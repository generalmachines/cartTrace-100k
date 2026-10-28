"""Triage: where did a failure come from, and what should the runner do about it (data/decisions/triage.yaml)?

Why both questions go through the decision engine. A store or our agent must never be blamed for our own failures,
and the retry policy has to read error text the rules cannot. Rules answer the clear signals (a refused call, a 429,
a timeout, a 5xx by who sent it); the decision model reads the rest. Every triage is logged, in the shared log and in
the episode's bundle, which is how the types get calibrated.

What changes behaviour. Only effective decisions (live types, no backend error), and only two answers: `skip_store`
and `pause_store` take the store out of the rest of the run, because a person or a later run should look at it first.
`retry` is left to the transport, whose RetryPolicy already retries what is safe to send twice, and `continue` hands
the error back to the agent, which is what happens anyway. While the types are shadow, nothing changes.

The evidence never carries secrets: error text is redacted with the trace rules before any decision sees it.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from carttrace.collect.ports import RunMode
from carttrace.core.canon import dumps
from carttrace.core.errors import ExternalServiceError, MCError, ProtocolError, SafetyError
from carttrace.decide.service import DecisionEngine
from carttrace.decide.types import Decision
from carttrace.protocols.tools import ToolResult
from carttrace.trace.redact import redact_text

Service = Literal["store", "sandbox", "model", "internal"]
MESSAGE_CHARS = 500
STORE_ACTIONS = frozenset({"skip_store", "pause_store"})
"""Effective answers of triage.next_action that take a store out of the rest of the run."""
SETUP_PHASES = frozenset({"discover", "open", "readback", "handoff"})
"""Episode phases that talk to a store or the sandbox directly, outside the agent."""


@dataclass(frozen=True, slots=True)
class Failure:
    """A failed call, as triage evidence."""

    service: Service
    tool: str | None
    status: int | None
    error_code: str | None
    message: str

    def evidence(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "tool": self.tool,
            "status": self.status,
            "error_code": self.error_code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class Triage:
    source: Decision
    action: Decision

    @property
    def effective_action(self) -> str | None:
        """The next action, when both decisions may change behaviour; None while either is shadow or failed."""
        if self.source.effective and self.action.effective and isinstance(self.action.answer, str):
            return self.action.answer
        return None

    @property
    def store_action(self) -> str | None:
        action = self.effective_action
        return action if action in STORE_ACTIONS else None

    def labels(self) -> list[str]:
        """Infra labels for the outcome: where the failure came from, as triage answered it."""
        return [f"triage:{self.source.answer}"]


def failure_of(
    exc: BaseException, *, phase: str, mode: RunMode, model_names: Collection[str], salt: str = ""
) -> Failure:
    """A raised exception as triage evidence. Who we were talking to comes from the phase and the error itself; the
    error code is the stable one carttrace.live puts on its errors when there is one."""
    status = exc.status if isinstance(exc, ExternalServiceError) else None
    code = getattr(exc, "code", None)
    if not isinstance(code, str):
        code = _code(exc, status)
    return Failure(
        service=_service(exc, phase=phase, mode=mode, model_names=model_names),
        tool=phase,
        status=status,
        error_code=code,
        message=_message(exc, salt=salt),
    )


def _code(exc: BaseException, status: int | None) -> str | None:
    if isinstance(exc, SafetyError):
        return "harness_blocked"
    if isinstance(exc, httpx.TimeoutException | TimeoutError):
        return "timeout"
    if isinstance(exc, ProtocolError):
        return "protocol_error"
    if status == 429:
        return "throttled"
    return None


def failure_of_result(tool: str, result: ToolResult, *, mode: RunMode, salt: str = "") -> Failure:
    """A tool call that came back as an error, as triage evidence."""
    content = result.content if isinstance(result.content, str) else _text_of(result.content)
    text, _ = redact_text(content[: MESSAGE_CHARS * 2], salt=salt)
    return Failure(
        service="store" if mode == "live" else "sandbox",
        tool=tool,
        status=_status_of(result),
        error_code=result.error_code,
        message=text[:MESSAGE_CHARS],
    )


async def triage(
    engine: DecisionEngine, failure: Failure, *, attempts: int, context: Mapping[str, Any] | None = None
) -> Triage:
    """Ask where the failure came from, then what to do next."""
    source = await engine.decide("triage.error_source", failure.evidence(), context=context)
    action = await engine.decide(
        "triage.next_action",
        {
            "error_source": source.answer,
            "error_code": failure.error_code,
            "status": failure.status,
            "attempts": attempts,
            "message": failure.message,
        },
        context=context,
    )
    return Triage(source=source, action=action)


STORE_CODES: dict[str, Literal["blocked", "throttled", "no_protocol"]] = {
    "robots_disallowed": "blocked",
    "blocked": "blocked",
    "throttled": "throttled",
    "not_found": "no_protocol",
    "redirected": "no_protocol",
    "protocol_error": "no_protocol",
    "negotiation_failed": "no_protocol",
}
"""carttrace.live's stable codes that are a store's outcome when they stop a live session from starting."""


def store_outcome(
    exc: BaseException, *, phase: str, mode: RunMode
) -> Literal["blocked", "throttled", "no_protocol"] | None:
    """The end reason a failure to start a live session earns by code, not by decision: robots.txt or a wall (401,
    403) is blocked, a 429 throttled, an endpoint that is not UCP over MCP no_protocol. Anything else, and anything
    after the session started, is ours to explain (harness_error)."""
    if mode != "live" or phase not in ("discover", "open"):
        return None
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code in STORE_CODES:
        return STORE_CODES[code]
    if isinstance(exc, ExternalServiceError) and exc.status in (401, 403):
        return "blocked"
    if isinstance(exc, ExternalServiceError) and exc.status == 429:
        return "throttled"
    return None


def _service(exc: BaseException, *, phase: str, mode: RunMode, model_names: Collection[str]) -> Service:
    if isinstance(exc, ExternalServiceError) and _names_a_model(exc.service, model_names):
        return "model"
    if isinstance(exc, SafetyError) or not isinstance(exc, MCError | httpx.HTTPError | TimeoutError | OSError):
        return "internal"
    if phase in SETUP_PHASES or phase == "agent":
        if phase == "handoff":
            return "sandbox"
        return "store" if mode == "live" else "sandbox"
    return "internal"


def _names_a_model(service: str, model_names: Collection[str]) -> bool:
    """carttrace.inference names a failed call "<endpoint> for role '<role>'" (carttrace.inference.calls.describe)."""
    return service in model_names or service.split(" for role ", 1)[0] in model_names


def _message(exc: BaseException, *, salt: str) -> str:
    text, _ = redact_text(f"{type(exc).__name__}: {exc}"[: MESSAGE_CHARS * 2], salt=salt)
    return text[:MESSAGE_CHARS]


def _status_of(result: ToolResult) -> int | None:
    for value in (result.structured, result.content):
        if isinstance(value, Mapping):
            status = value.get("status")
            if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
                return status
    return None


def _text_of(content: Any) -> str:
    if isinstance(content, list):
        parts = [
            str(block.get("text"))
            for block in content
            if isinstance(block, Mapping) and isinstance(block.get("text"), str)
        ]
        if parts:
            return " ".join(parts)
    try:
        return dumps(content)
    except (TypeError, ValueError):
        return str(content)

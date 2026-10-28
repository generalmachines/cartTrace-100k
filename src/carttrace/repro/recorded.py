"""A chat service that answers with a trace's own model turns, so a published journey can be run again without a model.

Why. A reader of CartTrace should be able to check that a trace is what the sandbox does, not only what we say it did:
the same world, task, seed and agent choices must reach the same store state. The agent's choices are the one part
that came from a model, and they are in the trace (turns.jsonl). So the Qwen agent is run as it was, with its model
replaced by this service, which returns each recorded completion in order; the shopper, the wallet, the consent rule,
the supervisor and the store are the real code. The replay then compares the sandbox's final state hash and events
chain head with the ones the trace published (outcome.json `state_digest`).
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from carttrace.core.errors import ValidationError
from carttrace.inference.service import ChatResult, ToolCall

MODEL_KINDS = frozenset({"note", "message", "tool_call"})
EMAIL_PLACEHOLDER = re.compile(r"email-[0-9a-f]{8,64}@redacted\.invalid")
PHONE_PLACEHOLDER = re.compile(r"phone-[0-9a-f]{8,64}")
"""The trace's placeholders for an email address and a phone number (carttrace.trace.redact): salted, so not reversible."""
SECRET = re.compile(r"redacted|jw[spd]-[0-9a-f]{12}")
"""What the redaction leaves where a secret the store issued was: the fixed marker (a cart token's `?key=`, a
signature) or a JWS part placeholder (an AP2 mandate, a checkout_jwt). The value is the store's or the wallet's own,
issued again by the replayed run, so it is read back from the replay's tool results, never guessed."""
SECRET_VALUE = {"redacted": r"[^\"\\\s&?#]+", "jw": r"[A-Za-z0-9_-]+"}


Phone = str | Mapping[str, str] | None
"""How a replay spells the persona's phone: one spelling everywhere, or one per field (`phone_field`), with `ANY_FIELD`
for the fields not named."""
ANY_FIELD = "*"


def phone_field(path: Sequence[str]) -> str:
    """The field a value sits in, as its last two keys (`buyer.phone_number`): an agent can write one number two ways
    in one request, once per field, and the placeholder keys a phone on its digits alone."""
    return ".".join(path[-2:])


def restore(value: Any, *, email: str | None, phone: Phone, path: tuple[str, ...] = ()) -> Any:
    """`value` with the placeholders put back to the persona's own email and phone. The redaction ran at write time
    over every file, the persona's fictional details included; the replay must send what the agent sent, and the
    persona (from the task's buyer block) is the only one whose details an agent was ever given."""
    if isinstance(value, str):
        if email:
            value = EMAIL_PLACEHOLDER.sub(email, value)
        spelled = phone.get(phone_field(path), phone.get(ANY_FIELD)) if isinstance(phone, Mapping) else phone
        if spelled:
            value = PHONE_PLACEHOLDER.sub(spelled, value)
        return value
    if isinstance(value, Mapping):
        return {k: restore(v, email=email, phone=phone, path=(*path, str(k))) for k, v in value.items()}
    if isinstance(value, list):
        return [restore(v, email=email, phone=phone, path=path) for v in value]
    return value


def phone_fields(turns: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Per agent tool call (`call_id`), how many phone placeholders each field of its arguments holds."""
    out: dict[str, dict[str, int]] = {}

    def walk(value: Any, path: tuple[str, ...], into: dict[str, int]) -> None:
        if isinstance(value, str):
            n = len(PHONE_PLACEHOLDER.findall(value))
            if n:
                into[phone_field(path)] = into.get(phone_field(path), 0) + n
        elif isinstance(value, Mapping):
            for k, v in value.items():
                walk(v, (*path, str(k)), into)
        elif isinstance(value, list):
            for v in value:
                walk(v, path, into)

    for turn in turns:
        if turn.get("actor") == "agent" and turn.get("kind") == "tool_call" and turn.get("call_id"):
            found: dict[str, int] = {}
            walk(turn.get("arguments"), ("arguments",), found)
            if found:
                out[str(turn["call_id"])] = found
    return out


def recorded_calls(
    turns: Iterable[Mapping[str, Any]], *, email: str | None = None, phone: Phone = None
) -> list[ChatResult]:
    """The model calls behind a trace's agent turns, in order. Every turn a call produced carries that call's seed
    (D69): its reasoning as a note, its text as a message, and each tool call; the first carries its usage."""
    calls: list[ChatResult] = []
    group: list[Mapping[str, Any]] = []
    seed: object = None
    for found in turns:
        if found.get("actor") != "agent" or found.get("kind") not in MODEL_KINDS or "seed" not in found:
            continue
        turn = restore(dict(found), email=email, phone=phone)
        if group and turn.get("seed") != seed:
            calls.append(_call(group))
            group = []
        seed = turn.get("seed")
        group.append(turn)
    if group:
        calls.append(_call(group))
    return calls


def call_times(turns: Iterable[Mapping[str, Any]]) -> list[int]:
    """When each recorded model call acted, in ms since the episode opened: its first tool call's `t_ms`, else its
    first turn's. Parallel to `recorded_calls`."""
    times: list[int] = []
    group: list[Mapping[str, Any]] = []
    seed: object = None
    for turn in [*turns, {"actor": "agent", "kind": "note", "seed": object()}]:
        if turn.get("actor") != "agent" or turn.get("kind") not in MODEL_KINDS or "seed" not in turn:
            continue
        if group and turn.get("seed") != seed:
            acted = next((g for g in group if g.get("kind") == "tool_call"), group[0])
            times.append(int(acted.get("t_ms") or 0))
            group = []
        seed = turn.get("seed")
        group.append(turn)
    return times


def _dated(result: object) -> datetime | None:
    """The `timestamp` a wallet result was dated with (AP2 payment_mandate_contents), if it has one."""
    if not isinstance(result, str) or "timestamp" not in result:
        return None
    found = re.search(r'"timestamp\\?"\s*:\s*\\?"(\d{4}-\d\d-\d\dT[0-9:.]+Z)', result)
    if found is None:
        return None
    try:
        return datetime.fromisoformat(found.group(1).replace("Z", "+00:00"))
    except ValueError:
        return None


class ReplayClock:
    """The run's clock on the trace's own timeline. A wallet dates what it signs (an AP2 mandate's `iat`) and an ACP
    request carries its timestamp, so a replay at today's time signs other bytes and reaches another state. `now()`
    is the episode's start plus where the trace was when the current model call acted; `monotonic()` is real, so the
    run's time caps and the writer's `t_ms` behave as in any run."""

    def __init__(
        self, started: datetime, turns: Iterable[Mapping[str, Any]] = (), wire: Iterable[Mapping[str, Any]] = ()
    ) -> None:
        import time

        self.started = started
        self.offset_ms = 0
        self._mono = time.monotonic
        listed = list(turns)
        dated = {str(t.get("tool_call_id")): _dated(t.get("result")) for t in listed if t.get("kind") == "tool_result"}
        for record in wire:  # a request the store got dated (an AP2 payment mandate's contents), by the call it served
            sent = _dated(json.dumps(record.get("request_body")))
            call_id = record.get("source_call_id")
            if sent is not None and isinstance(call_id, str) and dated.get(call_id) is None:
                dated[call_id] = sent
        pinned = {k: round((v - started).total_seconds() * 1000) for k, v in dated.items() if v is not None}
        made = [t for t in listed if t.get("actor") == "agent" and t.get("kind") == "tool_call"]
        lag = [
            round((stamp - started).total_seconds() * 1000) - int(t.get("t_ms") or 0)
            for t in made
            if (stamp := dated.get(str(t.get("call_id")))) is not None
        ]
        self.lag_ms = lag[0] if lag else 0
        """How far the writer's clock (`t_ms`) ran behind the episode's start, learnt from a dated result."""
        self.calls: list[tuple[str, int]] = [
            (str(t.get("tool")), pinned.get(str(t.get("call_id")), int(t.get("t_ms") or 0) + self.lag_ms)) for t in made
        ]
        """Every tool call the trace recorded, with when it was made, in order, on the episode's own time."""
        self.next = 0

    def at(self, offset_ms: int) -> None:
        self.offset_ms = offset_ms + self.lag_ms

    def calling(self, tool: str) -> None:
        """A surface is about to answer `tool`: the clock moves to when the trace made that call, so a date the
        store or the wallet writes falls in the same second as it did."""
        for i in range(self.next, len(self.calls)):
            if self.calls[i][0] == tool:
                self.offset_ms = self.calls[i][1]
                self.next = i + 1
                return

    def watch(self, surface: Any) -> Any:
        """`surface` with its calls moving this clock first."""
        call = surface.call

        async def timed(name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
            self.calling(name)
            return await call(name, arguments, **kwargs)

        with contextlib.suppress(AttributeError):  # a surface that takes no attribute keeps the model calls' times
            surface.call = timed
        return surface

    def now(self) -> datetime:
        return self.started + timedelta(milliseconds=self.offset_ms)

    def monotonic(self) -> float:
        return self._mono()


def _call(turns: Sequence[Mapping[str, Any]]) -> ChatResult:
    reasoning = next((str(t.get("text") or "") for t in turns if t.get("kind") == "note"), None)
    text = next((str(t.get("text") or "") for t in turns if t.get("kind") == "message"), "")
    tool_calls = [_tool_call(t) for t in turns if t.get("kind") == "tool_call"]
    first = turns[0]
    usage = next((dict(t["usage"]) for t in turns if isinstance(t.get("usage"), Mapping)), {})
    model = str(first.get("model") or "recorded")
    return ChatResult(
        role="agent",
        text=text,
        tool_calls=tool_calls,
        reasoning=reasoning,
        finish_reason=_str(first.get("finish_reason")),
        requested_model=model,
        served_model=model,
        usage={k: int(v) for k, v in usage.items() if isinstance(v, int)},
        latency_ms=int(next((t["latency_ms"] for t in turns if isinstance(t.get("latency_ms"), int)), 0)),
        endpoint="recorded",
    )


def _tool_call(turn: Mapping[str, Any]) -> ToolCall:
    raw = turn.get("arguments")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return ToolCall(
                id=str(turn.get("call_id")),
                name=str(turn.get("tool")),
                arguments={},
                arguments_error="recorded arguments did not parse",
                raw_arguments=raw,
            )
        raw = parsed
    arguments = dict(raw) if isinstance(raw, Mapping) else {}
    return ToolCall(id=str(turn.get("call_id")), name=str(turn.get("tool")), arguments=arguments)


def _str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _pattern(value: str, key: str | None) -> re.Pattern[str] | None:
    """A regex finding the live value `value` stands for: its text around each secret kept, each secret a wildcard.
    A value that is nothing but a secret is looked for under its own key in a JSON result."""
    parts = SECRET.split(value)
    secrets = SECRET.findall(value)
    if not secrets:
        return None
    body = "".join(
        re.escape(text) + (f"({SECRET_VALUE['redacted' if s == 'redacted' else 'jw']})" if i < len(secrets) else "")
        for i, (text, s) in enumerate(zip(parts, [*secrets, ""], strict=True))
    )
    if not "".join(parts).strip():
        if key is None:
            return None
        return re.compile(rf'"{re.escape(key)}"\s*:\s*"({body})"')
    return re.compile(f"({body})")


def live_values(value: Any, seen: str, key: str | None = None) -> Any:
    """`value` with each redacted secret put back to the live value the replayed run issued, found in `seen` (the
    tool results so far, newest last). A secret the replay has not issued stays as it is: the call then goes as the
    trace has it, and the state hashes will say the replay diverged."""
    if isinstance(value, str):
        pattern = _pattern(value, key)
        if pattern is None:
            return value
        found = pattern.findall(seen)
        if not found:
            return value
        last = found[-1]
        return last if isinstance(last, str) else last[0]
    if isinstance(value, Mapping):
        return {k: live_values(v, seen, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [live_values(v, seen, key) for v in value]
    return value


class RecordedChat:
    """ChatService over recorded calls. Asking past the end is a replay that went somewhere the trace did not."""

    def __init__(
        self, calls: Sequence[ChatResult], *, clock: ReplayClock | None = None, times: Sequence[int] = ()
    ) -> None:
        self.calls = list(calls)
        self.clock = clock
        self.times = list(times)
        self.asked = 0
        self.restored = 0
        """Redacted secrets put back from the replay's own results."""

    async def chat(
        self,
        role: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> ChatResult:
        if self.asked >= len(self.calls):
            raise ValidationError(
                f"the replay asked for model call {self.asked + 1}, but the trace recorded {len(self.calls)}",
                hint="the replay diverged from the trace; compare its turns with the published ones",
            )
        result = self.calls[self.asked]
        if self.clock is not None and self.asked < len(self.times):
            self.clock.at(self.times[self.asked])
        self.asked += 1
        if not result.tool_calls:
            return result
        seen = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "tool")
        calls = []
        for call in result.tool_calls:
            arguments = live_values(call.arguments, seen)
            self.restored += arguments != call.arguments
            calls.append(call.model_copy(update={"arguments": arguments}))
        return result.model_copy(update={"tool_calls": calls})

    @property
    def unused(self) -> int:
        return len(self.calls) - self.asked


class RecordedWorder:
    """A Worder (carttrace.agents.wording) that gives each shopper reply the words the trace recorded. The shopper's code
    still decides every reply (D118); only the words come from the trace, so a reply worded by model reads the same
    and one that fell back to the rules' line falls back again. Shopper words reach the store (an AP2 intent's
    description, say), so a replay worded otherwise would not reach the trace's state."""

    def __init__(self, turns: Iterable[Mapping[str, Any]], *, email: str | None = None, phone: Phone = None):
        self.replies: list[tuple[str, str]] = []
        for turn in turns:
            origin = turn.get("origin")
            if turn.get("actor") != "shopper" or turn.get("kind") != "message" or not isinstance(origin, Mapping):
                continue
            wording = origin.get("wording")
            if wording in ("model", "fallback"):
                text = restore(str(turn.get("text") or ""), email=email, phone=phone)
                self.replies.append((str(wording), text))
        self.used = 0

    async def __call__(
        self, messages: Sequence[Mapping[str, str]], *, max_tokens: int, temperature: float, seed: int
    ) -> Any:
        from carttrace.agents.wording import WordingReply

        if self.used >= len(self.replies):
            raise ValidationError("the replay asked the shopper for more replies than the trace recorded")
        wording, text = self.replies[self.used]
        self.used += 1
        if wording == "fallback":
            raise ValidationError("the trace's reply fell back to the rules' line")
        return WordingReply(text, "recorded")

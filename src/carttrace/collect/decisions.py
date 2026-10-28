"""One episode's view of the decision engine: every decision it makes is also written into its own bundle.

Why. The engine is shared by every episode of a run and logs to the shared decision log, which is what calibration
reads. The judge and the trace readers need the same decisions beside the episode they were made in, in the order
they happened among its turns, so the supervisor's, the shopper's and triage's decisions all pass through this
wrapper, which adds the episode's ids to the decision's context and appends the decision to `decisions.jsonl`.

Evidence shapes follow data/decisions/supervise.yaml: a step is a tool call
`{"tool", "arguments", "is_error", "error_code", "result"}` or a message `{"text"}`, results cut short, and a cart is
`{"lines", "total", "currency"}` or None. Steps arrive from the agent as mappings or objects; both are read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from carttrace.core.canon import dumps
from carttrace.decide.service import DecisionEngine
from carttrace.decide.types import Decision
from carttrace.trace.schema import Cart
from carttrace.trace.writer import TraceWriter

RESULT_CHARS = 400
"""How much of a tool result a supervision step shows."""
TEXT_CHARS = 600
RECENT_STEPS = 12
"""supervise.yaml: the last 12 steps are plenty."""


class EpisodeDecisions:
    """A DecisionEngine that writes each decision into the episode's bundle as well as the shared log."""

    def __init__(self, engine: DecisionEngine, writer: TraceWriter, context: Mapping[str, Any]) -> None:
        self.engine = engine
        self.writer = writer
        self.context = dict(context)
        self.recording = True
        """Turned off once the episode's outcome is written; later decisions reach the shared log only."""
        self.unrecorded = 0
        self.unavailable: list[dict[str, Any]] = []
        """Decisions that abstained because their model could not be reached (D115: the row is then quarantined)."""

    async def decide(
        self, type_id: str, evidence: Mapping[str, Any], *, context: Mapping[str, Any] | None = None
    ) -> Decision:
        decision = await self.engine.decide(type_id, evidence, context={**self.context, **(context or {})})
        self._record(decision)
        return decision

    async def decide_many(
        self, requests: Sequence[tuple[str, Mapping[str, Any]]], *, context: Mapping[str, Any] | None = None
    ) -> list[Decision]:
        decisions = await self.engine.decide_many(requests, context={**self.context, **(context or {})})
        for decision in decisions:
            self._record(decision)
        return decisions

    def _record(self, decision: Decision) -> None:
        if decision.decided_by == "abstain" and decision.error:
            self.unavailable.append({"type_id": decision.type_id, "decided_by": "abstain", "error": decision.error})
        if self.recording:
            self.writer.decision(decision)
        else:
            self.unrecorded += 1


def step_evidence(step: Any) -> dict[str, Any]:
    """One agent step as supervision evidence."""
    tool = _field(step, "tool")
    if isinstance(tool, str) and tool:
        arguments = _field(step, "arguments")
        return {
            "tool": tool,
            "arguments": _plain(arguments) if isinstance(arguments, Mapping) else {},
            "is_error": _field(step, "is_error") is True,
            "error_code": _str_or_none(_field(step, "error_code")),
            "result": _short(_field(step, "result"), RESULT_CHARS),
        }
    return {"text": _short(_field(step, "text") or _field(step, "content") or "", TEXT_CHARS)}


def steps_evidence(steps: Sequence[Any]) -> list[dict[str, Any]]:
    return [step_evidence(step) for step in list(steps)[-RECENT_STEPS:]]


def cart_evidence(cart: Any) -> dict[str, Any] | None:
    """A cart as supervision evidence: its lines, total and currency, or None when there is no cart."""
    if cart is None:
        return None
    if isinstance(cart, Cart):
        return {
            "lines": [
                {"variant_id": ln.variant_id, "title": ln.title, "quantity": ln.quantity, "unit_price": ln.unit_price}
                for ln in cart.lines
            ],
            "total": cart.total,
            "currency": cart.currency,
        }
    data = cart.model_dump(mode="json") if isinstance(cart, BaseModel) else cart
    if isinstance(data, Mapping):
        plain = _plain(data)
        if isinstance(plain, dict) and "lines" not in plain and isinstance(plain.get("line_items"), list):
            plain["lines"] = plain["line_items"]
        return plain if isinstance(plain, dict) else None
    return None


def _field(step: Any, name: str) -> Any:
    return step.get(name) if isinstance(step, Mapping) else getattr(step, name, None)


def _plain(value: Any) -> Any:
    """`value` as plain JSON; anything JSON cannot hold becomes its text, so evidence never fails to serialise."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def _short(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else dumps(_plain(value))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None

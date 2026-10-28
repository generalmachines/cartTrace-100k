"""What a decision engine promises, and where its decisions go. The router in `carttrace.decide.router` is the engine;
`carttrace.decide.replay` answers from a recorded log; tests use fakes. Sinks: `carttrace.decide.log`, or a trace writer's adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from carttrace.decide.types import Decision


class DecisionEngine(Protocol):
    async def decide(
        self, type_id: str, evidence: Mapping[str, Any], *, context: Mapping[str, Any] | None = None
    ) -> Decision:
        """Answer one decision. Never raises for a model failure: the Decision abstains and carries `error`.
        Raises ValidationError when the type is unknown or the evidence lacks a key the type needs."""
        ...

    async def decide_many(
        self, requests: Sequence[tuple[str, Mapping[str, Any]]], *, context: Mapping[str, Any] | None = None
    ) -> list[Decision]:
        """Answer a batch, concurrently within the GPU cap, in the order given."""
        ...


class DecisionSink(Protocol):
    """Where every decision is recorded: the JSONL decision log, a trace bundle's decisions.jsonl, or memory."""

    def write(self, decision: Decision) -> None:
        """Record one decision. The engine calls it once per decision; decide_many calls it in request order."""
        ...

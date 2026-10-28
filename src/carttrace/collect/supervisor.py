"""The decision supervisor: every few agent steps, is the episode looping, stuck, or already at the cart boundary?

Why in the collector. Supervision exists to free GPU slots from hopeless episodes (docs/ARCHITECTURE.md, "Engines that
use it"), which is the runner's concern, and the agent only has to ask. The agent may call `check` after every step;
the supervisor asks the decision engine only once `collect.supervise_every_steps` new steps have happened since it last
asked, so the cadence holds whatever the agent does.

When it stops an episode. Only on a yes that is `effective`: a live decision type with no backend error. The three
supervise types are shadow until they agree with human labels, so today every check is logged and none stops an
episode, which is the point of shadow (a false yes throws away an episode that might have finished).

A checkout row (D13) goes past the cart on purpose, so its supervisor never asks `supervise.boundary_reached`: the cart
boundary is where a live journey ends, not where an order begins.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from carttrace.agents.service import SupervisorVerdict
from carttrace.collect.decisions import cart_evidence, steps_evidence
from carttrace.decide.service import DecisionEngine
from carttrace.decide.types import Decision

CHECKS: tuple[str, ...] = ("supervise.looping", "supervise.stuck", "supervise.boundary_reached")
REASONS: Mapping[str, str] = {
    "supervise.looping": "looping",
    "supervise.stuck": "stuck",
    "supervise.boundary_reached": "boundary_reached",
}


GO_ON = SupervisorVerdict(stop=False)
EVIDENCE_SHA_CHARS = 16


def decision_id(decision: Decision) -> str:
    """`<type id>@<evidence sha, 16 hex>`: the name carttrace.agents gives a decision, which joins it to the decision log."""
    return f"{decision.type_id}@{decision.evidence_sha[:EVIDENCE_SHA_CHARS]}"


class DecisionSupervisor:
    """Supervisor for one episode. `checks` counts the times it asked the engine."""

    def __init__(self, engine: DecisionEngine, *, goal: str, every: int, boundary: bool = True) -> None:
        """`boundary` False leaves out the cart boundary check, for a checkout row."""
        if every < 1:
            raise ValueError("supervision needs at least one step between checks")
        self.boundary = boundary
        self.engine = engine
        self.goal = goal
        self.every = every
        self.checks = 0
        self._asked_at = 0

    async def check(self, steps: Sequence[Any], cart: Any) -> SupervisorVerdict:
        if len(steps) - self._asked_at < self.every:
            return GO_ON
        self._asked_at = len(steps)
        self.checks += 1
        recent = steps_evidence(steps)
        loop = {"goal": self.goal, "steps": recent}
        asks: list[tuple[str, Mapping[str, Any]]] = [("supervise.looping", loop), ("supervise.stuck", loop)]
        if self.boundary:
            asks.append(("supervise.boundary_reached", {**loop, "cart": cart_evidence(cart)}))
        decisions = await self.engine.decide_many(asks, context={"step": len(steps)})
        for decision in decisions:
            if decision.effective and decision.answer is True:
                return SupervisorVerdict(stop=True, reason=REASONS[decision.type_id], decision_id=decision_id(decision))
        return GO_ON

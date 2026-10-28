"""Supervision from the decision engine: supervise.looping and supervise.stuck as SupervisorVerdicts.

Why through carttrace.decide. An agent that repeats one search, or keeps calling a tool that fails, holds a GPU slot and a
store's goodwill for nothing, so the runner ends such episodes early (D9). Whether an agent is looping is a typed
decision: a rule answers the plain cases (the same call three times in six steps, five failures in a row) and Clef
reads the rest. A false stop throws away an episode that might have finished, so only an effective decision stops
one: a live type with no backend error. While a type is in shadow its answers are logged for calibration and never
stop anything.

The agent calls `check` after every step; the supervisor asks the engine only every `every_steps` steps
(collect.supervise_every_steps), with the last twelve steps as evidence, as data/decisions/supervise.yaml asks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from carttrace.agents.service import SupervisorVerdict

if TYPE_CHECKING:
    from carttrace.config.schema import Settings
    from carttrace.decide.service import DecisionEngine
    from carttrace.decide.types import Decision

TYPES = ("supervise.looping", "supervise.stuck")
"""The decision types a check asks, in order; the first effective yes stops the episode."""
WINDOW = 12
"""Recent steps shown to the decision: supervise.yaml says the last twelve are plenty."""


class SupervisorFromDecisions:
    """A Supervisor that asks the decision engine. One instance per episode, because it counts the steps."""

    def __init__(
        self,
        engine: DecisionEngine,
        *,
        goal: str,
        every_steps: int = 1,
        types: Sequence[str] = TYPES,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """`goal` is the shopper's opening; `context` (episode_id, task_id, ...) is logged with each decision."""
        if every_steps < 1:
            raise ValueError("every_steps is at least 1")
        self.engine = engine
        self.goal = goal
        self.every_steps = every_steps
        self.types = tuple(types)
        self.context = dict(context or {})
        self.decisions: list[Decision] = []
        """Every decision this supervisor asked for, in order."""
        self._checks = 0

    @classmethod
    def for_episode(
        cls,
        engine: DecisionEngine,
        settings: Settings,
        *,
        goal: str,
        context: Mapping[str, Any] | None = None,
    ) -> SupervisorFromDecisions:
        """Asking every collect.supervise_every_steps steps."""
        return cls(engine, goal=goal, every_steps=settings.collect.supervise_every_steps, context=context)

    async def check(self, steps: Sequence[Mapping[str, Any]], cart: Mapping[str, Any] | None) -> SupervisorVerdict:
        self._checks += 1
        if self._checks % self.every_steps:
            return SupervisorVerdict(stop=False)
        evidence = {"goal": self.goal, "steps": [dict(s) for s in steps[-WINDOW:]], "cart": cart}
        decisions = await self.engine.decide_many([(t, evidence) for t in self.types], context=self.context)
        self.decisions += decisions
        for decision in decisions:
            if decision.answer is True and decision.effective:
                reason = f"{decision.type_id} answered yes ({decision.decided_by})"
                return SupervisorVerdict(stop=True, reason=reason, decision_id=decision_id(decision))
        shadow = [d.type_id for d in decisions if d.answer is True]
        reason = f"shadow yes from {', '.join(shadow)}; not acted on" if shadow else ""
        return SupervisorVerdict(stop=False, reason=reason)


def decision_id(decision: Decision) -> str:
    """`<type id>@<first 16 hex of the evidence sha>`: joins a verdict to its line in the decision log."""
    return f"{decision.type_id}@{decision.evidence_sha[:16]}"

"""The decision engine of this kit: rules alone.

Why rules alone. Every CartTrace-100k row decided by rule (D115): the shopper's replies, the supervisor's stop checks
and error triage are code. A decision no rule settles abstains with the type's safe default, exactly as it did when
the trace was recorded, so a replay reaches the same state. No model is asked for a decision anywhere in this kit.

What the engine guarantees:
- a request is checked before anything runs: an unknown type or missing evidence raises ValidationError;
- a rule that returns an answer outside the type's space abstains and says why in `error`;
- every decision goes to the sink, and decide_many writes them in request order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

from carttrace.core.clock import Clock, SystemClock, iso
from carttrace.core.errors import ValidationError
from carttrace.decide.catalog import Catalog
from carttrace.decide.evidence import Evidence, plain_context, select_evidence
from carttrace.decide.service import DecisionSink
from carttrace.decide.space import AnswerSpace, resolve_space
from carttrace.decide.types import Answer, DecidedBy, Decision, DecisionType

OFFLINE_EVIDENCE: Final = "offline_evidence"
"""The context key an abstention keeps its evidence under (RouterConfig.keep_evidence)."""


@dataclass(frozen=True, slots=True)
class RouterConfig:
    keep_evidence: bool = False
    """An abstention keeps its evidence in the decision's context under `offline_evidence`, as the collector did."""


@dataclass(frozen=True, slots=True)
class _Request:
    dtype: DecisionType
    evidence: Evidence
    space: AnswerSpace
    abstain: Answer
    context: dict[str, Any]


class Router:
    """The DecisionEngine. Build it with `carttrace.decide.build_engine`."""

    def __init__(
        self, catalog: Catalog, *, sink: DecisionSink, clock: Clock | None = None, config: RouterConfig | None = None
    ) -> None:
        self.catalog = catalog
        self.config = config or RouterConfig()
        self._sink = sink
        self._clock: Clock = clock or SystemClock()

    async def decide(
        self, type_id: str, evidence: Mapping[str, Any], *, context: Mapping[str, Any] | None = None
    ) -> Decision:
        decision = self._resolve(self._prepare(type_id, evidence, context))
        self._sink.write(decision)
        return decision

    async def decide_many(
        self, requests: Sequence[tuple[str, Mapping[str, Any]]], *, context: Mapping[str, Any] | None = None
    ) -> list[Decision]:
        """Every request is checked before any is answered, so a bad batch fails whole."""
        prepared = [self._prepare(type_id, evidence, context) for type_id, evidence in requests]
        decisions = [self._resolve(r) for r in prepared]
        for decision in decisions:
            self._sink.write(decision)
        return decisions

    def _prepare(self, type_id: str, evidence: Mapping[str, Any], context: Mapping[str, Any] | None) -> _Request:
        dtype = self.catalog.get(type_id)
        chosen = select_evidence(dtype, evidence)
        space = resolve_space(dtype, chosen.values)
        abstain = space.check(dtype.abstain, source="the abstain value")
        return _Request(dtype, chosen, space, abstain, plain_context(context))

    def _resolve(self, request: _Request) -> Decision:
        started = self._clock.monotonic()
        rule_name = request.dtype.rule
        if rule_name is not None:
            raw = self.catalog.rules.get(rule_name)(request.evidence.values)
            if raw is not None:
                try:
                    answer = request.space.check(raw, source=f"rule {rule_name}")
                except ValidationError as exc:  # a rule bug: logged and visible, and never effective
                    return self._finish(request, started, "abstain", request.abstain, error=str(exc))
                return self._finish(request, started, "rule", answer, probabilities=request.space.certain(answer))
        if self.config.keep_evidence:
            request = replace(
                request, context={**request.context, OFFLINE_EVIDENCE: plain_context(request.evidence.values)}
            )
        return self._finish(request, started, "abstain", request.abstain)

    def _finish(
        self,
        request: _Request,
        started: float,
        decided_by: DecidedBy,
        answer: Answer,
        *,
        probabilities: dict[str, float] | None = None,
        error: str | None = None,
    ) -> Decision:
        return Decision(
            type_id=request.dtype.id,
            type_version=request.dtype.version,
            answer=answer,
            probabilities=probabilities or {},
            decided_by=decided_by,
            model_snapshot=None,
            evidence_sha=request.evidence.sha,
            latency_ms=max(0, round((self._clock.monotonic() - started) * 1000)),
            status=request.dtype.status,
            escalated=False,
            error=error,
            context=request.context,
            prompt_sha=None,
            decided_at=iso(self._clock.now()),
            stages=[],
        )

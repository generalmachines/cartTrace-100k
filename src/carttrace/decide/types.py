"""The contract of the router and decision engines: what a decision type is and what a decision returns.

Why typed. Collection, supervision, triage, alerting and judging all need many small decisions made fast. A decision
with a fixed answer space can be answered by a rule, or by Clef scoring label tokens in one forward pass, and either
way the result is a value with a probability that can be logged, replayed, calibrated against human labels and
compared across model versions. Free text cannot.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Kind = Literal["bool", "enum", "multi"]
DecidedBy = Literal["rule", "clef-flash", "clef", "fallback", "abstain"]
"""fallback (D61) is a configured model that is not the agent under test, asked only when both Clef stages failed; it
is never recorded as clef. D56's qwen stage is gone, and no decision log on record names it."""
ModelStageName = Literal["clef-flash", "clef", "fallback"]
Status = Literal["shadow", "live"]
Answer = str | list[str] | bool
"""What a decision answers: a bool for yes or no, one option for an enum, a list of options for a multi."""


class DecisionType(BaseModel):
    """One kind of decision, loaded from data/decisions/*.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    """Dotted id, e.g. "supervise.looping", "shopper.field", "triage.error_source", "judge.asked_size"."""
    kind: Kind
    options: list[str] = Field(default_factory=list)
    """Answer labels for enum and multi; for bool the labels are always yes and no."""
    question: str
    """The question Clef is asked, with {placeholders} filled from the evidence."""
    evidence: list[str] = Field(default_factory=list)
    """Evidence keys the question needs; the caller supplies them and the engine refuses when one is missing."""
    rule: str | None = None
    """Name of a function in carttrace.decide.rules that may answer outright, or return None to pass to Clef."""
    threshold: float = 0.8
    """Minimum probability of the winning answer for Clef's answer to stand."""
    escalate: bool = True
    """Ask clef when clef-flash is below threshold."""
    abstain: str | list[str] | bool
    """The safe default when nobody is confident, in the type's answer space."""
    status: Status = "shadow"
    """Shadow decisions are logged and evaluated but must not change what the system does."""
    version: int = 1
    options_from: str | None = None
    """An evidence key whose value adds options at call time, placed before the static options: a list of labels, or
    a mapping of label to a one-line note. shopper.field takes the hidden-spec field names this way."""
    option_notes: dict[str, str] = Field(default_factory=dict)
    """A one-line note per static option, shown to Clef beside the option."""
    max_chars: int = 6000
    """Character budget for the rendered evidence. A longer value keeps its head and tail, and the cut is marked."""
    description: str = ""
    """What the decision is for and who asks it, for people reading `carttrace decide show`."""


class StageAnswer(BaseModel):
    """What one model stage answered on the way to a decision, kept when a later stage decided or the engine abstained.

    Why. Until this record a decision kept only the stage that decided: when clef-flash was unsure and clef answered,
    clef-flash's answer was lost, so no threshold could be re-scored and no cascade calibrated on the items it deferred
    (Trust or Escalate, Jung et al. 2025; JEV-as-a-Judge, Li et al. 2026, which ran the fallback on accepted items for
    exactly this). Prediction-powered estimates also need a judge whose every answer is on record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: ModelStageName
    answer: Answer | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None
    """Probability of the stage's winning answer, the number the type's threshold is compared with."""
    served_model: str | None = None
    endpoint: str | None = None
    probabilities_from: Literal["logprobs", "text"] | None = None
    """How a fallback stage read its probabilities: "logprobs" from the next token's log probabilities, or "text" from
    the answer's text with the configured decide.fallback.text_confidence. None for Clef, which scores label tokens."""
    tokens: int | None = None
    latency_ms: int | None = None
    cost_usd: float | None = None
    """What the stage's calls cost, when its backend reports a cost (the Claude CLI fallback); None otherwise."""
    error: str | None = None
    """Why the stage gave no answer: a backend failure, a refused input, an open circuit breaker."""


class Decision(BaseModel):
    """One answered decision, as logged."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type_id: str
    type_version: int
    answer: str | list[str] | bool
    probabilities: dict[str, float] = Field(default_factory=dict)
    """Label to probability, from whoever decided; a rule's answer has probability 1.0."""
    decided_by: DecidedBy
    model_snapshot: str | None = None
    evidence_sha: str
    """sha256 of the canonical evidence the decision saw, so a replay can prove it saw the same thing."""
    latency_ms: int
    status: Status
    escalated: bool = False
    error: str | None = None
    """Set when a backend failed and the engine abstained because of it."""
    context: dict[str, Any] = Field(default_factory=dict)
    """Caller-supplied ids (episode_id, run_id, store_id) for joining the log back to traces."""
    prompt_sha: str | None = None
    """sha256 of the prompts and labels prepared for Clef; None when a rule decided first or no model is configured."""
    decided_at: str | None = None
    """RFC 3339 time the engine finished the decision, from its injected clock."""
    stages: list[StageAnswer] = Field(default_factory=list)
    """Every model stage the engine tried, in order, with its own answer and probabilities: the clef-flash answer of an
    escalated decision is here, beside the clef answer that decided. Empty when a rule decided."""

    @property
    def effective(self) -> bool:
        """Whether this decision may change behaviour: live types only, and never an abstention from an error."""
        return self.status == "live" and self.error is None

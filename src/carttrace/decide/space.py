"""The answers one decision may give: yes or no, or the type's options plus any its evidence supplies.

Why resolved per call. Most types have a fixed answer space, but some take options from the evidence at call time:
shopper.field asks which hidden-spec field the agent means, and every task has different fields. The space is resolved
once per decision, before any rule or model runs, so a rule's answer, the decision model's letters and the abstain value
are all checked against the same list, and a malformed options list is refused as bad evidence instead of reaching the
decision model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from carttrace.core.canon import dumps
from carttrace.core.errors import ValidationError
from carttrace.decide.types import Answer, DecisionType, Kind

YES, NO = "yes", "no"
"""The two labels of every bool decision, as the decision model scores them and as Decision.probabilities keys them."""
MAX_OPTIONS = 26
"""Enum options are lettered A to Z for the decision model, so one decision offers at most 26."""
MAX_OPTION_CHARS = 80
"""An option supplied by the evidence is a short label (a field name), never a sentence."""


@dataclass(frozen=True, slots=True)
class AnswerSpace:
    type_id: str
    kind: Kind
    options: tuple[str, ...] = ()
    """Options in the order the decision model sees them: supplied ones first, then the type's own. Empty for a bool."""
    notes: Mapping[str, str] = field(default_factory=dict)
    """A one-line note per option, where the type or the evidence gave one."""

    @property
    def labels(self) -> tuple[str, ...]:
        """The keys of Decision.probabilities: yes and no for a bool, the options otherwise."""
        return (YES, NO) if self.kind == "bool" else self.options

    def check(self, answer: object, *, source: str) -> Answer:
        """`answer` in canonical form (a multi's options in space order), or ValidationError naming its `source`."""
        if self.kind == "bool" and isinstance(answer, bool):
            return answer
        if self.kind == "enum" and isinstance(answer, str) and answer in self.options:
            return answer
        if self.kind == "multi" and isinstance(answer, list) and all(isinstance(a, str) for a in answer):
            chosen = set(answer)
            if chosen <= set(self.options):
                return [o for o in self.options if o in chosen]
        raise ValidationError(
            f"{self.type_id}: {source} gave {answer!r}, which is not in the answer space {list(self.labels)}",
            hint="a bool answers True or False, an enum one option, a multi a list of options",
        )

    def certain(self, answer: Answer) -> dict[str, float]:
        """The distribution of an answer given with certainty, over every label, so evaluation sees every option."""
        if self.kind == "bool":
            return {YES: 1.0 if answer is True else 0.0, NO: 0.0 if answer is True else 1.0}
        if self.kind == "enum":
            return {o: 1.0 if o == answer else 0.0 for o in self.options}
        chosen = set(answer) if isinstance(answer, list) else set()
        return {o: 1.0 if o in chosen else 0.0 for o in self.options}


def resolve_space(dtype: DecisionType, evidence: Mapping[str, Any]) -> AnswerSpace:
    """The answer space of one call. Raises ValidationError when the evidence supplies options that cannot be used."""
    if dtype.kind == "bool":
        return AnswerSpace(dtype.id, "bool")
    supplied, notes = _supplied(dtype, evidence) if dtype.options_from else ((), {})
    clash = sorted(set(supplied) & set(dtype.options))
    if clash:
        raise ValidationError(
            f"{dtype.id}: evidence {dtype.options_from!r} supplies {clash}, which the type already defines",
            hint=f"rename those options; {dtype.options} are reserved for {dtype.id}",
        )
    options = (*supplied, *dtype.options)
    if dtype.kind == "enum" and len(options) < 2:
        raise ValidationError(
            f"{dtype.id}: only {len(options)} option on offer, and an enum needs two to choose between",
            hint=f"supply options in {dtype.options_from!r}",
        )
    if len(options) > MAX_OPTIONS:
        raise ValidationError(
            f"{dtype.id}: {len(options)} options is more than the {MAX_OPTIONS} that the decision model can letter",
            hint=f"pass at most {MAX_OPTIONS - len(dtype.options)} options in {dtype.options_from!r}",
        )
    return AnswerSpace(dtype.id, dtype.kind, options, {**notes, **dtype.option_notes})


def _supplied(dtype: DecisionType, evidence: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, str]]:
    key = dtype.options_from
    value = evidence.get(key) if key else None
    if value is None:
        return (), {}
    if isinstance(value, Mapping):
        items = sorted(value.items())
        labels = [k for k, _ in items]
        notes = {k: (v if isinstance(v, str) else dumps(v)) for k, v in items if v not in (None, "")}
    elif isinstance(value, list):
        labels, notes = value, {}
    else:
        raise ValidationError(
            f"{dtype.id}: evidence {key!r} must list the options, or map each option to a note",
            hint=f"got {type(value).__name__}",
        )
    bad = [x for x in labels if not _usable(x)]
    if bad:
        raise ValidationError(
            f"{dtype.id}: evidence {key!r} has options that cannot be offered: {bad[:5]}",
            hint=f"each option is a non-empty single-line string of at most {MAX_OPTION_CHARS} characters",
        )
    repeated = sorted({x for x in labels if labels.count(x) > 1})
    if repeated:
        raise ValidationError(f"{dtype.id}: evidence {key!r} repeats options {repeated}", hint="list each option once")
    return tuple(labels), {k: " ".join(v.split()) for k, v in notes.items()}


def _usable(label: object) -> bool:
    return (
        isinstance(label, str) and 0 < len(label) <= MAX_OPTION_CHARS and label == label.strip() and label.isprintable()
    )

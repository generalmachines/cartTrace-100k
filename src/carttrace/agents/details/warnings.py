"""Warnings: whether a product's own label text says it is unsafe for someone with a condition.

Why this detail. A shopper who is pregnant, has asthma or takes warfarin should be steered away from a product whose
label warns against it, and an agent can only do that from the label text
(non-prescription-health.contraindications, skincare-makeup.warnings-and-directions). A product is:

- warned: a warning line ("Do not use if ...", "Ask a doctor before use if you ...", "Not suitable for ...") names
  the condition, or names "a medical condition" or "other medicines" generally, which covers every condition;
- clear: it has warning lines and none of them names the condition;
- unknown: it publishes no warnings at all. Unknown is never a pass.

Pregnancy and breastfeeding are named on their own; a general "medical condition" covers the illnesses and the
medicines, not them, because labels list pregnancy separately when it matters.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import lru_cache
from random import Random
from typing import Literal

from pydantic import JsonValue

from carttrace.agents.details.base import Proposal
from carttrace.agents.memo import per_product
from carttrace.agents.products import Product, Variant
from carttrace.agents.spec import Op
from carttrace.agents.text import fold, has_phrase

Status = Literal["warned", "clear", "unknown"]

CONDITIONS: dict[str, tuple[str, ...]] = {
    "pregnant": ("pregnan*",),
    "breastfeeding": ("breastfeed*", "breast-feed*", "breast feeding", "nursing", "lactating"),
    "asthma": ("asthma*",),
    "high blood pressure": ("high blood pressure", "hypertension", "blood pressure"),
    "heart disease": ("heart disease", "heart condition*", "heart problem*", "cardiovascular"),
    "liver disease": ("liver",),
    "kidney disease": ("kidney*", "renal"),
    "stomach ulcer": ("ulcer*",),
    "diabetes": ("diabet*",),
    "takes blood thinners": ("warfarin", "blood thinner*", "anticoagulant*", "blood-thinning"),
}
"""Each condition a shopper may mention (the `conditions` field) and the words a label uses for it."""
GENERAL = ("medical condition", "health condition", "any condition", "medication", "other medicines", "medicines")
"""Label phrases that cover every illness and medicine at once."""
_SPECIFIC_ONLY = frozenset({"pregnant", "breastfeeding"})
_SAY = {
    "pregnant": "I'm pregnant",
    "breastfeeding": "I'm breastfeeding",
    "asthma": "I have asthma",
    "high blood pressure": "I have high blood pressure",
    "heart disease": "I have a heart condition",
    "liver disease": "I have liver disease",
    "kidney disease": "I have kidney disease",
    "stomach ulcer": "I've had a stomach ulcer",
    "diabetes": "I'm diabetic",
    "takes blood thinners": "I take warfarin",
}
_STATE = {
    "pregnant": "pregnant",
    "breastfeeding": "breastfeeding",
    "asthma": "asthmatic",
    "high blood pressure": "on treatment for high blood pressure",
    "heart disease": "living with a heart condition",
    "liver disease": "living with liver disease",
    "kidney disease": "living with kidney disease",
    "stomach ulcer": "recovering from a stomach ulcer",
    "diabetes": "diabetic",
    "takes blood thinners": "on warfarin",
}
"""Each condition as it follows "I'm" in a category template ("What could I take? I'm {conditions}.")."""
WARNING_PHRASES = (
    "warning*",
    "do not use",
    "do not take",
    "don't use",
    "not suitable",
    "not recommended",
    "ask a doctor",
    "ask your doctor",
    "consult",
    "contraindicat*",
    "caution*",
    "stop use",
    "should not be used",
    "should not be taken",
    "avoid if",
    "talk to your doctor",
    "seek medical advice",
    "precaution*",
)
"""Phrases that open or carry a label warning."""


@dataclass(frozen=True, slots=True)
class WarningDetail:
    """The warnings decide: the shopper has a condition, and only the label says whether the product is safe."""

    name: str = "warnings"
    field: str = "conditions"
    op: Op = "safe_for"
    missing: str = "warnings"
    forbidden_slots: frozenset[str] = frozenset()
    hypotheses: tuple[str, ...] = ("contraindications", "warnings-and-directions", "side-effects")

    def applies(self, fields: Collection[str]) -> bool:
        return "conditions" in fields

    def propose(self, target: Product, variant: Variant, shelf: Sequence[Product], rng: Random) -> Proposal | None:
        """A condition the target's label is clear of and some comparable product warns against; a condition no
        label on the shelf mentions decides nothing, so it is never proposed."""
        others = [p for p in shelf if p.id != target.id]
        deciding = [
            c for c in CONDITIONS if status(target, c) == "clear" and any(status(p, c) == "warned" for p in others)
        ]
        return _proposal(rng.choice(deciding)) if deciding else None

    def impossible(self, pool: Sequence[Product], rng: Random) -> Proposal | None:
        everywhere = [c for c in CONDITIONS if pool and all(status(p, c) == "warned" for p in pool)]
        return _proposal(rng.choice(everywhere)) if everywhere else None

    def check(self, value: JsonValue, product: Product, variant: Variant) -> bool | None:
        wanted = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        verdicts = [status(product, c) for c in wanted if isinstance(c, str)]
        if not verdicts:
            return None
        if "warned" in verdicts:
            return False
        return None if "unknown" in verdicts else True


def status(product: Product, condition: str) -> Status:
    """What the product's own warnings say about one condition of CONDITIONS. Read once per product and condition
    within a memo run (carttrace.agents.memo)."""
    return per_product("warnings", product, condition, lambda: _status(product, condition))


def _status(product: Product, condition: str) -> Status:
    names = CONDITIONS.get(condition)
    if names is None:
        return "unknown"
    lines = warning_lines(product)
    if not lines:
        return "unknown"
    general = condition not in _SPECIFIC_ONLY
    for line in lines:
        folded = fold(line)
        if any(_named(folded, name) for name in names):
            return "warned"
        if general and any(has_phrase(folded, phrase) for phrase in GENERAL):
            return "warned"
    return "clear"


def warning_lines(product: Product) -> list[str]:
    """Every line of the product's text that carries a warning, and the line after a "Warnings:" heading."""
    lines = list(product.lines)
    out: list[str] = []
    for index, line in enumerate(lines):
        folded = fold(line)
        if not any(has_phrase(folded, phrase) for phrase in WARNING_PHRASES):
            continue
        out.append(line)
        if folded.rstrip().endswith(":") or len(folded.split()) <= 2:
            out.extend(lines[index + 1 : index + 3])
    return out


def _named(text: str, name: str) -> bool:
    return _name_pattern(name).search(text) is not None


@lru_cache(maxsize=1 << 12)
def _name_pattern(name: str) -> re.Pattern[str]:
    stem = re.escape(name.rstrip("*"))
    tail = "" if name.endswith("*") else r"(?![a-z0-9])"
    return re.compile(rf"(?<![a-z0-9]){stem}{tail}")


def _proposal(condition: str) -> Proposal:
    say = _SAY[condition]
    return Proposal(value=[condition], say=say, sentence=f"{say}, so it has to be safe for me.", slot=_STATE[condition])

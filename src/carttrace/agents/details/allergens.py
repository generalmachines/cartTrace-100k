"""Allergens: whether a product's own ingredient or allergen statement shows it is safe for someone who must avoid
a substance.

Why this detail. Allergen avoidance is a hard constraint in food, non-prescription health, skin and hair care, pet
food and jewellery (nickel), and a text-only agent can respect it only when the store publishes the ingredient list
or an allergen statement as text (non-prescription-health.excipients-and-allergens, food-non-alcohol.allergen-
statements). A product is:

- contains: its ingredient list or allergen statement names the substance or one of its names (whey and casein are
  milk; spelt and malt are gluten). Cross-contact ("may contain", "made in a facility that also processes") counts,
  because a shopper with an allergy avoids it;
- free: it says so ("gluten-free", "free from milk, soy and egg"), or it has an ingredient list or allergen
  statement that names none of them;
- unknown: it publishes neither, so nobody can tell from its data. Unknown is never a pass.

Errors here are not symmetric: calling a product free when it contains the allergen certifies an unsafe cart, so
every doubt reads as "contains". Negations are removed before looking ("free from gluten, dairy and soy" names no
allergen it contains), plant butters and milks are not dairy, and "malt" must be a word, so maltodextrin is not
gluten. A missing synonym makes a product look free, so add one when a store's wording is not covered.
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

Status = Literal["contains", "free", "unknown"]

ALLERGENS: dict[str, tuple[str, ...]] = {
    "peanuts": ("peanut*", "groundnut*", "arachis", "nut", "nuts"),
    "tree nuts": (
        "almond*",
        "hazelnut*",
        "walnut*",
        "cashew*",
        "pecan*",
        "pistachio*",
        "macadamia*",
        "brazil nut*",
        "tree nut*",
        "nut",
        "nuts",
    ),
    "milk": ("milk", "dairy", "whey", "casein*", "lactose", "butter", "cream", "cheese", "yogurt", "yoghurt"),
    "lactose": ("lactose", "milk", "whey", "dairy", "cream", "cheese", "yogurt", "yoghurt"),
    "gluten": ("gluten", "wheat", "barley", "rye", "spelt", "malt", "semolina", "triticale"),
    "soy": ("soy", "soya", "soybean*", "soyabean*"),
    "egg": ("egg", "eggs", "albumen", "albumin"),
    "sesame": ("sesame", "tahini"),
    "gelatin": ("gelatin", "gelatine"),
    "fragrance": ("fragrance", "parfum", "perfume"),
    "nickel": ("nickel",),
}
"""Each allergen of the shared vocabulary (data/categories/_fields.yaml) and the words that name it on a label.
A trailing * matches any ending (peanuts, peanut oil). A bare "nuts" on a label counts for both kinds, because a
shopper cannot tell which it means."""
FREE_WORDS: dict[str, tuple[str, ...]] = {"peanuts": ("nut",), "tree nuts": ("nut",), "milk": ("dairy",)}
"""Extra words a free-from claim uses for an allergen: "nut-free" covers both kinds of nut."""
NOT_THE_ALLERGEN = (
    "shea butter",
    "cocoa butter",
    "cacao butter",
    "mango butter",
    "kokum butter",
    "murumuru butter",
    "cupuacu butter",
    "peanut butter",
    "coconut milk",
    "coconut cream",
    "almond milk",
    "oat milk",
    "rice milk",
    "soy milk",
    "soya milk",
    "milk thistle",
    "cream of tartar",
    "buckwheat",
)
"""Phrases that contain an allergen's word without being it. Peanut butter is peanuts, so it is not dairy."""
INHERENT: dict[str, tuple[str, ...]] = {
    "fragrance": (
        "cologne*",
        "eau de",
        "fragrance*",
        "parfum*",
        "perfume*",
        "body mist",
        "hair mist",
        "scented",
        "toilette",
    ),
}
"""Products that are the allergen themselves, whatever their ingredient list says: a cologne is fragrance. On
2026-10-16 "a spray cologne that can't contain any fragrance" had four witnesses, colognes whose ingredient lists
spell the scent out by its components, and 14 rows built to pay placed no order."""
_SAY = {
    "lactose": "I'm lactose intolerant",
    "fragrance": "I react badly to fragrance",
}
STATEMENT_MARKERS = (
    "ingredient*",
    "inci",
    "composition",
    "allergen*",
    "may contain",
    "traces of",
    "contains:",
    "made in a facility",
    "made in a factory",
    "produced in a facility",
    "processed in a facility",
)
"""Phrases that open an ingredient list or an allergen statement. A line that starts with "Contains" counts too;
"this bottle contains 90 tablets" does not."""
_ACTIVE_ONLY = re.compile(r"(?<![a-z])active ingredients?")
"""A medicine's "Active ingredient: ibuprofen" names only the drug, not the excipients, so it is not a full list."""
_CROSS = ("may contain", "traces of", "made in a facility", "made in a factory", "produced in a", "processed in a")
_FREE_SPAN = re.compile(r"\bfree (?:from|of)\b[^.;:\n]*")
_NEGATED_WORD = re.compile(
    r"\b(?:no|without|zero|contains no|does not contain|doesn't contain|not made with)\s+(?:added\s+|any\s+)?[a-z-]+"
)
_FREE_SUFFIX = re.compile(r"\b[a-z]+(?:-| )free\b")


@dataclass(frozen=True, slots=True)
class AllergenDetail:
    """The ingredient list decides: the shopper must avoid a substance, and only the label says what is in it."""

    name: str = "allergens"
    field: str = "allergens"
    op: Op = "excludes"
    missing: str = "ingredients"
    forbidden_slots: frozenset[str] = frozenset()
    hypotheses: tuple[str, ...] = (
        "excipients-and-allergens",
        "allergen-statements",
        "full-ingredient-list",
        "ingredient-list",
        "ingredients-free-from",
        "fragrance-allergens",
        "protein-and-allergens",
        "allergy-information",
    )

    def applies(self, fields: Collection[str]) -> bool:
        return "allergens" in fields

    def propose(self, target: Product, variant: Variant, shelf: Sequence[Product], rng: Random) -> Proposal | None:
        """An allergen the target's label shows it is free of and some comparable product contains. An allergen no
        product on the shelf contains decides nothing ("allergic to nickel", for a sweet), so it is never proposed."""
        others = [p for p in shelf if p.id != target.id]
        deciding = [
            a for a in ALLERGENS if status(target, a) == "free" and any(status(p, a) == "contains" for p in others)
        ]
        return _proposal(rng.choice(deciding)) if deciding else None

    def impossible(self, pool: Sequence[Product], rng: Random) -> Proposal | None:
        everywhere = [a for a in ALLERGENS if pool and all(status(p, a) == "contains" for p in pool)]
        return _proposal(rng.choice(everywhere)) if everywhere else None

    def check(self, value: JsonValue, product: Product, variant: Variant) -> bool | None:
        wanted = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        verdicts = [status(product, a) for a in wanted if isinstance(a, str)]
        if not verdicts:
            return None
        if "contains" in verdicts:
            return False
        return None if "unknown" in verdicts else True


def status(product: Product, allergen: str) -> Status:
    """What the product's own data says about one allergen of ALLERGENS. Read once per product and allergen within a
    memo run (carttrace.agents.memo): proposing a task weighs every allergen against every product on the shelf."""
    return per_product("allergens", product, allergen, lambda: _status(product, allergen))


def _status(product: Product, allergen: str) -> Status:
    names = ALLERGENS.get(allergen)
    if names is None:
        return "unknown"
    inherent = INHERENT.get(allergen, ())
    labels = [_without_negations(_plain(label)) for label in (product.title, product.product_type or "")]
    if any(has_phrase(label, word) for label in labels for word in inherent):
        return "contains"
    found = statements(product)
    if any(_names(_without_negations(_plain(line)), names) for line in found):
        return "contains"
    claims = (*names, *FREE_WORDS.get(allergen, ()))
    if any(_claims_free(_plain(text), claims) for text in (product.text, *product.tags)):
        return "free"
    return "free" if found else "unknown"


def statements(product: Product) -> list[str]:
    """The product's ingredient list and allergen statements: each line that opens one, and the line after a
    heading ("Ingredients:", "Allergens") that ends there."""
    lines = list(product.lines)
    out: list[str] = []
    for index, line in enumerate(lines):
        folded = _ACTIVE_ONLY.sub(" ", fold(line))
        if not (folded.startswith("contains ") or any(has_phrase(folded, marker) for marker in STATEMENT_MARKERS)):
            continue
        out.append(line)
        if folded.rstrip().endswith(":") or len(folded.split()) <= 2:
            out.extend(lines[index + 1 : index + 2])
    return out


def cross_contact(product: Product) -> bool:
    """Whether the product warns of traces or a shared facility."""
    return any(any(has_phrase(line, c) for c in _CROSS) for line in statements(product))


def _names(text: str, names: Sequence[str]) -> bool:
    return any(_name_pattern(name).search(text) for name in names)


@lru_cache(maxsize=1 << 12)
def _name_pattern(name: str) -> re.Pattern[str]:
    stem = re.escape(name.rstrip("*"))
    tail = "" if name.endswith("*") else r"(?![a-z0-9])"
    return re.compile(rf"(?<![a-z0-9]){stem}{tail}")


def _without_negations(text: str) -> str:
    """The text with every "free from ..." clause, "no X", "without X" and "X-free" removed."""
    text = _FREE_SPAN.sub(" ", text)
    text = _NEGATED_WORD.sub(" ", text)
    return _FREE_SUFFIX.sub(" ", text)


def _claims_free(text: str, words: Sequence[str]) -> bool:
    return any(pattern.search(text) for word in words for pattern in _free_patterns(word))


@lru_cache(maxsize=1 << 12)
def _free_patterns(word: str) -> tuple[re.Pattern[str], ...]:
    """The three ways a label claims to be free of `word`, in the order they are tried."""
    stem = re.escape(word.rstrip("*"))
    return (
        re.compile(rf"(?<![a-z0-9]){stem}[a-z]*(?:-| )free\b"),
        re.compile(rf"\bfree (?:from|of)\b[^.;:\n]*?(?<![a-z0-9]){stem}"),
        re.compile(rf"\b(?:contains no|does not contain|doesn't contain|no)\s+(?:added\s+)?{stem}"),
    )


def _plain(text: str) -> str:
    folded = fold(text)
    for phrase in NOT_THE_ALLERGEN:
        folded = folded.replace(phrase, " ")
    return folded


def _proposal(allergen: str) -> Proposal:
    say = _SAY.get(allergen, f"I'm allergic to {allergen}")
    sentence = f"{say}, so it can't contain any {allergen}."
    return Proposal(value=[allergen], say=say, sentence=sentence, slot=allergen, aliases=("ingredients", "free_from"))

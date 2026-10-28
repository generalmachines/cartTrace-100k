"""Niche tasks only where the products carry the niche's data as a field (D104): the gate on open stand-ins' tasks.

Why. A category's niches (D92, data/categories/niches) are what its shoppers decide by beyond the basics: an
ingredient list, an allergen statement, a size chart, nutrition, warnings. The publishable stand-ins draw their
products from open catalogs, and the research of 2026-10-16 (docs/research/2026-10-16-open-fidelity.md, section 1)
found such data held as a structured field on at least 10% of the best pooled source's products for only 97 of the
227 category-niche pairs; for 55 more it appears only in title or description text, where a keyword pass finds it
about as often as it overcounts it, and for 75 nowhere. A task whose decision turns on a niche the stand-in's products
carry only as text (or not at all) is judged on data a reader may not be able to find, so the open lane writes such a
task only where every candidate product carries the niche's data as a field. Description text does not count until
rule extraction into `details` is verified (the research's fix 4).

What turns a template task on a niche. The template generator (carttrace.agents.tasks) decides by one of the data kinds below,
through its decisive detail or a predicate: a size chart (`fits`), an allergen or ingredient list (`excludes`, the
allergens detail, an active ingredient named from the description), warnings (`safe_for`), nutrition (`most`,
`least`) and a dietary claim (the persona's `dietary` mention). Each is a niche's data only in a category whose niche
spec names that kind (`niche_kinds`, built by the collector from carttrace.stores.niches: a data niche's kinds and the kinds
of the exposure types it is masked by). Store terms (returns, gift options, subscriptions, delivery: condition, offer
and policy niches) are the sandbox's store layer, which every stand-in serves, and option, price, stock and unit-price
needs read the basics (titles, options, prices), so neither is gated.

How a field is told from text. An open product's fixture carries its catalog's structured data under `metadata`
(carttrace.sandbox.catalogs.records.product_facts): `details` (ingredients, allergens, warnings, dosage, ...) and the
source's own `attributes`. A kind is carried when a detail of it, or an attribute whose key names it, states a value
that is not a placeholder ("Not Specified", "none", "[]").
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from carttrace.agents.products import Product
from carttrace.agents.spec import Decisive, Predicate

GATED_KINDS: Final = ("size_chart", "allergens", "ingredients", "warnings", "nutrition", "dietary")
"""The data kinds a template task can decide by that are a product niche's data."""
DECISIVE_GATED: Final[Mapping[str, str]] = {
    "size_chart": "size_chart",
    "allergens": "allergens",
    "warnings": "warnings",
    "nutrition": "nutrition",
}
"""A decisive detail (Decisive.detail) to the data kind it reads."""
_PLACEHOLDER: Final = re.compile(
    r"^\s*(|-+|n/?a|na|none|null|nil|\[\]|not specified|not applicable|no|unknown|attributes_\w+|0)\s*$", re.IGNORECASE
)
_DETAILS: Final[Mapping[str, tuple[str, ...]]] = {
    "size_chart": ("size_chart",),
    "allergens": ("allergens", "ingredients"),
    "ingredients": ("ingredients",),
    "warnings": ("warnings",),
    "nutrition": ("nutrition",),
    "dietary": ("allergens",),
}
"""The `details` keys that carry a kind as a field."""
_KEYS: Final[Mapping[str, re.Pattern[str]]] = {
    "size_chart": re.compile(r"size[ _-]?(chart|guide)", re.IGNORECASE),
    "allergens": re.compile(r"allergen|ingredient", re.IGNORECASE),
    "ingredients": re.compile(r"ingredient|\binci\b", re.IGNORECASE),
    "warnings": re.compile(r"warning|caution|contraindication", re.IGNORECASE),
    "nutrition": re.compile(
        r"nutrition|calorie|protein|carbohydrate|\bsugars?\b|\bfat\b|sodium|energy|dietary fib|serving size",
        re.IGNORECASE,
    ),
    "dietary": re.compile(
        r"allergen|dietary|\bdiet\b|gluten|vegan|vegetarian|halal|kosher|lactose|dairy|sugar[ -]?free", re.IGNORECASE
    ),
}
"""The `attributes` keys that carry a kind as a field (the research's attribute key rules, tools/open_fidelity.py)."""
_HTML_KEY: Final = re.compile(r"[<>\"=]|^.{61,}$")
"""Lazada keys that are HTML fragments, not field names."""


_OP_KINDS: Final[Mapping[str, str]] = {
    "fits": "size_chart",
    "excludes": "allergens",
    "safe_for": "warnings",
    "most": "nutrition",
    "least": "nutrition",
}
_MENTION_KINDS: Final[Mapping[str, str]] = {"dietary": "dietary", "active_ingredient": "ingredients"}
"""A `mentions` predicate's field to the kind it reads (a dietary claim, an active ingredient)."""


def predicate_kind(predicate: Predicate) -> str | None:
    """The gated data kind a predicate decides by, or None."""
    if predicate.op == "mentions":
        return _MENTION_KINDS.get(predicate.field)
    return _OP_KINDS.get(predicate.op)


def task_kinds(decisive: Decisive, predicates: Iterable[Predicate]) -> set[str]:
    """Every gated data kind a task decides by: its decisive detail's and its predicates'."""
    out = {k for p in predicates if (k := predicate_kind(p)) is not None}
    kind = DECISIVE_GATED.get(decisive.detail)
    if kind is not None:
        out.add(kind)
    return out


def carried(kind: str, product: Product) -> bool:
    """Whether the product states `kind` as a structured field: a detail of it, or an attribute whose key names it,
    with a value that is not a placeholder."""
    meta = product.metadata
    details = meta.get("details")
    if isinstance(details, Mapping) and any(_stated(details.get(k)) for k in _DETAILS.get(kind, ())):
        return True
    if any(_stated(meta.get(k)) for k in _DETAILS.get(kind, ())):
        return True
    attributes = meta.get("attributes")
    pattern = _KEYS.get(kind)
    if pattern is None or not isinstance(attributes, Mapping):
        return False
    return any(pattern.search(str(k)) and not _HTML_KEY.search(str(k)) and _stated(v) for k, v in attributes.items())


def _stated(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, Mapping | list | tuple):
        return bool(value)
    return not _PLACEHOLDER.match(str(value))


@dataclass(frozen=True, slots=True)
class NicheGate:
    """One category's gate: the gated data kinds that are its niches' data, each with the niches it decides."""

    category: str
    niches: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """Gated data kind to the category's niche ids whose data it is."""

    @classmethod
    def of(cls, category: str, niche_kinds: Mapping[str, Iterable[str]]) -> NicheGate:
        """The gate of a category from its data niches' kinds (niche id to the brief's data kinds it holds)."""
        by_kind: dict[str, list[str]] = {}
        for niche, kinds in sorted(niche_kinds.items()):
            for kind in sorted(set(kinds) & set(GATED_KINDS)):
                by_kind.setdefault(kind, []).append(niche)
        return cls(category, {k: tuple(v) for k, v in sorted(by_kind.items())})

    def gates(self, kind: str) -> bool:
        return kind in self.niches

    def allows(self, kind: str | None, products: Sequence[Product]) -> bool:
        """Whether a task may decide by `kind` over `products`: the kind is no niche's data here, or every product
        carries it as a field."""
        if kind is None or kind not in self.niches:
            return True
        return bool(products) and all(carried(kind, p) for p in products)

    def lacking(self, decisive: Decisive, predicates: Iterable[Predicate], pool: Sequence[Product]) -> list[str]:
        """The niches a task turns on that its candidates do not all carry as a field; empty when the task may be
        written."""
        out: list[str] = []
        for kind in sorted(task_kinds(decisive, predicates)):
            if not self.allows(kind, pool):
                out += [n for n in self.niches[kind] if n not in out]
        return out


def niche_mix(decisive: Decisive, predicates: Iterable[Predicate], gate: NicheGate) -> list[str]:
    """The category's niches a task turns on (for a report of the open tasks' niche mix)."""
    out: list[str] = []
    for kind in sorted(task_kinds(decisive, predicates)):
        out += [n for n in gate.niches.get(kind, ()) if n not in out]
    return out


__all__ = [
    "DECISIVE_GATED",
    "GATED_KINDS",
    "NicheGate",
    "carried",
    "niche_mix",
    "predicate_kind",
    "task_kinds",
]

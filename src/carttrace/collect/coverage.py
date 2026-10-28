"""Spreading a store's tasks over its catalog, and recording how much of the catalog they cover.

Why. A store's tasks used to be the first N the generator drew, so a store of 40 products was benchmarked on the few
the draw happened to favour, every run, and what the benchmark said about the store was what it said about four
products. Selection is a seam beside task writing (carttrace.agents.tasks writes each task; this module only chooses among
them), and it does three things:
- **Stratifies.** Products are put in a rotation order that interleaves strata: the product's type (the shopper's
  noun, the shelf it sits on, as carttrace.agents names it) and its price band (the store's cheapest third, middle third or
  dearest third). Tasks are then balanced by intent, and by whether the task's decisive detail (a size chart, an
  allergen list, warnings) is in the store's data or the task falls back to a variant option, in the share the
  candidates have.
- **Caps repeats.** No product carries more than `cap` of a store's tasks. Tasks written around a whole shelf (the
  deliberately impossible ones) keep the generator's share and are spread over product types.
- **Rotates.** The rotation order is a seeded permutation of the store's products keyed by the store; window `r` of it
  starts `r * count` products in, so successive windows (one per UTC day, or per autopilot round) tile the catalog
  and cover new products until it is exhausted, deterministically, with no state but the window number.

The candidates are drawn from the task generator itself: the first draw with the seed a store has always had, then
more draws (new seeds) until every product of the window has a task or `POOL_DRAWS` is reached. Same inputs, same
draws, same selection, same bytes.
"""

from __future__ import annotations

import bisect
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from carttrace.agents.nouns import noun_of
from carttrace.agents.products import read_products
from carttrace.core.ids import seeded_rng
from carttrace.trace.schema import TaskRef

Selection = Literal["first", "spread"]
PRICE_BANDS: Final = ("low", "mid", "high")
UNPRICED: Final = "unpriced"
POOL_DRAWS: Final = 8
"""Task generator draws per store at most: with ten tasks a draw, enough to find a task for most products of a
window in a catalog of 40, and few enough that building a manifest stays quick."""
_DIGITS = 4


@dataclass(frozen=True, slots=True)
class ProductFacts:
    """What selection needs to know about one product of the store's catalog."""

    id: str
    variants: frozenset[str]
    kind: str
    """The shopper's noun for it (carttrace.agents.nouns): products of one kind share a shelf."""
    band: str
    """low, mid or high: its cheapest available price against the store's other products; unpriced without one."""


def product_facts(fixtures: Sequence[Mapping[str, Any]]) -> list[ProductFacts]:
    """The catalog's products in id order, each with its kind and price band."""
    products = sorted(read_products(fixtures), key=lambda p: p.id)
    cheapest: dict[str, int] = {}
    for product in products:
        prices = [v.price for v in product.variants if v.available and v.price is not None]
        if prices:
            cheapest[product.id] = min(prices)
    ordered = sorted(cheapest.values())
    out = []
    for product in products:
        price = cheapest.get(product.id)
        band = (
            UNPRICED if price is None else PRICE_BANDS[min(2, 3 * bisect.bisect_left(ordered, price) // len(ordered))]
        )
        out.append(
            ProductFacts(
                id=product.id,
                variants=frozenset(v.id for v in product.variants),
                kind=noun_of(product).singular,
                band=band,
            )
        )
    return out


def rotation_order(facts: Sequence[ProductFacts], *, seed: int, store_id: str) -> list[str]:
    """Every product once, strata interleaved: the first product of each (kind, band) stratum, then the second of
    each, and so on, with products shuffled within a stratum and the strata shuffled, all seeded by the store."""
    strata: dict[tuple[str, str], list[str]] = {}
    for fact in facts:
        strata.setdefault((fact.kind, fact.band), []).append(fact.id)
    keys = sorted(strata)
    seeded_rng("rotation", seed, store_id, "strata").shuffle(keys)
    for key in keys:
        seeded_rng("rotation", seed, store_id, *key).shuffle(strata[key])
    order: list[str] = []
    depth = max((len(v) for v in strata.values()), default=0)
    for level in range(depth):
        order.extend(strata[key][level] for key in keys if level < len(strata[key]))
    return order


def window_of(order: Sequence[str], rotation: int | None, count: int) -> list[str]:
    """The rotation order from window `rotation`'s first product on, wrapping round: the order a window's tasks are
    chosen in. Window 0 (or None) starts at the beginning."""
    if not order:
        return []
    start = ((rotation or 0) * max(1, count)) % len(order)
    return [*order[start:], *order[:start]]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One generated task, described for selection."""

    task: TaskRef
    draw: int
    """Which generator draw it came from; 0 is the draw a store has always had."""
    position: int
    product: str | None
    """The one product the task is about; None for a task written around a whole shelf."""
    kind: str | None
    band: str | None
    decisive: bool | None
    """True when the store's data holds the task's decisive detail, False when a variant option decides it."""
    impossible: bool


def describe(tasks: Sequence[TaskRef], facts: Sequence[ProductFacts], *, draw: int = 0) -> list[Candidate]:
    """Each task with the product it is about, that product's stratum, and its decisive detail."""
    by_id = {f.id: f for f in facts}
    owner = {v: f.id for f in facts for v in f.variants} | {f.id: f.id for f in facts}
    out = []
    for position, task in enumerate(tasks):
        products = sorted({owner[entry] for entry in task.fixture if entry in owner})
        spec = task.hidden_spec
        impossible = spec.get("impossible") is True if "impossible" in spec else len(products) > 1
        product = products[0] if len(products) == 1 and not impossible else None
        first = by_id[products[0]] if products else None
        decisive = spec.get("decisive")
        detail = decisive.get("detail") if isinstance(decisive, Mapping) else None
        out.append(
            Candidate(
                task=task,
                draw=draw,
                position=position,
                product=product,
                kind=first.kind if first is not None else None,
                band=first.band if first is not None and product is not None else None,
                decisive=None if not isinstance(detail, str) else detail != "options",
                impossible=impossible,
            )
        )
    return out


def wanted_products(
    facts: Sequence[ProductFacts], *, count: int, rotation: int | None, seed: int, store_id: str
) -> list[str]:
    """The products of the window, in the order they are chosen: what the candidate draws try to cover."""
    order = window_of(rotation_order(facts, seed=seed, store_id=store_id), rotation, count)
    return order[:count]


def spread(
    candidates: Sequence[Candidate],
    facts: Sequence[ProductFacts],
    *,
    count: int,
    cap: int,
    rotation: int | None,
    seed: int,
    store_id: str,
) -> list[Candidate]:
    """`count` tasks spread over the window's products, at most `cap` per product, in the order they were drawn.

    Shelf tasks keep the share of the first draw (rounded half up), spread over product types. Each product in window
    order takes its best candidate: the intent used least so far, then the decisive kind (detail in the data or an
    option) the picks so far are short of, then the earliest draw. A pass over the window takes at most one task per
    product, and up to `cap` passes are made."""
    order = window_of(rotation_order(facts, seed=seed, store_id=store_id), rotation, count)
    first_draw = [c for c in candidates if c.draw == 0]
    share = Fraction(sum(c.impossible or c.product is None for c in first_draw), len(first_draw) or 1)
    shelf = [c for c in candidates if c.impossible or c.product is None]
    wanted_shelf = min(len(shelf), int(count * share + Fraction(1, 2)))
    possible = [c for c in candidates if not c.impossible and c.product is not None]
    present = Fraction(sum(c.decisive is True for c in possible), len(possible) or 1)
    picked: list[Candidate] = []
    intents: Counter[str] = Counter()
    uses: Counter[str] = Counter()
    taken: set[str] = set()
    by_product: dict[str, list[Candidate]] = {}
    for candidate in possible:
        by_product.setdefault(str(candidate.product), []).append(candidate)

    def take(candidate: Candidate) -> None:
        picked.append(candidate)
        taken.add(candidate.task.task_id)
        intents[candidate.task.intent] += 1
        if candidate.product is not None:
            uses[candidate.product] += 1

    for level in range(max(1, cap)):
        for product in order:
            if len(picked) >= count - wanted_shelf:
                break
            if uses[product] > level:
                continue
            options = [c for c in by_product.get(product, ()) if c.task.task_id not in taken]
            if not options:
                continue
            short = sum(c.decisive is True for c in picked) < present * (len(picked) + 1)
            take(
                min(options, key=lambda c: (intents[c.task.intent], (c.decisive is True) != short, c.draw, c.position))
            )
    rank = {product: i for i, product in enumerate(order)}
    first_of_kind: dict[str, int] = {}
    for fact in facts:
        first_of_kind[fact.kind] = min(first_of_kind.get(fact.kind, len(order)), rank.get(fact.id, len(order)))
    kinds_used = Counter(c.kind for c in picked)
    for _ in range(wanted_shelf):
        left = [c for c in shelf if c.task.task_id not in taken]
        if not left:
            break
        best = min(
            left,
            key=lambda c: (
                kinds_used[c.kind],
                first_of_kind.get(c.kind or "", len(order)),
                intents[c.task.intent],
                c.draw,
                c.position,
            ),
        )
        take(best)
        kinds_used[best.kind] += 1
    return sorted(picked, key=lambda c: (c.draw, c.position))


class StoreCoverage(BaseModel):
    """How much of a store's catalog its tasks in one mode cover."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selection: Selection
    window: int | None = None
    """The rotation window the tasks were chosen from; None when the plan does not rotate."""
    tasks: int
    asked: int | None
    """The tasks the plan asked for (tasks_per_store); None for every task drawn."""
    candidates: int
    """Distinct tasks the generator drew to choose from."""
    products: int
    """Products in the catalog the tasks were written around."""
    products_covered: list[str] = Field(default_factory=list)
    """The products a task is about, sorted."""
    share: float
    """products_covered over products."""
    product_types: int
    product_types_covered: list[str] = Field(default_factory=list)
    price_bands_covered: list[str] = Field(default_factory=list)
    decisive_in_data: int
    """Tasks whose decisive detail is in the store's data; the rest are decided by a variant option."""
    shelf_tasks: int
    """Tasks written around a whole shelf (the deliberately impossible ones)."""

    def line(self, store_id: str, kind: str) -> str:
        bands = ", ".join(self.price_bands_covered) or "none"
        pct = f"{100 * self.share:.1f}%"
        window = f", window {self.window}" if self.window is not None else ""
        return (
            f"{store_id}: {self.tasks} {kind} tasks ({self.selection}{window}) cover {len(self.products_covered)} of "
            f"{self.products} products ({pct}), {len(self.product_types_covered)} of {self.product_types} product "
            f"types, price bands {bands}"
        )


def coverage_of(
    chosen: Iterable[Candidate],
    facts: Sequence[ProductFacts],
    *,
    selection: Selection,
    window: int | None,
    asked: int | None,
    candidates: int,
) -> StoreCoverage:
    picks = list(chosen)
    covered = sorted({c.product for c in picks if c.product is not None})
    kinds = sorted({c.kind for c in picks if c.kind is not None})
    bands = [b for b in (*PRICE_BANDS, UNPRICED) if any(c.band == b for c in picks)]
    total = len(facts)
    return StoreCoverage(
        selection=selection,
        window=window,
        tasks=len(picks),
        asked=asked,
        candidates=candidates,
        products=total,
        products_covered=covered,
        share=round(len(covered) / total, _DIGITS) if total else 0.0,
        product_types=len({f.kind for f in facts}),
        product_types_covered=kinds,
        price_bands_covered=bands,
        decisive_in_data=sum(c.decisive is True for c in picks),
        shelf_tasks=sum(c.product is None for c in picks),
    )

"""The rules that choose what a store's tasks test (decision D30): which decision attributes, with which values, over
which candidate products, and why, read from the store's own data by code.

Why rules and not a model. The per-store writer of D27 let a model choose each task's needs and code only checked
the witnesses, which D20 rules out (no model writes predicates or hidden values), and a public benchmark has to say
for every task why it tests what it tests. So an attribute is chosen only when three things hold, and the reasons
are written into the hidden spec (`decision_attributes`):
- `varies_across_candidates`: among the products a shopper asking this would weigh, the attribute takes at least two
  values (two flavours, two prices, a product that is grass-fed and one that is not), so the data decides the task;
- `category_element` or `niche_check`: the attribute matters for the store's category, because a data element of
  the category names it (data/elements: the category's hypotheses, and the shared D19 elements) or one of the
  project's niche checks (data/agent/checks.yaml) reads it;
- `store_carries`: the store's brief exposes the data kind (carttrace.agents.brief), so no task turns on data the store
  does not have. Allergen-type needs (an allergen to avoid, an ingredient to have) stand only where the store carries
  ingredient or allergen data (`allergen_guard`).
A store-level attribute (a returns or shipping policy) is exempt from the first rule: it has one value per store.

How a need is built from an attribute is data too (data/agent/attributes.yaml, `mc.attributes.v1`): per data kind,
its builders, its shared element, how a category's hypotheses are matched to it, and the template opening each
builder writes. `offers` lists every attribute the rules accept for a store, each with its candidates and values;
`kind_scores` ranks the kinds (niche check severity, the element, how much of the store carries it); the writer
(carttrace.agents.intents_rules) spreads the offers under the D27 quotas. Everything is deterministic: products are read in
id order and every draw comes from a Random seeded by the store, the seed and what is drawn.

The surfaces an attribute is exposed on and the share of products that carry it come from the D19 inventory when
the store has one that names the attribute's element, else from what the needs were read from (a capture, which is
UCP over MCP; a fixture file, which is the store's own catalog).
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from pydantic import ValidationError as PydanticValidationError

from carttrace.agents.brief import KIND_IDS, StoreBrief, carried_by, kind_pattern, kind_spec
from carttrace.agents.checks import Check
from carttrace.agents.details import Proposal, detail
from carttrace.agents.measures import LEAST, MOST, basis_words, nutrient, quantity_of, unit_price
from carttrace.agents.memo import table
from carttrace.agents.nouns import Noun, noun_of
from carttrace.agents.predicates import OPTION_NAMES, holds, mentions, option_names
from carttrace.agents.products import Product, Variant
from carttrace.agents.slots import budget_above, budget_below
from carttrace.agents.spec import (
    CHOSEN_BECAUSE,
    EXPOSED_ON,
    ChosenBecause,
    DecisionAttribute,
    ExposedOn,
    HiddenSpec,
    IntentType,
    Predicate,
)
from carttrace.agents.text import fold, has_phrase
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.ids import seeded_rng
from carttrace.trace.inventory import PEOPLE_SURFACES, StoreDataInventory

RULES_SCHEMA: Final = "mc.attributes.v1"
RULE_VERSION: Final = "attributes/1"
"""The code's version of the rules; a spec's rule_version is this and the data file's version."""
DATA_DIR: Final = Path(__file__).resolve().parents[3] / "data"
DEFAULT_PATH: Final = DATA_DIR / "agent" / "attributes.yaml"
MAX_CANDIDATES: Final = 12
"""Most products one need is judged over: a shelf a shopper would weigh, not a whole catalog."""
MAX_OFFERS_PER_KIND: Final = 24
MAX_VALUES: Final = 6
ALLERGEN_KINDS: Final = frozenset({"allergens", "ingredients"})
"""The data a need about an allergen or an ingredient needs the store to carry."""
STORE_SCOPE: Final = frozenset({"policy"})
"""Builders of a store-level attribute: one value per store, so the first rule does not apply."""
POLICY_TOPICS: Final[Mapping[str, tuple[str, re.Pattern[str]]]] = {
    "returns": ("returns", re.compile(r"\breturns?\b|\brefunds?\b", re.I)),
    "shipping": ("shipping", re.compile(r"\bship(s|ping)?\b|\bdeliver", re.I)),
}
Builder = Literal[
    "option", "availability", "budget", "cheapest", "unit_price", "brand", "detail", "nutrient", "ingredient",
    "phrase", "policy", "none",
]  # fmt: skip
DETAIL_OF: Final[Mapping[str, str]] = {"allergens": "allergens", "size_chart": "size_chart", "warnings": "warnings"}
_PHRASE_END: Final = re.compile(r"[,.;:!?()\[\]|/]")
_INGREDIENTS: Final = re.compile(r"\bingredients?\b\s*:\s*(.+)", re.I)
_ITEM: Final = re.compile(r"^[a-z][a-z '-]{1,28}[a-z]$")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class KindRule(_Frozen):
    builders: list[Builder]
    element: str | None = None
    hypotheses: str = ""
    details: list[str] = Field(default_factory=list)
    field: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")
    stop: list[str] = Field(default_factory=list)
    extend: list[str] = Field(default_factory=list)
    intent: IntentType = "specify"
    intent_kind: str = "decide by the store's data"
    probe: str | None = None
    """What a D49 probe of a niche check reading this kind asks for: a phrase, or a policy topic."""

    @property
    def buildable(self) -> bool:
        return any(b != "none" for b in self.builders)


class Weights(_Frozen):
    severity: dict[str, float]
    category_element: float
    shared_element: float
    coverage: float
    distinct: float


class AttributeRules(_Frozen):
    schema_: Literal["mc.attributes.v1"] = Field(alias="schema")
    version: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    weights: Weights
    impossible_share: float = Field(ge=0, le=0.5)
    on_ask_share: float = Field(ge=0, le=1)
    openings: dict[str, str]
    kinds: dict[str, KindRule]

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    @property
    def rule_version(self) -> str:
        return f"{RULE_VERSION}@{self.version}"

    def kind(self, name: str) -> KindRule:
        found = self.kinds.get(name)
        if found is None:
            raise NotFoundError(
                f"no attribute rule for data kind {name!r}", hint="add it to data/agent/attributes.yaml"
            )
        return found


def load_rules(path: Path | None = None) -> AttributeRules:
    """The rules at `path` (default data/agent/attributes.yaml), parsed once per process and checked: every data kind
    of the brief has a rule, every rule's kind is the brief's, and every pattern compiles."""
    return _load((path or DEFAULT_PATH).resolve())


@functools.cache
def _load(path: Path) -> AttributeRules:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NotFoundError(f"no attribute rules at {path}", hint="restore data/agent/attributes.yaml") from None
    except yaml.YAMLError as exc:
        raise ValidationError(f"{path.name} is not valid YAML: {exc}") from exc
    try:
        rules = AttributeRules.model_validate(raw)
    except PydanticValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        where = ".".join(str(p) for p in first["loc"]) or "file"
        raise ValidationError(f"{path.name}: {where}: {first['msg']}", hint=f"follow {RULES_SCHEMA}") from None
    problems = [f"no rule for data kind {k}" for k in sorted(KIND_IDS - set(rules.kinds))]
    problems += [f"rule for unknown data kind {k}" for k in sorted(set(rules.kinds) - KIND_IDS)]
    for name, rule in rules.kinds.items():
        try:
            re.compile(rule.hypotheses)
        except re.error as exc:
            problems.append(f"{name}.hypotheses does not compile: {exc}")
        if "phrase" in rule.builders and (rule.field is None or kind_pattern(name) is None):
            problems.append(f"{name} builds phrases but has no field or no brief pattern")
    if problems:
        raise ValidationError(f"{path.name}: {'; '.join(problems[:5])}", hint="fix the attribute rules")
    return rules


# -- the category's elements -------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Element:
    """One data element of a category: a checklist hypothesis kept by data/elements/<slug>.yaml."""

    id: str
    statement: str
    details: tuple[str, ...] = ()


@functools.cache
def category_elements(slug: str, data_dir: Path = DATA_DIR) -> tuple[Element, ...]:
    """The category's own data elements: its hypotheses (data/categories/<slug>.yaml) that its elements file keeps
    (`hypotheses: all`, or a list). Empty for a category with neither file."""
    try:
        category = yaml.safe_load((data_dir / "categories" / f"{slug}.yaml").read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return ()
    try:
        elements = yaml.safe_load((data_dir / "elements" / f"{slug}.yaml").read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        elements = {"hypotheses": "all"}
    kept = elements.get("hypotheses", "all") if isinstance(elements, Mapping) else "all"
    out: list[Element] = []
    for h in category.get("hypotheses") or []:
        if not isinstance(h, Mapping) or not isinstance(h.get("id"), str):
            continue
        if kept != "all" and h["id"] not in (kept or []):
            continue
        details = tuple(str(d) for d in h.get("details") or [])
        out.append(Element(id=h["id"], statement=str(h.get("statement", "")), details=details))
    return tuple(out)


@functools.cache
def shared_elements(data_dir: Path = DATA_DIR) -> frozenset[str]:
    """The ids of the elements every category shares (data/elements/_elements.yaml)."""
    try:
        raw = yaml.safe_load((data_dir / "elements" / "_elements.yaml").read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return frozenset()
    elements = raw.get("elements") if isinstance(raw, Mapping) else None
    return frozenset(elements) if isinstance(elements, Mapping) else frozenset()


@dataclass(frozen=True, slots=True)
class ElementMatch:
    id: str
    """The element id: the category's own, else the shared one, else the kind's exposure taxonomy row."""
    own: bool
    """A hypothesis of the category names the kind."""
    shared: bool
    """A shared D19 element holds the kind."""

    @property
    def category_element(self) -> bool:
        return self.own or self.shared


def element_of(kind: str, category: str, rules: AttributeRules | None = None) -> ElementMatch:
    """The data element a kind is in this category: the first of the category's hypotheses whose statement matches
    the kind's pattern, else the first whose details name one of the kind's details, else the kind's shared D19
    element, else its exposure taxonomy row."""
    rule = (rules or load_rules()).kind(kind)
    elements = category_elements(category)
    pattern = re.compile(rule.hypotheses, re.I) if rule.hypotheses else None
    own = next((e for e in elements if pattern is not None and pattern.search(e.statement)), None)
    own = own or next((e for e in elements if set(e.details) & set(rule.details)), None)
    if own is not None:
        return ElementMatch(own.id, own=True, shared=bool(rule.element and rule.element in shared_elements()))
    if rule.element and rule.element in shared_elements():
        return ElementMatch(rule.element, own=False, shared=True)
    return ElementMatch(kind_spec(kind).row, own=False, shared=False)


# -- what the rules read about a store --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoreData:
    """Everything the rules read about one store: its brief, its products, the niche checks its data supports, where
    the products were read from, and its D19 inventory when it has one."""

    brief: StoreBrief
    products: tuple[Product, ...]
    checks: tuple[Check, ...] = ()
    """The niche checks an intent of this store may exercise (carttrace.agents.checks.agent_checks)."""
    source: str = "fixtures"
    """capture:<id> or fixtures: what the products were read from."""
    surfaces: tuple[ExposedOn, ...] = ()
    """Where those products were read: ucp_mcp for a capture, store_truth for a fixture file of the catalog."""
    inventory: StoreDataInventory | None = None
    seed: int = 0
    sellable: tuple[Product, ...] = field(init=False, default=())
    """The products with an available, priced variant: the only ones a need is written around."""

    def __post_init__(self) -> None:
        keep = tuple(p for p in self.products if any(v.available and v.price is not None for v in p.variants))
        object.__setattr__(self, "sellable", keep)

    @property
    def store_id(self) -> str:
        return self.brief.store_id

    @property
    def category(self) -> str:
        return self.brief.category

    @property
    def currency(self) -> str | None:
        return self.brief.currency

    def carries(self, kind: str) -> bool:
        return kind in self.brief.exposed_kinds

    def coverage(self, kind: str) -> float:
        return next((e.share for e in self.brief.exposes if e.kind == kind), 0.0)


def source_of_capture(capture_id: str) -> str:
    return f"capture:{capture_id}"


def allergen_guard(predicates: Sequence[Mapping[str, Any]], exposed: Sequence[str]) -> str | None:
    """Why a set of needs cannot stand on this store: one is about an allergen or an ingredient and the store carries
    neither ingredient nor allergen data; None when it may."""
    if ALLERGEN_KINDS & set(exposed):
        return None
    for p in predicates:
        if p.get("op") == "excludes" or p.get("field") in ("allergens", "ingredients"):
            return "the store carries no ingredient or allergen data, so no need may be about an allergen or ingredient"
    return None


@dataclass(frozen=True, slots=True)
class KindFacts:
    """Per data kind, what every attribute of the kind shares: its element, its checks, how much of the store
    carries it and where, and its rank."""

    kind: str
    element: ElementMatch
    check_ids: tuple[str, ...]
    coverage: float
    source: str
    exposed_on: tuple[ExposedOn, ...]
    carried: bool
    score: float

    def reasons(self, distinct: int, *, store_scope: bool) -> list[ChosenBecause]:
        found: set[ChosenBecause] = set()
        if distinct >= 2 and not store_scope:
            found.add("varies_across_candidates")
        if self.element.category_element:
            found.add("category_element")
        if self.carried:
            found.add("store_carries")
        if self.check_ids:
            found.add("niche_check")
        return [r for r in CHOSEN_BECAUSE if r in found]

    def relevant(self) -> bool:
        return self.element.category_element or bool(self.check_ids)


def kind_facts(data: StoreData, kind: str, rules: AttributeRules | None = None) -> KindFacts:
    lists = rules or load_rules()
    match = element_of(kind, data.category, lists)
    check_ids = tuple(c.id for c in data.checks if kind in c.kinds)
    coverage, source, surfaces = data.coverage(kind), data.source, data.surfaces
    if data.inventory is not None:
        found = inventory_exposure(data.inventory, match.id)
        if found is not None:
            coverage, surfaces = found
            source = f"inventory:{data.inventory.id}"
    carried = data.carries(kind)
    severity = max((lists.weights.severity.get(c.severity, 0.0) for c in data.checks if c.id in check_ids), default=0)
    w = lists.weights
    score = (
        severity
        + (w.category_element if match.own else w.shared_element if match.shared else 0.0)
        + w.coverage * coverage
    )
    return KindFacts(kind, match, check_ids, round(coverage, 4), source, surfaces, carried, round(score, 4))


def inventory_exposure(inventory: StoreDataInventory, element: str) -> tuple[float, tuple[ExposedOn, ...]] | None:
    """The best agent-surface coverage of an element in the inventory and the surfaces it is present or partial on
    (UCP over MCP; the store's pages, JSON-LD and microdata as `website`); None when the inventory does not hold the
    element."""
    cells = [c for c in inventory.cells if c.element == element]
    if not cells:
        return None
    seen: set[ExposedOn] = set()
    for cell in cells:
        if cell.state not in ("present", "partial"):
            continue
        if cell.surface == "ucp_mcp":
            seen.add("ucp_mcp")
        elif cell.surface in PEOPLE_SURFACES:
            seen.add("website")
    shares = [c.coverage for c in cells if c.coverage is not None and c.surface == "ucp_mcp"]
    coverage = max(shares) if shares else max((c.coverage or 0.0 for c in cells), default=0.0)
    return min(1.0, max(0.0, coverage)), tuple(s for s in EXPOSED_ON if s in seen)


# -- offers: the attributes the rules accept, with candidates and values ----------------------------------------------


@dataclass(frozen=True, slots=True)
class Value:
    """One value a need can ask for: what the predicate holds, how the opening says it, and for a detail its words."""

    value: JsonValue
    words: str
    proposal: Proposal | None = None
    op: str | None = None
    """The op when the builder has more than one (most or least)."""


@dataclass(frozen=True, slots=True)
class Offer:
    """An attribute the rules accept for a store: the kind, how the need is built, over which candidates (a shelf of
    one noun, one named product, or the store), and the values a need may ask for."""

    kind: str
    builder: Builder
    attribute: str
    """The hidden-spec field the need fills (flavor, budget, dietary), or `availability`."""
    shelf: tuple[Product, ...]
    values: tuple[Value, ...]
    impossible: tuple[Value, ...] = ()
    """Values no available candidate satisfies, for a task whose right outcome is a refusal."""
    distinct_values: int = 0
    noun: Noun | None = None
    product: Product | None = None
    option: str | None = None
    """The store's own option name behind an option or availability need."""
    need_field: str = ""
    """The predicate's field when it differs from the attribute (the option field of an availability need)."""
    need_op: str = ""

    @property
    def key(self) -> str:
        where = self.product.id if self.product is not None else (self.noun.singular if self.noun else "store")
        return f"{self.kind}:{self.attribute}:{where}"

    @property
    def store_scope(self) -> bool:
        return self.builder in STORE_SCOPE


def attribute_field(option_name: str) -> str:
    """The hidden-spec field a store's option name fills: the shared vocabulary's (Colour is color, Flavour is
    flavor), else the name made an identifier (Box is box, Nib Size is nib_size)."""
    key = fold(re.sub(r"\(.*?\)", "", option_name)).removeprefix("select ").strip()
    for candidate, names in OPTION_NAMES.items():
        if key in names:
            return candidate
    slug = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return slug if slug and slug[0].isalpha() else f"option_{slug}" if slug else "option"


def offers(data: StoreData, rules: AttributeRules | None = None) -> list[Offer]:
    """Every attribute the rules accept for the store, in data kind order: the store carries the kind, the kind
    matters for the category, and (for a product attribute) it takes at least two values among the candidates."""
    lists = rules or load_rules()
    out: list[Offer] = []
    for kind in sorted(lists.kinds):
        rule = lists.kinds[kind]
        if not rule.buildable or not data.carries(kind):
            continue
        if kind in ALLERGEN_KINDS and not ALLERGEN_KINDS & set(data.brief.exposed_kinds):
            continue
        if not kind_facts(data, kind, lists).relevant():
            continue
        found: list[Offer] = []
        for builder in rule.builders:
            make = _BUILDERS.get(builder)
            if make is not None:
                found += make(data, kind, rule)
        found = [o for o in found if (o.values or o.impossible) and (o.store_scope or o.distinct_values >= 2)]
        rng = seeded_rng("attributes.offers", data.store_id, data.seed, kind)
        if len(found) > MAX_OFFERS_PER_KIND:
            found = sorted(rng.sample(found, MAX_OFFERS_PER_KIND), key=lambda o: o.key)
        out += found
    return out


def kind_scores(data: StoreData, rules: AttributeRules | None = None) -> dict[str, KindFacts]:
    """The facts of every data kind the store's offers come from, the score included."""
    lists = rules or load_rules()
    return {kind: kind_facts(data, kind, lists) for kind in sorted(lists.kinds)}


def provenance(offer: Offer, facts: KindFacts, rules: AttributeRules | None = None) -> DecisionAttribute:
    return DecisionAttribute(
        attribute=offer.attribute,
        element_id=facts.element.id,
        check_ids=list(facts.check_ids),
        data_kind=offer.kind,
        chosen_because=facts.reasons(offer.distinct_values, store_scope=offer.store_scope),
        store_coverage=facts.coverage,
        distinct_values=offer.distinct_values,
        source=facts.source,
        exposed_on=list(facts.exposed_on),
        rule_version=(rules or load_rules()).rule_version,
    )


# -- the builders -----------------------------------------------------------------------------------------------------


def _shelves(data: StoreData, *, store_wide: bool = False) -> list[tuple[Noun | None, tuple[Product, ...]]]:
    """The sellable products grouped by the shopper's noun, each group of two or more a shelf; with `store_wide`, the
    whole store as one more shelf (noun None)."""
    by_noun: dict[str, list[Product]] = {}
    nouns: dict[str, Noun] = {}
    for product in data.sellable:
        noun = noun_of(product)
        by_noun.setdefault(noun.singular, []).append(product)
        nouns[noun.singular] = noun
    out: list[tuple[Noun | None, tuple[Product, ...]]] = [
        (nouns[k], tuple(v)) for k, v in sorted(by_noun.items()) if len(v) >= 2
    ]
    if store_wide and len(by_noun) > 1:
        out.append((None, data.sellable))
    return out


def _sample(products: Sequence[Product], rng: Random, size: int = MAX_CANDIDATES) -> tuple[Product, ...]:
    chosen = list(products) if len(products) <= size else rng.sample(list(products), size)
    return tuple(sorted(chosen, key=lambda p: p.id))


def _split(shelf: Sequence[Product], test: Callable[[Product], bool], rng: Random) -> tuple[Product, ...] | None:
    """A shelf of at most MAX_CANDIDATES with products that pass `test` and products that do not, or None when the
    shelf holds only one sort."""
    yes = [p for p in shelf if test(p)]
    no = [p for p in shelf if not test(p)]
    if not yes or not no:
        return None
    half = MAX_CANDIDATES // 2
    return _sample(_sample(yes, rng, half) + _sample(no, rng, MAX_CANDIDATES - min(len(yes), half)), rng)


def _available(product: Product) -> list[Variant]:
    return [v for v in product.variants if v.available and v.price is not None]


def _option_names(product: Product) -> list[str]:
    names = [name for v in product.variants for name, value in v.options if fold(value) not in ("default title", "")]
    return list(dict.fromkeys(names))


def _option_offers(data: StoreData, kind: str, rule: KindRule) -> list[Offer]:
    out = []
    for product in data.sellable:
        for name in _option_names(product):
            values = [x for x in dict.fromkeys(v.option(name) for v in product.variants) if x]
            possible = [x for x in values if any(v.option(name) == x and v.available for v in product.variants)]
            if len(values) < 2 or not possible:
                continue
            attribute = attribute_field(name)
            out.append(
                Offer(
                    kind=kind,
                    builder="option",
                    attribute=attribute,
                    shelf=(product,),
                    values=tuple(Value(x, x) for x in possible[:MAX_VALUES]),
                    distinct_values=len(values),
                    product=product,
                    option=name,
                    need_field=attribute,
                    need_op="eq",
                )
            )
    return out


def _unique_titles(product: Product) -> list[Variant]:
    """The variants of an option-less product whose title names it alone: a `mentions` of the title holds for that
    variant and no other ("Fine" is refused when "Extra-Fine" is a sibling)."""
    out = []
    for variant in product.variants:
        title = variant.title.strip()
        if not title or fold(title) == "default title":
            continue
        hits = [v for v in product.variants if mentions(product, v, title)]
        if hits == [variant] and not has_phrase(product.title, title):
            out.append(variant)
    return out


def _availability_offers(data: StoreData, kind: str, rule: KindRule) -> list[Offer]:
    """A variant of a product that is in stock beside one that is sold out (by option, or by a title that names it
    alone), and a sold-out product on a shelf of ones in stock, which only a refusal answers."""
    out: list[Offer] = _sold_out_products(data, kind)
    for product in data.sellable:
        if all(v.available for v in product.variants):
            continue
        names = _option_names(product)
        if names:
            name = names[0]
            values = [x for x in dict.fromkeys(v.option(name) for v in product.variants) if x]
            stocked = [x for x in values if any(v.option(name) == x and v.available for v in product.variants)]
            out_of = [x for x in values if x not in stocked]
            if stocked and out_of:
                out.append(
                    Offer(
                        kind=kind,
                        builder="availability",
                        attribute="availability",
                        shelf=(product,),
                        values=tuple(Value(x, x) for x in stocked[:MAX_VALUES]),
                        impossible=tuple(Value(x, x) for x in out_of[:MAX_VALUES]),
                        distinct_values=2,
                        product=product,
                        option=name,
                        need_field=attribute_field(name),
                        need_op="eq",
                    )
                )
            continue
        named = _unique_titles(product)
        stocked_v = [v for v in named if v.available]
        out_v = [v for v in named if not v.available]
        if stocked_v and out_v:
            out.append(
                Offer(
                    kind=kind,
                    builder="availability",
                    attribute="availability",
                    shelf=(product,),
                    values=tuple(Value(v.title, v.title) for v in stocked_v[:MAX_VALUES]),
                    impossible=tuple(Value(v.title, v.title) for v in out_v[:MAX_VALUES]),
                    distinct_values=2,
                    product=product,
                    need_field="variant",
                    need_op="mentions",
                )
            )
    return out


def _sold_out_products(data: StoreData, kind: str) -> list[Offer]:
    sellable = {p.id for p in data.sellable}
    by_noun: dict[str, list[Product]] = {}
    nouns: dict[str, Noun] = {}
    for product in data.products:
        if not product.variants:
            continue
        noun = noun_of(product)
        by_noun.setdefault(noun.singular, []).append(product)
        nouns[noun.singular] = noun
    out = []
    for singular, members in sorted(by_noun.items()):
        sold = [p for p in members if not any(v.available for v in p.variants)]
        stocked = [p for p in members if p.id in sellable]
        if not sold or not stocked:
            continue
        rng = seeded_rng("attributes.sold_out", data.store_id, data.seed, singular)
        named = _sample(sold, rng, MAX_VALUES)
        shelf = _sample(list(named) + list(_sample(stocked, rng, MAX_CANDIDATES - len(named))), rng)
        out.append(
            Offer(
                kind=kind,
                builder="availability",
                attribute="availability",
                shelf=shelf,
                values=(),
                impossible=tuple(Value(p.title, p.title) for p in named),
                distinct_values=2,
                noun=nouns[singular],
                need_field="product_name",
                need_op="is",
            )
        )
    return out


def _prices(shelf: Sequence[Product]) -> list[int]:
    return sorted({v.price for p in shelf for v in _available(p) if v.price is not None})


def _price_offers(data: StoreData, kind: str, builder: Builder) -> list[Offer]:
    currency = data.currency
    if currency is None:
        return []
    out = []
    for noun, shelf in _shelves(data):
        rng = seeded_rng("attributes.shelf", data.store_id, data.seed, kind, builder, noun.singular if noun else "")
        candidates = _sample(shelf, rng)
        prices = _prices(candidates)
        if len(prices) < 2:
            continue
        if builder == "cheapest":
            values: tuple[Value, ...] = (Value(None, "the cheapest"),)
            impossible: tuple[Value, ...] = ()
        else:
            budgets = []
            for price in prices[:-1]:
                amount = budget_above(price, currency, percent=0)
                if price <= amount < prices[-1]:
                    budgets.append(amount)
            values = tuple(Value({"amount": b, "currency": currency}, "") for b in dict.fromkeys(budgets))
            below = budget_below(prices[0], currency)
            impossible = (Value({"amount": below, "currency": currency}, ""),) if below else ()
        out.append(
            Offer(
                kind=kind,
                builder=builder,
                attribute="budget" if builder == "budget" else "price",
                shelf=candidates,
                values=values[:MAX_VALUES],
                impossible=impossible,
                distinct_values=len(prices),
                noun=noun,
                need_field="budget" if builder == "budget" else "price",
                need_op="max_price" if builder == "budget" else "cheapest",
            )
        )
    return out


def _budget_offers(data: StoreData, kind: str, rule: KindRule) -> list[Offer]:
    return _price_offers(data, kind, "budget")


def _cheapest_offers(data: StoreData, kind: str, rule: KindRule) -> list[Offer]:
    return _price_offers(data, kind, "cheapest")


def _unit_offers(data: StoreData, kind: str, rule: KindRule) -> list[Offer]:
    out = []
    for noun, shelf in _shelves(data):
        units: dict[str, list[Product]] = {}
        for product in shelf:
            held = {q.unit if q is not None else None for q in (quantity_of(product, v) for v in _available(product))}
            if len(held) == 1 and None not in held:
                units.setdefault(str(next(iter(held))), []).append(product)
        for unit, members in sorted(units.items()):
            rng = seeded_rng("attributes.shelf", data.store_id, data.seed, kind, unit, noun.singular if noun else "")
            candidates = _sample(members, rng)
            per_unit = {unit_price(p, v) for p in candidates for v in _available(p)} - {None}
            if len(candidates) < 2 and len({v.id for p in candidates for v in _available(p)}) < 2:
                continue
            if len(per_unit) < 2:
                continue
            basis = basis_words(unit)
            out.append(
                Offer(
                    kind=kind,
                    builder="unit_price",
                    attribute="unit_price",
                    shelf=candidates,
                    values=(Value({"unit": basis}, basis),),
                    distinct_values=len(per_unit),
                    noun=noun,
                    need_field="unit_price",
                    need_op="cheapest_per_unit",
                )
            )
    return out


def _brand_offers(data: StoreData, kind: str, rule: KindRule) -> list[Offer]:
    out = []
    for noun, shelf in _shelves(data):
        rng = seeded_rng("attributes.shelf", data.store_id, data.seed, kind, noun.singular if noun else "")
        by_brand: dict[str, list[Product]] = {}
        for product in shelf:
            if product.brand:
                by_brand.setdefault(product.brand, []).append(product)
        if len(by_brand) < 2:
            continue
        brands = sorted(by_brand, key=lambda b: (-len(by_brand[b]), fold(b)))[:MAX_CANDIDATES]
        picked: list[Product] = []
        for brand in brands:
            picked += list(_sample(by_brand[brand], rng, 2))
        candidates = _sample(picked, rng)
        present = sorted({p.brand for p in candidates if p.brand}, key=fold)
        if len(present) < 2:
            continue
        out.append(
            Offer(
                kind=kind,
                builder="brand",
                attribute="brand",
                shelf=candidates,
                values=tuple(Value(b, b) for b in present[:MAX_VALUES]),
                distinct_values=len(present),
                noun=noun,
                need_field="brand",
                need_op="brand",
            )
        )
    return out


def _detail_offers(data: StoreData, kind: str, rule: KindRule) -> list[Offer]:
    name = DETAIL_OF.get(kind)
    if name is None:
        return []
    reader = detail(name)
    out = []
    for noun, shelf in _shelves(data, store_wide=True):
        rng = seeded_rng("attributes.detail", data.store_id, data.seed, kind, noun.singular if noun else "")
        candidates = _sample(shelf, rng)
        found: dict[str, Value] = {}
        for target in candidates[:MAX_VALUES]:
            variants = _available(target)
            if not variants:
                continue
            proposal = reader.propose(target, variants[0], candidates, rng)
            if proposal is not None:
                found.setdefault(repr(proposal.value), Value(proposal.value, proposal.slot, proposal))
        if not found:
            continue
        readings = {
            reader.check(value.value, p, v) for value in found.values() for p in candidates for v in _available(p)
        } - {None}
        none = reader.impossible(candidates, rng)
        out.append(
            Offer(
                kind=kind,
                builder="detail",
                attribute=reader.field,
                shelf=candidates,
                values=tuple(found.values())[:MAX_VALUES],
                impossible=(Value(none.value, none.slot, none),) if none is not None else (),
                distinct_values=len(readings),
                noun=noun,
                need_field=reader.field,
                need_op=reader.op,
            )
        )
    return out


def _nutrient_offers(data: StoreData, kind: str, rule: KindRule) -> list[Offer]:
    out = []
    for noun, shelf in _shelves(data):
        for op, names in (("most", MOST), ("least", LEAST)):
            for name in names:
                stating = [p for p in shelf if nutrient(p, name) is not None]
                amounts = {nutrient(p, name) for p in stating}
                if len(stating) < 2 or len(amounts) < 2:
                    continue
                where = noun.singular if noun else ""
                rng = seeded_rng("attributes.shelf", data.store_id, data.seed, kind, op, name, where)
                candidates = _sample(stating, rng)
                out.append(
                    Offer(
                        kind=kind,
                        builder="nutrient",
                        attribute="nutrition",
                        shelf=candidates,
                        values=(Value({"nutrient": name}, name, op=op),),
                        distinct_values=len({nutrient(p, name) for p in candidates}),
                        noun=noun,
                        need_field="nutrition",
                        need_op=op,
                    )
                )
    return out


def _spellings(phrase: str) -> list[str]:
    """The other ways a phrase is written: with a hyphen for a space, or a space for a hyphen."""
    return [s for s in {phrase.replace("-", " "), phrase.replace(" ", "-")} if s != phrase]


def _holds_phrase(phrase: str, product: Product) -> bool:
    return any(mentions(product, v, phrase) for v in _available(product))


def _text_offers(
    data: StoreData, kind: str, rule: KindRule, builder: Builder, phrases_of: Callable[[Product], set[str]]
) -> list[Offer]:
    if rule.field is None:
        return []
    every: dict[str, int] = {}
    for product in data.sellable:
        for phrase in phrases_of(product):
            every[phrase] = every.get(phrase, 0) + 1
    ranked: list[str] = []
    for phrase in sorted(every, key=lambda p: (-every[p], p)):
        if not set(_spellings(phrase)) & set(ranked):
            ranked.append(phrase)
    out = []
    for noun, shelf in _shelves(data, store_wide=True):
        rng = seeded_rng("attributes.text", data.store_id, data.seed, kind, noun.singular if noun else "")
        for phrase in ranked:
            # a product that says it another way ("grass fed" for "grass-fed") is neither a yes nor a no
            others = _spellings(phrase)
            plain = [p for p in shelf if _holds_phrase(phrase, p) or not any(_holds_phrase(o, p) for o in others)]
            candidates = _split(plain, functools.partial(_holds_phrase, phrase), rng)
            if candidates is None:
                continue
            out.append(
                Offer(
                    kind=kind,
                    builder=builder,
                    attribute=rule.field,
                    shelf=candidates,
                    values=(Value(phrase, phrase),),
                    distinct_values=2,
                    noun=noun,
                    need_field=rule.field,
                    need_op="mentions",
                )
            )
    return out


def phrases_in(product: Product, kind: str, rule: KindRule) -> set[str]:
    """The phrases of a text kind the product's text and tags carry, as the brief's detector finds them: folded,
    extended past a bare start ("made in" to "made in canada"), and without the rule's stop phrases."""
    pattern = kind_pattern(kind)
    if pattern is None:
        return set()
    found: set[str] = set()
    for line in (*product.lines, *product.tags):
        for match in pattern.finditer(line):
            phrase = fold(match.group(0)).strip(" -:")
            if phrase in rule.extend:
                rest = _PHRASE_END.split(line[match.end() :], maxsplit=1)[0].split()[:3]
                if not rest:
                    continue
                phrase = fold(f"{phrase} {' '.join(rest)}")
            if phrase in rule.stop or len(phrase) < 3 or len(phrase) > 40:
                continue
            found.add(phrase)
    return found


def ingredients_in(product: Product) -> set[str]:
    """The items of the product's ingredient list (the line after "Ingredients:"), one to three plain words each."""
    found: set[str] = set()
    for line in product.lines:
        match = _INGREDIENTS.search(line)
        if match is None:
            continue
        for item in re.split(r"[,;()\[\]]|\band\b", match.group(1)):
            text = fold(item).strip(" .*:")
            if _ITEM.match(text) and len(text.split()) <= 3:
                found.add(text)
    return found


def _phrase_offers(data: StoreData, kind: str, rule: KindRule) -> list[Offer]:
    return _text_offers(data, kind, rule, "phrase", lambda p: phrases_in(p, kind, rule))


def _ingredient_offers(data: StoreData, kind: str, rule: KindRule) -> list[Offer]:
    return _text_offers(data, kind, rule, "ingredient", ingredients_in)


def _policy_offers(data: StoreData, kind: str, rule: KindRule) -> list[Offer]:
    topic = POLICY_TOPICS.get(kind)
    if topic is None or not any(topic[1].search(text) for text in data.brief.policies):
        return []
    rng = seeded_rng("attributes.policy", data.store_id, data.seed, kind)
    out = []
    for product in _sample(data.sellable, rng, 3):
        out.append(
            Offer(
                kind=kind,
                builder="policy",
                attribute="policy",
                shelf=(product,),
                values=(Value({"topic": topic[0]}, topic[0]),),
                distinct_values=1,
                product=product,
                need_field="policy",
                need_op="policy",
            )
        )
    return out


_BUILDERS: Final[Mapping[str, Callable[[StoreData, str, KindRule], list[Offer]]]] = {
    "option": _option_offers,
    "availability": _availability_offers,
    "budget": _budget_offers,
    "cheapest": _cheapest_offers,
    "unit_price": _unit_offers,
    "brand": _brand_offers,
    "detail": _detail_offers,
    "nutrient": _nutrient_offers,
    "ingredient": _ingredient_offers,
    "phrase": _phrase_offers,
    "policy": _policy_offers,
}


def probe_offer(data: StoreData, kind: str, rules: AttributeRules | None = None) -> tuple[Offer, bool] | None:
    """The offer a D49 probe of a niche check makes from `kind`, and whether no candidate satisfies it; None when the
    kind has no probe or the store has nothing for sale. Unlike `offers`, the store need not carry the kind: the
    probe asks what a shopper of the category asks, and a store that lacks it is what it measures. A phrase probe
    weighs a seeded sample of the store's largest shelf; a policy probe one seeded product."""
    lists = rules or load_rules()
    rule = lists.kinds.get(kind)
    if rule is None or rule.probe is None or not data.sellable:
        return None
    rng = seeded_rng("attributes.probe", data.store_id, data.seed, kind)
    if "policy" in rule.builders:
        product = _sample(data.sellable, rng, 1)[0]
        topic = rule.probe
        offer = Offer(
            kind=kind,
            builder="policy",
            attribute="policy",
            shelf=(product,),
            values=(Value({"topic": topic}, topic),),
            distinct_values=1,
            product=product,
            need_field="policy",
            need_op="policy",
        )
        return offer, False
    if rule.field is None:
        return None
    shelves = _shelves(data) or [(None, data.sellable)]
    noun, shelf = max(shelves, key=lambda s: (len(s[1]), s[0].singular if s[0] else ""))
    candidates = _sample(shelf, rng)
    phrase = rule.probe
    stating = sum(1 for p in candidates if _holds_phrase(phrase, p))
    offer = Offer(
        kind=kind,
        builder="phrase",
        attribute=rule.field,
        shelf=candidates,
        values=(Value(phrase, phrase),),
        distinct_values=2 if 0 < stating < len(candidates) else 1,
        noun=noun,
        need_field=rule.field,
        need_op="mentions",
    )
    return offer, stating == 0


# -- the template generator's draws ----------------------------------------------------------------------------------


DECISIVE_KINDS: Final[Mapping[str, str]] = {
    "size_chart": "size_chart",
    "allergens": "allergens",
    "warnings": "warnings",
    "options": "options",
    "price": "price",
    "nutrition": "nutrition",
    "stock": "availability",
}
"""The template generator's decisive details (carttrace.agents.tasks) as the brief's data kinds."""


def kind_of(detail_name: str, field_name: str | None) -> str | None:
    """The data kind a spec's decisive detail is: a unit price is price.unit, a bundle is bundle."""
    if field_name == "unit_price":
        return "price.unit"
    if detail_name == "options" and field_name == "bundle":
        return "bundle"
    return DECISIVE_KINDS.get(detail_name)


def distinct_among(predicate: Predicate, pool: Sequence[Product], stocked_out: frozenset[str] = frozenset()) -> int:
    """How many values the predicate's attribute takes among the pool: option values, prices, unit prices, nutrient
    amounts, brands, or for any other need the distinct readings (holds and does not) of its check."""
    variants = [(p, v) for p in pool for v in p.variants if v.id not in stocked_out]
    op = predicate.op
    if op == "eq":
        names = option_names(predicate.field, predicate.option)
        return len({fold(found) for _, v in variants if (found := v.option(*names))})
    if op in ("max_price", "cheapest"):
        return len({v.price for _, v in variants if v.available and v.price is not None})
    if op == "cheapest_per_unit":
        return len({unit_price(p, v) for p, v in variants if v.available} - {None})
    if op in ("most", "least"):
        name = predicate.value.get("nutrient") if isinstance(predicate.value, Mapping) else None
        return len({nutrient(p, name) for p in pool} - {None}) if isinstance(name, str) else 0
    if op == "brand":
        return len({fold(p.brand) for p in pool if p.brand})
    return len({holds(predicate, p, v) for p, v in variants} - {None})


def _coverage(kind: str, products: Sequence[Product]) -> float:
    """The share of the store's products that carry the data kind; remembered per product list within a memo run
    (carttrace.agents.memo), since every task of a store asks it of the same products."""
    memo = table("coverage")
    slot = (kind, tuple(map(id, products)))
    hit = memo.get(slot) if memo is not None else None
    if hit is None:
        share = sum(1 for p in products if carried_by(kind, p)) / len(products) if products else 0.0
        hit = (tuple(products), share)
        if memo is not None:
            memo[slot] = hit
    found: float = hit[1]
    return found


def describe(
    spec: HiddenSpec,
    pool: Sequence[Product],
    *,
    category: str,
    products: Sequence[Product],
    checks: Sequence[Check] = (),
    rules: AttributeRules | None = None,
) -> list[DecisionAttribute]:
    """The decision attribute of a task the template generator drew: its decisive detail as a data kind, the need
    that detail decides, and the same reasons and readings the rules record for their own draws. Empty when the spec
    names no need its detail decides. The generator is not told where its fixtures came from, so the source is
    `fixtures` and no surface is named."""
    lists = rules or load_rules()
    kind = kind_of(spec.decisive.detail, spec.decisive.field)
    if kind is None or kind not in lists.kinds:
        return []
    named = spec.predicate(spec.decisive.field) if spec.decisive.field else None
    need = named or next((p for p in spec.predicates if p.op == "eq" and p.value is not None), None)
    if kind == "availability":
        attribute = "availability"
        readings = {v.available and v.id not in spec.stocked_out for p in pool for v in p.variants}
        distinct = len(readings)
    elif need is not None:
        attribute, distinct = need.field, distinct_among(need, pool, spec.stocked_out)
    else:
        return []
    match = element_of(kind, category, lists)
    check_ids = [c.id for c in checks if kind in c.kinds and c.shared != "access"]
    coverage = _coverage(kind, products)
    found: set[ChosenBecause] = set()
    if distinct >= 2:
        found.add("varies_across_candidates")
    if match.category_element:
        found.add("category_element")
    if coverage > 0:
        found.add("store_carries")
    if check_ids:
        found.add("niche_check")
    return [
        DecisionAttribute(
            attribute=attribute,
            element_id=match.id,
            check_ids=check_ids,
            data_kind=kind,
            chosen_because=[r for r in CHOSEN_BECAUSE if r in found],
            store_coverage=round(coverage, 4),
            distinct_values=distinct,
            source="fixtures",
            exposed_on=[],
            rule_version=lists.rule_version,
        )
    ]


# -- why an attribute is refused -------------------------------------------------------------------------------------


def explain(data: StoreData, kind: str, attribute: str, rules: AttributeRules | None = None) -> list[str]:
    """The rules an attribute of `kind` fails for this store, in the order they are checked; empty when the rules
    offer it."""
    lists = rules or load_rules()
    if kind not in KIND_IDS:
        return [f"unknown data kind {kind!r}"]
    rule = lists.kind(kind)
    exposed = set(data.brief.exposed_kinds)
    failed: list[tuple[Callable[[], bool], str]] = [
        (lambda: not data.carries(kind), f"store_carries: the store does not expose {kind}"),
        (
            lambda: kind in ALLERGEN_KINDS and not ALLERGEN_KINDS & exposed,
            "store_carries: the store carries no ingredient or allergen data",
        ),
        (
            lambda: not kind_facts(data, kind, lists).relevant(),
            f"category_element, niche_check: no data element or niche check of {data.category} reads {kind}",
        ),
        (lambda: not rule.buildable, f"no rule builds a need from {kind} yet"),
    ]
    reason = next((text for test, text in failed if test()), None)
    if reason is not None:
        return [reason]
    known = sorted({o.attribute for o in offers(data, lists) if o.kind == kind})
    if attribute in known:
        return []
    if not attribute:
        return [f"varies_across_candidates: no attribute of {kind} takes two values among any candidates"]
    offered = f"; the rules offer {', '.join(known)}" if known else ""
    return [f"varies_across_candidates: {attribute!r} takes fewer than two values among any candidates{offered}"]


__all__ = [
    "ALLERGEN_KINDS",
    "DEFAULT_PATH",
    "MAX_CANDIDATES",
    "RULES_SCHEMA",
    "RULE_VERSION",
    "AttributeRules",
    "Element",
    "ElementMatch",
    "KindFacts",
    "KindRule",
    "Offer",
    "StoreData",
    "Value",
    "allergen_guard",
    "attribute_field",
    "category_elements",
    "describe",
    "distinct_among",
    "element_of",
    "explain",
    "ingredients_in",
    "inventory_exposure",
    "kind_facts",
    "kind_of",
    "kind_scores",
    "load_rules",
    "offers",
    "phrases_in",
    "probe_offer",
    "provenance",
    "shared_elements",
    "source_of_capture",
]

"""Tasks from fixtures and intents: an opening, a hidden spec of predicates, and the witnesses that satisfy it.

Why written this way. The benchmark asks, per category, whether a store exposes the detail that decides a purchase,
so every task is built around one: the shopper knows a measurement but not the size, so the size chart decides; the
shopper must avoid an allergen or has a condition, so the ingredient list or the warnings decide. Where a category's
fixtures expose no such detail, a structured variant option decides instead (decisive.detail = "options").

For each of the five first intents (replenish, specify, compare, discover, express) a task:
- picks a target variant (seeded), and the shelf a shopper would look at: the named product for replenish, every
  product of the same kind otherwise;
- states the decisive fact in the opening (three tasks in four, and every express task), or keeps it for when the
  agent asks; the target's other options are stated, kept for asking, or left as no preference;
- uses one of the category's templates when every slot can be filled from the store's data and none would give the
  answer away (a stated size makes a size chart moot), else a fallback opening from data/prompts/shopper-v2.yaml.
  Every filled slot becomes a predicate, so the opening never states a constraint the judge does not check;
- computes each predicate's `satisfied_by` and the witnesses from the fixture data with the checks the reference
  agent runs (carttrace.agents.predicates).

A seeded share of tasks (12 percent, the lab's preflight figure) is deliberately impossible: the shopper's
measurement is beyond every size, every product on the shelf contains their allergen or warns against their
condition, or the budget is below the cheapest product. Refusing is the right outcome there. Every task names at
least one witness or is marked impossible, and `validate_task` refuses any that does not.

A witness holds the units a line asks for. A world that says how many units a variant has (a sandbox snapshot's
`stock`) limits the shopper's count: a task drawn at three units of a variant the world holds one of asks for one, and
a variant holding fewer units than the line needs is never a witness, so a task none can fill is marked impossible on
purpose. The judge found 33 replica tasks of 2026-10-12 asking for more units than the world held (store-526
ep-e53644733cd4: three of a violin with one in stock), judged as the agent's failure. A world that says nothing about
stock limits nothing.

Compare tasks name their candidates. The category templates for compare turn on soft criteria (warranty, battery
life) no predicate can check, so instead the shopper names two to four products of one shelf from the store's own
catalog and one or two criteria that the fixture data settles exactly: a decisive detail (fits a 102 cm chest,
contains no milk, safe in pregnancy) and, now and then, the best of what is left (the cheapest, the cheapest per
tablet or per 100 g, the most protein or the least sugar per 100 g, carttrace.agents.measures). An impossible compare task
names candidates none of which qualifies. Where no named compare can be written, the shelf fallback opening is used.

Generator 2 adds the buyer block (carttrace.agents.buyer): the persona's address in the store's market, its reserved email
and phone, a delivery preference and the budget's scope, drawn from a Random of its own so a seed's products and
predicates are as before. A budget that covers shipping and tax ("all in") is set above the target's all-in price
with the preferred delivery, and its `satisfied_by` is computed with the sandbox's own market table, so witnesses stay
exact.

Generator 3 (decision D20) brings the variety the project asked for, all of it data (carttrace.agents.intents, data/agent):
- five more intents. gift: for someone else, with a delivery deadline (the buyer's stated preference) and a gift
  message; bundle: two or three products that go together, one of each (an `all_of` predicate and a line count);
  budget_max: a named product and a budget, and as many as the budget buys; substitute: a named product the spec's
  own sealed stockout fault takes out of stock, so the closest substitute (another product of its shelf meeting every
  other predicate) is right when the shopper allows one, and a refusal when they do not; policy: a specify task with a
  question the store's policies answer first. Each has witnesses from the fixtures and an impossible variant;
- the persona's constraints become predicates where the shelf's data makes them decide: their allergen as `excludes`,
  their condition as `safe_for`, their measurement as `fits`, their diet as a `dietary` mention;
- the persona's style (carttrace.agents.intents.Variety) sets how much is stated, how tight the budget is, which delivery it
  leans to, whether it buys two, and whether one on_ask detail is forgotten the first time it is asked;
- the buyer block carries the persona's own name and address.
Asked for no intents, the generator writes the five first intents as it always did: the same slots, the same draws.

Deterministic: fixtures are read in id order, every draw comes from a Random seeded by the store, the category, the
seed, the intent and the task's index, and task ids are hashes of the task's content.
"""

from __future__ import annotations

import json
import re
import string
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from random import Random
from typing import Any, Final, Protocol

from carttrace.agents.attributes import describe
from carttrace.agents.buyer import (
    CURRENCY_MARKETS,
    BuyerPlan,
    all_in_total,
    buyer_block,
    draw_plan,
    market_of,
    priced_market,
    stated_sentences,
    with_deadline,
)
from carttrace.agents.category import CategoryLike, CategorySpec, Template, category_spec
from carttrace.agents.checks import Check, load_checklists
from carttrace.agents.consistency import contradictions
from carttrace.agents.details import Detail, Proposal, details_for
from carttrace.agents.intents import (
    LEGACY,
    POLICY_TOPICS,
    IntentWords,
    Variety,
    load_intent_words,
    persona_constrains,
    persona_dietary,
    persona_fit,
    persona_impossible,
    persona_measures,
    persona_proposal,
    variety_for,
)
from carttrace.agents.measures import LEAST, MOST, basis_words, common_unit, nutrient
from carttrace.agents.memo import remembering, table
from carttrace.agents.niche_gate import DECISIVE_GATED, NicheGate
from carttrace.agents.nouns import (
    NAME_LEAD,
    NAME_LEAD_MAX,
    SHORT_TITLE_WORDS,
    Noun,
    kind_noun,
    names_product,
    noun_from,
    noun_of,
    plain_value,
    spoken_name,
    tidy,
)
from carttrace.agents.phrases import Phrase, Phrases, capital, fill, load_phrases
from carttrace.agents.predicates import CART_OPS, best, bundle_items, holds, item_holds, mentions, passing, satisfied_by
from carttrace.agents.products import Product, Variant, read_products
from carttrace.agents.slots import SlotValue, budget_above, budget_below, money, option_slots, text_slots
from carttrace.agents.spec import (
    ALL_INTENTS,
    INTENTS,
    NEW_INTENTS,
    SPEC_SCHEMA,
    SUPERLATIVES,
    Decisive,
    FulfillmentKind,
    HiddenSpec,
    IntentType,
    Op,
    Predicate,
    Source,
    validate_spec,
)
from carttrace.agents.text import fold, has_phrase
from carttrace.agents.wallet import Persona, Personas, load_personas
from carttrace.core.errors import ValidationError
from carttrace.core.ids import seeded_rng, stable_id
from carttrace.sandbox.catalogs.kinds import Kinds
from carttrace.trace.schema import StoreRef, TaskRef

GENERATOR = "mc.agents.tasks/5"
"""4 (2026-10-13, D30): every task records its decision attribute and why (`decision_attributes`), and every need
says the rules chose it (`chosen_by: rules`); the draws are the draws of 3.
3 (2026-10-12): ten intents, persona variety and constraints, the persona's name on its address, sealed faults.
2 (2026-10-11): the buyer block, budget scopes, and compare tasks that name their candidates."""
IMPOSSIBLE_SHARE = 0.12
"""Share of a plan kept deliberately impossible, as the lab's preflight did (10 to 15 percent asked for)."""
STATED_DECISIVE = 0.75
"""Share of possible tasks whose decisive fact is in the opening; the rest keep it for when the agent asks."""
TARGET_TRIES = 16
"""Target variants tried per task before settling for one whose data has no decisive detail."""
GATE_TRIES = 4
"""Draws again for a slot whose draft turned on a niche its products lack as a field (D104), before one with no
decisive detail."""
SET_OPS = CART_OPS
"""Ops not checked one variant at a time when finding witnesses."""
SECOND_CRITERION = 0.5
"""Share of named compare tasks with a decisive detail that also ask for the best of what is left."""
COMPARE_BUDGET = 0.3
"""Share of named compare tasks with a budget the shopper gives when asked."""
CORE_FEATURE_SHARE = 0.4
"""D117: of the open lane's specify-shaped tasks, the share that also turn on a union core fact of the target (made
of solid wood, machine washable, fits a model, BPA-free), stated in the opening, where the shelf has products without
it, so the fact decides between them. Drawn from a Random of its own, so every other draw is as before."""
CARE_FEATURES: Final[Mapping[str, str]] = {
    "machine wash": "I need to be able to machine wash it",
    "dishwasher safe": "it has to be dishwasher safe",
}
"""Care phrases a shopper decides by, to how they say it."""
SUBSTITUTE_OK = 0.7
"""Share of substitute tasks whose shopper accepts a substitute; the rest want a refusal when the product is out."""
SUBSTITUTE_STATED = 0.6
"""Share of substitute tasks that say in the opening whether a substitute will do."""
BUNDLE_BUDGET = 0.5
"""Share of bundle tasks with a budget."""
_CHEAPEST = re.compile(r"\bcheapest\b|\bleast expensive\b|\blowest price\b", re.IGNORECASE)
_OWN_SLOTS = frozenset({"product_type", "product_name", "brand", "budget", "quantity"})
"""Slots the generator fills itself rather than from a variant option or the product's text."""
UNCHECKED = re.compile(
    r"repeat delivery|subscri|auto-?ship|recurring|gift.?wrap|\breturn|exchange|drows|truer to size"
    r"|better value|deliver|refill|cover for|charger|replacement|spare|filter for|case for|parts? for|battery for"
    r"|add a message|\b(?:two|three|four|five|six|couple|dozen)\b",
    re.IGNORECASE,
)
"""Template wording that adds a constraint no predicate can check from product data: a subscription, returns, a
delivery date, "non-drowsy", an accessory for the named product ("refills for my ..."), a gift message, or a count
written as a word ("two more"). Such a template is not used, because the judge would pass a cart that ignores it."""
NO_CATEGORY_TEMPLATES: frozenset[IntentType] = frozenset({"compare", *NEW_INTENTS})
"""Intents whose category templates turn on soft criteria (warranty, battery life, coatings) that no predicate can
check; their tasks name their candidates and criteria (`_Writer.compare`), else use the fallback openings, whose one
criterion is the decisive detail. Generator 3's intents have their own openings (data/agent/intents.yaml)."""
NOUN_SLOTS = frozenset({"product_type", "noun", "nouns", "noun_bare"})
"""Placeholders that put the product's kind in the opening."""
SPOKEN_LOWER = frozenset({"color", "material", "scent", "flavor", "fit", "shade"})
"""Option fields whose store values a shopper says in lower case ("in navy", not "in Navy")."""


class _OwnWriter(Protocol):
    """A writer of one of generator 3's intents that is not specify-shaped (bundle, budget_max, substitute)."""

    def __call__(self, rng: Random, plan: BuyerPlan | None, variety: Variety, *, impossible: bool) -> _Draft | None: ...


class _HasStoreId(Protocol):
    @property
    def store_id(self) -> str: ...


class _HasId(Protocol):
    @property
    def id(self) -> str: ...


StoreLike = str | StoreRef | _HasStoreId | _HasId
"""A store id, a trace StoreRef, or anything with `store_id` or `id` (carttrace.stores.Store)."""


def generate_tasks(
    store: StoreLike,
    category: CategoryLike | Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    per_intent: int = 2,
    impossible_share: float = IMPOSSIBLE_SHARE,
    phrases: Phrases | None = None,
    persona: str | None = None,
    market: str | None = None,
    personas: Personas | None = None,
    intents: Sequence[IntentType] | None = None,
    intent_words: IntentWords | None = None,
    niche_gate: NicheGate | None = None,
    kinds: Kinds | None = None,
    gift_message: bool = True,
) -> list[TaskRef]:
    """`per_intent` tasks for each intent, written around `fixtures` (product dicts as UCP get_product or
    search_catalog returns them, or `fixtures_from_snapshot` of a sandbox snapshot), with a seeded share deliberately
    impossible.

    `intents` None writes the five first intents in their order, as every plan before generator 3 had them; a list
    (any of carttrace.agents.spec.ALL_INTENTS) writes those, round robin, so the first N tasks cover N intents.

    Each task's buyer block comes from `persona` (the default one of data/agent/personas.yaml when None) shopping in
    `market` (the store's own market code when None, else the market its catalog's currency implies), and the
    persona's style and constraints shape the task (carttrace.agents.intents). A task gets no buyer block when the persona
    has no address there, or the phrases are a v1 file.

    `niche_gate` (D104, carttrace.agents.niche_gate) writes a task that turns on one of the category's niches only where every
    candidate product carries the niche's data as a field, as the open lane's stand-ins are written: a draw that would
    turn on one the products hold only as text is drawn again, its details narrowed to those the products carry.

    `kinds` (D104, carttrace.sandbox.catalogs.kinds) writes the open lane's tasks: only around products whose title names a
    reviewed kind, compare tasks over two or more of one kind, no option value a shopper would not say, and a slot no
    draw can fill so is dropped, its reason recorded for `recording_drops`.

    `gift_message` False (D110) writes gift tasks for a store whose checkout takes no gift message or gift marking, as
    no sandbox platform pack models one: the shopper says who it is for and asks the checkout for nothing more
    (`gift.plain_openings`, the recipient predicate alone). On ep-b0ecea53f8e6 and ep-2b1bb02698b9 the store refused
    `gift_message` and `gift_note`, and the agent was judged on a wish no agent could meet.

    Raises ValidationError when no fixture product has an available, priced variant to write a task around."""
    if per_intent < 1:
        raise ValidationError("per_intent must be at least 1")
    if not 0 <= impossible_share <= 1:
        raise ValidationError("impossible_share is a fraction between 0 and 1")
    with remembering():
        return _generate(
            store,
            category,
            fixtures,
            seed=seed,
            per_intent=per_intent,
            impossible_share=impossible_share,
            phrases=phrases,
            persona=persona,
            market=market,
            personas=personas,
            intents=intents,
            intent_words=intent_words,
            niche_gate=niche_gate,
            kinds=kinds,
            gift_message=gift_message,
        )


def _generate(
    store: StoreLike,
    category: CategoryLike | Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    per_intent: int,
    impossible_share: float,
    phrases: Phrases | None,
    persona: str | None,
    market: str | None,
    personas: Personas | None,
    intents: Sequence[IntentType] | None,
    intent_words: IntentWords | None,
    niche_gate: NicheGate | None = None,
    kinds: Kinds | None = None,
    gift_message: bool = True,
) -> list[TaskRef]:
    """generate_tasks inside one memo run (carttrace.agents.memo): every predicate is checked once per product."""
    chosen = _intents(intents)
    writer = _Writer.build(
        _store_id(store),
        category_spec(category),
        fixtures,
        phrases or load_phrases(),
        intent_words or load_intent_words(),
        gate=niche_gate,
        kinds=kinds,
        gift_message=gift_message,
    )
    slug = writer.category.slug
    shopper = (personas or load_personas()).get(persona) if writer.phrases.buyer is not None else None
    variety = variety_for(shopper)
    where = market or market_of(store, writer.currency)
    priced = where is not None and priced_market(where, writer.currency)
    if intents is None:
        slots = [(intent, index) for intent in chosen for index in range(per_intent)]
    else:
        slots = [(intent, index) for index in range(per_intent) for intent in chosen]
    count = max(1, round(len(slots) * impossible_share)) if impossible_share > 0 else 0
    impossible = set(seeded_rng("tasks", writer.store_id, slug, seed, "impossible").sample(slots, count))

    def plan_of(intent: IntentType, index: int) -> BuyerPlan | None:
        if shopper is None or where is None:
            return None
        draws = seeded_rng("buyer", writer.store_id, slug, seed, intent, index, shopper.name, where)
        return draw_plan(shopper, where, draws, priced=priced, delivery=variety.delivery, intent=intent)

    def draft_of(intent: IntentType, index: int) -> _Draft | None:
        rng = seeded_rng("task", writer.store_id, slug, seed, intent, index)
        plan = plan_of(intent, index)
        draft = writer.impossible(intent, rng, plan, variety) if (intent, index) in impossible else None
        draft = draft or writer.possible(intent, rng, plan, variety)
        if draft is not None and writer.lacking(draft):
            draft = writer.gated_draw(intent, (intent, index) in impossible, plan, variety, seed=seed, index=index)
        return draft

    def task_of(intent: IntentType, index: int) -> TaskRef | None:
        """The slot's task, drawn again when its opening contradicts its spec or (open lane, D104) is unfit to
        publish; None, with the reason recorded (`recording_drops`), when no draw can fill the slot."""
        draft = draft_of(intent, index)
        if draft is None:
            why = "no product of a reviewed kind" if writer.kinds is not None else "no fixture product"
            _dropped(writer.store_id, intent, index, f"{why} can carry a {intent} task")
            return None
        task = writer.task(draft)
        if writer.contradicted(task, draft):
            try:
                task = writer.consistent_draw(
                    intent, (intent, index) in impossible, plan_of(intent, index), variety, seed=seed, index=index
                )
            except ValidationError as exc:
                _dropped(writer.store_id, intent, index, exc.message)
                return None
        return task

    tasks, unwritable = _filled(slots, chosen, per_intent, task_of)
    if not tasks:
        raise ValidationError(
            f"no fixture product can carry a {', '.join(dict.fromkeys(unwritable))} task for {slug}",
            hint="pass fixtures with at least one available, priced variant",
        )
    for task in tasks:
        validate_task(task)
    return tasks


_DROPS: ContextVar[list[str] | None] = ContextVar("task_drops", default=None)


@contextmanager
def recording_drops() -> Iterator[list[str]]:
    """The open lane's dropped task slots (D104) while the block runs, each as "STORE intent#index: why"."""
    found: list[str] = []
    token = _DROPS.set(found)
    try:
        yield found
    finally:
        _DROPS.reset(token)


def _dropped(store_id: str, intent: str, index: int, why: str) -> None:
    found = _DROPS.get()
    if found is not None:
        found.append(f"{store_id} {intent}#{index}: {why}")


def _filled(
    slots: Sequence[tuple[IntentType, int]],
    chosen: Sequence[IntentType],
    per_intent: int,
    task_of: Callable[[IntentType, int], TaskRef | None],
) -> tuple[list[TaskRef], list[IntentType]]:
    """Each slot's task, and the intents none could be written for. A kind no product can carry (a bundle in a store
    of one product) is skipped, and its slots are refilled from the kinds that can be written, in their order, so one
    unwritable kind never refuses a whole store (validation round 3, 2026-10-16: store-527's bundle slot dropped every
    row of the store). The caller notes the missing kinds."""
    tasks: list[TaskRef] = []
    unwritable: list[IntentType] = []
    for intent, index in slots:
        task = None if intent in unwritable else task_of(intent, index)
        if task is None:
            unwritable.append(intent)
            continue
        tasks.append(task)
    writable = [i for i in chosen if i not in unwritable]
    extra = per_intent
    while writable and len(tasks) < len(slots):
        before = len(tasks)
        for intent in writable:
            if len(tasks) >= len(slots):
                break
            task = task_of(intent, extra)
            if task is not None:
                tasks.append(task)
        extra += 1
        if len(tasks) == before or extra > per_intent * 4:
            break
    return tasks, unwritable


def _intents(intents: Sequence[IntentType] | None) -> tuple[IntentType, ...]:
    if intents is None:
        return INTENTS
    unknown = sorted(set(intents) - set(ALL_INTENTS))
    if unknown or not intents:
        raise ValidationError(f"unknown intents {unknown}" if unknown else "no intents", hint=f"use {ALL_INTENTS}")
    return tuple(i for i in ALL_INTENTS if i in set(intents))


def validate_task(task: TaskRef, fixtures: Sequence[Mapping[str, Any]] | None = None) -> None:
    """Refuse a task that could not be judged: its spec has no witness and is not impossible (or the reverse), or,
    when `fixtures` are given, its witnesses are not exactly the available variants that satisfy every predicate."""
    if task.hidden_spec.get("schema") != SPEC_SCHEMA:
        raise ValidationError(
            f"task {task.task_id} has no {SPEC_SCHEMA} hidden spec", hint="write it with generate_tasks"
        )
    spec = HiddenSpec.from_task(task)
    validate_spec(spec, where=f"task {task.task_id}")
    if fixtures is None:
        return
    candidates = set(spec.candidates)
    expected = witnesses_of(spec, [p for p in read_products(fixtures) if p.id in candidates])
    if sorted(expected) != sorted(spec.witnesses):
        raise ValidationError(
            f"task {task.task_id} names witnesses {spec.witnesses[:3]} but the fixtures give {expected[:3]}",
            hint="the fixtures changed since the task was written; regenerate the tasks",
        )


def witnesses_of(spec: HiddenSpec, pool: Sequence[Product]) -> list[str]:
    """The available variants in `pool` that satisfy every predicate of `spec`, in pool order. A budget that covers
    shipping and tax is checked all in, with the buyer's preferred delivery, as generation checked it; a variant the
    spec's own faults take out of stock, or one the store holds fewer units of than a line needs, is never one."""
    quantity = line_quantity(spec.predicates)
    stocked_out = spec.stocked_out
    pairs = [
        (p, v)
        for p in pool
        for v in p.variants
        if v.available
        and v.holds_units(quantity)
        and v.id not in stocked_out
        and not free_gift(v)
        and all(holds_in(spec, pr, p, v, quantity) is True for pr in spec.predicates if pr.op not in SET_OPS)
    ]
    superlative = next((p for p in spec.predicates if p.op in SUPERLATIVES), None)
    return best(superlative, pairs) if superlative is not None else [v.id for _, v in pairs]


def free_gift(variant: Variant) -> bool:
    """A variant priced at nothing: a gift with purchase or a free sample, which a store gives with an order and never
    sells alone. On 2026-10-16 a task's "merch" witnesses were $0 gift items no checkout would take by themselves."""
    return variant.price is not None and variant.price <= 0


def quantity_of(predicates: Sequence[Predicate]) -> int:
    """The units the shopper wants in all: the count predicate's value, else one."""
    found = next((p.value for p in predicates if p.op == "count"), 1)
    return found if isinstance(found, int) and not isinstance(found, bool) and found > 0 else 1


def line_quantity(predicates: Sequence[Predicate]) -> int:
    """The units of one cart line: one of each item for a bundle, else the count the shopper wants."""
    if any(p.op == "all_of" for p in predicates):
        return 1
    return quantity_of(predicates)


def holds_in(spec: HiddenSpec, predicate: Predicate, product: Product, variant: Variant, quantity: int) -> bool | None:
    """Whether a variant satisfies one predicate of `spec`, where the spec as a whole decides: a substitute for the
    named product counts when the spec allows one, a bundle's budget is checked against the item and the other items'
    cheapest prices, and an all-in budget against the sandbox checkout's total."""
    if predicate.op == "prefer":
        return _named(predicate, product) or spec.substitute_ok
    if predicate.op == "max_price":
        bundle = next((p for p in spec.predicates if p.op == "all_of"), None)
        if bundle is not None:
            return bundle_fits(predicate, bundle, product, variant)
        if spec.budget_scope == "total" and spec.buyer is not None:
            fulfillment = spec.buyer.fulfillment
            preference: FulfillmentKind = fulfillment.value if fulfillment is not None else "standard"
            return all_in_fits(predicate, variant, quantity, market=spec.buyer.market, preference=preference)
    return holds(predicate, product, variant, quantity=quantity)


def bundle_fits(budget: Predicate, bundle: Predicate, product: Product, variant: Variant) -> bool | None:
    """Whether a bundle bought with this variant of one item and the cheapest variant of each other item stays within
    the budget; None when the data cannot say."""
    value = budget.value
    amount = value.get("amount") if isinstance(value, Mapping) else None
    if not isinstance(amount, int) or variant.price is None:
        return None
    currency = value.get("currency") if isinstance(value, Mapping) else None
    if isinstance(currency, str) and variant.currency and currency != variant.currency:
        return False
    items = bundle_items(bundle.value)
    own = next((item for item in items if item_holds(item, product, variant)), None)
    if own is None:
        return False
    others = sum(int(str(item.get("min_price") or 0)) for item in items if item is not own)
    return variant.price + others <= amount


def _named(predicate: Predicate, product: Product) -> bool:
    value = predicate.value
    return isinstance(value, str) and (fold(product.title) == fold(value) or product.id == value)


def all_in_fits(
    predicate: Predicate, variant: Variant, quantity: int, *, market: str | None, preference: FulfillmentKind
) -> bool | None:
    """Whether the sandbox's checkout total for `quantity` of the variant, with the preferred delivery and tax, is
    within an all-in budget; None when the market table cannot price it."""
    value = predicate.value
    amount = value.get("amount") if isinstance(value, Mapping) else None
    if not isinstance(amount, int) or variant.price is None or variant.currency is None or market is None:
        return None
    currency = value.get("currency") if isinstance(value, Mapping) else None
    if isinstance(currency, str) and currency != variant.currency:
        return False
    memo = table("all_in_total")
    key = (variant.price, quantity, market, variant.currency, preference)
    if memo is None or key not in memo:
        total = all_in_total(variant.price, quantity, market=market, currency=variant.currency, preference=preference)
        if memo is not None:
            memo[key] = total
    else:
        total = memo[key]
    return None if total is None else total <= amount


def _all_in_ids(
    predicate: Predicate, pool: Sequence[Product], quantity: int, market: str | None, preference: FulfillmentKind
) -> list[str]:
    """The pool's variants whose all-in total fits the budget, in pool order. `all_in_fits` reads only a variant's
    price and currency, so each price is checked once; within a memo run (carttrace.agents.memo) the pool's answer is
    remembered whole."""
    memo = table("all_in")
    value = json.dumps(predicate.value, sort_keys=True)
    slot = (value, quantity, market, preference, tuple(map(id, pool)))
    hit = memo.get(slot) if memo is not None else None
    if hit is None:
        fits: dict[tuple[int | None, str | None], bool] = {}
        ids: list[str] = []
        for product in pool:
            for v in product.variants:
                price = (v.price, v.currency)
                ok = fits.get(price)
                if ok is None:
                    ok = fits[price] = all_in_fits(predicate, v, quantity, market=market, preference=preference) is True
                if ok:
                    ids.append(v.id)
        hit = (tuple(pool), ids)
        if memo is not None:
            memo[slot] = hit
    found: list[str] = hit[1]
    return list(found)


def _store_id(store: StoreLike) -> str:
    if isinstance(store, str):
        return store
    found = getattr(store, "store_id", None) or getattr(store, "id", None)
    if not isinstance(found, str) or not found:
        raise ValidationError("store must be an id, a StoreRef, or have a store_id or id attribute")
    return found


# -- drafting --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Draft:
    intent: IntentType
    pool: tuple[Product, ...]
    fixture: tuple[str, ...]
    decisive: Decisive
    predicates: tuple[Predicate, ...]
    witnesses: tuple[str, ...]
    opening: str
    template: str
    impossible: bool
    plan: BuyerPlan | None = None
    """The task's buyer draws; None writes no buyer block."""
    faults: tuple[dict[str, Any], ...] = ()
    """Sealed fault knobs the task's world must carry (a substitute task's stockout)."""


@dataclass(frozen=True, slots=True)
class _Focus:
    """What one task is about before its words are chosen."""

    intent: IntentType
    product: Product
    pool: tuple[Product, ...]
    noun: Noun
    detail: Detail | None = None
    proposal: Proposal | None = None
    decisive_stated: bool = False
    plan: BuyerPlan | None = None
    variety: Variety = LEGACY
    extra: Mapping[str, str] = field(default_factory=dict)
    """Slot values an intent's opening needs beyond the product's: a gift's recipient and message, a policy
    question."""

    @property
    def deciding(self) -> tuple[Detail, Proposal] | None:
        return (self.detail, self.proposal) if self.detail is not None and self.proposal is not None else None

    @property
    def total_scope(self) -> bool:
        """The shopper's budget, if they have one, covers shipping and tax."""
        return self.plan is not None and self.plan.scope == "total"

    @property
    def persona(self) -> Persona | None:
        return self.plan.persona if self.plan is not None else None


@dataclass(frozen=True, slots=True)
class _Opening:
    """The words of a task: a category template, or a fallback opening and the placeholders it has."""

    template: Template | None
    fallback: str = ""
    fallback_id: str = ""

    @property
    def names(self) -> frozenset[str]:
        if self.template is not None:
            return frozenset(self.template.slots)
        return frozenset(n for _, n, _, _ in string.Formatter().parse(self.fallback) if n)


@dataclass(frozen=True, slots=True)
class _Writer:
    store_id: str
    category: CategorySpec
    products: tuple[Product, ...]
    details: tuple[Detail, ...]
    currency: str
    phrases: Phrases
    intents: IntentWords
    gate: NicheGate | None = None
    """The open lane's niche gate (D104); None writes every task the data allows, text included."""
    kinds: Kinds | None = None
    """The open lane's product kinds (D104): tasks are written only around products whose title names a reviewed
    kind, a compare task needs two of one kind, and a slot no draw can fill well is dropped with its reason."""
    gift_message: bool = True
    """Whether the store's checkout takes a gift message (D110); a gift task asks for one only then."""

    def noun(self, product: Product) -> Noun:
        """The shopper's word for a product: its reviewed kind in the open lane, else `noun_of`."""
        if self.kinds is not None:
            found = kind_noun(product, self.kinds)
            if found is not None:
                return found
        return noun_of(product)

    def name(self, product: Product) -> str:
        """The product as the shopper names it in an opening or a reply (D119, `nouns.spoken_name`): its title when
        short, else its brand, key attribute and kind, with more of the title's words while another product of the
        store's would answer to the same words, and its first naming segment when none singles it out. The hidden
        spec keeps the full title, which is what is matched. Tasks had quoted whole 150 to 250 character Lazada
        titles ("Plus Size S-5XL Women Fishtail Denim Skirt ... Jean Midi Skirt Blue")."""
        noun = self.noun(product)
        if spoken_name(product.title, noun) == " ".join(product.title.split()):
            return product.title
        others = [fold(p.title) for p in self.products if p.id != product.id]
        for lead in range(NAME_LEAD, NAME_LEAD_MAX + 1):
            core = spoken_name(product.title, noun, lead=lead)
            if not any(has_phrase(other, fold(core)) for other in others):
                return spoken_name(product.title, noun, product.brand, lead=lead)
        first = re.split(r"\s+[-|/(]\s*|,\s*", " ".join(product.title.split()), maxsplit=1)[0].split()
        return " ".join(first[: 2 * SHORT_TITLE_WORDS]).strip(",.;:-")

    @classmethod
    def build(
        cls,
        store_id: str,
        category: CategorySpec,
        fixtures: Sequence[Mapping[str, Any]],
        phrases: Phrases,
        intents: IntentWords,
        *,
        gate: NicheGate | None = None,
        kinds: Kinds | None = None,
        gift_message: bool = True,
    ) -> _Writer:
        products = tuple(p for p in read_products(fixtures) if any(v.price is not None for v in p.variants))
        if kinds is not None:
            products = tuple(p for p in products if kind_noun(p, kinds) is not None)
        priced = [v for p in products for v in p.variants if v.available and v.price is not None]
        if not priced:
            raise ValidationError(
                f"no fixture product for {store_id} has an available, priced variant",
                hint="pass the store's products as get_product returns them",
            )
        currencies = sorted({v.currency for v in priced if v.currency})
        if not currencies:
            raise ValidationError("fixture prices name no currency", hint="pass prices as {amount, currency}")
        currency = max(currencies, key=lambda c: sum(1 for v in priced if v.currency == c))
        details = tuple(details_for(category.fields))
        return cls(store_id, category, products, details, currency, phrases, intents, gate, kinds, gift_message)

    # -- the niche gate (D104) ----------------------------------------------------------------------------------------

    def _gated(self, kind: str | None, products: Sequence[Product]) -> bool:
        """Whether the gate keeps a task from deciding by `kind` over `products` (never without a gate)."""
        return self.gate is not None and not self.gate.allows(kind, products)

    def _details_for(self, products: Sequence[Product]) -> list[Detail]:
        """The decisive details a task over `products` may turn on: all of them without a gate."""
        return [d for d in self.details if not self._gated(DECISIVE_GATED.get(d.name), products)]

    def lacking(self, draft: _Draft) -> list[str]:
        """The niches a draft turns on that its candidates do not all carry as a field; empty without a gate."""
        if self.gate is None:
            return []
        return self.gate.lacking(draft.decisive, draft.predicates, draft.pool)

    def contradicted(self, task: TaskRef, draft: _Draft) -> list[str]:
        """What the task's opening contradicts in its spec (carttrace.agents.consistency): delivery, deadline, budget and
        its scope, quantity, or a variant option of its candidates; in the open lane (D104) also what makes the task
        unfit to publish (`flaws`)."""
        found = contradictions(HiddenSpec.from_task(task), task.opening, draft.pool)
        return [*found, *self.flaws(task, draft)] if self.kinds is not None else found

    def flaws(self, task: TaskRef, draft: _Draft) -> list[str]:
        """What makes an open-lane task unfit to publish (D104): a compare task with fewer than two candidates of one
        kind, or an option value no person would say ("I'd like it in 1-2 days travel capacity9")."""
        out: list[str] = []
        if draft.intent == "compare":
            named = next((p.value for p in draft.predicates if p.op == "one_of"), None)
            count = len(named) if isinstance(named, list) else len(draft.pool)
            if count < 2:
                out.append("a compare task over fewer than two products of its kind")
        for predicate in draft.predicates:
            value = predicate.value
            if predicate.op == "eq" and isinstance(value, str):
                said = plain_value(value, speakable(value, predicate.field))
                if said is None:
                    out.append(f"the {predicate.field} {value!r} is not a value a shopper would say")
                elif said != value and _said_twice(predicate, said, draft):
                    out.append(f"the {predicate.field} {value!r} said as {said!r} names another option too")
            if predicate.source == "on_ask" and not _askable(predicate, draft):
                out.append(f"a hidden {predicate.field} no careful agent would ask about for a {draft.intent} task")
        return out

    def consistent_draw(
        self,
        intent: IntentType,
        impossible: bool,
        plan: BuyerPlan | None,
        variety: Variety,
        *,
        seed: int,
        index: int,
    ) -> TaskRef:
        """A draw again for a slot whose task's opening contradicts its spec (D104): up to GATE_TRIES draws of their
        own, each held to the niche gate too. Raises ValidationError when none is consistent, so a contradictory
        task is never written."""
        reasons: list[str] = []
        for attempt in range(GATE_TRIES):
            rng = seeded_rng("task-consistent", self.store_id, self.category.slug, seed, intent, index, attempt)
            draft = self.impossible(intent, rng, plan, variety) if impossible else None
            draft = draft or self.possible(intent, rng, plan, variety)
            if draft is None or self.lacking(draft):
                continue
            task = self.task(draft)
            reasons = self.contradicted(task, draft)
            if not reasons:
                return task
        raise ValidationError(
            f"every {intent} task drawn for {self.store_id} contradicts its own spec: {'; '.join(reasons)}",
            hint="the opening's words come from data/prompts/shopper-v2.yaml and the buyer draw; fix them there",
        )

    def gated_draw(
        self,
        intent: IntentType,
        impossible: bool,
        plan: BuyerPlan | None,
        variety: Variety,
        *,
        seed: int,
        index: int,
    ) -> _Draft | None:
        """A draw again for a slot whose first draft turned on a niche its products lack as a field (the hooks narrow
        the details to the shelf's, but a superlative may widen the shelf after): up to GATE_TRIES draws of their
        own, then one with no decisive detail at all, so the slot keeps its intent. Raises ValidationError if even
        that turns on a niche, which would be a gap in the hooks."""
        for attempt in range(GATE_TRIES):
            rng = seeded_rng("task-gate", self.store_id, self.category.slug, seed, intent, index, attempt)
            draft = self.impossible(intent, rng, plan, variety) if impossible else None
            draft = draft or self.possible(intent, rng, plan, variety)
            if draft is not None and not self.lacking(draft):
                return draft
        plain = replace(self, details=())
        rng = seeded_rng("task-gate", self.store_id, self.category.slug, seed, intent, index, "plain")
        draft = plain.impossible(intent, rng, plan, variety) if impossible else None
        draft = draft or plain.possible(intent, rng, plan, variety)
        if draft is not None and plain.lacking(draft):
            raise ValidationError(
                f"a {intent} task of {self.store_id} turns on {', '.join(plain.lacking(draft))} without the data",
                hint="the niche gate's hooks in carttrace.agents.tasks missed a predicate; gate it where it is drawn",
            )
        return draft

    def shelf(self, intent: IntentType, product: Product) -> tuple[Product, ...]:
        """The products a shopper with this intent would choose among: the named one for replenish, else every
        product of the same kind."""
        if intent == "replenish":
            return (product,)
        noun = self.noun(product)
        memo = table("shelf")
        if memo is None:
            return tuple(p for p in self.products if self.noun(p) == noun)
        slot = (id(self.products), noun)
        hit = memo.get(slot)
        if hit is None:
            hit = memo[slot] = (self.products, tuple(p for p in self.products if self.noun(p) == noun))
        shelf: tuple[Product, ...] = hit[1]
        return shelf

    # -- possible tasks ----------------------------------------------------------------------------------------------

    def possible(
        self, intent: IntentType, rng: Random, plan: BuyerPlan | None = None, variety: Variety = LEGACY
    ) -> _Draft | None:
        """A task with at least one witness. A compare task names its candidates where it can; otherwise targets
        whose data carries a decisive detail come first, and a target decided by its options is the fallback. Bundle,
        budget_max and substitute tasks have writers of their own; gift and policy are specify-shaped tasks with
        their own openings and extra predicates."""
        if intent == "compare":
            named = self.compare(rng, plan, variety, impossible=False)
            if named is not None:
                return named
        writer = self._own_writer(intent)
        if writer is not None:
            return writer(rng, plan, variety, impossible=False)
        if intent == "gift" and plan is not None:
            plan = with_deadline(plan)
        extra = self._extra_slots(intent, rng, plan)
        pairs = [(p, v) for p in self.products for v in p.variants if v.available and v.price is not None]
        rng.shuffle(pairs)
        fallback: _Draft | None = None
        for product, picked in pairs[:TARGET_TRIES]:
            pool = self.shelf(intent, product)
            if intent == "compare" and self.kinds is not None and len(pool) < 2:
                continue  # D104: "I'm deciding between your pregnancy pillows" over a shelf of one
            focus = _Focus(intent, product, pool, self.noun(product), plan=plan, variety=variety, extra=extra)
            focus, variant = self._decide(focus, picked, rng)
            draft = self._compose(focus, variant, rng)
            if not draft.witnesses:
                # a category template's wording can contradict the target (a strength it does not say); the
                # fallback openings state only what the target satisfies, so they always leave a witness
                draft = self._compose(focus, variant, rng, templates=False)
            if draft.witnesses and focus.deciding is not None:
                return draft
            if draft.witnesses and fallback is None:
                fallback = draft
        return fallback

    def _own_writer(self, intent: IntentType) -> _OwnWriter | None:
        """The writer of an intent that is not specify-shaped: bundle, budget_max or substitute; None otherwise."""
        writers: dict[str, _OwnWriter] = {
            "bundle": self.bundle,
            "budget_max": self.budget_max,
            "substitute": self.substitute,
        }
        return writers.get(intent)

    def _decide(self, focus: _Focus, variant: Variant, rng: Random) -> tuple[_Focus, Variant]:
        """The decisive detail of a target, if its data carries one: the persona's own measurement (which may pick
        another variant of the product), the persona's allergen or condition, else what the detail proposes; and
        whether the opening states it."""
        persona = focus.persona
        product = focus.product
        fitted = persona_fit(persona, product)
        if fitted is not None:
            variant = fitted[0]
        details = self._details_for(focus.pool)
        rng.shuffle(details)
        for detail in details:
            proposal = self._propose(
                detail, product, variant, self.shelf("specify", product), rng=rng, persona=persona, fitted=fitted
            )
            if proposal is not None:
                stated = focus.intent == "express" or rng.random() < focus.variety.stated_decisive
                return replace(focus, detail=detail, proposal=proposal, decisive_stated=stated), variant
        return focus, variant

    def _propose(
        self,
        detail: Detail,
        product: Product,
        variant: Variant,
        shelf: Sequence[Product],
        *,
        rng: Random,
        persona: Persona | None,
        fitted: tuple[Variant, Proposal] | None,
    ) -> Proposal | None:
        """What a detail proposes for a target. A persona's own constraints are binding: an allergen persona avoids
        only their allergens, a persona with a condition asks only about it, and a persona whose measurement the
        product's chart maps uses that measurement; where the data cannot make their constraint decide, the detail
        proposes nothing rather than a stranger's fact. Without such a constraint the detail proposes as it always
        did."""
        if detail.name == "size_chart" and persona_measures(persona, product):
            return fitted[1] if fitted is not None else None
        if persona_constrains(detail, persona):
            return persona_proposal(detail, persona, product, variant, shelf)
        return detail.propose(product, variant, shelf, rng)

    def _impossible_proposal(
        self, detail: Detail, pool: Sequence[Product], rng: Random, persona: Persona | None
    ) -> Proposal | None:
        """A value no product of the pool meets: the persona's own constraint where they have one for this detail
        (and nothing else, so a persona never claims a stranger's allergy), else what the detail finds."""
        if persona_constrains(detail, persona) or (
            detail.name == "size_chart" and any(persona_measures(persona, p) for p in pool)
        ):
            return persona_impossible(detail, persona, pool)
        return detail.impossible(pool, rng)

    def _extra_slots(self, intent: IntentType, rng: Random, plan: BuyerPlan | None) -> dict[str, str]:
        """The slot values a gift or policy opening needs, drawn once per task so the words and the predicates agree."""
        if intent == "gift":
            words = self.intents.gift
            return {"recipient": rng.choice(words.recipients), "message": rng.choice(words.messages)}
        if intent == "policy":
            policy = self.intents.policy
            topic = rng.choice(POLICY_TOPICS)
            home = plan.market if plan is not None else CURRENCY_MARKETS.get(self.currency)
            code = rng.choice(_other_countries(policy.countries, home))
            country = policy.countries[code]
            asked = policy.questions[topic]
            return {
                "topic": topic,
                "market": code,
                "country": country,
                "question": fill(asked.question, {"country": country}),
                "clause": fill(asked.clause, {"country": country}),
                "say": fill(asked.say, {"country": country}),
            }
        return {}

    def _extras(self, focus: _Focus) -> list[Predicate]:
        """The predicates a gift or policy task adds to a specify-shaped task: who it is for, that it is a gift and
        its message; or the policy question. Settled at checkout or by the agent's answer, never by a variant."""
        extra = focus.extra
        if focus.intent == "gift" and extra:
            words = self.intents.gift.say
            recipient, message = extra["recipient"], extra["message"]
            for_whom = Predicate(
                field="recipient",
                op="checkout",
                value=recipient,
                source="stated",
                say=fill(words["recipient"], {"recipient": recipient}),
            )
            if not self.gift_message:
                return [for_whom]
            return [
                for_whom,
                Predicate(field="gift", op="checkout", value=True, source="stated", say=words["gift"]),
                Predicate(
                    field="gift_message",
                    op="checkout",
                    value=message,
                    source="stated",
                    say=fill(words["gift_message"], {"message": message}),
                ),
            ]
        if focus.intent == "policy" and extra:
            value: dict[str, Any] = {"topic": extra["topic"]}
            if extra["topic"] == "shipping":
                value.update({"market": extra["market"], "country": extra["country"]})
            return [Predicate(field="policy", op="policy", value=value, source="stated", say=extra["say"])]
        return []

    # -- compare tasks that name their candidates --------------------------------------------------------------------

    def compare(
        self, rng: Random, plan: BuyerPlan | None, variety: Variety = LEGACY, *, impossible: bool
    ) -> _Draft | None:
        """A compare task naming two to four products of one shelf and one or two criteria the fixture data settles
        exactly: some candidate qualifies, or (`impossible`) none does. None when the store's data allows neither."""
        if self.phrases.compare is None:
            return None
        products = [p for p in self.products if _sellable(p)]
        rng.shuffle(products)
        for target in products[:TARGET_TRIES]:
            shelf = self.shelf("compare", target)
            others = [p for p in shelf if p.id != target.id and _sellable(p)]
            if not others:
                continue
            rng.shuffle(others)
            named = sorted([target, *others[: rng.randint(1, min(3, len(others)))]], key=lambda p: p.id)
            draft = self._named(target, tuple(named), shelf, rng=rng, plan=plan, variety=variety, impossible=impossible)
            if draft is not None and bool(draft.witnesses) != impossible:
                return draft
        return None

    def _named(
        self,
        target: Product,
        named: tuple[Product, ...],
        shelf: tuple[Product, ...],
        *,
        rng: Random,
        plan: BuyerPlan | None,
        variety: Variety,
        impossible: bool,
    ) -> _Draft | None:
        words = self.phrases.compare
        assert words is not None
        variant = rng.choice([v for v in target.variants if v.available and v.price is not None])
        persona = plan.persona if plan is not None else None
        found = self._compare_detail(target, variant, named, rng, impossible=impossible, persona=persona)
        if found is None and (self._details_for(named) or impossible):
            return None  # a category with decisive details compares on one, and an impossible task needs one
        best_of = None
        if found is None or (not impossible and rng.random() < SECOND_CRITERION):
            best_of = self._superlative(named, rng)
        detail, proposal = found if found is not None else (None, None)
        focus = _Focus("compare", target, shelf, self.noun(target), detail, proposal, True, plan, variety)
        titles = [p.title for p in named]
        listed = _listed([_the(self.name(p)) for p in named])
        kind_say = self.phrases.fact("kind").say.format(noun=focus.noun.with_article())
        predicates = [
            Predicate(field="product_type", op="kind", value=focus.noun.singular, source="stated", say=kind_say),
            Predicate(
                field="product_name",
                op="one_of",
                value=list(titles),
                source="stated",
                say=fill(words.candidates, {"candidates": listed}),
            ),
        ]
        if detail is not None and proposal is not None:
            predicates.append(self._decisive(focus, detail, proposal))
        if best_of is not None:
            predicates.append(best_of[0])
        if rng.random() < COMPARE_BUDGET:
            budget = self._budget_above(focus, variant, 1)
            predicates.append(self._budget(budget, "on_ask", total=focus.total_scope))
        predicates.append(self._quantity(1, "default"))
        criteria = " ".join(s for s in (proposal.sentence if proposal else "", best_of[1] if best_of else "") if s)
        index = rng.randrange(len(words.named))
        text = fill(words.named[index], {"candidates": listed, "criteria": f" {criteria}" if criteria else ""})
        if detail is not None:
            decisive = self._decisive_block(focus)
        else:
            assert best_of is not None
            field_name = best_of[0].field
            decisive = Decisive(detail="nutrition" if field_name == "nutrition" else "price", field=field_name)
        fixture = tuple(p.id for p in named)
        return self._finish(focus, fixture, decisive, predicates, opening=text, template=f"compare.named.{index}")

    def _compare_detail(
        self,
        target: Product,
        variant: Variant,
        named: Sequence[Product],
        rng: Random,
        *,
        impossible: bool,
        persona: Persona | None = None,
    ) -> tuple[Detail, Proposal] | None:
        """A decisive fact the candidates' data settles: one the target's variant meets and another candidate may
        not, or (`impossible`) one no candidate meets. A persona's own constraints are binding here as in `_propose`."""
        details = self._details_for(named)
        rng.shuffle(details)
        fitted = persona_fit(persona, target)
        for detail in details:
            if impossible:
                proposal = self._impossible_proposal(detail, named, rng, persona)
            else:
                proposal = self._propose(detail, target, variant, named, rng=rng, persona=persona, fitted=fitted)
            if proposal is not None:
                return detail, proposal
        return None

    def _superlative(self, named: Sequence[Product], rng: Random) -> tuple[Predicate, str]:
        """The best of the candidates by a measure every one of them states: price, price per unit, or a nutrient
        per 100 g (one whose candidates differ). The predicate and the sentence the opening adds."""
        words = self.phrases.compare
        assert words is not None
        pairs = [(p, v) for p in named for v in p.variants if v.available and v.price is not None]
        offered: list[tuple[Op, str, dict[str, str]]] = [("cheapest", "price", {})]
        unit = common_unit(pairs)
        if unit is not None:
            offered.append(("cheapest_per_unit", "unit_price", {"unit": basis_words(unit)}))
        directions: tuple[tuple[Op, tuple[str, ...]], ...] = (("most", MOST), ("least", LEAST))
        for op, names in directions if not self._gated("nutrition", named) else ():
            for name in names:
                stated = [nutrient(p, name) for p in named]
                if all(s is not None for s in stated) and len(set(stated)) > 1:
                    offered.append((op, "nutrition", {"nutrient": name}))
        op, field_name, slots = rng.choice(offered)
        phrase = words.criteria[op]
        value = dict(slots) or None
        say = phrase.say.format(**slots)
        return Predicate(field=field_name, op=op, value=value, source="stated", say=say), phrase.clause.format(**slots)

    # -- generator 3's own writers -----------------------------------------------------------------------------------

    def bundle(self, rng: Random, plan: BuyerPlan | None, variety: Variety, *, impossible: bool) -> _Draft | None:
        """Two or three products that go together, one of each: products of different kinds where the catalog has
        them, else different products of one kind. An impossible bundle has a budget below the cheapest way to buy
        it. None when the catalog has one sellable product."""
        sellable = [p for p in self.products if _sellable(p)]
        if len(sellable) < 2:
            return None
        plan = _items_scope(plan)
        by_noun: dict[str, list[Product]] = {}
        for product in sellable:
            by_noun.setdefault(self.noun(product).singular, []).append(product)
        wanted = rng.choice((2, 3)) if len(sellable) >= 3 else 2
        if len(by_noun) >= 2:
            nouns = rng.sample(sorted(by_noun), min(wanted, len(by_noun)))
            items = [rng.choice(by_noun[n]) for n in nouns]
            if len(items) < wanted:
                spare = [p for p in sellable if p not in items]
                items += rng.sample(spare, wanted - len(items))
        else:
            items = rng.sample(sellable, wanted)
        items = sorted(items, key=lambda p: p.id)
        floors = {p.id: min(v.price for v in p.variants if v.available and v.price is not None) for p in items}
        value: list[dict[str, Any]] = [
            {
                "title": p.title,
                "product_id": p.id,
                "variants": [v.id for v in p.variants if v.available and v.price is not None],
                "min_price": floors[p.id],
            }
            for p in items
        ]
        words = self.intents.bundle
        titles = _listed([_the(self.name(p)) for p in items])
        count = len(items)
        bundle_say = fill(words.say["bundle"], {"titles": titles})
        lines_say = words.say["lines"].format(count=count)
        quantity_say = words.say["quantity"].format(count=count)
        predicates = [
            Predicate(field="bundle", op="all_of", value=value, source="stated", say=bundle_say),
            Predicate(field="lines", op="lines", value=count, source="stated", say=lines_say),
            Predicate(field="quantity", op="count", value=count, source="stated", say=quantity_say),
        ]
        total = sum(floors.values())
        budget_clause = ""
        if impossible:
            amount = budget_below(total, self.currency)
            if amount is None:
                return None
            predicates.append(self._budget(amount, "stated"))
            budget_clause = f" {self._budget_phrase(False).clause.format(amount=money(amount, self.currency))}"
        elif rng.random() < BUNDLE_BUDGET:
            amount = budget_above(total, self.currency, percent=variety.budget_percent)
            source: Source = "stated" if rng.random() < 0.6 else "on_ask"
            predicates.append(self._budget(amount, source))
            if source == "stated":
                budget_clause = f" {self._budget_phrase(False).clause.format(amount=money(amount, self.currency))}"
        focus = _Focus("bundle", items[0], tuple(items), self.noun(items[0]), plan=plan, variety=variety)
        openings = self.intents.openings("bundle", self.category.slug)
        index = rng.randrange(len(openings))
        text = fill(
            openings[index], {"titles": titles, "count_words": words.count_words[str(count)], "budget": budget_clause}
        )
        decisive = (
            Decisive(detail="price", field="budget") if impossible else Decisive(detail="options", field="bundle")
        )
        fixture = tuple(p.id for p in items)
        return self._finish(focus, fixture, decisive, predicates, opening=text, template=f"intents.bundle.{index}")

    def budget_max(self, rng: Random, plan: BuyerPlan | None, variety: Variety, *, impossible: bool) -> _Draft | None:
        """A named product and a budget: as many as the budget buys (the count the cheapest variant gives), or, for an
        impossible task, a budget below one unit."""
        products = [p for p in self.products if _sellable(p)]
        rng.shuffle(products)
        words = self.intents.budget_max
        plan = _items_scope(plan)
        for target in products[:TARGET_TRIES]:
            prices = [v.price for v in target.variants if v.available and v.price is not None and v.price > 0]
            if not prices:
                continue  # a free or unpriced product buys no count against a budget
            floor = min(prices)
            if impossible:
                below = budget_below(floor, self.currency)
                if below is None:
                    continue
                amount, count = below, 1
            else:
                amount = budget_above(floor * rng.randint(2, 5), self.currency, percent=variety.budget_percent)
                count = amount // floor
            noun = self.noun(target)
            kind_say = self.phrases.fact("kind").say.format(noun=noun.with_article())
            predicates = [
                Predicate(
                    field="product_name",
                    op="is",
                    value=target.title,
                    source="stated",
                    say=f"I want {_the(self.name(target))}",
                ),
                Predicate(field="product_type", op="kind", value=noun.singular, source="on_ask", say=kind_say),
                self._budget(amount, "stated"),
                Predicate(field="quantity", op="count", value=count, source="stated", say=words.say["quantity"]),
            ]
            focus = _Focus("budget_max", target, (target,), self.noun(target), plan=plan, variety=variety)
            openings = self.intents.openings("budget_max", self.category.slug)
            index = rng.randrange(len(openings))
            text = fill(openings[index], {"budget": money(amount, self.currency), "name": self.name(target)})
            decisive = Decisive(detail="price", field="budget")
            draft = self._finish(
                focus, (target.id,), decisive, predicates, opening=text, template=f"intents.budget_max.{index}"
            )
            if bool(draft.witnesses) != impossible:
                return draft
        return None

    def substitute(self, rng: Random, plan: BuyerPlan | None, variety: Variety, *, impossible: bool) -> _Draft | None:
        """A named product the spec's own stockout fault takes out of stock. The shopper accepts a substitute (another
        product of the shelf meeting every other predicate) or wants nothing else, in which case a refusal is right
        and the task is impossible; an `impossible` draft always wants nothing else."""
        products = [p for p in self.products if _sellable(p)]
        rng.shuffle(products)
        words = self.intents.substitute
        for target in products[:TARGET_TRIES]:
            shelf = self.shelf("specify", target)
            others = [p for p in shelf if p.id != target.id and _sellable(p)]
            ok = False if impossible else rng.random() < SUBSTITUTE_OK
            if ok and not others:
                continue  # a shopper who accepts a substitute needs one to exist
            stated = rng.random() < SUBSTITUTE_STATED
            noun = self.noun(target)
            kind_say = self.phrases.fact("kind").say.format(noun=noun.with_article())
            ok_say = words.say["ok" if ok else "not_ok"]
            prefer_say = fill(words.say["prefer" if ok else "prefer_strict"], {"name": self.name(target)})
            ok_source: Source = "stated" if stated else "on_ask"
            predicates = [
                Predicate(field="product_type", op="kind", value=noun.singular, source="on_ask", say=kind_say),
                Predicate(field="product_name", op="prefer", value=target.title, source="stated", say=prefer_say),
                Predicate(field="substitute_ok", op="checkout", value=ok, source=ok_source, say=ok_say),
                self._quantity(1, "default"),
            ]
            faults = tuple(
                {"kind": "stockout", "variant": v.id, "id": f"stockout-{n}"}
                for n, v in enumerate((v for v in target.variants if v.available and v.price is not None), start=1)
            )
            focus = _Focus("substitute", target, tuple(shelf), self.noun(target), plan=plan, variety=variety)
            openings = self.intents.openings("substitute", self.category.slug)
            index = rng.randrange(len(openings))
            sentence = f"{capital(ok_say)}." if stated else ""
            text = fill(openings[index], {"name": self.name(target), "options": "", "substitute_sentence": sentence})
            decisive = Decisive(detail="stock", field="substitute_ok")
            draft = self._finish(
                focus,
                (target.id,),
                decisive,
                predicates,
                opening=text,
                template=f"intents.substitute.{index}",
                faults=faults,
            )
            if bool(draft.witnesses) == ok:
                return draft
        return None

    # -- composing a specify-shaped task -----------------------------------------------------------------------------

    def _compose(self, focus: _Focus, variant: Variant, rng: Random, *, templates: bool = True) -> _Draft:
        forbidden = focus.detail.forbidden_slots if focus.detail else frozenset()
        fields = self.category.fields
        options = {f: v for f, v in option_slots(variant, fields).items() if f not in forbidden}
        facts = self._facts(focus, variant, forbidden)
        opening = self._opening(focus, options, facts, rng, templates=templates)
        names = opening.names
        quantity = rng.choice((2, 3)) if "quantity" in names and opening.template is not None else 1
        two = False
        if quantity == 1 and focus.variety.two_share > 0 and rng.random() < focus.variety.two_share:
            quantity, two = 2, True
        if variant.stock is not None and 0 < variant.stock < quantity:
            quantity = variant.stock  # never ask for more units than the store holds (the draws stay as they were)
            two = two and quantity > 1
        budget = self._budget_above(focus, variant, quantity)
        predicates = self._identity(focus, names)
        deciding = focus.deciding
        if deciding is not None:
            predicates.append(self._decisive(focus, *deciding))
        if opening.template is not None:
            predicates += self._template_slots(opening.template, focus, options, facts)
        used = {p.field for p in predicates}
        sourced = self._option_predicates(used, options, focus.pool, names, rng=rng, variety=focus.variety)
        predicates += [p for p, _ in sourced]
        need = (
            None
            if self._gated("dietary", focus.pool)
            else persona_dietary(focus.persona, focus.product, variant, fields)
        )
        dietary_stated = False
        if need is not None and "dietary" not in {p.field for p in predicates}:
            dietary_stated = rng.random() < focus.variety.stated_decisive
            say = f"it needs to be {need}"
            predicates.append(
                Predicate(
                    field="dietary", op="mentions", value=need, source="stated" if dietary_stated else "on_ask", say=say
                )
            )
        budget_source = self._budget_source(opening, rng)
        if budget_source is not None:
            predicates.append(self._budget(budget, budget_source, total=focus.total_scope))
        stated_quantity = ("quantity" in names and opening.template is not None) or two
        predicates.append(self._quantity(quantity, "stated" if stated_quantity else "default"))
        if opening.template is not None and _CHEAPEST.search(opening.template.text):
            predicates.append(Predicate(field="price", op="cheapest", source="stated", say="it should be the cheapest"))
        predicates += self._extras(focus)
        if opening.template is not None:
            text = self._from_template(opening.template, focus, options, facts, quantity=quantity, budget=budget)
            template_id = opening.template.id
        else:
            stated = [p for p, said in sourced if said]
            text = self._from_fallback(opening.fallback, focus, stated, budget, budget_source == "stated")
            template_id = opening.fallback_id
        if two:
            text = f"{text} {capital(self.phrases.fact('quantity').say.format(count=quantity))}."
        if dietary_stated and need is not None:
            text = f"{text} It needs to be {need}."
        text = self._with_core_feature(focus, predicates, text)
        decisive = self._decisive_block(focus)
        return self._finish(focus, (focus.product.id,), decisive, predicates, opening=text, template=template_id)

    def _with_core_feature(self, focus: _Focus, predicates: list[Predicate], text: str) -> str:
        """The opening with a union core fact the task also turns on (D117), its predicate added; else as it was."""
        feature = self._core_feature(focus, predicates)
        if feature is None:
            return text
        predicates.append(feature)
        return f"{text} {capital(feature.say)}."

    def _core_feature(self, focus: _Focus, predicates: Sequence[Predicate]) -> Predicate | None:
        """A union core fact of the target the shopper decides by (D117), stated: the open lane's only (a gate is
        given), a seeded share of tasks, and only where the shelf holds products without it, so it decides."""
        if self.gate is None or len(focus.pool) < 2:
            return None
        rng = seeded_rng("core-feature", self.store_id, self.category.slug, focus.intent, focus.product.id)
        if rng.random() >= CORE_FEATURE_SHARE:
            return None
        used = {p.field for p in predicates}
        for name, phrase, say in core_features(focus.product):
            if name in used:
                continue
            holders = [p for p in focus.pool if _mentions_product(p, phrase)]
            if focus.product in holders and len(holders) < len(focus.pool):
                return Predicate(field=name, op="mentions", value=phrase, source="stated", say=say)
        return None

    def _facts(self, focus: _Focus, variant: Variant, forbidden: frozenset[str]) -> dict[str, SlotValue]:
        """The text slots a template may state, less those the detail forbids, and less an active ingredient read
        from the description where the niche gate does not count it (D104)."""
        facts = {
            f: v for f, v in text_slots(focus.product, variant, self.category.fields).items() if f not in forbidden
        }
        if "active_ingredient" in facts and self._gated("ingredients", focus.pool):
            del facts["active_ingredient"]
        return facts

    def _opening(
        self,
        focus: _Focus,
        options: Mapping[str, SlotValue],
        facts: Mapping[str, SlotValue],
        rng: Random,
        *,
        templates: bool = True,
    ) -> _Opening:
        allowed = templates and focus.intent not in NO_CATEGORY_TEMPLATES
        offered = self.category.templates.get(focus.intent, ()) if allowed else ()
        usable = [t for t in offered if self._usable(t, focus, options, facts)]
        if usable:
            return _Opening(rng.choice(usable))
        fallbacks, prefix = self._fallbacks(focus.intent)
        index = rng.randrange(len(fallbacks))
        return _Opening(None, fallbacks[index], f"{prefix}.{focus.intent}.{index}")

    def _fallbacks(self, intent: IntentType) -> tuple[list[str], str]:
        """The openings an intent falls back on: the shopper phrases' for the five first intents, and for generator
        3's the words of data/agent/intents.yaml (the category's own, else the generic ones)."""
        if intent == "gift" and not self.gift_message:
            return list(self.intents.gift.plain_openings), "intents.plain"
        if intent in NEW_INTENTS:
            return self.intents.openings(intent, self.category.slug), "intents"
        return self.phrases.openings[intent], "fallback"

    def _usable(
        self, template: Template, focus: _Focus, options: Mapping[str, SlotValue], facts: Mapping[str, SlotValue]
    ) -> bool:
        if UNCHECKED.search(template.text):
            return False
        slots = set(template.slots)
        if focus.detail is not None:
            if slots & focus.detail.forbidden_slots:
                return False
            if focus.detail.field in slots and not focus.decisive_stated:
                return False  # it would state a fact the shopper keeps until asked; a stated one is appended

        if focus.intent == "replenish" and not slots & {"product_name", "product_type"}:
            return False
        for slot in slots:
            if slot in ("product_type", "product_name", "budget", "quantity"):
                continue
            if slot == "brand" and focus.product.brand:
                continue
            if focus.detail is not None and slot == focus.detail.field and focus.proposal is not None:
                continue
            if slot not in options and slot not in facts:
                return False
        return True

    def _identity(self, focus: _Focus, names: frozenset[str]) -> list[Predicate]:
        """What the shopper asks for: the named product (replenish, or a template that names it) and its kind. An
        opening that does not name them ("the same one again", "I'm setting up my first kitchen") keeps them for
        when the agent asks."""
        out = []
        named = bool(names & {"product_name", "name"})
        if focus.intent == "replenish" or named:
            title = focus.product.title
            say = self.phrases.fact("is").say.format(value=self.name(focus.product))
            source: Source = "stated" if named else "on_ask"
            out.append(Predicate(field="product_name", op="is", value=title, source=source, say=say))
        if focus.intent != "replenish" or "product_type" in names:
            say = self.phrases.fact("kind").say.format(noun=focus.noun.with_article())
            kind_source: Source = "stated" if names & NOUN_SLOTS else "on_ask"
            out.append(
                Predicate(field="product_type", op="kind", value=focus.noun.singular, source=kind_source, say=say)
            )
        return out

    def _decisive(self, focus: _Focus, detail: Detail, proposal: Proposal) -> Predicate:
        return Predicate(
            field=detail.field,
            op=detail.op,
            value=proposal.value,
            source="stated" if focus.decisive_stated else "on_ask",
            say=proposal.say,
            aliases=list(proposal.aliases),
        )

    def _template_slots(
        self,
        template: Template,
        focus: _Focus,
        options: Mapping[str, SlotValue],
        facts: Mapping[str, SlotValue],
    ) -> list[Predicate]:
        out: list[Predicate] = []
        for slot in template.slots:
            if focus.detail is not None and slot == focus.detail.field:
                continue
            if slot == "brand" and focus.product.brand:
                say = self.phrases.fact("brand").say.format(value=focus.product.brand)
                out.append(Predicate(field="brand", op="brand", value=focus.product.brand, source="stated", say=say))
                continue
            if slot in _OWN_SLOTS:
                continue
            value = options.get(slot) or facts.get(slot)
            if value is not None:
                out.append(self._slot_predicate(slot, value, "stated"))
        return out

    def _slot_predicate(self, slot: str, value: SlotValue, source: Source) -> Predicate:
        spoken = _said_option(slot, value.text) if value.op == "eq" else _spoken(slot, value.text)
        if value.op == "eq":
            say = self.phrases.option(slot).say.format(value=spoken, label=self.phrases.label(slot))
        elif slot in self.phrases.facts:
            say = self.phrases.fact(slot).say.format(value=spoken)
        else:
            say = f"I'd like the {self.phrases.label(slot)} {spoken}"
        return Predicate(field=slot, op=value.op, value=value.value, option=value.option, source=source, say=say)

    def _option_predicates(
        self,
        used: set[str],
        options: Mapping[str, SlotValue],
        pool: Sequence[Product],
        names: frozenset[str],
        *,
        rng: Random,
        variety: Variety,
    ) -> list[tuple[Predicate, bool]]:
        """The target's other options: stated (only where the opening has an {options} slot), kept for asking, or
        no preference when the shelf offers a choice. The flag marks the stated ones. The persona's disclosure style
        moves the thresholds (carttrace.agents.intents.Variety)."""
        out = []
        for field_name, value in options.items():
            if field_name in used:
                continue
            draw = rng.random()
            if "options" in names and draw < variety.option_stated:
                out.append((self._slot_predicate(field_name, value, "stated"), True))
            elif draw < variety.option_asked:
                out.append((self._slot_predicate(field_name, value, "on_ask"), False))
            elif _choices(value, pool) > 1:
                label = self.phrases.label(field_name)
                predicate = Predicate(
                    field=field_name, op="eq", value=None, option=value.option, source="default", say=f"any {label}"
                )
                out.append((predicate, False))
        return out

    def _budget_source(self, opening: _Opening, rng: Random) -> Source | None:
        names = opening.names
        if opening.template is not None:
            if "budget" in names:
                return "stated"
            return "on_ask" if rng.random() < 0.3 else None
        draw = rng.random()
        if {"budget", "budget_sentence"} & names and draw < 0.5:
            return "stated"
        return "on_ask" if draw < 0.7 else None

    def _budget(self, amount: int, source: Source, *, total: bool = False) -> Predicate:
        """The budget predicate; an all-in one ("all in, with shipping") when it covers shipping and tax."""
        say = self._budget_phrase(total).say.format(amount=money(amount, self.currency))
        value = {"amount": amount, "currency": self.currency}
        return Predicate(field="budget", op="max_price", value=value, source=source, say=say)

    def _budget_phrase(self, total: bool) -> Phrase:
        return self.phrases.fact("budget_total" if total else "budget")

    def _budget_above(self, focus: _Focus, variant: Variant, quantity: int) -> int:
        """A round budget a little above what the target costs (the persona's budget style says how little): its
        items, or, for a budget that covers shipping and tax, the sandbox checkout's all-in total with the preferred
        delivery."""
        currency = variant.currency or self.currency
        price = (variant.price or 0) * quantity
        plan = focus.plan
        if focus.total_scope and plan is not None and variant.price is not None:
            all_in = all_in_total(
                variant.price, quantity, market=plan.market, currency=currency, preference=plan.preference
            )
            price = all_in if all_in is not None else price
        return budget_above(price, currency, percent=focus.variety.budget_percent)

    def _quantity(self, count: int, source: Source) -> Predicate:
        if count == 1 and source == "default":
            say = self.phrases.fact("quantity_one").say
        else:
            say = self.phrases.fact("quantity").say.format(count=count)
        return Predicate(field="quantity", op="count", value=count, source=source, say=say)

    def _decisive_block(self, focus: _Focus) -> Decisive:
        if focus.detail is None:
            return Decisive(detail="options")
        return Decisive(
            detail=focus.detail.name,
            field=focus.detail.field,
            missing=focus.detail.missing,
            hypothesis=self.category.hypothesis_for(focus.detail.hypotheses),
        )

    # -- words -------------------------------------------------------------------------------------------------------

    def _from_template(
        self,
        template: Template,
        focus: _Focus,
        options: Mapping[str, SlotValue],
        facts: Mapping[str, SlotValue],
        *,
        quantity: int,
        budget: int,
    ) -> str:
        amount = money(budget, self.currency)
        values: dict[str, str] = {
            "product_type": focus.noun.singular,
            "product_type_plural": focus.noun.plural,
            "product_name": self.name(focus.product),
            "brand": focus.product.brand or "",
            "budget": self.phrases.fact("budget_total_slot").say.format(amount=amount) if focus.total_scope else amount,
            "quantity": str(quantity),
        }
        values.update({k: _spoken(k, v.text) for k, v in facts.items()})
        values.update({k: _said_option(k, v.text) for k, v in options.items()})
        deciding = focus.deciding
        if deciding is not None:
            values[deciding[0].field] = deciding[1].slot
        text = fill(template.text.replace("{product_type}s", "{product_type_plural}"), values)
        if deciding is not None and focus.decisive_stated and deciding[0].field not in template.slots:
            text = f"{text} {deciding[1].sentence}"
        return text

    def _from_fallback(
        self, text: str, focus: _Focus, stated: Sequence[Predicate], budget: int, budget_stated: bool
    ) -> str:
        clauses = [
            self.phrases.option(p.field).clause.format(
                value=_said_option(p.field, str(p.value)), label=self.phrases.label(p.field)
            )
            for p in stated
        ]
        amount = money(budget, self.currency)
        decisive = f" {focus.proposal.sentence}" if focus.decisive_stated and focus.proposal else ""
        budget_phrase = self._budget_phrase(focus.total_scope)
        values = {
            "noun": focus.noun.with_article(),
            "noun_bare": focus.noun.singular,
            "nouns": focus.noun.plural,
            "name": self.name(focus.product),
            "brand": focus.product.brand or "",
            "from_brand": "",
            "options": (" " + ", ".join(clauses)) if clauses else "",
            "budget": f" {budget_phrase.clause.format(amount=amount)}" if budget_stated else "",
            "budget_sentence": f" {capital(budget_phrase.say.format(amount=amount))}." if budget_stated else "",
            "decisive": decisive,
            **focus.extra,
        }
        opening = fill(text, values)
        if decisive and "{decisive}" not in text:
            opening = f"{opening}{decisive}"
        return opening

    # -- impossible tasks --------------------------------------------------------------------------------------------

    def impossible(
        self, intent: IntentType, rng: Random, plan: BuyerPlan | None = None, variety: Variety = LEGACY
    ) -> _Draft | None:
        """A task no variant on its shelf satisfies: a decisive fact beyond the shelf's data (the persona's own
        constraint first, when the whole shelf fails it), else a budget below the cheapest product; for compare, named
        candidates none of which qualifies; for generator 3's intents, their own impossible form. None when none of
        these can be made from the fixtures."""
        if intent == "compare":
            named = self.compare(rng, plan, variety, impossible=True)
            if named is not None:
                return named
        writer = self._own_writer(intent)
        if writer is not None:
            return writer(rng, plan, variety, impossible=True)
        if intent == "gift" and plan is not None:
            plan = with_deadline(plan)
        extra = self._extra_slots(intent, rng, plan)
        persona = plan.persona if plan is not None else None
        products = list(self.products)
        rng.shuffle(products)
        for product in products[:TARGET_TRIES]:
            pool = self.shelf(intent, product)
            if intent == "compare" and self.kinds is not None and len(pool) < 2:
                continue  # D104: a compare task needs two products of one kind to choose between
            details = self._details_for(pool)
            rng.shuffle(details)
            for detail in details:
                proposal = self._impossible_proposal(detail, pool, rng, persona)
                if proposal is not None:
                    focus = _Focus(
                        intent, product, pool, self.noun(product), detail, proposal, True, plan, variety, extra
                    )
                    draft = self._impossible_draft(focus, None, rng)
                    if not draft.witnesses:
                        return draft
            prices = [v.price for p in pool for v in p.variants if v.available and v.price is not None]
            amount = budget_below(min(prices), self.currency) if prices else None
            if amount is not None:
                focus = _Focus(intent, product, pool, self.noun(product), plan=plan, variety=variety, extra=extra)
                draft = self._impossible_draft(focus, amount, rng)
                if not draft.witnesses:
                    return draft
        return None

    def _impossible_draft(self, focus: _Focus, budget: int | None, rng: Random) -> _Draft:
        templates, prefix = self._fallbacks(focus.intent)
        wanted = ("{budget}", "{budget_sentence}") if budget is not None else ("{decisive}",)
        fitting = [i for i, t in enumerate(templates) if any(w in t for w in wanted)]
        index = rng.choice(fitting) if fitting else rng.randrange(len(templates))
        opening = _Opening(None, templates[index], f"{prefix}.{focus.intent}.{index}")
        predicates = self._identity(focus, opening.names)
        deciding = focus.deciding
        if deciding is not None:
            predicates.append(self._decisive(focus, *deciding))
            decisive = self._decisive_block(focus)
        else:
            decisive = Decisive(detail="price", field="budget")
        if budget is not None:
            predicates.append(self._budget(budget, "stated", total=focus.total_scope))
        predicates.append(self._quantity(1, "default"))
        predicates += self._extras(focus)
        text = self._from_fallback(opening.fallback, focus, [], budget or 0, budget is not None)
        fixture = tuple(p.id for p in focus.pool)
        return self._finish(focus, fixture, decisive, predicates, opening=text, template=opening.fallback_id)

    # -- finishing ---------------------------------------------------------------------------------------------------

    def _finish(
        self,
        focus: _Focus,
        fixture: tuple[str, ...],
        decisive: Decisive,
        predicates: Sequence[Predicate],
        *,
        opening: str,
        template: str,
        faults: Sequence[Mapping[str, Any]] = (),
    ) -> _Draft:
        stocked_out = {str(f["variant"]) for f in faults if f.get("kind") == "stockout"}
        checked, found = self._witnessed(focus, predicates, stocked_out)
        if found and any(p.op in SUPERLATIVES for p in predicates):
            # an impossible draft stays impossible: only a possible one's best is looked for store-wide
            held = {p.id for p in focus.pool}
            kind = next((p for p in predicates if p.op == "kind" and isinstance(p.value, str)), None)
            noun = noun_from(str(kind.value)) if kind is not None else focus.noun
            extra = tuple(
                p for p in self.products if p.id not in held and noun is not None and names_product(noun, p.title)
            )
            if extra:
                wide = replace(focus, pool=(*focus.pool, *extra))
                _, wide_found = self._witnessed(wide, predicates, stocked_out)
                winners = {p.id for p in extra if any(v.id in wide_found for v in p.variants)}
                if winners:
                    focus = replace(focus, pool=(*focus.pool, *(p for p in extra if p.id in winners)))
                    checked, found = self._witnessed(focus, predicates, stocked_out)
        if focus.variety.forgets:
            checked = _forgetting_one(checked)
        return _Draft(
            intent=focus.intent,
            pool=focus.pool,
            fixture=fixture,
            decisive=decisive,
            predicates=tuple(checked),
            witnesses=tuple(found),
            opening=opening,
            template=template,
            impossible=not found,
            plan=focus.plan,
            faults=tuple(dict(f) for f in faults),
        )

    def _witnessed(
        self, focus: _Focus, predicates: Sequence[Predicate], stocked_out: set[str]
    ) -> tuple[list[Predicate], list[str]]:
        """Each predicate's satisfied_by over the focus's pool and the witnesses: the variants in every set, a
        superlative's best of them. Why the caller widens the pool for a superlative (2026-10-16): "the cheapest X you
        have" is the store's cheapest X, and the shopper and the judge read the witnesses, so a template task holds
        the store-wide best as a custom task does (carttrace.agents.intents_custom.rechecked), not the best of one noun's
        shelf."""
        quantity = line_quantity(predicates)
        checked = [
            p.model_copy(update={"satisfied_by": self._satisfied(p, focus, quantity, predicates)}) for p in predicates
        ]
        sets = sorted((set(p.satisfied_by) for p in checked if p.satisfied_by is not None), key=len)
        every = set.intersection(*sets) if sets else None  # the ids in every set: all(v.id in s for s in sets)
        pairs = [
            (p, v)
            for p in focus.pool
            for v in p.variants
            if v.available
            and v.holds_units(quantity)
            and v.id not in stocked_out
            and not free_gift(v)
            and (every is None or v.id in every)
        ]
        found = [v.id for _, v in pairs]
        superlative = next((p for p in checked if p.op in SUPERLATIVES), None)
        if superlative is not None:
            found = best(superlative, pairs)
            checked = [p.model_copy(update={"satisfied_by": found}) if p is superlative else p for p in checked]
        return checked, found

    def _satisfied(
        self, predicate: Predicate, focus: _Focus, quantity: int, predicates: Sequence[Predicate]
    ) -> list[str] | None:
        """The pool's variants that satisfy one predicate: a substitute's named product and, when allowed, the other
        shelf products that meet everything else; a bundle's budget by item; an all-in budget by the sandbox's
        checkout."""
        plan = focus.plan
        if predicate.op == "prefer":
            ok = any(p.field == "substitute_ok" and p.value is True for p in predicates)
            others = [p for p in predicates if p is not predicate and p.op not in SET_OPS]
            found: list[str] = []
            for p in focus.pool:
                if _named(predicate, p):
                    found += [v.id for v in p.variants]
                elif ok:
                    held = [{id(v) for v in passing(o, p, quantity=quantity)} for o in others]
                    found += [v.id for v in p.variants if all(id(v) in h for h in held)]
            return found
        bundle = next((p for p in predicates if p.op == "all_of"), None)
        if predicate.op == "max_price" and bundle is not None:
            return [v.id for p in focus.pool for v in p.variants if bundle_fits(predicate, bundle, p, v) is True]
        if predicate.op != "max_price" or not focus.total_scope or plan is None:
            return satisfied_by(predicate, focus.pool, quantity=quantity)
        return _all_in_ids(predicate, focus.pool, quantity, plan.market, plan.preference)

    def task(self, draft: _Draft) -> TaskRef:
        buyer = None
        opening = draft.opening
        if draft.plan is not None:
            budget = next((p for p in draft.predicates if p.op == "max_price"), None)
            scope = draft.plan.scope if budget is not None else "items"
            buyer = buyer_block(
                draft.plan,
                self.phrases,
                scope=scope,
                budget_source=None if budget is None else budget.source,
                key=f"{draft.template}|{opening}",
            )
            said = stated_sentences(buyer)
            opening = f"{opening} {said}" if said else opening
        spec = HiddenSpec(
            intent=draft.intent,
            decisive=draft.decisive,
            predicates=[p.model_copy(update={"chosen_by": "rules"}) for p in draft.predicates],
            witnesses=list(draft.witnesses),
            candidates=[p.id for p in draft.pool],
            impossible=draft.impossible,
            currency=self.currency,
            generator=GENERATOR,
            buyer=buyer,
            faults=list(draft.faults),
        )
        attributes = describe(
            spec,
            draft.pool,
            category=self.category.slug,
            products=self.products,
            checks=_niche_checks(self.category.slug),
        )
        hidden = spec.model_copy(update={"decision_attributes": attributes}).to_json()
        opening = tidy(opening)
        if draft.template.startswith(("fallback.", "compare.")):
            version = self.phrases.version
        elif draft.template.startswith("intents."):
            version = self.intents.version
        else:
            version = GENERATOR
        return TaskRef(
            task_id=stable_id("task", self.store_id, self.category.slug, draft.intent, opening, hidden),
            intent=draft.intent,
            category=self.category.slug,
            opening=opening,
            fixture=list(draft.fixture),
            hidden_spec=hidden,
            template=f"{draft.template}@{version}",
        )


def _niche_checks(slug: str) -> tuple[Check, ...]:
    """The niche checks of the category (data/agent/checks.yaml), for the provenance of a template task."""
    return tuple(load_checklists().for_category(slug))


def _items_scope(plan: BuyerPlan | None) -> BuyerPlan | None:
    """The plan with a budget for the items alone: a bundle's or a budget_max task's budget is arithmetic on prices,
    so it never covers shipping and tax."""
    return None if plan is None else replace(plan, scope="items")


_COUNTRY_MARKETS = {"GB": "GB", "DE": "EU", "FR": "EU", "IE": "EU"}
"""Country codes of data/agent/intents.yaml to the delivery market they are in (the personas' market keys)."""


def _other_countries(countries: Mapping[str, str], home: str | None) -> list[str]:
    """The country codes a shipping question may name: every one outside the store's own market, so the question is
    about shipping abroad; all of them when the home market is unknown or covers them all."""
    market = {"UK": "GB"}.get(home or "", home)
    out = [c for c in countries if _COUNTRY_MARKETS.get(c, c) != market]
    return out or list(countries)


def _forgetting_one(predicates: list[Predicate]) -> list[Predicate]:
    """The first on_ask predicate marked as forgotten the first time it is asked (a persona who forgets)."""
    index = next((i for i, p in enumerate(predicates) if p.source == "on_ask" and p.say), None)
    if index is None:
        return predicates
    return [p.model_copy(update={"forgets": True}) if i == index else p for i, p in enumerate(predicates)]


SPOKEN_WORDS = 4
SPOKEN_CHARS = 32
_GLUED = re.compile(r"[a-z]{3,}\d|\d[a-z]{4,}", re.IGNORECASE)
"""A word run into a number ("capacity9"): a seller's code, not a value."""


def speakable(value: str, field: str | None = None) -> bool:
    """Whether a shopper would say an option value aloud (D104): at most SPOKEN_WORDS words and SPOKEN_CHARS
    characters, no word run into a number, and a colour with no digits. Lazada's option values hold sellers' notes
    ("1-2 days travel capacity9", "rice white leather gun black buckle"), which an opening then asked for."""
    text = value.strip()
    if not text or len(text) > SPOKEN_CHARS or len(text.split()) > SPOKEN_WORDS or _GLUED.search(text):
        return False
    if field in ("color", "colour", "shade") and any(c.isdigit() for c in text):
        return False
    return any(c.isalnum() for c in text)


def _askable(predicate: Predicate, draft: _Draft) -> bool:
    """Whether a careful agent would ask the shopper for a need kept for asking (D104): an option only where the
    candidates offer a choice of it (a colour of a product sold in one colour is never asked), and a budget not when
    the shopper said "just pick one that works" (express). The judge fails an agent that ends with a hidden need
    unrevealed ("not revealed before ending: color"), so a need no agent would ask about measures the task."""
    if predicate.op == "max_price":
        return draft.intent != "express"
    if predicate.op != "eq" or not isinstance(predicate.value, str):
        return True
    field_name = fold(predicate.field)
    offered = {
        fold(value)
        for product in draft.pool
        for variant in product.variants
        if variant.available
        for name, value in variant.options
        if field_name in fold(name).replace("colour", "color")
    }
    return len(offered) > 1


def _the(title: str) -> str:
    """ "the" and a product's title, once: "the The Pastry Shop" was an opening of a published release (D104)."""
    return title if title[:4].lower() == "the " else f"the {title}"


def _sellable(product: Product) -> bool:
    return any(v.available and v.price is not None for v in product.variants)


def _listed(items: Sequence[str]) -> str:
    """ "the A", "the A and the B", "the A, the B and the C"."""
    if len(items) <= 2:
        return " and ".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _choices(value: SlotValue, pool: Sequence[Product]) -> int:
    """How many different values of this option the shelf offers."""
    if not value.option:
        return 0
    return len({found for p in pool for v in p.variants if v.available and (found := v.option(value.option))})


def _said_option(field_name: str, text: str) -> str:
    """An option value as the shopper says it (D119): a seller's note or model code cut to its plain value ("solid
    colors vests models-black" -> "black"), in the field's spoken case; the hidden spec keeps the store's value."""
    said = plain_value(text, speakable(text, field_name))
    return _spoken(field_name, said if said is not None else text)


def _said_twice(predicate: Predicate, said: str, draft: _Draft) -> bool:
    """Whether another value of the option on the task's candidates is said the same way, so the shopper's words
    would not single out the variant the spec asks for."""
    field_name = fold(predicate.field)
    for product in draft.pool:
        for variant in product.variants:
            for name, value in variant.options:
                if field_name not in fold(name).replace("colour", "color") or fold(value) == fold(str(predicate.value)):
                    continue
                if fold(plain_value(value, speakable(value, predicate.field)) or "") == fold(said):
                    return True
    return False


def _spoken(field_name: str, text: str) -> str:
    return text.lower() if field_name in SPOKEN_LOWER else text


def core_features(product: Product) -> list[tuple[str, str, str]]:
    """(field, phrase, how the shopper says it) for each union core fact of the product a shopper decides by (D117):
    its material, a care it allows, what it fits, a claim it carries, read from its stated details."""
    details = product.metadata.get("details")
    if not isinstance(details, Mapping):
        return []
    out: list[tuple[str, str, str]] = []
    material = str(details.get("material") or "").strip().lower()
    if material and len(material.split()) <= 3 and re.fullmatch(r"[a-z% -]+", material):
        out.append(("material", material, f"it should be made of {material}"))
    care = str(details.get("care") or "").lower()
    for phrase, say in CARE_FEATURES.items():
        if phrase in care:
            out.append(("care", phrase, say))
            break
    fits = str(details.get("compatibility") or "").strip()
    if (
        fits
        and len(fits.split()) <= 5
        and len(fits) <= 40
        and re.fullmatch(r"[\w .-]+", fits)
        and re.search(r"[A-Za-z]{3}", fits)
    ):
        out.append(("compatibility", fits, f"it has to be compatible with {fits}"))
    for claim in str(details.get("certifications") or "").split(","):
        word = claim.strip().lower()
        if word and len(word.split()) <= 2:
            out.append(("certifications", word, f"it needs to be {word}"))
            break
    return out


def _mentions_product(product: Product, phrase: str) -> bool:
    return any(mentions(product, v, phrase) for v in product.variants[:1])

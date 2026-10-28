"""The hidden spec, `mc.spec.v1`: what the shopper wants, as predicates the judge can check by variant id.

Why predicates with variant ids. The lab's accept-set verifier answered one question ("is the cart's variant in a
list") and readers found 73 false fails and 44 false passes in 383 journeys: hidden budgets and sizes were never
checked, and there was no notion of a correct refusal (lab/scoring/schema.py, 2026-09-26). Here each thing the
shopper wants is a predicate tagged with how the shopper discloses it:

- stated: said in the opening message;
- on_ask: said only when the agent asks about that field;
- default: never said; the shopper has no preference, and `value` is at most the plain assumption (one unit).

Each predicate carries `satisfied_by`, the variant ids among the task's candidates that satisfy it, computed from the
fixture data when the task was written. The judge checks a cart line by set membership and never has to read a size
chart or an ingredient list itself. `witnesses` are the available variants that satisfy every predicate; a task with
none is deliberately impossible (`impossible: true`), and refusing it is the right outcome.

The buyer block (optional, so every spec written before it still parses and judges the same) holds what the shopper
gives a checkout: a shipping address, an email and a phone number from their persona, how they want it delivered,
and whether their budget covers only the items or the total with shipping and tax. Each field carries a disclosure
source exactly like a predicate, so a judge can tell an address the shopper gave from one the agent made up. The
contact values are reserved ones only (carttrace.agents.reserved), so a spec can never carry a real person's details. The
block adds no required key, which is why the schema stays mc.spec.v1: a reader that does not know it loses nothing
it judged before, and the generator's version (`generator`) says which specs have one.

Generator 3 (2026-10-12) adds, each optional and left out of the JSON when unset, so every older spec keeps its bytes:
`forgets` on a predicate (the shopper does not have that on_ask fact to hand the first time and gives it when asked
again), `faults` (sealed sandbox knobs the task needs, a stockout of the named product for a substitute task; the
variants they take out of stock are never witnesses), and `voice` (what a model made of the template opening, kept so
a run never calls the model again). The intents grew from five to ten; a task's intent is in `intent`.

Decision D30 (2026-10-13) adds provenance, optional too and left out of the JSON when unset: `decision_attributes`,
one entry per attribute the task turns on (which attribute, its data element and niche checks, why the rules chose
it, how much of the store carries it, how many values it takes among the candidates, where that was read and on
which surfaces the store exposes it), and `chosen_by` on a predicate (`rules` when code chose the need, `model` when
a model suggested it and the rules accepted it). A spec written before keeps its bytes, and readers that do not know
the keys lose nothing they judged.

The format is JSON in `TaskRef.hidden_spec`, so any package can read it without importing this one (the judge sits
beside carttrace.agents on layer 5 and parses the same keys). A hand-written spec in the loose form
{"size": {"value": "M", "source": "on_ask"}} is read too, as equality predicates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from pydantic import ValidationError as PydanticValidationError

from carttrace.agents.reserved import EMAIL_HINT, PHONE_HINT, safe_email, safe_phone
from carttrace.core.errors import ValidationError
from carttrace.trace.schema import TaskRef

SPEC_SCHEMA = "mc.spec.v1"

Source = Literal["stated", "on_ask", "default"]
ChosenBy = Literal["rules", "model"]
"""Who chose a need: code from the store's data, or a model whose suggestion the rules accepted (D30)."""
ChosenBecause = Literal[
    "varies_across_candidates", "category_element", "store_carries", "niche_check", "checklist_item"
]  # fmt: skip
"""The rules a decision attribute passed, in this order (carttrace.agents.attributes); `checklist_item` marks a probe of
a niche check the store's data could not otherwise cover (D49)."""
CHOSEN_BECAUSE: tuple[ChosenBecause, ...] = (
    "varies_across_candidates", "category_element", "store_carries", "niche_check", "checklist_item"
)  # fmt: skip
StoreExposure = Literal["agent_view", "website_only", "absent"]
"""Where the store's capture-pass edition exposes a data kind (D49): on what an agent protocol surface returns (its
catalog: feed, tables, product .js), only on its product pages (tables and spec rows a person sees), or nowhere."""
STORE_EXPOSURES: tuple[StoreExposure, ...] = ("agent_view", "website_only", "absent")
ExposedOn = Literal["store_truth", "ucp_mcp", "ucp_rest", "webmcp", "acp", "website"]
"""Where a store exposes an attribute: its own catalog (store truth), an agent surface, or its web pages."""
EXPOSED_ON: tuple[ExposedOn, ...] = ("store_truth", "ucp_mcp", "ucp_rest", "webmcp", "acp", "website")
SOURCES: tuple[Source, ...] = ("stated", "on_ask", "default")
IntentType = Literal[
    "replenish", "specify", "compare", "discover", "express", "gift", "bundle", "budget_max", "substitute", "policy"
]  # fmt: skip
INTENTS: tuple[IntentType, ...] = ("replenish", "specify", "compare", "discover", "express")
"""The five intents every plan had before generator 3: what `generate_tasks` writes when asked for no others, in this
order, so the tasks of an older plan are the tasks it always had."""
NEW_INTENTS: tuple[IntentType, ...] = ("gift", "bundle", "budget_max", "substitute", "policy")
"""Generator 3's intents: a present for someone else with a deadline and a message; two or three items that go
together; the most of one product a budget buys; a named product the store has run out of, where the closest
substitute or a refusal is right; and a question the store's policies answer before the purchase."""
ALL_INTENTS: tuple[IntentType, ...] = (*INTENTS, *NEW_INTENTS)
Op = Literal[
    "kind",
    "is",
    "one_of",
    "all_of",
    "prefer",
    "brand",
    "eq",
    "mentions",
    "max_price",
    "count",
    "lines",
    "fits",
    "excludes",
    "safe_for",
    "cheapest",
    "cheapest_per_unit",
    "most",
    "least",
    "note",
    "checkout",
    "policy",
]
"""How a predicate is checked against a variant:

- kind: the product is the kind of thing asked for (`value` is the shopper's noun);
- is: the variant belongs to the named product (`value` is the product title);
- one_of: the variant belongs to one of the products the shopper named (`value` is their titles), as a compare task
  names its candidates;
- all_of: a bundle: the cart must hold one line from each product the shopper named (`value` is a list of
  {"title", "product_id", "variants"}); one variant satisfies it by belonging to any of them, and the judge checks
  that every item is in the cart;
- prefer: the variant belongs to the named product (`value` is its title), or, when the shopper allows a substitute
  (a `substitute_ok` predicate whose value is true), to another product of the shelf that satisfies every other
  predicate; a substitute task's named product is taken out of stock by the spec's `faults`;
- brand: the product's brand or vendor is `value`;
- eq: the variant's option `option` equals `value` (a `default` eq with no value means no preference);
- mentions: the product's title, options or description mention `value` ("ibuprofen", "200 mg", "tablet");
- max_price: unit price times the wanted quantity is at most `value` ({"amount": minor units, "currency": ...});
- count: the cart line's quantity is `value` (checked on the cart, so `satisfied_by` is None);
- lines: the cart holds at most `value` distinct items (a bundle sets it to its item count; checked on the cart);
- fits: the store's size chart maps the measurement in `value` ({"measure", "amount", "unit"}) to the variant's size;
- excludes: the product's ingredients contain none of the allergens in `value`;
- safe_for: the product's warnings say nothing against the conditions in `value`;
- cheapest: the variant is the cheapest of those that satisfy every other predicate;
- cheapest_per_unit: the variant costs least per unit of what it holds (per tablet, per 100 g, per 100 ml) of those
  that satisfy every other predicate (carttrace.agents.measures);
- most, least: the product's label states the most (or least) of a nutrient per 100 g (`value` is {"nutrient": ...})
  of those that satisfy every other predicate;
- note: a fact the shopper knows that only qualifies another predicate (the sizing system of a stated size); it is
  never checked on its own, so `satisfied_by` is None;
- checkout: a wish no variant settles, which the checkout or the agent's conduct does: who a gift is for, the gift
  message, whether a substitute is acceptable. Never checked on a variant (`satisfied_by` is None); the judge records
  it in the checkout scope;
- policy: a question the store's policies answer (`value` is {"topic": "returns"} or {"topic": "shipping",
  "market": ...}). The judge checks the agent's answer against the policy text the store sent when the trace holds
  one, and leaves it unsettled otherwise.
"""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class Predicate(_Frozen):
    field: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    """A field of the shared vocabulary (data/categories/_fields.yaml): size, budget, allergens, ..."""
    op: Op
    value: JsonValue = None
    source: Source
    say: str = ""
    """How the shopper words it, as a clause without a final stop: "my chest measures 102 cm"."""
    option: str | None = None
    """The store's own option name behind an `eq` predicate, e.g. "Colour"."""
    satisfied_by: list[str] | None = None
    """Variant ids among the candidates that satisfy this predicate; None when it is checked on the cart line."""
    aliases: list[str] = Field(default_factory=list)
    """Other fields a question may name when it is really about this one: someone who knows a measurement but not
    their size answers a question about size with the measurement."""
    forgets: bool = False
    """The shopper does not have this on_ask fact to hand the first time the agent asks, and gives it when asked
    again (a persona who forgets one detail). Left out of the JSON when false."""
    chosen_by: ChosenBy | None = None
    """Who chose this need (D30): `rules` (carttrace.agents.attributes chose it from the store's data) or `model` (a model
    suggested it and the same rules accepted it). Left out of the JSON when unset, as in every spec before D30."""


class Voice(_Frozen):
    """What a model made of the task's template opening (carttrace.agents.intents_voice), kept in the spec so a run is
    reproducible without calling the model again. `accepted` says whether the rewrite stands as the opening; when the
    guard refused it, `reason` says why and the template opening stands."""

    model: str
    accepted: bool
    template_opening: str
    reason: str = ""
    cost_usd: float | None = None
    prompt_sha256: str | None = None
    """sha256 of the prompt the voice model was sent (D69), so a trace records who wrote the opening and from what.
    None on specs voiced before D69, and left out of the spec then, so their task ids hold."""


class Decisive(_Frozen):
    """The detail of the store's data that decides the task: the part the benchmark is measuring."""

    detail: str
    """size_chart, allergens, warnings, or options (a structured variant option, the fallback)."""
    field: str | None = None
    """The hidden-spec field the detail decides (measurements, allergens, conditions, ...)."""
    missing: str | None = None
    """What an agent records when the store does not expose the detail: size_chart, ingredients, warnings."""
    hypothesis: str | None = None
    """The category checklist hypothesis this task tests, when the category names one."""


class DecisionAttribute(_Frozen):
    """One attribute a task turns on, and why the rules chose it (D30). The names are agreed with the judge, whose
    warehouse keys its check schema on `element_id` and `attribute`."""

    attribute: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    """The hidden-spec field of the need it decides: flavor, budget, unit_price, allergens, dietary, ..."""
    element_id: str = Field(min_length=1)
    """The data element (data/elements: a category hypothesis id, a shared D19 element) or, where none names it, the
    exposure taxonomy row (docs/STORE-EXPOSURE.md)."""
    check_ids: list[str] = Field(default_factory=list)
    """The niche checks (data/agent/checks.yaml) whose data the attribute is."""
    data_kind: str = Field(min_length=1)
    """The store brief's data kind (carttrace.agents.brief.DATA_KINDS)."""
    chosen_because: list[ChosenBecause] = Field(default_factory=list)
    store_coverage: float = Field(ge=0, le=1)
    """The share of the store's products that carry the data."""
    distinct_values: int = Field(ge=0)
    """How many values the attribute takes among the task's candidates (two readings, such as contains and free,
    for a yes-or-no attribute)."""
    source: str = Field(pattern=r"^(capture:|inventory:)\S+$|^fixtures$")
    """What the coverage and the surfaces were read from: capture:<id>, inventory:<id>, or fixtures."""
    exposed_on: list[ExposedOn] = Field(default_factory=list)
    rule_version: str = Field(min_length=1)
    """The rules' version (carttrace.agents.attributes.RULE_VERSION) that chose it."""
    store_exposure: StoreExposure | None = None
    """Where the store's edition exposes the data kind (D49): agent_view, website_only or absent; left out of the
    spec's bytes when not recorded (sets written before it)."""


SUPERLATIVES: frozenset[Op] = frozenset({"cheapest", "cheapest_per_unit", "most", "least"})
"""Ops that pick the best of the variants satisfying every other predicate, so they are checked over a set."""

FulfillmentKind = Literal["standard", "express", "pickup"]
FULFILLMENT_KINDS: tuple[FulfillmentKind, ...] = ("standard", "express", "pickup")
BudgetScope = Literal["items", "total"]
BUYER_FIELDS = ("shipping_address", "email", "phone", "fulfillment", "budget_scope")
"""The buyer block's fields, in the order an agent meets them at checkout."""


class Address(_Frozen):
    """A shipping address as the shopper gives it: fictional, the persona's address in the store's market."""

    name: str | None = None
    """An optional fictional name line; the shipped personas have none, because the sandbox keeps no names."""
    lines: list[str] = Field(min_length=1, max_length=2)
    """Street address lines: the first, and an optional second."""
    city: str = Field(min_length=1)
    region: str | None = None
    postal_code: str = Field(min_length=1)
    country: str = Field(pattern=r"^[A-Z]{2}$")
    """ISO 3166-1 alpha-2."""

    def words(self) -> str:
        """The address as a shopper writes it on one line: "1 Maple Way, Springfield, IL 62701, US"."""
        place = " ".join(part for part in (self.region, self.postal_code) if part)
        parts = [self.name, *self.lines, self.city, place, self.country]
        return ", ".join(part for part in parts if part)


class BuyerAddress(_Frozen):
    value: Address
    source: Source
    say: str = ""


class BuyerContact(_Frozen):
    value: str
    source: Source
    say: str = ""


class BuyerEmail(BuyerContact):
    @field_validator("value")
    @classmethod
    def _reserved(cls, value: str) -> str:
        if not safe_email(value):
            raise ValueError(f"a buyer email must be reserved: {EMAIL_HINT}")
        return value


class BuyerPhone(BuyerContact):
    @field_validator("value")
    @classmethod
    def _reserved(cls, value: str) -> str:
        if not safe_phone(value):
            raise ValueError(f"a buyer phone number must be reserved: {PHONE_HINT}")
        return value


class BuyerFulfillment(_Frozen):
    value: FulfillmentKind = "standard"
    """standard or express delivery, or pickup where the store offers it. A default one is the plain assumption,
    standard delivery."""
    deadline_days: int | None = Field(default=None, ge=1, le=60)
    """The most days the shopper can wait, met by an option whose stated delivery time is no longer."""
    source: Source = "default"
    say: str = ""


class BuyerBudgetScope(_Frozen):
    value: BudgetScope = "items"
    """items: the budget is for the items alone; total: it covers everything paid, shipping and tax included."""
    source: Source = "default"
    say: str = ""


BuyerField = BuyerAddress | BuyerContact | BuyerFulfillment | BuyerBudgetScope


class Buyer(_Frozen):
    """What the shopper gives a checkout, each field with its disclosure source."""

    persona: str | None = None
    """The persona (data/agent/personas.yaml) the details came from."""
    market: str | None = None
    """The delivery market the address is in."""
    shipping_address: BuyerAddress | None = None
    email: BuyerEmail | None = None
    phone: BuyerPhone | None = None
    fulfillment: BuyerFulfillment | None = None
    budget_scope: BuyerBudgetScope = Field(default_factory=BuyerBudgetScope)

    def field(self, name: str) -> BuyerField | None:
        if name not in BUYER_FIELDS:
            return None
        found: BuyerField | None = getattr(self, name)
        return found

    def fields(self) -> list[str]:
        """The fields this block holds, in BUYER_FIELDS order."""
        return [name for name in BUYER_FIELDS if self.field(name) is not None]

    def source_of(self, name: str) -> Source | None:
        found = self.field(name)
        return None if found is None else found.source


class HiddenSpec(_Frozen):
    schema_: Literal["mc.spec.v1"] = Field(default="mc.spec.v1", alias="schema")
    intent: IntentType | None = None
    decisive: Decisive = Field(default_factory=lambda: Decisive(detail="options"))
    predicates: list[Predicate] = Field(default_factory=list)
    witnesses: list[str] = Field(default_factory=list)
    """Available variant ids that satisfy every predicate. Empty exactly when the task is impossible."""
    candidates: list[str] = Field(default_factory=list)
    """Product ids the task was judged over: the shelf a shopper asking this would look at."""
    impossible: bool = False
    currency: str | None = None
    generator: str = ""
    """Which generator and version wrote the spec, e.g. "mc.agents.tasks/2"."""
    buyer: Buyer | None = None
    """What the shopper gives a checkout; None in specs written before the block existed."""
    faults: list[dict[str, Any]] = Field(default_factory=list)
    """Sealed fault knobs (carttrace.sandbox.faults) the task needs the sandbox world to carry: a substitute task's stockout
    of the named product's variants. Empty, and left out of the JSON, for every other task."""
    voice: Voice | None = None
    """The model's rewrite of the opening, when a voice pass ran; None (left out of the JSON) otherwise."""
    decision_attributes: list[DecisionAttribute] = Field(default_factory=list)
    """What the task turns on and why the rules chose it (D30); empty, and left out of the JSON, in older specs."""

    @property
    def budget_scope(self) -> BudgetScope:
        """Whether the budget covers the items alone (the default) or the total with shipping and tax."""
        return self.buyer.budget_scope.value if self.buyer is not None else "items"

    @property
    def stocked_out(self) -> frozenset[str]:
        """Variant ids the spec's own faults take out of stock: never witnesses, whatever the fixtures say."""
        return frozenset(
            str(f["variant"]) for f in self.faults if f.get("kind") == "stockout" and isinstance(f.get("variant"), str)
        )

    @property
    def substitute_ok(self) -> bool:
        """Whether the shopper accepts a substitute for a named product that is out of stock (a substitute task)."""
        found = self.predicate("substitute_ok")
        return found is not None and found.value is True

    def predicate(self, field: str) -> Predicate | None:
        return next((p for p in self.predicates if p.field == field), None)

    def by_source(self, *sources: Source) -> list[Predicate]:
        return [p for p in self.predicates if p.source in sources]

    def fields(self) -> list[str]:
        return [p.field for p in self.predicates]

    def alias_of(self, field: str) -> Predicate | None:
        """The predicate a question about `field` is really about, when `field` is one of its aliases."""
        return next((p for p in self.predicates if field in p.aliases), None)

    def to_json(self) -> dict[str, Any]:
        """The dict that goes in TaskRef.hidden_spec. A spec without a buyer block has no `buyer` key, one without
        faults, a voice or decision attributes has none of those keys, and a predicate that forgets nothing or names no
        chooser has no `forgets` or `chosen_by` key, so a spec that uses none of them is the same bytes as before they
        existed."""
        data: dict[str, Any] = self.model_dump(mode="json", by_alias=True)
        for key in ("buyer", "voice"):
            if data.get(key) is None:
                data.pop(key, None)
        if isinstance(data.get("voice"), dict) and data["voice"].get("prompt_sha256") is None:
            data["voice"].pop("prompt_sha256", None)
        for key in ("faults", "decision_attributes"):
            if not data.get(key):
                data.pop(key, None)
        for attribute in data.get("decision_attributes", []):
            if attribute.get("store_exposure") is None:
                attribute.pop("store_exposure", None)
        for predicate in data.get("predicates", []):
            if predicate.get("forgets") is False:
                predicate.pop("forgets", None)
            if predicate.get("chosen_by") is None:
                predicate.pop("chosen_by", None)
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> HiddenSpec:
        """A spec from TaskRef.hidden_spec: the mc.spec.v1 form, or the loose form of field to value."""
        if data.get("schema") == SPEC_SCHEMA:
            try:
                return cls.model_validate(dict(data))
            except PydanticValidationError as exc:
                first = exc.errors()[0]
                where = ".".join(str(p) for p in first["loc"]) or "spec"
                raise ValidationError(
                    f"hidden spec is not valid {SPEC_SCHEMA}: {where}: {first['msg']}",
                    hint="write specs with carttrace.agents.generate_tasks, or follow carttrace/agents/spec.py",
                ) from exc
        return cls(predicates=[_loose(field, raw) for field, raw in data.items()])

    @classmethod
    def from_task(cls, task: TaskRef) -> HiddenSpec:
        return cls.from_mapping(task.hidden_spec)


def _loose(field: str, raw: Any) -> Predicate:
    """One field of a hand-written spec: {"value": ..., "source": ...} or a bare value, which stays hidden until
    asked, because a value nobody tagged must not be volunteered."""
    entry: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {"value": raw}
    source = entry.get("source", "on_ask")
    if source not in SOURCES:
        raise ValidationError(
            f"hidden spec field {field!r} has source {source!r}", hint=f"use one of {', '.join(SOURCES)}"
        )
    value = entry.get("value")
    say = entry.get("say")
    if not isinstance(say, str) or not say:
        say = f"{field.replace('_', ' ')} {value}" if value is not None else ""
    try:
        return Predicate(field=field, op="eq", value=value, source=source, say=say)
    except PydanticValidationError as exc:
        raise ValidationError(
            f"hidden spec field {field!r} cannot be read: {exc.errors()[0]['msg']}",
            hint="field names are lowercase identifiers and values are JSON",
        ) from exc


def validate_spec(spec: HiddenSpec, *, where: str = "task") -> None:
    """Refuse a spec that could not be judged: no witness and not marked impossible, an impossible spec that names a
    witness, a field given twice, or a witness that does not satisfy every predicate it is checked against."""
    if spec.impossible and spec.witnesses:
        raise ValidationError(
            f"{where} is marked impossible but names witnesses {spec.witnesses[:3]}",
            hint="an impossible task has no variant that satisfies it; drop the witnesses or the flag",
        )
    if not spec.impossible and not spec.witnesses:
        raise ValidationError(
            f"{where} names no witness and is not marked impossible",
            hint="a task needs at least one variant that satisfies it, or impossible: true when refusal is right",
        )
    fields = spec.fields()
    repeated = sorted({f for f in fields if fields.count(f) > 1})
    if repeated:
        raise ValidationError(f"{where} gives fields {repeated} twice", hint="one predicate per field")
    if len([p for p in spec.predicates if p.op in SUPERLATIVES]) > 1:
        raise ValidationError(f"{where} asks for the best of the candidates twice", hint="one superlative per task")
    if spec.buyer is not None:
        _check_buyer(spec, spec.buyer, where)
    _check_shape(spec, where)
    if spec.faults:
        from carttrace.sandbox.faults import compile_faults  # the knobs are checked as the sandbox checks them

        compile_faults(spec.faults)
        held = spec.stocked_out & set(spec.witnesses)
        if held:
            raise ValidationError(
                f"{where} names witnesses {sorted(held)[:3]} that its own faults take out of stock",
                hint="a stocked-out variant is never a witness; regenerate the task",
            )
    for predicate in spec.predicates:
        if predicate.satisfied_by is None:
            continue
        outside = sorted(set(spec.witnesses) - set(predicate.satisfied_by))
        if outside:
            raise ValidationError(
                f"{where}: witnesses {outside[:3]} do not satisfy {predicate.field} ({predicate.op})",
                hint="witnesses are the variants that satisfy every predicate; regenerate the task",
            )


def _check_shape(spec: HiddenSpec, where: str) -> None:
    """Refuse a bundle without its line count or a line count without a bundle, a prefer with no substitute answer,
    and a policy predicate whose value names no topic: each would leave the judge or the shopper guessing."""
    bundle = next((p for p in spec.predicates if p.op == "all_of"), None)
    lines = next((p for p in spec.predicates if p.op == "lines"), None)
    if (bundle is None) != (lines is None):
        raise ValidationError(f"{where}: a bundle (all_of) and its line count (lines) go together", hint="add both")
    if bundle is not None:
        items = bundle.value if isinstance(bundle.value, list) else []
        if len(items) < 2 or lines is None or lines.value != len(items):
            raise ValidationError(
                f"{where}: a bundle names at least two items and its lines predicate counts them",
                hint="value is a list of {title, product_id, variants}; lines.value is its length",
            )
    if any(p.op == "prefer" for p in spec.predicates) and spec.predicate("substitute_ok") is None:
        raise ValidationError(f"{where}: a prefer predicate needs a substitute_ok predicate", hint="add one")
    for p in spec.predicates:
        if p.op == "policy" and not (isinstance(p.value, Mapping) and isinstance(p.value.get("topic"), str)):
            raise ValidationError(f"{where}: a policy predicate's value names a topic", hint='{"topic": "returns"}')


def _check_buyer(spec: HiddenSpec, buyer: Buyer, where: str) -> None:
    """Refuse a buyer block a judge could not read one way: an all-in budget scope with no budget to scope, or a
    pickup with a delivery deadline."""
    if buyer.budget_scope.value == "total" and spec.predicate("budget") is None:
        raise ValidationError(
            f"{where} says the budget covers shipping and tax, and gives no budget",
            hint="a budget scope of total needs a budget predicate; leave it as items otherwise",
        )
    fulfillment = buyer.fulfillment
    if fulfillment is not None and fulfillment.value == "pickup" and fulfillment.deadline_days is not None:
        raise ValidationError(f"{where} gives a pickup a delivery deadline", hint="only delivery has a deadline")


def intersect(sets: Iterable[Iterable[str]], universe: Iterable[str]) -> list[str]:
    """The ids in `universe` that are in every one of `sets`, in universe order."""
    keep = list(universe)
    for ids in sets:
        allowed = set(ids)
        keep = [i for i in keep if i in allowed]
    return keep

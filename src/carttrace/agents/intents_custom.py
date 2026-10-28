"""Custom intents, `mc.intents.custom.v1`: free-form shopping intents written per store by a model from the store's
brief, each with a hidden spec code validated against the capture, spread over the data kinds the store exposes.

Why a model writes them, and why code validates. The template generator (carttrace.agents.tasks) decides a task by one of a
few details, so a food store gets allergen tasks over and over and the other data it exposes is never tested. The
owner asked for the old lab's way back: an analyzer summarises the store (carttrace.agents.brief) and an intent writer
writes a hundred intents for that store, each turning on something the store exposes, in a persona's own words. D9 and
D20 keep free text out of what is judged, so the model proposes and code disposes: every predicate must be an
mc.spec.v1 predicate, every candidate product and witness variant must exist in the capture, the witnesses are
computed from the capture (never taken from the model), an impossible intent must be impossible, stated values must
be in the opening and kept-back values must not, and no intent may talk of paying. An intent that fails is rejected
with its reason, the reasons go back to the model in the next round, and the quotas (no data kind above a share of
the set, at least so many kinds when the store exposes them) are enforced by the same validator.

The set is written once per store and capture (var/agent/intents/<store>/<capture id>.json) and reused: a manifest
drawing from it calls no model. Costs are logged per call and capped per store.

Since D30 this is the comparison path (`carttrace agents intents write --selection model`): by default the rules of
carttrace.agents.attributes choose every need and a model only voices the opening (carttrace.agents.intents_rules), and a set
written that way names `selection: rules` and is filed beside this one as <capture id>.rules.json. A need a model
chose here says so (`chosen_by: model` on each predicate), so the two kinds of set can be told apart task by task.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from pydantic import ValidationError as PydanticValidationError

from carttrace.agents.attributes import allergen_guard
from carttrace.agents.brief import ALWAYS_KINDS, KIND_IDS, StoreBrief
from carttrace.agents.buyer import EXPRESS_WORDS, PICKUP_WORDS, buyer_block, draw_plan, market_options
from carttrace.agents.checks import Check, Checklists, agent_checks, load_checklists, must_cover
from carttrace.agents.consistency import contradictions
from carttrace.agents.intents_voice import MAX_CHARS, MAX_SENTENCES, VoiceReply, VoiceRunner, value_words
from carttrace.agents.nouns import names_product, noun_from
from carttrace.agents.phrases import Phrases, load_phrases
from carttrace.agents.predicates import CART_OPS
from carttrace.agents.products import Product, read_products
from carttrace.agents.spec import (
    SUPERLATIVES,
    BuyerAddress,
    BuyerContact,
    ChosenBy,
    DecisionAttribute,
    Decisive,
    FulfillmentKind,
    HiddenSpec,
    IntentType,
    Op,
    Predicate,
    Source,
    StoreExposure,
    validate_spec,
)
from carttrace.agents.tasks import witnesses_of
from carttrace.agents.text import fold
from carttrace.agents.wallet import Personas, load_personas
from carttrace.core.canon import digest, dumps, loads
from carttrace.core.clock import Clock, SystemClock, iso
from carttrace.core.errors import MCError, NotFoundError, ValidationError
from carttrace.core.files import append_jsonl, atomic_write_text
from carttrace.core.ids import seeded_rng, stable_id
from carttrace.trace.schema import TaskRef

INTENT_SET_SCHEMA: Final = "mc.intents.custom.v1"
INTENTS_DIR: Final = Path("agent") / "intents"
"""Under paths.var: <store>/<source>.json, one set per store and capture; log.jsonl beside them."""
GENERATOR: Final = "mc.agents.intents_custom/2"
DEFAULT_N: Final = 100
DEFAULT_BATCH: Final = 20
DEFAULT_MODEL: Final = "sonnet"
KIND_CAP_SHARE: Final = 0.15
"""No one decisive data kind may take more than this share of a store's intents."""
MIN_KINDS: Final = 8
"""At least this many data kinds per store, when the store exposes that many."""
MAX_ROUNDS_EXTRA: Final = 4
"""Rounds beyond ceil(n / batch) a writer may spend re-asking for rejected intents."""
WRITE_TIMEOUT_S: Final = 600.0
_PAYMENT: Final = re.compile(r"check ?out|pay now|buy now|purchase now|place (?:the|an|my) order|credit card", re.I)
_DIGITS: Final = re.compile(r"\d+")
_SENTENCE: Final = re.compile(r"[.!?](?:\s|$)")
_FENCE: Final = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


# -- what the model returns -------------------------------------------------------------------------------------------


class DraftPredicate(_Frozen):
    field: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    op: Op
    value: JsonValue = None
    source: Source
    say: str = ""
    option: str | None = None


class DraftFulfillment(_Frozen):
    value: FulfillmentKind = "standard"
    deadline_days: int | None = Field(default=None, ge=1, le=60)
    source: Source = "default"


class IntentDraft(_Frozen):
    """One intent as the model proposes it. Everything here is checked before it becomes a CustomIntent."""

    persona: str
    intent: IntentType
    intent_kind: str = Field(min_length=3, max_length=120)
    decisive_kind: str
    opening: str = Field(min_length=10)
    predicates: list[DraftPredicate] = Field(min_length=1)
    candidates: list[str] = Field(min_length=1)
    witnesses: list[str] = Field(default_factory=list)
    impossible: bool = False
    fulfillment: DraftFulfillment | None = None
    checks: list[str] = Field(default_factory=list)
    """The niche checklist ids the intent exercises (FB-07, CL-06); at least one where the store has a niche."""


# -- what is kept ---------------------------------------------------------------------------------------------------


class CustomIntent(_Frozen):
    id: str
    persona: str
    intent: IntentType
    intent_kind: str
    decisive_kind: str
    opening: str
    spec: dict[str, Any]
    """The hidden spec, mc.spec.v1, with witnesses computed from the capture."""
    checks: list[str] = Field(default_factory=list)
    """The niche checklist ids it exercises (data/agent/checks.yaml)."""
    needs_replica: bool = False
    """True when only the replica can run it: the spec carries faults, a checkout-stage wish (a `checkout`
    predicate, a stated delivery wish) or a checkout-stage check. A live row never gets such an intent."""
    chosen_by: ChosenBy = "model"
    """Who chose what the intent tests: the rules (D30), or a model (the D27 writer, or a suggestion the rules
    accepted)."""
    store_exposure: StoreExposure | None = None
    """Where the store's edition exposes the data the intent turns on (D49): agent_view, website_only or absent;
    None on sets written before it, and on the D27 model writer's."""
    probe: bool = False
    """A probe of a niche check (D49): the shopper asks for what the check names whether or not the store exposes it,
    so a store that lacks it meets the request; impossible when no candidate states it."""


class Rejected(_Frozen):
    reason: str
    intent_kind: str = ""
    decisive_kind: str = ""
    opening: str = ""


class CheckCoverage(_Frozen):
    """How one niche check of the store's category is tested by the set (D49): where the store exposes its data,
    how many intents exercise it, and when none does, why it cannot be asked of a shopper here."""

    exposure: StoreExposure
    kinds: list[str] = Field(default_factory=list)
    """The data kinds the check reads (data/agent/checks.yaml); empty for a check of no product data."""
    intents: int = 0
    probes: int = 0
    """Of `intents`, the probes written because no rule-built need covered the check."""
    reason: str = ""


class Quotas(_Frozen):
    n: int = Field(ge=1)
    kind_cap: int = Field(ge=1)
    """Most intents one decisive data kind may take."""
    min_kinds: int = Field(ge=1)
    """Fewest distinct decisive data kinds the set must spread over."""

    @classmethod
    def for_(cls, n: int, exposed: int, *, share: float = KIND_CAP_SHARE, min_kinds: int = MIN_KINDS) -> Quotas:
        return cls(n=n, kind_cap=max(1, math.ceil(n * share)), min_kinds=max(1, min(min_kinds, exposed, n)))


class IntentSet(_Frozen):
    schema_: Literal["mc.intents.custom.v1"] = Field(default=INTENT_SET_SCHEMA, alias="schema")
    id: str = ""
    """set-<sha12> of everything else."""
    store_id: str
    category: str
    source: str
    """The capture (or fixtures) hash the brief was read from; the set is filed under it."""
    brief_id: str
    model: str
    written_at: str
    generator: str = GENERATOR
    quotas: Quotas
    kinds: dict[str, int] = Field(default_factory=dict)
    """Decisive data kind to how many intents turn on it."""
    intent_kinds: dict[str, int] = Field(default_factory=dict)
    checks: dict[str, int] = Field(default_factory=dict)
    """Niche check id to how many intents exercise it."""
    uncovered: list[str] = Field(default_factory=list)
    """The checks the set had to cover (Critical and High ones the store exposes the data for, and the shared
    ones) and no intent exercises."""
    intents: list[CustomIntent] = Field(default_factory=list)
    rejected: list[Rejected] = Field(default_factory=list)
    calls: int = 0
    cost_usd: float = 0.0
    quotas_met: bool = False
    notes: list[str] = Field(default_factory=list)
    selection: Literal["rules", "model"] = "model"
    """Who chose the needs (D30): the rules, with a model voicing the openings, or the D27 model writer."""
    rule_version: str = ""
    """The rules' version (carttrace.agents.attributes) when the rules chose."""
    attributes: dict[str, int] = Field(default_factory=dict)
    """`<data kind>/<attribute>` to how many intents turn on it, when the rules chose."""
    voiced: int = 0
    """Openings a model voiced that the guard let stand."""
    suggested: int = 0
    """Intents a model suggested; `suggestions_accepted` of them passed the rules."""
    suggestions_accepted: int = 0
    check_coverage: dict[str, CheckCoverage] = Field(default_factory=dict)
    """Every niche check of the store's category (D49): its exposure, its intents, or the reason it has none."""

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> IntentSet:
        if data.get("schema") != INTENT_SET_SCHEMA:
            raise ValidationError(
                f"an intent set declares schema {data.get('schema')!r}; this reads {INTENT_SET_SCHEMA}"
            )
        try:
            found = cls.model_validate(dict(data))
        except PydanticValidationError as exc:
            first = exc.errors(include_url=False, include_input=False)[0]
            where = ".".join(str(p) for p in first["loc"]) or "set"
            raise ValidationError(f"intent set is not valid {INTENT_SET_SCHEMA}: {where}: {first['msg']}") from None
        return found


def set_id(intent_set: IntentSet) -> str:
    return f"set-{digest(intent_set.model_dump(mode='json', by_alias=True, exclude={'id'}), 12)}"


def intents_path(var: Path, store_id: str, source: str, selection: str = "model") -> Path:
    """Where a store's set for one capture is filed: <source>.json for the model writer, <source>.rules.json for
    the rules (D30), so the two can stand side by side for comparison."""
    suffix = ".rules.json" if selection == "rules" else ".json"
    return var / INTENTS_DIR / store_id / f"{source}{suffix}"


def intents_log(var: Path) -> Path:
    return var / INTENTS_DIR / "log.jsonl"


def save_intent_set(intent_set: IntentSet, var: Path) -> Path:
    sealed = intent_set.model_copy(update={"id": set_id(intent_set)})
    path = intents_path(var, sealed.store_id, sealed.source, sealed.selection)
    atomic_write_text(path, dumps(sealed.to_json()) + "\n")
    return path


def load_intent_set(path: Path) -> IntentSet:
    try:
        return IntentSet.from_mapping(loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise NotFoundError(
            f"no intent set at {path}", hint="write one with `carttrace agents intents write STORE`"
        ) from None
    except ValueError as exc:
        raise ValidationError(f"{path} is not JSON: {exc}") from exc


def latest_intent_set(var: Path, store_id: str) -> Path | None:
    """The store's newest intent set by the time it was written, or None."""
    folder = var / INTENTS_DIR / store_id
    if not folder.is_dir():
        return None
    found: list[tuple[str, Path]] = []
    for path in sorted(folder.glob("*.json")):
        try:
            found.append((load_intent_set(path).written_at, path))
        except MCError:
            continue
    return max(found)[1] if found else None


# -- validation -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Catalog:
    """What an intent is checked against: the capture's products by id, and every variant's product."""

    products: dict[str, Product]
    owner: dict[str, str]

    @classmethod
    def of(cls, fixtures: Sequence[Mapping[str, Any]], *, currency: str | None = None) -> Catalog:
        products = read_products(fixtures, currency=currency)
        return cls({p.id: p for p in products}, {v.id: p.id for p in products for v in p.variants})


@dataclass(frozen=True, slots=True)
class Validation:
    intent: CustomIntent | None
    reason: str | None


@dataclass(slots=True)
class Usage:
    """What a store's accepted intents have used so far: per data kind, per check, and the openings taken."""

    kinds: dict[str, int] = field(default_factory=dict)
    checks: dict[str, int] = field(default_factory=dict)
    openings: set[str] = field(default_factory=set)

    def take(self, intent: CustomIntent) -> None:
        self.kinds[intent.decisive_kind] = self.kinds.get(intent.decisive_kind, 0) + 1
        for check in intent.checks:
            self.checks[check] = self.checks.get(check, 0) + 1
        self.openings.add(fold(intent.opening))


def needs_replica(spec: Mapping[str, Any], *, checkout_stage: bool = False) -> bool:
    """Whether only the replica can run a task with this spec: it carries faults (a sealed stockout), a `checkout`
    predicate (a gift message, a substitute answer), a stated delivery wish, or (`checkout_stage`) exercises a check
    settled at checkout. A live row ends at a verified cart and holds none of these."""
    if spec.get("faults") or checkout_stage:
        return True
    raw_predicates = spec.get("predicates")
    predicates: list[Any] = raw_predicates if isinstance(raw_predicates, list) else []
    if any(isinstance(p, Mapping) and p.get("op") == "checkout" for p in predicates):
        return True
    raw_buyer = spec.get("buyer")
    buyer: Mapping[str, Any] = raw_buyer if isinstance(raw_buyer, Mapping) else {}
    raw_fulfillment = buyer.get("fulfillment")
    fulfillment: Mapping[str, Any] = raw_fulfillment if isinstance(raw_fulfillment, Mapping) else {}
    return fulfillment.get("source") == "stated"


@dataclass(frozen=True, slots=True)
class Checker:
    """What every draft of one store is checked against: the catalog, the brief, the personas and phrases, the
    market the buyer block is drawn in, and the set's quotas."""

    store_id: str
    catalog: Catalog
    brief: StoreBrief
    personas: Personas
    phrases: Phrases
    market: str | None
    quotas: Quotas
    checks: tuple[Check, ...] = ()
    """The niche checks an intent of this store may exercise (carttrace.agents.checks.agent_checks); empty for a category
    no P1 niche covers, and then intents name none."""

    @property
    def must(self) -> list[str]:
        """The check ids the set must cover."""
        return must_cover(self.checks)

    def validate(
        self, draft: IntentDraft, usage: Usage, *, require_checks: bool = True, probe: bool = False
    ) -> Validation:
        """The draft as a CustomIntent, or why it cannot stand. Nothing the model said about witnesses is trusted:
        they are computed from the catalog, and the model's own witnesses must be among them. `require_checks` False
        lets an intent name no niche check (the rules name the checks an attribute's data is, and some have none).
        `probe` is a D49 probe of a niche check: the store need not expose its kind, the check need not be one the
        store's data supports, and the allergen guard does not stand (asking is the point)."""
        reason = self._names(draft, usage, require_checks=require_checks, probe=probe)
        if reason is not None:
            return Validation(None, reason)
        built = self._spec(draft)
        if isinstance(built, str):
            return Validation(None, built)
        spec, pool = built
        wider = self._store_wide(spec, pool)
        if wider:
            draft = draft.model_copy(update={"candidates": [*draft.candidates, *wider]})
            built = self._spec(draft)
            if isinstance(built, str):
                return Validation(None, built)
            spec, pool = built
        found = witnesses_of(spec, pool)
        reason = self._witnesses(draft, found) or check_opening(draft.opening, spec, draft.fulfillment)
        if reason is not None:
            return Validation(None, reason)
        # a superlative is satisfied by the best of the rest: the witnesses, as the generator writes it
        predicates = [
            p.model_copy(update={"satisfied_by": list(found)}) if p.op in SUPERLATIVES else p for p in spec.predicates
        ]
        spec = spec.model_copy(update={"predicates": predicates, "witnesses": found, "impossible": not found})
        try:
            validate_spec(spec, where="intent")
        except ValidationError as exc:
            return Validation(None, exc.message)
        hidden = spec.to_json()
        named = list(dict.fromkeys(draft.checks))
        at_checkout = any(c.stage == "checkout" for c in self.checks if c.id in named)
        intent = CustomIntent(
            id=stable_id("intent", self.store_id, draft.opening, hidden),
            persona=draft.persona,
            intent=draft.intent,
            intent_kind=draft.intent_kind.strip(),
            decisive_kind=draft.decisive_kind,
            opening=draft.opening.strip(),
            spec=hidden,
            checks=named,
            needs_replica=needs_replica(hidden, checkout_stage=at_checkout),
        )
        return Validation(intent, None)

    def _names(
        self, draft: IntentDraft, usage: Usage, *, require_checks: bool = True, probe: bool = False
    ) -> str | None:
        exposed = self.brief.exposed_kinds
        used = usage.kinds
        known = {c.id for c in self.checks}
        unknown = sorted(set(draft.checks) - known)
        full = sorted(c for c in dict.fromkeys(draft.checks) if usage.checks.get(c, 0) >= self.quotas.kind_cap)
        checks: list[tuple[bool, str]] = [
            (
                require_checks and bool(known) and not draft.checks,
                f"name the checks the intent exercises, from {', '.join(sorted(known))}",
            ),
            (
                bool(unknown) and not probe,
                f"checks the store's niche has not, or its data does not support: {', '.join(unknown[:3])}",
            ),
            (bool(full), f"quota: checks {', '.join(full[:3])} already have {self.quotas.kind_cap} intents each"),
            (draft.persona not in self.personas.names(), f"unknown persona {draft.persona!r}"),
            (draft.decisive_kind not in KIND_IDS, f"unknown decisive_kind {draft.decisive_kind!r}"),
            (
                draft.decisive_kind not in exposed and not probe,
                f"the store does not expose {draft.decisive_kind}; choose from {', '.join(exposed)}",
            ),
            (
                used.get(draft.decisive_kind, 0) >= self.quotas.kind_cap,
                f"quota: {draft.decisive_kind} already has {self.quotas.kind_cap} intents; use another data kind",
            ),
            (fold(draft.opening) in usage.openings, "the opening repeats one already accepted"),
        ]
        found = next((reason for failed, reason in checks if failed), None)
        if found is not None or probe:
            return found
        return allergen_guard([p.model_dump() for p in draft.predicates], exposed)

    def _spec(self, draft: IntentDraft) -> tuple[HiddenSpec, list[Product]] | str:
        """The draft's spec with every predicate's satisfied_by computed over its candidates, or why it cannot be
        built: a candidate the capture lacks, a predicate that is not mc.spec.v1, a delivery wish the market lacks."""
        missing = sorted(c for c in draft.candidates if c not in self.catalog.products)
        if missing:
            return f"candidates not in the capture: {', '.join(missing[:3])}"
        pool = [self.catalog.products[c] for c in dict.fromkeys(draft.candidates)]
        try:
            predicates = [Predicate(**p.model_dump()) for p in draft.predicates]
        except PydanticValidationError as exc:
            return f"predicate: {exc.errors(include_url=False, include_input=False)[0]['msg']}"
        buyer = self._buyer(draft, predicates)
        if isinstance(buyer, str):
            return buyer
        spec = HiddenSpec(
            intent=draft.intent,
            decisive=Decisive(detail=draft.decisive_kind, field=_decisive_field(predicates)),
            predicates=predicates,
            candidates=[p.id for p in pool],
            impossible=draft.impossible,
            currency=self.brief.currency,
            generator=GENERATOR,
            buyer=buyer,
        )
        return _with_satisfied(spec, pool), pool

    def _store_wide(self, spec: HiddenSpec, pool: Sequence[Product]) -> list[str]:
        """The products outside the candidates that hold the store's best for a superlative ("the cheapest set you
        have"), to be added to them; empty for any other spec. Why: candidates are a sample of the shelf, so the best
        of the sample was the witness, and on 2026-10-16 an agent that found the store's cheaper set had it refused,
        and a set the shelf's noun missed (a "duo") could never count."""
        if not any(p.op in SUPERLATIVES for p in spec.predicates):
            return []
        held = {p.id for p in pool}
        everyone = [p for p in self.catalog.products.values() if p.id not in held and _same_kind(spec, p)]
        if not everyone:
            return []
        best = set(witnesses_of(spec, [*pool, *everyone]))
        owners = {self.catalog.owner[v] for v in best if v in self.catalog.owner}
        return [p.id for p in everyone if p.id in owners]

    def _witnesses(self, draft: IntentDraft, found: Sequence[str]) -> str | None:
        outside = sorted(w for w in draft.witnesses if w not in self.catalog.owner)
        wrong = sorted(w for w in draft.witnesses if w not in found)
        checks: list[tuple[bool, str]] = [
            (draft.impossible and bool(found), f"marked impossible but variants {found[:3]} satisfy every predicate"),
            (
                not draft.impossible and not found,
                "no available variant of the candidates satisfies every predicate; loosen it or mark it impossible",
            ),
            (bool(outside), f"witnesses not in the capture: {', '.join(outside[:3])}"),
            (bool(wrong), f"witnesses {wrong[:3]} do not satisfy the predicates (or are unavailable)"),
        ]
        return next((reason for failed, reason in checks if failed), None)

    def _buyer(self, draft: IntentDraft, predicates: Sequence[Predicate]) -> Any:
        """The spec's buyer block for the intent's persona in the market, with the intent's own delivery wish when
        it names one; None when the persona has no address there; a reason string when the wish cannot stand."""
        market = self.market
        if market is None or self.phrases.buyer is None:
            return None
        rng = seeded_rng("buyer", self.store_id, draft.persona, draft.opening)
        plan = draw_plan(self.personas.get(draft.persona), market, rng, priced=False)
        if plan is None:
            return None
        wish = draft.fulfillment
        if wish is not None and wish.value == "pickup" and "pickup" not in market_options(market):
            return f"the market {market} has no pickup counter"
        if wish is not None:
            plan = replace(
                plan,
                preference=wish.value,
                preference_source=wish.source,
                deadline_days=wish.deadline_days,
                scope="items",
            )
        else:
            source: Source = "on_ask" if plan.preference != "standard" else "default"
            plan = replace(plan, preference_source=source, deadline_days=None, scope="items")
        budget = next((p for p in predicates if p.op == "max_price"), None)
        return buyer_block(plan, self.phrases, scope="items", budget_source=None if budget is None else budget.source)


def with_provenance(
    intent: CustomIntent,
    store_id: str,
    *,
    chosen_by: ChosenBy,
    attributes: Sequence[DecisionAttribute] = (),
    model_fields: Sequence[str] = (),
    decisive_field: str | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
    generator: str | None = None,
) -> CustomIntent:
    """The intent with who chose each need (`chosen_by` on every predicate, `model` on `model_fields` whatever
    `chosen_by` says), why its attributes were chosen, and its id recomputed from the new spec."""
    spec = HiddenSpec.from_mapping(intent.spec)
    extra = aliases or {}
    predicates = [
        p.model_copy(
            update={
                "chosen_by": "model" if p.field in model_fields else chosen_by,
                "aliases": list(extra.get(p.field, p.aliases)),
            }
        )
        for p in spec.predicates
    ]
    update: dict[str, Any] = {"predicates": predicates, "decision_attributes": list(attributes)}
    if decisive_field is not None:
        update["decisive"] = spec.decisive.model_copy(update={"field": decisive_field})
    if generator is not None:
        update["generator"] = generator
    marked = spec.model_copy(update=update)
    validate_spec(marked, where="intent")
    hidden = marked.to_json()
    return intent.model_copy(
        update={
            "id": stable_id("intent", store_id, intent.opening, hidden),
            "spec": hidden,
            "chosen_by": "model" if model_fields or chosen_by == "model" else "rules",
            "needs_replica": needs_replica(hidden, checkout_stage=intent.needs_replica),
        }
    )


def validate_draft(
    draft: IntentDraft,
    *,
    store_id: str,
    catalog: Catalog,
    brief: StoreBrief,
    personas: Personas,
    phrases: Phrases,
    market: str | None,
    quotas: Quotas,
    used: Mapping[str, int],
    seen_openings: set[str],
    checks: Sequence[Check] = (),
    used_checks: Mapping[str, int] | None = None,
) -> Validation:
    """One draft through a Checker built for it; the writer keeps one Checker per store."""
    checker = Checker(store_id, catalog, brief, personas, phrases, market, quotas, tuple(checks))
    usage = Usage(kinds=dict(used), checks=dict(used_checks or {}), openings=set(seen_openings))
    return checker.validate(draft, usage)


def _decisive_field(predicates: Sequence[Predicate]) -> str | None:
    """The field the decisive data decides: the first predicate that is not identity, quantity or budget."""
    plain = {"product_type", "product_name", "quantity", "budget"}
    return next((p.field for p in predicates if p.field not in plain), None)


def _with_satisfied(spec: HiddenSpec, pool: Sequence[Product]) -> HiddenSpec:
    """Each predicate's satisfied_by from the pool, as the generator computes it; cart ops stay None."""
    from carttrace.agents.tasks import holds_in, line_quantity

    quantity = line_quantity(spec.predicates)
    checked = []
    for predicate in spec.predicates:
        if predicate.op in CART_OPS:
            checked.append(predicate.model_copy(update={"satisfied_by": None}))
            continue
        ids = [v.id for p in pool for v in p.variants if holds_in(spec, predicate, p, v, quantity) is True]
        checked.append(predicate.model_copy(update={"satisfied_by": ids}))
    return spec.model_copy(update={"predicates": checked})


def stated_words(spec: HiddenSpec) -> tuple[list[str], list[str]]:
    """The words a stated predicate puts in the opening, and the words a kept-back one must keep out of it."""
    keep: list[str] = []
    hide: list[str] = []
    for predicate in spec.predicates:
        words = value_words(predicate, spec.currency)
        value = predicate.value
        if predicate.op == "count" and isinstance(value, int) and value > 1:
            words = [str(value)]
        elif isinstance(value, Mapping):
            words += [str(value[k]) for k in ("unit", "nutrient") if isinstance(value.get(k), str)]
        (keep if predicate.source == "stated" else hide).extend(words)
    buyer = spec.buyer
    if buyer is not None:
        for name in ("shipping_address", "email", "phone"):
            found = buyer.field(name)
            if isinstance(found, BuyerAddress):
                hide.append(str(found.value.lines[0]))
            elif isinstance(found, BuyerContact):
                hide.append(found.value)
        if buyer.fulfillment is not None and buyer.fulfillment.source == "stated" and buyer.fulfillment.deadline_days:
            keep.append(str(buyer.fulfillment.deadline_days))
    lowered = {k.lower() for k in keep}
    return list(dict.fromkeys(keep)), [h for h in dict.fromkeys(hide) if len(h) >= 2 and h.lower() not in lowered]


def check_opening(opening: str, spec: HiddenSpec, wish: DraftFulfillment | None) -> str | None:
    """Why an opening cannot stand: a stated value missing, a kept-back value present, a digit no stated value
    carries, talk of paying, a stated delivery wish not in it, or too long."""
    text = opening.strip()
    lowered = text.lower()
    keep, hide = stated_words(spec)
    missing = [k for k in keep if k.lower() not in lowered]
    leaked = [h for h in hide if h.lower() in lowered]
    allowed = {d for k in keep for d in _DIGITS.findall(k)}
    foreign = sorted(set(_DIGITS.findall(text)) - allowed)
    words: tuple[str, ...] = ()
    if wish is not None and wish.source == "stated":
        words = EXPRESS_WORDS if wish.value == "express" else PICKUP_WORDS if wish.value == "pickup" else ()
    checks: list[tuple[bool, str]] = [
        (bool(missing), f"stated values missing from the opening: {', '.join(repr(m) for m in missing[:3])}"),
        (
            bool(leaked),
            f"the opening mentions values the shopper keeps for asking: {', '.join(repr(f) for f in leaked[:3])}",
        ),
        (
            bool(foreign),
            f"digits in the opening no stated value carries: {', '.join(foreign[:3])}; state them or drop them",
        ),
        (_PAYMENT.search(text) is not None, "the opening talks of checkout or payment"),
        (len(text) > MAX_CHARS or len(_SENTENCE.findall(text)) > MAX_SENTENCES, "the opening is too long"),
        (
            bool(words) and not any(w in lowered for w in words),
            f"a stated {wish.value if wish else ''} delivery must be in the opening ({', '.join(words[:3])}, ...)",
        ),
    ]
    return next((reason for failed, reason in checks if failed), None)


# -- the prompt -----------------------------------------------------------------------------------------------------


OPS_DOC: Final = """Predicate ops (mc.spec.v1), each checked by code against the catalog:
- kind: the product is this kind of thing; value is the product's `kind` from the brief (or a word of its title/type).
- is: the named product; value is its exact title. one_of: one of the named products; value is a list of titles.
- all_of: a bundle, one line per named product; value is [{"title", "product_id", "variants": [variant ids]}]; add a
  `lines` predicate (field "lines", value = item count) and a `quantity` count predicate with the same value.
- brand: the product's brand equals value.
- eq: a variant option equals value; set "option" to the store's option name (e.g. "Flavor", "Box").
- mentions: the product's title, tags, option values or description mention value (a phrase, e.g. "grass-fed",
  "FedEx 2-day", "Kosher"); use it for ingredients, dietary claims, certifications, origin, delivery and shipping text.
- max_price: unit price times quantity is at most value {"amount": minor units, "currency": "USD"}; field "budget".
- count: the cart line's quantity is value (an integer); field "quantity"; one per intent.
- excludes: the ingredient list names none of the allergens in value (a list); field "allergens".
- safe_for: the warnings say nothing against the conditions in value (a list); field "conditions".
- fits: the size chart maps value {"measure", "amount", "unit"} to the variant's size; field "measurements".
- cheapest | cheapest_per_unit {"unit": "100 g"} | most {"nutrient": "protein"} | least {"nutrient": "sugar"}: the best
  of the variants that satisfy everything else; at most one per intent; field "price", "unit_price" or "nutrition".
- checkout: a wish settled at checkout, never on a variant (a gift message, who it is for); field "gift_message" etc.
- policy: a question the store's policies answer; value {"topic": "returns"} or {"topic": "shipping", "market": "GB",
  "country": "the UK"}; field "policy".
- note: a fact that only qualifies another predicate; never checked.
Every predicate has "source": "stated" (in the opening, word for word), "on_ask" (said only when the agent asks about
that field; never in the opening) or "default" (never said; the plain assumption, e.g. one unit). Give each a "say":
how the shopper words it when asked, a clause without a final stop. One predicate per field; field names are lowercase
identifiers (product_type, product_name, budget, quantity, unit_price, allergens, flavor, pack_size, policy).
Prices in the opening are written exactly as the stated budget's words ($12 for 1200, $12.99 for 1299; never $12.00).
Each intent names "candidates": the product ids (from the brief) a shopper asking this would look at, and may name
"witnesses": variant ids you believe satisfy everything (optional; code recomputes them and checks yours).
An "impossible": true intent is one no available variant satisfies (a sold-out item, a budget under the cheapest, an
allergen every candidate contains); refusing it is the right outcome. Make about one in ten impossible.
"fulfillment" is optional: {"value": "standard"|"express"|"pickup", "deadline_days": n|null, "source": ...}; a stated
express or pickup wish must be in the opening in those words."""

PROMPT: Final = """You write shopping requests ("intents") for a benchmark that measures whether an online store
exposes the data a shopping agent needs. Each intent is one shopper's opening message plus a hidden spec that code
checks against the store's own catalog. Write intents for THIS store only, from what its brief shows it exposes.

Rules:
- Each intent turns on exactly one decisive data kind of the brief's `exposes` list (its "decisive_kind"), and its
  predicates must need that data to be settled (a unit-price intent needs pack sizes, a dietary intent needs the claim
  in the text, an availability intent needs a sold-out variant, a delivery intent needs the delivery text).
- Spread them: no data kind may take more than {kind_cap} of the {n} intents; the kinds that still need intents are
  listed below with their remaining allowance. Vary the intent kinds too: the ten base intents (replenish, specify,
  compare, discover, express, gift, bundle, budget_max, substitute, policy) are a start; propose your own free-form
  "intent_kind" in a few words (e.g. "find a gift under X with express delivery", "compare price per unit across pack
  sizes", "check if this ships to Z and the return window", "replace a part compatible with Y").
- Every predicate value must be checkable from the brief's products: name product ids and variant ids from the brief
  only, quote option values and phrases exactly as the brief shows them.
- Stated values appear word for word in the opening; on_ask and default values never do; use no digits in the opening
  that are not a stated value. Prices are written as $12.99 for 1299 minor units.
- The opening is first person, in the persona's own voice and situation, one to four sentences, natural, no bullet
  points, never about checkout, cards or paying: the request ends at wanting the item or the answer.
- Use several personas, each in character (their constraints are real: an allergen persona avoids its allergens).
- The niche checklist below is the source of task kinds: vary each check's "Shopper asks" for THIS store, with its
  own products, options and facts, and name in "checks" the check ids the intent exercises (at least one). Every
  Critical and High check whose data the store exposes, and the shared stock, price, delivered total, discounts and
  returns checks, must get an intent; no check takes more than {kind_cap}.

{ops}

The store brief (JSON):
{brief}

The category checklist (what this category's stores are expected to expose):
{checklist}

The niche checklist (id [severity, test, the data it needs]: what we check; Shopper asks):
{checks}
Checks still needing an intent: {needed_checks}

Personas (name: who they are; style; constraints):
{personas}

Decisive data kinds still open, with how many more intents each may take: {open_kinds}
Kinds that have no intent yet and should get one first: {needed_kinds}
{feedback}
Write {ask} intents now. Answer with one JSON object and nothing else:
{{"intents": [{{"persona": "...", "intent": "specify", "intent_kind": "...", "decisive_kind": "...", "opening": "...",
"predicates": [{{"field": "...", "op": "...", "value": ..., "source": "stated", "say": "...", "option": null}}],
"candidates": ["<product id>"], "witnesses": ["<variant id>"], "impossible": false, "fulfillment": null,
"checks": ["<check id>"]}}]}}
"""


def build_prompt(
    brief: StoreBrief,
    *,
    checklist: Sequence[Mapping[str, str]],
    personas: Personas,
    quotas: Quotas,
    used: Mapping[str, int],
    ask: int,
    rejected: Sequence[Rejected] = (),
    checks: Sequence[Check] = (),
    used_checks: Mapping[str, int] | None = None,
) -> str:
    taken = used_checks or {}
    rows = [
        f"- {c.id} [{c.severity}, {c.test}, needs {' or '.join(c.kinds) or 'no product data'}]: {c.check}; "
        f'Shopper asks: "{c.shopper_asks}"'
        for c in checks
    ]
    needed_checks = [c for c in must_cover(checks) if taken.get(c, 0) == 0]
    open_kinds = {k: quotas.kind_cap - used.get(k, 0) for k in brief.exposed_kinds if used.get(k, 0) < quotas.kind_cap}
    needed = [k for k in brief.exposed_kinds if used.get(k, 0) == 0 and k not in ALWAYS_KINDS]
    if len([k for k in used if used[k] > 0]) >= quotas.min_kinds:
        needed = []
    lines = [f"- {h.get('id', '')}: {h.get('statement', '')}" for h in checklist] or ["- (none)"]
    people = [_persona_line(name, personas.get(name)) for name in personas.names()]
    feedback = ""
    if rejected:
        shown = "\n".join(f"- {r.reason} (opening: {r.opening[:90]!r})" for r in rejected[:12])
        feedback = f"\nThe last round's rejected intents, with why. Do not repeat these mistakes:\n{shown}\n"
    body = {k: v for k, v in brief.to_json().items() if k not in ("schema", "id")}
    return PROMPT.format(
        kind_cap=quotas.kind_cap,
        n=quotas.n,
        ops=OPS_DOC,
        brief=dumps(body),
        checklist="\n".join(lines),
        personas="\n".join(people),
        open_kinds=dumps(open_kinds),
        needed_kinds=", ".join(needed[:10]) or "(the minimum of kinds is met)",
        checks="\n".join(rows) or "- (no P1 niche covers this category; name no checks)",
        needed_checks=", ".join(needed_checks) or "(every required check is covered)",
        feedback=feedback,
        ask=ask,
    )


def _persona_line(name: str, persona: Any) -> str:
    style = persona.style
    needs = persona.constraints.model_dump(exclude_defaults=True)
    line = (
        f"- {name}: {persona.display_name or name}, {persona.description.rstrip('.')}; {style.budget} budget, "
        f"{style.decisiveness}, {style.disclosure}, prefers {style.delivery} delivery"
    )
    return f"{line}; constraints {dumps(needs)}" if needs else line


def parse_drafts(text: str) -> tuple[list[IntentDraft], list[Rejected]]:
    """The intents in the model's reply: the drafts that parse, and a rejection for each that does not."""
    stripped = _FENCE.sub("", text.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        return [], [Rejected(reason="the reply holds no JSON object")]
    try:
        data = json.loads(stripped[start : end + 1])
    except ValueError as exc:
        return [], [Rejected(reason=f"the reply is not JSON: {str(exc)[:80]}")]
    items = data.get("intents") if isinstance(data, Mapping) else data
    if not isinstance(items, list):
        return [], [Rejected(reason='the reply has no "intents" list')]
    drafts: list[IntentDraft] = []
    rejected: list[Rejected] = []
    for item in items:
        if not isinstance(item, Mapping):
            rejected.append(Rejected(reason="an intent is not an object"))
            continue
        try:
            drafts.append(IntentDraft.model_validate(dict(item)))
        except PydanticValidationError as exc:
            first = exc.errors(include_url=False, include_input=False)[0]
            where = ".".join(str(p) for p in first["loc"]) or "intent"
            rejected.append(
                Rejected(
                    reason=f"{where}: {first['msg']}",
                    intent_kind=str(item.get("intent_kind", ""))[:120],
                    decisive_kind=str(item.get("decisive_kind", ""))[:40],
                    opening=str(item.get("opening", ""))[:200],
                )
            )
    return drafts, rejected


# -- the writer -------------------------------------------------------------------------------------------------------


def write_intents(
    brief: StoreBrief,
    fixtures: Sequence[Mapping[str, Any]],
    *,
    store_id: str,
    checklist: Sequence[Mapping[str, str]],
    runner: VoiceRunner,
    model: str = DEFAULT_MODEL,
    n: int = DEFAULT_N,
    batch: int = DEFAULT_BATCH,
    cap_usd: float | None = None,
    kind_cap_share: float = KIND_CAP_SHARE,
    min_kinds: int = MIN_KINDS,
    market: str | None = None,
    personas: Personas | None = None,
    phrases: Phrases | None = None,
    log: Path | None = None,
    clock: Clock | None = None,
    timeout_s: float = WRITE_TIMEOUT_S,
    checklists: Checklists | None = None,
) -> IntentSet:
    """`n` validated intents for the store, asked of the model in rounds of `batch`, each round told the kinds still
    open and the last round's rejections. Stops at `n`, at `cap_usd` of known cost, after ceil(n / batch) + 4 rounds,
    or after two rounds that gave nothing; what was accepted by then is the set, with a note. Raises ValidationError
    when nothing was accepted at all."""
    if n < 1 or batch < 1:
        raise ValidationError("n and batch are at least 1")
    quotas = Quotas.for_(n, len(brief.exposed_kinds), share=kind_cap_share, min_kinds=min_kinds)
    checker = Checker(
        store_id,
        Catalog.of(fixtures, currency=brief.currency),
        brief,
        personas or load_personas(),
        phrases or load_phrases(),
        market,
        quotas,
        tuple(agent_checks(brief, checklists or load_checklists())),
    )
    writer = _Writer(
        checker=checker,
        checklist=list(checklist),
        runner=runner,
        model=model,
        batch=batch,
        cap_usd=cap_usd,
        log=log,
        clock=clock or SystemClock(),
        timeout_s=timeout_s,
    )
    writer.run()
    return writer.result()


@dataclass(slots=True)
class _Writer:
    """The rounds of one store's writing and what they accumulate."""

    checker: Checker
    checklist: list[Mapping[str, str]]
    runner: VoiceRunner
    model: str
    batch: int
    cap_usd: float | None
    log: Path | None
    clock: Clock
    timeout_s: float
    accepted: list[CustomIntent] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    last_rejected: list[Rejected] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    intent_kinds: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    served: str = ""
    spent: float = 0.0
    calls: int = 0
    idle: int = 0

    @property
    def n(self) -> int:
        return self.checker.quotas.n

    def run(self) -> None:
        max_rounds = math.ceil(self.n / self.batch) + MAX_ROUNDS_EXTRA
        self.served = self.model
        for round_index in range(max_rounds):
            stop = self._stop()
            if stop is not None:
                if stop:
                    self.notes.append(stop)
                return
            self._round(round_index)
        self.notes.append(f"stopped after {max_rounds} rounds")

    def _stop(self) -> str | None:
        """Why the rounds end before the next one: enough intents (an empty reason), the cost cap, or two idle
        rounds; None to go on."""
        if len(self.accepted) >= self.n:
            return ""
        if self.cap_usd is not None and self.spent >= self.cap_usd:
            return f"stopped at the cost cap of {self.cap_usd:.2f} USD after {self.calls} calls"
        if self.idle >= 2:
            return "stopped after two rounds that gave no intent"
        return None

    def _round(self, round_index: int) -> None:
        ask = min(self.batch, self.n - len(self.accepted))
        prompt = build_prompt(
            self.checker.brief,
            checklist=self.checklist,
            personas=self.checker.personas,
            quotas=self.checker.quotas,
            used=self.usage.kinds,
            ask=ask,
            rejected=self.last_rejected,
            checks=self.checker.checks,
            used_checks=self.usage.checks,
        )
        record = {"round": round_index, "asked": ask, "accepted": 0, "rejected": 0, "cost_usd": None, "error": ""}
        try:
            reply = self.runner(prompt, model=self.model, timeout_s=self.timeout_s)
        except MCError as exc:
            self.notes.append(f"round {round_index}: the model call failed: {exc.message}")
            self.idle += 1
            self._log({**record, "error": exc.message})
            return
        self.calls += 1
        self.spent += float(reply.cost_usd or 0.0)
        self.served = reply.model or self.model
        drafts, broken = parse_drafts(reply.text)
        got = sum(self._take(draft, broken) for draft in drafts)
        self.rejected += broken
        self.last_rejected = broken
        self.idle = self.idle + 1 if got == 0 else 0
        self._log({**record, "accepted": got, "rejected": len(broken), "cost_usd": reply.cost_usd})

    def _take(self, draft: IntentDraft, rejected: list[Rejected]) -> int:
        """One draft validated and kept (1), or added to `rejected` with its reason (0)."""
        if len(self.accepted) >= self.n:
            return 0
        outcome = self.checker.validate(draft, self.usage)
        if outcome.intent is None:
            rejected.append(
                Rejected(
                    reason=outcome.reason or "rejected",
                    intent_kind=draft.intent_kind[:120],
                    decisive_kind=draft.decisive_kind[:40],
                    opening=draft.opening[:200],
                )
            )
            return 0
        intent = with_provenance(outcome.intent, self.checker.store_id, chosen_by="model")
        self.accepted.append(intent)
        self.usage.take(intent)
        key = fold(intent.intent_kind)
        self.intent_kinds[key] = self.intent_kinds.get(key, 0) + 1
        return 1

    def _log(self, record: Mapping[str, Any]) -> None:
        if self.log is None:
            return
        head = {"at": iso(self.clock.now()), "store_id": self.checker.store_id, "model": self.served}
        append_jsonl(self.log, {**head, **record, "spent_usd": round(self.spent, 6)})

    def result(self) -> IntentSet:
        store_id, quotas, brief = self.checker.store_id, self.checker.quotas, self.checker.brief
        if not self.accepted:
            raise ValidationError(
                f"no intent for {store_id} passed validation in {self.calls} calls",
                hint="read the rejections in the log, or try another model",
            )
        if len(self.accepted) < self.n:
            self.notes.append(f"{len(self.accepted)} of {self.n} intents written")
        used = self.usage.kinds
        uncovered = [c for c in self.checker.must if self.usage.checks.get(c, 0) == 0]
        met = len(used) >= quotas.min_kinds and all(c <= quotas.kind_cap for c in used.values()) and not uncovered
        if not met:
            self.notes.append(
                f"quotas not met: {len(used)} kinds of {quotas.min_kinds} needed; uncovered checks: "
                f"{', '.join(uncovered) or 'none'}"
            )
        draft_set = IntentSet(
            store_id=store_id,
            category=brief.category,
            source=brief.source,
            brief_id=brief.id,
            model=self.served,
            written_at=iso(self.clock.now()),
            quotas=quotas,
            kinds=dict(sorted(used.items(), key=lambda kv: (-kv[1], kv[0]))),
            intent_kinds=dict(sorted(self.intent_kinds.items(), key=lambda kv: (-kv[1], kv[0]))),
            checks=dict(sorted(self.usage.checks.items(), key=lambda kv: (-kv[1], kv[0]))),
            uncovered=uncovered,
            intents=self.accepted,
            rejected=self.rejected,
            calls=self.calls,
            cost_usd=round(self.spent, 6),
            quotas_met=met,
            notes=self.notes,
        )
        return draft_set.model_copy(update={"id": set_id(draft_set)})


def check_quotas(intent_set: IntentSet) -> list[str]:
    """What the set breaks of its own quotas: a kind over its cap, or too few kinds."""
    out = []
    for kind, count in intent_set.kinds.items():
        if count > intent_set.quotas.kind_cap:
            out.append(f"{kind} has {count} intents, above the cap of {intent_set.quotas.kind_cap}")
    if len(intent_set.kinds) < intent_set.quotas.min_kinds:
        out.append(f"{len(intent_set.kinds)} data kinds, fewer than the {intent_set.quotas.min_kinds} needed")
    for check, count in intent_set.checks.items():
        if count > intent_set.quotas.kind_cap:
            out.append(f"check {check} has {count} intents, above the cap of {intent_set.quotas.kind_cap}")
    if intent_set.uncovered:
        out.append(f"checks with no intent: {', '.join(intent_set.uncovered)}")
    return out


# -- tasks from a set -------------------------------------------------------------------------------------------------


def tasks_from_intents(intent_set: IntentSet, *, held: Sequence[str] | None = None) -> list[TaskRef]:
    """The set's intents as tasks, in the set's order; with `held` (product and variant ids a world holds), only the
    intents whose candidates and witnesses the world holds. Deterministic: the ids hash the content."""
    out = []
    keep = set(held) if held is not None else None
    for intent in intent_set.intents:
        spec = intent.spec
        named = [*spec.get("candidates", []), *spec.get("witnesses", [])]
        if keep is not None and not all(entry in keep for entry in named):
            continue
        out.append(
            TaskRef(
                task_id=stable_id(
                    "task", intent_set.store_id, intent_set.category, intent.intent, intent.opening, spec
                ),
                intent=intent.intent,
                category=intent_set.category,
                opening=intent.opening,
                fixture=list(spec.get("candidates", [])),
                hidden_spec=spec,
                template=f"custom@{intent_set.id}",
                intent_kind=intent.intent_kind,
                decisive_kind=intent.decisive_kind,
                needs_replica=intent.needs_replica,
            )
        )
    return out


def _same_kind(spec: HiddenSpec, product: Product) -> bool:
    """A product outside the candidates counts for a store-wide best only when its own title names the kind asked
    for: a category collection ("Body Cream") also holds a body scrub, which is no "cheapest body cream"."""
    kind = next((p for p in spec.predicates if p.op == "kind" and isinstance(p.value, str)), None)
    noun = noun_from(str(kind.value)) if kind is not None else None
    return kind is None or (noun is not None and names_product(noun, product.title))


def rechecked(task: TaskRef, products: Sequence[Product]) -> TaskRef | None:
    """A written intent's task checked again against the world it runs on, with this code's checks: its witnesses
    and satisfied_by computed afresh (a superlative over the whole store), or None when it no longer stands: it asks
    for a store label as a product kind ("the cheapest wholesale"), nothing it holds now satisfies it, or its opening
    contradicts its spec (carttrace.agents.consistency, D104). An impossible task is otherwise kept as written.

    Why (2026-10-16). Intent sets are written once and drawn every round, so a task the writer got wrong was run 15
    times over: of 942 overnight rows built to pay, 15 tasks with an order rate of a fifth or less held 230 rows. A
    row built to pay must be satisfiable by what the world serves."""
    spec = HiddenSpec.from_task(task)
    by_id = {p.id: p for p in products}
    if contradictions(spec, task.opening, [by_id[c] for c in spec.candidates if c in by_id]):
        return None  # its opening says what its spec does not (D104): run, it is judged against words it broke
    if spec.impossible:
        return task
    kind = next((p for p in spec.predicates if p.op == "kind" and isinstance(p.value, str)), None)
    if kind is not None:
        noun = noun_from(str(kind.value))
        if noun is None or not any(names_product(noun, p.title) for p in products):
            return None
    pool = [by_id[c] for c in dict.fromkeys(spec.candidates) if c in by_id]
    if any(p.op in SUPERLATIVES for p in spec.predicates):
        best = set(witnesses_of(spec, [*pool, *(p for p in products if _same_kind(spec, p))]))
        held = {p.id for p in pool}
        pool += [p for p in products if p.id not in held and any(v.id in best for v in p.variants)]
    checked = _with_satisfied(spec, pool)
    found = witnesses_of(checked, pool)
    if not found:
        return None
    candidates = [p.id for p in pool]
    if found == spec.witnesses and candidates == spec.candidates:
        same = all(
            a.satisfied_by == b.satisfied_by or a.op in SUPERLATIVES
            for a, b in zip(spec.predicates, checked.predicates, strict=True)
        )
        if same:
            return task
    predicates = [
        p.model_copy(update={"satisfied_by": found}) if p.op in SUPERLATIVES else p for p in checked.predicates
    ]
    fresh = checked.model_copy(update={"predicates": predicates, "witnesses": found, "candidates": candidates})
    hidden = fresh.to_json()
    return task.model_copy(
        update={"task_id": stable_id("task", task.task_id, hidden), "hidden_spec": hidden, "fixture": candidates}
    )


def custom_task_writer(var: Path, *, intent_set: Path | None = None) -> Any:
    """A generate_tasks in the collector's shape (store, category, fixtures, seed=, persona=) that draws from the
    store's custom intents (`intent_set`, else the store's newest under var/agent/intents) instead of the templates.
    The seed changes nothing: the set is what it is. Refuses, as generate_tasks does, when the store has no set or
    the world holds none of its products."""

    def write(
        store: Any, category: Any, fixtures: Sequence[Mapping[str, Any]], *, seed: int, persona: str | None = None
    ) -> list[TaskRef]:
        store_id = store if isinstance(store, str) else getattr(store, "store_id", None) or store.id
        path = intent_set or latest_intent_set(var, str(store_id))
        if path is None:
            raise ValidationError(
                f"no custom intents for {store_id}", hint=f"write them with `carttrace agents intents write {store_id}`"
            )
        found = load_intent_set(path)
        if found.store_id != store_id:
            raise ValidationError(f"{path} holds intents for {found.store_id}, not {store_id}")
        products = read_products(fixtures)
        held = [p.id for p in products] + [v.id for p in products for v in p.variants]
        tasks = [t for t in (rechecked(t, products) for t in tasks_from_intents(found, held=held)) if t is not None]
        if persona is not None and any(i.persona == persona for i in found.intents):
            tasks = [t for t in tasks if (t.hidden_spec.get("buyer") or {}).get("persona") == persona]
        if not tasks:
            raise ValidationError(
                f"the world holds none of the products the custom intents of {store_id} name",
                hint="the intents were written from another capture; write them again",
            )
        return tasks

    return write


__all__ = [
    "DEFAULT_BATCH",
    "DEFAULT_MODEL",
    "DEFAULT_N",
    "GENERATOR",
    "INTENTS_DIR",
    "INTENT_SET_SCHEMA",
    "KIND_CAP_SHARE",
    "MIN_KINDS",
    "Catalog",
    "CheckCoverage",
    "Checker",
    "CustomIntent",
    "IntentDraft",
    "IntentSet",
    "Quotas",
    "Rejected",
    "Usage",
    "VoiceReply",
    "build_prompt",
    "check_opening",
    "check_quotas",
    "custom_task_writer",
    "intents_log",
    "intents_path",
    "latest_intent_set",
    "load_intent_set",
    "needs_replica",
    "parse_drafts",
    "save_intent_set",
    "tasks_from_intents",
    "validate_draft",
    "with_provenance",
    "write_intents",
]

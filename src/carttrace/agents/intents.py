"""The intent creator: generator 3's intents and the variety a persona brings to a task, as data and deterministic code.

Why. The old lab had intent writer agents and many shoppers; this repository had five intents and one anonymous buyer,
so every task read alike. The project asked for buyers with different personas, hidden specs, addresses, names and kinds
of intent (D20). This module holds what makes tasks differ, so carttrace.agents.tasks can stay the one generator:

- the words of the five new intents (data/agent/intents.yaml): gift (for someone else, with a deadline and a message),
  bundle (two or three items that go together), budget_max (the most of one product a budget buys), substitute (the
  named product is out of stock on the replica, so the closest substitute or a refusal is right) and policy (a question
  the store's policies answer before the purchase), each with per-category openings where one fits better;
- `Variety`: what a persona's style (carttrace.agents.personas) changes in a task: how much is stated up front, how tight
  the budget is, which delivery it leans to, whether it buys two, whether it forgets one detail;
- the proposals a persona's constraints make: their allergen as the `excludes` predicate when the shelf's labels make
  it decide, their condition as `safe_for`, their measurement as `fits`, their diet as a `dietary` mention;
- `task_writer`: a generate_tasks with the plan's intents fixed, in the shape the collector's GenerateTasks port takes.

The model voice pass that rewrites an opening in the persona's words is carttrace.agents.intents_voice; nothing here calls a
model. Deterministic: every draw comes from the Random the generator hands in.
"""

from __future__ import annotations

import functools
import string
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from carttrace.agents.details.allergens import ALLERGENS as ALLERGEN_WORDS
from carttrace.agents.details.allergens import status as allergen_status
from carttrace.agents.details.base import Detail, Proposal
from carttrace.agents.details.size_chart import SizeChart, read_chart, to_mm, variant_size
from carttrace.agents.details.warnings import CONDITIONS as CONDITION_WORDS
from carttrace.agents.details.warnings import status as warning_status
from carttrace.agents.niche_gate import NicheGate
from carttrace.agents.personas import PersonaStyle
from carttrace.agents.products import Product, Variant
from carttrace.agents.spec import ALL_INTENTS, NEW_INTENTS, FulfillmentKind, IntentType
from carttrace.agents.text import has_phrase
from carttrace.agents.wallet import Persona
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.sandbox.catalogs.kinds import Kinds

INTENTS_SCHEMA: Final = "mc.intents.v1"
DEFAULT_PATH: Final = Path(__file__).resolve().parents[3] / "data" / "agent" / "intents.yaml"
"""data/agent/intents.yaml in this repository."""
NewIntent = Literal["gift", "bundle", "budget_max", "substitute", "policy"]
PolicyTopic = Literal["returns", "shipping"]
POLICY_TOPICS: Final[tuple[PolicyTopic, ...]] = ("returns", "shipping")
COMMON_SLOTS: Final = frozenset({"noun", "nouns", "noun_bare", "options", "budget", "decisive", "name"})
OPENING_SLOTS: Final[Mapping[str, frozenset[str]]] = {
    "gift": COMMON_SLOTS | {"recipient", "message"},
    "bundle": frozenset({"titles", "count_words", "budget"}),
    "budget_max": frozenset({"budget", "name"}),
    "substitute": frozenset({"name", "options", "substitute_sentence"}),
    "policy": COMMON_SLOTS | {"question", "clause"},
}
"""The placeholders each intent's openings may use; the loader refuses any other."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GiftWords(_Frozen):
    openings: list[str] = Field(min_length=1)
    plain_openings: list[str] = Field(min_length=1)
    """Openings with no gift message, for a store whose checkout takes none (D110)."""
    recipients: list[str] = Field(min_length=1)
    messages: list[str] = Field(min_length=1)
    say: dict[str, str]
    """recipient, gift and gift_message: how the shopper words each predicate."""


class BundleWords(_Frozen):
    openings: list[str] = Field(min_length=1)
    count_words: dict[str, str]
    """"2" to "two things", "3" to "three things"."""
    say: dict[str, str]
    """bundle, lines and quantity."""


class BudgetMaxWords(_Frozen):
    openings: list[str] = Field(min_length=1)
    say: dict[str, str]
    """quantity: "as many as my budget allows"."""


class SubstituteWords(_Frozen):
    openings: list[str] = Field(min_length=1)
    say: dict[str, str]
    """ok and not_ok (the substitute_ok predicate), prefer and prefer_strict (the product predicate)."""


class PolicyQuestion(_Frozen):
    question: str
    clause: str
    say: str


class PolicyWords(_Frozen):
    openings: list[str] = Field(min_length=1)
    questions: dict[str, PolicyQuestion]
    countries: dict[str, str]
    """Market or country code to how the shopper names it ("the UK")."""


class IntentWords(_Frozen):
    schema_: Literal["mc.intents.v1"] = Field(alias="schema")
    version: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    gift: GiftWords
    bundle: BundleWords
    budget_max: BudgetMaxWords
    substitute: SubstituteWords
    policy: PolicyWords
    categories: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    """Category slug to intent to its own openings, used instead of the generic ones."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    def openings(self, intent: str, category: str) -> list[str]:
        """The openings of `intent` for `category`: the category's own when it has any, else the generic ones."""
        own = self.categories.get(category, {}).get(intent)
        if own:
            return list(own)
        section = getattr(self, intent, None)
        return list(section.openings) if section is not None else []


def load_intent_words(path: Path | None = None) -> IntentWords:
    """The words at `path` (default: data/agent/intents.yaml), parsed once per process and checked."""
    return _load((path or DEFAULT_PATH).resolve())


@functools.cache
def _load(path: Path) -> IntentWords:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NotFoundError(f"no intent words at {path}", hint="restore data/agent/intents.yaml from git") from None
    except yaml.YAMLError as exc:
        raise ValidationError(f"{path.name} is not valid YAML: {exc}") from exc
    try:
        words = IntentWords.model_validate(raw)
    except PydanticValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "file"
        raise ValidationError(f"{path.name}: {where}: {first['msg']}", hint=f"follow {INTENTS_SCHEMA}") from exc
    problems = _problems(words)
    if problems:
        raise ValidationError(f"{path.name}: {'; '.join(sorted(set(problems)))}", hint="fix the template text")
    return words


NEEDED_SAYS: Final[Mapping[str, tuple[str, ...]]] = {
    "gift": ("recipient", "gift", "gift_message"),
    "bundle": ("bundle", "lines", "quantity"),
    "substitute": ("ok", "not_ok", "prefer", "prefer_strict"),
    "budget_max": ("quantity",),
}
"""The `say` entries each intent's words must give."""


def _problems(words: IntentWords) -> list[str]:
    out = _opening_problems(words)
    for intent, keys in NEEDED_SAYS.items():
        says = getattr(words, intent).say
        out += [f"{intent}.say needs {key}" for key in keys if key not in says]
    out += [f"policy.questions needs {topic}" for topic in POLICY_TOPICS if topic not in words.policy.questions]
    if not {"2", "3"} <= set(words.bundle.count_words):
        out.append("bundle.count_words needs 2 and 3")
    return out


def _opening_problems(words: IntentWords) -> list[str]:
    out: list[str] = []
    for slug, table in words.categories.items():
        for named in table:
            if named not in NEW_INTENTS:
                out.append(f"categories.{slug} names {named!r}, which is not one of {list(NEW_INTENTS)}")
    out += _unknown(words.gift.plain_openings, OPENING_SLOTS["gift"] - {"message"}, "gift.plain_openings")
    for intent in NEW_INTENTS:
        allowed = OPENING_SLOTS[intent]
        out += _unknown(words.openings(intent, ""), allowed, intent)
        for slug, table in words.categories.items():
            out += _unknown(table.get(intent, ()), allowed, f"{slug}.{intent}")
    return out


def _unknown(templates: Iterable[str], allowed: Collection[str], where: str) -> list[str]:
    out = []
    for template in templates:
        try:
            names = {n for _, n, _, _ in string.Formatter().parse(template) if n is not None}
        except ValueError as exc:
            out.append(f"{where}: broken placeholder in {template!r}: {exc}")
            continue
        out += [f"{where}: unknown placeholder {{{n}}} in {template!r}" for n in sorted(names - set(allowed))]
    return out


# -- variety from a persona's style -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Variety:
    """What a persona's style changes in task generation. The defaults are the first generator's draws, so a persona
    whose style is the default (or no persona at all) gets the tasks a seed always gave."""

    stated_decisive: float = 0.75
    """Share of possible tasks whose decisive fact is in the opening."""
    option_stated: float = 0.3
    """Below this draw an option the opening can carry is stated."""
    option_asked: float = 0.6
    """Below this draw (and above option_stated) an option is kept for when the agent asks; above, no preference."""
    budget_percent: int = 15
    """How far above the target's price a budget is set."""
    delivery: FulfillmentKind | None = None
    """The delivery the buyer plan leans to; None draws as every task did."""
    two_share: float = 0.0
    """Share of fallback openings that ask for two of the item."""
    forgets: bool = False
    """One on_ask detail is not to hand the first time the agent asks."""
    register: str = ""
    """How the persona speaks, for the voice pass; empty for the first generator's register."""

    @property
    def legacy(self) -> bool:
        return self == LEGACY


LEGACY: Final = Variety()
BUDGET_PERCENT: Final[Mapping[str, int]] = {"tight": 5, "normal": 15, "generous": 40}
REGISTERS: Final[Mapping[str, str]] = {
    "fast": "brief and decided: says what they want and expects it done",
    "deliberate": "thinks aloud, weighs the choice, and asks the assistant to lay out the options before deciding",
}


def variety_for(persona: Persona | None) -> Variety:
    """The variety a persona's style gives; LEGACY for None or for the default style."""
    if persona is None:
        return LEGACY
    style: PersonaStyle = persona.style
    stated, option_stated, option_asked = {
        "volunteers": (1.0, 0.6, 0.85),
        "mixed": (0.75, 0.3, 0.6),
        "asks": (0.3, 0.1, 0.8),
        "forgets": (0.3, 0.1, 0.8),
    }[style.disclosure]
    delivery: FulfillmentKind | None = None if style.delivery == "standard" else style.delivery
    return Variety(
        stated_decisive=stated,
        option_stated=option_stated,
        option_asked=option_asked,
        budget_percent=BUDGET_PERCENT[style.budget],
        delivery=delivery,
        two_share=0.25 if style.budget == "generous" else 0.0,
        forgets=style.disclosure == "forgets",
        register=REGISTERS[style.decisiveness] if style != PersonaStyle() else "",
    )


# -- a persona's constraints as proposals ------------------------------------------------------------------------------


def persona_proposal(
    detail: Detail, persona: Persona | None, target: Product, variant: Variant, shelf: Sequence[Product]
) -> Proposal | None:
    """The persona's own constraint as the decisive detail's proposal, when the shelf's data makes it decide: their
    allergen the target is free of and another product contains, or their condition the target's label is clear of
    and another warns against. None lets the detail propose as it always did."""
    if persona is None:
        return None
    others = [p for p in shelf if p.id != target.id]
    if detail.name == "allergens":
        for allergen in persona.constraints.allergens:
            if (
                allergen in ALLERGEN_WORDS
                and allergen_status(target, allergen) == "free"
                and any(allergen_status(p, allergen) == "contains" for p in others)
            ):
                return _allergen_proposal(allergen)
    if detail.name == "warnings":
        for condition in persona.constraints.conditions:
            if (
                condition in CONDITION_WORDS
                and warning_status(target, condition) == "clear"
                and any(warning_status(p, condition) == "warned" for p in others)
            ):
                return _warning_proposal(condition)
    return None


def persona_impossible(detail: Detail, persona: Persona | None, pool: Sequence[Product]) -> Proposal | None:
    """The persona's constraint as a value no product of the pool meets, when the data shows that."""
    if persona is None or not pool:
        return None
    if detail.name == "allergens":
        for allergen in persona.constraints.allergens:
            if allergen in ALLERGEN_WORDS and all(allergen_status(p, allergen) == "contains" for p in pool):
                return _allergen_proposal(allergen)
    if detail.name == "warnings":
        for condition in persona.constraints.conditions:
            if condition in CONDITION_WORDS and all(warning_status(p, condition) == "warned" for p in pool):
                return _warning_proposal(condition)
    return None


def persona_fit(persona: Persona | None, product: Product) -> tuple[Variant, Proposal] | None:
    """The variant of `product` the persona's own measurement fits by the product's chart, with the measurement as the
    `fits` proposal; None when the persona has no measurement the chart maps to exactly one size the product sells."""
    if persona is None or not persona.constraints.measurements:
        return None
    chart = read_chart(product)
    if chart is None:
        return None
    sizes = {k: v for v in product.variants if v.available and v.price is not None and (k := variant_size(v))}
    column = chart.column_for(set(sizes))
    for measure, text in persona.constraints.measurements.items():
        parsed = _measurement(text)
        if parsed is None or measure not in chart.measures():
            continue
        amount, unit = parsed
        rows = chart.fitting(measure, to_mm(amount, unit))
        if len(rows) != 1:
            continue
        key = _key_at(chart, rows[0], column)
        variant = sizes.get(key or "")
        if variant is not None:
            return variant, _fit_proposal(measure, amount, unit)
    return None


def persona_constrains(detail: Detail, persona: Persona | None) -> bool:
    """Whether the persona's own constraints decide what this detail may propose: an allergen persona only ever
    avoids their allergens, a persona with a condition only ever asks about that condition. False for the size
    chart, which `persona_measures` settles per product."""
    if persona is None:
        return False
    if detail.name == "allergens":
        return bool(persona.constraints.allergens)
    if detail.name == "warnings":
        return bool(persona.constraints.conditions)
    return False


def persona_measures(persona: Persona | None, product: Product) -> bool:
    """Whether the product's chart gives a measure the persona knows about themselves, so only their own measurement
    may decide a size task for it."""
    if persona is None or not persona.constraints.measurements:
        return False
    chart = read_chart(product)
    return chart is not None and any(m in chart.measures() for m in persona.constraints.measurements)


def persona_dietary(persona: Persona | None, product: Product, variant: Variant, fields: Collection[str]) -> str | None:
    """The persona's dietary need the product's own text claims ("vegan"), for a `dietary` mention predicate; None
    when the category has no dietary field, the persona no need, or the product does not say."""
    if persona is None or "dietary" not in fields:
        return None
    text = " ".join((product.title, product.product_type or "", *product.tags, *product.lines))
    return next((need for need in persona.constraints.dietary if has_phrase(text, need)), None)


def _measurement(text: str) -> tuple[Decimal, str] | None:
    amount, _, unit = text.partition(" ")
    try:
        return (Decimal(amount), unit) if unit in ("cm", "mm", "in") else None
    except InvalidOperation:
        return None


def _key_at(chart: SizeChart, row: int, column: int) -> str | None:
    keys = chart.rows[row].keys
    if column < len(keys) and keys[column]:
        return keys[column]
    return next((k for k in keys if k), None)


def _allergen_proposal(allergen: str) -> Proposal:
    from carttrace.agents.details.allergens import _proposal  # the detail's own wording, so a persona's allergen reads alike

    return _proposal(allergen)


def _warning_proposal(condition: str) -> Proposal:
    from carttrace.agents.details.warnings import _proposal

    return _proposal(condition)


def _fit_proposal(measure: str, amount: Decimal, unit: str) -> Proposal:
    from carttrace.agents.details.size_chart import _proposal

    return _proposal(measure, amount, unit)


# -- the generator façade ----------------------------------------------------------------------------------------------


def chosen_intents(names: Sequence[str] | None) -> tuple[IntentType, ...] | None:
    """The intents a plan names, in canonical order: None for the first generator's five, "all" for every one, else
    the names given. Refuses a name that is not an intent."""
    if names is None:
        return None
    wanted = set(names)
    if "all" in wanted:
        return ALL_INTENTS
    unknown = sorted(wanted - set(ALL_INTENTS))
    if unknown:
        raise ValidationError(f"unknown intents {unknown}", hint=f"choose from {', '.join(ALL_INTENTS)} or all")
    chosen = tuple(i for i in ALL_INTENTS if i in wanted)
    if not chosen:
        raise ValidationError("no intents chosen", hint=f"choose from {', '.join(ALL_INTENTS)} or all")
    return chosen


def task_writer(
    *,
    intents: Sequence[str] | None = None,
    per_intent: int = 2,
    niche_gates: Mapping[str, NicheGate] | None = None,
    kinds: Kinds | None = None,
    gift_message: bool = True,
) -> Any:
    """generate_tasks with the plan's intents fixed, in the shape the collector's GenerateTasks port takes (store,
    category, fixtures, seed=, persona=). With `intents` None it is generate_tasks itself. `niche_gates` (category
    slug to its gate, D104) gate each store's tasks by its category's niches, as the open lane writes them; `kinds`
    (D104) writes them only around products of a reviewed kind; `gift_message` False writes gift tasks that ask for
    no gift message (D110), for worlds whose checkout takes none."""
    from carttrace.agents.tasks import generate_tasks

    chosen = chosen_intents(intents)

    def write(
        store: Any, category: Any, fixtures: Sequence[Mapping[str, Any]], *, seed: int, persona: str | None = None
    ) -> Any:
        slug = getattr(category, "slug", None) or (category.get("slug") if isinstance(category, Mapping) else None)
        gate = niche_gates.get(str(slug)) if niche_gates is not None and slug else None
        return generate_tasks(
            store,
            category,
            fixtures,
            seed=seed,
            per_intent=per_intent,
            persona=persona,
            intents=chosen,
            niche_gate=gate,
            kinds=kinds,
            gift_message=gift_message,
        )

    return write

"""The shopper's words as data: data/prompts/shopper-v2.yaml, loaded once and checked.

Why data. Replies and openings are prompts in all but name: the agent reads them, and results depend on their
wording. Keeping them in a versioned file (CLAUDE.md: a prompt change is a code change) lets people who write well
improve them without touching code, and the version goes into every task's `template` field, so a changed reply can
be told apart in the traces. The loader refuses a placeholder the code would not fill, so a typo fails at load time
instead of reaching a shopper's mouth as "{sya}".

Version 2 (`mc.shopper-phrases.v2`) adds the buyer's checkout details and what the shopper refuses to make up, and the
words of compare tasks that name their candidates. A v1 file still loads, without those sections: its shopper answers
no buyer question, and generation writes no buyer block and no named compare task with it.
"""

from __future__ import annotations

import functools
import re
import string
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from carttrace.agents.spec import IntentType
from carttrace.core.errors import NotFoundError, ValidationError

PHRASES_SCHEMA = "mc.shopper-phrases.v2"
SCHEMAS = ("mc.shopper-phrases.v1", "mc.shopper-phrases.v2")
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "prompts" / "shopper-v2.yaml"
"""data/prompts/shopper-v2.yaml in this repository."""

REPLY_SLOTS = frozenset({"say", "Say", "label"})
CONSENT_SLOTS = frozenset({"total", "count", "held", "reason"})
CHOICE_SLOTS = frozenset({"option", "reason"})
OPTION_SLOTS = frozenset({"value", "label"})
FACT_SLOTS = frozenset({"value", "amount", "count", "noun"})
OPENING_SLOTS = frozenset(
    {"noun", "noun_bare", "nouns", "name", "brand", "from_brand", "options", "budget", "budget_sentence", "decisive"}
)
BUYER_SAY_SLOTS = frozenset({"address", "value", "days"})
COMPARE_SLOTS = frozenset({"candidates", "criteria"})
CRITERIA_SLOTS = frozenset({"unit", "nutrient"})
BUYER_WITHHELD = frozenset({"shipping_address", "email", "phone", "fulfillment", "budget_scope"})
BUYER_REFUSED = frozenset({"date_of_birth", "other_address", "recipient_name", "id_number"})
CRITERIA = frozenset({"cheapest", "cheapest_per_unit", "most", "least"})
_TIDY = ((re.compile(r"\s+"), " "), (re.compile(r"\s+([.,?!])"), r"\1"), (re.compile(r"([.?!])\.+"), r"\1"))


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class Replies(_Frozen):
    reveal: str
    """An on_ask fact, said when asked."""
    restate: str
    """A stated fact, asked about again."""
    default: str
    """A field the shopper has no preference on."""
    alias: str
    """A field the shopper cannot answer but a related fact settles (their size, when they know a measurement)."""
    unknown: str
    """A field that is not in the spec at all."""
    other: str
    """A question about something no field covers."""
    nothing_asked: str
    """A message that asks the shopper nothing."""
    forgot: str = "Sorry, I don't have that to hand right now. Could you ask me again in a moment?"
    """An on_ask fact the shopper forgets the first time it is asked (a predicate with `forgets`); the second question
    gets the reveal. A default, so a v1 file loads."""
    pay_go_ahead: str = "Yes, please go ahead. I'll approve the payment when my wallet asks me."
    """A yes or no question to charge the card or place the order, asked outside the wallet step (D108): the shopper
    approves inside the wallet step (D94), so they say to go on; it is not an offer to choose a product from."""


class Variants(_Frozen):
    """Other ways to say a reply line, so no set phrase is something an agent can pattern-match and no reply repeats
    word for word (D108; the 2026-10-16 prompt review found "Like I said, ..." twice in one reply and one refusal sent
    three times). Defaults, so a file without the section still loads."""

    restate: list[str] = Field(
        default_factory=lambda: ["Like I said, {say}.", "As I mentioned, {say}.", "{Say}, as I said."], min_length=1
    )
    """A stated fact, asked about again, in turn; the first is `replies.restate`'s words."""
    nothing_asked: list[str] = Field(
        default_factory=lambda: ["Okay, thanks.", "Thanks.", "Got it, thanks.", "Sounds good."], min_length=1
    )
    """A message that asks nothing, in turn."""
    again: list[str] = Field(
        default_factory=lambda: [
            "To say it again: {text}",
            "Sorry, just to repeat: {text}",
            "As I said before: {text}",
            "Once more: {text}",
        ],
        min_length=1,
    )
    """A reply that would repeat an earlier one word for word, led in so it does not. {text} is the reply."""


class ChoiceReplies(_Frozen):
    """What the shopper says when the agent lays out options or recommends one (D60). {option} names the option as
    the agent wrote it ("option 2, Travel"); {reason} is the need it gives, as a restate or reveal line. The defaults
    are the words of shopper-v2, so a file without the section still loads."""

    picked: str = "I'll go with {option}."
    """An option meets every need its words speak to."""
    confirmed: str = "Yes, {option} sounds right. Please go ahead with it."
    """The recommended option meets every need its words speak to, and no other option offered meets more."""
    none_fits: str = "None of those is quite what I'm after."
    """No option fits, and the need it misses is one the shopper keeps to themselves."""
    none_fits_named: str = "None of those works for me. {reason}"
    """No option fits, and the shopper says which need: one already stated, or one its disclosure allows."""
    declined: str = "I don't think that one is right for me."
    """The recommended option misses a need the shopper keeps to themselves, and no other option fits."""
    declined_named: str = "I don't think that one is right for me. {reason}"
    ask_for: str = "None of those works for me. {reason} Is that possible?"
    """No way of delivery offered is the one the shopper wants: they say which, and ask for it (D97)."""
    any_fits: str = "Any of those is fine with me."
    """An offer of ways to pay: no spec holds a preference, the persona's wallet decides (D97)."""


class ConsentReplies(_Frozen):
    """What the shopper answers when the agent asks to place an order for a stated total (checkout rows, D13)."""

    consented: str
    """The order matches what they asked for and the total is within their budget. {total} is the total approved."""
    over_budget: str
    """The total is more than they want to spend."""
    cart_mismatch: str
    """The order is not what they asked for: another item, quantity or extra lines (the fallback line)."""
    wrong_quantity: str = "I asked for {count}, not {held}, so please don't place the order like that."
    """A cart_mismatch where the units are wrong. {count} the units wanted, {held} the units in the order."""
    wrong_item: str = "That isn't the item I asked for, so please don't place the order.{reason}"
    """A cart_mismatch where a line is not a product they want. {reason} restates what they said, with a leading
    space, or is empty when they said nothing that tells it apart."""
    extra_items: str = "I only want the one item I asked for, so please don't place the order with anything else."
    """A cart_mismatch where the order holds more than one product and they asked for one."""
    repeat_plan: str = "I didn't ask for a repeat delivery, so please don't place the order on a subscription."
    """A cart_mismatch where a line is on a selling plan they did not ask for."""
    fewer_than_budget: str = "That's fewer than my budget buys, so please don't place the order like that."
    """A wrong quantity on a task whose count is what a budget buys (budget_max, D108): the shopper never named a
    count, so "I asked for 2" would claim words they never said."""
    more_than_budget: str = "That's more than my budget buys, so please don't place the order like that."
    """The same, with more units than the budget buys."""
    go_ahead: str = "That's what I want. Please go ahead and place the order."
    """The agent stopped at a ready cart on a checkout row, and the cart holds what the shopper asked for: they ask
    for the order. A default, so a file without it still loads."""
    total_mismatch: str
    """The total the agent stated is not the one the order shows."""
    currency_mismatch: str
    """The total is in a currency the shopper cannot compare with their budget."""


class Label(_Frozen):
    label: str
    none: str | None = None
    """What the shopper says when asked about this field and the spec has nothing on it."""


class Phrase(_Frozen):
    say: str
    clause: str = ""


class BuyerSay(_Frozen):
    """How the shopper words each checkout detail, as a clause."""

    shipping_address: str
    email: str
    phone: str
    standard: str
    express: str
    pickup: str
    deadline: str
    budget_total: str


class BuyerPhrases(_Frozen):
    say: BuyerSay
    withheld: dict[str, str]
    """A buyer field the shopper never gives (a default), to what they say when asked for it."""
    refuse: dict[str, str]
    """A detail the shopper does not have (a date of birth, a second address), to their plain refusal."""
    open_fulfillment: str = "I don't mind how it reaches me. Whatever you suggest is fine."
    say_variants: dict[str, list[str]] = Field(default_factory=dict)
    """Other ways to say a delivery wish (standard, express, pickup) or its deadline (D118): a task draws one with its
    own seed (carttrace.agents.buyer `buyer_block`), so openings do not all say "I'd like it sent express". A default, so a
    file without them still loads."""
    """The answer to a delivery question when the spec holds no delivery wish (D115): the agent's suggestion goes, and
    no way is named, so a shopper with no wish never says "shipping" and "pickup" in one breath. A default, so a file
    without the line still loads."""


class ComparePhrases(_Frozen):
    named: list[str] = Field(min_length=1)
    """Openings of a compare task that names its candidates."""
    candidates: str
    """The clause of the predicate that names them."""
    criteria: dict[str, Phrase]
    """Each criterion that picks the best candidate: its predicate clause (say) and its opening sentence (clause)."""


class Phrases(_Frozen):
    schema_: Literal["mc.shopper-phrases.v1", "mc.shopper-phrases.v2"] = Field(alias="schema")
    version: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    replies: Replies
    consent: ConsentReplies
    choice: ChoiceReplies = Field(default_factory=ChoiceReplies)
    """D60: the replies to an offer of options or a recommendation."""
    variants: Variants = Field(default_factory=Variants)
    """D108: other ways to say a restatement or an acknowledgement, and lead-ins for a reply said before."""
    labels: dict[str, Label]
    options: dict[str, Phrase]
    facts: dict[str, Phrase]
    openings: dict[IntentType, list[str]]
    buyer: BuyerPhrases | None = None
    """v2 only: the buyer's checkout details."""
    compare: ComparePhrases | None = None
    """v2 only: compare tasks that name their candidates."""

    def label(self, field: str) -> str:
        found = self.labels.get(field)
        return found.label if found else field.replace("_", " ")

    def none_for(self, field: str) -> str | None:
        found = self.labels.get(field)
        return found.none if found else None

    def option(self, field: str) -> Phrase:
        return self.options.get(field) or self.options["default"]

    def fact(self, key: str) -> Phrase:
        found = self.facts.get(key)
        if found is None:
            raise NotFoundError(f"shopper phrases {self.version} have no fact {key!r}", hint="add it under facts")
        return found


def load_phrases(path: Path | None = None) -> Phrases:
    """The phrases at `path` (default: data/prompts/shopper-v2.yaml), parsed once per process and checked."""
    return _load((path or DEFAULT_PATH).resolve())


@functools.cache
def _load(path: Path) -> Phrases:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NotFoundError(f"no shopper phrases at {path}", hint="restore data/prompts from git") from None
    except yaml.YAMLError as exc:
        raise ValidationError(f"{path.name} is not valid YAML: {exc}") from exc
    try:
        phrases = Phrases.model_validate(raw)
    except PydanticValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "file"
        raise ValidationError(f"{path.name}: {where}: {first['msg']}", hint=f"follow {PHRASES_SCHEMA}") from exc
    _check(path.name, phrases)
    return phrases


def _check(name: str, phrases: Phrases) -> None:
    problems: list[str] = []
    problems += _unknown(phrases.replies.model_dump().values(), REPLY_SLOTS)
    problems += _unknown(phrases.consent.model_dump().values(), CONSENT_SLOTS)
    problems += _unknown(phrases.choice.model_dump().values(), CHOICE_SLOTS)
    problems += _unknown(phrases.variants.restate, REPLY_SLOTS)
    problems += _unknown(phrases.variants.nothing_asked, frozenset())
    problems += _unknown(phrases.variants.again, frozenset({"text"}))
    problems += _unknown((p.say for p in phrases.options.values()), OPTION_SLOTS)
    problems += _unknown((p.clause for p in phrases.options.values()), OPTION_SLOTS)
    problems += _unknown((p.say for p in phrases.facts.values()), FACT_SLOTS)
    problems += _unknown((p.clause for p in phrases.facts.values()), FACT_SLOTS)
    problems += _unknown((t for ts in phrases.openings.values() for t in ts), OPENING_SLOTS)
    if "default" not in phrases.options:
        problems.append("options needs a default phrase")
    missing = [intent for intent, templates in phrases.openings.items() if not templates]
    if missing or len(phrases.openings) < 5:
        problems.append("openings needs at least one template for each of the five intents")
    problems += _v2_problems(phrases)
    if problems:
        raise ValidationError(f"{name}: {'; '.join(sorted(set(problems)))}", hint="fix the template text")


def _v2_problems(phrases: Phrases) -> list[str]:
    """What a v2 file lacks or misspells in its buyer and compare sections; a v1 file has neither."""
    problems: list[str] = []
    if phrases.schema_ == "mc.shopper-phrases.v1":
        return problems
    if phrases.buyer is None or phrases.compare is None:
        return ["a mc.shopper-phrases.v2 file needs its buyer and compare sections"]
    buyer, compare = phrases.buyer, phrases.compare
    problems += _unknown(buyer.say.model_dump().values(), BUYER_SAY_SLOTS)
    problems += _unknown((v for vs in buyer.say_variants.values() for v in vs), BUYER_SAY_SLOTS)
    problems += _unknown(buyer.withheld.values(), frozenset())
    problems += _unknown(buyer.refuse.values(), frozenset())
    problems += _unknown(compare.named, COMPARE_SLOTS)
    problems += _unknown([compare.candidates], frozenset({"candidates"}))
    problems += _unknown((p.say for p in compare.criteria.values()), CRITERIA_SLOTS)
    problems += _unknown((p.clause for p in compare.criteria.values()), CRITERIA_SLOTS)
    for section, have, need in (
        ("buyer.withheld", buyer.withheld, BUYER_WITHHELD),
        ("buyer.refuse", buyer.refuse, BUYER_REFUSED),
        ("compare.criteria", compare.criteria, CRITERIA),
    ):
        lacking = sorted(need - set(have))
        if lacking:
            problems.append(f"{section} needs {', '.join(lacking)}")
    problems += [f"facts needs {key}" for key in ("budget_total", "budget_total_slot") if key not in phrases.facts]
    return problems


def _unknown(templates: Iterable[str], allowed: frozenset[str]) -> list[str]:
    out = []
    for template in templates:
        try:
            names = {n for _, n, _, _ in string.Formatter().parse(template) if n is not None}
        except ValueError as exc:
            out.append(f"broken placeholder in {template!r}: {exc}")
            continue
        out += [f"unknown placeholder {{{n}}} in {template!r}" for n in sorted(names - allowed)]
    return out


def fill(template: str, values: Mapping[str, str]) -> str:
    """The template with its placeholders filled and its spacing tidied: no double spaces, no space before a stop."""
    text = template.format_map(dict(values))
    for pattern, replacement in _TIDY:
        text = pattern.sub(replacement, text)
    return text.strip()


def capital(text: str) -> str:
    return text[:1].upper() + text[1:]

"""The deterministic shopper: the opening from the task, and every reply decided from the hidden spec, never by a model.

Since D118 the shopper's model may word a decided reply (`use_wording`, carttrace.agents.wording): the rules below still
decide its field, value, choice, consent and `disclosed`; the model only says it in the persona's voice, code checks
the words keep exactly those facts, and the rules' line is sent when they do not. Without a voice every reply is the
rules' line, as before.

Why no free text. The lab's shopper was a model playing a persona, and readers kept finding it volunteering facts it
was never asked for, or inventing a preference the spec did not hold, which the judge then scored the agent against
(lab/shopper.py, 2026-09-18). D9 rules free-text model output out of control flow. Here a reply is assembled from
data/prompts/shopper-v1.yaml and the spec:

- a question about a field the shopper stated repeats it, and one about an on_ask field reveals it;
- a question about a default field gets "no preference", and one about a field the spec does not hold gets the
  field's plain "none" line or "no preference": the shopper never invents a value;
- a question about an alias (their size, when they know only a measurement) is answered with the fact behind it;
- a fact the persona forgets (a predicate with `forgets`, generator 3) is "not to hand" the first time it is asked
  and given the second time, so an agent that asks once and moves on never learns it;
- a message that asks nothing gets a short acknowledgement.

Which field a question is about is the shopper.field decision. The keyword rule (carttrace.agents.keywords) answers first;
when it finds nothing in a message that does ask something, the injectable `decide` function is consulted, normally
`engine_field_decider(build_engine())`. That type is in shadow (data/decisions/shopper.yaml), so its answers are
logged for calibration and change nothing until it goes live: `Decision.effective` is False for a shadow type.

Buyer details (the spec's buyer block, data/prompts/shopper-v2.yaml `buyer`). A checkout needs an address, an email,
maybe a phone number and a delivery choice, and the shopper holds them like any other fact: a stated one is in the
opening, an on_ask one is given only when the agent asks for that very field, and a default one is never given (a
phone number the shopper keeps to themselves). Each reply lists the details it gave in `disclosed`, which the judge
reads to tell a detail the agent was given from one it made up. A detail the shopper does not have at all (a date of
birth, a second address, a name when the address has none, an ID number) is refused plainly, never invented. With
`disclose_contact=False` (a live row, where no email or phone may ever be typed) email and phone are withheld too.

Consent (checkout rows, D13). When the agent asks to place an order and states its total, the shopper reads the
store's checkout as the agent platform shows it and answers by code: yes when the stated total is the checkout's own
total, the order is what they asked for (one item, a witness of the hidden spec, in the wanted quantity, with no
repeat delivery they did not ask for) and it is within their budget; no otherwise. The budget's scope decides what it
is compared with: the items alone (the default, as every budget was generated and judged), or the total with shipping
and tax when the shopper said their budget is "all in". An impossible task has no witness, so the shopper approves no
order on it.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from carttrace.agents.choice import (
    CHECKED_OPS,
    NAMING_OPS,
    Offer,
    Verdict,
    bold_names,
    delivery_offer,
    fulfillment_need,
    judge_alone,
    judge_options,
    label_of,
    names,
    names_a_thing,
    names_title,
    read_offer,
)
from carttrace.agents.decision_map import ShopperMap
from carttrace.agents.keywords import (
    BUYER_KEYWORDS,
    KEYWORDS,
    REFUSAL_KEYWORDS,
    asked_fields,
    question_text,
    requested_text,
)
from carttrace.agents.nouns import names_product, noun_from
from carttrace.agents.personas import Disclosure
from carttrace.agents.phrases import BuyerPhrases, Phrases, capital, fill, load_phrases
from carttrace.agents.predicates import best, bundle_items, option_names
from carttrace.agents.products import Product, Variant, read_products
from carttrace.agents.service import (
    ChoicePath,
    ChoiceRecord,
    ConsentReason,
    ConsentReply,
    ConsentRequest,
    ShopperReply,
    ShopperSource,
)
from carttrace.agents.slots import money
from carttrace.agents.spec import BUYER_FIELDS, SUPERLATIVES, Address, HiddenSpec, Predicate
from carttrace.agents.tasks import SET_OPS, holds_in, line_quantity, quantity_of
from carttrace.agents.text import fold, has_phrase, words
from carttrace.agents.wallet import load_personas
from carttrace.agents.wording import WORDING_CODE, ShopperVoice, Stance, facts_of
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.ids import seeded_rng
from carttrace.core.money import Money
from carttrace.trace.schema import Cart, CartLine, TaskRef

if TYPE_CHECKING:
    from carttrace.decide.service import DecisionEngine

SHOPPER_CODE = "mc.agents.shopper/8"
"""The shopper's code version (D69): raise it when a change to this module changes what a reply says or which
branch answers. A trace records it with the phrases version on every reply and in the header."""
FieldDecider = Callable[[str, Mapping[str, str]], Awaitable[str | None]]
"""(agent message, field name to label) -> the field it asks about, "other", or None for nothing."""
FIELD_TYPE = "shopper.field"
MAX_FIELDS = 24
"""shopper.field letters its options for Clef, two of which are none and other."""
CONTACT_FIELDS = frozenset({"email", "phone"})
REFUSED_FIELDS = frozenset(REFUSAL_KEYWORDS)
"""Details the shopper never has to give a shop, and refuses plainly when asked."""
IDENTITY_OPS = frozenset({"kind", "is", "one_of", "prefer", "all_of"})
"""Needs that name the product itself. The agent names the product in a question about something else ("How would
you like to receive the tuner?"), so its name asks about it only when nothing else in the question is asked (D97)."""
PAYMENT_WORDS = (
    "shop pay", "google pay", "apple pay", "paypal", "card", "saved card", "credit card", "debit card", "klarna",
    "afterpay", "wallet", "bank transfer", "link",
)  # fmt: skip
"""Words that name a way to pay. No spec holds a payment preference: the persona's wallet decides (D94)."""
_SHIPS_OR = re.compile(
    r"\b(?:shipped|sent|delivered|posted|mailed|ship(?:ping)? it|deliver(?:ing|y)?)\b[^?]{0,120}?\bor\b"
    r"[^?]{0,40}\b(?:pick|picked|collect|in[- ]store|store)\b",
    re.IGNORECASE,
)
"""A question that puts delivery against pickup ("shipped to your address, or picked up at the store?"): it names the
address as where a delivery would go, and asks how the shopper wants the order, not where."""
WAYS: Mapping[str, tuple[str, ...]] = {
    "pickup": ("pickup", "pick up", "pick it up", "picked up", "pick-up", "collect", "collection", "in store",
               "in-store", "store counter"),
    "express": ("express", "expedited", "next day", "next-day", "overnight", "priority", "courier"),
    "standard": ("standard", "ground", "regular shipping", "economy", "free shipping"),
    "ship": ("shipping", "ship it", "shipped", "ship", "delivery", "delivered", "deliver", "post", "posted", "mail"),
}  # fmt: skip
"""Each way to receive an order, as an agent words it; `ship` is shipping of no named speed (D115)."""
_NEGATED = re.compile(
    r"\b(?:no|not|n't|without|doesn't (?:offer|have|support)|does not (?:offer|have|support)|isn't|unavailable)\b"
    r"[^.?!]{0,40}$",
    re.IGNORECASE,
)
"""A negation just before a way's words in its sentence ("there's no in-store pickup"): that way is not offered."""
_RECEIVE = re.compile(
    r"\bhow (?:would|do|will|should) you (?:like|want|prefer)?\s*(?:to )?(?:receive|get|have)\b|"
    r"\b(?:delivery|shipping|fulfil?ment) (?:option|method|speed|choice)s?\b",
    re.IGNORECASE,
)
"""A question of how the order should reach the shopper, whatever ways it names."""
_SHIP_TO = re.compile(r"^please ship it to\s+", re.IGNORECASE)
_PAY = re.compile(r"\b(?:pay|paying|payment)\b", re.IGNORECASE)
_ASKS_WHERE = re.compile(r"\b(?:what|which|where)\b[^.?!]{0,40}\b(?:address|ship|send|deliver)", re.IGNORECASE)
_PAY_ASK = re.compile(
    r"\b(?:shall|should|can|may)\s+i\b[^?]{0,30}?\b(?:charge|pay|place (?:the |your )?order|complete (?:the |your )?"
    r"(?:order|purchase|checkout)|submit (?:the |your )?(?:order|payment))\b",
    re.IGNORECASE,
)
"""A yes or no question to pay or place the order ("Shall I charge your card for $161.99?"). On the v1 pilot it was
read as a recommendation and answered "I'll go with TKL Z-4 4-Channel Audio Mixer." (D108)."""
_COUNTED = re.compile(r"(?<![$£€\d.,])\b(\d{1,3})\s+([a-z][a-z-]*)", re.IGNORECASE)
"""A count and the noun it counts ("1 set", "2 sets"), not a price or a measure."""
NOT_COUNTED = frozenset({
    "day", "business", "week", "hour", "minute", "month", "year", "oz", "ml", "l", "g", "kg", "lb", "cm", "mm", "in",
    "inch", "percent", "x", "of", "and", "or", "to", "for", "at",
})  # fmt: skip
"""Words after a number that do not make it a count of the item: times, measures and joining words."""


class DeterministicShopper:
    """A Shopper whose every word comes from the task. One instance per episode."""

    def __init__(
        self,
        task: TaskRef,
        *,
        decide: FieldDecider | None = None,
        phrases: Phrases | None = None,
        disclose_contact: bool = True,
        disclosure: Disclosure | None = None,
    ) -> None:
        self.task = task
        self.disclosure = disclosure
        """The persona's disclosure style (carttrace.agents.personas): only `volunteers` names a need it kept for asking
        when it turns options down. None when the task names no persona the book holds."""
        self.spec = HiddenSpec.from_task(task)
        self.phrases = phrases or load_phrases()
        if self.spec.buyer is not None and self.phrases.buyer is None:
            raise ValidationError(
                f"task {task.task_id} has a buyer block, and shopper phrases {self.phrases.version} have no words "
                "for it",
                hint="use data/prompts/shopper-v2.yaml (the default)",
            )
        self.disclose_contact = disclose_contact
        self._decide = decide
        self._keywords = {**KEYWORDS, **BUYER_KEYWORDS, **REFUSAL_KEYWORDS} if self.phrases.buyer else KEYWORDS
        self.asked: list[tuple[str, ...]] = []
        """The fields each reply answered, in order: what the agent asked about."""
        self.revealed: set[str] = set()
        """on_ask fields the shopper has disclosed so far."""
        self.disclosed: set[str] = set()
        """Buyer details the shopper has given so far, stated ones included once asked."""
        self.forgotten: set[str] = set()
        """Fields the shopper did not have to hand the first time they were asked (a predicate with `forgets`)."""
        self._catalog: dict[str, tuple[Product, Variant]] = {}
        """The store's products as the sandbox serves them, by variant key (`use_catalog`); empty: witnesses only."""
        self._products: dict[str, Product] = {}
        """The same products by id, for reading which one an offered option names."""
        self._said: set[str] = set()
        """Every reply text sent so far: none is sent twice word for word (D108)."""
        self._restates = 0
        self._acks = 0
        self._counting = False
        """The reply being built answers a question of how many (D108)."""
        self._restated_now = False
        """A restatement's lead ("Like I said") has been said in the reply being built: at most one per reply."""
        self.decisions: ShopperMap | None = None
        """The task's build-time decision map on the row's world (D115, `use_map`); None: decided from the served
        catalog as it is read (rows built before the map, mask rows)."""
        self.lookups = 0
        """Product decisions answered from the decision map."""
        self.voice: ShopperVoice | None = None
        """Words each decided reply with the shopper's model (D118, `use_wording`); None: the rules' lines."""

    def opening(self) -> str:
        return self.task.opening

    @property
    def version(self) -> str:
        """The shopper's code and phrases version (D69): what a trace records beside every reply; with the wording
        code when a model words the replies (D118)."""
        base = f"{SHOPPER_CODE}+{self.phrases.version}"
        return f"{base}+{WORDING_CODE}" if self.voice is not None else base

    def use_wording(self, voice: ShopperVoice) -> None:
        """Word every reply's decided facts with `voice` from here on (D118). The rules still decide each reply:
        its fields, values, choice, consent and `disclosed` are the rules' own; only its words change."""
        self.voice = voice
        voice.opened(self.opening())

    async def reply(self, agent_message: str, *, cut: bool = False) -> ShopperReply:
        """The answer, with the branch that produced it (`rule`) and the shopper's version (D69)."""
        self._restated_now = False
        reply, rule = await self._reply(agent_message, cut=cut)
        decided = replace(
            reply, text=self._fresh(reply.text, rule), rule=rule, version=self.version, source=self._source(rule)
        )
        return await self._worded(decided, agent_message)

    # -- wording the decided facts (D118) --------------------------------------------------------------------------

    async def _worded(self, reply: ShopperReply, agent_message: str) -> ShopperReply:
        """`reply` with its text worded by the voice, when there is one; every other field as the rules decided."""
        if self.voice is None:
            return reply
        text, wording, model, usage = await self._word(reply.text, agent_message, self._stance(reply), reply.rule)
        return replace(reply, text=text, wording=wording, wording_model=model, usage=usage)

    async def _word(
        self, line: str, agent_message: str, stance: Stance | None, rule: str | None
    ) -> tuple[str, Any, str | None, dict[str, int] | None]:
        assert self.voice is not None
        self.voice.heard(agent_message)
        known, hidden = self._told()
        hidden += self._offered_labels(agent_message)
        labels = [self.phrases.label(name) for name in [*self.options(), *BUYER_FIELDS, *REFUSED_FIELDS]]
        facts = facts_of(line, known=known, hidden=hidden, labels=labels, stance=stance)
        worded = await self.voice.word(facts, rule=rule or "")
        return worded.text, worded.wording, worded.model, dict(worded.usage) or None

    @staticmethod
    def _stance(reply: ShopperReply) -> Stance | None:
        """The yes or no the rules decided, which the words must keep."""
        if reply.rule in ("go_ahead", "payment_go_ahead"):
            return "approve"
        if reply.rule == "cart_mismatch":
            return "refuse"
        path = reply.choice.path if reply.choice is not None else None
        if path in ("picked", "confirmed"):
            return "pick"
        if path in ("none_fits", "declined"):
            return "refuse"
        return None

    def _told(self) -> tuple[list[str], list[str]]:
        """What the agent has been told (stated and revealed values, given details) and what it has not (values kept
        for asking, details not given, the store's other products), as phrases the words are checked against."""
        known: list[str] = []
        hidden: list[str] = []
        for p in self.spec.predicates:
            value = p.value
            texts = (
                [value]
                if isinstance(value, str)
                else [v for v in value if isinstance(v, str)]
                if isinstance(value, list)
                else []
            )
            (known if p.source == "stated" or p.field in self.revealed else hidden).extend(texts)
        buyer = self.spec.buyer
        for name in ("shipping_address", "email", "phone"):
            found = buyer.field(name) if buyer is not None else None
            if found is None:
                continue
            held = found.value
            if isinstance(held, Address):
                texts = [t for t in (held.name, *held.lines, held.city, held.postal_code) if t]
            else:
                texts = [str(held)]
            (known if found.source == "stated" or name in self.disclosed else hidden).extend(texts)
        titles = [p.title for p in self._products.values() if len(words(p.title)) >= 2]
        hidden += [t for t in titles if t not in known]
        return known, hidden

    @staticmethod
    def _offered_labels(agent_message: str) -> list[str]:
        offer = read_offer(agent_message)
        return [label_of(o) for o in offer.options] if offer is not None else []

    def _source(self, rule: str) -> ShopperSource:
        """Where a reply of `rule` came from (D115): the decision map for a choice of items or delivery and a held
        cart, the keyword rules for a field, the safe default when nothing matched."""
        if rule in ("choice", "delivery_field", "go_ahead", "cart_mismatch"):
            return "lookup" if self.decisions is not None or rule == "delivery_field" else "matched"
        if rule == "other":
            return "default"
        if rule == "decided_field":
            return "model"
        return "matched"

    # -- saying it (D108) -----------------------------------------------------------------------------------------

    def _restate(self, values: Mapping[str, str]) -> str:
        """A stated fact said again: the restatements go through their variants in turn, and only the first in a
        reply leads with one; the rest are said plainly, so no reply says "Like I said" twice."""
        if self._restated_now:
            return fill(self.phrases.replies.reveal, values)
        lines = self._lines(self.phrases.replies.restate, self.phrases.variants.restate)
        self._restated_now = True
        self._restates += 1
        return fill(lines[(self._restates - 1) % len(lines)], values)

    def _ack(self) -> str:
        lines = self._lines(self.phrases.replies.nothing_asked, self.phrases.variants.nothing_asked)
        self._acks += 1
        return lines[(self._acks - 1) % len(lines)]

    @staticmethod
    def _lines(first: str, variants: Sequence[str]) -> list[str]:
        return [first, *(v for v in variants if v != first)]

    def _fresh(self, text: str, rule: str = "") -> str:
        """`text`, or, when it was said before word for word, the first of its other forms not yet said: another
        acknowledgement (two together once each has been said), or the reply with a lead-in ("To say it again: ...").
        On the v1 pilot one refusal was sent three times and "Okay, thanks." answered three sign-offs in a row."""
        if not text or text not in self._said:
            self._said.add(text)
            return text
        if rule in ("nothing_asked", "acknowledged"):
            acks = self._lines(self.phrases.replies.nothing_asked, self.phrases.variants.nothing_asked)
            forms = [*acks, *(f"{a} {b}" for a in acks for b in acks if a != b)]
        else:
            leads = self.phrases.variants.again
            forms = [fill(lead, {"text": text}) for lead in leads]
            forms += [fill(a, {"text": fill(b, {"text": text})}) for a in leads for b in leads if a != b]
        found = next((form for form in forms if form not in self._said), text)
        self._said.add(found)
        return found

    async def _reply(self, agent_message: str, *, cut: bool) -> tuple[ShopperReply, str]:  # noqa: PLR0911 - one return per branch
        asked = question_text(agent_message)
        requested = requested_text(agent_message)
        offer = None if cut else read_offer(agent_message)
        if not asked and offer is None:
            self.asked.append(())
            return ShopperReply(self._ack(), None, False, cut=cut), "nothing_asked"
        paying = self._paying(offer, requested)
        if paying is not None:
            self.asked.append(())
            return ShopperReply(paying[0], None, False, cut=cut), paying[1]
        if not requested and offer is None:
            # the agent said it holds the detail ("I'll use your profile's email"): a statement, acknowledged (D97)
            self.asked.append(())
            return ShopperReply(self._ack(), None, False, cut=cut), "acknowledged"
        fields: list[str] = []
        rule = "keyword_field"
        counted = self._counted(requested, offer)
        self._counting = bool(counted)
        if not counted and self._delivery_question(agent_message, requested, offer):
            return self._delivery_reply(agent_message, requested, offer, cut=cut), "delivery_field"
        if counted:
            fields, rule = counted, "quantity_field"
        elif requested:
            fields = self._ordered(self._buyer_view(self._held_as(self._asked_in(requested)), requested))
            if offer is not None:
                fields = [f for f in fields if self._holds(f)]
            if not fields and self._decide is not None:
                chosen = await self._decide(agent_message, self.options())
                fields = [chosen] if chosen and chosen not in ("none", "other") else []
                rule = "decided_field"
        if offer is not None and not counted and all(self._stated(f) for f in fields):
            return self._choose(offer, agent_message), "choice"
        self.asked.append(tuple(fields))
        if not fields:
            return ShopperReply(self.phrases.replies.other, None, False, cut=cut), "other"
        lines: list[str] = []
        revealed = False
        answered: set[str] = set()
        disclosed: list[str] = []
        for name in fields:
            line, said = self._answer(name, answered, disclosed)
            if line and line not in lines:
                lines.append(line)
            revealed = revealed or said
        self.disclosed.update(disclosed)
        order = {name: i for i, name in enumerate(BUYER_FIELDS)}
        given = tuple(sorted(dict.fromkeys(disclosed), key=lambda f: order.get(f, len(order))))
        return ShopperReply(" ".join(lines), fields[0], revealed, tuple(fields), given, cut=cut), rule

    # -- a delivery or fulfilment question (D115) ------------------------------------------------------------------

    def _delivery_question(self, message: str, requested: str, offer: Offer | None) -> bool:
        """The agent asks how the order should reach the shopper: its question (or the options it lays out) names a
        way to receive it while the message names two ways or more ("shipping or pickup?", "Standard or Express?"),
        or asks how they want to receive it. An offer of the store's products is not one, whatever it says of
        delivery. On 2026-10-17, "The checkout offers two ways to receive the order: shipping or pickup. Which would
        you prefer?" was weighed as an offer of products and got "None of those works for me. Like I said, I'm
        choosing between ..." (ep-ba077be3dfce, ep-73bf41962c22), and 9 order_not_placed cards had such a reply."""
        if self.phrases.buyer is None or self.spec.buyer is None:
            return False
        if offer is not None and any(self._named_product(o) is not None for o in offer.options):
            return False
        if offer is not None and offer.options and not any(_ways(o) for o in offer.options):
            return False
        asked = " ".join([requested, offer.asked if offer is not None else ""])
        if _RECEIVE.search(asked):
            return True
        return bool(_ways(asked)) and len(_ways(message)) >= 2

    def _delivery_reply(self, message: str, requested: str, offer: Offer | None, *, cut: bool) -> ShopperReply:
        """The delivery wish from the spec, in one way only: the wish when the message offers it, the wish asked for
        when it does not ("None of those works for me. I'll collect it from the store. Is that possible?"), or, with
        no wish, whatever the agent suggests. Other details the message asks for are answered beside it, but never a
        shipping address beside a collection (ep-20321f26e46e: "Please ship it to <address>. I'll collect it from
        the store.")."""
        buyer = self.spec.buyer
        assert buyer is not None and self.phrases.buyer is not None
        held = self._wish()
        lines: list[str] = []
        disclosed: list[str] = []
        revealed = False
        fields = ["fulfillment"]
        record: ChoiceRecord | None = None
        listed = offer.options if offer is not None else ()
        recommended = offer is not None and offer.recommended is not None
        if held is None or held.source == "default" or not held.say:
            lines.append(self.phrases.buyer.open_fulfillment)
        else:
            if held.source == "on_ask":
                self.revealed.add("fulfillment")
            reason = (
                self._restate(_clause(held.say, "fulfillment"))
                if held.source == "stated"
                else fill(self.phrases.replies.reveal, _clause(held.say, "fulfillment"))
            )
            pick = next((i for i, o in enumerate(listed) if _wish_offered(held.value, _offered(o), alone=True)), None)
            plain = next((i for i, o in enumerate(listed) if _offered(o) == {"ship"}), None)
            if pick is None and held.value != "pickup":
                pick = plain  # "shipping" of no named speed carries a standard or express wish
            if pick is not None:
                # the wish said as the spec holds it, never an option's words ("express 1–2 days)" is no answer)
                lines.append(reason)
                record = ChoiceRecord("picked", len(listed), pick + 1, recommended, ("fulfillment",))
            elif not listed:
                lines.append(reason)  # a question, not a list: the wish said plainly, whatever it names
            else:
                lines.append(fill(self.phrases.choice.ask_for, {"option": "", "reason": reason}))
                if listed:
                    missed = ("fulfillment",)
                    record = ChoiceRecord("none_fits", len(listed), None, recommended, missed, missed, missed)
            revealed = True
            if held.source in ("stated", "on_ask"):
                disclosed.append("fulfillment")
        others = self._buyer_view(self._held_as(self._asked_in(requested)), requested) if requested else []
        drop = {"fulfillment", "delivery_by"} | ({"shipping_address", "other_address"} if self._collects() else set())
        answered: set[str] = set()
        for name in self._ordered([f for f in others if f not in drop and f in BUYER_FIELDS]):
            line, said = self._answer(name, answered, disclosed)
            if line and line not in lines:
                lines.append(line)
                fields.append(name)
            revealed = revealed or said
        self.asked.append(tuple(fields))
        self.disclosed.update(disclosed)
        order = {name: i for i, name in enumerate(BUYER_FIELDS)}
        given = tuple(sorted(dict.fromkeys(disclosed), key=lambda f: order.get(f, len(order))))
        return ShopperReply(" ".join(lines), "fulfillment", revealed, tuple(fields), given, choice=record, cut=cut)

    def _wish(self) -> Any:
        """The delivery wish: the decision map's when the task carries one, else the spec's buyer block."""
        if self.decisions is not None and self.decisions.delivery is not None:
            return self.decisions.delivery
        return self.spec.buyer.fulfillment if self.spec.buyer is not None else None

    # -- an offer of options or a recommendation (D60) -----------------------------------------------------------

    def _choose(self, offer: Offer, message: str) -> ShopperReply:
        """Pick the option that meets the needs its words speak to, agree to a recommendation that does, or say none
        fits, naming the need only when it was stated, the offer asked about its field, or the persona volunteers."""
        unlocked = set(asked_fields(offer.asked, self.spec.fields(), self._values()))
        recommended = offer.recommended is not None
        for p in self.spec.predicates:
            if p.source == "on_ask" and p.forgets and p.field in unlocked and not self._to_hand(p):
                self.forgotten.add(p.field)
                self.asked.append((p.field,))
                record = ChoiceRecord("forgot", len(offer.options), recommended=recommended)
                text = fill(self.phrases.replies.forgot, {"label": self.phrases.label(p.field)})
                return ShopperReply(text, p.field, False, (p.field,), choice=record)
        delivery = self._delivery(offer)
        needs = delivery or [p for p in self.spec.predicates if p.op in CHECKED_OPS and self._usable(p)]
        currency, quantity = self.spec.currency, quantity_of(self.spec.predicates)
        verdicts = judge_options(
            offer.options, needs, context=offer.asked, currency=currency, quantity=quantity, message=message
        )
        unlocked |= self._settled_by_list(offer, verdicts, needs, delivery=bool(delivery))
        served = [self._named_product(option) for option in offer.options]
        needs = self._served_verdicts(served, verdicts, needs)
        pick: int | None = None
        rec_index = offer.recommended_option
        if offer.recommended is not None:
            alone = offer.recommended if offer.options else message
            rec = (
                verdicts[rec_index]
                if rec_index is not None
                else judge_alone(alone, needs, currency=currency, quantity=quantity)
            )
            if rec_index is None:
                needs = self._served_verdicts([self._named_product(alone)], [rec], needs)
            top = self._best(offer, verdicts, served)
            if "misses" not in rec.values():
                better = top is not None and _meets(verdicts[top]) > _meets(rec)
                if not better:
                    return self._confirm(offer, rec_index, rec, needs)
                pick = top
            elif top is not None:
                pick = top
            else:
                return self._unmet(offer, [rec], needs, unlocked, "declined")
        else:
            pick = self._best(offer, verdicts, served)
        if pick is None:
            return self._unmet(offer, verdicts, needs, unlocked, "none_fits")
        return self._picked(offer, pick, verdicts, needs)

    def _settled_by_list(
        self, offer: Offer, verdicts: Sequence[dict[str, Verdict]], needs: Sequence[Predicate], *, delivery: bool
    ) -> set[str]:
        """The needs an offer is a list of values for, which every option that does not name the wanted value misses,
        and which the offer therefore asks about (so the shopper may say what it wants). Why (D97): on the v1 pilot
        of 2026-10-16 a shopper who would collect from the store, offered "Standard or Express?", picked Express,
        because neither option named pickup and an option that says nothing was taken to fit; the judge counts 613
        of 1,417 failed fulfilment checks under v11 with that cause. A list of ways to deliver is about the way of
        delivery the buyer holds (unless it is a default, no preference); a list whose options name values the
        store serves for the need's own option ("7oz", "12oz" of Size) is about that option."""
        settled: set[str] = set()
        buyer = self.spec.buyer
        for need in needs:
            if delivery:
                held = buyer.fulfillment if buyer is not None else None
                about = need.field == "fulfillment" and held is not None and held.source != "default"
            else:
                about = need.op == "eq" and self._lists_values_of(offer, need)
            if not about:
                continue
            settled.add(need.field)
            for verdict in verdicts:
                if verdict.get(need.field) != "meets":
                    verdict[need.field] = "misses"
        return settled

    def _lists_values_of(self, offer: Offer, need: Predicate) -> bool:
        """Two options or more name a value the served catalog holds for the need's own option."""
        if not need.option or not self._products:
            return False
        name = fold(need.option)
        served = {
            value
            for product in self._products.values()
            for variant in product.variants
            for option, value in variant.options
            if fold(option) == name and value.strip()
        }
        return sum(1 for option in offer.options if any(has_phrase(option, v) for v in served)) >= 2

    def _paying(self, offer: Offer | None, question: str) -> tuple[str, str] | None:
        """The reply and its rule to a message about paying: a question of how to pay (`_pays`), or a yes or no
        question to pay or place the order outside the wallet step (D108); None for any other message."""
        if self._pays(offer, question):
            return self.phrases.choice.any_fits, "payment_offer"
        if _PAY_ASK.search(question) and (offer is None or not offer.options):
            return self.phrases.replies.pay_go_ahead, "payment_go_ahead"
        return None

    def _pays(self, offer: Offer | None, question: str) -> bool:
        """The agent asks how to pay: its question names two ways to pay, or its offer's options do while none names a
        product the store serves or a way to deliver. Why (D97): on the v1 pilot "How would you like to pay: Shop
        Pay, a saved card, or Google Pay?" was weighed against the product wanted and refused ("None of those works
        for me. I'm looking for a tuner."), and the agent's next turns, puzzling over it, ran into the token limit
        until the journey ended."""
        if _PAY.search(question) and sum(1 for w in PAYMENT_WORDS if has_phrase(question, w)) >= 2:
            return True
        if offer is None or delivery_offer(offer):
            return False
        if any(self._named_product(option) is not None for option in offer.options):
            return False
        return sum(1 for option in offer.options if any(has_phrase(option, w) for w in PAYMENT_WORDS)) >= 2

    def _delivery(self, offer: Offer) -> list[Predicate]:
        """The buyer's way of delivery, when the offer is of ways to deliver (two options or more name one): the one
        they asked for, or standard when they have no preference, as their answer to a question about it says. It is
        then the only need weighed (a shipping option is not the product), and a list of products that mentions free
        shipping is not chosen by it."""
        buyer = self.spec.buyer
        if self.phrases.buyer is None or buyer is None or buyer.fulfillment is None or not delivery_offer(offer):
            return []
        found = buyer.fulfillment
        return [fulfillment_need(found.value, found.source, found.say)]

    def _counted(self, question: str, offer: Offer | None) -> list[str]:
        """The quantity and budget fields the spec holds, when the message asks how many: its question, or its offer's
        options, name two counts or more of one noun ("hold at 1 set, or is 2 sets at $35.97 fine?"). Why (D108): on
        the v1 pilot that question was weighed as an offer of products and got "None of those works for me. Like I
        said, I want the Dip Dye Boy Pocket Tee & Shorts Set."; a buyer answers how many, and what they will spend."""
        texts = [question, *(offer.options if offer is not None else ())]
        counts: dict[str, set[str]] = {}
        for text in texts:
            for number, noun in _COUNTED.findall(text):
                singular = fold(noun).removesuffix("s")
                if singular and singular not in NOT_COUNTED:
                    counts.setdefault(singular, set()).add(number)
        if not any(len(numbers) >= 2 for numbers in counts.values()):
            return []
        return [f for f in ("quantity", "budget") if self.spec.predicate(f) is not None]

    def _holds(self, name: str) -> bool:
        """The spec holds the field: a need, an alias of one, or a buyer detail. Only those answer beside an offer
        (D108): "No preference on the pack size." answered a list of drum kits that named a pack."""
        if self.spec.predicate(name) is not None or self.spec.alias_of(name) is not None:
            return True
        buyer = self.spec.buyer
        return buyer is not None and name in BUYER_FIELDS and buyer.field(name) is not None

    def _stated(self, name: str) -> bool:
        """A field the shopper said in the opening: a question about it on an offer ("Which would you like, the $55
        moisturizer or a free one?" names the moisturizer) would only be told it again, so the offer is answered."""
        predicate = self.spec.predicate(name)
        if predicate is not None:
            return predicate.source == "stated"
        buyer = self.spec.buyer
        return buyer is not None and name in BUYER_FIELDS and buyer.source_of(name) == "stated"

    def _to_hand(self, predicate: Predicate) -> bool:
        return predicate.field in self.forgotten or predicate.field in self.revealed

    def _usable(self, predicate: Predicate) -> bool:
        """A need the shopper weighs options by: stated or kept for asking, never a default, and a detail the persona
        forgets only once it has been asked for it."""
        if predicate.source == "stated":
            return True
        return predicate.source == "on_ask" and (not predicate.forgets or self._to_hand(predicate))

    def _best(
        self, offer: Offer, verdicts: Sequence[Mapping[str, str]], served: Sequence[Product | None] = ()
    ) -> int | None:
        """The fitting option meeting the most needs; a seeded draw among equals, so a replay chooses the same. When
        some options name or price a thing, a course of action ("Walk away") is never the pick: if every product
        offered misses a need, none fits. When some options name a product the store serves, only those are picked:
        "Stop here, and you can come back later if the $46 set..." prices a thing but is not one (2026-10-16)."""
        things = [i for i, product in enumerate(served) if product is not None]
        things = things or [i for i, option in enumerate(offer.options) if names_a_thing(option)]
        allowed = things or range(len(verdicts))
        fitting = [i for i in allowed if "misses" not in verdicts[i].values()]
        if not fitting:
            return None
        top = max(_meets(verdicts[i]) for i in fitting)
        ties = [i for i in fitting if _meets(verdicts[i]) == top]
        rng = seeded_rng("shopper.choice", self.task.task_id, len(self.asked), list(offer.options))
        return ties[0] if len(ties) == 1 else rng.choice(ties)

    def _picked(
        self, offer: Offer, pick: int, verdicts: Sequence[Mapping[str, str]], needs: Sequence[Predicate]
    ) -> ShopperReply:
        label, named = self._label(offer.options[pick], verdicts[pick], needs)
        option = f"option {pick + 1}, {label}" if offer.numbered else label or f"option {pick + 1}"
        text = fill(self.phrases.choice.picked, {"option": option, "reason": ""})
        record = ChoiceRecord(
            "picked",
            len(offer.options),
            pick + 1,
            offer.recommended is not None,
            _checked(needs, verdicts),
            named=named,
        )
        return self._choice_reply(text, record, named, _decided(needs, [verdicts[pick]]))

    def _confirm(
        self, offer: Offer, index: int | None, verdicts: Mapping[str, str], needs: Sequence[Predicate]
    ) -> ShopperReply:
        sentence = offer.recommended or ""
        source = offer.options[index] if index is not None else sentence
        label, named = self._label(source, verdicts, needs)
        bold = bold_names(sentence)
        if index is None:
            label = bold[0] if bold else ""
            named = tuple(f for f in named if any(names(label, n) for n in needs if n.field == f))
        text = fill(self.phrases.choice.confirmed, {"option": label or "that one", "reason": ""})
        options = len(offer.options)
        record = ChoiceRecord(
            "confirmed",
            options,
            None if index is None else index + 1,
            True,
            _checked(needs, [verdicts]),
            named=named,
        )
        return self._choice_reply(text, record, named, _decided(needs, [verdicts]))

    def _unmet(
        self,
        offer: Offer,
        verdicts: Sequence[Mapping[str, str]],
        needs: Sequence[Predicate],
        unlocked: set[str],
        path: ChoicePath,
    ) -> ShopperReply:
        """None of the options fits (or the recommended one does not): the need most of them miss, named when the
        shopper stated it, the offer asked about its field, or the persona volunteers what it wants."""
        missed = [n for n in needs if any(v.get(n.field) == "misses" for v in verdicts)]
        if not missed:
            record = ChoiceRecord(path, len(offer.options), None, offer.recommended is not None)
            template = self.phrases.choice.declined if path == "declined" else self.phrases.choice.none_fits
            return self._choice_reply(template, record, (), ())
        need = max(missed, key=lambda n: sum(1 for v in verdicts if v.get(n.field) == "misses"))
        replies, choice = self.phrases.replies, self.phrases.choice
        reason = ""
        disclosed: list[str] = []
        if need.source == "stated" and need.say:
            reason = self._restate(_said(need))
        elif need.say and (need.field in unlocked or self.disclosure == "volunteers"):
            reason = fill(replies.reveal, _said(need))
        if reason and need.field == "budget":
            self._budget_scope_with(disclosed)
        named = (need.field,) if reason else ()
        if path == "declined":
            template = choice.declined_named if reason else choice.declined
        elif reason and need.field == "fulfillment":
            template = choice.ask_for  # what they want, and whether the store can do it (D97)
        else:
            template = choice.none_fits_named if reason else choice.none_fits
        text = fill(template, {"option": "", "reason": reason})
        record = ChoiceRecord(
            path,
            len(offer.options),
            None,
            offer.recommended is not None,
            _checked(needs, verdicts),
            (need.field,),
            named,
        )
        return self._choice_reply(text, record, named, (), disclosed)

    def _label(
        self, option: str, verdicts: Mapping[str, str], needs: Sequence[Predicate]
    ) -> tuple[str, tuple[str, ...]]:
        """The option's name as the shopper says it back, with the value of each need it meets by naming one that
        the name leaves out ("Supershine Hydrating Shampoo, Travel"), and the fields whose values it then gives."""
        label = label_of(option)
        named: list[str] = []
        for need in needs:
            if need.op not in NAMING_OPS or verdicts.get(need.field) != "meets":
                continue
            if not names(label, need) and need.op in ("eq", "brand", "mentions") and isinstance(need.value, str):
                label = f"{label}, {need.value}" if label else need.value
            if names(label, need):
                named.append(need.field)
        return label, tuple(named)

    def _choice_reply(
        self,
        text: str,
        record: ChoiceRecord,
        named: tuple[str, ...],
        decided: tuple[str, ...],
        disclosed: Sequence[str] = (),
    ) -> ShopperReply:
        """The reply's annotation reads as a field reply does: a reply that gives values names those fields and is
        revealed; one that gives none names the fields its choice turned on, so it is never read as a fallback."""
        given = list(disclosed)
        buyer = self.spec.buyer
        for name in named:
            predicate = self.spec.predicate(name)
            if predicate is not None and predicate.source == "on_ask":
                self.revealed.add(name)
            source = buyer.source_of(name) if buyer is not None and name in BUYER_FIELDS else None
            if source in ("stated", "on_ask"):
                given.append(name)
                if source == "on_ask":
                    self.revealed.add(name)
        named = tuple(n for n in named if self.spec.predicate(n) is not None or n in given)
        self.disclosed.update(given)
        disclosed = given
        fields = named or decided
        self.asked.append(fields)
        return ShopperReply(text, fields[0] if fields else None, bool(named), fields, tuple(disclosed), choice=record)

    async def go_ahead(self, agent_message: str, cart: Cart | None) -> ShopperReply:
        """The answer when the agent stops at a ready cart on a checkout row, whose journey ends with an order: the
        message's question answered as `reply` answers it; else, for a message that asks nothing the shopper can
        answer, a go-ahead to order the cart when it holds what they asked for (`_wanted`, the consent's own test),
        or what is wrong with it. An impossible task has no witness, so its cart is never approved. On 2026-10-15,
        40 checkout rows ended at cart_ready, 38 of them with the email or address the checkout needed never given."""
        self._restated_now = False
        reply, rule = await self._reply(agent_message, cut=False)
        if rule not in ("nothing_asked", "other") or cart is None or not cart.lines:
            decided = replace(
                reply, text=self._fresh(reply.text, rule), rule=rule, version=self.version, source=self._source(rule)
            )
        elif self._wanted(cart):
            text = self._fresh(self.phrases.consent.go_ahead)
            decided = ShopperReply(
                text, None, False, rule="go_ahead", version=self.version, source=self._source("go_ahead")
            )
        else:
            text = self._fresh(self._mismatch_text(cart))
            decided = ShopperReply(
                text, None, False, rule="cart_mismatch", version=self.version, source=self._source("cart_mismatch")
            )
        return await self._worded(decided, agent_message)

    async def consent(self, request: ConsentRequest) -> ConsentReply:
        """Approve or refuse an order for the total the agent stated, from the hidden spec and the store's checkout."""
        self._restated_now = False
        reason = self._consent_reason(request)
        replies = self.phrases.consent
        if reason == "consented":
            text = fill(replies.consented, {"total": money(request.total.amount, request.total.currency)})
            decided = ConsentReply(self._fresh(text), True, reason, version=self.version)
        elif reason == "cart_mismatch":
            text = self._fresh(self._mismatch_text(request.checkout))
            decided = ConsentReply(text, False, reason, version=self.version)
        else:
            decided = ConsentReply(self._fresh(getattr(replies, reason)), False, reason, version=self.version)
        if self.voice is None:
            return decided
        # the code's yes or no stands (`given`); the words keep it, or the rules' line is sent
        stance: Stance = "approve" if decided.given else "refuse"
        text, wording, model, usage = await self._word(decided.text, request.message, stance, "consent")
        return replace(decided, text=text, wording=wording, wording_model=model, usage=usage)

    def _mismatch_text(self, checkout: Cart) -> str:
        """What is wrong with the order, as a shopper would say it, checked in `_wanted`'s order. The bare "That isn't
        what I asked for" left the agent nothing to fix: on 2026-10-15, 327 of 1,457 sampled consent requests ended
        the journey on it, among them a tee carted once for a shopper who said "I'd like 2"."""
        replies = self.phrases.consent
        lines = checkout.lines
        bundle = next((p for p in self.spec.predicates if p.op == "all_of"), None)
        wrong = next((line for line in lines if not self._fits(line)), None)
        if wrong is not None:
            need = self._missed_by(wrong.variant_id)
            reason = ""
            if need is not None and need.say and need.source == "stated":
                reason = " " + self._restate(_said(need))
            elif need is not None and need.say and (self._to_hand(need) or self.disclosure == "volunteers"):
                reason = " " + fill(self.phrases.replies.reveal, _said(need))
            return fill(replies.wrong_item, {"reason": reason})
        if bundle is None and len({_key(line.variant_id) for line in lines}) > 1:
            return replies.extra_items
        wanted, held = quantity_of(self.spec.predicates), sum(line.quantity for line in lines)
        if lines and bundle is None and held != wanted:
            if self.spec.intent == "budget_max":
                # the count is what the budget buys; the shopper never named it (D108)
                return replies.fewer_than_budget if held < wanted else replies.more_than_budget
            return fill(replies.wrong_quantity, {"count": str(wanted), "held": str(held)})
        if lines and any(line.selling_plan is not None for line in lines):
            return replies.repeat_plan
        return replies.cart_mismatch

    def _consent_reason(self, request: ConsentRequest) -> ConsentReason:
        checkout = request.checkout
        stated = request.total
        if (checkout.currency or "").upper() != stated.currency:
            return "currency_mismatch"
        if checkout.total is None or checkout.total != stated.amount:
            return "total_mismatch"
        if not self._wanted(checkout):
            return "cart_mismatch"
        budget = self.budget()
        if budget is not None and budget.currency != stated.currency:
            return "currency_mismatch"
        if budget is not None and self._spent(checkout, stated) > budget.amount:
            return "over_budget"
        return "consented"

    def _spent(self, checkout: Cart, stated: Money) -> int:
        """What the budget is compared with: the stated total when it covers shipping and tax, else the items alone."""
        if self.spec.budget_scope == "total":
            return stated.amount
        return sum(line.unit_price * line.quantity for line in checkout.lines)

    def budget(self) -> Money | None:
        """The most the shopper will pay in all, from the spec's budget predicate; None when they named none."""
        predicate = next((p for p in self.spec.predicates if p.op == "max_price" or p.field == "budget"), None)
        value = predicate.value if predicate is not None else None
        if not isinstance(value, dict):
            return None
        amount, currency = value.get("amount"), value.get("currency") or self.spec.currency
        if not isinstance(amount, int) or isinstance(amount, bool) or not isinstance(currency, str):
            return None
        return Money(amount, currency.upper())

    def _wanted(self, checkout: Cart) -> bool:
        """The checkout holds what the shopper asked for: one item (or, for a bundle, one line of each item), every
        line a witness of the spec, in the wanted quantity, and on no selling plan unless they asked for repeat
        delivery."""
        lines = checkout.lines
        witnesses = {_key(w) for w in self.spec.witnesses}
        if not lines or not witnesses:
            return False
        if not all(self._fits(line) for line in lines):
            return False
        if sum(line.quantity for line in lines) != quantity_of(self.spec.predicates):
            return False
        bundle = next((p for p in self.spec.predicates if p.op == "all_of"), None)
        if bundle is None and len({_key(line.variant_id) for line in lines}) != 1:
            return False
        if bundle is not None and not _covers(bundle, lines):
            return False
        repeat = self.spec.predicate("subscription")
        wants_plan = repeat is not None and repeat.value is True
        return wants_plan or all(line.selling_plan is None for line in lines)

    def _missed_by(self, variant_id: str) -> Predicate | None:
        """The need a carted variant misses, the first in spec order, as a shopper would say why: on 2026-10-16 every
        refusal said "Like I said, <the first stated need>" ("a body lotion" for a body lotion that was not the
        cheapest), and agents argued that the item was what was asked for. From the served catalog when it holds the
        variant, else from each need's satisfied_by, else the first need the shopper stated."""
        missed = self.decisions.missed(_key(variant_id)) if self.decisions is not None else None
        if missed is not None and self.spec.predicate(missed) is not None:
            self.lookups += 1
            return self.spec.predicate(missed)
        found = self._catalog.get(_key(variant_id))
        if found is not None:
            return self._failing_need(found[0], [found[1]])
        key = _key(variant_id)
        for p in self.spec.predicates:
            if p.field != "quantity" and p.satisfied_by is not None and key not in {_key(v) for v in p.satisfied_by}:
                return p
        stated = (p for p in self.spec.predicates if p.source == "stated" and p.field not in ("quantity", "budget"))
        return next(stated, None)

    def _failing_need(self, product: Product, variants: Sequence[Variant]) -> Predicate | None:
        """The first need of the spec that none of `variants` of `product` meets: a superlative when none is as good
        as the witnesses, any other need when the data does not show it holds."""
        quantity = line_quantity(self.spec.predicates)
        for p in self.spec.predicates:
            if p.op in SUPERLATIVES:
                if not any(self._as_good(p, product, v) for v in variants):
                    return p
            elif p.op not in SET_OPS and not any(
                holds_in(self.spec, p, product, v, quantity) is True for v in variants
            ):
                return p
        return None

    def _as_good(self, superlative: Predicate, product: Product, variant: Variant) -> bool:
        """The variant is a witness, or ranks with them: the cheapest of the store's sets is the cheapest set even
        when the generator's shelf left it out."""
        witnesses = [self._catalog[k] for k in (_key(w) for w in self.spec.witnesses) if k in self._catalog]
        if _key(variant.id) in {_key(w) for w in self.spec.witnesses}:
            return True
        return bool(witnesses) and variant.id in best(superlative, [*witnesses, (product, variant)])

    def _named_product(self, text: str) -> Product | None:
        """The served product an option's words name, the most specific when several do (a title said whole beats
        one said in part, a longer title a shorter one); None without a catalog or when none is named. A one-word
        title ("Shampoo") names a kind, not a product, so it is never read as one."""
        scored: list[tuple[int, int, Product]] = []
        for product in self._products.values():
            if len(words(product.title)) < 2:
                continue
            if has_phrase(text, product.title):
                scored.append((1, len(product.title), product))
            elif names_title(text, product.title):
                scored.append((0, len(product.title), product))
        return max(scored, key=lambda s: (s[0], s[1], s[2].id))[2] if scored else None

    def _served_verdicts(
        self, served: Sequence[Product | None], verdicts: Sequence[dict[str, Verdict]], needs: Sequence[Predicate]
    ) -> list[Predicate]:
        """Each option naming a served product none of whose variants the shopper would approve misses the need it
        fails, so the shopper never picks or agrees to an item it will refuse at consent. Why: on 2026-10-16, 121 of
        296 give-ups on rows built to pay had the shopper pick or confirm an option and then refuse that very item,
        because the words of an option settle no ranking ("the cheapest") and no need its text leaves out. The need
        joins the needs weighed; whether the reply names it is still `_unmet`'s rule."""
        out = list(needs)
        for product, verdict in zip(served, verdicts, strict=True):
            if product is None or any(self._fits_variant(v.id) for v in product.variants):
                continue
            need = self._failing_need(product, product.variants)
            if need is None:
                continue
            verdict[need.field] = "misses"
            if all(n.field != need.field for n in out):
                out.append(need)
        return out

    def use_catalog(self, fixtures: Iterable[Mapping[str, Any]]) -> None:
        """Read the store's own products as its sandbox serves them, so an order of a variant outside the task's
        witnesses that meets every need is approved too. Why: the generator lists witnesses only on the task's shelf
        (products whose kind is the target's very noun), while the replica serves the whole catalog. On 2026-10-15
        the shopper refused a $15 bodysuit for "a bodysuit under $25" because the store calls it a graphic
        bodysuit; 96 of 171 classified refusals named a product off the shelf."""
        for product in read_products(fixtures, currency=self.spec.currency):
            self._products.setdefault(product.id, product)
            for variant in product.variants:
                self._catalog.setdefault(_key(variant.id), (product, variant))

    def use_map(self, doc: Mapping[str, Any]) -> None:
        """Decide from the task's build-time decision map (carttrace.agents.decision_map, D115) from here on."""
        self.decisions = ShopperMap.of(doc)

    def served_variants(self) -> dict[str, tuple[Product, Variant]]:
        """Every served variant by key (`use_catalog`), for the decision map's build."""
        return dict(self._catalog)

    def fits_variant(self, variant_id: str) -> bool:
        """The rule `_fits_variant` decides by, never the map: what the map is built from."""
        return self._fits_by_rule(variant_id)

    def failing_need(self, product: Product, variants: Sequence[Variant]) -> Predicate | None:
        return self._failing_need(product, variants)

    def _fits(self, line: CartLine) -> bool:
        return self._fits_variant(line.variant_id)

    def _fits_variant(self, variant_id: str) -> bool:
        """The decision map's answer for the variant when it holds one (D115), else the rule's."""
        if self.decisions is not None:
            found = self.decisions.decides(_key(variant_id))
            if found is not None:
                self.lookups += 1
                return found
        return self._fits_by_rule(variant_id)

    def _fits_by_rule(self, variant_id: str) -> bool:
        """The variant is a witness, or, read from the served catalog, an available variant that meets every need
        the way the generator checks a witness (carttrace.agents.tasks.holds_in) and, for a spec that ranks (the cheapest),
        ranks with the witnesses. A spec that names a bundle keeps to its witnesses: the generator paired them."""
        key = _key(variant_id)
        if key in {_key(w) for w in self.spec.witnesses}:
            return True
        found = self._catalog.get(key)
        predicates = self.spec.predicates
        if found is None or not self.spec.witnesses or any(p.op == "all_of" for p in predicates):
            return False
        product, variant = found
        quantity = line_quantity(predicates)
        if not variant.available or variant.id in self.spec.stocked_out or not variant.holds_units(quantity):
            return False
        if not all(
            holds_in(self.spec, p, product, variant, quantity) is True for p in predicates if p.op not in SET_OPS
        ):
            return False
        superlative = next((p for p in predicates if p.op in SUPERLATIVES), None)
        return superlative is None or (self._named_kind(product) and self._as_good(superlative, product, variant))

    def _named_kind(self, product: Product) -> bool:
        """A store-wide best is of products whose title names the kind asked for, as the writer ranks it
        (carttrace.agents.intents_custom): a body scrub filed under Body Cream is no cheapest body cream."""
        kind = next((p for p in self.spec.predicates if p.op == "kind" and isinstance(p.value, str)), None)
        noun = noun_from(str(kind.value)) if kind is not None else None
        return kind is None or (noun is not None and names_product(noun, product.title))

    def options(self) -> dict[str, str]:
        """The fields the shopper can answer about, with their labels: the evidence `fields` of shopper.field. The
        buyer block's fields come after the predicates and their aliases."""
        names = [p.field for p in self.spec.predicates]
        names += [a for p in self.spec.predicates for a in p.aliases if a not in names]
        if self.spec.buyer is not None:
            names += [f for f in self.spec.buyer.fields() if f not in names]
        return {name: self.phrases.label(name) for name in names[:MAX_FIELDS]}

    def _known_fields(self) -> list[str]:
        names = list(self._keywords)
        names += [n for n in self.options() if n not in self._keywords]
        return names

    def _buyer_view(self, fields: list[str], question: str = "") -> list[str]:
        """The asked fields as a checkout reads them: a question about another address is not one about the shipping
        address; one that puts delivery against pickup ("shipped to your address, or picked up?") asks how, not
        where; and with a delivery preference to give, a question about the delivery date is about that."""
        if self.phrases.buyer is None:
            return fields
        out = list(fields)
        if "other_address" in out:
            out = [f for f in out if f != "shipping_address"]
        if "shipping_address" in out and _SHIPS_OR.search(question) and not _ASKS_WHERE.search(question):
            out = [f for f in out if f != "shipping_address"]
            if "fulfillment" not in out:
                out.append("fulfillment")
        buyer = self.spec.buyer
        if (
            "delivery_by" in out
            and self.spec.predicate("delivery_by") is None
            and buyer is not None
            and buyer.fulfillment is not None
        ):
            out = [f for f in out if f != "delivery_by"]
            if "fulfillment" not in out:
                out.append("fulfillment")
        return out

    def _held_as(self, fields: list[str]) -> list[str]:
        """Asked fields as the spec holds them: a question about the size, when the spec holds no size but a pack
        size the store calls "Size", is about the pack size. On 2026-10-16 "Which size would you like?" got "No
        preference on the size" from shoppers who wanted the 7oz or the XL, and the order was then refused."""
        out: list[str] = []
        for name in fields:
            held = name
            if self.spec.predicate(name) is None and name not in BUYER_FIELDS:
                spoken = name.replace("_", " ")
                held = next(
                    (
                        p.field
                        for p in self.spec.predicates
                        if p.source != "default" and spoken in {fold(n) for n in option_names(p.field, p.option)}
                    ),
                    name,
                )
            if held not in out:
                out.append(held)
        return out

    def _asked_in(self, question: str) -> list[str]:
        """The fields `question` asks about: by keyword or a named value, with the product's own name counting only
        when nothing else is asked (IDENTITY_OPS)."""
        known = self._known_fields()
        found = asked_fields(question, known, self._values(identity=False), keywords=self._keywords)
        return found or asked_fields(question, known, self._values(), keywords=self._keywords)

    def _values(self, *, identity: bool = True) -> dict[str, list[str]]:
        """Values that, named in a question, mean it asks about their field (a colour, a size, a product name);
        without the product's own name when `identity` is False."""
        out: dict[str, list[str]] = {}
        for predicate in self.spec.predicates:
            if not identity and predicate.op in IDENTITY_OPS:
                continue
            value = predicate.value
            if isinstance(value, str):
                out[predicate.field] = [value]
            elif isinstance(value, list):
                out[predicate.field] = [v for v in value if isinstance(v, str)]
        return out

    def _ordered(self, fields: list[str]) -> list[str]:
        """Spec fields first, in spec order, then aliases, then fields the spec does not hold."""
        order = {name: i for i, name in enumerate(self.options())}
        return sorted(fields, key=lambda f: (order.get(f, len(order)), f))

    def _answer(self, name: str, answered: set[str], disclosed: list[str]) -> tuple[str, bool]:
        buyer_words = self.phrases.buyer
        if buyer_words is not None and (name in BUYER_FIELDS or name in REFUSED_FIELDS):
            return self._buyer_answer(name, buyer_words, answered, disclosed)
        return self._predicate_answer(name, answered, disclosed)

    def _predicate_answer(self, name: str, answered: set[str], disclosed: list[str]) -> tuple[str, bool]:
        replies = self.phrases.replies
        label = self.phrases.label(name)
        predicate = self.spec.predicate(name)
        if predicate is not None and predicate.field == "budget" and predicate.source != "default":
            self._budget_scope_with(disclosed)
        if predicate is None:
            behind = self.spec.alias_of(name)
            if behind is not None and behind.field not in answered:
                answered.add(behind.field)
                return self._alias(name, behind)
            if behind is not None:
                return "", False
            return self.phrases.none_for(name) or fill(replies.unknown, {"label": label}), False
        answered.add(name)
        if predicate.source == "default" or not predicate.say:
            # asked how many of a count they never named, the shopper wants one, as the consent holds them to (D108)
            one = self._counting and predicate.field == "quantity" and bool(predicate.say)
            return (fill(replies.reveal, _said(predicate)) if one else fill(replies.default, {"label": label})), False
        if predicate.source == "stated":
            return self._restate(_said(predicate)), True
        return self._reveal(name, predicate, label)

    def _reveal(self, name: str, predicate: Predicate, label: str) -> tuple[str, bool]:
        """An on_ask fact, given now; a fact the persona forgets is not to hand the first time it is asked."""
        if predicate.forgets and name not in self.forgotten:
            self.forgotten.add(name)
            return fill(self.phrases.replies.forgot, {"label": label}), False
        self.revealed.add(name)
        return fill(self.phrases.replies.reveal, _said(predicate)), True

    def _buyer_answer(
        self, name: str, words: BuyerPhrases, answered: set[str], disclosed: list[str]
    ) -> tuple[str, bool]:
        """A buyer detail by its source, a refusal of a detail the shopper does not have, never an invented one."""
        buyer = self.spec.buyer
        address = buyer.shipping_address if buyer is not None else None
        if name == "recipient_name" and address is not None and address.value.name:
            name = "shipping_address"  # the address has a name line, so the name is part of it
        if name in REFUSED_FIELDS:
            return words.refuse[name], False
        if name in answered:
            return "", False
        answered.add(name)
        found = buyer.field(name) if buyer is not None else None
        withheld = not self.disclose_contact and name in CONTACT_FIELDS
        if found is None or found.source == "default" or not found.say or withheld:
            return words.withheld[name], False
        disclosed.append(name)
        replies = self.phrases.replies
        say = found.say
        if name == "shipping_address" and self._collects():
            # D115: a shopper who collects gives their address as theirs, never as where to ship (ep-20321f26e46e)
            say = _SHIP_TO.sub("my address is ", say)
        if found.source == "on_ask":
            self.revealed.add(name)
        if found.source == "on_ask":
            return fill(replies.reveal, _clause(say, name)), True
        return self._restate(_clause(say, name)), True

    def _collects(self) -> bool:
        """The shopper's delivery wish is to collect from the store (stated or kept for asking)."""
        held = self.spec.buyer.fulfillment if self.spec.buyer is not None else None
        return held is not None and held.value == "pickup" and held.source != "default"

    def _budget_scope_with(self, disclosed: list[str]) -> None:
        """A budget said "all in" gives its scope with it."""
        buyer = self.spec.buyer
        if buyer is not None and buyer.budget_scope.value == "total" and buyer.budget_scope.source != "default":
            disclosed.append("budget_scope")

    def _alias(self, name: str, behind: Predicate) -> tuple[str, bool]:
        if behind.source == "default" or not behind.say:
            return fill(self.phrases.replies.default, {"label": self.phrases.label(name)}), False
        if behind.source == "on_ask":
            self.revealed.add(behind.field)
        values = {**_said(behind), "label": self.phrases.label(name)}
        return fill(self.phrases.replies.alias, values), True


def _ways(text: str) -> set[str]:
    """The ways to receive an order `text` names (WAYS)."""
    return {way for way, terms in WAYS.items() if any(has_phrase(text, t) for t in terms)}


def _offered(message: str) -> set[str]:
    """The ways a message offers: each way it names in a sentence that does not deny it ("no in-store pickup")."""
    out: set[str] = set()
    for sentence in re.split(r"(?<=[.?!])\s+|\n+", message):
        low = sentence.lower()
        for way, terms in WAYS.items():
            for word in terms:
                at = low.find(word)
                if at >= 0 and has_phrase(sentence, word) and not _NEGATED.search(low[:at]):
                    out.add(way)
                    break
    return out


def _wish_offered(wish: str, offered: set[str], *, alone: bool = False) -> bool:
    """The wish is one of the ways offered: shipping of no named speed is standard, and is express too when express
    is not named apart. `alone`: the ways of one listed option, where shipping of no named speed is standard only."""
    if wish == "pickup":
        return "pickup" in offered
    if wish == "express":
        return "express" in offered or (not alone and "ship" in offered and "standard" not in offered)
    if alone:
        return "standard" in offered
    return bool({"standard", "ship"} & offered)


def _meets(verdicts: Mapping[str, str]) -> int:
    return sum(1 for v in verdicts.values() if v == "meets")


def _checked(needs: Sequence[Predicate], verdicts: Sequence[Mapping[str, str]]) -> tuple[str, ...]:
    """The needs the options' words settled one way or the other, in spec order."""
    return tuple(n.field for n in needs if any(v.get(n.field, "unknown") != "unknown" for v in verdicts))


def _decided(needs: Sequence[Predicate], verdicts: Sequence[Mapping[str, str]]) -> tuple[str, ...]:
    """The needs the chosen option meets by its words: what the choice turned on."""
    return tuple(n.field for n in needs if any(v.get(n.field) == "meets" for v in verdicts))


def variant_key(variant_id: str) -> str:
    """How the shopper keys a variant (the decision map's keys)."""
    return _key(variant_id)


def _key(variant_id: str) -> str:
    """A variant id compared by its last segment, so a store's gid and the bare id it ends in are one variant."""
    return variant_id.rstrip("/").rsplit("/", 1)[-1]


def _covers(bundle: Predicate, lines: Sequence[CartLine]) -> bool:
    """Every item of a bundle has one line of one unit in the cart."""
    held = {_key(line.variant_id): line.quantity for line in lines}
    for item in bundle_items(bundle.value):
        variants = item.get("variants")
        keys = {_key(str(v)) for v in variants} if isinstance(variants, list) else set()
        if not any(held.get(k) == 1 for k in keys):
            return False
    return True


def _said(predicate: Predicate) -> dict[str, str]:
    return _clause(predicate.say, predicate.field)


def _clause(say: str, field: str) -> dict[str, str]:
    return {"say": say, "Say": capital(say), "label": field.replace("_", " ")}


def engine_field_decider(engine: DecisionEngine, *, context: Mapping[str, Any] | None = None) -> FieldDecider:
    """A FieldDecider that asks the decision engine's shopper.field. Only an effective decision (a live type, no
    backend error) names a field; a shadow or failed one is logged by the engine and changes nothing."""

    async def decide(agent_message: str, fields: Mapping[str, str]) -> str | None:
        evidence = {"agent_message": agent_message, "fields": dict(fields)}
        decision = await engine.decide(FIELD_TYPE, evidence, context=context)
        if not decision.effective or not isinstance(decision.answer, str) or decision.answer == "none":
            return None
        return decision.answer

    return decide


def shopper_for(
    task: TaskRef,
    *,
    engine: DecisionEngine | None = None,
    context: Mapping[str, Any] | None = None,
    phrases: Phrases | None = None,
    disclose_contact: bool = True,
) -> DeterministicShopper:
    """The shopper the runner uses: the keyword rule, then shopper.field on `engine` (the configured engine from
    carttrace.decide.build_engine when None). `context` (episode_id, task_id, ...) is logged with each decision. A live row
    passes `disclose_contact=False`, so the shopper never gives an email or phone an agent could type into a store."""
    if engine is None:
        from carttrace.decide.factory import build_engine

        engine = build_engine()
    ids = {"task_id": task.task_id, **(context or {})}
    return DeterministicShopper(
        task,
        decide=engine_field_decider(engine, context=ids),
        phrases=phrases,
        disclose_contact=disclose_contact,
        disclosure=disclosure_of(task),
    )


def opening_origin(task: TaskRef) -> dict[str, Any]:
    """The opening turn's `origin` (D69): who wrote the words the agent first reads, recorded rather than inferred. A
    voice rewrite the guard accepted, or a custom intent a model wrote, is the model's (with the voice model and its
    prompt's sha256 when the spec kept them); anything else is the template's. `version` is the spec's generator."""
    try:
        spec = HiddenSpec.from_task(task)
    except ValidationError:
        return {"rule": "opening"}
    voice = spec.voice
    origin: dict[str, Any] = {"rule": "opening", "version": spec.generator or None}
    if voice is not None and voice.accepted:
        origin |= {"author": "model", "voice_model": voice.model, "prompt_sha256": voice.prompt_sha256}
    elif task.intent_kind is not None and spec.generator.startswith("carttrace.agents.intents_custom"):
        origin |= {"author": "model"}
    else:
        origin |= {"author": "template"}
    return {k: v for k, v in origin.items() if v is not None}


def disclosure_of(task: TaskRef) -> Disclosure | None:
    """The disclosure style of the persona the task's buyer block names, from data/agent/personas.yaml; None when
    the task names no persona or one the book does not hold (a spec older than the personas file)."""
    buyer = HiddenSpec.from_task(task).buyer
    if buyer is None or buyer.persona is None:
        return None
    try:
        return load_personas().get(buyer.persona).style.disclosure
    except NotFoundError:
        return None

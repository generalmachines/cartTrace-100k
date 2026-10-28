"""The buyer block of a hidden spec: drawn from a persona and the task's seed, and read back by the shopper and agents.

Why the shopper holds these details. Until generator 2 the agent was handed the persona's email, phone and address in
its system prompt, so nothing measured whether an agent asks for what a checkout needs or makes it up. Now the
shopper holds them, each with a disclosure source like a predicate, and the store's own checkout says which ones it
needs. What is drawn, from a Random of its own (so the products and predicates a seed gives are the same as before):

- the shipping address: the persona's address in the store's market, kept until the agent asks;
- the email: the persona's reserved address, kept until asked;
- the phone: the persona's reserved number, kept until asked, or never given (a default) so that an agent which
  enters one anyway has invented it;
- the delivery preference: standard, express, or pickup where the market has a counter, sometimes with a deadline
  in days that only the matching option's stated delivery time meets; stated, kept until asked, or (for standard)
  the plain assumption;
- the budget scope: `items` (the default, as every budget was judged before) or `total`, when the budget covers
  shipping and tax too ("all in, with shipping"). A total budget is drawn only where the sandbox's market table can
  price the order exactly, so the witnesses stay exact.

Also here, because the reference agent needs them too: a persona's UCP postal address in the spec's shape and back,
which kind of fulfillment option a store's option is, and the delivery time an option's title states.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from random import Random
from typing import Any, Final

from carttrace.agents.phrases import Phrases, capital, fill
from carttrace.agents.spec import (
    Address,
    BudgetScope,
    Buyer,
    BuyerAddress,
    BuyerBudgetScope,
    BuyerEmail,
    BuyerFulfillment,
    BuyerPhone,
    FulfillmentKind,
    Source,
)
from carttrace.agents.wallet import MARKET_ALIASES, Persona
from carttrace.core.errors import ValidationError
from carttrace.core.ids import seeded_rng

TOTAL_SHARE: Final = 0.35
"""Share of budgets that cover shipping and tax too, where the market table can price the order."""
PHONE_KEPT: Final = 0.5
"""Share of tasks whose shopper gives a phone number when asked; the rest never give one."""
CURRENCY_MARKETS: Final[Mapping[str, str]] = {
    "USD": "US",
    "CAD": "CA",
    "GBP": "GB",
    "EUR": "EU",
    "INR": "IN",
    "AUD": "AU",
    "NZD": "NZ",
    "SGD": "SG",
    "MXN": "MX",
    "JPY": "JP",
    "PHP": "PH",
}
"""The delivery market a catalog's currency implies, when the store names none (persona markets, GB not UK)."""
EXPRESS_WORDS: Final = ("express", "expedited", "priority", "next day", "next-day", "overnight", "same day", "rush")
STANDARD_WORDS: Final = ("standard", "economy", "ground", "regular", "basic", "free shipping", "post")
PICKUP_WORDS: Final = ("pickup", "pick up", "pick-up", "collect", "counter", "in store", "in-store")
_DAYS: Final = re.compile(
    r"(\d+)\s*(?:-|to|–)\s*(\d+)\s*(?:business\s+|working\s+)?days?|(\d+)\s*(?:business\s+|working\s+)?days?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class BuyerPlan:
    """The draws of one task's buyer block, made before its words are written."""

    persona: Persona
    market: str
    address: Address
    preference: FulfillmentKind
    preference_source: Source
    deadline_days: int | None
    phone_source: Source
    scope: BudgetScope
    """total only where the market table can price the order; the generator falls back to items otherwise."""


def market_of(store: object, currency: str | None) -> str | None:
    """The delivery market a task's shopper lives in: the store's market code (carttrace.stores.Store.market_code, or a
    StoreRef's market), else the one its catalog's currency implies. GB, not UK, as personas.yaml writes it."""
    for name in ("market_code", "market"):
        found = getattr(store, name, None)
        if isinstance(found, str) and re.fullmatch(r"[A-Za-z]{2}", found.strip()):
            code = found.strip().upper()
            return MARKET_ALIASES.get(code, code)
    return CURRENCY_MARKETS.get((currency or "").upper())


def address_from(postal: Mapping[str, str], *, name: str | None = None) -> Address:
    """A persona's UCP postal address (data/agent/personas.yaml) as the spec's Address, with the persona's fictional
    name as its name line when it has one."""
    lines = [postal["street_address"], *([postal["extended_address"]] if postal.get("extended_address") else [])]
    return Address(
        name=name,
        lines=lines,
        city=postal["address_locality"],
        region=postal.get("address_region") or None,
        postal_code=postal["postal_code"],
        country=postal["address_country"],
    )


def ucp_address(address: Address) -> dict[str, str]:
    """The spec's Address as a UCP postal address, the shape a UCP checkout's shipping destination takes."""
    out: dict[str, str] = {"street_address": address.lines[0]}
    if len(address.lines) > 1:
        out["extended_address"] = address.lines[1]
    out["address_locality"] = address.city
    if address.region:
        out["address_region"] = address.region
    out["postal_code"] = address.postal_code
    out["address_country"] = address.country
    if address.name:
        first, _, last = address.name.partition(" ")
        out["first_name"] = first
        if last:
            out["last_name"] = last
    return out


def option_kind(title: str, option_id: str = "", method: str | None = None) -> FulfillmentKind | None:
    """Which kind a store's fulfillment option is, from its method and its words; None when they do not say."""
    if method == "pickup":
        return "pickup"
    words = f"{title} {option_id}".lower()
    if any(w in words for w in PICKUP_WORDS):
        return "pickup"
    if any(w in words for w in EXPRESS_WORDS):
        return "express"
    if any(w in words for w in STANDARD_WORDS):
        return "standard"
    return None


def stated_days(title: str) -> int | None:
    """The longest delivery time an option's title states, in days: "Standard (4-7 days)" is 7; None when none."""
    found = day_range(title)
    return None if found is None else found[1]


def day_range(title: str) -> tuple[int, int] | None:
    """(shortest, longest) delivery time an option's title states, in days: "Standard (4-7 days)" is (4, 7)."""
    found = _DAYS.search(title)
    if found is None:
        return None
    if found.group(3) is not None:
        return int(found.group(3)), int(found.group(3))
    return int(found.group(1)), int(found.group(2))


UNASKED_INTENTS: Final = frozenset({"express"})
"""Intents whose opening tells the agent to just pick (D108): no delivery wish is kept for asking on them."""


def draw_plan(
    persona: Persona,
    market: str,
    rng: Random,
    *,
    priced: bool,
    delivery: FulfillmentKind | None = None,
    intent: str | None = None,
) -> BuyerPlan | None:
    """One task's buyer draws, or None when the persona has no address in `market`. `priced` says whether the market
    table can price an order there, which a total budget needs. `delivery` is the persona's leaning (carttrace.agents.intents
    Variety): express or pickup weight the draw towards that kind; standard, or None, draws as every task did.

    A delivery wish kept for asking stays hidden only where a careful agent would ask about it (D108, the project's
    ruling on F5 of the 2026-10-16 prompt review): an `express` task's shopper says "Just pick one that works", so
    its wish is said in the opening instead. On the pilot, "a mixer under $200 ... Just pick one that works." hid a
    pickup, no agent asked, and every one was judged on it. The draws are the same; only where the wish is said
    moves."""
    postal = persona.address_for(market)
    if postal is None:
        return None
    options = market_options(market)
    kinds: list[FulfillmentKind] = ["standard", "standard", "express"]
    if "pickup" in options:
        kinds.append("pickup")  # only where the market has a counter to collect from
    if delivery == "express":
        kinds = ["express", "express", "standard"]
        if "pickup" in options:
            kinds.append("pickup")
    elif delivery == "pickup" and "pickup" in options:
        kinds = ["pickup", "pickup", "standard", "express"]
    preference: FulfillmentKind = rng.choice(kinds)
    draw = rng.random()
    source: Source = "stated" if draw < 0.4 else "on_ask" if draw < 0.8 or preference != "standard" else "default"
    if intent in UNASKED_INTENTS and source == "on_ask":
        source = "stated"
    deadline = _deadline(preference, options, rng) if source != "default" else None
    phone: Source = "on_ask" if rng.random() < PHONE_KEPT else "default"
    scope: BudgetScope = "total" if priced and rng.random() < TOTAL_SHARE else "items"
    if preference == "pickup":
        # "all in, with shipping" says the order is shipped; a shopper who collects has an items budget (D104)
        scope = "items"
    return BuyerPlan(
        persona=persona,
        market=market,
        address=address_from(postal, name=persona.display_name),
        preference=preference,
        preference_source=source,
        deadline_days=deadline,
        phone_source=phone,
        scope=scope,
    )


def buyer_block(
    plan: BuyerPlan, phrases: Phrases, *, scope: BudgetScope, budget_source: Source | None, key: str = ""
) -> Buyer:
    """The spec's buyer block from the plan. `scope` is the plan's, or items when the task has no budget; a total
    scope is disclosed with the budget, so it shares the budget's source. `key` (the task's own words) draws how the
    delivery wish and its deadline are said among the phrases' variants (D118); empty, the first wording."""
    if phrases.buyer is None:
        raise ValidationError(
            f"shopper phrases {phrases.version} have no buyer words", hint="use data/prompts/shopper-v2.yaml"
        )
    words = phrases.buyer.say
    address = plan.address
    fulfillment_say = _worded(phrases, plan.preference, getattr(words, plan.preference), key)
    if plan.deadline_days is not None:
        deadline = _worded(phrases, "deadline", words.deadline, key)
        fulfillment_say = f"{fill(deadline, {'days': str(plan.deadline_days)})}, so {fulfillment_say}"
    total = scope == "total" and budget_source is not None
    return Buyer(
        persona=plan.persona.name,
        market=plan.market,
        shipping_address=BuyerAddress(
            value=address, source="on_ask", say=fill(words.shipping_address, {"address": address.words()})
        ),
        email=BuyerEmail(
            value=plan.persona.email, source="on_ask", say=fill(words.email, {"value": plan.persona.email})
        ),
        phone=BuyerPhone(
            value=plan.persona.phone, source=plan.phone_source, say=fill(words.phone, {"value": plan.persona.phone})
        ),
        fulfillment=BuyerFulfillment(
            value=plan.preference,
            deadline_days=plan.deadline_days,
            source=plan.preference_source,
            say=fulfillment_say,
        ),
        budget_scope=BuyerBudgetScope(
            value="total" if total else "items",
            source=budget_source if total and budget_source is not None else "default",
            say=words.budget_total if total else "",
        ),
    )


def _worded(phrases: Phrases, kind: str, first: str, key: str) -> str:
    """One wording of a delivery clause: the first without a key, else a seeded draw over it and its variants."""
    variants = phrases.buyer.say_variants.get(kind, []) if phrases.buyer is not None else []
    if not key or not variants:
        return first
    return seeded_rng("buyer.say", key, kind).choice([first, *(v for v in variants if v != first)])


def stated_sentences(buyer: Buyer) -> str:
    """What the opening says of the buyer block: each stated field but the budget scope, which the budget's own
    wording carries."""
    said = [
        capital(found.say) + "."
        for name in ("shipping_address", "email", "phone", "fulfillment")
        if (found := buyer.field(name)) is not None and found.source == "stated" and found.say
    ]
    return " ".join(said)


def all_in_total(price: int, quantity: int, *, market: str, currency: str, preference: FulfillmentKind) -> int | None:
    """What the sandbox's checkout charges for `quantity` of a variant at `price` with the preferred fulfillment, tax
    included: the arithmetic of carttrace.sandbox.core.pricing over the market table. None when the market is not in the
    table or offers no option of that kind."""
    from carttrace.sandbox.core import pricing
    from carttrace.sandbox.core.market import UnknownMarket, market_for
    from carttrace.sandbox.core.state import Group, Line

    try:
        row = market_for(market, currency)
    except UnknownMarket:
        return None
    items = price * quantity
    if preference == "pickup":
        offered = pricing.pickup_options(row)
    else:
        offered = tuple(
            o for o in pricing.shipping_options(row, items) if option_kind(o["title"], o["id"]) == preference
        )
    if not offered:
        return None
    line = Line("l1", "v1", quantity, price)
    group = Group("g1", "pickup" if preference == "pickup" else "shipping", ("l1",), amount=int(offered[0]["amount"]))
    return pricing.totals([line], (), {}, row, [group]).total


def priced_market(market: str, currency: str) -> bool:
    """Whether the sandbox's market table can price an order in `market` and `currency`."""
    return all_in_total(100, 1, market=market, currency=currency, preference="standard") is not None


def market_options(market: str) -> dict[str, Any]:
    """The market table's options by kind: standard and express shipping, and pickup, each with its stated days."""
    from carttrace.sandbox.core.market import UnknownMarket
    from carttrace.sandbox.core.market import market as market_row

    try:
        row = market_row(market)
    except UnknownMarket:
        return {}
    out: dict[str, Any] = {}
    for option in row.shipping:
        kind = option_kind(option.title, option.id)
        if kind is not None and kind not in out:
            out[kind] = day_range(option.title)
    if row.pickup:
        out["pickup"] = None
    return out


def with_deadline(plan: BuyerPlan) -> BuyerPlan:
    """The plan with a delivery deadline the preferred option's stated time meets, for a task that needs one (a gift
    that must arrive in time): a stated express delivery one standard delivery cannot meet, or a stated standard
    delivery by its longest time plus a day. A pickup, or a market whose options state no times, keeps the plan."""
    options = market_options(plan.market)
    standard, express = options.get("standard"), options.get("express")
    preference = plan.preference
    if preference == "pickup" and standard is not None:
        preference = "standard"
    if preference == "express" and express is not None and standard is not None and express[1] < standard[0]:
        return replace(plan, preference="express", preference_source="stated", deadline_days=express[1])
    if preference != "pickup" and standard is not None:
        return replace(plan, preference="standard", preference_source="stated", deadline_days=standard[1] + 1)
    return plan


def _deadline(preference: FulfillmentKind, options: Mapping[str, Any], rng: Random) -> int | None:
    """Now and then, a deadline the preferred option's stated delivery time meets: for express, one standard
    delivery cannot meet even at its quickest; for standard, one its longest time meets. None otherwise."""
    if preference == "pickup" or rng.random() >= 0.5:
        return None
    standard, express = options.get("standard"), options.get("express")
    if preference == "express" and express is not None and standard is not None and express[1] < standard[0]:
        return int(rng.randint(express[1], standard[0] - 1))
    if preference == "standard" and standard is not None:
        return int(rng.randint(standard[1], standard[1] + 3))
    return None

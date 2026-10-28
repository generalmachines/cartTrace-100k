"""An opening must not contradict its hidden spec (D104): what the shopper says first is what the judge grades against.

Why (2026-10-16). The judge found open-layer tasks whose opening said "a mixer under $200 all in, with shipping. Just
pick one that works." while the spec's buyer collects in store (fulfillment pickup, on_ask). The agent shipped, as
the opening said, and the judge failed it for the wrong fulfilment. The generator drew a total budget (worded "all
in, with shipping") for a persona that collects; the buyer draw now gives a collecting shopper an items-only budget
(carttrace.agents.buyer.draw_plan), and every task is checked here before it is written (carttrace.agents.tasks) or run again from
a written intent set (carttrace.agents.intents_custom.rechecked).

What contradicts. Words of the opening against one need of the spec, read conservatively (a missed contradiction is
better than a refused honest task):
- fulfilment: shipping or delivery wished for while the shopper collects ("all in, with shipping", "ship it",
  "deliver it"), collecting while the shopper has it delivered ("I'll collect it", "pick it up"), or express while
  the delivery is not; a question about the shipping policy, or whether the store ships somewhere, is not a wish;
- budget scope: "all in" or "with shipping" while the budget covers the items alone, or the spec has no budget;
- budget: an amount stated as a ceiling ("under $50", "no more than $50") that is not the spec's budget;
- deadline: "within N days" where the spec's deadline is not N days;
- quantity: a count the shopper asks for ("I'd like 3", "get me two") that is not the spec's;
- variant: with the candidates at hand, an option value of the spec's option that is not the value it asks for, and
  the asked value nowhere in the opening ("in red" when the spec asks for navy).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Final

from carttrace.agents.products import Product
from carttrace.agents.spec import HiddenSpec
from carttrace.agents.text import fold
from carttrace.core.money import to_minor

_SHIP_WISH: Final = re.compile(
    r"all[ -]in,? with shipping|with shipping\b|including shipping|shipping included|\bship (?:it|them|this|to me)\b"
    r"|\bdeliver(?:ed)? (?:it|them|to me|to my)\b|\bdelivered to\b|\bsent express\b|\bposted to\b",
    re.IGNORECASE,
)
_PICKUP_WISH: Final = re.compile(
    r"\bcollect (?:it|them|in store|from)\b|\bpick (?:it|them) up\b|\bpick up (?:in|at|from)\b|\bin[- ]store pickup\b"
    r"|\bclick and collect\b",
    re.IGNORECASE,
)
_EXPRESS_WISH: Final = re.compile(r"\b(?:sent|ship(?:ped)?|deliver(?:ed)?) express\b|\bexpress (?:delivery|shipping)\b",
                                  re.IGNORECASE)  # fmt: skip
_NOT_EXPRESS: Final = re.compile(r"\b(?:no|not|don't need|needn't be) (?:\w+ )?express\b", re.IGNORECASE)
_ALL_IN: Final = re.compile(
    r"(?:[$£€]\d[\d,.]*|budget(?: is)?|total)\s*(?:\w+\s)?all[ -]in\b(?![ -](?:one|1\b))"
    r"|\ball[ -]in,? (?:with|including) shipping|with shipping\b|including shipping|shipping and tax",
    re.IGNORECASE,
)
"""A budget that covers more than the items; "All in One Cream" and "all in one go" are not one."""
_CEILING: Final = re.compile(
    r"(?:under|no more than|less than|at most|up to|budget (?:is|of)|or less than|within)\s+"
    r"(?P<cur>[$£€]|PHP\s?|USD\s?|GBP\s?|EUR\s?)(?P<amount>\d[\d,]*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_OR_LESS: Final = re.compile(r"(?P<cur>[$£€])(?P<amount>\d[\d,]*(?:\.\d{1,2})?) or less", re.IGNORECASE)
_WITHIN_DAYS: Final = re.compile(r"\bwithin (?P<n>\d+|one|two|three|four|five|six|seven|ten|fourteen) days?\b",
                                 re.IGNORECASE)  # fmt: skip
_COUNT: Final = re.compile(
    r"\b(?:i'd like|i would like|i want|i need|get me|buy|order|i'll take)\s+"
    r"(?P<n>\d+|two|three|four|five|six|seven|eight|nine|ten)\b(?!\s*(?:-|%|ml|g\b|oz|kg|cm|mm|in\b|inch|pack|count|"
    r"ct\b|pcs|piece|x\b|per|of (?:your|the|these|those)\b))",
    re.IGNORECASE,
)
_NUMBERS: Final = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                   "ten": 10, "fourteen": 14}  # fmt: skip


def _number(text: str) -> int:
    return int(text) if text.isdigit() else _NUMBERS[text.lower()]


def _budget_minor(spec: HiddenSpec) -> tuple[int, str] | None:
    budget = next((p for p in spec.predicates if p.op == "max_price"), None)
    value = budget.value if budget is not None else None
    amount = value.get("amount") if isinstance(value, dict) else None
    if isinstance(value, dict) and isinstance(amount, int):
        return amount, str(value.get("currency") or "")
    return None


def contradictions(spec: HiddenSpec, opening: str, products: Sequence[Product] = ()) -> list[str]:
    """Every need of `spec` the opening's words contradict, each as a short reason; empty when none does. With
    `products` (the candidates' data), a variant option the opening names is checked too."""
    out: list[str] = []
    out += _fulfilment(spec, opening)
    out += _budget(spec, opening)
    out += _deadline(spec, opening)
    out += _quantity(spec, opening)
    if products:
        out += _variant(spec, opening, products)
    return out


def task_contradictions(task: Any, products: Sequence[Product] = ()) -> list[str]:
    """`contradictions` of a TaskRef (or anything with `opening` and `hidden_spec`)."""
    return contradictions(HiddenSpec.from_mapping(task.hidden_spec), str(task.opening), products)


def _fulfilment(spec: HiddenSpec, opening: str) -> list[str]:
    found = spec.buyer.fulfillment if spec.buyer is not None else None
    if found is None:
        return []
    value = found.value
    out = []
    shipped, collected = _SHIP_WISH.search(opening), _PICKUP_WISH.search(opening)
    if value == "pickup" and shipped is not None:
        out.append(f"fulfilment: the opening wants it shipped ({shipped.group(0)!r}), the spec collects")
    if value != "pickup" and collected is not None:
        out.append(f"fulfilment: the opening collects ({collected.group(0)!r}), the spec has it {value}")
    if value != "express" and _EXPRESS_WISH.search(opening) and not _NOT_EXPRESS.search(opening):
        out.append(f"fulfilment: the opening wants express, the spec has it {value}")
    return out


def _budget(spec: HiddenSpec, opening: str) -> list[str]:
    out = []
    scope = spec.buyer.budget_scope.value if spec.buyer is not None else "items"
    budget = _budget_minor(spec)
    if _ALL_IN.search(opening) and (budget is None or scope != "total"):
        out.append("budget scope: the opening's budget is all in, the spec's covers the items alone or is absent")
    if budget is not None:
        stated = [m for m in (*_CEILING.finditer(opening), *_OR_LESS.finditer(opening))]
        amounts = {a for m in stated if (a := to_minor(m.group("amount").replace(",", ""), budget[1] or "USD"))}
        if amounts and budget[0] not in amounts:
            out.append(f"budget: the opening's ceiling {sorted(amounts)} is not the spec's {budget[0]}")
    return out


def _deadline(spec: HiddenSpec, opening: str) -> list[str]:
    found = _WITHIN_DAYS.search(opening)
    if found is None:
        return []
    days = spec.buyer.fulfillment.deadline_days if spec.buyer is not None and spec.buyer.fulfillment else None
    said = _number(found.group("n"))
    return [] if days == said else [f"deadline: the opening says within {said} days, the spec's deadline is {days}"]


def _quantity(spec: HiddenSpec, opening: str) -> list[str]:
    count = next((p for p in spec.predicates if p.op == "count"), None)
    if count is None or not isinstance(count.value, int):
        return []
    said = {_number(m.group("n")) for m in _COUNT.finditer(opening)}
    if said and count.value not in said:
        return [f"quantity: the opening asks for {sorted(said)}, the spec for {count.value}"]
    return []


def _variant(spec: HiddenSpec, opening: str, products: Sequence[Product]) -> list[str]:
    text = f" {fold(opening)} "
    out = []
    candidates = set(spec.candidates)
    for p in spec.predicates:
        if p.op != "eq" or not isinstance(p.value, str) or not p.option:
            continue
        values = {
            v
            for prod in products
            if not candidates or prod.id in candidates
            for var in prod.variants
            if (v := var.option(p.option)) is not None
        }
        wanted = fold(p.value)
        if _named(wanted, text):
            continue
        others = sorted(v for v in values if fold(v) != wanted and len(fold(v)) > 1 and _asked(fold(v), text))
        if others:
            out.append(f"variant: the opening names {p.option} {others[0]!r}, the spec asks for {p.value!r}")
    return out


def _asked(value: str, text: str) -> bool:
    """Whether the opening asks for an option value ("in red", "the 32 one", "size m"), not merely says the word ("my
    current bottle is nearly empty")."""
    return (
        re.search(rf"(?:\bin|\bthe|\ba|\ban|\bsize|\bcolou?r|\bshade|\bpack|,) {re.escape(value)}(?![\w-])", text)
        is not None
    )


def _named(value: str, text: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", text) is not None


__all__ = ["contradictions", "task_contradictions"]

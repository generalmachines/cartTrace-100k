"""Totals, discounts, shipping and tax: the arithmetic every cart, checkout and order uses, in integers.

Two audit bugs live here. Duplicate discount codes stacked (`["WELCOME10"] * 10` was 100% off), because the old store
summed the percentage of every code it was sent; now a code applies once, and only one code applies unless the
environment pack says codes combine. And IN orders paid 18% tax on top of tax-inclusive prices; now tax is added only
in a market whose prices exclude it, and an inclusive market reports the tax already inside the price without adding
it. Every percentage goes through `carttrace.core.money`, so rounding is half up and identical on replay.

Platforms (D42). A coupon's own rules (fixed amounts, free shipping, individual use, a minimum spend) are the store's
data and hold everywhere; how codes combine, in what order they apply and how free shipping is offered are the
platform's (`CoreRules`). With the plain rules and percent codes only, every number here is what it always was.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from carttrace.core.money import included_tax, percent_of, to_major, whole_percent_of
from carttrace.sandbox.core.config import PLAIN_RULES, CoreRules, Coupon
from carttrace.sandbox.core.market import Market
from carttrace.sandbox.core.state import Group, Totals

FREE_RATE_SUFFIX = "_free"
"""A separate free rate's id is its paid rate's id with this suffix (`standard_free`)."""


class Priced(Protocol):
    @property
    def unit_price(self) -> int: ...

    @property
    def quantity(self) -> int: ...


def subtotal(lines: Iterable[Priced]) -> int:
    return sum(ln.unit_price * ln.quantity for ln in lines)


def resolve_codes(
    requested: Sequence[str],
    table: Mapping[str, int],
    *,
    stackable: bool = False,
    refused: bool = False,
    coupons: Mapping[str, Coupon] | None = None,
    subtotal: int | None = None,
    ordered: bool = False,
    case_sensitive: bool = False,
    silent_repeat: bool = False,
    replaces: bool = False,
) -> tuple[tuple[str, ...], list[tuple[str, str]]]:
    """(codes that apply, [(code, why it was refused)]). Codes are case-insensitive, and a repeat is refused as a
    duplicate, unless the platform matches codes exactly as the store wrote them (`case_sensitive`) or lets a repeat
    pass quietly (`silent_repeat`), as Medusa does (ops/playgrounds/medusa/captures/store-api 43, 44).

    With `coupons` (code to rule) a code's own rules hold: a minimum spend over `subtotal` is refused as
    `minimum_not_met`; an individual-use code drops every code applied before it (each refused `not_combinable`) and
    lets none join after it. `ordered` applies codes by their rule's priority, and a `stop` rule refuses every code
    after it (`stopped`)."""
    rules = coupons if coupons is not None else {c: Coupon(c, "percent", int(p)) for c, p in table.items()}
    applied: list[str] = []
    refusals: list[tuple[str, str]] = []
    seen: set[str] = set()
    alone = False
    stopped = False
    raws = [str(r) for r in requested]
    if ordered:
        raws = sorted(raws, key=lambda r: rules[_norm(r)].priority if _norm(r) in rules else 0)
    for raw in raws:
        code = _norm(raw)
        if code in seen:
            if not silent_repeat:
                refusals.append((raw, "duplicate"))
            continue
        seen.add(code)
        rule = rules.get(code)
        if case_sensitive and raw.strip() != code:
            rule = None
        if refused or rule is None:
            refusals.append((raw, "invalid"))
        elif stopped:
            refusals.append((raw, "stopped"))
        elif rule.min_spend is not None and subtotal is not None and subtotal < rule.min_spend:
            refusals.append((raw, "minimum_not_met"))
        elif replaces and applied:
            applied = [code]  # Saleor: the newest code takes the order's one place (storefront 36, 42)
        elif alone or (applied and not stackable and not rule.individual_use):
            refusals.append((raw, "not_combinable"))
        else:
            if rule.individual_use:
                refusals += [(dropped, "not_combinable") for dropped in applied]
                applied = []
                alone = True
            applied.append(code)
            stopped = ordered and rule.stop
    return tuple(applied), refusals


def _norm(code: str) -> str:
    return str(code).strip().upper()


def discount(
    sub: int,
    codes: Iterable[str],
    table: Mapping[str, int],
    coupons: Mapping[str, Coupon] | None = None,
    *,
    compound: bool = False,
) -> int:
    """The merchandise discount of `codes`. Percent codes only (the shorthand table, or percent coupons): one rounding
    of the summed percentage, as always. With a fixed or free-shipping coupon among them, or where the platform
    compounds codes (`compound`, Medusa): each code's own amount, capped by what is left."""
    listed = list(codes)
    every_percent = coupons is None or all(coupons[c].kind == "percent" for c in listed if c in coupons)
    if every_percent and not compound:
        pct = sum(int(coupons[c].amount) if coupons and c in coupons else int(table.get(c, 0)) for c in listed)
        return min(sub, whole_percent_of(sub, min(pct, 100)))
    rules = coupons if coupons is not None else {c: Coupon(c, "percent", int(p)) for c, p in table.items()}
    return sum(code_amounts(sub, listed, rules, compound=compound))


def code_amounts(sub: int, codes: Sequence[str], coupons: Mapping[str, Coupon], *, compound: bool = False) -> list[int]:
    """Each code's own amount off `sub`, in order, never more than what is left (WooCommerce: WELCOME10 800 and FIVEOFF
    500 off 79.99, captures/diff-woo 64). With `compound` a percent code takes its share of what the codes before it
    left, not of `sub` (Medusa, ops/playgrounds/medusa/captures/store-api 48)."""
    out: list[int] = []
    left = sub
    for c in codes:
        rule = coupons.get(c)
        if rule is None or rule.kind == "free_shipping":
            amount = 0
        elif rule.kind == "percent":
            amount = whole_percent_of(left if compound else sub, min(rule.amount, 100))
        else:
            amount = rule.amount
        amount = max(0, min(amount, left))
        left -= amount
        out.append(amount)
    return out


def shares(
    codes: Sequence[str], total: int, sub: int, coupons: Mapping[str, Coupon], *, compound: bool = False
) -> list[int]:
    """Each applied code's share of a discount of `total`: split by percentage when every code is a percent one (the
    last code takes the rounding), else, or where codes compound, each code's own amount."""
    if not codes:
        return []
    if compound:
        return code_amounts(sub, codes, coupons, compound=True)
    if all(coupons.get(c, Coupon(c)).kind == "percent" for c in codes):
        pct = [int(coupons[c].amount) if c in coupons else 0 for c in codes]
        whole = sum(pct)
        out: list[int] = []
        left = total
        for i, p in enumerate(pct):
            share = left if i == len(codes) - 1 else (total * p // whole if whole else 0)
            left -= share
            out.append(share)
        return out
    return code_amounts(sub, codes, coupons)


def free_shipping_code(codes: Iterable[str], coupons: Mapping[str, Coupon]) -> bool:
    """Whether an applied code is a free-shipping coupon."""
    return any(c in coupons and coupons[c].kind == "free_shipping" for c in codes)


def shipping_options(
    market: Market,
    group_subtotal: int,
    *,
    rules: CoreRules = PLAIN_RULES,
    after_discount: int | None = None,
    free_code: bool = False,
) -> tuple[dict[str, Any], ...]:
    """The options a shipping group is offered, priced for the merchandise it carries.

    An option with a free-above threshold is free when the merchandise reaches it (after discounts where the platform
    compares it so, `after_discount`) or a free-shipping code is applied. Where the platform lists free shipping as a
    rate of its own (`separate_free_rate`), the paid option keeps its price and the free rate follows it, titled as
    the reference store titles it ("Standard (free from 50.00)"). A platform whose threshold counts the tax
    (`free_threshold_with_tax`, Medusa) adds it to the basis."""
    basis = after_discount if rules.free_after_discount and after_discount is not None else group_subtotal
    if rules.free_threshold_with_tax and not market.tax_inclusive:
        basis += percent_of(basis, market.tax_bp)
    out = []
    for o in market.shipping:
        qualifies = o.free_above is not None and (basis >= o.free_above or free_code)
        if not rules.separate_free_rate:
            out.append({"id": o.id, "title": o.title, "carrier": o.carrier, "amount": 0 if qualifies else o.amount})
            continue
        out.append({"id": o.id, "title": o.title, "carrier": o.carrier, "amount": o.amount})
        if qualifies and o.free_above is not None:
            title = f"{o.title} (free from {to_major(o.free_above, market.currency)})"
            out.append({"id": f"{o.id}{FREE_RATE_SUFFIX}", "title": title, "carrier": o.carrier, "amount": 0})
    return tuple(out)


def pickup_options(market: Market) -> tuple[dict[str, Any], ...]:
    return tuple({"id": p.id, "title": p.title, "carrier": "pickup", "amount": 0} for p in market.pickup)


def totals(
    lines: Sequence[Priced],
    codes: Sequence[str],
    table: Mapping[str, int],
    market: Market,
    groups: Sequence[Group] | None = None,
    *,
    coupons: Mapping[str, Coupon] | None = None,
    rules: CoreRules = PLAIN_RULES,
) -> Totals:
    """A cart's totals when `groups` is None; a checkout's or an order's otherwise. Fulfillment and tax stay None until
    every group has a selected option, because a store cannot quote shipping it has not chosen; a platform that taxes
    by region (`tax_upfront`, Medusa) charges the merchandise's tax from the first line, carts included, and one that
    taxes shipping (`tax_on_shipping`) adds the rate's tax once it is chosen. `coupons` are the codes' rules where
    they are more than the percent `table`."""
    sub = subtotal(lines)
    disc = discount(sub, codes, table, coupons, compound=rules.compound_codes)
    net = sub - disc
    inside = included_tax(net, market.tax_bp) if market.tax_inclusive else 0
    upfront = None if market.tax_inclusive or not rules.tax_upfront else percent_of(net, market.tax_bp)
    if groups is None:
        if upfront is not None:
            return Totals(subtotal=sub, discount=disc, tax=upfront, tax_included=inside, total=net + upfront)
        return Totals(subtotal=sub, discount=disc, tax_included=inside, total=net)
    amounts = [g.amount for g in groups]
    fulfillment = sum(a for a in amounts if a is not None) if amounts and None not in amounts else None
    tax = upfront
    if fulfillment is not None:
        tax = 0 if market.tax_inclusive else percent_of(net, market.tax_bp)
        if rules.tax_on_shipping and not market.tax_inclusive:
            tax += percent_of(fulfillment, market.tax_bp)
    return Totals(
        subtotal=sub,
        discount=disc,
        fulfillment=fulfillment,
        tax=tax,
        tax_included=inside,
        total=net + (fulfillment or 0) + (tax or 0),
    )


def adds_up(t: Totals, lines: Sequence[Priced]) -> list[str]:
    """Why these totals do not add up, or [] when they do."""
    problems = []
    sub = subtotal(lines)
    if t.subtotal != sub:
        problems.append(f"subtotal {t.subtotal} != sum of lines {sub}")
    if not 0 <= t.discount <= t.subtotal:
        problems.append(f"discount {t.discount} outside 0..{t.subtotal}")
    if t.fulfillment is not None and t.fulfillment < 0:
        problems.append("negative fulfillment")
    if t.tax is not None and t.tax < 0:
        problems.append("negative tax")
    if t.total != t.subtotal - t.discount + (t.fulfillment or 0) + (t.tax or 0):
        problems.append(f"total {t.total} != subtotal - discount + fulfillment + tax")
    if t.total < 0:
        problems.append("negative total")
    return problems

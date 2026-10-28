"""Checkouts: fulfillment groups, readiness, and the commands that create, update and cancel them.

A checkout is ready for completion only when its facts say so (a buyer email, every line in a fulfillment group with a
chosen option, a destination for every shipping group, stock held), and `settle` moves it to ready or back to
incomplete after every change. A checkout waiting on a human leaves escalation through any update that makes it whole
again; if it was waiting on the buyer's 3-D Secure step, the abandoned challenge's payment is voided on the way out.

How rates are offered is the platform's (`CoreRules`, D42): WooCommerce quotes them before any address and selects the
first, lists free shipping as a rate of its own once the merchandise after discounts reaches the threshold (or a
free-shipping code is applied), and keeps a destination it does not ship to with no rates at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from carttrace.sandbox.core import pricing
from carttrace.sandbox.core.basket import apply_codes, build_lines, merge_buyer, new_cart, reprice
from carttrace.sandbox.core.commands import CancelCheckout, CreateCheckout, GroupIn, UpdateCheckout
from carttrace.sandbox.core.context import Ctx
from carttrace.sandbox.core.errors import BusinessRule, IllegalTransition
from carttrace.sandbox.core.fault_effects import fire, peek
from carttrace.sandbox.core.handlers.rails import answer_request, void_open_requests
from carttrace.sandbox.core.machines import OPEN_CHECKOUT
from carttrace.sandbox.core.state import Cart, Checkout, Group, Line, Totals
from carttrace.sandbox.core.stock import held_for, release, rereserve, reserve

# -- fulfillment groups ------------------------------------------------------------------------------------------------


def resolve_groups(ctx: Ctx, ck: Checkout, updates: Sequence[GroupIn] | None) -> tuple[Group, ...]:
    """The checkout's groups after `updates`, with lines placed and options priced. Groups are merged by id."""
    existing = ck.groups
    if not existing and ctx.cfg.rules.default_shipping_group and ck.lines:
        existing = (Group("g1", "shipping", ()),)
    groups = {g.id: g for g in existing}
    order = [g.id for g in existing]
    asked: list[tuple[str, str]] = []
    line_ids = {ln.id for ln in ck.lines}
    for gi in updates or ():
        only = order[0] if len(order) == 1 and groups[order[0]].method == gi.method else None
        gid = gi.id or only or f"g{len(order) + 1}"
        g = groups.get(gid) or Group(gid, gi.method, ())
        if gid not in groups:
            order.append(gid)
        groups[gid] = _apply_update(ctx, g, gi, line_ids, asked)
    return _place_lines(ctx, ck, [groups[gid] for gid in order], asked)


def _unshippable(ctx: Ctx) -> set[str]:
    out = set(ctx.cfg.market.unshippable)
    for r in peek(ctx, "unshippable"):
        out |= {str(x).upper() for x in r.params.get("countries") or ()}
    return out


def _fire_unshippable(ctx: Ctx, country: str) -> None:
    """A destination an `unshippable` fault refuses fires it, as a sealed `fault.fired` (D66): the fault is standing and
    never consumed, so without the record a judge could not tell a fault that met an address from one that never
    did."""
    for rule in peek(ctx, "unshippable"):
        if country in {str(x).upper() for x in rule.params.get("countries") or ()}:
            fire(ctx, rule)


def _apply_update(ctx: Ctx, g: Group, gi: GroupIn, line_ids: set[str], asked: list[tuple[str, str]]) -> Group:
    if gi.method != g.method:
        g = Group(g.id, gi.method, ())
    if gi.line_ids is not None:
        for x in gi.line_ids:
            if x not in line_ids:
                ctx.emit("fulfillment.rejected", group_id=g.id, line_id=x, reason="line_not_found")
        g = replace(g, line_ids=tuple(x for x in gi.line_ids if x in line_ids), explicit_lines=True)
    if gi.destination is not None:
        d = gi.destination.model_dump(exclude_none=True)
        _fire_unshippable(ctx, str(d["address_country"]).upper())
        if d["address_country"] in _unshippable(ctx) and not ctx.cfg.rules.unshippable_no_rates:
            ctx.emit(
                "fulfillment.rejected", group_id=g.id, country=d["address_country"], reason="destination_not_shippable"
            )
        else:
            d.setdefault("id", "d1")
            g = replace(g, destination=d)
    if gi.location_id is not None:
        g = _located(ctx, g, gi.location_id)
    if gi.selected_option_id is not None:
        g = replace(g, selected_option_id=gi.selected_option_id)
        asked.append((g.id, gi.selected_option_id))
    return g


def _located(ctx: Ctx, g: Group, location_id: str) -> Group:
    """The group collecting at `location_id`, or unchanged with the refusal logged: an unknown counter, or one in a
    country the store does not serve (D110)."""
    place = ctx.cfg.market.location(location_id)
    if place is None:
        ctx.emit("fulfillment.rejected", group_id=g.id, location_id=location_id, reason="invalid_location")
        return g
    _fire_unshippable(ctx, place.address_country.upper())
    if place.address_country.upper() in _unshippable(ctx):
        ctx.emit(
            "fulfillment.rejected",
            group_id=g.id,
            location_id=location_id,
            country=place.address_country,
            reason="destination_not_shippable",
        )
        return g
    return replace(g, location_id=location_id, selected_option_id=location_id)


def _place_lines(ctx: Ctx, ck: Checkout, ordered: list[Group], asked: list[tuple[str, str]]) -> tuple[Group, ...]:
    """Named lines go to the groups that named them; every other line goes to the first group that named none."""
    line_ids = {ln.id for ln in ck.lines}
    claimed: set[str] = set()
    resolved: list[Group] = []
    for g in ordered:
        if g.explicit_lines:
            mine = tuple(x for x in g.line_ids if x in line_ids and x not in claimed)
            claimed |= set(mine)
            resolved.append(replace(g, line_ids=mine))
        else:
            resolved.append(g)
    rest = tuple(ln.id for ln in ck.lines if ln.id not in claimed)
    first_auto = True
    out: list[Group] = []
    for g in resolved:
        placed = g
        if not g.explicit_lines:
            placed = replace(g, line_ids=rest if first_auto else ())
            first_auto = False
        out.append(_priced(ctx, ck, placed, asked))
    return tuple(out)


def _priced(ctx: Ctx, ck: Checkout, g: Group, asked: list[tuple[str, str]]) -> Group:
    sub = pricing.subtotal(ln for ln in ck.lines if ln.id in g.line_ids)
    rules = ctx.cfg.rules
    shipping = g.method == "shipping"
    opts = _shipping_options(ctx, ck, g, sub) if shipping else _pickup_options(ctx)
    sel = g.selected_option_id
    chosen = next((o for o in opts if o["id"] == sel), None) if sel else None
    if sel and chosen is None and (g.id, sel) in asked:
        ctx.emit("fulfillment.rejected", group_id=g.id, option_id=sel, reason="invalid_option")
    if chosen is None and rules.preselect_first_rate and shipping and opts and g.line_ids:
        chosen = opts[0]
    if chosen is None and rules.cheapest_on_address and shipping and opts and g.line_ids and g.destination:
        chosen = min(opts, key=lambda o: int(o["amount"]))
    if g.method == "pickup" and chosen is not None:
        g = replace(g, location_id=chosen["id"])
    return replace(
        g,
        options=opts,
        selected_option_id=chosen["id"] if chosen else None,
        amount=chosen["amount"] if chosen else None,
    )


def _pickup_options(ctx: Ctx) -> tuple[dict[str, Any], ...]:
    """The store's counters a buyer may collect from: none in a country the store does not serve. Why (D110): with an
    `unshippable` fault on the buyer's country the replica still offered its counter there, so on ep-b7053b211a03 an
    agent switched a delivery the shopper asked for to a collection and the order of a row designed not to complete was
    placed. A store that will not ship to a country keeps no counter in it."""
    blocked = _unshippable(ctx)
    market = ctx.cfg.market
    keep = {p.id for p in market.pickup if p.address_country.upper() not in blocked}
    return tuple(o for o in pricing.pickup_options(market) if o["id"] in keep)


# -- readiness ---------------------------------------------------------------------------------------------------------


def _shipping_options(ctx: Ctx, ck: Checkout, g: Group, sub: int) -> tuple[dict[str, Any], ...]:
    """The rates a shipping group is offered: none before an address unless the platform quotes the store's own
    country, none for a destination the store does not ship to (a platform that keeps it), else the market's, with
    free shipping judged on the merchandise after the group's share of the discount."""
    rules = ctx.cfg.rules
    if not g.destination and not rules.rates_before_address:
        return ()
    country = str((g.destination or {}).get("address_country") or "").upper()
    if rules.unshippable_no_rates and country in _unshippable(ctx):
        return ()
    coupons = ctx.cfg.coupon_table()
    whole = pricing.subtotal(ck.lines)
    disc = pricing.discount(whole, ck.codes, ctx.cfg.discount_codes, coupons, compound=rules.compound_codes)
    after = sub - (disc * sub // whole if whole else 0)
    return pricing.shipping_options(
        ctx.cfg.market,
        sub,
        rules=rules,
        after_discount=after,
        free_code=pricing.free_shipping_code(ck.codes, coupons),
    )


def checkout_totals(ctx: Ctx, ck: Checkout) -> Totals:
    return pricing.totals(
        ck.lines,
        ck.codes,
        ctx.cfg.discount_codes,
        ctx.cfg.market,
        ck.groups,
        coupons=ctx.cfg.coupon_table(),
        rules=ctx.cfg.rules,
    )


def ready(ctx: Ctx, ck: Checkout) -> bool:
    if not ck.buyer.get("email") or not ck.lines or not ck.groups:
        return False
    res = ctx.tx.reservations.get(ck.reservation_id)
    if res is None or res.status != "held":
        return False
    covered = [x for g in ck.groups for x in g.line_ids]
    if sorted(covered) != sorted(ln.id for ln in ck.lines):
        return False
    for g in ck.groups:
        if g.line_ids and g.amount is None:
            return False
        if g.method == "shipping" and g.line_ids and not g.destination:
            return False
    return True


def void_pending_challenge(ctx: Ctx, ck: Checkout) -> None:
    """Void the payment parked on the buyer's 3-D Secure step, when the checkout moves on without it; the payment
    request it answered fails with it."""
    pay = ctx.tx.payments.get(ck.payment_id)
    if pay is not None and pay.status == "requires_action":
        answer_request(ctx, ctx.transition(ctx.tx.payments, pay, "voided"), "failed")


def settle(ctx: Ctx, ck: Checkout) -> Checkout:
    """Recompute groups and totals, store the checkout, and move it to ready or incomplete as its facts say."""
    if ck.escalation == "payment_action_required":
        void_pending_challenge(ctx, ck)
    ck = replace(ck, groups=resolve_groups(ctx, ck, None))
    ck = replace(ck, totals=checkout_totals(ctx, ck))
    ctx.tx.checkouts.put(ck.id, ck)
    target = "ready_for_complete" if ready(ctx, ck) else "incomplete"
    return ctx.transition(ctx.tx.checkouts, ck, target, escalation=None)


def open_checkout(ctx: Ctx, checkout_id: str) -> Checkout:
    ck = ctx.require(ctx.tx.checkouts, checkout_id, "checkout_not_found")
    ctx.refs["checkout_id"] = ck.id
    if ck.status not in OPEN_CHECKOUT:
        raise IllegalTransition(code="checkout_locked", message=f"checkout is {ck.status}", field="checkout_id")
    return ck


def cancel_open_checkout(ctx: Ctx, ck: Checkout, reason: str) -> None:
    """Cancel a checkout: release its hold, void a pending challenge, and say why."""
    res = ctx.tx.reservations.get(ck.reservation_id)
    if res is not None:
        release(ctx, res, "released")
    void_pending_challenge(ctx, ck)
    void_open_requests(ctx, ck.id)
    ctx.transition(ctx.tx.checkouts, ck, "canceled", escalation=None)
    ctx.emit("checkout.canceled_because", id=ck.id, reason=reason)


# -- commands ----------------------------------------------------------------------------------------------------------


def create_checkout(ctx: Ctx, c: CreateCheckout) -> None:
    cart = _source_cart(ctx, c)
    kid = ctx.new_id("checkout")
    lines = reprice(ctx, cart.lines)
    if c.lines is not None and c.cart_id is not None:
        lines = build_lines(ctx, lines, c.lines, kid)
    if not lines:
        if any(e.type == "line.rejected" and e.data.get("reason") in ("out_of_stock", "sold_out") for e in ctx.events):
            raise BusinessRule(code="merchandise_out_of_stock", message="nothing asked for is in stock", field="lines")
        raise BusinessRule(code="empty_cart", message="a checkout needs at least one line item", field="lines")
    prev = ctx.tx.checkouts.get(cart.checkout_id)
    if prev is not None and prev.status in OPEN_CHECKOUT:
        cancel_open_checkout(ctx, prev, reason="superseded")
    held_lines = _holdable(ctx, kid, lines)
    if not held_lines:
        raise BusinessRule(
            code="merchandise_out_of_stock", message="nothing in this cart can be reserved", field="lines"
        )
    codes = cart.codes if c.codes is None or c.cart_id is None else apply_codes(ctx, c.codes, kid, lines=lines)
    ck = Checkout(
        kid,
        c.session,
        cart.id,
        "incomplete",
        held_lines,
        codes,
        merge_buyer(cart.buyer, c.buyer),
        c.country or cart.country,
        (),
        Totals(),
        ctx.now,
    )
    res = reserve(ctx, kid, ck.lines)
    ck = replace(ck, reservation_id=res.id if res else None)
    ctx.tx.checkouts.put(kid, ck)
    ctx.emit("checkout.created", id=kid, cart_id=cart.id, session=c.session)
    if cart.status == "active":
        ctx.transition(ctx.tx.carts, cart, "converted", checkout_id=kid)
    else:
        ctx.tx.carts.put(cart.id, replace(cart, checkout_id=kid))
    ck = replace(ck, groups=resolve_groups(ctx, ck, c.fulfillment))
    ctx.tx.checkouts.put(kid, ck)
    settle(ctx, ck)
    ctx.refs.update(checkout_id=kid, cart_id=cart.id)


def _source_cart(ctx: Ctx, c: CreateCheckout) -> Cart:
    if c.cart_id is None:
        return new_cart(ctx, c.session, lines=c.lines or [], codes=c.codes or [], buyer=c.buyer, country=c.country)
    cart = ctx.require(ctx.tx.carts, c.cart_id, "cart_not_found")
    if cart.status == "canceled":
        raise IllegalTransition(
            code="cart_canceled", message=f"cart {cart.id} is canceled; it cannot be checked out", field="cart_id"
        )
    return cart


def _holdable(ctx: Ctx, owner: str, lines: tuple[Line, ...]) -> tuple[Line, ...]:
    """The lines whose stock can be held, line by line: a line that cannot be held is refused, the rest go on."""
    held: list[Line] = []
    for ln in lines:
        v = ctx.variant(ln.variant_id)
        if v.sellable >= ln.quantity + sum(x.quantity for x in held if x.variant_id == ln.variant_id):
            held.append(ln)
        else:
            ctx.emit(
                "line.rejected",
                owner=owner,
                variant_id=ln.variant_id,
                reason="out_of_stock",
                requested=ln.quantity,
                available=v.sellable,
            )
    return tuple(held)


def update_checkout(ctx: Ctx, c: UpdateCheckout) -> None:
    ck = open_checkout(ctx, c.checkout_id)
    ck = replace(ck, buyer=merge_buyer(ck.buyer, c.buyer))
    if c.lines is not None:
        ck = replace(ck, lines=build_lines(ctx, ck.lines, c.lines, ck.id, held=held_for(ctx, ck)))
        ck = rereserve(ctx, ck)
    if c.codes is not None:
        ck = replace(ck, codes=apply_codes(ctx, c.codes, ck.id, lines=ck.lines))
    ck = replace(ck, groups=resolve_groups(ctx, ck, c.fulfillment))
    ctx.tx.checkouts.put(ck.id, ck)
    ctx.emit("checkout.updated", id=ck.id)
    settle(ctx, ck)


def cancel_checkout(ctx: Ctx, c: CancelCheckout) -> None:
    ck = ctx.require(ctx.tx.checkouts, c.checkout_id, "checkout_not_found")
    ctx.refs["checkout_id"] = ck.id
    cancel_open_checkout(ctx, ck, reason="requested")

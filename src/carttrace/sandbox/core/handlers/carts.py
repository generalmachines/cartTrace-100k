"""Carts: create, update, cancel, and the environment's hand-off of a live cart into the sandbox.

Why a hand-off command. A live journey ends at a verified cart, because nothing is bought on a live store. The sandbox
then has to start from exactly that cart: the same variants, quantities and unit prices the shopper accepted. The
hand-off goes through `apply` like everything else, so it is in the event log and a replay reproduces it, and it is
refused as a whole if the sandbox cannot reproduce any line exactly, rather than starting checkout from a different
cart than the one the shopper saw.
"""

from __future__ import annotations

from dataclasses import replace

from carttrace.sandbox.core.basket import apply_codes, build_lines, cart_totals, merge_buyer, new_cart, reprice
from carttrace.sandbox.core.commands import CancelCart, CreateCart, HandoffCart, LineIn, UpdateCart
from carttrace.sandbox.core.context import Ctx
from carttrace.sandbox.core.errors import BusinessRule, IllegalTransition
from carttrace.sandbox.core.handlers.checkouts import cancel_open_checkout
from carttrace.sandbox.core.machines import OPEN_CHECKOUT


def create_cart(ctx: Ctx, c: CreateCart) -> None:
    new_cart(ctx, c.session, lines=c.lines, codes=c.codes, buyer=c.buyer, country=c.country)


def update_cart(ctx: Ctx, c: UpdateCart) -> None:
    cart = ctx.require(ctx.tx.carts, c.cart_id, "cart_not_found")
    if cart.status == "canceled":
        raise IllegalTransition(code="cart_canceled", message=f"cart {cart.id} is canceled", field="cart_id")
    ctx.refs["cart_id"] = cart.id
    if c.lines is not None:
        cart = replace(cart, lines=build_lines(ctx, cart.lines, c.lines, cart.id))
    if c.codes is not None:
        cart = replace(cart, codes=apply_codes(ctx, c.codes, cart.id, lines=cart.lines))
    cart = cart_totals(ctx, replace(cart, buyer=merge_buyer(cart.buyer, c.buyer), lines=reprice(ctx, cart.lines)))
    ctx.tx.carts.put(cart.id, cart)
    ctx.transition(ctx.tx.carts, cart, cart.status)
    ctx.emit("cart.updated", id=cart.id, lines=len(cart.lines))


def cancel_cart(ctx: Ctx, c: CancelCart) -> None:
    cart = ctx.require(ctx.tx.carts, c.cart_id, "cart_not_found")
    ctx.refs["cart_id"] = cart.id
    cart = ctx.transition(ctx.tx.carts, cart, "canceled")
    ck = ctx.tx.checkouts.get(cart.checkout_id)
    if ck is not None and ck.status in OPEN_CHECKOUT:  # a canceled cart cannot go on to complete its checkout
        cancel_open_checkout(ctx, ck, reason="cart_canceled")


def handoff_cart(ctx: Ctx, c: HandoffCart) -> None:
    if c.currency != ctx.tx.currency:
        raise BusinessRule(
            code="handoff_currency_mismatch",
            message=f"the live cart is in {c.currency}; this sandbox prices in {ctx.tx.currency}",
            field="currency",
            hint="build the envpack from the same cart (carttrace.sandbox.envpack.build_envpack_from_cart)",
        )
    want: dict[str, int] = {}
    price: dict[str, int] = {}
    for ln in c.lines:
        if price.setdefault(ln.variant_id, ln.unit_price) != ln.unit_price:
            raise BusinessRule(
                code="handoff_mismatch", message=f"{ln.variant_id} has two prices in one live cart", field="lines"
            )
        want[ln.variant_id] = want.get(ln.variant_id, 0) + ln.quantity
    lines_in = [LineIn(variant_id=vid, quantity=q) for vid, q in want.items()]
    cart = new_cart(ctx, c.session, lines=lines_in, buyer=c.buyer, country=c.country)
    got = {ln.variant_id: ln for ln in cart.lines}
    problems = []
    for vid, q in want.items():
        line = got.get(vid)
        if line is None or line.quantity != q:
            problems.append({"variant_id": vid, "reason": "not_reproduced", "quantity": q})
        elif line.unit_price != price[vid]:
            problems.append(
                {"variant_id": vid, "reason": "price_differs", "live": price[vid], "sandbox": line.unit_price}
            )
    if problems:
        raise BusinessRule(
            code="handoff_mismatch",
            message=f"the sandbox cannot reproduce {len(problems)} line(s) of the live cart",
            field="lines",
            details=problems,
            hint="build the envpack from the same cart so its snapshot holds every line at the live price",
        )
    ctx.emit(
        "cart.handed_off",
        id=cart.id,
        session=c.session,
        source=c.source,
        cart_ref=c.cart_ref,
        lines=[[vid, q, price[vid]] for vid, q in want.items()],
    )

"""The merchant's side of an order: fulfil, cancel, refund, edit and returns, with money, stock and webhooks moving
together.

Why together. A refund that leaves stock unreturned, or a cancel that never refunds, is exactly the kind of quiet
inconsistency a grader cannot see from the agent's transcript. Each command here changes the order, the payment and
the shelf in one commit, and the invariants hold refunds within what was captured.
"""

from __future__ import annotations

from dataclasses import replace

from carttrace.sandbox.core import pricing
from carttrace.sandbox.core.commands import CancelOrder, EditOrder, FulfillOrder, ReceiveReturn, RefundOrder, RequestReturn
from carttrace.sandbox.core.context import Ctx
from carttrace.sandbox.core.errors import BusinessRule, IllegalTransition, NotFound
from carttrace.sandbox.core.outbox import webhook
from carttrace.sandbox.core.state import Order

DAY_MS = 86_400_000


def _order(ctx: Ctx, order_id: str) -> Order:
    o = ctx.require(ctx.tx.orders, order_id, "order_not_found")
    ctx.refs["order_id"] = o.id
    return o


def _refund(ctx: Ctx, order: Order, amount: int) -> None:
    pay = ctx.require(ctx.tx.payments, order.payment_id, "payment_not_found")
    room = pay.captured - pay.refunded
    if amount > room:
        raise BusinessRule(
            code="refund_exceeds_capture", message=f"refund {amount} exceeds the {room} still captured", field="amount"
        )
    if amount <= 0:
        return
    refunded = pay.refunded + amount
    status = "refunded" if refunded == pay.captured else "partially_refunded"
    ctx.transition(ctx.tx.payments, pay, status, refunded=refunded)
    ctx.emit("refund.created", order_id=order.id, payment_id=pay.id, amount=amount)
    webhook(ctx, "order.refund_created", order.id)


def _restock(ctx: Ctx, variant_id: str, quantity: int) -> None:
    v = ctx.variant(variant_id)
    ctx.tx.inventory.put(variant_id, replace(v, on_hand=v.on_hand + quantity))
    ctx.emit("inventory.restocked", variant_id=variant_id, quantity=quantity)


def _refundable(ctx: Ctx, order: Order) -> int:
    pay = ctx.require(ctx.tx.payments, order.payment_id, "payment_not_found")
    return pay.captured - pay.refunded


def fulfill_order(ctx: Ctx, c: FulfillOrder) -> None:
    o = _order(ctx, c.order_id)
    ids = set(c.line_ids or [ln.id for ln in o.lines])
    unknown = ids - {ln.id for ln in o.lines}
    if unknown:
        raise NotFound(code="line_not_found", message=f"no order line {sorted(unknown)[0]!r}", field="line_ids")
    lines = tuple(replace(ln, fulfilled=ln.quantity) if ln.id in ids else ln for ln in o.lines)
    full = all(ln.fulfilled == ln.quantity for ln in lines)
    ctx.transition(ctx.tx.orders, o, "fulfilled" if full else "partially_fulfilled", lines=lines)
    ctx.emit("fulfillment.created", order_id=o.id, line_ids=sorted(ids))
    webhook(ctx, "order.fulfillment_created", o.id)


def cancel_order(ctx: Ctx, c: CancelOrder) -> None:
    o = _order(ctx, c.order_id)
    ctx.transition(ctx.tx.orders, o, "canceled")
    for ln in o.lines:
        _restock(ctx, ln.variant_id, ln.quantity)
    webhook(ctx, "order.canceled", o.id)
    _refund(ctx, o, _refundable(ctx, o))


def refund_order(ctx: Ctx, c: RefundOrder) -> None:
    o = _order(ctx, c.order_id)
    if o.status == "canceled":
        raise IllegalTransition(code="order_canceled", message="a canceled order was already refunded")
    _refund(ctx, o, c.amount)


def edit_order(ctx: Ctx, c: EditOrder) -> None:
    o = _order(ctx, c.order_id)
    line = next((ln for ln in o.lines if ln.id == c.line_id), None)
    if line is None:
        raise NotFound(code="line_not_found", message=f"no order line {c.line_id!r}", field="line_id")
    if c.quantity >= line.quantity:
        raise BusinessRule(
            code="edit_increase_unsupported", message="an order edit may only reduce a quantity", field="quantity"
        )
    edited = (replace(x, quantity=c.quantity) if x.id == line.id else x for x in o.lines)
    lines = tuple(ln for ln in edited if ln.quantity > 0)
    if not lines:
        raise BusinessRule(
            code="edit_empties_order",
            message="an edit cannot remove every line",
            hint="cancel the order instead",
        )
    totals = pricing.totals(
        lines,
        o.codes,
        ctx.cfg.discount_codes,
        ctx.cfg.market,
        o.groups,
        coupons=ctx.cfg.coupon_table(),
        rules=ctx.cfg.rules,
    )
    ctx.transition(ctx.tx.orders, o, "confirmed", lines=lines, totals=totals)
    _restock(ctx, line.variant_id, line.quantity - c.quantity)
    ctx.emit("order.edited", id=o.id, line_id=line.id, quantity=c.quantity, total=totals.total)
    webhook(ctx, "order.edited", o.id)
    _refund(ctx, o, max(0, o.totals.total - totals.total))


def request_return(ctx: Ctx, c: RequestReturn) -> None:
    o = _order(ctx, c.order_id)
    days = ctx.cfg.market.return_window_days
    if ctx.now > o.placed_at + days * DAY_MS:
        raise BusinessRule(code="return_window_closed", message=f"returns close {days} days after the order")
    ids = tuple(c.line_ids or [ln.id for ln in o.lines])
    ctx.transition(ctx.tx.orders, o, "return_requested", return_line_ids=ids)
    webhook(ctx, "order.return_requested", o.id)


def receive_return(ctx: Ctx, c: ReceiveReturn) -> None:
    o = _order(ctx, c.order_id)
    returned = tuple(replace(ln, returned=ln.quantity) if ln.id in o.return_line_ids else ln for ln in o.lines)
    o2 = ctx.transition(ctx.tx.orders, o, "returned", lines=returned)
    value = 0
    for ln in o.lines:
        if ln.id in o.return_line_ids:
            _restock(ctx, ln.variant_id, ln.quantity)
            value += ln.unit_price * ln.quantity
    webhook(ctx, "order.return_received", o2.id)
    _refund(ctx, o2, min(value, _refundable(ctx, o2)))

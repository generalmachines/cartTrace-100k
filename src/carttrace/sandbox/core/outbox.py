"""The webhook outbox: the order notifications a store sends, in order, append-only.

Why an outbox in the state. Webhooks are what a merchant's other systems react to, so a grader checks them like any
other fact: one `order.created` per order, a refund notification per refund. A `duplicate_webhook` fault delivers one
twice with the same id, the way real stores retry, so a consumer that is not idempotent shows up in the episode.
"""

from __future__ import annotations

from carttrace.sandbox.core.context import Ctx
from carttrace.sandbox.core.fault_effects import consume
from carttrace.sandbox.core.state import Webhook


def webhook(ctx: Ctx, topic: str, order_id: str) -> None:
    wid = ctx.new_id("wh")
    ctx.tx.add_webhook(Webhook(wid, topic, order_id, ctx.tx.seq))
    ctx.emit("webhook.enqueued", id=wid, topic=topic, order_id=order_id)
    if consume(ctx, "duplicate_webhook", ignore_on=True):
        ctx.tx.add_webhook(Webhook(wid, topic, order_id, ctx.tx.seq, delivery=2))
        ctx.emit("webhook.duplicated", sealed=True, id=wid, topic=topic, order_id=order_id)

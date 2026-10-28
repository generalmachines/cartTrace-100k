"""Stock reservations: a checkout holds its stock until it completes, is canceled, or the hold lapses.

Why holds. Two buyers of the last unit must not both complete. A checkout reserves every line or nothing; a held unit
is not sellable to anyone else; a hold expires after the pack's reservation window, and completion re-checks the shelf,
so a lapsed hold can never oversell. Every change to `reserved` goes through here, which keeps the invariant that a
variant's reserved count equals the sum of its held reservations.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from carttrace.sandbox.core.context import Ctx
from carttrace.sandbox.core.state import Checkout, Line, Reservation


def release(ctx: Ctx, res: Reservation, status: str) -> None:
    """Give a held reservation's stock back to the shelf and end it as `status` (released, expired, broken)."""
    if res.status != "held":
        return
    for vid, q in res.items:
        v = ctx.variant(vid)
        ctx.tx.inventory.put(vid, replace(v, reserved=v.reserved - q))
    ctx.transition(ctx.tx.reservations, res, status)


def break_holds(ctx: Ctx, variant_id: str, reason: str) -> None:
    """Break every hold on a variant whose stock is gone (a hard stockout, the shelf set below what is held)."""
    for r in list(ctx.tx.reservations.all().values()):
        if r.status == "held" and any(v == variant_id for v, _ in r.items):
            release(ctx, r, "broken")
            ctx.emit("reservation.broken_by", sealed=True, reservation_id=r.id, variant_id=variant_id, reason=reason)


def expire(ctx: Ctx) -> None:
    for r in list(ctx.tx.reservations.all().values()):
        if r.status == "held" and r.expires_at <= ctx.now:
            release(ctx, r, "expired")


def reserve(ctx: Ctx, checkout_id: str, lines: Iterable[Line]) -> Reservation | None:
    """Hold stock for every line, or nothing at all. None when any line cannot be held."""
    need: dict[str, int] = {}
    for ln in lines:
        need[ln.variant_id] = need.get(ln.variant_id, 0) + ln.quantity
    for vid, q in need.items():
        v = ctx.tx.inventory.get(vid)
        if v is None or v.sellable < q:
            return None
    res = Reservation(
        ctx.new_id("res"), checkout_id, "held", tuple(sorted(need.items())), ctx.now + ctx.cfg.reservation_ttl_ms
    )
    for vid, q in need.items():
        v = ctx.variant(vid)
        ctx.tx.inventory.put(vid, replace(v, reserved=v.reserved + q))
    ctx.tx.reservations.put(res.id, res)
    ctx.emit(
        "reservation.held",
        id=res.id,
        checkout_id=checkout_id,
        items=[list(i) for i in res.items],
        expires_at=res.expires_at,
    )
    return res


def held_for(ctx: Ctx, ck: Checkout) -> dict[str, int]:
    """What a checkout currently holds, per variant: stock it may keep when its lines change."""
    res = ctx.tx.reservations.get(ck.reservation_id)
    return dict(res.items) if res is not None and res.status == "held" else {}


def rereserve(ctx: Ctx, ck: Checkout) -> Checkout:
    """Release a checkout's hold and hold its current lines again."""
    old = ctx.tx.reservations.get(ck.reservation_id)
    if old is not None:
        release(ctx, old, "released")
    res = reserve(ctx, ck.id, ck.lines) if ck.lines else None
    return replace(ck, reservation_id=res.id if res else None)

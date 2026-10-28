"""The environment's own commands (setup and live changes to stock, prices and sessions) and reads.

Why commands and not direct edits. A stock change mid-episode is part of what the agent experienced, so it goes
through `apply` and into the event log like any other change, and a replay makes it again at the same moment.
"""

from __future__ import annotations

from dataclasses import replace

from carttrace.sandbox.core.commands import OpenSession, Read, RevokeToken, SetPrice, SetStock
from carttrace.sandbox.core.context import Ctx
from carttrace.sandbox.core.state import Session
from carttrace.sandbox.core.stock import break_holds
from carttrace.sandbox.core.trust import revoke


def set_stock(ctx: Ctx, c: SetStock) -> None:
    v = ctx.variant(c.variant_id)
    if c.on_hand < v.reserved:
        break_holds(ctx, v.id, reason="stock_set")
        v = ctx.variant(v.id)
    ctx.tx.inventory.put(v.id, replace(v, on_hand=c.on_hand))
    ctx.emit("inventory.set", variant_id=v.id, on_hand=c.on_hand)


def set_price(ctx: Ctx, c: SetPrice) -> None:
    v = ctx.variant(c.variant_id)
    ctx.tx.inventory.put(v.id, replace(v, price=c.price))
    ctx.emit("inventory.repriced", variant_id=v.id, previous=v.price, price=c.price)


def open_session(ctx: Ctx, c: OpenSession) -> None:
    permissions = tuple(sorted(set(c.permissions)))
    profile_id = c.profile_id or ctx.cfg.profile_id
    ctx.tx.sessions.put(c.session_id, Session(c.session_id, c.tier, permissions, profile_id=profile_id))
    ctx.emit("session.opened", session=c.session_id, tier=c.tier, permissions=list(permissions), profile_id=profile_id)


def revoke_token(ctx: Ctx, c: RevokeToken) -> None:
    s = ctx.require(ctx.tx.sessions, c.session_id, "session_not_found")
    revoke(ctx, s.id, sealed=False)


def read(ctx: Ctx, c: Read) -> None:
    ctx.emit("query.served", target=c.target, id=c.id)

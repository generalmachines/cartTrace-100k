"""Who may send which command: Shopify's agent trust tiers as state-level policy, and the merchant's and the
environment's own commands kept out of the agent's reach.

Why in the core. A tier is a fact about the session, so it lives in the state and every command is checked against it
in the same place, before any handler runs. An adapter cannot forget to check it, and a revoked token is refused the
same way on every protocol surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from carttrace.sandbox.core.context import Ctx
from carttrace.sandbox.core.errors import PolicyDenied
from carttrace.sandbox.core.state import Session

TIERS = ("anonymous", "signed", "token")
"""Shopify's agent trust tiers (shopify.dev/docs/agents, Spring '26): an anonymous agent, a signed agent (a verified
agent profile, RFC 9421 signatures), and an agent holding a buyer token. Ordered: each tier has what the one before
has."""

ACTORS = ("agent", "merchant", "system")

POLICY: Mapping[str, Mapping[str, str]] = {
    "complete_checkout": {"tier": "token", "permission": "purchase"},
    "confirm_payment_action": {"tier": "token", "permission": "purchase"},
    "fulfill_order": {"actor": "merchant"},
    "cancel_order": {"actor": "merchant"},
    "refund_order": {"actor": "merchant"},
    "edit_order": {"actor": "merchant"},
    "request_return": {"actor": "merchant"},
    "receive_return": {"actor": "merchant"},
    "request_payment": {"actor": "merchant"},
    "record_mandate": {"actor": "merchant"},
    "set_stock": {"actor": "system"},
    "set_price": {"actor": "system"},
    "open_session": {"actor": "system"},
    "revoke_token": {"actor": "system"},
    "handoff_cart": {"actor": "system"},
}
"""What each command needs. Unlisted agent commands need nothing beyond a session (anonymous is enough to search, build
a cart and open a checkout, and to have the PSP vault a credential, which charges nothing by itself). Only a
token-tier session with the `purchase` permission may complete a checkout or confirm the buyer's payment step; a
revoked token is no token. Merchant commands (fulfil, refund, return, edit, and issuing a payment request or keeping a
mandate) are the store's side, never the agent's. System commands are the environment's own (setup, faults, the live
cart hand-off, the issuer's answer to a challenge page), and a merchant may not send them either."""


def tier_at_least(have: str, need: str) -> bool:
    return TIERS.index(have) >= TIERS.index(need)


def default_session(ctx: Ctx, session_id: str) -> Session:
    return Session(session_id, ctx.cfg.default_tier, tuple(ctx.cfg.default_permissions), profile_id=ctx.cfg.profile_id)


def session(ctx: Ctx) -> Session:
    """The command's session, opened implicitly with the pack's default tier and profile the first time it acts."""
    sid = ctx.cmd.session
    s = ctx.tx.sessions.get(sid)
    if s is None:
        s = default_session(ctx, sid)
        ctx.tx.sessions.put(s.id, s)
        ctx.emit("session.opened", session=s.id, tier=s.tier, profile_id=s.profile_id, implicit=True)
    return s


def revoke(ctx: Ctx, session_id: str, *, sealed: bool) -> None:
    s = ctx.tx.sessions.get(session_id) or default_session(ctx, session_id)
    ctx.tx.sessions.put(session_id, replace(s, revoked=True))
    ctx.emit("session.token_revoked", sealed=sealed, session=session_id)


def authorize(ctx: Ctx) -> None:
    """Raise `PolicyDenied` unless the command's actor, and an agent's session, may send this command."""
    rule = POLICY.get(ctx.cmd.type, {})
    actor = ctx.cmd.actor
    need_actor = rule.get("actor")
    if need_actor == "system" and actor != "system":
        raise PolicyDenied(code="system_only", message=f"{ctx.cmd.type} is the environment's own command")
    if need_actor == "merchant" and actor not in ("merchant", "system"):
        raise PolicyDenied(
            code="merchant_only", message=f"{ctx.cmd.type} is the merchant's side of the order, not the agent's"
        )
    if actor != "agent":
        return
    s = session(ctx)
    tier = rule.get("tier")
    if tier is None:
        return
    if s.revoked:
        raise PolicyDenied(
            code="token_revoked",
            message="the buyer's token was revoked",
            hint="ask the buyer to authorize again",
        )
    if not tier_at_least(s.tier, tier):
        raise PolicyDenied(
            code="tier_insufficient", message=f"{ctx.cmd.type} needs a {tier}-tier agent; this session is {s.tier}"
        )
    perm = rule.get("permission")
    if perm and perm not in s.permissions:
        raise PolicyDenied(code="permission_missing", message=f"{ctx.cmd.type} needs the {perm!r} permission")

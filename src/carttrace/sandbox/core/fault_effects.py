"""How a scheduled fault takes effect inside `apply`: whether it is due, how it is counted, and what it does.

Why here and not in the handlers. A fault is the environment's own doing, so it fires in one place, before the
command's handler runs, and its firing is a sealed event the agent never sees. Pre-effects change the world (stock,
prices, a token, a product's text); intercepts refuse the command outright (a 409, a 5xx, a 429). The other kinds
(declines, challenges, duplicate webhooks, refused codes, unshippable countries) are consulted by the handler that
reaches that point, through `consume` (counted) and `peek` (standing, not counted).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from carttrace.sandbox.core.config import Config, FaultRule
from carttrace.sandbox.core.context import Ctx
from carttrace.sandbox.core.errors import Conflict, Unavailable
from carttrace.sandbox.core.event import Event
from carttrace.sandbox.core.state import State
from carttrace.sandbox.core.stock import break_holds
from carttrace.sandbox.core.trust import revoke
from carttrace.sandbox.core.tx import Tx

PRE_EFFECTS = ("stockout", "reprice", "revoke_token", "injected_text")
INTERCEPTS = ("conflict", "unavailable", "rate_limited")


def touched_variants(ctx: Ctx) -> set[str]:
    """Variants the command names, or that sit in the cart or checkout it names."""
    cmd = ctx.cmd
    lines: Sequence[Any] = getattr(cmd, "lines", None) or ()
    out = {ln.variant_id for ln in lines if getattr(ln, "variant_id", None)}
    for ent in (
        ctx.tx.carts.get(getattr(cmd, "cart_id", None)),
        ctx.tx.checkouts.get(getattr(cmd, "checkout_id", None)),
    ):
        if ent is not None:
            out |= {ln.variant_id for ln in ent.lines}
    return out


def due(ctx: Ctx, rule: FaultRule, *, ignore_on: bool = False) -> bool:
    if rule.at_start:
        return False  # it fired as the episode opened (`start_effects`), never on a command
    fired = ctx.tx.faults_fired.get(rule.id) or 0
    if rule.times is not None and fired >= rule.times:
        return False
    if rule.at_seq is not None and ctx.tx.seq < rule.at_seq:
        return False
    if rule.after is not None and ctx.tx.seq <= rule.after:
        return False
    if rule.on and not ignore_on and ctx.cmd.type not in rule.on:
        return False
    return not (rule.variant is not None and rule.variant not in touched_variants(ctx))


def fire(ctx: Ctx, rule: FaultRule) -> None:
    ctx.tx.faults_fired.put(rule.id, (ctx.tx.faults_fired.get(rule.id) or 0) + 1)
    ctx.emit("fault.fired", sealed=True, fault_id=rule.id, kind=rule.kind, command=ctx.cmd.type)


def consume(ctx: Ctx, kind: str, *, ignore_on: bool = False) -> FaultRule | None:
    """The first due fault of this kind, counted as fired; None when none is due."""
    for rule in ctx.cfg.faults:
        if rule.kind == kind and due(ctx, rule, ignore_on=ignore_on):
            fire(ctx, rule)
            return rule
    return None


def peek(ctx: Ctx, kind: str) -> list[FaultRule]:
    """Standing faults (refused codes, unshippable countries) that apply now, without counting a firing."""
    return [r for r in ctx.cfg.faults if r.kind == kind and due(ctx, r, ignore_on=not r.on)]


def run_scheduled(ctx: Ctx) -> None:
    """Fire every due pre-effect, then the first due intercept, which refuses the command by raising."""
    for rule in ctx.cfg.faults:
        if rule.kind in PRE_EFFECTS and due(ctx, rule):
            fire(ctx, rule)
            _pre_effect(ctx, rule)
    for rule in ctx.cfg.faults:
        if rule.kind in INTERCEPTS and due(ctx, rule):
            fire(ctx, rule)
            _intercept(rule)


def start_effects(state: State, config: Config) -> tuple[State, list[Event]]:
    """The state an episode opens on: every `at_start` rule fired once, in schedule order, each as a sealed
    `fault.fired` (with `at_start`) followed by what it changed. The same pack gives the same state and events, so a
    replay that opens the same way reaches the same digest."""
    rules = [r for r in config.faults if r.at_start]
    if not rules:
        return state, []
    tx = Tx(state)
    events: list[Event] = []
    for rule in rules:
        tx.faults_fired.put(rule.id, (tx.faults_fired.get(rule.id) or 0) + 1)
        data = {"fault_id": rule.id, "kind": rule.kind, "command": None, "at_start": True}
        events.append(Event("fault.fired", data, sealed=True))
        _change_world(tx, rule, events)
    return tx.freeze(), events


def _pre_effect(ctx: Ctx, rule: FaultRule) -> None:
    p = rule.params
    if rule.kind == "revoke_token":
        revoke(ctx, str(p.get("session") or ctx.cmd.session), sealed=True)
        return
    if rule.kind == "stockout" and p.get("hard"):
        vid = str(p.get("variant") or rule.variant)
        if ctx.tx.inventory.get(vid) is not None:
            break_holds(ctx, vid, reason="stockout")
    _change_world(ctx.tx, rule, ctx.events)


def _change_world(tx: Tx, rule: FaultRule, events: list[Event]) -> None:
    """What a stockout, a reprice or injected text does to the world, with its sealed event. No command needed, so an
    episode can open on it."""
    p = rule.params
    if rule.kind in ("stockout", "reprice"):
        vid = str(p.get("variant") or rule.variant)
        v = tx.inventory.get(vid)
        if v is None:
            return
        if rule.kind == "stockout":
            tx.inventory.put(vid, replace(v, on_hand=v.reserved))
            events.append(Event("inventory.stockout", {"variant_id": vid, "on_hand": v.reserved}, sealed=True))
        else:
            price = int(p["price"])
            tx.inventory.put(vid, replace(v, price=price))
            data = {"variant_id": vid, "previous": v.price, "price": price}
            events.append(Event("inventory.repriced", data, sealed=True))
    elif rule.kind == "injected_text":
        pid = str(p["product_id"])
        tx.overlays.put(pid, str(p["text"]))
        events.append(Event("content.injected", {"product_id": pid}, sealed=True))


def _intercept(rule: FaultRule) -> None:
    p = rule.params
    if rule.kind == "conflict":
        raise Conflict(code="conflict", message=str(p.get("message") or "the resource changed; retry"), retryable=True)
    if rule.kind == "unavailable":
        raise Unavailable(
            code="server_error",
            message=str(p.get("message") or "the store is temporarily unavailable"),
            retryable=True,
            details=({"status": int(p.get("status", 503))},),
        )
    retry_after = int(p.get("retry_after", 182))
    raise Unavailable(
        code="rate_limited",
        message=f"Rate limit exceeded, retry after {retry_after} seconds",
        retryable=True,
        details=({"retry_after": retry_after},),
    )

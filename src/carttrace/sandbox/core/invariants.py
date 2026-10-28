"""The invariants every commit must keep. `apply` runs `check(before, after)` and refuses to commit a state that breaks
one, so a bug in the core shows up as a refused command with an `invariant` error, never as a corrupt store a grader
then trusts.

What is held:
- statuses move only along the transition tables, and nothing is ever deleted;
- totals add up; stock is never negative and reservations never exceed it; the sum of held reservations is each
  variant's reserved count;
- one order per idempotency key and per checkout; a completed checkout has lines, an order and a captured payment;
- payment amounts match the payment's status: nothing is authorized or captured before authorization, capture never
  exceeds authorization, refunds never exceed capture, a challenge has an id, a decline has a reason, and a checkout
  holds at most one payment that has or may still take money;
- a vaulted credential is consumed only by a captured payment of its own checkout within its allowance, and a
  payment charged to one stays within it;
- a payment request is paid only by a captured payment of exactly its amount, pending only while that payment waits on
  the buyer, and a payment that answers a request is of its checkout and its amount;
- a session's tier is a known tier and its profile id is a profile id;
- the webhook outbox, the idempotency store and the recorded mandates are append-only.

Checks walk only what changed (by object identity, since state is copied on write), so they stay cheap on a catalog
of thousands of variants.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

from carttrace.sandbox.core.machines import MACHINES, reachable
from carttrace.sandbox.core.pricing import adds_up
from carttrace.sandbox.core.state import Payment, State
from carttrace.sandbox.core.trust import TIERS

_PROFILE = re.compile(r"^pp-[0-9a-f]{12}$")
BEFORE_AUTHORIZATION = frozenset({"pending", "requires_action", "authenticated", "declined", "authentication_failed"})
CAPTURED = frozenset({"captured", "partially_refunded", "refunded"})
LIVE_PAYMENT = frozenset({"requires_action", "authenticated", "authorized"} | CAPTURED)
"""Statuses of a payment that has taken, or may still take, the buyer's money."""
CHALLENGED = frozenset({"requires_action", "authenticated", "authentication_failed"})
DECIDED = frozenset({"frictionless", "frictionless_declined"})


def _changed(before: Mapping[str, Any], after: Mapping[str, Any]) -> Iterator[str]:
    if before is after:
        return
    for k, v in after.items():
        if before.get(k) is not v:
            yield k


def check(before: State, after: State) -> list[str]:
    """Every invariant `after` breaks, given that it was committed on top of `before`. Empty when the commit is
    sound."""
    return [
        *_statuses(before, after),
        *_stock(before, after),
        *_totals(before, after),
        *_orders(before, after),
        *_payments(before, after),
        *_credentials(before, after),
        *_requests(before, after),
        *_checkouts(before, after),
        *_sessions(before, after),
        *_append_only(before, after),
    ]


def _statuses(before: State, after: State) -> Iterator[str]:
    for name, table in MACHINES.items():
        b, a = getattr(before, name), getattr(after, name)
        for k in b:
            if k not in a:
                yield f"{name}/{k} was deleted"
        for k in _changed(b, a):
            ent = a[k]
            if ent.status not in table:
                yield f"{name}/{k} has unknown status {ent.status}"
            elif k in b and b[k].status != ent.status and ent.status not in reachable(name, b[k].status):
                yield f"{name}/{k} moved {b[k].status} -> {ent.status}, not a legal transition"


def _stock(before: State, after: State) -> Iterator[str]:
    held: dict[str, int] = {}
    for r in after.reservations.values():
        if r.status == "held":
            for vid, q in r.items:
                held[vid] = held.get(vid, 0) + q
    touched = set(_changed(before.inventory, after.inventory)) | set(held)
    if before.reservations is not after.reservations:
        for r in before.reservations.values():
            touched.update(v for v, _ in r.items)
    for vid in sorted(touched):
        v = after.inventory.get(vid)
        if v is None:
            yield f"inventory/{vid} is reserved but does not exist"
            continue
        if v.on_hand < 0:
            yield f"inventory/{vid} stock {v.on_hand} < 0"
        if v.reserved < 0 or v.reserved > v.on_hand:
            yield f"inventory/{vid} reserved {v.reserved} outside 0..{v.on_hand}"
        if v.reserved != held.get(vid, 0):
            yield f"inventory/{vid} reserved {v.reserved} != held reservations {held.get(vid, 0)}"
        if v.price < 0:
            yield f"inventory/{vid} negative price"


def _totals(before: State, after: State) -> Iterator[str]:
    for name in ("carts", "checkouts", "orders"):
        b, a = getattr(before, name), getattr(after, name)
        for k in _changed(b, a):
            ent = a[k]
            for p in adds_up(ent.totals, ent.lines):
                yield f"{name}/{k}: {p}"


def _orders(before: State, after: State) -> Iterator[str]:
    if before.orders is after.orders:
        return
    keys: dict[tuple[str, str], str] = {}
    by_checkout: dict[str, str] = {}
    for o in after.orders.values():
        if o.idempotency_key:
            k2 = (o.session, o.idempotency_key)
            if k2 in keys:
                yield f"orders {keys[k2]} and {o.id} share idempotency key {o.idempotency_key}"
            keys[k2] = o.id
        if o.checkout_id in by_checkout:
            yield f"orders {by_checkout[o.checkout_id]} and {o.id} come from one checkout {o.checkout_id}"
        by_checkout[o.checkout_id] = o.id
        if not o.lines:
            yield f"orders/{o.id} has no lines"
    for k in _changed(before.orders, after.orders):
        o = after.orders[k]
        pay = after.payments.get(o.payment_id)
        if pay is None or pay.checkout_id != o.checkout_id:
            yield f"orders/{k} is not paid by a payment of its own checkout"
        elif pay.currency != o.currency:
            yield f"orders/{k} is in {o.currency} but its payment is in {pay.currency}"


def _payment_amounts(k: str, p: Payment) -> Iterator[str]:
    if not 0 <= p.captured <= p.authorized:
        yield f"payments/{k} captured {p.captured} outside 0..authorized {p.authorized}"
    if not 0 <= p.refunded <= p.captured:
        yield f"payments/{k} refunded {p.refunded} outside 0..captured {p.captured}"
    if p.authorized > p.amount:
        yield f"payments/{k} authorized {p.authorized} > amount {p.amount}"
    if p.status in BEFORE_AUTHORIZATION and (p.authorized or p.captured):
        yield f"payments/{k} is {p.status} but holds money"
    if p.status == "authorized" and (p.authorized != p.amount or p.captured):
        yield f"payments/{k} is authorized but authorized {p.authorized} of {p.amount}, captured {p.captured}"
    if p.status in CAPTURED and p.captured != p.authorized:
        yield f"payments/{k} is {p.status} but captured {p.captured} of {p.authorized} authorized"
    if p.status == "captured" and p.refunded:
        yield f"payments/{k} is captured but refunded {p.refunded}"
    if p.status == "partially_refunded" and not 0 < p.refunded < p.captured:
        yield f"payments/{k} is partially_refunded but refunded {p.refunded} of {p.captured}"
    if p.status == "refunded" and p.refunded != p.captured:
        yield f"payments/{k} is refunded but refunded {p.refunded} of {p.captured}"
    if p.status == "voided" and p.captured:
        yield f"payments/{k} is voided but captured {p.captured}"


def _payments(before: State, after: State) -> Iterator[str]:
    changed = list(_changed(before.payments, after.payments))
    for k in changed:
        p = after.payments[k]
        yield from _payment_amounts(k, p)
        # a 3-D Secure decision with no challenge (frictionless, D71; accepted or declined at once, D111) has none
        if p.status in CHALLENGED and not p.challenge_id and p.authentication not in DECIDED:
            yield f"payments/{k} is {p.status} without a challenge"
        if p.status in ("declined", "authentication_failed") and not p.decline_code:
            yield f"payments/{k} is {p.status} without a reason"
        if p.checkout_id not in after.checkouts:
            yield f"payments/{k} belongs to no checkout"
        yield from _payment_links(k, p, after)
    if changed:
        live: dict[str, list[str]] = {}
        for p in after.payments.values():
            if p.status in LIVE_PAYMENT:
                live.setdefault(p.checkout_id, []).append(p.id)
        for kid in sorted({after.payments[k].checkout_id for k in changed}):
            if len(live.get(kid, ())) > 1:
                yield f"checkouts/{kid} has {len(live[kid])} payments that hold or may take money"


def _payment_links(k: str, p: Payment, after: State) -> Iterator[str]:
    """A payment that answers a request, or charges a vaulted credential, stays inside what it was allowed. A
    declined payment charged nothing, so it is not held to the allowance: a Shop Token presented for another
    checkout declines under the spec's checkout scope, and that decline must exist to be read back (D73)."""
    if p.status == "declined":
        return
    if p.request_id is not None:
        req = after.payment_requests.get(p.request_id)
        if req is None:
            yield f"payments/{k} answers an unknown payment request"
        elif req.checkout_id != p.checkout_id or req.amount != p.amount or req.currency != p.currency:
            yield f"payments/{k} does not match payment request {req.id} (checkout, amount or currency)"
    if p.credential_id is not None:
        cred = after.credentials.get(p.credential_id)
        if cred is None:
            yield f"payments/{k} charges an unknown vaulted credential"
        elif cred.checkout_id != p.checkout_id or p.amount > cred.max_amount or cred.currency != p.currency:
            yield f"payments/{k} is outside the allowance of credential {cred.id}"


def _credentials(before: State, after: State) -> Iterator[str]:
    for k in _changed(before.credentials, after.credentials):
        c = after.credentials[k]
        if c.max_amount <= 0:
            yield f"credentials/{k} has no allowance"
        pay = after.payments.get(c.payment_id) if c.payment_id else None
        if c.status == "consumed" and (pay is None or pay.status not in CAPTURED or pay.credential_id != k):
            yield f"credentials/{k} is consumed without a captured payment charged to it"
        if c.status == "active" and c.payment_id is not None:
            yield f"credentials/{k} is active but names a payment"


def _requests(before: State, after: State) -> Iterator[str]:
    for k in _changed(before.payment_requests, after.payment_requests):
        r = after.payment_requests[k]
        if r.checkout_id not in after.checkouts:
            yield f"payment_requests/{k} belongs to no checkout"
        pay = after.payments.get(r.payment_id) if r.payment_id else None
        if pay is not None and pay.request_id != k:
            yield f"payment_requests/{k} names payment {pay.id}, which answers another request"
        if r.status == "paid" and (pay is None or pay.status not in CAPTURED or pay.amount != r.amount):
            yield f"payment_requests/{k} is paid without a captured payment of its amount"
        if r.status == "pending" and (pay is None or pay.status not in ("pending", "requires_action")):
            yield f"payment_requests/{k} is pending but no payment for it waits"
        if r.status in ("open", "void") and r.payment_id is not None:
            yield f"payment_requests/{k} is {r.status} but names a payment"


def _checkouts(before: State, after: State) -> Iterator[str]:
    for k in _changed(before.checkouts, after.checkouts):
        c = after.checkouts[k]
        pay = after.payments.get(c.payment_id) if c.payment_id else None
        if c.status == "completed":
            if not c.lines:
                yield f"checkouts/{k} completed with no lines"
            if not c.order_id or c.order_id not in after.orders:
                yield f"checkouts/{k} completed without an order"
            if pay is None or pay.status not in CAPTURED:
                yield f"checkouts/{k} completed without a captured payment"
        if c.status in ("ready_for_complete", "complete_in_progress") and not c.lines:
            yield f"checkouts/{k} is {c.status} with no lines"
        waiting = c.status == "requires_escalation" and c.escalation == "payment_action_required"
        if waiting and (pay is None or pay.status != "requires_action"):
            yield f"checkouts/{k} waits on the buyer's payment step but no payment is waiting"
        if c.escalation is not None and c.status != "requires_escalation":
            yield f"checkouts/{k} is {c.status} but still names an escalation"


def _sessions(before: State, after: State) -> Iterator[str]:
    for k in _changed(before.sessions, after.sessions):
        s = after.sessions[k]
        if s.tier not in TIERS:
            yield f"sessions/{k} has unknown tier {s.tier}"
        if s.profile_id is not None and not _PROFILE.match(s.profile_id):
            yield f"sessions/{k} has a malformed profile id"


def _append_only(before: State, after: State) -> Iterator[str]:
    if after.webhooks[: len(before.webhooks)] != before.webhooks:
        yield "the webhook outbox was rewritten"
    for k, rec in before.idempotency.items():
        if after.idempotency.get(k) is not rec:
            yield f"idempotency record {k} was changed"
    if before.mandates is not after.mandates:
        for k, m in before.mandates.items():
            if after.mandates.get(k) is not m:
                yield f"mandate {k} was changed"

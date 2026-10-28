"""Every entity's legal transitions, declared as data.

Why data. The lab's old store moved statuses by assignment wherever it liked, which is how `requires_escalation`
became a state with no way out and how a canceled cart could still complete a checkout. Here a status only changes
through `Ctx.transition`, which checks this table, and the commit checks it again by comparing every entity's status
before and after. Anything not listed is illegal. ACP publishes no transition table and UCP's is prose, so these
tables are our reading of both (UCP 2026-08-25 checkout statuses; Shopify's Cart, Checkout and Order surfaces).

A self-loop (a -> a) is listed where an update without a status change is legal.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache

from carttrace.sandbox.core.errors import IllegalTransition

CART: Mapping[str, frozenset[str]] = {
    "active": frozenset({"active", "converted", "canceled"}),
    "converted": frozenset({"converted", "canceled"}),
    "canceled": frozenset(),
}
"""A cart becomes `converted` when a checkout is created from it. Canceling a converted cart cancels its open
checkout."""

CHECKOUT: Mapping[str, frozenset[str]] = {
    "incomplete": frozenset({"incomplete", "ready_for_complete", "requires_escalation", "canceled"}),
    "ready_for_complete": frozenset(
        {"ready_for_complete", "incomplete", "requires_escalation", "complete_in_progress", "canceled"}
    ),
    "requires_escalation": frozenset({"requires_escalation", "incomplete", "ready_for_complete", "canceled"}),
    "complete_in_progress": frozenset({"completed", "ready_for_complete", "requires_escalation"}),
    "completed": frozenset(),
    "canceled": frozenset(),
}
"""`requires_escalation` has three exits: the buyer's step is done (back to ready), the agent fixes what was missing
(back to incomplete or ready), or the checkout is canceled. It used to have none."""

PAYMENT: Mapping[str, frozenset[str]] = {
    "pending": frozenset({"requires_action", "authenticated", "authentication_failed", "authorized", "declined"}),
    "requires_action": frozenset({"authenticated", "authentication_failed", "voided"}),
    "authenticated": frozenset({"authorized", "declined", "voided"}),
    "authorized": frozenset({"captured", "voided"}),
    "captured": frozenset({"partially_refunded", "refunded"}),
    "partially_refunded": frozenset({"partially_refunded", "refunded"}),
    "declined": frozenset(),
    "authentication_failed": frozenset(),
    "voided": frozenset(),
    "refunded": frozenset(),
}
"""The PSP's side of a payment. A 3-D Secure challenge parks a payment in `requires_action` until the buyer's bank
answers (or, frictionless, D71, authenticates it at once from `pending`): `authenticated` goes on to authorization
(which can still decline, for insufficient funds), while `authentication_failed` ends it; a 3-D Secure decision
with no challenge (D111) declines from `pending` straight to `authentication_failed`. A challenge abandoned
because the checkout was canceled or retried is `voided`, as is an authenticated payment whose stock was lost before
it could be authorized."""

ORDER: Mapping[str, frozenset[str]] = {
    "on_hold": frozenset({"on_hold", "confirmed", "partially_fulfilled", "fulfilled", "canceled"}),
    "confirmed": frozenset({"confirmed", "partially_fulfilled", "fulfilled", "canceled"}),
    "partially_fulfilled": frozenset({"partially_fulfilled", "fulfilled"}),
    "fulfilled": frozenset({"return_requested"}),
    "return_requested": frozenset({"returned"}),
    "returned": frozenset(),
    "canceled": frozenset(),
}
"""Shopify's fulfillment side of an order. Money (refunds) lives on the payment's own machine. `on_hold` is an order
placed with an offline method the merchant must see paid first (WooCommerce bank transfer and cheque, D42); it is
placed in that state, never moved into it, and leaves it as a confirmed order would move on."""

RESERVATION: Mapping[str, frozenset[str]] = {
    "held": frozenset({"committed", "released", "expired", "broken"}),
    "committed": frozenset(),
    "released": frozenset(),
    "expired": frozenset(),
    "broken": frozenset(),
}

CREDENTIAL: Mapping[str, frozenset[str]] = {
    "active": frozenset({"active", "consumed"}),
    "consumed": frozenset(),
}
"""A vaulted credential is one-time: the capture of a payment charged to it consumes it. A declined attempt leaves it
active, as a PSP leaves a token usable after an issuer's decline; its expiry is a time, checked at completion."""

PAYMENT_REQUEST: Mapping[str, frozenset[str]] = {
    "open": frozenset({"open", "pending", "paid", "failed", "void"}),
    "pending": frozenset({"pending", "paid", "failed", "void"}),
    "paid": frozenset(),
    "failed": frozenset(),
    "void": frozenset(),
}
"""A payment request is answered once. A payment that waits on the buyer's 3-D Secure step leaves it pending; a
decline, a failed authentication or a payment voided on the way fails it (the protocol issues a fresh request for the
next attempt); a request still open when its checkout completes some other way, or is canceled, is void."""

MACHINES: Mapping[str, Mapping[str, frozenset[str]]] = {
    "carts": CART,
    "checkouts": CHECKOUT,
    "payments": PAYMENT,
    "orders": ORDER,
    "reservations": RESERVATION,
    "credentials": CREDENTIAL,
    "payment_requests": PAYMENT_REQUEST,
}
INITIAL: Mapping[str, str] = {
    "carts": "active",
    "checkouts": "incomplete",
    "payments": "pending",
    "orders": "confirmed",
    "reservations": "held",
    "credentials": "active",
    "payment_requests": "open",
}
OPEN_CHECKOUT = ("incomplete", "ready_for_complete", "requires_escalation")
"""Checkout statuses an agent may still change."""


def legal(machine: str, before: str, after: str) -> bool:
    return after in MACHINES[machine].get(before, frozenset())


def check(machine: str, entity_id: str, before: str, after: str) -> None:
    """Raise `IllegalTransition` unless `before -> after` is in the machine's table."""
    if not legal(machine, before, after):
        noun = machine[:-1]
        raise IllegalTransition(
            code=f"{noun}_{before}",
            message=f"{noun} {entity_id} is {before}; it cannot become {after}",
            field="status",
        )


@cache
def reachable(machine: str, before: str) -> frozenset[str]:
    """Every status reachable from `before` in one or more legal moves. A commit may chain several: a completion goes
    ready -> complete_in_progress -> completed, and a payment pending -> authorized -> captured."""
    table = MACHINES[machine]
    seen: set[str] = set()
    todo = [before]
    while todo:
        s = todo.pop()
        for nxt in table.get(s, ()):
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return frozenset(seen)

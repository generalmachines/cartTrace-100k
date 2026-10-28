"""Completion and the buyer's payment step: stock re-checked, the payment authenticated, authorized and captured, the
order placed, or the payment declined or parked on a 3-D Secure challenge.

How the PSP decides, in order. A due `payment_decline` fault declines; a due `three_ds` fault asks for a challenge
(or, where the store decides 3-D Secure at once, D111, authenticates and approves or declines with no challenge);
otherwise the token decides (`carttrace.sandbox.core.psp.TOKENS`): a challenge token parks the payment in `requires_action`,
a decline token declines at authorization, any other credential is approved. After a challenge the buyer's answer
comes back as `confirm_payment_action` (from the agent's token session, or from the environment when a challenge page
answers): the issuer passes or fails the authentication as the token or fault scripted it, and a passed payment goes
on to authorization, where a `payment_decline` fault scheduled on `confirm_payment_action` or a decline token still
declines (3-D Secure proves who the buyer is, not that the card has funds).

A decline or a failed authentication returns its error instead of raising it, so its side effects stay: a declined
payment really was declined, and the checkout is ready for another attempt.

Before the PSP decides, a completion's limits are checked, and refused before anything changes: a vaulted credential
(`vt-...`) is charged only by its own session, for its checkout, unexpired, unused, and for no more than its allowance;
a payment that names a payment request must find it open, unexpired, for this checkout and for exactly the checkout's
total. The amount checks run after the stock re-check, on the total the buyer will actually pay. The request then
follows the payment (pending on a challenge, paid on capture, failed on a decline), and a capture consumes the
credential and voids the checkout's other open requests.

Shop Pay (D73). The dev.shopify.shop_pay handler charges only a Shop Token minted for this checkout
(`mint_shop_token`): anything else is refused before anything changes. A minted token is a payment like any other, so
the spec's three rules are the PSP's own declines, in the event log and the PSP's record: single-use
(`shop_token_used`, a token whose payment captured), checkout-scoped (`shop_token_checkout_mismatch`, minted for
another checkout) and time-limited (`shop_token_expired`, past its validity on the logical clock). The wallet
credential the token was minted from decides the issuer's answer, as a card token does.
"""

from __future__ import annotations

from dataclasses import replace

from carttrace.sandbox.core import brokers, psp
from carttrace.sandbox.core.commands import CompleteCheckout, ConfirmPaymentAction
from carttrace.sandbox.core.context import Ctx
from carttrace.sandbox.core.errors import BusinessRule, CoreError, IllegalTransition
from carttrace.sandbox.core.fault_effects import consume
from carttrace.sandbox.core.handlers.checkouts import checkout_totals, resolve_groups, settle
from carttrace.sandbox.core.handlers.rails import answer_request, consume_credential, void_open_requests
from carttrace.sandbox.core.outbox import webhook
from carttrace.sandbox.core.state import Checkout, Order, OrderLine, Payment, PaymentRequest, VaultedCredential
from carttrace.sandbox.core.stock import reserve

RETRY_HINT = "try another payment credential, or ask the buyer"


def ensure_stock(ctx: Ctx, ck: Checkout) -> tuple[Checkout, CoreError | None]:
    """The completion re-check: the reservation must still be held (a lapsed one is renewed if the shelf still has the
    stock), every variant must still be for sale, and every price must still be the one the buyer was quoted."""
    res = ctx.tx.reservations.get(ck.reservation_id)
    if res is None or res.status != "held":
        renewed = reserve(ctx, ck.id, ck.lines)
        if renewed is None:
            ck = replace(ck, reservation_id=None)
            ctx.tx.checkouts.put(ck.id, ck)
            ck = ctx.transition(ctx.tx.checkouts, ck, "incomplete", escalation=None)
            ctx.emit("checkout.stock_lost", id=ck.id)
            return ck, BusinessRule(
                code="merchandise_out_of_stock", message="the reservation lapsed and the stock is gone", field="lines"
            )
        ck = replace(ck, reservation_id=renewed.id)
        ctx.tx.checkouts.put(ck.id, ck)
    for ln in ck.lines:
        if not ctx.variant(ln.variant_id).available:
            return ck, BusinessRule(
                code="merchandise_out_of_stock", message=f"{ln.variant_id} is no longer for sale", field="lines"
            )
    repriced = tuple(replace(ln, unit_price=ctx.variant(ln.variant_id).price) for ln in ck.lines)
    if repriced != ck.lines:
        ck = replace(ck, lines=repriced)
        ck = replace(ck, groups=resolve_groups(ctx, ck, None))
        ck = replace(ck, totals=checkout_totals(ctx, ck))
        ctx.tx.checkouts.put(ck.id, ck)
        ctx.emit("checkout.repriced", id=ck.id, total=ck.totals.total)
        return ck, BusinessRule(
            code="price_changed",
            message="a price changed since the checkout was quoted",
            hint="review the new total and complete again",
        )
    return ck, None


def finalize(ctx: Ctx, ck: Checkout, pay: Payment) -> None:
    """Authorize and capture, commit the held stock, place the order and notify. An order paid by a method the
    platform holds (`CoreRules.on_hold_methods`: WooCommerce bank transfer and cheque, captures/diff-woo 112 and 117)
    is placed `on_hold`; the ledger books it as it books cash on delivery, so a cancel or refund works as on any
    order."""
    pay = ctx.transition(ctx.tx.payments, pay, "authorized", authorized=pay.amount)
    pay = ctx.transition(ctx.tx.payments, pay, "captured", captured=pay.amount)
    res = ctx.require(ctx.tx.reservations, ck.reservation_id, "reservation_not_found")
    for vid, q in res.items:
        v = ctx.variant(vid)
        ctx.tx.inventory.put(vid, replace(v, on_hand=v.on_hand - q, reserved=v.reserved - q))
    ctx.transition(ctx.tx.reservations, res, "committed")
    oid = ctx.new_id("order")
    key = ctx.cmd.idempotency_key if ctx.cmd.type == "complete_checkout" else None
    held = pay.instrument in ctx.cfg.rules.on_hold_methods
    order = Order(
        oid,
        1000 + len(ctx.tx.orders.all()) + 1,
        ck.id,
        ck.session,
        key,
        "on_hold" if held else "confirmed",
        tuple(OrderLine(ln.id, ln.variant_id, ln.quantity, ln.unit_price) for ln in ck.lines),
        ck.totals,
        ck.codes,
        ck.groups,
        dict(ck.buyer),
        pay.id,
        pay.currency,
        ctx.now,
    )
    ctx.tx.orders.put(oid, order)
    ctx.emit("order.created", id=oid, checkout_id=ck.id, total=ck.totals.total, currency=pay.currency)
    if held:
        ctx.emit("order.on_hold", id=oid, method=pay.instrument, reason="awaiting_payment")
    ctx.transition(ctx.tx.checkouts, ck, "completed", order_id=oid, escalation=None)
    answer_request(ctx, pay, "paid")
    consume_credential(ctx, pay)
    void_open_requests(ctx, ck.id)
    webhook(ctx, "order.created", oid)
    ctx.refs["order_id"] = oid


def _accepted(ctx: Ctx, method: str | None) -> None:
    if method is not None and method not in ctx.cfg.market.accepts:
        raise BusinessRule(
            code="payment_instrument_unsupported",
            message=f"instrument type {method!r} is not accepted here; accepted: {', '.join(ctx.cfg.market.accepts)}",
            field="payment.instrument",
        )


def _method(
    ctx: Ctx, instrument: str | None, token: str | None, namespace: str
) -> tuple[str | None, psp.TokenSpec | None]:
    """The payment method type and the checked credential. Raises before anything changes."""
    method = instrument.lower() if instrument else None
    _accepted(ctx, method)
    spec = psp.credential_for(namespace, token, method)
    if spec is not None and method is None:
        method = spec.method
        _accepted(ctx, method)
    return method, spec


def _vaulted(ctx: Ctx, c: CompleteCheckout, ck: Checkout, rule: psp.HandlerRule) -> VaultedCredential | None:
    """The vaulted credential a completion presents, checked against its allowance's checkout, expiry and use; None
    for a raw test token. Raises before anything changes."""
    token = c.payment.token
    if token is None or not token.startswith(psp.VAULT_PREFIX):
        if rule.vaulted:
            raise BusinessRule(
                code="delegated_credential_required",
                message="this handler charges only credentials the payment processor vaulted with an allowance",
                field="payment.token",
                hint="vault the credential first (ACP delegate_payment) and pay with the vt- token it returns",
            )
        return None
    cred = ctx.tx.credentials.get(token)
    if cred is None or cred.session != c.session:
        raise BusinessRule(
            code="payment_credential_invalid",
            message="the payment credential is not one this store's payment processor issued",
            field="payment.token",
            hint="use the token the payment processor returned when it vaulted your credential",
        )
    if cred.status != "active":
        raise BusinessRule(
            code="credential_used",
            message="this vaulted credential was already charged; it is good for one payment",
            field="payment.token",
            hint="vault the credential again for a new allowance",
        )
    if cred.checkout_id != ck.id:
        raise BusinessRule(
            code="allowance_checkout_mismatch",
            message="the vaulted credential's allowance is for another checkout",
            field="payment.token",
            hint="vault the credential for this checkout",
        )
    if cred.expires_at < ctx.now:
        raise BusinessRule(
            code="allowance_expired",
            message="the vaulted credential's allowance has expired",
            field="payment.token",
            hint="vault the credential again with a later expiry",
        )
    return cred


def _shop_token(ctx: Ctx, c: CompleteCheckout, rule: psp.HandlerRule) -> VaultedCredential | None:
    """The minted Shop Token a completion through dev.shopify.shop_pay presents; None on every other handler. Raises
    before anything changes: a handler that takes only minted tokens refuses a raw credential, and a token this store
    never minted (or another session's) is not a credential it knows."""
    if not rule.shop:
        return None
    token = c.payment.token
    if token is None or not token.startswith(psp.SHOP_TOKEN_PREFIX):
        raise BusinessRule(
            code="shop_token_required",
            message="this handler charges only single-use Shop Tokens minted for one checkout",
            field="payment.token",
            hint="mint a Shop Token from the wallet's Shop Pay credential (the Shop Pay interface), then pay with it",
        )
    cred = ctx.tx.credentials.get(token)
    if cred is None or cred.session != c.session:
        raise BusinessRule(
            code="payment_credential_invalid",
            message="the payment credential is not one this store's payment processor issued",
            field="payment.token",
            hint="pay with the Shop Token minted for this checkout",
        )
    return cred


def _shop_decline(ctx: Ctx, cred: VaultedCredential, ck: Checkout) -> str | None:
    """The decline code a minted Shop Token earns under the spec's three rules (D73), or None when it may be
    processed: already captured once (single-use), minted for another checkout (checkout-scoped), or past its
    validity (time-limited; at the expiry instant the token is expired, as JWT's exp reads)."""
    if cred.status != "active":
        return psp.SHOP_DECLINE_USED
    if cred.checkout_id != ck.id:
        return psp.SHOP_DECLINE_CHECKOUT
    if cred.expires_at <= ctx.now:
        return psp.SHOP_DECLINE_EXPIRED
    return None


def _issuer_spec(ctx: Ctx, pay: Payment) -> psp.TokenSpec | None:
    """The credential spec that decides the issuer's answer for a payment: the presented token's own, or for a Shop
    Token the wallet credential it was minted from (the minted id is no test token)."""
    if pay.token is None:
        return None
    if pay.token.startswith(psp.SHOP_TOKEN_PREFIX) and pay.credential_id is not None:
        cred = ctx.tx.credentials.get(pay.credential_id)
        return psp.TOKENS.get(cred.token) if cred is not None else None
    return psp.TOKENS.get(pay.token)


def _requested(ctx: Ctx, c: CompleteCheckout, ck: Checkout, rule: psp.HandlerRule) -> PaymentRequest | None:
    """The payment request a completion answers, checked for its checkout, its use and its expiry; None when the
    completion names none. Raises before anything changes."""
    rid = c.payment.request_id
    if rid is None:
        if rule.requested:
            raise BusinessRule(
                code="payment_request_required",
                message="this handler takes only payments that answer the store's payment request",
                field="payment.request_id",
                hint="ask the store for a payment request (an MPP challenge, an AP2 cart) and answer it",
            )
        return None
    req = ctx.tx.payment_requests.get(rid)
    if req is None or req.session != c.session:
        raise BusinessRule(
            code="payment_request_not_found",
            message="this store issued no such payment request",
            field="payment.request_id",
            hint="answer a payment request this store issued for this checkout",
        )
    if req.checkout_id != ck.id:
        raise BusinessRule(
            code="payment_request_mismatch",
            message="the payment request is for another checkout",
            field="payment.request_id",
        )
    if req.status != "open":
        raise BusinessRule(
            code="payment_request_used",
            message=f"the payment request was already answered ({req.status}); each is good for one payment",
            field="payment.request_id",
            details=({"status": req.status},),
            hint="ask the store for a new payment request",
        )
    if req.expires_at < ctx.now:
        raise BusinessRule(
            code="payment_request_expired",
            message="the payment request has expired",
            field="payment.request_id",
            hint="ask the store for a new payment request",
        )
    return req


def _within_limits(ctx: Ctx, ck: Checkout, cred: VaultedCredential | None, req: PaymentRequest | None) -> None:
    """The final total against the credential's allowance and the request's amount. Raises when it is outside."""
    total, currency = ck.totals.total, ctx.cfg.market.currency
    if cred is not None and cred.currency != currency:
        raise BusinessRule(
            code="allowance_currency_mismatch",
            message=f"the allowance is in {cred.currency}; this checkout is in {currency}",
            field="payment.token",
        )
    if cred is not None and total > cred.max_amount:
        raise BusinessRule(
            code="allowance_exceeded",
            message=f"the checkout total {total} exceeds the allowance of {cred.max_amount} {currency}",
            field="payment.token",
            details=({"total": total, "max_amount": cred.max_amount},),
            hint="vault the credential again with an allowance that covers the total, or reduce the cart",
        )
    if req is not None and (req.currency != currency or req.amount != total):
        raise BusinessRule(
            code="payment_amount_mismatch",
            message=f"the payment request is for {req.amount} {req.currency}; the checkout total is {total} {currency}",
            field="payment.request_id",
            details=({"requested": req.amount, "total": total},),
            hint="ask the store for a new payment request at the current total",
        )


def _decline(ctx: Ctx, ck: Checkout, pay: Payment, code: str) -> BusinessRule:
    ctx.transition(ctx.tx.payments, pay, "declined", decline_code=code)
    ctx.transition(ctx.tx.checkouts, ck, "ready_for_complete", escalation=None)
    answer_request(ctx, pay, "failed")
    return BusinessRule(
        code="payment_declined",
        message=f"the payment was declined ({code})",
        details=({"decline_code": code, **brokers.decline_details(ctx.cfg.psp, code)},),
        hint=RETRY_HINT,
    )


def _challenge(ctx: Ctx, ck: Checkout, pay: Payment, outcome: str) -> None:
    chal = ctx.new_id("chal")
    ctx.transition(
        ctx.tx.payments, pay, "requires_action", challenge_id=chal, challenge_outcome=outcome, authentication="pending"
    )
    ctx.transition(ctx.tx.checkouts, ck, "requires_escalation", escalation="payment_action_required")
    ctx.emit("payment.challenge_issued", payment_id=pay.id, challenge_id=chal, checkout_id=ck.id)
    ctx.emit("payment.challenge_scripted", sealed=True, challenge_id=chal, outcome=outcome)
    ctx.refs.update(action_required=True, challenge_id=chal)


THREE_DS_DECLINE = "authentication_failed"
"""The core's decline code for a payment the issuer authenticated and declined (D111)."""


def _authenticate(ctx: Ctx, ck: Checkout, pay: Payment, spec: psp.TokenSpec | None, outcome: str) -> CoreError | None:
    """A payment the issuer wants 3-D Secure for: decided at once where the store decides it (D111), else parked on
    a challenge the buyer answers (D71)."""
    if ctx.cfg.three_ds is not None:
        return _three_ds_decided(ctx, ck, pay, spec, outcome)
    _challenge(ctx, ck, pay, outcome)
    return None


def _three_ds_decided(
    ctx: Ctx, ck: Checkout, pay: Payment, spec: psp.TokenSpec | None, outcome: str
) -> CoreError | None:
    """3-D Secure as a plain accept or decline: the issuer authenticates without a challenge
    and approves (frictionless, then authorization as any payment), or authenticates and declines, answered in each
    rail's own decline shape. The scripted outcome is kept on the payment (sealed), so the same card is declined again
    on a retry, as its bank would; another card is another bank."""
    decided = ctx.cfg.three_ds
    assert decided is not None
    google_pay = (spec is not None and spec.kind == "google_pay") or pay.namespace == "com.google.pay"
    rail = "google_pay" if google_pay else decided.rail
    accepted = outcome == "pass" and not decided.abandon
    if accepted:
        pay = ctx.transition(
            ctx.tx.payments, pay, "authenticated", authentication="frictionless", challenge_outcome=outcome
        )
        ctx.emit(
            "payment.three_ds", payment_id=pay.id, checkout_id=ck.id, result="accepted", decline_code=None, rail=rail
        )
        return _issuer_answer(ctx, ck, pay, spec)
    how = "frictionless_declined"
    pay = ctx.transition(
        ctx.tx.payments,
        pay,
        "authentication_failed",
        decline_code=THREE_DS_DECLINE,
        authentication=how,
        challenge_outcome=outcome,
    )
    ctx.emit(
        "payment.three_ds",
        payment_id=pay.id,
        checkout_id=ck.id,
        result="declined",
        decline_code=THREE_DS_DECLINE,
        rail=rail,
    )
    ctx.transition(ctx.tx.checkouts, ck, "ready_for_complete", escalation=None)
    answer_request(ctx, pay, "failed")
    return BusinessRule(
        code="payment_authentication_failed",
        message="the buyer's bank did not authenticate the payment",
        details=(
            {
                "decline_code": THREE_DS_DECLINE,
                **brokers.decline_details(ctx.cfg.psp, THREE_DS_DECLINE, authentication=how),
            },
        ),
        hint=RETRY_HINT,
    )


def _challenge_due(ctx: Ctx, ck: Checkout, pay: Payment, spec: psp.TokenSpec | None) -> str | None:
    """How the issuer will answer the challenge this payment meets, or None when it meets none: a due three_ds
    fault, else a challenge an earlier payment of the checkout still owes, else the token's own."""
    forced = consume(ctx, "three_ds")
    if forced is not None:
        return str(forced.params.get("result", "pass"))
    return _owed_challenge(ctx, ck, pay) or (spec.challenge if spec else None)


def _owed_challenge(ctx: Ctx, ck: Checkout, pay: Payment) -> str | None:
    """The challenge an earlier payment of this checkout with the same credential was asked for and never passed
    (its challenge was voided by a new completion, or failed): that card's bank still asks for it, with the same
    sealed answer; another card is another bank. Without it a forced
    3-D Secure fault (times 1) was bypassed by completing again, and the next payment was approved unauthenticated
    (the v1 review's #5)."""
    earlier = [
        p for p in ctx.tx.payments.all().values() if p.checkout_id == ck.id and p.id != pay.id and p.token == pay.token
    ]
    if any(p.authentication == "challenge_passed" for p in earlier):
        return None
    asked = [p for p in earlier if p.challenge_outcome is not None]
    return asked[-1].challenge_outcome if asked else None


def _forced_decline(ctx: Ctx) -> str | None:
    """The decline code of a due `payment_decline` fault, counted as fired; None when none is due."""
    fault = consume(ctx, "payment_decline")
    return None if fault is None else str(fault.params.get("code", "generic_decline"))


def _issuer_answer(ctx: Ctx, ck: Checkout, pay: Payment, spec: psp.TokenSpec | None) -> CoreError | None:
    """Authorization by the token: a decline token declines, any other credential is approved and captured. A token
    whose issuer authenticates without a challenge (frictionless 3-D Secure, D71) is authenticated first."""
    if spec is not None and spec.frictionless and pay.status == "pending":
        pay = ctx.transition(ctx.tx.payments, pay, "authenticated", authentication="frictionless")
    if spec is not None and spec.decline_code is not None:
        return _decline(ctx, ck, pay, spec.decline_code)
    finalize(ctx, ck, pay)
    return None


def complete_checkout(ctx: Ctx, c: CompleteCheckout) -> CoreError | None:
    ck = ctx.require(ctx.tx.checkouts, c.checkout_id, "checkout_not_found")
    ctx.refs["checkout_id"] = ck.id
    if ck.status == "requires_escalation":
        ck = settle(ctx, ck)
    if ck.status == "completed":
        raise IllegalTransition(
            code="checkout_completed", message="checkout is already completed; its order stands", field="status"
        )
    if ck.status == "canceled":
        raise IllegalTransition(code="checkout_canceled", message="checkout was canceled", field="status")
    if ck.status != "ready_for_complete":
        raise IllegalTransition(
            code="checkout_not_ready",
            message=f"checkout is {ck.status}; buyer email, a destination and a shipping option are required",
            field="status",
        )
    namespace = ctx.cfg.handlers.get(c.payment.handler)
    if namespace is None:
        ctx.transition(ctx.tx.checkouts, ck, "requires_escalation", escalation="payment_handler_unavailable")
        return BusinessRule(
            code="payment_handler_unavailable",
            message=f"handler {c.payment.handler!r} is not served here",
            hint=f"pay with one of {sorted(ctx.cfg.handlers)}, or a human can finish at continue_url",
        )
    rule = psp.HANDLER_RULES.get(namespace, psp.HandlerRule())
    cred = _vaulted(ctx, c, ck, rule)
    req = _requested(ctx, c, ck, rule)
    shop = _shop_token(ctx, c, rule)
    spec: psp.TokenSpec | None
    token: str | None
    method: str | None
    credential_id: str | None
    if shop is not None:
        spec = psp.TOKENS[shop.token]
        token, method, credential_id = shop.id, spec.method, shop.id
    else:
        token = cred.token if cred is not None else c.payment.token
        method, spec = _method(ctx, c.payment.instrument, token, namespace)
        credential_id = cred.id if cred is not None else None
    ck, err = ensure_stock(ctx, ck)
    if err is not None:
        return err
    _within_limits(ctx, ck, cred if cred is not None else shop, req)
    ck = ctx.transition(ctx.tx.checkouts, ck, "complete_in_progress")
    pay = Payment(
        ctx.new_id("pay"),
        ck.id,
        "pending",
        c.payment.handler,
        namespace,
        method,
        ctx.cfg.market.currency,
        ck.totals.total,
        token=token,
        brand=spec.brand if spec else None,
        request_id=req.id if req is not None else None,
        credential_id=credential_id,
        idempotency_key=c.idempotency_key,
    )
    ctx.tx.payments.put(pay.id, pay)
    ctx.emit(
        "payment.created",
        id=pay.id,
        checkout_id=ck.id,
        amount=pay.amount,
        handler=pay.handler,
        namespace=namespace,
        instrument=method,
        brand=pay.brand,
        request_id=pay.request_id,
        credential_id=pay.credential_id,
    )
    ck = replace(ck, payment_id=pay.id)
    ctx.tx.checkouts.put(ck.id, ck)
    ctx.refs["payment_id"] = pay.id
    answer_request(ctx, pay, "pending")
    if shop is not None:
        violation = _shop_decline(ctx, shop, ck)
        if violation is not None:
            return _decline(ctx, ck, pay, violation)
    declined = _forced_decline(ctx)
    if declined is not None:
        return _decline(ctx, ck, pay, declined)
    outcome = _challenge_due(ctx, ck, pay, spec)
    if outcome is not None:
        return _authenticate(ctx, ck, pay, spec, outcome)
    return _issuer_answer(ctx, ck, pay, spec)


def confirm_payment_action(ctx: Ctx, c: ConfirmPaymentAction) -> CoreError | None:
    ck = ctx.require(ctx.tx.checkouts, c.checkout_id, "checkout_not_found")
    ctx.refs["checkout_id"] = ck.id
    pay = ctx.tx.payments.get(ck.payment_id)
    waiting = ck.status == "requires_escalation" and ck.escalation == "payment_action_required"
    if not waiting or pay is None or pay.status != "requires_action":
        raise IllegalTransition(
            code="no_action_pending", message="this checkout is not waiting for the buyer's payment step"
        )
    ctx.refs.update(payment_id=pay.id, challenge_id=pay.challenge_id)
    if not c.approved or pay.challenge_outcome != "pass":
        reason = "challenge_abandoned" if not c.approved else "authentication_failed"
        how = "challenge_abandoned" if not c.approved else "challenge_failed"
        pay = ctx.transition(ctx.tx.payments, pay, "authentication_failed", decline_code=reason, authentication=how)
        ctx.transition(ctx.tx.checkouts, ck, "ready_for_complete", escalation=None)
        answer_request(ctx, pay, "failed")
        return BusinessRule(
            code="payment_authentication_failed",
            message="the buyer's bank did not authenticate the payment"
            if c.approved
            else "the buyer did not complete the payment step",
            details=({"decline_code": reason, **brokers.decline_details(ctx.cfg.psp, reason, authentication=how)},),
            hint=RETRY_HINT,
        )
    pay = ctx.transition(ctx.tx.payments, pay, "authenticated", authentication="challenge_passed")
    ck = ctx.transition(ctx.tx.checkouts, ck, "ready_for_complete", escalation=None)
    ck, err = ensure_stock(ctx, ck)
    if err is not None:
        pay = ctx.transition(ctx.tx.payments, pay, "voided")
        answer_request(ctx, pay, "failed")
        return err
    ck = ctx.transition(ctx.tx.checkouts, ck, "complete_in_progress")
    declined = _forced_decline(ctx)
    if declined is not None:
        return _decline(ctx, ck, pay, declined)
    return _issuer_answer(ctx, ck, pay, _issuer_spec(ctx, pay))

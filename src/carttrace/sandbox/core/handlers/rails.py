"""The payment rails beyond a card token: credentials the PSP vaulted with an allowance, payment requests, and the
mandates a store keeps.

Why in the core. Each of these is a fact a later command depends on and a grader checks: a delegated credential may be
charged once, for one checkout, within its allowance and before its expiry (ACP); a payment request is answered once,
for exactly its amount, before it expires (an MPP challenge, an AP2 cart); a mandate is what a party signed. Kept in
the state and changed only here, they replay with the log, and a surface cannot forget a rule: a replayed MPP
credential finds its request already answered, an AP2 payment for a stale cart finds the amount it was issued for.

Nothing here knows the protocols by name; a surface says which protocol a request or mandate belongs to, as data.
"""

from __future__ import annotations

from carttrace.core.canon import digest
from carttrace.sandbox.core import psp
from carttrace.sandbox.core.commands import MintShopToken, RecordMandate, RequestPayment, VaultCredential
from carttrace.sandbox.core.context import Ctx
from carttrace.sandbox.core.errors import BusinessRule, IllegalTransition, NotFound
from carttrace.sandbox.core.state import Mandate, Payment, PaymentRequest, VaultedCredential

REQUESTABLE = ("ready_for_complete", "requires_escalation")
"""Checkout statuses a payment request may be issued for: its total is final, and completion may still come."""


def vault_credential(ctx: Ctx, c: VaultCredential) -> None:
    spec = psp.TOKENS.get(c.token)
    if spec is None:
        raise BusinessRule(
            code="payment_credential_invalid",
            message="the payment processor cannot vault a credential it did not issue",
            field="token",
            hint="vault a token the shopper's wallet holds; card numbers are never accepted",
        )
    if c.expires_at <= ctx.now:
        raise BusinessRule(
            code="allowance_expired",
            message="the allowance expires before the credential could be used",
            field="expires_at",
            hint="set the allowance's expiry in the future",
        )
    vid = ctx.new_id("vt")
    cred = VaultedCredential(
        vid, c.session, "active", c.token, c.max_amount, c.currency, c.checkout_id, c.expires_at, c.merchant, ctx.now
    )
    ctx.tx.credentials.put(vid, cred)
    ctx.emit(
        "credential.vaulted",
        id=vid,
        session=c.session,
        checkout_id=c.checkout_id,
        max_amount=c.max_amount,
        currency=c.currency,
        expires_at=c.expires_at,
        brand=spec.brand,
    )
    ctx.refs["credential_id"] = vid


def mint_shop_token(ctx: Ctx, c: MintShopToken) -> None:
    """Mint a single-use Shop Token for one checkout (D73). The wallet's credential must be a shop_token the PSP
    knows; the checkout must be this session's, payable, and total exactly what the buyer confirmed (the spec's
    checkout and merchant scope: "context about the checkout and merchant, preventing usage outside of the verified
    checkout"). The token's validity runs from now for the credential's TTL on the logical clock (the spec's
    time-limit), and a capture consumes it (the spec's single-use). The token id is `shop_` plus the digest the other
    core ids take, so the same commands mint the same token on every run."""
    spec = psp.TOKENS.get(c.token)
    if spec is None or spec.kind != "shop_pay":
        raise BusinessRule(
            code="payment_credential_invalid",
            message="the Shop Pay interface does not know this credential",
            field="token",
            hint="mint from the Shop Pay credential the shopper's wallet holds; card numbers are never accepted",
        )
    ck = ctx.require(ctx.tx.checkouts, c.checkout_id, "checkout_not_found")
    if ck.session != c.session:
        raise NotFound(code="checkout_not_found", message=f"no checkout {c.checkout_id!r}", field="checkout_id")
    ctx.refs["checkout_id"] = ck.id
    if ck.status not in REQUESTABLE or ck.totals.fulfillment is None:
        raise IllegalTransition(
            code="checkout_not_ready",
            message=f"checkout is {ck.status}; a Shop Token mints only for a final total",
            field="status",
            hint="give the checkout a buyer email, a destination and a shipping option first",
        )
    if c.currency != ctx.cfg.market.currency or c.amount != ck.totals.total:
        raise BusinessRule(
            code="payment_amount_mismatch",
            message=f"a Shop Token is scoped to the checkout total, {ck.totals.total} {ctx.cfg.market.currency}",
            field="amount",
            hint="ask the buyer to confirm the checkout's current total in the Shop Pay interface",
        )
    if ctx.cfg.shop_id is None:
        raise BusinessRule(
            code="payment_handler_unavailable",
            message="this store serves no Shop Pay handler to mint a token for",
            field="token",
        )
    ttl = spec.ttl_ms if spec.ttl_ms is not None else psp.SHOP_TOKEN_TTL_MS
    n = ctx.tx.id_counter + 1
    ctx.tx.id_counter = n
    token = f"{psp.SHOP_TOKEN_PREFIX}{digest(['shop', ctx.tx.seed, n], 12)}"
    cred = VaultedCredential(
        token, c.session, "active", c.token, c.amount, c.currency, ck.id, ctx.now + ttl, ctx.cfg.shop_id, ctx.now
    )
    ctx.tx.credentials.put(token, cred)
    ctx.emit(
        "shop_token.issued",
        id=token,
        session=c.session,
        checkout_id=ck.id,
        amount=c.amount,
        currency=c.currency,
        expires_at=cred.expires_at,
        merchant=cred.merchant,
    )
    ctx.refs["credential_id"] = token


def request_payment(ctx: Ctx, c: RequestPayment) -> None:
    ck = ctx.require(ctx.tx.checkouts, c.checkout_id, "checkout_not_found")
    if ck.session != c.session:
        raise NotFound(code="checkout_not_found", message=f"no checkout {c.checkout_id!r}", field="checkout_id")
    ctx.refs["checkout_id"] = ck.id
    if ck.status not in REQUESTABLE or ck.totals.fulfillment is None:
        raise IllegalTransition(
            code="checkout_not_ready",
            message=f"checkout is {ck.status}; a payment can be requested only for a final total",
            field="status",
            hint="give the checkout a buyer email, a destination and a shipping option first",
        )
    if ctx.tx.payment_requests.get(c.request_id) is not None:
        raise BusinessRule(
            code="payment_request_exists",
            message="a payment request with this id was already issued",
            field="request_id",
        )
    if c.currency != ctx.cfg.market.currency or c.amount != ck.totals.total:
        raise BusinessRule(
            code="payment_amount_mismatch",
            message=f"a payment request must ask for the checkout total, {ck.totals.total} {ctx.cfg.market.currency}",
            field="amount",
        )
    if c.expires_at <= ctx.now:
        raise BusinessRule(code="payment_request_expired", message="the request expires before it is issued")
    req = PaymentRequest(
        c.request_id, ck.session, ck.id, c.protocol, "open", c.amount, c.currency, c.expires_at, ctx.now
    )
    ctx.tx.payment_requests.put(req.id, req)
    ctx.emit(
        "payment_request.issued",
        id=req.id,
        checkout_id=ck.id,
        protocol=c.protocol,
        amount=c.amount,
        currency=c.currency,
        expires_at=c.expires_at,
    )
    ctx.refs["request_id"] = req.id


def record_mandate(ctx: Ctx, c: RecordMandate) -> None:
    existing = ctx.tx.mandates.get(c.mandate_id)
    if existing is not None:
        if existing.digest != c.digest or existing.kind != c.kind:
            raise BusinessRule(
                code="mandate_conflict",
                message=f"mandate {c.mandate_id!r} was recorded with other contents",
                field="mandate_id",
                hint="a mandate id names one mandate; a changed mandate is a new mandate with a new id",
            )
        ctx.emit("mandate.repeated", id=c.mandate_id, kind=c.kind)
        ctx.refs["mandate_id"] = c.mandate_id
        return
    for ref in c.ref_ids:
        if ctx.tx.mandates.get(ref) is None:
            raise NotFound(code="mandate_not_found", message=f"no recorded mandate {ref!r}", field="ref_ids")
    if c.checkout_id is not None:
        ck = ctx.require(ctx.tx.checkouts, c.checkout_id, "checkout_not_found")
        if ck.session != c.session:
            raise NotFound(code="checkout_not_found", message=f"no checkout {c.checkout_id!r}", field="checkout_id")
    if c.request_id is not None:
        ctx.require(ctx.tx.payment_requests, c.request_id, "payment_request_not_found")
    m = Mandate(
        c.mandate_id,
        c.session,
        c.kind,
        c.protocol,
        c.digest,
        c.signer,
        ctx.now,
        c.checkout_id,
        c.request_id,
        tuple(c.ref_ids),
        c.expires_at,
    )
    ctx.tx.mandates.put(m.id, m)
    ctx.emit(
        "mandate.recorded",
        id=m.id,
        kind=m.kind,
        protocol=m.protocol,
        digest=m.digest,
        signer=m.signer,
        checkout_id=m.checkout_id,
        request_id=m.request_id,
        ref_ids=list(m.ref_ids),
    )
    ctx.refs["mandate_id"] = m.id


# -- what payments do to requests and credentials ----------------------------------------------------------------------


def answer_request(ctx: Ctx, pay: Payment, status: str) -> None:
    """Move the payment request a payment answers to `status` (pending, paid or failed)."""
    if pay.request_id is None:
        return
    req = ctx.require(ctx.tx.payment_requests, pay.request_id, "payment_request_not_found")
    if req.status != status:
        ctx.transition(ctx.tx.payment_requests, req, status, payment_id=pay.id)


def consume_credential(ctx: Ctx, pay: Payment) -> None:
    """A captured payment uses up the vaulted credential it charged."""
    if pay.credential_id is None:
        return
    cred = ctx.require(ctx.tx.credentials, pay.credential_id, "credential_not_found")
    ctx.transition(ctx.tx.credentials, cred, "consumed", payment_id=pay.id)


def void_open_requests(ctx: Ctx, checkout_id: str) -> None:
    """Requests still open for a checkout that completed some other way, or was canceled, can never be answered."""
    for req in list(ctx.tx.payment_requests.all().values()):
        if req.checkout_id == checkout_id and req.status == "open":
            ctx.transition(ctx.tx.payment_requests, req, "void")

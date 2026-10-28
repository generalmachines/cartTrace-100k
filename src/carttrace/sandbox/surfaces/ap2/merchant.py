"""AP2's merchant role in the sandbox, for the vendored releases: mandates in, merchant-signed carts out, the user's
payment authorization verified and settled through the core PSP.

v0.1.0 (pydantic types, mirrored in `models`):
- an `IntentMandate` is accepted (unsigned in v0.1.0, so it must require the user's cart confirmation), checked for
  expiry and the merchants it allows, recorded, and answered with the catalog's matching items;
- a basket becomes a core checkout. Presenting the intent mandate with it links the cart to the intent and holds the
  basket to the intent's skus. Once the checkout is ready (an email, an address, a shipping option) the merchant
  issues a payment request for its total and returns a `CartMandate` whose `merchant_authorization` is an ES256 JWT
  over the cart contents' hash;
- a `PaymentMandate` must carry the user's authorization: a JWS by the shopper's key over the hashes of the cart
  mandate this merchant issued and of the payment mandate contents. A cart the merchant did not issue (a tampered
  price, another item) or a total other than the request's is refused; otherwise the payment settles in the core.

v0.2.0 (JSON Schemas, `code/sdk/schemas/ap2`):
- an open checkout mandate (a JWS by the shopper) is the intent: its allowed merchants and line item constraints;
- a basket becomes a checkout, and the merchant returns its `checkout_jwt` (ES256) and `checkout_hash`; the payment
  request's id is the checkout hash, which the payment mandate names as `transaction_id`;
- completion takes the user's closed checkout mandate and payment mandate (JWS by the shopper): the checkout JWT must
  verify under the merchant's key and hash to the checkout hash, and the payment must name this payee, the request's
  amount and the transaction. The merchant answers with a checkout receipt and a payment receipt.

What a reader sees. Every reply about a checkout or cart, at either release, carries the merchant's checkout
(`checkout`: its line items, totals and currency), whether it is still incomplete, signed, settled or failed (D66 for
v0.2.0, D71 for v0.1.0, which also reads a cart by its id): AP2 types no read of a basket before it is signed, and an
agent, the harness and the runner's read-back all need its lines and total.

Pickup. W3C PaymentRequest, which both releases' carts use, has a `shipping_type` of `pickup`; the store's pickup
locations are offered beside its shipping rates (`shipping_options`, ids such as `pickup-1`, amount 0), and a basket
whose `shipping_option_id` names one is collected in store with no shipping address. Before 2026-10-15 the merchant
shipped only and asked every pickup order for an address: 37 of the 50 AP2 rows whose shopper would collect from the
store gave up that day (store-516.example: "the store requires a pickup location").

The challenge step. Neither release's types define a challenge object (v0.2.0 receipts are Success or Error), so when
the issuer asks for 3-D Secure the sandbox answers 202 `challenge_required` with the challenge page the buyer completes,
and the outcome is read afterwards. Every mandate in or out is recorded in the core (`record_mandate`) by its digest,
with the key that signed it, so the event log says what each party authorized.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

import pydantic

from carttrace.core.cards import looks_like_card_number
from carttrace.core.ids import stable_id
from carttrace.core.money import exponent, to_major
from carttrace.protocols.registry import registry
from carttrace.sandbox.core.config import REQUESTED_NAMESPACE
from carttrace.sandbox.core.errors import BusinessRule, CoreError, InvalidCommand, NotFound, PolicyDenied
from carttrace.sandbox.core.guard import find_card_number
from carttrace.sandbox.core.state import Checkout, Mandate, PaymentRequest
from carttrace.sandbox.events.runtime import Outcome
from carttrace.sandbox.presented import presented
from carttrace.sandbox.surfaces.ap2 import models
from carttrace.sandbox.surfaces.ap2.keys import BadSignature, PartyKey, mandate_hash, party_key, read_jws, text_hash
from carttrace.sandbox.surfaces.errors import UnsupportedVersion
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.keys import opaque_key
from carttrace.sandbox.surfaces.problems import json_path, problems
from carttrace.sandbox.surfaces.ucp.catalog import Catalog
from carttrace.sandbox.surfaces.ucp.messages import fault_for
from carttrace.sandbox.surfaces.values import obj, text

SERVED = ("v0.1.0", "v0.2.0")
CART_TTL_MS = 15 * 60 * 1000
"""How long a merchant-signed cart (and its payment request) stays payable, in logical milliseconds."""
SCHEMAS = "code/sdk/schemas/ap2"
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MANDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
TOKEN_PREFIXES = ("tok_", "vt-")
METHODS = frozenset({"card", "basic-card"})
"""Payment method names a v0.1.0 payment response may name; the cart's method data offers `CARD`."""
ERRORS: Mapping[str, type[CoreError]] = {
    "invalid": InvalidCommand,
    "not_found": NotFound,
    "policy": PolicyDenied,
    "business": BusinessRule,
}


def check_version(version: str | None) -> str:
    if version not in SERVED:
        raise UnsupportedVersion(
            f"the sandbox has no AP2 merchant for {version!r}", hint=f"serve a profile at one of {list(SERVED)}"
        )
    return str(version)


def iso(ms: int) -> str:
    return (EPOCH + timedelta(milliseconds=ms)).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class Ap2Reply:
    status: int
    body: dict[str, Any]


class Refused(Exception):
    """A request the merchant refuses: the reply to send and the typed error to log."""

    def __init__(self, status: int, code: str, message: str, kind: str = "invalid", **extra: Any) -> None:
        super().__init__(message)
        self.reply = Ap2Reply(status, {"error": code, "error_description": message, **extra})
        self.error: CoreError = ERRORS[kind](code=code, message=message, field=extra.get("path"))


class Ap2Merchant:
    """The merchant of one store at one AP2 release."""

    def __init__(self, host: SandboxHost) -> None:
        self.host = host
        self.version = check_version(host.require("ap2"))
        self.catalog = Catalog(host.pack)
        self.merchant = presented(host.pack).merchant_id
        self.processor = presented(host.pack).acp_psp
        """The card processor the payment request names: the store's presented PSP (D106)."""
        self.merchant_key: PartyKey = party_key(host.pack.id, "merchant")
        self.shopper_key: PartyKey = party_key(host.pack.id, "shopper")

    # -- shared steps --------------------------------------------------------------------------------------------------

    @property
    def now_s(self) -> int:
        return self.host.state.clock // 1000

    def refuse(self, command: str, refused: Refused) -> Ap2Reply:
        """Log a refusal as one entry and answer it."""
        self.host.reject(command, refused.error)
        return refused.reply

    def _record(self, session: str, mandate_id: str, kind: str, digest: str, **fields: Any) -> Outcome:
        command: dict[str, Any] = {
            "type": "record_mandate",
            "actor": "merchant",
            "session": session,
            "mandate_id": mandate_id,
            "kind": kind,
            "protocol": "ap2",
            "digest": digest,
            **fields,
        }
        out = self.host.execute(command)
        if not out.ok:
            err = out.error or {}
            raise Refused(409, str(err.get("code")), str(err.get("message")), "business")
        return out

    def _checkout_of(self, session: str, checkout_id: str | None) -> Checkout | None:
        ck = self.host.state.checkouts.get(checkout_id or "")
        return ck if ck is not None and ck.session == session else None

    def _amount(self, minor: int) -> dict[str, Any]:
        """A W3C PaymentCurrencyAmount: v0.1.0 types its value as a float, in major units."""
        currency = self.host.state.currency
        return {"currency": currency, "value": float(to_major(minor, currency) or "0")}

    def _products(self, words: str, skus: Sequence[str] | None) -> list[dict[str, Any]]:
        """Catalog items matching an intent: by its words, or exactly its skus."""
        state = self.host.state
        wanted = set(skus or ())
        products = self.catalog.search(state, words) if words.strip() and not wanted else self.catalog.products(state)
        found = []
        for p in products:
            for v in p.variants:
                if wanted and v.id not in wanted:
                    continue
                title = self.catalog.variant_title(state, v.id)
                found.append({"sku": v.id, "title": title, "price": self._amount(v.price), "available": v.available})
        return found[:20]

    def _basket(
        self, session: str, body: Mapping[str, Any], *, allowed: frozenset[str] | None
    ) -> tuple[Checkout | None, Outcome]:
        """Create or update the core checkout a basket request describes (`cart_id` or `checkout_id` to update)."""
        lines: dict[str, int] = {}
        for i, raw in enumerate(body.get("items") or ()):
            item = obj(raw)
            sku, qty = text(item.get("sku") or item.get("id")), item.get("quantity", 1)
            if not sku or not isinstance(qty, int) or isinstance(qty, bool) or qty < 1:
                raise Refused(
                    400,
                    "invalid_basket",
                    "Each item needs a sku and a whole quantity of at least 1",
                    path=f"$.items[{i}]",
                )
            if allowed is not None and sku not in allowed:
                raise Refused(
                    403,
                    "not_in_intent",
                    "An item is outside what the intent mandate allows",
                    "policy",
                    path=f"$.items[{i}].sku",
                )
            lines[sku] = lines.get(sku, 0) + qty
        fields: dict[str, Any] = {}
        if lines:
            fields["lines"] = [{"variant_id": v, "quantity": q} for v, q in lines.items()]
        buyer = {"email": text(body.get("payer_email")), "phone_number": text(body.get("payer_phone"))}
        if any(buyer.values()):
            fields["buyer"] = {k: v for k, v in buyer.items() if v}
        group = _group(body, self._pickup_ids())
        if group is not None:
            fields["fulfillment"] = [group]
        target_id = text(body.get("cart_id") or body.get("checkout_id")) or None
        if target_id is not None:
            ck = self._checkout_of(session, target_id)
            if ck is None:
                raise Refused(404, "cart_not_found", "No cart with this id", "not_found")
            out = self.host.execute({"type": "update_checkout", "session": session, "checkout_id": ck.id, **fields})
        elif "lines" in fields:
            out = self.host.execute({"type": "create_checkout", "session": session, **fields})
        else:
            raise Refused(400, "invalid_basket", "A new cart needs items", path="$.items")
        fault = fault_for(out.error, tier=self.host.tier(session))
        if fault is not None:
            raise Refused(fault.status, fault.reason, fault.message, "business")
        return self._checkout_of(session, str(out.refs.get("checkout_id") or target_id or "")), out

    def _pickup_ids(self) -> frozenset[str]:
        return frozenset(p.id for p in self.host.pack.config().market.pickup)

    def _pickup_options(self) -> list[dict[str, Any]]:
        """The store's pickup locations as W3C shipping options: collected in store, no address, no charge."""
        return [
            {
                "id": p.id,
                "label": f"Pickup: {p.title} (collect in store, no shipping address)",
                "amount": self._amount(0),
            }
            for p in self.host.pack.config().market.pickup
        ]

    def _incomplete(self, ck: Checkout, out: Outcome, key: str) -> Ap2Reply:
        """A cart that cannot be signed yet: what it still needs, the shipping options once it has an address, and
        the store's pickup locations, which need none."""
        pickup = next((g for g in ck.groups if g.method == "pickup"), None)
        group = next((g for g in ck.groups if g.method == "shipping"), None)
        missing = [] if ck.buyer.get("email") else ["payer_email"]
        if pickup is not None:
            if pickup.amount is None:
                missing.append("shipping_option_id")
        elif group is None or not group.destination:
            missing.append("shipping_address")
        elif group.amount is None:
            missing.append("shipping_option_id")
        options = [
            {"id": str(o["id"]), "label": str(o["title"]), "amount": self._amount(int(o["amount"]))}
            for o in (group.options if group is not None else ())
        ]
        options += [o for o in self._pickup_options() if o["id"] not in {x["id"] for x in options}]
        body: dict[str, Any] = {key: ck.id, "status": "incomplete", "missing": missing, "shipping_options": options}
        if out.error is not None:
            # what the store refused in this update (an address it does not ship to), as AP2's errors say it
            body["error"], body["error_description"] = str(out.error.get("code")), str(out.error.get("message"))
        unshipped = [e["data"] for e in out.messages() if e["type"] == "fulfillment.rejected"]
        if unshipped and "error" not in body:
            where = unshipped[0].get("country") or unshipped[0].get("location_id") or "there"
            body["error"] = str(unshipped[0].get("reason") or "destination_not_shippable")
            body["error_description"] = f"the store does not deliver to {where}"
        refused = [e["data"] for e in out.messages() if e["type"] == "line.rejected"]
        if refused:
            body["refused_items"] = [{"sku": r.get("variant_id"), "reason": r.get("reason")} for r in refused]
        body["checkout"] = self._checkout_payload(ck)
        return Ap2Reply(200, body)

    def _issue_request(self, session: str, ck: Checkout, request_id: str) -> PaymentRequest:
        out = self.host.execute(
            {
                "type": "request_payment",
                "actor": "merchant",
                "session": session,
                "request_id": request_id,
                "checkout_id": ck.id,
                "protocol": "ap2",
                "amount": ck.totals.total,
                "currency": self.host.state.currency,
                "expires_at": self.host.state.clock + CART_TTL_MS,
            }
        )
        if not out.ok:
            err = out.error or {}
            raise Refused(409, str(err.get("code")), str(err.get("message")), "business")
        return self.host.state.payment_requests[request_id]

    def _issued(self, session: str, request_id: str) -> tuple[PaymentRequest, Checkout, Mandate]:
        """The payment request a mandate names, its checkout, and the cart the merchant signed for it."""
        req = self.host.state.payment_requests.get(request_id)
        ck = self._checkout_of(session, req.checkout_id) if req is not None and req.session == session else None
        cart = next(
            (m for m in self.host.state.mandates.values() if m.kind == "cart" and m.request_id == request_id), None
        )
        if req is None or ck is None or cart is None:
            raise Refused(404, "cart_not_found", "This merchant signed no cart with this payment request", "not_found")
        if req.expires_at < self.host.state.clock:
            raise Refused(410, "cart_expired", "The signed cart has expired; ask for a new one", "business")
        if req.status != "open":
            raise Refused(
                409, "request_used", f"The cart's payment request was already answered ({req.status})", "business"
            )
        return req, ck, cart

    def _settle(
        self, session: str, ck: Checkout, req: PaymentRequest, *, token: str, key: str, base: str
    ) -> tuple[int, dict[str, Any]]:
        """Pay the request in the core: (HTTP status, what happened)."""
        out = self.host.execute(
            {
                "type": "complete_checkout",
                "session": session,
                "checkout_id": ck.id,
                "idempotency_key": key,
                "payment": {"handler": REQUESTED_NAMESPACE, "token": token, "request_id": req.id},
            }
        )
        fault = fault_for(out.error, tier=self.host.tier(session))
        if fault is not None:
            return fault.status, {"status": "failed", "error": fault.reason, "error_description": fault.message}
        if out.refs.get("action_required"):
            return 202, self._status(ck.id, base)
        status = self._status(ck.id, base)
        if status["status"] in ("failed", "pending") and out.error is not None:
            # a completion the store refused before any payment (stock gone at completion) fails as AP2's error
            # receipt says, rather than waiting as pending with no reason (the payments check, D94)
            status = {k: v for k, v in status.items() if k != "checkout_status"}
            status["status"] = "failed"
            status["error"], status["error_description"] = str(out.error.get("code")), str(out.error.get("message"))
        return (200 if status["status"] == "success" else 402), status

    def _status(self, checkout_id: str, base: str) -> dict[str, Any]:
        """Where a checkout's payment stands: success, a challenge the buyer must complete, a failure, or pending."""
        ck = self.host.state.checkouts[checkout_id]
        pay = self.host.state.payments.get(ck.payment_id or "")
        if ck.status == "completed":
            return {"status": "success", "order_id": ck.order_id, "payment_id": ck.payment_id}
        if pay is not None and pay.status == "requires_action":
            chal = str(pay.challenge_id)
            challenge = {
                "type": "3ds",
                "id": chal,
                "url": f"{base}/3ds/{chal}",
                "display_text": "The buyer's bank asks the buyer to confirm this payment",
            }
            return {"status": "challenge_required", "challenge": challenge, "payment_id": pay.id}
        if pay is not None and pay.status in ("declined", "authentication_failed", "voided"):
            reason = pay.decline_code or pay.status
            return {"status": "failed", "error": pay.status, "error_description": f"the payment failed ({reason})"}
        return {"status": "pending", "checkout_status": ck.status}

    # -- v0.1.0 --------------------------------------------------------------------------------------------------------

    def _accept_intent_v1(self, session: str, raw: Mapping[str, Any]) -> tuple[str, models.IntentMandate]:
        intent = _model(models.IntentMandate, raw, models.INTENT_MANDATE_DATA_KEY)
        expires = _epoch_ms(intent.intent_expiry)
        if expires is None or expires <= self.host.state.clock:
            raise Refused(
                400,
                "intent_expired",
                "intent_expiry must be a future ISO 8601 instant",
                "business",
                path="$.intent_expiry",
            )
        if not intent.user_cart_confirmation_required:
            raise Refused(
                400,
                "confirmation_required",
                "An unsigned intent mandate must require the user's cart confirmation (AP2 v0.1.0)",
                "policy",
            )
        if intent.merchants and self.merchant.lower() not in {m.lower() for m in intent.merchants}:
            raise Refused(403, "merchant_not_allowed", "This merchant is not among the intent's merchants", "policy")
        digest = mandate_hash(dict(raw))
        mid = stable_id("im", digest)
        self._record(session, mid, "intent", digest, signer=None, expires_at=expires)
        return mid, intent

    def intent_v1(self, session: str, body: Mapping[str, Any]) -> Ap2Reply:
        try:
            _no_card_number(body)
            mid, intent = self._accept_intent_v1(session, _member(body, models.INTENT_MANDATE_DATA_KEY))
        except Refused as r:
            return self.refuse("record_mandate", r)
        products = self._products(intent.natural_language_description, intent.skus)
        reply = {"intent_mandate_id": mid, "status": "accepted", "intent_expiry": intent.intent_expiry}
        return Ap2Reply(201, {**reply, "products": products})

    def cart_v1(self, session: str, body: Mapping[str, Any], base: str) -> Ap2Reply:
        try:
            _no_card_number(body)
            intent_id, allowed = None, None
            if isinstance(body.get(models.INTENT_MANDATE_DATA_KEY), Mapping):
                intent_id, intent = self._accept_intent_v1(session, body[models.INTENT_MANDATE_DATA_KEY])
                allowed = frozenset(intent.skus) if intent.skus else None
            ck, out = self._basket(session, body, allowed=allowed)
            if ck is None:
                err = out.error or {}
                raise Refused(400, str(err.get("code") or "cart_refused"), str(err.get("message") or "no cart"))
            if ck.status != "ready_for_complete":
                return self._incomplete(ck, out, "cart_id")
            req = self._issue_request(session, ck, stable_id("pr", ck.id, self.host.state.seq))
            mandate = self._cart_mandate(ck, req)
            digest = mandate_hash(mandate)
            self._record(
                session,
                stable_id("cm", digest),
                "cart",
                digest,
                signer=self.merchant_key.kid,
                checkout_id=ck.id,
                request_id=req.id,
                ref_ids=[intent_id] if intent_id else [],
                expires_at=req.expires_at,
            )
        except Refused as r:
            return self.refuse("create_checkout", r)
        reply = {models.CART_MANDATE_DATA_KEY: mandate, "cart_id": ck.id, "status": "ready"}
        return Ap2Reply(201, {**reply, "checkout": self._checkout_payload(ck)})

    def _cart_mandate(self, ck: Checkout, req: PaymentRequest) -> dict[str, Any]:
        """The cart the merchant signs: a W3C PaymentRequest for the checkout's total, valid until the request."""
        state = self.host.state
        window = self.host.pack.market().return_window_days
        items = [
            {
                "label": f"{self.catalog.variant_title(state, ln.variant_id)} x{ln.quantity}",
                "amount": self._amount(ln.unit_price * ln.quantity),
                "refund_period": window,
            }
            for ln in ck.lines
        ]
        t = ck.totals
        for label, amount in (("Discount", -t.discount), ("Shipping", t.fulfillment), ("Tax", t.tax)):
            if amount:
                items.append({"label": label, "amount": self._amount(amount), "refund_period": window})
        group = next((g for g in ck.groups if g.method == "shipping"), None)
        pickup = next((g for g in ck.groups if g.method == "pickup"), None)
        chosen = group or pickup
        options = [
            {
                "id": str(o["id"]),
                "label": str(o["title"]),
                "amount": self._amount(int(o["amount"])),
                "selected": chosen is not None and o["id"] == chosen.selected_option_id,
            }
            for o in (chosen.options if chosen is not None else ())
        ]
        request: dict[str, Any] = {
            "method_data": [
                {"supported_methods": "CARD", "data": {"payment_processor": self.processor, "livemode": False}}
            ],
            "details": {
                "id": req.id,
                "display_items": items,
                "shipping_options": options,
                "total": {"label": "Total", "amount": self._amount(t.total), "refund_period": window},
            },
            "options": {
                "request_payer_email": True,
                "request_shipping": pickup is None,
                "shipping_type": "pickup" if pickup is not None and group is None else "shipping",
            },
        }
        if group is not None and group.destination:
            request["shipping_address"] = _contact(group.destination)
        contents = {
            "id": ck.id,
            "user_cart_confirmation_required": True,
            "payment_request": request,
            "cart_expiry": iso(req.expires_at),
            "merchant_name": self.merchant,
        }
        claims = {
            "iss": self.merchant,
            "sub": self.merchant,
            "aud": "payment_processor",
            "iat": self.now_s,
            "exp": req.expires_at // 1000,
            "jti": req.id,
            "cart_hash": mandate_hash(contents),
        }
        return {"contents": contents, "merchant_authorization": self.merchant_key.sign(claims)}

    def payment_v1(self, session: str, body: Mapping[str, Any], base: str) -> Ap2Reply:
        try:
            _no_card_number(body)
            raw = _member(body, models.PAYMENT_MANDATE_DATA_KEY)
            mandate = _model(models.PaymentMandate, raw, models.PAYMENT_MANDATE_DATA_KEY)
            contents = mandate.payment_mandate_contents
            contents_hash = mandate_hash(dict(obj(raw.get("payment_mandate_contents"))))
            pmid = contents.payment_mandate_id
            if not _MANDATE_ID.match(pmid) or looks_like_card_number(pmid):
                raise Refused(
                    400,
                    "invalid_mandate_id",
                    "payment_mandate_id takes 1 to 128 letters, digits, _ . : -",
                    path="$.payment_mandate_contents.payment_mandate_id",
                )
            req, ck, cart = self._issued(session, contents.payment_details_id)
            if isinstance(body.get(models.CART_MANDATE_DATA_KEY), Mapping):
                self._verify_cart(body[models.CART_MANDATE_DATA_KEY], cart)
            self._verify_user_v1(mandate.user_authorization, cart.digest, contents_hash)
            self._check_total(req, contents.payment_details_total)
            token = self._check_response_v1(contents, req)
            self._record(
                session,
                pmid,
                "payment",
                contents_hash,
                signer=self.shopper_key.kid,
                checkout_id=ck.id,
                request_id=req.id,
                ref_ids=[cart.id],
            )
        except Refused as r:
            return self.refuse("complete_checkout", r)
        status, fields = self._settle(session, ck, req, token=token, key=f"ap2|{opaque_key(pmid)}", base=base)
        checkout = self._checkout_payload(self.host.state.checkouts[ck.id])
        return Ap2Reply(status, {"payment_mandate_id": pmid, "cart_id": ck.id, **fields, "checkout": checkout})

    def _check_response_v1(self, contents: models.PaymentMandateContents, req: PaymentRequest) -> str:
        """The payment response names the cart's request, a method the cart offers, this merchant, and an opaque
        credential; returns the credential."""
        response = contents.payment_response
        if response.request_id != req.id:
            raise Refused(400, "request_mismatch", "payment_response.request_id must name the cart's payment request")
        if response.method_name.lower() not in METHODS:
            raise Refused(400, "method_unsupported", "This merchant takes the CARD payment method")
        if contents.merchant_agent.lower() != self.merchant.lower():
            raise Refused(400, "merchant_mismatch", "merchant_agent must name this merchant")
        token = text(obj(response.details).get("token"))
        if not token.startswith(TOKEN_PREFIXES):
            raise Refused(
                400,
                "invalid_token",
                "payment_response.details.token takes a credential token from the shopper's wallet",
                path="$.payment_mandate_contents.payment_response.details.token",
            )
        return token

    def payment_status_v1(self, session: str, payment_mandate_id: str, base: str) -> Ap2Reply:
        m = self.host.state.mandates.get(payment_mandate_id)
        if m is None or m.session != session or m.kind != "payment" or m.checkout_id is None:
            self.host.reject("read", NotFound(code="mandate_not_found", message="No payment mandate with this id"))
            return Ap2Reply(404, {"error": "mandate_not_found", "error_description": "No payment mandate with this id"})
        self.host.execute({"type": "read", "session": session, "target": "checkout", "id": m.checkout_id})
        checkout = self._checkout_payload(self.host.state.checkouts[m.checkout_id])
        status = self._status(m.checkout_id, base)
        return Ap2Reply(200, {"payment_mandate_id": m.id, "cart_id": m.checkout_id, **status, "checkout": checkout})

    def cart_status_v1(self, session: str, cart_id: str, base: str) -> Ap2Reply:
        """A cart read by its id (the sandbox's REST binding, D71): where its payment stands and the merchant's
        checkout behind it, so the cart can be read back before and after it is paid, as at v0.2.0."""
        ck = self._checkout_of(session, cart_id)
        if ck is None:
            self.host.reject("read", NotFound(code="cart_not_found", message="No cart with this id"))
            return Ap2Reply(404, {"error": "cart_not_found", "error_description": "No cart with this id"})
        self.host.execute({"type": "read", "session": session, "target": "checkout", "id": ck.id})
        reply: dict[str, Any] = {"cart_id": ck.id, **self._status(ck.id, base)}
        if reply.get("status") == "pending" and ck.status != "ready_for_complete":
            reply["status"] = "incomplete"
        reply["checkout"] = self._checkout_payload(self.host.state.checkouts[ck.id])
        return Ap2Reply(200, reply)

    def _verify_cart(self, sent: Any, issued: Mandate) -> None:
        """A cart mandate presented back must be the one the merchant issued: signed by it, over these contents."""
        cart = obj(sent)
        try:
            claims = self.merchant_key.verify(text(cart.get("merchant_authorization")))
        except BadSignature:
            raise Refused(403, "cart_tampered", "The cart's merchant authorization does not verify", "policy") from None
        if claims.get("cart_hash") != mandate_hash(cart.get("contents")) or mandate_hash(dict(cart)) != issued.digest:
            raise Refused(403, "cart_tampered", "The cart mandate is not the cart this merchant signed", "policy")

    def _verify_user_v1(self, authorization: str | None, cart_hash: str, contents_hash: str) -> None:
        """The user's authorization: a JWS by the shopper's key binding the issued cart and these payment contents."""
        if not authorization:
            raise Refused(401, "authorization_required", "A payment mandate needs the user's authorization", "policy")
        try:
            claims = self.shopper_key.verify(authorization)
        except BadSignature:
            raise Refused(
                403, "authorization_invalid", "The user's authorization does not verify under the user's key", "policy"
            ) from None
        data = claims.get("transaction_data")
        if claims.get("aud") != self.merchant or not isinstance(data, list) or len(data) != 2:
            raise Refused(403, "authorization_invalid", "The user's authorization is not for this merchant", "policy")
        if data[0] != cart_hash:
            raise Refused(403, "cart_tampered", "The user authorized a cart this merchant did not issue", "policy")
        if data[1] != contents_hash:
            raise Refused(
                403,
                "authorization_invalid",
                "The user's authorization does not cover these payment mandate contents",
                "policy",
            )

    def _check_total(self, req: PaymentRequest, total: models.PaymentItem) -> None:
        minor = _minor(total.amount.value, total.amount.currency)
        if total.amount.currency.upper() != req.currency or minor != req.amount:
            says = f"The payment total must be the cart's, {to_major(req.amount, req.currency)} {req.currency}"
            raise Refused(
                409, "amount_mismatch", says, "business", path="$.payment_mandate_contents.payment_details_total"
            )

    # -- v0.2.0 --------------------------------------------------------------------------------------------------------

    def _accept_intent_v2(self, session: str, token: str) -> tuple[str, dict[str, Any]]:
        payload = self._user_jws(token, "open_checkout_mandate.json", "open_checkout_mandate")
        exp = payload.get("exp")
        if isinstance(exp, int) and exp * 1000 <= self.host.state.clock:
            raise Refused(400, "intent_expired", "The open checkout mandate has expired", "business")
        allowed = [
            obj(c) for c in payload.get("constraints") or () if obj(c).get("type") == "checkout.allowed_merchants"
        ]
        names = {text(obj(m).get(k)).lower() for c in allowed for m in c.get("allowed") or () for k in ("id", "name")}
        if allowed and self.merchant.lower() not in names:
            raise Refused(
                403, "merchant_not_allowed", "This merchant is not among the mandate's allowed merchants", "policy"
            )
        digest = text_hash(token)
        mid = stable_id("im", digest)
        self._record(
            session,
            mid,
            "intent",
            digest,
            signer=self.shopper_key.kid,
            expires_at=exp * 1000 if isinstance(exp, int) else None,
        )
        return mid, payload

    def intent_v2(self, session: str, body: Mapping[str, Any]) -> Ap2Reply:
        try:
            mid, payload = self._accept_intent_v2(session, text(body.get("open_checkout_mandate")))
        except Refused as r:
            return self.refuse("record_mandate", r)
        skus = sorted(_acceptable(payload))
        products = self._products("", skus) if skus else []
        return Ap2Reply(201, {"intent_mandate_id": mid, "status": "accepted", "products": products})

    def checkout_v2(self, session: str, body: Mapping[str, Any], base: str) -> Ap2Reply:
        try:
            _no_card_number(body)
            intent_id, allowed = None, None
            if isinstance(body.get("open_checkout_mandate"), str):
                intent_id, payload = self._accept_intent_v2(session, body["open_checkout_mandate"])
                acceptable = _acceptable(payload)
                allowed = frozenset(acceptable) if acceptable else None
            ck, out = self._basket(session, body, allowed=allowed)
            if ck is None:
                err = out.error or {}
                raise Refused(400, str(err.get("code") or "checkout_refused"), str(err.get("message") or "no checkout"))
            if ck.status != "ready_for_complete":
                return self._incomplete(ck, out, "checkout_id")
            checkout = self._checkout_payload(ck)
            expires_at = self.host.state.clock + CART_TTL_MS
            jti = stable_id("jti", ck.id, self.host.state.seq)
            claims = {**checkout, "iss": self.merchant, "iat": self.now_s, "exp": expires_at // 1000, "jti": jti}
            checkout_jwt = self.merchant_key.sign(claims)
            checkout_hash = text_hash(checkout_jwt)
            req = self._issue_request(session, ck, checkout_hash)
            self._record(
                session,
                stable_id("cm", checkout_hash),
                "cart",
                checkout_hash,
                signer=self.merchant_key.kid,
                checkout_id=ck.id,
                request_id=req.id,
                ref_ids=[intent_id] if intent_id else [],
                expires_at=req.expires_at,
            )
        except Refused as r:
            return self.refuse("create_checkout", r)
        reply = {"checkout_id": ck.id, "checkout_jwt": checkout_jwt, "checkout_hash": checkout_hash}
        return Ap2Reply(201, {**reply, "checkout": checkout})

    def _checkout_payload(self, ck: Checkout) -> dict[str, Any]:
        """The checkout the merchant signs; its items are UCP items (AP2 v0.2.0 `types/item.json`)."""
        state = self.host.state
        t = ck.totals
        totals = [{"type": "subtotal", "amount": t.subtotal}]
        for kind, amount in (
            ("discount", -t.discount if t.discount else None),
            ("fulfillment", t.fulfillment),
            ("tax", t.tax),
        ):
            if amount is not None:
                totals.append({"type": kind, "amount": amount})
        totals.append({"type": "total", "amount": t.total})
        line_items = [
            {
                "id": ln.id,
                "item": {
                    "id": ln.variant_id,
                    "title": self.catalog.variant_title(state, ln.variant_id),
                    "price": ln.unit_price,
                },
                "quantity": ln.quantity,
            }
            for ln in ck.lines
        ]
        merchant = {"id": self.merchant, "name": self.merchant}
        return {
            "id": ck.id,
            "merchant": merchant,
            "currency": state.currency,
            "line_items": line_items,
            "totals": totals,
        }

    def complete_v2(self, session: str, checkout_id: str, body: Mapping[str, Any], base: str) -> Ap2Reply:
        try:
            ck = self._checkout_of(session, checkout_id)
            if ck is None:
                raise Refused(404, "checkout_not_found", "No checkout with this id", "not_found")
            checkout_token = text(body.get("checkout_mandate"))
            payment_token = text(body.get("payment_mandate"))
            closed = self._user_jws(checkout_token, "checkout_mandate.json", "checkout_mandate")
            payment = self._user_jws(payment_token, "payment_mandate.json", "payment_mandate")
            _no_card_number({"checkout_mandate": closed, "payment_mandate": payment})
            checkout_hash = self._verify_checkout_jwt(text(closed.get("checkout_jwt")), ck, closed)
            req, _, cart = self._issued(session, checkout_hash)
            token = self._check_payment_v2(payment, req, checkout_hash)
            checkout_digest, payment_digest = text_hash(checkout_token), text_hash(payment_token)
            closed_id = stable_id("km", checkout_digest)
            fields = {"signer": self.shopper_key.kid, "checkout_id": ck.id, "request_id": req.id}
            self._record(session, closed_id, "checkout", checkout_digest, ref_ids=[cart.id], **fields)
            self._record(
                session, stable_id("pm", payment_digest), "payment", payment_digest, ref_ids=[closed_id], **fields
            )
        except Refused as r:
            return self.refuse("complete_checkout", r)
        key = f"ap2|{opaque_key(checkout_hash)}"
        status, outcome = self._settle(session, ck, req, token=token, key=key, base=base)
        reply: dict[str, Any] = {"checkout_id": ck.id, **outcome}
        reply.update(self._receipts(ck.id, checkout_digest, payment_digest, outcome))
        reply["checkout"] = self._checkout_payload(self.host.state.checkouts[ck.id])
        return Ap2Reply(status, reply)

    def _verify_checkout_jwt(self, checkout_jwt: str, ck: Checkout, closed: Mapping[str, Any]) -> str:
        """The checkout JWT verifies under the merchant's key, is for this checkout, and hashes to the mandate's
        checkout hash; returns the hash."""
        try:
            signed = self.merchant_key.verify(checkout_jwt)
        except BadSignature:
            raise Refused(
                403, "checkout_tampered", "The checkout JWT does not verify under this merchant's key", "policy"
            ) from None
        checkout_hash = text_hash(checkout_jwt)
        if signed.get("id") != ck.id or closed.get("checkout_hash") != checkout_hash:
            raise Refused(
                403, "checkout_tampered", "The checkout mandate is not over the checkout this merchant signed", "policy"
            )
        return checkout_hash

    def _check_payment_v2(self, mandate: Mapping[str, Any], req: PaymentRequest, checkout_hash: str) -> str:
        """The payment mandate names this transaction, this payee and the request's amount; returns its credential."""
        if mandate.get("transaction_id") != checkout_hash:
            raise Refused(
                403, "transaction_mismatch", "The payment mandate's transaction_id is not this checkout", "policy"
            )
        if text(obj(mandate.get("payee")).get("id")).lower() != self.merchant.lower():
            raise Refused(403, "payee_mismatch", "The payment mandate pays another payee", "policy")
        amount = obj(mandate.get("payment_amount"))
        if text(amount.get("currency")).upper() != req.currency or amount.get("amount") != req.amount:
            says = f"The payment amount must be the checkout's, {req.amount} {req.currency} in minor units"
            raise Refused(409, "amount_mismatch", says, "business", path="$.payment_mandate.payment_amount")
        exp = mandate.get("exp")
        if isinstance(exp, int) and exp * 1000 < self.host.state.clock:
            raise Refused(400, "mandate_expired", "The payment mandate has expired", "business")
        token = text(obj(mandate.get("payment_instrument")).get("id"))
        if looks_like_card_number(token):
            raise Refused(
                400,
                "card_number_refused",
                "A card number was sent; this merchant takes payment tokens only",
                path="$.payment_mandate.payment_instrument.id",
            )
        if not token.startswith(TOKEN_PREFIXES):
            raise Refused(
                400,
                "invalid_token",
                "payment_instrument.id takes a credential token from the shopper's wallet",
                path="$.payment_mandate.payment_instrument.id",
            )
        return token

    def checkout_status_v2(self, session: str, checkout_id: str, base: str) -> Ap2Reply:
        ck = self._checkout_of(session, checkout_id)
        if ck is None:
            self.host.reject("read", NotFound(code="checkout_not_found", message="No checkout with this id"))
            return Ap2Reply(404, {"error": "checkout_not_found", "error_description": "No checkout with this id"})
        self.host.execute({"type": "read", "session": session, "target": "checkout", "id": ck.id})
        outcome = self._status(ck.id, base)
        digests = {m.kind: m.digest for m in self.host.state.mandates.values() if m.checkout_id == ck.id}
        reply: dict[str, Any] = {"checkout_id": ck.id, **outcome}
        if "checkout" in digests and "payment" in digests:
            reply.update(self._receipts(ck.id, digests["checkout"], digests["payment"], outcome))
        reply["checkout"] = self._checkout_payload(self.host.state.checkouts[ck.id])
        return Ap2Reply(200, reply)

    def _receipts(
        self, checkout_id: str, checkout_ref: str, payment_ref: str, outcome: Mapping[str, Any]
    ) -> dict[str, Any]:
        """The v0.2.0 checkout and payment receipts of a settled or failed checkout; none while a challenge waits."""
        status = outcome.get("status")
        iat = self.now_s
        psp = f"psp.{self.merchant}"
        if status == "success":
            payment_id = str(outcome.get("payment_id"))
            checkout_receipt = {
                "status": "Success",
                "iss": self.merchant,
                "iat": iat,
                "reference": checkout_ref,
                "order_id": str(outcome.get("order_id")),
            }
            payment_receipt = {
                "status": "Success",
                "iss": psp,
                "iat": iat,
                "reference": payment_ref,
                "payment_id": payment_id,
                "psp_confirmation_id": payment_id,
                "network_confirmation_id": stable_id("net", payment_id),
            }
            return {"checkout_receipt": checkout_receipt, "payment_receipt": payment_receipt}
        if status != "failed":
            return {}
        error, description = str(outcome.get("error")), str(outcome.get("error_description"))
        payment_id = str(self.host.state.checkouts[checkout_id].payment_id or "none")
        failure = {"error": error, "error_description": description}
        return {
            "checkout_receipt": {
                "status": "Error",
                "iss": self.merchant,
                "iat": iat,
                "reference": checkout_ref,
                **failure,
            },
            "payment_receipt": {
                "status": "Error",
                "iss": psp,
                "iat": iat,
                "reference": payment_ref,
                "payment_id": payment_id,
                **failure,
            },
        }

    def _user_jws(self, token: str, schema: str, name: str) -> dict[str, Any]:
        """A mandate signed by the shopper's key, checked against the release's schema."""
        if not token:
            raise Refused(400, "mandate_required", f"{name} is required, as a JWS the user signed", path=f"$.{name}")
        try:
            payload = self.shopper_key.verify(token)
        except BadSignature:
            try:
                read_jws(token)
            except BadSignature:
                raise Refused(400, "mandate_malformed", f"{name} is not a compact JWS", path=f"$.{name}") from None
            raise Refused(
                403, "authorization_invalid", f"{name} does not verify under the user's key", "policy", path=f"$.{name}"
            ) from None
        found = problems(registry().validator("ap2", self.version, f"{SCHEMAS}/{schema}"), payload)
        if found:
            path = found[0].path.replace("$", f"$.{name}", 1)
            raise Refused(400, found[0].code, f"{name}: {found[0].content}", path=path)
        return payload


def _member(body: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    raw = body.get(key)
    if not isinstance(raw, Mapping):
        raise Refused(400, "mandate_required", f"The body carries the mandate under {key!r}", path=f"$['{key}']")
    return raw


def _model[M: pydantic.BaseModel](cls: type[M], raw: Mapping[str, Any], key: str) -> M:
    """A v0.1.0 type validated, or `Refused` naming the first field by path and saying what it wants, not what it
    held."""
    try:
        return cls.model_validate(dict(raw))
    except pydantic.ValidationError as exc:
        first = exc.errors(include_input=False, include_url=False)[0]
        path = json_path(first["loc"])
        code = "field_required" if first["type"] == "missing" else f"invalid_{first['type']}"
        raise Refused(400, code, f"{key} {path}: {first['msg']}", path=path) from None


def _no_card_number(body: Mapping[str, Any]) -> None:
    if find_card_number(dict(body)) is not None:
        raise Refused(400, "card_number_refused", "A card number was sent; this merchant takes payment tokens only")


def address_lines(value: object) -> list[str]:
    """A ContactAddress's `address_line` as its list of lines. W3C types it a list, which is the only list-typed field
    of the address; a single string is one line, never its characters (a string iterated would ship to
    `["6", "3,  , S, a, ..."]`). Blank lines and anything not a string are left out."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [x.strip() for x in value if isinstance(x, str) and x.strip()]


def _group(body: Mapping[str, Any], pickup_ids: frozenset[str] = frozenset()) -> dict[str, Any] | None:
    """The core fulfillment group a basket's address and option make (a W3C ContactAddress to the core's address); an
    option naming one of the store's pickup locations makes a pickup group, which needs no address."""
    address = obj(body.get("shipping_address"))
    option = text(body.get("shipping_option_id")) or None
    if option is not None and option in pickup_ids:
        return {"id": "g1", "method": "pickup", "location_id": option, "selected_option_id": option}
    if not address and option is None:
        return None
    group: dict[str, Any] = {"id": "g1", "method": "shipping"}
    if address:
        country = text(address.get("country")).upper()
        if not (len(country) == 2 and country.isalpha()):
            raise Refused(
                400,
                "invalid_address",
                "shipping_address.country takes an ISO 3166-1 alpha-2 code",
                path="$.shipping_address.country",
            )
        lines = address_lines(address.get("address_line"))
        dest = {
            "first_name": text(address.get("recipient")),
            "street_address": lines[0] if lines else "",
            "extended_address": ", ".join(lines[1:]),
            "address_locality": text(address.get("city")),
            "address_region": text(address.get("region")),
            "postal_code": text(address.get("postal_code")),
            "address_country": country,
        }
        group["destination"] = {k: v for k, v in dest.items() if v}
    if option is not None:
        group["selected_option_id"] = option
    return group


def _contact(dest: Mapping[str, str]) -> dict[str, Any]:
    """A core address as a W3C ContactAddress."""
    lines = [x for x in (dest.get("street_address"), dest.get("extended_address")) if x]
    out: dict[str, Any] = {
        "recipient": dest.get("first_name"),
        "address_line": lines or None,
        "city": dest.get("address_locality"),
        "region": dest.get("address_region"),
        "postal_code": dest.get("postal_code"),
        "country": dest.get("address_country"),
    }
    return {k: v for k, v in out.items() if v}


def _acceptable(payload: Mapping[str, Any]) -> set[str]:
    """The item ids an open checkout mandate's line item constraints accept."""
    out: set[str] = set()
    for c in payload.get("constraints") or ():
        if obj(c).get("type") != "checkout.line_items":
            continue
        for req in obj(c).get("items") or ():
            out |= {text(obj(item).get("id")) for item in obj(req).get("acceptable_items") or ()}
    return out


def _minor(value: float, currency: str) -> int | None:
    """A major-unit amount in minor units, only when it carries no more precision than the currency has."""
    try:
        scaled = Decimal(str(value)) * (10 ** exponent(currency))
    except InvalidOperation:
        return None
    return int(scaled) if scaled == scaled.to_integral_value() else None


def _epoch_ms(value: str) -> int | None:
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return None if when.tzinfo is None else (when - EPOCH) // timedelta(milliseconds=1)

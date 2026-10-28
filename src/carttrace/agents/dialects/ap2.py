"""AP2 v0.2.0: an intent mandate, a merchant-signed checkout and the shopper's payment mandate (D66, the
`ap2_mandates` payment rail).

What differs from UCP, and why the agents still run the same loop:
- AP2 has no catalog: the store's UCP catalog is listed beside it, as for ACP. Nor a cart: the merchant's checkout is
  the cart, made from a basket with `create_checkout(items, ...)` and updated by naming its `checkout_id`.
- The shopper's intent comes first. `send_intent_mandate` sends an open checkout mandate the shopper's wallet signs:
  the line items the shopper wants (a `checkout.line_items` constraint), and the merchant answers with the intent's id,
  which the basket names (`intent_mandate_id`), so the merchant holds the checkout to what the shopper allowed.
- The checkout says what it still needs (`missing`: payer_email, shipping_address, shipping_option_id) until it is
  ready; then the merchant signs it (`checkout_jwt`, `checkout_hash`) and its totals are the total the shopper is
  asked to approve (`Ap2Entry`).
- Payment is `complete_checkout(checkout_id, token)`: the wallet's credential, issued against the shopper's approval of
  exactly the checkout's total (`MandateIssuer`), and the shopper's wallet signs the closed checkout mandate and the
  payment mandate over it (`WalletSigner`), refusing to sign a payment it did not issue a credential for. The merchant
  answers with a checkout receipt and a payment receipt and the order id; a 3-D Secure step answers 202
  `challenge_required` with the bank's page, after which `get_checkout_status` says how it ended.

Shapes are the sandbox's own REST binding of AP2's types (AP2 publishes no HTTP API): each reply names its
`checkout_id` and `status`, and carries the merchant's checkout (`checkout`: its line items as UCP items, totals and
currency) so a reader sees what it holds before it is signed.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from carttrace.agents.buyer import option_kind, stated_days
from carttrace.agents.checkout import total_of
from carttrace.agents.dialects.base import Call, Dialect
from carttrace.agents.entry import ASK_ORDER, Details, Option
from carttrace.agents.tools import payload_of
from carttrace.agents.wallet import (
    Consent,
    Credential,
    CredentialKind,
    CredentialRequest,
    MandateRequest,
    Persona,
    Refusal,
    Wallet,
)
from carttrace.core.errors import SafetyError
from carttrace.core.money import Money, exponent
from carttrace.protocols.tools import ToolResult
from carttrace.trace.schema import Cart, CartLine

INTENT: Final = "send_intent_mandate"
CHECKOUT: Final = "create_checkout"
COMPLETE: Final = "complete_checkout"
STATUS: Final = "get_checkout_status"
TOOLS: Final = frozenset({CHECKOUT, COMPLETE, STATUS})
NAMESPACE: Final = "ap2.card"
"""What the wallet calls the card instrument a payment mandate names (`payment_instrument.type` card)."""
STATUSES: Final[Mapping[str, str]] = {
    "incomplete": "incomplete",
    "success": "completed",
    "challenge_required": "challenge_required",
    "failed": "failed",
    "pending": "pending",
}
"""The merchant's reply status to the checkout status the agents read; a signed checkout is `ready`."""
MISSING: Final[Mapping[str, str]] = {
    "payer_email": "email",
    "shipping_address": "shipping_address",
    "shipping_option_id": "fulfillment",
}
GROUP_ID: Final = "g1"


@dataclass(frozen=True, slots=True)
class Ap2Dialect(Dialect):
    """AP2 v0.2.0's merchant checkout and mandates, with the store's catalog read through UCP's catalog tools."""

    create: str = CHECKOUT
    read: str = STATUS
    create_checkout: str = CHECKOUT
    update_checkout: str = CHECKOUT
    get_checkout: str = STATUS
    complete_checkout: str = COMPLETE
    get_order: str | None = None
    checkout_tools: frozenset[str] = field(default=TOOLS)
    order_tools: frozenset[str] = frozenset()
    payment_key: str = "token"
    intent: str | None = INTENT

    @property
    def name(self) -> str:
        return "ap2"

    def offered(self, tools: Collection[str]) -> bool:
        return {INTENT, CHECKOUT} <= set(tools)

    # -- the cart: the merchant's checkout -----------------------------------------------------------------------------

    def lines_arguments(self, lines: Sequence[tuple[str, int]], *, currency: str | None = None) -> dict[str, Any]:
        del currency
        return {"items": [{"sku": vid, "quantity": qty} for vid, qty in lines]}

    def with_intent(self, calls: list[Call], intent_id: str) -> list[Call]:
        return [(name, {**arguments, "intent_mandate_id": intent_id}) for name, arguments in calls]

    def intent_arguments(self, items: Sequence[tuple[str, str, int]]) -> dict[str, Any]:
        """An open checkout mandate's constraints: one line item requirement per item the shopper wants."""
        requirements = [
            {"id": f"line-{i}", "acceptable_items": [{"id": vid, "title": title or vid}], "quantity": qty}
            for i, (vid, title, qty) in enumerate(items, start=1)
        ]
        return {"constraints": [{"type": "checkout.line_items", "items": requirements}]}

    def intent_id(self, result: ToolResult) -> str | None:
        data = payload_of(result) if not result.is_error else None
        found = data.get("intent_mandate_id") if isinstance(data, Mapping) else None
        return found if isinstance(found, str) and found else None

    def read_arguments(self, cart_id: str) -> dict[str, Any]:
        return {"checkout_id": cart_id}

    def id_arguments(self, ident: str) -> dict[str, Any]:
        return {"checkout_id": ident}

    def is_cart_tool(self, name: str) -> bool:
        return name in (CHECKOUT, STATUS)

    def cart_in(self, result: ToolResult) -> Cart | None:
        if result.is_error:
            return None
        doc = self.checkout_doc(payload_of(result))
        return None if doc is None else self.view_of(doc)

    # -- documents -----------------------------------------------------------------------------------------------------

    def checkout_doc(self, payload: Any) -> Mapping[str, Any] | None:
        """A merchant reply read as one checkout document: the merchant's checkout (lines, totals, currency) under the
        reply's id and status, with what the reply adds (missing, options, the signed checkout, the order, the
        bank's challenge, a failure)."""
        if not isinstance(payload, Mapping) or not isinstance(payload.get("checkout_id"), str):
            return None
        inner = payload.get("checkout")
        doc: dict[str, Any] = dict(inner) if isinstance(inner, Mapping) else {}
        doc.setdefault("line_items", [])
        doc["id"] = payload["checkout_id"]
        raw = payload.get("status")
        doc["status"] = STATUSES.get(str(raw), str(raw)) if raw is not None else "ready"
        for key in ("missing", "shipping_options", "checkout_jwt", "checkout_hash", "challenge", "error"):
            if key in payload:
                doc[key] = payload[key]
        if isinstance(payload.get("error_description"), str):
            doc["error_description"] = payload["error_description"]
        order = payload.get("order_id")
        if doc["status"] == "completed" and isinstance(order, str) and order:
            doc["order"] = {"id": order}
        for key in ("checkout_receipt", "payment_receipt"):
            if isinstance(payload.get(key), Mapping):
                doc[key] = dict(payload[key])
        return doc

    def order_doc(self, payload: Any) -> Mapping[str, Any] | None:
        return None

    def view_of(self, doc: Mapping[str, Any]) -> Cart | None:
        return _cart(doc, cart_id=str(doc.get("id") or "") or None)

    def order_view(self, doc: Mapping[str, Any]) -> Cart | None:
        order_id = self.order_of(doc)
        return None if order_id is None else _cart(doc, cart_id=order_id)

    def order_of(self, doc: Mapping[str, Any]) -> str | None:
        order = doc.get("order")
        found = order.get("id") if isinstance(order, Mapping) else None
        return found if doc.get("status") == "completed" and isinstance(found, str) and found else None

    def order_label(self, doc: Mapping[str, Any]) -> str | None:
        return self.order_of(doc)

    def challenge_of(self, doc: Mapping[str, Any]) -> str | None:
        challenge = doc.get("challenge")
        url = challenge.get("url") if isinstance(challenge, Mapping) else None
        return url if doc.get("status") == "challenge_required" and isinstance(url, str) and url else None

    def payment_failure(self, doc: Mapping[str, Any]) -> str | None:
        if doc.get("status") != "failed":
            return None
        said = doc.get("error_description") or doc.get("error")
        return str(said) if said else "the payment failed"

    def handlers_of(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        """The one way a payment mandate pays on this rail: a card credential, named by the mandate."""
        del doc
        return {NAMESPACE: [{"id": "card", "version": "v0.2.0"}]}

    # -- opening a checkout with saved details ------------------------------------------------------------------------

    def open_checkout_calls(self, cart_id: str, details: Mapping[str, Any]) -> list[Call]:
        arguments: dict[str, Any] = {"checkout_id": cart_id}
        if isinstance(details.get("email"), str):
            arguments["payer_email"] = details["email"]
        address = details.get("address")
        if isinstance(address, Mapping):
            arguments["shipping_address"] = _contact_address(address)
        return [(CHECKOUT, arguments)]

    # -- paying --------------------------------------------------------------------------------------------------------

    def complete_target(self, arguments: Mapping[str, Any]) -> str | None:
        target = arguments.get("checkout_id")
        return target if isinstance(target, str) and target else None

    def tokens_of(self, arguments: Mapping[str, Any]) -> set[str]:
        token = arguments.get("token")
        return {token} if isinstance(token, str) and token else set()

    def credential_request(
        self,
        checkout_id: str,
        handler_id: str,
        namespace: str,
        amount: Money,
        doc: Mapping[str, Any],
        *,
        now_ms: int,
    ) -> CredentialRequest:
        del handler_id, namespace, now_ms
        contents = {k: doc[k] for k in ("checkout_hash",) if k in doc}
        return MandateRequest(checkout_id, "payment", contents, amount)

    def credential_answer(self, credential: Credential, *, token: str | None = None) -> dict[str, Any]:
        return {
            self.payment_key: token or credential.token,
            "amount": {"amount": credential.amount.amount, "currency": credential.amount.currency},
            "use": f"call {COMPLETE} with this checkout_id and this token: the shopper's wallet signs the checkout "
            "mandate and the payment mandate for this total",
        }

    def complete_arguments(self, checkout_id: str, answer: Mapping[str, Any]) -> dict[str, Any]:
        return {"checkout_id": checkout_id, "token": answer[self.payment_key]}

    def refresh_calls(self, checkout_id: str) -> list[Call]:
        """The merchant signs the checkout again at its current total when asked to update it with nothing."""
        return [(CHECKOUT, {"checkout_id": checkout_id})]


def _cart(doc: Mapping[str, Any], *, cart_id: str | None) -> Cart | None:
    """The merchant checkout's lines and totals as the trace's `Cart`, or None when a line cannot be read."""
    currency = doc.get("currency")
    code = currency.upper() if isinstance(currency, str) and currency else None
    lines: list[CartLine] = []
    for raw in doc.get("line_items") or ():
        item = raw.get("item") if isinstance(raw, Mapping) else None
        if not isinstance(item, Mapping):
            return None
        variant, price, quantity = item.get("id"), item.get("price"), raw.get("quantity")
        if not isinstance(variant, str) or not _whole(price) or not _whole(quantity) or code is None:
            return None
        title = item.get("title") if isinstance(item.get("title"), str) else ""
        lines.append(CartLine(variant_id=variant, title=str(title), quantity=quantity, unit_price=price, currency=code))
    totals = {str(t.get("type")): t.get("amount") for t in doc.get("totals") or () if isinstance(t, Mapping)}
    subtotal, total = totals.get("subtotal"), totals.get("total")
    return Cart(
        cart_id=cart_id,
        lines=lines,
        subtotal=subtotal if _whole(subtotal) else None,
        total=total if _whole(total) else None,
        currency=code,
    )


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _contact_address(address: Mapping[str, Any]) -> dict[str, Any]:
    """A UCP postal address as a W3C ContactAddress, the shape AP2's basket takes; no recipient name is invented."""
    lines = [str(address[k]) for k in ("street_address", "extended_address") if address.get(k)]
    out: dict[str, Any] = {
        "address_line": lines,
        "city": str(address.get("address_locality") or ""),
        "postal_code": str(address.get("postal_code") or ""),
        "country": str(address.get("address_country") or ""),
    }
    if address.get("address_region"):
        out["region"] = str(address["address_region"])
    return out


def _minor(value: Any, currency: str | None) -> int | None:
    """A major-unit amount (AP2's W3C PaymentCurrencyAmount value) in minor units."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        scaled = Decimal(str(value)) * (10 ** exponent(currency))
    except InvalidOperation:
        return None
    return int(scaled) if scaled == scaled.to_integral_value() else None


# -- the buyer's details through the merchant's checkout ------


class Ap2Entry:
    """AP2: the merchant's checkout is the cart, so `create` reads it by updating it with nothing; details go in with
    `create_checkout(checkout_id, payer_email, shipping_address)`, the option with `shipping_option_id`."""

    def __init__(self, dialect: Ap2Dialect | None = None) -> None:
        self.dialect = dialect or Ap2Dialect()

    @property
    def name(self) -> str:
        return "ap2"

    def offered(self, tools: Collection[str]) -> bool:
        return {INTENT, CHECKOUT, COMPLETE} <= set(tools)

    def create(self, cart_id: str) -> Call:
        return CHECKOUT, {"checkout_id": cart_id}

    def missing(self, doc: Mapping[str, Any]) -> list[str]:
        named = {MISSING[str(m)] for m in doc.get("missing") or () if str(m) in MISSING}
        return [f for f in ASK_ORDER if f in named]

    def enter(self, doc: Mapping[str, Any], details: Details) -> list[Call]:
        arguments: dict[str, Any] = {"checkout_id": str(doc["id"])}
        if details.email is not None:
            arguments["payer_email"] = details.email
        if details.address is not None and details.method != "pickup":
            address = details.address
            contact: dict[str, Any] = {
                "address_line": list(address.lines),
                "city": address.city,
                "postal_code": address.postal_code,
                "country": address.country,
            }
            if address.region:
                contact["region"] = address.region
            arguments["shipping_address"] = contact
        return [(CHECKOUT, arguments)] if len(arguments) > 1 else []

    def options(self, doc: Mapping[str, Any]) -> list[Option]:
        found: list[Option] = []
        currency = doc.get("currency") if isinstance(doc.get("currency"), str) else None
        for option in doc.get("shipping_options") or ():
            if not isinstance(option, Mapping) or not isinstance(option.get("id"), str):
                continue
            amount = option.get("amount")
            value = amount.get("value") if isinstance(amount, Mapping) else None
            code = amount.get("currency") if isinstance(amount, Mapping) else currency
            minor = _minor(value, code if isinstance(code, str) else currency)
            if minor is None:
                continue
            title = option["label"] if isinstance(option.get("label"), str) else option["id"]
            kind = option_kind(title, option["id"], "shipping")
            method = "pickup" if kind == "pickup" else "shipping"
            found.append(
                Option(
                    method_id=method,
                    method=method,
                    group_id=GROUP_ID,
                    option_id=option["id"],
                    title=title,
                    amount=minor,
                    kind=kind,
                    days=stated_days(title),
                )
            )
        return sorted(found, key=lambda o: o.amount)

    def select(self, doc: Mapping[str, Any], option: Option) -> Call:
        return CHECKOUT, {"checkout_id": str(doc["id"]), "shipping_option_id": option.option_id}

    def selected(self, doc: Mapping[str, Any]) -> Option | None:
        """The merchant's reply does not say which option is selected; a signed checkout has one."""
        del doc
        return None

    def ready(self, doc: Mapping[str, Any]) -> bool:
        return isinstance(doc.get("checkout_jwt"), str) and doc.get("status") == "ready"

    def total(self, doc: Mapping[str, Any]) -> Money | None:
        return total_of(doc)


# -- the wallet's side ------


class MandateIssuer:
    """AP2 credentials: the persona's card token, issued for exactly the approved total of one merchant-signed checkout,
    for the payment mandate to name. The wallet's consent gate runs first."""

    @property
    def kind(self) -> CredentialKind:
        return "ap2_mandate"

    def refusal(self, persona: Persona, request: CredentialRequest) -> Refusal | None:
        if not isinstance(request, MandateRequest):
            return Refusal("credential_kind_mismatch", f"a {request.kind} request reached the AP2 mandate issuer")
        if _card(persona) is None:
            kinds = ", ".join(i.kind for i in persona.instruments)
            return Refusal(
                "handler_unsupported",
                f"the shopper's wallet holds no card for an AP2 payment mandate (it holds: {kinds})",
                hint="this persona cannot pay here",
            )
        return None

    def issue(self, persona: Persona, request: CredentialRequest, consent: Consent, *, sequence: int) -> Credential:
        assert isinstance(request, MandateRequest)
        card = _card(persona)
        assert card is not None
        return Credential(
            kind=self.kind,
            checkout_id=request.checkout_id,
            amount=request.amount,
            token=card.token,
            instrument=card.kind,
            instrument_id=f"wallet-instrument-{sequence}",
            consent=consent.seq,
            handler_id="card",
            brand=card.brand,
        )


def _card(persona: Persona) -> Any:
    """The persona's card, else its Google Pay token: the sandbox's PSP takes any opaque test credential through the
    payment-request handler both MPP and AP2 settle through, so a wallet that holds only Google Pay still pays."""
    return next((i for i in persona.instruments if i.kind == "card"), None) or next(
        (i for i in persona.instruments if i.kind == "google_pay"), None
    )


class WalletSigner:
    """The shopper's wallet signing AP2 mandates for the user, gated on the shopper's consent: an open checkout mandate
    (the intent: what the shopper asked for) is signed as asked; a closed checkout mandate and a payment mandate only
    once the wallet issued an AP2 credential, and a payment mandate only for the token and the amount it issued, which
    the shopper approved. The key is the shopper's test key in the pack's world (`ShopperWallet`). `wallet` is set
    when the episode's wallet is made, after the surface opens."""

    def __init__(self, pack_id: str, wallet: Wallet | None = None) -> None:
        from carttrace.sandbox.surfaces.ap2 import ShopperWallet

        self.key = ShopperWallet(pack_id)
        self.wallet = wallet

    @property
    def kid(self) -> str:
        return self.key.kid

    def sign(self, payload: Mapping[str, Any], *, typ: str = "JWT") -> str:
        kind = payload.get("vct")
        issued_any = any(c.kind == "ap2_mandate" for c in (self.wallet.issued if self.wallet is not None else ()))
        if kind is None and "transaction_data" in payload and not issued_any:
            # AP2 v0.1.0's user authorization binds the cart's and the payment's hashes (D71)
            raise SafetyError(
                "the shopper's wallet signs a payment authorization only for a credential it issued against the "
                "shopper's approval of that amount",
                hint="have the shopper sign the mandate on their device, then get the credential",
            )
        if kind in ("mandate.checkout.1", "mandate.payment.1"):
            issued = [c for c in (self.wallet.issued if self.wallet is not None else ()) if c.kind == "ap2_mandate"]
            if kind == "mandate.payment.1":
                instrument = payload.get("payment_instrument")
                amount = payload.get("payment_amount")
                token = instrument.get("id") if isinstance(instrument, Mapping) else None
                minor = amount.get("amount") if isinstance(amount, Mapping) else None
                issued = [c for c in issued if c.token == token and c.amount.amount == minor]
            if not issued:
                raise SafetyError(
                    "the shopper's wallet signs a checkout or payment mandate only for a credential it issued against "
                    "the shopper's approval of that amount",
                    hint="have the shopper sign the mandate on their device, then get the credential",
                )
        return self.key.sign(payload, typ=typ)


# -- AP2 v0.1.0 (D71) ------


INTENT_V1: Final = "send_intent_mandate"
CART_V1: Final = "request_cart_mandate"
PAY_V1: Final = "send_payment_mandate"
STATUS_V1: Final = "get_payment_status"
TOOLS_V1: Final = frozenset({CART_V1, PAY_V1, STATUS_V1})
CART_MANDATE_V1: Final = "ap2.mandates.CartMandate"
"""The member a v0.1.0 reply carries the merchant-signed cart under (its A2A DataPart key)."""


@dataclass(frozen=True, slots=True)
class Ap2V1Dialect(Ap2Dialect):
    """AP2 v0.1.0: an unsigned intent mandate (the shopper's words and the skus, requiring the shopper's confirmation
    of the cart), a cart the merchant signs (`cart_mandate`, a W3C PaymentRequest for the total), and a payment mandate
    whose user authorization the shopper's wallet signs over the cart's and the payment's hashes. The sandbox's REST
    binding reads a cart by its id (`get_payment_status(cart_id)`), and every reply carries the merchant's checkout
    behind the cart, as at v0.2.0."""

    create: str = CART_V1
    read: str = STATUS_V1
    create_checkout: str = CART_V1
    update_checkout: str = CART_V1
    get_checkout: str = STATUS_V1
    complete_checkout: str = PAY_V1
    checkout_tools: frozenset[str] = field(default=TOOLS_V1)
    intent: str | None = INTENT_V1

    @property
    def name(self) -> str:
        return "ap2_v1"

    def offered(self, tools: Collection[str]) -> bool:
        return {INTENT_V1, CART_V1} <= set(tools)

    def intent_arguments(self, items: Sequence[tuple[str, str, int]]) -> dict[str, Any]:
        """A v0.1.0 IntentMandate: what the shopper asked for in words, the skus it allows, and the shopper's
        confirmation of the cart required (an unsigned intent must require it). The surface dates it."""
        words = "; ".join(f"{qty} x {title or vid}" for vid, title, qty in items)
        mandate = {
            "natural_language_description": words,
            "user_cart_confirmation_required": True,
            "skus": [vid for vid, _, _ in items],
            "requires_refundability": False,
        }
        return {"intent_mandate": mandate}

    def read_arguments(self, cart_id: str) -> dict[str, Any]:
        return {"cart_id": cart_id}

    def id_arguments(self, ident: str) -> dict[str, Any]:
        return {"cart_id": ident}

    def is_cart_tool(self, name: str) -> bool:
        return name in (CART_V1, STATUS_V1)

    def checkout_doc(self, payload: Any) -> Mapping[str, Any] | None:
        """A v0.1.0 reply read as one checkout document under the cart's id; a signed cart (the merchant's
        `CartMandate`, kept as `cart_mandate`) is ready."""
        if not isinstance(payload, Mapping) or not isinstance(payload.get("cart_id"), str):
            return None
        doc = dict(Ap2Dialect.checkout_doc(self, {**payload, "checkout_id": payload["cart_id"]}) or {})
        signed = payload.get(CART_MANDATE_V1)
        if isinstance(signed, Mapping):
            doc["cart_mandate"] = dict(signed)
        return doc

    def handlers_of(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        del doc
        return {NAMESPACE: [{"id": "card", "version": "v0.1.0"}]}

    def open_checkout_calls(self, cart_id: str, details: Mapping[str, Any]) -> list[Call]:
        return [_as_cart(call) for call in Ap2Dialect.open_checkout_calls(self, cart_id, details)]

    def complete_target(self, arguments: Mapping[str, Any]) -> str | None:
        target = arguments.get("cart_id")
        return target if isinstance(target, str) and target else None

    def credential_answer(self, credential: Credential, *, token: str | None = None) -> dict[str, Any]:
        out = Ap2Dialect.credential_answer(self, credential, token=token)
        out["use"] = (
            f"call {PAY_V1} with this cart_id and this token: the shopper's wallet signs the payment mandate for "
            "this total"
        )
        return out

    def complete_arguments(self, checkout_id: str, answer: Mapping[str, Any]) -> dict[str, Any]:
        return {"cart_id": checkout_id, "token": answer[self.payment_key]}

    def refresh_calls(self, checkout_id: str) -> list[Call]:
        """The merchant signs the cart again at its current total when asked to update it with nothing."""
        return [(CART_V1, {"cart_id": checkout_id})]


class Ap2V1Entry(Ap2Entry):
    """AP2 v0.1.0: the cart is read by updating it with nothing; details go in with `request_cart_mandate(cart_id,
    payer_email, shipping_address)`, the option with `shipping_option_id`; a signed cart is ready."""

    def __init__(self, dialect: Ap2V1Dialect | None = None) -> None:
        super().__init__(dialect or Ap2V1Dialect())

    @property
    def name(self) -> str:
        return "ap2_v1"

    def offered(self, tools: Collection[str]) -> bool:
        return {INTENT_V1, CART_V1, PAY_V1} <= set(tools)

    def create(self, cart_id: str) -> Call:
        return CART_V1, {"cart_id": cart_id}

    def enter(self, doc: Mapping[str, Any], details: Details) -> list[Call]:
        return [_as_cart(call) for call in super().enter(doc, details)]

    def select(self, doc: Mapping[str, Any], option: Option) -> Call:
        return _as_cart(super().select(doc, option))

    def ready(self, doc: Mapping[str, Any]) -> bool:
        return isinstance(doc.get("cart_mandate"), Mapping) and doc.get("status") == "ready"

    def options(self, doc: Mapping[str, Any]) -> list[Option]:
        """The delivery options, from the reply's own list when it has one, else from the signed cart mandate's
        payment request (`contents.payment_request.details.shipping_options`, W3C PaymentShippingOption), which is
        where v0.1.0 lists them once the cart is signed. Read only from the top level, a v0.1.0 cart showed no
        option, and the reference agent gave up "no express option that arrives within 3 days" on every AP2 v0.1.0
        row with a deadline (10 judge cards on 2026-10-17, ep-9bd1568d89c6), while an express row with no deadline
        was ordered at the merchant's default, standard."""
        found = super().options(doc)
        if found or "shipping_options" in doc:
            return found
        listed = mandate_shipping_options(doc.get("cart_mandate"))
        return super().options({**doc, "shipping_options": listed}) if listed else []


def mandate_shipping_options(mandate: Any) -> list[Mapping[str, Any]]:
    """The shipping options a v0.1.0 CartMandate's payment request lists; empty when it lists none."""
    contents = mandate.get("contents") if isinstance(mandate, Mapping) else None
    request = contents.get("payment_request") if isinstance(contents, Mapping) else None
    details = request.get("details") if isinstance(request, Mapping) else None
    options = details.get("shipping_options") if isinstance(details, Mapping) else None
    return [o for o in options or () if isinstance(o, Mapping)]


def _as_cart(call: Call) -> Call:
    """A v0.2.0 checkout call as v0.1.0's cart call: the same basket, naming the cart."""
    _, arguments = call
    out = {k: v for k, v in arguments.items() if k != "checkout_id"}
    return CART_V1, {"cart_id": arguments["checkout_id"], **out}

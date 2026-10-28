"""The AP2 surface as an agent holds it: a `ToolSurface` (kind `ap2`) on a sandbox store's AP2 merchant, and the
shopper's test wallet that signs for the user.

Who signs. AP2's mandates carry the user's signature, made on the user's device after the user consents; an agent
cannot make one and must not hold the key. So the surface takes a `signer` (any object with `kid` and
`sign(payload) -> JWS`): the agent side's wallet, which asks for consent before it signs. `ShopperWallet(pack_id)` is
the test wallet holding the shopper's derived key (`ap2.keys`); a harness wraps it with its consent gate. Without a
signer, every tool that needs the user's signature refuses with `signer_required`, and nothing is sent.

The tools, at v0.1.0:
- `send_intent_mandate(intent_mandate)`: an `IntentMandate` as v0.1.0 types it; returns its id and matching items;
- `request_cart_mandate(items, shipping_address?, shipping_option_id?, payer_email?, cart_id?, intent_mandate_id?)`:
  the merchant-signed `CartMandate` (the surface keeps it for the payment), or what the cart still needs;
- `send_payment_mandate(cart_id, token, method_name?)`: the surface builds the `PaymentMandateContents` for the kept
  cart, the wallet signs the user's authorization over the cart's and the contents' hashes, and the merchant settles;
- `get_payment_status(payment_mandate_id)`.

At v0.2.0:
- `send_intent_mandate(constraints)`: an open checkout mandate the wallet signs;
- `create_checkout(items, ..., checkout_id?, intent_mandate_id?)`: the merchant-signed `checkout_jwt`;
- `complete_checkout(checkout_id, token, instrument_type?)`: the closed checkout mandate and payment mandate the
  wallet signs, and the receipts;
- `get_checkout_status(checkout_id)`.

A 202 `challenge_required` is a normal result (the buyer completes the 3-D Secure page, then the status tools show the
outcome); an HTTP error is an error result whose `error_code` is the merchant's `error`.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any, ClassVar, Protocol
from urllib.parse import quote

import httpx

from carttrace.core.canon import dumps
from carttrace.core.clock import Clock, iso
from carttrace.core.ids import stable_id
from carttrace.protocols.tools import SurfaceKind, ToolResult, ToolSpec
from carttrace.sandbox.presented import presented
from carttrace.sandbox.surfaces.ap2 import models
from carttrace.sandbox.surfaces.ap2.keys import PartyKey, mandate_hash, party_key, read_jws
from carttrace.sandbox.surfaces.ap2.merchant import check_version
from carttrace.sandbox.surfaces.ap2.rest import AP2_PATH
from carttrace.sandbox.surfaces.client import JSON, SandboxSurface, WireSink
from carttrace.sandbox.surfaces.host import SandboxHost

INTENT_TTL = timedelta(days=1)
"""How long an undated v0.1.0 intent lasts, as the user's device dates it: a day, so a pack whose logical clock
starts at its capture is not refused for an expired intent."""


class MandateSigner(Protocol):
    """What signs for the user: the wallet, after the user consents."""

    @property
    def kid(self) -> str: ...

    def sign(self, payload: Mapping[str, Any], *, typ: str = "JWT") -> str: ...


class ShopperWallet:
    """The shopper's test wallet in one pack's world: it signs whatever it is asked to. Wrap it with consent."""

    def __init__(self, pack_id: str) -> None:
        self.key: PartyKey = party_key(pack_id, "shopper")

    @property
    def kid(self) -> str:
        return self.key.kid

    def sign(self, payload: Mapping[str, Any], *, typ: str = "JWT") -> str:
        return self.key.sign(payload, typ=typ)

    def jwk(self) -> dict[str, str]:
        return self.key.jwk()


ADDRESS = {
    "type": "object",
    "description": "A W3C ContactAddress: recipient, address_line, city, region, postal_code, country (two letters)",
}
ITEMS = {
    "type": "array",
    "description": "The basket: product variant ids (sku) and whole quantities",
    "items": {
        "type": "object",
        "properties": {"sku": {"type": "string"}, "quantity": {"type": "integer", "minimum": 1}},
        "required": ["sku"],
    },
}
BASKET = {
    "items": ITEMS,
    "shipping_address": ADDRESS,
    "shipping_option_id": {
        "type": "string",
        "description": "An id from the shipping options the merchant listed; a pickup location's id (pickup-1) is "
        "collected in store and needs no shipping_address",
    },
    "payer_email": {"type": "string"},
    "intent_mandate_id": {"type": "string", "description": "The intent this basket answers, from send_intent_mandate"},
}
TOKEN = {"type": "string", "description": "The payment credential token the shopper's wallet issued"}
CONSTRAINTS = {
    "type": "array",
    "description": "The open checkout mandate's constraints (AP2 v0.2.0 open_checkout_mandate.json): at least one "
    "checkout.line_items",
    "items": {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "type": {"const": "checkout.line_items"},
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "A name for this requirement, e.g. line-1"},
                                "acceptable_items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {"id": {"type": "string"}, "title": {"type": "string"}},
                                        "required": ["id", "title"],
                                    },
                                },
                                "quantity": {"type": "integer", "minimum": 1},
                            },
                            "required": ["id", "acceptable_items", "quantity"],
                        },
                    },
                },
                "required": ["type", "items"],
            },
            {
                "type": "object",
                "properties": {
                    "type": {"const": "checkout.allowed_merchants"},
                    "allowed": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
                        },
                    },
                },
                "required": ["type", "allowed"],
            },
        ]
    },
}
"""The constraint shapes AP2 v0.2.0's open checkout mandate takes, from its vendored schema, so an agent can write one
(D66: with only `{"type": "object"}` the agent under test sent 40 shapes and none matched)."""


def _tools(version: str) -> tuple[ToolSpec, ...]:
    if version == "v0.1.0":
        return (
            ToolSpec(
                name="send_intent_mandate",
                description="Tell the merchant what the user wants, as an AP2 v0.1.0 IntentMandate (the user confirmed "
                "the description). Returns the intent's id and the merchant's matching items.",
                input_schema={
                    "type": "object",
                    "properties": {"intent_mandate": models.IntentMandate.model_json_schema()},
                    "required": ["intent_mandate"],
                },
            ),
            ToolSpec(
                name="request_cart_mandate",
                description="Ask the merchant to sign a cart for a basket. Returns the CartMandate (items, total, "
                "expiry, merchant signature), or what the cart still needs (an email, an address, an option).",
                input_schema={
                    "type": "object",
                    "properties": {**BASKET, "cart_id": {"type": "string", "description": "A cart to update"}},
                },
            ),
            ToolSpec(
                name="send_payment_mandate",
                description="Pay for a signed cart: the user's wallet signs a PaymentMandate for exactly the cart's "
                "total with the given credential, and the merchant settles it.",
                input_schema={
                    "type": "object",
                    "properties": {"cart_id": {"type": "string"}, "token": TOKEN, "method_name": {"type": "string"}},
                    "required": ["cart_id", "token"],
                },
            ),
            ToolSpec(
                name="get_payment_status",
                description="Where a cart's payment stands: success with the order, a challenge the buyer must "
                "complete, a failure, or a cart not yet paid, with the cart's items and totals. Name the cart "
                "(cart_id) or the payment (payment_mandate_id).",
                input_schema={
                    "type": "object",
                    "properties": {"cart_id": {"type": "string"}, "payment_mandate_id": {"type": "string"}},
                },
            ),
        )
    return (
        ToolSpec(
            name="send_intent_mandate",
            description="Tell the merchant what the user allows, as an AP2 v0.2.0 open checkout mandate the user's "
            "wallet signs. It must hold one checkout.line_items constraint, one requirement per item: "
            '{"type": "checkout.line_items", "items": [{"id": "line-1", "acceptable_items": [{"id": "<variant id>", '
            '"title": "<title>"}], "quantity": 1}]}; a checkout.allowed_merchants constraint may name the store. '
            "Returns the intent's id, which create_checkout takes as intent_mandate_id.",
            input_schema={"type": "object", "properties": {"constraints": CONSTRAINTS}, "required": ["constraints"]},
        ),
        ToolSpec(
            name="create_checkout",
            description="Ask the merchant to sign a checkout for a basket. Returns the checkout, its merchant-signed "
            "checkout_jwt and checkout_hash, or what the checkout still needs.",
            input_schema={
                "type": "object",
                "properties": {**BASKET, "checkout_id": {"type": "string", "description": "A checkout to update"}},
            },
        ),
        ToolSpec(
            name="complete_checkout",
            description="Complete a signed checkout: the user's wallet signs the checkout mandate and a payment "
            "mandate for exactly its total with the given credential. Returns the checkout and payment receipts.",
            input_schema={
                "type": "object",
                "properties": {
                    "checkout_id": {"type": "string"},
                    "token": TOKEN,
                    "instrument_type": {"type": "string"},
                },
                "required": ["checkout_id", "token"],
            },
        ),
        ToolSpec(
            name="get_checkout_status",
            description="Where a checkout's payment stands, with the receipts once settled.",
            input_schema={
                "type": "object",
                "properties": {"checkout_id": {"type": "string"}},
                "required": ["checkout_id"],
            },
        ),
    )


class Ap2Surface(SandboxSurface):
    """AP2 as the shopping agent meets the merchant, at the store's AP2 release."""

    kind: SurfaceKind = "ap2"
    protocol: ClassVar[str] = "ap2"

    def __init__(
        self,
        client: httpx.AsyncClient,
        host: SandboxHost,
        *,
        recorder: WireSink | None = None,
        session_id: str | None = None,
        clock: Clock | None = None,
        signer: MandateSigner | None = None,
    ) -> None:
        super().__init__(client, host, recorder=recorder, session_id=session_id, clock=clock)
        self.ap2 = check_version(host.require("ap2"))
        self.signer = signer
        self.merchant = presented(host.pack).merchant_id
        self._intents: dict[str, Any] = {}
        """Intent mandate id to the mandate as sent (an object at v0.1.0, a JWS at v0.2.0)."""
        self._carts: dict[str, dict[str, Any]] = {}
        """Cart or checkout id to what the merchant signed for it."""

    async def list_tools(self) -> list[ToolSpec]:
        return [t.model_copy(deep=True) for t in _tools(self.ap2)]

    async def call(self, name: str, arguments: dict[str, Any], *, idempotency_key: str | None = None) -> ToolResult:
        self._calls += 1
        handlers = {
            "send_intent_mandate": self._intent,
            "request_cart_mandate": self._cart,
            "create_checkout": self._cart,
            "send_payment_mandate": self._pay_v1,
            "complete_checkout": self._pay_v2,
            "get_payment_status": self._status,
            "get_checkout_status": self._status,
        }
        names = {t.name for t in _tools(self.ap2)}
        handler = handlers.get(name) if name in names else None
        if handler is None:
            return ToolResult(content=f"unknown tool {name!r}", is_error=True, error_code="protocol_error")
        return await handler(arguments)

    # -- tools ---------------------------------------------------------------------------------------------------------

    async def _intent(self, arguments: Mapping[str, Any]) -> ToolResult:
        if self.ap2 == "v0.1.0":
            mandate = arguments.get("intent_mandate")
            if not isinstance(mandate, Mapping):
                return _local("send_intent_mandate needs an intent_mandate object")
            sent: Any = dict(mandate)
            if not sent.get("intent_expiry"):  # the user's device dates the intent when the agent leaves it undated
                sent["intent_expiry"] = iso(self.clock.now() + INTENT_TTL)
            result = await self._post("/intent_mandates", {models.INTENT_MANDATE_DATA_KEY: sent})
        else:
            if self.signer is None:
                return _no_signer()
            payload = {
                "vct": "mandate.checkout.open.1",
                "constraints": list(arguments.get("constraints") or []),
                "cnf": {"kid": self.signer.kid},
                "iat": int(self.clock.now().timestamp()),
            }
            sent = self.signer.sign(payload, typ="vc+jwt")
            result = await self._post("/intent_mandates", {"open_checkout_mandate": sent})
        intent_id = (result.structured or {}).get("intent_mandate_id") if not result.is_error else None
        if isinstance(intent_id, str):
            self._intents[intent_id] = sent
        return result

    async def _cart(self, arguments: Mapping[str, Any]) -> ToolResult:
        body = {k: v for k, v in arguments.items() if k in BASKET or k in ("cart_id", "checkout_id")}
        intent_id = body.pop("intent_mandate_id", None)
        if isinstance(intent_id, str):
            sent = self._intents.get(intent_id)
            if sent is None:
                return _local("no intent mandate with this id was sent from this surface")
            key = models.INTENT_MANDATE_DATA_KEY if self.ap2 == "v0.1.0" else "open_checkout_mandate"
            body[key] = sent
        path = "/cart_mandates" if self.ap2 == "v0.1.0" else "/checkouts"
        result = await self._post(path, body)
        doc = result.structured or {}
        if not result.is_error and self.ap2 == "v0.1.0" and isinstance(doc.get(models.CART_MANDATE_DATA_KEY), Mapping):
            self._carts[str(doc.get("cart_id"))] = dict(doc[models.CART_MANDATE_DATA_KEY])
        elif not result.is_error and isinstance(doc.get("checkout_jwt"), str):
            self._carts[str(doc.get("checkout_id"))] = dict(doc)
        return result

    async def _pay_v1(self, arguments: Mapping[str, Any]) -> ToolResult:
        cart = self._carts.get(str(arguments.get("cart_id")))
        token = arguments.get("token")
        if cart is None:
            return _local("no signed cart with this id was received; call request_cart_mandate first")
        if not isinstance(token, str) or not token:
            return _local("send_payment_mandate needs a token")
        if self.signer is None:
            return _no_signer()
        details = cart["contents"]["payment_request"]["details"]
        pmid = stable_id("pmd", self.host.pack.id, self.session_id, self._calls)
        contents = {
            "payment_mandate_id": pmid,
            "payment_details_id": details["id"],
            "payment_details_total": details["total"],
            "payment_response": {
                "request_id": details["id"],
                "method_name": str(arguments.get("method_name") or "CARD"),
                "details": {"token": token},
            },
            "merchant_agent": cart["contents"]["merchant_name"],
            "timestamp": iso(self.clock.now()),
        }
        claims = {
            "aud": cart["contents"]["merchant_name"],
            "iat": int(self.clock.now().timestamp()),
            "nonce": pmid,
            "transaction_data": [mandate_hash(cart), mandate_hash(contents)],
        }
        mandate = {"payment_mandate_contents": contents, "user_authorization": self.signer.sign(claims, typ="kb+jwt")}
        body = {models.PAYMENT_MANDATE_DATA_KEY: mandate, models.CART_MANDATE_DATA_KEY: cart}
        return await self._post("/payment_mandates", body)

    async def _pay_v2(self, arguments: Mapping[str, Any]) -> ToolResult:
        signed = self._carts.get(str(arguments.get("checkout_id")))
        token = arguments.get("token")
        if signed is None:
            return _local("no signed checkout with this id was received; call create_checkout first")
        if not isinstance(token, str) or not token:
            return _local("complete_checkout needs a token")
        if self.signer is None:
            return _no_signer()
        checkout_jwt = str(signed["checkout_jwt"])
        _, claims = read_jws(checkout_jwt)
        total = next((t["amount"] for t in claims.get("totals") or () if t.get("type") == "total"), None)
        merchant = claims.get("merchant") or {"id": self.merchant, "name": self.merchant}
        times = {"iat": claims.get("iat"), "exp": claims.get("exp")}
        closed = {
            "vct": "mandate.checkout.1",
            "checkout_jwt": checkout_jwt,
            "checkout_hash": signed["checkout_hash"],
            **times,
        }
        payment = {
            "vct": "mandate.payment.1",
            "transaction_id": signed["checkout_hash"],
            "payee": {"id": merchant.get("id"), "name": merchant.get("name")},
            "payment_amount": {"amount": total, "currency": claims.get("currency")},
            "payment_instrument": {"id": token, "type": str(arguments.get("instrument_type") or "card")},
            **times,
        }
        body = {
            "checkout_mandate": self.signer.sign(closed, typ="vc+jwt"),
            "payment_mandate": self.signer.sign(payment, typ="vc+jwt"),
        }
        path = f"/checkouts/{quote(str(arguments.get('checkout_id')), safe='')}/complete"
        return await self._post(path, body)

    async def _status(self, arguments: Mapping[str, Any]) -> ToolResult:
        if self.ap2 == "v0.1.0" and not arguments.get("payment_mandate_id") and arguments.get("cart_id"):
            target, path = arguments.get("cart_id"), "/carts/{}"
        elif self.ap2 == "v0.1.0":
            target, path = arguments.get("payment_mandate_id"), "/payment_mandates/{}"
        else:
            target, path = arguments.get("checkout_id"), "/checkouts/{}"
        if not isinstance(target, str) or not target:
            return _local("this tool needs an id")
        headers = {"accept": JSON, **self._headers()}
        url = AP2_PATH + path.format(quote(target, safe=""))
        response, parsed, latency = await self._send("GET", url, headers=headers, protocol="ap2", version=self.version)
        return _result(response, parsed, latency)

    async def _post(self, path: str, body: Mapping[str, Any]) -> ToolResult:
        headers = {"accept": JSON, "content-type": JSON, **self._headers()}
        response, parsed, latency = await self._send(
            "POST", f"{AP2_PATH}{path}", body=dict(body), headers=headers, protocol="ap2", version=self.version
        )
        return _result(response, parsed, latency)


def _result(response: httpx.Response, parsed: Any, latency: int) -> ToolResult:
    failed = response.status_code >= 400
    doc = parsed if isinstance(parsed, Mapping) else None
    code = str(doc.get("error")) if failed and doc is not None and doc.get("error") else None
    return ToolResult(
        content=response.text,
        is_error=failed,
        error_code=(code or ("store_error" if response.status_code >= 500 else "protocol_error")) if failed else None,
        structured=doc,
        latency_ms=latency,
    )


def _local(message: str) -> ToolResult:
    return ToolResult(content=message, is_error=True, error_code="protocol_error")


def _no_signer() -> ToolResult:
    text = "the user's wallet must sign this mandate, and this surface holds no signer: open it with a wallet"
    return ToolResult(
        content=dumps({"error": "signer_required", "error_description": text}),
        is_error=True,
        error_code="signer_required",
    )

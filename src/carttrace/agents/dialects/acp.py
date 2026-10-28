"""ACP: a checkout session instead of a cart, a delegated payment token instead of a handler token, and the store's
catalog read beside it.

What differs from UCP, and why the agents still run the same loop:
- ACP has no catalog and no cart. A store that speaks ACP publishes its products elsewhere (in the sandbox, the
  replica's own UCP catalog is listed beside the ACP tools), and the first thing an agent makes is a checkout session
  holding the items. So the dialect's `create` is `create_checkout_session`, `read` is `get_checkout_session`, and
  the session's line items are the cart the agent reads back and the judge sees (`view_of`).
- The buyer's details go into the same session (`AcpEntry`: `buyer`, the fulfillment address and the selected
  option, each under the names the release uses, which the dialect reads from the tools' own schemas), so the
  reference agent's entry logic is unchanged.
- Payment is a delegated token (D13): the wallet issues the persona's card credential for an allowance of exactly the
  approved total (`DelegatedTokenIssuer`, behind the wallet's consent gate), the harness vaults it at the store's PSP
  through the surface's own `delegate_payment` (`delegation`, on the wire like any store call), and the agent pays with
  the `vt-` token the PSP answered, never with the credential itself (`credential_answer`, `tokens_of`).
- The order comes back inside the completed session (`order`), so there is no order tool; `order_view` reads it.

Shapes are data read from the listed tools' schemas (`from_specs`): items with quantities (2025-09-29) or repeated
line items with a currency and capabilities (2026-01-30 on), `fulfillment_address` or `fulfillment_details`, the three
ways of selecting an option, and a payment provider or payment handlers. A dialect built without schemas takes the
newest shapes.

Names. The personas have none, and the shopper refuses to give one, so the address's `name`, which every ACP release
requires as a member, is sent empty: the store keeps contact facts only, and nothing the shopper did not give is
invented (the judge counts an invented name against the agent).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Self

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
    DelegatedTokenRequest,
    Persona,
    Refusal,
)
from carttrace.core.money import Money
from carttrace.protocols.tools import ToolResult, ToolSpec
from carttrace.trace.schema import Cart, CartLine

SESSION_TOOLS: frozenset[str] = frozenset(
    {
        "create_checkout_session",
        "update_checkout_session",
        "get_checkout_session",
        "complete_checkout_session",
        "cancel_checkout_session",
    }
)
CART_TOOLS: frozenset[str] = frozenset({"create_checkout_session", "update_checkout_session", "get_checkout_session"})
"""Session tools whose result is the cart the agent is building (the session before payment)."""
DELEGATE = "delegate_payment"
HANDLER_NAMESPACE = "dev.acp.tokenized.card"
"""The sandbox's ACP payment handler's name (2026-01-30 on): a tokenized card that needs a delegated token."""
PROVIDER_NAMESPACE = "acp.payment_provider"
"""The namespace the wallet knows a release's `payment_provider` (up to 2026-01-16) under; the entry id is the
provider ACP names (stripe)."""
ALLOWANCE_DAYS = 1
"""How long a delegated token's allowance lasts: a day after the session was made, so a replay of an old pack, whose
logical clock starts at its capture, is not refused for an expired allowance."""
MISSING_PATHS: Mapping[str, str] = {
    "$.buyer.email": "email",
    "$.buyer": "email",
    "$.buyer.phone_number": "phone",
    "$.fulfillment_address": "shipping_address",
    "$.fulfillment_details.address": "shipping_address",
    "$.fulfillment_details": "shipping_address",
    "$.fulfillment_option_id": "fulfillment",
    "$.selected_fulfillment_options": "fulfillment",
}
"""An ACP session's `missing` message params to the buyer field each one asks for."""
GROUP_ID = "g1"
"""ACP has one fulfillment group per session: every item ships together."""
RISK_SIGNAL: Mapping[str, Any] = {"type": "card_testing", "score": 0, "action": "authorized"}
"""The agent platform's own fraud signal on a delegation, as ACP's PSP request carries it (required from 2025-09-29 to
2026-01-30): a sandbox persona's card is never being tested, so the one signal ACP defines scores nothing."""
_URL = re.compile(r"https?://\S+")
Selection = Literal["option_id", "nested", "flat"]
AddressMember = Literal["fulfillment_address", "fulfillment_details"]


@dataclass(frozen=True, slots=True)
class AcpDialect(Dialect):
    """ACP's checkout sessions, with the store's catalog read through UCP's catalog tools beside them."""

    create: str = "create_checkout_session"
    read: str = "get_checkout_session"
    create_checkout: str = "update_checkout_session"
    update_checkout: str = "update_checkout_session"
    get_checkout: str = "get_checkout_session"
    complete_checkout: str = "complete_checkout_session"
    get_order: str | None = None
    delegate: str | None = DELEGATE
    checkout_tools: frozenset[str] = SESSION_TOOLS
    order_tools: frozenset[str] = frozenset()
    payment_key: str = "payment_data"
    line_items: bool = True
    """Items are `line_items` without a quantity, with a currency and capabilities (2026-01-30 on); else `items`
    with quantities."""
    address_member: AddressMember = "fulfillment_details"
    selection: Selection = "flat"
    handlers: bool = True
    """The session declares payment handlers and a completion names one (2026-01-30 on); else a payment provider."""
    names_required: bool = False
    """The release requires a buyer's first and last name (2025-09-29); the agent has none to give."""

    @property
    def name(self) -> str:
        return "acp"

    @classmethod
    def from_specs(cls, specs: Mapping[str, ToolSpec]) -> Self:
        create = _payload(specs.get("create_checkout_session"))
        update = _payload(specs.get("update_checkout_session"))
        complete = _payload(specs.get("complete_checkout_session"))
        selected = update.get("selected_fulfillment_options")
        items = selected.get("items") if isinstance(selected, Mapping) else None
        nested = isinstance(items, Mapping) and "shipping" in (items.get("properties") or {})
        payment = complete.get("payment_data")
        payment_props = payment.get("properties") if isinstance(payment, Mapping) else None
        buyer = update.get("buyer")
        required = buyer.get("required") if isinstance(buyer, Mapping) else None
        return cls(
            line_items="line_items" in create if create else True,
            address_member="fulfillment_address" if "fulfillment_address" in update else "fulfillment_details",
            selection="option_id" if "fulfillment_option_id" in update else "nested" if nested else "flat",
            handlers="handler_id" in payment_props if isinstance(payment_props, Mapping) else True,
            names_required=isinstance(required, list) and "first_name" in required,
        )

    def offered(self, tools: Collection[str]) -> bool:
        return "create_checkout_session" in set(tools)

    # -- the cart: a checkout session ----------------------------------------------------------------------------------

    def lines_arguments(self, lines: Sequence[tuple[str, int]], *, currency: str | None = None) -> dict[str, Any]:
        if not self.line_items:
            return {"payload": {"items": [{"id": vid, "quantity": qty} for vid, qty in lines]}}
        items = [{"id": vid} for vid, qty in lines for _ in range(qty)]
        payload: dict[str, Any] = {"line_items": items, "capabilities": {}}
        if currency:
            payload["currency"] = currency.lower()
        return {"payload": payload}

    def is_cart_tool(self, name: str) -> bool:
        return name in CART_TOOLS

    def cart_in(self, result: ToolResult) -> Cart | None:
        if result.is_error:
            return None
        doc = self.checkout_doc(payload_of(result))
        return None if doc is None else self.view_of(doc)

    # -- the checkout: a session's documents ---------------------------------------------------------------------------

    def order_doc(self, payload: Any) -> Mapping[str, Any] | None:
        return None

    def view_of(self, doc: Mapping[str, Any]) -> Cart | None:
        return session_cart(doc, cart_id=str(doc.get("id") or "") or None)

    def order_view(self, doc: Mapping[str, Any]) -> Cart | None:
        order_id = self.order_of(doc)
        return None if order_id is None else session_cart(doc, cart_id=order_id)

    def order_label(self, doc: Mapping[str, Any]) -> str | None:
        order = doc.get("order")
        if not isinstance(order, Mapping):
            return None
        number = order.get("order_number")
        return str(number) if isinstance(number, str) and number else None

    def challenge_of(self, doc: Mapping[str, Any]) -> str | None:
        """The bank's challenge page a session names: its `continue_url` while authentication is required, or the
        URL the requires_3ds message states on releases without one."""
        asked = [m for m in _messages(doc) if m.get("code") == "requires_3ds"]
        if not asked:
            return None
        url = doc.get("continue_url")
        if isinstance(url, str) and "/3ds/" in url:
            return url
        for message in asked:
            found = _URL.search(str(message.get("content") or ""))
            if found is not None:
                return found.group(0).rstrip(".")
        return None

    def payment_failure(self, doc: Mapping[str, Any]) -> str | None:
        return next((str(m.get("content") or "") for m in _messages(doc) if m.get("code") == "payment_declined"), None)

    def handlers_of(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        capabilities = doc.get("capabilities")
        payment = capabilities.get("payment") if isinstance(capabilities, Mapping) else None
        listed = payment.get("handlers") if isinstance(payment, Mapping) else None
        out: dict[str, Any] = {}
        for handler in listed if isinstance(listed, list) else ():
            if isinstance(handler, Mapping) and isinstance(handler.get("id"), str):
                namespace = str(handler.get("name") or HANDLER_NAMESPACE)
                out.setdefault(namespace, []).append(
                    {k: handler[k] for k in ("id", "version", "config") if k in handler}
                )
        provider = doc.get("payment_provider")
        if not out and isinstance(provider, Mapping) and isinstance(provider.get("provider"), str):
            entry: dict[str, Any] = {"id": provider["provider"]}
            if isinstance(provider.get("merchant_id"), str):
                entry["config"] = {"merchant_id": provider["merchant_id"]}
            out[PROVIDER_NAMESPACE] = [entry]
        return out

    # -- opening a checkout with saved details ------------------------------------------------------------------------

    def open_checkout_calls(self, cart_id: str, details: Mapping[str, Any]) -> list[Call]:
        entry = AcpEntry(dialect=self)
        given = Details(
            email=details.get("email"),
            phone=details.get("phone_number"),
            address=None,
            method="shipping",
        )
        payload = entry.payload({"id": cart_id}, given)
        address = details.get("address")
        if isinstance(address, Mapping):
            payload[self.address_member] = entry.address_member(_ucp_to_acp_address(address))
        return [(self.update_checkout, {"id": cart_id, "payload": payload})]

    # -- paying --------------------------------------------------------------------------------------------------------

    def complete_arguments(self, checkout_id: str, answer: Mapping[str, Any]) -> dict[str, Any]:
        return {"id": checkout_id, "payload": {"payment_data": dict(answer[self.payment_key])}}

    def tokens_of(self, arguments: Mapping[str, Any]) -> set[str]:
        payload = arguments.get("payload")
        data = payload.get("payment_data") if isinstance(payload, Mapping) else None
        if not isinstance(data, Mapping):
            return set()
        token = data.get("token")
        instrument = data.get("instrument")
        credential = instrument.get("credential") if isinstance(instrument, Mapping) else None
        if isinstance(credential, Mapping) and isinstance(credential.get("token"), str):
            token = credential["token"]
        return {str(token)} if isinstance(token, str) and token else set()

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
        del namespace
        created = doc.get("created_at")
        start = _epoch_ms(created) if isinstance(created, str) else None
        expires = (start if start is not None else now_ms) + ALLOWANCE_DAYS * 24 * 3600 * 1000
        return DelegatedTokenRequest(
            checkout_id,
            amount,
            merchant_of(doc) or "",
            _iso(expires),
            handler_id=handler_id,
        )

    def delegation(self, credential: Credential, request: CredentialRequest) -> dict[str, Any]:
        assert isinstance(request, DelegatedTokenRequest)
        return {
            "payment_method": {
                "type": "card",
                "card_number_type": "network_token",
                "number": credential.token,
                "display_card_funding_type": "credit",
                "display_brand": credential.brand or "visa",
                "metadata": {},
            },
            "allowance": {
                "reason": "one_time",
                "max_amount": request.allowance.amount,
                "currency": request.allowance.currency.lower(),
                "checkout_session_id": request.checkout_id,
                "merchant_id": request.merchant_id,
                "expires_at": request.expires_at,
            },
            "risk_signals": [RISK_SIGNAL],
            "metadata": {},
        }

    def vaulted(self, result: ToolResult) -> str | None:
        if result.is_error:
            return None
        data = payload_of(result)
        token = data.get("id") if isinstance(data, Mapping) else None
        return token if isinstance(token, str) and token.startswith("vt-") else None

    def credential_answer(self, credential: Credential, *, token: str | None = None) -> dict[str, Any]:
        vault = token or credential.token
        if self.handlers:
            payment: dict[str, Any] = {
                "handler_id": credential.handler_id,
                "instrument": {"type": "card", "credential": {"type": "spt", "token": vault}},
            }
        else:
            payment = {"token": vault, "provider": credential.handler_id or "stripe"}
        return {
            self.payment_key: payment,
            "amount": {"amount": credential.amount.amount, "currency": credential.amount.currency},
            "use": f"put this payment_data in the payload of {self.complete_checkout} for this session",
        }


# -- the session as a cart ------


def session_cart(doc: Mapping[str, Any], *, cart_id: str | None) -> Cart | None:
    """A checkout session's line items and totals as the trace's `Cart`, or None when a line cannot be read. Each
    release names the unit price differently: `unit_amount` where lines carry details, else the line's base amount
    over its quantity."""
    currency = doc.get("currency")
    code = currency.upper() if isinstance(currency, str) and currency else None
    lines: list[CartLine] = []
    for raw in doc.get("line_items") or ():
        if not isinstance(raw, Mapping):
            return None
        item = raw.get("item") if isinstance(raw.get("item"), Mapping) else {}
        variant = item.get("id") if isinstance(item, Mapping) else None
        quantity = _as_int(raw.get("quantity", item.get("quantity") if isinstance(item, Mapping) else None))
        price = _unit_price(raw, quantity)
        if not isinstance(variant, str) or quantity is None or price is None or code is None:
            return None
        title = raw.get("name") if isinstance(raw.get("name"), str) else ""
        lines.append(CartLine(variant_id=variant, title=str(title), quantity=quantity, unit_price=price, currency=code))
    totals = {str(t.get("type")): t.get("amount") for t in doc.get("totals") or () if isinstance(t, Mapping)}
    return Cart(
        cart_id=cart_id,
        lines=lines,
        subtotal=_as_int(totals.get("subtotal")),
        total=_as_int(totals.get("total")),
        currency=code,
    )


def merchant_of(doc: Mapping[str, Any]) -> str | None:
    """The merchant id a session's payment handler or provider names, for a delegated token's allowance."""
    capabilities = doc.get("capabilities")
    payment = capabilities.get("payment") if isinstance(capabilities, Mapping) else None
    for handler in (payment.get("handlers") if isinstance(payment, Mapping) else None) or ():
        config = handler.get("config") if isinstance(handler, Mapping) else None
        merchant = config.get("merchant_id") if isinstance(config, Mapping) else None
        if isinstance(merchant, str) and merchant:
            return merchant
    provider = doc.get("payment_provider")
    merchant = provider.get("merchant_id") if isinstance(provider, Mapping) else None
    return merchant if isinstance(merchant, str) and merchant else None


def _unit_price(line: Mapping[str, Any], quantity: int | None) -> int | None:
    unit = _as_int(line.get("unit_amount"))
    if unit is not None:
        return unit
    base = _as_int(line.get("base_amount"))
    if base is None:
        base = next(
            (
                _as_int(t.get("amount"))
                for t in line.get("totals") or ()
                if isinstance(t, Mapping) and t.get("type") == "items_base_amount"
            ),
            None,
        )
    if base is not None and quantity is not None and quantity > 0 and base % quantity == 0:
        return base // quantity
    return None


def _as_int(value: Any) -> int | None:
    """`value` when it is a whole number (a bool is not), else None."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _messages(doc: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [m for m in doc.get("messages") or () if isinstance(m, Mapping)]


def _payload(spec: ToolSpec | None) -> Mapping[str, Any]:
    """The properties of a session tool's `payload`, as the surface lists them (the release's request schema)."""
    if spec is None:
        return {}
    payload = spec.input_schema.get("properties", {}).get("payload")
    props = payload.get("properties") if isinstance(payload, Mapping) else None
    return props if isinstance(props, Mapping) else {}


def _epoch_ms(value: str) -> int | None:
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        return None
    return int((when - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(milliseconds=1))


def _iso(ms: int) -> str:
    when = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=ms)
    return when.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _ucp_to_acp_address(address: Mapping[str, Any]) -> Mapping[str, str]:
    """A persona's saved UCP postal address as the ACP entry's address parts."""
    return {
        "line_one": str(address.get("street_address") or ""),
        "line_two": str(address.get("extended_address") or ""),
        "city": str(address.get("address_locality") or ""),
        "state": str(address.get("address_region") or ""),
        "postal_code": str(address.get("postal_code") or ""),
        "country": str(address.get("address_country") or ""),
    }


# -- the buyer's details through a session ------


class AcpEntry:
    """ACP: the session is the cart, so `create` reads it; details go in with `update_checkout_session`, under the
    names the release uses; options are the session's `fulfillment_options`, shipping only."""

    def __init__(self, specs: Mapping[str, ToolSpec] | None = None, *, dialect: AcpDialect | None = None) -> None:
        self.dialect = dialect or (AcpDialect.from_specs(specs) if specs else AcpDialect())

    @property
    def name(self) -> str:
        return "acp"

    def offered(self, tools: Collection[str]) -> bool:
        return {"create_checkout_session", "update_checkout_session"} <= set(tools)

    def create(self, cart_id: str) -> Call:
        return "get_checkout_session", {"id": cart_id}

    def missing(self, doc: Mapping[str, Any]) -> list[str]:
        found: set[str] = set()
        for message in _messages(doc):
            if message.get("code") != "missing":
                continue
            field = MISSING_PATHS.get(str(message.get("param") or ""))
            if field is not None:
                found.add(field)
        return [f for f in ASK_ORDER if f in found]

    def enter(self, doc: Mapping[str, Any], details: Details) -> list[Call]:
        payload = self.payload(doc, details)
        if not payload:
            return []
        return [(self.dialect.update_checkout, {"id": str(doc["id"]), "payload": payload})]

    def payload(self, doc: Mapping[str, Any], details: Details) -> dict[str, Any]:
        """The update body for the disclosed details: `buyer` for contact facts, the address under the release's
        member when the shopper ships (a pickup needs none, and ACP's sandbox ships only)."""
        del doc
        payload: dict[str, Any] = {}
        buyer = {k: v for k, v in (("email", details.email), ("phone_number", details.phone)) if v is not None}
        if buyer:
            payload["buyer"] = buyer
        if details.address is not None and details.method != "pickup":
            address = details.address
            parts = {
                "line_one": address.lines[0],
                "line_two": address.lines[1] if len(address.lines) > 1 else "",
                "city": address.city,
                "state": address.region or "",
                "postal_code": address.postal_code,
                "country": address.country,
            }
            payload[self.dialect.address_member] = self.address_member(parts)
        return payload

    def address_member(self, parts: Mapping[str, str]) -> dict[str, Any]:
        """The address as the release's member takes it: the shopper's parts, a name left empty (the personas have
        none, and the schema requires the member), under `fulfillment_address` or inside `fulfillment_details`."""
        address: dict[str, Any] = {"name": "", "line_one": parts["line_one"]}
        if parts.get("line_two"):
            address["line_two"] = parts["line_two"]
        address.update(
            city=parts["city"], state=parts.get("state", ""), country=parts["country"], postal_code=parts["postal_code"]
        )
        if self.dialect.address_member == "fulfillment_address":
            return address
        return {"name": "", "address": address}

    def options(self, doc: Mapping[str, Any]) -> list[Option]:
        found: list[Option] = []
        for option in doc.get("fulfillment_options") or ():
            if not isinstance(option, Mapping) or not isinstance(option.get("id"), str):
                continue
            amount = _as_int(option.get("total"))
            if amount is None:
                amount = next(
                    (
                        _as_int(t.get("amount"))
                        for t in option.get("totals") or ()
                        if isinstance(t, Mapping) and t.get("type") == "total"
                    ),
                    None,
                )
            if amount is None:
                continue
            kind = str(option.get("type") or "shipping")
            title = option["title"] if isinstance(option.get("title"), str) else option["id"]
            found.append(
                Option(
                    method_id="fulfillment",
                    method=kind,
                    group_id=GROUP_ID,
                    option_id=option["id"],
                    title=title,
                    amount=amount,
                    kind=option_kind(title, option["id"], kind),
                    days=stated_days(title),
                )
            )
        return sorted(found, key=lambda o: o.amount)

    def select(self, doc: Mapping[str, Any], option: Option) -> Call:
        d = self.dialect
        if d.selection == "option_id":
            payload: dict[str, Any] = {"fulfillment_option_id": option.option_id}
        else:
            items = [str(li["id"]) for li in doc.get("line_items") or () if isinstance(li, Mapping) and "id" in li]
            if d.selection == "nested":
                chosen: dict[str, Any] = {
                    "type": "shipping",
                    "shipping": {"option_id": option.option_id, "item_ids": items},
                }
            else:
                chosen = {"type": "shipping", "option_id": option.option_id, "item_ids": items}
            payload = {"selected_fulfillment_options": [chosen]}
        return d.update_checkout, {"id": str(doc["id"]), "payload": payload}

    def selected(self, doc: Mapping[str, Any]) -> Option | None:
        chosen = doc.get("fulfillment_option_id")
        if not isinstance(chosen, str):
            for pick in doc.get("selected_fulfillment_options") or ():
                if not isinstance(pick, Mapping):
                    continue
                shipping = pick.get("shipping")
                chosen = shipping.get("option_id") if isinstance(shipping, Mapping) else pick.get("option_id")
                if isinstance(chosen, str):
                    break
        return next((o for o in self.options(doc) if o.option_id == chosen), None)

    def ready(self, doc: Mapping[str, Any]) -> bool:
        return doc.get("status") == "ready_for_payment"

    def total(self, doc: Mapping[str, Any]) -> Money | None:
        return total_of(doc)


# -- the delegated token's issuer ------


class DelegatedTokenIssuer:
    """ACP delegated tokens: the persona's card credential, issued for an allowance of exactly the approved total, for
    one checkout session. The wallet's consent gate runs first; the harness then vaults the credential at the store's
    PSP (`AcpDialect.delegation`) and the agent pays with the vaulted token, never with this one."""

    @property
    def kind(self) -> CredentialKind:
        return "acp_delegated_token"

    def refusal(self, persona: Persona, request: CredentialRequest) -> Refusal | None:
        if not isinstance(request, DelegatedTokenRequest):
            return Refusal("credential_kind_mismatch", f"a {request.kind} request reached the ACP token issuer")
        if _card(persona) is None:
            kinds = ", ".join(i.kind for i in persona.instruments)
            return Refusal(
                "handler_unsupported",
                f"the shopper's wallet holds no card to delegate (it holds: {kinds})",
                hint="ACP's payment handler takes a tokenized card; this persona cannot pay here",
            )
        return None

    def issue(self, persona: Persona, request: CredentialRequest, consent: Consent, *, sequence: int) -> Credential:
        assert isinstance(request, DelegatedTokenRequest)
        card = _card(persona)
        assert card is not None
        return Credential(
            kind=self.kind,
            checkout_id=request.checkout_id,
            amount=request.allowance,
            token=card.token,
            instrument="card",
            instrument_id=f"wallet-instrument-{sequence}",
            consent=consent.seq,
            handler_id=request.handler_id,
            brand=card.brand,
        )


def _card(persona: Persona) -> Any:
    return next((i for i in persona.instruments if i.kind == "card"), None)

"""The commands the core accepts, protocol-neutral, validated strictly before anything is applied.

Why strict. Adapters (UCP over MCP and REST, ACP, MPP, AP2) translate their wire arguments into these. A wrong type is
refused here with a typed `InvalidCommand` naming the field, so the adapter can answer its protocol's own "invalid
params" instead of a traceback. Strict means strict: `"2"` is not a quantity, a tuple is not a list, an unknown field
is an error rather than silently dropped, and a payment credential is an opaque `tok_...` token or nothing.

Every command carries the session it acts for, who is acting (the agent, the merchant, or the environment itself), an
optional idempotency key, and an optional `at` (logical milliseconds). The runtime stamps `at` from an injected clock
and records it, so replaying the command log reproduces every timestamp.

A refusal never echoes what the command said: field paths are sanitised and messages come from the schema, so a card
number or a secret pasted into a command cannot reach the event log through its own error (`carttrace.sandbox.core.guard`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

import pydantic
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from carttrace.core.canon import digest
from carttrace.sandbox.core.errors import InvalidCommand
from carttrace.sandbox.core.guard import find_card_number, safe_name, safe_path

Id = Annotated[str, Field(min_length=1, max_length=200)]
Country = Annotated[str, Field(pattern=r"^[A-Z]{2}$")]
Currency = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
Qty = Annotated[int, Field(ge=0, le=10_000)]
Code = Annotated[str, Field(min_length=1, max_length=64)]
RawToken = Annotated[str, Field(pattern=r"^tok_[a-z0-9_]{1,60}$")]
"""An opaque test credential. A card number cannot match this, by design."""
Token = Annotated[str, Field(pattern=r"^(?:tok_[a-z0-9_]{1,60}|vt-[0-9a-f]{12}|shop_[0-9a-f]{12})$")]
"""What a payment presents: an opaque test credential, the token of one the PSP vaulted (`vt-...`, from
`vault_credential`), or a Shop Token the wallet's platform minted (`shop_...`, from `mint_shop_token`). No shape can
hold a card number."""
Method = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z_-]{0,39}$")]
ProfileId = Annotated[str, Field(pattern=r"^pp-[0-9a-f]{12}$")]
ProtocolName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]{0,31}$")]
Digest = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{43}$")]
"""base64url sha256, unpadded. Not hex: a long hex run of decimal digits could pass for a card number."""
Tier = Literal["anonymous", "signed", "token"]
Actor = Literal["agent", "merchant", "system"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class LineIn(_M):
    line_id: Id | None = None
    """An existing line's id, exactly as the store returned it (adapters strip their own decorations)."""
    variant_id: Id | None = None
    quantity: Qty

    @model_validator(mode="after")
    def _one_ref(self) -> Self:
        if self.line_id is None and self.variant_id is None:
            raise ValueError("a line needs line_id (to change an existing line) or variant_id (to add one)")
        return self


class Buyer(_M):
    email: Annotated[str, Field(pattern=r"^[^@\s]+@[^@\s]+$", max_length=254)] | None = None
    phone_number: Annotated[str, Field(max_length=40)] | None = None


class Address(_M):
    id: Id | None = None
    first_name: Annotated[str, Field(max_length=200)] | None = None
    last_name: Annotated[str, Field(max_length=200)] | None = None
    street_address: Annotated[str, Field(max_length=500)] | None = None
    extended_address: Annotated[str, Field(max_length=500)] | None = None
    address_locality: Annotated[str, Field(max_length=200)] | None = None
    address_region: Annotated[str, Field(max_length=200)] | None = None
    postal_code: Annotated[str, Field(max_length=40)] | None = None
    address_country: Country


class GroupIn(_M):
    id: Id | None = None
    method: Literal["shipping", "pickup"] = "shipping"
    line_ids: list[Id] | None = None
    """Lines this group fulfils. None means every line no other group claims."""
    destination: Address | None = None
    location_id: Id | None = None
    selected_option_id: Id | None = None


class PaymentIn(_M):
    handler: Id
    """The handler id or namespace the store declared, e.g. "shopify.card" or "com.google.pay"."""
    instrument: Method | None = None
    """The payment method type: card, cod, upi. Case is ignored."""
    token: Token | None = None
    """The opaque test credential (`carttrace.sandbox.core.psp.TOKENS`), or a vaulted one's token, charged within its
    allowance. The fake handler and cash on delivery need none."""
    request_id: Id | None = None
    """The payment request this payment answers (an MPP challenge, an AP2 cart): it must still be open, unexpired, for
    this checkout, and for exactly the checkout's total."""


class HandoffLine(_M):
    variant_id: Id
    quantity: Annotated[int, Field(ge=1, le=10_000)]
    unit_price: Annotated[int, Field(ge=0)]
    """The live cart's unit price in minor units; the hand-off is refused if the sandbox's price differs."""


class _Cmd(_M):
    session: Id = "default"
    actor: Actor = "agent"
    idempotency_key: Annotated[str, Field(min_length=1, max_length=255)] | None = None
    at: Annotated[int, Field(ge=0)] | None = None


class OpenSession(_Cmd):
    type: Literal["open_session"] = "open_session"
    session_id: Id
    tier: Tier
    permissions: list[Id] = Field(default_factory=list)
    profile_id: ProfileId | None = None
    """The ProtocolProfile the session speaks under; the pack's own profile when absent."""


class RevokeToken(_Cmd):
    type: Literal["revoke_token"] = "revoke_token"
    session_id: Id


class Read(_Cmd):
    type: Literal["read"] = "read"
    target: Literal["catalog", "product", "cart", "checkout", "order"]
    id: Annotated[str, Field(max_length=500)] | None = None


class CreateCart(_Cmd):
    type: Literal["create_cart"] = "create_cart"
    lines: list[LineIn] = Field(default_factory=list)
    codes: list[Code] = Field(default_factory=list)
    buyer: Buyer | None = None
    country: Country | None = None


class UpdateCart(_Cmd):
    type: Literal["update_cart"] = "update_cart"
    cart_id: Id
    lines: list[LineIn] | None = None
    """A full replacement, as UCP and the live servers do: lines not named are removed."""
    codes: list[Code] | None = None
    buyer: Buyer | None = None


class CancelCart(_Cmd):
    type: Literal["cancel_cart"] = "cancel_cart"
    cart_id: Id


class HandoffCart(_Cmd):
    """The environment re-creates a live cart in the sandbox, so a live journey continues to checkout here."""

    type: Literal["handoff_cart"] = "handoff_cart"
    lines: list[HandoffLine] = Field(min_length=1)
    currency: Currency
    buyer: Buyer | None = None
    country: Country | None = None
    source: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.:-]{0,63}$")] = "live"
    """Where the cart was read, e.g. "live:get_cart"."""
    cart_ref: Annotated[str, Field(pattern=r"^[^?\s]{1,200}$")] | None = None
    """The live cart id as the trace recorded it, without any query part (where a cart token's secret key lives)."""


class CreateCheckout(_Cmd):
    type: Literal["create_checkout"] = "create_checkout"
    cart_id: Id | None = None
    lines: list[LineIn] | None = None
    """Without a cart_id, a checkout from lines directly (a cart is made for it, as UCP allows)."""
    codes: list[Code] | None = None
    buyer: Buyer | None = None
    country: Country | None = None
    fulfillment: list[GroupIn] | None = None

    @model_validator(mode="after")
    def _source(self) -> Self:
        if self.cart_id is None and not self.lines:
            raise ValueError("a checkout needs cart_id or lines")
        return self


class UpdateCheckout(_Cmd):
    type: Literal["update_checkout"] = "update_checkout"
    checkout_id: Id
    lines: list[LineIn] | None = None
    codes: list[Code] | None = None
    buyer: Buyer | None = None
    fulfillment: list[GroupIn] | None = None
    """Groups are merged by id: a group named again keeps what it had unless the update says otherwise."""


class CancelCheckout(_Cmd):
    type: Literal["cancel_checkout"] = "cancel_checkout"
    checkout_id: Id


class CompleteCheckout(_Cmd):
    type: Literal["complete_checkout"] = "complete_checkout"
    checkout_id: Id
    payment: PaymentIn


class ConfirmPaymentAction(_Cmd):
    type: Literal["confirm_payment_action"] = "confirm_payment_action"
    checkout_id: Id
    approved: bool = True
    """The buyer's own step (a 3-D Secure challenge, a bank app). True when the buyer completed it."""


class FulfillOrder(_Cmd):
    type: Literal["fulfill_order"] = "fulfill_order"
    order_id: Id
    line_ids: list[Id] | None = None


class CancelOrder(_Cmd):
    type: Literal["cancel_order"] = "cancel_order"
    order_id: Id
    reason: Annotated[str, Field(max_length=64)] = "other"


class RefundOrder(_Cmd):
    type: Literal["refund_order"] = "refund_order"
    order_id: Id
    amount: Annotated[int, Field(ge=1)]


class EditOrder(_Cmd):
    type: Literal["edit_order"] = "edit_order"
    order_id: Id
    line_id: Id
    quantity: Qty


class RequestReturn(_Cmd):
    type: Literal["request_return"] = "request_return"
    order_id: Id
    line_ids: list[Id] | None = None


class ReceiveReturn(_Cmd):
    type: Literal["receive_return"] = "receive_return"
    order_id: Id


class VaultCredential(_Cmd):
    """The PSP vaults an opaque test credential behind a token of its own, usable once, for one checkout, within an
    allowance (ACP's delegated payment). The credential itself is never shown again."""

    type: Literal["vault_credential"] = "vault_credential"
    token: RawToken
    max_amount: Annotated[int, Field(ge=1)]
    """Minor units of `currency`: the most a completion may charge."""
    currency: Currency
    checkout_id: Id
    """The checkout the allowance is for; a completion of any other checkout is refused."""
    expires_at: Annotated[int, Field(ge=0)]
    """Logical milliseconds."""
    merchant: Annotated[str, Field(min_length=1, max_length=256)] | None = None


class MintShopToken(_Cmd):
    """The wallet's platform mints a single-use Shop Token from a persona's Shop Pay credential, scoped to one
    checkout and the store's merchant id, valid for a short logical window (Shop Pay's Path A, D73:
    vendor/dev.shopify.shop_pay/2026-04-08/spec.md). The credential itself is never shown again; the minted token is
    what a completion through the dev.shopify.shop_pay handler presents."""

    type: Literal["mint_shop_token"] = "mint_shop_token"
    token: RawToken
    """The wallet's Shop Pay credential (`tok_shop_pay_...` in `carttrace.sandbox.core.psp.TOKENS`)."""
    checkout_id: Id
    """The checkout the token is scoped to; its total must be exactly `amount` (the spec's "context about the
    checkout"). A completion of any other checkout declines."""
    amount: Annotated[int, Field(ge=1)]
    """Minor units of `currency`: the checkout's total the buyer confirmed in the Shop Pay interface."""
    currency: Currency


class RequestPayment(_Cmd):
    """The store asks to be paid an exact amount for one checkout, once, before a deadline: an MPP challenge or an AP2
    merchant-signed cart. Sent by the merchant for the checkout's session."""

    type: Literal["request_payment"] = "request_payment"
    request_id: Id
    checkout_id: Id
    protocol: ProtocolName
    amount: Annotated[int, Field(ge=0)]
    currency: Currency
    expires_at: Annotated[int, Field(ge=0)]


class RecordMandate(_Cmd):
    """The store keeps a mandate it received or issued, by digest, after its surface verified the signature."""

    type: Literal["record_mandate"] = "record_mandate"
    mandate_id: Id
    kind: Literal["intent", "cart", "checkout", "payment"]
    protocol: ProtocolName
    digest: Digest
    signer: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    checkout_id: Id | None = None
    request_id: Id | None = None
    ref_ids: list[Id] = Field(default_factory=list)
    expires_at: Annotated[int, Field(ge=0)] | None = None


class SetStock(_Cmd):
    type: Literal["set_stock"] = "set_stock"
    variant_id: Id
    on_hand: Annotated[int, Field(ge=0)]


class SetPrice(_Cmd):
    type: Literal["set_price"] = "set_price"
    variant_id: Id
    price: Annotated[int, Field(ge=0)]


Command = (
    OpenSession
    | RevokeToken
    | Read
    | CreateCart
    | UpdateCart
    | CancelCart
    | HandoffCart
    | CreateCheckout
    | UpdateCheckout
    | CancelCheckout
    | CompleteCheckout
    | ConfirmPaymentAction
    | FulfillOrder
    | CancelOrder
    | RefundOrder
    | EditOrder
    | RequestReturn
    | ReceiveReturn
    | VaultCredential
    | MintShopToken
    | RequestPayment
    | RecordMandate
    | SetStock
    | SetPrice
)
"""Every command, discriminated by `type`."""

COMMANDS: tuple[type[_Cmd], ...] = (
    OpenSession,
    RevokeToken,
    Read,
    CreateCart,
    UpdateCart,
    CancelCart,
    HandoffCart,
    CreateCheckout,
    UpdateCheckout,
    CancelCheckout,
    CompleteCheckout,
    ConfirmPaymentAction,
    FulfillOrder,
    CancelOrder,
    RefundOrder,
    EditOrder,
    RequestReturn,
    ReceiveReturn,
    VaultCredential,
    MintShopToken,
    RequestPayment,
    RecordMandate,
    SetStock,
    SetPrice,
)
TYPES: tuple[str, ...] = tuple(str(cls.model_fields["type"].default) for cls in COMMANDS)
MUTATING = frozenset(TYPES) - {"read"}
"""Commands an idempotency key protects. A read changes nothing, so it needs none."""

_ADAPTER: TypeAdapter[Command] = TypeAdapter(Annotated[Command, Field(discriminator="type")])


def parse(raw: object) -> Command:
    """A JSON object (or an already typed command) to a typed command, or an `InvalidCommand` naming every bad field."""
    if isinstance(raw, COMMANDS):
        _refuse_card_numbers(raw.model_dump(mode="json"))
        return _ADAPTER.validate_python(raw)  # an instance passes through unchanged
    if not isinstance(raw, Mapping):
        raise InvalidCommand(
            code="invalid_command",
            message=f"a command is a JSON object, got {type(raw).__name__}",
            hint="send an object with a type field",
        )
    kind = raw.get("type")
    if kind not in TYPES:
        shown = safe_name(kind)
        raise InvalidCommand(
            code="unknown_command",
            message=f"unknown command type {shown!r}" if shown else "unknown command type",
            field="type",
            hint=f"use one of {', '.join(TYPES)}",
        )
    _refuse_card_numbers(raw)
    try:
        return _ADAPTER.validate_python(dict(raw))
    except pydantic.ValidationError as e:
        details = tuple({"field": safe_path(err["loc"][1:]) or "type", "message": err["msg"]} for err in e.errors())
        first = details[0] if details else {"field": "type", "message": "invalid"}
        raise InvalidCommand(
            code="invalid_argument",
            message=f"{first['field']}: {first['message']}",
            field=first["field"],
            details=details,
        ) from None


def raw_type(command: object) -> str | None:
    """The type a raw command claims, only when it is identifier-shaped: what a refusal may safely log."""
    claimed = command.get("type") if isinstance(command, Mapping) else getattr(command, "type", None)
    return safe_name(claimed)


def _refuse_card_numbers(body: Mapping[str, Any]) -> None:
    path = find_card_number(body)
    if path is not None:
        raise InvalidCommand(
            code="card_number_refused",
            message=f"{path}: a card number was sent; this store takes payment tokens only",
            field=path,
            hint="pay with a token the shopper's wallet issued, never a card number",
        )


def dump(cmd: Command) -> dict[str, Any]:
    """The command as JSON, every field included: what the log records and what a replay re-parses."""
    out: dict[str, Any] = cmd.model_dump(mode="json")
    return out


def body_hash(cmd: Command) -> str:
    """What makes two requests 'the same request' for idempotency: everything except the key and the timestamp."""
    return digest(cmd.model_dump(mode="json", exclude={"at", "idempotency_key"}))

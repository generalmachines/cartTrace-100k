"""The store's state as typed, immutable values: inventory, sessions, carts, checkouts, reservations, payments, orders,
the PSP's vaulted credentials, payment requests, recorded mandates, the idempotency store and the webhook outbox.

Why immutable. `apply` takes a state and returns a new one; the old one is never touched, so a grader can hold the
state before an episode and the state after and diff them, and a failed invariant simply means the new state is
dropped. Maps are copied on write (`carttrace.sandbox.core.tx`), so a command that touches one cart copies the cart map and not
the catalog, and the top-level maps of a `State` are read-only views, so nothing can edit a committed state in place.

Why only mutable facts. Product titles, descriptions and option structures never change during an episode and live in
the environment pack; the state carries only what a journey can change (price, stock, reservations, availability), so
a state hash is small and a diff names exactly what moved.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from carttrace.core.canon import digest


def plain(obj: Any) -> Any:
    """A dataclass tree to JSON-able dicts and lists, without `dataclasses.asdict`'s deep copy of every leaf."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [plain(v) for v in obj]
    return obj


@dataclass(frozen=True, slots=True)
class Variant:
    id: str
    product_id: str
    title: str
    price: int
    """Minor units of `currency`."""
    currency: str
    available: bool
    on_hand: int
    reserved: int = 0
    requires_selling_plan: bool = False

    @property
    def sellable(self) -> int:
        return max(0, self.on_hand - self.reserved) if self.available else 0


@dataclass(frozen=True, slots=True)
class Line:
    id: str
    variant_id: str
    quantity: int
    unit_price: int


@dataclass(frozen=True, slots=True)
class Totals:
    subtotal: int = 0
    discount: int = 0
    fulfillment: int | None = None
    """None until every fulfillment group has a selected option: the store cannot quote shipping it has not chosen."""
    tax: int | None = None
    """Tax added on top. None until fulfillment is known; 0 in a tax-inclusive market."""
    tax_included: int = 0
    """Tax already inside the prices of a tax-inclusive market. Shown, never added."""
    total: int = 0


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    tier: str
    """anonymous, signed or token (`carttrace.sandbox.core.trust.TIERS`)."""
    permissions: tuple[str, ...] = ()
    revoked: bool = False
    profile_id: str | None = None
    """The ProtocolProfile id (`pp-...`) the session speaks under, so a grader knows which versions it saw."""


@dataclass(frozen=True, slots=True)
class Cart:
    id: str
    session: str
    status: str
    lines: tuple[Line, ...]
    codes: tuple[str, ...]
    buyer: Mapping[str, str]
    country: str | None
    totals: Totals
    created_at: int
    checkout_id: str | None = None


@dataclass(frozen=True, slots=True)
class Group:
    id: str
    method: str
    """shipping or pickup. One checkout may hold both (Shopify mixed fulfillment)."""
    line_ids: tuple[str, ...]
    destination: Mapping[str, str] | None = None
    location_id: str | None = None
    options: tuple[Mapping[str, Any], ...] = ()
    selected_option_id: str | None = None
    amount: int | None = None
    explicit_lines: bool = False
    """True when the agent named this group's lines; otherwise it takes every line no other group claims."""


@dataclass(frozen=True, slots=True)
class Checkout:
    id: str
    session: str
    cart_id: str
    status: str
    lines: tuple[Line, ...]
    codes: tuple[str, ...]
    buyer: Mapping[str, str]
    country: str | None
    groups: tuple[Group, ...]
    totals: Totals
    created_at: int
    reservation_id: str | None = None
    payment_id: str | None = None
    order_id: str | None = None
    escalation: str | None = None
    """Why the checkout needs a human: payment_handler_unavailable, payment_action_required. None otherwise."""


@dataclass(frozen=True, slots=True)
class Reservation:
    id: str
    checkout_id: str
    status: str
    items: tuple[tuple[str, int], ...]
    expires_at: int


@dataclass(frozen=True, slots=True)
class Payment:
    id: str
    checkout_id: str
    status: str
    handler: str
    """The handler id the agent named, as the store declared it (e.g. "gpay", "shopify.card")."""
    namespace: str
    """The handler's namespace (e.g. "com.google.pay"), which decides the credentials it takes."""
    instrument: str | None
    """The payment method type: card, cod, upi."""
    currency: str
    amount: int
    """What the buyer was asked to pay; the authorization once authorized."""
    token: str | None = None
    """The opaque test credential (`tok_...`). Never a card number: commands refuse anything else."""
    brand: str | None = None
    challenge_id: str | None = None
    """Set when a 3-D Secure challenge was issued for this payment."""
    challenge_outcome: str | None = None
    """Sealed: how the issuer will answer the challenge, pass or fail. Adapters never show it to the agent."""
    decline_code: str | None = None
    """Why the payment was declined or failed authentication."""
    authorized: int = 0
    captured: int = 0
    refunded: int = 0
    request_id: str | None = None
    """The payment request this payment answers (`PaymentRequest`), when it answers one."""
    credential_id: str | None = None
    """The vaulted credential it charges (`VaultedCredential`); `token` is then the credential the vault holds."""
    authentication: str = "none"
    """How 3-D Secure ended (D71, `carttrace.sandbox.core.brokers.AUTHENTICATIONS`): none, frictionless, pending,
    challenge_passed, challenge_failed or challenge_abandoned."""
    idempotency_key: str | None = None
    """The idempotency key of the completion that made it, which the store passes to its processor."""


@dataclass(frozen=True, slots=True)
class VaultedCredential:
    """An opaque test credential the sandbox's PSP holds behind a token of its own, with an allowance: who may charge
    it, for which checkout, up to how much, in which currency and until when (ACP's delegated payment)."""

    id: str
    session: str
    status: str
    """active, or consumed once a payment charged to it was captured."""
    token: str
    """The `tok_...` credential it stands for. Never shown to the agent again."""
    max_amount: int
    currency: str
    checkout_id: str
    expires_at: int
    """Logical milliseconds; a completion after this is refused."""
    merchant: str | None
    created_at: int
    payment_id: str | None = None


@dataclass(frozen=True, slots=True)
class PaymentRequest:
    """An amount the store asked to be paid for one checkout: single use and expiring. An MPP 402 challenge and an AP2
    merchant-signed cart are payment requests; the payment that answers one must match its amount and currency."""

    id: str
    """The protocol's own reference: an MPP challenge id, an AP2 payment details id or checkout hash."""
    session: str
    checkout_id: str
    protocol: str
    status: str
    """open, pending (a payment for it waits on the buyer's step), paid, failed or void."""
    amount: int
    currency: str
    expires_at: int
    issued_at: int
    payment_id: str | None = None


@dataclass(frozen=True, slots=True)
class Mandate:
    """A signed statement of consent the store received or issued (a user's intent, a merchant's cart, a user's
    payment authorization), kept by its digest so a grader sees exactly what was authorized. Never changed."""

    id: str
    session: str
    kind: str
    """intent, cart, checkout or payment."""
    protocol: str
    digest: str
    """base64url sha256 of the mandate's canonical JSON."""
    signer: str | None
    """The key id whose signature was verified, or None for an unsigned mandate."""
    recorded_at: int
    checkout_id: str | None = None
    request_id: str | None = None
    ref_ids: tuple[str, ...] = ()
    """Other mandates this one binds (the intent a cart answers, the cart a payment authorizes)."""
    expires_at: int | None = None


@dataclass(frozen=True, slots=True)
class OrderLine:
    id: str
    variant_id: str
    quantity: int
    unit_price: int
    fulfilled: int = 0
    returned: int = 0


@dataclass(frozen=True, slots=True)
class Order:
    id: str
    number: int
    checkout_id: str
    session: str
    idempotency_key: str | None
    status: str
    lines: tuple[OrderLine, ...]
    totals: Totals
    codes: tuple[str, ...]
    groups: tuple[Group, ...]
    buyer: Mapping[str, str]
    payment_id: str
    currency: str
    placed_at: int
    return_line_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IdemRecord:
    session: str
    key: str
    command: str
    body_hash: str
    seq: int
    outcome: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Webhook:
    id: str
    topic: str
    order_id: str
    seq: int
    delivery: int = 1


MAPS = (
    "inventory",
    "sessions",
    "carts",
    "checkouts",
    "reservations",
    "payments",
    "orders",
    "credentials",
    "payment_requests",
    "mandates",
    "idempotency",
    "faults_fired",
    "overlays",
)
"""The top-level maps of a `State`, each keyed by id."""


@dataclass(frozen=True, slots=True)
class State:
    seed: int
    currency: str
    seq: int = 0
    clock: int = 0
    """Logical milliseconds: the command's `at`, or the previous clock plus the pack's tick."""
    id_counter: int = 0
    inventory: Mapping[str, Variant] = field(default_factory=dict)
    sessions: Mapping[str, Session] = field(default_factory=dict)
    carts: Mapping[str, Cart] = field(default_factory=dict)
    checkouts: Mapping[str, Checkout] = field(default_factory=dict)
    reservations: Mapping[str, Reservation] = field(default_factory=dict)
    payments: Mapping[str, Payment] = field(default_factory=dict)
    orders: Mapping[str, Order] = field(default_factory=dict)
    credentials: Mapping[str, VaultedCredential] = field(default_factory=dict)
    """Credentials the PSP vaulted with an allowance, by their `vt-` token."""
    payment_requests: Mapping[str, PaymentRequest] = field(default_factory=dict)
    mandates: Mapping[str, Mandate] = field(default_factory=dict)
    idempotency: Mapping[str, IdemRecord] = field(default_factory=dict)
    webhooks: tuple[Webhook, ...] = ()
    faults_fired: Mapping[str, int] = field(default_factory=dict)
    overlays: Mapping[str, str] = field(default_factory=dict)
    """Text injected into a product's description by an `injected_text` fault, keyed by product id."""

    def __post_init__(self) -> None:
        for name in MAPS:
            value = getattr(self, name)
            if not isinstance(value, MappingProxyType):
                object.__setattr__(self, name, MappingProxyType(dict(value)))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = plain(self)
        return out

    def hash(self) -> str:
        """sha256 of the state's canonical JSON. Two runs that agree on every fact agree on this."""
        return digest(self.to_dict())

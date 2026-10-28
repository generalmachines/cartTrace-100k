"""The sandbox's protocol-neutral state core: typed immutable state, one `apply(state, command, config)`, the
transition tables and trust policy as data, idempotency, stock reservations, the PSP simulator's test tokens, its
vaulted credentials and payment requests, recorded mandates, and invariants checked on every commit.

Nothing here knows UCP, ACP, MPP or AP2, and nothing here imports another part of the sandbox. Adapters translate
their protocol into these commands and render these states back.
"""

from carttrace.sandbox.core.apply import apply, initial_state, outcome, reject
from carttrace.sandbox.core.commands import MUTATING, TYPES, Command, parse
from carttrace.sandbox.core.config import (
    DELEGATED_NAMESPACE,
    FAKE_HANDLERS,
    FAKE_NAMESPACE,
    REQUESTED_NAMESPACE,
    Config,
    FaultRule,
)
from carttrace.sandbox.core.errors import (
    BusinessRule,
    Conflict,
    CoreError,
    IllegalTransition,
    InvalidCommand,
    InvariantViolation,
    NotFound,
    PolicyDenied,
    Unavailable,
)
from carttrace.sandbox.core.event import Event
from carttrace.sandbox.core.invariants import check as check_invariants
from carttrace.sandbox.core.machines import (
    CART,
    CHECKOUT,
    CREDENTIAL,
    INITIAL,
    MACHINES,
    ORDER,
    PAYMENT,
    PAYMENT_REQUEST,
    RESERVATION,
)
from carttrace.sandbox.core.market import MARKETS, Market, PickupLocation, ShippingOption, UnknownMarket, market, market_for
from carttrace.sandbox.core.psp import DECLINE_CODES, HANDLER_KINDS, HANDLER_RULES, TOKENS, HandlerRule, TokenSpec
from carttrace.sandbox.core.state import (
    Cart,
    Checkout,
    Group,
    Line,
    Mandate,
    Order,
    OrderLine,
    Payment,
    PaymentRequest,
    Reservation,
    Session,
    State,
    Totals,
    Variant,
    VaultedCredential,
    Webhook,
)
from carttrace.sandbox.core.trust import POLICY, TIERS

__all__ = [
    "CART",
    "CHECKOUT",
    "CREDENTIAL",
    "DECLINE_CODES",
    "DELEGATED_NAMESPACE",
    "FAKE_HANDLERS",
    "FAKE_NAMESPACE",
    "HANDLER_KINDS",
    "HANDLER_RULES",
    "INITIAL",
    "MACHINES",
    "MARKETS",
    "MUTATING",
    "ORDER",
    "PAYMENT",
    "PAYMENT_REQUEST",
    "POLICY",
    "REQUESTED_NAMESPACE",
    "RESERVATION",
    "TIERS",
    "TOKENS",
    "TYPES",
    "BusinessRule",
    "Cart",
    "Checkout",
    "Command",
    "Config",
    "Conflict",
    "CoreError",
    "Event",
    "FaultRule",
    "Group",
    "HandlerRule",
    "IllegalTransition",
    "InvalidCommand",
    "InvariantViolation",
    "Line",
    "Mandate",
    "Market",
    "NotFound",
    "Order",
    "OrderLine",
    "Payment",
    "PaymentRequest",
    "PickupLocation",
    "PolicyDenied",
    "Reservation",
    "Session",
    "ShippingOption",
    "State",
    "TokenSpec",
    "Totals",
    "Unavailable",
    "UnknownMarket",
    "Variant",
    "VaultedCredential",
    "Webhook",
    "apply",
    "check_invariants",
    "initial_state",
    "market",
    "market_for",
    "outcome",
    "parse",
    "reject",
]

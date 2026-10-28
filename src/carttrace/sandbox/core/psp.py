"""The PSP simulator's test credentials: opaque tokens that decide what the issuer does. Never card numbers.

Why tokens. The repo's rule is that no card number appears anywhere, test cards included. A real PSP's test mode keys
its behaviour on magic card numbers; this one keys it on tokens such as `tok_visa_3ds_challenge`, which a task hands
the agent the way a wallet or a vault would. The token says three things: which card network it stands for, which
handler namespaces may present it (a Google Pay token only through `com.google.pay`), and how the issuer answers:
approve, decline with a reason, or ask for a 3-D Secure challenge that the buyer's bank then passes or fails.

Faults can force the same outcomes on any credential (`payment_decline`, `three_ds` in `carttrace.sandbox.faults`); the
handler in `carttrace.sandbox.core.handlers.payments` consults the faults first and the token second.

The PSP also vaults a credential behind a `vt-` token of its own with an allowance (ACP's delegated payment), and
serves two handlers of its own for the payment protocols: one that takes only vaulted credentials, and one whose every
payment answers a payment request (an MPP challenge, an AP2 cart). `HANDLER_RULES` says which is which.

Shop Pay (D73). A `shop_pay` credential (`tok_shop_pay_...`) is what a persona's Shop Wallet holds. It never pays
directly: the wallet's platform mints a single-use Shop Token from it (`mint_shop_token`), as the spec's Path A has
Shop Pay return one token per checkout (vendor/dev.shopify.shop_pay/2026-04-08/spec.md, "Token Acquisition Path A").
The minted `shop_...` token is the only credential the `dev.shopify.shop_pay` handler charges, and the spec's three
rules are the PSP's declines: single-use (`shop_token_used`), checkout-scoped (`shop_token_checkout_mismatch`) and
time-limited (`shop_token_expired`). A credential's `ttl_ms` keys the token's validity on the logical clock, the way
a card token keys the issuer's answer; `tok_shop_pay_expired` mints a token whose validity has already run out.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from carttrace.sandbox.core.config import (
    DELEGATED_NAMESPACE,
    FAKE_NAMESPACE,
    REQUESTED_NAMESPACE,
    SHOP_PAY_NAMESPACE,
)
from carttrace.sandbox.core.errors import BusinessRule

SHOP_TOKEN_PREFIX = "shop_"  # noqa: S105 - a token prefix, not a password
"""How a minted Shop Token reads (the spec's `shop_abc123...`), beside the PSP's own `vt-` vault tokens."""
SHOP_TOKEN_TTL_MS = 15 * 60 * 1000
"""How long a minted Shop Token stays valid, on the logical clock: the spec says tokens are time-limited and should
be used promptly after generation, without naming a period; fifteen minutes is the sandbox's own choice."""


@dataclass(frozen=True, slots=True)
class TokenSpec:
    token: str
    brand: str | None
    """Card network the token stands for: visa, mastercard, amex. None for a Shop Pay wallet credential, whose
    instrument schema carries no brand (the spec's `display` holds wallet metadata, not a network)."""
    kind: str
    """card, google_pay or shop_pay: which handler namespaces take it (`HANDLER_KINDS`)."""
    method: str = "card"
    """The payment method type it pays with, checked against the market's accepted methods."""
    challenge: str | None = None
    """pass or fail when the issuer asks for a 3-D Secure challenge; None for a frictionless payment."""
    decline_code: str | None = None
    """Set when authorization declines, with the issuer's reason."""
    frictionless: bool = False
    """The issuer authenticates the buyer with 3-D Secure without a challenge (EMV 3DS transStatus Y at lookup, D71)."""
    ttl_ms: int | None = None
    """A shop_token credential's minted token validity in logical milliseconds; None takes SHOP_TOKEN_TTL_MS."""


TOKENS: Mapping[str, TokenSpec] = MappingProxyType(
    {
        t.token: t
        for t in (
            TokenSpec("tok_visa_ok", "visa", "card"),
            TokenSpec("tok_visa_decline_insufficient_funds", "visa", "card", decline_code="insufficient_funds"),
            TokenSpec("tok_visa_decline_do_not_honor", "visa", "card", decline_code="do_not_honor"),
            TokenSpec("tok_visa_3ds_challenge", "visa", "card", challenge="pass"),
            TokenSpec("tok_visa_3ds_fail", "visa", "card", challenge="fail"),
            TokenSpec("tok_visa_3ds_frictionless", "visa", "card", frictionless=True),
            TokenSpec("tok_mc_3ds_challenge", "mastercard", "card", challenge="pass"),
            TokenSpec("tok_mc_ok", "mastercard", "card"),
            TokenSpec("tok_amex_ok", "amex", "card"),
            TokenSpec("tok_gpay_ok", "visa", "google_pay"),
            TokenSpec("tok_shopify_card_ok", "visa", "card"),
            TokenSpec("tok_shop_pay_ok", None, "shop_pay", method="shop_pay"),
            TokenSpec("tok_shop_pay_declined", None, "shop_pay", method="shop_pay", decline_code="do_not_honor"),
            TokenSpec("tok_shop_pay_expired", None, "shop_pay", method="shop_pay", ttl_ms=0),
        )
    }
)
"""Every credential the simulator knows. Anything else is refused as an invalid credential."""

HANDLER_KINDS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        FAKE_NAMESPACE: frozenset({"card", "google_pay"}),
        "dev.shopify.card": frozenset({"card"}),
        "com.google.pay": frozenset({"google_pay"}),
        SHOP_PAY_NAMESPACE: frozenset({"shop_pay"}),
    }
)
"""Which token kinds each handler namespace takes. A namespace not listed here (a handler a live store declared that
the simulator has no rules for) takes every test token, so a pack built from a new store still pays."""


@dataclass(frozen=True, slots=True)
class HandlerRule:
    """What a handler namespace demands of a payment beyond a valid credential."""

    vaulted: bool = False
    """Only a credential this PSP vaulted (`vt-...`) may be presented: a raw test token is refused."""
    requested: bool = False
    """The payment must answer a payment request (`PaymentIn.request_id`)."""
    shop: bool = False
    """Only a Shop Token minted for this checkout (`mint_shop_token`) may be presented (dev.shopify.shop_pay, D73):
    a raw wallet credential is refused, and a minted one declines under the spec's three rules."""


HANDLER_RULES: Mapping[str, HandlerRule] = MappingProxyType(
    {
        DELEGATED_NAMESPACE: HandlerRule(vaulted=True),
        REQUESTED_NAMESPACE: HandlerRule(requested=True),
        SHOP_PAY_NAMESPACE: HandlerRule(shop=True),
    }
)
"""The PSP's own handlers for the payment protocols (`EnvPack.handlers` serves them when the profile names one), and
the Shop Pay handler's rule. Any other namespace demands nothing more: a vaulted credential or a request may still be
presented through it."""
VAULT_PREFIX = "vt-"
"""Tokens the PSP issues for vaulted credentials; `stable_id("vt", ...)` makes them."""

DECLINE_CODES = (
    "generic_decline",
    "insufficient_funds",
    "do_not_honor",
    "expired_card",
    "processing_error",
)
"""Reasons an issuer gives for a decline, as faults may name them. Tokens use the same vocabulary."""
SHOP_DECLINE_USED = "shop_token_used"
SHOP_DECLINE_CHECKOUT = "shop_token_checkout_mismatch"
SHOP_DECLINE_EXPIRED = "shop_token_expired"
"""The Shop Token declines (D73), the spec's three token rules in the sandbox's own PSP vocabulary: the spec names
the rules (single-use, checkout-scoped, time-limited) and no codes. Not in DECLINE_CODES: no fault may name them."""
DECLINE_CLASSES: Mapping[str, str] = MappingProxyType(
    {
        "generic_decline": "buyer_fixable",
        "insufficient_funds": "buyer_fixable",
        "do_not_honor": "buyer_fixable",
        "expired_card": "buyer_fixable",
        "processing_error": "retryable",
        SHOP_DECLINE_USED: "buyer_fixable",
        SHOP_DECLINE_CHECKOUT: "buyer_fixable",
        SHOP_DECLINE_EXPIRED: "buyer_fixable",
    }
)
"""Each decline code's class (D66): buyer_fixable when only the buyer can clear it (another card, more funds, the
bank's say; for a Shop Token, a token minted again), retryable when the same credential may pay on another try (a
processing error at the issuer)."""

CHALLENGE_OUTCOMES = ("pass", "fail")
CREDENTIAL_FREE_METHODS = ("cod", "bank_transfer", "cheque")
"""Payment methods that need no credential at checkout: cash on delivery is paid at the door, a bank transfer or a
cheque after the order (WooCommerce's offline gateways). A market takes the latter two only where it lists them."""


def credential_for(namespace: str, token: str | None, method: str | None) -> TokenSpec | None:
    """The token a completion presents, checked against its handler. None when no credential is needed: the fake
    handler without a token pays as it always has, and cash on delivery needs none. Raises `BusinessRule` for a
    missing, unknown or misdirected credential; nothing has changed when it does."""
    if token is None:
        if namespace == FAKE_NAMESPACE or method in CREDENTIAL_FREE_METHODS:
            return None
        raise BusinessRule(
            code="payment_credential_missing",
            message=f"the {namespace} handler needs a payment credential",
            field="payment.token",
            hint="pass the token the shopper's wallet issued as payment.token",
        )
    spec = TOKENS.get(token)
    if spec is None:
        raise BusinessRule(
            code="payment_credential_invalid",
            message="the payment credential is not one this store's payment processor issued",
            field="payment.token",
            hint="use a token the shopper's wallet issued; card numbers are never accepted",
        )
    kinds = HANDLER_KINDS.get(namespace)
    if kinds is not None and spec.kind not in kinds:
        raise BusinessRule(
            code="payment_credential_invalid",
            message=f"a {spec.kind} credential cannot be presented through the {namespace} handler",
            field="payment.token",
            hint="pay through the handler that issued the credential",
        )
    if method is not None and method != spec.method:
        raise BusinessRule(
            code="payment_instrument_mismatch",
            message=f"the credential pays by {spec.method}, but the instrument says {method}",
            field="payment.instrument",
            hint=f"set payment.instrument to {spec.method!r} or leave it out",
        )
    return spec

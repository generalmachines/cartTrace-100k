"""Payment brokers: the processor behind a store's PSP handler, answering in Stripe's, Adyen's, Braintree's or PayPal's
own shapes (D71).

Why shapes and not new behaviour. The sandbox's PSP decides every payment the same way, from the opaque test token
and the scheduled faults (`carttrace.sandbox.core.psp`, `handlers.payments`): approve, decline with a reason, or ask for a
3-D Secure challenge that passes or fails, or authenticate without one (frictionless). What differs between real
processors is how they say it: the object, the status names, the decline vocabulary, how 3-D Secure and SCA show, and
how a retry is made safe. A row on the `ucp_psp` rail names its broker (`policy.psp` in the pack, `psp` in the header),
and the same decision is rendered in that broker's documented test-mode shape, in two places: the agent's decline
details (`processor`, `processor_code`, `processor_message` beside the core's own `decline_code`, which every agent
reads) and the harness's PSP record (`processor`, one broker-shaped answer per payment).

Nothing here is a card number. The shapes carry the processor's own ids, derived from the sandbox's payment id
(`stable_id`), and the opaque token is never copied into them.

Sources, each shape from the broker's public documentation (read 2026-10-14):
- Stripe: the PaymentIntent object (https://docs.stripe.com/api/payment_intents/object) with its latest charge
  expanded (https://docs.stripe.com/api/charges/object: `outcome`, `payment_method_details.card.three_d_secure`
  with `authentication_flow` challenge or frictionless and `result`); decline codes
  (https://docs.stripe.com/declines/codes); 3-D Secure (https://docs.stripe.com/payments/3d-secure: status
  `requires_action` with `next_action`, and `payment_intent_authentication_failure` back to `requires_payment_method`);
  idempotency (https://docs.stripe.com/api/idempotent_requests: the `Idempotency-Key` header, a replay answered with
  `Idempotent-Replayed: true`).
- Adyen: the Checkout API /payments response (https://docs.adyen.com/api-explorer/Checkout/71/post/payments:
  `pspReference`, `resultCode`, `refusalReason`, `refusalReasonCode`, `action`, `additionalData`); result codes
  (https://docs.adyen.com/online-payments/payment-result-codes: Authorised, Refused, ChallengeShopper, Cancelled);
  refusal reasons (https://docs.adyen.com/development-resources/refusal-reasons: 2 Refused, 4 Acquirer Error, 6
  Expired Card, 11 3D Not Authenticated, 12 Not enough balance, 16 Shopper Cancelled); 3-D Secure 2
  (https://docs.adyen.com/online-payments/3d-secure/native-3ds2: action type threeDS2, subtype challenge);
  idempotency (https://docs.adyen.com/development-resources/api-idempotency: the `Idempotency-Key` header).
- Braintree: the Transaction response (https://developer.paypal.com/braintree/docs/reference/response/transaction:
  `status`, `processorResponseCode`, `processorResponseText`, `processorResponseType`, `gatewayRejectionReason`,
  `threeDSecureInfo`); authorization responses
  (https://developer.paypal.com/braintree/docs/reference/general/processor-responses/authorization-responses: 1000
  Approved, 2000 Do Not Honor, 2001 Insufficient Funds, 2004 Expired Card, 2046 Declined, 2099 Cardholder
  Authentication Required, 3000 Processor Network Unavailable - Try Again); 3-D Secure
  (https://developer.paypal.com/braintree/docs/guides/3d-secure/server-side: a failed authentication is
  `gateway_rejected` with reason `three_d_secure`). Braintree takes no idempotency key; it guards a retry with
  duplicate transaction checking (https://developer.paypal.com/braintree/articles/control-panel/transactions/
  duplicate-checking), which the record names.
- PayPal: Orders v2 (https://developer.paypal.com/docs/api/orders/v2/: order `status` COMPLETED or
  PAYER_ACTION_REQUIRED with a `payer-action` link, the capture's `status` and `processor_response.response_code`,
  whose enum is checked against PayPal's published OpenAPI document, checkout_orders_v2.json); the codes used are
  0000 APPROVED, 0500 DO_NOT_HONOR, 5100 GENERIC_DECLINE, 5120 INSUFFICIENT_FUNDS, 5400 EXPIRED_CARD, 9100
  DECLINED_PLEASE_RETRY and 5650 SCA_REQUIRED; 3-D Secure results
  (https://developer.paypal.com/docs/checkout/advanced/customize/3d-secure/response-parameters/:
  `authentication_result.liability_shift` POSSIBLE, NO or UNKNOWN, `three_d_secure.authentication_status` Y, N, C);
  idempotency (https://developer.paypal.com/api/rest/reference/idempotency/: the `PayPal-Request-Id` header).

The EMV 3-D Secure transaction status (Y authenticated, C challenge required, N not authenticated, U unable) is what
each broker reports under its own name, and `authentication` below is the sandbox's word for the outcome.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Final

from carttrace.core.ids import stable_id
from carttrace.core.money import exponent

AUTHENTICATIONS: Final = (
    "none",
    "frictionless",
    "challenge_passed",
    "challenge_failed",
    "challenge_abandoned",
    "pending",
    "frictionless_declined",
)
"""How 3-D Secure ended for a payment: not asked, passed without a challenge, a challenge passed, failed or abandoned,
or a challenge still waiting for the buyer; or authenticated without a challenge and declined (transStatus N, D111)."""


@dataclass(frozen=True, slots=True)
class Decline:
    """A broker's words for one of the core's decline codes."""

    code: str
    """The broker's own code (Stripe's decline_code, Adyen's refusalReasonCode, Braintree's processor response code,
    PayPal's processor response code)."""
    message: str
    """The broker's documented text for it."""
    extra: Mapping[str, str] = MappingProxyType({})
    """What else the broker says beside the code (Stripe's error `code`, Braintree's decline type, PayPal's name)."""


@dataclass(frozen=True, slots=True)
class Facts:
    """One payment as the core decided it, which every broker renders."""

    payment_id: str
    checkout_id: str
    order_id: str | None
    status: str
    """The core payment status: captured, declined, authentication_failed, requires_action, voided, pending."""
    decline_code: str | None
    authentication: str
    amount: int
    captured: int
    currency: str
    brand: str | None
    idempotency_key: str | None
    """The key the store's completion carried, which the store passes to its processor."""
    replayed: int
    """How many completions with that key the store answered from its idempotency record."""


@dataclass(frozen=True, slots=True)
class Broker:
    name: str
    title: str
    idempotency: str
    """How a retry is made safe: the header the broker reads, or `duplicate_transaction_check`."""
    declines: Mapping[str, Decline]
    authentication_failed: Decline
    abandoned: Decline
    render: Callable[[Facts], dict[str, Any]]


def _major(amount: int, currency: str) -> str:
    """Minor units as the decimal string Braintree and PayPal take ("12.00")."""
    digits = exponent(currency)
    return str((Decimal(amount) / (Decimal(10) ** digits)).quantize(Decimal(1).scaleb(-digits)))


def _ref(prefix: str, facts: Facts, kind: str, length: int = 24) -> str:
    return stable_id(prefix, facts.payment_id, kind, length=length).replace("-", "_", 1)


def _emv(authentication: str) -> tuple[str | None, str | None]:
    """The EMV 3-D Secure transStatus at lookup and after the challenge for an outcome."""
    return {
        "none": (None, None),
        "frictionless": ("Y", None),
        "challenge_passed": ("C", "Y"),
        "challenge_failed": ("C", "N"),
        "challenge_abandoned": ("C", "U"),
        "pending": ("C", None),
        "frictionless_declined": ("N", None),
    }[authentication]


# -- Stripe ------------------------------------------------------------------------------------------------------------

STRIPE_DECLINES: Mapping[str, Decline] = MappingProxyType(
    {
        "insufficient_funds": Decline(
            "insufficient_funds", "Your card has insufficient funds.", MappingProxyType({"code": "card_declined"})
        ),
        "do_not_honor": Decline("do_not_honor", "Your card was declined.", MappingProxyType({"code": "card_declined"})),
        "generic_decline": Decline(
            "generic_decline", "Your card was declined.", MappingProxyType({"code": "card_declined"})
        ),
        "expired_card": Decline("expired_card", "Your card has expired.", MappingProxyType({"code": "expired_card"})),
        "processing_error": Decline(
            "processing_error",
            "An error occurred while processing your card. Try again in a little bit.",
            MappingProxyType({"code": "processing_error"}),
        ),
    }
)


def _stripe(f: Facts) -> dict[str, Any]:
    status = {
        "captured": "succeeded",
        "requires_action": "requires_action",
        "pending": "processing",
        "voided": "canceled",
    }.get(f.status, "requires_payment_method")
    error: dict[str, Any] | None = None
    if f.status == "declined":
        d = STRIPE.declines.get(f.decline_code or "", STRIPE.declines["generic_decline"])
        error = {"type": "card_error", "code": d.extra["code"], "decline_code": d.code, "message": d.message}
    elif f.status == "authentication_failed":
        d = STRIPE.abandoned if f.authentication == "challenge_abandoned" else STRIPE.authentication_failed
        error = {"type": "invalid_request_error", "code": d.code, "message": d.message}
    charged = f.status in ("captured", "declined")
    charge: dict[str, Any] | None = None
    if charged:
        flow = {"frictionless": "frictionless", "challenge_passed": "challenge"}.get(f.authentication)
        three_ds = (
            {"authentication_flow": flow, "result": "authenticated", "version": "2.2.0"} if flow is not None else None
        )
        declined = f.status == "declined"
        charge = {
            "id": _ref("ch", f, "charge"),
            "object": "charge",
            "amount": f.amount,
            "amount_captured": f.captured,
            "captured": not declined,
            "currency": f.currency.lower(),
            "status": "failed" if declined else "succeeded",
            "failure_code": error["code"] if error is not None else None,
            "outcome": {
                "type": "issuer_declined" if declined else "authorized",
                "network_status": "declined_by_network" if declined else "approved_by_network",
                "reason": error["decline_code"] if error is not None else None,
            },
            "payment_method_details": {
                "type": "card",
                "card": {"brand": f.brand or "unknown", "three_d_secure": three_ds},
            },
        }
    return {
        "id": _ref("pi", f, "intent"),
        "object": "payment_intent",
        "amount": f.amount,
        "amount_received": f.captured,
        "currency": f.currency.lower(),
        "capture_method": "automatic",
        "confirmation_method": "automatic",
        "metadata": {"checkout_id": f.checkout_id, "order_id": f.order_id},
        "status": status,
        "next_action": {"type": "use_stripe_sdk"} if f.status == "requires_action" else None,
        "last_payment_error": error,
        "latest_charge": charge,
    }


# -- Adyen -------------------------------------------------------------------------------------------------------------

ADYEN_DECLINES: Mapping[str, Decline] = MappingProxyType(
    {
        "insufficient_funds": Decline("12", "Not enough balance"),
        "do_not_honor": Decline("2", "Refused"),
        "generic_decline": Decline("2", "Refused"),
        "expired_card": Decline("6", "Expired Card"),
        "processing_error": Decline("4", "Acquirer Error"),
    }
)


def _adyen(f: Facts) -> dict[str, Any]:
    lookup, after = _emv(f.authentication)
    out: dict[str, Any] = {
        "pspReference": _reference(stable_id("psp", f.payment_id, length=16).removeprefix("psp-")),
        "merchantReference": f.checkout_id,
        "amount": {"value": f.amount, "currency": f.currency},
        "paymentMethod": {"type": "scheme", "brand": f.brand or "unknown"},
    }
    if f.status == "captured":
        out["resultCode"] = "Authorised"
    elif f.status == "requires_action":
        out["resultCode"] = "ChallengeShopper"
        out["action"] = {"type": "threeDS2", "subtype": "challenge", "paymentMethodType": "scheme"}
    elif f.status == "authentication_failed":
        abandoned = f.authentication == "challenge_abandoned"
        d = ADYEN.abandoned if abandoned else ADYEN.authentication_failed
        out.update(
            resultCode="Cancelled" if abandoned else "Refused", refusalReason=d.message, refusalReasonCode=d.code
        )
    elif f.status == "declined":
        d = ADYEN.declines.get(f.decline_code or "", ADYEN.declines["generic_decline"])
        out.update(resultCode="Refused", refusalReason=d.message, refusalReasonCode=d.code)
    elif f.status == "voided":
        out["resultCode"] = "Cancelled"
    else:
        out["resultCode"] = "Pending"
    if lookup is not None:
        authenticated = (after or lookup) == "Y"
        out["additionalData"] = {
            "threeDOffered": "true",
            "threeDAuthenticated": "true" if authenticated else "false",
            "threeDAuthenticatedResponse": after or lookup,
            "threeDSVersion": "2.2.0",
            "liabilityShift": "true" if authenticated else "false",
        }
    return out


# -- Braintree ---------------------------------------------------------------------------------------------------------

BRAINTREE_DECLINES: Mapping[str, Decline] = MappingProxyType(
    {
        "insufficient_funds": Decline("2001", "Insufficient Funds", MappingProxyType({"type": "soft_declined"})),
        "do_not_honor": Decline("2000", "Do Not Honor", MappingProxyType({"type": "soft_declined"})),
        "generic_decline": Decline("2046", "Declined", MappingProxyType({"type": "soft_declined"})),
        "expired_card": Decline("2004", "Expired Card", MappingProxyType({"type": "hard_declined"})),
        "processing_error": Decline(
            "3000", "Processor Network Unavailable - Try Again", MappingProxyType({"type": "soft_declined"})
        ),
    }
)
BRAINTREE_3DS_STATUS: Mapping[str, str] = MappingProxyType(
    {
        "frictionless": "authenticate_successful",
        "challenge_passed": "authenticate_successful",
        "challenge_failed": "authenticate_failed",
        "challenge_abandoned": "authenticate_unable_to_authenticate",
        "pending": "challenge_required",
        "frictionless_declined": "authenticate_failed",
    }
)


def _braintree(f: Facts) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": stable_id("bt", f.payment_id, length=8).removeprefix("bt-"),
        "type": "sale",
        "amount": _major(f.amount, f.currency),
        "currencyIsoCode": f.currency,
        "orderId": f.checkout_id,
        "creditCard": {"cardType": (f.brand or "unknown").title()},
        "gatewayRejectionReason": None,
    }
    if f.status == "captured":
        out.update(
            status="submitted_for_settlement",
            processorResponseCode="1000",
            processorResponseText="Approved",
            processorResponseType="approved",
        )
    elif f.status == "declined":
        d = BRAINTREE.declines.get(f.decline_code or "", BRAINTREE.declines["generic_decline"])
        out.update(
            status="processor_declined",
            processorResponseCode=d.code,
            processorResponseText=d.message,
            processorResponseType=d.extra["type"],
        )
    elif f.status == "authentication_failed":
        out.update(status="gateway_rejected", gatewayRejectionReason="three_d_secure")
    elif f.status == "voided":
        out["status"] = "voided"
    else:
        out["status"] = "authorizing"
    if f.authentication != "none":
        lookup, after = _emv(f.authentication)
        passed = f.authentication in ("frictionless", "challenge_passed")
        info: dict[str, Any] = {
            "status": BRAINTREE_3DS_STATUS[f.authentication],
            "enrolled": "Y",
            "liabilityShiftPossible": True,
            "liabilityShifted": passed,
            "threeDSecureVersion": "2.2.0",
            "lookup": {"transStatus": lookup},
        }
        if after is not None:
            info["authentication"] = {"transStatus": after}
        out["threeDSecureInfo"] = info
    return out


# -- PayPal ------------------------------------------------------------------------------------------------------------

PAYPAL_DECLINES: Mapping[str, Decline] = MappingProxyType(
    {
        "insufficient_funds": Decline("5120", "INSUFFICIENT_FUNDS"),
        "do_not_honor": Decline("0500", "DO_NOT_HONOR"),
        "generic_decline": Decline("5100", "GENERIC_DECLINE"),
        "expired_card": Decline("5400", "EXPIRED_CARD"),
        "processing_error": Decline("9100", "DECLINED_PLEASE_RETRY"),
    }
)
PAYPAL_LIABILITY: Mapping[str, str] = MappingProxyType(
    {
        "frictionless": "POSSIBLE",
        "challenge_passed": "POSSIBLE",
        "challenge_failed": "NO",
        "challenge_abandoned": "UNKNOWN",
        "pending": "UNKNOWN",
        "frictionless_declined": "NO",
    }
)


def _paypal(f: Facts) -> dict[str, Any]:
    value = {"currency_code": f.currency, "value": _major(f.amount, f.currency)}
    card: dict[str, Any] = {"brand": (f.brand or "unknown").upper()}
    if f.authentication != "none":
        lookup, after = _emv(f.authentication)
        card["authentication_result"] = {
            "liability_shift": PAYPAL_LIABILITY[f.authentication],
            "three_d_secure": {"enrollment_status": "Y", "authentication_status": after or lookup},
        }
    order_id = _reference(stable_id("pp", f.payment_id, length=17).removeprefix("pp-"))
    unit: dict[str, Any] = {"reference_id": f.checkout_id, "amount": value}
    out: dict[str, Any] = {"id": order_id, "intent": "CAPTURE", "payment_source": {"card": card}}
    if f.status in ("captured", "declined"):
        declined = f.status == "declined"
        d = PAYPAL.declines.get(f.decline_code or "", PAYPAL.declines["generic_decline"]) if declined else None
        capture = {
            "id": _reference(stable_id("ppc", f.payment_id, length=17).removeprefix("ppc-")),
            "status": "DECLINED" if declined else "COMPLETED",
            "amount": value,
            "final_capture": True,
            "processor_response": {"response_code": d.code if d is not None else "0000"},
        }
        unit["payments"] = {"captures": [capture]}
        out["status"] = "COMPLETED"
        if d is not None:
            out["processor_response_name"] = d.message
    elif f.status == "requires_action":
        out["status"] = "PAYER_ACTION_REQUIRED"
        out["links"] = [{"rel": "payer-action", "method": "GET", "href": f"sandbox://3ds/{f.payment_id}"}]
    elif f.status == "authentication_failed":
        out["status"] = "PAYER_ACTION_REQUIRED"
        out["details"] = [{"issue": "CONTINGENCY_NOT_SUCCESSFUL", "description": PAYPAL.authentication_failed.message}]
    elif f.status == "voided":
        out["status"] = "VOIDED"
    else:
        out["status"] = "APPROVED"
    out["purchase_units"] = [unit]
    return out


STRIPE: Final = Broker(
    "stripe",
    "Stripe",
    "Idempotency-Key",
    STRIPE_DECLINES,
    Decline(
        "payment_intent_authentication_failure",
        "The provided PaymentMethod has failed authentication. You can provide payment_method_data or a new "
        "PaymentMethod to attempt to fulfill this PaymentIntent again.",
    ),
    Decline("authentication_not_handled", "The customer did not complete the required authentication."),
    _stripe,
)
ADYEN: Final = Broker(
    "adyen",
    "Adyen",
    "Idempotency-Key",
    ADYEN_DECLINES,
    Decline("11", "3D Not Authenticated"),
    Decline("16", "Shopper Cancelled"),
    _adyen,
)
BRAINTREE: Final = Broker(
    "braintree",
    "Braintree",
    "duplicate_transaction_check",
    BRAINTREE_DECLINES,
    Decline("three_d_secure", "Gateway Rejected: three_d_secure"),
    Decline("2099", "Cardholder Authentication Required"),
    _braintree,
)
PAYPAL: Final = Broker(
    "paypal",
    "PayPal",
    "PayPal-Request-Id",
    PAYPAL_DECLINES,
    Decline("CONTINGENCY_NOT_SUCCESSFUL", "3D-Secure payer authentication is unsuccessful."),
    Decline("5650", "SCA_REQUIRED"),
    _paypal,
)
BROKERS: Mapping[str, Broker] = MappingProxyType({b.name: b for b in (STRIPE, ADYEN, BRAINTREE, PAYPAL)})
"""Every broker a `ucp_psp` row can name, in the order a plan lists them."""
BROKER_NAMES: Final = tuple(BROKERS)


def broker(name: str) -> Broker:
    found = BROKERS.get(name)
    if found is None:
        raise ValueError(f"no payment broker {name!r}; brokers are {list(BROKERS)}")
    return found


def decline_details(name: str | None, code: str, *, authentication: str | None = None) -> dict[str, str]:
    """What a refusal's details add for the broker behind the PSP: its name, code and text for the core's decline
    code (or for a failed or abandoned 3-D Secure step); nothing when the store names no broker."""
    if name is None or name not in BROKERS:
        return {}
    b = BROKERS[name]
    if authentication == "challenge_abandoned":
        d = b.abandoned
    elif authentication in ("challenge_failed", "frictionless_declined"):
        d = b.authentication_failed
    else:
        d = b.declines.get(code, b.declines["generic_decline"])
    return {"processor": b.name, "processor_code": d.code, "processor_message": d.message}


def processor_answer(name: str, facts: Facts) -> dict[str, Any]:
    """The broker's answer for one payment, with how it guards a retry: the header and key it reads, or its duplicate
    check, and how many completions were answered from the store's idempotency record."""
    b = broker(name)
    header = None if b.idempotency == "duplicate_transaction_check" else b.idempotency
    return {
        "broker": b.name,
        "response": b.render(facts),
        "idempotency": {
            "mechanism": "duplicate_transaction_check" if header is None else "header",
            "header": header,
            "key": facts.idempotency_key,
            "replayed": facts.replayed,
        },
    }


def _reference(digest_text: str) -> str:
    """A processor reference from a hex digest, upper case, always holding a letter: an all-digit 16 or 17 character
    reference can pass Luhn, and the no-card-numbers rule then flags (or redacts) the PSP's own record (the v1
    review's #14: 83 in 200,000 Adyen references were Luhn-valid)."""
    out = digest_text.upper()
    return out if not out.isdigit() else "R" + out[1:]

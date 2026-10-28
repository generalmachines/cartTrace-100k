"""The wallet tools an agent is given on a sandbox checkout row: each payment rail's own, official credential step,
under its official name and schema, served offline by the persona's wallet (decision D94).

Why. Until D94 every rail gave the agent the same four tools we wrote (request_consent, wallet_payment_handlers,
wallet_request_credential, wallet_complete_challenge), so a published trace showed a payment no real agent makes. So each rail
now hands the agent the step its own protocol defines for the agent's side, and the shopper's approval happens where
that step's official flow asks the buyer (a wallet's payment sheet, a delegation the platform confirms, a mandate the
user signs on their device). Nothing reaches a payment provider: the credential is the persona's opaque test token,
and every answer is the official response shape. The research is docs/research/2026-10-16-real-wallets.md.

The kits, by the dialect the store speaks (`kit_for`):
- UCP (ucp, ucp_checkout, webmcp; rails ucp_psp and shop_pay), by payment handler:
  - `tokenize`: UCP's Tokenization API (`operationId: tokenize`, vendor/ucp/2026-08-25/source/handlers/tokenization):
    `{credential, binding: {type: dev.ucp.shopping.checkout, id}}` answered `{token}`; the credential is the wallet's
    saved card, which the platform fills (the agent never holds one);
  - `load_payment_data`: the Google Pay API's PaymentsClient.loadPaymentData(PaymentDataRequest), for a
    `com.google.pay` handler; the payment sheet is where the shopper approves the total; answered PaymentData with
    `tokenizationData {type: PAYMENT_GATEWAY, token}`, or a PaymentsError (`statusCode` CANCELED, ...);
  - `shop_pay_payment_request`: Shop Pay's Token Acquisition Path A (vendor/dev.shopify.shop_pay/2026-04-08/spec.md):
    the payment request is shown in the Shop Pay interface, the buyer confirms, and a single-use `shop_token` comes
    back. Shop Pay publishes the flow but no request schema, so its arguments are ours.
- ACP: `delegate_payment` (POST /agentic_commerce/delegate_payment), the official DelegatePaymentRequest with the
  allowance the agent states; the platform's vault fills `payment_method` and `risk_signals`, as ChatGPT's does, after
  the shopper confirms; answered by the PSP's own `{id: vt_..., created, metadata}` or ACP's Error.
- AP2 v0.1.0: the sample shopping agent's tools (google-agentic-commerce/AP2 v0.1.0 samples, Apache-2.0):
  `get_payment_methods`, `get_payment_credential_token`, `create_payment_mandate` and
  `sign_mandates_on_user_device`, whose user authorization is the shopper's consent.
- AP2 v0.2.0: `create_checkout_mandate` (the user signs `mandate.checkout.1` over the merchant's checkout_jwt, the
  consent) and the credentials provider's MCP `issue_payment_credential`.
- MPP: `create_credential`, the client payment method's `create_credential(challenge)` (pympp 0.11.0
  `mpp.runtime.Method`): the client enabler answers the store's 402 challenge with the MPP Credential
  `{challenge, payload}` once the shopper approves the charge.
- Every rail: `three_ds_challenge`, named for UCP's `dev.ucp.common.payment.three_ds_challenge` Action: the platform
  shows the bank's page to the buyer. How the agent asks for it is ours on every rail.

What stays ours, and is documented as such: the consent turn written into the trace (no protocol defines one; the
judge reads it), a few arguments the official calls carry in their transport (AP2's checkout or cart id, the
tokenizer's handler), the error codes for a shopper who says no, and the 3-D Secure tool's shape.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from carttrace.core.canon import dumps
from carttrace.protocols.tools import ToolResult

TOKENIZE: Final = "tokenize"
GOOGLE_PAY: Final = "load_payment_data"
SHOP_PAY: Final = "shop_pay_payment_request"
DELEGATE: Final = "delegate_payment"
AP2_METHODS: Final = "get_payment_methods"
AP2_TOKEN: Final = "get_payment_credential_token"  # noqa: S105 (a tool name)
AP2_MANDATE: Final = "create_payment_mandate"
AP2_SIGN: Final = "sign_mandates_on_user_device"
AP2_CHECKOUT_MANDATE: Final = "create_checkout_mandate"
AP2_ISSUE: Final = "issue_payment_credential"
MPP_CREDENTIAL: Final = "create_credential"
CHALLENGE: Final = "three_ds_challenge"

WALLET_TOOLS: Final = frozenset(
    {
        TOKENIZE,
        GOOGLE_PAY,
        SHOP_PAY,
        DELEGATE,
        AP2_METHODS,
        AP2_TOKEN,
        AP2_MANDATE,
        AP2_SIGN,
        AP2_CHECKOUT_MANDATE,
        AP2_ISSUE,
        MPP_CREDENTIAL,
        CHALLENGE,
    }
)
"""Every wallet tool name of every kit: reserved beside the store's tools on a checkout row."""
CONSENT_TOOLS: Final = frozenset(
    {TOKENIZE, GOOGLE_PAY, SHOP_PAY, DELEGATE, AP2_SIGN, AP2_CHECKOUT_MANDATE, MPP_CREDENTIAL}
)
"""The tools whose official flow asks the shopper to approve the payment: each writes the consent turn."""
STANDS_IN: Final = frozenset({DELEGATE})
"""Store tools the wallet's own tool of the same name stands in for on a checkout row: ACP's delegate_payment is the
agent platform's call to the PSP, made with the vault's credential, so the agent calls the wallet's and the store's
copy (which would want a card) is not shown."""
CHECKOUT_BINDING: Final = "dev.ucp.shopping.checkout"
"""The capability a UCP token is bound to (common/types/binding.json)."""
MPP_PROBLEMS: Final = "https://paymentauth.org/problems"
"""The base of MPP's problem types (pympp 0.11.0 mpp.errors)."""


@dataclass(frozen=True, slots=True)
class Kit:
    """The wallet tools one rail gives the agent, in the order its flow uses them."""

    name: str
    tools: tuple[str, ...]


KITS: Final[Mapping[str, Kit]] = {
    "ucp": Kit("ucp", (TOKENIZE, GOOGLE_PAY, SHOP_PAY, CHALLENGE)),
    "acp": Kit("acp", (DELEGATE, CHALLENGE)),
    "ap2_v1": Kit("ap2_v1", (AP2_METHODS, AP2_TOKEN, AP2_MANDATE, AP2_SIGN, CHALLENGE)),
    "ap2": Kit("ap2", (AP2_CHECKOUT_MANDATE, AP2_ISSUE, CHALLENGE)),
    "mpp": Kit("mpp", (MPP_CREDENTIAL, CHALLENGE)),
}
_KIT_OF: Final[Mapping[str, str]] = {
    "ucp": "ucp",
    "ucp_checkout": "ucp",
    "webmcp": "ucp",
    "website": "ucp",
    "acp": "acp",
    "ap2": "ap2",
    "ap2_v1": "ap2_v1",
    "mpp": "mpp",
}


def kit_for(dialect: str) -> Kit:
    """The wallet kit of a dialect (UCP's for one this table does not name)."""
    return KITS[_KIT_OF.get(dialect, "ucp")]


# -- definitions -------------------------------------------------------------------------------------------------------

_STRING: Final = {"type": "string"}
_CHECKOUT_ID: Final = {"type": "string", "description": "The checkout's id, exactly as the store returned it."}


def _function(name: str, description: str, properties: Mapping[str, Any], required: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": dict(properties), "required": list(required)},
        },
    }


_AMOUNT_ACP: Final = {
    "type": "object",
    "description": "ACP Allowance: how the delegated card may be used.",
    "properties": {
        "reason": {"type": "string", "enum": ["one_time"]},
        "max_amount": {"type": "integer", "description": "The checkout session's total, in minor units."},
        "currency": {"type": "string", "description": "ISO 4217, lowercase (usd)."},
        "checkout_session_id": {"type": "string"},
        "merchant_id": {"type": "string", "description": "The merchant the session's payment handler names."},
        "expires_at": {"type": "string", "format": "date-time"},
    },
    "required": ["reason", "max_amount", "currency", "checkout_session_id", "merchant_id", "expires_at"],
}
_ADDRESS_ACP: Final = {
    "type": "object",
    "description": "ACP Address (optional).",
    "properties": {k: _STRING for k in ("name", "line_one", "line_two", "city", "state", "country", "postal_code")},
}

DEFINITIONS: Final[Mapping[str, dict[str, Any]]] = {
    TOKENIZE: _function(
        TOKENIZE,
        "UCP Tokenization API (operationId tokenize) of a checkout's card payment handler (dev.shopify.card, or any "
        "handler that takes a card token): convert the shopper's saved card into a token bound to one checkout. "
        "Your platform's wallet supplies the credential; you never see or send card data. Before it tokenizes, the "
        "wallet shows the shopper the checkout and its current total to confirm. Answers {token}. Pay with it in the "
        "store's paying call: a UCP instrument {id, handler_id, type: card, selected: true, credential: {type: token, "
        "token}} in complete_checkout's checkout.payment.instruments, or a storefront's place_order payment_token.",
        {
            "handler_id": {
                "type": "string",
                "description": "The id of the checkout's payment handler whose tokenizer to use, as the checkout's "
                "payment handlers name it.",
            },
            "binding": {
                "type": "object",
                "description": "UCP Binding: the resource the token is bound to.",
                "properties": {
                    "type": {"type": "string", "enum": [CHECKOUT_BINDING]},
                    "id": {"type": "string", "description": "The checkout's id."},
                },
                "required": ["type", "id"],
            },
            "identity": {
                "type": "object",
                "description": "UCP Payment Identity, when tokenizing on behalf of the business (optional).",
                "properties": {"access_token": _STRING},
            },
        },
        ["handler_id", "binding"],
    ),
    GOOGLE_PAY: _function(
        GOOGLE_PAY,
        "Google Pay API PaymentsClient.loadPaymentData, for a checkout's com.google.pay payment handler: shows the "
        "shopper the Google Pay payment sheet for the total you state, and returns PaymentData once they confirm, or "
        "a PaymentsError (statusCode CANCELED when they close the sheet). Build the PaymentDataRequest from the "
        "handler's config (api_version, merchant_info, allowed_payment_methods). Pay with "
        "paymentMethodData.tokenizationData.token as a UCP instrument {id, handler_id, type: card, selected: true, "
        "credential: {type: PAYMENT_GATEWAY, token}}.",
        {
            "apiVersion": {"type": "integer", "enum": [2]},
            "apiVersionMinor": {"type": "integer", "enum": [0]},
            "merchantInfo": {
                "type": "object",
                "properties": {"merchantId": _STRING, "merchantName": _STRING},
            },
            "allowedPaymentMethods": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["CARD"]},
                        "parameters": {"type": "object"},
                        "tokenizationSpecification": {"type": "object"},
                    },
                },
            },
            "transactionInfo": {
                "type": "object",
                "properties": {
                    "totalPriceStatus": {"type": "string", "enum": ["FINAL", "ESTIMATED", "NOT_CURRENTLY_KNOWN"]},
                    "totalPrice": {"type": "string", "description": "The checkout's total in major units (38.39)."},
                    "currencyCode": {"type": "string", "description": "ISO 4217 (USD)."},
                    "countryCode": _STRING,
                    "transactionId": {"type": "string", "description": "The checkout's id."},
                    "checkoutOption": {"type": "string", "enum": ["DEFAULT", "COMPLETE_IMMEDIATE_PURCHASE"]},
                },
                "required": ["totalPriceStatus", "totalPrice", "currencyCode", "transactionId"],
            },
        },
        ["apiVersion", "apiVersionMinor", "allowedPaymentMethods", "transactionInfo"],
    ),
    SHOP_PAY: _function(
        SHOP_PAY,
        "Shop Pay one-time payment request (Token Acquisition Path A of the dev.shopify.shop_pay handler): the "
        "payment request for the checkout (its totals, lines and delivery) is shown to the shopper in the Shop Pay "
        "interface, and once they confirm it a single-use Shop Pay token scoped to this checkout comes back. Pay "
        "with it as a UCP instrument {id, handler_id, type: shop_pay, selected: true, credential: {type: shop_token, "
        "token}}.",
        {
            "handler_id": {"type": "string", "description": "The checkout's Shop Pay handler id (shop_pay)."},
            "shop_id": {"type": "string", "description": "The merchant's shop_id from the handler's config."},
            "checkout_id": _CHECKOUT_ID,
        },
        ["handler_id", "checkout_id"],
    ),
    DELEGATE: _function(
        DELEGATE,
        "ACP Delegate Payment (POST /agentic_commerce/delegate_payment) at the store's payment processor: vault the "
        "shopper's card for one checkout session within an allowance. Your platform's vault supplies payment_method "
        "and risk_signals after the shopper confirms the allowance; you send the allowance (max_amount is the "
        "session's total). Answers the PSP's {id, created, metadata}, or an ACP Error {type, code, message, param}. "
        "Pay with the id in complete_checkout_session's payment_data, as the session's payment handler takes it.",
        {"allowance": _AMOUNT_ACP, "billing_address": _ADDRESS_ACP, "metadata": {"type": "object"}},
        ["allowance"],
    ),
    AP2_METHODS: _function(
        AP2_METHODS,
        "AP2 credentials provider (payment method collector): the shopper's payment methods eligible for the cart, "
        "by alias. Answers {payment_method_aliases: [...]}.",
        {"user_email": {"type": "string", "description": "The shopper's email."}},
        ["user_email"],
    ),
    AP2_TOKEN: _function(
        AP2_TOKEN,
        "AP2 credentials provider: a payment credential token for one of the shopper's payment methods. It pays "
        "only once the shopper has signed the payment mandate that names it (sign_mandates_on_user_device). "
        "Answers {status, token}.",
        {"user_email": _STRING, "payment_method_alias": _STRING},
        ["user_email", "payment_method_alias"],
    ),
    AP2_MANDATE: _function(
        AP2_MANDATE,
        "AP2: create the PaymentMandate for the merchant-signed CartMandate, paying with one payment method. "
        "Answers {ap2.mandates.PaymentMandate: {payment_mandate_contents}}, not yet signed.",
        {"payment_method_alias": _STRING, "user_email": _STRING, "cart_id": {"type": "string"}},
        ["payment_method_alias", "user_email", "cart_id"],
    ),
    AP2_SIGN: _function(
        AP2_SIGN,
        "AP2: show the CartMandate and the PaymentMandate to the shopper on their device; when they confirm, the "
        "device signs both (user_authorization). Then send the payment mandate's token to the merchant with "
        "send_payment_mandate.",
        {"cart_id": {"type": "string"}},
        ["cart_id"],
    ),
    AP2_CHECKOUT_MANDATE: _function(
        AP2_CHECKOUT_MANDATE,
        "AP2 v0.2: show the merchant-signed checkout to the shopper on their device; when they confirm, the device "
        "signs the checkout mandate (mandate.checkout.1 over checkout_jwt and checkout_hash).",
        {"checkout_id": _CHECKOUT_ID},
        ["checkout_id"],
    ),
    AP2_ISSUE: _function(
        AP2_ISSUE,
        "AP2 v0.2 credentials provider (MCP): issue the payment credential for a checkout the shopper signed a "
        "checkout mandate for. Answers {payment_token}. Pay with complete_checkout(checkout_id, token).",
        {
            "checkout_id": _CHECKOUT_ID,
            "checkout_jwt_hash": {"type": "string", "description": "The checkout's checkout_hash."},
            "payment_mandate_chain_id": _STRING,
            "open_checkout_hash": _STRING,
            "payment_nonce": _STRING,
        },
        ["checkout_id", "checkout_jwt_hash"],
    ),
    MPP_CREDENTIAL: _function(
        MPP_CREDENTIAL,
        "MPP client payment method create_credential(challenge): answer the store's HTTP 402 Payment challenge "
        "(from request_payment) with an MPP Credential {challenge, payload}, once the shopper approves the charge. "
        "Then call pay with the challenge's id and payload.token.",
        {"challenge_id": {"type": "string", "description": "The id of the challenge request_payment returned."}},
        ["challenge_id"],
    ),
    CHALLENGE: _function(
        CHALLENGE,
        "Show the bank's 3-D Secure challenge that the checkout carries (UCP's dev.ucp.common.payment."
        "three_ds_challenge Action, a continue_url handoff, an ACP session's authentication step, or an AP2 "
        "merchant's challenge) to the shopper, who answers it with their bank. Then read the checkout again.",
        {"checkout_id": _CHECKOUT_ID, "url": {"type": "string", "description": "The challenge URL, if given."}},
        ["checkout_id"],
    ),
}


def definitions(names: Sequence[str]) -> list[dict[str, Any]]:
    return [copy.deepcopy(DEFINITIONS[n]) for n in names]


# -- official answer shapes ------------------------------------------------------------------------------------------


def ok(data: Mapping[str, Any]) -> ToolResult:
    body = dict(data)
    return ToolResult(content=dumps(body), structured=body)


def _err(code: str, body: Mapping[str, Any]) -> ToolResult:
    data = dict(body)
    return ToolResult(content=dumps(data), is_error=True, error_code=code, structured=data)


def ucp_error(code: str, content: str, severity: str = "recoverable") -> ToolResult:
    """UCP's error response (common/types/error_response.json): status error and its messages."""
    message = {"type": "error", "code": code, "content_type": "plain", "content": content, "severity": severity}
    return _err(code, {"ucp": {"status": "error"}, "messages": [message]})


def google_pay_error(status: str, message: str) -> ToolResult:
    """Google Pay's PaymentsError: statusCode (CANCELED, BUYER_ACCOUNT_ERROR, DEVELOPER_ERROR, ...) and message."""
    return _err(status.lower(), {"statusCode": status, "statusMessage": message})


ACP_ERROR_CODES: Final = ("invalid_card", "duplicate_request", "idempotency_conflict", "too_many_requests")
"""The delegate_payment Error codes every vendored ACP release allows (2025-09-29 to 2026-04-17)."""
ACP_ERROR_TYPES: Final = ("invalid_request", "rate_limit_exceeded", "processing_error", "service_unavailable")


def acp_error(code: str, message: str, *, kind: str = "invalid_request", param: str | None = None) -> ToolResult:
    """ACP's Error (schema.delegate_payment.json): type, code, message and the JSONPath of the field, with `type`
    and `code` from the schema's enums. Our own reason (a shopper who said no, a missing allowance field) has no code
    there, so it leads `message` under `invalid_card`, the enum's one refusal of the payment method, and stays the
    result's `error_code` for the trace (the v1 review's #25)."""
    kind = kind if kind in ACP_ERROR_TYPES else "invalid_request"
    sent = (
        code if code in ACP_ERROR_CODES else ("too_many_requests" if kind == "service_unavailable" else "invalid_card")
    )
    text = message if sent == code else f"{code}: {message}"
    body: dict[str, Any] = {"type": kind, "code": sent, "message": text}
    if param:
        body["param"] = param
    return _err(code, body)


def ap2_error(code: str, description: str) -> ToolResult:
    """An AP2 error as its v0.2.0 receipts write one: status Error, error and error_description."""
    return _err(code, {"status": "Error", "error": code, "error_description": description})


def mpp_problem(
    slug: str, title: str, detail: str, *, status: int = 402, challenge_id: str | None = None
) -> ToolResult:
    """An MPP problem details answer (RFC 9457, as pympp's PaymentError.to_problem_details writes it)."""
    body: dict[str, Any] = {"type": f"{MPP_PROBLEMS}/{slug}", "title": title, "status": status, "detail": detail}
    if challenge_id:
        body["challengeId"] = challenge_id
    return _err(slug, body)

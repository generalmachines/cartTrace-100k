"""AP2 v0.1.0's protocol types, as the sandbox validates them: a field-for-field mirror of the vendored pydantic models.

Why a mirror. AP2 v0.1.0 published no JSON Schemas; its types are the pydantic models in `vendor/ap2/v0.1.0/src/ap2/
types` (mandate.py, payment_request.py, contact_picker.py). Vendored files are spec data and are never imported as
code, so the same models are written here, with the same names, types, defaults and optional fields (and, like the
originals, they ignore unknown fields). `tests/sandbox/surfaces/test_ap2.py` loads the vendored modules in
a test and checks that both produce the same JSON Schema, so a drift is a failing test, not a silent difference.

v0.2.0 published JSON Schemas (`code/sdk/schemas/ap2/*.json`); its mandates and receipts are checked against those
files directly (`carttrace.protocols.registry`).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

CART_MANDATE_DATA_KEY = "ap2.mandates.CartMandate"
INTENT_MANDATE_DATA_KEY = "ap2.mandates.IntentMandate"
PAYMENT_MANDATE_DATA_KEY = "ap2.mandates.PaymentMandate"
"""The data keys AP2 v0.1.0 carries each mandate under (an A2A DataPart's key); the sandbox's REST binding uses the
same keys as its JSON members."""


class ContactAddress(BaseModel):
    city: str | None = None
    country: str | None = None
    dependent_locality: str | None = None
    organization: str | None = None
    phone_number: str | None = None
    postal_code: str | None = None
    recipient: str | None = None
    region: str | None = None
    sorting_code: str | None = None
    address_line: list[str] | None = None


class PaymentCurrencyAmount(BaseModel):
    currency: str = Field(..., description="The three-letter ISO 4217 currency code.")
    value: float = Field(..., description="The monetary value.")


class PaymentItem(BaseModel):
    label: str
    amount: PaymentCurrencyAmount
    pending: bool | None = None
    refund_period: int = 30


class PaymentShippingOption(BaseModel):
    id: str
    label: str
    amount: PaymentCurrencyAmount
    selected: bool | None = False


class PaymentOptions(BaseModel):
    request_payer_name: bool | None = False
    request_payer_email: bool | None = False
    request_payer_phone: bool | None = False
    request_shipping: bool | None = True
    shipping_type: str | None = None


class PaymentMethodData(BaseModel):
    supported_methods: str
    data: dict[str, Any] | None = Field(default_factory=dict)


class PaymentDetailsModifier(BaseModel):
    supported_methods: str
    total: PaymentItem | None = None
    additional_display_items: list[PaymentItem] | None = None
    data: Any | None = None


class PaymentDetailsInit(BaseModel):
    id: str
    display_items: list[PaymentItem]
    shipping_options: list[PaymentShippingOption] | None = None
    modifiers: list[PaymentDetailsModifier] | None = None
    total: PaymentItem


class PaymentRequest(BaseModel):
    method_data: list[PaymentMethodData]
    details: PaymentDetailsInit
    options: PaymentOptions | None = None
    shipping_address: ContactAddress | None = None


class PaymentResponse(BaseModel):
    request_id: str
    method_name: str
    details: dict[str, Any] | None = None
    shipping_address: ContactAddress | None = None
    shipping_option: PaymentShippingOption | None = None
    payer_name: str | None = None
    payer_email: str | None = None
    payer_phone: str | None = None


class IntentMandate(BaseModel):
    user_cart_confirmation_required: bool = True
    natural_language_description: str
    merchants: list[str] | None = None
    skus: list[str] | None = None
    requires_refundability: bool | None = False
    intent_expiry: str


class CartContents(BaseModel):
    id: str
    user_cart_confirmation_required: bool
    payment_request: PaymentRequest
    cart_expiry: str
    merchant_name: str


class CartMandate(BaseModel):
    contents: CartContents
    merchant_authorization: str | None = None


class PaymentMandateContents(BaseModel):
    payment_mandate_id: str
    payment_details_id: str
    payment_details_total: PaymentItem
    payment_response: PaymentResponse
    merchant_agent: str
    timestamp: str
    """Required here, where the vendored model fills it from the wall clock: a server never invents a time."""


class PaymentMandate(BaseModel):
    payment_mandate_contents: PaymentMandateContents
    user_authorization: str | None = None

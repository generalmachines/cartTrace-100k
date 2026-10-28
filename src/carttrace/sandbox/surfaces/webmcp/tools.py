"""The storefront's WebMCP tools, as data: what the replica's pages register, and how.

Why data. One definition per tool feeds three readers that must agree: the page's script, which registers the
imperative tools with `document.modelContext.registerTool()` from this table (`imperative_definitions`); the checkout
forms, whose markup is generated from the declarative tools' fields, so the input schema the browser synthesises from
them is exactly the one written here (`form_schema`); and the tests, which hold the page to both.

The set is what a real storefront would expose to an agent, each a step a shopper takes on its pages:
- read-only, annotated `readOnlyHint`: search_products, get_product, compare_products, get_cart, get_checkout;
- the cart: add_to_cart, update_cart_line, remove_cart_line, apply_discount;
- the checkout: start_checkout, and three declarative forms (set_buyer_contact, set_shipping_address,
  choose_fulfillment) marked with the explainer's `toolname`, `tooldescription`, `toolparamdescription` and
  `toolautosubmit`, so both registration styles are exercised;
- place_order, annotated `consequentialHint`: it pays. It stays imperative because only the imperative API can carry
  annotations, and an agent's harness may ask the shopper before a consequential call.

Descriptions follow the draft's best practices (README, "Best Practices"): what the tool does and what it needs, in
positive terms, prices in minor units, plain enum values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal

Style = Literal["imperative", "declarative"]
FieldKind = Literal["text", "email", "tel", "select"]

READ_ONLY: Mapping[str, bool] = MappingProxyType({"readOnlyHint": True})
CONSEQUENTIAL: Mapping[str, bool] = MappingProxyType({"consequentialHint": True})
MINOR_UNITS = "in minor units of the store's currency (2500 is 25.00 USD)"


@dataclass(frozen=True, slots=True)
class FormField:
    """One control of a declarative tool's form."""

    name: str
    label: str
    description: str
    kind: FieldKind = "text"
    required: bool = False
    max_length: int | None = None
    pattern: str | None = None
    """An HTML `pattern`, matched against the whole value."""
    options: tuple[tuple[str, str], ...] = ()
    """(value, label) of a select's options; filled from the store for the fulfillment form."""


@dataclass(frozen=True, slots=True)
class ToolDef:
    name: str
    title: str
    description: str
    style: Style
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    """The imperative tools' JSON Schema. Declarative tools have theirs from their form (`form_schema`)."""
    annotations: Mapping[str, bool] | None = None
    fields: tuple[FormField, ...] = ()


def _obj(properties: Mapping[str, Any], required: Sequence[str] = ()) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "object", "properties": dict(properties), "additionalProperties": False}
    if required:
        out["required"] = list(required)
    return out


ID = {"type": "string", "minLength": 1, "maxLength": 200}
QUERY_MAX = 256
"""The longest query the search box takes: what real storefront search boxes take (Shopify's and BigCommerce's
search inputs hold 255 or 256 characters). At 200 a shopper's quoted Lazada title (up to 250 characters) was refused
"query must satisfy maxLength 200" (D119: 160 refusals in the 2026-10-17 traces)."""
CURSOR_MAX = 64
"""The longest `cursor` the tool takes. The storefront hands out its own short page token for an upstream cursor
longer than this (`StorefrontOperations.page_token`), so a next_cursor always fits: Shopify-shaped packs' base64
cursors (30 characters) were refused "cursor must satisfy maxLength 20" by the tool's own schema (D119: 430 times)."""

SEARCH = ToolDef(
    name="search_products",
    title="Search products",
    description=(
        "Searches this store's catalog by keywords and shows the matching products: each with its id, title, "
        f"description, price range and variants (prices {MINOR_UNITS}, availability, options) and the details the "
        "store states. Narrow by price or category if you like; pass next_cursor from a result to see the next page."
    ),
    style="imperative",
    input_schema=_obj(
        {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": QUERY_MAX,
                "description": f"Words to search for, up to {QUERY_MAX} characters.",
            },
            "min_price": {"type": "integer", "minimum": 0, "description": f"Lowest price, {MINOR_UNITS}."},
            "max_price": {"type": "integer", "minimum": 0, "description": f"Highest price, {MINOR_UNITS}."},
            "category": {"type": "string", "maxLength": 200, "description": "A category to search within."},
            "cursor": {"type": "string", "maxLength": CURSOR_MAX, "description": "next_cursor from an earlier page."},
        },
        ["query"],
    ),
    annotations=READ_ONLY,
)
PRODUCT = ToolDef(
    name="get_product",
    title="Open a product",
    description=(
        "Opens a product's page and shows everything the store states about it: its description and details (such "
        f"as size chart, material, care, ingredients), its options, and its variants with prices {MINOR_UNITS} and "
        'availability. Pass options such as {"Size": "M"} to see which variants have those values.'
    ),
    style="imperative",
    input_schema=_obj(
        {
            "product_id": {**ID, "description": "A product id, or a variant id, from search results."},
            "options": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": 'Option name to the value wanted, for example {"Color": "Black", "Size": "M"}.',
            },
        },
        ["product_id"],
    ),
    annotations=READ_ONLY,
)
COMPARE = ToolDef(
    name="compare_products",
    title="Compare products",
    description=(
        "Shows 2 to 4 products side by side on the attributes they share: price range, options such as size, and "
        "the details the store states for them. Attributes only one of them states are listed after the table."
    ),
    style="imperative",
    input_schema=_obj(
        {
            "product_ids": {
                "type": "array",
                "items": ID,
                "minItems": 2,
                "maxItems": 4,
                "uniqueItems": True,
                "description": "The products to compare, by product id.",
            }
        },
        ["product_ids"],
    ),
    annotations=READ_ONLY,
)
ADD = ToolDef(
    name="add_to_cart",
    title="Add to cart",
    description=(
        "Adds a variant to the shopping cart (it starts one if there is none) and shows the cart: its lines with "
        f"their ids, quantities and prices {MINOR_UNITS}, and its totals."
    ),
    style="imperative",
    input_schema=_obj(
        {
            "variant_id": {**ID, "description": "The variant to add, from a product's variants."},
            "quantity": {"type": "integer", "minimum": 1, "maximum": 99, "description": "How many. Default 1."},
        },
        ["variant_id"],
    ),
)
UPDATE_LINE = ToolDef(
    name="update_cart_line",
    title="Change a quantity",
    description="Sets the quantity of one cart line, by the line id the cart shows, and shows the cart.",
    style="imperative",
    input_schema=_obj(
        {
            "line_id": {**ID, "description": "The cart line's id."},
            "quantity": {"type": "integer", "minimum": 1, "maximum": 99, "description": "The new quantity."},
        },
        ["line_id", "quantity"],
    ),
)
REMOVE_LINE = ToolDef(
    name="remove_cart_line",
    title="Remove from cart",
    description="Removes one line from the cart, by the line id the cart shows, and shows the cart.",
    style="imperative",
    input_schema=_obj({"line_id": {**ID, "description": "The cart line's id."}}, ["line_id"]),
)
CART = ToolDef(
    name="get_cart",
    title="View cart",
    description=f"Shows the shopping cart: its id, lines (line id, variant, quantity, price {MINOR_UNITS}) and totals.",
    style="imperative",
    input_schema=_obj({}),
    annotations=READ_ONLY,
)
DISCOUNT = ToolDef(
    name="apply_discount",
    title="Apply a discount code",
    description=(
        "Applies a discount code to the checkout, or to the cart before checkout has started, and shows the new "
        "totals, or why the code was not accepted."
    ),
    style="imperative",
    input_schema=_obj(
        {"code": {"type": "string", "minLength": 1, "maxLength": 64, "description": "The code."}}, ["code"]
    ),
)
START = ToolDef(
    name="start_checkout",
    title="Check out",
    description=(
        "Starts checkout for the cart and shows the checkout: its lines, totals so far, and what is still missing "
        "before the order can be placed (contact details, a shipping address, a delivery choice)."
    ),
    style="imperative",
    input_schema=_obj({}),
)
CONTACT = ToolDef(
    name="set_buyer_contact",
    title="Contact details",
    description="Fills in and submits the checkout's contact form: the buyer's email, and their name and phone number.",
    style="declarative",
    fields=(
        FormField("name", "Full name", "The buyer's full name, if they give one.", max_length=200),
        FormField("email", "Email", "The buyer's email address, for the order confirmation.", "email", True, 254),
        FormField("phone", "Phone", "The buyer's phone number, with its country code.", "tel", max_length=40),
    ),
)
ADDRESS = ToolDef(
    name="set_shipping_address",
    title="Shipping address",
    description=(
        "Fills in and submits the checkout's shipping address form. The checkout then shows the delivery options "
        "for that address with their prices."
    ),
    style="declarative",
    fields=(
        FormField("first_name", "First name", "The recipient's first name.", max_length=200),
        FormField("last_name", "Last name", "The recipient's last name.", max_length=200),
        FormField("street_address", "Street address", "Street and number.", required=True, max_length=500),
        FormField("extended_address", "Apartment, suite", "Apartment, suite or unit, if any.", max_length=500),
        FormField("address_locality", "City", "City or town.", required=True, max_length=200),
        FormField("address_region", "State or region", "State, province or region, if the country has them.",
                  max_length=200),
        FormField("postal_code", "Postal code", "Postal or ZIP code.", required=True, max_length=40),
        FormField("address_country", "Country", "Two-letter ISO country code, for example US.", required=True,
                  max_length=2, pattern="[A-Za-z]{2}"),
    ),
)  # fmt: skip
FULFILLMENT = ToolDef(
    name="choose_fulfillment",
    title="Delivery",
    description=(
        "Fills in and submits the checkout's delivery form: ship the order (choose a shipping option) or collect it "
        "at one of the store's pickup locations."
    ),
    style="declarative",
    fields=(
        FormField("method", "Delivery method", "shipping, or pickup at a store location.", "select", True),
        FormField("option_id", "Shipping option", "The shipping option, when the method is shipping.", "select"),
        FormField("location_id", "Pickup location", "The pickup location, when the method is pickup.", "select"),
    ),
)
CHECKOUT = ToolDef(
    name="get_checkout",
    title="View checkout",
    description=(
        f"Shows the checkout: its status, lines, totals {MINOR_UNITS}, contact details, address, delivery choice, "
        "and what is still missing before the order can be placed, with the tool that fills each in."
    ),
    style="imperative",
    input_schema=_obj({}),
    annotations=READ_ONLY,
)
PLACE = ToolDef(
    name="place_order",
    title="Place order",
    description=(
        "Pays for the checkout and places the order, with a payment token from the buyer's wallet. If the buyer's "
        "bank asks them to confirm (3-D Secure), the page shows the bank's step to the buyer and waits for it. Shows "
        "the order confirmation, or why the payment did not go through."
    ),
    style="imperative",
    input_schema=_obj(
        {
            "payment_token": {
                "type": "string",
                "minLength": 5,
                "maxLength": 64,
                "description": "The payment token the buyer's wallet gave for this purchase.",
            },
            "payment_handler": {"type": "string", "description": "The payment method to pay with."},
        },
        ["payment_token"],
    ),
    annotations=CONSEQUENTIAL,
)

TOOLS: tuple[ToolDef, ...] = (
    SEARCH,
    PRODUCT,
    COMPARE,
    ADD,
    UPDATE_LINE,
    REMOVE_LINE,
    CART,
    DISCOUNT,
    START,
    CONTACT,
    ADDRESS,
    FULFILLMENT,
    CHECKOUT,
    PLACE,
)
BY_NAME: Mapping[str, ToolDef] = MappingProxyType({t.name: t for t in TOOLS})
DECLARATIVE: tuple[ToolDef, ...] = tuple(t for t in TOOLS if t.style == "declarative")


def form_schema(tool: ToolDef) -> dict[str, Any]:
    """The input schema the shim synthesises from a declarative tool's form (`carttrace/sandbox/browser/js`): strings, a
    select's values as `enum` with `anyOf` const/title pairs, `required`, and each field's description."""
    properties: dict[str, Any] = {}
    for f in tool.fields:
        prop: dict[str, Any] = {"type": "string"}
        if f.kind == "select" and f.options:
            prop["anyOf"] = [{"type": "string", "const": value, "title": label} for value, label in f.options]
            prop["enum"] = [value for value, _ in f.options]
        else:
            if f.required:
                prop["minLength"] = 1  # the form marks required text fields minlength="1"
            if f.max_length is not None:
                prop["maxLength"] = f.max_length
            if f.pattern is not None:
                prop["pattern"] = f"^(?:{f.pattern})$"
        prop["description"] = f.description
        properties[f.name] = prop
    out: dict[str, Any] = {"type": "object", "properties": properties}
    required = [f.name for f in tool.fields if f.required]
    if required:
        out["required"] = required
    return out


def with_choices(tool: ToolDef, choices: Mapping[str, Sequence[tuple[str, str]]]) -> ToolDef:
    """A declarative tool whose selects offer the store's own choices (field name to (value, label) pairs). A select
    the store has no choices for is left out of the form (a market without pickup locations has no location field)."""
    fields = []
    for f in tool.fields:
        offered = choices.get(f.name) if f.kind == "select" else None
        if offered is None:
            fields.append(f)
        elif offered:
            fields.append(replace(f, options=tuple(offered)))
    return replace(tool, fields=tuple(fields))


def imperative_definitions(handlers: Sequence[str]) -> list[dict[str, Any]]:
    """The registerTool() dictionaries the page registers (without `execute`, which the page's script adds), with
    place_order's payment methods named from the store's handlers."""
    out = []
    for t in TOOLS:
        if t.style != "imperative":
            continue
        schema = dict(t.input_schema)
        if t.name == PLACE.name and handlers:
            props = dict(schema["properties"])
            props["payment_handler"] = {
                **props["payment_handler"],
                "enum": list(handlers),
                "description": f"The payment method to pay with. Default {handlers[0]}.",
            }
            schema["properties"] = props
        entry: dict[str, Any] = {"name": t.name, "title": t.title, "description": t.description, "inputSchema": schema}
        if t.annotations is not None:
            entry["annotations"] = dict(t.annotations)
        out.append(entry)
    return out

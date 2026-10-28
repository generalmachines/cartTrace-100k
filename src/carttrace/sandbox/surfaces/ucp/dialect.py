"""Per-release UCP dialects: what a sandbox document looks like at one UCP version, as data.

Why data. A live store speaks one UCP release, and the same journey must be answered the way that release answers it
(docs/ARCHITECTURE.md, "Protocol versions"). Between 2026-04-08 and 2026-08-25 most of the wire is the same shape at a
different `$id`; what does change is small and nameable: shipping destinations gained a `type` discriminator, a
fulfillment option's `description` became a description object, the error response moved to `common/`, and 2026-08-25
added the payment authentication extension, under which a 3-D Secure challenge is an Action on a
`complete_in_progress` checkout instead of a buyer handoff at `continue_url`. Each of those is a field below. Renderers
read the flags; code branches only where semantics differ (the 3-D Secure flow).

The checkout-only releases (D71). 2026-01-11 and 2026-01-23 define checkout and no cart, catalog or order operation:
a checkout is made from line items whose `item.id` "should match the id provided in the id field in the product feed",
and the order a completion places comes back inside the checkout (`order`: id and permalink). So at these releases the
sandbox serves the checkout tools of the release itself, in its own shapes, and the store's product feed through the
catalog tools of the first release that defines them (`feed_version`, 2026-04-08), listed beside them and answered as
that release answers. What else differs is again a field: 2026-01-11 lists active capabilities as an array of
`{name, version}` and carries the payment handlers inside the checkout's `payment.handlers`, takes no request `meta`,
and puts a completion's `payment` beside its `id`; both releases total a discount as a positive amount and define no
error response, so a refusal is answered with the release's envelope and its messages alone.

A release without a dialect is refused when an app is built, naming the releases that are served.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from carttrace.sandbox.surfaces.errors import UnsupportedVersion

CHECKOUT = "dev.ucp.shopping.checkout"
CART = "dev.ucp.shopping.cart"
ORDER = "dev.ucp.shopping.order"
SEARCH = "dev.ucp.shopping.catalog.search"
LOOKUP = "dev.ucp.shopping.catalog.lookup"
FULFILLMENT = "dev.ucp.shopping.fulfillment"
DISCOUNT = "dev.ucp.shopping.discount"
PAYMENT_AUTHENTICATION = "dev.ucp.common.payment.authentication"
THREE_DS_ACTION = "dev.ucp.common.payment.three_ds_challenge"
SHOPPING_SERVICE = "dev.ucp.shopping"


@dataclass(frozen=True, slots=True)
class CapabilityDoc:
    """Where a capability is specified, and which capabilities it extends (UCP overview, "Extensions")."""

    spec: str
    """Path under https://ucp.dev/<version>/specification/."""
    extends: tuple[str, ...] = ()


CAPABILITY_DOCS: Mapping[str, CapabilityDoc] = MappingProxyType(
    {
        CHECKOUT: CapabilityDoc("shopping/checkout"),
        CART: CapabilityDoc("shopping/cart"),
        ORDER: CapabilityDoc("shopping/order"),
        SEARCH: CapabilityDoc("shopping/catalog/search"),
        LOOKUP: CapabilityDoc("shopping/catalog/lookup"),
        FULFILLMENT: CapabilityDoc("shopping/extensions/fulfillment", (CHECKOUT,)),
        DISCOUNT: CapabilityDoc("shopping/extensions/discount", (CHECKOUT, CART)),
        PAYMENT_AUTHENTICATION: CapabilityDoc("payment/extensions/authentication", (CHECKOUT,)),
    }
)
"""The capabilities the sandbox's UCP surfaces implement, with their spec pages and parents."""

ResponseKind = Literal["cart", "checkout", "order", "search", "lookup"]

RELEVANT: Mapping[ResponseKind, tuple[str, ...]] = MappingProxyType(
    {
        "cart": (CART, DISCOUNT),
        "checkout": (CHECKOUT, FULFILLMENT, DISCOUNT, PAYMENT_AUTHENTICATION),
        "order": (ORDER,),
        "search": (SEARCH,),
        "lookup": (LOOKUP,),
    }
)
"""Which capabilities a response of each kind lists in `ucp.capabilities` (UCP overview, "Response Capability
Selection"): its root capability and the extensions of that root."""


@dataclass(frozen=True, slots=True)
class UcpDialect:
    """One UCP release as the sandbox renders it."""

    version: str
    error_schema: str
    """The release's error response schema, relative to vendor/ucp/<version>/; empty where it defines none."""
    profile_schema: str
    """The release's business discovery profile schema (`/.well-known/ucp`), with its fragment."""
    typed_destinations: bool
    """Fulfillment destinations carry `type` (`shipping_address`, `business_location`) in responses."""
    option_description: Literal["text", "object"]
    """A fulfillment option's `description`: a string, or a description object (`{"plain": ...}`)."""
    pickup_destinations: bool
    """A pickup method lists its locations in `destinations`. At 2026-04-08 a destination is
    `oneOf [shipping_destination, retail_location]`, and a postal address takes any extra field, so every retail
    location also matches the shipping branch and no pickup destination can validate (an upstream defect, pinned by
    a test). The sandbox then omits them and names the locations in an info message instead."""
    payment_actions: bool
    """The release defines the payment authentication extension, so a 3-D Secure challenge can be an Action."""
    mcp_structured: bool
    """The release's MCP binding requires the UCP payload in `result.structuredContent` (2026-08-25,
    `transports/mcp_tool_call.json`), so it is sent even over an MCP version that predates the field (2025-03-26,
    whose results take extra members)."""
    implied: tuple[str, ...]
    """Capabilities the sandbox always serves at this release, whether or not the profile names them: the extensions
    a checkout needs to be completable here (fulfillment) and the ones the core implements (discounts, orders)."""
    defaults: tuple[str, ...]
    """What the sandbox declares when the profile names no capabilities at all."""
    feed_version: str | None = None
    """A checkout-only release: the release whose catalog tools serve the store's product feed beside its checkout
    tools (D71). None where the release defines a catalog of its own."""
    capability_list: bool = False
    """Active capabilities are an array of `{name, version}` (2026-01-11), not a registry keyed by name."""
    handlers_in_payment: bool = False
    """The checkout carries its payment handlers in `payment.handlers` (2026-01-11), not in `ucp.payment_handlers`."""
    top_level_payment: bool = False
    """complete_checkout takes `payment` beside `id` (2026-01-11), not inside `checkout`."""
    signed_discount: bool = True
    """A discount total is negative (2026-04-08 on, `signed_amount`); the checkout-only releases total it positive."""
    order_tool: bool = True
    """The release has get_order; else the order is read back from the completed checkout's `order`."""

    @property
    def checkout_only(self) -> bool:
        return self.feed_version is not None

    def destination(self, kind: Literal["shipping_address", "business_location"]) -> dict[str, str]:
        """The discriminator a destination carries in this release, or nothing where the release has none."""
        return {"type": kind} if self.typed_destinations else {}

    def option_text(self, text: str) -> str | dict[str, str]:
        return {"plain": text} if self.option_description == "object" else text


CHECKOUT_ONLY_DEFAULTS = (CHECKOUT, FULFILLMENT, DISCOUNT)
"""What a checkout-only release declares by default: checkout and the extensions the sandbox implements on it."""
FEED_VERSION = "2026-04-08"
"""The first UCP release with catalog operations, whose tools serve a checkout-only release's product feed."""

DIALECTS: Mapping[str, UcpDialect] = MappingProxyType(
    {
        "2026-01-11": UcpDialect(
            version="2026-01-11",
            error_schema="",
            profile_schema="source/discovery/profile_schema.json",
            typed_destinations=False,
            option_description="text",
            pickup_destinations=False,
            payment_actions=False,
            mcp_structured=False,
            implied=(FULFILLMENT, DISCOUNT),
            defaults=CHECKOUT_ONLY_DEFAULTS,
            feed_version=FEED_VERSION,
            capability_list=True,
            handlers_in_payment=True,
            top_level_payment=True,
            signed_discount=False,
            order_tool=False,
        ),
        "2026-01-23": UcpDialect(
            version="2026-01-23",
            error_schema="",
            profile_schema="source/discovery/profile_schema.json#/$defs/business_profile",
            typed_destinations=False,
            option_description="text",
            pickup_destinations=False,
            payment_actions=False,
            mcp_structured=False,
            implied=(FULFILLMENT, DISCOUNT),
            defaults=CHECKOUT_ONLY_DEFAULTS,
            feed_version=FEED_VERSION,
            signed_discount=False,
            order_tool=False,
        ),
        "2026-04-08": UcpDialect(
            version="2026-04-08",
            error_schema="source/schemas/shopping/types/error_response.json",
            profile_schema="source/discovery/profile_schema.json#/$defs/business_profile",
            typed_destinations=False,
            option_description="text",
            pickup_destinations=False,
            payment_actions=False,
            mcp_structured=False,
            implied=(FULFILLMENT, DISCOUNT, ORDER),
            defaults=(CHECKOUT, CART, SEARCH, LOOKUP, FULFILLMENT, DISCOUNT, ORDER),
        ),
        "2026-08-25": UcpDialect(
            version="2026-08-25",
            error_schema="source/schemas/common/types/error_response.json",
            profile_schema="source/schemas/profile.json#/$defs/business_schema",
            typed_destinations=True,
            option_description="object",
            pickup_destinations=True,
            payment_actions=True,
            mcp_structured=True,
            implied=(FULFILLMENT, DISCOUNT, ORDER),
            defaults=(CHECKOUT, CART, SEARCH, LOOKUP, FULFILLMENT, DISCOUNT, ORDER),
        ),
    }
)
"""Every UCP release the sandbox serves. Adding one is a row here plus the tests that validate its documents."""


def served_versions() -> list[str]:
    return sorted(DIALECTS)


def dialect_for(version: str | None) -> UcpDialect:
    """The dialect of a UCP release, or `UnsupportedVersion` naming the releases that are served."""
    dialect = DIALECTS.get(version or "")
    if dialect is None:
        raise UnsupportedVersion(
            f"the sandbox has no UCP dialect for {version!r}",
            hint=f"serve a profile at one of {served_versions()} (presets ucp-2026-01-11, ucp-2026-01-23, "
            "ucp-2026-04-08, shopify-live-2026-10)",
        )
    return dialect

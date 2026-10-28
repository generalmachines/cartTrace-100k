"""Platform dialects: what a store's commerce platform adds to the protocol on the wire, as data (decision D29).

Why. UCP and MCP fix the shape of a cart or a tool list only so far. The live Shopify stores of the 2026-10-12 pilot
(var/live/delta-01/report.md) all answered alike on everything the specifications leave open: MCP 2025-11-25 with
logging, prompts and resources capabilities and no session id; every tool with an `outputSchema` and a long
instructional description; keyed cart gids and `invalid_cart_id` for a malformed one; `expires_at`, an always-present
`messages` list, `display_text` on line totals and seven policy link types on every cart; an opaque search cursor and
an info message when a price filter is applied. None of that is UCP; all of it is what an agent sees. A platform row
names each of those choices, so the sandbox can answer as the store's platform does, and a pack with no platform keeps
the sandbox's own plain dialect, which the surface tests pin.

Which platform a pack speaks is the pack's platform pack when it names one (`policy.platform_pack`, D42), else its
`store.platform` (a replica built from a capture carries the store's; a twin cut from a catalog says `sandbox`). The
rows are frozen data; code reads a flag and branches only where the behaviour differs. Changing a row changes every
store of that platform at once, which is the point.

Platform packs (D42). a replica should not only show the store as it is, and
not only on Shopify, so the traces teach more than one platform's habits. `with_platform_pack(pack, name)` serves a
store's products and market under another platform's rules: WooCommerce from the reference store's recorded probes
(`woocommerce.py`), a generic Magento-like pack from the Magento code reading (`magento.py`), or the sandbox's plain
dialect (`generic`). The pack keeps its products; what its own platform recorded (the store's ranking of searches,
its tools/list) is not replayed under another platform's rules, since a WooCommerce store neither ranks nor lists
as Shopify does. The core's side of each platform is `carttrace.sandbox.core.config.CORE_RULES`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from carttrace.protocols.profile import ProtocolProfile
from carttrace.sandbox.core.config import clamps_stock
from carttrace.sandbox.envpack.pack import EnvPack
from carttrace.sandbox.platforms.shopify import (
    PlatformToolList,
    ToolDoc,
    market_excludes,
    platform_tool_list,
    tool_docs,
)
from carttrace.sandbox.platforms.spec import PackSpec, pack_spec, pack_specs

__all__ = [
    "BIGCOMMERCE",
    "GENERIC",
    "MAGENTO",
    "MEDUSA",
    "MINOR_UNITS",
    "PLATFORMS",
    "PLATFORM_PACKS",
    "PRESTASHOP",
    "SALEOR",
    "SHOPIFY",
    "VENDURE",
    "WOOCOMMERCE",
    "PackSpec",
    "Platform",
    "PlatformToolList",
    "ToolDoc",
    "market_excludes",
    "pack_spec",
    "pack_specs",
    "platform_named",
    "platform_of",
    "platform_profile",
    "platform_tool_list",
    "served_platform",
    "tool_docs",
    "with_platform_pack",
]

MINOR_UNITS = (
    "Prices in the response are integers in the currency's ISO 4217 minor units, paired with\n"
    'a currency code: {"amount": 600, "currency": "USD"} is $6.00 and {"amount": 2500,\n'
    '"currency": "USD"} is $25.00. Convert to major units before quoting a price to a buyer\n'
    "(divide by 100 for two-decimal currencies such as USD and EUR; zero-decimal currencies\n"
    "such as JPY are already whole units)."
)
"""The paragraph every live Shopify tool description ends with (13 of 13 tools, 10 of 10 stores). Every platform's
descriptions carry it, because an agent that misreads minor units misquotes every price."""


@dataclass(frozen=True, slots=True)
class Platform:
    """One platform's choices, each with the live evidence that set it (SHOPIFY) or the sandbox's plain default."""

    name: str
    default_mcp: str
    """The MCP version served when the profile names none."""
    mcp_capabilities: Mapping[str, Any]
    """What `initialize` advertises."""
    sessions: bool
    """Whether `initialize` issues an `Mcp-Session-Id` that later requests must carry. Live Shopify issues none."""
    notification_status: int
    """The HTTP status a JSON-RPC notification gets: 202 as MCP suggests, 200 as live Shopify answers."""
    notification_body: str | None
    """What travels with it: nothing (202), or the `{}` JSON object live Shopify sends with its 200."""
    text_refusals: bool
    """A call whose arguments fail the tool's schema is refused as a text content block (live Shopify: "Missing
    required arguments: catalog") rather than as a UCP error document."""
    tools_file: str | None
    """A package data file holding the platform's recorded tools/list, listed whole at its own versions by a store
    with no recording of its own (D40) and read for descriptions and output schemas at others (D29); None for the
    binding's."""
    available_filter: bool
    """`catalog.filters.available` is listed (default true) and honoured: only sale-ready items come back."""
    plain_content_type: bool
    """Every error and warning message says `content_type: plain` (live Shopify, all 14 cart messages seen)."""
    keyed_cart_ids: bool
    """Cart ids are `gid://shopify/Cart/<token>?key=<key>` and line ids `gid://shopify/CartLine/<id>?cart=<token>`;
    a malformed id is `invalid_cart_id`, an unknown one `cart_not_found`."""
    cart_expires_days: int | None
    """`expires_at` on every cart, this many days after it was made; None leaves the field out."""
    messages_always: bool
    """`messages` is present (an empty list) on every document, not only when there is something to say."""
    line_display_text: bool
    """Line item totals carry `display_text` as the cart's totals do."""
    links: tuple[tuple[str, str], ...]
    """The policy links every cart and checkout carries, as (type, title)."""
    full_ucp_block: bool
    """The `ucp` block of every response lists the cart and catalog capabilities with their spec and schema links,
    the payment handlers with their configs and the `services` registry, as the discovery document does."""
    discounts_block: bool
    """Cart replies carry a `discounts` block even when no code was applied. Live Shopify never sends one."""
    opaque_cursors: bool
    """Search cursors are opaque base64 tokens rather than decimal offsets."""
    price_filter_note: bool
    """A search with a price filter and no `context.currency` carries the shop-currency info message."""
    clamp_stock: bool
    """A line over the stock is added at the stock with a `merchandise_not_enough_stock` warning instead of refused."""
    buyer_location_warning: bool
    """A line whose product's tags exclude the buyer's market (or whose market cannot be told, no context given) is
    added with a `product_unavailable_in_buyer_location` warning."""
    storefront_search: bool
    """A search the store was never asked ranks as live Shopify's catalog search does, as near as product text can
    tell (carttrace.sandbox.surfaces.ucp.catalog, D39), instead of the sandbox's own word count."""
    delivery_group_gids: bool = False
    """An update_cart whose fulfillment names a delivery group by anything but a `CartDeliveryGroup` gid is refused
    as `invalid_input` with the cart left as it was (live Shopify, `SHOPIFY_EVIDENCE` "delivery_group.invalid")."""
    search_all_words: bool = False
    """Every query word must appear, as a substring and as written, in a product's title or text: no stemming, no
    typo or synonym tolerance (WooCommerce: 'black sandal' finds 0, 'slippers' 0 where 'slipper' finds 4)."""
    search_total_count: bool = False
    """Search pagination carries `total_count` beside `has_next_page` (WooCommerce's UCP plugin)."""
    facets: bool = False
    """A search answers with layered navigation facets and automatic price ranges (Magento, unverified wire shape)."""
    block_extends: bool = False
    """The response `ucp` block names the capability each extension extends (WooCommerce's UCP plugin)."""
    server_info: Mapping[str, str] | None = None
    """The MCP `serverInfo` the platform's server names; None keeps the sandbox's."""
    mcp_instructions: str | None = None
    """The MCP `instructions` its initialize answers with; None sends none."""
    wording: Mapping[str, str] = field(default_factory=dict)
    """The platform's own words for refusals an agent meets (keys `discount.<reason>`, `stock.<case>`,
    `line.<reason>`, `order.status`), formatted with the event's facts; a key it lacks keeps the sandbox's words."""
    order_states: Mapping[str, str] = field(default_factory=dict)
    """The core's order statuses in the platform's own names, said on a completed checkout and an order; empty says
    nothing."""
    checkout_discounts: bool = True
    """Checkouts carry a `discounts` block even when no code was applied (WooCommerce's UCP plugin sends none)."""
    readiness: Mapping[str, tuple[str, str, str]] = field(default_factory=dict)
    """What an incomplete checkout lacks, in the platform's own code, words and field, by what is missing (`email`,
    `fulfillment`, `address`); the field may name the method as `{at}`. A key it lacks keeps UCP's `missing`."""
    supported_versions: bool = False
    """The UCP discovery document lists the older releases the store also serves, and the store's `name`."""
    acp_discovery: bool = False
    """`/.well-known/acp.json` is the platform plugin's document instead of the release's."""
    profile_versions: Mapping[str, str] = field(default_factory=dict)
    """The protocol versions a store of this platform speaks where they differ from the store's own: applied by
    `platform_profile` when a store is served under this platform's pack."""
    sold_out_warning: bool = False
    """A line the store will not sell is answered with a `merchandise_out_of_stock` warning and left out of the cart,
    which is still made (live Shopify, D52); otherwise UCP's `out_of_stock` error at the line."""
    accepts: tuple[str, ...] = ()
    """Payment methods the platform's stores take beyond the market's (WooCommerce's offline gateways)."""
    product_views: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """The product shape the platform's catalog tools answer with, per view (`listing`, `detail`, `lookup`): the
    product, variant, media and option value fields and how many media items. A replica's product the store never
    showed in that view is served in this shape (D55); empty serves a product's document as it is."""
    dialect: str | None = None
    """The platform plugin whose UCP documents the surface answers in, request refusals and reply shapes alike
    (`medusa`: carttrace.sandbox.platforms.medusa, D68); None answers in the sandbox's own documents."""
    unserved: tuple[str, ...] = ()
    """UCP capabilities the platform's plugin does not serve, left out of discovery and every reply (Medusa's plugin
    serves no discount capability)."""
    search_phrase: bool = False
    """A search needs the whole query, case-folded, inside a product's title, description or handle (Medusa's UCP
    plugin: 'black sandal' finds 0 where 'Chelsea boot' finds 5)."""
    search_prefix_words: bool = False
    """Every query word must begin a word of the product's indexed text, variants and option values included
    (PrestaShop's word index, code reading)."""
    search_field_prefix: bool = False
    """Every query word, stemmed, must begin a word of one field: the product's name, a variant's name or the
    description (Vendure's default Postgres search strategy, code reading)."""
    search_whole_words: bool = False
    """Every query word must be a whole word of the product's name, description, variant names or option values
    (Saleor's websearch over the `simple` configuration, recorded: ops/playgrounds/saleor/captures/storefront)."""
    page_default: int | None = None
    """A search page's size when the request names none; None keeps the sandbox's."""
    page_max: int | None = None
    """The largest search page the platform serves; None keeps the sandbox's."""
    evidence: Mapping[str, str] = field(default_factory=dict)
    """Each behaviour of the row and the probe or source that set it."""
    extra: Mapping[str, Any] = field(default_factory=dict)


def frozen(value: Any) -> Any:
    """A spec value as a platform row holds it: mappings read-only, lists as tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: frozen(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(frozen(v) for v in value)
    return value


SEARCH_FLAGS: Mapping[str, str] = MappingProxyType(
    {
        "storefront": "storefront_search",
        "all_words": "search_all_words",
        "phrase": "search_phrase",
        "prefix_words": "search_prefix_words",
        "field_prefix": "search_field_prefix",
        "whole_words": "search_whole_words",
    }
)
"""A spec's search mode as the platform row's flag (`search.mode`)."""


def platform_of(spec: PackSpec) -> Platform:
    """The platform row a pack spec names (data/sandbox/packs/<name>.yaml): every flag a surface reads, from the
    section that holds it. Stock clamping is the core's (`CORE_RULES`), so the row asks the core."""
    mcp, ucp, acp = spec.serves.mcp, spec.serves.ucp, spec.serves.acp
    docs, errors, checkout, search = spec.documents, spec.errors, spec.checkout, spec.search
    pins = {"ucp": ucp.pinned, "mcp": mcp.pinned, "acp": acp.pinned}
    flags = dict.fromkeys(SEARCH_FLAGS.values(), False)
    if search.mode in SEARCH_FLAGS:
        flags[SEARCH_FLAGS[search.mode]] = True
    return Platform(
        name=spec.name,
        default_mcp=mcp.default,
        mcp_capabilities=frozen(mcp.capabilities),
        sessions=mcp.sessions,
        notification_status=mcp.notification_status,
        notification_body=mcp.notification_body,
        text_refusals=mcp.text_refusals,
        tools_file=mcp.tools_file,
        available_filter=search.available_filter,
        plain_content_type=docs.plain_content_type,
        keyed_cart_ids=errors.keyed_ids,
        cart_expires_days=checkout.cart_expires_days,
        messages_always=docs.messages_always,
        line_display_text=docs.line_display_text,
        links=tuple((str(t), str(title)) for t, title in checkout.policy_links),
        full_ucp_block=docs.full_ucp_block,
        discounts_block=checkout.discounts_block,
        opaque_cursors=search.opaque_cursors,
        price_filter_note=search.price_filter_note,
        clamp_stock=clamps_stock(spec.name),
        buyer_location_warning=checkout.buyer_location_warning,
        storefront_search=flags["storefront_search"],
        delivery_group_gids=errors.delivery_group_gids,
        search_all_words=flags["search_all_words"],
        search_total_count=search.total_count,
        facets=search.facets,
        block_extends=docs.block_extends,
        server_info=frozen(mcp.server_info) if mcp.server_info is not None else None,
        mcp_instructions=mcp.instructions,
        wording=MappingProxyType(dict(spec.wording)),
        order_states=MappingProxyType(dict(checkout.order_states)),
        checkout_discounts=checkout.checkout_discounts,
        readiness=MappingProxyType({k: (r.code, r.text, r.path) for k, r in checkout.readiness.items()}),
        supported_versions=ucp.supported_versions,
        acp_discovery=acp.discovery == "plugin" and docs.dialect is None,
        profile_versions=MappingProxyType({k: v for k, v in pins.items() if v is not None}),
        sold_out_warning=checkout.sold_out_warning,
        accepts=tuple(checkout.accepts),
        product_views=frozen(docs.product_views),
        dialect=docs.dialect,
        unserved=tuple(ucp.unserved),
        search_phrase=flags["search_phrase"],
        search_prefix_words=flags["search_prefix_words"],
        search_field_prefix=flags["search_field_prefix"],
        search_whole_words=flags["search_whole_words"],
        page_default=search.page_default,
        page_max=search.page_max,
        evidence=MappingProxyType(spec.evidence()),
    )


GENERIC = platform_of(pack_spec("generic"))
"""The sandbox's own dialect: what every pack spoke before platforms, kept exactly for packs that name none."""
SHOPIFY = platform_of(pack_spec("shopify"))
"""Live Shopify, October 2026 (data/sandbox/packs/shopify.yaml): every flag cites the live bundle that showed it
(D29, D46, D52, D55)."""
WOOCOMMERCE = platform_of(pack_spec("woocommerce"))
"""WooCommerce 11.2.1 with its MCP adapter and UCP/ACP plugins, 2026-10-12 (woocommerce.yaml): every flag cites a
recorded probe under ops/playgrounds/woocommerce/captures."""
MAGENTO = platform_of(pack_spec("magento"))
"""A generic Magento-like pack read in the Magento 2.4.8 source (magento.yaml), never seen on a wire."""
MEDUSA = platform_of(pack_spec("medusa"))
"""Medusa 2.21.2 with the Finance District agentic commerce plugin 1.1.3 (medusa.yaml): the plugin's documents are
its own (`dialect`), every flag cites a recorded probe under ops/playgrounds/medusa/captures."""
SALEOR = platform_of(pack_spec("saleor"))
"""Saleor 3.20.119 (saleor.yaml): the sandbox's plain UCP shapes with Saleor's recorded behaviour behind them."""
VENDURE = platform_of(pack_spec("vendure"))
"""A Vendure-like pack read at Vendure 3.4.0's tag (vendure.yaml), never seen on a wire."""
PRESTASHOP = platform_of(pack_spec("prestashop"))
"""A PrestaShop-like pack read at PrestaShop 8.2.1's tag (prestashop.yaml), never seen on a wire."""
BIGCOMMERCE = platform_of(pack_spec("bigcommerce"))
"""A BigCommerce-like pack at the UCP release its scouted stores serve (bigcommerce.yaml); cart rules unverified."""

PLATFORMS: Mapping[str, Platform] = MappingProxyType(
    {
        GENERIC.name: GENERIC,
        SHOPIFY.name: SHOPIFY,
        WOOCOMMERCE.name: WOOCOMMERCE,
        MAGENTO.name: MAGENTO,
        MEDUSA.name: MEDUSA,
        SALEOR.name: SALEOR,
        VENDURE.name: VENDURE,
        PRESTASHOP.name: PRESTASHOP,
        BIGCOMMERCE.name: BIGCOMMERCE,
    }
)
PLATFORM_PACKS = (
    "shopify",
    "woocommerce",
    "magento",
    "medusa",
    "saleor",
    "vendure",
    "prestashop",
    "bigcommerce",
    "generic",
)
"""The platform packs a manifest's platform axis crosses sandbox rows with (D42), in canonical order."""


def platform_named(name: str | None) -> Platform:
    """The platform row a pack's platform names; the plain sandbox dialect for any other or none."""
    return PLATFORMS.get((name or "").lower(), GENERIC)


def served_platform(pack: EnvPack) -> Platform:
    """The platform a pack is served with: its platform pack, else its store's platform."""
    return platform_named(pack.platform_name)


def with_platform_pack(pack: EnvPack, name: str | None) -> EnvPack:
    """`pack` served under platform pack `name`: the same products and market under that platform's rules. The pack
    itself when `name` is None or is the platform the pack already speaks (a Shopify replica under `shopify`, a twin
    under `generic`), so those rows keep their worlds and episode ids. Otherwise a derived pack naming the platform
    pack, with the store's recorded tool lists left out (its own platform's) and the platform's payment methods added
    to the market's."""
    if name is None:
        return pack
    wanted = _known(name)
    if wanted is served_platform(pack):
        return pack
    market = pack.market()
    accepts = tuple(dict.fromkeys((*market.accepts, *wanted.accepts)))
    changes: dict[str, Any] = {"policy": {"platform_pack": wanted.name}, "tool_lists": []}
    if accepts != market.accepts:
        changes["market"] = replace(market, accepts=accepts)
    return pack.derive(**changes)


def platform_profile(profile: ProtocolProfile, name: str | None) -> ProtocolProfile:
    """The profile a store speaks under platform pack `name`: its own, with the versions the platform's stores speak
    where they differ (WooCommerce: MCP 2025-06-18, ACP 2026-04-17). Unchanged for a platform with none."""
    if name is None:
        return profile
    changes: dict[str, Any] = {k: v for k, v in _known(name).profile_versions.items() if getattr(profile, k) != v}
    if not changes:
        return profile
    if "ucp" in changes and profile.ucp_capabilities:
        changes["ucp_capabilities"] = dict.fromkeys(profile.ucp_capabilities, changes["ucp"])
    said = ", ".join(f"{k.upper()} {v}" for k, v in sorted(changes.items()) if isinstance(v, str))
    note = (
        f"{profile.note}; under the {name} platform pack: {said}" if profile.note else f"{name} platform pack: {said}"
    )
    return profile.with_versions(note, **changes)


def _known(name: str) -> Platform:
    found = PLATFORMS.get(name.lower())
    if found is None:
        raise ValueError(f"no platform pack {name!r}; platform packs are {list(PLATFORM_PACKS)}")
    return found

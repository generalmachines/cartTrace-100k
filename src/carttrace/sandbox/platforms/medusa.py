"""Medusa's wire facts, wording and document shapes, each from the Medusa playground's recorded probes (decision D68).

Why a Medusa pack. The project asked (2026-10-13) for the sandbox to fit every e-commerce platform type, so a trace teaches more than Shopify's and WooCommerce's habits. Medusa is the open-source headless
platform we run: Medusa 2.21.2 with the Finance District agentic commerce plugin 1.1.3 (ops/playgrounds/medusa),
seeded with the same 100-product footwear store as a sandbox pack (ops/playgrounds/catalog). Two recordings set this
pack, cited in `EVIDENCE` by run and file number:
- `store-api`: Medusa's own Store API, 114 probes (storeapi.py): search and filters, cart writes and refusals,
  promotions, shipping by region, the payment collection and session flow, regions and currencies, error shapes,
  and the plugin's own UCP search with the same queries;
- `ucp-2026-08-25` and `acp-2026-01-30`: the plugin's UCP and ACP surfaces end to end (smoke.py).
The plugin's source was read where a recording left a rule open (`P`, the installed package), and each such reading
says so.

What is here. The core's side of Medusa (codes stack and compound, case-sensitive codes, a repeat is a no-op, free
shipping from the item total after discounts with tax, tax by region from the first line and on shipping, a line over
the stock refused whole, one shipping group from the start, the cheapest rate once an address is known) is
`carttrace.sandbox.core.config.CORE_RULES["medusa"]`. This module holds the surface's side: what the plugin refuses before it
runs (`refusal`), and how it shapes every UCP document (`reshape`): its envelope, product, cart, checkout, order and
error shapes, offset pagination and phrase search. The plugin serves UCP over REST only; the sandbox serves the same
documents over its MCP binding, as the WooCommerce pack serves UCP's shopper tools behind WooCommerce's MCP.

What is not ported, with why, is in `NOT_PORTED`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from carttrace.core.money import percent_of
from carttrace.sandbox.core.market import countries_served
from carttrace.sandbox.core.state import Cart, Checkout, Order
from carttrace.sandbox.platforms.spec import pack_spec

SPEC = pack_spec("medusa")
"""The pack's spec (data/sandbox/packs/medusa.yaml): every constant below is read from it."""
CAPTURES = "ops/playgrounds/medusa/captures"
P = "@financedistrict/medusa-plugin-agentic-commerce 1.1.3, .medusa/server/src/"
"""The plugin's installed source, read in the playground's app image (mc-pg-medusa-app:2.21.2-fd1.1.3)."""
MEDUSA_ACP_VERSION = str(SPEC.serves.acp.pinned)
"""The plugin's ACP release (acp-2026-01-30 01)."""
CHECKOUT_TTL_MS = int(SPEC.extra["checkout_ttl_ms"])
"""A checkout session expires six hours after it was made (ucp-2026-08-25 14; P lib/formatters/ucp.js)."""
SEARCH_PAGE = int(SPEC.search.page_default or 20)
SEARCH_MAX = int(SPEC.search.page_max or 100)
"""The plugin's default and largest search page (store-api 106; P api/validation-schemas.js CatalogSearchSchema)."""

EVIDENCE: Mapping[str, str] = MappingProxyType(SPEC.evidence())
"""Each ported behaviour and the recorded probe (or plugin source) that set it, captures under CAPTURES."""

NOT_PORTED: Mapping[str, str] = MappingProxyType({k: v.what for k, v in SPEC.not_ported.items()})

WORDING: Mapping[str, str] = MappingProxyType(dict(SPEC.wording))
"""Medusa's own words (store-api 30, 31, 37, 44, 46)."""

READINESS: Mapping[str, tuple[str, str, str]] = MappingProxyType(
    {k: (r.code, r.text, r.path) for k, r in SPEC.checkout.readiness.items()}
)
"""An incomplete checkout's messages as the plugin words them (ucp-2026-08-25 14; P lib/formatters/ucp.js)."""

ORDER_STATES: Mapping[str, str] = MappingProxyType(dict(SPEC.checkout.order_states))
"""The core's order statuses in Medusa's (ucp-2026-08-25 21, store-api 112)."""

ENVELOPE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {k: tuple(v) for k, v in SPEC.extra["envelope_capabilities"].items()}
)
"""What every plugin reply's envelope lists, in its order, by served release (`*` for the rest): six capabilities
at 2026-04-08 and 2026-08-25 (ucp-2026-08-25 12), checkout, order and fulfillment at 2026-01-23 (ucp-2026-01-23 09)."""
ENVELOPE_CAPABILITIES = ENVELOPE["*"]
CAPABILITY_LIST = frozenset(SPEC.extra["capability_list_releases"])
"""Releases whose envelope lists capabilities as `{name, version}` entries (2026-01-11's shape)."""
LINE_TOTAL: Mapping[str, str] = MappingProxyType(dict(SPEC.extra["line_total_type"]))
"""The type of a line's one total by served release: `line_total` from 2026-04-08, `subtotal` at 2026-01-23
(ucp-2026-01-23 09), whose total types have no `line_total`."""
UNSERVED = tuple(SPEC.serves.ucp.unserved)
"""Capabilities the plugin does not serve (ucp-2026-08-25 01)."""
UNSERVED_FIELDS = SPEC.refused_fields()
"""Body fields of the capabilities in `UNSERVED`, refused whatever the release defines (ucp-2026-08-25 18)."""

ALLOWED: Mapping[str, frozenset[str]] = MappingProxyType(
    {k: frozenset(v) for k, v in SPEC.extra["validator_fields"].items()}
)
"""The top-level fields each plugin body takes (P api/validation-schemas.js); Medusa's validator refuses others."""
IGNORED_ARGUMENTS = frozenset(SPEC.extra["ignored_arguments"])
"""Transport parameters the sandbox's binding adds beside the body: the path id and the request headers."""

ACP_DISCOVERY: Mapping[str, Any] = MappingProxyType(dict(SPEC.extra["acp_discovery"]))
"""The plugin's ACP discovery document (acp-2026-01-30 01), its version, base URL and currency filled when served;
the recording lists `eur`, the plugin's default, for a USD store, which the pack does not copy."""


@dataclass(frozen=True, slots=True)
class Wire:
    """A reply as the plugin sends it: an HTTP status and a body."""

    status: int
    body: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Facts:
    """What shaping a document needs beyond the document: the served release, the store's base URL and market, and
    the resource it shows."""

    version: str
    base: str
    tax_bp: int
    tax_on_shipping: bool = True
    country: str | None = None
    store: str = ""
    cart: Cart | None = None
    checkout: Checkout | None = None
    order: Order | None = None
    request: Mapping[str, Any] | None = None
    """The call's payload as the catalog or the core read it: a search's offset and limit, a write's line items."""


def error_doc(
    version: str, code: str, content: str, *, severity: str = "unrecoverable", path: str | None = None
) -> dict[str, Any]:
    """An error as the plugin's ucpErrorFor writes it (P lib/formatters/error-formatters.js)."""
    message: dict[str, Any] = {"type": "error", "code": code, "content": content, "severity": severity}
    if path is not None:
        message["path"] = path
    return {"ucp": {"version": version, "status": "error"}, "messages": [message]}


def invalid_data(text: str) -> Wire:
    """Medusa's request validator's refusal (ucp-2026-08-25 18, store-api 22, 36): its own shape, not UCP's."""
    return Wire(400, {"type": "invalid_data", "message": f"Invalid request: {text}"})


def missing_agent(version: str) -> Wire:
    """No UCP-Agent header (ucp-2026-08-25 05)."""
    return Wire(400, error_doc(version, "missing_ucp_agent", "Missing UCP-Agent header for platform identification"))


def refusal(
    tool: str,
    arguments: Mapping[str, Any],
    payload_key: str | None,
    *,
    version: str,
    market: str,
    served: frozenset[str] = frozenset(),
    currency: str | None = None,
) -> Wire | None:
    """What the plugin refuses before it runs: a body field neither its schema nor the served UCP release defines
    (`discounts`, whose capability it does not serve), a create priced in a currency the store does not sell in,
    and an address in a country no region serves. None lets the call run.

    `served` is what the served release defines for this body: its request schema's fields and the fields of the
    extensions the plugin serves (`accepted`). The plugin's own validator is stricter (`fields.strict`: it refuses
    `context` on a search or an update and `cart_id` on a create), but the sandbox lists the release's schema to the
    agent, and a replica that lists a field and then refuses it teaches the agent to drop it: on 2026-10-15 after 15:00
    the medusa pack refused `context` 239 times in 155 episodes, and in 40 of the 41 episodes that then met "do not
    have a price" the agent had first been told `context` was unrecognized and had dropped it from its creates.
    `currency` is the store's: a create with no `context.currency` is priced in it, as UCP's cart says ("falls back to
    geo-IP if omitted"), and only a currency the store does not price in gets the plugin's "do not have a price"
    (store-api 64)."""
    payload = arguments.get(payload_key) if payload_key is not None else None
    body: Mapping[str, Any] = payload if isinstance(payload, Mapping) else {}
    allowed = ALLOWED.get(tool)
    if allowed is not None:
        extra = [k for k in body if k not in allowed and k not in served]
        if extra:
            return invalid_data("Unrecognized fields: " + ", ".join(f"'{k}'" for k in extra))
    if tool in ("create_cart", "create_checkout") and body.get("line_items"):
        context = body.get("context")
        asked = context.get("currency") if isinstance(context, Mapping) else None
        if isinstance(asked, str) and asked and currency is not None and asked.upper() != currency.upper():
            ids = [str((li.get("item") or {}).get("id")) for li in body["line_items"] if isinstance(li, Mapping)]
            text = f"Variants with IDs {', '.join(ids)} do not have a price"
            return Wire(500, error_doc(version, "internal_error", text))
    served_countries = countries_served(market)
    for where, country in _countries(body):
        if country and country.upper() not in served_countries:
            listed = ", ".join(c.lower() for c in served_countries)
            text = f'Country "{country.lower()}" is not served by any region. Supported countries: {listed}.'
            return Wire(400, error_doc(version, "country_not_supported", text, severity="recoverable", path=where))
    return None


EXTENSION_FIELDS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {k: MappingProxyType(dict(v)) for k, v in SPEC.extra["extension_fields"].items()}
)
"""Body fields an extension adds to a tool, by the capability that defines them (UCP cart.json: "Adds cart_id to
create_checkout for cart-to-checkout conversion"; fulfillment.json on checkout). The release's tool schema lists only
the base fields."""


def accepted(tool: str, schema_fields: frozenset[str], capabilities: frozenset[str]) -> frozenset[str]:
    """The body fields the served release defines for `tool` on this plugin: its request schema's own, and those of
    each extension the release defines and the plugin serves (`ENVELOPE_CAPABILITIES`), never one of a capability
    the plugin does not serve (`UNSERVED`)."""
    extensions = {
        name
        for name, capability in EXTENSION_FIELDS.get(tool, {}).items()
        if capability in capabilities and capability in ENVELOPE_CAPABILITIES
    }
    return frozenset(schema_fields | extensions) - UNSERVED_FIELDS


def _countries(body: Mapping[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    address = body.get("shipping_address")
    if isinstance(address, Mapping) and isinstance(address.get("address_country"), str):
        out.append(("$.shipping_address.address_country", address["address_country"]))
    fulfillment = body.get("fulfillment")
    methods = fulfillment.get("methods") if isinstance(fulfillment, Mapping) else None
    for i, m in enumerate(methods if isinstance(methods, list) else ()):
        for j, d in enumerate((m.get("destinations") or []) if isinstance(m, Mapping) else ()):
            if isinstance(d, Mapping) and isinstance(d.get("address_country"), str):
                out.append((f"$.fulfillment.methods[{i}].destinations[{j}].address_country", d["address_country"]))
    return out


def schema_problem(content: str) -> Wire:
    """A body the sandbox's release schema refuses, worded as Medusa's validator words it ("Field 'variant_id' is
    required", store-api 36)."""
    m = re.fullmatch(r"'([^']+)' is required", content)
    return invalid_data(f"Field '{m.group(1)}' is required" if m else content)


def search_payload(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """A search as the plugin reads it: `offset` pages (a cursor is ignored, store-api 110), 20 to a page and no more
    than 100, spec price and category filters ignored (store-api 108), `filters.category` matched by name
    (store-api 109). Returned in the sandbox's terms: the offset as the catalog's decimal cursor."""
    pagination = catalog.get("pagination")
    paging: Mapping[str, Any] = pagination if isinstance(pagination, Mapping) else {}
    limit = paging.get("limit")
    offset = paging.get("offset")
    size = (
        min(limit, SEARCH_MAX) if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0 else SEARCH_PAGE
    )
    out: dict[str, Any] = {k: v for k, v in catalog.items() if k not in ("pagination", "filters")}
    page: dict[str, Any] = {"limit": size}
    if isinstance(offset, int) and not isinstance(offset, bool) and offset > 0:
        page["cursor"] = str(offset)
    out["pagination"] = page
    raw = catalog.get("filters")
    category = raw.get("category") if isinstance(raw, Mapping) else None
    if isinstance(category, str) and category:
        out["filters"] = {"categories": [category]}
    return out


def phrase_matched(products: Sequence[Any], query: str) -> list[Any]:
    """The products whose title, description or handle holds the whole query, case-folded (store-api 94 to 105).
    Each field is matched on its own, as the plugin's filter tests them one by one."""
    q = query.casefold()
    if not q:
        return list(products)
    out = []
    for p in products:
        plain = p.description.get("plain")
        body = plain if plain is not None else " ".join(p.description.values())
        handle = str(p.facts.get("handle") or handle_of(p.title, p.id))
        if q in p.title.casefold() or q in str(body).casefold() or q in handle.casefold():
            out.append(p)
    return out


def handle_of(title: str, ident: str) -> str:
    """A product's handle as the playground's seed writes one: the title's slug and the id's last segment."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    tail = ident.rsplit("/", 1)[-1].lower()
    return f"{slug}-{tail}" if tail and tail != slug else slug


# -- documents ---------------------------------------------------------------------------------------------------------


def envelope(version: str, *, status: str = "success", handlers: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The plugin's `ucp` member: its capabilities at the served release (`ENVELOPE`: six at 2026-08-25, ucp-2026-08-25
    12; three at 2026-01-23, ucp-2026-01-23 09), in the release's own form, and on a checkout the payment handlers
    (the sandbox's, see NOT_PORTED)."""
    names = ENVELOPE.get(version, ENVELOPE_CAPABILITIES)
    capabilities: Any = (
        [{"name": name, "version": version} for name in names]
        if version in CAPABILITY_LIST
        else {name: [{"version": version}] for name in names}
    )
    out: dict[str, Any] = {"version": version, "status": status, "capabilities": capabilities}
    if handlers is not None:
        out["payment_handlers"] = dict(handlers)
    return out


def reshape(tool: str, doc: Mapping[str, Any], facts: Facts) -> Wire:  # noqa: PLR0911 - one per kind
    """The sandbox's UCP document for `tool` as the plugin shapes it."""
    if _is_error(doc):
        return _error(doc, facts)
    if tool == "search_catalog":
        return Wire(200, _search(doc, facts))
    if tool == "lookup_catalog":
        return Wire(200, _lookup(doc, facts))
    if tool == "get_product":
        return Wire(200, _product(doc, facts))
    if tool in ("create_cart", "get_cart", "update_cart", "cancel_cart"):
        refused = _stock_refused(doc, facts)
        return refused or Wire(201 if tool == "create_cart" else 200, _cart(doc, facts))
    if tool in ("create_checkout", "get_checkout", "update_checkout", "complete_checkout", "cancel_checkout"):
        refused = _stock_refused(doc, facts)
        return refused or Wire(201 if tool == "create_checkout" else 200, _checkout(doc, facts))
    if tool == "get_order":
        return Wire(200, _order(doc, facts))
    return Wire(200, dict(doc))


def _is_error(doc: Mapping[str, Any]) -> bool:
    ucp = doc.get("ucp")
    return isinstance(ucp, Mapping) and ucp.get("status") == "error"


NOT_FOUND_TEXT: Mapping[str, str] = MappingProxyType(dict(SPEC.extra["not_found_text"]))


def _error(doc: Mapping[str, Any], facts: Facts) -> Wire:
    """A sandbox error document as the plugin answers: a refused input is Medusa's validator's 400, an unknown
    resource a 404, a line it cannot sell or an item it does not have the 500 its catch-all sends."""
    messages = [m for m in doc.get("messages") or () if isinstance(m, Mapping) and m.get("type") == "error"]
    first = messages[0] if messages else {}
    code = str(first.get("code") or "")
    path = str(first.get("path") or "")
    if code == "not_found" and not path.startswith("$.line_items"):
        noun = next((n for n in NOT_FOUND_TEXT if n in str(first.get("content") or "").lower()), "order")
        return Wire(404, error_doc(facts.version, "not_found", NOT_FOUND_TEXT[noun]))
    if code in ("out_of_stock", "merchandise_out_of_stock"):
        return Wire(500, error_doc(facts.version, "internal_error", WORDING["stock.out_of_stock"]))
    if code == "not_found":
        ids = ", ".join(_named_items(messages, facts.request)) or "requested"
        text = f"Variants {ids} do not exist or belong to a product that is not published"
        return Wire(500, error_doc(facts.version, "internal_error", text))
    if code in ("invalid_argument", "field_required", "missing", "invalid_id", "invalid_cursor"):
        return schema_problem(str(first.get("content") or ""))
    return Wire(500, error_doc(facts.version, "internal_error", str(first.get("content") or "Internal error")))


def _stock_refused(doc: Mapping[str, Any], facts: Facts) -> Wire | None:
    """A cart or checkout write with a line Medusa cannot sell: the plugin's catch-all 500 in Medusa's words, the
    resource not shown (ucp-2026-08-25 10; store-api 30, 31, 37)."""
    for m in doc.get("messages") or ():
        refused = isinstance(m, Mapping) and m.get("type") == "error" and m.get("code") == "out_of_stock"
        if refused and (m.get("path") is None or str(m.get("path")).startswith("$.line_items")):
            return Wire(500, error_doc(facts.version, "internal_error", WORDING["stock.out_of_stock"]))
    return None


def _named_items(messages: Sequence[Mapping[str, Any]], request: Mapping[str, Any] | None) -> list[str]:
    """The item ids the request named at each not-found message's `$.line_items[i]` path (store-api 28: Medusa
    names the variants it does not have)."""
    lines = (request or {}).get("line_items")
    out: list[str] = []
    for m in messages:
        found = re.match(r"\$\.line_items\[(\d+)\]", str(m.get("path") or ""))
        if found and isinstance(lines, list) and int(found.group(1)) < len(lines):
            entry = lines[int(found.group(1))]
            item = entry.get("item") if isinstance(entry, Mapping) else None
            if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                out.append(item["id"])
    return out


def _search(doc: Mapping[str, Any], facts: Facts) -> dict[str, Any]:
    products = [product(p) for p in doc.get("products") or () if isinstance(p, Mapping)]
    request = facts.request or {}
    paging = request.get("pagination") if isinstance(request.get("pagination"), Mapping) else {}
    assert isinstance(paging, Mapping)
    limit = int(paging.get("limit") or SEARCH_PAGE)
    cursor = paging.get("cursor")
    offset = int(cursor) if isinstance(cursor, str) and cursor.isdigit() else 0
    given = doc.get("pagination")
    total = given.get("total_count") if isinstance(given, Mapping) else None
    count = int(total) if isinstance(total, int) else offset + len(products)
    return {
        "ucp": {"version": facts.version, "status": "success"},
        "products": products,
        "pagination": {"total": count, "limit": limit, "offset": offset, "has_more": offset + limit < count},
    }


def _lookup(doc: Mapping[str, Any], facts: Facts) -> dict[str, Any]:
    found = doc.get("products")
    if found is None and isinstance(doc.get("product"), Mapping):
        found = [doc["product"]]
    products = [product(p) for p in found or () if isinstance(p, Mapping)]
    return {"ucp": {"version": facts.version, "status": "success"}, "products": products, "messages": []}


def _product(doc: Mapping[str, Any], facts: Facts) -> dict[str, Any]:
    """get_product, which the plugin has no route for (its recordings serve search and lookup only): UCP's
    get_product response holding the plugin's product shape. It had answered as a lookup, a `products` list with no
    `product`, which the release's response schema refuses (the pack conformance suite, D101)."""
    found = doc.get("product")
    if not isinstance(found, Mapping):
        listed = [p for p in doc.get("products") or () if isinstance(p, Mapping)]
        found = listed[0] if listed else None
    out: dict[str, Any] = {"ucp": {"version": facts.version, "status": "success"}, "messages": []}
    if isinstance(found, Mapping):
        out["product"] = product(found)
    return out


def product(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A product as the plugin formats one (P lib/formatters/ucp.js formatUcpProduct; ucp-2026-08-25 06, 08)."""
    variants: list[dict[str, Any]] = []
    for v in doc.get("variants") or ():
        if not isinstance(v, Mapping):
            continue
        price = v.get("price") if isinstance(v.get("price"), Mapping) else None
        available = _available(v)
        variants.append(
            {
                "id": v.get("id"),
                "title": str(v.get("title") or ""),
                "sku": v.get("sku") if isinstance(v.get("sku"), str) else None,
                "price": _money(price) if price is not None else None,
                "availability": {"available": available, "status": "in_stock" if available else "out_of_stock"},
            }
        )
    amounts = [v["price"]["amount"] for v in variants if v["price"] is not None]
    currency = next((v["price"]["currency"] for v in variants if v["price"] is not None), "eur")
    price_range = (
        {"min": {"amount": min(amounts), "currency": currency}, "max": {"amount": max(amounts), "currency": currency}}
        if amounts
        else None
    )
    description = doc.get("description")
    text = description.get("plain") if isinstance(description, Mapping) else description
    title = str(doc.get("title") or "")
    return {
        "id": doc.get("id"),
        "title": title,
        "description": str(text or ""),
        "handle": str(doc.get("handle") or handle_of(title, str(doc.get("id") or ""))),
        "categories": [_leaf(c) for c in doc.get("categories") or ()],
        "price_range": price_range,
        "variants": variants,
        "media": [{"url": m.get("url"), "type": "image"} for m in doc.get("media") or () if isinstance(m, Mapping)],
    }


def _available(v: Mapping[str, Any]) -> bool:
    availability = v.get("availability")
    if isinstance(availability, Mapping) and isinstance(availability.get("available"), bool):
        return bool(availability["available"])
    return bool(v.get("available", True))


def _money(price: Mapping[str, Any]) -> dict[str, Any]:
    """A catalog price as the plugin sends it: minor units and Medusa's lower-case currency code."""
    return {"amount": int(price.get("amount") or 0), "currency": str(price.get("currency") or "eur").lower()}


def _leaf(category: Any) -> str:
    """A category's name: the last step of a `A > B` path."""
    value = category.get("value") if isinstance(category, Mapping) else category
    return str(value or "").split(">")[-1].strip()


def _lines(doc: Mapping[str, Any], version: str) -> list[dict[str, Any]]:
    """Each line with its one total, typed as the plugin types it at the served release (`LINE_TOTAL`)."""
    kind = LINE_TOTAL.get(version, LINE_TOTAL["*"])
    out = []
    for li in doc.get("line_items") or ():
        if not isinstance(li, Mapping):
            continue
        item = li.get("item") if isinstance(li.get("item"), Mapping) else {}
        assert isinstance(item, Mapping)
        price = int(item.get("price") or 0)
        quantity = li.get("quantity")
        qty = int(quantity.get("total", 0)) if isinstance(quantity, Mapping) else int(quantity or 0)
        out.append(
            {
                "id": li.get("id"),
                "item": {"id": item.get("id"), "title": str(item.get("title") or ""), "price": price},
                "quantity": qty,
                "totals": [{"type": kind, "display_text": "Line total", "amount": price * qty}],
            }
        )
    return out


def _totals(doc: Mapping[str, Any], facts: Facts) -> list[dict[str, Any]]:
    """The plugin's totals: subtotal, then shipping, tax and discount where above zero, then the total. Shipping and
    discount are Medusa's `shipping_total` and `discount_total`, which carry their tax (ucp-2026-08-25 17)."""
    by_type = {str(t.get("type")): int(t.get("amount") or 0) for t in doc.get("totals") or () if isinstance(t, Mapping)}
    subtotal = by_type.get("subtotal", 0)
    shipping = by_type.get("fulfillment", 0)
    tax = by_type.get("tax", 0)
    discount = -by_type.get("discount", 0)
    out = [{"type": "subtotal", "display_text": "Subtotal", "amount": subtotal}]
    if shipping > 0:
        shown = shipping + (percent_of(shipping, facts.tax_bp) if facts.tax_on_shipping else 0)
        out.append({"type": "fulfillment", "display_text": "Shipping", "amount": shown})
    if tax > 0:
        out.append({"type": "tax", "display_text": "Tax", "amount": tax})
    if discount > 0:
        out.append(
            {"type": "discount", "display_text": "Discount", "amount": -(discount + percent_of(discount, facts.tax_bp))}
        )
    out.append({"type": "total", "display_text": "Total", "amount": by_type.get("total", 0)})
    return out


def _links(base: str) -> list[dict[str, str]]:
    """Two untitled policy links on the storefront (ucp-2026-08-25 12)."""
    return [{"type": "terms_of_service", "url": f"{base}/terms"}, {"type": "privacy_policy", "url": f"{base}/privacy"}]


def _address(raw: Mapping[str, Any]) -> dict[str, Any]:
    """A postal address as the plugin's medusaToUcpAddress writes one: Medusa's lower-case country code."""
    keep = (
        "first_name",
        "last_name",
        "street_address",
        "extended_address",
        "address_locality",
        "address_region",
        "address_country",
        "postal_code",
        "phone_number",
    )
    out = {k: raw[k] for k in keep if raw.get(k)}
    if isinstance(out.get("address_country"), str):
        out["address_country"] = out["address_country"].lower()
    return out


def _destination(doc: Mapping[str, Any]) -> Mapping[str, Any] | None:
    fulfillment = doc.get("fulfillment")
    for m in (fulfillment.get("methods") or []) if isinstance(fulfillment, Mapping) else ():
        for d in (m.get("destinations") or []) if isinstance(m, Mapping) else ():
            if isinstance(d, Mapping) and d.get("street_address"):
                return d
    return None


def _cart(doc: Mapping[str, Any], facts: Facts) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ucp": envelope(facts.version),
        "id": doc.get("id"),
        "currency": doc.get("currency"),
        "line_items": _lines(doc, facts.version),
        "totals": _totals(doc, facts),
        "messages": _kept(doc.get("messages")),
        "links": _links(facts.base),
    }
    buyer = doc.get("buyer")
    if isinstance(buyer, Mapping) and buyer.get("email"):
        out["buyer"] = {"email": buyer["email"]}
    if facts.country:
        out["shipping_address"] = {"address_country": facts.country.lower()}
    return out


def _kept(messages: Any) -> list[Any]:
    """The sandbox's messages the plugin also sends: all but the order-status note, which it never adds."""
    return [dict(m) for m in messages or () if isinstance(m, Mapping) and m.get("code") != "order_status"]


def _checkout(doc: Mapping[str, Any], facts: Facts) -> dict[str, Any]:
    status = str(doc.get("status") or "incomplete")
    ucp = doc.get("ucp") if isinstance(doc.get("ucp"), Mapping) else {}
    assert isinstance(ucp, Mapping)
    handlers = ucp.get("payment_handlers")
    payment = doc.get("payment")
    listed = isinstance(payment, Mapping) and not isinstance(handlers, Mapping)
    out: dict[str, Any] = {
        "ucp": envelope(facts.version, handlers=None if listed else handlers or {}),
        "id": doc.get("id"),
        "status": status,
        "currency": doc.get("currency"),
        "line_items": _lines(doc, facts.version),
        "totals": _totals(doc, facts),
        "messages": [*_kept(doc.get("messages")), *_status_notes(status, facts)],
        "links": _links(facts.base),
    }
    handoff = doc.get("continue_url")
    if status == "requires_escalation" and isinstance(handoff, str) and handoff:
        # UCP's checkout schema: continue_url MUST be provided when status is requires_escalation. Dropping it left a
        # 3-D Secure challenge on Medusa with no page to answer it at (2026-10-17: the PSP issued the challenge and no
        # wallet could serve it, once Medusa's lower-case "usd" no longer stopped rows before payment, D104)
        out["continue_url"] = handoff
    if listed and isinstance(payment, Mapping):
        # UCP 2026-01-11 lists the handlers in the checkout's `payment`; dropping it left a Medusa checkout with none
        # to pay through (2026-10-16: 3 rows built to pay on ucp_psp at 2026-01-11, no order)
        out["payment"] = dict(payment)
    destination = _destination(doc)
    buyer = doc.get("buyer") if isinstance(doc.get("buyer"), Mapping) else {}
    assert isinstance(buyer, Mapping)
    named = {k: destination[k] for k in ("first_name", "last_name") if destination is not None and destination.get(k)}
    shown = {**named, **{k: buyer[k] for k in ("first_name", "last_name", "email", "phone_number") if buyer.get(k)}}
    if shown:
        out["buyer"] = shown
    if destination is not None:
        out["shipping_address"] = _address(destination)
    elif facts.country:
        out["shipping_address"] = {"address_country": facts.country.lower()}
    fulfillment = _fulfillment(doc, completed=status == "completed")
    if fulfillment is not None:
        out["fulfillment"] = fulfillment
    created = facts.checkout.created_at if facts.checkout is not None else None
    if created is not None:
        out["expires_at"] = _iso(created + CHECKOUT_TTL_MS)
    order = doc.get("order")
    if isinstance(order, Mapping) and isinstance(order.get("id"), str):
        oid = order["id"]
        out["order"] = {
            "id": oid,
            "checkout_id": doc.get("id"),
            "permalink_url": f"{facts.base}/orders/{oid}",
            "links": [{"type": "self", "url": f"{facts.base}/ucp/orders/{oid}"}],
        }
    return out


def _status_notes(status: str, facts: Facts) -> list[dict[str, Any]]:
    """The info message the plugin adds for the session's state (P lib/formatters/ucp.js buildCheckoutMessages)."""
    if status == "completed":
        return [{"type": "info", "code": "checkout_completed", "content": "Checkout completed successfully."}]
    if status == "canceled":
        return [{"type": "info", "code": "checkout_canceled", "content": "This checkout session has been canceled."}]
    if status == "ready_for_complete":
        text = "Checkout is ready. POST /checkout-sessions/{id}/complete with payment.instruments[]."
        return [{"type": "info", "code": "ready_for_complete", "content": text}]
    return []


def _fulfillment(doc: Mapping[str, Any], *, completed: bool) -> dict[str, Any] | None:
    """The plugin's fulfillment: one shipping method and group, each option with one fulfillment total titled as the
    option; options are emptied once the order is placed (ucp-2026-08-25 20)."""
    raw = doc.get("fulfillment")
    if not isinstance(raw, Mapping):
        return None
    methods = []
    for m in raw.get("methods") or ():
        if not isinstance(m, Mapping):
            continue
        destinations = [
            {"id": d.get("id"), **_address(d)} for d in m.get("destinations") or () if isinstance(d, Mapping)
        ]
        groups = []
        for g in m.get("groups") or ():
            if not isinstance(g, Mapping):
                continue
            options = [] if completed else [_option(o) for o in g.get("options") or () if isinstance(o, Mapping)]
            groups.append(
                {
                    "id": g.get("id"),
                    "line_item_ids": list(g.get("line_item_ids") or ()),
                    "options": options,
                    "selected_option_id": g.get("selected_option_id"),
                }
            )
        methods.append(
            {
                "id": m.get("id"),
                "type": m.get("type"),
                "line_item_ids": list(m.get("line_item_ids") or ()),
                "destinations": destinations,
                "selected_destination_id": destinations[0]["id"] if destinations else None,
                "groups": groups,
            }
        )
    return {"methods": methods}


def _option(o: Mapping[str, Any]) -> dict[str, Any]:
    title = str(o.get("title") or "Shipping")
    amount = next((int(t.get("amount") or 0) for t in o.get("totals") or () if isinstance(t, Mapping)), 0)
    out: dict[str, Any] = {
        "id": o.get("id"),
        "title": title,
        "totals": [{"type": "fulfillment", "display_text": title, "amount": amount}],
    }
    if o.get("carrier"):
        out["carrier"] = o["carrier"]
    return out


def _order(doc: Mapping[str, Any], facts: Facts) -> dict[str, Any]:
    """An order as the plugin formats one (ucp-2026-08-25 21): Medusa's display number, its status, no checkout id
    (the plugin reads a field the order does not carry), a self link."""
    order = facts.order
    oid = doc.get("id")
    placed = _iso(order.placed_at) if order is not None else None
    out: dict[str, Any] = {
        "ucp": envelope(facts.version),
        "id": oid,
        "display_id": order.number if order is not None else None,
        "checkout_id": None,
        "permalink_url": f"{facts.base}/orders/{oid}",
        "status": ORDER_STATES.get(order.status, "pending") if order is not None else "pending",
        "currency": doc.get("currency"),
        "line_items": _lines(doc, facts.version),
        "totals": _totals(doc, facts),
        "fulfillment_status": "fulfilled" if order is not None and order.status == "fulfilled" else "not_fulfilled",
        "fulfillment_events": [],
        "created_at": placed,
        "updated_at": placed,
        "links": [{"type": "self", "url": f"{facts.base}/ucp/orders/{oid}"}],
    }
    if order is not None and order.buyer.get("email"):
        out["buyer"] = {"email": order.buyer["email"]}
    destination = next((g.destination for g in order.groups if g.destination), None) if order is not None else None
    if destination:
        out["shipping_address"] = _address(destination)
    return out


def _iso(ms: int) -> str:
    """A time as the plugin writes one: JavaScript's toISOString, milliseconds and `Z`."""
    stamp = datetime.fromtimestamp(ms / 1000, tz=UTC)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms % 1000:03d}Z"


def acp_discovery(version: str, api_base_url: str, currency: str) -> dict[str, Any]:
    """The plugin's ACP discovery document at the served release, base URL and store currency."""
    doc = ACP_DISCOVERY
    protocol = dict(doc["protocol"])
    capabilities = dict(doc["capabilities"])
    return {
        "protocol": {**protocol, "version": version, "supported_versions": [version]},
        "api_base_url": api_base_url,
        "transports": list(doc["transports"]),
        "capabilities": {**capabilities, "payment": {"handlers": []}, "supported_currencies": [currency.lower()]},
    }

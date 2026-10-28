"""What each storefront page shows, as JSON: the view a page renders and its tool answers with.

Why one view per page. The draft has a tool answer with whatever its page's script resolves; a storefront's tools
answer with what the page now shows. So each page has one view model, built from the store's own documents (the UCP
documents the sandbox's operations render from the core state), and the page renders exactly that view
(`carttrace.sandbox.surfaces.webmcp.html`). The facts are the store's: product documents pass through untouched (ids,
descriptions, prices in minor units, variants, options, details), which is also what the protocol agents' readers
take. What changes is the shape around them: a page has no `ucp` envelope or JSON paths, so messages lose their
`path`, the checkout's fulfillment is shown as the one delivery choice the page offers, and what a checkout still
lacks names the tool that fills it in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlencode

from carttrace.sandbox.core.market import Market

Json = dict[str, Any]

MISSING_TOOLS: tuple[tuple[str, str], ...] = (
    ("destinations", "set_shipping_address"),
    ("shipping_address", "set_shipping_address"),
    ("selected_option_id", "choose_fulfillment"),
    ("$.buyer", "set_buyer_contact"),
    ("$.fulfillment", "set_shipping_address or choose_fulfillment"),
    ("$.line_items", "add_to_cart"),
)
"""Which tool fills in what a checkout lacks, by the UCP path of its `missing` message (first match wins)."""
MISSING_CODES: dict[str, str] = {
    "missing_email": "$.buyer.email",
    "missing_phone": "$.buyer.phone_number",
    "missing_shipping_address": "$.fulfillment.methods[0].destinations",
}
"""A platform pack's own `missing_` codes (WooCommerce's) as the UCP path they stand for, so the page lists them as
missing with their form: without it a WooCommerce-pack pickup checkout waited for an email no form was named for
(the payments check, D94)."""
DOC_ONLY = ("ucp", "links", "continue_url", "actions")
"""Members of a UCP document a page has no use for."""


def messages_of(doc: Mapping[str, Any]) -> list[Json]:
    """A document's messages as a page states them: type, code, text and severity."""
    out = []
    for m in doc.get("messages") or ():
        if isinstance(m, Mapping):
            shown = {k: m[k] for k in ("type", "code", "content", "severity") if k in m}
            out.append(shown)
    return out


def metadata_of(product: Mapping[str, Any]) -> Mapping[str, Any]:
    """A product's business-defined data (UCP `metadata`), or an empty one."""
    meta = product.get("metadata")
    return meta if isinstance(meta, Mapping) else {}


def first_error(doc: Mapping[str, Any]) -> Json | None:
    for m in doc.get("messages") or ():
        if isinstance(m, Mapping) and m.get("type") == "error":
            return {"code": str(m.get("code") or "error"), "message": str(m.get("content") or "")}
    return None


def search_view(query: str, doc: Mapping[str, Any], *, filters: Mapping[str, Any]) -> Json:
    params = {"q": query, **{k: str(v) for k, v in filters.items() if v is not None}}
    pagination = doc.get("pagination") or {}
    view: Json = {
        "page": "search",
        "url": f"/shop/search?{urlencode(params)}",
        "query": query,
        "products": list(doc.get("products") or ()),
    }
    if pagination.get("has_next_page") and pagination.get("cursor") is not None:
        view["next_cursor"] = str(pagination["cursor"])
    if doc.get("messages"):
        view["messages"] = messages_of(doc)
    return view


def product_view(product: Mapping[str, Any]) -> Json:
    return {"page": "product", "url": f"/shop/products/{quote(str(product['id']), safe='')}", "product": dict(product)}


def compare_view(ids: Sequence[str], doc: Mapping[str, Any]) -> Json:
    """2 to 4 products on the attributes at least two of them state, then what only one states."""
    by_id = {str(p.get("id")): p for p in doc.get("products") or () if isinstance(p, Mapping)}
    products = [by_id[i] for i in ids if i in by_id]
    facts = [_facts(p) for p in products]
    names: list[str] = []
    for f in facts:
        names += [n for n in f if n not in names]
    rows: list[Json] = []
    alone: dict[str, list[str]] = {}
    for name in names:
        holders = [i for i, f in enumerate(facts) if name in f]
        if len(holders) >= 2:
            values = [{"product_id": str(p["id"]), "value": f.get(name)} for p, f in zip(products, facts, strict=True)]
            rows.append({"attribute": name, "values": values})
        else:
            alone.setdefault(str(products[holders[0]]["id"]), []).append(name)
    view: Json = {
        "page": "compare",
        "url": f"/shop/compare?{urlencode({'ids': ','.join(ids)})}",
        "products": [
            {"id": str(p["id"]), "title": p.get("title"), "price_range": p.get("price_range")} for p in products
        ],
        "rows": rows,
    }
    if alone:
        view["stated_by_one"] = [{"product_id": pid, "attributes": attrs} for pid, attrs in alone.items()]
    missing = [i for i in ids if i not in by_id]
    if missing:
        view["not_found"] = missing
    return view


def _facts(p: Mapping[str, Any]) -> dict[str, Any]:
    """A product's comparable attributes, in a fixed order: price, options, then the store's stated details."""
    out: dict[str, Any] = {}
    if isinstance(p.get("price_range"), Mapping):
        out["price"] = p["price_range"]
    for o in p.get("options") or ():
        if isinstance(o, Mapping) and isinstance(o.get("name"), str):
            out[o["name"]] = [v.get("label") for v in o.get("values") or () if isinstance(v, Mapping)]
    meta = metadata_of(p)
    for key in ("brand", "product_type"):
        if isinstance(meta.get(key), str):
            out[key] = meta[key]
    for group in ("details", "attributes"):
        entries = meta.get(group)
        if isinstance(entries, Mapping):
            for k, v in entries.items():
                out.setdefault(str(k), v)
    if isinstance(p.get("rating"), Mapping):
        out["rating"] = p["rating"]
    return out


def cart_view(doc: Mapping[str, Any] | None, *, locked: bool = False) -> Json:
    view: Json = {"page": "cart", "url": "/shop/cart"}
    if doc is None:
        view["cart"] = None
        view["note"] = "The cart is empty."
        return view
    view["cart"] = {k: v for k, v in doc.items() if k not in (*DOC_ONLY, "messages")}
    if doc.get("messages"):
        view["messages"] = messages_of(doc)
    if locked:
        view["note"] = (
            "Checkout has started for this cart: continue it with get_checkout and place_order, or change the cart, "
            "which cancels that checkout so you start it again."
        )
    return view


def checkout_view(
    doc: Mapping[str, Any], *, market: Market, handlers: Sequence[str], challenge: str | None = None
) -> Json:
    """The checkout page: the checkout with one delivery choice, what is missing and the tool for each, the payment
    methods the page offers, and the bank's challenge page when one is waiting."""
    shown = {k: v for k, v in doc.items() if k not in (*DOC_ONLY, "messages", "fulfillment")}
    shown["delivery"] = delivery_of(doc.get("fulfillment"), market)
    view: Json = {"page": "checkout", "url": "/shop/checkout", "checkout": shown}
    missing = []
    others = []
    for m in doc.get("messages") or ():
        if not isinstance(m, Mapping):
            continue
        code = str(m.get("code") or "")
        if code == "missing" or code in MISSING_CODES:
            path = str(m.get("path") or MISSING_CODES.get(code, ""))
            missing.append({"what": str(m.get("content") or ""), "use": _tool_for(path)})
        else:
            others.append({k: m[k] for k in ("type", "code", "content", "severity") if k in m})
    view["missing"] = missing
    if others:
        view["messages"] = others
    view["payment_handlers"] = list(handlers)
    if challenge is not None:
        view["challenge"] = {"url": challenge, "note": "The buyer's bank asks the buyer to confirm this payment."}
    return view


def delivery_of(fulfillment: Any, market: Market) -> Json:
    """The checkout's one delivery choice as the page shows it, and the pickup locations it offers."""
    methods = fulfillment.get("methods") if isinstance(fulfillment, Mapping) else None
    out: Json = {"method": None}
    if isinstance(methods, list) and methods and isinstance(methods[0], Mapping):
        m = methods[0]
        out["method"] = m.get("type")
        group = (m.get("groups") or [{}])[0]
        if m.get("type") == "shipping":
            dests = m.get("destinations") or []
            if dests:
                out["address"] = {k: v for k, v in dests[0].items() if k not in ("id", "type")}
            out["options"] = [
                {
                    "id": o.get("id"),
                    "title": o.get("title"),
                    "amount": _option_amount(o),
                }
                for o in group.get("options") or ()
                if isinstance(o, Mapping)
            ]
            out["selected_option"] = group.get("selected_option_id")
        else:
            out["location"] = m.get("selected_destination_id") or group.get("selected_option_id")
    out["pickup_locations"] = [{"id": p.id, "title": p.title} for p in market.pickup]
    return out


def order_view(doc: Mapping[str, Any]) -> Json:
    order = {k: v for k, v in doc.items() if k not in (*DOC_ONLY, "permalink_url")}
    return {
        "page": "order",
        "url": f"/shop/orders/{quote(str(doc.get('id')), safe='')}",
        "order": order,
        "note": f"Order {doc.get('label', '')} is placed. A confirmation goes to the buyer's email.".replace("  ", " "),
    }


def error_view(page: str, url: str, code: str, message: str, **extra: Any) -> Json:
    return {"page": page, "url": url, "error": {"code": code, "message": message}, **extra}


def _option_amount(option: Mapping[str, Any]) -> int | None:
    """An option's price: its totals entry of type total, or its one priced entry (Medusa's pack writes
    `fulfillment`, D94)."""
    priced = [t for t in option.get("totals") or () if isinstance(t, Mapping) and isinstance(t.get("amount"), int)]
    found = next((t for t in priced if t.get("type") == "total"), priced[0] if len(priced) == 1 else None)
    return int(found["amount"]) if found is not None else None


def _tool_for(path: str) -> str:
    return next((tool for marker, tool in MISSING_TOOLS if marker in path), "get_checkout")

"""The storefront's HTML: the page shell, one fragment per view, and the declarative checkout forms.

Why server-rendered fragments. A tool answers with what its page shows, so the page must show exactly the tool's
view. The view is rendered here once, in Python, both for a page load and for every tool call (the storefront's API
returns the view and its fragment together; the page's script swaps the fragment in), so there is one template per
page and nothing for the page's script to format. Prices are shown from minor units by `carttrace.core.money`.

What a person sees in a screenshot is the replica: the store's name with "sandbox replica", its products, cart and
checkout, and the checkout's three forms. The forms are the declarative WebMCP tools: `toolname`, `tooldescription`,
`toolparamdescription` and `toolautosubmit` (declarative-api-explainer.md), with their fields from
`carttrace.sandbox.surfaces.webmcp.tools`. They stay in the page outside the swapped fragment, so a tool's form is never
replaced while it runs. Product images are not loaded: their URLs point at the live store's CDN, which the offline
browser refuses; the page names them instead.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

from carttrace.core.money import Money
from carttrace.sandbox.surfaces.webmcp.tools import FormField, ToolDef
from carttrace.sandbox.surfaces.webmcp.views import metadata_of

STATIC = Path(__file__).resolve().parent / "static"
STYLE = """
body{font:15px/1.45 system-ui,sans-serif;margin:0;color:#222;background:#fbfaf7}
header{display:flex;gap:16px;align-items:center;padding:12px 24px;background:#24313d;color:#fff}
header a{color:#fff;text-decoration:none}header .store{font-weight:600}header .tag{opacity:.7;font-size:13px}
header form{margin-left:auto}main,#checkout-forms{max-width:980px;margin:0 auto;padding:16px 24px}
.card{border:1px solid #ddd;border-radius:6px;padding:12px;margin:10px 0;background:#fff}
table{border-collapse:collapse;width:100%;margin:8px 0}td,th{border:1px solid #ddd;padding:4px 8px;text-align:left}
.price{font-weight:600}.muted{color:#666;font-size:13px}.error{color:#9b1c1c;font-weight:600}
.missing li{color:#7a4b00}form.tool{border:1px solid #ccd;border-radius:6px;padding:12px;margin:12px 0;background:#fff}
form.tool label{display:block;margin:6px 0}form.tool input,form.tool select{margin-left:8px}
body:not([data-page=checkout]) #checkout-forms{display:none}iframe{width:100%;height:220px;border:1px solid #ccd}
"""


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def money(amount: object, currency: object) -> str:
    """An amount in minor units as the page shows it. A platform pack's wire may name the currency in its own case
    (Medusa's store API says "usd"), so the code is read case-blind; a page never fails over how a code is spelled
    (2026-10-15: every Medusa-pack WebMCP search that found a product ended VOID)."""
    if not isinstance(amount, int) or isinstance(amount, bool) or not isinstance(currency, str):
        return ""
    code = currency.strip().upper()
    return f"{Money(amount, code).major()} {code}"


@cache
def page_script(*, bare: bool = False) -> str:
    """The storefront's script; `bare` without its comment lines, which name the sandbox's own modules (a stand-in's
    pages, D106)."""
    text = (STATIC / "storefront.js").read_text(encoding="utf-8")
    if not bare:
        return text
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))


def document(
    *,
    store: str,
    title: str,
    view: Mapping[str, Any],
    fragment: str,
    forms: str,
    config: Mapping[str, Any],
    tag: str | None = "sandbox replica: nothing here is a real sale",
    bare: bool = False,
) -> str:
    """A whole page: the shell, the view's fragment, the checkout forms, and the script that registers the tools."""
    data = json.dumps(config, sort_keys=True, separators=(",", ":")).replace("<", "\\u003c")
    page = esc(view.get("page") or "home")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><link rel="icon" href="data:,">'
        f"<title>{esc(title)}</title><style>{STYLE}</style></head>"
        f'<body data-page="{page}"><header><a class="store" href="/shop">{esc(store)}</a>'
        + (f'<span class="tag">{esc(tag)}</span>' if tag else "")
        + '<a href="/shop/cart">Cart</a><a href="/shop/checkout">Checkout</a>'
        '<form action="/shop/search" method="get"><input name="q" aria-label="Search" placeholder="Search">'
        "<button>Search</button></form></header>"
        f'<main id="view">{fragment}</main><section id="checkout-forms">{forms}</section>'
        f'<script type="application/json" id="mc-storefront">{data}</script>'
        f"<script>{page_script(bare=bare)}</script></body></html>"
    )


def title_of(store: str, view: Mapping[str, Any]) -> str:
    page = str(view.get("page") or "home")
    if page == "product":
        return f"{(view.get('product') or {}).get('title') or 'Product'} | {store}"
    if page == "search":
        return f"Search: {view.get('query', '')} | {store}"
    return f"{page.capitalize()} | {store}"


def fragment(view: Mapping[str, Any], currency: str) -> str:
    """The `<main>` content of a view."""
    page = view.get("page")
    body = {
        "home": _home,
        "search": _search,
        "product": _product,
        "compare": _compare,
        "cart": _cart,
        "checkout": _checkout,
        "order": _order,
    }.get(str(page), _home)(view, currency)
    error = view.get("error")
    if isinstance(error, Mapping):
        body = f'<p class="error" role="alert">{esc(error.get("message") or error.get("code"))}</p>{body}'
    return body + _messages(view.get("messages"))


def _home(view: Mapping[str, Any], currency: str) -> str:
    return (
        f"<h1>{esc(view.get('store') or 'Welcome')}</h1>"
        "<p>Search the catalog above, or ask your assistant to shop for you.</p>"
    )


def _messages(messages: object) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    items = "".join(f"<li>{esc(m.get('content'))}</li>" for m in messages if isinstance(m, Mapping))
    return f'<ul class="messages muted">{items}</ul>'


def _text(description: object) -> str:
    if isinstance(description, Mapping):
        for form in ("plain", "markdown", "html"):
            if isinstance(description.get(form), str):
                return str(description[form])
    return "" if description is None else str(description)


def _range(product: Mapping[str, Any], currency: str) -> str:
    r = product.get("price_range") or {}
    lo, hi = (r.get("min") or {}).get("amount"), (r.get("max") or {}).get("amount")
    cur = (r.get("min") or {}).get("currency") or currency
    if lo == hi:
        return money(lo, cur)
    return f"{money(lo, cur)} to {money(hi, cur)}"


def _card(p: Mapping[str, Any], currency: str) -> str:
    pid = str(p.get("id"))
    available = any((v.get("availability") or {}).get("available") for v in p.get("variants") or ())
    text = _text(p.get("description"))
    short = text if len(text) <= 240 else text[:240].rsplit(" ", 1)[0] + "..."
    return (
        f'<article class="card" data-product-id="{esc(pid)}"><h3><a href="/shop/products/{esc(quote(pid, safe=""))}">'
        f'{esc(p.get("title"))}</a></h3><p class="price">{esc(_range(p, currency))}</p>'
        f'<p>{esc(short)}</p><p class="muted">{"In stock" if available else "Sold out"} &middot; id {esc(pid)}</p>'
        "</article>"
    )


def _search(view: Mapping[str, Any], currency: str) -> str:
    results = view.get("products") or []
    head = f'<h1>Results for &ldquo;{esc(view.get("query"))}&rdquo;</h1><p class="muted">{len(results)} shown</p>'
    more = '<p class="muted">More results on the next page.</p>' if view.get("next_cursor") else ""
    return head + "".join(_card(p, currency) for p in results) + more


def _details(product: Mapping[str, Any]) -> str:
    meta = metadata_of(product)
    rows = []
    for key in ("brand", "product_type"):
        if isinstance(meta.get(key), str):
            rows.append((key.replace("_", " ").capitalize(), meta[key]))
    for group in ("details", "attributes"):
        entries = meta.get(group)
        if isinstance(entries, Mapping):
            rows += [(str(k).replace("_", " ").capitalize(), v) for k, v in entries.items()]
    if not rows:
        return ""
    body = "".join(f"<tr><th>{esc(k)}</th><td>{esc(_flat(v))}</td></tr>" for k, v in rows)
    return f"<h2>Details</h2><table>{body}</table>"


def _flat(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(_flat(v) for v in value)
    if isinstance(value, Mapping):
        return "; ".join(f"{k}: {_flat(v)}" for k, v in value.items())
    return "" if value is None else str(value)


def _product(view: Mapping[str, Any], currency: str) -> str:
    p = view.get("product") or {}
    variants = []
    for v in p.get("variants") or ():
        price = v.get("price") or {}
        options = ", ".join(f"{o.get('name')}: {o.get('label')}" for o in v.get("options") or ())
        stock = "In stock" if (v.get("availability") or {}).get("available") else "Sold out"
        variants.append(
            f'<tr data-variant-id="{esc(v.get("id"))}"><td>{esc(v.get("title"))}</td><td>{esc(options)}</td>'
            f"<td>{esc(money(price.get('amount'), price.get('currency') or currency))}</td><td>{stock}</td>"
            f'<td class="muted">{esc(v.get("id"))}</td></tr>'
        )
    media = [m.get("url") for m in p.get("media") or () if isinstance(m, Mapping)]
    images = f'<p class="muted">{len(media)} image(s); not loaded here.</p>' if media else ""
    table = (
        "<h2>Variants</h2><table><tr><th>Variant</th><th>Options</th><th>Price</th><th>Stock</th><th>Id</th></tr>"
        + "".join(variants)
        + "</table>"
    )
    return (
        f'<h1>{esc(p.get("title"))}</h1><p class="price">{esc(_range(p, currency))}</p>'
        f"<p>{esc(_text(p.get('description')))}</p>{images}{_details(p)}{table}"
    )


def _compare(view: Mapping[str, Any], currency: str) -> str:
    products = view.get("products") or []
    head = "".join(f"<th>{esc(p.get('title'))}</th>" for p in products)
    rows = []
    for row in view.get("rows") or ():
        cells = "".join(f"<td>{esc(_shown(c.get('value'), currency))}</td>" for c in row.get("values") or ())
        rows.append(f"<tr><th>{esc(str(row.get('attribute')).replace('_', ' '))}</th>{cells}</tr>")
    alone = "".join(
        f"<li>{esc(a.get('product_id'))}: {esc(', '.join(a.get('attributes') or ()))}</li>"
        for a in view.get("stated_by_one") or ()
    )
    tail = f"<h2>Stated by one product only</h2><ul>{alone}</ul>" if alone else ""
    return f"<h1>Compare</h1><table><tr><th></th>{head}</tr>{''.join(rows)}</table>{tail}"


def _shown(value: object, currency: str) -> str:
    if isinstance(value, Mapping) and "min" in value and "max" in value:
        lo, hi = value.get("min") or {}, value.get("max") or {}
        cur = lo.get("currency") or currency
        a, b = money(lo.get("amount"), cur), money(hi.get("amount"), cur)
        return a if a == b else f"{a} to {b}"
    if value is None:
        return "-"
    return _flat(value)


def _lines(lines: Sequence[Mapping[str, Any]], currency: str) -> str:
    rows = []
    for ln in lines:
        item = ln.get("item") or {}
        qty = ln.get("quantity")
        if isinstance(qty, Mapping):
            qty = qty.get("total")
        total = next((t.get("amount") for t in ln.get("totals") or () if t.get("type") == "total"), None)
        rows.append(
            f'<tr data-line-id="{esc(ln.get("id"))}"><td>{esc(item.get("title"))}</td><td>{esc(qty)}</td>'
            f"<td>{esc(money(item.get('price'), currency))}</td><td>{esc(money(total, currency))}</td>"
            f'<td class="muted">{esc(ln.get("id"))}</td></tr>'
        )
    return (
        "<table><tr><th>Item</th><th>Qty</th><th>Price</th><th>Total</th><th>Line</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _totals(totals: object, currency: str) -> str:
    entries = totals if isinstance(totals, list) else []
    rows = "".join(
        f"<tr><th>{esc(t.get('display_text') or t.get('type'))}</th>"
        f"<td>{esc(money(t.get('amount'), currency))}</td></tr>"
        for t in entries
        if isinstance(t, Mapping)
    )
    return f'<table class="totals">{rows}</table>'


def _cart(view: Mapping[str, Any], currency: str) -> str:
    cart = view.get("cart")
    note = f'<p class="muted">{esc(view.get("note"))}</p>' if view.get("note") else ""
    if not isinstance(cart, Mapping):
        return f"<h1>Your cart</h1>{note}"
    cur = str(cart.get("currency") or currency)
    return (
        f"<h1>Your cart</h1>{note}{_lines(cart.get('line_items') or [], cur)}{_totals(cart.get('totals'), cur)}"
        f'<p class="muted">cart {esc(cart.get("id"))}</p>'
    )


def _checkout(view: Mapping[str, Any], currency: str) -> str:
    ck = view.get("checkout")
    if not isinstance(ck, Mapping):
        return "<h1>Checkout</h1><p>There is no checkout yet: add something to the cart, then check out.</p>"
    cur = str(ck.get("currency") or currency)
    buyer = ck.get("buyer") or {}
    contact = ", ".join(str(buyer[k]) for k in ("email", "phone_number") if buyer.get(k)) or "not given yet"
    delivery = ck.get("delivery") or {}
    where = "not chosen yet"
    if delivery.get("method") == "shipping":
        address = delivery.get("address") or {}
        where = "Ship to " + (", ".join(str(v) for v in address.values() if v) or "an address not given yet")
        chosen = next(
            (o for o in delivery.get("options") or () if o.get("id") == delivery.get("selected_option")), None
        )
        if chosen is not None:
            where += f" by {chosen.get('title')} ({money(chosen.get('amount'), cur)})"
    elif delivery.get("method") == "pickup":
        where = f"Pick up at {delivery.get('location') or 'a location not chosen yet'}"
    missing = "".join(
        f'<li>{esc(m.get("what"))} <span class="muted">({esc(m.get("use"))})</span></li>'
        for m in view.get("missing") or ()
    )
    order = ck.get("order") if isinstance(ck.get("order"), Mapping) else None
    placed = f"<p>Order {esc(order.get('label'))} placed.</p>" if order else ""
    return (
        f'<h1>Checkout</h1><p class="muted">checkout {esc(ck.get("id"))}: {esc(ck.get("status"))}</p>{placed}'
        f"{_lines(ck.get('line_items') or [], cur)}{_totals(ck.get('totals'), cur)}"
        f"<p>Contact: {esc(contact)}</p><p>Delivery: {esc(where)}</p>"
        + (f'<h2>Still needed</h2><ul class="missing">{missing}</ul>' if missing else "")
        + '<div id="challenge"></div>'
    )


def _order(view: Mapping[str, Any], currency: str) -> str:
    o = view.get("order") or {}
    cur = str(o.get("currency") or currency)
    return (
        f"<h1>Thank you: order {esc(o.get('label'))}</h1><p>{esc(view.get('note'))}</p>"
        f"{_lines(o.get('line_items') or [], cur)}{_totals(o.get('totals'), cur)}"
    )


# -- the declarative checkout forms ------------------------------------------------------------------------------------


def forms(tools: Sequence[ToolDef], *, country: str) -> str:
    """The checkout's declarative tools, one form each, in the order given."""
    return "<h2>Checkout details</h2>" + "".join(_form(t, country=country) for t in tools)


def _form(tool: ToolDef, *, country: str) -> str:
    fields = "".join(_field(f, country=country) for f in tool.fields)
    return (
        f'<form class="tool" toolname="{esc(tool.name)}" tooldescription="{esc(tool.description)}" toolautosubmit>'
        f'<h3>{esc(tool.title)}</h3>{fields}<button type="submit">Save {esc(tool.title.lower())}</button></form>'
    )


def _field(f: FormField, *, country: str) -> str:
    attrs = f'name="{esc(f.name)}" toolparamdescription="{esc(f.description)}"'
    if f.required:
        attrs += " required"
    if f.kind == "select":
        options = "".join(f'<option value="{esc(v)}">{esc(label)}</option>' for v, label in f.options)
        return f"<label>{esc(f.label)}<select {attrs}>{options}</select></label>"
    if f.required:
        attrs += ' minlength="1"'
    if f.max_length is not None:
        attrs += f' maxlength="{f.max_length}"'
    if f.pattern is not None:
        attrs += f' pattern="{esc(f.pattern)}"'
    if f.name == "address_country":
        attrs += f' value="{esc(country)}"'
    return f'<label>{esc(f.label)}<input type="{f.kind}" {attrs}></label>'

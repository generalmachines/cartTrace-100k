"""A replica's own website: the pages a website-path agent reads, carrying the store's schema.org JSON-LD as captured
(D67).

Why. A live store without a protocol is shopped through its pages (D31), and the page reader trusts, in order, the
theme's variant data, then JSON-LD and microdata, then OpenGraph (D36). 259 of the 307 stores the 2026-W42 capture
read carry JSON-LD, so a live website session reads the store's Product, Offer, AggregateRating and Brand blocks. A
replica that served no pages could not be shopped the same way, so live and sandbox website sessions could not be
compared. Here every replica serves its products as pages under SITE, each with:
- the JSON-LD blocks the capture kept from the store's own product page (`_jsonld` in the product's facts, reviewers'
  names redacted), verbatim, so the agent reads the same structured data the live page gave;
- the theme's variant data (`application/json`, Shopify's shape: ids, options, price in minor units, `available`)
  rendered from the sandbox state, which the reader trusts first, so a sold-out draw (D52) or a sealed fault shows on
  the page even where the captured JSON-LD says InStock;
- the title, price, availability and description as text, and an add-to-cart form the website tools never post.

Routes (all GET, read-only, logging nothing): `/site/` (a GET search form and the product links), `/site/search?q=`
(the replica's own search, as its UCP search ranks), `/site/robots.txt` (everything allowed), `/site/sitemap.xml`
(every product page), and each product at its store's own path (`/products/<handle>` on Shopify, the page address the
capture read elsewhere). The collector reaches them through carttrace.live's website surface under a reserved host name, so
the agent's tools are the live ones byte for byte (carttrace.collect.bindings).
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final
from urllib.parse import quote, urlsplit

import httpx
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, Response
from starlette.routing import Route

from carttrace.core.canon import dumps
from carttrace.core.money import Money
from carttrace.sandbox.envpack.pack import EnvPack
from carttrace.sandbox.presented import presented
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.ucp.catalog import Catalog, ProductView, VariantView
from carttrace.trace.jsonld import types_of

SITE: Final = "/site"
JSONLD: Final = "_jsonld"
"""Where a replica's product facts keep the JSON-LD blocks its page carried (carttrace.trace.capture_v2, D67)."""
SEARCH_RESULTS: Final = 24
SHOPIFY_GID: Final = "gid://shopify/"


def page_path(product: ProductView) -> str:
    """The product's page on the replica's site: the store's own path when its facts name its page, else
    /products/<handle or id>."""
    url = product.facts.get("url")
    if isinstance(url, str) and url.strip():
        path = urlsplit(url.strip()).path
        if path and path != "/":
            return path
    handle = product.facts.get("handle")
    slug = handle if isinstance(handle, str) and handle.strip() else page_id(product.id)
    return f"/products/{quote(str(slug), safe='-_.')}"


def page_id(ident: str) -> str:
    """An id as the store's page writes it: Shopify's numeric id inside its gid, anything else as it is."""
    return ident.rsplit("/", 1)[-1] if ident.startswith(SHOPIFY_GID) else ident


def jsonld_of(product: ProductView) -> list[Any]:
    blocks = product.facts.get(JSONLD)
    return list(blocks) if isinstance(blocks, list) else []


def served_types(catalog: Catalog, host: SandboxHost) -> list[str]:
    """The schema.org types every product page of the replica serves between them, for the trace header."""
    return types_of([jsonld_of(p) for p in catalog.products(host.sandbox.state)])


class Site:
    """The replica's website on one served store."""

    def __init__(self, host: SandboxHost, catalog: Catalog) -> None:
        self.host = host
        self.catalog = catalog
        self._paths: dict[str, str] | None = None

    def routes(self) -> list[Route]:
        return [
            Route(f"{SITE}/robots.txt", self._robots, methods=["GET"]),
            Route(f"{SITE}/sitemap.xml", self._sitemap, methods=["GET"]),
            Route(f"{SITE}/search", self._search, methods=["GET"]),
            Route(f"{SITE}/", self._home, methods=["GET"]),
            Route(SITE + "/{path:path}", self._product, methods=["GET"]),
        ]

    # -- reads ---------------------------------------------------------------------------------------------------------

    def products(self) -> list[ProductView]:
        return self.catalog.products(self.host.sandbox.state)

    def paths(self) -> dict[str, str]:
        """Page path to product id, built once: the catalog's products never change within a served pack."""
        if self._paths is None:
            found: dict[str, str] = {}
            for product in self.products():
                found.setdefault(page_path(product), product.id)
            self._paths = found
        return self._paths

    def product_at(self, path: str) -> ProductView | None:
        pid = self.paths().get(path) or self.paths().get(path.rstrip("/"))
        if pid is None:
            return None
        return next((p for p in self.products() if p.id == pid), None)

    # -- routes --------------------------------------------------------------------------------------------------------

    async def _robots(self, request: Request) -> Response:
        origin = _origin(request)
        return PlainTextResponse(f"User-agent: *\nAllow: /\n\nSitemap: {origin}/sitemap.xml\n")

    async def _sitemap(self, request: Request) -> Response:
        origin = _origin(request)
        urls = "".join(f"<url><loc>{html.escape(origin + path)}</loc></url>" for path in sorted(self.paths()))
        body = f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>'
        return Response(body, media_type="application/xml")

    async def _home(self, request: Request) -> Response:
        domain = html.escape(presented(self.host.pack).domain)
        links = "".join(
            f'<li><a href="{html.escape(page_path(p))}">{html.escape(p.title)}</a></li>' for p in self.products()
        )
        body = (
            f'<form action="/search" method="get" role="search"><label>Search <input type="search" name="q"></label>'
            f'<button type="submit">Search</button></form><h2>Products</h2><ul class="products">{links}</ul>'
        )
        return _page(f"{domain}", body)

    async def _search(self, request: Request) -> Response:
        query = request.query_params.get("q", "").strip()
        items = ""
        if query:
            page = self.catalog.search_page(self.host.sandbox.state, {"query": query, "pagination": {"limit": 24}})
            for doc in page.products[:SEARCH_RESULTS]:
                hit = self.catalog.find(self.host.sandbox.state, str(doc.get("id") or ""))
                if hit is None:
                    continue
                product = hit[0]
                price = _price_text(product.featured)
                items += (
                    f'<li class="product"><a href="{html.escape(page_path(product))}">{html.escape(product.title)}</a>'
                    f' <span class="price">{html.escape(price)}</span></li>'
                )
        found = f'<ul class="results">{items}</ul>' if items else "<p>No products matched your search.</p>"
        form = (
            f'<form action="/search" method="get" role="search"><input type="search" name="q" '
            f'value="{html.escape(query)}"><button type="submit">Search</button></form>'
        )
        return _page(f"Search: {query}" if query else "Search", form + found)

    async def _product(self, request: Request) -> Response:
        path = "/" + str(request.path_params.get("path") or "")
        product = self.product_at(path)
        if product is None:
            return _page("Page not found", "<p>This page does not exist.</p>", status=404)
        return HTMLResponse(product_page(product, path))


def product_page(product: ProductView, path: str) -> str:
    """One product's page: its captured JSON-LD in the head, the theme's variant data from the state, and the facts
    as a person sees them."""
    featured = product.featured
    blocks = "".join(
        f'<script type="application/ld+json">{_script_json(block)}</script>' for block in jsonld_of(product)
    )
    theme = {
        "id": page_id(product.id),
        "title": product.title,
        "handle": product.facts.get("handle") if isinstance(product.facts.get("handle"), str) else None,
        "options": list(product.option_names),
        "variants": [_theme_variant(v, product.option_names) for v in product.variants],
    }
    picker = "".join(
        f'<option value="{html.escape(page_id(v.id))}"{"" if v.available else " disabled"}>'
        f"{html.escape(v.title)}{'' if v.available else ' (sold out)'}</option>"
        for v in product.variants
    )
    stock = "In stock" if any(v.available for v in product.variants) else "Sold out"
    description = next(iter(product.description.values()), "")
    head = f'<title>{html.escape(product.title)}</title><link rel="canonical" href="{html.escape(path)}">{blocks}'
    body = (
        f'<main><h1>{html.escape(product.title)}</h1><p class="price">{html.escape(_price_text(featured))}</p>'
        f'<p class="availability">{stock}</p>'
        f'<form action="/cart/add" method="post" class="product-form"><label>Options <select name="id">{picker}'
        f'</select></label><input type="number" name="quantity" value="1" min="1">'
        f'<button type="submit" name="add">Add to cart</button></form>'
        f'<div class="description">{html.escape(_plain(description))}</div>'
        f'<script type="application/json" data-product-json>{_script_json(theme)}</script></main>'
    )
    return _document(head, body)


def _theme_variant(variant: VariantView, names: Sequence[str]) -> dict[str, Any]:
    labels = [variant.options[n] for n in names if n in variant.options] or list(variant.options.values())
    doc = variant.document or {}
    out: dict[str, Any] = {
        "id": page_id(variant.id),
        "title": variant.title,
        "price": variant.price,
        "available": variant.available,
        "options": labels,
    }
    for index, label in enumerate(labels[:3], start=1):
        out[f"option{index}"] = label
    sku = doc.get("sku") if isinstance(doc, Mapping) else None
    if isinstance(sku, str) and sku:
        out["sku"] = sku
    return out


def _price_text(variant: VariantView) -> str:
    return f"{Money(variant.price, variant.currency).major()} {variant.currency}"


def _plain(text: str) -> str:
    """Description HTML as text: tags dropped, entities read."""
    return html.unescape(re.sub(r"<[^>]+>", " ", text)).strip()


def _script_json(value: Any) -> str:
    """JSON inside a script element: `</` cannot close it."""
    return dumps(value).replace("</", "<\\/")


def _origin(request: Request) -> str:
    return f"{request.url.scheme}://{request.url.netloc}"


def _document(head: str, body: str) -> str:
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8">{head}</head><body>{body}</body></html>'


def _page(title: str, body: str, *, status: int = 200) -> Response:
    return HTMLResponse(
        _document(f"<title>{html.escape(title)}</title>", f"<main><h1>{html.escape(title)}</h1>{body}</main>"),
        status_code=status,
    )


def pack_jsonld_types(pack: EnvPack) -> list[str]:
    """The schema.org types a pack's website pages serve between them (its products' captured JSON-LD), for the
    trace header; empty for a pack whose products carry none (a synthetic catalog's)."""
    return types_of([facts.get(JSONLD) for facts in pack.snapshot.products.values() if facts.get(JSONLD)])


REWRITTEN_TYPES: Final = ("text/", "json", "xml", "javascript")


class SiteBridge(httpx.AsyncBaseTransport):
    """The replica's website addressed as https://<host>/...: each request for `host` is answered in process by the
    sandbox app under SITE, and the store's own absolute addresses in the captured JSON-LD (https://<domain>/...)
    are rewritten to the replica's host, so every address a page names is one the agent may ask for. A request for
    any other host is refused before the app sees it."""

    def __init__(self, app: Any, host: str, *, domains: Sequence[str] = ()) -> None:
        self.host = host.lower()
        self._inner = httpx.ASGITransport(app=app)
        names = {d.lower().removeprefix("www.") for d in domains if d}
        self._from = sorted({f"//{n}" for n in names} | {f"//www.{n}" for n in names}, key=len, reverse=True)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host.lower() != self.host:
            return httpx.Response(421, text=f"this bridge answers {self.host} only", request=request)
        target = request.url.copy_with(path=SITE + (request.url.path or "/"))
        forwarded = httpx.Request(request.method, target, headers=request.headers, content=request.content)
        response = await self._inner.handle_async_request(forwarded)
        body = await response.aread()
        kind = response.headers.get("content-type", "")
        if self._from and any(t in kind for t in REWRITTEN_TYPES):
            for name in self._from:
                body = body.replace(name.encode("utf-8"), f"//{self.host}".encode())
        dropped = ("content-length", "content-encoding")
        headers = [(k, v) for k, v in response.headers.items() if k.lower() not in dropped]
        return httpx.Response(response.status_code, headers=headers, content=body, request=request)

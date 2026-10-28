"""The storefront as the sandbox app serves it: pages under `/shop`, the page's API under `/shop/api`, and the WebMCP
entry beside `/.well-known/ucp`.

Why the pages are the sandbox's. A store ships WebMCP on its own storefront, so the replica's storefront is part of the
replica: `build_app` mounts it for every pack, speaking the profile's WebMCP draft (the newest vendored one when the
profile names none). It is served like any other page of the sandbox app, so a browser reaches it only through the
sandbox browser's offline routing (`carttrace.sandbox.browser.routing`) and acts for the session that names
(`MC-Sandbox-Session`), as the other surfaces do.

Routes:
- `GET /shop`, `/shop/search?q=`, `/shop/products/{id}`, `/shop/compare?ids=`, `/shop/cart`, `/shop/checkout`,
  `/shop/orders/{id}`: whole pages, read from the state (a page load logs nothing; only tools do);
- `POST /shop/api/{tool}`: one tool's work for the page's script: `{"view", "html", "title"}`, with the HTTP status of a
  protocol fault (429 when rate limited, for example) and the error inside the view.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.routing import Route

from carttrace.core.canon import dumps_bytes, loads
from carttrace.sandbox.browser.webmcp import WebMcpDraft, webmcp_draft
from carttrace.sandbox.presented import presented
from carttrace.sandbox.surfaces.access import DEFAULT_SESSION, AccessRefused, named_session
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.ucp.operations import UcpOperations
from carttrace.sandbox.surfaces.values import body_of
from carttrace.sandbox.surfaces.webmcp.html import document, forms, fragment, title_of
from carttrace.sandbox.surfaces.webmcp.operations import Answer, StorefrontOperations
from carttrace.sandbox.surfaces.webmcp.tools import DECLARATIVE, TOOLS, imperative_definitions
from carttrace.sandbox.surfaces.webmcp.views import Json, error_view

SHOP = "/shop"
API = "/shop/api"
SPEC_URL = "https://webmachinelearning.github.io/webmcp/"
JSON = "application/json"


class Storefront:
    """The replica's storefront on one served store."""

    def __init__(self, host: SandboxHost, ops: UcpOperations) -> None:
        self.host = host
        self.draft: WebMcpDraft = webmcp_draft(host.profile.webmcp)
        self.operations = StorefrontOperations(host, ops)
        shown = presented(host.pack)
        self.store = shown.store_name if shown.neutral else shown.domain
        self.tag = None if shown.neutral else "sandbox replica: nothing here is a real sale"
        declarative = [self.operations.tool(t.name) or t for t in DECLARATIVE]
        self._forms = forms([t for t in declarative if t is not None], country=_country(host))
        self._config = {
            "draft": self.draft.version,
            "tools": imperative_definitions(self.operations.handlers),
        }

    @property
    def version(self) -> str:
        return self.draft.version

    def advertisement(self, base: str) -> dict[str, Any]:
        """The storefront's entry in the sandbox's discovery document: WebMCP has no discovery of its own (tools live
        in the page), so the sandbox names the page, the draft and what the page registers."""
        return {
            "version": self.draft.version,
            "spec": SPEC_URL,
            "storefront": f"{base}{SHOP}",
            "api": "document.modelContext",
            "imperative": [t.name for t in TOOLS if t.style == "imperative"],
            "declarative": [t.name for t in DECLARATIVE],
        }

    # -- routes --------------------------------------------------------------------------------------------------------

    def routes(self) -> list[Route]:
        return [
            Route(SHOP, self._home, methods=["GET"]),
            Route(f"{SHOP}/search", self._search, methods=["GET"]),
            Route(SHOP + "/products/{product_id:path}", self._product, methods=["GET"]),
            Route(f"{SHOP}/compare", self._compare, methods=["GET"]),
            Route(f"{SHOP}/cart", self._cart, methods=["GET"]),
            Route(f"{SHOP}/checkout", self._checkout, methods=["GET"]),
            Route(SHOP + "/orders/{order_id}", self._order, methods=["GET"]),
            Route(API + "/{op}", self._api, methods=["POST"]),
        ]

    async def _home(self, request: Request) -> Response:
        return self._page(request, "home", {})

    async def _search(self, request: Request) -> Response:
        return self._page(request, "search", {"q": request.query_params.get("q", "")})

    async def _product(self, request: Request) -> Response:
        return self._page(request, "product", {"id": str(request.path_params["product_id"])})

    async def _compare(self, request: Request) -> Response:
        return self._page(request, "compare", {"ids": request.query_params.get("ids", "")})

    async def _cart(self, request: Request) -> Response:
        return self._page(request, "cart", {})

    async def _checkout(self, request: Request) -> Response:
        return self._page(request, "checkout", {})

    async def _order(self, request: Request) -> Response:
        return self._page(request, "order", {"id": str(request.path_params["order_id"])})

    def _page(self, request: Request, route: str, params: Mapping[str, str]) -> Response:
        try:
            session = named_session(request) or DEFAULT_SESSION
        except AccessRefused as e:
            text = f"<!doctype html><title>Refused</title><p>{html.escape(e.message)}</p>"
            return HTMLResponse(text, status_code=e.status)
        base = _base(request)
        view = self.operations.page(route, params, session=session, base=base)
        status = 404 if isinstance(view.get("error"), Mapping) and route in ("product", "order") else 200
        return HTMLResponse(self.render(view), status_code=status)

    def render(self, view: Json) -> str:
        return document(
            store=self.store,
            title=title_of(self.store, view),
            view=view,
            fragment=fragment(view, self.operations.currency),
            forms=self._forms,
            config=self._config,
            tag=self.tag,
            bare=self.tag is None,
        )

    async def _api(self, request: Request) -> Response:
        op = str(request.path_params["op"])
        try:
            session = named_session(request) or DEFAULT_SESSION
        except AccessRefused as e:
            return _json({"view": error_view("home", SHOP, "forbidden", e.message)}, e.status)
        raw = await body_of(request)
        if raw is None:
            return _json({"view": error_view("home", SHOP, "request_too_large", "The request is too large")}, 413)
        try:
            arguments = loads(raw) if raw.strip() else {}
        except ValueError:
            return _json({"view": error_view("home", SHOP, "invalid_json", "The request is not JSON")}, 400)
        answer = self.operations.run(op, arguments, session=session, base=_base(request))
        return self.answer(answer)

    def answer(self, answer: Answer) -> Response:
        view = answer.view
        body = {"view": view, "html": fragment(view, self.operations.currency), "title": title_of(self.store, view)}
        return _json(body, answer.status)


def _json(body: Mapping[str, Any], status: int = 200) -> Response:
    return Response(dumps_bytes(dict(body)), status_code=status, media_type=JSON)


def _base(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _country(host: SandboxHost) -> str:
    """The country a shopper in the store's market lives in: its pickup country, else the market's own code."""
    market = host.pack.market()
    return market.pickup[0].address_country if market.pickup else market.code

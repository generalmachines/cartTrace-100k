"""The sandbox as a server: one envpack served at one profile's versions, as an ASGI app, in process or on a port.

`build_app(envpack, profile=None)` is the whole store, serving exactly the protocols its profile names:
- UCP: over MCP at `/api/ucp/mcp`, the REST binding under `/api/ucp`, the business profile at `/.well-known/ucp`, and
  the WebMCP storefront under `/shop`, whose pages register the tools at the draft the profile names (or the newest
  vendored one) and which the business profile names beside its `ucp` block;
- ACP: the checkout session API under `/api/acp`, the PSP's delegate payment under `/psp`, `/.well-known/acp.json`
  where the release defines it;
- MPP: checkout payment behind HTTP 402 under `/api/mpp`;
- AP2: the merchant role (mandates in, signed carts out, payment settled) under `/api/ap2`;
- Shop Pay, where the store declares the handler or its profile adopts it (D73): the Shop Pay interface under
  `/psp/shop_pay`, which mints the wallet platform's single-use Shop Tokens;
and always the pages its documents link to, the 3-D Secure challenge page, and the localhost-only control routes under
`/_sandbox`. A request to the prefix of a protocol the store does not serve is answered 404 with what it does serve.

`open_sandbox_surface(envpack, kind=...)` hands an agent a `ToolSurface` on that store, in process through
`httpx.ASGITransport`, so a journey opens no socket; a kind the store's profile does not serve is refused plainly. The
`webmcp` kind runs the storefront's tools in the sandbox browser (`carttrace.sandbox.browser`). `serve` runs the same app
with uvicorn on `settings.sandbox.host`, for agents in another process.

Why one app per pack. A store is one world: its catalog, its profile and its fault schedule are the pack, and its
state is one `Sandbox`. Two agents that must share a world (a handed-off cart, a UCP checkout paid over MPP, a second
session racing the first for the last unit) share one app; pass it to `open_sandbox_surface(..., app=app)`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import httpx
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from carttrace.config.http import async_client
from carttrace.core.canon import dumps_bytes
from carttrace.core.clock import Clock
from carttrace.core.errors import ValidationError
from carttrace.core.registry import names, resolve
from carttrace.protocols.profile import ProtocolProfile
from carttrace.protocols.tools import SurfaceKind
from carttrace.sandbox.core.config import SHOP_PAY_NAMESPACE
from carttrace.sandbox.envpack.pack import EnvPack
from carttrace.sandbox.presented import Guard, UnreversibleName, presented, rename, unreversible
from carttrace.sandbox.surfaces.access import LoopbackOrigins
from carttrace.sandbox.surfaces.acp.dialect import dialect_for as acp_dialect
from carttrace.sandbox.surfaces.acp.rest import ACP_PATH, PSP_PATH, AcpServer
from carttrace.sandbox.surfaces.ap2.rest import AP2_PATH, Ap2Server
from carttrace.sandbox.surfaces.client import SandboxSurface, WireSink
from carttrace.sandbox.surfaces.control import control_routes
from carttrace.sandbox.surfaces.errors import UnsupportedSurface
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.mcp.endpoint import McpEndpoint
from carttrace.sandbox.surfaces.mpp.endpoint import MPP_PATH, MppServer
from carttrace.sandbox.surfaces.pages import page_routes
from carttrace.sandbox.surfaces.shop_pay import shop_pay_routes
from carttrace.sandbox.surfaces.site import Site
from carttrace.sandbox.surfaces.ucp.catalog import Catalog
from carttrace.sandbox.surfaces.ucp.envelope import MCP_PATH, REST_PATH
from carttrace.sandbox.surfaces.ucp.operations import UcpOperations
from carttrace.sandbox.surfaces.ucp.rest import rest_routes
from carttrace.sandbox.surfaces.webmcp.routes import Storefront

SURFACE_KIND = "sandbox.surface"
"""The registry kind of the agent's side of a surface (`carttrace.sandbox.PLUGINS`): ucp_mcp, ucp_rest, acp_rest, mpp, ap2
and webmcp (the storefront's tools in the sandbox browser)."""
PREFIXES: Mapping[str, tuple[str, ...]] = {
    "ucp": (REST_PATH,),
    "acp": (ACP_PATH, PSP_PATH),
    "mpp": (MPP_PATH,),
    "ap2": (AP2_PATH,),
}
"""The path prefixes each protocol is served under."""


def build_app(
    envpack: EnvPack,
    *,
    profile: ProtocolProfile | None = None,
    log_path: Path | None = None,
    clock: Clock | None = None,
) -> Starlette:
    """The sandbox store for `envpack`, speaking `profile`'s versions (the pack's own when None).

    Raises `UnsupportedVersion` for a profile that names a protocol version with no dialect (a UCP release, an ACP
    API-Version, an MPP or AP2 version), an MCP version without a streamable HTTP transport, or no protocol at all.
    `log_path` appends the event log to a file (`events.jsonl`); `clock` replaces the logical clock."""
    host = SandboxHost(envpack, profile=profile, log_path=log_path, clock=clock)
    routes: list[BaseRoute] = []
    ops: UcpOperations | None = None
    mcp: McpEndpoint | None = None
    storefront: Storefront | None = None
    if host.serves("ucp"):
        ops = UcpOperations(host)
        mcp = McpEndpoint(host, ops)
        storefront = Storefront(host, ops)
        ucp_ops, shop = ops, storefront

        async def discovery(request: Request) -> Response:
            base = str(request.base_url).rstrip("/")
            doc = ucp_ops.envelope.discovery(base)
            doc["webmcp"] = shop.advertisement(base)
            return Response(dumps_bytes(doc), media_type="application/json")

        routes += [
            Route("/.well-known/ucp", discovery, methods=["GET"]),
            Route(MCP_PATH, mcp.handle, methods=["GET", "POST", "DELETE"]),
        ]
        routes += rest_routes(ops)
        routes += storefront.routes()
    if host.serves("acp"):
        routes += AcpServer(host, acp_dialect(host.require("acp"))).routes()
    if host.serves("mpp"):
        routes += MppServer(host).routes()
    if host.serves("ap2"):
        routes += Ap2Server(host).routes()
    if SHOP_PAY_NAMESPACE in host.pack.handlers().values():
        routes += shop_pay_routes(host)
    site = Site(host, ops.catalog if ops is not None else Catalog(host.pack))
    routes += site.routes()
    routes += [*control_routes(host), *page_routes(host), *_unserved(host)]
    app = Starlette(
        routes=routes,
        middleware=[Middleware(VisibilityGuard, host=host), Middleware(LoopbackOrigins, rpc_path=MCP_PATH)],
    )
    app.state.host = host
    app.state.ops = ops
    app.state.mcp = mcp
    app.state.webmcp = storefront
    app.state.site = site
    return app


def _unserved(host: SandboxHost) -> list[Route]:
    """A plain 404 under the prefix of every protocol this store does not serve, saying what it serves."""

    async def refuse(request: Request) -> Response:
        protocol = request.scope["route"].name if "route" in request.scope else "this protocol"
        body = {
            "error": f"this store does not serve {protocol}",
            "serves": host.protocols(),
            "hint": "serve the pack under a profile that names it",
        }
        return Response(dumps_bytes(body), status_code=404, media_type="application/json")

    out = []
    for protocol, prefixes in PREFIXES.items():
        if host.serves(protocol):
            continue
        for prefix in prefixes:
            methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
            out.append(Route(f"{prefix}/{{rest:path}}", refuse, methods=methods, name=protocol.upper()))
            out.append(Route(prefix, refuse, methods=methods, name=protocol.upper()))
    return out


CONTROL_PREFIX = "/_sandbox/"
"""The harness's own routes (state, events, reset), which no agent is served; the guard leaves them alone."""


class VisibilityGuard:
    """The server's boundary for what an agent is served (D106). For a store presenting an identity of its own, the
    sandbox core's internal names a response echoes are served by their presented names (`served_names`, both ways:
    a request naming the presented one reaches the core as the internal one). Then every response is checked
    against the row's guard before a byte of it leaves: its headers and its body. A forbidden token raises
    `VisibilityLeak` in the app, so the response is never served; the guard keeps the hit, and the episode ends the
    row with it. The harness's own routes (`CONTROL_PREFIX`) pass untouched."""

    def __init__(self, app: ASGIApp, host: SandboxHost) -> None:
        self.app = app
        self.host = host

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = str(scope.get("path") or "")
        if scope["type"] != "http" or path.startswith(CONTROL_PREFIX):
            await self.app(scope, receive, send)
            return
        names = dict(presented(self.host.pack).served_names)
        guard = self.host.guard
        if not names and guard is None:
            await self.app(scope, receive, send)
            return
        if names:
            scope, receive = await _renamed_request(scope, receive, {v: k for k, v in names.items()})
        await self.app(scope, receive, _Checked(send, names, guard, path))


SENT_BODY: Final = "mc.sent_body"
"""The scope key of a request's body as the agent sent it, before the boundary renamed it: what a signature over the
request (ACP's `Signature`) was made over (D109)."""


async def _renamed_request(scope: Scope, receive: Receive, names: Mapping[str, str]) -> tuple[Scope, Receive]:
    """The request with its body read whole and every presented name in it back to the internal one."""
    body = b""
    while True:
        message = await receive()
        if message["type"] != "http.request":
            break
        body += message.get("body") or b""
        if not message.get("more_body"):
            break
    renamed = rename(body, names)
    headers = [(k, v) for k, v in scope.get("headers") or () if k.lower() != b"content-length"]
    if body or any(k.lower() == b"content-length" for k, _ in scope.get("headers") or ()):
        headers.append((b"content-length", str(len(renamed)).encode()))
    sent = False

    async def replay() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": renamed, "more_body": False}
        return await receive()

    return {**scope, "headers": headers, SENT_BODY: body}, replay


class _Checked:
    """The response's `send`, renamed and checked (see VisibilityGuard): a whole body is buffered, renamed, checked
    and sent with its length; an event stream is renamed and checked chunk by chunk."""

    def __init__(self, send: Send, names: Mapping[str, str], guard: Guard | None, path: str) -> None:
        self.send = send
        self.names = names
        self.guard = guard
        self.path = path
        self.start: Message | None = None
        self.stream = False
        self.chunks: list[bytes] = []

    async def __call__(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = list(message.get("headers") or ())
            if self.guard is not None:
                self.guard.check(b"\n".join(k + b": " + v for k, v in headers), f"{self.path} headers")
            self.stream = any(k.lower() == b"content-type" and v.startswith(b"text/event-stream") for k, v in headers)
            self.start = {**message, "headers": headers}
            return
        if message["type"] != "http.response.body":
            await self.send(message)
            return
        body = message.get("body") or b""
        if self.stream:
            await self._flush(self._renamed(body), more=bool(message.get("more_body")))
            return
        self.chunks.append(body)
        if not message.get("more_body"):
            await self._flush(self._renamed(b"".join(self.chunks)), more=False, sized=True)

    def _renamed(self, body: bytes) -> bytes:
        """`body` under its served names, once none of them is already in it (D109: else the way back in would turn
        it into an internal name the core never wrote)."""
        clash = unreversible(body, self.names)
        if clash is not None:
            raise UnreversibleName(clash, f"{self.path} body")
        return rename(body, self.names)

    async def _flush(self, body: bytes, *, more: bool, sized: bool = False) -> None:
        if self.guard is not None:
            self.guard.check(body, f"{self.path} body")
        if self.start is not None:
            start = self.start
            if sized:
                headers = [(k, v) for k, v in start["headers"] if k.lower() != b"content-length"]
                start = {**start, "headers": [*headers, (b"content-length", str(len(body)).encode())]}
            await self.send(start)
            self.start = None
        await self.send({"type": "http.response.body", "body": body, "more_body": more})


def public_origin(pack: EnvPack) -> str:
    """The origin a store's surfaces call themselves, which every link they hand the agent is built on (checkout and
    order links, policy URLs, discovery endpoints): the store's presented origin (D106), a stand-in's own `.example`
    domain, else `IN_PROCESS_URL`. Either way the request goes through the ASGI transport in process, never a
    socket."""
    return presented(pack).origin


def host_of(app: Starlette) -> SandboxHost:
    """The served store of an app `build_app` made."""
    host = getattr(app.state, "host", None)
    if not isinstance(host, SandboxHost):
        raise ValidationError("this app was not built by build_app", hint="build it with carttrace.sandbox.build_app")
    return host


async def open_sandbox_surface(
    envpack: EnvPack,
    *,
    kind: SurfaceKind = "ucp_mcp",
    recorder: WireSink | None = None,
    session_id: str | None = None,
    app: Starlette | None = None,
    profile: ProtocolProfile | None = None,
    clock: Clock | None = None,
    agent_profile: str | None = None,
    options: Mapping[str, Any] | None = None,
    guard: Guard | None = None,
) -> SandboxSurface:
    """A connected `ToolSurface` (mode "sandbox", version the profile's version of the kind's protocol) on a store
    for `envpack`.

    The surface talks HTTP to the app through `httpx.ASGITransport`: no socket, no network. `recorder` receives one
    wire record per exchange (a `TraceWriter` fits). `session_id` names the sandbox session the surface acts for (the
    one a live cart was handed off to); without it the surface acts for a session of its own. `app` shares an existing
    store between surfaces; it must serve `envpack`. `agent_profile` is the profile URL every UCP call names in
    `meta.ucp-agent.profile` (`carttrace.protocols.ucp_meta`). `options` are the kind's own keyword arguments (ACP:
    `api_version`, `sign`; AP2: `signer`; webmcp: the fields of `WebMcpOptions`, such as `challenge`, `detail_filter`,
    `on_hidden`, `limits` and `screenshots`). A kind the store's profile does not serve raises `UnsupportedSurface`."""
    served = names(SURFACE_KIND)
    if kind not in served:
        raise UnsupportedSurface(f"the sandbox does not serve {kind} surfaces yet", hint=f"use one of {served}")
    factory = resolve(SURFACE_KIND, kind)
    if app is None:
        app = build_app(envpack, profile=profile, clock=None)
    host = host_of(app)
    if guard is not None:
        host.guard = guard
    if envpack not in (host.source, host.pack):
        raise ValidationError(
            f"the app serves {host.pack.id}, not {envpack.id}", hint="open the surface on the app built for this pack"
        )
    protocol = getattr(factory, "protocol", None)
    if isinstance(protocol, str):
        host.require(protocol)
    opener = getattr(factory, "open", None)
    if not callable(opener):
        raise UnsupportedSurface(f"{SURFACE_KIND} {kind!r} is not a SandboxSurface", hint="fix carttrace.sandbox.PLUGINS")
    client = async_client(base_url=public_origin(host.pack), transport=httpx.ASGITransport(app=app))
    try:
        surface = opener(
            client,
            host,
            app.state,
            recorder=recorder,
            session_id=session_id,
            clock=clock,
            agent_profile=agent_profile,
            options=options,
        )
        if not isinstance(surface, SandboxSurface):
            raise UnsupportedSurface(f"{SURFACE_KIND} {kind!r} is not a SandboxSurface", hint="fix carttrace.sandbox.PLUGINS")
        await surface.connect()
    except BaseException:
        await client.aclose()
        raise
    return surface


def serve(
    envpack: EnvPack,
    *,
    host: str,
    port: int,
    profile: ProtocolProfile | None = None,
    log_path: Path | None = None,
) -> None:
    """Run the store on `host:port` until interrupted (uvicorn). Control routes still answer localhost only."""
    import uvicorn

    uvicorn.run(build_app(envpack, profile=profile, log_path=log_path), host=host, port=port, log_level="warning")

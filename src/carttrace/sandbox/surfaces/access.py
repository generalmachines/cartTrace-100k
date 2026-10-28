"""The harness's own levers and who may pull them: control routes, the 3-D Secure challenge page, and naming the
sandbox session a protocol session acts for. Localhost only.

Why localhost. The sandbox binds `settings.sandbox.host` (127.0.0.1 by default), and anything that resets an episode,
reads sealed state or answers a payment challenge is the environment's, never the agent's: the agent reaches the store
only through its tools. Checking the client address on every such route keeps that true even when an operator binds
the server more widely. An in-process client (`httpx.ASGITransport`) reports 127.0.0.1.

Naming a session. A live journey hands its cart to the sandbox for one session (`Sandbox.handoff(cart,
session="buyer")`), and the agent must continue as that session to see it, since carts and checkouts are isolated per
session. The harness says so with the `MC-Sandbox-Session` header: on `initialize` for MCP (the MCP session then acts
for it), and on each REST request. Without it, an MCP session acts for a session of its own and a REST or stateless
MCP caller acts for `agent`. A session's trust tier comes from the envpack (`start.sessions`, `start.default_tier`) or
from `POST /_sandbox/session`, so it is anonymous unless a test sets it.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from carttrace.core.canon import dumps_bytes
from carttrace.core.errors import MCError

SESSION_HEADER = "mc-sandbox-session"
SESSION_HEADERS = (SESSION_HEADER, "x-session-id")
"""The header names a harness may name its session by: the legacy one, and the one a stand-in's surfaces send
(visibility.yaml `identity.session_header`, D106)."""
DEFAULT_SESSION = "agent"
"""The session a caller acts for when nothing names one and the transport has no sessions (REST, MCP 2026-07-28)."""
LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


class AccessRefused(MCError):
    """A harness lever pulled by a client that may not, or with a malformed value. `status` is the HTTP answer."""

    def __init__(self, message: str, *, status: int, hint: str = "") -> None:
        super().__init__(message, hint=hint)
        self.status = status


def is_local(request: Request) -> bool:
    """True only for a loopback client; a request whose client is unknown is not local."""
    client = request.client
    if client is None:
        return False
    host = client.host.removeprefix("::ffff:")
    return host in LOCAL_HOSTS or host.startswith("127.")


def require_local(request: Request) -> None:
    if not is_local(request):
        raise AccessRefused(
            "control and challenge routes answer localhost clients only",
            status=403,
            hint="run the harness on the sandbox's own host",
        )


def named_session(request: Request) -> str | None:
    """The sandbox session the harness names for this request, or None when it names none."""
    value = next((request.headers[h] for h in SESSION_HEADERS if h in request.headers), None)
    if value is None:
        return None
    require_local(request)
    if not _SESSION_ID.match(value):
        raise AccessRefused(
            f"{SESSION_HEADER} must be 1-64 letters, digits, '_', '.', ':' or '-'",
            status=400,
            hint="name the session the envpack declares, e.g. buyer",
        )
    return value


def valid_session_id(value: object) -> bool:
    return isinstance(value, str) and bool(_SESSION_ID.match(value))


def loopback_origin(origin: str) -> bool:
    """An `Origin` whose host is the loopback interface. Only a page served from this machine may call."""
    return (urlsplit(origin).hostname or "") in LOCAL_HOSTS


class LoopbackOrigins:
    """Refuse every request a browser sends from a page that is not on the loopback interface (DNS rebinding).

    Why every route. A page on any web site can make the buyer's browser resolve its own name to 127.0.0.1 and then
    call this server with `Origin: http://that-site` and `Host: that-site`; the TCP client is then local, so the
    localhost check alone would let it reset an episode or answer a 3-D Secure challenge. MCP requires the same check
    on its endpoint (MCP transports, "Security Warning"). Clients that are not browsers send no `Origin` and are not
    affected; the MCP path is answered with a JSON-RPC error that has no id, as MCP allows.
    """

    def __init__(self, app: ASGIApp, *, rpc_path: str) -> None:
        self.app = app
        self.rpc_path = rpc_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            origin = next((v.decode("latin-1") for k, v in scope["headers"] if k == b"origin"), None)
            if origin is not None and not loopback_origin(origin):
                text = "Forbidden: this server answers pages served from this machine only"
                body = (
                    {"jsonrpc": "2.0", "error": {"code": -32600, "message": text}}
                    if scope.get("path") == self.rpc_path
                    else {"error": text, "hint": "call the sandbox from a local page or a non-browser client"}
                )
                await Response(dumps_bytes(body), status_code=403, media_type="application/json")(scope, receive, send)
                return
        await self.app(scope, receive, send)

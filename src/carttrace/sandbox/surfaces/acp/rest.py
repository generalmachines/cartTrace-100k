"""The ACP REST binding: the merchant's checkout session API under `/api/acp`, the PSP's delegate payment under `/psp`,
and `/.well-known/acp.json` where the release defines it.

What travels in headers, checked here, each refusal logged as one entry and answered with the release's `Error`:
- `Authorization: Bearer <api key>` is required by every release; the key is the store's test key
  (`acp.signing.AcpKeys`). A missing or wrong key is 401.
- `API-Version` is required by every release, and the store serves exactly its profile's version: a missing version
  or another one is 400, naming the served version in `supported_versions` where the release has the field.
- `Idempotency-Key` on a POST is required from 2026-04-17 (400 `idempotency_key_required`) and optional before; at
  most 255 characters.
- `Signature` and `Timestamp`: checked whenever a request carries a signature, and demanded only when the release's
  dialect says so (none of the vendored releases does). A signature that does not verify is 401.
- `Request-Id` and `Idempotency-Key` are echoed; a replay carries `Idempotent-Replayed: true` where the release
  defines the header.

The session a call acts for is named by the harness's `MC-Sandbox-Session` header (`access.named_session`), else
`agent`, as for UCP REST.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from carttrace.core.canon import dumps_bytes, loads, sha256_hex
from carttrace.sandbox.core.errors import CoreError, InvalidCommand, PolicyDenied
from carttrace.sandbox.platforms import medusa
from carttrace.sandbox.platforms.woocommerce import acp_discovery
from carttrace.sandbox.surfaces.access import DEFAULT_SESSION, AccessRefused, named_session
from carttrace.sandbox.surfaces.acp.dialect import AcpDialect
from carttrace.sandbox.surfaces.acp.operations import COMMANDS, AcpCall, AcpOperations, AcpReply
from carttrace.sandbox.surfaces.acp.schemas import psp_error
from carttrace.sandbox.surfaces.acp.signing import AcpKeys
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.values import body_of

JSON = "application/json"
ACP_PATH = "/api/acp"
PSP_PATH = "/psp"
DISCOVERY_PATH = "/.well-known/acp.json"
MAX_KEY = 255


@dataclass(frozen=True, slots=True)
class AcpRoute:
    """One operation of the binding: its method, its path below its API's prefix, and whether it reads a body."""

    op: str
    method: str
    path: str
    api: str
    body: bool


ROUTES: tuple[AcpRoute, ...] = (
    AcpRoute("create", "POST", "/checkout_sessions", "checkout", True),
    AcpRoute("update", "POST", "/checkout_sessions/{id}", "checkout", True),
    AcpRoute("get", "GET", "/checkout_sessions/{id}", "checkout", False),
    AcpRoute("complete", "POST", "/checkout_sessions/{id}/complete", "checkout", True),
    AcpRoute("cancel", "POST", "/checkout_sessions/{id}/cancel", "checkout", True),
    AcpRoute("delegate", "POST", "/agentic_commerce/delegate_payment", "psp", True),
)
"""The paths of the vendored OpenAPI documents (`openapi.agentic_checkout.yaml`, `openapi.delegate_payment.yaml`),
which no release moved."""


def route_path(route: AcpRoute) -> str:
    return f"{PSP_PATH if route.api == 'psp' else ACP_PATH}{route.path}"


class AcpServer:
    """The ACP store of one host at its profile's API-Version."""

    def __init__(self, host: SandboxHost, dialect: AcpDialect) -> None:
        self.host = host
        self.dialect = dialect
        self.ops = AcpOperations(host, dialect)
        self.keys = AcpKeys.for_pack(host.pack.id)

    def routes(self) -> list[Route]:
        out = [Route(route_path(r), _Handler(self, r).handle, methods=[r.method]) for r in ROUTES]
        if self.dialect.discovery or self.host.platform.dialect == "medusa":
            out.append(Route(DISCOVERY_PATH, self.discovery, methods=["GET"]))
        return out

    async def discovery(self, request: Request) -> Response:
        """The seller's discovery document (2026-04-17): the one version served, REST only, checkout and the PSP. A
        platform whose ACP plugin writes its own (WooCommerce's UCP/ACP Agent: capabilities as a list of names,
        captures/plugins-ucp-acp-agent 02) serves that one, from the recorded data (D42)."""
        base = str(request.base_url).rstrip("/")
        if self.host.platform.acp_discovery:
            return _json(200, acp_discovery(self.dialect.version, f"{base}{ACP_PATH}"))
        if self.host.platform.dialect == "medusa":
            currency = self.host.pack.snapshot.currency
            return _json(200, medusa.acp_discovery(self.dialect.version, f"{base}{ACP_PATH}", currency))
        doc = {
            "protocol": {"name": "acp", "version": self.dialect.version, "supported_versions": [self.dialect.version]},
            "api_base_url": f"{base}{ACP_PATH}",
            "transports": ["rest"],
            "capabilities": {
                "services": ["checkout", "delegate_payment"],
                "intervention_types": ["3ds"],
                "supported_currencies": [self.host.state.currency.lower()],
            },
        }
        return _json(200, doc)


class HeaderRefused(Exception):
    """A request the binding refuses on its headers or body framing, with the reply to send."""

    def __init__(self, reply: AcpReply, error: CoreError | None) -> None:
        super().__init__(reply.body.get("message"))
        self.reply = reply
        self.error = error


class _Handler:
    def __init__(self, server: AcpServer, route: AcpRoute) -> None:
        self.server = server
        self.route = route

    async def handle(self, request: Request) -> Response:
        s = self.server
        try:
            session = named_session(request) or DEFAULT_SESSION
        except AccessRefused as e:
            refusal = s.ops.render.error("forbidden" if e.status == 403 else "invalid_session", e.message)
            return _json(e.status, psp_error(refusal, s.dialect.version) if self.route.api == "psp" else refusal)
        try:
            raw = await self._checked(request)
            body = self._body(raw)
        except HeaderRefused as refused:
            if refused.error is not None:
                s.host.reject(COMMANDS[self.route.op], refused.error)
            return self._reply(request, refused.reply)
        call = AcpCall(
            op=self.route.op,
            session=session,
            body=body,
            target=request.path_params.get("id"),
            base=str(request.base_url).rstrip("/"),
            key=request.headers.get("idempotency-key") if self.route.method == "POST" else None,
            body_digest=sha256_hex(raw),
        )
        return self._reply(request, s.ops.call(call))

    async def _checked(self, request: Request) -> bytes:
        """The raw body, once every header the release requires has passed."""
        s = self.server
        render = s.ops.render
        if not s.keys.bearer_ok(request.headers.get("authorization")):
            text = "Authorization must be Bearer and the store's API key"
            raise HeaderRefused(
                AcpReply(401, render.error("unauthorized", text)), PolicyDenied(code="invalid_api_key", message=text)
            )
        version = request.headers.get("api-version")
        if version != s.dialect.version:
            code = "missing_api_version" if not version else "unsupported_api_version"
            text = f"This store serves API-Version {s.dialect.version}"
            reply = AcpReply(400, render.error(code, text, versions=True))
            raise HeaderRefused(reply, InvalidCommand(code=code, message=text, field="API-Version"))
        if self.route.method == "POST":
            self._idempotency(request)
        raw = await body_of(request) if self.route.body else b""
        if raw is None:
            text = "The request body is larger than this store reads"
            raise HeaderRefused(AcpReply(413, render.error("request_too_large", text)), None)
        self._signature(request, raw)
        return raw

    def _idempotency(self, request: Request) -> None:
        render = self.server.ops.render
        key = request.headers.get("idempotency-key")
        if key is None and self.server.dialect.idempotency.required:
            text = "Idempotency-Key header is required"
            reply = AcpReply(400, render.error("idempotency_key_required", text))
            raise HeaderRefused(reply, InvalidCommand(code="idempotency_key_required", message=text))
        if key is not None and not 0 < len(key) <= MAX_KEY:
            text = f"Idempotency-Key takes 1 to {MAX_KEY} characters"
            reply = AcpReply(400, render.error("invalid_idempotency_key", text))
            raise HeaderRefused(reply, InvalidCommand(code="invalid_idempotency_key", message=text))

    def _signature(self, request: Request, raw: bytes) -> None:
        s = self.server
        signature = request.headers.get("signature")
        if signature is None and not s.dialect.signature_required:
            return
        timestamp = request.headers.get("timestamp") or ""
        # signed over the bytes the agent sent, not the ones the server boundary renamed for the core (D109)
        sent = request.scope.get("mc.sent_body", raw)
        ok = (
            signature is not None
            and bool(timestamp)
            and s.keys.verify(signature, method=request.method, path=request.url.path, timestamp=timestamp, body=sent)
        )
        if not ok:
            code = "signature_required" if signature is None else "invalid_signature"
            text = "The request signature is missing" if signature is None else "The request signature does not verify"
            raise HeaderRefused(
                AcpReply(401, s.ops.render.error(code, text)), PolicyDenied(code=code, message=text, field="Signature")
            )

    def _body(self, raw: bytes) -> Mapping[str, Any]:
        if not raw.strip():
            return {}
        try:
            body = loads(raw)
        except ValueError:
            body = None
        if not isinstance(body, dict):
            text = "The request body must be a JSON object"
            reply = AcpReply(400, self.server.ops.render.error("invalid_json", text))
            raise HeaderRefused(reply, InvalidCommand(code="invalid_json", message=text))
        return body

    def _reply(self, request: Request, reply: AcpReply) -> Response:
        headers: dict[str, str] = {}
        request_id = request.headers.get("request-id")
        if request_id is not None:
            headers["Request-Id"] = request_id
        key = request.headers.get("idempotency-key")
        if key is not None:
            headers["Idempotency-Key"] = key
        if reply.replayed and self.server.dialect.idempotency.replayed_header:
            headers["Idempotent-Replayed"] = "true"
        if reply.retry_after is not None:
            headers["Retry-After"] = str(reply.retry_after)
        body: Mapping[str, Any] = reply.body
        if self.route.api == "psp" and reply.status >= 400 and isinstance(body, Mapping) and "code" in body:
            body = psp_error(body, self.server.dialect.version)
        return _json(reply.status, body, headers)


def _json(status: int, body: Any, headers: Mapping[str, str] | None = None) -> Response:
    return Response(dumps_bytes(body), status_code=status, media_type=JSON, headers=dict(headers or {}))

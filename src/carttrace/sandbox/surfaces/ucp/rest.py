"""The UCP REST binding, routed from the release's own OpenAPI document onto the same operations as MCP.

Why from the OpenAPI. The paths, methods, operation ids, required headers and which operations create a resource are
all in `source/services/shopping/rest.openapi.json` of each release, so the routes are read from it rather than
written out: a release that moved a path would be served at its own path without code changes. Each REST request is
turned into the same tool arguments an MCP call carries (`meta` from the `UCP-Agent` and `Idempotency-Key` headers,
`id` from the path, the body as the domain payload) and checked against the same release schema.

What REST answers. Headers the OpenAPI marks required (`UCP-Agent`, `Request-Id`, and `Idempotency-Key` on writes)
are a protocol error when missing: 400 with `{code, content}`. Business outcomes are 200 (201 when a resource was
made) with the UCP document, error responses included; protocol faults are their HTTP status with `{code, content}`
and `Retry-After` (UCP checkout REST binding, "Status Codes").
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from carttrace.core.canon import dumps_bytes, loads
from carttrace.core.errors import MCError
from carttrace.protocols.registry import registry
from carttrace.sandbox.platforms import medusa
from carttrace.sandbox.surfaces.access import DEFAULT_SESSION, AccessRefused, named_session
from carttrace.sandbox.surfaces.ucp.dialect import DIALECTS
from carttrace.sandbox.surfaces.ucp.envelope import REST_PATH
from carttrace.sandbox.surfaces.ucp.operations import UcpOperations
from carttrace.sandbox.surfaces.ucp.schemas import FEED_TOOLS
from carttrace.sandbox.surfaces.values import body_of

JSON = "application/json"
_PROFILE = re.compile(r'(?:^|,)\s*profile\s*=\s*"((?:[^"\\]|\\.)*)"')
HTTP_METHODS = ("get", "put", "post", "delete", "patch")


@dataclass(frozen=True, slots=True)
class RestOp:
    """One operation of a release's REST binding, as its OpenAPI document declares it."""

    name: str
    method: str
    path: str
    """The path template, e.g. /carts/{id}."""
    headers: tuple[str, ...]
    """Headers the OpenAPI marks required."""
    body: bool
    created: bool
    """The operation answers 201 when it makes a resource."""


@cache
def rest_operations(version: str) -> dict[str, RestOp]:
    """Every operation of the release's REST binding, by operation id (the same names as the MCP tools)."""
    doc = registry().ucp_rest_openapi(version)
    shared = doc.get("components", {}).get("parameters", {})
    out: dict[str, RestOp] = {}
    for path, item in doc.get("paths", {}).items():
        for method, op in item.items():
            if method not in HTTP_METHODS or not isinstance(op, Mapping):
                continue
            headers = [p for p in _parameters(op, item, shared) if p.get("in") == "header" and p.get("required")]
            name = str(op.get("operationId"))
            out[name] = RestOp(
                name=name,
                method=method.upper(),
                path=str(path),
                headers=tuple(str(p["name"]) for p in headers),
                body="requestBody" in op,
                created="201" in (op.get("responses") or {}),
            )
    feed = DIALECTS[version].feed_version if version in DIALECTS else None
    if feed is not None:  # a checkout-only release: its product feed at the catalog release's paths (D71)
        out.update({n: op for n, op in rest_operations(feed).items() if n in FEED_TOOLS and n not in out})
    return out


def rest_routes(ops: UcpOperations, *, prefix: str = REST_PATH) -> list[Route]:
    """One route per operation of the release's OpenAPI that the sandbox serves."""
    routes: list[Route] = []
    for name, op in rest_operations(ops.dialect.version).items():
        if name in ops.ops:
            routes.append(Route(f"{prefix}{_routed(op.path)}", RestOperation(ops, op).handle, methods=[op.method]))
    return routes


def _routed(path: str) -> str:
    """An OpenAPI path template as a Starlette route. The `{id}` parameter takes slashes: a keyed Shopify cart id is
    `gid://shopify/Cart/<token>?key=<key>`, a client sends it percent-encoded as one segment, and the server sees the
    path decoded, so a one-segment parameter never matched it and every read of a keyed cart was a 404
    (sbx-replica-0003, 2026-10-13: no REST cart row of the round was read back)."""
    return path.replace("{id}", "{id:path}")


def _parameters(op: Mapping[str, Any], item: Mapping[str, Any], shared: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out = []
    for p in [*(item.get("parameters") or ()), *(op.get("parameters") or ())]:
        ref = p.get("$ref") if isinstance(p, Mapping) else None
        resolved = shared.get(str(ref).rsplit("/", 1)[-1]) if isinstance(ref, str) else p
        if isinstance(resolved, Mapping):
            out.append(resolved)
    return out


class BadRequest(MCError):
    """A REST request the binding refuses before reading it: a missing required header, a body that is not a JSON
    object, or one too large to read. `code` is the UCP protocol error code sent back."""

    def __init__(self, code: str, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class RestOperation:
    def __init__(self, ops: UcpOperations, op: RestOp) -> None:
        self.ops = ops
        self.name = op.name
        self.required = op.headers
        self.body = op.body
        self.created = op.created

    async def handle(self, request: Request) -> Response:
        try:
            session = named_session(request) or DEFAULT_SESSION
            arguments = await self._arguments(request)
        except AccessRefused as e:
            return _protocol_error(e.status, "forbidden" if e.status == 403 else "invalid_session", e.message)
        except BadRequest as e:
            if e.code == "invalid_profile_url" and self.ops.platform.dialect == "medusa":
                wire = medusa.missing_agent(self.ops.dialect.version)
                return Response(dumps_bytes(wire.body), status_code=wire.status, media_type=JSON)
            return _protocol_error(e.status, e.code, e.message)
        reply = self.ops.call(self.name, arguments, session=session, base_url=str(request.base_url))
        if reply.fault is not None:
            f = reply.fault
            headers = {"Retry-After": str(f.retry_after)} if f.retry_after is not None else {}
            body = f.data() or {"code": f.reason, "content": f.message}
            return Response(dumps_bytes(body), status_code=f.status, media_type=JSON, headers=headers)
        status = reply.status if reply.status is not None else 201 if self.created and reply.created else 200
        return Response(dumps_bytes(reply.doc), status_code=status, media_type=JSON)

    async def _arguments(self, request: Request) -> dict[str, Any]:
        """The tool arguments a REST request carries: `meta` from its headers, `id` from its path, the body as the
        domain payload. A missing required header or a body that is not a JSON object is a 400."""
        if self.ops.platform.dialect == "medusa" and not request.headers.get("ucp-agent"):
            raise BadRequest("invalid_profile_url", "UCP-Agent header is required")  # the plugin checks it first
        missing = [h for h in self.required if not request.headers.get(h)]
        if missing:
            code = "invalid_profile_url" if missing[0] == "UCP-Agent" else "missing_header"
            raise BadRequest(code, f"{missing[0]} header is required")
        profile = parse_ucp_agent(request.headers.get("ucp-agent", ""))
        if profile is None and self.ops.platform.dialect == "medusa":
            profile = request.headers.get("ucp-agent", "")  # the plugin reads the header to name the platform only
        if profile is None:
            raise BadRequest("invalid_profile_url", 'UCP-Agent must be a structured dictionary: profile="<URL>"')
        meta: dict[str, Any] = {"ucp-agent": {"profile": profile}}
        key = request.headers.get("idempotency-key")
        if key:
            meta["idempotency-key"] = key
        arguments: dict[str, Any] = {"meta": meta}
        if "id" in request.path_params:
            arguments["id"] = str(request.path_params["id"])
        if not self.body:
            return arguments
        raw = await body_of(request)
        if raw is None:
            raise BadRequest("request_too_large", "The request body is larger than this store reads", status=413)
        try:
            payload = loads(raw) if raw.strip() else {}
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            raise BadRequest("invalid_json", "The request body must be a JSON object")
        payload_key = self.ops.book.payload_key(self.name)
        if payload_key is not None:
            arguments[payload_key] = payload
        return arguments


def parse_ucp_agent(value: str) -> str | None:
    """The profile URL of a `UCP-Agent` header (RFC 8941 dictionary: `profile="https://..."`)."""
    m = _PROFILE.search(value)
    return m.group(1).replace('\\"', '"').replace("\\\\", "\\") if m else None


def _protocol_error(status: int, code: str, content: str) -> Response:
    return Response(dumps_bytes({"code": code, "content": content}), status_code=status, media_type=JSON)

"""The sandbox's surfaces as an agent holds them: `ToolSurface`s that speak to a sandbox app in process.

This module holds the base every surface shares and the two UCP surfaces; the ACP, MPP and AP2 surfaces live beside
their servers (`acp.client`, `mpp.client`, `ap2.client`) and are chosen by kind through `carttrace.sandbox.PLUGINS`.

Why in process and over HTTP. The agent must meet the sandbox through the same wire it meets a live store on (UCP
over MCP streamable HTTP, or UCP REST), so the trace records the same kind of exchanges and the judge reads them the
same way. Routing that HTTP through `httpx.ASGITransport` keeps it inside the process: no socket is opened, nothing
can leave the machine, and the in-process host name (`sandbox.invalid`, a name reserved never to resolve) makes any
accidental real request fail at once.

What is recorded. Each HTTP exchange is one wire record (`carttrace.trace.WireRecord` fields without `seq` and `t_ms`,
which the trace writer assigns): MCP lifecycle messages as protocol `mcp` at the negotiated MCP version, tool calls
as protocol `ucp` at the profile's UCP version. Durations come from the injected clock, so a frozen clock records the
same bytes on every run. Redaction is the recorder's job (`TraceWriter.wire` redacts at write time).

What a call returns. A `ToolResult` whose `content` is the text the store sent (the UCP document as canonical JSON),
whose `structured` is the document when the transport carried it structured, and whose `error_code` is the store's
own code when the call ended in an error (`not_found`, `rate_limited`...), `throttled` for a 429, `store_error` for a
5xx and `protocol_error` for a refusal of the request itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Protocol
from urllib.parse import quote

import httpx
from mcp_types import CLIENT_CAPABILITIES_META_KEY, CLIENT_INFO_META_KEY, PROTOCOL_VERSION_META_KEY

from carttrace import __version__
from carttrace.core.canon import dumps, dumps_bytes, loads
from carttrace.core.clock import Clock, SystemClock
from carttrace.core.errors import ProtocolError
from carttrace.protocols.tools import Mode, SurfaceKind, ToolResult, ToolSpec
from carttrace.protocols.ucp_meta import (
    KEYED_TOOLS,
    PLACEHOLDER_AGENT_PROFILE,
    idempotency_key,
    requires_idempotency_key,
    schema_without_meta,
    with_meta,
)
from carttrace.sandbox.presented import presented
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.mcp.versions import McpVersion, mcp_version
from carttrace.sandbox.surfaces.ucp.envelope import MCP_PATH, REST_PATH
from carttrace.sandbox.surfaces.ucp.operations import UcpOperations
from carttrace.sandbox.surfaces.ucp.rest import rest_operations

CLIENT_INFO: Mapping[str, str] = {"name": "mc-agent", "version": __version__}
JSON = "application/json"


class WireSink(Protocol):
    """Where a surface records its exchanges. `carttrace.trace.TraceWriter` is one."""

    def wire(self, record: Mapping[str, Any]) -> Any: ...


class SandboxSurface:
    """What every sandbox surface shares: the app's host, one HTTP client, the recorder and the clock.

    Subclasses speak one protocol (`protocol`, a `ProtocolProfile` family the store must serve) and set `kind`.
    `open_sandbox_surface` builds them through `open`, which hands each the server-side objects it needs."""

    mode: Mode = "sandbox"
    kind: SurfaceKind = "ucp_mcp"
    protocol: ClassVar[str] = "ucp"

    def __init__(
        self,
        client: httpx.AsyncClient,
        host: SandboxHost,
        *,
        recorder: WireSink | None = None,
        session_id: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.client = client
        self.host = host
        self.recorder = recorder
        self.session_id = session_id
        """The sandbox session this surface acts for, when the harness names one."""
        self.clock: Clock = clock or SystemClock()
        self._calls = 0
        self._closed = False

    @classmethod
    def open(
        cls,
        client: httpx.AsyncClient,
        host: SandboxHost,
        app_state: Any,
        *,
        recorder: WireSink | None = None,
        session_id: str | None = None,
        clock: Clock | None = None,
        agent_profile: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> SandboxSurface:
        """The surface on an app's host. `app_state` is the app's `state` (server-side objects such as UCP's
        operations); `options` are this kind's own keyword arguments."""
        return cls(client, host, recorder=recorder, session_id=session_id, clock=clock, **dict(options or {}))

    @property
    def version(self) -> str:
        """The protocol version this surface speaks: the served profile's."""
        return str(getattr(self.host.profile, self.protocol))

    async def connect(self) -> None:
        """Open whatever the transport needs before the first call."""

    async def list_tools(self) -> list[ToolSpec]:
        raise NotImplementedError

    async def call(self, name: str, arguments: dict[str, Any], *, idempotency_key: str | None = None) -> ToolResult:
        """Call one tool with the protocol's metadata filled by this surface. `idempotency_key` replaces the derived
        key; it is for the harness (replaying a completion, a fault scenario), never for the agent."""
        raise NotImplementedError

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self.client.aclose()

    @property
    def session_header(self) -> str:
        """The header this surface names its session in: the store's presented one (D106)."""
        return presented(self.host.pack).session_header

    def _headers(self) -> dict[str, str]:
        if self.session_id is None:
            return {}
        return {self.session_header: self.session_id}

    def _next_key(self) -> str:
        """A deterministic idempotency key, unique per call in this surface's session; never one that looks like a
        card number (see carttrace.protocols.ucp_meta.idempotency_key)."""
        return idempotency_key(self.host.pack.id, self.kind, self.session_id, self._calls)

    async def _send(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        content: bytes | None = None,
        headers: Mapping[str, str],
        protocol: str,
        version: str,
    ) -> tuple[httpx.Response, Any, int]:
        """Send one request, record the exchange, and return (response, parsed body, latency in ms). `content` sends
        exact bytes (a signed body); otherwise `body` is sent as canonical JSON."""
        start = self.clock.monotonic()
        if content is None and body is not None:
            content = dumps_bytes(body)
        response = await self.client.request(method, path, content=content, headers=dict(headers))
        latency = max(0, round((self.clock.monotonic() - start) * 1000))
        parsed = _parse(response)
        if self.recorder is not None:
            self.recorder.wire(
                {
                    "surface": self.kind,
                    "protocol": protocol,
                    "version": version,
                    "method": method,
                    "url": str(response.request.url),
                    "request_headers": dict(response.request.headers),
                    "request_body": body,
                    "status": response.status_code,
                    "response_headers": dict(response.headers),
                    "response_body": parsed if parsed is not None else response.text or None,
                    "duration_ms": latency,
                    "error": None,
                }
            )
        return response, parsed, latency


class UcpSurface(SandboxSurface):
    """What the two UCP surfaces share: the release's operations and the `meta` the client fills on every call."""

    protocol: ClassVar[str] = "ucp"

    def __init__(
        self,
        client: httpx.AsyncClient,
        host: SandboxHost,
        ops: UcpOperations,
        *,
        recorder: WireSink | None = None,
        session_id: str | None = None,
        clock: Clock | None = None,
        profile_url: str | None = None,
    ) -> None:
        super().__init__(client, host, recorder=recorder, session_id=session_id, clock=clock)
        self.ops = ops
        self.profile_url = profile_url or PLACEHOLDER_AGENT_PROFILE
        """The agent profile every call names in `meta.ucp-agent.profile`, as the live client does."""
        self._keyed: frozenset[str] = KEYED_TOOLS

    @classmethod
    def open(
        cls,
        client: httpx.AsyncClient,
        host: SandboxHost,
        app_state: Any,
        *,
        recorder: WireSink | None = None,
        session_id: str | None = None,
        clock: Clock | None = None,
        agent_profile: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> SandboxSurface:
        return cls(
            client,
            host,
            app_state.ops,
            recorder=recorder,
            session_id=session_id,
            clock=clock,
            profile_url=agent_profile,
            **dict(options or {}),
        )

    def _idempotency(self, name: str, given: str | None, *, required: bool = False) -> str | None:
        """The harness's key when it gives one; else a deterministic key, unique per call in this surface's session,
        for the tools that need one."""
        self._calls += 1
        if given is not None:
            return given
        if name not in self._keyed and not required:
            return None
        return self._next_key()


class McpSurface(UcpSurface):
    """UCP over MCP streamable HTTP, at the version the store negotiates."""

    kind: SurfaceKind = "ucp_mcp"

    def __init__(self, *args: Any, client_version: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.requested = client_version or self.host.mcp.version
        """The MCP version this client asks for; the store answers with its own."""
        self.negotiated: McpVersion = self.host.mcp
        self.mcp_session: str | None = None
        """The store's session id, or None for a store that issues none (live Shopify): each request then names the
        sandbox session itself and carries the protocol version header, as the official client does."""
        self._connected = False
        self._next_id = 0

    async def connect(self) -> None:
        """`initialize` then `notifications/initialized` at the handshake versions; `server/discover` at 2026-07-28,
        which has no handshake."""
        if mcp_version(self.requested).modern:
            self.negotiated = mcp_version(self.requested)
            response, parsed, _ = await self._rpc("server/discover", {}, lifecycle=True)
            result = _result(parsed)
            if self.negotiated.version not in (result or {}).get("supportedVersions", ()):
                raise _refused("server/discover", response)
            return
        params = {"protocolVersion": self.requested, "capabilities": {}, "clientInfo": dict(CLIENT_INFO)}
        response, parsed, _ = await self._rpc("initialize", params, lifecycle=True)
        result = _result(parsed)
        if response.status_code != 200 or result is None:
            raise _refused("initialize", response)
        self.negotiated = mcp_version(str(result.get("protocolVersion")))
        self.mcp_session = response.headers.get("mcp-session-id")
        self._connected = True
        await self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, lifecycle=True)

    async def list_tools(self) -> list[ToolSpec]:
        response, parsed, _ = await self._rpc("tools/list", {}, lifecycle=True)
        result = _result(parsed)
        if result is None:
            raise _refused("tools/list", response)
        listed = [t for t in result.get("tools") or () if isinstance(t, Mapping)]
        self._keyed = frozenset(str(t["name"]) for t in listed if requires_idempotency_key(t.get("inputSchema") or {}))
        return [
            ToolSpec(
                name=str(t["name"]),
                description=str(t.get("description") or ""),
                input_schema=schema_without_meta(t.get("inputSchema") or {}),
                output_schema=t.get("outputSchema"),
                annotations=dict(t.get("annotations") or {}),
            )
            for t in listed
        ]

    async def call(self, name: str, arguments: dict[str, Any], *, idempotency_key: str | None = None) -> ToolResult:
        key = self._idempotency(name, idempotency_key)
        filled = with_meta(arguments, profile_url=self.profile_url, idempotency=key)
        response, parsed, latency = await self._rpc("tools/call", {"name": name, "arguments": filled})
        if isinstance(parsed, Mapping) and isinstance(parsed.get("result"), Mapping):
            result = parsed["result"]
            content = result.get("content") or []
            text = next((str(c.get("text")) for c in content if isinstance(c, Mapping) and c.get("type") == "text"), "")
            structured = result.get("structuredContent")
            is_error = bool(result.get("isError"))
            code = _document_error(structured if structured is not None else _json(text)) if is_error else None
            return ToolResult(
                content=text,
                is_error=is_error,
                error_code=(code or "store_error") if is_error else None,
                structured=structured,
                latency_ms=latency,
            )
        error = parsed.get("error") if isinstance(parsed, Mapping) else None
        shown = dumps(error) if error is not None else response.text
        return ToolResult(
            content=shown,
            is_error=True,
            error_code=_transport_error(response.status_code, error),
            latency_ms=latency,
        )

    async def aclose(self) -> None:
        if not self._closed and self.mcp_session is not None:
            await self._send(
                "DELETE",
                MCP_PATH,
                headers=self._session_headers(),
                protocol="mcp",
                version=self.negotiated.version,
            )
        await super().aclose()

    async def _rpc(
        self, method: str, params: dict[str, Any], *, lifecycle: bool = False
    ) -> tuple[httpx.Response, Any, int]:
        self._next_id += 1
        if self.negotiated.modern:
            params = {
                **params,
                "_meta": {
                    PROTOCOL_VERSION_META_KEY: self.negotiated.version,
                    CLIENT_CAPABILITIES_META_KEY: {},
                    CLIENT_INFO_META_KEY: dict(CLIENT_INFO),
                },
            }
        body = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params}
        return await self._post(body, lifecycle=lifecycle)

    async def _post(self, body: dict[str, Any], *, lifecycle: bool) -> tuple[httpx.Response, Any, int]:
        headers = {"content-type": JSON, "accept": f"{JSON}, text/event-stream", **self._session_headers()}
        if self.negotiated.modern or body.get("method") == "initialize" or self.mcp_session is None:
            headers.update(self._headers())
        protocol, version = ("mcp", self.negotiated.version) if lifecycle else ("ucp", self.version)
        return await self._send("POST", MCP_PATH, body=body, headers=headers, protocol=protocol, version=version)

    def _session_headers(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.mcp_session is not None:
            out["mcp-session-id"] = self.mcp_session
        if self.negotiated.modern or (self._connected and self.negotiated.version_header):
            out["mcp-protocol-version"] = self.negotiated.version
        return out


class RestSurface(UcpSurface):
    """UCP over the REST binding: the same tools, each sent to its OpenAPI operation."""

    kind: SurfaceKind = "ucp_rest"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._requests = 0
        self.routes = rest_operations(self.host.ucp.version)

    async def list_tools(self) -> list[ToolSpec]:
        """REST has no tool listing; the tools are the release's, the same ones MCP lists, with `meta` filled by
        this surface and so hidden from the agent."""
        tools = [t for t in self.ops.tools() if t.name in self.routes]
        self._keyed = frozenset(t.name for t in tools if requires_idempotency_key(t.input_schema))
        return [t.model_copy(update={"input_schema": schema_without_meta(t.input_schema)}) for t in tools]

    async def call(self, name: str, arguments: dict[str, Any], *, idempotency_key: str | None = None) -> ToolResult:
        op = self.routes.get(name)
        if op is None or name not in self.ops.ops:
            return ToolResult(content=f"unknown tool {name!r}", is_error=True, error_code="protocol_error")
        self._requests += 1
        path = op.path
        if "{id}" in path:
            target = arguments.get("id")
            if not isinstance(target, str) or not target:
                return ToolResult(content="this tool needs an id", is_error=True, error_code="protocol_error")
            path = path.replace("{id}", quote(target, safe=""))
        headers = {"accept": JSON, "request-id": f"req-{self._requests}", **self._headers()}
        profile = self.profile_url
        headers["ucp-agent"] = 'profile="' + profile.replace("\\", "\\\\").replace('"', '\\"') + '"'
        key = self._idempotency(name, idempotency_key, required="Idempotency-Key" in op.headers)
        if key is not None:
            headers["idempotency-key"] = key
        body = None
        if op.body:
            payload_key = self.ops.book.payload_key(name)
            payload = arguments.get(payload_key) if payload_key is not None else None
            body = payload if isinstance(payload, Mapping) else {}
            headers["content-type"] = JSON
        response, parsed, latency = await self._send(
            op.method, f"{REST_PATH}{path}", body=body, headers=headers, protocol="ucp", version=self.version
        )
        doc_error = _document_error(parsed) if isinstance(parsed, Mapping) and "ucp" in parsed else None
        failed = response.status_code >= 400
        is_error = failed or _is_error_document(parsed)
        code: str | None = None
        if failed:
            code = _transport_error(response.status_code, parsed)
        elif is_error:
            code = doc_error or "store_error"
        return ToolResult(
            content=response.text,
            is_error=is_error,
            error_code=code,
            structured=parsed if isinstance(parsed, Mapping) and "ucp" in parsed else None,
            latency_ms=latency,
        )


def _result(parsed: Any) -> Mapping[str, Any] | None:
    result = parsed.get("result") if isinstance(parsed, Mapping) else None
    return result if isinstance(result, Mapping) else None


def _refused(method: str, response: httpx.Response) -> ProtocolError:
    return ProtocolError(
        f"the sandbox refused {method} ({response.status_code}): {response.text[:200]}",
        hint="check the profile's MCP version (carttrace sandbox tools ENVPACK shows what the store serves)",
    )


def _parse(response: httpx.Response) -> Any:
    if not response.content or "json" not in response.headers.get("content-type", ""):
        return None
    try:
        return loads(response.content)
    except ValueError:
        return None


def _json(text: str) -> Any:
    try:
        return loads(text) if text else None
    except ValueError:
        return None


def _is_error_document(doc: Any) -> bool:
    ucp = doc.get("ucp") if isinstance(doc, Mapping) else None
    return isinstance(ucp, Mapping) and ucp.get("status") == "error"


def _document_error(doc: Any) -> str | None:
    """The first error message's code in a UCP document."""
    for m in (doc.get("messages") if isinstance(doc, Mapping) else None) or ():
        if isinstance(m, Mapping) and m.get("type") == "error" and isinstance(m.get("code"), str):
            return str(m["code"])
    return None


def _transport_error(status: int, error: Any) -> str:
    """A transport-level refusal as a stable code: the store's own reason where it gave one."""
    data = error.get("data") if isinstance(error, Mapping) and "data" in error else error
    reason = data.get("code") if isinstance(data, Mapping) else None
    if status == 429 or reason == "rate_limited":
        return "throttled"
    if isinstance(reason, str) and reason:
        return reason
    return "store_error" if status >= 500 else "protocol_error"

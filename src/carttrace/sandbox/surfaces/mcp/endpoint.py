"""The UCP-over-MCP endpoint: MCP streamable HTTP at the profile's MCP version, answering in JSON.

Why not the SDK's server. The official `mcp` SDK (2.3.0) is used for what it does exactly: its per-version request
models (`mcp_types.methods.validate_client_request`) check every incoming request, and its error codes are the ones
sent. Its server runtime is not used, for three reasons found in its source:
- it negotiates from a fixed table (`ServerRunner._negotiate_initialize` takes any handshake version the client asks
  for), so it cannot be a store that speaks only the profile's version;
- its session ids are `uuid4`, so recorded wire bytes would differ on every run;
- its stateful mode runs sessions in a task group that only an ASGI lifespan starts, which `httpx.ASGITransport`
  (the in-process, socket-free surface) never sends, and it shapes every pre-2026 result with the 2025-11-25 models.
This handler is the faithful alternative the build allowed: one POST endpoint, JSON responses, sessions with
deterministic ids, and per-version behaviour from `carttrace.sandbox.surfaces.mcp.versions`.

What it answers, by MCP version:
- handshake versions (2025-03-26, 2025-06-18, 2025-11-25): `initialize` opens a session (`Mcp-Session-Id`, a hash of
  the pack, the episode and a counter) and is answered with the profile's version whatever the client asked; later
  requests must carry the session id (400 without, 404 for an unknown or expired one) and, when they send
  `MCP-Protocol-Version`, the session's version (400 otherwise). Batches are taken only at 2025-03-26. DELETE ends a
  session; GET is 405, since the server opens no SSE stream.
- 2026-07-28: no handshake and no sessions. Each request names the version in its header and its `_meta` (400 with
  -32020 when they differ, -32022 for an unsupported one) and `server/discover` lists what is served.

A session belongs to the episode it was opened in: after `/_sandbox/reset` it answers 404, as an expired MCP session
does, and the client must initialize again. UCP protocol faults (rate limits, unavailability, 401, 403, 409) are a
JSON-RPC error with the matching HTTP status, and `Retry-After` when the store gave one; business outcomes are tool
results whose `structuredContent` (2025-06-18 and later) and text content hold the UCP document.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pydantic
from mcp_types import CLIENT_CAPABILITIES_META_KEY, PROTOCOL_VERSION_META_KEY, SERVER_INFO_META_KEY
from mcp_types.methods import validate_client_request
from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS, KNOWN_PROTOCOL_VERSIONS
from starlette.requests import Request
from starlette.responses import Response

from carttrace import __version__
from carttrace.core.canon import dumps, dumps_bytes, loads
from carttrace.core.ids import stable_id
from carttrace.sandbox.core.guard import safe_name
from carttrace.sandbox.presented import presented
from carttrace.sandbox.surfaces.access import DEFAULT_SESSION, AccessRefused, named_session
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.mcp.jsonrpc import (
    HEADER_MISMATCH,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    UNSUPPORTED_PROTOCOL_VERSION,
    Message,
    RequestId,
    classify,
    failure,
    result,
)
from carttrace.sandbox.surfaces.mcp.versions import McpVersion
from carttrace.sandbox.surfaces.ucp.operations import Reply, UcpOperations, UnknownTool
from carttrace.sandbox.surfaces.values import body_of, obj

SESSION_ID_HEADER = "mcp-session-id"
VERSION_HEADER = "mcp-protocol-version"
SERVER_INFO: Mapping[str, str] = {"name": "mc-sandbox", "version": __version__}
JSON = "application/json"


@dataclass(slots=True)
class McpSession:
    id: str
    core: str
    """The sandbox session its tool calls act for."""
    version: McpVersion
    episode: int
    initialized: bool = False


@dataclass(frozen=True, slots=True)
class Answer:
    """One JSON-RPC response, with the HTTP status and headers it travels with when it is answered alone."""

    body: dict[str, Any]
    status: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)


class Refused(Exception):
    """A request answered with an error before it reaches a method: the answer travels up to `handle`."""

    def __init__(self, answer: Answer) -> None:
        super().__init__(answer.body.get("error", {}).get("message", "refused"))
        self.answer = answer


class McpEndpoint:
    """The MCP endpoint of one served store. Mount `handle` for GET, POST and DELETE."""

    def __init__(self, host: SandboxHost, ops: UcpOperations) -> None:
        self.host = host
        self.ops = ops
        self.version = host.mcp
        self.sessions: dict[str, McpSession] = {}
        self._opened: dict[int, int] = {}

    async def handle(self, request: Request) -> Response:
        if request.method == "GET":
            return Response(status_code=405, headers={"Allow": "POST, DELETE"})
        try:
            if request.method == "DELETE":
                return self._delete(request)
            return await self._post(request)
        except Refused as r:
            return self._respond(r.answer)
        except AccessRefused as e:
            return self._respond(Answer(self._fail(None, INVALID_REQUEST, e.message), e.status))

    # -- transport -----------------------------------------------------------------------------------------------------

    async def _post(self, request: Request) -> Response:
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != JSON:
            raise self._refuse(None, INVALID_REQUEST, "Unsupported Media Type: send application/json", 415)
        if not _accepts_json(request.headers.get("accept")):
            raise self._refuse(None, INVALID_REQUEST, "Not Acceptable: accept application/json", 406)
        raw = await body_of(request)
        if raw is None:
            raise self._refuse(None, INVALID_REQUEST, "Payload Too Large", 413)
        try:
            body = loads(raw)
        except ValueError:
            raise self._refuse(None, PARSE_ERROR, "Parse error", 400) from None
        header = request.headers.get(VERSION_HEADER)
        if self.version.modern:
            return self._respond(self._modern(request, body, header))
        return self._handshake_era(request, body, header)

    def _handshake_era(self, request: Request, body: object, header: str | None) -> Response:
        if header is not None and header not in KNOWN_PROTOCOL_VERSIONS:
            raise self._refuse(None, INVALID_REQUEST, "Bad Request: unsupported MCP-Protocol-Version", 400)
        sid = request.headers.get(SESSION_ID_HEADER)
        if isinstance(body, list):
            return self._batch(request, body, sid)
        msg = classify(body)
        if msg.kind == "invalid":
            raise self._refuse(msg.id, INVALID_REQUEST, "Invalid Request", 400)
        if msg.kind == "request" and msg.method == "initialize":
            if sid is not None and self.stateful:
                raise self._refuse(msg.id, INVALID_REQUEST, "Bad Request: initialize opens a new session", 400)
            return self._initialize(request, msg)
        if sid is None and header is not None and header not in HANDSHAKE_PROTOCOL_VERSIONS:
            data = {"requested": header, "supported": [self.version.version]}
            raise self._refuse(msg.id, UNSUPPORTED_PROTOCOL_VERSION, "Unsupported protocol version", 400, data)
        session = self._session(sid, msg.id) if self.stateful else self._stateless(request)
        if header is not None and header != session.version.version:
            text = f"Bad Request: this session speaks MCP {session.version.version}"
            raise self._refuse(msg.id, INVALID_REQUEST, text, 400)
        return self._respond(self._one(request, msg, session.core, session))

    @property
    def stateful(self) -> bool:
        """Whether this store issues MCP session ids. Live Shopify issues none: every request stands alone, and the
        harness names the sandbox session on each (`MC-Sandbox-Session`) or acts for the default one."""
        return self.host.platform.sessions

    def _stateless(self, request: Request) -> McpSession:
        """The session a request of a store without session ids acts for: never stored, never expired."""
        core = named_session(request) or DEFAULT_SESSION
        return McpSession("", core, self.version, self.host.episode, initialized=True)

    def _batch(self, request: Request, body: list[Any], sid: str | None) -> Response:
        session = self._session(sid, None) if self.stateful else self._stateless(request)
        if not session.version.batching:
            raise self._refuse(None, INVALID_REQUEST, f"Batches are not part of MCP {session.version.version}", 400)
        if not body:
            raise self._refuse(None, INVALID_REQUEST, "Invalid Request: an empty batch", 400)
        answers = [self._one(request, classify(item), session.core, session) for item in body]
        bodies = [a.body for a in answers if a is not None]
        return Response(dumps_bytes(bodies), media_type=JSON) if bodies else Response(status_code=202)

    def _delete(self, request: Request) -> Response:
        if not self.stateful:
            return Response(status_code=405, headers={"Allow": "POST"})
        session = self._session(request.headers.get(SESSION_ID_HEADER), None)
        self.sessions.pop(session.id, None)
        return Response(status_code=200)

    def _session(self, sid: str | None, rid: RequestId | None) -> McpSession:
        """The live session a request names: 400 without an id, 404 for an unknown one or one of an older episode."""
        if sid is None:
            raise self._refuse(rid, INVALID_REQUEST, "Bad Request: Mcp-Session-Id header is required", 400)
        session = self.sessions.get(sid)
        if session is not None and session.episode != self.host.episode:
            self.sessions.pop(sid, None)
            session = None
        if session is None:
            raise self._refuse(rid, INVALID_REQUEST, "Session not found", 404)
        return session

    def _initialize(self, request: Request, msg: Message) -> Response:
        try:
            validate_client_request("initialize", self.version.version, msg.params)
        except (KeyError, pydantic.ValidationError):
            raise self._refuse(msg.id, INVALID_PARAMS, "Invalid params", 200) from None
        platform = self.host.platform
        answer: dict[str, Any] = {
            "protocolVersion": self.version.version,
            "capabilities": _plain(platform.mcp_capabilities),
            "serverInfo": self._server_info(),
        }
        if platform.mcp_instructions is not None:
            answer["instructions"] = platform.mcp_instructions
        body = result(msg.id, answer)
        if not self.stateful:
            return Response(dumps_bytes(body), media_type=JSON)
        core = named_session(request)
        episode = self.host.episode
        n = self._opened.get(episode, 0) + 1
        self._opened[episode] = n
        sid = stable_id("mcps", self.host.pack.id, episode, n)
        self.sessions[sid] = McpSession(sid, core or sid, self.version, episode)
        return Response(dumps_bytes(body), media_type=JSON, headers={"Mcp-Session-Id": sid})

    # -- 2026-07-28 ----------------------------------------------------------------------------------------------------

    def _modern(self, request: Request, body: object, header: str | None) -> Answer | None:
        if isinstance(body, list):
            raise self._refuse(None, INVALID_REQUEST, "Batches are not part of MCP 2026-07-28", 400)
        msg = classify(body)
        if msg.kind in ("notification", "response"):
            return None
        if msg.kind == "invalid":
            raise self._refuse(msg.id, INVALID_REQUEST, "Invalid Request", 400)
        params = obj(msg.params)
        meta = obj(params.get("_meta"))
        named = meta.get(PROTOCOL_VERSION_META_KEY)
        if msg.method == "initialize" or (header is not None and header == named and header != self.version.version):
            asked = header if msg.method != "initialize" else params.get("protocolVersion")
            data = {"requested": asked if isinstance(asked, str) else "", "supported": [self.version.version]}
            raise self._refuse(msg.id, UNSUPPORTED_PROTOCOL_VERSION, "Unsupported protocol version", 400, data)
        if header is None or named != header:
            text = "MCP-Protocol-Version must be sent and match _meta's protocol version"
            raise self._refuse(msg.id, HEADER_MISMATCH, text, 400)
        if not isinstance(meta.get(CLIENT_CAPABILITIES_META_KEY), Mapping):
            raise self._refuse(msg.id, INVALID_PARAMS, "Invalid params: _meta needs clientCapabilities", 200)
        return self._one(request, msg, named_session(request) or DEFAULT_SESSION, None)

    # -- methods -------------------------------------------------------------------------------------------------------

    def _one(self, request: Request, msg: Message, core: str, session: McpSession | None) -> Answer | None:
        """Answer one message. Notifications and responses get no answer; a request's error is its answer."""
        if msg.kind in ("notification", "response"):
            if session is not None and msg.method == "notifications/initialized":
                session.initialized = True
            return None
        version = session.version if session is not None else self.version
        try:
            method = self._method(msg, version)
            return self._dispatch(request, msg, method, core, version)
        except Refused as r:
            return Answer(r.answer.body)

    def _method(self, msg: Message, version: McpVersion) -> str:
        """The method a request names, checked against the version's request models (the official SDK's)."""
        if msg.kind == "invalid":
            raise self._refuse(msg.id, INVALID_REQUEST, "Invalid Request", 200)
        method = str(msg.method)
        if method == "initialize":
            raise self._refuse(msg.id, INVALID_REQUEST, "initialize cannot be sent inside a session", 200)
        try:
            validate_client_request(method, version.version, msg.params)
        except KeyError:
            raise self._refuse(msg.id, METHOD_NOT_FOUND, "Method not found", 200, safe_name(method)) from None
        except pydantic.ValidationError:
            raise self._refuse(msg.id, INVALID_PARAMS, "Invalid params", 200) from None
        return method

    def _dispatch(self, request: Request, msg: Message, method: str, core: str, version: McpVersion) -> Answer:
        if method == "tools/call":
            return self._call(request, msg, core, version)
        if method == "ping":
            return Answer(result(msg.id, {}))
        if method == "server/discover":
            return Answer(result(msg.id, self._stamp(version, self._discover())))
        if method == "tools/list":
            return Answer(result(msg.id, self._stamp(version, self._tools(version))))
        raise self._refuse(msg.id, METHOD_NOT_FOUND, "Method not found", 200, safe_name(method))

    def _tools(self, version: McpVersion) -> dict[str, Any]:
        """The tools/list result: the store's recorded list verbatim when the pack or its platform has one at these
        versions (D40), else the release's tools as `UcpOperations.tools` documents them."""
        listed = self.ops.listing
        out: dict[str, Any] = {
            "tools": [copy.deepcopy(t) for t in listed.tools] if listed is not None else self._documented(version)
        }
        if version.modern:
            out.update(cacheScope="public", ttlMs=0)
        return out

    def _documented(self, version: McpVersion) -> list[dict[str, Any]]:
        tools = []
        for t in self.ops.tools():
            entry: dict[str, Any] = {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
            if version.output_schema and t.output_schema is not None:
                entry["outputSchema"] = t.output_schema
            tools.append(entry)
        return tools

    def _discover(self) -> dict[str, Any]:
        return {
            "supportedVersions": [self.version.version],
            "capabilities": {"tools": {"listChanged": False}},
            "cacheScope": "public",
            "ttlMs": 0,
        }

    def _call(self, request: Request, msg: Message, core: str, version: McpVersion) -> Answer:
        params = msg.params or {}
        name = str(params.get("name"))
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise self._refuse(msg.id, INVALID_PARAMS, "Invalid params: arguments must be an object", 200)
        try:
            reply = self.ops.call(name, arguments, session=core, base_url=str(request.base_url))
        except UnknownTool:
            shown = safe_name(name)
            raise self._refuse(
                msg.id, INVALID_PARAMS, f"Unknown tool: {shown}" if shown else "Unknown tool", 200
            ) from None
        if reply.fault is not None:
            f = reply.fault
            headers = {"Retry-After": str(f.retry_after)} if f.retry_after is not None else {}
            return Answer(self._fail(msg.id, f.rpc_code, f.message, f.data()), f.status, headers)
        shaped = tool_result(reply, version, structured=self.host.ucp.mcp_structured)
        return Answer(result(msg.id, self._stamp(version, shaped)))

    # -- responses -----------------------------------------------------------------------------------------------------

    def _stamp(self, version: McpVersion, value: dict[str, Any]) -> dict[str, Any]:
        """2026-07-28 results carry `resultType` and the server's identity in `_meta`."""
        if not version.modern:
            return value
        return {**value, "resultType": "complete", "_meta": {SERVER_INFO_META_KEY: self._server_info()}}

    def _server_info(self) -> dict[str, str]:
        """The MCP server the store names: its platform pack's own, else its presented one (D106), else the
        sandbox's."""
        return dict(self.host.platform.server_info or presented(self.host.pack).mcp_server or SERVER_INFO)

    def _fail(self, rid: RequestId | None, code: int, message: str, data: Any = None) -> dict[str, Any]:
        return failure(rid, code, message, data, omit_unknown_id=self.version.error_id_optional)

    def _refuse(self, rid: RequestId | None, code: int, message: str, status: int, data: Any = None) -> Refused:
        return Refused(Answer(self._fail(rid, code, message, data), status))

    def _respond(self, answer: Answer | None) -> Response:
        if answer is None:
            platform = self.host.platform
            if platform.notification_body is None:
                return Response(status_code=platform.notification_status)
            return Response(platform.notification_body, status_code=platform.notification_status, media_type=JSON)
        return Response(
            dumps_bytes(answer.body), status_code=answer.status, media_type=JSON, headers=dict(answer.headers)
        )


def _plain(value: Mapping[str, Any]) -> dict[str, Any]:
    """A platform's capabilities table as plain JSON (its rows are read-only mappings)."""
    return {k: _plain(v) if isinstance(v, Mapping) else v for k, v in value.items()}


def tool_result(reply: Reply, version: McpVersion, *, structured: bool = False) -> dict[str, Any]:
    """A UCP document as an MCP tool result: canonical JSON text for every client, and `structuredContent` where the
    MCP version has it or the UCP release's binding requires it (`structured`)."""
    if reply.text is not None:
        return {"content": [{"type": "text", "text": reply.text}], "isError": True}
    doc = reply.doc or {}
    out: dict[str, Any] = {"content": [{"type": "text", "text": dumps(doc)}], "isError": reply.is_error}
    if version.structured or structured:
        out["structuredContent"] = doc
    return out


def _accepts_json(accept: str | None) -> bool:
    if accept is None:
        return True
    kinds = {part.split(";")[0].strip().lower() for part in accept.split(",")}
    return bool(kinds & {JSON, "application/*", "*/*"})

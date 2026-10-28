"""The ACP surface as an agent holds it: a `ToolSurface` (kind `acp_rest`) on a sandbox store's ACP REST binding.

The tools are the five checkout operations under the names ACP's own MCP binding gives them (2026-04-17,
`openrpc.agentic_checkout.json`), applied at every release, plus the PSP's `delegate_payment`: create, get, update,
complete and cancel a checkout session, and vault a test credential for one. Each takes `id` (the checkout session)
where the operation has one, and `payload`, the release's own request body schema with every reference inlined.

What the surface fills, never the agent: `Authorization` with the store's test API key, `API-Version`, `Request-Id`,
a deterministic `Idempotency-Key` on every POST (unique per call in this surface's session, so a replay of the episode
sends the same keys), and `Signature` with `Timestamp` when the store requires signatures or the harness asks
(`sign=True`). A harness may ask for another `api_version` to see the store refuse it.

What a call returns. `content` is the body the store sent, `structured` the parsed JSON, `is_error` true for an HTTP
error (an ACP `Error`), with `error_code` its `code`. A session whose messages carry an error (a declined payment) is a
business outcome, not a protocol error, as in UCP.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar
from urllib.parse import quote

import httpx

from carttrace.core.canon import dumps_bytes
from carttrace.core.clock import Clock, iso
from carttrace.protocols.tools import SurfaceKind, ToolResult, ToolSpec
from carttrace.sandbox.surfaces.acp.dialect import AcpDialect, dialect_for
from carttrace.sandbox.surfaces.acp.rest import ROUTES, AcpRoute, route_path
from carttrace.sandbox.surfaces.acp.schemas import Api, defines, inlined
from carttrace.sandbox.surfaces.acp.signing import AcpKeys
from carttrace.sandbox.surfaces.client import JSON, SandboxSurface, WireSink
from carttrace.sandbox.surfaces.host import SandboxHost


@dataclass(frozen=True, slots=True)
class AcpTool:
    name: str
    op: str
    payload: tuple[Api, str] | None
    description: str


TOOLS: tuple[AcpTool, ...] = (
    AcpTool(
        "create_checkout_session",
        "create",
        ("checkout", "CheckoutSessionCreateRequest"),
        "Create a checkout session from items, and optionally the buyer and the fulfillment address "
        "(ACP POST /checkout_sessions). The store answers with the authoritative session: line items, totals, "
        "fulfillment options, payment handlers and messages.",
    ),
    AcpTool(
        "get_checkout_session",
        "get",
        None,
        "Retrieve the latest state of a checkout session (ACP GET /checkout_sessions/{id}).",
    ),
    AcpTool(
        "update_checkout_session",
        "update",
        ("checkout", "CheckoutSessionUpdateRequest"),
        "Change a checkout session: its items (a full replacement), the buyer, the fulfillment address or the "
        "selected fulfillment option (ACP POST /checkout_sessions/{id}).",
    ),
    AcpTool(
        "complete_checkout_session",
        "complete",
        ("checkout", "CheckoutSessionCompleteRequest"),
        "Pay for a checkout session with a delegated payment token and place the order "
        "(ACP POST /checkout_sessions/{id}/complete). Get the token from delegate_payment first.",
    ),
    AcpTool(
        "cancel_checkout_session",
        "cancel",
        ("checkout", "CancelSessionRequest"),
        "Cancel a checkout session that is not completed (ACP POST /checkout_sessions/{id}/cancel).",
    ),
    AcpTool(
        "delegate_payment",
        "delegate",
        ("psp", "DelegatePaymentRequest"),
        "Vault the buyer's payment credential with the store's payment processor for one checkout session, within "
        "an allowance (maximum amount, currency, expiry), and get the delegated token to complete with "
        "(ACP POST /agentic_commerce/delegate_payment). The credential is the token the shopper's wallet holds, "
        "sent as payment_method.number with card_number_type network_token; card numbers are refused.",
    ),
)
ROUTE_OF: Mapping[str, AcpRoute] = {r.op: r for r in ROUTES}


class AcpSurface(SandboxSurface):
    """ACP over its REST binding, at the API-Version the store serves (or the one the harness asks for)."""

    kind: SurfaceKind = "acp_rest"
    protocol: ClassVar[str] = "acp"

    def __init__(
        self,
        client: httpx.AsyncClient,
        host: SandboxHost,
        *,
        recorder: WireSink | None = None,
        session_id: str | None = None,
        clock: Clock | None = None,
        api_version: str | None = None,
        sign: bool = False,
    ) -> None:
        super().__init__(client, host, recorder=recorder, session_id=session_id, clock=clock)
        self.served: AcpDialect = dialect_for(host.require("acp"))
        self.api_version = api_version or self.served.version
        """The API-Version this client sends; the store refuses any other than its own."""
        self.keys = AcpKeys.for_pack(host.pack.id)
        self.sign = sign or self.served.signature_required

    @property
    def version(self) -> str:
        return self.api_version

    async def list_tools(self) -> list[ToolSpec]:
        return [self._spec(t) for t in TOOLS]

    def _spec(self, tool: AcpTool) -> ToolSpec:
        route = ROUTE_OF[tool.op]
        props: dict[str, Any] = {}
        required: list[str] = []
        if "{id}" in route.path:
            props["id"] = {"type": "string", "description": "The checkout session id the store returned"}
            required.append("id")
        payload = tool.payload
        if payload is not None and defines(self.served.version, payload[0], payload[1]):
            props["payload"] = inlined(self.served.version, payload[0], payload[1])
            if tool.op != "cancel":
                required.append("payload")
        schema: dict[str, Any] = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        return ToolSpec(name=tool.name, description=tool.description, input_schema=schema)

    async def call(self, name: str, arguments: dict[str, Any], *, idempotency_key: str | None = None) -> ToolResult:
        tool = next((t for t in TOOLS if t.name == name), None)
        if tool is None:
            return ToolResult(content=f"unknown tool {name!r}", is_error=True, error_code="protocol_error")
        route = ROUTE_OF[tool.op]
        self._calls += 1
        path = signed_path = route_path(route)
        if "{id}" in path:
            target = arguments.get("id")
            if not isinstance(target, str) or not target:
                return ToolResult(content="this tool needs an id", is_error=True, error_code="protocol_error")
            path = path.replace("{id}", quote(target, safe=""))
            signed_path = signed_path.replace("{id}", target)
        payload = arguments.get("payload")
        body: Mapping[str, Any] | None = None
        if route.body and (isinstance(payload, Mapping) or tool.op != "cancel"):
            body = payload if isinstance(payload, Mapping) else {}
        content = dumps_bytes(body) if body is not None else b""
        headers = {
            "accept": JSON,
            "authorization": f"Bearer {self.keys.api_key}",
            "api-version": self.api_version,
            "request-id": f"req-{self._calls}",
            **self._headers(),
        }
        if body is not None:
            headers["content-type"] = JSON
        if route.method == "POST":
            headers["idempotency-key"] = idempotency_key or self._next_key()
        if self.sign:
            stamp = iso(self.clock.now())
            headers["timestamp"] = stamp
            headers["signature"] = self.keys.sign(method=route.method, path=signed_path, timestamp=stamp, body=content)
        response, parsed, latency = await self._send(
            route.method,
            path,
            body=body,
            content=content if body is not None else None,
            headers=headers,
            protocol="acp",
            version=self.api_version,
        )
        failed = response.status_code >= 400
        code = None
        if failed:
            reason = parsed.get("code") if isinstance(parsed, Mapping) else None
            code = str(reason) if isinstance(reason, str) and reason else _status_code(response.status_code)
        return ToolResult(
            content=response.text,
            is_error=failed,
            error_code=code,
            structured=parsed if isinstance(parsed, Mapping) else None,
            latency_ms=latency,
        )


def _status_code(status: int) -> str:
    if status == 429:
        return "throttled"
    return "store_error" if status >= 500 else "protocol_error"

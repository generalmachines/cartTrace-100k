"""The WebMCP surface as an agent holds it: a `ToolSurface` whose tools are the ones the storefront page registers,
run in the sandbox browser.

Why a surface like the others. The protocol agents (carttrace.agents) take any `ToolSurface`: they list its tools, send
their schemas to the model and call them by name. So the storefront's WebMCP tools reach the agent exactly as the page
declared them (name, description, inputSchema, annotations from `getTools()`), and a call runs the page's own execute
callback through `executeTool()`; the agent's loop does not change and never sees the page.

What a call returns. `content` is the JSON string `executeTool()` resolved with, byte for byte; `structured` is that
JSON parsed. A tool that answers with an `error` in its JSON (the draft lets tool authors report failures there) is an
error result with the storefront's code, `throttled` for a rate limit. A rejected execution is an error result named
by its DOMException: `protocol_error` for NotFoundError and DataError (no such tool, input that is not an object),
`store_error` for OperationError and AbortError (the page's tool failed, or its page unloaded). A page that hangs,
crashes or loads without end raises a `BrowserFailure`, which ends the episode as an infrastructure error (VOID).

Which stores serve it. The storefront is the UCP store's own pages (its tools run the UCP operations), so the kind's
protocol is `ucp`: a store whose profile serves no UCP has no storefront, and the kind is refused as plainly as an ACP
surface on a UCP-only store is.

Options (`open_sandbox_surface(..., kind="webmcp", options={...})`, the fields of `WebMcpOptions`):
- `challenge`: how the buyer answers a 3-D Secure page (approve or decline), from the persona;
- `detail_filter` and `on_hidden`: the detail_hidden condition. The filter is applied by the storefront to every
  catalog document before any page or tool answer is built from it, so no copy shows the hidden detail; `on_hidden`
  receives, after each call, how many statements of each detail were removed (`carttrace.collect.surfaces.HiddenHook`);
- `limits` (default: settings `sandbox.browser`), `screenshots` (a directory for a PNG of every page shown) and
  `seed` (default: the pack's).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

import httpx

from carttrace.core.clock import Clock
from carttrace.core.errors import ValidationError
from carttrace.protocols.tools import SurfaceKind, ToolResult, ToolSpec
from carttrace.sandbox.browser.environment import BrowserLimits, Environment
from carttrace.sandbox.browser.runtime import CallResult, ChallengePolicy, RuntimeInfo, SandboxBrowser
from carttrace.sandbox.surfaces.access import DEFAULT_SESSION
from carttrace.sandbox.surfaces.client import SandboxSurface, WireSink
from carttrace.sandbox.surfaces.errors import UnsupportedSurface
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.webmcp.operations import Attached, ViewFilter
from carttrace.sandbox.surfaces.webmcp.routes import Storefront

HiddenHook = Callable[[str, Mapping[str, int]], None]
"""(tool, how many statements of each hidden detail its answer lost), as `carttrace.collect.surfaces.HiddenHook`."""
DOM_ERRORS: Mapping[str, str] = {
    "NotFoundError": "protocol_error",
    "DataError": "protocol_error",
    "NotSupportedError": "protocol_error",
    "InvalidStateError": "protocol_error",
    "OperationError": "store_error",
    "AbortError": "store_error",
}
STORE_CODES: Mapping[str, str] = {"rate_limited": "throttled"}
EMPTY_SCHEMA: Mapping[str, Any] = {"type": "object", "properties": {}}


@dataclass(frozen=True, slots=True)
class WebMcpOptions:
    """The webmcp kind's own options, as `open_sandbox_surface(options={...})` names them."""

    challenge: ChallengePolicy = "approve"
    detail_filter: ViewFilter | None = None
    on_hidden: HiddenHook | None = None
    limits: BrowserLimits | None = None
    screenshots: Path | None = None
    seed: int | None = None

    @classmethod
    def parse(cls, options: Mapping[str, Any] | None) -> WebMcpOptions:
        """The options a caller passed by name, or `ValidationError` naming the fields that exist."""
        try:
            return cls(**dict(options or {}))
        except TypeError as exc:
            names = ", ".join(f.name for f in fields(cls))
            raise ValidationError(f"webmcp options: {exc}", hint=f"pass only {names}") from exc


class WebMcpSurface(SandboxSurface):
    """The storefront's WebMCP tools in the sandbox browser."""

    kind: SurfaceKind = "webmcp"
    protocol: ClassVar[str] = "ucp"

    def __init__(
        self,
        client: httpx.AsyncClient,
        host: SandboxHost,
        storefront: Storefront,
        *,
        recorder: WireSink | None = None,
        session_id: str | None = None,
        clock: Clock | None = None,
        options: WebMcpOptions | None = None,
    ) -> None:
        super().__init__(client, host, recorder=recorder, session_id=session_id, clock=clock)
        self.storefront = storefront
        self.options = options or WebMcpOptions()
        self.session = self.session_id or DEFAULT_SESSION
        """The sandbox session the page's requests act for, as a storefront's cookie names its shopper."""
        self.browser: SandboxBrowser | None = None
        self.runtime: RuntimeInfo | None = None
        self._attached: Attached | None = None

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
        """The surface on the storefront `build_app` put on the app's state. `agent_profile` is not used: the page
        names the placeholder profile in its own UCP calls, as a storefront's script would."""
        storefront = getattr(app_state, "webmcp", None)
        if not isinstance(storefront, Storefront):
            raise UnsupportedSurface(
                f"this store has no WebMCP storefront: its profile ({host.profile.id}) serves {host.describe()}",
                hint="the storefront is the UCP store's pages; serve the pack under a profile that names UCP",
            )
        return cls(
            client,
            host,
            storefront,
            recorder=recorder,
            session_id=session_id,
            clock=clock,
            options=WebMcpOptions.parse(options),
        )

    @property
    def version(self) -> str:
        """The WebMCP draft the storefront's pages speak."""
        return self.storefront.version

    async def connect(self) -> None:
        """Open the sandbox browser on the storefront's home page."""
        limits = self.options.limits
        if limits is None:
            from carttrace.config import settings

            limits = BrowserLimits.from_settings(settings())
        if self.options.detail_filter is not None:
            self._attached = self.storefront.operations.attach(self.session, self.options.detail_filter)
        self.browser = SandboxBrowser(
            self.client,
            session=self.session,
            draft=self.storefront.version,
            environment=Environment.for_pack(self.host.pack, seed=self.options.seed),
            limits=limits,
            sink=self.recorder,
            clock=self.clock,
            challenge=self.options.challenge,
            screenshots=self.options.screenshots,
            session_header=self.session_header,
        )
        try:
            self.runtime = await self.browser.open("/shop")
        except BaseException:
            self.storefront.operations.detach(self.session)
            raise
        self._report_hidden("storefront")

    async def list_tools(self) -> list[ToolSpec]:
        """The page's registered tools, as getTools() lists them."""
        return [
            ToolSpec(
                name=str(t["name"]),
                description=str(t.get("description") or ""),
                input_schema=dict(t.get("inputSchema") or EMPTY_SCHEMA),
                annotations=dict(t.get("annotations") or {}),
            )
            for t in await self._browser().tools()
        ]

    async def call(self, name: str, arguments: dict[str, Any], *, idempotency_key: str | None = None) -> ToolResult:
        """Run one of the page's tools. `idempotency_key` is not used: the page keys its own order submissions, as a
        storefront does (`carttrace.sandbox.surfaces.webmcp.operations.order_key`)."""
        self._calls += 1
        result = await self._browser().call(name, arguments)
        self._report_hidden(name)
        return tool_result(result)

    async def aclose(self) -> None:
        if self._closed:
            return
        try:
            if self.browser is not None:
                await self.browser.close()
        finally:
            self.storefront.operations.detach(self.session)
            await super().aclose()

    def _browser(self) -> SandboxBrowser:
        if self.browser is None:
            raise ValidationError("the WebMCP surface is not connected", hint="open it with open_sandbox_surface")
        return self.browser

    def _report_hidden(self, name: str) -> None:
        if self._attached is None:
            return
        counts = self._attached.take()
        if counts and self.options.on_hidden is not None:
            self.options.on_hidden(name, counts)


def tool_result(call: CallResult) -> ToolResult:
    """A page's answer as the agent reads it."""
    if call.result is None:
        code = DOM_ERRORS.get(call.error or "", "store_error")
        text = f"{call.error or 'Error'}: {call.message}" if call.message else str(call.error or "Error")
        return ToolResult(content=text, is_error=True, error_code=code, latency_ms=call.duration_ms)
    try:
        parsed = json.loads(call.result)
    except ValueError:
        parsed = None
    error = parsed.get("error") if isinstance(parsed, Mapping) else None
    if isinstance(error, Mapping):
        code = str(error.get("code") or "store_error")
        return ToolResult(
            content=call.result,
            is_error=True,
            error_code=STORE_CODES.get(code, code),
            structured=parsed,
            latency_ms=call.duration_ms,
        )
    return ToolResult(content=call.result, structured=parsed, latency_ms=call.duration_ms)

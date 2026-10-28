"""`ToolSurface`: the one shape every protocol endpoint takes when an agent talks to it.

Why one contract. The protocol agent must not care whether its tools come from a live store's UCP MCP endpoint, the
sandbox's UCP surface at an older version, an ACP checkout session, an MPP 402 endpoint, or WebMCP tools registered by
a page in the sandbox browser. Each of those is a `ToolSurface`: it lists tools with JSON Schemas and runs one call at
a time, returning a `ToolResult` that carries both what the agent sees and what the trace records. The safety
allowlist for live stores wraps a surface; it is not a different kind of surface.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

SurfaceKind = Literal["ucp_mcp", "ucp_rest", "acp_rest", "mpp", "ap2", "webmcp", "store_mcp", "website"]
"""`website` is a store's own pages, read through carttrace.live.website (D31): no protocol, GET only."""
Mode = Literal["live", "sandbox"]


class ToolSpec(BaseModel):
    """A tool as the agent sees it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    output_schema: dict[str, Any] | None = None
    annotations: dict[str, Any] = Field(default_factory=dict)
    """MCP tool annotations (readOnlyHint, destructiveHint, ...) as the surface declared them."""


class ToolResult(BaseModel):
    """One call's outcome. `content` is what the agent reads; `error_code` is set exactly when `is_error` is."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: Any
    is_error: bool = False
    error_code: str | None = None
    """A stable machine code: "harness_blocked", "throttled", "protocol_error", "store_error", "timeout", or the
    protocol's own error code."""
    structured: Any = None
    """MCP structuredContent when the surface returned it."""
    latency_ms: int = 0


@runtime_checkable
class ToolSurface(Protocol):
    """A protocol endpoint the agent can call. Implementations record their own wire exchanges."""

    @property
    def kind(self) -> SurfaceKind: ...

    @property
    def mode(self) -> Mode: ...

    @property
    def version(self) -> str:
        """The protocol version this surface speaks, from the session's ProtocolProfile."""
        ...

    async def list_tools(self) -> list[ToolSpec]: ...

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...

    async def aclose(self) -> None: ...

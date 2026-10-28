"""The MCP versions the sandbox serves over streamable HTTP, and what each one changes on the wire, as data.

Why pinned to the profile. A store speaks one MCP version (live Shopify stores negotiated 2025-11-25 on all 140
pilot episodes of 2026-10-12, var/live/delta-01/report.md section 1.1; the platform table in carttrace.sandbox.platforms
makes that the Shopify default), and the sandbox must answer as that store would: whatever version a client asks for
in `initialize`, the server answers with its own, and the client decides whether it can go on (MCP lifecycle,
"Version Negotiation"). The
official SDK negotiates from a fixed table instead (`ServerRunner._negotiate_initialize` accepts any handshake version
the client names), which is one reason this sandbox runs its own JSON-RPC handler (see `carttrace.sandbox.surfaces`).

What differs between versions:
- 2025-03-26: streamable HTTP's first version. JSON-RPC batches are allowed. Tool results carry `content` only.
- 2025-06-18: batches removed; tool results may carry `structuredContent`; clients send `MCP-Protocol-Version` on
  every request after `initialize`.
- 2025-11-25: as 2025-06-18 for everything the sandbox uses.
- 2026-07-28: no handshake and no sessions. Every request carries `_meta` with the protocol version and client
  capabilities and the `MCP-Protocol-Version` header, `server/discover` lists versions, and every result carries
  `resultType`.

2024-11-05 used the older HTTP+SSE transport, which the sandbox does not serve.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from carttrace.sandbox.surfaces.errors import UnsupportedVersion

DEFAULT_MCP = "2025-06-18"
"""The version a profile that names none is served at on a pack with no platform (the sandbox's plain dialect). A
platform names its own default (`Platform.default_mcp`): Shopify's is 2025-11-25."""


@dataclass(frozen=True, slots=True)
class McpVersion:
    version: str
    modern: bool
    """2026-07-28 and later: per-request `_meta`, `server/discover`, no `initialize`, no sessions."""
    structured: bool
    """Tool results carry `structuredContent` beside the text content."""
    batching: bool
    """A POST may carry a JSON-RPC batch."""
    error_id_optional: bool
    """An error answering a message whose id is unknown leaves `id` out (2025-11-25 and later); before that the
    schema requires an id, and JSON-RPC 2.0's `null` is sent."""
    version_header: bool
    """Clients send `MCP-Protocol-Version` on every request after `initialize` (2025-06-18 and later)."""
    output_schema: bool = True
    """A listed tool may carry `outputSchema` (2025-06-18 and later; the 2025-03-26 Tool has no such member)."""


VERSIONS: Mapping[str, McpVersion] = MappingProxyType(
    {
        "2025-03-26": McpVersion(
            "2025-03-26",
            modern=False,
            structured=False,
            batching=True,
            error_id_optional=False,
            version_header=False,
            output_schema=False,
        ),
        "2025-06-18": McpVersion(
            "2025-06-18",
            modern=False,
            structured=True,
            batching=False,
            error_id_optional=False,
            version_header=True,
        ),
        "2025-11-25": McpVersion(
            "2025-11-25",
            modern=False,
            structured=True,
            batching=False,
            error_id_optional=True,
            version_header=True,
        ),
        "2026-07-28": McpVersion(
            "2026-07-28",
            modern=True,
            structured=True,
            batching=False,
            error_id_optional=True,
            version_header=True,
        ),
    }
)


def served_mcp() -> list[str]:
    return sorted(VERSIONS)


def mcp_version(version: str | None) -> McpVersion:
    """The served MCP version a profile names (the default when it names none), or `UnsupportedVersion`."""
    found = VERSIONS.get(version or DEFAULT_MCP)
    if found is None:
        raise UnsupportedVersion(
            f"the sandbox serves no streamable HTTP transport for MCP {version!r}",
            hint=f"serve the profile at one of {served_mcp()}",
        )
    return found

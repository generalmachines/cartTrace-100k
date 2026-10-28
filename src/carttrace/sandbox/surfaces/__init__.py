"""The sandbox's protocol surfaces: UCP over MCP and UCP REST, and the payment protocols ACP, MPP and AP2, on the
state core, at the versions of one profile.

Surfaces hold no state of their own. Every tool call becomes one command on the core (`Sandbox.execute`), or one
logged refusal, and every reply is rendered from the state the command left, in the dialect of the profile's UCP
release (`ucp.dialect`). MCP and REST share the operations layer (`ucp.operations`) and differ only in how a reply
travels: a JSON-RPC tool result (or a JSON-RPC error with the HTTP status of a protocol fault), or an HTTP response.

Map:
- `ucp.dialect`: what each UCP release changes on the wire, as data; `ucp.schemas`: the release's tools and the
  argument check every call passes; `ucp.guard`: card numbers refused before anything else.
- `ucp.requests` and `ucp.render`: UCP payloads to core commands, core state to UCP documents; `ucp.catalog`: the
  catalog from the envpack snapshot and the state; `ucp.messages`: UCP messages and protocol faults; `ucp.envelope`:
  the `ucp` block and `/.well-known/ucp`.
- `ucp.operations`: the thirteen operations; `ucp.rest`: the REST binding, routed from the release's OpenAPI.
- `mcp.endpoint`: MCP streamable HTTP at the profile's MCP version; `mcp.versions`: what each MCP version changes;
  `mcp.jsonrpc`: framing.
- `acp`: the ACP checkout and delegated payment REST binding at each vendored API-Version (dialects as data in
  `acp/<version>.yaml`); `mpp`: the HTTP 402 payment flow with pympp's challenge, credential and receipt; `ap2`: AP2's
  merchant role at v0.1.0 and v0.2.0 with the parties' derived signing keys. Each has its own `client` surface.
- `keys`: test secrets derived from the envpack id, and idempotency keys spelled so no card check misreads them;
  `problems`: schema problems in plain words, shared by every protocol's argument check.
- `host`: the served store (pack, profile, `Sandbox`, episode); `access`: localhost-only levers and naming a session;
  `control`: `/_sandbox/*`; `pages`: the pages documents link to and the 3-D Secure challenge page; `client`: the
  in-process `ToolSurface`s.

Why the MCP server is a handler of its own and not the SDK's runtime: see `mcp.endpoint`. In short, the official SDK
(2.3.0) cannot pin negotiation to one version, issues uuid4 session ids, and needs an ASGI lifespan that the in-process
transport never sends; its per-version request models and error codes are used as they are.
"""

from carttrace.sandbox.presented import IN_PROCESS_URL
from carttrace.sandbox.surfaces.access import DEFAULT_SESSION, SESSION_HEADER
from carttrace.sandbox.surfaces.client import McpSurface, RestSurface, SandboxSurface, WireSink
from carttrace.sandbox.surfaces.errors import UnsupportedSurface, UnsupportedVersion
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.mcp.versions import DEFAULT_MCP, McpVersion, mcp_version, served_mcp
from carttrace.sandbox.surfaces.ucp.dialect import DIALECTS, UcpDialect, dialect_for, served_versions
from carttrace.sandbox.surfaces.ucp.operations import Reply, UcpOperations, UnknownTool

__all__ = [
    "DEFAULT_MCP",
    "DEFAULT_SESSION",
    "DIALECTS",
    "IN_PROCESS_URL",
    "SESSION_HEADER",
    "McpSurface",
    "McpVersion",
    "Reply",
    "RestSurface",
    "SandboxHost",
    "SandboxSurface",
    "UcpDialect",
    "UcpOperations",
    "UnknownTool",
    "UnsupportedSurface",
    "UnsupportedVersion",
    "WireSink",
    "dialect_for",
    "mcp_version",
    "served_mcp",
    "served_versions",
]

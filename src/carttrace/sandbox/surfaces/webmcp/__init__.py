"""The replica's WebMCP storefront: pages that register WebMCP tools, run in the sandbox browser for a text agent.

Stores ship WebMCP on their storefront before they ship a server-side agent protocol, so the benchmark measures it on
the replica of each store: server-rendered pages (home, search, product, compare, cart, checkout, order) built from the
envpack's catalog and market table, whose own script registers the tools of `tools.TOOLS` with
`document.modelContext` at the profile's WebMCP draft, imperatively and, for the checkout's forms, declaratively.
Every tool is the UCP operation, and so the core command, a UCP agent's call becomes (`operations`), so state, events
and grading are the same on both protocols.

Map:
- `tools`: the tool catalog as data; `views`: what each page shows, as JSON; `html`: how it is drawn;
- `operations`: tools to UCP operations, the page's current cart and checkout, the hidden-detail seam;
- `routes`: `Storefront`, the routes `build_app` mounts (on `app.state.webmcp`) and its discovery entry;
- `client`: `WebMcpSurface` and `WebMcpOptions`, the agent's side; `journey`: scripted journeys; `cli`: `carttrace sandbox
  webmcp`.
"""

from carttrace.sandbox.surfaces.webmcp.client import WebMcpOptions, WebMcpSurface, tool_result
from carttrace.sandbox.surfaces.webmcp.operations import Answer, Attached, StorefrontOperations, ViewFilter, order_key
from carttrace.sandbox.surfaces.webmcp.routes import API, SHOP, Storefront
from carttrace.sandbox.surfaces.webmcp.tools import BY_NAME, DECLARATIVE, TOOLS, FormField, ToolDef, form_schema

__all__ = [
    "API",
    "BY_NAME",
    "DECLARATIVE",
    "SHOP",
    "TOOLS",
    "Answer",
    "Attached",
    "FormField",
    "Storefront",
    "StorefrontOperations",
    "ToolDef",
    "ViewFilter",
    "WebMcpOptions",
    "WebMcpSurface",
    "form_schema",
    "order_key",
    "tool_result",
]

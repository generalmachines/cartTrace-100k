"""Live rows on a store's WebMCP tools (D67): which live protocol surfaces a plan names, and the row surface of each.

Why a module of its own. Live rows ran on UCP over MCP (or, for a store without a protocol, its website, D31). The
2026-10-13 scan found Shopify storefronts registering twelve WebMCP tools through Shopify's adapter, so a plan may now
name `webmcp` beside or instead of `ucp_mcp` and get one live row per task on each. The rest of the live planner is
unchanged; this keeps the new kind's rules in one place.

A live WebMCP row opens carttrace.live.open_webmcp_surface: the store's home page in headless Chromium, the page's own tools
behind live.allow_tools["webmcp"] (read, search and cart tools only), no navigation, and only Storefront API GraphQL
queries and cart line mutations on the wire. A store whose page registers no tool ends the row as `not_found`, the
store's outcome; `carttrace live webmcp-scan` says beforehand which stores register tools.
"""

from __future__ import annotations

from collections.abc import Sequence

from carttrace.collect.profiles import DEFAULT_SURFACE, LIVE_WEBMCP, LIVE_WEBMCP_VERSION, RowSurface

LIVE_PROTOCOL_KINDS = (DEFAULT_SURFACE, LIVE_WEBMCP)
"""The live surfaces a UCP store's rows may run on, in plan order."""


def live_protocol_kinds(surfaces: Sequence[str]) -> list[str]:
    """The live protocol surfaces a plan names, in canonical order: ucp_mcp, then webmcp."""
    return [kind for kind in LIVE_PROTOCOL_KINDS if kind in surfaces]


def live_row_surface(kind: str, *, ucp_version: str | None) -> RowSurface:
    """The row surface of a live row on `kind`: UCP over MCP at the store's captured release (None until captured),
    or the page's WebMCP tools."""
    if kind == LIVE_WEBMCP:
        return RowSurface(kind=LIVE_WEBMCP, version=LIVE_WEBMCP_VERSION)
    return RowSurface(kind=DEFAULT_SURFACE, version=ucp_version)


def webmcp_note(store_id: str) -> str:
    return (
        f"{store_id}: live webmcp rows run the tools its page registers in headless Chromium (D67); read, search and "
        "cart tools only, nothing past the cart"
    )

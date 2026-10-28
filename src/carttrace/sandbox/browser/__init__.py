"""The sandbox browser: headless Chromium for the replica's WebMCP storefront and its 3-D Secure pages.

The agent never sees a page (D2). This runtime loads the replica's pages, gives them the WebMCP API of the profile's
draft (Chromium's own when it conforms, else the vendored shim), lists the tools the pages register and runs them as
an in-page agent would, deterministically, offline, and within bounds.

Map:
- `runtime`: `SandboxBrowser` (open, tools, call, screenshot, close) and what it reports (`RuntimeInfo`, `CallResult`)
- `webmcp`: the drafts it hosts (`DRAFTS`, `webmcp_draft`), the shim and probe sources, the in-page agent
- `environment`: `Environment` (origin, locale, timezone, viewport, frozen time, seed) and `BrowserLimits`
- `routing`: `Relay`, the offline route handler; `slots`: the host-wide cap on open browsers
- `record`: the trace records it writes; `errors`: its typed failures (all VOID the episode)
"""

from carttrace.sandbox.browser.environment import MARKET_LOCALES, ORIGIN, BrowserLimits, Environment
from carttrace.sandbox.browser.errors import (
    BrowserFailure,
    BrowserTimeout,
    BrowserUnavailable,
    NoBrowserSlot,
    PageBudgetExceeded,
)
from carttrace.sandbox.browser.runtime import CallResult, ChallengePolicy, RuntimeInfo, SandboxBrowser
from carttrace.sandbox.browser.slots import BrowserSlots, browser_slots
from carttrace.sandbox.browser.webmcp import DEFAULT_WEBMCP, DRAFTS, WebMcpDraft, served_webmcp, webmcp_draft

__all__ = [
    "DEFAULT_WEBMCP",
    "DRAFTS",
    "MARKET_LOCALES",
    "ORIGIN",
    "BrowserFailure",
    "BrowserLimits",
    "BrowserSlots",
    "BrowserTimeout",
    "BrowserUnavailable",
    "CallResult",
    "ChallengePolicy",
    "Environment",
    "NoBrowserSlot",
    "PageBudgetExceeded",
    "RuntimeInfo",
    "SandboxBrowser",
    "WebMcpDraft",
    "browser_slots",
    "served_webmcp",
    "webmcp_draft",
]

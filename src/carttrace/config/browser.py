"""The one headless Chromium launch policy: how every part of the repository starts Playwright's Chromium.

Why one policy. Two parts drive a browser: the media export (an Excalidraw scene to SVG and PNG, carttrace.media) and the
sandbox browser (WebMCP storefront pages and 3-D Secure challenges, carttrace.sandbox.browser). Both must give the same bytes
for the same input and both must never reach the network, and before this module only the export knew how. The rules
live here, as data both pass to Playwright, so a third user gets them by asking rather than by copying:
- headless, with the GPU off: shared GPUs belong to the model servers (CLAUDE.md, "Shared GPUs");
- no background traffic of the browser's own (component updates, sync, metrics, default apps);
- every host name resolves to nothing, so a request that escaped a page's route handler still cannot leave the
  machine; Playwright's routes intercept requests before name resolution, so served origins are unaffected;
- a fixed viewport, scale factor, colour scheme, locale and timezone, and service workers blocked.

Playwright itself is imported only where a browser is started, so this module stays importable anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

NO_GPU_ARGS: tuple[str, ...] = ("--disable-gpu",)
QUIET_ARGS: tuple[str, ...] = (
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-default-apps",
    "--disable-sync",
    "--metrics-recording-only",
    "--no-first-run",
    "--mute-audio",
)
OFFLINE_ARGS: tuple[str, ...] = ("--host-resolver-rules=MAP * ~NOTFOUND",)
"""Every name fails to resolve. Playwright's routes still answer the origins a caller serves itself."""
INSTALL_HINT = "run `uv run playwright install chromium chromium-headless-shell`"


@dataclass(frozen=True, slots=True)
class Viewport:
    width: int = 1280
    height: int = 800


def launch_options(*, extra_args: Sequence[str] = ()) -> dict[str, Any]:
    """Keyword arguments for `chromium.launch` (sync or async API)."""
    return {"headless": True, "args": [*NO_GPU_ARGS, *QUIET_ARGS, *OFFLINE_ARGS, *extra_args]}


def context_options(
    *, viewport: Viewport | None = None, locale: str = "en-US", timezone: str = "UTC", scale: float = 1
) -> dict[str, Any]:
    """Keyword arguments for `browser.new_context`: everything that changes how a page lays out or formats."""
    size = viewport or Viewport()
    return {
        "viewport": {"width": size.width, "height": size.height},
        "device_scale_factor": scale,
        "locale": locale,
        "timezone_id": timezone,
        "color_scheme": "light",
        "reduced_motion": "reduce",
        "service_workers": "block",
        "accept_downloads": False,
    }


def chromium_problem() -> str | None:
    """Why headless Chromium cannot start here, or None when it can. Launches and closes it once (about 0.3 s)."""
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # the dependency is pinned, so this means a broken environment
        return f"playwright is not importable: {exc}"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_options())
            browser.close()
    except PlaywrightError as exc:
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return f"headless Chromium does not start: {first} ({INSTALL_HINT})"
    return None

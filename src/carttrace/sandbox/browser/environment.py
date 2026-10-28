"""What one episode's browser is: its limits, and everything that makes its pages give the same bytes.

Why both here. A page's output depends on more than the store's state: the viewport it lays out in, the locale and
timezone it formats in, the time `Date.now()` reads and the numbers `Math.random()` draws. The environment fixes all
of them from the envpack (the store's market gives the locale and timezone, the pack's seed the randomness, its
starting clock the time), so the same episode in the same Chromium build gives the same pages. The limits bound what
an episode may spend, so a hung or runaway page ends as an infrastructure error, never as a hang.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

from carttrace.config.browser import Viewport
from carttrace.config.schema import Settings, runner_backstop_s
from carttrace.core.clock import iso
from carttrace.sandbox.envpack.pack import EnvPack

ORIGIN = "http://localhost"
"""Where the sandbox's pages are served in the browser. A loopback name, so the page is a secure context (WebMCP is
[SecureContext]) and its `Origin` passes the sandbox's loopback check; and a name that, should a request ever escape
the route handler, could reach this machine only, never a store."""
UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
DEFAULT_TIME = datetime(2026, 9, 21, tzinfo=UTC)

MARKET_LOCALES: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "US": ("en-US", "America/New_York"),
        "CA": ("en-CA", "America/Toronto"),
        "IN": ("en-IN", "Asia/Kolkata"),
        "UK": ("en-GB", "Europe/London"),
        "EU": ("de-DE", "Europe/Berlin"),
        "AU": ("en-AU", "Australia/Sydney"),
        "NZ": ("en-NZ", "Pacific/Auckland"),
        "SG": ("en-SG", "Asia/Singapore"),
        "MX": ("es-MX", "America/Mexico_City"),
        "JP": ("ja-JP", "Asia/Tokyo"),
        "PH": ("en-PH", "Asia/Manila"),
    }
)
"""The locale and timezone a shopper in each market's browser has, keyed like `carttrace.sandbox.core.market.MARKETS`."""


BROWSER_TAIL_S = 600.0
"""What a browser must outlive the runner's backstop by: the model call in flight when the agent is stopped (up to
300 s under load), then the cart and order read back through the page."""


def episode_budget_s(settings: Settings) -> float:
    """The browser's whole-episode budget: the configured one, but never shorter than the runner's backstop
    (`runner_backstop_s`, three time caps) plus the tail.

    The agent's time cap leaves out waits for a GPU slot, so a session's wall time can reach the runner's backstop,
    not just the cap. A browser held to the time cap plus 600 s (2100 s under a 1500 s cap) voided 34 of 482 sandbox
    sessions in the hour to 2026-10-16 00:30Z, every one a Shopify replica lasting 2106 to 2709 s ("the episode's
    browser budget of N s is spent", then readback:BrowserTimeout). Before that, a 900 s budget voided 131 of the
    2026-10-15 WebMCP rows. Each page call and navigation keeps its own short timeout, so a hung page still ends
    fast."""
    backstop = runner_backstop_s(settings.collect.time_cap_s)
    return max(settings.sandbox.browser.episode_timeout_s, backstop + BROWSER_TAIL_S)


@dataclass(frozen=True, slots=True)
class BrowserLimits:
    call_timeout_s: float = 20.0
    navigation_timeout_s: float = 20.0
    episode_timeout_s: float = 900.0
    max_pages: int = 200
    max_browsers: int = 4
    slot_wait_s: float = 1800.0
    lock_dir: Path | None = None
    """Where the host-wide slot files live; None means no host-wide cap (tests that open one browser at a time)."""
    native_webmcp: bool = False
    """Launch Chromium with its own WebMCP feature on, so the probe can find and judge it."""

    @classmethod
    def from_settings(cls, settings: Settings) -> BrowserLimits:
        b = settings.sandbox.browser
        return cls(
            call_timeout_s=b.call_timeout_s,
            navigation_timeout_s=b.navigation_timeout_s,
            episode_timeout_s=episode_budget_s(settings),
            max_pages=b.max_pages,
            max_browsers=b.max_browsers,
            slot_wait_s=b.slot_wait_s,
            lock_dir=settings.paths.var / "locks" / "browser",
            native_webmcp=b.native_webmcp,
        )


@dataclass(frozen=True, slots=True)
class Environment:
    origin: str = ORIGIN
    locale: str = "en-US"
    timezone: str = "UTC"
    viewport: Viewport = field(default_factory=Viewport)
    time: datetime = DEFAULT_TIME
    """What every page's `Date.now()` reads, frozen for the whole episode."""
    seed: int = 0

    @classmethod
    def for_pack(cls, pack: EnvPack, *, seed: int | None = None) -> Environment:
        """The browser of a shopper in the pack's market, at the pack's starting time (`start.clock_start_ms`, the
        core's logical clock at the start of an episode), drawing from the pack's seed."""
        market = pack.market()
        locale, timezone = MARKET_LOCALES.get(market.code, ("en-US", "UTC"))
        start = pack.doc["start"]
        clock_ms = int(start.get("clock_start_ms") or 0)
        return cls(
            locale=locale,
            timezone=timezone,
            time=UNIX_EPOCH + timedelta(milliseconds=clock_ms) if clock_ms else DEFAULT_TIME,
            seed=int(start.get("seed") or 0) if seed is None else seed,
        )

    def record(self) -> dict[str, object]:
        """The environment as the trace records it."""
        return {
            "origin": self.origin,
            "locale": self.locale,
            "timezone": self.timezone,
            "viewport": {"width": self.viewport.width, "height": self.viewport.height},
            "time": iso(self.time),
            "seed": self.seed,
        }

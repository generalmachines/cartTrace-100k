"""The sandbox browser's failures, typed so an episode ends the right way.

Why typed. A replica page that hangs, crashes or loads without end is our infrastructure failing, not the store and
not the agent, so it must end the episode as a harness error (the judge's VOID) and never as a hang or as a tool
result the agent might read as the store's answer. Every one of these is an `ExternalServiceError` with no HTTP
status, which the protocol agents end on as `harness_error` (`carttrace.agents.tools.failure_reason`). Chromium missing on a
machine is a configuration problem and says how to install it.
"""

from __future__ import annotations

from carttrace.core.errors import ConfigError, ExternalServiceError

SERVICE = "sandbox browser"


class BrowserUnavailable(ConfigError):
    """Headless Chromium is not installed or does not start on this machine."""


class BrowserFailure(ExternalServiceError):
    """The browser or a page in it failed: it crashed, refused to load, or broke the runtime's contract."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(SERVICE, message, hint=hint or "the replica page failed; the episode is VOID, not scored")


class BrowserTimeout(BrowserFailure):
    """A tool call, a page load, or the episode's whole browser budget ran out."""


class PageBudgetExceeded(BrowserFailure):
    """The episode loaded more documents than `sandbox.browser.max_pages` allows."""


class NoBrowserSlot(BrowserFailure):
    """Every host-wide browser slot stayed taken for `sandbox.browser.slot_wait_s`."""

"""The caps that hold right now: each endpoint's host-wide cap and the collector's episode target, followed live (D114).

Why live. Infra caps the shared GPU servers by time of day (Qwen 16 by day and 56 from 22:00 to 08:00, the two decision
models 7), and a call above the cap is refused, not queued. A collector runs for days, and a restart cuts rows off
mid-run, so a running process must follow the caps without one:
- the schedule (`endpoints.<name>.schedule`, in `gpu.timezone`) is evaluated on every call against the clock, so every
  process on the host changes cap at the same minute;
- the config files are read again every `gpu.recheck_s` (60 s), so an edit to a cap or a schedule reaches a running
  process within a minute. A read that fails (a file half written, a typo) keeps the last good caps and says why in
  `last_error`; it never stops the process.

Only the caps come from a re-read: the endpoint with the same name and the same server. Everything else a process
holds keeps the settings it started with. CARTTRACE_GPU_MAX_INFLIGHT still lowers every cap, because the re-read goes through
the same loader with the same environment.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from carttrace.config.loader import load_settings
from carttrace.config.schema import Endpoint, ModelEndpoint, Settings
from carttrace.core.clock import Clock, SystemClock


class LiveCaps:
    """Caps by the clock and the config files as they are now. Thread-unsafe by design: one event loop reads it."""

    def __init__(
        self,
        settings: Settings,
        *,
        clock: Clock | None = None,
        reload: Callable[[], Settings] | None = None,
        recheck_s: float | None = None,
    ) -> None:
        self.clock = clock or SystemClock()
        self._settings = settings
        self._reload = reload
        self.recheck_s = settings.gpu.recheck_s if recheck_s is None else recheck_s
        self._checked = self.clock.monotonic()
        self.last_error: str | None = None
        """Why the last re-read failed, or None when it worked (or none ran yet)."""

    @classmethod
    def from_disk(cls, settings: Settings, *, clock: Clock | None = None) -> LiveCaps:
        """Caps that re-read config.json and config.local.json at settings.home, under this process's environment."""
        return cls(settings, clock=clock, reload=lambda: load_settings(settings.home, dict(os.environ)))

    @property
    def settings(self) -> Settings:
        """The settings the caps come from, read again when `recheck_s` has passed."""
        if self._reload is not None and self.clock.monotonic() - self._checked >= self.recheck_s:
            self._checked = self.clock.monotonic()
            try:
                self._settings = self._reload()
            except Exception as exc:  # a bad edit must not stop collection: keep the last good caps
                self.last_error = f"{type(exc).__name__}: {exc}"[:300]
            else:
                self.last_error = None
        return self._settings

    def endpoint(self, endpoint: Endpoint | ModelEndpoint) -> Endpoint:
        """The endpoint as configured now: the one of the same name on the same server, else `endpoint` itself."""
        name = endpoint.name if isinstance(endpoint, ModelEndpoint) else ""
        current = self.settings.endpoints.get(name) if name else None
        if current is None or current.endpoint_id != endpoint.endpoint_id:
            return endpoint
        return current

    def cap(self, endpoint: Endpoint | ModelEndpoint) -> int:
        """The endpoint's host-wide cap right now."""
        settings = self.settings
        return settings.cap_now(self.endpoint(endpoint), self.clock.now())

    def collect_target(self) -> int:
        """Sandbox episodes the collector keeps in flight right now."""
        return self.settings.collect_target(self.clock.now())

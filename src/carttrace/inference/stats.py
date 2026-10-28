"""Per-process counters for every model call, per endpoint and role: what the runner copies into the status file.

Why here. Every model call passes through this package, so this is the one place that sees all of them: calls,
errors, tokens in and out, latency, and the model that actually answered. The runner reads `snapshot()` and writes
it into var/status/collector.json, and the watch alerts from that file (idle compute, a failing endpoint). This
module never imports carttrace.collect; the snapshot's names follow the status file's EndpointLoad so the copy is direct.
How many slots other processes hold is not known here: that comes from `GpuSlots.in_flight()`.

Counted per HTTP request to a model server, retries excluded: one chat turn is one call, a Clef score over two labels
is one call and over N > 2 labels N calls (plus a /tokenize call the first time each label is seen). Rates (calls per
minute, tokens per second) cover the last minute, or the time since the counters started when that is shorter. Error
counts and latency percentiles cover the last ten minutes. Time comes from an injected Clock, so tests freeze it.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from carttrace.config.schema import ModelEndpoint
from carttrace.core.clock import Clock, SystemClock

RATE_WINDOW_S = 60.0
ERROR_WINDOW_S = 600.0


class EndpointStats(BaseModel):
    """One endpoint and role, as of one snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint_id: str
    role: str
    endpoint: str
    """The endpoint's name in config.json."""
    cap: int
    """The endpoint's max_inflight."""
    calls: int
    """Since the counters started."""
    errors: int
    tokens_in: int
    tokens_out: int
    calls_per_min: float
    tokens_in_per_s: float
    tokens_out_per_s: float
    errors_last_10m: int
    p50_latency_ms: int | None
    p95_latency_ms: int | None
    served_models: list[str] = Field(default_factory=list)
    """Every model id the server answered with, sorted. More than one means an alias moved."""
    last_error: str | None = None
    refusals: int = 0
    """Calls the server refused as over its parallel limit, since the counters started; each was retried (D114)."""


@dataclass(frozen=True, slots=True)
class _Event:
    t: float
    ok: bool
    latency_ms: int
    tokens_in: int
    tokens_out: int


@dataclass(slots=True)
class _Series:
    endpoint: str
    cap: int
    calls: int = 0
    errors: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    last_error: str | None = None
    refusals: int = 0
    served: set[str] = field(default_factory=set)
    events: deque[_Event] = field(default_factory=deque)


class InferenceStats:
    """Thread-safe counters. One per process (`process_stats()`); tests make their own with a FrozenClock."""

    def __init__(self, clock: Clock | None = None) -> None:
        self.clock = clock or SystemClock()
        self._lock = threading.Lock()
        self._series: dict[tuple[str, str], _Series] = {}
        self._started = self.clock.monotonic()

    def record(
        self,
        endpoint: ModelEndpoint,
        *,
        ok: bool,
        latency_ms: int,
        usage: Mapping[str, int] | None = None,
        served_model: str | None = None,
        error: str | None = None,
    ) -> None:
        usage = usage or {}
        tokens_in = int(usage.get("prompt_tokens", 0))
        tokens_out = int(usage.get("completion_tokens", 0))
        now = self.clock.monotonic()
        with self._lock:
            key = (endpoint.endpoint_id, endpoint.role)
            series = self._series.get(key)
            if series is None:
                series = self._series[key] = _Series(endpoint=endpoint.name, cap=endpoint.max_inflight)
            series.cap = endpoint.max_inflight
            series.calls += 1
            series.tokens_in += tokens_in
            series.tokens_out += tokens_out
            if not ok:
                series.errors += 1
                series.last_error = (error or "error")[:300]
            if served_model:
                series.served.add(served_model)
            series.events.append(_Event(now, ok, latency_ms, tokens_in, tokens_out))
            _prune(series.events, now - ERROR_WINDOW_S)

    def refused(self, endpoint: ModelEndpoint) -> None:
        """Count one refusal (back-pressure): not a call, not an error."""
        with self._lock:
            key = (endpoint.endpoint_id, endpoint.role)
            series = self._series.get(key)
            if series is None:
                series = self._series[key] = _Series(endpoint=endpoint.name, cap=endpoint.max_inflight)
            series.refusals += 1

    def snapshot(self) -> list[EndpointStats]:
        """Every endpoint and role seen so far, sorted by endpoint id then role."""
        now = self.clock.monotonic()
        span = max(1.0, min(RATE_WINDOW_S, now - self._started))
        with self._lock:
            out = []
            for (endpoint_id, role), series in sorted(self._series.items()):
                _prune(series.events, now - ERROR_WINDOW_S)
                recent = [e for e in series.events if e.t >= now - RATE_WINDOW_S]
                latencies = sorted(e.latency_ms for e in series.events if e.ok)
                out.append(
                    EndpointStats(
                        endpoint_id=endpoint_id,
                        role=role,
                        endpoint=series.endpoint,
                        cap=series.cap,
                        calls=series.calls,
                        errors=series.errors,
                        tokens_in=series.tokens_in,
                        tokens_out=series.tokens_out,
                        calls_per_min=round(len(recent) * 60.0 / span, 3),
                        tokens_in_per_s=round(sum(e.tokens_in for e in recent) / span, 3),
                        tokens_out_per_s=round(sum(e.tokens_out for e in recent) / span, 3),
                        errors_last_10m=sum(1 for e in series.events if not e.ok),
                        p50_latency_ms=_percentile(latencies, 0.50),
                        p95_latency_ms=_percentile(latencies, 0.95),
                        served_models=sorted(series.served),
                        last_error=series.last_error,
                        refusals=series.refusals,
                    )
                )
            return out

    def reset(self) -> None:
        with self._lock:
            self._series.clear()
            self._started = self.clock.monotonic()


def _prune(events: deque[_Event], cutoff: float) -> None:
    while events and events[0].t < cutoff:
        events.popleft()


def _percentile(sorted_values: list[int], q: float) -> int | None:
    """Nearest-rank percentile: an observed value, never an interpolation."""
    if not sorted_values:
        return None
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[rank - 1]


_PROCESS = InferenceStats()


def process_stats() -> InferenceStats:
    """The counters every service in this process records into by default."""
    return _PROCESS

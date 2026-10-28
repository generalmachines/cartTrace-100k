"""The collector status file: the runner writes it, the watch, the pulse and the media flows read it.

Why a file and not a socket. The watch runs from a systemd timer every few minutes and must work when the runner is
dead, which is exactly when it matters most. The runner rewrites `var/status/collector.json` atomically every 15 s;
a reader judges staleness from `updated_at`, so a dead runner shows up as an old file, not a missing answer.

How the counters add up, which the watch's arithmetic relies on (`remaining = total - done - failed`):
- `episodes_total` is what this run will run: the manifest's episodes, less those the run decided to skip (a store
  that blocked us, throttled us out or has no protocol is not asked again in the same run);
- `episodes_done` counts finished episodes, including those a previous run of the same manifest finished
  (`episodes_resumed`), and `episodes_failed` counts our own failures (`harness_error`, or a bundle that could not be
  written); together they are finished work, so a run that ends with work left shows it as remaining.

Endpoint load is per model server and role: rates and latencies from carttrace.inference's per-process counters, and calls
in flight from the host-wide GPU slots, which every process on the host shares. A role the run depends on is listed
before its first call, at zero, so an endpoint that never gets a call still shows as idle.
"""

from __future__ import annotations

import os
import socket
from collections import Counter, deque
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from carttrace.core.canon import pretty
from carttrace.core.clock import Clock, iso
from carttrace.core.errors import ValidationError
from carttrace.core.files import atomic_write_text

if TYPE_CHECKING:
    from carttrace.config.schema import ModelEndpoint
    from carttrace.inference.stats import EndpointStats

STATUS_SCHEMA = "mc.collector-status.v1"
STATUS_FILE = "collector.json"
DONE_WINDOW_S = 600.0
"""`done_last_10m` counts episodes finished in this many seconds."""
ERROR_CHARS = 300
RECENT_ERRORS = 20
"""How many of the newest errors the status file keeps (D122), beside `last_error`."""

State = Literal["starting", "running", "draining", "paused", "finished", "failed"]


class EndpointLoad(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint_id: str
    role: str
    cap: int
    in_flight: int
    calls_per_min: float
    tokens_out_per_s: float = 0.0
    tokens_in_per_s: float = 0.0
    errors_last_10m: int = 0
    p50_latency_ms: int | None = None


class CollectorStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_: Literal["mc.collector-status.v1"] = Field(default="mc.collector-status.v1", alias="schema")
    run_id: str
    manifest_id: str
    pid: int
    host: str
    state: State
    started_at: str
    updated_at: str
    episodes_total: int
    """Episodes this run will run: the manifest's, less the ones it skipped (`episodes_skipped`)."""
    episodes_done: int
    """Finished episodes, `episodes_resumed` included."""
    episodes_running: int
    episodes_failed: int
    """Episodes our own code or infrastructure failed: harness_error, or no bundle could be written."""
    done_last_10m: int
    """Episodes this process finished, done or failed, in the last ten minutes."""
    end_reasons: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)
    by_mode: dict[str, int] = Field(default_factory=dict)
    target_concurrency: int
    endpoints: list[EndpointLoad] = Field(default_factory=list)
    last_error: str | None = None
    note: str = ""
    episodes_skipped: int = 0
    """Episodes left out of this run because their store blocked us, throttled us out, has no agent protocol, or
    was skipped by an effective triage decision. A new run of the manifest tries them again."""
    episodes_resumed: int = 0
    """Episodes a previous run of the same manifest had already finished."""
    episodes_paused: int = 0
    """Queued live episodes waiting for a pause in the control file to lift."""
    live_concurrency: int | None = None
    """Live episodes allowed in flight at once (collect.live_concurrency); None when the run has no live rows."""
    recent_errors: list[str] = Field(default_factory=list)
    """The newest errors, oldest first, at most RECENT_ERRORS, each "<UTC time> <episode>: <error>" (D122): one bad
    episode's message no longer hides the others'."""
    episodes_requeued: int = 0
    """Episodes set aside to run again, never counted as failed (D122): v1 rows a drain cut (their bundles are in the
    quarantine as `drained`). They stay in the work left, and a new run of the manifest runs them."""


class StatusBoard:
    """The runner's counters. Every status file is a `snapshot` of them; time comes from the injected clock."""

    def __init__(
        self,
        *,
        run_id: str,
        manifest_id: str,
        total: int,
        target_concurrency: int,
        live_concurrency: int | None,
        clock: Clock,
        host: str | None = None,
        pid: int | None = None,
    ) -> None:
        self.run_id = run_id
        self.manifest_id = manifest_id
        self.clock = clock
        self.started_at = iso(clock.now())
        self.host = host if host is not None else socket.gethostname()
        self.pid = pid if pid is not None else os.getpid()
        self.state: State = "starting"
        self.total = total
        self.target_concurrency = target_concurrency
        self.live_concurrency = live_concurrency
        self.done = 0
        self.failed = 0
        self.running = 0
        self.skipped = 0
        self.resumed = 0
        self.paused = 0
        self.requeued = 0
        self.end_reasons: Counter[str] = Counter()
        self.by_category: Counter[str] = Counter()
        self.by_mode: Counter[str] = Counter()
        self.last_error: str | None = None
        self.recent_errors: deque[str] = deque(maxlen=RECENT_ERRORS)
        self.note = ""
        self._finished: deque[float] = deque()

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.done - self.failed)

    def resumed_episode(self, *, end_reason: str | None, category: str, mode: str) -> None:
        """An episode a previous run finished: done, and in the breakdowns, but not in `done_last_10m`."""
        self.resumed += 1
        self.done += 1
        self._count(end_reason, category, mode)

    def started(self) -> None:
        self.running += 1

    def finished(self, *, end_reason: str | None, category: str, mode: str, failed: bool) -> None:
        """An episode this process ran has ended, with a bundle (`end_reason`) or without one (failed, None)."""
        self.running = max(0, self.running - 1)
        if failed:
            self.failed += 1
        else:
            self.done += 1
        self._count(end_reason, category, mode)
        self._finished.append(self.clock.monotonic())

    def returned(self) -> None:
        """An episode that started but was put back in the queue: its store was paused before any traffic, or the
        agent's model server failed it and its re-run will replace its bundle (mc-twc)."""
        self.running = max(0, self.running - 1)

    def requeued_episode(self) -> None:
        """An episode that started and was set aside to run again (D122: a drain cut it): neither done nor failed,
        so it stays in the work left, and no error is named for it."""
        self.running = max(0, self.running - 1)
        self.requeued += 1

    def skip(self, count: int) -> None:
        """Episodes the run will not run after all: they leave the total."""
        self.skipped += count
        self.total = max(0, self.total - count)

    def error(self, message: str) -> None:
        self.last_error = message[:ERROR_CHARS]
        self.recent_errors.append(f"{iso(self.clock.now())} {self.last_error}")

    def done_last_10m(self) -> int:
        cutoff = self.clock.monotonic() - DONE_WINDOW_S
        while self._finished and self._finished[0] < cutoff:
            self._finished.popleft()
        return len(self._finished)

    def snapshot(self, endpoints: Iterable[EndpointLoad] = ()) -> CollectorStatus:
        return CollectorStatus(
            run_id=self.run_id,
            manifest_id=self.manifest_id,
            pid=self.pid,
            host=self.host,
            state=self.state,
            started_at=self.started_at,
            updated_at=iso(self.clock.now()),
            episodes_total=self.total,
            episodes_done=self.done,
            episodes_running=self.running,
            episodes_failed=self.failed,
            done_last_10m=self.done_last_10m(),
            end_reasons=dict(sorted(self.end_reasons.items())),
            by_category=dict(sorted(self.by_category.items())),
            by_mode=dict(sorted(self.by_mode.items())),
            target_concurrency=self.target_concurrency,
            endpoints=list(endpoints),
            last_error=self.last_error,
            note=self.note,
            episodes_skipped=self.skipped,
            episodes_resumed=self.resumed,
            episodes_paused=self.paused,
            live_concurrency=self.live_concurrency,
            episodes_requeued=self.requeued,
            recent_errors=list(self.recent_errors),
        )

    def _count(self, end_reason: str | None, category: str, mode: str) -> None:
        if end_reason is not None:
            self.end_reasons[end_reason] += 1
        self.by_category[category] += 1
        self.by_mode[mode] += 1


def endpoint_loads(
    stats: Iterable[EndpointStats],
    in_flight: Mapping[str, int],
    expected: Iterable[ModelEndpoint] = (),
    *,
    caps: Mapping[str, int] | None = None,
) -> list[EndpointLoad]:
    """One load per endpoint and role: every role carttrace.inference has counted, plus the `expected` routes the run
    depends on that have made no call yet. In flight is per server, shared by its roles, from the GPU slots. `caps`
    (per endpoint id) is each server's cap right now, by its schedule (D114); without it, the configured one."""
    caps = caps or {}
    loads: dict[tuple[str, str], EndpointLoad] = {}
    for item in stats:
        loads[(item.endpoint_id, item.role)] = EndpointLoad(
            endpoint_id=item.endpoint_id,
            role=item.role,
            cap=caps.get(item.endpoint_id, item.cap),
            in_flight=in_flight.get(item.endpoint_id, 0),
            calls_per_min=item.calls_per_min,
            tokens_out_per_s=item.tokens_out_per_s,
            tokens_in_per_s=item.tokens_in_per_s,
            errors_last_10m=item.errors_last_10m,
            p50_latency_ms=item.p50_latency_ms,
        )
    for route in expected:
        key = (route.endpoint_id, route.role)
        if key not in loads:
            loads[key] = EndpointLoad(
                endpoint_id=route.endpoint_id,
                role=route.role,
                cap=caps.get(route.endpoint_id, route.max_inflight),
                in_flight=in_flight.get(route.endpoint_id, 0),
                calls_per_min=0.0,
            )
    return [loads[key] for key in sorted(loads)]


def status_path(var_dir: Path) -> Path:
    """Where the runner writes, and the watch reads: <paths.var>/status/collector.json."""
    return var_dir / "status" / STATUS_FILE


def write_status(path: Path, status: CollectorStatus) -> None:
    """The whole file or nothing: a reader never sees half a status."""
    atomic_write_text(path, pretty(status.model_dump(mode="json", by_alias=True)))


def read_status(path: Path) -> CollectorStatus | None:
    """The status a runner last wrote here, or None when none has. Raises ValidationError for a file that does not
    match this version of the contract."""
    if not path.exists():
        return None
    try:
        return CollectorStatus.model_validate_json(path.read_bytes())
    except PydanticValidationError as exc:
        raise ValidationError(
            f"{path} does not match {STATUS_SCHEMA}: {exc.errors()[0]['msg']}",
            hint="the runner and this reader are on different versions of carttrace.collect.status; update both",
        ) from exc

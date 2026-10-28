"""The scheduler: which episode starts next, within the run's limits, and how a run stops starting them.

The limits (docs/ARCHITECTURE.md, "Concurrency and compute", and CLAUDE.md, "Polite traffic"):
- sandbox and reference episodes: at most `collect.target_concurrency` in flight, in manifest order. With
  `concurrency` (D114: the followed endpoint's cap right now, times collect.concurrency_per_slot) the lane follows it
  live, checked every `follow_every_s`: a raise starts more workers at once; a cut lets the running episodes finish
  and retires workers as they do, until no more than the new target run;
- browser episodes (a WebMCP row): at most `browser_concurrency` (config `sandbox.browser.max_browsers`) in flight,
  in their own lane, in manifest order. Such an episode holds a headless Chromium for its whole run, so starting more
  of them than the host-wide slot pool (carttrace.sandbox.browser.slots) holds only queues them at the pool until
  `slot_wait_s` runs out and they die as NoBrowserSlot (mc-3r1.34: 199 of 205 harness errors in sbx-replica-0003,
  32 rows against 4 browsers). The lane starts only as many as there are browsers; the pool stays the backstop for
  contention across processes;
- live episodes: at most `collect.live_concurrency` in flight across stores, and one at a time per store, so a
  store sees one session from this process (carttrace.live's throttle enforces the same across processes, with the daily
  cap and the per-platform rate on top).

Pauses and skips. A live episode that finds the gate closed for its store (the control file's brake) goes back to the
front of that store's queue, and the store is tried again after `pause_recheck_s`; the run waits, in state "paused" when
nothing else can run, and resumes when the brake is released. A store that blocked us, throttled us out, has no
protocol, or that an effective triage decision says to skip loses the rest of its live episodes for this run; they
leave the run's total, and a later run of the manifest tries them again.

Model server outages (mc-twc). A sandbox or browser episode whose agent's model server was down (label
`model_unavailable`: no connection, no reply in time, a 5xx after the inference retries) goes back to the front of its
lane, at most `OUTAGE_RETRIES` times, and its re-run overwrites the void bundle (same run and episode id). Its lane
then starts nothing for `OUTAGE_PAUSE_S`, doubling to `OUTAGE_PAUSE_MAX_S` while the server stays down; after each
wait one episode alone tries the server, and only when it gets an answer do the other workers start again. Without
this, one outage on 2026-10-12 (run-dbcd65a3488f) ended every Qwen row of the consumer-electronics twin run, 1,500 of
them, as harness_error within ten minutes, about three seconds each. Live rows are not run again: a re-run would be
a second session on the store.

Draining. `drain()` stops every worker from taking another episode; the ones running finish. `cancel_running()`
cancels them, and each records `harness_error` (label `drained`) on its way out.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from carttrace.collect.episode import Episode, EpisodeReport
from carttrace.collect.manifest import ManifestRow
from carttrace.collect.profiles import MODEL_UNAVAILABLE, STOREFRONT
from carttrace.collect.status import StatusBoard

PAUSE_RECHECK_S = 60.0
"""How long a paused store waits before the gate is checked again."""

OUTAGE_PAUSE_S = 30.0
"""How long a lane waits after an episode found the agent's model server down, before one episode tries it again."""
OUTAGE_PAUSE_MAX_S = 600.0
"""The longest wait between tries while the server stays down (the wait doubles from OUTAGE_PAUSE_S)."""
OUTAGE_RETRIES = 3
"""How many times one row is run again after the model server failed it; then its harness_error stands."""

BROWSER_SURFACES: frozenset[str] = frozenset({STOREFRONT})
"""Surface kinds whose episode holds a sandbox browser (one headless Chromium) for its whole run."""


def needs_browser(row: ManifestRow) -> bool:
    """Whether the row's episode opens a sandbox browser: it names a surface that runs in one."""
    return row.surface is not None and row.surface.kind in BROWSER_SURFACES


@dataclass(frozen=True, slots=True)
class Skip:
    store_id: str
    reason: str
    episodes: int


class Scheduler:
    def __init__(
        self,
        rows: Sequence[ManifestRow],
        *,
        episode: Callable[[ManifestRow], Episode],
        on_report: Callable[[ManifestRow, EpisodeReport], None],
        board: StatusBoard,
        target_concurrency: int,
        live_concurrency: int,
        browser_concurrency: int,
        pause_recheck_s: float = PAUSE_RECHECK_S,
        outage_pause_s: float = OUTAGE_PAUSE_S,
        concurrency: Callable[[], int] | None = None,
        follow_every_s: float = 60.0,
    ) -> None:
        self._episode = episode
        self._concurrency = concurrency
        self.follow_every_s = follow_every_s
        self._on_report = on_report
        self.board = board
        self.pause_recheck_s = pause_recheck_s
        self._sandbox: deque[ManifestRow] = deque(row for row in rows if row.mode != "live" and not needs_browser(row))
        self._browser: deque[ManifestRow] = deque(row for row in rows if row.mode != "live" and needs_browser(row))
        self._live: dict[str, deque[ManifestRow]] = {}
        for row in rows:
            if row.mode == "live":
                self._live.setdefault(row.store_id, deque()).append(row)
        self.target = max(1, concurrency()) if concurrency is not None else target_concurrency
        """Sandbox episodes the lane runs at once right now."""
        self._sandbox_workers = min(self.target, len(self._sandbox))
        self._sandbox_alive = 0
        self._browser_workers = min(browser_concurrency, len(self._browser))
        self._live_workers = min(live_concurrency, len(self._live))
        self._active: set[str] = set()
        self._paused_until: dict[str, float] = {}
        self.skips: list[Skip] = []
        self._running: set[asyncio.Task[EpisodeReport]] = set()
        self._changed: asyncio.Event | None = None
        self._follow_wake = asyncio.Event()
        self._tasks: list[asyncio.Future[None]] = []
        self.draining = False
        self.outage_pause_s = outage_pause_s
        self._outage_until = 0.0
        """Loop time before which no sandbox or browser episode starts: the model server was found down."""
        self._outage_wait = 0.0
        """The current wait between tries; 0 while the server answers."""
        self._probing = False
        """One episode is trying the server after a wait; the other workers wait for its answer."""
        self._reruns: dict[str, int] = {}
        self.outages = 0
        """Episodes the model server failed, each run again or, past OUTAGE_RETRIES, left as harness_error."""

    async def run(self) -> None:
        """Run every episode, or until drained."""
        self._changed = asyncio.Event()
        self.board.target_concurrency = self.target
        self._update_board()
        for _ in range(self._sandbox_workers):
            self._start_sandbox_worker()
        self._tasks += [asyncio.ensure_future(self._queue_worker(self._browser)) for _ in range(self._browser_workers)]
        self._tasks += [asyncio.ensure_future(self._live_worker()) for _ in range(self._live_workers)]
        if self._concurrency is not None and self._sandbox:
            self._tasks.append(asyncio.ensure_future(self._follow()))
        while not all(task.done() for task in self._tasks):  # the follower may add workers while these run
            await asyncio.gather(*self._tasks)
        self._update_board()

    def _start_sandbox_worker(self) -> None:
        self._sandbox_alive += 1
        self._tasks.append(asyncio.ensure_future(self._sandbox_worker()))

    async def _sandbox_worker(self) -> None:
        try:
            await self._queue_worker(self._sandbox, lane_full=self._sandbox_surplus)
        finally:
            self._sandbox_alive -= 1
            self._follow_wake.set()

    def _sandbox_surplus(self) -> bool:
        """Whether the sandbox lane has more workers than its target right now: the caller then retires. Read at
        every episode boundary, so a cut stops new starts at once; a raise waits for the follower's next check."""
        self._retarget()
        return self._sandbox_alive > self.target

    def _retarget(self) -> None:
        if self._concurrency is None:
            return
        target = max(1, self._concurrency())
        if target != self.target:
            self.target = target
            self.board.target_concurrency = target
            self._update_board()

    async def _follow(self) -> None:
        """Every follow_every_s, read the target and grow the sandbox lane to it (a cut retires workers as their
        episodes end, in _queue_worker). Ends with the lane."""
        while not self._lane_done():
            self._follow_wake.clear()
            try:
                async with asyncio.timeout(self.follow_every_s):
                    await self._follow_wake.wait()
            except TimeoutError:
                pass
            if self._lane_done():
                return
            self._retarget()
            while self._sandbox_alive < min(self.target, self._sandbox_alive + len(self._sandbox)):
                self._start_sandbox_worker()

    def _lane_done(self) -> bool:
        return self.draining or not (self._sandbox or self._sandbox_alive)

    def drain(self) -> None:
        """Start no more episodes; the running ones finish. The board says so from now on."""
        self.draining = True
        self._update_board()
        self._wake()
        self._follow_wake.set()

    def cancel_running(self) -> int:
        """Cancel the running episodes; each records harness_error on its way out. Returns how many."""
        for task in self._running:
            task.cancel()
        return len(self._running)

    @property
    def running(self) -> int:
        return len(self._running)

    @property
    def queued(self) -> int:
        return len(self._sandbox) + len(self._browser) + sum(len(q) for q in self._live.values())

    # -- workers ---------------------------------------------------------------------------------------------------

    async def _queue_worker(self, queue: deque[ManifestRow], lane_full: Callable[[], bool] | None = None) -> None:
        """Run one lane's rows in manifest order until it empties or the run drains, holding back while the agent's
        model server is down. `lane_full` true retires the worker before its next row."""
        loop = asyncio.get_running_loop()
        while not self.draining and queue:
            if lane_full is not None and lane_full():
                return
            if self._outage_wait:
                left = self._outage_until - loop.time()
                if left > 0 or self._probing:
                    await self._wait(left if left > 0 else None)
                    continue
                self._probing = True
            probe = self._probing
            row = queue.popleft()
            try:
                report = await self._execute(row)
            finally:
                if probe:
                    self._probing = False
            if MODEL_UNAVAILABLE in report.labels:
                self.outages += 1
                if report.paused:
                    queue.appendleft(row)
                if not self._outage_wait or probe or self._outage_until <= loop.time():
                    self._outage_wait = min(OUTAGE_PAUSE_MAX_S, max(self.outage_pause_s, self._outage_wait * 2))
                    self._outage_until = loop.time() + self._outage_wait
            elif probe:
                self._outage_wait = 0.0
            self._update_board()
            self._wake()

    async def _live_worker(self) -> None:
        loop = asyncio.get_running_loop()
        while not self.draining:
            store_id = self._next_store(loop.time())
            if store_id is None:
                if not any(self._live.values()):
                    return
                await self._wait(self._next_wake(loop.time()))
                continue
            row = self._live[store_id].popleft()
            self._active.add(store_id)
            try:
                report = await self._execute(row)
            finally:
                self._active.discard(store_id)
            if report.paused:
                self._live[store_id].appendleft(row)
                if report.skip_store:
                    self._skip(store_id, report.skip_store)
                else:
                    self._paused_until[store_id] = loop.time() + self.pause_recheck_s
            else:
                self._paused_until.pop(store_id, None)
                if report.skip_store:
                    self._skip(store_id, report.skip_store)
            self._update_board()
            self._wake()

    async def _execute(self, row: ManifestRow) -> EpisodeReport:
        episode = self._episode(row)
        self.board.started()
        task = asyncio.ensure_future(episode.run())
        self._running.add(task)
        self._update_board()
        try:
            await asyncio.wait([task])
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.wait([task])
            raise
        finally:
            self._running.discard(task)
        report = episode.report
        if not task.cancelled() and task.exception() is not None:
            exc = task.exception()
            report.failed = True
            report.error = f"{type(exc).__name__}: {exc}"[:300]
        if self._rerun(row, report):
            # its bundle will be replaced by the re-run's, so it is neither counted nor sent to the results
            report.paused = True
        self._on_report(row, report)
        return report

    def _rerun(self, row: ManifestRow, report: EpisodeReport) -> bool:
        """Whether a sandbox row the model server failed goes back in its lane (at most OUTAGE_RETRIES times)."""
        if row.mode == "live" or MODEL_UNAVAILABLE not in report.labels:
            return False
        done = self._reruns.get(row.episode_id, 0)
        if done >= OUTAGE_RETRIES:
            return False
        self._reruns[row.episode_id] = done + 1
        return True

    # -- choosing --------------------------------------------------------------------------------------------------

    def _next_store(self, now: float) -> str | None:
        for store_id, queue in self._live.items():
            if queue and store_id not in self._active and self._paused_until.get(store_id, 0.0) <= now:
                return store_id
        return None

    def _next_wake(self, now: float) -> float | None:
        """Seconds until a paused store may be tried again, or None to wait for a change."""
        times = [
            until
            for store_id, until in self._paused_until.items()
            if self._live.get(store_id) and store_id not in self._active
        ]
        return max(0.0, min(times) - now) if times else None

    async def _wait(self, seconds: float | None) -> None:
        assert self._changed is not None
        self._changed.clear()
        try:
            async with asyncio.timeout(seconds):
                await self._changed.wait()
        except TimeoutError:
            return

    def _wake(self) -> None:
        if self._changed is not None:
            self._changed.set()

    def _skip(self, store_id: str, reason: str) -> None:
        left = len(self._live.get(store_id, ()))
        self._live[store_id] = deque()
        self._paused_until.pop(store_id, None)
        if left:
            self.board.skip(left)
        self.skips.append(Skip(store_id, reason, left))

    def _update_board(self) -> None:
        paused = sum(len(self._live[s]) for s in self._paused_until if s in self._live)
        self.board.paused = paused
        if self._outage_wait:
            self.board.note = (
                f"the agent's model server is down: waiting {self._outage_wait:g} s between tries, "
                f"{self.outages} episodes failed by it so far"
            )
        elif self.board.note.startswith("the agent's model server is down"):
            self.board.note = ""
        if self.draining:
            self.board.state = "draining"
        elif not self._running and (
            self._outage_wait or (paused and not self._sandbox and not self._browser and paused == self.queued)
        ):
            self.board.state = "paused"
        else:
            self.board.state = "running"

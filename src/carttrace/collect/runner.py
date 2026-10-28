"""`run_manifest`: run every episode of a manifest, resumably, within the run's limits, and keep the status current.

What a run does, in order:
1. Refuses a manifest with live rows unless all three live gates are open (CARTTRACE_LIVE=1 in the environment, live.enabled,
   the --live flag). carttrace.live's LiveGate stays the authority on every call; this is the early refusal.
2. Decides the data rights of each catalog the manifest's sandbox rows use, refusing a restricted one that is not
   allowed, so no episode starts under rights nobody decided.
3. Finishes what a crash left: staged bundles with an outcome are uploaded, unfinished ones discarded so their
   episodes run again (`recover_staged(..., discard_unfinished=True)`).
4. Resumes: episodes whose bundle is complete in storage are skipped and counted as done. With
   `retry_harness_errors`, those that ended in harness_error run again, as new bundles beside the old.
5. Runs the rest through the scheduler, rewriting var/status/collector.json every collect.status_every_s, and
   appending a results record (traces collected, latest outcome) for the sheet sync after every bundle.
6. Quarantines (D111, carttrace.collect.quarantine): a v1 sandbox or reference bundle (a row with an experiment block) that
   ended in harness_error, a model
   error, a crash or `drained` is moved to quarantine/<date>/<run>/<episode>/ as soon as it is uploaded, gets no
   results row, and its row runs again under the same episode id, up to collect.quarantine_retries times (counted
   over every run of the manifest from the quarantine's reason files; drained attempts are not counted).

Stopping. SIGTERM or SIGINT (or `RunControl.drain()`) drains: no episode starts after it, running ones get until
`drain_timeout_s` (14 minutes, inside the collector unit's 15 minute TimeoutStopSec) to finish, then are cancelled and
recorded as harness_error with the label `drained`, and a final status is written. A second signal cancels them at
once, unless it comes within SIGNAL_REPEAT_S of the first: that is the same stop delivered twice (D122). A drained
run ends "finished", with the episodes it did not reach left over; a new run of the manifest picks them up. A v1 row
cut by the drain is set aside, never counted: its bundle goes to the quarantine (D111), the status counts it as
requeued rather than failed and names no error for it, and the next run of the manifest runs it again.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from carttrace import __version__
from carttrace.collect.bindings import Bindings, default_bindings
from carttrace.collect.episode import Episode, EpisodeReport, RunContext
from carttrace.collect.manifest import Manifest, ManifestRow
from carttrace.collect.open_lane import check_open_rights
from carttrace.collect.ports import LiveGate, RunMode
from carttrace.collect.quarantine import DRAINED, attempts, listed, move, problem
from carttrace.collect.results import FinishedEpisode, ResultsTally, ResultWriter, recorded_episodes
from carttrace.collect.rights import sandbox_rights
from carttrace.collect.scheduler import OUTAGE_PAUSE_S, PAUSE_RECHECK_S, Scheduler
from carttrace.collect.status import CollectorStatus, EndpointLoad, StatusBoard, endpoint_loads, status_path, write_status
from carttrace.collect.worlds import SandboxWorlds, configured_worlds, replica_rights, rights_key
from carttrace.config.schema import ModelEndpoint, Settings
from carttrace.core.clock import Clock, SystemClock, iso
from carttrace.core.errors import ConfigError, SafetyError, ValidationError
from carttrace.core.ids import stable_id
from carttrace.decide.service import DecisionEngine
from carttrace.inference.caps import LiveCaps
from carttrace.inference.slots import gpu_slots
from carttrace.inference.stats import EndpointStats, process_stats
from carttrace.storage.service import BlobStore
from carttrace.trace.provenance import ResultMode
from carttrace.trace.reader import resolved, scan
from carttrace.trace.schema import BundleSummary, ModelRef, Rights
from carttrace.trace.writer import recover_staged

DRAIN_TIMEOUT_S = 14 * 60.0
"""How long running episodes get to finish after a drain starts: inside the unit's TimeoutStopSec of 15 minutes, with
a minute left to record the stragglers and write the final status."""
SIGNAL_REPEAT_S = 5.0
"""A second SIGTERM or SIGINT this soon after the first is the same stop, not a request to cancel (D122). systemd's
default KillMode=control-group signals every process of the unit, and `uv run` forwards the TERM it got to the
collector, so one `systemctl stop` reached the runner twice within milliseconds and cut every running episode at once
(2026-10-17: 45, 112 and 144 drained harness errors in three stops, the last drain lasting two seconds)."""
LIVE_AGENTS = ("qwen", "reference")
"""Who may shop a live row: the Qwen agent, or the reference agent for a read-only smoke of a store's surface."""
REFERENCE_MODEL = ModelRef(role="agent", requested="reference-agent", endpoint="local")
"""The scripted reference agent, named in headers where a model would be."""
DECIDE_ROLES: Mapping[str, str] = {"clef-flash": "decide.flash", "clef": "decide.full"}

InferenceSnapshot = Callable[[], Iterable[EndpointStats]]


class RunSummary(BaseModel):
    """What a run did, for the CLI and for callers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    manifest_id: str
    state: str
    started_at: str
    ended_at: str
    episodes_total: int
    episodes_done: int
    episodes_failed: int
    episodes_skipped: int
    episodes_resumed: int
    episodes_left: int
    """Episodes neither done nor failed: left by a drain, or waiting on a pause when the run stopped."""
    bundles: int
    """Bundles this run wrote and uploaded."""
    end_reasons: dict[str, int] = Field(default_factory=dict)
    recovered: list[str] = Field(default_factory=list)
    """Staged bundles from an earlier crash that this run uploaded before it started."""
    drained: bool = False
    quarantined: int = 0
    """Bundles this run moved to the quarantine (D111)."""
    retried: int = 0
    """Rows this run ran again after a quarantine."""
    notes: list[str] = Field(default_factory=list)


class RunControl:
    """A handle on a running `run_manifest`: `drain()` once to stop starting episodes, twice to cancel running ones."""

    def __init__(self, *, drain_timeout_s: float = DRAIN_TIMEOUT_S, signal_repeat_s: float = SIGNAL_REPEAT_S) -> None:
        self.drain_timeout_s = drain_timeout_s
        self.signal_repeat_s = signal_repeat_s
        self.requests = 0
        self.repeats = 0
        """Signals taken as a repeat of the first (the same stop delivered twice), not as a request to cancel."""
        self._first_signal: float | None = None
        self._scheduler: Scheduler | None = None
        self._deadline: asyncio.Task[None] | None = None

    def signalled(self) -> None:
        """A SIGTERM or SIGINT: the first drains; one within `signal_repeat_s` of the first is the same stop and
        changes nothing; a later one cancels the running episodes, as a second `drain()` does."""
        now = asyncio.get_running_loop().time()
        if self._first_signal is not None and now - self._first_signal < self.signal_repeat_s:
            self.repeats += 1
            return
        if self._first_signal is None:
            self._first_signal = now
        self.drain()

    def drain(self) -> None:
        self.requests += 1
        scheduler = self._scheduler
        if scheduler is None:
            return
        if self.requests == 1:
            scheduler.drain()
            self._deadline = asyncio.get_running_loop().create_task(self._cancel_after(scheduler))
        else:
            scheduler.cancel_running()

    def attach(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler
        if self.requests:
            self.requests -= 1
            self.drain()

    def detach(self) -> None:
        self._scheduler = None
        if self._deadline is not None:
            self._deadline.cancel()

    async def _cancel_after(self, scheduler: Scheduler) -> None:
        await asyncio.sleep(self.drain_timeout_s)
        scheduler.cancel_running()


def closed_live_gates(settings: Settings, live_flag: bool, env: Mapping[str, str]) -> list[str]:
    """The live gates that are closed, by name: empty when a live run may start."""
    closed = []
    if env.get("CARTTRACE_LIVE") != "1":
        closed.append("CARTTRACE_LIVE=1 is not set in the environment")
    if not settings.live.enabled:
        closed.append("live.enabled is false")
    if not live_flag:
        closed.append("the --live flag was not given")
    return closed


def code_version(home: Path) -> str:
    """`git describe` of the checkout, -dirty when it has uncommitted changes; the package version without git."""
    try:
        done = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--tags"],  # noqa: S607 - git from PATH, as carttrace.protocols.fetch
            cwd=home,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return f"mc-{__version__}"
    described = done.stdout.strip()
    return described if done.returncode == 0 and described else f"mc-{__version__}"


async def run_manifest(
    manifest: Manifest,
    *,
    settings: Settings,
    live_flag: bool,
    bindings: Bindings | None = None,
    engine: DecisionEngine | None = None,
    store: BlobStore | None = None,
    clock: Clock | None = None,
    run_id: str | None = None,
    control: RunControl | None = None,
    handle_signals: bool = True,
    retry_harness_errors: bool = False,
    live_agent: str = "qwen",
    inference_stats: InferenceSnapshot | None = None,
    in_flight: Callable[[], Mapping[str, int]] | None = None,
    version: str | None = None,
    env: Mapping[str, str] | None = None,
    pause_recheck_s: float = PAUSE_RECHECK_S,
    outage_pause_s: float = OUTAGE_PAUSE_S,
    status_file: Path | None = None,
    caps: LiveCaps | None = None,
) -> RunSummary:
    """Run `manifest` to the end, or until drained. Raises SafetyError before anything starts when the manifest has
    live rows and a live gate is closed, or uses a restricted catalog that is not allowed.

    Every keyword after `live_flag` is for tests and tools: the defaults are the configured implementations, the
    system clock, the process's inference counters and GPU slots, and the checkout's git version; `status_file`
    replaces var/status/collector.json for a tool that must not overwrite the collector's (the payments check).
    """
    refuse_closed_live_gates(manifest, settings, live_flag, os.environ if env is None else env)
    bindings = bindings or default_bindings(settings)
    clock = clock or SystemClock()
    if store is None:
        from carttrace.storage import store as configured_store

        store = configured_store(settings)
    if live_agent not in LIVE_AGENTS:
        raise ValidationError(f"live_agent {live_agent!r} is not one of {LIVE_AGENTS}", hint="qwen or reference")
    run = _Run(manifest, settings=settings, bindings=bindings, store=store, clock=clock, run_id=run_id)
    run.live_agent = live_agent
    run.status = _StatusFile(
        status_file or status_path(settings.paths.var),
        run.board,
        routes=run.routes,
        inference_stats=inference_stats,
        in_flight=in_flight,
        settings=settings,
        clock=clock,
    )
    run.status.write()
    follow = None
    if settings.collect.concurrency_follows is not None:  # D114: the sandbox lane follows the GPU cap live
        follow = caps or LiveCaps.from_disk(settings, clock=clock)
    try:
        pending = run.resume(retry_harness_errors=retry_harness_errors)
        ctx = run.context(
            pending,
            engine=engine or _engine(settings, offline=decisions_offline(settings, manifest)),
            gate=bindings.live_gate(settings, live_flag=live_flag) if any(r.mode == "live" for r in pending) else None,
            version=version or code_version(settings.home),
        )

        def scheduler(rows: list[ManifestRow]) -> Scheduler:
            return Scheduler(
                rows,
                episode=lambda row: Episode(row, ctx),
                on_report=run.on_report,
                board=run.board,
                target_concurrency=settings.collect.target_concurrency,
                live_concurrency=settings.collect.live_concurrency,
                browser_concurrency=settings.sandbox.browser.max_browsers,
                pause_recheck_s=pause_recheck_s,
                outage_pause_s=outage_pause_s,
                concurrency=follow.collect_target if follow is not None else None,
                follow_every_s=settings.gpu.recheck_s,
            )

        control = control or RunControl()
        await run.execute(scheduler(pending), control, handle_signals=handle_signals)
        await run.retry_quarantined(scheduler, control, handle_signals=handle_signals)
    except BaseException as exc:
        run.failed(exc)
        raise
    return run.summary()


def refuse_closed_live_gates(manifest: Manifest, settings: Settings, live_flag: bool, env: Mapping[str, str]) -> None:
    live_rows = sum(row.mode == "live" for row in manifest.rows)
    closed = closed_live_gates(settings, live_flag, env) if live_rows else []
    if closed:
        raise SafetyError(
            f"manifest {manifest.id} has {live_rows} live rows and the live gate is closed: " + "; ".join(closed),
            hint="set CARTTRACE_LIVE=1 and pass --live to run live rows, or build a manifest without them",
        )


class _Run:
    """One run's state: its board, its notes, the rights and models it decided, its results file."""

    def __init__(
        self,
        manifest: Manifest,
        *,
        settings: Settings,
        bindings: Bindings,
        store: BlobStore,
        clock: Clock,
        run_id: str | None,
    ) -> None:
        self.manifest = manifest
        self.settings = settings
        self.bindings = bindings
        self.store = store
        self.clock = clock
        self.worlds = configured_worlds(settings, bindings)
        self.rights = _sandbox_rights(manifest, bindings, self.worlds)
        check_open_rights(manifest, self.rights)
        self.models, self.routes = _models(settings, manifest)
        self.live_agent = "qwen"
        """Which agent shops live rows (`run_manifest(live_agent=...)`)."""
        self.started_at = iso(clock.now())
        self.run_id = run_id or stable_id("run", manifest.id, self.started_at)
        live = any(row.mode == "live" for row in manifest.rows)
        self.board = StatusBoard(
            run_id=self.run_id,
            manifest_id=manifest.id,
            total=len(manifest.rows),
            target_concurrency=settings.collect.target_concurrency,
            live_concurrency=settings.collect.live_concurrency if live else None,
            clock=clock,
        )
        self.status: _StatusFile | None = None
        self.notes: list[str] = []
        self.recovered: list[str] = []
        self.results = ResultWriter(
            db_path=settings.collect.results_db,
            file_path=settings.collect.results_path if settings.collect.results_file_fallback else None,
            tally=ResultsTally(),
            clock=clock,
            run_id=self.run_id,
            manifest_id=manifest.id,
            mode="live" if live else "sandbox",
            started_at=self.started_at,
        )
        self.drained = False
        self.retries = settings.collect.quarantine_retries
        self.attempts: dict[str, int] = {}
        """Episode id to its quarantined attempts that count toward the retry limit (D111)."""
        self.requeue: list[ManifestRow] = []
        self.quarantined = 0
        self.retried = 0
        self.playground_bundles = 0
        """Bundles of playground rows, which the results database never holds."""
        if live and not settings.live.agent_profile_url:
            self.notes.append(
                "live.agent_profile_url is unset: live UCP calls carry no agent profile, which Shopify requires; "
                "publish one (carttrace live profile) and set it"
            )

    def resume(self, *, retry_harness_errors: bool) -> list[ManifestRow]:
        """Finish what a crash staged, then the rows still to run; what storage already holds counts as done."""
        recovered = recover_staged(self.settings.trace.staging_dir, self.store, discard_unfinished=True)
        self.recovered = list(recovered.uploaded)
        self.notes += [f"staged {staged.episode_id}: {staged.reason}" for staged in recovered.unfinished]
        # A bundle this reader cannot parse (an import from another tool, or newer code) is skipped and noted, never
        # a crash: on 2026-10-15 old-lab bundles imported under episodes/ stopped every collector start.
        unreadable: list[tuple[str, str]] = []
        summaries = list(scan(self.store, unreadable=unreadable))
        if unreadable:
            self.notes.append(
                f"skipped {len(unreadable)} unreadable bundles, first: {unreadable[0][0]}: {unreadable[0][1]}"
            )
        self.results.tally = ResultsTally.from_summaries(summaries)
        done = resolved(s for s in summaries if s.complete)
        pending: list[ManifestRow] = []
        held = attempts(listed(self.store, (r.episode_id for r in self.manifest.rows)))
        self.attempts = dict(held)
        spent = 0
        for row in self.manifest.rows:
            found = done.get(row.episode_id)
            if found is None and held[row.episode_id] > self.retries:
                spent += 1  # quarantined past its retries: left for a person, never counted (D111)
                continue
            if found is None or (retry_harness_errors and found.end_reason == "harness_error"):
                pending.append(row)
            else:
                self.board.resumed_episode(end_reason=found.end_reason, category=found.category, mode=row.mode)
        if self.board.resumed:
            self.notes.append(f"resumed: {self.board.resumed} episodes already in storage")
        if spent:
            self.notes.append(
                f"{spent} rows quarantined more than collect.quarantine_retries={self.retries} times are not run again"
            )
        self._backfill(done)
        return pending

    def _backfill(self, done: Mapping[str, BundleSummary]) -> None:
        """Write the results row of every finished bundle of this manifest the database lacks: one uploaded by
        recover_staged, one whose close failed, one whose results write failed (final review #8). Counting reads the
        database, so a bundle missing from it was never counted and never re-run."""
        rows = {r.episode_id: r for r in self.manifest.rows if r.mode in ("live", "sandbox", "reference")}
        rows = {k: r for k, r in rows.items() if k in done}
        have = recorded_episodes(self.settings.collect.results_db, rows)
        added = 0
        for episode_id, row in sorted(rows.items()):
            found = done[episode_id]
            if episode_id in have or found.end_reason is None or found.ended_at is None:
                continue
            self.results.add(
                FinishedEpisode(
                    store_id=found.store_id,
                    episode_id=episode_id,
                    end_reason=found.end_reason,
                    ended_at=found.ended_at,
                    mode=cast(ResultMode, row.mode),
                    provenance=found.provenance,
                    store_source=found.store_source,
                    surface=found.surface,
                    agent=found.agent,
                )
            )
            added += 1
        if added:
            self.notes.append(f"back-filled {added} results rows of bundles already in storage")
        self._note_result_errors()

    def context(
        self, pending: list[ManifestRow], *, engine: DecisionEngine, gate: LiveGate | None, version: str
    ) -> RunContext:
        self.results.start(version)
        self._note_result_errors()
        return RunContext(
            settings=self.settings,
            bindings=self.bindings,
            store=self.store,
            engine=engine,
            clock=self.clock,
            manifest=self.manifest,
            run_id=self.run_id,
            worlds=self.worlds,
            staging_dir=self.settings.trace.staging_dir,
            code_version=version,
            sandbox_rights=self.rights,
            models=self.models,
            model_names=_model_names(self.settings),
            gate=gate,
            salt=_salt(self.settings),
            live_agent=self.live_agent,
            playgrounds=_playgrounds(self.settings, pending),
        )

    def on_report(self, row: ManifestRow, report: EpisodeReport) -> None:
        if report.paused:
            self.board.returned()
            return
        v1 = row.experiment is not None and row.mode in ("sandbox", "reference")
        found = problem(report.end_reason, report.labels) if v1 else None
        if v1 and DRAINED in report.labels and self._set_aside(row, report):
            return
        self.board.finished(end_reason=report.end_reason, category=report.category, mode=row.mode, failed=report.failed)
        if report.error:
            self.board.error(f"{report.episode_id}: {report.error}")
        if row.mode == "playground":
            # A store we own (D44) is never a store's result: its bundles are the record.
            self.playground_bundles += int(report.prefix is not None)
            return
        if report.prefix is not None and found is not None:
            self._quarantine(row, report, found)
            return
        if report.prefix is not None and report.end_reason is not None and report.ended_at is not None:
            self.results.add(
                FinishedEpisode(
                    store_id=report.store_id,
                    episode_id=report.episode_id,
                    end_reason=report.end_reason,
                    ended_at=report.ended_at,
                    mode=row.mode,
                    provenance=report.provenance,
                    store_source=report.store_source,
                    surface=report.surface,
                    agent=report.agent,
                )
            )
            self._note_result_errors()

    def _set_aside(self, row: ManifestRow, report: EpisodeReport) -> bool:
        """D122: a v1 row the drain cut is not a failure. Its bundle, when it has one, goes to the quarantine as
        `drained` (D111, not counted toward the row's retries); the board counts the row as requeued, so it stays in
        the work left, and names no error for it; the next run of the manifest runs it again. False when its bundle
        could not be moved: then it is counted as before, so nothing is hidden."""
        if report.prefix is not None and not self._quarantine(row, report, DRAINED):
            return False
        self.board.requeued_episode()
        return True

    def _quarantine(self, row: ManifestRow, report: EpisodeReport, found: str) -> bool:
        """D111: move the bundle out of episodes/, write no results row, and queue the row again while it has
        retries left. False when the move failed (the bundle stays where it is, and the board says so)."""
        assert report.prefix is not None
        try:
            move(self.store, report.prefix, reason=found, at=iso(self.clock.now()), episode_id=report.episode_id)
        except Exception as exc:
            self.board.error(f"{report.episode_id}: quarantine failed: {type(exc).__name__}: {exc}")
            self.notes.append(f"{report.episode_id}: not quarantined ({type(exc).__name__}), left at {report.prefix}")
            return False
        self.quarantined += 1
        if found != DRAINED:
            self.attempts[report.episode_id] = self.attempts.get(report.episode_id, 0) + 1
        if found != DRAINED and self.attempts[report.episode_id] <= self.retries:
            self.requeue.append(row)
        return True

    async def retry_quarantined(
        self, scheduler: Callable[[list[ManifestRow]], Scheduler], control: RunControl, *, handle_signals: bool
    ) -> None:
        """Run the quarantined rows again, pass after pass, until none is left with retries or the run drains."""
        while self.requeue and not self.drained:
            rows, self.requeue = self.requeue, []
            self.retried += len(rows)
            self.board.total += len(rows)
            self.notes.append(f"ran {len(rows)} quarantined rows again (same episode ids)")
            await self.execute(scheduler(rows), control, handle_signals=handle_signals)

    def _note_result_errors(self) -> None:
        for message in self.results.errors:
            self.board.error(message)
            self.notes.append(message)
        self.results.errors.clear()

    async def execute(self, scheduler: Scheduler, control: RunControl, *, handle_signals: bool) -> None:
        assert self.status is not None
        control.attach(scheduler)
        restore = _install_signals(control, self.notes) if handle_signals else _nothing
        ticker = asyncio.ensure_future(self.status.loop(self.settings.collect.status_every_s))
        try:
            await scheduler.run()
        finally:
            control.detach()
            restore()
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
        self.notes += [f"skipped {s.store_id}: {s.reason} ({s.episodes} episodes)" for s in scheduler.skips]
        self.drained = scheduler.draining
        if control.repeats:
            self.notes.append(
                f"{control.repeats} repeated stop signals within {control.signal_repeat_s:g} s of the first were "
                "taken as the same stop, not as a request to cancel"
            )
            control.repeats = 0
        if self.board.requeued:
            self.notes.append(f"{self.board.requeued} episodes cut by the drain were set aside to run again")
        self.board.state = "finished"
        self.board.note = _note(drained=self.drained, left=self.board.remaining, skipped=self.board.skipped)
        self.status.write()

    def failed(self, exc: BaseException) -> None:
        self.board.state = "failed"
        self.board.error(f"{type(exc).__name__}: {exc}")
        self.board.note = "the runner itself failed; a new run of the manifest resumes from what is in storage"
        if self.status is not None:
            self.status.write()

    def summary(self) -> RunSummary:
        final = self.board.snapshot()
        return RunSummary(
            run_id=self.run_id,
            manifest_id=self.manifest.id,
            state=final.state,
            started_at=self.started_at,
            ended_at=final.updated_at,
            episodes_total=final.episodes_total,
            episodes_done=final.episodes_done,
            episodes_failed=final.episodes_failed,
            episodes_skipped=final.episodes_skipped,
            episodes_resumed=final.episodes_resumed,
            episodes_left=self.board.remaining,
            bundles=self.results.bundles + self.playground_bundles,
            end_reasons=final.end_reasons,
            recovered=self.recovered,
            drained=self.drained,
            quarantined=self.quarantined,
            retried=self.retried,
            notes=self.notes,
        )


def decisions_offline(settings: Settings, manifest: Manifest) -> bool:
    """Whether the run's episodes decide by rule alone, asking no decision model (D115): collect.decisions off, or a
    sandbox v1 manifest, whose decisions are all in shadow and whose Clef belongs to the judge."""
    return settings.collect.decisions == "off" or manifest.plan.v1 is not None


def _engine(settings: Settings, *, offline: bool = False) -> DecisionEngine:
    """The run's decision engine; `offline`: rules alone, no scorer built and so no Clef or Clef-flash client, each
    abstention keeping its evidence for an offline answer (D115)."""
    from carttrace.decide.factory import build_engine

    if not offline:
        return build_engine(settings)
    rules = settings.model_copy(update={"decide": settings.decide.model_copy(update={"order": ["rule"]})})
    return build_engine(rules, keep_evidence=True)


class _StatusFile:
    """Writes the board to var/status/collector.json, with endpoint load from carttrace.inference and the GPU slots."""

    def __init__(
        self,
        path: Path,
        board: StatusBoard,
        *,
        routes: list[ModelEndpoint],
        inference_stats: InferenceSnapshot | None,
        in_flight: Callable[[], Mapping[str, int]] | None,
        settings: Settings,
        clock: Clock | None = None,
    ) -> None:
        self.path = path
        self.board = board
        self.routes = routes
        self._stats = inference_stats
        self._in_flight = in_flight
        self._settings = settings
        self._caps = LiveCaps.from_disk(settings, clock=clock)

    def write(self) -> CollectorStatus:
        status = self.board.snapshot(self._loads())
        write_status(self.path, status)
        return status

    async def loop(self, every_s: float) -> None:
        """Rewrite the file every `every_s` seconds. A write that fails is recorded on the board and tried again next
        time; if it keeps failing the file goes stale, which is exactly what the watch alerts on."""
        while True:
            await asyncio.sleep(every_s)
            try:
                self.write()
            except OSError as exc:
                self.board.error(f"status file {self.path} not written: {exc.strerror or exc}")

    def _loads(self) -> list[EndpointLoad]:
        stats = process_stats().snapshot() if self._stats is None else self._stats()
        in_flight = gpu_slots(self._settings).in_flight() if self._in_flight is None else self._in_flight()
        caps = {route.endpoint_id: self._caps.cap(route) for route in self.routes}
        return endpoint_loads(stats, in_flight, self.routes, caps=caps)


def _sandbox_rights(manifest: Manifest, bindings: Bindings, worlds: SandboxWorlds) -> dict[str, Rights]:
    """Rights per catalog the manifest's sandbox rows use, and per capture its replica worlds use (each replica
    carries its own store's rights, read from its pack), decided before any episode starts (`rights_key`)."""
    rights: dict[str, Rights] = {}
    for catalog in sorted({world.catalog for world in manifest.worlds.values() if world.capture is None}):
        restricted = worlds.check_catalog(catalog)
        rights[catalog] = sandbox_rights(bindings.licence_rights(catalog), restricted=restricted)
    for envpack, world in sorted(manifest.worlds.items()):
        if world.capture is not None and rights_key(world) not in rights:
            store, profile = manifest.stores[world.store_id], manifest.profiles[world.profile]
            rights[rights_key(world)] = replica_rights(worlds.load(envpack, store, world, profile))
    return rights


def _models(settings: Settings, manifest: Manifest) -> tuple[dict[RunMode, tuple[ModelRef, ...]], list[ModelEndpoint]]:
    """The models each mode's header names, and the routes the status lists before their first call."""
    order = () if decisions_offline(settings, manifest) else settings.decide.order  # D115: no decider is asked
    deciders = [_route(settings, DECIDE_ROLES[s]) for s in order if s in DECIDE_ROLES]
    decide_routes = [route for route in deciders if route is not None]
    decide_refs = tuple(_ref(route) for route in decide_routes)
    routes = list(decide_routes)
    models: dict[RunMode, tuple[ModelRef, ...]] = {"reference": (REFERENCE_MODEL, *decide_refs)}
    if any(row.mode != "reference" for row in manifest.rows):
        agent = _route(settings, "agent")
        if agent is None:
            raise ConfigError(
                "the manifest has rows for the protocol agent but no route for the agent role",
                hint='add "agent": "<endpoint>" to roles in config.json',
            )
        params: dict[str, Any] = {"temperature": settings.agents.temperature, "max_tokens": settings.agents.max_tokens}
        if settings.agents.context_budget_tokens > 0:
            params["context_budget_tokens"] = settings.agents.context_budget_tokens  # D82, in the header as given
        agent_ref = _ref(agent, params)
        models["live"] = models["sandbox"] = models["playground"] = (agent_ref, *decide_refs)
        routes.insert(0, agent)
    return models, routes


def _playgrounds(settings: Settings, rows: Sequence[ManifestRow]) -> dict[str, Any]:
    """The playground stores the run's playground rows shop (data/playgrounds/stores.json); empty without such rows."""
    if not any(row.mode == "playground" for row in rows):
        return {}
    from carttrace.collect.playgrounds import load_playgrounds

    return dict(load_playgrounds(settings))


def _route(settings: Settings, role: str) -> ModelEndpoint | None:
    try:
        return settings.role(role)
    except ConfigError:
        return None


def _ref(route: ModelEndpoint, params: Mapping[str, object] | None = None) -> ModelRef:
    return ModelRef(role=route.role, requested=route.model, endpoint=route.name, params=dict(params or {}))


def _model_names(settings: Settings) -> frozenset[str]:
    names = set(settings.endpoints) | set(settings.roles)
    names |= {endpoint.endpoint_id for endpoint in settings.endpoints.values()}
    return frozenset(names)


def _salt(settings: Settings) -> str:
    name = settings.trace.redaction_salt_env
    if not name:
        return ""
    from carttrace.config import secret

    return secret(name) or ""


def _install_signals(control: RunControl, notes: list[str]) -> Callable[[], None]:
    """SIGTERM and SIGINT drain the run, for as long as it runs (`RunControl.signalled`: a repeat within
    SIGNAL_REPEAT_S is the same stop). Off the main thread signals cannot be handled, and the note says so instead of
    the run pretending to drain."""
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, control.signalled)
        except (NotImplementedError, RuntimeError, ValueError) as exc:
            notes.append(f"{sig.name} is not handled in this thread ({type(exc).__name__}); stop the run another way")
            continue
        installed.append(sig)

    def restore() -> None:
        for sig in installed:
            loop.remove_signal_handler(sig)

    return restore


def _nothing() -> None:
    return None


def _note(*, drained: bool, left: int, skipped: int) -> str:
    parts = []
    if drained:
        parts.append(f"drained with {left} episodes left; a new run of the manifest resumes them")
    elif left:
        parts.append(f"{left} episodes did not finish")
    if skipped:
        parts.append(f"{skipped} episodes skipped with their stores")
    return "; ".join(parts)

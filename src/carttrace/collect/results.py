"""The collector's result writer: one `mc.stores.result.v1` row per finished bundle, into the results database.

Why totals, not increments. `carttrace.stores.results` folds a store's records by time, and a record that sets a trace count
replaces the one before it. So each record the collector writes carries the store's whole count, which starts from
the bundles already in storage (read once when a run starts) and grows as the run finishes episodes. The count is
kept per store and per provenance (carttrace.trace.provenance): a twin episode never raises the count a live or replica
record carries, which is how twin outcomes once reached the sheet as a real store's.

What counts. `traces_collected` counts a store's complete bundles that did not end in `harness_error`: a harness
error is our failure, not a trace of the store. `latest_outcome` is the end reason of the store's most recent episode
of that provenance, whatever it was. Rows go to the results database (carttrace.stores.resultsdb) under its file lock; the
JSONL file is written too only behind collect.results_file_fallback, and a failed write is recorded, never fatal.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from carttrace.core.clock import Clock, iso
from carttrace.core.files import append_jsonl_locked
from carttrace.stores.results import RESULT_SCHEMA, StoreResult
from carttrace.stores.resultsdb import ResultRow, ResultsDB, RunRow
from carttrace.trace.provenance import Provenance, ResultMode
from carttrace.trace.schema import BundleSummary, EndReason

NOT_A_TRACE: frozenset[str] = frozenset({"harness_error"})

Key = tuple[str, Provenance]
"""Store id and provenance: what a count is kept per."""


@dataclass(slots=True)
class ResultsTally:
    """Per store and provenance: how many traces storage holds, and the latest episode's end reason with its end."""

    counts: dict[Key, int] = field(default_factory=dict)
    latest: dict[Key, tuple[str, str]] = field(default_factory=dict)
    """(store id, provenance) to (ended_at, end_reason)."""

    @classmethod
    def from_summaries(cls, summaries: Iterable[BundleSummary]) -> ResultsTally:
        tally = cls()
        for summary in summaries:
            if summary.complete and summary.end_reason is not None:
                tally._add(
                    (summary.store_id, summary.provenance), summary.end_reason, summary.ended_at or summary.started_at
                )
        return tally

    def add(self, store_id: str, provenance: Provenance, end_reason: str, ended_at: str) -> tuple[int, str]:
        """Count one more finished episode; returns the store's (traces_collected, latest_outcome) of that
        provenance."""
        key = (store_id, provenance)
        self._add(key, end_reason, ended_at)
        return self.counts.get(key, 0), self.latest[key][1]

    def _add(self, key: Key, end_reason: str, ended_at: str) -> None:
        if end_reason not in NOT_A_TRACE:
            self.counts[key] = self.counts.get(key, 0) + 1
        before = self.latest.get(key)
        if before is None or ended_at >= before[0]:
            self.latest[key] = (ended_at, end_reason)


def result_record(
    store_id: str, *, observed_at: str, traces_collected: int, latest_outcome: EndReason
) -> dict[str, Any]:
    """One `mc.stores.result.v1` line with the collection fields only, checked against the schema, so the fold never
    mistakes it for a preflight."""
    record: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "store_id": store_id,
        "observed_at": observed_at,
        "traces_collected": traces_collected,
        "latest_outcome": latest_outcome,
    }
    checked = StoreResult.model_validate(record)
    record["observed_at"] = checked.observed_at
    return record


def append_result(path: Path, record: dict[str, Any]) -> None:
    """The file fallback: one line, under the lock the preflight's file writer takes too."""
    append_jsonl_locked(path, record)


@dataclass(frozen=True, slots=True)
class FinishedEpisode:
    """What the writer needs to know about one finished bundle, as the episode's report carries it."""

    store_id: str
    episode_id: str
    end_reason: str
    ended_at: str
    mode: ResultMode
    provenance: Provenance
    store_source: str
    surface: str = ""
    agent: str = ""


class ResultWriter:
    """Writes a results row after every bundle: to the database when one is configured, to the file as a fallback."""

    def __init__(
        self,
        *,
        db_path: Path | None,
        file_path: Path | None,
        tally: ResultsTally,
        clock: Clock,
        run_id: str,
        manifest_id: str,
        mode: ResultMode,
        started_at: str,
        code_version: str = "",
    ) -> None:
        self.db_path = db_path
        self.file_path = file_path
        self.tally = tally
        self.clock = clock
        self.run = RunRow(run_id, manifest_id, mode, started_at, code_version)
        self.errors: list[str] = []
        """Writes that failed, one line each; the runner notes them. Never raised."""
        self.bundles = 0
        self._run_written = False

    def start(self, code_version: str) -> None:
        """Record the run itself once its code version is known; a failure here is noted like any other."""
        self.run = RunRow(self.run.run_id, self.run.manifest_id, self.run.mode, self.run.started_at, code_version)
        if self.db_path is None:
            return
        try:
            with ResultsDB(self.db_path) as db:
                db.write_run(self.run)
            self._run_written = True
        except (OSError, sqlite3.Error) as exc:
            self.errors.append(f"results database {self.db_path} not written for {self.run.run_id}: {exc}")

    def add(self, finished: FinishedEpisode) -> ResultRow:
        """Count the episode and write its row; returns the row whether or not the write succeeded."""
        self.bundles += 1
        traces, latest = self.tally.add(finished.store_id, finished.provenance, finished.end_reason, finished.ended_at)
        record = result_record(
            finished.store_id,
            observed_at=iso(self.clock.now()),
            traces_collected=traces,
            latest_outcome=cast(EndReason, latest),
        )
        row = ResultRow(
            record=StoreResult.model_validate(record),
            mode=finished.mode,
            provenance=finished.provenance,
            store_source=finished.store_source,
            surface=finished.surface,
            agent=finished.agent,
            end_reason=finished.end_reason,
            run_id=self.run.run_id,
            episode_id=finished.episode_id,
        )
        if self.db_path is not None:
            try:
                with ResultsDB(self.db_path) as db:
                    if not self._run_written:
                        db.write_run(self.run)
                        self._run_written = True
                    db.write(row)
            except (OSError, sqlite3.Error) as exc:
                self.errors.append(f"results database {self.db_path} not written for {finished.store_id}: {exc}")
        if self.file_path is not None:
            try:
                append_result(self.file_path, record)
            except OSError as exc:
                message = f"results file {self.file_path} not written for {finished.store_id}: {exc.strerror or exc}"
                self.errors.append(message)
        return row


def recorded_episodes(db_path: Path | None, episode_ids: Iterable[str], *, counted_only: bool = False) -> set[str]:
    """The episodes the per-store results database holds a row for; with `counted_only`, only rows that are not
    harness_error, so a void that will be re-run is not counted done (final review #6)."""
    wanted = sorted(set(episode_ids))
    if not wanted or db_path is None or not db_path.is_file():
        return set()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
    try:
        conn.execute("CREATE TEMP TABLE wanted (episode_id TEXT PRIMARY KEY)")
        conn.executemany("INSERT OR IGNORE INTO wanted VALUES (?)", [(e,) for e in wanted])
        sql = "SELECT DISTINCT r.episode_id FROM results r JOIN wanted w ON r.episode_id = w.episode_id"
        if counted_only:
            sql += " WHERE r.end_reason IS NULL OR r.end_reason != 'harness_error'"
        return {str(row[0]) for row in conn.execute(sql)}
    finally:
        conn.close()

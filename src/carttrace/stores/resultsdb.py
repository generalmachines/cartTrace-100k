"""The per-store results database (var/stores/results.sqlite): every observation about a store, by provenance.

Why a database, and why provenance is a column. The results file was one append-only JSONL that reached 73,000 lines
in a day, and every line carried a real store id whether the episode shopped the store, a replica of its capture or
a twin cut from a public catalog. Folding it per store mixed those, so the sheet sync could report twin outcomes as
the store's own (store-305 said store-305.example and meant an Open Food Facts twin), and nobody could analyse 73,000
lines in one document. Here every row names its provenance (carttrace.trace.provenance) and three views, `store_live`,
`store_replica` and `store_twin`, never mix them; a row whose provenance could not be resolved is `unknown` and in no
view at all. Per-store status reads `STORE_PROVENANCES` (live and replica); twins are stand-in numbers for
aggregates only.

What a row is. The `record` column holds the `mc.stores.result.v1` object as code has always written it (StoreResult
in carttrace.stores.results), so folding and the sheet planner keep their shape; the other columns are the indexable facts
of the observation: the run and episode it came from, mode (live, sandbox, reference), provenance, the store's data
source (team_sheet, abo, off, wands, capture, ...), surface, agent, the outcome (a preflight's outcome, or the judge's
class once one is recorded), the trace's end reason, and the judge version.

Writers. stdlib sqlite3 in WAL mode; every write takes the repository's file lock (carttrace.core.files.file_lock) on a
`.lock` file beside the database, so the collector and a preflight never write at once, and a busy timeout covers
readers. A row's key is a stable hash of what it says, so re-running a migration or a resumed run never duplicates.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carttrace.core.canon import dumps
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.files import file_lock
from carttrace.core.ids import stable_id
from carttrace.stores.results import COLLECTION_FIELDS, StoreResult, fold_results
from carttrace.trace.provenance import PROVENANCES, STORE_PROVENANCES, Provenance, ResultMode

DB_FILE = Path("var") / "stores" / "results.sqlite"
"""Under the repository root; config.json collect.results_db names it."""
NOT_A_TRACE: frozenset[str] = frozenset({"harness_error"})
"""End reasons that are our failure, never a trace of the store (carttrace.collect.results counts the same way)."""
VIEWS: Mapping[Provenance, str] = {"live": "store_live", "replica": "store_replica", "twin": "store_twin"}
"""One view per provenance; there is deliberately none for unknown."""

COLUMNS = (
    "store_id",
    "run_id",
    "episode_id",
    "mode",
    "provenance",
    "store_source",
    "surface",
    "agent",
    "outcome",
    "end_reason",
    "observed_at",
    "judge_version",
)

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    manifest_id TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('live', 'sandbox', 'reference')),
    started_at TEXT NOT NULL,
    code_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    row_key TEXT PRIMARY KEY,
    store_id TEXT NOT NULL,
    run_id TEXT,
    episode_id TEXT,
    mode TEXT NOT NULL CHECK (mode IN ('live', 'sandbox', 'reference')),
    provenance TEXT NOT NULL CHECK (provenance IN ('live', 'replica', 'twin', 'unknown')),
    store_source TEXT NOT NULL,
    surface TEXT NOT NULL,
    agent TEXT NOT NULL,
    outcome TEXT,
    end_reason TEXT,
    observed_at TEXT NOT NULL,
    judge_version TEXT,
    record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS results_by_store ON results (store_id, provenance, observed_at);
CREATE INDEX IF NOT EXISTS results_by_episode ON results (episode_id);
CREATE INDEX IF NOT EXISTS results_by_run ON results (run_id);
"""
_VIEW = "CREATE VIEW IF NOT EXISTS {view} AS SELECT * FROM results WHERE provenance = '{provenance}';"


@dataclass(frozen=True, slots=True)
class ResultRow:
    """One row of `results`: the facts beside the `mc.stores.result.v1` record they index."""

    record: StoreResult
    mode: ResultMode
    provenance: Provenance
    store_source: str
    surface: str = ""
    agent: str = ""
    outcome: str | None = None
    end_reason: str | None = None
    run_id: str | None = None
    episode_id: str | None = None
    judge_version: str | None = None

    @property
    def store_id(self) -> str:
        return self.record.store_id

    @property
    def observed_at(self) -> str:
        return self.record.observed_at

    @property
    def key(self) -> str:
        return stable_id("result", self.store_id, self.observed_at, self.episode_id, self.provenance, self.text)

    @property
    def text(self) -> str:
        """The record as canonical JSON with only the fields it set, as the file wrote it, so a collection record
        read back never looks like a preflight (an empty `protocols` would otherwise replace a real one)."""
        return dumps(self.record.model_dump(mode="json", by_alias=True, exclude_unset=True, exclude_none=True))

    def columns(self) -> dict[str, Any]:
        return {
            "store_id": self.store_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "mode": self.mode,
            "provenance": self.provenance,
            "store_source": self.store_source,
            "surface": self.surface,
            "agent": self.agent,
            "outcome": self.outcome,
            "end_reason": self.end_reason,
            "observed_at": self.observed_at,
            "judge_version": self.judge_version,
        }


@dataclass(frozen=True, slots=True)
class RunRow:
    run_id: str
    manifest_id: str
    mode: ResultMode
    started_at: str
    code_version: str


class ResultsDB:
    """The database at `path`, opened on entry. Reads need no lock; writes take the file lock beside it."""

    def __init__(self, path: Path, *, timeout_s: float = 30.0) -> None:
        self.path = path
        self.lock_path = path.with_suffix(".lock")
        self.timeout_s = timeout_s
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> ResultsDB:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=self.timeout_s, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={int(self.timeout_s * 1000)}")
        with self._locked():
            conn.executescript(_DDL + "".join(_VIEW.format(view=v, provenance=p) for p, v in VIEWS.items()))
        self._conn = conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise ValidationError("the results database is not open", hint="use `with ResultsDB(path) as db:`")
        return self._conn

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with file_lock(self.lock_path):
            yield

    # -- writing --------------------------------------------------------------------------------------------------

    def write(self, row: ResultRow) -> bool:
        """Insert one row; False when a row with the same key is already there."""
        return self.write_many([row]) == 1

    def write_many(self, rows: Iterable[ResultRow]) -> int:
        """Insert rows in one transaction under the file lock; returns how many were new."""
        values = [(r.key, *r.columns().values(), r.text) for r in rows]
        if not values:
            return 0
        placeholders = ", ".join("?" for _ in range(len(COLUMNS) + 2))
        sql = f"INSERT OR IGNORE INTO results (row_key, {', '.join(COLUMNS)}, record) VALUES ({placeholders})"  # noqa: S608
        with self._locked():
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                before = self._count()
                conn.executemany(sql, values)
                after = self._count()
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return after - before

    def write_run(self, run: RunRow) -> None:
        with self._locked():
            self.conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, manifest_id, mode, started_at, code_version) "
                "VALUES (?, ?, ?, ?, ?)",
                (run.run_id, run.manifest_id, run.mode, run.started_at, run.code_version),
            )

    def _count(self) -> int:
        row = self.conn.execute("SELECT count(*) AS n FROM results").fetchone()
        return int(row["n"])

    # -- reading --------------------------------------------------------------------------------------------------

    def rows(
        self,
        *,
        store_id: str | None = None,
        provenances: Sequence[Provenance] | None = None,
        limit: int | None = None,
    ) -> list[ResultRow]:
        """Rows oldest first, filtered by store and provenance. No filter means every row, unknown included."""
        where, args = _filters(store_id, provenances)
        sql = f"SELECT {', '.join(COLUMNS)}, record FROM results{where} ORDER BY observed_at, rowid"  # noqa: S608
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [_row_of(found) for found in self.conn.execute(sql, args)]

    def runs(self) -> list[RunRow]:
        found = self.conn.execute("SELECT * FROM runs ORDER BY started_at, run_id")
        return [RunRow(r["run_id"], r["manifest_id"], r["mode"], r["started_at"], r["code_version"]) for r in found]

    def counts(self) -> dict[Provenance, int]:
        """Rows per provenance, every provenance listed."""
        counts: dict[Provenance, int] = dict.fromkeys(PROVENANCES, 0)
        for found in self.conn.execute("SELECT provenance, count(*) AS n FROM results GROUP BY provenance"):
            counts[found["provenance"]] = int(found["n"])
        return counts

    def view(self, name: str, *, store_id: str | None = None) -> list[ResultRow]:
        """Rows of one provenance's view, by the view's name (store_twin) or its provenance (twin); a view is a table
        of one provenance and nothing else."""
        views: dict[str, str] = {str(k): v for k, v in VIEWS.items()} | {v: v for v in VIEWS.values()}
        if name not in views:
            raise ValidationError(f"no view {name!r}", hint=f"one of {sorted(VIEWS.values())}")
        where, args = _filters(store_id, None)
        sql = f"SELECT {', '.join(COLUMNS)}, record FROM {views[name]}{where} ORDER BY observed_at, rowid"  # noqa: S608
        return [_row_of(found) for found in self.conn.execute(sql, args)]

    def fold(self, provenances: Sequence[Provenance] = STORE_PROVENANCES) -> dict[str, StoreResult]:
        """Every store's folded result over the given provenances only, keyed by store id.

        Preflight fields fold as records do in a file (carttrace.stores.results.fold_results). The trace count and the latest
        outcome are computed from the episode rows of those provenances, never read from a record, so a count never
        includes an episode of another provenance.
        """
        if "unknown" in provenances:
            raise ValidationError("unknown provenance is never folded", hint="pick from live, replica, twin")
        rows = self.rows(provenances=provenances)
        folded = fold_results(r.record for r in rows)
        episodes: dict[str, tuple[int, str | None]] = {}
        for row in rows:
            if row.episode_id is None or row.end_reason is None:
                continue
            count, _ = episodes.get(row.store_id, (0, None))
            episodes[row.store_id] = (count + (row.end_reason not in NOT_A_TRACE), row.end_reason)
        for store_id, result in list(folded.items()):
            found = episodes.get(store_id)
            if found is None:
                if result.model_fields_set & set(COLLECTION_FIELDS):
                    folded[store_id] = result.model_copy(update={"traces_collected": None, "latest_outcome": None})
                continue
            folded[store_id] = result.model_copy(update={"traces_collected": found[0], "latest_outcome": found[1]})
        return folded


def load_store_results(path: Path, provenances: Sequence[Provenance] = STORE_PROVENANCES) -> dict[str, StoreResult]:
    """`ResultsDB.fold` over the database at `path`; the file must exist."""
    if not path.is_file():
        raise NotFoundError(f"results database {path} does not exist", hint="run `carttrace stores results migrate` once")
    with ResultsDB(path) as db:
        return db.fold(provenances)


def export_parquet(path: Path, out: Path, *, provenances: Sequence[Provenance] | None = None) -> int:
    """Write the rows (the indexed columns plus the record as JSON text) to a parquet file; returns the row count."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    with ResultsDB(path) as db:
        rows = db.rows(provenances=provenances)
    table = pa.Table.from_pylist([{**r.columns(), "record": r.text} for r in rows], schema=_arrow_schema())
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")
    return int(table.num_rows)


def _arrow_schema() -> Any:
    import pyarrow as pa

    return pa.schema([(name, pa.string()) for name in (*COLUMNS, "record")])


def _filters(store_id: str | None, provenances: Sequence[Provenance] | None) -> tuple[str, list[str]]:
    clauses: list[str] = []
    args: list[str] = []
    if store_id is not None:
        clauses.append("store_id = ?")
        args.append(store_id)
    if provenances is not None:
        unknown = [p for p in provenances if p not in PROVENANCES]
        if unknown:
            raise ValidationError(f"unknown provenance {unknown}", hint=f"one of {list(PROVENANCES)}")
        clauses.append(f"provenance IN ({', '.join('?' for _ in provenances)})")
        args.extend(provenances)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", args


def _row_of(found: sqlite3.Row) -> ResultRow:
    record = StoreResult.model_validate_json(found["record"])
    return ResultRow(
        record=record,
        mode=found["mode"],
        provenance=found["provenance"],
        store_source=found["store_source"],
        surface=found["surface"],
        agent=found["agent"],
        outcome=found["outcome"],
        end_reason=found["end_reason"],
        run_id=found["run_id"],
        episode_id=found["episode_id"],
        judge_version=found["judge_version"],
    )


def parse_provenances(text: str | None) -> tuple[Provenance, ...] | None:
    """`--provenance live,replica` as a tuple; None (no option) means the store's own, STORE_PROVENANCES."""
    if text is None:
        return None
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    bad = [n for n in names if n not in PROVENANCES]
    if bad or not names:
        raise ValidationError(f"unknown provenance {bad or text!r}", hint=f"one or more of {', '.join(PROVENANCES)}")
    return tuple(n for n in PROVENANCES if n in names)

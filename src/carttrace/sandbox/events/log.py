"""The append-only event log: every event the core reports, numbered, stamped, hash-chained, and optionally written to
`events.jsonl` as it happens.

Why a chain. A grader that reads the log needs to know it is the whole log, in order, unedited. Each record carries the
hash of the one before it, so the last hash (the digest) commits to everything; two runs with the same pack and the
same commands have the same digest, and a log that was cut or edited does not verify.

Sinks. A caller that keeps its own copy of the log (the collector, which writes it into the episode's trace bundle as
`events.jsonl`) attaches a sink: it is handed every record already in the log at once, then each new one as it is
chained, in log order and under the log's lock, so its copy is the log, line for line.

The file is appended one canonical line per record (`carttrace.core.files.append_jsonl`) and never rewritten. A reset starts
a new episode in the same file (`episode` counts up) with a fresh chain; nothing earlier is touched. Records are plain
JSON: `{episode, env, idx, seq, at, type, data, sealed, prev, hash}`.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from carttrace.core.canon import digest, dumps
from carttrace.core.files import append_jsonl, read_jsonl
from carttrace.sandbox.core.event import Event

GENESIS = "0" * 64
EventRecord = dict[str, Any]
"""One log record, as JSON."""
EventSink = Callable[[EventRecord], object]
"""Something handed a copy of every record of the log, in order (see "Sinks")."""


class EventLog:
    def __init__(
        self,
        path: Path | None = None,
        *,
        episode: int = 0,
        env_id: str | None = None,
        sinks: Iterable[EventSink] = (),
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.episode = episode
        self.env_id = env_id
        self._records: list[EventRecord] = []
        self._head = GENESIS
        self._lock = threading.Lock()
        self._sinks: list[EventSink] = list(sinks)

    def append(self, seq: int, at: int, events: Iterable[Event]) -> list[EventRecord]:
        """Number, stamp and chain `events`, write them, and return copies of the records."""
        out = []
        with self._lock:
            for e in events:
                rec: EventRecord = {
                    "episode": self.episode,
                    "env": self.env_id,
                    "idx": len(self._records),
                    "seq": seq,
                    "at": at,
                    "type": e.type,
                    "data": json.loads(dumps(dict(e.data))),
                    "sealed": e.sealed,
                    "prev": self._head,
                }
                rec["hash"] = digest(rec)
                self._head = rec["hash"]
                self._records.append(rec)
                if self.path is not None:
                    append_jsonl(self.path, rec)
                for sink in self._sinks:
                    sink(_copy(rec))
                out.append(_copy(rec))
        return out

    def attach(self, sink: EventSink) -> None:
        """Hand `sink` every record so far, then every new one as it is chained. Attaching the same sink twice is
        one attachment (a bound method equals itself on every access)."""
        with self._lock:
            if any(s == sink for s in self._sinks):
                return
            for rec in self._records:
                sink(_copy(rec))
            self._sinks.append(sink)

    def detach(self, sink: EventSink) -> None:
        """Stop handing records to `sink`; a sink that was never attached is ignored."""
        with self._lock:
            self._sinks = [s for s in self._sinks if s != sink]

    def since(self, n: int = 0, *, include_sealed: bool = True) -> list[EventRecord]:
        """Records with idx >= n. Without sealed ones, for anything the agent may read."""
        with self._lock:
            recs = self._records[max(0, n) :]
            return [_copy(r) for r in recs if include_sealed or not r["sealed"]]

    @property
    def digest(self) -> str:
        """The hash of the last record: it commits to the whole episode."""
        return self._head

    def __len__(self) -> int:
        return len(self._records)

    def new_episode(self, env_id: str | None = None) -> EventLog:
        """A fresh chain for the next episode, appending to the same file and handing records to the same sinks."""
        with self._lock:
            sinks = list(self._sinks)
        return EventLog(self.path, episode=self.episode + 1, env_id=env_id, sinks=sinks)


def _copy(rec: EventRecord) -> EventRecord:
    out: EventRecord = json.loads(dumps(rec))
    return out


def verify(records: Iterable[EventRecord]) -> tuple[bool, str]:
    """Recompute one episode's chain. (True, digest) when it holds, or (False, why it does not)."""
    head = GENESIS
    for i, r in enumerate(records):
        if r.get("idx") != i or r.get("prev") != head:
            return False, f"record {i} is out of order or its predecessor is missing"
        body = {k: v for k, v in r.items() if k != "hash"}
        if digest(body) != r.get("hash"):
            return False, f"record {i} was edited"
        head = str(r["hash"])
    return True, head


def read_events(path: Path, episode: int | None = None) -> list[EventRecord]:
    """The records of an `events.jsonl`, all episodes or one. A torn last line (a writer killed mid-append) is
    skipped."""
    return [r for r in read_jsonl(Path(path)) if episode is None or r.get("episode") == episode]

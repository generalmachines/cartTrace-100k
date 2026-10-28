"""The trace reader: summaries from a store listing, and whole bundles read lazily and checked against bundle.json.

Why two entry points. Pulses, charts and run status count over thousands of bundles and must not pay for turns and
wire records, so `scan` reads two small files per bundle and learns everything else from one listing. The judge,
evals and exports read one bundle at a time, in full, and need proof that it is the bundle that was written:
`open_bundle` reads the small files at once, fetches a log only when it is iterated, and `verify` checks every file
against its sha256 in bundle.json. Both accept a bundle that is still arriving, because collection uploads bundle.json
last and the store is read while collection runs.

A sandbox bundle may carry events.jsonl, the sandbox's own hash-chained event log (D33): `events` reads it,
`check_events` recomputes its chain against the outcome's `state_digest`, and `verify` reports the file when the chain
does not hold. A bundle without it (every live one, and every one written before it) reads and verifies as before.
agent.json (D69) is optional the same way: `agent` reads it, and None means a bundle written before it.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from carttrace.core.canon import sha256_hex
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.files import JsonlError, iter_jsonl_numbered
from carttrace.trace.events import EventsCheck, verify_events
from carttrace.trace.exposure import ExposureSheet
from carttrace.trace.layout import (
    AGENT_FILE,
    DECISIONS_FILE,
    EPISODE_FILE,
    EPISODES_PREFIX,
    EVENTS_FILE,
    EXPOSURE_FILE,
    MANIFEST_FILE,
    OUTCOME_FILE,
    PRUNED_FILE,
    STATE_FILE,
    TURNS_FILE,
    WIRE_FILE,
    TraceStore,
    parse_json_as,
    validate_as,
)
from carttrace.trace.provenance import agent_of, provenance_of, store_source_of, surface_of
from carttrace.trace.schema import (
    AgentDefinition,
    BundleManifest,
    BundleSummary,
    EpisodeHeader,
    Outcome,
    SandboxEvent,
    StateSnapshot,
    Turn,
    WireRecord,
)

Record = Turn | WireRecord | StateSnapshot | dict[str, Any]
"""One line of any of the four logs; decisions stay plain dicts, because carttrace.decide owns their shape."""


def scan(
    store: TraceStore, prefix: str = EPISODES_PREFIX, *, unreadable: list[tuple[str, str]] | None = None
) -> Iterator[BundleSummary]:
    """A summary of every bundle under `prefix`, in key order, read from episode.json and outcome.json alone.

    With `unreadable` given, a bundle whose header or outcome does not parse is appended there as (path, reason) and
    skipped, so one bundle written by newer code than this reader cannot stop a whole scan; without it, the error is
    raised as before.

    A bundle whose bundle.json has not arrived is listed with complete=False; its end_reason and ended_at stay None
    until outcome.json arrives. A directory without episode.json (a consumer's `judge/` folder) is not a bundle.
    """
    for path, names in sorted(_directories(store, prefix).items()):
        if EPISODE_FILE not in names:
            continue
        try:
            header = _read(store, path, EPISODE_FILE, EpisodeHeader)
            outcome = _read(store, path, OUTCOME_FILE, Outcome) if OUTCOME_FILE in names else None
        except ValidationError as exc:
            if unreadable is None:
                raise
            unreadable.append((path, str(exc).splitlines()[0][:300]))
            continue
        yield BundleSummary(
            episode_id=header.episode_id,
            run_id=header.run_id,
            mode=header.mode,
            store_id=header.store.store_id,
            category=header.store.category,
            intent=header.task.intent,
            condition=header.condition.name,
            profile_id=header.profile.id,
            started_at=header.started_at,
            ended_at=None if outcome is None else outcome.ended_at,
            end_reason=None if outcome is None else outcome.end_reason,
            complete=MANIFEST_FILE in names,
            path=path,
            platform=header.store.platform,
            provenance=provenance_of(header),
            agent=agent_of(header),
            surface=surface_of(header),
            manifest_id=header.manifest_id,
            code_version=header.code_version,
            store_source=store_source_of(header),
            pruned=PRUNED_FILE in names,
            infra_labels=() if outcome is None else tuple(outcome.infra_labels),
        )


def resolved(summaries: Iterable[BundleSummary]) -> dict[str, BundleSummary]:
    """One bundle per episode id where several exist (an outage rerun after midnight, a retry under a new run id): the
    one that is not harness_error, then the latest to end, then the path, so every reader counts the same one (final
    review #13)."""
    out: dict[str, BundleSummary] = {}
    for found in summaries:
        held = out.get(found.episode_id)
        if held is None or _rank(found) > _rank(held):
            out[found.episode_id] = found
    return out


def _rank(found: BundleSummary) -> tuple[bool, str, str]:
    return (found.end_reason != "harness_error", found.ended_at or "", found.path)


def completed_episodes(store: TraceStore, prefix: str = EPISODES_PREFIX) -> dict[str, str]:
    """Episode id to bundle path for every bundle whose bundle.json has arrived, from one listing and no reads.

    This is what a resumed runner skips. If an episode id appears under two paths, the later path wins.
    """
    return {
        path.rpartition("/")[2]: path
        for path, names in sorted(_directories(store, prefix).items())
        if EPISODE_FILE in names and MANIFEST_FILE in names
    }


def open_bundle(store: TraceStore, path: str) -> Bundle:
    """One bundle, by the `path` a BundleSummary gives. Reads episode.json, outcome.json and bundle.json now and
    the logs only when they are iterated."""
    path = path.rstrip("/")
    names = frozenset(key.rpartition("/")[2] for key in store.list(f"{path}/") if key.rpartition("/")[0] == path)
    if EPISODE_FILE not in names:
        raise NotFoundError(
            f"no trace bundle at {path!r}: it has no {EPISODE_FILE}",
            hint="pass a path from scan(), such as episodes/<date>/<run_id>/<episode_id>",
        )
    return Bundle(
        store=store,
        path=path,
        files=names,
        header=_read(store, path, EPISODE_FILE, EpisodeHeader),
        outcome=_read(store, path, OUTCOME_FILE, Outcome) if OUTCOME_FILE in names else None,
        manifest=_read(store, path, MANIFEST_FILE, BundleManifest) if MANIFEST_FILE in names else None,
    )


@dataclass(frozen=True, slots=True)
class Bundle:
    """One episode's bundle in a store. Consumers read it; they write their own files beside it, never into it."""

    store: TraceStore
    path: str
    """Store key prefix, e.g. episodes/2026-10-11/<run_id>/<episode_id>."""
    files: frozenset[str]
    """The bundle's own files that were in the store when it was opened."""
    header: EpisodeHeader
    outcome: Outcome | None
    """None while the episode's outcome has not arrived."""
    manifest: BundleManifest | None
    """None until bundle.json arrives, which is last."""

    @property
    def complete(self) -> bool:
        return self.manifest is not None and self.manifest.complete

    @property
    def pruned(self) -> bool:
        """A tombstone (D79, carttrace.trace.prune): its logs were removed on purpose, so they read as empty and `verify`
        names them as missing; pruned.json says what went and why."""
        return PRUNED_FILE in self.files

    def turns(self) -> Iterator[Turn]:
        return self._typed(TURNS_FILE, Turn)

    def wire(self) -> Iterator[WireRecord]:
        return self._typed(WIRE_FILE, WireRecord)

    def state(self) -> Iterator[StateSnapshot]:
        return self._typed(STATE_FILE, StateSnapshot)

    def decisions(self) -> Iterator[dict[str, Any]]:
        return self._decisions()

    def agent(self) -> AgentDefinition | None:
        """What the agent was given (agent.json, D69): its system prompt and tool definitions; None on a bundle
        written before D69, or by an agent that records none."""
        if AGENT_FILE not in self.files:
            return None
        return _read(self.store, self.path, AGENT_FILE, AgentDefinition)

    def exposure(self) -> ExposureSheet | None:
        """What the store exposed per data type and surface (exposure.json, D93); None on a live bundle or one
        written before D93."""
        if EXPOSURE_FILE not in self.files:
            return None
        return _read(self.store, self.path, EXPOSURE_FILE, ExposureSheet)

    @property
    def has_events(self) -> bool:
        """The bundle carries the sandbox's event log (a sandbox episode whose sandbox opened)."""
        return EVENTS_FILE in self.files

    def events(self) -> Iterator[SandboxEvent]:
        """The sandbox's own event log, in chain order; nothing on a bundle without events.jsonl."""
        return self._typed(EVENTS_FILE, SandboxEvent)

    def check_events(self) -> EventsCheck | None:
        """The chain of events.jsonl recomputed, and its head compared with the outcome's state_digest; None on a
        bundle without the file."""
        if not self.has_events:
            return None
        digest_of = None if self.outcome is None else self.outcome.state_digest
        return verify_events(self.events(), digest_of)

    def timeline(self) -> Iterator[tuple[str, Record]]:
        """Every record of the four logs in the order it was written, as (file name, record). The writer numbers
        all four from one counter, so merging by seq restores the episode's single order."""
        logs: list[Iterator[tuple[int, str, Record]]] = [
            ((r.seq, TURNS_FILE, r) for r in self.turns()),
            ((r.seq, WIRE_FILE, r) for r in self.wire()),
            ((r.seq, STATE_FILE, r) for r in self.state()),
            ((int(r["seq"]), DECISIONS_FILE, r) for r in self.decisions()),
        ]
        for _, name, record in heapq.merge(*logs, key=lambda entry: entry[0]):
            yield name, record

    def verify(self) -> list[str]:
        """The files that do not match bundle.json, sorted: changed, missing, or (for bundle.json itself) written
        for another episode. Empty when the bundle is intact."""
        if self.manifest is None:
            raise ValidationError(
                f"{self.path} has no {MANIFEST_FILE} yet, so there is nothing to verify it against",
                hint="its upload is still running or was interrupted; run recover_staged where it was written",
            )
        bad = {MANIFEST_FILE} if self.manifest.episode_id != self.header.episode_id else set()
        for name, expected in self.manifest.files.items():
            key = f"{self.path}/{name}"
            if not self.store.exists(key) or sha256_hex(self.store.get_bytes(key)) != expected:
                bad.add(name)
        if EVENTS_FILE in self.manifest.files and EVENTS_FILE not in bad and self.has_events:
            try:
                checked = self.check_events()
            except ValidationError:
                checked = None
                bad.add(EVENTS_FILE)
            if checked is not None and not checked.ok:
                bad.add(EVENTS_FILE)
        return sorted(bad)

    def _lines(self, name: str) -> Iterator[tuple[str, Any]]:
        """(where, record) for each record of one log. A torn final line, from a writer killed mid-append, is
        skipped, as for every JSONL file (carttrace.core.files); any other bad line raises ValidationError."""
        if name not in self.files:
            return
        key = f"{self.path}/{name}"
        try:
            for number, record in iter_jsonl_numbered(self.store.get_bytes(key), source=key):
                yield f"{key} line {number}", record
        except JsonlError as exc:
            raise ValidationError(
                exc.message,
                hint="bundle files are written once and never edited; compare it with bundle.json using verify()",
            ) from exc

    def _typed[M: BaseModel](self, name: str, model: type[M]) -> Iterator[M]:
        for where, obj in self._lines(name):
            yield validate_as(model, obj, where=where)

    def _decisions(self) -> Iterator[dict[str, Any]]:
        for where, obj in self._lines(DECISIONS_FILE):
            if not isinstance(obj, dict) or not isinstance(obj.get("seq"), int):
                raise ValidationError(f"{where} is not a decision record", hint="decision records are JSON objects")
            yield obj


def _directories(store: TraceStore, prefix: str) -> dict[str, set[str]]:
    """File names per directory, from one listing."""
    out: dict[str, set[str]] = {}
    for key in store.list(prefix):
        directory, _, name = key.rpartition("/")
        out.setdefault(directory, set()).add(name)
    return out


def _read[M: BaseModel](store: TraceStore, path: str, name: str, model: type[M]) -> M:
    key = f"{path}/{name}"
    return parse_json_as(model, store.get_bytes(key), where=key)

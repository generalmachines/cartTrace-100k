"""Quarantine (D111): a bundle that ended in a system problem never counts and never ships.

Why. A harness error, a model error, a crash or a collector restart's `drained` cut is our failure,
not a trace of the agent or the store. Such rows used to stay among the traces and be counted apart (the preflight's
"errors under 1%", the dashboard's error column), so every reader had to know the rule. Now the collector moves a v1
row's bundle (a row with an experiment block; older lanes keep theirs) out of `episodes/` to
`quarantine/<date>/<run>/<episode>/` as soon as it is uploaded, writes no results row for it, and runs the row again:
the same manifest row, so the same episode id (task, store, seed, world, surface and rail are the row's), up to
`collect.quarantine_retries` times (default 2). A second quarantine of one episode under one run goes to
`<episode>~2`, and so on. A later run of the manifest finds the row missing from storage and runs it again too, until
its quarantined attempts (drained ones aside: a restart is not the row's fault) pass the limit. Every reader that
counts from `episodes/` (progress, the dashboard's fold, the preflight, the release) never sees a quarantined bundle;
the quarantine is counted apart (`counts`).

`carttrace collect quarantine --sweep` applies the rule to the bundles already in storage (dry-run unless `--apply`).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from carttrace.core.errors import MCError
from carttrace.storage.service import BlobStore
from carttrace.trace.layout import EPISODES_PREFIX, EXPERIMENT_FILE
from carttrace.trace.reader import scan

QUARANTINE_PREFIX: Final = "quarantine/"
REASON_FILE: Final = "quarantine.json"
"""Written inside a quarantined bundle: why, from where, and when."""
SCHEMA: Final = "mc.collect.quarantine.v1"

HARNESS_ERROR: Final = "harness_error"
MODEL_ERROR: Final = "model_error"
CRASH: Final = "crash"
DRAINED: Final = "drained"
DECIDER_UNAVAILABLE: Final = "decider_unavailable"
"""A decision the shopper or the supervisor needed could not reach its model (the judge's gate of the same name, which
voids the row): our failure, run again (sandbox v1 final, D115)."""
PROBLEMS: Final = (DRAINED, MODEL_ERROR, CRASH, DECIDER_UNAVAILABLE, HARNESS_ERROR)
SHOPPER_DECISIONS: Final = frozenset({"shopper.field"})
SUPERVISOR_DECISIONS: Final = frozenset({"supervise.looping", "supervise.stuck", "supervise.boundary_reached"})
CAP_ENDS: Final = frozenset({"step_cap", "time_cap"})
"""The judge's rule (carttrace.judge.gates `_decider_down`, judge-1.0), read here without importing the frozen judge: a
shopper decision that abstained on a backend error, or a supervisor one on a journey that then ran to a cap."""
"""The system problems, in the order a bundle is named by the first that applies."""


def problem(end_reason: str | None, labels: Iterable[str] = ()) -> str | None:
    """The system problem a finished bundle shows, or None: a `drained` label (a collector restart cut it), a model
    error (an infra label naming the model, as `model_unavailable`: the preflight's rule since D96), a crash (an infra
    label naming one), or an end in harness_error."""
    found = [str(x) for x in labels]
    if DRAINED in found:
        return DRAINED
    if any("model" in label for label in found):
        return MODEL_ERROR
    if any(CRASH in label for label in found):
        return CRASH
    if DECIDER_UNAVAILABLE in found:
        return DECIDER_UNAVAILABLE
    if end_reason == HARNESS_ERROR:
        return HARNESS_ERROR
    return None


def decider_down(decisions: Iterable[Mapping[str, Any]], end_reason: str | None) -> bool:
    """Whether a decision the journey needed abstained because its model could not be reached: a shopper decision, or
    a supervisor decision on a journey that ran to a step or time cap (the judge's `decider_unavailable`)."""
    failed = [d for d in decisions if d.get("decided_by") == "abstain" and d.get("error")]
    if any(d.get("type_id") in SHOPPER_DECISIONS for d in failed):
        return True
    return end_reason in CAP_ENDS and any(d.get("type_id") in SUPERVISOR_DECISIONS for d in failed)


def quarantine_prefix(prefix: str) -> str:
    """`episodes/<date>/<run>/<episode>` to `quarantine/<date>/<run>/<episode>`."""
    path = prefix.rstrip("/")
    if not path.startswith(EPISODES_PREFIX):
        raise MCError(f"{prefix!r} is not a bundle under {EPISODES_PREFIX}")
    return QUARANTINE_PREFIX + path[len(EPISODES_PREFIX) :]


def move(store: BlobStore, prefix: str, *, reason: str, at: str, episode_id: str) -> str:
    """Move the bundle at `prefix` into the quarantine, with a reason file; returns its new prefix. Every key is
    written at its new place before any old one is removed, so a crash part way leaves a copy, never a loss; a bundle
    left in both places is moved again by the next sweep (the copy is overwritten with the same bytes)."""
    first = quarantine_prefix(prefix)
    to, n = first, 1
    while store.exists(f"{to}/{REASON_FILE}"):  # a retry under the same run quarantined again: its own folder
        n += 1
        to = f"{first}~{n}"
    base = prefix.rstrip("/") + "/"
    keys = store.list(base)
    for key in keys:
        store.put_bytes(to + "/" + key[len(base) :], store.get_bytes(key))
    body = {"schema": SCHEMA, "episode_id": episode_id, "problem": reason, "from": prefix.rstrip("/"), "at": at}
    store.put_bytes(f"{to}/{REASON_FILE}", (json.dumps(body, sort_keys=True) + "\n").encode())
    for key in keys:
        store.delete(key)
    return to


@dataclass(frozen=True, slots=True)
class Quarantined:
    episode_id: str
    path: str
    problem: str


def listed(store: BlobStore, episode_ids: Iterable[str] | None = None) -> list[Quarantined]:
    """The quarantined bundles (of `episode_ids` only, when given), from their reason files."""
    wanted = None if episode_ids is None else set(episode_ids)
    out: list[Quarantined] = []
    for key in store.list(QUARANTINE_PREFIX):
        if not key.endswith("/" + REASON_FILE):
            continue
        path = key[: -len(REASON_FILE) - 1]
        episode_id = path.rsplit("/", 1)[-1].split("~", 1)[0]
        if wanted is not None and episode_id not in wanted:
            continue
        try:
            body = json.loads(store.get_bytes(key))
        except (MCError, ValueError):
            body = {}
        out.append(Quarantined(episode_id, path, str(body.get("problem") or HARNESS_ERROR)))
    return out


def attempts(found: Iterable[Quarantined]) -> Counter[str]:
    """Each episode's quarantined attempts that count toward its retry limit: every problem but `drained`."""
    return Counter(q.episode_id for q in found if q.problem != DRAINED)


def counts(found: Iterable[Quarantined]) -> dict[str, int]:
    """Quarantined bundles by problem, for the reports that count them apart."""
    by = Counter(q.problem for q in found)
    return {p: by[p] for p in PROBLEMS if by[p]}


@dataclass(frozen=True, slots=True)
class SweepItem:
    episode_id: str
    path: str
    problem: str
    moved_to: str | None


def sweep(store: BlobStore, *, apply: bool, at: str, prefix: str = EPISODES_PREFIX) -> list[SweepItem]:
    """Every complete v1 bundle (one with experiment.json, as the collector quarantines only v1 rows) under `prefix`
    that shows a system problem; moved to the quarantine when `apply`. Older lanes' bundles stay where they are."""
    unreadable: list[tuple[str, str]] = []
    out: list[SweepItem] = []
    for summary in scan(store, prefix, unreadable=unreadable):
        if not summary.complete or summary.end_reason is None:
            continue
        if not store.exists(f"{summary.path.rstrip('/')}/{EXPERIMENT_FILE}"):
            continue
        found = problem(summary.end_reason, _labels(store, summary.path))
        if found is None and _decider_voided(store, summary.path, summary.end_reason):
            found = DECIDER_UNAVAILABLE
        if found is None:
            continue
        moved = move(store, summary.path, reason=found, at=at, episode_id=summary.episode_id) if apply else None
        out.append(SweepItem(summary.episode_id, summary.path, found, moved))
    return out


JUDGE_DIR: Final = "judge/"
DECISIONS_FILE: Final = "decisions.jsonl"


def _decider_voided(store: BlobStore, path: str, end_reason: str | None) -> bool:
    """A bundle the decision model's outage shaped: its judge card's gate says `decider_unavailable`, or, with no card,
    its own decisions.jsonl shows the outage by the judge's rule (`decider_down`). Before D115 such a row finished as
    an ordinary trace (2026-10-17: 333 Qwen and 8 reference v1 traces judged VOID decider_unavailable)."""
    base = path.rstrip("/") + "/"
    cards = [k for k in store.list(base + JUDGE_DIR) if k.endswith(".json")]
    for key in cards:
        try:
            card = json.loads(store.get_bytes(key))
        except (MCError, ValueError):
            continue
        gate = card.get("gate") if isinstance(card, Mapping) else None
        reasons = gate.get("reasons") if isinstance(gate, Mapping) else None
        if any(isinstance(r, Mapping) and r.get("label") == DECIDER_UNAVAILABLE for r in reasons or ()):
            return True
    if not store.exists(base + DECISIONS_FILE):
        return False
    try:
        lines = store.get_bytes(base + DECISIONS_FILE).decode().splitlines()
    except (MCError, UnicodeDecodeError):
        return False
    decisions: list[Mapping[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, Mapping):
            decisions.append(row)
    return decider_down(decisions, end_reason)


def _labels(store: BlobStore, path: str) -> Sequence[str]:
    try:
        body: Mapping[str, Any] = json.loads(store.get_bytes(f"{path.rstrip('/')}/outcome.json"))
    except (MCError, ValueError):
        return ()
    return [str(x) for x in body.get("infra_labels") or ()]

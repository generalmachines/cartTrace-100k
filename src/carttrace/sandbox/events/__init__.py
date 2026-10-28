"""The sandbox's record of what happened, and the runtime that writes it: an append-only, hash-chained event log
(`events.jsonl`), the running `Sandbox` with its lock and optional injected clock, replay from recorded commands or
from the log alone to the same state hash and digest, the state and events-since-N reads, `diff(before, after)`, and
the grader's diff check (expected inside the change, the change inside expected plus allowed)."""

from carttrace.sandbox.events.diff import ANY, BOOKKEEPING, DiffCheck, check, diff, flatten, matches, path
from carttrace.sandbox.events.log import GENESIS, EventLog, EventRecord, EventSink, read_events, verify
from carttrace.sandbox.events.runtime import (
    EVENTS_SCHEMA,
    MESSAGE_EVENTS,
    REJECTED,
    Outcome,
    Sandbox,
    audit_episode,
    commands_from_events,
    epoch_ms,
    replay,
    replay_events,
)

__all__ = [
    "ANY",
    "BOOKKEEPING",
    "EVENTS_SCHEMA",
    "GENESIS",
    "MESSAGE_EVENTS",
    "REJECTED",
    "DiffCheck",
    "EventLog",
    "EventRecord",
    "EventSink",
    "Outcome",
    "Sandbox",
    "audit_episode",
    "check",
    "commands_from_events",
    "diff",
    "epoch_ms",
    "flatten",
    "matches",
    "path",
    "read_events",
    "replay",
    "replay_events",
    "verify",
]

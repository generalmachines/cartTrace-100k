"""Checking a bundle's events.jsonl: the sandbox's hash chain, recomputed from the bundle alone (D33).

Why here and not in carttrace.sandbox. The judge, the bench and `carttrace trace verify` read bundles, and a bundle is a layer 3
contract that must not need the sandbox to be checked: the chain rule is small (each line's `hash` is the sha256 of
the canonical JSON of its other fields, and its `prev` is the hash before it), so it is restated here.
`tests/trace/test_events.py` holds it equal to `carttrace.sandbox.events.verify` on a real sandbox log.

What a check can prove. A line no redaction rule touched is recomputed in full. A line a rule changed (a persona's
email in a checkout command body, say) keeps the sandbox's own `prev` and `hash`, so the links still hold, but its own
hash is taken on trust and counted in `trusted`. The head must equal the outcome's `state_digest.events` when the
outcome names one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from carttrace.core.canon import digest
from carttrace.trace.schema import SandboxEvent, StateDigest

GENESIS = "0" * 64
"""The `prev` of an episode's first line (carttrace.sandbox.events.GENESIS)."""


@dataclass(frozen=True, slots=True)
class EventsCheck:
    ok: bool
    head: str
    """The hash of the last line, or GENESIS for an empty log."""
    count: int
    trusted: int = 0
    """Redacted lines whose own hash could not be recomputed; their links were checked."""
    problem: str = ""
    """What failed, when ok is False."""


def verify_events(lines: Iterable[SandboxEvent], digest_of: StateDigest | None = None) -> EventsCheck:
    """Recompute the chain of every episode in `lines`, in order, and compare the head with `digest_of`."""
    head, expect_idx, episode = GENESIS, 0, None
    count = trusted = 0
    for line in lines:
        if line.episode != episode:
            episode, head, expect_idx = line.episode, GENESIS, 0
        if line.idx != expect_idx or line.prev != head:
            problem = f"line {count + 1} is out of order or its predecessor is gone"
            return EventsCheck(False, head, count, trusted, problem)
        if line.redactions:
            trusted += 1
        else:
            body = line.model_dump(mode="json")
            body.pop("hash")
            if digest(body) != line.hash:
                return EventsCheck(False, head, count, trusted, f"line {count + 1} was edited")
        head, expect_idx, count = line.hash, expect_idx + 1, count + 1
    if digest_of is not None:
        if digest_of.count != count:
            problem = f"the outcome counts {digest_of.count} events, found {count}"
            return EventsCheck(False, head, count, trusted, problem)
        if digest_of.events != head:
            return EventsCheck(
                False, head, count, trusted, f"the chain ends at {head[:12]}, the outcome names {digest_of.events[:12]}"
            )
    return EventsCheck(True, head, count, trusted)

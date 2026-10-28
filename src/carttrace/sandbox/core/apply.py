"""`apply(state, command, config) -> (state', events)`: the only way the store changes.

Why one function. Every judgment about a journey should come from what the store's state did, the way tau2-bench,
AppWorld and ORO grade from backend state. That only works if every change goes through one door that validates the
command, enforces trust policy, deduplicates retries, fires the environment's sealed faults, moves statuses only along
the transition tables, checks the invariants, and says what happened as events. Adapters for UCP, ACP, MPP and AP2
translate their wire calls into commands here; none of them holds state of its own.

Shape of a call. The command is parsed (a malformed one is a `command.rejected` with a typed validation error, never a
traceback); the clock ticks; expired reservations are released; scheduled faults fire; the session's tier is checked;
an idempotency key is looked up; then the handler runs on a working copy. A handler that raises leaves only the
bookkeeping (sequence, clock, faults fired, expiries) behind. A handler that returns an error keeps its side effects:
a declined payment really was declined. The new state is committed only if every invariant holds.

Pure and deterministic: no wall clock, no randomness. Ids come from the seed and a counter, time from the command's
`at` or the previous time plus the pack's tick.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from carttrace.sandbox.core import commands as C
from carttrace.sandbox.core.config import Config
from carttrace.sandbox.core.context import Ctx
from carttrace.sandbox.core.errors import (
    BusinessRule,
    Conflict,
    CoreError,
    IllegalTransition,
    InvalidCommand,
    InvariantViolation,
    NotFound,
    PolicyDenied,
)
from carttrace.sandbox.core.errors import from_dict as error_from_dict
from carttrace.sandbox.core.event import Event
from carttrace.sandbox.core.fault_effects import run_scheduled
from carttrace.sandbox.core.handlers import HANDLERS
from carttrace.sandbox.core.invariants import check as check_invariants
from carttrace.sandbox.core.state import IdemRecord, Session, State, Variant
from carttrace.sandbox.core.stock import expire
from carttrace.sandbox.core.trust import authorize
from carttrace.sandbox.core.tx import Tx

STORED_ERRORS = (BusinessRule, IllegalTransition, NotFound, PolicyDenied)
"""Outcomes an idempotency key remembers. Retryable failures (an injected 5xx, a 409, a rate limit) and malformed
requests are not remembered, so a retry with the same key can succeed."""

REF_MAPS = {
    "cart_id": "carts",
    "checkout_id": "checkouts",
    "payment_id": "payments",
    "order_id": "orders",
    "credential_id": "credentials",
    "request_id": "payment_requests",
    "mandate_id": "mandates",
}
"""Refs that name an entity, and the map it lives in."""


def _bookkeeping(state: State, cmd: C.Command | None, config: Config) -> Tx:
    tx = Tx(state)
    tx.seq = state.seq + 1
    at = cmd.at if cmd is not None else None
    tx.clock = max(state.clock, at) if at is not None else state.clock + config.tick_ms
    return tx


def apply(state: State, command: object, config: Config) -> tuple[State, list[Event]]:
    """Apply one command. Always returns a state and events; the last event is `command.applied` or
    `command.rejected`."""
    try:
        cmd = C.parse(command)
    except InvalidCommand as e:
        return reject(state, C.raw_type(command), e, config)
    tx = _bookkeeping(state, cmd, config)
    events: list[Event] = [Event("command", {"body": C.dump(cmd)})]
    ctx = Ctx(tx, cmd, config, events)
    error: CoreError | None = None
    replayed = False
    try:
        expire(ctx)
        run_scheduled(ctx)
        authorize(ctx)
        actor = cmd.session if cmd.actor == "agent" else f"@{cmd.actor}"
        idem_key = f"{actor}|{cmd.idempotency_key}" if cmd.idempotency_key and cmd.type in C.MUTATING else None
        rec = ctx.tx.idempotency.get(idem_key)
        if rec is not None:
            if rec.body_hash != C.body_hash(cmd) or rec.command != cmd.type:
                raise Conflict(
                    code="idempotency_conflict",
                    message="this idempotency key was used with a different request",
                    field="idempotency_key",
                    hint="use a new idempotency key for a different request",
                )
            ctx.refs.update(rec.outcome.get("refs") or {})
            events.append(Event("command.replayed", {"original_seq": rec.seq, "outcome": dict(rec.outcome)}))
            replayed = True
            stored = rec.outcome.get("error")
            error = error_from_dict(stored) if stored else None
        else:
            error = _run_handler(ctx, cmd)
            if idem_key is not None and (error is None or isinstance(error, STORED_ERRORS)):
                remembered = {"ok": error is None, "refs": dict(ctx.refs), "error": error.to_dict() if error else None}
                record = IdemRecord(actor, str(cmd.idempotency_key), cmd.type, C.body_hash(cmd), ctx.tx.seq, remembered)
                ctx.tx.idempotency.put(idem_key, record)
    except CoreError as e:
        error = e
    new = ctx.tx.freeze()
    problems = check_invariants(state, new)
    if problems:
        keep = _bookkeeping(state, cmd, config).freeze()
        err = InvariantViolation(
            code="invariant_violated", message=problems[0], details=tuple({"problem": p} for p in problems)
        )
        return keep, [events[0], Event("command.rejected", {"command": cmd.type, "error": err.to_dict(), "refs": {}})]
    refs = dict(ctx.refs)
    if error is None:
        events.append(Event("command.applied", {"command": cmd.type, "refs": refs, "replayed": replayed}))
    else:
        events.append(
            Event(
                "command.rejected",
                {"command": cmd.type, "error": error.to_dict(), "refs": refs, "replayed": replayed},
            )
        )
    return new, events


def _run_handler(ctx: Ctx, cmd: C.Command) -> CoreError | None:
    """Run the command's handler on a fresh working copy over the bookkeeping. If it raises, drop its changes, its
    events and any ref to an entity that no longer exists, and keep the bookkeeping."""
    base = ctx.tx.freeze()
    ctx.tx = Tx(base)
    n_events = len(ctx.events)
    try:
        return HANDLERS[type(cmd)](ctx, cmd)
    except CoreError as e:
        del ctx.events[n_events:]
        ctx.tx = Tx(base)
        for key, name in REF_MAPS.items():
            if key in ctx.refs and ctx.refs[key] not in ctx.tx.read(name):
                del ctx.refs[key]
        return e


def reject(state: State, command_type: str | None, error: CoreError, config: Config) -> tuple[State, list[Event]]:
    """Record a command that could not become a core command (malformed, or refused by an adapter before
    translation) the way `apply` records one it could not parse: the sequence and clock move, and a
    `command.rejected` with the typed error is the only event."""
    tx = _bookkeeping(state, None, config)
    return tx.freeze(), [Event("command.rejected", {"command": command_type, "error": error.to_dict(), "refs": {}})]


def initial_state(
    *, seed: int, currency: str, inventory: Mapping[str, Variant], clock: int, sessions: Iterable[Session] = ()
) -> State:
    return State(
        seed=seed, currency=currency, clock=clock, inventory=dict(inventory), sessions={s.id: s for s in sessions}
    )


def outcome(events: Sequence[Event]) -> dict[str, Any]:
    """The last event of an apply, as {ok, command, refs, error, replayed}."""
    last = events[-1]
    return {
        "ok": last.type == "command.applied",
        "command": last.data.get("command"),
        "refs": dict(last.data.get("refs") or {}),
        "error": last.data.get("error"),
        "replayed": bool(last.data.get("replayed")),
    }

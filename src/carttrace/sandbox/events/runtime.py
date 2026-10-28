"""The running sandbox: one environment pack, its current state, its event log, a lock, and the API the protocol
surfaces and a grader use (execute a command, hand off a live cart, read the state, read events since N, diff two
states, reset, replay).

Why a lock and not a database. A sandbox episode is small and every command must see the state the previous one left.
Concurrent agent sessions against one store serialise here, so stock and reservations are decided in one order, the
order is the log's order, and replaying the recorded commands reproduces it exactly.

Determinism. Ids come from the pack's seed and a counter, and time from the logical clock, which advances the pack's
tick per command unless a `Clock` is injected; an injected clock's readings are stamped onto the commands as `at` and
recorded, so a replay needs no clock at all. The log opens every episode with an `episode.started` record naming the
pack, the seed and the initial state's hash, and every accepted command's full body is in its `command` record, so the
log alone is enough to replay an episode and prove it reaches the same digest (`replay_events`).

What is recorded. A command that fails validation is recorded only as its type (when identifier-shaped) and its typed
error, never as the raw input, so nothing a refused command carried (a card number, a secret) is kept for replay.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from carttrace.core.clock import Clock
from carttrace.core.errors import ValidationError
from carttrace.sandbox.core.apply import apply, outcome, reject
from carttrace.sandbox.core.commands import dump, parse, raw_type
from carttrace.sandbox.core.errors import CoreError, InvalidCommand
from carttrace.sandbox.core.errors import from_dict as error_from_dict
from carttrace.sandbox.core.event import Event
from carttrace.sandbox.core.fault_effects import start_effects
from carttrace.sandbox.core.state import State
from carttrace.sandbox.envpack.handoff import handoff_command
from carttrace.sandbox.envpack.pack import EnvPack
from carttrace.sandbox.events.diff import BOOKKEEPING, diff
from carttrace.sandbox.events.log import EventLog, EventRecord, EventSink, verify
from carttrace.trace.schema import Cart

EVENTS_SCHEMA = "mc.sandbox.events.v1"
REJECTED = "__rejected__"
"""The type of a recorded stand-in for a command that never became a core command."""
MESSAGE_EVENTS = (
    "line.rejected",
    "line.clamped",
    "discount.rejected",
    "fulfillment.rejected",
    "checkout.repriced",
    "checkout.stock_lost",
)
"""Events an adapter turns into protocol messages: refusals and changes the agent can act on."""
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def epoch_ms(clock: Clock) -> int:
    """The clock's reading in whole milliseconds since the epoch, in integer arithmetic."""
    return (clock.now() - _EPOCH) // timedelta(milliseconds=1)


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one command did, as an adapter renders it."""

    ok: bool
    command: str | None
    refs: Mapping[str, Any]
    """Ids of what the command touched: cart_id, checkout_id, payment_id, order_id, challenge_id, action_required."""
    error: Mapping[str, Any] | None
    """The typed error as JSON (`CoreError.to_dict`), or None."""
    replayed: bool
    """True when an idempotency key answered with the original outcome."""
    seq: int
    events: tuple[EventRecord, ...] = ()

    def messages(self) -> list[EventRecord]:
        """The events an adapter turns into protocol messages, never sealed ones."""
        return [e for e in self.events if not e["sealed"] and e["type"] in MESSAGE_EVENTS]


class Sandbox:
    def __init__(
        self,
        pack: EnvPack,
        *,
        log_path: Path | None = None,
        clock: Clock | None = None,
        seed: int | None = None,
        episode: int = 0,
    ) -> None:
        """`seed` overrides the pack's seed. `clock`, when given, replaces the pack's tick: each command is stamped
        with its reading in epoch milliseconds, and the logical clock never moves backwards, so a clock that reads
        earlier than the pack's `clock_start_ms` holds time still. `episode` numbers the log's first episode; a replay
        of episode N of a file passes N, because the number is part of every hashed record."""
        self._pack = pack
        self._config = pack.config()
        self._clock = clock
        self._lock = threading.RLock()
        self._log = EventLog(log_path, episode=episode, env_id=pack.id)
        self._commands: list[Any] = []
        self._open_episode(seed)

    def _open_episode(self, seed: int | None) -> None:
        """Log `episode.started` over the pack's initial state, then fire the schedule's `at_start` faults as sealed
        events at the same sequence number: the episode starts from the world they leave."""
        s = self._pack.initial_state(seed)
        data = {"schema": EVENTS_SCHEMA, "env": self._pack.id, "seed": s.seed, "state": s.hash()}
        self._log.append(s.seq, s.clock, [Event("episode.started", {**data, "profile": self._pack.profile.id})])
        s, events = start_effects(s, self._config)
        if events:
            self._log.append(s.seq, s.clock, events)
        self._state = self._initial = s

    # -- commands ------------------------------------------------------------------------------------------------------

    def execute(self, command: object) -> Outcome:
        """Apply one command: a JSON object or a typed command from `carttrace.sandbox.core.commands`."""
        with self._lock:
            stamped = self._stamp(command)
            try:
                parsed = parse(stamped)
            except InvalidCommand as e:
                return self._record_rejection(raw_type(stamped), e)
            self._commands.append(dump(parsed))
            return self._commit(*apply(self._state, parsed, self._config))

    def reject(self, command_type: str | None, error: CoreError) -> Outcome:
        """Log a command an adapter refused before it could become a core command (a malformed wire argument)."""
        with self._lock:
            return self._record_rejection(command_type, error)

    def execute_many(self, commands: Iterable[object]) -> list[Outcome]:
        return [self.execute(c) for c in commands]

    def handoff(self, cart: Cart, *, session: str = "buyer", country: str | None = None) -> Outcome:
        """Re-create a live cart for `session`, as a system command in the log. The pack should come from
        `build_envpack_from_cart` on the same cart; the outcome's refs name the new cart."""
        return self.execute(handoff_command(cart, session=session, country=country))

    def _stamp(self, command: object) -> object:
        if self._clock is None:
            return command
        if isinstance(command, Mapping) and command.get("at") is None:
            return {**command, "at": epoch_ms(self._clock)}
        if isinstance(command, BaseModel) and getattr(command, "at", 0) is None:
            return command.model_copy(update={"at": epoch_ms(self._clock)})
        return command

    def _record_rejection(self, command_type: str | None, error: CoreError) -> Outcome:
        self._commands.append({"type": REJECTED, "command": command_type, "error": error.to_dict()})
        return self._commit(*reject(self._state, command_type, error, self._config))

    def _commit(self, new: State, events: Sequence[Event]) -> Outcome:
        self._state = new
        records = self._log.append(new.seq, new.clock, events)
        o = outcome(events)
        return Outcome(o["ok"], o["command"], o["refs"], o["error"], o["replayed"], new.seq, tuple(records))

    # -- reads ---------------------------------------------------------------------------------------------------------

    @property
    def pack(self) -> EnvPack:
        return self._pack

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    @property
    def initial(self) -> State:
        """The state this episode started from."""
        with self._lock:
            return self._initial

    @property
    def episode(self) -> int:
        return self._log.episode

    @property
    def commands(self) -> list[Any]:
        """Every command of this episode as recorded for replay: parsed bodies, and stand-ins for refusals."""
        with self._lock:
            return list(self._commands)

    def state_dict(self) -> dict[str, Any]:
        return self.state.to_dict()

    def state_hash(self) -> str:
        return self.state.hash()

    def attach(self, sink: EventSink) -> None:
        """Hand `sink` every record of the log so far (the opening and its at_start faults included), then each new
        one as it is chained, across resets (`EventLog.attach`). The collector writes a bundle's events.jsonl this
        way."""
        with self._lock:
            self._log.attach(sink)

    def detach(self, sink: EventSink) -> None:
        with self._lock:
            self._log.detach(sink)

    def events_since(self, n: int = 0, *, include_sealed: bool = True) -> list[EventRecord]:
        return self._log.since(n, include_sealed=include_sealed)

    @property
    def log_length(self) -> int:
        return len(self._log)

    @property
    def digest(self) -> str:
        """The hash of the episode's last log record: it commits to every event so far."""
        return self._log.digest

    def diff(
        self,
        before: State | Mapping[str, Any] | None = None,
        after: State | Mapping[str, Any] | None = None,
        *,
        ignore: Iterable[str] = BOOKKEEPING,
    ) -> dict[str, tuple[Any, Any]]:
        """What changed between two states (default: from the episode's start to now)."""
        b = before if before is not None else self.initial
        a = after if after is not None else self.state
        return diff(
            b.to_dict() if isinstance(b, State) else b, a.to_dict() if isinstance(a, State) else a, ignore=ignore
        )

    # -- episodes ------------------------------------------------------------------------------------------------------

    def reset(self, pack: EnvPack | None = None, *, seed: int | None = None) -> None:
        """A new episode from a pack (default: the same one). The catalog is the pack's, exactly."""
        with self._lock:
            if pack is not None:
                self._pack, self._config = pack, pack.config()
            self._log = self._log.new_episode(self._pack.id)
            self._commands = []
            self._open_episode(seed)


def replay(pack: EnvPack, commands: Iterable[Any], *, seed: int | None = None, episode: int = 0) -> Sandbox:
    """Run recorded commands on a fresh sandbox. Same pack, seed, episode and commands: same state hash, same
    digest."""
    sb = Sandbox(pack, seed=seed, episode=episode)
    for c in commands:
        if isinstance(c, Mapping) and c.get("type") == REJECTED:
            sb.reject(c.get("command"), error_from_dict(c["error"]))
        else:
            sb.execute(c)
    return sb


def commands_from_events(records: Iterable[EventRecord]) -> list[Any]:
    """One episode's commands, recovered from its log: each `command` record's body, and a stand-in for each command
    that was refused before it parsed."""
    opened = {int(r["seq"]) for r in records if r["type"] == "episode.started"}
    by_seq: dict[int, list[EventRecord]] = {}
    for r in records:
        if int(r["seq"]) not in opened:  # the opening and its at_start faults: the pack replays them, not a command
            by_seq.setdefault(int(r["seq"]), []).append(r)
    out: list[Any] = []
    for seq in sorted(by_seq):
        recs = by_seq[seq]
        head = next((r for r in recs if r["type"] == "command"), None)
        if head is not None:
            out.append(head["data"]["body"])
        else:
            last = recs[-1]["data"]
            out.append({"type": REJECTED, "command": last.get("command"), "error": last["error"]})
    return out


def audit_episode(pack: EnvPack, records: Sequence[EventRecord]) -> tuple[bool, str]:
    """Check one episode's log end to end: the chain verifies, and replaying its commands over `pack` reaches the
    same digest. (True, digest), or (False, what failed)."""
    ok, head = verify(records)
    if not ok:
        return False, head
    again = replay_events(pack, records)
    if again.digest != head:
        return False, f"replay reached {again.digest[:12]}, the log ends at {head[:12]}"
    return True, head


def replay_events(pack: EnvPack, records: Sequence[EventRecord]) -> Sandbox:
    """Replay one episode from its log alone. Refuses a log that was not written over this pack."""
    start = next((r for r in records if r["type"] == "episode.started"), None)
    if start is None or start["data"].get("env") != pack.id:
        found = None if start is None else start["data"].get("env")
        raise ValidationError(
            f"the log was not written over {pack.id} (it names {found!r})",
            hint="replay an episode with the envpack its episode.started record names",
        )
    return replay(pack, commands_from_events(records), seed=int(start["data"]["seed"]), episode=int(start["episode"]))

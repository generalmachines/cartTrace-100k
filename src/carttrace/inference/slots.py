"""The host-wide GPU cap: at most `max_inflight` calls in flight per model server, across every process on this host.

Why file locks. The cap protects shared GPUs (CLAUDE.md, "Shared GPUs"; docs/DECISIONS.md D6), and collection runs
as several processes at once: the runner, a judge pass, a load test. An in-process semaphore cannot see the other
processes, and a lock server would be one more thing to keep alive. So a slot is a file,
var/locks/gpu/<endpoint_id>/slot-NNN.lock, one per unit of the endpoint's cap, and a call holds a slot by holding an
exclusive flock on it. The kernel drops a flock when its process dies, so a crashed worker never leaks a slot, and
counting held slots is a non-blocking probe of each file.

How a call waits. Inside one process an asyncio semaphore lets at most `max_inflight` waiters at the files, so a
process with 500 queued decisions polls with at most `max_inflight` of them. Those try every slot without blocking
(LOCK_EX | LOCK_NB) and back off exponentially, 2 ms doubling to 25 ms, while all are taken. A slot is released when
the call ends, by return, exception or cancellation. Slots belong to a server (`endpoint_id`), not a role, so two
roles on one server share its cap. The locks are only host-wide on a local filesystem: var/ must not be a network
mount.

A cap that moves (D114). The cap is read on every try, from `caps` (carttrace.inference.caps.LiveCaps in production: the
endpoint's time-of-day schedule and the config files as they are now), so a running process follows it:
- raising it makes slots NNN up to the new cap and lets more waiters at them;
- lowering it lets the calls in flight finish and starts none above the new cap. A try only takes a slot below the
  cap, and a call that took one then counts every held slot of the server, from every process; while more than the
  cap are held (calls above it still finishing) it gives its slot back and waits. Each call locks before it counts,
  so of two calls racing for the last place at least one sees the other: the count is never exceeded, at worst both
  step back and try again.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
import re
import weakref
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

from carttrace.config.loader import settings as process_settings
from carttrace.config.schema import ModelEndpoint, Settings
from carttrace.inference.caps import LiveCaps
from carttrace.inference.service import GpuSlots

POLL_INITIAL_S = 0.002
POLL_MAX_S = 0.025
GATE_RECHECK_S = 1.0
"""How often a call waiting at the in-process gate reads the cap again, so a raise opens it before any call ends."""
SLOT_GLOB = "slot-*.lock"
_SLOT_INDEX = re.compile(r"^slot-(\d+)\.lock$")


def slot_name(index: int) -> str:
    return f"slot-{index:03d}.lock"


def lock_root(settings: Settings) -> Path:
    """Where the slot directories live: <paths.var>/locks/gpu."""
    return settings.paths.var / "locks" / "gpu"


class FileGpuSlots:
    """GpuSlots on flock'd files. Share one instance per process (`gpu_slots()`), so its waiters share one gate.

    `caps` gives an endpoint's cap right now; without it the cap is the endpoint's max_inflight."""

    def __init__(
        self,
        root: Path,
        *,
        caps: Callable[[ModelEndpoint], int] | None = None,
        poll_initial_s: float = POLL_INITIAL_S,
        poll_max_s: float = POLL_MAX_S,
        gate_recheck_s: float = GATE_RECHECK_S,
    ) -> None:
        self.root = root
        self.caps = caps
        self.gate_recheck_s = gate_recheck_s
        self.poll_initial_s = poll_initial_s
        self.poll_max_s = poll_max_s
        self._gates: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, _Gate]]
        self._gates = weakref.WeakKeyDictionary()
        self._made: dict[str, int] = {}
        self._turn = 0

    def __repr__(self) -> str:
        return f"FileGpuSlots({str(self.root)!r})"

    def cap(self, endpoint: ModelEndpoint) -> int:
        """The endpoint's cap right now."""
        return max(1, self.caps(endpoint)) if self.caps is not None else endpoint.max_inflight

    def slot(self, endpoint: ModelEndpoint) -> AbstractAsyncContextManager[None]:
        return self._hold(endpoint)

    def in_flight(self) -> dict[str, int]:
        """Slots held right now, per endpoint id, by every process on this host."""
        if not self.root.is_dir():
            return {}
        return {
            directory.name: sum(1 for path in sorted(directory.glob(SLOT_GLOB)) if _is_held(path))
            for directory in sorted(self.root.iterdir())
            if directory.is_dir()
        }

    @asynccontextmanager
    async def _hold(self, endpoint: ModelEndpoint) -> AsyncIterator[None]:
        gate = self._gate(endpoint.endpoint_id)
        gate.resize(self.cap(endpoint))
        await gate.acquire(lambda: gate.resize(self.cap(endpoint)), self.gate_recheck_s)
        try:
            fd = await self._acquire(endpoint, gate)
        except BaseException:
            gate.release()
            raise
        try:
            yield
        finally:
            try:
                _release(fd)
            finally:
                gate.release()

    def _gate(self, endpoint_id: str) -> _Gate:
        loop = asyncio.get_running_loop()
        gates = self._gates.get(loop)
        if gates is None:
            gates = self._gates[loop] = {}
        gate = gates.get(endpoint_id)
        if gate is None:
            gate = gates[endpoint_id] = _Gate(1)
        return gate

    async def _acquire(self, endpoint: ModelEndpoint, gate: _Gate) -> int:
        directory = self.root / endpoint.endpoint_id
        self._turn += 1
        turn = os.getpid() + self._turn
        delay = self.poll_initial_s
        while True:
            cap = self.cap(endpoint)
            gate.resize(cap)
            self._make(directory, endpoint.endpoint_id, cap)
            fd = _try_slots(directory, cap, turn % cap)
            if fd is not None:
                if _held(directory, cap) == 0 or _held(directory, 0) <= cap:
                    return fd
                _release(fd)  # calls above a lowered cap are still finishing: step back until they have
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.poll_max_s)

    def _make(self, directory: Path, endpoint_id: str, cap: int) -> None:
        if self._made.get(endpoint_id, 0) >= cap:
            return
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(cap):  # every slot exists up front, so the directory shows the cap at a glance
            (directory / slot_name(index)).touch(exist_ok=True)
        self._made[endpoint_id] = cap


class _Gate:
    """An asyncio semaphore whose size can change: at most `limit` holders, waiters served in order. Shrinking it
    takes nothing from a holder; it only stops new ones until the holders are fewer than the new limit."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, limit)
        self.active = 0
        self._waiters: deque[asyncio.Future[None]] = deque()

    def resize(self, limit: int) -> None:
        self.limit = max(1, limit)
        self._wake()

    async def acquire(self, recheck: Callable[[], None], every_s: float) -> None:
        """Take a place, calling `recheck` (which may resize the gate) every `every_s` while waiting."""
        if self.active < self.limit and not self._waiters:
            self.active += 1
            return
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            while not waiter.done():
                await asyncio.wait([waiter], timeout=every_s)
                if not waiter.done():
                    recheck()
            await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():  # granted just as it was cancelled: give the place back
                self.release()
            else:
                waiter.cancel()
                with contextlib.suppress(ValueError):
                    self._waiters.remove(waiter)
            raise

    def release(self) -> None:
        self.active -= 1
        self._wake()

    def _wake(self) -> None:
        while self._waiters and self.active < self.limit:
            waiter = self._waiters.popleft()
            if not waiter.done():
                self.active += 1
                waiter.set_result(None)


def _held(directory: Path, from_index: int) -> int:
    """Held slots of the server numbered `from_index` and up, by every process (a slot above the cap is held only by
    a call that started before the cap was lowered)."""
    count = 0
    for path in directory.glob(SLOT_GLOB):
        match = _SLOT_INDEX.match(path.name)
        if match is not None and int(match.group(1)) >= from_index and _is_held(path):
            count += 1
    return count


def _try_slots(directory: Path, cap: int, start: int) -> int | None:
    """The fd of the first free slot, locked, starting at `start`; None when every slot is held."""
    for offset in range(cap):
        fd = os.open(directory / slot_name((start + offset) % cap), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        except BaseException:
            os.close(fd)
            raise
        return fd
    return None


def _release(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _is_held(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


_SHARED: dict[Path, FileGpuSlots] = {}


def gpu_slots(settings: Settings | None = None) -> GpuSlots:
    """The process's GPU slots under <paths.var>/locks/gpu, one shared instance per lock directory, following the caps
    live (carttrace.inference.caps)."""
    settings = settings or process_settings()
    root = lock_root(settings)
    slots = _SHARED.get(root)
    if slots is None:
        slots = _SHARED[root] = FileGpuSlots(root, caps=LiveCaps.from_disk(settings).cap)
    return slots

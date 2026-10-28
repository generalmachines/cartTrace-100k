"""The host-wide cap on open browsers: at most `sandbox.browser.max_browsers` Chromium processes across every process
on this host.

Why a host-wide cap. Collection runs many episodes at once and in several processes (the runner, a bench scan, a CLI
journey), and each WebMCP episode opens its own Chromium. Without a cap a busy run opens hundreds of renderers and
starves the machine. As with the GPU slots (`carttrace.inference.slots`, which are keyed to model servers and so are not
reused here), a slot is a file, `<paths.var>/locks/browser/slot-NNN.lock`, held by an exclusive flock that the kernel
drops when its process dies, so a crashed episode never leaks a slot. Inside one process an asyncio semaphore keeps
waiters off the files. A waiter that gets no slot within `slot_wait_s` ends its episode as an infrastructure error.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from carttrace.sandbox.browser.errors import NoBrowserSlot

POLL_INITIAL_S = 0.01
POLL_MAX_S = 0.25


def slot_name(index: int) -> str:
    return f"slot-{index:03d}.lock"


class BrowserSlots:
    """Slots under one directory. Share one instance per directory (`browser_slots`), so its waiters share a gate."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._gates: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[int, asyncio.Semaphore]]
        self._gates = weakref.WeakKeyDictionary()

    @asynccontextmanager
    async def hold(self, cap: int, *, wait_s: float) -> AsyncIterator[int]:
        """Hold one slot while the block runs; yields the slot's index. Raises `NoBrowserSlot` after `wait_s`."""
        loop = asyncio.get_running_loop()
        gates = self._gates.setdefault(loop, {})
        gate = gates.setdefault(cap, asyncio.Semaphore(cap))
        try:
            await asyncio.wait_for(gate.acquire(), timeout=wait_s)
        except TimeoutError as exc:
            raise NoBrowserSlot(f"no browser slot of {cap} came free in {wait_s:g} s") from exc
        try:
            index, fd = await self._acquire(cap, deadline=loop.time() + wait_s)
            try:
                yield index
            finally:
                _release(fd)
        finally:
            gate.release()

    def held(self) -> int:
        """Slots held right now by every process on this host."""
        if not self.root.is_dir():
            return 0
        return sum(1 for path in sorted(self.root.glob("slot-*.lock")) if _is_held(path))

    async def _acquire(self, cap: int, *, deadline: float) -> tuple[int, int]:
        self.root.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()
        delay = POLL_INITIAL_S
        while True:
            for index in range(cap):
                fd = _try(self.root / slot_name(index))
                if fd is not None:
                    return index, fd
            if loop.time() >= deadline:
                raise NoBrowserSlot(f"all {cap} browser slots under {self.root} stayed held")
            await asyncio.sleep(delay)
            delay = min(delay * 2, POLL_MAX_S)


def _try(path: Path) -> int | None:
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise
    return fd


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


_SHARED: dict[Path, BrowserSlots] = {}


def browser_slots(root: Path) -> BrowserSlots:
    """The shared slots of one lock directory."""
    key = root.resolve()
    slots = _SHARED.get(key)
    if slots is None:
        slots = _SHARED[key] = BrowserSlots(key)
    return slots

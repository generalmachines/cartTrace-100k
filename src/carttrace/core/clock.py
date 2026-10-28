"""Clocks are injected, never read from the module.

Why. A timestamp read inside domain code makes a test depend on when it runs and makes a replay differ from the
original. Code that needs time takes a `Clock`. Production passes `SystemClock()`; tests and replays pass a
`FrozenClock` and move it explicitly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """The current time, timezone-aware, in UTC."""
        ...

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin that never goes backwards. Use for durations only."""
        ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


@dataclass(slots=True)
class FrozenClock:
    """A clock that moves only when told to. `now` and `monotonic` advance together."""

    current: datetime = field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    _mono: float = 0.0

    def __post_init__(self) -> None:
        if self.current.tzinfo is None:
            raise ValueError("FrozenClock needs a timezone-aware datetime")

    def now(self) -> datetime:
        return self.current

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a clock never moves backwards")
        self.current += timedelta(seconds=seconds)
        self._mono += seconds


def iso(dt: datetime) -> str:
    """RFC 3339 in UTC with a Z suffix and millisecond precision: the one timestamp format in files we write."""
    if dt.tzinfo is None:
        raise ValueError("refusing to format a naive datetime")
    return dt.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")

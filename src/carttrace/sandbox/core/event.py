"""One thing that happened, as the core reports it. The runtime (`carttrace.sandbox.events`) numbers it, stamps it with the
logical clock and appends it to the hash-chained log; the core only says what happened and whether it is sealed.

Sealed events are the environment's own doing (a fault firing) or bookkeeping the agent must not see. They are in the
log for the grader and never in what an adapter shows the agent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Event:
    type: str
    data: Mapping[str, Any] = field(default_factory=dict)
    """JSON values only: the log hashes them as canonical JSON."""
    sealed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": dict(self.data), "sealed": self.sealed}

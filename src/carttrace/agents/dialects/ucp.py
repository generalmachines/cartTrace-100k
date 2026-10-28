"""UCP over MCP or REST: the base dialect under its own name. 2026-04-08 and 2026-08-25 take the same shapes, so one
class serves both releases; the checkout readers it uses are the ones the harness always had (`carttrace.agents.checkout`)."""

from __future__ import annotations

from dataclasses import dataclass

from carttrace.agents.dialects.base import Dialect


@dataclass(frozen=True, slots=True)
class UcpDialect(Dialect):
    """How the reference agent speaks UCP over MCP (2026-04-08 and 2026-08-25 take the same shapes)."""

    @property
    def name(self) -> str:
        return "ucp"

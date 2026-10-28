"""Typed errors that say what happened and what to do next.

Every error the system raises on purpose derives from `MCError` and carries a `hint`: the next action for whoever reads
it (set a variable, lift a pause, re-vendor a protocol). Callers catch the narrowest type they can handle; nothing
catches `Exception` to carry on silently.
"""

from __future__ import annotations


class MCError(Exception):
    """Base for every deliberate error. `hint` is the next action, in one sentence."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message} ({self.hint})" if self.hint else self.message


class ConfigError(MCError):
    """A setting is missing or malformed."""


class ValidationError(MCError):
    """Input does not match its schema or contract."""


class NotFoundError(MCError):
    """A named thing (a protocol version, an episode, a store) does not exist."""


class SafetyError(MCError):
    """An action was refused because it could change something on a live store or break a safety rule."""


class ProtocolError(MCError):
    """A peer spoke a protocol wrongly, or we cannot speak the version it asked for."""


class ExternalServiceError(MCError):
    """A service we depend on (a model server, S3, Slack, a store) failed or is unreachable."""

    def __init__(
        self, service: str, message: str, *, hint: str = "", status: int | None = None, unavailable: bool = False
    ) -> None:
        super().__init__(f"{service}: {message}", hint=hint)
        self.service = service
        self.status = status
        self.unavailable = unavailable
        """The service was down rather than refusing us: no connection, no reply in time, a 429 or a 5xx. A caller
        may wait and try the same work again; a refusal (4xx, a reply it cannot read) would fail again."""

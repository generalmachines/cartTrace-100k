"""The refusals the core tells apart, each with a stable code an adapter maps onto its protocol's error vocabulary.

Why typed. The lab's old sandbox let a malformed argument (`quantity: "two"`, `catalog: "shoes"`) reach `int()` or
`.get()` and answered HTTP 500 with a traceback, 70 of them in the logs by 2026-10-09. A grader cannot tell "the agent
sent garbage" from "the sandbox broke" when both are a 500. Here every refusal is data: a kind, a code, a message, the
field it is about, whether retrying the same request can succeed, and a hint naming the next action.

Each kind derives from `CoreError`, which derives from `carttrace.core.errors.MCError`. The two kinds that mean what a core
error type already means also derive from it: `InvalidCommand` is a `carttrace.core.errors.ValidationError` and `NotFound` is
a `carttrace.core.errors.NotFoundError`, so a caller outside the sandbox can catch them by the shared type.

`to_dict` is what the event log and the idempotency store keep, and `from_dict` rebuilds the same error from it, so a
replayed refusal is byte-identical to the original.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from carttrace.core.errors import MCError, NotFoundError, ValidationError


class CoreError(MCError):
    """A refusal the core decided on purpose. Raised inside a handler, or returned when the side effects must stay."""

    kind: ClassVar[str] = "error"

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str | None = None,
        retryable: bool = False,
        details: Sequence[Mapping[str, Any]] = (),
        hint: str = "",
    ) -> None:
        super().__init__(message, hint=hint)
        self.code = code
        self.field = field
        self.retryable = retryable
        self.details: tuple[dict[str, Any], ...] = tuple(dict(d) for d in details)

    def to_dict(self) -> dict[str, Any]:
        """The error as JSON. Optional parts appear only when set, so the log stays small and stable."""
        out: dict[str, Any] = {
            "kind": self.kind,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.field is not None:
            out["field"] = self.field
        if self.details:
            out["details"] = [dict(d) for d in self.details]
        if self.hint:
            out["hint"] = self.hint
        return out


class InvalidCommand(CoreError, ValidationError):
    """The command itself is malformed: a missing field, a wrong type, a negative quantity, a card number."""

    kind = "validation"


class NotFound(CoreError, NotFoundError):
    """A cart, checkout, order, variant or session the command names does not exist."""

    kind = "not_found"


class IllegalTransition(CoreError):
    """The entity exists but the requested move is not in its transition table."""

    kind = "illegal_transition"


class PolicyDenied(CoreError):
    """The session's trust tier or permissions do not allow this command, or the actor may not send it."""

    kind = "policy_denied"


class Conflict(CoreError):
    """An idempotency key reused with a different body, or an injected 409."""

    kind = "conflict"


class Unavailable(CoreError):
    """An injected 5xx or 429. Retryable by definition."""

    kind = "unavailable"


class BusinessRule(CoreError):
    """The command was well formed and legal but the store refused it: out of stock, declined, price changed."""

    kind = "business"


class InvariantViolation(CoreError):
    """A commit would have broken an invariant. Never the agent's fault; the core refused to commit its own bug."""

    kind = "invariant"


KINDS: Mapping[str, type[CoreError]] = {
    c.kind: c
    for c in (
        CoreError,
        InvalidCommand,
        NotFound,
        IllegalTransition,
        PolicyDenied,
        Conflict,
        Unavailable,
        BusinessRule,
        InvariantViolation,
    )
}


def from_dict(d: Mapping[str, Any]) -> CoreError:
    """Rebuild an error from `to_dict` output. An unknown kind becomes a plain `CoreError` rather than failing."""
    cls = KINDS.get(str(d.get("kind")), CoreError)
    field = d.get("field")
    return cls(
        code=str(d["code"]),
        message=str(d.get("message") or ""),
        field=None if field is None else str(field),
        retryable=bool(d.get("retryable")),
        details=tuple(d.get("details") or ()),
        hint=str(d.get("hint") or ""),
    )

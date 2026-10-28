"""Reading what a caller sent: a body no larger than the limit, and members that should be an object or a string, or
an empty default when they are not.

Why a module. Requests reach the surfaces as parsed JSON, and a member that should be an object can be anything; each
reader would otherwise repeat the same `isinstance` test, and a type checker cannot narrow a `.get()` call. These
helpers say once what a missing or mistyped member reads as, and how much of a body both bindings read.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from starlette.requests import Request

MAX_BODY_BYTES = 4 * 1024 * 1024
"""The largest request body either binding reads, as the official MCP SDK's default; larger ones are answered 413."""


async def body_of(request: Request) -> bytes | None:
    """The request body, or None when it is larger than `MAX_BODY_BYTES` (checked before reading when the client
    declared a length)."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        return None
    raw = await request.body()
    return raw if len(raw) <= MAX_BODY_BYTES else None


def obj(value: object) -> Mapping[str, Any]:
    """`value` when it is a JSON object, else an empty one."""
    return value if isinstance(value, Mapping) else {}


def text(value: object, default: str = "") -> str:
    """`value` when it is a string, else `default`."""
    return value if isinstance(value, str) else default

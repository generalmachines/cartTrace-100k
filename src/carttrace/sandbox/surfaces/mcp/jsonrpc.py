"""JSON-RPC 2.0 framing for the MCP endpoint: telling requests, notifications and responses apart, and building
results and errors with the codes the official MCP SDK defines (`mcp_types`).

MCP narrows JSON-RPC in one way that matters here: a request id is a string or an integer and is never null, so a
message with `"id": null` is an invalid request rather than a notification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from mcp_types import (
    HEADER_MISMATCH,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    UNSUPPORTED_PROTOCOL_VERSION,
)

__all__ = [
    "HEADER_MISMATCH",
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "UNSUPPORTED_PROTOCOL_VERSION",
    "Message",
    "RequestId",
    "classify",
    "failure",
    "result",
]

RequestId = str | int
Kind = Literal["request", "notification", "response", "invalid"]


@dataclass(frozen=True, slots=True)
class Message:
    kind: Kind
    id: RequestId | None = None
    method: str | None = None
    params: Mapping[str, Any] | None = None


def classify(obj: object) -> Message:
    """One JSON-RPC message, by shape."""
    if not isinstance(obj, Mapping) or obj.get("jsonrpc") != "2.0":
        return Message("invalid", _id(obj))
    method = obj.get("method")
    params = obj.get("params")
    if params is not None and not isinstance(params, Mapping):
        return Message("invalid", _id(obj))
    if isinstance(method, str):
        if "id" not in obj:
            return Message("notification", None, method, params)
        rid = _id(obj)
        return Message("request", rid, method, params) if rid is not None else Message("invalid")
    if "id" in obj and ("result" in obj or "error" in obj):
        return Message("response", _id(obj))
    return Message("invalid", _id(obj))


def _id(obj: object) -> RequestId | None:
    rid = obj.get("id") if isinstance(obj, Mapping) else None
    if isinstance(rid, bool):
        return None
    return rid if isinstance(rid, str | int) else None


def result(rid: RequestId | None, value: Mapping[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "result": dict(value)}


def failure(
    rid: RequestId | None, code: int, message: str, data: Any = None, *, omit_unknown_id: bool = False
) -> dict[str, Any]:
    """A JSON-RPC error. An unknown id is `null` (JSON-RPC 2.0), or left out where the MCP version allows it."""
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    out: dict[str, Any] = {"jsonrpc": "2.0"}
    if rid is not None or not omit_unknown_id:
        out["id"] = rid
    out["error"] = err
    return out

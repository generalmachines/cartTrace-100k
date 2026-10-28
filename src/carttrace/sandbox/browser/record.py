"""What the sandbox browser writes to the trace: one wire record per exchange, in the shape every surface writes.

Why the same shape. The judge reads a WebMCP bundle the way it reads a UCP one, from `wire.jsonl` (`carttrace.trace.WireRecord`
fields without `seq` and `t_ms`, which the writer assigns), and redaction happens once, at write time, in the
recorder (`TraceWriter.wire`). The browser records:
- `executeTool` and `getTools` (protocol `webmcp`, at the page's draft): the tool, its arguments, and the JSON the
  page answered, or the DOMException name it rejected with;
- every request the page made to the sandbox (protocol `http`): documents by their status and the sha256 of their
  bytes, API calls with their JSON bodies (a page fragment of HTML by its sha256, since the JSON beside it holds the
  same facts);
- the browser's own events (protocol `browser`): the launch and its environment, the WebMCP runtime chosen, console
  errors and warnings, uncaught page errors, refused requests, popups, dialogs, a 3-D Secure answer, the close.
Every record carries the draft's version, since all of them belong to one WebMCP surface. Durations come from the
injected clock, so a frozen clock records the same bytes on every run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Protocol

from carttrace.core.clock import Clock

SURFACE = "webmcp"


class WireSink(Protocol):
    """Where records go. `carttrace.trace.TraceWriter` is one; so is `carttrace.sandbox.surfaces.client.WireSink`."""

    def wire(self, record: Mapping[str, Any]) -> Any: ...


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Wire:
    """Writes the browser's records to a sink (or nowhere), stamped with the surface and the draft's version."""

    def __init__(self, sink: WireSink | None, *, version: str, clock: Clock) -> None:
        self.sink = sink
        self.version = version
        self.clock = clock

    def write(
        self,
        protocol: str,
        method: str,
        url: str,
        *,
        request_headers: Mapping[str, str] | None = None,
        request_body: Any = None,
        status: int | None = None,
        response_headers: Mapping[str, str] | None = None,
        response_body: Any = None,
        duration_ms: int = 0,
        error: str | None = None,
    ) -> None:
        if self.sink is None:
            return
        self.sink.wire(
            {
                "surface": SURFACE,
                "protocol": protocol,
                "version": self.version,
                "method": method,
                "url": url,
                "request_headers": dict(request_headers or {}),
                "request_body": request_body,
                "status": status,
                "response_headers": dict(response_headers or {}),
                "response_body": response_body,
                "duration_ms": duration_ms,
                "error": error,
            }
        )

    def browser(self, method: str, url: str, *, body: Any = None, error: str | None = None) -> None:
        """One of the browser's own events."""
        self.write("browser", method, url, response_body=body, error=error)

    def elapsed_ms(self, start: float) -> int:
        return max(0, round((self.clock.monotonic() - start) * 1000))

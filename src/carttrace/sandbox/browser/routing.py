"""Offline routing: the browser reaches the sandbox store and nothing else.

Why route every request. A replica page is built from a real store's data, and real data links out: product images on
a CDN, fonts, analytics. A sandbox browser must never fetch any of it, both for the safety rules (nothing touches a
live store from the sandbox) and for determinism (a CDN's answer is not ours to fix). So every request the browser
makes goes through one Playwright route handler:
- a request for the sandbox origin is answered by the store's ASGI app, in process, through the same `httpx` client
  the other surfaces use (`httpx.ASGITransport`): no socket, and the app sees a loopback client and a loopback
  `Origin`, so its localhost-only routes (the 3-D Secure page) and its DNS-rebinding check behave as for any local
  page;
- every other request is aborted, and recorded;
- WebSockets are closed before they connect.
Name resolution is turned off in Chromium as well (`carttrace.config.browser.OFFLINE_ARGS`), so a request that somehow
escaped the handler still could not leave the machine.

The handler also names the sandbox session every request acts for (`MC-Sandbox-Session`), as a storefront's cookie
names its shopper, and counts documents against the episode's page budget.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx

from carttrace.sandbox.browser.record import Wire, sha256_of
from carttrace.sandbox.surfaces.access import SESSION_HEADER

FORWARDED = ("content-type", "accept", "origin")
"""Request headers passed on to the app; Chromium's own (user agent, client hints) would only add noise."""
KEPT_RESPONSE = ("content-type", "location", "cache-control")
PROBE_PATH = "/__mc/webmcp-probe"
"""The blank page the WebMCP probe runs in, answered by the router itself."""
PROBE_PAGE = (
    b'<!doctype html><html lang="en"><head><meta charset="utf-8"><link rel="icon" href="data:,">'
    b"<title>probe</title></head><body></body></html>"
)


class Relay:
    """The route handler of one browser context."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        origin: str,
        session: str,
        wire: Wire,
        max_pages: int,
        session_header: str = SESSION_HEADER,
    ) -> None:
        """`session_header` names the header the session goes in: the store's presented one (D106)."""
        self.client = client
        self.session_header = session_header
        self.origin = origin.rstrip("/")
        parts = urlsplit(self.origin)
        self._scheme, self._netloc = parts.scheme, parts.netloc
        self.session = session
        self.wire = wire
        self.max_pages = max_pages
        self.documents = 0
        self.blocked: list[str] = []
        self.over_budget: str | None = None
        """The first document refused for going over the page budget."""
        self.failures: list[str] = []
        """Requests the app failed to answer at all (it raised); the runtime ends the episode on the next call."""

    def serves(self, url: str) -> bool:
        parts = urlsplit(url)
        return parts.scheme == self._scheme and parts.netloc == self._netloc

    async def handle(self, route: Any) -> None:
        """Answer one request (a Playwright `Route`)."""
        request = route.request
        url = request.url
        if not self.serves(url):
            self.blocked.append(url)
            self.wire.browser("blocked", url, body={"resource": request.resource_type}, error="offline routing")
            await _answer(route.abort("blockedbyclient"))
            return
        if urlsplit(url).path == PROBE_PATH:
            await _answer(
                route.fulfill(status=200, body=PROBE_PAGE, headers={"content-type": "text/html; charset=utf-8"})
            )
            return
        document = request.resource_type == "document"
        if document:
            self.documents += 1
            if self.documents > self.max_pages:
                self.over_budget = self.over_budget or url
                self.wire.browser("blocked", url, body={"resource": "document"}, error="page budget exceeded")
                await _answer(route.abort("blockedbyclient"))
                return
        headers = {k: v for k, v in request.headers.items() if k in FORWARDED}
        headers[self.session_header] = self.session
        body = request.post_data_buffer
        start = self.wire.clock.monotonic()
        try:
            response = await self.client.request(request.method, url, headers=headers, content=body)
        except Exception as exc:  # the app raised: our failure, recorded and raised by the runtime
            reason = f"{type(exc).__name__}: {exc}"[:300]
            self.failures.append(f"{request.method} {url}: {reason}")
            self.wire.write("http", request.method, url, request_headers=headers, error=reason)
            await _answer(route.fulfill(status=500, body=b"sandbox error", headers={"content-type": "text/plain"}))
            return
        kept = {k: v for k, v in response.headers.items() if k in KEPT_RESPONSE}
        self.wire.write(
            "http",
            request.method,
            url,
            request_headers={k: v for k, v in headers.items() if k != "accept"},
            request_body=_body(body, headers.get("content-type", "")),
            status=response.status_code,
            response_headers=kept,
            response_body=_shown(response.content, kept.get("content-type", ""), document=document),
            duration_ms=self.wire.elapsed_ms(start),
        )
        await _answer(route.fulfill(status=response.status_code, headers=kept, body=response.content))

    async def refuse_socket(self, socket: Any) -> None:
        """Close a WebSocket before it connects (a Playwright `WebSocketRoute`)."""
        self.blocked.append(socket.url)
        self.wire.browser("blocked", socket.url, body={"resource": "websocket"}, error="offline routing")
        await socket.close(code=1008, reason="the sandbox browser is offline")


async def _answer(reply: Awaitable[None]) -> None:
    """Send a route's answer; a page closed meanwhile (the episode ended) needs none."""
    try:
        await reply
    except Exception as exc:
        if not _closed(exc):
            raise


def _closed(exc: BaseException) -> bool:
    text = str(exc)
    return "has been closed" in text or "Target closed" in text or "Route is already handled" in text


def _body(raw: bytes | None, content_type: str) -> Any:
    if not raw:
        return None
    if "json" in content_type:
        try:
            return json.loads(raw)
        except ValueError:
            pass
    return raw.decode("utf-8", "replace")[:4000]


def _shown(raw: bytes, content_type: str, *, document: bool) -> Any:
    """A response as the trace keeps it: a document by its digest, JSON as JSON (an HTML fragment inside it by its
    digest), anything else by its digest."""
    if not document and "json" in content_type:
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        if isinstance(data, Mapping):
            return {k: ({"sha256": sha256_of(str(v).encode())} if k == "html" else v) for k, v in data.items()}
        if data is not None:
            return data
    return {"sha256": sha256_of(raw), "bytes": len(raw)}

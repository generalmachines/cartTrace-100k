"""The one HTTP client factory and the one retry policy for the whole repository.

Why one factory. In the lab, eight places built their own model client (folded into one on 2026-10-09), and each
carried its own user agent, timeouts and retry rules. Stores must see one stable identity (docs/ARCHITECTURE.md,
"Identity"), and whether a request may be sent twice is a safety question, not a per-module habit. So every client
comes from `async_client` or `sync_client`: it sends our user agent from settings (live.user_agent), has explicit
timeouts, does not follow redirects unless asked, and never retries on its own.

Retrying is explicit, through `RetryPolicy` and `send_with_retry`, with deterministic exponential backoff (no
jitter, so a test can assert the exact delays):
- a failure to connect is retried for any request, because the request never left;
- any other transport error, and the policy's retry statuses, are retried only for idempotent requests (GET, HEAD,
  OPTIONS, PUT, DELETE, TRACE), or when the caller vouches that a POST has no side effects (`idempotent=True`, as model
  inference does);
- a Retry-After header in seconds is honoured, and a Retry-After longer than the policy's longest delay ends the
  retries, so a 429 is recorded by the caller instead of waited out.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field

import httpx

from carttrace.config.schema import Settings

IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE", "TRACE"})
DEFAULT_TIMEOUT_S = 30.0
CONNECT_TIMEOUT_S = 10.0

RetryHook = Callable[[int, float, str], None]
"""Called before each retry with (attempt that failed, delay in seconds, reason), so callers can record retries."""


def user_agent(settings: Settings | None = None) -> str:
    """Our user agent from settings (live.user_agent)."""
    if settings is None:
        from carttrace.config.loader import settings as process_settings

        settings = process_settings()
    return settings.live.user_agent


def make_timeout(total_s: float | None = None, *, connect_s: float = CONNECT_TIMEOUT_S) -> httpx.Timeout:
    """`total_s` for reading, writing and waiting for a pooled connection; connecting gets at most `connect_s`."""
    total = DEFAULT_TIMEOUT_S if total_s is None else total_s
    return httpx.Timeout(total, connect=min(connect_s, total))


def _headers(settings: Settings | None, agent: str | None, headers: Mapping[str, str] | None) -> dict[str, str]:
    """The caller's headers plus our user agent. Only the explicit `user_agent` argument can change the agent; a
    User-Agent among `headers` is dropped, so no module can quietly give stores a second identity."""
    merged = {k: v for k, v in (headers or {}).items() if k.lower() != "user-agent"}
    merged["User-Agent"] = agent or user_agent(settings)
    return merged


def async_client(
    *,
    settings: Settings | None = None,
    user_agent: str | None = None,
    base_url: str = "",
    headers: Mapping[str, str] | None = None,
    timeout: float | httpx.Timeout | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    limits: httpx.Limits | None = None,
    follow_redirects: bool = False,
    proxy: str | None = None,
) -> httpx.AsyncClient:
    """An httpx.AsyncClient with our user agent and explicit timeouts. Pass `transport` in tests
    (httpx.MockTransport) so nothing leaves the process."""
    return httpx.AsyncClient(
        base_url=base_url,
        headers=_headers(settings, user_agent, headers),
        timeout=timeout if isinstance(timeout, httpx.Timeout) else make_timeout(timeout),
        transport=transport,
        limits=limits or httpx.Limits(max_connections=100, max_keepalive_connections=20),
        follow_redirects=follow_redirects,
        proxy=proxy,
    )


def sync_client(
    *,
    settings: Settings | None = None,
    user_agent: str | None = None,
    base_url: str = "",
    headers: Mapping[str, str] | None = None,
    timeout: float | httpx.Timeout | None = None,
    transport: httpx.BaseTransport | None = None,
    limits: httpx.Limits | None = None,
    follow_redirects: bool = False,
    proxy: str | None = None,
) -> httpx.Client:
    """The synchronous twin of `async_client`, for CLI probes and other code with no event loop."""
    return httpx.Client(
        base_url=base_url,
        headers=_headers(settings, user_agent, headers),
        timeout=timeout if isinstance(timeout, httpx.Timeout) else make_timeout(timeout),
        transport=transport,
        limits=limits or httpx.Limits(max_connections=100, max_keepalive_connections=20),
        follow_redirects=follow_redirects,
        proxy=proxy,
    )


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times to try, how long to wait between tries, and which failures are worth another try."""

    attempts: int = 3
    """Total tries, the first included. 1 means never retry."""
    base_delay_s: float = 0.5
    multiplier: float = 2.0
    max_delay_s: float = 8.0
    retry_statuses: frozenset[int] = field(default_factory=lambda: frozenset({429, 502, 503, 504}))
    retry_timeouts: bool = True
    """Retry read, write and pool timeouts of idempotent requests. Off for model calls: a server that timed out is
    usually overloaded, and a retry doubles its load."""

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("RetryPolicy.attempts counts the first try, so it is at least 1")
        if self.base_delay_s < 0 or self.max_delay_s < 0 or self.multiplier < 1:
            raise ValueError("RetryPolicy delays are non-negative and the multiplier is at least 1")

    @classmethod
    def never(cls) -> RetryPolicy:
        return cls(attempts=1)

    def backoff_s(self, retry: int) -> float:
        """The wait before retry number `retry` (1 for the first retry): base * multiplier^(retry-1), capped."""
        if retry < 1:
            raise ValueError("retries are numbered from 1")
        return min(self.max_delay_s, self.base_delay_s * self.multiplier ** (retry - 1))

    def next_delay(self, failed_attempt: int, *, retry_after_s: float | None = None) -> float | None:
        """Seconds to wait before trying again after `failed_attempt` failed, or None when the tries are used up or
        the server asked for a longer wait than this policy allows."""
        if failed_attempt >= self.attempts:
            return None
        delay = self.backoff_s(failed_attempt)
        if retry_after_s is not None:
            if retry_after_s > self.max_delay_s:
                return None
            delay = max(delay, retry_after_s)
        return delay

    def retries_status(self, status: int) -> bool:
        return status in self.retry_statuses

    def retries_error(self, exc: httpx.TransportError, *, idempotent: bool) -> bool:
        if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout | httpx.PoolTimeout):
            return True  # the request never reached the server
        if isinstance(exc, httpx.UnsupportedProtocol | httpx.LocalProtocolError | httpx.ProxyError):
            return False  # our request is wrong; another try fails the same way
        if isinstance(exc, httpx.TimeoutException):
            return idempotent and self.retry_timeouts
        return idempotent


def retry_after_s(response: httpx.Response) -> float | None:
    """Retry-After in seconds. The HTTP-date form is not read (it needs a clock); the policy's own backoff applies."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _is_idempotent(request: httpx.Request, idempotent: bool | None) -> bool:
    return request.method.upper() in IDEMPOTENT_METHODS if idempotent is None else idempotent


async def send_with_retry(
    client: httpx.AsyncClient,
    request: httpx.Request,
    policy: RetryPolicy | None = None,
    *,
    idempotent: bool | None = None,
    hold: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: RetryHook | None = None,
) -> httpx.Response:
    """Send `request`, retrying under `policy`. `hold` wraps each try (carttrace.inference passes its GPU slot, so a slot is
    never held through a backoff sleep). Returns the last response, retryable status or not; raises the last
    transport error when the tries run out."""
    policy = policy or RetryPolicy()
    safe = _is_idempotent(request, idempotent)
    attempt = 0
    while True:
        attempt += 1
        try:
            async with hold() if hold is not None else nullcontext():
                response = await client.send(request)
        except httpx.TransportError as exc:
            delay = policy.next_delay(attempt) if policy.retries_error(exc, idempotent=safe) else None
            if delay is None:
                raise
            reason = type(exc).__name__
        else:
            if not (safe and policy.retries_status(response.status_code)):
                return response
            delay = policy.next_delay(attempt, retry_after_s=retry_after_s(response))
            if delay is None:
                return response
            reason = f"HTTP {response.status_code}"
            await response.aclose()
        if on_retry is not None:
            on_retry(attempt, delay, reason)
        await sleep(delay)


def send_with_retry_sync(
    client: httpx.Client,
    request: httpx.Request,
    policy: RetryPolicy | None = None,
    *,
    idempotent: bool | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: RetryHook | None = None,
) -> httpx.Response:
    """The synchronous twin of `send_with_retry`."""
    policy = policy or RetryPolicy()
    safe = _is_idempotent(request, idempotent)
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.send(request)
        except httpx.TransportError as exc:
            delay = policy.next_delay(attempt) if policy.retries_error(exc, idempotent=safe) else None
            if delay is None:
                raise
            reason = type(exc).__name__
        else:
            if not (safe and policy.retries_status(response.status_code)):
                return response
            delay = policy.next_delay(attempt, retry_after_s=retry_after_s(response))
            if delay is None:
                return response
            reason = f"HTTP {response.status_code}"
            response.close()
        if on_retry is not None:
            on_retry(attempt, delay, reason)
        sleep(delay)

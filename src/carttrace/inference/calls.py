"""The one way inference reaches a model server: a GPU slot, the inference retry policy, timing, stats and errors.

Why one path. Chat and decision-model calls differ only in the request body and in how the reply is read. Everything
else must be the same for both, or the cap and the status file stop meaning what they say:
- every request holds a GPU slot while it is on the wire, and never during a backoff sleep;
- it is timed by the injected clock from the moment the slot is held;
- it lands in the per-process stats with its tokens and the model that answered;
- it fails as ExternalServiceError naming the role, the endpoint and what to do next.

Retries: a connection that never opened, a dropped connection, and 502 or 503 are retried up to three tries with
backoff (0.5 s, then 1 s). Timeouts are not retried: a server that timed out is usually overloaded, and a retry
doubles its load.

Refusals are back-pressure, not failures (D114). The GPU gateway refuses a call above its parallel limit instead of
queueing it. A refusal is HTTP 429, or any other error status but 401, 403 and 404 whose body speaks of the limit
(concurrency, parallel, too many, rate limit, capacity, busy, refused). It is retried with backoff (1 s doubling to
30 s, jittered, or the server's Retry-After when longer, up to 60 s), without holding a GPU slot while it waits,
for up to `REFUSAL_PATIENCE_S` in all. Only then does it fail, as `unavailable`, which the collector treats as an
outage and runs the row again (carttrace.collect.scheduler), so a refusal never ends a row as a model error. The wait counts
as GPU queueing, not as the agent's time. Each refusal is counted in the stats (`refusals`), never as an error.

HTTP clients come from carttrace.config.http, one per event loop, so a process that runs several event loops in turn never
reuses a client bound to a closed loop.
"""

from __future__ import annotations

import asyncio
import random
import re
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from carttrace.config.http import RetryPolicy, async_client, make_timeout, retry_after_s, send_with_retry
from carttrace.config.schema import ModelEndpoint, Settings
from carttrace.config.secrets import secret
from carttrace.core.canon import loads
from carttrace.core.clock import Clock, SystemClock
from carttrace.core.errors import ConfigError, ExternalServiceError, ValidationError
from carttrace.inference.service import GpuSlots
from carttrace.inference.slots import gpu_slots
from carttrace.inference.stats import InferenceStats, process_stats

INFERENCE_RETRY = RetryPolicy(
    attempts=3,
    base_delay_s=0.5,
    multiplier=2.0,
    max_delay_s=8.0,
    retry_statuses=frozenset({502, 503}),
    retry_timeouts=False,
)
REFUSAL_PATIENCE_S = 900.0
"""How long one call keeps retrying refusals before it fails as unavailable."""
REFUSAL_BASE_S = 1.0
REFUSAL_MAX_S = 30.0
REFUSAL_RETRY_AFTER_MAX_S = 60.0
_REFUSAL_TEXT = re.compile(
    r"concurren|parallel|too many|rate.?limit|capacity|limit (?:exceeded|reached)|over (?:the )?(?:limit|cap)|"
    r"in.?flight|busy|refus",
    re.IGNORECASE,
)
_NEVER_REFUSALS = frozenset({401, 403, 404})


def is_refusal(status: int, body: str) -> bool:
    """Whether an error reply is the server refusing load (back-pressure) rather than refusing the request."""
    if status == 429:
        return True
    if status < 400 or status in _NEVER_REFUSALS:
        return False
    return _REFUSAL_TEXT.search(body) is not None


def refusal_delay(refusals: int, retry_after: float | None, *, jitter: float = 1.0) -> float:
    """The wait after the `refusals`-th refusal in a row: backoff, scaled by `jitter` (0.5 to 1), or Retry-After
    when the server asks for longer, up to REFUSAL_RETRY_AFTER_MAX_S."""
    backoff = min(REFUSAL_MAX_S, REFUSAL_BASE_S * 2 ** (refusals - 1)) * jitter
    if retry_after is not None:
        backoff = max(backoff, min(retry_after, REFUSAL_RETRY_AFTER_MAX_S))
    return float(backoff)


CONNECTION_LIMITS = httpx.Limits(max_connections=256, max_keepalive_connections=64)


@dataclass(frozen=True, slots=True)
class Reply:
    payload: dict[str, Any]
    latency_ms: int
    status: int


def describe(endpoint: ModelEndpoint) -> str:
    """How errors name a call: the endpoint and the role, e.g. "qwen for role 'agent'"."""
    return f"{endpoint.name or endpoint.endpoint_id} for role {endpoint.role!r}"


def usage_of(payload: Mapping[str, Any]) -> dict[str, int]:
    """The integer usage counters, plus `cached_prompt_tokens` when the server reports a prompt cache split."""
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return {}
    out = {k: v for k, v in usage.items() if isinstance(v, int) and not isinstance(v, bool)}
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, Mapping) else None
    if isinstance(cached, int) and not isinstance(cached, bool):
        out["cached_prompt_tokens"] = cached
    return out


def served_model_of(payload: Mapping[str, Any]) -> str | None:
    model = payload.get("model")
    return model if isinstance(model, str) and model else None


class ModelClient:
    """Posts JSON to model servers. `transport` (httpx.MockTransport) keeps tests offline; `sleep` keeps them fast."""

    def __init__(
        self,
        settings: Settings,
        *,
        slots: GpuSlots | None = None,
        stats: InferenceStats | None = None,
        clock: Clock | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        refusal_patience_s: float = REFUSAL_PATIENCE_S,
        rng: random.Random | None = None,
    ) -> None:
        self.settings = settings
        self.refusal_patience_s = refusal_patience_s
        self._rng = rng or random.Random()  # noqa: S311 - backoff jitter, not a secret
        self.slots = slots or gpu_slots(settings)
        self.stats = stats or process_stats()
        self.clock = clock or SystemClock()
        self.retry = retry or INFERENCE_RETRY
        self._transport = transport
        self._sleep = sleep or asyncio.sleep
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]
        self._clients = weakref.WeakKeyDictionary()
        self._keys: dict[str, str] = {}

    async def post[T](
        self, endpoint: ModelEndpoint, path: str, body: Mapping[str, Any], read: Callable[[Reply], T]
    ) -> T:
        """POST `body` to the endpoint and turn the reply into a result with `read`. A reply `read` cannot make sense
        of (it raises) counts as an endpoint error in the stats."""
        client = self._client()
        try:
            request = client.build_request(
                "POST",
                endpoint.base_url + path,
                json=dict(body),
                headers=self._auth(endpoint),
                timeout=make_timeout(endpoint.timeout_s),
            )
        except (TypeError, ValueError) as exc:  # httpx refuses NaN, Infinity and values JSON cannot hold
            raise ValidationError(
                f"the request to {describe(endpoint)} is not plain JSON: {exc}",
                hint="send messages and tools as str, finite numbers, lists and dicts",
            ) from exc
        held_at: list[float] = []

        @asynccontextmanager
        async def hold() -> AsyncIterator[None]:
            async with self.slots.slot(endpoint):
                held_at.append(self.clock.monotonic())
                yield

        try:
            response = await self._send(endpoint, client, request, hold)
        except httpx.TimeoutException as exc:
            raise self._failure(
                endpoint,
                held_at,
                f"no reply from {path} within {endpoint.timeout_s:g} s",
                hint=f"the server may be overloaded; check `carttrace config check --probe --endpoint {endpoint.name}` "
                "and the GPU in-flight count",
                unavailable=True,
            ) from exc
        except httpx.TransportError as exc:
            raise self._failure(
                endpoint,
                held_at,
                f"cannot reach {endpoint.base_url}: {type(exc).__name__}",
                hint=f"check the server is up: `carttrace config check --probe --endpoint {endpoint.name}`",
                unavailable=True,
            ) from exc
        except httpx.HTTPError as exc:  # the reply arrived but could not be read, e.g. a corrupt compressed body
            raise self._failure(
                endpoint,
                held_at,
                f"the reply from {path} could not be read: {type(exc).__name__}",
                hint="retry; if it repeats, check the server or any proxy in between",
            ) from exc
        latency_ms = self._elapsed_ms(held_at)
        if response.status_code >= 400 and is_refusal(response.status_code, response.text):
            raise self._failure(
                endpoint,
                held_at,
                f"refused for {self.refusal_patience_s:g} s: HTTP {response.status_code} from {path}: "
                f"{_snippet(response)}",
                hint="the GPU gateway is over its parallel limit; check endpoints."
                f"{endpoint.name}.max_inflight and schedule against infra's limits",
                status=response.status_code,
                unavailable=True,
            )
        if response.status_code >= 400:
            raise self._failure(
                endpoint,
                held_at,
                f"HTTP {response.status_code} from {path}: {_snippet(response)}",
                hint=_status_hint(endpoint, response.status_code),
                status=response.status_code,
                unavailable=response.status_code == 429 or response.status_code >= 500,
            )
        payload = self._payload(endpoint, held_at, path, response)
        reply = Reply(payload=payload, latency_ms=latency_ms, status=response.status_code)
        try:
            result = read(reply)
        except Exception as exc:
            self.stats.record(endpoint, ok=False, latency_ms=latency_ms, error=str(exc))
            raise
        self.stats.record(
            endpoint, ok=True, latency_ms=latency_ms, usage=usage_of(payload), served_model=served_model_of(payload)
        )
        return result

    async def _send(
        self,
        endpoint: ModelEndpoint,
        client: httpx.AsyncClient,
        request: httpx.Request,
        hold: Callable[[], AbstractAsyncContextManager[None]],
    ) -> httpx.Response:
        """Send under the retry policy, and wait out refusals (back-pressure) for up to refusal_patience_s. Returns
        the last response, a refusal included when the patience ran out."""
        waited = 0.0
        refusals = 0
        while True:
            response = await send_with_retry(client, request, self.retry, idempotent=True, hold=hold, sleep=self._sleep)
            if response.status_code < 400 or not is_refusal(response.status_code, response.text):
                return response
            refusals += 1
            self.stats.refused(endpoint)
            delay = refusal_delay(refusals, retry_after_s(response), jitter=0.5 + self._rng.random() / 2)
            if waited + delay > self.refusal_patience_s:
                return response
            await response.aclose()
            await self._sleep(delay)
            waited += delay

    async def aclose(self) -> None:
        """Close this event loop's HTTP client. Safe to call more than once."""
        client = self._clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            await client.aclose()

    def _client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None or client.is_closed:
            client = async_client(settings=self.settings, transport=self._transport, limits=CONNECTION_LIMITS)
            self._clients[loop] = client
        return client

    def _auth(self, endpoint: ModelEndpoint) -> dict[str, str]:
        variable = endpoint.api_key_env
        if not variable:
            return {}
        key = self._keys.get(variable)
        if key is None:
            found = secret(variable, settings=self.settings)
            if not found:
                raise ConfigError(
                    f"{describe(endpoint)} needs the secret {variable}, which is not set",
                    hint=f"export {variable}, or add it to a file listed in ops.secrets_files",
                )
            key = self._keys[variable] = found
        return {"Authorization": f"Bearer {key}"}

    def _elapsed_ms(self, held_at: list[float]) -> int:
        return round((self.clock.monotonic() - held_at[-1]) * 1000) if held_at else 0

    def _failure(
        self,
        endpoint: ModelEndpoint,
        held_at: list[float],
        message: str,
        *,
        hint: str,
        status: int | None = None,
        unavailable: bool = False,
    ) -> ExternalServiceError:
        error = ExternalServiceError(describe(endpoint), message, hint=hint, status=status, unavailable=unavailable)
        self.stats.record(endpoint, ok=False, latency_ms=self._elapsed_ms(held_at), error=str(error))
        return error

    def _payload(
        self, endpoint: ModelEndpoint, held_at: list[float], path: str, response: httpx.Response
    ) -> dict[str, Any]:
        try:
            payload = loads(response.content)
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            raise self._failure(
                endpoint,
                held_at,
                f"{path} replied with something other than a JSON object",
                hint=f"check endpoints.{endpoint.name}.base_url points at the right server",
            )
        return payload


def _snippet(response: httpx.Response) -> str:
    text = " ".join(response.text.split())
    return text[:200] or "(empty body)"


def _status_hint(endpoint: ModelEndpoint, status: int) -> str:
    if status in (401, 403):
        variable = endpoint.api_key_env or "the endpoint's api_key_env"
        return f"the server refused the key; check {variable}"
    if status == 404:
        return f"check endpoints.{endpoint.name}.base_url and model"
    if status < 500:
        return f"the server refused the request; check endpoints.{endpoint.name}.model and the request size"
    return "the server failed; retry later, and check its logs if it repeats"

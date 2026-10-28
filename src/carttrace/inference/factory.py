"""`chat_service()`: the chat service callers ask for, chosen by config.json components["inference.chat"].

With no settings given it returns one shared instance per process, so every caller shares the HTTP connections and
the concurrency gate; with explicit settings it builds a new one the caller owns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from carttrace.config.loader import settings as process_settings
from carttrace.config.schema import Settings
from carttrace.core.errors import ConfigError
from carttrace.core.registry import resolve
from carttrace.inference.service import ChatResult, ChatService

CHAT_KIND = "inference.chat"

_SHARED: dict[str, tuple[Settings, object]] = {}


def chat_service(settings: Settings | None = None) -> ChatService:
    if settings is not None:
        return _build(settings)
    current = process_settings()
    shared = _SHARED.get(CHAT_KIND)
    if shared is not None and shared[0] is current:
        return cast(ChatService, shared[1])
    service = _build(current)
    _SHARED[CHAT_KIND] = (current, service)
    return service


def _build(settings: Settings) -> ChatService:
    name = settings.component(CHAT_KIND)
    service: object = resolve(CHAT_KIND, name)(settings)
    if not callable(getattr(service, "chat", None)):
        raise ConfigError(f"{CHAT_KIND} implementation {name!r} has no chat() method", hint="fix its PLUGINS entry")
    return cast(ChatService, service)


async def chat(
    role: str,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None = None,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    seed: int | None = None,
) -> ChatResult:
    """One turn for `role` on the process's shared ChatService: `await chat("agent", messages, tools)`."""
    return await chat_service().chat(role, messages, tools, max_tokens=max_tokens, temperature=temperature, seed=seed)

"""ChatService over any OpenAI-compatible /v1/chat/completions server: vLLM, our Qwen gateway on model-server, and
Anthropic's OpenAI-compatible API.

Why a pure parser. A recorded turn must read exactly like a live one, and the agent loop must never see a tool call
whose arguments silently changed shape. `parse_chat_response` takes the reply payload and returns a ChatResult:
- text is the message content (a list of text parts is joined);
- reasoning is `reasoning_content` or `reasoning`; when a server leaves `<think>...</think>` in the content instead,
  it is moved to reasoning, so traces separate the two the same way whichever server answered;
- each tool call's arguments are parsed as a JSON object. Arguments that do not parse, parse to something other than
  an object, or hold NaN give {} with `arguments_error` set and the raw text kept. They never raise: a malformed
  call is the agent's mistake to record, not a crash;
- the served model is the reply's `model` field, which can differ from the requested one (model-server :8080 answers
  every alias with qwen3.8-27b);
- usage keeps the integer counters, plus `cached_prompt_tokens` when the server reports a cache split;
- a reply with no text, no reasoning and no tool call raises ExternalServiceError naming the role and the endpoint.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx

from carttrace.config.http import RetryPolicy
from carttrace.config.schema import ModelEndpoint, Settings
from carttrace.core.canon import loads
from carttrace.core.clock import Clock
from carttrace.core.errors import ConfigError, ExternalServiceError
from carttrace.core.ids import stable_id
from carttrace.inference.calls import ModelClient, describe, served_model_of, usage_of
from carttrace.inference.service import ChatResult, GpuSlots, ToolCall
from carttrace.inference.stats import InferenceStats

CHAT_PATH = "/v1/chat/completions"
THINK_START = "<think>"
THINK_END = "</think>"


def build_request(
    endpoint: ModelEndpoint,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None,
    *,
    max_tokens: int,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    """The request body: the endpoint's extra_body, then the fields this call sets, which always win."""
    body: dict[str, Any] = dict(endpoint.extra_body)
    body.update(model=endpoint.model, messages=list(messages), max_tokens=max_tokens, temperature=temperature)
    if tools:
        body["tools"] = list(tools)
        body.setdefault("tool_choice", "auto")
    if seed is not None:
        body["seed"] = seed
    return body


def parse_chat_response(endpoint: ModelEndpoint, payload: dict[str, Any], latency_ms: int) -> ChatResult:
    service = describe(endpoint)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ExternalServiceError(service, "the reply carried no choices", hint=_EMPTY_HINT)
    choice = choices[0]
    raw_message = choice.get("message")
    message = raw_message if isinstance(raw_message, dict) else {}
    text = _content_text(service, message.get("content"))
    reasoning = _first_text(message.get("reasoning_content"), message.get("reasoning"))
    if reasoning is None and THINK_END in text:
        before, _, after = text.rpartition(THINK_END)
        reasoning = before.replace(THINK_START, "").strip() or None
        text = after.strip()
    elif reasoning is None and text.lstrip().startswith(THINK_START):  # cut off while still thinking
        reasoning = text.lstrip().removeprefix(THINK_START).strip() or None
        text = ""
    tool_calls = parse_tool_calls(service, message.get("tool_calls"))
    if not text.strip() and not reasoning and not tool_calls:
        raise ExternalServiceError(service, "empty reply: no text, reasoning or tool call", hint=_EMPTY_HINT)
    finish = choice.get("finish_reason")
    return ChatResult(
        role=endpoint.role,
        text=text,
        tool_calls=tool_calls,
        reasoning=reasoning,
        finish_reason=finish if isinstance(finish, str) else None,
        requested_model=endpoint.model,
        served_model=served_model_of(payload),
        usage=usage_of(payload),
        latency_ms=latency_ms,
        endpoint=endpoint.name or None,
    )


_EMPTY_HINT = "retry the turn; if it repeats, check the server's logs for this model"


def parse_tool_calls(service: str, raw: object) -> list[ToolCall]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ExternalServiceError(service, "tool_calls in the reply is not a list", hint=_EMPTY_HINT)
    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        call = item if isinstance(item, dict) else {}
        raw_function = call.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}
        raw_name = function.get("name")
        name = raw_name if isinstance(raw_name, str) else ""
        arguments, error, raw_text = parse_arguments(function.get("arguments"))
        if not name:
            error = error or "the tool call names no function"
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = stable_id("call", index, name, raw_text or "")
        calls.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
                arguments_error=error,
                raw_arguments=raw_text if error else None,
            )
        )
    return calls


def parse_arguments(raw: object) -> tuple[dict[str, Any], str | None, str | None]:
    """(arguments, error, raw text). Never raises: a call whose arguments are unusable gets {} and an error."""
    if raw is None:
        return {}, None, None
    if isinstance(raw, dict):
        return dict(raw), None, None
    if not isinstance(raw, str):
        return {}, f"arguments are a JSON {_kind(raw)}, not a string or an object", json.dumps(raw)
    value, error = _decode(raw)
    if error is not None:
        return {}, error, raw
    if not isinstance(value, dict):
        return {}, f"arguments are a JSON {_kind(value)}, not an object", raw
    return value, None, raw


def _decode(text: str) -> tuple[object, str | None]:
    """The JSON value of `text`, or an error. Blank text is an empty object: a call to a tool with no parameters."""
    if not text.strip():
        return {}, None
    try:
        return loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"arguments are not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
    except ValueError:
        return None, "arguments contain NaN or Infinity, which JSON does not allow"


def _kind(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "null" if value is None else type(value).__name__


def _content_text(service: str, content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        )
    raise ExternalServiceError(service, f"message content is a {type(content).__name__}", hint=_EMPTY_HINT)


def _first_text(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


class OpenAICompatChat:
    """ChatService for every role whose endpoint has kind "chat"."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: ModelClient | None = None,
        slots: GpuSlots | None = None,
        stats: InferenceStats | None = None,
        clock: Clock | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or ModelClient(
            settings, slots=slots, stats=stats, clock=clock, transport=transport, retry=retry, sleep=sleep
        )

    async def chat(
        self,
        role: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> ChatResult:
        endpoint = self.settings.role(role)
        if endpoint.kind != "chat":
            raise ConfigError(
                f"role {role!r} routes to {endpoint.name!r}, a {endpoint.kind} endpoint; chat needs a chat endpoint",
                hint=f"point roles.{role} at an endpoint of kind chat, or use label_scorer() for {endpoint.kind}",
            )
        body = build_request(endpoint, messages, tools, max_tokens=max_tokens, temperature=temperature, seed=seed)
        return await self.client.post(
            endpoint, CHAT_PATH, body, lambda reply: parse_chat_response(endpoint, reply.payload, reply.latency_ms)
        )

    async def aclose(self) -> None:
        await self.client.aclose()

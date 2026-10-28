"""What inference promises. Implementations: `carttrace.inference.openai_chat`, `carttrace.inference.clef`, `carttrace.inference.recorded`.

Two kinds of call, because the system asks models two kinds of thing:
- `ChatService.chat(role, messages, tools)`: a conversation turn (the agent; the shopper only if a template cannot
  answer). The caller names the role, never the model.
- `LabelScorer.score(role, query, labels)`: the probability of each label as the next token, renormalised over the
  labels, from Clef's /generative_scoring. This is what the decision engine is built on: one forward pass for two
  labels and one per label for more, no decoding, no free text.

Every call passes through the host-wide cap for its endpoint (`gpu_slot`), and every result carries the served model
next to the requested one, because one server can answer an alias with a different model.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from carttrace.config.schema import ModelEndpoint

ProbabilitySource = Literal["logprobs", "text"]


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any]
    """Parsed JSON arguments. A call whose arguments did not parse has {} here and `arguments_error` set."""
    arguments_error: str | None = None
    raw_arguments: str | None = None
    """The arguments exactly as the model emitted them, kept only when they did not parse, so the trace shows the
    mistake."""


class ChatResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning: str | None = None
    finish_reason: str | None = None
    requested_model: str
    served_model: str | None
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: int
    """Model time of the call that answered: from holding the GPU slot to the reply, waits for a slot excluded."""
    endpoint: str | None = None
    """Name of the config.json endpoint that answered (qwen, sonnet, ...)."""


class ScoreResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    probabilities: dict[str, float]
    """Label to probability, summing to 1 over the labels asked."""
    served_model: str | None
    latency_ms: int
    """Wall time of the whole score call, slot waits included: its per-label calls overlap, so their own times do not
    add up to anything useful."""
    usage: dict[str, int] = Field(default_factory=dict)
    requested_model: str | None = None
    endpoint: str | None = None
    """Name of the config.json endpoint that answered (clef_flash, clef, ...)."""
    probabilities_from: ProbabilitySource | None = None
    """Where the probabilities came from when not from Clef's /generative_scoring (None): "logprobs", a chat
    endpoint's next-token log probabilities; "text", only the answer's text, with a configured confidence."""
    cost_usd: float | None = None
    """What the call cost, when the backend reports it (the Claude CLI does); None for our own model servers."""


class ChatService(Protocol):
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
        """Raises ExternalServiceError naming the role and endpoint when the server fails or replies with nothing."""
        ...


class LabelScorer(Protocol):
    async def score(self, role: str, query: str, labels: Sequence[str]) -> ScoreResult:
        """Each label must tokenise to exactly one token on the role's model; raises ValidationError otherwise."""
        ...


class GpuSlots(Protocol):
    def slot(self, endpoint: ModelEndpoint) -> AbstractAsyncContextManager[None]:
        """Hold one of the endpoint's `max_inflight` slots, shared by every process on this host, for one call."""
        ...

    def in_flight(self) -> dict[str, int]:
        """Slots held right now, per endpoint id, across all processes."""
        ...


async def _never() -> AsyncIterator[None]:  # pragma: no cover - typing helper only
    yield None

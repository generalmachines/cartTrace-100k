"""The agent's context budget (D82): older tool results become short stubs in the model's view, never in the trace.

Why it exists. Every model call resends the whole conversation, so a sandbox order row sent a median of 470k input
tokens in 15 calls on 2026-10-16, and the shared GPUs' prefill is what limits throughput. Successful rows peaked at a
median of 36k tokens of context (p95 83k, p99 121k), but the rows that wandered (time_cap, step_cap) kept growing to
140k to 540k, almost all of it catalog search results the agent had read long before
(docs/research/2026-10-16-agent-context.md). `agents.context_budget_tokens` bounds what one call may send.

Opt-in only. The budget is 0 (off) by default: a compacted history is one a real agent would not see, so the project
keeps it out of the dataset (2026-10-16), a release leaves out every episode run with it (rule context_budget), and
benchmark rows never set it. It exists for experiments that measure what trimming costs.

How it works, deterministically. Before a call the harness estimates its prompt: the previous call's measured
prompt_tokens plus what was added since at CHARS_PER_TOKEN, which errs high. Over the budget, the oldest tool results
are replaced in the model's view by a stub naming the tool, its arguments, the result's size and a digest of the ids
it held (product and variant ids with their titles, the checkout's status), until the estimate is at COMPACT_TO of
the budget. A stub stays a stub, so the prompt's prefix up to the newest stub is stable between compactions and a
server's prefix cache can keep reusing it. Never stubbed: the system prompt, every shopper and assistant message, the
assistant's tool calls (so the tool-call pairing is untouched), the newest KEEP_RESULTS results, and results too
small to be worth it. The trace bundle keeps every result in full: the stubs exist only in the request.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from carttrace.core.canon import dumps, loads

CHARS_PER_TOKEN = 3.0
"""Characters per token assumed for text added since the last measured call. Qwen's tokenizer read 3.43 characters a
token at the median successful row's peak call on 2026-10-16 and 3.09 at the tenth percentile, so 3.0 errs high."""
COMPACT_TO = 0.6
"""A compaction stubs results until the estimate is this share of the budget, so the next one is several calls away
(each changes the prompt from its first new stub on, which a prefix cache must then recompute)."""
KEEP_RESULTS = 3
"""The newest tool results are never stubbed: the agent is acting on them."""
MIN_STUB_CHARS = 2_000
"""A result shorter than this in the model's view is never stubbed: a stub saves too little to be worth the loss."""
DIGEST_CHARS = 1_200
"""Most characters a stub's digest of ids may take."""
ARGUMENT_CHARS = 200
"""Most characters of a stub's restatement of the call's arguments."""
_LABELS = ("title", "name", "label", "handle", "sku")
_SCALARS = 8
"""Most top-level scalar fields (status, totals, ids) a digest names."""


@dataclass(frozen=True, slots=True)
class ResultSource:
    """What one tool message in the history came from: the call and the result as the surface returned it."""

    tool: str
    arguments: Mapping[str, Any]
    content: Any
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Compaction:
    """One compaction: the message indexes newly stubbed, and the estimated prompt tokens before and after."""

    stubbed: tuple[int, ...]
    before: int
    after: int


def message_chars(message: Mapping[str, Any]) -> int:
    """A message's characters as the request carries them: its content and its tool calls."""
    return len(str(message.get("content") or "")) + sum(len(str(call)) for call in message.get("tool_calls") or ())


def history_chars(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(message_chars(message) for message in messages)


def tools_chars(tools: Sequence[Mapping[str, Any]]) -> int:
    """The tool definitions' characters, which every call carries as well."""
    return len(dumps(list(tools)))


class ContextGauge:
    """Estimates the next call's prompt tokens from the last measured one. Before any measurement it counts every
    character at CHARS_PER_TOKEN; after one, only the difference from what that call sent."""

    def __init__(self) -> None:
        self.sent_chars: int | None = None
        self.sent_tokens: int | None = None

    def measured(self, chars: int, prompt_tokens: int | None) -> None:
        """Record what a call sent (its messages' characters) and the prompt tokens the server counted for it; a
        server that reports no count leaves the estimate on characters alone."""
        if prompt_tokens is None or prompt_tokens <= 0:
            return
        self.sent_chars, self.sent_tokens = chars, prompt_tokens

    def estimate(self, chars: int, fixed_chars: int) -> int:
        """The prompt tokens of a call whose messages take `chars` characters beside `fixed_chars` of tool
        definitions."""
        if self.sent_chars is None or self.sent_tokens is None:
            return round((chars + fixed_chars) / CHARS_PER_TOKEN)
        return max(0, round(self.sent_tokens + (chars - self.sent_chars) / CHARS_PER_TOKEN))


def view(messages: Sequence[dict[str, Any]], stubs: Mapping[int, str]) -> list[dict[str, Any]]:
    """The history as the model is sent it: each stubbed tool message's content replaced by its stub."""
    if not stubs:
        return list(messages)
    return [{**m, "content": stubs[i]} if i in stubs else m for i, m in enumerate(messages)]


def compact(
    messages: Sequence[dict[str, Any]],
    sources: Mapping[int, ResultSource],
    stubs: dict[int, str],
    *,
    budget: int,
    estimate: Callable[[list[dict[str, Any]]], int],
) -> Compaction | None:
    """Stub the oldest tool results until `estimate` of the view is at COMPACT_TO of `budget`; None when the view is
    within the budget already. `stubs` is the episode's own map of message index to stub and gains the new ones."""
    current = view(messages, stubs)
    before = estimate(current)
    if budget <= 0 or before <= budget:
        return None
    results = [i for i, m in enumerate(messages) if m.get("role") == "tool" and i in sources]
    protected = set(results[-KEEP_RESULTS:])
    target = budget * COMPACT_TO
    after = before
    added: list[int] = []
    for i in results:
        if after <= target:
            break
        content = str(messages[i].get("content") or "")
        if i in protected or i in stubs or len(content) < MIN_STUB_CHARS:
            continue
        stubs[i] = stub(sources[i], len(content), budget)
        added.append(i)
        current[i] = {**messages[i], "content": stubs[i]}
        after = estimate(current)
    return Compaction(stubbed=tuple(added), before=before, after=after)


def stub(source: ResultSource, chars: int, budget: int) -> str:
    """What the model reads in place of a result it read before: the call, the size, and the ids it held."""
    arguments = dumps(dict(source.arguments)) if source.arguments else "{}"
    if len(arguments) > ARGUMENT_CHARS:
        arguments = arguments[:ARGUMENT_CHARS] + "..."
    kind = "an error" if source.is_error else f"{chars} characters"
    head = (
        f"[earlier result shortened by the harness to keep this conversation within its {budget}-token budget] "
        f"{source.tool}({arguments}) returned {kind}"
    )
    body = digest(source.content)
    tail = "Call the tool again if you need the full result."
    return f"{head}: {body}. {tail}" if body else f"{head}. {tail}"


def digest(content: Any) -> str:
    """The parts of a result an agent acts on later, in order: its top-level counts and short fields, then each
    object's id with its title, cut at DIGEST_CHARS with how many ids were left out."""
    data = _parsed(content)
    if isinstance(data, str):
        text = " ".join(data.split())
        return text[:300] + ("..." if len(text) > 300 else "")
    head: list[str] = []
    if isinstance(data, Mapping):
        for key in sorted(data):
            value = data[key]
            if isinstance(value, list):
                head.append(f"{len(value)} {key}")
            elif isinstance(value, bool | int | float | str) and len(str(value)) <= 60 and len(head) < _SCALARS:
                head.append(f"{key} {value}")
    elif isinstance(data, list):
        head.append(f"{len(data)} items")
    ids: list[str] = []
    seen: set[str] = set()
    for label, ident in _ids(data):
        if ident not in seen:
            seen.add(ident)
            ids.append(f"{label} {ident}" if label else ident)
    parts = ", ".join(head)
    room = DIGEST_CHARS - len(parts)
    shown: list[str] = []
    used = 0
    for entry in ids:
        if used + len(entry) + 2 > room:
            break
        shown.append(entry)
        used += len(entry) + 2
    listed = "; ".join(shown)
    if len(shown) < len(ids):
        listed += f"; and {len(ids) - len(shown)} more ids"
    return "; ids: ".join(p for p in (parts, listed) if p)


def _parsed(content: Any) -> Any:
    if isinstance(content, str):
        stripped = content.lstrip()
        if stripped.startswith(("{", "[")):
            try:
                return loads(content)
            except ValueError:
                return content
    return content


def _ids(data: Any) -> Iterator[tuple[str, str]]:
    """Every (title, id) pair in the result, depth first in the result's own order."""
    if isinstance(data, Mapping):
        ident = data.get("id")
        if isinstance(ident, str | int) and not isinstance(ident, bool):
            label = next((str(data[k]) for k in _LABELS if isinstance(data.get(k), str) and data[k]), "")
            yield label[:80], str(ident)
        for value in data.values():
            if isinstance(value, Mapping | list):
                yield from _ids(value)
    elif isinstance(data, list):
        for item in data:
            yield from _ids(item)

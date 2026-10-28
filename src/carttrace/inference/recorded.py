"""Cassettes: model calls recorded once and replayed by the canonical digest of the request.

Why. Judging and reports must re-run without touching a GPU, and tests must never reach one (CLAUDE.md). A cassette
is a JSONL file with one canonical line per call: {"key", "kind", "request", "result"}. The key is the digest of the
canonical request (the role, messages, tools and sampling parameters of a chat; the role, query and labels of a
score), so a replay answers only the exact call that was recorded, and a changed prompt is a loud miss
(NotFoundError), never a stale answer. The result is the ChatResult or ScoreResult as returned, so a replay carries
the recorded served model, usage and latency, byte for byte.

Modes: "replay" (the default) answers from the file and calls nothing; "record" calls the wrapped service and appends
the call. Recording a call whose identical result is already there writes nothing, so re-running a recording script
does not grow the file; a changed result is appended and the later line wins. One process writes a cassette at a
time. With components["inference.chat"] or ["inference.scorer"] set to "recorded", the factories replay
<paths.var>/cassettes/chat.jsonl and score.jsonl.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pydantic

from carttrace.config.schema import Settings
from carttrace.core.canon import digest
from carttrace.core.errors import ConfigError, NotFoundError, ValidationError
from carttrace.core.files import append_jsonl, read_jsonl
from carttrace.inference.service import ChatResult, ChatService, LabelScorer, ScoreResult

CassetteMode = Literal["replay", "record"]
KEY_VERSION = 1
KEY_LENGTH = 32


def chat_request(
    role: str,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None,
    *,
    max_tokens: int,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    """What a chat call is keyed on: everything that can change its answer."""
    return {
        "v": KEY_VERSION,
        "kind": "chat",
        "role": role,
        "messages": list(messages),
        "tools": list(tools or []),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }


def score_request(role: str, query: str, labels: Sequence[str]) -> dict[str, Any]:
    return {"v": KEY_VERSION, "kind": "score", "role": role, "query": query, "labels": list(labels)}


def request_key(request: dict[str, Any]) -> str:
    try:
        return digest(request, KEY_LENGTH)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"a {request.get('kind')} request is not plain JSON, so it cannot be recorded: {exc}",
            hint="pass messages, tools and labels as JSON values (str, numbers, lists, dicts)",
        ) from exc


class Cassette:
    """One cassette file, read once and appended to as calls are recorded."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._rows: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(path):
            if not (isinstance(row, dict) and isinstance(row.get("key"), str) and "result" in row):
                raise ValidationError(f"{path} holds a line that is not a cassette record", hint="re-record it")
            self._rows[row["key"]] = row

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, key: str) -> dict[str, Any] | None:
        return self._rows.get(key)

    def put(self, row: dict[str, Any]) -> bool:
        """Append `row` unless the same record is already the latest for its key. True when a line was written."""
        key = row["key"]
        if self._rows.get(key) == row:
            return False
        append_jsonl(self.path, row)
        self._rows[key] = row
        return True


def _cassette(cassette: Cassette | Path) -> Cassette:
    return cassette if isinstance(cassette, Cassette) else Cassette(cassette)


def _check_mode(mode: str, inner: object | None) -> None:
    if mode not in ("replay", "record"):
        raise ConfigError(f"unknown cassette mode {mode!r}", hint="use replay or record")
    if mode == "record" and inner is None:
        raise ConfigError("recording needs the service to record", hint="pass inner=chat_service() or label_scorer()")


def _miss(cassette: Cassette, kind: str, role: str, key: str) -> NotFoundError:
    return NotFoundError(
        f"no recorded {kind} call {key} for role {role!r} in {cassette.path}",
        hint="record it with mode='record'; the prompt, tools, labels or parameters may have changed",
    )


class RecordedChat:
    """ChatService that replays a cassette, or records the calls of `inner` into it."""

    def __init__(
        self, cassette: Cassette | Path, *, mode: CassetteMode = "replay", inner: ChatService | None = None
    ) -> None:
        _check_mode(mode, inner)
        self.cassette = _cassette(cassette)
        self.mode = mode
        self.inner = inner

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
        request = chat_request(role, messages, tools, max_tokens=max_tokens, temperature=temperature, seed=seed)
        key = request_key(request)
        if self.inner is None or self.mode == "replay":
            row = self.cassette.get(key)
            if row is None or row.get("kind") != "chat":
                raise _miss(self.cassette, "chat", role, key)
            return _validated(ChatResult, row, self.cassette)
        result = await self.inner.chat(role, messages, tools, max_tokens=max_tokens, temperature=temperature, seed=seed)
        self.cassette.put({"key": key, "kind": "chat", "request": request, "result": result.model_dump(mode="json")})
        return result


class RecordedScorer:
    """LabelScorer that replays a cassette, or records the calls of `inner` into it."""

    def __init__(
        self, cassette: Cassette | Path, *, mode: CassetteMode = "replay", inner: LabelScorer | None = None
    ) -> None:
        _check_mode(mode, inner)
        self.cassette = _cassette(cassette)
        self.mode = mode
        self.inner = inner

    async def score(self, role: str, query: str, labels: Sequence[str]) -> ScoreResult:
        request = score_request(role, query, labels)
        key = request_key(request)
        if self.inner is None or self.mode == "replay":
            row = self.cassette.get(key)
            if row is None or row.get("kind") != "score":
                raise _miss(self.cassette, "score", role, key)
            return _validated(ScoreResult, row, self.cassette)
        result = await self.inner.score(role, query, labels)
        self.cassette.put({"key": key, "kind": "score", "request": request, "result": result.model_dump(mode="json")})
        return result


def _validated[R: (ChatResult, ScoreResult)](model: type[R], row: dict[str, Any], cassette: Cassette) -> R:
    try:
        return model.model_validate(row["result"])
    except pydantic.ValidationError:
        raise ValidationError(
            f"{cassette.path} holds a malformed {row.get('kind')} record {row.get('key')}", hint="re-record it"
        ) from None


def cassette_path(settings: Settings, kind: Literal["chat", "score"]) -> Path:
    return settings.paths.var / "cassettes" / f"{kind}.jsonl"


def recorded_chat(settings: Settings) -> RecordedChat:
    """The "recorded" inference.chat component: replays <paths.var>/cassettes/chat.jsonl."""
    return RecordedChat(cassette_path(settings, "chat"))


def recorded_scorer(settings: Settings) -> RecordedScorer:
    """The "recorded" inference.scorer component: replays <paths.var>/cassettes/score.jsonl."""
    return RecordedScorer(cassette_path(settings, "score"))

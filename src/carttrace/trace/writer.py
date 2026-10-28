"""The trace writer: stages one episode's bundle on local disk while it runs, then uploads it whole, manifest last.

Why stage, then upload. An episode writes hundreds of small records over minutes, and each one must survive a killed
runner. A blob store has no append, so records are appended and flushed to local files instead, and the bundle goes
to the store only when the episode ends: file by file, with bundle.json last. A reader that finds bundle.json knows
every other file is already there; one that does not knows the bundle is still arriving. The staging copy is removed
only after the upload, so a crash at any point leaves a finished bundle that `recover_staged` uploads, or an
unfinished one it reports, and never a lost one. A lock on each staging directory keeps recovery away from a writer
that is still alive, and the kernel drops it when the writer dies.

Why the writer owns `seq` and `t_ms`. One counter per episode, shared by turns, wire, state and decisions, merges the
four logs back into the single order things happened and makes a snapshot's `raw_ref` unambiguous; within one file
seq only increases, with gaps where another file took a number. Times are milliseconds since the writer opened, from
the injected clock, so a frozen clock writes the same bytes every time.

Why every record is redacted, not only wire records. A cart id with its secret key, or a shopper's email, reaches
tool arguments and results, state snapshots and the outcome as easily as the wire. One table and one set of stable
placeholders apply everywhere, so a redacted cart id in a turn still matches the one in the wire log. Turns, wire
records and snapshots each list the rules that changed them in `redactions`; `Outcome.counts["redacted"]` counts the
records any rule changed.

The sandbox's event log (D33). A sandbox episode hands the writer every record of the sandbox's own hash-chained log
as it is chained (`event`), and each goes to `events.jsonl` at once, redacted, without taking a number from the
episode's counter: the line keeps the sandbox's own `seq`, `idx`, `prev` and `hash`, so a reader checks the chain
from the bundle alone. The file exists only once a first event arrives, so a live bundle never has one, and
bundle.json lists it only when it exists. Records may arrive from the sandbox's worker threads, so the file has its
own lock.

What the writer adds on its own (D69), so every agent and every surface gets it without saying so:
- **Call ids.** A tool_call turn gets `call_id` `call-<seq>` (its own seq, so it is unique and deterministic) unless
  the caller names one; a tool_result turn that names none cites the oldest unanswered call of the same tool, which
  is the call it answers, since an episode runs its calls one at a time; every wire record written while a call is
  unanswered (and no other turn came between) names it in `source_call_id`. `done` ends an episode and is answered
  by nothing, so it opens no window.
- **Error classes.** A failed tool result gets `error_type` from carttrace.trace.errors' closed set and `error_message`,
  its text cut to 300 characters; `error_code` stays as the surface said it.
- **Cost counts.** `cached_prompt_tokens` from usage, and `gpu_ms` and `cost_usd_micros` from each call's cost, are
  summed into the outcome's counts beside the token counts.
- **The served model.** The first agent turn that names the model that served it (`served`) fills the header's
  agent ModelRef `served`, which was null on every header before; the staged episode.json is rewritten in place,
  which is safe because nothing is uploaded before close.
- **agent.json.** `agent_definition` writes what the agent was given once per episode, and records the prompt's
  sha256 and file in the header's agent ModelRef params.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import shutil
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Final, Literal, Self

from pydantic import BaseModel

from carttrace.core.canon import dumps_line, pretty, sha256_hex
from carttrace.core.clock import Clock
from carttrace.core.errors import MCError, NotFoundError, ValidationError
from carttrace.core.files import append_jsonl, atomic_write_text
from carttrace.trace.errors import error_type_of, message_of
from carttrace.trace.experiment import ExperimentConfig
from carttrace.trace.exposure import ExposureSheet
from carttrace.trace.layout import (
    AGENT_FILE,
    BUNDLE_FILES,
    DECISIONS_FILE,
    EPISODE_FILE,
    EVENTS_FILE,
    EXPERIMENT_FILE,
    EXPOSURE_FILE,
    LOG_FILES,
    MANIFEST_FILE,
    OPTIONAL_FILES,
    OUTCOME_FILE,
    STATE_FILE,
    TURNS_FILE,
    UPLOAD_ORDER,
    WIRE_FILE,
    TraceStore,
    bundle_prefix,
    parse_json_as,
    validate_as,
)
from carttrace.trace.redact import redact_json, redact_wire
from carttrace.trace.schema import (
    AGENT_SCHEMA,
    Actor,
    AgentDefinition,
    BundleManifest,
    CallCost,
    Cart,
    EpisodeHeader,
    Outcome,
    SandboxEvent,
    StateSnapshot,
    Turn,
    TurnKind,
    TurnOrigin,
    WireRecord,
)

COUNT_KEYS = (
    "turns",
    "tool_calls",
    "tool_errors",
    "prompt_tokens",
    "completion_tokens",
    "cached_prompt_tokens",
    "gpu_ms",
    "cost_usd_micros",
    "decisions",
    "wire",
    "snapshots",
    "redacted",
)
"""What the writer counts as it records, and fills into `Outcome.counts` where the caller left a key out."""

_LOCK_FILE = ".lock"
_OWNED = ("seq", "t_ms")
_UNANSWERED = frozenset({"done"})
"""Tools whose call no result answers: the harness's `done` ends the episode, so it opens no wire window."""


WriterState = Literal["open", "closing", "sealed", "closed", "abandoned"]
"""open takes records; closing and sealed are a close() that has started (and may be retried); abandoned is a
context block that failed before an outcome, whose staged files now belong to recover_staged."""


class TraceWriter:
    """Writes one episode's bundle. Create it with `TraceWriter.open(...)`; use it from one thread or event loop.

    Each record method returns the record as written: numbered, stamped and redacted. Used as a context manager it
    closes on the way out, and when the block fails before an outcome was recorded it leaves the staged files for
    `recover_staged` to report instead of hiding the failure behind a second one.
    """

    def __init__(
        self,
        store: TraceStore,
        header: EpisodeHeader,
        *,
        prefix: str,
        path: Path,
        clock: Clock,
        salt: str,
        lock: IO[str],
    ) -> None:
        """Internal: `open` validates, stages and locks, then hands the pieces here."""
        self.header = header
        """The header as written, redacted."""
        self.prefix = prefix
        """Where the bundle goes in the store: episodes/<date>/<run_id>/<episode_id>."""
        self.path = path
        """The staging directory."""
        self._store = store
        self._clock = clock
        self._salt = salt
        self._lock: IO[str] | None = lock
        self._origin = clock.monotonic()
        self._seq = 0
        self._wire_seqs: set[int] = set()
        self._outcome: Outcome | None = None
        self._state: WriterState = "open"
        self._counts: dict[str, int] = dict.fromkeys(COUNT_KEYS, 0)
        self._events_lock = threading.Lock()
        self._events = 0
        self._pending: list[tuple[str, str | None]] = []
        """(call_id, tool) of every tool call no result has answered yet, oldest first."""
        self._open_call: str | None = None
        """The call the next wire record was made for: the last turn written was its tool_call."""
        self._agent_written = False
        self._exposure_written = False
        self._experiment_written = False

    @classmethod
    def open(
        cls,
        store: TraceStore,
        header: EpisodeHeader,
        *,
        staging_dir: Path,
        clock: Clock,
        redaction_salt: str = "",
    ) -> TraceWriter:
        """Stage a new bundle in `staging_dir/<episode_id>/` and write its episode.json.

        `redaction_salt` keys the email, phone and reviewer placeholders. Empty keeps them comparable across machines
        and runs; set it from a secret before bundles leave the team, because an unsalted hash of a phone number can
        be reversed by trying every number.
        """
        prefix = bundle_prefix(header)
        data, fired = redact_json(header.model_dump(by_alias=True), salt=redaction_salt)
        redacted = validate_as(EpisodeHeader, data, where=f"the header of episode {header.episode_id}")
        path = staging_dir / header.episode_id
        path.mkdir(parents=True, exist_ok=True)
        lock = _try_lock(path)
        if lock is None:
            raise ValidationError(
                f"episode {header.episode_id} is already being written in {path}",
                hint="one writer per episode; let the other one finish",
            )
        try:
            leftovers = _staged_files(path)
            if leftovers:
                raise ValidationError(
                    f"{path} already holds a staged bundle ({', '.join(leftovers)})",
                    hint="run recover_staged: it uploads a finished bundle and reports an unfinished one, which "
                    "must be removed (discard_unfinished=True) before the episode is re-run",
                )
            atomic_write_text(path / EPISODE_FILE, pretty(redacted.model_dump(mode="json", by_alias=True)))
            for name in LOG_FILES:
                (path / name).touch()
        except BaseException:
            lock.close()
            raise
        writer = cls(store, redacted, prefix=prefix, path=path, clock=clock, salt=redaction_salt, lock=lock)
        writer._counts["redacted"] += bool(fired)
        return writer

    @property
    def elapsed_ms(self) -> int:
        """Milliseconds since the writer opened, from the injected clock: the `t_ms` the next record gets."""
        return max(0, round((self._clock.monotonic() - self._origin) * 1000))

    def counts(self) -> dict[str, int]:
        """What the writer has recorded so far, by kind (see COUNT_KEYS)."""
        return dict(self._counts)

    def turn(
        self,
        actor: Actor,
        kind: TurnKind,
        *,
        text: str | None = None,
        tool: str | None = None,
        surface: str | None = None,
        arguments: dict[str, Any] | None = None,
        result: Any = None,
        is_error: bool = False,
        error_code: str | None = None,
        model: str | None = None,
        usage: dict[str, int] | None = None,
        latency_ms: int | None = None,
        seed: int | None = None,
        call_id: str | None = None,
        tool_call_id: str | None = None,
        finish_reason: str | None = None,
        cost: CallCost | Mapping[str, Any] | None = None,
        origin: TurnOrigin | Mapping[str, Any] | None = None,
        served: str | None = None,
    ) -> Turn:
        """Append one turn: a message, a tool call, a tool result or a note. Its `redactions` lists the rules that
        changed it; `seed` and `finish_reason` are the model call's that produced it, `cost` that call's on its first
        turn. A tool call is given a `call_id` and a result the `tool_call_id` it answers when the caller names none,
        and a failed result its `error_type` and `error_message` (D69). `served` is the model the endpoint said
        answered an agent call: it fills the header's agent ModelRef and is not itself a field of the turn."""
        self._require_open()
        if kind == "tool_call" and call_id is None:
            call_id = f"call-{self._seq}"
        if kind == "tool_result":
            tool_call_id = self._answer(tool, tool_call_id)
        error_type = error_type_of(error_code, result) if kind == "tool_result" and is_error else None
        error_message = message_of(result) if error_type is not None else None
        record = self._append(
            TURNS_FILE,
            Turn,
            {
                "actor": actor,
                "kind": kind,
                "text": text,
                "tool": tool,
                "surface": surface,
                "arguments": arguments,
                "result": result,
                "is_error": is_error,
                "error_code": error_code,
                "model": model,
                "usage": usage,
                "latency_ms": latency_ms,
                "seed": seed,
                "call_id": call_id,
                "tool_call_id": tool_call_id,
                "finish_reason": finish_reason,
                "cost": _plain(cost),
                "error_type": error_type,
                "error_message": error_message,
                "origin": _plain(origin),
            },
        )
        if kind == "tool_call" and record.call_id is not None:
            self._pending.append((record.call_id, tool))
        self._open_call = record.call_id if kind == "tool_call" and tool not in _UNANSWERED else None
        self._counts["turns"] += 1
        self._counts["tool_calls"] += record.kind == "tool_call"
        self._counts["tool_errors"] += record.kind == "tool_result" and record.is_error
        for key in ("prompt_tokens", "completion_tokens", "cached_prompt_tokens"):
            self._counts[key] += (record.usage or {}).get(key, 0)
        if record.cost is not None:
            self._counts["gpu_ms"] += record.cost.gpu_ms or 0
            self._counts["cost_usd_micros"] += record.cost.usd_micros or 0
        if served and actor == "agent":
            self._serve("agent", served)
        return record

    def exposure_sheet(self, sheet: ExposureSheet) -> ExposureSheet:
        """Write exposure.json once (D93): what the store exposed per data type and surface, redacted like every record,
        in canonical JSON so the same world, task and surfaces give the same bytes."""
        self._require_open()
        if self._exposure_written:
            raise ValidationError(
                f"episode {self.header.episode_id} already has its exposure sheet",
                hint="the episode writes its sheet once, when its world and surfaces are known",
            )
        if sheet.episode_id != self.header.episode_id:
            raise ValidationError(
                f"exposure sheet of {sheet.episode_id} handed to episode {self.header.episode_id}",
                hint="build the sheet from this episode's own header",
            )
        redacted, fired = redact_json(sheet.model_dump(mode="json", by_alias=True), salt=self._salt)
        written = validate_as(ExposureSheet, redacted, where=f"the exposure sheet of {self.header.episode_id}")
        atomic_write_text(self.path / EXPOSURE_FILE, dumps_line(written.model_dump(mode="json", by_alias=True)) + "\n")
        self._exposure_written = True
        self._counts["redacted"] += bool(fired)
        return written

    def experiment_config(self, config: ExperimentConfig) -> ExperimentConfig:
        """Write experiment.json once (D96): what kind of trace this is, for the judge, in canonical JSON so the same
        row and header give the same bytes. Redacted like every record."""
        self._require_open()
        if self._experiment_written:
            raise ValidationError(
                f"episode {self.header.episode_id} already has its experiment config",
                hint="the episode writes its experiment config once, after its header",
            )
        if config.episode_id != self.header.episode_id:
            raise ValidationError(
                f"experiment config of {config.episode_id} handed to episode {self.header.episode_id}",
                hint="build the config from this episode's own row and header",
            )
        redacted, fired = redact_json(config.model_dump(mode="json", by_alias=True), salt=self._salt)
        written = validate_as(ExperimentConfig, redacted, where=f"the experiment config of {self.header.episode_id}")
        atomic_write_text(self.path / EXPERIMENT_FILE, pretty(written.model_dump(mode="json", by_alias=True)) + "\n")
        self._experiment_written = True
        self._counts["redacted"] += bool(fired)
        return written

    def agent_definition(self, definition: AgentDefinition | Mapping[str, Any]) -> AgentDefinition:
        """Write agent.json once (D69): the system prompt and every tool definition the agent was given, redacted, and
        name the prompt's sha256 and file in the header's agent ModelRef params."""
        self._require_open()
        if self._agent_written:
            raise ValidationError(
                f"episode {self.header.episode_id} already has its agent definition",
                hint="the agent records what it was given once, when it has listed its tools",
            )
        data = (
            definition.model_dump(by_alias=True)
            if isinstance(definition, BaseModel)
            else {"schema": AGENT_SCHEMA, **definition}
        )
        redacted, fired = redact_json(data, salt=self._salt)
        written = validate_as(AgentDefinition, redacted, where=f"the agent definition of {self.header.episode_id}")
        # Compact, not pretty: tool schemas are deep, and indentation doubled the file (119 KB to 55 KB in the smoke).
        atomic_write_text(self.path / AGENT_FILE, dumps_line(written.model_dump(mode="json", by_alias=True)) + "\n")
        self._agent_written = True
        self._counts["redacted"] += bool(fired)
        extra = {
            "system_prompt_sha256": written.system_prompt_sha256,
            **({"prompt_sha256": written.prompt_sha256} if written.prompt_sha256 else {}),
            **({"prompt_file": written.prompt_file} if written.prompt_file else {}),
        }
        self._amend_models(written.role, params=extra)
        return written

    def amend_header(self, **fields: Any) -> EpisodeHeader:
        """Set header fields learned after the bundle opened (the shopper's version, say), rewriting the staged
        episode.json before anything is uploaded. Only the D69 version fields may change this way."""
        self._require_open()
        unknown = sorted(set(fields) - _AMENDABLE)
        if unknown:
            raise ValidationError(
                f"the header field(s) {', '.join(unknown)} cannot change after the bundle opened",
                hint=f"pass them to TraceWriter.open; only {', '.join(sorted(_AMENDABLE))} are learned later",
            )
        return self._rewrite_header({**self.header.model_dump(by_alias=True), **fields})

    def _serve(self, role: str, served: str) -> None:
        if any(m.role == role and m.served is None for m in self.header.models):
            self._amend_models(role, served=served)

    def _amend_models(self, role: str, *, served: str | None = None, params: Mapping[str, Any] | None = None) -> None:
        models = []
        for ref in self.header.models:
            update: dict[str, Any] = {}
            if ref.role == role and served is not None and ref.served is None:
                update["served"] = served
            if ref.role == role and params:
                update["params"] = {**ref.params, **params}
            models.append(ref.model_copy(update=update).model_dump())
        self._rewrite_header({**self.header.model_dump(by_alias=True), "models": models})

    def _rewrite_header(self, data: Mapping[str, Any]) -> EpisodeHeader:
        redacted, _ = redact_json(dict(data), salt=self._salt)
        header = validate_as(EpisodeHeader, redacted, where=f"the header of episode {self.header.episode_id}")
        if header != self.header:
            atomic_write_text(self.path / EPISODE_FILE, pretty(header.model_dump(mode="json", by_alias=True)))
            self.header = header
        return header

    def _answer(self, tool: str | None, tool_call_id: str | None) -> str | None:
        """The call a tool result answers: the one it names, else the oldest unanswered call of the same tool."""
        for index, (pending_id, pending_tool) in enumerate(self._pending):
            if (tool_call_id is not None and pending_id == tool_call_id) or (
                tool_call_id is None and pending_tool == tool
            ):
                del self._pending[index]
                return pending_id
        return tool_call_id

    def wire(self, record: Mapping[str, Any]) -> WireRecord:
        """Append one protocol exchange. Pass every WireRecord field except `seq` and `t_ms`, which the writer
        assigns. The record is redacted before it is staged, and its `redactions` lists the rules that fired."""
        self._require_open()
        _refuse_owned(record, "wire record")
        source = {"source_call_id": self._open_call} if "source_call_id" not in record and self._open_call else {}
        raw = validate_as(
            WireRecord,
            {**record, **source, "seq": self._seq, "t_ms": self.elapsed_ms},
            where=f"wire record {self._seq}",
        )
        redacted, fired = redact_wire(raw, salt=self._salt)
        self._write(WIRE_FILE, redacted.model_dump(mode="json", by_alias=True), redacted=bool(fired))
        self._wire_seqs.add(redacted.seq)
        self._counts["wire"] += 1
        return redacted

    def snapshot(
        self,
        source: str,
        cart: Cart | None = None,
        *,
        raw_ref: int | None = None,
        selection: Mapping[str, Any] | None = None,
    ) -> StateSnapshot:
        """Append one cart or state snapshot, or a website selection (D31). `raw_ref` is the seq of the wire record it
        was read from. Its `redactions` lists the rules that changed it."""
        self._require_open()
        if raw_ref is not None and raw_ref not in self._wire_seqs:
            raise ValidationError(
                f"raw_ref {raw_ref} is not the seq of a wire record in episode {self.header.episode_id}",
                hint="pass the seq of the WireRecord that wire() returned",
            )
        record = self._append(
            STATE_FILE,
            StateSnapshot,
            {
                "source": source,
                "cart": None if cart is None else cart.model_dump(),
                "raw_ref": raw_ref,
                **({"selection": dict(selection)} if selection is not None else {}),
            },
        )
        self._counts["snapshots"] += 1
        return record

    def decision(self, record: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
        """Append one decision-engine record (a carttrace.decide Decision, or its dump). The line is the redacted decision
        dict plus the writer-owned `seq` and `t_ms`; carttrace.decide owns every other field, so nothing else is added."""
        self._require_open()
        data = record.model_dump(by_alias=True) if isinstance(record, BaseModel) else dict(record)
        _refuse_owned(data, "decision record")
        redacted, fired = redact_json(data, salt=self._salt)
        line: dict[str, Any] = {**redacted, "seq": self._seq, "t_ms": self.elapsed_ms}
        self._write(DECISIONS_FILE, line, redacted=bool(fired))
        self._counts["decisions"] += 1
        return line

    def event(self, record: Mapping[str, Any]) -> SandboxEvent:
        """Append one record of the sandbox's event log to events.jsonl, as the sandbox chained it, its `data`
        redacted. A record the rules changed lists them in `redactions` and keeps the sandbox's `prev` and `hash`."""
        self._require_open()
        fields = {k: v for k, v in record.items() if k != "redactions"}
        data, fired = redact_json(fields.get("data") or {}, salt=self._salt)
        line = validate_as(
            SandboxEvent, {**fields, "data": data, "redactions": fired}, where=f"sandbox event {fields.get('idx')}"
        )
        with self._events_lock:
            self._require_open()
            target = self.path / EVENTS_FILE
            size = target.stat().st_size if target.exists() else 0
            try:
                append_jsonl(target, line.model_dump(mode="json"))
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"a sandbox event is not canonical JSON: {exc}",
                    hint="the sandbox logs JSON values only; this is a bug in carttrace.sandbox",
                ) from exc
            except OSError:
                os.truncate(target, size)
                raise
            self._events += 1
            self._counts["redacted"] += bool(fired)
        return line

    @property
    def events_written(self) -> int:
        """Lines in events.jsonl so far."""
        with self._events_lock:
            return self._events

    def outcome(self, outcome: Outcome) -> Outcome:
        """Record how the episode ended, once; close() writes it. Any COUNT_KEYS entry the caller left out of
        `counts` is filled from what the writer recorded, so every bundle carries the same counts."""
        self._require_open()
        if self._outcome is not None:
            raise ValidationError(
                f"episode {self.header.episode_id} already has an outcome",
                hint="record the outcome once, when the episode ends",
            )
        data = outcome.model_dump()
        data["counts"] = {**self.counts(), **outcome.counts}
        redacted, _ = redact_json(data, salt=self._salt)
        self._outcome = validate_as(Outcome, redacted, where=f"the outcome of episode {self.header.episode_id}")
        return self._outcome

    def close(self) -> str:
        """Finish the bundle and return its store prefix.

        Writes outcome.json, then bundle.json with the sha256 of every file, uploads every file with bundle.json
        last, and removes the staging copy only after the upload. When any step fails the writer lets go of the
        staging directory, so calling close() again retries, and `recover_staged` can finish it after a crash.
        """
        if self._state == "closed":
            return self.prefix
        if self._outcome is None:
            raise ValidationError(
                f"episode {self.header.episode_id} has no outcome yet",
                hint="call outcome(...) before close(); use end_reason='harness_error' when our own code failed",
            )
        if self._state != "open" and not self.path.is_dir():
            return self._finished_elsewhere()
        if self._state == "open":
            self._state = "closing"
        try:
            self._hold_lock()
            if self._state == "closing":
                atomic_write_text(self.path / OUTCOME_FILE, pretty(self._outcome.model_dump(mode="json")))
                _write_manifest(self.path, self.header.episode_id)
                self._state = "sealed"
            _upload(self.path, self._store, self.prefix)
            shutil.rmtree(self.path)
            self._state = "closed"
        finally:
            self._release()
        return self.prefix

    async def aclose(self) -> str:
        """close() on a worker thread, so an upload does not stall the event loop that runs the other episodes."""
        return await asyncio.to_thread(self.close)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._state == "closed":
            return
        if exc_type is None or self._outcome is not None:
            self.close()
        else:
            self._state = "abandoned"
            self._release()

    def _append[M: BaseModel](self, name: str, model: type[M], fields: Mapping[str, Any]) -> M:
        """A turn or a snapshot: redacted, then numbered, stamped and given the names of the rules that fired."""
        self._require_open()
        redacted, fired = redact_json(dict(fields), salt=self._salt)
        record = validate_as(
            model,
            {**redacted, "redactions": fired, "seq": self._seq, "t_ms": self.elapsed_ms},
            where=f"{name} record",
        )
        self._write(name, record.model_dump(mode="json", by_alias=True), redacted=bool(fired))
        return record

    def _write(self, name: str, line: dict[str, Any], *, redacted: bool) -> None:
        target = self.path / name
        size = target.stat().st_size
        try:
            append_jsonl(target, line)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"a {name} record is not canonical JSON: {exc}",
                hint="record JSON values only: no NaN or Infinity, no bytes, no objects",
            ) from exc
        except OSError:
            # A full disk can stop a write halfway. Cut the half record off, so a log that goes on being written
            # never holds a broken line before good ones (readers forgive only a torn last line).
            os.truncate(target, size)
            raise
        self._seq += 1
        self._counts["redacted"] += redacted

    @property
    def records(self) -> int:
        """Records written so far (turns, wire records, snapshots, decisions, events): the episode's progress."""
        return self._seq

    @property
    def is_open(self) -> bool:
        """Whether the writer still takes records: false once close() has started."""
        return self._state == "open"

    def _require_open(self) -> None:
        if self._state != "open":
            raise ValidationError(
                f"the writer for episode {self.header.episode_id} is {self._state} and takes no more records",
                hint="record everything before close(); a new episode, or a re-run, needs a new writer",
            )

    def _hold_lock(self) -> None:
        if self._lock is None:
            self._lock = _try_lock(self.path)
            if self._lock is None:
                raise ValidationError(
                    f"another process holds {self.path}",
                    hint="it is finishing this bundle (recover_staged, probably); let it, then check the store",
                )

    def _release(self) -> None:
        if self._lock is not None:
            self._lock.close()  # closing the descriptor drops the lock
            self._lock = None

    def _finished_elsewhere(self) -> str:
        """After a failed close, recover_staged may have uploaded the bundle and removed the staging copy."""
        if self._store.exists(f"{self.prefix}/{MANIFEST_FILE}"):
            self._state = "closed"
            return self.prefix
        raise NotFoundError(
            f"the staging copy {self.path} is gone and the store has no {self.prefix}/{MANIFEST_FILE}",
            hint="something removed the staging directory before its upload finished; the episode must be re-run",
        )


_AMENDABLE = frozenset({"shopper_version", "generator_version", "judge_code_version"})


def _plain(value: BaseModel | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)


def _refuse_owned(record: Mapping[str, Any], what: str) -> None:
    owned = [key for key in _OWNED if key in record]
    if owned:
        raise ValidationError(
            f"a {what} may not set {', '.join(owned)}",
            hint="leave them out: the writer numbers every record and stamps it from its clock",
        )


def _try_lock(directory: Path) -> IO[str] | None:
    """An exclusive lock on a staging directory, or None while another writer holds it."""
    handle = (directory / _LOCK_FILE).open("a", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    except BaseException:
        handle.close()
        raise
    return handle


def _staged_files(path: Path) -> list[str]:
    return sorted(p.name for p in path.iterdir() if not p.name.startswith("."))


V1_REQUIRED: Final = (EXPERIMENT_FILE, EXPOSURE_FILE)
"""Files a sandbox v1 bundle cannot seal without, unless it ended harness_error (D98, final review #7)."""


def _write_manifest(path: Path, episode_id: str) -> BundleManifest:
    missing = [name for name in BUNDLE_FILES if not (path / name).is_file()]
    if not missing:
        missing = _v1_missing(path)
    if missing:
        raise ValidationError(
            f"{path} is missing {', '.join(missing)}",
            hint="a staging directory is written only by its TraceWriter; this one cannot be sealed",
        )
    present = [*BUNDLE_FILES, *(name for name in OPTIONAL_FILES if (path / name).is_file())]
    files = {name: sha256_hex((path / name).read_bytes()) for name in present}
    manifest = BundleManifest(episode_id=episode_id, files=files, complete=True)
    atomic_write_text(path / MANIFEST_FILE, pretty(manifest.model_dump(mode="json", by_alias=True)))
    return manifest


def _v1_missing(path: Path) -> list[str]:
    """A v1 bundle's required files it lacks: its header names sandbox_version v1 and its outcome is not
    harness_error (a row that failed to write them ends harness_error with a label, and seals)."""
    header = parse_json_as(EpisodeHeader, (path / EPISODE_FILE).read_bytes(), where=str(path / EPISODE_FILE))
    if header.sandbox_version != "v1":
        return []
    outcome = parse_json_as(Outcome, (path / OUTCOME_FILE).read_bytes(), where=str(path / OUTCOME_FILE))
    if outcome.end_reason == "harness_error":
        return []
    return [name for name in V1_REQUIRED if not (path / name).is_file()]


def _upload(path: Path, store: TraceStore, prefix: str) -> None:
    """Every staged file to the store, bundle.json last. Every file is checked against bundle.json before the first
    byte leaves, so a staging copy that changed after sealing never becomes half a bundle in the store."""
    manifest_bytes = (path / MANIFEST_FILE).read_bytes()
    manifest = parse_json_as(BundleManifest, manifest_bytes, where=str(path / MANIFEST_FILE))
    names = [name for name in UPLOAD_ORDER if name in manifest.files]
    names += sorted(set(manifest.files) - set(UPLOAD_ORDER))
    staged: list[tuple[str, bytes]] = []
    for name in names:
        if "/" in name or name.startswith(".") or not (path / name).is_file():
            raise ValidationError(
                f"bundle.json in {path} lists {name!r}, which is not a staged file",
                hint="keep the staging directory for inspection; it cannot be uploaded as it is",
            )
        data = (path / name).read_bytes()
        if sha256_hex(data) != manifest.files[name]:
            raise ValidationError(
                f"staged {name} in {path} does not match bundle.json",
                hint="the staging copy changed after it was sealed; keep it for inspection and do not upload it",
            )
        staged.append((name, data))
    for name, data in staged:
        store.put_bytes(f"{prefix}/{name}", data)
    store.put_bytes(f"{prefix}/{MANIFEST_FILE}", manifest_bytes)


@dataclass(frozen=True, slots=True)
class StagedBundle:
    """A staged bundle that `recover_staged` left where it was, and why."""

    episode_id: str
    path: Path
    reason: str


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    uploaded: tuple[str, ...]
    """Store prefixes of the bundles uploaded and removed from staging, in directory order."""
    unfinished: tuple[StagedBundle, ...]
    """Everything else: still being written, never ended, or failing its own manifest."""


def recover_staged(staging_dir: Path, store: TraceStore, *, discard_unfinished: bool = False) -> RecoveryReport:
    """Upload every finished bundle that crashed writers left in `staging_dir`, and report the rest.

    A staged bundle is finished once outcome.json exists, because close() writes it first and nothing is appended
    after it; bundle.json is written here when the crash came before it. A directory held by a live writer (or by
    another recovery) is never touched. An unfinished bundle (its episode never ended) stays where it is, or is
    removed with `discard_unfinished=True` so its episode can be re-run. A finished bundle that fails its own
    manifest, or whose upload fails, is never removed. A directory without episode.json is reported and left alone,
    without even taking its lock: it is either a writer that is still opening, or not ours.
    """
    uploaded: list[str] = []
    unfinished: list[StagedBundle] = []
    if not staging_dir.is_dir():
        return RecoveryReport(uploaded=(), unfinished=())
    for path in sorted(p for p in staging_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        names = _staged_files(path)
        if EPISODE_FILE not in names:
            reason = "nothing staged yet" if not names else f"not a staged bundle: it has no {EPISODE_FILE}"
            unfinished.append(StagedBundle(path.name, path, reason))
            continue
        lock = _try_lock(path)
        if lock is None:
            unfinished.append(StagedBundle(path.name, path, "held by a live writer or another recovery"))
            continue
        try:
            if not (path / OUTCOME_FILE).is_file():
                reason = "no outcome.json: the episode never ended"
                if discard_unfinished:
                    shutil.rmtree(path)
                    reason += "; removed"
                unfinished.append(StagedBundle(path.name, path, reason))
                continue
            uploaded.append(_finish(path, store))
        except MCError as exc:
            unfinished.append(StagedBundle(path.name, path, f"not uploaded: {exc}"))
        finally:
            lock.close()
    return RecoveryReport(uploaded=tuple(uploaded), unfinished=tuple(unfinished))


def _finish(path: Path, store: TraceStore) -> str:
    if not (path / EPISODE_FILE).is_file():
        raise ValidationError(f"{path} has no {EPISODE_FILE}", hint="it was not staged by a TraceWriter")
    header = parse_json_as(EpisodeHeader, (path / EPISODE_FILE).read_bytes(), where=str(path / EPISODE_FILE))
    prefix = bundle_prefix(header)
    if not (path / MANIFEST_FILE).is_file():
        _write_manifest(path, header.episode_id)
    _upload(path, store, prefix)
    shutil.rmtree(path)
    return prefix

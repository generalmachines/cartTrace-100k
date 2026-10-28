"""The decision log: every decision the engine makes, one canonical JSON line each, readable back as Decisions.

Why a log. A decision type goes live only after it agrees with human labels, and a judge's verdict is trusted only if
each Clef answer behind it can be traced: the type and version, the hash of the evidence, the answer and the
probabilities, who decided (a rule, clef-flash, clef, or an abstention), the model snapshot and the latency. Lines are
`Decision.model_dump(mode="json")`, so they parse straight back into Decisions. Several processes (the runner, the
watch, the judge) may share `var/decisions/decisions.jsonl`, so each line goes through
`carttrace.core.files.append_jsonl_locked`, which holds an exclusive advisory lock on the log file itself for the append, and
no lock file is left beside a log that lives in a trace bundle.
Evidence itself is never written here, only its sha256 (see D10).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from carttrace.config.schema import Settings
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.files import append_jsonl_locked, read_jsonl
from carttrace.decide.service import DecisionSink
from carttrace.decide.types import Decision

LOG_NAME = "decisions.jsonl"


class DecisionLog:
    """An append-only JSONL file of decisions. A DecisionSink, and iterable as the Decisions it holds."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, decision: Decision) -> None:
        append_jsonl_locked(self.path, decision.model_dump(mode="json"))

    def __iter__(self) -> Iterator[Decision]:
        return iter_decisions(self.path)


class MemorySink:
    """Keeps decisions in a list: for tests, `carttrace decide try`, and callers that copy them into a trace bundle."""

    def __init__(self) -> None:
        self.decisions: list[Decision] = []

    def write(self, decision: Decision) -> None:
        self.decisions.append(decision)


class TeeSink:
    """Writes each decision to several sinks in turn, for example the run's log and the episode's bundle."""

    def __init__(self, *sinks: DecisionSink) -> None:
        self.sinks: Sequence[DecisionSink] = sinks

    def write(self, decision: Decision) -> None:
        for sink in self.sinks:
            sink.write(decision)


def read_decisions(path: Path) -> list[Decision]:
    """Every decision in a log file, or in every *.jsonl file of a directory in name order. A torn last line (a writer
    killed mid-append) is skipped; any other bad line raises ValidationError naming the file and line."""
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    elif path.exists():
        files = [path]
    else:
        raise NotFoundError(f"no decision log at {path}", hint="decision logs are written under var/decisions/")
    return [d for f in files for d in iter_decisions(f)]


def iter_decisions(path: Path) -> Iterator[Decision]:
    for number, record in enumerate(read_jsonl(path), start=1):
        try:
            yield Decision.model_validate(record)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"{path} record {number} is not a decision: {exc.errors()[0]['msg']}",
                hint="decision logs hold Decision.model_dump(mode='json') lines only",
            ) from exc


def default_log_path(settings: Settings) -> Path:
    return Path(settings.var_dir) / "decisions" / LOG_NAME


def jsonl_sink(settings: Settings) -> DecisionLog:
    """The `jsonl` block of decide.sink: the shared log under var/decisions/."""
    return DecisionLog(default_log_path(settings))


def memory_sink(settings: Settings) -> MemorySink:
    """The `memory` block of decide.sink: nothing reaches disk, for dry runs and CI."""
    del settings
    return MemorySink()

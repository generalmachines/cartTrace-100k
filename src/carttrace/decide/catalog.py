"""Decision types are data: this loads data/decisions/*.yaml and refuses any type that could not be answered.

Why check at load time. A type whose abstain value is outside its own answer space, whose question names evidence no
caller is asked for, or whose rule does not exist would fail on the first episode that asks it, or worse, log answers
that mean nothing. The catalog checks every type as it loads, so a bad edit fails `carttrace decide types` and the test
suite instead of a collection run.

Layout: one YAML file per engine (supervise.yaml, triage.yaml, judge.yaml and so on), each a list of types whose ids
start with the file's name. An engine with many types splits them by topic as `<prefix>-<topic>.yaml`
(judge-sizing.yaml holds judge.* types). Bump a type's `version` whenever its question, options or evidence change,
because human labels and calibration are kept per version.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from carttrace.core.canon import digest
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.decide.template import MIN_SHARE, placeholders, shown_keys, without_placeholders
from carttrace.decide.rules import RULES, RuleBook
from carttrace.decide.space import MAX_OPTIONS
from carttrace.decide.types import DecisionType

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "decisions"
"""data/decisions in this repository; callers with settings pass `settings.data_dir / "decisions"`."""
MAX_CHARS_CEILING = 60_000
"""No decision reads more evidence than this; a bigger budget is a typo, and Clef's context would not hold it."""

_ID = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
_WORD = re.compile(r"^[a-z][a-z0-9_]*$")


class Catalog:
    """Every decision type by id, each checked against the rule book it will run with."""

    def __init__(self, types: Iterable[DecisionType], *, rules: RuleBook = RULES) -> None:
        listed = list(types)
        problems = catalog_problems(listed, rules)
        if problems:
            raise _invalid(problems)
        self.rules = rules
        self._types = {t.id: t for t in sorted(listed, key=lambda t: t.id)}

    def get(self, type_id: str) -> DecisionType:
        dtype = self._types.get(type_id)
        if dtype is None:
            close = difflib.get_close_matches(type_id, list(self._types), n=3)
            hint = f"did you mean {', '.join(close)}?" if close else f"known types: {', '.join(self._types)}"
            raise ValidationError(f"unknown decision type {type_id!r}", hint=hint)
        return dtype

    def ids(self) -> list[str]:
        return list(self._types)

    def __iter__(self) -> Iterator[DecisionType]:
        return iter(self._types.values())

    def __len__(self) -> int:
        return len(self._types)

    def __contains__(self, type_id: object) -> bool:
        return type_id in self._types

    @property
    def sha(self) -> str:
        """Hash of every type definition: two catalogs with the same sha ask the same questions the same way."""
        return digest([t.model_dump(mode="json") for t in self])


def load_catalog(directory: Path | None = None, *, rules: RuleBook = RULES) -> Catalog:
    """Every type in `directory` (default: this repository's data/decisions), checked. All problems in all files are
    reported together."""
    root = directory if directory is not None else DEFAULT_DIR
    files = sorted(root.glob("*.yaml"))
    if not files:
        raise NotFoundError(
            f"no decision types in {root}", hint="decision types live in data/decisions/*.yaml under paths.data"
        )
    types: list[DecisionType] = []
    problems: list[str] = []
    for path in files:
        for index, entry in enumerate(_entries(path, problems)):
            try:
                dtype = DecisionType.model_validate(entry)
            except PydanticValidationError as exc:
                problems.append(f"{path.name} entry {index + 1}: {_summary(exc)}")
                continue
            prefix = dtype.id.split(".", 1)[0]
            if path.stem.split("-", 1)[0] != prefix:
                problems.append(f"{path.name}: {dtype.id} belongs in {prefix}.yaml or {prefix}-<topic>.yaml")
            types.append(dtype)
    problems.extend(catalog_problems(types, rules))
    if problems:
        raise _invalid(problems)
    return Catalog(types, rules=rules)


def catalog_problems(types: Iterable[DecisionType], rules: RuleBook = RULES) -> list[str]:
    """Everything wrong with a set of types, one sentence each; empty when every type can be answered."""
    problems: list[str] = []
    seen: set[str] = set()
    for dtype in types:
        if dtype.id in seen:
            problems.append(f"{dtype.id}: defined more than once")
        seen.add(dtype.id)
        problems.extend(f"{dtype.id}: {p}" for p in type_problems(dtype, rules))
    return problems


def type_problems(dtype: DecisionType, rules: RuleBook = RULES) -> list[str]:
    """Everything wrong with one type, one sentence each."""
    problems: list[str] = []
    if not _ID.match(dtype.id):
        problems.append("id must be dotted lowercase words, such as supervise.looping")
    if dtype.version < 1:
        problems.append("version must be 1 or more")
    if not 0.0 < dtype.threshold <= 1.0:
        problems.append("threshold must be above 0 and at most 1")
    if not dtype.question.strip():
        problems.append("question is empty")
    problems.extend(_evidence_problems(dtype))
    problems.extend(_option_problems(dtype))
    problems.extend(_rule_problems(dtype, rules))
    return problems


def _evidence_problems(dtype: DecisionType) -> list[str]:
    problems: list[str] = []
    if not dtype.evidence:
        problems.append("declares no evidence keys")
    bad = [k for k in dtype.evidence if not _WORD.match(k)]
    if bad:
        problems.append(f"evidence keys must be lowercase words: {bad}")
    repeated = sorted({k for k in dtype.evidence if dtype.evidence.count(k) > 1})
    if repeated:
        problems.append(f"evidence keys repeat: {repeated}")
    undeclared = sorted(set(placeholders(dtype.question)) - set(dtype.evidence))
    if undeclared:
        problems.append(f"question names evidence the type does not declare: {undeclared}")
    stripped = without_placeholders(dtype.question)
    if "{" in stripped or "}" in stripped:
        problems.append("question has a brace that is not a {placeholder} naming an evidence key")
    floor = MIN_SHARE * max(1, len(shown_keys(dtype)))
    if not floor <= dtype.max_chars <= MAX_CHARS_CEILING:
        problems.append(f"max_chars must be between {floor} ({MIN_SHARE} per shown key) and {MAX_CHARS_CEILING}")
    return problems


def _option_problems(dtype: DecisionType) -> list[str]:
    if dtype.kind == "bool":
        problems = []
        if dtype.options or dtype.options_from or dtype.option_notes:
            problems.append("a bool type takes no options, options_from or option_notes; its labels are yes and no")
        if not isinstance(dtype.abstain, bool):
            problems.append(f"abstain must be yes or no, got {dtype.abstain!r}")
        return problems
    return [*_choice_problems(dtype), *_abstain_problems(dtype)]


def _choice_problems(dtype: DecisionType) -> list[str]:
    problems: list[str] = []
    options = dtype.options
    bad = [o for o in options if not _WORD.match(o)]
    if bad:
        problems.append(f"options must be lowercase words: {bad}")
    repeated = sorted({o for o in options if options.count(o) > 1})
    if repeated:
        problems.append(f"options repeat: {repeated}")
    if dtype.options_from is None and len(options) < 2:
        problems.append("an enum or multi type needs at least two options, or options_from")
    if dtype.options_from is not None and dtype.options_from not in dtype.evidence:
        problems.append(f"options_from {dtype.options_from!r} is not a declared evidence key")
    if dtype.options_from is not None and not options:
        problems.append("a type with options_from still needs a static option to hold its abstain value")
    if len(options) > MAX_OPTIONS:
        problems.append(f"at most {MAX_OPTIONS} options, got {len(options)}")
    stray = sorted(set(dtype.option_notes) - set(options))
    if stray:
        problems.append(f"option_notes name options that do not exist: {stray}")
    if any("\n" in note for note in dtype.option_notes.values()):
        problems.append("option_notes must be one line each")
    return problems


def _abstain_problems(dtype: DecisionType) -> list[str]:
    options, abstain = dtype.options, dtype.abstain
    if dtype.kind == "enum" and not (isinstance(abstain, str) and abstain in options):
        return [f"abstain must be one of the type's own options {options}, got {abstain!r}"]
    if dtype.kind == "multi" and not (
        isinstance(abstain, list) and set(abstain) <= set(options) and len(set(abstain)) == len(abstain)
    ):
        return [f"abstain must be a list of distinct options from {options}, got {abstain!r}"]
    return []


def _rule_problems(dtype: DecisionType, rules: RuleBook) -> list[str]:
    if dtype.rule is None:
        return []
    if dtype.rule not in rules:
        return [f"rule {dtype.rule!r} does not exist; known rules: {rules.names()}"]
    missing = [k for k in rules.get(dtype.rule).needs if k not in dtype.evidence]
    return [f"rule {dtype.rule!r} reads evidence the type does not declare: {missing}"] if missing else []


def _entries(path: Path, problems: list[str]) -> list[Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        problems.append(f"{path.name}: not valid YAML: {exc}")
        return []
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return raw
    problems.append(f"{path.name}: must hold a list of decision types, got {type(raw).__name__}")
    return []


def _summary(exc: PydanticValidationError) -> str:
    return "; ".join(f"{'.'.join(str(x) for x in e['loc']) or 'entry'}: {e['msg']}" for e in exc.errors())


def _invalid(problems: list[str]) -> ValidationError:
    return ValidationError(
        "invalid decision types:\n  " + "\n  ".join(problems),
        hint="fix data/decisions/*.yaml; `carttrace decide types` checks them again",
    )

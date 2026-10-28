"""Reading vendored spec documents (JSON, and YAML for OpenAPI) into plain JSON values.

Why not `json.load` and `yaml.safe_load`. Everything read here ends up hashed, diffed or shown to an agent, so it must
be plain JSON: strings, numbers, booleans, null, lists and string-keyed objects. PyYAML implements YAML 1.1, which
turns an unquoted 2026-04-17 into a date, `yes` and `on` into booleans and `200:` into an integer key; the loader below
keeps the YAML 1.2 core meaning instead (dates stay strings, only true and false are booleans, every key is a string).
ACP 2026-01-30 also puts a tab between `description:` and its value, which YAML 1.2 allows as separation and PyYAML
rejects; that one shape is repaired and recorded rather than failing the whole version. JSON is parsed as written,
except that NaN and Infinity are refused because canonical JSON cannot hold them. A name repeated inside one object
keeps its last value, as every common parser does, and is recorded so the lint can report it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from carttrace.core.errors import ValidationError

STRUCTURED_SUFFIXES = (".json", ".yaml", ".yml")
_MAX_TAB_FIXES = 64


@dataclass(slots=True)
class LoadNotes:
    """What the loader had to tolerate in one document: repeated names and repaired YAML tabs."""

    duplicate_keys: list[str] = field(default_factory=list)
    tab_fixes: list[int] = field(default_factory=list)
    """1-based line numbers where a separation tab was replaced by a space."""

    @property
    def clean(self) -> bool:
        return not self.duplicate_keys and not self.tab_fixes


def is_structured(path: str | Path) -> bool:
    return str(path).endswith(STRUCTURED_SUFFIXES)


def load_document(path: Path, notes: LoadNotes | None = None) -> Any:
    """The JSON value of a vendored .json, .yaml or .yml file. Raises ValidationError when it does not parse."""
    notes = notes if notes is not None else LoadNotes()
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{path.name} is not UTF-8", hint="re-vendor it; upstream files are UTF-8") from exc
    if path.suffix == ".json":
        return _load_json(text, path, notes)
    if path.suffix in (".yaml", ".yml"):
        return _load_yaml(text, path, notes)
    raise ValidationError(f"{path.name} is not a JSON or YAML document")


def _load_json(text: str, path: Path, notes: LoadNotes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                notes.duplicate_keys.append(key)
            out[key] = value
        return out

    def no_constants(name: str) -> Any:
        raise ValueError(f"{name} is not JSON")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=no_constants)
    except ValueError as exc:
        raise ValidationError(f"{path.name} is not valid JSON: {exc}", hint="re-vendor it, or report upstream") from exc


class _CoreLoader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 core scalars: no timestamps, and only true/false are booleans."""


_CoreLoader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag not in ("tag:yaml.org,2002:timestamp", "tag:yaml.org,2002:bool")]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_CoreLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def _load_yaml(text: str, path: Path, notes: LoadNotes) -> Any:
    for _ in range(_MAX_TAB_FIXES):
        try:
            return _plain(yaml.load(text, Loader=_CoreLoader))  # noqa: S506 - _CoreLoader is a SafeLoader
        except yaml.MarkedYAMLError as exc:
            repaired = _separation_tab_repaired(text, exc)
            if repaired is None:
                raise ValidationError(f"{path.name} is not valid YAML: {exc}", hint="report upstream") from exc
            text, line = repaired
            notes.tab_fixes.append(line)
    raise ValidationError(f"{path.name} needs more than {_MAX_TAB_FIXES} tab repairs", hint="report upstream")


def _separation_tab_repaired(text: str, exc: yaml.MarkedYAMLError) -> tuple[str, int] | None:
    """Replace the tab PyYAML stopped at when it only separates a mapping key's colon from its value."""
    mark = exc.problem_mark
    if mark is None:
        return None
    lines = text.split("\n")
    line = lines[mark.line]
    head, tail = line[: mark.column], line[mark.column :]
    if not tail.startswith("\t") or not head.rstrip(" \t").endswith(":"):
        return None
    lines[mark.line] = head.rstrip(" \t") + " " + tail.lstrip(" \t")
    return "\n".join(lines), mark.line + 1


def _plain(value: Any) -> Any:
    """String keys everywhere, so the value is JSON (YAML allows `200:` as an integer key)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value

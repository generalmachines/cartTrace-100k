"""Why a request does not match its schema, said without repeating what the request held.

Why shared. Every surface checks what a caller sent against the vendored schema of the version it serves (UCP tool
inputs, ACP request bodies, AP2 mandates), and each must report the same way: the field, by an RFC 9535 JSONPath, and
what the schema wants, in words built from the schema and never from the value. A card number or a secret pasted into
a field therefore cannot come back in a reply or reach the event log through its own refusal (the same rule as
`carttrace.sandbox.core.guard`). Paths drop a leading payload key, the way stores report them (`$.line_items[0].quantity`
for `cart.line_items[0].quantity`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jsonschema.exceptions import ValidationError as SchemaError
from jsonschema.exceptions import best_match

MAX_PROBLEMS = 10
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_KEY = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True, slots=True)
class Problem:
    """One reason a request does not match its schema."""

    path: str
    """RFC 9535 JSONPath into the request, relative to the domain payload."""
    code: str
    """`field_required`, or `invalid_<field>` (`invalid_quantity`), as stores name them."""
    content: str


def problems(validator: Any, instance: Any, payload: str | None = None) -> list[Problem]:
    """Every problem `validator` finds with `instance`, deterministic and at most `MAX_PROBLEMS`."""
    found: dict[tuple[str, str], Problem] = {}
    for err in validator.iter_errors(instance):
        for p in _problems(_specific(err), payload):
            found.setdefault((p.path, p.code), p)
    return [found[k] for k in sorted(found)][:MAX_PROBLEMS]


def _specific(err: SchemaError) -> SchemaError:
    """For a failed `oneOf`/`anyOf`, the sub-error that explains it best; otherwise the error itself."""
    while err.context:
        better = best_match(err.context)
        if better is None or better is err:
            break
        err = better
    return err


def _problems(err: SchemaError, payload: str | None) -> list[Problem]:
    parts = list(err.absolute_path)
    keyword = str(err.validator)
    if keyword == "required":
        present = err.instance if isinstance(err.instance, Mapping) else {}
        missing = [str(n) for n in _strings(err.validator_value) if n not in present]
        return [
            Problem(json_path([*parts, n], payload), "field_required", f"{_label([*parts, n])} is required")
            for n in missing
        ]
    return [Problem(json_path(parts, payload), _code(parts), f"{_label(parts)} {_says(err)}")]


SAYS: Mapping[str, Callable[[Any], str]] = {
    "type": lambda v: "must be of type " + (" or ".join(_strings(v)) if isinstance(v, list) else str(v)),
    "minimum": lambda v: f"must be at least {v}",
    "maximum": lambda v: f"must be at most {v}",
    "exclusiveMinimum": lambda v: f"must be more than {v}",
    "exclusiveMaximum": lambda v: f"must be less than {v}",
    "minItems": lambda v: f"must have at least {v} item(s)",
    "maxItems": lambda v: f"must have at most {v} item(s)",
    "minLength": lambda v: f"must have at least {v} character(s)",
    "maxLength": lambda v: f"must have at most {v} character(s)",
    "minProperties": lambda v: f"must have at least {v} field(s)",
    "maxProperties": lambda v: f"must have at most {v} field(s)",
    "enum": lambda v: f"must be one of {sorted(map(str, v)) if isinstance(v, list) else v}",
    "const": lambda v: f"must be {v!r}",
    "pattern": lambda v: f"must match the pattern {v}",
    "format": lambda v: f"must be a valid {v}",
    "additionalProperties": lambda v: "has fields this schema does not allow",
    "unevaluatedProperties": lambda v: "has fields this schema does not allow",
    "uniqueItems": lambda v: "must not repeat items",
    "dependentRequired": lambda v: "is missing a field it depends on",
}
"""What each JSON Schema keyword wants, in words built from the schema's own value, never the instance's."""


def _says(err: SchemaError) -> str:
    keyword = str(err.validator)
    say = SAYS.get(keyword)
    return say(err.validator_value) if say is not None else f"does not match the schema ({keyword})"


def _strings(value: object) -> list[str]:
    return [str(x) for x in value] if isinstance(value, list) else []


def _code(parts: Sequence[object]) -> str:
    """`invalid_<last named field>`, or `invalid_argument` when no field is named."""
    names = [p for p in parts if isinstance(p, str) and _IDENT.match(p)]
    return f"invalid_{names[-1].lower()}" if names else "invalid_argument"


def _label(parts: Sequence[object]) -> str:
    named = [p for p in parts if isinstance(p, str)]
    shown = named[-1] if named and _KEY.match(named[-1]) else None
    return f"'{shown}'" if shown else "the request"


def json_path(parts: Iterable[object], payload: str | None = None) -> str:
    """RFC 9535 JSONPath for an instance path, dropping a leading domain payload key. Keys that are not plain
    identifiers are shown as `['?']`, so nothing the caller wrote as a key is echoed."""
    items = list(parts)
    if payload is not None and items and items[0] == payload:
        items = items[1:]
    out = "$"
    for p in items:
        if isinstance(p, int) and not isinstance(p, bool):
            out += f"[{p}]"
        elif isinstance(p, str) and _IDENT.match(p):
            out += f".{p}"
        elif isinstance(p, str) and _KEY.match(p):
            out += f"['{p}']"
        else:
            out += "['?']"
    return out

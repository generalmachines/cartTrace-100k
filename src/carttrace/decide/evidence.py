"""What a decision saw: exactly the evidence keys its type declares, as plain JSON, and the sha256 of that.

Why only the declared keys. Callers often pass more than a type needs (a whole status file, every turn so far).
Hashing everything would give two identical questions different hashes, so replays and human labels would no longer
join on `evidence_sha`; showing everything would let a rule or Clef read evidence the log cannot prove it saw. A
decision sees the keys its type declares and nothing else, and `evidence_sha` is the hash of exactly that, written as
canonical JSON. The same round trip turns pydantic models and tuples into plain JSON, so a rule and the prompt read the
very value that was hashed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from carttrace.core.canon import dumps, sha256_hex
from carttrace.core.errors import ValidationError
from carttrace.decide.types import DecisionType


@dataclass(frozen=True, slots=True)
class Evidence:
    values: Mapping[str, Any]
    """The declared keys only, as plain JSON, read-only at the top level."""
    sha: str
    """sha256 hex of the canonical JSON of `values`."""


def select_evidence(dtype: DecisionType, evidence: Mapping[str, Any]) -> Evidence:
    """The evidence a decision of `dtype` sees. Raises ValidationError when a declared key is missing or a value is not
    JSON. A key the caller cannot fill is passed as None, never left out, so a missing key is always a caller bug."""
    if not isinstance(evidence, Mapping):
        raise ValidationError(f"{dtype.id}: evidence must be a mapping of key to value, got {type(evidence).__name__}")
    missing = [k for k in dtype.evidence if k not in evidence]
    if missing:
        raise ValidationError(
            f"{dtype.id} needs evidence {missing}",
            hint=f"pass every key the type declares, {dtype.evidence}, with None for a value that is unknown",
        )
    text = canonical_text({k: evidence[k] for k in dtype.evidence}, what=f"{dtype.id} evidence")
    return Evidence(MappingProxyType(json.loads(text)), sha256_hex(text))


def plain_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    """The caller's context as plain JSON, because it is logged with every decision."""
    if context is None:
        return {}
    if not isinstance(context, Mapping):
        raise ValidationError(f"decision context must be a mapping, got {type(context).__name__}")
    value: dict[str, Any] = json.loads(canonical_text(context, what="decision context"))
    return value


def canonical_text(value: Any, *, what: str) -> str:
    """Canonical JSON of `value`, with pydantic models dumped first. ValidationError when it is not JSON."""
    try:
        return dumps(_plain(value))
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{what} is not plain JSON: {exc}",
            hint="pass strings, numbers, booleans, None, lists and dicts with string keys; dump models with "
            "model_dump(mode='json')",
        ) from exc


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value

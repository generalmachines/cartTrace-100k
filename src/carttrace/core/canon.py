"""Canonical JSON and content hashes: the one serialisation every deterministic artefact goes through.

Why one function. A manifest, an envpack, a decision log or a report is only reproducible if the same value always
becomes the same bytes. Python's default `json.dumps` keeps insertion order and accepts NaN, so two equal dicts can
serialise differently and a float can produce text another parser rejects. Everything that is hashed or compared byte
for byte is written with `dumps` here: sorted keys, no whitespace, UTF-8 kept as is, NaN and Infinity refused.

`loads` is the way back for text from outside (a model server's reply, a config file). Python's `json.loads` accepts
NaN and Infinity, and reads 1e999 as infinity; a value holding one would fail much later, far from its cause, when a
trace or a cassette tries to write it. So `loads` refuses them where they arrive.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def dumps(value: Any) -> str:
    """Canonical JSON text. Raises ValueError on NaN or Infinity, and TypeError on values JSON cannot hold."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def dumps_line(value: Any) -> str:
    """`dumps` for one line of a JSONL file: the same text with U+2028 and U+2029 written as \\u escapes.

    Why. JSON allows both raw inside a string, and `dumps` keeps UTF-8 as is, but JavaScript and Python's
    `str.splitlines` treat them as line breaks, so a product description holding one splits a trace line in two for
    every such reader (D69). The escape reads back as the same character, so the value is unchanged; `dumps` itself
    keeps them raw, because every digest and id already made over such a value must keep its bytes."""
    return dumps(value).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def dumps_bytes(value: Any) -> bytes:
    return dumps(value).encode("utf-8")


def loads(text: str | bytes) -> Any:
    """The value of JSON text, refusing what `dumps` cannot write back: NaN, Infinity, -Infinity, and numbers too
    large for a float.

    Text that is not JSON raises json.JSONDecodeError; a refused number raises a plain ValueError, so a caller can
    tell the two apart. Both are ValueErrors.
    """
    return json.loads(text, parse_constant=_refuse_constant, parse_float=_finite)


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a JSON value")


def _finite(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"{text} does not fit in a float")
    return number


def sha256_hex(data: bytes | str) -> str:
    """Hex sha256 of raw bytes, or of a string's UTF-8 encoding."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def digest(value: Any, length: int = 64) -> str:
    """Hex sha256 of a value's canonical JSON, cut to `length` characters (at least 8)."""
    if not 8 <= length <= 64:
        raise ValueError(f"digest length must be between 8 and 64, got {length}")
    return sha256_hex(dumps_bytes(value))[:length]


def pretty(value: Any) -> str:
    """Canonical content with indentation, for files people read. Same key order as `dumps`, so diffs stay small."""
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"

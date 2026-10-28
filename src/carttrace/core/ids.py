"""Stable ids and seeded randomness.

Why hashes and not uuid4. An id derived from what a thing is (a namespace plus the parts that identify it) is the same
on every re-run, so a resumed run skips work it already did and two machines agree on names without talking. Random
choices come from a `random.Random` seeded the same way, so a manifest drawn twice from the same inputs is identical.
"""

from __future__ import annotations

import random
import re

from carttrace.core.canon import digest

_NAMESPACE = re.compile(r"^[a-z][a-z0-9]{0,15}$")


def stable_id(namespace: str, *parts: object, length: int = 12) -> str:
    """`<namespace>-<hex>` from the namespace and the parts, e.g. stable_id("ep", run_id, 17) -> "ep-3f2a...".

    Parts are serialised as canonical JSON, so ("a", 1) and ("a", "1") give different ids, and the order of the parts
    matters.
    """
    if not _NAMESPACE.match(namespace):
        raise ValueError(
            f"id namespace must be 1-16 lowercase letters or digits starting with a letter, got {namespace!r}"
        )
    return f"{namespace}-{digest([namespace, *[_plain(p) for p in parts]], length)}"


def seeded_rng(namespace: str, *parts: object) -> random.Random:
    """A Random whose sequence depends only on the namespace and the parts. Never share one across unrelated draws."""
    return random.Random(int(digest([namespace, *[_plain(p) for p in parts]], 16), 16))  # noqa: S311 - reproducible draws, not secrets


def _plain(part: object) -> object:
    if part is None or isinstance(part, str | int | float | bool):
        return part
    if isinstance(part, list | tuple):
        # a plain str is its own plain form: task specs hold long lists of variant ids
        return [p if type(p) is str else _plain(p) for p in part]
    if isinstance(part, dict):
        return {str(k): _plain(v) for k, v in part.items()}
    raise TypeError(f"id parts must be JSON scalars, lists or dicts, got {type(part).__name__}")

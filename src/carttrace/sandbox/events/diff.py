"""State diffs and the diff check a grader runs: expected is inside the change, and the change is inside expected plus
allowed.

Why this shape. A journey is judged right when the store changed the way the task needed (every expected change is
there) and in no other way that matters (every change is either expected or explicitly allowed). That catches both
the agent that never placed the order and the one that placed it and also bought something else, canceled a
stranger's cart, or applied a code it was told not to. It is how ORO and tau2-bench grade, from backend state rather
than prose.

Paths are slash-joined (`checkouts/checkout-ab12.../status`). A list of records with ids (cart lines, order lines) is
keyed by those ids, so reordering is not a change. Patterns use `*` for one path segment and `**` for any depth. A key
that itself holds a slash (a live store's `gid://shopify/ProductVariant/...` id) is escaped (`/` as `%2F`, `%` as
`%25`), so every segment is one segment; build exact paths with `path(...)`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Final

MISSING = "<absent>"
BOOKKEEPING = ("seq", "clock", "id_counter", "faults_fired/**", "idempotency/**", "sessions/**", "webhooks/**")
"""Changes every command makes or that only the environment cares about. Ignored by default; pass ignore=() to see
them."""


class _AnyValue:
    """Matches whatever value an expected path ends at."""

    def __repr__(self) -> str:
        return "ANY"


ANY: Final = _AnyValue()


def segment(key: object) -> str:
    """One path segment: the key with `%` and `/` escaped."""
    return str(key).replace("%", "%25").replace("/", "%2F")


def path(*parts: object) -> str:
    """An exact diff path from its parts, e.g. path("inventory", "gid://shopify/ProductVariant/1", "on_hand")."""
    return "/".join(segment(p) for p in parts)


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}/{segment(k)}" if prefix else segment(k)))
    elif isinstance(obj, list | tuple):
        keyed = bool(obj) and all(isinstance(x, Mapping) and "id" in x for x in obj)
        if keyed and len({x["id"] for x in obj}) == len(obj):
            for x in obj:
                out.update(flatten(x, f"{prefix}/{segment(x['id'])}"))
        elif obj and all(isinstance(x, list | tuple | Mapping) for x in obj):
            for i, x in enumerate(obj):
                out.update(flatten(x, f"{prefix}/{i}"))
        else:
            out[prefix] = list(obj)
    else:
        out[prefix] = obj
    return out


@lru_cache(maxsize=512)
def _rx(pattern: str) -> re.Pattern[str]:
    parts = [".*" if seg == "**" else re.escape(seg).replace(r"\*", "[^/]*") for seg in pattern.split("/")]
    return re.compile("^" + "/".join(parts).replace("/.*", "(/.*)?") + "$")


def matches(path: str, pattern: str) -> bool:
    return bool(_rx(pattern).match(path))


def diff(
    before: Mapping[str, Any], after: Mapping[str, Any], *, ignore: Iterable[str] = BOOKKEEPING
) -> dict[str, tuple[Any, Any]]:
    """Every path whose value differs: {path: (before, after)}, with MISSING for a path on one side only."""
    a, b = flatten(before), flatten(after)
    ign = [_rx(p) for p in ignore]
    out = {}
    for p in sorted(set(a) | set(b)):
        va, vb = a.get(p, MISSING), b.get(p, MISSING)
        if va != vb and not any(r.match(p) for r in ign):
            out[p] = (va, vb)
    return out


@dataclass(frozen=True, slots=True)
class DiffCheck:
    ok: bool
    missing: list[str] = field(default_factory=list)
    """Expected changes that did not happen."""
    mismatched: list[str] = field(default_factory=list)
    """Expected paths that changed, but to the wrong value."""
    unexpected: list[str] = field(default_factory=list)
    """Changes neither expected nor allowed."""


def check(
    delta: Mapping[str, tuple[Any, Any]], expected: Mapping[str, object], allowed: Iterable[str] = ()
) -> DiffCheck:
    """`expected`: {pattern: the value the path must end at, or ANY}. `allowed`: patterns that may change or not."""
    missing, mismatched, unexpected = [], [], []
    for pat, want in expected.items():
        hits = [p for p in delta if matches(p, pat)]
        if not hits:
            missing.append(pat)
        elif want is not ANY and not any(delta[p][1] == want for p in hits):
            mismatched.append(f"{pat}: wanted {want!r}, got {[delta[p][1] for p in hits][:3]!r}")
    pats = [*expected, *allowed]
    for p in delta:
        if not any(matches(p, pat) for pat in pats):
            unexpected.append(p)
    return DiffCheck(not (missing or mismatched or unexpected), missing, mismatched, unexpected)

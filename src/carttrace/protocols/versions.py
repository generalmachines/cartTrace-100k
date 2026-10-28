"""Version strings sort the way people read them.

Why not plain string order. Protocol versions here are dates (UCP, ACP, MCP: "2026-04-08"), release tags (AP2:
"v0.2.0") or a package release (MPP: "pympp-0.11.0"). Dates sort correctly as strings, but "v0.10.0" would sort before
"v0.9.0". Splitting a version into runs of digits and non-digits and comparing the digit runs as numbers orders all
three shapes correctly, and the same input always gives the same order.
"""

from __future__ import annotations

import re

_RUNS = re.compile(r"(\d+)")


def version_key(version: str) -> tuple[tuple[int, int | str], ...]:
    """A sort key: digit runs compare as numbers, other runs as text, and numbers sort before text at a position."""
    parts: list[tuple[int, int | str]] = []
    for run in _RUNS.split(version):
        if not run:
            continue
        parts.append((0, int(run)) if run.isdigit() else (1, run))
    return tuple(parts)


def sort_versions(versions: list[str]) -> list[str]:
    return sorted(versions, key=lambda v: (version_key(v), v))

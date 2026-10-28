"""How a decision type's question names its evidence, for the catalog's checks."""

from __future__ import annotations

import re

from carttrace.decide.types import DecisionType

MIN_SHARE = 200
"""Every shown evidence value may keep at least this many characters; the catalog sizes max_chars to match."""

_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")


def placeholders(template: str) -> list[str]:
    """Evidence keys a question names as {key}, in order of appearance."""
    return _PLACEHOLDER.findall(template)


def without_placeholders(template: str) -> str:
    return _PLACEHOLDER.sub("", template)


def shown_keys(dtype: DecisionType) -> list[str]:
    """Evidence keys shown beside the question: every declared key except those it names and the options source."""
    named = set(placeholders(dtype.question))
    return [k for k in dtype.evidence if k not in named and k != dtype.options_from]

"""Schema.org JSON-LD as a page carries it: the blocks of a captured page, their types, and their reviewers' names
redacted (D67).

Why here. Two readers need the same blocks and sit on layers that may not import each other: the live page reader
(carttrace.pages.html, which reads them out of a page it fetched) and the capture v2 reader (carttrace.trace.capture_v2, which reads
them out of the raw product pages the capture pass kept, so a replica's website pages carry the JSON-LD its store
served). Both parse a block's text with `parse_block`.

Redaction. A Review's author is usually a person's name ("Jane D."), and CLAUDE.md asks for personal names to be
redacted at write time. `without_reviewers` replaces the name of every Person who authored a review with
REDACTED_NAME before a block is written into an envpack; nothing else in the block changes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from typing import Any, Final

REDACTED_NAME: Final = "[redacted]"
MAX_BLOCK_CHARS: Final = 2_000_000
"""A block longer than this is not read: the page reader's own bound for a data script (D36)."""
_SCRIPT = re.compile(
    r"<script\b[^>]*\btype\s*=\s*[\"']?application/ld\+json[\"']?[^>]*>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL
)
_REVIEW_KEYS: Final = frozenset({"review", "reviews"})
_NAME_KEYS: Final = frozenset({"name", "givenName", "familyName", "alternateName"})


def parse_block(raw: str) -> Any | None:
    """One ld+json script's JSON, with the HTML comment and CDATA wrappers and a trailing semicolon some themes leave
    taken off; None when it is not JSON."""
    text = raw.strip().removeprefix("<!--").removesuffix("-->").strip()
    text = text.removeprefix("<![CDATA[").removesuffix("]]>").strip().rstrip(";").strip()
    if not text or len(text) > MAX_BLOCK_CHARS:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def page_blocks(html: str) -> list[Any]:
    """Every JSON-LD block a page carries, in page order, as parsed JSON; blocks that do not parse are left out."""
    blocks = []
    for match in _SCRIPT.finditer(html):
        value = parse_block(match.group(1))
        if value is not None:
            blocks.append(value)
    return blocks


def types_of(blocks: Any) -> list[str]:
    """The schema.org types the blocks name anywhere (Product, Offer, AggregateRating, BreadcrumbList, ...), sorted,
    without their vocabulary prefix."""
    found: set[str] = set()
    for node in _nodes(blocks):
        kinds = node.get("@type")
        for kind in kinds if isinstance(kinds, list) else [kinds]:
            if isinstance(kind, str) and kind.strip():
                found.add(kind.rsplit("/", 1)[-1].rsplit(":", 1)[-1].strip())
    return sorted(found)


def without_reviewers(value: Any, *, review: bool = False) -> Any:
    """The blocks with each review author's name redacted (a Person, or a bare name string), everything else as is.
    `review` marks the items of a `review` list, which some themes write without an @type."""
    if isinstance(value, list):
        return [without_reviewers(v, review=review) for v in value]
    if not isinstance(value, Mapping):
        return value
    is_review = review or _is_review(value)
    out: dict[str, Any] = {}
    for key, item in value.items():
        if is_review and key == "author":
            out[key] = _redacted_author(item)
        else:
            out[key] = without_reviewers(item, review=key in _REVIEW_KEYS)
    return out


def _is_review(node: Mapping[str, Any]) -> bool:
    kinds = node.get("@type")
    return any(isinstance(k, str) and k.endswith("Review") for k in (kinds if isinstance(kinds, list) else [kinds]))


def _redacted_author(author: Any) -> Any:
    if isinstance(author, str):
        return REDACTED_NAME
    if isinstance(author, list):
        return [_redacted_author(a) for a in author]
    if isinstance(author, Mapping):
        kinds = author.get("@type")
        listed = kinds if isinstance(kinds, list) else [kinds]
        if any(isinstance(k, str) and "Organization" in k for k in listed):
            return dict(author)
        return {k: (REDACTED_NAME if k in _NAME_KEYS else v) for k, v in author.items()}
    return author


def _nodes(value: Any, depth: int = 0) -> Iterator[Mapping[str, Any]]:
    if depth > 40:
        return
    if isinstance(value, list):
        for item in value:
            yield from _nodes(item, depth + 1)
    elif isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _nodes(item, depth + 1)

"""Plain-text helpers and the small detectors that lift decision-critical details out of free text.

Why detectors at all. Sources state some details in fields (ABO `material`, Open Food Facts `allergens`) and many
only in prose: care instructions in a bullet point, a dosage line in a supplement listing, a size in a shoe's
title. A detector here is a regular expression that finds the line saying it, so the detail reaches `Details` as the
source's own words, never as something we made up; the product's provenance records which field it was read from.
They are deliberately conservative: a missed detail is a store that does not state it, while a false one would
credit a store with information it never gave.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable

from carttrace.core.canon import digest

_TAG = re.compile(r"<[^>]+>")
_MOJIBAKE = re.compile("[ÂÃâðï][\u0080-¿]")

CARE = re.compile(
    r"\b(machine[- ]wash(?:able)?|hand[- ]wash|dry[- ]clean|spot[- ]clean|tumble[- ]dry|wipe (?:clean|with)"
    r"|do not (?:bleach|iron|tumble)|dishwasher[- ]safe|care instructions?|washing instructions?)\b",
    re.IGNORECASE,
)
DIRECTIONS = re.compile(
    r"\b(per serving|serving size|servings? per|daily dose|recommended (?:dose|dosage|use)|directions(?: for use)?:"
    r"|take \d|(?:tablets?|capsules?|softgels?|gummies) (?:a|per) day|\d+ (?:tablets?|capsules?|softgels?) daily)",
    re.IGNORECASE,
)
"""How much to take, and how often: the dosage a shopper checks first."""
STRENGTH = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|iu)\b", re.IGNORECASE)
"""Strength per dose ("2000 mcg"): the dosage detail when a listing gives no directions."""
WARNINGS = re.compile(
    r"\b(warnings?:|caution|keep out of (?:the )?reach of children|not suitable for|choking hazard|do not use if"
    r"|consult (?:a|your) (?:doctor|physician))",
    re.IGNORECASE,
)
ALLERGENS = re.compile(
    r"\b(allergens?|contains:? (?:milk|soy|tree nuts|nuts|peanuts|wheat|gluten|eggs?|fish|shellfish|sesame)"
    r"|may contain|gluten[- ]free|nut[- ]free|dairy[- ]free|lactose[- ]free|soy[- ]free)",
    re.IGNORECASE,
)
INGREDIENTS = re.compile(r"^\s*(?:full\s+)?ingredients?\s*[:\-]", re.IGNORECASE)
DIMENSIONS = re.compile(r"\d\s*(?:cm|mm|in|inch(?:es)?|ft|\"|''|')?\s*[A-Za-z]{0,2}\s*[x×]\s*\d", re.IGNORECASE)
COMPATIBLE = re.compile(r"\b(compatible with|compatibility|for use with|fits models?)\b", re.IGNORECASE)

_SIZE_US = re.compile(
    r"(?<![\w.])(\d{1,2}(?:\.5)?)\s*(?:(?:[A-F]{1,2}|E{2,4}|M|W|N|Wide|Narrow|X-Wide)\s*(?:\((?:M|W)\))?\s*)?US\b"
)
_SIZE_UK = re.compile(r"(?<![\w.])(\d{1,2}(?:\.5)?)\s*UK\b")
_SIZE_EU = re.compile(r"(?<![\w.])(\d{2}(?:\.5)?)\s*EU\b")
_SIZE_RING = re.compile(r"\bSize\s+(\d{1,2}(?:\.5)?)\b")
_SIZE_WIDTH = re.compile(r"^(\d{1,2}(?:\.5)?)\s*(?:M|W|N|D|B|EE|EEE|Wide|Narrow|X-Wide|Medium)$")
"""A shoe size with a width but no system ("12 M"), as US listings write it: the size, with its system unstated."""
_SIZE_ALPHA = re.compile(
    r"^(XXS|XS|S|M|L|XL|XXL|XXXL|[2-6]XL|X-Small|Small|Medium|Large|X-Large|XX-Large|XXX-Large)$", re.IGNORECASE
)


def clean(text: str | None, *, limit: int = 4000) -> str:
    """One line of plain text: entities decoded, tags and runs of whitespace removed, cut at `limit` characters."""
    if not text:
        return ""
    flat = " ".join(_TAG.sub(" ", html.unescape(text)).split())
    return flat[:limit].rstrip()


def paragraphs(parts: Iterable[str | None], *, limit: int = 8000) -> str:
    """Several cleaned pieces of text as one description, one piece per line, cut at `limit` characters."""
    lines = [c for c in (clean(p) for p in parts) if c]
    return "\n".join(dict.fromkeys(lines))[:limit].rstrip()


def fix_mojibake(text: str) -> str:
    """UTF-8 that was read as Latin-1 (`ï¼\\x9a` for a full-width colon), read back as UTF-8. Text that is not
    mojibake, or that does not round-trip, is returned unchanged."""
    if not text or not _MOJIBAKE.search(text):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def find_line(lines: Iterable[str], pattern: re.Pattern[str], *, limit: int = 300) -> str | None:
    """The first of `lines` (already cleaned with `clean`) that matches `pattern`, cut at `limit` characters."""
    for line in lines:
        if line and pattern.search(line):
            return line[:limit].rstrip()
    return None


def number(value: float) -> str:
    """A measurement as people write it: at most two decimals, no trailing zeros (87.0 -> "87", 3.0409 -> "3.04")."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def slug(text: str, *, limit: int = 48) -> str:
    """A lowercase ASCII slug for store keys and sandbox domains; a hash when nothing ASCII is left."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    out = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")[:limit].strip("-")
    return out or f"x{digest(text, 8)}"


def title_size(title: str) -> tuple[str, str | None, tuple[int, int]] | None:
    """The size a listing title names, its size system, and where it sits in the title.

    The leftmost of a US, UK or EU shoe size wins ("8 UK (42 EU) (9 US)" is a UK 8), then a ring "Size 8", then a
    letter size as the title's last comma-separated part ("..., Navy, Medium"). Call it only for apparel, footwear
    and jewellery, where "3 UK" is a size and not a plug.
    """
    found: list[tuple[int, int, str, str | None]] = []
    for pattern, system in ((_SIZE_US, "US"), (_SIZE_UK, "UK"), (_SIZE_EU, "EU")):
        match = pattern.search(title)
        if match:
            found.append((match.start(), match.end(), match.group(0).strip(), system))
    if found:
        start, end, label, named = min(found)
        return label, named, (start, end)
    ring = _SIZE_RING.search(title)
    if ring:
        return ring.group(0), None, (ring.start(), ring.end())
    last = title.rsplit(",", 1)
    if len(last) == 2:
        tail, start = last[1].strip(), len(last[0]) + 1
        if _SIZE_ALPHA.match(tail):
            return tail, "alpha", (start, len(title))
        if _SIZE_WIDTH.match(tail):
            return tail, None, (start, len(title))
    return None


def without(text: str, span: tuple[int, int] | None = None, phrase: str | None = None) -> str:
    """`text` with a span and a phrase (case-insensitive, whole words) removed, punctuation tidied: the part of a
    listing title its variants share."""
    out = text if span is None else text[: span[0]] + " " + text[span[1] :]
    if phrase:
        out = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*,\s*(?=,|$)", "", out)
    out = re.sub(r"[\s,;/-]+$", "", " ".join(out.split()))
    return out.strip(" ,;-")

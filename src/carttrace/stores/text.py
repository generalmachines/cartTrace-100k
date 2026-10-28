"""Text normalisation shared by the sheet snapshot, merchant grouping and the category files.

Why one module. Three small operations decide whether two strings are "the same": Unicode composition (a sheet cell
may hold "é" as one code point or as "e" plus an accent), whitespace (a header typed with a line break in it), and
accent folding for names ("Café Dumont" and "Cafe Dumont" are one company). Each is done once, here, so the snapshot, the
merchant keys and the category slugs agree with each other.
"""

from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def nfc(text: str) -> str:
    """Canonical composition, so equal-looking cells compare equal."""
    return unicodedata.normalize("NFC", text)


def collapse_ws(text: str) -> str:
    """Every run of whitespace (spaces, tabs, line breaks, no-break spaces) becomes one space; ends are trimmed."""
    return " ".join(text.split())


def fold(text: str) -> str:
    """Accents removed and case folded: "Café Dumont" -> "cafe dumont", "Élan" -> "elan"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def slugify(text: str) -> str:
    """A lowercase ASCII slug: "Clothing (adult)" -> "clothing-adult", "Store queue" -> "store-queue"."""
    return _NON_ALNUM.sub("-", fold(text).replace("&", " and ")).strip("-")

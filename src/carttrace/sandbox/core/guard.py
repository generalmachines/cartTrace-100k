"""Card numbers never enter the sandbox, and nothing a refused command said is echoed into the log.

Why. The repo's rule is that no card number appears anywhere, test cards included. Payment credentials are opaque
tokens, and the command schema refuses anything else as a token, but an agent can still be talked into pasting a card
number into an address line or a discount code. The event log keeps every accepted command's body, so a card number
that got past validation would be written to disk and replayed into every copy of the log. `find_card_number` scans
every free-text string of a raw command for a 13 to 19 digit run (spaces and dashes allowed) that passes the Luhn
check, and the command is refused before anything records it.

What is skipped. Id fields (`id`, `*_id`, `*_ids`, `*_ref`, `session`) are skipped, because Shopify ids are long digit
strings and some would pass the Luhn check by chance. A phone number is checked only for runs of 16 or more digits,
since E.164 numbers have at most 15. Refusals name the field by a sanitised path, never by the value, and a key that is
not identifier-shaped is shown as `?`, so a card number used as a key is not echoed either.

What a card number is (`luhn_ok`, `looks_like_card_number`) lives in carttrace.core.cards and is re-exported here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from carttrace.core.cards import PAN_DIGITS, is_identifier_key, looks_like_card_number, luhn_ok

__all__ = [
    "PAN_DIGITS",
    "PHONE_KEYS",
    "find_card_number",
    "looks_like_card_number",
    "luhn_ok",
    "safe_name",
    "safe_path",
    "safe_segment",
]

_SAFE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
PHONE_KEYS = frozenset({"phone_number", "phone"})


def _is_id_key(key: str) -> bool:
    """An identifier field (carttrace.core.cards.is_identifier_key, the one rule every card check shares) or the session."""
    return key == "session" or is_identifier_key(key)


def find_card_number(value: object, path: tuple[str | int, ...] = ()) -> str | None:
    """The sanitised path of the first free-text string in a JSON value that holds a card number, or None."""
    if isinstance(value, Mapping):
        for k, v in value.items():
            key = str(k)
            if _is_id_key(key):
                continue
            found = find_card_number(v, (*path, key))
            if found is not None:
                return found
        return None
    if isinstance(value, list | tuple):
        for i, v in enumerate(value):
            found = find_card_number(v, (*path, i))
            if found is not None:
                return found
        return None
    if isinstance(value, str) and not value.startswith("gid://"):
        # A gid:// id's digits are an id, whatever key holds it; seen 2026-10-11 when Open Food Facts barcodes, held
        # in lookup_catalog's `ids`, passed the Luhn check and the judge called them card numbers.
        last = next((p for p in reversed(path) if isinstance(p, str)), "")
        if looks_like_card_number(value, min_digits=16 if last in PHONE_KEYS else PAN_DIGITS[0]):
            return safe_path(path) or "command"
    return None


def safe_segment(seg: object) -> str:
    """A path segment fit for a log: an index or an identifier-shaped key, else `?`."""
    if isinstance(seg, int) and not isinstance(seg, bool):
        return str(seg)
    s = str(seg)
    return s if _SAFE.match(s) else "?"


def safe_path(loc: Sequence[object]) -> str:
    return ".".join(safe_segment(s) for s in loc)


def safe_name(value: object) -> str | None:
    """A command type as the caller gave it, only when it is identifier-shaped; otherwise None."""
    return value if isinstance(value, str) and _SAFE.match(value) else None

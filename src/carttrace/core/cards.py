"""What counts as a payment card number: the one rule everything that keeps card numbers out agrees on.

Why in core. The repo's rule is that no card number appears anywhere, test cards included. The sandbox refuses a
command that carries one (carttrace.sandbox.core.guard), and any other check that keeps card numbers out of what we store
must recognise exactly the same thing: a run of 13 to 19 digits, single spaces or dashes allowed between them, that
passes the Luhn check.
"""

from __future__ import annotations

import re

PAN_DIGITS = (13, 19)
"""The fewest and the most digits a card number has."""

_RUN = re.compile(r"\d(?:[ -]?\d)*")


def luhn_ok(digits: str) -> bool:
    """The Luhn (mod 10) check every payment card number passes."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def looks_like_card_number(text: str, *, min_digits: int = PAN_DIGITS[0]) -> bool:
    """True when `text` holds a maximal digit run of card length that passes the Luhn check."""
    for m in _RUN.finditer(text):
        digits = m.group().replace(" ", "").replace("-", "")
        if min_digits <= len(digits) <= PAN_DIGITS[1] and luhn_ok(digits):
            return True
    return False


_ID_KEY = re.compile(r"^(?:id|ids|cursor|session|handle)$|(?:_id|_ids|_ref|Id|Ids)$")


def is_identifier_key(key: str) -> bool:
    """True for a JSON key whose value is an identifier (id, ids, cursor, handle, *_id, *Id ...). Any digit string can
    be an id, and roughly one in ten passes the Luhn check, so card checks leave these values alone."""
    return bool(_ID_KEY.search(key.removesuffix("[]")))

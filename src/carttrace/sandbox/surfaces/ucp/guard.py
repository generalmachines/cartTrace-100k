"""Card numbers are refused at the surface, before a tool call's arguments go anywhere.

Why here as well as in the core. The core refuses a command that carries a card number (`carttrace.sandbox.core.guard`),
but a UCP request carries free text the core never sees: a search query, `context.intent`, an instrument's `display`,
a `pan_credential` that UCP 2026-08-25 defines and the sandbox does not take. A search query in particular becomes a
`read` command's `id`, which the core's scan skips because ids are long digit strings. So every tool call's arguments
are scanned whole, with the core's own rule (`carttrace.core.cards`), and a call that carries a card number is refused
without echoing it.

What is skipped. The core skips id fields (`id`, `*_id`, `*_ids`, `*_ref`); UCP adds `ids` (catalog lookup, which takes
product and variant gids) and `idempotency-key`, both opaque identifiers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from carttrace.sandbox.core.guard import find_card_number

OPAQUE_KEYS = frozenset({"ids", "idempotency-key"})


def card_number_field(arguments: Mapping[str, Any]) -> str | None:
    """The sanitised path of the first field holding a card number, or None."""
    return find_card_number(_without_opaque(arguments))


def _without_opaque(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _without_opaque(v) for k, v in value.items() if str(k) not in OPAQUE_KEYS}
    if isinstance(value, list | tuple):
        return [_without_opaque(v) for v in value]
    return value

"""The opaque alias an agent is shown in place of a persona's test token, and how the sandbox's PSP reads it back.

Why. The persona's test tokens name what they are built to do (`tok_visa_3ds_fail`, `tok_visa_decline_...`), and the
wallet handed them to the agent before it paid: in ep-edf2330df6b3 the agent wrote "The token name
'tok_visa_3ds_fail' suggests that 3-D Secure will fail", so a designed failure was not blind. The wallet now shows
an alias in a processor's token shape, a digest of the episode's session and the token, and the sandbox resolves it
back to the test token below every surface (`carttrace.sandbox.surfaces.host.SandboxHost.execute`), so the token's fault
stays inside the PSP. Both sides compute it from the same two values, so nothing is registered and replays need
nothing new: the core's commands carry the test token, as before.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Final

from carttrace.core.canon import digest

ALIAS_PREFIX: Final = "tok_1"
"""A processor's token shape (Stripe's `tok_1...`): an alias reads as any card token does."""


LETTERS: Final = "ghijkmnpqrstuvwx"
"""One letter per hex digit, none of them hex, for every sixth character of an alias."""


def wallet_alias(key: str, token: str) -> str:
    """The alias of `token` in the episode whose session is `key`: stable, opaque, and naming nothing.

    Every sixth character is a letter, so no run of digits is ever long enough to read as a card number: a plain
    hex digest can hold one (payments-check's ep-7d9255cce287 was refused `card_number_refused` on
    tok_170e5438541710577..., whose 13-digit run passes the Luhn check)."""
    raw = digest(["wallet-alias", key, token], 23)
    return ALIAS_PREFIX + "".join(LETTERS[int(c, 16)] if i % 6 == 5 else c for i, c in enumerate(raw))


def aliases(key: str, tokens: Iterable[str]) -> dict[str, str]:
    """Alias to token, for every token in `tokens`, in the episode `key`."""
    return {wallet_alias(key, t): t for t in tokens}


def unalias(value: Any, table: Mapping[str, str]) -> Any:
    """`value` with every string that is an alias in `table` replaced by its token, anywhere in a JSON value."""
    if isinstance(value, str):
        return table.get(value, value)
    if isinstance(value, list):
        return [unalias(v, table) for v in value]
    if isinstance(value, Mapping):
        return {k: unalias(v, table) for k, v in value.items()}
    return value

"""Layer 0: the primitives every other package shares. Standard library only.

- `canon`: canonical JSON and content hashes, and `loads`, its strict inverse
- `ids`: stable ids and seeded randomness
- `clock`: injected clocks and the one timestamp format
- `files`: atomic writes and JSONL, including a locked append for logs several processes share
- `aio`: `gather_all`, awaitables run together that fail as one, with the first failure as itself
- `money`: integer minor units plus a currency code
- `cards`: what counts as a payment card number, for the checks that keep them out
- `errors`: typed errors with a next-action hint
"""

from carttrace.core.aio import gather_all
from carttrace.core.canon import digest, dumps, dumps_bytes, loads, pretty, sha256_hex
from carttrace.core.cards import looks_like_card_number, luhn_ok
from carttrace.core.clock import Clock, FrozenClock, SystemClock, iso
from carttrace.core.errors import (
    ConfigError,
    ExternalServiceError,
    MCError,
    NotFoundError,
    ProtocolError,
    SafetyError,
    ValidationError,
)
from carttrace.core.files import (
    JsonlError,
    append_jsonl,
    append_jsonl_locked,
    atomic_write_bytes,
    atomic_write_text,
    iter_jsonl_bytes,
    iter_jsonl_numbered,
    read_jsonl,
)
from carttrace.core.ids import seeded_rng, stable_id
from carttrace.core.money import Money, exponent, to_major, to_minor

__all__ = [
    "Clock",
    "ConfigError",
    "ExternalServiceError",
    "FrozenClock",
    "JsonlError",
    "MCError",
    "Money",
    "NotFoundError",
    "ProtocolError",
    "SafetyError",
    "SystemClock",
    "ValidationError",
    "append_jsonl",
    "append_jsonl_locked",
    "atomic_write_bytes",
    "atomic_write_text",
    "digest",
    "dumps",
    "dumps_bytes",
    "exponent",
    "gather_all",
    "iso",
    "iter_jsonl_bytes",
    "iter_jsonl_numbered",
    "loads",
    "looks_like_card_number",
    "luhn_ok",
    "pretty",
    "read_jsonl",
    "seeded_rng",
    "sha256_hex",
    "stable_id",
    "to_major",
    "to_minor",
]

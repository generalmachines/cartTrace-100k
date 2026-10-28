"""The test keys a sandbox store holds, derived from its envpack id: the same pack gives the same keys on every machine.

Why derived. A store's secrets (the API key its ACP clients present, the key it checks ACP request signatures with, the
HMAC key that binds an MPP challenge to its amount, the signing keys of AP2's parties) must come out the same on every
run, or a replay would issue other challenge ids and signatures and reach another digest. They must also differ
between packs, so a credential minted against one world proves nothing in another. So every secret is HMAC-SHA256 of
a fixed root over the pack id and a purpose.

They are test keys: anyone holding the pack id can derive them, which is what lets a harness (the agent side's
wallet, a test) sign as the user or verify as the merchant without a key exchange. Nothing outside the sandbox trusts
them, and none of them is a card credential.
"""

from __future__ import annotations

import hashlib
import hmac

from carttrace.core.canon import digest

ROOT = b"mc.sandbox.test-keys.v1"
"""The fixed root every test secret hangs from. Changing it changes every derived key, so it is versioned."""
_LETTERS = str.maketrans("0123456789abcdef", "abcdefghijklmnop")


def opaque_key(*parts: object, length: int = 24) -> str:
    """A digest of `parts` spelled in the letters a to p, for the idempotency keys surfaces derive. The core refuses
    a command holding a digit run that passes the Luhn check (`carttrace.sandbox.core.guard`) and scans `idempotency_key`,
    so a hex digest there would, now and then, read as a card number; one with no digits never can."""
    return digest(list(parts), length).translate(_LETTERS)


def secret(pack_id: str, purpose: str) -> bytes:
    """32 bytes for one purpose (`acp.api_key`, `acp.signing`, `mpp.challenge`, `ap2.merchant`...) of one pack."""
    return hmac.new(ROOT, f"{pack_id}|{purpose}".encode(), hashlib.sha256).digest()


def secret_text(pack_id: str, purpose: str, *, prefix: str) -> str:
    """A secret as printable text (`acp_test_<hex>`), for keys that travel in headers."""
    return f"{prefix}{secret(pack_id, purpose).hex()}"

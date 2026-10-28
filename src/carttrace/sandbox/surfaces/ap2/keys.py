"""AP2's parties' test keys, derived from the envpack id, and the compact JWS and hashes their mandates are signed with.

Why derived and deterministic. AP2's mandates are signed: the merchant signs the cart (v0.1.0 `merchant_authorization`,
v0.2.0 `checkout_jwt`), the user signs the payment authorization (v0.1.0 `user_authorization`, v0.2.0 the closed
checkout and payment mandates). The sandbox must verify those signatures, and a replay must produce byte-identical
signatures, or the event log would differ. So each party's key is an ECDSA P-256 key whose private scalar is derived
from the pack id and the party's role (`carttrace.sandbox.surfaces.keys`), and every signature is RFC 6979 deterministic
ECDSA. P-256 with ES256 is what AP2 v0.2.0's JWK type (`types/jwk.json`) prescribes.

What the sandbox reads a signed mandate as (AP2 publishes the claims, not a byte format; written down here so a wallet
built elsewhere interoperates):
- a compact JWS, `base64url(header) "." base64url(payload) "." base64url(R || S)`, header
  `{"alg": "ES256", "kid": "<pack id>#ap2-<role>", "typ": ...}`; the payload is the JSON the mandate defines;
- a hash is `base64url(sha256(RFC 8785 JCS of the object))`, unpadded, and the hash of a JWS is the sha256 of its
  ASCII text (AP2 v0.2.0 `checkout_hash`);
- selective disclosure (SD-JWT) is not modeled: a mandate is disclosed whole.

These are test keys: anyone with the pack id can derive them, which is what lets the agent side's wallet sign as the
shopper without a key exchange. Nothing outside the sandbox trusts them.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature

from carttrace.core.canon import dumps_bytes, loads
from carttrace.sandbox.surfaces.keys import secret

Role = Literal["merchant", "shopper", "credentials_provider", "payment_processor"]
ROLES: tuple[Role, ...] = ("merchant", "shopper", "credentials_provider", "payment_processor")
"""AP2's parties (v0.1.0 README roles): the sandbox is the merchant and its payment processor; the shopper's key signs
for the user's device, the credentials provider's for the wallet."""
_ORDER = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
"""The order of the P-256 group: a private scalar is in [1, order - 1]."""


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def unb64url(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def mandate_hash(value: Any) -> str:
    """base64url sha256 of the RFC 8785 canonical JSON of `value`: what a mandate's hashes are over."""
    return b64url(hashlib.sha256(rfc8785.dumps(value)).digest())


def text_hash(text: str) -> str:
    """base64url sha256 of a string's ASCII bytes: the hash of a JWS (AP2 v0.2.0 `checkout_hash`)."""
    return b64url(hashlib.sha256(text.encode("ascii")).digest())


class BadSignature(ValueError):
    """A JWS that does not parse, names another key, or does not verify."""


@dataclass(frozen=True, slots=True)
class PartyKey:
    """One party's ES256 key."""

    kid: str
    private: ec.EllipticCurvePrivateKey

    def sign(self, payload: Mapping[str, Any], *, typ: str = "JWT") -> str:
        header = {"alg": "ES256", "kid": self.kid, "typ": typ}
        signing_input = f"{b64url(dumps_bytes(header))}.{b64url(dumps_bytes(dict(payload)))}"
        der = self.private.sign(signing_input.encode("ascii"), ec.ECDSA(hashes.SHA256(), deterministic_signing=True))
        r, s = decode_dss_signature(der)
        return f"{signing_input}.{b64url(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))}"

    def verify(self, token: str) -> dict[str, Any]:
        """The payload of a JWS this key signed. Raises `BadSignature` otherwise."""
        header, payload = read_jws(token)
        if header.get("alg") != "ES256" or header.get("kid") != self.kid:
            raise BadSignature("the JWS names another algorithm or key")
        head, body, sig = token.split(".")
        raw = unb64url(sig)
        if len(raw) != 64:
            raise BadSignature("an ES256 signature is 64 bytes")
        der = encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
        try:
            self.private.public_key().verify(der, f"{head}.{body}".encode("ascii"), ec.ECDSA(hashes.SHA256()))
        except InvalidSignature:
            raise BadSignature("the signature does not verify") from None
        return payload

    def jwk(self) -> dict[str, str]:
        """The public key as AP2 v0.2.0's JWK type (`types/jwk.json`) writes it."""
        numbers = self.private.public_key().public_numbers()
        return {
            "kty": "EC",
            "crv": "P-256",
            "x": b64url(numbers.x.to_bytes(32, "big")),
            "y": b64url(numbers.y.to_bytes(32, "big")),
            "use": "sig",
            "alg": "ES256",
            "kid": self.kid,
        }


def read_jws(token: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(header, payload) of a compact JWS, without verifying it. Raises `BadSignature` for one that does not parse."""
    if not isinstance(token, str) or token.count(".") != 2 or len(token) > 64 * 1024:
        raise BadSignature("a compact JWS has three base64url parts")
    head, body, _ = token.split(".")
    try:
        header, payload = loads(unb64url(head)), loads(unb64url(body))
    except ValueError:
        raise BadSignature("the JWS header or payload is not JSON") from None
    if not isinstance(header, dict) or not isinstance(payload, dict):
        raise BadSignature("the JWS header and payload must be JSON objects")
    return header, payload


@cache
def party_key(pack_id: str, role: Role) -> PartyKey:
    """The ES256 key of one AP2 party in one pack's world."""
    scalar = int.from_bytes(secret(pack_id, f"ap2.{role}"), "big") % (_ORDER - 1) + 1
    return PartyKey(kid=f"{pack_id}#ap2-{role}", private=ec.derive_private_key(scalar, ec.SECP256R1()))

"""The ACP store's test credentials: the API key its clients present, and the key it checks request signatures with.

What ACP says. Every request carries `Authorization: Bearer <api key>` (required in every release). `Signature` and
`Timestamp` are optional headers in every vendored release; the OpenAPI calls the signature an HMAC over the request
("detached signature over canonical JSON") and does not fix the construction. No vendored release requires one, so a
store checks a signature when a request carries one, and demands one only when its dialect says signatures are
required.

The sandbox's construction, which its client surface and any other client must follow:
`Signature: base64url(HMAC-SHA256(key, Timestamp "\\n" METHOD "\\n" path "\\n" body))`, unpadded, with `Timestamp` an
RFC 3339 instant and `body` the exact bytes sent (empty for a GET). The timestamp is bound into the signature but not
checked for freshness: the sandbox runs on a logical clock it does not share with the agent, and retries are made safe
by `Idempotency-Key`, not by a time window.

Both keys derive from the envpack id (`carttrace.sandbox.surfaces.keys`), so a harness holding the pack can authenticate as
the store's platform client without a key exchange.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass

from carttrace.sandbox.surfaces.keys import secret, secret_text

API_KEY_PREFIX = "acp_test_"


@dataclass(frozen=True, slots=True)
class AcpKeys:
    """One store's ACP test credentials."""

    api_key: str
    signing_key: bytes

    @classmethod
    def for_pack(cls, pack_id: str) -> AcpKeys:
        return cls(
            api_key=secret_text(pack_id, "acp.api_key", prefix=API_KEY_PREFIX),
            signing_key=secret(pack_id, "acp.signing"),
        )

    def sign(self, *, method: str, path: str, timestamp: str, body: bytes) -> str:
        """The `Signature` header value for one request."""
        message = b"\n".join([timestamp.encode(), method.upper().encode(), path.encode(), body])
        mac = hmac.new(self.signing_key, message, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(mac).decode("ascii").rstrip("=")

    def verify(self, signature: str, *, method: str, path: str, timestamp: str, body: bytes) -> bool:
        expected = self.sign(method=method, path=path, timestamp=timestamp, body=body)
        return hmac.compare_digest(expected.encode(), signature.strip().encode())

    def bearer_ok(self, authorization: str | None) -> bool:
        """True when `Authorization` is `Bearer <this store's API key>`."""
        if not authorization:
            return False
        scheme, _, value = authorization.strip().partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(value.strip().encode(), self.api_key.encode())

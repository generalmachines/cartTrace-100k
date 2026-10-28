"""The sandbox's MPP payment method, `sandbox`: a charge paid with an opaque test credential, built for the `mpp` SDK.

Why a method of its own. The SDK (pympp 0.11.0) ships `tempo` (an on-chain transfer) and `stripe` (a Stripe shared
payment token), and the specs vendored with it add `card` (a Visa network token encrypted to the merchant's RSA key).
Each settles through a real network, and the card method carries a network token that is itself card-number shaped.
The sandbox pays through its own PSP simulator, which keys behaviour on opaque test tokens (`tok_visa_ok`, or a `vt-`
token its vault issued), so it registers a method that says exactly that and nothing more: a simulated method in the
SDK's own terms (`mpp.server.method.Method` and `mpp.runtime.Method`), with no money and no chain.

The method, as a payment method specification must state it (MPP core, "Payment Method Specifications"):
- identifier `sandbox`; intent `charge` (draft-payment-intent-charge-00);
- request: the charge intent's shared fields, `amount` (minor units, as a string), `currency` (ISO 4217, lowercase),
  `recipient` (the store's domain), `description`, `externalId` (the checkout id); `methodDetails.livemode` is false;
- payload: `{"token": "<opaque test credential>"}`; a card number is never accepted;
- verification: the challenge binding (HMAC, the core spec's recommended construction, keyed by the store's test key),
  expiry against the store's logical clock, single use per challenge, amount and currency equal to the checkout's
  total, the token known to the PSP; settlement: authorization and capture in the core, or its decline or 3-D Secure
  step;
- receipt: `{status: success, method: sandbox, timestamp, reference: <payment id>, externalId: <checkout id>}`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mpp import Challenge, Credential

METHOD = "sandbox"
INTENT = "charge"
CHALLENGE_TTL_MS = 5 * 60 * 1000
"""How long a challenge stays payable, in the store's logical milliseconds: the core spec's five-minute default."""


def payload(token: str) -> dict[str, str]:
    """The credential payload of the sandbox method."""
    return {"token": token}


class SandboxMethod:
    """The client side of the `sandbox` method, for the SDK's runtime (`mpp.client.PaymentTransport`) or by hand: it
    answers a sandbox charge challenge with the opaque test credential it holds."""

    name = METHOD
    intents: Mapping[str, Any] = {INTENT: None}
    """The intents it pays (the SDK's runtime reads the names)."""

    def __init__(self, token: str) -> None:
        self.token = token

    async def create_credential(self, challenge: Challenge) -> Credential:
        return credential(challenge, self.token)


def credential(challenge: Challenge, token: str) -> Credential:
    """The credential that answers `challenge` with `token`: the challenge echoed unchanged, and the payload."""
    return Credential(challenge=challenge.to_echo(), payload=payload(token))

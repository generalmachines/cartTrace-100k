"""MPP: a UCP checkout paid over HTTP 402 (D66, the `mpp` payment rail).

What differs from UCP, and why the agents still run the same loop:
- The catalog, the cart and the checkout are the store's UCP tools, unchanged; only UCP's own complete_checkout is not
  offered, so the payment goes over MPP (`carttrace.sandbox.surfaces.mpp`).
- Paying takes two calls. `request_payment(checkout_id)` asks the store for its 402 challenge (the amount in minor
  units, the currency, the recipient, an expiry), and `pay(challenge_id, token)` answers it with the wallet's opaque
  test credential, which the surface builds into MPP's `Authorization: Payment` credential. The dialect's
  `prepare_payment` is the first call and `complete_checkout` the second, so the reference agent asks for the
  challenge just before it pays, and the harness guards `pay` as it guards every paying call: only a credential the
  wallet issued for the checkout's approved total goes through.
- The wallet's credential is the persona's card token, issued against the shopper's approval of exactly the checkout's
  total (`ChallengeCredentialIssuer`); the challenge's amount must be that total, or the store refuses it.
- The paid reply is MPP's checkout resource with the order and the `Payment-Receipt`; a 3-D Secure step answers 402
  `payment-action-required`, after which the UCP checkout carries the bank's challenge as it does on UCP.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from carttrace.agents.checkout import CHECKOUT_TOOLS, checkout_doc
from carttrace.agents.dialects.base import Dialect
from carttrace.agents.tools import payload_of
from carttrace.agents.wallet import (
    Consent,
    Credential,
    CredentialKind,
    CredentialRequest,
    PaymentChallengeRequest,
    Persona,
    Refusal,
)
from carttrace.core.money import Money
from carttrace.protocols.tools import ToolResult

REQUEST: Final = "request_payment"
PAY: Final = "pay"
STATUS: Final = "get_payment"
NAMESPACE: Final = "mpp.sandbox"
"""What the wallet calls MPP's `sandbox` payment method, the one method the sandbox's MPP endpoint takes."""
METHOD_VERSION: Final = "pympp-0.11.0"


@dataclass(frozen=True, slots=True)
class MppDialect(Dialect):
    """UCP's catalog, cart and checkout, paid over MPP's HTTP 402."""

    complete_checkout: str = PAY
    prepare_payment: str | None = REQUEST
    checkout_tools: frozenset[str] = field(default=CHECKOUT_TOOLS | {PAY, STATUS})
    payment_key: str = "token"

    @property
    def name(self) -> str:
        return "mpp"

    def offered(self, tools: Collection[str]) -> bool:
        return {REQUEST, PAY, self.create} <= set(tools)

    def checkout_doc(self, payload: Any) -> Mapping[str, Any] | None:
        """A UCP checkout, or MPP's checkout resource read as one: its id, status, total and order."""
        doc = checkout_doc(payload)
        if doc is not None:
            return doc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("checkout_id"), str):
            return None
        currency = payload.get("currency")
        out: dict[str, Any] = {
            "id": payload["checkout_id"],
            "status": payload.get("status"),
            "line_items": [],
            "currency": currency.upper() if isinstance(currency, str) else None,
        }
        total = payload.get("total")
        if isinstance(total, int) and not isinstance(total, bool):
            out["totals"] = [{"type": "total", "amount": total}]
        order = payload.get("order")
        if isinstance(order, Mapping) and isinstance(order.get("id"), str):
            number = order.get("number")
            out["order"] = {"id": order["id"], **({"label": f"#{number}"} if number is not None else {})}
        if isinstance(payload.get("receipt"), Mapping):
            out["receipt"] = dict(payload["receipt"])
        return out

    def handlers_of(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        """The one way the store takes payment on this rail: MPP's sandbox method."""
        del doc
        return {NAMESPACE: [{"id": "sandbox", "version": METHOD_VERSION}]}

    def complete_target(self, arguments: Mapping[str, Any]) -> str | None:
        """`pay` names a challenge, not a checkout: the harness takes the store's latest checkout."""
        del arguments
        return None

    def tokens_of(self, arguments: Mapping[str, Any]) -> set[str]:
        token = arguments.get("token")
        return {token} if isinstance(token, str) and token else set()

    def credential_request(
        self,
        checkout_id: str,
        handler_id: str,
        namespace: str,
        amount: Money,
        doc: Mapping[str, Any],
        *,
        now_ms: int,
    ) -> CredentialRequest:
        del handler_id, namespace, doc, now_ms
        return PaymentChallengeRequest(checkout_id, {}, amount)

    def credential_answer(self, credential: Credential, *, token: str | None = None) -> dict[str, Any]:
        return {
            self.payment_key: token or credential.token,
            "amount": {"amount": credential.amount.amount, "currency": credential.amount.currency},
            "use": f"call {REQUEST} with this checkout_id for the store's HTTP 402 challenge, check that its amount is "
            f"this total, then call {PAY} with the challenge's id and this token",
        }

    def prepared(self, result: ToolResult) -> dict[str, Any] | None:
        data = payload_of(result)
        challenges = data.get("challenges") if isinstance(data, Mapping) else None
        for challenge in challenges if isinstance(challenges, list) else ():
            if isinstance(challenge, Mapping) and isinstance(challenge.get("id"), str):
                return {"challenge_id": challenge["id"]}
        return None

    def complete_arguments(self, checkout_id: str, answer: Mapping[str, Any]) -> dict[str, Any]:
        del checkout_id
        return {"challenge_id": answer.get("challenge_id"), "token": answer[self.payment_key]}

    def order_label(self, doc: Mapping[str, Any]) -> str | None:
        order = doc.get("order")
        label = order.get("label") if isinstance(order, Mapping) else None
        return label if isinstance(label, str) and label else None


class ChallengeCredentialIssuer:
    """MPP credentials: the persona's card token, issued for exactly the approved total of one checkout, for the
    surface to build into an answer to the store's 402 challenge. The wallet's consent gate runs first."""

    @property
    def kind(self) -> CredentialKind:
        return "mpp_credential"

    def refusal(self, persona: Persona, request: CredentialRequest) -> Refusal | None:
        if not isinstance(request, PaymentChallengeRequest):
            return Refusal("credential_kind_mismatch", f"a {request.kind} request reached the MPP credential issuer")
        if _card(persona) is None:
            kinds = ", ".join(i.kind for i in persona.instruments)
            return Refusal(
                "handler_unsupported",
                f"the shopper's wallet holds no card for MPP's sandbox method (it holds: {kinds})",
                hint="this persona cannot pay here",
            )
        return None

    def issue(self, persona: Persona, request: CredentialRequest, consent: Consent, *, sequence: int) -> Credential:
        assert isinstance(request, PaymentChallengeRequest)
        card = _card(persona)
        assert card is not None
        return Credential(
            kind=self.kind,
            checkout_id=request.checkout_id,
            amount=request.amount,
            token=card.token,
            instrument=card.kind,
            instrument_id=f"wallet-instrument-{sequence}",
            consent=consent.seq,
            handler_id="sandbox",
            brand=card.brand,
        )


def _card(persona: Persona) -> Any:
    """The persona's card, else its Google Pay token: the sandbox's PSP takes any opaque test credential through the
    payment-request handler both MPP and AP2 settle through, so a wallet that holds only Google Pay still pays."""
    return next((i for i in persona.instruments if i.kind == "card"), None) or next(
        (i for i in persona.instruments if i.kind == "google_pay"), None
    )

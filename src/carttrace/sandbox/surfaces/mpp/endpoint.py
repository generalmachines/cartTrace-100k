"""MPP in the sandbox: a checkout's payment behind HTTP 402, as pympp 0.11.0 and the specs vendored with it define it.

Routes, under `/api/mpp`:
- `POST /checkouts/{id}/complete`, the paid resource. Without a credential the store answers 402 with a fresh
  `WWW-Authenticate: Payment` challenge for the checkout's total and an RFC 9457 `payment-required` problem; with an
  `Authorization: Payment` credential that answers a challenge it issued, it captures the payment in the core and
  answers 200 with the order and a `Payment-Receipt`.
- `GET /checkouts/{id}`, the checkout's payment status, with the receipt once it is paid (after a 3-D Secure step the
  buyer completed at the challenge page, the agent learns the outcome here).

Every challenge is a core payment request (`request_payment`): issuing one is the only change an unpaid request makes
(MPP core, "Idempotency and Side Effects"), and it is in the event log. Its id is the HMAC binding the core spec
recommends, keyed by the store's test key, over the realm, method, intent, request and expiry, so a credential whose
echo was altered (another amount, another checkout) is not a challenge this store issued. The request carries the
amount the store asked for; the core refuses a payment that answers it for any other total, and a request already
answered, so a replayed credential is refused whatever the transport did.

What the store answers (MPP core, "Status Codes"):
- no credential, or a payable checkout and a credential that fails: 402 with a fresh challenge and the problem
  (`malformed-credential`, `invalid-challenge` for an unknown, altered or already used challenge, `payment-expired`,
  `payment-insufficient` or `verification-failed` for a wrong amount, `invalid-payload`, `verification-failed` for an
  issuer's decline, `payment-action-required` when the issuer asks for 3-D Secure);
- a challenge for another payment method: 400 `method-unsupported`;
- a checkout that cannot be paid now (not ready, already completed): 409, since no challenge can be built for it;
- a checkout that is not the caller's: 404; a session whose trust tier may not complete: 403.

Time is the store's logical clock: a challenge expires `CHALLENGE_TTL_MS` after it was issued, in the clock the event
log stamps, so a replay expires the same challenges. No receipt is ever sent on an error response.
"""

from __future__ import annotations

import base64
import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from mpp import Challenge, Credential, Receipt, generate_challenge_id
from mpp._parsing import ParseError
from mpp.errors import (
    InvalidChallengeError,
    InvalidPayloadError,
    MalformedCredentialError,
    PaymentActionRequiredError,
    PaymentError,
    PaymentExpiredError,
    PaymentInsufficientError,
    PaymentMethodUnsupportedError,
    PaymentRequiredError,
    VerificationFailedError,
)
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from carttrace.core.canon import dumps_bytes, loads
from carttrace.sandbox.core.config import REQUESTED_NAMESPACE
from carttrace.sandbox.core.errors import BusinessRule, CoreError, IllegalTransition, InvalidCommand, NotFound
from carttrace.sandbox.core.guard import find_card_number
from carttrace.sandbox.core.state import Checkout
from carttrace.sandbox.presented import presented
from carttrace.sandbox.surfaces.access import DEFAULT_SESSION, AccessRefused, named_session
from carttrace.sandbox.surfaces.errors import UnsupportedVersion
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.keys import opaque_key, secret_text
from carttrace.sandbox.surfaces.mpp.method import CHALLENGE_TTL_MS, INTENT
from carttrace.sandbox.surfaces.ucp.messages import fault_for

MPP_PATH = "/api/mpp"
SERVED = ("pympp-0.11.0",)
"""MPP versions with a server here: the pympp release the specs were vendored for."""
PROBLEM = "application/problem+json"
JSON = "application/json"
TOKEN_PREFIXES = ("tok_", "vt-")
PAYABLE = ("ready_for_complete", "requires_escalation")
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def served_versions() -> list[str]:
    return list(SERVED)


def check_version(version: str | None) -> str:
    if version not in SERVED:
        raise UnsupportedVersion(
            f"the sandbox has no MPP server for {version!r}", hint=f"serve a profile at one of {list(SERVED)}"
        )
    return str(version)


def iso(ms: int) -> str:
    return (EPOCH + timedelta(milliseconds=ms)).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Rejected(Exception):
    """A credential refused before it reached the core: the problem to answer with, and the typed error to log."""

    def __init__(self, problem: PaymentError, code: str = "invalid_challenge", *, business: bool = False) -> None:
        super().__init__(str(problem))
        self.problem = problem
        kind = BusinessRule if business else InvalidCommand
        self.error: CoreError = kind(code=code, message=str(problem))


class Unsupported(Exception):
    """A credential for a payment method this store does not take: 400, with no challenge."""

    def __init__(self, problem: PaymentError, challenge_id: str) -> None:
        super().__init__(str(problem))
        self.problem = problem
        self.challenge_id = challenge_id


@dataclass(frozen=True, slots=True)
class Answer:
    """One HTTP answer of the MPP endpoint."""

    status: int
    body: Any
    headers: Mapping[str, str]
    media_type: str = JSON

    def response(self) -> Response:
        body = dumps_bytes(self.body)
        return Response(body, status_code=self.status, media_type=self.media_type, headers=dict(self.headers))


class MppServer:
    """The MPP endpoint of one store."""

    def __init__(self, host: SandboxHost) -> None:
        self.host = host
        self.version = check_version(host.require("mpp"))
        self.realm = presented(host.pack).realm
        self.method = presented(host.pack).mpp_method
        self.secret = secret_text(host.pack.id, "mpp.challenge", prefix="")
        """The HMAC key that binds a challenge to its parameters. A test key; it never leaves the store."""

    def routes(self) -> list[Route]:
        return [
            Route(f"{MPP_PATH}/checkouts/{{id}}/complete", self.complete, methods=["POST"]),
            Route(f"{MPP_PATH}/checkouts/{{id}}", self.status, methods=["GET"]),
        ]

    # -- routes --------------------------------------------------------------------------------------------------------

    async def complete(self, request: Request) -> Response:
        try:
            session = named_session(request) or DEFAULT_SESSION
        except AccessRefused as e:
            return _problem_response(e.status, "forbidden", e.message)
        ck = self._checkout(str(request.path_params["id"]), session)
        if ck is None:
            self.host.reject("complete_checkout", NotFound(code="checkout_not_found", message="Checkout not found"))
            return _problem_response(404, "not-found", "No checkout with this id")
        header = _payment_scheme(request.headers.get("authorization"))
        base = str(request.base_url).rstrip("/")
        if header is None:
            return self._challenge(ck, base, PaymentRequiredError(self.realm, "a payment for this checkout")).response()
        return self._pay(ck, header, base).response()

    async def status(self, request: Request) -> Response:
        try:
            session = named_session(request) or DEFAULT_SESSION
        except AccessRefused as e:
            return _problem_response(e.status, "forbidden", e.message)
        ck = self._checkout(str(request.path_params["id"]), session)
        if ck is None:
            self.host.reject("read", NotFound(code="checkout_not_found", message="Checkout not found"))
            return _problem_response(404, "not-found", "No checkout with this id")
        out = self.host.execute({"type": "read", "session": session, "target": "checkout", "id": ck.id})
        fault = fault_for(out.error, tier=self.host.tier(session))
        if fault is not None:
            return _problem_response(fault.status, fault.reason, fault.message)
        base = str(request.base_url).rstrip("/")
        receipt = self._receipt(self.host.state.checkouts[ck.id])
        headers = {"Payment-Receipt": receipt.to_payment_receipt()} if receipt is not None else {}
        return Answer(200, self._resource(ck.id, base, receipt), headers).response()

    # -- the 402 flow --------------------------------------------------------------------------------------------------

    def _checkout(self, checkout_id: str, session: str) -> Checkout | None:
        ck = self.host.state.checkouts.get(checkout_id)
        return ck if ck is not None and ck.session == session else None

    def _request(self, ck: Checkout) -> dict[str, Any]:
        """The charge intent's request for a checkout's total (`method` docstring)."""
        lines = sum(ln.quantity for ln in ck.lines)
        return {
            "amount": str(ck.totals.total),
            "currency": self.host.state.currency.lower(),
            "recipient": self.realm,
            "description": f"{lines} item(s) from {self.realm}",
            "externalId": ck.id,
            "methodDetails": {"livemode": False},
        }

    def _challenge(self, ck: Checkout, base: str, problem: PaymentError, *, retry: bool = False) -> Answer:
        """A fresh challenge for the checkout's total, issued as a core payment request, with `problem`; or 409 when
        the checkout cannot be paid now."""
        if ck.status not in PAYABLE or ck.totals.fulfillment is None:
            self.host.reject(
                "request_payment",
                IllegalTransition(code="checkout_not_ready", message=f"checkout is {ck.status}", field="status"),
            )
            return self._not_payable(ck, base, problem if retry else None)
        clock = self.host.state.clock
        request = self._request(ck)
        challenge = Challenge.create(
            secret_key=self.secret,
            realm=self.realm,
            method=self.method,
            intent=INTENT,
            request=request,
            expires=iso(clock + CHALLENGE_TTL_MS),
            description=str(request["description"]),
        )
        out = self.host.execute(
            {
                "type": "request_payment",
                "actor": "merchant",
                "session": ck.session,
                "request_id": challenge.id,
                "checkout_id": ck.id,
                "protocol": "mpp",
                "amount": ck.totals.total,
                "currency": self.host.state.currency,
                "expires_at": clock + CHALLENGE_TTL_MS,
            }
        )
        if not out.ok:
            return self._not_payable(ck, base, problem if retry else None)
        headers = {"WWW-Authenticate": challenge.to_www_authenticate(self.realm), "Cache-Control": "no-store"}
        return Answer(402, problem.to_problem_details(challenge.id), headers, PROBLEM)

    def _not_payable(self, ck: Checkout, base: str, problem: PaymentError | None) -> Answer:
        state = self.host.state.checkouts[ck.id]
        detail = (
            f"The checkout is {state.status}; nothing is left to pay"
            if state.status in ("completed", "canceled")
            else "The checkout cannot be paid yet: it needs a buyer email, a destination and a shipping option"
        )
        body: dict[str, Any] = {
            "type": "about:blank",
            "title": "Conflict",
            "status": 409,
            "detail": detail,
            "checkout": self._resource(ck.id, base, self._receipt(state)),
        }
        if problem is not None:
            body["payment"] = problem.to_problem_details()
        return Answer(409, body, {}, PROBLEM)

    def _pay(self, ck: Checkout, header: str, base: str) -> Answer:
        try:
            challenge_id, token = self._verify(ck, header)
        except Unsupported as u:
            self.host.reject("complete_checkout", InvalidCommand(code="method_unsupported", message=str(u.problem)))
            return Answer(400, u.problem.to_problem_details(u.challenge_id), {}, PROBLEM)
        except Rejected as r:
            self.host.reject("complete_checkout", r.error)
            return self._challenge(ck, base, r.problem, retry=True)
        return self._settle(ck, challenge_id, token, base)

    def _verify(self, ck: Checkout, header: str) -> tuple[str, str]:
        """(challenge id, token) of a credential that answers an open challenge this store issued for this checkout's
        current total, in the order the specs ask: binding and expiry before the payment proof. Raises `Rejected` or
        `Unsupported`."""
        try:
            cred = Credential.from_authorization(header)
        except ParseError as e:
            raise Rejected(MalformedCredentialError(str(e)), "malformed_credential") from None
        echo = cred.challenge
        if echo.method != self.method:
            raise Unsupported(PaymentMethodUnsupportedError(echo.method), echo.id)
        request = self._authentic(cred)
        if request is None:
            raise Rejected(InvalidChallengeError(echo.id, "it was not issued by this store, or was altered"))
        if request.get("externalId") != ck.id:
            raise Rejected(InvalidChallengeError(echo.id, "it was issued for another checkout"))
        expires = _epoch_ms(echo.expires)
        if expires is None or expires < self.host.state.clock:
            raise Rejected(PaymentExpiredError(echo.expires), "payment_expired", business=True)
        issued = self.host.state.payment_requests.get(echo.id)
        if issued is None or issued.status != "open":
            reason = "it is unknown to this store" if issued is None else "it was already used"
            raise Rejected(InvalidChallengeError(echo.id, reason), business=True)
        wrong = self._wrong_amount(ck, request)
        if wrong is not None:
            raise Rejected(wrong, "payment_amount_mismatch", business=True)
        body = dict(cred.payload) if isinstance(cred.payload, Mapping) else {}
        if find_card_number(body) is not None:
            problem = InvalidPayloadError("a card number was sent; this store takes payment tokens only")
            raise Rejected(problem, "card_number_refused")
        token = body.get("token")
        if not isinstance(token, str) or not token.startswith(TOKEN_PREFIXES):
            problem = InvalidPayloadError("payload.token takes a credential token from the shopper's wallet")
            raise Rejected(problem, "invalid_payload")
        return echo.id, token

    def _authentic(self, cred: Credential) -> dict[str, Any] | None:
        """The echoed request when the echo is a challenge this store issued, unaltered; None otherwise."""
        echo = cred.challenge
        if echo.realm != self.realm or echo.intent != INTENT:
            return None
        request = _b64_object(echo.request)
        opaque = _b64_object(echo.opaque) if echo.opaque else None
        if request is None or (echo.opaque and opaque is None):
            return None
        try:
            expected = generate_challenge_id(
                secret_key=self.secret,
                realm=echo.realm,
                method=echo.method,
                intent=echo.intent,
                request=request,
                expires=echo.expires,
                digest=echo.digest,
                opaque=opaque,
                header=echo.header,
            )
        except (TypeError, ValueError):
            return None
        return request if _same(expected, echo.id) else None

    def _wrong_amount(self, ck: Checkout, request: Mapping[str, Any]) -> PaymentError | None:
        """The checkout's total and currency against the challenge's: the total may have changed since."""
        currency = self.host.state.currency.lower()
        if request.get("currency") != currency or request.get("recipient") != self.realm:
            return VerificationFailedError("the challenge's currency or recipient is not this checkout's")
        try:
            amount = int(str(request.get("amount")))
        except ValueError:
            return VerificationFailedError("the challenge's amount is not an amount")
        total = ck.totals.total
        if amount < total:
            return PaymentInsufficientError(f"the challenge was for {amount}; the checkout total is now {total}")
        if amount > total:
            return VerificationFailedError(f"the challenge was for {amount}; the checkout total is now {total}")
        return None

    def _settle(self, ck: Checkout, challenge_id: str, token: str, base: str) -> Answer:
        out = self.host.execute(
            {
                "type": "complete_checkout",
                "session": ck.session,
                "checkout_id": ck.id,
                "idempotency_key": f"mpp|{opaque_key(challenge_id)}",
                "payment": {"handler": REQUESTED_NAMESPACE, "token": token, "request_id": challenge_id},
            }
        )
        fault = fault_for(out.error, tier=self.host.tier(ck.session))
        if fault is not None:
            status = 403 if fault.status in (401, 403) else fault.status
            headers = {"Retry-After": str(fault.retry_after)} if fault.retry_after is not None else {}
            return Answer(status, _problem(status, fault.reason, fault.message), headers, PROBLEM)
        after = self.host.state.checkouts[ck.id]
        if after.status == "completed":
            receipt = self._receipt(after)
            headers = {"Payment-Receipt": receipt.to_payment_receipt()} if receipt is not None else {}
            return Answer(200, self._resource(ck.id, base, receipt), headers)
        if out.refs.get("action_required"):
            page = f"{base}/3ds/{out.refs.get('challenge_id')}"
            problem = PaymentActionRequiredError(f"the buyer's bank asks for a 3-D Secure challenge at {page}")
            problem.details = {"actionUrl": page, "checkoutId": ck.id}
            return self._challenge(after, base, problem, retry=True)
        err = out.error or {}
        detail = str(err.get("message") or "the payment was not taken")
        return self._challenge(after, base, VerificationFailedError(detail), retry=True)

    def _receipt(self, ck: Checkout) -> Receipt | None:
        """The receipt of a checkout paid through MPP: the payment's id as reference, the order time as timestamp."""
        if ck.status != "completed" or ck.payment_id is None:
            return None
        pay = self.host.state.payments.get(ck.payment_id)
        req = self.host.state.payment_requests.get(pay.request_id or "") if pay is not None else None
        order = self.host.state.orders.get(ck.order_id or "")
        if pay is None or req is None or req.protocol != "mpp" or order is None:
            return None
        return Receipt(
            status="success",
            timestamp=EPOCH + timedelta(milliseconds=order.placed_at),
            reference=pay.id,
            method=self.method,
            external_id=ck.id,
        )

    def _resource(self, checkout_id: str, base: str, receipt: Receipt | None) -> dict[str, Any]:
        """The checkout as this endpoint shows it: its status and total, its order once placed, and the receipt."""
        state = self.host.state
        ck = state.checkouts[checkout_id]
        out: dict[str, Any] = {
            "checkout_id": ck.id,
            "status": ck.status,
            "total": ck.totals.total,
            "currency": state.currency.lower(),
        }
        order = state.orders.get(ck.order_id or "")
        if order is not None:
            out["order"] = {"id": order.id, "number": order.number, "permalink_url": f"{base}/orders/{order.id}"}
        if receipt is not None:
            out["receipt"] = {
                "status": receipt.status,
                "method": receipt.method,
                "timestamp": iso(state.orders[str(ck.order_id)].placed_at),
                "reference": receipt.reference,
                "externalId": receipt.external_id,
            }
        return out


def _payment_scheme(header: str | None) -> str | None:
    """The `Payment` credential among an Authorization header's schemes (RFC 9110 allows several)."""
    for part in (header or "").split(","):
        if part.strip().lower().startswith("payment "):
            return part.strip()
    return None


def _b64_object(value: str | None) -> dict[str, Any] | None:
    if not value or len(value) > 16 * 1024:
        return None
    try:
        decoded = loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))
    except ValueError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _same(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def _epoch_ms(value: str | None) -> int | None:
    if not value:
        return None
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return None if when.tzinfo is None else (when - EPOCH) // timedelta(milliseconds=1)


def _problem(status: int, slug: str, detail: str) -> dict[str, Any]:
    """An RFC 9457 problem under MPP's problem base, for answers that are not payment errors (404, 403, 429)."""
    title = slug.replace("-", " ").replace("_", " ").capitalize()
    return {"type": f"https://paymentauth.org/problems/{slug}", "title": title, "status": status, "detail": detail}


def _problem_response(status: int, slug: str, detail: str) -> Response:
    return Answer(status, _problem(status, slug, detail), {}, PROBLEM).response()

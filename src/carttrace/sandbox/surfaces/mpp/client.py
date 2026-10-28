"""The MPP surface as an agent holds it: a `ToolSurface` (kind `mpp`) on a sandbox store's 402 payment endpoint.

Three tools, so the agent sees what it is asked to pay before it pays, as MPP requires of a client ("Clients MUST verify
before authorizing payment: the amount, the recipient, the currency, the validity window"):
- `request_payment(checkout_id)`: ask to pay for a checkout. The store answers 402 with its challenge, which this tool
  returns decoded (id, amount in minor units, currency, recipient, description, expiry) as a normal result.
- `pay(challenge_id, token)`: answer a challenge this surface received with an opaque test credential. The surface
  builds the credential with the `sandbox` method (`mpp.method`), echoing the challenge exactly, and retries the paid
  request with `Authorization: Payment ...`. A 200 returns the order and the decoded `Payment-Receipt`; a 402 is an
  error result with the problem type as `error_code`, and the store's fresh challenge is remembered for the next try.
- `get_payment(checkout_id)`: the checkout's payment status, with the receipt once paid (after the buyer's 3-D Secure
  step, for instance).

The surface keeps the challenges it was given (client-side state, as any MPP client does); the store keeps nothing but
its payment requests in the core.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar
from urllib.parse import quote

import httpx
from mpp import Challenge, Receipt
from mpp._parsing import ParseError

from carttrace.core.canon import dumps
from carttrace.core.clock import Clock
from carttrace.protocols.tools import SurfaceKind, ToolResult, ToolSpec
from carttrace.sandbox.surfaces.client import JSON, SandboxSurface, WireSink
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.mpp.endpoint import MPP_PATH, check_version
from carttrace.sandbox.surfaces.mpp.method import credential

CHECKOUT_ID = {"type": "string", "description": "The checkout id the store returned (UCP, ACP or AP2)"}
TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="request_payment",
        description=(
            "Ask to pay for a checkout over MPP. The store answers HTTP 402 with a payment challenge: check its amount "
            "(minor units), currency and recipient before you pay it."
        ),
        input_schema={"type": "object", "properties": {"checkout_id": CHECKOUT_ID}, "required": ["checkout_id"]},
    ),
    ToolSpec(
        name="pay",
        description=(
            "Pay a challenge from request_payment with the MPP credential that answers it (its payload.token). "
            "Returns the order and the payment receipt, or the store's problem and a fresh challenge."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "challenge_id": {"type": "string", "description": "The id of a challenge request_payment returned"},
                "token": {"type": "string", "description": "The credential's payload.token to pay with"},
            },
            "required": ["challenge_id", "token"],
        },
    ),
    ToolSpec(
        name="get_payment",
        description="The payment status of a checkout, with the receipt once it is paid.",
        input_schema={"type": "object", "properties": {"checkout_id": CHECKOUT_ID}, "required": ["checkout_id"]},
    ),
)


class MppSurface(SandboxSurface):
    """MPP over HTTP 402 at the store's MPP version."""

    kind: SurfaceKind = "mpp"
    protocol: ClassVar[str] = "mpp"

    def __init__(
        self,
        client: httpx.AsyncClient,
        host: SandboxHost,
        *,
        recorder: WireSink | None = None,
        session_id: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(client, host, recorder=recorder, session_id=session_id, clock=clock)
        check_version(host.require("mpp"))
        self._challenges: dict[str, tuple[str, Challenge]] = {}
        """Challenge id to (the paid path it answers, the challenge as received)."""

    async def list_tools(self) -> list[ToolSpec]:
        return [t.model_copy(deep=True) for t in TOOLS]

    async def call(self, name: str, arguments: dict[str, Any], *, idempotency_key: str | None = None) -> ToolResult:
        self._calls += 1
        if name == "request_payment":
            return await self._request(arguments)
        if name == "pay":
            return await self._pay(arguments)
        if name == "get_payment":
            return await self._status(arguments)
        return ToolResult(content=f"unknown tool {name!r}", is_error=True, error_code="protocol_error")

    def _path(self, arguments: Mapping[str, Any], suffix: str = "") -> str | None:
        target = arguments.get("checkout_id")
        if not isinstance(target, str) or not target:
            return None
        return f"{MPP_PATH}/checkouts/{quote(target, safe='')}{suffix}"

    async def _request(self, arguments: Mapping[str, Any]) -> ToolResult:
        path = self._path(arguments, "/complete")
        if path is None:
            return ToolResult(content="request_payment needs a checkout_id", is_error=True, error_code="protocol_error")
        response, parsed, latency = await self._post(path, {})
        return self._result(path, response, parsed, latency, expect_challenge=True)

    async def _pay(self, arguments: Mapping[str, Any]) -> ToolResult:
        chosen = self._challenges.get(str(arguments.get("challenge_id")))
        token = arguments.get("token")
        if chosen is None:
            return ToolResult(
                content="no challenge with this id was received; call request_payment first",
                is_error=True,
                error_code="protocol_error",
            )
        if not isinstance(token, str) or not token:
            return ToolResult(content="pay needs a token", is_error=True, error_code="protocol_error")
        path, challenge = chosen
        auth = credential(challenge, token).to_authorization()
        response, parsed, latency = await self._post(path, {challenge.credential_header.lower(): auth})
        return self._result(path, response, parsed, latency)

    async def _status(self, arguments: Mapping[str, Any]) -> ToolResult:
        path = self._path(arguments)
        if path is None:
            return ToolResult(content="get_payment needs a checkout_id", is_error=True, error_code="protocol_error")
        headers = {"accept": JSON, **self._headers()}
        response, parsed, latency = await self._send("GET", path, headers=headers, protocol="mpp", version=self.version)
        return self._result(path, response, parsed, latency)

    async def _post(self, path: str, extra: Mapping[str, str]) -> tuple[httpx.Response, Any, int]:
        headers = {"accept": JSON, **self._headers(), **extra}
        return await self._send("POST", path, headers=headers, protocol="mpp", version=self.version)

    def _result(
        self, path: str, response: httpx.Response, parsed: Any, latency: int, *, expect_challenge: bool = False
    ) -> ToolResult:
        challenges = self._remember(path, response)
        body: dict[str, Any] = dict(parsed) if isinstance(parsed, Mapping) else {}
        if challenges:
            body["challenges"] = challenges
        receipt = response.headers.get("payment-receipt")
        if receipt:
            try:
                parsed_receipt = Receipt.from_payment_receipt(receipt)
            except ParseError:
                body["receipt_error"] = "the store sent a Payment-Receipt that does not parse"
            else:
                body["receipt"] = _receipt(parsed_receipt)
        paid = response.status_code == 200
        asked = response.status_code == 402 and expect_challenge and bool(challenges)
        failed = not (paid or asked)
        code = _problem_type(body) if failed else None
        return ToolResult(
            content=dumps(body),
            is_error=failed,
            error_code=code or ("protocol_error" if failed else None),
            structured=body,
            latency_ms=latency,
        )

    def _remember(self, path: str, response: httpx.Response) -> list[dict[str, Any]]:
        """Parse and keep every Payment challenge of a 402, and describe each for the agent."""
        shown: list[dict[str, Any]] = []
        for header in response.headers.get_list("www-authenticate"):
            try:
                challenge = Challenge.from_www_authenticate(header)
            except ParseError:
                continue
            self._challenges[challenge.id] = (path, challenge)
            request = challenge.request
            shown.append(
                {
                    "id": challenge.id,
                    "method": challenge.method,
                    "intent": challenge.intent,
                    "amount": request.get("amount"),
                    "currency": request.get("currency"),
                    "recipient": request.get("recipient"),
                    "description": challenge.description,
                    "expires": challenge.expires,
                }
            )
        return shown


def _receipt(r: Receipt) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": r.status,
        "method": r.method,
        "timestamp": r.timestamp.isoformat().replace("+00:00", "Z"),
        "reference": r.reference,
    }
    if r.external_id:
        out["externalId"] = r.external_id
    return out


def _problem_type(body: Mapping[str, Any]) -> str | None:
    """The problem type slug of an error body: the payment problem a 409 carries (a replayed credential), else the
    body's own; None for a problem without a type (`about:blank`)."""
    nested = body.get("payment")
    for doc in (nested if isinstance(nested, Mapping) else {}, body):
        kind = doc.get("type")
        if isinstance(kind, str) and kind != "about:blank":
            return kind.rsplit("/", 1)[-1]
    return None

"""The Shop Pay interface: where the wallet's platform mints a single-use Shop Token for one checkout (D73).

Why an endpoint of its own. The spec's Token Acquisition Path A hands the platform a single-use Shop Token per
checkout "through the Shop Pay interface" (vendor/dev.shopify.shop_pay/2026-04-08/spec.md), which is Shop Pay's side,
never the merchant's: no UCP tool of the store mints one. Shop Pay publishes no HTTP shape for it, so this binding is
the sandbox's own, as AP2's REST binding is. It is served only where the store declares the Shop Pay handler or its
profile adopts it (a store that takes no Shop Pay payments has no Shop Pay interface to speak of).

One route: `POST /psp/shop_pay/tokens` `{checkout_id, token, amount, currency}` mints from the wallet's
`tok_shop_pay_...` credential, scoped to the checkout, its exact total and the store's `shop_id`, and answers 201 with
the `shop_...` token and its expiry on the logical clock. The mint is one core command (`mint_shop_token`), so it is
in the event log and replays like every other; the same body is idempotent by a key derived from it. A refusal is the
core's typed error as JSON, with the same status vocabulary the control routes use.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from carttrace.core.canon import dumps_bytes, loads
from carttrace.sandbox.surfaces.access import DEFAULT_SESSION, AccessRefused, named_session
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.keys import opaque_key
from carttrace.sandbox.surfaces.values import body_of

JSON = "application/json"
TOKENS_PATH = "/psp/shop_pay/tokens"


def shop_pay_routes(host: SandboxHost) -> list[Route]:
    server = ShopPayServer(host)
    return [Route(TOKENS_PATH, server.mint, methods=["POST"])]


class ShopPayServer:
    """The Shop Pay interface of one host."""

    def __init__(self, host: SandboxHost) -> None:
        self.host = host

    async def mint(self, request: Request) -> Response:
        try:
            session = named_session(request) or DEFAULT_SESSION
        except AccessRefused as e:
            return _json(e.status, _error("invalid_session", e.message))
        raw = await body_of(request)
        if raw is None:
            return _json(413, _error("request_too_large", "The request body is larger than this store reads"))
        body: Any = {}
        if raw.strip():
            try:
                body = loads(raw)
            except ValueError:
                body = None
        if not isinstance(body, dict):
            return _json(400, _error("invalid_json", "The request body must be a JSON object"))
        out = self.host.execute(
            {
                "type": "mint_shop_token",
                "session": session,
                "token": body.get("token"),
                "checkout_id": body.get("checkout_id"),
                "amount": body.get("amount"),
                "currency": body.get("currency"),
                "idempotency_key": f"shop|{session}|{opaque_key(_digest_parts(body), length=16)}",
            }
        )
        if out.error is not None:
            status = 400 if out.error.get("kind") == "validation" else 422
            return _json(status, _error(str(out.error.get("code")), str(out.error.get("message")), out.error))
        token = str(out.refs["credential_id"])
        cred = self.host.state.credentials[token]
        body_out = {
            "token": token,
            "type": "shop_token",
            "checkout_id": cred.checkout_id,
            "amount": cred.max_amount,
            "currency": cred.currency,
            "shop_id": cred.merchant,
            "expires_at_ms": cred.expires_at,
        }
        return _json(201, body_out)


def _digest_parts(body: dict[str, Any]) -> tuple[Any, ...]:
    return (body.get("token"), body.get("checkout_id"), body.get("amount"), body.get("currency"))


def _error(code: str, message: str, full: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if isinstance(full, dict):
        for key in ("field", "details", "hint"):
            if full.get(key):
                error[key] = full[key]
    return {"error": error}


def _json(status: int, body: Any) -> Response:
    return Response(dumps_bytes(body), status_code=status, media_type=JSON)

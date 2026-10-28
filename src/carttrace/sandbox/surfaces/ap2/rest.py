"""The sandbox's REST binding of AP2's merchant role, under `/api/ap2`.

Why a binding of our own. AP2 is transport-agnostic (its samples carry mandates in A2A messages, as DataParts keyed
`ap2.mandates.IntentMandate` and so on), and neither vendored release publishes an HTTP API. The sandbox serves the
merchant role over plain JSON with the releases' own types and data keys, so the agent meets AP2's objects exactly as
the release defines them.

Routes at v0.1.0:
- `POST /intent_mandates` `{"ap2.mandates.IntentMandate": ...}`: accepted, recorded, answered with matching items;
- `POST /cart_mandates` `{items, shipping_address, shipping_option_id, payer_email, cart_id?,
  "ap2.mandates.IntentMandate"?}`: the merchant-signed `ap2.mandates.CartMandate`, or what the cart still needs;
- `POST /payment_mandates` `{"ap2.mandates.PaymentMandate": ..., "ap2.mandates.CartMandate"?}`: verified and settled;
- `GET /payment_mandates/{id}`: where the payment stands.

Routes at v0.2.0:
- `POST /intent_mandates` `{"open_checkout_mandate": <JWS>}`;
- `POST /checkouts` `{items, ..., checkout_id?, open_checkout_mandate?}`: the `checkout_jwt` and `checkout_hash`;
- `POST /checkouts/{id}/complete` `{"checkout_mandate": <JWS>, "payment_mandate": <JWS>}`: the receipts;
- `GET /checkouts/{id}`: where the payment stands, with the receipts once settled.

At both: `GET /jwks`, the merchant's and its payment processor's public keys, so an agent can verify a cart's
signature. The session a call acts for is named by `MC-Sandbox-Session`, else `agent`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from carttrace.core.canon import dumps_bytes, loads
from carttrace.sandbox.surfaces.access import DEFAULT_SESSION, AccessRefused, named_session
from carttrace.sandbox.surfaces.ap2.keys import party_key
from carttrace.sandbox.surfaces.ap2.merchant import Ap2Merchant, Ap2Reply
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.values import body_of

AP2_PATH = "/api/ap2"
JSON = "application/json"
Handler = Callable[[Request], Awaitable[Response]]


class Ap2Server:
    """The AP2 merchant of one store at its profile's release."""

    def __init__(self, host: SandboxHost) -> None:
        self.host = host
        self.merchant = Ap2Merchant(host)

    def routes(self) -> list[Route]:
        m = self.merchant
        posts: dict[str, Callable[[str, Mapping[str, Any], str], Ap2Reply]]
        gets: dict[str, Callable[[str, str, str], Ap2Reply]]
        if m.version == "v0.1.0":
            posts = {
                "/intent_mandates": lambda s, b, base: m.intent_v1(s, b),
                "/cart_mandates": m.cart_v1,
                "/payment_mandates": m.payment_v1,
            }
            gets = {"/payment_mandates/{id}": m.payment_status_v1, "/carts/{id}": m.cart_status_v1}
        else:
            posts = {
                "/intent_mandates": lambda s, b, base: m.intent_v2(s, b),
                "/checkouts": m.checkout_v2,
            }
            gets = {"/checkouts/{id}": m.checkout_status_v2}
        out = [Route(f"{AP2_PATH}{path}", self._post(fn), methods=["POST"]) for path, fn in posts.items()]
        out += [Route(f"{AP2_PATH}{path}", self._get(fn), methods=["GET"]) for path, fn in gets.items()]
        if m.version == "v0.2.0":
            out.append(Route(f"{AP2_PATH}/checkouts/{{id}}/complete", self._complete, methods=["POST"]))
        out.append(Route(f"{AP2_PATH}/jwks", self.jwks, methods=["GET"]))
        return out

    def _post(self, fn: Callable[[str, Mapping[str, Any], str], Ap2Reply]) -> Handler:
        async def run(request: Request) -> Response:
            session, body, refused = await _read(request)
            if refused is not None:
                return refused
            return _json(fn(session, body, _base(request)))

        return run

    def _get(self, fn: Callable[[str, str, str], Ap2Reply]) -> Handler:
        async def run(request: Request) -> Response:
            try:
                session = named_session(request) or DEFAULT_SESSION
            except AccessRefused as e:
                return _json(Ap2Reply(e.status, {"error": "forbidden", "error_description": e.message}))
            return _json(fn(session, str(request.path_params["id"]), _base(request)))

        return run

    async def _complete(self, request: Request) -> Response:
        session, body, refused = await _read(request)
        if refused is not None:
            return refused
        return _json(self.merchant.complete_v2(session, str(request.path_params["id"]), body, _base(request)))

    async def jwks(self, request: Request) -> Response:
        keys = [party_key(self.host.pack.id, role).jwk() for role in ("merchant", "payment_processor")]
        return _json(Ap2Reply(200, {"keys": keys}))


async def _read(request: Request) -> tuple[str, Mapping[str, Any], Response | None]:
    """The session and the JSON body of a POST, or the response refusing them."""
    try:
        session = named_session(request) or DEFAULT_SESSION
    except AccessRefused as e:
        return "", {}, _json(Ap2Reply(e.status, {"error": "forbidden", "error_description": e.message}))
    raw = await body_of(request)
    if raw is None:
        return session, {}, _json(Ap2Reply(413, {"error": "request_too_large", "error_description": "body too large"}))
    try:
        body = loads(raw) if raw.strip() else {}
    except ValueError:
        body = None
    if not isinstance(body, dict):
        reply = Ap2Reply(400, {"error": "invalid_json", "error_description": "The body must be a JSON object"})
        return session, {}, _json(reply)
    return session, body, None


def _base(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _json(reply: Ap2Reply) -> Response:
    return Response(dumps_bytes(reply.body), status_code=reply.status, media_type=JSON)

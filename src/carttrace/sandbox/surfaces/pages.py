"""The pages UCP documents point at: the storefront, policy pages, continue URLs, order permalinks and the 3-D Secure
challenge page, plus `/health`.

Why serve them. A checkout must carry `links` (privacy and terms pages) and, when it needs the buyer, a `continue_url`;
an order carries a `permalink_url`; a 3-D Secure challenge carries the URL of a buyer-facing surface. A URL that leads
nowhere would teach an agent that links are decoration. These pages are small, read-only views of the state, built
from the pack's market table (fees, return window), so they say only what the sandbox actually does.

The challenge page is the environment's: `POST /3ds/{challenge}` answers the issuer's challenge as the environment
(`confirm_payment_action` from the system actor, which the core's trust policy allows), with `approved` saying whether
the buyer completed the step. The sealed script (pass or fail) decides the authentication, as a real issuer would.
The browser runtime completes it according to the persona; tests post to it directly. It is localhost only.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qs

from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.routing import Route

from carttrace.core.canon import dumps_bytes, loads
from carttrace.core.money import Money
from carttrace.sandbox.core.state import Checkout, Payment
from carttrace.sandbox.presented import presented
from carttrace.sandbox.surfaces.access import AccessRefused, require_local
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.ucp.ids import CartIds

JSON = "application/json"


def page_routes(host: SandboxHost) -> list[Route]:
    p = Pages(host)
    return [
        Route("/", p.home, methods=["GET"]),
        Route("/health", p.health, methods=["GET"]),
        Route("/policies/{name}", p.policy, methods=["GET"]),
        Route("/cart/c/{token}", p.cart, methods=["GET"]),
        Route("/cart/{cart_id}", p.cart, methods=["GET"]),
        Route("/checkout/{checkout_id}", p.checkout, methods=["GET"]),
        Route("/orders/{order_id}", p.order, methods=["GET"]),
        Route("/3ds/{challenge_id}", p.challenge, methods=["GET", "POST"]),
    ]


class Pages:
    def __init__(self, host: SandboxHost) -> None:
        self.host = host

    def _money(self, amount: int) -> str:
        cur = self.host.state.currency
        return f"{Money(amount, cur).major()} {cur}"

    async def home(self, request: Request) -> Response:
        shown = presented(self.host.pack)
        if shown.neutral:
            return _page(shown.store_name, f"<p>Welcome to {html.escape(shown.store_name)}.</p>")
        domain = html.escape(shown.domain)
        return _page(f"{domain} (sandbox)", f"<p>A sandbox replica of {domain}. Nothing here is a real sale.</p>")

    async def health(self, request: Request) -> Response:
        body = {
            "ok": True,
            "livemode": False,
            "env": self.host.pack.id,
            "profile": self.host.profile.id,
            "ucp": self.host.ucp.version,
            "mcp": self.host.mcp.version,
            "episode": self.host.episode,
        }
        return Response(dumps_bytes(body), media_type=JSON)

    async def policy(self, request: Request) -> Response:
        texts = self._policies()
        name = str(request.path_params["name"])
        if name not in texts:
            return _page("Not found", "<p>No such policy.</p>", status=404)
        return _page(name.replace("-", " ").capitalize(), f"<p>{html.escape(texts[name])}</p>")

    def _policies(self) -> dict[str, str]:
        market = self.host.pack.market()
        ships = []
        for o in market.shipping:
            free = f", free from {self._money(o.free_above)}" if o.free_above is not None else ""
            ships.append(f"{o.title}: {self._money(o.amount)}{free}.")
        if market.unshippable:
            ships.append(f"No shipping to {', '.join(market.unshippable)}.")
        shown = presented(self.host.pack)
        if shown.neutral:
            return {
                "shipping-policy": " ".join(ships),
                "refund-policy": f"Returns are accepted within {market.return_window_days} days of delivery.",
                "privacy-policy": "We use your contact details only to process and deliver your order.",
                "terms-of-service": "Orders are subject to availability and to our confirmation of payment.",
                "subscription-policy": "We do not offer subscriptions; every order is a single purchase.",
                "contact-information": f"Write to support@{shown.domain}."
                + (f" Our address: {shown.store_address}." if shown.store_address else ""),
                "legal-notice": f"{shown.store_name}, an online store"
                + (f", {shown.store_address}." if shown.store_address else "."),
            }
        return {
            "shipping-policy": " ".join(ships),
            "refund-policy": f"Returns are accepted within {market.return_window_days} days of delivery.",
            "privacy-policy": "The sandbox keeps contact details for the episode only and stores no names.",
            "terms-of-service": "This is a sandbox: no order is real, no money moves, no goods ship.",
            "subscription-policy": "The sandbox sells no subscriptions; every order is a single purchase.",
            "contact-information": "The sandbox has no staff; questions go nowhere.",
            "legal-notice": "A sandbox replica for benchmarking shopping agents; not the store it copies.",
        }

    async def cart(self, request: Request) -> Response:
        ident = str(request.path_params.get("cart_id") or "")
        token = request.path_params.get("token")
        if isinstance(token, str):
            ids = CartIds(self.host.pack.id, keyed=True)
            core = ids.by_token(token, self.host.state.carts)
            ident = core if core is not None and request.query_params.get("key") == ids.key(core) else ""
        cart = self.host.state.carts.get(ident)
        if cart is None or cart.status == "canceled":
            return _page("Cart not found", "<p>This cart does not exist or has expired.</p>", status=404)
        lines = sum(ln.quantity for ln in cart.lines)
        return _page("Your cart", f"<p>{lines} item(s), {self._money(cart.totals.total)}.</p>")

    async def checkout(self, request: Request) -> Response:
        ck = self.host.state.checkouts.get(str(request.path_params["checkout_id"]))
        if ck is None:
            return _page("Checkout not found", "<p>This checkout does not exist.</p>", status=404)
        status = html.escape(ck.status.replace("_", " "))
        return _page("Checkout", f"<p>Status: {status}. Total {self._money(ck.totals.total)}.</p>")

    async def order(self, request: Request) -> Response:
        o = self.host.state.orders.get(str(request.path_params["order_id"]))
        if o is None:
            return _page("Order not found", "<p>This order does not exist.</p>", status=404)
        return _page(
            f"Order #{o.number}", f"<p>Status: {html.escape(o.status)}. Total {self._money(o.totals.total)}.</p>"
        )

    async def challenge(self, request: Request) -> Response:
        try:
            require_local(request)
        except AccessRefused as e:
            return _page("Forbidden", f"<p>{html.escape(e.message)}</p>", status=e.status)
        found = self._pending(str(request.path_params["challenge_id"]))
        if found is None:
            return _page("No challenge", "<p>No 3-D Secure challenge is waiting here.</p>", status=404)
        ck, pay = found
        if request.method == "GET":
            amount = self._money(pay.amount)
            form = (
                f"<p>Your bank asks you to confirm a payment of {amount}.</p>"
                '<form method="post"><button name="approved" value="1">Confirm</button>'
                '<button name="approved" value="0">Cancel</button></form>'
            )
            return _page("Confirm your payment", form)
        approved = _approved(await request.body(), request.headers.get("content-type", ""))
        out = self.host.execute(
            {
                "type": "confirm_payment_action",
                "actor": "system",
                "session": ck.session,
                "checkout_id": ck.id,
                "approved": approved,
            }
        )
        state = self.host.state
        body: dict[str, Any] = {
            "ok": out.ok,
            "checkout_status": state.checkouts[ck.id].status,
            "payment_status": state.payments[pay.id].status,
        }
        return Response(dumps_bytes(body), media_type=JSON)

    def _pending(self, challenge_id: str) -> tuple[Checkout, Payment] | None:
        state = self.host.state
        for pay in state.payments.values():
            if pay.challenge_id == challenge_id and pay.status == "requires_action":
                ck = state.checkouts.get(pay.checkout_id)
                if ck is not None:
                    return ck, pay
        return None


def _approved(raw: bytes, content_type: str) -> bool:
    """`approved` from a form post (`approved=1`) or JSON (`{"approved": true}`); a body that says nothing approves,
    as a buyer who submits the challenge form does."""
    if content_type.split(";", maxsplit=1)[0].strip().lower() == JSON:
        try:
            body = loads(raw) if raw.strip() else {}
        except ValueError:
            return False
        value = body.get("approved", True) if isinstance(body, Mapping) else True
        return value is True
    fields = parse_qs(raw.decode("utf-8", "replace"))
    return fields.get("approved", ["1"])[-1] in ("1", "true", "yes")


def _page(title: str, body: str, *, status: int = 200) -> HTMLResponse:
    text = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title></head>'
        f"<body><h1>{html.escape(title)}</h1>{body}</body></html>"
    )
    return HTMLResponse(text, status_code=status)

"""The storefront's backend: each WebMCP tool as the same UCP operation, and so the same core command, a UCP agent's
call becomes.

Why through the UCP operations. Grading reads the core: its state, its event log, its digest. A purchase made through
the storefront's tools must leave exactly the state and the events the same purchase made over UCP leaves, or the two
protocols could not be compared on one store. So the storefront translates each tool into the UCP call a careful UCP
client would make (`UcpOperations.call`, which validates it against the served release, refuses card numbers, runs
one core command or logs one refusal, and renders the result from state) and builds its page from the document that
comes back. A refusal the storefront makes itself (no cart yet, a cart locked to its checkout, arguments that fail the
tool's schema) is logged as one refusal too, so every tool call is still one entry in the log.

What the page keeps for the shopper, as a real storefront's cookie would: the session's current cart (its newest one
not canceled, unless its checkout completed) and that cart's checkout. A cart changed after checkout started cancels
that checkout (D87), as a storefront's cart page does; while a payment completes the cart stays locked to it, and
after the order a new add_to_cart starts a new cart.

Idempotency. A storefront page sends its own idempotency key with an order, so a resubmission places one order: the
key is derived from the store, the session, the checkout and how many payments that checkout has seen
(`order_key`), so a retry before a payment is recorded reuses it and a second attempt after a decline gets a new one.

Details hidden by an experiment (the collect layer's detail filter) are cut from every catalog document before a view
is built from it, so neither the page nor any tool answer can show them (`StorefrontOperations.attach`).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from carttrace.protocols.ucp_meta import PLACEHOLDER_AGENT_PROFILE, idempotency_key
from carttrace.sandbox.core.config import FAKE_NAMESPACE, SHOP_PAY_NAMESPACE
from carttrace.sandbox.core.errors import IllegalTransition, InvalidCommand, NotFound
from carttrace.sandbox.core.machines import OPEN_CHECKOUT
from carttrace.sandbox.core.state import Cart, Checkout, Line, State
from carttrace.sandbox.presented import presented
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.ucp.catalog import lookup_docs
from carttrace.sandbox.surfaces.ucp.guard import card_number_field
from carttrace.sandbox.surfaces.ucp.operations import Reply, UcpOperations
from carttrace.sandbox.surfaces.ucp.render import method_id
from carttrace.sandbox.surfaces.webmcp.tools import BY_NAME, CURSOR_MAX, FULFILLMENT, ToolDef, form_schema, with_choices
from carttrace.sandbox.surfaces.webmcp.views import (
    Json,
    cart_view,
    checkout_view,
    compare_view,
    error_view,
    first_error,
    order_view,
    product_view,
    search_view,
)

ViewFilter = Callable[[Any, Counter[str]], Any]
"""The collect layer's detail filter for one episode: (catalog document, counts) to the document without the hidden
details, adding to `counts` how many statements of each it removed. `carttrace.collect.surfaces.without_details` fits, with
the episode's `HiddenSet` bound."""
INTERNAL = frozenset({"payment_result"})
"""Storefront operations the page calls that are not tools: the checkout read after the bank's challenge."""
CARD_NAMESPACE = "dev.shopify.card"
INSTRUMENT_ID = "instrument-1"


def order_key(pack_id: str, session: str, checkout_id: str, attempt: int) -> str:
    """The idempotency key a storefront's order submission carries (UCP types the key as a UUID)."""
    return idempotency_key(pack_id, "webmcp-order", session, checkout_id, attempt)


@dataclass(frozen=True, slots=True)
class Answer:
    """What the storefront API answers a page with: the view, and the HTTP status (a protocol fault's own)."""

    view: Json
    status: int = 200


@dataclass(slots=True)
class Attached:
    """One session's detail filter and what it has removed since the surface last asked."""

    apply: ViewFilter
    counts: Counter[str] = field(default_factory=Counter)

    def take(self) -> dict[str, int]:
        out = dict(sorted(self.counts.items()))
        self.counts.clear()
        return out


class StorefrontOperations:
    def __init__(self, host: SandboxHost, ops: UcpOperations) -> None:
        self.host = host
        self.ops = ops
        self.market = host.pack.market()
        self.currency = host.pack.snapshot.currency
        self.handlers = payment_handlers(ops)
        self.placed: dict[str, int] = {}
        """place_order calls per checkout: each is a new attempt with its own idempotency key. Counting payments
        instead reused the first key after a refusal that made no payment (a price change at completion), so the
        store replayed the refusal to the re-approved attempt (the payments check, D94)."""
        self.filters: dict[str, Attached] = {}
        self.page_tokens: dict[tuple[str, str], Mapping[str, Any]] = {}
        """(session, page token) to the upstream pagination it stands for (`page_token`)."""
        choices: dict[str, list[tuple[str, str]]] = {
            "method": [("shipping", "Ship it")] + ([("pickup", "Pick up in store")] if self.market.pickup else []),
            "option_id": [(o.id, o.title) for o in self.market.shipping],
            "location_id": [(p.id, p.title) for p in self.market.pickup],
        }
        self.fulfillment_tool = with_choices(FULFILLMENT, choices)
        self._update_cart_needs_id = _payload_requires(ops, "update_cart", "id")

    # -- the seam for hidden details -----------------------------------------------------------------------------------

    def attach(self, session: str, view_filter: ViewFilter) -> Attached:
        """Cut what `view_filter` removes from every catalog document shown to `session`, on pages and in tool
        answers alike."""
        attached = Attached(view_filter)
        self.filters[session] = attached
        return attached

    def detach(self, session: str) -> None:
        self.filters.pop(session, None)

    def shown(self, session: str, doc: Any) -> Any:
        """A catalog document as `session` may see it. Two copies of one fact in one call count once, as the collect
        filter counts them (the larger count of each detail, not the sum)."""
        attached = self.filters.get(session)
        if attached is None or doc is None:
            return doc
        found: Counter[str] = Counter()
        out = attached.apply(doc, found)
        attached.counts |= found
        return out

    # -- tools ---------------------------------------------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self.host.state

    def tool(self, name: str) -> ToolDef | None:
        if name == FULFILLMENT.name:
            return self.fulfillment_tool
        return BY_NAME.get(name)

    def schema(self, tool: ToolDef) -> Mapping[str, Any]:
        return form_schema(tool) if tool.style == "declarative" else tool.input_schema

    def run(self, name: str, arguments: Any, *, session: str, base: str) -> Answer:
        """Run one tool (or an internal page operation) for `session`; `base` is the origin pages are served at."""
        tool = self.tool(name)
        if tool is None and name not in INTERNAL:
            return self._refuse(name, "unknown_tool", "This store has no such tool", page="home", url="/shop")
        args = dict(arguments) if isinstance(arguments, Mapping) else {}
        if not isinstance(arguments, Mapping):
            return self._refuse(name, "invalid_argument", "The input must be a JSON object")
        if card_number_field(args) is not None:
            text = "A card number was sent; this store takes payment tokens only"
            return self._refuse(name, "card_number_refused", text)
        if tool is not None:
            if tool.style == "declarative":
                args = {k: v for k, v in args.items() if not (isinstance(v, str) and v.strip() == "")}
            problem = _problem(self.schema(tool), args)
            if problem is not None:
                return self._refuse(name, "invalid_argument", problem)
        handler: Callable[[Mapping[str, Any], str, str], Answer] = getattr(self, f"_{name}")
        return handler(args, session, base)

    # -- catalog -------------------------------------------------------------------------------------------------------

    def _search_products(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        query = str(args["query"])
        catalog: dict[str, Any] = {"query": query}
        filters: dict[str, Any] = {}
        price = {k: args[f"{k}_price"] for k in ("min", "max") if f"{k}_price" in args}
        if price:
            filters["price"] = price
        if "category" in args:
            filters["categories"] = [str(args["category"])]
        if filters:
            catalog["filters"] = filters
        if "cursor" in args:
            cursor = str(args["cursor"])
            catalog["pagination"] = dict(self.page_tokens.get((session, cursor), {"cursor": cursor}))
        reply = self._ucp("search_catalog", session, base, catalog=catalog)
        shown = {k: args.get(k) for k in ("min_price", "max_price", "category")}
        if reply.fault is not None or reply.doc is None:
            return self._fault(reply, "search", "/shop/search")
        doc = self.shown(session, reply.doc)
        view = search_view(query, doc, filters=shown)
        upstream = next_page(doc)
        if upstream is not None:
            view["next_cursor"] = self.page_token(session, upstream)
        if reply.is_error:
            view["error"] = first_error(doc) or {"code": "search_failed", "message": "The search did not run"}
        return Answer(view)

    def page_token(self, session: str, upstream: Mapping[str, Any]) -> str:
        """The next_cursor the page hands out for the next page's upstream pagination (`next_page`), always within
        the tool's own `cursor` schema: an upstream cursor that fits is handed out as it is; a longer one, or an
        offset page (Medusa's), gets a short token of the page's own (as a storefront's `?page=` link stands for its
        backend's cursor), read back to the upstream pagination when it is passed in. Stable for a session."""
        cursor = upstream.get("cursor")
        if isinstance(cursor, str) and len(cursor) <= CURSOR_MAX and set(upstream) == {"cursor"}:
            return cursor
        key = json.dumps(dict(upstream), sort_keys=True)
        token = "pg-" + hashlib.sha256(f"{session}\x00{key}".encode()).hexdigest()[:16]
        self.page_tokens[(session, token)] = dict(upstream)
        return token

    def _get_product(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        ident = str(args["product_id"])
        catalog: dict[str, Any] = {"id": ident}
        options = args.get("options")
        if isinstance(options, Mapping) and options:
            catalog["selected"] = [{"name": str(k), "label": str(v)} for k, v in options.items()]
        reply = self._ucp("get_product", session, base, catalog=catalog)
        url = product_view({"id": ident})["url"]
        if reply.fault is not None or reply.doc is None:
            return self._fault(reply, "product", url)
        doc = self.shown(session, reply.doc)
        found = None if reply.is_error else _one_product(doc, ident)
        if found is None:
            error = first_error(doc) or {"code": "not_found", "message": "No product has this id"}
            return Answer(error_view("product", url, error["code"], error["message"]))
        return Answer(product_view(found))

    def _compare_products(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        ids = [str(i) for i in args["product_ids"]]
        reply = self._ucp("lookup_catalog", session, base, catalog={"ids": ids})
        if reply.fault is not None or reply.doc is None:
            return self._fault(reply, "compare", "/shop/compare")
        doc = self.shown(session, reply.doc)
        view = compare_view(ids, doc)
        if len(view["products"]) < 2:
            view["error"] = {"code": "not_found", "message": "Fewer than two of these products are in this store"}
        return Answer(view)

    # -- the cart ------------------------------------------------------------------------------------------------------

    def _add_to_cart(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        variant = str(args["variant_id"])
        quantity = int(args.get("quantity", 1))
        cart = self.current_cart(session)
        if cart is not None and self.locked(cart):
            refused = self._reopen("add_to_cart", cart, session, base)
            if refused is not None:
                return refused
        if cart is None:
            # The page is the store's own, so it names the store's currency as the buyer's context (UCP context,
            # cart.json since 2026-04-08), as Medusa's storefront does: Medusa prices a line only in a region's
            # currency, and a cart created without one is refused ("Variants with IDs ... do not have a price").
            line = {"item": {"id": variant}, "quantity": quantity}
            reply = self._ucp(
                "create_cart", session, base, cart={"line_items": [line], "context": {"currency": self.currency}}
            )
            return self._cart_answer(reply)
        lines = _line_items(cart.lines)
        match = next((li for li in lines if li["item"]["id"] == variant), None)
        if match is not None:
            match["quantity"] += quantity
        else:
            lines.append({"item": {"id": variant}, "quantity": quantity})
        return self._cart_answer(self._update_cart(cart, lines, session, base))

    def _update_cart_line(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        found = self._cart_line("update_cart_line", args, session, base)
        if isinstance(found, Answer):
            return found
        cart, line_id = found
        lines = _line_items(cart.lines)
        for li in lines:
            if li["id"] == line_id:
                li["quantity"] = int(args["quantity"])
        return self._cart_answer(self._update_cart(cart, lines, session, base))

    def _remove_cart_line(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        found = self._cart_line("remove_cart_line", args, session, base)
        if isinstance(found, Answer):
            return found
        cart, line_id = found
        lines = [li for li in _line_items(cart.lines) if li["id"] != line_id]
        return self._cart_answer(self._update_cart(cart, lines, session, base))

    def _get_cart(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        cart = self.current_cart(session)
        if cart is None:
            self.host.reject("get_cart", NotFound(code="cart_not_found", message="No cart yet", field="cart_id"))
            return Answer(cart_view(None))
        reply = self._ucp("get_cart", session, base, id=self.wire_id(cart))
        if reply.fault is not None or reply.doc is None:
            return self._fault(reply, "cart", "/shop/cart")
        if reply.is_error:
            return self._cart_answer(reply)
        return Answer(cart_view(reply.doc, locked=self.locked(cart)))

    def _apply_discount(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        code = str(args["code"]).strip()
        ck = self.open_checkout(session)
        if ck is not None:
            codes = [*ck.codes, code] if code.upper() not in ck.codes else list(ck.codes)
            payload = {"line_items": _line_items(ck.lines), "discounts": {"codes": codes}}
            return self._checkout_answer(self._ucp("update_checkout", session, base, id=ck.id, checkout=payload))
        cart = self.current_cart(session)
        if cart is None:
            return self._refuse("apply_discount", "cart_not_found", "Add something to the cart first", page="cart")
        codes = [*cart.codes, code] if code.upper() not in cart.codes else list(cart.codes)
        return self._cart_answer(self._update_cart(cart, _line_items(cart.lines), session, base, codes=codes))

    def _update_cart(
        self, cart: Cart, lines: list[dict[str, Any]], session: str, base: str, *, codes: Sequence[str] | None = None
    ) -> Reply:
        payload: dict[str, Any] = {"line_items": lines}
        if self._update_cart_needs_id:
            payload["id"] = self.wire_id(cart)
        if codes is not None:
            payload["discounts"] = {"codes": list(codes)}
        return self._ucp("update_cart", session, base, id=self.wire_id(cart), cart=payload)

    def _cart_line(self, tool: str, args: Mapping[str, Any], session: str, base: str) -> tuple[Cart, str] | Answer:
        cart = self.current_cart(session)
        if cart is None:
            return self._refuse(tool, "cart_not_found", "The cart is empty", page="cart", url="/shop/cart")
        if self.locked(cart):
            refused = self._reopen(tool, cart, session, base)
            if refused is not None:
                return refused
        line_id = self.ops.ids.parse_line(str(args["line_id"]))
        if all(ln.id != line_id for ln in cart.lines):
            return self._refuse(tool, "line_not_found", "The cart has no line with this id", page="cart")
        return cart, line_id

    def _cart_answer(self, reply: Reply) -> Answer:
        if reply.fault is not None or reply.doc is None:
            return self._fault(reply, "cart", "/shop/cart")
        if reply.is_error:
            error = first_error(reply.doc) or {"code": "cart_failed", "message": "The cart did not change"}
            failed: Json = {"page": "cart", "url": "/shop/cart", "cart": None, "error": error}
            messages = cart_view(reply.doc).get("messages")
            if messages:
                failed["messages"] = messages
            return Answer(failed)
        view = cart_view(reply.doc)
        rejected = first_error(reply.doc)
        if rejected is not None:
            view["error"] = rejected
        return Answer(view)

    # -- the checkout --------------------------------------------------------------------------------------------------

    def _start_checkout(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        cart = self.current_cart(session)
        if cart is None or not cart.lines:
            return self._refuse("start_checkout", "empty_cart", "Add something to the cart first", page="cart")
        checkout: dict[str, Any] = {"cart_id": self.wire_id(cart), "line_items": []}
        if self.ops.platform.dialect == "medusa":
            # The plugin's own validator refuses cart_id on a create and prices lines only in a currency the request
            # names, so the page sends the cart's lines and the store's currency, as a storefront on the plugin must.
            # The pack itself takes cart_id since D81 (data/sandbox/packs/medusa.yaml, create_checkout.cart_id).
            lines = [{"item": {"id": ln.variant_id}, "quantity": ln.quantity} for ln in cart.lines]
            checkout = {"line_items": lines, "context": {"currency": self.currency}}
        reply = self._ucp("create_checkout", session, base, checkout=checkout)
        return self._checkout_answer(reply)

    def _set_buyer_contact(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        ck = self.open_checkout(session)
        if ck is None:
            return self._no_checkout("set_buyer_contact", session)
        buyer: dict[str, str] = {"email": str(args["email"]).strip()}
        if args.get("phone"):
            buyer["phone_number"] = str(args["phone"]).strip()
        first, _, last = str(args.get("name") or "").strip().partition(" ")
        if first:
            buyer["first_name"] = first
        if last.strip():
            buyer["last_name"] = last.strip()
        payload = {"line_items": _line_items(ck.lines), "buyer": buyer}
        return self._checkout_answer(self._ucp("update_checkout", session, base, id=ck.id, checkout=payload))

    def _set_shipping_address(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        ck = self.open_checkout(session)
        if ck is None:
            return self._no_checkout("set_shipping_address", session)
        address = {k: str(v).strip() for k, v in args.items() if isinstance(v, str) and v.strip()}
        address["address_country"] = address["address_country"].upper()
        method: dict[str, Any] = {"type": "shipping", "destinations": [address], **_method_lines(ck)}
        if ck.groups:
            method["id"] = method_id(ck.groups[0].id)
        payload = {"line_items": _line_items(ck.lines), "fulfillment": {"methods": [method]}}
        return self._checkout_answer(self._ucp("update_checkout", session, base, id=ck.id, checkout=payload))

    def _choose_fulfillment(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        ck = self.open_checkout(session)
        if ck is None:
            return self._no_checkout("choose_fulfillment", session)
        kind = str(args["method"])
        group = ck.groups[0] if ck.groups else None
        method: dict[str, Any] = {"type": kind, **_method_lines(ck)}
        if group is not None:
            method["id"] = method_id(group.id)
        if kind == "pickup":
            if not args.get("location_id"):
                return self._refuse("choose_fulfillment", "missing", "Choose a pickup location", page="checkout")
            method["selected_destination_id"] = str(args["location_id"])
        else:
            if not args.get("option_id"):
                return self._refuse("choose_fulfillment", "missing", "Choose a shipping option", page="checkout")
            selection: dict[str, Any] = {"selected_option_id": str(args["option_id"])}
            if group is not None:
                selection = {"id": group.id, **selection}
            method["groups"] = [selection]
        payload = {"line_items": _line_items(ck.lines), "fulfillment": {"methods": [method]}}
        return self._checkout_answer(self._ucp("update_checkout", session, base, id=ck.id, checkout=payload))

    def _get_checkout(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        ck = self.latest_checkout(session)
        if ck is None:
            self.host.reject(
                "get_checkout", NotFound(code="checkout_not_found", message="No checkout yet", field="checkout_id")
            )
            return Answer(checkout_view_empty())
        reply = self._ucp("get_checkout", session, base, id=ck.id)
        return self._checkout_answer(reply, reading=True, base=base)

    def _place_order(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        ck = self.open_checkout(session)
        if ck is None:
            return self._no_checkout("place_order", session)
        handler = str(args.get("payment_handler") or self.handlers[0])
        if handler not in self.handlers:
            text = f"This store takes payment through {', '.join(self.handlers)}"
            return self._refuse("place_order", "payment_handler_unavailable", text, page="checkout")
        attempt = self.placed.get(ck.id, 0)
        self.placed[ck.id] = attempt + 1
        token = str(args["payment_token"])
        instrument: dict[str, Any] = {
            "id": INSTRUMENT_ID,
            "handler_id": handler,
            "type": "card",
            "selected": True,
            "credential": {"type": "token", "token": token},
        }
        if self._namespace(handler) == SHOP_PAY_NAMESPACE:
            # D73: a Shop Pay method is paid with the single-use Shop Token the wallet minted, in the handler's own
            # instrument shape (the spec's `shop_pay` instrument with a `shop_token` credential).
            instrument["type"] = "shop_pay"
            instrument["credential"] = {"type": "shop_token", "token": token}
        reply = self._ucp(
            "complete_checkout",
            session,
            base,
            id=ck.id,
            checkout={"payment": {"instruments": [instrument]}},
            idempotency=order_key(self.host.pack.id, session, ck.id, attempt),
        )
        return self._payment_answer(reply, ck.id, base)

    def _namespace(self, handler_id: str) -> str | None:
        """The namespace of a payment method the page offers, from the store's declared handler entries."""
        for ns, entries in self.ops.envelope.handlers.items():
            if any(str(e.get("id")) == handler_id for e in entries):
                return ns
        return None

    def _payment_result(self, args: Mapping[str, Any], session: str, base: str) -> Answer:
        ck = self.latest_checkout(session)
        if ck is None:
            return self._no_checkout("payment_result", session)
        reply = self._ucp("get_checkout", session, base, id=ck.id)
        return self._payment_answer(reply, ck.id, base)

    def _payment_answer(self, reply: Reply, checkout_id: str, base: str) -> Answer:
        if reply.fault is not None or reply.doc is None:
            return self._fault(reply, "checkout", "/shop/checkout")
        ck = self.state.checkouts[checkout_id]
        if ck.status == "completed" and ck.order_id is not None:
            return Answer(order_view(self.ops.render.order(self.state, self.state.orders[ck.order_id], base)))
        challenge = self.challenge_of(ck)
        view = checkout_view(reply.doc, market=self.market, handlers=self.handlers, challenge=challenge)
        if challenge is None:
            view["error"] = _failure(reply.doc) or {"code": "order_not_placed", "message": "The order was not placed"}
        return Answer(view)

    def _checkout_answer(self, reply: Reply, *, reading: bool = False, base: str = "") -> Answer:
        if reply.fault is not None or reply.doc is None:
            return self._fault(reply, "checkout", "/shop/checkout")
        doc = reply.doc
        if reply.is_error:
            error = first_error(doc) or {"code": "checkout_failed", "message": "The checkout did not change"}
            return Answer({**checkout_view_empty(), "error": error})
        ck = self.state.checkouts.get(str(doc.get("id") or ""))
        if reading and ck is not None and ck.status == "completed" and ck.order_id is not None:
            order = self.ops.render.order(self.state, self.state.orders[ck.order_id], base)
            return Answer(order_view(order))
        challenge = self.challenge_of(ck) if ck is not None else None
        view = checkout_view(doc, market=self.market, handlers=self.handlers, challenge=challenge)
        failure = None if reading else _failure(doc)
        if failure is not None:
            view["error"] = failure
        return Answer(view)

    def _no_checkout(self, tool: str, session: str) -> Answer:
        latest = self.latest_checkout(session)
        if latest is not None and latest.status == "completed":
            return self._refuse(tool, "order_placed", "The order for this checkout is placed already", page="checkout")
        return self._refuse(tool, "checkout_not_found", "Start checkout first (start_checkout)", page="checkout")

    # -- what the page keeps -------------------------------------------------------------------------------------------

    def current_cart(self, session: str) -> Cart | None:
        """The session's newest cart not canceled, unless its checkout already became an order."""
        carts = [c for c in self.state.carts.values() if c.session == session and c.status != "canceled"]
        if not carts:
            return None
        cart = max(carts, key=lambda c: (c.created_at, c.id))
        ck = self.state.checkouts.get(cart.checkout_id) if cart.checkout_id else None
        return None if ck is not None and ck.status == "completed" else cart

    def wire_id(self, cart: Cart) -> str:
        """The cart's id as the store spells it to a client, which is what the UCP operations read: the core's own id,
        or on a keyed platform the `gid://shopify/Cart/<token>?key=<key>` the cart page shows. Sending the core's id
        there was refused as `invalid_cart_id`, so on a Shopify replica every storefront call naming the current
        cart failed (sbx-replica-0003, 2026-10-13: get_cart {} was `cart_not_found` in the store's log)."""
        return self.ops.ids.wire_cart(cart.id)

    def locked(self, cart: Cart) -> bool:
        ck = self.state.checkouts.get(cart.checkout_id) if cart.checkout_id else None
        return ck is not None and (ck.status in OPEN_CHECKOUT or ck.status == "complete_in_progress")

    def open_checkout(self, session: str) -> Checkout | None:
        ck = self.latest_checkout(session)
        return ck if ck is not None and ck.status in OPEN_CHECKOUT else None

    def latest_checkout(self, session: str) -> Checkout | None:
        found = [k for k in self.state.checkouts.values() if k.session == session and k.status != "canceled"]
        return max(found, key=lambda k: (k.created_at, k.id)) if found else None

    def challenge_of(self, ck: Checkout) -> str | None:
        """The bank's challenge page waiting on a checkout, as a path on the sandbox origin."""
        pay = self.state.payments.get(ck.payment_id) if ck.payment_id else None
        waiting = ck.status == "requires_escalation" and ck.escalation == "payment_action_required"
        if waiting and pay is not None and pay.status == "requires_action" and pay.challenge_id:
            return f"/3ds/{pay.challenge_id}"
        return None

    # -- pages without a tool (navigations: views of the state, no command) --------------------------------------------

    def page(self, route: str, params: Mapping[str, str], *, session: str, base: str) -> Json:
        """What a page loaded by URL shows. A load is not a tool call: it reads the state and logs nothing, as the
        sandbox's other pages do (`carttrace.sandbox.surfaces.pages`)."""
        build = {
            "search": self._search_page,
            "product": self._product_page,
            "compare": self._compare_page,
            "cart": self._cart_page,
            "checkout": self._checkout_page,
            "order": self._order_page,
        }.get(route)
        if build is None:
            return {"page": "home", "url": "/shop", "store": presented(self.host.pack).domain}
        return build(params, session, base)

    def _search_page(self, params: Mapping[str, str], session: str, base: str) -> Json:
        query = params.get("q", "")
        doc: dict[str, Any] = {"products": [], "pagination": {"has_next_page": False}}
        if query.strip():
            page = self.ops.catalog.search_page(self.state, {"query": query})
            doc = {"products": page.products, "pagination": page.pagination}
        return search_view(query, self.shown(session, doc), filters={})

    def _product_page(self, params: Mapping[str, str], session: str, base: str) -> Json:
        hit = self.ops.catalog.find(self.state, params.get("id", ""))
        if hit is None:
            return error_view("product", "/shop", "not_found", "No product has this id")
        product, variant = hit
        detail = self.ops.catalog.detail({"id": params.get("id", "")}, product, variant)
        return product_view(self.shown(session, {"product": detail})["product"])

    def _compare_page(self, params: Mapping[str, str], session: str, base: str) -> Json:
        ids = [i for i in params.get("ids", "").split(",") if i][:4]
        docs, _ = lookup_docs(self.ops.catalog, self.state, ids)
        return compare_view(ids, self.shown(session, {"products": docs}))

    def _cart_page(self, params: Mapping[str, str], session: str, base: str) -> Json:
        cart = self.current_cart(session)
        if cart is None:
            return cart_view(None)
        return cart_view(self.ops.render.cart(self.state, cart, [], base), locked=self.locked(cart))

    def _checkout_page(self, params: Mapping[str, str], session: str, base: str) -> Json:
        ck = self.latest_checkout(session)
        if ck is None:
            return checkout_view_empty()
        doc = self.ops.render.checkout(self.state, ck, [], base)
        return checkout_view(doc, market=self.market, handlers=self.handlers, challenge=self.challenge_of(ck))

    def _order_page(self, params: Mapping[str, str], session: str, base: str) -> Json:
        order = self.state.orders.get(params.get("id", ""))
        if order is None or order.session != session:
            return error_view("order", "/shop", "not_found", "No order has this id")
        return order_view(self.ops.render.order(self.state, order, base))

    # -- shared steps --------------------------------------------------------------------------------------------------

    def _ucp(self, name: str, session: str, base: str, *, idempotency: str | None = None, **arguments: Any) -> Reply:
        meta: dict[str, Any] = {"ucp-agent": {"profile": PLACEHOLDER_AGENT_PROFILE}}
        if idempotency is not None:
            meta["idempotency-key"] = idempotency
        return self.ops.call(name, {"meta": meta, **arguments}, session=session, base_url=base)

    def _refuse(self, tool: str, code: str, message: str, *, page: str = "checkout", url: str | None = None) -> Answer:
        """A call the storefront refused itself: one logged refusal, and the page saying why."""
        error = (
            IllegalTransition(code=code, message=message, field="cart_id")
            if code in ("checkout_started", "order_placed")
            else NotFound(code=code, message=message, field="id")
            if code.endswith("_not_found")
            else InvalidCommand(code=code, message=message, field="arguments")
        )
        self.host.reject(tool, error)
        return Answer(error_view(page, url or f"/shop/{page}", code, message))

    def _reopen(self, tool: str, cart: Cart, session: str, base: str) -> Answer | None:
        """A cart edited after checkout started: the open checkout is canceled and the edit goes through, as a
        storefront's cart page lets a buyer go back and change the order; None when it did. A checkout that is
        completing (a 3-D Secure challenge outstanding) stays locked. Why: on 2026-10-16, 29 webmcp rows built to pay
        met checkout_started on remove_cart_line, add_to_cart or update_cart_line after the shopper refused the
        item, and with no tool to cancel the agent could not swap it."""
        ck = self.state.checkouts.get(cart.checkout_id) if cart.checkout_id else None
        if ck is None or ck.status not in OPEN_CHECKOUT:
            return self._locked(tool, cart)
        reply = self._ucp("cancel_checkout", session, base, idempotency=f"reopen-{ck.id}", id=ck.id)
        if reply.fault is not None or reply.doc is None or reply.is_error:
            return self._locked(tool, cart)
        return None

    def _locked(self, tool: str, cart: Cart) -> Answer:
        text = "Checkout has started for this cart: continue with get_checkout and place_order"
        return self._refuse(tool, "checkout_started", text, page="cart", url="/shop/cart")

    def _fault(self, reply: Reply, page: str, url: str) -> Answer:
        fault = reply.fault
        if fault is None:
            return Answer(error_view(page, url, "store_error", "The store did not answer"), status=500)
        view = error_view(page, url, fault.reason, fault.message)
        if fault.retry_after is not None:
            view["error"]["retry_after"] = fault.retry_after
        return Answer(view, status=fault.status)


def checkout_view_empty() -> Json:
    return {"page": "checkout", "url": "/shop/checkout", "checkout": None, "missing": []}


def payment_handlers(ops: UcpOperations) -> list[str]:
    """The payment methods the page offers: the handler entries the store declares (their ids, as its discovery
    document lists them), its card handler first and the sandbox's own handler last."""
    by_namespace = ops.envelope.handlers
    order = sorted(by_namespace, key=lambda ns: (ns != CARD_NAMESPACE, ns == FAKE_NAMESPACE, ns))
    return [str(e["id"]) for ns in order for e in by_namespace[ns]]


def _line_items(lines: Sequence[Line]) -> list[dict[str, Any]]:
    """A cart's or checkout's lines as a UCP update sends them back: every line, by id and item, with its quantity."""
    return [{"id": ln.id, "item": {"id": ln.variant_id}, "quantity": ln.quantity} for ln in lines]


def _method_lines(ck: Checkout) -> dict[str, Any]:
    """The lines a fulfillment method covers, named as the UCP releases that require them want
    (`line_item_ids`): the first group's when the checkout has one, else every line. Without it the Shopify pack's
    UCP refused every set_shipping_address and choose_fulfillment ("'line_item_ids' is required"), so no
    Shopify-pack WebMCP row of 2026-10-15 placed an order."""
    ids = list(ck.groups[0].line_ids) if ck.groups and ck.groups[0].line_ids else [ln.id for ln in ck.lines]
    return {"line_item_ids": ids}


def _failure(doc: Mapping[str, Any]) -> Json | None:
    """What went wrong with a checkout call: the first error that is not a field still to fill in. A platform pack
    may name a field still to fill in its own words (WooCommerce's `missing_email`, `missing_shipping_address`), so
    every `missing_` code is readiness too; read as failures they made 17 Woo-pack start_checkout calls and 14
    set_buyer_contact calls of 2026-10-15 look refused while the checkout simply waited for the next form."""
    for m in doc.get("messages") or ():
        if not isinstance(m, Mapping) or m.get("type") != "error":
            continue
        code = str(m.get("code") or "error")
        if code in ("missing", "out_of_stock") or code.startswith("missing_"):
            continue
        return {"code": code, "message": str(m.get("content") or "")}
    return None


def _payload_requires(ops: UcpOperations, tool: str, member: str) -> bool:
    """Whether the served release's tool requires `member` in its payload (UCP 2026-04-08's update_cart needs the
    cart's id there as well as in `id`)."""
    spec = ops.book.spec(tool)
    key = ops.book.payload_key(tool)
    if spec is None or key is None:
        return False
    payload = spec.input_schema.get("properties", {}).get(key, {})
    return member in (payload.get("required") or ())


def next_page(doc: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The upstream pagination of a search's next page, or None on the last: UCP's `cursor` (when `has_next_page`),
    or a pack's offset paging (Medusa's `has_more`, `offset`, `limit`), whose next page is asked for by offset. The
    storefront had read only UCP's, so it offered no second page under the Medusa pack."""
    paging = doc.get("pagination")
    if not isinstance(paging, Mapping):
        return None
    if paging.get("has_next_page") and paging.get("cursor") is not None:
        return {"cursor": str(paging["cursor"])}
    offset, limit = paging.get("offset"), paging.get("limit")
    if paging.get("has_more") is not True or not isinstance(offset, int) or not isinstance(limit, int):
        return None
    if isinstance(offset, bool) or isinstance(limit, bool) or limit <= 0:
        return None
    return {"offset": offset + limit, "limit": limit}


def _problem(schema: Mapping[str, Any], args: Mapping[str, Any]) -> str | None:
    """The first way `args` fails the tool's schema, in words taken from the schema, never from the input."""
    validator = jsonschema.Draft202012Validator(dict(schema))
    errors = sorted(validator.iter_errors(dict(args)), key=lambda e: (list(e.absolute_path), e.validator))
    if not errors:
        return None
    err = errors[0]
    where = ".".join(str(p) for p in err.absolute_path) or "input"
    if err.validator == "required":
        missing = [r for r in err.validator_value if r not in args]
        return f"{', '.join(missing) or where} is required"
    if err.validator == "additionalProperties":
        extra = sorted(set(args) - set(schema.get("properties", {})))
        return f"this tool takes no {', '.join(extra)}"
    return f"{where} must satisfy {err.validator} {json_text(err.validator_value)}"


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))[:200]


def _one_product(doc: Mapping[str, Any], ident: str) -> Mapping[str, Any] | None:
    """The product a get_product answer holds. UCP answers with `product`; a platform pack may answer with a
    `products` list instead (Medusa's plugin formats get_product as a lookup, ucp-2026-08-25 06), so the storefront
    takes the listed product with this id, or the only one listed. Without this every Medusa-pack product page read
    "No product has this id"."""
    one = doc.get("product")
    if isinstance(one, Mapping):
        return one
    listed = [p for p in doc.get("products") or () if isinstance(p, Mapping)]
    match = [p for p in listed if p.get("id") == ident or p.get("handle") == ident]
    if match:
        return match[0]
    return listed[0] if len(listed) == 1 else None

"""Core state to UCP documents: carts, checkouts, orders and error responses, in one release's dialect.

Why from state. A document is rendered from the state the command left, never from the command, so a get after an
update, a replayed idempotent request and a document built after a fault all show the same store. Messages come from
two places: the command's own events and refusal (passed in by the operation), and facts the state shows whatever
the command was (what an incomplete checkout still lacks, a last payment attempt that failed, a pending 3-D Secure
challenge), so a `get_checkout` explains the checkout as well as the call that changed it.

Release differences handled here (`UcpDialect`): typed destinations and description objects in 2026-08-25, and the
3-D Secure flow. The core parks a challenged payment as `requires_escalation` with `payment_action_required`. When the
payment authentication extension is active (2026-08-25, declared), that is a `complete_in_progress` checkout carrying
a `dev.ucp.common.payment.three_ds_challenge` Action whose URL is the sandbox's challenge page; otherwise it is a
buyer handoff, `requires_escalation` with `continue_url` at the challenge page and a `requires_buyer_input` message,
which is how 2026-04-08 (and a 2026-08-25 store that does not declare the extension) must say it.

Fulfillment is one method per core fulfillment group, each with its one group, so a method id (`fm-<group>`) and its
group id map back to the core group exactly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from carttrace.core.money import to_major
from carttrace.sandbox.core import brokers, pricing
from carttrace.sandbox.core.config import SHOP_PAY_NAMESPACE, Coupon
from carttrace.sandbox.core.market import Market
from carttrace.sandbox.core.state import Cart, Checkout, Group, Line, Order, Payment, State, Totals
from carttrace.sandbox.platforms import GENERIC, Platform
from carttrace.sandbox.surfaces.ucp.catalog import Catalog
from carttrace.sandbox.surfaces.ucp.dialect import THREE_DS_ACTION, ResponseKind, UcpDialect
from carttrace.sandbox.surfaces.ucp.envelope import Envelope
from carttrace.sandbox.surfaces.ucp.ids import CartIds
from carttrace.sandbox.surfaces.ucp.messages import PLAIN_WORDING, Message, Wording, error, info

METHOD_PREFIX = "fm-"
ADDRESS_FIELDS = (
    "first_name",
    "last_name",
    "street_address",
    "extended_address",
    "address_locality",
    "address_region",
    "postal_code",
    "address_country",
)
TERMINAL = ("completed", "canceled")
FAILED_PAYMENT = {
    "declined": "Card declined",
    "authentication_failed": "The buyer's bank did not authenticate the payment",
}
DECLINE_WORDS = {
    "insufficient_funds": "insufficient funds",
    "do_not_honor": "the issuer did not honor it",
    "expired_card": "the card has expired",
    "processing_error": "a processing error",
    "generic_decline": "the issuer declined it",
    "challenge_abandoned": "the buyer did not complete the challenge",
    "authentication_failed": "the challenge failed",
    "shop_token_used": "the Shop Token was already used",
    "shop_token_checkout_mismatch": "the Shop Token is not for this checkout",
    "shop_token_expired": "the Shop Token has expired",
}
"""A decline code in words (the sandbox's own phrasing of the PSP's vocabulary; the Shop Token entries are the spec's
three token rules, D73)."""


def method_id(group_id: str) -> str:
    return f"{METHOD_PREFIX}{group_id}"


def group_of_method(method: str) -> str:
    """The core group a rendered method id names (`fm-g1` is `g1`); other ids name a group directly."""
    return method.removeprefix(METHOD_PREFIX)


class Renderer:
    def __init__(
        self,
        dialect: UcpDialect,
        envelope: Envelope,
        catalog: Catalog,
        market: Market,
        codes: Mapping[str, int],
        *,
        platform: Platform = GENERIC,
        ids: CartIds | None = None,
        coupons: Mapping[str, Coupon] | None = None,
        wording: Wording = PLAIN_WORDING,
        psp: str | None = None,
    ):
        """`codes` is the pack's discount table (code to percent off), for each applied code's share; `coupons` the
        codes' rules where a pack has more than percentages. `platform` and `ids` decide how a cart is spelled on the
        wire (carttrace.sandbox.platforms, carttrace.sandbox.surfaces.ucp.ids); `wording` is the platform's own words."""
        self.dialect = dialect
        self.envelope = envelope
        self.catalog = catalog
        self.market = market
        self.codes = codes
        self.coupons: Mapping[str, Coupon] = (
            coupons if coupons is not None else {c: Coupon(c, "percent", int(p)) for c, p in codes.items()}
        )
        self.platform = platform
        self.ids = ids or CartIds("", keyed=False)
        self.wording = wording
        self.psp = psp
        """The payment broker behind the store's PSP (D71), whose words a failed payment's message adds."""

    # -- shared pieces -------------------------------------------------------------------------------------------------

    def line_items(self, state: State, lines: Sequence[Line], *, cart_id: str | None = None) -> list[dict[str, Any]]:
        """Line items; a cart's (`cart_id` given) carry the platform's line ids."""
        return [
            {
                "id": self.ids.wire_line(ln.id, cart_id) if cart_id is not None else ln.id,
                "item": self._item(state, ln.variant_id, ln.unit_price),
                "quantity": ln.quantity,
                "totals": _line_totals(ln.unit_price * ln.quantity, labelled=self.platform.line_display_text),
            }
            for ln in lines
        ]

    def _item(self, state: State, variant_id: str, unit_price: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": variant_id,
            "title": self.catalog.variant_title(state, variant_id),
            "price": unit_price,
        }
        image = self.catalog.image(variant_id)
        if image is not None:
            item["image_url"] = image
        return item

    def totals(self, t: Totals, *, cart: bool = False, label: str = "Shipping") -> list[dict[str, Any]]:
        """UCP totals: exactly one subtotal and one total, with discount (negative), fulfillment and tax between."""
        out: list[dict[str, Any]] = [{"type": "subtotal", "display_text": "Subtotal", "amount": t.subtotal}]
        if t.discount:
            amount = -t.discount if self.dialect.signed_discount else t.discount
            out.append({"type": "discount", "display_text": "Discount", "amount": amount})
        if not cart and t.fulfillment is not None:
            out.append({"type": "fulfillment", "display_text": label, "amount": t.fulfillment})
        if t.tax is not None:
            text = "Tax included in prices" if self.market.tax_inclusive else "Tax"
            out.append({"type": "tax", "display_text": text, "amount": t.tax})
        out.append({"type": "total", "display_text": "Total", "amount": t.total})
        return out

    def links(self, base: str) -> list[dict[str, Any]]:
        """The policy links the platform's carts and checkouts carry (seven types on live Shopify)."""
        return [
            {"type": kind, "url": f"{base}/policies/{kind.replace('_', '-')}", "title": title}
            for kind, title in self.platform.links
        ]

    def discounts(self, codes: Sequence[str], discount: int, subtotal: int = 0) -> dict[str, Any]:
        """`discounts.codes` as applied, and each code's share of the discount: split by percentage when every code is
        a percent one, else each code's own amount (a fixed amount, nothing for free shipping)."""
        amounts = pricing.shares(codes, discount, subtotal, self.coupons)
        applied = [
            {"code": code, "title": f"{code} ({self._says(code)})", "amount": share}
            for code, share in zip(codes, amounts, strict=True)
        ]
        return {"codes": list(codes), "applied": applied}

    def _says(self, code: str) -> str:
        rule = self.coupons.get(code, Coupon(code))
        if rule.kind == "fixed_cart":
            return f"{to_major(rule.amount, self.market.currency)} off"
        if rule.kind == "free_shipping":
            return "free shipping"
        return f"{rule.amount}% off"

    def order_notes(self, state: State, order_id: str | None) -> list[Message]:
        """The order's status in the platform's own names (WooCommerce: processing, on-hold), where it has them."""
        order = state.orders.get(order_id) if order_id is not None else None
        name = self.platform.order_states.get(order.status) if order is not None else None
        if name is None:
            return []
        text = self.wording.say("order.status", status=name) or f"Order status: {name}"
        return [info(text, code="order_status", path="$.order")]

    def said(self, messages: Sequence[Message]) -> list[Message]:
        """Messages as the platform words them: live Shopify marks every error and warning `content_type: plain`."""
        if not self.platform.plain_content_type:
            return list(messages)
        return [{**m, "content_type": "plain"} if m.get("type") in ("error", "warning") else m for m in messages]

    # -- documents -----------------------------------------------------------------------------------------------------

    def cart(self, state: State, cart: Cart, messages: Sequence[Message], base: str) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "ucp": self.envelope.block("cart"),
            "id": self.ids.wire_cart(cart.id),
            "line_items": self.line_items(state, cart.lines, cart_id=cart.id),
            "currency": state.currency,
            "totals": self.totals(cart.totals, cart=True),
            "links": self.links(base),
        }
        if cart.buyer:
            doc["buyer"] = dict(cart.buyer)
        if self.envelope.active("dev.ucp.shopping.discount") and (self.platform.discounts_block or cart.codes):
            doc["discounts"] = self.discounts(cart.codes, cart.totals.discount, cart.totals.subtotal)
        if messages or self.platform.messages_always:
            doc["messages"] = self.said(messages)
        if cart.status != "canceled":
            doc["continue_url"] = f"{base}{self.ids.continue_path(cart.id)}"
        days = self.platform.cart_expires_days
        if days is not None:
            doc["expires_at"] = _rfc3339(cart.created_at + days * DAY_MS)
        return doc

    def checkout(self, state: State, ck: Checkout, messages: Sequence[Message], base: str) -> dict[str, Any]:
        pay = state.payments.get(ck.payment_id) if ck.payment_id else None
        status = ck.status
        continue_url: str | None = None if ck.status in TERMINAL else f"{base}/checkout/{ck.id}"
        notes: list[Message] = []
        actions: dict[str, Any] | None = None
        challenge = _pending_challenge(ck, pay)
        if challenge is not None and pay is not None:
            page = f"{base}/3ds/{challenge}"
            if self.envelope.actions:
                status = "complete_in_progress"
                at = f"$.actions['{THREE_DS_ACTION}'][0]"
                actions = {
                    THREE_DS_ACTION: [{"id": challenge, "config": {"payment_instrument_id": pay.id, "url": page}}]
                }
                notes.append(
                    info(
                        "The buyer's bank asks for a 3-D Secure challenge: present it, then get the checkout",
                        code="payment_authentication",
                        path=at,
                    )
                )
            else:
                continue_url = page
                notes.append(
                    error(
                        "payment_authentication_required",
                        "The buyer's bank asks for a 3-D Secure challenge; hand off to the buyer at continue_url",
                        severity="requires_buyer_input",
                    )
                )
        elif ck.status == "requires_escalation" and ck.escalation == "payment_handler_unavailable":
            notes.append(
                error(
                    "payment_handler_unavailable",
                    "The payment handler named is not served here; pay through a declared handler or hand off to the "
                    "buyer at continue_url",
                    severity="requires_buyer_input",
                    path="$.payment.instruments[0].handler_id",
                )
            )
        notes += self._readiness(state, ck) + _payment_failure(ck, pay, self.psp) + self._pickup_note(ck.groups)
        doc: dict[str, Any] = {
            "ucp": self.envelope.block("checkout"),
            "id": ck.id,
            "status": status,
            "currency": state.currency,
            "line_items": self.line_items(state, ck.lines),
            "totals": self.totals(ck.totals, label=_fulfillment_label(ck.groups)),
            "links": self.links(base),
        }
        if ck.buyer:
            doc["buyer"] = dict(ck.buyer)
        fulfillment = self.fulfillment(ck.groups)
        if fulfillment is not None:
            doc["fulfillment"] = fulfillment
        if self.envelope.active("dev.ucp.shopping.discount") and (self.platform.checkout_discounts or ck.codes):
            doc["discounts"] = self.discounts(ck.codes, ck.totals.discount, ck.totals.subtotal)
        payment = self._payment(pay)
        if payment is not None:
            doc["payment"] = payment
        if ck.order_id is not None and ck.status == "completed":
            order = state.orders.get(ck.order_id)
            doc["order"] = {
                "id": ck.order_id,
                "label": f"#{order.number}" if order is not None else ck.order_id,
                "permalink_url": f"{base}/orders/{ck.order_id}",
            }
            notes += self.order_notes(state, ck.order_id)
        if actions is not None:
            doc["actions"] = actions
        merged = _unique([*messages, *notes])
        if merged or self.platform.messages_always:
            doc["messages"] = self.said(merged)
        if continue_url is not None:
            doc["continue_url"] = continue_url
        return doc

    def _payment(self, pay: Payment | None) -> dict[str, Any] | None:
        """The checkout's `payment`: its last attempt's instrument; at 2026-01-11 the handlers too, and always."""
        if self.dialect.handlers_in_payment:
            listed: dict[str, Any] = {"handlers": self.envelope.payment_handlers()}
            if pay is not None:
                listed.update(selected_instrument_id=pay.id, instruments=[_instrument(pay, flag=False)])
            return listed
        return {"instruments": [_instrument(pay)]} if pay is not None else None

    def fulfillment(self, groups: Sequence[Group]) -> dict[str, Any] | None:
        methods = [self._method(g) for g in groups]
        return {"methods": methods} if methods else None

    def _method(self, g: Group) -> dict[str, Any]:
        m: dict[str, Any] = {"id": method_id(g.id), "type": g.method, "line_item_ids": list(g.line_ids)}
        if g.method == "shipping" and g.destination:
            dest = {k: str(g.destination[k]) for k in ADDRESS_FIELDS if g.destination.get(k)}
            dest_id = str(g.destination.get("id") or "d1")
            m["destinations"] = [{**self.dialect.destination("shipping_address"), "id": dest_id, **dest}]
            m["selected_destination_id"] = dest_id
        elif g.method == "pickup":
            if self.dialect.pickup_destinations:
                m["destinations"] = [
                    {
                        **self.dialect.destination("business_location"),
                        "id": p.id,
                        "name": p.title,
                        "address": {"address_country": p.address_country},
                    }
                    for p in self.market.pickup
                ]
            if g.location_id is not None:
                m["selected_destination_id"] = g.location_id
        group: dict[str, Any] = {
            "id": g.id,
            "line_item_ids": list(g.line_ids),
            "options": [self._option(o) for o in g.options],
        }
        if g.selected_option_id is not None:
            group["selected_option_id"] = g.selected_option_id
        m["groups"] = [group]
        return m

    def _option(self, o: Mapping[str, Any]) -> dict[str, Any]:
        carrier = str(o.get("carrier") or "")
        text = "Collect at the store counter" if carrier == "pickup" else f"Ships with {carrier}"
        out: dict[str, Any] = {
            "id": str(o["id"]),
            "title": str(o["title"]),
            "description": self.dialect.option_text(text),
            "totals": [{"type": "total", "amount": int(o["amount"])}],
        }
        if carrier and carrier != "pickup":
            out["carrier"] = carrier
        return out

    def _pickup_note(self, groups: Sequence[Group]) -> list[Message]:
        """Where a release cannot list pickup locations as destinations, say which ids `selected_destination_id`
        takes."""
        if self.dialect.pickup_destinations or not self.market.pickup:
            return []
        names = ", ".join(f"{p.id} ({p.title})" for p in self.market.pickup)
        return [
            info(f"Pickup locations: {names}", code="pickup_locations", path=f"$.fulfillment.methods[{i}]")
            for i, g in enumerate(groups)
            if g.method == "pickup"
        ]

    def _readiness(self, state: State, ck: Checkout) -> list[Message]:
        """What an incomplete checkout still lacks, each pointed at the field to send (UCP "missing")."""
        if ck.status != "incomplete":
            return []
        out: list[Message] = []
        if not ck.lines:
            out.append(error("missing", "A checkout needs at least one line item", path="$.line_items"))
        if not ck.buyer.get("email"):
            out.append(self._missing("email", "Buyer email is required", "$.buyer.email"))
        if not ck.groups:
            text = "Choose shipping or pickup for the line items"
            out.append(self._missing("fulfillment", text, "$.fulfillment.methods", at="$.fulfillment.methods[0]"))
        for i, g in enumerate(ck.groups):
            at = f"$.fulfillment.methods[{i}]"
            if g.method == "shipping" and g.line_ids and not g.destination:
                out.append(self._missing("address", "A shipping address is required", f"{at}.destinations", at=at))
            elif g.line_ids and g.amount is None:
                out.append(error("missing", "Select a fulfillment option", path=f"{at}.groups[0].selected_option_id"))
        covered = {x for g in ck.groups for x in g.line_ids}
        if ck.groups and any(ln.id not in covered for ln in ck.lines):
            out.append(error("missing", "Some line items have no fulfillment method", path="$.fulfillment.methods"))
        res = state.reservations.get(ck.reservation_id) if ck.reservation_id else None
        if ck.lines and (res is None or res.status != "held"):
            out.append(error("out_of_stock", "The stock for this checkout could not be held", path="$.line_items"))
        return out

    def _missing(self, key: str, text: str, path: str, *, at: str = "") -> Message:
        """A readiness message: UCP's `missing`, or the platform's own code, words and field (WooCommerce's UCP
        plugin: `missing_email`, `missing_shipping_address`, captures/diff-woo 142)."""
        own = self.platform.readiness.get(key)
        if own is None:
            return error("missing", text, path=path)
        code, words, where = own
        return error(code, words, path=where.format(at=at))

    def order(self, state: State, o: Order, base: str) -> dict[str, Any]:
        lines = [
            {
                "id": ln.id,
                "item": self._item(state, ln.variant_id, ln.unit_price),
                "quantity": {"original": ln.quantity, "total": ln.quantity - ln.returned, "fulfilled": ln.fulfilled},
                "totals": _line_totals(ln.unit_price * ln.quantity),
                "status": _line_status(ln.quantity - ln.returned, ln.fulfilled),
            }
            for ln in o.lines
        ]
        doc: dict[str, Any] = {
            "ucp": self.envelope.block("order"),
            "id": o.id,
            "label": f"#{o.number}",
            "checkout_id": o.checkout_id,
            "permalink_url": f"{base}/orders/{o.id}",
            "line_items": lines,
            "fulfillment": {"expectations": [self._expectation(g, o) for g in o.groups if g.line_ids], "events": []},
            "currency": o.currency,
            "totals": self.totals(o.totals, label=_fulfillment_label(o.groups)),
        }
        notes = [{**m, "path": "$"} for m in self.order_notes(state, o.id)]
        if notes:
            doc["messages"] = self.said(notes)
        return doc

    def _expectation(self, g: Group, o: Order) -> dict[str, Any]:
        qty = {ln.id: ln.quantity - ln.returned for ln in o.lines}
        if g.method == "shipping" and g.destination:
            destination = {k: str(g.destination[k]) for k in ADDRESS_FIELDS if g.destination.get(k)}
        else:
            loc = self.market.location(g.location_id or "")
            destination = {"address_country": loc.address_country if loc is not None else self.market.code}
        out: dict[str, Any] = {
            "id": g.id,
            "line_items": [{"id": x, "quantity": qty[x]} for x in g.line_ids if qty.get(x, 0) > 0],
            "method_type": g.method,
            "destination": destination,
        }
        chosen = next((opt for opt in g.options if opt.get("id") == g.selected_option_id), None)
        if chosen is not None:
            out["description"] = str(chosen.get("title"))
        return out

    def error(self, kind: ResponseKind, messages: Sequence[Message], base: str) -> dict[str, Any]:
        """An error response: no resource could be established; `ucp.status` is "error"."""
        return {
            "ucp": self.envelope.block(kind, status="error"),
            "messages": self.said(messages),
            "continue_url": f"{base}/",
        }


def _line_totals(amount: int, *, labelled: bool = False) -> list[dict[str, Any]]:
    if labelled:
        return [
            {"type": "subtotal", "display_text": "Subtotal", "amount": amount},
            {"type": "total", "display_text": "Total", "amount": amount},
        ]
    return [{"type": "subtotal", "amount": amount}, {"type": "total", "amount": amount}]


DAY_MS = 24 * 60 * 60 * 1000


def _rfc3339(ms: int) -> str:
    """Logical milliseconds as live Shopify spells a time: whole seconds, UTC, `Z` (`2026-11-11T20:52:45Z`)."""
    return datetime.fromtimestamp(ms // 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _line_status(total: int, fulfilled: int) -> str:
    if total == 0:
        return "removed"
    if fulfilled >= total:
        return "fulfilled"
    return "partial" if fulfilled > 0 else "processing"


def _fulfillment_label(groups: Sequence[Group]) -> str:
    methods = {g.method for g in groups}
    return "Pickup" if methods == {"pickup"} else "Shipping"


def _pending_challenge(ck: Checkout, pay: Payment | None) -> str | None:
    waiting = ck.status == "requires_escalation" and ck.escalation == "payment_action_required"
    if waiting and pay is not None and pay.status == "requires_action":
        return pay.challenge_id
    return None


def _payment_failure(ck: Checkout, pay: Payment | None, psp: str | None = None) -> list[Message]:
    """The last payment attempt failed and the checkout is waiting for another one; with a broker behind the PSP
    (D71), what the broker said, by its own code and words. A Shop Pay payment is no card (D73), so its failure does
    not say one."""
    if pay is None or ck.status != "ready_for_complete" or pay.status not in FAILED_PAYMENT:
        return []
    why = DECLINE_WORDS.get(pay.decline_code or "", "")
    head = FAILED_PAYMENT[pay.status]
    if pay.status == "declined" and pay.namespace == SHOP_PAY_NAMESPACE:
        head = "Payment declined"
    text = head + (f": {why}" if why else "")
    said = brokers.decline_details(psp, pay.decline_code or "generic_decline", authentication=pay.authentication)
    if said:
        text += f" ({brokers.broker(said['processor']).title} {said['processor_code']}: {said['processor_message']})"
    return [error("payment_failed", text, path="$.payment.instruments[0]")]


def _instrument(pay: Payment, *, flag: bool = True) -> dict[str, Any]:
    """The instrument of the checkout's payment attempt, display-safe: never its credential. `flag` marks it
    `selected`; 2026-01-11 names the selected instrument in `payment.selected_instrument_id` instead, and its card
    instrument requires `brand` and `last_digits`, which is empty: an opaque test token has no card number behind it."""
    out: dict[str, Any] = {"id": pay.id, "handler_id": pay.handler, "type": pay.instrument or "card"}
    if flag:
        out["selected"] = True
    else:
        out.update(brand=pay.brand or "unknown", last_digits="")
    if pay.brand:
        out["display"] = {"brand": pay.brand}
    return out


def _unique(messages: Sequence[Message]) -> list[Message]:
    seen: set[tuple[Any, ...]] = set()
    out: list[Message] = []
    for m in messages:
        key = (m.get("type"), m.get("code"), m.get("path"))
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out

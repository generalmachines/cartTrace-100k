"""UCP messages and protocol faults: how the core's events and refusals reach an agent.

Two layers, as UCP defines them (overview, "Error Handling"; checkout, cart and catalog MCP bindings):
- Protocol faults stop a request before it is processed: a rate limit or unavailability, an idempotency key reused
  with another payload (409), a session whose trust tier or permissions do not allow the operation (401, 403), or
  the sandbox's own invariant failure (500). Each transport answers these itself: an HTTP status plus a JSON-RPC error
  (-32000, or -32603 for a 500) over MCP, an HTTP status with a `{code, content}` body over REST.
- Business outcomes are documents: the resource with `messages`, or an error response with `ucp.status: "error"`.
  Codes follow the UCP standard ones where one fits (`not_found`, `out_of_stock`, `address_undeliverable`,
  `payment_failed`, `missing`, the discount extension's `discount_code_*`); elsewhere the core's own code is kept, since
  it already says what happened. Severity tells the platform what to do next.

Nothing here echoes a value the agent sent except a discount code, which the surface guard has already checked for
card numbers; item ids are pointed at by `path` instead of being repeated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

Severity = Literal["recoverable", "requires_buyer_input", "requires_buyer_review", "unrecoverable"]
Message = dict[str, Any]


def error(code: str, content: str, *, severity: Severity = "recoverable", path: str | None = None) -> Message:
    out: Message = {"type": "error", "code": code, "content": content, "severity": severity}
    if path is not None:
        out["path"] = path
    return out


def warning(code: str, content: str, *, path: str | None = None) -> Message:
    out: Message = {"type": "warning", "code": code, "content": content}
    if path is not None:
        out["path"] = path
    return out


def info(content: str, *, code: str | None = None, path: str | None = None) -> Message:
    out: Message = {"type": "info", "content": content}
    if code is not None:
        out["code"] = code
    if path is not None:
        out["path"] = path
    return out


# -- protocol faults ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fault:
    """A refusal the transport answers itself, never as a UCP document."""

    status: int
    """HTTP status: the primary signal on both transports."""
    rpc_code: int
    """JSON-RPC error code over MCP: -32000, or -32603 for a 500."""
    message: str
    reason: str
    """A short machine code: rate_limited, unavailable, conflict, idempotency_conflict, unauthorized, forbidden,
    internal, request_too_large."""
    retry_after: int | None = None
    bare: bool = False
    """Answered with no `data`, as live Shopify answers an internal error (store-398 ep-7b185304b778 seq 32, D70)."""

    def data(self) -> dict[str, Any] | None:
        if self.bare:
            return None
        out: dict[str, Any] = {"code": self.reason, "content": self.message}
        if self.retry_after is not None:
            out["retry_after"] = self.retry_after
        return out


PROTOCOL_ERROR = -32000
INTERNAL_ERROR = -32603
INVALID_PARAMS = -32602


def fault_for(err: Mapping[str, Any] | None, *, tier: str | None) -> Fault | None:
    """The protocol fault a core refusal maps to (UCP overview, "Protocol Errors"), or None for a business outcome.

    `tier` is the acting session's trust tier: an anonymous agent asked for a token-tier operation was never
    authenticated (401), while a signed or token agent without the permission was (403)."""
    if err is None:
        return None
    build = FAULTS.get(str(err.get("kind")))
    return build(err, tier) if build is not None else None


def _detail(err: Mapping[str, Any], key: str) -> int | None:
    return next((int(d[key]) for d in err.get("details") or () if isinstance(d, Mapping) and key in d), None)


def _unavailable(err: Mapping[str, Any], tier: str | None) -> Fault:
    if err.get("code") == "rate_limited":
        return Fault(429, PROTOCOL_ERROR, "Rate limit exceeded", "rate_limited", _detail(err, "retry_after"))
    status = _detail(err, "status") or 503
    rpc = INTERNAL_ERROR if status == 500 else PROTOCOL_ERROR
    return Fault(status, rpc, str(err.get("message") or "Service unavailable"), "unavailable")


def _conflict(err: Mapping[str, Any], tier: str | None) -> Fault:
    reason = "idempotency_conflict" if err.get("code") == "idempotency_conflict" else "conflict"
    return Fault(409, PROTOCOL_ERROR, str(err.get("message") or "Conflict"), reason)


def _denied(err: Mapping[str, Any], tier: str | None) -> Fault:
    code = err.get("code")
    text = str(err.get("message") or "")
    if code == "token_revoked" or (code == "tier_insufficient" and tier in (None, "anonymous")):
        return Fault(401, PROTOCOL_ERROR, text or "Unauthorized", "unauthorized")
    return Fault(403, PROTOCOL_ERROR, text or "Forbidden", "forbidden")


def _internal(err: Mapping[str, Any], tier: str | None) -> Fault:
    return Fault(500, INTERNAL_ERROR, "Internal error", "internal")


FAULTS: Mapping[str, Callable[[Mapping[str, Any], str | None], Fault]] = {
    "unavailable": _unavailable,
    "conflict": _conflict,
    "policy_denied": _denied,
    "invariant": _internal,
}
"""Core refusal kinds that are protocol faults, by kind. Every other kind is a business outcome."""


def batch_too_large(limit: int) -> Fault:
    """A lookup over the batch limit (catalog lookup, "Batch Size"): -32602 over MCP, 400 over REST."""
    return Fault(400, INVALID_PARAMS, f"at most {limit} identifiers per lookup", "request_too_large")


# -- business refusals -------------------------------------------------------------------------------------------------

REFUSALS: Mapping[str, tuple[str, Severity]] = {
    "cart_not_found": ("not_found", "unrecoverable"),
    "cart_canceled": ("not_found", "unrecoverable"),
    "checkout_not_found": ("not_found", "unrecoverable"),
    "order_not_found": ("not_found", "unrecoverable"),
    "empty_cart": ("missing", "recoverable"),
    "merchandise_out_of_stock": ("out_of_stock", "recoverable"),
    "payment_declined": ("payment_failed", "recoverable"),
    "payment_authentication_failed": ("payment_failed", "recoverable"),
    "payment_handler_unavailable": ("payment_handler_unavailable", "requires_buyer_input"),
    "checkout_locked": ("checkout_locked", "unrecoverable"),
    "checkout_completed": ("checkout_completed", "unrecoverable"),
    "checkout_canceled": ("checkout_canceled", "unrecoverable"),
}
"""Core codes that have a UCP name or a fixed severity. Any other core code is kept as is, recoverable."""

FIELD_PATHS: Mapping[str, str] = {
    "lines": "$.line_items",
    "codes": "$.discounts.codes",
    "country": "$.context.address_country",
    "fulfillment": "$.fulfillment.methods",
    "payment.token": "$.payment.instruments[0].credential.token",
    "payment.instrument": "$.payment.instruments[0].type",
    "payment.handler": "$.payment.instruments[0].handler_id",
    "status": "$.status",
}
"""Where a core refusal's field points in a UCP request, for fields named whole."""
LINE_FIELDS: Mapping[str, str] = {"quantity": ".quantity", "variant_id": ".item.id", "line_id": ".id"}
GROUP_FIELDS: Mapping[str, str] = {
    "destination": ".destinations[0]",
    "selected_option_id": ".groups[0].selected_option_id",
    "location_id": ".selected_destination_id",
    "line_ids": ".line_item_ids",
}


def ucp_path(field: str) -> str | None:
    """The JSONPath in a UCP request that a core command's field (`lines.0.quantity`, `buyer.email`) came from."""
    if field in FIELD_PATHS:
        return FIELD_PATHS[field]
    parts = field.split(".")
    head, rest = parts[0], parts[1:]
    if head == "buyer" and len(rest) == 1:
        return f"$.buyer.{rest[0]}"
    if head == "codes" and len(rest) == 1 and rest[0].isdigit():
        return f"$.discounts.codes[{rest[0]}]"
    if head == "lines" and rest and rest[0].isdigit():
        return f"$.line_items[{rest[0]}]" + (LINE_FIELDS.get(rest[1], "") if len(rest) > 1 else "")
    if head == "fulfillment" and rest and rest[0].isdigit():
        tail = GROUP_FIELDS.get(rest[1], "") if len(rest) > 1 else ""
        extra = f".{rest[2]}" if len(rest) > 2 and rest[1] == "destination" else ""
        return f"$.fulfillment.methods[{rest[0]}]{tail}{extra}"
    return None


def refusal(err: Mapping[str, Any], *, path: str | None = None) -> Message:
    """A core refusal as a UCP error message. `path` overrides the one its field implies. A malformed field the core
    refused is named the way UCP stores name it (`invalid_token` for `payment.token`)."""
    code = str(err.get("code"))
    ucp_code, severity = REFUSALS.get(code, (code, "recoverable"))
    field = err.get("field")
    if err.get("kind") == "validation" and code == "invalid_argument" and isinstance(field, str):
        last = field.rsplit(".", 1)[-1]
        ucp_code = f"invalid_{last}" if last.isidentifier() and not last.isdigit() else code
    where = path if path is not None else ucp_path(field) if isinstance(field, str) else None
    return error(ucp_code, _plain(str(err.get("message") or code)), severity=severity, path=where)


def _plain(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


# -- events ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Paths:
    """Where an event points in the request that caused it: each request line by its line id or item id, each
    discount code by its text, each fulfillment method by the core group it became."""

    lines: tuple[tuple[str | None, str | None], ...] = ()
    """(line id, item id) of each request line item, in request order."""
    codes: tuple[str, ...] = ()
    methods: tuple[str | None, ...] = ()
    """The core group id each request fulfillment method maps to, in request order (None for a new group)."""

    def line(self, *, line_id: str | None = None, variant_id: str | None = None) -> str | None:
        for i, (lid, vid) in enumerate(self.lines):
            if (line_id is not None and lid == line_id) or (line_id is None and vid is not None and vid == variant_id):
                return f"$.line_items[{i}]"
        return None

    def code(self, code: str) -> str | None:
        want = code.strip().upper()
        for i, c in enumerate(self.codes):
            if c.strip().upper() == want:
                return f"$.discounts.codes[{i}]"
        return None

    def method(self, group_id: str | None) -> str:
        for i, gid in enumerate(self.methods):
            if gid is not None and gid == group_id:
                return f"$.fulfillment.methods[{i}]"
        return "$.fulfillment.methods[0]" if self.methods else "$.fulfillment"


DISCOUNT_REASONS: Mapping[str, tuple[str, str]] = {
    "invalid": ("discount_code_invalid", "is not a valid code here"),
    "duplicate": ("discount_code_already_applied", "is already applied"),
    "not_combinable": ("discount_code_combination_disallowed", "cannot be combined with another code"),
    "minimum_not_met": ("discount_code_minimum_not_met", "needs a larger order"),
    "stopped": ("discount_code_combination_disallowed", "was not applied: a rule before it stops further rules"),
}


@dataclass(frozen=True, slots=True)
class Wording:
    """A platform's own words for the refusals an agent meets (carttrace.sandbox.platforms `Platform.wording`), and how it
    writes an amount. Empty keeps the sandbox's words."""

    texts: Mapping[str, str]
    money: Callable[[int], str] = str
    sold_out_warning: bool = False
    """A line refused as sold out is a `merchandise_out_of_stock` warning with no path, and the cart is made without
    it (live Shopify, D52: "The product '<title>' is already sold out.", 210 of 210 such messages); otherwise UCP's
    `out_of_stock` error at the line."""

    def say(self, key: str, **facts: Any) -> str | None:
        template = self.texts.get(key)
        if template is None:
            return None
        try:
            return template.format(**facts)
        except (KeyError, IndexError):
            return None


PLAIN_WORDING = Wording({})


def event_messages(
    events: Sequence[Mapping[str, Any]],
    *,
    paths: Paths,
    title: Callable[[str], str],
    wording: Wording = PLAIN_WORDING,
) -> list[Message]:
    """Messages for the agent-visible refusals and changes among a command's events (`Outcome.messages()`), in the
    platform's words where it has its own (the codes stay UCP's)."""
    out: list[Message] = []
    for e in events:
        d = e.get("data") or {}
        kind = e.get("type")
        if kind == "line.rejected":
            out.append(_line_message(d, paths, title, wording))
        elif kind == "line.clamped":
            added = int(d.get("added") or 0)
            noun = "1 item was" if added == 1 else f"{added} items were"
            text = (
                wording.say("stock.clamped", added=added, noun=noun)
                or f"Only {noun} added to your cart due to availability."
            )
            out.append(warning("merchandise_not_enough_stock", text))  # live names no path (store-310 ep-ad499feca1d4)
        elif kind == "discount.rejected":
            reason = str(d.get("reason"))
            code, says = DISCOUNT_REASONS.get(reason, ("discount_code_invalid", "was not applied"))
            sent = str(d.get("code"))
            own = wording.say(
                f"discount.{reason}",
                code=sent.strip().lower(),
                other=str(d.get("with_code") or "").lower(),
                min_spend=wording.money(int(d["min_spend"])) if isinstance(d.get("min_spend"), int) else "",
            )
            out.append(warning(code, own or f"Code '{sent}' {says}", path=paths.code(sent)))
        elif kind == "fulfillment.rejected":
            out.append(_fulfillment_message(d, paths))
        elif kind == "checkout.repriced":
            out.append(
                warning("price_changed", "A price changed since the checkout was quoted; the totals are updated")
            )
        elif kind == "checkout.stock_lost":
            out.append(error("out_of_stock", "The reserved stock lapsed and is gone", path="$.line_items"))
    return out


def _line_message(d: Mapping[str, Any], paths: Paths, title: Callable[[str], str], wording: Wording) -> Message:
    reason = str(d.get("reason"))
    vid = d.get("variant_id")
    path = paths.line(line_id=d.get("line_id"), variant_id=None if vid is None else str(vid))
    if reason == "sold_out" or (reason == "out_of_stock" and wording.sold_out_warning and not d.get("available")):
        return _sold_out_message(title(str(vid)) if vid is not None else "This item", path, wording)
    if reason == "out_of_stock":
        name = title(str(vid)) if vid is not None else "This item"
        text = f"{name} is out of stock: {d.get('requested')} requested, {d.get('available')} available"
        case = "stock.not_enough" if int(d.get("available") or 0) > 0 else "stock.out_of_stock"
        own = wording.say(case, name=name, available=d.get("available"), requested=d.get("requested"))
        return error("out_of_stock", own or text, path=path)
    if reason == "options_required":
        text = "This product has options: item.id takes one of its variants' ids"
        own = wording.say("line.options_required")
        return error("options_required", own or text, path=None if path is None else f"{path}.item.id")
    if reason == "not_found":
        text = "This item is not in the store's catalog; item.id takes a variant id from the catalog"
        return error("not_found", text, path=None if path is None else f"{path}.item.id")
    if reason == "line_not_found":
        text = "This store never issued that line item id for this resource"
        return error("not_found", text, path=None if path is None else f"{path}.id")
    return error("invalid_line_item", "The line item id and item id name different items", path=path)


def _sold_out_message(name: str, path: str | None, wording: Wording) -> Message:
    """A variant the store will not sell, in the platform's words: the store's `stock.sold_out` sentence, else its
    out-of-stock one (WooCommerce's, diff-woo 41), else the sandbox's. No count of units is said, since none remain."""
    text = wording.say("stock.sold_out", name=name) or wording.say("stock.out_of_stock", name=name, available=0)
    if wording.sold_out_warning:
        return warning("merchandise_out_of_stock", text or f"The product '{name}' is already sold out.")
    return error("out_of_stock", text or f"{name} is sold out", path=path)


def _fulfillment_message(d: Mapping[str, Any], paths: Paths) -> Message:
    reason = str(d.get("reason"))
    at = paths.method(None if d.get("group_id") is None else str(d.get("group_id")))
    if reason == "destination_not_shippable":
        return error(
            "address_undeliverable", f"This store does not ship to {d.get('country')}", path=f"{at}.destinations"
        )
    if reason == "invalid_option":
        return error(
            "invalid_option", "No such fulfillment option in this group", path=f"{at}.groups[0].selected_option_id"
        )
    if reason == "invalid_location":
        return error("invalid_location", "No such pickup location", path=f"{at}.selected_destination_id")
    if reason == "line_not_found":
        return error("not_found", "A line item id this method names is not in the checkout", path=f"{at}.line_item_ids")
    return error(reason, "The fulfillment change was refused", path=at)

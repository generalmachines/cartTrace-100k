"""A UCP cart reply, from a live store or the sandbox, as the normalised `Cart` traces and the judge read.

Why one converter in the trace package. The live client and the sandbox both return UCP carts, and the judge compares
the final cart with the hidden spec; if each wrote its own parser, a live cart and a sandbox cart of the same lines
could normalise differently and the benchmark would compare formats instead of journeys. The shape below is the one a
live Shopify store returned on 2026-09-29 (UCP 2026-08-25): an `id`, `line_items` whose `item` carries the variant id,
title and unit price in minor units, a `currency`, and `totals` typed subtotal and total. The converter accepts the
cart itself or a reply that wraps it under "cart", "result" or MCP's "structuredContent".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from carttrace.core.errors import ValidationError
from carttrace.trace.schema import Cart, CartLine

_WRAPPERS = ("structuredContent", "result", "cart")


def from_ucp(payload: Mapping[str, Any]) -> Cart:
    """The normalised cart in a UCP cart reply. Raises ValidationError when there is no cart or a line has no
    variant id, quantity or integer price, because a guessed line would let the judge pass a cart it never saw."""
    cart = _unwrap(payload)
    currency = _str(cart.get("currency"))
    lines = [_line(raw, currency, i) for i, raw in enumerate(cart.get("line_items") or [])]
    totals = {str(t.get("type")): t.get("amount") for t in cart.get("totals") or [] if isinstance(t, Mapping)}
    return Cart(
        cart_id=_str(cart.get("id")),
        lines=lines,
        subtotal=_int_or_none(totals.get("subtotal")),
        total=_int_or_none(totals.get("total")),
        currency=currency,
    )


def _unwrap(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    current: Any = payload
    for _ in range(len(_WRAPPERS) + 1):
        if isinstance(current, Mapping) and "line_items" in current:
            return current
        nested = next((current[k] for k in _WRAPPERS if isinstance(current, Mapping) and k in current), None)
        if nested is None:
            break
        current = nested
    raise ValidationError("reply holds no UCP cart (no line_items)", hint="pass the get_cart or update_cart reply")


def _line(raw: Any, currency: str | None, index: int) -> CartLine:
    if not isinstance(raw, Mapping):
        raise ValidationError(f"cart line {index} is not an object")
    raw_item = raw.get("item")
    item: Mapping[str, Any] = raw_item if isinstance(raw_item, Mapping) else {}
    variant = _str(item.get("id")) or _str(raw.get("variant_id"))
    quantity = raw.get("quantity")
    price = item.get("price", raw.get("unit_price"))
    if (
        variant is None
        or not isinstance(quantity, int)
        or not isinstance(price, int)
        or not _is_int(quantity)
        or not _is_int(price)
    ):
        raise ValidationError(
            f"cart line {index} lacks a variant id, an integer quantity or an integer price in minor units",
            hint="UCP carts carry item.id, quantity and item.price; a line without them cannot be judged",
        )
    line_currency = _str(item.get("currency")) or currency
    if line_currency is None:
        raise ValidationError(f"cart line {index} has no currency and the cart names none")
    raw_options = item.get("options")
    options: Mapping[str, Any] = raw_options if isinstance(raw_options, Mapping) else {}
    return CartLine(
        product_id=_str(item.get("product_id")),
        variant_id=variant,
        title=_str(item.get("title")) or "",
        quantity=quantity,
        unit_price=price,
        currency=line_currency,
        options={str(k): str(v) for k, v in options.items()},
        selling_plan=_str(raw.get("selling_plan_id") or item.get("selling_plan_id")),
    )


def from_page_cart(payload: Mapping[str, Any]) -> Cart:
    """A cart as a store page's WebMCP get_cart shows it (D67): Shopify's storefront adapter answers `line_items`
    whose `item` carries the variant id, title and `unit_price` {amount in minor units, currency}, and the cart's
    `total_price` the same way, wrapped in MCP's structuredContent. Raises ValidationError like `from_ucp`."""
    cart = _unwrap(payload)
    raw_total = cart.get("total_price")
    total: Mapping[str, Any] = raw_total if isinstance(raw_total, Mapping) else {}
    currency = _str(cart.get("currency")) or _str(total.get("currency"))
    lines = []
    for index, raw in enumerate(cart.get("line_items") or []):
        raw_item = raw.get("item") if isinstance(raw, Mapping) else None
        item: Mapping[str, Any] = raw_item if isinstance(raw_item, Mapping) else {}
        raw_price = item.get("unit_price")
        price: Mapping[str, Any] = raw_price if isinstance(raw_price, Mapping) else {}
        variant, amount = _str(item.get("id")), price.get("amount")
        quantity = raw.get("quantity") if isinstance(raw, Mapping) else None
        line_currency = _str(price.get("currency")) or currency
        if variant is None or not _is_int(quantity) or not _is_int(amount) or line_currency is None:
            raise ValidationError(
                f"page cart line {index} lacks a variant id, an integer quantity, a unit price or its currency",
                hint="the page's get_cart lines carry item.id, quantity and item.unit_price {amount, currency}",
            )
        variant_title = _str(item.get("variant_title"))
        title = " - ".join(t for t in (_str(item.get("title")), variant_title) if t and t != "Default Title")
        lines.append(
            CartLine(
                variant_id=variant,
                title=title,
                quantity=cast(int, quantity),
                unit_price=cast(int, amount),
                currency=line_currency,
            )
        )
    return Cart(cart_id=_str(cart.get("id")), lines=lines, total=_int_or_none(total.get("amount")), currency=currency)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_or_none(value: Any) -> int | None:
    return int(value) if _is_int(value) else None


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None

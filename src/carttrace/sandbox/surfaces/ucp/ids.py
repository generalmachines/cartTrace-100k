"""Cart and line ids as a platform spells them on the wire, and the way back to the core's ids.

Why. The core names a cart `cart-<hex>` and a line `line-<hex>`, which is enough for the state and the event log.
Live Shopify names a cart `gid://shopify/Cart/<token>?key=<key>` and a line `gid://shopify/CartLine/<id>?cart=<token>`
(var/live/delta-01/report.md section 4), and three live calls failed with `invalid_cart_id` because an agent dropped
the `?key=` part. An agent that cannot make that mistake on the sandbox is not being measured on it. So on a keyed
platform the surface renders the core's ids in the store's shape and reads them back, and a malformed or keyless id is
refused the way the store refuses it, while the core and the event log keep their own ids and digests.

The token and the key are derived from the pack and the core id, so they are the same on every run; the key is a
secret in form only (32 hex digits, which the trace writer redacts like a real one) and never in substance.

The ids inside a cart update are read the same way (D46): a line gid sent without its `?cart=` part and a delivery
group id that is not a `gid://shopify/CartDeliveryGroup/` gid were both refused live, each in its own shape
(`carttrace.sandbox.platforms.shopify.SHOPIFY_EVIDENCE`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from carttrace.core.canon import digest

CART_GID = "gid://shopify/Cart/"
LINE_GID = "gid://shopify/CartLine/"
DELIVERY_GROUP_GID = "gid://shopify/CartDeliveryGroup/"
TOKEN_LENGTH = 24
KEY_LENGTH = 32
_CART = re.compile(r"^gid://shopify/Cart/(?P<token>[A-Za-z0-9_-]+)(?:\?key=(?P<key>[A-Za-z0-9_-]+))?$")
_LINE = re.compile(r"^gid://shopify/CartLine/(?P<id>[A-Za-z0-9_-]+)(?:\?cart=(?P<token>[A-Za-z0-9_-]*))?$")
_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"

Reading = Literal["ok", "malformed", "unknown"]


@dataclass(frozen=True, slots=True)
class ParsedCartId:
    core_id: str | None
    reading: Reading
    """`malformed` for an id not in the platform's shape (or keyed and sent without its key); `unknown` for a
    well-formed id naming no cart of this store; `ok` otherwise."""


class CartIds:
    """The id mapping of one served store. With `keyed` off every method is the identity."""

    def __init__(self, pack_id: str, *, keyed: bool) -> None:
        self.pack_id = pack_id
        self.keyed = keyed

    # -- rendering ----------------------------------------------------------------------------------------------------

    def token(self, core_id: str) -> str:
        """The cart token: 24 URL-safe characters from the pack and the core id."""
        raw = digest(["cart-token", self.pack_id, core_id], 48)
        return "".join(_B64[int(raw[i : i + 2], 16) % len(_B64)] for i in range(0, 2 * TOKEN_LENGTH, 2))

    def key(self, core_id: str) -> str:
        return digest(["cart-key", self.pack_id, core_id], KEY_LENGTH)

    def wire_cart(self, core_id: str) -> str:
        if not self.keyed:
            return core_id
        return f"{CART_GID}{self.token(core_id)}?key={self.key(core_id)}"

    def wire_line(self, core_line_id: str, core_cart_id: str) -> str:
        if not self.keyed:
            return core_line_id
        return f"{LINE_GID}{core_line_id.partition('-')[2] or core_line_id}?cart={self.token(core_cart_id)}"

    def continue_path(self, core_id: str) -> str:
        """The path of the cart's continue URL under the store's base: `/cart/c/<token>?key=<key>` on Shopify."""
        if not self.keyed:
            return f"/cart/{core_id}"
        return f"/cart/c/{self.token(core_id)}?key={self.key(core_id)}"

    # -- reading ------------------------------------------------------------------------------------------------------

    def parse_cart(self, wire_id: str, known: Iterable[str]) -> ParsedCartId:
        """The core cart id a wire id names, among the `known` core ids of this store."""
        if not self.keyed:
            return ParsedCartId(wire_id, "ok")
        m = _CART.match(wire_id)
        if m is None or m.group("key") is None:
            return ParsedCartId(None, "malformed")
        token = m.group("token")
        for core_id in known:
            if self.token(core_id) == token:
                if m.group("key") != self.key(core_id):
                    return ParsedCartId(None, "unknown")
                return ParsedCartId(core_id, "ok")
        return ParsedCartId(None, "unknown")

    def by_token(self, token: str, known: Iterable[str]) -> str | None:
        """The core cart id of a bare token (a continue URL's path part)."""
        return next((core_id for core_id in known if self.token(core_id) == token), None)

    def keyless_line(self, wire_id: str) -> bool:
        """A line gid sent without its `?cart=` part, which live Shopify refuses as "Invalid id: <id>" (store-305
        ep-e4e9532bc58b seq 48). False on a platform without keyed ids, and for an id not in the line gid shape at
        all, which the core refuses as a line it never issued."""
        if not self.keyed:
            return False
        m = _LINE.match(wire_id)
        return m is not None and m.group("token") is None

    def parse_line(self, wire_id: str) -> str:
        """The core line id a wire line id names; an id not in the platform's shape is returned as it is, so the core
        refuses it as a line it never issued."""
        if not self.keyed:
            return wire_id
        m = _LINE.match(wire_id)
        return f"line-{m.group('id')}" if m is not None else wire_id


def delivery_group_gid(value: object) -> bool:
    """A Shopify delivery group id: `gid://shopify/CartDeliveryGroup/<id>`. Live Shopify refused `delivery_speed`
    as "is not a valid CartDeliveryGroup GID" (store-180 ep-da1cd727b196 seq 23); what it does with a well-formed one
    the sandbox never issued is unobserved."""
    return isinstance(value, str) and value.startswith(DELIVERY_GROUP_GID) and len(value) > len(DELIVERY_GROUP_GID)

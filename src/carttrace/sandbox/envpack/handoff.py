"""Hand a live cart to the sandbox: build a pack from the cart, and seed the sandbox with the same cart.

Why. A live journey ends at a verified cart (decision D3: nothing is bought on a live store). Payment then runs in the
sandbox at the versions the store declared, and it must start from exactly the cart the shopper accepted: the same
variants, quantities and unit prices. `build_envpack_from_cart` puts every line's variant in the pack's snapshot at the
live price, and `handoff_command` is the system command that re-creates the cart through `apply`, so the hand-off is in
the event log and replays like any other command.

What is never carried over. A live cart id can hold a secret (`?key=` on Shopify cart tokens); only the part before
any `?` is kept as `cart_ref`, and nothing else from the live store's session travels with the cart.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from carttrace.protocols.profile import ProtocolProfile
from carttrace.sandbox.core.apply import apply, outcome
from carttrace.sandbox.core.errors import from_dict as error_from_dict
from carttrace.sandbox.core.event import Event
from carttrace.sandbox.core.market import Market
from carttrace.sandbox.core.state import State
from carttrace.sandbox.envpack.pack import EnvPack, EnvPackError, build_envpack
from carttrace.trace.schema import Cart, StoreRef, TaskRef

CART_REF_MAX = 200


def cart_currency(cart: Cart) -> str:
    """The one currency of a live cart. Refuses a cart whose lines disagree with each other or with the cart."""
    found = {ln.currency.upper() for ln in cart.lines} | ({cart.currency.upper()} if cart.currency else set())
    if len(found) != 1:
        raise EnvPackError(
            f"a live cart needs exactly one currency, found {sorted(found) or 'none'}",
            hint="hand off carts the live store priced in one currency",
        )
    return found.pop()


def cart_variants(cart: Cart, *, stock: int = 100) -> list[dict[str, Any]]:
    """The variants a live cart holds, as snapshot variants at the cart's own unit prices, with a shelf at least as
    deep as the quantity the cart asks for."""
    if not cart.lines:
        raise EnvPackError("the live cart has no lines", hint="only a verified, non-empty cart is handed off")
    out: dict[str, dict[str, Any]] = {}
    qty: dict[str, int] = {}
    for ln in cart.lines:
        seen = out.get(ln.variant_id)
        if seen is not None and seen["price"] != ln.unit_price:
            raise EnvPackError(
                f"variant {ln.variant_id!r} has two prices in one live cart", hint="re-read the cart before hand-off"
            )
        qty[ln.variant_id] = qty.get(ln.variant_id, 0) + ln.quantity
        out[ln.variant_id] = {
            "id": ln.variant_id,
            "product_id": ln.product_id or "",
            "title": ln.title,
            "price": ln.unit_price,
            "available": True,
            "stock": max(stock, qty[ln.variant_id]),
            "options": dict(ln.options),
        }
    return list(out.values())


def build_envpack_from_cart(
    cart: Cart,
    *,
    store: str | StoreRef | Mapping[str, Any],
    profile: ProtocolProfile,
    payment_handlers: Mapping[str, Any] | None = None,
    market: Market | str = "US",
    other_variants: Iterable[Mapping[str, Any]] = (),
    products: Mapping[str, Mapping[str, Any]] | None = None,
    stock: int = 100,
    tasks: Iterable[TaskRef | Mapping[str, Any]] = (),
    faults: Iterable[Mapping[str, Any]] = (),
    start: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
    source: str = "live:cart",
    captured_at: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> EnvPack:
    """A pack whose snapshot holds every variant of the live cart at its live price, plus `other_variants` the journey
    touched (search results, alternatives). Where both name a variant, the cart's price wins: it is the latest one the
    store showed. The profile and handler configs are the store's own, as captured during the journey."""
    in_cart = cart_variants(cart, stock=stock)
    ids = {v["id"] for v in in_cart}
    rest = [dict(v) for v in other_variants if str(v["id"]) not in ids]
    return build_envpack(
        store=store,
        variants=[*in_cart, *rest],
        currency=cart_currency(cart),
        profile=profile,
        payment_handlers=payment_handlers,
        products=products,
        market=market,
        tasks=tasks,
        faults=faults,
        start=start,
        policy=policy,
        snapshot_source=source,
        captured_at=captured_at,
        provenance=provenance,
    )


def safe_cart_ref(cart_id: str | None) -> str | None:
    """The live cart id without its query (where a cart token's secret key lives), or None if nothing usable is left."""
    if not cart_id:
        return None
    ref = cart_id.split("?", 1)[0].strip()
    return ref if 0 < len(ref) <= CART_REF_MAX and not any(ch.isspace() for ch in ref) else None


def handoff_command(
    cart: Cart,
    *,
    session: str = "buyer",
    source: str = "live",
    country: str | None = None,
) -> dict[str, Any]:
    """The system command that re-creates `cart` in the sandbox for the agent's `session`."""
    cmd: dict[str, Any] = {
        "type": "handoff_cart",
        "actor": "system",
        "session": session,
        "currency": cart_currency(cart),
        "lines": [
            {"variant_id": ln.variant_id, "quantity": ln.quantity, "unit_price": ln.unit_price} for ln in cart.lines
        ],
        "source": source,
        "cart_ref": safe_cart_ref(cart.cart_id),
    }
    if country is not None:
        cmd["country"] = country
    return cmd


def seed_state_from_cart(
    pack: EnvPack, cart: Cart, *, session: str = "buyer", seed: int | None = None
) -> tuple[State, list[Event]]:
    """The pack's initial state with the live cart handed off to `session`, and the events that did it. Raises the
    core's typed error when the sandbox cannot reproduce the cart exactly."""
    state, events = apply(pack.initial_state(seed), handoff_command(cart, session=session), pack.config())
    result = outcome(events)
    if not result["ok"]:
        raise error_from_dict(result["error"])
    return state, events

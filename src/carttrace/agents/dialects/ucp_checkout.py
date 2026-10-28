"""UCP's checkout-only releases, 2026-01-11 and 2026-01-23 (D71): no cart, no catalog operation of their own, no
get_order.

What differs from the later releases, and why the agents still run the same loop:
- The store's products come from its feed, which the sandbox serves through 2026-04-08's catalog tools listed beside
  the checkout tools, so searching and reading a product are UCP's as always.
- There is no cart: the first thing an agent makes is a checkout holding the items (`create_checkout` with
  `line_items`), as on ACP, and the checkout's lines are the cart the agent reads back and the judge sees. Its `id`
  goes back in every update's payload, which both releases require.
- The order comes back inside the completed checkout (`order`: id and permalink); there is no order tool.
- Completion takes an idempotency key as a parameter of its own (`idempotency_key`, a UUID), derived here from the
  checkout and the wallet's instrument so a retry with a new credential is a new request.
- 2026-01-11 also requires a checkout's `currency` and a `payment` object on create and update, puts the completion's
  `payment` beside its `id` with `selected_instrument_id`, and types a card instrument with `brand` and `last_digits`,
  which is empty: the sandbox's credentials are opaque tokens with no card number behind them.

Shapes are read from the listed tools' schemas (`from_specs`), so one class speaks both releases.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Self

from carttrace.agents.checkout import line_inputs
from carttrace.agents.dialects.base import Call, Dialect
from carttrace.agents.entry import Details, Option, UcpEntry, checkout_only
from carttrace.agents.tools import payload_of
from carttrace.agents.wallet import Credential
from carttrace.core.canon import digest
from carttrace.protocols.tools import ToolResult, ToolSpec
from carttrace.trace.schema import Cart

CHECKOUT_TOOLS: Final = frozenset({"create_checkout", "update_checkout", "get_checkout"})
"""Checkout tools whose result is the cart the agent is building (the checkout before payment)."""


def _checkout_props(spec: ToolSpec | None) -> tuple[Mapping[str, Any], list[str]]:
    if spec is None:
        return {}, []
    checkout = spec.input_schema.get("properties", {}).get("checkout")
    if not isinstance(checkout, Mapping):
        return {}, []
    props = checkout.get("properties")
    required = checkout.get("required")
    return (props if isinstance(props, Mapping) else {}), (list(required) if isinstance(required, list) else [])


def idempotency_uuid(*parts: object) -> str:
    """A deterministic UUID-shaped key for these parts, as the releases' `idempotency_key` takes it."""
    return str(uuid.UUID(hex=digest(["ucp_checkout", *[str(p) for p in parts]], 32)))


@dataclass(frozen=True, slots=True)
class UcpCheckoutDialect(Dialect):
    """UCP 2026-01-11 and 2026-01-23: the checkout is the cart, with the store's feed read through the catalog tools."""

    create: str = "create_checkout"
    read: str = "get_checkout"
    create_checkout: str = "update_checkout"
    update_checkout: str = "update_checkout"
    get_checkout: str = "get_checkout"
    get_order: str | None = None
    order_tools: frozenset[str] = frozenset()
    top_level_payment: bool = False
    """The completion's `payment` sits beside its `id` (2026-01-11), with `selected_instrument_id`."""
    keyed: bool = True
    """The completion takes `idempotency_key` as a parameter of its own."""
    currency_required: bool = False
    """create and update require the checkout's `currency` and a `payment` object (2026-01-11)."""

    @property
    def name(self) -> str:
        return "ucp_checkout"

    @classmethod
    def from_specs(cls, specs: Mapping[str, ToolSpec]) -> Self:
        complete = specs.get("complete_checkout")
        props = complete.input_schema.get("properties", {}) if complete is not None else {}
        _, required = _checkout_props(specs.get("create_checkout"))
        return cls(
            top_level_payment="payment" in props,
            keyed="idempotency_key" in props if props else True,
            currency_required="currency" in required,
        )

    def offered(self, tools: Collection[str]) -> bool:
        return checkout_only(tools) and self.search in set(tools)

    def required(self, currency: str | None) -> dict[str, Any]:
        """What 2026-01-11 requires of a checkout write besides its lines."""
        if not self.currency_required:
            return {}
        return {"payment": {}, **({"currency": currency.upper()} if currency else {})}

    # -- the cart: a checkout ------------------------------------------------------------------------------------------

    def lines_arguments(self, lines: Sequence[tuple[str, int]], *, currency: str | None = None) -> dict[str, Any]:
        items = [{"item": {"id": vid}, "quantity": qty} for vid, qty in lines]
        return {"checkout": {"line_items": items, **self.required(currency)}}

    def is_cart_tool(self, name: str) -> bool:
        return name in CHECKOUT_TOOLS

    def cart_in(self, result: ToolResult) -> Cart | None:
        if result.is_error:
            return None
        doc = self.checkout_doc(payload_of(result))
        return None if doc is None else self.view_of(doc)

    def open_checkout_calls(self, cart_id: str, details: Mapping[str, Any]) -> list[Call]:
        """The checkout already exists (it is the cart): read it; the entry gives it the buyer's details."""
        del details
        return [(self.get_checkout, self.id_arguments(cart_id))]

    # -- documents -----------------------------------------------------------------------------------------------------

    def order_doc(self, payload: Any) -> Mapping[str, Any] | None:
        return None

    def order_view(self, doc: Mapping[str, Any]) -> Cart | None:
        order_id = self.order_of(doc)
        view = self.view_of(doc) if order_id is not None else None
        return None if view is None else view.model_copy(update={"cart_id": order_id})

    def order_label(self, doc: Mapping[str, Any]) -> str | None:
        order = doc.get("order")
        label = order.get("label") if isinstance(order, Mapping) else None
        return label if isinstance(label, str) and label else self.order_of(doc)

    # -- paying --------------------------------------------------------------------------------------------------------

    def credential_answer(self, credential: Credential, *, token: str | None = None) -> dict[str, Any]:
        out = Dialect.credential_answer(self, credential, token=token)
        if self.top_level_payment:
            instrument = dict(out[self.payment_key])
            instrument.pop("selected", None)
            display = instrument.get("display")
            brand = display.get("brand") if isinstance(display, Mapping) else None
            instrument.update(brand=str(brand or credential.brand or "unknown"), last_digits="")
            out[self.payment_key] = instrument
            out["use"] = f"put this instrument in {self.complete_checkout} as payment.instruments"
        return out

    def complete_arguments(self, checkout_id: str, answer: Mapping[str, Any]) -> dict[str, Any]:
        instrument = dict(answer[self.payment_key])
        if self.top_level_payment:
            payment = {"selected_instrument_id": str(instrument.get("id") or ""), "instruments": [instrument]}
            out: dict[str, Any] = {"id": checkout_id, "payment": payment}
        else:
            out = {"id": checkout_id, "checkout": {"payment": {"instruments": [instrument]}}}
        if self.keyed:
            out["idempotency_key"] = idempotency_uuid(checkout_id, instrument.get("id"), instrument.get("credential"))
        return out

    def tokens_of(self, arguments: Mapping[str, Any]) -> set[str]:
        if not self.top_level_payment:
            return Dialect.tokens_of(self, arguments)
        return Dialect.tokens_of(self, {"checkout": {"payment": arguments.get("payment")}})


class UcpCheckoutEntry(UcpEntry):
    """The buyer's details through a checkout-only release's update_checkout: the base UCP entry's payload, with the
    checkout's `id` in it (both releases require it) and, at 2026-01-11, its currency and a payment object."""

    def __init__(self, specs: Mapping[str, Any] | None = None) -> None:
        _, required = _checkout_props((specs or {}).get("update_checkout"))
        self.currency_required = "currency" in required

    @property
    def name(self) -> str:
        return "ucp_checkout"

    def offered(self, tools: Collection[str]) -> bool:
        return checkout_only(tools)

    def create(self, cart_id: str) -> Call:
        return "get_checkout", {"id": cart_id}

    def _complete(self, doc: Mapping[str, Any], call: Call) -> Call:
        name, arguments = call
        payload = dict(arguments.get("checkout") or {})
        payload["id"] = str(doc["id"])
        payload.setdefault("line_items", line_inputs(doc))
        if self.currency_required:
            payload.setdefault("payment", {})
            if isinstance(doc.get("currency"), str):
                payload.setdefault("currency", doc["currency"])
        return name, {**arguments, "checkout": payload}

    def enter(self, doc: Mapping[str, Any], details: Details) -> list[Call]:
        return [self._complete(doc, c) for c in super().enter(doc, details)]

    def select(self, doc: Mapping[str, Any], option: Option) -> Call:
        return self._complete(doc, super().select(doc, option))

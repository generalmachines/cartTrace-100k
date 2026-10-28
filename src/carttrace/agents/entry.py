"""Entering the buyer's details through a store's own checkout operations, whatever protocol the store speaks.

Why a seam and not harness tools. What the benchmark measures here is the store's protocol: whether its checkout
says what it still needs, and whether an agent can give it the shopper's address, contact details and delivery
choice through the store's own operations. A harness tool for buyer details would measure our harness instead. So an
agent reads which checkout the surfaces offer and speaks it through a `CheckoutEntry`:

- `missing(doc)`: the buyer fields the store's checkout says it still needs, in the order an agent asks for them
  (delivery first, because pickup needs no address);
- `enter(doc, details)`: the calls that give the checkout what the shopper disclosed;
- `options(doc)` and `select(doc, option)`: the fulfillment options the checkout offers, each with its kind
  (standard, express, pickup) and its stated delivery time, and the call that picks one;
- `ready(doc)` and `total(doc)`: whether it can be completed, and the total the shopper is asked to approve.

Built: UCP here (create_checkout, update_checkout: `buyer` for contact, a shipping method's `destinations` for the
address, `groups[].selected_option_id` for the option); ACP (`carttrace.agents.dialects.acp.AcpEntry`: a checkout session's
`buyer`, its fulfillment address and selected option, under the names the release uses) and WebMCP
(`carttrace.agents.dialects.webmcp.WebMcpEntry`: set_buyer_contact, set_shipping_address, choose_fulfillment), each one
class registered in carttrace.agents PLUGINS under `agents.checkout_entry`; `entry_for` picks the first whose tools a surface
lists, shaped by the tools' schemas when it takes them.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

from carttrace.agents.buyer import option_kind, stated_days, ucp_address
from carttrace.agents.checkout import line_inputs, total_of
from carttrace.agents.spec import Address, FulfillmentKind
from carttrace.core.money import Money

ENTRY_KIND: Final = "agents.checkout_entry"
"""The registry kind of the checkout entries (`carttrace.agents.PLUGINS`)."""
ASK_ORDER: Final = ("fulfillment", "shipping_address", "email", "phone")
"""The buyer fields a checkout can report missing, in the order an agent asks the shopper for them."""
Method = Literal["shipping", "pickup"]


@dataclass(frozen=True, slots=True)
class Details:
    """What the shopper disclosed, to be entered: nothing here is ever a value the shopper did not give."""

    email: str | None = None
    phone: str | None = None
    address: Address | None = None
    method: Method | None = None
    """Ship or collect, from the shopper's delivery preference."""


@dataclass(frozen=True, slots=True)
class Option:
    """One fulfillment option a checkout offers, and where it sits."""

    method_id: str
    method: str
    group_id: str
    option_id: str
    title: str
    amount: int
    kind: FulfillmentKind | None
    """standard, express or pickup, from the option's method and words; None when they do not say."""
    days: int | None
    """The longest delivery time its title states."""


Call = tuple[str, dict[str, Any]]
"""A tool name and its arguments."""


class CheckoutEntry(Protocol):
    """How one protocol's checkout takes the buyer's details."""

    @property
    def name(self) -> str: ...

    def offered(self, tools: Collection[str]) -> bool:
        """Whether a surface listing `tools` speaks this checkout."""
        ...

    def create(self, cart_id: str) -> Call:
        """The call that opens a checkout from the cart, with no buyer details: the store says what it needs."""
        ...

    def missing(self, doc: Mapping[str, Any]) -> list[str]:
        """The buyer fields (ASK_ORDER) the checkout reports it still needs."""
        ...

    def enter(self, doc: Mapping[str, Any], details: Details) -> list[Call]:
        """The calls that give the checkout the disclosed details; empty when there is nothing to give."""
        ...

    def options(self, doc: Mapping[str, Any]) -> list[Option]:
        """Every fulfillment option the checkout offers, cheapest first."""
        ...

    def select(self, doc: Mapping[str, Any], option: Option) -> Call:
        """The call that picks one option."""
        ...

    def selected(self, doc: Mapping[str, Any]) -> Option | None:
        """The option the checkout has selected, if any."""
        ...

    def ready(self, doc: Mapping[str, Any]) -> bool:
        """Whether the checkout can be completed."""
        ...

    def total(self, doc: Mapping[str, Any]) -> Money | None:
        """The checkout's total, shipping and tax included."""
        ...


_INDEXED = re.compile(r"\[\d+\]")
UCP_MISSING: Final[Mapping[str, str]] = {
    "$.buyer": "email",
    "$.buyer.email": "email",
    "$.buyer.phone_number": "phone",
    "$.fulfillment.methods": "fulfillment",
    "$.fulfillment.methods.destinations": "shipping_address",
    "$.fulfillment.methods.groups.selected_option_id": "fulfillment",
    "$.shipping_address": "shipping_address",
}
"""A UCP checkout's `missing` message paths (indices dropped) to the buyer field each one asks for; Medusa's pack names
the address at `$.shipping_address` (D94)."""


def checkout_only(tools: Collection[str]) -> bool:
    """Whether the listed tools are a checkout-only UCP release's (2026-01-11, 2026-01-23, D71): a checkout, and no
    cart, order or AP2 intent; `carttrace.agents.dialects.ucp_checkout` speaks those."""
    listed = set(tools)
    return {"create_checkout", "update_checkout"} <= listed and not listed & {
        "create_cart",
        "get_order",
        "send_intent_mandate",
    }


class UcpEntry:
    """UCP 2026-04-08 and 2026-08-25: create_checkout from the cart, then update_checkout, which replaces the whole
    checkout, so every call sends the line items back."""

    @property
    def name(self) -> str:
        return "ucp"

    def offered(self, tools: Collection[str]) -> bool:
        return {"create_checkout", "update_checkout"} <= set(tools) and not checkout_only(tools)

    def create(self, cart_id: str) -> Call:
        return "create_checkout", {"checkout": {"cart_id": cart_id, "line_items": []}}

    def missing(self, doc: Mapping[str, Any]) -> list[str]:
        found: set[str] = set()
        for message in doc.get("messages") or ():
            code = str(message.get("code") or "") if isinstance(message, Mapping) else ""
            if code != "missing" and not code.startswith("missing_"):
                # a platform's own readiness code names the field too (WooCommerce's missing_email, D94)
                continue
            path = _INDEXED.sub("", str(message.get("path") or ""))
            field = UCP_MISSING.get(path)
            if field is not None:
                found.add(field)
        return [f for f in ASK_ORDER if f in found]

    def enter(self, doc: Mapping[str, Any], details: Details) -> list[Call]:
        payload: dict[str, Any] = {}
        buyer = {k: v for k, v in (("email", details.email), ("phone_number", details.phone)) if v is not None}
        if buyer:
            payload["buyer"] = buyer
        if details.method is not None:
            method: dict[str, Any] = {"type": details.method}
            existing = _method_id(doc)
            if existing is not None:
                method["id"] = existing
            # the method names the lines it covers, which a store's recorded schema may require (live Shopify's)
            covered = next((m.get("line_item_ids") for m in _methods(doc) if m.get("id") == existing), None)
            ids = [str(x["id"]) for x in doc.get("line_items") or () if isinstance(x, Mapping) and x.get("id")]
            if isinstance(covered, list) and covered:
                method["line_item_ids"] = list(covered)
            elif ids:
                method["line_item_ids"] = ids
            if details.method == "shipping" and details.address is not None:
                method["destinations"] = [ucp_address(details.address)]
            payload["fulfillment"] = {"methods": [method]}
        if not payload:
            return []
        payload["line_items"] = line_inputs(doc)
        return [("update_checkout", {"id": str(doc["id"]), "checkout": payload})]

    def options(self, doc: Mapping[str, Any]) -> list[Option]:
        found: list[Option] = []
        for method in _methods(doc):
            for group in method.get("groups") or ():
                if not isinstance(group, Mapping) or not isinstance(group.get("id"), str):
                    continue
                for option in group.get("options") or ():
                    made = _option(method, group, option)
                    if made is not None:
                        found.append(made)
        return sorted(found, key=lambda o: o.amount)

    def select(self, doc: Mapping[str, Any], option: Option) -> Call:
        group = {"id": option.group_id, "selected_option_id": option.option_id}
        method: dict[str, Any] = {"id": option.method_id, "type": option.method, "groups": [group]}
        covered = next((m.get("line_item_ids") for m in _methods(doc) if m.get("id") == option.method_id), None)
        if isinstance(covered, list) and covered:  # the lines it covers, which a store's recorded schema may require
            method["line_item_ids"] = list(covered)
        checkout = {"line_items": line_inputs(doc), "fulfillment": {"methods": [method]}}
        return "update_checkout", {"id": str(doc["id"]), "checkout": checkout}

    def selected(self, doc: Mapping[str, Any]) -> Option | None:
        chosen = {
            str(g.get("selected_option_id"))
            for m in _methods(doc)
            for g in m.get("groups") or ()
            if isinstance(g, Mapping) and g.get("selected_option_id")
        }
        return next((o for o in self.options(doc) if o.option_id in chosen), None)

    def ready(self, doc: Mapping[str, Any]) -> bool:
        return doc.get("status") == "ready_for_complete"

    def total(self, doc: Mapping[str, Any]) -> Money | None:
        return total_of(doc)


def entry_for(tools: Collection[str], *, specs: Mapping[str, Any] | None = None) -> CheckoutEntry | None:
    """The checkout entry for a surface listing `tools`: the first registered under `agents.checkout_entry` that the
    tools offer, or None when the surface speaks none of them. `specs` (tool name to its `ToolSpec`) shapes an entry
    whose protocol changes its request bodies by release (ACP), when its class takes them."""
    import inspect

    from carttrace.core.registry import names, resolve

    for name in names(ENTRY_KIND):
        cls = resolve(ENTRY_KIND, name)
        try:
            takes_specs = "specs" in inspect.signature(cls).parameters
        except (TypeError, ValueError):
            takes_specs = False
        made: CheckoutEntry = cls(specs=specs) if takes_specs else cls()
        if made.offered(tools):
            return made
    return None


def pick(options: Sequence[Option], preference: FulfillmentKind, deadline: int | None) -> Option | None:
    """The option that matches the shopper's preference and meets their deadline, cheapest first. When no option's
    words say its kind, standard is the cheapest shipping option and express the dearest."""
    kinds = [o for o in options if o.kind == preference]
    if not kinds and preference != "pickup":
        shipping = [o for o in options if o.method == "shipping"]
        if shipping and all(o.kind is None for o in shipping):
            kinds = [shipping[0] if preference == "standard" else shipping[-1]]
    if deadline is not None:
        kinds = [o for o in kinds if o.days is not None and o.days <= deadline]
    return kinds[0] if kinds else None


def _methods(doc: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    fulfillment = doc.get("fulfillment")
    methods = fulfillment.get("methods") if isinstance(fulfillment, Mapping) else None
    return [m for m in methods or () if isinstance(m, Mapping) and isinstance(m.get("id"), str)]


def _method_id(doc: Mapping[str, Any]) -> str | None:
    """The checkout's one fulfillment method, which an update names so it changes that method instead of adding one."""
    methods = _methods(doc)
    return str(methods[0]["id"]) if len(methods) == 1 else None


def _option(method: Mapping[str, Any], group: Mapping[str, Any], option: Any) -> Option | None:
    if not isinstance(option, Mapping) or not isinstance(option.get("id"), str):
        return None
    priced = [t for t in option.get("totals") or () if isinstance(t, Mapping) and isinstance(t.get("amount"), int)]
    # the entry of type total; an option priced by one entry of another type (Medusa's pack writes `fulfillment`,
    # D94) is priced by it
    total = next((t for t in priced if t.get("type") == "total"), priced[0] if len(priced) == 1 else None)
    if total is None:
        return None
    amount = int(total["amount"])
    title = option["title"] if isinstance(option.get("title"), str) else option["id"]
    kind = str(method.get("type") or "shipping")
    return Option(
        method_id=str(method["id"]),
        method=kind,
        group_id=str(group["id"]),
        option_id=option["id"],
        title=title,
        amount=amount,
        kind=option_kind(title, option["id"], kind),
        days=stated_days(title),
    )

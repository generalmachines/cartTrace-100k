"""WebMCP: a storefront's page tools, which answer with the view the page shows.

What differs from UCP, and why the agents still run the same loop:
- The tools are the page's (`search_products`, `get_product`, `compare_products`, `add_to_cart`, `get_cart`,
  `start_checkout`, the three declarative forms, `get_checkout`, `place_order`), with flat arguments and no ids where
  the page has one cart and one checkout open (`get_cart`, `get_checkout` and `place_order` take none).
- A tool answers with its page's view: the cart under `cart`, the checkout under `checkout` with what it still lacks
  (`missing`, each naming the form that fills it in), the payment methods the page offers (`payment_handlers`), and
  the order page once the order is placed (`order`). `checkout_doc` reads a checkout view, and reads the order page as
  the completed checkout it stands for, so the harness and the reference agent see the same facts they see on UCP.
- The page answers the bank's 3-D Secure step itself (the sandbox browser does, as the persona would), so a
  `place_order` answer is the payment's outcome: `challenge_of` is never a URL for the wallet to answer.
- Payment is the UCP handler token the wallet issues (D13): `place_order` takes `payment_token` and the page's
  `payment_handler` id. The page names its payment methods by the store's handler entry ids, so the wallet, which
  knows handler namespaces, reads the namespace from the id's words (`namespace_of`).
- The buyer's details go through the page's forms (`WebMcpEntry`: `set_buyer_contact`, `set_shipping_address`,
  `choose_fulfillment`), which take the shopper's contact facts and address; the contact form's name is left out,
  because the shopper has none to give.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from carttrace.agents.buyer import option_kind, stated_days
from carttrace.agents.checkout import order_view, total_of
from carttrace.agents.dialects.base import Call, Dialect
from carttrace.agents.entry import ASK_ORDER, Details, Option
from carttrace.agents.wallet import Credential
from carttrace.core.money import Money
from carttrace.trace.schema import Cart

CHECKOUT_TOOLS: frozenset[str] = frozenset(
    {
        "start_checkout",
        "set_buyer_contact",
        "set_shipping_address",
        "choose_fulfillment",
        "apply_discount",
        "get_checkout",
        "place_order",
    }
)
ORDER_TOOLS: frozenset[str] = frozenset({"place_order", "get_checkout"})
"""Tools whose answer is the order page once the order is placed."""
CART_TOOLS: frozenset[str] = frozenset({"add_to_cart", "update_cart_line", "remove_cart_line", "get_cart"})
MISSING_USES: Mapping[str, tuple[str, ...]] = {
    "set_buyer_contact": ("email",),
    "set_shipping_address": ("shipping_address",),
    "choose_fulfillment": ("fulfillment",),
    "set_shipping_address or choose_fulfillment": ("fulfillment",),
}
"""The form a checkout view says fills in what is missing, to the buyer fields it asks for."""
NAMESPACE_WORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("gpay", "google"), "com.google.pay"),
    (("shop_pay", "shoppay"), "dev.shopify.shop_pay"),
    (("fake", "sandbox"), "dev.ucp.sandbox.fake"),
    (("card",), "dev.shopify.card"),
)
"""Words in a page's payment method id to the handler namespace the wallet knows it as."""
PAYMENT_ERRORS = ("payment", "order_not_placed", "declined")
"""Error codes of a place_order answer that mean the payment did not go through."""


@dataclass(frozen=True, slots=True)
class WebMcpDialect(Dialect):
    """How an agent speaks a storefront's WebMCP tools."""

    search: str = "search_products"
    product: str = "get_product"
    lookup: str = "compare_products"
    create: str = "add_to_cart"
    read: str = "get_cart"
    create_checkout: str = "start_checkout"
    update_checkout: str = "set_shipping_address"
    get_checkout: str = "get_checkout"
    complete_checkout: str = "place_order"
    get_order: str | None = None
    checkout_tools: frozenset[str] = CHECKOUT_TOOLS
    order_tools: frozenset[str] = ORDER_TOOLS
    payment_key: str = "payment_token"

    @property
    def name(self) -> str:
        return "webmcp"

    def offered(self, tools: Collection[str]) -> bool:
        return {"search_products", "add_to_cart"} <= set(tools)

    # -- the catalog and the cart --------------------------------------------------------------------------------------

    def search_arguments(self, query: str) -> dict[str, Any]:
        return {"query": query}

    def next_search_arguments(self, arguments: Mapping[str, Any], payload: Any) -> dict[str, Any] | None:
        """The storefront's next page of results: its `next_cursor`, passed back as `cursor`."""
        cursor = payload.get("next_cursor") if isinstance(payload, Mapping) else None
        return {**arguments, "cursor": cursor} if isinstance(cursor, str) and cursor else None

    def product_arguments(self, product_id: str, selected: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
        arguments: dict[str, Any] = {"product_id": product_id}
        if selected:
            arguments["options"] = {name: label for name, label in selected}
        return arguments

    def lookup_arguments(self, product_ids: Sequence[str]) -> dict[str, Any]:
        return {"product_ids": list(product_ids)}

    def cart_arguments(self, variant_id: str, quantity: int, *, currency: str | None = None) -> dict[str, Any]:
        del currency
        return {"variant_id": variant_id, "quantity": quantity}

    def lines_arguments(self, lines: Sequence[tuple[str, int]], *, currency: str | None = None) -> dict[str, Any]:
        """The page adds one line a call; `cart_calls` makes one call per line, so only a single line has arguments."""
        if len(lines) != 1:
            raise ValueError("add_to_cart takes one line a call; use cart_calls for more")
        return self.cart_arguments(*lines[0], currency=currency)

    def cart_calls(self, lines: Sequence[tuple[str, int]], *, currency: str | None = None) -> list[Call]:
        """One add_to_cart per line: the page has one cart, and each call adds a line to it and shows the cart."""
        return [(self.create, self.cart_arguments(vid, qty, currency=currency)) for vid, qty in lines]

    def read_arguments(self, cart_id: str) -> dict[str, Any]:
        del cart_id
        return {}

    def id_arguments(self, ident: str) -> dict[str, Any]:
        del ident
        return {}

    def is_cart_tool(self, name: str) -> bool:
        return name in CART_TOOLS

    # -- the checkout --------------------------------------------------------------------------------------------------

    def open_checkout_calls(self, cart_id: str, details: Mapping[str, Any]) -> list[Call]:
        del cart_id
        contact = {k: v for k, v in (("email", details.get("email")), ("phone", details.get("phone_number"))) if v}
        address = details.get("address")
        calls: list[Call] = [("start_checkout", {})]
        if contact:
            calls.append(("set_buyer_contact", contact))
        if isinstance(address, Mapping):
            calls.append(("set_shipping_address", address_arguments(address)))
        return calls

    def checkout_doc(self, payload: Any) -> Mapping[str, Any] | None:
        """The checkout a view shows, with what the view says around it (what is missing, the payment methods, an
        error); an order page is the completed checkout it stands for."""
        if not isinstance(payload, Mapping):
            return None
        checkout = payload.get("checkout")
        if isinstance(checkout, Mapping) and isinstance(checkout.get("id"), str):
            return {
                **checkout,
                "missing": list(payload.get("missing") or ()),
                "payment_handlers": list(payload.get("payment_handlers") or ()),
                "challenge": payload.get("challenge"),
                "error": payload.get("error"),
            }
        order = self.order_doc(payload)
        if order is None:
            return None
        return {
            "id": str(order["checkout_id"]),
            "status": "completed",
            "currency": order.get("currency"),
            "line_items": _bought_lines(order),
            "totals": list(order.get("totals") or ()),
            "order": {k: order[k] for k in ("id", "label") if k in order},
            "missing": [],
            "payment_handlers": [],
            "challenge": None,
            "error": None,
        }

    def order_doc(self, payload: Any) -> Mapping[str, Any] | None:
        order = payload.get("order") if isinstance(payload, Mapping) else None
        if not isinstance(order, Mapping):
            return None
        if all(isinstance(order.get(k), str) for k in ("id", "checkout_id")) and "line_items" in order:
            return order
        return None

    def order_view(self, doc: Mapping[str, Any]) -> Cart | None:
        return order_view(doc)

    def challenge_of(self, doc: Mapping[str, Any]) -> str | None:
        del doc
        return None

    def payment_failure(self, doc: Mapping[str, Any]) -> str | None:
        error = doc.get("error")
        if not isinstance(error, Mapping):
            return None
        code = str(error.get("code") or "")
        if any(word in code for word in PAYMENT_ERRORS):
            return str(error.get("message") or code)
        return None

    def handlers_of(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for method in doc.get("payment_handlers") or ():
            if isinstance(method, str) and method:
                out.setdefault(namespace_of(method), []).append({"id": method})
        return out

    # -- paying --------------------------------------------------------------------------------------------------------

    def complete_target(self, arguments: Mapping[str, Any]) -> str | None:
        del arguments
        return None

    def tokens_of(self, arguments: Mapping[str, Any]) -> set[str]:
        token = arguments.get("payment_token")
        return {token} if isinstance(token, str) and token else set()

    def credential_answer(self, credential: Credential, *, token: str | None = None) -> dict[str, Any]:
        return {
            self.payment_key: token or credential.token,
            "payment_handler": credential.handler_id,
            "amount": {"amount": credential.amount.amount, "currency": credential.amount.currency},
            "use": f"pass payment_token and payment_handler to {self.complete_checkout}",
        }

    def complete_arguments(self, checkout_id: str, answer: Mapping[str, Any]) -> dict[str, Any]:
        del checkout_id
        out: dict[str, Any] = {"payment_token": answer[self.payment_key]}
        handler = answer.get("payment_handler")
        if isinstance(handler, str) and handler:
            out["payment_handler"] = handler
        return out


def namespace_of(method_id: str) -> str:
    """The handler namespace a page's payment method id stands for, by its words; the id itself when none match."""
    lowered = method_id.lower()
    for words, namespace in NAMESPACE_WORDS:
        if any(word in lowered for word in words):
            return namespace
    return method_id


def address_arguments(address: Mapping[str, Any]) -> dict[str, Any]:
    """A UCP postal address as the shipping address form takes it (the same field names, without the contact names
    the shopper does not give)."""
    keys = (
        "street_address",
        "extended_address",
        "address_locality",
        "address_region",
        "postal_code",
        "address_country",
    )
    return {k: str(address[k]) for k in keys if isinstance(address.get(k), str) and address[k]}


def _bought_lines(order: Mapping[str, Any]) -> list[dict[str, Any]]:
    """An order's lines with the quantity bought as a number, as a checkout's lines carry it."""
    out: list[dict[str, Any]] = []
    for raw in order.get("line_items") or ():
        if not isinstance(raw, Mapping):
            continue
        quantity = raw.get("quantity")
        count = quantity.get("total", quantity.get("original")) if isinstance(quantity, Mapping) else quantity
        out.append({**raw, "quantity": count})
    return out


# -- the buyer's details through the page's forms ------


class WebMcpEntry:
    """WebMCP: `start_checkout` opens the checkout; the contact and address forms take the disclosed details; the
    delivery form chooses a shipping option, or a pickup location. A storefront says what is missing by naming the
    form that fills it in, and shows no shipping options until it has an address, so an agent that ships must give
    the address first: `missing` asks for it with the delivery choice."""

    @property
    def name(self) -> str:
        return "webmcp"

    def offered(self, tools: Collection[str]) -> bool:
        return {"set_buyer_contact", "set_shipping_address", "choose_fulfillment"} <= set(tools)

    def create(self, cart_id: str) -> Call:
        del cart_id
        return "start_checkout", {}

    def missing(self, doc: Mapping[str, Any]) -> list[str]:
        found: set[str] = set()
        for entry in doc.get("missing") or ():
            use = entry.get("use") if isinstance(entry, Mapping) else None
            found.update(MISSING_USES.get(str(use or ""), ()))
        delivery = doc.get("delivery")
        has_address = isinstance(delivery, Mapping) and isinstance(delivery.get("address"), Mapping)
        if "fulfillment" in found and not has_address:
            found.add("shipping_address")
        return [f for f in ASK_ORDER if f in found]

    def enter(self, doc: Mapping[str, Any], details: Details) -> list[Call]:
        del doc
        calls: list[Call] = []
        contact = {k: v for k, v in (("email", details.email), ("phone", details.phone)) if v is not None}
        if contact:
            calls.append(("set_buyer_contact", contact))
        if details.address is not None and details.method != "pickup":
            address = details.address
            arguments: dict[str, Any] = {"street_address": address.lines[0]}
            if len(address.lines) > 1:
                arguments["extended_address"] = address.lines[1]
            arguments["address_locality"] = address.city
            if address.region:
                arguments["address_region"] = address.region
            arguments["postal_code"] = address.postal_code
            arguments["address_country"] = address.country
            calls.append(("set_shipping_address", arguments))
        return calls

    def options(self, doc: Mapping[str, Any]) -> list[Option]:
        delivery = doc.get("delivery")
        if not isinstance(delivery, Mapping):
            return []
        found: list[Option] = []
        for option in delivery.get("options") or ():
            if not isinstance(option, Mapping) or not isinstance(option.get("id"), str):
                continue
            amount = option.get("amount")
            if not isinstance(amount, int) or isinstance(amount, bool):
                continue
            title = option["title"] if isinstance(option.get("title"), str) else option["id"]
            found.append(
                Option(
                    method_id="delivery",
                    method="shipping",
                    group_id="delivery",
                    option_id=option["id"],
                    title=title,
                    amount=amount,
                    kind=option_kind(title, option["id"], "shipping"),
                    days=stated_days(title),
                )
            )
        found.sort(key=lambda o: o.amount)
        for location in delivery.get("pickup_locations") or ():
            if not isinstance(location, Mapping) or not isinstance(location.get("id"), str):
                continue
            title = location["title"] if isinstance(location.get("title"), str) else location["id"]
            found.append(
                Option(
                    method_id="delivery",
                    method="pickup",
                    group_id="pickup",
                    option_id=location["id"],
                    title=title,
                    amount=0,
                    kind="pickup",
                    days=None,
                )
            )
        return found

    def select(self, doc: Mapping[str, Any], option: Option) -> Call:
        del doc
        if option.method == "pickup":
            return "choose_fulfillment", {"method": "pickup", "location_id": option.option_id}
        return "choose_fulfillment", {"method": "shipping", "option_id": option.option_id}

    def selected(self, doc: Mapping[str, Any]) -> Option | None:
        delivery = doc.get("delivery")
        if not isinstance(delivery, Mapping):
            return None
        if delivery.get("method") == "pickup":
            location = delivery.get("location")
            return next((o for o in self.options(doc) if o.method == "pickup" and o.option_id == location), None)
        chosen = delivery.get("selected_option")
        return next((o for o in self.options(doc) if o.method == "shipping" and o.option_id == chosen), None)

    def ready(self, doc: Mapping[str, Any]) -> bool:
        return doc.get("status") == "ready_for_complete"

    def total(self, doc: Mapping[str, Any]) -> Money | None:
        return total_of(doc)

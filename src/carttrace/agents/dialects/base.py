"""What a dialect is: how an agent speaks one protocol's tools and reads its documents, with UCP's shapes as the base.

Why one object and not branches in the agents. The reference agent and the checkout harness read the store's replies
at a dozen points (which product a search listed, which cart a call returned, what a checkout still lacks, whether
the payment went through, which order was placed), and the Qwen agent's harness guards the paying call. Each of those
reads differs by protocol only in shape: UCP's checkout is a document with `messages`, ACP's is a checkout session
with the same facts under other names, and a WebMCP storefront answers with the view its page shows. So the shapes
live here, one class per protocol, and the agents' loops never ask which protocol they are on.

The base speaks UCP (2026-04-08 and 2026-08-25 take the same shapes), through the readers in `carttrace.agents.checkout`;
`carttrace.agents.dialects.acp` and `carttrace.agents.dialects.webmcp` override what differs. A dialect is frozen data: it holds
tool names and shape flags, never state.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from carttrace.agents.checkout import (
    CHECKOUT_TOOLS,
    COMPLETE,
    ORDER_TOOLS,
    challenge_of,
    checkout_doc,
    handlers_of,
    order_doc,
    order_of,
    order_view,
    payment_failure,
    view_of,
)
from carttrace.agents.tools import cart_in, is_cart_tool
from carttrace.agents.wallet import (
    SHOP_PAY_NAMESPACE,
    Credential,
    CredentialRequest,
    HandlerTokenRequest,
    ShopTokenRequest,
)
from carttrace.core.money import Money
from carttrace.protocols.tools import ToolResult, ToolSpec
from carttrace.trace.schema import Cart

Call = tuple[str, dict[str, Any]]
"""A tool name and its arguments."""


@dataclass(frozen=True, slots=True)
class Dialect:
    """How an agent speaks UCP over MCP or REST; the base the other dialects override.

    Tool names: the catalog (`search`, `product`, `lookup`), the cart (`create`, `read`), the checkout
    (`create_checkout`, `update_checkout`, `get_checkout`, `complete_checkout`, `get_order`) and, where the protocol
    has one, the PSP's `delegate`. A name that is None is an operation the protocol does not have."""

    search: str = "search_catalog"
    product: str = "get_product"
    lookup: str = "lookup_catalog"
    create: str = "create_cart"
    read: str = "get_cart"
    create_checkout: str = "create_checkout"
    update_checkout: str = "update_checkout"
    get_checkout: str = "get_checkout"
    complete_checkout: str = COMPLETE
    get_order: str | None = "get_order"
    delegate: str | None = None
    """The PSP's delegate payment tool (ACP), which the harness calls for the wallet; None elsewhere."""
    checkout_tools: frozenset[str] = field(default=CHECKOUT_TOOLS)
    """Tools whose successful results are checkouts the harness keeps (`CheckoutBook`)."""
    order_tools: frozenset[str] = field(default=ORDER_TOOLS)
    """Tools whose successful results may be orders."""
    payment_key: str = "instrument"
    """The member of the payment answer (`credential_answer`) that the paying call takes."""
    browse: str | None = None
    """A tool that lists a page's links by kind, for a store whose search finds nothing (the website's list_links)."""
    page: str | None = None
    """A tool that reads one page as text (the website's open_page)."""
    index: str | None = None
    """A tool that searches the store's own product index (the website's browse_index over its sitemaps)."""
    boundary: str = "cart"
    """Where the journey ends: a cart, or on a store's website a selection (D31), which `done` names."""
    prepare_payment: str | None = None
    """A tool the paying call needs answered first, for what the payment answers (MPP's request_payment: the store's
    402 challenge); None where the protocol pays in one call."""
    intent: str | None = None
    """A tool that states the shopper's intent before the cart (AP2's send_intent_mandate); None elsewhere."""

    @property
    def name(self) -> str:
        return "ucp"

    @classmethod
    def from_specs(cls, specs: Mapping[str, ToolSpec]) -> Self:
        """The dialect as the listed tools' schemas shape it; UCP has one shape per release the agents share."""
        del specs
        return cls()

    def offered(self, tools: Collection[str]) -> bool:
        """Whether a surface listing `tools` speaks this dialect."""
        return {self.search, self.create} <= set(tools)

    # -- the catalog and the cart: arguments ---------------------------------------------------------------------------

    def search_arguments(self, query: str) -> dict[str, Any]:
        return {"catalog": {"query": query}}

    def next_search_arguments(self, arguments: Mapping[str, Any], payload: Any) -> dict[str, Any] | None:
        """The same search's next page, when the reply says there is one: UCP's `cursor` with `has_next_page`, or an
        offset page (`has_more`, `offset`, `limit`) as Medusa's Store API pages; None on the last page."""
        paging = payload.get("pagination") if isinstance(payload, Mapping) else None
        catalog = arguments.get("catalog")
        if not isinstance(paging, Mapping) or not isinstance(catalog, Mapping):
            return None
        cursor, offset, limit = paging.get("cursor"), paging.get("offset"), paging.get("limit")
        if paging.get("has_next_page") is True and isinstance(cursor, str) and cursor:
            return {"catalog": {**catalog, "pagination": {"cursor": cursor}}}
        if paging.get("has_more") is True and isinstance(offset, int) and isinstance(limit, int) and limit > 0:
            return {"catalog": {**catalog, "pagination": {"offset": offset + limit, "limit": limit}}}
        return None

    def product_arguments(self, product_id: str, selected: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
        """get_product for one product; `selected` (option name, value) pairs narrow it to the variants that carry
        them, as UCP's interactive option selection does."""
        catalog: dict[str, Any] = {"id": product_id}
        if selected:
            catalog["selected"] = [{"name": name, "label": label} for name, label in selected]
        return {"catalog": catalog}

    def lookup_arguments(self, product_ids: Sequence[str]) -> dict[str, Any]:
        return {"catalog": {"ids": list(product_ids)}}

    def cart_arguments(self, variant_id: str, quantity: int, *, currency: str | None = None) -> dict[str, Any]:
        """The call that starts the cart with one line; `currency` is the shopper's, for protocols whose cart is a
        checkout session priced in a stated currency."""
        return self.lines_arguments([(variant_id, quantity)], currency=currency)

    def lines_arguments(self, lines: Sequence[tuple[str, int]], *, currency: str | None = None) -> dict[str, Any]:
        """The call that starts the cart with one line per (variant id, quantity): a bundle's items (D20), or the one
        item of every other task."""
        del currency
        return {"cart": {"line_items": [{"item": {"id": vid}, "quantity": qty} for vid, qty in lines]}}

    def cart_calls(self, lines: Sequence[tuple[str, int]], *, currency: str | None = None) -> list[Call]:
        """The calls that make a cart holding `lines`, in order; the last one's reply is the cart. One call where the
        protocol starts a cart with every line at once (UCP, ACP); a page that adds one line a call overrides it."""
        return [(self.create, self.lines_arguments(lines, currency=currency))]

    def item_calls(self, items: Sequence[tuple[str, str, int]], *, currency: str | None = None) -> list[Call]:
        """The calls that hold `items`, each (product id, variant id, quantity): the cart's calls by default; a
        website names the product page of each variant it chooses."""
        return self.cart_calls([(variant_id, quantity) for _, variant_id, quantity in items], currency=currency)

    def with_intent(self, calls: list[Call], intent_id: str) -> list[Call]:
        """The cart calls naming the intent the shopper's wallet signed (AP2); unchanged where there is none."""
        del intent_id
        return calls

    def intent_arguments(self, items: Sequence[tuple[str, str, int]]) -> dict[str, Any]:
        """The intent call for `items` (variant id, title, quantity); only dialects with an `intent` tool have one."""
        raise NotImplementedError(f"the {self.name} dialect states no intent")

    def intent_id(self, result: ToolResult) -> str | None:
        """The id of the intent the store accepted, from the intent call's reply."""
        del result
        return None

    def prepare_arguments(self, checkout_id: str) -> dict[str, Any]:
        """The `prepare_payment` call for a checkout."""
        return {"checkout_id": checkout_id}

    def prepared(self, result: ToolResult) -> dict[str, Any] | None:
        """What the `prepare_payment` reply gives the paying call (MPP: the challenge's id), merged into the wallet's
        answer; None when the reply gives nothing to pay."""
        del result
        return None

    def browse_arguments(self, url: str, kind: str) -> dict[str, Any]:
        return {"url": url, "kind": kind}

    def index_arguments(self, query: str) -> dict[str, Any]:
        return {"query": query}

    def read_arguments(self, cart_id: str) -> dict[str, Any]:
        return {"id": cart_id}

    def id_arguments(self, ident: str) -> dict[str, Any]:
        return {"id": ident}

    # -- the catalog and the cart: documents ---------------------------------------------------------------------------

    def is_cart_tool(self, name: str) -> bool:
        """Whether a tool's result is the cart the episode is building."""
        return is_cart_tool(name)

    def cart_in(self, result: ToolResult) -> Cart | None:
        """The cart a successful cart call returned, or None."""
        return cart_in(result)

    # -- the checkout: arguments ---------------------------------------------------------------------------------------

    def checkout_arguments(self, cart_id: str, details: Mapping[str, Any]) -> dict[str, Any]:
        """A checkout from the cart (whose lines win), with the buyer's contact facts and a shipping address."""
        buyer = {"email": details["email"], "phone_number": details["phone_number"]}
        method = {"type": "shipping", "destinations": [dict(details["address"])]}
        return {
            "checkout": {"cart_id": cart_id, "line_items": [], "buyer": buyer, "fulfillment": {"methods": [method]}}
        }

    def open_checkout_calls(self, cart_id: str, details: Mapping[str, Any]) -> list[Call]:
        """The calls that open a checkout from the cart with the shopper's saved details (email, phone_number,
        address), in order; the last one's reply is the checkout."""
        return [(self.create_checkout, self.checkout_arguments(cart_id, details))]

    def refresh_calls(self, checkout_id: str) -> list[Call]:
        """The calls that read a checkout again, as the store now prices it, ready to pay (a store that repriced it at
        payment answers with the new total)."""
        return [(self.get_checkout, self.id_arguments(checkout_id))]

    def complete_arguments(self, checkout_id: str, answer: Mapping[str, Any]) -> dict[str, Any]:
        """The paying call, from the payment the wallet steps gave (`credential_answer`)."""
        return {"id": checkout_id, "checkout": {"payment": {"instruments": [dict(answer[self.payment_key])]}}}

    # -- the checkout: documents ---------------------------------------------------------------------------------------

    def checkout_doc(self, payload: Any) -> Mapping[str, Any] | None:
        """The checkout a reply shows, or None (an error reply shows none)."""
        return checkout_doc(payload)

    def order_doc(self, payload: Any) -> Mapping[str, Any] | None:
        """The order a reply shows, or None."""
        return order_doc(payload)

    def view_of(self, doc: Mapping[str, Any]) -> Cart | None:
        """A checkout's lines and totals as a cart (`cart_id` is the checkout's id)."""
        return view_of(doc)

    def order_view(self, doc: Mapping[str, Any]) -> Cart | None:
        """An order's lines and totals as a cart (`cart_id` is the order's id)."""
        return order_view(doc)

    def order_of(self, doc: Mapping[str, Any]) -> str | None:
        """The id of the order a completed checkout placed."""
        return order_of(doc)

    def order_label(self, doc: Mapping[str, Any]) -> str | None:
        """How the store names the order to a shopper (#1001), when the document says."""
        order = doc.get("order")
        label = order.get("label") if isinstance(order, Mapping) else doc.get("label")
        return label if isinstance(label, str) and label else None

    def challenge_of(self, doc: Mapping[str, Any]) -> str | None:
        """The URL of the bank's 3-D Secure challenge a checkout carries, for the wallet to answer; None when the
        protocol's page answers it itself, or none is waiting."""
        return challenge_of(doc)

    def payment_failure(self, doc: Mapping[str, Any]) -> str | None:
        """What the store said about a failed payment attempt, or None."""
        return payment_failure(doc)

    def handlers_of(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        """The payment handlers a checkout declares: namespace to its entries, as the wallet reads them."""
        return handlers_of(doc)

    # -- paying --------------------------------------------------------------------------------------------------------

    def complete_target(self, arguments: Mapping[str, Any]) -> str | None:
        """The checkout a paying call names, or None when the protocol's paying call names none (the harness then
        takes the store's latest checkout)."""
        target = arguments.get("id")
        return target if isinstance(target, str) and target else None

    def tokens_of(self, arguments: Mapping[str, Any]) -> set[str]:
        """The credential tokens a paying call presents."""
        checkout = arguments.get("checkout")
        payment = checkout.get("payment") if isinstance(checkout, Mapping) else None
        instruments = payment.get("instruments") if isinstance(payment, Mapping) else None
        found: set[str] = set()
        for instrument in instruments if isinstance(instruments, list) else ():
            credential = instrument.get("credential") if isinstance(instrument, Mapping) else None
            token = credential.get("token") if isinstance(credential, Mapping) else None
            if isinstance(token, str):
                found.add(token)
        return found

    def credential_request(
        self,
        checkout_id: str,
        handler_id: str,
        namespace: str,
        amount: Money,
        doc: Mapping[str, Any],
        *,
        now_ms: int,
    ) -> CredentialRequest:
        """The wallet request for a credential that pays `amount` for `checkout_id` through `handler_id`. The Shop
        Pay handler takes a single-use token minted per checkout (D73), not the persona's credential itself."""
        del doc, now_ms
        if namespace == SHOP_PAY_NAMESPACE:
            return ShopTokenRequest(checkout_id, handler_id, amount)
        return HandlerTokenRequest(checkout_id, handler_id, namespace, amount)

    def credential_answer(self, credential: Credential, *, token: str | None = None) -> dict[str, Any]:
        """The payment the paying call takes once the wallet steps issued a credential, under
        `payment_key`, and where it goes. `token` replaces the credential's own when the store's PSP vaulted it or
        the Shop Pay interface minted a single-use one from it (D73)."""
        instrument = credential.ucp_instrument()
        if token is not None:
            kind = "shop_token" if credential.kind == "shop_pay_token" else "token"
            instrument["credential"] = {"type": kind, "token": token}
        return {
            self.payment_key: instrument,
            "amount": {"amount": credential.amount.amount, "currency": credential.amount.currency},
            "use": f"put this instrument in {self.complete_checkout} as checkout.payment.instruments",
        }

    def delegation(self, credential: Credential, request: CredentialRequest) -> dict[str, Any]:
        """The PSP request that vaults `credential` for `request`'s allowance (ACP's delegate payment); only dialects
        with a `delegate` tool have one."""
        raise NotImplementedError(f"the {self.name} dialect delegates no payment")

    def vaulted(self, result: ToolResult) -> str | None:
        """The token the PSP vaulted a credential behind, from the delegate reply."""
        del result
        return None

"""Checkout on the agent's side: the store's UCP checkout documents as the harness reads them, and the wallet tools a
checkout row adds to the agent's own (decisions D13, D94).

Why wallet tools. On a checkout row the agent goes past the cart: it creates the checkout, gives the shopper's
delivery details, chooses a fulfillment option, gets the shopper's approval of the order and its total, pays, answers
3-D Secure, and ends with an order. Two of those steps are not the store's: the shopper's approval and the wallet.
Since D94 the agent reaches them through each rail's own official credential step (`carttrace.agents.wallet_tools`): UCP's
tokenize, Google Pay's loadPaymentData or Shop Pay's payment request for a UCP handler, ACP's delegate_payment, AP2's
mandate signing on the user's device and its credentials provider, MPP's create_credential, each under its official
name and in its official request and response shapes, and the shopper approves inside the step whose official flow
asks the buyer. The buyer's details are not among them: they go to the store through its own checkout operations
(carttrace.agents.entry), because that protocol is what the benchmark measures. The Qwen agent and the reference agent call
the same tools, so both leave the same turns in the trace.

What the tools read. Never the agent's word: the harness keeps the store's latest reply for each checkout and order
(`CheckoutBook`), from the results of the agent's own calls, and every wallet tool reads the checkout from there.

The guard. The wallet's tokens are a persona's opaque test tokens, so a model that saw one could present it again.
The harness therefore lets the paying call through only when it pays with a credential the wallet issued for that
checkout after the shopper approved it, at its current total, and not presented before (`CheckoutHarness.guard`);
anything else is refused below the agent as `harness_blocked`, never sent, and the judge still checks every order
against the shopper's approval.

Protocols. The documents here are UCP's; the harness reads them through the dialect the agent speaks
(`carttrace.agents.dialects`, given with `speak`). On ACP the wallet's card is vaulted at the store's PSP through the
surface's own delegate_payment, with the agent's allowance, and the agent pays with the vaulted token. On a store's
Shop Pay handler the wallet's Shop Pay credential is minted into a single-use Shop Token at the store's Shop Pay
interface (D73).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal

import httpx

from carttrace.agents import wallet_tools as wt
from carttrace.agents.service import ConsentRequest, Shopper, TurnSink
from carttrace.agents.slots import money
from carttrace.agents.tools import REFUSED, payload_of
from carttrace.agents.wallet import (
    SHOP_PAY_NAMESPACE,
    ChallengeOutcome,
    Credential,
    CredentialRequest,
    DelegatedTokenRequest,
    PaymentChallengeRequest,
    Refusal,
    ShopTokenRequest,
    Wallet,
)
from carttrace.core.canon import digest, dumps
from carttrace.core.clock import Clock, SystemClock, iso
from carttrace.core.errors import MCError, ValidationError
from carttrace.core.money import Money, to_major, to_minor
from carttrace.protocols.tools import ToolResult
from carttrace.trace.carts import from_ucp
from carttrace.trace.schema import Cart

if TYPE_CHECKING:
    from carttrace.agents.dialects.base import Dialect

StoreCall = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]
"""How the harness calls one of the store's own tools for the wallet (ACP's delegate_payment): the agent's surface
call, recorded as any store call is."""

THREE_DS_ACTION: Final = "dev.ucp.common.payment.three_ds_challenge"
"""The 3-D Secure challenge Action of UCP's payment authentication extension (2026-08-25,
common/payment_authentication.json)."""
CHALLENGE_HANDOFF: Final = "payment_authentication_required"
"""The message code of a checkout handing the bank's challenge to the buyer at continue_url (releases without the
extension)."""
PAYMENT_FAILED: Final = "payment_failed"
"""The message code of a checkout whose last payment attempt was declined or failed authentication."""
COMPLETE: Final = "complete_checkout"
CHECKOUT_TOOLS: Final = frozenset(
    {"create_checkout", "update_checkout", "get_checkout", "complete_checkout", "cancel_checkout"}
)
ORDER_TOOLS: Final = frozenset({"get_order"})

HARNESS_TOOLS: Final = wt.WALLET_TOOLS
"""Every wallet tool of every rail (`carttrace.agents.wallet_tools`), reserved beside the store's tools."""
CONSENT_TOOLS: Final = wt.CONSENT_TOOLS
"""The wallet tools whose call asks the shopper to approve the payment."""
HARNESS_SURFACE: Final = "harness"
"""What the trace names as the surface of a wallet tool's turns."""
GOOGLE_PAY_NAMESPACE: Final = "com.google.pay"
HARNESS_REFUSALS: Final = frozenset(
    {"shop_pay_unavailable", "shop_pay_refused", "shop_pay_failed", "challenge_unavailable"}
)
"""Wallet refusals that are the harness's failing (an interface or the bank's page unreachable), never the shopper's
or the store's: the row ends harness_error."""
_WRAPPERS = ("structuredContent", "result", "checkout", "order")


# -- reading the store's documents -------------------------------------------------------------------------------------


def unwrap(data: Any, *, needs: Sequence[str]) -> Mapping[str, Any] | None:
    """The document in a reply that carries every key in `needs`, under any of the usual wrappers, or None."""
    current = data
    for _ in range(len(_WRAPPERS) + 1):
        if not isinstance(current, Mapping):
            return None
        if all(key in current for key in needs) and isinstance(current.get("id"), str):
            return current
        current = next((current[k] for k in _WRAPPERS if k in current), None)
    return None


def checkout_doc(data: Any) -> Mapping[str, Any] | None:
    """The UCP checkout in a reply, or None (an error response holds none)."""
    doc = unwrap(data, needs=("id", "status", "line_items"))
    return None if doc is None or _status(doc) == "error" else doc


def order_doc(data: Any) -> Mapping[str, Any] | None:
    """The UCP order in a get_order reply, or None."""
    doc = unwrap(data, needs=("id", "checkout_id", "line_items"))
    return None if doc is None or _status(doc) == "error" else doc


def _status(doc: Mapping[str, Any]) -> str | None:
    ucp = doc.get("ucp")
    status = ucp.get("status") if isinstance(ucp, Mapping) else None
    return status if isinstance(status, str) else None


def total_of(doc: Mapping[str, Any]) -> Money | None:
    """The document's total (its `totals` entry of type total) in its currency."""
    currency = doc.get("currency")
    for entry in doc.get("totals") or ():
        if isinstance(entry, Mapping) and entry.get("type") == "total":
            amount = entry.get("amount")
            if isinstance(amount, int) and not isinstance(amount, bool) and isinstance(currency, str):
                return Money(amount, currency.upper())
    return None


def handlers_of(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The payment handlers a checkout declares (`ucp.payment_handlers`): namespace to its entries. UCP 2026-01-11
    lists them in the checkout's `payment.handlers` instead, each naming its namespace (`name`), D71."""
    ucp = doc.get("ucp")
    handlers = ucp.get("payment_handlers") if isinstance(ucp, Mapping) else None
    if isinstance(handlers, Mapping) and handlers:
        return {str(k): v for k, v in handlers.items()}
    payment = doc.get("payment")
    listed = payment.get("handlers") if isinstance(payment, Mapping) else None
    out: dict[str, Any] = {}
    for entry in listed if isinstance(listed, list) else ():
        if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
            out.setdefault(entry["name"], []).append(dict(entry))
    return out


def challenge_of(doc: Mapping[str, Any]) -> str | None:
    """The URL of the 3-D Secure challenge a checkout carries: its Action, or the continue_url of a buyer handoff."""
    actions = doc.get("actions")
    listed = actions.get(THREE_DS_ACTION) if isinstance(actions, Mapping) else None
    for action in listed or ():
        config = action.get("config") if isinstance(action, Mapping) else None
        url = config.get("url") if isinstance(config, Mapping) else None
        if isinstance(url, str) and url:
            return url
    handoff = any(m.get("code") == CHALLENGE_HANDOFF for m in _messages(doc))
    url = doc.get("continue_url")
    return url if handoff and isinstance(url, str) and url else None


def order_of(doc: Mapping[str, Any]) -> str | None:
    """The id of the order a completed checkout placed."""
    order = doc.get("order")
    oid = order.get("id") if isinstance(order, Mapping) else None
    return oid if doc.get("status") == "completed" and isinstance(oid, str) and oid else None


def payment_failure(doc: Mapping[str, Any]) -> str | None:
    """What the store said about a failed payment attempt (declined, or authentication failed), or None."""
    return next((str(m.get("content") or "") for m in _messages(doc) if m.get("code") == PAYMENT_FAILED), None)


@dataclass(frozen=True, slots=True)
class FulfillmentChoice:
    """One fulfillment option a checkout offers, and where it sits: what update_checkout must name to select it."""

    method_id: str
    group_id: str
    option_id: str
    title: str
    amount: int


def shipping_options(doc: Mapping[str, Any]) -> list[FulfillmentChoice]:
    """Every option the checkout's shipping methods offer, cheapest first (ties keep the store's order)."""
    fulfillment = doc.get("fulfillment")
    methods = fulfillment.get("methods") if isinstance(fulfillment, Mapping) else None
    found: list[FulfillmentChoice] = []
    for method in methods or ():
        if not isinstance(method, Mapping) or method.get("type") != "shipping" or not isinstance(method.get("id"), str):
            continue
        for group in method.get("groups") or ():
            if not isinstance(group, Mapping) or not isinstance(group.get("id"), str):
                continue
            for option in group.get("options") or ():
                amount = _option_total(option)
                if amount is None or not isinstance(option.get("id"), str):
                    continue
                title = option.get("title")
                found.append(
                    FulfillmentChoice(
                        method_id=method["id"],
                        group_id=group["id"],
                        option_id=option["id"],
                        title=title if isinstance(title, str) else option["id"],
                        amount=amount,
                    )
                )
    return sorted(found, key=lambda c: c.amount)


def _option_total(option: Any) -> int | None:
    """An option's price: its totals entry of type total, or its one priced entry of another type (Medusa's pack
    writes `fulfillment`, D94)."""
    if not isinstance(option, Mapping):
        return None
    priced = [
        e
        for e in option.get("totals") or ()
        if isinstance(e, Mapping) and isinstance(e.get("amount"), int) and not isinstance(e.get("amount"), bool)
    ]
    found = next((e for e in priced if e.get("type") == "total"), priced[0] if len(priced) == 1 else None)
    return int(found["amount"]) if found is not None else None


def selected_option(doc: Mapping[str, Any]) -> FulfillmentChoice | None:
    """The shipping option the checkout has selected, if any."""
    fulfillment = doc.get("fulfillment")
    methods = fulfillment.get("methods") if isinstance(fulfillment, Mapping) else None
    chosen = {
        g.get("selected_option_id")
        for m in methods or ()
        if isinstance(m, Mapping)
        for g in m.get("groups") or ()
        if isinstance(g, Mapping)
    }
    return next((c for c in shipping_options(doc) if c.option_id in chosen), None)


def line_inputs(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The checkout's lines as update_checkout takes them back (a full replacement keeps only what it is sent)."""
    out: list[dict[str, Any]] = []
    for line in doc.get("line_items") or ():
        item = line.get("item") if isinstance(line, Mapping) else None
        if isinstance(item, Mapping) and isinstance(item.get("id"), str) and isinstance(line.get("quantity"), int):
            entry: dict[str, Any] = {"item": {"id": item["id"]}, "quantity": line["quantity"]}
            if isinstance(line.get("id"), str):
                entry["id"] = line["id"]
            out.append(entry)
    return out


def view_of(doc: Mapping[str, Any]) -> Cart | None:
    """A checkout's lines and totals as a cart (`cart_id` is the checkout's id), or None when it cannot be read."""
    try:
        return from_ucp(doc)
    except ValidationError:
        return None


def order_view(doc: Mapping[str, Any]) -> Cart | None:
    """An order's lines and totals as a cart (`cart_id` is the order's id), or None when it cannot be read. A UCP
    order line's quantity is an object (original, total, fulfilled); the lines bought are its `total` (after
    returns), else its `original`."""
    lines: list[dict[str, Any]] = []
    for raw in doc.get("line_items") or ():
        if not isinstance(raw, Mapping):
            return None
        quantity = raw.get("quantity")
        count = quantity.get("total", quantity.get("original")) if isinstance(quantity, Mapping) else quantity
        lines.append({**raw, "quantity": count})
    return view_of({**doc, "line_items": lines})


def _messages(doc: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [m for m in doc.get("messages") or () if isinstance(m, Mapping)]


@dataclass(slots=True)
class CheckoutBook:
    """The store's latest word on each checkout and order in this episode, from the results of the agent's calls."""

    checkouts: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    orders: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    last_checkout: str | None = None
    last_order: str | None = None

    def observe(self, tool: str, result: ToolResult, dialect: Dialect | None = None) -> None:
        """Keep what a checkout or order tool's result says, read in `dialect` (UCP when None). Errors and other
        tools change nothing."""
        d = dialect or _ucp()
        if result.is_error or tool not in d.checkout_tools | d.order_tools:
            return
        data = payload_of(result)
        if tool in d.order_tools:
            order = d.order_doc(data)
            if order is not None:
                self.orders[str(order["id"])] = order
                self.last_order = str(order["id"])
        if tool not in d.checkout_tools:
            return
        doc = d.checkout_doc(data)
        if doc is None:
            return
        self.checkouts[str(doc["id"])] = doc
        self.last_checkout = str(doc["id"])
        placed = d.order_of(doc)
        if placed is not None:
            self.last_order = placed

    def get(self, checkout_id: str) -> Mapping[str, Any] | None:
        return self.checkouts.get(checkout_id)

    @property
    def order_id(self) -> str | None:
        """The order the store placed, as a completed checkout or get_order said."""
        return self.last_order


# -- the wallet tools --------------------------------------------------------------------------------------------------

Status = Literal["approved", "declined", "failed"]
HarnessRun = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]
"""How the reference agent calls a wallet tool: the call and its result written to the trace."""


@dataclass(frozen=True, slots=True)
class Authorized:
    """Where the reference agent's run of a rail's wallet steps ended: approved with the payment the paying call
    takes (`answer`, as the dialect's `credential_answer` gives it), declined by the shopper, or failed."""

    status: Status
    answer: Mapping[str, Any] | None = None
    reason: str = ""


class CheckoutHarness:
    """The wallet tools of a checkout row, over one wallet and one shopper, and the guard on paying. One per
    episode."""

    def __init__(self, wallet: Wallet, shopper: Shopper, sink: TurnSink, *, clock: Clock | None = None) -> None:
        self.wallet = wallet
        self.shopper = shopper
        self.sink = sink
        self.clock: Clock = clock or SystemClock()
        self.book = CheckoutBook()
        self.dialect: Dialect | None = None
        """The protocol the agent speaks, once it has listed its surfaces' tools (`speak`); UCP until then."""
        self.store: StoreCall | None = None
        self._paid_with: set[int] = set()
        """Indexes into `wallet.issued` of the credentials a completion already presented."""
        self._vaulted: dict[int, str] = {}
        """Indexes into `wallet.issued` to the token the agent pays with when it is not the credential's own: the
        one the store's PSP vaulted it behind (ACP), or the Shop Token minted from it (Shop Pay, D73)."""
        self._challenges: dict[str, Mapping[str, Any]] = {}
        """MPP: each 402 challenge request_payment returned, by id."""
        self._mandates: dict[str, dict[str, Any]] = {}
        """AP2 v0.1.0: the payment mandate contents created for each cart, by cart id."""
        self._reply = ""
        """What the shopper last said when asked to approve a payment."""
        self.failed: str | None = None
        """Why the harness itself failed a wallet step (its own exception, the Shop Pay interface or the bank's page
        unreachable, no PSP to vault at), or None. The agents end the row as harness_error on it: such a failure is
        never the agent's outcome (the v1 review's #9)."""
        self._actions: dict[str, str] = {}
        """MPP: the bank's challenge page a payment-action-required answer named, by checkout id."""
        self._unwritten: Callable[[], None] | None = None
        """Writes the agent's paying call to the trace once the shopper has answered the wallet's sheet (`call`)."""

    def speak(self, dialect: Dialect, *, store: StoreCall | None = None) -> None:
        """Read the store's documents in `dialect`, and reach the store's own tools through `store` where the wallet's
        protocol step needs one (ACP's delegate_payment)."""
        self.dialect = dialect
        self.store = store

    @property
    def spoken(self) -> Dialect:
        return self.dialect if self.dialect is not None else _ucp()

    @property
    def kit(self) -> wt.Kit:
        return wt.kit_for(self.spoken.name)

    def tools(self) -> tuple[str, ...]:
        """The wallet tools this row gives the agent: its rail's kit, and of UCP's handler tools only those for an
        instrument the shopper's wallet holds (a platform offers Google Pay or Shop Pay only to a buyer who has it);
        never three_ds_challenge where the wallet offers none (D111)."""
        kinds = {i.kind for i in self.wallet.persona.instruments}
        held = {wt.GOOGLE_PAY: "google_pay", wt.SHOP_PAY: "shop_pay"}
        listed = tuple(n for n in self.kit.tools if n not in held or held[n] in kinds)
        if not getattr(self.wallet, "offers_challenge", True):
            # the store's PSP decides 3-D Secure at once (D111): no bank page, so no challenge tool
            listed = tuple(n for n in listed if n != wt.CHALLENGE)
        if self.wallet.prefer == SHOP_PAY_NAMESPACE and wt.SHOP_PAY in listed:
            # a Shop Pay row (D73, D89): the shopper pays this store with Shop Pay, so the wallet offers that step
            # alone; offering every instrument let the agent pay a Shop Pay row with a card built to decline
            return tuple(n for n in listed if n not in (wt.TOKENIZE, wt.GOOGLE_PAY))
        return listed

    def handles(self, name: str) -> bool:
        return name in self.tools()

    def definitions(self) -> list[dict[str, Any]]:
        return wt.definitions(self.tools())

    @staticmethod
    def stands_in(name: str) -> bool:
        """Whether a store tool of this name is replaced by the wallet's own (ACP's delegate_payment)."""
        return name in wt.STANDS_IN

    @staticmethod
    def asks_shopper(name: str) -> bool:
        return name in wt.CONSENT_TOOLS

    def guard(self, tool: str, arguments: Mapping[str, Any]) -> ToolResult | None:
        """Refuse, below the agent and before anything is sent, a completion that does not pay with a credential the
        wallet issued for this checkout, at the checkout's current total, and not presented before. A token the agent
        saw once, or no token at all (the sandbox's own handler takes none), never pays without the shopper's
        approval. None lets the call go to the store."""
        d = self.spoken
        if tool != d.complete_checkout:
            return None
        checkout_id = d.complete_target(arguments) or self.book.last_checkout
        doc = self.book.get(checkout_id) if isinstance(checkout_id, str) else None
        total = total_of(doc) if doc is not None else None
        presented = d.tokens_of(arguments)
        for index, credential in enumerate(self.wallet.issued):
            token = self._vaulted.get(index, credential.token)
            fresh = index not in self._paid_with and credential.checkout_id == checkout_id
            if fresh and token in presented and (total is None or credential.amount == total):
                self._paid_with.add(index)
                return None
        steps = " or ".join(n for n in self.tools() if n in wt.CONSENT_TOOLS) or "the wallet"
        content = (
            f"not sent: pay only with a token the wallet issued for this checkout after the shopper approved its "
            f"current total ({steps}); each one pays once"
        )
        return ToolResult(content=content, is_error=True, error_code=REFUSED)

    def observe(self, tool: str, result: ToolResult) -> None:
        self.book.observe(tool, result, self.spoken)
        if self.spoken.prepare_payment is None:
            return
        data = payload_of(result)
        if not isinstance(data, Mapping):
            return
        for challenge in data.get("challenges") or ():
            # a 402 from request_payment, or the fresh challenge a refused pay answers with
            if isinstance(challenge, Mapping) and isinstance(challenge.get("id"), str):
                self._challenges[challenge["id"]] = dict(challenge)
        details = data.get("details")
        if isinstance(details, Mapping) and isinstance(details.get("actionUrl"), str):
            # MPP's payment-action-required problem names the bank's page and the checkout it holds
            checkout = details.get("checkoutId") or self.book.last_checkout
            if isinstance(checkout, str):
                self._actions[checkout] = details["actionUrl"]

    @property
    def checkout_id(self) -> str | None:
        return self.book.last_checkout

    @property
    def order_id(self) -> str | None:
        return self.book.order_id

    def checkout_view(self) -> Cart | None:
        """The latest checkout as the shopper is shown it (the consent's view), or None before one exists."""
        checkout_id = self.book.last_checkout
        doc = self.book.get(checkout_id) if checkout_id is not None else None
        return self.spoken.view_of(doc) if doc is not None else None

    @staticmethod
    def brief(wallet: Wallet) -> str:
        """The shopper's saved checkout details, as the Qwen agent's system prompt carries them for a task without a
        buyer block. A task with one never gets them: its shopper gives each detail only when asked."""
        details = wallet.shipping_details()
        return (
            "Checkout details the shopper saved in their wallet, for the store's checkout and nothing else: "
            f"email {details['email']}; phone {details['phone_number']}; shipping address (a UCP postal address) "
            f"{dumps(details['address'])}."
        )

    async def call(
        self, name: str, arguments: Mapping[str, Any], *, write_call: Callable[[], None] | None = None
    ) -> ToolResult:
        """Run one wallet tool. Never raises for what the agent sent: a bad call is an error result it can read, in
        the shape of the official step it imitates.

        `write_call` writes the agent's call to the trace. A paying step's call is written just after the shopper
        answers the wallet's sheet, so the record reads sheet, answer, paying call, result: the paying call on
        record is the one the shopper's answer released, and the approval comes before every paying call and the
        completion on every rail (the v1 release's consent_before_completion read the shopper approving after
        shop_pay_payment_request). A call refused before any sheet is written before its result, as any other."""
        if write_call is not None and name not in wt.CONSENT_TOOLS:
            write_call()
        else:
            self._unwritten = write_call
        try:
            return await self._run(name, arguments)
        finally:
            self._write_call()

    def _write_call(self) -> None:
        write, self._unwritten = self._unwritten, None
        if write is not None:
            write()

    async def _run(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        handlers: dict[str, Callable[[Mapping[str, Any]], Awaitable[ToolResult]]] = {
            wt.TOKENIZE: self._tokenize,
            wt.GOOGLE_PAY: self._google_pay,
            wt.SHOP_PAY: self._shop_pay,
            wt.DELEGATE: self._delegate_payment,
            wt.AP2_METHODS: self._ap2_methods,
            wt.AP2_TOKEN: self._ap2_token,
            wt.AP2_MANDATE: self._ap2_mandate,
            wt.AP2_SIGN: self._ap2_sign,
            wt.AP2_CHECKOUT_MANDATE: self._ap2_checkout_mandate,
            wt.AP2_ISSUE: self._ap2_issue,
            wt.MPP_CREDENTIAL: self._mpp_credential,
            wt.CHALLENGE: self._challenge,
        }
        handler = handlers.get(name) if self.handles(name) else None
        if handler is None:
            return _error("unknown_tool", f"there is no wallet tool named {name} on this checkout")
        try:
            return await handler(arguments)
        except (MCError, httpx.HTTPError) as exc:
            self.failed = f"{name} failed: {exc}"
            return _error("harness_error", f"{name} failed: {exc}", "try again, or tell the shopper what failed")

    # -- shared steps ------------------------------------------------------------------------------------------------

    def _doc(self, checkout_id: Any) -> tuple[str, Mapping[str, Any]] | None:
        if not isinstance(checkout_id, str) or not checkout_id:
            return None
        doc = self.book.get(checkout_id)
        return (checkout_id, doc) if doc is not None else None

    async def _approve(self, checkout_id: str, doc: Mapping[str, Any], total: Money, sheet: str) -> bool | None:
        """Show the shopper the store's checkout and `total` on the wallet's confirmation (`sheet`), record their
        answer in the wallet, and say whether they approved; None when the checkout has nothing to show."""
        view = self.spoken.view_of(doc)
        if view is None:
            return None
        request = ConsentRequest(checkout_id=checkout_id, total=total, checkout=view, message=sheet)
        # the wallet's own screen, which the model never wrote: a system turn, so no judge credits the agent with
        # stating the total (the v1 review's #34)
        self.sink.turn("system", "message", text=sheet)
        reply = await self.shopper.consent(request)
        self.sink.turn(
            "shopper",
            "message",
            text=reply.text,
            arguments={"consent": reply.annotation(request)},
            origin=reply.origin(),
            **reply.turn_extras(),
        )
        self._write_call()
        self._reply = reply.text.strip()
        if reply.given:
            self.wallet.approve(checkout_id, total)
        else:
            self.wallet.refuse(checkout_id)
        return reply.given

    def _why(self, refused: str) -> str:
        """A refusal's words with the shopper's own, as the platform shows the agent what the buyer said."""
        return f'{refused}; the shopper said: "{self._reply}"' if self._reply else refused

    def _sheet(self, where: str, total: Money, what: str = "") -> str:
        """What the wallet's confirmation screen says."""
        tail = f" {what}" if what else ""
        return f"{where}: pay {_shown(total)}{tail}. Confirm?"

    def _index_of(self, credential: Credential) -> int:
        return next(i for i, c in enumerate(self.wallet.issued) if c is credential)

    # -- UCP ---------------------------------------------------------------------------------------------------------

    def _handler(self, doc: Mapping[str, Any], handler_id: Any) -> str | None:
        return _namespace(self.spoken.handlers_of(doc), handler_id) if isinstance(handler_id, str) else None

    async def _tokenize(self, arguments: Mapping[str, Any]) -> ToolResult:  # noqa: PLR0911 - one official error per refusal
        binding = arguments.get("binding")
        bound = binding if isinstance(binding, Mapping) else {}
        if bound.get("type") != wt.CHECKOUT_BINDING:
            return wt.ucp_error("invalid", f"binding.type must be {wt.CHECKOUT_BINDING}, with the checkout's id")
        found = self._doc(bound.get("id"))
        if found is None:
            return wt.ucp_error("not_found", f"no checkout {bound.get('id')!r} has come back from the store")
        checkout_id, doc = found
        handler_id = arguments.get("handler_id")
        namespace = self._handler(doc, handler_id)
        if namespace is None or not isinstance(handler_id, str):
            return wt.ucp_error("not_found", f"the checkout declares no payment handler {handler_id!r}")
        if namespace in (GOOGLE_PAY_NAMESPACE, SHOP_PAY_NAMESPACE):
            other = wt.GOOGLE_PAY if namespace == GOOGLE_PAY_NAMESPACE else wt.SHOP_PAY
            return wt.ucp_error("invalid", f"the {namespace} handler issues its own token: use {other}")
        instrument = self.wallet.persona.instrument_for(namespace)
        if instrument is None:
            kinds = ", ".join(i.kind for i in self.wallet.persona.instruments)
            return wt.ucp_error(
                "invalid",
                f"the wallet holds no credential the {namespace} handler takes (it holds: {kinds})",
                "requires_buyer_input",
            )
        total = total_of(doc)
        if total is None:
            return wt.ucp_error("invalid", "the checkout shows no total to confirm")
        what = "with your saved card" if instrument.kind == "card" else "on delivery or by transfer"
        given = await self._approve(checkout_id, doc, total, self._sheet("Your wallet", total, what))
        if not given:
            return wt.ucp_error(
                "buyer_declined", self._why("the shopper did not confirm the payment"), "requires_buyer_input"
            )
        issued = await self._issue(checkout_id, doc, handler_id, namespace, total)
        if isinstance(issued, ToolResult):
            return _as_ucp(issued)
        credential, token = issued
        return wt.ok({"token": token or credential.token})

    async def _google_pay(self, arguments: Mapping[str, Any]) -> ToolResult:  # noqa: PLR0911 - one official error per refusal
        info = arguments.get("transactionInfo")
        info = info if isinstance(info, Mapping) else {}
        if arguments.get("apiVersion") != 2 or arguments.get("apiVersionMinor") != 0:
            return wt.google_pay_error("DEVELOPER_ERROR", "apiVersion must be 2 and apiVersionMinor 0")
        found = self._doc(info.get("transactionId") or self.book.last_checkout)
        if found is None:
            return wt.google_pay_error("DEVELOPER_ERROR", "transactionInfo.transactionId names no checkout")
        checkout_id, doc = found
        handlers = self.spoken.handlers_of(doc).get(GOOGLE_PAY_NAMESPACE)
        entry = handlers[0] if isinstance(handlers, list) and handlers else handlers
        handler_id = entry.get("id") if isinstance(entry, Mapping) else None
        if not isinstance(handler_id, str):
            return wt.google_pay_error("DEVELOPER_ERROR", "the checkout declares no com.google.pay payment handler")
        currency = info.get("currencyCode")
        minor = to_minor(info.get("totalPrice"), currency) if isinstance(currency, str) else None
        if minor is None or not isinstance(currency, str):
            return wt.google_pay_error("DEVELOPER_ERROR", "transactionInfo needs totalPrice and currencyCode")
        instrument = self.wallet.persona.instrument_for(GOOGLE_PAY_NAMESPACE)
        if instrument is None:
            return wt.google_pay_error("BUYER_ACCOUNT_ERROR", "the shopper has no payment method in Google Pay")
        stated = Money(minor, currency.upper())
        given = await self._approve(checkout_id, doc, stated, self._sheet("Google Pay", stated))
        if not given:
            return wt.google_pay_error("CANCELED", self._why("the shopper closed the payment sheet"))
        issued = await self._issue(checkout_id, doc, handler_id, GOOGLE_PAY_NAMESPACE, stated)
        if isinstance(issued, ToolResult):
            return wt.google_pay_error("DEVELOPER_ERROR", issued.content if isinstance(issued.content, str) else "")
        credential, _ = issued
        # what Google Pay's PaymentMethodData shows a buyer: the network and four digits that stand for the card's
        # last four (cardDetails), read from the alias and naming no card; the v1 release showed "Visa (Google Pay
        # test)", a test's name
        brand = credential.brand or "visa"
        last4 = f"{int(digest(['gpay-last4', credential.token], 8), 16) % 10_000:04d}"
        method: dict[str, Any] = {
            "type": "CARD",
            "description": f"{brand.title()} \u2022\u2022\u2022\u2022 {last4}",
            "info": {"cardNetwork": brand.upper(), "cardDetails": last4},
            "tokenizationData": {"type": "PAYMENT_GATEWAY", "token": credential.token},
        }
        return wt.ok({"apiVersion": 2, "apiVersionMinor": 0, "paymentMethodData": method})

    async def _shop_pay(self, arguments: Mapping[str, Any]) -> ToolResult:  # noqa: PLR0911 - one official error per refusal
        found = self._doc(arguments.get("checkout_id"))
        if found is None:
            return wt.ucp_error(
                "not_found", f"no checkout {arguments.get('checkout_id')!r} has come back from the store"
            )
        checkout_id, doc = found
        handler_id = arguments.get("handler_id")
        if self._handler(doc, handler_id) != SHOP_PAY_NAMESPACE or not isinstance(handler_id, str):
            return wt.ucp_error("not_found", f"the checkout declares no {SHOP_PAY_NAMESPACE} handler {handler_id!r}")
        if self.wallet.persona.instrument_for(SHOP_PAY_NAMESPACE) is None:
            return wt.ucp_error("invalid", "the shopper has no Shop Pay account", "requires_buyer_input")
        total = total_of(doc)
        if total is None:
            return wt.ucp_error("invalid", "the checkout shows no total for the payment request")
        given = await self._approve(checkout_id, doc, total, self._sheet("Shop Pay", total))
        if not given:
            return wt.ucp_error(
                "buyer_declined", self._why("the shopper did not confirm in Shop Pay"), "requires_buyer_input"
            )
        issued = await self._issue(checkout_id, doc, handler_id, SHOP_PAY_NAMESPACE, total)
        if isinstance(issued, ToolResult):
            return _as_ucp(issued)
        _, token = issued
        return wt.ok({"type": "shop_token", "token": token})

    # -- ACP ---------------------------------------------------------------------------------------------------------

    async def _delegate_payment(self, arguments: Mapping[str, Any]) -> ToolResult:  # noqa: PLR0911 - one official error per refusal
        allowance = arguments.get("allowance")
        given_allowance = allowance if isinstance(allowance, Mapping) else {}
        missing = [
            k
            for k in ("reason", "max_amount", "currency", "checkout_session_id", "merchant_id", "expires_at")
            if k not in given_allowance
        ]
        if missing:
            return wt.acp_error(
                "invalid_request", f"allowance.{missing[0]} is required", param=f"$.allowance.{missing[0]}"
            )
        found = self._doc(given_allowance.get("checkout_session_id"))
        if found is None:
            return wt.acp_error("invalid_request", "no such checkout session", param="$.allowance.checkout_session_id")
        checkout_id, doc = found
        amount = _money(given_allowance.get("max_amount"), given_allowance.get("currency"))
        if amount is None or given_allowance.get("reason") != "one_time":
            return wt.acp_error("invalid_request", "allowance needs reason one_time, max_amount and currency")
        if not any(i.kind == "card" for i in self.wallet.persona.instruments):
            return wt.acp_error(
                "invalid_card", "the shopper's wallet holds no card to delegate", param="$.payment_method"
            )
        given = await self._approve(checkout_id, doc, amount, self._sheet("Pay", amount, "with your saved card"))
        if not given:
            return wt.acp_error(
                "user_declined", self._why("the shopper did not confirm the payment"), param="$.allowance"
            )
        handlers = self.spoken.handlers_of(doc)
        handler_id = next(
            (
                e.get("id")
                for es in handlers.values()
                for e in (es if isinstance(es, list) else [es])
                if isinstance(e, Mapping) and isinstance(e.get("id"), str)
            ),
            None,
        )
        request = DelegatedTokenRequest(
            checkout_id,
            amount,
            str(given_allowance["merchant_id"]),
            str(given_allowance["expires_at"]),
            handler_id=handler_id,
        )
        issued = self.wallet.issue(request)
        if isinstance(issued, Refusal):
            return wt.acp_error(issued.code, issued.message, param="$.allowance")
        extra = {k: arguments[k] for k in ("billing_address", "metadata") if isinstance(arguments.get(k), Mapping)}
        return await self._vault(issued, request, extra)

    async def _vault(self, credential: Credential, request: CredentialRequest, extra: Mapping[str, Any]) -> ToolResult:
        """Vault the wallet's credential at the store's PSP through the store's own delegate_payment, the vault's
        payment method beside the agent's allowance, and answer what the PSP answered. The approval it spent stays
        spent when the PSP refuses: as after a decline, the agent asks the shopper before trying again."""
        d = self.spoken
        if self.store is None or d.delegate is None:
            self.failed = "no payment processor to vault at"
            return wt.acp_error(
                "processing_error", "the store's payment processor cannot be reached", kind="service_unavailable"
            )
        body = {**d.delegation(credential, request), **extra}
        result = await self.store(d.delegate, {"payload": body})
        token = d.vaulted(result)
        if not result.is_error and token is not None:
            self._vaulted[self._index_of(credential)] = token
        return result

    # -- AP2 ---------------------------------------------------------------------------------------------------------

    def _aliases(self) -> list[tuple[str, str, str]]:
        """The shopper's AP2-eligible payment methods: (alias, kind, token), card first, then Google Pay."""
        out: list[tuple[str, str, str]] = []
        for kind in ("card", "google_pay"):
            for i in self.wallet.persona.instruments:
                if i.kind == kind:
                    label = f"{(i.brand or 'card').title()} card" if kind == "card" else "Google Pay"
                    out.append((label, kind, self.wallet.shown(i.token)))
        return out

    async def _ap2_methods(self, arguments: Mapping[str, Any]) -> ToolResult:
        del arguments
        return wt.ok({"payment_method_aliases": [a for a, _, _ in self._aliases()]})

    async def _ap2_token(self, arguments: Mapping[str, Any]) -> ToolResult:
        alias = arguments.get("payment_method_alias")
        token = next((t for a, _, t in self._aliases() if a == alias), None)
        if token is None:
            return wt.ap2_error("payment_method_not_found", f"no payment method {alias!r}; ask get_payment_methods")
        return wt.ok({"status": "success", "token": token})

    async def _ap2_mandate(self, arguments: Mapping[str, Any]) -> ToolResult:
        found = self._doc(arguments.get("cart_id"))
        if found is None:
            return wt.ap2_error("cart_mandate_missing", "no merchant-signed cart with this cart_id")
        cart_id, doc = found
        alias = arguments.get("payment_method_alias")
        method = next(((a, k, t) for a, k, t in self._aliases() if a == alias), None)
        total = total_of(doc)
        if method is None or total is None:
            return wt.ap2_error("payment_method_not_found", f"no payment method {alias!r}; ask get_payment_methods")
        cart = doc.get("cart_mandate")
        contents_of = cart.get("contents") if isinstance(cart, Mapping) else None
        details_id = contents_of.get("id") if isinstance(contents_of, Mapping) else None
        contents = {
            "payment_mandate_id": f"pm_{digest([cart_id, len(self._mandates)], 12)}",
            "payment_details_id": details_id if isinstance(details_id, str) else cart_id,
            "payment_details_total": {
                "label": "Total",
                "amount": {"currency": total.currency, "value": float(to_major(total.amount, total.currency) or 0)},
            },
            "payment_response": {
                "request_id": details_id if isinstance(details_id, str) else cart_id,
                "method_name": "CARD",
                "details": {"token": {"value": method[2], "url": "https://credentials-provider.example/"}},
                "payer_email": arguments.get("user_email"),
            },
            "merchant_agent": "merchant_agent",
            "timestamp": iso(self.clock.now()),
        }
        self._mandates[cart_id] = contents
        return wt.ok({"ap2.mandates.PaymentMandate": {"payment_mandate_contents": contents}})

    async def _ap2_sign(self, arguments: Mapping[str, Any]) -> ToolResult:
        found = self._doc(arguments.get("cart_id"))
        if found is None:
            return wt.ap2_error("cart_mandate_missing", "no merchant-signed cart with this cart_id")
        cart_id, doc = found
        contents = self._mandates.get(cart_id)
        total = total_of(doc)
        if contents is None or total is None:
            return wt.ap2_error("payment_mandate_missing", "create the payment mandate first (create_payment_mandate)")
        given = await self._approve(cart_id, doc, total, self._sheet("Sign on your device", total))
        if not given:
            return wt.ap2_error("user_declined", self._why("the shopper did not sign the mandates"))
        issued = await self._issue(cart_id, doc, "card", "ap2.card", total)
        if isinstance(issued, ToolResult):
            return wt.ap2_error(str(issued.error_code), _said(issued))
        credential, _ = issued
        contents["payment_response"]["details"]["token"]["value"] = credential.token
        cart_hash = digest(doc.get("cart_mandate") or {"cart_id": cart_id}, 64)
        mandate_hash = digest(contents, 64)
        signed = {"payment_mandate_contents": contents, "user_authorization": f"{cart_hash}_{mandate_hash}"}
        return wt.ok({"ap2.mandates.PaymentMandate": signed})

    async def _ap2_checkout_mandate(self, arguments: Mapping[str, Any]) -> ToolResult:
        found = self._doc(arguments.get("checkout_id"))
        if found is None:
            return wt.ap2_error("mandate_scope_mismatch", "no merchant-signed checkout with this checkout_id")
        checkout_id, doc = found
        total = total_of(doc)
        jwt = doc.get("checkout_jwt")
        if total is None or not isinstance(jwt, str):
            return wt.ap2_error("merchant_authorization_missing", "the checkout is not signed by the merchant yet")
        given = await self._approve(checkout_id, doc, total, self._sheet("Sign on your device", total))
        if not given:
            return wt.ap2_error("user_declined", self._why("the shopper did not sign the checkout mandate"))
        claims = {
            "vct": "mandate.checkout.1",
            "checkout_jwt": jwt,
            "checkout_hash": doc.get("checkout_hash"),
            "iat": int(self.clock.now().timestamp()),
        }
        return wt.ok({"ap2.mandates.CheckoutMandate": claims, "status": "signed"})

    async def _ap2_issue(self, arguments: Mapping[str, Any]) -> ToolResult:
        found = self._doc(arguments.get("checkout_id"))
        if found is None:
            return wt.ap2_error("mandate_scope_mismatch", "no merchant-signed checkout with this checkout_id")
        checkout_id, doc = found
        expected = doc.get("checkout_hash")
        if isinstance(expected, str) and arguments.get("checkout_jwt_hash") != expected:
            return wt.ap2_error("mandate_scope_mismatch", "checkout_jwt_hash is not the checkout's checkout_hash")
        total = total_of(doc)
        if total is None:
            return wt.ap2_error("mandate_required", "the checkout shows no total")
        issued = await self._issue(checkout_id, doc, "card", "ap2.card", total)
        if isinstance(issued, ToolResult):
            code = "mandate_required" if issued.error_code in ("no_consent", "consent_spent") else issued.error_code
            return wt.ap2_error(str(code), _said(issued))
        credential, _ = issued
        return wt.ok({"payment_token": credential.token})

    # -- MPP ---------------------------------------------------------------------------------------------------------

    async def _mpp_credential(self, arguments: Mapping[str, Any]) -> ToolResult:
        challenge_id = arguments.get("challenge_id")
        challenge = self._challenges.get(challenge_id) if isinstance(challenge_id, str) else None
        if challenge is None:
            return wt.mpp_problem(
                "invalid-challenge", "Invalid Challenge", "no challenge with this id; call request_payment"
            )
        found = self._doc(self.book.last_checkout)
        currency = challenge.get("currency")
        minor = _int(challenge.get("amount"))
        if found is None or minor is None or not isinstance(currency, str):
            return wt.mpp_problem("invalid-challenge", "Invalid Challenge", "the challenge names no checkout or amount")
        checkout_id, doc = found
        if not any(i.kind in ("card", "google_pay") for i in self.wallet.persona.instruments):
            return wt.mpp_problem(
                "method-unsupported", "Method Unsupported", "the wallet holds nothing this payment method takes"
            )
        amount = Money(minor, currency.upper())
        recipient = challenge.get("recipient")
        given = await self._approve(checkout_id, doc, amount, self._sheet("Pay", amount, f"to {recipient}"))
        if not given:
            return wt.mpp_problem(
                "payment-error",
                "Payment Error",
                self._why("the shopper declined the charge"),
                challenge_id=challenge_id,
            )
        request = PaymentChallengeRequest(checkout_id, challenge, amount)
        issued = self.wallet.issue(request)
        if isinstance(issued, Refusal):
            return wt.mpp_problem("payment-error", "Payment Error", issued.message, challenge_id=challenge_id)
        echoed = {k: challenge[k] for k in ("id", "method", "intent", "expires") if k in challenge}
        return wt.ok({"challenge": echoed, "payload": {"token": issued.token}})

    # -- issuing -----------------------------------------------------------------------------------------------------

    async def _issue(
        self, checkout_id: str, doc: Mapping[str, Any], handler_id: str, namespace: str, amount: Money
    ) -> tuple[Credential, str | None] | ToolResult:
        """The wallet's credential for an approved amount through one handler, and the token the agent pays with
        when it is not the credential's own (a minted Shop Token)."""
        now_ms = int(self.clock.now().timestamp() * 1000)
        request = self.spoken.credential_request(checkout_id, handler_id, namespace, amount, doc, now_ms=now_ms)
        issued = self.wallet.issue(request)
        if isinstance(issued, Refusal):
            return _refused(issued)
        token: str | None = None
        if isinstance(request, ShopTokenRequest):
            minted = await self.wallet.mint_shop_token(issued, request)
            if isinstance(minted, Refusal):
                if minted.code in HARNESS_REFUSALS:
                    self.failed = f"{minted.code}: {minted.message}"
                return _refused(minted)
            token = minted
            self._vaulted[self._index_of(issued)] = token
        return issued, token

    # -- 3-D Secure --------------------------------------------------------------------------------------------------

    async def _challenge(self, arguments: Mapping[str, Any]) -> ToolResult:
        found = self._doc(arguments.get("checkout_id"))
        if found is None:
            return _error("unknown_checkout", "no such checkout has come back from the store in this conversation")
        checkout_id, doc = found
        url = self.challenge_for(checkout_id, doc)
        if url is None:
            return _error(
                "no_challenge",
                f"no 3-D Secure challenge is waiting on checkout {checkout_id!r}",
                "read the checkout with get_checkout to see its status",
            )
        given = arguments.get("url")
        if isinstance(given, str) and given and given != url:
            return _error("challenge_mismatch", "that is not the challenge the store's checkout carries")
        outcome = await self.wallet.present_challenge(url)
        if isinstance(outcome, Refusal):
            if outcome.code in HARNESS_REFUSALS:
                self.failed = f"{outcome.code}: {outcome.message}"
            return _refused(outcome)
        return wt.ok(_answered(checkout_id, outcome))

    def challenge_for(self, checkout_id: str, doc: Mapping[str, Any]) -> str | None:
        """The bank's challenge page waiting on a checkout: what the checkout carries, else the page an MPP
        payment-action-required answer named for it (a checkout read may show the handoff without its URL)."""
        return self.spoken.challenge_of(doc) or self._actions.get(checkout_id)

    # -- the reference agent's run of the steps ----------------------------------------------------------------------

    async def authorize(self, checkout_id: str, run: HarnessRun) -> Authorized:  # noqa: PLR0911, PLR0912 - one branch per rail
        """Run this rail's wallet steps for a checkout as the reference agent does, each call through `run`, and say
        how they ended; approved carries the payment for the dialect's paying call."""
        found = self._doc(checkout_id)
        if found is None:
            return Authorized("failed", reason="no checkout")
        _, doc = found
        d = self.spoken
        email = self.wallet.persona.email
        steps: list[tuple[str, dict[str, Any]]]
        kit = self.kit.name
        if kit == "acp":
            total = total_of(doc)
            if total is None:
                return Authorized("failed", reason="no total")
            now_ms = int(self.clock.now().timestamp() * 1000)
            asked = d.credential_request(checkout_id, "", "", total, doc, now_ms=now_ms)
            assert isinstance(asked, DelegatedTokenRequest)
            allowance = {
                "reason": "one_time",
                "max_amount": total.amount,
                "currency": total.currency.lower(),
                "checkout_session_id": checkout_id,
                "merchant_id": asked.merchant_id,
                "expires_at": asked.expires_at,
            }
            steps = [(wt.DELEGATE, {"allowance": allowance})]
        elif kit == "ap2_v1":
            methods = await run(wt.AP2_METHODS, {"user_email": email})
            aliases = (methods.structured or {}).get("payment_method_aliases") if not methods.is_error else None
            if not aliases:
                return Authorized("failed", reason="no payment method")
            alias = aliases[0]
            steps = [
                (wt.AP2_TOKEN, {"user_email": email, "payment_method_alias": alias}),
                (wt.AP2_MANDATE, {"payment_method_alias": alias, "user_email": email, "cart_id": checkout_id}),
                (wt.AP2_SIGN, {"cart_id": checkout_id}),
            ]
        elif kit == "ap2":
            steps = [
                (wt.AP2_CHECKOUT_MANDATE, {"checkout_id": checkout_id}),
                (wt.AP2_ISSUE, {"checkout_id": checkout_id, "checkout_jwt_hash": doc.get("checkout_hash")}),
            ]
        elif kit == "mpp":
            challenge = next(reversed(list(self._challenges)), None)
            if challenge is None:
                return Authorized("failed", reason="no challenge")
            steps = [(wt.MPP_CREDENTIAL, {"challenge_id": challenge})]
        else:
            ucp = self._ucp_step(checkout_id, doc)
            if ucp is None:
                return Authorized("failed", reason="nothing the store takes")
            steps = [ucp]
        before = len(self.wallet.issued)
        for name, arguments in steps:
            result = await run(name, arguments)
            if result.is_error:
                declined = result.error_code in _DECLINED
                return Authorized("declined" if declined else "failed", reason=_said(result))
        if len(self.wallet.issued) == before:
            return Authorized("failed", reason="no credential")
        index = len(self.wallet.issued) - 1
        credential = self.wallet.issued[index]
        return Authorized("approved", d.credential_answer(credential, token=self._vaulted.get(index)))

    def _ucp_step(self, checkout_id: str, doc: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
        """The UCP wallet call for the handler the wallet prefers among those the checkout declares and the shopper
        can pay through."""
        every = self.wallet.offers(self.spoken.handlers_of(doc))
        offers = [o for o in every if o.payable]
        if not offers:
            # nothing payable: ask the card handler's tokenizer anyway, so the refusal is UCP's own error answer
            card = next((o for o in every if o.namespace not in (GOOGLE_PAY_NAMESPACE, SHOP_PAY_NAMESPACE)), None)
            if card is None:
                return None
            binding = {"type": wt.CHECKOUT_BINDING, "id": checkout_id}
            return wt.TOKENIZE, {"handler_id": card.handler_id, "binding": binding}
        offer = offers[0]
        if offer.namespace == GOOGLE_PAY_NAMESPACE:
            total = total_of(doc)
            if total is None:
                return None
            config = _config(self.spoken.handlers_of(doc).get(GOOGLE_PAY_NAMESPACE))
            raw = config.get("merchant_info")
            merchant: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
            request = {
                "apiVersion": 2,
                "apiVersionMinor": 0,
                "merchantInfo": {
                    k: v
                    for k, v in (
                        ("merchantId", merchant.get("merchant_id")),
                        ("merchantName", merchant.get("merchant_name")),
                    )
                    if v
                },
                "allowedPaymentMethods": [
                    {
                        "type": "CARD",
                        "parameters": dict(m.get("parameters") or {}),
                        "tokenizationSpecification": dict(m.get("tokenization_specification") or {}),
                    }
                    for m in config.get("allowed_payment_methods") or ()
                    if isinstance(m, Mapping)
                ]
                or [{"type": "CARD", "parameters": {}}],
                "transactionInfo": {
                    "totalPriceStatus": "FINAL",
                    "totalPrice": to_major(total.amount, total.currency),
                    "currencyCode": total.currency,
                    "transactionId": checkout_id,
                    "checkoutOption": "COMPLETE_IMMEDIATE_PURCHASE",
                },
            }
            return wt.GOOGLE_PAY, request
        if offer.namespace == SHOP_PAY_NAMESPACE:
            config = _config(self.spoken.handlers_of(doc).get(SHOP_PAY_NAMESPACE))
            out: dict[str, Any] = {"handler_id": offer.handler_id, "checkout_id": checkout_id}
            if isinstance(config.get("shop_id"), str):
                out["shop_id"] = config["shop_id"]
            return wt.SHOP_PAY, out
        binding = {"type": wt.CHECKOUT_BINDING, "id": checkout_id}
        return wt.TOKENIZE, {"handler_id": offer.handler_id, "binding": binding}


_DECLINED: Final = frozenset({"buyer_declined", "canceled", "user_declined", "payment-error"})
"""The error codes a wallet step answers when the shopper said no."""


def _ucp() -> Dialect:
    """The UCP dialect, the default when an agent has not said what it speaks (imported late: the dialects read the
    documents through this module)."""
    from carttrace.agents.dialects.ucp import UcpDialect

    return UcpDialect()


def _config(entries: Any) -> Mapping[str, Any]:
    entry = entries[0] if isinstance(entries, list) and entries else entries
    config = entry.get("config") if isinstance(entry, Mapping) else None
    return config if isinstance(config, Mapping) else {}


def _namespace(handlers: Mapping[str, Any], handler_id: str) -> str | None:
    """The namespace of the handler a checkout declares under `handler_id` (an entry id, or the namespace itself)."""
    for namespace, entries in handlers.items():
        if namespace == handler_id:
            return namespace
        for entry in entries if isinstance(entries, list) else [entries]:
            if isinstance(entry, Mapping) and entry.get("id") == handler_id:
                return namespace
    return None


def _money(amount: Any, currency: Any) -> Money | None:
    if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0 or not isinstance(currency, str):
        return None
    code = currency.strip().upper()
    return Money(amount, code) if len(code) == 3 and code.isalpha() else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _shown(value: Money) -> str:
    return money(value.amount, value.currency)


def _said(result: ToolResult) -> str:
    data = result.structured if isinstance(result.structured, Mapping) else {}
    for key in ("message", "error_description", "statusMessage", "detail"):
        if isinstance(data.get(key), str):
            return str(data[key])
    messages = data.get("messages")
    if isinstance(messages, list) and messages and isinstance(messages[0], Mapping):
        return str(messages[0].get("content") or "")
    return result.content if isinstance(result.content, str) else ""


def _as_ucp(result: ToolResult) -> ToolResult:
    return wt.ucp_error(str(result.error_code or "invalid"), _said(result))


def _answered(checkout_id: str, outcome: ChallengeOutcome) -> dict[str, Any]:
    return {
        "checkout_id": checkout_id,
        "shopper_completed_challenge": outcome.approved,
        "bank": dict(outcome.answer),
        "next": "read the checkout again to see whether the payment went through",
    }


def _error(code: str, message: str, hint: str = "") -> ToolResult:
    data: dict[str, Any] = {"error": code, "message": message}
    if hint:
        data["hint"] = hint
    return ToolResult(content=dumps(data), is_error=True, error_code=code, structured=data)


def _refused(refusal: Refusal) -> ToolResult:
    return _error(refusal.code, refusal.message, refusal.hint)

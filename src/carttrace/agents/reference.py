"""The reference agent: deterministic, no model, knows the hidden spec, and acts only on what the surfaces return.

Why it exists. When the Qwen agent fails a task, the failure is either the agent's (it never asked the size, it
misread the chart) or the store's (the chart is an image, the ingredients are not published). The reference agent
separates the two: it knows exactly what the shopper wants, so whatever stops it is something the store did not
expose. It is the "perfect agent" of the benchmark, limited only by the store's data.

What it does, in order, every step written to the TurnSink:
1. asks the shopper about each on_ask field, as a careful agent would (it already knows the answers; a reply that
   does not reveal the field is noted, because that is a shopper bug);
2. searches the catalog with the spec's own words: the product's name for a replenish task, else the shopper's noun;
3. reads each candidate with get_product (or lookup_catalog, or the search result itself), keeping the variants the
   search listed, because UCP's get_product may show only the selected variant; when the product offers option
   values no known variant carries, it narrows get_product with `selected` to the values the spec needs (the size
   the product's own chart gives, the colour asked for). It checks every predicate against the returned data with
   carttrace.agents.predicates, the same checks task generation used;
4. when the decisive detail is missing from a product, asks the store's policy tool once (a size guide often lives
   there), and otherwise records the detail as missing for that product;
5. adds the variant it can justify to a cart (the cheapest, when the shopper asked for the cheapest), reads the cart
   back, and ends with cart_ready; or ends with no_match, saying what it could not verify.

On a checkout row (D13: given a wallet, sandbox only) it goes on from the verified cart, through the same harness tools
the Qwen agent has (carttrace.agents.checkout):
6. creates the checkout from the cart. When the task's spec has a buyer block (generator 2), it opens the checkout
   with no buyer details, asks the shopper for exactly the fields the store's checkout reports missing (delivery
   first, then the address only for a delivery, then email and phone), enters what the shopper disclosed through
   the store's own operations (carttrace.agents.entry: update_checkout on UCP), and picks the option that matches the
   delivery preference and meets any deadline. A detail the shopper withholds is never entered. Older specs keep the
   persona's saved email, phone and address, and the cheapest shipping option;
7. checks the checkout's total against the shopper's budget (items alone, or all in when they said so) and, when it
   is over, tries the next variant it can justify instead of asking; otherwise asks the shopper to approve the order,
   stating the checkout's own total, and stops when they say no;
8. takes the handler the wallet recommends, a credential for the approved total, and completes the checkout;
9. hands a 3-D Secure challenge (an Action, or a continue_url handoff) to the wallet and reads the checkout again;
10. on a decline or a failed challenge tells the shopper and stops: it never tries another card unasked; a decline
    the issuer says a retry may clear (a processing error, D66) it tries once more, after asking the shopper to
    approve the total again; otherwise reads the order back and ends with order_placed.

On the payment rails (D66) the same steps take the rail's calls: AP2 states the shopper's intent (an open checkout
mandate naming the items) before the merchant's checkout is made, and pays with complete_checkout, the wallet signing
the mandates; MPP asks for the store's HTTP 402 challenge (request_payment) just before it pays it (pay), and when the
payment does not come back paid reads the UCP checkout to see where it stands, a 3-D Secure challenge included.

A compare task names its candidates (a `one_of` predicate): it searches each by name, and among the variants it can
justify keeps the best by the task's criterion (the cheapest, the cheapest per unit, the most or least of a nutrient),
reading the measure from the store's own data (carttrace.agents.measures).

Generator 3's intents (D20): a bundle (an `all_of` predicate) searches each named item and carts one justified variant
of every item in one cart, or refuses naming the item it could not justify; a substitute task (a `prefer` predicate)
tries the named product first and, when the catalog lists it with nothing for sale or the store will not add it (the
replica's sealed stockout, true from the first read in a collection, or one that fires when the line is added),
carts the closest substitute it can justify against every other predicate when the shopper allows one, else refuses
saying the product is out of stock; a policy question is answered in the final message from the policies the store's
catalog replies attached or its policy tool returned, and recorded as a missing detail when the store publishes none.

Its result carries `missing_details`: detail name to the products whose data lacked it. That is the store's side of
the benchmark.

Which protocol it speaks is read from the tools the surfaces list (`carttrace.agents.dialects`): UCP's catalog and cart, ACP's
checkout sessions with the catalog beside them, or a WebMCP storefront's page tools. The dialect names the tools and
shapes their arguments and documents; the steps above are the same on each.

On a store's website (D31, `boundary` "selection") step 5 chooses the variant (choose_variant) instead of carting it
and ends with selection_ready, naming the selection; the runner reads it back. When the store's search finds nothing
(it shows none, robots.txt disallows it, or its results page lists no product link) it searches the store's own
product index (browse_index, the store's sitemaps) with the same words, keeping the hits that match the most of them;
when that finds nothing it browses: the homepage's product links, then the product links of the categories whose
names share a word with what it looks for. A policy question is answered from the store's policy
page, found among the homepage's policy links.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, NoReturn

import httpx

from carttrace.agents.checkout import HARNESS_SURFACE, CheckoutHarness, total_of
from carttrace.agents.details import detail as detail_named
from carttrace.agents.details.size_chart import measurement_of, read_chart, size_key
from carttrace.agents.dialects import Dialect, UcpDialect, dialect_for
from carttrace.agents.entry import CheckoutEntry, Details, entry_for, pick
from carttrace.agents.nouns import plural_of, singular_of
from carttrace.agents.phrases import Phrases, load_phrases
from carttrace.agents.predicates import CART_OPS, DETAIL_OPS, best, bundle_items, holds, item_holds, option_names
from carttrace.agents.products import Product, Variant, products_in_reply, read_product
from carttrace.agents.service import AgentResult, DeclaredStatus, Limits, Shopper, ShopperReply, Supervisor, TurnSink
from carttrace.agents.shopper import opening_origin
from carttrace.agents.slots import money
from carttrace.agents.spec import SUPERLATIVES, Buyer, HiddenSpec, Predicate
from carttrace.agents.tasks import line_quantity
from carttrace.agents.text import fold, lines_of, pipe_tables, read_html
from carttrace.agents.tools import FATAL_CODES, failure_reason, payload_of, result_text, transport_arguments, with_transport
from carttrace.agents.wallet import Wallet
from carttrace.agents.wallet_tools import CHALLENGE
from carttrace.config.schema import Settings
from carttrace.core.canon import dumps
from carttrace.core.clock import Clock, SystemClock
from carttrace.core.errors import MCError
from carttrace.core.money import Money
from carttrace.protocols.tools import ToolResult, ToolSpec, ToolSurface
from carttrace.trace.schema import Cart, EndReason, TaskRef

QUESTIONS: Mapping[str, str] = {
    "product_name": "Which product was it?",
    "product_type": "What kind of product are you looking for?",
    "budget": "What's your budget?",
    "quantity": "How many would you like?",
    "measurements": "What are your measurements?",
    "allergens": "Do you have any allergies I should know about?",
    "conditions": "Do you have any health conditions I should know about?",
    "brand": "Which brand would you like?",
}
"""What the reference agent asks about an on_ask field; any other field gets "Which <label> would you like?"."""
BUYER_QUESTIONS: Mapping[str, str] = {
    "fulfillment": "How would you like to receive it: standard delivery, express, or would you rather collect it?",
    "shipping_address": "What address should I ship it to?",
    "email": "What email should the store use for your order?",
    "phone": "What phone number should the store use for your order?",
}
"""What the reference agent asks when the store's checkout reports a buyer field missing."""
ENTRY_ROUNDS = 4
"""Rounds of reading what the checkout still needs and entering it: delivery, address, email and phone at most."""
CHECKOUT_TRIES = 3
"""Variants a checkout row tries when the one before came to more than the shopper's budget."""
READ_ROUNDS = 3
"""Batches of candidates read in detail before the agent gives up on a search's results (D104)."""
SEARCH_PAGES = 3
"""Pages of one search's results the agent reads (D104)."""
QUERY_CHARS = 200
"""The longest query the agent types into a store's search, cut at a word: what every storefront search box takes
(the WebMCP storefront's takes 256). Typed whole, a 250-character Lazada title was refused "query must satisfy
maxLength 200" on 14 reference rows of 2026-10-17 (D119); its first 200 characters still sit inside the title."""
POLICY_QUERIES: Mapping[str, str] = {"size_chart": "size chart", "ingredients": "ingredients", "warnings": "warnings"}
"""What it asks a store's policy tool when a product lacks the decisive detail."""
POLICY_TOPIC_QUERIES: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "returns": ("return policy", ("return", "refund")),
    "shipping": ("shipping policy", ("ship", "deliver")),
}
"""A policy task's topic: what it asks the store's policy tool, and the words a policy text about it carries."""
POLICY_TOOLS = ("search_shop_policies_and_faqs",)
BROWSE_CATEGORIES = 4
"""Categories a website search falls back to reading, the closest to the shopper's words first."""
BROWSE_ERRORS = frozenset({"no_search", "robots_disallowed", "not_found"})
"""Search errors after which the reference agent browses the store instead."""
_FAILURES = (MCError, httpx.HTTPError, OSError, TimeoutError)
RETRY_WORDS = ("processing_error", "processing error")
"""What a store says of a decline a retry with the same credential may clear (carttrace.sandbox.core.psp.DECLINE_CLASSES)."""
DECLINE_WORDS = ("declin", "authenticat", "do_not_honor", "insufficient_funds")
"""What a store says of a payment its issuer turned down, which another instrument may pay (D115)."""
PAY_TRIES = 2
"""Payments a checkout row attempts: the first, and one retry after a retryable decline, with renewed approval."""
SAME_CALLS = 2
"""Times the agent sends one call with the same arguments while the store's state is unchanged (no write between
them). A third is a loop of the agent's own and ends the row as harness_error (2026-10-17, run-95e4a1f77934)."""
CART_RESERVE = 4
"""Steps a candidate read leaves for the cart: the AP2 intent, the cart's create and update, and its read back."""
CHECKOUT_RESERVE = 10
"""Steps a checkout row's read leaves besides, for the checkout (the agent needs nine, carttrace.collect.conditions)."""
READ_PREFIXES = ("get_", "search", "lookup", "list_", "browse")
"""Calls that read the store's state and change nothing: repeats of one are counted between writes."""

__all__ = ["ReferenceAgent", "UcpDialect"]


class ReferenceAgent:
    """A ProtocolAgent with no model. The supervisor, if given, is never needed: the agent cannot loop."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        clock: Clock | None = None,
        ask: bool = True,
        max_candidates: int = 8,
        dialect: Dialect | None = None,
        phrases: Phrases | None = None,
        wallet: Wallet | None = None,
        recover: bool | None = None,
    ) -> None:
        """`settings` give only our agent profile URL (live.agent_profile_url) for UCP's `meta`. A `wallet` makes the
        episode a checkout row: the journey goes on from the verified cart to an order. `dialect` fixes how the agent
        speaks; by default it is read from the tools the surfaces list (carttrace.agents.dialects). `recover` is the row's
        designed expectation for a payment that fails (experiment.json `expected.recoverable`, D107): True, the
        reference retries a retryable decline on a fresh payment request and pays a declined instrument's order with
        another one the wallet holds; False or None, it stops after a decline as before (a retryable one aside)."""
        self.recover = recover
        self.transport = transport_arguments(settings.live.agent_profile_url if settings is not None else None)
        self.wallet = wallet
        self.clock: Clock = clock or SystemClock()
        self.ask = ask
        self.max_candidates = max_candidates
        self.dialect = dialect
        self.phrases = phrases or load_phrases()

    async def run(
        self,
        task: TaskRef,
        surfaces: Sequence[ToolSurface],
        shopper: Shopper,
        sink: TurnSink,
        *,
        limits: Limits,
        supervisor: Supervisor | None = None,
    ) -> AgentResult:
        run = _Run(self, HiddenSpec.from_task(task), shopper, sink, limits)
        run.opening = opening_origin(task)
        try:
            return await run.go(surfaces)
        except _Stop as stop:
            return stop.result


class _Stop(Exception):
    def __init__(self, result: AgentResult) -> None:
        super().__init__(result.end_reason)
        self.result = result


class _OverBudget(Exception):
    """The checkout's total is more than the shopper's budget allows: the next variant may fit."""

    def __init__(self, shown: str) -> None:
        super().__init__(shown)
        self.shown = shown


@dataclass
class _Checked:
    """What one product's data showed."""

    product: Product
    justified: list[Variant] = field(default_factory=list)
    unknown: dict[str, str] = field(default_factory=dict)
    """Predicate field to what was missing for it (the detail's name, or the predicate's field)."""


class _Run:
    def __init__(
        self, agent: ReferenceAgent, spec: HiddenSpec, shopper: Shopper, sink: TurnSink, limits: Limits
    ) -> None:
        self.agent = agent
        self.dialect: Dialect = agent.dialect or UcpDialect()
        """What the agent speaks: the agent's own, or the dialect the listed tools offer once they are listed."""
        self.spec = spec
        self.shopper = shopper
        self.sink = sink
        self.limits = limits
        self.clock = agent.clock
        self.started = self.clock.monotonic()
        self.tools: dict[str, ToolSurface] = {}
        self.specs: dict[str, ToolSpec] = {}
        self.steps = 0
        self.tool_calls = 0
        self.cart_id: str | None = None
        self.missing: dict[str, list[str]] = {}
        self.quantity = line_quantity(spec.predicates)
        self.checked = [p for p in spec.predicates if p.op not in CART_OPS]
        self.decisive = next((p for p in spec.predicates if p.op in DETAIL_OPS), None)
        self.bundle = next((p for p in spec.predicates if p.op == "all_of"), None)
        self.prefer = next((p for p in spec.predicates if p.op == "prefer"), None)
        self.policy = next((p for p in spec.predicates if p.op == "policy"), None)
        self.policies: list[str] = []
        """Policy texts the store sent: the `policies` of its catalog replies, and its policy tool's answer."""
        self.out_of_stock: str | None = None
        """The named product the store would not add to a cart (a substitute task's stockout)."""
        self.lines: list[tuple[Variant, Product]] = []
        """What the cart holds, one entry per line, once a cart is made."""
        self.harness = (
            CheckoutHarness(agent.wallet, shopper, sink, clock=agent.clock) if agent.wallet is not None else None
        )
        self.disclosed: set[str] = set()
        """Buyer details the shopper has given in this episode."""
        self.search_failed = False
        """The store's search answered with an error browsing can stand in for (a website without search)."""
        self.opening: dict[str, Any] | None = None
        """The opening turn's origin (D69): who wrote the task's first message."""
        self.sent: dict[str, int] = {}
        self.read_cut = False
        """The steps left ended the candidate reads before every candidate the search found was read."""
        """Each call and its arguments to how often it was sent since the last write (`SAME_CALLS`)."""

    async def go(self, surfaces: Sequence[ToolSurface]) -> AgentResult:
        self.sink.turn("shopper", "message", text=self.shopper.opening(), origin=self.opening)
        self._note("reference agent: it knows the hidden spec and uses only what the store's surfaces return")
        await self._list(surfaces)
        if self.agent.dialect is None:
            self.dialect = dialect_for(self.specs)
        self._note(f"the surfaces speak {self.dialect.name}: {', '.join(sorted(self.tools))}")
        if self.harness is not None:
            self.harness.speak(self.dialect, store=self._store)
        if self.dialect.search not in self.tools:
            self._miss(f"tool:{self.dialect.search}", [])
            self._note(f"no surface offers {self.dialect.search}, so nothing can be found")
            return self._end("no_protocol")
        if self.agent.ask:
            await self._ask()
        found = await self._search()
        checked = await self._read_candidates(found)
        checked = await self._consult_policies(checked)
        if self.policy is not None:
            await self._read_policy()
        self._record_missing(checked)
        if self.bundle is not None:
            return await self._bundle_cart(checked)
        self._sold_out(checked)
        choices = self._choices(checked)
        if not choices and self.read_cut:
            # the steps ran out before the candidates did: no refusal is justified by what was read
            self._note(f"step cap reached after {self.steps} steps, before every candidate was read")
            return self._end("step_cap")
        return await self._cart_from(choices)

    async def _read_candidates(self, found: Sequence[Mapping[str, Any]]) -> list[_Checked]:
        """The candidates read in detail: the first `_candidate_cap`, then, while none of them fits, the next as many,
        up to READ_ROUNDS batches. Why (D104): a search for "backpack" on an open stand-in listed 17, and the one in
        the coffee colour the shopper asked for was the twelfth; a careful shopper reads on before giving up. A
        superlative ("whichever is cheapest") is the best of every candidate, so then they are all read."""
        cap = self._candidate_cap()
        checked = await self._check_all(found[:cap])
        read = cap
        named = self.spec.predicate("product_name")
        best_of = any(p.op in SUPERLATIVES for p in self.spec.predicates) and (
            named is None or not isinstance(named.value, list)
        )
        # a substitute task reads on only while nothing it may cart was found (2026-10-17: the named product's sold-out
        # listing is never justified, so "nothing justified" read 24 candidates after the second was a fit, and the
        # 30-step cap ended the row between the cart and its read back), and not at all once the named product is
        # sold out and the shopper wants nothing else
        while (
            self.bundle is None
            and (best_of or not self._choices(checked))
            and not self._settled(checked)
            and read < min(len(found), cap * READ_ROUNDS)
            and self._steps_left()
        ):
            more = found[read : read + cap]
            read += len(more)
            checked += await self._check_all(more)
        return checked

    async def _check_all(self, found: Sequence[Mapping[str, Any]]) -> list[_Checked]:
        """Each candidate read in turn while the steps left cover the cart (and a checkout row's checkout) after it."""
        out: list[_Checked] = []
        for raw in found:
            if not self._steps_left():
                self.read_cut = True
                self._note(f"read {len(out)} more candidates; the steps left are kept for the cart")
                break
            out.append(await self._check(raw))
        return out

    def _steps_left(self) -> bool:
        """Whether one more candidate read (a get_product and its narrowing) still leaves the cart its steps."""
        reserve = CART_RESERVE + (CHECKOUT_RESERVE if self.harness is not None else 0) + 2
        return self.limits.step_cap - self.steps >= reserve

    def _settled(self, checked: Sequence[_Checked]) -> bool:
        """A substitute task whose named product the store lists with nothing for sale, when the shopper wants no
        other: reading more candidates cannot change the refusal."""
        if self.prefer is None or self.spec.substitute_ok:
            return False
        return any(_named_by(self.prefer, c.product) and not _for_sale(c.product) for c in checked)

    def _sold_out(self, checked: Sequence[_Checked]) -> None:
        """A substitute task's named product the catalog lists with no variant for sale (a world that opens with its
        stockout): it is out of stock without trying a cart, and the substitutes come next."""
        if self.prefer is None:
            return
        for entry in checked:
            product = entry.product
            if _named_by(self.prefer, product) and not _for_sale(product):
                self.out_of_stock = product.title
                self._note(f"the store lists {product.title} with no variant for sale: out of stock")
                return

    async def _cart_from(self, choices: Sequence[tuple[Variant, Product]]) -> AgentResult:
        """The first choice the store carts (and, with a wallet, checks out within the budget), else a refusal."""
        limit = CHECKOUT_TRIES if self.harness is not None else 2
        if self.prefer is not None:
            limit = max(limit, len(choices))  # the named product first, then every substitute the shopper allows
        over: _OverBudget | None = None
        for variant, product in choices[:limit]:
            cart = await self._add(variant, product)
            if cart is None:
                if self.prefer is not None and _named_by(self.prefer, product):
                    self.out_of_stock = product.title
                    self._note(f"the store would not add {product.title} ({variant.id}) to a cart: out of stock")
                continue
            if self.harness is not None:
                try:
                    return await self._checkout(self.harness, cart, variant, product)
                except _OverBudget as exc:
                    over = exc
                    self._note(f"the checkout for {variant.id} comes to {exc.shown}, over the shopper's budget")
                    continue
            return self._finish_cart(cart, variant, product)
        if over is not None:
            self._halt(
                "Everything I found that fits comes to more than your budget at checkout, so I haven't placed an order."
            )
        return self._finish_refusal()

    # -- tools -------------------------------------------------------------------------------------------------------

    async def _list(self, surfaces: Sequence[ToolSurface]) -> None:
        for surface in surfaces:
            try:
                specs = await surface.list_tools()
            except _FAILURES as exc:
                self._note(f"listing {surface.kind} tools failed: {exc}")
                raise _Stop(self._result("harness_error")) from None
            for spec in specs:
                if spec.name not in self.tools:
                    self.tools[spec.name] = surface
                    self.specs[spec.name] = spec

    async def _call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self._repeat_guard(name, arguments)
        self._step()
        surface = self.tools[name]
        self.sink.turn("agent", "tool_call", tool=name, surface=str(surface.kind), arguments=arguments)
        self.tool_calls += 1
        guarded = self.harness.guard(name, arguments) if self.harness is not None else None
        try:
            if guarded is not None:
                result = guarded
            else:
                result = await surface.call(name, with_transport(self.specs[name], arguments, self.agent.transport))
        except _FAILURES as exc:
            self._note(f"{name} failed: {exc}")
            raise _Stop(self._result(failure_reason(exc))) from None
        self.sink.turn(
            "tool",
            "tool_result",
            tool=name,
            surface=str(surface.kind),
            result=result.content,
            is_error=result.is_error,
            error_code=result.error_code,
            latency_ms=result.latency_ms,
        )
        reason = FATAL_CODES.get(result.error_code or "")
        if reason is not None:
            self._note(f"{name} answered {result.error_code}; the episode cannot go on")
            raise _Stop(self._result(reason))
        if self.harness is not None:
            self.harness.observe(name, result)
        return result

    async def _store(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """A store call the harness makes for the wallet (ACP's delegate_payment): sent and written to the trace as
        the agent's own calls are, with a note saying the wallet asked for it."""
        if name not in self.tools:
            return ToolResult(content=f"no surface offers {name}", is_error=True, error_code="protocol_error")
        self._note(f"the wallet vaults its credential at the store's payment processor ({name})")
        return await self._call(name, arguments)

    async def _harness_call(self, harness: CheckoutHarness, name: str, arguments: dict[str, Any]) -> ToolResult:
        """One of the checkout row's harness tools, written to the trace as the Qwen agent's are: a paying step's call
        once the shopper has answered the wallet's sheet (`CheckoutHarness.call`)."""
        self.sent.clear()  # a wallet step may change what the store's checkout answers
        self._step()

        def write_call() -> None:
            self.sink.turn("agent", "tool_call", tool=name, surface=HARNESS_SURFACE, arguments=arguments)

        result = await harness.call(name, arguments, write_call=write_call)
        self.sink.turn(
            "tool",
            "tool_result",
            tool=name,
            surface=HARNESS_SURFACE,
            result=result.content,
            is_error=result.is_error,
            error_code=result.error_code,
        )
        if harness.failed is not None:
            self._note(f"the wallet failed, not the agent: {harness.failed}")
            raise _Stop(self._result("harness_error"))
        return result

    def _repeat_guard(self, name: str, arguments: Mapping[str, Any]) -> None:
        """No call is sent a third time with the same arguments while nothing was written in between (`SAME_CALLS`):
        a write (a cart, a checkout update, a payment) starts the count again, since a read after it may answer
        otherwise. The guard is the agent's own: a deterministic agent that would repeat itself is looping, which is
        the harness's failure, never the store's or the shopper's."""
        if not name.startswith(READ_PREFIXES):
            self.sent.clear()
            return
        key = dumps([name, arguments])
        self.sent[key] = self.sent.get(key, 0) + 1
        if self.sent[key] > SAME_CALLS:
            self._note(f"the reference agent would send {name} a third time with the same arguments; stopped as a loop")
            raise _Stop(self._result("harness_error"))

    def _step(self) -> None:
        if self.steps >= self.limits.step_cap:
            self._note(f"step cap reached after {self.steps} steps")
            raise _Stop(self._result("step_cap"))
        if self.clock.monotonic() - self.started >= self.limits.time_cap_s:
            self._note("time cap reached")
            raise _Stop(self._result("time_cap"))
        self.steps += 1

    # -- the shopper -------------------------------------------------------------------------------------------------

    async def _ask(self) -> None:
        for predicate in self.spec.by_source("on_ask"):
            label = self.agent.phrases.label(predicate.field)
            question = QUESTIONS.get(predicate.field, f"Which {label} would you like?")
            reply = await self._question(question)
            if predicate.field not in reply.fields:
                self._note(f"the shopper did not answer about {predicate.field}; the reference agent knows it anyway")

    async def _question(self, question: str) -> ShopperReply:
        """One question to the shopper, and their reply, both written to the trace."""
        self._step()
        self.sink.turn("agent", "message", text=question)
        reply = await self.shopper.reply(question)
        self.sink.turn(
            "shopper",
            "message",
            text=reply.text,
            arguments=reply.annotation(),
            origin=reply.origin(),
            **reply.turn_extras(),
        )
        self.disclosed.update(reply.disclosed)
        return reply

    # -- the catalog -------------------------------------------------------------------------------------------------

    def _queries(self) -> list[str]:
        named = self.spec.predicate("product_name")
        kind = self.spec.predicate("product_type")
        brand = self.spec.predicate("brand")
        queries = []
        if self.bundle is not None:
            queries += [
                str(item["title"]) for item in bundle_items(self.bundle.value) if isinstance(item.get("title"), str)
            ]
        if named is not None and isinstance(named.value, str):
            queries.append(named.value)
        if named is not None and isinstance(named.value, list):
            queries += [title for title in named.value if isinstance(title, str)]
        if kind is not None and isinstance(kind.value, str):
            noun = kind.value.removeprefix("pair of ")
            if brand is not None and isinstance(brand.value, str):
                queries.append(f"{brand.value} {noun}")
            queries.append(noun)
        return list(dict.fromkeys(q for q in queries if q.strip()))

    async def _search(self) -> list[Mapping[str, Any]]:
        """The candidates: every product the shopper's queries return, up to max_candidates (a bundle searches each of
        its items, whatever the count). When the shopper's noun finds nothing, its plural is tried: a store's search
        matches whole words, and a store may write only the plural ("Chomplings")."""
        found: dict[str, Mapping[str, Any]] = {}
        for query in self._queries():
            await self._search_one(query, found)
            if self.bundle is None and len(found) >= self.agent.max_candidates:
                break
        if not found and not self.search_failed:
            for query in self._plural_queries():
                await self._search_one(query, found)
        if not self.search_failed:
            await self._narrower_search(found)
        if not found and self.dialect.index in self.tools:  # a website whose search lists nothing, or has none
            await self._index_search(found)
        if not found and self.dialect.browse in self.tools:
            await self._browse(found)
        if self.bundle is not None:
            found = self._items_first(found)
        found = self._named_first(found)
        if not found:
            self._note("the catalog search found nothing for the shopper's words")
        return list(found.values())

    def _candidate_cap(self) -> int:
        """How many candidates are read in detail: the agent's cap, or every item of a bundle when there are more."""
        if self.bundle is None:
            return self.agent.max_candidates
        return max(self.agent.max_candidates, len(bundle_items(self.bundle.value)))

    async def _search_one(self, query: str, found: dict[str, Mapping[str, Any]]) -> None:
        """One search; for the shopper's noun (the kind), its next pages too, up to SEARCH_PAGES (D104)."""
        if self.search_failed:
            return
        query = clipped_query(query)
        kind = self.spec.predicate("product_type")
        noun = kind.value.removeprefix("pair of ") if kind is not None and isinstance(kind.value, str) else None
        paged = noun is not None and (query == noun or query.endswith(f" {noun}") or query == plural_of(noun))
        arguments: dict[str, Any] | None = self.dialect.search_arguments(query)
        pages = 0
        while arguments is not None:
            result = await self._call(self.dialect.search, arguments)
            if result.is_error and (result.error_code or "") in BROWSE_ERRORS:
                self.search_failed = True
                self._note(f"the store's search answered {result.error_code}; the agent browses the store instead")
            payload = payload_of(result) if not result.is_error else None
            self._collect_policies(payload)
            for raw in products_in_reply(payload) if payload is not None else []:
                found.setdefault(str(raw["id"]), raw)
            pages += 1
            # D104: a store's search pages its results; the one that fits may be on the second page (a coffee-coloured
            # backpack was the 12th of 17 "backpack" results on an open stand-in that served 10 a page)
            more = paged and pages < SEARCH_PAGES and len(found) < self._candidate_cap() * READ_ROUNDS
            arguments = self.dialect.next_search_arguments(arguments, payload) if more and payload else None

    async def _index_search(self, found: dict[str, Mapping[str, Any]]) -> None:
        """Products from the store's own product index (its sitemaps), for each of the shopper's queries: the hits
        that match the most of the query's words. A store that publishes no index ends the attempt."""
        assert self.dialect.index is not None
        for query in [*self._queries(), *self._plural_queries()]:
            result = await self._call(self.dialect.index, self.dialect.index_arguments(query))
            if result.is_error:
                self._note(f"the store's product index answered {result.error_code}; the agent browses instead")
                return
            replies = products_in_reply(payload_of(result))
            best = max((len(r.get("matched") or ()) for r in replies), default=0)
            for raw in replies:
                if len(raw.get("matched") or ()) == best:
                    found.setdefault(str(raw["id"]), raw)
            if self.bundle is None and len(found) >= self.agent.max_candidates:
                return

    async def _browse(self, found: dict[str, Mapping[str, Any]]) -> None:
        """Products found by browsing: the homepage's product links, then those of the categories closest to the
        shopper's words, keeping the links that share a word with them."""
        assert self.dialect.browse is not None
        wanted = {w for q in [*self._queries(), *self._plural_queries()] for w in _words(q)}
        listed = await self._links("/", "product")
        categories = sorted(
            await self._links("/", "category"), key=lambda link: -len(wanted & _words(str(link.get("text") or "")))
        )
        for link in categories[:BROWSE_CATEGORIES]:
            if len(found) >= self.agent.max_candidates:
                break
            listed += await self._links(str(link["url"]), "product")
        for link in listed:
            text = str(link.get("text") or "")
            if wanted & _words(text) and len(found) < self.agent.max_candidates:
                found.setdefault(str(link["url"]), {"id": str(link["url"]), "title": text, "url": link["url"]})
        if not found:
            self._note("browsing the store's links found nothing that names what the shopper wants")

    async def _links(self, url: str, kind: str) -> list[Mapping[str, Any]]:
        assert self.dialect.browse is not None
        result = await self._call(self.dialect.browse, self.dialect.browse_arguments(url, kind))
        payload = payload_of(result) if not result.is_error else None
        links = payload.get("links") if isinstance(payload, Mapping) else None
        return [link for link in links or () if isinstance(link, Mapping) and isinstance(link.get("url"), str)]

    async def _narrower_search(self, found: dict[str, Mapping[str, Any]]) -> None:
        """Shorter queries for what the full words did not find: a long name's first three and first two words, and
        a phrase's last word (the noun) in the singular and plural. A bundle tries them for each item it has not
        found yet. Why (D104, 2026-10-16): stores' searches match whole words, phrases or prefixes, and on the open
        stand-ins of round 8 the reference gave up on 14 of 180 baseline rows the store held, after one or two
        queries ("Raider 150 Fi air filter orinal suzuki", "ignition coil") where a shorter one finds the product."""
        if self.bundle is not None:
            for item in bundle_items(self.bundle.value):
                title, pid = item.get("title"), str(item.get("product_id"))
                if isinstance(title, str) and pid not in found and not _titled(found, title):
                    for query in _narrower(title):
                        await self._search_one(query, found)
                        if pid in found or _titled(found, title):
                            break
            return
        await self._find_named(found)
        if found:
            return
        for query in self._queries():
            for narrow in _narrower(query):
                await self._search_one(narrow, found)
                if found:
                    return

    async def _find_named(self, found: dict[str, Mapping[str, Any]]) -> None:
        """Shorter queries for each candidate a compare task names that the full title did not find (D104): every one
        must be read for "whichever is cheapest" to hold."""
        named = self.spec.predicate("product_name")
        if named is None or not isinstance(named.value, list):
            return
        for title in (t for t in named.value if isinstance(t, str)):
            if not _titled(found, title):
                for query in _narrower(title):
                    await self._search_one(query, found)
                    if _titled(found, title):
                        break

    def _plural_queries(self) -> list[str]:
        kind = self.spec.predicate("product_type")
        if kind is None or not isinstance(kind.value, str):
            return []
        noun = kind.value.removeprefix("pair of ")
        plural = plural_of(noun)
        return [plural] if plural != noun else []

    def _named_first(self, found: dict[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        """The candidates a compare task names ahead of everything else the searches returned, so the candidate cap
        never drops one (D104: the second of two named products was the 14th of 20 results)."""
        named = self.spec.predicate("product_name")
        if named is None or not isinstance(named.value, list):
            return found
        titles = {fold(str(t)) for t in named.value if isinstance(t, str)}
        own = {k: v for k, v in found.items() if fold(str(v.get("title") or "")) in titles}
        return {**own, **{k: v for k, v in found.items() if k not in own}}

    def _items_first(self, found: dict[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        """A bundle's own items ahead of everything else the searches returned, so the candidate cap never drops one."""
        assert self.bundle is not None
        titles = {fold(str(item.get("title") or "")) for item in bundle_items(self.bundle.value)}
        ids = {str(item.get("product_id")) for item in bundle_items(self.bundle.value)}
        own = {k: v for k, v in found.items() if k in ids or fold(str(v.get("title") or "")) in titles}
        return {**own, **{k: v for k, v in found.items() if k not in own}}

    async def _check(self, raw: Mapping[str, Any]) -> _Checked:
        """Read one candidate: its detail from get_product (or lookup_catalog), its variants from the search result
        and the detail together, because get_product may show only the selected variant. When the product offers
        option values none of the known variants carries, narrow get_product to the values the spec needs."""
        product_id = str(raw["id"])
        detailed: Mapping[str, Any] = raw
        if self.dialect.product in self.tools:
            detailed = merged(await self._fetch(self.dialect.product_arguments(product_id)) or raw, raw)
        elif self.dialect.lookup in self.tools:
            result = await self._call(self.dialect.lookup, self.dialect.lookup_arguments([product_id]))
            replies = products_in_reply(payload_of(result)) if not result.is_error else []
            found = next((r for r in replies if str(r.get("id")) == product_id), None)
            detailed = merged(found, raw) if found is not None else raw
        product = self._read(detailed, product_id)
        checked = self._judge(product)
        selected = self._selection(product, detailed)
        narrows = self.dialect.product in self.tools and self.dialect.boundary != "selection"  # a page shows all
        if checked.justified or not selected or not narrows:
            return checked
        narrowed = await self._fetch(self.dialect.product_arguments(product_id, selected))
        if narrowed is None:
            return checked
        return self._judge(self._read(merged(detailed, narrowed), product_id))

    async def _fetch(self, arguments: dict[str, Any]) -> Mapping[str, Any] | None:
        result = await self._call(self.dialect.product, arguments)
        payload = payload_of(result) if not result.is_error else None
        self._collect_policies(payload)
        replies = products_in_reply(payload) if payload is not None else []
        return replies[0] if replies else None

    def _collect_policies(self, payload: Any) -> None:
        """The policies a catalog reply attached (UCP `policies` at the response root), kept as text for a policy
        question: "<type>: <description>"."""
        for entry in policies_in(payload):
            text = policy_text(entry)
            if text and text not in self.policies:
                self.policies.append(text)

    async def _read_policy(self) -> None:
        """The store's answer to the shopper's policy question: what its catalog replies attached, else what its
        policy tool says to the topic's query; a store that publishes neither has the detail missing."""
        assert self.policy is not None
        topic = _topic(self.policy)
        query, words = POLICY_TOPIC_QUERIES.get(topic, (topic, (topic,)))
        if not _about(self.policies, words):
            tool = next((t for t in POLICY_TOOLS if t in self.tools), None)
            if tool is not None:
                result = await self._call(tool, {"query": query})
                if not result.is_error:
                    lines, _ = _policy_text(payload_of(result))
                    self.policies.extend(line for line in lines if line not in self.policies)
        if not _about(self.policies, words) and self.dialect.browse in self.tools and self.dialect.page in self.tools:
            await self._policy_page(topic, words)
        if not _about(self.policies, words):
            self._miss("policies", [])

    async def _policy_page(self, topic: str, words: Sequence[str]) -> None:
        """A website's policy page on the topic, from the homepage's policy links, read as text."""
        assert self.dialect.page is not None
        links = await self._links("/", "policy")
        page = next(
            (link for link in links if any(w in f"{link.get('text')} {link.get('url')}".lower() for w in words)), None
        )
        if page is None:
            return
        result = await self._call(self.dialect.page, {"url": page["url"]})
        payload = payload_of(result) if not result.is_error else None
        text = payload.get("text") if isinstance(payload, Mapping) else None
        if isinstance(text, str) and text.strip():
            self.policies.append(f"{topic}: {' '.join(text.split())[:2000]}")

    def _policy_answer(self) -> str:
        """What the final message says to the shopper's policy question, from the store's own text or its absence."""
        if self.policy is None:
            return ""
        topic = _topic(self.policy)
        _, words = POLICY_TOPIC_QUERIES.get(topic, (topic, (topic,)))
        found = _about(self.policies, words)
        if found:
            return f" On {topic}, the store says: {found[0][:400]}"
        return f" The store does not publish its {topic} policy to agents, so I can't answer that part."

    def _read(self, raw: Mapping[str, Any], product_id: str) -> Product:
        try:
            return read_product(raw, currency=self.spec.currency)
        except MCError as exc:
            self._note(f"product {product_id} could not be read: {exc}")
            return read_product({"id": product_id})

    def _selection(self, product: Product, raw: Mapping[str, Any]) -> list[tuple[str, str]]:
        """The option values to narrow get_product to: those the spec's option predicates name, and the size the
        product's own chart gives for the shopper's measurement. Empty when the known variants already carry every
        option value the product offers."""
        offered = _offered(raw)
        seen = {(fold(name), fold(value)) for variant in product.variants for name, value in variant.options}
        if all((fold(n), fold(v)) in seen for n, values in offered.items() for v in values):
            return []
        wanted: list[tuple[str, str]] = []
        for predicate in self.checked:
            if predicate.op == "eq" and isinstance(predicate.value, str):
                names = {fold(x) for x in option_names(predicate.field, predicate.option)}
                name = next((n for n in offered if fold(n) in names), None)
                if name is not None:
                    wanted.append((name, predicate.value))
            elif predicate.op == "fits":
                size = _fitting_size(product, predicate.value, offered)
                if size is not None:
                    wanted.append(size)
        return wanted

    def _judge(self, product: Product) -> _Checked:
        checked = _Checked(product)
        for variant in product.variants:
            if not variant.available:
                continue
            verdicts = {p.field: holds(p, product, variant, quantity=self.quantity) for p in self.checked}
            if all(v is True for v in verdicts.values()):
                checked.justified.append(variant)
            elif not any(v is False for v in verdicts.values()):
                for name, verdict in verdicts.items():
                    if verdict is None:
                        checked.unknown.setdefault(name, self._missing_name(name))
        return checked

    def _missing_name(self, field_name: str) -> str:
        predicate = self.spec.predicate(field_name)
        if predicate is not None and predicate.op in DETAIL_OPS:
            return detail_named(DETAIL_OPS[predicate.op]).missing
        return field_name

    async def _consult_policies(self, checked: list[_Checked]) -> list[_Checked]:
        """When the decisive detail is missing from every candidate that could otherwise fit, ask the store's
        policy tool once (a size guide often lives there) and read the answer as part of each product."""
        decisive = self.decisive
        if decisive is None or any(c.justified for c in checked):
            return checked
        lacking = {id(c) for c in checked if decisive.field in c.unknown}
        tool = next((t for t in POLICY_TOOLS if t in self.tools), None)
        missing = self._missing_name(decisive.field)
        query = POLICY_QUERIES.get(missing)
        if not lacking or tool is None or query is None:
            return checked
        result = await self._call(tool, {"query": query})
        if result.is_error:
            return checked
        lines, tables = _policy_text(payload_of(result))
        out = []
        for entry in checked:
            if id(entry) in lacking:
                enriched = replace(
                    entry.product,
                    lines=(*entry.product.lines, *lines),
                    tables=(*entry.product.tables, *tables),
                )
                out.append(self._judge(enriched))
            else:
                out.append(entry)
        return out

    def _choices(self, checked: Sequence[_Checked]) -> list[tuple[Variant, Product]]:
        """Every justified variant in search order; only the best of them when the shopper asked for the best (the
        cheapest, the cheapest per unit, the most or least of a nutrient). For a substitute task, the named product's
        variants first, then, when the shopper allows a substitute, the other products' variants that meet every
        other predicate."""
        pairs = [(v, c.product) for c in checked for v in c.justified]
        if self.prefer is not None:
            named = [(v, p) for v, p in pairs if _named_by(self.prefer, p)]
            pairs = named + (self._substitutes(checked) if self.spec.substitute_ok else [])
        superlative = next((p for p in self.spec.predicates if p.op in SUPERLATIVES), None)
        if superlative is not None and pairs:
            keep = set(best(superlative, [(p, v) for v, p in pairs]))
            pairs = [(v, p) for v, p in pairs if v.id in keep]
        return pairs

    def _substitutes(self, checked: Sequence[_Checked]) -> list[tuple[Variant, Product]]:
        """The closest substitutes for a named product: every other product's available variants that meet every
        predicate but the name, in search order."""
        assert self.prefer is not None
        others = [p for p in self.checked if p.op != "prefer"]
        out: list[tuple[Variant, Product]] = []
        for entry in checked:
            if _named_by(self.prefer, entry.product):
                continue
            for variant in entry.product.variants:
                if variant.available and all(
                    holds(p, entry.product, variant, quantity=self.quantity) is True for p in others
                ):
                    out.append((variant, entry.product))
        return out

    # -- the cart ----------------------------------------------------------------------------------------------------

    async def _add(self, variant: Variant, product: Product) -> Cart | None:
        return await self._add_lines([(variant, product)])

    async def _add_lines(self, lines: Sequence[tuple[Variant, Product]]) -> Cart | None:
        """A cart with one line per item, read back; None when the store did not make it or does not hold every line
        (an out-of-stock line is left out of the cart the store returns)."""
        if self.dialect.create not in self.tools:
            self._miss(f"tool:{self.dialect.create}", [])
            self._note(f"no surface offers {self.dialect.create}")
            return None
        d = self.dialect
        wanted = [(v.id, self.quantity) for v, _ in lines]
        items = [(p.id, v.id, self.quantity) for v, p in lines]
        cart: Cart | None = None
        calls = d.item_calls(items, currency=self.spec.currency)
        if d.intent is not None and d.intent in self.tools:
            intent_id = await self._intent(lines)
            if intent_id is None:
                return None
            calls = d.with_intent(calls, intent_id)
        for name, arguments in calls:
            result = await self._call(name, arguments)
            cart = d.cart_in(result)
            if cart is None:
                self._note(f"{name} did not return a cart for {wanted}: {result_text(result, limit=200)}")
                return None
        if cart is None:
            return None
        self.cart_id = cart.cart_id or self.cart_id
        if d.read in self.tools and cart.cart_id:
            read = await self._call(d.read, d.read_arguments(cart.cart_id))
            cart = d.cart_in(read) or cart
        held = {(line.variant_id, line.quantity) for line in cart.lines}
        lacking = [vid for vid, qty in wanted if (vid, qty) not in held]
        if lacking:
            self._note(f"the cart read back does not hold {self.quantity} of {', '.join(lacking)}")
            return None
        self.lines = list(lines)
        return cart

    async def _intent(self, lines: Sequence[tuple[Variant, Product]]) -> str | None:
        """The shopper's intent stated to the store before its checkout is made (AP2's open checkout mandate, which the
        shopper's wallet signs): the items the shopper wants, and the id the store gave it."""
        d = self.dialect
        assert d.intent is not None
        items = [(v.id, f"{p.title} ({v.label()})", self.quantity) for v, p in lines]
        result = await self._call(d.intent, d.intent_arguments(items))
        found = d.intent_id(result)
        if found is None:
            self._note(f"{d.intent} gave no intent id: {result_text(result, limit=200)}")
        return found

    async def _bundle_cart(self, checked: Sequence[_Checked]) -> AgentResult:
        """A bundle: one justified variant of every named item in one cart, else a refusal naming the item."""
        assert self.bundle is not None
        picked: list[tuple[Variant, Product]] = []
        for item in bundle_items(self.bundle.value):
            found = next(((v, c.product) for c in checked for v in c.justified if item_holds(item, c.product, v)), None)
            if found is None:
                title = str(item.get("title") or item.get("product_id"))
                self._note(f"no variant of {title} could be justified from the store's data")
                return self._finish_refusal(f"I couldn't find {title} in this store, so I haven't added the set.")
            picked.append(found)
        cart = await self._add_lines(picked)
        if cart is None:
            return self._finish_refusal()
        variant, product = picked[0]
        if self.harness is not None:
            try:
                return await self._checkout(self.harness, cart, variant, product)
            except _OverBudget as exc:
                self._note(f"the checkout for the set comes to {exc.shown}, over the shopper's budget")
                self._halt("The set comes to more than your budget at checkout, so I haven't placed an order.")
        return self._finish_cart(cart, variant, product)

    # -- the checkout --------------------------------------------------------------------------------------------------

    async def _checkout(self, harness: CheckoutHarness, cart: Cart, variant: Variant, product: Product) -> AgentResult:
        """From the verified cart to an order. Every stop the store, the shopper or the bank decides raises `_Stop`
        with the message that tells the shopper why; a total over the shopper's budget raises `_OverBudget`."""
        buyer = self.spec.buyer
        if buyer is not None:
            doc, where = await self._buyer_checkout(buyer, cart)
        else:
            doc, where = await self._open_checkout(harness, cart), "to your saved address"
        checkout_id = str(doc["id"])
        total = total_of(doc)
        if total is None:
            self._halt("The store's checkout does not show a total, so I can't ask you to approve it.")
        shown = money(total.amount, total.currency)
        if self._over_budget(doc, total):
            raise _OverBudget(shown)
        entry = entry_for(self.tools, specs=self.specs)
        choice = entry.selected(doc) if entry is not None else None
        delivery = f", delivered {choice.title.lower()}" if choice is not None else ""
        ask = (
            f"I'm ready to order {self._items_words(product, variant)}{delivery}, {where}. "
            f"The total is {shown}, shipping and tax included. Please confirm it in your wallet."
        )
        self.sink.turn("agent", "message", text=ask)
        paid: Mapping[str, Any] | None = None
        for attempt in range(PAY_TRIES):
            answer = await self._authorized(harness, checkout_id, shown)
            paid = await self._pay(harness, checkout_id, answer)
            if attempt + 1 == PAY_TRIES or (paid is not None and self.dialect.order_of(paid) is not None):
                break
            if self._retryable(paid):
                self._note("the issuer declined with a processing error, which a retry may clear")
                failure = self.dialect.payment_failure(paid) if paid is not None else None
                self.sink.turn(
                    "agent",
                    "message",
                    text=f"Your bank couldn't process the payment ({failure}). I'll ask your wallet once more; the "
                    f"total is still {shown}.",
                )
                await self._fresh_request(checkout_id)
                continue
            if self.agent.recover is True and self._other_instrument(harness, paid):
                failure = self.dialect.payment_failure(paid) if paid is not None else None
                self._note("the row is designed to recover and the wallet holds another instrument: paying with it")
                self.sink.turn(
                    "agent",
                    "message",
                    text=f"Your payment didn't go through ({failure}). I'll pay with the other payment method in your "
                    f"wallet instead; the total is still {shown}.",
                )
                await self._fresh_request(checkout_id)
                continue
            moved = await self._moved(checkout_id, total)
            if moved is None:
                break
            before, total = shown, moved
            shown = money(total.amount, total.currency)
            self._note(f"the store changed the checkout's total at payment, from {before} to {shown}")
            self.sink.turn(
                "agent",
                "message",
                text=f"The store changed the price at checkout: the total is now {shown}, not {before}. Please "
                "confirm the new total in your wallet.",
            )
        return await self._placed(paid, product, variant, shown, where)

    async def _moved(self, checkout_id: str, approved: Money) -> Money | None:
        """The checkout's new total when the store refused the payment because a price changed (it repriced the
        checkout), read again through the store's own checkout tools; None when the total is the one approved."""
        doc: Mapping[str, Any] | None = None
        for name, arguments in self.dialect.refresh_calls(checkout_id):
            if name not in self.tools:
                return None
            doc = await self._checkout_call(name, arguments) or doc
        total = total_of(doc) if doc is not None else None
        if total is None or total == approved or (doc is not None and self.dialect.payment_failure(doc)):
            return None
        return total

    def _other_instrument(self, harness: CheckoutHarness, doc: Mapping[str, Any] | None) -> bool:
        """After a payment the issuer declined (a decline or a failed 3-D Secure, never a reprice): set the declined
        instrument aside, and whether the wallet still holds one to pay with."""
        if doc is None or self.dialect.order_of(doc) is not None:
            return False
        failure = (self.dialect.payment_failure(doc) or "").lower()
        if not any(word in failure for word in DECLINE_WORDS):
            return False
        issued = harness.wallet.issued
        return bool(issued) and harness.wallet.set_aside(issued[-1].token)

    async def _fresh_request(self, checkout_id: str) -> None:
        """Before paying again: the checkout read again through the store's own calls, so a rail that answers each
        payment request once (AP2: "request_used", ep-27bdfa84e6bb) signs a new one for the same basket."""
        for name, arguments in self.dialect.refresh_calls(checkout_id):
            if name in self.tools:
                await self._checkout_call(name, arguments)

    def _retryable(self, doc: Mapping[str, Any] | None) -> bool:
        """Whether the payment failed in a way the same credential may pay on another try: a processing error."""
        if doc is None or self.dialect.order_of(doc) is not None:
            return False
        failure = (self.dialect.payment_failure(doc) or "").lower()
        return any(word in failure for word in RETRY_WORDS)

    def _over_budget(self, doc: Mapping[str, Any], total: Money) -> bool:
        """Whether the checkout costs more than the shopper's budget: its total when the budget is all in, else its
        items alone."""
        budget = self.spec.predicate("budget")
        value = budget.value if budget is not None else None
        amount = value.get("amount") if isinstance(value, Mapping) else None
        if not isinstance(amount, int) or (isinstance(value, Mapping) and value.get("currency") != total.currency):
            return False
        if self.spec.budget_scope == "total":
            return total.amount > amount
        view = self.dialect.view_of(doc)
        items = sum(line.unit_price * line.quantity for line in view.lines) if view is not None else 0
        return items > amount

    async def _buyer_checkout(self, buyer: Buyer, cart: Cart) -> tuple[Mapping[str, Any], str]:
        """A checkout opened with no buyer details, given exactly what the store reports missing and the shopper
        discloses, through the store's own operations; then the option that matches the delivery preference. The
        checkout, and where the order goes in words."""
        entry = entry_for(self.tools, specs=self.specs)
        if entry is None or cart.cart_id is None:
            self._miss(f"tool:{self.dialect.update_checkout}", [])
            self._halt("I couldn't check out: the store does not offer checkout to agents.")
        name, arguments = entry.create(cart.cart_id)
        doc = await self._checkout_call(name, arguments)
        if doc is None:
            self._halt("I couldn't start the checkout: the store did not open one for your cart.")
        known: dict[str, bool] = {}
        entered: set[str] = set()
        for _ in range(ENTRY_ROUNDS):
            missing = entry.missing(doc)
            for name in missing:
                if name == "shipping_address" and known.get("fulfillment") and _collects(buyer):
                    known[name] = True  # nothing to give: a collected order ships nowhere
                elif name not in known:
                    known[name] = await self._buyer_field(buyer, name)
            new = [f for f in missing if known.get(f) and f not in entered]
            if not new:
                break
            entered.update(new)
            for call_name, call_arguments in entry.enter(doc, self._details(buyer, new)):
                doc = await self._checkout_call(call_name, call_arguments) or doc
        refused = [f for f in entry.missing(doc) if not known.get(f) and f != "fulfillment"]
        if refused:
            words = ", ".join(self.agent.phrases.label(f) for f in refused)
            self._halt(f"I couldn't check out without your {words}, so I haven't placed an order.")
        doc = await self._choose_option(entry, buyer, doc)
        if not entry.ready(doc):
            self._halt("I couldn't complete the checkout details the store asks for.")
        return doc, _destination_words(buyer)

    async def _buyer_field(self, buyer: Buyer, field: str) -> bool:
        """Whether the shopper has given a buyer field the checkout needs: stated in the opening, or disclosed when
        asked now. A delivery preference the shopper leaves open is the plain assumption, standard delivery."""
        found = buyer.field(field)
        if found is not None and found.source == "stated":
            return True
        if field in self.disclosed:
            return True
        question = BUYER_QUESTIONS.get(field)
        if question is None:
            return False
        reply = await self._question(question)
        if field in reply.disclosed:
            return True
        return field == "fulfillment"

    def _details(self, buyer: Buyer, fields: Sequence[str]) -> Details:
        """The disclosed values of `fields`, as the checkout takes them. An address goes with the shipping method it
        belongs to, which the delivery preference decides."""
        preference = buyer.fulfillment.value if buyer.fulfillment is not None else "standard"
        method: Literal["shipping", "pickup"] = "pickup" if preference == "pickup" else "shipping"
        address = buyer.shipping_address.value if buyer.shipping_address is not None else None
        return Details(
            email=buyer.email.value if "email" in fields and buyer.email is not None else None,
            phone=buyer.phone.value if "phone" in fields and buyer.phone is not None else None,
            address=address if "shipping_address" in fields and method == "shipping" else None,
            method=method if {"fulfillment", "shipping_address"} & set(fields) else None,
        )

    async def _choose_option(self, entry: CheckoutEntry, buyer: Buyer, doc: Mapping[str, Any]) -> Mapping[str, Any]:
        """Select the option that matches the shopper's delivery preference and meets their deadline."""
        fulfillment = buyer.fulfillment
        preference = fulfillment.value if fulfillment is not None else "standard"
        deadline = fulfillment.deadline_days if fulfillment is not None else None
        options = entry.options(doc)
        if not options and preference != "pickup" and deadline is None and entry.ready(doc):
            # the store signed the checkout with its delivery already set and lists no choice (an AP2 merchant
            # under some packs, D94): there is nothing to choose
            return doc
        choice = pick(options, preference, deadline)
        if choice is None:
            wanted = f"{preference} option" + (f" that arrives within {deadline} days" if deadline else "")
            self._halt(f"The store's checkout offers no {wanted}, so I haven't placed an order.")
        current = entry.selected(doc)
        if current is not None and current.option_id == choice.option_id and current.group_id == choice.group_id:
            return doc
        name, arguments = entry.select(doc, choice)
        chosen = await self._checkout_call(name, arguments)
        rest = [o for o in options if o != choice]
        while chosen is None and rest:
            # D104: a store may list an option it then refuses ("standard_free" on a WooCommerce storefront); the
            # next option of the shopper's kind that meets the deadline does as well
            choice = pick(rest, preference, deadline)
            if choice is None:
                break
            rest = [o for o in rest if o != choice]
            name, arguments = entry.select(doc, choice)
            chosen = await self._checkout_call(name, arguments)
        if chosen is None:
            self._halt("I couldn't choose the delivery option at the store's checkout.")
        return chosen

    async def _open_checkout(self, harness: CheckoutHarness, cart: Cart) -> Mapping[str, Any]:
        """A checkout from the cart, with the shopper's saved details and the cheapest shipping option selected."""
        d = self.dialect
        missing = [
            name for name in (d.create_checkout, d.update_checkout, d.complete_checkout) if name not in self.tools
        ]
        for name in missing:
            self._miss(f"tool:{name}", [])
        entry = entry_for(self.tools, specs=self.specs)
        if missing or cart.cart_id is None or entry is None:
            self._halt("I couldn't check out: the store does not offer checkout to agents.")
        details = harness.wallet.shipping_details()
        doc: Mapping[str, Any] | None = None
        for name, arguments in d.open_checkout_calls(cart.cart_id, details):
            doc = await self._checkout_call(name, arguments) or doc
        if doc is None:
            self._halt("I couldn't start the checkout: the store did not open one for your cart.")
        choice = next((o for o in entry.options(doc) if o.method == "shipping"), None)
        if choice is not None and entry.selected(doc) is None:
            name, arguments = entry.select(doc, choice)
            doc = await self._checkout_call(name, arguments)
        if doc is None or not entry.ready(doc):
            self._halt("I couldn't complete the checkout details the store asks for.")
        return doc

    async def _authorized(self, harness: CheckoutHarness, checkout_id: str, shown: str) -> Mapping[str, Any]:
        """The payment for the paying call, from the rail's own wallet steps (D94), the shopper approving where the
        official flow asks the buyer. A rail whose payment answers the store's request first (MPP's 402 challenge)
        asks for it before the wallet answers it."""
        d = self.dialect
        extra: dict[str, Any] = {}
        if d.prepare_payment is not None:
            if d.prepare_payment not in self.tools:
                self._halt("I couldn't pay: the store does not offer its payment request to agents.")
            asked = await self._call(d.prepare_payment, d.prepare_arguments(checkout_id))
            prepared = d.prepared(asked)
            if prepared is None:
                self._halt("I couldn't pay: the store did not ask for a payment I could answer.")
            extra = prepared

        async def run(name: str, arguments: dict[str, Any]) -> ToolResult:
            return await self._harness_call(harness, name, arguments)

        outcome = await harness.authorize(checkout_id, run)
        if outcome.status == "declined":
            self._halt(f"I haven't placed the order for {shown}, as you asked. Your cart is still there.")
        if outcome.status != "approved" or outcome.answer is None:
            self._halt(f"I couldn't pay: your wallet did not give me a payment credential ({outcome.reason}).")
        return {**outcome.answer, **extra}

    async def _pay(
        self, harness: CheckoutHarness, checkout_id: str, answer: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        """Complete the checkout; a 3-D Secure challenge goes to the wallet, and the checkout is read again. A rail
        whose payment answers the store's request first (MPP's 402 challenge) asks for it just before paying, and reads
        the checkout when the payment does not come back paid."""
        d = self.dialect
        doc = await self._checkout_call(d.complete_checkout, d.complete_arguments(checkout_id, answer))
        if d.prepare_payment is not None and (doc is None or d.order_of(doc) is None) and d.get_checkout in self.tools:
            doc = await self._checkout_call(d.get_checkout, d.id_arguments(checkout_id)) or doc
        if doc is None or harness.challenge_for(checkout_id, doc) is None:
            return doc
        await self._harness_call(harness, CHALLENGE, {"checkout_id": checkout_id})
        if d.get_checkout not in self.tools:
            return None
        return await self._checkout_call(d.get_checkout, d.id_arguments(checkout_id))

    async def _placed(
        self, doc: Mapping[str, Any] | None, product: Product, variant: Variant, shown: str, where: str
    ) -> AgentResult:
        """The order read back and the journey's end; or, without one, the shopper told why, and nothing retried."""
        d = self.dialect
        order_id = d.order_of(doc) if doc is not None else None
        if order_id is None:
            failure = d.payment_failure(doc) if doc is not None else None
            if failure:
                self._halt(
                    f"Your payment didn't go through ({failure}). I haven't placed the order, and I won't try "
                    "another card unless you ask me to."
                )
            self._halt("I couldn't place the order: the store did not confirm it.")
        label = (d.order_label(doc) if doc is not None else None) or order_id
        if d.get_order is not None and d.get_order in self.tools:
            result = await self._call(d.get_order, d.id_arguments(order_id))
            order = payload_of(result) if not result.is_error else None
            label = str(order.get("label") or label) if isinstance(order, Mapping) else label
        sent = "It's ready to collect from the store." if where == COLLECT else f"It ships {where}."
        message = f"I placed your order {label}: {self._items_words(product, variant)}, for {shown}. {sent}"
        message += self._substitution_words() + self._policy_answer()
        self.sink.turn(
            "agent",
            "tool_call",
            tool="done",
            arguments={"status": "order_placed", "message": message, "order_id": order_id},
        )
        self.sink.turn("agent", "message", text=message)
        return self._result("order_placed", declared="order_placed", message=message)

    async def _checkout_call(self, name: str, arguments: dict[str, Any]) -> Mapping[str, Any] | None:
        """A checkout tool call, and the checkout it returned (None for an error response or no checkout)."""
        """A checkout tool call, and the checkout it returned (None for a reply that shows none). A storefront answers a
        failed payment with the checkout page and an error on it, so an error result is still read for the checkout
        it carries; UCP's and ACP's error replies carry none."""
        result = await self._call(name, arguments)
        doc = self.dialect.checkout_doc(payload_of(result))
        if doc is None:
            self._note(f"{name} returned no checkout: {result_text(result, limit=200)}")
        return doc

    def _halt(self, message: str) -> NoReturn:
        """End a checkout row without an order: the message tells the shopper why, and nothing else is tried."""
        self.sink.turn("agent", "tool_call", tool="done", arguments={"status": "cannot_continue", "message": message})
        self.sink.turn("agent", "message", text=message)
        raise _Stop(self._result("gave_up", declared="cannot_continue", message=message))

    # -- endings -----------------------------------------------------------------------------------------------------

    def _finish_cart(self, cart: Cart, variant: Variant, product: Product) -> AgentResult:
        checked = [p.field for p in self.checked if p.source != "default"]
        labels = ", ".join(self.agent.phrases.label(f) for f in checked) or "what you asked for"
        if self.dialect.boundary == "selection":
            return self._finish_selection(cart, variant, product, labels)
        message = (
            f"I added {self._items_words(product, variant)} to your cart. I checked it against what you told me: "
            f"{labels}." + self._substitution_words() + self._policy_answer()
        )
        self.sink.turn("agent", "tool_call", tool="done", arguments={"status": "cart_ready", "message": message})
        self.sink.turn("agent", "message", text=message)
        return self._result("cart_ready", declared="cart_ready", message=message, cart_id=cart.cart_id)

    def _finish_selection(self, cart: Cart, variant: Variant, product: Product, labels: str) -> AgentResult:
        """A website journey's end (D31): the chosen variant, named by the selection's id; nothing is carted."""
        message = (
            f"I chose {self._items_words(product, variant)} on the store's website. I checked it against what you "
            f"told me: {labels}. Nothing has been added to a cart or bought; it is ready for you to buy."
            + self._substitution_words()
            + self._policy_answer()
        )
        arguments = {"status": "selection_ready", "message": message, "selection_id": cart.cart_id}
        self.sink.turn("agent", "tool_call", tool="done", arguments=arguments)
        self.sink.turn("agent", "message", text=message)
        return self._result("selection_ready", declared="selection_ready", message=message, cart_id=cart.cart_id)

    def _items_words(self, product: Product, variant: Variant) -> str:
        """What the cart holds, in words: a bundle's items one of each, else the item and its quantity."""
        if self.bundle is not None and len(self.lines) > 1:
            listed = ", ".join(f"{p.title} ({v.label()})" for v, p in self.lines)
            return f"{listed}, one of each"
        return f"{product.title} ({variant.label()}), quantity {self.quantity}"

    def _substitution_words(self) -> str:
        if self.out_of_stock is None or not self.lines:
            return ""
        chosen = ", ".join(p.title for _, p in self.lines)
        return f" {self.out_of_stock} is out of stock, so I chose {chosen} instead, as you said a similar one would do."

    def _record_missing(self, checked: Sequence[_Checked]) -> None:
        """Note each detail the store's data lacked, with the products that lacked it: the store's side of a
        failure, written to the trace whether or not a cart is made."""
        for entry in checked:
            for name in sorted(set(entry.unknown.values())):
                self._miss(name, [entry.product.id])
        for name, products in sorted(self.missing.items()):
            if products:
                self.sink.turn(
                    "agent",
                    "note",
                    text=f"missing detail {name} on {len(products)} products",
                    arguments={"missing": name, "products": products},
                )

    def _finish_refusal(self, message: str | None = None) -> AgentResult:
        if message is None and self.out_of_stock is not None:
            tail = (
                " and I couldn't find a similar one I can show fits"
                if self.spec.substitute_ok
                else ", and you asked me not to get anything else"
            )
            message = f"{self.out_of_stock} is out of stock{tail}, so I haven't added anything."
        elif message is None and self.missing:
            gaps = ", ".join(sorted(self.missing))
            message = f"I couldn't find anything I can show fits everything you need. The store does not say: {gaps}."
        elif message is None:
            message = "I couldn't find anything in this store that fits everything you need."
        message += self._policy_answer()
        self.sink.turn("agent", "tool_call", tool="done", arguments={"status": "no_match", "message": message})
        self.sink.turn("agent", "message", text=message)
        return self._result("gave_up", declared="no_match", message=message)

    def _miss(self, name: str, products: Sequence[str]) -> None:
        """Record products lacking a detail, kept sorted so the result does not depend on the store's search order."""
        listed = self.missing.setdefault(name, [])
        listed.extend(p for p in products if p not in listed)
        listed.sort()

    def _note(self, text: str) -> None:
        self.sink.turn("agent", "note", text=text)

    def _end(self, reason: EndReason) -> AgentResult:
        return self._result(reason)

    def _result(
        self,
        reason: EndReason,
        *,
        declared: DeclaredStatus | None = None,
        message: str = "",
        cart_id: str | None = None,
    ) -> AgentResult:
        return AgentResult(
            end_reason=reason,
            final_message=message,
            steps=self.steps,
            tool_calls=self.tool_calls,
            cart_id=cart_id or self.cart_id,
            declared_status=declared,
            missing_details={k: list(v) for k, v in sorted(self.missing.items())},
            checkout_id=self.harness.checkout_id if self.harness is not None else None,
            order_id=self.harness.order_id if self.harness is not None else None,
        )


COLLECT = "for collection from the store"


def _destination_words(buyer: Buyer) -> str:
    return COLLECT if _collects(buyer) else "to the address you gave me"


def _collects(buyer: Buyer) -> bool:
    """The shopper wants to collect the order: no address is needed."""
    return buyer.fulfillment is not None and buyer.fulfillment.value == "pickup"


def _for_sale(product: Product) -> bool:
    """Whether any variant of the product is for sale: none listed, or none available, is nothing to buy."""
    return any(v.available for v in product.variants)


def _words(text: str) -> set[str]:
    """The words of a text that name something (three letters or more), folded."""
    return {w for w in fold(text).replace("-", " ").split() if len(w) >= 3 and w.isalpha()}


def _named_by(predicate: Predicate, product: Product) -> bool:
    value = predicate.value
    return isinstance(value, str) and (fold(product.title) == fold(value) or product.id == value)


def _topic(predicate: Predicate) -> str:
    value = predicate.value
    topic = value.get("topic") if isinstance(value, Mapping) else None
    return str(topic) if isinstance(topic, str) else "policy"


def _about(texts: Sequence[str], words: Sequence[str]) -> list[str]:
    return [t for t in texts if any(w in t.lower() for w in words)]


def policies_in(payload: Any) -> list[Mapping[str, Any]]:
    """The UCP `policies` a catalog reply carries at its root (or under MCP's structuredContent / result)."""
    data = payload
    for _ in range(3):
        if not isinstance(data, Mapping):
            return []
        found = data.get("policies")
        if isinstance(found, list):
            return [p for p in found if isinstance(p, Mapping)]
        inner = next((data[k] for k in ("structuredContent", "result") if isinstance(data.get(k), Mapping)), None)
        if inner is None:
            return []
        data = inner
    return []


def policy_text(entry: Mapping[str, Any]) -> str:
    """A UCP policy as one line: its type and its description (plain, markdown or html, whichever it gives)."""
    description = entry.get("description")
    if isinstance(description, Mapping):
        text = next((str(description[k]) for k in ("plain", "markdown", "html") if description.get(k)), "")
    else:
        text = str(description or "")
    text = " ".join(_strings(text).split()) if text else ""
    if "</" in text:
        lines, _ = read_html(text)
        text = " ".join(lines)
    kind = entry.get("type")
    return f"{kind}: {text}".strip(": ") if isinstance(kind, str) else text


def _policy_text(payload: Any) -> tuple[tuple[str, ...], tuple[tuple[tuple[str, ...], ...], ...]]:
    """A policy tool's answer as product text: its lines and its tables."""
    text = payload if isinstance(payload, str) else _strings(payload)
    if "</" in text:
        lines, tables = read_html(text)
    else:
        lines = lines_of(text)
        tables = pipe_tables(lines)
    return tuple(lines), tuple(tuple(tuple(row) for row in table) for table in tables)


def _strings(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return "\n".join(_strings(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(_strings(v) for v in value)
    return ""


def merged(primary: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """`primary` with every variant of `extra` it does not list itself added after its own, so a get_product reply
    that shows one selected variant keeps the variants the search result listed."""
    out = dict(primary)
    own = [v for v in primary.get("variants") or [] if isinstance(v, Mapping)]
    ids = {str(v.get("id")) for v in own}
    more = [v for v in extra.get("variants") or [] if isinstance(v, Mapping) and str(v.get("id")) not in ids]
    out["variants"] = [*own, *more]
    return out


def _offered(raw: Mapping[str, Any]) -> dict[str, list[str]]:
    """Option name to the values the product says it offers (UCP's product-level `options`)."""
    out: dict[str, list[str]] = {}
    for option in raw.get("options") or []:
        if not isinstance(option, Mapping) or not isinstance(option.get("name"), str):
            continue
        values = [v.get("label") if isinstance(v, Mapping) else v for v in option.get("values") or []]
        out[option["name"]] = [str(v) for v in values if isinstance(v, str | int) and str(v)]
    return out


def _fitting_size(product: Product, value: Any, offered: Mapping[str, list[str]]) -> tuple[str, str] | None:
    """The size option value the product's own chart gives for the shopper's measurement, when there is one."""
    chart = read_chart(product)
    wanted = measurement_of(value)
    if chart is None or wanted is None:
        return None
    measure, mm = wanted
    keys = {k for i in chart.fitting(measure, mm) for k in chart.rows[i].keys if k}
    for name, values in offered.items():
        if "size" in fold(name):
            match = next((v for v in values if size_key(v) in keys), None)
            if match is not None:
                return (name, match)
    return None


def clipped_query(query: str, limit: int = QUERY_CHARS) -> str:
    """`query` as a search box takes it: whole when it fits in `limit` characters, else cut at the last word that
    fits (a single word longer than the box is cut at the box)."""
    if len(query) <= limit:
        return query
    cut = query[: limit + 1].rsplit(" ", 1)[0] if " " in query[: limit + 1] else query[:limit]
    return cut[:limit].rstrip(" ,;:-/")


def _narrower(query: str) -> list[str]:
    """Shorter forms of a search query, most specific first (`_narrower_search`)."""
    words = query.split()
    out: list[str] = []
    if len(words) > 3:
        out += [" ".join(words[:3]), " ".join(words[:2])]
    if len(words) >= 2:
        head = words[-1].strip(",.()").lower()
        if len(head) > 2:
            # a kind bought in pairs is asked for in the plural ("running shoes"), which a title may say singly
            out += [head, singular_of(head), plural_of(head)]
    return list(dict.fromkeys(q for q in out if q.strip() and q != query))


def _titled(found: Mapping[str, Mapping[str, Any]], title: str) -> bool:
    return fold(title) in {fold(str(v.get("title") or "")) for v in found.values()}

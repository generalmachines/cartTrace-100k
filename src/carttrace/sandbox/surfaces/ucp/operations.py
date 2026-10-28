"""The UCP operations: every tool a release defines, as one core command and one document.

Why one entry per call. Each tool call becomes exactly one entry in the sandbox's event log: the command it translates
to (a `read` for gets, searches and lookups, so faults that fire on any command fire on them too), or a logged
refusal when the call is refused before translation (arguments that fail the release's schema, a card number, a
resource of another session). Replaying the log therefore replays the whole conversation with the store, refusals
included, to the same digest.

What a call returns. A `Reply` holds either a UCP document (the resource, or an error response when no resource can
be established) or a protocol `Fault` the transport answers itself (`carttrace.sandbox.surfaces.ucp.messages`). MCP and
REST share this layer and differ only in how they carry a reply.

Sessions. Every call acts for one core session. Carts, checkouts and orders belong to the session that made them:
another session asking for one is told it does not exist, the same answer as for an id that was never issued, so one
agent's session can neither see nor change another's.

Rules from the UCP 2026-08-25 specification applied at every release served:
- cart to checkout: `cart_id` converts a cart; cart contents win over the checkout payload, and an open checkout
  already made from the cart is returned instead of a second one (cart, "Cart-to-Checkout Conversion");
- a canceled cart is not found afterwards (cart, "Cancel Cart");
- a search needs a query or filters (catalog search, "Search Inputs");
- a lookup over the batch limit is refused as invalid params (catalog lookup, "Batch Size");
- while a 3-D Secure Action is outstanding (2026-08-25 with payment authentication declared), update and complete
  are refused with a recoverable error and the checkout is left as it is (checkout, "Accepted completion").

The store's own errors (D70). A replica episode's pack lists the calls its draw fails at its store's live rates
(`EnvPack.failing_calls`: the n-th call of a tool in a session, counted from the calls that pass the schema). Such a
call is logged as refused and answered as the store answered live: Shopify's create_cart "Throttled" as an error cart
(an empty id, no lines, `ucp.status` "error", one error message of code `error`, severity requires_buyer_input:
store-137 ep-422a0968cd25 seq 25), and an HTTP 500 as JSON-RPC -32603 "Internal error" with no data (store-398
ep-7b185304b778 seq 32).
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from carttrace.core.cards import looks_like_card_number
from carttrace.core.errors import ProtocolError
from carttrace.protocols.registry import registry
from carttrace.protocols.tools import ToolSpec
from carttrace.sandbox.core.commands import MUTATING
from carttrace.sandbox.core.config import SHOP_PAY_NAMESPACE
from carttrace.sandbox.core.errors import IllegalTransition, InvalidCommand, NotFound, Unavailable
from carttrace.sandbox.core.machines import OPEN_CHECKOUT
from carttrace.sandbox.core.state import Cart, Checkout, Order, State
from carttrace.sandbox.events.runtime import Outcome
from carttrace.sandbox.platforms import MINOR_UNITS, ToolDoc, market_excludes, medusa, tool_docs
from carttrace.sandbox.platforms.woocommerce import money_text
from carttrace.sandbox.presented import presented
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.ucp.catalog import LOOKUP_LIMIT, Catalog, lookup_docs, served_catalog
from carttrace.sandbox.surfaces.ucp.dialect import ResponseKind
from carttrace.sandbox.surfaces.ucp.envelope import Envelope
from carttrace.sandbox.surfaces.ucp.guard import card_number_field
from carttrace.sandbox.surfaces.ucp.ids import CartIds, delivery_group_gid
from carttrace.sandbox.surfaces.ucp.listing import Listing, listed_book, served_listing
from carttrace.sandbox.surfaces.ucp.messages import (
    INTERNAL_ERROR,
    Fault,
    Message,
    Paths,
    Wording,
    batch_too_large,
    error,
    event_messages,
    fault_for,
    info,
    refusal,
    warning,
)
from carttrace.sandbox.surfaces.ucp.render import Renderer
from carttrace.sandbox.surfaces.ucp.requests import (
    PaymentChoice,
    buyer,
    country,
    discount_codes,
    fulfillment_groups,
    line_inputs,
    payment,
)
from carttrace.sandbox.surfaces.ucp.schemas import TRANSPORT_PARAMS, Problem, ToolBook, release_tools, tool_book
from carttrace.sandbox.surfaces.values import obj, text

KINDS: Mapping[str, ResponseKind] = {
    "create_cart": "cart",
    "get_cart": "cart",
    "update_cart": "cart",
    "cancel_cart": "cart",
    "create_checkout": "checkout",
    "get_checkout": "checkout",
    "update_checkout": "checkout",
    "complete_checkout": "checkout",
    "cancel_checkout": "checkout",
    "get_order": "order",
    "search_catalog": "search",
    "lookup_catalog": "lookup",
    "get_product": "lookup",
}
"""The response kind of each operation: which capabilities its documents list, error responses included."""

STATE_COVERED = frozenset({"payment_declined", "payment_authentication_failed", "payment_handler_unavailable"})
"""Refusals whose outcome the rendered checkout already explains from state; not repeated as a second message."""


class UnknownTool(ProtocolError):
    """A tool name the served release does not define. MCP answers -32602, REST has no route for it."""


@dataclass(frozen=True, slots=True)
class Reply:
    """A UCP document, or a protocol fault the transport answers itself."""

    doc: dict[str, Any] | None = None
    fault: Fault | None = None
    created: bool = False
    """A new resource was made (REST answers 201)."""
    text: str | None = None
    """A refusal a platform sends as plain text (live Shopify: "Missing required arguments: catalog"): MCP answers it
    as the tool result's only content; REST, which has no text result, still answers `doc`."""
    status: int | None = None
    """The HTTP status a platform's plugin answers `doc` with (Medusa's 400, 404 and 500, D68); None lets the binding
    choose (200, or 201 for a new resource)."""

    @property
    def is_error(self) -> bool:
        """The call ended in an error: a fault, an error status, or an error response (`ucp.status` is "error")."""
        if self.fault is not None or (self.status is not None and self.status >= 400):
            return True
        ucp = (self.doc or {}).get("ucp")
        return isinstance(ucp, Mapping) and ucp.get("status") == "error"

    def error_code(self) -> str | None:
        """The first error's code, for `ToolResult.error_code`."""
        if self.fault is not None:
            return self.fault.reason
        if self.doc is not None and isinstance(self.doc.get("type"), str) and self.status is not None:
            return str(self.doc["type"])
        for m in (self.doc or {}).get("messages") or ():
            if isinstance(m, Mapping) and m.get("type") == "error":
                return str(m.get("code"))
        return None


@dataclass(frozen=True, slots=True)
class Call:
    name: str
    arguments: Mapping[str, Any]
    session: str
    base: str

    def payload(self, key: str) -> Mapping[str, Any]:
        value = self.arguments.get(key)
        return value if isinstance(value, Mapping) else {}

    @property
    def target(self) -> str:
        return str(self.arguments.get("id") or "")

    @property
    def idempotency_key(self) -> str | None:
        meta = self.arguments.get("meta")
        key = meta.get("idempotency-key") if isinstance(meta, Mapping) else None
        if key is None:
            key = self.arguments.get("idempotency_key")  # 2026-01-11: a parameter of complete and cancel
        return key if isinstance(key, str) and 0 < len(key) <= 255 else None


Op = Callable[[Call], Reply]


THROTTLED_TEXT = "Throttled"
"""Live Shopify's whole message when it throttles a call (store-111, store-116, store-137 create_cart, D70)."""


class UcpOperations:
    """The tools of the served release over one host."""

    def __init__(self, host: SandboxHost) -> None:
        self.host = host
        self.dialect = host.ucp
        self.platform = host.platform
        self.listing: Listing | None = served_listing(
            host.pack, self.platform, mcp=host.mcp.version, ucp=self.dialect.version
        )
        """The store's own recorded tools/list, listed verbatim (D40); None lists the release's binding."""
        self.book: ToolBook = (
            tool_book(self.dialect.version) if self.listing is None else listed_book(self.dialect.version, self.listing)
        )
        defined = set(registry().ucp_capabilities(self.dialect.version))
        self.capabilities = frozenset(defined)
        """The capabilities the served release defines."""
        self.envelope = Envelope(
            self.dialect,
            host.profile,
            host.pack,
            defined=defined,
            full=self.platform.full_ucp_block,
            extends=self.platform.block_extends,
            supported_versions=self.platform.supported_versions,
            unserved=self.platform.unserved,
        )
        self.catalog: Catalog = served_catalog(host.pack, self.platform)
        config = host.pack.config()
        self.market = config.market
        self.core_rules = config.rules
        self.core_handlers = config.handlers
        """Handler id to namespace, for what a completion's handler_id resolves to (the Shop Pay instrument check)."""
        currency = config.market.currency
        self.wording = Wording(
            self.platform.wording,
            money=lambda amount: money_text(amount, currency),
            sold_out_warning=self.platform.sold_out_warning,
        )
        """The platform's own words for refusals (WooCommerce's coupon and stock sentences, Magento's options)."""
        self.accepted = config.market.accepts
        self.ids = CartIds(host.pack.id, keyed=self.platform.keyed_cart_ids)
        self.render = Renderer(
            self.dialect,
            self.envelope,
            self.catalog,
            config.market,
            config.discount_codes,
            platform=self.platform,
            ids=self.ids,
            coupons=config.coupon_table(),
            wording=self.wording,
            psp=config.psp,
        )
        self.docs: Mapping[str, ToolDoc] = (
            tool_docs(self.platform.tools_file) if self.platform.tools_file is not None else {}
        )
        """The platform's own descriptions and output schemas, by tool name (empty for the plain dialect)."""
        self.outputs: dict[str, dict[str, Any] | None] = {
            t.name: t.output_schema for t in release_tools(self.dialect.version, outputs=True)
        }
        """The release's result schemas, listed as `outputSchema` where the platform has none of its own."""
        self.calls_made: dict[tuple[str, str], int] = {}
        """(session, tool) to the calls that reached the store, so a drawn store error fails the n-th one (D70)."""

    @cached_property
    def ops(self) -> dict[str, Op]:
        table: dict[str, Op] = {
            "search_catalog": self._search_catalog,
            "lookup_catalog": self._lookup_catalog,
            "get_product": self._get_product,
            "create_cart": self._create_cart,
            "get_cart": self._get_cart,
            "update_cart": self._update_cart,
            "cancel_cart": self._cancel_cart,
            "create_checkout": self._create_checkout,
            "get_checkout": self._get_checkout,
            "update_checkout": self._update_checkout,
            "complete_checkout": self._complete_checkout,
            "cancel_checkout": self._cancel_checkout,
            "get_order": self._get_order,
        }
        return {name: table[name] for name in self.book.names() if name in table}

    def tools(self) -> list[ToolSpec]:
        """The tools this store lists. With a recorded list (D40): the recorded tools the sandbox implements, in the
        recorded order, each exactly as recorded. Without one: the release's tools in binding order, each with the
        release's own input schema, the platform's description and output schema (the release's result schema where
        the platform has none), and the inputs the platform adds (`catalog.filters.available` on Shopify). The
        minor-units paragraph live Shopify ends every description with is on every platform's, because a misread
        price is the one mistake every journey can make."""
        if self.listing is not None:
            return [t for t in self.listing.specs() if t.name in self.ops]
        return [self._documented(t) for t in self.book.specs if t.name in self.ops]

    def _documented(self, spec: ToolSpec) -> ToolSpec:
        doc = self.docs.get(spec.name)
        description = doc.description if doc is not None and doc.description else with_minor_units(spec.description)
        output = doc.output_schema if doc is not None and doc.output_schema is not None else self.outputs.get(spec.name)
        schema = spec.input_schema
        if self.platform.available_filter and doc is not None and doc.input_schema is not None:
            schema = with_available_filter(schema, doc.input_schema)
        return spec.model_copy(
            update={
                "description": description,
                "output_schema": _object_schema(output) if output is not None else None,
                "input_schema": copy.deepcopy(schema),
            }
        )

    def call(self, name: str, arguments: Mapping[str, Any], *, session: str, base_url: str) -> Reply:
        """Run one tool call for `session`. Raises `UnknownTool` for a name the release does not define."""
        op = self.ops.get(name)
        if op is None:
            raise UnknownTool(
                f"unknown tool {name!r}", hint=f"the tools at UCP {self.dialect.version} are {list(self.ops)}"
            )
        call = Call(name, arguments, session, base_url.rstrip("/"))
        card = card_number_field(arguments)
        if card is not None:
            text = "A card number was sent; this store takes payment credentials as tokens only"
            return self._refuse_input(call, [Problem("$", "card_number_refused", text)], code="card_number_refused")
        if self.platform.dialect == "medusa":
            return self._medusa(call, op)
        problems = self.book.check(name, arguments)
        if problems:
            refused = self._refuse_input(call, problems)
            if self.platform.text_refusals:
                text = refusal_text(problems, self.book.payload_key(name))
                return Reply(refused.doc, text=text)
            return refused
        failed = self._store_error(call)
        return failed if failed is not None else op(call)

    def _store_error(self, call: Call) -> Reply | None:
        """The store's own error when the episode's draw fails this call (D70), logged as a refusal; else None."""
        key = (call.session, call.name)
        n = self.calls_made[key] = self.calls_made.get(key, 0) + 1
        kind = self.host.pack.failing_calls().get((call.name, n))
        if kind is None:
            return None
        self.host.reject(
            call.name, Unavailable(code=f"store_{kind}", message=f"the store answered {call.name} call {n}: {kind}")
        )
        if kind == "server_error":
            return Reply(fault=Fault(500, INTERNAL_ERROR, "Internal error", "internal", bare=True))
        said = [error("error", THROTTLED_TEXT, severity="requires_buyer_input")]
        if KINDS[call.name] != "cart":
            return Reply(self.render.error(KINDS[call.name], said, call.base))
        doc = {
            "ucp": self.envelope.block("cart", status="error"),
            "currency": self.host.pack.config().market.currency,
            "id": "",
            "line_items": [],
            "links": self.render.links(call.base),
            "messages": self.render.said(said),
            "totals": [],
        }
        return Reply(doc)

    # -- a platform plugin's own documents -----------------------------------------------------------------------------

    def _medusa(self, call: Call, op: Op) -> Reply:
        """A call as Medusa's UCP plugin answers it (carttrace.sandbox.platforms.medusa, D68): what its validator refuses
        before it runs, a search read its way, then the sandbox's document in the plugin's shape and status. Every
        refusal is logged as one entry, as every other refusal is."""
        key = self.book.payload_key(call.name)
        refused = medusa.refusal(
            call.name,
            call.arguments,
            key,
            version=self.dialect.version,
            market=self.market.code,
            served=medusa.accepted(call.name, self._body_fields(call.name, key), self.capabilities),
            currency=self.market.currency,
        )
        if refused is not None:
            text = str(refused.body.get("message") or _first_content(refused.body))
            self.host.reject(call.name, InvalidCommand(code="refused_by_platform", message=text, field="$"))
            return Reply(refused.body, status=refused.status)
        if call.name == "search_catalog":
            payload = medusa.search_payload(call.payload("catalog"))
            call = Call(call.name, {**call.arguments, "catalog": payload}, call.session, call.base)
        problems = self.book.check(call.name, call.arguments)
        if problems:
            self._refuse_input(call, problems)
            wire = medusa.schema_problem(problems[0].content)
            return Reply(wire.body, status=wire.status)
        reply = op(call)
        if reply.fault is not None or reply.doc is None:
            return reply
        wire = medusa.reshape(call.name, reply.doc, self._medusa_facts(call, reply.doc))
        return Reply(wire.body, status=wire.status, created=wire.status == 201)

    def _body_fields(self, name: str, key: str | None) -> frozenset[str]:
        """The top-level body fields the served release's tool schema lists for `name`."""
        spec = self.book.spec(name)
        props = spec.input_schema.get("properties", {}) if spec is not None and key is not None else {}
        body = props.get(key) if isinstance(props, Mapping) else None
        fields = body.get("properties") if isinstance(body, Mapping) else None
        return frozenset(fields) if isinstance(fields, Mapping) else frozenset()

    def _medusa_facts(self, call: Call, doc: Mapping[str, Any]) -> medusa.Facts:
        """What the plugin's shapes need beyond the document: the resource the document shows, the shopper's
        country, the market's tax."""
        state = self.state
        ident = str(doc.get("id") or "")
        cart = state.carts.get(self.ids.parse_cart(ident, state.carts).core_id or "") if ident else None
        checkout = state.checkouts.get(ident)
        order = state.orders.get(ident)
        country = (checkout.country if checkout is not None else None) or (cart.country if cart is not None else None)
        return medusa.Facts(
            version=self.dialect.version,
            base=call.base,
            tax_bp=self.market.tax_bp,
            tax_on_shipping=self.core_rules.tax_on_shipping,
            country=country,
            store=presented(self.host.pack).domain,
            cart=cart,
            checkout=checkout,
            order=order,
            request=call.payload(self.book.payload_key(call.name) or ""),
        )

    # -- shared steps --------------------------------------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self.host.state

    def _run(self, call: Call, command: Mapping[str, Any]) -> tuple[Outcome, Fault | None]:
        cmd = {**command, "session": call.session}
        key = call.idempotency_key
        if key is not None and cmd["type"] in MUTATING:
            cmd["idempotency_key"] = key
        out = self.host.execute(cmd)
        return out, fault_for(out.error, tier=self.host.tier(call.session))

    def _read(self, call: Call, target: str, ident: str | None) -> Fault | None:
        _, fault = self._run(call, {"type": "read", "target": target, "id": None if ident is None else ident[:500]})
        return fault

    def _refuse_input(self, call: Call, problems: Sequence[Problem], *, code: str = "invalid_argument") -> Reply:
        """Log a call refused before translation and answer with the release's error response."""
        first = problems[0]
        details = tuple({"field": p.path, "message": p.content} for p in problems)
        self.host.reject(call.name, InvalidCommand(code=code, message=first.content, field=first.path, details=details))
        messages = [error(p.code, p.content, path=p.path) for p in problems]
        return Reply(self.render.error(KINDS[call.name], messages, call.base))

    def _not_found(self, call: Call, noun: str, *, path: str = "$.id", malformed: bool = False) -> Reply:
        """The answer for an id that was never issued, was canceled, or belongs to another session. A keyed platform
        names a cart it cannot find `cart_not_found`, and an id not in its shape (a cart gid without its `?key=`)
        `invalid_cart_id`, in its own words and with no path (var/live/delta-01/report.md section 4)."""
        code = f"{noun}_not_found"
        text = f"{noun.capitalize()} not found"
        self.host.reject(call.name, NotFound(code=code, message=text, field=f"{noun}_id"))
        if noun == "cart" and self.ids.keyed:
            shown = (
                ("invalid_cart_id", INVALID_CART_TEXT)
                if malformed
                else ("cart_not_found", "The requested cart does not exist")
            )
            msg = error(shown[0], shown[1], severity="unrecoverable")
        else:
            msg = error("not_found", text, severity="unrecoverable", path=path)
        return Reply(self.render.error(KINDS[call.name], [msg], call.base))

    def _find_cart(self, call: Call, wire_id: str, *, path: str = "$.id") -> Cart | Reply:
        """The session's live cart a wire id names, or the store's answer when there is none."""
        parsed = self.ids.parse_cart(wire_id, self.state.carts)
        cart = self._cart(parsed.core_id, call.session) if parsed.core_id is not None else None
        if cart is None:
            return self._not_found(call, "cart", path=path, malformed=parsed.reading == "malformed")
        return cart

    def _core_lines(self, raw: object) -> tuple[list[dict[str, Any]], tuple[tuple[str | None, str | None], ...]]:
        """Request line items with the platform's line ids read back to the core's."""
        lines, refs = line_inputs(raw)
        if not self.ids.keyed:
            return lines, refs
        for entry in lines:
            if isinstance(entry.get("line_id"), str):
                entry["line_id"] = self.ids.parse_line(entry["line_id"])
        mapped = tuple((self.ids.parse_line(lid) if lid is not None else None, item) for lid, item in refs)
        return lines, mapped

    def _location_notes(self, cart: Cart, context: object) -> list[Message]:
        """A warning per line whose product's market tags exclude the buyer, or any market-limited line when the
        request named no country (live Shopify, store-181 ep-50f972162b39: the line is still added)."""
        if not self.platform.buyer_location_warning:
            return []
        where = country(context) or cart.country
        return [
            warning("product_unavailable_in_buyer_location", LOCATION_TEXT)
            for ln in cart.lines
            if market_excludes(self.catalog.tags(ln.variant_id), where)
        ]

    def _failed(self, call: Call, out: Outcome, notes: Sequence[Message], paths: Paths) -> Reply:
        """No resource came out of a create: an error response with what the core said. When the core refused a
        checkout for having no lines, it dropped the events of the lines it rejected, so each line whose item is not
        in the inventory is named here from the state instead."""
        messages = [*notes, *self._events(out, paths), *self._unknown_items(out, paths)]
        if out.error is not None:
            msg = refusal(out.error)
            messages.append(msg if out.error.get("kind") == "validation" else _as_unrecoverable(msg))
        return Reply(self.render.error(KINDS[call.name], messages or [_generic(call)], call.base))

    def _unknown_items(self, out: Outcome, paths: Paths) -> list[Message]:
        if out.error is None or out.error.get("code") not in ("empty_cart", "merchandise_out_of_stock"):
            return []
        inventory = self.state.inventory
        return [
            error("not_found", "This item is not in the store's catalog", path=f"$.line_items[{i}].item.id")
            for i, (_, item) in enumerate(paths.lines)
            if item is not None and item not in inventory
        ]

    def _events(self, out: Outcome, paths: Paths) -> list[Message]:
        state = self.state
        return event_messages(
            out.messages(),
            paths=paths,
            title=lambda vid: self.catalog.variant_title(state, vid),
            wording=self.wording,
        )

    def _cart(self, cart_id: str, session: str) -> Cart | None:
        cart = self.state.carts.get(cart_id)
        return cart if cart is not None and cart.session == session and cart.status != "canceled" else None

    def _checkout(self, checkout_id: str, session: str) -> Checkout | None:
        ck = self.state.checkouts.get(checkout_id)
        return ck if ck is not None and ck.session == session else None

    def _order(self, order_id: str, session: str) -> Order | None:
        order = self.state.orders.get(order_id)
        return order if order is not None and order.session == session else None

    # -- catalog -------------------------------------------------------------------------------------------------------

    def _search_catalog(self, call: Call) -> Reply:
        req = call.payload("catalog")
        query = text(req.get("query"))
        filters = obj(req.get("filters"))
        if not query.strip() and not filters and self.platform.dialect != "medusa":
            return self._refuse_input(
                call, [Problem("$.query", "missing", "A search needs a query or filters")], code="missing"
            )
        cursor = obj(req.get("pagination")).get("cursor")
        if cursor is not None and not (isinstance(cursor, str) and cursor and self.catalog.known_cursor(req)):
            problem = Problem(
                "$.pagination.cursor", "invalid_cursor", "Use a cursor from an earlier page of this search"
            )
            return self._refuse_input(call, [problem])
        fault = self._read(call, "catalog", query)
        if fault is not None:
            return Reply(fault=fault)
        page = self.catalog.search_page(self.state, req)
        doc: dict[str, Any] = {
            "ucp": self.envelope.block("search"),
            "products": page.products,
            "pagination": page.pagination,
        }
        if page.policies:
            doc["policies"] = page.policies
        doc.update(page.layered)
        context = obj(req.get("context"))
        if (
            self.platform.price_filter_note
            and isinstance(filters.get("price"), Mapping)
            and not context.get("currency")
        ):
            doc["messages"] = [info(PRICE_FILTER_TEXT.format(currency=self.state.currency))]
        return Reply(self._catalog_doc(doc))

    def _lookup_catalog(self, call: Call) -> Reply:
        req = call.payload("catalog")
        ids = [str(i) for i in req.get("ids") or () if isinstance(i, str)]
        if len(dict.fromkeys(ids)) > LOOKUP_LIMIT:
            too_many = batch_too_large(LOOKUP_LIMIT)
            refused = InvalidCommand(code="request_too_large", message=too_many.message, field="ids")
            self.host.reject(call.name, refused)
            return Reply(fault=too_many)
        fault = self._read(call, "product", ",".join(ids))
        if fault is not None:
            return Reply(fault=fault)
        docs, missing = lookup_docs(self.catalog, self.state, ids)
        docs = _price_filtered(docs, req.get("filters"))
        doc: dict[str, Any] = {"ucp": self.envelope.block("lookup"), "products": docs}
        if missing:
            doc["messages"] = [info(_shown(i), code="not_found") for i in missing]
        if self.catalog.has_policies:
            known = self.catalog.index(self.state).products
            targets = [(f"$.products[{i}]", known[d["id"]]) for i, d in enumerate(docs) if d["id"] in known]
            doc["policies"] = self.catalog.policies(targets)
            if not doc["policies"]:
                del doc["policies"]
        return Reply(self._catalog_doc(doc))

    def _get_product(self, call: Call) -> Reply:
        req = call.payload("catalog")
        ident = str(req.get("id") or "")
        fault = self._read(call, "product", ident)
        if fault is not None:
            return Reply(fault=fault)
        hit = self.catalog.find(self.state, ident)
        if hit is None:
            msg = error("not_found", "No product or variant has this id", severity="unrecoverable", path="$.id")
            return Reply(self.render.error("lookup", [msg], call.base))
        product, variant = hit
        detail = self.catalog.detail(req, product, variant)
        kept = _price_filtered([detail], req.get("filters"))
        if not kept:
            msg = error("not_found", "No variant of this product is inside the price filter", severity="unrecoverable")
            return Reply(self.render.error("lookup", [msg], call.base))
        doc: dict[str, Any] = {"ucp": self.envelope.block("lookup"), "product": kept[0]}
        policies = self.catalog.policies([("$.product", product)])
        if policies:
            doc["policies"] = policies
        return Reply(self._catalog_doc(doc))

    def _catalog_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        """A catalog reply as the platform sends it: live Shopify always carries `messages` (an empty list)."""
        if self.platform.messages_always:
            doc["messages"] = self.render.said(doc.get("messages") or [])
        return doc

    # -- carts ---------------------------------------------------------------------------------------------------------

    def _create_cart(self, call: Call) -> Reply:
        req = call.payload("cart")
        lines, refs = self._core_lines(req.get("line_items"))
        codes = discount_codes(req)
        paths = Paths(lines=refs, codes=tuple(codes or ()))
        cmd = {
            "type": "create_cart",
            "lines": lines,
            "codes": codes or [],
            "buyer": buyer(req.get("buyer")),
            "country": country(req.get("context")),
        }
        out, fault = self._run(call, cmd)
        if fault is not None:
            return Reply(fault=fault)
        cart = self._cart(str(out.refs.get("cart_id") or ""), call.session)
        if cart is None:
            return self._failed(call, out, [], paths)
        notes = [*self._events(out, paths), *self._location_notes(cart, req.get("context"))]
        return Reply(self.render.cart(self.state, cart, notes, call.base), created=True)

    def _get_cart(self, call: Call) -> Reply:
        cart = self._find_cart(call, call.target)
        if isinstance(cart, Reply):
            return cart
        fault = self._read(call, "cart", cart.id)
        if fault is not None:
            return Reply(fault=fault)
        return Reply(self.render.cart(self.state, cart, [], call.base))

    def _update_cart(self, call: Call) -> Reply:
        req = call.payload("cart")
        body_id = req.get("id")
        if isinstance(body_id, str) and body_id != call.target:
            problem = Problem("$.id", "invalid_id", "cart.id must name the same cart as the id parameter")
            return self._refuse_input(call, [problem])
        cart = self._find_cart(call, call.target)
        if isinstance(cart, Reply):
            return cart
        refused = self._refused_ids(call, cart, req)
        if refused is not None:
            return refused
        lines, refs = self._core_lines(req.get("line_items"))
        codes = discount_codes(req)
        cmd: dict[str, Any] = {
            "type": "update_cart",
            "cart_id": cart.id,
            "lines": lines,
            "buyer": buyer(req.get("buyer")),
        }
        if codes is not None:
            cmd["codes"] = codes
        out, fault = self._run(call, cmd)
        if fault is not None:
            return Reply(fault=fault)
        paths = Paths(lines=refs, codes=tuple(codes or ()))
        return self._cart_reply(call, out, cart.id, paths, context=req.get("context"))

    def _refused_ids(self, call: Call, cart: Cart, req: Mapping[str, Any]) -> Reply | None:
        """An update naming an id inside its payload the platform does not accept, answered as live Shopify answered
        it, with the cart left as it was and one logged refusal (D46, `SHOPIFY_EVIDENCE`): a delivery group that is
        not a `CartDeliveryGroup` gid is an `invalid_input` error response; a line gid without its `?cart=` part
        returns the cart unchanged with an "Invalid id" error message. None when every id is in the platform's
        shape. The order of the two checks is unobserved."""
        if self.platform.delivery_group_gids:
            bad = _bad_delivery_group(req.get("fulfillment"))
            if bad is not None:
                path, got = bad
                text = DELIVERY_GROUP_TEXT.format(got=got)
                self.host.reject(call.name, InvalidCommand(code="invalid_input", message=text, field=path))
                msg = error("invalid_input", text, severity="unrecoverable", path=path)
                return Reply(self.render.error(KINDS[call.name], [msg], call.base))
        raw = req.get("line_items")
        for i, entry in enumerate(raw if isinstance(raw, list) else ()):
            line_id = entry.get("id") if isinstance(entry, Mapping) else None
            if isinstance(line_id, str) and self.ids.keyless_line(line_id):
                text = f"Invalid id: {line_id}"
                self.host.reject(
                    call.name, InvalidCommand(code="invalid_line_id", message=text, field=f"$.line_items[{i}].id")
                )
                msg = error("error", text, severity="requires_buyer_input")
                return Reply(self.render.cart(self.state, cart, [msg], call.base))
        return None

    def _cancel_cart(self, call: Call) -> Reply:
        cart = self._find_cart(call, call.target)
        if isinstance(cart, Reply):
            return cart
        out, fault = self._run(call, {"type": "cancel_cart", "cart_id": cart.id})
        if fault is not None:
            return Reply(fault=fault)
        return self._cart_reply(call, out, cart.id, Paths())

    def _cart_reply(self, call: Call, out: Outcome, cart_id: str, paths: Paths, *, context: object = None) -> Reply:
        """The cart as the command left it (a canceled cart is returned as it was before deletion)."""
        cart = self.state.carts[cart_id]
        messages = self._events(out, paths)
        if out.error is not None:
            messages.append(refusal(out.error))
        if cart.status != "canceled":
            messages += self._location_notes(cart, context)
        return Reply(self.render.cart(self.state, cart, messages, call.base))

    # -- checkouts -----------------------------------------------------------------------------------------------------

    def _create_checkout(self, call: Call) -> Reply:
        req = call.payload("checkout")
        groups, methods, notes = fulfillment_groups(req, ())
        codes = discount_codes(req)
        source = self._checkout_source(call, req, codes)
        if isinstance(source, Reply):
            return source
        cmd, refs = source
        if groups is not None:
            cmd["fulfillment"] = groups
        paths = Paths(lines=refs, codes=tuple(codes or ()), methods=methods)
        out, fault = self._run(call, cmd)
        if fault is not None:
            return Reply(fault=fault)
        ck = self._checkout(str(out.refs.get("checkout_id") or ""), call.session)
        if ck is None:
            return self._failed(call, out, notes, paths)
        return self._checkout_reply(call, out, ck.id, paths, notes, created=True)

    def _checkout_source(
        self, call: Call, req: Mapping[str, Any], codes: list[str] | None
    ) -> tuple[dict[str, Any], tuple[tuple[str | None, str | None], ...]] | Reply:
        """The create command's source: a cart to convert (its contents win over the payload's), or the payload's
        own lines. An open checkout already made from the cart is answered as it is."""
        cart_id = req.get("cart_id") if isinstance(req.get("cart_id"), str) else None
        if cart_id is None:
            lines, refs = line_inputs(req.get("line_items"))
            if not lines:
                problem = Problem("$.line_items", "missing", "A checkout needs line items, or a cart_id to convert")
                return self._refuse_input(call, [problem], code="missing")
            cmd = {
                "type": "create_checkout",
                "lines": lines,
                "codes": codes,
                "buyer": buyer(req.get("buyer")),
                "country": country(req.get("context")),
            }
            return cmd, refs
        cart = self._find_cart(call, cart_id, path="$.cart_id")
        if isinstance(cart, Reply):
            return cart
        existing = self.state.checkouts.get(cart.checkout_id) if cart.checkout_id else None
        if existing is not None and existing.status in OPEN_CHECKOUT:
            fault = self._read(call, "checkout", existing.id)
            if fault is not None:
                return Reply(fault=fault)
            note = info("This cart already has an open checkout; it is returned instead of a new one")
            return Reply(self.render.checkout(self.state, existing, [note], call.base))
        cmd = {
            "type": "create_checkout",
            "cart_id": cart.id,
            "buyer": buyer(req.get("buyer"), keep_out=cart.buyer),
            "country": cart.country or country(req.get("context")),
        }
        if not cart.codes and codes is not None:
            cmd["codes"] = codes
        return cmd, ()

    def _get_checkout(self, call: Call) -> Reply:
        ck = self._checkout(call.target, call.session)
        if ck is None:
            return self._not_found(call, "checkout")
        fault = self._read(call, "checkout", ck.id)
        if fault is not None:
            return Reply(fault=fault)
        return Reply(self.render.checkout(self.state, ck, [], call.base))

    def _update_checkout(self, call: Call) -> Reply:
        ck = self._checkout(call.target, call.session)
        if ck is None:
            return self._not_found(call, "checkout")
        busy = self._in_progress(call, ck)
        if busy is not None:
            return busy
        req = call.payload("checkout")
        lines, refs = line_inputs(req.get("line_items"))
        groups, methods, notes = fulfillment_groups(req, ck.groups)
        codes = discount_codes(req)
        cmd: dict[str, Any] = {
            "type": "update_checkout",
            "checkout_id": ck.id,
            "lines": lines,
            "buyer": buyer(req.get("buyer")),
        }
        if groups is not None:
            cmd["fulfillment"] = groups
        if codes is not None:
            cmd["codes"] = codes
        out, fault = self._run(call, cmd)
        if fault is not None:
            return Reply(fault=fault)
        return self._checkout_reply(
            call, out, ck.id, Paths(lines=refs, codes=tuple(codes or ()), methods=methods), notes
        )

    def _complete_checkout(self, call: Call) -> Reply:
        ck = self._checkout(call.target, call.session)
        if ck is None:
            return self._not_found(call, "checkout")
        busy = self._in_progress(call, ck)
        if busy is not None:
            return busy
        source = call.arguments if self.dialect.top_level_payment else call.payload("checkout")
        choice = payment(source, self.accepted)
        if choice is None:
            problem = Problem("$.payment.instruments", "missing", "Completing needs a selected payment instrument")
            return self._refuse_input(call, [problem], code="missing")
        invalid = self._shop_pay_instrument(call, source, choice)
        if invalid is not None:
            return invalid
        out, fault = self._run(call, {"type": "complete_checkout", "checkout_id": ck.id, "payment": choice.payment})
        if fault is not None:
            return Reply(fault=fault)
        return self._checkout_reply(call, out, ck.id, Paths(), [], choice=choice)

    def _shop_pay_instrument(self, call: Call, source: Mapping[str, Any], choice: PaymentChoice) -> Reply | None:
        """A Shop Pay instrument checked against the vendored instrument schema of the handler's declared version
        (D73), as the spec's merchant processing has it: the instrument must conform to the Shop Pay instrument
        schema before the token is processed (vendor/dev.shopify.shop_pay/2026-04-08/spec.md). None when the payment
        is not through the store's Shop Pay handler, when the instrument conforms, or when the declared version is
        not vendored (the core's own rules still hold). Every problem is one logged refusal in the platform's
        shapes."""
        namespace = self.core_handlers.get(choice.payment["handler"])
        if namespace != SHOP_PAY_NAMESPACE:
            return None
        entries = self.envelope.handlers.get(SHOP_PAY_NAMESPACE) or []
        version = str(entries[0].get("version")) if entries else ""
        reg = registry()
        if not reg.has(SHOP_PAY_NAMESPACE, version):
            return None
        pay = source.get("payment")
        instruments = pay.get("instruments") if isinstance(pay, Mapping) else None
        instrument = (
            instruments[choice.index] if isinstance(instruments, list) and choice.index < len(instruments) else None
        )
        validator = reg.validator(SHOP_PAY_NAMESPACE, version, "shop_pay.json")
        errors = sorted(validator.iter_errors(instrument), key=lambda e: [str(p) for p in e.absolute_path])
        if not errors:
            return None
        problems = [
            Problem(
                f"$.payment.instruments[{choice.index}]{_json_path(err)}",
                "invalid_instrument",
                str(err.message),
            )
            for err in errors
        ]
        return self._refuse_input(call, problems)

    def _cancel_checkout(self, call: Call) -> Reply:
        ck = self._checkout(call.target, call.session)
        if ck is None:
            return self._not_found(call, "checkout")
        out, fault = self._run(call, {"type": "cancel_checkout", "checkout_id": ck.id})
        if fault is not None:
            return Reply(fault=fault)
        return self._checkout_reply(call, out, ck.id, Paths(), [])

    def _in_progress(self, call: Call, ck: Checkout) -> Reply | None:
        """While a 3-D Secure Action is outstanding the checkout is `complete_in_progress`: a new update or completion
        leaves it unchanged and is answered with the current checkout and a recoverable error."""
        if not self.envelope.actions or not (
            ck.status == "requires_escalation" and ck.escalation == "payment_action_required"
        ):
            return None
        text = "The checkout is completing: process the outstanding 3-D Secure Action, then get the checkout"
        self.host.reject(call.name, IllegalTransition(code="checkout_in_progress", message=text, field="status"))
        msg = error("checkout_in_progress", text, path="$.status")
        return Reply(self.render.checkout(self.state, self.state.checkouts[ck.id], [msg], call.base))

    def _checkout_reply(
        self,
        call: Call,
        out: Outcome,
        checkout_id: str,
        paths: Paths,
        notes: Sequence[Message],
        *,
        created: bool = False,
        choice: PaymentChoice | None = None,
    ) -> Reply:
        messages = [*notes, *self._events(out, paths)]
        err = out.error
        if err is not None and str(err.get("code")) not in STATE_COVERED:
            messages.append(refusal(err, path=_payment_path(err, choice)) if choice else refusal(err))
        ck = self.state.checkouts[checkout_id]
        return Reply(self.render.checkout(self.state, ck, messages, call.base), created=created)

    # -- orders --------------------------------------------------------------------------------------------------------

    def _get_order(self, call: Call) -> Reply:
        order = self._order(call.target, call.session)
        if order is None:
            return self._not_found(call, "order")
        fault = self._read(call, "order", order.id)
        if fault is not None:
            return Reply(fault=fault)
        return Reply(self.render.order(self.state, order, call.base))


INVALID_CART_TEXT = "Invalid id format. Expected a Shopify Cart GID (e.g., 'gid://shopify/Cart/Z2NwLXV...')."
"""Live Shopify's words for a cart id without its key (store-178 ep-475bdc4b5b0f seq 21, 30, 41)."""
DELIVERY_GROUP_TEXT = 'is not a valid CartDeliveryGroup GID (got: "{got}")'
"""Live Shopify's words for a delivery group id outside the gid shape (store-180 ep-da1cd727b196 seq 23)."""
PRICE_FILTER_TEXT = (
    "Price filter applied using shop currency ({currency}). Pass context.currency to specify the intended denomination."
)
"""Live Shopify's words on 14 searches with a price filter (store-310 ep-1362cef036a5 seq 9, among others)."""
LOCATION_TEXT = "The product is unavailable in the current buyer location."
"""Live Shopify's words (store-181 ep-50f972162b39 seq 14)."""


def refusal_text(problems: Sequence[Problem], payload_key: str | None) -> str:
    """Schema problems as live Shopify words them in a text-only refusal: "Missing required arguments: catalog" when
    a top-level argument is missing, else "Invalid arguments: object at `/cart` is missing required properties:
    line_items" (store-302 ep-fae69c1ddff8 seq 19), and the same pointer form for any other problem."""
    top: list[str] = []
    nested: dict[str, list[str]] = {}
    other: list[str] = []
    for p in problems:
        missing = _required_name(p.content) if p.code == "field_required" else None
        segments = _segments(p.path)
        if missing is not None and (not segments or (len(segments) == 1 and segments[0] in TRANSPORT_PARAMS)):
            top.append(missing)
        elif missing is not None:
            parent = _pointer(segments[:-1], payload_key)
            nested.setdefault(parent, []).append(segments[-1])
        else:
            other.append(f"value at `{_pointer(segments, payload_key)}` is invalid: {p.content}")
    if top:
        return "Missing required arguments: " + ", ".join(dict.fromkeys(top))
    parts = [f"object at `{at}` is missing required properties: {', '.join(names)}" for at, names in nested.items()]
    return "Invalid arguments: " + "; ".join([*parts, *other])


def _bad_delivery_group(fulfillment: object) -> tuple[str, str] | None:
    """The path and value of the first delivery group id in an update's `fulfillment` that is not a Shopify
    `CartDeliveryGroup` gid, as the path live Shopify gave it (relative to the cart payload)."""
    methods = fulfillment.get("methods") if isinstance(fulfillment, Mapping) else None
    for i, method in enumerate(methods if isinstance(methods, list) else ()):
        groups = method.get("groups") if isinstance(method, Mapping) else None
        for j, group in enumerate(groups if isinstance(groups, list) else ()):
            ident = group.get("id") if isinstance(group, Mapping) else None
            if ident is not None and not delivery_group_gid(ident):
                return f"$.fulfillment.methods[{i}].groups[{j}].id", str(ident)
    return None


def _required_name(content: str) -> str | None:
    if content.startswith("'") and content.endswith("' is required"):
        return content[1 : -len("' is required")]
    return None


def _segments(path: str) -> list[str]:
    """`$.line_items[0].item.id` as ["line_items", "0", "item", "id"]."""
    out: list[str] = []
    for part in path.removeprefix("$").replace("[", ".").replace("]", "").split("."):
        if part:
            out.append(part)
    return out


def _pointer(segments: Sequence[str], payload_key: str | None) -> str:
    """A JSON Pointer into the tool's arguments; payload paths are relative to the payload, `meta` and `id` not."""
    head = [] if (segments and segments[0] in TRANSPORT_PARAMS) or payload_key is None else [payload_key]
    return "/" + "/".join([*head, *segments])


PAYMENT_FIELDS: Mapping[str, str] = {
    "payment.token": ".credential.token",
    "payment.instrument": ".type",
    "payment.handler": ".handler_id",
}


def _payment_path(err: Mapping[str, Any], choice: PaymentChoice | None) -> str | None:
    field = str(err.get("field") or "")
    if choice is None or field not in PAYMENT_FIELDS:
        return None
    return choice.path(PAYMENT_FIELDS[field])


def _json_path(err: Any) -> str:
    """A jsonschema error's instance path as a JSON path suffix (`.credential.type`, `[0]`)."""
    return "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in err.absolute_path)


def _as_unrecoverable(message: Message) -> Message:
    """In an error response no resource exists to act on (checkout, "Error Handling")."""
    return {**message, "severity": "unrecoverable"} if message.get("type") == "error" else message


def _first_content(body: Mapping[str, Any]) -> str:
    """The first message's words of a plugin's error document."""
    for m in body.get("messages") or ():
        if isinstance(m, Mapping) and m.get("content"):
            return str(m["content"])
    return "refused"


def _generic(call: Call) -> Message:
    return error("not_created", f"{call.name} did not create a resource", severity="unrecoverable")


def _shown(ident: str) -> str:
    """An identifier as a lookup's not-found note shows it, unless it looks like a card number."""
    return "an identifier that looks like a card number" if looks_like_card_number(ident) else ident


def _object_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """An output schema as MCP lists it: `type` is required and is `object` (a bare `oneOf` of objects, as the UCP
    binding's results and live Shopify's checkout result are written, gains it)."""
    out: dict[str, Any] = copy.deepcopy(dict(schema))
    out.setdefault("type", "object")
    return out


def with_minor_units(description: str) -> str:
    """A tool description with the minor-units paragraph appended, once."""
    text = description.rstrip()
    return text if MINOR_UNITS in text else f"{text}\n\n{MINOR_UNITS}\n"


def with_available_filter(schema: Mapping[str, Any], platform_schema: Mapping[str, Any]) -> dict[str, Any]:
    """The binding's input schema with the platform's `catalog.filters.available` added (default true, its words),
    when the binding has a `catalog.filters` object to add it to and the platform lists the field."""
    theirs = _filters(platform_schema)
    available = theirs.get("available") if theirs is not None else None
    out: dict[str, Any] = copy.deepcopy(dict(schema))
    ours = _filters(out)
    if ours is None or not isinstance(available, Mapping) or "available" in ours:
        return out
    ours["available"] = dict(available)
    return out


def _filters(schema: Mapping[str, Any]) -> dict[str, Any] | None:
    """The `properties` object of `catalog.filters` in a tool input schema, when it has one."""
    catalog = (schema.get("properties") or {}).get("catalog") if isinstance(schema.get("properties"), Mapping) else None
    if not isinstance(catalog, Mapping):
        return None
    filters = (
        (catalog.get("properties") or {}).get("filters") if isinstance(catalog.get("properties"), Mapping) else None
    )
    if not isinstance(filters, Mapping):
        return None
    properties = filters.get("properties")
    return properties if isinstance(properties, dict) else None


def _price_filtered(products: list[dict[str, Any]], filters: object) -> list[dict[str, Any]]:
    """Products with only the variants inside `filters.price`, dropping products left with none (catalog lookup,
    "Filters")."""
    price = filters.get("price") if isinstance(filters, Mapping) else None
    if not isinstance(price, Mapping):
        return products
    lo, hi = price.get("min"), price.get("max")

    def inside(v: Mapping[str, Any]) -> bool:
        amount = int(v["price"]["amount"])
        return (not isinstance(lo, int) or amount >= lo) and (not isinstance(hi, int) or amount <= hi)

    out = []
    for p in products:
        variants = [v for v in p["variants"] if inside(v)]
        if variants:
            out.append({**p, "variants": variants})
    return out

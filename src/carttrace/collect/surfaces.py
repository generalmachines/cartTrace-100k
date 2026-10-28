"""What the runner puts between a protocol surface and the agent or the trace: taps, a buffer and two wrappers.

UCP's `meta` (our agent profile, idempotency keys) is not filled here: the live and sandbox surfaces both fill it
through `carttrace.protocols.ucp_meta`, below the agent.

Why each exists:
- `WireBuffer` holds a live store's discovery exchanges until the bundle is open. The header has to name the profile
  the store declared, which discovery is what finds, so the writer opens after it; nothing is lost, because the
  buffered records are written first, in order.
- `WireTap` passes records to the writer and remembers the seq of the last one, so a cart snapshot read back after
  the journey points at the exact wire record it came from (`raw_ref`).
- `ObservedSurface` reports every failed call to the episode's triage. It changes nothing the agent sees.
- `DetailFilter` is the detail_hidden condition: it removes the hidden details from what the agent reads, after the
  surface has recorded the store's full reply on the wire. The judge sees what the store sent; the agent does not.
  When the condition names catalog details it removes exactly those (their entries and their sentences, with
  `carttrace.collect.matchers` and `carttrace.collect.excise`) and reports, per call, how many statements of each it removed: the
  wire says what the store had, and the report says what the agent was kept from, so a store that never stated a
  detail can be told from one whose detail we hid.
- `CatalogOnly` is the catalog an ACP row reads beside its checkout surface: ACP has no catalog operation, so the
  replica's UCP catalog tools are listed with the ACP tools, and nothing else of that surface is.
- `view_filter` and `webmcp_options` are the same detail_hidden condition on a WebMCP storefront, which cuts the detail
  server-side itself, and the storefront's other options (the persona's 3-D Secure answer, limits, seed).
- `ReadBack` is how the runner reads the cart, the checkout and the order back on each surface kind, into the same
  snapshot shapes the judge reads for UCP.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from carttrace.agents.dialects import Dialect, dialect_for_surface
from carttrace.collect.excise import excise
from carttrace.collect.matchers import HiddenSet
from carttrace.collect.ports import WireSink
from carttrace.core.canon import dumps
from carttrace.core.errors import ValidationError
from carttrace.protocols.tools import Mode, SurfaceKind, ToolResult, ToolSpec, ToolSurface
from carttrace.trace.carts import from_page_cart, from_ucp
from carttrace.trace.schema import Cart
from carttrace.trace.writer import TraceWriter

CATALOG_TOOLS = frozenset({"search_catalog", "lookup_catalog", "get_product"})
"""UCP tools whose replies carry product details."""
POLICY_TOOLS = frozenset({"search_shop_policies_and_faqs"})
"""Store MCP tools that answer policy and FAQ questions."""
DETAIL_KEYS: Mapping[str, frozenset[str]] = {
    "description": frozenset({"description", "description_html", "descriptionHtml", "body_html"}),
    "attributes": frozenset({"attributes"}),
    "metafields": frozenset({"metafields", "details"}),
}
"""What each hidden detail removes from a product: every key of these names, at any depth of a catalog reply.
Metafields are a store's structured product data; a sandbox store built from a catalog carries the same data as UCP
`metadata.details` (size chart, allergens, care...), so hiding metafields hides those too, or the agent would still
read the detail and the experiment would measure nothing."""
NO_POLICY = "No matching policy or FAQ answer was found."
"""What a policy lookup answers while policies are hidden: what a store that publishes none would say."""
ENTRY_KEYS = frozenset({"attributes", "metafields", "details"})
"""Product data keyed by name: a store's attributes and metafields, and a catalog store's `metadata.details`. An
entry goes when its name names a hidden detail; every other entry stays as it is."""
JSON_STYLES: tuple[tuple[bool, tuple[str, str], bool], ...] = tuple(
    (sort_keys, separators, ascii_only)
    for sort_keys in (True, False)
    for separators in ((",", ":"), (", ", ": "))
    for ascii_only in (False, True)
)
"""(sort_keys, separators, ensure_ascii) of the JSON layouts a reply's text is recognised in and written back in."""

FailureHook = Callable[[str, dict[str, Any], ToolResult, int], Awaitable[None]]
"""(tool, arguments, failed result, how many times this call has now failed)."""
HiddenHook = Callable[[str, Mapping[str, int]], None]
"""(tool, how many statements of each hidden detail the filter removed from its reply)."""
Rewrite = Callable[[Any, HiddenSet, Counter[str]], Any]


class WireBuffer:
    """A WireSink that keeps records until a writer can take them."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def wire(self, record: Mapping[str, Any]) -> dict[str, Any]:
        copied = json.loads(dumps(dict(record)))
        self.records.append(copied)
        return dict(copied)

    def replay(self, sink: WireSink) -> None:
        """Write every buffered record to `sink`, in order, once."""
        records, self.records = self.records, []
        for record in records:
            sink.wire(record)


class WireTap:
    """A WireSink in front of a TraceWriter that remembers the seq of the last record written."""

    def __init__(self, writer: TraceWriter) -> None:
        self.writer = writer
        self.last_seq: int | None = None
        self.written: list[Any] = []
        """Every WireRecord written through the tap, in order (a website selection's proof names the GET's seq)."""

    def wire(self, record: Mapping[str, Any]) -> Any:
        written = self.writer.wire(record)
        self.last_seq = written.seq
        self.written.append(written)
        return written


class _Delegate:
    """Forwards the ToolSurface properties and lifecycle to the wrapped surface."""

    def __init__(self, inner: ToolSurface) -> None:
        self.inner = inner

    @property
    def kind(self) -> SurfaceKind:
        return self.inner.kind

    @property
    def mode(self) -> Mode:
        return self.inner.mode

    @property
    def version(self) -> str:
        return self.inner.version

    async def list_tools(self) -> list[ToolSpec]:
        return await self.inner.list_tools()

    async def aclose(self) -> None:
        await self.inner.aclose()


class ObservedSurface(_Delegate):
    """Calls go through unchanged; each failed one is reported with how often that same call has failed."""

    def __init__(self, inner: ToolSurface, on_failure: FailureHook) -> None:
        super().__init__(inner)
        self.on_failure = on_failure
        self._failures: dict[str, int] = {}

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = await self.inner.call(name, arguments)
        if result.is_error:
            key = dumps([name, arguments]) if _is_json(arguments) else name
            self._failures[key] = self._failures.get(key, 0) + 1
            await self.on_failure(name, arguments, result, self._failures[key])
        return result


class DetailFilter(_Delegate):
    """The detail_hidden condition: hidden details never reach the agent.

    With `details` (catalog detail names) the filter hides exactly those: in catalog replies their entries under
    attributes, metafields and details and their sentences, lines and tables in descriptions, and in policy answers
    their statements when `hide` names policies. Each reply it changed is reported to `on_hidden`. Without details it
    hides the locations in `hide` whole, as detail_hidden did before hypotheses named details."""

    def __init__(
        self,
        inner: ToolSurface,
        hide: Iterable[str],
        *,
        details: Iterable[str] = (),
        on_hidden: HiddenHook | None = None,
    ) -> None:
        super().__init__(inner)
        self.hide = frozenset(hide)
        names = tuple(details)
        self.details = HiddenSet(names) if names else None
        self.keys = frozenset().union(*(DETAIL_KEYS.get(name, frozenset()) for name in self.hide))
        self.on_hidden = on_hidden

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = await self.inner.call(name, arguments)
        if self.details is not None:
            return result if result.is_error else self._without_details(name, result, self.details)
        if name in POLICY_TOOLS and "policies" in self.hide and not result.is_error:
            return result.model_copy(update={"content": [{"type": "text", "text": NO_POLICY}], "structured": None})
        if name in CATALOG_TOOLS and self.keys and not result.is_error:
            return result.model_copy(
                update={
                    "content": _strip_content(result.content, self.keys),
                    "structured": strip(result.structured, self.keys),
                }
            )
        return result

    def _without_details(self, name: str, result: ToolResult, hidden: HiddenSet) -> ToolResult:
        rewrite: Rewrite
        if name in CATALOG_TOOLS:
            rewrite = without_details
        elif name in POLICY_TOOLS and "policies" in self.hide:
            rewrite = _text_without
        else:
            return result
        content, in_content = _content_without(result.content, rewrite, hidden)
        in_structured: Counter[str] = Counter()
        structured = None if result.structured is None else rewrite(result.structured, hidden, in_structured)
        counts = in_content | in_structured  # two copies of one reply: the larger count of each detail, not the sum
        if not counts:
            return result
        if self.on_hidden is not None:
            self.on_hidden(name, dict(sorted(counts.items())))
        return result.model_copy(update={"content": content, "structured": structured})


def without_details(value: Any, hidden: HiddenSet, counts: Counter[str]) -> Any:
    """A catalog reply without the hidden details: their entries under attributes, metafields and details, and their
    statements in descriptions, at any depth. A mapping or list the removal empties is dropped too, as a store that
    never had the detail sends none. `counts` gains the statements removed, per detail."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in DETAIL_KEYS["description"]:
                new = _description_without(item, hidden, counts, as_html=None if key == "description" else True)
            elif key in ENTRY_KEYS:
                new = _entries_without(item, hidden, counts)
            else:
                new = without_details(item, hidden, counts)
            if isinstance(item, Mapping | list) and item and not new:
                continue
            out[key] = new
        return out
    if isinstance(value, list):
        return [without_details(item, hidden, counts) for item in value]
    return value


def _description_without(item: Any, hidden: HiddenSet, counts: Counter[str], *, as_html: bool | None) -> Any:
    """A description, as text or as a {plain, html, markdown} object, read the way carttrace.agents reads each form; the
    forms repeat one text, so each detail counts once."""
    if isinstance(item, str):
        text, found = excise(item, hidden, as_html=as_html)
        counts.update(found)
        return text
    if not isinstance(item, Mapping):
        return item
    forms: dict[str, Any] = {}
    most: Counter[str] = Counter()
    for form, text in item.items():
        if isinstance(text, str):
            forms[form], found = excise(text, hidden, as_html=form == "html")
            most |= found
        else:
            forms[form] = text
    counts.update(most)
    return forms


def _entries_without(item: Any, hidden: HiddenSet, counts: Counter[str]) -> Any:
    """Attributes, metafields or details without the entries whose name names a hidden detail: a mapping of name to
    value, or a list of {key or name, value} entries as Shopify sends metafields."""
    if isinstance(item, Mapping):
        kept: dict[str, Any] = {}
        for key, value in item.items():
            named = hidden.keyed(str(key))
            counts.update(named)
            if not named:
                kept[key] = value
        return kept
    if isinstance(item, list):
        entries: list[Any] = []
        for entry in item:
            label = _entry_name(entry)
            named = hidden.keyed(label) if label is not None else frozenset()
            counts.update(named)
            if not named:
                entries.append(entry)
        return entries
    return item


def _entry_name(entry: Any) -> str | None:
    """The name of one entry of a list of metafields or attributes: its "key", else its "name"."""
    if isinstance(entry, Mapping):
        for field in ("key", "name"):
            if isinstance(entry.get(field), str):
                return str(entry[field])
    return None


def _text_without(value: Any, hidden: HiddenSet, counts: Counter[str]) -> Any:
    """A policy answer without the statements of hidden details, in every string it holds."""
    if isinstance(value, str):
        text, found = excise(value, hidden)
        counts.update(found)
        return text
    if isinstance(value, Mapping):
        return {key: _text_without(item, hidden, counts) for key, item in value.items()}
    if isinstance(value, list):
        return [_text_without(item, hidden, counts) for item in value]
    return value


def _content_without(content: Any, rewrite: Rewrite, hidden: HiddenSet) -> tuple[Any, Counter[str]]:
    """MCP content without the hidden details, as `_strip_content` strips it: inside JSON text, whether it is the
    content itself or a text block, and in any other value. Text that is not JSON is cut as text."""
    counts: Counter[str] = Counter()
    if isinstance(content, str):
        return _json_text_without(content, rewrite, hidden, counts), counts
    if not isinstance(content, list):
        return rewrite(content, hidden, counts), counts
    out: list[Any] = []
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "text" and isinstance(block.get("text"), str):
            out.append({**block, "text": _json_text_without(block["text"], rewrite, hidden, counts)})
        else:
            out.append(rewrite(block, hidden, counts))
    return out, counts


def _json_text_without(text: str, rewrite: Rewrite, hidden: HiddenSet, counts: Counter[str]) -> str:
    """JSON text rewritten inside the JSON and written back in the layout it came in, or the very same string when
    nothing was hidden, so the two arms of a pair read the same bytes wherever the detail was not."""
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict | list):
        cut, in_text = excise(text, hidden)
        counts.update(in_text)
        return cut
    found: Counter[str] = Counter()
    rewritten = rewrite(parsed, hidden, found)
    if not found:
        return text
    counts.update(found)
    for sort_keys, separators, ascii_only in JSON_STYLES:
        if json.dumps(parsed, sort_keys=sort_keys, separators=separators, ensure_ascii=ascii_only) == text:
            return json.dumps(rewritten, sort_keys=sort_keys, separators=separators, ensure_ascii=ascii_only)
    return json.dumps(rewritten, separators=(",", ":"), ensure_ascii=False)


def strip(value: Any, keys: frozenset[str]) -> Any:
    """`value` without any mapping key in `keys`, at any depth. Other values are returned as they are."""
    if isinstance(value, Mapping):
        return {k: strip(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, list):
        return [strip(item, keys) for item in value]
    return value


def _strip_content(content: Any, keys: frozenset[str]) -> Any:
    """MCP content blocks whose text is JSON are stripped inside the JSON, and so is content that is itself a JSON
    string (the sandbox surface's `content` is the store's document as text); anything else is stripped as a value.

    Seen 2026-10-11: the first detail_hidden run stripped `structured` but passed the JSON text in `content` through,
    and the protocol agent, which reads `content`, saw every hidden detail."""
    if isinstance(content, str):
        return _strip_json_text(content, keys)
    if not isinstance(content, list):
        return strip(content, keys)
    out: list[Any] = []
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "text" and isinstance(block.get("text"), str):
            out.append({**block, "text": _strip_json_text(block["text"], keys)})
        else:
            out.append(strip(block, keys))
    return out


def _strip_json_text(text: str, keys: frozenset[str]) -> str:
    """JSON text with the keys removed inside it; text that is not a JSON object or array comes back unchanged."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return text
    if not isinstance(parsed, dict | list):
        return text
    return json.dumps(strip(parsed, keys), ensure_ascii=False)


def _is_json(value: Any) -> bool:
    try:
        dumps(value)
    except (TypeError, ValueError):
        return False
    return True


# -- a catalog beside a checkout protocol ------------------------------------------------------------------------------

CATALOG_ONLY = CATALOG_TOOLS | POLICY_TOOLS
"""What a catalog companion offers: the store's catalog and policy tools, and nothing that writes."""
COMPANIONS: Mapping[str, str] = {"acp_rest": "ucp_rest", "mpp": "ucp_rest", "ap2": "ucp_rest"}
"""Surface kinds that pay for a checkout but read no catalog, to the UCP binding the store's catalog is read through
beside them. ACP publishes products in a feed, not in an operation; in the sandbox the replica's own UCP catalog is
that feed. WebMCP's storefront has its own catalog pages."""


def companion_kind(kind: str, profile: Any) -> str | None:
    """The catalog companion a row on `kind` needs, when its store serves UCP; None when the kind has its own catalog
    or the store has none to read."""
    found = COMPANIONS.get(kind)
    return found if found is not None and isinstance(getattr(profile, "ucp", None), str) and profile.ucp else None


class CatalogOnly(_Delegate):
    """A UCP surface that lists its catalog tools only: the companion an ACP row reads the store's products through.
    Every other tool is refused here and never sent, so the cart, checkout and payment go through the row's own
    surface, which is what the row measures."""

    async def list_tools(self) -> list[ToolSpec]:
        return [t for t in await self.inner.list_tools() if t.name in CATALOG_ONLY]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in CATALOG_ONLY:
            return ToolResult(
                content=f"{name} is not offered here: this surface reads the catalog only; the checkout goes through "
                "the store's checkout surface",
                is_error=True,
                error_code="unknown_tool",
            )
        return await self.inner.call(name, arguments)


class WithoutTools(_Delegate):
    """A surface that does not offer some of its tools: on the MPP rail (D66) the store's UCP checkout is paid over
    MPP's HTTP 402, so UCP's own complete_checkout is neither listed nor sent."""

    def __init__(self, inner: ToolSurface, hidden: Iterable[str]) -> None:
        super().__init__(inner)
        self.hidden = frozenset(hidden)

    async def list_tools(self) -> list[ToolSpec]:
        return [t for t in await self.inner.list_tools() if t.name not in self.hidden]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name in self.hidden:
            return ToolResult(
                content=f"{name} is not offered here: this store takes payment over its payment rail",
                is_error=True,
                error_code="unknown_tool",
            )
        return await self.inner.call(name, arguments)


# -- the detail_hidden condition on a storefront ----------------------------------------------------------------------

ViewFilter = Callable[[Any, Counter[str]], Any]
"""(catalog document, counts) to the document without the hidden details, as `WebMcpOptions.detail_filter` takes
it: the storefront applies it to every document before a page or a tool answer is built, for the row's session."""


def view_filter(hide: Iterable[str], details: Iterable[str] = ()) -> ViewFilter | None:
    """The detail_hidden condition as a storefront's own filter: the same cut `DetailFilter` makes on a UCP reply
    (named details with their counts, or whole locations), applied server-side so no copy of a page shows the detail.
    A row on a storefront is not wrapped in `DetailFilter` as well: it would count every statement twice. None when
    the condition hides nothing a catalog document holds."""
    names = tuple(details)
    locations = frozenset(hide)
    if names:
        hidden = HiddenSet(names)
        return lambda doc, counts: without_details(doc, hidden, counts)
    keys = frozenset().union(*(DETAIL_KEYS.get(name, frozenset()) for name in locations))
    if keys:
        return lambda doc, counts: strip(doc, keys)
    return None


def webmcp_options(
    *,
    challenge: str,
    hide: Iterable[str],
    details: Iterable[str],
    on_hidden: HiddenHook,
    limits: Any,
    seed: int,
) -> dict[str, Any]:
    """The webmcp kind's options for one row (`WebMcpOptions` by name): how the persona answers the bank's 3-D Secure
    page (`challenge`: approve or decline), the condition's detail filter and the hook it reports to, the browser
    limits from config, and the episode's seed, so the same row records the same bytes."""
    options: dict[str, Any] = {"challenge": challenge, "limits": limits, "seed": seed}
    cut = view_filter(hide, details)
    if cut is not None:
        options["detail_filter"] = cut
        options["on_hidden"] = on_hidden
    return options


def challenge_policy(three_ds: Any) -> str:
    """A persona's 3-D Secure behaviour (complete or abandon) as the sandbox browser's challenge policy."""
    return "decline" if three_ds == "abandon" else "approve"


# -- read-backs --------------------------------------------------------------------------------------------------------

Call = tuple[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ReadBack:
    """The runner's read-backs on one surface kind: which call reads the cart, the checkout and the order back from the
    store after the agent ends, and how each reply becomes the StateSnapshot the judge reads.

    The snapshots take the same shapes on every kind, and their sources name the read's role (`get_cart`,
    `get_checkout`, `get_order`), as the judge's cart reads name them; the wire record each snapshot points at names
    the surface's own tool. UCP reads a cart, a checkout and an order by id. ACP has no cart and no order tool: the
    session is read by id, its lines are the cart before payment and the checkout after, and the order it carries is
    the order. A WebMCP storefront's `get_cart` and `get_checkout` take no id (the page has one of each open), and once
    the order is placed `get_checkout` shows the order page, which stands for the completed checkout and the order."""

    kind: str
    dialect: Dialect
    """carttrace.agents' dialect for the kind: how its documents are read."""

    @classmethod
    def for_kind(cls, kind: str, *, ucp: str | None = None, ap2: str | None = None) -> ReadBack:
        """The read-backs on `kind`; a UCP binding served at a checkout-only release (`ucp` 2026-01-11 or 2026-01-23,
        D71) reads the checkout as the cart and its order from it, as that release has no cart or order tool, and AP2
        at v0.1.0 reads its cart by id (D71)."""
        from carttrace.sandbox.surfaces.ucp.dialect import DIALECTS

        if kind == "ap2" and ap2 == "v0.1.0":
            from carttrace.agents.dialects.ap2 import Ap2V1Dialect

            return cls(kind, Ap2V1Dialect())
        served = DIALECTS.get(ucp or "")
        if kind in ("ucp_mcp", "ucp_rest") and served is not None and served.checkout_only:
            from carttrace.agents.dialects.ucp_checkout import UcpCheckoutDialect

            return cls(kind, UcpCheckoutDialect())
        return cls(kind, dialect_for_surface(kind))

    def cart_call(self, cart_id: str) -> Call:
        return self.dialect.read, self.dialect.read_arguments(cart_id)

    def cart_gone(self, payload: Any) -> bool:
        """The store shows no cart any more, and that is not a failure: a storefront starts a new, empty cart once
        its checkout became an order, and the checkout and the order carry the lines."""
        return (
            self.kind == "webmcp"
            and isinstance(payload, Mapping)
            and payload.get("cart") is None
            and "structuredContent" not in payload  # a live page's own get_cart (D67) answers its cart, empty or not
        )

    def cart_of(self, payload: Any) -> Cart | None:
        if self.kind in ("acp_rest", "ap2"):
            doc = self.dialect.checkout_doc(payload)
            return None if doc is None else self.dialect.view_of(doc)
        if not isinstance(payload, Mapping):
            return None
        try:
            return from_ucp(payload)
        except ValidationError:
            if self.kind != "webmcp":
                return None
        try:
            return from_page_cart(payload)
        except ValidationError:
            return None

    def refusal_of(self, payload: Any) -> str | None:
        """The code of the error a cart reply carries in place of a cart, when the store answered without marking the
        call failed: a UCP error message (Shopify's `invalid_cart_id`, `cart_not_found`), or the error a storefront
        page shows. A read that comes back so is a failed read, named for the store's reason."""
        if not isinstance(payload, Mapping):
            return None
        error = payload.get("error")
        if isinstance(error, Mapping) and error.get("code"):
            return str(error["code"])
        for message in payload.get("messages") or ():
            if isinstance(message, Mapping) and message.get("type") == "error" and message.get("code"):
                return str(message["code"])
        return None

    def checkout_call(self, checkout_id: str) -> Call:
        return self.dialect.get_checkout, self.dialect.id_arguments(checkout_id)

    def checkout_of(self, payload: Any) -> Mapping[str, Any] | None:
        return self.dialect.checkout_doc(payload)

    def checkout_view(self, doc: Mapping[str, Any]) -> Cart | None:
        return self.dialect.view_of(doc)

    def order_in(self, doc: Mapping[str, Any]) -> str | None:
        return self.dialect.order_of(doc)

    def order_call(self, order_id: str) -> Call | None:
        """The call that reads the order by id, or None when the kind has no order tool and the order comes with the
        checkout reply (`order_from_checkout`)."""
        if self.dialect.get_order is None:
            return None
        return self.dialect.get_order, self.dialect.id_arguments(order_id)

    def order_of(self, payload: Any) -> Cart | None:
        """The order an order call answered, as a cart."""
        doc = self.dialect.order_doc(payload)
        return None if doc is None else self.dialect.order_view(doc)

    def order_from_checkout(self, payload: Any, doc: Mapping[str, Any]) -> Cart | None:
        """The order a checkout reply carries, as a cart: ACP's session holds the order's facts in its own lines; a
        storefront's reply is the order page."""
        if self.kind == "webmcp":
            return self.order_of(payload)
        return self.dialect.order_view(doc)

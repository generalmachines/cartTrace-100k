"""The catalog as UCP shows it: products from the envpack snapshot, priced and stocked from the core's current state.

Why two sources. Titles, descriptions and options never change during an episode and live in the pack
(`snapshot.products`, `snapshot.variants`); prices, stock and availability can (a `reprice` or `stockout` fault, a
completed order), and live only in the state. Reading both means a search after a fault shows the faulted world, and a
reset shows the pack's world again.

What a product holds. Product facts are whatever the capture or the catalog source recorded, so they are read
tolerantly: `title`, `description` (text or a `{plain, html, markdown}` object), `handle`, `url`, `media` or `images`,
`tags` and `categories`. A product with no facts is named after its first variant. Text a fault injected
(`injected_text`, a prompt injection test) is appended to the product's description, as a compromised store would
serve it.

Search ranks by query words found in the title (2 points each) and elsewhere in the product's text (1 point), then by
product id, so the same query over the same world always pages the same way. On a platform whose search is the
store's own (`storefront_search`, live Shopify) a query the store was never asked ranks closer to how it would: whole
words, plurals folded to their singular, any query word enough to list a product, each word weighted by how rare it
is in the store (idf) times the best field it is found in (title 3, product type 2, vendor, tags and variant titles
and option values 1, description 0.5), then by the product's standing in the store's own recorded answers (what it
listed often and early: a best-seller proxy), then by product id (D39). Chosen on half of 2,033 distinct live
first-page searches of 43 Shopify stores (2026-10-12 and 13) and measured on the other half: top-10 overlap with live
0.34 for the word count, 0.36 for this ranking, 0.48 with the first half replayed from the search memory
(carttrace.sandbox.catalogs.search_memory). Matching every query word, as the storefront's search page does, scored 0.26,
and leaving the description out 0.35, so the catalog search is neither.

Under the WooCommerce platform pack (D42, `all_words`) a search lists what WooCommerce's lists: every query word, as
written, a substring of the title or description ('black sandal' 0, 'slippers' 0, 'slipper' 4 on the reference
store), and a store served under another platform than its own replays none of its recorded search answers, which are
its own platform's ranking. Under the Magento-like pack (`facets`) a search answers with layered navigation counts
and automatic price ranges beside the products.

Lookup and get_product follow the UCP catalog lookup capability: one featured variant per product with `inputs`
correlation for lookups, and option selection with relaxation and availability signals for get_product.

Replicas of real stores (decision D12). A product built from a store capture (`carttrace.sandbox.catalogs.replica`) holds
the store's own document as its facts, its variant documents under `variants`. Such a product keeps the store's order
of variants, options and option values, and every field the store sent (list prices, SKUs, barcodes, media sizes, a
variant's own description and metadata, fields UCP does not name), with its availability wording and price range while
the state agrees with them; prices and availability still come from the state, so a fault or an order changes them.
The policies the store attached to a product (`_policies`) are served at the response root with `applies_to` targets.
A store shows less in a search listing than in get_product (often no metadata, a few variants, sometimes shorter text),
and an agent that reads a listing must not learn more on a replica than it would on the store, so each view keeps the
store's own shape (`_listing`, `_detail`): the fields it sent, its values where they differ from the full document,
and the most variants a listing showed.
The snapshot's `recorded` answers replay what the store answered during the capture, because a store's ranking,
listing and selection cannot be recomputed from text: a search the store was asked returns its products, variants and
pagination in its order; the same search paged another way is cut from the store's ranking (followed by this
module's own hits); a get_product the store was asked keeps its selection and variant list, and the store's own
availability signals for option values whose variants the capture never saw. Anything never asked falls back to the
rules above.
"""

from __future__ import annotations

import base64
import binascii
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from carttrace.core.canon import dumps, loads
from carttrace.core.money import exponent
from carttrace.sandbox.core.state import State
from carttrace.sandbox.envpack.doc import AgentViewDoc, RecordedAnswer, RecordedProduct
from carttrace.sandbox.envpack.pack import EnvPack
from carttrace.sandbox.platforms import Platform, platform_named
from carttrace.sandbox.platforms.magento import facets, price_ranges
from carttrace.sandbox.platforms.medusa import phrase_matched
from carttrace.sandbox.presented import presented
from carttrace.sandbox.views import description_text, handle_for
from carttrace.trace.products import price_of

_WORD = re.compile(r"[a-z0-9]+")
DEFAULT_PAGE = 10
MAX_PAGE = 50
LOOKUP_LIMIT = 50
"""Identifiers per lookup_catalog call; more is refused (-32602 over MCP, 400 over REST)."""
POLICIES = "_policies"
"""Where a replica's product facts keep the policies the store attached to the product in its catalog replies."""
DERIVED = frozenset({"id", "title", "description", "price_range", "variants", "options", "selected"})
"""Fields of a store's product document that are rendered from the state and the snapshot, never copied."""
LISTING = "_listing"
DETAIL = "_detail"
"""Where a replica's product facts keep the shape the store gave a product in search listings and in get_product:
`fields` and `variant_fields` (the keys it sent), `values` and `variant_values` (what it sent where that differs
from the full document) and, for listings, `max_variants`. A key a shape leaves out restricts nothing: the platform's
view (`Catalog.views`) fills it, else the document keeps what it has."""
LOOKUP = "_lookup"
"""The lookup_catalog view, which no capture records per product: served in the platform's lookup shape."""
VIEW_NAMES = {LISTING: "listing", DETAIL: "detail", LOOKUP: "lookup"}
"""A view's key in a product's facts, to its name in a platform's `product_views`."""
RENDER = "render"
"""In a view shape the catalog serves a stand-in with (D100): the recorded view's format, the stand-in's domain and
each product's taxonomy category, which `_viewed` writes the document's values from."""
ADDED = "_added"
"""Where a detail_added overlay (carttrace.sandbox.catalogs.overlay) names the fields it placed on a product, per view name:
served in that view whatever its shape, since the overlay is the experiment's one difference."""
SHAPE_KEYS = (
    "fields",
    "variant_fields",
    "media_fields",
    "max_media",
    "variant_media_fields",
    "option_value_fields",
    "category_values",
)
PRODUCT_REQUIRED = frozenset({"id", "title", "description", "price_range", "variants", "selected"})
VARIANT_REQUIRED = frozenset({"id", "title", "description", "price", "inputs"})
"""What UCP requires of a product and a variant (and the per-response members), kept whatever shape a view has."""
STATE_FIELDS = frozenset({"id", "variants", "options", "selected", "price", "availability"})
"""Fields a view's recorded values never replace: they come from the state or the request."""


@dataclass(frozen=True, slots=True)
class VariantView:
    id: str
    product_id: str
    title: str
    price: int
    currency: str
    available: bool
    """For sale and in stock now."""
    options: Mapping[str, str]
    document: Mapping[str, Any] | None = None
    """The store's own variant document, for a replica's product; None for any other product."""


@dataclass(frozen=True, slots=True)
class ProductView:
    id: str
    title: str
    description: Mapping[str, str]
    facts: Mapping[str, Any]
    variants: tuple[VariantView, ...]
    option_names: tuple[str, ...] = field(default=())

    @property
    def featured(self) -> VariantView:
        """The representative variant: the first one for sale, else the first one."""
        return next((v for v in self.variants if v.available), self.variants[0])

    @property
    def captured(self) -> bool:
        """The facts are a store's own document (a replica's product)."""
        return is_store_document(self.facts)

    def text(self) -> str:
        """Everything search matches besides the title: description, variants, options, tags, brand, categories."""
        parts = [*self.description.values(), *(v.title for v in self.variants)]
        parts += [label for v in self.variants for label in v.options.values()]
        parts += [str(t) for t in self.facts.get("tags") or () if isinstance(t, str)]
        parts += [str(self.facts[k]) for k in ("brand", "product_type") if isinstance(self.facts.get(k), str)]
        parts += [c["value"] for c in _categories(self.facts)]
        return " ".join([self.title, *parts]).lower()


class Catalog:
    """Products of one pack, read against one state."""

    def __init__(
        self,
        pack: EnvPack,
        *,
        opaque_cursors: bool = False,
        available_filter: bool = False,
        storefront_search: bool = False,
        all_words: bool = False,
        recorded_search: bool = True,
        total_count: bool = False,
        facets: bool = False,
        views: Mapping[str, Mapping[str, Any]] | None = None,
        phrase: bool = False,
        prefix_words: bool = False,
        field_prefix: bool = False,
        whole_words: bool = False,
        page_default: int | None = None,
        page_max: int | None = None,
    ) -> None:
        """`opaque_cursors` hands out base64 cursors as live Shopify does instead of decimal offsets;
        `available_filter` honours `filters.available` (default true: only sale-ready products and variants);
        `storefront_search` ranks a search no recorded answer covers as live Shopify's catalog search would;
        `all_words` matches as WooCommerce's search does (every word, as a substring, no stemming, D42);
        `recorded_search` replays the store's recorded search answers (off for a store served under another
        platform's pack, whose search is that platform's); `total_count` adds the hit count to pagination
        (WooCommerce's UCP plugin); `facets` answers with Magento's layered navigation counts and price ranges;
        `views` is the platform's product shape per view (`Platform.product_views`), in which a replica's product
        is served where the store's own recorded shape says nothing (D55); `phrase` matches as Medusa's UCP plugin
        does (the whole query inside a title, description or handle, D68), and `page_default` and `page_max` are the
        platform's page sizes where they differ from the sandbox's."""
        self.views: Mapping[str, Mapping[str, Any]] = views or {}
        self.phrase = phrase
        self.prefix_words = prefix_words
        self.field_prefix = field_prefix
        self.whole_words = whole_words
        self.page_default = page_default or DEFAULT_PAGE
        self.page_max = page_max or MAX_PAGE
        self.opaque_cursors = opaque_cursors
        self.available_filter = available_filter
        self.storefront_search = storefront_search
        self.all_words = all_words
        self.total_count = total_count
        self.facets = facets
        snap = pack.snapshot
        self._products: Mapping[str, Mapping[str, Any]] = snap.products
        self._variants = snap.variants
        self.currency = snap.currency
        self._documents: dict[str, tuple[int, Mapping[str, Any]]] = {}
        """A replica's variant id to its place in the store's order and its document."""
        for facts in self._products.values():
            if is_store_document(facts):
                for i, doc in enumerate(facts["variants"]):
                    if isinstance(doc, Mapping) and isinstance(doc.get("id"), str):
                        self._documents.setdefault(doc["id"], (i, doc))
        self._searches: dict[str, RecordedAnswer] = {}
        self._details: dict[str, RecordedAnswer] = {}
        pages: dict[str, list[RecordedAnswer]] = {}
        for answer in snap.recorded:
            if answer.tool == "search_catalog":
                if not recorded_search:
                    continue
                self._searches.setdefault(dumps(answer.request), answer)
                pages.setdefault(_ranking_key(answer.request), []).append(answer)
            else:
                self._details.setdefault(dumps(answer.request), answer)
        self._rankings = {key: ranking_of(found) for key, found in pages.items()}
        self._standing = standing(self._searches.values())
        """How high the store's own recorded answers listed each product: its default order, as near as the
        sandbox can know it, for `storefront_ranked` to break ties by."""
        self.has_policies = any(facts.get(POLICIES) for facts in self._products.values())
        """Some product carries a store's policies (a replica's); otherwise no response lists any."""
        shapes = [facts.get(DETAIL) for facts in self._products.values() if is_store_document(facts)]
        fields = [s.get("option_value_fields") for s in shapes if isinstance(s, Mapping)]
        self._detail_value_fields: frozenset[str] | None = (
            frozenset(str(f) for found in fields if isinstance(found, list) for f in found) if fields else None
        )
        """The option value fields the store's get_product replies carried, across every product it was asked about:
        the shape a replica gives a product its capture never read in detail. Live Shopify sends labels only (store-307
        get_product seq 14 and 17 on 2026-10-12), so a replica must not invent `available` and `exists` for one."""

    def products(self, state: State) -> list[ProductView]:
        """Every product, ordered by product id, with variants ordered by variant id (a replica's in the store's
        order)."""
        by_product: dict[str, list[VariantView]] = {}
        for vid in sorted(self._variants):
            sv = self._variants[vid]
            live = state.inventory.get(vid)
            if live is None:
                continue
            pid = sv.product_id or vid
            placed = self._documents.get(vid)
            view = VariantView(
                vid,
                pid,
                sv.title or vid,
                live.price,
                live.currency,
                live.sellable > 0,
                dict(sv.options),
                placed[1] if placed is not None else None,
            )
            by_product.setdefault(pid, []).append(view)
        return [self._product(pid, self._ordered(vs), state) for pid, vs in sorted(by_product.items())]

    def _ordered(self, variants: list[VariantView]) -> tuple[VariantView, ...]:
        """A replica product's variants in the store's order; any other product's as given (by id)."""
        if not any(v.document is not None for v in variants):
            return tuple(variants)
        last = len(variants) + len(self._documents)
        return tuple(sorted(variants, key=lambda v: (self._documents.get(v.id, (last, {}))[0], v.id)))

    def _product(self, pid: str, variants: tuple[VariantView, ...], state: State) -> ProductView:
        facts = dict(self._products.get(pid) or {})
        title = str(facts.get("title") or variants[0].title)
        description = _description(facts.get("description"), title)
        overlay = state.overlays.get(pid)
        if overlay:
            description = {k: f"{v} {overlay}" for k, v in description.items()}
        names: list[str] = []
        if is_store_document(facts):
            for option in facts.get("options") or ():
                name = option.get("name") if isinstance(option, Mapping) else None
                if isinstance(name, str) and name and name not in names:
                    names.append(name)
        for v in variants:
            names += [n for n in v.options if n not in names]
        return ProductView(pid, title, description, facts, variants, tuple(names))

    def variant_title(self, state: State, variant_id: str) -> str:
        """A line item's title: the product's title, with the variant's when it says more."""
        sv = self._variants.get(variant_id)
        if sv is None:
            return variant_id
        facts = self._products.get(sv.product_id) or {}
        product = str(facts.get("title") or "")
        variant = sv.title
        if product and variant and variant != "Default Title" and not variant.startswith(product):
            return f"{product} - {variant}"
        return variant if variant and variant != "Default Title" else product or variant_id

    def tags(self, variant_id: str) -> list[str]:
        """The tags of a variant's product, as the store sent them (market limits such as `NO_Australia` among them)."""
        sv = self._variants.get(variant_id)
        facts = self._products.get(sv.product_id) if sv is not None else None
        return [t for t in (facts or {}).get("tags") or () if isinstance(t, str)]

    def image(self, variant_id: str) -> str | None:
        sv = self._variants.get(variant_id)
        facts = self._products.get(sv.product_id) if sv is not None else None
        media = _media(facts or {})
        return str(media[0]["url"]) if media else None

    # -- search --------------------------------------------------------------------------------------------------------

    def search(  # noqa: PLR0911 - one return per platform rule
        self, state: State, query: str, *, price: Mapping[str, Any] | None = None, categories: Sequence[str] = ()
    ) -> list[ProductView]:
        if self.storefront_search:
            return storefront_ranked(self._filtered(state, price, categories), query, self._standing)
        if self.phrase:
            return phrase_matched(self._filtered(state, price, categories), query)
        if self.prefix_words:
            return prefix_words_matched(self._filtered(state, price, categories), query)
        if self.field_prefix:
            return field_prefix_matched(self._filtered(state, price, categories), query)
        if self.whole_words:
            return whole_words_matched(self._filtered(state, price, categories), query)
        if self.all_words:
            return all_words_matched(self._filtered(state, price, categories), query)
        words = _WORD.findall(query.lower())
        hits: list[tuple[int, ProductView]] = []
        for p in self._filtered(state, price, categories):
            score = _score(p, words)
            if words and score == 0:
                continue
            hits.append((score, p))
        hits.sort(key=lambda h: (-h[0], h[1].id))
        return [p for _, p in hits]

    def _filtered(self, state: State, price: Mapping[str, Any] | None, categories: Sequence[str]) -> list[ProductView]:
        """Every product inside the price filter (with only its variants inside it) and the category filter."""
        out: list[ProductView] = []
        for product in self.products(state):
            p = _price_filtered(product, price)
            if p is not None and not (categories and not _in_categories(p, categories)):
                out.append(p)
        return out

    def known_cursor(self, request: Mapping[str, Any]) -> bool:
        """Whether a search request's cursor is one this store hands out: none, an offset from one of its own pages,
        or the store's own cursor for a page of a search a replica recorded."""
        req = search_request(request)
        cursor = req["cursor"]
        if cursor is None or cursor.isdigit() or _opaque_offset(cursor) is not None:
            return True
        ranking = self._rankings.get(_ranking_key(req))
        return ranking is not None and cursor in ranking.offsets

    def search_page(self, state: State, request: Mapping[str, Any]) -> SearchPage:
        """One page of a search whose cursor `known_cursor` accepts: the store's recorded answer when it was asked
        exactly this, a page of its recorded ranking when it was asked this search paged another way, else this
        module's own ranking."""
        req = search_request(request)
        index = self.index(state)
        exact = self._searches.get(dumps(req))
        if exact is not None:
            pagination = dict(exact.pagination) if exact.pagination is not None else {"has_next_page": False}
            return self._page(_listed(index, exact.products), pagination)
        raw_filters = request.get("filters")
        filters: Mapping[str, Any] = raw_filters if isinstance(raw_filters, Mapping) else {}
        price = filters.get("price") if isinstance(filters.get("price"), Mapping) else None
        categories = [str(c) for c in filters.get("categories") or () if isinstance(c, str)]
        own = self.search(state, str(request.get("query") or ""), price=price or None, categories=categories)
        if self.available_filter and filters.get("available", True) is not False:
            own = [p for p in (_sellable(p) for p in own) if p is not None]
        limit = min(int(req["limit"] or self.page_default), self.page_max)
        cursor = req["cursor"]
        ranking = self._rankings.get(_ranking_key(req))
        entries: list[tuple[ProductView, tuple[VariantView, ...]]]
        cursors: Mapping[int, str] = {}
        if ranking is None:
            entries = [(p, _listing_variants(p, self.view(LISTING))) for p in own]
            offset = _offset(cursor)
        else:
            recorded = _listed(index, ranking.hits)
            shown = {p.id for p, _ in recorded}
            entries = recorded + [(p, _listing_variants(p, self.view(LISTING))) for p in own if p.id not in shown]
            offset = ranking.offsets[cursor] if cursor in ranking.offsets else _offset(cursor)
            cursors = ranking.cursors
        page = entries[offset : offset + limit]
        more = offset + limit < len(entries)
        nxt = cursors.get(offset + limit) or self._cursor(offset + limit, limit)
        paging: dict[str, Any] = {"has_next_page": more, **({"cursor": nxt} if more else {})}
        if self.total_count:
            paging["total_count"] = len(entries)
        out = self._page(page, paging)
        if self.facets:
            out.layered.update(layered(entries, self.currency))
        return out

    def _cursor(self, offset: int, limit: int) -> str:
        """The cursor of the page starting at `offset`: a decimal offset, or on a platform with opaque cursors a
        base64 token naming the offset and the page, as live Shopify's wrap a GraphQL cursor and a page number."""
        if not self.opaque_cursors:
            return str(offset)
        raw = dumps({"offset": offset, "page": offset // max(1, limit) + 1}).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    def detail(self, request: Mapping[str, Any], product: ProductView, variant: VariantView | None) -> dict[str, Any]:
        """A get_product product: the store's recorded answer when it was asked exactly this, else `detail_doc`."""
        req = detail_request(request)
        recorded = self._details.get(dumps(req))
        doc = detail_doc(
            product, variant, req["selected"], req["preferences"], recorded=recorded, view=self.view(DETAIL)
        )
        keep = self._detail_value_fields
        own = product.facts.get(DETAIL)
        view = self.view(DETAIL) or {}
        unshaped = product.captured and not (isinstance(own, Mapping) and "fields" in own)
        if unshaped and keep is not None and isinstance(doc.get("options"), list):
            doc["options"] = [_option_values(o, set(keep) | {"label"}) for o in doc["options"]]
        unshaped = unshaped or (not product.captured and isinstance(view.get(RENDER), Mapping))
        bare = view.get("unselected_option_value_fields")
        if unshaped and not req["selected"] and bare is not None and isinstance(doc.get("options"), list):
            doc["options"] = [_option_values(o, {str(f) for f in bare} | {"label"}) for o in doc["options"]]
        return doc

    def view(self, key: str) -> Mapping[str, Any] | None:
        """The platform's shape for view `key` (LISTING, DETAIL, LOOKUP), or None when it has none."""
        return self.views.get(VIEW_NAMES[key])

    def listing(self, product: ProductView) -> dict[str, Any]:
        """The document a search listing shows for `product`, as `search_page` lists it when no recorded answer picks
        its variants. The benchmark's parity reader compares it with a live listing of the same product (D55)."""
        return self._page([(product, _listing_variants(product, self.view(LISTING)))], {}).products[0]

    def policies(self, targets: Sequence[tuple[str, ProductView]]) -> list[dict[str, Any]]:
        """The policies a replica's store attached to these products, for the response root: one entry per policy,
        its `applies_to` naming every target it covers (`$.products[2]`, `$.product`). Empty for any other store."""
        by_policy: dict[str, dict[str, Any]] = {}
        for path, product in targets:
            for policy in product.facts.get(POLICIES) or ():
                if isinstance(policy, Mapping):
                    entry = by_policy.setdefault(dumps(policy), {**policy, "applies_to": []})
                    if path not in entry["applies_to"]:
                        entry["applies_to"].append(path)
        return list(by_policy.values())

    def _page(
        self, entries: Sequence[tuple[ProductView, tuple[VariantView, ...]]], pagination: dict[str, Any]
    ) -> SearchPage:
        docs = [
            shaped(
                product_doc(p, [variant_doc(v, option_names=p.option_names) for v in vs]),
                p,
                LISTING,
                self.view(LISTING),
            )
            for p, vs in entries
        ]
        policies = self.policies([(f"$.products[{i}]", p) for i, (p, _) in enumerate(entries)])
        return SearchPage(docs, pagination, policies)

    def index(self, state: State) -> Index:
        """Every product and variant by id, built once for a call that resolves several identifiers."""
        return Index(self.products(state))

    def find(self, state: State, ident: str) -> tuple[ProductView, VariantView | None] | None:
        """The product an id names: (product, variant) for a variant id, (product, None) for a product id."""
        return self.index(state).find(ident)


class Index:
    def __init__(self, products: Sequence[ProductView]) -> None:
        self.products = {p.id: p for p in products}
        self.variants = {v.id: (p, v) for p in products for v in p.variants}

    def find(self, ident: str) -> tuple[ProductView, VariantView | None] | None:
        if ident in self.products:
            return self.products[ident], None
        hit = self.variants.get(ident)
        return (hit[0], hit[1]) if hit is not None else None


@dataclass(frozen=True, slots=True)
class SearchPage:
    products: list[dict[str, Any]]
    pagination: dict[str, Any]
    policies: list[dict[str, Any]]
    layered: dict[str, Any] = field(default_factory=dict)
    """Magento's layered navigation over every hit (`facets`, `price_ranges`); empty on other platforms."""


@dataclass(frozen=True, slots=True)
class Ranking:
    """A store's results for one recorded search (query and filters), across the pages its capture read."""

    hits: tuple[RecordedProduct, ...]
    """In the store's order, each product once, with the variants the store listed for it."""
    offsets: Mapping[str, int]
    """A store cursor to the offset of the page it asks for."""
    cursors: Mapping[int, str]
    """An offset to the store cursor that asks for the page starting there."""


def search_request(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """A search as a replica matches it: the query in lowercase with single spaces, the filters, the page size and
    the cursor. The context (the shopper's market) is left out: a replica sells in the one market it was built for."""
    raw_pagination = catalog.get("pagination")
    pagination: Mapping[str, Any] = raw_pagination if isinstance(raw_pagination, Mapping) else {}
    raw_filters = catalog.get("filters")
    limit = pagination.get("limit")
    cursor = pagination.get("cursor")
    return {
        "query": " ".join(str(catalog.get("query") or "").split()).casefold(),
        "filters": dict(raw_filters) if isinstance(raw_filters, Mapping) else {},
        "limit": limit if isinstance(limit, int) and not isinstance(limit, bool) else None,
        "cursor": cursor if isinstance(cursor, str) and cursor else None,
    }


def detail_request(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """A get_product as a replica matches it: the id, the selections and preferences in order, and the filters."""
    raw_filters = catalog.get("filters")
    selected = [
        {"name": s["name"], "label": s["label"]}
        for s in catalog.get("selected") or ()
        if isinstance(s, Mapping) and isinstance(s.get("name"), str) and isinstance(s.get("label"), str)
    ]
    return {
        "id": str(catalog.get("id") or ""),
        "selected": selected,
        "preferences": [str(p) for p in catalog.get("preferences") or () if isinstance(p, str)],
        "filters": dict(raw_filters) if isinstance(raw_filters, Mapping) else {},
    }


def _opaque_offset(cursor: str) -> int | None:
    """The offset an opaque cursor this module handed out names, or None for any other text."""
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        value = loads(raw)
    except (ValueError, binascii.Error):
        return None
    offset = value.get("offset") if isinstance(value, Mapping) else None
    return offset if isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0 else None


def _offset(cursor: str | None) -> int:
    if cursor is None:
        return 0
    if cursor.isdigit():
        return int(cursor)
    return _opaque_offset(cursor) or 0


def _sellable(p: ProductView) -> ProductView | None:
    """The product with only its variants for sale, or None when none is (`filters.available` true)."""
    kept = tuple(v for v in p.variants if v.available)
    if not kept:
        return None
    return (
        p if len(kept) == len(p.variants) else ProductView(p.id, p.title, p.description, p.facts, kept, p.option_names)
    )


def _ranking_key(request: Mapping[str, Any]) -> str:
    return dumps({"query": request.get("query"), "filters": request.get("filters")})


def ranking_of(pages: Sequence[RecordedAnswer]) -> Ranking:
    """A store's ranking for one search from its recorded pages: from the first page (no cursor) of each page size,
    follow the store's next cursors; the chain with the most products wins, ties to the smaller page size."""
    best: Ranking | None = None
    best_key: tuple[int, int] | None = None
    for limit in sorted({_limit(p) for p in pages}):
        group = {p.request.get("cursor"): p for p in pages if _limit(p) == limit}
        hits: list[RecordedProduct] = []
        seen: set[str] = set()
        offsets: dict[str, int] = {}
        cursors: dict[int, str] = {}
        page = group.get(None)
        while page is not None:
            hits += [h for h in page.products if h.id not in seen]
            seen |= {h.id for h in page.products}
            pagination = page.pagination or {}
            nxt = pagination.get("cursor") if pagination.get("has_next_page") is True else None
            if not isinstance(nxt, str) or not nxt or nxt in offsets:
                break
            offsets[nxt] = len(hits)
            cursors[len(hits)] = nxt
            page = group.get(nxt)
        key = (-len(hits), limit)
        if best_key is None or key < best_key:
            best, best_key = Ranking(tuple(hits), offsets, cursors), key
    return best or Ranking((), {}, {})


def _limit(answer: RecordedAnswer) -> int:
    limit = answer.request.get("limit")
    return limit if isinstance(limit, int) and not isinstance(limit, bool) else 0


def _listed(index: Index, hits: Sequence[RecordedProduct]) -> list[tuple[ProductView, tuple[VariantView, ...]]]:
    """Recorded products as they stand in the state, each with the variants the store listed (all of them, featured
    first, when none of those is in the replica). A product the replica lacks is left out."""
    out: list[tuple[ProductView, tuple[VariantView, ...]]] = []
    for hit in hits:
        product = index.products.get(hit.id)
        if product is None:
            continue
        by_id = {v.id: v for v in product.variants}
        listed = tuple(by_id[i] for i in hit.variants if i in by_id)
        out.append((product, listed or _featured_first(product)))
    return out


def _featured_first(p: ProductView) -> tuple[VariantView, ...]:
    first = p.featured
    return (first, *[v for v in p.variants if v.id != first.id])


def _listing_variants(p: ProductView, platform: Mapping[str, Any] | None = None) -> tuple[VariantView, ...]:
    """A listing's variants: featured first, at most as many as the store's listings of a replica's product showed,
    else as many as the platform's listings show (`platform`, live Shopify ten, D55)."""
    variants = _featured_first(p)
    if not p.captured:
        return variants
    shape = p.facts.get(LISTING)
    cap = shape.get("max_variants") if isinstance(shape, Mapping) else None
    if cap is None and platform is not None:
        cap = platform.get("max_variants")
    return variants[:cap] if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0 else variants


def shaped(doc: dict[str, Any], p: ProductView, view: str, platform: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A replica product's document in the shape the store gave `view` (LISTING, DETAIL or LOOKUP): only the fields
    it sent there, with its own values where they differed from the full document. A view the store never showed
    this product in (no recorded `fields`; a shape may still hold the view's own values) is served in the platform's
    view (`platform`, D55); with neither, the document keeps what it has. A field a detail_added overlay placed is
    kept. A stand-in's product (not a replica's) is served in the recorded agent view its pack names (`platform`
    carrying RENDER, D100), else as it is."""
    if not p.captured:
        render = platform.get(RENDER) if platform else None
        return _viewed(doc, p, view, platform or {}, render) if isinstance(render, Mapping) else doc
    own = p.facts.get(view)
    mine: Mapping[str, Any] = own if isinstance(own, Mapping) else {}
    if not mine and not platform:
        return doc
    fill = {} if "fields" in mine else (platform or {})
    shape = {key: mine[key] if key in mine else fill.get(key) for key in SHAPE_KEYS}
    added = p.facts.get(ADDED)
    extra = {str(f) for f in added.get(VIEW_NAMES[view]) or ()} if isinstance(added, Mapping) else set()
    values = {k: v for k, v in (mine.get("values") or {}).items() if _replaceable(str(k), p)}
    return view_shaped(doc, shape, values=values, variant_values=mine.get("variant_values") or {}, extra=extra)


def _viewed(
    doc: Mapping[str, Any], p: ProductView, view: str, shape: Mapping[str, Any], render: Mapping[str, Any]
) -> dict[str, Any]:
    """A stand-in's product in a recorded agent view (D100, carttrace.sandbox.views): the view's field names and nesting,
    values written in its format, and nothing the open record does not state. The description's html member holds
    the open description (and the folded facts) as text, a handle and url sit on the stand-in's own domain, the
    category is the Shopify taxonomy id mapped from our category and the product type, the compare-at range is the
    format's empty one, and media only come from a source licensed to show them. No SKU, compare-at price, review
    or alt text is made up: those paths stay absent."""
    fmt = render["format"]
    out = dict(doc)
    text = _view_text(p, fmt)
    description = {"html": text}
    out["description"] = description
    handle = handle_for(p.title, p.id)
    out["handle"] = handle
    out["url"] = str(fmt["url"]).format(domain=render["domain"], handle=handle)
    ranged = out["price_range"]
    if fmt["list_price_range"] == "zero":
        out["list_price_range"] = {end: price(0, ranged[end]["currency"]) for end in ("min", "max")}
    else:
        out["list_price_range"] = {end: dict(ranged[end]) for end in ("min", "max")}
    category = (render.get("categories") or {}).get(p.id)
    if isinstance(category, str) and category:
        out["categories"] = [{"taxonomy": fmt["category_taxonomy"], "value": f"{fmt['category_prefix']}{category}"}]
    else:
        out.pop("categories", None)
    source = (p.facts.get("provenance") or {}).get("source") if isinstance(p.facts.get("provenance"), Mapping) else None
    media = _media(p.facts) if source in (fmt.get("media_sources") or ()) else []
    if media:
        out["media"] = media
    else:
        out.pop("media", None)
    default = fmt.get("default_option") if not p.option_names else None
    if isinstance(default, Mapping):
        name, label = str(default["name"]), str(default["label"])
        signals = {"available": any(v.available for v in p.variants), "exists": True} if view == DETAIL else {}
        out["options"] = [{"name": name, "values": [{"label": label, **signals}]}]
        if "selected" in out:
            out["selected"] = [{"name": name, "label": label}]
    kept = [str(f) for f in fmt.get("availability_fields") or ()]
    variants = []
    for v in out.get("variants") or ():
        nv = {k: value for k, value in v.items() if k not in ("availability", "description", "media")}
        availability = v.get("availability") if isinstance(v.get("availability"), Mapping) else {}
        nv["availability"] = {k: availability[k] for k in kept if k in availability}
        nv["description"] = dict(description)
        if isinstance(default, Mapping) and not v.get("options"):
            nv["options"] = [{"name": str(default["name"]), "label": str(default["label"])}]
            nv["title"] = str(default["label"])
        if media:
            nv["media"] = [{"type": m["type"], "url": m["url"]} for m in media[:1]]
        variants.append(nv)
    most = shape.get("max_variants") if view == LISTING else None
    if isinstance(most, int) and not isinstance(most, bool) and most > 0:
        variants = variants[:most]
    out["variants"] = variants
    return view_shaped(out, shape)


def _view_text(p: ProductView, fmt: Mapping[str, Any]) -> str:
    """The description a view shows: the open record's own (empty when it has none, never the title), or the text a
    fault put in its place, with the format's folded facts."""
    raw = p.facts.get("description")
    own = raw if isinstance(raw, str) else ""
    if dict(p.description) != _description(raw, p.title):
        own = p.description.get("plain") or next(iter(p.description.values()), "")
    return description_text(own, p.facts, [str(f) for f in fmt.get("fold") or ()])


def view_shaped(
    doc: Mapping[str, Any],
    shape: Mapping[str, Any],
    *,
    values: Mapping[str, Any] | None = None,
    variant_values: Mapping[str, Any] | None = None,
    extra: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """A product document cut to one view's `shape` (SHAPE_KEYS; a key the shape lacks restricts nothing), with a
    view's own `values` and `variant_values` in place of the document's and the `extra` fields kept whatever the
    shape. `shaped` serves a replica's product with it; replica fidelity cuts the capture's truth with it, so the
    replica is scored on what the store's agent view shows (D55)."""
    named = shape.get("fields")
    fields = set(doc) if named is None else {str(f) for f in named} | PRODUCT_REQUIRED | set(extra)
    out = {k: v for k, v in doc.items() if k in fields}
    for key, value in (values or {}).items():
        if key in fields and key not in STATE_FIELDS:
            out[key] = value
    if isinstance(out.get("media"), list):
        out["media"] = _shown_media(out["media"], shape.get("media_fields"), shape.get("max_media"))
    if isinstance(out.get("categories"), list) and shape.get("category_values") is not None:
        _keep_categories(out, str(shape["category_values"]))
    out["variants"] = _shaped_variants(doc.get("variants") or (), shape, variant_values or {})
    value_fields = shape.get("option_value_fields")
    if isinstance(out.get("options"), list) and value_fields is not None:
        keep = {str(f) for f in value_fields} | {"label"}
        out["options"] = [_option_values(option, keep) for option in out["options"]]
    return out


def _keep_categories(out: dict[str, Any], pattern: str) -> None:
    """Only the categories whose value the platform's agent view lists (live Shopify: the taxonomy category)."""
    kinds = re.compile(pattern)
    shown = [c for c in out["categories"] if isinstance(c, Mapping) and kinds.search(str(c.get("value") or ""))]
    if shown:
        out["categories"] = shown
    else:
        del out["categories"]


def _shaped_variants(
    variants: Iterable[Mapping[str, Any]], shape: Mapping[str, Any], variant_values: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Each variant with the fields the view shows (all when the shape names none), the view's own values where the
    store sent them, and its media as the view shows them."""
    named = shape.get("variant_fields")
    keep = None if named is None else {str(f) for f in named} | VARIANT_REQUIRED
    out = []
    for variant in variants:
        kept = {k: v for k, v in variant.items() if keep is None or k in keep}
        for key, value in (variant_values.get(str(variant.get("id"))) or {}).items():
            if (keep is None or key in keep) and key not in STATE_FIELDS:
                kept[key] = value
        if isinstance(kept.get("media"), list) and shape.get("variant_media_fields") is not None:
            kept["media"] = _shown_media(kept["media"], shape["variant_media_fields"], None)
        out.append(kept)
    return out


def _shown_media(media: list[Any], fields: Any, most: Any) -> list[Any]:
    """Media items as a view shows them: at most `most` (None: all), each with only `fields` (None: all)."""
    items = media[:most] if isinstance(most, int) and not isinstance(most, bool) and most >= 0 else media
    if fields is None:
        return list(items)
    keep = {str(f) for f in fields}
    return [{k: v for k, v in m.items() if k in keep} if isinstance(m, Mapping) else m for m in items]


def _option_values(option: Any, keep: set[str]) -> Any:
    """An option with only the value fields the store sent: a store that leaves `available` and `exists` unset (live
    Shopify does on get_product) gets an answer that leaves them unset too."""
    if not isinstance(option, Mapping):
        return option
    values = [
        {k: v for k, v in value.items() if k in keep} if isinstance(value, Mapping) else value
        for value in option.get("values") or ()
    ]
    return {**option, "values": values}


def _replaceable(key: str, p: ProductView) -> bool:
    """A view's own value may stand in for a field only while the state has not changed what it describes: the price
    range while no price moved, the description while no fault injected text into it."""
    if key == "price_range":
        return _prices_unchanged(p)
    if key == "description":
        return dict(p.description) == _description(p.facts.get("description"), p.title)
    return True


def _prices_unchanged(p: ProductView) -> bool:
    return all(v.document is not None and price_of(v.document.get("price"))[0] == v.price for v in p.variants)


def is_store_document(facts: Mapping[str, Any]) -> bool:
    """Product facts that are a store's own document, as a replica keeps them: their variants are listed in them."""
    return isinstance(facts.get("variants"), list)


def _description(raw: object, title: str) -> dict[str, str]:
    if isinstance(raw, str) and raw.strip():
        return {"plain": raw}
    if isinstance(raw, Mapping):
        kept = {k: str(raw[k]) for k in ("plain", "html", "markdown") if isinstance(raw.get(k), str) and raw[k]}
        if kept:
            return kept
    return {"plain": title}


def _media(facts: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in facts.get("media") or ():
        if isinstance(m, Mapping) and isinstance(m.get("url"), str):
            item: dict[str, Any] = {"type": str(m.get("type") or "image"), "url": m["url"]}
            if isinstance(m.get("alt_text"), str):
                item["alt_text"] = m["alt_text"]
            out.append(item)
    for img in facts.get("images") or ():
        url = img if isinstance(img, str) else img.get("url") if isinstance(img, Mapping) else None
        if isinstance(url, str):
            out.append({"type": "image", "url": url})
    return out


FIELD_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("title", 3.0),
    ("product_type", 2.0),
    ("vendor", 1.0),
    ("tags", 1.0),
    ("variants", 1.0),
    ("description", 0.5),
)
"""How much a query word found in each field counts in `storefront_ranked`, before its rarity; a word counts once,
at its best field."""
_TOKEN = re.compile(r"[^\W_]+")


def all_words_matched(products: Sequence[ProductView], query: str) -> list[ProductView]:
    """`products` WooCommerce's search would list for `query` (D42): every word of the query, case-folded, found as a
    substring of the product's title or description, with no stemming and no typo or synonym tolerance ('black
    sandal' finds 0, 'slippers' 0 where 'slipper' finds 4: ops/playgrounds/woocommerce/captures/diff-woo 26, 23, 24).
    Variant titles and option values are not searched, as WordPress searches a product's title, excerpt and content
    only. WooCommerce lists matches newest first; the sandbox has no publication dates, so matches keep the catalog's
    product order (by id). An empty query lists every product."""
    words = [w for w in query.casefold().split() if w]
    out: list[ProductView] = []
    for p in products:
        plain = p.description.get("plain")
        body = plain if plain is not None else " ".join(p.description.values())
        text = f"{p.title} {body}".casefold()
        if all(w in text for w in words):
            out.append(p)
    return out


def prefix_words_matched(products: Sequence[ProductView], query: str) -> list[ProductView]:
    """`products` a word-index search lists for `query` (PrestaShop's Search::find, D68): every query word must begin
    some word the product's index holds (its title, description, variants, option values, tags, brand and
    categories), so 'slipp' finds a slipper and 'slippers' does not, and an option value such as 'black' counts. A
    word no product's index begins is dropped from the query rather than failing it (Search.php: "If nothing was
    found after X retries, skip this keyword"); a query left with no word finds nothing. Matches keep the catalog's
    product order; an empty query lists every product."""
    words = [w for w in _TOKEN.findall(query.casefold()) if w]
    if not words:
        return list(products)
    indexes = [(p, set(_TOKEN.findall(p.text().casefold()))) for p in products]
    kept = [w for w in words if any(t.startswith(w) for _, index in indexes for t in index)]
    if not kept:
        return []
    return [p for p, index in indexes if all(any(t.startswith(w) for t in index) for w in kept)]


def whole_words_matched(products: Sequence[ProductView], query: str) -> list[ProductView]:
    """`products` Saleor's search lists for `query` (a websearch query over the `simple` text search configuration,
    D68): every word, case-folded, must be a whole word of the product's name, description, a variant's name or an
    option value; no stemming, no prefix ('slippers' finds 0 where 'slipper' finds 4, 'black sandal' finds the sandals
    with a Black variant: ops/playgrounds/saleor/captures/storefront 05, 06, 08). Matches keep the catalog's product
    order; an empty query lists every product."""
    words = _TOKEN.findall(query.casefold())
    out: list[ProductView] = []
    for p in products:
        plain = p.description.get("plain")
        parts = [p.title, plain if plain is not None else " ".join(p.description.values())]
        parts += [v.title for v in p.variants] + [label for v in p.variants for label in v.options.values()]
        index = set(_TOKEN.findall(" ".join(parts).casefold()))
        if all(w in index for w in words):
            out.append(p)
    return out


def stem(word: str) -> str:
    """A word as Postgres's English text search keeps it, as near as a few suffix rules come to its Snowball
    stemmer: plurals and -ing and -ed endings dropped ('slippers' and 'slipper' are one lexeme)."""
    w = word.casefold()
    for suffix, keep in (("sses", "ss"), ("ies", "i"), ("ing", ""), ("ed", ""), ("s", "")):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3 and not (suffix == "s" and w.endswith("ss")):
            return w[: len(w) - len(suffix)] + keep
    return w


def field_prefix_matched(products: Sequence[ProductView], query: str, *, min_length: int = 2) -> list[ProductView]:
    """`products` Vendure's default search lists for `query` (its Postgres strategy, D68): every word, stemmed, as a
    prefix (`'word':*` joined with `&`), all inside one field: the product's name, a variant's name, or the
    description; a query of `min_length` characters or fewer filters nothing. So 'black sandal' finds a sandal whose
    variant is named 'Black' only if 'sandal' is in that variant's name too, and 'slippers' finds a slipper."""
    if len(query.strip()) <= min_length:
        return list(products)
    terms = [stem(w) for w in _TOKEN.findall(query.casefold())]
    out: list[ProductView] = []
    for p in products:
        plain = p.description.get("plain")
        fields = [
            p.title,
            *(v.title for v in p.variants),
            plain if plain is not None else " ".join(p.description.values()),
        ]
        for text in fields:
            lexemes = {stem(t) for t in _TOKEN.findall(str(text).casefold())}
            if all(any(lx.startswith(term) for lx in lexemes) for term in terms):
                out.append(p)
                break
    return out


def layered(entries: Sequence[tuple[ProductView, tuple[VariantView, ...]]], currency: str) -> dict[str, Any]:
    """Magento's layered navigation over a search's hits (carttrace.sandbox.platforms.magento): facet counts per option
    value over the sellable variants, and automatic price ranges over each product's lowest sellable price."""
    prices = [
        min(v.price for v in p.variants if v.available) for p, _ in entries if any(v.available for v in p.variants)
    ]
    return {"facets": facets(_once_per_product(entries)), "price_ranges": price_ranges(prices, exponent(currency))}


def _once_per_product(entries: Sequence[tuple[ProductView, tuple[VariantView, ...]]]) -> list[dict[str, str]]:
    """One {option: value} row per product and distinct sellable option value, so a product counts once per value."""
    rows: list[dict[str, str]] = []
    for p, _ in entries:
        pairs = dict.fromkeys((n, label) for v in p.variants if v.available for n, label in v.options.items())
        rows += [{n: label} for n, label in pairs]
    return rows


def storefront_ranked(
    products: Sequence[ProductView], query: str, standing: Mapping[str, float] | None = None
) -> list[ProductView]:
    """`products` ranked for `query` as live Shopify's catalog search ranks, approximately (module docstring): every
    product holding any query word, by the sum over query words of idf times the best field's weight, then by the
    product's `standing` in the store's own answers (higher first), then by id. An empty query lists every product
    by standing, then id."""
    known = standing or {}
    words = list(dict.fromkeys(storefront_words(query)))
    if not words:
        return sorted(products, key=lambda p: (-known.get(p.id, 0.0), p.id))
    indexed = [(p, search_fields(p)) for p in products]
    total = len(indexed)
    found = {w: sum(1 for _, fields in indexed if any(w in f for f in fields.values())) for w in words}
    rarity = {w: math.log(1 + total / (1 + found[w])) for w in words}
    hits: list[tuple[float, float, str, ProductView]] = []
    for p, fields in indexed:
        score = 0.0
        for w in words:
            score += rarity[w] * max((weight for name, weight in FIELD_WEIGHTS if w in fields[name]), default=0.0)
        if score > 0:
            hits.append((-score, -known.get(p.id, 0.0), p.id, p))
    hits.sort(key=lambda h: h[:3])
    return [h[3] for h in hits]


def standing(answers: Iterable[RecordedAnswer]) -> dict[str, float]:
    """Each product's standing in a store's recorded search answers: the sum over the answers listing it of one over
    its place (1 for first, 1/2 for second, ...). A best-seller proxy: what the store lists often and early."""
    out: dict[str, float] = {}
    for answer in answers:
        for place, hit in enumerate(answer.products, start=1):
            out[hit.id] = out.get(hit.id, 0.0) + 1.0 / place
    return out


def search_fields(p: ProductView) -> dict[str, frozenset[str]]:
    """The words of each field `storefront_ranked` matches, folded as `storefront_words` folds a query."""
    facts = p.facts
    variants = [v.title for v in p.variants] + [label for v in p.variants for label in v.options.values()]
    text = {
        "title": p.title,
        "product_type": _string(facts.get("product_type")),
        "vendor": _string(facts.get("vendor")) or _string(facts.get("brand")),
        "tags": " ".join(t for t in facts.get("tags") or () if isinstance(t, str)),
        "variants": " ".join(variants),
        "description": " ".join(p.description.values()),
    }
    return {name: frozenset(storefront_words(value)) for name, value in text.items()}


def storefront_words(text: str) -> list[str]:
    """Whole words of `text`, case-folded, each plural folded to its singular ("accessories" to "accessory",
    "boxes" to "box", "bands" to "band"), so a query and a product's text meet whichever number either uses."""
    return [_singular(w) for w in _TOKEN.findall(text.casefold())]


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _score(p: ProductView, words: Sequence[str]) -> int:
    title = set(_WORD.findall(p.title.lower()))
    rest = set(_WORD.findall(p.text()))
    return sum(2 if w in title else 1 if w in rest else 0 for w in words)


def _price_filtered(p: ProductView, price: Mapping[str, Any] | None) -> ProductView | None:
    """The product with only the variants inside the price filter, or None when none is (catalog search, "Filters")."""
    if not price:
        return p
    lo, hi = price.get("min"), price.get("max")
    kept = tuple(
        v
        for v in p.variants
        if (not isinstance(lo, int) or v.price >= lo) and (not isinstance(hi, int) or v.price <= hi)
    )
    if not kept:
        return None
    return ProductView(p.id, p.title, p.description, p.facts, kept, p.option_names)


def _categories(facts: Mapping[str, Any]) -> list[dict[str, str]]:
    """A product's categories as UCP categories (`{value, taxonomy?}`), from strings or objects."""
    out: list[dict[str, str]] = []
    for c in facts.get("categories") or ():
        if isinstance(c, str) and c:
            out.append({"value": c})
        elif isinstance(c, Mapping) and isinstance(c.get("value"), str):
            item = {"value": str(c["value"])}
            if isinstance(c.get("taxonomy"), str):
                item["taxonomy"] = str(c["taxonomy"])
            out.append(item)
    return out


def _in_categories(p: ProductView, wanted: Sequence[str]) -> bool:
    """Filter categories match a category's value, or any step of a `A > B` path (catalog search, "Filters")."""
    have: set[str] = set()
    for c in _categories(p.facts):
        value = c["value"].lower()
        have |= {value, *(part.strip() for part in value.split(">"))}
    return any(w.lower().strip() in have for w in wanted)


def _rating(facts: Mapping[str, Any]) -> dict[str, Any] | None:
    """The product's rating when it has the UCP shape: a value and a scale maximum."""
    r = facts.get("rating")
    if not isinstance(r, Mapping):
        return None
    value, top = r.get("value"), r.get("scale_max")
    if not (_number(value) and _number(top)):
        return None
    out: dict[str, Any] = {"value": value, "scale_max": top}
    if isinstance(r.get("count"), int) and not isinstance(r.get("count"), bool):
        out["count"] = r["count"]
    return out


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value >= 0


def _metadata(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Business-defined product data (UCP `metadata`): what the catalog stated (attributes, details such as a size
    chart or allergens), with the brand and product type."""
    out = dict(facts["metadata"]) if isinstance(facts.get("metadata"), Mapping) else {}
    for key in ("brand", "product_type"):
        if isinstance(facts.get(key), str):
            out.setdefault(key, facts[key])
    return out


def served_catalog(pack: EnvPack, platform: Platform | None = None, *, viewed: bool = True) -> Catalog:
    """The catalog the UCP surface serves `pack` with: its platform's search, cursor, filter and product view rules
    (`platform`, else the one the pack is served under), and the recorded agent view a stand-in's pack names (D100)
    in place of the platform's product view (`viewed` off reads the documents before any view, as the exposure
    sheet does). One constructor for the surface and for the benchmark's parity reader (carttrace.bench.served_parity), so
    both read the same agent view."""
    row = platform or platform_named(pack.platform_name)
    agent = pack.agent_view if viewed else None
    views = row.product_views if agent is None else agent_shapes(agent, presented(pack).domain)
    return Catalog(
        pack,
        opaque_cursors=row.opaque_cursors,
        available_filter=row.available_filter,
        storefront_search=row.storefront_search,
        all_words=row.search_all_words,
        recorded_search=not pack.transplanted,
        total_count=row.search_total_count,
        phrase=row.search_phrase,
        prefix_words=row.search_prefix_words,
        field_prefix=row.search_field_prefix,
        whole_words=row.search_whole_words,
        page_default=row.page_default,
        page_max=row.page_max,
        facets=row.facets,
        views=views,
    )


def agent_shapes(view: AgentViewDoc, domain: str) -> dict[str, dict[str, Any]]:
    """A recorded agent view's shapes as the catalog serves them: each with what `_viewed` writes values from."""
    render = {"format": view.format, "domain": domain, "categories": view.categories}
    return {name: {**shape, RENDER: render} for name, shape in view.views.items()}


# -- documents ---------------------------------------------------------------------------------------------------------


def price(amount: int, currency: str) -> dict[str, Any]:
    return {"amount": amount, "currency": currency}


def variant_doc(v: VariantView, *, option_names: Sequence[str] = ()) -> dict[str, Any]:
    if v.document is not None:
        return _store_variant_doc(v)
    names = [n for n in option_names if n in v.options] or list(v.options)
    out: dict[str, Any] = {
        "id": v.id,
        "title": v.title,
        "description": {"plain": v.title},
        "price": price(v.price, v.currency),
        "availability": {"available": v.available, "status": "in_stock" if v.available else "out_of_stock"},
    }
    if names:
        out["options"] = [{"name": n, "label": v.options[n]} for n in names]
    return out


def _store_variant_doc(v: VariantView) -> dict[str, Any]:
    """A replica's variant: the store's document as captured, priced and stocked from the state."""
    assert v.document is not None
    out = {k: value for k, value in v.document.items() if k != "inputs" and not k.startswith("_")}
    out["id"] = v.id
    out["title"] = v.title
    out["price"] = price(v.price, v.currency)
    captured = v.document.get("availability")
    if isinstance(captured, Mapping) and captured.get("available") is v.available:
        out["availability"] = dict(captured)
    else:
        out["availability"] = {"available": v.available, "status": "in_stock" if v.available else "out_of_stock"}
    out.setdefault("description", {"plain": v.title})
    if "options" not in out and v.options:
        out["options"] = [{"name": n, "label": label} for n, label in v.options.items()]
    return out


def _store_product_doc(p: ProductView, variants: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """A replica's product: every field of the store's document, with the description, price range, options and
    variants rendered from the state (the store's price range while no price has moved)."""
    out = {k: value for k, value in p.facts.items() if k not in DERIVED and not k.startswith("_")}
    prices = [v.price for v in p.variants]
    cur = p.variants[0].currency
    captured = p.facts.get("price_range")
    unchanged = _prices_unchanged(p)
    out["id"] = p.id
    out["title"] = p.title
    out["description"] = dict(p.description)
    if unchanged and isinstance(captured, Mapping):
        out["price_range"] = dict(captured)
    else:
        out["price_range"] = {"min": price(min(prices), cur), "max": price(max(prices), cur)}
    out["variants"] = list(variants)
    if p.option_names:
        out["options"] = [{"name": n, "values": [{"label": label} for label in _labels(p, n)]} for n in p.option_names]
    return out


def product_doc(p: ProductView, variants: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """A product as search and lookup return it, with the variant documents given."""
    if p.captured:
        return _store_product_doc(p, variants)
    prices = [v.price for v in p.variants]
    cur = p.variants[0].currency
    out: dict[str, Any] = {
        "id": p.id,
        "title": p.title,
        "description": dict(p.description),
        "price_range": {"min": price(min(prices), cur), "max": price(max(prices), cur)},
        "variants": list(variants),
    }
    for key in ("handle", "url"):
        if isinstance(p.facts.get(key), str):
            out[key] = p.facts[key]
    media = _media(p.facts)
    if media:
        out["media"] = media
    tags = [t for t in p.facts.get("tags") or () if isinstance(t, str)]
    if tags:
        out["tags"] = tags
    categories = _categories(p.facts)
    if categories:
        out["categories"] = categories
    rating = _rating(p.facts)
    if rating is not None:
        out["rating"] = rating
    metadata = _metadata(p.facts)
    if metadata:
        out["metadata"] = metadata
    if p.option_names:
        out["options"] = [{"name": n, "values": [{"label": label} for label in _labels(p, n)]} for n in p.option_names]
    return out


def _labels(p: ProductView, name: str) -> list[str]:
    """An option's values: a replica's in the store's order (every value it listed, also those of variants the capture
    never saw), then any value only a variant names."""
    out: list[str] = []
    if p.captured:
        for option in p.facts.get("options") or ():
            if isinstance(option, Mapping) and option.get("name") == name:
                for value in option.get("values") or ():
                    label = value.get("label") if isinstance(value, Mapping) else value
                    if isinstance(label, str) and label not in out:
                        out.append(label)
    for v in p.variants:
        label = v.options.get(name)
        if label is not None and label not in out:
            out.append(label)
    return out


def lookup_docs(
    catalog: Catalog, state: State, ids: Sequence[str], *, index: Index | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """(products, ids not found). Each product appears once, carrying one featured or exact variant per resolving id,
    each variant with the `inputs` that resolved to it (catalog lookup, "Client Correlation")."""
    order: list[str] = []
    products: dict[str, ProductView] = {}
    picked: dict[str, dict[str, list[dict[str, str]]]] = {}
    missing: list[str] = []
    index = index or catalog.index(state)
    for ident in dict.fromkeys(ids):
        hit = index.find(ident)
        if hit is None:
            missing.append(ident)
            continue
        product, variant = hit
        chosen, match = (variant, "exact") if variant is not None else (product.featured, "featured")
        if product.id not in products:
            products[product.id] = product
            order.append(product.id)
        picked.setdefault(product.id, {}).setdefault(chosen.id, []).append({"id": ident, "match": match})
    docs = []
    for pid in order:
        p = products[pid]
        variants = []
        for v in p.variants:
            if v.id in picked[pid]:
                variants.append({**variant_doc(v, option_names=p.option_names), "inputs": picked[pid][v.id]})
        docs.append(shaped(product_doc(p, variants), p, LOOKUP, catalog.view(LOOKUP)))
    return docs, missing


def detail_doc(
    p: ProductView,
    variant: VariantView | None,
    selected: Sequence[Mapping[str, Any]],
    preferences: Sequence[str],
    *,
    recorded: RecordedAnswer | None = None,
    view: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A get_product product: effective selections after relaxation, the variants that match them (the requested or
    featured one first), and option values marked `available` and `exists` relative to those selections. With the
    store's `recorded` answer to this request, its selection and variant list instead. `view` is the platform's
    get_product shape (D55)."""
    if recorded is not None:
        listed = _listed(Index([p]), [h for h in recorded.products if h.id == p.id])
        if listed and listed[0][1]:
            return shaped(_recorded_detail(p, listed[0][1], recorded), p, DETAIL, view)
    return shaped(_computed_detail(p, variant, selected, preferences), p, DETAIL, view)


def _computed_detail(
    p: ProductView,
    variant: VariantView | None,
    selected: Sequence[Mapping[str, Any]],
    preferences: Sequence[str],
) -> dict[str, Any]:
    if variant is not None:
        effective = dict(variant.options)
        featured = variant
    else:
        effective = _relaxed(p, _wanted(selected), preferences)
        matching = [v for v in p.variants if _matches(v, effective)]
        featured = next((v for v in matching if v.available), matching[0] if matching else p.featured)
        if not selected:
            effective = dict(featured.options)
    rest = [v for v in p.variants if v.id != featured.id and _matches(v, effective)]
    doc = product_doc(p, [variant_doc(v, option_names=p.option_names) for v in (featured, *rest)])
    doc["selected"] = [{"name": n, "label": effective[n]} for n in p.option_names if n in effective]
    if p.option_names:
        doc["options"] = [
            {
                "name": n,
                "values": [{"label": label, **_signals(p, {**effective, n: label})} for label in _labels(p, n)],
            }
            for n in p.option_names
        ]
    return doc


def _recorded_detail(p: ProductView, listed: Sequence[VariantView], recorded: RecordedAnswer) -> dict[str, Any]:
    """The store's answer to a get_product it was asked: its selection and its variants, priced from the state, with
    availability signals from the state where the replica has a matching variant and the store's own elsewhere."""
    effective = {s["name"]: s["label"] for s in recorded.selected or () if "name" in s and "label" in s}
    doc = product_doc(p, [variant_doc(v, option_names=p.option_names) for v in listed])
    doc["selected"] = [{"name": n, "label": label} for n, label in effective.items()]
    if p.option_names:
        stored = _stored_signals(recorded)
        doc["options"] = [
            {
                "name": n,
                "values": [
                    {"label": label, **_signals(p, {**effective, n: label}, stored.get((n, label)))}
                    for label in _labels(p, n)
                ],
            }
            for n in p.option_names
        ]
    return doc


def _stored_signals(recorded: RecordedAnswer) -> dict[tuple[str, str], dict[str, bool]]:
    out: dict[tuple[str, str], dict[str, bool]] = {}
    for option in recorded.options or ():
        name = option.get("name")
        for value in option.get("values") or ():
            if isinstance(name, str) and isinstance(value, Mapping) and isinstance(value.get("label"), str):
                signals = {k: value[k] for k in ("available", "exists") if isinstance(value.get(k), bool)}
                if len(signals) == 2:
                    out[(name, value["label"])] = signals
    return out


def _wanted(selected: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for s in selected:
        name, label = s.get("name"), s.get("label")
        if isinstance(name, str) and isinstance(label, str) and name not in out:
            out[name] = label
    return out


def _matches(v: VariantView, selection: Mapping[str, str]) -> bool:
    return all(v.options.get(n) == label for n, label in selection.items())


def _relaxed(p: ProductView, wanted: dict[str, str], preferences: Sequence[str]) -> dict[str, str]:
    """Drop selections until some variant matches: from the end of `preferences` first, then the latest selection."""
    current = dict(wanted)
    order = [n for n in reversed(list(preferences)) if n in current]
    order += [n for n in reversed(list(current)) if n not in order]
    while current and not any(_matches(v, current) for v in p.variants):
        current.pop(order.pop(0))
    return current


def _signals(p: ProductView, selection: Mapping[str, str], stored: Mapping[str, bool] | None = None) -> dict[str, bool]:
    """Whether a variant matching `selection` is for sale and exists; the store's own `stored` signals when the
    replica has no matching variant (one the capture never saw)."""
    matching = [v for v in p.variants if _matches(v, selection)]
    if not matching and stored is not None:
        return dict(stored)
    return {"available": any(v.available for v in matching), "exists": bool(matching)}

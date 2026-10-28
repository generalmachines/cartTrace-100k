"""Capture v2 read back: a store capture pass's edition directory as the store's catalog, in UCP's product shape.

Why a reader here. The capture pass (`carttrace live capture-all`) writes one directory per store and edition under
<paths.var>/store/captures/<edition>/<store>/: `report.json` (written last, so a directory without one is a pass
still running or one that failed), `canvas.json` (the store, its rights), `tables/*.parquet` (products, variants,
offers, reviews_summary, policies, probes, in the declared schemas of carttrace.live.tables) and `raw/` (every page and feed
as served, gzipped, listed in raw/manifest.json). Two consumers need the same catalog out of it and sit on one layer,
so neither may import the other: the live loop's task fixtures (carttrace.collect, `carttrace live fixtures`) and the sandbox
replica (carttrace.sandbox.catalogs.capture_v2). Both read it here, through `store_catalog`, so a live task's products are
always products of the replica it pairs with.

What a product holds. The tables are the truth: ids, titles, vendor, product type, tags, collections (as categories),
each variant's price, compare-at price (as `list_price` when it is higher), availability, options, SKU and barcode,
and the review summary (as `rating`). The tables keep only counts of description characters and image alt texts, so
the description and the media (with alt text) are read from the raw Shopify feed pages and product .js documents
when the capture holds them; other platforms' products carry what the tables hold. Option values carry `available`:
whether any variant for sale has that value. A product the tables give no variant (product pages whose offers had no
variant id) gets one variant per priced offer, marked `_offer`; those exist only in the sandbox, so live fixtures
leave such products out.

Listings (D38). The capture took some category and collection grids for products (Store 167's
eucalyptus-collection, 37 of its pages; 8 of Store 041's), and a task naming one names a page nothing can be chosen on.
The pass records each detailed page's kind in `products.page_kind`; `store_catalog` leaves out every product whose
page was a listing, says how many in its notes and in `StoreCatalog.listings`, so fixtures, replicas and the intent
writers never see one. A table without the column (written before it, and not backfilled) is "unknown": nothing is
left out. `listing_ids` names a capture's listings (products and their variants, as the tables and as UCP write
them), for screening intents already written.

Ids. A Shopify store's numeric ids are written as its UCP server writes them (gid://shopify/Product/<id>), so a
fixture, a replica and a live reply name a product the same way.

Which products. A big store indexes tens of thousands of products, more than a task generator or an envpack needs.
`store_catalog` keeps at most `limit` (DEFAULT_PRODUCTS): the products the pass read in detail first, then a seeded
spread of the rest, the same every time for the same capture. Fixtures are cut from that same selection, so the
replica always holds every product a live task names. A variant id names one variant: a product whose variant ids
an earlier product already holds keeps only its own (store-011's colour pages each listed the same skus, and its world
did not build), and one left with none is left out, with a note.

Availability. One reading of schema.org's availability words for every writer and reader (`availability_word`,
`available_of`): the URL prefix, case, spaces, underscores and hyphens do not matter, so "OnlineOnly", "In Stock",
"instock" and "LimitedAvailability" read for sale. Tables written before it carry page-made variants written sold
out from such offers; `corrected_variants` reads them again from their offers, never changing a cached row.

Partial captures. A pass the store stopped part way (blocked or throttled) after it read the catalog still holds
true rows. `CaptureV2.partial_usable` says when they are enough for a sandbox replica (PARTIAL_MIN_PRODUCTS read in
detail and indexed, and no block on the catalog itself); only the replica path asks (`newest_captures_v2`'s
`partial_min`), and live sessions and their fixtures stay on complete captures.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import pyarrow.parquet as pq

from carttrace.core.canon import digest, dumps
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.ids import stable_id
from carttrace.trace.jsonld import page_blocks, without_reviewers
from carttrace.trace.schema import Rights

STORE_CAPTURES: Final = Path("store") / "captures"
"""Under paths.var: <edition>/<store>/ per capture."""
ID_PREFIX: Final = "cv2-"
REPORT = "report.json"
CANVAS = "canvas.json"
TABLES = "tables"
RAW = "raw"
COMPLETE: Final = "complete"
DEFAULT_PRODUCTS: Final = 500
"""Products a store's catalog keeps at most, for its fixtures and its replica."""
RATING_SCALE: Final = 5
"""Review summaries in the tables are on a five-star scale."""
RAW_MAX_BYTES: Final = 64 * 1024 * 1024
"""A raw page larger than this, uncompressed, is not read for descriptions."""
QUOTE_CHARS: Final = 200
STAMP_COLUMNS: Final = frozenset({"store_id", "edition", "captured_at", "expires_at", "purpose"})
LISTING: Final = "listing"
"""The products table's `page_kind` for a page that lists products rather than showing one (carttrace.live.page_kind)."""
PAGE_KIND: Final = "page_kind"
POLICY_TYPES: Final = {
    "returns": "dev.ucp.shopping.policy.return",
    "warranty": "dev.ucp.shopping.policy.warranty",
}
"""Fact prefix to the UCP policy type; any other fact is typed under our own reverse-DNS name."""
OWN_POLICY_TYPE: Final = "ai.generalmachines.policy"
PARTIAL_MIN_PRODUCTS: Final = 20
"""Products a partial capture must have read in detail, and indexed, before a sandbox replica may be built from it
(the module docstring's partial captures)."""
PARTIAL_STATUSES: Final = frozenset({"blocked", "throttled"})
"""Stop reasons a partial capture may have: the store stopped answering part way. Robots and off_store captures read
what the store did not allow, or another site, and are never partial."""
CATALOG_KINDS: Final = ("product-", "feed-", "collection", "sitemap-", "graphql-", "woo-", "meta", "home")
"""Raw manifest kinds of the answers that are the catalog itself: product pages and their .js, the platform feeds,
collections, sitemaps and the homepage. A store that blocked one of them blocked its products."""
PAGE_SOURCES: Final = frozenset({"jsonld", "microdata", "og"})
"""Offer sources of a product page's markup (carttrace.live.generic), the offers a page-made variant was written from."""
SITEMAP_PAGE: Final = "sitemap_page"
"""The products table's source of a product read from its page (carttrace.live.generic), not from a platform index."""
IN_STOCK: Final = frozenset({"InStock", "LimitedAvailability", "OnlineOnly"})
ORDERABLE: Final = frozenset({"PreOrder", "PreSale", "BackOrder", "MadeToOrder"})
"""Not in stock now, but a cart and a checkout take it (schema.org's words for an order that ships later)."""
OUT_OF_STOCK: Final = frozenset({"OutOfStock", "SoldOut", "Discontinued", "InStoreOnly"})
_AVAILABILITY_WORDS: Final = {
    **{w.lower(): w for w in IN_STOCK | ORDERABLE | OUT_OF_STOCK},
    "available": "InStock",
    "oos": "OutOfStock",
    "unavailable": "OutOfStock",
}
"""Each schema.org availability word, folded (lowercase, no spaces, underscores or hyphens), to its own spelling;
plus the plain words pages write in microdata text and OpenGraph."""


@dataclass(frozen=True, slots=True)
class CaptureV2:
    """One store's capture in one edition, as its report and canvas describe it."""

    path: Path
    store_id: str
    edition: str
    status: str
    """The report's stop_reason: complete, blocked, robots, throttled, off_store, ..."""
    captured_at: str
    expires_at: str | None
    domain: str
    platform: str
    market: str | None
    category: str | None
    agent_surfaces: tuple[str, ...]
    """The agent protocols the store declared (ucp_mcp, store_mcp, ...)."""
    wall: str | None
    robots_disallowed: tuple[str, ...]
    products_indexed: int
    rights: Rights
    products_detailed: int = 0
    """Product pages the pass read with Product markup (the report's products_detailed)."""
    stop_kind: str | None = None
    """The raw manifest kind of the answer that stopped a blocked or throttled pass (`wellknown-a2a_agent_card`,
    `page-returns`, `product-<id>`, `feed-004`, ...), None when the pass was not stopped so or kept no such answer."""

    @property
    def id(self) -> str:
        """cv2-<sha12> of the store, the edition and the capture time: what a manifest names a replica world by."""
        return ID_PREFIX + digest([self.store_id, self.edition, self.captured_at], 12)

    @property
    def complete(self) -> bool:
        return self.status == COMPLETE

    @property
    def usable(self) -> bool:
        """Complete, with products, behind no wall: a capture a replica and fixtures can be made from."""
        return self.complete and self.products_indexed > 0 and self.wall is None

    @property
    def partial(self) -> bool:
        """The pass stopped before its end (blocked, throttled, ...): a replica built from it holds what was read
        before the stop, which an experiment records (`capture_complete` false)."""
        return not self.complete

    def partial_usable(self, min_products: int = PARTIAL_MIN_PRODUCTS) -> bool:
        """Whether a stopped pass still read enough of the catalog for a sandbox replica (never a live session):
        stopped blocked or throttled, with at least `min_products` products indexed and read in detail, and not
        blocked on the catalog itself. A 429 anywhere is a rate, not a wall, so a throttled pass passes wherever it
        stopped; a 401, a 403 or a challenge on a product page, a feed, a collection or a sitemap means the store
        walls its products from our reader, and that is the store's recorded outcome. A block on a page beside the
        catalog (/.well-known/agent-card.json, a returns page, an order-tracking page) leaves the products read
        before it as true as any. In 2026-W42 this brought in 19 stores: 10 blocked beside the catalog (seven at the
        agent card, store-278 and store-607 among them; store-145 and store-586 at /a/returns; store-a155 at its shipping page) and 9
        throttled after
        22 to 200 detailed products; store-193, store-285 and store-294, blocked on product pages, stay out."""
        if min_products <= 0 or self.status not in PARTIAL_STATUSES:
            return False
        if self.products_indexed < min_products or self.products_detailed < min_products:
            return False
        walled = self.stop_kind is None or self.stop_kind.startswith(CATALOG_KINDS)
        return not (self.status == "blocked" and walled)

    def replica_usable(self, partial_min: int = 0) -> bool:
        """`usable`, or with `partial_min` above zero, `partial_usable(partial_min)`: what a sandbox replica may be
        built from. Zero keeps the complete-only rule."""
        return self.usable or (partial_min > 0 and self.partial_usable(partial_min))

    @property
    def ucp(self) -> bool:
        return "ucp_mcp" in self.agent_surfaces


@dataclass(frozen=True, slots=True)
class StoreCatalog:
    """A capture's products as UCP documents, the store's policies, and its currency."""

    capture: CaptureV2
    currency: str | None
    products: list[dict[str, Any]]
    policies: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)
    listings: int = 0
    """Products left out because their page was a listing (D38)."""
    jsonld: dict[str, list[Any]] = field(default_factory=dict)
    """UCP product id to the schema.org JSON-LD blocks its product page carried as the capture kept it, reviewers'
    names redacted (D67); read only when `store_catalog` is asked for it."""


def availability_word(value: Any) -> str | None:
    """A schema.org availability as its own word ("InStock", "OutOfStock", "PreOrder", ...), whatever its spelling:
    the URL prefix stripped (https://schema.org/, http://schema.org/, schema:, and the "schema.orgInStock" a page
    wrote without its slash), case, spaces, underscores and hyphens folded. None when it is empty or no word of
    schema.org's. One reading for every writer and reader: in 2026-W42, 459 offers said "OnlineOnly" (store-606,
    store-a135, store-a136), 175 said "In Stock" in microdata text (store-591), 98 "instock" (store-a012), 44 "schema.orgInStock"
    (store-349) and 47 "LimitedAvailability", and three writers each kept their own list, so nine stores read every
    variant sold out."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    lowered = text.lower()
    for prefix in ("https://schema.org/", "http://schema.org/", "https://schema.org", "http://schema.org", "schema:"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :]
            break
    else:
        if lowered.startswith("schema.org"):
            text = text[len("schema.org") :].lstrip("/")
        elif "/" in text:
            text = text.rstrip("/").rsplit("/", 1)[-1]
    folded = "".join(ch for ch in text.lower() if ch.isalnum())
    return _AVAILABILITY_WORDS.get(folded)


def available_of(value: Any, *, orderable: bool = True) -> bool | None:
    """Whether a schema.org availability means the item can be bought: True in stock (and, with `orderable`, on
    pre-order or back order, which a cart takes), False out of stock, None when the value says nothing schema.org
    names. The capture's own writers count pre-orders as for sale; a website reader that wants stock now passes
    `orderable=False`."""
    word = availability_word(value)
    if word is None:
        return None
    if word in IN_STOCK:
        return True
    if word in ORDERABLE:
        return orderable
    return False


def captures_root(var: Path) -> Path:
    return var / STORE_CAPTURES


def is_capture_v2(path: Path) -> bool:
    """Whether `path` is a capture v2 store directory (it holds report.json and tables/)."""
    return path.is_dir() and (path / REPORT).is_file() and (path / TABLES).is_dir()


def read_capture_v2(path: Path) -> CaptureV2:
    """The capture in a store directory. Raises NotFoundError when it has no report (still running, or failed)."""
    report_path = path / REPORT
    if not report_path.is_file():
        raise NotFoundError(f"{path} holds no {REPORT}", hint="the pass is still running there, or it failed")
    report = _json_object(report_path)
    canvas = _json_object(path / CANVAS) if (path / CANVAS).is_file() else {}
    raw_store = canvas.get("store")
    store: Mapping[str, Any] = raw_store if isinstance(raw_store, Mapping) else {}
    raw_rights = canvas.get("rights")
    try:
        rights = Rights.model_validate(
            {k: v for k, v in raw_rights.items() if k in Rights.model_fields} if isinstance(raw_rights, Mapping) else {}
        )
    except ValueError:
        rights = Rights(platform_terms="unknown", basis="capture v2 canvas rights did not validate; nothing allowed")
    store_id = _text(report.get("store")) or _text(store.get("store_id")) or path.name
    edition = _text(report.get("edition")) or path.parent.name
    captured_at = _text(report.get("captured_at")) or _text(canvas.get("captured_at")) or ""
    return CaptureV2(
        path=path,
        store_id=store_id,
        edition=edition,
        status=_text(report.get("stop_reason")) or "unknown",
        captured_at=captured_at,
        expires_at=_text(report.get("expires_at")),
        domain=_text(store.get("domain")) or "",
        platform=_text(store.get("platform")) or "unknown",
        market=_text(store.get("market")),
        category=_text(store.get("category")),
        agent_surfaces=tuple(str(s) for s in report.get("agent_surfaces") or () if isinstance(s, str)),
        wall=_text(report.get("wall")),
        robots_disallowed=tuple(str(s) for s in report.get("robots_disallowed") or () if isinstance(s, str)),
        products_indexed=int(report.get("products_indexed") or 0),
        rights=rights,
        products_detailed=int(report.get("products_detailed") or 0),
        stop_kind=_stop_kind(path) if report.get("stop_reason") in PARTIAL_STATUSES else None,
    )


def _stop_kind(path: Path) -> str | None:
    """The manifest kind of the last blocked or throttled answer a pass kept (the one that stopped it), or None."""
    manifest = path / RAW / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        entries = _json_object(manifest).get("entries")
    except ValidationError:
        return None
    for entry in reversed(entries if isinstance(entries, list) else []):
        if isinstance(entry, Mapping) and entry.get("status") in PARTIAL_STATUSES:
            return _text(entry.get("kind"))
    return None


def capture_v2_dirs(root: Path, *, edition: str | None = None, store_ids: Sequence[str] = ()) -> list[Path]:
    """Store directories with a report under `root` (one edition's when given), sorted."""
    if not root.is_dir():
        return []
    editions = [root / edition] if edition is not None else sorted(p for p in root.iterdir() if p.is_dir())
    wanted = set(store_ids)
    out: list[Path] = []
    for folder in editions:
        if not folder.is_dir():
            continue
        out += [p for p in sorted(folder.iterdir()) if (not wanted or p.name in wanted) and (p / REPORT).is_file()]
    return out


def read_captures_v2(root: Path, *, edition: str | None = None, store_ids: Sequence[str] = ()) -> list[CaptureV2]:
    """Every capture with a report under `root`; a report that does not parse is skipped (its pass is rerun)."""
    out = []
    for path in capture_v2_dirs(root, edition=edition, store_ids=store_ids):
        try:
            out.append(read_capture_v2(path))
        except (NotFoundError, ValidationError):
            continue
    return out


def newest_edition(root: Path) -> str | None:
    """The edition holding the newest capture under `root`, or None when there is none."""
    found = read_captures_v2(root)
    return max(found, key=lambda c: (c.captured_at, c.edition)).edition if found else None


def newest_captures_v2(
    root: Path,
    *,
    edition: str | None = None,
    store_ids: Sequence[str] = (),
    category: str | None = None,
    partial_min: int = 0,
) -> dict[str, CaptureV2]:
    """Store id to its newest usable capture (complete, with products, no wall), of one edition when given and of
    one category when given (a capture that names no category matches any). With `partial_min` above zero a partial
    capture that read at least that many products counts too (`CaptureV2.partial_usable`): only the sandbox replica
    path asks for that, never a live pool or its fixtures."""
    out: dict[str, CaptureV2] = {}
    for cap in read_captures_v2(root, edition=edition, store_ids=store_ids):
        if not cap.replica_usable(partial_min) or (category is not None and cap.category not in (None, category)):
            continue
        held = out.get(cap.store_id)
        if held is None or (cap.captured_at, cap.edition) > (held.captured_at, held.edition):
            out[cap.store_id] = cap
    return dict(sorted(out.items()))


def find_capture_v2(root: Path, ident: str | Path) -> CaptureV2:
    """A capture by its directory or by its cv2- id under `root`."""
    candidate = Path(ident)
    if is_capture_v2(candidate):
        return read_capture_v2(candidate)
    text = str(ident)
    if text.startswith(ID_PREFIX):
        for cap in read_captures_v2(root):
            if cap.id == text:
                return cap
        raise NotFoundError(f"no capture {text} under {root}", hint="list them with `carttrace live eligible`")
    raise NotFoundError(f"{text} is neither a capture v2 directory nor a {ID_PREFIX} id")


def table_rows(capture: CaptureV2, name: str) -> list[dict[str, Any]]:
    """One table's rows without the stamp columns; an absent table has none."""
    path = capture.path / TABLES / f"{name}.parquet"
    if not path.is_file():
        return []
    return _read_rows(path)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """A parquet table's rows, its columns in the file's order with the stamp columns left out. Only the kept columns
    are read: a big store's variants and offers tables hold hundreds of thousands of rows."""
    columns = [name for name in pq.read_schema(path).names if name not in STAMP_COLUMNS]
    if not columns:
        return [{} for _ in range(pq.read_metadata(path).num_rows)]
    # no thread pool, as carttrace.live.tables.read_parquet: its threads aborted `carttrace live replicas` at exit (2026-10-14)
    rows: list[dict[str, Any]] = pq.read_table(path, columns=columns, use_threads=False).to_pylist()
    return rows


def _shared_rows(capture: CaptureV2, name: str) -> list[dict[str, Any]]:
    """`table_rows` for this module's own read-only use, remembered per process while the file is unchanged: a
    manifest reads one store's catalog twice (its size, then its replica), and the tables are most of the cost.
    Callers here never change a row; anything outside gets fresh rows from `table_rows`."""
    path = capture.path / TABLES / f"{name}.parquet"
    try:
        stat = path.stat()
    except OSError:
        return []
    if not path.is_file():
        return []
    return _cached_rows(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _cached_rows(path: str, mtime_ns: int, size: int) -> list[dict[str, Any]]:
    """One store's four catalog tables at a time: the next store's evict them."""
    return _read_rows(Path(path))


# -- the catalog -------------------------------------------------------------------------------------------------------


def store_catalog(
    capture: CaptureV2, *, limit: int = DEFAULT_PRODUCTS, raw: bool = True, jsonld: bool = False
) -> StoreCatalog:
    """The capture's products as UCP product documents (the module docstring says which and what they hold), its
    policies as UCP policy objects, and its currency (the one most variants are priced in). With `jsonld`, also each
    product page's schema.org JSON-LD as the raw page held it (D67), which a replica's website pages serve."""
    notes: list[str] = []
    rows = _shared_rows(capture, "products")
    products = [p for p in rows if p.get(PAGE_KIND) != LISTING]
    listings = len(rows) - len(products)
    if listings:
        notes.append(f"left out {listings} listing pages (page_kind listing): a grid is not a product")
    offers = _shared_rows(capture, "offers")
    variants = corrected_variants(rows, _shared_rows(capture, "variants"), offers)
    currency = _currency(variants) or _currency(offers)
    if currency is None:
        return StoreCatalog(
            capture, None, [], store_policies(capture), [*notes, "no priced variant or offer"], listings
        )
    by_product: dict[str, list[dict[str, Any]]] = {}
    for row in variants:
        if _priced(row, currency) and _text(row.get("variant_id")) is not None:
            by_product.setdefault(str(row["product_id"]), []).append(row)
    offered: dict[str, list[dict[str, Any]]] = {}
    for row in offers:
        if _priced(row, currency):
            offered.setdefault(str(row["product_id"]), []).append(row)
    sellable = [
        p
        for p in products
        if _text(p.get("product_id")) and (str(p["product_id"]) in by_product or offered.get(str(p["product_id"])))
    ]
    chosen = _choose(capture, sellable, limit)
    if len(sellable) > len(chosen):
        notes.append(f"kept {len(chosen)} of {len(sellable)} sellable products")
    details = _raw_details(capture, {str(p["product_id"]) for p in chosen}, notes) if raw else {}
    ratings = {str(r["product_id"]): r for r in _shared_rows(capture, "reviews_summary") if r.get("rating") is not None}
    docs = []
    seen: set[str] = set()
    repeated = dropped = 0
    for row in chosen:
        pid = str(row["product_id"])
        rows_of = by_product.get(pid) or []
        own: list[dict[str, Any]] = []
        for v in rows_of:
            # once across products and within one (store-304 and store-334 list a variant twice on one product)
            if str(v["variant_id"]) not in seen:
                seen.add(str(v["variant_id"]))
                own.append(v)
        repeated += len(rows_of) - len(own)
        if rows_of and not own:
            dropped += 1
            continue
        docs.append(
            _product_doc(
                capture,
                row,
                variants=own,
                offers=offered.get(pid) or [],
                currency=currency,
                detail=details.get(pid, {}),
                rating=ratings.get(pid),
            )
        )
    if repeated:
        notes.append(
            f"left out {repeated} variants whose id an earlier product already holds"
            + (f", and {dropped} products left with none" if dropped else "")
            + ": a variant id names one variant (store-011's colour pages each listed the same skus)"
        )
    blocks = _raw_jsonld(capture, {str(p["product_id"]) for p in chosen}, notes) if jsonld else {}
    by_doc = {ucp_id(capture, "Product", pid): found for pid, found in blocks.items()}
    return StoreCatalog(capture, currency, docs, store_policies(capture), notes, listings, by_doc)


def variant_availability(capture: CaptureV2) -> list[bool | None]:
    """Each variant row's availability as the catalog reads it (`corrected_variants`), in the table's order."""
    corrected = corrected_variants(
        _shared_rows(capture, "products"), _shared_rows(capture, "variants"), _shared_rows(capture, "offers")
    )
    return [row.get("available") if isinstance(row.get("available"), bool) else None for row in corrected]


def corrected_variants(
    products: Sequence[Mapping[str, Any]], variants: Sequence[dict[str, Any]], offers: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The variant rows with the page writer's old availability misreading undone: a variant carttrace.live.generic made
    from a product page's offer (its product's source is `sitemap_page`; its id `<pid>#<n>` names the n-th markup
    offer of the product in the offers table, or it is that offer's sku) that the tables say is not available while
    its offer's availability word reads for sale (`available_of`) is available. The writer took only "InStock",
    "PreOrder" and "BackOrder", so "OnlineOnly", "In Stock", "instock" and "LimitedAvailability" offers were written
    sold out (nine 2026-W42 stores read every variant sold out). A corrected row is a copy; the rows given are never
    changed (they may be the module's cached ones)."""
    pages = {str(p.get("product_id")) for p in products if p.get("source") == SITEMAP_PAGE}
    if not pages:
        return list(variants)
    by_product: dict[str, list[Mapping[str, Any]]] = {}
    for row in offers:
        pid = str(row.get("product_id"))
        if pid in pages and row.get("source") in PAGE_SOURCES:
            by_product.setdefault(pid, []).append(row)
    out: list[dict[str, Any]] = []
    for row in variants:
        pid = str(row.get("product_id"))
        offer = _offer_of(row, by_product.get(pid, ())) if row.get("available") is False and pid in pages else None
        if offer is not None and available_of(offer.get("availability")) is True:
            out.append({**row, "available": True})
        else:
            out.append(row)
    return out


def _offer_of(variant: Mapping[str, Any], offers: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The markup offer a page-made variant was written from: `<pid>#<n>` is the n-th, else the offer of its sku."""
    vid = _text(variant.get("variant_id")) or ""
    pid = str(variant.get("product_id"))
    index = vid.removeprefix(f"{pid}#")
    if index != vid and index.isdigit():
        n = int(index)
        return offers[n - 1] if 1 <= n <= len(offers) else None
    return next((o for o in offers if _text(o.get("sku")) == vid), None)


def has_page_kinds(capture: CaptureV2) -> bool:
    """Whether the capture's products table records page kinds (a pass from D38 on, or a backfilled one)."""
    path = capture.path / TABLES / "products.parquet"
    return path.is_file() and PAGE_KIND in pq.read_schema(path).names


def listing_ids(capture: CaptureV2) -> frozenset[str]:
    """Every id a task could name a listing by: each listing product's id and its variants' ids, as the tables write
    them and as the store's UCP server would (gid://shopify/... on Shopify). Empty when the capture records no kinds."""
    listed = {str(p["product_id"]) for p in table_rows(capture, "products") if p.get(PAGE_KIND) == LISTING}
    if not listed:
        return frozenset()
    out: set[str] = set()
    for pid in listed:
        out |= {pid, ucp_id(capture, "Product", pid)}
    for row in table_rows(capture, "variants"):
        vid = _text(row.get("variant_id"))
        if vid is not None and str(row.get("product_id")) in listed:
            out |= {vid, ucp_id(capture, "ProductVariant", vid)}
    return frozenset(out)


def fixture_products(catalog: StoreCatalog) -> list[dict[str, Any]]:
    """The catalog's products in the live fixtures' shape (id, title, variants, product_type, vendor, description,
    tags), leaving out offer-made variants (they exist only in the sandbox) and products left with none."""
    out = []
    for doc in catalog.products:
        real = [v for v in doc["variants"] if not v.get("_offer")]
        if not real:
            continue
        product: dict[str, Any] = {
            "id": doc["id"],
            "title": doc["title"],
            "variants": [
                {
                    "id": v["id"],
                    "title": v["title"],
                    "price": dict(v["price"]),
                    "availability": {"available": bool(v["availability"]["available"])},
                    "options": [dict(o) for o in v.get("options") or ()],
                }
                for v in real
            ],
            "product_type": doc.get("product_type") or "",
            "vendor": doc.get("vendor") or "",
        }
        if isinstance(doc.get("description"), Mapping):
            product["description"] = dict(doc["description"])
        if doc.get("tags"):
            product["tags"] = list(doc["tags"])
        out.append(product)
    return out


def store_policies(capture: CaptureV2) -> list[dict[str, Any]]:
    """The store's policy facts (returns window, fees, transit and processing days, free shipping threshold) as UCP
    policy objects: a type, a plain description quoting the page, the page's url and the fact itself. Rows the pass
    marked as conflicting are left out; the page's own statement is kept."""
    out: dict[str, dict[str, Any]] = {}
    for row in table_rows(capture, "policies"):
        fact = _text(row.get("fact"))
        if fact is None or row.get("kind") == "conflict":
            continue
        value = row.get("value")
        label = fact.replace("_", " ")
        text = f"{label}: {value}" if value is not None else label
        quote = _text(row.get("quote"))
        if quote:
            text += f' ("{" ".join(quote.split())[:QUOTE_CHARS]}")'
        prefix = fact.split("_", 1)[0]
        policy: dict[str, Any] = {
            "type": POLICY_TYPES.get(prefix, f"{OWN_POLICY_TYPE}.{fact}"),
            "description": {"plain": text},
            "fact": fact,
        }
        if value is not None:
            policy["value"] = value
        url = _text(row.get("url"))
        if url and url.startswith(("http://", "https://")):
            policy["url"] = url
        out.setdefault(dumps([fact, value, url]), policy)
    return list(out.values())


def ucp_id(capture: CaptureV2, kind: str, raw_id: str) -> str:
    """A table id as the store's UCP server writes it: gid://shopify/<kind>/<id> for a Shopify store's numeric id."""
    if capture.platform == "shopify" and raw_id.isdigit():
        return f"gid://shopify/{kind}/{raw_id}"
    return raw_id


# -- internals ---------------------------------------------------------------------------------------------------------


def _choose(capture: CaptureV2, products: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """At most `limit` products: those read in detail first, then a seeded spread, kept in the store's order."""
    if len(products) <= limit:
        return list(products)
    ranked = sorted(
        range(len(products)),
        key=lambda i: (
            not products[i].get("sampled"),
            stable_id("cv2pick", capture.store_id, products[i]["product_id"]),
        ),
    )
    kept = sorted(ranked[: max(0, limit)])
    return [products[i] for i in kept]


def _product_doc(
    capture: CaptureV2,
    row: Mapping[str, Any],
    *,
    variants: Sequence[Mapping[str, Any]],
    offers: Sequence[Mapping[str, Any]],
    currency: str,
    detail: Mapping[str, Any],
    rating: Mapping[str, Any] | None,
) -> dict[str, Any]:
    pid = str(row["product_id"])
    title = _text(row.get("title")) or pid
    docs = [_variant_doc(capture, v, currency) for v in variants] or [
        _offer_variant(capture, pid, i, o, currency=currency, title=title) for i, o in enumerate(offers)
    ]
    doc: dict[str, Any] = {"id": ucp_id(capture, "Product", pid), "title": title}
    for key in ("handle", "url", "vendor", "product_type"):
        value = _text(row.get(key))
        if value:
            doc[key] = value
    if isinstance(detail.get("description"), str) and detail["description"].strip():
        doc["description"] = {"html": detail["description"]}
    tags = [t for t in row.get("tags") or () if isinstance(t, str) and t.strip()]
    if tags:
        doc["tags"] = tags
    collections = [c for c in row.get("collections") or () if isinstance(c, str) and c.strip()]
    if collections:
        doc["categories"] = [{"value": c} for c in collections]
    media = detail.get("media")
    if isinstance(media, list) and media:
        doc["media"] = media
    names = [n for n in row.get("option_names") or () if isinstance(n, str) and n]
    options = _options(names, docs)
    if options:
        doc["options"] = options
    doc["variants"] = docs
    if rating is not None and isinstance(rating.get("rating"), int | float):
        summary: dict[str, Any] = {"value": float(rating["rating"]), "scale_max": RATING_SCALE}
        if isinstance(rating.get("count"), int):
            summary["count"] = rating["count"]
        doc["rating"] = summary
    return doc


def _variant_doc(capture: CaptureV2, row: Mapping[str, Any], currency: str) -> dict[str, Any]:
    vid = str(row["variant_id"])
    options = _option_pairs(row.get("options"))
    title = _text(row.get("title")) or " / ".join(v for _, v in options) or vid
    price = int(row["price_minor"])
    doc: dict[str, Any] = {
        "id": ucp_id(capture, "ProductVariant", vid),
        "title": title,
        "price": {"amount": price, "currency": currency},
        "availability": {"available": row.get("available") is not False},
    }
    if options:
        doc["options"] = [{"name": n, "label": v} for n, v in options]
    compare = row.get("compare_at_minor")
    if isinstance(compare, int) and compare > price:
        doc["list_price"] = {"amount": compare, "currency": currency}
    sku = _text(row.get("sku"))
    if sku:
        doc["sku"] = sku
    barcode = _text(row.get("barcode"))
    if barcode:
        doc["barcodes"] = [{"type": "gtin", "value": barcode}]
    return doc


def _offer_variant(
    capture: CaptureV2, pid: str, index: int, row: Mapping[str, Any], *, currency: str, title: str
) -> dict[str, Any]:
    """A variant made from a product page's offer, for a product the tables give no variant: sandbox only."""
    price = int(row["price_minor"])
    raw_id = _text(row.get("variant_id")) or f"{pid}-offer-{index + 1}"
    doc: dict[str, Any] = {
        "id": ucp_id(capture, "ProductVariant", raw_id),
        "title": title if index == 0 else f"{title} ({index + 1})",
        "price": {"amount": price, "currency": currency},
        "availability": {"available": available_of(row.get("availability")) is not False},
        "_offer": True,
    }
    compare = row.get("compare_at_minor")
    if isinstance(compare, int) and compare > price:
        doc["list_price"] = {"amount": compare, "currency": currency}
    sku = _text(row.get("sku"))
    if sku:
        doc["sku"] = sku
    gtin = _text(row.get("gtin"))
    if gtin:
        doc["barcodes"] = [{"type": "gtin", "value": gtin}]
    return doc


def _options(names: Sequence[str], variants: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each option with its values in the variants' order, a value available when a variant for sale has it."""
    order = list(names)
    for v in variants:
        order += [o["name"] for o in v.get("options") or () if o["name"] not in order]
    out = []
    for name in order:
        values: dict[str, bool] = {}
        for v in variants:
            for o in v.get("options") or ():
                if o["name"] == name:
                    values[o["label"]] = values.get(o["label"], False) or bool(v["availability"]["available"])
        if values:
            out.append({"name": name, "values": [{"label": k, "available": a} for k, a in values.items()]})
    return out


def _option_pairs(raw: Any) -> list[tuple[str, str]]:
    value = raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw) if raw.strip() else {}
        except ValueError:
            return []
    if not isinstance(value, Mapping):
        return []
    return [(str(k), str(v)) for k, v in value.items() if isinstance(v, str | int | float) and str(v).strip()]


def _raw_details(capture: CaptureV2, wanted: set[str], notes: list[str]) -> dict[str, dict[str, Any]]:
    """Description html and media (with alt text) per product id, from the raw Shopify feed pages and product .js
    documents the pass kept; a product .js wins over the feed, because it carries the media's alt text."""
    manifest_path = capture.path / RAW / "manifest.json"
    if not wanted or not manifest_path.is_file():
        return {}
    try:
        entries = _json_object(manifest_path).get("entries") or []
    except ValidationError:
        notes.append("raw manifest did not parse; descriptions and media left out")
        return {}
    out: dict[str, dict[str, Any]] = {}
    unread = 0
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, Mapping) or entry.get("status") != "ok" or not isinstance(entry.get("key"), str):
            continue
        kind = str(entry.get("kind") or "")
        feed, single = kind.startswith("feed-"), kind.startswith("product-") and kind.endswith("-js")
        if not (feed or single):
            continue
        data = _raw_json(capture.path / RAW / Path(entry["key"]).name)
        if data is None:
            unread += 1
            continue
        items = data.get("products") if feed else [data]
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, Mapping):
                continue
            pid = _text(item.get("id"))
            if pid is None or pid not in wanted:
                continue
            found = out.setdefault(pid, {})
            text = item.get("description") if single else item.get("body_html")
            if isinstance(text, str) and text.strip() and (single or "description" not in found):
                found["description"] = text
            media = _media(item.get("media") if single else item.get("images"))
            if media and (single or "media" not in found):
                found["media"] = media
    if unread:
        notes.append(f"{unread} raw pages could not be read")
    return out


def _raw_jsonld(capture: CaptureV2, wanted: set[str], notes: list[str]) -> dict[str, list[Any]]:
    """Each wanted product's JSON-LD blocks, from the product page the pass kept (raw entry kind product-<id>, HTML),
    with reviewers' names redacted; a product whose page carried none is left out."""
    manifest_path = capture.path / RAW / "manifest.json"
    if not wanted or not manifest_path.is_file():
        return {}
    try:
        entries = _json_object(manifest_path).get("entries") or []
    except ValidationError:
        notes.append("raw manifest did not parse; JSON-LD left out")
        return {}
    out: dict[str, list[Any]] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, Mapping) or entry.get("status") != "ok" or not isinstance(entry.get("key"), str):
            continue
        kind = str(entry.get("kind") or "")
        pid = kind.removeprefix("product-")
        html_page = "html" in str(entry.get("content_type") or "") or str(entry["key"]).endswith(".html.gz")
        if pid == kind or pid not in wanted or pid in out or not html_page:
            continue
        text = _raw_text(capture.path / RAW / Path(entry["key"]).name)
        blocks = page_blocks(text) if text is not None else []
        if blocks:
            out[pid] = without_reviewers(blocks)
    return out


def _raw_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rb") as handle:
            data = handle.read(RAW_MAX_BYTES + 1)
    except (OSError, EOFError):
        return None
    return None if len(data) > RAW_MAX_BYTES else data.decode("utf-8", errors="replace")


def _media(raw: Any) -> list[dict[str, Any]]:
    out = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, Mapping):
            continue
        url = _text(item.get("src")) or _text(item.get("url"))
        if url is None:
            continue
        if url.startswith("//"):
            url = "https:" + url
        media: dict[str, Any] = {"type": str(item.get("media_type") or "image"), "url": url}
        alt = _text(item.get("alt"))
        if alt:
            media["alt_text"] = alt
        out.append(media)
    return out


def _raw_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rb") as handle:
            data = handle.read(RAW_MAX_BYTES + 1)
        if len(data) > RAW_MAX_BYTES:
            return None
        value = json.loads(data)
    except (OSError, EOFError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def _currency(rows: Iterable[Mapping[str, Any]]) -> str | None:
    counts = Counter(
        str(r["currency"]) for r in rows if isinstance(r.get("currency"), str) and isinstance(r.get("price_minor"), int)
    )
    if not counts:
        return None
    return min(counts, key=lambda c: (-counts[c], c))


def _priced(row: Mapping[str, Any], currency: str) -> bool:
    price = row.get("price_minor")
    return (
        isinstance(price, int)
        and not isinstance(price, bool)
        and price >= 0
        and row.get("currency") == currency
        and _text(row.get("product_id")) is not None
    )


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidationError(f"{path} is not readable JSON: {exc}", hint="the capture pass rewrites it") from None
    if not isinstance(value, dict):
        raise ValidationError(f"{path} is not a JSON object")
    return value


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None

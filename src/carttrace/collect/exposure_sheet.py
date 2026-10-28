"""The store exposure sheet (D93, carttrace.trace.exposure): for one sandbox episode, what its store exposes, per data type of
the category's vocabulary and per surface of the episode, computed from the world and the surfaces' own renderers.

Why.The judge must tell three cases apart, and this module is where the sandbox says which:
- the store does not expose it (`absent` in the world, or `not_served` on the surface): a store fault, or not
  applicable;
- we hid it (`withheld`, `by_us`): an exposure mask (D63) or the detail_hidden condition (D47), designed;
- it was exposed (`exposed`, with the tools and fields that carried it) and the agent did not use it: the agent's.

How each column is read, never guessed:
- **The vocabulary** is the category's: the curated types of data/bench/union.yaml, the exposure mask's table
  (`exposure.TYPES`), the data elements of data/elements mapped to union ids (`ELEMENTS`), and every type of the
  category's corpus (var/bench/corpus/<category>.json, D76), or of its union (var/bench/union/<category>.json, D62)
  where no corpus was built. Keys resolve to ids through the union's own merge map and key reader
  (carttrace.agents.union_types, carttrace.agents.doc_keys), so a replica's `metadata.material` and a live reply's are one id.
- **The world** is the store's own pack before its mask: each product's facts read as the corpus reads a pack.
- **A surface** is read from what it renders for every product: UCP over MCP and REST from the served catalog's
  search listing, lookup and get_product documents with the root policies (the Medusa plugin's reshape applied where
  the platform is Medusa), the storefront (WebMCP) from the same listing and detail documents without policies, the
  replica website from each product page as served (its JSON-LD and text). The detail_hidden cut is the surface's own
  (`view_filter`), applied where the episode applies it. An ACP, MPP or AP2 row has no catalog of its own: its
  companion's column is the catalog, and its own column says it serves none.
- **Why not served:** the mask's withheld types are `withheld` (cause `mask`); a type the condition's cut removed is
  `withheld` (cause `detail_hidden`); a type the unshaped document carries and the platform's product view (D55)
  dropped is `not_served` (cause `platform_view`); anything else the world has and the surface does not carry is
  `not_served` (cause `surface`).
- **Checkout** is read from the served pack and its platform pack: handlers, the instruments the store takes, the
  market's shipping and pickup, the platform's policy links, the discount codes and gift cards. Reading only: the
  rails and wallets are other modules' (carttrace.collect.payment, carttrace.sandbox).

Deterministic: the same world, task, surfaces and vocabulary give the same bytes. Readings are cached per pack id, so
the many episodes of one world in a process read it once.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from carttrace.agents.doc_keys import HEADING, STRUCTURED, mark_document, mark_schema_nodes, read_text, served_document
from carttrace.agents.doc_keys import slug as key_slug
from carttrace.agents.spec import HiddenSpec
from carttrace.agents.union_types import TypeBook, load_types
from carttrace.collect.exposure import BY_ID, TYPES, TYPES_VERSION, carried
from carttrace.collect.payment import store_accepts
from carttrace.collect.surfaces import COMPANIONS, view_filter
from carttrace.collect.union_core import PROVENANCE as UNION_CORE
from carttrace.collect.union_core import origins as core_origins
from carttrace.core.canon import digest, loads, sha256_hex
from carttrace.pages.html import parse_page
from carttrace.protocols.profile import ProtocolProfile
from carttrace.sandbox.core.config import core_rules
from carttrace.sandbox.envpack import EnvPack
from carttrace.sandbox.platforms import Platform, platform_named
from carttrace.sandbox.platforms import medusa as medusa_pack
from carttrace.sandbox.presented import source_ids
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.site import page_path, product_page
from carttrace.sandbox.surfaces.ucp.catalog import POLICIES, Catalog, Index, ProductView, lookup_docs, served_catalog
from carttrace.sandbox.surfaces.ucp.operations import UcpOperations
from carttrace.trace.exposure import (
    Cause,
    Checkout,
    Coverage,
    ExposureSheet,
    Feature,
    LiveSeen,
    Products,
    Role,
    SurfaceCell,
    SurfaceInfo,
    TaskProduct,
    TypeRow,
    ViewUsed,
    Vocabulary,
    VocabularySource,
    Withholding,
)
from carttrace.trace.schema import EpisodeHeader, ExposureMask

READER: Final = "exposure-sheet-2"
"""Bump whenever what a reading reads changes."""
FIELDS: Final = 5
"""Field paths kept per type and surface: enough to point a judge at the place."""
REPO: Final = Path(__file__).resolve().parents[3]
ELEMENTS_FILE: Final = REPO / "data" / "elements" / "_elements.yaml"
ELEMENTS: Final[Mapping[str, str]] = {
    "reviews.rating": "reviews.rating",
    "reviews.text": "reviews.text",
    "questions.answers": "reviews.qa",
    "size.chart": "product.size_chart",
    "fit.guidance": "product.fit_notes",
    "variants.options": "variant.options",
    "variants.availability": "variant.availability",
    "price": "price.current",
    "price.unit": "price.unit",
    "identifiers.gtin": "identity.gtin",
    "identity.brand": "identity.brand",
    "images.alt": "media.alt_text",
    "shipping.cost": "delivery.cost",
    "delivery.estimate": "delivery.estimate",
    "returns.policy": "returns.policy",
    "faq": "store.faq",
    "agent_files.llms_txt": "store.llms_txt",
    "agent_files.agents_md": "store.agents_md",
    "agent_files.ucp_discovery": "store.ucp_discovery",
    "agent_files.robots": "store.robots",
    "agent_files.sitemap": "store.sitemap",
}
"""The shared data elements of the live data audit (data/elements/_elements.yaml, D19) and the union id each one is.
`reviews.verified`, `comparison` and `search` have no union type: they are computed by the audit, not shown."""
STORE_PREFIXES: Final = ("store.", "policy.", "agent.cart.", "store.payment-handler")
"""Type ids that are about the store, not a product: read from checkout, never counted over products."""
UCP_KINDS: Final = frozenset({"ucp_mcp", "ucp_rest"})
CATALOG_TOOLS: Final = ("search_catalog", "lookup_catalog", "get_product")
STOREFRONT_TOOLS: Final[Mapping[str, str]] = {"search_catalog": "search_products", "get_product": "get_product"}
"""The storefront's tools and the UCP call each one renders (carttrace.sandbox.surfaces.webmcp.operations)."""
WEBSITE_TOOL: Final = "read_product"
CHECKOUT_KINDS: Final = frozenset({"ucp_mcp", "ucp_rest", "webmcp", "acp_rest", "ap2", "mpp"})
LINK_KINDS: Final = frozenset({"ucp_mcp", "ucp_rest", "mpp"})
"""Kinds whose cart and checkout replies carry the platform's policy links: UCP's, and MPP's (it pays a UCP checkout).
The storefront's views drop `links` (webmcp/views.py DOC_ONLY), AP2's merchant sends none, and ACP sends its own
(`return_policy`, `seller_shop_policies`) whatever the platform pack."""
PICKUP_KINDS: Final = frozenset({"ucp_mcp", "ucp_rest", "webmcp", "ap2", "mpp"})
"""Kinds whose checkout offers the market's pickup: ACP ships only (acp/requests.py refuses pickup); MPP pays a UCP
checkout; AP2 offers pickup since D81."""
DISCOUNT_CAPABILITY: Final = "dev.ucp.shopping.discount"
GIFT_CARD: Final = re.compile(r"\bgift[ -]?cards?\b|\be-?gift\b", re.IGNORECASE)
ROLE_ORDER: Final[tuple[Role, ...]] = ("candidate", "witness")


# -- the vocabulary --------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VocabType:
    id: str
    name: str
    group: str
    origin: str
    element: str | None = None


@dataclass(frozen=True, slots=True)
class Vocab:
    """The category's types, where they came from, and what the live stores of the category showed of each."""

    id: str
    category: str | None
    types: Mapping[str, VocabType]
    sources: tuple[VocabularySource, ...]
    live: Mapping[str, Mapping[str, LiveSeen]] = field(default_factory=dict)
    """Store id to type id to what that live store showed (the union's per-store layers)."""

    def model(self) -> Vocabulary:
        return Vocabulary(id=self.id, category=self.category, types=len(self.types), sources=list(self.sources))


def vocabulary(category: str | None, var: Path | None, *, book: TypeBook | None = None) -> Vocab:
    """The vocabulary of `category`: curated union types, the mask's table, the mapped data elements, and the
    category's corpus (or union) types when `var` holds them. Cached by the files' sizes and times."""
    corpus = _file(var, "bench/corpus", category)
    union = _file(var, "bench/union", category)
    book = book or load_types()
    key = (category, corpus, _stamp(corpus), union, _stamp(union), book.version, len(book.types))
    found = _VOCABS.get(key)
    if found is None:
        if len(_VOCABS) >= _VOCABS_MAX:
            _VOCABS.clear()
        found = _VOCABS[key] = _vocabulary(category, corpus, union, book)
    return found


_VOCABS: dict[tuple[Any, ...], Vocab] = {}
_VOCABS_MAX: Final = 64


def _file(var: Path | None, folder: str, category: str | None) -> Path | None:
    if var is None or category is None:
        return None
    path = var / folder / f"{category}.json"
    return path if path.is_file() else None


def _stamp(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


def _vocabulary(category: str | None, corpus: Path | None, union: Path | None, book: TypeBook) -> Vocab:
    types: dict[str, VocabType] = {}
    for spec in book.types:
        types[spec.id] = VocabType(spec.id, spec.name, spec.group, "curated")
    for row in TYPES:
        if row.id not in types:
            name = f"{' '.join(row.kinds).replace('_', ' ')} (an exposure kind no union type curates yet)"
            types[row.id] = VocabType(row.id, name, "Product facts", "auto")
    sources = [
        VocabularySource(name="union.yaml", version=book.version, types=len(book.types)),
        VocabularySource(name="exposure-types", version=TYPES_VERSION, types=len(TYPES)),
        VocabularySource(name="elements", version=_sha(ELEMENTS_FILE), types=len(ELEMENTS)),
    ]
    live: dict[str, dict[str, LiveSeen]] = {}
    read_from = corpus or union
    if read_from is not None:
        raw = loads(read_from.read_bytes())
        entries = raw.get("entries") if corpus is not None else raw.get("types")
        kept = 0
        for entry in entries or ():
            if corpus is not None and entry.get("status") != "type":
                continue
            type_id = str(entry["id"])
            kept += 1
            if type_id not in types:
                origin = "curated" if book.spec(type_id) is not None else "auto"
                types[type_id] = VocabType(type_id, str(entry.get("name") or type_id), str(entry.get("group")), origin)
        folder = "corpus" if corpus is not None else "union"
        sources.append(VocabularySource(name=f"{folder}/{category}.json", version=_sha(read_from), types=kept))
    if union is not None:
        live = _live(loads(union.read_bytes()))
    for element, type_id in ELEMENTS.items():
        found = types.get(type_id)
        if found is not None:
            types[type_id] = VocabType(found.id, found.name, found.group, found.origin, element)
    ordered = dict(sorted(types.items()))
    vocab_id = "xv-" + digest([READER, [(s.name, s.version) for s in sources], list(ordered)], 12)
    return Vocab(vocab_id, category, ordered, tuple(sources), live)


def _live(union: Mapping[str, Any]) -> dict[str, dict[str, LiveSeen]]:
    out: dict[str, dict[str, LiveSeen]] = defaultdict(dict)
    for entry in union.get("types") or ():
        for per in entry.get("per_store") or ():
            layers = per.get("layers") or {}
            seen = LiveSeen(
                website=_share(layers.get("store_truth")),
                agent_view=_share(layers.get("agent_view")),
            )
            if seen.website is not None or seen.agent_view is not None:
                out[str(per.get("store_id"))][str(entry["id"])] = seen
    return dict(out)


def _share(cell: Any) -> float | None:
    if not isinstance(cell, Mapping):
        return None
    if cell.get("store_level"):
        return 1.0
    share = cell.get("share")
    return round(float(share), 4) if isinstance(share, int | float) and share > 0 else None


def _sha(path: Path) -> str:
    return sha256_hex(path.read_bytes())[:16] if path.is_file() else "missing"


# -- readings ---------------------------------------------------------------------------------------------------------

Found = dict[str, dict[str, set[str]]]
"""Type id to tool to the field paths that carried it."""


@dataclass(slots=True)
class Resolver:
    """Observed keys to type ids, through the union's merge map, kept to what the sheet reports: curated ids,
    vocabulary ids, and structured keys' own ids (a structured key is a type in the union's scheme; a free label or a
    heading is one only once the vocabulary has it)."""

    book: TypeBook
    vocab: Vocab
    cache: dict[tuple[str, str], str | None] = field(default_factory=dict)

    def __call__(self, key: str, kind: str) -> str | None:
        if (key, kind) in self.cache:
            return self.cache[(key, kind)]
        resolved = self.book.resolve(key, kind)
        keep: str | None = resolved.type_id
        if resolved.origin != "curated" and resolved.type_id not in self.vocab.types:
            keep = None if (resolved.free or resolved.heading or kind == HEADING) else resolved.type_id
        self.cache[(key, kind)] = keep
        return keep


def read_doc(
    doc: Mapping[str, Any], policies: Sequence[Mapping[str, Any]], tool: str, resolve: Resolver, out: Found
) -> None:
    """Every type one served product document carries, by the union's key reader and by the mask table's detectors,
    with the policies served beside it. A document read once is not read again: a world's episodes differ in a few
    variants' stock at most, so most documents repeat across them (`_DOCS`, keyed by content and vocabulary)."""
    key = digest([resolve.vocab.id, doc, policies], 24)
    found = _DOCS.get(key)
    if found is None:
        found = _read_doc(doc, policies, resolve)
        if len(_DOCS) >= _DOCS_MAX:
            _DOCS.clear()
        _DOCS[key] = found
    for type_id, paths in found.items():
        out.setdefault(type_id, {}).setdefault(tool, set()).update(paths)


_DOCS: dict[str, Mapping[str, frozenset[str]]] = {}
_DOCS_MAX: Final = 100_000


def _read_doc(
    doc: Mapping[str, Any], policies: Sequence[Mapping[str, Any]], resolve: Resolver
) -> Mapping[str, frozenset[str]]:
    pid = str(doc.get("id") or "")
    out: dict[str, set[str]] = {}

    def mark(key: str, kind: str, path: str, _value: Any, **_: Any) -> None:
        type_id = resolve(key, kind)
        if type_id is not None:
            out.setdefault(type_id, set()).add(path)

    # a surface's metadata groups (`details`, `attributes`) are read as the world's are, each entry a label of its own
    # (D117: the surface columns had read `metadata.details` as one label and missed every detail it served)
    mark_document(served_document(doc), "ucp", pid, mark)
    for policy in policies:
        fact = policy.get("fact")
        if isinstance(fact, str) and fact:
            mark(f"policy:{key_slug(fact)}", STRUCTURED, f"policies[].{fact}", None)
    for type_id in sorted(carried(doc, policies, TYPES)):
        out.setdefault(type_id, set()).add(f"detector {' '.join(BY_ID[type_id].kinds)}")
    return {t: frozenset(paths) for t, paths in out.items()}


def read_page(html: str, url: str, tool: str, resolve: Resolver, out: Found) -> None:
    """A product page as served: its schema.org nodes and its text's labels, as the union reads a page."""
    facts = parse_page(html, url)

    def mark(key: str, kind: str, path: str, _value: Any, **_: Any) -> None:
        type_id = resolve(key, kind)
        if type_id is not None:
            out.setdefault(type_id, {}).setdefault(tool, set()).add(path)

    mark_schema_nodes([*facts.jsonld, *facts.microdata], mark)
    read_text(facts.main_text.split("\n"), lambda key, kind, path, value: mark(key, kind, "main text", value))


@dataclass(frozen=True, slots=True)
class WorldRead:
    """Per product, what the world's facts carry, and per surface family what each tool renders for it."""

    products: tuple[str, ...]
    world: Mapping[str, Found]
    ucp: Mapping[str, Found]
    ucp_unviewed: Mapping[str, Found] | None
    """What UCP would render without the platform's product view (D55); None for a platform without one."""
    storefront_unviewed: Mapping[str, Found] | None
    ucp_hidden: Mapping[str, Found] | None
    storefront: Mapping[str, Found]
    storefront_hidden: Mapping[str, Found] | None
    website: Mapping[str, Found]
    tools: frozenset[str]
    """The catalog tools the store's UCP surface lists."""
    variants: Mapping[str, str]
    """Variant id to product id."""
    gift_cards: tuple[str, ...]
    core_values: Mapping[str, Mapping[str, Mapping[str, str]]] = field(default_factory=dict)
    """Product id to its recorded union core values (D117, carttrace.collect.union_core.origins)."""
    cored: bool = False
    """Whether the world was built with a union core (its provenance names one)."""


def read_world(
    served: EnvPack,
    base: EnvPack,
    profile: ProtocolProfile | None,
    resolve: Resolver,
    *,
    hide: Sequence[str] = (),
    details: Sequence[str] = (),
    kinds: Sequence[str] = ("ucp_mcp",),
) -> WorldRead:
    """Read the world (`base`, before any mask) and what the surface families of `kinds` render of `served`: UCP
    always (every catalog companion is UCP), the storefront and the website only when the episode mounts them."""
    families = frozenset(k for k in kinds if k in ("webmcp", "website"))
    key = (
        served.id,
        base.id,
        None if profile is None else profile.id,
        tuple(hide),
        tuple(details),
        resolve.vocab.id,
        families,
    )
    found = _READS.get(key)
    if found is None:
        found = _read_world(served, base, profile, resolve, hide=hide, details=details, families=families)
        if len(_READS) >= _READS_MAX:
            _READS.clear()
        _READS[key] = found
    return found


_READS: dict[tuple[Any, ...], WorldRead] = {}
_READS_MAX: Final = 32


def _read_world(
    served: EnvPack,
    base: EnvPack,
    profile: ProtocolProfile | None,
    resolve: Resolver,
    *,
    hide: Sequence[str],
    details: Sequence[str],
    families: frozenset[str],
) -> WorldRead:
    world: dict[str, Found] = {}
    gift_cards: list[str] = []
    for pid, facts in sorted(base.snapshot.products.items()):
        out: Found = {}
        policies = [p for p in facts.get(POLICIES) or () if isinstance(p, Mapping)]
        read_doc({"id": pid, **served_document(facts)}, policies, "world", resolve, out)
        world[pid] = out
        if GIFT_CARD.search(f"{facts.get('title') or ''} {facts.get('product_type') or ''}"):
            gift_cards.append(pid)
    host = SandboxHost(served, profile=profile)
    platform = host.platform
    if host.profile.ucp is not None:
        ops = UcpOperations(host)
        tools = frozenset(t.name for t in ops.tools() if t.name in CATALOG_TOOLS)
        catalog = ops.catalog
    else:
        tools, catalog = frozenset(CATALOG_TOOLS), served_catalog(host.pack, platform)
    state = host.pack.initial_state()
    products = catalog.products(state)
    cut = view_filter(hide, details)
    ucp = _ucp_reads(catalog, state, products, platform, resolve)
    unviewed = storefront_unviewed = None
    if platform.product_views or host.pack.agent_view is not None:
        plain = served_catalog(host.pack, _unviewed(platform), viewed=False)
        unviewed = _ucp_reads(plain, state, products, platform, resolve)
        if "webmcp" in families:
            storefront_unviewed = _storefront_reads(plain, products, platform, resolve)
    hidden = None if cut is None else _ucp_reads(catalog, state, products, platform, resolve, cut=cut)
    storefront: dict[str, Found] = {}
    storefront_hidden = None
    if "webmcp" in families:
        storefront = _storefront_reads(catalog, products, platform, resolve)
        storefront_hidden = None if cut is None else _storefront_reads(catalog, products, platform, resolve, cut=cut)
    website: dict[str, Found] = {}
    for product in products if "website" in families else ():
        out = {}
        read_page(product_page(product, page_path(product)), page_path(product), WEBSITE_TOOL, resolve, out)
        website[product.id] = out
    variants = {v.id: p.id for p in products for v in p.variants}
    return WorldRead(
        products=tuple(sorted(world)),
        world=world,
        ucp=ucp,
        ucp_unviewed=unviewed,
        storefront_unviewed=storefront_unviewed,
        ucp_hidden=hidden,
        storefront=storefront,
        storefront_hidden=storefront_hidden,
        website=website,
        tools=tools,
        variants=variants,
        gift_cards=tuple(gift_cards),
        core_values={pid: core_origins(f) for pid, f in sorted(base.snapshot.products.items())},
        cored=UNION_CORE in (base.doc.get("provenance") or {}),
    )


def _unviewed(platform: Platform) -> Platform:
    """The platform without its product view, so a type the view dropped can be told from one the world lacks."""
    return replace(platform, product_views={})


Cut = Any
"""A detail filter: (document, counts) to the document without the hidden details."""


def _documents(
    catalog: Catalog, state: Any, product: ProductView, platform: Platform, index: Index
) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    """(tool, product document, root policies) for one product, as the UCP tools render them."""
    policies = catalog.policies([("$.product", product)])
    found, _ = lookup_docs(catalog, state, [product.id], index=index)
    docs = [
        ("search_catalog", catalog.listing(product)),
        ("lookup_catalog", found[0] if found else None),
        ("get_product", catalog.detail({"id": product.id}, product, None)),
    ]
    out: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for tool, doc in docs:
        if doc is None:
            continue
        if platform.dialect == "medusa":
            out.append((tool, medusa_pack.product(doc), []))
        else:
            out.append((tool, doc, policies))
    return out


def _ucp_reads(
    catalog: Catalog,
    state: Any,
    products: Sequence[ProductView],
    platform: Platform,
    resolve: Resolver,
    *,
    cut: Cut | None = None,
) -> dict[str, Found]:
    out: dict[str, Found] = {}
    index = Index(products)
    for product in products:
        found: Found = {}
        for tool, rendered, served_policies in _documents(catalog, state, product, platform, index):
            doc: Mapping[str, Any] = rendered
            policies: Sequence[Mapping[str, Any]] = served_policies
            if cut is not None:
                reply = cut({"products": [doc], "policies": policies}, Counter())
                doc = (reply.get("products") or [{}])[0]
                policies = [p for p in reply.get("policies") or () if isinstance(p, Mapping)]
            read_doc(doc, policies, tool, resolve, found)
        out[product.id] = found
    return out


def _storefront_reads(
    catalog: Catalog, products: Sequence[ProductView], platform: Platform, resolve: Resolver, *, cut: Cut | None = None
) -> dict[str, Found]:
    """The storefront's search_products and get_product documents: the UCP listing and detail, no root policies
    (its views drop them), its own server-side cut applied to each document."""
    out: dict[str, Found] = {}
    for product in products:
        found: Found = {}
        docs = {
            "search_catalog": catalog.listing(product),
            "get_product": catalog.detail({"id": product.id}, product, None),
        }
        for ucp_tool, rendered in docs.items():
            doc = medusa_pack.product(rendered) if platform.dialect == "medusa" else rendered
            if cut is not None:
                doc = cut(doc, Counter())
            read_doc(doc, (), STOREFRONT_TOOLS[ucp_tool], resolve, found)
        out[product.id] = found
    return out


# -- the sheet --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SheetInputs:
    """What one episode's sheet is computed from."""

    header: EpisodeHeader
    served: EnvPack
    """The pack the surfaces serve (masked, faulted, as the episode runs it)."""
    base: EnvPack
    """The store's own world: the same pack before its exposure mask."""
    profile: ProtocolProfile | None
    var: Path | None
    row_kind: str
    hide: tuple[str, ...] = ()
    details: tuple[str, ...] = ()


def build_sheet(inputs: SheetInputs, *, book: TypeBook | None = None) -> ExposureSheet:
    """The episode's exposure sheet."""
    header = inputs.header
    book = book or load_types()
    vocab = vocabulary(header.store.category, inputs.var, book=book)
    resolve = Resolver(book, vocab)
    kinds = _kinds(header, inputs.row_kind)
    read = read_world(
        inputs.served, inputs.base, inputs.profile, resolve, hide=inputs.hide, details=inputs.details, kinds=kinds
    )
    filtered = _filtered_kind(inputs.row_kind) if (inputs.hide or inputs.details) else None
    spec = HiddenSpec.from_mapping(header.task.hidden_spec) if header.task.hidden_spec else None
    candidates = sorted(set(spec.candidates)) if spec is not None else []
    witness_products = sorted({read.variants.get(v, v) for v in (spec.witnesses if spec is not None else ())})
    mask = header.exposure
    masked = _masked_ids(mask)
    platform = platform_named(inputs.served.platform_name)
    checkout = _checkout(inputs, kinds, platform, read)
    typed = {f.id: set(_feature_types(f, book)) for f in checkout}
    rows: list[TypeRow] = []
    seen_live = vocab.live.get(header.store.store_id, {})
    ids = sorted(
        {
            *_ids(read.world),
            *(t for m in _families(read) for t in _ids(m)),
            *masked,
            *seen_live,
            *(t for ids in typed.values() for t in ids),
        }
    )
    for type_id in ids:
        row = _row(
            type_id,
            vocab=vocab,
            book=book,
            read=read,
            kinds=kinds,
            filtered=filtered,
            candidates=set(candidates),
            witnesses=set(witness_products),
            masked=masked,
            mask=mask,
            details=inputs.details,
            hide=inputs.hide,
            live=seen_live.get(type_id),
            features=[f for f in checkout if type_id in typed[f.id]],
        )
        rows.append(row)
    present = {r.id for r in rows}
    task_products = _task_products(candidates, witness_products, read, kinds, filtered)
    return ExposureSheet(
        episode_id=header.episode_id,
        store_id=header.store.store_id,
        category=header.store.category,
        envpack=inputs.served.id,
        world=inputs.base.id,
        platform_pack=header.platform_pack,
        agent_view=_view_used(inputs.served),
        presented_ids=source_ids(inputs.served),
        mask=None if mask is None else mask.mask,
        detail_hidden=sorted({*inputs.details, *inputs.hide}),
        vocabulary=vocab.model(),
        surfaces=[_surface_info(k, header, read) for k in kinds],
        products=Products(all=len(read.products), candidates=candidates, witnesses=witness_products),
        types=rows,
        absent=[t for t in vocab.types if t not in present],
        task_products=task_products,
        checkout=Checkout(rail=header.rail, features=checkout),
    )


def _view_used(pack: EnvPack) -> ViewUsed | None:
    """The recorded agent view the pack's catalog documents are served in (D100), or None."""
    view = pack.agent_view
    if view is None:
        return None
    return ViewUsed(
        ref=view.ref, template=view.template, store=view.store, own=list(view.own), source_ids=dict(view.ids)
    )


def _kinds(header: EpisodeHeader, row_kind: str) -> list[str]:
    """Every surface kind of the episode, the row's own first."""
    found = [row_kind, *(s.kind for s in header.surfaces)]
    return list(dict.fromkeys(k for k in found if k))


def _filtered_kind(row_kind: str) -> str | None:
    """The surface the detail_hidden cut applies to on this row (carttrace.collect.episode._agent_surfaces): a storefront
    cuts server-side, an ACP, MPP or AP2 row's companion catalog is wrapped, UCP rows are wrapped; a website row
    never takes the condition."""
    if row_kind in COMPANIONS:
        return COMPANIONS[row_kind]
    if row_kind in (*UCP_KINDS, "webmcp"):
        return row_kind
    return None


def _masked_ids(mask: ExposureMask | None) -> set[str]:
    """Every union id the mask withholds: each withheld row and the curated ids it withholds with it."""
    if mask is None:
        return set()
    out: set[str] = set()
    for type_id in mask.withheld:
        row = BY_ID.get(type_id)
        out.update(row.union if row is not None and row.union else (type_id,))
    return out


def _ids(found: Mapping[str, Found]) -> set[str]:
    return {t for per in found.values() for t in per}


def _families(read: WorldRead) -> list[Mapping[str, Found]]:
    return [read.ucp, read.storefront, read.website]


def _feature_types(feature: Feature, book: TypeBook) -> list[str]:
    """The union type ids a checkout feature is (its link types, its handler namespaces), resolved as the union
    resolves a cart reply's keys."""
    if feature.id == "policy.links":
        return sorted({book.resolve(f"cart:links[].type={key_slug(v)}", STRUCTURED).type_id for v in feature.values})
    if feature.id == "payment.handlers":
        return sorted({book.resolve(f"payment_handler:{key_slug(v)}", STRUCTURED).type_id for v in feature.values})
    return []


def _surface_reading(read: WorldRead, kind: str, filtered: str | None) -> Mapping[str, Found] | None:
    if kind in UCP_KINDS:
        return read.ucp_hidden if filtered == kind and read.ucp_hidden is not None else read.ucp
    if kind == "webmcp":
        return read.storefront_hidden if filtered == kind and read.storefront_hidden is not None else read.storefront
    if kind == "website":
        return read.website
    return None


def _unfiltered(read: WorldRead, kind: str) -> Mapping[str, Found] | None:
    if kind in UCP_KINDS:
        return read.ucp
    if kind == "webmcp":
        return read.storefront
    return None


def _coverage(found: Mapping[str, Found], type_id: str, candidates: set[str], witnesses: set[str]) -> Coverage:
    having = {pid for pid, per in found.items() if type_id in per}
    return Coverage(products=len(having), candidates=len(having & candidates), witnesses=len(having & witnesses))


def _row(
    type_id: str,
    *,
    vocab: Vocab,
    book: TypeBook,
    read: WorldRead,
    kinds: Sequence[str],
    filtered: str | None,
    candidates: set[str],
    witnesses: set[str],
    masked: set[str],
    mask: ExposureMask | None,
    details: Sequence[str],
    hide: Sequence[str],
    live: LiveSeen | None,
    features: Sequence[Feature],
) -> TypeRow:
    known = vocab.types.get(type_id)
    spec = book.spec(type_id)
    name = known.name if known else (spec.name if spec else type_id)
    group = known.group if known else (spec.group if spec else "Other")
    world = _coverage(read.world, type_id, candidates, witnesses)
    store_level = type_id.startswith(STORE_PREFIXES) and not world.products
    if type_id in masked and mask is not None and type_id in mask.counts and world.products == 0:
        world = Coverage(products=mask.counts[type_id].before)
    withheld: Withholding | None = None
    if type_id in masked and mask is not None:
        withheld = Withholding(by="mask", mask=mask.mask)
    cells: dict[str, SurfaceCell] = {}
    for kind in kinds:
        if store_level or (features and not world.products):
            cells[kind] = _store_cell(kind, features)
            continue
        found = _surface_reading(read, kind, filtered)
        if found is None:
            status = "not_served" if world.products else "absent"
            cells[kind] = SurfaceCell(status=status, cause="surface" if world.products else None)
            continue
        cover = _coverage(found, type_id, candidates, witnesses)
        tools = sorted({t for per in found.values() for t in per.get(type_id, {})})
        plain = _unfiltered(read, kind) if filtered == kind else None
        cut_from = _cut_from(plain, found, type_id, candidates, witnesses)
        thinned = None
        if mask is not None and type_id in mask.collateral:
            thinned = _cut_from(read.world, found, type_id, candidates, witnesses)
        if cut_from is not None:
            withheld = withheld or Withholding(by="detail_hidden", details=sorted({*details, *hide}))
        if cover.products:
            paths = sorted({p for per in found.values() for f in per.get(type_id, {}).values() for p in f})
            cause: Cause | None = "detail_hidden" if cut_from is not None else "mask" if thinned else None
            cells[kind] = SurfaceCell(
                status="exposed",
                cause=cause,
                coverage=cover,
                withheld_from=cut_from or thinned,
                tools=tools,
                fields=paths[:FIELDS],
            )
            continue
        if not world.products:
            cells[kind] = SurfaceCell(status="absent", coverage=cover)
            continue
        if type_id in masked:
            cells[kind] = SurfaceCell(status="withheld", by_us=True, cause="mask", coverage=cover, withheld_from=world)
            continue
        if cut_from is not None:
            cells[kind] = SurfaceCell(
                status="withheld", by_us=True, cause="detail_hidden", coverage=cover, withheld_from=cut_from
            )
            continue
        unviewed = read.ucp_unviewed if kind in UCP_KINDS else read.storefront_unviewed if kind == "webmcp" else None
        if unviewed is not None and _ids_of(unviewed, type_id):
            cells[kind] = SurfaceCell(status="not_served", cause="platform_view", coverage=cover)
            continue
        cells[kind] = SurfaceCell(status="not_served", cause="surface", coverage=cover)
    layer, value_origins = _layer(type_id, read, world)
    return TypeRow(
        id=type_id,
        name=name,
        group=group,
        scope="store" if store_level else "product",
        origin="curated" if spec is not None else "auto",
        element=known.element if known else None,
        world=world,
        live=live,
        withheld=withheld,
        surfaces=cells,
        layer=layer,
        value_origins=value_origins,
    )


def _layer(type_id: str, read: WorldRead, world: Coverage) -> tuple[Any, dict[str, int]]:
    """A stand-in's union layer of the type and its values' origins over the world's products (D117), from each
    product's recorded core values; (None, {}) for a world built without the core."""
    if not read.cored:
        return None, {}
    counts: Counter[str] = Counter()
    layers: set[str] = set()
    for record in read.core_values.values():
        origins = {v["origin"] for v in record.values() if v.get("type") == type_id}
        layers |= {v["layer"] for v in record.values() if v.get("type") == type_id}
        if origins:
            counts[sorted(origins)[0]] += 1
    if not counts:
        return ("base" if world.products else None), {}
    return ("experimental" if "experimental" in layers else "core"), dict(sorted(counts.items()))


def _cut_from(
    plain: Mapping[str, Found] | None,
    found: Mapping[str, Found],
    type_id: str,
    candidates: set[str],
    witnesses: set[str],
) -> Coverage | None:
    """The products the detail_hidden cut removed the type from on a filtered surface, or None when it removed it
    from none (or the surface is not filtered)."""
    if plain is None:
        return None
    gone = {pid for pid, per in plain.items() if type_id in per and type_id not in found.get(pid, {})}
    if not gone:
        return None
    return Coverage(products=len(gone), candidates=len(gone & candidates), witnesses=len(gone & witnesses))


def _ids_of(found: Mapping[str, Found], type_id: str) -> bool:
    return any(type_id in per for per in found.values())


def _store_cell(kind: str, features: Sequence[Feature]) -> SurfaceCell:
    """A store-level type on one surface: exposed where a checkout feature that is it reaches the agent."""
    on = [f for f in features if f.surfaces.get(kind)]
    if on:
        return SurfaceCell(status="exposed", tools=["checkout"], fields=sorted({_feature_field(f) for f in on}))
    if any(f.offered for f in features):
        return SurfaceCell(status="not_served", cause="surface")
    return SurfaceCell(status="absent")


def _feature_field(feature: Feature) -> str:
    return {"policy.links": "links[]", "payment.handlers": "ucp.payment_handlers", "discount.codes": "discounts"}.get(
        feature.id, feature.id
    )


def _task_products(
    candidates: Sequence[str], witnesses: Sequence[str], read: WorldRead, kinds: Sequence[str], filtered: str | None
) -> list[TaskProduct]:
    out: list[TaskProduct] = []
    for pid in sorted({*candidates, *witnesses}):
        roles = [r for r in ROLE_ORDER if pid in (candidates if r == "candidate" else witnesses)]
        surfaces: dict[str, list[str]] = {}
        for kind in kinds:
            found = _surface_reading(read, kind, filtered)
            if found is not None:
                surfaces[kind] = sorted(found.get(pid, {}))
        out.append(TaskProduct(id=pid, roles=roles, world=sorted(read.world.get(pid, {})), surfaces=surfaces))
    return out


def _surface_info(kind: str, header: EpisodeHeader, read: WorldRead) -> SurfaceInfo:
    version = next((s.version for s in header.surfaces if s.kind == kind), None)
    if kind in UCP_KINDS:
        tools = sorted(read.tools)
    elif kind == "webmcp":
        tools = sorted(STOREFRONT_TOOLS.values())
    elif kind == "website":
        tools = [WEBSITE_TOOL]
    else:
        tools = []
    return SurfaceInfo(kind=kind, version=version, product_tools=tools, catalog_from=COMPANIONS.get(kind))


# -- checkout ---------------------------------------------------------------------------------------------------------


def _checkout(inputs: SheetInputs, kinds: Sequence[str], platform: Platform, read: WorldRead) -> list[Feature]:
    """What checkout exposes on this world, read from the served pack and its platform pack."""
    pack = inputs.served
    header = inputs.header
    checkout_kinds = [k for k in kinds if k in CHECKOUT_KINDS]
    declared = set(pack.payment_handlers)
    handlers = sorted(set(pack.handlers().values()))
    ours = sorted(set(handlers) - declared)
    on = dict.fromkeys(checkout_kinds, bool(handlers))
    features = [
        Feature(
            id="payment.handlers",
            offered=bool(handlers),
            surfaces=on,
            values=handlers,
            by_us=bool(ours),
            note=f"served by the sandbox, not declared by the store: {', '.join(ours)}" if ours else None,
        )
    ]
    plan = header.payment
    accepts = list(plan.store_accepts) if plan is not None else list(store_accepts(header.rail, handlers))
    wallet = [w.kind for w in plan.wallet] if plan is not None else []
    features.append(
        Feature(
            id="payment.wallets",
            offered=bool(accepts),
            surfaces=dict.fromkeys(checkout_kinds, bool(accepts)),
            values=accepts,
            note=f"the persona's wallet holds {', '.join(sorted(set(wallet)))}" if wallet else None,
        )
    )
    market = pack.market()
    shipping = [o.id for o in market.shipping]
    features.append(
        Feature(
            id="fulfilment.shipping",
            offered=bool(shipping),
            platform_supports=True,
            surfaces=dict.fromkeys(checkout_kinds, bool(shipping)),
            values=shipping,
        )
    )
    pickup = [p.id for p in market.pickup]
    medusa = platform.dialect == "medusa"
    features.append(
        Feature(
            id="fulfilment.pickup",
            offered=bool(pickup),
            platform_supports=None if medusa else True,
            surfaces={k: bool(pickup) and k in PICKUP_KINDS for k in checkout_kinds},
            values=pickup,
            note="Medusa's plugin shapes one shipping method; pickup there is unverified" if medusa else None,
        )
    )
    links = [kind for kind, _ in platform.links]
    features.append(
        Feature(
            id="policy.links",
            offered=bool(links),
            platform_supports=bool(platform.links),
            surfaces={k: k == "acp_rest" or (bool(links) and k in LINK_KINDS) for k in checkout_kinds},
            values=links,
            note=None if links else f"the {platform.name} pack serves no policy links on carts and checkouts",
        )
    )
    config = pack.config()
    codes = sorted({*config.discount_codes, *config.coupons})
    supports = DISCOUNT_CAPABILITY not in platform.unserved
    rules = core_rules(pack.platform_name)
    stacks = config.stack_discounts if rules.stack_codes is None else rules.stack_codes
    features.append(
        Feature(
            id="discount.codes",
            offered=bool(codes),
            platform_supports=supports,
            surfaces={k: bool(codes) and (supports or k not in UCP_KINDS) for k in checkout_kinds},
            values=codes,
            note=f"codes {'stack' if stacks else 'do not stack'} on {pack.platform_name or 'this platform'}",
        )
    )
    features.append(
        Feature(
            id="gift_card.redeem",
            offered=False,
            platform_supports=False,
            surfaces=dict.fromkeys(checkout_kinds, False),
            by_us=True,
            note="the sandbox core models no gift card redemption at checkout on any platform pack",
        )
    )
    features.append(
        Feature(
            id="gift.message",
            offered=False,
            platform_supports=False,
            surfaces=dict.fromkeys(checkout_kinds, False),
            by_us=True,
            note="the sandbox core models no gift message, gift note or gift marking at checkout on any platform pack",
        )
    )
    catalog_kinds = [k for k in kinds if _surface_reading(read, k, None) is not None]
    features.append(
        Feature(
            id="gift_card.products",
            offered=bool(read.gift_cards),
            platform_supports=True,
            surfaces={k: bool(read.gift_cards) for k in catalog_kinds},
            values=list(read.gift_cards),
            note=None if read.gift_cards else "no product of the world is a gift card",
        )
    )
    return features


__all__ = [
    "ELEMENTS",
    "READER",
    "SheetInputs",
    "Vocab",
    "build_sheet",
    "read_world",
    "vocabulary",
]

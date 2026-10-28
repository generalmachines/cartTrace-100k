"""Sandbox store snapshots cut from a catalog: one store, one category, a seeded sample of its products.

Why a snapshot and not the whole catalog. A sandbox world is an envpack (`carttrace.sandbox.envpack`, mc.envpack.v2): a
store, its products and variants, a market, and provenance. `build_store_snapshot(source, category, seed=, size=)`
picks one store of the category (a Lazada seller, an ABO brand, a WANDS partition) and `size` of its products, both
with `carttrace.core.ids.seeded_rng`, and returns exactly the arguments `build_envpack` takes. Same inputs, same bytes: the
same catalog, category, seed and size always give the same pack id.

Why the licence travels in the pack. Decision D10: every bundle carries its data rights. The snapshot's provenance
holds the catalog (source, category, store, seed, size, the raw files' sha256, the content hash of the index it was
cut from), the source's licence and the `Rights` it grants; `envpack_rights(pack)` reads them back for the bundle
header, and gives the stricter of the recorded rights and the source's current licence, so a licence tightened later
wins and a pack without rights grants nothing. A caller's own provenance can be added to a pack but never replaces
these keys (`PROVENANCE_KEYS`).

A replica of a real store (`carttrace.sandbox.catalogs.replica`, decision D12) is a StoreSnapshot too, of source `replica`:
it also carries the store's captured ProtocolProfile and payment handlers, the catalog answers the store gave during
its capture (`recorded`), and the capture's time, all of which `envpack_kwargs` passes on only when set, so a catalog
snapshot builds exactly the pack it always did. Its provenance names the capture (`capture`) and its rights, a live
store's.
"""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from carttrace.config.schema import Settings
from carttrace.core.errors import MCError, NotFoundError
from carttrace.core.ids import seeded_rng
from carttrace.protocols.profile import ProtocolProfile
from carttrace.sandbox.catalogs.categories import known_categories
from carttrace.sandbox.catalogs.errors import CatalogError, NotCoveredError
from carttrace.sandbox.catalogs.index import ensure_index
from carttrace.sandbox.catalogs.licence import licence_rights, rights_for, strictest
from carttrace.sandbox.catalogs.records import ProductRecord, VariantRecord, product_facts
from carttrace.sandbox.catalogs.settings import CatalogSettings, resolve
from carttrace.sandbox.catalogs.source import open_source
from carttrace.sandbox.catalogs.text import slug
from carttrace.sandbox.envpack import EnvPack, build_envpack
from carttrace.sandbox.envpack.doc import MarketDoc
from carttrace.trace.schema import Rights, StoreRef

SNAPSHOT_SCHEMA: Final = "mc.catalog-snapshot.v1"
PROVENANCE_KEYS = ("catalog", "capture", "licence", "rights")
"""Keys of an envpack's provenance that a catalog snapshot (or a replica) owns; a caller's provenance never replaces
them."""


class StoreSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    products: int


class StoreSnapshot(BaseModel):
    """A sandbox store cut from a catalog, as the arguments of `carttrace.sandbox.build_envpack`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_: Literal["mc.catalog-snapshot.v1"] = Field(default=SNAPSHOT_SCHEMA, alias="schema")
    source: str
    category: str
    store_key: str
    seed: int
    size: int
    store: dict[str, str]
    """The envpack StoreDoc: domain (under .invalid for a catalog store; the store's own host for a replica, which is
    a copy of that store and is only ever served in process), store_id, platform, category."""
    currency: str
    market: MarketDoc
    variants: list[dict[str, Any]]
    products: dict[str, dict[str, Any]]
    provenance: dict[str, Any]
    rights: Rights
    profile: ProtocolProfile | None = None
    """A replica's: the versions the store spoke when it was captured."""
    payment_handlers: dict[str, Any] = Field(default_factory=dict)
    """A replica's: the payment handlers the store declared, as declared."""
    recorded: list[dict[str, Any]] = Field(default_factory=list)
    """A replica's: the catalog answers the store gave during its capture (envpack `RecordedAnswer` fields)."""
    tool_lists: list[dict[str, Any]] = Field(default_factory=list)
    """A replica's: the store's own tools/list replies as recorded live (envpack `RecordedToolList` fields, D40)."""
    captured_at: str | None = None
    snapshot_source: str | None = None
    """Where the snapshot was read, when it is not `catalog:<source>` (a replica's is `capture:<capture id>`)."""

    def envpack_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for `build_envpack`; add profile, tasks, faults, start or policy as needed. A replica's
        profile, payment handlers, recorded answers, recorded tool lists and capture time come with it."""
        out: dict[str, Any] = {
            "store": dict(self.store),
            "variants": [dict(v) for v in self.variants],
            "currency": self.currency,
            "products": {k: dict(v) for k, v in self.products.items()},
            "market": self.market.to_market(),
            "snapshot_source": self.snapshot_source or f"catalog:{self.source}",
            "provenance": dict(self.provenance),
        }
        if self.profile is not None:
            out["profile"] = self.profile
        if self.payment_handlers:
            out["payment_handlers"] = dict(self.payment_handlers)
        if self.recorded:
            out["recorded"] = [dict(a) for a in self.recorded]
        if self.tool_lists:
            out["tool_lists"] = [dict(t) for t in self.tool_lists]
        if self.captured_at is not None:
            out["captured_at"] = self.captured_at
        return out

    def to_envpack(self, **extra: Any) -> EnvPack:
        """The sealed envpack. `extra` adds build_envpack arguments (profile, tasks, faults, start, policy, ...);
        an extra `provenance` is merged under the catalog's own keys."""
        kwargs = self.envpack_kwargs()
        theirs = dict(extra.pop("provenance", None) or {})
        clash = sorted(k for k in extra if k in ("store", "variants", "currency", "products", "snapshot_source"))
        if clash:
            raise CatalogError(f"cannot replace {clash} of a catalog snapshot", hint="build another snapshot")
        kwargs["provenance"] = {**{k: v for k, v in theirs.items() if k not in PROVENANCE_KEYS}, **kwargs["provenance"]}
        return build_envpack(**{**kwargs, **extra})

    def store_ref(self) -> StoreRef:
        """The trace StoreRef for episodes on this store."""
        return StoreRef(
            store_id=self.store["store_id"],
            domain=self.store["domain"],
            category=self.category,
            platform=self.store.get("platform") or "sandbox",
            market=self.market.code,
        )


def catalog_stores(
    source_name: str, category_slug: str, *, settings: Settings | CatalogSettings | None = None
) -> list[StoreSummary]:
    """The stores a source offers in a category, largest first (ties by key). A restricted source's sellers are
    listed only with the research allowance."""
    source = open_source(source_name, settings=settings)
    source.require_allowed()
    _check_category(source_name, category_slug, settings)
    stores = ensure_index(source).stores(category_slug)
    rule = source.manifest.stores
    floor = rule.min_products if rule else 1
    offered = [StoreSummary(key=k, products=n) for k, n in stores.items() if n >= floor]
    return sorted(offered, key=lambda s: (-s.products, s.key))


def build_store_snapshot(
    source_name: str,
    category_slug: str,
    *,
    seed: int,
    size: int,
    store: str | None = None,
    settings: Settings | CatalogSettings | None = None,
) -> StoreSnapshot:
    """One sandbox store of `category_slug` from `source_name`: a seeded store (or `store`) and a seeded sample of at
    most `size` of its products, ready for `build_envpack`.

    Raises RestrictedCatalogError for a restricted source without the research allowance, NotCoveredError when the
    source has no store in the category, and NotFetchedError when the source was never fetched.
    """
    if size < 1:
        raise NotFoundError(f"a store needs at least one product, got size={size}", hint="pass size >= 1")
    cfg = resolve(settings)
    source = open_source(source_name, settings=cfg)
    source.require_allowed()
    _check_category(source_name, category_slug, cfg)
    if category_slug not in source.categories.categories():
        raise NotCoveredError(
            f"{source_name} maps nothing to {category_slug!r}", hint=f"its categories: {source.categories.categories()}"
        )
    index = ensure_index(source)
    stores = index.stores(category_slug)
    rule = source.manifest.stores
    floor = rule.min_products if rule else 1
    offered = {k: n for k, n in stores.items() if n >= floor} or stores
    if store is None:
        chosen = _choose_store(source_name, category_slug, offered, seed=seed, size=size)
    elif store in stores:
        chosen = store
    else:
        largest = sorted(offered, key=lambda k: (-offered[k], k))[:5]
        raise NotCoveredError(
            f"{source_name} has no store {store!r} in {category_slug!r}", hint=f"largest stores: {largest}"
        )
    products = index.products(category_slug, chosen)
    rng = seeded_rng("catpick", source_name, category_slug, chosen, seed, size)
    picked = sorted(rng.sample(products, min(size, len(products))), key=lambda p: p.id)
    licence = source.licence
    rights = rights_for(source_name, licence)
    store_slug = slug(chosen)
    market = source.manifest.sandbox_market()
    provenance = {
        "catalog": {
            "source": source_name,
            "title": source.manifest.title,
            "category": category_slug,
            "store": chosen,
            "seed": seed,
            "size": size,
            "products": len(picked),
            "files": index.manifest.files,
            "content": index.manifest.content,
            "normaliser": index.manifest.normaliser,
            "prices": source.manifest.prices,
        },
        "licence": licence.model_dump(mode="json", exclude_none=True),
        "rights": rights.model_dump(mode="json"),
    }
    return StoreSnapshot(
        source=source_name,
        category=category_slug,
        store_key=chosen,
        seed=seed,
        size=size,
        store={
            "domain": f"{store_slug}.{source_name}.sandbox.invalid",
            "store_id": f"{source_name}:{store_slug}",
            "platform": "sandbox",
            "category": category_slug,
        },
        currency=market.currency,
        market=MarketDoc.from_market(market),
        variants=[_variant(p, v) for p in picked for v in p.variants],
        products={p.id: product_facts(p) for p in picked},
        provenance=provenance,
        rights=rights,
    )


def envpack_rights(pack: EnvPack, *, settings: Settings | CatalogSettings | None = None) -> Rights:
    """The data rights an episode on `pack` carries in its bundle header.

    The pack's recorded rights, tightened by the source's current licence when the pack was cut from a catalog. A pack
    with no recorded rights, or whose catalog licence cannot be read any more, grants nothing.
    """
    provenance = pack.doc.get("provenance") or {}
    recorded_raw = provenance.get("rights")
    if not isinstance(recorded_raw, dict):
        return Rights()
    recorded = Rights.model_validate(recorded_raw)
    catalog = provenance.get("catalog")
    source = catalog.get("source") if isinstance(catalog, dict) else None
    if not isinstance(source, str):
        return recorded
    try:
        current = licence_rights(source, settings=settings)
    except MCError:
        return strictest(recorded, Rights(store_source=source, platform_terms="sandbox"))
    return strictest(recorded, current)


def _check_category(source_name: str, category_slug: str, settings: Settings | CatalogSettings | None) -> None:
    known = known_categories(resolve(settings).categories_dir)
    if category_slug not in known:
        raise NotFoundError(f"no category {category_slug!r}", hint=f"choose one of {known}")


def _choose_store(source: str, category: str, offered: dict[str, int], *, seed: int, size: int) -> str:
    """A seeded store among those that can fill `size` products, or the largest when none can."""
    if not offered:
        raise NotCoveredError(f"{source} has no store in {category!r}")
    fits = sorted(k for k, n in offered.items() if n >= size)
    if not fits:
        return min(offered, key=lambda k: (-offered[k], k))
    rng = seeded_rng("catstore", source, category, seed, size)
    return fits[rng.randrange(len(fits))]


def _variant(product: ProductRecord, variant: VariantRecord) -> dict[str, Any]:
    return {
        "id": variant.id,
        "product_id": product.id,
        "title": f"{product.title} - {variant.title}" if variant.title != product.title else product.title,
        "price": variant.price,
        "available": variant.available,
        "stock": variant.stock,
        "options": dict(variant.options),
    }

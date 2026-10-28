"""Catalog sources for sandbox stores, with their licences carried into every bundle.

Why. Training data must come from sources we hold rights to (decision D10), and a sandbox store needs real product
detail: sizes with their system, materials, care, ingredients, allergens, dosage, compatibility, dimensions. This
package turns licensed public catalogs into sandbox stores, one category at a time, and makes every store say what
its data may be used for.

Sources (kind `sandbox.catalog`, see `plugins`; each pinned in data/catalogs/<name>.json):
- abo: Amazon Berkeley Objects listings, CC BY 4.0; stores are brands; synthetic prices
- wands: Wayfair's WANDS products, MIT; stores are deterministic partitions; synthetic prices
- off, obf: Open Food Facts and Open Beauty Facts, ODbL 1.0 (share-alike: no training until a lawyer has read it);
  stores are brands; synthetic prices
- lazada: Lazada listings from ShoppingBench, Apache-2.0 release, training and publication allowed since D90
  (amending D11); stores are sellers; real prices in PHP, one market (PH)
- esci, shopify-taxonomy: reference datasets with a licence record and no stores
- open: the open product pools (kind pool, D91): per category, the products of every publishable catalog (abo,
  lazada, wands), and stand-in stores drawn from them (`build_pool_snapshot`, `pool_market`, `ensure_pools`)

Use it
    `open_source("abo").fetch()` downloads the pinned files into <catalogs_dir>/abo/raw (sha256 and size checked,
    budgeted); `--relock` takes a newer upstream file and rewrites the pin. `source.iter_products("footwear")`
    yields normalised `ProductRecord`s. `ensure_index(source)` writes per-category shards and store counts once.
    `build_store_snapshot("abo", "footwear", seed=7, size=40)` cuts a seeded store; `.to_envpack(profile=...,
    tasks=...)` seals it. `envpack_rights(pack)` is the `Rights` a bundle header on that pack carries.
    `licence_rights("wands")` maps a source's licence onto Rights (store_source = the source, platform_terms
    "sandbox").

Replicas (decision D12): `build_replica(capture, market="US")` turns a store capture (`carttrace live capture`) into the
StoreSnapshot of that store's replica, whose envpack (`replica_envpack`) answers the captured requests with the
store's own documents; `replica_fidelity(capture, pack)` replays a sample of them and scores the answers.
`ReplicaSource` (`open_replica()`) is the `replica` block of kind sandbox.catalog: the newest capture of each store.
`add_elements(pack, additions, inventory_id=...)` is the detail_added overlay (decision D15): the replica served with
data elements its pages show and its agent surface lacks, from a store's data inventory (`carttrace live inventory`).
Search memory (D39): `memories_from_store(trace_store)` keeps every search a live bundle recorded, per store
(`SearchMemory`, saved by `save_memory` under <paths.var>/live/search-memory), and `with_search_memory(snapshot,
memory)` makes a replica replay those answers, holding the products only live listed.
Agent surface (D40): `surfaces_from(trace_store, captures_dir=...)` keeps the tools/list each store sent live, per
store and version pair (`AgentSurface`, saved by `save_surface` under <paths.var>/live/agent-surface), and
`with_agent_surface(snapshot, surface)` makes a replica list exactly those tools.
Sold out (D52): `records_from(trace_store, captures_dir=...)` counts, per store, the cart writes live refused as sold
out (`SoldOutRecord`, saved by `save_record` under <paths.var>/live/sold-out), `rate_for(record, market)` is the rate
a replica world in that market answers sold out at (the store's own, else its category's, else every store's), and
`draw_sold_out(pack, seed, witnesses=...)` draws an episode's sold-out variants from it, sparing a witness.

CLI (`register`, for carttrace/sandbox/cli.py): `carttrace sandbox catalogs list [SOURCE] [--category C]`, `fetch SOURCE
[--relock] [--from PATH]`, `build SOURCE [--record] [--category C --seed N --size N --store KEY --out DIR]`, and
`carttrace sandbox replica build CAPTURE`, `carttrace sandbox replica fidelity CAPTURE`, `carttrace sandbox replica memory`,
`carttrace sandbox replica surface` and `carttrace sandbox replica sold-out`.
"""

from carttrace.sandbox.catalogs.agent_surface import (
    AgentSurface,
    build_agent_surface,
    load_surface,
    newest_surface,
    save_surface,
    surfaces_from,
    with_agent_surface,
)
from carttrace.sandbox.catalogs.categories import EXCLUDED, CategoryMap, known_categories, load_category_map
from carttrace.sandbox.catalogs.errors import (
    BudgetError,
    CatalogError,
    IntegrityError,
    NotCoveredError,
    NotFetchedError,
    RestrictedCatalogError,
)
from carttrace.sandbox.catalogs.fidelity import Difference, FidelityReport, replica_fidelity
from carttrace.sandbox.catalogs.index import (
    CatalogIndex,
    CategoryCount,
    IndexManifest,
    build_index,
    current_index,
    ensure_index,
    record_coverage,
)
from carttrace.sandbox.catalogs.licence import Licence, licence_rights, rights_for, strictest
from carttrace.sandbox.catalogs.manifest import SourceFile, SourceManifest, StoreRule, list_manifests, load_manifest
from carttrace.sandbox.catalogs.overlay import Added, add_elements
from carttrace.sandbox.catalogs.pools import (
    OPEN,
    POOL_CAP,
    CategoryPool,
    PoolManifest,
    Pools,
    build_pool_snapshot,
    build_pools,
    capacity,
    check_pool,
    counted_sources,
    current_pools,
    ensure_pools,
    pool_market,
    pool_rows,
)
from carttrace.sandbox.catalogs.records import (
    DETAILS,
    Details,
    ProductRecord,
    Provenance,
    Rating,
    VariantRecord,
    product_facts,
)
from carttrace.sandbox.catalogs.replica import (
    LIVE_LICENCE,
    REPLICA,
    ReplicaSource,
    build_replica,
    open_replica,
    replica_envpack,
)
from carttrace.sandbox.catalogs.search_memory import (
    SearchMemory,
    build_search_memory,
    load_memory,
    memories_from_store,
    newest_memory,
    save_memory,
    with_search_memory,
)
from carttrace.sandbox.catalogs.settings import CatalogSettings
from carttrace.sandbox.catalogs.snapshot import (
    StoreSnapshot,
    StoreSummary,
    build_store_snapshot,
    catalog_stores,
    envpack_rights,
)
from carttrace.sandbox.catalogs.sold_out import (
    Draw,
    SoldOutRecord,
    build_records,
    draw_sold_out,
    load_record,
    newest_record,
    rate_for,
    records_from,
    save_record,
    with_sold_out,
)
from carttrace.sandbox.catalogs.source import KIND, BaseSource, CatalogSource, FetchReport, open_source

__all__ = [
    "DETAILS",
    "EXCLUDED",
    "KIND",
    "LIVE_LICENCE",
    "OPEN",
    "POOL_CAP",
    "REPLICA",
    "Added",
    "AgentSurface",
    "BaseSource",
    "BudgetError",
    "CatalogError",
    "CatalogIndex",
    "CatalogSettings",
    "CatalogSource",
    "CategoryCount",
    "CategoryMap",
    "CategoryPool",
    "Details",
    "Difference",
    "Draw",
    "FetchReport",
    "FidelityReport",
    "IndexManifest",
    "IntegrityError",
    "Licence",
    "NotCoveredError",
    "NotFetchedError",
    "PoolManifest",
    "Pools",
    "ProductRecord",
    "Provenance",
    "Rating",
    "ReplicaSource",
    "RestrictedCatalogError",
    "SearchMemory",
    "SoldOutRecord",
    "SourceFile",
    "SourceManifest",
    "StoreRule",
    "StoreSnapshot",
    "StoreSummary",
    "VariantRecord",
    "add_elements",
    "build_agent_surface",
    "build_index",
    "build_pool_snapshot",
    "build_pools",
    "build_records",
    "build_replica",
    "build_search_memory",
    "build_store_snapshot",
    "capacity",
    "catalog_stores",
    "check_pool",
    "counted_sources",
    "current_index",
    "current_pools",
    "draw_sold_out",
    "ensure_index",
    "ensure_pools",
    "envpack_rights",
    "known_categories",
    "licence_rights",
    "list_manifests",
    "load_category_map",
    "load_manifest",
    "load_memory",
    "load_record",
    "load_surface",
    "memories_from_store",
    "newest_memory",
    "newest_record",
    "newest_surface",
    "open_replica",
    "open_source",
    "pool_market",
    "pool_rows",
    "product_facts",
    "rate_for",
    "record_coverage",
    "records_from",
    "replica_envpack",
    "replica_fidelity",
    "rights_for",
    "save_memory",
    "save_record",
    "save_surface",
    "strictest",
    "surfaces_from",
    "with_agent_surface",
    "with_search_memory",
    "with_sold_out",
]

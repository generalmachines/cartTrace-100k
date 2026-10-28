"""Open product pools (D91): per category, the products of every publishable catalog, and pooled stores cut from them.

Why pools. The publishable lane (D85) cut each stand-in store from one catalog, so a category's variety was whatever
that catalog had: a clothing-adult twin of ABO held 14 products of one type, and gifts-collectibles had no open
source at all. Since D90 Lazada (ShoppingBench) may be published too, and the project asked that each
stand-in store draw its products from its category's pool across all publishable sources. The pool's members are the
`members` of the pool manifest data/catalogs/open.json (kind pool): abo, lazada and wands. A member must allow
publication and not be restricted or share-alike, and the pool's licence may grant nothing a member does not, or the
pool refuses to open (`check_pool`): Open Food Facts and Open Beauty Facts stay out until their review (D85).

What a pool holds. `build_pools` reads each member's index once and keeps, per category, at most `POOL_CAP` products
a member, chosen by the sha256 of their ids (a uniform sample that no code change reorders), under
var/catalogs/open-pools/pool-<key>/, keyed like an index by everything that decides its bytes (the members' index
contents, the cap, the rule). Beside each category's shard it writes a summary: full and pooled products and
variants per source, distinct product types, the price range per currency, a sample of titles, and how many of the
publishable target's 4,000 traces the category can carry before a product repeats more than ten times
(`carry`).

Markets stay honest. Lazada's prices are real pesos in the Philippines; ABO's and WANDS' are our synthetic dollars.
A pooled store holds one group's products, never both: the members with real prices in the world's market (Lazada
for PH), else the synthetic-price members, which serve any market as twins always have (D25). `pool_market` decides
which a stand-in store is, seeded per store, in proportion to the two groups' pooled products in its category, so
the lane draws every pooled product about as often; a store already in PH is always a PH store.

A pooled store (`build_pool_snapshot`) takes `size` products of its group. A stand-in for a store whose capture we
hold (`match_capture`, D104) takes, within each member's share, the products most like that store's own catalog
(carttrace.sandbox.catalogs.matching: its titles' and types' words and its price band), sampled from a shortlist twice its
share, which held the repeat limit on the target's stand-ins; one whose store has no capture is spread over the
group's product types (each type in seeded order gives one product before any gives a second), so twins in one
category hold different products and kinds. Its provenance names the pool, the members' index contents, how many
products each member gave and, for a matched pick, what it matched; its rights are the pool licence's (store_source
`open`), and each product's own provenance names its source.
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import multiprocessing
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from carttrace.config.schema import Settings
from carttrace.core.canon import digest, loads, pretty
from carttrace.core.errors import NotFoundError, SafetyError
from carttrace.core.files import atomic_write_text
from carttrace.core.ids import seeded_rng
from carttrace.sandbox.catalogs.categories import known_categories
from carttrace.sandbox.catalogs.errors import CatalogError, NotCoveredError
from carttrace.sandbox.catalogs.index import CatalogIndex, current_index, modules_digest
from carttrace.sandbox.catalogs.kinds import Kinds, load_kinds
from carttrace.sandbox.catalogs.licence import rights_for
from carttrace.sandbox.catalogs.manifest import SourceManifest, load_manifest
from carttrace.sandbox.catalogs.matching import SPREAD_RULE, Scorer, StoreProfile, load_profile, match_pick
from carttrace.sandbox.catalogs.records import ProductRecord, product_facts
from carttrace.sandbox.catalogs.settings import CatalogSettings, resolve
from carttrace.sandbox.catalogs.snapshot import StoreSnapshot, _variant
from carttrace.sandbox.catalogs.source import open_source
from carttrace.sandbox.catalogs.synthetic import Shared, band_for, load_shared, price_note, synthetic_price
from carttrace.sandbox.catalogs.vetting import VET_RULE, Pricer, Verdict, Vetting, dropped, kind_of
from carttrace.sandbox.core.market import MARKETS, Market, market, market_for
from carttrace.sandbox.envpack.doc import MarketDoc

OPEN: Final = "open"
"""The pool manifest's name, and the catalog name a publishable plan gives to draw from the pools."""
POOL_SCHEMA: Final = "mc.open-pool.v1"
POOL_RULE: Final = "open-pool.v2"
"""Name of the selection rule; change it (v3) whenever the selection or the summary changes. v2 (D104): every pooled
product is vetted (carttrace.sandbox.catalogs.vetting) and each category's summary counts the products dropped per reason."""
POOL_DIR: Final = "open-pools"
CURRENT: Final = "CURRENT"
MANIFEST: Final = "manifest.json"
POOL_CAP: Final = 5000
"""Products a member gives a category's pool, at most."""
TRACES_TARGET: Final = 4000
"""The publishable target's traces per category (the v1 open lane)."""
TASKS_PER_STORE: Final = 25
_FIRST: Final = (
    "amber", "ash", "birch", "blue", "brass", "briar", "cedar", "clover", "copper", "coral", "crest", "dune", "elm",
    "ember", "fern", "flint", "glen", "golden", "harbor", "hazel", "heron", "indigo", "ivy", "juniper", "kestrel",
    "lark", "linden", "maple", "marble", "meadow", "mill", "moss", "north", "oak", "orchard", "pine", "quarry", "raven",
    "reed", "ridge", "river", "rowan", "sage", "slate", "stone", "summit", "thistle", "willow",
)  # fmt: skip
_SECOND: Final = (
    "and-co", "bay", "brook", "court", "crossing", "dale", "field", "ford", "gate", "grove", "hall", "haven", "hill",
    "hollow", "lane", "lodge", "park", "point", "row", "square", "street", "vale", "way", "wharf", "wood", "yard",
)  # fmt: skip
_TRADE: Final = (
    "goods",
    "supply",
    "market",
    "trading",
    "house",
    "store",
    "shop",
    "outfitters",
    "provisions",
    "mercantile",
)
STAND_IN_TLD: Final = "example"
"""Every stand-in's domain is under .example (RFC 2606): reserved, so it names no real store and never resolves."""


def stand_in_domain(category: str, seed: int) -> str:
    """The domain a stand-in of the open pools shows the agent, on every surface: a neutral, brand-like name under
    .example (`copper-lane-goods.example`), stable for its category and seed but naming neither, nor the sandbox.
    The release's roster pseudonyms (`twin-...`) stay release-side; nothing an agent reads encodes the category."""
    h = hashlib.sha256(f"stand-in-domain\n{category}\n{seed}".encode()).digest()
    first, second, trade = _FIRST[h[0] % len(_FIRST)], _SECOND[h[1] % len(_SECOND)], _TRADE[h[2] % len(_TRADE)]
    return f"{first}-{second}-{trade}.{STAND_IN_TLD}"


STORE_SIZE: Final = 40
"""Products a stand-in store holds (carttrace.collect.worlds DEFAULT_SIZE)."""
REPEAT_LIMIT: Final = 10
"""How often one product may appear across a category's stores before the variety is called thin."""
SAMPLE: Final = 8
SYNTHETIC: Final = "synthetic"
_ID = re.compile(rb'"id":"([^"]+)"')


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourcePool(_Frozen):
    """One member's part of a category's pool."""

    products: int = Field(ge=0)
    """In the member's index (the whole category)."""
    variants: int = Field(ge=0)
    pooled: int = Field(ge=0)
    """Kept in the pool (at most POOL_CAP)."""
    pooled_variants: int = Field(ge=0)
    product_types: int = Field(ge=0)
    """Distinct product types (the last step of the source's category path) in the member's index."""
    currency: str
    market: str
    prices: str
    """source (the catalog's own) or synthetic."""
    price_min: int | None = None
    """Lowest variant price in the pool, minor units of `currency`."""
    price_max: int | None = None


class Capacity(_Frozen):
    """How many of the target's traces a category can carry with real product variety."""

    stores: int
    """Stand-in stores the target needs: traces over tasks per store."""
    slots: int
    """Product appearances across those stores."""
    distinct: int
    """Pooled products available."""
    repeats: float
    """Appearances per product at the target."""
    carry: int
    """Traces the pool carries before a product repeats more than REPEAT_LIMIT times (capped at the target)."""
    thin: bool
    """repeats above REPEAT_LIMIT."""


class CategoryPool(_Frozen):
    schema_: str = Field(default=POOL_SCHEMA, alias="schema")
    category: str
    sources: dict[str, SourcePool] = Field(default_factory=dict)
    groups: dict[str, list[str]] = Field(default_factory=dict)
    """Currency to the members whose products are priced in it."""
    product_types: int = 0
    """Distinct product types across members (types read the same way in two members count once)."""
    top_types: list[tuple[str, int]] = Field(default_factory=list)
    """The ten largest product types in the pool, with their pooled products."""
    sample: list[dict[str, Any]] = Field(default_factory=list)
    dropped: dict[str, int] = Field(default_factory=dict)
    """Products the vetting dropped (D104), per reason."""
    capacity: Capacity
    sha256: str
    """sha256 of the shard's lines as written."""


class PoolManifest(_Frozen):
    schema_: str = Field(default=POOL_SCHEMA, alias="schema")
    key: str
    rule: str = POOL_RULE
    cap: int
    members: dict[str, str]
    """Member to the content hash of the index it was read from."""
    categories: dict[str, CategoryPool]
    counted: dict[str, dict[str, int]] = Field(default_factory=dict)
    """Catalogs counted but not pooled (licence under review): source to products per category."""


@dataclass(frozen=True, slots=True)
class Pools:
    """A built, current set of pools."""

    path: Path
    manifest: PoolManifest

    def shard(self, category: str) -> Path:
        return self.path / f"{category}.jsonl.gz"

    def category(self, category: str) -> CategoryPool:
        found = self.manifest.categories.get(category)
        if found is None:
            raise NotCoveredError(f"the open pools hold no products in {category!r}", hint=f"held: {self.covered()}")
        return found

    def covered(self) -> list[str]:
        return sorted(c for c, p in self.manifest.categories.items() if any(s.pooled for s in p.sources.values()))

    def records(self, category: str) -> list[ProductRecord]:
        """The pooled products of a category, sorted by source and id."""
        return list(_records(str(self.shard(category)), self.manifest.categories[category].sha256))


# -- members and their licence -------------------------------------------------------------------------------------


def pool_manifest(settings: Settings | CatalogSettings | None = None) -> SourceManifest:
    found = load_manifest(OPEN, settings=settings)
    if found.kind != "pool":
        raise CatalogError(f"{OPEN}.json is a {found.kind}, not a pool", hint="set kind: pool and list its members")
    return found


def check_pool(settings: Settings | CatalogSettings | None = None) -> list[SourceManifest]:
    """The pool's member manifests, after refusing a pool that could leak rights: a member that is not a catalog,
    is restricted or share-alike, or does not allow publication, and a pool licence that allows what a member does
    not (commercial use, training, publication)."""
    pool = pool_manifest(settings)
    members = [load_manifest(name, settings=settings) for name in pool.members]
    for m in members:
        lic = m.licence
        problem = (
            "is not a catalog"
            if m.kind != "catalog"
            else "is restricted"
            if lic.restricted
            else "is share-alike"
            if lic.share_alike
            else "does not allow publication"
            if not lic.publication_allowed
            else None
        )
        if problem is not None:
            raise SafetyError(
                f"pool member {m.name} {problem}; a pooled store's bundles would carry rights it cannot grant (D91)",
                hint=f"remove {m.name} from data/catalogs/{OPEN}.json members",
            )
    lic = pool.licence
    for flag in ("commercial_use", "training_allowed", "publication_allowed"):
        if getattr(lic, flag) and not all(getattr(m.licence, flag) for m in members):
            raise SafetyError(
                f"the pool licence allows {flag} that a member's does not (D91)",
                hint=f"set {flag} false in data/catalogs/{OPEN}.json",
            )
    return members


def counted_sources(settings: Settings | CatalogSettings | None = None) -> list[str]:
    """Catalogs we hold but do not pool: those whose licence does not allow publication yet (off, obf)."""
    from carttrace.sandbox.catalogs.manifest import list_manifests

    pooled = set(pool_manifest(settings).members)
    return [
        m.name
        for m in list_manifests(settings=settings)
        if m.kind == "catalog" and m.name not in pooled and not m.licence.restricted
    ]


# -- building ------------------------------------------------------------------------------------------------------


def pools_root(settings: Settings | CatalogSettings | None = None) -> Path:
    return resolve(settings).catalogs_dir / POOL_DIR


def _member_indexes(members: Sequence[SourceManifest], settings: CatalogSettings) -> dict[str, CatalogIndex]:
    out: dict[str, CatalogIndex] = {}
    for m in members:
        index = current_index(open_source(m.name, settings=settings))
        if index is None:
            raise NotFoundError(
                f"{m.name} has no current index, so its pool cannot be read",
                hint=f"run `carttrace sandbox catalogs build {m.name} --record`",
            )
        out[m.name] = index
    return out


def pools_key(indexes: Mapping[str, CatalogIndex], cap: int, cfg: CatalogSettings | None = None) -> str:
    vetted = _vetting_key(cfg) if cfg is not None else {}
    return digest(
        {"rule": POOL_RULE, "cap": cap, "members": {k: v.manifest.content for k, v in indexes.items()}, **vetted}, 16
    )


def _vetting_key(cfg: CatalogSettings) -> dict[str, str]:
    """What the vetting reads besides the indexes: its rule, the lexicon and the shared price rules."""
    return {
        "vet": VET_RULE,
        "kinds": _kinds(cfg).digest,
        "shared": load_shared(cfg.data_dir).digest(),
        "code": modules_digest("carttrace.sandbox.catalogs.kinds", "carttrace.sandbox.catalogs.vetting"),
    }


def _kinds(cfg: CatalogSettings) -> Kinds:
    return load_kinds(cfg.data_dir / "kinds.yaml")


def current_pools(settings: Settings | CatalogSettings | None = None) -> Pools | None:
    """The pools if built from exactly the members' current indexes, else None."""
    cfg = resolve(settings)
    root = pools_root(cfg)
    try:
        name = (root / CURRENT).read_text(encoding="utf-8").strip()
        manifest = PoolManifest.model_validate(loads((root / name / MANIFEST).read_bytes()))
    except (FileNotFoundError, ValueError):
        return None
    members = check_pool(cfg)
    try:
        indexes = _member_indexes(members, cfg)
    except NotFoundError:
        return None
    if manifest.key != pools_key(indexes, manifest.cap, cfg):
        return None
    return Pools(root / name, manifest)


def ensure_pools(settings: Settings | CatalogSettings | None = None, *, workers: int | None = None) -> Pools:
    return current_pools(settings) or build_pools(settings, workers=workers)


def build_pools(
    settings: Settings | CatalogSettings | None = None, *, cap: int = POOL_CAP, workers: int | None = None
) -> Pools:
    """Read every member's index once and write every category's pool. Deterministic: the same indexes and cap give
    the same bytes, whatever `workers` is."""
    cfg = resolve(settings)
    members = check_pool(cfg)
    indexes = _member_indexes(members, cfg)
    key = pools_key(indexes, cap, cfg)
    root = pools_root(cfg)
    final = root / f"pool-{key}"
    tmp = root / f".tmp-{key}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    by_name = {m.name: m for m in members}
    jobs = [
        (name, category, str(index.shard(category)), cap)
        for name, index in sorted(indexes.items())
        for category in sorted(index.categories())
    ]
    count = workers or min(32, len(jobs) or 1)
    if count <= 1:
        scanned = [_scan_shard(job) for job in jobs]
    else:
        # spawn, not fork: a manifest build that first needs the pools is a threaded process
        with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn")) as pool:
            scanned = list(pool.map(_scan_shard, jobs))
    per_category: dict[str, dict[str, _Scan]] = {}
    for scan in scanned:
        per_category.setdefault(scan.category, {})[scan.source] = scan
    parsed = {
        category: {source: [ProductRecord.from_line(line) for line in scan.kept] for source, scan in found.items()}
        for category, found in per_category.items()
    }
    vetting = Vetting.of(
        (r for found in parsed.values() for records in found.values() for r in records),
        _kinds(cfg),
        load_shared(cfg.data_dir).synthetic.price,
    )
    categories: dict[str, CategoryPool] = {}
    for category in known_categories(cfg.categories_dir):
        found = per_category.get(category)
        if found:
            verdicts = {s: [vetting.check(r) for r in parsed[category][s]] for s in sorted(found)}
            categories[category] = _write_category(tmp, category, found, by_name, verdicts)
    counted = _counted(cfg)
    manifest = PoolManifest(
        key=key,
        cap=cap,
        members={k: v.manifest.content for k, v in sorted(indexes.items())},
        categories=categories,
        counted=counted,
    )
    atomic_write_text(tmp / MANIFEST, pretty(manifest.model_dump(mode="json", by_alias=True)))
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    atomic_write_text(root / CURRENT, final.name + "\n")
    for old in root.glob("pool-*"):
        if old != final and old.is_dir():
            shutil.rmtree(old, ignore_errors=True)
    return Pools(final, manifest)


def _counted(cfg: CatalogSettings) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for name in counted_sources(cfg):
        index = current_index(open_source(name, settings=cfg))
        if index is not None:
            out[name] = {c: n.products for c, n in sorted(index.manifest.categories.items())}
    return out


@dataclass(slots=True)
class _Scan:
    source: str
    category: str
    products: int
    variants: int
    types: Counter[str]
    kept: list[bytes]
    """The pooled lines, sorted by product id."""


def _rank(product_id: bytes) -> bytes:
    return hashlib.sha256(b"open-pool\n" + product_id).digest()


def _scan_shard(job: tuple[str, str, str, int]) -> _Scan:
    """One member's category: every product counted, the `cap` products of lowest id hash kept."""
    source, category, path, cap = job
    heap: list[tuple[bytes, bytes, bytes]] = []
    products = variants = 0
    types: Counter[str] = Counter()
    with gzip.open(path, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            doc = loads(line)
            products += 1
            variants += len(doc.get("variants") or ())
            path_ = doc.get("category_path") or []
            types[_type_of(path_)] += 1
            found = _ID.search(line)
            pid = found.group(1) if found else str(doc.get("id")).encode()
            item = (bytes(255 - b for b in _rank(pid)), pid, line)  # inverted: heap[0] is the largest rank kept
            if len(heap) < cap:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    kept = [line for _, _, line in sorted(heap, key=lambda t: t[1])]
    return _Scan(source, category, products, variants, types, kept)


def _type_of(path: Sequence[Any]) -> str:
    """A product's type for variety counts: the last step of its category path, folded."""
    return str(path[-1]).strip().lower() if path else ""


def _write_category(
    directory: Path,
    category: str,
    scans: Mapping[str, _Scan],
    members: Mapping[str, SourceManifest],
    verdicts: Mapping[str, Sequence[Verdict]],
) -> CategoryPool:
    sha = hashlib.sha256()
    records: list[ProductRecord] = []
    with (
        (directory / f"{category}.jsonl.gz").open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz,
    ):
        for source in sorted(scans):
            for verdict in verdicts[source]:
                if verdict.record is None:
                    continue
                body = (verdict.record.to_line() + "\n").encode("utf-8")
                sha.update(body)
                gz.write(body)
                records.append(verdict.record)
    sources: dict[str, SourcePool] = {}
    groups: dict[str, list[str]] = {}
    for source, scan in sorted(scans.items()):
        mine = [r for r in records if r.source == source]
        prices = [v.price for r in mine for v in r.variants]
        m = members[source]
        sources[source] = SourcePool(
            products=scan.products,
            variants=scan.variants,
            pooled=len(mine),
            pooled_variants=sum(len(r.variants) for r in mine),
            product_types=len(scan.types),
            currency=m.currency or "",
            market=m.market or "",
            prices=m.prices or "",
            price_min=min(prices) if prices else None,
            price_max=max(prices) if prices else None,
        )
        groups.setdefault(m.currency or "", []).append(source)
    types = Counter(_type_of(r.category_path) for r in records)
    all_types = set().union(*(set(s.types) for s in scans.values()))
    rng = seeded_rng("openpool-sample", category)
    sample = [_sample_row(r) for r in sorted(rng.sample(records, min(SAMPLE, len(records))), key=lambda r: r.id)]
    pool = CategoryPool(
        category=category,
        sources=sources,
        groups={k: sorted(v) for k, v in sorted(groups.items())},
        product_types=len(all_types),
        top_types=[(t or "(none)", n) for t, n in sorted(types.items(), key=lambda kv: (-kv[1], kv[0]))[:10]],
        sample=sample,
        dropped=dropped(v for found in verdicts.values() for v in found),
        capacity=capacity(len(records)),
        sha256=sha.hexdigest(),
    )
    atomic_write_text(directory / f"{category}.json", pretty(pool.model_dump(mode="json", by_alias=True)))
    return pool


def _sample_row(record: ProductRecord) -> dict[str, Any]:
    v = record.variants[0]
    return {
        "source": record.source,
        "title": record.title[:120],
        "product_type": record.category_path[-1] if record.category_path else None,
        "price": v.price,
        "currency": v.currency,
    }


def capacity(distinct: int) -> Capacity:
    """How many of the target's traces `distinct` pooled products carry before one repeats more than REPEAT_LIMIT
    times across the category's stores (TASKS_PER_STORE tasks a store, STORE_SIZE products a store)."""
    stores = -(-TRACES_TARGET // TASKS_PER_STORE)
    slots = stores * STORE_SIZE
    repeats = round(slots / distinct, 1) if distinct else float(slots)
    carry = min(TRACES_TARGET, distinct * REPEAT_LIMIT * TASKS_PER_STORE // STORE_SIZE)
    return Capacity(
        stores=stores, slots=slots, distinct=distinct, repeats=repeats, carry=carry, thin=repeats > REPEAT_LIMIT
    )


@lru_cache(maxsize=4)
def _records(path: str, sha: str) -> tuple[ProductRecord, ...]:
    """A category shard's records, kept for the few categories a manifest build is working on (keyed by the shard's
    sha256, so a rebuilt pool is never served from memory)."""
    with gzip.open(path, "rb") as fh:
        return tuple(ProductRecord.from_line(line) for line in fh if line.strip())


# -- stand-in stores -----------------------------------------------------------------------------------------------


def _group(pool: CategoryPool, members: Mapping[str, SourceManifest], market_code: str) -> list[str]:
    """The members a store in `market_code` draws from: those with real prices in that market, else the synthetic
    ones. Empty when neither has products in the category."""
    code = _code(market_code)
    own = [s for s, p in pool.sources.items() if p.prices != SYNTHETIC and _code(p.market) == code and p.pooled]
    if own:
        return sorted(own)
    return sorted(s for s, p in pool.sources.items() if p.prices == SYNTHETIC and p.pooled)


def _code(code: str) -> str:
    try:
        return market(code).code
    except NotFoundError:
        return code.upper()


MIXED: Final = "mixed"
HOME: Final = "home"
MIXED_MARKET: Final = "US"
"""The one market a mixed stand-in store sells in: synthetic US dollars for every member."""
MAX_MEMBER_SHARE: Final = 0.6
"""The most of a mixed store one member may fill while another member has products left, so Lazada's pools do not
drown ABO's and WANDS' in a category."""


def pool_market(
    category: str, store_id: str, *, seed: int, home: str, settings: Settings | CatalogSettings | None = None
) -> str:
    """The market a stand-in store of `category` is built in. Mixed pools (the default): always MIXED_MARKET, every
    member's products priced in its synthetic dollars. Home pools (sandbox.open_pool_markets home): `home`, or the
    market of a real-price member (PH for Lazada), drawn per store in proportion to the pooled products each group
    holds; a store whose home has a real-price member stays home."""
    cfg = resolve(settings)
    if cfg.open_pool_markets != HOME:
        return MIXED_MARKET
    pools = ensure_pools(cfg)
    pool = pools.category(category)
    real = sorted({p.market for p in pool.sources.values() if p.prices != SYNTHETIC and p.pooled})
    if _code(home) in {_code(m) for m in real}:
        return home
    synthetic = sum(p.pooled for p in pool.sources.values() if p.prices == SYNTHETIC)
    weights = [(home, synthetic)] + [
        (m, sum(p.pooled for p in pool.sources.values() if p.prices != SYNTHETIC and p.market == m)) for m in real
    ]
    total = sum(w for _, w in weights)
    if total == 0:
        return home
    draw = seeded_rng("openpool-market", category, store_id, seed).randrange(total)
    for code, weight in weights:
        if draw < weight:
            return code
        draw -= weight
    return home


def member_quotas(available: Mapping[str, int], size: int) -> dict[str, int]:
    """How many of a mixed store's `size` products each member gives. In proportion to its pooled products, with two
    caps while others still have room: no member fills more than MAX_MEMBER_SHARE of the store, and none gives more
    than its pool can repeat at the target without a product appearing more than REPEAT_LIMIT times. When the caps
    leave the store short, the share cap is lifted first, then the repeat cap. A member whose pool can take one
    product a store gives at least one. Integer arithmetic on counts only."""
    members = sorted(m for m, n in available.items() if n > 0)
    want = min(size, sum(available[m] for m in members))
    stores = -(-TRACES_TARGET // TASKS_PER_STORE)
    share = max(1, size * int(MAX_MEMBER_SHARE * 100) // 100)
    repeat = {m: available[m] * REPEAT_LIMIT // stores for m in members}
    tiers = [
        {m: min(available[m], share, repeat[m]) for m in members},
        {m: min(available[m], repeat[m]) for m in members},
        {m: available[m] for m in members},
    ]
    if len(members) == 1:
        tiers = [tiers[-1]]
    quota = dict.fromkeys(members, 0)
    if want >= len(members):  # every member whose pool can take it gives at least one product
        quota.update({m: 1 for m in members if tiers[0][m] >= 1})
    for cap in tiers:
        while sum(quota.values()) < want:
            room = [m for m in members if quota[m] < cap[m]]
            if not room:
                break
            left = want - sum(quota.values())
            weight = sum(available[m] for m in room)
            gave = 0
            for m in room:
                give = min(cap[m] - quota[m], left * available[m] // weight)
                quota[m] += give
                gave += give
            if gave == 0:
                top = max(room, key=lambda m: (available[m] - quota[m], m))
                quota[top] += 1
    return quota


def mixed_plan(pool: CategoryPool, size: int = STORE_SIZE) -> dict[str, Any]:
    """A mixed store's split of a category and what it means at the target: each member's products a store, and how
    often each of its pooled products appears across the target's stores."""
    available = {s: p.pooled for s, p in pool.sources.items()}
    quota = member_quotas(available, size)
    stores = -(-TRACES_TARGET // TASKS_PER_STORE)
    repeats = {s: round(stores * q / available[s], 1) for s, q in quota.items() if available[s]}
    worst = max(repeats.values(), default=0.0)
    return {"per_store": quota, "products": sum(quota.values()), "repeats": repeats, "worst": worst,
            "thin": worst > REPEAT_LIMIT}  # fmt: skip


def repriced(record: ProductRecord, shared: Shared, pricer: Pricer | None = None) -> ProductRecord:
    """A real-price member's product in a mixed store, priced in the synthetic currency as ABO's and WANDS' are: by its
    own price's rank among its category's products of its kind, placed in its kind's US band (`Pricer`, D104), or,
    without a pricer, a synthetic price from its category's band (D91). Its own price is kept as provenance
    (provenance.derived.source_price), and provenance names the rule: derived.price for a ranked price,
    synthetic.price for a drawn one."""
    rule = shared.synthetic.price
    if record.currency == rule.currency:
        return record
    own = sorted({(v.price, v.currency) for v in record.variants})
    kept = "; ".join(f"{p} {c} minor units" for p, c in own)
    synthetic = dict(record.provenance.synthetic)
    derived = dict(record.provenance.derived)
    band = band_for(rule, record.category)
    if pricer is not None:
        price, note = pricer.price(record, band)
        derived["price"] = note
        derived["source_price"] = f"{kept}: {record.source}'s own price, ranked not converted (D104)"
    else:
        price = synthetic_price(rule, band, source=record.source, product_id=record.id)
        synthetic["price"] = price_note(rule, band)
        derived["source_price"] = f"{kept}: {record.source}'s own price, not converted (D91 mixed pools)"
    variants = [v.model_copy(update={"price": price, "currency": rule.currency}) for v in record.variants]
    provenance = record.provenance.model_copy(update={"synthetic": synthetic, "derived": derived})
    return record.model_copy(update={"variants": variants, "provenance": provenance})


def pick(records: Sequence[ProductRecord], *, category: str, seed: int, size: int) -> list[ProductRecord]:
    """`size` products spread over the records' product types: types in seeded order, one product of each before a
    second of any, each type's products in seeded order. Sorted by id."""
    by_type: dict[str, list[ProductRecord]] = {}
    for r in sorted(records, key=lambda r: (r.source, r.id)):
        by_type.setdefault(_type_of(r.category_path), []).append(r)
    rng = seeded_rng("openpool-pick", category, seed, size)
    order = sorted(by_type)
    rng.shuffle(order)
    queues = {t: rng.sample(by_type[t], len(by_type[t])) for t in order}
    out: list[ProductRecord] = []
    while len(out) < size and any(queues.values()):
        for t in order:
            if queues[t] and len(out) < size:
                out.append(queues[t].pop())
    return sorted(out, key=lambda r: r.id)


def matched(
    records: Sequence[ProductRecord], scorer: Scorer | None, *, category: str, seed: int, size: int
) -> list[ProductRecord]:
    """`size` of `records`: the ones most like the scorer's store (`match_pick`, D104), topped up by the spread pick
    when fewer distinct titles are left than asked for; without a scorer, the spread pick (`pick`). Sorted by id."""
    if scorer is None:
        return pick(records, category=category, seed=seed, size=size)
    out = match_pick(records, scorer, category=category, seed=seed, size=size)
    if len(out) < size:
        held = {r.id for r in out}
        rest = [r for r in records if r.id not in held]
        out += pick(rest, category=category, seed=seed, size=size - len(out))
    return sorted(out, key=lambda r: r.id)


def build_pool_snapshot(
    category: str,
    *,
    seed: int,
    size: int,
    market_code: str = "US",
    match_capture: str | None = None,
    settings: Settings | CatalogSettings | None = None,
) -> StoreSnapshot:
    """A stand-in store of `category`, ready for build_envpack. Mixed pools (the default): `size` products across
    every member (`member_quotas`), all priced in synthetic dollars (`repriced`) and sold in MIXED_MARKET. Home pools:
    `size` products of one member group for a world in `market_code` (see `_group`). Within each member's share the
    products are those most like the store whose capture `match_capture` names (D104, `matched`), else spread over the
    member's product types; the provenance's `pick` says which. A product with no variant for sale (`sellable`) is
    never drawn. Raises NotCoveredError when there is nothing to draw,
    and NotFoundError when the named capture is gone."""
    if size < 1:
        raise NotFoundError(f"a store needs at least one product, got size={size}", hint="pass size >= 1")
    cfg = resolve(settings)
    members = {m.name: m for m in check_pool(cfg)}
    pools = ensure_pools(cfg)
    pool = pools.category(category)
    mixed = cfg.open_pool_markets != HOME
    profile = _profile(cfg, match_capture)
    if mixed:
        shared = load_shared(cfg.data_dir)
        records = pools.records(category)
        kinds = _kinds(cfg)
        # prices rank among every pooled product of the category, so leaving the unsellable ones out moves no price
        pricer = Pricer.of(records, kinds, shared.synthetic.price)
        records = [r for r in records if sellable(r)]
        scorer = Scorer.of(profile, records) if profile is not None else None
        quota = member_quotas({s: p.pooled for s, p in pool.sources.items()}, size)
        picked = sorted(
            (
                repriced(r, shared, pricer)
                for member, n in quota.items()
                for r in matched(
                    _shelved([x for x in records if x.source == member], kinds, n),
                    scorer,
                    category=f"{category}:{member}",
                    seed=seed,
                    size=n,
                )
            ),
            key=lambda r: r.id,
        )
        group = sorted(quota)
        currency = shared.synthetic.price.currency
        store_market = market_for(MIXED_MARKET, currency)
    else:
        group = _group(pool, members, market_code)
        records = [r for r in pools.records(category) if r.source in group and sellable(r)]
        scorer = Scorer.of(profile, records) if profile is not None else None
        picked = matched(records, scorer, category=category, seed=seed, size=size)
        currency = picked[0].currency if picked else ""
        store_market = _store_market(members[group[0]], market_code) if group else MARKETS[MIXED_MARKET]
    if not picked:
        raise NotCoveredError(f"the open pool of {category!r} has no products for market {market_code}")
    licence = pool_manifest(cfg).licence
    rights = rights_for(OPEN, licence)
    counts = Counter(r.source for r in picked)
    key = f"pool-{category}-{seed}"
    provenance = {
        "catalog": {
            "source": OPEN,
            "title": pool_manifest(cfg).title,
            "category": category,
            "store": key,
            "seed": seed,
            "size": size,
            "products": len(picked),
            "pool": pools.manifest.key,
            "rule": POOL_RULE,
            "markets": MIXED if mixed else HOME,
            "sources": dict(sorted(counts.items())),
            "contents": {s: pools.manifest.members[s] for s in group},
            "prices": {s: members[s].prices for s in group},
            # D104: what the products were picked by; left out of a spread pick so its packs keep their ids
            **({"pick": pick_basis(profile, match_capture)} if match_capture is not None else {}),
        },
        "licence": licence.model_dump(mode="json", exclude_none=True),
        "rights": rights.model_dump(mode="json"),
    }
    return StoreSnapshot(
        source=OPEN,
        category=category,
        store_key=key,
        seed=seed,
        size=size,
        store={
            "domain": stand_in_domain(category, seed),
            "store_id": f"{OPEN}:{key}",
            "platform": "sandbox",
            "category": category,
        },
        currency=currency,
        market=MarketDoc.from_market(store_market),
        variants=[_variant(p, v) for p in picked for v in p.variants],
        products={p.id: product_facts(p) for p in picked},
        provenance=provenance,
        rights=rights,
    )


def sellable(record: ProductRecord) -> bool:
    """A pooled product a stand-in may shelve: one with at least one variant for sale. Why (2026-10-17): 4,116 of the
    pools' 186,334 products list every variant unavailable at the source (a sold-out Lazada listing), and a stand-in
    that shelved one served a product nothing of which could be bought, which no task's witness may be and which a
    designed stockout (carttrace.sandbox.catalogs.sold_out, the task's sealed stockout) is the only honest way to show. A
    stand-in's own sold-out answers come from those draws, never from the source's stale listing."""
    return any(v.available for v in record.variants)


def _shelved(records: Sequence[ProductRecord], kinds: Kinds, wanted: int) -> list[ProductRecord]:
    """A member's products a stand-in shelves first: those whose title names a reviewed kind (D104), since only they
    can carry a task; all of the member's products when fewer than `wanted` do."""
    named = [r for r in records if kind_of(kinds, r) is not None]
    return named if len(named) >= wanted else list(records)


def _profile(cfg: CatalogSettings, capture_id: str | None) -> StoreProfile | None:
    if capture_id is None:
        return None
    if cfg.store_captures_dir is None:
        raise NotFoundError(
            f"a stand-in matched to capture {capture_id} needs the store captures",
            hint="build it through CatalogSettings.from_settings, which names <paths.var>/store/captures",
        )
    return load_profile(cfg.store_captures_dir, capture_id)


def pick_basis(profile: StoreProfile | None, capture_id: str | None) -> dict[str, Any]:
    """What a stand-in's products were picked by (D104): the matched store's profile, or the spread pick and why."""
    if profile is not None:
        return profile.basis()
    if capture_id is not None:
        return {"rule": SPREAD_RULE, "capture": capture_id, "why": "the capture holds no titled product to match"}
    return {"rule": SPREAD_RULE, "why": "no capture of the store to match"}


def _store_market(member: SourceManifest, market_code: str) -> Market:
    """A real-price member's own market row; else the member's market priced in its currency."""
    if member.prices != SYNTHETIC:
        return member.sandbox_market()
    return MARKETS.get(_code(market_code)) or member.sandbox_market()


def pool_rows(pools: Pools) -> Iterator[dict[str, Any]]:
    """One summary row per category, for the research note and `carttrace sandbox catalogs pools`."""
    for category, pool in sorted(pools.manifest.categories.items()):
        cap = pool.capacity
        yield {
            "category": category,
            "sources": {s: (p.products, p.pooled) for s, p in pool.sources.items()},
            "product_types": pool.product_types,
            "pooled": cap.distinct,
            "repeats": cap.repeats,
            "carry": cap.carry,
            "thin": cap.thin,
            "mixed": mixed_plan(pool),
        }


__all__ = [
    "OPEN",
    "POOL_CAP",
    "CategoryPool",
    "PoolManifest",
    "Pools",
    "build_pool_snapshot",
    "build_pools",
    "capacity",
    "check_pool",
    "counted_sources",
    "current_pools",
    "ensure_pools",
    "matched",
    "member_quotas",
    "mixed_plan",
    "pick",
    "pick_basis",
    "pool_market",
    "pool_rows",
    "repriced",
]

"""The catalog source interface, the base every source shares, and choosing a source by name.

Why a Protocol plus a base. A source is a Lego block (`carttrace.core.registry`, kind `sandbox.catalog`): ABO, WANDS, Open
Food Facts, Open Beauty Facts and Lazada each read a different upstream format, and callers ask for a source by name
and use only the interface (`name`, `licence`, `fetch`, `iter_products`). What they share lives in `BaseSource`: the
manifest, the category map, the synthetic price and stock rules, the restricted-source gate, and fetching the
manifest's files with pins and budgets. A subclass only turns raw files into `ProductRecord`s.

The restricted gate. A source whose licence is restricted (Lazada) refuses to fetch, read or index unless
settings.sandbox.allow_restricted_catalogs is true, and that allowance never changes the rights its bundles carry.
"""

from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

import httpx

from carttrace.config.schema import Settings
from carttrace.core.clock import Clock, SystemClock
from carttrace.core.errors import ConfigError, NotFoundError
from carttrace.core.registry import resolve as resolve_component
from carttrace.sandbox.catalogs.categories import EXCLUDED, CategoryMap, load_category_map
from carttrace.sandbox.catalogs.download import Fetched, Progress, RawDigests, download, http_client, import_local
from carttrace.sandbox.catalogs.errors import (
    BudgetError,
    CatalogError,
    IntegrityError,
    NotFetchedError,
    RestrictedCatalogError,
)
from carttrace.sandbox.catalogs.licence import Licence
from carttrace.sandbox.catalogs.manifest import SourceFile, SourceManifest, manifest_path, read_manifest, write_manifest
from carttrace.sandbox.catalogs.records import ProductRecord, Provenance, VariantRecord
from carttrace.sandbox.catalogs.settings import CatalogSettings, resolve
from carttrace.sandbox.catalogs.synthetic import (
    Shared,
    band_for,
    load_shared,
    price_note,
    stock_note,
    synthetic_price,
    synthetic_stock,
)

KIND = "sandbox.catalog"
RAW = "raw"


@runtime_checkable
class CatalogSource(Protocol):
    """A catalog a sandbox store can be cut from."""

    @property
    def name(self) -> str: ...

    @property
    def licence(self) -> Licence: ...

    def fetch(
        self,
        dest: Path | None = None,
        *,
        relock: bool = False,
        local: Path | None = None,
        transport: httpx.BaseTransport | None = None,
        clock: Clock | None = None,
        progress: Progress | None = None,
    ) -> FetchReport:
        """Download the official files into `dest` (default <catalogs_dir>/<name>/raw), verified against their
        pins; `relock` takes newer upstream bytes and rewrites the pins; `local` takes the file from a local path."""
        ...

    def iter_products(self, category_slug: str | None = None) -> Iterator[ProductRecord]:
        """Normalised products in one of our categories (all mapped categories when None)."""
        ...


UNPLACED_EXAMPLES = 3


@dataclass(slots=True)
class Unplaced:
    """A source key the category map does not place: how many rows carry it and a few of their titles."""

    key: str
    rows: int = 0
    examples: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FetchReport:
    source: str
    files: list[Fetched] = field(default_factory=list)
    relocked: bool = False
    """The manifest's pins were rewritten."""

    def lines(self) -> list[str]:
        out = [f"{f.name}: {f.status} {f.size} bytes sha256 {f.sha256[:16]}" for f in self.files]
        if self.relocked:
            out.append(f"relocked data/catalogs/{self.source}.json")
        return out


class BaseSource:
    """What every catalog source shares. Subclasses set NORMALISER and implement `records`."""

    NORMALISER: ClassVar[str] = "base.v0"
    """Name and version of the subclass's parsing rules; part of every index key, so a parsing change rebuilds."""
    REPORTS_UNPLACED: ClassVar[bool] = False
    """Whether `records` notes the keys its category map leaves unplaced (`note_unplaced`)."""
    UNPLACED_FIELD: ClassVar[str] = "category"
    """What an unplaced key is, in words, for the model that places it (product type, category path, ...)."""

    def __init__(
        self,
        *,
        manifest: SourceManifest,
        categories: CategoryMap,
        shared: Shared,
        settings: CatalogSettings,
    ) -> None:
        self.manifest = manifest
        self.categories = categories
        self.shared = shared
        self.settings = settings
        self.skipped: Counter[str] = Counter()
        """Why upstream rows were left out during the last `records` pass, by reason."""
        self.unplaced: dict[str, Unplaced] = {}
        """Source keys the category map matched nothing for during the last `records` pass (excluded keys are not
        here), for parsers that report them (`REPORTS_UNPLACED`): what `carttrace collect open-products classify` asks
        about (D91)."""

    # -- identity ------------------------------------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def licence(self) -> Licence:
        return self.manifest.licence

    @property
    def root(self) -> Path:
        return self.settings.catalogs_dir / self.name

    @property
    def raw_dir(self) -> Path:
        return self.root / RAW

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, {self.licence.spdx})"

    # -- the restricted gate -------------------------------------------------------------------------------------------

    def require_allowed(self) -> None:
        if self.licence.restricted and not self.settings.allow_restricted:
            raise RestrictedCatalogError(
                f"{self.name} is a restricted catalog ({self.licence.spdx}): {self.licence.basis}",
                hint="for internal research only, set sandbox.allow_restricted_catalogs to true in "
                "config.local.json; its bundles stay barred from training and publication",
            )

    # -- fetching ------------------------------------------------------------------------------------------------------

    def fetch(
        self,
        dest: Path | None = None,
        *,
        relock: bool = False,
        local: Path | None = None,
        transport: httpx.BaseTransport | None = None,
        clock: Clock | None = None,
        progress: Progress | None = None,
    ) -> FetchReport:
        self.require_allowed()
        dest = dest or self.raw_dir
        if local is not None and len(self.manifest.files) != 1:
            raise CatalogError(f"{self.name} has {len(self.manifest.files)} files; --from takes one")
        self._check_total_budget()
        unpinned = [f.name for f in self.manifest.files if f.sha256 is None]
        if unpinned and not relock:
            raise IntegrityError(
                f"{self.name}: {unpinned} have no sha256 pin yet, so nothing can be verified",
                hint=f"run `carttrace sandbox catalogs fetch {self.name} --relock` once to pin what upstream serves",
            )
        fetched: list[Fetched] = []
        with http_client(user_agent=self.settings.user_agent, transport=transport) as client:
            for spec in self.manifest.files:
                want_sha = None if relock else spec.sha256
                want_size = None if relock else spec.size
                target = dest / spec.name
                if local is not None:
                    got = import_local(
                        local, target, want_sha256=want_sha, want_size=want_size, budget=self.manifest.budget_bytes
                    )
                else:
                    url = (spec.latest or spec.url) if relock else spec.url
                    got = download(
                        client,
                        url,
                        target,
                        want_sha256=want_sha,
                        want_size=want_size,
                        budget=self.manifest.budget_bytes,
                        progress=progress,
                    )
                RawDigests(dest).record(spec.name, got.sha256)
                fetched.append(got)
        changed = relock and self._relock(fetched, clock or SystemClock())
        return FetchReport(self.name, fetched, relocked=changed)

    def _relock(self, fetched: Sequence[Fetched], clock: Clock) -> bool:
        """Rewrite the manifest's pins to the fetched bytes. Returns whether anything changed."""
        day = clock.now().date().isoformat()
        by_name = {f.name: f for f in fetched}
        files: list[SourceFile] = []
        for spec in self.manifest.files:
            got = by_name[spec.name]
            if (got.sha256, got.size) == (spec.sha256, spec.size):
                files.append(spec)
                continue
            update: dict[str, Any] = {"sha256": got.sha256, "size": got.size, "fetched": day}
            if got.version_id and got.final_url:
                update["url"] = _versioned(got.final_url, got.version_id)
                update["latest"] = spec.latest or spec.url
            files.append(spec.model_copy(update=update))
        if files == list(self.manifest.files):
            return False
        total = sum(f.size or 0 for f in files)
        budget = max(self.manifest.budget_bytes, total)
        updated = SourceManifest.model_validate(
            {
                **self.manifest.model_dump(mode="json", by_alias=True, exclude_none=True),
                "files": [f.model_dump(mode="json", exclude_none=True) for f in files],
                "budget_bytes": budget,
            }
        )
        write_manifest(manifest_path(self.settings.data_dir, self.name), updated)
        self.manifest = updated
        return True

    def _check_total_budget(self) -> None:
        limit = self.shared.total_budget_bytes
        held = sum(p.stat().st_size for p in self.settings.catalogs_dir.rglob("*") if p.is_file())
        mine = sum(p.stat().st_size for p in self.raw_dir.rglob("*") if p.is_file()) if self.raw_dir.exists() else 0
        incoming = sum(f.size or 0 for f in self.manifest.files)
        if held - mine + incoming > limit:
            raise BudgetError(
                f"fetching {self.name} would hold {held - mine + incoming} bytes under {self.settings.catalogs_dir}, "
                f"over the {limit} byte total",
                hint="remove indexes or sources you no longer need, or raise total_budget_bytes in _shared.yaml",
            )

    # -- reading -------------------------------------------------------------------------------------------------------

    def raw_path(self, name: str) -> Path:
        """A fetched raw file, or NotFetchedError naming the command that fetches it."""
        self.manifest.file(name)
        path = self.raw_dir / name
        if not path.is_file():
            raise NotFetchedError(
                f"{self.name}: {name} is not under {self.raw_dir}", hint=f"run `carttrace sandbox catalogs fetch {self.name}`"
            )
        return path

    def raw_sha256(self) -> dict[str, str]:
        """sha256 of each raw file as it is on disk now (cached by size and modification time)."""
        digests = RawDigests(self.raw_dir)
        return {f.name: digests.sha256(self.raw_path(f.name).name) for f in self.manifest.files}

    def iter_products(self, category_slug: str | None = None) -> Iterator[ProductRecord]:
        """Products in one category (all mapped categories when None). A category is read from the source's current
        index when there is one, which holds exactly these records in the same order, so asking for one category of
        Lazada does not re-read 2.7 million documents."""
        self.require_allowed()
        if category_slug is not None and category_slug not in self.categories.categories():
            raise NotFoundError(
                f"{self.name} maps nothing to category {category_slug!r}",
                hint=f"its categories: {self.categories.categories()}",
            )
        if category_slug is not None:
            from carttrace.sandbox.catalogs.index import current_index

            index = current_index(self)
            if index is not None:
                if category_slug in index.manifest.categories:
                    yield from (ProductRecord.from_line(line) for line in index.lines(category_slug))
                return
        for record in self.records():
            if category_slug is None or record.category == category_slug:
                yield record

    def records(self) -> Iterator[ProductRecord]:
        """Every product the source maps to one of our categories, in a deterministic order."""
        raise NotImplementedError

    def note_unplaced(self, key: str, title: str) -> None:
        """Count a row whose key the category map does not place, keeping a few titles as examples. Not one of the
        RECORD_HELPERS: it changes no record, so a parser may call it without changing any index."""
        found = self.unplaced.get(key)
        if found is None:
            found = self.unplaced[key] = Unplaced(key)
        found.rows += 1
        if len(found.examples) < UNPLACED_EXAMPLES and title and title not in found.examples:
            found.examples.append(title)

    # -- helpers for subclasses ----------------------------------------------------------------------------------------

    def category_of(self, value: str | Sequence[str]) -> str | None:
        """Our slug for a source category value, or None when it is unmapped or excluded."""
        found = self.categories.assign(value)
        return None if found in (None, EXCLUDED) else found

    def price(self, product_id: str, category: str, key: str | Sequence[str]) -> tuple[int, str]:
        """A synthetic price for a product (minor units of the rule's currency) and its provenance note."""
        rule = self.shared.synthetic.price
        if rule.currency != self.manifest.currency:
            raise CatalogError(
                f"{self.name} prices in {self.manifest.currency}, synthetic prices are in {rule.currency}",
                hint="synthetic prices only suit sources sold in the rule's currency",
            )
        band = band_for(rule, category, self.categories.band(key))
        return synthetic_price(rule, band, source=self.name, product_id=product_id), price_note(rule, band)

    def stock(self, variant_id: str) -> int:
        return synthetic_stock(self.shared.synthetic.stock, source=self.name, variant_id=variant_id)

    def stock_note(self) -> str:
        return stock_note(self.shared.synthetic.stock)

    def variant(
        self,
        variant_id: str,
        *,
        title: str,
        options: dict[str, str],
        price: int,
        sku: str | None = None,
        barcode: str | None = None,
    ) -> VariantRecord:
        """A variant with a synthetic shelf; available exactly when the shelf is not empty."""
        stock = self.stock(variant_id)
        currency = self.manifest.currency or ""
        return VariantRecord(
            id=variant_id,
            title=title,
            options=options,
            price=price,
            currency=currency,
            available=stock > 0,
            stock=stock,
            sku=sku,
            barcode=barcode,
        )

    def provenance(
        self,
        upstream_ids: Sequence[str],
        *,
        synthetic: dict[str, str] | None = None,
        derived: dict[str, str] | None = None,
        upstream_url: str | None = None,
    ) -> Provenance:
        return Provenance(
            source=self.name,
            licence=self.licence.spdx,
            upstream_ids=list(upstream_ids),
            synthetic=dict(synthetic or {}),
            derived=dict(derived or {}),
            upstream_url=upstream_url,
        )

    def gid(self, kind: str, local_id: str) -> str:
        """A Shopify-style global id, so sandbox ids read like the live ones: gid://<source>/<kind>/<id>."""
        return f"gid://{self.name}/{kind}/{local_id}"


def _versioned(url: str, version_id: str) -> str:
    base = url.split("?", 1)[0]
    return f"{base}?versionId={version_id}"


# -- choosing a source by name -----------------------------------------------------------------------------------------

SourceFactory = Callable[..., CatalogSource]


def source_factory(name: str, plugins: dict[str, dict[str, str]] | None = None) -> SourceFactory:
    """The class registered as `name` for kind sandbox.catalog in carttrace.sandbox's PLUGINS, through `carttrace.core.registry`.
    An explicit `plugins` table wins (tests, experiments)."""
    if plugins is None:
        return cast(SourceFactory, resolve_component(KIND, name))
    target = plugins.get(KIND, {}).get(name)
    if target is None:
        raise ConfigError(
            f"unknown {KIND} implementation {name!r}", hint=f"choose one of {sorted(plugins.get(KIND, {}))}"
        )
    module_name, _, attr = target.partition(":")
    return cast(SourceFactory, getattr(importlib.import_module(module_name), attr))


def open_source(
    name: str, *, settings: Settings | CatalogSettings | None = None, plugins: dict[str, dict[str, str]] | None = None
) -> BaseSource:
    """The named source, ready to fetch or read. Restricted sources open, but refuse to fetch or read unless
    allowed."""
    cfg = resolve(settings)
    if name == "replica":
        raise CatalogError(
            "replica is not a licensed catalog: its stores are captured from the stores themselves",
            hint="read captures with carttrace.sandbox.catalogs.open_replica(), or build one with `carttrace sandbox replica build`",
        )
    manifest = read_manifest(manifest_path(cfg.data_dir, name))
    if manifest.kind != "catalog":
        raise NotFoundError(f"{name} is a reference dataset, not a catalog", hint="it records a licence only")
    factory = source_factory(name, plugins)
    source = factory(
        manifest=manifest,
        categories=load_category_map(name, data_dir=cfg.data_dir, categories_dir=cfg.categories_dir),
        shared=load_shared(cfg.data_dir),
        settings=cfg,
    )
    if not isinstance(source, BaseSource):
        raise ConfigError(f"{name} does not build a catalog source", hint="register a BaseSource subclass")
    return source

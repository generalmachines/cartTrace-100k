"""Source manifests, `data/catalogs/<source>.json`: where a catalog comes from, pinned, and under what licence.

Why pinned like the protocol lock. A sandbox store is only reproducible if the catalog it was cut from is: the same
URL, the same bytes. Each file a source downloads is listed with its official URL, sha256 and size, and `fetch`
refuses bytes that differ (`carttrace.sandbox.catalogs.download`). Some upstreams replace a file in place every day (the
Open Food Facts and Open Beauty Facts dumps); such a file is marked `moving`, and taking a newer copy is an explicit
`--relock`, which rewrites the pin here. Where upstream keeps versions (the Open Food Facts S3 bucket answers with an
`x-amz-version-id`), the relock pins the versioned URL, so the file stays fetchable byte for byte.

A manifest of kind `reference` records a licence for a dataset we use without building stores from it (ESCI queries,
Shopify's taxonomy), so `licence_rights` covers it too. A manifest of kind `pool` (`open`, D91) names the catalogs a
pooled store draws its products from, and the licence its bundles carry: what every member allows, with every
member's attribution (carttrace.sandbox.catalogs.pools refuses a pool licence that grants more than a member's).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError

from carttrace.config.schema import Settings
from carttrace.core.canon import loads, pretty
from carttrace.core.files import atomic_write_text
from carttrace.sandbox.catalogs.errors import CatalogError
from carttrace.sandbox.catalogs.licence import Licence
from carttrace.sandbox.catalogs.settings import CatalogSettings, resolve
from carttrace.sandbox.core.market import Market, market_for
from carttrace.sandbox.envpack.doc import MarketDoc

MANIFEST_SCHEMA: Final = "mc.catalog-source.v1"
SourceName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{0,31}$")]
FileName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
HttpsUrl = Annotated[str, Field(pattern=r"^https://[^\s]+$")]
Day = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceFile(_Frozen):
    """One file a source downloads, pinned."""

    name: FileName
    """Name under <catalogs_dir>/<source>/raw/."""
    url: HttpsUrl
    """What `fetch` downloads: the official file, at a versioned URL where upstream offers one."""
    sha256: Sha256 | None = None
    """The bytes `fetch` accepts. None only before the first `--relock`."""
    size: int | None = Field(default=None, ge=0)
    fetched: Day | None = None
    """UTC day the pinned bytes were downloaded."""
    latest: HttpsUrl | None = None
    """Where `--relock` looks for a newer copy, when that differs from `url`."""
    moving: bool = False
    """Upstream replaces the file at `url` in place, so a later fetch may need `--relock`."""
    note: str = ""


class StoreRule(_Frozen):
    """What counts as one store in this source."""

    unit: Literal["brand", "seller", "partition"]
    """brand (ABO, Open Food Facts), seller (Lazada shops), or partition (no seller or brand: deterministic buckets)."""
    partitions: int = Field(default=8, ge=1)
    """For unit partition: buckets per category."""
    min_products: int = Field(default=3, ge=1)
    """A store needs at least this many products in a category to be offered as a store."""


class SourceManifest(_Frozen):
    schema_: Literal["mc.catalog-source.v1"] = Field(default=MANIFEST_SCHEMA, alias="schema")
    name: SourceName
    title: str = Field(min_length=1)
    kind: Literal["catalog", "reference", "pool"]
    homepage: HttpsUrl | None = None
    """Where the dataset lives; every catalog and reference has one, a pool has none of its own."""
    members: list[SourceName] = Field(default_factory=list)
    """A pool's catalogs (kind pool only)."""
    licence: Licence
    files: list[SourceFile] = Field(default_factory=list)
    budget_bytes: int = Field(default=0, ge=0)
    """The most this source may download; fetch refuses anything larger."""
    market: Annotated[str, Field(pattern=r"^[A-Z]{2}$")] | None = None
    """Sandbox market code the stores sell in (carttrace.sandbox.core.market), or the code of `market_row`."""
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | None = None
    market_row: MarketDoc | None = None
    """A market the sandbox table lacks (PH for Lazada), carried whole."""
    prices: Literal["source", "synthetic"] | None = None
    """source: the catalog's own prices; synthetic: seeded prices from data/catalogs/_shared.yaml."""
    stores: StoreRule | None = None
    countries: list[str] = Field(default_factory=list)
    """Open Food Facts style country tags a product must carry to be kept (empty keeps all)."""
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _complete(self) -> Self:
        names = [f.name for f in self.files]
        if len(names) != len(set(names)):
            raise ValueError("file names repeat")
        if self.kind == "pool":
            if not self.members or self.files or len(set(self.members)) != len(self.members):
                raise ValueError("a pool names its member catalogs, once each, and downloads no files")
            return self
        if self.members:
            raise ValueError("only a pool names members")
        if self.homepage is None:
            raise ValueError("a catalog or reference needs a homepage")
        if self.kind == "reference":
            return self
        missing = [k for k in ("market", "currency", "prices", "stores") if getattr(self, k) is None]
        if missing or not self.files:
            raise ValueError(f"a catalog source needs files, market, currency, prices and stores; missing {missing}")
        pinned = sum(f.size or 0 for f in self.files)
        if pinned > self.budget_bytes:
            raise ValueError(f"pinned files total {pinned} bytes, over budget_bytes {self.budget_bytes}")
        if self.market_row is not None and (
            self.market_row.code != self.market or self.market_row.currency != self.currency
        ):
            raise ValueError("market_row must have the manifest's market code and currency")
        return self

    def sandbox_market(self) -> Market:
        """The market row stores from this source sell in, priced in the catalog's currency."""
        if self.market_row is not None:
            return self.market_row.to_market()
        return market_for(self.market, self.currency)

    def file(self, name: str) -> SourceFile:
        for f in self.files:
            if f.name == name:
                return f
        raise CatalogError(f"{self.name} lists no file {name!r}", hint=f"files: {[f.name for f in self.files]}")


def manifest_path(data_dir: Path, name: str) -> Path:
    return data_dir / f"{name}.json"


def read_manifest(path: Path) -> SourceManifest:
    """A manifest file, validated. Refuses a file whose name is not its `name`."""
    try:
        raw = loads(path.read_bytes())
    except FileNotFoundError:
        raise CatalogError(f"no source manifest {path}", hint="list sources with `carttrace sandbox catalogs list`") from None
    except ValueError as exc:
        raise CatalogError(f"{path.name} is not JSON: {exc}") from None
    try:
        manifest = SourceManifest.model_validate(raw)
    except PydanticValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "document"
        raise CatalogError(f"{path.name}: {where}: {first['msg']}", hint="fix the manifest by hand") from None
    if path.stem != manifest.name:
        raise CatalogError(f"{path.name} describes {manifest.name!r}", hint="name the file <name>.json")
    return manifest


def load_manifest(name: str, *, settings: Settings | CatalogSettings | None = None) -> SourceManifest:
    return read_manifest(manifest_path(resolve(settings).data_dir, name))


def list_manifests(*, settings: Settings | CatalogSettings | None = None) -> list[SourceManifest]:
    """Every source manifest under data/catalogs, sorted by name. Files starting with `_` and category or coverage
    files are not manifests."""
    data_dir = resolve(settings).data_dir
    paths = sorted(p for p in data_dir.glob("*.json") if not p.name.startswith("_") and p.stem.count(".") == 0)
    return [read_manifest(p) for p in paths]


def write_manifest(path: Path, manifest: SourceManifest) -> None:
    """Rewrite a manifest after a relock: indented canonical JSON, so the diff shows only the pin that moved."""
    atomic_write_text(path, pretty(manifest.model_dump(mode="json", by_alias=True, exclude_none=True)))

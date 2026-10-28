"""Per-category indexes of a source: normalised products, one gzipped JSON Lines shard per category.

Why an index. Reading a raw source is slow (Lazada is 2.7 million documents in 1.5 GB of gzip, Open Food Facts a
1.3 GB CSV), and a store is cut from one category of it. `build_index` reads the source once and writes, for each of
our categories, every normalised product (`<category>.jsonl.gz`) and the stores in it with their product counts
(`<category>.stores.json`), plus a manifest with coverage: products, variants, stores and the share of products
stating each decision-critical detail. Building a store then reads one shard.

Why keyed by its inputs. An index is named by a key over everything that decides its bytes: the raw files' sha256,
the parser's version and source code, the category map and the synthetic rules. It is built in a temporary
directory, moved to `index/idx-<key>`, and only then named in `index/CURRENT`, so a reader sees a whole index or
none, and an index built from other inputs or by other code is never read. The same inputs always give the same
shard bytes (gzip is written with no name and a zero time). Packs record the index's `content` hash instead of its
key, so a code change that produces the same products leaves every pack id as it was.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import importlib
import inspect
import os
import shutil
import textwrap
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from carttrace.core.canon import digest, dumps, loads, pretty
from carttrace.core.files import atomic_write_text
from carttrace.sandbox.catalogs.categories import known_categories
from carttrace.sandbox.catalogs.errors import CatalogError, NotCoveredError
from carttrace.sandbox.catalogs.records import DETAILS, ProductRecord
from carttrace.sandbox.catalogs.source import BaseSource

INDEX_SCHEMA: Final = "mc.catalog-index.v1"
INDEX_DIR = "index"
CURRENT = "CURRENT"
MANIFEST = "manifest.json"
BP = 10_000


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CategoryCount(_Frozen):
    sha256: str
    """sha256 of the category's records as written (uncompressed canonical lines): what the shard holds."""
    products: int = Field(ge=0)
    variants: int = Field(ge=0)
    stores: int = Field(ge=0)
    stores_at_least_20: int = Field(ge=0)
    """Stores with 20 or more products in the category: the ones a typical snapshot can fill."""
    details_bp: dict[str, int] = Field(default_factory=dict)
    """Share of products stating each detail, in basis points."""


class IndexManifest(_Frozen):
    schema_: Literal["mc.catalog-index.v1"] = Field(default=INDEX_SCHEMA, alias="schema")
    source: str
    key: str
    """A hash of the inputs, the parser's code included: an index whose key differs from today's is never read."""
    content: str
    """A hash of what the index holds (every category's records): the same products give the same content, whatever
    code produced them, so packs name this and not `key`."""
    normaliser: str
    files: dict[str, str]
    """Raw file name to sha256: the bytes this index was built from."""
    category_map: str
    synthetic: str
    categories: dict[str, CategoryCount]
    skipped: dict[str, int] = Field(default_factory=dict)
    """Upstream rows left out, by reason (no English title, unmapped category, ...)."""


PARSER_MODULES = (
    "carttrace.sandbox.catalogs.categories",
    "carttrace.sandbox.catalogs.records",
    "carttrace.sandbox.catalogs.synthetic",
    "carttrace.sandbox.catalogs.text",
)
"""Modules every parser's records depend on, besides the parser's own: category matching, the record shape, the
synthetic rules and the text detectors."""
RECORD_HELPERS = ("category_of", "price", "stock", "variant", "provenance", "gid")
"""The `BaseSource` methods parsers build records with. The rest of `BaseSource` (fetching, gating, reading an
index) does not change a record, so editing it rebuilds nothing."""


def code_digest(source: BaseSource) -> str:
    """A hash of the code that decides a source's records: the parser's module, `PARSER_MODULES` and the
    `RECORD_HELPERS`, so changing a parser rebuilds its index even when nobody remembered to bump NORMALISER.
    Docstrings and comments are left out: rewording one never invalidates a gigabyte of index."""
    modules = [_code_of(name) for name in sorted({type(source).__module__, *PARSER_MODULES})]
    helpers = [_code_of_object(getattr(BaseSource, name)) for name in RECORD_HELPERS]
    return digest([*modules, *helpers], 16)


def modules_digest(*modules: str) -> str:
    """A hash of named modules' code, docstrings and comments left out, for a key that must change with the code that
    decides a derived file's bytes (the pools' vetting, D104)."""
    return digest([_code_of(name) for name in sorted(modules)], 16)


def _code_of(module: str) -> str:
    """A module's code without docstrings or comments; its name alone when the source is not on disk."""
    try:
        return _stripped(inspect.getsource(importlib.import_module(module)))
    except (OSError, TypeError):
        return module


def _code_of_object(obj: object) -> str:
    try:
        return _stripped(textwrap.dedent(inspect.getsource(obj)))  # type: ignore[arg-type]
    except (OSError, TypeError):
        return repr(obj)


def _stripped(source: str) -> str:
    """Code as an AST dump without docstrings (module, class, function and attribute ones: bare strings do
    nothing)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and any(_is_docstring(stmt) for stmt in body):
            node.body = [stmt for stmt in body if not _is_docstring(stmt)] or [ast.Pass()]  # type: ignore[attr-defined]
    return ast.dump(tree, annotate_fields=False)


def _is_docstring(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def index_key(source: BaseSource) -> str:
    """A hash of every input that decides the index's bytes."""
    m = source.manifest
    return digest(
        {
            "normaliser": source.NORMALISER,
            "code": code_digest(source),
            "files": source.raw_sha256(),
            "map": source.categories.digest(),
            "synthetic": source.shared.digest(),
            "market": [m.market, m.currency, m.prices, m.countries],
            "stores": m.stores.model_dump(mode="json") if m.stores else None,
        },
        16,
    )


@dataclass(frozen=True, slots=True)
class CatalogIndex:
    """A built, current index of one source."""

    path: Path
    manifest: IndexManifest

    def shard(self, category: str) -> Path:
        return self.path / f"{category}.jsonl.gz"

    def categories(self) -> list[str]:
        return [c for c, n in self.manifest.categories.items() if n.products]

    def stores(self, category: str) -> dict[str, int]:
        """Store key to its product count in `category`, sorted by key."""
        path = self.path / f"{category}.stores.json"
        if not path.is_file():
            raise NotCoveredError(
                f"{self.manifest.source} has no products in {category!r}", hint=f"covered: {self.categories()}"
            )
        data = loads(path.read_bytes())
        if not isinstance(data, dict):
            raise CatalogError(f"{path} is not a stores table", hint="rebuild the index")
        return {str(k): int(v) for k, v in data.items()}

    def lines(self, category: str) -> Iterator[bytes]:
        path = self.shard(category)
        if not path.is_file():
            raise NotCoveredError(f"{self.manifest.source} has no products in {category!r}")
        with gzip.open(path, "rb") as fh:
            for line in fh:
                if line.strip():
                    yield line

    def products(self, category: str, store: str | None = None) -> list[ProductRecord]:
        """Products of a category (one store's when `store` is given), sorted by id. Only matching lines are parsed
        as records; the store test reads the raw line first."""
        needle = None if store is None else f'"store":{_json_string(store)}'.encode()
        out = [ProductRecord.from_line(line) for line in self.lines(category) if needle is None or needle in line]
        return sorted((p for p in out if store is None or p.store == store), key=lambda p: p.id)


def current_index(source: BaseSource) -> CatalogIndex | None:
    """The source's index if one exists and was built from exactly the current inputs, else None."""
    pointer = source.root / INDEX_DIR / CURRENT
    try:
        name = pointer.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    path = source.root / INDEX_DIR / name
    try:
        manifest = IndexManifest.model_validate(loads((path / MANIFEST).read_bytes()))
    except (FileNotFoundError, ValueError):
        return None
    if manifest.key != index_key(source):
        return None
    return CatalogIndex(path, manifest)


def ensure_index(source: BaseSource) -> CatalogIndex:
    """The current index, built first when there is none or the inputs changed. A restricted source's index is
    opened only with the research allowance, like its raw files."""
    source.require_allowed()
    return current_index(source) or build_index(source)


def build_index(source: BaseSource) -> CatalogIndex:
    """Read the whole source once and write its per-category index. Returns the new current index."""
    source.require_allowed()
    key = index_key(source)
    base = source.root / INDEX_DIR
    final = base / f"idx-{key}"
    _sweep_temporaries(base)
    tmp = base / f".tmp-{key}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    writers: dict[str, _Shard] = {}
    try:
        for record in source.records():
            shard = writers.get(record.category)
            if shard is None:
                shard = writers[record.category] = _Shard(tmp, record.category)
            shard.add(record)
        counts = {category: writers[category].close() for category in sorted(writers)}
        manifest = IndexManifest(
            source=source.name,
            key=key,
            content=digest({category: count.sha256 for category, count in counts.items()}, 16),
            normaliser=source.NORMALISER,
            files=source.raw_sha256(),
            category_map=source.categories.digest(),
            synthetic=source.shared.digest(),
            categories=counts,
            skipped=dict(sorted(source.skipped.items())),
        )
        atomic_write_text(tmp / MANIFEST, pretty(manifest.model_dump(mode="json", by_alias=True)))
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
    except BaseException:
        for shard in writers.values():
            shard.abort()
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    atomic_write_text(base / CURRENT, final.name + "\n")
    for old in base.glob("idx-*"):
        if old != final and old.is_dir():
            shutil.rmtree(old, ignore_errors=True)
    return CatalogIndex(final, manifest)


def _sweep_temporaries(base: Path) -> None:
    """Remove the temporary directories of builds whose process is gone (killed mid-build), never a live one's."""
    for tmp in base.glob(".tmp-*-*"):
        pid = tmp.name.rsplit("-", 1)[-1]
        if pid.isdigit() and not _alive(int(pid)):
            shutil.rmtree(tmp, ignore_errors=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _Shard:
    """One category's gzipped JSON Lines file, with its counts kept on the way."""

    def __init__(self, directory: Path, category: str) -> None:
        self._dir = directory
        self._category = category
        self._raw: IO[bytes] = (directory / f"{category}.jsonl.gz").open("wb")
        self._gz = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
        self._sha = hashlib.sha256()
        self.products = 0
        self.variants = 0
        self.stores: Counter[str] = Counter()
        self.details: Counter[str] = Counter()

    def add(self, record: ProductRecord) -> None:
        line = record.to_line().encode("utf-8") + b"\n"
        self._sha.update(line)
        self._gz.write(line)
        self.products += 1
        self.variants += len(record.variants)
        self.stores[record.store] += 1
        self.details.update(record.details.present())

    def close(self) -> CategoryCount:
        self._gz.close()
        self._raw.close()
        stores = dict(sorted(self.stores.items()))
        atomic_write_text(self._dir / f"{self._category}.stores.json", pretty(stores))
        return CategoryCount(
            sha256=self._sha.hexdigest(),
            products=self.products,
            variants=self.variants,
            stores=len(stores),
            stores_at_least_20=sum(1 for n in stores.values() if n >= 20),
            details_bp={d: self.details[d] * BP // self.products for d in DETAILS if self.details[d]},
        )

    def abort(self) -> None:
        for handle in (self._gz, self._raw):
            try:
                handle.close()
            except (OSError, ValueError):
                continue


COVERAGE_SCHEMA = "mc.catalog-coverage.v1"


def coverage_path(source: BaseSource) -> Path:
    """data/catalogs/<source>.coverage.json, checked in beside the manifest; for a restricted source the file stays
    under catalogs_dir instead (decision D11: its rates are never quoted as store norms)."""
    if source.licence.restricted:
        return source.root / "coverage.json"
    return source.settings.data_dir / f"{source.name}.coverage.json"


def coverage_document(source: BaseSource, index: CatalogIndex, known: list[str]) -> dict[str, object]:
    """What an index covers: per category the products, variants, stores and detail shares, which of our categories
    the source does not cover at all, and the licence the numbers come under."""
    m = index.manifest
    lic = source.licence
    return {
        "schema": COVERAGE_SCHEMA,
        "source": m.source,
        "licence": {
            "spdx": lic.spdx,
            "restricted": lic.restricted,
            "training_allowed": lic.training_allowed,
            "publication_allowed": lic.publication_allowed,
        },
        "content": m.content,
        "normaliser": m.normaliser,
        "files": m.files,
        "categories": {c: n.model_dump(mode="json") for c, n in m.categories.items()},
        "not_covered": [c for c in known if c not in index.categories()],
        "skipped": m.skipped,
    }


def record_coverage(source: BaseSource, index: CatalogIndex) -> Path:
    """Write the coverage document (`coverage_path`). Deterministic: the same index, the same bytes."""
    path = coverage_path(source)
    known = known_categories(source.settings.categories_dir)
    atomic_write_text(path, pretty(coverage_document(source, index, known)))
    return path


def _json_string(text: str) -> str:
    return dumps(text)

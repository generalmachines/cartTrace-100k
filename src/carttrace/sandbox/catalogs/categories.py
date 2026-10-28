"""Category maps, `data/catalogs/<source>.categories.yaml`: a source's own categories onto our 25.

Why a map per source, as data. Benchmark categories are the 25 of the store queue (data/categories/*.yaml), and
every catalog files its products differently: ABO by product type, WANDS and Lazada by a category path, Open Food
Facts by tags. Which of our categories a source category belongs to is a judgement (are kitchen appliances Kitchen /
dining or Consumer electronics?), so the maps are YAML with the reasons in comments, and this module only applies
them. A map that names an unknown category, or files one key under two categories, is refused at load time.

How a key is matched, by the map's `match`:
- exact: the source value equals the key (ABO product_type);
- prefix: the source path starts with the key on whole segments, and the longest key wins (Lazada, WANDS);
- tags: the product carries several tags, and the first key in file order that it carries wins (Open Food Facts).

`(excluded)` collects keys that are deliberately out of scope (alcohol, digital goods, fabric swatches). A product
whose keys match nothing is unmapped; both are counted in coverage and never reach a store.

An optional `price_bands` block narrows synthetic prices for source keys whose category band is too wide (a phone
case is Consumer electronics, but not a 700 dollar one), matched the same way.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from pydantic import ValidationError as PydanticValidationError

from carttrace.core.canon import digest
from carttrace.sandbox.catalogs.errors import CatalogError

MAP_SCHEMA: Final = "mc.catalog-categories.v1"
EXCLUDED = "(excluded)"
Match = Literal["exact", "prefix", "tags"]


class CategoryMap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_: Literal["mc.catalog-categories.v1"] = Field(default=MAP_SCHEMA, alias="schema")
    source: str
    match: Match
    field: str
    """What the keys are matched against, for readers: product_type, category path, categories_tags."""
    separator: str = " > "
    """Path separator of the source's category paths (prefix maps)."""
    rules: dict[str, list[str]]
    """Our category slug, or (excluded), to the source keys filed under it. File order is the tags priority."""
    price_bands: dict[str, Annotated[list[int], Field(min_length=2, max_length=2)]] = Field(default_factory=dict)
    """Source key to [low, high] in whole units of the synthetic price currency."""

    _lookup: dict[str, str] = PrivateAttr(default_factory=dict)
    _order: dict[str, int] = PrivateAttr(default_factory=dict)
    _cache: dict[str, str | None] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _one_category_per_key(self) -> Self:
        seen: dict[str, str] = {}
        for category, keys in self.rules.items():
            for key in keys:
                if not key.strip():
                    raise ValueError(f"{category}: an empty key")
                if key in seen:
                    raise ValueError(f"key {key!r} is filed under both {seen[key]!r} and {category!r}")
                seen[key] = category
        for key, band in self.price_bands.items():
            if not 0 < band[0] <= band[1]:
                raise ValueError(f"price band for {key!r} must be 0 < low <= high, got {band}")
        return self

    def model_post_init(self, context: Any, /) -> None:
        position = 0
        for category, keys in self.rules.items():
            for key in keys:
                self._lookup[self._norm(key)] = category
                self._order[self._norm(key)] = position
                position += 1

    # -- matching ------------------------------------------------------------------------------------------------------

    def assign(self, value: str | Sequence[str]) -> str | None:
        """Our category slug for a source value, EXCLUDED, or None when nothing matches.

        exact and prefix take one string (a path for prefix); tags takes the product's tags in any order.
        """
        if self.match == "tags":
            tags = [value] if isinstance(value, str) else list(value)
            hits = [(self._order[t], self._lookup[t]) for t in (self._norm(x) for x in tags) if t in self._lookup]
            return min(hits)[1] if hits else None
        if not isinstance(value, str):
            raise CatalogError(f"{self.source}: a {self.match} map assigns one value, got {type(value).__name__}")
        if value not in self._cache:
            exact = self.match == "exact"
            self._cache[value] = self._lookup.get(self._norm(value)) if exact else self._longest_prefix(value)
        return self._cache[value]

    def band(self, value: str | Sequence[str]) -> tuple[int, int] | None:
        """The price band for a source value from `price_bands`, matched like `assign`."""
        key = self._band_key(value)
        if key is None:
            return None
        lo, hi = self.price_bands[key]
        return lo, hi

    def _band_key(self, value: str | Sequence[str]) -> str | None:
        if self.match == "tags":
            tags = {value} if isinstance(value, str) else set(value)
            return next((k for k in self.price_bands if k in tags), None)
        if not isinstance(value, str):
            return None
        if self.match == "exact":
            return value if value in self.price_bands else None
        return max((k for k in self.price_bands if self._starts(value, k)), key=len, default=None)

    def categories(self) -> list[str]:
        """Our categories this source files products under, in file order."""
        return [c for c in self.rules if c != EXCLUDED and self.rules[c]]

    def digest(self) -> str:
        """A hash of what the map decides (rules, matching and bands), for index keys."""
        return digest(
            {"match": self.match, "separator": self.separator, "rules": self.rules, "bands": self.price_bands}, 16
        )

    def split(self, path: str) -> list[str]:
        """A path's segments. Splits on the separator with its spaces, so "Indoor/Outdoor" stays one segment."""
        return [part.strip() for part in path.split(self.separator) if part.strip()]

    def _norm(self, key: str) -> str:
        return self.separator.join(self.split(key)) if self.match == "prefix" else key.strip()

    def _starts(self, path: str, prefix: str) -> bool:
        parts, want = self.split(path), self.split(prefix)
        return bool(want) and parts[: len(want)] == want

    def _longest_prefix(self, path: str) -> str | None:
        parts = self.split(path)
        for n in range(len(parts), 0, -1):
            hit = self._lookup.get(self.separator.join(parts[:n]))
            if hit is not None:
                return hit
        return None


def known_categories(categories_dir: Path) -> list[str]:
    """The slugs of the 25 benchmark categories: the names of data/categories/*.yaml, without reading carttrace.stores
    (a higher layer) or the files' contents."""
    slugs = sorted(p.stem for p in categories_dir.glob("*.yaml") if not p.name.startswith("_"))
    if not slugs:
        raise CatalogError(f"no category files in {categories_dir}", hint="restore data/categories from git")
    return slugs


def map_path(data_dir: Path, source: str) -> Path:
    return data_dir / f"{source}.categories.yaml"


def load_category_map(source: str, *, data_dir: Path, categories_dir: Path) -> CategoryMap:
    """A source's map, validated against the known categories."""
    path = map_path(data_dir, source)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CatalogError(f"no category map {path}", hint=f"write {path.name} beside {source}.json") from None
    except yaml.YAMLError as exc:
        raise CatalogError(f"{path.name} is not valid YAML: {exc}") from None
    try:
        cmap = CategoryMap.model_validate(raw)
    except PydanticValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "document"
        raise CatalogError(f"{path.name}: {where}: {first['msg']}") from None
    if cmap.source != source:
        raise CatalogError(f"{path.name} maps source {cmap.source!r}", hint=f"set source: {source}")
    _check_slugs(path.name, cmap.rules, known_categories(categories_dir))
    return cmap


def _check_slugs(where: str, rules: dict[str, list[str]], known: Iterable[str]) -> None:
    allowed = {*known, EXCLUDED}
    unknown = sorted(c for c in rules if c not in allowed)
    if unknown:
        raise CatalogError(
            f"{where} names categories that are not in data/categories: {unknown}",
            hint="use the slugs of data/categories/*.yaml, or (excluded)",
        )

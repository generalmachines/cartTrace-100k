"""Synthetic prices and stock for sources that publish neither, drawn from seeded rules and always labelled.

Why synthetic, and why seeded. ABO, WANDS, Open Food Facts and Open Beauty Facts describe products but sell nothing,
and a sandbox store needs a price and a shelf for every variant. The draws come from `carttrace.core.ids.seeded_rng`, keyed
by the source and the product or variant id, so the same product always gets the same price in every store and on
every machine, and no float reaches the result: a price is one of a fixed ladder of retail price points (9.99, 24.99,
...) inside the product's band, picked with integer draws. The rules live in data/catalogs/_shared.yaml; every
generated field is named in the product's `provenance.synthetic` with the rule and band that made it, so nothing
synthetic can pass for a merchant's own fact.

Bands are whole units of the rule's currency per category, narrowed per source key by a map's `price_bands` (a phone
case is Consumer electronics, but costs tens, not hundreds).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError

from carttrace.core.canon import digest
from carttrace.core.ids import seeded_rng
from carttrace.core.money import to_minor
from carttrace.sandbox.catalogs.errors import CatalogError

SHARED_FILE = "_shared.yaml"
SHARED_SCHEMA: Final = "mc.catalog-shared.v1"
Band = Annotated[list[int], Field(min_length=2, max_length=2)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PriceRule(_Frozen):
    rule: str = Field(min_length=1)
    """Name recorded in provenance, e.g. synthetic-price.v1. Change it whenever the ladder or bands change."""
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    ladder: list[int] = Field(min_length=2)
    """Retail price points in minor units, ascending."""
    bands: dict[str, Band]
    """Category slug to [low, high] in whole units."""
    default_band: Band = Field(default_factory=lambda: [5, 100])

    @model_validator(mode="after")
    def _ascending(self) -> Self:
        if self.ladder != sorted(set(self.ladder)) or self.ladder[0] <= 0:
            raise ValueError("the price ladder must be positive and strictly ascending")
        for name, band in {**self.bands, "default_band": self.default_band}.items():
            if not 0 < band[0] <= band[1]:
                raise ValueError(f"band {name} must be 0 < low <= high, got {band}")
        return self


class StockRule(_Frozen):
    rule: str = Field(min_length=1)
    max: int = Field(ge=1)
    """Largest shelf a variant gets."""
    out_of_stock_bp: int = Field(ge=0, le=10_000)
    """Share of variants drawn with an empty shelf, in basis points."""


class Synthetic(_Frozen):
    price: PriceRule
    stock: StockRule


class Shared(_Frozen):
    """data/catalogs/_shared.yaml: rules every source shares."""

    schema_: Literal["mc.catalog-shared.v1"] = Field(default=SHARED_SCHEMA, alias="schema")
    total_budget_bytes: int = Field(ge=0)
    """The most all sources together may hold under catalogs_dir (downloads and indexes)."""
    synthetic: Synthetic

    def digest(self) -> str:
        return digest(self.synthetic.model_dump(mode="json"), 16)


def load_shared(data_dir: Path) -> Shared:
    path = data_dir / SHARED_FILE
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CatalogError(f"no {path}", hint="restore data/catalogs from git") from None
    except yaml.YAMLError as exc:
        raise CatalogError(f"{SHARED_FILE} is not valid YAML: {exc}") from None
    try:
        return Shared.model_validate(raw)
    except PydanticValidationError as exc:
        first = exc.errors()[0]
        raise CatalogError(f"{SHARED_FILE}: {'.'.join(str(p) for p in first['loc'])}: {first['msg']}") from None


def band_for(rule: PriceRule, category: str, override: tuple[int, int] | None = None) -> tuple[int, int]:
    """The band a product is priced in: the source key's own band if the map gives one, else its category's."""
    if override is not None:
        return override
    lo, hi = rule.bands.get(category, rule.default_band)
    return lo, hi


def synthetic_price(rule: PriceRule, band: tuple[int, int], *, source: str, product_id: str) -> int:
    """A price in minor units of `rule.currency`: one ladder point inside the band, the same for a product forever."""
    lo, hi = (to_minor(b, rule.currency) or 0 for b in band)
    points = [p for p in rule.ladder if lo <= p <= hi]
    if not points:
        points = [min(rule.ladder, key=lambda p: (abs(p - lo), p))]
    rng = seeded_rng("catprice", source, product_id)
    return points[rng.randrange(len(points))]


def synthetic_stock(rule: StockRule, *, source: str, variant_id: str) -> int:
    """Units on the shelf for a variant: zero for a seeded few, else between 1 and `rule.max`."""
    rng = seeded_rng("catstock", source, variant_id)
    if rng.randrange(10_000) < rule.out_of_stock_bp:
        return 0
    return 1 + rng.randrange(rule.max)


def price_note(rule: PriceRule, band: tuple[int, int]) -> str:
    return f"{rule.rule}: a seeded ladder price in {band[0]}-{band[1]} {rule.currency}"


def stock_note(rule: StockRule) -> str:
    return f"{rule.rule}: seeded shelf of 0-{rule.max}, empty for {rule.out_of_stock_bp / 100:g}% of variants"

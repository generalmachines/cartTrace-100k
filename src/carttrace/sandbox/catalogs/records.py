"""The normalised product record every catalog source yields, whatever its upstream format.

Why one record. Five sources (ABO listings, WANDS rows, Open Food Facts and Open Beauty Facts dumps, Lazada documents)
spell a product five ways. Store building, coverage counting and the sandbox surfaces read one shape: a product with
variants priced in integer minor units of one currency, plain-text facts, and the decision-critical details an agent
needs in its category (size and size system, material, care, ingredients, allergens, warnings, dosage,
compatibility, dimensions), each None when the source does not state it.

Why provenance on every product. A sandbox store must never pass synthetic facts off as a merchant's. Where a source
has no prices or stock (ABO, WANDS, Open Food Facts, Open Beauty Facts), they are drawn from seeded rules
(`carttrace.sandbox.catalogs.synthetic`) and `provenance.synthetic` names every generated field and the rule that made it,
next to the source's licence and the upstream ids the product came from. Surfaces show the product, not its
provenance.

Records are written to index shards one canonical JSON line each (`to_line`), so the same catalog gives the same
bytes.
"""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from carttrace.core.canon import dumps, loads

DETAILS = (
    "size",
    "size_system",
    "material",
    "care",
    "ingredients",
    "allergens",
    "warnings",
    "dosage",
    "compatibility",
    "dimensions",
    "weight",
    "pack_size",
)
"""Decision-critical details, in the order coverage reports them."""

Currency = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
Text = Annotated[str, Field(max_length=20_000)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Details(_Frozen):
    """What decides a purchase in a category, as plain text, or None when the source does not say."""

    size: str | None = None
    """The size or sizes on offer ("6, 7, 8 UK", "Queen", "XL")."""
    size_system: str | None = None
    """US, UK, EU, alpha, ... (the vocabulary of data/categories/_fields.yaml size_system)."""
    material: str | None = None
    care: str | None = None
    ingredients: str | None = None
    allergens: str | None = None
    warnings: str | None = None
    dosage: str | None = None
    compatibility: str | None = None
    """What it fits or works with ("Xiaomi Redmi Y2", "Toyota Vios 2019")."""
    dimensions: str | None = None
    weight: str | None = None
    pack_size: str | None = None
    """Net quantity or count ("500 g", "90 tablets")."""

    def present(self) -> list[str]:
        """Names of the details this product states."""
        return [name for name in DETAILS if getattr(self, name)]

    def stated(self) -> dict[str, str]:
        """Only the stated details, in DETAILS order."""
        return {name: str(getattr(self, name)) for name in self.present()}


class VariantRecord(_Frozen):
    id: str = Field(min_length=1, max_length=200)
    title: str
    options: dict[str, str] = Field(default_factory=dict)
    """Option name to value, e.g. {"Color": "Navy", "Size": "7.5 B US"}."""
    price: int = Field(ge=0)
    """Minor units of `currency`."""
    currency: Currency
    available: bool = True
    stock: int = Field(default=0, ge=0)
    sku: str | None = None
    barcode: str | None = None


class Rating(_Frozen):
    value: float = Field(ge=0)
    scale_max: float = Field(default=5, ge=1)
    count: int | None = Field(default=None, ge=0)


class Provenance(_Frozen):
    source: str
    licence: str
    """SPDX id of the source's licence (or LicenseRef-...)."""
    upstream_ids: list[str] = Field(min_length=1)
    """The ids of the upstream rows this product was built from (listings, SKUs)."""
    synthetic: dict[str, str] = Field(default_factory=dict)
    """Generated field to the rule that made it, e.g. {"price": "synthetic-price.v1 band 49-1999 USD"}."""
    derived: dict[str, str] = Field(default_factory=dict)
    """Detail or structure to the upstream field it came from, where that is not obvious ({"care": "bullet_point"})."""
    upstream_url: str | None = None
    """The product's page upstream, for audit only; never shown to an agent."""


class ProductRecord(_Frozen):
    id: str = Field(min_length=1, max_length=200)
    source: str
    category: str
    """Our category slug (data/categories/<slug>.yaml)."""
    category_path: list[str] = Field(default_factory=list)
    """The source's own category, as path segments or a single type."""
    store: str = Field(min_length=1, max_length=200)
    """Store key within the source: a brand, a seller id or a partition."""
    title: Text = Field(min_length=1)
    brand: str | None = None
    description: Text = ""
    attributes: dict[str, str] = Field(default_factory=dict)
    details: Details = Field(default_factory=Details)
    variants: list[VariantRecord] = Field(min_length=1)
    rating: Rating | None = None
    tags: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    """Upstream image references (ABO image ids, Open Food Facts and Lazada image URLs), recorded but never
    downloaded; a text-only agent does not need them, so snapshots leave them out of what a surface shows."""
    provenance: Provenance

    @model_validator(mode="after")
    def _one_currency_unique_variants(self) -> Self:
        if len({v.currency for v in self.variants}) != 1:
            raise ValueError(f"{self.id}: variants are priced in more than one currency")
        ids = [v.id for v in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.id}: variant ids repeat")
        return self

    @property
    def currency(self) -> str:
        return self.variants[0].currency

    def option_names(self) -> list[str]:
        """Option names in the order the variants first use them."""
        names: dict[str, None] = {}
        for v in self.variants:
            names.update(dict.fromkeys(v.options))
        return list(names)

    def to_line(self) -> str:
        """One canonical JSON line (no newline)."""
        return dumps(self.model_dump(mode="json"))

    @classmethod
    def from_line(cls, line: str | bytes) -> ProductRecord:
        return cls.model_validate(loads(line))


def product_facts(record: ProductRecord) -> dict[str, Any]:
    """The product as an envpack snapshot holds it, in UCP's product shape where UCP has a field: title, a plain-text
    description, categories (with taxonomy "merchant"), tags, rating, and `metadata` (UCP's "business-defined custom
    data") holding the store's own `attributes` and the decision-critical `details`, so a surface can show them and an
    experiment can hide one by deleting one key. `brand` and `product_type` are there for readers of Shopify-style
    feeds. `provenance` (licence, synthetic fields, upstream ids) is for audit and must never reach an agent.
    Variants live in the snapshot's variant table, not here."""
    metadata: dict[str, Any] = {}
    if record.attributes:
        metadata["attributes"] = dict(record.attributes)
    stated = record.details.stated()
    if stated:
        metadata["details"] = stated
    facts: dict[str, Any] = {"id": record.id, "title": record.title, "description": record.description}
    if record.brand:
        facts["brand"] = record.brand
    if record.category_path:
        facts["product_type"] = record.category_path[-1]
        facts["categories"] = [{"value": " > ".join(record.category_path), "taxonomy": "merchant"}]
    if record.tags:
        facts["tags"] = list(record.tags)
    if record.rating is not None:
        facts["rating"] = record.rating.model_dump(mode="json", exclude_none=True)
    if metadata:
        facts["metadata"] = metadata
    facts["provenance"] = record.provenance.model_dump(mode="json", exclude_none=True)
    return facts

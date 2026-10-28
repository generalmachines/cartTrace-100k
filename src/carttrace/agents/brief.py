"""The store brief, `mc.brief.v1`: what one store's capture exposes, read by code, so an intent writer can turn on the
data this store has and not the data every store of its category is supposed to have.

Why. The template generator makes one of a few details decisive (variant options, allergens, price, warnings), so a
food store got allergen tasks over and over while the delivery promise, the unit price, the certifications and the
subscription plan it exposes were never tested. The old lab had an analyzer summarise a store before its intent writer wrote tasks; this is that
analyzer, deterministic and without a model: the products, their kinds and option dimensions, the price bands, which
data kinds of the exposure taxonomy (docs/STORE-EXPOSURE.md) the catalog exposes and which it lacks, and twenty to
forty representative products with the decisive facts an intent can turn on, each fact a short quote of the store's
own text. The brief is cached per capture hash, so a store is analysed once per capture.

The data kinds are the rows of the exposure taxonomy an intent can be decided by (`DATA_KINDS`). Each detector reads a
product's normalised text (carttrace.agents.products) and answers with a quote or None; a kind is exposed when enough products
carry it. Detectors reuse the decisive details where one exists (allergens, size charts, warnings), so the brief never
says a store exposes a size chart the size-chart reader cannot read.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from carttrace.agents.details.allergens import ALLERGENS, statements
from carttrace.agents.details.allergens import status as allergen_status
from carttrace.agents.details.size_chart import read_chart
from carttrace.agents.details.warnings import warning_lines
from carttrace.agents.measures import quantity_of
from carttrace.agents.nouns import noun_of
from carttrace.agents.products import Product, Variant, read_products
from carttrace.agents.text import fold
from carttrace.core.canon import digest, dumps, loads
from carttrace.core.errors import ValidationError
from carttrace.core.files import atomic_write_text
from carttrace.trace.capture import StoreCapture

BRIEF_SCHEMA: Final = "mc.brief.v1"
BRIEFS_DIR: Final = Path("agent") / "briefs"
"""Under paths.var: <store>/<source>.json, one brief per capture."""
MIN_REPRESENTATIVES: Final = 20
MAX_REPRESENTATIVES: Final = 40
MAX_VARIANTS_SHOWN: Final = 12
MAX_TAGS_SHOWN: Final = 8
MAX_OPTION_VALUES: Final = 20
QUOTE_CHARS: Final = 160
EXPOSED_SHARE: Final = 0.1
"""A kind counts as exposed when at least this share of products (or three products) carry it."""
EXPOSED_PRODUCTS: Final = 3

Detector = Callable[[Product, Mapping[str, Any]], str | None]
"""(the product, its raw document) -> a quote of the data, or None when the product does not expose the kind."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class KindSpec(_Frozen):
    """One data kind an intent can turn on: its id, the exposure taxonomy row it is, and what it means."""

    kind: str
    row: str
    """The taxonomy row (docs/STORE-EXPOSURE.md section 3), e.g. `price.unit`."""
    about: str
    """What a shopper would ask that this data settles."""


class ExposedKind(_Frozen):
    kind: str
    products: int
    share: float
    examples: list[str] = Field(default_factory=list)
    """Up to three product ids that carry it."""


class BriefVariant(_Frozen):
    id: str
    label: str
    price: int | None
    available: bool


class BriefProduct(_Frozen):
    id: str
    title: str
    kind: str
    """The shopper's noun for it (carttrace.agents.nouns), the value a `kind` predicate takes."""
    brand: str | None = None
    product_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    price_min: int | None = None
    price_max: int | None = None
    variants: list[BriefVariant] = Field(default_factory=list)
    facts: dict[str, str] = Field(default_factory=dict)
    """Data kind to a short quote of the product's own text that carries it."""


class PriceBands(_Frozen):
    min: int
    median: int
    max: int
    low: int
    """Available variants priced in the lowest third of the range."""
    mid: int
    high: int
    low_below: int
    """The price below which a variant is in the low band; `mid_below` likewise for the mid band."""
    mid_below: int


class StoreBrief(_Frozen):
    schema_: Literal["mc.brief.v1"] = Field(default=BRIEF_SCHEMA, alias="schema")
    id: str = ""
    """brief-<sha12> of everything else, so a brief names its content."""
    store_id: str
    domain: str
    category: str
    source: str
    """What the brief was read from: a capture id (cap-<sha12>), or fixtures-<sha12> of a fixture file."""
    currency: str | None
    products: int
    variants: int
    available_variants: int
    product_types: dict[str, int] = Field(default_factory=dict)
    """Shopper nouns to how many products are that kind of thing."""
    options: dict[str, list[str]] = Field(default_factory=dict)
    """Option dimensions (Size, Flavor, Pack) to the values the catalog offers, in catalog order."""
    price_bands: PriceBands | None = None
    exposes: list[ExposedKind] = Field(default_factory=list)
    """The data kinds the catalog exposes, most common first."""
    lacks: list[str] = Field(default_factory=list)
    """The data kinds no product (or too few) exposes, so no intent may turn on them."""
    policies: list[str] = Field(default_factory=list)
    """Policy text the capture holds (UCP `policies` of catalog replies), when any."""
    representatives: list[BriefProduct] = Field(default_factory=list)

    @property
    def exposed_kinds(self) -> list[str]:
        return [e.kind for e in self.exposes]

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> StoreBrief:
        if data.get("schema") != BRIEF_SCHEMA:
            raise ValidationError(f"a brief declares schema {data.get('schema')!r}; this code reads {BRIEF_SCHEMA}")
        try:
            brief = cls.model_validate(dict(data))
        except PydanticValidationError as exc:
            first = exc.errors(include_url=False, include_input=False)[0]
            where = ".".join(str(p) for p in first["loc"]) or "brief"
            raise ValidationError(f"brief is not valid {BRIEF_SCHEMA}: {where}: {first['msg']}") from None
        expected = brief_id(brief)
        if brief.id != expected:
            raise ValidationError(f"brief id {brief.id} does not match its content ({expected})", hint="rebuild it")
        return brief


# -- the data kinds ---------------------------------------------------------------------------------------------------


def _quote(text: str, start: int = 0) -> str:
    cut = text[max(0, start - 40) :].strip()
    return cut if len(cut) <= QUOTE_CHARS else cut[: QUOTE_CHARS - 1].rstrip() + "…"


def _grep(lines: Iterable[str], pattern: re.Pattern[str]) -> str | None:
    for line in lines:
        found = pattern.search(line)
        if found:
            return _quote(line, found.start())
    return None


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _text_lines(product: Product) -> list[str]:
    return [*product.lines, *product.tags]


_PATTERNS: Final[Mapping[str, re.Pattern[str]]] = {
    "ingredients": _rx(r"\bingredients?\b\s*:|\bmade (with|from)\b|\bcontains?\b\s*:"),
    "nutrition": _rx(
        r"nutrition(al)? (facts|information)|\bcalories\b|\bper serving\b|per 100 ?(g|ml)|\bprotein\b.{0,12}\d"
        r"|\d\s?g (of )?(sugar|protein|fib(er|re)|carb)"
    ),
    "dietary": _rx(
        r"\bvegan\b|gluten.?free|\bketo\b|\bpaleo\b|\bkosher\b|\bhalal\b|\borganic\b|non.?gmo|dairy.?free"
        r"|sugar.?free|\bwhole30\b|plant.?based|nut.?free|soy.?free"
    ),
    "materials": _rx(
        r"\bmaterials?\b\s*:|100% (cotton|wool|linen|silk|polyester|leather)"
        r"|\b(stainless steel|solid oak|merino|cashmere|organic cotton|recycled)\b"
    ),
    "dimensions": _rx(
        r"\bdimensions?\b|\d+(\.\d+)?\s?(cm|mm|in|inch|inches|\")\s?[x×]\s?\d|\b(width|height|depth|length)\b\s*:?\s*\d"
    ),
    "weight": _rx(r"\b\d+(\.\d+)?\s?(oz|g|kg|lb|lbs|grams?|ounces?|pounds?)\b"),
    "care": _rx(r"machine wash|hand wash|dry clean|care instructions|do not bleach|tumble dry|wipe clean"),
    "compatibility": _rx(
        r"compatible with|\bfits\b (the |all |most )?[A-Za-z0-9]|works with|for use with|\bmodel (number|no\.?)\b"
        r"|designed for"
    ),
    "dosage": _rx(r"\bdirections?\b\s*:|\bdosage\b|take \d|\bservings? per\b|\bserving size\b|\b\d+\s?mg\b"),
    "reviews": _rx(r"\b\d(\.\d)?\s?(out of 5|/5|stars?)\b|\b\d+\s?(customer )?reviews?\b|\brated\b"),
    "returns": _rx(r"\breturns?\b|\brefunds?\b|\bexchanges?\b"),
    "shipping": _rx(r"\bship(s|ped|ping)?\b|\bdeliver(y|ed|s)?\b"),
    "delivery_estimate": _rx(
        r"\d+\s?(-|to|–)?\s?\d*\s?(business |working )?days?\b|ships? (within|in|out)|same.?day|next.?day|\d-day\b"
        r"|overnight|\bprocess(ing)? time\b|\bcut.?off\b"
    ),
    "shipping_restrictions": _rx(
        r"(do|does|can)( not|'t|not) ship|cannot ship|\bPO boxes?\b|international (addresses|orders|shipping)"
        r"|perishable|cool packs?|ice packs?|\bheat\b.{0,30}(ship|transit)|only ship"
    ),
    "subscription": _rx(
        r"\bsubscri(be|ption)\b|auto.?ship|\brecurring\b|subscribe (and|&) save|every \d+ (weeks?|months?)"
    ),
    "bundle": _rx(
        r"\bbundles?\b|variety pack|gift set|\bkits?\b|\bsampler\b|\bassort(ed|ment)\b|\bbox of\b|multi.?pack"
    ),
    "gift": _rx(r"\bgifts?\b|gift wrap|gift message|gift card"),
    "certifications": _rx(
        r"certif(ied|ication)|\bUSDA\b|non.?gmo|fair.?trade|cruelty.?free|\bFDA\b|\bB.?Corp\b|\bGOTS\b|\bOEKO\b"
        r"|\bFSC\b|\bNSF\b|\bUL listed\b"
    ),
    "origin": _rx(
        r"\bmade in\b|\bproduct of\b|\bsourced (from|in)\b|grass.?fed|wild.?caught|\bimported\b|\bhandmade\b"
        r"|\bsmall.?batch\b"
    ),
    "warranty": _rx(r"\bwarrant(y|ies)\b|\bguarantee\b|lifetime"),
    "stock_text": _rx(
        r"sold out|back in stock|pre.?order|backorder|low stock|only \d+ left|limited (edition|run|stock)"
        r"|ships in \d+ weeks?"
    ),
    # The kinds of the drafted niches (docs/research/niche-checklists-2026-10-13.md, D54). A phrase kind's pattern
    # is also where its needs take their phrases from (carttrace.agents.attributes.phrases_in), so each alternative is a
    # phrase a shopper would say whole: "14k gold", "lab grown diamond", "dry hair", "cordless".
    "metal": _rx(
        r"\b(9|10|14|18|22|24) ?(k|kt|ct|karat) (solid |recycled )?(yellow |white |rose )?gold( vermeil| plated"
        r"| filled)?\b|\bsolid (yellow |white |rose )?gold\b|\bsterling silver\b|\b925 silver\b"
        r"|\bgold (vermeil|plated|filled)\b|\bgold-(plated|filled)\b|\b(950|solid) platinum\b"
    ),
    "stone": _rx(
        r"\b(lab[ -]grown|lab[ -]created|natural) (diamond|sapphire|emerald|ruby|stone|gemstone)\b"
        r"|\bdiamond\b(?![ -]?(quilt|pattern|stitch|print|weave|plate|knit|shape|check))|\bsapphire\b|\bmoissanite\b"
        r"|\bcubic zirconia\b|\bfreshwater pearl\b|\bbirthstone\b|\bgemstone\b|\bcarat weight\b|\bctw\b"
    ),
    "personalization": _rx(
        r"\b(personali[sz]ed|personali[sz]able|engraved|engravable|monogrammed|embossed|hand-?stamped)\b"
    ),
    "hair_type": _rx(
        r"\b(curly|coily|wavy|straight|fine|thin|thick|dry|damaged|oily|frizzy|colou?r[ -]treated|bleached"
        r"|textured|all) hair( types?)?\b|\bhair types?\b"
    ),
    "scent": _rx(
        r"\b(top|heart|middle|base) notes?\b|\bnotes? of\b|\bolfactory\b|\b(fragrance|scent) (family|notes|profile)\b"
        r"|\beau de (parfum|toilette|cologne)\b|\bextrait de parfum\b"
    ),
    "power": _rx(
        r"\b(cordless|corded|rechargeable|battery[ -]powered|dual[ -]voltage|brushless|solar[ -]powered)\b"
        r"|\b\d{1,3}(\.\d)? ?(v|volts?)\b|\b\d{2,4} ?watts?\b"
    ),
    "condition": _rx(r"\b(refurbished|pre-?owned|open[ -]box|like new|b-stock|second[ -]hand)\b"),
    "pet_type": _rx(
        r"\b(dogs|cats|puppies|kittens|small animals|small dogs|large dogs|senior dogs|senior cats|rabbits|birds"
        r"|guinea pigs|hamsters|ferrets)\b"
    ),
    "age_range": _rx(
        r"\bages? \d+|\b\d+ ?(\+|plus) ?(months?|mos?|years?|yrs?)\b|\b\d+ ?(-|to) ?\d+ ?(months?|mos?|years?|yrs?)\b"
        r"|\b\d+ ?(months?|years?) and up\b|\b(newborn|infant|toddler|preemie)s?\b"
    ),
    "capacity": _rx(r"\b\d+(\.\d+)? ?(l|litres?|liters?|ltr|qt|quarts?)\b|\bcapacity\b|\bholds (up to )?\d+"),
    "identifiers": _rx(
        r"\bISBN(-1[03])?\b|\bMPN\b|\b(part|model|item) (number|no\.?|#)|\bOEM (part )?(number|no)\b|\bUPC\b"
        r"|\bEAN\b|\bGTIN\b"
    ),
    "safety_standard": _rx(
        r"\bASTM ?F ?\d+|\bEN ?71\b|\bCPSIA\b|\bCPSC\b|\bJPMA\b|\bUKCA\b|\bFMVSS ?\d+|\bECE ?R ?\d+|\bR129\b"
        r"|\bFCC\b|\bUL (listed|certified)\b|\bCE (marked|certified)\b|\bBPA[ -]free\b|\bphthalate[ -]free\b"
        r"|\blead[ -]free\b"
    ),
}


def _by_pattern(kind: str) -> Detector:
    pattern = _PATTERNS[kind]

    def detect(product: Product, raw: Mapping[str, Any]) -> str | None:
        return _grep(_text_lines(product), pattern)

    return detect


def _options(product: Product, raw: Mapping[str, Any]) -> str | None:
    names = [name for v in product.variants for name, value in v.options if fold(value) not in ("default title", "")]
    seen = list(dict.fromkeys(names))
    return ", ".join(seen) if seen else None


def _price(product: Product, raw: Mapping[str, Any]) -> str | None:
    priced = [v for v in product.variants if v.price is not None]
    return f"{len(priced)} priced variants" if priced else None


def _list_price(product: Product, raw: Mapping[str, Any]) -> str | None:
    listed = raw.get("list_price_range")
    low = listed.get("min") if isinstance(listed, Mapping) else None
    amount = low.get("amount") if isinstance(low, Mapping) else None
    if isinstance(amount, int) and amount > 0:
        return f"list price {amount} minor units"
    for variant in raw.get("variants") or []:
        if not isinstance(variant, Mapping):
            continue
        price, list_price = variant.get("price"), variant.get("list_price")
        if not isinstance(price, Mapping) or not isinstance(list_price, Mapping):
            continue
        listed_amount = list_price.get("amount")
        if listed_amount not in (None, 0) and listed_amount != price.get("amount"):
            return f"list price {listed_amount} against price {price.get('amount')}"
    return None


def _unit_price(product: Product, raw: Mapping[str, Any]) -> str | None:
    for variant in product.variants:
        held = quantity_of(product, variant)
        if held is not None and held.amount > 0:
            amount = held.amount
            shown = str(amount.numerator) if amount.denominator == 1 else f"{float(amount):g}"
            return f"{variant.label() or variant.title}: {shown} {held.unit}"
    return None


def _availability(product: Product, raw: Mapping[str, Any]) -> str | None:
    out = [v for v in product.variants if not v.available]
    if out:
        return f"{len(out)} of {len(product.variants)} variants unavailable"
    return _grep(_text_lines(product), _PATTERNS["stock_text"])


def _allergens(product: Product, raw: Mapping[str, Any]) -> str | None:
    found = statements(product)
    if found:
        return _quote(found[0])
    known = [a for a in ALLERGENS if allergen_status(product, a) != "unknown"]
    return f"free-from claims: {', '.join(known[:4])}" if known else None


def _size_chart(product: Product, raw: Mapping[str, Any]) -> str | None:
    chart = read_chart(product)
    if chart is None:
        return None
    return f"size chart with {', '.join(chart.measures())}"


def _warnings(product: Product, raw: Mapping[str, Any]) -> str | None:
    lines = warning_lines(product)
    return _quote(lines[0]) if lines else None


def _description(product: Product, raw: Mapping[str, Any]) -> str | None:
    text = " ".join(product.lines)
    return _quote(text) if len(text) >= 80 else None


def _brand(product: Product, raw: Mapping[str, Any]) -> str | None:
    return product.brand


def _identifiers(product: Product, raw: Mapping[str, Any]) -> str | None:
    """A GTIN on a variant (UCP `barcodes`, as the capture pass writes a barcode), else an ISBN, MPN or part number
    in the text: what an agent matches a named model, book or part against."""
    variants = [v for v in raw.get("variants") or [] if isinstance(v, Mapping)]
    coded = [v for v in variants if any(isinstance(b, Mapping) and b.get("value") for b in v.get("barcodes") or [])]
    if coded:
        return f"gtin on {len(coded)} of {len(variants)} variants"
    return _grep(_text_lines(product), _PATTERNS["identifiers"])


def _media_alt(product: Product, raw: Mapping[str, Any]) -> str | None:
    for item in raw.get("media") or []:
        alt = item.get("alt_text") if isinstance(item, Mapping) else None
        if isinstance(alt, str) and alt.strip():
            return _quote(alt)
    return None


DATA_KINDS: Final[tuple[KindSpec, ...]] = (
    KindSpec(kind="options", row="variants.options", about="which size, flavour, pack or colour to pick"),
    KindSpec(kind="price", row="price", about="whether it fits a budget, or is the cheapest"),
    KindSpec(kind="price.list", row="price.list", about="whether it is on sale against a list price"),
    KindSpec(kind="price.unit", row="price.unit", about="the price per unit, per pack size or per 100 g"),
    KindSpec(kind="availability", row="variants.availability", about="in stock, on preorder, or sold out"),
    KindSpec(kind="description", row="content.description", about="what the product says about itself"),
    KindSpec(kind="brand", row="identity.brand", about="who makes it"),
    KindSpec(kind="ingredients", row="content.ingredients", about="what it is made of"),
    KindSpec(kind="allergens", row="content.allergens", about="whether it is safe for an allergy"),
    KindSpec(kind="nutrition", row="content.nutrition", about="protein, sugar, calories per serving"),
    KindSpec(kind="dietary", row="content.badges", about="vegan, gluten-free, keto, kosher claims"),
    KindSpec(kind="certifications", row="content.badges", about="certified organic, non-GMO, fair trade"),
    KindSpec(kind="origin", row="content.description", about="where and how it is made or sourced"),
    KindSpec(kind="size_chart", row="size.chart", about="which size fits a measurement"),
    KindSpec(kind="materials", row="content.specs", about="what it is made from"),
    KindSpec(kind="dimensions", row="content.specs", about="whether it fits a space"),
    KindSpec(kind="weight", row="content.specs", about="how much it weighs or holds"),
    KindSpec(kind="care", row="content.specs", about="how to wash or look after it"),
    KindSpec(kind="compatibility", row="content.compatibility", about="whether it fits or works with another thing"),
    KindSpec(kind="warnings", row="content.warnings", about="whether it is safe for a condition or an age"),
    KindSpec(kind="dosage", row="content.dosage", about="how much to take and how many servings a pack holds"),
    KindSpec(kind="reviews", row="reviews.rating", about="how others rated it"),
    KindSpec(kind="returns", row="returns.policy", about="whether and how it can be returned"),
    KindSpec(kind="shipping", row="shipping.cost", about="what delivery costs and where it ships"),
    KindSpec(kind="delivery_estimate", row="delivery.estimate", about="when it arrives, processing time, cut-offs"),
    KindSpec(kind="shipping_restrictions", row="shipping.restrictions", about="where it cannot ship, perishables"),
    KindSpec(kind="warranty", row="policy.warranty", about="what is guaranteed and for how long"),
    KindSpec(kind="subscription", row="catalog.subscription", about="repeat delivery and its terms"),
    KindSpec(kind="bundle", row="catalog.bundle", about="sets, kits and variety packs"),
    KindSpec(kind="gift", row="catalog.bundle", about="gift sets, wrap and messages"),
    KindSpec(kind="media_alt", row="images.alt", about="what the pictures show, as text"),
    # the drafted niches' kinds (D54)
    KindSpec(kind="metal", row="content.specs", about="which metal and purity, solid or plated"),
    KindSpec(kind="stone", row="content.specs", about="which stone, natural or lab grown, and its carat"),
    KindSpec(kind="personalization", row="catalog.bundle", about="whether it can be engraved or monogrammed"),
    KindSpec(kind="hair_type", row="content.category_facts", about="which hair type it is for"),
    KindSpec(kind="scent", row="content.category_facts", about="its notes, family and concentration"),
    KindSpec(kind="power", row="content.specs", about="cordless or corded, voltage, battery, wattage"),
    KindSpec(kind="condition", row="content.category_facts", about="new, refurbished or pre-owned"),
    KindSpec(kind="pet_type", row="content.category_facts", about="which animal, life stage or size it is for"),
    KindSpec(kind="age_range", row="content.category_facts", about="which age it suits"),
    KindSpec(kind="capacity", row="content.specs", about="how much it holds, in litres or quarts"),
    KindSpec(kind="identifiers", row="identifiers.gtin", about="the GTIN, ISBN, model or part number"),
    KindSpec(kind="safety_standard", row="content.badges", about="the safety standard it meets"),
)  # fmt: skip
"""Every data kind an intent's `decisive_kind` may name, in display order."""

KIND_IDS: Final[frozenset[str]] = frozenset(k.kind for k in DATA_KINDS)
ALWAYS_KINDS: Final[frozenset[str]] = frozenset({"options", "price"})
"""Kinds every priced catalog exposes, so they never count towards the minimum of kinds a store must spread over."""

_DETECTORS: Final[Mapping[str, Detector]] = {
    "options": _options,
    "price": _price,
    "price.list": _list_price,
    "price.unit": _unit_price,
    "availability": _availability,
    "description": _description,
    "brand": _brand,
    "allergens": _allergens,
    "size_chart": _size_chart,
    "warnings": _warnings,
    "media_alt": _media_alt,
    **{kind: _by_pattern(kind) for kind in _PATTERNS if kind != "stock_text"},
    "identifiers": _identifiers,
}


def carried_by(kind: str, product: Product, raw: Mapping[str, Any] | None = None) -> bool:
    """Whether one product carries one data kind, by the kind's own detector."""
    detector = _DETECTORS.get(kind)
    return detector is not None and bool(detector(product, raw or {}))


def kind_pattern(kind: str) -> re.Pattern[str] | None:
    """The pattern a text kind's detector reads (dietary, origin, subscription, ...), or None for a kind read from
    structured data; the attribute rules (carttrace.agents.attributes) take their phrases from the same pattern, so a need
    never asks for a phrase the brief could not have found."""
    return _PATTERNS.get(kind)


def kind_spec(kind: str) -> KindSpec:
    found = next((k for k in DATA_KINDS if k.kind == kind), None)
    if found is None:
        raise ValidationError(f"unknown data kind {kind!r}", hint=f"choose from {', '.join(sorted(KIND_IDS))}")
    return found


def facts_of(product: Product, raw: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Every data kind the product exposes, with a quote, in DATA_KINDS order."""
    document = raw or {}
    out: dict[str, str] = {}
    for spec in DATA_KINDS:
        found = _DETECTORS[spec.kind](product, document)
        if found:
            out[spec.kind] = found
    return out


# -- building a brief -------------------------------------------------------------------------------------------------


def brief_of(
    fixtures: Sequence[Mapping[str, Any]],
    *,
    store_id: str,
    domain: str,
    category: str,
    source: str,
    currency: str | None = None,
    policies: Sequence[str] = (),
) -> StoreBrief:
    """The brief of a catalog given as product dicts (UCP's shape, as a capture's documents or a fixture file).
    Deterministic: products are read in id order and every count and quote comes from the data."""
    fixtures = [raw for raw in fixtures if raw.get("page_kind") != "listing"]  # a grid is not a product (D38)
    products = read_products(fixtures, currency=currency)
    if not products:
        raise ValidationError(f"no products to brief for {store_id}", hint="capture the store first")
    raws = {str(_unwrapped_id(raw)): raw for raw in fixtures}
    facts = {p.id: facts_of(p, raws.get(p.id)) for p in products}
    priced = [v for p in products for v in p.variants if v.price is not None]
    available = [v for v in priced if v.available]
    money = currency or next((v.currency for v in priced if v.currency), None)
    kinds: dict[str, int] = {}
    for product in products:
        kinds[noun_of(product).singular] = kinds.get(noun_of(product).singular, 0) + 1
    exposes, lacks = _exposure(products, facts)
    draft = StoreBrief(
        store_id=store_id,
        domain=domain,
        category=category,
        source=source,
        currency=money,
        products=len(products),
        variants=sum(len(p.variants) for p in products),
        available_variants=len(available),
        product_types=dict(sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))),
        options=_option_dimensions(products),
        price_bands=_bands(available or priced),
        exposes=exposes,
        lacks=lacks,
        policies=[_quote(p) for p in policies][:10],
        representatives=_representatives(products, facts),
    )
    return draft.model_copy(update={"id": brief_id(draft)})


def brief_of_capture(capture: StoreCapture) -> StoreBrief:
    """The brief of a capture: its products' documents, its currency, and any policy text its replies carried."""
    documents = [p.document for p in capture.products]
    policies: list[str] = []
    for call in capture.calls:
        reply = call.reply if isinstance(call.reply, Mapping) else {}
        for policy in reply.get("policies") or []:
            text = policy.get("text") or policy.get("content") if isinstance(policy, Mapping) else policy
            if isinstance(text, str) and text.strip():
                policies.append(text.strip())
    return brief_of(
        documents,
        store_id=capture.store.store_id,
        domain=capture.store.domain,
        category=capture.category,
        source=capture.id,
        currency=capture.currency,
        policies=policies,
    )


def brief_id(brief: StoreBrief) -> str:
    body = brief.model_dump(mode="json", by_alias=True, exclude={"id"})
    return f"brief-{digest(body, 12)}"


def brief_path(var: Path, store_id: str, source: str) -> Path:
    return var / BRIEFS_DIR / store_id / f"{source}.json"


def cached_brief(capture: StoreCapture, var: Path) -> StoreBrief:
    """The capture's brief, read from var/agent/briefs/<store>/<capture id>.json when it is there, else built and
    saved. A cached brief that no longer validates is rebuilt."""
    path = brief_path(var, capture.store.store_id, capture.id)
    if path.is_file():
        try:
            return StoreBrief.from_mapping(loads(path.read_text(encoding="utf-8")))
        except (ValidationError, ValueError):
            pass
    brief = brief_of_capture(capture)
    atomic_write_text(path, dumps(brief.to_json()) + "\n")
    return brief


def load_brief(path: Path) -> StoreBrief:
    try:
        return StoreBrief.from_mapping(loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise ValidationError(f"no brief at {path}", hint="write one with `carttrace agents brief STORE`") from None
    except ValueError as exc:
        raise ValidationError(f"{path} is not JSON: {exc}") from exc


def fixtures_source(fixtures: Sequence[Mapping[str, Any]]) -> str:
    """The source name of a brief built from a fixture file: fixtures-<sha12> of its canonical content."""
    return f"fixtures-{digest(list(fixtures), 12)}"


# -- the parts --------------------------------------------------------------------------------------------------------


def _unwrapped_id(raw: Mapping[str, Any]) -> Any:
    for key in ("id",):
        if key in raw:
            return raw[key]
    for key in ("product", "result", "structuredContent"):
        inner = raw.get(key)
        if isinstance(inner, Mapping) and "id" in inner:
            return inner["id"]
    return None


def _exposure(
    products: Sequence[Product], facts: Mapping[str, Mapping[str, str]]
) -> tuple[list[ExposedKind], list[str]]:
    exposes: list[ExposedKind] = []
    lacks: list[str] = []
    total = len(products)
    for spec in DATA_KINDS:
        carrying = [p.id for p in products if spec.kind in facts[p.id]]
        count = len(carrying)
        share = round(count / total, 3) if total else 0.0
        if count and (count >= EXPOSED_PRODUCTS or share >= EXPOSED_SHARE):
            exposes.append(ExposedKind(kind=spec.kind, products=count, share=share, examples=carrying[:3]))
        else:
            lacks.append(spec.kind)
    exposes.sort(key=lambda e: (-e.products, e.kind))
    return exposes, lacks


def _option_dimensions(products: Sequence[Product]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for product in products:
        for variant in product.variants:
            for name, value in variant.options:
                if fold(value) in ("default title", ""):
                    continue
                values = out.setdefault(name, [])
                if value not in values and len(values) < MAX_OPTION_VALUES:
                    values.append(value)
    return dict(sorted(out.items()))


def _bands(variants: Sequence[Variant]) -> PriceBands | None:
    prices = sorted(v.price for v in variants if v.price is not None)
    if not prices:
        return None
    low, high = prices[0], prices[-1]
    span = high - low
    low_below = low + span // 3
    mid_below = low + (2 * span) // 3
    return PriceBands(
        min=low,
        median=int(statistics.median(prices)),
        max=high,
        low=sum(1 for p in prices if p <= low_below),
        mid=sum(1 for p in prices if low_below < p <= mid_below),
        high=sum(1 for p in prices if p > mid_below),
        low_below=low_below,
        mid_below=mid_below,
    )


def _representatives(products: Sequence[Product], facts: Mapping[str, Mapping[str, str]]) -> list[BriefProduct]:
    """Twenty to forty products (every one when the catalog is small), spread over product kinds, the richest in
    facts first within a kind, in a deterministic order."""
    by_kind: dict[str, list[Product]] = {}
    for product in products:
        by_kind.setdefault(noun_of(product).singular, []).append(product)
    for members in by_kind.values():
        members.sort(key=lambda p: (-len(facts[p.id]), p.id))
    kinds = sorted(by_kind, key=lambda k: (-len(by_kind[k]), k))
    wanted = min(MAX_REPRESENTATIVES, len(products))
    chosen: list[Product] = []
    while len(chosen) < wanted:
        progressed = False
        for kind in kinds:
            members = by_kind[kind]
            if members:
                chosen.append(members.pop(0))
                progressed = True
                if len(chosen) >= wanted:
                    break
        if not progressed:
            break
    chosen.sort(key=lambda p: p.id)
    return [_brief_product(p, facts[p.id]) for p in chosen]


def _brief_product(product: Product, facts: Mapping[str, str]) -> BriefProduct:
    priced = [v.price for v in product.variants if v.price is not None]
    return BriefProduct(
        id=product.id,
        title=product.title,
        kind=noun_of(product).singular,
        brand=product.brand,
        product_type=product.product_type,
        tags=list(product.tags[:MAX_TAGS_SHOWN]),
        price_min=min(priced) if priced else None,
        price_max=max(priced) if priced else None,
        variants=[
            BriefVariant(id=v.id, label=v.label(), price=v.price, available=v.available)
            for v in product.variants[:MAX_VARIANTS_SHOWN]
        ],
        facts={k: v for k, v in facts.items() if k not in ("description", "price", "brand")},
    )


__all__ = [
    "ALWAYS_KINDS",
    "BRIEFS_DIR",
    "BRIEF_SCHEMA",
    "DATA_KINDS",
    "KIND_IDS",
    "BriefProduct",
    "BriefVariant",
    "ExposedKind",
    "KindSpec",
    "PriceBands",
    "StoreBrief",
    "brief_id",
    "brief_of",
    "brief_of_capture",
    "brief_path",
    "cached_brief",
    "carried_by",
    "facts_of",
    "fixtures_source",
    "kind_pattern",
    "kind_spec",
    "load_brief",
]

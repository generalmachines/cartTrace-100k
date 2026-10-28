"""The union core of an open stand-in (D117): each category's common data types, filled on every product in layers.

Why. A stand-in carried only what its open catalog had, about 3 to 22% of the data types its category's real stores
show (var/bench/union, D62), so no task could turn on a material, a care label, a fit or a warranty, and the judge's
category questions rarely applied. The core is what most real stores of the category show; a stand-in now serves it.

The layers, recorded per value in the world (each product's `provenance.union_core`) and in the
exposure sheet (`layer` beside `origin`):
- **base**: the catalog's own product facts, as the pool sets them (title, description, images, brand, its own
  attributes and details, price, variants). Never changed: a core value is only ever added where the base states
  none, and never contradicts the base's text.
- **core**: the category's curated union types (data/bench/union.yaml ids, not the auto `label.*` or `heading.*`
  tail) that at least `sandbox.v1.union_core_share` of its real stores show (data/bench/union_core.json, frozen from
  the union so a world rebuilds the same), filled on every product in order: (a) `catalog`, a base field that states
  it (a detail, or an attribute whose key names it, D104's text cleaning applied); (b) `derived`, a deterministic rule
  over the product's own title and description ("40 x 30 cm", "machine washable", "100% cotton"); (c) `synthetic`,
  only where the category's stores usually show it, seeded by the product id, from a vocabulary checked never to
  contradict the text. Safety facts (ingredients, allergens, nutrition, warnings, directions) are never synthetic, nor
  are sizes and weights (a size drawn without the product's own kind read absurd: 57 x 43 x 21 cm work gloves).
  Store-level types (returns, shipping cost, threshold and estimate, warranty) come from the store's own market and
  spec (D107), one value for every product, served as the UCP catalog's root policies. Served on every surface as
  the platform pack serves product data: UCP `metadata.details` (Shopify's recorded view folds them into the
  description, D100), the storefront's product page, the website's page.
- **experimental**: the types a later experiment may withhold or show per row (data/sandbox/union_layers.yaml, per
  category). Each maps to the detail names the detail_hidden cut (`carttrace.collect.surfaces.view_filter`) removes from a
  surface's replies, so an experiment switches a type per row with no new world. In v1 nothing is withheld
  (`mask_bp` 0): the layer only says which types may be.

Never filled, with the reason the measure reports: reviews (a review is a shopper's, never ours), media and alt text
(no licensed image), compare-at prices (a sale that never was), GTINs (a real namespace), variant options (they are
the base's structure), and store files.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import yaml

from carttrace.core.canon import digest, dumps, loads, sha256_hex
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.ids import seeded_rng

RULE: Final = "union-core.v1"
REPO: Final = Path(__file__).resolve().parents[3]
CORE_FILE: Final = REPO / "data" / "bench" / "union_core.json"
LAYERS_FILE: Final = REPO / "data" / "sandbox" / "union_layers.yaml"
CORE_SCHEMA: Final = "mc.union-core.v1"
LAYERS_SCHEMA: Final = "mc.union-layers.v1"
Origin = Literal["catalog", "derived", "synthetic", "store"]
Layer = Literal["base", "core", "experimental"]
PROVENANCE: Final = "union_core"
"""The key under each product's `provenance` (never served, D10) and the pack's provenance that records the fill."""

# -- where each core type lives ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Slot:
    """Where a core type lives in a store's data, and how it is filled."""

    where: Literal["detail", "policy", "base", "unfilled"]
    key: str = ""
    """The `metadata.details` key (a detail) or the policy fact (a policy)."""
    synthetic: bool = False
    """Whether a synthetic value may stand where the catalog and the text say nothing."""
    reason: str = ""
    """Why an unfilled type is left as the base has it."""


SLOTS: Final[Mapping[str, Slot]] = {
    # product facts, in metadata.details
    "product.materials": Slot("detail", "material", synthetic=True),
    "product.care": Slot("detail", "care", synthetic=True),
    "product.measurements": Slot("detail", "dimensions"),
    "product.weight": Slot("detail", "weight"),
    "product.capacity": Slot("detail", "capacity"),
    "product.compatibility": Slot("detail", "compatibility"),
    "product.whats_included": Slot("detail", "included", synthetic=True),
    "product.features": Slot("detail", "features"),
    "product.origin": Slot("detail", "origin"),
    "product.scent_notes": Slot("detail", "scent"),
    "product.ingredients": Slot("detail", "ingredients"),
    "product.allergens": Slot("detail", "allergens"),
    "product.nutrition": Slot("detail", "nutrition"),
    "product.warnings": Slot("detail", "warnings"),
    "product.directions": Slot("detail", "directions"),
    "badge.claims": Slot("detail", "certifications"),
    "policy.warranty": Slot("detail", "warranty", synthetic=True),
    # store terms, the UCP catalog's root policies, one value a store
    "returns.window": Slot("policy", "returns_window_days"),
    "returns.policy": Slot("policy", "returns_window_days"),
    "returns.fee": Slot("policy", "returns_fee_minor"),
    "delivery.free_threshold": Slot("policy", "free_shipping_threshold_minor"),
    "delivery.estimate": Slot("policy", "transit_days_max"),
    "delivery.cost": Slot("policy", "shipping_cost_minor"),
    "policy.shipping": Slot("policy", "shipping_cost_minor"),
    # the base serves these (or the store's surfaces do); a core type the base lacks is not invented
    **{
        t: Slot("base", reason="a base field: served as the pool sets it")
        for t in (
            "product.title",
            "product.description",
            "price.current",
            "variant.availability",
            "variant.options",
            "variant.size",
            "variant.colour",
            "identity.brand",
            "identity.product_type",
            "identity.tags",
            "nav.collections",
            "identity.sku",
            "price.taxable",
            "delivery.requires_shipping",
            "delivery.regions",
            "delivery.pickup",
            "delivery.restrictions",
            "stock.count",
            "stock.preorder",
            "stock.tracking",
            "stock.pickup",
            "price.cart_total",
            "product.specs",
        )
    },
    **{
        t: Slot("unfilled", reason="a shopper's review is never written by us")
        for t in ("reviews.rating", "reviews.count", "reviews.text", "reviews.qa")
    },
    **{
        t: Slot("unfilled", reason="no licensed image to show or describe")
        for t in ("media.images", "media.alt_text", "media.video", "media.model_3d")
    },
    "price.compare_at": Slot("unfilled", reason="a compare-at price would claim a sale that never was (D100)"),
    "price.promo_banner": Slot("unfilled", reason="a promotion the store never ran"),
    "identity.gtin": Slot("unfilled", reason="a GTIN is a real namespace; a made-up one may name another product"),
    "product.size_chart": Slot("unfilled", reason="a chart must match the brand's own sizing; only the base's"),
}
"""Each curated type a core can hold, to where it lives. A type not listed is a store page or file (store.*,
policy.privacy, nav.*): served by the store's surfaces as before, never per product."""

SAFETY: Final = frozenset({"ingredients", "allergens", "nutrition", "warnings", "directions"})
"""Details never synthetic: a made-up ingredient or warning could mislead a reader of the data about real safety."""


# -- the frozen core ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoreFile:
    id: str
    """`uc-<sha12>` of the file's bytes: what a world's recipe names."""
    categories: Mapping[str, Mapping[str, tuple[int, int]]]
    """Category to curated type id to (stores showing it, stores read)."""
    source: Mapping[str, Any]

    def core(self, category: str, share: float) -> tuple[str, ...]:
        """The category's core at `share`: curated types at least that share of its read stores show, by id."""
        found = self.categories.get(category) or {}
        return tuple(sorted(t for t, (n, of) in found.items() if of > 0 and n / of >= share))

    def share(self, category: str, type_id: str) -> float:
        n, of = (self.categories.get(category) or {}).get(type_id, (0, 0))
        return n / of if of else 0.0


def build_core_file(union_json: Path, curated: Sequence[str]) -> dict[str, Any]:
    """The core file's body from the union's union.json (`carttrace bench union`): per category, every curated type's
    stores and stores read. The threshold is applied when a world is built (`sandbox.v1.union_core_share`)."""
    raw = loads(union_json.read_bytes())
    ids = set(curated)
    categories: dict[str, dict[str, list[int]]] = {}
    for slug, body in sorted((raw.get("categories") or {}).items()):
        categories[slug] = {
            str(t["id"]): [int(t.get("stores") or 0), int(t.get("of") or 0)]
            for t in sorted(body.get("types") or (), key=lambda t: str(t["id"]))
            if str(t["id"]) in ids
        }
    return {
        "schema": CORE_SCHEMA,
        "source": {
            "file": "var/bench/union/union.json",
            "sha256": sha256_hex(union_json.read_bytes()),
            "inputs": raw.get("inputs") or {},
            "types_file": raw.get("types_file"),
        },
        "categories": categories,
    }


def load_core(path: Path | None = None) -> CoreFile:
    return _load_core((path or CORE_FILE).resolve())


@functools.cache
def _load_core(path: Path) -> CoreFile:
    if not path.is_file():
        raise NotFoundError(f"no union core at {path}", hint="run `carttrace sandbox union-core --write`")
    body = path.read_bytes()
    raw = loads(body)
    if raw.get("schema") != CORE_SCHEMA:
        raise ValidationError(f"{path.name} is not {CORE_SCHEMA}")
    cats = {
        str(c): {str(t): (int(v[0]), int(v[1])) for t, v in types.items()} for c, types in raw["categories"].items()
    }
    return CoreFile(id="uc-" + sha256_hex(body)[:12], categories=cats, source=raw.get("source") or {})


@dataclass(frozen=True, slots=True)
class Layers:
    experimental: Mapping[str, tuple[str, ...]]

    def layer(self, category: str, type_id: str) -> Layer:
        return "experimental" if type_id in self.experimental.get(category, ()) else "core"

    def details(self, category: str) -> tuple[str, ...]:
        """The detail names the detail_hidden cut withholds to switch the category's experimental types off for a row
        (`carttrace.collect.surfaces.view_filter(details=...)`), with no new world."""
        out = [SLOTS[t].key for t in self.experimental.get(category, ()) if t in SLOTS and SLOTS[t].where == "detail"]
        return tuple(dict.fromkeys(out))


def load_layers(path: Path | None = None) -> Layers:
    return _load_layers((path or LAYERS_FILE).resolve())


@functools.cache
def _load_layers(path: Path) -> Layers:
    if not path.is_file():
        return Layers({})
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("schema") != LAYERS_SCHEMA:
        raise ValidationError(f"{path.name} is not {LAYERS_SCHEMA}")
    cats = raw.get("experimental") or {}
    return Layers({str(c): tuple(sorted(str(t) for t in types or ())) for c, types in cats.items()})


@dataclass(frozen=True, slots=True)
class CoreSpec:
    """What a world's recipe names (World.union_core): the rule, the frozen core and the threshold."""

    core: CoreFile
    share: float

    @property
    def ref(self) -> str:
        return f"{RULE}@{self.core.id}@{round(self.share * 10_000)}"


def spec_of(ref: str | None, core: CoreFile | None = None) -> CoreSpec | None:
    if ref is None:
        return None
    rule, core_id, bp = ref.split("@")
    found = core or load_core()
    if rule != RULE or core_id != found.id:
        raise ValidationError(
            f"world names union core {ref}, but this checkout holds {RULE}@{found.id}",
            hint="rebuild the manifest, or restore data/bench/union_core.json",
        )
    return CoreSpec(found, int(bp) / 10_000)


# -- reading the base --------------------------------------------------------------------------------------------------

_PLACEHOLDER: Final = re.compile(
    r"^\s*(|-+|n/?a|na|none|null|nil|\[\]|not specified|not applicable|no|unknown|other|others|attributes_\w+|0)\s*$",
    re.IGNORECASE,
)
_TAG: Final = re.compile(r"<[^>]+>")


def clean(value: Any) -> str | None:
    """A base value as text, tags and runs of space removed (D104's cleaning), or None when it states nothing."""
    if value is None or isinstance(value, bool | Mapping | list):
        return None
    text = " ".join(_TAG.sub(" ", str(value)).split())
    if not text or _PLACEHOLDER.match(text) or len(text) > 300:
        return None
    return text


ATTRIBUTE_KEYS: Final[Mapping[str, re.Pattern[str]]] = {
    "material": re.compile(r"material|fabric", re.IGNORECASE),
    "care": re.compile(r"\bcare\b|wash|clean", re.IGNORECASE),
    "capacity": re.compile(r"capacity|volume", re.IGNORECASE),
    "compatibility": re.compile(
        r"compatib|fits? |vehicle model|car model|motorcycle model|device model", re.IGNORECASE
    ),
    "origin": re.compile(r"country of origin|country origin|made in", re.IGNORECASE),
    "scent": re.compile(r"scent|fragrance", re.IGNORECASE),
    "warranty": re.compile(r"warranty (length|period)|product warranty", re.IGNORECASE),
    "certifications": re.compile(r"dietary needs|organic|certification", re.IGNORECASE),
}
"""Attribute keys (ABO, WANDS, Lazada) that state a detail. Weight and dimensions are composed below."""
_WANDS_DIMS: Final = (
    ("W", re.compile(r"overall width", re.IGNORECASE)),
    ("D", re.compile(r"overall depth", re.IGNORECASE)),
    ("H", re.compile(r"overall height", re.IGNORECASE)),
)
_NO_WARRANTY: Final = re.compile(r"^\s*(no|none|n/?a|no warranty)\s*$", re.IGNORECASE)


def _from_attributes(key: str, attributes: Mapping[str, Any]) -> tuple[str, str] | None:  # noqa: PLR0912 - one rule a detail
    """(value, attribute key) a catalog attribute states for detail `key`."""
    if key == "dimensions":
        parts = []
        for axis, pattern in _WANDS_DIMS:
            hit = next((clean(v) for k, v in sorted(attributes.items()) if pattern.search(str(k))), None)
            if hit is not None and re.fullmatch(r"\d+(\.\d+)?", hit):
                parts.append(f"{axis} {hit}")
        return (" x ".join(parts) + " in", "Overall width/depth/height") if len(parts) == 3 else None
    if key == "weight":
        for k, v in sorted(attributes.items()):
            text = clean(v)
            if (
                text is not None
                and re.search(r"product weight, lb", str(k), re.IGNORECASE)
                and re.fullmatch(r"\d+(\.\d+)?", text)
            ):
                return f"{text} lb", str(k)
        return None
    named = ATTRIBUTE_KEYS.get(key)
    if named is None:
        return None
    for k, v in sorted(attributes.items()):
        if not named.search(str(k)) or re.search(r"[<>\"=]", str(k)):
            continue
        text = clean(v)
        if text is None:
            continue
        if key == "warranty":
            if _NO_WARRANTY.match(str(v)):
                continue
            length = next(
                (clean(x) for kk, x in attributes.items() if re.search("warranty length", str(kk), re.I)), None
            )
            text = f"{length} warranty" if length and key == "warranty" and re.search(r"\d", length) else text
            if text.lower() in ("yes", "set"):
                continue
        if key == "certifications" and re.search("organic", str(k), re.IGNORECASE):
            if text.lower() not in ("yes", "organic", "certified organic"):
                continue
            text = "Organic"
        if key == "certifications" and text.lower() in ("healthier choice", "no", "none"):
            continue
        return text, str(k)
    return None


# -- deriving from the product's own text ------------------------------------------------------------------------------

MATERIALS: Final = (
    "100% cotton", "organic cotton", "cotton", "linen", "polyester", "nylon", "spandex", "wool", "merino wool",
    "cashmere", "silk", "leather", "faux leather", "suede", "denim", "velvet", "fleece", "bamboo", "solid wood",
    "wood", "oak", "walnut", "pine", "acacia", "teak", "mdf", "plywood", "metal", "steel", "stainless steel",
    "aluminum", "aluminium", "iron", "brass", "copper", "glass", "ceramic", "porcelain", "stoneware", "marble",
    "plastic", "abs", "silicone", "rubber", "acrylic", "resin", "paper", "cardboard", "canvas", "rattan", "wicker",
    "foam", "memory foam", "microfiber", "acetate", "titanium", "sterling silver", "gold", "polypropylene",
    "pla", "plant-based", "compostable", "biodegradable", "polycarbonate", "tritan", "melamine", "enamel",
    "cast iron", "vinyl", "pvc", "polyurethane", "pu", "latex", "jute", "sisal", "felt", "down", "feather", "mesh",
    "stone", "granite", "concrete", "crystal", "pearl", "alloy", "zinc", "chrome", "carbon fiber", "fiberglass",
    "terracotta", "clay", "porcelain", "beeswax", "soy wax", "wax", "fabric", "textile", "upholstered", "knit",
)  # fmt: skip
"""Material words a title or description may say; the longest said wins."""
_MATERIAL: Final = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in sorted(MATERIALS, key=len, reverse=True)) + r")\b", re.IGNORECASE
)
_PERCENT: Final = re.compile(r"\b(\d{1,3})\s?%\s?(" + "|".join(re.escape(m) for m in MATERIALS) + r")\b", re.IGNORECASE)
_CARE: Final = re.compile(
    r"\b(machine[ -]washable|machine wash(?:able)?(?: cold| warm)?|hand wash(?: only)?|dry clean(?: only)?|"
    r"dishwasher[ -]safe|wipe clean(?: with a damp cloth)?|spot clean(?: only)?|do not bleach|tumble dry(?: low)?)\b",
    re.IGNORECASE,
)
_UNIT: Final = r"(cm|mm|in|inch|inches|m)\b"
_NUM: Final = r"\d+(?:\.\d+)?"
_DIMS: Final = re.compile(
    rf"\b({_NUM})\s?(?:{_UNIT})?\s?[x×*]\s?({_NUM})\s?(?:{_UNIT})?(?:\s?[x×*]\s?({_NUM}))?\s?{_UNIT}", re.IGNORECASE
)
_WEIGHT: Final = re.compile(rf"\b({_NUM})\s?(kg|g|lbs?|oz|pounds?|grams?)\b(?![-/])", re.IGNORECASE)
_CAPACITY: Final = re.compile(rf"\b({_NUM})\s?(ml|l|liters?|litres?|fl\.? ?oz|qt|quarts?|gallons?|cups?)\b", re.I)
_COMPAT: Final = re.compile(
    r"\b(?:compatible with|fits|for use with|designed for)\s+"
    r"([A-Z0-9][\w\-./ ]{2,40}?)(?=[,.;:()\n|]|\s(?:and|with)\b|$)"
)
_INCLUDED: Final = re.compile(
    r"\b((?:set|pack) of \d+(?: [a-z]+){1,3}|"
    r"(?:includes|comes with):?\s+(?:an? |one |two |three |four |\d+ )[a-z][a-z -]{2,50}[a-z])",
    re.IGNORECASE,
)
_COUNT: Final = re.compile(r"\b(\d+)\s?(?:pcs|pieces|pack|count|ct|pc)\b", re.IGNORECASE)
_ORIGIN: Final = re.compile(r"\bmade in (the )?([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?)")
_SCENT: Final = re.compile(r"\b(?:scent|fragrance)(?: notes?)?\s*:\s*([^.;\n|]{3,60})", re.IGNORECASE)
_LABELLED: Final[Mapping[str, re.Pattern[str]]] = {
    k: re.compile(rf"\b{label}\s*:\s*([^\n|]{{3,240}})", re.IGNORECASE)
    for k, label in (
        ("ingredients", "ingredients"),
        ("allergens", r"(?:allergens?|contains)"),
        ("nutrition", r"(?:nutrition(?: facts)?|nutritional information)"),
        ("warnings", r"(?:warnings?|caution)"),
        ("directions", r"(?:directions(?: for use)?|how to use)"),
        ("features", r"(?:features|key features)"),
        ("warranty", r"warranty"),
    )
}
CLAIMS: Final = (
    "organic", "vegan", "gluten-free", "gluten free", "dairy-free", "non-gmo", "bpa-free", "bpa free",
    "cruelty-free", "hypoallergenic", "sugar-free", "sugar free", "fair trade", "keto", "halal", "kosher",
    "fragrance-free", "paraben-free", "waterproof", "water-resistant", "fsc certified", "oeko-tex",
)  # fmt: skip
_CLAIM: Final = re.compile(r"\b(" + "|".join(re.escape(c) for c in CLAIMS) + r")\b", re.IGNORECASE)
_NEGATED: Final = re.compile(r"\b(?:not|non|no)\s+$", re.IGNORECASE)


def _derive(key: str, title: str, description: str) -> tuple[str, str] | None:  # noqa: PLR0911, PLR0912 - one rule a detail
    """(value, rule) the product's own text states for detail `key`."""
    text = f"{title}\n{description}"
    if key == "material":
        pct = _PERCENT.search(text)
        if pct is not None:
            return f"{pct.group(1)}% {pct.group(2).lower()}", "text: percent material"
        found = _MATERIAL.search(title) or _MATERIAL.search(description)
        return (found.group(1).lower(), "text: material word") if found else None
    if key == "care":
        found = _CARE.search(text)
        return (found.group(1)[0].upper() + found.group(1)[1:].lower(), "text: care phrase") if found else None
    if key == "dimensions":
        found = _DIMS.search(text)
        if found is None:
            return None
        nums = [g for g in found.groups()[:-1] if g and re.fullmatch(_NUM, g)]
        return " x ".join(nums) + f" {found.group(found.lastindex or 0).lower()}", "text: N x N unit"
    if key == "weight":
        found = _WEIGHT.search(text)
        return (f"{found.group(1)} {found.group(2).lower()}", "text: N unit") if found else None
    if key == "capacity":
        found = _CAPACITY.search(text)
        return (f"{found.group(1)} {found.group(2)}", "text: N volume") if found else None
    if key == "compatibility":
        found = _COMPAT.search(text)
        return (found.group(1).strip(), "text: compatible with / fits") if found else None
    if key == "included":
        found = _INCLUDED.search(text)
        return (found.group(1).strip(), "text: includes / set of") if found else None
    if key == "origin":
        found = _ORIGIN.search(text)
        return (found.group(2), "text: made in") if found else None
    if key == "scent":
        found = _SCENT.search(text)
        return (found.group(1).strip(), "text: scent:") if found else None
    if key == "certifications":
        said = []
        for m in _CLAIM.finditer(text):
            if not _NEGATED.search(text[max(0, m.start() - 5) : m.start()]):
                word = m.group(1).lower().replace(" free", "-free")
                if word not in said:
                    said.append(word)
        return (", ".join(said[:4]), "text: claim words") if said else None
    pattern = _LABELLED.get(key)
    if pattern is not None:
        found = pattern.search(description)
        return (found.group(1).strip().rstrip("."), f"text: {key}:") if found else None
    return None


# -- synthetic values, consistent with the product ---------------------------------------------------------------------

SYNTHETIC_MATERIALS: Final[Mapping[str, tuple[str, ...]]] = {
    "clothing-adult": ("cotton", "cotton blend", "polyester", "linen blend"),
    "footwear": ("synthetic upper, rubber sole", "leather upper, rubber sole", "mesh upper, rubber sole"),
    "bags-accessories": ("polyester", "nylon", "faux leather", "canvas"),
    "furniture": ("engineered wood", "solid wood", "metal frame", "upholstered fabric"),
    "home-decor-textiles": ("cotton", "polyester", "linen blend", "ceramic", "glass"),
    "kitchen-dining": ("stainless steel", "ceramic", "glass", "silicone", "bamboo"),
    "garden-outdoor-home": ("powder-coated steel", "resin", "polypropylene", "acacia wood"),
    "toys-games": ("plastic", "wood", "plush fabric"),
    "baby-kids": ("cotton", "plastic", "silicone"),
    "sports-fitness": ("polyester", "nylon", "rubber", "steel"),
    "tools-hardware": ("steel", "aluminum", "plastic"),
    "pet-supplies": ("polyester", "nylon", "plastic"),
    "jewelry-watches": ("stainless steel", "alloy", "sterling silver"),
    "general-merchandise": ("plastic", "metal", "fabric"),
    "gifts-collectibles": ("resin", "ceramic", "metal"),
    "arts-crafts-music": ("wood", "paper", "plastic"),
    "computers-accessories": ("plastic", "aluminum"),
    "consumer-electronics": ("plastic", "aluminum"),
    "automotive-parts": ("steel", "rubber", "plastic"),
    "books-stationery": ("paper",),
}
CARE_BY_MATERIAL: Final = (
    (re.compile(r"cotton|linen|polyester|nylon|fleece|fabric|plush|canvas|microfiber|spandex", re.I),
     "Machine wash cold, tumble dry low"),
    (re.compile(r"wool|cashmere|silk", re.I), "Hand wash cold, dry flat"),
    (re.compile(r"leather|suede", re.I), "Wipe clean with a dry cloth"),
    (re.compile(r"wood|mdf|plywood|rattan|wicker|bamboo|acacia", re.I), "Wipe clean with a dry cloth"),
    (re.compile(r"steel|aluminum|aluminium|metal|iron|brass|copper|alloy|silver|titanium", re.I),
     "Wipe with a damp cloth and dry"),
    (re.compile(r"glass|ceramic|porcelain|stoneware|silicone", re.I), "Dishwasher safe"),
    (re.compile(r"plastic|resin|rubber|acrylic|polypropylene|abs", re.I), "Wipe clean with a damp cloth"),
)  # fmt: skip
WARRANTY: Final[Mapping[str, str]] = {
    "furniture": "1-year limited warranty", "garden-outdoor-home": "1-year limited warranty",
    "tools-hardware": "2-year limited warranty", "computers-accessories": "1-year limited warranty",
    "consumer-electronics": "1-year limited warranty", "kitchen-dining": "1-year limited warranty",
    "sports-fitness": "1-year limited warranty", "automotive-parts": "1-year limited warranty",
    "jewelry-watches": "1-year limited warranty", "bags-accessories": "1-year limited warranty",
    "footwear": "90-day warranty against manufacturing defects", "toys-games": "90-day warranty against defects",
    "baby-kids": "1-year limited warranty", "home-decor-textiles": "90-day warranty against defects",
}  # fmt: skip
"""A store's warranty by category (one for every product of the store, D107); a category not listed sells nothing
warranted, so its stand-in states none rather than a meaningless one on a loaf of bread."""


def _says_any(pattern: re.Pattern[str], title: str, description: str) -> bool:
    return pattern.search(f"{title}\n{description}") is not None


def _synthetic(  # noqa: PLR0911 - one rule a detail
    key: str, product_id: str, category: str, *, title: str, description: str, filled: Mapping[str, str]
) -> tuple[str, str] | None:
    """(value, rule) for detail `key`, seeded by the product id; None where any value could contradict the text."""
    rng = seeded_rng(RULE, product_id, key)
    if key == "material":
        choices = SYNTHETIC_MATERIALS.get(category)
        if not choices or _says_any(_MATERIAL, title, description):
            return None  # the text names a material we did not read, or the category sells none to name
        return rng.choice(choices), f"synthetic: {category} material vocabulary"
    if key == "care":
        material = filled.get("material") or ""
        if _says_any(_CARE, title, description) or not material:
            return None
        found = next((care for pattern, care in CARE_BY_MATERIAL if pattern.search(material)), None)
        return (found, "synthetic: care of its material") if found else None
    if key == "included":
        if _COUNT.search(title) or _INCLUDED.search(description):
            return None
        return "1 item, as pictured", "synthetic: one item"
    if key == "warranty":
        found = WARRANTY.get(category)
        return (found, "store: the store's warranty") if found else None
    return None


# -- the fill ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Filled:
    products: dict[str, dict[str, Any]]
    summary: dict[str, Any]


def fill_products(  # noqa: PLR0912, PLR0915 - the three origins in order, per detail and product
    products: Mapping[str, Mapping[str, Any]],
    category: str,
    spec: CoreSpec,
    *,
    layers: Layers | None = None,
    policies: Sequence[Mapping[str, Any]] = (),
) -> Filled:
    """Every product with its core filled, the base untouched: a value added only where the base states none, each
    recorded in the product's `provenance.union_core` as {detail: {type, origin, layer, rule}}, and the store's
    policies (`store_policies`) attached as its root policies when it has none. Deterministic."""
    layered = layers or load_layers()
    core = spec.core.core(category, spec.share)
    slots = [(t, SLOTS[t]) for t in core if t in SLOTS]
    details_wanted = list(dict.fromkeys((s.key, t) for t, s in slots if s.where == "detail"))
    policy_types = sorted({t for t, s in slots if s.where == "policy"})
    out: dict[str, dict[str, Any]] = {}
    counts: dict[str, dict[str, int]] = {}
    for pid, facts in sorted(products.items()):
        doc = loads(dumps(facts))
        meta = doc.setdefault("metadata", {})
        details = dict(meta.get("details") or {})
        attributes = meta.get("attributes") or {}
        title = str(doc.get("title") or "")
        description = str(doc.get("description") or "") if isinstance(doc.get("description"), str) else ""
        record: dict[str, dict[str, str]] = {}
        for key, type_id in details_wanted:
            layer = layered.layer(category, type_id)
            stated = clean(details.get(key))
            got: tuple[str, Origin, str] | None = None
            if stated is not None:
                got = (stated, "catalog", f"details.{key}")
            elif (hit := _from_attributes(key, attributes)) is not None:
                got = (hit[0], "catalog", f"attributes.{hit[1]}")
            elif (hit := _derive(key, title, description)) is not None:
                got = (hit[0], "derived", hit[1])
            elif SLOTS[type_id].synthetic and key not in SAFETY:
                filled_now = {k: v for k, v in details.items() if isinstance(v, str)}
                hit = _synthetic(key, pid, category, title=title, description=description, filled=filled_now)
                if hit is not None:
                    origin: Origin = "store" if hit[1].startswith("store:") else "synthetic"
                    got = (hit[0], origin, hit[1])
            if got is None:
                counts.setdefault(type_id, {}).setdefault("none", 0)
                counts[type_id]["none"] += 1
                continue
            value, origin, rule = got
            if stated is None:
                details[key] = value  # added beside the base, never in its place
            record[key] = {"type": type_id, "origin": origin, "layer": layer, "rule": rule}
            counts.setdefault(type_id, {}).setdefault(origin, 0)
            counts[type_id][origin] += 1
        if details:
            meta["details"] = details
        elif not meta:
            doc.pop("metadata", None)
        if policy_types and policies and not doc.get("_policies"):
            doc["_policies"] = [dict(p) for p in policies]
            for type_id in policy_types:
                fact = SLOTS[type_id].key
                if any(p.get("fact") == fact for p in policies):
                    record[f"policy:{fact}:{type_id}"] = {
                        "type": type_id, "origin": "store", "layer": layered.layer(category, type_id),
                        "rule": f"store spec: {fact}",
                    }  # fmt: skip
                    counts.setdefault(type_id, {}).setdefault("store", 0)
                    counts[type_id]["store"] += 1
        prov = doc.setdefault("provenance", {})
        prov[PROVENANCE] = record
        out[pid] = doc
    summary = {
        "rule": RULE,
        "core": spec.ref,
        "category": category,
        "types": list(core),
        "filled": {t: dict(sorted(c.items())) for t, c in sorted(counts.items())},
        "experimental": list(layered.experimental.get(category, ())),
        "digest": digest([RULE, spec.ref, sorted(out)], 12),
    }
    return Filled(out, summary)


def store_policies(market: Any, category: str, *, currency_exponent: int = 2) -> list[dict[str, Any]]:
    """The store's terms as the UCP catalog's root policies, read from its market (the checkout's own table, so a
    policy never says what its checkout does not do): its return window, free returns, the cheapest shipping and the
    free-shipping threshold, the slowest standard transit, and its warranty where the category sells one."""
    del currency_exponent
    out: list[dict[str, Any]] = []
    days = int(getattr(market, "return_window_days", 0) or 0)
    if days > 0:
        out.append(_policy("returns_window_days", "return_policy", days,
                           f"Returns are accepted within {days} days of delivery."))  # fmt: skip
        out.append(_policy("returns_fee_minor", "return_policy", 0, "Returns are free."))
    shipping = sorted(getattr(market, "shipping", ()) or (), key=lambda o: int(o.amount))
    if shipping:
        # Every option the checkout offers, never the cheapest alone: a policy naming only Standard read as "no
        # express" and agents gave up express tasks the checkout could meet (2026-10-17 trace test, ep-9d1a2c9b1b64)
        listed = "; ".join(f"{o.title} {_money(int(o.amount), market)}" for o in shipping)
        out.append(_policy("shipping_cost_minor", "shipping_policy", int(shipping[0].amount),
                           f"Shipping options: {listed}."))  # fmt: skip
        free = sorted(int(o.free_above) for o in shipping if getattr(o, "free_above", None) is not None)
        if free:
            out.append(_policy("free_shipping_threshold_minor", "shipping_policy", free[0],
                               f"Free shipping on orders over {_money(free[0], market)}."))  # fmt: skip
        spans = [m for o in shipping if (m := re.search(r"(\d+)\s*-\s*(\d+)\s*days", str(o.title)))]
        if spans:
            low, high = min(int(m.group(1)) for m in spans), max(int(m.group(2)) for m in spans)
            out.append(_policy("transit_days_max", "shipping_policy", high,
                               f"Delivery takes {low} to {high} days, by the shipping option chosen."))  # fmt: skip
    warranty = WARRANTY.get(category)
    if warranty:
        out.append(_policy("warranty", "warranty", None, f"Every product comes with a {warranty}."))
    return out


def _policy(fact: str, kind: str, value: Any, text: str) -> dict[str, Any]:
    body: dict[str, Any] = {"description": {"plain": text}, "fact": fact, "type": kind}
    if value is not None:
        body["value"] = value
    return body


def _money(minor: int, market: Any) -> str:
    currency = str(getattr(market, "currency", "USD"))
    symbol = {"USD": "$", "GBP": "£", "EUR": "€"}.get(currency, f"{currency} ")
    return f"{symbol}{minor / 100:,.2f}"


def origins(facts: Mapping[str, Any]) -> Mapping[str, Mapping[str, str]]:
    """A product's recorded core values (detail or policy to its type, origin, layer and rule); empty for a product
    built before the core."""
    prov = facts.get("provenance")
    found = prov.get(PROVENANCE) if isinstance(prov, Mapping) else None
    return found if isinstance(found, Mapping) else {}


__all__ = [
    "CORE_FILE",
    "LAYERS_FILE",
    "RULE",
    "SLOTS",
    "CoreFile",
    "CoreSpec",
    "Filled",
    "Layers",
    "Slot",
    "build_core_file",
    "fill_products",
    "load_core",
    "load_layers",
    "origins",
    "spec_of",
    "store_policies",
]


def write_layers(core: CoreFile, share: float, path: Path, categories_dir: Path) -> None:
    """data/sandbox/union_layers.yaml's first draft: per category, the core product types its niche spec
    (data/categories/niches, D92) names as what its shoppers decide by and a mask may withhold. The team edits it;
    in v1 nothing is withheld (`mask_bp` 0), so it only labels the sheets' `layer`."""
    from carttrace.collect.exposure import BY_ID
    from carttrace.stores.niches import load_niches, niches_dir

    book = load_niches(niches_dir(categories_dir))
    experimental: dict[str, list[str]] = {}
    for cat in sorted(core.categories):
        spec = book.get(cat)
        masked: set[str] = set()
        for niche in spec.niches if spec is not None else ():
            base = book.niche(niche.id)
            if base.kind == "data":
                masked |= set(niche.mask or base.mask) | set(base.types)
        experimental[cat] = [
            t for t in core.core(cat, share) if t in masked and t in BY_ID and t in SLOTS and SLOTS[t].where == "detail"
        ]
    head = (
        "# The union layers of the open stand-ins (D117). base: the catalog's own facts, never changed. core: the\n"
        "# category's union core (data/bench/union_core.json at sandbox.v1.union_core_share), always served in v1.\n"
        "# experimental: the core types below, which a later experiment may withhold or show per row through the\n"
        "# detail_hidden cut (carttrace.collect.union_core.Layers.details); in v1 nothing is withheld (mask_bp 0).\n"
    )
    body = {"schema": LAYERS_SCHEMA, "experimental": experimental}
    path.write_text(head + yaml.safe_dump(body, sort_keys=True, width=110), encoding="utf-8")

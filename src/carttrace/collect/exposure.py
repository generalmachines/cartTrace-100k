"""Exposure masks (D63): a sandbox world that serves some of its store's data types and withholds the others, so the
benchmark can read which data an agent was given and how it reacted.

Why a mask, and why on the world.detail_hidden hides one detail from one reply filter and only where a task turns on it (D47); a mask
is a property of the store itself: a world whose replica serves its own products with each withheld data type removed
from the pack, so every surface the world mounts (UCP over MCP, UCP REST, the ACP row's UCP catalog, the WebMCP
storefront) serves it without the type, and the wire records the store answering without it. The task is any task of
the store; the mask is recorded on the world's recipe and on the episode's header (`exposure`), and the analytics read
the pattern from the frames.

What a data type is. One row of `TYPES`: a curated id of the category data-type union (D62, data/bench/union.yaml,
which `carttrace bench union` reads stores against), the store brief's data kinds whose detectors say whether a product
carries it (carttrace.agents.brief, D49's kinds, the union type's `kinds`), the other curated ids it withholds with it, and
how a world withholds it:
- `details`: the catalog details whose statements the detail_hidden matchers cut (carttrace.collect.matchers);
- `text`: the sentences, lines and tags the kind's own pattern reads are cut, so the detector cannot find them;
- `keys`: structured keys removed at any depth of a product's facts (a review rating, image alt text, barcodes);
- `names`: a pattern over the names of metadata, metafield and attribute entries and of the policies a store attached
  to its products (a `processing_days` policy is a delivery promise); such entries go;
- `shipping_titles`: the market's shipping option titles lose what the pattern reads ("Standard (4-7 days)").
Text is only ever cut, never rewritten (carttrace.collect.excise). The table is the one place the mapping lives; a kind the
union curates no type of yet keeps an id in the union's scheme until it does. Options, prices, availability, the
description as a whole and shipping costs are never withheld: without them a store cannot be shopped at all, which
measures a broken store rather than missing data.

Honest by construction. A mask is planned on a world only when every type it withholds is served there before the cut
(some product carries it, read by the detectors on what the world serves: its listings, its get_product answers and
the policies served with them, under the world's platform view) and served nowhere after it. `served_counts` reads
both from the surface's own catalog (`served_catalog`), so the check reads what an agent would be sent. A type that
survives the cut (a title that names it, say) is not withheld on that world, and the manifest says so.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from carttrace.agents.brief import EXPOSED_PRODUCTS, EXPOSED_SHARE, carried_by, kind_pattern
from carttrace.agents.products import Grid, read_product
from carttrace.collect.excise import excise
from carttrace.collect.matchers import MATCHERS, HiddenSet
from carttrace.core.canon import digest, dumps
from carttrace.core.errors import ValidationError
from carttrace.sandbox.envpack import EnvPack
from carttrace.sandbox.surfaces.ucp.catalog import POLICIES, served_catalog
from carttrace.trace.schema import Condition, ExposureCount, ExposureDesign, ExposureMask

TYPES_VERSION: Final = "exposure-types-2026-10-14"
CONDITION: Final = "exposure"
"""The condition name of every row of an exposure design: its params name the design, the mask and what it withholds,
so a reader pairs rows by store, task, seed and platform without knowing how they were made."""
DESIGNS: Final[tuple[ExposureDesign, ...]] = ("baseline", "leave_one_out", "random")
RANDOM_MASKS: Final = 3
"""Seeded random masks per store, beside its baseline and one leave-one-out mask per type it shows."""
DESCRIPTION_KEYS: Final = frozenset({"description", "description_html", "descriptionHtml", "body_html"})
ENTRY_KEYS: Final = frozenset({"attributes", "metafields", "details", "metadata"})
"""Product data keyed by name: a store's attributes, metafields and metadata, and a catalog store's details."""
TAG_KEYS: Final = frozenset({"tags"})
PATTERN_PREFIX: Final = "~"
"""Before a type id where the cut set names a kind's pattern rather than a catalog detail."""
_PARENTHETICAL: Final = re.compile(r"\s*\(([^()]*)\)")


@dataclass(frozen=True, slots=True)
class DataType:
    """One data type a store may show, how its presence is read and how a world withholds it."""

    id: str
    """The category data-type union's curated id (D62, data/bench/union.yaml) the row withholds."""
    kinds: tuple[str, ...]
    """The store brief's data kinds (carttrace.agents.brief.DATA_KINDS) whose detectors read it: the union type's `kinds`."""
    union: tuple[str, ...] = ()
    """Every curated union id the cut withholds with it (the id itself first): one row withholds a kind whole, so a
    review rating, its count and its text go together. Empty for a kind the union has no curated type of yet, whose id
    follows the union's scheme."""
    details: tuple[str, ...] = ()
    text: bool = False
    keys: tuple[str, ...] = ()
    names: str | None = None
    shipping_titles: bool = False

    @property
    def patterns(self) -> tuple[re.Pattern[str], ...]:
        """The text patterns of its kinds, when the type cuts text."""
        if not self.text:
            return ()
        return tuple(p for k in self.kinds if (p := kind_pattern(k)) is not None)

    @property
    def name_pattern(self) -> re.Pattern[str] | None:
        return re.compile(self.names, re.IGNORECASE) if self.names else None


def _t(type_id: str, kinds: str, union: str = "", **how: Any) -> DataType:
    """A row: its union id, its kinds (space-separated) and the other curated ids it withholds with it."""
    return DataType(id=type_id, kinds=tuple(kinds.split()), union=(type_id, *union.split()), **how)


def _uncurated(type_id: str, kinds: str, **how: Any) -> DataType:
    return DataType(id=type_id, kinds=tuple(kinds.split()), **how)


TYPES: Final[tuple[DataType, ...]] = (
    _t(
        "reviews.rating",
        "reviews",
        "reviews.count reviews.text",
        text=True,
        keys=(
            "rating",
            "ratings",
            "review_count",
            "reviews_count",
            "reviews",
            "review",
            "aggregate_rating",
            "aggregateRating",
            "reviewCount",
            "ratingValue",
        ),
        names=r"\brating|\breviews?\b|review_count",
    ),
    _t("product.size_chart", "size_chart", "store.size_guide", details=("size_chart",), names=r"size[ _]?(chart|gu)"),
    _t(
        "delivery.estimate",
        "delivery_estimate",
        text=True,
        names=r"processing|transit|deliver|dispatch|lead[ _]?time|ship(ping)?[ _]?time|handling|cut[ _-]?off",
        shipping_titles=True,
    ),
    _t(
        "returns.policy",
        "returns",
        "returns.window returns.fee returns.final_sale",
        text=True,
        names=r"\breturns?|refund|exchange",
    ),
    _t("policy.warranty", "warranty", text=True, names=r"warrant|guarantee"),
    # an ingredient list states what a product contains, and the allergen reader reads it so: both go
    _t("product.allergens", "allergens", details=("allergens", "ingredients"), names=r"allergen|ingredient"),
    _t("product.ingredients", "ingredients", details=("ingredients",), text=True, names=r"ingredient|\binci\b"),
    _t("product.nutrition", "nutrition", text=True, names=r"nutrition"),
    _t("product.warnings", "warnings", details=("warnings",), names=r"warning|contraindication|caution"),
    _t("product.directions", "dosage", details=("dosage",), text=True, names=r"dosage|directions"),
    _t("product.materials", "materials", details=("material",), text=True, names=r"material|fabric|composition"),
    _t("product.care", "care", details=("care",), text=True, names=r"\bcare\b|wash"),
    _t("product.measurements", "dimensions", details=("dimensions",), text=True, names=r"dimension"),
    _t("product.weight", "weight", details=("weight",), text=True, names=r"\bweight\b"),
    _t("product.compatibility", "compatibility", details=("compatibility",), text=True, names=r"compatib"),
    _t("badge.claims", "dietary certifications", text=True, names=r"dietary|\bdiet\b|certif"),
    _t("product.origin", "origin", text=True, names=r"origin|made[ _]in"),
    _t("offer.subscription", "subscription", "policy.subscription", text=True, names=r"subscri|selling[ _]?plan"),
    _t("offer.bundle", "bundle", "product.whats_included", text=True),
    _t("offer.gift_options", "gift", text=True, names=r"\bgift"),
    _t("offer.personalisation", "personalization", text=True, names=r"personali[sz]|engrav|monogram"),
    _t("product.scent_notes", "scent", text=True, names=r"scent|fragrance[ _]notes|\bnotes\b"),
    _t("product.power", "power", text=True, names=r"voltage|battery|power[ _]source|wattage"),
    _t("product.capacity", "capacity", text=True, names=r"capacity"),
    # the union curates it by its exposure row (shipping.restrictions) and names no kind
    _t("delivery.restrictions", "shipping_restrictions", text=True),
    _t("media.alt_text", "media_alt", keys=("alt_text", "altText", "alt")),
    _t("identity.brand", "brand", keys=("vendor", "brand")),
    _t(
        "identity.gtin",
        "identifiers",
        text=True,
        keys=("barcodes", "barcode", "gtin", "gtin8", "gtin12", "gtin13", "gtin14", "upc", "ean", "isbn", "mpn"),
    ),
    _t(
        "price.compare_at",
        "price.list",
        "price.discount",
        keys=("list_price", "list_price_range", "compare_at_price", "compare_at_price_range"),
    ),
    # kinds the brief reads that the union curates no type of yet (D62): ids in its scheme, no curated ids withheld
    _uncurated("product.metal", "metal", text=True, names=r"\bmetal\b|purity"),
    _uncurated("product.stone", "stone", text=True, names=r"\bstone|carat"),
    _uncurated("product.hair_type", "hair_type", text=True, names=r"hair[ _]?type"),
    _uncurated("product.condition", "condition", text=True, names=r"\bcondition\b"),
    _uncurated("product.pet_type", "pet_type", text=True, names=r"species|life[ _]?stage|breed"),
    _uncurated("product.age_range", "age_range", text=True, names=r"\bage[ _]?(range|grading)?\b"),
    _uncurated("badge.safety_standard", "safety_standard", text=True, names=r"safety[ _]standard"),
)
"""Every data type an exposure mask may withhold, in display order, keyed by the category data-type union's curated
ids (D62). A test holds each id and its kinds to data/bench/union.yaml."""
NEVER_WITHHELD: Final[Mapping[str, str]] = {
    "variant.options": "without option values no variant can be chosen",
    "price.current": "a store without prices cannot be shopped",
    "price.unit": "read from titles and option values, which are never cut",
    "variant.availability": "stock is the state's, and hiding it tests stock rather than data",
    "product.description": "withholding the whole description withholds most other types at once",
    "delivery.cost": "shipping costs are the checkout's arithmetic",
}
BY_ID: Final[Mapping[str, DataType]] = {t.id: t for t in TYPES}


def data_type(type_id: str) -> DataType:
    found = BY_ID.get(type_id)
    if found is None:
        raise ValidationError(
            f"{type_id!r} is not an exposure data type",
            hint=f"choose from {', '.join(BY_ID)}; never {', '.join(NEVER_WITHHELD)}",
        )
    return found


def mask_id(withheld: Iterable[str]) -> str:
    """One id per set of withheld types, the same on every store: xm-<sha12>."""
    return "xm-" + digest(["exposure-mask", sorted(set(withheld))], 12)


def ordered(type_ids: Iterable[str]) -> list[str]:
    """Type ids in table order, each once."""
    wanted = set(type_ids)
    return [t.id for t in TYPES if t.id in wanted]


def exposure_condition(design: ExposureDesign, withheld: Sequence[str]) -> Condition:
    """The condition of one exposure row: what it withholds, as a mask id and the type ids, and how it was chosen."""
    return Condition(
        name=CONDITION,
        params={"design": design, "mask": mask_id(withheld), "withheld": ordered(withheld)},
    )


def design_masks(
    shown: Sequence[str], *, seed: int, random_masks: int = RANDOM_MASKS
) -> list[tuple[ExposureDesign, tuple[str, ...]]]:
    """The masks that cover one store's space: the full-exposure baseline, each type it shows withheld on its own,
    and `random_masks` seeded draws of two or more types (never all of them, never a set already planned). The draw
    ranks types by a hash of the seed, the draw and the type, so the same seed always gives the same masks."""
    types = ordered(shown)
    out: list[tuple[ExposureDesign, tuple[str, ...]]] = [("baseline", ())]
    out += [("leave_one_out", (t,)) for t in types]
    if len(types) < 3:
        return out
    seen = {withheld for _, withheld in out}
    for draw in range(random_masks * 8):
        if len(out) - 1 - len(types) >= random_masks:
            break
        size = 2 + int(digest(["exposure-size", seed, draw], 8), 16) % (len(types) - 2)
        ranked = sorted(types, key=lambda t: digest(["exposure-rank", seed, draw, t], 12))
        pick = tuple(ordered(ranked[:size]))
        if pick not in seen:
            seen.add(pick)
            out.append(("random", pick))
    return out


# -- what a world serves ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Served:
    """What a world serves: its products, per type how many of them carry it, and per product the types it carries."""

    products: int
    counts: Mapping[str, int]
    carried: Mapping[str, frozenset[str]] = field(default_factory=dict)
    """Product id to the types its served documents carry."""

    def updated(self, carried: Mapping[str, frozenset[str]]) -> Served:
        """These counts with some products read again: what a world serves once a cut changed only those."""
        merged = {**self.carried, **carried}
        counts: Counter[str] = Counter()
        for found in merged.values():
            counts.update(found)
        return Served(self.products, {t: counts.get(t, 0) for t in self.counts}, merged)

    def shows(self, type_id: str) -> bool:
        """Whether any product serves the type: something to withhold, or something left after a cut."""
        return self.counts.get(type_id, 0) > 0

    def exposes(self, type_id: str) -> bool:
        """Whether the world exposes the type as the store brief counts a kind exposed (three products, or a tenth
        of them): the types an exposure design spans, so a mask never withholds what one product in a thousand
        mentions in passing."""
        count = self.counts.get(type_id, 0)
        return count > 0 and (count >= EXPOSED_PRODUCTS or count >= EXPOSED_SHARE * self.products)


def served_counts(pack: EnvPack, types: Sequence[DataType] = TYPES) -> Served:
    """Per type, how many of the pack's products carry it in what the world serves: the search listing and the
    get_product answer the surface's own catalog renders for each product (under the pack's platform view, D55), with
    the policies served beside them. A document read once is not read again (`_seen`)."""
    carried = served_types(pack, types=types)
    counts: Counter[str] = Counter()
    for found in carried.values():
        counts.update(found)
    return Served(products=len(carried), counts={t.id: counts.get(t.id, 0) for t in types}, carried=carried)


def served_types(
    pack: EnvPack, *, only: Collection[str] | None = None, types: Sequence[DataType] = TYPES
) -> dict[str, frozenset[str]]:
    """Per product (of `only`, or all), the types its served documents carry. A product's listing and get_product
    answer are rendered from its own facts and the platform's view, so a product a cut left alone serves what it
    served before and need not be read again."""
    catalog = served_catalog(pack)
    state = pack.initial_state()
    ids = tuple(t.id for t in types)
    out: dict[str, frozenset[str]] = {}
    for product in catalog.products(state):
        if only is not None and product.id not in only:
            continue
        docs = [catalog.listing(product), catalog.detail({"id": product.id}, product, None)]
        policies = catalog.policies([("$.product", product)])
        found: set[str] = set()
        for doc in docs:
            found |= _seen_carried(doc, policies, types, ids)
        out[product.id] = frozenset(found)
    return out


_SEEN: dict[str, frozenset[str]] = {}
_SEEN_MAX: Final = 200_000


def _seen_carried(
    doc: Mapping[str, Any], policies: Sequence[Mapping[str, Any]], types: Sequence[DataType], ids: tuple[str, ...]
) -> frozenset[str]:
    key = digest([ids, doc, policies], 24)
    found = _SEEN.get(key)
    if found is None:
        if len(_SEEN) >= _SEEN_MAX:
            _SEEN.clear()
        found = _SEEN[key] = frozenset(carried(doc, policies, types))
    return found


def carried(doc: Mapping[str, Any], policies: Sequence[Mapping[str, Any]], types: Sequence[DataType]) -> set[str]:
    """The types one served product document carries: by its kind's detector (on the product and on each variant
    document, which an agent reads as well), by a structured key, by a named entry or by a policy served with it."""
    readings = _readings(doc)
    keys, labels = _shape(doc)
    out: set[str] = set()
    for spec in types:
        if _carries(spec, readings, keys, labels, policies):
            out.add(spec.id)
    return out


def _carries(
    spec: DataType,
    readings: Sequence[tuple[Any, Mapping[str, Any]]],
    keys: frozenset[str],
    labels: Sequence[str],
    policies: Sequence[Mapping[str, Any]],
) -> bool:
    if spec.keys and not keys.isdisjoint(spec.keys):
        return True
    names = _names(spec)
    if names is not None and any(names.search(label) for label in labels):
        return True
    if any(_policy_carries(policy, spec) for policy in policies):
        return True
    return any(carried_by(kind, product, raw) for kind in spec.kinds for product, raw in readings)


def _readings(doc: Mapping[str, Any]) -> list[tuple[Any, Mapping[str, Any]]]:
    out: list[tuple[Any, Mapping[str, Any]]] = []
    if doc.get("id") is not None:
        out.append((read_product(doc), doc))
    own = dumps(doc.get("description"))
    for variant in doc.get("variants") or ():
        if isinstance(variant, Mapping) and variant.get("id") is not None and _own_text(variant, own):
            out.append((read_product(variant), variant))
    return out


def _own_text(variant: Mapping[str, Any], product_description: str) -> bool:
    """Whether a variant's document says anything of its own a detector could read (a description other than its
    title or its product's, metadata, tags): a variant without is read through its product, and reading it again
    would only cost time."""
    if any(variant.get(k) for k in (*ENTRY_KEYS, *TAG_KEYS)):
        return True
    for key in DESCRIPTION_KEYS:
        texts = _strings(variant.get(key))
        if texts and dumps(variant.get(key)) != product_description and texts != [str(variant.get("title") or "")]:
            return True
    return False


def _shape(doc: Any) -> tuple[frozenset[str], list[str]]:
    """In one walk of a document: every key that holds a value, and the label of every named entry (a metadata,
    metafield or attribute key, or a listed entry's key or name)."""
    keys: set[str] = set()
    labels: list[str] = []
    stack: list[Any] = [doc]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            for key, item in value.items():
                if item in (None, "", [], {}):
                    continue
                keys.add(str(key))
                if key in ENTRY_KEYS:
                    labels.extend(_entry_labels(item))
                if isinstance(item, Mapping | list):
                    stack.append(item)
        elif isinstance(value, list):
            stack.extend(item for item in value if isinstance(item, Mapping | list))
    return frozenset(keys), labels


def _entry_labels(entries: Any) -> list[str]:
    if isinstance(entries, Mapping):
        return [_label(str(k)) for k, v in entries.items() if v not in (None, "", [], {})]
    if isinstance(entries, list):
        return [_label(name) for e in entries if (name := _entry_name(e)) is not None]
    return []


_NAMES: dict[str, re.Pattern[str] | None] = {}


def _names(spec: DataType) -> re.Pattern[str] | None:
    """The type's entry-name pattern, compiled once."""
    if spec.id not in _NAMES:
        _NAMES[spec.id] = spec.name_pattern
    return _NAMES[spec.id]


def _policy_carries(policy: Mapping[str, Any], spec: DataType) -> bool:
    names = _names(spec)
    if names is not None and any(names.search(_label(str(policy.get(k) or ""))) for k in ("fact", "type", "title")):
        return True
    texts = _strings(policy.get("description"))
    return any(pattern.search(text) for pattern in spec.patterns for text in texts)


# -- withholding -----------------------------------------------------------------------------------------------------


class CutSet(HiddenSet):
    """What one mask cuts from product text: the catalog details of its types, as detail_hidden reads them, and the
    patterns of its text types, named `~<type id>`. Read by carttrace.collect.excise exactly as a HiddenSet is."""

    def __init__(self, types: Sequence[DataType]) -> None:
        names = sorted({d for t in types for d in t.details})
        self.details = frozenset(names)
        self._matchers = tuple(MATCHERS[name] for name in MATCHERS if name in self.details)
        self.patterns: dict[str, re.Pattern[str]] = {
            f"{PATTERN_PREFIX}{t.id}#{i}": pattern for t in types for i, pattern in enumerate(t.patterns)
        }
        self.names: dict[str, re.Pattern[str]] = {
            PATTERN_PREFIX + t.id: pattern for t in types if (pattern := t.name_pattern) is not None
        }
        self.owner: dict[str, str] = {name: name[1:].split("#")[0] for name in [*self.patterns, *self.names]}
        for spec in types:
            for detail in spec.details:
                self.owner.setdefault(detail, spec.id)

    def keyed(self, key: str) -> frozenset[str]:
        label = _label(key)
        found = set(super().keyed(key)) if self.details else set()
        found |= {name for name, pattern in self.names.items() if pattern.search(label)}
        found |= {name for name, pattern in self.patterns.items() if pattern.search(label)}
        return frozenset(found)

    def stated(self, text: str) -> frozenset[str]:
        found = set(super().stated(text)) if self.details else set()
        found |= {name for name, pattern in self.patterns.items() if pattern.search(text)}
        return frozenset(found)

    def chart(self, grid: Grid) -> bool:
        return bool(self.details) and super().chart(grid)

    def types_of(self, counts: Mapping[str, int]) -> Counter[str]:
        """Counts by cut-set name, as counts by type id."""
        out: Counter[str] = Counter()
        for name, count in counts.items():
            owner = self.owner.get(name)
            if owner is not None:
                out[owner] += count
        return out


@dataclass(frozen=True, slots=True)
class Withheld:
    """A world with data types withheld: the new pack, and per type how many statements, entries and keys went."""

    pack: EnvPack
    removed: Mapping[str, int]
    changed: frozenset[str] = frozenset()
    """The products whose facts the cut changed."""


def withhold(pack: EnvPack, withheld: Sequence[str]) -> Withheld:
    """`pack` with every type in `withheld` cut from its products' facts (every view's values and every variant's
    document included, and the policies attached to them) and, for a type that says so, from its market's shipping
    titles. A new pack, its provenance naming the base pack and what went; the same pack when nothing is withheld."""
    types = [data_type(t) for t in dict.fromkeys(withheld)]
    if not types:
        return Withheld(pack=pack, removed={})
    types = [data_type(t) for t in ordered(t.id for t in types)]
    doc = pack.doc
    cut = _Cut(types)
    products: dict[str, Any] = doc["snapshot"]["products"]
    changed: set[str] = set()
    for pid in sorted(products):
        new = _cut_facts(cut, products[pid])
        if new != products[pid]:
            changed.add(pid)
        products[pid] = new
    if any(t.shipping_titles for t in types):
        for option in doc["market"].get("shipping") or ():
            if isinstance(option, dict) and isinstance(option.get("title"), str):
                option["title"] = cut.title(option["title"])
    removed = {t.id: cut.counts.get(t.id, 0) for t in types}
    provenance = dict(doc.get("provenance") or {})
    provenance[CONDITION] = {"base": pack.id, "types": TYPES_VERSION, "withheld": [t.id for t in types]}
    doc["provenance"] = provenance
    return Withheld(pack=EnvPack.from_bytes(dumps(doc).encode("utf-8")), removed=removed, changed=frozenset(changed))


_CUTS: dict[str, tuple[Any, Counter[str]]] = {}
_CUTS_MAX: Final = 200_000


def _cut_facts(cut: _Cut, facts: Any) -> Any:
    """One product's facts after the cut, remembered by content (D72): a store's platform packs and payment rails
    serve the same product under many worlds, and the cut is a pure function of the facts and the withheld types, so
    a product cut once is not cut again. The counts the cut made are added again on every use. The facts returned
    are shared between worlds and only ever serialized, never changed."""
    key = digest([[t.id for t in cut.types], facts], 24)
    found = _CUTS.get(key)
    if found is None:
        before = Counter(cut.counts)
        new = cut.facts(facts)
        if len(_CUTS) >= _CUTS_MAX:
            _CUTS.clear()
        _CUTS[key] = (new, cut.counts - before)
        return new
    cut.counts.update(found[1])
    return found[0]


@dataclass(slots=True)
class _Cut:
    types: Sequence[DataType]
    cutset: CutSet = field(init=False)
    keys: dict[str, str] = field(init=False)
    counts: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self.cutset = CutSet(self.types)
        self.keys = {key: t.id for t in self.types for key in t.keys}

    def facts(self, facts: Any) -> Any:
        return self._value(copy.deepcopy(facts))

    def _value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for key, item in value.items():
                owner = self.keys.get(key)
                if owner is not None and item not in (None, "", [], {}):
                    self.counts[owner] += 1
                    continue
                new = self._member(key, item)
                if isinstance(item, Mapping | list) and item and not new:
                    continue  # a container the cut emptied goes, as a store that never had the data sends none
                out[key] = new
            return out
        if isinstance(value, list):
            return [self._value(item) for item in value]
        return value

    def _member(self, key: str, item: Any) -> Any:
        if key in DESCRIPTION_KEYS:
            return self._description(item, as_html=None if key == "description" else True)
        if key in ENTRY_KEYS:
            return self._entries(item)
        if key in TAG_KEYS and isinstance(item, list):
            return [tag for tag in item if not (isinstance(tag, str) and self._stated(tag))]
        if key == POLICIES and isinstance(item, list):
            return [p for p in (self._policy(policy) for policy in item) if p is not None]
        return self._value(item)

    def _description(self, item: Any, *, as_html: bool | None) -> Any:
        if isinstance(item, str):
            return self._text(item, as_html=as_html)
        if not isinstance(item, Mapping):
            return self._value(item)
        return {
            form: self._text(text, as_html=form == "html") if isinstance(text, str) else text
            for form, text in item.items()
        }

    def _text(self, text: str, *, as_html: bool | None = None) -> str:
        cut, found = excise(text, self.cutset, as_html=as_html)
        self.counts.update(self.cutset.types_of(found))
        return cut

    def _stated(self, text: str) -> bool:
        found = self.cutset.stated(text)
        self.counts.update(self.cutset.types_of(dict.fromkeys(found, 1)))
        return bool(found)

    def _entries(self, item: Any) -> Any:
        """Named entries without those named for a withheld type, and the rest with their text cut."""
        if isinstance(item, Mapping):
            kept: dict[str, Any] = {}
            for key, value in item.items():
                owner = self.keys.get(str(key))
                if owner is not None and value not in (None, "", [], {}):
                    self.counts[owner] += 1
                    continue
                if self._keyed(str(key)):
                    continue
                new = self._text(value) if isinstance(value, str) else self._value(value)
                if isinstance(value, Mapping | list) and value and not new:
                    continue
                kept[key] = new
            return kept
        if isinstance(item, list):
            entries: list[Any] = []
            for entry in item:
                name = _entry_name(entry)
                if name is not None and self._keyed(name):
                    continue
                entries.append(self._value(entry))
            return entries
        return self._value(item)

    def _keyed(self, key: str) -> bool:
        found = self.cutset.keyed(key)
        self.counts.update(self.cutset.types_of(dict.fromkeys(found, 1)))
        return bool(found)

    def _policy(self, policy: Any) -> Any:
        """A policy without what states a withheld type, or None when it is about one (its fact or type is named
        for it) or the cut left it nothing to say."""
        if not isinstance(policy, Mapping):
            return policy
        for spec in self.types:
            pattern = _names(spec)
            if pattern is not None and any(pattern.search(_label(str(policy.get(k) or ""))) for k in ("fact", "type")):
                self.counts[spec.id] += 1
                return None
        out = self._value(policy)
        description = out.get("description")
        if "description" in policy and not _strings(description):
            return None
        return out

    def title(self, title: str) -> str:
        """A shipping option's title without the parenthetical a withheld type's pattern reads."""

        def drop(found: re.Match[str]) -> str:
            return "" if self._stated(found.group(1)) else found.group(0)

        return _PARENTHETICAL.sub(drop, title).strip() or title


def _entry_name(entry: Any) -> str | None:
    if isinstance(entry, Mapping):
        for name in ("key", "name"):
            if isinstance(entry.get(name), str):
                return str(entry[name])
    return None


def _label(text: str) -> str:
    return re.sub(r"[_./:\-]+", " ", text)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        return [s for item in value.values() for s in _strings(item)]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    return []


# -- one mask on one world -------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MaskedWorld:
    """A world with a mask applied and checked: its pack and the mask its header carries, or why it was refused."""

    pack: EnvPack | None
    mask: ExposureMask | None
    refused: str | None = None


def masked_world(pack: EnvPack, before: Served, design: ExposureDesign, withheld: Sequence[str]) -> MaskedWorld:
    """The world `pack` (whose served counts are `before`) with `withheld` cut, when the mask is honest there: each
    withheld type was served before and no product serves it after. Otherwise no pack, and the reason."""
    types = ordered(withheld)
    missing = [t for t in types if not before.shows(t)]
    if missing:
        return MaskedWorld(None, None, f"{', '.join(missing)} not served in this world, so nothing to withhold")
    cut = withhold(pack, types)
    after = before.updated(served_types(cut.pack, only=cut.changed)) if types else before
    left = [f"{t} on {after.counts[t]} products" for t in types if after.shows(t)]
    if left:
        return MaskedWorld(None, None, f"the cut leaves {', '.join(left)}")
    served = [t.id for t in TYPES if t.id not in types and after.shows(t.id)]
    absent = [t.id for t in TYPES if not before.shows(t.id)]
    counts = {
        t: ExposureCount(before=before.counts.get(t, 0), after=after.counts.get(t, 0))
        for t in ordered([*types, *served])
    }
    mask = ExposureMask(
        design=design,
        mask=mask_id(types),
        withheld=types,
        served=served,
        absent=absent,
        types=TYPES_VERSION,
        products=before.products,
        counts=counts,
        collateral=[t for t in served if after.counts[t] < before.counts[t]],
    )
    return MaskedWorld(cut.pack, mask)


def apply_mask(pack: EnvPack, mask: ExposureMask) -> EnvPack:
    """The pack a world's recipe names: its base pack with the mask's withheld types cut (the same pack for a
    baseline mask). The planner checked the mask on this very cut, so a rebuild gives the same pack id."""
    if mask.types != TYPES_VERSION:
        raise ValidationError(
            f"mask {mask.mask} was planned on type table {mask.types}, not {TYPES_VERSION}",
            hint="build a new manifest: the table that cut the planned world has changed",
        )
    return withhold(pack, mask.withheld).pack


def mask_onto(masked: MaskedWorld, base: EnvPack, onto: EnvPack) -> EnvPack:
    """`onto` under `masked`'s mask, where `masked` is the cut made on `base` (D72). A payment rail's world (D66)
    serves its platform's world over other protocols: when `onto` holds the same products and market as `base`, the
    cut already made is carried over instead of cut again, and the pack is the one `apply_mask(onto, ...)` gives,
    byte for byte (the provenance names `onto` as the base). Otherwise `onto` is cut on its own."""
    if masked.pack is None or masked.mask is None:
        raise ValidationError(f"mask refused on {base.id}: {masked.refused}", hint="place only a mask that was honest")
    if not masked.mask.withheld:
        return apply_mask(onto, masked.mask)
    if not onto.same_goods(base):
        return apply_mask(onto, masked.mask)
    doc, cut = onto.doc, masked.pack.doc
    doc["snapshot"], doc["market"] = cut["snapshot"], cut["market"]
    provenance = dict(doc.get("provenance") or {})
    provenance[CONDITION] = {**cut["provenance"][CONDITION], "base": onto.id}
    doc["provenance"] = provenance
    return EnvPack.from_bytes(dumps(doc).encode("utf-8"))


def wire_types(payload: Any, types: Sequence[str]) -> Counter[str]:
    """Per type, how many product documents in one reply (a search, a lookup or a get_product answer, at any depth)
    carry it, with the policies served at the reply's root: the proof, read on a trace's wire, that a withheld type
    never reached it."""
    specs = [data_type(t) for t in ordered(types)]
    policies = _root_policies(payload)
    docs = list(_product_docs(payload))
    counts: Counter[str] = Counter()
    for doc in docs:
        counts.update(carried(doc, policies, specs))
    if not docs:
        counts.update(s.id for s in specs if any(_policy_carries(p, s) for p in policies))
    return counts


def _root_policies(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        found = payload.get("policies")
        if isinstance(found, list):
            return [p for p in found if isinstance(p, Mapping)]
        return [p for item in payload.values() for p in _root_policies(item)]
    if isinstance(payload, list):
        return [p for item in payload for p in _root_policies(item)]
    return []


def _product_docs(payload: Any) -> Iterable[Mapping[str, Any]]:
    """Every mapping that reads as a product: an id beside a title and variants."""
    if isinstance(payload, Mapping):
        if payload.get("id") is not None and "title" in payload and isinstance(payload.get("variants"), list):
            yield payload
            return
        for item in payload.values():
            yield from _product_docs(item)
    elif isinstance(payload, list):
        for item in payload:
            yield from _product_docs(item)

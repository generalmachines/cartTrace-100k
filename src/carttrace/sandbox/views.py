"""Recorded agent views (D100): the product documents live stores sent agents, as data a stand-in renders in.

Why. A stand-in (an open twin, D85 and D91) serves open products through the protocol and its platform pack, and its
catalog documents were the sandbox's own: a plain-text description, a merchant category path, and a `metadata`
object no live Shopify document carries, with no handle, url or compare-at range. Every store with a recorded agent
view is Shopify over UCP MCP (176 stores, 77,640 product documents), and they are nearly one template
(docs/research/2026-10-16-open-fidelity.md). A stand-in of such a store now serves its open products in that store's
recorded shape, so the agent reads what it read live.

The template is data, data/sandbox/views/shopify-ucp-mcp.yaml (written by tools/agent_views.py from the live
bundles, structure only), so it sits beside the platform packs as data. It holds:
- `views`: the dominant search_catalog (listing), get_product (detail) and lookup_catalog (lookup) shapes, in the
  catalog's shape keys (carttrace.sandbox.surfaces.ucp.catalog SHAPE_KEYS);
- `stores`: every store with a recorded view, the views it recorded and where its own shape differs;
- `format`: how values are written (the description's html member carries text without tags, as every live
  document's does; variants repeat the product's description; availability is `available` alone; an empty
  compare-at range is zero; a product without options shows Shopify's Title option);
- `unfilled`: the paths the open data cannot fill (media without a licensed image, SKUs, compare-at prices, alt
  text), left absent as a real store leaves them;
- `recorded`: a recorded document's paths and types per view, the reference the tests hold a rendering to.

The view is served only under the Shopify pack, and there the world's products and variants carry Shopify-shaped
ids (the world's presented ids, D106, carttrace.sandbox.presented) and Shopify's variant titles, so carts and tasks name
items as the catalog does. Under any other pack, and for a store without a recorded view, a stand-in keeps its platform
pack's documents and the open ids. A view is resolved per store: its
own shape where it recorded the view, else the dominant one. Nothing here invents a value: a field the open record
does not state is left out.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from carttrace.core.canon import digest
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.sandbox.envpack.doc import AgentViewDoc, EnvPackDoc
from carttrace.sandbox.envpack.pack import EnvPack

SCHEMA: Final = "mc.agent-view.v1"
TEMPLATE_FILE: Final = "shopify-ucp-mcp.yaml"
TAXONOMY_FILE: Final = "shopify-taxonomy.json"
VIEWS: Final = ("listing", "detail", "lookup")
VIEW_PACK: Final = "shopify"
"""The only platform pack a recorded Shopify view is served under: under any other the world keeps that pack's own
documents, since Shopify documents beside another platform's search and checkout are a mix no real store sends."""
ViewName = Literal["listing", "detail", "lookup"]
DETAIL_LABELS: Final = {
    "size": "Size",
    "size_system": "Size system",
    "material": "Material",
    "care": "Care",
    "ingredients": "Ingredients",
    "allergens": "Allergens",
    "warnings": "Warnings",
    "dosage": "Dosage",
    "compatibility": "Compatibility",
    "dimensions": "Dimensions",
    "weight": "Weight",
    "pack_size": "Pack size",
    "capacity": "Capacity",
    "included": "What's included",
    "features": "Features",
    "origin": "Origin",
    "scent": "Scent",
    "nutrition": "Nutrition",
    "directions": "Directions",
    "certifications": "Certifications",
    "warranty": "Warranty",
}
"""How a stated detail reads when it is folded into the description, as live stores write "Label: value"."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class DefaultOption(_Frozen):
    name: str
    label: str


class ViewFormat(_Frozen):
    description: Literal["text"]
    """`text`: the description's html member holds the text without markup, as every recorded document's does (no
    tag in 77,640), lines run together."""
    variant_description: Literal["product"]
    """Each variant repeats the product's description (238,807 of 238,807 recorded variants)."""
    availability_fields: list[str]
    list_price_range: Literal["zero", "price_range"]
    """What a product with no compare-at price carries: zero amounts (live Shopify's), or its price range."""
    url: str
    """With {domain} and {handle}."""
    default_option: DefaultOption | None = None
    """The option a product without options shows (Shopify's Title / Default Title)."""
    category_taxonomy: str
    category_prefix: str
    fold: list[Literal["details", "attributes"]] = Field(default_factory=list)
    """Open facts written into the description as "Label: value." sentences, since the view has no field for them."""
    media_sources: list[str] = Field(default_factory=list)
    """Open sources whose image URLs may be served (licensed for it and given as URLs); others serve no media."""


class Recorded(_Frozen):
    from_: str = Field(alias="from")
    documents: int
    paths: list[str]


class StoreRow(_Frozen):
    bundles: int
    recorded: dict[ViewName, int]
    """The views the store's live bundles hold documents of, with how many."""
    differs: dict[ViewName, dict[str, Any]] = Field(default_factory=dict)
    """Shape keys where the store's own view differs from the dominant one (None: the store's view lacks the key)."""
    format: dict[str, Any] = Field(default_factory=dict)


class ViewTemplate(_Frozen):
    schema_: Literal["mc.agent-view.v1"] = Field(alias="schema")
    id: str
    title: str
    platform: str
    surface: str
    source: dict[str, Any]
    format: ViewFormat
    views: dict[ViewName, dict[str, Any]]
    unfilled: list[str]
    recorded: dict[ViewName, Recorded]
    stores: dict[str, StoreRow]

    def for_store(self, store_id: str) -> ResolvedView | None:
        """The store's view: its own shape for each view it recorded, the dominant one elsewhere; None for a store
        with no recorded view."""
        row = self.stores.get(store_id)
        if row is None:
            return None
        views: dict[str, dict[str, Any]] = {}
        for name in VIEWS:
            shape = dict(self.views[name])
            if name in row.recorded:
                for key, value in row.differs.get(name, {}).items():
                    if value is None:
                        shape.pop(key, None)
                    else:
                        shape[key] = value
            views[name] = shape
        fmt = {**self.format.model_dump(mode="json"), **row.format}
        ViewFormat.model_validate(fmt)
        own = [n for n in VIEWS if n in row.recorded]
        sealed = digest({"template": self.id, "views": views, "format": fmt}, 8)
        return ResolvedView(template=self.id, store=store_id, own=tuple(own), views=views, format=fmt, digest=sealed)


@dataclass(frozen=True, slots=True)
class ResolvedView:
    template: str
    store: str
    own: tuple[str, ...]
    views: Mapping[str, Mapping[str, Any]]
    format: Mapping[str, Any]
    digest: str

    @property
    def ref(self) -> str:
        return f"{self.template}:{self.store}@{self.digest}"


@dataclass(frozen=True, slots=True)
class AgentViews:
    """The recorded views of one data directory, with the taxonomy their categories are mapped onto."""

    template: ViewTemplate
    taxonomy: Mapping[str, str]
    """Shopify taxonomy id (`hb-3-2-9`) to its name, for the subtrees our categories name."""
    roots: Mapping[str, tuple[str, ...]]
    """Our category slug to its Shopify taxonomy ids (data/categories/<slug>.yaml), in file order."""

    def view_of(self, store_id: str) -> ResolvedView | None:
        return self.template.for_store(store_id)

    def resolve(self, ref: str) -> ResolvedView:
        """The view a world names. Raises ValidationError when the template no longer gives the store that view."""
        name, _, rest = ref.partition(":")
        store, _, sealed = rest.partition("@")
        found = self.view_of(store) if name == self.template.id else None
        if found is None or found.digest != sealed:
            now = found.ref if found is not None else "none"
            raise ValidationError(
                f"agent view {ref} is not the one data/sandbox/views gives now ({now})",
                hint="the template was regenerated since the manifest was built; build a new manifest",
            )
        return found

    def apply(self, pack: EnvPack, ref: str, category: str) -> EnvPack:
        """`pack` served in the view `ref` names: the view, each product's taxonomy category, and variant titles as
        Shopify writes them (Default Title for a product without options, else its option values joined by " / "),
        so catalog, cart and checkout name every item one way. The ids are the world's presented ones (D106,
        carttrace.sandbox.presented: Shopify gids under the Shopify pack), applied before the view."""
        view = self.resolve(ref)
        roots = self.roots.get(category, ())
        doc = pack.doc
        snapshot = doc["snapshot"]
        categories: dict[str, str] = {}
        for pid, facts in sorted(snapshot["products"].items()):
            found = taxonomy_category(roots, self.taxonomy, facts.get("product_type"), _merchant_path(facts))
            if found:
                categories[pid] = found
        for variant in snapshot["variants"].values():
            options = variant.get("options") or {}
            variant["title"] = " / ".join(str(v) for v in options.values()) if options else DEFAULT_VARIANT_TITLE
        shown = doc["policy"].get("presented") or {}
        agent = AgentViewDoc(
            ref=view.ref,
            template=view.template,
            store=view.store,
            own=list(view.own),  # type: ignore[arg-type]
            views={k: dict(v) for k, v in view.views.items()},
            format=dict(view.format),
            categories=categories,
            ids=dict(shown.get("ids") or {}),
        )
        doc["policy"] = {**doc["policy"], "agent_view": agent.model_dump(mode="json")}
        return EnvPack(EnvPackDoc.model_validate(doc))


DEFAULT_VARIANT_TITLE: Final = "Default Title"
"""Shopify's title for the one variant of a product without options; a cart line names the product's title."""


def load_views(directory: Path, categories_dir: Path) -> AgentViews | None:
    """The views under `directory`, or None when it holds no template (views off)."""
    path = directory / TEMPLATE_FILE
    if not path.is_file():
        return None
    return _load(path, path.stat().st_mtime_ns, directory / TAXONOMY_FILE, categories_dir)


@lru_cache(maxsize=4)
def _load(path: Path, _mtime: int, taxonomy_path: Path, categories_dir: Path) -> AgentViews:
    try:
        template = ViewTemplate.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValueError) as exc:
        raise ValidationError(f"{path} is not an {SCHEMA} template: {exc}") from None
    if not taxonomy_path.is_file():
        raise NotFoundError(f"{taxonomy_path} is missing", hint="run tools/agent_views.py taxonomy")
    table = json.loads(taxonomy_path.read_text(encoding="utf-8"))["categories"]
    roots: dict[str, tuple[str, ...]] = {}
    for spec_path in sorted(categories_dir.glob("[a-z]*.yaml")):
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        paths = (spec.get("shopify_taxonomy") or {}).get("paths") or ()
        roots[spec_path.stem] = tuple(str(p["id"]) for p in paths if isinstance(p, Mapping) and p.get("id"))
    return AgentViews(template=template, taxonomy=table, roots=roots)


# -- values ------------------------------------------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset({"and", "for", "the", "of", "with", "other", "a", "an", "in", "accessories", "supplies", "products"})


def _words(text: object) -> set[str]:
    out = set()
    for w in _WORD.findall(str(text or "").lower()):
        if w in _STOP:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def taxonomy_category(roots: Sequence[str], table: Mapping[str, str], product_type: object, path: object) -> str | None:
    """The Shopify taxonomy id for a product of our category (`roots`, its taxonomy ids): the category under those
    roots whose name shares most words with the product type (three each) and its merchant path (one each), the
    fewest unmatched words, then the shallowest and the first id; the first root when no name shares a word. A
    best effort, like data/categories' own mapping; None for a category that names no taxonomy id."""
    if not roots:
        return None
    kind, trail = _words(product_type), _words(path)
    best: tuple[int, int, int, str] | None = None
    for node, name in table.items():
        if not any(node == r or node.startswith(r + "-") for r in roots):
            continue
        words = _words(name)
        score = 3 * len(kind & words) + len(trail & words)
        if score == 0:
            continue
        key = (-score, len(words - kind - trail), node.count("-"), node)
        if best is None or key < best:
            best = key
    return best[3] if best is not None else roots[0]


def document_paths(value: Any, prefix: str = "") -> set[str]:
    """Every path of a document with its JSON type (`variants[].price.amount:int`), list items read one by one: the
    structure a recorded document and a rendering are compared by."""
    out: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.add(f"{path}:{_json_type(item)}")
            out |= document_paths(item, path)
    elif isinstance(value, list):
        for item in value[:50]:
            out |= document_paths(item, prefix + "[]")
    return out


def _json_type(value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    names = {int: "int", float: "number", str: "str", list: "list", dict: "object"}
    for kind, name in names.items():
        if isinstance(value, kind):
            return name
    return "object" if isinstance(value, Mapping) else "null"


def _merchant_path(facts: Mapping[str, Any]) -> str:
    for c in facts.get("categories") or ():
        if isinstance(c, Mapping) and c.get("taxonomy") == "merchant":
            return str(c.get("value") or "")
    return ""


def handle_for(title: str, product_id: str) -> str:
    """A product's handle on the stand-in: its title's slug (at most 60 characters) and its id's last segment, so two
    products of one title never share a url."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60].strip("-")
    tail = re.sub(r"[^a-z0-9]+", "-", product_id.rsplit("/", 1)[-1].lower()).strip("-")
    return f"{slug}-{tail}" if slug and tail else slug or tail or "product"


def description_text(description: str, facts: Mapping[str, Any], fold: Sequence[str]) -> str:
    """The description as the view's html member carries it (`text`): the open description's lines run together,
    then each folded fact as a "Label: value." sentence. Empty when the record states nothing."""
    parts = [" ".join(line.split()) for line in description.splitlines()]
    out = " ".join(p for p in parts if p)
    metadata = facts.get("metadata") if isinstance(facts.get("metadata"), Mapping) else {}
    facts_said: list[str] = []
    for kind in fold:
        stated = metadata.get(kind) if isinstance(metadata, Mapping) else None
        if not isinstance(stated, Mapping):
            continue
        order = list(DETAIL_LABELS) if kind == "details" else []
        ranked = sorted(stated.items(), key=lambda kv: (order.index(kv[0]) if kv[0] in order else len(order), kv[0]))
        for key, value in ranked:
            text = " ".join(str(value).split())
            if not text:
                continue
            label = DETAIL_LABELS.get(str(key), str(key)) if kind == "details" else str(key)
            facts_said.append(f"{label}: {text.rstrip('.')}.")
    if out and facts_said and out[-1] not in ".!?":
        out += "."
    return " ".join(p for p in (out, *facts_said) if p)

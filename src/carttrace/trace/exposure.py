"""The store exposure sheet, `mc.exposure.v1`: what a sandbox store exposes, per data type and per surface, written into
every sandbox bundle as exposure.json (D93).

Why a sheet.The header's `exposure` block (D63) names the 36 types
an exposure mask may cut, counted over the served products, on exposure rows only. A judge needs more: every type of
the category's vocabulary (the data-type union and the corpus, D62 and D76), on every surface of the episode, with
the tools and fields that carried it, over the task's own candidates and witnesses, and who removed it when it is
missing. So each type, on each surface, has one status:
- `exposed`: some product (or, for a store-level type, the surface's checkout or policy reply) carries it there;
- `withheld`: the store's world has it and we removed it on purpose (`by_us` true): an exposure mask (D63) or the
  detail_hidden condition (D47). Designed; never the store's fault, never the agent's for not finding it;
- `not_served`: the store's world has it and this surface does not carry it (`by_us` false): the platform's product
  view (D55) or the surface's own document shape drops it, as the real store's platform does. A store fault;
- `absent`: the store's world does not have it at all. Not applicable, or a store fault if the task needs it.

`exposed` with the agent never reading it is the agent's miss; the sheet says where it was (tools and fields).

Checkout is read the same way (`checkout.features`): payment handlers, wallets accepted, shipping and pickup, policy
links, discount codes and gift cards, each with whether the world offers it, whether the platform pack supports it and
which surfaces expose it.

The sheet is written by carttrace.collect (carttrace.collect.exposure_sheet) from the world and the surfaces' own renderers, never
by a model, and its bytes are a function of the world, the task, the surfaces and the vocabulary. `load(bundle)` reads
it back typed; a bundle written before D93, or a live one, has none.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from carttrace.trace.layout import EXPOSURE_FILE, parse_json_as

if TYPE_CHECKING:
    from carttrace.trace.reader import Bundle

SCHEMA = "mc.exposure.v1"

Status = Literal["exposed", "withheld", "not_served", "absent"]
"""What one surface did with one type: carried it, had it removed by our design, did not carry what the world has, or
had nothing to carry because the world has none."""
Cause = Literal["mask", "detail_hidden", "platform_view", "surface"]
"""Why a type the world has is not exposed on a surface: our exposure mask (D63), our detail_hidden condition (D47),
the platform's product view (D55), or the surface's own shape (it renders no field that carries it)."""
Scope = Literal["product", "store"]
Role = Literal["candidate", "witness"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Coverage(_Frozen):
    """How many products carry a type: of all the world's products, of the task's candidates, of its witnesses."""

    products: int = Field(default=0, ge=0)
    candidates: int = Field(default=0, ge=0)
    witnesses: int = Field(default=0, ge=0)


class SurfaceCell(_Frozen):
    """One type on one surface."""

    status: Status
    by_us: bool = False
    """True only when we removed it on purpose (status `withheld`)."""
    cause: Cause | None = None
    """Why the world's type is not exposed here; None when it is absent, or exposed whole. On an `exposed` cell,
    `detail_hidden` says our cut removed it from some products (`withheld_from`) and others still serve it, and
    `mask` that the mask's cut thinned it (a collateral type of the mask, D63: text that stated a withheld type
    stated this one too)."""
    coverage: Coverage | None = None
    """Products this surface serves it on (product-level types only)."""
    withheld_from: Coverage | None = None
    """Products our design removed it from on this surface: the world's products under a mask, the products the
    detail_hidden cut took it from. Read `withheld_from.witnesses` for whether the task's own products lost it."""
    tools: list[str] = Field(default_factory=list)
    """The surface's tools (or REST operations) whose output carries it: `search_catalog`, `get_product`, ..."""
    fields: list[str] = Field(default_factory=list)
    """Where in their documents: field paths in the union's key scheme (`metadata.material`, `variants[].list_price`,
    `text` for a line of the description, `policies[]`), at most FIELDS of them."""


class LiveSeen(_Frozen):
    """What the live store itself showed, from the category union (D62): the share of products read that showed the
    type on its website (store truth) and to agents (agent view); None where that layer was not read or showed none."""

    website: float | None = None
    agent_view: float | None = None


class Withholding(_Frozen):
    """What our design removed: an exposure mask's id, or the detail_hidden condition's details."""

    by: Literal["mask", "detail_hidden"]
    mask: str | None = None
    details: list[str] = Field(default_factory=list)


class TypeRow(_Frozen):
    """One data type of the vocabulary that the store's world, a surface or the live store shows."""

    id: str
    """The union's stable id (data/bench/union.yaml curated ids, else the union's id scheme, D62)."""
    name: str
    group: str
    scope: Scope
    origin: Literal["curated", "auto"]
    element: str | None = None
    """The data element (data/elements, the live data audit's id) this type is, where one maps."""
    world: Coverage
    """In the store's own world: the pack's products as captured, before any platform view, mask or condition."""
    live: LiveSeen | None = None
    withheld: Withholding | None = None
    surfaces: dict[str, SurfaceCell] = Field(default_factory=dict)
    """Keyed by surface kind (`ucp_mcp`, `ucp_rest`, `webmcp`, `acp_rest`, ...), every surface of the episode."""
    layer: Literal["base", "core", "experimental"] | None = None
    """An open stand-in's union layer of the type (D117): `base`, the catalog's own facts; `core`, the category's
    union core the sandbox filled; `experimental`, a core type a later experiment may withhold per row. None on a
    world built before the core, and left out of the file then."""
    value_origins: dict[str, int] = Field(default_factory=dict)
    """Of the world's products carrying the type, how many hold a value of each origin (D117): `catalog` (a base
    field), `derived` (a rule over the product's own text), `synthetic` (seeded, made to agree with the text) and
    `store` (the store's own terms). Left out when empty."""

    @model_serializer(mode="wrap")
    def _without_empty_layer(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if data.get("layer") is None:
            data.pop("layer", None)
        if not data.get("value_origins"):
            data.pop("value_origins", None)
        return data


class VocabularySource(_Frozen):
    name: str
    """`union.yaml`, `corpus/<category>.json`, `union/<category>.json`, `exposure-types`, `elements`."""
    version: str
    """The source's own version, or the sha256 (first 16) of its bytes."""
    types: int = Field(ge=0)


class Vocabulary(_Frozen):
    """The category's data types the sheet reads, and where they came from."""

    id: str
    """xv-<sha12> over the sources' versions and the type ids: the same vocabulary, the same id."""
    category: str | None
    types: int = Field(ge=0)
    sources: list[VocabularySource] = Field(default_factory=list)


class SurfaceInfo(_Frozen):
    """One surface of the episode and the tools through which it serves products."""

    kind: str
    version: str | None = None
    product_tools: list[str] = Field(default_factory=list)
    """Tools whose output carries product documents on this surface."""
    catalog_from: str | None = None
    """The surface whose catalog this one serves when it has none of its own (an ACP row reads UCP's)."""


class TaskProduct(_Frozen):
    """One product the task is about, and the types it carries in the world and on each surface."""

    id: str
    roles: list[Role]
    world: list[str] = Field(default_factory=list)
    surfaces: dict[str, list[str]] = Field(default_factory=dict)


class Products(_Frozen):
    all: int = Field(ge=0)
    candidates: list[str] = Field(default_factory=list)
    witnesses: list[str] = Field(default_factory=list)


class Feature(_Frozen):
    """One thing checkout may expose: whether the world offers it, whether the platform pack supports it, and on which
    surfaces an agent can reach it."""

    id: str
    """`payment.handlers`, `payment.wallets`, `fulfilment.shipping`, `fulfilment.pickup`, `policy.links`,
    `discount.codes`, `gift_card.redeem`, `gift.message`, `gift_card.products`."""
    offered: bool
    """The world offers it (the store declared it, or the sandbox serves it for this world)."""
    platform_supports: bool | None = None
    """The platform pack can serve it at all; None where no platform rule speaks to it."""
    surfaces: dict[str, bool] = Field(default_factory=dict)
    """Per surface kind of the episode: reachable there."""
    values: list[str] = Field(default_factory=list)
    """What it holds: handler namespaces, instruments, option ids, link types, codes."""
    by_us: bool = False
    """True when the sandbox, not the store, decided it (an adopted rail's handler, a designed fault)."""
    note: str | None = None


class Checkout(_Frozen):
    rail: str | None = None
    features: list[Feature] = Field(default_factory=list)

    def feature(self, feature_id: str) -> Feature | None:
        return next((f for f in self.features if f.id == feature_id), None)


class ViewUsed(_Frozen):
    """The recorded agent view a stand-in's catalog documents were served in (D100, data/sandbox/views)."""

    ref: str
    """`<template>:<store>@<digest>`, as experiment.json names it."""
    template: str
    store: str
    """The store whose recorded view it is: the one the stand-in stands in for."""
    own: list[str] = Field(default_factory=list)
    """The views (listing, detail, lookup) in the store's own recorded shape; the rest are the template's dominant."""
    source_ids: dict[str, str] = Field(default_factory=dict)
    """Each Shopify-shaped product and variant id the world served to the open source's own id it stands for."""


class ExposureSheet(_Frozen):
    schema_: Literal["mc.exposure.v1"] = Field(default="mc.exposure.v1", alias="schema")
    episode_id: str
    store_id: str
    category: str | None = None
    envpack: str | None = None
    """The pack the episode ran on (masked when a mask applies)."""
    world: str | None = None
    """The store's own world the `world` columns read: the base pack id before any mask."""
    platform_pack: str | None = None
    agent_view: ViewUsed | None = None
    """The recorded agent view the UCP documents were served in (D100); None when they were the platform pack's.
    Where it applies, the types the view has no field for read `not_served` with cause `platform_view`."""
    presented_ids: dict[str, str] = Field(default_factory=dict)
    """Each product and variant id the stand-in presented (D106, its pack's id scheme) to the open source's own id;
    left out of the file when the world presents its source ids (every world but a stand-in)."""
    mask: str | None = None
    detail_hidden: list[str] = Field(default_factory=list)
    vocabulary: Vocabulary
    surfaces: list[SurfaceInfo] = Field(default_factory=list)
    products: Products
    types: list[TypeRow] = Field(default_factory=list)
    """Every vocabulary type something shows, by id."""
    absent: list[str] = Field(default_factory=list)
    """Vocabulary ids nothing shows: not in the world, on no surface, never seen on the live store."""
    task_products: list[TaskProduct] = Field(default_factory=list)
    checkout: Checkout = Field(default_factory=Checkout)

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    @model_serializer(mode="wrap")
    def _without_empty_ids(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if not data.get("presented_ids"):
            data.pop("presented_ids", None)
        return data

    def type(self, type_id: str) -> TypeRow | None:
        """One type's row, or None when the vocabulary has it as absent or not at all."""
        return next((t for t in self.types if t.id == type_id), None)

    def status(self, type_id: str, surface: str) -> Status | None:
        """The type's status on one surface: `absent` for a vocabulary id nothing shows, None for an id outside the
        vocabulary or a surface the episode did not mount."""
        row = self.type(type_id)
        if row is None:
            return "absent" if type_id in self.absent else None
        cell = row.surfaces.get(surface)
        return None if cell is None else cell.status


def load(bundle: Bundle | Path | str) -> ExposureSheet | None:
    """The bundle's exposure sheet, typed; None when it has none (live, or written before D93). `bundle` is an opened
    Bundle, a bundle directory or the exposure.json path itself."""
    if isinstance(bundle, Path | str):
        path = Path(bundle)
        if path.is_dir():
            path = path / EXPOSURE_FILE
        if not path.is_file():
            return None
        return parse_json_as(ExposureSheet, path.read_bytes(), where=str(path))
    return bundle.exposure()


__all__ = [
    "SCHEMA",
    "Cause",
    "Checkout",
    "Coverage",
    "ExposureSheet",
    "Feature",
    "LiveSeen",
    "Products",
    "Role",
    "Scope",
    "Status",
    "SurfaceCell",
    "SurfaceInfo",
    "TaskProduct",
    "TypeRow",
    "Vocabulary",
    "VocabularySource",
    "Withholding",
    "load",
]

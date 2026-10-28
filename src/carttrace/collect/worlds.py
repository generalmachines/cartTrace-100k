"""Sandbox worlds: the environment pack a sandbox or reference row runs in, made from a catalog and a store's profile.

Why a world per store, built when the manifest is. A sandbox row is a queue store's twin: a catalog store from the
same category (carttrace.sandbox.catalogs cuts it, seeded by the store, so twins in one category hold different products)
speaking the queue store's protocol profile and delivery market. The bundle header names the queue store it stands
in for; the pack keeps the catalog store's own identity, under a .invalid domain, because a sandbox world never
borrows a real host. The manifest names each world by its envpack id and keeps the recipe beside it, and the tasks
are written around the world's own products, so the tasks and the world cannot drift apart.

At run time the pack is loaded from `var/envpacks/<id>.json`, or rebuilt from its recipe when the file is missing,
and either way it must be the pack the manifest names: a catalog that changed on disk gives a different id, and the
row fails loudly instead of running in another world. The catalog's provenance (source, licence, rights) stays in the
pack under its own keys; the collector adds the twin's, never over them.

A catalog snapshot that is a replica capture of the store (D12) may carry the store's own protocol profile; the
manifest asks `captured` for it before a world is built, so the replica speaks the store's versions
(carttrace.collect.profiles). A licensed catalog carries none.

Restricted catalogs (Lazada, internal research only) build worlds only when `sandbox.allow_restricted_catalogs` is
on; their bundles are never trainable or publishable (`carttrace.collect.rights`).

Replicas (decision D12). A plan whose catalog is `replica` gives a store that has a capture (`carttrace live capture`) a
world built from that capture: the store's own products and answers (carttrace.sandbox.catalogs.build_replica), under the
store's real host, speaking the profile the store was captured at. The recipe names the capture (`capture`), so the
world is rebuilt from it when its pack is missing, and its episodes carry the capture's rights (a live store's: never
training, never publication, raw content expiring), read from the pack (`replica_rights`). A store with no capture
falls back to a twin from the configured default catalog, and the manifest notes it.

Capture v2. A store whose newest capture is a capture
pass edition (<paths.var>/store/captures/<edition>/<store>/, carttrace.trace.capture_v2) gets its replica from that instead:
the world names the capture's `cv2-` id, and the replica carries no profile of its own, so the store's preflight
profile (else the preset) is the one it speaks. The newer of the two kinds of capture wins.

Sold out (D52). A replica world names the rate its store answered "sold out" at live in the world's market
(`World.sold_out`, from `carttrace sandbox replica sold-out`), and its pack carries it; each episode draws its own sold-out
variants from it as it opens (`sold_out_world`), sparing a witness so the task can still be bought.

Exposure (D63). A replica world of an exposure design names its mask (`World.exposure`): the data types it withholds
are cut from the pack once it is served under its platform pack (carttrace.collect.exposure), so every surface the world
mounts serves the store without them, and its sold-out rate and platform rules hold as on any replica world.

Store replies (D70). A replica world names its store's replies record (`World.store_replies`, from `carttrace sandbox replica
store-replies`): its pack serves the payment handlers the store's live cart replies declared, in place of the
sandbox's fake one, and names the rates of the store's own errors; each episode draws the calls that fail as it
opens (`store_errors_world`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer
from pydantic import ValidationError as PydanticValidationError

from carttrace.collect.bindings import Bindings
from carttrace.collect.exposure import CONDITION as EXPOSURE_PROVENANCE
from carttrace.collect.exposure import apply_mask
from carttrace.collect.ports import BuildStoreSnapshot, CatalogRestricted, Fixtures, PoolMarket, ReplicaSnapshot
from carttrace.collect.profiles import CapturedStore, captured_store
from carttrace.collect.union_core import PROVENANCE as UNION_CORE
from carttrace.collect.union_core import CoreSpec, fill_products, load_core, spec_of, store_policies
from carttrace.config.schema import Settings
from carttrace.core.errors import NotFoundError, SafetyError, ValidationError
from carttrace.protocols.profile import ProtocolProfile
from carttrace.sandbox import EnvPack, UnknownMarket, build_envpack, market
from carttrace.sandbox.catalogs.agent_surface import SURFACE_DIR, load_surface, newest_surface, with_agent_surface
from carttrace.sandbox.catalogs.pools import OPEN
from carttrace.sandbox.catalogs.search_memory import MEMORY_DIR, load_memory, newest_memory, with_search_memory
from carttrace.sandbox.catalogs.snapshot import StoreSnapshot
from carttrace.sandbox.catalogs.sold_out import SOLD_OUT_DIR, Draw, draw_sold_out, newest_record, rate_for, with_sold_out
from carttrace.sandbox.catalogs.store_replies import (
    REPLIES_DIR,
    StoreRepliesRecord,
    draw_store_errors,
    replies_ref,
    with_store_replies,
)
from carttrace.sandbox.catalogs.store_replies import load_record as load_replies
from carttrace.sandbox.catalogs.store_replies import newest_record as newest_replies
from carttrace.sandbox.envpack import Snapshot
from carttrace.sandbox.faults import opening_knobs
from carttrace.sandbox.platforms import platform_named, with_platform_pack
from carttrace.sandbox.presented import stand_in_identity, with_presented
from carttrace.sandbox.surfaces import UnsupportedVersion, dialect_for, mcp_version
from carttrace.sandbox.views import VIEW_PACK, AgentViews, load_views
from carttrace.trace.capture import CAPTURES_DIR, find_capture, latest_capture, load_capture
from carttrace.trace.capture_v2 import ID_PREFIX, captures_root, find_capture_v2, newest_captures_v2, store_catalog
from carttrace.trace.schema import ExposureMask, Rights, SoldOutRate, StoreRef, StoreReplies, TaskRef

DEFAULT_SIZE = 40
"""Products a sandbox store's catalog snapshot asks for."""
ENVPACKS_DIR = "envpacks"
OWN_ARGUMENTS = ("store", "variants", "currency", "products", "snapshot_source")
"""What a catalog snapshot fixes about its store; the collector sets the profile and the market."""
REPLICA = "replica"
"""The catalog name of real stores' replicas (carttrace.sandbox PLUGINS, sandbox.catalog)."""
TASK_FAULT_PREFIX = "task-"
"""Before the id of every fault a task's spec seals into its row's world."""


class World(BaseModel):
    """How to build one sandbox world again: the manifest keeps one per envpack id."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    store_id: str
    """The queue store the world is a twin of."""
    catalog: str
    """Catalog source name (carttrace.sandbox.catalogs), e.g. "abo"."""
    category: str
    """Category slug the snapshot was cut from."""
    snapshot_seed: int
    size: int = Field(ge=1)
    market: str
    """Delivery market code as the queue writes it (US, GB, EU, ...)."""
    profile: str
    """Key of the world's profile in the manifest's profiles table."""
    capture: str | None = None
    """The capture (cap-<sha12>) a replica world is built from; None for a catalog twin. Left out of the manifest
    when None, so manifests made before replicas keep their ids."""
    search_memory: str | None = None
    """The search memory (sm-<sha12>, carttrace.sandbox.catalogs.search_memory) a replica world replays the store's live
    searches from (D39); None for a twin and for a store no live row has searched yet. Left out when None, so
    manifests made before memories keep their ids."""
    agent_surface: str | None = None
    """The agent surface (as-<sha12>, carttrace.sandbox.catalogs.agent_surface) a replica world lists its store's own
    tools from (D40); None for a twin and for a store no live session has listed yet, which lists its platform's.
    Left out when None, so manifests made before surfaces keep their ids."""
    platform_pack: str | None = None
    """The platform pack the world is served under (carttrace.sandbox.platforms.with_platform_pack, D42); None serves its
    own platform. Left out when None, so manifests made before the platform axis keep their ids."""
    sold_out: SoldOutRate | None = None
    """The rate a replica world's cart writes answer sold out at, from the store's sold-out record in the world's
    market (carttrace.sandbox.catalogs.sold_out, D52), with its evidence; None for a twin and for a store with no record.
    Left out when None, so manifests made before it keep their ids."""
    adopted: list[str] = Field(default_factory=list)
    """The protocol families the world serves for a payment rail that its store does not speak (D66: acp, ap2, mpp),
    added to its profile as if the store had adopted them. Left out when empty."""
    exposure: ExposureMask | None = None
    """The exposure mask a replica world is served under (D63, carttrace.collect.exposure): the data types it withholds are
    cut from the pack after its platform pack, and the mask's evidence says they were there and are gone. None for
    every world of a plan without an exposure design. Left out when None, so other manifests keep their ids."""
    store_replies: StoreReplies | None = None
    """What a replica world serves from its store's live replies (D70): the payment handlers, read from the record it
    names, and the rates of the store's own errors. None for a twin and for a store with no live session. Left out
    when None, so manifests made before it keep their ids."""
    pick_from: str | None = None
    """The capture (cv2-<sha12>) of the store an open-pool stand-in's products were matched to (D104,
    carttrace.sandbox.catalogs.matching); None for a spread pick (a store with no capture) and every other world. Left out
    when None, so other manifests keep their ids."""
    agent_view: str | None = None
    """The recorded agent view a stand-in serves its products in (D100, carttrace.sandbox.views): `<template>:<store>@<digest>`
    for a twin of a store whose agent view was recorded live, served under the Shopify pack, which then reads as that
    store read to agents; None for a replica, for a twin of a store with no recorded view, and under any other pack.
    Structure only (field names, nesting and value conventions), never a value the store sent. Left out when None,
    so other manifests keep their ids."""
    home: str | None = None
    """A pooled world's store home market (catalog `open`, D91), which decides the member group its products come
    from: Lazada's for a PH store, else the synthetic-price catalogs', whatever market a cross-border row delivers
    to. None for every other world, and left out, so other manifests keep their ids."""
    union_core: str | None = None
    """The union core an open stand-in's products are filled with (D117, carttrace.collect.union_core): `<rule>@<core
    file id>@<share in bp>`. None for every other world and for a plan built before it, and left out, so other
    manifests keep their ids."""

    @model_serializer(mode="wrap")
    def _without_empty_capture(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        for name in (
            "capture",
            "search_memory",
            "agent_surface",
            "platform_pack",
            "sold_out",
            "exposure",
            "store_replies",
            "pick_from",
            "agent_view",
            "home",
            "union_core",
        ):
            if out.get(name) is None:
                out.pop(name, None)
        if not out.get("adopted"):
            out.pop("adopted", None)
        return out


def rights_key(world: World) -> str:
    """The key of a world's rights in a run's rights table: its catalog for a twin (one licence for every store cut
    from it), its catalog and capture for a replica (each capture carries its own store's rights)."""
    return world.catalog if world.capture is None else f"{world.catalog}:{world.capture}"


def task_faults(task: TaskRef) -> list[dict[str, Any]]:
    """The sealed fault knobs a task's hidden spec says its world carries (a substitute task's stockout of the named
    product), as that world opens with them: each change to the world holds from the first read
    (carttrace.sandbox.faults.opening_knobs), and every id is prefixed `task-` so it cannot meet one of the world's own.
    Empty for a task with none. A spec whose faults are not a list of valid knobs is refused."""
    raw = task.hidden_spec.get("faults")
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValidationError(f"task {task.task_id}: hidden_spec.faults is not a list", hint="regenerate the task")
    return opening_knobs(raw, prefix=TASK_FAULT_PREFIX)


def sealed_world(pack: EnvPack, task: TaskRef) -> tuple[EnvPack, str | None]:
    """The pack a sandbox row runs on: the row's world `pack` with the task's sealed faults joined to its schedule,
    and a note naming both packs and the faults; `pack` itself and None when the task has none.

    Why at open time and not in the manifest: a world is shared by every task written around it, while the faults are
    the task's. The derived pack's id covers the faults (an envpack's id is its schedule's too), the header keeps the
    row's world and the task with its spec, so base pack plus task rebuild it exactly, and the note says which id ran.
    Round 1 of the sandbox ran substitute rows without them: the named variant stayed in stock and was carted, 71 of
    the round's 126 WRONG verdicts."""
    knobs = task_faults(task)
    if not knobs:
        return pack, None
    sealed = pack.derive(faults=[*pack.doc["faults"], *knobs])
    ids = ", ".join(str(k["id"]) for k in knobs)
    return sealed, f"task faults sealed into {sealed.id} (from {pack.id}): {ids}"


def sold_out_world(pack: EnvPack, task: TaskRef, seed: int) -> tuple[EnvPack, Draw | None, str | None]:
    """The pack a replica row runs on once its episode's sold-out variants are drawn (D52): the world's rate drawn by
    the row's seed over every variant for sale, a witness spared when the draw left the task none to buy (unless the
    task is impossible). The pack, the draw and a note naming the derived pack; `pack`, None and None for a world
    that names no rate.

    Why at open time: the world is shared by every task written around it, while the draw is the episode's, and
    the spared witness is the task's. The header names the rate (from the world's recipe), the note the variants."""
    spec = task.hidden_spec
    raw = spec.get("witnesses")
    witnesses = [str(w) for w in raw] if isinstance(raw, list) else []
    derived, draw = draw_sold_out(pack, seed, witnesses=witnesses, impossible=spec.get("impossible") is True)
    if draw is None:
        return pack, None, None
    rate = draw.rate
    spared = f"; spared {', '.join(draw.spared)} so the task keeps a witness for sale" if draw.spared else ""
    note = (
        f"sold out at cart write: {len(draw.variants)} of {draw.candidates} variants for sale, drawn at "
        f"{rate.rate_bp / 100:.2f}% ({rate.source}: {rate.stratum}, {rate.refused} of {rate.writes} live writes "
        f"refused) with seed {seed}{spared}; the catalog still lists them available. Pack {derived.id} from {pack.id}"
    )
    return derived, draw, note


def store_errors_world(pack: EnvPack, seed: int) -> tuple[EnvPack, str | None]:
    """The pack a replica row runs on once its episode's failing calls are drawn from its store's error rates (D70),
    and a note naming them; `pack` and None for a world that names no rate."""
    derived, calls = draw_store_errors(pack, seed)
    if calls is None:
        return pack, None
    failing = ", ".join(f"{c.tool} call {c.call} ({c.kind})" for c in calls) or "none"
    note = f"store errors drawn with seed {seed} at the store's live rates: {failing}. Pack {derived.id} from {pack.id}"
    return derived, note


def replica_rights(pack: EnvPack) -> Rights:
    """The rights a replica world's episodes carry: its capture's, which the pack's provenance holds."""
    raw = (pack.doc.get("provenance") or {}).get("rights")
    if not isinstance(raw, Mapping):
        raise ValidationError(
            f"replica pack {pack.id} carries no rights", hint="rebuild it from its capture with build_replica"
        )
    return Rights.model_validate(raw)


@dataclass(frozen=True, slots=True)
class Replicas:
    """Where captures are and how a replica is built from one, for a plan whose catalog is `replica`."""

    captures_dir: Path
    """<paths.var>/captures."""
    snapshot: ReplicaSnapshot
    twin_catalog: str | None
    """The catalog a store with no capture falls back to (sandbox.default_catalog); None plans it no sandbox rows."""
    v2_dir: Path | None = None
    """<paths.var>/store/captures, the capture pass's editions; None reads old captures only."""
    memory_dir: Path | None = None
    """<paths.var>/live/search-memory, the stores' search memories; None builds replicas without them."""
    surface_dir: Path | None = None
    """<paths.var>/live/agent-surface, the stores' recorded tool lists; None builds replicas without them."""
    sold_out_dir: Path | None = None
    """<paths.var>/live/sold-out, the stores' sold-out records; None builds replicas that never answer sold out."""
    replies_dir: Path | None = None
    """<paths.var>/live/store-replies, the stores' replies records; None builds replicas without them (D70)."""
    partial_min: int = 0
    """Products a capture pass stopped part way (blocked or throttled beside the catalog) must have read for a replica
    to be built from it (carttrace.trace.capture_v2.CaptureV2.partial_usable); 0 builds replicas from complete passes only."""


@dataclass(frozen=True, slots=True)
class CaptureInfo:
    """What the planner reads from a store's capture: its id, how many products it holds, its profile."""

    path: Path
    id: str
    products: int
    profile: ProtocolProfile | None
    """The store's profile as captured; None for a capture v2, which asked no UCP server (the preflight's is used)."""
    listings: int = 0
    """Listing pages the capture's catalog left out (D38); 0 for an old capture, which read no pages."""


@dataclass(frozen=True, slots=True)
class BuiltWorld:
    pack: EnvPack
    fixtures: list[dict[str, Any]]
    """The world's products in UCP's shape, for task generation."""


def configured_worlds(settings: Settings, bindings: Bindings, *, save: bool = True) -> SandboxWorlds:
    """The worlds of a manifest build or a run: packs kept under <paths.var>/envpacks (unless `save` is off, for a
    planner's probe), replicas built from the captures under <paths.var>/captures, and twins cut from the configured
    catalogs (a store with no capture falls back to sandbox.default_catalog)."""
    twin = settings.sandbox.default_catalog
    return SandboxWorlds(
        snapshot=bindings.store_snapshot,
        fixtures=bindings.fixtures,
        restricted=bindings.catalog_restricted,
        allow_restricted=settings.sandbox.allow_restricted_catalogs,
        pool_market=bindings.pool_market,
        views=_views(settings),
        union_core=_union_core(settings),
        directory=settings.paths.var / ENVPACKS_DIR if save else None,
        replicas=Replicas(
            captures_dir=settings.paths.var / CAPTURES_DIR,
            snapshot=bindings.replica_snapshot,
            twin_catalog=None if twin == REPLICA else twin,
            v2_dir=captures_root(settings.paths.var),
            memory_dir=settings.paths.var / MEMORY_DIR,
            surface_dir=settings.paths.var / SURFACE_DIR,
            sold_out_dir=settings.paths.var / SOLD_OUT_DIR,
            replies_dir=settings.paths.var / REPLIES_DIR,
            # D96: a capture stopped at a wall or a throttle after enough products were read still makes a replica
            partial_min=settings.sandbox.v1.partial_min_products,
        ),
    )


def _union_core(settings: Settings) -> CoreSpec | None:
    """The union core open stand-ins are filled with (D117): the frozen core at `sandbox.v1.union_core_share`."""
    share = settings.sandbox.v1.union_core_share
    return CoreSpec(load_core(), share) if share < 1 else None


def presents(world: World) -> bool:
    """Whether the world is a stand-in of the open pools, which presents a neutral identity of its own (D106): its
    `.example` domain, its pack's id scheme, and names that say nothing of the sandbox or the store it stands in for."""
    return world.catalog == OPEN and world.capture is None


def _views(settings: Settings) -> AgentViews | None:
    """The recorded agent views `sandbox.agent_views` names (D100); None when it names none or holds no template."""
    where = settings.sandbox.agent_views
    return None if where is None else load_views(where, settings.paths.data / "categories")


def sandbox_market(code: str) -> str | None:
    """The sandbox market table's code for a delivery market (GB is UK there), or None when it has no row."""
    try:
        return market(code).code
    except UnknownMarket:
        return None


def envpack_arguments(raw: Any, *, world: World) -> dict[str, Any]:
    """`build_envpack` arguments from what build_store_snapshot returned: a catalog StoreSnapshot, through its
    `envpack_kwargs()`, or an envpack Snapshot (or its JSON), which names no store and gets a sandbox one."""
    kwargs_of = getattr(raw, "envpack_kwargs", None)
    if callable(kwargs_of):
        found = kwargs_of()
        if not isinstance(found, Mapping) or not {"store", "variants", "currency"} <= set(found):
            raise ValidationError(
                "a catalog snapshot's envpack_kwargs() lacks store, variants or currency",
                hint="return the arguments build_envpack takes",
            )
        return dict(found)
    snap = as_snapshot(raw)
    return {
        "store": {
            "domain": f"{world.store_id.lower()}.{world.catalog}.sandbox.invalid",
            "store_id": f"{world.catalog}:{world.store_id}",
            "platform": "sandbox",
            "category": world.category,
        },
        "variants": [{"id": vid, **variant.model_dump()} for vid, variant in sorted(snap.variants.items())],
        "currency": snap.currency,
        "products": snap.products,
        "snapshot_source": snap.source,
        "captured_at": snap.captured_at,
    }


def as_snapshot(value: Any) -> Snapshot:
    """An envpack store snapshot from the model, another model, or its JSON."""
    if isinstance(value, Snapshot):
        return value
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    if not isinstance(data, Mapping):
        raise ValidationError(
            f"build_store_snapshot returned a {type(value).__name__}, not a store snapshot",
            hint="return a catalog StoreSnapshot, or an envpack Snapshot",
        )
    try:
        return Snapshot.model_validate(dict(data))
    except PydanticValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        where = ".".join(str(part) for part in first["loc"]) or "snapshot"
        raise ValidationError(
            f"build_store_snapshot returned an invalid snapshot: {where}: {first['msg']}",
            hint="return a catalog StoreSnapshot, or envpack Snapshot fields",
        ) from None


@dataclass(slots=True)
class SandboxWorlds:
    """Builds, saves and loads sandbox worlds. One instance serves a whole manifest build or run."""

    snapshot: BuildStoreSnapshot
    fixtures: Fixtures
    restricted: CatalogRestricted
    allow_restricted: bool
    directory: Path | None = None
    """Where packs are saved and loaded (<paths.var>/envpacks); None keeps them in memory only."""
    replicas: Replicas | None = None
    """How replica worlds are built; None when the plan has no replicas."""
    _arguments: dict[tuple[str, str, str, int, int, str | None, str | None], dict[str, Any]] = field(
        default_factory=dict
    )
    _captured: dict[tuple[str, str, str, int, int, str | None, str | None], CapturedStore | None] = field(
        default_factory=dict
    )
    _picks: dict[tuple[str, str], str | None] = field(default_factory=dict)
    _checked: dict[str, bool] = field(default_factory=dict)
    _replicas: dict[tuple[str, str, str | None, str | None], dict[str, Any]] = field(default_factory=dict)
    _memories: dict[str, str | None] = field(default_factory=dict)
    _surfaces: dict[str, str | None] = field(default_factory=dict)
    _sold_out: dict[tuple[str, str], SoldOutRate | None] = field(default_factory=dict)
    _replies: dict[str, StoreReplies | None] = field(default_factory=dict)
    pool_market: PoolMarket | None = None
    """carttrace.sandbox.catalogs.pool_market, for a plan on the open pools (D91): the market a stand-in store is built in."""
    views: AgentViews | None = None
    """The recorded agent views stand-ins are served in (D100, data/sandbox/views); None serves none."""
    union_core: CoreSpec | None = None
    """The union core open stand-ins are planned with (D117, `sandbox.v1.union_core_share`); None fills none."""

    def union_core_ref(self) -> str | None:
        """What a stand-in's recipe names for its union core, or None."""
        return None if self.union_core is None else self.union_core.ref

    def check_catalog(self, catalog: str) -> bool:
        """Whether the catalog is restricted. Raises SafetyError for a restricted one while they are not allowed.
        Replicas are not a licensed catalog: each capture carries its own store's rights."""
        if catalog == REPLICA:
            return False
        restricted = self._checked.get(catalog)
        if restricted is None:
            restricted = self._checked[catalog] = self.restricted(catalog)
        if restricted and not self.allow_restricted:
            raise SafetyError(
                f"catalog {catalog!r} is restricted to internal research",
                hint="choose another catalog, or set sandbox.allow_restricted_catalogs in config.local.json for "
                "internal research; its bundles are never trained on or published",
            )
        return restricted

    def capture_of(self, store: StoreRef, category: str, *, partial_min: int | None = None) -> CaptureInfo | None:
        """The newest capture with products of a store in a category (filed under its queue id or its host), or
        None when it has none or no replicas are configured. A capture pass stopped part way counts when it read at
        least `partial_min` products (default: the replicas' own `partial_min`; 0 keeps complete passes only): this
        is the sandbox replica path, and a live session never runs on such a capture."""
        if self.replicas is None:
            return None
        host = store.domain.strip().lower()
        ids = [store.store_id, host, host.removeprefix("www.")]
        path = latest_capture(self.replicas.captures_dir, ids, category)
        capture = load_capture(path) if path is not None else None
        if self.replicas.v2_dir is not None:
            least = self.replicas.partial_min if partial_min is None else partial_min
            v2 = newest_captures_v2(
                self.replicas.v2_dir, store_ids=[store.store_id], category=category, partial_min=least
            )
            found = v2.get(store.store_id)
            if found is not None and (capture is None or found.captured_at > capture.captured_at):
                catalog = store_catalog(found, raw=False)
                if catalog.products:
                    return CaptureInfo(
                        path=found.path,
                        id=found.id,
                        products=len(catalog.products),
                        profile=None,
                        listings=catalog.listings,
                    )
        if path is None or capture is None:
            return None
        return CaptureInfo(path=path, id=capture.id, products=len(capture.products), profile=capture.profile)

    def build(self, store: StoreRef, world: World, profile: ProtocolProfile) -> BuiltWorld:
        """The world's pack, cut from its catalog (or built from its capture), served under its platform pack and with
        its exposure mask's withheld types cut, and its products as task fixtures (the same products under every
        platform and mask: tasks are written on the store as it is)."""
        made = self._built(store, world, profile)
        pack = made.pack if world.platform_pack is None else with_platform_pack(made.pack, world.platform_pack)
        if presents(world):
            served = platform_named(pack.platform_name)
            identity = stand_in_identity(
                pack.store.domain,
                world.platform_pack,
                psp=pack.doc["policy"].get("psp"),
                has_server=bool(served.server_info),
            )
            pack = with_presented(pack, identity)
        if world.agent_view is not None:
            if self.views is None:
                raise ValidationError(
                    f"world of {store.store_id} names agent view {world.agent_view}, but no views are configured",
                    hint="build stand-in worlds through configured_worlds, whose views read sandbox.agent_views",
                )
            pack = self.views.apply(pack, world.agent_view, world.category)
        if world.exposure is not None:
            pack = apply_mask(pack, world.exposure)
        return made if pack is made.pack else BuiltWorld(pack=pack, fixtures=made.fixtures)

    def _built(self, store: StoreRef, world: World, profile: ProtocolProfile) -> BuiltWorld:
        if world.capture is not None:
            return self._replica(store, world, profile)
        self.check_catalog(world.catalog)
        code = sandbox_market(world.market)
        if code is None:
            raise ValidationError(
                f"the sandbox has no market {world.market!r} for store {store.store_id}",
                hint="add the market to carttrace.sandbox's market table, or leave the store out of sandbox rows",
            )
        arguments = dict(self._snapshot_arguments(world))
        own = dict(arguments.pop("provenance", None) or {})
        twin = {
            "twin_of": store.store_id,
            "world": {"catalog": world.catalog, "category": world.category, "seed": world.snapshot_seed},
        }
        own_arguments = {k: v for k, v in arguments.items() if k in (*OWN_ARGUMENTS, "captured_at")}
        pack = build_envpack(**own_arguments, profile=profile, market=code, provenance={**twin, **own})
        core = spec_of(world.union_core)
        if core is not None:
            # D117: the category's union core on every product, the base untouched, each value's origin recorded
            policies = store_policies(pack.market(), world.category)
            filled = fill_products(own_arguments["products"], world.category, core, policies=policies)
            own_arguments["products"] = filled.products
            pack = build_envpack(
                **own_arguments, profile=profile, market=code, provenance={**twin, **own, UNION_CORE: filled.summary}
            )
        return BuiltWorld(pack=pack, fixtures=list(self.fixtures(pack.snapshot)))

    def _replica(self, store: StoreRef, world: World, profile: ProtocolProfile) -> BuiltWorld:
        """A replica world: the store's capture as a pack, speaking `profile` in the world's market."""
        if self.replicas is None or world.capture is None:
            raise ValidationError(
                f"world of {store.store_id} names capture {world.capture} but no replicas are configured",
                hint="build and run replica manifests through carttrace.collect.make_manifest and run_manifest",
            )
        code = sandbox_market(world.market)
        if code is None:
            raise ValidationError(f"the sandbox has no market {world.market!r} for store {store.store_id}")
        try:
            dialect_for(profile.ucp)
            mcp_version(profile.mcp)
        except UnsupportedVersion as exc:
            raise ValidationError(
                f"capture {world.capture} speaks versions the sandbox cannot serve: {exc.message}", hint=exc.hint
            ) from None
        key = (world.capture, code, world.search_memory, world.agent_surface)
        arguments = self._replicas.get(key)
        if arguments is None:
            try:
                if world.capture.startswith(ID_PREFIX) and self.replicas.v2_dir is not None:
                    path = find_capture_v2(self.replicas.v2_dir, world.capture).path
                else:
                    path = find_capture(self.replicas.captures_dir, world.capture)
            except NotFoundError as exc:
                raise NotFoundError(
                    f"capture {world.capture} of {store.store_id} is gone: {exc.message}",
                    hint="raw captures expire after live.retention_days_raw days; capture the store again and build "
                    "a new manifest",
                ) from None
            raw = self._with_surface(
                store, world, self._with_memory(store, world, self.replicas.snapshot(path, market=code))
            )
            arguments = self._replicas[key] = envpack_arguments(raw, world=world)
        own = dict(arguments.get("provenance") or {})
        origin = {"replica_of": store.store_id, "world": {"catalog": REPLICA, "capture": world.capture}}
        pack = build_envpack(
            **{k: v for k, v in arguments.items() if k not in ("profile", "market", "provenance")},
            profile=profile,
            market=code,
            provenance={**origin, **own},
        )
        if world.sold_out is not None:
            pack = with_sold_out(pack, world.sold_out)
        if world.store_replies is not None:
            pack = with_store_replies(pack, self._replies_record(store, world.store_replies))
        return BuiltWorld(pack=pack, fixtures=list(self.fixtures(pack.snapshot)))

    def pick_capture_of(self, store_id: str, category: str) -> str | None:
        """The capture an open-pool stand-in for this store matches its products to (D104): the store's newest capture
        of the category with products, as a replica would be built from; None when it has none or no replicas are
        configured, and the stand-in's products are spread over the pool's types as before."""
        if self.replicas is None or self.replicas.v2_dir is None:
            return None
        key = (store_id, category)
        if key not in self._picks:
            found = newest_captures_v2(
                self.replicas.v2_dir, store_ids=[store_id], category=category, partial_min=self.replicas.partial_min
            ).get(store_id)
            self._picks[key] = found.id if found is not None else None
        return self._picks[key]

    def view_of(self, store: StoreRef, platform_pack: str | None) -> str | None:
        """The recorded agent view a stand-in of `store` serves under `platform_pack` (D100): the store's, when its
        agent view was recorded live and the pack is Shopify's, the platform it was recorded on; None otherwise."""
        if self.views is None or platform_pack != VIEW_PACK:
            return None
        found = self.views.view_of(store.store_id)
        return found.ref if found is not None else None

    def replies_of(self, store: StoreRef) -> StoreReplies | None:
        """What the store's replica worlds serve from its newest store replies record (D70); None when it has none."""
        if self.replicas is None or self.replicas.replies_dir is None:
            return None
        if store.store_id not in self._replies:
            record = newest_replies(self.replicas.replies_dir, store.store_id)
            self._replies[store.store_id] = replies_ref(record) if record is not None else None
        return self._replies[store.store_id]

    def _replies_record(self, store: StoreRef, ref: StoreReplies) -> StoreRepliesRecord:
        if self.replicas is None or self.replicas.replies_dir is None:
            raise ValidationError(
                f"world of {store.store_id} names store replies {ref.record}, which cannot be applied here",
                hint="build replica worlds through configured_worlds, whose replicas know where the records are",
            )
        return load_replies(self.replicas.replies_dir, store.store_id, ref.record)

    def sold_out_of(self, store: StoreRef, market: str) -> SoldOutRate | None:
        """The rate the store's replica worlds in `market` answer sold out at, from its newest sold-out record; None
        when it has none."""
        if self.replicas is None or self.replicas.sold_out_dir is None:
            return None
        key = (store.store_id, market)
        if key not in self._sold_out:
            record = newest_record(self.replicas.sold_out_dir, store.store_id)
            self._sold_out[key] = rate_for(record, market) if record is not None else None
        return self._sold_out[key]

    def memory_of(self, store: StoreRef) -> str | None:
        """The id of the store's newest search memory, which its replica worlds replay; None when it has none."""
        if self.replicas is None or self.replicas.memory_dir is None:
            return None
        if store.store_id not in self._memories:
            memory = newest_memory(self.replicas.memory_dir, store.store_id)
            self._memories[store.store_id] = memory.id if memory is not None else None
        return self._memories[store.store_id]

    def _with_memory(self, store: StoreRef, world: World, raw: Any) -> Any:
        """The replica snapshot with the world's search memory replayed in it, when the world names one."""
        if world.search_memory is None:
            return raw
        if self.replicas is None or self.replicas.memory_dir is None or not isinstance(raw, StoreSnapshot):
            raise ValidationError(
                f"world of {store.store_id} names search memory {world.search_memory}, which cannot be applied here",
                hint="build replica worlds through configured_worlds, whose replicas know where memories are",
            )
        memory = load_memory(self.replicas.memory_dir, store.store_id, world.search_memory)
        return with_search_memory(raw, memory)

    def surface_of(self, store: StoreRef) -> str | None:
        """The id of the store's newest agent surface, whose tools its replica worlds list; None when it has none."""
        if self.replicas is None or self.replicas.surface_dir is None:
            return None
        if store.store_id not in self._surfaces:
            surface = newest_surface(self.replicas.surface_dir, store.store_id)
            self._surfaces[store.store_id] = surface.id if surface is not None else None
        return self._surfaces[store.store_id]

    def _with_surface(self, store: StoreRef, world: World, raw: Any) -> Any:
        """The replica snapshot listing its store's recorded tools, when the world names a surface."""
        if world.agent_surface is None:
            return raw
        if self.replicas is None or self.replicas.surface_dir is None or not isinstance(raw, StoreSnapshot):
            raise ValidationError(
                f"world of {store.store_id} names agent surface {world.agent_surface}, which cannot be applied here",
                hint="build replica worlds through configured_worlds, whose replicas know where surfaces are",
            )
        surface = load_surface(self.replicas.surface_dir, store.store_id, world.agent_surface)
        return with_agent_surface(raw, surface)

    def save(self, pack: EnvPack) -> None:
        if self.directory is not None:
            pack.save(self.directory)

    def load(self, envpack_id: str, store: StoreRef, world: World, profile: ProtocolProfile) -> EnvPack:
        """The pack a manifest row names: from its file, or rebuilt from its recipe. Raises ValidationError when the
        rebuilt world is not the one named, so a row never runs somewhere else."""
        self.check_catalog(world.catalog)
        if self.directory is not None:
            path = self.directory / f"{envpack_id}.json"
            if path.is_file():
                return EnvPack.load(path)
        pack = self.build(store, world, profile).pack
        if pack.id != envpack_id:
            raise ValidationError(
                f"store {store.store_id}'s sandbox world rebuilt as {pack.id}, not {envpack_id}",
                hint="the catalog or the code that builds worlds changed since the manifest was made; build a new "
                "manifest, or restore var/envpacks from where the manifest was built",
            )
        self.save(pack)
        return pack

    def unmasked(self, pack: EnvPack, store: StoreRef, world: World, profile: ProtocolProfile) -> EnvPack:
        """The world `pack` is before its exposure mask (D93's exposure sheet reads the store's own world): `pack`
        itself when its recipe withholds nothing, else the pack its mask was cut from, from its file when saved, or
        rebuilt from the recipe without the mask and saved."""
        if world.exposure is None or not world.exposure.withheld:
            return pack
        cut = pack.doc.get("provenance", {}).get(EXPOSURE_PROVENANCE)
        base_id = cut.get("base") if isinstance(cut, dict) else None
        if self.directory is not None and isinstance(base_id, str):
            path = self.directory / f"{base_id}.json"
            if path.is_file():
                return EnvPack.load(path)
        base = self.build(store, world.model_copy(update={"exposure": None}), profile).pack
        self.save(base)
        return base

    def captured(self, world: World) -> CapturedStore | None:
        """The store's own protocols when the world's catalog snapshot is a replica capture of it, else None.
        Raises what cutting the snapshot raises (NotFoundError for a category the catalog lacks)."""
        self.check_catalog(world.catalog)
        self._snapshot_arguments(world)
        return self._captured[_snapshot_key(world)]

    def _snapshot_arguments(self, world: World) -> dict[str, Any]:
        key = _snapshot_key(world)
        found = self._arguments.get(key)
        if found is None:
            if world.catalog == OPEN:
                matched = {"match": world.pick_from} if world.pick_from is not None else {}
                raw = self.snapshot(
                    OPEN,
                    world.category,
                    seed=world.snapshot_seed,
                    size=world.size,
                    market=world.home or world.market,
                    **matched,
                )
            else:
                raw = self.snapshot(world.catalog, world.category, seed=world.snapshot_seed, size=world.size)
            found = envpack_arguments(raw, world=world)
            self._captured[key] = captured_store(raw, found)
            self._arguments[key] = found
        return found


def _snapshot_key(world: World) -> tuple[str, str, str, int, int, str | None, str | None]:
    """A world's catalog snapshot depends on these alone: the market and the profile are set on the pack after (a
    pooled world's products depend on its store's home market, D91, and on the capture they were matched to, D104)."""
    return (world.store_id, world.catalog, world.category, world.snapshot_seed, world.size, world.home, world.pick_from)

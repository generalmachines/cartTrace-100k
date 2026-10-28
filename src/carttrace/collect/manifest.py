"""Manifests: every episode a collection run will run, decided up front, as one canonical file named by its content.

Why a manifest. A run must be resumable, auditable and repeatable: which store, which task, under which condition,
protocol profile and mode, which repeat, with which seed. So all of that is decided before the first request, from
explicit inputs only (the store queue, preflight results, the categories, generated tasks, the conditions, the
repeats and a seed), and written as canonical JSON whose id is `mf-<sha12>` of its content. The same inputs give the
same bytes; a changed input gives a new id.

Episode ids come from what a row is, never from the run: `stable_id("ep", store, task, condition, profile, mode,
repeat, seed, market, world, handoff)`. A resumed run, or a later manifest that shares rows, finds those episodes
already in storage and skips them.

Rows are ordered store by store (queue order), live then sandbox then reference, task by task, repeat by repeat, with
a row's conditions next to each other, so the two halves of a pair run close together. A row's seed depends on the
store, the task and the repeat only, so the conditions of one pair share it. Stores, tasks, profiles and sandbox
worlds are tables the rows point into, which keeps a manifest of tens of thousands of rows small.

What leaves a store out of a mode is recorded, not guessed: it gets a note saying why (no preflight, a preflight that
found no UCP over MCP, a market the sandbox does not have, a category the catalog lacks, a persona with no address in
the store's market).

Checkout rows (D13) run in the sandbox only. A plan that asks for live rows and the checkout condition together is
refused when it is made, and a manifest that holds a live row whose condition authorises an order does not validate,
so no live episode can ever be told to buy. The persona a checkout row pays as is resolved when the manifest is built
(the plan's, or the default in data/agent/personas.yaml) and written into the row's condition.

Each store keeps its own protocol profile (carttrace.collect.profiles): a profiles file, the replica capture or the store's
preflight, before the plan's preset, which is only the fallback.

Multi-protocol runs and coverage (D15). A plan may name `surfaces`: every sandbox and reference row is then crossed with
each listed surface the store serves and the sandbox implements, and names it (`row.surface`); a surface the sandbox
has no implementation of yet is left out with a note naming it, so new surfaces join when they register. A live row
only ever runs on carttrace.live's UCP over MCP, read-only, and a manifest holding any other live surface does not validate.
A plan may spread each store's tasks over its catalog (`selection="spread"`, with a rotation window and a cap per
product, carttrace.collect.coverage), and the manifest records per store and mode what the tasks cover (`coverage`). And it
may pair live rows with replica rows (`pair_replica`): each store's tasks are drawn once, from its live fixtures, and
its sandbox rows run those same tasks on the replica of its capture (whatever catalog the plan names for twins), with
the same persona, seed and condition; each such row names the live row it pairs with (`replica_of`) and both rows carry
one `pair_id`, so the bench can join the pair and check the replica's fidelity on outcomes. A store with no live rows
gets no replica rows; a store whose sandbox world is a stand-in (a twin cut from a catalog, not a capture of the
store) is not paired, and the manifest says so. Seen 2026-10-12: a paired pilot built on twins drew tasks separately
per mode, so its 90 live rows and 100 sandbox rows shared no task, and a store with no live rows still got sandbox
rows. The split is the safety rule's: live rows measure discovery, catalog reads and cart building through what the
live store serves; its replica measures checkout and payment through every surface, so a paired plan may ask for the
checkout condition, which only the replica rows take.

Personas and intents (D20). A plan may cross each store's tasks with personas (`personas`: all of them, a count chosen
per store by seed, or names): each task is then written for one persona, chosen per store and task by seed, and the
row's condition params name it (`persona`), as the task's buyer block does, so the runner pays as that persona on a
checkout row and a reader can pair episodes by shopper. A plan may draw its tasks from any of the ten intents
(`intents`) and may have a model rewrite each kept task's opening in its persona's voice (`voice: model`,
carttrace.agents.intents_voice), applied after selection so only the tasks that run cost a call; the rewrite is stored in the
task, so running the manifest calls no model.

Compatibility. A plan that sets none of these serialises exactly as plans did before them: its rows name no surface
and its manifest holds no coverage and no new notes, so the ids of manifests and episodes already built do not change.
A row on UCP over MCP has the episode id it always had, so a surfaces plan's ucp_mcp rows resume episodes a plain plan
already ran.

Replicas (decision D12) are chosen explicitly: a plan whose catalog is `replica` (`carttrace collect manifest --catalog
replica`) runs each store's sandbox and reference rows on the replica of its newest capture in its category, speaking
the profile the store was captured at, with tasks written around the store's own products. A store with no capture
runs on a twin from sandbox.default_catalog instead, and the manifest says so; with the default `replica` it gets no
sandbox rows, so no store is shopped on another source's products. Any other catalog plans twins only, so a
new capture never changes a plan that did not ask for replicas.

Platform packs (D42, D68). A plan may name `platforms` (shopify, woocommerce, magento, medusa, saleor, vendure,
prestashop, bigcommerce, generic): every sandbox and
reference row is then crossed with each, its world the store's world under that platform's pack
(carttrace.sandbox.platforms.with_platform_pack) and its profile the store's with the platform's versions
(`platform_profile`). Each row names its `platform_pack`, and so does its episode header, so analytics compare agents
across platforms on the same products. A platform the world already speaks (a Shopify replica under `shopify`, a twin
under `generic`) keeps the world and its episode ids; only such a row pairs with a live row. Live rows ignore the
axis: a live store speaks its own platform.

Orders and rails (D66). A plan whose `boundary` is `order` (the CLI's default) makes every sandbox and reference row a
checkout row: its condition gains the judge's convention `{"checkout": true, "persona": ...}`, so the journey runs on
from the cart through checkout, the shopper's consent and payment to the order read back. `cart` keeps cart-boundary
rows for comparison, and None (every plan before D66) builds what it always built. A replica row paired with a live
row keeps the cart boundary, since the live half never buys. A plan may name `rails` (carttrace.collect.rails): every order
row is crossed with each, its world serving the store's catalog through the rail's protocol (adopted where the store
does not speak it), and each row names its `rail` and whether it was `adopted`. `checkout_faults` copies the rates of
config's sandbox.checkout_faults, which each order row's episode draws from by its seed.

The publishable lane (D85, carttrace.collect.open_lane). A plan marked `publishable` runs on twins of a licensed catalog
only, and its manifest refuses to validate with a live, playground, paired or hand-off row, or a world that is a
replica or carries a live session's records; the licence itself is checked where it can be read, by the builder and
the runner. Its twins may take the exposure design.

Rail versions and brokers (D71). `rail_versions` crosses a rail's rows with versions of its protocol (ucp_psp with UCP
releases, ap2_mandates with AP2 v0.1.0 and v0.2.0), each where the store's profile allows it (`version_allowed`), and
each row names its `rail_version`; a checkout-only UCP release runs on ucp_mcp only. `psps` crosses ucp_psp rows with
the payment brokers behind the store's PSP (stripe, adyen, braintree, paypal), and each row names its `psp`. With
`draw_axes` each row draws one version and one broker by seed instead, so the rows do not multiply.

The surface x rail matrix (D89, carttrace.collect.cells). With `draw_cells` each order task takes `cells_per_task` valid
(surface, rail) cells, the ones the store's rows have used least so far among those its persona can pay on, so every
cell is measured on purpose and not in proportion to what stores declare; a UCP binding the store's capture did not
record is served as adopted, and each row says so (`surface_adopted`). The shop_pay rail is planned only where Shop
Pay is real: under the shopify platform pack, or on a Shopify store's own platform.
"""

from __future__ import annotations

import multiprocessing
import re
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic import ValidationError as PydanticValidationError

from carttrace.agents.intents import chosen_intents
from carttrace.agents.memo import remembering
from carttrace.agents.tasks import recording_drops
from carttrace.collect.cells import CELL_SET, CellDraw, cell_of, shop_pay_platform, valid_cell
from carttrace.collect.conditions import (
    CONDITIONS,
    MARKET_ALIASES,
    ConditionName,
    HiddenDetail,
    authorises_checkout,
    baseline,
    checkout,
    cross_border,
    cross_border_market,
    detail_hidden,
    hidden_details,
    hides_decisive,
    variant_profile,
    version_variant,
)
from carttrace.collect.coverage import (
    POOL_DRAWS,
    Candidate,
    ProductFacts,
    Selection,
    StoreCoverage,
    coverage_of,
    describe,
    product_facts,
    spread,
    wanted_products,
)
from carttrace.collect.exposure import (
    BY_ID,
    RANDOM_MASKS,
    MaskedWorld,
    Served,
    design_masks,
    exposure_condition,
    mask_onto,
    masked_world,
    ordered,
    served_counts,
)
from carttrace.collect.live_webmcp import live_protocol_kinds, live_row_surface, webmcp_note
from carttrace.collect.payment import built_to_fail, can_pay, limited_wallet, pays_somewhere, planned_payment
from carttrace.collect.playgrounds import is_playground_id
from carttrace.collect.ports import RUN_MODES, GenerateTasks, PersonaBook, RunMode
from carttrace.collect.profiles import (
    DEFAULT_SURFACE,
    LIVE_SURFACES,
    LIVE_WEBMCP,
    SURFACE_KINDS,
    WEBSITE,
    WEBSITE_VERSION,
    CapturedStore,
    RowSurface,
    StoreProtocols,
    registered_surfaces,
    spoken_surfaces,
    store_protocols,
    surface_version,
)
from carttrace.collect.rails import (
    CHECKOUT_ONLY_UCP,
    RAIL_NAMES,
    SHOP_PAY_NS,
    adopted_by,
    rail,
    rail_kinds,
    rail_profile,
    rail_surfaces,
    version_allowed,
)
from carttrace.collect.replica_site import website_condition_note
from carttrace.collect.stand_ins import with_stand_ins
from carttrace.collect.store_spec import StoreSpec
from carttrace.collect.v1_rows import RowExperiment, V1Recipe
from carttrace.collect.worlds import (
    DEFAULT_SIZE,
    REPLICA,
    BuiltWorld,
    CaptureInfo,
    SandboxWorlds,
    World,
    presents,
    sandbox_market,
)
from carttrace.config.schema import CheckoutFaults
from carttrace.core.canon import digest, dumps, loads, pretty
from carttrace.core.errors import MCError, NotFoundError, ValidationError
from carttrace.core.files import atomic_write_text
from carttrace.core.ids import seeded_rng, stable_id
from carttrace.protocols.profile import ProtocolProfile
from carttrace.sandbox import EnvPack
from carttrace.sandbox.catalogs.pools import OPEN, stand_in_domain
from carttrace.sandbox.core.brokers import BROKER_NAMES
from carttrace.sandbox.platforms import PLATFORM_PACKS, platform_profile
from carttrace.sandbox.presented import id_scheme_for, presented_ids_in
from carttrace.stores.categories import Category, CategoryCatalog
from carttrace.stores.model import Store
from carttrace.stores.results import StoreResult
from carttrace.trace.schema import Condition, ExposureDesign, PaymentFaultKind, StoreRef, StoreReplies, TaskRef

MANIFEST_SCHEMA = "mc.collect.manifest.v1"
MANIFESTS_DIR = "manifests"
DEFAULT_PROFILE_PRESET = "shopify-live-2026-10"
DEFAULT_VARIANT_PRESET = "ucp-2026-04-08"
DEFAULT_HOME_MARKET = "US"
"""The delivery market of a store whose queue market is Global."""
WEBSITE_CONDITIONS: tuple[str, ...] = ("baseline", "cross_border")
"""Conditions a live website row takes (D31)."""
PLAYGROUND_CONDITIONS: tuple[str, ...] = ("baseline", "cross_border", "checkout")
"""Conditions a playground row takes (D44): a store we own may be bought from, so checkout too (its UCP rows; a
website row ends at a selection). Sealed faults and hidden details belong to sandbox worlds."""
LIVE_SURFACE = "ucp_mcp"
"""The one live surface today: UCP over MCP on Shopify (carttrace.live)."""
SEED_SPACE = 2**31
CAPTURE = "capture it with `carttrace live capture {domain} --category {category} --live`"
PERSONAS_SPEC = re.compile(r"^(?:all|[1-9]\d*|[a-z][a-z0-9-]*(?:,[a-z][a-z0-9-]*)*)$")
"""`personas`: all, a count, or persona slugs separated by commas."""
Voicer = Callable[[str, Sequence[TaskRef]], list[TaskRef]]
"""(store id, the tasks kept for a store) -> the same tasks with their openings in their personas' voices
(carttrace.agents.intents_voice.voice_tasks, bound by carttrace.collect.inputs)."""


WORLD_MODES: frozenset[str] = frozenset({"sandbox", "reference"})
"""Modes whose rows run in a sandbox world (an envpack); a live row reaches the store, a playground row our own."""
MarketContext = Literal["fill", "none"]
LIVE_MARKET_CONTEXT: MarketContext = "none"
"""What a live row's market context is when its manifest names none."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _modes(*modes: RunMode) -> list[RunMode]:
    return list(modes)


def _conditions(*names: ConditionName) -> list[ConditionName]:
    return list(names)


class ManifestPlan(_Frozen):
    """The choices a manifest was built with, kept in it so it can be rebuilt and audited. Selections are sorted and
    modes and conditions put in their canonical order, so one selection always gives one manifest."""

    stores: list[str] = Field(default_factory=list)
    """Queue ids (store-004) or domains; empty means every store in the chosen categories."""
    categories: list[str] = Field(default_factory=list)
    """Category slugs; empty means every category."""
    modes: list[RunMode] = Field(default_factory=lambda: _modes("sandbox"), min_length=1)
    conditions: list[ConditionName] = Field(default_factory=lambda: _conditions("baseline"), min_length=1)
    repeats: int = Field(default=1, ge=1)
    """Episodes per store, task and condition (collect.repeats), each with its own seed."""
    seed: int = 0
    """Base seed. A new seed gives new worlds, new task draws, new row seeds, and so new episodes."""
    tasks_per_store: int | None = Field(default=None, ge=1)
    """Keep the first N tasks generated for each store and mode; None keeps them all."""
    limit: int | None = Field(default=None, ge=1)
    """At most this many rows, cut between whole groups (a store, mode, task and repeat with all its conditions)."""
    catalog: str = "abo"
    """Catalog source for sandbox and reference rows (sandbox.default_catalog), or `replica` for each store's
    replica, built from its newest capture."""
    catalog_size: int = Field(default=DEFAULT_SIZE, ge=1)
    profile: str = DEFAULT_PROFILE_PRESET
    """Preset a store is planned under when no profile captured from it is given."""
    variant: str = DEFAULT_VARIANT_PRESET
    """Preset whose UCP release the version_variant condition moves to."""
    cross_border_markets: list[str] = Field(default_factory=lambda: ["US", "GB"], min_length=1)
    """Preferred delivery markets for cross_border, in order; the first that is not the store's own is used."""
    hypotheses: list[str] = Field(default_factory=list)
    """Limit detail_hidden to these hypothesis ids; empty means every hypothesis that can be hidden."""
    handoff: bool = False
    """Live rows hand their verified cart to a sandbox checkout."""
    persona: str | None = None
    """The persona checkout rows pay as (data/agent/personas.yaml); None is the file's default persona."""
    require_preflight: bool = True
    """Live rows only for stores whose preflight found UCP over MCP. Off, Shopify stores without a preflight result
    are planned too, and their episodes record what discovery finds."""
    surfaces: list[str] = Field(default_factory=list)
    """Sandbox surfaces (sandbox.surface kinds) every sandbox and reference row is crossed with, among those the
    store serves; empty runs every row on UCP over MCP and names no surface, as manifests did before surfaces."""
    selection: Selection = "first"
    """first: the first tasks_per_store tasks generated; spread: tasks spread over the store's catalog by product
    type, price band and decisive detail, at most product_cap per product (carttrace.collect.coverage)."""
    rotation: int | None = Field(default=None, ge=0)
    """The rotation window spread selection starts from: a UTC day's ordinal, or an autopilot round; None is window
    0. Successive windows cover new products."""
    product_cap: int = Field(default=1, ge=1)
    """At most this many of a store's tasks per product, under spread selection."""
    pair_replica: bool = False
    """Sandbox rows run the live rows' tasks on the store's replica (its capture, when it has one, whatever `catalog`
    names for twins), each naming the live row it pairs with, both rows sharing a pair id. A store with no live rows
    gets no sandbox rows. The checkout condition, when asked, goes to the replica rows only."""
    personas: str | None = None
    """Cross each store's tasks with personas (data/agent/personas.yaml): "all", a count ("4": that many chosen per
    store by seed, among those with an address in the store's market) or names ("card-ok,ananya-moorcroft"). Each
    task is written for one persona, chosen per store and task by seed, named in the row's condition params and in the
    task's buyer block. None writes every task for `persona` (or the file's default), as every plan did."""
    intents: list[str] = Field(default_factory=list)
    """The intents tasks are drawn from (carttrace.agents.spec.ALL_INTENTS, or "all"), round robin so the first N tasks
    cover N intents; empty writes the five first intents as every plan did. `custom`, alone, draws each store's tasks
    from its own written intents (carttrace.agents.intents_custom, `carttrace agents intents write`) instead of the templates."""
    intent_set: str | None = None
    """A custom intent set file every store's tasks are drawn from (one store's set); None draws each store's newest
    set under var/agent/intents when the intents are custom."""
    voice: Literal["template", "model"] = "template"
    """model: a signed-in Claude model rewrites each kept task's opening in its persona's voice behind the code guard
    (carttrace.agents.intents_voice), and the rewrite is stored in the task; template keeps the generator's words."""
    platforms: list[str] = Field(default_factory=list)
    """Platform packs (carttrace.sandbox.platforms.PLATFORM_PACKS) every sandbox and reference row is crossed with, D42;
    empty runs each world under its own platform and names none, as every plan did. Live rows ignore it."""
    market_context: MarketContext | None = None
    """What the harness adds to a live row's cart and catalog calls (D43): `fill` puts the row's market's country and
    currency where the agent left them out (carttrace.live.market); `none` sends what the agent sends, so the store prices
    us as it prices any visitor from our network position. None is the default, `none`, and leaves the rows' bytes
    as they were."""
    boundary: Literal["order", "cart"] | None = None
    """Where sandbox and reference rows end (D66): `order`, every row goes on to an order; `cart`, at a verified cart
    (checkout condition rows still order). None builds rows as plans before D66 did."""
    rails: list[str] = Field(default_factory=list)
    """Payment rails (carttrace.collect.rails) every order row is crossed with, D66; empty pays over the row's own surface
    and names no rail."""
    checkout_faults: CheckoutFaults | None = None
    """The checkout-time fault rates order rows draw from (config's sandbox.checkout_faults, copied when the manifest
    was built); None draws none."""
    exposure: Literal["design"] | None = None
    """`design` (D63, carttrace.collect.exposure): each replica store's sandbox and reference rows run under exposure masks
    instead of the plan's conditions: the full-exposure baseline, each data type the store shows withheld on its own,
    and `exposure_random` seeded masks, every mask on every platform pack where it is honest, each task on one of the
    store's surfaces in turn. The boundary applies to them (an order boundary makes them order rows), and so do the
    rails, one a row: each task takes the plan's rails in turn, then the rail's surfaces. Live and playground rows
    keep the plan's conditions. None plans no masks."""
    exposure_random: int = Field(default=RANDOM_MASKS, ge=0)
    """Seeded random masks per store in an exposure design."""
    exposure_types: list[str] = Field(default_factory=list)
    """D88: the data types an exposure design spans, among those each store's world shows (a store benchmark's chosen
    types); empty spans every type the store shows, as every design did."""
    rail_versions: dict[str, list[str]] = Field(default_factory=dict)
    """Rail to the versions of its protocol its rows are crossed with (D71); a rail not named runs at its one
    version."""
    psps: list[str] = Field(default_factory=list)
    """Payment brokers ucp_psp rows are crossed with (D71, carttrace.sandbox.core.brokers); empty pays through the sandbox's
    own PSP and names none."""
    draw_axes: bool = False
    """Draw each row's rail version and broker by seed instead of crossing every row with each (D71), so every
    version and broker is in use and the rows do not multiply (the sandbox loop's default)."""
    draw_rails: bool = False
    """Draw each row's payment rail instead of crossing every row with each (D72), as draw_platforms draws the
    platform pack: the fast fallback round's rows (an exposure design already takes the rails one a task)."""
    draw_cells: bool = False
    """D89: each order task takes `cells_per_task` (surface, rail) cells of the matrix (carttrace.collect.cells), the ones
    this store's rows have used least so far among those its persona's wallet can pay on (ties by seed), instead of the
    rails in turn and then their surfaces; a UCP binding the store does not serve itself is served as adopted, and
    every row names `surface_adopted`. The sandbox loop's default."""
    cells_per_task: int = Field(default=1, ge=1)
    """D89: how many cells each task of a `draw_cells` plan takes (an exposure design's tasks already run every mask,
    so one; the fallback round's take several)."""
    payment_fault_bp: int | None = Field(default=None, ge=0, le=10_000)
    """D80: of a plan's order rows whose tasks are crossed with personas, or drawn from custom intents (D86), the
    share in basis points whose payment is designed to fail (config's sandbox.payment_fault_bp, copied when the
    manifest was built): each such task is written for a persona built to fail (a decline, a failed or abandoned 3-D
    Secure challenge), or for one whose wallet holds nothing one of the plan's rails takes, paired with that rail
    (`no_instrument`). Every other task is written for a persona that pays, on a rail its wallet can pay on; a custom
    intent written for a persona built to fail is taken only as a fault task, at the share. None draws personas as
    plans before D80 did (about 40% of them built to fail), still never pairing a wallet with a rail it cannot pay on
    by accident."""
    publishable: bool = False
    """The publishable lane (D85, carttrace.collect.open_lane): every row is a sandbox or reference row on a twin cut from
    the plan's catalog, whose licence must allow publication, and nothing in the manifest comes from a live store or
    a capture of one (no replica, no live pair or hand-off, no custom intents, no store's live replies). The manifest
    refuses to validate otherwise, and the runner refuses to start it unless every catalog's rights allow publication.
    An exposure design is allowed on it: a mask withholds data types of the twin's own catalog products."""
    v1: V1Recipe | None = None
    """D96: the sandbox v1 design this manifest was cut under (carttrace.collect.v1_cut): its layer and its baseline, mask
    and designed-failure shares, as config gave them. None on every other plan."""
    draw_platforms: bool = False
    """Draw each row's platform pack instead of crossing every row with each (D72): a task's variants that differ in
    nothing but their platform keep one, the pack the store's rows have used least so far (ties by seed), so every
    pack is covered and the rows do not multiply (the sandbox loop's default). Off crosses, as every plan did."""
    store_specs: dict[str, StoreSpec] = Field(default_factory=dict)
    """Store id to the one spec it is served under on every row (carttrace.collect.store_spec, owner 2026-10-16: one store,
    one pack and one protocol set): its pack replaces `platforms`, its versions `rail_versions`, its broker `psps`,
    and its profile declares every rail it pays on. A store not named is planned under the plan's own options."""

    @field_validator("exposure_types")
    @classmethod
    def _exposure_types(cls, value: list[str], info: ValidationInfo) -> list[str]:
        unknown = sorted(set(value) - set(BY_ID))
        if unknown:
            raise ValueError(f"exposure_types: not exposure data types: {unknown}")
        if value and info.data.get("exposure") is None:
            raise ValueError("exposure_types chooses the types an exposure design spans: set exposure to design")
        return ordered(value)

    @field_validator("stores", "categories", "hypotheses", "intents")
    @classmethod
    def _sorted(cls, value: list[str]) -> list[str]:
        return sorted(set(value))

    @field_validator("intents")
    @classmethod
    def _custom_alone(cls, value: list[str]) -> list[str]:
        if CUSTOM_INTENTS in value and len(value) > 1:
            raise ValueError("intents: custom draws from the store's own intents and goes with no template intent")
        return value

    @field_validator("personas")
    @classmethod
    def _personas(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not PERSONAS_SPEC.match(text):
            raise ValueError("personas is 'all', a count, or persona slugs separated by commas")
        if text == "all" or text.isdigit():
            return text
        return ",".join(sorted(set(text.split(","))))

    @field_validator("modes")
    @classmethod
    def _modes(cls, value: list[RunMode]) -> list[RunMode]:
        return [mode for mode in RUN_MODES if mode in value]

    @field_validator("conditions")
    @classmethod
    def _conditions(cls, value: list[ConditionName]) -> list[ConditionName]:
        return [name for name in CONDITIONS if name in value]

    @field_validator("platforms")
    @classmethod
    def _platforms(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(PLATFORM_PACKS))
        if unknown:
            raise ValueError(f"not platform packs: {unknown}; platform packs are {list(PLATFORM_PACKS)}")
        return [name for name in PLATFORM_PACKS if name in value]

    @field_validator("rails")
    @classmethod
    def _rails(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(RAIL_NAMES))
        if unknown:
            raise ValueError(f"not payment rails: {unknown}; rails are {list(RAIL_NAMES)}")
        return [name for name in RAIL_NAMES if name in value]

    @field_validator("psps")
    @classmethod
    def _psps(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(BROKER_NAMES))
        if unknown:
            raise ValueError(f"not payment brokers: {unknown}; brokers are {list(BROKER_NAMES)}")
        return [name for name in BROKER_NAMES if name in value]

    @field_validator("rail_versions")
    @classmethod
    def _rail_versions(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for name, versions in sorted(value.items()):
            if name not in RAIL_NAMES:
                raise ValueError(f"rail_versions: not a payment rail: {name!r}")
            known = rail(name).versions
            unknown = sorted(set(versions) - set(known))
            if unknown:
                raise ValueError(f"rail_versions: {name} is not crossed with {unknown}; it takes {list(known)}")
            out[name] = [v for v in known if v in versions]
        return out

    @field_validator("surfaces")
    @classmethod
    def _surfaces(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(SURFACE_KINDS))
        if unknown:
            raise ValueError(f"not surface kinds: {unknown}; surface kinds are {list(SURFACE_KINDS)}")
        return [kind for kind in SURFACE_KINDS if kind in value]

    @model_validator(mode="after")
    def _no_live_checkout(self) -> Self:
        if "live" in self.modes and "checkout" in self.conditions and not self.pair_replica:
            raise ValueError(
                "the checkout condition runs only in the sandbox: a live journey ends at a verified cart and nothing "
                "is ever bought on a live store (D3, D13). Plan live rows and checkout rows in separate manifests, "
                "or pair them with --pair-replica, where only the replica rows take checkout"
            )
        return self

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.selection != "spread" and (self.rotation is not None or self.product_cap != 1):
            raise ValueError("rotation and product_cap choose among spread tasks: set selection to spread")
        if self.pair_replica and not ("live" in self.modes and "sandbox" in self.modes):
            raise ValueError("pair_replica pairs live rows with their replica rows: plan both live and sandbox modes")
        if self.intent_set is not None and self.intents and self.intents != [CUSTOM_INTENTS]:
            raise ValueError("intent_set draws custom intents: leave intents empty or set it to custom")
        if self.custom_intents and self.personas is not None:
            raise ValueError("custom intents carry their own personas: leave personas unset")
        if (self.rails or self.checkout_faults is not None) and self.boundary != "order":
            raise ValueError("rails and checkout faults are how an order is paid: set boundary to order")
        if self.exposure is not None and self.catalog != REPLICA and not self.publishable:
            raise ValueError(
                "an exposure design masks a store's own replica, or a publishable twin's catalog products: set "
                "catalog to replica, or plan the publishable lane"
            )
        if self.exposure is not None and self.pair_replica:
            raise ValueError("an exposure design is not paired with live rows: leave pair_replica off")
        uncrossed = sorted(set(self.rail_versions) - set(self.rails))
        if uncrossed:
            raise ValueError(f"rail_versions names rails the plan does not cross: {uncrossed}")
        if self.draw_rails and not self.rails:
            raise ValueError("draw_rails draws among the plan's payment rails: name them in rails")
        if self.draw_platforms and not self.platforms:
            raise ValueError("draw_platforms draws among the plan's platform packs: name them in platforms")
        if self.payment_fault_bp is not None and self.boundary != "order":
            raise ValueError("payment_fault_bp is the share of order rows built to fail payment: set boundary to order")
        if self.psps and "ucp_psp" not in self.rails:
            raise ValueError("payment brokers are behind the ucp_psp rail's PSP handler: add ucp_psp to rails")
        return self

    @model_validator(mode="after")
    def _store_specs(self) -> Self:
        for sid, spec in self.store_specs.items():
            if spec.platform not in self.platforms:
                raise ValueError(f"store_specs: {sid}'s pack {spec.platform} is not among the plan's platforms")
            if sorted(set(spec.rail_versions) - set(self.rails)):
                raise ValueError(f"store_specs: {sid} names versions of rails the plan does not cross")
            if spec.psp is not None and "ucp_psp" not in self.rails:
                raise ValueError(f"store_specs: {sid}'s broker is behind the ucp_psp rail: add ucp_psp to rails")
        return self

    @model_validator(mode="after")
    def _cells(self) -> Self:
        if self.draw_cells and not self.rails:
            raise ValueError("draw_cells draws (surface, rail) cells: name the plan's payment rails in rails")
        if self.draw_cells and self.draw_rails:
            raise ValueError("draw_cells already draws each task's rail with its surface: leave draw_rails off")
        if self.cells_per_task != 1 and not self.draw_cells:
            raise ValueError("cells_per_task is how many cells a draw_cells task takes: set draw_cells")
        return self

    @model_validator(mode="after")
    def _publishable(self) -> Self:
        problem = self.publishable_problem()
        if problem is not None:
            raise ValueError(f"a publishable plan (D85) {problem}")
        return self

    def publishable_problem(self) -> str | None:
        """Why this plan cannot be in the publishable lane (completing "a publishable plan ..."), or None when it can
        or is not publishable. The catalog's licence is checked where the licence can be read (carttrace.collect.open_lane):
        this is the plan's own shape, so a replica, a live half, a hand-off or a store's own intents never get in."""
        if not self.publishable:
            return None
        rules = (
            (self.catalog == REPLICA, "runs on twins of a licensed catalog, never on replicas of live stores"),
            (any(mode not in WORLD_MODES for mode in self.modes), "plans sandbox and reference rows only"),
            (self.pair_replica, "pairs no live rows"),
            (self.handoff, "takes no hand-off of a live cart"),
            (
                self.custom_intents,
                "draws template intents only: a store's custom intents are written from its live data",
            ),
        )
        return next((reason for broken, reason in rules if broken), None)

    @property
    def custom_intents(self) -> bool:
        """Whether tasks are drawn from each store's written intents rather than the template generator."""
        return self.intents == [CUSTOM_INTENTS] or self.intent_set is not None

    @property
    def extended(self) -> bool:
        """Whether the plan sets an option added for multi-protocol runs and coverage; only such a plan's manifest
        holds coverage and the notes that go with it."""
        return bool(self.surfaces) or self.selection != "first" or self.pair_replica

    @model_serializer(mode="wrap")
    def _without_unset_options(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """A plan that names no persona, and sets none of the options added since, serializes as plans did before
        they existed, so the ids of manifests already built (which hash their plan) do not change."""
        data: dict[str, Any] = handler(self)
        for name, unset in UNSET_OPTIONS.items():
            if name in data and data[name] == unset:
                data.pop(name)
        return data


UNSET_OPTIONS: Mapping[str, Any] = {
    "persona": None,
    "surfaces": [],
    "selection": "first",
    "rotation": None,
    "product_cap": 1,
    "pair_replica": False,
    "personas": None,
    "intents": [],
    "intent_set": None,
    "voice": "template",
    "platforms": [],
    "market_context": None,
    "boundary": None,
    "rails": [],
    "checkout_faults": None,
    "exposure": None,
    "exposure_random": RANDOM_MASKS,
    "exposure_types": [],
    "rail_versions": {},
    "psps": [],
    "draw_axes": False,
    "draw_rails": False,
    "draw_cells": False,
    "cells_per_task": 1,
    "draw_platforms": False,
    "store_specs": {},
    "publishable": False,
    "payment_fault_bp": None,
    "v1": None,
}
"""Plan options added after the first manifests, and the value at which a plan leaves each out of its bytes."""
CUSTOM_INTENTS = "custom"
"""The `intents` value that draws tasks from each store's written intents (carttrace.agents.intents_custom)."""


class ManifestRow(_Frozen):
    """One episode."""

    episode_id: str
    store_id: str
    task_id: str
    condition: Condition
    profile: str
    """Key into Manifest.profiles."""
    mode: RunMode
    repeat: int = Field(ge=0)
    seed: int
    market: str
    """The shopper's delivery market for this episode."""
    envpack: str | None = None
    """The sandbox world (env-<sha12>) of a sandbox or reference row; None for a live row."""
    handoff: bool = False
    surface: RowSurface | None = None
    """The surface the row runs on; None (a plan with no surfaces) is UCP over MCP at the profile's release."""
    replica_of: str | None = None
    """On a sandbox row of a pair_replica plan: the episode id of the live row it is the replica of."""
    pair_id: str | None = None
    """On both rows of a live and replica pair: the pair's id (pair-<sha12> of the store, task, condition, repeat,
    seed and market the two share), the same on each, which the episode header carries so the bench can join them."""
    platform_pack: str | None = None
    """The platform pack a sandbox or reference row's world is served under (D42), when the plan crosses platforms;
    the episode header carries it. None: the world's own platform, as every row before the axis."""

    market_context: MarketContext | None = None
    """On a live row whose plan named one: `fill` or `none` (ManifestPlan.market_context). None is `none`."""
    rail: str | None = None
    """The payment rail an order row is paid over (D66), when the plan crosses rails; the episode header carries it."""
    adopted: bool | None = None
    """On a row with a rail: whether the store itself does not speak the rail's protocol, so the replica serves it as
    if the store had adopted it."""
    rail_version: str | None = None
    """On a row of a plan that crosses its rail with versions (D71): the rail protocol's version its world serves."""
    psp: str | None = None
    """On a ucp_psp row of a plan that crosses brokers (D71): the payment broker behind the store's PSP."""
    payment_fault: PaymentFaultKind | None = None
    """On an order row whose payment is designed to fail (D80): how, as planned from its persona's wallet and its rail
    (decline, 3ds_fail, 3ds_abandon, no_instrument). None on a row built to pay, and on every row before D80. It
    does not enter the episode id: the persona and the rail that decide it already do."""
    surface_adopted: bool | None = None
    """On a row of a `draw_cells` plan (D89): whether the store itself does not serve the row's surface, so the
    replica serves it as adopted (a UCP binding its capture did not record; ACP's or AP2's own surface where the rail
    is adopted). None on every other row. Like `adopted`, it does not enter the episode id."""
    experiment: RowExperiment | None = None
    """D96: on a sandbox v1 row, what kind of trace it is (baseline, mask or designed failure), what was removed or
    injected and its paired row, which the episode writes into experiment.json. Not in the episode id."""

    @model_serializer(mode="wrap")
    def _without_unset_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """A row that names no surface, pairs with nothing and names no platform pack, market context, rail, rail
        version or broker serializes as rows did before any of them existed."""
        data: dict[str, Any] = handler(self)
        names = ("surface", "replica_of", "pair_id", "platform_pack", "market_context", "rail", "adopted")
        for name in (*names, "rail_version", "psp", "payment_fault", "surface_adopted", "experiment"):
            if data.get(name) is None:
                data.pop(name, None)
        return data

    @property
    def market_fill(self) -> bool:
        """Whether the harness fills this live row's market into the calls that leave it out (carttrace.live.market)."""
        return self.mode == "live" and (self.market_context or LIVE_MARKET_CONTEXT) == "fill"


class Manifest(_Frozen):
    """A sealed manifest. Build one with `build_manifest`; validation refuses one whose id is not its content."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: Literal["mc.collect.manifest.v1"] = Field(default="mc.collect.manifest.v1", alias="schema")
    id: str
    plan: ManifestPlan
    sources: dict[str, str] = Field(default_factory=dict)
    """What the manifest was built from, by content hash: the queue snapshot, the results file, the categories."""
    stores: dict[str, StoreRef]
    """Store id to the store, with its own delivery market."""
    tasks: dict[str, dict[str, TaskRef]]
    """Store id to task id to the task."""
    profiles: dict[str, ProtocolProfile]
    """Profile key (pk-...) to the profile, its source and note included."""
    worlds: dict[str, World] = Field(default_factory=dict)
    """Envpack id to the recipe of a sandbox world."""
    rows: list[ManifestRow]
    notes: list[str] = Field(default_factory=list)
    coverage: dict[str, dict[str, StoreCoverage]] = Field(default_factory=dict)
    """Store id to "live" or "sandbox" (sandbox and reference rows share their tasks) to what the tasks cover; kept
    for plans that set an option added for coverage (ManifestPlan.extended)."""
    served: dict[str, ServedPrices] = Field(default_factory=dict)
    """Store id to what its price probe saw (carttrace.live.prices, D43): the currency the store priced us in, which its
    live tasks are written in, for each store with live rows whose fixtures came from a probe."""
    shopper_maps: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """`<envpack>/<task id>` to the shopper's build-time decision map on that world (carttrace.agents.decision_map,
    `mc.shopper-map.v1`, D115): which variant the shopper accepts and why it refuses each other, its answer per field,
    its delivery wish and substitute policy. The runtime shopper looks them up. Left out when empty, so older
    manifests keep their ids."""

    @model_serializer(mode="wrap")
    def _without_empty_coverage(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """A manifest with no coverage and no served prices serializes, and so hashes, as manifests did before
        either was recorded."""
        data: dict[str, Any] = handler(self)
        for name in ("coverage", "served", "shopper_maps"):
            if not data.get(name):
                data.pop(name, None)
        return data

    @model_validator(mode="after")
    def _sealed(self) -> Self:
        expected = manifest_id(self.body())
        if self.id != expected:
            raise ValueError(f"manifest id {self.id} does not match its content ({expected})")
        seen: set[str] = set()
        for row in self.rows:
            if row.episode_id in seen:
                raise ValueError(f"episode {row.episode_id} appears twice")
            seen.add(row.episode_id)
            problem = self._row_problem(row)
            if problem is not None:
                raise ValueError(f"row {row.episode_id} {problem}")
        live = {row.episode_id: row for row in self.rows if row.mode == "live"}
        for row in self.rows:
            paired = live.get(row.replica_of) if row.replica_of is not None else None
            if row.replica_of is not None and (
                row.mode != "sandbox"
                or paired is None
                or (paired.store_id, paired.task_id) != (row.store_id, row.task_id)
            ):
                raise ValueError(
                    f"row {row.episode_id} names {row.replica_of} as its live row, which is not a live row of its "
                    "store and task in this manifest"
                )
        for env, world in self.worlds.items():
            if world.profile not in self.profiles:
                raise ValueError(f"world {env} names a profile the manifest does not hold")
        self._pairs_sealed()
        self._publishable_sealed()
        return self

    def _publishable_sealed(self) -> None:
        """A publishable manifest (D85) holds nothing a live store gave: every row a sandbox or reference row on a
        twin of the plan's catalog, no pair or hand-off, and every world free of a capture and of the records live
        sessions leave (search memory, agent surface, sold-out rate, store replies). The licence is the runner's and
        the builder's to check (carttrace.collect.open_lane), since only they can read it."""
        if not self.plan.publishable:
            return
        for row in self.rows:
            if row.mode not in WORLD_MODES or row.replica_of is not None or row.pair_id is not None or row.handoff:
                raise ValueError(
                    f"row {row.episode_id} is a {row.mode} row, a live pair or a hand-off in a publishable manifest "
                    "(D85): only sandbox and reference rows on licensed twins may be published"
                )
        for env, world in self.worlds.items():
            problem = publishable_world_problem(world, self.plan.catalog)
            if problem is not None:
                raise ValueError(f"world {env} of a publishable manifest (D85) {problem}")

    def _row_problem(self, row: ManifestRow) -> str | None:  # noqa: PLR0911 - one rule per return
        """What makes a row unrunnable or unsafe, or None: a table it names is missing, a sandbox row has no world
        (or a live or playground row has one), a live row would buy, take a market context it cannot, or run on a
        surface other than carttrace.live's."""
        if row.store_id not in self.stores or row.task_id not in self.tasks.get(row.store_id, {}):
            return "names a store or task the manifest does not hold"
        if row.profile not in self.profiles:
            return "names a profile the manifest does not hold"
        needs_world = row.mode in WORLD_MODES
        if needs_world != (row.envpack is not None) or (row.envpack and row.envpack not in self.worlds):
            return "lacks its sandbox world, or is a live or playground row with one"
        live = _live_row_problem(row) if row.mode == "live" else None
        if live is not None:
            return f"is a live row {live}"
        if row.mode == "playground" and row.platform_pack is not None:
            return "is a playground row under a platform pack: a store we own speaks its own platform (D44)"
        if (row.mode == "playground") != is_playground_id(row.store_id):
            return "mixes a playground store and a mode that is not playground: playground stores take playground rows"
        if row.mode != "live" and row.market_context is not None:
            return "names a market context, which only a live row takes"
        if row.rail is not None and row.mode not in WORLD_MODES:
            return "names a payment rail, which only a sandbox or reference row takes (D66)"
        if row.rail is not None and not valid_cell(row.surface.kind if row.surface else None, row.rail):
            return f"pays over {row.rail} on {_kind_of(row.surface)}, which is no cell of the matrix (D89)"
        if (
            row.rail is not None
            and rail(row.rail).handler
            and not shop_pay_platform(row.platform_pack, self.stores[row.store_id].platform)
        ):
            pack = row.platform_pack
            where = f"the {pack} platform pack" if pack else f"its own {self.stores[row.store_id].platform} platform"
            return f"pays over {row.rail} under {where}: Shop Pay is Shopify's (D89)"
        return None

    def _pairs_sealed(self) -> None:
        """Every pair id is on exactly one live row and the one replica row that names it, and is the id of what
        they share, so a reader joining by pair id never joins two episodes of different tasks or conditions."""
        pairs: dict[str, list[ManifestRow]] = {}
        for row in self.rows:
            if row.pair_id is not None:
                pairs.setdefault(row.pair_id, []).append(row)
        for pair_id, members in pairs.items():
            replica = next((r for r in members if r.mode == "sandbox"), None)
            partner = next((r for r in members if r.mode == "live"), None)
            if (
                len(members) != 2
                or replica is None
                or partner is None
                or replica.replica_of != partner.episode_id
                or pair_id != pair_id_of(partner)
            ):
                raise ValueError(
                    f"pair {pair_id} is not one live row and the one replica row that names it, sharing the store, "
                    "task, condition, repeat, seed and market"
                )

    def body(self) -> dict[str, Any]:
        """Everything but the id: what the id hashes."""
        return self.model_dump(mode="json", by_alias=True, exclude={"id"})

    def store(self, row: ManifestRow) -> StoreRef:
        """The row's store, with the row's delivery market. A row on a stand-in of the open pools (D100) names the
        stand-in's own `.example` domain, the one its surfaces serve, never the real store it stands in for: the
        header, the agent's "Store:" line and every link then name one store, and none of them a real one."""
        store = self.stores[row.store_id].model_copy(update={"market": row.market})
        world = self.world(row) if row.mode in WORLD_MODES else None
        if world is not None and presents(world):
            return store.model_copy(update={"domain": stand_in_domain(world.category, world.snapshot_seed)})
        return store

    def task(self, row: ManifestRow) -> TaskRef:
        """The row's task, in its world's ids: a stand-in presents its products under its pack's id scheme (D106,
        Shopify gids under the Shopify pack), so its row's task names them so too."""
        task = self.tasks[row.store_id][row.task_id]
        world = self.world(row) if row.mode in WORLD_MODES else None
        if world is None or not presents(world):
            return task
        scheme = id_scheme_for(world.platform_pack)
        return TaskRef.model_validate(presented_ids_in(task.model_dump(mode="json"), scheme))

    def shopper_map(self, row: ManifestRow) -> dict[str, Any] | None:
        """The row's shopper decision map (D115), or None when the manifest holds none for its world and task."""
        if row.envpack is None:
            return None
        return self.shopper_maps.get(f"{row.envpack}/{row.task_id}")

    def profile(self, row: ManifestRow) -> ProtocolProfile:
        return self.profiles[row.profile]

    def world(self, row: ManifestRow) -> World | None:
        return None if row.envpack is None else self.worlds[row.envpack]

    def surface(self, row: ManifestRow) -> RowSurface:
        """The surface the row runs on: the one it names, else UCP over MCP at its profile's release."""
        return row.surface or RowSurface(kind=DEFAULT_SURFACE, version=self.profiles[row.profile].ucp)

    def counts(self) -> dict[str, int]:
        """Rows per mode, in mode order."""
        return {mode: n for mode in RUN_MODES if (n := sum(r.mode == mode for r in self.rows))}

    def to_text(self) -> str:
        """The file's bytes: canonical key order, indented, newline-terminated."""
        return pretty(self.model_dump(mode="json", by_alias=True))


def publishable_world_problem(world: World, catalog: str) -> str | None:
    """Why a world cannot be in a publishable manifest of `catalog` (completing "world ... "), or None: it must be a
    twin cut from that catalog, with nothing a live session or a capture recorded. A twin whose products were matched
    to its store's capture (`pick_from`, D104) is still one: the capture chose which open products it holds, and none
    of the capture's content is in it."""
    live_records = (world.search_memory, world.agent_surface, world.sold_out, world.store_replies)
    rules = (
        (world.capture is not None or world.catalog == REPLICA, f"is a replica of capture {world.capture}"),
        (world.catalog != catalog, f"is cut from catalog {world.catalog}, not the plan's {catalog}"),
        (any(record is not None for record in live_records), "carries records of live sessions"),
    )
    return next((reason for broken, reason in rules if broken), None)


def _live_row_problem(row: ManifestRow) -> str | None:
    """Why a live row may not be in a manifest, completing "is a live row ...", or None when it may."""
    if authorises_checkout(row.condition):
        return "whose condition authorises an order (D3)"
    if row.platform_pack is not None:
        return "under a platform pack: a live store speaks its own platform (D42)"
    if row.rail is not None:
        return "on a payment rail: nothing is paid on a live store (D66)"
    if row.surface is not None and row.surface.kind not in LIVE_SURFACES:
        return f"on {row.surface.kind}: a live store is reached only through carttrace.live's UCP over MCP, read-only"
    return None


class ServedPrices(_Frozen):
    """What one store's price probe saw before the round (carttrace.live.prices): its currency to us, when, and how it
    ended; the episode header of each of its live rows names the currency (`served_currency`)."""

    currency: str | None = None
    """The currency the store prices us in: its probe cart's, else its catalog's."""
    catalog_currency: str | None = None
    cart_currency: str | None = None
    observed_at: str = ""
    ended: str = ""
    products: int = Field(default=0, ge=0)


def manifest_id(body: Mapping[str, Any]) -> str:
    return f"mf-{digest(body, 12)}"


def pair_id_of(row: ManifestRow) -> str:
    """The id of the pair a row belongs to: what a live row and its replica row share (store, task, condition, repeat,
    seed and market), and nothing of the mode, world or profile that tells them apart, so both compute the same id
    and a pair keeps it when the replica is rebuilt from a newer capture."""
    return stable_id(
        "pair", row.store_id, row.task_id, row.condition.model_dump(mode="json"), row.repeat, row.seed, row.market
    )


def profile_key(profile: ProtocolProfile) -> str:
    """A profile's key in a manifest: its whole content (source and note too), where its version id is not."""
    return stable_id("pk", profile.model_dump(mode="json"))


def seal(**parts: Any) -> Manifest:
    """A Manifest from its parts (plain JSON), with the id computed from them. Empty coverage and served tables are
    left out, as the manifest's own bytes leave them out."""
    optional = ("coverage", "served", "shopper_maps")
    body = {"schema": MANIFEST_SCHEMA, **{k: v for k, v in parts.items() if k not in optional or v}}
    return Manifest.model_validate({**body, "id": manifest_id(body)})


@dataclass(slots=True)
class _Draft:
    """The tables and rows while a manifest is being built."""

    stores: dict[str, StoreRef] = field(default_factory=dict)
    tasks: dict[str, dict[str, TaskRef]] = field(default_factory=dict)
    profiles: dict[str, ProtocolProfile] = field(default_factory=dict)
    worlds: dict[str, World] = field(default_factory=dict)
    groups: list[list[ManifestRow]] = field(default_factory=list)
    notes: dict[str, None] = field(default_factory=dict)
    """Notes in the order first made, once each."""
    coverage: dict[str, dict[str, StoreCoverage]] = field(default_factory=dict)

    def note(self, text: str) -> None:
        self.notes.setdefault(text, None)

    def absorb(self, other: _Draft) -> None:
        """Another draft's part added after this one's, as if it had been planned into this draft (D72): notes once
        each in the order first made, tables by their keys, row groups after these."""
        for text in other.notes:
            self.note(text)
        for key, ref in other.stores.items():
            self.stores.setdefault(key, ref)
        for store_id, table in other.tasks.items():
            self.add_tasks(store_id, list(table.values()))
        for key, profile in other.profiles.items():
            self.profiles.setdefault(key, profile)
        for env, world in other.worlds.items():
            self.worlds.setdefault(env, world)
        self.groups.extend(other.groups)
        for store_id, kinds in other.coverage.items():
            self.coverage.setdefault(store_id, {}).update(kinds)

    def profile(self, profile: ProtocolProfile) -> str:
        key = profile_key(profile)
        self.profiles.setdefault(key, profile)
        return key

    def add_tasks(self, store_id: str, tasks: Sequence[TaskRef]) -> None:
        table = self.tasks.setdefault(store_id, {})
        for task in tasks:
            if table.setdefault(task.task_id, task) != task:
                raise ValidationError(
                    f"generate_tasks gave store {store_id} two different tasks with id {task.task_id}",
                    hint="a task id names one task per store; put what makes two tasks differ into their ids",
                )


def build_manifest(
    plan: ManifestPlan,
    *,
    stores: Sequence[Store],
    categories: CategoryCatalog,
    tasks: GenerateTasks,
    presets: Mapping[str, ProtocolProfile],
    worlds: SandboxWorlds | None = None,
    results: Mapping[str, StoreResult] | None = None,
    profiles: Mapping[str, ProtocolProfile] | None = None,
    fixtures: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    sources: Mapping[str, str] | None = None,
    personas: PersonaBook | None = None,
    surfaces: Collection[str] | None = None,
    voice: Voicer | None = None,
    fixture_notes: Mapping[str, str] | None = None,
    served: Mapping[str, Mapping[str, Any]] | None = None,
    playgrounds: Mapping[str, Sequence[str]] | None = None,
    jobs: int = 1,
) -> Manifest:
    """The manifest for `plan` over `stores` (the queue, in queue order).

    `results` are the folded preflight results by store id (carttrace.stores.load_results), whose captured profiles a
    store's rows speak; `profiles` are profiles captured from stores, by store id, which come first; `fixtures` are
    live stores' products by store id (UCP product dicts, from a static scan), which live tasks are written around;
    `presets` are the named profiles (carttrace.protocols.load_presets), the plan's being the fallback for a store with no
    profile of its own; `personas` are the shoppers checkout rows pay as and tasks are crossed with
    (carttrace.agents.load_personas), needed when the plan has the checkout condition or names personas; `surfaces` are the
    sandbox surface kinds that have an implementation (the sandbox.surface registry when None); `voice` rewrites the
    kept tasks' openings when the plan's voice is model (the caller binds the model runner); `fixture_notes` are
    notes on a store's fixtures (the listings they left out, D38), made for each store live rows are planned for;
    `served` is each store's price probe (the served fixtures' metadata, carttrace.collect.livepool.served_table), kept for
    the stores with live rows; `playgrounds` are the surfaces of each playground store (D44, data/playgrounds), whose
    rows a plan with the playground mode plans on them; `jobs` plans that many stores at once in forked processes
    (D72), each store's part merged back in queue order, so the manifest is the same bytes whatever it is.
    Deterministic apart from a voice pass: the only other side effect is `worlds` saving the packs it builds, when it
    has a directory. Raises ValidationError for a plan that cannot be built at all (sandbox rows without worlds, an
    unknown preset, store, category or persona) and SafetyError for a restricted catalog that is not allowed; a store
    that cannot take part in a mode is left out with a note.
    """
    sandbox_modes = [mode for mode in plan.modes if mode in WORLD_MODES]
    if sandbox_modes and worlds is None:
        raise ValidationError("sandbox and reference rows need sandbox worlds", hint="pass worlds=SandboxWorlds(...)")
    if plan.personas is not None and personas is None:
        raise ValidationError(
            "crossing personas needs the personas book", hint="pass personas=carttrace.agents.load_personas()"
        )
    if plan.voice == "model" and voice is None:
        raise ValidationError(
            "a model voice needs a voicer", hint="pass voice=, as carttrace.collect.inputs.make_manifest does"
        )
    buyer = _persona(plan, personas)
    base = _preset(presets, plan.profile)
    target = _preset(presets, plan.variant) if "version_variant" in plan.conditions else None
    if sandbox_modes and worlds is not None:
        worlds.check_catalog(plan.catalog)
    registered = (
        frozenset(surfaces)
        if surfaces is not None
        else (registered_surfaces() if plan.surfaces or plan.rails else None)
    )
    job = _StoreJob(
        plan=plan,
        categories=categories,
        tasks=tasks,
        worlds=worlds if sandbox_modes else None,
        results=results or {},
        profiles=profiles,
        fixtures=fixtures or {},
        personas=personas,
        voice=voice,
        playgrounds=playgrounds or {},
        fixture_notes=fixture_notes,
        base=base,
        target=target,
        buyer=buyer,
        registered=registered or frozenset(),
    )
    draft = _Draft()
    for planned in _planned(job, _selected(stores, plan, categories), jobs):
        draft.absorb(planned)
    rows = _limited(draft.groups, plan.limit)
    if plan.pair_replica:
        rows = _paired(rows, draft)
    used_stores = {row.store_id for row in rows}
    live_stores = {row.store_id for row in rows if row.mode == "live"}
    used_tasks = {(row.store_id, row.task_id) for row in rows}
    used_worlds = {row.envpack for row in rows if row.envpack}
    used_profiles = {row.profile for row in rows} | {draft.worlds[env].profile for env in used_worlds}
    return seal(
        plan=plan.model_dump(mode="json"),
        sources=dict(sorted((sources or {}).items())),
        stores={sid: ref.model_dump(mode="json") for sid, ref in sorted(draft.stores.items()) if sid in used_stores},
        tasks={
            sid: {tid: task.model_dump(mode="json") for tid, task in sorted(table.items()) if (sid, tid) in used_tasks}
            for sid, table in sorted(draft.tasks.items())
            if sid in used_stores
        },
        profiles={key: p.model_dump(mode="json") for key, p in sorted(draft.profiles.items()) if key in used_profiles},
        worlds={env: w.model_dump(mode="json") for env, w in sorted(draft.worlds.items()) if env in used_worlds},
        rows=[row.model_dump(mode="json") for row in rows],
        notes=list(draft.notes),
        coverage={
            sid: {kind: c.model_dump(mode="json") for kind, c in sorted(table.items())}
            for sid, table in sorted(draft.coverage.items())
            if sid in used_stores
        },
        served={
            sid: ServedPrices.model_validate(dict(entry)).model_dump(mode="json")
            for sid, entry in sorted((served or {}).items())
            if sid in live_stores
        },
    )


@dataclass(frozen=True, slots=True)
class _StoreJob:
    """What planning one store reads, the same for every store of a build: shared by forked workers (D72)."""

    plan: ManifestPlan
    categories: CategoryCatalog
    tasks: GenerateTasks
    worlds: SandboxWorlds | None
    results: Mapping[str, StoreResult]
    profiles: Mapping[str, ProtocolProfile] | None
    fixtures: Mapping[str, Sequence[Mapping[str, Any]]]
    personas: PersonaBook | None
    voice: Voicer | None
    playgrounds: Mapping[str, Sequence[str]]
    fixture_notes: Mapping[str, str] | None
    base: ProtocolProfile
    target: ProtocolProfile | None
    buyer: tuple[str, frozenset[str]] | None
    registered: frozenset[str]


def _plan_store(job: _StoreJob, store: Store) -> _Draft:
    """One store's part of the manifest, in a draft of its own: its notes, tables and row groups."""
    job = _with_spec(job, store.id)
    plan, worlds = job.plan, job.worlds
    draft = _Draft()
    store = _pooled_home(plan, store, worlds, draft)
    category = job.categories.get(store.category_slug)
    captured = _captured(plan, store, category, worlds) if worlds is not None else None
    spec = plan.store_specs.get(store.id)
    # a store planned under its own spec (a stand-in) speaks that spec alone, never what the real store's capture or
    # preflight found: one protocol set a store, and nothing of the real store's identity in a publishable world
    protocols = (
        StoreProtocols(job.base, f"preset {plan.profile} as the store's spec ({spec.platform} pack)")
        if spec is not None
        else store_protocols(
            store.id,
            fallback=job.base,
            fallback_name=plan.profile,
            explicit=job.profiles,
            captured=captured,
            result=job.results.get(store.id),
        )
    )
    if plan.extended:
        draft.note(protocols.note(store.id))
    roster, faulting = _roster(plan, job.personas, store, draft)
    planner = _StorePlanner(
        plan,
        store,
        category,
        draft,
        protocols,
        job.target,
        job.buyer,
        job.registered,
        roster=roster,
        voice=job.voice,
        persona_markets=_markets(job.personas) if plan.boundary == "order" else {},
        faulting=faulting,
        book=job.personas,
    )
    store_fixtures = [dict(f) for f in job.fixtures.get(store.id, ())]
    if is_playground_id(store.id):
        _plan_playground(planner, job.tasks, store_fixtures, job.playgrounds.get(store.id))
        return draft
    if "playground" in plan.modes:
        draft.note(f"{store.id}: no playground rows: it is not a playground (data/playgrounds/stores.json)")
    if "live" in plan.modes:
        _note_fixtures(draft, store.id, job.fixture_notes)
        _plan_live(planner, job.tasks, store_fixtures, job.results.get(store.id))
    if worlds is not None:
        planner.sandbox([mode for mode in plan.modes if mode in WORLD_MODES], job.tasks, worlds)
    return draft


def _with_spec(job: _StoreJob, store_id: str) -> _StoreJob:
    """The job as one store is planned under its own spec (`ManifestPlan.store_specs`): one pack, its versions, its
    broker, and a profile that declares every rail it pays on, so all of the store's worlds speak one protocol set."""
    spec = job.plan.store_specs.get(store_id)
    if spec is None:
        return job
    update: dict[str, Any] = {
        "platforms": [spec.platform],
        "draw_platforms": False,
        "rail_versions": {k: list(v) for k, v in spec.rail_versions.items()},
        "psps": [spec.psp] if spec.psp is not None else [],
    }
    plan = job.plan.model_copy(update=update)
    return replace(job, plan=plan, base=spec.profile(job.base, plan.rails))


_FORKED: tuple[_StoreJob, Sequence[Store]] | None = None
"""The job a forked worker plans from, set in the parent just before the pool forks (closures do not pickle)."""


@dataclass(frozen=True, slots=True)
class _Failed:
    """An MCError raised in a worker, carried back whole: a pickled MCError loses its hint."""

    kind: type[MCError]
    message: str
    hint: str


def _plan_forked(at: int) -> _Draft | _Failed:
    if _FORKED is None:
        raise RuntimeError("a planning worker started without its job")
    job, stores = _FORKED
    try:
        return _plan_store(job, stores[at])
    except MCError as exc:
        return _Failed(type(exc), exc.message, exc.hint)


def _planned(job: _StoreJob, stores: Sequence[Store], jobs: int) -> Iterator[_Draft]:
    """Each store's draft in queue order: planned here one after another, or `jobs` at a time in forked workers
    (D72: round 28 of 2026-10-14 built its 11 stores one after another for over 30 minutes while the GPUs waited).
    A worker's error is raised here as the store's own, as the build in one process raises it."""
    global _FORKED  # noqa: PLW0603 - the only way to hand a closure to a forked worker
    if jobs <= 1 or len(stores) <= 1:
        for store in stores:
            yield _plan_store(job, store)
        return
    _FORKED = (job, stores)
    try:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=min(jobs, len(stores)), mp_context=context) as pool:
            for found in pool.map(_plan_forked, range(len(stores))):
                if isinstance(found, _Failed):
                    error = found.kind.__new__(found.kind)
                    MCError.__init__(error, found.message, hint=found.hint)
                    raise error
                yield found
    finally:
        _FORKED = None


def _plan_playground(
    planner: _StorePlanner, tasks: GenerateTasks, fixtures: list[dict[str, Any]], surfaces: Sequence[str] | None
) -> None:
    """A store we own (D44): playground rows only, never live rows, never a sandbox world of it."""
    if "playground" in planner.plan.modes and surfaces is not None:
        planner.playground(tasks, fixtures, surfaces)
    else:
        planner.draft.note(f"{planner.store.id}: a playground store takes playground rows only")


def _plan_live(
    planner: _StorePlanner, tasks: GenerateTasks, fixtures: list[dict[str, Any]], result: StoreResult | None
) -> None:
    """A store's live rows: on UCP over MCP when its preflight found it, else on its website when the plan names that
    surface and the preflight allows (D31), else a note saying why none."""
    plan, store = planner.plan, planner.store
    reason = live_ineligible(store, result, require_preflight=plan.require_preflight)
    website = WEBSITE in plan.surfaces and reason is not None
    why_not = website_ineligible(result, require_preflight=plan.require_preflight) if website else None
    if reason is None:
        planner.live(tasks, fixtures)
    elif website and why_not is None:
        planner.live_website(tasks, fixtures)
    else:
        planner.draft.note(f"{store.id}: no live rows: {why_not or reason}")


def _note_fixtures(draft: _Draft, store_id: str, notes: Mapping[str, str] | None) -> None:
    """The note on a store's live fixtures (the listings they left out, D38), when there is one."""
    if notes and store_id in notes:
        draft.note(notes[store_id])


def _pooled_home(plan: ManifestPlan, store: Store, worlds: SandboxWorlds | None, draft: _Draft) -> Store:
    """The store as a plan on the open pools stands it in (D91): in the market `pool_market` draws for it, PH when its
    products are Lazada's, so they are sold where they are priced. Any other plan, or a store the pools do not cover,
    keeps its own market."""
    if plan.catalog != OPEN or worlds is None or worlds.pool_market is None:
        return store
    home = store.market_code or DEFAULT_HOME_MARKET
    try:
        chosen = worlds.pool_market(store.category_slug, store.id, seed=plan.seed, home=home)
    except NotFoundError:
        return store
    if chosen == home:
        return store
    draft.note(f"{store.id}: its stand-in store is a {chosen} store of the open pools, priced in that market (D91)")
    return store.model_copy(update={"market_code": chosen})


def _captured(plan: ManifestPlan, store: Store, category: Category, worlds: SandboxWorlds) -> CapturedStore | None:
    """The store's protocols from its replica capture, when the plan's catalog snapshot of it is one. A snapshot that
    cannot be cut gives none here, and neither does a store in a market the sandbox lacks; building the store's world
    says why in a note."""
    if plan.catalog == REPLICA or sandbox_market(store.market_code or DEFAULT_HOME_MARKET) is None:
        return None  # a replica plan's profile is its capture's, read where the world is built
    recipe = World(
        store_id=store.id,
        catalog=plan.catalog,
        category=category.slug,
        snapshot_seed=derived_seed("world", plan.seed, store.id),
        size=plan.catalog_size,
        market=store.market_code or DEFAULT_HOME_MARKET,
        profile="",
        pick_from=worlds.pick_capture_of(store.id, category.slug) if plan.catalog == OPEN else None,
        home=(store.market_code or DEFAULT_HOME_MARKET) if plan.catalog == OPEN else None,
    )
    try:
        return worlds.captured(recipe)
    except (NotFoundError, ValidationError):
        return None


Variant = tuple[Condition, str, str]
"""A condition with the delivery market and the profile key it runs under."""
Placed = tuple[Condition, str, str, str | None, RowSurface | None, str | None, bool]
OnRail = tuple[str | None, bool | None, str | None, str | None]
"""A placed variant's payment rail, whether it is adopted, its crossed rail version and its broker (D66, D71)."""
NO_RAIL: OnRail = (None, None, None, None)
"""A variant placed in its world (None for a live row), on its surface (None when the plan names no surfaces), under
its platform pack (None when the plan crosses none), and whether the world is the store's own (it may pair)."""


RAIL_AXIS = "rail"
PLATFORM_AXIS = "platform"
"""The axes a plan may draw per row instead of crossing (D72), in the order they are drawn."""
LATER = "later:"
"""The prefix of a placeholder for a world built when a row is planned on it (D72); never in a row."""


@dataclass(frozen=True, slots=True)
class _MaskOnRail:
    """A masked world on a payment rail, built only when a row is planned on it (D72): the rail's world of the
    platform pack, given the cut already made on the platform's world (`base`, cut as `masked`)."""

    worlds: SandboxWorlds
    market: str
    key: str
    kind: str
    platform: str | None
    adopted: tuple[str, ...]
    base: EnvPack
    masked: MaskedWorld
    where: Mapping[str, Any]


class _Design:
    """One store's exposure design while it is placed: the rows' kind, the store's surfaces and protocols, and what
    has been placed, each placement's payment rail and whether it is adopted, and the worlds per design."""

    def __init__(self, kind: str, surfaces: Sequence[str | None], protocols: StoreProtocols) -> None:
        self.kind = kind
        self.surfaces = surfaces
        self.protocols = protocols
        self.placed: list[Placed] = []
        self.rails: list[OnRail] = []
        self.designs: dict[str, int] = {}


@dataclass(slots=True)
class _StorePlanner:
    """One store's part of the manifest: its tasks and its rows in every mode the plan asks for."""

    plan: ManifestPlan
    store: Store
    category: Category
    draft: _Draft
    protocols: StoreProtocols
    target: ProtocolProfile | None
    buyer: tuple[str, frozenset[str]] | None = None
    """The persona checkout rows pay as, and the delivery markets it has an address in."""
    registered: frozenset[str] = frozenset()
    """The sandbox surface kinds that have an implementation."""
    roster: Sequence[str] | None = None
    """The personas this store's tasks are crossed with, in the order tasks take them; None writes every task for the
    plan's persona, as every plan did."""
    voice: Voicer | None = None
    """Rewrites the kept tasks' openings in their personas' voices; None keeps the generator's words."""
    built: dict[tuple[str, str, str | None], BuiltWorld] = field(default_factory=dict)
    recipes: dict[tuple[str, str, str | None], World] = field(default_factory=dict)
    """The recipe of each world in `built`, by the same key, which an exposure mask is added to."""
    live_tasks: list[TaskRef] = field(default_factory=list)
    replica_only: list[TaskRef] = field(default_factory=list)
    """The tasks left off the live rows because only the replica can run them (faults, needs_replica); a paired plan
    runs them on its replica rows, unpaired."""
    live_rows: dict[tuple[str, int, str, str], str] = field(default_factory=dict)
    """(task, repeat, condition, market) to the live row's episode id, for the replica rows that pair with it."""
    task_personas: dict[str, str | None] = field(default_factory=dict)
    """Task id to the persona it was written for (None: the plan's), for the rows' condition params."""
    persona_markets: Mapping[str, frozenset[str]] = field(default_factory=dict)
    """Persona to the delivery markets it has an address in, for a plan whose rows end at an order (D66)."""
    axis_uses: dict[str, dict[str, int]] = field(default_factory=dict)
    """Per drawn axis (rail, platform), rows per value drawn so far for this store (D72), which the next draw evens
    out."""
    later: dict[str, _MaskOnRail] = field(default_factory=dict)
    """Placeholder to the masked rail world it stands for, built when a row is planned on it (D72)."""
    resolved: dict[str, str | None] = field(default_factory=dict)
    """Placeholder to the envpack id it was built as, or None when it did not build."""
    faulting: Sequence[str] = ()
    """D80: the personas the payment fault share's tasks are written for; empty when the plan sets no share."""
    book: PersonaBook | None = None
    """The personas, for which rails each one's wallet can pay on (D80)."""
    fault_tasks: set[str] = field(default_factory=set)
    """D80: the tasks of the payment fault share, by id."""
    unpaired: int = 0
    shop_pay: bool = False
    """D80: the store's sandbox worlds serve Shop Pay on UCP (its profile or its live replies declare the handler)."""
    served: StoreProtocols | None = None
    """What the store itself serves, for each sandbox row's `surface_adopted` (D89)."""
    cells: CellDraw | None = None
    """D89: this store's cell quota (carttrace.collect.cells): rows per (surface, rail) cell so far and each task's cells,
    kept so its reference rows run on the cells its sandbox rows did; made on the first draw."""
    no_shop_pay: set[str] = field(default_factory=set)
    """D89: the platforms already noted as carrying no Shop Pay rows for this store."""
    optionless: int = 0
    """D121: drawn tasks left unwritten because they need an option value no pack of the store shows."""

    @property
    def planned(self) -> ProtocolProfile:
        return self.protocols.profile

    @property
    def home(self) -> str:
        return self.store.market_code or DEFAULT_HOME_MARKET

    @property
    def ref(self) -> StoreRef:
        return StoreRef(
            store_id=self.store.id,
            domain=self.store.domain,
            category=self.category.slug,
            platform=self.store.platform_key,
            market=self.home,
        )

    def live_website(self, tasks: GenerateTasks, fixtures: list[dict[str, Any]]) -> None:
        """Live rows on the store's own website (D31): a store that declares no agent protocol is shopped through
        its pages, read-only, ending at a selection. Only conditions a website row can hold are planned: baseline
        and cross_border (detail_hidden filters protocol replies, which a page is not)."""
        surface = RowSurface(kind=WEBSITE, version=WEBSITE_VERSION)
        variants = [v for v in self._variants("live") if v[0].name in WEBSITE_CONDITIONS]
        dropped = sorted({c for c in self.plan.conditions if c not in WEBSITE_CONDITIONS})
        if dropped:
            self.draft.note(f"{self.store.id}: website rows take {', '.join(WEBSITE_CONDITIONS)} only, not {dropped}")
        self.draft.note(f"{self.store.id}: no agent protocol declared; live rows shop its website (D31)")
        generated = self._tasks(tasks, fixtures, "live")
        if generated and variants:
            self.live_tasks = list(generated)
            self._groups(
                generated, ["live"], [(c, m, p, None, surface, None, True) for c, m, p in variants], docs=fixtures
            )

    def playground(self, tasks: GenerateTasks, fixtures: list[dict[str, Any]], surfaces: Sequence[str]) -> None:
        """Rows on a store we own (D44), on each of its surfaces the plan names (all of them when it names none): its
        website, ending at a verified selection, under baseline and cross_border; its UCP REST plugin under those and
        checkout, which may buy, since the store is ours."""
        kinds = [k for k in (self.plan.surfaces or surfaces) if k in surfaces]
        unserved = [k for k in self.plan.surfaces if k not in surfaces]
        if unserved:
            self.draft.note(f"{self.store.id}: the playground serves {', '.join(surfaces)}, not {', '.join(unserved)}")
        if not kinds:
            self.draft.note(f"{self.store.id}: no playground rows: it serves none of the plan's surfaces")
            return
        dropped = sorted({c for c in self.plan.conditions if c not in PLAYGROUND_CONDITIONS})
        if dropped:
            self.draft.note(f"{self.store.id}: playground rows take {', '.join(PLAYGROUND_CONDITIONS)}, not {dropped}")
        generated = self._tasks(tasks, fixtures, "playground")
        if not generated:
            return
        placed: list[Placed] = []
        for condition, market, key in self._variants("playground"):
            for kind in kinds:
                if kind == WEBSITE:
                    if condition.name in WEBSITE_CONDITIONS:
                        placed.append(
                            (
                                condition,
                                market,
                                key,
                                None,
                                RowSurface(kind=WEBSITE, version=WEBSITE_VERSION),
                                None,
                                True,
                            )
                        )
                    continue
                version = surface_version(kind, self.draft.profiles[key])
                if version is None:
                    self.draft.note(f"{self.store.id}: no playground {kind} rows: its profile names no version")
                    continue
                placed.append((condition, market, key, None, RowSurface(kind=kind, version=version), None, True))
        self._groups(generated, ["playground"], placed, docs=fixtures)

    def live(self, tasks: GenerateTasks, fixtures: list[dict[str, Any]]) -> None:
        surfaces: list[RowSurface | None] = [None]
        if self.plan.surfaces:
            kinds = live_protocol_kinds(self.plan.surfaces)
            if not kinds:
                self.draft.note(
                    f"{self.store.id}: no live rows: a live store is reached only through UCP over MCP (carttrace.live), "
                    "which the plan's surfaces leave out"
                )
                return
            others = [kind for kind in self.plan.surfaces if kind not in LIVE_SURFACES]
            if others:
                self.draft.note(
                    f"{self.store.id}: live rows run on ucp_mcp only, read-only; {', '.join(others)} are measured on "
                    "the replica"
                )
            version = self.planned.ucp if self.protocols.from_store else None
            surfaces = [live_row_surface(kind, ucp_version=version) for kind in kinds]
            if LIVE_WEBMCP in kinds:
                self.draft.note(webmcp_note(self.store.id))
        generated = self._tasks(tasks, fixtures, "live")
        if generated:
            self.live_tasks = list(generated)
            self._groups(
                generated,
                ["live"],
                [(c, m, p, None, surface, None, True) for c, m, p in self._variants("live") for surface in surfaces],
                docs=fixtures,
            )

    def sandbox(self, modes: Sequence[RunMode], tasks: GenerateTasks, worlds: SandboxWorlds) -> None:
        kind = "/".join(modes)
        if self.plan.pair_replica and not self.live_tasks:
            # Seen 2026-10-12: a store whose live tasks could not be written (three fixture products) still got ten
            # sandbox rows of its own in a paired pilot. A pair needs its live half, so the store gets no rows.
            self.draft.note(
                f"{self.store.id}: no {kind} rows: no replica pairs, the store has no live rows to pair with"
            )
            return
        if sandbox_market(self.home) is None:
            self.draft.note(f"{self.store.id}: no {kind} rows: the sandbox has no market {self.home}")
            return
        source = self._catalog(worlds, kind)
        if source is None:
            return
        catalog, capture = source
        own = capture.profile if capture is not None else None  # a capture v2 carries none: the preflight's is used
        planned = own if own is not None else self.planned
        home_world = self._world(worlds, self.home, self.draft.profile(planned), kind, catalog=catalog, capture=capture)
        if home_world is None:
            return
        protocols = StoreProtocols(own, "replica capture") if own is not None else self.protocols
        self.served = protocols if protocols.served is not None else replace(protocols, served=self.protocols.served)
        replies = self._replies(worlds)
        self.shop_pay = SHOP_PAY_NS in planned.ucp_payment_handlers or (
            replies is not None and SHOP_PAY_NS in replies.handlers
        )
        if capture is not None and own is not None and self.plan.extended:
            self.draft.note(f"{self.store.id}: {kind} rows speak profile {own.id} of its capture {capture.id}")
        surfaces = self._surfaces(kind, protocols)
        if not surfaces:
            return
        generated = self._replica_tasks(home_world.fixtures, kind, catalog) if self.plan.pair_replica else None
        if generated is None:
            generated = self._tasks(tasks, home_world.fixtures, kind)
        if not generated:
            return
        if self.plan.exposure is not None:
            design = _Design(kind, surfaces, protocols)
            self._exposure(worlds, self.draft.profile(planned), design, catalog=catalog, capture=capture)
            placed, on_rails = design.placed, design.rails if self.plan.rails else None
        else:
            context = (kind, surfaces, protocols)
            placed, crossed = self._conditions(worlds, planned, context, catalog=catalog, capture=capture)
            on_rails = crossed if self.plan.rails else None
        turns = self.plan.exposure is not None
        self._groups(generated, modes, placed, docs=home_world.fixtures, rails=on_rails, surface_turns=turns)

    def _conditions(
        self,
        worlds: SandboxWorlds,
        planned: ProtocolProfile,
        context: tuple[str, Sequence[str | None], StoreProtocols],
        *,
        catalog: str,
        capture: CaptureInfo | None,
    ) -> tuple[list[Placed], list[OnRail]]:
        """Each condition of the plan placed in its world under every platform pack and payment rail, on each of the
        store's surfaces, with each placement's rail and whether it is adopted. `context` is the rows' kind, the
        store's surfaces and its protocols."""
        kind, surfaces, protocols = context
        placed: list[Placed] = []
        on_rails: list[OnRail] = []
        platforms: list[str | None] = list(self.plan.platforms) or [None]
        for condition, market, key in self._variants("sandbox", planned):
            for platform in platforms:
                served = self._platform_key(key, platform)
                rails = self._plan_rails(platform)
                for name, version in [(n, v) for n in rails for v in self._rail_versions(n, served, protocols, kind)]:
                    key_on, added = self._rail_key(served, name, version)
                    world = self._world(
                        worlds, market, key_on, kind, catalog=catalog, capture=capture, platform=platform, adopted=added
                    )
                    if world is None:
                        continue
                    found = self._placed(
                        (condition, market, key_on),
                        world.pack.id,
                        self._railed_surfaces(name, version, surfaces, protocols, kind),
                        kind,
                        platform,
                        own=world.pack.doc["policy"].get("platform_pack") is None,
                    )
                    copies = self._on_rail(worlds, name, version, protocols)
                    placed.extend(found * len(copies))
                    on_rails.extend(at for at in copies for _ in found)
        return placed, on_rails

    def _plan_rails(self, platform: str | None) -> list[str | None]:
        """The plan's rails a world under platform pack `platform` is placed on: Shop Pay only where it is real, under
        the shopify pack or on a Shopify store's own platform (D89), each other platform noted once."""
        rails: list[str | None] = list(self.plan.rails) or [None]
        handlers = [n for n in rails if n is not None and rail(n).handler]
        if not handlers or shop_pay_platform(platform, self.store.platform_key):
            return rails
        where = platform or f"its own platform ({self.store.platform_key})"
        if where not in self.no_shop_pay:
            self.no_shop_pay.add(where)
            self.draft.note(
                f"{self.store.id}: no {', '.join(handlers)} rows under {where}: Shop Pay is Shopify's (D89)"
            )
        return [n for n in rails if n not in handlers]

    def _rail_key(self, key: str, name: str | None, version: str | None = None) -> tuple[str, tuple[str, ...]]:
        """The profile key a world on rail `name` speaks (D66), at `version` when the plan crosses the rail's versions
        (D71), and the protocols it moves for it, if any."""
        if name is None:
            return key, ()
        base = self.draft.profiles[key]
        served = rail_profile(base, name, version)
        return self.draft.profile(served), tuple(sorted(served.differs_from(base)))

    def _railed_surfaces(
        self,
        name: str | None,
        version: str | None,
        surfaces: Sequence[str | None],
        protocols: StoreProtocols,
        kind: str,
    ) -> list[str | None]:
        """The row surfaces of a variant on rail `name` at `version`: a checkout-only UCP release on MCP only (D71)."""
        kinds = list(surfaces) if name is None else self._rail_surfaces(name, protocols, kind)
        return [k for k in kinds if k in (None, "ucp_mcp")] if version in CHECKOUT_ONLY_UCP else kinds

    def _on_rail(
        self, worlds: SandboxWorlds, name: str | None, version: str | None, protocols: StoreProtocols
    ) -> list[OnRail]:
        """What each copy of a variant on rail `name` names: whether it is adopted, its version, and one copy per
        payment broker where the rail pays through the store's PSP and the plan crosses brokers (D71). A handler
        rail (D73) is unadopted where the store's replies record declares the handler even when its profile does not
        (D70: the capture pass never opens a cart, so the record is the newer witness)."""
        adopted = None if name is None else adopted_by(protocols, name)
        if name is not None and adopted and rail(name).handler:
            record = self._replies(worlds)
            if record is not None and rail(name).protocol in record.handlers:
                adopted = False
        brokers: list[str | None] = (
            list(self.plan.psps) if name is not None and rail(name).brokers and self.plan.psps else [None]
        )
        return [(name, adopted, version, broker) for broker in brokers]

    def _replies(self, worlds: SandboxWorlds) -> StoreReplies | None:
        """The store's live replies record (D70), which says which payment handlers it declares; none for a
        publishable plan (D85), whose rows take nothing a live session recorded."""
        return None if self.plan.publishable else worlds.replies_of(self.ref)

    def _rail_versions(self, name: str | None, key: str, protocols: StoreProtocols, kind: str) -> list[str | None]:
        """The versions rail `name` is crossed with for this store (D71): those the plan names that the store's
        profile allows, each other one left out with a note; [None] where the plan names none."""
        wanted = self.plan.rail_versions.get(name or "") if name is not None else None
        if not wanted or name is None:
            return [None]
        base = self.draft.profiles[key]
        out: list[str | None] = []
        for version in wanted:
            why = version_allowed(base, protocols, name, version)
            if why is None:
                out.append(version)
            else:
                self.draft.note(f"{self.store.id}: no {kind} {name} rows at {version}: {why}")
        return out

    def _rail_surfaces(self, name: str, protocols: StoreProtocols, kind: str) -> list[str | None]:
        """The row surfaces rail `name` runs on for this store: its own (adopted where the store lacks it), or the
        plan's UCP bindings the store serves; each one left out with a note."""
        out: list[str | None] = []
        spoken = spoken_surfaces()
        unrailed = sorted(set(self.plan.surfaces) - rail_kinds(self.plan.rails))
        if unrailed:
            self.draft.note(f"surfaces {', '.join(unrailed)}: no payment rail of the plan runs on them (D66)")
        for surface in rail_surfaces(name, self.plan.surfaces):
            if surface is None:
                out.append(None)
            elif surface not in self.registered or surface not in spoken:
                self.draft.note(f"rail {name}: surface {surface} has no sandbox implementation or agent dialect")
            elif not (rail(name).own_surface or protocols.serves(surface) or self._adoptable(surface, protocols)):
                self.draft.note(f"{self.store.id}: no {kind} {surface} rows on rail {name}: the store does not serve")
            else:
                out.append(surface)
        return out

    def _adoptable(self, surface: str, protocols: StoreProtocols) -> bool:
        """D89: on a plan that draws cells, a UCP binding the store does not serve itself is served as adopted where its
        profile speaks UCP (the binding is the same protocol over another transport), and the row says so."""
        return self.plan.draw_cells and surface_version(surface, protocols.profile) is not None

    def _platform_key(self, key: str, platform: str | None) -> str:
        """The profile key a world under `platform` speaks: the store's profile with the platform's versions."""
        if platform is None:
            return key
        return self.draft.profile(platform_profile(self.draft.profiles[key], platform))

    def _catalog(self, worlds: SandboxWorlds, kind: str) -> tuple[str, CaptureInfo | None] | None:
        """The catalog this store's sandbox worlds are cut from, and the capture a replica is built from: the plan's
        catalog and no capture, or for a replica plan the store's newest capture, else the twin catalog it falls back
        to (noted), else None, with a note saying the store has no rows. A paired plan runs on the store's capture
        whatever catalog it names for twins, since the pair is with the replica; with no capture, the twin cut from
        the plan's catalog is a stand-in, which `_replica_tasks` says cannot pair."""
        catalog = self.plan.catalog
        if catalog != REPLICA and not self.plan.pair_replica:
            return catalog, None
        capture = worlds.capture_of(self.ref, self.category.slug)
        if capture is not None:
            if capture.listings:
                self.draft.note(
                    f"{self.store.id}: {capture.listings} listing pages left out of its replica {capture.id}"
                )
            return REPLICA, capture
        if catalog != REPLICA:
            return catalog, None
        twin = worlds.replicas.twin_catalog if worlds.replicas is not None else None
        slug, how = self.category.slug, CAPTURE.format(domain=self.store.domain, category=self.category.slug)
        if twin is None or twin == REPLICA:
            self.draft.note(f"{self.store.id}: no {kind} rows: no capture in {slug}; {how}")
            return None
        worlds.check_catalog(twin)
        self.draft.note(
            f"{self.store.id}: no capture in {slug}, so its {kind} rows run on a twin cut from {twin}; {how}"
        )
        return twin, None

    def _placed(
        self,
        variant: Variant,
        envpack: str,
        surfaces: Sequence[str | None],
        kind: str,
        platform: str | None = None,
        *,
        own: bool,
    ) -> list[Placed]:
        """One variant placed in its world (`envpack`, or a world built later, D72) on each of the store's surfaces
        (unnamed, when the plan names none), under its platform pack; `own` says the world is the store's own."""
        condition, market, key = variant
        out: list[Placed] = []
        for surface in surfaces:
            if surface is None:
                out.append((condition, market, key, envpack, None, platform, own))
                continue
            version = surface_version(surface, self.draft.profiles[key])
            if version is None:
                self.draft.note(f"{self.store.id}: no {kind} {surface} rows under {condition.name}: no version")
                continue
            if surface == WEBSITE and condition.name not in WEBSITE_CONDITIONS:
                self.draft.note(website_condition_note(self.store.id, kind, condition.name))
                continue
            row_surface = RowSurface(kind=surface, version=version)
            out.append((condition, market, key, envpack, row_surface, platform, own))
        return out

    def _surfaces(self, kind: str, protocols: StoreProtocols) -> list[str | None]:
        """The surfaces this store's sandbox rows run on: [None] for a plan that names none (UCP over MCP, unnamed),
        else each listed surface the sandbox implements and the store serves, with a note for each left out."""
        if not self.plan.surfaces:
            return [None]
        out: list[str | None] = []
        spoken = spoken_surfaces()
        for surface in self.plan.surfaces:
            if surface not in self.registered:
                self.draft.note(
                    f"surface {surface}: the sandbox has no implementation registered under sandbox.surface yet, so "
                    "its rows are left out"
                )
            elif surface not in spoken:
                self.draft.note(
                    f"surface {surface}: no agent dialect speaks it yet (carttrace.agents.dialects), so its rows are left out"
                )
            elif not protocols.serves(surface):
                served = ", ".join(protocols.surfaces()) or "none"
                self.draft.note(f"{self.store.id}: no {kind} {surface} rows: the store serves {served}")
            else:
                out.append(surface)
        if not out:
            self.draft.note(f"{self.store.id}: no {kind} rows: it serves none of the plan's surfaces")
        return out

    def _world(
        self,
        worlds: SandboxWorlds,
        market: str,
        key: str,
        kind: str,
        *,
        catalog: str,
        capture: CaptureInfo | None,
        platform: str | None = None,
        adopted: tuple[str, ...] = (),
    ) -> BuiltWorld | None:
        found = self.built.get((market, key, platform))
        if found is not None:
            return found
        recipe = World(
            store_id=self.store.id,
            catalog=catalog,
            category=self.category.slug,
            snapshot_seed=0 if capture is not None else derived_seed("world", self.plan.seed, self.store.id),
            size=capture.products if capture is not None else self.plan.catalog_size,
            market=market,
            profile=key,
            capture=capture.id if capture is not None else None,
            search_memory=worlds.memory_of(self.ref) if capture is not None else None,
            agent_surface=worlds.surface_of(self.ref) if capture is not None else None,
            platform_pack=platform,
            sold_out=worlds.sold_out_of(self.ref, market) if capture is not None else None,
            adopted=list(adopted),
            store_replies=worlds.replies_of(self.ref) if capture is not None and platform is None else None,
            pick_from=worlds.pick_capture_of(self.store.id, self.category.slug) if catalog == OPEN else None,
            agent_view=worlds.view_of(self.ref, platform) if capture is None and catalog != REPLICA else None,
            home=self.home if catalog == OPEN else None,
            union_core=worlds.union_core_ref() if catalog == OPEN and capture is None else None,
        )
        try:
            made = worlds.build(self.ref, recipe, self.draft.profiles[key])
        except (NotFoundError, ValidationError) as exc:
            self.draft.note(f"{self.store.id}: no {kind} rows in market {market}: {exc.message}")
            return None
        worlds.save(made.pack)
        self.draft.worlds[made.pack.id] = recipe
        self.built[(market, key, platform)] = made
        self.recipes[(market, key, platform)] = recipe
        return made

    def _exposure(
        self,
        worlds: SandboxWorlds,
        key: str,
        design: _Design,
        *,
        catalog: str,
        capture: CaptureInfo | None,
    ) -> None:
        """The store's exposure design (D63) placed on its worlds: every mask on every platform pack's world where it
        is honest (each withheld type served there before the cut and nowhere after it), each such world saved with
        its mask on its recipe, and built for each payment rail of the plan (D66) so a row can pay over any of them.
        The types the design spans are those some platform's world shows."""
        if capture is None and not self.plan.publishable:
            self.draft.note(
                f"{self.store.id}: no {design.kind} exposure rows: it has no replica of its own, and a mask withholds "
                "a store's own data from its own products"
            )
            return
        bases: list[tuple[str | None, str, BuiltWorld, World, Served]] = []
        platforms: list[str | None] = list(self.plan.platforms) or [None]
        for named in platforms:
            served_key = self._platform_key(key, named)
            base = self._world(
                worlds, self.home, served_key, design.kind, catalog=catalog, capture=capture, platform=named
            )
            if base is not None:
                recipe = self.recipes[(self.home, served_key, named)]
                bases.append((named, served_key, base, recipe, served_counts(base.pack)))
        shown = ordered(t for *_, before in bases for t in before.counts if before.exposes(t))
        if self.plan.exposure_types:
            chosen = set(self.plan.exposure_types)
            unshown = [t for t in self.plan.exposure_types if t not in shown]
            shown = [t for t in shown if t in chosen]
            if unshown:
                self.draft.note(
                    f"{self.store.id}: chosen exposure types it does not show, so no mask withholds them: "
                    f"{', '.join(unshown)}"
                )
        seed = derived_seed("exposure", self.plan.seed, self.store.id)
        held: set[str] = set()
        where = {"catalog": catalog, "capture": capture}
        # the baseline and each type alone first; random masks then draw only on types a cut withheld somewhere, so
        # no random mask is refused on every platform for a type no cut can remove
        for mask in design_masks(shown, seed=seed, random_masks=0):
            if self._mask(worlds, bases, mask, design, where):
                held.update(mask[1])
        for mask in design_masks(ordered(held), seed=seed, random_masks=self.plan.exposure_random):
            if mask[0] == "random":
                self._mask(worlds, bases, mask, design, where)
        spread = ", ".join(f"{name} {count}" for name, count in design.designs.items())
        self.draft.note(
            f"{self.store.id}: exposure design over {len(shown)} data types it shows ({', '.join(shown) or 'none'}), "
            f"{len(held)} withheld honestly somewhere; worlds by design: {spread or 'none'}"
        )

    def _mask(
        self,
        worlds: SandboxWorlds,
        bases: Sequence[tuple[str | None, str, BuiltWorld, World, Served]],
        mask: tuple[ExposureDesign, tuple[str, ...]],
        design: _Design,
        where: Mapping[str, Any],
    ) -> bool:
        """One mask placed on every platform's world where it is honest, and on that world as each payment rail of
        the plan serves it: a rail changes the world's protocols, never its products, so the check on the platform's
        world holds for each rail's. Each world is saved with the mask on its recipe; whether any was placed."""
        name, withheld = mask
        condition = exposure_condition(name, withheld)
        any_placed = False
        for platform, served_key, base, recipe, before in bases:
            rails = self._plan_rails(platform)
            masked = masked_world(base.pack, before, name, withheld)
            if masked.pack is None or masked.mask is None:
                self.draft.note(
                    f"{self.store.id}: no {design.kind} exposure rows withholding {', '.join(withheld)} under "
                    f"{platform or 'its own platform'}: {masked.refused}"
                )
                continue
            for rail_name, version in [
                (n, v) for n in rails for v in self._rail_versions(n, served_key, design.protocols, design.kind)
            ]:
                key_on, added = self._rail_key(served_key, rail_name, version)
                if rail_name is None:
                    worlds.save(masked.pack)
                    self.draft.worlds[masked.pack.id] = recipe.model_copy(update={"exposure": masked.mask})
                    envpack = masked.pack.id
                else:
                    # D72: the rail's world is built only when a row is planned on it, and it takes the cut made
                    # once on the platform's world, since a rail changes the world's protocols, never its products
                    envpack = self._later(
                        _MaskOnRail(worlds, self.home, key_on, design.kind, platform, added, base.pack, masked, where)
                    )
                kinds = self._railed_surfaces(rail_name, version, design.surfaces, design.protocols, design.kind)
                found = self._placed((condition, self.home, key_on), envpack, kinds, design.kind, platform, own=False)
                copies = self._on_rail(worlds, rail_name, version, design.protocols)
                design.placed.extend(found * len(copies))
                design.rails.extend(at for at in copies for _ in found)
            design.designs[name] = design.designs.get(name, 0) + 1
            any_placed = True
        return any_placed

    def _tasks(self, tasks: GenerateTasks, fixtures: list[dict[str, Any]], kind: str) -> list[TaskRef]:
        # One memo run (carttrace.agents.memo) for every draw of the store's tasks: each persona and spread draw reads the
        # same fixtures, so their products and predicate checks are computed once (2026-10-13, a fifteen-minute
        # manifest of one replica store).
        with remembering():
            return self._tasks_of(tasks, fixtures, kind)

    def _tasks_of(self, tasks: GenerateTasks, fixtures: list[dict[str, Any]], kind: str) -> list[TaskRef]:
        mode = kind if kind in ("live", "playground") else "sandbox"
        seed = derived_seed("tasks", self.plan.seed, self.store.id, mode)
        try:
            with recording_drops() as drops:
                generated = self._generate(tasks, fixtures, seed)
        except MCError as exc:
            self.draft.note(f"{self.store.id}: no {kind} rows: generate_tasks refused: {exc.message}")
            return []
        for why, count in Counter(drops).items():
            # D104: an open-lane slot no draw could fill well, per persona draw that dropped it
            self.draft.note(f"{self.store.id}: task slot dropped ({count} draws): {why.split(' ', 1)[1]}")
        self._note_unwritten(generated, kind)
        if mode in ("live", "playground"):
            # Live pilot 02 (2026-10-12): 9 of 10 substitute rows came out WRONG on live stores, whose stock no sealed
            # fault can touch. A spec with faults, or a custom intent that needs the replica, never gets a live row.
            kept = [task for task in generated if not _replica_only(task)]
            self.replica_only = _unique([task for task in generated if _replica_only(task)])
            if len(kept) < len(generated):
                self.draft.note(
                    f"{self.store.id}: {len(generated) - len(kept)} tasks left off {mode} rows: their specs carry "
                    "sealed faults or a checkout-stage requirement only the replica can hold"
                )
            generated = kept
        unique = _unique(generated)
        if len(unique) < len(generated):
            # Seen 2026-10-11: a store with few qualifying products gave the same task twice, and identical tasks make
            # identical rows, which the manifest refuses. A task id names its content, so keeping the first is exact.
            self.draft.note(f"{self.store.id}: {len(generated) - len(unique)} repeated tasks dropped")
            generated = unique
        if self.plan.selection == "spread":
            generated = self._spread(tasks, fixtures, generated, mode, kind)
        else:
            if self.plan.tasks_per_store is not None:
                generated = generated[: self.plan.tasks_per_store]
            if self.plan.extended and generated:
                facts = product_facts(fixtures)
                self._cover(mode, describe(generated, facts), facts, candidates=len(unique))
        if self.optionless:
            packs = ", ".join(sorted(self.plan.platforms))
            self.draft.note(
                f"{self.store.id}: {self.optionless} drawn task(s) left unwritten: they need an option value the "
                f"{packs} pack does not show (D121)"
            )
            self.optionless = 0
        generated = self._voiced(generated)
        if not generated:
            self.draft.note(f"{self.store.id}: no {kind} rows: generate_tasks gave no tasks")
            return []
        self.draft.stores[self.store.id] = self.ref
        self.draft.add_tasks(self.store.id, generated)
        return generated

    def _note_unwritten(self, generated: Sequence[TaskRef], kind: str) -> None:
        """Name the intents the plan asked for that no product of the store could carry: the generator skipped them
        and refilled their slots from the others (validation round 3, 2026-10-16), so the store keeps its rows."""
        if not self.plan.intents or self.plan.custom_intents:
            return
        asked = chosen_intents(self.plan.intents) or ()
        missing = [i for i in asked if i not in {t.intent for t in generated}]
        if missing:
            self.draft.note(
                f"{self.store.id}: {kind} tasks of {', '.join(missing)} could not be written (no product carries one); "
                "their slots went to the other intents"
            )

    def _generate(self, tasks: GenerateTasks, fixtures: list[dict[str, Any]], seed: int) -> list[TaskRef]:
        """One draw of the store's tasks (`_draw`), less any task that needs an option value where every pack the
        store is served under shows none (D121: on round 46 the Medusa pack's store-011 stand-in lost 24 of its tasks at
        the replay, "the store does not say: color, size", and came in at 65 of 100 rows). Spare and later draws then
        fill the store with tasks it can show."""
        drawn = self._draw(tasks, fixtures, seed)
        if not hides_options(self.plan.platforms):
            return drawn
        shown = [task for task in drawn if not needs_option(task)]
        self.optionless += len(drawn) - len(shown)
        return shown

    def _draw(self, tasks: GenerateTasks, fixtures: list[dict[str, Any]], seed: int) -> list[TaskRef]:
        """One draw of the store's tasks. Without a roster, every task is written for the plan's persona (which pays
        on checkout rows, so the specs' buyer blocks hold its details), or a custom intent for its own persona, under
        the plan's payment fault share when it sets one (D86). With a roster, each slot's task comes from the
        persona the store's roster order gives that slot, round robin, so a store's tasks cross its personas evenly
        and the same inputs give the same crossing."""
        if not self.roster:
            generated = list(tasks(self.store, self.category, fixtures, seed=seed, persona=self.plan.persona))
            for task in generated:
                # a custom intent is written for a persona of its own, named in its buyer block
                own = (task.hidden_spec.get("buyer") or {}).get("persona") if task.intent_kind is not None else None
                self.task_personas[task.task_id] = own if isinstance(own, str) else self.plan.persona
            if self.plan.payment_fault_bp is not None and any(task.intent_kind is not None for task in generated):
                return self._custom_shared(generated)
            return generated
        if self.faulting:
            return self._generate_shared(tasks, fixtures, seed)
        drawn = [list(tasks(self.store, self.category, fixtures, seed=seed, persona=name)) for name in self.roster]
        out: list[TaskRef] = []
        for index in range(min(len(d) for d in drawn)):
            which = index % len(self.roster)
            task = drawn[which][index]
            self.task_personas[task.task_id] = self.roster[which]
            out.append(task)
        return out

    def _generate_shared(self, tasks: GenerateTasks, fixtures: list[dict[str, Any]], seed: int) -> list[TaskRef]:
        """One draw of the store's tasks under a payment fault share (D80): each slot is a fault slot with the plan's
        probability, drawn by seed, and takes the next persona of the fault pool, else the next paying one."""
        assert self.roster is not None
        names = list(dict.fromkeys([*self.roster, *self.faulting]))
        drawn = {n: list(tasks(self.store, self.category, fixtures, seed=seed, persona=n)) for n in names}
        share = self.plan.payment_fault_bp or 0
        taken = {False: 0, True: 0}
        out: list[TaskRef] = []
        for index in range(min(len(d) for d in drawn.values())):
            rng = seeded_rng("payment-fault", self.plan.seed, self.store.id, seed, index)
            fault = rng.randrange(10_000) < share
            pool = self.faulting if fault else self.roster
            name = pool[taken[fault] % len(pool)]
            taken[fault] += 1
            task = drawn[name][index]
            self.task_personas[task.task_id] = name
            if fault:
                self.fault_tasks.add(task.task_id)
            out.append(task)
        return out

    def _custom_shared(self, generated: list[TaskRef]) -> list[TaskRef]:
        """D86: a payment fault share over a store's custom intents, each written for a persona of its own. A task
        whose persona is built to fail, or pays on none of the plan's rails, can only be a fault task; any other pays.
        Each slot is a fault slot with the plan's probability, drawn by seed, and takes the next task of its pool (each
        pool shuffled by seed per store), until the pool a slot needs runs out, so a written intent built to fail
        enters at the share and never as a paying row. The draw does not depend on the task draw's seed: the set is
        what it is, and a spread's later draws give the same tasks."""
        rails: list[str | None] = list(self.plan.rails) or [None]
        paying: list[TaskRef] = []
        failing: list[TaskRef] = []
        for task in generated:
            payer = self._payer(self.task_personas.get(task.task_id))
            built = payer is not None and (built_to_fail(payer, rails) or not pays_somewhere(payer, rails))
            (failing if built else paying).append(task)
        rng = seeded_rng("custom-payment-fault", self.plan.seed, self.store.id)
        rng.shuffle(paying)
        rng.shuffle(failing)
        share = self.plan.payment_fault_bp or 0
        if not paying:
            self.draft.note(f"{self.store.id}: no custom intent is written for a persona built to pay (D86)")
            return []
        if share and not failing:
            self.draft.note(f"{self.store.id}: no custom intent is written for a persona built to fail; all pay (D86)")
        pools = {False: paying, True: failing}
        taken = {False: 0, True: 0}
        out: list[TaskRef] = []
        for index in range(len(generated)):
            draw = seeded_rng("payment-fault", self.plan.seed, self.store.id, CUSTOM_INTENTS, index)
            fault = bool(failing) and draw.randrange(10_000) < share
            if taken[fault] >= len(pools[fault]):
                break
            task = pools[fault][taken[fault]]
            taken[fault] += 1
            if fault:
                self.fault_tasks.add(task.task_id)
            out.append(task)
        if len(out) < len(generated):
            self.draft.note(
                f"{self.store.id}: {len(generated) - len(out)} of {len(generated)} custom intents left out (D86): "
                f"{len(failing)} are written for a persona built to fail, and the fault share takes {taken[True]}"
            )
        return out

    def _voiced(self, generated: list[TaskRef]) -> list[TaskRef]:
        """The kept tasks through the plan's voice, keeping each task's persona under its new id."""
        if self.voice is None or not generated:
            return generated
        voiced = self.voice(self.store.id, generated)
        if len(voiced) != len(generated):
            raise ValidationError(
                f"{self.store.id}: the voice pass returned {len(voiced)} tasks for {len(generated)}",
                hint="a voicer keeps every task, rewritten or not",
            )
        for before, after in zip(generated, voiced, strict=True):
            self.task_personas[after.task_id] = self.task_personas.get(before.task_id, self.plan.persona)
            if before.task_id in self.fault_tasks:
                self.fault_tasks.add(after.task_id)
        return voiced

    def _spread(
        self, tasks: GenerateTasks, fixtures: list[dict[str, Any]], first: list[TaskRef], mode: str, kind: str
    ) -> list[TaskRef]:
        """Tasks spread over the catalog (carttrace.collect.coverage): candidates from the first draw, then from more draws
        until every product of the rotation window has one, then the stratified choice among them."""
        facts = product_facts(fixtures)
        count = self.plan.tasks_per_store or len(first)
        window = wanted_products(
            facts, count=count, rotation=self.plan.rotation, seed=self.plan.seed, store_id=self.store.id
        )
        pool: list[Candidate] = describe(first, facts, draw=0)
        seen = {c.task.task_id for c in pool}
        for draw in range(1, POOL_DRAWS):
            covered = {c.product for c in pool}
            if all(product in covered for product in window):
                break
            seed = derived_seed("tasks", self.plan.seed, self.store.id, mode, draw)
            try:
                more = _unique(self._generate(tasks, fixtures, seed))
            except MCError as exc:
                self.draft.note(
                    f"{self.store.id}: {kind} task draw {draw} refused, choosing among fewer: {exc.message}"
                )
                break
            fresh = [task for task in more if task.task_id not in seen]
            seen |= {task.task_id for task in fresh}
            pool += describe(fresh, facts, draw=draw)
        chosen = spread(
            pool,
            facts,
            count=count,
            cap=self.plan.product_cap,
            rotation=self.plan.rotation,
            seed=self.plan.seed,
            store_id=self.store.id,
        )
        if chosen and len(chosen) < count:
            self.draft.note(
                f"{self.store.id}: {len(chosen)} of {count} {kind} tasks: the catalog gives no more distinct products "
                f"at {self.plan.product_cap} task(s) each"
            )
        if chosen:
            self._cover(mode, chosen, facts, candidates=len(pool))
        return [c.task for c in chosen]

    def _cover(self, mode: str, chosen: Sequence[Candidate], facts: Sequence[ProductFacts], *, candidates: int) -> None:
        coverage = coverage_of(
            chosen,
            facts,
            selection=self.plan.selection,
            window=self.plan.rotation,
            asked=self.plan.tasks_per_store,
            candidates=candidates,
        )
        self.draft.coverage.setdefault(self.store.id, {})[mode] = coverage
        self.draft.note(coverage.line(self.store.id, mode))

    def _replica_tasks(self, fixtures: list[dict[str, Any]], kind: str, catalog: str) -> list[TaskRef] | None:
        """The live rows' tasks this store's sandbox world can run: every product and witness variant they name is in
        the world, as it is in a replica of the store. None, with a note, when the world holds none of them (a
        stand-in), and the sandbox rows get tasks of their own. A store with no live tasks never gets here: `sandbox`
        plans no rows for it."""
        facts = product_facts(fixtures)
        held = {f.id for f in facts} | {v for f in facts for v in f.variants}
        priced = fixture_currency(fixtures)
        runnable = [task for task in self.live_tasks if _held_by(task, held)]
        foreign = [task for task in runnable if _task_currency(task) not in (None, priced)]
        if foreign:
            # D43: live tasks are written in the currency the store served us; a replica priced in its capture's
            # currency cannot run them without a currency mismatch, and nothing is ever converted.
            self.draft.note(
                f"{self.store.id}: {len(foreign)} live tasks have no replica rows: they are priced in "
                f"{', '.join(sorted({str(_task_currency(t)) for t in foreign}))} as the store served us, and its "
                f"{kind} world is priced in {priced or 'no currency'}"
            )
            runnable = [task for task in runnable if task not in foreign]
            if not runnable:
                return None  # the note above says why; the replica rows run tasks of their own, unpaired
        if not runnable:
            self.draft.note(
                f"{self.store.id}: no replica pairs: its {kind} world (catalog {catalog}) does not hold the "
                "live tasks' products, so it is a stand-in, not a replica"
            )
            return None
        missing = sum(1 for task in self.live_tasks if not _held_by(task, held))
        if missing:
            self.draft.note(
                f"{self.store.id}: {missing} live tasks have no replica rows: the world lacks their products"
            )
        if self.plan.extended:
            self._cover("sandbox", describe(runnable, facts), facts, candidates=len(self.live_tasks))
        return runnable + self._replica_only_tasks(held, kind)

    def _replica_only_tasks(self, held: Collection[str], kind: str) -> list[TaskRef]:
        """The tasks only the replica can run (faults, a custom intent that needs_replica), which the live rows left
        off: in a paired plan they still run on the replica rows, with no live half and so no pair id."""
        extra = [task for task in self.replica_only if _held_by(task, held)]
        if self.plan.tasks_per_store is not None:
            extra = extra[: self.plan.tasks_per_store]
        extra = self._voiced(extra)
        if extra:
            self.draft.add_tasks(self.store.id, extra)
            self.draft.note(
                f"{self.store.id}: {len(extra)} replica-only tasks run on {kind} rows unpaired: no live row holds them"
            )
        return extra

    def _variants(
        self, kind: Literal["live", "sandbox", "playground"], profile: ProtocolProfile | None = None
    ) -> list[Variant]:
        """Each condition of the plan this store can run in `kind`, with its market and profile key. `profile` is
        the one the store is planned under when it is not the plan's (a replica speaks its capture's)."""
        own = profile or self.planned
        planned = self.draft.profile(own)
        out: list[Variant] = []
        for name in self.plan.conditions:
            if kind == "playground" and name not in PLAYGROUND_CONDITIONS:
                continue
            if name == "baseline":
                out.append((baseline(), self.home, planned))
            elif name == "cross_border":
                out.extend(self._cross_border(kind, planned))
            elif name == "version_variant":
                out.extend(self._version_variant(kind, own))
            elif name == "checkout":
                out.extend(self._checkout(kind, planned))
            else:
                out.extend((detail_hidden(detail), self.home, planned) for detail in self._details(kind))
        return out

    def _checkout(self, kind: str, planned: str) -> list[Variant]:
        if kind == "live" and self.plan.pair_replica:
            self.draft.note(
                f"{self.store.id}: no live checkout rows: nothing is bought on a live store (D3); its replica rows "
                "take the checkout condition, unpaired"
            )
            return []
        if kind == "live":
            raise ValidationError(
                f"{self.store.id}: a live row cannot take the checkout condition", hint="checkout is sandbox only (D3)"
            )
        if self.buyer is None:
            return []
        name, markets = self.buyer
        home = MARKET_ALIASES.get(self.home.upper(), self.home.upper())
        if home not in markets:
            self.draft.note(f"{self.store.id}: no {kind} checkout rows: persona {name} has no address in {self.home}")
            return []
        return [(checkout(name), self.home, planned)]

    def _cross_border(self, kind: str, planned: str) -> list[Variant]:
        other = cross_border_market(self.home, self.plan.cross_border_markets)
        if other is None or (kind != "live" and sandbox_market(other) is None):
            self.draft.note(f"{self.store.id}: no {kind} cross_border rows: no other market to deliver to")
            return []
        return [(cross_border(self.home, other), other, planned)]

    def _version_variant(self, kind: str, own: ProtocolProfile) -> list[Variant]:
        if kind == "live":
            self.draft.note(f"{self.store.id}: no live version_variant rows: a live store speaks its own versions")
            return []
        if self.target is None:
            return []
        derived = variant_profile(own, self.target, target_name=self.plan.variant)
        if derived is None:
            self.draft.note(f"{self.store.id}: no version_variant rows: {self.plan.variant} speaks the same UCP")
            return []
        return [
            (
                version_variant(own, derived, target_name=self.plan.variant),
                self.home,
                self.draft.profile(derived),
            )
        ]

    def _details(self, kind: str) -> list[HiddenDetail]:
        details, notes = hidden_details(self.category, self.plan.hypotheses)
        for note in notes:
            self.draft.note(f"{self.category.slug}: {note}")
        if not details:
            self.draft.note(
                f"{self.store.id}: no {kind} detail_hidden rows: no hypothesis of its category can be hidden"
            )
        return details

    def _groups(
        self,
        generated: Sequence[TaskRef],
        modes: Sequence[RunMode],
        variants: Sequence[Placed],
        *,
        docs: Sequence[Mapping[str, Any]] = (),
        rails: Sequence[OnRail] | None = None,
        surface_turns: bool = False,
    ) -> None:
        """The rows of every task under every variant, grouped per task and repeat. A detail_hidden variant takes
        only a task it really hides something from (D47, `hides_decisive` over the task's products in `docs`).
        `rails` names each variant's payment rail and whether it is adopted (D66), when the plan crosses rails; a
        plan whose boundary is order makes each world row a checkout row (`_ordered`). With `surface_turns` (an
        exposure design) each task runs on one of the variants' surfaces, in turn by its place in `generated` from a
        start drawn by the store, so every mask and platform of the store meets the same task on the same surface and
        no surface always takes a store's first task."""
        index = {str(d.get("id")): d for d in docs if isinstance(d, Mapping)}
        honest: dict[tuple[str, str], bool] = {}
        left_out = kept = 0
        start = derived_seed("surface-turn", self.store.id)
        for mode in modes:
            for turn, task in enumerate(generated):
                persona = self.task_personas.get(task.task_id)
                fits = self._rail_fit(task.task_id, persona) if rails is not None else None
                fresh = self.cells is None or task.task_id not in self.cells.drawn
                mine = self._members(variants, rails, start + turn if surface_turns else None, fits, task.task_id)
                for repeat in range(self.plan.repeats):
                    seed = derived_seed("row", self.plan.seed, self.store.id, task.task_id, repeat)
                    group = []
                    for at, envpack in self._kept(variants, rails, mine, task.task_id, repeat, fits=fits):
                        condition, market, key, _, surface, platform, own = variants[at]
                        rail_name, adopted, rail_version, psp = rails[at] if rails is not None else NO_RAIL
                        if condition.name == "detail_hidden":
                            probe = (task.task_id, dumps(condition.params))
                            if probe not in honest:
                                honest[probe] = hides_decisive(task.hidden_spec, condition, index)
                            left_out, kept = (left_out, kept + 1) if honest[probe] else (left_out + 1, kept)
                            if not honest[probe]:
                                continue
                        if self.roster is not None or task.intent_kind is not None:
                            # a roster's task, or a custom intent, is written for a persona of its own
                            condition = _with_persona(condition, persona)
                        pair = (task.task_id, repeat, dumps(condition.model_dump(mode="json")), market)
                        paired = mode == "sandbox" and self.plan.pair_replica and _on_live_surface(surface) and own
                        replica_of = self.live_rows.get(pair) if paired else None
                        if mode in WORLD_MODES and replica_of is None and self.plan.boundary == "order":
                            condition = self._ordered(condition, persona, market)
                        row = make_row(
                            store_id=self.store.id,
                            task_id=task.task_id,
                            condition=condition,
                            profile=key,
                            mode=mode,
                            repeat=repeat,
                            seed=seed,
                            market=market,
                            envpack=envpack,
                            handoff=self.plan.handoff and mode == "live" and not _on_website(surface),
                            surface=surface,
                            replica_of=replica_of,
                            platform_pack=platform,
                            market_context=self.plan.market_context if mode == "live" else None,
                            rail=rail_name,
                            adopted=adopted,
                            rail_version=rail_version,
                            psp=psp,
                            payment_fault=self._planned_fault(condition, rail_name),
                            surface_adopted=self._surface_adopted(surface, rail_name),
                        )
                        if mode == "live":
                            self.live_rows[pair] = row.episode_id
                        self._count_cell(row, fresh=fresh)
                        group.append(row)
                    if group:
                        self.draft.groups.append(group)
        self._note_unpaired()
        if left_out:
            self.draft.note(
                f"{self.store.id}: {left_out} detail_hidden rows left out, {kept} kept (D47): a row is planned only "
                "when the condition hides its task's decisive detail and the task's products state it"
            )

    def _drawn(
        self, variants: Sequence[Placed], rails: Sequence[OnRail] | None, task_id: str, repeat: int
    ) -> set[int] | None:
        """On a plan that draws its rail versions and brokers (`draw_axes`, D71): the variants a task's row keeps,
        one (version, broker) per variant that differs in nothing else, drawn by the plan's seed, the store, the task
        and the repeat, so every version and broker is in use and rows do not multiply. None keeps every variant."""
        if not self.plan.draw_axes or rails is None:
            return None
        groups: dict[str, list[int]] = {}
        for at, (condition, market, _, _, surface, platform, _) in enumerate(variants):
            kind = surface.kind if surface is not None else None
            base = dumps([condition.model_dump(mode="json"), market, kind, platform, rails[at][0]])
            groups.setdefault(base, []).append(at)
        kept: set[int] = set()
        for base, members in groups.items():
            choices = sorted({(rails[at][2] or "", rails[at][3] or "") for at in members})
            pick = choices[
                seeded_rng("axes", self.plan.seed, self.store.id, task_id, repeat, base).randrange(len(choices))
            ]
            kept.update(at for at in members if (rails[at][2] or "", rails[at][3] or "") == pick)
        return kept

    def _kept(
        self,
        variants: Sequence[Placed],
        rails: Sequence[OnRail] | None,
        members: Sequence[int],
        task_id: str,
        repeat: int,
        *,
        fits: Callable[[str | None], bool] | None = None,
    ) -> list[tuple[int, str | None]]:
        """The places among `members` a task's row keeps once its rail version, broker and platform are drawn, each
        with its world's envpack id; a variant whose world did not build is dropped (its note says why)."""
        drawn = self._drawn(variants, rails, task_id, repeat)
        packs = self._platformed(variants, rails, members, task_id, repeat, fits=fits)
        out: list[tuple[int, str | None]] = []
        for at in members:
            if (drawn is not None and at not in drawn) or (packs is not None and at not in packs):
                continue
            later = variants[at][3]
            envpack = self._envpack(later)
            if later is None or envpack is not None:
                out.append((at, envpack))
        return out

    def _platformed(
        self,
        variants: Sequence[Placed],
        rails: Sequence[OnRail] | None,
        members: Sequence[int],
        task_id: str,
        repeat: int,
        *,
        fits: Callable[[str | None], bool] | None = None,
    ) -> set[int] | None:
        """On a plan that draws its payment rails or platform packs (`draw_rails`, `draw_platforms`, D72): the
        variants of a task's row that keep their rail and pack, one of each per set of variants that differ in
        nothing else, the one this store's rows have used least so far (the rail first, then the pack; ties drawn by
        the plan's seed, the store, the task and the repeat), so every rail and pack is in use and the rows do not
        multiply. None keeps every variant."""
        axes = [
            axis for axis, on in ((RAIL_AXIS, self.plan.draw_rails), (PLATFORM_AXIS, self.plan.draw_platforms)) if on
        ]
        if not axes:
            return None

        def value(at: int, axis: str) -> str:
            if axis == RAIL_AXIS:
                return (rails[at][0] if rails is not None else None) or ""
            return variants[at][5] or ""

        groups: dict[str, list[int]] = {}
        for at in members:
            condition, market, _, _, surface, _, _ = variants[at]
            kind = surface.kind if surface is not None else None
            kept_axes = [value(at, axis) for axis in (RAIL_AXIS, PLATFORM_AXIS) if axis not in axes]
            groups.setdefault(dumps([condition.model_dump(mode="json"), market, kind, kept_axes]), []).append(at)
        kept: set[int] = set()
        for base, group in groups.items():
            chosen = group
            for axis in axes:
                uses = self.axis_uses.setdefault(axis, {})
                choices = list(dict.fromkeys(value(at, axis) for at in chosen))
                if axis == RAIL_AXIS and fits is not None:
                    # D80: only a rail the task's persona can pay on (or, on a no_instrument fault row, cannot)
                    choices = [c for c in choices if fits(c or None)] or choices
                least = min(uses.get(c, 0) for c in choices)
                tied = [c for c in choices if uses.get(c, 0) == least]
                rng = seeded_rng(axis, self.plan.seed, self.store.id, task_id, repeat, base)
                pick = tied[rng.randrange(len(tied))]
                uses[pick] = least + 1
                chosen = [at for at in chosen if value(at, axis) == pick]
            kept.update(chosen)
        return kept

    def _later(self, job: _MaskOnRail) -> str:
        """A placeholder for a world built only when a row is planned on it (D72), resolved by `_envpack`."""
        token = f"{LATER}{len(self.later)}"
        self.later[token] = job
        return token

    def _envpack(self, envpack: str | None) -> str | None:
        """The world a placed variant names: its envpack id, or the world a placeholder stands for, built once (the
        same store, mask, platform and profile is the same pack) and None when it does not build."""
        if envpack is None or not envpack.startswith(LATER):
            return envpack
        if envpack not in self.resolved:
            job = self.later[envpack]
            on_rail = self._world(
                job.worlds, job.market, job.key, job.kind, platform=job.platform, adopted=job.adopted, **job.where
            )
            if on_rail is None:
                self.resolved[envpack] = None
            else:
                pack = mask_onto(job.masked, job.base, on_rail.pack)
                job.worlds.save(pack)
                recipe = self.recipes[(job.market, job.key, job.platform)]
                self.draft.worlds[pack.id] = recipe.model_copy(update={"exposure": job.masked.mask})
                self.resolved[envpack] = pack.id
        return self.resolved[envpack]

    def _payer(self, persona: str | None) -> Any:
        """The persona a task's order rows pay as (its own, else the plan's, else the book's default); None without a
        persona book."""
        if self.book is None:
            return None
        name = persona or (self.buyer[0] if self.buyer is not None else None) or self.book.default
        try:
            return self.book.get(name)
        except NotFoundError:
            return None

    def _rail_fit(self, task_id: str, persona: str | None) -> Callable[[str | None], bool] | None:
        """D80: which rails a task's order rows may be paid over. A row is paired only with a rail its persona's
        wallet can pay on; a task of the fault share whose persona pays everywhere it can (a wallet some rail takes
        nothing from) is paired with a rail it cannot pay on instead, which makes it an explicit no_instrument fault
        row. None when the plan's rows do not order."""
        if self.plan.boundary != "order" and "checkout" not in self.plan.conditions:
            return None
        payer = self._payer(persona)
        if payer is None:
            return None
        rails: list[str | None] = list(self.plan.rails) or [None]
        deliberate = task_id in self.fault_tasks and not built_to_fail(payer, rails) and limited_wallet(payer, rails)
        shop_pay = self.shop_pay
        return lambda name: can_pay(payer, name, shop_pay=shop_pay) != deliberate

    def _note_unpaired(self) -> None:
        if self.unpaired:
            self.draft.note(
                f"{self.store.id}: {self.unpaired} rail placements left out (D80): the task's persona's wallet holds "
                "nothing the rail takes, and the task is not a no_instrument fault row"
            )
            self.unpaired = 0

    def _members(
        self,
        variants: Sequence[Placed],
        rails: Sequence[OnRail] | None,
        turn: int | None,
        fits: Callable[[str | None], bool] | None,
        task_id: str,
    ) -> list[int]:
        """The variants a task's rows are placed on: on an exposure design's `turn`, that turn's rail and surface;
        else every variant whose rail fits the task (D80). A paying task no rail fits gets no rows (its persona
        pays nowhere the store is served); a task of the fault share keeps them, each a planned fault row."""
        keep = task_id in self.fault_tasks
        if self.plan.draw_cells and rails is not None:
            return self._celled(variants, rails, fits, task_id, keep=keep)
        if turn is not None:
            chosen = _on_turn(variants, rails, turn, fits)
            if fits is not None and rails is not None and not keep:
                chosen = self._fitting(rails, chosen, fits, keep=False)
            return chosen
        return self._fitting(rails, range(len(variants)), fits, keep=keep)

    def _celled(
        self,
        variants: Sequence[Placed],
        rails: Sequence[OnRail],
        fits: Callable[[str | None], bool] | None,
        task_id: str,
        *,
        keep: bool,
    ) -> list[int]:
        """D89: the variants of a task's rows on the cells it draws: `cells_per_task` of the valid (surface, rail)
        cells the store's variants hold, among those whose rail the task's persona can pay on (D80; a no_instrument
        fault task's `fits` names the rails it cannot), each the cell this store's rows have used least so far, ties
        drawn by the plan's seed, the store and the task. A paying task no cell fits gets no rows; a fault task no
        cell fits keeps them all, as `_fitting` does. The draw is made once a task, so its reference rows run on the
        cells its sandbox rows did."""
        cells = [cell_of(_kind_of(v[4]), rails[at][0] or "") for at, v in enumerate(variants)]
        present = [c for c in dict.fromkeys(cells) if c in CELL_SET]
        draw = self._cell_draw()
        fresh = task_id not in draw.drawn
        count = self.plan.cells_per_task
        chosen = set(draw.draw(task_id, present, key=(self.store.id,), pays=fits, fault=keep, count=count))
        if fresh and not chosen:
            self.unpaired += len(variants)
        return [at for at, c in enumerate(cells) if c in chosen]

    def _cell_draw(self) -> CellDraw:
        if self.cells is None:
            self.cells = CellDraw(seed=self.plan.seed)
        return self.cells

    def _count_cell(self, row: ManifestRow, *, fresh: bool) -> None:
        """D89: a row of the mode a task drew its cells in (`fresh`) counted on its cell, for the next task's draw."""
        if fresh and self.plan.draw_cells and row.rail is not None:
            self._cell_draw().count(cell_of(_kind_of(row.surface), row.rail))

    def _surface_adopted(self, surface: RowSurface | None, rail_name: str | None) -> bool | None:
        """D89: on a row of a cells plan, whether the store does not serve the row's surface itself; None elsewhere."""
        if not self.plan.draw_cells or rail_name is None or self.served is None:
            return None
        return not self.served.serves(_kind_of(surface))

    def _fitting(
        self,
        rails: Sequence[OnRail] | None,
        members: Iterable[int],
        fits: Callable[[str | None], bool] | None,
        *,
        keep: bool,
    ) -> list[int]:
        """The members whose rail fits the task (D80), the rest counted for the store's note; when none fits, all of
        them for a fault task (`keep`), else none."""
        members = list(members)
        if rails is None or fits is None:
            return members
        kept = [at for at in members if fits(rails[at][0])]
        if not kept and keep:
            return members
        self.unpaired += len(members) - len(kept)
        return kept

    def _planned_fault(self, condition: Condition, rail_name: str | None) -> PaymentFaultKind | None:
        """D80: how an order row's payment is designed to fail, from its persona's wallet and its rail; None when it
        is built to pay or the row does not order."""
        if not authorises_checkout(condition):
            return None
        persona = condition.params.get("persona")
        payer = self._payer(persona if isinstance(persona, str) else None)
        return planned_payment(payer, rail_name, shop_pay=self.shop_pay).fault if payer is not None else None

    def _ordered(self, condition: Condition, persona: str | None, market: str) -> Condition:
        """A world row's condition on a plan whose boundary is order (D66): it authorises an order, paid as the task's
        persona (else the plan's), the judge's convention. A persona with no address in the row's market cannot be
        delivered to, so its row keeps the cart boundary, and the manifest says so."""
        if authorises_checkout(condition):
            return condition
        name = persona or (self.buyer[0] if self.buyer is not None else None)
        markets = self.persona_markets.get(name or "", frozenset())
        code = MARKET_ALIASES.get(market.upper(), market.upper())
        if name is None or code not in markets:
            self.draft.note(
                f"{self.store.id}: some rows keep the cart boundary: persona {name} has no address in {market}"
            )
            return condition
        return condition.model_copy(update={"params": {**condition.params, "checkout": True, "persona": name}})


def make_row(
    *,
    store_id: str,
    task_id: str,
    condition: Condition,
    profile: str,
    mode: RunMode,
    repeat: int,
    seed: int,
    market: str,
    envpack: str | None,
    handoff: bool,
    surface: RowSurface | None = None,
    replica_of: str | None = None,
    platform_pack: str | None = None,
    market_context: MarketContext | None = None,
    rail: str | None = None,
    adopted: bool | None = None,
    rail_version: str | None = None,
    psp: str | None = None,
    payment_fault: PaymentFaultKind | None = None,
    surface_adopted: bool | None = None,
) -> ManifestRow:
    """A row, with its episode id derived from everything that makes it this episode. The surface enters the id only
    when it is not UCP over MCP, the surface every row ran on before rows named one, so those rows keep their ids;
    the live row a replica row pairs with does not enter it, since pairing does not change the episode. A market
    context enters it only when it is `fill`, which is not the live default."""
    parts: list[Any] = [
        store_id,
        task_id,
        condition.model_dump(mode="json"),
        profile,
        mode,
        repeat,
        seed,
        market,
        envpack,
        handoff,
    ]
    if surface is not None and surface.kind != DEFAULT_SURFACE:
        parts.append({"surface": surface.kind})
    if market_context == "fill":
        parts.append({"market_context": market_context})
    if rail is not None:
        parts.append({"rail": rail})
    if rail is not None and rail_version is not None:
        parts.append({"rail_version": rail_version})
    if rail is not None and psp is not None:
        parts.append({"psp": psp})
    return ManifestRow(
        episode_id=stable_id("ep", *parts),
        store_id=store_id,
        task_id=task_id,
        condition=condition,
        profile=profile,
        mode=mode,
        repeat=repeat,
        seed=seed,
        market=market,
        envpack=envpack,
        handoff=handoff,
        surface=surface,
        replica_of=replica_of,
        platform_pack=platform_pack,
        market_context=market_context if mode == "live" else None,
        rail=rail,
        adopted=adopted if rail is not None else None,
        rail_version=rail_version if rail is not None else None,
        psp=psp if rail is not None else None,
        payment_fault=payment_fault if authorises_checkout(condition) else None,
        surface_adopted=surface_adopted if rail is not None else None,
    )


def hides_options(platforms: Sequence[str]) -> bool:
    """Whether every pack of `platforms` shows no option values (its spec's `documents.serves_options`, D121); False
    for no pack, or a pack with no spec file."""
    from carttrace.sandbox.platforms.spec import PACKS_DIR, pack_spec

    names = [name for name in platforms if (PACKS_DIR / f"{name}.yaml").is_file()]
    return (
        bool(names)
        and len(names) == len(platforms)
        and not any(pack_spec(name).documents.serves_options for name in names)
    )


def needs_option(task: TaskRef) -> bool:
    """Whether a task's spec turns on a variant's option value (a predicate naming an option: a colour, a size)."""
    predicates = task.hidden_spec.get("predicates") or []
    return any(isinstance(p, dict) and p.get("option") for p in predicates)


def _unique(tasks: Sequence[TaskRef]) -> list[TaskRef]:
    seen: set[str] = set()
    out: list[TaskRef] = []
    for task in tasks:
        if task.task_id not in seen:
            seen.add(task.task_id)
            out.append(task)
    return out


def fixture_currency(fixtures: Sequence[Mapping[str, Any]]) -> str | None:
    """The currency most of these products' variants are priced in (ties alphabetically), or None for none."""
    counts: dict[str, int] = {}
    for product in fixtures:
        for variant in product.get("variants") or ():
            price = variant.get("price") if isinstance(variant, Mapping) else None
            code = price.get("currency") if isinstance(price, Mapping) else None
            if isinstance(code, str) and code:
                counts[code.upper()] = counts.get(code.upper(), 0) + 1
    return max(sorted(counts), key=lambda c: counts[c]) if counts else None


def _task_currency(task: TaskRef) -> str | None:
    code = task.hidden_spec.get("currency")
    return code.upper() if isinstance(code, str) and code else None


def _replica_only(task: TaskRef) -> bool:
    """Whether only a sandbox or replica row can run the task: its spec carries faults (a substitute task's sealed
    stockout) or it is a custom intent marked as needing the replica."""
    return task.needs_replica or bool(task.hidden_spec.get("faults"))


def _held_by(task: TaskRef, held: Collection[str]) -> bool:
    """Whether a world holding these product and variant ids holds everything the task names."""
    witnesses = task.hidden_spec.get("witnesses")
    named = (
        [*task.fixture, *(w for w in witnesses if isinstance(w, str))] if isinstance(witnesses, list) else task.fixture
    )
    return bool(named) and all(entry in held for entry in named)


def _kind_of(surface: RowSurface | None) -> str:
    return DEFAULT_SURFACE if surface is None else surface.kind


def _on_turn(
    variants: Sequence[Placed],
    rails: Sequence[OnRail] | None,
    turn: int,
    fits: Callable[[str | None], bool] | None = None,
) -> list[int]:
    """The places of the variants whose turn it is for the `turn`th task: the payment rails taken in turn (D66, so a
    plan's rails are covered evenly without crossing every row with each), among those the task's persona can pay on
    when `fits` says (D80), then that rail's surfaces in turn."""
    if not variants:
        return []
    named = [rails[at][0] if rails is not None else None for at in range(len(variants))]
    order = list(dict.fromkeys(named))
    if fits is not None:
        order = [name for name in order if fits(name)] or order
    chosen = order[turn % len(order)]
    kinds = list(dict.fromkeys(_kind_of(v[4]) for at, v in enumerate(variants) if named[at] == chosen))
    kind = kinds[(turn // len(order)) % len(kinds)]
    return [at for at, v in enumerate(variants) if named[at] == chosen and _kind_of(v[4]) == kind]


def _on_live_surface(surface: RowSurface | None) -> bool:
    return surface is None or surface.kind in LIVE_SURFACES


def _on_website(surface: RowSurface | None) -> bool:
    return surface is not None and surface.kind == WEBSITE


def website_ineligible(result: StoreResult | None, *, require_preflight: bool) -> str | None:
    """Why a store without UCP cannot be shopped through its website live, or None when it can: a preflight that met
    a wall, robots.txt or 429s keeps it out (those are recorded as its outcome already); no preflight at all does
    only when the plan requires one."""
    if result is None or result.preflight is None:
        return "no preflight result" if require_preflight else None
    if result.preflight not in ("ok", "no_protocol"):
        return f"preflight {result.preflight}"
    return None


def live_ineligible(store: Store, result: StoreResult | None, *, require_preflight: bool) -> str | None:
    """Why a store cannot have live rows, or None when it can."""
    if result is None or result.preflight is None:
        if require_preflight:
            return "no preflight result"
        return None if store.platform_key == "shopify" else "no preflight result, and not a Shopify store"
    if result.preflight != "ok":
        return f"preflight {result.preflight}"
    if LIVE_SURFACE not in result.protocols:
        return "the preflight found no UCP over MCP"
    return None


def derived_seed(namespace: str, *parts: object) -> int:
    """A seed drawn from a namespace and parts, the same on every machine and run."""
    return seeded_rng(namespace, *parts).randrange(SEED_SPACE)


def _selected(stores: Sequence[Store], plan: ManifestPlan, categories: CategoryCatalog) -> list[Store]:
    """The plan's stores in queue order. An unknown store or category fails the build instead of shrinking it."""
    known = {c.slug for c in categories}
    unknown = [slug for slug in plan.categories if slug not in known]
    if unknown:
        raise ValidationError(f"unknown categories {unknown}", hint="see `carttrace stores categories`")
    wanted = set(plan.stores)
    found: set[str] = set()
    picked: list[Store] = []
    # D119: a later open stand-in of a queue store (store-388~2) is planned as a store of its own, after its queue store
    for store in with_stand_ins(stores, plan.stores) if plan.catalog != REPLICA else stores:
        if plan.categories and store.category_slug not in plan.categories:
            continue
        if wanted:
            hits = {store.id, store.domain} & wanted
            if not hits:
                continue
            found |= hits
        if store.category_slug not in known:
            raise ValidationError(
                f"store {store.id} is in category {store.category_slug!r}, which has no category file",
                hint="add data/categories/<slug>.yaml, or leave the store out",
            )
        picked.append(store)
    missing = sorted(wanted - found)
    if missing:
        raise ValidationError(
            f"no store {missing} in the chosen categories of the queue",
            hint="pass queue ids (store-004) or domains, as `carttrace stores show` lists them",
        )
    return picked


def _paired(rows: Sequence[ManifestRow], draft: _Draft) -> list[ManifestRow]:
    """The rows with a pair id on each live row and the replica row that names it, both still in the manifest (a
    limit may have cut the replica half), and a note per store counting its pairs. Pair ids never enter an episode
    id, so adding them changes no episode."""
    live = {row.episode_id for row in rows if row.mode == "live"}
    pairs = {row.replica_of: pair_id_of(row) for row in rows if row.replica_of is not None and row.replica_of in live}
    out: list[ManifestRow] = []
    counts: dict[str, int] = {}
    for row in rows:
        pair_id = pairs.get(row.episode_id) if row.mode == "live" else pairs.get(row.replica_of or "")
        if pair_id is None:
            out.append(row)
            continue
        out.append(row.model_copy(update={"pair_id": pair_id}))
        if row.mode == "live":
            counts[row.store_id] = counts.get(row.store_id, 0) + 1
    for store_id, count in sorted(counts.items()):
        draft.note(
            f"{store_id}: {count} replica pairs: each live row and its replica row share a task, persona, seed and "
            "condition under one pair_id"
        )
    return out


def _limited(groups: Sequence[list[ManifestRow]], limit: int | None) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for group in groups:
        if limit is not None and len(rows) + len(group) > limit:
            break
        rows.extend(group)
    return rows


def _with_persona(condition: Condition, persona: str | None) -> Condition:
    """The condition with the task's persona in its params, so a reader can pair episodes by shopper and a checkout
    row pays as the persona its task was written for."""
    if persona is None:
        return condition
    return condition.model_copy(update={"params": {**condition.params, "persona": persona}})


def _roster(
    plan: ManifestPlan, personas: PersonaBook | None, store: Store, draft: _Draft
) -> tuple[list[str] | None, list[str]]:
    """The personas a store's tasks are crossed with, in the order slots take them: those the plan names (all of
    them, a seeded count, or the named ones) that have an address in the store's market, shuffled by seed per store;
    None when the plan crosses nothing. With a payment fault share (D80), the second list holds the personas the
    share's tasks are written for (built to fail, or holding nothing one of the plan's rails takes) and the first only
    personas built to pay; without one, the second is empty and the first is drawn as before D80."""
    if plan.personas is None or personas is None:
        return None, []
    home = store.market_code or DEFAULT_HOME_MARKET
    code = MARKET_ALIASES.get(home.upper(), home.upper())
    markets = personas.markets()
    able = sorted(n for n, ms in markets.items() if code in {MARKET_ALIASES.get(m.upper(), m.upper()) for m in ms})
    spec = plan.personas
    wanted: list[str] | None
    if spec == "all":
        wanted = able
    elif spec.isdigit():
        wanted = able if int(spec) >= len(able) else None
    else:
        names = spec.split(",")
        unknown = sorted(set(names) - set(markets))
        if unknown:
            raise ValidationError(f"no personas {unknown}", hint=f"choose from {', '.join(sorted(markets))}")
        wanted = [n for n in names if n in able]
        left_out = sorted(set(names) - set(wanted))
        if left_out:
            draft.note(f"{store.id}: personas {', '.join(left_out)} have no address in {home}, so they are left out")
    if plan.payment_fault_bp is not None:
        return _fault_roster(plan, personas, store, draft, able if wanted is None else wanted)
    rng = seeded_rng("personas", plan.seed, store.id)
    chosen = rng.sample(able, int(spec)) if wanted is None else rng.sample(wanted, len(wanted))
    if not chosen:
        draft.note(f"{store.id}: no persona has an address in {home}; its tasks are written for the plan's persona")
        return None, []
    return chosen, []


def _fault_roster(
    plan: ManifestPlan, personas: PersonaBook, store: Store, draft: _Draft, among: Sequence[str]
) -> tuple[list[str] | None, list[str]]:
    """D80: the paying personas a store's tasks take (the plan's count of them, or all), and those the fault share's
    tasks take (as many, at most), each shuffled by seed per store. A persona built to fail on any rail is never a
    payer; one whose wallet some rail cannot take pays on the other rails and is a `no_instrument` fault on that one."""
    rails: list[str | None] = list(plan.rails) or [None]
    faulty = [n for n in among if built_to_fail(personas.get(n), rails)]
    limited = [n for n in among if n not in faulty and limited_wallet(personas.get(n), rails)]
    payers = [n for n in among if n not in faulty and pays_somewhere(personas.get(n), rails)]
    spec = plan.personas or "all"
    count = int(spec) if spec.isdigit() else len(payers)
    rng = seeded_rng("personas", plan.seed, store.id)
    chosen = rng.sample(payers, min(count, len(payers)))
    pool = faulty + limited
    picked = rng.sample(pool, min(max(1, (count + 1) // 2), len(pool))) if plan.payment_fault_bp else []
    if not chosen:
        draft.note(f"{store.id}: no persona built to pay has an address in its market; its tasks take the plan's")
        return None, []
    if plan.payment_fault_bp and not picked:
        draft.note(f"{store.id}: no persona built to fail has an address in its market; every task pays (D80)")
    return chosen, picked


def _markets(personas: PersonaBook | None) -> dict[str, frozenset[str]]:
    """Persona to the delivery markets it has an address in, under one spelling of each market (GB, not UK)."""
    if personas is None:
        return {}
    return {
        name: frozenset(MARKET_ALIASES.get(m.upper(), m.upper()) for m in markets)
        for name, markets in personas.markets().items()
    }


def _persona(plan: ManifestPlan, personas: PersonaBook | None) -> tuple[str, frozenset[str]] | None:
    """The persona checkout rows pay as, with its markets; None when the plan has no checkout condition and its rows
    end at a cart."""
    if "checkout" not in plan.conditions and plan.boundary != "order":
        return None
    if personas is None:
        raise ValidationError(
            "checkout rows need the personas they pay as", hint="pass personas=carttrace.agents.load_personas()"
        )
    name = plan.persona or personas.default
    markets = personas.markets()
    if name not in markets:
        raise ValidationError(f"no persona {name!r}", hint=f"choose one of {', '.join(sorted(markets))}")
    return name, frozenset(MARKET_ALIASES.get(m.upper(), m.upper()) for m in markets[name])


def _preset(presets: Mapping[str, ProtocolProfile], name: str) -> ProtocolProfile:
    if name not in presets:
        raise ValidationError(f"no protocol preset {name!r}", hint=f"choose one of {sorted(presets)}")
    return presets[name]


def manifest_path(var_dir: Path, manifest: Manifest) -> Path:
    """Where a manifest is kept by default: <paths.var>/manifests/<id>.json."""
    return var_dir / MANIFESTS_DIR / f"{manifest.id}.json"


def save_manifest(manifest: Manifest, path: Path) -> Path:
    """Write the manifest's canonical text. Never overwrites a different manifest."""
    text = manifest.to_text()
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValidationError(
            f"{path} exists and holds another manifest", hint="manifests are named by their id; choose another path"
        )
    atomic_write_text(path, text)
    return path


def load_manifest(path: Path) -> Manifest:
    """A manifest from its file. Refuses one that does not validate, or whose id is not its content."""
    try:
        data = loads(path.read_bytes())
    except FileNotFoundError:
        raise NotFoundError(f"no manifest at {path}", hint="build one with `carttrace collect manifest`") from None
    except ValueError as exc:
        raise ValidationError(
            f"{path} is not JSON: {exc}", hint="manifests are written by `carttrace collect manifest`"
        ) from exc
    if not isinstance(data, Mapping) or data.get("schema") != MANIFEST_SCHEMA:
        found = data.get("schema") if isinstance(data, Mapping) else type(data).__name__
        raise ValidationError(
            f"{path} declares schema {found!r}; this code reads {MANIFEST_SCHEMA}",
            hint="build the manifest again with this version of mc",
        )
    try:
        return Manifest.model_validate(data)
    except PydanticValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        where = ".".join(str(part) for part in first["loc"]) or "manifest"
        raise ValidationError(
            f"{path} is not a valid manifest: {where}: {first['msg']}",
            hint="never edit a manifest by hand; build a new one with `carttrace collect manifest`",
        ) from None

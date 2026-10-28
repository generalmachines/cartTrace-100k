"""One episode, from the first request to the closed bundle.

The steps, in order:
1. Live rows check the gate (`LiveGate.check`): a closed gate or a paused store means the episode does not start and
   goes back to the queue. Nothing has been sent, so nothing is recorded.
2. Live rows run discovery with its exchanges buffered, so the header can name the profile the store declares.
3. The bundle opens, with the rights decided for the episode (`carttrace.collect.rights`).
4. A store that disallows us (robots, 401 or 403), throttles us out, or has no UCP over MCP ends the episode here, as
   `blocked`, `throttled` or `no_protocol`, and the store is skipped for the rest of the run.
5. The surface opens: carttrace.live's UCP client for a live row, the sandbox's in-process surface over the row's world for
   the others. A detail_hidden row reads it through the detail filter; every row reads it through the observer that
   triages failed calls. When the filter removes catalog details from a reply, a system note right after that call
   says how many statements of each it removed, and the outcome's counts carry the episode's totals
   (`hidden.<detail>`), so a judge can tell a store that never stated a detail from one whose detail we hid, even
   after retention has purged the turns. A WebMCP row is not wrapped in the filter: the storefront is opened with the
   same cut as its own `detail_filter` and reports to the same hook, so no copy of a page shows the detail and
   nothing is counted twice. An ACP row, whose protocol has no catalog, reads the store's catalog beside its checkout
   through a second surface on the same served store (the replica's UCP catalog tools, nothing else of UCP), and the
   filter wraps that one. A sandbox row whose task's spec carries sealed faults (a substitute task's stockout) runs
   on its world with those faults joined to the schedule, true from the episode's first read
   (`carttrace.collect.worlds.sealed_world`); a system note names the derived pack and the faults, and the sandbox's event
   log has each as a sealed `fault.fired` at the opening. A live row whose task carries faults never runs. A replica
   row's world names the rate its store answered "sold out" at live (D52): the episode draws, by its seed, which
   variants its cart writes refuse as sold out while the catalog lists them available, a system note names them and
   any witness spared so the task can still be bought, and the header names the rate (`sold_out`). Its world also
   names its store's replies record (D70): the seed draws which calls answer with the store's own error, a system
   note names them, and the header names the handlers served and the rates (`store_replies`).
   As soon as a sandbox row's surface opens, the episode's writer is attached to the sandbox's own event log
   (`Bindings.watch_sandbox`, D33): every record so far (the opening, its at_start faults) and each one after goes to
   the bundle's events.jsonl as the sandbox chains it, redacted. A live row has no sandbox and no events file.
6. The agent runs with the deterministic shopper, the decision supervisor and the run's limits. The runner holds a
   hard backstop of three time caps (carttrace.config.schema.runner_backstop_s), in case an agent does not stop itself.
7. The cart is read back through the surface (never through the filter) and snapshotted with the wire record it came
   from: that is the verified cart.
8. A live row that asks for it hands a cart_ready cart to a sandbox checkout.
9. The outcome is written and the bundle closed: uploaded, bundle.json last. On a sandbox row the sink is detached as
   the outcome is recorded, and the outcome's `state_digest` names the sandbox's final state hash and the head of the
   events chain, so a replay of the same row can be shown to reach the same place.

Checkout rows (D13, sandbox and reference only). The condition authorises an order (`params["checkout"]`, the judge's
convention). Before the agent starts, the row's sandbox session gets the buyer token with the purchase permission and
the agent gets the persona's wallet (`Bindings.checkout`); the step cap grows by `CHECKOUT_STEPS`, and the supervisor
does not ask whether the cart boundary is reached. After the cart, the checkout and the order are read back
(get_checkout, get_order) and snapshotted with their wire records: the order's lines are the final cart, and the
checkout's status and payment instrument and the order's total are the payment state the judge reads. A live row
whose condition authorises an order is never run: the manifest refuses one, and so does the episode.

Orders and payment rails (D66). A sandbox row of a plan whose boundary is order is a checkout row, as above. On a plan
that crosses payment rails the row names its rail, and the header records it with its protocol, its version and
whether the store itself speaks it (`rail`, `rail_protocol`, `rail_version`, `adopted`). The rail decides what opens:
ucp_psp and acp_delegated pay over the row's own surface; ap2_mandates opens the AP2 merchant with the shopper's wallet
as its signer (`WalletSigner`, which signs a payment mandate only for a credential the wallet issued against the
shopper's approval); mpp opens the store's MPP endpoint beside the row's UCP surface, on the same served store, and the
agent is not offered UCP's own complete_checkout. A row whose manifest names checkout fault rates draws at most one
fault by its episode id (carttrace.collect.checkout_faults): the header names the rates, the draw and the fault, the world
carries its knobs from the first read, a system note names the derived pack, and the outcome says whether it fired
(`checkout_faults`, from the store's own `fault.fired` records). Before the outcome a checkout row writes the PSP's own
record of every payment it decided, as a harness wire record (surface `psp`): the opaque token, the amount in minor
units, the currency, the outcome, the checkout, the order, and where in events.jsonl it was decided. A ucp_psp row
that names a payment broker (D71) runs on its world with the broker set in its policy (a system note names the derived
pack); its header names `psp`, and the PSP's record adds `psp`, how 3-D Secure ended (`authentication`) and the
broker's own answer for each payment (`processor`).

Surfaces (D15). A sandbox row opens the surface it names (`row.surface`, the sandbox.surface kind; UCP over MCP when it
names none), and the header's `surfaces` records that kind at its version first, which is what the bench pairs and
splits rows by, then the catalog companion when the row has one. A live row runs on carttrace.live's UCP over MCP only: one
that names any other surface is refused before anything is sent, as one that authorises an order is. The read-backs
go through the row's own surface on every kind (`carttrace.collect.surfaces.ReadBack`: UCP's get_cart, get_checkout and
get_order; ACP's checkout session, whose lines are the cart and whose `order` is the order; a storefront's get_cart and
get_checkout, which shows the order page once the order is placed) into the same snapshot shapes, so the judge reads
every kind the same way. A WebMCP row's storefront is opened with the persona's 3-D Secure answer (the page answers
the bank itself), the browser limits from config and the episode's seed.

Failures. Anything raised after the bundle opened is triaged and recorded: the episode ends `harness_error`, except a
401, 403 or 429 from a live store outside the agent, which is the store's (`blocked`, `throttled`). A cancelled episode
(the runner draining past its deadline) is recorded as `harness_error` with the label `drained` before the
cancellation goes on: its surfaces are closed first, for up to CANCEL_CLOSE_S, while the writer still takes their
closing exchanges, then the bundle is closed (D122: closed the other way round, every drained episode reported
"closing a surface failed: the writer ... is closed", 2026-10-17). An upload already under way when the cancellation
comes is left to finish.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast, get_args

from pydantic import ValidationError as PydanticValidationError

from carttrace.agents.products import fixtures_from_snapshot
from carttrace.agents.profile import BuyerProfile, buyer_profile
from carttrace.collect.bindings import Bindings
from carttrace.collect.checkout_faults import PSP_SURFACE, fault_world, firings, psp_authorisations, three_ds_decision
from carttrace.collect.checkout_faults import draw as draw_faults
from carttrace.collect.conditions import (
    CHECKOUT_STEPS,
    added_of,
    added_world,
    authorises_checkout,
    details_of,
    hidden_of,
    persona_of,
)
from carttrace.collect.decisions import EpisodeDecisions
from carttrace.collect.exposure_sheet import SheetInputs, build_sheet
from carttrace.collect.handoff import hand_off, payload_of
from carttrace.collect.manifest import Manifest, ManifestRow, derived_seed
from carttrace.collect.payment import PERSONA_FAULTS, PlannedPayment, planned_payment
from carttrace.collect.ports import AgentResult, Discovery, LiveGate, RunMode, SandboxWatch
from carttrace.collect.profiles import (
    LIVE_SURFACES,
    LIVE_WEBMCP,
    MODEL_UNAVAILABLE,
    NO_AGENT_DIALECT,
    STOREFRONT,
    WEBSITE,
    spoken_surfaces,
    surface_version,
)
from carttrace.collect.quarantine import DECIDER_UNAVAILABLE, decider_down
from carttrace.collect.rails import RAILS, rail_version
from carttrace.collect.replica_site import site_jsonld_types
from carttrace.collect.rights import live_rights
from carttrace.collect.selection import SOURCE as SELECTION_SOURCE
from carttrace.collect.selection import WireSeen, selection_record
from carttrace.collect.shopper_maps import mapped
from carttrace.collect.supervisor import DecisionSupervisor
from carttrace.collect.surfaces import (
    CatalogOnly,
    DetailFilter,
    ObservedSurface,
    ReadBack,
    WireBuffer,
    WireTap,
    WithoutTools,
    challenge_policy,
    companion_kind,
    webmcp_options,
)
from carttrace.collect.triage import Failure, Triage, failure_of, failure_of_result, store_outcome, triage
from carttrace.collect.v1_cut import forced_faults, forced_kind
from carttrace.collect.v1_stamp import designed_recovery, experiment_config, order_stages, persona_fired
from carttrace.collect.worlds import (
    SandboxWorlds,
    World,
    presents,
    rights_key,
    sealed_world,
    sold_out_world,
    store_errors_world,
    task_faults,
)
from carttrace.config.schema import BACKSTOP_MIN_S, Settings, runner_backstop_s
from carttrace.core.clock import Clock, iso
from carttrace.core.errors import MCError, SafetyError, ValidationError
from carttrace.core.files import read_jsonl
from carttrace.decide.service import DecisionEngine
from carttrace.protocols.tools import ToolResult, ToolSurface
from carttrace.protocols.ucp_meta import PLACEHOLDER_AGENT_PROFILE
from carttrace.sandbox.browser import BrowserFailure, BrowserLimits, NoBrowserSlot
from carttrace.sandbox.envpack import EnvPack
from carttrace.sandbox.presented import Guard, VisibilityLeak, forbidden_tokens
from carttrace.trace.experiment import ThreeDs
from carttrace.trace.inventory import INVENTORY_DIR
from carttrace.trace.layout import EXPOSURE_FILE, TURNS_FILE, TraceStore
from carttrace.trace.provenance import Provenance, agent_of, provenance_of, store_source_of, surface_of
from carttrace.trace.schema import (
    SCHEMA_MINOR,
    Cart,
    CheckoutFaultDraw,
    EndReason,
    EpisodeHeader,
    ModelRef,
    Outcome,
    PaymentFault,
    PaymentPlan,
    Rights,
    SurfaceRef,
    WalletInstrument,
)
from carttrace.trace.writer import TraceWriter


class EpisodeIdle(MCError):
    """The episode's bundle got no record for collect.idle_timeout_s: something hung (D122)."""


END_REASONS: frozenset[str] = frozenset(get_args(EndReason))
STORE_REFUSALS: frozenset[str] = frozenset({"robots_disallowed"})
"""Refusals before any request that are still the store's outcome: robots.txt disallows us, so the episode ends
blocked and is recorded, and the store is skipped for the rest of the run."""
DAILY_CAP = "daily_cap"
"""carttrace.live's code for a store that has had today's live sessions: its rows wait for another run."""
LABEL_CHARS = 120
ERROR_CHARS = 300
CANCEL_CLOSE_S = 10.0
"""How long a cancelled episode spends closing its surfaces (an MCP session's DELETE, a browser) before its bundle is
closed without them."""
IDLE_TIMEOUT = "idle_timeout"
"""The label of an episode cut because its bundle got no record for collect.idle_timeout_s (D122)."""
IDLE_CHECK_S = 15.0
"""How often the idle deadline looks at the bundle."""
BROWSER_RETRY_S = (5.0, 15.0)
"""The waits before the second and later tries to open a WebMCP storefront on a fresh browser (D122)."""
SURFACE_CLOSE_UNRECORDED = "surface_close_unrecorded"
"""A surface closed after the bundle: its closing exchange could not be recorded, which is expected then, not an
error."""


@dataclass(frozen=True, slots=True)
class RunContext:
    """What every episode of a run shares."""

    settings: Settings
    bindings: Bindings
    store: TraceStore
    engine: DecisionEngine
    clock: Clock
    manifest: Manifest
    run_id: str
    worlds: SandboxWorlds
    staging_dir: Path
    code_version: str
    sandbox_rights: Mapping[str, Rights]
    """Rights by catalog name, decided once when the run starts."""
    models: Mapping[RunMode, tuple[ModelRef, ...]]
    model_names: frozenset[str]
    """Endpoint and role names of the model servers, so a failure naming one is attributed to models."""
    gate: LiveGate | None = None
    salt: str = ""
    live_agent: str = "qwen"
    """Which agent shops live rows: the Qwen agent, or the reference agent for a read-only smoke (`--live-agent`)."""
    playgrounds: Mapping[str, Any] = field(default_factory=dict)
    """The playground stores a run's playground rows shop (carttrace.collect.playgrounds, D44), by store id."""


@dataclass(slots=True)
class EpisodeReport:
    """How one episode went, for the runner's counters."""

    episode_id: str
    store_id: str
    category: str
    mode: RunMode
    prefix: str | None = None
    """The bundle's storage prefix, once it is uploaded."""
    end_reason: str | None = None
    ended_at: str | None = None
    failed: bool = False
    """Our failure: the episode ended harness_error, or its bundle could not be opened or uploaded."""
    paused: bool = False
    """Not started: the live gate is closed or the store is paused; or, set by the scheduler, a sandbox episode the
    agent's model server failed, run again in place. The row goes back to the queue."""
    skip_store: str | None = None
    """Why the store's other rows should not run in this run, if they should not."""
    error: str | None = None
    labels: list[str] = field(default_factory=list)
    provenance: Provenance = "unknown"
    """Where the store came from (carttrace.trace.provenance), read from the header once the bundle opens."""
    store_source: str = ""
    surface: str = ""
    agent: str = ""


async def run_episode(row: ManifestRow, ctx: RunContext) -> EpisodeReport:
    return await Episode(row, ctx).run()


class Episode:
    """One episode. `report` is filled as it goes, so a runner that cancels `run` can still read how it ended."""

    served_pack: Any = None
    guard: Guard | None = None
    """The visibility guard the row's served store holds (D106), once its surfaces open."""
    """The envpack a sandbox row's store serves, faults and all: the shopper reads its catalog to judge an order."""

    authorisations: Sequence[Mapping[str, Any]] = ()
    """The PSP's record of every payment it decided, once `_psp_record` read it as the episode ends."""

    def __init__(self, row: ManifestRow, ctx: RunContext) -> None:
        self.row = row
        self.ctx = ctx
        self.manifest = ctx.manifest
        self.store_ref = ctx.manifest.store(row)
        self.task = ctx.manifest.task(row)
        self.profile = ctx.manifest.profile(row)
        self.report = EpisodeReport(row.episode_id, row.store_id, self.store_ref.category, row.mode)
        self.phase = "start"
        self.writer: TraceWriter | None = None
        self.decisions: EpisodeDecisions | None = None
        self.discovery: Discovery | None = None
        self.buffer = WireBuffer()
        """A live store's discovery exchanges, held until the bundle opens."""
        self.surfaces: list[ToolSurface] = []
        self.labels: list[str] = []
        self.hidden: Counter[str] = Counter()
        """Statements of each catalog detail the detail filter kept from the agent, over the episode."""
        self.checkout = authorises_checkout(row.condition)
        """The row's condition authorises an order (D13): the journey goes on from the cart, in the sandbox only."""
        self.surface = ctx.manifest.surface(row)
        """The surface the row runs on: the one it names, else UCP over MCP at its profile's release."""
        self.companion = (
            companion_kind(self.surface.kind, self.profile) if row.mode not in ("live", "playground") else None
        )
        """The UCP binding the store's catalog is read through beside a checkout protocol that has none (ACP)."""
        self._readback: ReadBack | None = None
        self.ended = False
        """The outcome is written."""
        self.closing = False
        """The upload has started; it finishes on its own thread even if this episode is cancelled."""
        self.started = ctx.clock.monotonic()
        self.watch: SandboxWatch | None = None
        """The sandbox behind a sandbox row's surface, once it opened and its event log feeds events.jsonl."""
        self.events_failed = False
        """An event could not be written: events.jsonl stops there and the outcome names no state digest."""
        self.order_ref: str | None = None
        """The order a checkout row placed, as the store named it (a playground row reads it back from wc/v3)."""
        self.order_unverified = False
        self.jsonld_types: list[str] | None = None
        """A sandbox website row's JSON-LD types, read from its world before the bundle opens (D67)."""
        self._site_pack: Any = None
        """The order was taken from the agent's completion, not the checkout read; only the admin API confirms it."""
        self.rail = RAILS.get(row.rail) if row.rail is not None else None
        """The payment rail the row's order is paid over (D66), or None."""
        self.fault_draw: CheckoutFaultDraw | None = None
        """The checkout fault the row drew (D66), decided when its header is written."""
        self.fault_events: list[dict[str, Any]] = []
        """The store's `fault.fired` records, as they came, for the outcome's checkout_faults."""
        self.payment_events: list[dict[str, Any]] = []
        """The store's payment records, as they came, for the PSP's record."""
        self.signer: Any = None
        """The shopper's wallet signing AP2 mandates on an ap2_mandates row, once it has a wallet."""

    @property
    def readback(self) -> ReadBack:
        """How the row's own surface is read back, looked up when first needed, so a kind no agent dialect speaks
        (MPP, AP2) is refused inside the episode (`no_agent_dialect`), recorded in its bundle, never before it."""
        if self._readback is None:
            self._readback = ReadBack.for_kind(self.surface.kind, ucp=self.profile.ucp, ap2=self.profile.ap2)
        return self._readback

    async def run(self) -> EpisodeReport:
        if self.row.mode == "live" and self.checkout:
            return self._refused_live_checkout()
        if self.row.mode == "live" and self.surface.kind not in LIVE_SURFACES:
            return self._refused_live_surface()
        if self.row.mode == "live" and added_of(self.row.condition):
            return self._refused_live_added()
        if self.row.mode == "live" and self.task.hidden_spec.get("faults"):
            return self._refused_live_faults()
        try:
            if self.row.mode == "live":
                await self._live()
            elif self.row.mode == "playground":
                await self._playground()
            else:
                await self._sandbox()
        except asyncio.CancelledError:
            await self._close_surfaces_first()
            self._end_now("drained")
            raise
        finally:
            await self._close_surfaces()
            self.report.labels = sorted(set(self.labels))
        return self.report

    async def _close_surfaces_first(self) -> None:
        """On a cancellation: close the surfaces while the writer still records their closing exchanges, for up to
        CANCEL_CLOSE_S; a second cancellation or the time limit leaves the rest to be closed after the bundle."""
        if self.writer is None or self.closing or self.ended:
            return
        try:
            async with asyncio.timeout(CANCEL_CLOSE_S):
                await self._close_surfaces()
        except (TimeoutError, asyncio.CancelledError):
            self.labels.append("surface_close_cut")

    def _refused_live_checkout(self) -> EpisodeReport:
        """Nothing is bought on a live store (D3): a live row that authorises an order never runs, and sends nothing."""
        self.report.failed = True
        self.report.error = (
            "a live row's condition authorises an order; checkout runs in the sandbox only, so it did not run"
        )
        self.report.labels = ["live_checkout_refused"]
        return self.report

    def _refused_live_surface(self) -> EpisodeReport:
        """A live store is reached only through carttrace.live's allowlisted UCP over MCP client: a live row naming another
        surface never runs, and sends nothing."""
        self.report.failed = True
        self.report.error = (
            f"a live row names surface {self.surface.kind}; live rows run on UCP over MCP only, read-only, so it did "
            "not run"
        )
        self.report.labels = ["live_surface_refused"]
        return self.report

    def _refused_live_added(self) -> EpisodeReport:
        """detail_added serves a store's replica with data the store lacks; a live store cannot be given it, and a
        live row that ran anyway would be a baseline mislabelled as the added arm."""
        self.report.failed = True
        self.report.error = "a live row cannot take detail_added: the condition runs on the store's replica only"
        self.report.labels = ["live_detail_added_refused"]
        return self.report

    def _refused_live_faults(self) -> EpisodeReport:
        """A task's sealed faults (a substitute task's stockout) hold only in a sandbox world; a live store cannot be
        made to carry them, and a live row that ran anyway would judge an agent against a stockout the store never
        had. The manifest plans such tasks off live rows; this is the episode's own refusal, before anything is sent."""
        self.report.failed = True
        self.report.error = "a live row's task carries sealed faults, which only a sandbox world holds; it did not run"
        self.report.labels = ["live_faults_refused"]
        return self.report

    # -- the two openings ------------------------------------------------------------------------------------------

    async def _live(self) -> None:
        gate = self.ctx.gate
        if gate is None:
            raise SafetyError("a live row reached an episode without a live gate", hint="run it with --live")
        store_id, platform = self.row.store_id, self.store_ref.platform
        self.phase = "gate"
        try:
            gate.check(self.store_ref.domain, store_id=store_id)
        except SafetyError as exc:
            self._paused(exc)
            return
        self.phase = "discover"
        failure: BaseException | None = None
        try:
            self.discovery = await self.ctx.bindings.discover(
                self.store_ref.domain, gate=gate, recorder=self.buffer, store_id=store_id, platform=platform
            )
        except SafetyError as exc:
            # A refusal before anything was sent (a pause, a busy store, today's cap) waits or skips; one the store
            # earned (robots.txt disallows us) is its outcome, recorded like any other.
            if not self.buffer.records and code_of(exc) not in STORE_REFUSALS:
                self._paused(exc)
                return
            failure = exc
        except Exception as exc:  # recorded as the episode's outcome once the bundle is open
            failure = exc
        if self.discovery is not None and self.discovery.profile is not None:
            self.profile = self.discovery.profile
        if not self._open_live():
            return
        if failure is not None:
            await self._failed(failure)
            return
        await self._guarded(self._live_journey(gate))

    def _open_live(self) -> bool:
        """Open a live bundle and write the buffered discovery exchanges into it first."""
        endpoint = self.discovery.ucp_endpoint if self.discovery is not None else None
        if self.surface.kind in (WEBSITE, LIVE_WEBMCP):
            endpoint = f"https://{getattr(self.discovery, 'host', None) or self.store_ref.domain}/"
        if not self._open(url=endpoint or f"https://{self.store_ref.domain}"):
            return False
        assert self.writer is not None
        self.buffer.replay(self.writer)
        return True

    async def _live_journey(self, gate: LiveGate) -> None:
        assert self.discovery is not None and self.writer is not None
        if self.surface.kind == WEBSITE:
            await self._website_journey(gate)
            return
        if self.surface.kind == LIVE_WEBMCP:
            await self._webmcp_journey(gate)
            return
        ended = discovery_end(self.discovery)
        if ended is not None:
            self.labels.extend(f"discovery:{note}"[:LABEL_CHARS] for note in self.discovery.notes)
            self.report.skip_store = ended
            await self._finish(ended)
            return
        self.phase = "open"
        tap = WireTap(self.writer)
        surface = await self.ctx.bindings.open_ucp_surface(
            self.store_ref.domain,
            gate=gate,
            recorder=tap,
            store_id=self.row.store_id,
            platform=self.store_ref.platform,
            discovery=self.discovery,
            market=self.row.market if self.row.market_fill else None,
        )
        self.surfaces.append(surface)
        await self._journey(surface, tap, extras=await self._policy_companion(gate, tap))

    async def _policy_companion(self, gate: LiveGate, tap: WireTap) -> list[ToolSurface]:
        """The store's Storefront MCP policy tool beside a live UCP row, when live.policy_companion is on and the
        store is a Shopify store (D67); none otherwise, and none when the endpoint does not answer (noted)."""
        if not self.ctx.settings.live.policy_companion or self.store_ref.platform != "shopify":
            return []
        assert self.writer is not None
        try:
            companion = await self.ctx.bindings.open_store_mcp_surface(
                self.store_ref.domain,
                gate=gate,
                recorder=tap,
                store_id=self.row.store_id,
                platform=self.store_ref.platform,
                discovery=self.discovery,
            )
        except MCError as exc:
            self.writer.turn("system", "note", text=f"no Storefront MCP policy tool: {exc}"[:400], arguments={})
            return []
        self.surfaces.append(companion)
        return [companion]

    async def _website_journey(self, gate: LiveGate) -> None:
        """A store with no agent protocol, shopped through its own pages (D31): discovery's walls and 429s end it as
        they end any live row; a missing UCP document does not, since the row never asks for one."""
        assert self.discovery is not None and self.writer is not None
        ended = website_end(self.discovery)
        if ended is not None:
            self.labels.extend(f"discovery:{note}"[:LABEL_CHARS] for note in self.discovery.notes)
            self.report.skip_store = ended
            await self._finish(ended)
            return
        self.phase = "open"
        tap = WireTap(self.writer)
        currency = self.task.hidden_spec.get("currency")
        hints, edition = self._capture_hints()
        surface = await self.ctx.bindings.open_website_surface(
            self.store_ref.domain,
            gate=gate,
            recorder=tap,
            store_id=self.row.store_id,
            platform=self.store_ref.platform,
            discovery=self.discovery,
            currency=currency if isinstance(currency, str) else None,
            hints=hints,
            hints_edition=edition,
            market=self.row.market,
        )
        self.surfaces.append(surface)
        await self._journey(surface, tap)

    async def _webmcp_journey(self, gate: LiveGate) -> None:
        """A store's page-registered WebMCP tools, run in its page behind the allowlist (D67). Discovery's walls and
        429s end it as they end a website row; a missing UCP document does not, since the page's tools need none."""
        assert self.discovery is not None and self.writer is not None
        ended = website_end(self.discovery)
        if ended is not None:
            self.labels.extend(f"discovery:{note}"[:LABEL_CHARS] for note in self.discovery.notes)
            self.report.skip_store = ended
            await self._finish(ended)
            return
        self.phase = "open"
        tap = WireTap(self.writer)
        surface = await self.ctx.bindings.open_webmcp_surface(
            self.store_ref.domain,
            gate=gate,
            recorder=tap,
            store_id=self.row.store_id,
            platform=self.store_ref.platform,
            discovery=self.discovery,
        )
        self.surfaces.append(surface)
        await self._journey(surface, tap)

    def _capture_hints(self) -> tuple[list[tuple[str, str]], str | None]:
        """The (url, title) of every product our latest capture of the store holds, to seed the website's product
        index (D34) when collect.website_capture_hints is on, and the edition they come from. The note written here
        records every hint handed to the surface, so the judge can tell them from what the agent found itself; the
        index marks each result from them source "capture"."""
        if not self.ctx.settings.collect.website_capture_hints:
            return [], None
        catalog = self.ctx.bindings.capture_catalog(self.row.store_id)
        products = getattr(catalog, "products", None) or ()
        hints = [(str(p.url), str(p.title or "")) for p in products if getattr(p, "url", "")]
        if not hints:
            return [], None
        edition = str(getattr(catalog, "edition", "") or "") or None
        assert self.writer is not None
        self.writer.turn(
            "system",
            "note",
            text=(
                f"the website's product index is seeded with {len(hints)} product addresses from our capture "
                f"{edition} of this store; browse_index marks each result only the capture holds source capture"
            ),
            arguments={"source": "capture", "edition": edition, "hints": [url for url, _ in hints]},
        )
        return hints, edition

    async def _playground(self) -> None:
        """A store we own and run on this machine (D44): its website through the website road under its reserved
        name, or its UCP REST plugin on loopback. Never a live store: no live gate, its own rights, and a checkout row
        may buy with its offline gateways, the order read back from its admin API."""
        playground = self.ctx.playgrounds.get(self.row.store_id)
        if playground is None:
            self.report.failed = True
            self.report.error = f"{self.row.store_id} is not a playground this run knows (data/playgrounds/stores.json)"
            self.labels.append("playground_unknown")
            return
        url = f"https://{playground.host}/" if self.surface.kind == WEBSITE else str(playground.origin)
        if not self._open(url=url):
            return
        await self._guarded(self._playground_journey(playground))

    async def _playground_journey(self, playground: Any) -> None:
        assert self.writer is not None
        bridged = (
            f", addressed as https://{playground.host} through the playground bridge"
            if self.surface.kind == WEBSITE
            else ""
        )
        self.writer.turn(
            "system",
            "note",
            text=f"{playground.store_id} is a playground store we own, served at {playground.origin}{bridged}",
            arguments={"playground": playground.store_id, "origin": playground.origin, "host": playground.host},
        )
        self.phase = "open"
        tap = WireTap(self.writer)
        currency = self.task.hidden_spec.get("currency")
        surface = await self.ctx.bindings.open_playground_surface(
            playground,
            kind=self.surface.kind,
            settings=self.ctx.settings,
            recorder=tap,
            session_id=self.row.episode_id,
            agent_profile=self._agent_profile(),
            currency=currency if isinstance(currency, str) else None,
            market=self.row.market,
        )
        self.surfaces.append(surface)
        wallet: Any = None
        if self.checkout:
            if self.surface.kind == WEBSITE:
                raise ValidationError(
                    "a playground website row ends at a selection and cannot take checkout",
                    hint="checkout rows run on the playground's UCP surface; build a new manifest",
                )
            wallet = self.ctx.bindings.playground_wallet(persona_of(self.row.condition), market=self.row.market)
        await self._journey(surface, tap, wallet=wallet)

    async def _playground_order(self, cart: Cart | None, end_reason: str) -> tuple[Cart | None, str]:
        """On a playground checkout row that placed an order: the order as the admin API holds it becomes the final
        cart; one the checkout read never showed and the admin API cannot find was not placed, as far as we know."""
        if self.row.mode != "playground" or self.order_ref is None:
            return cart, end_reason
        admin = await self._admin_order(self.order_ref)
        if admin is None and self.order_unverified:
            self.labels.append("order_unconfirmed")
            return cart, "harness_error"
        return admin or cart, end_reason

    async def _admin_order(self, order: str) -> Cart | None:
        """The order as the playground's admin API (wc/v3) holds it, the ground truth after a checkout row: recorded
        as the harness's own read (no buyer name, email or address kept) and snapshotted as the final order."""
        assert self.writer is not None
        playground = self.ctx.playgrounds[self.row.store_id]
        self.phase = "admin_readback"
        read = await self.ctx.bindings.read_playground_order(playground, self.ctx.settings.home, order)
        record = self.writer.wire(
            {
                "surface": "wc_v3",
                "protocol": "harness",
                "version": "",
                "method": "GET",
                "url": read.url,
                "request_body": {"by": "harness", "reason": "admin_order_readback", "order": order},
                "status": read.status,
                "response_body": read.summary,
                "error": None,
            }
        )
        if read.cart is None:
            self.labels.append("admin_order_unread")
            return None
        self.writer.snapshot("playground:admin:get_order", read.cart, raw_ref=record.seq)
        return cast(Cart, read.cart)

    async def _sandbox(self) -> None:
        world = self.manifest.world(self.row)
        if self.row.envpack is None or world is None:
            raise ValidationError(f"sandbox row {self.row.episode_id} names no world", hint="build a new manifest")
        if self.surface.kind == WEBSITE or self.checkout or self.surface.kind in spoken_surfaces():
            # the header names the handlers a checkout row's world serves (D80), and the agent's context the store's
            # date, so the world is read before the bundle opens
            self._site_pack = await asyncio.to_thread(
                self.ctx.worlds.load, self.row.envpack, self.store_ref, world, self.profile
            )
        if self.surface.kind == WEBSITE:
            self.jsonld_types = site_jsonld_types(self._site_pack)
        if not self._open(url=f"sandbox://{self.row.envpack}/{self.surface.kind}"):
            return
        if self.surface.kind not in spoken_surfaces():
            await self._no_agent_dialect()
            return
        await self._guarded(self._sandbox_journey())

    async def _no_agent_dialect(self) -> None:
        """A sandbox row on a surface no agent dialect speaks (MPP, AP2 today) is ours, never the agent's or the
        store's: it ends harness_error with the label `no_agent_dialect` before the agent starts, so no agent turn is
        spent, and the judge voids it. The planner leaves such surfaces out; this is the backstop for a manifest
        built before that."""
        assert self.writer is not None
        self.labels.append(NO_AGENT_DIALECT)
        self.report.error = f"no agent dialect speaks {self.surface.kind} surfaces yet, so the row did not run"
        self.writer.turn("system", "note", text=self.report.error, arguments={"surface": self.surface.kind})
        await self._finish("harness_error")

    async def _sandbox_journey(self) -> None:
        assert self.writer is not None and self.row.envpack is not None
        world = self.manifest.world(self.row)
        assert world is not None
        self.phase = "world"
        pack = loaded = self._site_pack or await asyncio.to_thread(
            self.ctx.worlds.load, self.row.envpack, self.store_ref, world, self.profile
        )
        if added_of(self.row.condition):
            inventories = self.ctx.settings.paths.var / INVENTORY_DIR
            pack, note = await asyncio.to_thread(added_world, pack, self.row.condition, inventory_dir=inventories)
            self.writer.turn("system", "note", text=note, arguments={"envpack": pack.id})
        base = pack
        pack, sealed = await asyncio.to_thread(sealed_world, pack, self.task)
        if sealed is not None:
            faults = [str(k["id"]) for k in task_faults(self.task)]
            self.writer.turn(
                "system", "note", text=sealed, arguments={"envpack": pack.id, "from": base.id, "faults": faults}
            )
        drawn_from = pack
        pack, draw, said = await asyncio.to_thread(sold_out_world, pack, self.task, self.row.seed)
        if draw is not None and said is not None:
            facts = {
                "envpack": pack.id,
                "from": drawn_from.id,
                "sold_out": list(draw.variants),
                "spared": list(draw.spared),
            }
            self.writer.turn("system", "note", text=said, arguments=facts)
        faulted_from = pack
        pack, faulted = await asyncio.to_thread(
            fault_world, pack, self.fault_draw, three_ds="fail" if self._decides_three_ds() else "pass"
        )
        if faulted is not None and self.fault_draw is not None:
            ids = [fid for fault in self.fault_draw.faults for fid in fault.ids]
            self.writer.turn(
                "system", "note", text=faulted, arguments={"envpack": pack.id, "from": faulted_from.id, "faults": ids}
            )
        pack = await self._store_errors_drawn(pack)
        pack = await self._three_ds_world(await self._brokered(pack))
        await self._serve(pack, loaded, world)
        self.phase = "open"
        tap = WireTap(self.writer)
        paying = self.rail.payment_surface if self.rail is not None else None
        app = self._app(pack, shared=self.companion is not None or paying is not None)
        opened = await self._open_row_surface(pack, tap, app)
        self.surfaces.append(opened)
        self.watch = self.ctx.bindings.watch_sandbox(opened, self._sandbox_event)
        catalog: ToolSurface | None = None
        if self.companion is not None:
            catalog = await self.ctx.bindings.open_sandbox_surface(
                pack,
                kind=self.companion,
                recorder=tap,
                session_id=self.row.episode_id,
                agent_profile=self._agent_profile(),
                app=app,
            )
            self.surfaces.append(catalog)
        payment: ToolSurface | None = None
        if paying is not None:
            payment = await self.ctx.bindings.open_sandbox_surface(
                pack,
                kind=paying,
                recorder=tap,
                session_id=self.row.episode_id,
                agent_profile=self._agent_profile(),
                app=app,
            )
            self.surfaces.append(payment)
        wallet: Any = None
        if self.checkout:
            self.phase = "checkout_setup"
            wallet = await self.ctx.bindings.checkout(
                opened,
                persona=persona_of(self.row.condition),
                market=self.row.market,
                session_id=self.row.episode_id,
                recorder=tap,
                clock=self.ctx.clock,
                prefer=self.rail.protocol if self.rail is not None and self.rail.handler else None,
            )
            self._fit_wallet(wallet)
        await self._journey(opened, tap, wallet=wallet, catalog=catalog, payment=payment)
        self._guard_held()

    async def _open_row_surface(self, pack: EnvPack, tap: WireTap, app: Any) -> ToolSurface:
        """The row's own surface. A WebMCP storefront whose page timed out or aborted while it loaded is opened again
        on a fresh browser, up to collect.browser_open_tries times with a wait between (D122: under host load three
        rows of 2026-10-17 ended VOID on "Page.goto: Timeout 20000ms exceeded"); a note says each retry. A browser
        that is not installed, or no slot freed in sandbox.browser.slot_wait_s, is not tried again."""
        tries = self.ctx.settings.collect.browser_open_tries if self.surface.kind == STOREFRONT else 1
        for attempt in range(1, tries + 1):
            try:
                return await self.ctx.bindings.open_sandbox_surface(
                    pack,
                    kind=self.surface.kind,
                    recorder=tap,
                    session_id=self.row.episode_id,
                    agent_profile=self._agent_profile(),
                    options=self._surface_options(pack.id),
                    app=app,
                )
            except BrowserFailure as exc:
                if attempt >= tries or isinstance(exc, NoBrowserSlot):
                    raise
                wait = BROWSER_RETRY_S[min(attempt, len(BROWSER_RETRY_S)) - 1]
                assert self.writer is not None
                self.writer.turn(
                    "system",
                    "note",
                    text=f"the storefront did not open ({type(exc).__name__}); opening it again on a fresh browser "
                    f"in {wait:g} s",
                    arguments={"browser_retry": attempt, "error": type(exc).__name__},
                )
                self.labels.append(f"browser_retry:{type(exc).__name__}"[:LABEL_CHARS])
                await asyncio.sleep(wait)
        raise AssertionError("unreachable: the last try returns or raises")

    def _shopper(self) -> Any:
        """The row's shopper, its version written into the header (D69) once it is known."""
        assert self.decisions is not None and self.writer is not None
        shopper = self.ctx.bindings.shopper(self.task, self.decisions, live=self.row.mode == "live")
        if self.served_pack is not None and hasattr(shopper, "use_catalog"):
            # an order of a served product outside the task's shelf that meets every need is approved (2026-10-15)
            shopper.use_catalog(fixtures_from_snapshot(self.served_pack.snapshot))
        decided = self.manifest.shopper_map(self.row) if mapped(self.row) else None
        if decided is not None and hasattr(shopper, "use_map"):
            shopper.use_map(decided)  # D115: the shopper looks its decisions up in the task's build-time map
        if self.row.mode != "live" and hasattr(shopper, "use_wording"):
            # D118: the rules decide each reply, the shopper role's model words it (never on a live row)
            buyer = getattr(getattr(shopper, "spec", None), "buyer", None)
            voice = self.ctx.bindings.shopper_voice(
                self.ctx.settings, self.task.task_id, getattr(buyer, "persona", None)
            )
            if voice is not None:
                shopper.use_wording(voice)
        version = getattr(shopper, "version", None)
        if isinstance(version, str) and version:
            self.writer.amend_header(shopper_version=version)
        return shopper

    def _recovery(self, wallet: Any) -> dict[str, Any]:
        """D115: on a reference checkout row, the row's designed expectation for a failing payment, which the
        reference follows (`recover`); nothing on any other row."""
        if wallet is None or self.row.mode != "reference" or self.writer is None:
            return {}
        recover = designed_recovery(self.row, self.writer.header)
        return {} if recover is None else {"recover": recover}

    def _fit_wallet(self, wallet: Any) -> None:
        """The row's wallet given to AP2's signer (D66), and offering no challenge tool where the PSP decides 3-D
        Secure at once (D111): there is no bank page to hand the shopper."""
        if self._decides_three_ds() and hasattr(wallet, "offers_challenge"):
            wallet.offers_challenge = False
        if self.signer is not None:
            self.signer.wallet = wallet

    def _decides_three_ds(self) -> bool:
        """D111: whether the row's PSP decides 3-D Secure at once, accepted or declined, with no challenge page and no
        challenge tool: every v1 row, and any sandbox row where config's sandbox.three_ds says
        decision (the payments check)."""
        if self.row.mode not in ("sandbox", "reference"):
            return False
        v1 = self.row.experiment is not None and self.manifest.plan.v1 is not None
        return v1 or self.ctx.settings.sandbox.three_ds == "decision"

    async def _three_ds_world(self, pack: Any) -> Any:
        """The pack an order row runs on where its PSP decides 3-D Secure at once (D111): the row's rail on its
        policy, and whether the persona would abandon authentication, with a system note naming both packs; `pack`
        itself on every other row."""
        if not self.checkout or not self._decides_three_ds() or self.writer is None:
            return pack
        try:
            persona = self.ctx.bindings.personas().get(persona_of(self.row.condition))
        except (MCError, AttributeError):
            persona = None
        policy = {"three_ds": {"rail": self.row.rail, "abandon": getattr(persona, "three_ds", None) == "abandon"}}
        decided = await asyncio.to_thread(pack.derive, policy=policy)
        said = f"3-D Secure decided by the PSP at once, accepted or declined (D111); pack {decided.id} from {pack.id}"
        self.writer.turn("system", "note", text=said, arguments={"envpack": decided.id, "from": pack.id})
        return decided

    async def _brokered(self, pack: Any) -> Any:
        """The pack a row naming a payment broker runs on: the broker set in its policy (D71), with a system note
        naming both packs; `pack` itself on every other row."""
        if self.row.psp is None or self.writer is None:
            return pack
        brokered = await asyncio.to_thread(pack.derive, policy={"psp": self.row.psp})
        said = f"payment broker {self.row.psp} behind the store's PSP (D71); pack {brokered.id} from {pack.id}"
        facts = {"envpack": brokered.id, "from": pack.id, "psp": self.row.psp}
        self.writer.turn("system", "note", text=said, arguments=facts)
        return brokered

    def _sandbox_event(self, record: Mapping[str, Any]) -> None:
        """One record of the sandbox's event log, into events.jsonl. Called under the sandbox's lock, possibly on its
        worker thread: a record that cannot be written stops the file there and is labelled, and the store goes on
        answering the agent."""
        if self.events_failed or self.writer is None:
            return
        kind = str(record.get("type") or "")
        if kind == "fault.fired":
            self.fault_events.append(dict(record))
        elif kind.startswith("payment.") or kind == "command.replayed":
            self.payment_events.append(dict(record))
        try:
            self.writer.event(record)
        except (MCError, OSError) as exc:
            self.events_failed = True
            self.labels.append(f"events_unwritten:{type(exc).__name__}"[:LABEL_CHARS])

    def _state_digest(self) -> dict[str, Any] | None:
        """Detach the event sink and name what the sandbox's world ended as: its state hash and the chain's head."""
        watch, self.watch = self.watch, None
        if watch is None or self.writer is None:
            return None
        watch.detach(self._sandbox_event)
        if self.events_failed:
            return None
        return {
            "state": watch.state_hash(),
            "events": watch.digest,
            "count": self.writer.events_written,
            "env": watch.pack.id,
        }

    def _surface_options(self, pack_id: str) -> dict[str, Any] | None:
        """The row's surface options: a storefront takes the persona's 3-D Secure answer, the condition's detail
        filter, the browser limits from config and the episode's seed; AP2's merchant takes the shopper's wallet as
        the signer of its mandates (D66); a replica's website the task's currency and the row's market, as a live
        website row is given them (D67); the other kinds take none from a row."""
        if self.surface.kind == "ap2":
            from carttrace.agents.dialects.ap2 import WalletSigner

            self.signer = WalletSigner(pack_id)
            return {"signer": self.signer}
        if self.surface.kind == WEBSITE:
            currency = self.task.hidden_spec.get("currency")
            return {"currency": currency if isinstance(currency, str) else None, "market": self.row.market}
        if self.surface.kind != STOREFRONT:
            return None
        persona = persona_of(self.row.condition)
        three_ds = getattr(self.ctx.bindings.personas().get(persona), "three_ds", None) if self.checkout else None
        return webmcp_options(
            challenge=challenge_policy(three_ds),
            hide=hidden_of(self.row.condition),
            details=details_of(self.row.condition),
            on_hidden=self._details_hidden,
            limits=BrowserLimits.from_settings(self.ctx.settings),
            seed=self.row.seed,
        )

    # -- the journey -----------------------------------------------------------------------------------------------

    async def _journey(
        self,
        surface: ToolSurface,
        tap: WireTap,
        *,
        wallet: Any = None,
        catalog: ToolSurface | None = None,
        payment: ToolSurface | None = None,
        extras: Sequence[ToolSurface] = (),
    ) -> None:
        assert self.writer is not None
        settings, row = self.ctx.settings, self.row
        agent_surfaces = self._agent_surfaces(surface, catalog, payment)
        agent_surfaces += [ObservedSurface(extra, self._tool_failed) for extra in extras]
        assert self.decisions is not None
        shopper = self._shopper()
        supervisor = DecisionSupervisor(
            self.decisions,
            goal=self.task.opening,
            every=settings.collect.supervise_every_steps,
            boundary=not self.checkout,
        )
        limits = self.ctx.bindings.limits(
            step_cap=self._step_cap(),
            time_cap_s=settings.collect.time_cap_s,
            max_tokens=settings.agents.max_tokens,
        )
        extra: dict[str, Any] = {"wallet": wallet} if wallet is not None else {}
        if self.row.experiment is not None:
            extra["single_prompt"] = True  # D98: every v1 row reads agent-v6.md (D108), cart and order alike
        profile = self._buyer_profile() if wallet is not None else None
        if profile is not None:
            extra["profile"] = profile
        extra |= self._recovery(wallet)
        agent = self.ctx.bindings.agent(
            self.agent_mode, seed=agent_seed(row), clock=self.ctx.clock, context=self._agent_context(), **extra
        )
        self.phase = "agent"
        cap = settings.collect.time_cap_s
        result: AgentResult | None = None
        try:
            async with asyncio.timeout(runner_backstop_s(cap, min_s=BACKSTOP_MIN_S)) as deadline:
                result = await agent.run(
                    self.task, agent_surfaces, shopper, self.writer, limits=limits, supervisor=supervisor
                )
        except TimeoutError:
            if not deadline.expired():
                raise
            self.labels.append("runner_time_cap")
        # read by name: an agent result from before mc-twc (the reference agent's, a test double) has no such field
        if result is not None and getattr(result, "model_unavailable", False) is True:
            self.labels.append(MODEL_UNAVAILABLE)
        end_reason = "time_cap" if result is None else self._agent_end(result)
        if self.surface.kind == WEBSITE:
            await self._website_end(surface, tap, result, end_reason)
            return
        cart: Cart | None = None
        cart_id = None if result is None else _str_or_none(result.cart_id)
        if cart_id is not None:
            self.phase = "readback"
            cart = await self._read_cart(surface, tap, cart_id, claimed=end_reason == "cart_ready")
        elif end_reason == "cart_ready":
            self.labels.append("no_cart_id")
        if self.checkout and result is not None:
            self.phase = "order_readback"
            ordered, end_reason = await self._read_order(surface, tap, result, end_reason)
            cart = ordered or cart
            cart, end_reason = await self._playground_order(cart, end_reason)
        handoff: dict[str, Any] | None = None
        handing = row.mode == "live" and row.handoff and self.surface.kind != LIVE_WEBMCP
        if handing and end_reason == "cart_ready" and cart is not None and cart.lines:
            self.phase = "handoff"
            handoff = await self._handoff(cart, tap)
        await self._finish(end_reason, cart=cart, handoff=handoff)

    def _agent_surfaces(
        self, surface: ToolSurface, catalog: ToolSurface | None, payment: ToolSurface | None
    ) -> list[ToolSurface]:
        """What the agent shops through: the row's surface (without the tools its payment rail does not offer, and
        through the detail filter on a detail_hidden row), the rail's payment surface, and the catalog companion, each
        observed for failed calls."""
        hidden, details = hidden_of(self.row.condition), details_of(self.row.condition)
        agent_surface: ToolSurface = surface
        if self.rail is not None and self.rail.hides:
            agent_surface = WithoutTools(agent_surface, self.rail.hides)
        if (hidden or details) and self.surface.kind != STOREFRONT and catalog is None:
            agent_surface = DetailFilter(agent_surface, hidden, details=details, on_hidden=self._details_hidden)
        out: list[ToolSurface] = [ObservedSurface(agent_surface, self._tool_failed)]
        if payment is not None:
            out.append(ObservedSurface(payment, self._tool_failed))
        if catalog is not None:
            shown: ToolSurface = CatalogOnly(catalog)
            if hidden or details:
                shown = DetailFilter(shown, hidden, details=details, on_hidden=self._details_hidden)
            out.append(ObservedSurface(shown, self._tool_failed))
        return out

    async def _website_end(
        self, surface: ToolSurface, tap: WireTap, result: AgentResult | None, end_reason: str
    ) -> None:
        """A website journey's end: the agent's selection read back by the harness with a GET of each chosen page,
        and written as the "website:selection" snapshot the judge grades (carttrace.collect.selection). The outcome's
        final cart stays null; its selection block is the snapshot's."""
        selection_id = None if result is None else _str_or_none(result.cart_id)
        if end_reason == "cart_ready":
            end_reason = "selection_ready"
        if end_reason != "selection_ready" or selection_id is None:
            if end_reason == "selection_ready":
                self.labels.append("no_selection_id")
                end_reason = "gave_up"
            await self._finish(end_reason)
            return
        self.phase = "readback"
        assert self.writer is not None
        mark = len(tap.written)
        reply = await surface.call(self.readback.dialect.read, self.readback.dialect.read_arguments(selection_id))
        seen = [WireSeen(w.seq, w.method, w.url, w.status) for w in tap.written[mark:]]
        payload = payload_of(reply)
        if reply.is_error and reply.error_code in ("blocked", "throttled"):
            self.labels.append(f"selection_reread:{reply.error_code}")
            end_reason = str(reply.error_code)
            self.report.skip_store = end_reason
        elif reply.is_error and reply.error_code != "selection_unverified":
            self.labels.append(f"selection_readback_failed:{reply.error_code or 'error'}"[:LABEL_CHARS])
        doc = payload if isinstance(payload, Mapping) else {}
        record = selection_record(doc, seen, await self._selection_catalog())
        if record is None:
            self.labels.append("selection_unreadable")
            await self._finish("gave_up" if end_reason == "selection_ready" else end_reason)
            return
        if not record.selection["verified"]:
            self.labels.append("selection_unverified")
        self.writer.snapshot(SELECTION_SOURCE, record.cart, raw_ref=record.raw_ref, selection=record.selection)
        await self._finish(end_reason, selection=record.selection)

    def _admin_confirms(self, doc: Mapping[str, Any], order_id: str | None) -> bool:
        """Whether a completed checkout whose read names no order is the playground's to confirm: its admin API holds
        the ground truth, and the order id the completion answered is what to look for."""
        return self.row.mode == "playground" and doc.get("status") == "completed" and order_id is not None

    async def _selection_catalog(self) -> Any:
        """The ids a website selection is named in: a playground's own (its Store API), a replica's own (its pack,
        D67), else our capture's."""
        playground = self.ctx.playgrounds.get(self.row.store_id) if self.row.mode == "playground" else None
        if playground is not None:
            return await self.ctx.bindings.playground_catalog(playground)
        replica = next((getattr(s, "selection_catalog", None) for s in self.surfaces), None)
        if replica is not None:
            return replica
        return self.ctx.bindings.capture_catalog(self.row.store_id)

    def _agent_end(self, result: AgentResult) -> str:
        reason = str(result.end_reason)
        if reason not in END_REASONS:
            self.labels.append(f"agent_end_reason_unknown:{reason}"[:LABEL_CHARS])
            return "harness_error"
        if reason == "order_placed" and self.row.mode == "live":
            self.labels.append("live_order_claimed")
            return "harness_error"
        return reason

    async def _read_cart(self, surface: ToolSurface, tap: WireTap, cart_id: str, *, claimed: bool) -> Cart | None:
        """Read the agent's cart back by the id the store returned (a keyed Shopify gid whole, `?key=` included) and
        snapshot it as `get_cart`. A read that writes no snapshot always says why in a label: the call failed, the reply
        carried the store's error instead of a cart, the cart could not be read, or a cart the agent called ready was
        gone. Seen 2026-10-13 (sbx-replica-0003): Shopify-pack REST and storefront rows wrote no snapshot and the judge
        voided them as cart_unread."""
        assert self.writer is not None
        read = await self._read_back(surface, tap, self.readback.cart_call(cart_id), "cart")
        if read is None:
            return None
        payload, raw_ref = read
        if self.readback.cart_gone(payload):
            # The storefront moved past its cart; the checkout and the order carry the lines.
            if claimed:
                self.labels.append("cart_gone")
            return None
        cart = self.readback.cart_of(payload)
        if cart is None:
            refused = self.readback.refusal_of(payload)
            self.labels.append(f"cart_readback_failed:{refused}"[:LABEL_CHARS] if refused else "cart_unreadable")
            return None
        self.writer.snapshot(f"{self.writer.header.mode}:get_cart", cart, raw_ref=raw_ref)
        return cart

    async def _read_order(
        self, surface: ToolSurface, tap: WireTap, result: AgentResult, end_reason: str
    ) -> tuple[Cart | None, str]:
        """Read the checkout and the order back from the store and snapshot them; the cart they show (the order's lines
        when there is one), and the end reason. The store's word decides it: order_placed stays only when the store
        shows the order, and an order the agent did not claim (a step cap reached just after paying, say) still makes
        the episode order_placed, so the judge checks it against the shopper's approval like any other."""
        assert self.writer is not None
        checkout_id = _str_or_none(getattr(result, "checkout_id", None))
        order_id = _str_or_none(getattr(result, "order_id", None))
        if checkout_id is None:
            if end_reason == "order_placed":
                self.labels.append("no_checkout_id")
            return None, end_reason
        final: Cart | None = None
        read = await self._read_back(surface, tap, self.readback.checkout_call(checkout_id), "checkout")
        if read is not None:
            payload, raw_ref = read
            doc = self.readback.checkout_of(payload)
            view = self.readback.checkout_view(doc) if doc is not None else None
            if doc is None or view is None:
                self.labels.append("checkout_unreadable")
            else:
                self.writer.snapshot(f"{self.writer.header.mode}:get_checkout", view, raw_ref=raw_ref)
                final = view
                placed = self.readback.order_in(doc)
                if placed is None and self._admin_confirms(doc, order_id):
                    # The playground's UCP plugin shows a completed checkout without its order on a read (2026-10-13
                    # smoke); the order the completion named is taken, and the admin API must confirm it.
                    placed = order_id
                    self.order_unverified = True
                if placed is None and end_reason == "order_placed":
                    self.labels.append("order_unconfirmed")
                    return final, "harness_error"
                if placed is not None and end_reason != "order_placed":
                    self.labels.append(f"order_unclaimed:{end_reason}"[:LABEL_CHARS])
                    end_reason = "order_placed"
                order_id = placed or order_id
                self.order_ref = _order_number(doc) or order_id
                if order_id is not None and self.readback.order_call(order_id) is None:
                    # No order tool on this kind: the order came with the checkout reply, under the same wire record.
                    ordered = self.readback.order_from_checkout(payload, doc)
                    if ordered is None:
                        self.labels.append("order_unreadable")
                    else:
                        self.writer.snapshot(f"{self.writer.header.mode}:get_order", ordered, raw_ref=raw_ref)
                        final = ordered
                    return final, end_reason
        if order_id is not None:
            final = await self._read_placed(surface, tap, order_id) or final
        return final, end_reason

    async def _read_placed(self, surface: ToolSurface, tap: WireTap, order_id: str) -> Cart | None:
        assert self.writer is not None
        call = self.readback.order_call(order_id)
        if call is None:
            return None
        read = await self._read_back(surface, tap, call, "order")
        if read is None:
            return None
        payload, raw_ref = read
        view = self.readback.order_of(payload)
        if view is None:
            self.labels.append("order_unreadable")
            return None
        self.writer.snapshot(f"{self.writer.header.mode}:get_order", view, raw_ref=raw_ref)
        return view

    async def _read_back(
        self, surface: ToolSurface, tap: WireTap, call: tuple[str, dict[str, Any]], what: str
    ) -> tuple[Any, int | None] | None:
        """One read-back call through the surface (never the filter): its payload and the seq of its wire record."""
        tool, arguments = call
        before = tap.last_seq
        result = await surface.call(tool, arguments)
        raw_ref = tap.last_seq if tap.last_seq != before else None
        if result.is_error:
            self.labels.append(f"{what}_readback_failed:{result.error_code or 'error'}"[:LABEL_CHARS])
            await self._call_failed(tool, result)
            return None
        return payload_of(result), raw_ref

    async def _handoff(self, cart: Cart, tap: WireTap) -> dict[str, Any] | None:
        assert self.writer is not None
        try:
            handoff = await hand_off(
                cart,
                store=self.store_ref,
                profile=self.profile,
                discovery=self.discovery,
                rights=self.writer.header.rights,
                worlds=self.ctx.worlds,
                open_surface=self.ctx.bindings.open_sandbox_surface,
                recorder=tap,
                session_id=f"{self.row.episode_id}.handoff",
                profile_url=self._agent_profile(),
                clock=self.ctx.clock,
                opened=self.surfaces,
            )
        except (MCError, ValueError) as exc:
            # The journey reached its cart; a hand-off that fails is ours, recorded, and the outcome stays cart_ready.
            self.labels.append(f"handoff_failed:{type(exc).__name__}")
            self.report.error = _clip(str(exc))
            return None
        if handoff.get("error_code"):
            self.labels.append(f"handoff_checkout_refused:{handoff['error_code']}"[:LABEL_CHARS])
            refused = ToolResult(content="", is_error=True, error_code=str(handoff["error_code"]))
            await self._call_failed("create_checkout", refused, mode="sandbox")
        return handoff

    def _details_hidden(self, tool: str, counts: Mapping[str, int]) -> None:
        """The detail filter kept catalog details from the agent in one reply. Why a note turn: the condition's params
        say what the experiment meant to hide before anything ran, and only the reply shows whether the store had it
        to hide; a system note sits in the same ordered log as the call, right after the wire record holding the
        store's full reply, as the runner's other notes do."""
        assert self.writer is not None
        listed = ", ".join(f"{name} {count}" for name, count in counts.items())
        self.writer.turn(
            "system",
            "note",
            text=f"detail_hidden removed from the {tool} reply: {listed}",
            arguments={"tool": tool, "hidden": dict(counts)},
        )
        self.hidden.update(counts)

    async def _tool_failed(self, tool: str, arguments: dict[str, Any], result: ToolResult, attempts: int) -> None:
        """An agent's call failed. The evidence names the tool and the error, never the arguments."""
        del arguments
        await self._call_failed(tool, result, attempts=attempts)

    async def _call_failed(
        self, tool: str, result: ToolResult, *, attempts: int = 1, mode: RunMode | None = None
    ) -> None:
        """Triage a failed call, the agent's or the runner's own (the read-back, the hand-off)."""
        failure = failure_of_result(tool, result, mode=mode or self.row.mode, salt=self.ctx.salt)
        decided = await self._triage(failure, attempts)
        if decided is not None and decided.store_action and self.row.mode == "live" and mode != "sandbox":
            self.report.skip_store = decided.store_action

    # -- endings ---------------------------------------------------------------------------------------------------

    async def _guarded(self, journey: Awaitable[None]) -> None:
        """Run the part of the episode after the bundle opened; whatever it raises is recorded as its outcome. A
        journey whose bundle gets no record for collect.idle_timeout_s is cut and recorded as `idle_timeout` (D122)."""
        try:
            await self._within_idle_deadline(journey)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # one episode's failure is recorded in its bundle, never the end of the run
            await self._failed(exc)

    async def _within_idle_deadline(self, journey: Awaitable[None]) -> None:
        """Await `journey`, cancelling it when the bundle's record count stays still for collect.idle_timeout_s:
        then raise EpisodeIdle, which the episode records as harness_error with the label `idle_timeout`."""
        limit = self.ctx.settings.collect.idle_timeout_s
        # waiting for a browser slot is queueing, not a hang: a storefront row's opening may wait slot_wait_s more
        queueing = self.ctx.settings.sandbox.browser.slot_wait_s if self.surface.kind == STOREFRONT else 0.0
        task = asyncio.ensure_future(journey)
        seen, still = self._records(), 0.0
        try:
            while not task.done():
                await asyncio.wait([task], timeout=min(IDLE_CHECK_S, limit))
                if task.done():
                    break
                now = self._records()
                still = 0.0 if now != seen else still + min(IDLE_CHECK_S, limit)
                seen = now
                if still >= limit + (queueing if self.phase == "open" else 0.0):
                    task.cancel()
                    await asyncio.wait([task])
                    if self.ended:  # it was cut while finishing: the outcome stands
                        return
                    self.labels.append(IDLE_TIMEOUT)
                    raise EpisodeIdle(
                        f"no record for {limit:g} s in phase {self.phase}: the episode was cut and its row runs again",
                        hint="a hung page, tool or model call; collect.idle_timeout_s sets the limit",
                    )
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.wait([task])
            raise
        task.result()

    def _records(self) -> int:
        return self.writer.records if self.writer is not None else 0

    async def _failed(self, exc: BaseException) -> None:
        failure = failure_of(
            exc, phase=self.phase, mode=self.row.mode, model_names=self.ctx.model_names, salt=self.ctx.salt
        )
        self.report.error = failure.message
        self.labels.append(f"{self.phase}:{type(exc).__name__}"[:LABEL_CHARS])
        if isinstance(exc, VisibilityLeak):
            self.labels.append(f"visibility_leak:{exc.kind}"[:LABEL_CHARS])
        decided = await self._triage(failure, 1)
        if decided is not None:
            self.labels.extend(decided.labels())
            if decided.store_action and self.row.mode == "live":
                self.report.skip_store = decided.store_action
        end = store_outcome(exc, phase=self.phase, mode=self.row.mode)
        if end is not None:
            self.report.skip_store = end
        if self.ended:
            self.report.failed = True  # the outcome was written; its bundle stays staged for recover_staged
            return
        await self._finish(end or "harness_error")

    async def _triage(self, failure: Failure, attempts: int) -> Triage | None:
        recording = self.decisions is not None and self.decisions.recording
        try:
            if recording and self.decisions is not None:
                return await triage(self.decisions, failure, attempts=attempts)
            return await triage(self.ctx.engine, failure, attempts=attempts, context=self._context())
        except MCError as exc:
            self.labels.append(f"triage_failed:{type(exc).__name__}")
            return None

    async def _finish(
        self,
        end_reason: str,
        *,
        cart: Cart | None = None,
        handoff: dict[str, Any] | None = None,
        selection: dict[str, Any] | None = None,
    ) -> None:
        if selection is not None:
            boundary = "selection"
        elif handoff is not None and handoff.get("checkout_id"):
            boundary = "checkout_handoff"
        elif end_reason == "order_placed":
            boundary = "order"
        elif end_reason == "cart_ready":
            boundary = "cart"
        else:
            boundary = "none"
        self._record_outcome(end_reason, boundary=boundary, cart=cart, handoff=handoff, selection=selection)
        # Surfaces close first: closing one ends its MCP session with a request the bundle must record. Seen
        # 2026-10-11: closed after the writer, every episode logged "the writer is closed" and lost that exchange.
        await self._close_surfaces()
        await self._close_writer()

    async def _close_writer(self) -> None:
        assert self.writer is not None
        self.closing = True
        upload = asyncio.ensure_future(self.writer.aclose())
        try:
            self.report.prefix = await asyncio.shield(upload)
        except MCError as exc:
            self.report.failed = True
            self.report.error = _clip(f"bundle not uploaded, left staged for recovery: {exc}")
            self.labels.append(f"close:{type(exc).__name__}")

    def _end_now(self, label: str) -> None:
        """Record and close at once: the episode was cancelled, and another await could be cancelled too."""
        self.labels.append(label)
        if self.writer is None and self.buffer.records:
            self._open_live()  # requests went out during discovery: they are recorded whatever happens
        if self.writer is None:
            if not self.report.paused:
                self.report.failed = True
                self.report.error = self.report.error or f"cancelled before its bundle opened ({label})"
            return
        if self.closing:
            return  # the upload runs on in its thread; recover_staged finishes it if the process stops first
        if not self.ended:
            self._record_outcome("harness_error", boundary="none", cart=None, handoff=None)
        try:
            self.report.prefix = self.writer.close()
        except MCError as exc:
            self.report.failed = True
            self.report.error = _clip(f"bundle not uploaded, left staged for recovery: {exc}")

    def _record_outcome(
        self,
        end_reason: str,
        *,
        boundary: str,
        cart: Cart | None,
        handoff: dict[str, Any] | None,
        selection: dict[str, Any] | None = None,
    ) -> None:
        assert self.writer is not None
        if self.decisions is not None:
            self.decisions.recording = False
            if self.row.experiment is not None and decider_down(self.decisions.unavailable, end_reason):
                # D115: a shopper or supervisor decision whose model was down shaped the row: ours, re-run, never a
                # finished trace (the judge voids it as decider_unavailable)
                self.labels.append(DECIDER_UNAVAILABLE)
                end_reason, boundary = "harness_error", "none"
        self._psp_record()
        state_digest = self._state_digest()
        outcome = Outcome.model_validate(
            {
                "end_reason": end_reason,
                "boundary": boundary,
                "final_cart": None if cart is None else cart.model_dump(),
                "handoff": handoff,
                "infra_labels": sorted(set(self.labels)),
                "counts": {f"hidden.{name}": count for name, count in sorted(self.hidden.items())},
                "duration_ms": max(0, round((self.ctx.clock.monotonic() - self.started) * 1000)),
                "ended_at": iso(self.ctx.clock.now()),
                **({"selection": selection} if selection is not None else {}),
                **({"state_digest": state_digest} if state_digest is not None else {}),
                **self._fault_firings(),
            }
        )
        if self.row.experiment is not None and not self._experiment_written():
            # D98: a v1 row whose experiment.json or exposure.json could not be written is a void the loop re-runs
            outcome = outcome.model_copy(
                update={"end_reason": "harness_error", "infra_labels": sorted(set(self.labels))}
            )
        recorded = self.writer.outcome(outcome)
        self.ended = True
        self.report.end_reason = recorded.end_reason
        self.report.ended_at = recorded.ended_at
        self.report.failed = recorded.end_reason == "harness_error"

    def _fault_firings(self) -> dict[str, Any]:
        """The outcome's checkout_faults: whether the row's drawn fault fired, from the store's own records."""
        found = firings(self.fault_draw, self.fault_events)
        return {} if found is None else {"checkout_faults": [f.model_dump(mode="json") for f in found]}

    def _psp_record(self) -> None:
        """On a sandbox checkout row whose sandbox opened: the PSP's own record of every payment it decided (D66), as a
        harness wire record, so the charged amount joins the shopper's consent (by checkout id) and the order read back
        (by order id). Written once, before the event sink is detached."""
        watch = self.watch
        if watch is None or self.writer is None or not self.checkout or self.events_failed:
            return
        try:
            found = psp_authorisations(watch.state, self.payment_events, self.row.psp)
        except (AttributeError, TypeError) as exc:  # a test's fake sandbox holds no core state
            self.labels.append(f"psp_unread:{type(exc).__name__}"[:LABEL_CHARS])
            return
        self.authorisations = found
        self.writer.wire(
            {
                "surface": PSP_SURFACE,
                "protocol": "harness",
                "version": "",
                "method": "READ",
                "url": f"sandbox://{watch.pack.id}/psp/payments",
                "request_body": {"by": "harness", "reason": "psp_authorisations"},
                "status": 200,
                "response_body": {"authorisations": found},
                "error": None,
            }
        )

    def _paused(self, exc: SafetyError) -> None:
        self.report.paused = True
        self.report.error = _clip(str(exc))
        if code_of(exc) == DAILY_CAP:
            self.report.skip_store = DAILY_CAP

    async def _close_surfaces(self) -> None:
        while self.surfaces:
            surface = self.surfaces.pop()
            try:
                await surface.aclose()
            except Exception as exc:  # a surface that fails to close is reported; the bundle is already whole
                if self.writer is not None and not self.writer.is_open:
                    # closed after the bundle (a cancellation cut its close short): the exchange has nowhere to go
                    self.labels.append(SURFACE_CLOSE_UNRECORDED)
                    continue
                self.labels.append(f"surface_close:{type(exc).__name__}")
                self.report.error = self.report.error or _clip(f"closing a surface failed: {exc}")

    # -- the header ------------------------------------------------------------------------------------------------

    def _open(self, *, url: str) -> bool:
        """Open the bundle. False when it cannot be opened: nothing was recorded, and the report says why."""
        self.phase = "open_writer"
        try:
            self.writer = TraceWriter.open(
                self.ctx.store,
                self._header(url),
                staging_dir=self.ctx.staging_dir,
                clock=self.ctx.clock,
                redaction_salt=self.ctx.salt,
            )
        except MCError as exc:
            self.report.failed = True
            self.report.error = _clip(f"the bundle could not be opened: {exc}")
            return False
        header = self.writer.header
        self.report.provenance = provenance_of(header)
        self.report.store_source = store_source_of(header)
        self.report.surface = surface_of(header)
        self.report.agent = agent_of(header)
        self.decisions = EpisodeDecisions(self.ctx.engine, self.writer, self._context())
        return True

    def _experiment_written(self) -> bool:
        """Write experiment.json (v1 rows) and say whether the bundle holds both v1 files; a failure is labelled."""
        try:
            self._experiment()
        except (MCError, PydanticValidationError, KeyError) as exc:
            self.labels.append("experiment_unwritten")
            text = f"no experiment.json: {type(exc).__name__}: {str(exc)[:200]}"
            assert self.writer is not None
            self.writer.turn("system", "note", text=text, arguments={"experiment": "failed"})
            return False
        assert self.writer is not None
        return "exposure_sheet_failed" not in self.labels and (self.writer.path / EXPOSURE_FILE).is_file()

    def _experiment(self) -> None:
        """D96: a sandbox v1 row writes experiment.json as it ends, so the judge reads what kind of trace it is
        (baseline or designed failure, cart or order), what was removed or injected, the expected outcome and, on an
        order row, the turns where checkout opened, the shopper consented and payment was sent (D98)."""
        assert self.writer is not None
        turns = self.writer.path / TURNS_FILE
        found = order_stages(read_jsonl(turns)) if turns.is_file() else None
        fired_list = firings(self.fault_draw, self.fault_events)
        fired = any(f.fired for f in fired_list) if fired_list else None
        header = self.writer.header
        # final review #11: a persona fault counts as a designed failure only where it acted
        persona = persona_fired(header.payment_fault, self.authorisations, read_jsonl(turns) if turns.is_file() else ())
        decided = three_ds_decision(self.payment_events)
        config = experiment_config(
            self.row,
            self.manifest,
            header,
            stages=found,
            fired=fired,
            persona=persona,
            three_ds=ThreeDs.model_validate(decided) if decided is not None else None,
        )
        if config is not None:
            self.writer.experiment_config(config)

    def _header(self, url: str) -> EpisodeHeader:
        settings, row = self.ctx.settings, self.row
        live = row.mode == "live"
        mode = "live" if live else "playground" if row.mode == "playground" else "sandbox"
        return EpisodeHeader.model_validate(
            {
                "schema": "mc.trace.v1",
                "episode_id": row.episode_id,
                "run_id": self.ctx.run_id,
                "manifest_id": self.manifest.id,
                "mode": mode,
                "started_at": iso(self.ctx.clock.now()),
                "store": self.store_ref.model_dump(),
                "task": self.task.model_dump(),
                "condition": row.condition.model_dump(),
                "profile": self.profile.model_dump(),
                "surfaces": [
                    SurfaceRef(
                        kind=self.surface.kind,
                        mode=mode,
                        url=url,
                        version=self._surface_version(),
                    ).model_dump(),
                    *self._companion_refs(),
                    *self._payment_refs(),
                ],
                "models": [self._model(m).model_dump() for m in self.ctx.models.get(self.agent_mode, ())],
                "seed": row.seed,
                "limits": {
                    "step_cap": self._step_cap(),
                    "time_cap_s": settings.collect.time_cap_s,
                    "max_tokens": settings.agents.max_tokens,
                },
                "code_version": self.ctx.code_version,
                "envpack": row.envpack,
                "rights": self._rights().model_dump(),
                "pair_id": row.pair_id,
                "platform_pack": row.platform_pack,
                "boundary": "selection" if self.surface.kind == WEBSITE else None,
                "market_context": self._market_context(),
                "served_currency": self._served_currency(),
                "sold_out": self._sold_out_rate(),
                "schema_minor": SCHEMA_MINOR,
                "generator_version": self._generator_version(),
                "judge_code_version": settings.judge.version,
                **self._rail_fields(),
                **self._checkout_fault_draw(),
                **self._payment_fields(),
                "jsonld_types": self.jsonld_types,
                "exposure": self._exposure(),
                "store_replies": self._store_replies(),
                **self._sandbox_version(),
            }
        )

    def _sandbox_version(self) -> dict[str, Any]:
        """D96: a v1 row's version and the locked spec it was built under; nothing on any other row."""
        recipe = self.manifest.plan.v1
        if self.row.experiment is None or recipe is None:
            return {}
        return {"sandbox_version": recipe.sandbox_version, "sandbox_spec": recipe.spec}

    def _rail_fields(self) -> dict[str, Any]:
        """The header's rail, its protocol and version, and whether it is adopted (D66); nothing off a rail."""
        if self.rail is None or self.row.rail is None:
            return {}
        return {
            "rail": self.rail.name,
            "rail_protocol": self.rail.protocol,
            "rail_version": rail_version(self.rail.name, self.profile),
            "adopted": bool(self.row.adopted),
            **({"surface_adopted": self.row.surface_adopted} if self.row.surface_adopted is not None else {}),
            **({"psp": self.row.psp} if self.row.psp is not None else {}),
        }

    def _checkout_fault_draw(self) -> dict[str, Any]:
        """The checkout fault a sandbox checkout row draws by its episode id from the manifest's rates (D66), kept for
        the world it opens on and the outcome; nothing on any other row."""
        rates = self.manifest.plan.checkout_faults
        if rates is None or not self.checkout or self.row.mode not in ("sandbox", "reference"):
            return {}
        forced = forced_kind(self.row.condition)
        if forced is not None:
            # D96: a v1 row built to meet this checkout fault draws it for certain; no other v1 row draws any
            rates = forced_faults(rates, forced)
        self.fault_draw = draw_faults(
            rates, episode_id=self.row.episode_id, task=self.task, country=self._buyer_country(), rail=self.row.rail
        )
        return {"checkout_faults": self.fault_draw.model_dump(mode="json")}

    def _payment_fields(self) -> dict[str, Any]:
        """D80: a sandbox order row's payment as designed (its rail, what the store takes, the persona's wallet and
        the instrument it pays with) and the fault it is designed to meet, for the dashboard and the judge to tell an
        intended failure from the agent's. Read after the checkout fault draw, which it names. Nothing on any other
        row."""
        if not self.checkout or self.row.mode not in ("sandbox", "reference") or self._site_pack is None:
            return {}
        try:
            persona = self.ctx.bindings.personas().get(persona_of(self.row.condition))
            handlers = sorted(set(self._site_pack.handlers().values()))
        except (MCError, AttributeError):
            if self.row.experiment is not None:
                raise  # a v1 row's header must carry its payment design (final review #11)
            return {}
        planned = planned_payment(persona, self.row.rail, handlers)
        payment = PaymentPlan(
            rail=self.row.rail,
            store_accepts=list(planned.store_accepts),
            handlers=handlers,
            wallet=[WalletInstrument(kind=k, token=t) for k, t in planned.wallet],
            pays_with=WalletInstrument(kind=planned.pays_with[0], token=planned.pays_with[1])
            if planned.pays_with is not None
            else None,
        )
        fault = self._payment_fault(planned, persona)
        return {"payment": payment.model_dump(mode="json"), "payment_fault": fault}

    def _payment_fault(self, planned: PlannedPayment, persona: Any) -> dict[str, Any] | None:
        """How the row's payment is designed to fail (D80): the persona's token first (it fails before any checkout
        fault can), then no usable instrument, then the checkout fault draw's decline or challenge; None when the row
        is built to pay."""
        intended = self.row.payment_fault is not None
        if planned.fault in PERSONA_FAULTS:
            token = planned.pays_with[1] if planned.pays_with is not None else None
            # recoverable when the wallet holds another instrument the rail takes (final review #11: a designed
            # decline paid with Google Pay instead read as unrecoverable in the header and recoverable in experiment)
            other = any(t != token and k in planned.store_accepts for k, t in planned.wallet)
            found = PaymentFault(
                kind=planned.fault,
                source="persona",
                planned=intended,
                token=token,
                code=planned.code,
                recoverable=other,
            )
            return found.model_dump(mode="json")
        if planned.fault == "no_instrument":
            source = "plan" if self.row.payment_fault == "no_instrument" else "unplanned"
            return PaymentFault(kind="no_instrument", source=source, planned=intended).model_dump(mode="json")
        drawn = self.fault_draw.faults if self.fault_draw is not None else []
        for fault in drawn:
            if fault.kind == "payment_decline":
                code = f"payment_decline:{fault.decline_class or 'buyer_fixable'}"
                recoverable = fault.decline_class == "retryable"
            elif fault.kind == "three_ds" and self._decides_three_ds():
                # D111: the issuer authenticates and declines at once; another instrument the rail takes still pays
                token = planned.pays_with[1] if planned.pays_with is not None else None
                code = "three_ds:declined"
                recoverable = any(t != token and k in planned.store_accepts for k, t in planned.wallet)
            elif fault.kind == "three_ds":
                code, recoverable = "three_ds", getattr(persona, "three_ds", "complete") == "complete"
            else:
                continue
            found = PaymentFault(
                kind="checkout_fault", source="checkout_fault", planned=True, code=code, recoverable=recoverable
            )
            return found.model_dump(mode="json")
        return None

    def _buyer_country(self) -> str | None:
        """The country of the persona's delivery address in the row's market, which an unshippable fault refuses."""
        try:
            persona = self.ctx.bindings.personas().get(persona_of(self.row.condition))
            address = persona.address_for(self.row.market)
        except (MCError, AttributeError):
            if self.row.experiment is not None:
                raise  # a v1 row's unshippable fault needs the buyer's country (final review #11)
            return None
        country = address.get("address_country") if isinstance(address, Mapping) else None
        return country if isinstance(country, str) and country else None

    def _generator_version(self) -> str | None:
        """The generator that wrote the task's hidden spec (D69), as the spec names it; None for a hand-written one."""
        generator = self.task.hidden_spec.get("generator")
        return generator if isinstance(generator, str) and generator else None

    def _sold_out_rate(self) -> dict[str, Any] | None:
        """The rate a replica row's world answers sold out at (D52), as its recipe names it; None elsewhere."""
        live = self.row.mode in ("live", "playground")
        world = None if live or self.row.envpack is None else self.manifest.worlds.get(self.row.envpack)
        rate = world.sold_out if world is not None else None
        return rate.model_dump(mode="json") if rate is not None else None

    def _exposure(self) -> dict[str, Any] | None:
        """The exposure mask a sandbox row's world is served under (D63), as its recipe names it; None elsewhere."""
        live = self.row.mode in ("live", "playground")
        world = None if live or self.row.envpack is None else self.manifest.worlds.get(self.row.envpack)
        mask = world.exposure if world is not None else None
        return mask.model_dump(mode="json") if mask is not None else None

    async def _serve(self, pack: EnvPack, loaded: EnvPack, world: World) -> None:
        """Serve `pack` (the world as the episode runs it, from `loaded`, the row's world) and write exposure.json
        (D93): what the store exposes per data type and surface, from the world before its mask and the pack the
        surfaces serve. A sheet that cannot be computed is noted in the turns, never fatal to the episode: the
        journey is the data, the sheet its reading."""
        assert self.writer is not None
        self.served_pack = pack
        cond = self.row.condition
        try:
            base = await asyncio.to_thread(self.ctx.worlds.unmasked, loaded, self.store_ref, world, self.profile)
            inputs = SheetInputs(
                header=self.writer.header,
                served=pack,
                base=base,
                profile=None,
                var=self.ctx.settings.paths.var,
                row_kind=self.surface.kind,
                hide=tuple(hidden_of(cond)),
                details=tuple(details_of(cond)),
            )
            sheet = await asyncio.to_thread(build_sheet, inputs)
            self.writer.exposure_sheet(sheet)
        except Exception as exc:  # recorded in the bundle, and the episode goes on without a sheet
            text = f"no exposure sheet: {type(exc).__name__}: {str(exc)[:200]}"
            self.writer.turn("system", "note", text=text, arguments={"exposure_sheet": "failed"})
            if self.row.experiment is not None:
                # D98: a v1 bundle needs its sheet; the row ends harness_error (re-run), never a sheetless trace
                self.labels.append("exposure_sheet_failed")

    async def _store_errors_drawn(self, pack: EnvPack) -> EnvPack:
        """The pack with the episode's failing calls drawn at its store's error rates (D70), noted when it names any."""
        assert self.writer is not None
        drawn, failing = await asyncio.to_thread(store_errors_world, pack, self.row.seed)
        if failing is not None:
            self.writer.turn("system", "note", text=failing, arguments={"envpack": drawn.id, "from": pack.id})
        return drawn

    def _store_replies(self) -> dict[str, Any] | None:
        """What a replica row's world serves from its store's live replies (D70), as its recipe names it."""
        live = self.row.mode in ("live", "playground")
        world = None if live or self.row.envpack is None else self.manifest.worlds.get(self.row.envpack)
        found = world.store_replies if world is not None else None
        return found.model_dump(mode="json") if found is not None else None

    def _market_context(self) -> str | None:
        """A live UCP row's market context as the header records it (D43); None on any other row."""
        if self.row.mode != "live" or self.surface.kind == WEBSITE:
            return None
        return "fill" if self.row.market_fill else "none"

    def _served_currency(self) -> str | None:
        """The currency the store priced us in before the round, from the manifest's price probes (D43)."""
        served = self.manifest.served.get(self.row.store_id) if self.row.mode == "live" else None
        return served.currency if served is not None else None

    def _rights(self) -> Rights:
        if self.row.mode == "playground":
            return cast(Rights, self.ctx.playgrounds[self.row.store_id].rights)
        if self.row.mode == "live":
            # The queue's platform comes from page code, discovery's from the store's own replies; if either says
            # Shopify, its API terms apply, and so does the retention purge.
            seen = {self.store_ref.platform, self.discovery.platform if self.discovery else None}
            platform = "shopify" if "shopify" in seen else next((p for p in seen if p), None)
            return live_rights(platform)
        world = self.manifest.world(self.row)
        assert world is not None
        rights = self.ctx.sandbox_rights.get(rights_key(world))
        if rights is None:
            raise ValidationError(
                f"no data rights were decided for {rights_key(world)!r} before the run started",
                hint="the runner decides rights for every catalog its manifest uses; this is a bug in carttrace.collect",
            )
        return rights

    def _surface_version(self) -> str:
        """The version the header records for the row's surface: a live store's UCP release as discovery found it, a
        sandbox surface's from the row (or its profile's UCP release on a row that names no surface)."""
        if self.surface.kind == WEBSITE or self.row.mode == "playground":
            return self.surface.version or "unknown"
        if self.row.mode == "live" or self.row.surface is None:
            return self.profile.ucp or "unknown"
        return self.surface.version or "unknown"

    def _companion_refs(self) -> list[dict[str, Any]]:
        """The catalog companion's reference, after the row's own surface: the bench reads the first."""
        if self.companion is None or self.row.envpack is None:
            return []
        version = surface_version(self.companion, self.profile) or "unknown"
        url = f"sandbox://{self.row.envpack}/{self.companion}#catalog"
        return [SurfaceRef(kind=self.companion, mode="sandbox", url=url, version=version).model_dump()]

    def _payment_refs(self) -> list[dict[str, Any]]:
        """The payment rail's own surface, after the row's and its catalog's (MPP, D66)."""
        kind = self.rail.payment_surface if self.rail is not None else None
        if kind is None or self.row.envpack is None:
            return []
        version = surface_version(kind, self.profile) or "unknown"
        url = f"sandbox://{self.row.envpack}/{kind}#payment"
        return [SurfaceRef(kind=kind, mode="sandbox", url=url, version=version).model_dump()]

    def _step_cap(self) -> int:
        """collect.step_cap, and on a checkout row the steps checkout, payment and the order read back need too."""
        return self.ctx.settings.collect.step_cap + (CHECKOUT_STEPS if self.checkout else 0)

    def _context(self) -> dict[str, Any]:
        return {
            "episode_id": self.row.episode_id,
            "run_id": self.ctx.run_id,
            "manifest_id": self.manifest.id,
            "store_id": self.row.store_id,
        }

    def _app(self, pack: EnvPack, *, shared: bool) -> Any:
        """The served store the row's surfaces share, when they must share one or the row is guarded (D106), with the
        row's guard set on it; else None (each surface serves its own)."""
        guard = self._guard()
        if not shared and guard is None:
            return None
        app = self.ctx.bindings.sandbox_app(pack)
        served = getattr(getattr(app, "state", None), "host", None)
        if guard is not None and served is not None and hasattr(served, "guard"):
            served.guard = guard
            self.guard = guard
        return app

    def _guard(self) -> Guard | None:
        """The row's visibility guard (D106): on a stand-in of the open pools, every token of its roster store, its
        pool, the sandbox's own names and the source catalogs' ids, which no response may carry; None elsewhere."""
        world = self.manifest.world(self.row) if self.row.mode in ("sandbox", "reference") else None
        if world is None or not presents(world):
            return None
        roster = self.manifest.stores[self.row.store_id]
        return forbidden_tokens(
            roster_domain=roster.domain,
            roster_id=roster.store_id,
            category=world.category,
            seed=world.snapshot_seed,
        )

    def _guard_held(self) -> None:
        """End the row when its guard refused a response (D106), even one the surface turned into an error result:
        the agent never saw the token, but the row is not a fair trace of the store."""
        if self.guard is not None and self.guard.hits:
            kind, where = self.guard.hits[0]
            raise VisibilityLeak(kind, where)

    def _agent_context(self) -> str:
        """What the protocol agent is told about where it shops: the store, the shopper's market and the currency.
        Some stores refuse cart writes without a buyer country, and cross_border means nothing to an agent that is
        never told the shopper's market. A live row whose market context is `none` (D43) shops as a visitor from
        our network position: the agent is told the currency the store shows it, which its task is written in, and
        no country, so it has none to send. The text is recorded in the header's agent ModelRef params."""
        currency = self.task.hidden_spec.get("currency")
        known = isinstance(currency, str) and bool(currency)
        if self.row.mode == "live" and not self.row.market_fill and self.surface.kind != WEBSITE:
            shown = f" The store shows this shopper prices in {currency}, and the shopper pays in {currency}."
            return f"Store: {self.store_ref.domain}.{shown if known else ''}"
        pays = f" and pays in {currency}" if known else ""
        date = self._store_date()
        today = f" Today's date at the store is {date} (UTC)." if date is not None else ""
        return f"Store: {self.store_ref.domain}. The shopper is in {self.row.market}{pays}.{today}{self._journey_end()}"

    def _journey_end(self) -> str:
        """D98: on a sandbox v1 row, where the journey ends, the one thing the single prompt leaves to the context."""
        if self.row.experiment is None:
            return ""
        if self.checkout:
            return " This journey ends with an order."
        return " This journey ends at the cart: build and read back the cart the shopper chose; do not open a checkout."

    def _store_date(self) -> str | None:
        """The date a sandbox store believes it is: its world's pinned logical clock (`start.clock_start_ms`,
        2026-09-21 by default), which the agent is told the way it is told the market (D80). On 2026-10-15, 43 AP2
        v0.1.0 intents were refused as intent_expired because the agent dated them by its own guess (2026-05-01,
        2026-08-26), before the store's clock. None off the sandbox, or before the world is read."""
        pack = self._site_pack
        if pack is None or self.row.mode not in ("sandbox", "reference"):
            return None
        start = pack.doc.get("start")
        ms = start.get("clock_start_ms") if isinstance(start, Mapping) else None
        if not isinstance(ms, int) or ms <= 0:
            return None
        return datetime.fromtimestamp(ms / 1000, tz=UTC).date().isoformat()

    def _buyer_profile(self) -> BuyerProfile | None:
        """The shopper's buyer profile the agent is given (D77): only on a sandbox checkout row, from the source
        agents.buyer_profile names, for a task with a buyer block. Never on a live row (CLAUDE.md: no email or phone
        is typed into a live store) nor a playground, whose store is a copy of a real one."""
        if self.row.mode != "sandbox" or not self.checkout:
            return None
        return buyer_profile(self.task, source=self.ctx.settings.agents.buyer_profile)

    def _model(self, ref: ModelRef) -> ModelRef:
        if ref.role != "agent" or self.agent_mode == "reference":
            return ref
        params = {**ref.params, "context": self._agent_context(), "seed": agent_seed(self.row)}
        profile = self._buyer_profile()
        if profile is not None:
            # what the agent was given (D69): the source, the persona and the details, the text redacted at write time
            params["buyer_profile"] = {**profile.record(), "text": profile.text()}
        return ref.model_copy(update={"params": params})

    @property
    def agent_mode(self) -> RunMode:
        """The mode whose agent shops this row: the row's, except a live row the run gives the reference agent."""
        if self.row.mode == "live" and self.ctx.live_agent == "reference":
            return "reference"
        return self.row.mode

    def _agent_profile(self) -> str:
        """Our agent profile for sandbox calls: the published one, or a placeholder that can never resolve (RFC 2606),
        since the sandbox requires one and never fetches it."""
        return self.ctx.settings.live.agent_profile_url or PLACEHOLDER_AGENT_PROFILE


def agent_seed(row: ManifestRow) -> int:
    """The seed the agent's per-call sampling seeds are drawn from (carttrace.agents.qwen turn_seed): one drawn from the
    pair id on either half of a live and replica pair, so both halves send the same seed at the same call and differ
    only by what the store answered (2026-10-13: 151 of 273 disagreeing pairs began with a different first query
    before any store answer differed); the row's own seed on any other row."""
    if row.pair_id is not None:
        return derived_seed("agent", row.pair_id)
    return row.seed


def discovery_end(discovery: Discovery) -> str | None:
    """The end reason discovery decides on its own, or None when the store can be shopped: robots.txt disallowing us,
    a 401 or 403 (a wall), or a 429 end the episode as the store's; a store that declares no UCP over MCP has no
    protocol for us; one that never answered at all is an infrastructure failure, ours to explain."""
    if discovery.robots_allowed is False:
        return "blocked"
    if discovery.ucp_endpoint:
        return None
    statuses = [status for status in discovery.statuses.values() if isinstance(status, int)]
    codes = {str(code) for code in (getattr(discovery, "errors", None) or {}).values()}
    if "blocked" in codes or any(status in (401, 403) for status in statuses):
        return "blocked"
    if "throttled" in codes or 429 in statuses:
        return "throttled"
    if discovery.robots_allowed is None and not statuses:
        return "harness_error"
    return "no_protocol"


def website_end(discovery: Discovery) -> str | None:
    """The end reason discovery decides for a website row: a wall (401, 403, a challenge) or a 429 ends it as the
    store's; one that never answered at all is ours. A store with no UCP document is what the row is for, and
    robots.txt is asked again for every page the agent reads."""
    statuses = [status for status in discovery.statuses.values() if isinstance(status, int)]
    codes = {str(code) for code in (getattr(discovery, "errors", None) or {}).values()}
    if "blocked" in codes or any(status in (401, 403) for status in statuses):
        return "blocked"
    if "throttled" in codes or 429 in statuses:
        return "throttled"
    if discovery.robots_allowed is None and not statuses:
        return "harness_error"
    return None


def code_of(exc: BaseException) -> str | None:
    """The stable error code carttrace.live puts on its refusals and store errors."""
    code = getattr(exc, "code", None)
    return code if isinstance(code, str) else None


def _order_number(doc: Mapping[str, Any]) -> str | None:
    """The order number a checkout document shows beside its order id (the playground's UCP plugin: `order.label`),
    which the store's admin API reads the order by."""
    order = doc.get("order")
    label = order.get("label") if isinstance(order, Mapping) else None
    return label if isinstance(label, str) and label.isdigit() else None


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _clip(text: str) -> str:
    return text if len(text) <= ERROR_CHARS else text[: ERROR_CHARS - 1] + "…"

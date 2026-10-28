"""What the collector drives in other packages, as Protocols: the runner depends on shapes, never on a build.

Why ports. The agents, the live client, the sandbox surfaces and the catalog sources are built beside the collector,
on lower layers (docs/ARCHITECTURE.md). The runner names here exactly what it calls and what it reads back, by the
contracted names, and `carttrace.collect.bindings` connects them to the real packages when a run starts. Tests hand the
runner fakes with the same shapes, so every path of the runner runs offline.

Results are read by attribute and checked where they are used, because a value that crosses a package boundary is
data: an agent that returns an end reason the trace schema does not know ends its episode as `harness_error` with a
label saying so, instead of failing the run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from carttrace.config.schema import Settings
from carttrace.core.clock import Clock
from carttrace.decide.service import DecisionEngine
from carttrace.protocols.profile import ProtocolProfile
from carttrace.protocols.tools import ToolSurface
from carttrace.sandbox.envpack import EnvPack
from carttrace.stores.categories import Category
from carttrace.stores.model import Store
from carttrace.trace.schema import Rights, TaskRef
from carttrace.trace.writer import TraceWriter

RunMode = Literal["live", "sandbox", "reference", "playground"]
"""live: the protocol agent on a live store. sandbox: the protocol agent on a sandbox store. reference: the scripted
reference agent on the same sandbox store, which separates what a store cannot do from what our agent did not do.
playground: the protocol agent on a real store we own and run on this machine (carttrace.collect.playgrounds, D44), never a
live one."""

RUN_MODES: tuple[RunMode, ...] = ("live", "sandbox", "reference", "playground")


class WireSink(Protocol):
    """Where a surface records its exchanges. A TraceWriter is one; so are the collector's taps and buffers."""

    def wire(self, record: Mapping[str, Any]) -> Any: ...


class Shopper(Protocol):
    """carttrace.agents' shopper: the task's opening, then one reply per agent message, answered from the hidden spec."""

    def opening(self) -> str: ...

    async def reply(self, agent_message: str, *, cut: bool = False) -> Any: ...


class Supervisor(Protocol):
    """What the agent asks between steps. The verdict has `stop`, `reason` and `decision_id`."""

    async def check(self, steps: Sequence[Any], cart: Any) -> Any: ...


class AgentResult(Protocol):
    """carttrace.agents' AgentResult, read by attribute."""

    @property
    def end_reason(self) -> str: ...

    @property
    def final_message(self) -> str | None: ...

    @property
    def cart_id(self) -> str | None: ...

    @property
    def checkout_id(self) -> str | None:
        """The checkout the store returned last, on a checkout row."""
        ...

    @property
    def order_id(self) -> str | None:
        """The order the store placed, on a checkout row."""
        ...


class ProtocolAgent(Protocol):
    """carttrace.agents' ProtocolAgent: QwenProtocolAgent for live and sandbox rows, ReferenceAgent for reference rows."""

    async def run(
        self,
        task: TaskRef,
        surfaces: Sequence[ToolSurface],
        shopper: Shopper,
        sink: TraceWriter,
        *,
        limits: Any,
        supervisor: Supervisor | None = None,
    ) -> AgentResult: ...


class LiveGate(Protocol):
    """carttrace.live's LiveGate. `check` raises SafetyError while any gate is closed or the store is paused."""

    def check(self, domain: str, *, store_id: str | None = None) -> None: ...


class Discovery(Protocol):
    """carttrace.live's Discovery, read by attribute."""

    @property
    def domain(self) -> str: ...

    @property
    def robots_allowed(self) -> bool | None:
        """None when robots.txt never answered."""
        ...

    @property
    def ucp_endpoint(self) -> str | None: ...

    @property
    def ucp_profile(self) -> Any: ...

    @property
    def profile(self) -> ProtocolProfile | None: ...

    @property
    def platform(self) -> str | None: ...

    @property
    def statuses(self) -> Mapping[str, int]: ...

    @property
    def notes(self) -> Sequence[str]: ...


class MakeLiveGate(Protocol):
    def __call__(self, settings: Settings, *, live_flag: bool) -> LiveGate: ...


class Discover(Protocol):
    """carttrace.live.discover. `store_id` lets a per-store pause apply, `platform` names the throttle bucket."""

    async def __call__(
        self,
        domain: str,
        *,
        gate: LiveGate,
        recorder: WireSink | None = None,
        store_id: str | None = None,
        platform: str | None = None,
    ) -> Discovery: ...


class OpenUcpSurface(Protocol):
    """carttrace.live.open_ucp_surface. Given this run's `discovery`, the session goes where discovery found the store; given
    the row's `market`, cart and catalog calls that leave the buyer's country or currency out get the market's."""

    async def __call__(
        self,
        domain: str,
        *,
        gate: LiveGate,
        recorder: WireSink | None = None,
        store_id: str | None = None,
        platform: str | None = None,
        discovery: Discovery | None = None,
        market: str | None = None,
    ) -> ToolSurface: ...


class OpenWebsiteSurface(Protocol):
    """carttrace.live.open_website_surface: a store without an agent protocol, read through its own pages (D31), GET only;
    `currency` is the shopper's, for a page that names none; `hints` are (url, title) of the products our capture
    `hints_edition` holds, browse_index entries marked "capture" (D34); `market` is the shopper's, whose locale the
    store's product index prefers (D38)."""

    async def __call__(
        self,
        domain: str,
        *,
        gate: LiveGate,
        recorder: WireSink | None = None,
        store_id: str | None = None,
        platform: str | None = None,
        discovery: Discovery | None = None,
        currency: str | None = None,
        hints: Sequence[tuple[str, str]] = (),
        hints_edition: str | None = None,
        market: str | None = None,
    ) -> ToolSurface: ...


class OpenPlaygroundSurface(Protocol):
    """A playground's surface of one kind (D44): its storefront pages through the website road under its reserved
    name (`website`), or its UCP REST plugin on loopback (`ucp_rest`); `currency` and `market` are the shopper's."""

    async def __call__(
        self,
        playground: Any,
        *,
        kind: str,
        settings: Settings,
        recorder: WireSink | None = None,
        session_id: str | None = None,
        agent_profile: str | None = None,
        currency: str | None = None,
        market: str | None = None,
    ) -> ToolSurface: ...


class PlaygroundCatalogs(Protocol):
    """A playground's own ids for naming a website selection (carttrace.collect.playgrounds.PlaygroundCatalog), read from its
    Store API."""

    async def __call__(self, playground: Any) -> Any: ...


class ReadPlaygroundOrder(Protocol):
    """An order as the playground's admin API holds it (carttrace.collect.playgrounds.read_order)."""

    async def __call__(self, playground: Any, home: Path, order: str) -> Any: ...


class OpenWallet(Protocol):
    """The persona's wallet for a playground checkout row (carttrace.agents' wallet_for): no sandbox session to grant, and
    no card is ever needed, since the playground's gateways are offline."""

    def __call__(self, persona: str | None, *, market: str) -> Any: ...


class CaptureCatalogs(Protocol):
    """carttrace.live.load_capture_catalog over the run's blob store: the store's captured ids, or None (D31)."""

    def __call__(self, store_id: str) -> Any: ...


class OpenSandboxSurface(Protocol):
    """carttrace.sandbox.open_sandbox_surface. `options` are the kind's own (webmcp: the persona's 3-D Secure answer, the
    storefront's detail filter, the browser limits, the seed); `app` shares one served store between two surfaces of
    an episode (an ACP row's checkout and the catalog it reads beside it), as `sandbox_app` builds it."""

    async def __call__(
        self,
        envpack: EnvPack,
        *,
        kind: str = "ucp_mcp",
        recorder: WireSink | None = None,
        session_id: str | None = None,
        agent_profile: str | None = None,
        options: Mapping[str, Any] | None = None,
        app: Any = None,
    ) -> ToolSurface: ...


EventSink = Callable[[Mapping[str, Any]], object]
"""Where the sandbox hands each record of its event log: the episode's TraceWriter.event, which writes events.jsonl."""


class SandboxWatch(Protocol):
    """The running sandbox behind a surface (carttrace.sandbox's Sandbox), as an episode reads it when its outcome is
    recorded: the final state's hash and the head of the event chain."""

    @property
    def pack(self) -> EnvPack: ...

    @property
    def digest(self) -> str: ...

    @property
    def state(self) -> Any:
        """The sandbox's current core state (payments, checkouts, orders), read by attribute: the PSP's own record of
        what it decided is built from it (D66)."""
        ...

    def state_hash(self) -> str: ...

    def detach(self, sink: EventSink) -> None: ...


class WatchSandbox(Protocol):
    """Attach `sink` to the sandbox behind `surface` (every record so far, then each new one, D33) and return the
    sandbox; None when the surface is not the sandbox's own (a test's fake store), which then writes no events."""

    def __call__(self, surface: ToolSurface, sink: EventSink) -> SandboxWatch | None: ...


class BuildSandboxApp(Protocol):
    """carttrace.sandbox.build_app: one served store for an envpack, which `OpenSandboxSurface` opens surfaces on."""

    def __call__(self, envpack: EnvPack) -> Any: ...


class GenerateTasks(Protocol):
    """carttrace.agents.generate_tasks: tasks written around `fixtures`, product dicts as UCP get_product returns them.
    `persona` is the one the plan's checkout rows pay as, so the spec's buyer block holds that persona's details."""

    def __call__(
        self,
        store: Store,
        category: Category,
        fixtures: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        persona: str | None = None,
    ) -> list[TaskRef]: ...


class BuildStoreSnapshot(Protocol):
    """carttrace.sandbox.catalogs.build_store_snapshot: one catalog store of a category, as `build_envpack` arguments
    (`envpack_kwargs()`), or an envpack `Snapshot`. `match` names the capture of the store an open-pool stand-in's
    products are matched to (D104); it is passed only when set, so a binding without it serves every other world."""

    def __call__(
        self,
        source_name: str,
        category_slug: str,
        *,
        seed: int,
        size: int,
        market: str | None = None,
        match: str | None = None,
    ) -> Any: ...


class PoolMarket(Protocol):
    """carttrace.sandbox.catalogs.pool_market: the market a stand-in store of the open pools is built in (D91)."""

    def __call__(self, category: str, store_id: str, *, seed: int, home: str) -> str: ...


class ReplicaSnapshot(Protocol):
    """carttrace.sandbox.catalogs.build_replica: a captured store's replica as a store snapshot (`envpack_kwargs()`, with
    the store's captured profile, payment handlers and recorded answers), selling in `market`."""

    def __call__(self, capture: Path, *, market: str) -> Any: ...


class Fixtures(Protocol):
    """Product dicts in UCP's shape from an envpack snapshot (carttrace.agents fixtures_from_snapshot): what
    generate_tasks writes sandbox tasks around."""

    def __call__(self, snapshot: Any) -> list[dict[str, Any]]: ...


class LicenceRights(Protocol):
    """carttrace.sandbox.catalogs.licence_rights: the Rights fields a source's licence allows."""

    def __call__(self, source_name: str) -> Rights | Mapping[str, Any]: ...


class CatalogRestricted(Protocol):
    """Whether a catalog source's licence is restricted (Lazada, decision D11: internal research only)."""

    def __call__(self, source_name: str) -> bool: ...


class MakeAgent(Protocol):
    """The agent for one episode: its sampling seeded by the row's seed, its time cap on the run's clock, and the
    `context` it is told (the store, the shopper's market and currency), which agents that take none ignore. A
    checkout row passes the persona's `wallet` too, which makes the agent go on from the cart to an order, and a
    sandbox checkout row the shopper's buyer `profile` (carttrace.agents.profile, D77), only when it has one."""

    def __call__(
        self, mode: RunMode, *, seed: int, clock: Clock, context: str = "", wallet: Any = None, profile: Any = None
    ) -> ProtocolAgent: ...


class PersonaBook(Protocol):
    """carttrace.agents' Personas: the synthetic shoppers checkout rows pay as."""

    @property
    def default(self) -> str: ...

    def markets(self) -> Mapping[str, Sequence[str]]:
        """Persona name to the delivery markets it has an address in."""
        ...

    def get(self, name: str | None = None) -> Any:
        """One persona (the default when None), read by attribute: `three_ds` is how it answers the bank's
        challenge (complete or abandon), which a WebMCP storefront's page needs to know before the agent starts."""
        ...


class LoadPersonas(Protocol):
    """carttrace.agents.load_personas: data/agent/personas.yaml, checked."""

    def __call__(self) -> PersonaBook: ...


class OpenCheckout(Protocol):
    """Prepare one checkout row's sandbox session and return the persona's wallet (carttrace.agents' Wallet) for the agent.

    The session gets the buyer token with the purchase permission, Shopify's token tier, as the environment grants it
    (logged in the sandbox's own event log), and the wallet reaches the bank's 3-D Secure page and the store's Shop
    Pay interface (D73) through the same in-process app as the surface, recording those exchanges with `recorder`.
    `prefer` is the handler namespace the row's payment rail asks the wallet to recommend first."""

    async def __call__(
        self,
        surface: ToolSurface,
        *,
        persona: str | None,
        market: str,
        session_id: str,
        recorder: WireSink | None,
        clock: Clock,
        prefer: str | None = None,
    ) -> Any: ...


class MakeLimits(Protocol):
    """carttrace.agents.Limits(step_cap, time_cap_s, max_tokens)."""

    def __call__(self, *, step_cap: int, time_cap_s: int, max_tokens: int) -> Any: ...


class MakeShopper(Protocol):
    """carttrace.agents' deterministic shopper for one task, asking its field mapping through `engine`. On a live row
    (`live`) the shopper never gives an email or phone number, because none may be typed into a live store."""

    def __call__(self, task: TaskRef, engine: DecisionEngine, *, live: bool = False) -> Shopper: ...

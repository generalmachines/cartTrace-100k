"""The collector's bindings: each port in `ports.py` connected to its implementation, by the contracted name.

Why one table resolved late. The runner needs twelve entry points from carttrace.agents, carttrace.live and carttrace.sandbox. Naming them
in one frozen record means a test swaps any of them for a fake without patching modules, and a sandbox-only run never
imports carttrace.live. `default_bindings` resolves each name on first call, so a missing or renamed entry point fails with
a ConfigError that names it, at the moment it is needed, instead of as an import error far from the cause.

The agents' constructors are not fixed by the contracts, so each is built with the keywords its signature accepts from
what the runner offers (the settings, the row's seed, the run's clock, a checkout row's wallet), and nothing is
guessed: QwenProtocolAgent samples with the row's seed, and both agents keep their time cap on the run's clock. The
shopper comes from carttrace.agents' `shopper_for(task, engine=...)`, given the episode's engine so its field mapping
decisions land in the bundle.

Checkout rows (D13) add two: the personas (carttrace.agents' `load_personas`, data/agent/personas.yaml), which a manifest
checks its checkout rows against, and `checkout`, which grants the row's sandbox session the buyer token with the
purchase permission and builds the persona's wallet (`wallet_for`), its 3-D Secure page reached through the sandbox
surface's own in-process client (`HttpChallengePage`). Both default to the real implementations, so a test that
builds its own bindings gets them unless it passes fakes.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from carttrace.collect.ports import (
    BuildSandboxApp,
    BuildStoreSnapshot,
    CaptureCatalogs,
    CatalogRestricted,
    Discover,
    Discovery,
    EventSink,
    Fixtures,
    GenerateTasks,
    LicenceRights,
    LiveGate,
    LoadPersonas,
    MakeAgent,
    MakeLimits,
    MakeLiveGate,
    MakeShopper,
    OpenCheckout,
    OpenPlaygroundSurface,
    OpenSandboxSurface,
    OpenUcpSurface,
    OpenWallet,
    OpenWebsiteSurface,
    PersonaBook,
    PlaygroundCatalogs,
    PoolMarket,
    ProtocolAgent,
    ReadPlaygroundOrder,
    ReplicaSnapshot,
    RunMode,
    SandboxWatch,
    Shopper,
    WatchSandbox,
    WireSink,
)
from carttrace.config.schema import Settings
from carttrace.core.clock import Clock
from carttrace.core.errors import ConfigError, ValidationError
from carttrace.decide.service import DecisionEngine
from carttrace.protocols.tools import ToolSurface
from carttrace.sandbox.envpack import EnvPack
from carttrace.stores.categories import Category
from carttrace.stores.model import Store
from carttrace.trace.schema import Rights, TaskRef

AGENTS = "carttrace.agents"
LIVE = "carttrace.live"
WEBSITE_KIND = "website"
"""A sandbox row on this kind reads its replica's pages through carttrace.live's website tools (carttrace.collect.replica_site)."""
SANDBOX = "carttrace.sandbox"
CATALOGS = "carttrace.sandbox.catalogs"
FIXTURES = "carttrace.agents.products"
OPEN_POOL = "open"
"""carttrace.sandbox.catalogs.OPEN: the catalog name of the open product pools (D91)."""
DEFAULT_POOL_MARKET = "US"
AGENT_CLASSES: Mapping[RunMode, str] = {
    "live": "QwenProtocolAgent",
    "sandbox": "QwenProtocolAgent",
    "reference": "ReferenceAgent",
    "playground": "QwenProtocolAgent",
}
SHOPPER_FACTORY = "shopper_for"
"""carttrace.agents' factory for the runner's DeterministicShopper, asking shopper.field through the engine it is given."""
PERSONAS_FILE = Path("agent") / "personas.yaml"
"""Under settings.paths.data."""
PLAYGROUND_RPS: Mapping[str, float] = {"default": 10.0}
"""Requests per second to a playground: it is ours, on this machine, and only its own capacity bounds it."""
PLAYGROUND_SESSIONS = 100_000
"""Sessions per day on a playground: no cap worth the name."""
PURCHASE = "purchase"
"""The permission a token-tier sandbox session needs to complete a checkout (carttrace.sandbox.core.trust)."""


def personas_at(path: Path | None) -> LoadPersonas:
    """The personas at `path` (carttrace.agents' own default when None)."""

    def personas() -> PersonaBook:
        return cast(PersonaBook, contract(AGENTS, "load_personas")(path))

    return personas


def checkout_with(path: Path | None) -> OpenCheckout:
    """A checkout binding paying as the personas at `path` (carttrace.agents' own default when None)."""

    async def checkout(
        surface: ToolSurface,
        *,
        persona: str | None,
        market: str,
        session_id: str,
        recorder: WireSink | None,
        clock: Clock,
        prefer: str | None = None,
    ) -> Any:
        sandbox_surface = contract(SANDBOX, "SandboxSurface")
        if not isinstance(surface, sandbox_surface):
            raise ConfigError(
                f"a checkout row needs the sandbox's own surface, not a {type(surface).__name__}",
                hint="checkout rows run in the sandbox only (D13); open the surface with carttrace.sandbox",
            )
        granted = surface.host.open_session(session_id, "token", [PURCHASE])
        if not granted.ok:
            raise ValidationError(
                f"the sandbox would not grant session {session_id} the buyer token: {granted.error}",
                hint="check the envpack's start sessions and the sandbox's trust policy",
            )
        page = contract(AGENTS, "HttpChallengePage")(surface.client, recorder=recorder, clock=clock)
        shop_pay = contract(AGENTS, "HttpShopPay")(
            surface.client, session_id, recorder=recorder, clock=clock, session_header=surface.session_header
        )
        return contract(AGENTS, "wallet_for")(
            persona,
            market=market,
            challenge_page=page,
            shop_pay=shop_pay,
            prefer=prefer,
            path=path,
            alias_key=session_id,
        )

    return checkout


def wallet_with(path: Path | None) -> OpenWallet:
    """The wallet of a persona at `path` (carttrace.agents' own default when None), for a playground checkout row."""

    def wallet(persona: str | None, *, market: str) -> Any:
        return contract(AGENTS, "wallet_for")(persona, market=market, path=path)

    return wallet


async def open_playground_surface(
    playground: Any,
    *,
    kind: str,
    settings: Settings,
    recorder: WireSink | None = None,
    session_id: str | None = None,
    agent_profile: str | None = None,
    currency: str | None = None,
    market: str | None = None,
) -> ToolSurface:
    """A playground's surface (D44). `website`: carttrace.live's website surface on its reserved host, behind a gate that
    opens for that host only, through the loopback bridge, with a throttle of its own (it is ours, so no daily cap
    worth the name). `ucp_rest`: carttrace.sandbox's remote UCP REST surface at the endpoint its discovery declares."""
    if kind == "website":
        from carttrace.core.clock import SystemClock

        clock = SystemClock()
        throttle = contract(f"{LIVE}.throttle", "Throttle")(
            settings.paths.var / "locks" / "playground", PLAYGROUND_RPS, daily_cap=PLAYGROUND_SESSIONS, clock=clock
        )
        robots = contract(f"{LIVE}.robots", "RobotsCache").from_settings(settings, clock=clock)
        surface = await contract(LIVE, "open_website_surface")(
            playground.host,
            gate=contract(LIVE, "PlaygroundGate")(settings, hosts=[playground.host]),
            recorder=recorder,
            store_id=playground.store_id,
            platform=playground.platform,
            currency=currency,
            market=market,
            clock=clock,
            transport=contract(LIVE, "LoopbackBridge")(playground.host, playground.origin),
            throttle=throttle,
            robots=robots,
        )
        return cast(ToolSurface, surface)
    if kind == "ucp_rest":
        surface = await contract(SANDBOX, "open_remote_rest_surface")(
            playground.origin,
            discovery_path=playground.ucp_discovery,
            recorder=recorder,
            session_id=session_id,
            agent_profile=agent_profile,
            note=f"declared by playground {playground.store_id}'s discovery document",
        )
        return cast(ToolSurface, surface)
    raise ConfigError(f"a playground has no {kind} surface", hint="playground rows run on website or ucp_rest")


async def playground_catalog(playground: Any) -> Any:
    """The playground's own ids, read from its Store API (carttrace.collect.playgrounds)."""
    from carttrace.collect.playgrounds import PlaygroundCatalog, playground_fixtures

    return PlaygroundCatalog(await asyncio.to_thread(playground_fixtures, playground))


async def read_playground_order(playground: Any, home: Path, order: str) -> Any:
    """An order as the playground's admin API holds it (carttrace.collect.playgrounds.read_order)."""
    from carttrace.collect.playgrounds import read_order

    return await read_order(playground, home, order)


def watch_sandbox(surface: ToolSurface, sink: EventSink) -> SandboxWatch | None:
    """The sandbox behind a surface carttrace.sandbox opened, with `sink` attached to its event log (D33); None for any
    other surface. A replica's website (carttrace.collect.replica_site) names its sandbox as `sandbox_host`."""
    host: Any = getattr(surface, "sandbox_host", None)
    if host is None:
        if not isinstance(surface, contract(SANDBOX, "SandboxSurface")):
            return None
        host = getattr(surface, "host")  # noqa: B009 - a SandboxSurface's host, which the ToolSurface contract lacks
    sandbox = host.sandbox
    sandbox.attach(sink)
    return cast(SandboxWatch, sandbox)


@dataclass(frozen=True, slots=True)
class Bindings:
    """Every entry point of another package the collector calls. Build the real one with `default_bindings`."""

    agent: MakeAgent
    limits: MakeLimits
    shopper: MakeShopper
    generate_tasks: GenerateTasks
    live_gate: MakeLiveGate
    discover: Discover
    open_ucp_surface: OpenUcpSurface
    open_sandbox_surface: OpenSandboxSurface
    store_snapshot: BuildStoreSnapshot
    fixtures: Fixtures
    licence_rights: LicenceRights
    catalog_restricted: CatalogRestricted
    pool_market: PoolMarket | None = None
    """carttrace.sandbox.catalogs.pool_market (D91); None plans every stand-in store of the open pools in its own market."""
    personas: LoadPersonas = field(default=personas_at(None))
    checkout: OpenCheckout = field(default=checkout_with(None))
    replica_snapshot: ReplicaSnapshot = field(default_factory=lambda: replica_snapshot)
    """carttrace.sandbox.catalogs.build_replica, for worlds built from store captures."""
    sandbox_app: BuildSandboxApp = field(default_factory=lambda: sandbox_app)
    """carttrace.sandbox.build_app, for an episode whose two surfaces (an ACP checkout and the catalog beside it) must be one
    store."""
    open_website_surface: OpenWebsiteSurface = field(default_factory=lambda: open_website_surface)
    """carttrace.live.open_website_surface, for a live row on a store's own pages (D31)."""
    open_webmcp_surface: Any = field(default_factory=lambda: open_webmcp_surface)
    """carttrace.live.open_webmcp_surface, for a live row on the WebMCP tools a store's page registers (D67)."""
    open_store_mcp_surface: Any = field(default_factory=lambda: open_store_mcp_surface)
    """carttrace.live.open_store_mcp_surface, the Storefront MCP policy tool beside a live UCP row (live.policy_companion)."""
    capture_catalog: CaptureCatalogs = field(default_factory=lambda: no_capture_catalog)
    """The store's captured ids a website selection is named in; none unless the run binds the blob store's."""
    watch_sandbox: WatchSandbox = field(default_factory=lambda: watch_sandbox)
    """Attaches a sandbox row's event sink to the sandbox behind its surface, so the bundle carries events.jsonl."""
    open_playground_surface: OpenPlaygroundSurface = field(default_factory=lambda: open_playground_surface)
    """A playground's website (carttrace.live, under its reserved name) or UCP REST surface (carttrace.sandbox, on loopback), D44."""
    playground_catalog: PlaygroundCatalogs = field(default_factory=lambda: playground_catalog)
    """A playground's own ids, which its website selections are named in."""
    read_playground_order: ReadPlaygroundOrder = field(default_factory=lambda: read_playground_order)
    """An order as the playground's admin API holds it, read after a checkout row."""
    playground_wallet: OpenWallet = field(default=wallet_with(None))
    """The persona's wallet for a playground checkout row."""
    shopper_voice: Callable[[Settings, str, str | None], Any] = field(default_factory=lambda: shopper_voice)
    """D118: the row's shopper voice (carttrace.agents.wording.ShopperVoice) when `agents.shopper.wording` is model, else
    None: the rules decide every reply, the shopper role's model words it."""


async def open_website_surface(
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
) -> ToolSurface:
    """carttrace.live.open_website_surface, resolved when first called."""
    fn = contract(LIVE, "open_website_surface")
    extra = _accepted_by(
        fn,
        store_id=store_id,
        platform=platform,
        discovery=discovery,
        currency=currency,
        hints=tuple(hints) or None,
        hints_edition=hints_edition,
        market=market,
    )
    return cast(ToolSurface, await fn(domain, gate=gate, recorder=recorder, **extra))


async def open_webmcp_surface(
    domain: str,
    *,
    gate: LiveGate,
    recorder: WireSink | None = None,
    store_id: str | None = None,
    platform: str | None = None,
    discovery: Discovery | None = None,
) -> ToolSurface:
    """carttrace.live.open_webmcp_surface, resolved when first called."""
    fn = contract(LIVE, "open_webmcp_surface")
    extra = _accepted_by(fn, store_id=store_id, platform=platform, discovery=discovery)
    return cast(ToolSurface, await fn(domain, gate=gate, recorder=recorder, **extra))


async def open_store_mcp_surface(
    domain: str,
    *,
    gate: LiveGate,
    recorder: WireSink | None = None,
    store_id: str | None = None,
    platform: str | None = None,
    discovery: Discovery | None = None,
) -> ToolSurface:
    """carttrace.live.open_store_mcp_surface, resolved when first called."""
    fn = contract(LIVE, "open_store_mcp_surface")
    return cast(
        ToolSurface,
        await fn(domain, gate=gate, recorder=recorder, store_id=store_id, platform=platform, discovery=discovery),
    )


def no_capture_catalog(store_id: str) -> Any:
    """No captured ids: a website selection keeps synthetic ids."""
    del store_id
    return None


def capture_catalogs(settings: Settings) -> CaptureCatalogs:
    """The stores' captured ids from the configured blob store (carttrace.live.load_capture_catalog), each store read once."""
    cache: dict[str, Any] = {}

    def load(store_id: str) -> Any:
        if store_id not in cache:
            from carttrace.storage.factory import store as blob_store

            cache[store_id] = contract(LIVE, "load_capture_catalog")(blob_store(settings), store_id)
        return cache[store_id]

    return load


def replica_snapshot(capture: Path, *, market: str) -> Any:
    """A captured store's replica snapshot, resolved when first called: carttrace.sandbox.catalogs.build_replica for an old
    capture file, build_replica_v2 for a capture v2 store directory (carttrace.sandbox.catalogs.capture_v2)."""
    return contract(f"{CATALOGS}.capture_v2", "build_any_replica")(capture, market=market)


def sandbox_app(envpack: EnvPack) -> Any:
    """One served sandbox store for `envpack` (carttrace.sandbox.build_app), resolved when first called."""
    return contract(SANDBOX, "build_app")(envpack)


def shopper_voice(settings: Settings, task_id: str, persona: str | None) -> Any:
    """carttrace.agents.wording's ShopperVoice for one row on the process's QwenWorder (D118), or None when the shopper
    words nothing by model (`agents.shopper.wording` rules)."""
    config = settings.agents.shopper
    if config.wording != "model":
        return None
    wording = f"{AGENTS}.wording"
    worder = contract(wording, "worder_for")(settings)
    return contract(wording, "ShopperVoice").from_settings(worder, config, task_id=task_id, persona=persona)


def contract(module: str, name: str) -> Any:
    """`module.name`, imported now. Raises ConfigError naming the contract when either is missing."""
    try:
        found = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module and not module.startswith(f"{exc.name}."):
            raise
        raise ConfigError(
            f"{module} is not available, so the collector cannot call {module}.{name}",
            hint=f"build {module} to its contract; tests run the collector with fake bindings",
        ) from exc
    target = getattr(found, name, None)
    if target is None:
        raise ConfigError(
            f"{module} defines no {name}, which the collector calls by its contracted name",
            hint=f"export {name} from {module}, or update carttrace.collect.bindings if the contract changed",
        )
    return target


def construct(factory: Callable[..., Any], **offered: Any) -> Any:
    """Call `factory` with the offered keyword arguments its signature accepts (all of them if it takes **kwargs)."""
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return factory()
    return factory(**_accepted(parameters, offered))


def default_bindings(settings: Settings) -> Bindings:
    """The real implementations, each resolved on first use."""

    def agent(
        mode: RunMode,
        *,
        seed: int,
        clock: Clock,
        context: str = "",
        wallet: Any = None,
        profile: Any = None,
        single_prompt: bool = False,
        recover: bool | None = None,
    ) -> ProtocolAgent:
        factory = contract(AGENTS, AGENT_CLASSES[mode])
        offered: dict[str, Any] = {"settings": settings, "seed": seed, "clock": clock, "context": context}
        if single_prompt and "single_prompt" in _accepted(_parameters(factory), {"single_prompt": True}):
            # D98: a sandbox v1 cart row reads the one v1 prompt; an agent without prompts (the reference) ignores it
            offered["single_prompt"] = True
        if profile is not None and "profile" in _accepted(_parameters(factory), {"profile": profile}):
            # the buyer profile of a sandbox checkout row (D77); an agent that takes none (the reference agent, which
            # reads the hidden spec) is built without it
            offered["profile"] = profile
        if recover is not None and "recover" in _accepted(_parameters(factory), {"recover": recover}):
            offered["recover"] = recover  # D115: the reference follows the row's designed expectation
        if wallet is not None:
            if "wallet" not in _accepted(_parameters(factory), {"wallet": wallet}):
                raise ConfigError(
                    f"carttrace.agents.{AGENT_CLASSES[mode]} takes no wallet, so it cannot run a checkout row",
                    hint="give the agent a `wallet` keyword (carttrace.agents.wallet.Wallet)",
                )
            offered["wallet"] = wallet
        return cast(ProtocolAgent, construct(factory, **offered))

    def limits(*, step_cap: int, time_cap_s: int, max_tokens: int) -> Any:
        return contract(AGENTS, "Limits")(step_cap=step_cap, time_cap_s=time_cap_s, max_tokens=max_tokens)

    def shopper(task: TaskRef, engine: DecisionEngine, *, live: bool = False) -> Shopper:
        # A live row's shopper withholds email and phone: CLAUDE.md, no contact detail is ever typed into a live store.
        return cast(Shopper, contract(AGENTS, SHOPPER_FACTORY)(task, engine=engine, disclose_contact=not live))

    def generate_tasks(
        store: Store, category: Category, fixtures: Any, *, seed: int, persona: str | None = None
    ) -> list[TaskRef]:
        tasks = contract(AGENTS, "generate_tasks")(store, category, fixtures, seed=seed, persona=persona)
        return [t if isinstance(t, TaskRef) else TaskRef.model_validate(t) for t in tasks]

    def live_gate(settings: Settings, *, live_flag: bool) -> LiveGate:
        return cast(LiveGate, contract(LIVE, "LiveGate")(settings, live_flag=live_flag))

    async def discover(
        domain: str,
        *,
        gate: LiveGate,
        recorder: WireSink | None = None,
        store_id: str | None = None,
        platform: str | None = None,
    ) -> Discovery:
        fn = contract(LIVE, "discover")
        extra = _accepted_by(fn, store_id=store_id, platform=platform)
        return cast(Discovery, await fn(domain, gate=gate, recorder=recorder, **extra))

    async def open_ucp_surface(
        domain: str,
        *,
        gate: LiveGate,
        recorder: WireSink | None = None,
        store_id: str | None = None,
        platform: str | None = None,
        discovery: Discovery | None = None,
        market: str | None = None,
    ) -> ToolSurface:
        fn = contract(LIVE, "open_ucp_surface")
        extra = _accepted_by(fn, store_id=store_id, platform=platform, discovery=discovery, market=market)
        return cast(ToolSurface, await fn(domain, gate=gate, recorder=recorder, **extra))

    async def open_sandbox_surface(
        envpack: EnvPack,
        *,
        kind: str = "ucp_mcp",
        recorder: WireSink | None = None,
        session_id: str | None = None,
        agent_profile: str | None = None,
        options: Mapping[str, Any] | None = None,
        app: Any = None,
        guard: Any = None,
    ) -> ToolSurface:
        if kind == WEBSITE_KIND:
            from carttrace.collect.replica_site import open_replica_website

            given = dict(options or {})
            return cast(
                ToolSurface,
                await open_replica_website(
                    envpack,
                    settings=settings,
                    recorder=recorder,
                    currency=given.get("currency"),
                    market=given.get("market"),
                ),
            )
        fn = contract(SANDBOX, "open_sandbox_surface")
        extra = _accepted_by(fn, options=options, app=app, guard=guard)
        opened = await fn(
            envpack, kind=kind, recorder=recorder, session_id=session_id, agent_profile=agent_profile, **extra
        )
        return cast(ToolSurface, opened)

    def store_snapshot(
        source_name: str,
        category_slug: str,
        *,
        seed: int,
        size: int,
        market: str | None = None,
        match: str | None = None,
    ) -> Any:
        if source_name == OPEN_POOL:
            build = contract(CATALOGS, "build_pool_snapshot")
            extra = {"match_capture": match} if match is not None else {}
            return build(category_slug, seed=seed, size=size, market_code=market or DEFAULT_POOL_MARKET, **extra)
        return contract(CATALOGS, "build_store_snapshot")(source_name, category_slug, seed=seed, size=size)

    def pool_market(category: str, store_id: str, *, seed: int, home: str) -> str:
        return str(contract(CATALOGS, "pool_market")(category, store_id, seed=seed, home=home))

    def fixtures(snapshot: Any) -> list[dict[str, Any]]:
        return list(contract(FIXTURES, "fixtures_from_snapshot")(snapshot))

    def licence_rights(source_name: str) -> Rights | Mapping[str, Any]:
        return cast(Rights | Mapping[str, Any], contract(CATALOGS, "licence_rights")(source_name))

    personas_path = settings.paths.data / PERSONAS_FILE
    return Bindings(
        agent=agent,
        limits=limits,
        shopper=shopper,
        generate_tasks=generate_tasks,
        live_gate=live_gate,
        discover=discover,
        open_ucp_surface=open_ucp_surface,
        open_sandbox_surface=open_sandbox_surface,
        store_snapshot=store_snapshot,
        fixtures=fixtures,
        licence_rights=licence_rights,
        catalog_restricted=catalog_restricted,
        pool_market=pool_market,
        personas=personas_at(personas_path),
        checkout=checkout_with(personas_path),
        replica_snapshot=replica_snapshot,
        sandbox_app=sandbox_app,
        open_website_surface=open_website_surface,
        capture_catalog=capture_catalogs(settings),
        playground_wallet=wallet_with(personas_path),
    )


def catalog_restricted(source_name: str) -> bool:
    """The `restricted` flag of a catalog source's licence, from its manifest (a catalog, or the open pools' pool
    manifest, D91)."""
    source = contract(CATALOGS, "load_manifest")(source_name)
    restricted = getattr(getattr(source, "licence", None), "restricted", None)
    if not isinstance(restricted, bool):
        raise ConfigError(
            f"catalog source {source_name!r} has no licence.restricted flag",
            hint="every CatalogSource declares its Licence (spdx, training_allowed, ..., restricted)",
        )
    return restricted


def _parameters(fn: Callable[..., Any]) -> Mapping[str, inspect.Parameter]:
    try:
        return inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}


def _accepted_by(fn: Callable[..., Any], **offered: Any) -> dict[str, Any]:
    """The offered keywords `fn` takes, values that are None left out: extras a contract does not fix are passed only
    where the implementation asks for them."""
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in _accepted(parameters, offered).items() if v is not None}


def _accepted(parameters: Mapping[str, inspect.Parameter], offered: Mapping[str, Any]) -> dict[str, Any]:
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(offered)
    return {
        name: value
        for name, value in offered.items()
        if name in parameters
        and parameters[name].kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }

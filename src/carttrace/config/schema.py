"""The shape of every setting: pydantic models that mirror config.json section by section.

Why typed settings in one place. The lab grew about 45 configuration variables with no single reference, and moved
models by rewriting the environment. Here every setting is a field with a reason, every model is a role in one table,
and a role with no route refuses to start instead of borrowing another role's model. Every model forbids unknown keys,
so a misspelt key fails at load time instead of silently keeping a default. `config.schema.json` is generated from
`ConfigFile` (`carttrace config schema`), so an editor checks config.json against exactly what the loader accepts.

Secrets never live in these models. A field whose name ends in `_env` holds the name of a variable, never its value,
and `carttrace.config.secrets` resolves it. The loader in `carttrace.config.loader` fills `Settings`; relative paths resolve against
the repository root through the validation context, so a model validated without a root keeps them relative.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_validator,
    model_validator,
)

from carttrace.core.canon import pretty
from carttrace.core.errors import ConfigError

EndpointKind = Literal["chat", "clef", "images", "videos"]
DecideStep = Literal["rule", "clef-flash", "clef", "fallback"]
FALLBACK_ROLE = "decide.fallback"
"""The fallback decision stage's role. It resolves to decide.fallback.endpoint, never to an entry of roles (D61)."""
CLAUDE_CLI = "claude_cli"
"""The decide.fallback.endpoint value that names the signed-in Claude Code CLI instead of an entry of endpoints."""

ENV_NAME = r"^[A-Z_][A-Z0-9_]*$"
"""What a field ending in `_env` must hold: a variable name, so a pasted key fails validation instead of loading."""

_ENDPOINT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_ROLE_NAME = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
_ID_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")

SCHEMA_FILE = "config.schema.json"
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def _home(info: ValidationInfo) -> Path | None:
    context = info.context if isinstance(info.context, dict) else {}
    home = context.get("home")
    return home if isinstance(home, Path) else None


def _against_home(value: Path, info: ValidationInfo) -> Path:
    value = value.expanduser()
    home = _home(info)
    return value if home is None or value.is_absolute() else home / value


def _storage_url(value: str, info: ValidationInfo) -> str:
    """file:// URLs and bare paths become absolute file URIs under the repository root; other schemes pass through.

    The path is percent-decoded first, so an already-encoded URL is not encoded twice, and file://localhost/abs
    means /abs as RFC 8089 says.
    """
    if "://" in value and not value.startswith("file://"):
        return value
    rest = unquote(value.removeprefix("file://"))
    if rest.startswith("localhost/"):
        rest = rest.removeprefix("localhost")
    if not rest:
        raise ValueError("a file:// storage URL needs a path, e.g. file://var/store")
    path = _against_home(Path(rest), info)
    return path.as_uri() if path.is_absolute() else f"file://{rest}"


def _base_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("base_url must be an http:// or https:// server root")
    if parts.username or parts.password:
        raise ValueError("base_url must not carry credentials; name the key's variable in api_key_env instead")
    if parts.query or parts.fragment:
        raise ValueError("base_url must not carry a query or a fragment")
    return value.rstrip("/")


def _blob_url(value: str) -> str:
    if not value.startswith(("s3://", "file://")):
        raise ValueError("media_url must be an s3:// or file:// URL")
    return value


RepoPath = Annotated[Path, AfterValidator(_against_home)]
EnvName = Annotated[str, Field(pattern=ENV_NAME)]
StorageUrl = Annotated[str, AfterValidator(_storage_url)]
BaseUrl = Annotated[str, AfterValidator(_base_url)]
BlobUrl = Annotated[str, AfterValidator(_blob_url)]
PositiveRate = Annotated[float, Field(gt=0)]


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True, coerce_numbers_to_str=True)


class Paths(_Section):
    data: RepoPath
    """Checked-in inputs: store queue snapshots, categories and decision types."""
    vendor: RepoPath
    """Vendored protocol specs, filled by `carttrace protocols vendor` from protocols.lock.toml."""
    var: RepoPath
    """Run output and local state: traces when storage is local, status, locks, decisions, media, cassettes."""


class Components(_Section):
    """Which implementation of each replaceable part runs, by its name in the package's PLUGINS table."""

    storage_blobstore: str = Field(alias="storage.blobstore")
    """local or s3. Must agree with the scheme of storage.url."""
    inference_chat: str = Field(alias="inference.chat")
    """openai_compat, or recorded to replay var/cassettes/chat.jsonl."""
    inference_scorer: str = Field(alias="inference.scorer")
    """clef, or recorded to replay var/cassettes/score.jsonl."""
    inference_fallback_scorer: str = Field(default="chat_logprobs", alias="inference.fallback_scorer")
    """chat_logprobs: the LabelScorer of the qwen decision stage, reading label probabilities from a chat endpoint's
    logprobs."""
    ops_notifier: str = Field(alias="ops.notifier")
    """slack_webhook or file."""
    stores_sheet: str = Field(alias="stores.sheet")
    """google_sheets or csv."""
    media_renderer: str = Field(alias="media.renderer")
    """excalidraw or matplotlib."""
    media_imagery: str = Field(alias="media.imagery")
    """h3, qwen_image or none."""
    media_explainer: str = Field(alias="media.explainer")
    """sonnet, claude_cli or template: who writes the report's explanation."""
    media_planner: str = Field(default="template", alias="media.planner")
    """claude_cli or template: who plans the update video's scenes and the Slack post's words."""
    media_video: str = Field(default="infographic", alias="media.video")
    """infographic (scenes drawn by code, recorded in Chromium), h3 (one H3 clip per storyboard shot) or none."""
    decide_sink: str = Field(default="jsonl", alias="decide.sink")
    """jsonl (var/decisions/) or memory: where the decision engine logs every decision."""

    def kinds(self) -> dict[str, str]:
        """Kind to implementation name, e.g. {"storage.blobstore": "local", ...}."""
        dumped: dict[str, str] = self.model_dump(by_alias=True)
        return dumped

    def name_for(self, kind: str) -> str:
        kinds = self.kinds()
        if kind not in kinds:
            raise ConfigError(
                f"no component kind {kind!r} in config.json components",
                hint=f"choose one of {sorted(kinds)}, or add the kind to config.json and carttrace.config.schema.Components",
            )
        return kinds[kind]


class Storage(_Section):
    url: StorageUrl
    """Where run output goes: file://<path> (relative to the repository root) or s3://bucket/prefix. CARTTRACE_STORAGE
    overrides it."""
    media_url: BlobUrl | None = None
    """s3://bucket/prefix for rendered media, shared through time-limited links."""
    media_link_ttl_s: int = Field(default=604800, ge=1, le=604800)
    """Lifetime of a media link in seconds. S3 SigV4 presigned URLs cannot outlive seven days."""
    aws_profile: str | None = Field(default=None, min_length=1)
    """AWS profile (in ~/.aws) for s3:// stores when the environment sets no AWS_PROFILE, as under systemd. Set it per
    machine in config.local.json; the profile's keys stay in ~/.aws, never here."""
    share_url: BlobUrl | None = None
    """s3://bucket/prefix the team shares through (D116): the official CartTrace traces under
    <share_url>/carttrace-100k/<spec>/ (`carttrace release share`) and the human labels under <share_url>/labels/ (`carttrace labels
    sync`). Unset, both syncs refuse to run."""


_CLOCK_TIME = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"


def _minute_of(clock_time: str) -> int:
    hours, minutes = clock_time.split(":")
    return int(hours) * 60 + int(minutes)


class CapWindow(_Section):
    """A time of day when an endpoint's cap differs from its max_inflight (D114), in gpu.timezone."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", use_attribute_docstrings=True, coerce_numbers_to_str=True, populate_by_name=True
    )

    start: str = Field(alias="from", pattern=_CLOCK_TIME)
    """HH:MM the window opens, inclusive: at 22:00 a 22:00 window is open."""
    end: str = Field(alias="to", pattern=_CLOCK_TIME)
    """HH:MM the window closes, exclusive: at 08:00 a window to 08:00 is closed. Earlier than `from` means the window
    runs past midnight."""
    max_inflight: int = Field(ge=1)
    """The host-wide cap while the window is open."""

    @model_validator(mode="after")
    def _not_empty(self) -> CapWindow:
        if self.start == self.end:
            raise ValueError("a cap window's from and to must differ; leave the window out to keep max_inflight")
        return self

    def covers(self, minute: int) -> bool:
        """Whether the window is open at `minute` past local midnight."""
        start, end = _minute_of(self.start), _minute_of(self.end)
        return start <= minute < end if start < end else minute >= start or minute < end


class Endpoint(_Section):
    """One model server and the model it serves."""

    kind: EndpointKind
    """chat (/v1/chat/completions), clef (/generative_scoring), images or videos."""
    base_url: BaseUrl
    """Server root without a trailing slash, e.g. http://models.example:8080."""
    model: str = Field(min_length=1)
    """The model id sent in every request. The served model is recorded separately from each reply."""
    max_inflight: int = Field(default=32, ge=1)
    """Host-wide cap for this server, enforced across processes by carttrace.inference. CARTTRACE_GPU_MAX_INFLIGHT lowers it. It
    applies whenever no `schedule` window is open."""
    schedule: list[CapWindow] = Field(default_factory=list)
    """Time-of-day caps (D114), e.g. [{"from": "22:00", "to": "08:00", "max_inflight": 56}], in gpu.timezone. The
    first open window wins; outside every window the cap is max_inflight. A running process follows it live, and so
    does the host-wide slot count across processes (carttrace.inference.slots)."""
    timeout_s: float = Field(default=120.0, gt=0)
    """Read timeout for one call."""
    api_key_env: EnvName | None = None
    """Name of the variable holding the key, never the key itself."""
    extra_body: dict[str, JsonValue] = Field(default_factory=dict)
    """Extra fields merged into every request body for this server, e.g. {"chat_template_kwargs": {"enable_thinking":
    true}}. The fields a call sets itself (model, messages, max_tokens, ...) always win."""

    def cap_at(self, local: datetime) -> int:
        """The cap at local time `local` (already in gpu.timezone): the first open schedule window's, else
        max_inflight."""
        minute = local.hour * 60 + local.minute
        for window in self.schedule:
            if window.covers(minute):
                return window.max_inflight
        return self.max_inflight

    @property
    def endpoint_id(self) -> str:
        """The cap is per server, not per role: two roles on one server share its slots."""
        bare = self.base_url.removeprefix("http://").removeprefix("https://").replace("/", "_").replace(":", "_")
        return _ID_UNSAFE.sub("_", bare)


class ModelEndpoint(Endpoint):
    """One role's resolved route to a model server: what `Settings.role(name)` returns."""

    role: str
    """agent, shopper, decide.flash, decide.full, media.video, media.explainer, ..."""
    name: str = ""
    """The endpoint's name in config.json `endpoints` (qwen, clef_flash, ...)."""

    def api_key(self) -> str | None:
        """The key from the variable named by `api_key_env`: the environment first, then ops.secrets_files."""
        if not self.api_key_env:
            return None
        from carttrace.config.secrets import secret

        return secret(self.api_key_env)


class Live(_Section):
    enabled: bool = False
    """One of three gates on live traffic; the others are an explicit --live flag and no pause in the control file.
    CARTTRACE_LIVE=1 sets it."""
    control_file: RepoPath
    """JSON {"global_pause": bool, "paused_stores": [...]}: the brake every live path checks. CARTTRACE_CONTROL_FILE
    overrides it."""
    user_agent: str = Field(min_length=1)
    """Our stable identity, carttrace/<version> (+<info url>). Every HTTP client from carttrace.config.http sends it."""
    agent_profile_url: str | None = None
    """Our UCP agent profile, sent as meta.ucp-agent.profile on every live UCP call."""
    per_store_daily_sessions: int = Field(ge=0)
    """At most this many live sessions per store per day."""
    platform_rps: dict[str, PositiveRate]
    """Host-wide requests per second per platform. `default` applies to any platform not listed."""
    allow_tools: dict[str, list[str]]
    """The live safety allowlist: surface kind to the tool names a live call may use. Everything else is refused."""
    request_timeout_s: float = Field(default=30.0, gt=0)
    """Timeout for one live HTTP request."""
    retention_days_raw: int = Field(default=30, ge=1)
    """Days a live bundle's raw content (turns, wire, state) is kept before `carttrace trace purge` removes it. Shopify's
    API terms (6.2.1, 6.2.3) limit storing merchant data; scores, ids and judge outputs are kept."""
    signing_key_env: EnvName | None = None
    """Name of the variable holding our ECDSA P-256 private key (PEM) for RFC 9421 request signing, Shopify's Signed
    tier. Unset means requests go unsigned (Anonymous tier)."""
    agent_profile_path: RepoPath | None = None
    """Our UCP agent profile document as published at agent_profile_url; generated by `carttrace live profile`."""
    scan_products_per_category: int = Field(default=20, ge=1)
    """Products a static scan samples per store and category."""
    per_store_rps: PositiveRate = 0.5
    """The per-store ceiling of a store capture pass (`carttrace live capture-all`): requests per second to one store, on
    top of its platform bucket, so a big index never takes the whole platform bucket. A store on an independent
    platform has this bucket alone, in captures and in agent sessions alike (D57)."""
    per_store_daily_requests: int = Field(default=2500, ge=1)
    """Requests a store capture pass may send one store per UTC day, across every road to it; a larger pass continues
    the next day."""
    independent_platforms: list[str] = Field(
        default_factory=lambda: ["custom", "woocommerce", "magento", "shopware", "sap_commerce"]
    )
    """Self-hosted platforms whose stores share no host: in a capture pass and in agent sessions (D57) their requests
    draw on the per-store bucket (per_store_rps) only, not on a platform-wide bucket (Shopify's limit is per IP across
    its shops; a WooCommerce store's is its own)."""

    policy_companion: bool = False
    """D67: a live UCP over MCP row on a Shopify store also lists the store's Storefront MCP policy tool
    (search_shop_policies_and_faqs at /api/mcp, read-only), the one tool that endpoint still serves and the only one
    UCP lacks. Off by default: a replica does not serve it, so a row with it is not the twin of a replica row."""

    @model_validator(mode="after")
    def _has_default_rate(self) -> Live:
        if "default" not in self.platform_rps:
            raise ValueError("platform_rps needs a 'default' rate for platforms it does not list")
        return self


BACKSTOP_FACTOR = 3.0
"""The agent's time cap leaves out waits for a GPU slot (carttrace.agents.qwen), so on busy GPUs an episode runs longer than
the cap by the clock; the runner's backstop only catches an agent that hangs, so it allows three caps of wall time."""
BACKSTOP_MIN_S = 30.0
"""The runner's backstop is at least this much more than the time cap."""


def runner_backstop_s(time_cap_s: float, *, min_s: float = BACKSTOP_MIN_S) -> float:
    """The wall time the collector's runner gives an episode's agent before stopping it: BACKSTOP_FACTOR time caps,
    and at least min_s more than one. Anything that must outlive the agent (the sandbox browser) derives from this, so
    the two cannot drift apart."""
    return max(time_cap_s + min_s, time_cap_s * BACKSTOP_FACTOR)


class Gpu(_Section):
    """How the shared GPU servers' caps are read (D114)."""

    timezone: str = "UTC"
    """The IANA zone every endpoint's `schedule` is written in. Infra's hours are in it; UTC unless they say so."""
    recheck_s: float = Field(default=60.0, gt=0)
    """How often a running process reads the caps again (config files and schedule), and how often the collector
    resizes its episode pool to follow them."""

    @field_validator("timezone")
    @classmethod
    def _known_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"gpu.timezone must be an IANA zone such as UTC or Europe/London, not {value!r}") from None
        return value

    def local(self, now: datetime) -> datetime:
        """`now` (timezone-aware) in gpu.timezone."""
        return now.astimezone(ZoneInfo(self.timezone))


class Collect(_Section):
    target_concurrency: int = Field(ge=1)
    """Episodes in flight at once, when concurrency_follows is unset."""
    concurrency_follows: str | None = None
    """An endpoint name (e.g. qwen): the collector's sandbox episodes in flight follow that endpoint's cap right now,
    times concurrency_per_slot, instead of target_concurrency, and the pool resizes live (D114)."""
    concurrency_per_slot: float = Field(default=1.5, gt=0)
    """Episodes per slot of the followed endpoint: an episode spends part of its time off the GPU (tools, store
    calls), so a few more episodes than slots keep the slots busy."""

    def target_for(self, cap: int | None) -> int:
        """Episodes in flight for the followed endpoint's cap `cap` (None when nothing is followed)."""
        if self.concurrency_follows is None or cap is None:
            return self.target_concurrency
        return max(1, math.floor(cap * self.concurrency_per_slot))

    step_cap: int = Field(ge=1)
    """Agent steps before an episode ends with step_cap."""
    time_cap_s: int = Field(ge=1)
    """Wall-clock seconds before an episode ends with time_cap."""
    status_every_s: float = Field(gt=0)
    """How often the runner rewrites var/status/collector.json."""
    supervise_every_steps: int = Field(ge=1)
    """Agent steps between supervision decisions."""
    repeats: int = Field(ge=1)
    """Episodes per manifest row."""
    live_concurrency: int = Field(default=4, ge=1)
    """Live episodes in flight at once, across stores; each store still gets one session at a time."""
    results_db: RepoPath = Path("var/stores/results.sqlite")
    """The results database (carttrace.stores.resultsdb): where the collector and the preflight write mc.stores.result.v1
    rows with their provenance, and what `carttrace stores sync` and the dashboard read."""
    results_path: RepoPath = Path("var/stores/results.jsonl")
    """The JSONL file the results database replaced; written beside it only when results_file_fallback is true, and
    what `carttrace stores results migrate` reads once."""
    results_file_fallback: bool = False
    """Also append every result to results_path, as before the database. Off: the database is the only writer."""
    decisions: Literal["on", "off"] = "off"
    """Sandbox v1 final (D115): whether an episode's decisions (supervision, triage, the shopper's field mapping) may
    ask a decision model. "off", the default and always for a v1 row, decides by rule alone and calls no Clef or
    Clef-flash at all: every such type is in shadow, so a model's answer changed nothing, and Clef is the judge's
    alone. A decision no rule answers abstains with its type's safe default and keeps its evidence in the decision's
    context (`offline_evidence`), so a calibration can ask the model offline later."""
    idle_timeout_s: float = Field(default=1800.0, gt=0)
    """D122: an episode whose bundle gets no new record (a turn, a wire record, a sandbox event) for this long is cut
    as `idle_timeout` and its row runs again (a v1 row is quarantined, D111). Longer than one model call can take
    (endpoints' timeout_s, 900 s for Qwen) so that only a hang trips it: 2026-10-17, a WebMCP row sat 30 minutes
    inside its 45-minute backstop with a live renderer and nothing happening."""
    browser_open_tries: int = Field(default=3, ge=1, le=10)
    """D122: how many times a WebMCP row opens its storefront on a fresh browser before the row fails: a page that
    timed out or aborted while loading (BrowserTimeout, under host load) is tried again after 5 s, then 15 s."""
    quarantine_retries: int = Field(default=2, ge=0, le=10)
    """D111: how many times a row whose bundle was quarantined (harness_error, model error, crash or drained,
    carttrace.collect.quarantine) runs again, as the same episode id. Drained attempts do not count toward it."""
    website_capture_hints: bool = True
    """Seed a live website episode's product index (browse_index, D34) with the product URLs and titles of our
    latest capture of the store, each marked source "capture" in the trace so the judge can tell what the agent found
    itself from what it was handed. On for evaluation; off to measure what an agent finds on its own."""


class SandboxBrowser(_Section):
    """The sandbox browser (carttrace.sandbox.browser): headless Chromium for WebMCP storefronts and 3-D Secure pages."""

    max_browsers: int = Field(default=4, ge=1, le=64)
    """Chromium processes open at once across every process on this host (file locks under paths.var/locks/browser).
    The collector starts at most this many WebMCP episodes at once (carttrace.collect.scheduler, mc-3r1.34); an episode
    that still finds the pool taken by another process waits for a slot rather than open one more."""
    slot_wait_s: float = Field(default=1800.0, gt=0)
    """How long an episode waits for a browser slot before it ends as an infrastructure error. Generous on purpose:
    a browser is held for a whole episode (minutes), and the wait is queueing, not part of the episode's browser
    budget, so a wait behind other processes' episodes must not burn the row."""
    call_timeout_s: float = Field(default=20.0, gt=0)
    """One WebMCP tool call, from executeTool to its result. A page that hangs ends the episode as VOID."""
    navigation_timeout_s: float = Field(default=20.0, gt=0)
    """One page load."""
    episode_timeout_s: float = Field(default=900.0, gt=0)
    """Everything one browser does in an episode, opening included."""
    max_pages: int = Field(default=200, ge=1)
    """Documents (pages and frames) one episode may load; more ends it as an infrastructure error."""
    native_webmcp: bool = False
    """Launch Chromium with its experimental WebMCP feature on, so a native document.modelContext is probed and used
    when it conforms to the profile's WebMCP draft. Off: the vendored shim always runs."""


class CheckoutFaultRate(_Section):
    """One checkout-time fault's rate (D66): a design parameter, since live never pays and so gives no rate."""

    rate_bp: int = Field(default=0, ge=0, le=10_000)
    """Basis points of sandbox order rows that draw this fault."""
    assumed: bool = True
    """True: the rate is chosen, not measured. Every rate here is, until a measured source replaces it."""
    source: str = ""
    """The public source the rate is set against, with what it says, or why there is none."""


class CheckoutFaults(_Section):
    """The checkout-time faults sandbox order rows draw (D66), at most one per row, by the row's seed. Rates are kept
    low, so most journeys can complete; the five kinds together stay under a tenth of the rows."""

    three_ds: CheckoutFaultRate = Field(default_factory=CheckoutFaultRate)
    """The buyer's bank asks for a 3-D Secure challenge at completion (it passes when the buyer completes it)."""
    payment_decline: CheckoutFaultRate = Field(default_factory=CheckoutFaultRate)
    """The payment is declined: buyer_fixable (insufficient funds, expired card: every completion is declined) or
    retryable (processing error: the decline fires once, so a retry with the shopper's renewed approval pays)."""
    reprice: CheckoutFaultRate = Field(default_factory=CheckoutFaultRate)
    """A price changes at completion: the store refuses with price_changed and the new total."""
    stockout: CheckoutFaultRate = Field(default_factory=CheckoutFaultRate)
    """The stock is gone at completion: the reservation is broken and the store refuses as out of stock."""
    unshippable: CheckoutFaultRate = Field(default_factory=CheckoutFaultRate)
    """The store does not ship to the buyer's country: the destination is refused."""
    buyer_fixable_bp: int = Field(default=5000, ge=0, le=10_000)
    """Of the declines, basis points that are buyer_fixable; the rest are retryable."""
    reprice_bp: int = Field(default=1000, ge=1, le=10_000)
    """How much a price change raises the unit price, in basis points of it (at least one minor unit)."""


class SandboxV1Failures(_Section):
    """How the one designed-failure budget of sandbox v1 (D96) splits across its sources, by weight (tenths of the
    budget by default: impossible 3, payment 4, checkout 3). Persona payment
    faults and impossible tasks fail a whole task, so they take whole tasks (a store's draw rounds them by seed); the
    checkout faults are drawn per row and take what is left, which makes each store's budget exact."""

    payment: int = Field(default=4, ge=0)
    """A persona built to fail payment (a decline, a failed or abandoned 3-D Secure challenge) or a wallet that holds
    nothing the row's rail takes (no_instrument), D80."""
    checkout: int = Field(default=3, ge=0)
    """A checkout-time fault forced on the row (D66): 3-D Secure, a decline, a reprice, a stockout, no shipping."""
    impossible: int = Field(default=3, ge=0)
    """A task nothing in the store satisfies, so a refusal is the right answer."""
    checkout_kinds: dict[str, int] = Field(
        default_factory=lambda: {"three_ds": 3, "payment_decline": 3, "reprice": 1, "stockout": 1, "unshippable": 1}
    )
    """The checkout fault kinds' weights among the checkout rows: D66's rates (300, 300, 100, 100, 100 bp) as ratios."""


class SandboxV1(_Section):
    """Sandbox v1 (D96): each store's exact replica, 5 personas x 5 tasks, each task on the
    surface x rail cells, baseline rows half through payment and half to a verified cart (D98), designed failures
    through payment, with baseline, mask and designed-failure shares in one budget,
    toward a finite target (data/sandbox/v1-target.yaml)."""

    enabled: bool = True
    """The sandbox loop builds v1 rounds; false builds the loop's rounds as before D96 (CARTTRACE_LOOP_V1=0 does too)."""
    target: RepoPath = Path("data/sandbox/v1-target.yaml")
    """The finite target: stores, traces per store, per-category totals and how stores are chosen."""
    personas: int = Field(default=5, ge=1)
    """Personas a store's tasks are written for (data/agent/personas.yaml, chosen per store by seed)."""
    tasks_per_persona: int = Field(default=5, ge=1)
    cells_per_task: int = Field(default=4, ge=1)
    """Surface x rail cells each task runs on (the surface x rail matrix of branch sandbox/rails-matrix owns which)."""
    baseline_bp: int = Field(default=9600, ge=0, le=10_000)
    """Rows with the store's full data, nothing withheld or injected, ending at the order or the cart as
    `order_share_bp` splits them."""
    mask_bp: int = Field(default=0, ge=0, le=10_000)
    """0 in v1: the exposure-mask arm moved to the later benchmark phase; the code path stays.
    Above 0: rows with one data type the task turns on withheld, each paired with the same task's full-exposure row on
    the same store, seed, surface and rail."""
    designed_failure_bp: int = Field(default=400, ge=0, le=10_000)
    """Negative rows, every source in one budget (impossible tasks inside it, not on top); owner 2026-10-17: 3-4% of the
    rows, so 400 bp (4 negatives a 100-row store), down from D96's 10%."""
    failures: SandboxV1Failures = Field(default_factory=SandboxV1Failures)
    order_share_bp: int = Field(default=5000, ge=0, le=10_000)
    """D98: the share of each store's baseline rows that run boundary=order, end to end through payment; the rest run
    boundary=cart and end at a verified cart (the agent stops before checkout). Designed payment and checkout faults
    are always order rows. Paying journeys cost about 2.4 times a cart journey's GPU (judge-v11 study s12: median 337
    s against 142, 61 turns against 23), so half the baseline at the cart gives about 1.35 times the rows an hour."""
    mask_types: dict[str, list[str]] = Field(default_factory=dict)
    """Category slug to the data types (carttrace.collect.exposure.TYPES ids) a mask row of that category may withhold, one a
    row, on a task whose decision turns on it. Empty until the project decides which types may be hidden (2026-10-16):
    a category with none plans its mask share as baseline rows. Basic product fields are never maskable."""
    cross_border_bp: int = Field(default=0, ge=0, le=10_000)
    """Of the baseline rows, the share whose shopper delivers to another market (the row's market axis, D96). 0 plans
    every row at home."""
    candidate_tasks: int = Field(default=48, ge=1)
    """Tasks the planner writes per store before v1 picks its 25: room for the fault, impossible and mask choices."""
    candidate_fault_bp: int = Field(default=2000, ge=0, le=10_000)
    """D80's share the candidate draw is written at, so enough tasks are built to fail for the payment source."""
    partial_min_products: int = Field(default=20, ge=1)
    """A capture stopped by a wall or a throttle after this many products were read in detail still makes a replica."""
    reference: bool = True
    """Rows also run under the scripted reference agent (no GPU), as a sample of `reference_share_bp`."""
    reference_share_bp: int = Field(default=1000, ge=0, le=10_000)
    """D111: the share of Qwen rows that get a reference twin, in basis points (1000: one reference row per ten Qwen
    rows), chosen by code, exact per store where the counts allow and spread over category, boundary, kind, cell and
    persona (carttrace.collect.v1_cut.sample_references). 10000 gives every Qwen row its twin, as before D111. The build-time
    offline replay (D104) runs every row whatever this says."""
    union_core_share: float = Field(default=0.3, ge=0, le=1)
    """D117: a category's union core is the curated union types at least this share of its real stores show
    (data/bench/union_core.json); every open stand-in's products carry them, filled from the catalog, their own text,
    then a seeded synthetic value. 1.0 fills nothing beyond what every store shows."""
    open_platforms: list[str] = Field(
        default_factory=lambda: [
            "shopify",
            "woocommerce",
            "magento",
            "medusa",
            "saleor",
            "vendure",
            "prestashop",
            "bigcommerce",
            "generic",
        ]
    )
    """The open lane's packs, drawn per row (D85): a twin has no platform of its own."""

    @model_validator(mode="after")
    def _shares(self) -> SandboxV1:
        total = self.baseline_bp + self.mask_bp + self.designed_failure_bp
        if total != 10_000:
            raise ValueError(f"sandbox.v1 shares add to {total} basis points, not 10000")
        return self


class Sandbox(_Section):
    host: str = "127.0.0.1"
    """The sandbox server binds here. Control routes refuse any other client address."""
    port: int = Field(default=8787, ge=1, le=65535)
    catalogs_dir: RepoPath = Path("var/catalogs")
    """Downloaded catalog sources and the per-store catalogs built from them."""
    default_catalog: str = "replica"
    """Catalog source for sandbox stores when a manifest names none. `replica`: each store's own captured products,
    and a store with no capture gets no sandbox rows. A dataset catalog (abo, off, wands) as a default put stand-in
    products in categories they do not belong to."""
    allow_restricted_catalogs: bool = False
    """Lets sources whose licence forbids training or commercial use build stores for internal research.
    Their bundles are always marked training_allowed=false and publication_allowed=false."""
    open_pool_markets: Literal["mixed", "home"] = "mixed"
    """How a stand-in store of the open pools (D91) is built. mixed (the project's choice, 2026-10-16): every member's
    products in one US store, Lazada's priced from the same synthetic USD bands as ABO's and WANDS', its own peso
    price kept as provenance. home: one member group a store, Lazada's only in a PH store priced in pesos."""
    agent_views: RepoPath | None = Path("data/sandbox/views")
    """The recorded agent views (D100, carttrace.sandbox.views): a stand-in of a store whose agent view was recorded live
    serves its open products in that store's recorded shape. None serves every stand-in in its platform pack's
    documents, as before."""
    browser: SandboxBrowser = Field(default_factory=SandboxBrowser)
    """Limits of the sandbox browser that hosts WebMCP storefront pages."""
    checkout_faults: CheckoutFaults = Field(default_factory=CheckoutFaults)
    """The checkout-time faults sandbox order rows draw, with each rate's source (D66). A manifest copies them when it
    plans order rows, so its rows draw the same faults whatever this file says later."""
    payment_fault_bp: int = Field(default=1000, ge=0, le=10_000)
    """D80: of the sandbox order rows whose tasks are crossed with personas, the share in basis points whose payment is
    designed to fail: a persona built to fail (a decline, a failed or abandoned 3-D Secure challenge) or a wallet
    paired on purpose with a rail that takes nothing it holds (no_instrument). Every other row's persona pays, on a
    rail its wallet can pay on. 1000 (a tenth), down from the about 40% of personas.yaml built to fail, by the
    project's decision of 2026-10-15 (he wants payments). A manifest copies it when it plans order rows."""
    three_ds: Literal["challenge", "decision"] = "challenge"
    """How a sandbox order row meets 3-D Secure outside v1: challenge parks the payment on a bank page the agent hands
    the shopper (D71, D94); decision has the PSP authenticate and approve or decline at once, with no page and no
    challenge tool (D111). A v1 row always decides, and so does the payments check."""
    v1: SandboxV1 = Field(default_factory=SandboxV1)
    """Sandbox v1 (D96): the rounds the sandbox loop builds by default, toward a finite target."""


class AgentIntents(_Section):
    """The per-store intent writer (carttrace.agents.intents_custom): a model writes free-form intents from a store's brief,
    code validates each against the capture, and the set is kept under var/agent/intents."""

    model: str = Field(default="sonnet", min_length=1)
    """The signed-in Claude CLI model the writer asks (sonnet, opus, haiku, or a full model id)."""
    n: int = Field(default=100, ge=1)
    """Intents per store."""
    batch: int = Field(default=20, ge=1)
    """Intents asked of the model per call."""
    cap_usd: float = Field(default=2.0, gt=0)
    """Most USD one store's writing may spend; the writer stops there with what it has."""
    kind_cap_share: float = Field(default=0.15, gt=0, le=1)
    """No one decisive data kind may take more than this share of a store's intents."""
    min_kinds: int = Field(default=8, ge=1)
    """At least this many data kinds per store, when the store exposes that many."""


class AgentShopper(_Section):
    """How the simulated shopper says what its rules decided (D118). The rules decide every fact of a reply (which
    field, its value or none, accept or refuse, consent and its total, what is disclosed); with `wording` model, the
    `shopper` role's model words those facts in the persona's voice, code checks the words keep exactly those facts,
    and the rules' own line is sent when they do not."""

    wording: Literal["model", "rules"] = "rules"
    """`model`: the shopper's model words each reply, behind the fact check; `rules`: the rules' lines, as before."""
    max_tokens: int = Field(default=120, ge=16, le=512)
    """Per wording call; thinking is off."""
    temperature: float = Field(default=0.8, ge=0, le=2)
    retries: int = Field(default=1, ge=0, le=3)
    """Wording calls after a first one the fact check refused, before the rules' line is sent."""
    history: int = Field(default=6, ge=0, le=20)
    """Most recent conversation messages the wording prompt shows (the opening is always shown)."""


class Agents(_Section):
    prompt: RepoPath = Path("data/prompts/agent-v6.md")
    """The protocol agent's system prompt. A prompt change is a code change: it is versioned in git."""
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1)
    """Per agent turn."""
    context_chars: int = 0
    """Cap on the conversation history sent per model call, in characters (experiments; 0 keeps the full history).
    Over the cap the oldest tool results are replaced by a placeholder until the history fits; the system prompt,
    the shopper's opening and the newest turns are never cut."""
    context_budget_tokens: int = Field(default=0, ge=0)
    """Opt-in experiments only (D82; 0, the default, is off): the most prompt tokens one agent call may send. Over it,
    the oldest tool results are shortened in the model's view to stubs naming the tool, its arguments and the ids the
    result held, until the call is at 60% of the budget; the trace keeps every result in full. Off by default because
    a compacted history is one a real agent would not see: a release leaves such rows out (rule context_budget), and
    benchmark runs keep it off. 96,000 (the p97 of successful sandbox rows' peak context on 2026-10-16) is the value
    measured for an experiment arm."""
    buyer_profile: Literal["persona", "none"] = "persona"
    """Where a sandbox checkout row's buyer profile comes from (D77): `persona` gives the agent the shopper's email,
    phone, shipping address and name (the task's buyer block, drawn from the row's persona) in its context, so it
    fills the checkout without asking; `none` keeps them with the shopper, who gives each only when asked (the
    comparison arm). A live row never gets one: no email or phone is ever typed into a live store."""
    intents: AgentIntents = Field(default_factory=AgentIntents)
    shopper: AgentShopper = Field(default_factory=AgentShopper)
    """The simulated shopper's wording (D118)."""
    """The per-store intent writer's model, size and cost cap."""
    under_test: list[str] = Field(default_factory=list)
    """Every model a run may use as its agent, beside roles.agent's own. The agent under test never decides or judges
    (D61): the decision router refuses to start when its fallback endpoint or model is one of these."""


class Judge(_Section):
    version: str = Field(default="judge-v1", min_length=1)
    """Name the judge writes its output under, next to each bundle (judge/<version>.json)."""
    use_clef: bool = True
    """Ask Clef the decision questions state cannot answer. False gives a code-only card."""
    panel: list[str] = Field(default_factory=list)
    """More decision engines asked the same questions (clef, or another decide stage but clef-flash), each recorded on
    the step beside the primary's answer and never counted; `carttrace judge calibrate --panel` reports where they
    disagree, and the labelling queue puts those questions first."""
    order: list[str] = Field(default_factory=lambda: ["rule", "clef"], min_length=1)
    """The decide stages the judge asks, in order: rules, then the full Clef model. Never clef-flash, and never the
    shared decide.order, so a collection setting cannot route the judge to another model."""

    @model_validator(mode="after")
    def _full_clef_only(self) -> Judge:
        # the judge uses the full Clef model only, never clef-flash.
        if "clef-flash" in (*self.order, *self.panel):
            raise ValueError(
                "the judge asks the full Clef model only: leave clef-flash out of judge.order and judge.panel"
            )
        return self


class BenchScorecard(_Section):
    """The store scorecard (carttrace.bench.scorecard)."""

    count_unvalidated_clef: bool = True
    """Count Clef answers of types not yet promoted (no human validation yet) in a score's value, marked "Clef
    unvalidated: x%". False keeps them out of the value and marks the score partial."""


class Bench(_Section):
    min_stores_verified: int = Field(default=3, ge=1)
    """Stores a checklist item's paired lift must hold in before it counts as verified."""
    must_lift_points: float = Field(default=10.0, gt=0)
    """Success difference, in percentage points, that makes a verified item a MUST."""
    should_lift_points: float = Field(default=3.0, gt=0)
    """Smallest difference that makes an item a SHOULD; below it the item is dropped."""
    scorecard: BenchScorecard = Field(default_factory=BenchScorecard)


class DecideFallbackCli(_Section):
    """The signed-in Claude Code CLI as the fallback stage (decide.fallback.endpoint "claude_cli"): no API key."""

    model: str = Field(default="haiku", min_length=1)
    """The CLI model alias (haiku, sonnet) or full name the fallback asks."""
    binary: str = Field(default="claude", min_length=1)
    """The CLI to run: a name on PATH or a path."""
    timeout_s: float = Field(default=60.0, gt=0)
    """Seconds one call may take before its process is stopped and the stage counts as failed."""
    max_budget_usd: float = Field(default=0.05, gt=0)
    """The CLI's own spending cap for one call."""
    max_inflight: int = Field(default=8, ge=1)
    """CLI processes running at once in one process; each is a Node process of its own."""


class DecideFallback(_Section):
    """The fallback stage (D61): a model that is not the agent under test answers a decision only when both Clef
    stages failed or were skipped by their breakers, and only for the decision types listed here. Off by default."""

    endpoint: str | None = None
    """Where the fallback asks: an endpoint of kind chat in config.json endpoints (label probabilities from its
    logprobs), or "claude_cli" for the signed-in Claude Code CLI (decide.fallback.cli). None: no fallback, and a
    decision abstains when Clef gives no answer, as before D56. It may never be the agent's endpoint or model."""
    types: list[str] = Field(default_factory=list)
    """Decision type ids or shell patterns (shopper.*, supervise.*) the fallback may answer. Empty: none. The judge's
    own types are left out on purpose: their routing is the judge session's to choose."""
    text_confidence: float | None = Field(default=None, gt=0, le=1)
    """The probability given to the fallback's answer when it is read from text only (the CLI, or a chat endpoint
    without logprobs): the agreement measured against clef-flash on recorded evidence, since no per-item probability
    exists. None: such an answer is refused and the decision abstains."""
    cli: DecideFallbackCli = Field(default_factory=DecideFallbackCli)

    @property
    def uses_cli(self) -> bool:
        """Whether the fallback asks the signed-in Claude Code CLI rather than an HTTP endpoint."""
        return self.endpoint == CLAUDE_CLI


class Decide(_Section):
    order: list[DecideStep] = Field(min_length=1)
    """Who answers, in order: rule, then clef-flash, then clef on escalation, then fallback when both Clef stages
    failed (only for decide.fallback.types, and only when decide.fallback.endpoint is set)."""
    breaker_failures: int = Field(ge=1)
    """Consecutive Clef failures that open the circuit breaker."""
    breaker_cooldown_s: float = Field(gt=0)
    """Seconds an open breaker waits before trying Clef again."""
    fallback: DecideFallback = Field(default_factory=DecideFallback)

    @field_validator("order", mode="before")
    @classmethod
    def _no_qwen(cls, value: Any) -> Any:
        if isinstance(value, list) and "qwen" in value:
            raise ValueError(
                "decide.order no longer takes qwen: the agent under test never decides (D61); name fallback instead "
                "and set decide.fallback.endpoint to a model that is not the agent"
            )
        return value

    @model_validator(mode="after")
    def _no_repeats(self) -> Decide:
        if len(set(self.order)) != len(self.order):
            raise ValueError("decide.order lists a step twice")
        if "fallback" in self.order and self.order[-1] != "fallback":
            raise ValueError("decide.order must list fallback last: it answers only after the Clef stages failed")
        return self


class OpsPulse(_Section):
    """The progress digest (`carttrace ops pulse`, D65): one card every few hours, the one stream the team reads on Slack."""

    style: Literal["image", "text"] = "image"
    """image: one card drawn by code, uploaded to storage.media_url, and a one-line post; text: the same lines as
    Slack text. An image that cannot be drawn or uploaded falls back to the one line."""
    every_hours: int = Field(default=2, ge=1, le=24)
    """The digest's period: mc-pulse.timer fires at 00:00 UTC and every this many hours (it must divide 24), and a
    digest with no earlier post on record covers the period before it."""
    target_good_traces: int = Field(default=100_000, ge=1)
    """The goal the progress bar fills towards: release-quality traces (D65)."""
    frames_fresh_hours: float = Field(default=2.0, gt=0)
    """Good traces are read from the analytics frames of the current judge version when they were built this
    recently, and from the judge cards otherwise."""
    model: str = Field(default="haiku", min_length=1)
    """The model the signed-in Claude CLI runs to phrase the LEARNED, CHANGED and PROBLEMS lines from computed facts.
    Every number is computed by code; when the model fails, code phrases the lines."""
    timeout_s: float = Field(default=180.0, gt=0)
    """One phrasing call; past this, code phrases the lines."""

    @field_validator("every_hours")
    @classmethod
    def _divides_the_day(cls, value: int) -> int:
        if 24 % value:
            raise ValueError("ops.pulse.every_hours must divide 24 (1, 2, 3, 4, 6, 8, 12 or 24)")
        return value


Stream = Literal[
    "digest",
    "carttrace",
    "breakage",
    "watch",
    "unit_failed",
    "haiku_watch",
    "manager",
    "audit",
    "daily",
    "autopilot",
    "notify",
]
"""Who posts: the progress digest, the CartTrace update, breakage alerts, the watch's other alerts, other units'
failures, the Haiku watcher, the pipeline manager, the auditor, the daily media update, the autopilot and
`carttrace ops notify`."""


class OpsWatch(_Section):
    """Hysteresis for the watch's alerts. An alert opens once it has fired for `open_after_minutes` and closes once it has been clear for
    `close_after_minutes`; in between, nothing is posted, so an incident is one open and one close."""

    open_after_minutes: float = Field(default=9.0, ge=0)
    """An alert must fire this long before it opens: three watch runs five minutes apart by default."""
    close_after_minutes: float = Field(default=14.0, ge=0)
    """An open alert must stay clear this long before it closes."""
    open_now_rules: list[str] = Field(
        default_factory=lambda: ["collector_stopped", "stalled", "gpu_idle", "loop_unit_down"]
    )
    """Rules that open on the first run they fire: each already waits out its own window (stall_minutes,
    idle_minutes), or means collection stopped."""


class OpsCartTrace(_Section):
    """The CartTrace update (`carttrace dashboard carttrace-update`): what CartTrace-100k has collected, posted every few
    hours and once a day in short, from the dashboard's own /api/carttrace numbers (v1 rows of the locked spec)."""

    every_hours: int = Field(default=3, ge=1, le=24)
    """mc-carttrace-update.timer fires at 00:00 UTC and every this many hours (it must divide 24)."""
    daily_hour_utc: int = Field(default=8, ge=0, le=23)
    """mc-carttrace-daily.timer posts the short daily summary (the last 24 hours) at this hour, UTC."""
    screenshot: bool = True
    """Attach a screenshot of the /carttrace page's top section, uploaded to storage.media_url; without it, or when
    it fails, the post links the section instead."""
    top: int = Field(default=5, ge=1, le=12)
    """Categories shown at the top and at the bottom of the coverage line."""
    timeout_s: float = Field(default=60.0, gt=0)
    """Reading /api/carttrace and taking the screenshot, each."""

    @field_validator("every_hours")
    @classmethod
    def _divides_the_day(cls, value: int) -> int:
        if 24 % value:
            raise ValueError("ops.carttrace.every_hours must divide 24 (1, 2, 3, 4, 6, 8, 12 or 24)")
        return value


class OpsSlack(_Section):
    """Which stream reaches Slack (D65). one stream on Slack, the digest, plus a short alert
    when something breaks; everything else keeps working and writes to the file notifier (var/notify)."""

    streams: dict[Stream, str] = Field(default_factory=dict)
    """Stream to ops.notifier block (slack_webhook, file). A stream not listed uses components["ops.notifier"]."""
    breakage_rules: list[str] = Field(
        default_factory=lambda: ["collector_stopped", "stalled", "gpu_idle", "episode_errors", "loop_unit_down"]
    )
    """The watch rules whose alerts are breakage: collection stopped or stalled, the GPU idle with work queued, a
    harness fault spike, a loop unit not running. Their posts go to the breakage stream, cut to two lines; every
    other rule's go to the watch stream."""
    breakage_units: list[str] = Field(default_factory=lambda: ["mc-collector", "mc-live-loop", "mc-sandbox-loop"])
    """Units whose failure (mc-alert@) is breakage; any other unit's failure goes to the unit_failed stream and shows
    in the digest's PROBLEMS line."""
    channel: str = ""
    """The channel the webhook posts to, as people know it (e.g. #carttrace), for the dashboard's 15.0 Slack
    history; the webhook URL itself names no channel."""


class OpsAudit(_Section):
    """The hourly auditor (`carttrace ops audit`): what it reads, which model reviews it, and when a finding counts."""

    model: str = Field(default="opus", min_length=1)
    """The model the signed-in Claude CLI runs for the QA pass, as an alias (opus, sonnet) or a full name."""
    hours: float = Field(default=6.0, gt=0, le=168)
    """The window: journals, judge cards and the backlog are read this far back. A timer's journal is read over at
    least one of its periods, so the daily units are always seen."""
    timeout_s: float = Field(default=600.0, gt=0)
    """One model call; past this the report is written by code alone, and says so."""
    unjudged_after_minutes: float = Field(default=30.0, gt=0)
    """A finished episode with no judge card after this long is a judging gap."""
    unjudged_fail: int = Field(default=500, ge=1)
    """This many late cards or more is a failure rather than a warning."""
    min_cards: int = Field(default=50, ge=1)
    """Runs with fewer cards than this are too small for the outcome shares below to mean anything."""
    max_void_share: float = Field(default=0.05, ge=0, le=1)
    """A run whose share of VOID cards is above this is anomalous."""
    max_no_answer_share: float = Field(default=0.25, ge=0, le=1)
    """A run whose share of NO_ANSWER cards is above this is anomalous."""
    max_runner_error_share: float = Field(default=0.05, ge=0, le=1)
    """A run whose share of harness_error episodes (or the collector's failed episodes) is above this is anomalous."""
    min_free_gb: float = Field(default=50.0, ge=0)
    """Free disk under the var directory below this fails the hygiene check."""
    max_var_gb: float = Field(default=1000.0, gt=0)
    """The var directory above this warns."""
    max_backlog_added: int = Field(default=100, ge=1)
    """More backlog records than this in one window warns: something is appending in a loop."""
    queue_quiet_hours: float = Field(default=3.0, gt=0)
    """A running queue whose log has not moved for this long warns."""
    repeat_hours: float = Field(default=6.0, gt=0)
    """An issue still open is posted again this often."""
    all_clear_hours: float = Field(default=6.0, gt=0)
    """The one-line all-clear is posted at most this often."""
    max_prompt_chars: int = Field(default=60000, ge=2000)
    """The facts handed to the model are cut to fit this, dropping quoted log lines first."""


class Ops(_Section):
    slack_webhook_env: EnvName
    """Name of the variable holding the Slack webhook URL."""
    secrets_files: list[RepoPath] = Field(default_factory=list)
    """KEY=VALUE files read, in order, when a secret is not in the environment. ~ is expanded."""
    stall_minutes: float = Field(gt=0)
    """Alert when no episode finishes for this long while work is queued."""
    idle_fraction: float = Field(ge=0, le=1)
    """Alert when fewer than this fraction of GPU slots are busy while work is queued..."""
    idle_minutes: float = Field(gt=0)
    """...for this many minutes."""
    repeat_critical_hours: float | None = Field(default=None, gt=0)
    """Post a critical alert that stays open again this often. Unset: the carttrace.ops default (3 hours)."""
    error_rate: float | None = Field(default=None, gt=0, le=1)
    """Error fraction at which an error spike counts. Unset: the carttrace.ops default (0.10)."""
    error_min_count: int | None = Field(default=None, ge=1)
    """Fewer errors than this are never a spike, whatever the rate. Unset: the carttrace.ops default (5)."""
    error_window_minutes: float | None = Field(default=None, gt=0)
    """Window the error rate is measured over. Unset: the carttrace.ops default (30 minutes)."""
    down_after_failures: int | None = Field(default=None, ge=1)
    """Consecutive failed probes before an endpoint counts as down. Unset: the carttrace.ops default (2)."""
    probe_cache_s: float | None = Field(default=None, ge=0)
    """How long a probe result is reused. Unset: the carttrace.ops default (120 s)."""
    trace_prefix: str | None = None
    """Storage key prefix the pulse and the watch scan for trace bundles. Unset: the trace writer's episodes/."""
    pulse: OpsPulse = Field(default_factory=OpsPulse)
    audit: OpsAudit = Field(default_factory=OpsAudit)
    """The hourly auditor that reads every automation's logs (D21)."""
    slack: OpsSlack = Field(default_factory=OpsSlack)
    """Which stream reaches Slack and which writes to the file notifier (D65)."""
    watch: OpsWatch = Field(default_factory=OpsWatch)
    """When the watch's alerts open and close (hysteresis)."""
    carttrace: OpsCartTrace = Field(default_factory=OpsCartTrace)
    """The CartTrace update: every few hours and once a day."""
    loop_units: list[str] = Field(default_factory=lambda: ["mc-live-loop", "mc-sandbox-loop"])
    """Always-on units the watch expects running; one that failed or is gone opens loop_unit_down."""


class MediaClaudeCli(_Section):
    """The Claude Code CLI as a model client (`claude -p`): how media reaches Sonnet without an API key."""

    binary: str = Field(default="claude", min_length=1)
    """The CLI's executable, looked up on PATH."""
    model: str = Field(default="sonnet", min_length=1)
    """An alias (sonnet, opus) or a full model name, passed as --model."""
    timeout_s: float = Field(default=300.0, gt=0)
    """One call; the process is stopped after this and the template stands in."""
    max_budget_usd: float | None = Field(default=1.0, gt=0)
    """Spending cap per call (--max-budget-usd); unset sends none."""
    effort: str | None = None
    """low, medium, high, xhigh or max (--effort); unset uses the CLI's default."""
    structured: bool = False
    """Ask for structured output (--json-schema). Off: the schema is written into the system prompt, and the reply is
    parsed and validated by media, which guards it either way."""


class Media(_Section):
    daily_update_hour_utc: int = Field(ge=0, le=23)
    """Hour (UTC) the daily update runs."""
    palette: str = Field(min_length=1)
    """Design system for rendered media and reports."""
    art: str = Field(default="general-machines", min_length=1)
    """The look of the update video and its cards (palette and fonts), and the art direction for image and video
    models (H3, Qwen-Image): a file in data/media/art/."""
    claude_cli: MediaClaudeCli = Field(default_factory=MediaClaudeCli)
    video_art: bool = False
    """Put the decorative hero art (media.imagery) behind the update video's scenes. Off: the video is drawn by code
    alone."""


class Stores(_Section):
    sheet_id: str = Field(min_length=1)
    """The team's store queue Google Sheet."""
    queue_tab: str = Field(min_length=1)
    evidence_tab: str = Field(min_length=1)
    credentials_env: EnvName
    """Name of the variable holding the Google service-account key: a path to the JSON file, or the JSON itself."""
    sync_tab: str | None = None
    """Tab the sync writes machine-owned columns into; None means the evidence tab."""
    id_column: str = Field(default="ID", min_length=1)
    """Header of the column holding the store id (store-001, ...)."""


class Trace(_Section):
    staging_dir: RepoPath = Path("var/staging/traces")
    """Where episodes are written before their bundle is uploaded to storage. Must be a local filesystem."""
    redaction_salt_env: EnvName | None = None
    """Name of the variable holding a secret salt for redaction placeholders. Unset means unsalted, which is fine
    inside the team but must be set before bundles leave it."""


class Autopilot(_Section):
    """The deterministic scheduler (carttrace.autopilot), run by mc-autopilot.timer. Every choice it makes is a function of
    this section, the store queue and what storage already holds, so the same state always gives the same next run."""

    enabled: bool = True
    """False pauses the scheduler; a run already going finishes."""
    every_minutes: int = Field(default=10, ge=1, le=1440)
    """How often mc-autopilot.timer ticks."""
    categories: list[str] = Field(default_factory=list)
    """Category slugs to work through, in this order; empty means every category a licensed catalog covers."""
    catalogs: dict[str, str] = Field(default_factory=dict)
    """Category slug to catalog source; a category not named here uses the licensed source with the most products."""
    stores_per_run: int = Field(default=10, ge=1)
    tasks_per_store: int = Field(default=4, ge=1)
    repeats: int = Field(default=3, ge=1)
    modes: list[str] = Field(default_factory=lambda: ["sandbox", "reference"])
    """Run modes for planned rows: sandbox and reference; live only when `live` is true and the live gates are open."""
    conditions: list[str] = Field(default_factory=lambda: ["baseline", "detail_hidden"])
    daily_episode_cap: int = Field(default=20000, ge=0)
    """The most episodes the scheduler starts per UTC day; 0 means no cap."""
    live: bool = False
    """Plan live rows for stores added in live mode. The three live gates (CARTTRACE_LIVE=1, live.enabled, the control file)
    still decide whether any live request is sent."""
    surfaces: list[str] = Field(default_factory=list)
    """Sandbox surfaces each store's sandbox and reference rows are crossed with, among those the store serves
    (carttrace.collect: ucp_mcp, ucp_rest, ...); empty keeps every row on UCP over MCP. Live rows stay on UCP over MCP."""
    product_rotation: bool = True
    """Spread each store's tasks over its catalog (product type, price band, decisive detail; one task per product),
    and once every planned store is done, plan the next window of each store's products, so coverage grows run by run
    without repeating a product until the catalog is used up. False takes the first tasks generated, as before."""


class Dashboard(_Section):
    """The team dashboard (carttrace.dashboard), served by mc-dashboard.service."""

    host: str = "127.0.0.1"
    """Address to bind. Loopback by default; set this machine's Tailscale address in config.local.json to share it
    with the team on the tailnet, never a public interface."""
    port: int = Field(default=8790, ge=1, le=65535)
    token_env: EnvName = "CARTTRACE_DASHBOARD_TOKEN"  # noqa: S105 - the name of a variable, not a secret
    """Name of the variable holding the token that actions (adding a store, pausing the scheduler) require."""
    refresh_s: int = Field(default=30, ge=5)
    """How often open pages refresh their data."""
    review_queue: RepoPath = Path("var/review/queue.csv")
    """The judge questions 8.0 Review asks people, a labelling sheet `carttrace dashboard review-queue` writes offline."""
    review_answers: RepoPath = Path("var/review/answers.jsonl")
    """The answers people give on 8.0 Review, one JSON line each; `carttrace dashboard review-export` turns them into a
    filled sheet for `carttrace judge labels --import`."""
    labels_dir: RepoPath = Path("var/labels")
    """The team's human labels in the one label schema (data/labels/label.schema.json, D116): append-only
    <labeller>/<yyyy-mm-dd>.jsonl, the merged view under merged/, and the sync's state. 8.0 Review writes here."""
    viewer_logs: RepoPath = Path("var/viewer/logs")
    """Where `carttrace dashboard viewer-export` writes one Inspect AI eval log per run, and what mc-trace-viewer.service
    serves with `inspect view`. Derived data: deleting it only costs a re-export."""
    viewer_state: RepoPath = Path("var/viewer/export-state.json")
    """The exporter's record of what each log was built from, so a timer tick rewrites only the runs that changed."""
    viewer_port: int = Field(default=8791, ge=1, le=65535)
    """The port `inspect view` listens on, on dashboard.host (the same bind rule as the dashboard)."""
    viewer_url: BaseUrl | None = None
    """The trace viewer's address as teammates open it, for example http://127.0.0.1:8791. Episode pages and the
    trace list link each episode to its sample there; unset, no link is shown."""
    viewer_days: int = Field(default=2, ge=1)
    """Day folders mc-trace-export.timer re-checks each tick (today and the day before by default)."""
    viewer_workers: int = Field(default=4, ge=1, le=64)
    """Processes that export runs in parallel; the CPUs are shared with collection, so keep it modest."""
    explorer_port: int = Field(default=8792, ge=1, le=65535)
    """The port Invariant Explorer (ops/explorer/, mc-explorer.service) publishes on dashboard.host."""
    explorer_url: BaseUrl | None = None
    """Explorer's address as teammates open it, for example http://127.0.0.1:8792. Episode pages and the trace
    list link each pushed episode to its trace there; unset, no link is shown."""
    explorer_data: RepoPath = Path("var/explorer")
    """Explorer's pinned source (src/), its database (database/) and uploaded images (images/)."""
    explorer_state: RepoPath = Path("var/explorer/push-state.sqlite")
    """`carttrace dashboard explorer-push`'s record of each pushed episode: its trace id, dataset and what it was built
    from. The dashboard reads trace ids from it for its links."""
    explorer_days: int = Field(default=2, ge=1)
    """Day folders mc-explorer-push.timer re-checks each tick."""
    explorer_workers: int = Field(default=4, ge=1, le=64)
    """Processes that turn bundles into traces while the push runs."""
    explorer_gpu_usd_per_hour: float = Field(default=0.0, ge=0.0)
    """The price of an hour of our own GPU, for the cost on each trace: billed API cost plus GPU time at this rate.
    0 counts billed cost only."""


class Price(_Section):
    """What one model costs through an API, in US dollars per million tokens (OpenRouter's list price)."""

    openrouter_id: str = Field(min_length=1)
    """The model's id on OpenRouter (qwen/qwen3.8-27b), which `carttrace dashboard prices --refresh` looks up."""
    input: Decimal = Field(ge=0)
    """Per million prompt tokens."""
    output: Decimal = Field(ge=0)
    """Per million completion tokens."""
    cache_read: Decimal | None = Field(default=None, ge=0)
    """Per million prompt tokens served from the provider's prompt cache; None when the provider has no cache price."""


def _default_prices() -> dict[str, Price]:
    haiku = Price(openrouter_id="anthropic/claude-haiku-5.5", input=Decimal("0.10"), output=Decimal("0.50"))
    return {
        "qwen3.8-27b": Price(
            openrouter_id="qwen/qwen3.8-27b", input=Decimal("0.24"), output=Decimal("2.20"), cache_read=Decimal("0.05")
        ),
        "clef": haiku.model_copy(update={"cache_read": Decimal("0.01")}),
        "clef_flash": haiku.model_copy(update={"cache_read": Decimal("0.01")}),
    }


def _default_alternatives() -> dict[str, Price]:
    return {
        "qwen3.6-35b-a3b": Price(
            openrouter_id="qwen/qwen3.6-35b-a3b",
            input=Decimal("0.15"),
            output=Decimal("1.00"),
            cache_read=Decimal("0.05"),
        ),
        "qwen3.5-35b-a3b": Price(
            openrouter_id="qwen/qwen3.5-35b-a3b",
            input=Decimal("0.15"),
            output=Decimal("1.00"),
            cache_read=Decimal("0.05"),
        ),
        "qwen3.5-27b": Price(openrouter_id="qwen/qwen3.5-27b", input=Decimal("0.26"), output=Decimal("2.60")),
        "qwen3.6-27b": Price(openrouter_id="qwen/qwen3.6-27b", input=Decimal("0.45"), output=Decimal("2.70")),
        "qwen3.8-flash": Price(
            openrouter_id="qwen/qwen3.8-flash",
            input=Decimal("0.15"),
            output=Decimal("0.47"),
            cache_read=Decimal("0.016"),
        ),
        "claude-haiku-5.5": Price(
            openrouter_id="anthropic/claude-haiku-5.5",
            input=Decimal("0.10"),
            output=Decimal("0.50"),
            cache_read=Decimal("0.01"),
        ),
    }


def _default_judge_alternatives() -> dict[str, Price]:
    return {
        "claude-sonnet-5.5": Price(
            openrouter_id="anthropic/claude-sonnet-5.5",
            input=Decimal("2"),
            output=Decimal("10"),
            cache_read=Decimal("0.10"),
        )
    }


class Costs(_Section):
    """What running the traces would cost on rented infrastructure (11.0 Costs in the dashboard). Everything here runs
    locally; these prices turn the token counts every bundle records into an API-equivalent bill."""

    prices: dict[str, Price] = Field(default_factory=_default_prices)
    """Keyed by our agent model (the header's served model) or our decision endpoint (clef, clef_flash). Clef is our
    own model with no list price, so it is priced as the API judge named here. A model with no price costs nothing on
    the page and is listed as unpriced."""
    prices_as_of: str = Field(default="2026-10-15", pattern=r"^\d{4}-\d{2}-\d{2}$")
    """The day the prices above were copied from OpenRouter."""
    alternatives: dict[str, Price] = Field(default_factory=_default_alternatives)
    """Cheaper agent models the what-if table prices the same tokens at."""
    judge_alternatives: dict[str, Price] = Field(default_factory=_default_judge_alternatives)
    """Other API judges the Clef table prices the same calls at."""
    snapshot_dir: RepoPath = Path("var/costs")
    """Where `carttrace dashboard prices --refresh` writes dated OpenRouter price snapshots; the newest overrides the prices
    above for every id it lists."""
    clef_output_tokens: int = Field(default=5, ge=0)
    """A Clef call records one token total; this many of them are counted as output (a yes or no answer), the rest as
    input."""
    cache_hit_share: float = Field(default=0.8, ge=0.0, le=1.0)
    """For the prompt-caching what-if: the share of prompt tokens a provider's prefix cache would serve. An agent call
    repeats the whole conversation so far, so most of each prompt was sent before."""
    gpu_name: str = "H100 80GB"
    gpu_usd_per_hour: Decimal = Field(default=Decimal("2.49"), ge=0)
    """On-demand rental price of one GPU, for the in-house what-if."""
    gpu_concurrency: int = Field(default=32, ge=1)
    """Agent requests one GPU serves at once (the load-tested cap, data/ops/capacity.json): summed request time over
    this is GPU time."""


class Release(_Section):
    """The public trace release (carttrace.release, `carttrace release build|check`). Only twins cut from catalogs whose licence
    allows publication are ever released (D10, D25, D47); everything here narrows that, nothing widens it."""

    name: str = Field(default="carttrace-traces", pattern=r"^[a-z][a-z0-9-]{1,63}$")
    """The dataset's name. Its canary GUID is derived from it, so changing it changes the canary."""
    pretty_name: str = "CartTrace agent shopping traces"
    repo_id: str = "ORG/carttrace-traces"
    """The Hugging Face repository the project uploads to, for the card's quick start. A placeholder until chosen."""
    catalogs: list[str] = Field(default_factory=lambda: ["abo", "lazada", "open", "wands"])
    """Catalog sources a released twin may be cut from (open: the open product pools of abo, lazada and wands, D91;
    lazada since D90). A bundle from any other source is refused, whatever its flags say: Open Food Facts is
    share-alike, replicas and live stores are never released."""
    exclude_store_domains: list[str] = Field(default_factory=lambda: ["amazon.com", "wayfair.com", "lazada.com.ph"])
    """Real stores whose twins are never released: the catalogs' own publishers. Their catalog's text names them
    ("visit amazon.com/amazonessentials"), so a pseudonym standing in for that domain would say which store it is."""
    judge_version: str | None = "judge-v9"
    """The judge version whose cards label the release (the analytics frames' version); None uses judge.version. A
    bundle without that card is left out and counted, never labelled by another version."""
    category_fit: bool = True
    """Release only twins whose task products fit the store category they stand in for (carttrace.release.fit)."""
    category_fit_depth: int = Field(default=2, ge=1, le=6)
    """How many levels of the catalog's own category path decide a product's home category."""
    exclude_task_authors: list[str] = Field(default_factory=list)
    """Task authors whose episodes are left out (`sonnet`: model-written openings and specs), pending legal review."""
    blocked_strings: list[str] = Field(default_factory=lambda: ["jews.com"])
    """Any bundle holding one of these, in any file, is left out whole."""
    cross_store_share: float = Field(default=0.10, ge=0.0, lt=1.0)
    """Share of the hash space of pseudonymous stores held out whole as test_cross_store."""
    cross_template_share: float = Field(default=0.10, ge=0.0, lt=1.0)
    """Share of the hash space of task templates, among the remaining stores, held out whole as test_cross_template."""
    validation_share: float = Field(default=0.05, ge=0.0, lt=1.0)
    """Share of the hash space of task ids, among the rest, that is validation; the remainder is train."""
    split_seed: str = "split-v1"
    """Keys every split hash. Changing it reshuffles every split, so it changes only with a new major release."""
    sample_sessions: int = Field(default=1500, ge=0)
    """Train episodes written as Session Trace Simple Format files under traces_sample/."""
    shard_bytes: int = Field(default=256_000_000, ge=1_000_000)
    """A parquet shard is closed once it reaches this size on disk (the Hub suggests 100 to 500 MB)."""
    row_group_bytes: int = Field(default=64_000_000, ge=100_000, le=100_000_000)
    """Uncompressed bytes buffered per row group, at most 100 MB so the dataset viewer can page it."""
    wire_row_group_bytes: int = Field(default=16_000_000, ge=100_000, le=100_000_000)
    """The same for the wire config, whose bodies are large: small row groups keep a reader's memory low."""
    workers: int = Field(default=16, ge=1, le=128)
    """Processes that read and clean bundles in parallel; the CPUs are shared, so keep it modest."""
    out_dir: RepoPath = Path("var/release")
    """Releases are written under <out_dir>/<version>/ unless --out names another directory."""
    private_dir: RepoPath = Path("var/release-private")
    """Where the pseudonym map (real store id and domain to pseudonym) goes, as <private_dir>/<version>/. It must
    never be inside a release directory; build and check both refuse that."""


class ConfigFile(_Section):
    """config.json exactly. `carttrace config schema` writes config.schema.json from this model."""

    schema_ref: str | None = Field(default=None, alias="$schema")
    """Path of the generated schema, for editors."""
    paths: Paths
    components: Components
    storage: Storage
    endpoints: dict[str, Endpoint]
    """Model servers by name. Names are lowercase identifiers so CARTTRACE__ENDPOINTS__<NAME>__<KEY> can address them."""
    gpu: Gpu = Field(default_factory=Gpu)
    roles: dict[str, str]
    """Role to endpoint name. A role with no route refuses to run instead of borrowing another role's model."""
    live: Live
    collect: Collect
    decide: Decide
    ops: Ops
    media: Media
    stores: Stores
    trace: Trace = Field(default_factory=Trace)
    sandbox: Sandbox = Field(default_factory=Sandbox)
    agents: Agents = Field(default_factory=Agents)
    judge: Judge = Field(default_factory=Judge)
    bench: Bench = Field(default_factory=Bench)
    autopilot: Autopilot = Field(default_factory=Autopilot)
    dashboard: Dashboard = Field(default_factory=Dashboard)
    costs: Costs = Field(default_factory=Costs)
    release: Release = Field(default_factory=Release)

    @model_validator(mode="after")
    def _routes_resolve(self) -> ConfigFile:
        bad_endpoints = sorted(n for n in self.endpoints if not _ENDPOINT_NAME.match(n))
        if bad_endpoints:
            raise ValueError(f"endpoint names must be lowercase identifiers: {bad_endpoints}")
        follows = self.collect.concurrency_follows
        if follows is not None and follows not in self.endpoints:
            raise ValueError(f"collect.concurrency_follows names {follows!r}, which is not in endpoints")
        bad_roles = sorted(r for r in self.roles if not _ROLE_NAME.match(r))
        if bad_roles:
            raise ValueError(f"role names must be dotted lowercase identifiers: {bad_roles}")
        missing = sorted(f"{r} -> {e}" for r, e in self.roles.items() if e not in self.endpoints)
        if missing:
            raise ValueError(f"roles name endpoints that are not in endpoints: {missing}")
        if FALLBACK_ROLE in self.roles:
            raise ValueError(
                f"roles.{FALLBACK_ROLE} is no longer read (D61): the fallback decision stage asks "
                "decide.fallback.endpoint; remove the role and set that instead"
            )
        if CLAUDE_CLI in self.endpoints:
            raise ValueError(f"no endpoint may be named {CLAUDE_CLI}: decide.fallback.endpoint uses it for the CLI")
        fallback = self.decide.fallback.endpoint
        if fallback is not None and fallback != CLAUDE_CLI:
            endpoint = self.endpoints.get(fallback)
            if endpoint is None:
                raise ValueError(
                    f"decide.fallback.endpoint names {fallback!r}, which is not in endpoints; name one of "
                    f"{sorted(self.endpoints)} or {CLAUDE_CLI}"
                )
            if endpoint.kind != "chat":
                raise ValueError(f"decide.fallback.endpoint {fallback!r} is a {endpoint.kind} endpoint; it needs chat")
        return self


class Settings(ConfigFile):
    """The effective settings: config.json, then config.local.json, then the environment, validated and frozen.

    Treat it as read-only; `carttrace.config.settings()` hands the same object to every caller in the process.
    """

    home: Path
    """The repository root every relative path resolved against."""

    def role(self, name: str) -> ModelEndpoint:
        """The route for a role, with `role` set to the role's name. Raises ConfigError naming what to set. The
        fallback decision role resolves to decide.fallback.endpoint (D61)."""
        if name == FALLBACK_ROLE:
            return self._fallback_route()
        endpoint_name = self.roles.get(name)
        if endpoint_name is None:
            env = "CARTTRACE__ROLES__" + name.upper().replace(".", "_")
            raise ConfigError(
                f"no model route for role {name!r}",
                hint=f'add "{name}": "<endpoint>" to roles in config.json or config.local.json, '
                f"or set {env}=<endpoint>",
            )
        endpoint = self.endpoint(endpoint_name)
        return ModelEndpoint(role=name, name=endpoint_name, **endpoint.model_dump())

    def _fallback_route(self) -> ModelEndpoint:
        name = self.decide.fallback.endpoint
        if name is None or name == CLAUDE_CLI:
            raise ConfigError(
                f"role {FALLBACK_ROLE} has no HTTP endpoint: decide.fallback.endpoint is {name!r}",
                hint="set decide.fallback.endpoint to a chat endpoint that is not the agent's, or leave fallback out "
                "of decide.order; the claude_cli fallback is asked through its own scorer",
            )
        return ModelEndpoint(role=FALLBACK_ROLE, name=name, **self.endpoint(name).model_dump())

    def cap_now(self, endpoint: Endpoint, now: datetime) -> int:
        """The endpoint's cap at `now` (timezone-aware), by its schedule in gpu.timezone."""
        return endpoint.cap_at(self.gpu.local(now))

    def collect_target(self, now: datetime) -> int:
        """Sandbox episodes the collector keeps in flight at `now`: collect.target_concurrency, or the followed
        endpoint's cap times collect.concurrency_per_slot."""
        follows = self.collect.concurrency_follows
        followed = self.endpoints.get(follows) if follows is not None else None
        cap = self.cap_now(followed, now) if followed is not None else None
        return self.collect.target_for(cap)

    def endpoint(self, name: str) -> Endpoint:
        try:
            return self.endpoints[name]
        except KeyError:
            raise ConfigError(
                f"no endpoint {name!r} in config.json endpoints", hint=f"choose one of {sorted(self.endpoints)}"
            ) from None

    def component(self, kind: str) -> str:
        """The implementation name config.json chooses for a component kind, e.g. component("inference.chat")."""
        return self.components.name_for(kind)

    def secret_names(self) -> dict[str, str]:
        """Each config location that names a secret, mapped to the variable name it holds."""
        found: dict[str, str] = {}
        _collect_env_names(self, "", found)
        return dict(sorted(found.items()))

    # Shortcuts with the names of the skeleton contract, so early callers keep working.
    @property
    def data_dir(self) -> Path:
        return self.paths.data

    @property
    def vendor_dir(self) -> Path:
        return self.paths.vendor

    @property
    def var_dir(self) -> Path:
        return self.paths.var

    @property
    def storage_url(self) -> str:
        return self.storage.url

    @property
    def live_enabled(self) -> bool:
        return self.live.enabled

    @property
    def control_file(self) -> Path:
        return self.live.control_file

    @property
    def user_agent(self) -> str:
        return self.live.user_agent

    @property
    def agent_profile_url(self) -> str | None:
        return self.live.agent_profile_url

    @property
    def slack_webhook_env(self) -> str:
        return self.ops.slack_webhook_env

    @property
    def s3_media_prefix(self) -> str | None:
        return self.storage.media_url


def _collect_env_names(model: BaseModel, prefix: str, found: dict[str, str]) -> None:
    for field_name in type(model).model_fields:
        value = getattr(model, field_name)
        path = f"{prefix}{field_name}"
        if isinstance(value, BaseModel):
            _collect_env_names(value, f"{path}.", found)
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, BaseModel):
                    _collect_env_names(item, f"{path}.{key}.", found)
        elif field_name.endswith("_env") and isinstance(value, str):
            found[path] = value


def json_schema() -> dict[str, Any]:
    """The JSON Schema of config.json, generated from `ConfigFile`."""
    schema = ConfigFile.model_json_schema(by_alias=True, mode="validation")
    schema["$schema"] = JSON_SCHEMA_DIALECT
    schema["title"] = "CartTrace config.json"
    schema["description"] = (
        "Generated by `carttrace config schema` from carttrace.config.schema.ConfigFile. Do not edit by hand; per-machine "
        "overrides go in config.local.json and CARTTRACE__SECTION__KEY variables."
    )
    return schema


def json_schema_text() -> str:
    """The schema as the bytes written to config.schema.json: canonical key order, indented, newline-terminated."""
    return pretty(json_schema())

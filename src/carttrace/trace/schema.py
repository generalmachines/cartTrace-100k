"""The trace bundle, `mc.trace.v1`: the contract between collection and everything that reads it.

Why a versioned schema. Collection is generic: it records what happened and computes no score. The judge, the store
benchmark, agent evals and training exports all read the same bundles, and any of them may be re-run later over
traces already collected. That only works if the bundle is a stable, versioned contract that consumers depend on,
never the other way round. A consumer writes next to a bundle under its own versioned name; it never edits raw files.

Bundle layout (one directory per episode):
    episode.json   EpisodeHeader: what was run, against what, under which versions and models
    turns.jsonl    Turn records: the conversation and every tool call and result, in order
    wire.jsonl     WireRecord: raw protocol exchanges, redacted at write time
    state.jsonl    StateSnapshot: cart and state snapshots as the store or the sandbox reported them
    decisions.jsonl  decision-engine records made during the episode (supervision, shopper mapping, triage): each line
                   is the decision dict carttrace.decide wrote, plus the writer-owned `seq` and `t_ms`
    events.jsonl   SandboxEvent: the sandbox's own hash-chained event log, one record per line (sandbox bundles only;
                   optional, so a bundle without it, every live one and every one written before D33, stays valid)
    exposure.json  ExposureSheet (carttrace.trace.exposure, D93): what the store exposed per data type and surface, and what
                   checkout offered (sandbox bundles since D93 only; optional)
    outcome.json   Outcome: how the episode ended, decided by code; on a sandbox bundle its `state_digest` names the
                   sandbox's final state hash and the head of the events.jsonl chain
    bundle.json    BundleManifest: the sha256 of every file above, written last

Ordering. The writer owns `seq` and `t_ms` on every line of the four logs. `seq` is one counter per episode, shared by
turns, wire, state and decisions: it increases within each file, with gaps where another file took a number, and
merging the four files by seq gives the order things happened. `t_ms` is milliseconds since the writer opened, from
its clock. events.jsonl is not numbered by the writer: its `seq` is the sandbox's own command counter and its `idx` the
record's place in the sandbox's chain, so it is never merged into the timeline by seq.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from carttrace.protocols.profile import ProtocolProfile

SCHEMA = "mc.trace.v1"
SCHEMA_MINOR = 1
"""The minor version of mc.trace.v1 a new header names in `schema_minor` (D69). Minor versions only add optional
fields, so a reader of v1 reads every minor: 0 is every bundle written before D69 (the field is absent there), 1 adds
the agent definition (agent.json), call ids, finish reasons, per-call cost, turn origins, typed error classes, the
served model in the header and the kept product region of website pages."""
AGENT_SCHEMA = "mc.trace.agent.v1"

Mode = Literal["live", "sandbox", "playground"]
"""live: a store we do not own, read-only. sandbox: our in-process store. playground: a real store we own and run on
this machine (D44), which may be bought from with its offline gateways."""
Actor = Literal["shopper", "agent", "tool", "system"]
TurnKind = Literal["message", "tool_call", "tool_result", "note"]
ErrorType = Literal[
    "harness_blocked",  # our own safety layer refused the call below the agent; nothing was sent
    "invalid_arguments",  # the agent's arguments could not be read, or do not fit the tool's schema
    "unknown_tool",  # the agent called a tool no surface offers
    "tool_unavailable",  # the store lists the tool but will not run it for this caller
    "not_found",  # the thing asked for does not exist (a product, a cart, a page)
    "out_of_stock",  # the store refused a write because the item is sold out or short
    "payment_declined",  # a payment, credential or 3-D Secure step was declined or failed
    "needs_input",  # the store needs something before it goes on (buyer details, a choice, an approval)
    "unverified",  # the harness re-read a selection or an order and it did not hold
    "throttled",  # 429 or the throttle's wait ran out; never retried past budget
    "blocked",  # 401, 403, a bot wall, or robots.txt disallowing the path
    "gated",  # login, age, prescription or consultation required
    "timeout",  # the call ran out of time
    "network_error",  # DNS, TLS or a refused connection
    "protocol_error",  # the reply is not the protocol (bad JSON-RPC, wrong version, no index)
    "store_error",  # the store answered with a failure of its own (5xx, isError, ucp.status error)
    "other",  # a failure none of the above names; error_code keeps what the surface said
]
"""The closed set of tool-failure classes (D69), on every failed tool result as `error_type`. `error_code` keeps the
surface's own code verbatim (the judge, triage and the bench read it); `error_type` is the one a reader can group by
across surfaces and protocols. carttrace.trace.errors maps codes onto it."""
EndReason = Literal[
    "cart_ready",  # the agent reached the live boundary: a verified cart the shopper accepted
    "selection_ready",  # on a store's website (D31): the agent chose a variant; the harness re-read and judged it
    "order_placed",  # sandbox only: checkout completed and an order exists
    "refused_correctly",  # the agent declined because nothing satisfies the spec (decided later by the judge)
    "gave_up",  # the agent stopped without a cart and without a justified refusal
    "step_cap",  # hit the step limit
    "time_cap",  # hit the wall-clock limit
    "supervisor_stop",  # the decision engine judged the episode stuck or looping and stopped it
    "blocked",  # bot wall, robots disallow, or 401/403 from the store
    "throttled",  # repeated 429s exhausted the retry budget
    "no_protocol",  # the store exposes no agent protocol we can use
    "gated",  # age gate, consultation, login or prescription required
    "harness_error",  # our own bug or infrastructure failure; never counted against store or agent
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StoreRef(_Frozen):
    store_id: str
    """The queue id, e.g. "store-001", or a sandbox store id."""
    domain: str
    category: str
    platform: str | None = None
    market: str | None = None
    """Shopper delivery market for this episode, e.g. "US"."""


class TaskRef(_Frozen):
    task_id: str
    intent: Literal[
        "replenish", "specify", "compare", "discover", "express", "gift", "bundle", "budget_max", "substitute", "policy"
    ]  # fmt: skip
    """The five intents every task had, and generator 3's five (carttrace.agents.spec.ALL_INTENTS)."""
    category: str
    opening: str
    """The shopper's first message."""
    fixture: list[str] = Field(default_factory=list)
    """Product or variant ids the task was written around."""
    hidden_spec: dict[str, Any] = Field(default_factory=dict)
    """Predicates the shopper wants, each tagged stated, on_ask or default. Judged later; never shown to the agent."""
    template: str = ""
    """Template id and version the task text came from."""
    intent_kind: str | None = None
    """A custom intent's free-form kind ("compare price per unit across pack sizes"), when a model wrote the task
    from the store's brief (carttrace.agents.intents_custom); None for a template task. Left out of the JSON when None."""
    decisive_kind: str | None = None
    """The data kind of the exposure taxonomy the task turns on (price.unit, delivery_estimate, allergens, ...),
    recorded on custom tasks; None for a template task, whose spec's decisive detail says. Left out when None."""
    needs_replica: bool = False
    """Only the replica can run it (a custom intent with faults or a checkout-stage requirement); a manifest plans
    it on sandbox rows only, never on a live row. Left out of the JSON when False."""

    @model_serializer(mode="wrap")
    def _without_unset_kinds(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """A task with no custom kinds serializes as tasks did before the kinds existed, so every task id and
        trace header already written keeps its bytes."""
        data: dict[str, Any] = handler(self)
        for name in ("intent_kind", "decisive_kind"):
            if data.get(name) is None:
                data.pop(name, None)
        if data.get("needs_replica") is False:
            data.pop("needs_replica", None)
        return data


class Condition(_Frozen):
    name: str
    """baseline, cross_border, network_alt, version_variant, detail_hidden, ..."""
    params: dict[str, Any] = Field(default_factory=dict)


class SurfaceRef(_Frozen):
    kind: str
    mode: Mode
    url: str
    version: str


class ModelRef(_Frozen):
    role: str
    requested: str
    served: str | None = None
    endpoint: str
    params: dict[str, Any] = Field(default_factory=dict)


class CallCost(_Frozen):
    """What one model call cost (D69), on the call's first turn. Money is an integer: millionths of a US dollar."""

    usd_micros: int | None = None
    """Millionths of a USD the call cost: 0 on our own GPU servers, the backend's figure where it reports one, None
    where it is not known (a paid API that reports nothing)."""
    gpu_ms: int | None = None
    """The GPU-time proxy on our own servers: milliseconds the call held its GPU slot (the call's model time, slot
    waits excluded). Shared GPUs batch requests, so it bounds the call's share from above. None off our GPUs."""
    basis: Literal["own_gpu", "reported", "unpriced"] = "own_gpu"


class TurnOrigin(_Frozen):
    """What produced a turn that no model call of the agent made (D69): the shopper's rule or branch, the version of
    the code and phrases behind it, and for model-voiced text the voice model and its prompt's sha256, so who wrote
    the words (task_author) is recorded rather than inferred."""

    rule: str
    """opening, keyword_field, decided_field, nothing_asked, other, choice, consent."""
    version: str | None = None
    """The shopper's version (carttrace.agents.shopper SHOPPER_CODE plus the phrases version) for a reply; for the opening,
    the generator that wrote the task."""
    author: Literal["template", "model", "rules"] | None = None
    """Who wrote the words: the template, a model (an accepted voice rewrite, a custom intent), or the shopper's
    rules (every reply)."""
    voice_model: str | None = None
    """The model that voiced the text, when a model did."""
    prompt_sha256: str | None = None
    """sha256 of the prompt the voice model was sent, when the task recorded it (tasks voiced before D69 did not)."""
    wording: Literal["model", "fallback"] | None = None
    """On a shopper reply whose facts the rules decided (D118): `model` when the shopper's model worded them and the
    words passed the code's fact check, `fallback` when the rules' own line was sent instead. None when the shopper
    words nothing by model (`agents.shopper.wording` rules, and every reply before D118)."""

    @model_serializer(mode="wrap")
    def _without_unset(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        return {k: v for k, v in data.items() if v is not None}


class AgentDefinition(_Frozen):
    """agent.json (D69): what the agent was given, once per episode, in the spirit of ATIF's `agent` block: the system
    prompt as sent, the prompt file it came from with its sha256, and every tool definition exactly as the model
    received it, the store's surfaces, the harness's tools and `done` alike (a website episode's tools are recorded
    nowhere else). Written by the agent once it has listed its tools; absent on bundles written before D69."""

    schema_: Literal["mc.trace.agent.v1"] = Field(default="mc.trace.agent.v1", alias="schema")
    name: str
    """The agent implementation (qwen-protocol, reference)."""
    role: str = "agent"
    system_prompt: str
    """The system prompt text as sent, context included, redacted at write time like every record."""
    system_prompt_sha256: str
    """sha256 of `system_prompt` as the agent sent it (before redaction)."""
    prompt_file: str | None = None
    """The versioned prompt file the prompt came from, relative to paths.data (prompts/agent-v3.md)."""
    prompt_sha256: str | None = None
    """sha256 of the prompt file's text as read (stripped), so a prompt edit shows as a new sha."""
    tool_definitions: list[dict[str, Any]] = Field(default_factory=list)
    """Every tool as sent to the model, in the order sent (OpenAI function format)."""
    tool_surfaces: dict[str, str] = Field(default_factory=dict)
    """Tool name as exposed to the model to what serves it: a surface kind, `harness` or `done`."""
    params: dict[str, Any] = Field(default_factory=dict)
    """temperature, max_tokens, the episode's seed base."""


class Rights(_Frozen):
    """What this episode's data may be used for, decided before collection and carried with the bundle.

    Why on every bundle. Shopify's API terms (updated 2026-10-07, section 2.3.24) bar using anything derived from its
    APIs (which include its MCP servers) or from a merchant store to train or fine-tune models, except with Shopify's
    written consent or that merchant's consent; inference-only use is not restricted. Clauses 2.3.4 and 6.2.5 bear on
    benchmarking. Store-selection sources (Channel3, Store Leads) carry their own limits. The export and publication
    steps read these fields and refuse what is not allowed; nothing defaults to allowed.
    """

    store_source: str = "team_sheet"
    """Where the store was selected from: team_sheet, httparchive, channel3, manual, sandbox."""
    platform_terms: Literal["shopify_api", "woocommerce", "custom", "sandbox", "unknown"] = "unknown"
    merchant_consent: Literal["none", "requested", "granted", "denied"] = "none"
    training_allowed: bool = False
    """True only with the platform's written consent or the merchant's (or for sandbox-only content we own)."""
    publication_allowed: bool = False
    """True only after the publication gate: merchant permission and legal review."""
    basis: str = ""
    """Reference to the consent or ruling that set the flags above."""


class SoldOutRate(_Frozen):
    """The chance a replica answers a variant's cart write "sold out" while its catalog lists it available, and the
    live evidence behind it (D52, carttrace.sandbox.catalogs.sold_out). The same block names the rate on a world's recipe,
    in its pack's policy and on an episode's header, so analytics can stratify by it."""

    record: str
    """The store's sold-out record (so-<sha12>) the rate was read from."""
    market: str
    """The market of the world the rate was chosen for (US, GB, ...)."""
    rate_bp: int = Field(ge=0, le=10_000)
    """Basis points: the share of cart writes the evidence below had refused as sold out."""
    source: Literal["store", "category", "overall"]
    """Whose record the rate is: the store's own, its category's stores, or every live store."""
    statistic: Literal["own", "median_of_stores"] = "own"
    """`own`: the share of the store's own writes refused; `median_of_stores`: the median of the store rates of its
    category (or of every store), over stores with enough writes of their own."""
    stores: int | None = None
    """How many stores a median is over; None for the store's own rate."""
    stratum: str
    """Which sessions the counts pool, e.g. `store-092 US`, `store-092 all markets`, `category bags-accessories`."""
    writes: int = Field(ge=0)
    """Cart writes (create or update with lines) in the evidence; for a median, those stores' writes pooled."""
    refused: int = Field(ge=0)
    """Of them, writes the store answered with a sold-out message."""
    sessions: int = Field(ge=0)
    """Live sessions with at least one cart write in the evidence."""
    refused_sessions: int = Field(ge=0)
    """Of them, sessions with at least one write refused as sold out."""


RailName = Literal["ucp_psp", "acp_delegated", "ap2_mandates", "mpp", "shop_pay"]
"""The payment rails a sandbox order row pays over (D66): UCP checkout with a PSP handler token, ACP delegated payment,
AP2 intent, cart and payment mandates, MPP's HTTP 402 challenge, credential and receipt, or a UCP checkout paid with
a Shop Pay single-use token (D73)."""
CheckoutFaultKind = Literal["three_ds", "payment_decline", "reprice", "stockout", "unshippable"]
"""The checkout-time faults an order row may draw (D66): a 3-D Secure challenge, a declined payment, a price change at
checkout, stock gone at checkout, and no shipping to the buyer's address."""
DeclineClass = Literal["buyer_fixable", "retryable"]
"""A declined payment the buyer must fix (another card, more funds: every completion is declined), or one a retry with
the same credential can clear (the decline fires once)."""


class CheckoutFault(_Frozen):
    """One checkout-time fault an order row's episode drew (D66), as its world's schedule holds it."""

    kind: CheckoutFaultKind
    ids: list[str]
    """The fault ids in the episode's pack schedule (`checkout-...`): one per witness variant for a price change or a
    stockout, else one. The store's event log names the one that fired (`fault.fired`, `fault_id`)."""
    rate_bp: int = Field(ge=0, le=10_000)
    """The kind's rate, in basis points of order rows, from the manifest's checkout fault rates."""
    assumed: bool = True
    """The rate is a design parameter, not a measurement: live never pays, so no live row can give one."""
    source: str = ""
    """The public source the rate is set against, or why there is none."""
    decline_class: DeclineClass | None = None
    """On a payment_decline: buyer_fixable or retryable."""
    code: str | None = None
    """On a payment_decline: the issuer's decline code (carttrace.sandbox.core.psp.DECLINE_CODES)."""
    variants: list[str] = Field(default_factory=list)
    """On a reprice or stockout: the task's witness variants it is set on, so it fires whichever the agent buys."""
    raise_bp: int | None = None
    """On a reprice: how much each variant's unit price rises, in basis points of it; the store's event log has each
    new price (`inventory.repriced`)."""
    countries: list[str] = Field(default_factory=list)
    """On unshippable: the destination countries the store stops shipping to (the buyer's)."""


class CheckoutFaultDraw(_Frozen):
    """What an order row drew among the checkout-time faults (D66): every kind's rate, the seeded draw, and the fault
    it chose (none when the draw fell past every rate, the common case)."""

    rates: dict[str, int]
    """Kind to its rate in basis points, as the manifest named them."""
    draw_bp: int = Field(ge=0, lt=10_000)
    """The episode's seeded draw in [0, 10000); the kinds' rates are laid end to end in `rates` order."""
    faults: list[CheckoutFault] = Field(default_factory=list)
    """At most one fault; empty when none was drawn, or the drawn kind cannot apply (a price change on a task with no
    witness)."""
    skipped: str | None = None
    """Why a drawn kind was not applied, when it was not."""


PaymentFaultKind = Literal["decline", "3ds_fail", "3ds_abandon", "no_instrument", "checkout_fault"]
"""How an order row's payment is designed to fail (D80): the bank declines the persona's instrument (an expired Shop
Pay token included), the bank fails 3-D Secure, the shopper abandons the challenge, the wallet holds nothing the
rail takes, or the checkout fault the row drew (D66) declines or challenges the payment."""
PaymentFaultSource = Literal["persona", "plan", "checkout_fault", "unplanned"]
"""Where a designed payment fault comes from (D80): the persona's wallet (its token is built to fail), the plan (a
no_instrument fault row the planner paired on purpose), the row's checkout fault draw, or `unplanned`: the wallet
cannot pay the store's handlers though the planner expected it to, which is a harness bug to look at, not a fault."""


class WalletInstrument(_Frozen):
    """One instrument of the persona's wallet (D80): its kind and the sandbox PSP's opaque test token."""

    kind: str
    """card, google_pay, shop_pay, or offline (a pay-later gateway)."""
    token: str


class PaymentPlan(_Frozen):
    """What an order row's payment is made of (D80): the rail, what the store takes on it, the wallet, and the
    instrument the wallet pays with. Present on every sandbox order row since D80."""

    rail: str | None = None
    """The payment rail (D66), None on an order row of a plan that crosses no rails (UCP checkout)."""
    store_accepts: list[str]
    """Instrument kinds the row's checkout takes: the rail's credential (ACP a card; AP2 and MPP a card or Google Pay;
    Shop Pay its own), or on UCP what the world's payment handlers take."""
    handlers: list[str] = Field(default_factory=list)
    """The payment handler namespaces the row's world serves, sorted."""
    wallet: list[WalletInstrument]
    """The persona's instruments, in its order of preference."""
    pays_with: WalletInstrument | None = None
    """The instrument the wallet pays with; None when it holds nothing the store takes (no_instrument)."""


class PaymentFault(_Frozen):
    """How an order row's payment is designed to fail (D80), so a reader tells an intended failure from the agent's.
    Absent (null) on a row built to pay."""

    kind: PaymentFaultKind
    source: PaymentFaultSource
    planned: bool
    """The manifest planned the row as a fault row (its `payment_fault`), as the payment fault share asks."""
    token: str | None = None
    """The opaque test token built to fail (decline, 3ds_fail, 3ds_abandon); None otherwise."""
    code: str | None = None
    """The decline code (insufficient_funds, do_not_honor, shop_token_expired), or the checkout fault's kind and
    class (payment_decline:buyer_fixable, three_ds)."""
    recoverable: bool = False
    """The agent can still complete the order: a retryable decline retried with renewed consent, or a checkout fault's
    3-D Secure challenge the persona completes."""


class FaultFiring(_Frozen):
    """Whether one drawn checkout fault fired, read from the store's own event log (D66). A fault that never fired is
    recognisable here (`fired` false), so a judge can void the row as fault_unfired."""

    kind: CheckoutFaultKind
    ids: list[str]
    fired: bool
    fault_id: str | None = None
    """The id of the first knob that fired."""
    event_seq: int | None = None
    """The sandbox's command counter of its first `fault.fired` record (events.jsonl `seq`)."""
    event_idx: int | None = None
    """The record's place in events.jsonl (`idx`)."""
    times: int = 0
    """How many `fault.fired` records it left."""
    decline_class: DeclineClass | None = None


ExposureDesign = Literal["baseline", "leave_one_out", "random"]
"""How an exposure mask was chosen (D63): the store's full exposure, one of its data types withheld, or a seeded draw
of several."""


class ExposureCount(_Frozen):
    """How many of a world's products served one data type before its exposure cut, and after it."""

    before: int = Field(ge=0)
    after: int = Field(ge=0)


class ExposureMask(_Frozen):
    """Which data types a sandbox world serves and which it withholds (D63, carttrace.collect.exposure), with the evidence
    that each withheld type was there to withhold and is gone from what the world serves. The same block is on the
    world's recipe and on the episode's header, so analytics read the pattern of what an agent was given."""

    design: ExposureDesign
    mask: str
    """xm-<sha12> of the withheld type ids: one id per set, the same on every store."""
    withheld: list[str] = Field(default_factory=list)
    """Type ids the world withholds: each was served before the cut (`counts[t].before` > 0) and is not after it."""
    served: list[str] = Field(default_factory=list)
    """Type ids the world serves: some product carries each after the cut."""
    absent: list[str] = Field(default_factory=list)
    """Type ids of the table the store's data does not show in this world, so there is nothing to withhold."""
    types: str
    """The version of the type table the ids are from (carttrace.collect.exposure.TYPES_VERSION)."""
    products: int = Field(ge=0)
    """Products the world serves."""
    counts: dict[str, ExposureCount] = Field(default_factory=dict)
    """Per withheld and served type: products that carried it before the cut and after it."""
    collateral: list[str] = Field(default_factory=list)
    """Served types the cut thinned (fewer products carry them after it): text that stated a withheld type also
    stated these."""


class StoreErrorRate(_Frozen):
    """How often one of a store's tools answered with an error of the store's own making on live rows (D70), the rate
    a replica of the store draws that error at, and the live counts behind it."""

    kind: Literal["throttled", "server_error"]
    """`throttled`: Shopify's isError reply "Throttled" (a cart that was not made); `server_error`: an HTTP 5xx."""
    tool: str
    rate_bp: int = Field(ge=0, le=10_000)
    """Basis points of the tool's calls."""
    source: Literal["store", "overall"]
    """The store's own calls, or (for a tool the store was called too few times on) the median of every store's."""
    stores: int | None = None
    """How many stores an overall median is over; None for the store's own rate."""
    calls: int = Field(ge=0)
    faults: int = Field(ge=0)


class StoreReplies(_Frozen):
    """What a replica serves from its store's own live replies beyond its capture (D70): the payment handlers its
    cart replies declared and the rates of its own errors. Named on a world's recipe and on an episode's header."""

    record: str
    """The store's replies record (sr-<sha12>, carttrace.sandbox.catalogs.store_replies)."""
    handlers: list[str] = Field(default_factory=list)
    """The payment handler namespaces served, as the store's live cart replies declared them; empty when none."""
    handlers_from: str | None = None
    """The live reply they were read from: `<episode> seq <n>`."""
    errors: list[StoreErrorRate] = Field(default_factory=list)
    """Every rate above zero, by kind and tool."""


class EpisodeHeader(_Frozen):
    schema_: Literal["mc.trace.v1"] = Field(default="mc.trace.v1", alias="schema")
    schema_minor: int = Field(default=0, ge=0)
    """The minor version of mc.trace.v1 (SCHEMA_MINOR): 0, and absent from the file, on every bundle before D69."""
    episode_id: str
    run_id: str
    manifest_id: str
    mode: Mode
    started_at: str
    store: StoreRef
    task: TaskRef
    condition: Condition
    profile: ProtocolProfile
    surfaces: list[SurfaceRef] = Field(default_factory=list)
    models: list[ModelRef] = Field(default_factory=list)
    seed: int
    limits: dict[str, int] = Field(default_factory=dict)
    """step_cap, time_cap_s, max_tokens."""
    code_version: str
    """git describe of the repo that ran the episode, with -dirty when uncommitted."""
    envpack: str | None = None
    """Sandbox envpack id (env-<sha>) when the episode ran in or handed off to the sandbox."""
    rights: Rights = Field(default_factory=Rights)
    """Data-use limits for this episode. Exports and publication must check them."""
    pair_id: str | None = None
    """The manifest row's pair id when the episode is one half of a live and replica pair (carttrace.collect pair_replica):
    the same on both episodes, so the bench joins them by it. None on every other episode."""
    boundary: Literal["selection"] | None = None
    """Where the journey is meant to end when it is not a cart: "selection" on a store's website (D31), where the
    agent chooses a variant and nothing is carted. None on every other episode."""
    platform_pack: str | None = None
    """The platform pack a sandbox episode's store was served under (carttrace.sandbox.platforms, D42): its products and
    market under that platform's rules, so analytics compare agents across platforms on the same products. None: the
    store's own platform (`store.platform`), as every episode before the platform axis."""
    market_context: Literal["fill", "none"] | None = None
    """On a live episode (D43): `fill` when the harness put the row's market into the cart and catalog calls that
    left it out (carttrace.live.market, its FILLED wire records), `none` when the store saw only what the agent sent and
    priced us as any visitor from our network position. None on every other episode."""
    served_currency: str | None = None
    """On a live episode whose tasks were written from a price probe (carttrace.live.prices, D43): the currency the store
    priced us in before the round, which the task's budget and prices are written in. None otherwise."""
    sold_out: SoldOutRate | None = None
    """On a sandbox episode of a replica world (D52): the rate at which its cart writes answer "sold out" for
    variants the catalog lists available, with the live evidence behind it. None on every other episode."""
    rail: RailName | None = None
    """On a sandbox row of a plan that crosses payment rails (D66): the rail its order is paid over. None on every
    other episode."""
    rail_protocol: str | None = None
    """The rail's protocol family (ucp, acp, ap2, mpp)."""
    rail_version: str | None = None
    """The rail's protocol version the replica served (UCP release, ACP API-Version, AP2 release, MPP release)."""
    psp: Literal["stripe", "adyen", "braintree", "paypal"] | None = None
    """On a ucp_psp row of a plan that crosses payment brokers (D71): the broker behind the store's PSP, whose
    test-mode shapes the PSP's record carries. None on every other row."""
    adopted: bool | None = None
    """True when the real store does not speak the rail's protocol and the replica served its catalog through it as if
    the store had adopted it; false when the store speaks it. Rows with adopted true are scored for the agent on the
    rail and kept out of the store's own ledger."""
    surface_adopted: bool | None = None
    """On a row of a manifest that draws (surface, rail) cells (D89): true when the real store does not serve the row's
    surface itself (a UCP binding its capture did not record, or a rail's own surface it lacks) and the replica served
    it as adopted; false when the store serves it. None on every other episode."""
    checkout_faults: CheckoutFaultDraw | None = None
    """On a sandbox order row whose manifest names checkout fault rates (D66): every rate, the seeded draw and the
    fault it chose."""
    jsonld_types: list[str] | None = None
    """On a sandbox website episode (D67): the schema.org types the replica's pages serve in the JSON-LD its store's
    pages carried when captured (Product, Offer, AggregateRating, ...); empty when they carry none. None on every
    other episode."""
    exposure: ExposureMask | None = None
    """On a sandbox episode of an exposure row (D63): the data types its world served and withheld, and how the mask
    was chosen. None on every other episode."""

    shopper_version: str | None = None
    """The deterministic shopper's code and phrases version (D69), e.g. `mc.agents.shopper/1+shopper-v2`."""
    generator_version: str | None = None
    """The generator that wrote the task's hidden spec (its `generator`, e.g. `mc.agents.tasks/4`) (D69)."""
    judge_code_version: str | None = None
    """The judge version configured when the episode ran (judge.version, e.g. `judge-v10`) (D69): the judge this
    bundle was collected for; a card names the one that actually judged it."""
    store_replies: StoreReplies | None = None
    """On a sandbox episode of a replica world (D70): the payment handlers and the error rates its store's live replies
    showed, which the replica serves. None on every other episode."""
    payment: PaymentPlan | None = None
    """On a sandbox order row since D80: the rail, what the store takes, the persona's wallet and the instrument it
    pays with. None on every other episode."""
    payment_fault: PaymentFault | None = None
    """On a sandbox order row since D80: how its payment is designed to fail, written as null when it is built to pay
    (a header with `payment` always carries the key). Absent on every other episode."""
    sandbox_version: str | None = None
    """D96: "v1" on a row built under the locked sandbox v1 spec; absent on every other episode."""
    sandbox_spec: str | None = None
    """D96: the locked v1 spec's digest (v1s-<sha12>, data/sandbox/v1-spec.lock.json) the row was built under."""

    @model_serializer(mode="wrap")
    def _without_unset_pair(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """A header of an unpaired episode serializes as headers did before pairs, so recorded bundles' bytes hold;
        so does one with no boundary, platform pack, market context, served currency, sold-out rate, rail, checkout
        fault draw, JSON-LD types, exposure mask or store replies of its own, and one written before D69 (minor 0, no
        versions)."""
        data: dict[str, Any] = handler(self)
        paid = data.get("payment") is not None
        for name in _HEADER_OPTIONAL:
            if data.get(name) is None and not (paid and name == "payment_fault"):
                data.pop(name, None)
        if data.get("schema_minor") == 0:
            data.pop("schema_minor", None)
        return data


_HEADER_OPTIONAL = (
    "pair_id",
    "boundary",
    "platform_pack",
    "market_context",
    "served_currency",
    "sold_out",
    "rail",
    "rail_protocol",
    "rail_version",
    "adopted",
    "surface_adopted",
    "psp",
    "checkout_faults",
    "jsonld_types",
    "exposure",
    "store_replies",
    "payment",
    "payment_fault",
    "shopper_version",
    "generator_version",
    "judge_code_version",
    "sandbox_version",
    "sandbox_spec",
)


class Turn(_Frozen):
    seq: int
    """The episode's one counter, shared with wire, state and decisions: increasing in this file, with gaps."""
    t_ms: int
    """Milliseconds since the writer opened."""
    actor: Actor
    kind: TurnKind
    text: str | None = None
    tool: str | None = None
    surface: str | None = None
    arguments: dict[str, Any] | None = None
    result: Any = None
    is_error: bool = False
    error_code: str | None = None
    model: str | None = None
    """Served model that produced this turn, for agent and shopper turns."""
    usage: dict[str, int] | None = None
    latency_ms: int | None = None
    seed: int | None = None
    """The sampling seed the model call that produced this turn sent: since D69 on every turn the call produced,
    before it on the call's first turn only. The two halves of a live and replica pair send the same seed at the same
    call index, so their first calls differ only by what the store answered. None on every other turn, and left out
    of the record then."""
    redactions: list[str] = Field(default_factory=list)
    """Names of the redaction rules that changed this record. Bundles written before the field read as []."""
    call_id: str | None = None
    """On a tool_call turn (D69): the call's id in this episode, unique and deterministic (`call-<seq>`), which its
    result cites in `tool_call_id` and its wire exchanges in `source_call_id`."""
    tool_call_id: str | None = None
    """On a tool_result turn (D69): the `call_id` of the call it answers, so parallel calls stay unambiguous."""
    finish_reason: str | None = None
    """On every turn an agent model call produced (D69): why the model stopped (stop, tool_calls, length)."""
    cost: CallCost | None = None
    """On a model call's first turn (D69), beside its usage: what the call cost."""
    error_type: ErrorType | None = None
    """On a failed tool result (D69): the closed class of the failure; `error_code` keeps the surface's own code."""
    error_message: str | None = None
    """On a failed tool result (D69): the failure's message as text, at most 300 characters."""
    origin: TurnOrigin | None = None
    """On a shopper turn (D69): the rule or branch that produced it and its version, and for the opening who wrote
    it."""

    @model_serializer(mode="wrap")
    def _without_unset_seed(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """A turn with no seed serializes as turns did before seeds were recorded, so recorded bundles' bytes hold;
        so does one without any of the D69 fields."""
        data: dict[str, Any] = handler(self)
        for name in _TURN_OPTIONAL:
            if data.get(name) is None:
                data.pop(name, None)
        return data


_TURN_OPTIONAL = (
    "seed",
    "call_id",
    "tool_call_id",
    "finish_reason",
    "cost",
    "error_type",
    "error_message",
    "origin",
)


class WireRecord(_Frozen):
    seq: int
    """The episode's one counter, shared with turns, state and decisions: increasing in this file, with gaps."""
    t_ms: int
    """Milliseconds since the writer opened."""
    surface: str
    protocol: str
    version: str
    method: str
    url: str
    request_headers: dict[str, str] = Field(default_factory=dict)
    request_body: Any = None
    status: int | None = None
    response_headers: dict[str, str] = Field(default_factory=dict)
    response_body: Any = None
    duration_ms: int = 0
    error: str | None = None
    redactions: list[str] = Field(default_factory=list)
    """Names of the redaction rules that changed this record."""
    source_call_id: str | None = None
    """The `call_id` of the agent's tool call this exchange was made for (D69); None for the runner's own exchanges
    (discovery, read-backs) and on every record before D69, and left out of the line then."""

    @model_serializer(mode="wrap")
    def _without_unset_source(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if data.get("source_call_id") is None:
            data.pop("source_call_id", None)
        return data


class CartLine(_Frozen):
    product_id: str | None = None
    variant_id: str
    title: str = ""
    quantity: int
    unit_price: int
    """Minor units in `currency`."""
    currency: str
    options: dict[str, str] = Field(default_factory=dict)
    selling_plan: str | None = None


class Cart(_Frozen):
    cart_id: str | None = None
    lines: list[CartLine] = Field(default_factory=list)
    subtotal: int | None = None
    total: int | None = None
    currency: str | None = None


class StateSnapshot(_Frozen):
    seq: int
    """The episode's one counter, shared with turns, wire and decisions: increasing in this file, with gaps."""
    t_ms: int
    """Milliseconds since the writer opened."""
    source: str
    """"live:get_cart", "live:cart_reply", "sandbox:state", ..."""
    cart: Cart | None = None
    raw_ref: int | None = None
    """seq of the WireRecord the snapshot was read from."""
    redactions: list[str] = Field(default_factory=list)
    """Names of the redaction rules that changed this record. Bundles written before the field read as []."""
    selection: dict[str, Any] | None = None
    """On a website episode (source "website:selection"): the variant the agent chose, as the harness re-read it
    (`carttrace.collect.selection`: product_id, variant_id or options, url, quantity, price_seen, availability_seen,
    verified, proof). Left out of the line when None, so snapshots written before it keep their bytes."""

    @model_serializer(mode="wrap")
    def _without_unset_selection(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if data.get("selection") is None:
            data.pop("selection", None)
        return data


class SandboxEvent(_Frozen):
    """One line of events.jsonl: one record of the sandbox's event log (`carttrace.sandbox.events`, schema
    mc.sandbox.events.v1), as the sandbox chained it, with its `data` redacted at write time.

    A line no redaction rule changed is the sandbox's record byte for byte, so `hash` is the sha256 of the canonical
    JSON of every other field and `prev` the hash of the line before (64 zeros on an episode's first line). A line a
    rule changed lists the rules in `redactions` and keeps the sandbox's `prev` and `hash`, so the chain still links
    but that line's own hash can no longer be recomputed (carttrace.trace.events.verify_events says how many lines it took on
    trust)."""

    episode: int
    """The sandbox episode the record belongs to: 0 for a collected episode, which never resets."""
    env: str | None = None
    """The envpack id the sandbox ran."""
    idx: int
    """The record's place in its episode's chain, from 0."""
    seq: int
    """The sandbox's command counter when the record was chained: 0 for the opening and its at_start faults, then one
    per command; the records of one command share it."""
    at: int
    """The sandbox's logical clock, milliseconds since the epoch."""
    type: str
    """episode.started, command, command.applied, command.rejected, fault.fired, inventory.stockout, ..."""
    data: dict[str, Any] = Field(default_factory=dict)
    sealed: bool = False
    """The environment's own doing (a fault firing) or bookkeeping: logged for the grader, never shown to the agent."""
    prev: str
    hash: str
    redactions: list[str] = Field(default_factory=list)
    """Names of the redaction rules that changed `data`. Left out of the line when empty."""

    @model_serializer(mode="wrap")
    def _without_unset_redactions(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if not data.get("redactions"):
            data.pop("redactions", None)
        return data


class StateDigest(_Frozen):
    """What the sandbox's world ended as: enough to prove a replay of the same row reached the same place."""

    state: str
    """sha256 of the sandbox's final state (`State.hash`), when the outcome was recorded."""
    events: str
    """The hash of the last line of events.jsonl: the head of the chain, which commits to every event."""
    count: int
    """Lines in events.jsonl."""
    env: str | None = None
    """The envpack id the sandbox ran."""


class Outcome(_Frozen):
    end_reason: EndReason
    boundary: Literal["none", "cart", "checkout_handoff", "order", "selection"]
    final_cart: Cart | None = None
    handoff: dict[str, Any] | None = None
    """Live cart to sandbox checkout: {"envpack": ..., "checkout_id": ..., "profile": ...}."""
    infra_labels: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    """turns, tool_calls, tool_errors, prompt_tokens, completion_tokens, decisions, wire, snapshots, and redacted: the
    records any redaction rule changed. Since D69 also cached_prompt_tokens (where the endpoint reports a cache
    split), gpu_ms and cost_usd_micros, summed over the episode's model calls. A detail_hidden episode that names
    catalog details adds hidden.<detail>: the statements of that detail the collector's filter kept from the agent
    (none means the store never stated it)."""
    duration_ms: int = 0
    ended_at: str
    selection: dict[str, Any] | None = None
    """On a website episode (boundary "selection", D31): the verified selection record, the same as the
    "website:selection" state snapshot holds; final_cart is null there. Left out when None."""
    state_digest: StateDigest | None = None
    """On a sandbox episode whose sandbox opened (D33): its final state hash and the head of events.jsonl. Left out
    when None, so outcomes written before it, and every live one, keep their bytes."""
    checkout_faults: list[FaultFiring] | None = None
    """On a sandbox order row that drew a checkout fault (D66): whether each fired, and where in the store's event log.
    Left out when None."""

    @model_serializer(mode="wrap")
    def _without_unset_selection(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        for name in ("selection", "state_digest", "checkout_faults"):
            if data.get(name) is None:
                data.pop(name, None)
        return data


DRAINED_LABEL = "drained"
"""The infra label the runner writes on an episode it cancelled while draining for a restart (carttrace.collect.episode):
it ends as harness_error so the judge voids it, but it is deliberate, never a fault and never lost work, since a new
run of the manifest resumes it. Ops counts it apart."""


def is_drained(end_reason: str | None, infra_labels: Iterable[str]) -> bool:
    """Whether an outcome is a deliberate drain rather than a harness fault."""
    return end_reason == "harness_error" and DRAINED_LABEL in infra_labels


class BundleSummary(_Frozen):
    """What a scan of bundles yields without reading turns or wire records: enough for counts, pulses and charts."""

    episode_id: str
    run_id: str
    mode: Mode
    store_id: str
    category: str
    intent: str
    condition: str
    profile_id: str
    started_at: str
    ended_at: str | None
    end_reason: str | None
    """None while the episode has no outcome yet."""
    complete: bool
    path: str
    """Storage key prefix of the bundle."""
    platform: str | None = None
    """The store's platform as the header names it (EpisodeHeader.store.platform); None when it names none."""
    provenance: Literal["live", "replica", "twin", "unknown"] = "unknown"
    """Where the store came from (carttrace.trace.provenance): the store itself, a replica of its capture, or a twin."""
    agent: str = ""
    """The agent model, served when recorded, else requested; empty when the header names none."""
    surface: str = ""
    """The surface kinds the episode spoke to, joined with "+"."""
    manifest_id: str = ""
    code_version: str = ""
    store_source: str = ""
    """Where the store's data came from (carttrace.trace.provenance.store_source_of): a catalog, capture, team_sheet, ..."""
    pruned: bool = False
    """A tombstone (D79, carttrace.trace.prune): its logs were removed on purpose; header, outcome and bundle.json stay."""
    infra_labels: tuple[str, ...] = ()
    """The outcome's infra labels (`drained`, model labels); empty while the episode has no outcome."""

    @property
    def drained(self) -> bool:
        """Cancelled on purpose by a draining runner: not a harness fault, not a loss."""
        return is_drained(self.end_reason, self.infra_labels)

    @property
    def harness_fault(self) -> bool:
        """Ended in harness_error for a reason other than a drain: our own failure."""
        return self.end_reason == "harness_error" and not self.drained


class BundleManifest(_Frozen):
    schema_: Literal["mc.trace.v1"] = Field(default="mc.trace.v1", alias="schema")
    episode_id: str
    files: dict[str, str]
    """File name to sha256 of its bytes."""
    complete: bool

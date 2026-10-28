"""The environment pack document, `mc.envpack.v2`: the file format, as pydantic models at the I/O boundary.

What a pack holds:
- `store`: which store the world is a copy of;
- `profile`: the `ProtocolProfile` the store speaks, so every surface mounts at those exact versions;
- `payment_handlers`: the handler configs exactly as the live store declared them (raw JSON, namespace to entries);
- `snapshot`: the products and variants a journey touched, with prices, currency and availability, and for a replica
  of a real store the catalog answers the store gave during its capture (`recorded`, decision D12);
- `market`: the one market row (tax, shipping, pickup, payment methods, returns);
- `tasks`: the tasks the pack was built for, as trace `TaskRef`s;
- `start`: what every episode starts from (seed, logical clock, tick, reservation window, sessions and tiers);
- `policy`: discount codes, whether they stack, and whether the sandbox's fake handler is served;
- `faults`: the sealed fault schedule, as knobs (`carttrace.sandbox.faults`);
- `tool_lists`: for a replica, the store's own MCP `tools/list` replies as recorded live, one per MCP and UCP version
  pair, which the UCP MCP surface lists verbatim (decision D40);
- `provenance`: where the pieces came from (trace ids, capture times, data rights), carried as given.

Why strict. A pack is the whole world an episode starts from and its id names that world, so a pack that loads must be
exactly what `build_envpack` writes: unknown keys, wrong types, a market priced in another currency than the catalog,
or an inconsistent handler declaration are refused when the pack is built or loaded, never discovered mid-episode.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from carttrace.protocols.profile import ProtocolProfile
from carttrace.sandbox.core.market import Market, PickupLocation, ShippingOption
from carttrace.sandbox.faults import compile_faults
from carttrace.trace.schema import SoldOutRate, StoreErrorRate, TaskRef

SCHEMA: Final = "mc.envpack.v2"

Currency = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
Country = Annotated[str, Field(pattern=r"^[A-Z]{2}$")]
Name = Annotated[str, Field(min_length=1, max_length=200)]
Minor = Annotated[int, Field(ge=0)]
Tier = Literal["anonymous", "signed", "token"]


class _Doc(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class StoreDoc(_Doc):
    domain: Annotated[str, Field(min_length=1, max_length=253)]
    store_id: Name | None = None
    """The queue id (e.g. "store-001") when the store is in the team's queue."""
    platform: Name | None = None
    category: Name | None = None


class SnapshotVariant(_Doc):
    product_id: str = ""
    title: str = ""
    price: Minor
    """Minor units of the snapshot's currency, as the store showed it."""
    available: bool = True
    stock: Minor = 100
    """Live stores rarely publish stock, so the sandbox gives each variant a shelf of this size."""
    requires_selling_plan: bool = False
    options: dict[str, str] = Field(default_factory=dict)


class RecordedProduct(_Doc):
    id: str
    """The product the store listed."""
    variants: list[str] = Field(default_factory=list)
    """The variants it listed with it, in its order."""


class RecordedAnswer(_Doc):
    """One catalog answer a store gave during its capture, which a replica replays: the store's ranking, the variants
    it listed per product, its pagination and its selection cannot be recomputed from product text, so they are kept.
    Prices and availability are not: they are rendered from the state, so faults and orders still change them."""

    tool: Literal["search_catalog", "get_product"]
    request: dict[str, Any]
    """The request as the replica matches it (carttrace.sandbox.surfaces.ucp.catalog search_request, detail_request)."""
    products: list[RecordedProduct] = Field(default_factory=list)
    pagination: dict[str, Any] | None = None
    """A search's pagination as the store answered it: has_next_page and the store's own cursor."""
    selected: list[dict[str, str]] | None = None
    """A get_product's effective selection."""
    options: list[dict[str, Any]] | None = None
    """A get_product's option values with the store's availability signals, used for values whose variants the
    capture never saw."""


class RecordedToolList(_Doc):
    """One `tools/list` reply a store sent live, kept verbatim: what its agent read before its first call. A replica
    lists exactly these tools at these versions, so the agent's first prompt is the one the live agent got (D40)."""

    mcp: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    """The MCP version the reply was sent at."""
    ucp: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    """The UCP release the store served on that endpoint."""
    tools: list[dict[str, Any]] = Field(min_length=1)
    """The reply's `tools`, in the store's order, every member as sent."""
    source: Annotated[str, Field(min_length=1, max_length=200)]
    """Where it was read: `live:<episode>#<seq>`, `capture:<capture>#<seq>` or `platform:<name>`."""
    recorded_at: str | None = None
    """RFC 3339 time of the episode or capture that recorded it."""


class Snapshot(_Doc):
    currency: Currency
    source: Annotated[str, Field(min_length=1, max_length=64)] = "fixture"
    """Where the snapshot was read: "fixture", "live:cart", "live:get_cart", ..."""
    captured_at: str | None = None
    """RFC 3339 time a live snapshot was taken."""
    variants: dict[str, SnapshotVariant] = Field(default_factory=dict)
    """Keyed by variant id."""
    products: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Product facts an adapter shows (titles, descriptions, options), keyed by product id, as captured."""
    recorded: list[RecordedAnswer] = Field(default_factory=list)
    """A replica's recorded catalog answers. Left out of the canonical form when empty, so packs built before the
    field existed keep their bytes and their ids."""


class ShippingDoc(_Doc):
    id: Name
    title: str
    carrier: str
    amount: Minor
    free_above: Minor | None = None


class PickupDoc(_Doc):
    id: Name
    title: str
    address_country: Country


class MarketDoc(_Doc):
    code: Annotated[str, Field(pattern=r"^[A-Z]{2}$")]
    currency: Currency
    tax_bp: Annotated[int, Field(ge=0, le=10_000)]
    tax_inclusive: bool
    shipping: list[ShippingDoc] = Field(min_length=1)
    pickup: list[PickupDoc] = Field(default_factory=list)
    accepts: list[Annotated[str, Field(pattern=r"^[a-z][a-z_]{0,39}$")]] = Field(
        default_factory=lambda: ["card", "cod"]
    )
    return_window_days: Annotated[int, Field(ge=0)] = 30
    unshippable: list[Country] = Field(default_factory=list)

    @classmethod
    def from_market(cls, m: Market) -> MarketDoc:
        return cls.model_validate(m.to_dict())

    def to_market(self) -> Market:
        return Market(
            code=self.code,
            currency=self.currency,
            tax_bp=self.tax_bp,
            tax_inclusive=self.tax_inclusive,
            shipping=tuple(ShippingOption(o.id, o.title, o.carrier, o.amount, o.free_above) for o in self.shipping),
            pickup=tuple(PickupLocation(p.id, p.title, p.address_country) for p in self.pickup),
            accepts=tuple(self.accepts),
            return_window_days=self.return_window_days,
            unshippable=tuple(self.unshippable),
        )


class SessionDoc(_Doc):
    id: Name
    tier: Tier
    permissions: list[Name] = Field(default_factory=list)
    revoked: bool = False
    profile_id: Annotated[str, Field(pattern=r"^pp-[0-9a-f]{12}$")] | None = None
    """The profile the session speaks under; the pack's own profile when absent."""


class StartDoc(_Doc):
    seed: int = 11
    clock_start_ms: Minor = 1_790_000_000_000
    """Logical milliseconds since the epoch when an episode starts (2026-09-21)."""
    tick_ms: Minor = 1000
    """How far the logical clock moves per command that does not carry its own `at`."""
    reservation_ttl_ms: Minor = 15 * 60 * 1000
    default_tier: Tier = "anonymous"
    default_permissions: list[Name] = Field(default_factory=list)
    sessions: list[SessionDoc] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_sessions(self) -> Self:
        ids = [s.id for s in self.sessions]
        if len(ids) != len(set(ids)):
            raise ValueError("session ids repeat")
        return self


class CouponDoc(_Doc):
    """A discount code with more to it than a percentage (carttrace.sandbox.core.config.Coupon)."""

    code: Annotated[str, Field(pattern=r"^[A-Z0-9][A-Z0-9_-]{0,63}$")]
    kind: Literal["percent", "fixed_cart", "free_shipping"] = "percent"
    amount: Minor = 0
    """Percent off for a percent coupon (at most 100), minor units off for a fixed_cart one, 0 for free_shipping."""
    individual_use: bool = False
    min_spend: Minor | None = None
    priority: int = 0
    stop: bool = False

    @model_validator(mode="after")
    def _percent(self) -> Self:
        if self.kind == "percent" and self.amount > 100:
            raise ValueError(f"coupon {self.code}: a percent coupon takes at most 100")
        return self


class SoldOutDoc(_Doc):
    """Which variants a replica answers "sold out" at cart write while its catalog lists them available (D52).

    A world names its rate and no variants; an episode's pack (carttrace.collect.worlds.sold_out_world) adds the seed it was
    drawn with, the variants the draw refused and the ones it spared so the task can still be bought. The core reads
    the variants alone, so the same pack refuses the same lines on every run."""

    rate: SoldOutRate
    seed: int | None = None
    """The episode seed the variants were drawn with; None on a world, which refuses nothing yet."""
    variants: list[str] = Field(default_factory=list)
    """The variants refused as sold out at cart write, sorted."""
    spared: list[str] = Field(default_factory=list)
    """Variants the draw picked but kept for sale, so a task with witnesses keeps one that can be bought."""

    @model_validator(mode="after")
    def _sorted(self) -> Self:
        if self.variants != sorted(set(self.variants)) or self.spared != sorted(set(self.spared)):
            raise ValueError("sold-out variants and spared variants are sorted and unique")
        if set(self.variants) & set(self.spared):
            raise ValueError("a spared variant is not sold out")
        return self


class DrawnFault(_Doc):
    """One call an episode's draw answers with its store's own error (D70): the `call`-th call of `tool` in a
    session."""

    tool: str
    call: int = Field(ge=1)
    kind: Literal["throttled", "server_error"]


class StoreErrorsDoc(_Doc):
    """The errors a replica answers some calls with, at the rates its store answered them live (D70).

    A world names the rates and no calls; an episode's pack (carttrace.collect.worlds.store_errors_world) adds the seed it was
    drawn with and the calls that fail. The surface reads the calls alone, so the same pack fails the same calls on
    every run."""

    rates: list[StoreErrorRate]
    seed: int | None = None
    """The episode seed the calls were drawn with; None on a world, which fails nothing yet."""
    calls: list[DrawnFault] = Field(default_factory=list)
    """Sorted by tool and call."""

    @model_validator(mode="after")
    def _sorted(self) -> Self:
        keys = [(c.tool, c.call) for c in self.calls]
        if keys != sorted(set(keys)):
            raise ValueError("drawn store errors are sorted by tool and call, one per call")
        return self


class AgentViewDoc(_Doc):
    """The recorded agent view a stand-in's open products are served in (D100, carttrace.sandbox.views): the shape the store
    it stands in for sent agents live, so its catalog documents carry that store's fields, nesting and value
    conventions instead of the sandbox's own document."""

    ref: Name
    """`<template>:<store>@<digest>`, as experiment.json and exposure.json name it."""
    template: Name
    """The template file's id under data/sandbox/views (shopify-ucp-mcp)."""
    store: Name
    """The store whose recorded view this is."""
    own: list[Literal["listing", "detail", "lookup"]]
    """The views the store's own recordings give; the others are the template's dominant view."""
    views: dict[str, dict[str, Any]]
    """View name to its shape (carttrace.sandbox.surfaces.ucp.catalog SHAPE_KEYS)."""
    format: dict[str, Any]
    """How values are written (carttrace.sandbox.views.ViewFormat)."""
    categories: dict[str, str] = Field(default_factory=dict)
    """Product id to its Shopify taxonomy category id (`ap-2-3`), mapped from our category and its product type."""
    ids: dict[str, str] = Field(default_factory=dict)
    """Each Shopify-shaped product and variant id the world serves to the open source's own id it stands for."""


class PresentedDoc(_Doc):
    """The identity a stand-in presents to agents (D106, carttrace.sandbox.presented): what every surface calls the store,
    its links' origin, its MCP server, its payment names and its id scheme, resolved from data/sandbox/visibility.yaml
    when the world is built."""

    store_name: Name
    domain: Name
    origin: Name
    merchant_id: Name
    realm: Name
    mcp_server: dict[str, str] = Field(default_factory=dict)
    """The MCP serverInfo; empty keeps the platform pack's own."""
    session_header: Name
    acp_psp: Name
    mpp_method: Annotated[str, Field(pattern=r"^[a-z]+$")]
    """An MPP payment method identifier: lowercase letters only (MPP core, "Method Identifier Format")."""
    card_namespace: Name
    card_handler_id: Name
    shop_id: Name
    id_scheme: Literal["shopify", "numeric", "prefixed", "graphql"]
    version: Name
    """The visibility.yaml version it was resolved from."""
    ids: dict[str, str] = Field(default_factory=dict)
    """Each presented product and variant id to the open source's own id."""
    served_names: dict[str, str] = Field(default_factory=dict)
    """The sandbox core's own names a response can echo, to the names they are served by (the server rewrites them
    at its boundary, both ways)."""
    market_names: dict[str, str] = Field(default_factory=dict)
    """The market table's carrier and pickup location names, to the names the stand-in's world gives them."""
    store_address: str | None = None
    """The store's own address as its policy pages state it, for the world's market; None states none."""


class ThreeDsDoc(_Doc):
    """3-D Secure decided at once, accepted or declined, with no challenge page (D111, core.config.ThreeDs)."""

    rail: Name | None = None
    abandon: bool = False


class PolicyDoc(_Doc):
    discount_codes: dict[str, Annotated[int, Field(ge=0, le=100)]] = Field(default_factory=lambda: {"WELCOME10": 10})
    """Code to percent off. Codes are upper-case: shoppers' codes are matched case-insensitively."""
    stack_discounts: bool = False
    fake_handler: bool = True
    """Serve the sandbox's own credential-free handler (`dev.ucp.sandbox.fake`) beside the declared ones."""
    coupons: list[CouponDoc] = Field(default_factory=list)
    """Codes with rules of their own (fixed amounts, free shipping, individual use, minimum spend, priority). A code
    here and in `discount_codes` is this one. Left out of the canonical form when empty, so packs keep their ids."""
    platform_pack: Name | None = None
    """The platform pack the store is served under (carttrace.sandbox.platforms, D42) when it is not the store's own
    platform: the same products and market, another platform's behaviour. None serves `store.platform`'s. Left out
    of the canonical form when None."""
    sold_out: SoldOutDoc | None = None
    """The store's sold-out answers at cart write (D52); None answers from stock alone. Left out of the canonical
    form when None, so packs keep their ids."""
    store_errors: StoreErrorsDoc | None = None
    """The store's own errors, drawn per episode (D70); None answers every call. Left out of the canonical form when
    None, so packs keep their ids."""
    psp: Literal["stripe", "adyen", "braintree", "paypal"] | None = None
    """The payment broker behind the store's PSP (D71): the processor whose test-mode shapes its payments are recorded
    in. None is the sandbox's own PSP. Left out of the canonical form when None, so packs keep their ids."""
    agent_view: AgentViewDoc | None = None
    """The recorded agent view a stand-in is served in (D100); None serves the platform pack's documents. Left out of
    the canonical form when None, so packs keep their ids."""
    presented: PresentedDoc | None = None
    """The identity a stand-in presents (D106); None presents the legacy one. Left out of the canonical form when
    None, so packs keep their ids."""
    three_ds: ThreeDsDoc | None = None
    """3-D Secure as a plain accept or decline (D111); None parks a payment on a challenge page (D71). Left out of the
    canonical form when None, so packs keep their ids."""

    @model_validator(mode="after")
    def _upper_codes(self) -> Self:
        if any(code != code.upper() for code in self.discount_codes):
            raise ValueError("discount codes must be upper-case")
        codes = [c.code for c in self.coupons]
        if len(codes) != len(set(codes)):
            raise ValueError("coupon codes repeat")
        return self


def handler_entries(declared: Any) -> list[Mapping[str, Any]]:
    """A namespace's declared entries, whether the store wrote one object or a list of them."""
    items = declared if isinstance(declared, list) else [declared]
    return [e for e in items if isinstance(e, Mapping)]


def declared_handlers(payment_handlers: Mapping[str, Any], profile: ProtocolProfile) -> dict[str, str]:
    """Handler id to namespace for every handler the store declared: each namespace by its own name, and each entry by
    its `id`, which is what a UCP `handler_id` names. Raises ValueError when one id names two namespaces."""
    out: dict[str, str] = {}
    for ns in sorted(set(payment_handlers) | set(profile.ucp_payment_handlers)):
        names = [ns] + [str(e["id"]) for e in handler_entries(payment_handlers.get(ns)) if isinstance(e.get("id"), str)]
        for name in names:
            if out.setdefault(name, ns) != ns:
                raise ValueError(f"handler id {name!r} is declared by both {out[name]!r} and {ns!r}")
    return out


class EnvPackDoc(_Doc):
    schema_: Literal["mc.envpack.v2"] = Field(default=SCHEMA, alias="schema")
    store: StoreDoc
    profile: ProtocolProfile
    payment_handlers: dict[str, Any] = Field(default_factory=dict)
    """As the live store declared them: namespace to one entry or a list of entries ({id, version, config, ...})."""
    snapshot: Snapshot
    market: MarketDoc
    tasks: list[TaskRef] = Field(default_factory=list)
    start: StartDoc = Field(default_factory=StartDoc)
    policy: PolicyDoc = Field(default_factory=PolicyDoc)
    faults: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    tool_lists: list[RecordedToolList] = Field(default_factory=list)
    """A replica's recorded tool lists, one per (mcp, ucp). Left out of the canonical form when empty, so packs built
    before the field existed keep their bytes and their ids."""

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.market.currency != self.snapshot.currency:
            raise ValueError(
                f"the market is priced in {self.market.currency} but the snapshot in {self.snapshot.currency}"
            )
        for ns, declared in self.payment_handlers.items():
            items = declared if isinstance(declared, list) else [declared]
            if not ns or not items or not all(isinstance(e, Mapping) for e in items):
                raise ValueError(f"payment handler {ns!r} must be declared as an object or a list of objects")
        declared_handlers(self.payment_handlers, self.profile)
        compile_faults(self.faults)  # raises FaultConfigError, a typed error that pydantic passes through
        pairs = [(t.mcp, t.ucp) for t in self.tool_lists]
        if len(pairs) != len(set(pairs)):
            raise ValueError("tool_lists holds two lists for one MCP and UCP version pair")
        return self

    def tool_list(self, *, mcp: str, ucp: str) -> RecordedToolList | None:
        """The recorded tool list at exactly these versions, or None."""
        return next((t for t in self.tool_lists if t.mcp == mcp and t.ucp == ucp), None)

    def dump(self) -> dict[str, Any]:
        """The document as JSON, every field included, keys as the file spells them; an empty `snapshot.recorded` and
        empty `tool_lists` are left out, as packs written before they existed spell them."""
        out: dict[str, Any] = self.model_dump(mode="json", by_alias=True)
        if not out["snapshot"]["recorded"]:
            del out["snapshot"]["recorded"]
        if not out["tool_lists"]:
            del out["tool_lists"]
        unset_names: tuple[tuple[str, Any], ...] = (
            ("coupons", []),
            ("platform_pack", None),
            ("sold_out", None),
            ("store_errors", None),
            ("psp", None),
            ("agent_view", None),
            ("presented", None),
            ("three_ds", None),
        )
        for name, unset in unset_names:
            if out["policy"].get(name) == unset:
                del out["policy"][name]
        return out

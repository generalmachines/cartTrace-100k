"""A platform pack as data: data/sandbox/packs/<pack>.yaml, validated by this schema (decision D101).

Why data. Nine packs (D42, D68) each encoded their differences from the protocol in code, ad hoc: a refusal table in
one module, a flag in another, an evidence string beside both. D81 found the Medusa pack refusing `context` and
`cart_id`, which the vendored UCP spec defines at every served release; the rule had come from reading the plugin's
source, no probe showed it, and nothing checked it against the spec. A spec file says, in one place and in one shape
for every pack, what the pack serves, which request fields it accepts, renames, requires, ignores or refuses, how it
reshapes documents, which error shapes and codes it answers with, and its checkout specifics, each with the spec
section it follows and, where it departs from the spec, the source that shows the platform does so. The conformance
suite (`carttrace.sandbox.platforms.conformance`, `carttrace sandbox packs-check`) asks every pack every operation the spec
defines and fails on any departure the spec file does not declare.

The rules the schema enforces:
- every divergence cites a source (`Cite`): a recorded wire example (a live bundle under var/store/episodes or
  var/store/captures, or a playground recording under ops/playgrounds), the platform's public docs or its source at a
  pinned tag (a URL), or `assumed` with why;
- a pack never refuses a field the served release defines (D81), except a field of a capability it does not serve
  (`serves.ucp.unserved`): a `refuse` rule must name such a capability;
- what data cannot express stays in code, and the spec says where and why (`code`).

`pack_spec(name)` loads and caches one file; `pack_specs()` all of them; `json_schema()` is the generated JSON Schema
(data/sandbox/packs/pack.schema.json, kept current by a test).
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA: Final = "mc.sandbox.pack.v1"
PACKS_DIR: Final = Path(__file__).resolve().parents[4] / "data" / "sandbox" / "packs"
SCHEMA_FILE: Final = "pack.schema.json"
WIRE_ROOTS: Final = ("var/store/episodes/", "var/store/captures/", "ops/playgrounds/")
"""Where a recorded wire example lives: live bundles, live captures, playground recordings."""
UCP_RELEASES: Final = ("2026-01-11", "2026-01-23", "2026-04-08", "2026-08-25")
"""The UCP releases the sandbox serves (carttrace.sandbox.surfaces.ucp.dialect.DIALECTS); a test keeps them equal."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# -- sources -----------------------------------------------------------------------------------------------------------


class Cite(_Model):
    """Where a behaviour or a divergence comes from. Exactly one of `source`, `wire`, `docs` or `assumed`."""

    source: str | None = None
    """A key of the spec's `sources`, which says where it is once."""
    wire: str | None = None
    """A recorded wire example: a path under var/store/episodes or var/store/captures (live), or ops/playgrounds
    (a playground recording)."""
    docs: str | None = None
    """The platform's public docs, or its source at a pinned tag: an https URL."""
    assumed: str | None = None
    """Why it is assumed: what was not observed or read, and what the pack does instead."""
    at: str | None = None
    """Where inside the source: a probe or seq number, a line range, a section."""
    note: str | None = None
    """What the source shows, in a sentence."""

    @model_validator(mode="after")
    def _one(self) -> Cite:
        given = [k for k in ("source", "wire", "docs", "assumed") if getattr(self, k)]
        if len(given) != 1:
            raise ValueError(f"a citation names exactly one of source, wire, docs or assumed, not {given or 'none'}")
        if self.wire is not None and not self.wire.startswith(WIRE_ROOTS):
            raise ValueError(f"a wire citation is a recording under {', '.join(WIRE_ROOTS)}: {self.wire!r}")
        if self.docs is not None and not self.docs.startswith("https://"):
            raise ValueError(f"a docs citation is an https URL: {self.docs!r}")
        return self

    @property
    def kind(self) -> str:
        return next(k for k in ("source", "wire", "docs", "assumed") if getattr(self, k))


class Source(_Model):
    """A named source, cited by key."""

    wire: str | None = None
    docs: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def _one(self) -> Source:
        if (self.wire is None) == (self.docs is None):
            raise ValueError("a source is one wire recording or one docs URL")
        Cite(wire=self.wire, docs=self.docs)
        return self


# -- what the pack serves ----------------------------------------------------------------------------------------------


class UcpServes(_Model):
    releases: tuple[str, ...]
    """The UCP releases the pack is served at: the store's own when the platform pins none, else the pinned one."""
    pinned: str | None = None
    """The release every store of this platform speaks (`Platform.profile_versions["ucp"]`), or None."""
    unserved: tuple[str, ...] = ()
    """Capabilities the platform does not serve: left out of discovery and replies, and their fields refused."""
    supported_versions: bool = False
    """The discovery document lists the older releases the store also serves, and the store's `name`."""
    cite: Cite | None = None


class AcpServes(_Model):
    pinned: str | None = None
    """The ACP API-Version every store of this platform speaks; None keeps the store's own."""
    discovery: Literal["release", "plugin"] = "release"
    """`/.well-known/acp.json` is the release's document, or the platform plugin's own (WooCommerce's recorded one;
    a dialect pack's plugin writes its own, carttrace.sandbox.platforms.medusa.acp_discovery)."""
    cite: Cite | None = None


class McpServes(_Model):
    default: str
    """The MCP version served when the store's profile names none."""
    capabilities: dict[str, Any] = Field(default_factory=lambda: {"tools": {"listChanged": False}})
    sessions: bool = True
    notification_status: int = 202
    notification_body: str | None = None
    server_info: dict[str, str] | None = None
    instructions: str | None = None
    tools_file: str | None = None
    """A package data file with the platform's recorded tools/list (carttrace.sandbox.platforms)."""
    text_refusals: bool = False
    """A call whose arguments fail the tool's schema is refused as plain text over MCP."""
    pinned: str | None = None
    cite: Cite | None = None


class Serves(_Model):
    ucp: UcpServes
    acp: AcpServes = AcpServes()
    mcp: McpServes


# -- request fields ----------------------------------------------------------------------------------------------------

Rule = Literal["accept", "rename", "require", "ignore", "refuse"]


class FieldRule(_Model):
    """What the pack does with one request field, against the spec section the field is defined in."""

    rule: Rule
    spec: str
    """The spec section that defines the field (UCP's page and heading, or the release's schema file)."""
    to: str | None = None
    """`rename`: the field the platform reads in its place."""
    capability: str | None = None
    """`refuse`: the unserved capability the field belongs to (the only refusal of a defined field D81 allows)."""
    what: str = ""
    cite: Cite | None = None
    """Required for every rule but `accept`: a departure from the spec cites its source."""

    @model_validator(mode="after")
    def _cited(self) -> FieldRule:
        if self.rule != "accept" and self.cite is None:
            raise ValueError(f"a {self.rule} rule departs from the spec, so it cites a source")
        if self.rule == "rename" and not self.to:
            raise ValueError("a rename names the field read in its place (`to`)")
        if self.rule == "refuse" and not self.capability:
            raise ValueError(
                "a pack refuses a field the release defines only when it does not serve the field's capability "
                "(D81); name it in `capability`"
            )
        return self


class OperationSpec(_Model):
    default: Literal["spec"] = "spec"
    """Every field the release's tool schema and the served extensions define is accepted as the spec says."""
    fields: dict[str, FieldRule] = Field(default_factory=dict)
    """Fields by their path in the payload (`context`, `pagination.cursor`, `fulfillment.methods[].line_item_ids`)."""


# -- documents, errors, checkout ---------------------------------------------------------------------------------------

FindingKind = Literal["refused_valid", "accepted_invalid", "error_shape", "doc_shape"]


class Divergence(_Model):
    """A documented departure from the release's schemas that the conformance suite accepts."""

    id: str
    what: str
    spec: str
    """The spec section departed from."""
    cite: Cite
    protocol: Literal["ucp", "acp"] = "ucp"
    kind: FindingKind = "doc_shape"
    operations: tuple[str, ...] = ("*",)
    releases: tuple[str, ...] = ("*",)
    where: str = "*"
    """A glob on the finding's place: the request field, the document pointer and message, or the error shape."""
    detail: str = "*"
    """A glob on the finding's detail."""

    def matches(self, release: str, operation: str, kind: str, where: str, detail: str) -> bool:
        return (
            kind == self.kind
            and any(fnmatch.fnmatchcase(operation, p) for p in self.operations)
            and any(fnmatch.fnmatchcase(release, p) for p in self.releases)
            and fnmatch.fnmatchcase(where, self.where)
            and fnmatch.fnmatchcase(detail, self.detail)
        )


class DocumentShape(_Model):
    """How the pack shapes UCP documents, named so a reader sees the reshaping without reading code."""

    dialect: str | None = None
    """The code that reshapes every reply (`medusa`: carttrace.sandbox.platforms.medusa.reshape); None keeps the sandbox's
    own documents."""
    field_names: str = "spec"
    nesting: str = "spec"
    description: Literal["spec", "text"] = "spec"
    """A product's description: the spec's description object, or a plain string."""
    money: str = "spec"
    """Minor units and an ISO 4217 code as the spec says, or how the pack writes them."""
    currency_case: Literal["upper", "lower"] = "upper"
    options: str = "spec"
    """Option and variant layout."""
    serves_options: bool = True
    """Whether the catalog shows a variant's option values (colour, size, ...). False: an agent cannot read which
    variant is which colour or size, so no task that needs an option value is written for a world under this pack
    (D121)."""
    media: str = "spec"
    pagination: str = "spec"
    product_views: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """The product shape per view (`listing`, `detail`, `lookup`, D55): `Platform.product_views`."""
    full_ucp_block: bool = False
    """Every reply's `ucp` block lists the cart and catalog capabilities with spec and schema links, the payment
    handlers and the `services` registry, as the discovery document does (live Shopify)."""
    block_extends: bool = False
    """The block names the capability each extension extends."""
    messages_always: bool = False
    """`messages` is on every document, an empty list when there is nothing to say."""
    line_display_text: bool = False
    plain_content_type: bool = False
    """Every message says `content_type: plain`."""
    divergences: tuple[Divergence, ...] = ()
    cite: Cite | None = None


class ErrorShape(_Model):
    """An error the pack answers in a shape that is not the release's error response."""

    id: str
    shape: str
    """The body's members, as the suite names them (`{message,type}`), or `mcp:text` for a plain-text refusal."""
    status: int | None = None
    operations: tuple[str, ...] = ("*",)
    what: str
    cite: Cite


class ErrorCode(_Model):
    """A platform's own error or message code and the spec code it stands for."""

    code: str
    spec: str
    """The release's code it means (`missing`, `out_of_stock`, `invalid_input`)."""
    path: str | None = None
    """The field it names, as the release's `missing` would."""
    what: str = ""
    cite: Cite


class Errors(_Model):
    shape: Literal["ucp", "platform"] = "ucp"
    shapes: tuple[ErrorShape, ...] = ()
    codes: tuple[ErrorCode, ...] = ()
    keyed_ids: bool = False
    """Cart ids are the platform's keyed gids; a malformed one is `invalid_cart_id`, an unknown one `cart_not_found`."""
    delivery_group_gids: bool = False
    cite: Cite | None = None


class Readiness(_Model):
    code: str
    text: str
    path: str


class Practice(_Model):
    """One checkout practice: what the pack does, and its source."""

    rule: str
    cite: Cite | None = None


class Checkout(_Model):
    """Checkout specifics the core and the surface read."""

    line_items: Practice = Practice(rule="spec")
    fulfillment: Practice = Practice(rule="spec")
    pickup: Practice = Practice(rule="spec")
    policy_links: tuple[tuple[str, str], ...] = ()
    """The policy links every cart and checkout carries, as (type, title)."""
    links_cite: Cite | None = None
    discounts: Practice = Practice(rule="spec")
    discounts_block: bool = True
    """Cart replies carry a `discounts` block even when no code was applied."""
    checkout_discounts: bool = True
    gift_cards: Practice = Practice(rule="not served: the sandbox has no gift card capability")
    payment_handlers: Practice = Practice(rule="the store's declared handlers")
    accepts: tuple[str, ...] = ()
    """Payment methods the platform's stores take beyond the market's."""
    cart_expires_days: int | None = None
    buyer_location_warning: bool = False
    """A line whose product's market tags exclude the buyer is added with a warning."""
    sold_out_warning: bool = False
    """A line the store will not sell is a `merchandise_out_of_stock` warning and left out; else UCP's error."""
    readiness: dict[str, Readiness] = Field(default_factory=dict)
    order_states: dict[str, str] = Field(default_factory=dict)


class Search(_Model):
    """How the platform's catalog search finds and pages products (a rule the spec leaves to the business)."""

    mode: Literal["sandbox", "storefront", "all_words", "phrase", "prefix_words", "field_prefix", "whole_words"]
    """sandbox: the sandbox's word count; storefront: live Shopify's catalog search, approximately (D39); all_words:
    every word as a substring; phrase: the whole query in one field; prefix_words: every word begins an indexed word;
    field_prefix: every stemmed word begins a word of one field; whole_words: every word whole."""
    page_default: int | None = None
    page_max: int | None = None
    available_filter: bool = False
    """`filters.available` is listed and honoured."""
    price_filter_note: bool = False
    """A price filter with no `context.currency` carries the shop-currency info message."""
    opaque_cursors: bool = False
    total_count: bool = False
    """Pagination carries `total_count`."""
    facets: bool = False
    """Layered navigation facets and price ranges beside the products."""
    cite: Cite | None = None


class CodeNote(_Model):
    """Hand-written code the spec cannot replace, and why."""

    name: str
    """The module or function (`carttrace.sandbox.platforms.medusa.reshape`)."""
    why: str


class NotPorted(_Model):
    what: str
    cite: Cite | None = None


# -- the pack ----------------------------------------------------------------------------------------------------------


class PackSpec(_Model):
    """One platform pack, whole."""

    schema_: Literal["mc.sandbox.pack.v1"] = Field(alias="schema")
    name: str
    title: str
    basis: Literal["live", "recorded", "code", "scouted", "plain"]
    """Where its rules came from: live stores, a playground's recordings, code read at a tag, scouted stores, or none
    (the sandbox's plain dialect)."""
    summary: str
    sources: dict[str, Source] = Field(default_factory=dict)
    serves: Serves
    search: Search
    operations: dict[str, OperationSpec] = Field(default_factory=dict)
    documents: DocumentShape = DocumentShape()
    errors: Errors = Errors()
    checkout: Checkout = Checkout()
    wording: dict[str, str] = Field(default_factory=dict)
    """The platform's own words for refusals (keys `discount.<reason>`, `stock.<case>`, `line.<reason>`,
    `order.status`), formatted with the event's facts."""
    wording_cite: Cite | None = None
    behaviours: dict[str, Cite] = Field(default_factory=dict)
    """Each platform rule the pack follows where the spec leaves it open, and its source (`Platform.evidence`)."""
    not_ported: dict[str, NotPorted] = Field(default_factory=dict)
    code: tuple[CodeNote, ...] = ()
    extra: dict[str, Any] = Field(default_factory=dict)
    """Platform facts a module reads that have no section (the Medusa plugin's validator table, its ACP document)."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _consistent(self) -> PackSpec:
        for cite in self.citations():
            if cite.source is not None and cite.source not in self.sources:
                raise ValueError(f"{self.name}: a citation names source {cite.source!r}, which `sources` lacks")
        unknown = [r for r in self.serves.ucp.releases if r not in UCP_RELEASES]
        if unknown:
            raise ValueError(f"{self.name}: serves UCP releases the sandbox has no dialect for: {unknown}")
        if self.serves.ucp.pinned is not None and self.serves.ucp.releases != (self.serves.ucp.pinned,):
            raise ValueError(f"{self.name}: a pinned UCP release is the only one served")
        for op, spec in self.operations.items():
            for name, rule in spec.fields.items():
                if rule.rule == "refuse" and rule.capability not in self.serves.ucp.unserved:
                    raise ValueError(
                        f"{self.name}: {op} refuses {name}, a field of {rule.capability}, which the pack serves; "
                        "a pack never refuses a field the served release defines (D81)"
                    )
        ids = [d.id for d in self.divergences]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.name}: divergence ids repeat: {sorted({i for i in ids if ids.count(i) > 1})}")
        return self

    # -- reads ---------------------------------------------------------------------------------------------------------

    def citations(self) -> list[Cite]:
        """Every citation in the spec."""
        out: list[Cite] = []

        def walk(node: Any) -> None:
            if isinstance(node, Cite):
                out.append(node)
            elif isinstance(node, BaseModel):
                for name in type(node).model_fields:
                    walk(getattr(node, name))
            elif isinstance(node, Mapping):
                for v in node.values():
                    walk(v)
            elif isinstance(node, list | tuple):
                for v in node:
                    walk(v)

        walk(self)
        return out

    def resolve(self, cite: Cite) -> Cite:
        """A citation with its named source filled in."""
        if cite.source is None:
            return cite
        src = self.sources[cite.source]
        return Cite(wire=src.wire, docs=src.docs, at=cite.at, note=cite.note)

    def said(self, cite: Cite) -> str:
        """A citation as one line of text: the note, then where."""
        full = self.resolve(cite)
        where = full.wire or full.docs or f"assumed: {full.assumed}"
        name = f"{cite.source} " if cite.source else ""
        at = f" {full.at}" if full.at else ""
        place = f"{name}{at.strip()}" if cite.source else f"{where}{at}"
        return f"{full.note}; {place}" if full.note else place

    @property
    def divergences(self) -> list[Divergence]:
        """Every departure the suite accepts: declared document and error divergences, and each field rule that
        refuses, requires, renames or ignores a field, as the finding it explains."""
        out = list(self.documents.divergences)
        for shape in self.errors.shapes:
            out.append(
                Divergence(
                    id=f"error.{shape.id}",
                    what=shape.what,
                    spec="the release's error response",
                    cite=shape.cite,
                    kind="error_shape",
                    operations=shape.operations,
                    where=shape.shape,
                )
            )
        for op, spec in self.operations.items():
            for name, rule in spec.fields.items():
                if rule.rule == "accept" or rule.cite is None:
                    continue
                payload = _payload_of(op)
                at = f"{payload}.{name}" if payload else name
                kind: FindingKind = "accepted_invalid" if rule.rule in ("ignore", "rename") else "refused_valid"
                out.append(
                    Divergence(
                        id=f"field.{op}.{name}",
                        what=rule.what or f"{rule.rule} {name}",
                        spec=rule.spec,
                        cite=rule.cite,
                        kind=kind,
                        operations=(op,),
                        where=f"{at}*",
                    )
                )
        return out

    def evidence(self) -> dict[str, str]:
        """Each behaviour and the line of text that cites it (`Platform.evidence`): the citation's note, or what an
        assumption says, or where the source is."""
        return {key: self.text_of(cite) for key, cite in self.behaviours.items()}

    def text_of(self, cite: Cite) -> str:
        if cite.note:
            return cite.note
        if cite.assumed:
            return cite.assumed
        full = self.resolve(cite)
        place = full.wire or full.docs or ""
        return (
            f"{place}{full.at or ''}" if (full.at or "").startswith("#") else " ".join(p for p in (place, full.at) if p)
        )

    def refused_fields(self) -> frozenset[str]:
        """Top-level body fields the pack refuses on every operation: those of the capabilities it does not serve."""
        return frozenset(
            name.split(".")[0]
            for spec in self.operations.values()
            for name, r in spec.fields.items()
            if r.rule == "refuse"
        )

    def error_code(self, code: str) -> ErrorCode | None:
        """The spec code a platform's own code stands for, or None when it is the spec's own."""
        return next((c for c in self.errors.codes if c.code == code), None)

    def field_rule(self, operation: str, name: str) -> FieldRule | None:
        spec = self.operations.get(operation)
        return spec.fields.get(name) if spec is not None else None


def _payload_of(tool: str) -> str | None:
    if tool.endswith("_cart"):
        return "cart"
    if tool.endswith("_checkout"):
        return "checkout"
    if tool in ("search_catalog", "lookup_catalog", "get_product"):
        return "catalog"
    return None


# -- loading -----------------------------------------------------------------------------------------------------------


def load(path: Path) -> PackSpec:
    """A spec file, validated; a typed error naming the file when it is not one."""
    from carttrace.core.errors import ValidationError

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return PackSpec.model_validate(raw)
    except ValueError as exc:
        raise ValidationError(f"{path} is not a valid pack spec: {exc}", hint="see docs/sandbox/PACKS.md") from exc


@cache
def pack_spec(name: str) -> PackSpec:
    """The spec of pack `name` (data/sandbox/packs/<name>.yaml)."""
    spec = load(PACKS_DIR / f"{name}.yaml")
    if spec.name != name:
        raise ValueError(f"{name}.yaml names the pack {spec.name!r}")
    return spec


def pack_specs() -> dict[str, PackSpec]:
    """Every spec under data/sandbox/packs, by name."""
    return {p.stem: pack_spec(p.stem) for p in sorted(PACKS_DIR.glob("*.yaml"))}


def json_schema() -> dict[str, Any]:
    """The JSON Schema of a pack spec, generated from the model (data/sandbox/packs/pack.schema.json)."""
    out = PackSpec.model_json_schema(by_alias=True)
    out["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    out["$id"] = f"https://carttrace.invalid/schemas/{SCHEMA}.json"
    return out


def schema_text() -> str:
    return json.dumps(json_schema(), indent=2, sort_keys=True) + "\n"

"""The environment pack: one immutable, content-named bundle that is everything an episode starts from.

Its id is `env-<sha12>`, the first twelve hex digits of the sha256 of its canonical JSON (`carttrace.core.canon`), so two
episodes that name the same pack started from byte-identical worlds, and anything that changes the world (a price, a
handler config, the fault schedule, the profile) is a new id.

Why it fixes the reset leak. The lab's old server rebuilt carts and ids on reset but left its catalog as the last
episode had left it unless it was seeded again: a stockout fault, a price change or a completed order's stock decrement
carried into the next episode. Here reset is `pack.initial_state()`, a pure function of an immutable document, so the
catalog after a reset is the catalog in the pack, exactly, every time.

Files. `save` writes `<dir>/<id>.json` holding exactly the canonical bytes; `load` refuses a file whose bytes are not
the canonical form of a valid v2 document or whose content does not hash to its name, so a pack that loads is the pack
that was saved, byte for byte. `public()` is the pack without its fault schedule and hidden task specs: what may be
shown to anything the agent can read.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pydantic

from carttrace.core.canon import dumps, sha256_hex
from carttrace.core.errors import ValidationError
from carttrace.core.files import atomic_write_bytes
from carttrace.protocols.profile import ProtocolProfile
from carttrace.sandbox.core.apply import initial_state
from carttrace.sandbox.core.config import (
    DELEGATED_NAMESPACE,
    FAKE_HANDLERS,
    FAKE_NAMESPACE,
    REQUESTED_NAMESPACE,
    SHOP_PAY_HANDLER_ID,
    SHOP_PAY_NAMESPACE,
    Config,
    Coupon,
    ThreeDs,
    adopted_shop_id,
    core_rules,
)
from carttrace.sandbox.core.market import Market, market_for
from carttrace.sandbox.core.state import Session, State, Variant
from carttrace.sandbox.envpack.doc import (
    SCHEMA,
    AgentViewDoc,
    EnvPackDoc,
    MarketDoc,
    PresentedDoc,
    RecordedToolList,
    Snapshot,
    StoreDoc,
    declared_handlers,
    handler_entries,
)
from carttrace.sandbox.faults import compile_faults
from carttrace.trace.schema import StoreRef, TaskRef

ID_LENGTH = 12
DEFAULT_PROFILE = ProtocolProfile(ucp="2026-08-25", mcp="2025-06-18", note="sandbox default")
"""What a pack speaks when no live store's profile is given: the newest UCP core over MCP 2025-06-18."""
PROTOCOL_HANDLERS: Mapping[str, str] = MappingProxyType(
    {"acp": DELEGATED_NAMESPACE, "mpp": REQUESTED_NAMESPACE, "ap2": REQUESTED_NAMESPACE}
)
"""The PSP handler each payment protocol pays through, served when the profile names the protocol: ACP charges
credentials the PSP vaulted (delegated payment), MPP and AP2 answer payment requests (a 402 challenge, a signed
cart)."""
DERIVABLE = ("profile", "payment_handlers", "market", "tasks", "faults", "start", "policy", "provenance", "tool_lists")
MERGED = ("start", "policy")
"""Parts `derive` merges key by key; the other derivable parts are replaced whole."""


class EnvPackError(ValidationError):
    """A pack that does not build, does not load, or is not the pack its name says."""


class EnvPack:
    """Immutable. Build one with `build_envpack`, `EnvPack.load` or `EnvPack.from_bytes`."""

    __slots__ = ("_doc", "_json", "_sha")

    _doc: EnvPackDoc
    _json: str
    _sha: str

    def __init__(self, doc: EnvPackDoc) -> None:
        text = dumps(doc.dump())
        object.__setattr__(self, "_doc", doc)
        object.__setattr__(self, "_json", text)
        object.__setattr__(self, "_sha", sha256_hex(text))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("an environment pack is immutable; derive or build a new one")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EnvPack) and other._sha == self._sha

    def __hash__(self) -> int:
        return hash(self._sha)

    def __repr__(self) -> str:
        return f"EnvPack({self.id}, store={self._doc.store.domain!r}, profile={self._doc.profile.id})"

    # -- identity ------------------------------------------------------------------------------------------------------

    @property
    def id(self) -> str:
        return f"env-{self._sha[:ID_LENGTH]}"

    @property
    def sha256(self) -> str:
        """The full sha256 of the canonical bytes, for integrity checks beyond the short id."""
        return self._sha

    # -- reading -------------------------------------------------------------------------------------------------------

    @property
    def doc(self) -> dict[str, Any]:
        """A fresh copy of the document as JSON; changing it changes nothing."""
        out: dict[str, Any] = json.loads(self._json)
        return out

    @property
    def profile(self) -> ProtocolProfile:
        return self._doc.profile.model_copy(deep=True)

    @property
    def store(self) -> StoreDoc:
        return self._doc.store.model_copy(deep=True)

    @property
    def snapshot(self) -> Snapshot:
        return self._doc.snapshot.model_copy(deep=True)

    @property
    def presented_doc(self) -> PresentedDoc | None:
        """The identity a stand-in presents (D106), or None; read it through carttrace.sandbox.presented.presented."""
        found = self._doc.policy.presented
        return found.model_copy(deep=True) if found is not None else None

    @property
    def agent_view(self) -> AgentViewDoc | None:
        """The recorded agent view a stand-in is served in (D100), or None."""
        found = self._doc.policy.agent_view
        return found.model_copy(deep=True) if found is not None else None

    def same_goods(self, other: EnvPack) -> bool:
        """Whether both packs hold the same snapshot (products, variants, recorded answers) and market, whatever
        their profiles: a payment rail's world and its platform's world (D72)."""
        return self._doc.snapshot == other._doc.snapshot and self._doc.market == other._doc.market

    def tool_list(self, *, mcp: str, ucp: str) -> RecordedToolList | None:
        """The store's own tools/list reply recorded at these MCP and UCP versions, or None (D40)."""
        found = self._doc.tool_list(mcp=mcp, ucp=ucp)
        return found.model_copy(deep=True) if found is not None else None

    @property
    def payment_handlers(self) -> dict[str, Any]:
        """The handler configs as the live store declared them."""
        out: dict[str, Any] = self.doc["payment_handlers"]
        return out

    @property
    def tasks(self) -> list[TaskRef]:
        return [t.model_copy(deep=True) for t in self._doc.tasks]

    def market(self) -> Market:
        return self._doc.market.to_market()

    def handlers(self) -> dict[str, str]:
        """Handler id to namespace for every handler this pack serves: the fake one when the policy says so, the
        store's declared ones, and the PSP's own handler for each payment protocol the profile names
        (`PROTOCOL_HANDLERS`). A profile that adopts Shop Pay (D73) serves it under the spec's own handler id too,
        since a UCP `handler_id` names the declared entry's id."""
        out = dict(FAKE_HANDLERS) if self._doc.policy.fake_handler else {}
        shown = self._doc.policy.presented
        if out and shown is not None:
            # a stand-in lists the card handler under its presented names (D106); inside, it is the fake one
            out = {shown.card_namespace: FAKE_NAMESPACE, shown.card_handler_id: FAKE_NAMESPACE}
        out.update(declared_handlers(self._doc.payment_handlers, self._doc.profile))
        profile = self._doc.profile
        for family, namespace in PROTOCOL_HANDLERS.items():
            if getattr(profile, family) is not None:
                out.setdefault(namespace, namespace)
        if SHOP_PAY_NAMESPACE in profile.ucp_payment_handlers:
            out.setdefault(SHOP_PAY_HANDLER_ID, SHOP_PAY_NAMESPACE)
        return out

    def shop_id(self) -> str | None:
        """The Shop Pay merchant id this pack mints Shop Tokens for (D73): the store's declared handler config's
        `shop_id`, or the sandbox's own for this world (`adopted_shop_id`) where the profile adopts the handler and the
        store declared none; None where no Shop Pay handler is served."""
        entries = handler_entries(self._doc.payment_handlers.get(SHOP_PAY_NAMESPACE))
        for entry in entries:
            config = entry.get("config")
            found = config.get("shop_id") if isinstance(config, Mapping) else None
            if isinstance(found, str) and found:
                return found
        if SHOP_PAY_NAMESPACE in self._doc.profile.ucp_payment_handlers:
            shown = self._doc.policy.presented
            return shown.shop_id if shown is not None else adopted_shop_id(self.id)
        return None

    @property
    def platform_name(self) -> str | None:
        """The platform whose behaviour the store is served with: the pack's platform pack (D42), else the store's own
        platform as captured (`sandbox` for a twin cut from a catalog)."""
        return self._doc.policy.platform_pack or self._doc.store.platform

    @property
    def transplanted(self) -> bool:
        """Whether the store is served under another platform's pack than its own: its products and market stay, but
        nothing its own platform recorded (its ranking of searches, its tool list) is replayed."""
        pack = self._doc.policy.platform_pack
        return pack is not None and pack != (self._doc.store.platform or "")

    def failing_calls(self) -> dict[tuple[str, int], str]:
        """(tool, n) to the store error the episode's draw answers the n-th call of that tool in a session with (D70).
        Empty for a world, which draws nothing yet, and for a pack served under another platform's rules, whose
        store's errors were never seen in that platform's words."""
        found = self._doc.policy.store_errors
        if found is None or self.transplanted:
            return {}
        return {(c.tool, c.call): c.kind for c in found.calls}

    def coupons(self) -> dict[str, Coupon]:
        """Every code's rule: each percent code of `policy.discount_codes`, then the typed coupons over them."""
        pol = self._doc.policy
        out = {code: Coupon(code, "percent", int(pct)) for code, pct in pol.discount_codes.items()}
        for c in pol.coupons:
            out[c.code] = Coupon(c.code, c.kind, c.amount, c.individual_use, c.min_spend, c.priority, c.stop)
        return out

    def config(self) -> Config:
        d = self._doc
        rules = core_rules(self.platform_name)
        return Config(
            market=d.market.to_market(),
            discount_codes=MappingProxyType(dict(d.policy.discount_codes)),
            stack_discounts=d.policy.stack_discounts,
            default_tier=d.start.default_tier,
            default_permissions=tuple(d.start.default_permissions),
            reservation_ttl_ms=d.start.reservation_ttl_ms,
            tick_ms=d.start.tick_ms,
            handlers=MappingProxyType(self.handlers()),
            faults=compile_faults(d.faults),
            profile_id=d.profile.id,
            clamp_stock=rules.clamp_stock,
            coupons=MappingProxyType(self.coupons()),
            rules=rules,
            sold_out=frozenset(d.policy.sold_out.variants) if d.policy.sold_out is not None else frozenset(),
            psp=d.policy.psp,
            three_ds=ThreeDs(rail=d.policy.three_ds.rail, abandon=d.policy.three_ds.abandon)
            if d.policy.three_ds is not None
            else None,
            shop_id=self.shop_id(),
        )

    def initial_state(self, seed: int | None = None) -> State:
        """The state every episode on this pack starts from. `seed` overrides the pack's own (ids change with it)."""
        d = self._doc
        cur = d.snapshot.currency
        inventory = {
            vid: Variant(vid, v.product_id, v.title, v.price, cur, v.available, v.stock, 0, v.requires_selling_plan)
            for vid, v in d.snapshot.variants.items()
        }
        sessions = [
            Session(s.id, s.tier, tuple(sorted(set(s.permissions))), s.revoked, s.profile_id or d.profile.id)
            for s in d.start.sessions
        ]
        return initial_state(
            seed=d.start.seed if seed is None else seed,
            currency=cur,
            inventory=inventory,
            clock=d.start.clock_start_ms,
            sessions=sessions,
        )

    def public(self) -> dict[str, Any]:
        """The pack as the agent may see it: the fault schedule and every task's hidden spec are sealed."""
        d = self.doc
        d["faults"] = f"sealed ({len(d['faults'])})"
        for t in d["tasks"]:
            t["hidden_spec"] = "sealed"
        return d

    def derive(self, **changes: Any) -> EnvPack:
        """A new pack with parts replaced (`faults=[...]`, `profile=...`) or merged (`start={...}`, `policy={...}`).
        The old pack is untouched."""
        unknown = set(changes) - set(DERIVABLE)
        if unknown:
            raise EnvPackError(f"cannot derive {sorted(unknown)}", hint=f"derive one of {DERIVABLE}, or build anew")
        doc = self.doc
        for key, value in changes.items():
            try:
                part = _part(key, value, doc["snapshot"]["currency"])
            except pydantic.ValidationError as exc:
                raise _invalid(exc, key) from None
            doc[key] = {**doc[key], **part} if key in MERGED else part
        return _from_dict(doc)

    # -- bytes and files -----------------------------------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        return self._json.encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> EnvPack:
        """A pack from its canonical bytes. Refuses anything that is not exactly what `to_bytes` writes."""
        try:
            parsed = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EnvPackError(f"not JSON: {exc}", hint="an envpack file is written by EnvPack.save") from None
        pack = _from_dict(parsed)
        if pack.to_bytes() != data:
            raise EnvPackError(
                "the bytes are not the canonical form of this pack",
                hint="never edit a pack by hand; derive a new one with EnvPack.derive or build_envpack",
            )
        return pack

    def save(self, directory: Path) -> Path:
        """Write `<directory>/<id>.json`. Never overwrites a different file: the name is the content."""
        path = Path(directory) / f"{self.id}.json"
        if path.exists():
            if path.read_bytes() != self.to_bytes():
                raise EnvPackError(
                    f"{path} exists and does not hold {self.id}", hint="the file was edited or is corrupt; remove it"
                )
            return path
        atomic_write_bytes(path, self.to_bytes())
        return path

    @classmethod
    def load(cls, path: Path) -> EnvPack:
        p = Path(path)
        pack = cls.from_bytes(p.read_bytes())
        if p.stem != pack.id:
            raise EnvPackError(
                f"{p.name} does not hold the pack its name says (content is {pack.id})",
                hint="load packs only from files EnvPack.save wrote, under their own names",
            )
        return pack


def _from_dict(doc: object) -> EnvPack:
    if not isinstance(doc, Mapping) or doc.get("schema") != SCHEMA:
        found = doc.get("schema") if isinstance(doc, Mapping) else type(doc).__name__
        raise EnvPackError(f"not an {SCHEMA} document (schema {found!r})", hint="rebuild it with build_envpack")
    try:
        return EnvPack(EnvPackDoc.model_validate(doc))
    except pydantic.ValidationError as exc:
        raise _invalid(exc) from None


def _invalid(exc: pydantic.ValidationError, part: str = "") -> EnvPackError:
    """A pydantic refusal as a typed envpack error naming the first bad field."""
    first = exc.errors()[0]
    where = ".".join(str(p) for p in (part, *first["loc"]) if p != "") or "document"
    return EnvPackError(f"invalid envpack: {where}: {first['msg']}", hint="rebuild it with build_envpack")


def _part(key: str, value: Any, currency: str) -> Any:
    """A derivable part as the document stores it."""
    if key == "profile":
        return (value if isinstance(value, ProtocolProfile) else ProtocolProfile.model_validate(value)).model_dump(
            mode="json"
        )
    if key == "market":
        return _market_doc(value, currency).model_dump(mode="json")
    if key == "tasks":
        return _tasks(value)
    if key == "faults":
        compile_faults(value)
        return [dict(f) for f in value]
    if key == "policy":
        return _policy(value)
    return _json(value, key)


def _json(value: Any, part: str) -> Any:
    """`value` as plain canonical JSON, or a typed refusal naming the part that holds something JSON cannot."""
    try:
        return json.loads(dumps(value))
    except (TypeError, ValueError) as exc:
        hint = "pass strings, whole numbers, lists and objects"
        raise EnvPackError(f"{part} is not plain JSON: {exc}", hint=hint) from None


def _market_doc(market: Market | str, currency: str) -> MarketDoc:
    m = market if isinstance(market, Market) else market_for(market, currency)
    return MarketDoc.from_market(m)


def _task(t: TaskRef | Mapping[str, Any]) -> TaskRef:
    return t if isinstance(t, TaskRef) else TaskRef.model_validate(t)


def _tasks(tasks: Iterable[TaskRef | Mapping[str, Any]]) -> list[dict[str, Any]]:
    try:
        return [_task(t).model_dump(mode="json") for t in tasks]
    except pydantic.ValidationError as exc:
        raise _invalid(exc, "tasks") from None


def _policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(policy)
    if "discount_codes" in out:
        out["discount_codes"] = {str(k).strip().upper(): v for k, v in dict(out["discount_codes"]).items()}
    return out


def _store_doc(store: str | StoreRef | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(store, str):
        return {"domain": store}
    if isinstance(store, StoreRef):
        return {
            "domain": store.domain,
            "store_id": store.store_id,
            "platform": store.platform,
            "category": store.category,
        }
    return dict(store)


def _variants(variants: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for v in variants:
        vid = str(v["id"])
        if vid in out:
            raise EnvPackError(f"variant {vid!r} appears twice", hint="give each variant once in the snapshot")
        out[vid] = {
            "product_id": str(v.get("product_id") or ""),
            "title": str(v.get("title") or ""),
            "price": v["price"],
            "available": v.get("available", True),
            "stock": v.get("stock", 100),
            "requires_selling_plan": v.get("requires_selling_plan", False),
            "options": dict(v.get("options") or {}),
        }
    return out


def build_envpack(
    *,
    store: str | StoreRef | Mapping[str, Any],
    variants: Iterable[Mapping[str, Any]],
    currency: str,
    profile: ProtocolProfile | None = None,
    payment_handlers: Mapping[str, Any] | None = None,
    products: Mapping[str, Mapping[str, Any]] | None = None,
    market: Market | str = "US",
    tasks: Iterable[TaskRef | Mapping[str, Any]] = (),
    faults: Iterable[Mapping[str, Any]] = (),
    start: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
    snapshot_source: str = "fixture",
    captured_at: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    recorded: Iterable[Mapping[str, Any]] = (),
    tool_lists: Iterable[Mapping[str, Any]] = (),
) -> EnvPack:
    """Build and seal a v2 pack.

    `variants` are dicts with `id`, `price` (minor units of `currency`) and optionally `product_id`, `title`,
    `available`, `stock` (default 100), `requires_selling_plan` and `options`. `market` is a market code
    (`carttrace.sandbox.core.market.MARKETS`) or a `Market` row; a code is priced in `currency`. `payment_handlers` are the
    configs as the live store declared them; when given, the fake handler is served only if `policy` says
    `fake_handler: true`. `recorded` are a replica's catalog answers (`RecordedAnswer` fields), replayed by the
    sandbox's catalog. `tool_lists` are a replica's recorded tools/list replies (`RecordedToolList` fields), listed
    verbatim by its UCP MCP surface. The same arguments always give the same pack id.
    """
    cur = currency.upper()
    handlers = _json(dict(payment_handlers or {}), "payment_handlers")
    pol = _policy({"fake_handler": not handlers, **(policy or {})})
    faults_list = [dict(f) for f in faults]
    compile_faults(faults_list)
    doc = {
        "schema": SCHEMA,
        "store": _store_doc(store),
        "profile": (profile or DEFAULT_PROFILE).model_dump(mode="json"),
        "payment_handlers": handlers,
        "snapshot": {
            "currency": cur,
            "source": snapshot_source,
            "captured_at": captured_at,
            "variants": _variants(variants),
            "products": _json({str(k): dict(p) for k, p in (products or {}).items()}, "products"),
            "recorded": _json([dict(a) for a in recorded], "recorded"),
        },
        "market": _market_doc(market, cur).model_dump(mode="json"),
        "tasks": _tasks(tasks),
        "start": dict(start or {}),
        "policy": pol,
        "faults": faults_list,
        "provenance": _json(dict(provenance or {}), "provenance"),
        "tool_lists": _json([dict(t) for t in tool_lists], "tool_lists"),
    }
    return _from_dict(doc)

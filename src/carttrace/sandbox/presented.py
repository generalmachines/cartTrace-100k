"""The identity a sandbox store presents to agents (D106): one object per world, read by every surface.

Why. A stand-in of the open pools (D85, D91) stands in for a real roster store, and its bytes told the agent so: the
host `sandbox.invalid`, an MCP server named `mc-sandbox`, an ACP handler whose PSP is `sandbox` and whose merchant was
`pool-<category>-<seed>.open.sandbox.invalid`, an MPP realm of the same name and method `sandbox`, product ids
`gid://abo/...` that name the dataset, and the real store's domain in the agent's context. About sixteen modules
built such strings on their own. Now one `Presented` is resolved per world from data/sandbox/visibility.yaml when the
world is built (at manifest build), kept in the pack (`policy.presented`), and every surface reads it through
`presented(pack)`: the store's name and domain, its origin, its merchant id and realm, its MCP server, its card
handler, the PSP and MPP method names, the Shop Pay shop id, and the id scheme its products and variants carry.

A stand-in presents neutral, realistic values: a brand-like `.example` domain (`pools.stand_in_domain`), the pack's
own MCP server name, ids in the pack's own shape (Shopify gids, Medusa's prefixed ids, Saleor's GraphQL ids, numbers
elsewhere), Stripe as the ACP PSP (or the broker the pack names, D71), and a card handler under example.com. No value
names the sandbox, the category, the pool, its seed, the roster store or the source catalog. Every other world (a
replica, a twin of another catalog) presents what it always did: `presented(pack)` gives today's values.

The guard. `forbidden_tokens` lists a row's tokens by kind (the roster store's domain and id, the pool key, the
sandbox's own names, the source catalogs' id prefixes) and `Guard` checks a response's bytes against them: the
sandbox server refuses to serve a response that holds one and ends the row (`VisibilityLeak`, harness_error, with
the token's kind). It is a substring test over the lowercased bytes, plus a scan of the hosts in them for the
category slug.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

import yaml

from carttrace.core.errors import SafetyError, ValidationError
from carttrace.sandbox.core.config import FAKE_NAMESPACE, adopted_shop_id
from carttrace.sandbox.envpack.doc import EnvPackDoc, PresentedDoc
from carttrace.sandbox.envpack.pack import EnvPack

VISIBILITY_FILE: Final = Path(__file__).resolve().parents[3] / "data" / "sandbox" / "visibility.yaml"
IN_PROCESS_URL: Final = "http://sandbox.invalid"
"""What every other world calls itself; carttrace.sandbox.surfaces.client re-exports it."""
LEGACY_SERVER: Final = "mc-sandbox"
LEGACY_SESSION_HEADER: Final = "mc-sandbox-session"
LEGACY_PSP: Final = "sandbox"
LEGACY_MPP_METHOD: Final = "sandbox"
LEGACY_CARD_HANDLER_ID: Final = "fake"
IdScheme = Literal["source", "shopify", "numeric", "prefixed", "graphql"]
_OPEN_GID = re.compile(r"^gid://(?!shopify/)(?P<source>[a-z0-9_-]+)/(?P<kind>Product|ProductVariant)/(?P<rest>.+)$")
_SHOPIFY_GID = re.compile(r"^gid://shopify/(?P<kind>Product|ProductVariant)/\d+$")


@dataclass(frozen=True, slots=True)
class Presented:
    """What a store calls itself on every surface an agent reads."""

    store_name: str
    domain: str
    origin: str
    """The base every link a surface hands out is built on (`https://<domain>`, or `IN_PROCESS_URL`)."""
    merchant_id: str
    realm: str
    mcp_server: Mapping[str, str] | None
    """The MCP `serverInfo`; None keeps the platform pack's own, else the sandbox's legacy one."""
    session_header: str
    acp_psp: str
    mpp_method: str
    card_namespace: str
    """The namespace the card handler is listed under; internally it is always FAKE_NAMESPACE."""
    card_handler_id: str
    shop_id: str
    id_scheme: IdScheme
    served_names: Mapping[str, str]
    """Internal name to served name, rewritten at the server's boundary (empty: nothing is rewritten)."""
    market_names: Mapping[str, str]
    """The market's carrier and pickup location names to the ones the world serves (empty: the market's own)."""
    store_address: str | None
    """The address the store's policy pages give, or None."""
    neutral: bool
    """A stand-in's identity (resolved from visibility.yaml); False for every other world's legacy one."""

    @property
    def vault_source(self) -> str:
        """The `source` an ACP vault token's metadata names: the store's payment processor, never the sandbox's."""
        return "payment_processor" if self.neutral else "sandbox_psp"

    def card_note(self) -> dict[str, Any]:
        """The card handler's config as it is served."""
        if self.neutral:
            return {"livemode": False}
        return {"livemode": False, "note": "the sandbox's own handler; it charges nobody"}


def presented(pack: EnvPack) -> Presented:
    """The identity `pack` presents: its own (`policy.presented`), else the legacy one every pack had."""
    doc = pack.presented_doc
    if doc is not None:
        return Presented(
            store_name=doc.store_name,
            domain=doc.domain,
            origin=doc.origin,
            merchant_id=doc.merchant_id,
            realm=doc.realm,
            mcp_server=dict(doc.mcp_server) if doc.mcp_server else None,
            session_header=doc.session_header,
            acp_psp=doc.acp_psp,
            mpp_method=doc.mpp_method,
            card_namespace=doc.card_namespace,
            card_handler_id=doc.card_handler_id,
            shop_id=doc.shop_id,
            id_scheme=doc.id_scheme,
            served_names=dict(doc.served_names),
            market_names=dict(doc.market_names),
            store_address=doc.store_address,
            neutral=True,
        )
    domain = pack.store.domain
    return Presented(
        store_name=domain,
        domain=domain,
        origin=IN_PROCESS_URL,
        merchant_id=domain,
        realm=domain,
        mcp_server=None,
        session_header=LEGACY_SESSION_HEADER,
        acp_psp=LEGACY_PSP,
        mpp_method=LEGACY_MPP_METHOD,
        card_namespace=FAKE_NAMESPACE,
        card_handler_id=LEGACY_CARD_HANDLER_ID,
        shop_id=adopted_shop_id(pack.id),
        id_scheme="source",
        served_names={},
        market_names={},
        store_address=None,
        neutral=False,
    )


# -- the contract ------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=2)
def visibility(path: Path = VISIBILITY_FILE) -> dict[str, Any]:
    """data/sandbox/visibility.yaml, checked for the parts the code reads."""
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"{path} does not read: {exc}") from None
    if not isinstance(doc, dict) or doc.get("schema") != "mc.visibility.v1":
        raise ValidationError(f"{path} is not an mc.visibility.v1 file")
    for key in ("audiences", "forbidden", "identity"):
        if not isinstance(doc.get(key), dict):
            raise ValidationError(f"{path}: `{key}` must be a mapping")
    return doc


def _per_pack(table: Mapping[str, Any], pack_name: str | None) -> Any:
    return table.get(pack_name or "generic", table.get("default"))


OpenScheme = Literal["shopify", "numeric", "prefixed", "graphql"]


def id_scheme_for(platform_pack: str | None) -> OpenScheme:
    """The id scheme a stand-in served under `platform_pack` gives its products and variants."""
    found = _per_pack(visibility()["identity"]["id_scheme"], platform_pack)
    if found not in ("shopify", "numeric", "prefixed", "graphql"):
        raise ValidationError(f"visibility.yaml names an unknown id scheme {found!r}")
    return found  # type: ignore[no-any-return]


def stand_in_identity(domain: str, platform_pack: str | None, *, psp: str | None, has_server: bool) -> PresentedDoc:
    """A stand-in's presented identity: its `.example` domain and the per-pack values of visibility.yaml. A pack
    whose spec names its own MCP server (`has_server`, WooCommerce's) keeps it."""
    identity = visibility()["identity"]
    server = None if has_server else dict(_per_pack(identity["mcp_server"], platform_pack))
    handler = identity["card_handler"]
    words = domain.split(".", 1)[0].replace("-and-co", " & co").split("-")
    return PresentedDoc(
        store_name=" ".join(w.capitalize() if w != "&" else w for w in words),
        domain=domain,
        origin=f"https://{domain}",
        merchant_id=domain,
        realm=domain,
        mcp_server={str(k): str(v) for k, v in (server or {}).items()},
        session_header=str(identity["session_header"]),
        acp_psp=psp or str(identity["acp_psp"]),
        mpp_method=str(identity["mpp_method"]),
        card_namespace=str(handler["namespace"]),
        card_handler_id=str(handler["id"]),
        shop_id=str(identity["shop_id"]),
        id_scheme=id_scheme_for(platform_pack),
        version=str(visibility()["version"]),
        served_names={str(k): str(v) for k, v in (identity.get("served_names") or {}).items()},
        market_names={str(k): str(v) for k, v in (identity.get("market_names") or {}).items()},
    )


# -- ids ---------------------------------------------------------------------------------------------------------------


def _digits(open_id: str, n: int) -> int:
    h = int(hashlib.sha256(f"presented-id\n{open_id}".encode()).hexdigest()[:16], 16)
    low = 10 ** (n - 1)
    return int(low + h % (9 * low))


def presented_id(open_id: str, scheme: IdScheme) -> str:
    """The id an open product or variant id (`gid://abo/Product/B07...`) is presented as under `scheme`: stable, from
    the open id's sha256. Any other id is returned as it is."""
    match = _OPEN_GID.match(open_id)
    if match is None or scheme == "source":
        return open_id
    kind = match.group("kind")
    if scheme == "shopify":
        return f"gid://shopify/{kind}/{_digits(open_id, 13)}"
    if scheme == "numeric":
        return str(_digits(open_id, 9))
    if scheme == "prefixed":
        alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
        h = int(hashlib.sha256(f"presented-id\n{open_id}".encode()).hexdigest(), 16)
        tail = "".join(alphabet[(h >> (5 * i)) & 31] for i in range(26))
        return f"{'prod' if kind == 'Product' else 'variant'}_01{tail[2:]}"
    raw = f"{kind}:{_digits(open_id, 9)}".encode()
    return base64.b64encode(raw).decode()


def presented_ids_in(value: Any, scheme: IdScheme) -> Any:
    """`value` (a task, a fault, any JSON) with every open product and variant id in it, as a whole string, presented
    under `scheme`: a row on a stand-in reads its task in its world's ids."""
    if scheme == "source":
        return value
    if isinstance(value, str):
        return presented_id(value, scheme) if _OPEN_GID.match(value) else value
    if isinstance(value, list):
        return [presented_ids_in(v, scheme) for v in value]
    if isinstance(value, Mapping):
        return {k: presented_ids_in(v, scheme) for k, v in value.items()}
    return value


def with_presented(pack: EnvPack, doc: PresentedDoc) -> EnvPack:
    """`pack` presenting `doc`: its products and variants under the doc's id scheme (each product's open id kept in
    its `provenance.source_id`, the whole map in `policy.presented.ids`) and the identity in `policy.presented`.
    Raises ValidationError when two ids present as one."""
    data = pack.doc
    snapshot = data["snapshot"]
    ids: dict[str, str] = {}
    for open_id in [*snapshot["products"], *snapshot["variants"]]:
        new = presented_id(open_id, doc.id_scheme)
        if new in ids and ids[new] != open_id:
            raise ValidationError(f"{open_id} and {ids[new]} present as one id {new}")
        ids[new] = open_id
    moved = {v: k for k, v in ids.items()}
    products: dict[str, Any] = {}
    for open_id, raw in sorted(snapshot["products"].items()):
        new = moved[open_id]
        facts = {**raw, "id": new} if "id" in raw or new != open_id else dict(raw)
        provenance = facts.get("provenance")
        if new != open_id:
            facts["provenance"] = {**(provenance if isinstance(provenance, Mapping) else {}), "source_id": open_id}
        products[new] = facts
    variants = {
        moved[vid]: {**v, "product_id": moved.get(v.get("product_id"), v.get("product_id"))}
        for vid, v in sorted(snapshot["variants"].items())
    }
    snapshot["products"], snapshot["variants"] = products, variants
    names = dict(doc.market_names)
    market = data["market"]
    for option in market.get("shipping") or ():
        option["carrier"] = names.get(option.get("carrier"), option.get("carrier"))
        option["title"] = names.get(option.get("title"), option.get("title"))
    for place in market.get("pickup") or ():
        place["title"] = names.get(place.get("title"), place.get("title"))
    data["tasks"] = presented_ids_in(data.get("tasks") or [], doc.id_scheme)
    data["faults"] = presented_ids_in(data.get("faults") or [], doc.id_scheme)
    kept = {k: v for k, v in sorted(ids.items()) if k != v}
    addresses = visibility()["identity"].get("store_address") or {}
    address = addresses.get(market.get("code"), addresses.get("default"))
    full = doc.model_copy(update={"ids": kept, "store_address": str(address) if address else None})
    data["policy"] = {**data["policy"], "presented": full.model_dump(mode="json")}
    return EnvPack(EnvPackDoc.model_validate(data))


def source_ids(pack: EnvPack) -> dict[str, str]:
    """Each presented product and variant id of `pack` to the open source's own id; empty when it presents none."""
    doc = pack.presented_doc
    return dict(doc.ids) if doc is not None else {}


def rename(data: bytes, names: Mapping[str, str]) -> bytes:
    """`data` with every JSON string that is exactly one of `names` replaced by its counterpart, also inside a JSON
    document carried as a string (an MCP tool result's text, where the quotes are escaped)."""
    for old, new in names.items():
        data = data.replace(f'"{old}"'.encode(), f'"{new}"'.encode())
        data = data.replace(f'\\"{old}\\"'.encode(), f'\\"{new}\\"'.encode())
    return data


def unreversible(data: bytes, names: Mapping[str, str]) -> str | None:
    """The first served name (a value of `names`) that `data`, a body the core produced, already holds as a JSON
    string, or None. The boundary's rewrite is undone on the way back in (a request's served names become the internal
    ones), so a served name the core emits itself would come back as an internal name it never wrote: a signed
    document the agent echoes (AP2 v0.1.0's cart mandate) would no longer be the one signed (D109)."""
    for new in names.values():
        if f'"{new}"'.encode() in data or f'\\"{new}\\"'.encode() in data:
            return new
    return None


class UnreversibleName(SafetyError):
    """A served name of the boundary's rewrite is a string the core emits on its own (D109): the rewrite could not be
    undone on the way back in, so the response is not served."""

    def __init__(self, name: str, where: str) -> None:
        super().__init__(
            f"server boundary: {where} already holds the served name {name!r}, so a request echoing it would not "
            "reach the core as it was sent",
            hint="serve that value through carttrace.sandbox.presented, or give visibility.yaml's served_names a name "
            "nothing else emits",
        )
        self.name = name
        self.where = where


# -- the guard ---------------------------------------------------------------------------------------------------------


class VisibilityLeak(SafetyError):
    """A response held a token its row's agent may not see (D106): it was not served, and the row ends."""

    def __init__(self, kind: str, where: str) -> None:
        super().__init__(
            f"visibility guard: a {kind} token in {where} was refused before it was served",
            hint="present the value through carttrace.sandbox.presented, never format it in the surface",
        )
        self.kind = kind
        self.where = where


@dataclass(frozen=True, slots=True)
class Guard:
    """A row's forbidden tokens, by kind, and the category slug no served host may hold."""

    tokens: tuple[tuple[str, str], ...]
    category: str | None = None
    hits: list[tuple[str, str]] = field(default_factory=list)
    """Every refusal, (kind, where), so the episode can end the row after the surface's own error handling."""

    def find(self, data: bytes | str) -> str | None:
        """The kind of the first forbidden token in `data`, or None."""
        text = (data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data).lower()
        if '\\"' in text:
            text = text.replace('\\"', '"')  # a JSON document carried as a string (an MCP tool result's text)
        for token, kind in self.tokens:
            if token in text:
                return kind
        if self.category:
            for match in _HOST.finditer(text):
                host = match.group(1)
                if host.endswith((".example", ".invalid")) and self.category in host:
                    return "category_host"
        return None

    def check(self, data: bytes | str, where: str) -> None:
        kind = self.find(data)
        if kind is not None:
            self.hits.append((kind, where))
            raise VisibilityLeak(kind, where)


_HOST = re.compile(r"[a-z][a-z0-9+.-]*://([a-z0-9.-]+)")


def forbidden_tokens(
    *, roster_domain: str | None, roster_id: str | None, category: str | None, seed: int | None
) -> Guard:
    """The guard of a row on a stand-in: the tokens of visibility.yaml's `forbidden`, filled from its roster store
    and its world."""
    table = visibility()["forbidden"]
    out: list[tuple[str, str]] = []

    def add(kind: str, tokens: Iterable[str]) -> None:
        out.extend((t.lower(), kind) for t in tokens if t)

    if roster_domain:
        bare = roster_domain.lower().removeprefix("www.")
        add("roster_domain", {roster_domain, bare})
    if roster_id:
        add("roster_id", [roster_id])
    if category is not None and seed is not None:
        add("pool_key", [f"pool-{category}-{seed}", "open:pool-"])
    add("sandbox", [str(t) for t in table["sandbox"]])
    add("source_id", [str(t) for t in table["source_id"]])
    return Guard(tokens=tuple(dict.fromkeys(out)), category=category)

"""Replicas: a real store's sandbox, built from a capture of the store itself (decision D12).

Why a catalog source of its own. A catalog twin is cut from a licensed public catalog and answers how agents behave in a
category; a replica answers how this store performs. `build_replica(capture, market=...)` turns a store capture
(`carttrace.trace.capture`, made by `carttrace live capture`) into the StoreSnapshot an envpack holds:
- the store's real products and variants as captured: ids, titles, prices in minor units of the store's currency,
  options, availability, descriptions, media, metadata, every field the store sent, each product's facts being the
  store's own document (get_product's reply completed with what only search results carried);
- the shape the store gave each view (`_listing`, `_detail`: the fields a search listing or get_product carried, their
  values where they differ, how many variants a listing showed) and the policies it attached to each product
  (`_policies`), so the sandbox's UCP surface answers the way the store did (carttrace.sandbox.surfaces.ucp.catalog);
- the catalog answers the store gave during the capture (`recorded`): which products a search returned in which
  order with which variants and which pagination, and which selection and variants a get_product returned, so the
  replica answers the captured requests with equivalent documents;
- the store's captured ProtocolProfile and payment handlers, so every surface mounts at the store's own versions;
- provenance naming the capture and its rights: a live store's (D10), training and publication false, and for
  Shopify raw content that expires after live.retention_days_raw days (`carttrace collect purge` removes replica packs).

`ReplicaSource` is the `sandbox.catalog` block named "replica": it has the CatalogSource shape, reads the newest
capture of each store under <paths.var>/captures, and is never fetched (a capture is taken live, read-only and polite,
by `carttrace live capture`). `replica_fidelity` (carttrace.sandbox.catalogs.fidelity) checks a replica against its capture.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import httpx

from carttrace.config.schema import Settings
from carttrace.core.canon import dumps
from carttrace.core.clock import Clock
from carttrace.sandbox.catalogs.download import Progress
from carttrace.sandbox.catalogs.errors import CatalogError
from carttrace.sandbox.catalogs.licence import Licence
from carttrace.sandbox.catalogs.records import ProductRecord, Provenance, Rating, VariantRecord
from carttrace.sandbox.catalogs.settings import CatalogSettings, resolve
from carttrace.sandbox.catalogs.snapshot import StoreSnapshot
from carttrace.sandbox.catalogs.source import FetchReport
from carttrace.sandbox.catalogs.text import clean
from carttrace.sandbox.core.market import Market, market_for
from carttrace.sandbox.envpack import EnvPack
from carttrace.sandbox.envpack.doc import MarketDoc
from carttrace.sandbox.surfaces.ucp.catalog import (
    DETAIL,
    LISTING,
    POLICIES,
    STATE_FIELDS,
    detail_request,
    search_request,
)
from carttrace.trace.capture import CapturedProduct, StoreCapture, load_capture, newest_captures
from carttrace.trace.products import RESPONSE_ONLY, merge_documents, product_documents, reply_payload
from carttrace.trace.rights import LIVE_BASIS

REPLICA: Final = "replica"
SEARCH_TOOL = "search_catalog"
DETAIL_TOOL = "get_product"
LIVE_LICENCE = Licence(
    spdx="LicenseRef-live-store-terms",
    name="A live store's content, under its platform's terms (D10, D12)",
    commercial_use=True,
    training_allowed=False,
    publication_allowed=False,
    verified="2026-10-11: docs/DECISIONS.md D10 and D12",
    basis=f"{LIVE_BASIS}; each capture carries the rights of its own store's platform (Shopify: shopify_api)",
)
"""What a replica may be used for, before each capture's own rights narrow it: inference and internal benchmarking,
never training and never publication."""
SIGNALS = ("available", "exists")
"""get_product's per-response availability signals on option values; a replica derives them per request."""
DEFAULT_MARKET = "US"
PACKS_DIR = "envpacks"
"""Where replica packs are saved under paths.var: beside the packs the collector's manifests name (carttrace.collect)."""
DEFAULT_SHELF = 100
"""Stock a captured variant for sale is given: live stores do not publish stock (as `SnapshotVariant.stock`)."""
_KEPT = STATE_FIELDS | set(RESPONSE_ONLY)
"""Fields a view's own values are never recorded for: they come from the state or the request."""
_PRODUCTS_AT = re.compile(r"^\$\.products\[(\d+)\]")
_PRODUCT_AT = re.compile(r"^\$\.product(?![A-Za-z_])")


def build_replica(capture: StoreCapture | Path, *, market: str | Market) -> StoreSnapshot:
    """The store snapshot of a capture's replica, selling in `market` (a market code, or a Market row) priced in the
    store's own currency. Raises CatalogError when the capture holds no product."""
    cap = capture if isinstance(capture, StoreCapture) else load_capture(Path(capture))
    if not cap.products or cap.currency is None:
        raise CatalogError(
            f"capture {cap.id} holds no products, so it makes no replica",
            hint="capture the store again with other queries or a larger budget (carttrace live capture --products N)",
        )
    row = market if isinstance(market, Market) else market_for(market, cap.currency)
    shapes = _shapes(cap)
    policies = _policies(cap)
    variants: list[dict[str, Any]] = []
    products: dict[str, dict[str, Any]] = {}
    for product in cap.products:
        facts = store_facts(product)
        facts.update(shapes.get(product.id, {}))
        if policies.get(product.id):
            facts[POLICIES] = policies[product.id]
        products[product.id] = facts
        variants += [
            {
                "id": v.id,
                "product_id": product.id,
                "title": v.title,
                "price": v.price,
                "available": v.available,
                "options": {o.name: o.label for o in v.options},
            }
            for v in product.variants
        ]
    recorded = recorded_answers(cap)
    provenance = {
        "capture": {
            "id": cap.id,
            "store_id": cap.store.store_id,
            "domain": cap.store.domain,
            "category": cap.category,
            "captured_at": cap.captured_at,
            "ended": cap.ended,
            "queries": list(cap.queries),
            "products": len(cap.products),
            "calls": len(cap.calls),
            "recorded": len(recorded),
        },
        "rights": cap.rights.model_dump(mode="json"),
    }
    return StoreSnapshot(
        source=REPLICA,
        category=cap.category,
        store_key=cap.store.store_id,
        seed=0,
        size=len(cap.products),
        store={
            "domain": cap.store.domain,
            "store_id": cap.store.store_id,
            "platform": cap.store.platform or "unknown",
            "category": cap.category,
        },
        currency=cap.currency,
        market=MarketDoc.from_market(row),
        variants=variants,
        products=products,
        provenance=provenance,
        rights=cap.rights,
        profile=cap.profile,
        payment_handlers=dict(cap.payment_handlers),
        recorded=recorded,
        captured_at=cap.captured_at,
        snapshot_source=f"capture:{cap.id}",
    )


def replica_market(capture: StoreCapture) -> str:
    """The market a replica sells in when none is named: the one the capture asked for, else the store's, else US."""
    return capture.market or capture.store.market or DEFAULT_MARKET


def replica_envpack(capture: StoreCapture | Path, *, market: str | Market, **extra: Any) -> EnvPack:
    """The sealed envpack of a capture's replica. `extra` adds build_envpack arguments (tasks, faults, start, policy,
    another profile for an experiment); provenance is merged under the replica's own keys."""
    return build_replica(capture, market=market).to_envpack(**extra)


def store_facts(product: CapturedProduct) -> dict[str, Any]:
    """A captured product's facts for its replica: the store's document without what describes one response (the
    effective selection, lookup correlation, option availability signals), with only the variants the capture kept
    (those with a price in the store's currency), in the store's order."""
    doc = product.document
    kept = {v.id for v in product.variants}
    facts = {k: v for k, v in doc.items() if k not in RESPONSE_ONLY and not k.startswith("_")}
    facts["variants"] = [
        {k: v for k, v in variant.items() if k not in RESPONSE_ONLY}
        for variant in doc.get("variants") or ()
        if isinstance(variant, Mapping) and variant.get("id") in kept
    ]
    if isinstance(doc.get("options"), list):
        facts["options"] = [_without_signals(o) for o in doc["options"] if isinstance(o, Mapping)]
    return facts


def recorded_answers(capture: StoreCapture) -> list[dict[str, Any]]:
    """The catalog answers a capture recorded, as envpack `RecordedAnswer` fields: each successful search_catalog and
    get_product, keyed by its request as the replica matches it, the first answer winning for a repeated request."""
    out: dict[str, dict[str, Any]] = {}
    for call in capture.successful(SEARCH_TOOL, DETAIL_TOOL):
        payload = reply_payload(call.reply)
        if payload is None or _failed(payload):
            continue
        catalog = call.arguments.get("catalog")
        request = catalog if isinstance(catalog, Mapping) else {}
        docs = product_documents(payload)
        listed = [{"id": d["id"], "variants": _variant_ids(d)} for d in docs]
        if call.tool == SEARCH_TOOL:
            answer: dict[str, Any] = {"tool": SEARCH_TOOL, "request": search_request(request), "products": listed}
            pagination = payload.get("pagination")
            if isinstance(pagination, Mapping):
                answer["pagination"] = dict(pagination)
        else:
            if not docs:
                continue
            answer = {"tool": DETAIL_TOOL, "request": detail_request(request), "products": listed[:1]}
            selected = docs[0].get("selected")
            if isinstance(selected, list):
                answer["selected"] = [
                    {"name": s["name"], "label": s["label"]}
                    for s in selected
                    if isinstance(s, Mapping) and isinstance(s.get("name"), str) and isinstance(s.get("label"), str)
                ]
            if isinstance(docs[0].get("options"), list):
                answer["options"] = [dict(o) for o in docs[0]["options"] if isinstance(o, Mapping)]
        out.setdefault(dumps([answer["tool"], answer["request"]]), answer)
    return list(out.values())


def _shapes(capture: StoreCapture) -> dict[str, dict[str, Any]]:
    """Per product, the shape the store gave it in search listings and in get_product replies."""
    views: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for call in capture.successful(SEARCH_TOOL, DETAIL_TOOL):
        payload = reply_payload(call.reply)
        if payload is None or _failed(payload):
            continue
        view = LISTING if call.tool == SEARCH_TOOL else DETAIL
        for doc in product_documents(payload):
            views.setdefault(doc["id"], {}).setdefault(view, []).append(doc)
    by_id = {p.id: p for p in capture.products}
    out: dict[str, dict[str, Any]] = {}
    for pid, seen in views.items():
        product = by_id.get(pid)
        if product is None:
            continue
        full = store_facts(product)
        out[pid] = {view: _shape(docs, full, view) for view, docs in sorted(seen.items())}
    return out


def _shape(docs: Sequence[Mapping[str, Any]], full: Mapping[str, Any], view: str) -> dict[str, Any]:
    fields: set[str] = set()
    variant_fields: set[str] = set()
    value_fields: set[str] = set()
    most = 0
    for doc in docs:
        fields |= {str(k) for k in doc}
        variants = [v for v in doc.get("variants") or () if isinstance(v, Mapping)]
        variant_fields |= {str(k) for v in variants for k in v}
        most = max(most, len(variants))
        for option in doc.get("options") or ():
            if isinstance(option, Mapping):
                value_fields |= {str(k) for v in option.get("values") or () if isinstance(v, Mapping) for k in v}
    shown = merge_documents(docs[0], *docs[1:])
    values = {k: v for k, v in shown.items() if k not in _KEPT and full.get(k) != v}
    full_variants = {str(v.get("id")): v for v in full.get("variants") or () if isinstance(v, Mapping)}
    variant_values: dict[str, dict[str, Any]] = {}
    for variant in shown.get("variants") or ():
        mine = full_variants.get(str(variant.get("id")))
        if mine is None:
            continue
        differs = {k: v for k, v in variant.items() if k not in _KEPT and mine.get(k) != v}
        if differs:
            variant_values[str(variant["id"])] = differs
    shape: dict[str, Any] = {"fields": sorted(fields), "variant_fields": sorted(variant_fields)}
    if value_fields:
        shape["option_value_fields"] = sorted(value_fields)
    if values:
        shape["values"] = values
    if variant_values:
        shape["variant_values"] = variant_values
    if view == LISTING:
        shape["max_variants"] = most
    return shape


def _policies(capture: StoreCapture) -> dict[str, list[dict[str, Any]]]:
    """Per product, the policies the store's catalog replies attached to it (by `applies_to`, or to every product of a
    reply when a policy names no target), each once, without its targets."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for call in capture.successful(SEARCH_TOOL, DETAIL_TOOL):
        payload = reply_payload(call.reply)
        if payload is None or _failed(payload):
            continue
        ids = [d["id"] for d in product_documents(payload)]
        for policy in payload.get("policies") or ():
            if not isinstance(policy, Mapping):
                continue
            body = {k: v for k, v in policy.items() if k != "applies_to"}
            for pid in covered_products(policy.get("applies_to"), ids, single="product" in payload):
                out.setdefault(pid, {}).setdefault(dumps(body), body)
    return {pid: list(found.values()) for pid, found in out.items()}


def covered_products(targets: Any, ids: Sequence[str], *, single: bool) -> list[str]:
    """The products a policy's `applies_to` names in one catalog reply whose products are `ids` (`single` for a
    get_product reply, whose one product is `$.product`): every product when it names none."""
    if not isinstance(targets, list) or not targets:
        return list(ids)
    out: list[str] = []
    for target in targets:
        if not isinstance(target, str):
            continue
        at = _PRODUCTS_AT.match(target)
        if at is not None:
            out += [ids[int(at.group(1))]] if int(at.group(1)) < len(ids) else []
        elif single and _PRODUCT_AT.match(target) and ids:
            out.append(ids[0])
        else:
            out += list(ids)  # a wildcard, a filter or the root: every product, over-crediting rather than hiding
    return list(dict.fromkeys(out))


def _failed(payload: Mapping[str, Any]) -> bool:
    ucp = payload.get("ucp")
    return isinstance(ucp, Mapping) and ucp.get("status") == "error"


def _variant_ids(doc: Mapping[str, Any]) -> list[str]:
    return [str(v["id"]) for v in doc.get("variants") or () if isinstance(v, Mapping) and isinstance(v.get("id"), str)]


def _without_signals(option: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(option)
    values = option.get("values")
    if isinstance(values, list):
        out["values"] = [
            {k: v for k, v in value.items() if k not in SIGNALS} if isinstance(value, Mapping) else value
            for value in values
        ]
    return out


# -- the catalog source ------------------------------------------------------------------------------------------------


class ReplicaSource:
    """The `sandbox.catalog` block "replica": captured stores, read from <paths.var>/captures. Never fetched."""

    def __init__(self, *, settings: CatalogSettings) -> None:
        self.settings = settings

    @property
    def name(self) -> str:
        return REPLICA

    @property
    def licence(self) -> Licence:
        return LIVE_LICENCE

    @property
    def root(self) -> Path | None:
        return self.settings.captures_dir

    def fetch(
        self,
        dest: Path | None = None,
        *,
        relock: bool = False,
        local: Path | None = None,
        transport: httpx.BaseTransport | None = None,
        clock: Clock | None = None,
        progress: Progress | None = None,
    ) -> FetchReport:
        """Refused: a replica is captured from its store, live and read-only, never downloaded."""
        raise CatalogError(
            "a replica is captured from its own store, not fetched",
            hint="run `carttrace live capture DOMAIN --category SLUG --live`, then `carttrace sandbox replica build CAPTURE`",
        )

    def captures(self, category_slug: str | None = None) -> list[Path]:
        """The newest capture with products of each store under the captures root (of one category when given)."""
        if self.root is None:
            return []
        return list(newest_captures(self.root, category=category_slug).values())

    def iter_products(self, category_slug: str | None = None) -> Iterator[ProductRecord]:
        """Every product of each store's newest capture, as normalised records (prices as captured; stock is not
        published, so a variant for sale gets the sandbox's usual shelf)."""
        for path in self.captures(category_slug):
            cap = load_capture(path)
            for product in cap.products:
                yield product_record(cap, product)


def open_replica(*, settings: Settings | CatalogSettings | None = None) -> ReplicaSource:
    """The replica source over the configured captures (<paths.var>/captures)."""
    return ReplicaSource(settings=resolve(settings))


def product_record(capture: StoreCapture, product: CapturedProduct) -> ProductRecord:
    """A captured product as the normalised record every catalog source yields."""
    doc = product.document
    variant_docs = {str(v.get("id")): v for v in doc.get("variants") or () if isinstance(v, Mapping)}
    variants = []
    for v in product.variants:
        raw = variant_docs.get(v.id, {})
        barcodes = [b.get("value") for b in raw.get("barcodes") or () if isinstance(b, Mapping)]
        variants.append(
            VariantRecord(
                id=v.id,
                title=v.title or v.id,
                options={o.name: o.label for o in v.options},
                price=v.price,
                currency=v.currency,
                available=v.available,
                stock=DEFAULT_SHELF if v.available else 0,
                sku=raw.get("sku") if isinstance(raw.get("sku"), str) else None,
                barcode=next((b for b in barcodes if isinstance(b, str)), None),
            )
        )
    categories = [c.get("value") for c in doc.get("categories") or () if isinstance(c, Mapping)]
    path = next((c for c in categories if isinstance(c, str)), None)
    raw_metadata = doc.get("metadata")
    metadata: Mapping[str, Any] = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    return ProductRecord(
        id=product.id,
        source=REPLICA,
        category=capture.category,
        category_path=[part.strip() for part in path.split(">")] if path else [],
        store=capture.store.store_id,
        title=product.title or product.id,
        description=_plain(doc.get("description")),
        attributes={str(k): str(v) for k, v in metadata.items() if isinstance(v, str | int | float | bool)},
        variants=variants,
        rating=_rating(doc.get("rating")),
        tags=[t for t in doc.get("tags") or () if isinstance(t, str)],
        images=[m["url"] for m in doc.get("media") or () if isinstance(m, Mapping) and isinstance(m.get("url"), str)],
        provenance=Provenance(
            source=REPLICA,
            licence=LIVE_LICENCE.spdx,
            upstream_ids=[capture.id, product.id],
            upstream_url=doc.get("url") if isinstance(doc.get("url"), str) else None,
        ),
    )


def _plain(description: Any) -> str:
    if isinstance(description, str):
        return clean(description, limit=20_000)
    if isinstance(description, Mapping):
        for key in ("plain", "markdown", "html"):
            if isinstance(description.get(key), str) and description[key]:
                return clean(description[key], limit=20_000)
    return ""


def _rating(raw: Any) -> Rating | None:
    if not isinstance(raw, Mapping):
        return None
    value, top, count = raw.get("value"), raw.get("scale_max", 5), raw.get("count")
    if not isinstance(value, int | float) or not isinstance(top, int | float) or value < 0 or top < 1:
        return None
    return Rating(value=float(value), scale_max=float(top), count=count if isinstance(count, int) else None)

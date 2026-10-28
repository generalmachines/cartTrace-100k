"""Search memory: what a store's own search answered on live rows, kept per store so its replica answers the same.

Why. The paired runs of 2026-10-12 and 13 (var/live/pairs-01/report.md, `carttrace bench pairs`) agreed on outcome in 60%
of 683 pairs; 122 of the 273 disagreements began with the replica's search returning other products than the live
store for the same query, 35 with none in common. A store's ranking cannot be recomputed from product text, and a
capture asks only a handful of queries, but every live row asks the store more: 2,871 live search_catalog calls in
the bundles of those two days. So every live search a bundle recorded is kept here, and the replica built with it
replays them (D39):
- `answers`: per request as a replica matches it (carttrace.sandbox.surfaces.ucp.catalog `search_request`: the query
  folded, the filters, the page size, the cursor), the products the store listed in its order, the variants it
  listed with each, and its pagination with the store's own cursor, so the next page the agent asks for is the
  store's next page. When live answered the same request more than once, the latest answer wins;
- `products`: the store's listing document of every product an answer lists, the latest one seen. A replica is cut
  from a capture that keeps at most 500 products (capture v2 keeps 500 of 25,000 at store-004), and on 2026-10-13 a
  third of what live search listed was outside it, so a replayed answer could not show it. The replica adds those
  products from their listing documents, marked with `SOURCE_KEY`, so they are found and sold but never drawn as a
  task's product (carttrace.agents.products leaves them out of fixtures). A product the capture holds keeps its capture
  facts, with two things only the live wire shows (D55): its listing shows the media the live listing showed (live
  Shopify lists one image with its alt text, which the capture's feeds often lack: store-048 ep-7f9dd7b61780 seq 9,
  "Cara Loafer - Brown Sugar Leopard Suede Side View" in the listing, no alt on any of the feed's six images), and
  its facts gain the store's taxonomy category beside the capture's collections.

A memory is canonical JSON whose id is `sm-<sha12>` of its content, saved as <paths.var>/live/search-memory/<store>/
<id>.json, so a manifest names the memory its replica worlds were built with and a rebuilt world is the same world.
Nothing here reaches a live store: it reads bundles already written, whose wire records were redacted at write time.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from carttrace.core.canon import digest, dumps, loads
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.files import atomic_write_text
from carttrace.sandbox.catalogs.snapshot import StoreSnapshot
from carttrace.sandbox.envpack.doc import RecordedAnswer
from carttrace.sandbox.surfaces.ucp.catalog import LISTING, POLICIES, search_request
from carttrace.trace.bundle_index import live_bundles
from carttrace.trace.layout import EPISODES_PREFIX, TraceStore
from carttrace.trace.products import product_documents, reply_payload
from carttrace.trace.reader import Bundle
from carttrace.trace.schema import WireRecord

SCHEMA: Final = "mc.search-memory.v1"
ID_PREFIX: Final = "sm-"
MEMORY_DIR: Final = Path("live") / "search-memory"
"""Where memories are saved under paths.var: one directory per store, one file per memory id."""
SOURCE_KEY: Final = "_search_memory"
"""The facts key of a product a replica holds only because live search listed it; its value is the memory's id."""
SEARCH_TOOL: Final = "search_catalog"


class SearchMemory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: Literal["mc.search-memory.v1"] = Field(default=SCHEMA, alias="schema")
    id: str
    store_id: str
    domains: list[str]
    """The store's hosts as its live episodes named them (lowercase, without www.), sorted: a capture filed under the
    host rather than the queue id is the same store's."""
    episodes: list[str]
    """The live episodes whose searches it holds, sorted."""
    days: list[str]
    """The days (YYYY-MM-DD of an episode's start) its searches were asked, sorted."""
    answers: list[RecordedAnswer]
    """One per request, sorted by request."""
    products: dict[str, dict[str, Any]]
    """Listing document by product id, for every product an answer lists."""


@dataclass(frozen=True, slots=True)
class LiveSearch:
    """One search_catalog call a live bundle recorded, with the store's answer."""

    store_id: str
    domain: str
    episode_id: str
    started_at: str
    seq: int
    request: dict[str, Any]
    """As a replica matches it (`search_request`)."""
    products: tuple[dict[str, Any], ...]
    """The product documents the store listed, in its order."""
    pagination: dict[str, Any] | None


def live_searches(bundle: Bundle) -> list[LiveSearch]:
    """Every successful search_catalog call of a live bundle, in wire order; [] for a sandbox bundle."""
    header = bundle.header
    if header.mode != "live":
        return []
    out: list[LiveSearch] = []
    for record in bundle.wire():
        found = _search(record)
        if found is None:
            continue
        request, products, pagination = found
        out.append(
            LiveSearch(
                store_id=header.store.store_id,
                domain=header.store.domain,
                episode_id=header.episode_id,
                started_at=header.started_at,
                seq=record.seq,
                request=request,
                products=products,
                pagination=pagination,
            )
        )
    return out


def _search(record: WireRecord) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], dict[str, Any] | None] | None:
    """(request, listed documents, pagination) of a successful search_catalog exchange, else None."""
    params = _search_params(record)
    payload = _answer(record.response_body) if params is not None else None
    if params is None or payload is None:
        return None
    arguments = params.get("arguments")
    catalog = arguments.get("catalog") if isinstance(arguments, Mapping) else None
    request = search_request(catalog if isinstance(catalog, Mapping) else {})
    pagination = payload.get("pagination")
    return request, tuple(product_documents(payload)), dict(pagination) if isinstance(pagination, Mapping) else None


def _search_params(record: WireRecord) -> Mapping[str, Any] | None:
    """The JSON-RPC params of a tools/call to search_catalog that got an HTTP 200, else None."""
    body = record.request_body
    if record.status != 200 or record.error or not isinstance(body, Mapping) or body.get("method") != "tools/call":
        return None
    params = body.get("params")
    return params if isinstance(params, Mapping) and params.get("name") == SEARCH_TOOL else None


def _answer(reply: Any) -> Mapping[str, Any] | None:
    """The catalog document of a reply that is no JSON-RPC error, no tool error and no UCP error, else None."""
    if not isinstance(reply, Mapping) or reply.get("error") is not None:
        return None
    result = reply.get("result")
    if isinstance(result, Mapping) and result.get("isError") is True:
        return None
    payload = reply_payload(reply)
    if payload is None or not isinstance(payload.get("products"), list):
        return None
    ucp = payload.get("ucp")
    return None if isinstance(ucp, Mapping) and ucp.get("status") == "error" else payload


def build_search_memory(store_id: str, searches: Iterable[LiveSearch]) -> SearchMemory:
    """A store's memory from its live searches. The latest answer to a request wins (by episode start, then
    episode id, then wire seq), and so does the latest listing document of a product. Searches of other stores are
    ignored. The same searches give the same bytes whatever order they come in."""
    mine = sorted((s for s in searches if s.store_id == store_id), key=lambda s: (s.started_at, s.episode_id, s.seq))
    answers: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for s in mine:
        listed = [{"id": d["id"], "variants": _variant_ids(d)} for d in s.products]
        answer: dict[str, Any] = {"tool": SEARCH_TOOL, "request": s.request, "products": listed}
        if s.pagination is not None:
            answer["pagination"] = s.pagination
        answers[dumps(s.request)] = answer
        for doc in s.products:
            documents[doc["id"]] = doc
    kept = [answers[key] for key in sorted(answers)]
    listed_ids = {p["id"] for a in kept for p in a["products"]}
    content: dict[str, Any] = {
        "schema": SCHEMA,
        "store_id": store_id,
        "domains": sorted({host(s.domain) for s in mine}),
        "episodes": sorted({s.episode_id for s in mine}),
        "days": sorted({s.started_at[:10] for s in mine}),
        "answers": kept,
        "products": {pid: documents[pid] for pid in sorted(listed_ids)},
    }
    return SearchMemory.model_validate({**content, "id": ID_PREFIX + digest(content, 12)})


def memories_from_store(
    store: TraceStore,
    prefix: str = EPISODES_PREFIX,
    *,
    stores: Iterable[str] = (),
    paths: Iterable[str] | None = None,
) -> dict[str, SearchMemory]:
    """One memory per store from every live bundle under `prefix` (only those of `stores`, when named), by store id.
    Sandbox bundles are skipped from their summaries alone, so a scan over tens of thousands of them reads two small
    files each; `paths` (a bundle index's live bundles, D72) names the live bundles instead, with no scan."""
    wanted = frozenset(stores)
    searches: list[LiveSearch] = []
    for bundle in live_bundles(store, prefix, paths=paths):
        if not wanted or bundle.header.store.store_id in wanted:
            searches += live_searches(bundle)
    found = sorted({s.store_id for s in searches})
    return {store_id: build_search_memory(store_id, searches) for store_id in found}


def memory_path(directory: Path, store_id: str, memory_id: str) -> Path:
    return Path(directory) / store_id / f"{memory_id}.json"


def save_memory(memory: SearchMemory, directory: Path) -> Path:
    """Write a memory as canonical JSON under `directory`/<store>/<id>.json; an existing file is the same bytes."""
    path = memory_path(directory, memory.store_id, memory.id)
    if not path.is_file():
        atomic_write_text(path, dumps(memory.model_dump(mode="json", by_alias=True)) + "\n")
    return path


def load_memory(directory: Path, store_id: str, memory_id: str) -> SearchMemory:
    path = memory_path(directory, store_id, memory_id)
    try:
        raw = loads(path.read_bytes())
    except FileNotFoundError:
        raise NotFoundError(
            f"no search memory {memory_id} of {store_id} under {directory}",
            hint="build it again from the live bundles with `carttrace sandbox replica memory`",
        ) from None
    memory = SearchMemory.model_validate(raw)
    if memory.id != memory_id or memory.store_id != store_id:
        raise ValidationError(f"{path} holds memory {memory.id} of {memory.store_id}", hint="rebuild the memory")
    return memory


def newest_memory(directory: Path, store_id: str) -> SearchMemory | None:
    """The store's memory with the latest day, then the most episodes, then the highest id; None when it has none."""
    found = [load_memory(directory, store_id, p.stem) for p in sorted((Path(directory) / store_id).glob("sm-*.json"))]
    if not found:
        return None
    return max(found, key=lambda m: (m.days[-1] if m.days else "", len(m.episodes), m.id))


def with_search_memory(snapshot: StoreSnapshot, memory: SearchMemory) -> StoreSnapshot:
    """A replica's snapshot that replays the memory: its answers first (they win over the capture's own answer to
    the same request, being later), the capture's other answers after, and every product an answer lists that the
    capture lacks, built from its live listing document. A product listed in another currency than the replica's
    is left out (its prices would be wrong), so its answers simply skip it. The provenance names the memory."""
    own = str(snapshot.store.get("store_id") or snapshot.store_key)
    if memory.store_id != own and host(str(snapshot.store.get("domain") or "")) not in memory.domains:
        raise ValidationError(
            f"search memory {memory.id} is of {memory.store_id}, not of {own}",
            hint="pass the replica's own store's memory",
        )
    products = {k: dict(v) for k, v in snapshot.products.items()}
    variants = [dict(v) for v in snapshot.variants]
    policies = _shared_policies(products.values())
    added = carried = 0
    for pid, doc in memory.products.items():
        if pid in products:
            carried += _carry_live_listing(products[pid], doc)
            continue
        rows = _variant_rows(pid, doc, snapshot.currency)
        if rows is None:
            continue
        facts = dict(doc)
        facts[LISTING] = {"fields": sorted(doc), "max_variants": len(rows)}
        facts[SOURCE_KEY] = memory.id
        if policies is not None:
            facts[POLICIES] = policies
        products[pid] = facts
        variants += rows
        added += 1
    answers = [a.model_dump(mode="json") for a in memory.answers]
    asked = {dumps(a["request"]) for a in answers}
    recorded = answers + [a for a in snapshot.recorded if dumps(a.get("request")) not in asked]
    provenance = {
        **snapshot.provenance,
        "search_memory": {
            "id": memory.id,
            "answers": len(answers),
            "products_added": added,
            "listing_media": carried,
            "episodes": len(memory.episodes),
            "days": list(memory.days),
        },
    }
    return snapshot.model_copy(
        update={
            "products": products,
            "variants": variants,
            "recorded": recorded,
            "provenance": provenance,
            "size": len(products),
        }
    )


def _carry_live_listing(facts: dict[str, Any], doc: Mapping[str, Any]) -> int:
    """What a live listing shows of a captured product that its capture lacks: the listing's media as that view's
    own value (`values`; the capture's media stay its facts), and any category it lists that the facts lack (the
    store's taxonomy category beside the capture's collections). 1 when it changed the product, else 0."""
    changed = 0
    media = doc.get("media")
    if isinstance(media, list) and media and media != facts.get("media"):
        own = facts.get(LISTING)
        shape = dict(own) if isinstance(own, Mapping) else {}
        values = dict(shape.get("values") or {})
        if values.get("media") != media:
            values["media"] = [dict(m) if isinstance(m, Mapping) else m for m in media]
            shape["values"] = values
            facts[LISTING] = shape
            changed = 1
    held = [c for c in facts.get("categories") or () if isinstance(c, Mapping)]
    new = [dict(c) for c in doc.get("categories") or () if isinstance(c, Mapping) and c not in held]
    if new:
        facts["categories"] = [*held, *new]
        changed = 1
    return changed


def host(domain: str) -> str:
    """A store's host as memories compare it: lowercase, without a leading www."""
    return domain.strip().lower().removeprefix("www.")


def _variant_ids(doc: Mapping[str, Any]) -> list[str]:
    return [str(v["id"]) for v in doc.get("variants") or () if isinstance(v, Mapping) and isinstance(v.get("id"), str)]


def _variant_rows(pid: str, doc: Mapping[str, Any], currency: str) -> list[dict[str, Any]] | None:
    """The snapshot variants of a listed product, or None when it lists none priced in `currency`."""
    rows: list[dict[str, Any]] = []
    for v in doc.get("variants") or ():
        if not isinstance(v, Mapping) or not isinstance(v.get("id"), str):
            continue
        price = v.get("price")
        if not isinstance(price, Mapping) or price.get("currency") != currency:
            return None
        amount = price.get("amount")
        if not isinstance(amount, int) or isinstance(amount, bool):
            return None
        availability = v.get("availability")
        available = isinstance(availability, Mapping) and availability.get("available") is True
        options = {
            str(o["name"]): str(o["label"])
            for o in v.get("options") or ()
            if isinstance(o, Mapping) and isinstance(o.get("name"), str) and isinstance(o.get("label"), str)
        }
        rows.append(
            {
                "id": v["id"],
                "product_id": pid,
                "title": str(v.get("title") or doc.get("title") or v["id"]),
                "price": amount,
                "available": available,
                "options": options,
            }
        )
    return rows or None


def _shared_policies(products: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]] | None:
    """The policies every product of the capture carries, when they all carry the same (a capture v2's store-wide
    policies), for products added from the memory; None otherwise."""
    seen = Counter(dumps(p.get(POLICIES)) for p in products)
    if len(seen) != 1:
        return None
    only = loads(next(iter(seen)))
    return [dict(p) for p in only] if isinstance(only, list) and only else None

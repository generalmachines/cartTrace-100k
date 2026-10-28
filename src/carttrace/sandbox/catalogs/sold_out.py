"""Sold-out record: how often a store's cart refused a variant as sold out on live rows, kept per store so its
replica refuses at the same rate (D52).

Why. A live Shopify store answers a cart write for a variant it will not sell with HTTP 200, the line left out of
`line_items` and a `merchandise_out_of_stock` warning ("The product '<title>' is already sold out."), while its
catalog lists the variant available. The judge session of 2026-10-13 counted it on every bundle: about one live
session in two hundred on the store's home market, one in ten cross-border, concentrated on a dozen stores (store-092
half its sessions). The sandbox refused 22 of 143,772 cart writes, because a replica sells every variant its catalog
lists. A benchmark whose sandbox never says "sold out" never measures what an agent does when a store does, and judge-
v10 (D45) already reads the answer. So every live bundle's cart writes are counted here, and the replica built with a
store's record draws, per episode, which variants answer sold out.

What is counted, per live session (`session_counts`):
- a cart write is a create_cart or update_cart call (add_to_cart, add_cart_lines on other surfaces) that sends at least
  one line; it is refused sold out when its reply carries a `merchandise_out_of_stock` (or `out_of_stock`) message, or
  fails with one, that says no units remain (a message naming units still available is short stock, not sold out, as
  judge-v10 reads it);
- the refused variants are the lines it sent that the reply's cart does not hold, with whether the session's own
  catalog replies had listed each available (559 of 562 kept refusals of 2026-10-12 and 13 had been);
- the market is the one the store saw: the market the harness filled on the cart and catalog calls (every row before
  D43, and `market_context: fill`), else `served` (D43: the store saw only our network position).

Rates (`rate_for`). Cross-border sessions with a filled market are kept as evidence but not used by default: they
are where nearly every sold-out answer of 2026-10-12 and 13 came from (store-069 refused 71 of 71 writes for a GB
shopper and 0 of 12 for a US one), the market was the harness's fill rather than anything the agent sent, and D43
removed the fill, so the judge session took them for inflated. `filled_cross_border=True` uses them. Sessions with
no market filled (`served`) are kept as evidence and not used either, unless `served=True`: the store saw our
network position, not the world's market, and answered as it answers a cross-border shopper. Over the live bundles of
2026-10-15 (2,228 sessions with a cart write) the home market filled refused 51 of 1,055 writes (4.8%), a filled
cross-border market 590 of 1,196 (49.3%) and `served` 1,190 of 1,914 (62.2%); the six stores with both a filled home
and a `served` stratum refused 0 home writes of 34 and 457 `served` writes of 515. store-557.example (store-557) had
only `served` sessions, 39 of 42 writes refused, so its US replica refused 93% of cart writes and 17 creates on the
Medusa pack that day met "Some variant does not have the required inventory". A world in
market M gets, in order, with every rate citing its counts: the store's own sessions in M (at least `min_writes`
writes), the store's own sessions in every admissible market, the median of its category's store rates, the median
of every store's rate. A median counts only stores with at least `min_writes` admissible writes of their own, so one
store cannot set a category's rate: pooled, Store 092's 51 refusals were 41% of the jewelry writes and 5% of all of
them, while every other store refused none. The pooled figures stay in the record as evidence. Rates are basis
points of cart writes.

A record is canonical JSON whose id is `so-<sha12>` of its content, saved as <paths.var>/live/sold-out/<store>/
<id>.json, so a manifest names the record its replica worlds were built with. A store with a capture and no live
session gets a record too, holding the baselines alone. Nothing here reaches a live store: it reads bundles already
written, redacted at write time.

The draw (`draw_sold_out`). A world's pack names its rate (`policy.sold_out`) and refuses nothing; the pack an episode
runs on lists the variants that answer sold out, each available variant drawn by a hash of (episode seed, variant id)
against the rate. The catalog is untouched, so it lists them available as live does. When the draw would leave none
of a task's witnesses for sale, the witness whose draw came closest to selling is spared, so a task stays winnable
unless it was written impossible (no witnesses).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from carttrace.core.canon import digest, dumps, loads
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.files import atomic_write_text
from carttrace.sandbox.catalogs.search_memory import host
from carttrace.sandbox.envpack.pack import EnvPack
from carttrace.trace.bundle_index import live_bundles
from carttrace.trace.capture import capture_files, load_capture
from carttrace.trace.capture_v2 import read_captures_v2
from carttrace.trace.layout import EPISODES_PREFIX, TraceStore
from carttrace.trace.products import reply_payload
from carttrace.trace.reader import Bundle
from carttrace.trace.schema import SoldOutRate, WireRecord

SCHEMA: Final = "mc.sold-out.v1"
ID_PREFIX: Final = "so-"
SOLD_OUT_DIR: Final = Path("live") / "sold-out"
"""Where records are saved under paths.var: one directory per store, one file per record id."""
CART_WRITES: Final = frozenset({"create_cart", "update_cart", "add_to_cart", "add_cart_lines"})
SOLD_OUT_CODES: Final = frozenset({"merchandise_out_of_stock", "out_of_stock"})
"""Shopify's code on a live cart reply, and UCP's, as judge-v10 reads them (carttrace.judge.stock)."""
SOME_LEFT: Final = re.compile(r"\b([1-9]\d*) (?:available|remaining)\b")
"""Short stock ("3 requested, 1 available", WooCommerce's "(3 remaining)"): the store would sell what it has."""
SERVED: Final = "served"
"""The market of a session the harness filled no market on (D43): what the store served our network position."""
MIN_WRITES: Final = 8
"""Cart writes a stratum needs before its own rate is used rather than the next fallback."""
BASIS: Final = 10_000
DRAW: Final = "soldout"
"""The namespace of a variant's draw."""
KEPT_REFUSALS: Final = 50
"""Refused variants a record lists as examples, most refused first."""

Source = Literal["store", "category", "overall"]
MEDIAN: Final = "median_of_stores"
"""The statistic a category or overall fallback uses: the median of the store rates, not the pooled share."""


class Counts(BaseModel):
    """Cart writes and sessions, and how many of each the store refused as sold out."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    writes: int = 0
    refused: int = 0
    sessions: int = 0
    """Sessions with at least one cart write."""
    refused_sessions: int = 0
    variants: int = 0
    """Distinct variants the sessions' writes sent (summed over sessions)."""
    refused_variants: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(**{k: getattr(self, k) + getattr(other, k) for k in Counts.model_fields})

    @property
    def rate_bp(self) -> int:
        """Refused writes in basis points of writes, rounded half up; 0 with no writes."""
        return (self.refused * BASIS * 2 + self.writes) // (self.writes * 2) if self.writes else 0


class Stratum(BaseModel):
    """One market's sessions of a store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    market: str
    """The market the store saw: the filled one, or `served` when no market was filled (D43)."""
    filled: bool
    """Whether the harness filled the market on the session's calls (every row before D43)."""
    cross_border: bool
    """Whether that market is not the store's home market."""
    used: bool
    """False for a filled cross-border stratum, unless the record was built with them (`filled_cross_border`), and
    for the `served` stratum, unless built with `served`."""
    counts: Counts


class Median(BaseModel):
    """The median of the rates of the stores with enough writes of their own, with their writes pooled beside it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rate_bp: int
    """The median store rate in basis points (the mean of the middle two, rounded half up, for an even count)."""
    stores: int
    """Stores with at least `min_writes` admissible writes, whose rates the median is over."""
    counts: Counts
    """Those stores' admissible writes pooled, cited with the median."""


def median_of(rates: Mapping[str, Counts], least: int) -> Median:
    """The median rate over the stores with at least `least` writes; 0 over no stores."""
    kept = [c for c in rates.values() if c.writes >= least]
    ordered = sorted(c.rate_bp for c in kept)
    n = len(ordered)
    mid = 0 if n == 0 else ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2] + 1) // 2
    pooled = Counts()
    for c in kept:
        pooled = pooled + c
    return Median(rate_bp=mid, stores=n, counts=pooled)


class Refusal(BaseModel):
    """One variant the store refused as sold out, with what its catalog had said of it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variant_id: str
    times: int
    """Writes that sent it and were refused."""
    markets: list[str]
    listed_available: int
    """Refusals in sessions whose own catalog replies had listed the variant available."""
    listed_unavailable: int
    words: str
    """The store's message, the first time."""


class SoldOutRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: Literal["mc.sold-out.v1"] = Field(default=SCHEMA, alias="schema")
    id: str
    store_id: str
    category: str | None
    home_market: str | None
    domains: list[str]
    episodes: int
    """Live sessions of the store read (with or without a cart write)."""
    days: list[str]
    strata: list[Stratum]
    """By market, sorted."""
    store: Counts
    """The store's admissible strata pooled."""
    category_baseline: Counts
    """Every store of the category, admissible strata pooled: evidence, not the rate (see `statistic`)."""
    overall_baseline: Counts
    """Every live store, admissible strata pooled: evidence, not the rate."""
    category_median: Median
    """The median rate of the category's stores with at least `min_writes` writes: the category fallback."""
    overall_median: Median
    """The median rate of every store with at least `min_writes` writes: the last fallback."""
    statistic: Literal["median_of_stores"] = MEDIAN
    """What the category and overall fallbacks use."""
    min_writes: int
    filled_cross_border: bool
    """Whether filled cross-border sessions were used (False by default: see the module's Rates)."""
    refusals: list[Refusal]
    served: bool = False
    """Whether sessions with no market filled were used (False by default: see the module's Rates)."""


@dataclass(frozen=True, slots=True)
class SessionCount:
    """One live session's cart writes and the store's sold-out answers to them."""

    store_id: str
    domain: str
    category: str | None
    home_market: str | None
    market: str
    filled: bool
    day: str
    counts: Counts
    refused: tuple[tuple[str, bool | None, str], ...]
    """(variant id, whether the session's catalog listed it available, the store's words) per refused variant."""

    @property
    def cross_border(self) -> bool:
        return self.market not in (SERVED, self.home_market)

    def used(self, filled_cross_border: bool = False, served: bool = False) -> bool:
        """Whether the session counts toward a rate: every one but a filled cross-border one and one with no market
        filled, unless those are asked for too."""
        if self.market == SERVED:
            return served
        return filled_cross_border or not (self.filled and self.cross_border)


def session_counts(bundle: Bundle) -> SessionCount | None:
    """A live bundle's cart writes and sold-out answers; None for a sandbox bundle."""
    header = bundle.header
    if header.mode != "live":
        return None
    params = header.condition.params
    home = _text(params.get("home_market")) or header.store.market
    filled = header.market_context != "none"
    market = (_text(params.get("market")) or header.store.market or "") if filled else SERVED
    listed: dict[str, bool] = {}
    writes = refused = 0
    sent_all: set[str] = set()
    refused_all: dict[str, tuple[bool | None, str]] = {}
    for record in bundle.wire():
        call = _tool_call(record)
        if call is None:
            continue
        name, arguments = call
        if name not in CART_WRITES:
            _listings(record.response_body, listed)
            continue
        sent = _sent(arguments)
        if not sent:
            continue
        writes += 1
        sent_all |= set(sent)
        words = _sold_out(record)
        if words is None:
            continue
        refused += 1
        held = _held(record.response_body)
        for vid in sent:
            if vid not in held:
                refused_all.setdefault(vid, (listed.get(vid), words))
    counts = Counts(
        writes=writes,
        refused=refused,
        sessions=1 if writes else 0,
        refused_sessions=1 if refused else 0,
        variants=len(sent_all),
        refused_variants=len(refused_all),
    )
    return SessionCount(
        store_id=header.store.store_id,
        domain=header.store.domain,
        category=header.store.category,
        home_market=home,
        market=market,
        filled=filled,
        day=header.started_at[:10],
        counts=counts,
        refused=tuple((vid, seen, words) for vid, (seen, words) in sorted(refused_all.items())),
    )


def build_records(
    sessions: Iterable[SessionCount],
    *,
    stores: Mapping[str, tuple[str | None, str | None]] = {},
    min_writes: int = MIN_WRITES,
    filled_cross_border: bool = False,
    served: bool = False,
) -> dict[str, SoldOutRecord]:
    """One record per store that has a live session or is named in `stores` (store id to its category and home
    market, for stores with a capture and no live session). The same sessions give the same bytes in any order."""

    def used(x: SessionCount) -> bool:
        return x.used(filled_cross_border, served)

    found = sorted(sessions, key=lambda s: (s.store_id, s.day, s.market, s.counts.model_dump_json(), s.refused))
    by_store: dict[str, list[SessionCount]] = defaultdict(list)
    for s in found:
        by_store[s.store_id].append(s)
    overall = _pooled(s for s in found if used(s))
    categories: dict[str | None, Counts] = {}
    per_store: dict[str, Counts] = {}
    store_category: dict[str, str | None] = {}
    for s in found:
        if used(s):
            categories[s.category] = categories.get(s.category, Counts()) + s.counts
            per_store[s.store_id] = per_store.get(s.store_id, Counts()) + s.counts
            store_category.setdefault(s.store_id, s.category)
    overall_median = median_of(per_store, min_writes)
    out: dict[str, SoldOutRecord] = {}
    for store_id in sorted(set(by_store) | set(stores)):
        mine = by_store.get(store_id, [])
        category = next((s.category for s in mine if s.category), None) or stores.get(store_id, (None, None))[0]
        home = next((s.home_market for s in mine if s.home_market), None) or stores.get(store_id, (None, None))[1]
        content: dict[str, Any] = {
            "schema": SCHEMA,
            "store_id": store_id,
            "category": category,
            "home_market": home,
            "domains": sorted({host(s.domain) for s in mine}),
            "episodes": len(mine),
            "days": sorted({s.day for s in mine}),
            "strata": [x.model_dump(mode="json") for x in _strata(mine, filled_cross_border, served)],
            "store": _pooled(s for s in mine if used(s)).model_dump(mode="json"),
            "category_baseline": categories.get(category, Counts()).model_dump(mode="json"),
            "overall_baseline": overall.model_dump(mode="json"),
            "category_median": median_of(
                {k: v for k, v in per_store.items() if store_category[k] == category}, min_writes
            ).model_dump(mode="json"),
            "overall_median": overall_median.model_dump(mode="json"),
            "statistic": MEDIAN,
            "min_writes": min_writes,
            "filled_cross_border": filled_cross_border,
            "refusals": [r.model_dump(mode="json") for r in _refusals(mine)],
            "served": served,
        }
        out[store_id] = SoldOutRecord.model_validate({**content, "id": ID_PREFIX + digest(content, 12)})
    return out


def rate_for(record: SoldOutRecord, market: str) -> SoldOutRate:
    """The rate a replica world of the store in `market` answers sold out at, and the evidence that set it: the
    store's own sessions in that market, else in every admissible market, else the median of its category's store
    rates, else the median of every store's (each over stores with enough writes of their own)."""
    least = record.min_writes
    own = next((x for x in record.strata if x.used and x.market == market), None)
    chosen: tuple[Source, str, Counts, int, int | None]
    if own is not None and own.counts.writes >= least:
        chosen = ("store", f"{record.store_id} {market}", own.counts, own.counts.rate_bp, None)
    elif record.store.writes >= least:
        chosen = ("store", f"{record.store_id} all markets", record.store, record.store.rate_bp, None)
    elif record.category_median.stores:
        m = record.category_median
        chosen = ("category", f"median of {_stores(m.stores)} of {record.category}", m.counts, m.rate_bp, m.stores)
    else:
        m = record.overall_median
        chosen = ("overall", f"median of {_stores(m.stores)}, every category", m.counts, m.rate_bp, m.stores)
    source, stratum, counts, rate_bp, stores = chosen
    return SoldOutRate(
        record=record.id,
        market=market,
        rate_bp=rate_bp,
        source=source,
        statistic="own" if stores is None else MEDIAN,
        stores=stores,
        stratum=stratum,
        writes=counts.writes,
        refused=counts.refused,
        sessions=counts.sessions,
        refused_sessions=counts.refused_sessions,
    )


def records_from(
    store: TraceStore,
    prefix: str = EPISODES_PREFIX,
    *,
    captures_dir: Path | None = None,
    v2_dir: Path | None = None,
    stores: Iterable[str] = (),
    min_writes: int = MIN_WRITES,
    filled_cross_border: bool = False,
    served: bool = False,
    paths: Iterable[str] | None = None,
) -> dict[str, SoldOutRecord]:
    """One record per store from every live bundle under `prefix` and every store with a capture under the given
    roots; only those of `stores` when named, though the baselines always pool every live store. `paths` (a bundle
    index's live bundles, D72) names the live bundles instead, with no scan."""
    sessions: list[SessionCount] = []
    for bundle in live_bundles(store, prefix, paths=paths):
        found = session_counts(bundle)
        if found is not None:
            sessions.append(found)
    captured = dict(captured_stores(captures_dir, v2_dir)) if captures_dir is not None else {}
    records = build_records(
        sessions, stores=captured, min_writes=min_writes, filled_cross_border=filled_cross_border, served=served
    )
    wanted = frozenset(stores)
    return {k: v for k, v in records.items() if not wanted or k in wanted}


def captured_stores(
    captures_dir: Path, v2_dir: Path | None = None
) -> Iterator[tuple[str, tuple[str | None, str | None]]]:
    """(store id, (category, market)) of every store with a capture, old or v2, so a replica with no live session
    still gets the baselines."""
    for path in capture_files(captures_dir):
        try:
            capture = load_capture(path)
        except (NotFoundError, ValidationError):
            continue  # a purge record or a capture this code cannot read holds no store to keep
        yield capture.store.store_id, (capture.store.category, capture.store.market)
    if v2_dir is not None:
        for v2 in read_captures_v2(v2_dir):
            yield v2.store_id, (v2.category, v2.market)


def record_path(directory: Path, store_id: str, record_id: str) -> Path:
    return Path(directory) / store_id / f"{record_id}.json"


def save_record(record: SoldOutRecord, directory: Path) -> Path:
    """Write a record as canonical JSON under `directory`/<store>/<id>.json; an existing file is the same bytes."""
    path = record_path(directory, record.store_id, record.id)
    if not path.is_file():
        atomic_write_text(path, dumps(record.model_dump(mode="json", by_alias=True)) + "\n")
    return path


def load_record(directory: Path, store_id: str, record_id: str) -> SoldOutRecord:
    path = record_path(directory, store_id, record_id)
    try:
        raw = loads(path.read_bytes())
    except FileNotFoundError:
        raise NotFoundError(
            f"no sold-out record {record_id} of {store_id} under {directory}",
            hint="build it again from the live bundles with `carttrace sandbox replica sold-out`",
        ) from None
    record = SoldOutRecord.model_validate(raw)
    if record.id != record_id or record.store_id != store_id:
        raise ValidationError(f"{path} holds record {record.id} of {record.store_id}", hint="rebuild the record")
    return record


def newest_record(directory: Path, store_id: str) -> SoldOutRecord | None:
    """The store's record read the current way, then with the latest live day, the most sessions and the highest id;
    None when it has none. A record written before D81 holds no `served` field: it counted sessions with no market
    filled, so it pooled more writes and outranked the store's own refreshed record. On 2026-10-16 store-146's newest
    pick was still so-a543742f110f (83.75%, a category median over those sessions) beside so-d0076d5189df (0), and
    335 of 942 overnight rows built to pay ran in such worlds, ordering 46.9% of the time against 66.9%."""
    found = [load_record(directory, store_id, p.stem) for p in sorted((Path(directory) / store_id).glob("so-*.json"))]
    if not found:
        return None
    return max(
        found,
        key=lambda r: (
            "served" in r.model_fields_set,
            max(r.days, default=""),
            r.episodes,
            r.overall_baseline.writes,
            r.id,
        ),
    )


def with_sold_out(pack: EnvPack, rate: SoldOutRate) -> EnvPack:
    """A world's pack naming the rate its episodes draw sold-out variants at (`policy.sold_out`); it refuses nothing
    itself."""
    return pack.derive(policy={"sold_out": {"rate": rate.model_dump(mode="json")}})


@dataclass(frozen=True, slots=True)
class Draw:
    """What one episode's draw decided."""

    rate: SoldOutRate
    seed: int
    variants: tuple[str, ...]
    """Refused as sold out at cart write, sorted."""
    spared: tuple[str, ...]
    candidates: int
    """Variants the pack had for sale, which were drawn."""


def draw_sold_out(
    pack: EnvPack, seed: int, *, witnesses: Sequence[str] = (), impossible: bool = False
) -> tuple[EnvPack, Draw | None]:
    """The pack an episode with `seed` runs on: its world's rate drawn over every variant for sale, the sold-out ones
    listed in `policy.sold_out`. When every witness for sale was drawn and the task is not impossible, the witness
    drawn nearest to selling is spared. The pack itself and None when its world names no rate."""
    policy = pack.doc["policy"].get("sold_out")
    if not isinstance(policy, Mapping):
        return pack, None
    rate = SoldOutRate.model_validate(policy["rate"])
    variants = pack.snapshot.variants
    candidates = sorted(vid for vid, v in variants.items() if v.available and v.stock > 0)
    scores = {vid: variant_draw(seed, vid) for vid in candidates}
    sold = {vid for vid, score in scores.items() if score < rate.rate_bp}
    spared: list[str] = []
    fitting = [w for w in dict.fromkeys(witnesses) if w in scores]
    if fitting and not impossible and all(w in sold for w in fitting):
        keep = max(fitting, key=lambda w: (scores[w], w))
        sold.discard(keep)
        spared.append(keep)
    draw = Draw(rate, seed, tuple(sorted(sold)), tuple(spared), len(candidates))
    if not sold and not spared:
        return pack, draw
    derived = pack.derive(
        policy={"sold_out": {"rate": policy["rate"], "seed": seed, "variants": sorted(sold), "spared": spared}}
    )
    return derived, draw


def sold_out_as_live(pack: EnvPack, bundle: Bundle) -> EnvPack:
    """`pack` selling out exactly the variants the live bundle's store refused as sold out, for the wire fidelity gate:
    the replica's answer to the same refusal is then scored against the store's. The pack itself when the store
    refused none the pack holds."""
    found = session_counts(bundle)
    held = pack.snapshot.variants
    variants = sorted({vid for vid, _, _ in found.refused if vid in held}) if found is not None else []
    if found is None or not variants:
        return pack
    counts = found.counts
    rate = SoldOutRate(
        record=f"live:{bundle.header.episode_id}",
        market=found.market,
        rate_bp=counts.rate_bp,
        source="store",
        stratum=f"{bundle.header.episode_id} as live",
        writes=counts.writes,
        refused=counts.refused,
        sessions=counts.sessions,
        refused_sessions=counts.refused_sessions,
    )
    policy = {"rate": rate.model_dump(mode="json"), "seed": 0, "variants": variants, "spared": []}
    return pack.derive(policy={"sold_out": policy})


def variant_draw(seed: int, variant_id: str) -> int:
    """A variant's draw for an episode, uniform over [0, 10000): it is sold out when the draw is under the rate."""
    return int(digest([DRAW, seed, variant_id], 8), 16) % BASIS


def _stores(n: int) -> str:
    return f"{n} store" if n == 1 else f"{n} stores"


def _pooled(sessions: Iterable[SessionCount]) -> Counts:
    out = Counts()
    for s in sessions:
        out = out + s.counts
    return out


def _strata(sessions: Sequence[SessionCount], filled_cross_border: bool, served: bool) -> list[Stratum]:
    groups: dict[tuple[str, bool], list[SessionCount]] = defaultdict(list)
    for s in sessions:
        groups[(s.market, s.filled)].append(s)
    return [
        Stratum(
            market=market,
            filled=filled,
            cross_border=group[0].cross_border,
            used=group[0].used(filled_cross_border, served),
            counts=_pooled(group),
        )
        for (market, filled), group in sorted(groups.items())
    ]


def _refusals(sessions: Sequence[SessionCount]) -> list[Refusal]:
    times: dict[str, int] = defaultdict(int)
    markets: dict[str, set[str]] = defaultdict(set)
    seen: dict[str, list[bool | None]] = defaultdict(list)
    words: dict[str, str] = {}
    for s in sessions:
        for vid, listed, said in s.refused:
            times[vid] += 1
            markets[vid].add(s.market)
            seen[vid].append(listed)
            words.setdefault(vid, said)
    ranked = sorted(times, key=lambda v: (-times[v], v))[:KEPT_REFUSALS]
    return [
        Refusal(
            variant_id=vid,
            times=times[vid],
            markets=sorted(markets[vid]),
            listed_available=sum(1 for x in seen[vid] if x is True),
            listed_unavailable=sum(1 for x in seen[vid] if x is False),
            words=words[vid],
        )
        for vid in ranked
    ]


def _tool_call(record: WireRecord) -> tuple[str, Mapping[str, Any]] | None:
    body = record.request_body
    if not isinstance(body, Mapping) or body.get("method") != "tools/call":
        return None
    params = body.get("params")
    if not isinstance(params, Mapping) or not isinstance(params.get("name"), str):
        return None
    arguments = params.get("arguments")
    return str(params["name"]), arguments if isinstance(arguments, Mapping) else {}


def _sent(arguments: Mapping[str, Any]) -> list[str]:
    """The variant ids a cart write sends: UCP's `cart.line_items[].item.id` (or the lines at the top level)."""
    cart = arguments.get("cart")
    holder = cart if isinstance(cart, Mapping) else arguments
    out: list[str] = []
    for line in holder.get("line_items") or ():
        item = line.get("item") if isinstance(line, Mapping) else None
        ident = item.get("id") if isinstance(item, Mapping) else None
        if isinstance(ident, str) and ident:
            out.append(ident)
    return list(dict.fromkeys(out))


def _payload(reply: Any) -> Mapping[str, Any] | None:
    payload = reply_payload(reply)
    if isinstance(payload, Mapping) and isinstance(payload.get("structuredContent"), Mapping):
        inner: Mapping[str, Any] = payload["structuredContent"]
        return inner
    return payload


def _sold_out(record: WireRecord) -> str | None:
    """The store's words when a cart write's reply refused something as sold out (no units left), else None."""
    payload = _payload(record.response_body)
    if payload is None:
        return None
    found = [m for m in payload.get("messages") or () if isinstance(m, Mapping) and m.get("code") in SOLD_OUT_CODES]
    error = payload.get("error")
    if isinstance(error, Mapping) and error.get("code") in SOLD_OUT_CODES:
        found.append(error)
    texts = [str(m.get("content") or m.get("message") or "") for m in found]
    if not found or any(SOME_LEFT.search(t) for t in texts):
        return None
    return texts[0]


def _held(reply: Any) -> set[str]:
    payload = _payload(reply)
    out: set[str] = set()
    for line in (payload or {}).get("line_items") or ():
        item = line.get("item") if isinstance(line, Mapping) else None
        if isinstance(item, Mapping) and isinstance(item.get("id"), str):
            out.add(item["id"])
    return out


def _listings(reply: Any, listed: dict[str, bool]) -> None:
    """Record each variant a catalog reply lists with its availability (the latest listing wins)."""
    payload = _payload(reply)
    if payload is None:
        return
    products = payload.get("products")
    docs = products if isinstance(products, list) else [payload.get("product")]
    for doc in docs:
        for variant in (doc.get("variants") if isinstance(doc, Mapping) else None) or ():
            if not isinstance(variant, Mapping) or not isinstance(variant.get("id"), str):
                continue
            availability = variant.get("availability")
            available = availability.get("available") if isinstance(availability, Mapping) else None
            if isinstance(available, bool):
                listed[variant["id"]] = available


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None

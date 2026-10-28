"""Store replies record: what a store's own live replies carry that its capture does not, kept per store so its
replica serves the same (D70): the payment handlers its cart replies declare, and how often it answered a call with
an error of its own making.

Why. The judge session's code-only re-judge (judge-v11 store ledger, 2026-10-13) read two gaps no capture could close:
- **payment handlers.** Live Shopify lists `ucp.payment_handlers` on every cart reply (Google Pay, Shopify's card
  handler and Shop Pay, each with its spec, schema and config: store-111 ep-d0df49b419cb seq 20), while 111 of 117 replica
  packs held none, so a replica listed each namespace as a bare `{id, version}` stub beside the sandbox's own fake
  handler. The capture pass reads pages and feeds, not carts, so only a live cart reply shows them;
- **store errors.** Of 1,641 live agent-view journeys, 365 had a failed call the same tool never recovered from, but
  only 8 of those were the store's own doing: Shopify answering create_cart "Throttled" (38 calls on store-111, store-116
  and store-137, ep-422a0968cd25 seq 25). The rest were the harness's own wire records (237 journeys with a market fill
  record, 16 with a refused send), discovery documents (llms.txt and agents.md 3xx and 404s), and refusals of what
  the agent sent (missing arguments, a cart the agent had cancelled, an id that never existed), which the replica's
  own validation already answers. Beside them, one HTTP 500 "Internal error" (store-398 ep-7b185304b778 seq 32) and
  three connect timeouts, all recovered at the next call.

What is counted, per live session (`session_replies`): every MCP tool call the store answered (the harness's own
records, our in-process hand-off and the discovery documents left out), by tool; the calls that failed by the store's
doing, by kind: `throttled` (an isError reply whose message says "Throttled"), `server_error` (an HTTP 5xx) and
`timeout` (the call never got an answer: the network path, kept as evidence and never ported); and the newest cart
reply's `ucp.payment_handlers`, exactly as the store sent them (redacted at write time).

Rates (`error_rates`). A ported kind on a tool the store was called at least `min_calls` times on is the store's own
share of those calls; on a tool it was called on fewer times, the median of every store's share over stores with
enough calls (so one store's throttling never sets another's rate; the median is 0 while most stores never throttle).
Rates are basis points of the tool's calls.

A record is canonical JSON whose id is `sr-<sha12>` of its content, saved as <paths.var>/live/store-replies/<store>/
<id>.json, so a manifest names the record its replica worlds were built with. Nothing here reaches a live store: it
reads bundles already written.

The draw (`draw_store_errors`). A world's pack names the rates and fails nothing; the pack an episode runs on lists
the calls that fail, the n-th call of a tool failing when a hash of (episode seed, kind, tool, n) falls under the
rate, for the first `MAX_DRAWN_CALLS` calls of each tool. The UCP surface answers a listed call in its store's words
(carttrace.sandbox.surfaces.ucp.operations); a pack served under another platform's rules (D42) answers every call.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from carttrace.core.canon import digest, dumps, loads
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.files import atomic_write_text
from carttrace.sandbox.catalogs.search_memory import host
from carttrace.sandbox.envpack.doc import DrawnFault
from carttrace.sandbox.envpack.pack import EnvPack
from carttrace.trace.bundle_index import live_bundles
from carttrace.trace.layout import EPISODES_PREFIX, TraceStore
from carttrace.trace.products import reply_payload
from carttrace.trace.reader import Bundle
from carttrace.trace.schema import StoreErrorRate, StoreReplies, WireRecord

SCHEMA: Final = "mc.store-replies.v1"
ID_PREFIX: Final = "sr-"
REPLIES_DIR: Final = Path("live") / "store-replies"
"""Where records are saved under paths.var: one directory per store, one file per record id."""
FaultKind = Literal["throttled", "server_error", "timeout"]
PortedKind = Literal["throttled", "server_error"]
PORTED: Final[tuple[PortedKind, ...]] = ("throttled", "server_error")
"""The kinds a replica answers with. A timeout is the network between us and the store, never a reply it wrote."""
THROTTLED: Final = "Throttled"
"""Live Shopify's whole message when it throttles a cart write (code `error`, severity requires_buyer_input)."""
CART_TOOLS: Final = frozenset({"create_cart", "update_cart", "get_cart", "cancel_cart"})
HARNESS_PROTOCOL: Final = "harness"
MIN_CALLS: Final = 20
"""Calls of a tool a store needs before its own rate is used rather than the median of every store's."""
BASIS: Final = 10_000
MAX_DRAWN_CALLS: Final = 60
"""Calls of each tool an episode's draw decides; later calls are always answered (no journey has made more)."""
DRAW: Final = "storeerror"
KEPT_EXAMPLES: Final = 5
"""Faults a record cites per kind."""


class ToolCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    calls: int = 0
    faults: dict[str, int] = Field(default_factory=dict)
    """Kind to the calls that failed so."""


class HandlersSeen(BaseModel):
    """The payment handlers a store's newest cart reply declared, and where."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    handlers: dict[str, Any]
    """Namespace to its entries, exactly as the store sent them."""
    episode_id: str
    seq: int
    started_at: str
    replies: int
    """Cart replies of the store's sessions that declared handlers."""
    sets: int
    """Distinct namespace sets among them (1 when the store always declared the same ones)."""


class FaultExample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    tool: str
    episode_id: str
    seq: int
    words: str


class OverallMedian(BaseModel):
    """The median rate of one kind on one tool over every store called on it at least `min_calls` times."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    tool: str
    rate_bp: int
    stores: int


class StoreRepliesRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: Literal["mc.store-replies.v1"] = Field(default=SCHEMA, alias="schema")
    id: str
    store_id: str
    domains: list[str]
    episodes: int
    """Live agent-view sessions of the store read."""
    days: list[str]
    payment_handlers: HandlersSeen | None
    tools: dict[str, ToolCounts]
    """Tool to its calls and faults, over the store's sessions."""
    examples: list[FaultExample]
    overall: list[OverallMedian]
    min_calls: int


@dataclass(slots=True)
class SessionReplies:
    """One live session's tool calls, the store's own errors among them, and its newest declared handlers."""

    store_id: str
    domain: str
    episode_id: str
    started_at: str
    calls: Counter[str] = field(default_factory=Counter)
    faults: Counter[tuple[str, str]] = field(default_factory=Counter)
    """(kind, tool) to calls."""
    examples: list[FaultExample] = field(default_factory=list)
    handlers: tuple[int, dict[str, Any]] | None = None
    """(seq, handlers) of the session's newest cart reply that declared any."""
    handler_sets: list[tuple[str, ...]] = field(default_factory=list)


def session_replies(bundle: Bundle) -> SessionReplies | None:
    """A live agent-view bundle's tool calls and the store's own errors; None for a sandbox or website bundle."""
    header = bundle.header
    if header.mode != "live" or any(s.kind == "website" for s in header.surfaces):
        return None
    out = SessionReplies(header.store.store_id, header.store.domain, header.episode_id, header.started_at)
    for record in bundle.wire():
        tool = _tool(record)
        if tool is None:
            continue
        out.calls[tool] += 1
        kind = fault_kind(record)
        if kind is not None:
            out.faults[(kind, tool)] += 1
            words = _words(record, kind)
            out.examples.append(
                FaultExample(kind=kind, tool=tool, episode_id=header.episode_id, seq=record.seq, words=words)
            )
        if tool in CART_TOOLS:
            declared = declared_handlers(record)
            if declared:
                out.handlers = (record.seq, declared)
                out.handler_sets.append(tuple(sorted(declared)))
    return out


def fault_kind(record: WireRecord) -> FaultKind | None:
    """The store's own error on one tool call, or None: a call that never got an answer, an HTTP 5xx, or Shopify's
    "Throttled" isError reply. Refusals of what the agent sent are the store answering, not failing."""
    if record.error is not None:
        return "timeout" if "timeout" in record.error.lower() else None
    if record.status is not None and record.status >= 500:
        return "server_error"
    body = record.response_body
    result = body.get("result") if isinstance(body, Mapping) else None
    if not (isinstance(result, Mapping) and result.get("isError") is True):
        return None
    payload = reply_payload(body)
    for message in (payload or {}).get("messages") or ():
        if isinstance(message, Mapping) and str(message.get("content") or "").strip() == THROTTLED:
            return "throttled"
    return None


def declared_handlers(record: WireRecord) -> dict[str, Any]:
    """The payment handlers a successful reply's `ucp` block declares, namespace to entries; {} when none."""
    if record.status != 200 or record.error is not None:
        return {}
    payload = reply_payload(record.response_body)
    ucp = payload.get("ucp") if isinstance(payload, Mapping) else None
    handlers = ucp.get("payment_handlers") if isinstance(ucp, Mapping) else None
    if not isinstance(handlers, Mapping):
        return {}
    return {
        str(ns): [dict(e) for e in entries if isinstance(e, Mapping)]
        for ns, entries in handlers.items()
        if isinstance(entries, list) and any(isinstance(e, Mapping) for e in entries)
    }


def build_records(sessions: Iterable[SessionReplies], *, min_calls: int = MIN_CALLS) -> dict[str, StoreRepliesRecord]:
    """One record per store with a live agent-view session. The same sessions give the same bytes in any order."""
    found = sorted(sessions, key=lambda s: (s.store_id, s.started_at, s.episode_id))
    by_store: dict[str, list[SessionReplies]] = defaultdict(list)
    for s in found:
        by_store[s.store_id].append(s)
    tools = {store_id: _tools(mine) for store_id, mine in by_store.items()}
    overall = _overall(tools, min_calls)
    out: dict[str, StoreRepliesRecord] = {}
    for store_id, mine in sorted(by_store.items()):
        content: dict[str, Any] = {
            "schema": SCHEMA,
            "store_id": store_id,
            "domains": sorted({host(s.domain) for s in mine}),
            "episodes": len(mine),
            "days": sorted({s.started_at[:10] for s in mine}),
            "payment_handlers": _handlers(mine),
            "tools": {tool: counts.model_dump(mode="json") for tool, counts in sorted(tools[store_id].items())},
            "examples": [e.model_dump(mode="json") for e in _examples(mine)],
            "overall": [m.model_dump(mode="json") for m in overall],
            "min_calls": min_calls,
        }
        out[store_id] = StoreRepliesRecord.model_validate({**content, "id": ID_PREFIX + digest(content, 12)})
    return out


def error_rates(record: StoreRepliesRecord) -> list[StoreErrorRate]:
    """Every ported kind's rate above zero on each tool, the store's own where it was called enough, else the median
    of every store's."""
    medians = {(m.kind, m.tool): m for m in record.overall}
    out: list[StoreErrorRate] = []
    for tool in sorted(set(record.tools) | {t for _, t in medians}):
        counts = record.tools.get(tool, ToolCounts())
        for kind in PORTED:
            faults = counts.faults.get(kind, 0)
            median = medians.get((kind, tool))
            if counts.calls >= record.min_calls:
                rate = StoreErrorRate(
                    kind=kind, tool=tool, rate_bp=_bp(faults, counts.calls), source="store", calls=counts.calls,
                    faults=faults,
                )  # fmt: skip
            elif median is not None and median.stores:
                rate = StoreErrorRate(
                    kind=kind, tool=tool, rate_bp=median.rate_bp, source="overall", stores=median.stores,
                    calls=counts.calls, faults=faults,
                )  # fmt: skip
            else:
                continue
            if rate.rate_bp > 0:
                out.append(rate)
    return out


def replies_ref(record: StoreRepliesRecord) -> StoreReplies:
    """What a world's recipe and an episode's header name of a record: the handlers served and the error rates."""
    seen = record.payment_handlers
    return StoreReplies(
        record=record.id,
        handlers=sorted(seen.handlers) if seen is not None else [],
        handlers_from=f"{seen.episode_id} seq {seen.seq}" if seen is not None else None,
        errors=error_rates(record),
    )


def with_store_replies(pack: EnvPack, record: StoreRepliesRecord) -> EnvPack:
    """A replica world's pack serving its store's own replies: the payment handlers its live cart replies declared,
    in place of the sandbox's fake one, and the rates its episodes draw the store's errors at. The pack itself when
    the record shows neither."""
    changes: dict[str, Any] = {}
    policy: dict[str, Any] = {}
    seen = record.payment_handlers
    if seen is not None:
        changes["payment_handlers"] = dict(seen.handlers)
        policy["fake_handler"] = False
    rates = error_rates(record)
    if rates:
        policy["store_errors"] = {"rates": [r.model_dump(mode="json") for r in rates]}
    if policy:
        changes["policy"] = policy
    return pack.derive(**changes) if changes else pack


def draw_store_errors(pack: EnvPack, seed: int) -> tuple[EnvPack, list[DrawnFault] | None]:
    """The pack an episode with `seed` runs on: its world's error rates drawn over the first `MAX_DRAWN_CALLS` calls
    of each tool, the failing calls listed in `policy.store_errors`. The pack itself and None when its world names no
    rate; the pack itself and [] when the draw failed no call."""
    policy = pack.doc["policy"].get("store_errors")
    if not isinstance(policy, Mapping):
        return pack, None
    rates = [StoreErrorRate.model_validate(r) for r in policy.get("rates") or ()]
    taken: dict[tuple[str, int], str] = {}
    for rate in sorted(rates, key=lambda r: (r.tool, r.kind)):
        for n in range(1, MAX_DRAWN_CALLS + 1):
            if (rate.tool, n) not in taken and call_draw(seed, rate.kind, rate.tool, n) < rate.rate_bp:
                taken[(rate.tool, n)] = rate.kind
    calls = [DrawnFault.model_validate({"tool": t, "call": n, "kind": k}) for (t, n), k in sorted(taken.items())]
    if not calls:
        return pack, []
    drawn = {"rates": policy["rates"], "seed": seed, "calls": [c.model_dump(mode="json") for c in calls]}
    return pack.derive(policy={"store_errors": drawn}), calls


def store_errors_as_live(pack: EnvPack, bundle: Bundle) -> EnvPack:
    """`pack` failing exactly the calls the live bundle's store failed with a ported kind, for the wire fidelity gate:
    the replica's answer to the same call is then scored against the store's. A call is counted among its tool's calls
    the store's schema let through (a text refusal of missing or invalid arguments is the schema's, as the replica's
    is). The pack itself when the store failed none."""
    counts: Counter[str] = Counter()
    calls: list[dict[str, Any]] = []
    for record in sorted(bundle.wire(), key=lambda r: r.seq):
        tool = _tool(record)
        if tool is None or _schema_refusal(record):
            continue
        counts[tool] += 1
        kind = fault_kind(record)
        if kind in PORTED:
            calls.append({"tool": tool, "call": counts[tool], "kind": kind})
    if not calls:
        return pack
    rates = [
        StoreErrorRate(
            kind=c["kind"], tool=c["tool"], rate_bp=BASIS, source="store", calls=counts[c["tool"]], faults=1
        ).model_dump(mode="json")
        for c in calls
    ]
    policy = {"rates": rates, "seed": 0, "calls": sorted(calls, key=lambda c: (c["tool"], c["call"]))}
    return pack.derive(policy={"store_errors": policy})


def call_draw(seed: int, kind: str, tool: str, n: int) -> int:
    """The n-th call of a tool's draw for an episode, uniform over [0, 10000): it fails when under the rate."""
    return int(digest([DRAW, seed, kind, tool, n], 8), 16) % BASIS


def records_from(
    store: TraceStore,
    prefix: str = EPISODES_PREFIX,
    *,
    stores: Iterable[str] = (),
    min_calls: int = MIN_CALLS,
    paths: Iterable[str] | None = None,
) -> dict[str, StoreRepliesRecord]:
    """One record per store from every live bundle under `prefix`; only those of `stores` when named, though the
    medians always pool every live store. `paths` (a bundle index's live bundles, D72) names the live bundles
    instead, with no scan."""
    sessions: list[SessionReplies] = []
    for bundle in live_bundles(store, prefix, paths=paths):
        found = session_replies(bundle)
        if found is not None:
            sessions.append(found)
    records = build_records(sessions, min_calls=min_calls)
    wanted = frozenset(stores)
    return {k: v for k, v in records.items() if not wanted or k in wanted}


def record_path(directory: Path, store_id: str, record_id: str) -> Path:
    return Path(directory) / store_id / f"{record_id}.json"


def save_record(record: StoreRepliesRecord, directory: Path) -> Path:
    """Write a record as canonical JSON under `directory`/<store>/<id>.json; an existing file is the same bytes."""
    path = record_path(directory, record.store_id, record.id)
    if not path.is_file():
        atomic_write_text(path, dumps(record.model_dump(mode="json", by_alias=True)) + "\n")
    return path


def load_record(directory: Path, store_id: str, record_id: str) -> StoreRepliesRecord:
    path = record_path(directory, store_id, record_id)
    try:
        raw = loads(path.read_bytes())
    except FileNotFoundError:
        raise NotFoundError(
            f"no store replies record {record_id} of {store_id} under {directory}",
            hint="build it again from the live bundles with `carttrace sandbox replica store-replies`",
        ) from None
    record = StoreRepliesRecord.model_validate(raw)
    if record.id != record_id or record.store_id != store_id:
        raise ValidationError(f"{path} holds record {record.id} of {record.store_id}", hint="rebuild the record")
    return record


def newest_record(directory: Path, store_id: str) -> StoreRepliesRecord | None:
    """The store's record with the latest live day, then the most sessions, then the highest id; None when it has
    none."""
    found = [load_record(directory, store_id, p.stem) for p in sorted((Path(directory) / store_id).glob("sr-*.json"))]
    if not found:
        return None
    return max(found, key=lambda r: (max(r.days, default=""), r.episodes, r.id))


def _tool(record: WireRecord) -> str | None:
    """The tool an MCP tools/call to the store names; None for anything else (the harness's own records, our
    in-process hand-off, discovery documents, initialize and tools/list)."""
    if record.protocol == HARNESS_PROTOCOL:
        return None
    hostname = urlsplit(record.url).hostname or ""
    if hostname == "invalid" or hostname.endswith(".invalid"):
        return None
    body = record.request_body
    if not isinstance(body, Mapping) or body.get("method") != "tools/call":
        return None
    params = body.get("params")
    name = params.get("name") if isinstance(params, Mapping) else None
    return name if isinstance(name, str) and name else None


def _schema_refusal(record: WireRecord) -> bool:
    """Whether the store refused the call's arguments before acting on them: Shopify's text-only isError answers
    ("Missing required arguments: catalog", "Invalid arguments: ...")."""
    body = record.response_body
    result = body.get("result") if isinstance(body, Mapping) else None
    if not (isinstance(result, Mapping) and result.get("isError") is True):
        return False
    content = result.get("content")
    blocks = content if isinstance(content, list) else []
    texts = [str(c.get("text") or "") for c in blocks if isinstance(c, Mapping)]
    return any(t.startswith(("Missing required arguments", "Invalid arguments")) for t in texts)


def _words(record: WireRecord, kind: str) -> str:
    if kind == "timeout":
        return str(record.error or "")[:200]
    if kind == "throttled":
        return THROTTLED
    body = record.response_body
    error = body.get("error") if isinstance(body, Mapping) else None
    message = error.get("message") if isinstance(error, Mapping) else None
    return f"HTTP {record.status}" + (f": {message}" if isinstance(message, str) else "")


def _tools(sessions: Sequence[SessionReplies]) -> dict[str, ToolCounts]:
    calls: Counter[str] = Counter()
    faults: dict[str, Counter[str]] = defaultdict(Counter)
    for s in sessions:
        calls.update(s.calls)
        for (kind, tool), n in s.faults.items():
            faults[tool][kind] += n
    return {tool: ToolCounts(calls=n, faults=dict(sorted(faults[tool].items()))) for tool, n in calls.items()}


def _overall(tools: Mapping[str, Mapping[str, ToolCounts]], least: int) -> list[OverallMedian]:
    out: list[OverallMedian] = []
    names = sorted({tool for counts in tools.values() for tool in counts})
    for tool in names:
        enough = [counts[tool] for counts in tools.values() if tool in counts and counts[tool].calls >= least]
        for kind in PORTED:
            ordered = sorted(_bp(c.faults.get(kind, 0), c.calls) for c in enough)
            n = len(ordered)
            mid = 0 if n == 0 else ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2] + 1) // 2
            out.append(OverallMedian(kind=kind, tool=tool, rate_bp=mid, stores=n))
    return out


def _handlers(sessions: Sequence[SessionReplies]) -> dict[str, Any] | None:
    newest = None
    replies = 0
    sets: set[tuple[str, ...]] = set()
    for s in sessions:
        replies += len(s.handler_sets)
        sets |= set(s.handler_sets)
        if s.handlers is not None:
            newest = (s, s.handlers)
    if newest is None:
        return None
    session, (seq, handlers) = newest
    seen = HandlersSeen(
        handlers=handlers,
        episode_id=session.episode_id,
        seq=seq,
        started_at=session.started_at,
        replies=replies,
        sets=len(sets),
    )
    return seen.model_dump(mode="json")


def _examples(sessions: Sequence[SessionReplies]) -> list[FaultExample]:
    kept: dict[str, list[FaultExample]] = defaultdict(list)
    for s in sessions:
        for e in s.examples:
            if len(kept[e.kind]) < KEPT_EXAMPLES:
                kept[e.kind].append(e)
    return [e for kind in sorted(kept) for e in kept[kind]]


def _bp(part: int, whole: int) -> int:
    """`part` in basis points of `whole`, rounded half up; 0 of nothing."""
    return (part * BASIS * 2 + whole) // (whole * 2) if whole else 0


__all__ = [
    "PORTED",
    "REPLIES_DIR",
    "SCHEMA",
    "StoreRepliesRecord",
    "build_records",
    "call_draw",
    "declared_handlers",
    "draw_store_errors",
    "error_rates",
    "fault_kind",
    "load_record",
    "newest_record",
    "records_from",
    "replies_ref",
    "save_record",
    "session_replies",
    "store_errors_as_live",
    "with_store_replies",
]

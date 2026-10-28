"""Agent surface: the tools/list a store sent live, kept per store so its replica lists exactly the same (D40).

Why. In all 682 compared live and replica pairs of 2026-10-12 and 13 the replica's agent got a first prompt about
18,600 tokens longer than the live agent's: the replica listed the vendored binding's input schemas and live Shopify
its own, shorter ones (D39, Open). The tool definitions are most of a first prompt, so a pair whose halves list
different tools can never start alike. Every live session lists the store's tools before its first call, so every
live bundle and every capture that reached the store's MCP endpoint holds the store's own list, verbatim, in its
wire records. This module keeps, per store and per (MCP version, UCP release), the latest of them:
- sources: live bundles (`wire.jsonl`), old captures (`var/captures/<store>/cap-*.json`, whose `wire` keeps the
  session) and capture v2 editions (`wire.jsonl.gz`, which hold a tools/list only when the pass opened the store's MCP
  session). A reply counts when it got HTTP 200, is no JSON-RPC error, lists at least one tool and was not paged;
- versions: the MCP version the request named in `MCP-Protocol-Version` (else the session's initialize reply), and the
  UCP release the record was labelled with;
- the latest wins: by the recording's time (the episode's start or the capture's time), then its source id, then
  the record's seq;
- redacted schema positions (the `signature` member of Shopify's Apple Pay schema, redacted at write time) are
  restored as the string schema their siblings have (`carttrace.sandbox.surfaces.ucp.listing`), and their paths are kept.

A surface is canonical JSON whose id is `as-<sha12>` of its content, saved as <paths.var>/live/agent-surface/<store>/
<id>.json, so a manifest names the surface its replica worlds were built with and a rebuilt world is the same world.
`with_agent_surface(snapshot, surface)` puts its lists on a replica's snapshot, and so in its pack (`tool_lists`).
Nothing here reaches a live store: it reads bundles and captures already written, redacted at write time.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from carttrace.core.canon import digest, dumps, loads
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.files import atomic_write_text
from carttrace.sandbox.catalogs.search_memory import host
from carttrace.sandbox.catalogs.snapshot import StoreSnapshot
from carttrace.sandbox.envpack.doc import RecordedToolList
from carttrace.sandbox.surfaces.ucp.listing import restore_redacted
from carttrace.trace.bundle_index import live_bundles
from carttrace.trace.capture import capture_files, load_capture
from carttrace.trace.capture_v2 import read_captures_v2
from carttrace.trace.layout import EPISODES_PREFIX, TraceStore
from carttrace.trace.reader import Bundle
from carttrace.trace.schema import WireRecord

SCHEMA: Final = "mc.agent-surface.v1"
ID_PREFIX: Final = "as-"
SURFACE_DIR: Final = Path("live") / "agent-surface"
"""Where surfaces are saved under paths.var: one directory per store, one file per surface id."""
LIST_METHOD: Final = "tools/list"
VERSION_HEADER: Final = "mcp-protocol-version"


class AgentSurface(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: Literal["mc.agent-surface.v1"] = Field(default=SCHEMA, alias="schema")
    id: str
    store_id: str
    domains: list[str]
    """The store's hosts as its recordings named them (lowercase, without www.), sorted."""
    seen: int
    """How many tools/list replies of the store were read."""
    distinct: int
    """How many different lists they held (1 when the store always listed the same tools)."""
    tool_lists: list[RecordedToolList]
    """The latest list per (mcp, ucp), sorted by versions."""
    restored: list[str] = Field(default_factory=list)
    """The redacted schema positions restored, as `<mcp>/<ucp>/<tool>/<pointer>`."""


@dataclass(frozen=True, slots=True)
class LiveList:
    """One tools/list reply a store sent, where it was read and when."""

    store_id: str
    domain: str
    source: str
    """`live:<episode>#<seq>`, `capture:<capture>#<seq>`."""
    recorded_at: str
    mcp: str
    ucp: str
    tools: tuple[Mapping[str, Any], ...]


def lists_in(
    records: Iterable[WireRecord], *, store_id: str, domain: str, source: str, recorded_at: str
) -> list[LiveList]:
    """Every countable tools/list reply among one session's wire records, in record order."""
    out: list[LiveList] = []
    negotiated: str | None = None
    for record in records:
        body = record.request_body
        if not isinstance(body, Mapping):
            continue
        if body.get("method") == "initialize":
            negotiated = _negotiated(record.response_body) or negotiated
            continue
        tools = _listed(record) if body.get("method") == LIST_METHOD else None
        if tools is None:
            continue
        headers = {k.lower(): v for k, v in record.request_headers.items()}
        mcp = headers.get(VERSION_HEADER) or negotiated
        if not mcp or not record.version or record.protocol != "ucp":
            continue
        out.append(LiveList(store_id, domain, f"{source}#{record.seq}", recorded_at, mcp, record.version, tools))
    return out


def live_lists(bundle: Bundle) -> list[LiveList]:
    """The tools/list replies of a live bundle; [] for a sandbox bundle."""
    header = bundle.header
    if header.mode != "live":
        return []
    return lists_in(
        bundle.wire(),
        store_id=header.store.store_id,
        domain=header.store.domain,
        source=f"live:{header.episode_id}",
        recorded_at=header.started_at,
    )


def capture_lists(captures_dir: Path, v2_dir: Path | None = None, *, stores: Iterable[str] = ()) -> list[LiveList]:
    """The tools/list replies of every old capture under `captures_dir` and every capture v2 under `v2_dir`."""
    wanted = tuple(dict.fromkeys(stores))
    out: list[LiveList] = []
    for path in capture_files(captures_dir, wanted):
        try:
            capture = load_capture(path)
        except (NotFoundError, ValidationError):
            continue  # a purge record or a capture this code cannot read holds no session to keep
        out += lists_in(
            capture.wire,
            store_id=capture.store.store_id,
            domain=capture.store.domain,
            source=f"capture:{capture.id}",
            recorded_at=capture.captured_at,
        )
    if v2_dir is not None:
        for v2 in read_captures_v2(v2_dir, store_ids=wanted):
            out += lists_in(
                _v2_wire(v2.path),
                store_id=v2.store_id,
                domain=v2.domain,
                source=f"capture:{v2.id}",
                recorded_at=v2.captured_at,
            )
    return out


def build_agent_surface(store_id: str, lists: Iterable[LiveList]) -> AgentSurface:
    """A store's surface from its recorded lists: the latest per (mcp, ucp). Lists of other stores are ignored, and
    the same lists give the same bytes whatever order they come in."""
    mine = sorted((x for x in lists if x.store_id == store_id), key=lambda x: (x.recorded_at, x.source))
    if not mine:
        raise NotFoundError(f"no tools/list reply of {store_id} was recorded", hint="run a live row on the store first")
    latest: dict[tuple[str, str], LiveList] = {}
    for x in mine:
        latest[(x.mcp, x.ucp)] = x
    kept: list[dict[str, Any]] = []
    restored: list[str] = []
    for (mcp, ucp), x in sorted(latest.items()):
        tools, paths = restore_redacted(x.tools)
        restored += [f"{mcp}/{ucp}{p}" for p in paths]
        kept.append({"mcp": mcp, "ucp": ucp, "tools": tools, "source": x.source, "recorded_at": x.recorded_at})
    content: dict[str, Any] = {
        "schema": SCHEMA,
        "store_id": store_id,
        "domains": sorted({host(x.domain) for x in mine}),
        "seen": len(mine),
        "distinct": len({digest(list(x.tools)) for x in mine}),
        "tool_lists": kept,
        "restored": restored,
    }
    return AgentSurface.model_validate({**content, "id": ID_PREFIX + digest(content, 12)})


def surfaces_from(
    store: TraceStore,
    prefix: str = EPISODES_PREFIX,
    *,
    captures_dir: Path | None = None,
    v2_dir: Path | None = None,
    stores: Iterable[str] = (),
    paths: Iterable[str] | None = None,
) -> dict[str, AgentSurface]:
    """One surface per store from every live bundle under `prefix` and every capture under the given roots (only
    those of `stores`, when named), by store id. Sandbox bundles are skipped from their summaries alone; `paths` (a
    bundle index's live bundles, D72) names the live bundles instead, with no scan."""
    wanted = frozenset(stores)
    lists: list[LiveList] = []
    for bundle in live_bundles(store, prefix, paths=paths):
        if not wanted or bundle.header.store.store_id in wanted:
            lists += live_lists(bundle)
    if captures_dir is not None:
        lists += capture_lists(captures_dir, v2_dir, stores=sorted(wanted))
    found = sorted({x.store_id for x in lists if not wanted or x.store_id in wanted})
    return {store_id: build_agent_surface(store_id, lists) for store_id in found}


def surface_path(directory: Path, store_id: str, surface_id: str) -> Path:
    return Path(directory) / store_id / f"{surface_id}.json"


def save_surface(surface: AgentSurface, directory: Path) -> Path:
    """Write a surface as canonical JSON under `directory`/<store>/<id>.json; an existing file is the same bytes."""
    path = surface_path(directory, surface.store_id, surface.id)
    if not path.is_file():
        atomic_write_text(path, dumps(surface.model_dump(mode="json", by_alias=True)) + "\n")
    return path


def load_surface(directory: Path, store_id: str, surface_id: str) -> AgentSurface:
    path = surface_path(directory, store_id, surface_id)
    try:
        raw = loads(path.read_bytes())
    except FileNotFoundError:
        raise NotFoundError(
            f"no agent surface {surface_id} of {store_id} under {directory}",
            hint="build it again from the live bundles with `carttrace sandbox replica surface`",
        ) from None
    surface = AgentSurface.model_validate(raw)
    if surface.id != surface_id or surface.store_id != store_id:
        raise ValidationError(f"{path} holds surface {surface.id} of {surface.store_id}", hint="rebuild the surface")
    return surface


def newest_surface(directory: Path, store_id: str) -> AgentSurface | None:
    """The store's surface with the latest recording, then the most replies read, then the highest id; None when it
    has none."""
    found = [load_surface(directory, store_id, p.stem) for p in sorted((Path(directory) / store_id).glob("as-*.json"))]
    if not found:
        return None
    return max(found, key=lambda s: (max(t.recorded_at or "" for t in s.tool_lists), s.seen, s.id))


def with_agent_surface(snapshot: StoreSnapshot, surface: AgentSurface) -> StoreSnapshot:
    """A replica's snapshot that lists its store's recorded tools (its pack's `tool_lists`). The provenance names
    the surface."""
    own = str(snapshot.store.get("store_id") or snapshot.store_key)
    if surface.store_id != own and host(str(snapshot.store.get("domain") or "")) not in surface.domains:
        raise ValidationError(
            f"agent surface {surface.id} is of {surface.store_id}, not of {own}",
            hint="pass the replica's own store's surface",
        )
    provenance = {
        **snapshot.provenance,
        "agent_surface": {
            "id": surface.id,
            "lists": [f"{t.mcp}/{t.ucp}" for t in surface.tool_lists],
            "seen": surface.seen,
            "restored": len(surface.restored),
        },
    }
    lists = [t.model_dump(mode="json") for t in surface.tool_lists]
    return snapshot.model_copy(update={"tool_lists": lists, "provenance": provenance})


def _listed(record: WireRecord) -> tuple[Mapping[str, Any], ...] | None:
    """The tools of a countable tools/list exchange: HTTP 200, a result with tools, not a page of a longer list."""
    body = record.request_body
    params = body.get("params") if isinstance(body, Mapping) else None
    if record.status != 200 or record.error or (isinstance(params, Mapping) and params.get("cursor")):
        return None
    reply = record.response_body
    result = reply.get("result") if isinstance(reply, Mapping) and reply.get("error") is None else None
    if not isinstance(result, Mapping) or result.get("nextCursor"):
        return None
    tools = result.get("tools")
    if not isinstance(tools, list):
        return None
    named = tuple(t for t in tools if isinstance(t, Mapping) and isinstance(t.get("name"), str) and t["name"])
    return named or None


def _negotiated(reply: Any) -> str | None:
    result = reply.get("result") if isinstance(reply, Mapping) else None
    version = result.get("protocolVersion") if isinstance(result, Mapping) else None
    return version if isinstance(version, str) and version else None


def _v2_wire(capture_dir: Path) -> Iterator[WireRecord]:
    """A capture v2's wire records, when it kept any."""
    path = Path(capture_dir) / "wire.jsonl.gz"
    if not path.is_file():
        return
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield WireRecord.model_validate(loads(line))

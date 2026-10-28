"""One CartTrace-100k episode, read from the dataset's parquet tables, written back as the bundle files a replay reads.

Why. The dataset ships tables (messages, turns, wire), one row per journey, turn or exchange, while the sandbox writes
and replays bundles (episode.json, turns.jsonl, wire.jsonl, outcome.json). This module joins one episode's rows back
into those files, so `carttrace replay` takes either.

What a replay needs from the tables, and where it comes from:
- each model call's turns, grouped: the turns' `seed` (one per model call, D69);
- which call each tool result answers: the turn's `tool_call_id`;
- which call each wire exchange served: the wire row's `source_call_id`;
- how the shopper's words were made: the turn's `wording` (model or fallback);
- what the replay must reach: the messages row's `final_state` (the sandbox's state hash and the head and length of
  its event chain), with the world, start time and end reason beside it.
A table without `seed` is grouped by the release's turn order instead: a new call starts at an agent turn that follows
the shopper or a system note, which reads most traces right, and the replay's hashes say when it did not.

The release shows each stand-in under its roster domain (`store_domain`, ct-<serial>.example), while its world keeps
the domain it was served under. Where the agent copied that domain into a call (the merchant id of an ACP delegated
payment), replaying the published spelling reaches a different state, so the bundle takes the world's domain back
in the turns and the wire. Both names are public: one is on the row, the other in the world file.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carttrace.core.errors import NotFoundError, ValidationError

TABLES = ("messages", "turns", "wire")


def _rows(release: Path, table: str, episode_id: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.dataset as ds
    except ImportError:  # pragma: no cover - the dataset extra
        raise ValidationError("reading the dataset needs pyarrow", hint="uv sync --extra dataset") from None
    folder = release / "data" / table
    if not folder.is_dir():
        raise NotFoundError(f"no {table} table under {release}/data", hint="download the dataset's data/ folder")
    found = ds.dataset(str(folder), format="parquet").to_table(filter=ds.field("episode_id") == episode_id)
    return list(found.to_pylist())


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def turns_from_rows(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The bundle's turns from the dataset's turn rows, in seq order."""
    out: list[dict[str, Any]] = []
    group, last_actor = 0, None
    for row in sorted(turns, key=lambda r: int(r["seq"])):
        turn: dict[str, Any] = {
            "seq": row["seq"],
            "t_ms": row.get("t_ms"),
            "actor": row.get("actor"),
            "kind": row.get("kind"),
            "text": row.get("text"),
            "tool": row.get("tool"),
            "surface": row.get("surface"),
            "arguments": _json(row.get("arguments")),
            "result": row.get("result")
            if not isinstance(row.get("result"), (dict, list))
            else json.dumps(row["result"]),
            "is_error": row.get("is_error"),
            "error_code": row.get("error_code"),
            "model": row.get("model"),
            "latency_ms": row.get("latency_ms"),
        }
        if row.get("prompt_tokens") is not None or row.get("completion_tokens") is not None:
            turn["usage"] = {k: int(row[k]) for k in ("prompt_tokens", "completion_tokens") if row.get(k) is not None}
        if turn["kind"] == "tool_call":
            turn["call_id"] = f"call-{row['seq']}"
        if turn["kind"] == "tool_result" and row.get("tool_call_id"):
            turn["tool_call_id"] = row["tool_call_id"]
        if row.get("finish_reason") is not None:
            turn["finish_reason"] = row["finish_reason"]
        if turn["actor"] == "shopper" and turn["kind"] == "message" and row.get("wording"):
            turn["origin"] = {"wording": row["wording"]}
        elif turn["actor"] == "shopper" and turn["kind"] == "message" and out:  # not the opening
            turn["origin"] = {"wording": "model"}
        if turn["actor"] == "agent" and turn["kind"] in ("message", "tool_call") and turn["model"]:
            if row.get("seed") is not None:
                turn["seed"] = row["seed"]
            else:
                if last_actor in ("shopper", "system") or group == 0:
                    group += 1
                turn["seed"] = f"group-{group}"
        if turn["actor"] != "tool":
            last_actor = turn["actor"]
        out.append(turn)
    return out


def wire_from_rows(wire: Sequence[Mapping[str, Any]], turns: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The bundle's wire records; a row without `source_call_id` is joined to the last tool call before it."""
    calls = sorted((int(t["seq"]), str(t["call_id"])) for t in turns if t.get("kind") == "tool_call")
    out = []
    for row in sorted(wire, key=lambda r: int(r["seq"])):
        record = {k: _json(v) if k.endswith(("_body", "_headers")) else v for k, v in row.items()}
        if not record.get("source_call_id"):
            before = [c for s, c in calls if s < int(row["seq"])]
            record["source_call_id"] = before[-1] if before else None
        out.append(record)
    return out


def world_domain(worlds: Path | None, world_id: str) -> str | None:
    """The domain `world_id`'s world was served under, from its file in the worlds folder, if there is one."""
    if worlds is None:
        return None
    from carttrace.repro.worlds import world_files

    try:
        envpack, _ = world_files(worlds, world_id)
    except NotFoundError:
        return None
    domain = (json.loads(envpack.read_text(encoding="utf-8")).get("store") or {}).get("domain")
    return str(domain) if domain else None


def served_domain(records: list[dict[str, Any]], published: str | None, served: str | None) -> list[dict[str, Any]]:
    """`records` with the roster domain `published` spelled as the world's `served` domain."""
    if not published or not served or published == served:
        return records
    name = re.compile(rf"(?<![\w.-]){re.escape(published)}(?![\w-])")
    return [json.loads(name.sub(served, json.dumps(r, default=str))) for r in records]


def write_bundle(release: Path, episode_id: str, out: Path, *, worlds: Path | None = None) -> Path:
    """Write `episode_id`'s bundle files from the dataset at `release` into `out`/<episode id>/; returns that folder.
    With `worlds`, the stand-in's roster domain is spelled as its world's domain again."""
    rows = {table: _rows(release, table, episode_id) for table in TABLES}
    if not rows["messages"]:
        raise NotFoundError(f"no episode {episode_id} in {release}", hint="check the id against the messages table")
    episode = rows["messages"][0]
    domains = (episode.get("store_domain"), world_domain(worlds, str(episode["world_id"])))
    turns = served_domain(turns_from_rows(rows["turns"]), *domains)
    folder = out / episode_id
    folder.mkdir(parents=True, exist_ok=True)
    header = {
        "episode_id": episode_id,
        "started_at": episode["started_at"],
        "envpack": episode["world_id"],
    }
    outcome = {"end_reason": episode.get("end_reason"), "state_digest": _json(episode.get("final_state")) or {}}
    (folder / "episode.json").write_text(json.dumps(header, indent=1))
    (folder / "outcome.json").write_text(json.dumps(outcome, indent=1))
    lines = {
        "turns.jsonl": turns,
        "wire.jsonl": served_domain(wire_from_rows(rows["wire"], turns), *domains),
    }
    for name, records in lines.items():
        (folder / name).write_text("".join(json.dumps(r, default=str) + "\n" for r in records))
    return folder

"""An index of the store's complete bundles by mode, kept up to date by reading only what is new.

Why. The replica refreshes (D39 search memories, D40 agent surfaces, D52 sold-out records, D70 store replies) each
read every live bundle, and finding the live ones meant listing the whole archive and reading two small files of every
bundle, sandbox ones included. On 2026-10-14 the archive held 201,461 bundles and 1,374,785 keys: the listing alone
took 154 s and each refresh about 200 s, four of them before every sandbox build, while the collector ran dry (D72).

Here the index remembers every complete bundle it has seen (complete: bundle.json arrived, so nothing more is written
to it) and which of them are live. An update lists only the date folders that can still receive bundles (from the
day before the newest one it closed to today; storage keys are `episodes/<UTC date of started_at>/...`, so an older
folder only grows by an episode that started before its midnight and ended after the next update) and reads the
header of a bundle it has not seen. A bundle still arriving is left for the next update. The first update lists
everything once.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from carttrace.core.canon import dumps, loads
from carttrace.core.errors import ValidationError
from carttrace.core.files import atomic_write_text
from carttrace.trace.layout import EPISODE_FILE, EPISODES_PREFIX, MANIFEST_FILE, TraceStore, parse_json_as
from carttrace.trace.reader import Bundle, open_bundle, scan
from carttrace.trace.schema import EpisodeHeader

INDEX_SCHEMA: Final = "mc.trace.bundle_index.v1"
REOPEN_DAYS = 1
"""Date folders before the newest closed one that are listed again: an episode that started before midnight and
ended after an update lands in the earlier folder."""


class BundleIndex(BaseModel):
    """Complete bundles by date folder, and which are live. Paths are the bundles' storage paths, sorted."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: Literal["mc.trace.bundle_index.v1"] = Field(default="mc.trace.bundle_index.v1", alias="schema")
    prefix: str = EPISODES_PREFIX
    dates: dict[str, list[str]] = Field(default_factory=dict)
    """Date folder (2026-10-14) to its complete bundles' paths."""
    live: list[str] = Field(default_factory=list)
    """The complete live bundles' paths, sorted."""
    updated_at: str = ""

    @property
    def bundles(self) -> int:
        return sum(len(paths) for paths in self.dates.values())


class IndexUpdate(BaseModel):
    """What one update read: the index after it, and how much of the archive it had to look at."""

    model_config = ConfigDict(frozen=True)

    index: BundleIndex
    listed_dates: list[str]
    new_bundles: int
    new_live: list[str]
    arriving: int
    """Bundles seen that are not complete yet."""
    unreadable: int = 0
    """Complete bundles whose header this code cannot read, left out as a scan leaves them."""


def load_index(path: Path) -> BundleIndex | None:
    """The index at `path`, or None when there is none yet. Refuses a file that is not an index."""
    if not path.is_file():
        return None
    raw = loads(path.read_bytes())
    if not isinstance(raw, dict) or raw.get("schema") != INDEX_SCHEMA:
        raise ValidationError(f"{path} is not a {INDEX_SCHEMA} file", hint="remove it; the next update rebuilds it")
    return BundleIndex.model_validate(raw)


def save_index(index: BundleIndex, path: Path) -> Path:
    atomic_write_text(path, dumps(index.model_dump(mode="json", by_alias=True)) + "\n")
    return path


def update_index(store: TraceStore, index: BundleIndex | None, *, now: datetime) -> IndexUpdate:
    """The index with every complete bundle that arrived since `index` was made (all of them when it is None),
    listing only the date folders that can still grow."""
    prefix = index.prefix if index is not None else EPISODES_PREFIX
    dates = {d: set(paths) for d, paths in (index.dates if index is not None else {}).items()}
    live = set(index.live if index is not None else ())
    listing = _listing(store, prefix, sorted(dates), now.date(), full=index is None)
    new_live: list[str] = []
    new = arriving = unreadable = 0
    for day, folder in listing.items():
        known = dates.setdefault(day, set())
        for path, names in sorted(folder.items()):
            if path in known or EPISODE_FILE not in names:
                continue
            if MANIFEST_FILE not in names:
                arriving += 1
                continue
            try:
                header = parse_json_as(EpisodeHeader, store.get_bytes(f"{path}/{EPISODE_FILE}"), where=path)
            except ValidationError:
                unreadable += 1  # written by newer code than this reader, as scan skips it; read again next time
                continue
            known.add(path)
            new += 1
            if header.mode == "live":
                live.add(path)
                new_live.append(path)
    made = BundleIndex(
        prefix=prefix,
        dates={d: sorted(paths) for d, paths in sorted(dates.items()) if paths},
        live=sorted(live),
        updated_at=now.isoformat(),
    )
    return IndexUpdate(
        index=made,
        listed_dates=sorted(listing),
        new_bundles=new,
        new_live=sorted(new_live),
        arriving=arriving,
        unreadable=unreadable,
    )


def _listing(
    store: TraceStore, prefix: str, known: list[str], today: date, *, full: bool
) -> dict[str, dict[str, set[str]]]:
    """Date folder to bundle path to its file names: every folder on a full listing, else the folders from the day
    before the newest known one to today."""
    if full or not known:
        keys: Iterable[str] = store.list(prefix)
    else:
        start = date.fromisoformat(known[-1]) - timedelta(days=REOPEN_DAYS)
        days = [start + timedelta(days=n) for n in range((today - start).days + 1)]
        keys = [key for day in days for key in store.list(f"{prefix}{day.isoformat()}/")]
    out: dict[str, dict[str, set[str]]] = {}
    for key in keys:
        directory, _, name = key.rpartition("/")
        day = directory.removeprefix(prefix).split("/", 1)[0]
        out.setdefault(day, {}).setdefault(directory, set()).add(name)
    return out


def live_bundles(
    store: TraceStore, prefix: str = EPISODES_PREFIX, *, paths: Iterable[str] | None = None
) -> Iterator[Bundle]:
    """Every live bundle under `prefix`, found by a scan of the archive; or, given `paths` (an index's `live`), those
    bundles alone, with no listing of the archive."""
    if paths is None:
        for summary in scan(store, prefix, unreadable=[]):
            if summary.mode == "live":
                yield open_bundle(store, summary.path)
        return
    for path in paths:
        yield open_bundle(store, path)

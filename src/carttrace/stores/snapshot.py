"""Snapshots of the team sheet: `data/stores/<YYYY-MM-DD>/<tab-slug>.csv` plus a `manifest.json` with every sha256.

Why snapshots and not live reads. The store queue is a Google Sheet that people edit all day, but manifests, splits and
benchmarks must name exactly which version of the queue they were built from. `carttrace stores pull` exports the workbook
(a read-only GET of the sheet's xlsx export), parses every tab with the standard-library reader in `carttrace.stores.xlsx`,
and writes one CSV per tab: tables keep only their header and data rows, empty rows are removed, and the manifest
records where each kept row and column sat in the sheet, so the sync can write back by cell. The same workbook bytes
and the same `fetched_at` always give the same files, byte for byte; the raw workbook is not kept, because it also
carries comments and the names of the people who wrote them.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from carttrace.core.canon import pretty, sha256_hex
from carttrace.core.clock import Clock, iso
from carttrace.core.errors import ConfigError, ExternalServiceError, NotFoundError, ValidationError
from carttrace.core.files import atomic_write_bytes, atomic_write_text
from carttrace.stores.settings import StoresSettings
from carttrace.stores.tables import Tab, tab_from_grid
from carttrace.stores.text import slugify
from carttrace.stores.xlsx import column_index, column_letter, read_workbook

SNAPSHOT_SCHEMA = "mc.stores.snapshot.v1"
PARSER = "mc.stores.xlsx/1"
"""Bump when the reader or the tab normalisation changes what a cell becomes."""
MANIFEST = "manifest.json"
_SHEET_ID = re.compile(r"^[A-Za-z0-9_-]{20,}$")
_SNAPSHOT_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EXPORT_HOSTS = ("docs.google.com",)
_EXPORT_HOST_SUFFIXES = (".googleusercontent.com",)


class SourceRef(BaseModel):
    """Where the workbook bytes came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["export", "file"]
    """export: downloaded from the sheet's xlsx export URL; file: a workbook someone exported by hand."""
    url: str
    """The export URL of the sheet, which a file export also came from."""
    sha256: str
    bytes: int


class TabEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    slug: str
    file: str
    layout: Literal["table", "sheet"]
    state: str
    header_row: int | None
    columns: list[str]
    """Sheet column letter of each CSV column."""
    rows: list[tuple[int, int]]
    """Sheet rows of the CSV's data rows, as inclusive runs: [(4, 628)] is rows 4 to 628."""
    row_count: int


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: Literal["mc.stores.snapshot.v1"] = Field(default="mc.stores.snapshot.v1", alias="schema")
    sheet_id: str
    date: str
    """UTC date of `fetched_at`, which is also the snapshot's directory name."""
    fetched_at: str
    source: SourceRef
    parser: str = PARSER
    tabs: list[TabEntry]
    files: dict[str, str]
    """File name to the sha256 of its bytes."""

    def tab(self, name: str) -> TabEntry:
        for entry in self.tabs:
            if entry.name == name:
                return entry
        raise NotFoundError(
            f"snapshot {self.date} has no tab named {name!r}",
            hint=f"tabs are {[t.name for t in self.tabs]}; check stores.queue_tab and stores.evidence_tab in config",
        )


def pull_snapshot(
    settings: StoresSettings,
    *,
    clock: Clock,
    xlsx: Path | None = None,
    fetched_at: datetime | None = None,
    client: httpx.Client | None = None,
) -> tuple[Path, SnapshotManifest]:
    """Export the sheet (or read a workbook exported by hand) and write that day's snapshot. Returns its directory.

    A client passed in must follow redirects; the default one, from carttrace.config.http, does.
    """
    source_kind: Literal["export", "file"]
    if xlsx is not None:
        if not xlsx.is_file():
            raise NotFoundError(f"{xlsx} does not exist", hint="pass the .xlsx file you exported from the sheet")
        data, source_kind = xlsx.read_bytes(), "file"
    elif client is not None:
        data, source_kind = download_workbook(settings.sheet_id, client=client), "export"
    else:
        from carttrace.config.http import sync_client

        with sync_client(user_agent=settings.user_agent, timeout=settings.http_timeout_s, follow_redirects=True) as own:
            data, source_kind = download_workbook(settings.sheet_id, client=own), "export"
    manifest, files = build_snapshot(
        data,
        sheet_id=settings.sheet_id,
        fetched_at=fetched_at or clock.now(),
        source_kind=source_kind,
        anchor=settings.id_column,
    )
    return write_snapshot(settings.stores_dir, manifest, files), manifest


def export_url(sheet_id: str) -> str:
    """The read-only xlsx export of a sheet shared by link."""
    if not _SHEET_ID.match(sheet_id):
        raise ConfigError(f"{sheet_id!r} does not look like a Google Sheets id", hint="set stores.sheet_id in config")
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"


def download_workbook(sheet_id: str, *, client: httpx.Client) -> bytes:
    """GET the xlsx export, retried under the shared policy. Google answers with a redirect to a googleusercontent.com
    download, so the client must follow redirects; the final host is checked before the bytes are trusted."""
    from carttrace.config.http import send_with_retry_sync

    url = export_url(sheet_id)
    try:
        response = send_with_retry_sync(client, client.build_request("GET", url))
    except httpx.HTTPError as exc:
        raise ExternalServiceError(
            "google_sheets", f"export request failed: {type(exc).__name__}", hint="check the network"
        ) from exc
    host = urlsplit(str(response.url)).hostname or ""
    if host not in _EXPORT_HOSTS and not host.endswith(_EXPORT_HOST_SUFFIXES):
        raise ExternalServiceError(
            "google_sheets", f"export ended at an unexpected host {host!r}", hint="pass --xlsx with a file you exported"
        )
    if response.status_code != 200:
        raise ExternalServiceError(
            "google_sheets",
            f"export returned HTTP {response.status_code}",
            hint="the sheet must be shared by link for reading, or pass --xlsx with a file you exported",
            status=response.status_code,
        )
    if not response.content.startswith(b"PK\x03\x04"):
        kind = response.headers.get("content-type", "unknown content")
        raise ExternalServiceError(
            "google_sheets",
            f"export did not return an xlsx workbook ({kind})",
            hint="Google served a sign-in page: share the sheet by link, or pass --xlsx with a file you exported",
        )
    return response.content


def build_snapshot(
    workbook: bytes,
    *,
    sheet_id: str,
    fetched_at: datetime,
    source_kind: Literal["export", "file"],
    anchor: str = "ID",
) -> tuple[SnapshotManifest, dict[str, bytes]]:
    """Parse a workbook into the snapshot's files and manifest, without touching the disk."""
    if fetched_at.tzinfo is None:
        raise ValueError("fetched_at must be timezone-aware")
    parsed = read_workbook(workbook)
    files: dict[str, bytes] = {}
    entries: list[TabEntry] = []
    used: set[str] = set()
    for position, sheet in enumerate(parsed.sheets, start=1):
        tab = tab_from_grid(sheet.name, sheet.rows, merges=sheet.merges, anchor=anchor)
        slug = _unique_slug(slugify(sheet.name) or f"tab-{position}", used)
        file_name = f"{slug}.csv"
        files[file_name] = _csv_bytes(tab.header, tab.rows)
        entries.append(
            TabEntry(
                name=sheet.name,
                slug=slug,
                file=file_name,
                layout=tab.layout,
                state=sheet.state,
                header_row=tab.header_row,
                columns=[column_letter(c) for c in tab.columns],
                rows=_runs(tab.row_numbers),
                row_count=len(tab.rows),
            )
        )
    when = fetched_at.astimezone(UTC)
    manifest = SnapshotManifest(
        sheet_id=sheet_id,
        date=when.date().isoformat(),
        fetched_at=iso(when),
        source=SourceRef(kind=source_kind, url=export_url(sheet_id), sha256=sha256_hex(workbook), bytes=len(workbook)),
        tabs=entries,
        files={name: sha256_hex(data) for name, data in sorted(files.items())},
    )
    return manifest, files


def write_snapshot(stores_dir: Path, manifest: SnapshotManifest, files: Mapping[str, bytes]) -> Path:
    """Write the files, then the manifest last, so a reader never trusts a half-written snapshot. Files an earlier
    pull of the same day listed and this one does not are removed; files the snapshot never listed are left alone."""
    target = stores_dir / manifest.date
    previous = _try_manifest(target)
    for name, data in sorted(files.items()):
        atomic_write_bytes(target / name, data)
    if previous is not None:
        for stale in sorted(set(previous.files) - set(files)):
            (target / stale).unlink(missing_ok=True)
    atomic_write_text(target / MANIFEST, pretty(manifest.model_dump(mode="json", by_alias=True)))
    return target


def snapshot_dirs(stores_dir: Path) -> list[Path]:
    """Every snapshot under `stores_dir`, oldest first."""
    if not stores_dir.is_dir():
        return []
    return sorted(p for p in stores_dir.iterdir() if _SNAPSHOT_DIR.match(p.name) and (p / MANIFEST).is_file())


def latest_snapshot(stores_dir: Path) -> Path:
    found = snapshot_dirs(stores_dir)
    if not found:
        raise NotFoundError(f"no store snapshot under {stores_dir}", hint="run `carttrace stores pull`")
    return found[-1]


def read_manifest(snapshot_dir: Path) -> SnapshotManifest:
    path = snapshot_dir / MANIFEST
    if not path.is_file():
        raise NotFoundError(f"{path} does not exist", hint="pass a directory written by `carttrace stores pull`")
    try:
        return SnapshotManifest.model_validate_json(path.read_bytes())
    except PydanticValidationError as exc:
        raise ValidationError(f"{path} is not a {SNAPSHOT_SCHEMA} manifest: {exc.errors()[0]['msg']}") from exc


def verify_snapshot(snapshot_dir: Path) -> SnapshotManifest:
    """The manifest, after checking that every file still has the sha256 it was written with."""
    manifest = read_manifest(snapshot_dir)
    for name in manifest.files:
        _checked_bytes(snapshot_dir, manifest, name)
    return manifest


def read_snapshot_tab(snapshot_dir: Path, name: str) -> Tab:
    """One tab of a snapshot, read back with its sheet rows and columns. Refuses a file whose bytes changed."""
    manifest = read_manifest(snapshot_dir)
    entry = manifest.tab(name)
    data = _checked_bytes(snapshot_dir, manifest, entry.file)
    lines = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
    header, rows = (tuple(lines[0]), lines[1:]) if lines else ((), [])
    numbers = tuple(n for first, last in entry.rows for n in range(first, last + 1))
    if len(numbers) != len(rows) or any(len(row) != len(header) for row in rows):
        raise ValidationError(f"{entry.file} does not match its manifest entry", hint="re-run `carttrace stores pull`")
    return Tab(
        name=entry.name,
        layout=entry.layout,
        header=header,
        columns=tuple(column_index(letter) for letter in entry.columns),
        header_row=entry.header_row,
        rows=tuple(tuple(row) for row in rows),
        row_numbers=numbers,
    )


def _checked_bytes(snapshot_dir: Path, manifest: SnapshotManifest, name: str) -> bytes:
    path = snapshot_dir / name
    if not path.is_file():
        raise NotFoundError(f"snapshot file {path} is missing", hint="restore it from git or re-run `carttrace stores pull`")
    data = path.read_bytes()
    if sha256_hex(data) != manifest.files.get(name):
        raise ValidationError(
            f"{path} changed since it was pulled (sha256 does not match manifest.json)",
            hint="snapshots are read-only: restore the file from git, or re-run `carttrace stores pull`",
        )
    return data


def _try_manifest(snapshot_dir: Path) -> SnapshotManifest | None:
    try:
        return read_manifest(snapshot_dir)
    except NotFoundError:
        return None


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _runs(numbers: Sequence[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for n in numbers:
        if runs and n == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], n)
        else:
            runs.append((n, n))
    return runs


def _unique_slug(slug: str, used: set[str]) -> str:
    candidate, n = slug, 1
    while candidate in used:
        n += 1
        candidate = f"{slug}-{n}"
    used.add(candidate)
    return candidate

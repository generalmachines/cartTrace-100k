"""Filling vendor/ from the lock, checking it offline, and relocking after a pin changes.

Why three operations. CI and every test need an offline answer to "is vendor/ exactly what the lock says"
(`check`). A fresh checkout or a damaged tree needs the files back, byte for byte, from the pinned upstream refs
(`vendor`, which writes nothing for a source whose bytes differ from the lock, because a moved tag or a re-uploaded
file is exactly what the lock exists to catch). A person who changes a pin needs the file list and hashes rewritten
(`vendor(relock=True)`), and nothing else in the lock may change when they do. A url source that cannot be fetched
during a relock is recorded with its error and the run carries on, because a payment handler's site being down must
not block vendoring the protocols.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

import httpx
from pydantic import ValidationError as PydanticValidationError

from carttrace.core.canon import sha256_hex
from carttrace.core.errors import ExternalServiceError, MCError, ProtocolError, ValidationError
from carttrace.core.files import atomic_write_bytes
from carttrace.protocols.fetch import Fetched, Fetcher, GitFetcher, UrlFetcher, http_client
from carttrace.protocols.lock import CheckReport, LockedFile, Source, check_vendored, read_lock, write_lock
from carttrace.protocols.paths import ProtocolPaths

Status = Literal["ok", "relocked", "failed", "unavailable"]


@dataclass(frozen=True, slots=True)
class SourceResult:
    id: str
    status: Status
    message: str = ""


@dataclass(slots=True)
class VendorReport:
    results: list[SourceResult] = field(default_factory=list)
    check: CheckReport | None = None
    lock_written: bool = False

    @property
    def ok(self) -> bool:
        failed = any(r.status == "failed" for r in self.results)
        return not failed and (self.check is None or self.check.ok)

    def lines(self) -> list[str]:
        out = [f"{r.id}: {r.status}" + (f" ({r.message})" if r.message else "") for r in self.results]
        if self.check is not None:
            out += self.check.problems + [f"warning: {w}" for w in self.check.warnings] + [self.check.summary()]
        return out


def check(paths: ProtocolPaths, *, only: Iterable[str] = ()) -> CheckReport:
    """Offline: every locked file present with its sha256, and nothing unlocked under vendor/."""
    lock = read_lock(paths.lock)
    wanted = list(only)
    return check_vendored(lock, paths.root, lock.select(wanted) if wanted else None)


def fetch(
    paths: ProtocolPaths, *, only: Iterable[str] = (), git: Fetcher | None = None, http: httpx.Client | None = None
) -> VendorReport:
    """Fetch every selected source at its pin, verify each file's sha256 against the lock, and write vendor/.

    A source whose upstream bytes differ from the lock is reported and left untouched on disk.
    """
    return _run(paths, only=only, relock=False, git=git, http=http)


def relock(
    paths: ProtocolPaths, *, only: Iterable[str] = (), git: Fetcher | None = None, http: httpx.Client | None = None
) -> VendorReport:
    """Refetch every selected source at its pin, rewrite its files and hashes in the lock, and write vendor/.

    A tag is resolved to its current commit, which the lock records in full. A url source that cannot be fetched keeps
    its previous files and records the error, and the run carries on.
    """
    return _run(paths, only=only, relock=True, git=git, http=http)


def _run(
    paths: ProtocolPaths,
    *,
    only: Iterable[str],
    relock: bool,
    git: Fetcher | None,
    http: httpx.Client | None,
) -> VendorReport:
    wanted = list(only)
    lock = read_lock(paths.lock)
    selected = lock.select(wanted)
    report = VendorReport()
    updated: list[Source] = []
    with _fetchers(git, http) as fetchers:
        for src in selected:
            result, new_src = _one(src, fetchers[src.kind], paths.root, relock=relock)
            report.results.append(result)
            if new_src is not None:
                updated.append(new_src)
    if relock:
        lock = lock.replace(updated)
        write_lock(lock, paths.lock)
        report.lock_written = True
    report.check = check_vendored(lock, paths.root, lock.select(wanted) if wanted else None)
    return report


@contextmanager
def _fetchers(git: Fetcher | None, http: httpx.Client | None) -> Iterator[dict[str, Fetcher]]:
    with tempfile.TemporaryDirectory(prefix="mc-protocols-") as tmp:
        client = http or http_client()
        try:
            yield {"git": git or GitFetcher(Path(tmp)), "url": UrlFetcher(client)}
        finally:
            if http is None:
                client.close()


def _one(src: Source, fetcher: Fetcher, root: Path, *, relock: bool) -> tuple[SourceResult, Source | None]:
    """Fetch one source; returns its result and, on relock, its rewritten lock entry."""
    if not relock and not src.files:
        if src.error:
            return SourceResult(src.id, "unavailable", f"nothing locked; the last relock failed: {src.error}"), None
        return SourceResult(src.id, "failed", f"nothing locked; run --relock --only {src.id}"), None
    try:
        fetched = fetcher.fetch(src, discover=relock)
        if relock:
            new_src, note = _relocked(src, fetched)
        else:
            _verify(src, fetched)
            new_src, note = None, ""
        _write_dest(root / src.dest, fetched.files)
    except MCError as exc:
        if relock and src.kind == "url" and isinstance(exc, ExternalServiceError):
            return SourceResult(src.id, "unavailable", str(exc)), src.model_copy(update={"error": exc.message})
        return SourceResult(src.id, "failed", str(exc)), None
    return SourceResult(src.id, "relocked" if relock else "ok", note), new_src


def _relocked(src: Source, fetched: Fetched) -> tuple[Source, str]:
    if src.version_from == "commit_date" and fetched.commit_date != src.version:
        raise ValidationError(
            f"{src.id}: the pinned commit is dated {fetched.commit_date}, the lock says version {src.version}",
            hint=f"set version = {fetched.commit_date!r} and dest to match, then relock",
        )
    note = ""
    if src.kind == "git" and src.commit and fetched.commit != src.commit:
        note = f"commit moved from {src.commit[:12]} to {(fetched.commit or '')[:12]}"
    try:
        files = [
            LockedFile(path=rel, sha256=sha256_hex(data), url=fetched.urls.get(rel))
            for rel, data in sorted(fetched.files.items())
        ]
        new = src.model_copy(
            update={"files": files, "commit": fetched.commit if src.kind == "git" else None, "error": None}
        )
        return Source.model_validate(new.model_dump()), note
    except PydanticValidationError as exc:
        raise ValidationError(f"{src.id}: upstream produced an unusable file list: {exc.errors()[0]['msg']}") from exc


def _verify(src: Source, fetched: Fetched) -> None:
    """Refuse upstream bytes the lock does not vouch for."""
    if src.kind == "git" and src.version_from == "commit_date" and fetched.commit_date != src.version:
        raise ProtocolError(f"{src.id}: commit is dated {fetched.commit_date}, not {src.version}")
    bad = [f.path for f in src.files if sha256_hex(fetched.files[f.path]) != f.sha256]
    if bad:
        raise ProtocolError(
            f"{src.id}: upstream bytes differ from the lock for {len(bad)} files, first {bad[0]}",
            hint=f"nothing was written; if upstream changed on purpose, relock with --only {src.id}",
        )


def _write_dest(dest: Path, files: dict[str, bytes]) -> None:
    """Replace a source's directory wholesale, so a file dropped upstream does not linger here."""
    for rel in files:
        parts = PurePosixPath(rel).parts
        if not rel or rel.startswith("/") or ".." in parts:
            raise ValidationError(f"refusing to write {rel!r} outside {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    try:
        for rel, data in sorted(files.items()):
            atomic_write_bytes(staging / rel, data)
        old = None
        if dest.exists():
            old = dest.with_name(f"{staging.name}.old")
            dest.rename(old)
        staging.rename(dest)
        if old is not None:
            shutil.rmtree(old)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

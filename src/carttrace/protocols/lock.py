"""The pins in protocols.lock.toml: where every vendored protocol file came from and what it hashes to.

Why a lock. Every schema the sandbox validates against, every tool list a surface shows the agent and every diff the
benchmark reports rests on files copied from upstream repositories and sites. Before this lock, the lab found those
files by hand on whatever day someone looked, and two agents reading the same spec got different enums. The lock names
one upstream ref per source (a tag with its full commit sha, a bare commit, or a URL) and the sha256 of every file, so
a moved tag or a re-uploaded file fails the check instead of quietly changing what a version means.

The file is TOML so it reads with the standard library. Only `dump_lock` writes it: one fixed shape, sources sorted by
family and version, files sorted by path, and a round-trip check before anything reaches disk, so a relock diff shows
only what changed upstream.
"""

from __future__ import annotations

import itertools
import json
import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError

from carttrace.core.canon import sha256_hex
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.files import atomic_write_text
from carttrace.protocols.versions import version_key

SourceKind = Literal["git", "url"]
VENDOR_README = "README.md"
"""The one file allowed directly under vendor/ that no source owns: it tells a reader the tree is generated."""

HEADER = """\
# protocols.lock.toml: every vendored protocol file, where it came from, and what it hashes to.
#
# Written by `carttrace protocols vendor --relock`; checked offline by `carttrace protocols vendor --check` and by CI.
# To add or bump a protocol, edit a pin (tag, commit, url, include, version, dest) by hand, then run
# `carttrace protocols vendor --relock --only <id>`; it refetches at the pin, records the full commit sha and rehashes.
# A url source whose fetch fails records `error` and keeps the files it last had; the check warns and carries on.
"""

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9.-]*$")

# The order fields are written in. Anything not listed here is not part of the lock format.
_SOURCE_FIELDS = (
    "id",
    "protocol",
    "version",
    "kind",
    "repo",
    "tag",
    "commit",
    "url",
    "include",
    "follow_refs",
    "version_from",
    "dest",
    "note",
    "error",
    "files",
)
_FILE_FIELDS = ("path", "url", "sha256")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LockedFile(_Frozen):
    path: str
    """POSIX path relative to the source's `dest`."""
    sha256: str
    url: str | None = None
    """Where a url source downloaded this file from. Unset for git sources, whose files come from the checkout."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_relpath(self.path, what="locked file")
        if not _SHA256.match(self.sha256):
            raise ValueError(f"{self.path}: sha256 must be 64 lowercase hex characters")
        return self


class Source(_Frozen):
    """One upstream origin of vendored files: a git checkout at a pinned commit, or a set of URLs."""

    id: str
    protocol: str
    """The family, which is also the directory under vendor/: ucp, acp, ap2, mcp, mpp, webmcp, or a payment-handler
    namespace such as com.google.pay."""
    version: str
    kind: SourceKind
    repo: str | None = None
    tag: str | None = None
    commit: str | None = None
    """Full 40-character commit sha. A tag is resolved to it on relock and checked against it on every fetch."""
    url: str | None = None
    """url sources: the primary document. Its directory is the base that `follow_refs` stays inside."""
    include: list[str] = Field(default_factory=list)
    """git sources: glob patterns relative to the repository root; each must match at least one file."""
    follow_refs: bool = False
    """url sources: also fetch every `$ref` target under the primary document's directory, recursively."""
    version_from: Literal["commit_date"] | None = None
    """git sources whose version is the commit's UTC date (WebMCP has no releases)."""
    dest: str
    """Directory relative to the repository root, under vendor/<protocol>/<version>."""
    note: str = ""
    error: str | None = None
    """url sources: why the last relock could not fetch. Files stay as last locked (none if never fetched)."""
    files: list[LockedFile] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not _ID.match(self.id):
            raise ValueError(f"source id {self.id!r} must be lowercase letters, digits, dots and dashes")
        _check_relpath(self.dest, what=f"{self.id}: dest")
        if self.kind == "git":
            if not self.repo or self.url:
                raise ValueError(f"{self.id}: a git source needs repo and no url")
            if not (self.tag or self.commit):
                raise ValueError(f"{self.id}: a git source needs a tag or a commit")
            if self.commit is not None and not _SHA1.match(self.commit):
                raise ValueError(f"{self.id}: commit must be the full 40-character sha, got {self.commit!r}")
            if not self.include:
                raise ValueError(f"{self.id}: a git source needs include globs")
            for pattern in self.include:
                _check_relpath(pattern, what=f"{self.id}: include")
        else:
            if not self.url or self.repo or self.tag or self.commit or self.include:
                raise ValueError(f"{self.id}: a url source needs url and no repo, tag, commit or include")
            if self.version_from is not None:
                raise ValueError(f"{self.id}: version_from applies to git sources only")
        paths = [f.path for f in self.files]
        if paths != sorted(set(paths)):
            raise ValueError(f"{self.id}: files must be unique and sorted by path")
        return self

    @property
    def ref(self) -> str:
        """The upstream ref in one phrase, for messages: a short commit with its tag, or the URL."""
        if self.kind == "url":
            return str(self.url)
        sha = (self.commit or "")[:12] or "unresolved"
        return f"{self.tag} ({sha})" if self.tag else sha


class Lock(_Frozen):
    lock_version: Literal[1] = 1
    source: list[Source] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = [s.id for s in self.source]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate source ids: {dupes}")
        dests = sorted((PurePosixPath(s.dest), s.id) for s in self.source)
        for (a, a_id), (b, b_id) in itertools.pairwise(dests):
            if a == b or a in b.parents:
                raise ValueError(f"sources {a_id} and {b_id} write overlapping destinations {a} and {b}")
        return self

    def get(self, source_id: str) -> Source:
        for s in self.source:
            if s.id == source_id:
                return s
        raise NotFoundError(f"no source {source_id!r} in the lock", hint="run `carttrace protocols list` to see the ids")

    def select(self, only: Iterable[str] = ()) -> list[Source]:
        """The sources named in `only` (every source when it is empty), in lock order. Unknown ids raise."""
        wanted = list(only)
        if not wanted:
            return list(self.source)
        chosen = {self.get(i).id for i in wanted}
        return [s for s in self.source if s.id in chosen]

    def replace(self, updated: Iterable[Source]) -> Lock:
        """This lock with some sources swapped for new versions of themselves (matched by id)."""
        by_id = {s.id: s for s in updated}
        return Lock(source=[by_id.get(s.id, s) for s in self.source])

    def sorted(self) -> Lock:
        return Lock(source=sorted(self.source, key=lambda s: (s.protocol, version_key(s.version), s.id)))


def _check_relpath(value: str, *, what: str) -> None:
    p = PurePosixPath(value)
    if not value or value.startswith("/") or "\\" in value or ".." in p.parts or str(p) != value.rstrip("/"):
        raise ValueError(f"{what} {value!r} must be a relative POSIX path without '..'")


# ---------------------------------------------------------------------------------------------------------------------
# Reading and writing


def read_lock(path: Path) -> Lock:
    if not path.is_file():
        raise NotFoundError(f"no protocol lock at {path}", hint="run from the repository root, or restore the lock")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        return Lock.model_validate(raw)
    except (tomllib.TOMLDecodeError, PydanticValidationError) as exc:
        raise ValidationError(f"{path.name} is malformed: {exc}", hint="fix the pin by hand, then relock") from exc


def dump_lock(lock: Lock) -> str:
    """The lock as TOML text: header, version, then each source in sorted order with its fields in a fixed order."""
    out = [HEADER, f"lock_version = {lock.lock_version}"]
    for src in lock.sorted().source:
        data = src.model_dump(exclude_defaults=True)
        out += ["", "[[source]]"]
        for key in _SOURCE_FIELDS:
            if key == "files":
                out += _files_lines(src.files)
            elif key in data:
                out.append(f"{key} = {_scalar(data[key])}")
    text = "\n".join(out) + "\n"
    if Lock.model_validate(tomllib.loads(text)) != lock.sorted():
        raise ValidationError("the lock writer did not round-trip", hint="this is a bug in carttrace.protocols.lock")
    return text


def write_lock(lock: Lock, path: Path) -> None:
    atomic_write_text(path, dump_lock(lock))


def _files_lines(files: list[LockedFile]) -> list[str]:
    if not files:
        return ["files = []"]
    rows = []
    for f in files:
        data = f.model_dump(exclude_none=True)
        rows.append("  { " + ", ".join(f"{k} = {_scalar(data[k])}" for k in _FILE_FIELDS if k in data) + " },")
    return ["files = [", *rows, "]"]


def _scalar(value: Any) -> str:
    """A TOML value. A JSON string literal is a valid TOML basic string, escapes included."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "[" + ", ".join(_scalar(v) for v in value) + "]"
    raise TypeError(f"the lock has no TOML form for {value!r}")


# ---------------------------------------------------------------------------------------------------------------------
# The offline check


@dataclass(slots=True)
class CheckReport:
    """What `--check` found. Problems fail the check; warnings (recorded upstream outages) do not."""

    sources: int = 0
    files: int = 0
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        head = f"{self.files} files in {self.sources} sources"
        if self.problems:
            return f"lock check failed: {len(self.problems)} problems ({head})"
        return f"ok: {head} match the lock" + (f", {len(self.warnings)} warnings" if self.warnings else "")


def check_vendored(lock: Lock, root: Path, sources: Iterable[Source] | None = None) -> CheckReport:
    """Compare the vendored tree with the lock without touching the network.

    Every locked file must exist with its sha256, every file under a source's dest must be locked, and every file
    under vendor/ must belong to some source. A url source with a recorded fetch error and no files is a warning.
    """
    chosen = lock.source if sources is None else list(sources)
    report = CheckReport(sources=len(chosen))
    for src in chosen:
        _check_source(src, root, report)
    if sources is None:
        _check_strays(lock, root, report)
    report.problems.sort()
    report.warnings.sort()
    return report


def _check_source(src: Source, root: Path, report: CheckReport) -> None:
    expected_prefix = PurePosixPath("vendor", src.protocol, src.version)
    dest_rel = PurePosixPath(src.dest)
    if dest_rel != expected_prefix and expected_prefix not in dest_rel.parents:
        report.problems.append(f"{src.id}: dest {src.dest} is not under {expected_prefix}")
    if src.error:
        report.warnings.append(f"{src.id}: last relock could not fetch {src.ref}: {src.error}")
    if not src.files:
        if not src.error:
            report.problems.append(f"{src.id}: no files locked (run `carttrace protocols vendor --relock --only {src.id}`)")
        return
    dest = root / src.dest
    locked = {f.path: f.sha256 for f in src.files}
    for rel, sha in locked.items():
        p = dest / rel
        if not p.is_file():
            report.problems.append(f"{src.id}: missing {src.dest}/{rel}")
        elif sha256_hex(p.read_bytes()) != sha:
            report.problems.append(f"{src.id}: sha256 mismatch for {src.dest}/{rel}")
        else:
            report.files += 1
    if dest.is_dir():
        for p in sorted(dest.rglob("*")):
            if p.is_file() and p.relative_to(dest).as_posix() not in locked:
                report.problems.append(f"{src.id}: unlocked file {p.relative_to(root).as_posix()}")


def _check_strays(lock: Lock, root: Path, report: CheckReport) -> None:
    vendor = root / "vendor"
    if not vendor.is_dir():
        return
    dests = [PurePosixPath(s.dest) for s in lock.source]
    for p in sorted(vendor.rglob("*")):
        if not p.is_file():
            continue
        rel = PurePosixPath(p.relative_to(root).as_posix())
        if rel == PurePosixPath("vendor", VENDOR_README):
            continue
        if not any(d in rel.parents for d in dests):
            report.problems.append(f"no source owns {rel}")

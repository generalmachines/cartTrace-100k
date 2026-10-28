"""Atomic writes and JSONL, the two ways run output reaches disk.

Why atomic. A runner killed mid-write must never leave a half file that a reader then trusts: a status file read by the
watch, a manifest read by a resumed run, an outcome read by the judge. Whole files are written to a temporary sibling
and renamed into place, which POSIX makes atomic. The temporary is created with mode 0o666 so the kernel applies the
process umask, as it does for any other file the process creates. tempfile.mkstemp creates 0o600, and the rename
carried that to the target, so checked-in files such as config.schema.json came out readable by their owner alone.

Why JSONL. Append-only logs (turns, wire, decisions) are one canonical line per record, flushed per line, and a reader
skips a torn last line instead of failing. A log that several processes append to (the shared decision log) takes an
exclusive flock on the log file itself for each append, so two writers' lines never interleave.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from carttrace.core.canon import dumps_line
from carttrace.core.errors import ValidationError

_TEMP_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
_TEMP_NUMBERS = itertools.count()
"""Numbers the temporaries of one process; with the pid they make names that no other live writer uses."""


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """`data` at `path`, whole or not at all. The file ends with mode 0o666 less the process umask, like any file
    the process creates, whatever mode an earlier version of it had."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = _create_temporary(path)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def _create_temporary(path: Path) -> tuple[int, Path]:
    """A new dot-named file beside `path`, open for writing. O_EXCL steps past a leftover that a crashed process with
    the same pid left under a name, instead of writing into it."""
    while True:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{next(_TEMP_NUMBERS)}")
        try:
            return os.open(tmp, _TEMP_FLAGS, 0o666), tmp
        except FileExistsError:
            continue


def append_jsonl(path: Path, record: Any) -> None:
    """Append one canonical JSON line and flush it. The caller owns ordering; concurrent writers need their own file,
    or `append_jsonl_locked`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(dumps_line(record) + "\n")
        fh.flush()


def append_jsonl_locked(path: Path, record: Any) -> None:
    """`append_jsonl` for a log that several processes append to: the line is written under an exclusive flock on
    the file itself, held for this one append and dropped when the file closes.

    POSIX advisory locking, so it keeps out only writers that take the same lock, as every caller of this function
    does. No lock file is left beside the log. A record that is not canonical JSON is refused before the file is
    opened.
    """
    import fcntl  # POSIX only, so imported here: carttrace.core itself must import anywhere

    line = dumps_line(record) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(line)
        fh.flush()


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """An exclusive flock on `path` (created empty if missing) for the block: one writer at a time across processes
    for whatever the lock guards, such as the results database beside it. Advisory, like `append_jsonl_locked`."""
    import fcntl  # POSIX only, so imported here: carttrace.core itself must import anywhere

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class JsonlError(ValidationError, ValueError):
    """A JSONL line that is neither a record nor a torn last line, so the file is not ours or was edited.

    Also a ValueError, as json.JSONDecodeError is, so handlers written for that still catch it.
    """

    def __init__(self, line: int, problem: str, *, source: str = "") -> None:
        where = f"{source} line {line}" if source else f"line {line}"
        super().__init__(
            f"{where} is {problem}",
            hint="JSONL logs are appended one canonical line at a time and never edited; only a torn last line is "
            "skipped, so find what else wrote to this file",
        )
        self.line = line
        """1-based line number in the file."""
        self.problem = problem
        """What is wrong with the line: "not UTF-8", or "not JSON: <reason>"."""
        self.source = source


def iter_jsonl_numbered(data: bytes, *, source: str = "") -> Iterator[tuple[int, Any]]:
    """(line number, record) for every complete record in JSONL bytes, in order, counting lines from 1.

    Blank lines are skipped. A last line that does not parse is a writer killed mid-append, and is skipped too; it
    may even stop inside a multi-byte character, which is why each line is decoded on its own. Any other line that is
    not UTF-8 JSON raises JsonlError, naming `source` and the line.
    """
    lines = data.split(b"\n")
    for number, raw in enumerate(lines, start=1):
        try:
            text = raw.decode("utf-8")
            if not text.strip():
                continue
            record = json.loads(text)
        except UnicodeDecodeError as exc:
            if number == len(lines):
                return
            raise JsonlError(number, "not UTF-8", source=source) from exc
        except json.JSONDecodeError as exc:
            if number == len(lines):
                return
            raise JsonlError(number, f"not JSON: {exc.msg}", source=source) from exc
        yield number, record


def iter_jsonl_bytes(data: bytes, *, source: str = "") -> Iterator[Any]:
    """Every complete record in JSONL bytes, in order: a torn last line is skipped, any other bad line raises
    JsonlError. See `iter_jsonl_numbered`, which also yields each record's line number."""
    for _, record in iter_jsonl_numbered(data, source=source):
        yield record


def read_jsonl(path: Path) -> Iterator[Any]:
    """Every complete record of a JSONL file in order, or nothing when the file does not exist. A torn final line (a
    writer killed mid-append) is skipped; any other bad line raises JsonlError naming the file and the line."""
    if not path.exists():
        return
    yield from iter_jsonl_bytes(path.read_bytes(), source=str(path))

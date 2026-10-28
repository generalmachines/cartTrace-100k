"""Verified downloads of catalog files: streamed to disk, hashed on the way, refused unless they match their pin.

Why our own loop. Catalog dumps are large (the Open Food Facts CSV is over a gigabyte), so a file is streamed to a
temporary sibling and never held in memory, which `carttrace.config.http.send_with_retry` (it reads whole bodies) cannot
do. Everything else is the repository's: the client comes from `carttrace.config.http.sync_client` (our user agent,
explicit timeouts), and retries follow its `RetryPolicy` (a dropped connection or a 429/5xx is tried again with
deterministic backoff; a long Retry-After is not waited out).

What is refused, before anything lands under its real name:
- a URL that is not https, or a redirect that leaves https;
- a body larger than the source's budget, by Content-Length up front or by count while streaming;
- bytes whose sha256 or size differ from the manifest's pin (a moved upstream file must be taken with `--relock`).
A refused download leaves nothing behind. A file already on disk with the pinned hash is not downloaded again.

`import_local` takes the same file from a local path (an existing copy, a mirror) under the same checks, linking it
when the filesystem allows so a gigabyte is not copied twice. `RawDigests` caches each raw file's sha256 by size and
modification time, so checking an index against its inputs does not re-read gigabytes.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from carttrace.config.http import RetryPolicy, retry_after_s, sync_client
from carttrace.core.canon import loads, pretty
from carttrace.core.errors import ExternalServiceError, ValidationError
from carttrace.core.files import atomic_write_text
from carttrace.sandbox.catalogs.errors import BudgetError, IntegrityError

CHUNK = 1 << 20
READ_TIMEOUT_S = 120.0
DIGESTS_FILE = ".sha256.json"
Progress = Callable[[str, int, int | None], None]
"""Called as bytes arrive: (file name, bytes so far, expected total or None)."""
Status = Literal["downloaded", "present", "imported"]


@dataclass(frozen=True, slots=True)
class Fetched:
    """One file on disk, verified."""

    name: str
    path: Path
    sha256: str
    size: int
    status: Status
    version_id: str | None = None
    """The upstream object version (S3 x-amz-version-id) when the server reported one."""
    final_url: str | None = None


def http_client(*, user_agent: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """The client catalog downloads use: the repository's one factory, following redirects (dataset hosts redirect
    to their storage buckets), which `download` then requires to stay on https."""
    return sync_client(
        user_agent=user_agent,
        timeout=httpx.Timeout(READ_TIMEOUT_S, connect=10.0),
        transport=transport,
        follow_redirects=True,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def check_pin(name: str, sha256: str, size: int, *, want_sha256: str | None, want_size: int | None) -> None:
    """Raise IntegrityError unless the bytes are the pinned ones. A file without a pin passes (first relock)."""
    if want_size is not None and size != want_size:
        raise IntegrityError(
            f"{name}: {size} bytes, the manifest pins {want_size}",
            hint="upstream changed or the download is truncated; to take the new file, fetch with --relock",
        )
    if want_sha256 is not None and sha256 != want_sha256:
        raise IntegrityError(
            f"{name}: sha256 {sha256[:16]}..., the manifest pins {want_sha256[:16]}...",
            hint="upstream replaced the file; to take the new file, fetch with --relock",
        )


def download(
    client: httpx.Client,
    url: str,
    dest: Path,
    *,
    want_sha256: str | None,
    want_size: int | None,
    budget: int,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    progress: Progress | None = None,
) -> Fetched:
    """Download `url` to `dest`, verified against the pin (when given) and the byte budget."""
    if not url.startswith("https://"):
        raise ValidationError(f"refusing to fetch {url}: catalog files come over https only")
    if dest.exists() and want_sha256 is not None:
        size = dest.stat().st_size
        if size == want_size or want_size is None:
            sha = sha256_file(dest)
            if sha == want_sha256:
                return Fetched(dest.name, dest, sha, size, "present")
    policy = policy or RetryPolicy(attempts=4, max_delay_s=30.0)
    host = httpx.URL(url).host
    attempt = 0
    while True:
        attempt += 1
        try:
            fetched = _stream_once(client, url, dest, budget=budget, want_size=want_size, progress=progress)
        except _Retryable as exc:
            delay = policy.next_delay(attempt, retry_after_s=exc.retry_after)
            if delay is None:
                raise ExternalServiceError(host, exc.reason, status=exc.status, hint="try again later") from None
            sleep(delay)
            continue
        except httpx.TransportError as exc:
            delay = policy.next_delay(attempt) if policy.retries_error(exc, idempotent=True) else None
            if delay is None:
                raise ExternalServiceError(host, f"GET {url} failed: {exc}", hint="check the network") from exc
            sleep(delay)
            continue
        try:
            check_pin(dest.name, fetched.sha256, fetched.size, want_sha256=want_sha256, want_size=want_size)
        except IntegrityError:
            fetched.path.unlink(missing_ok=True)
            raise
        fetched.path.replace(dest)
        return Fetched(dest.name, dest, fetched.sha256, fetched.size, "downloaded", fetched.version_id, fetched.url)


class _Retryable(Exception):
    def __init__(self, reason: str, *, status: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class _Part:
    path: Path
    sha256: str
    size: int
    version_id: str | None
    url: str


def _stream_once(
    client: httpx.Client, url: str, dest: Path, *, budget: int, want_size: int | None, progress: Progress | None
) -> _Part:
    """One GET streamed to `.<name>.part` beside `dest`. The part file is removed on any failure."""
    part = dest.with_name(f".{dest.name}.part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    host = httpx.URL(url).host
    with client.stream("GET", url) as response:
        status = response.status_code
        if status in (429, 500, 502, 503, 504):
            raise _Retryable(f"HTTP {status} for {url}", status=status, retry_after=retry_after_s(response))
        if status != 200:
            raise ExternalServiceError(
                host, f"HTTP {status} for {url}", status=status, hint="the file moved upstream; fix its url"
            )
        final = str(response.url)
        if not final.startswith("https://"):
            raise ExternalServiceError(host, f"{url} redirected off https to {final}")
        declared = response.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > budget:
            raise BudgetError(
                f"{dest.name}: {int(declared)} bytes is over the source's budget of {budget}",
                hint="raise budget_bytes in the source manifest only if the larger file is intended",
            )
        h = hashlib.sha256()
        size = 0
        try:
            with part.open("wb") as fh:
                for chunk in response.iter_bytes(CHUNK):
                    size += len(chunk)
                    if size > budget:
                        raise BudgetError(
                            f"{dest.name}: more than {budget} bytes, the source's budget",
                            hint="raise budget_bytes in the source manifest only if the larger file is intended",
                        )
                    h.update(chunk)
                    fh.write(chunk)
                    if progress is not None:
                        progress(dest.name, size, want_size)
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        version = response.headers.get("x-amz-version-id")
    return _Part(part, h.hexdigest(), size, version if version and version != "null" else None, final)


def import_local(source: Path, dest: Path, *, want_sha256: str | None, want_size: int | None, budget: int) -> Fetched:
    """Take a raw file from a local path under the same checks as a download, as a hard link when possible."""
    if not source.is_file():
        raise IntegrityError(f"{source} is not a file", hint="pass the path of the downloaded file to --from")
    size = source.stat().st_size
    if size > budget:
        raise BudgetError(f"{source.name}: {size} bytes is over the source's budget of {budget}")
    sha = sha256_file(source)
    check_pin(dest.name, sha, size, want_sha256=want_sha256, want_size=want_size)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.samefile(source):
        return Fetched(dest.name, dest, sha, size, "present")
    part = dest.with_name(f".{dest.name}.part")
    part.unlink(missing_ok=True)
    try:
        os.link(source, part)
    except OSError:
        shutil.copyfile(source, part)
    part.replace(dest)
    return Fetched(dest.name, dest, sha, size, "imported")


class RawDigests:
    """sha256 of each raw file of a source, cached in `<raw>/.sha256.json` by size and modification time."""

    def __init__(self, raw_dir: Path) -> None:
        self._dir = raw_dir
        self._path = raw_dir / DIGESTS_FILE

    def _load(self) -> dict[str, dict[str, object]]:
        try:
            data = loads(self._path.read_bytes())
        except (FileNotFoundError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def sha256(self, name: str) -> str:
        path = self._dir / name
        stat = path.stat()
        cache = self._load()
        hit = cache.get(name)
        if isinstance(hit, dict) and hit.get("size") == stat.st_size and hit.get("mtime_ns") == stat.st_mtime_ns:
            sha = hit.get("sha256")
            if isinstance(sha, str):
                return sha
        sha = sha256_file(path)
        self.record(name, sha)
        return sha

    def record(self, name: str, sha256: str) -> None:
        stat = (self._dir / name).stat()
        cache = self._load()
        cache[name] = {"sha256": sha256, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        atomic_write_text(self._path, pretty(cache))

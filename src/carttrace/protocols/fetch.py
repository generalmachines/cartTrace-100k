"""Fetching one lock source from upstream: a shallow, sparse checkout of one git commit, or plain HTTPS downloads.

Why so narrow. Vendoring is the only network use in this package, and it must never surprise anyone: it touches
GitHub and the official spec sites named in the lock, nothing else, and never a store. A git source is fetched at its
pinned commit only (depth 1, blobs filtered, a sparse checkout of the include globs), so a moved tag cannot change what
lands in vendor/. A url source downloads the listed files, and on relock follows `$ref`s only under the primary
document's own directory. Fetchers return bytes; deciding what to keep and what to hash is `carttrace.protocols.vendor`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urldefrag, urljoin

import httpx

from carttrace.config.http import RetryPolicy, send_with_retry_sync, sync_client
from carttrace.core.errors import ExternalServiceError, ProtocolError, ValidationError
from carttrace.protocols.lock import Source
from carttrace.protocols.walk import iter_refs

GIT_TIMEOUT_S = 300.0
HTTP_TIMEOUT_S = 60.0


@dataclass(frozen=True, slots=True)
class Fetched:
    """What one source produced upstream: file bytes by path relative to the source's dest."""

    files: dict[str, bytes]
    urls: dict[str, str] = field(default_factory=dict)
    """url sources: where each file came from."""
    commit: str | None = None
    """git sources: the commit that was checked out (a tag's commit on relock)."""
    commit_date: str | None = None
    """git sources: the commit's committer date in UTC, as YYYY-MM-DD."""


class Fetcher(Protocol):
    def fetch(self, source: Source, *, discover: bool) -> Fetched:
        """Fetch a source. With `discover`, find the file set (include globs, followed refs) instead of reading only
        the files the lock already lists, and accept a tag's current commit."""
        ...


# ---------------------------------------------------------------------------------------------------------------------
# git


class GitFetcher:
    """Shallow, sparse, blob-filtered checkouts of pinned commits under a scratch directory."""

    def __init__(self, workdir: Path, *, git: str | None = None, timeout_s: float = GIT_TIMEOUT_S) -> None:
        self._git = git
        self._workdir = workdir
        self._timeout_s = timeout_s

    def resolve_tag(self, repo: str, tag: str) -> str:
        """The commit a tag points at upstream, peeling an annotated tag."""
        out = self._run("ls-remote", repo, f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}")
        refs = {name: sha for sha, _, name in (line.partition("\t") for line in out.splitlines() if line)}
        sha = refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")
        if not sha:
            raise ProtocolError(f"tag {tag} does not exist at {repo}", hint="check the tag name in the lock")
        return sha

    def fetch(self, source: Source, *, discover: bool) -> Fetched:
        if source.repo is None:
            raise ValidationError(f"{source.id}: a git source needs a repo")
        commit = self._pinned_commit(source, discover=discover)
        checkout = self._workdir / source.id
        if checkout.exists():
            shutil.rmtree(checkout)
        checkout.mkdir(parents=True)
        patterns = source.include if discover else [f.path for f in source.files]
        self._run("init", "--quiet", cwd=checkout)
        self._run("remote", "add", "origin", source.repo, cwd=checkout)
        self._run(
            "sparse-checkout", "set", "--no-cone", *[_sparse_pattern(p, glob=discover) for p in patterns], cwd=checkout
        )
        self._run("fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", commit, cwd=checkout)
        self._run("checkout", "--quiet", "FETCH_HEAD", cwd=checkout)
        head = self._run("rev-parse", "HEAD", cwd=checkout)
        if head != commit:
            raise ProtocolError(f"{source.id}: fetched {head}, the lock pins {commit}", hint="relock or fix the pin")
        when = datetime.fromisoformat(self._run("log", "-1", "--format=%cI", "HEAD", cwd=checkout))
        files = _collect_globs(source, checkout) if discover else _collect_paths(source, checkout)
        return Fetched(files=files, commit=commit, commit_date=when.astimezone(UTC).date().isoformat())

    def _pinned_commit(self, source: Source, *, discover: bool) -> str:
        commit = source.commit
        if source.tag and source.repo:
            tagged = self.resolve_tag(source.repo, source.tag)
            if discover or commit is None:
                commit = tagged
            elif tagged != commit:
                raise ProtocolError(
                    f"{source.id}: tag {source.tag} now points at {tagged}, the lock pins {commit}",
                    hint=f"if the move is intended, run `carttrace protocols vendor --relock --only {source.id}`",
                )
        if commit is None:
            raise ValidationError(f"{source.id}: no commit pinned", hint="add a tag or commit, then relock")
        return commit

    def _run(self, *args: str, cwd: Path | None = None) -> str:
        git = self._git or shutil.which("git")
        if not git:
            raise ExternalServiceError("git", "git is not installed", hint="install git, then rerun the vendor command")
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
        try:
            done = subprocess.run(  # noqa: S603 - fixed git argv from the lock, never a shell
                [git, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=self._timeout_s, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise ExternalServiceError("git", f"git {args[0]} timed out", hint="check the network, then rerun") from exc
        if done.returncode != 0:
            detail = (done.stderr or done.stdout).strip().splitlines()
            raise ExternalServiceError(
                "git",
                f"git {' '.join(args[:2])} failed: {detail[-1] if detail else done.returncode}",
                hint="check the repo URL and that the pinned commit exists upstream",
            )
        return done.stdout.strip()


def _sparse_pattern(value: str, *, glob: bool) -> str:
    """A non-cone sparse-checkout pattern anchored at the root. Globs keep wildcards; literal paths are escaped."""
    if glob:
        return "/" + value
    escaped = "".join("\\" + ch if ch in "*?[]\\!#" else ch for ch in value)
    return "/" + escaped


def _collect_globs(source: Source, checkout: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for pattern in source.include:
        matched = sorted(p for p in checkout.glob(pattern) if _is_tracked_file(p, checkout))
        if not matched:
            raise ProtocolError(
                f"{source.id}: include {pattern!r} matches nothing at {source.ref}",
                hint="the repository layout differs at this ref; fix include in the lock",
            )
        for p in matched:
            files[p.relative_to(checkout).as_posix()] = p.read_bytes()
    return dict(sorted(files.items()))


def _collect_paths(source: Source, checkout: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for locked in source.files:
        p = checkout / locked.path
        if not _is_tracked_file(p, checkout):
            raise ProtocolError(
                f"{source.id}: {locked.path} does not exist at {source.ref}",
                hint="upstream no longer has this file at the pin; relock or fix the pin",
            )
        files[locked.path] = p.read_bytes()
    return files


def _is_tracked_file(p: Path, checkout: Path) -> bool:
    """A regular file inside the checkout. Symlinks are refused so a hostile repo cannot point us at local files."""
    rel = p.relative_to(checkout)
    return p.is_file() and not p.is_symlink() and ".git" not in rel.parts


# ---------------------------------------------------------------------------------------------------------------------
# url


def http_client(*, transport: httpx.BaseTransport | None = None, user_agent: str | None = None) -> httpx.Client:
    """The client vendoring uses: the repository's one client factory, with our user agent from settings, following
    redirects (spec sites redirect to their CDNs), which `_get` then requires to stay on https."""
    return sync_client(
        user_agent=user_agent,
        timeout=HTTP_TIMEOUT_S,
        transport=transport,
        follow_redirects=True,
        headers={"Accept": "application/json, */*;q=0.5"},
    )


class UrlFetcher:
    """Downloads with the repository's retry policy: GETs are idempotent, so dropped connections and 429/5xx retry."""

    def __init__(self, client: httpx.Client, *, retry: RetryPolicy | None = None) -> None:
        self._client = client
        self._retry = retry or RetryPolicy()

    def fetch(self, source: Source, *, discover: bool) -> Fetched:
        if source.url is None:
            raise ValidationError(f"{source.id}: a url source needs a url")
        base = source.url.rsplit("/", 1)[0] + "/"
        if not discover:
            files: dict[str, bytes] = {}
            urls: dict[str, str] = {}
            for locked in source.files:
                url = locked.url or urljoin(base, locked.path)
                files[locked.path], urls[locked.path] = self._get(url), url
            return Fetched(files=files, urls=urls)
        return self._crawl(source, source.url, base)

    def _crawl(self, source: Source, start: str, base: str) -> Fetched:
        files: dict[str, bytes] = {}
        urls: dict[str, str] = {}
        queue, seen = deque([start]), {start}
        while queue:
            url = queue.popleft()
            data = self._get(url)
            rel = url.removeprefix(base)
            files[rel], urls[rel] = data, url
            if not (source.follow_refs and rel.endswith(".json")):
                continue
            try:
                doc = json.loads(data)
            except ValueError as exc:
                raise ProtocolError(f"{source.id}: {url} is not JSON", hint="drop follow_refs for this source") from exc
            targets = sorted({urldefrag(site.target).url for site in iter_refs(doc, url)})
            for target in targets:
                if target.startswith(base) and target not in seen:
                    seen.add(target)
                    queue.append(target)
        return Fetched(files=dict(sorted(files.items())), urls=dict(sorted(urls.items())))

    def _get(self, url: str) -> bytes:
        if not url.startswith("https://"):
            raise ValidationError(f"refusing to fetch {url}: vendored files come over https only")
        host = httpx.URL(url).host
        try:
            resp = send_with_retry_sync(self._client, self._client.build_request("GET", url), self._retry)
        except httpx.HTTPError as exc:
            raise ExternalServiceError(host, f"GET {url} failed: {exc}", hint="check the network, then relock") from exc
        if resp.status_code != 200:
            raise ExternalServiceError(
                host,
                f"HTTP {resp.status_code} for {url}",
                status=resp.status_code,
                hint="the file moved or is gone upstream; fix the url in the lock",
            )
        if not str(resp.url).startswith("https://"):
            raise ExternalServiceError(host, f"{url} redirected off https to {resp.url}")
        return resp.content

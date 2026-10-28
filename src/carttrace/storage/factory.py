"""Choosing a blob store: by URL scheme for `open_store`, by config.json components for `store(settings)`.

Why both. A tool that is handed a URL (a bundle path from a report, a media prefix) needs the store that URL names,
so `open_store` picks the built-in implementation for its scheme. Run output follows config: `store(settings)` uses
the implementation config.json names under components["storage.blobstore"] and the URL in storage.url (CARTTRACE_STORAGE),
and refuses a pair that disagrees, such as a local store asked to open s3://, instead of guessing which one was
meant. Implementations are resolved through the registry (kind "storage.blobstore"), so boto3 is imported only when
S3 is actually chosen.

`storage.aws_profile` reaches every store opened here: `store` and `media_store` read it from settings, and callers
holding only a URL pass it as `aws_profile`. It goes to implementations whose constructor takes `profile` (S3), so a
local store never sees it.
"""

from __future__ import annotations

import inspect
from typing import cast

from carttrace.config.loader import settings as process_settings
from carttrace.config.schema import Settings
from carttrace.core.errors import ConfigError
from carttrace.core.registry import resolve
from carttrace.storage.service import BlobStore

KIND = "storage.blobstore"
SCHEMES = {"file": "local", "s3": "s3"}
"""The built-in implementation for each URL scheme."""


def scheme_of(url: str) -> str:
    """The URL's scheme; a plain path counts as file."""
    head, sep, _ = url.partition("://")
    return head.lower() if sep else "file"


def open_store(url: str, *, name: str | None = None, aws_profile: str | None = None) -> BlobStore:
    """The store at `url`. `name` picks the implementation; by default it is the built-in one for the URL's scheme.
    `aws_profile` (config `storage.aws_profile`) goes to an implementation that takes `profile`."""
    scheme = scheme_of(url)
    chosen = name or SCHEMES.get(scheme)
    if chosen is None:
        raise ConfigError(f"no blob store for {scheme}:// URLs", hint="use file://<path> or s3://bucket/prefix")
    factory = resolve(KIND, chosen)
    accepted = getattr(factory, "schemes", None)
    if isinstance(accepted, frozenset) and scheme not in accepted:
        raise ConfigError(
            f"the {chosen!r} blob store cannot open {scheme}:// URLs",
            hint=f"set components.{KIND} to {SCHEMES.get(scheme, '<a store for ' + scheme + '>')!r} "
            "(CARTTRACE__COMPONENTS__STORAGE_BLOBSTORE), or point storage.url (CARTTRACE_STORAGE) at a URL it opens",
        )
    if aws_profile and "profile" in inspect.signature(factory).parameters:
        return cast(BlobStore, factory(url, profile=aws_profile))
    return cast(BlobStore, factory(url))


def store(settings: Settings | None = None) -> BlobStore:
    """The store for run output: config.json's storage.url through its chosen implementation."""
    settings = settings or process_settings()
    return open_store(settings.storage.url, name=settings.component(KIND), aws_profile=settings.storage.aws_profile)


def media_store(settings: Settings | None = None) -> BlobStore:
    """The store for rendered media (storage.media_url), opened by its scheme."""
    settings = settings or process_settings()
    if not settings.storage.media_url:
        raise ConfigError("storage.media_url is not set", hint="set it to s3://bucket/prefix in config.json")
    return open_store(settings.storage.media_url, aws_profile=settings.storage.aws_profile)

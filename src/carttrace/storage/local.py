"""BlobStore on a local directory (file:// URLs): the default store, and what tests and laptops run on.

Why atomic writes. A runner killed mid-write must never leave half a file that a reader then trusts, so every put
goes through `carttrace.core.files.atomic_write_bytes`: the bytes land in a dot-named temporary beside the target and are
renamed into place. Listings skip dot-named files and directories, so a reader never sees a temporary, and a crashed
writer's leftover temporary is invisible to every key operation.

URLs: file:///abs/path is the store at that path. config.json may say file://var/store, which the settings loader
resolves against the repository root; given to `open_store` directly, a relative form resolves against the working
directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
from urllib.parse import unquote, urlsplit

from carttrace.core.errors import ConfigError, NotFoundError
from carttrace.core.files import atomic_write_bytes
from carttrace.storage.keys import validate_key, validate_prefix, validate_ttl


def root_of(url: str | Path) -> Path:
    """The absolute directory a file:// URL (or a plain path) names."""
    if isinstance(url, Path):
        path = url
    elif url.startswith("file://"):
        parts = urlsplit(url)
        if parts.query or parts.fragment:
            raise ConfigError(f"file URL {url!r} carries a query or fragment", hint="use file:///abs/path")
        rest = unquote(url.removeprefix("file://"))
        path = Path(rest.removeprefix("localhost")) if rest.startswith("localhost/") else Path(rest)
    elif "://" in url:
        raise ConfigError(f"{url!r} is not a file:// URL", hint="the local store takes file:///abs/path")
    else:
        path = Path(url)
    path = path.expanduser()
    return path if path.is_absolute() else path.absolute()


class LocalBlobStore:
    """A directory of files, one per key."""

    schemes: ClassVar[frozenset[str]] = frozenset({"file"})

    def __init__(self, url: str | Path) -> None:
        self.root = root_of(url)

    def __repr__(self) -> str:
        return f"LocalBlobStore({self.url!r})"

    @property
    def url(self) -> str:
        return self.root.as_uri()

    def path(self, key: str) -> Path:
        """The file that holds `key`, for tools that must hand a real path to another program."""
        return self.root.joinpath(*validate_key(key).split("/"))

    def put_bytes(self, key: str, data: bytes) -> None:
        atomic_write_bytes(self.path(key), data)

    def get_bytes(self, key: str) -> bytes:
        path = self.path(key)
        try:
            return path.read_bytes()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            raise NotFoundError(f"no blob {key!r} in {self.url}", hint="check the key, or list the store") from None

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def list(self, prefix: str = "") -> list[str]:
        prefix = validate_prefix(prefix)
        directory = prefix.rpartition("/")[0]
        base = self.root.joinpath(*directory.split("/")) if directory else self.root
        if not base.is_dir():
            return []
        keys: list[str] = []
        for dirpath, dirnames, filenames in base.walk():
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                if name.startswith(".") or not (dirpath / name).is_file():
                    continue
                key = (dirpath / name).relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    keys.append(key)
        return sorted(keys)

    def delete(self, key: str) -> None:
        path = self.path(key)
        if path.is_file():
            path.unlink(missing_ok=True)

    def uri(self, key: str) -> str:
        return self.path(key).as_uri()

    def presign(self, key: str, ttl_s: int) -> str:
        validate_ttl(ttl_s)
        return self.uri(key)

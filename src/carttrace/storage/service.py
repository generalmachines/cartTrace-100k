"""What a blob store promises. Implementations: `carttrace.storage.local.LocalBlobStore`, `carttrace.storage.s3.S3BlobStore`.

Why one small interface. Trace bundles, status files, decision logs and media all end up either on this machine
(file://) or in S3 (s3://), and the code that writes them must not care which. The interface is the handful of
operations both backends do the same way: whole-object put and get, existence, a sorted listing by key prefix,
delete, a stable URI, and a time-limited link. Keys are relative POSIX paths checked by `carttrace.storage.keys`, so a key
means the same object on every backend and can never escape the store's root.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BlobStore(Protocol):
    @property
    def url(self) -> str:
        """The store's root as a URL: file:///abs/path or s3://bucket/prefix."""
        ...

    def put_bytes(self, key: str, data: bytes) -> None:
        """Write the whole object, replacing any previous one. Readers never see a partial object."""
        ...

    def get_bytes(self, key: str) -> bytes:
        """The whole object. Raises NotFoundError when there is none."""
        ...

    def exists(self, key: str) -> bool: ...

    def list(self, prefix: str = "") -> list[str]:
        """Every key that starts with `prefix` (a plain string prefix, as in S3), sorted by code point."""
        ...

    def delete(self, key: str) -> None:
        """Remove the object. Deleting a key that does not exist is not an error."""
        ...

    def uri(self, key: str) -> str:
        """A stable name for the object: file:///abs/path/key or s3://bucket/prefix/key."""
        ...

    def presign(self, key: str, ttl_s: int) -> str:
        """A link that reads the object for `ttl_s` seconds. Only S3 makes a real time-limited link; the local store
        returns its file:// URI."""
        ...

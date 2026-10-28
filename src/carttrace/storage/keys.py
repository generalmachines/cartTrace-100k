"""Blob keys: relative POSIX paths, checked once here so no backend can be walked outside its root.

Why so strict. A key reaches the local store as a filesystem path and S3 as an object name, and parts of keys come
from data we do not control (store domains and ids from the team sheet). An absolute path, a `..` segment or a
backslash would let a key escape the store's root on disk, or name different objects on Linux, Windows and S3.
Empty segments ("a//b"), `.` segments and trailing slashes name the same file in two ways. Segments that start with
a dot are reserved: atomic writes park their temporary files there, and listings skip them on every backend.
"""

from __future__ import annotations

from carttrace.core.errors import ValidationError

MAX_KEY_BYTES = 1024
"""S3's limit on an object name, applied everywhere so a key that works locally also works in S3."""

_HINT = "use a relative path like traces/ep-1234/episode.json"


def validate_key(key: str) -> str:
    """The key itself when it is safe; raises ValidationError naming the rule it breaks."""
    if not isinstance(key, str) or not key:
        raise ValidationError("a blob key must be a non-empty string", hint=_HINT)
    _check_characters(key)
    if key.startswith("/"):
        raise ValidationError(f"blob key {key!r} is absolute", hint=_HINT)
    for segment in key.split("/"):
        _check_segment(key, segment)
    return key


def validate_prefix(prefix: str) -> str:
    """A listing prefix: empty, or a key, or a key with a trailing slash, or a partial last segment."""
    if prefix == "":
        return prefix
    if not isinstance(prefix, str):
        raise ValidationError("a blob prefix must be a string", hint=_HINT)
    _check_characters(prefix)
    if prefix.startswith("/"):
        raise ValidationError(f"blob prefix {prefix!r} is absolute", hint=_HINT)
    segments = prefix.split("/")
    for segment in segments[:-1]:
        _check_segment(prefix, segment)
    last = segments[-1]
    if last in (".", "..") or last.startswith("."):
        raise ValidationError(f"blob prefix {prefix!r} ends in a reserved segment", hint=_HINT)
    return prefix


def _check_characters(value: str) -> None:
    if "\\" in value:
        raise ValidationError(f"blob key {value!r} contains a backslash", hint="use / between segments")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValidationError(f"blob key {value!r} contains a control character", hint=_HINT)
    if len(value.encode("utf-8")) > MAX_KEY_BYTES:
        raise ValidationError(f"blob key is longer than {MAX_KEY_BYTES} bytes", hint="shorten the key")


def _check_segment(value: str, segment: str) -> None:
    if segment == "":
        raise ValidationError(f"blob key {value!r} has an empty segment", hint="remove the doubled or trailing /")
    if segment in (".", ".."):
        raise ValidationError(f"blob key {value!r} contains {segment!r}", hint=_HINT)
    if segment.startswith("."):
        raise ValidationError(
            f"blob key {value!r} has a segment starting with a dot", hint="dot segments are reserved for temporaries"
        )


MAX_LINK_TTL_S = 604800
"""Seven days: the longest an S3 SigV4 presigned URL can live. Applied to every backend so code tested locally
cannot ask S3 for a link it will refuse."""


def validate_ttl(ttl_s: int) -> int:
    if isinstance(ttl_s, bool) or not isinstance(ttl_s, int) or not 1 <= ttl_s <= MAX_LINK_TTL_S:
        raise ValidationError(
            f"a link lifetime must be a whole number of seconds from 1 to {MAX_LINK_TTL_S}",
            hint="storage.media_link_ttl_s in config.json holds the usual value",
        )
    return ttl_s


def is_valid_key(key: str) -> bool:
    """Whether `validate_key` would accept `key`. Listings use it to skip objects other tools wrote (S3 folder
    markers, hidden files) instead of returning keys that every other operation refuses."""
    try:
        validate_key(key)
    except ValidationError:
        return False
    return True

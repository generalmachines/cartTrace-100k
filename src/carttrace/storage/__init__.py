"""Layer 2: BlobStore (local, s3). Every byte of run output goes through it.

- `store(settings)`: the store for run output (config.json storage.url, or CARTTRACE_STORAGE, through components)
- `open_store(url)`: the store a URL names, by scheme
- `media_store(settings)`: the store for rendered media (storage.media_url)
- `BlobStore`: put_bytes, get_bytes, exists, list, delete, uri, presign; keys are relative POSIX paths
"""

from carttrace.storage.factory import media_store, open_store, scheme_of, store
from carttrace.storage.keys import MAX_LINK_TTL_S, is_valid_key, validate_key, validate_prefix, validate_ttl
from carttrace.storage.service import BlobStore

PLUGINS = {
    "storage.blobstore": {
        "local": "carttrace.storage.local:LocalBlobStore",
        "s3": "carttrace.storage.s3:S3BlobStore",
    },
}

__all__ = [
    "MAX_LINK_TTL_S",
    "PLUGINS",
    "BlobStore",
    "is_valid_key",
    "media_store",
    "open_store",
    "scheme_of",
    "store",
    "validate_key",
    "validate_prefix",
    "validate_ttl",
]

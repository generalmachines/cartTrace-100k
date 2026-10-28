"""Finding `$ref`s in a JSON document without mistaking data for schema.

Why a walker of our own. A spec document mixes schema with data: `enum`, `const`, `default` and `examples` hold
instances, which may legally contain a key named "$ref" that is not a reference. OpenAPI and OpenRPC documents embed
schemas at many depths. Every consumer here (the URL crawler that follows a handler's references, the ref check that
proves each version resolves offline) needs the same answer to "which strings are references, against which base",
so it lives in one place. A string `$id` on the way down moves the base, as JSON Schema says it does.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

DATA_KEYWORDS = frozenset({"enum", "const", "default", "examples", "example"})
"""Keywords whose values are instances, never schemas. Vendor extensions (`x-...`) are skipped for the same reason."""


@dataclass(frozen=True, slots=True)
class RefSite:
    pointer: str
    """JSON pointer of the object that holds the `$ref`, e.g. "/properties/items"."""
    ref: str
    """The `$ref` value as written."""
    base: str
    """The base URI in force where the reference appears."""

    @property
    def target(self) -> str:
        return urljoin(self.base, self.ref)


def iter_refs(node: Any, base: str, pointer: str = "") -> Iterator[RefSite]:
    """Every `$ref` string in `node`, in document order, with the base URI that applies to it."""
    if isinstance(node, dict):
        ident = node.get("$id")
        if isinstance(ident, str) and ident:
            base = urljoin(base, ident)
        ref = node.get("$ref")
        if isinstance(ref, str):
            yield RefSite(pointer or "/", ref, base)
        for key, value in node.items():
            if key in DATA_KEYWORDS or key.startswith("x-") or key in ("$ref", "$id"):
                continue
            yield from iter_refs(value, base, f"{pointer}/{escape(str(key))}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from iter_refs(item, base, f"{pointer}/{i}")


def escape(token: str) -> str:
    """One JSON pointer reference token (RFC 6901)."""
    return token.replace("~", "~0").replace("/", "~1")


def unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def resolve_pointer(doc: Any, pointer: str) -> Any:
    """The value at `pointer` ("" or "/" is the whole document). Raises KeyError or IndexError when it is absent."""
    if pointer in ("", "/"):
        return doc
    if not pointer.startswith("/"):
        raise KeyError(f"not a JSON pointer: {pointer!r}")
    node = doc
    for raw in pointer[1:].split("/"):
        token = unescape(raw)
        node = node[int(token)] if isinstance(node, list) else node[token]
    return node

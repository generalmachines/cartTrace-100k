"""Turning a schema full of `$ref`s into one self-contained schema, for tool definitions an agent reads.

Why inline. A UCP tool's arguments are spread over dozens of files (checkout, line item, item, buyer, postal address,
payment instruments) joined by relative and absolute `$ref`s. An agent sees a tool list as text and cannot follow a
reference to another file, and live Shopify stores list their UCP tools fully inlined. So every reference is replaced
by the schema it points at, walking only the keywords that hold subschemas (data under `enum`, `const`, `default` and
`examples` is copied as is). A target that refers back to itself cannot be inlined; it goes once into `$defs` and is
referenced locally. Each embedded resource loses `$id`, `$schema`, `$anchor` and its own `$defs`, so a local reference
can only mean the bundle's own `$defs`. A `$ref` with sibling keywords keeps them: annotations are laid over the
inlined target, anything else wraps the target in `allOf` so validation means the same as before.

A projection (a function from one schema object to another) runs on every object before it is walked; UCP uses it to
apply `ucp_request` for one operation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urldefrag, urljoin

from referencing import Registry
from referencing.exceptions import Unresolvable

from carttrace.core.errors import ValidationError
from carttrace.protocols.walk import unescape

Projection = Callable[[dict[str, Any]], dict[str, Any]]

ANNOTATIONS = frozenset({"title", "description", "examples", "default", "deprecated", "readOnly", "writeOnly"})
ALWAYS_DROPPED = frozenset(
    {"$id", "$schema", "$anchor", "$dynamicAnchor", "$comment", "$defs", "definitions", "$vocabulary"}
)
_ONE = (
    "additionalProperties",
    "contains",
    "propertyNames",
    "not",
    "if",
    "then",
    "else",
    "unevaluatedItems",
    "unevaluatedProperties",
    "contentSchema",
    "additionalItems",
)
_MANY = ("allOf", "anyOf", "oneOf", "prefixItems")
_MAP = ("properties", "patternProperties", "dependentSchemas")
_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def bundle(
    schema: Any,
    base: str,
    registry: Registry[Any],
    *,
    project: Projection | None = None,
    drop: frozenset[str] = frozenset(),
) -> Any:
    """A self-contained copy of `schema` (found at `base`) with every reference resolved through `registry`."""
    state = _Bundler(registry=registry, project=project, drop=ALWAYS_DROPPED | drop)
    out = state.schema(schema, base)
    if state.defs:
        if not isinstance(out, dict):
            out = {"allOf": [out]}
        out = {**out, "$defs": dict(sorted(state.defs.items()))}
    return json.loads(json.dumps(out))


@dataclass
class _Bundler:
    registry: Registry[Any]
    project: Projection | None
    drop: frozenset[str]
    defs: dict[str, Any] = field(default_factory=dict)
    _names: dict[str, str] = field(default_factory=dict)
    _stack: list[str] = field(default_factory=list)
    _done: dict[str, Any] = field(default_factory=dict)
    _recursive: set[str] = field(default_factory=set)

    def schema(self, node: Any, base: str) -> Any:
        if not isinstance(node, dict):
            return node
        if self.project is not None:
            node = self.project(node)
        ident = node.get("$id")
        if isinstance(ident, str) and ident:
            base = urljoin(base, ident)
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in self.drop or key == "$ref":
                continue
            out[key] = self._keyword(key, value, base)
        ref = node.get("$ref")
        if not isinstance(ref, str):
            return out
        return _merge(self._target(urljoin(base, ref)), out)

    def _keyword(self, key: str, value: Any, base: str) -> Any:
        if key in _MAP and isinstance(value, dict):
            return {name: self.schema(sub, base) for name, sub in value.items()}
        if key in _MANY and isinstance(value, list):
            return [self.schema(sub, base) for sub in value]
        if key in _ONE or key == "items":
            return [self.schema(s, base) for s in value] if isinstance(value, list) else self.schema(value, base)
        if key == "dependencies" and isinstance(value, dict):
            return {n: self.schema(sub, base) if isinstance(sub, dict | bool) else sub for n, sub in value.items()}
        return value

    def _target(self, target: str) -> Any:
        if target in self._stack:
            self._recursive.add(target)
            return {"$ref": f"#/$defs/{self._name(target)}"}
        if target in self._done:
            return self._done[target]
        try:
            resolved = self.registry.resolver().lookup(target)
        except Unresolvable as exc:
            raise ValidationError(
                f"$ref {target} does not resolve inside the vendored tree",
                hint="run `carttrace protocols show <family> <version>`; it lists every unresolved ref",
            ) from exc
        self._stack.append(target)
        out = self.schema(resolved.contents, self._base_of(target))
        self._stack.pop()
        if target in self._recursive:
            self.defs[self._name(target)] = out
            out = {"$ref": f"#/$defs/{self._name(target)}"}
        self._done[target] = out
        return out

    def _base_of(self, target: str) -> str:
        """The base URI in force at a reference target: its document, moved by every `$id` above the target."""
        doc_uri, fragment = urldefrag(target)
        root = self.registry.get_or_retrieve(doc_uri).value.contents
        if not fragment.startswith("/"):
            return doc_uri
        base, node = doc_uri, root
        for raw in fragment[1:].split("/"):
            if isinstance(node, dict) and isinstance(node.get("$id"), str):
                base = urljoin(base, node["$id"])
            node = node[int(raw)] if isinstance(node, list) else node[unescape(raw)]
        return base

    def _name(self, target: str) -> str:
        if target not in self._names:
            doc_uri, fragment = urldefrag(target)
            stem = doc_uri.rstrip("/").rsplit("/", 1)[-1].split(".", 1)[0]
            tail = fragment.rstrip("/").rsplit("/", 1)[-1] if fragment else ""
            name = _NAME_CHARS.sub("_", f"{stem}.{tail}" if tail else stem) or "def"
            taken = set(self._names.values())
            candidate, n = name, 2
            while candidate in taken:
                candidate, n = f"{name}_{n}", n + 1
            self._names[target] = candidate
        return self._names[target]


def _merge(target: Any, siblings: dict[str, Any]) -> Any:
    """A resolved `$ref` combined with the keywords written next to it."""
    if not siblings:
        return target
    if isinstance(target, dict) and set(siblings) <= ANNOTATIONS:
        return {**target, **siblings}
    return {"allOf": [target], **siblings}

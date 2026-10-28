"""Resolving `$ref`s inside one vendored version, and never on the network.

Why scoped registries. A validator built with jsonschema's defaults fetches any `$ref` it cannot find, which would make
validation depend on a website being up and on what it serves that day. Here every vendored document is registered
under a fixed, unresolvable URI (`https://vendor.invalid/<family>/<version>/<path>`) and under its own `$id`, and
anything else raises instead of being fetched. Registration is per tree: UCP 2026-01-11 and 2026-01-23 ship both
`source/` (what authors write) and `spec/` (what their generator emitted), and the two trees declare the same `$id`s
with different content, so one registry per version would silently keep whichever file was read last. A tree is the
first directory of a document's path inside the version, and a reference resolves only among its tree's documents.

Why repairs. Some upstream documents write references relative to their own file while declaring an `$id` somewhere
else: ACP uses placeholder ids (https://example.com/schemas/cart/bundle.schema.json) and refers to sibling files by
name, and UCP's discovery profile up to 2026-04-08 declares /schemas/discovery/ but refers to ../schemas/ucp.json.
By the JSON Schema rules those references point at nothing. When such a reference does resolve against the file's own
path, that file is registered under the URI the reference actually produces, and the repair is recorded, so the
validators work the way the authors' tooling does and the lint names every place it was needed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urldefrag, urljoin, urlsplit

from referencing import Registry, Resource, Specification
from referencing.exceptions import NoSuchResource, Unresolvable
from referencing.jsonschema import DRAFT202012, specification_with

from carttrace.protocols.walk import iter_refs

VENDOR_BASE = "https://vendor.invalid/"
"""Base of every vendored document's URI. `.invalid` is reserved (RFC 2606), so no lookup can ever leave the host."""


def tree_of(relpath: str) -> str:
    """The tree a document belongs to: its first directory inside the version, or "" for files at the top."""
    parts = PurePosixPath(relpath).parts
    return parts[0] if len(parts) > 1 else ""


def document_uri(family: str, version: str, relpath: str) -> str:
    return f"{VENDOR_BASE}{family}/{version}/{relpath}"


def specification_of(doc: Any) -> Specification[Any]:
    """The JSON Schema dialect a document declares, defaulting to 2020-12. Non-objects get the opaque dialect."""
    if not isinstance(doc, Mapping):
        return Specification.OPAQUE
    dialect = doc.get("$schema")
    if isinstance(dialect, str):
        return specification_with(dialect, default=DRAFT202012)
    return DRAFT202012


@dataclass(frozen=True, slots=True)
class TreeRefs:
    """The offline reference registry for one tree of one version, and what building it had to tolerate."""

    family: str
    version: str
    tree: str
    registry: Registry[Any]
    duplicate_ids: tuple[str, ...] = ()
    """`$id`s declared by more than one document in this tree, each with the paths that declare it."""
    repairs: tuple[str, ...] = ()
    """References that resolve only against their file's own path, one sentence each."""

    def base_of(self, relpath: str) -> str:
        return document_uri(self.family, self.version, relpath)


def build_tree_refs(family: str, version: str, tree: str, docs: Mapping[str, Any]) -> TreeRefs:
    """Register every document of one tree. `docs` maps relpath to parsed document, for this tree only."""
    resources: dict[str, Resource[Any]] = {}
    declared: dict[str, list[str]] = {}
    for relpath in sorted(docs):
        doc = docs[relpath]
        resources[document_uri(family, version, relpath)] = specification_of(doc).create_resource(doc)
        ident = doc.get("$id") if isinstance(doc, Mapping) else None
        if isinstance(ident, str) and urlsplit(ident).scheme:
            declared.setdefault(urldefrag(ident).url, []).append(relpath)
    registry: Registry[Any] = Registry(retrieve=_retriever(docs, declared))  # type: ignore[call-arg]  # attrs alias
    registry = registry.with_resources(resources.items()).crawl()
    aliases, repairs = _file_relative_aliases(family, version, docs, registry, resources)
    if aliases:
        registry = registry.with_resources(aliases.items()).crawl()
    dupes = tuple(f"{i} in {', '.join(paths)}" for i, paths in sorted(declared.items()) if len(paths) > 1)
    return TreeRefs(family, version, tree, registry, dupes, tuple(repairs))


def _retriever(docs: Mapping[str, Any], declared: Mapping[str, list[str]]) -> Callable[[str], Resource[Any]]:
    """Map a URI nobody registered onto a vendored file by the `$id` layout, or refuse. Never fetches.

    A document with `$id` https://ucp.dev/schemas/shopping/checkout.json at source/schemas/shopping/checkout.json
    says https://ucp.dev/schemas/ lives at source/schemas/; a reference to a sibling URL whose file declares no `$id`
    of its own is then found by path.
    """
    prefixes = sorted({p for ident, paths in declared.items() for p in _prefix_pairs(ident, paths[0])}, reverse=True)

    def retrieve(uri: str) -> Resource[Any]:
        for url_prefix, path_prefix in prefixes:
            if uri.startswith(url_prefix):
                candidate = path_prefix + uri.removeprefix(url_prefix)
                if candidate in docs:
                    return specification_of(docs[candidate]).create_resource(docs[candidate])
        raise NoSuchResource(ref=uri)  # type: ignore[call-arg]  # attrs field

    return retrieve


def _prefix_pairs(ident: str, relpath: str) -> list[tuple[str, str]]:
    """(URL prefix, path prefix) when the `$id` and the path end in the same segments."""
    url_parts = urlsplit(ident).path.split("/")
    path_parts = relpath.split("/")
    common = 0
    while common < min(len(url_parts), len(path_parts)) and url_parts[-1 - common] == path_parts[-1 - common]:
        common += 1
    if common == 0:
        return []
    url_prefix = ident[: len(ident) - len("/".join(url_parts[-common:]))]
    path_prefix = "/".join(path_parts[:-common])
    return [(url_prefix, f"{path_prefix}/" if path_prefix else "")]


def _file_relative_aliases(
    family: str,
    version: str,
    docs: Mapping[str, Any],
    registry: Registry[Any],
    resources: Mapping[str, Resource[Any]],
) -> tuple[dict[str, Resource[Any]], list[str]]:
    """Aliases that make file-relative references resolve, with one sentence per repaired reference."""
    aliases: dict[str, Resource[Any]] = {}
    repairs: list[str] = []
    for relpath in sorted(docs):
        own = document_uri(family, version, relpath)
        resolver = registry.resolver(base_uri=own)
        for site in iter_refs(docs[relpath], own):
            broken = urldefrag(site.target).url
            if site.ref.startswith("#") or broken in resources:
                continue
            try:
                resolver.lookup(site.target)
                continue
            except Unresolvable:
                pass
            intended = urldefrag(urljoin(own, site.ref)).url
            if intended not in resources or aliases.get(broken, resources[intended]) is not resources[intended]:
                continue
            aliases[broken] = resources[intended]
            repairs.append(f"{relpath}: {site.pointer}: $ref {site.ref!r} resolves only against the file's own path")
    return aliases, repairs


def unresolved_refs(refs: TreeRefs, relpath: str, doc: Any) -> list[str]:
    """Every `$ref` in one document that does not resolve inside its tree, as one sentence each."""
    problems = []
    resolver = refs.registry.resolver(base_uri=refs.base_of(relpath))
    for site in iter_refs(doc, refs.base_of(relpath)):
        try:
            resolver.lookup(site.target)
        except Unresolvable as exc:
            problems.append(f"{relpath}: {site.pointer}: $ref {site.ref!r} does not resolve ({type(exc).__name__})")
    return problems

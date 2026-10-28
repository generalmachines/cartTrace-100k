"""The registry: for a protocol family and version, its files, schemas, validators and tool definitions.

Why one object. The sandbox renders each surface at the versions of a `ProtocolProfile`, the live client validates what
a store returned, the judge checks a cart against the schema the store declared, and the diff compares two releases.
All of them ask the same questions of vendor/ ("which schema validates this message at this version", "which tools
does UCP 2026-04-08 list over MCP"), and all of them must get answers that never touch the network and never change
between runs. Documents are parsed once and cached; callers get copies, so nobody can edit what the next caller reads.
Versions are directories under vendor/<family>/, so a relocked version is visible without code changes.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import urljoin

import jsonschema

from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.protocols.bundle import bundle
from carttrace.protocols.documents import LoadNotes, is_structured, load_document
from carttrace.protocols.paths import configured
from carttrace.protocols.profile import ProtocolProfile
from carttrace.protocols.refs import TreeRefs, build_tree_refs, tree_of, unresolved_refs
from carttrace.protocols.tools import ToolSpec
from carttrace.protocols.ucp import TOOL_HIDDEN_KEYS, binding_path, capabilities, operation_of, projection
from carttrace.protocols.versions import sort_versions
from carttrace.protocols.walk import resolve_pointer

DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_KNOWN_DIALECTS = frozenset(
    {
        DRAFT_2020_12,
        "https://json-schema.org/draft/2019-09/schema",
        "http://json-schema.org/draft-07/schema",
        "http://json-schema.org/draft-06/schema",
        "http://json-schema.org/draft-04/schema",
    }
)


class SchemaValidator(Protocol):
    """The part of a jsonschema validator callers use."""

    def is_valid(self, instance: Any) -> bool: ...

    def validate(self, instance: Any) -> None: ...

    def iter_errors(self, instance: Any) -> Iterator[Any]: ...


@dataclass(slots=True)
class VersionLint:
    """What reading every document of one version found. `clean` means every reference resolves offline."""

    family: str
    version: str
    documents: int = 0
    unresolved: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    """References that resolve only against their file's own path (an upstream `$id` mismatch)."""
    duplicate_ids: list[str] = field(default_factory=list)
    duplicate_keys: list[str] = field(default_factory=list)
    yaml_tab_fixes: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.unresolved and not self.duplicate_ids


class Registry:
    """Read-only access to one vendored tree. Build one per vendor directory; `registry()` returns the shared one."""

    def __init__(self, vendor_dir: Path) -> None:
        self.vendor_dir = vendor_dir
        self._files: dict[tuple[str, str], list[str]] = {}
        self._docs: dict[tuple[str, str, str], Any] = {}
        self._notes: dict[tuple[str, str, str], LoadNotes] = {}
        self._trees: dict[tuple[str, str, str], TreeRefs] = {}
        self._tools: dict[tuple[str, str, bool], list[ToolSpec]] = {}

    # -- what is vendored -------------------------------------------------------------------------------------------

    def families(self) -> list[str]:
        if not self.vendor_dir.is_dir():
            return []
        return sorted(p.name for p in self.vendor_dir.iterdir() if p.is_dir() and not p.name.startswith("."))

    def versions(self, family: str) -> list[str]:
        root = self.vendor_dir / _name(family)
        if not root.is_dir():
            raise NotFoundError(
                f"no protocol family {family!r} is vendored",
                hint=f"choose one of {self.families()}, or add a source to protocols.lock.toml",
            )
        return sort_versions([p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])

    def has(self, family: str, version: str) -> bool:
        return (self.vendor_dir / _name(family) / _name(version)).is_dir()

    def files(self, family: str, version: str) -> list[str]:
        """Every vendored file of one version, as sorted POSIX paths relative to vendor/<family>/<version>/.

        Paths with a component starting with "." are left out: no upstream file vendored here is one, and a relock
        stages its files in a hidden sibling directory until it renames them into place.
        """
        key = (family, version)
        if key not in self._files:
            root = self._version_dir(family, version)
            rels = (p.relative_to(root) for p in root.rglob("*") if p.is_file())
            self._files[key] = sorted(r.as_posix() for r in rels if not any(x.startswith(".") for x in r.parts))
        return list(self._files[key])

    def path(self, family: str, version: str, relpath: str) -> Path:
        rel = PurePosixPath(relpath)
        if rel.is_absolute() or ".." in rel.parts or not relpath:
            raise ValidationError(f"{relpath!r} is not a path inside a vendored version")
        p = self._version_dir(family, version) / rel
        if not p.is_file():
            raise NotFoundError(
                f"{family} {version} has no file {relpath}",
                hint=f"run `carttrace protocols show {family} {version}` to list its files",
            )
        return p

    # -- documents and schemas --------------------------------------------------------------------------------------

    def document(self, family: str, version: str, relpath: str) -> Any:
        """The parsed JSON or YAML document, shared and cached. Do not modify it; `schema` returns a copy."""
        key = (family, version, relpath)
        if key not in self._docs:
            path = self.path(family, version, relpath)
            if not is_structured(path):
                raise ValidationError(f"{family} {version} {relpath} is not a JSON or YAML document")
            notes = LoadNotes()
            self._docs[key] = load_document(path, notes)
            self._notes[key] = notes
        return self._docs[key]

    def schema(self, family: str, version: str, ref: str) -> Any:
        """A copy of a document, or of the part a JSON pointer fragment names.

        For example `schema("mcp", "2025-06-18", "schema/2025-06-18/schema.json#/definitions/Tool")`.
        """
        relpath, _, fragment = ref.partition("#")
        doc = self.document(family, version, relpath)
        try:
            part = resolve_pointer(doc, fragment) if fragment else doc
        except (KeyError, IndexError, ValueError) as exc:
            raise NotFoundError(f"{family} {version} {relpath} has nothing at #{fragment}") from exc
        return copy.deepcopy(part)

    def validator(self, family: str, version: str, ref: str, *, formats: bool = False) -> SchemaValidator:
        """A jsonschema validator for a document or a fragment of one, resolving every `$ref` inside the version.

        `formats` asserts `format` (uri, email, date-time); JSON Schema 2020-12 treats it as an annotation by default.
        """
        relpath, _, fragment = ref.partition("#")
        doc = self.document(family, version, relpath)
        refs = self.refs(family, version, tree_of(relpath))
        dialect = doc.get("$schema") if isinstance(doc, dict) else None
        dialect = dialect if isinstance(dialect, str) and dialect.rstrip("#") in _KNOWN_DIALECTS else DRAFT_2020_12
        target = refs.base_of(relpath) + (f"#{fragment}" if fragment else "")
        root = {"$schema": dialect, "$ref": target}
        cls = jsonschema.validators.validator_for(root, default=jsonschema.Draft202012Validator)
        checker = cls.FORMAT_CHECKER if formats else None
        return cls(root, registry=refs.registry, format_checker=checker)  # type: ignore[no-any-return]

    def errors(self, family: str, version: str, ref: str, instance: Any, *, formats: bool = False) -> list[str]:
        """Every validation error as "<path>: <message>", sorted, or an empty list when the instance is valid."""
        out = []
        for err in self.validator(family, version, ref, formats=formats).iter_errors(instance):
            where = "/" + "/".join(str(p) for p in err.absolute_path)
            out.append(f"{where}: {err.message}")
        return sorted(out)

    def refs(self, family: str, version: str, tree: str) -> TreeRefs:
        """The offline `$ref` registry for one tree of one version (see `carttrace.protocols.refs`)."""
        key = (family, version, tree)
        if key not in self._trees:
            docs = {
                rel: self.document(family, version, rel)
                for rel in self.files(family, version)
                if is_structured(rel) and tree_of(rel) == tree
            }
            self._trees[key] = build_tree_refs(family, version, tree, docs)
        return self._trees[key]

    def lint(self, family: str, version: str) -> VersionLint:
        """Parse every document of a version and resolve every reference in it; report what had to be tolerated."""
        out = VersionLint(family, version)
        structured = [rel for rel in self.files(family, version) if is_structured(rel)]
        for tree in sorted({tree_of(rel) for rel in structured}):
            refs = self.refs(family, version, tree)
            out.repairs += refs.repairs
            out.duplicate_ids += refs.duplicate_ids
            for rel in (r for r in structured if tree_of(r) == tree):
                doc = self.document(family, version, rel)
                notes = self._notes[(family, version, rel)]
                out.duplicate_keys += [f"{rel}: {k}" for k in notes.duplicate_keys]
                out.yaml_tab_fixes += [f"{rel}: line {n}" for n in notes.tab_fixes]
                out.unresolved += unresolved_refs(refs, rel, doc)
                out.documents += 1
        return out

    # -- per-protocol views ----------------------------------------------------------------------------------------

    def ucp_mcp_tools(self, version: str, *, service: str = "shopping", outputs: bool = False) -> list[ToolSpec]:
        """The tools a UCP MCP endpoint lists at this version, from the release's OpenRPC binding.

        One tool per method, in binding order: its name, its description (or summary), and an input schema whose
        properties are the method's params, every `$ref` inlined and `ucp_request` applied for the method's operation.
        With `outputs`, each tool also carries its result schema, projected with `ucp_response`.
        """
        key = (version, service, outputs)
        if key not in self._tools:
            self._tools[key] = self._build_ucp_tools(version, service, outputs=outputs)
        return [t.model_copy(deep=True) for t in self._tools[key]]

    def _build_ucp_tools(self, version: str, service: str, *, outputs: bool) -> list[ToolSpec]:
        relpath = self._ucp_binding(version, service, "mcp")
        doc = self.document("ucp", version, relpath)
        refs = self.refs("ucp", version, tree_of(relpath))
        base = refs.base_of(relpath)
        if isinstance(doc.get("$id"), str):
            base = urljoin(base, doc["$id"])
        tools = []
        for method in doc.get("methods", []):
            name, op = method["name"], operation_of(method["name"])
            props: dict[str, Any] = {}
            required = []
            for param in method.get("params", []):
                schema = param.get("schema", {})
                if param.get("description") and isinstance(schema, dict) and "description" not in schema:
                    schema = {**schema, "description": param["description"]}
                props[param["name"]] = schema
                if param.get("required"):
                    required.append(param["name"])
            raw: dict[str, Any] = {"type": "object", "properties": props}
            if required:
                raw["required"] = required
            schema_in = bundle(raw, base, refs.registry, project=projection("request", op), drop=TOOL_HIDDEN_KEYS)
            schema_out = None
            result = method.get("result", {}).get("schema")
            if outputs and result is not None:
                schema_out = bundle(
                    result, base, refs.registry, project=projection("response", op), drop=TOOL_HIDDEN_KEYS
                )
            tools.append(
                ToolSpec(
                    name=name,
                    description=method.get("description") or method.get("summary") or "",
                    input_schema={"$schema": DRAFT_2020_12, **schema_in},
                    output_schema=schema_out,
                )
            )
        return tools

    def ucp_mcp_openrpc(self, version: str, *, service: str = "shopping") -> dict[str, Any]:
        """The release's MCP binding (OpenRPC) as published, references untouched."""
        return dict(self.schema("ucp", version, self._ucp_binding(version, service, "mcp")))

    def ucp_rest_openapi(self, version: str, *, service: str = "shopping") -> dict[str, Any]:
        """The release's REST binding (OpenAPI) as published, references untouched."""
        return dict(self.schema("ucp", version, self._ucp_binding(version, service, "rest")))

    def ucp_capabilities(self, version: str) -> dict[str, str]:
        """Capability name (dev.ucp.shopping.cart, ...) to the schema that defines it in this release."""
        docs = {rel: self.document("ucp", version, rel) for rel in self.files("ucp", version) if is_structured(rel)}
        return capabilities(docs)

    def _ucp_binding(self, version: str, service: str, transport: str) -> str:
        relpath = binding_path(self.files("ucp", version), service, "mcp" if transport == "mcp" else "rest")
        if relpath is None:
            raise NotFoundError(
                f"UCP {version} has no {transport} binding for service {service!r}",
                hint="services live under source/services/<service>/ in each release",
            )
        return relpath

    def acp_apis(self, version: str) -> list[str]:
        """The OpenAPI documents of one ACP version: agentic_checkout, delegate_payment, ..."""
        prefix = f"spec/{version}/openapi/openapi."
        return sorted(
            rel.removeprefix(prefix).removesuffix(".yaml")
            for rel in self.files("acp", version)
            if rel.startswith(prefix) and rel.endswith(".yaml")
        )

    def acp_openapi(self, version: str, api: str = "agentic_checkout") -> dict[str, Any]:
        """One ACP OpenAPI document, parsed from YAML; the default is the agentic checkout API."""
        if api not in self.acp_apis(version):
            raise NotFoundError(f"ACP {version} has no {api} API", hint=f"choose one of {self.acp_apis(version)}")
        return dict(self.schema("acp", version, f"spec/{version}/openapi/openapi.{api}.yaml"))

    def mcp_schema(self, version: str) -> dict[str, Any]:
        """The MCP JSON Schema for one protocol version (its definitions cover every request, result and type)."""
        return dict(self.schema("mcp", version, f"schema/{version}/schema.json"))

    # -- profiles --------------------------------------------------------------------------------------------------

    def check_profile(self, profile: ProtocolProfile) -> list[str]:
        """What this tree cannot serve for a profile: versions not vendored, capabilities a release does not define."""
        problems = []
        wanted: list[tuple[str, str]] = []
        for family in ("ucp", "mcp", "acp", "ap2", "mpp", "webmcp"):
            value = getattr(profile, family)
            if value is not None:
                wanted.append((family, value))
        wanted += [("ucp", v) for v in profile.ucp_capabilities.values()]
        wanted += list(profile.ucp_payment_handlers.items())
        for family, version in sorted(set(wanted)):
            if not self.has(family, version):
                problems.append(f"{family} {version} is not vendored")
        for name, version in sorted(profile.ucp_capabilities.items()):
            if self.has("ucp", version) and name not in self.ucp_capabilities(version):
                problems.append(f"UCP {version} does not define capability {name}")
        return problems

    def _version_dir(self, family: str, version: str) -> Path:
        if not self.has(family, version):
            known = self.versions(family)
            raise NotFoundError(f"{family} {version} is not vendored", hint=f"vendored versions: {known}")
        return self.vendor_dir / family / version


def _name(value: str) -> str:
    """A family or version as one plain directory name, so no lookup can leave vendor/."""
    if not value or value.startswith(".") or "/" in value or "\\" in value:
        raise ValidationError(f"{value!r} is not a protocol family or version name")
    return value


_SHARED: dict[Path, Registry] = {}


def registry(vendor_dir: Path | None = None) -> Registry:
    """The shared registry for a vendor directory, by default the one settings name (`paths.vendor`)."""
    path = (vendor_dir or configured().vendor).resolve()
    if path not in _SHARED:
        _SHARED[path] = Registry(path)
    return _SHARED[path]

"""UCP specifics: where each release keeps its bindings, and what a request may contain for one operation.

Why a module of its own. UCP moved its files between releases (2026-01-11 and 2026-01-23 name the MCP binding
`services/shopping/openrpc.json`; 2026-04-08 renamed it `mcp.openrpc.json`), and its source schemas describe the
response: a checkout requires `id`, `status` and `totals`, which no client sends on create. Each property says how it
behaves in a request with a `ucp_request` annotation (`omit`, `required`, `optional`, or a map from operation to one of
those). The rules below are the ones the UCP authors' own SDK generator applies (python-sdk preprocess_schemas.py,
`eval_prop_inclusion`): a map that does not name the operation omits the property, and a `transition` entry uses its
target state. UCP 2026-01-11 shipped the generated request schemas in `spec/`, and the tests check this projection
against them. `ucp_response` uses the same grammar for responses.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from carttrace.protocols.bundle import Projection

Direction = Literal["request", "response"]
ANNOTATION_KEYS = frozenset({"ucp_request", "ucp_response", "ucp_shared_request"})
"""UCP's direction annotations. The projection consumes them."""
TOOL_HIDDEN_KEYS = ANNOTATION_KEYS | {"name", "version"}
"""Keywords left out of agent-facing schemas: the annotations, and the entity metadata (`name`, `version`) UCP puts on
capability schemas, which is not JSON Schema and reads like a field."""

_OPERATIONS = {
    "get": "read",
    "create": "create",
    "update": "update",
    "complete": "complete",
    "cancel": "cancel",
    "search": "search",
    "lookup": "lookup",
}


def binding_path(files: Iterable[str], service: str, transport: Literal["mcp", "rest"]) -> str | None:
    """The OpenRPC (MCP) or OpenAPI (REST) binding of one service in one release, by either naming scheme."""
    names = ("mcp.openrpc.json", "openrpc.json") if transport == "mcp" else ("rest.openapi.json", "openapi.json")
    present = set(files)
    for name in names:
        candidate = f"source/services/{service}/{name}"
        if candidate in present:
            return candidate
    return None


def operation_of(method: str) -> str:
    """The operation a binding method performs: create_checkout is create, get_cart is read."""
    verb = method.split("_", 1)[0]
    return _OPERATIONS.get(verb, verb)


def capabilities(docs: dict[str, Any]) -> dict[str, str]:
    """Capability name to the schema that defines it, from each source schema's top-level `name`."""
    found: dict[str, str] = {}
    for relpath in sorted(docs):
        doc = docs[relpath]
        if not relpath.startswith("source/schemas/") or not isinstance(doc, dict):
            continue
        name = doc.get("name")
        if isinstance(name, str) and name.startswith("dev.ucp.") and name not in found:
            found[name] = relpath
    return dict(sorted(found.items()))


def projection(direction: Direction, op: str) -> Projection:
    """Keep, require or drop each property of an object schema for one operation in one direction."""
    key = "ucp_request" if direction == "request" else "ucp_response"

    def project(node: dict[str, Any]) -> dict[str, Any]:
        props = node.get("properties")
        if not isinstance(props, dict):
            return node
        listed = node.get("required")
        base_required = [r for r in listed if isinstance(r, str)] if isinstance(listed, list) else []
        kept: dict[str, Any] = {}
        required = [r for r in base_required if r not in props]
        for name, sub in props.items():
            include, is_required = _inclusion(sub, key, op, name in base_required)
            if include:
                kept[name] = sub
                if is_required:
                    required.append(name)
        out = {k: v for k, v in node.items() if k != "required"}
        out["properties"] = kept
        if required:
            out["required"] = _ordered(required, base_required)
        return out

    return project


def _inclusion(prop: Any, key: str, op: str, base_required: bool) -> tuple[bool, bool]:
    """(include, required) for one property, by the UCP SDK's rules."""
    if not isinstance(prop, dict) or key not in prop:
        return True, base_required
    marker = prop[key]
    if isinstance(marker, dict):
        marker = marker.get(op)
        if isinstance(marker, dict):
            transition = marker.get("transition")
            marker = transition.get("to") if isinstance(transition, dict) else None
        if marker is None:
            return False, False
    if marker == "omit":
        return False, False
    if marker == "required":
        return True, True
    if marker == "optional":
        return True, False
    return True, base_required


def _ordered(required: list[str], base: list[str]) -> list[str]:
    """Required names in the base schema's order first, then newly required ones in property order, no repeats."""
    seen: set[str] = set()
    out = []
    for name in [*[b for b in base if b in required], *required]:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out

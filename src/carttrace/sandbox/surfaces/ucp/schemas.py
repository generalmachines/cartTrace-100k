"""The tools one UCP release lists, and the check every call's arguments pass before anything reaches the core.

Why the release's own schemas. The UCP bindings require a conforming server to "validate tool inputs against UCP
schemas" (checkout, cart and catalog MCP bindings, "Conformance"), and the point of the sandbox is that the same
request can be valid at one release and refused at another: `update_cart` at 2026-04-08 requires `cart.id` in the
payload, while 2026-08-25 omits it. So the tool list is exactly `carttrace.protocols.registry.ucp_mcp_tools(version)` (names
and request-projected input schemas from the release's OpenRPC binding), and each call is checked against its tool's
input schema with a compiled Draft 2020-12 validator.

What a refusal says. A problem names the field by an RFC 9535 JSONPath and says what the schema wants, built from the
schema and never from the value (`carttrace.sandbox.surfaces.problems`), so a card number or a secret pasted into a field
cannot come back in the reply or the log. Paths are relative to the domain payload (`$.line_items[0].quantity` for
`cart.line_items[0].quantity`); `meta` and `id` keep their names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

import jsonschema

from carttrace.protocols.registry import registry
from carttrace.protocols.tools import ToolSpec
from carttrace.sandbox.surfaces.problems import MAX_PROBLEMS, Problem, json_path, problems

__all__ = ["MAX_PROBLEMS", "TRANSPORT_PARAMS", "Problem", "ToolBook", "json_path", "tool_book"]

TRANSPORT_PARAMS = ("meta", "id")
"""Tool parameters that are not the domain payload: request metadata and the target resource id."""


class ToolBook:
    """The tools of one UCP release, each with its compiled argument validator."""

    def __init__(self, version: str, specs: Sequence[ToolSpec]) -> None:
        self.version = version
        self.specs: tuple[ToolSpec, ...] = tuple(specs)
        self._by_name = {t.name: t for t in self.specs}
        self._validators: dict[str, Any] = {}

    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def spec(self, name: str) -> ToolSpec | None:
        return self._by_name.get(name)

    def payload_key(self, name: str) -> str | None:
        """The domain payload parameter of a tool (`cart`, `checkout`, `catalog`), or None when it takes only an id."""
        spec = self._by_name.get(name)
        props = spec.input_schema.get("properties", {}) if spec is not None else {}
        rest = [p for p in props if p not in TRANSPORT_PARAMS]
        return rest[0] if len(rest) == 1 else None

    def check(self, name: str, arguments: Mapping[str, Any]) -> list[Problem]:
        """Every problem with `arguments` for tool `name`, deterministic and at most `MAX_PROBLEMS`."""
        return problems(self._validator(name), dict(arguments), self.payload_key(name))

    def _validator(self, name: str) -> Any:
        if name not in self._validators:
            schema = self._by_name[name].input_schema
            cls = jsonschema.validators.validator_for(schema, default=jsonschema.Draft202012Validator)
            self._validators[name] = cls(schema)
        return self._validators[name]


FEED_TOOLS = ("search_catalog", "lookup_catalog", "get_product")
"""The catalog tools that serve a checkout-only release's product feed (D71)."""


def release_tools(version: str, *, outputs: bool = False) -> list[ToolSpec]:
    """The tools a release lists in the sandbox: its binding's own, and at a checkout-only release (2026-01-11,
    2026-01-23) the product feed's catalog tools from the first release that defines them, after its checkout ones."""
    from carttrace.sandbox.surfaces.ucp.dialect import DIALECTS

    own = registry().ucp_mcp_tools(version, outputs=outputs)
    dialect = DIALECTS.get(version)
    feed = dialect.feed_version if dialect is not None else None
    if feed is None:
        return own
    names = {t.name for t in own}
    extra = [t for t in registry().ucp_mcp_tools(feed, outputs=outputs) if t.name in FEED_TOOLS and t.name not in names]
    return [*own, *extra]


@cache
def tool_book(version: str) -> ToolBook:
    """The shared tool book of a release, built once per process from the vendored binding."""
    return ToolBook(version, release_tools(version))

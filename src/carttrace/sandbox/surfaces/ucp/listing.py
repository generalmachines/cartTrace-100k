"""The tools a served store lists over MCP: its own recorded `tools/list`, verbatim, when one is known (decision D40).

Why verbatim. The agent's first prompt is its system prompt, the shopper's opening and the tool definitions, and the
definitions are what a store lists. In all 682 live and replica pairs of 2026-10-12 and 13 the replica's first prompt
was about 18,600 tokens longer than the live one, because the replica listed the vendored binding's input schemas
(110 KB over 13 tools) where live Shopify lists its own compact ones (48 KB). A pair whose halves start from different
prompts cannot sample alike whatever the seed (D39). So a replica lists what its store listed: names, descriptions,
input and output schemas, annotations and order, every member as recorded.

Where the list comes from, first found:
- the pack's `tool_lists` (envpack `RecordedToolList`), at exactly the served MCP and UCP versions: a replica built
  with its store's agent surface (`carttrace.sandbox.catalogs.agent_surface`);
- the platform's recorded list (`Platform.tools_file`) at the versions it was recorded at: the live-shaped default
  for a store of that platform with no recording of its own (every Shopify store of 2026-10-12 and 13 sent it, byte
  for byte, on all 1,475 live lists);
- none: the release's binding, as before (D29 for a platform's descriptions and output schemas).

Validation. A call is checked against the input schema the store listed, as the store checks it, so a request live
would refuse is refused here too. The binding's own arguments that the store does not list (UCP's `signals`, the
sandbox's `meta.idempotency-key` on every tool) are still accepted, each checked by the binding's definition, but
never listed, so they never enter the prompt.

What redaction took. Traces redact at write time, and two redactions hit Shopify's list, in all 1,475 recordings:
- a JSON member named `signature` (Shopify's Apple Pay `payment_data` schema has one) becomes `"redacted"` where a
  schema belongs. A redacted value in a `properties` position is restored as `{"type": "string"}`, the schema of each
  of its siblings (`data`, `version`) and the schema an unredacted copy of the list holds there;
- the `?key=` part of a cart gid, which create_checkout's `cart_id` description gives as an example. It is restored
  as the example's own text (`RESTORED_TEXT`), as the unredacted copy has it.
The unredacted copy is the same Shopify list captured by the lab before redaction existed (agentic-shopping-lab
tests/fixtures/ucp-tools-list.json, an older and shorter edition). Restored paths are kept with the recording.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

from carttrace.core.canon import digest, dumps, loads
from carttrace.protocols.registry import registry
from carttrace.protocols.tools import ToolSpec
from carttrace.sandbox.envpack.pack import EnvPack
from carttrace.sandbox.platforms import Platform, platform_tool_list
from carttrace.sandbox.surfaces.ucp.schemas import ToolBook
from carttrace.trace.redact import REDACTED

RESTORED_SCHEMA: Mapping[str, str] = {"type": "string"}
"""What a redacted schema position becomes (see the module docstring)."""
RESTORED_TEXT: Mapping[str, str] = {
    f"gid://shopify/Cart/abc123?key={REDACTED}": "gid://shopify/Cart/abc123?key=secret",
}
"""Example text the cart-key redaction rewrote inside a description, and the store's own words for it."""


@dataclass(frozen=True, slots=True)
class Listing:
    """What one served store lists, and how its calls are checked."""

    tools: tuple[dict[str, Any], ...]
    """The `tools/list` entries, as recorded, in the recorded order."""
    source: str
    """Where the list was read (`RecordedToolList.source`)."""

    def names(self) -> tuple[str, ...]:
        return tuple(str(t["name"]) for t in self.tools)

    def specs(self) -> list[ToolSpec]:
        """The listed tools as ToolSpecs (what REST and the in-process clients read)."""
        return [tool_spec(t) for t in self.tools]

    def validation_specs(self, binding: Mapping[str, ToolSpec]) -> list[ToolSpec]:
        """The listed tools with the input schemas their calls are checked against: the store's, accepting the
        binding's arguments the store does not list."""
        out = []
        for spec in self.specs():
            theirs = binding.get(spec.name)
            schema = accepting(spec.input_schema, theirs.input_schema) if theirs is not None else spec.input_schema
            out.append(spec.model_copy(update={"input_schema": schema}))
        return out


def listing(tools: Sequence[Mapping[str, Any]], source: str) -> Listing:
    """A listing from recorded `tools/list` entries, redacted schema positions restored; entries without a name are
    not tools and are left out."""
    named = [t for t in tools if isinstance(t.get("name"), str) and t["name"]]
    restored, _ = restore_redacted(named)
    return Listing(tuple(restored), source)


def served_listing(pack: EnvPack, platform: Platform, *, mcp: str, ucp: str) -> Listing | None:
    """What a store lists at these versions: the pack's own recording, else its platform's recorded list when it was
    recorded at these versions, else None (the binding's tools)."""
    recorded = pack.tool_list(mcp=mcp, ucp=ucp)
    if recorded is not None:
        return listing(recorded.tools, recorded.source)
    found = platform_tool_list(platform.tools_file) if platform.tools_file is not None else None
    if found is not None and found.mcp == mcp and found.ucp == ucp:
        return listing(found.tools, found.source)
    return None


def listed_book(version: str, served: Listing) -> ToolBook:
    """The tool book a listing's calls are checked with, shared by every app serving the same list."""
    return _listed_book(version, dumps(list(served.tools)), digest(list(served.tools)))


@cache
def _listed_book(version: str, tools: str, key: str) -> ToolBook:
    del key  # in the cache key only, so equal lists share one book
    binding = {t.name: t for t in registry().ucp_mcp_tools(version)}
    served = Listing(tuple(loads(tools)), "")
    return ToolBook(version, served.validation_specs(binding))


def tool_spec(entry: Mapping[str, Any]) -> ToolSpec:
    """One `tools/list` entry as a ToolSpec, its input schema whole (`meta` included)."""
    schema = entry.get("inputSchema")
    output = entry.get("outputSchema")
    annotations = entry.get("annotations")
    return ToolSpec(
        name=str(entry["name"]),
        description=str(entry.get("description") or ""),
        input_schema=copy.deepcopy(dict(schema)) if isinstance(schema, Mapping) else {"type": "object"},
        output_schema=copy.deepcopy(dict(output)) if isinstance(output, Mapping) else None,
        annotations=copy.deepcopy(dict(annotations)) if isinstance(annotations, Mapping) else {},
    )


def accepting(listed: Mapping[str, Any], binding: Mapping[str, Any]) -> dict[str, Any]:
    """The listed schema, plus every property the binding defines at the same place and the listed one lacks, with
    the binding's definition. `required` stays the listed schema's, so nothing the store treats as optional becomes
    required, and nothing listed changes."""
    out: dict[str, Any] = copy.deepcopy(dict(listed))
    ours = binding.get("properties")
    if not isinstance(ours, Mapping):
        return out
    theirs = out.get("properties")
    if not isinstance(theirs, dict):
        return out
    for name, definition in ours.items():
        if not isinstance(definition, Mapping):
            continue
        if name not in theirs:
            theirs[name] = {} if _refers(definition) else copy.deepcopy(dict(definition))
        elif isinstance(theirs[name], Mapping):
            theirs[name] = accepting(theirs[name], definition)
    return out


def _refers(schema: Any) -> bool:
    """Whether a schema holds a `$ref`, which would not resolve once copied under another root."""
    if isinstance(schema, Mapping):
        return "$ref" in schema or any(_refers(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_refers(v) for v in schema)
    return False


def restore_redacted(tools: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Recorded entries with every redacted `properties` member restored as a string schema and every redacted
    example text restored as written, and the JSON pointers of what was restored (`/<tool>/inputSchema/...`)."""
    restored: list[str] = []
    out = []
    for entry in tools:
        name = str(entry.get("name") or "")
        out.append(_restore(copy.deepcopy(dict(entry)), f"/{name}", restored))
    return out, restored


def _restore(value: Any, path: str, restored: list[str]) -> Any:
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{path}/{key}"
            if key == "properties" and isinstance(child, dict):
                for prop, schema in child.items():
                    if schema == REDACTED:
                        child[prop] = dict(RESTORED_SCHEMA)
                        restored.append(f"{here}/{prop}")
                    else:
                        child[prop] = _restore(schema, f"{here}/{prop}", restored)
            else:
                value[key] = _restore(child, here, restored)
        return value
    if isinstance(value, list):
        return [_restore(item, f"{path}/{i}", restored) for i, item in enumerate(value)]
    if isinstance(value, str) and REDACTED in value:
        text = value
        for redacted, own in RESTORED_TEXT.items():
            text = text.replace(redacted, own)
        if text != value:
            restored.append(path)
        return text
    return value

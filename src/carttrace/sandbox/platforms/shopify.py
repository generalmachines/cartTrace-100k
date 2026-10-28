"""The Shopify pack's facts, read from its spec (data/sandbox/packs/shopify.yaml).

The tools/list Shopify stores answered, which the private sandbox serves word for word under this pack, is held
back from this kit: without it the pack's tools carry the protocol's own descriptions. Descriptions are what the
agent reads, never what the store does with a call, so a replay reaches the same state either way.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources
from types import MappingProxyType
from typing import Any

from carttrace.sandbox.platforms.spec import pack_spec

SPEC = pack_spec("shopify")
"""The pack's spec (data/sandbox/packs/shopify.yaml): every fact below is read from it (D101)."""
SHOPIFY_TOOLS_FILE = str(SPEC.serves.mcp.tools_file)
SHOPIFY_MCP_CAPABILITIES: Mapping[str, Any] = MappingProxyType(dict(SPEC.serves.mcp.capabilities))
"""What all 140 live initialize replies advertised (report section 1.1)."""
SHOPIFY_LINKS = tuple((str(t), str(title)) for t, title in SPEC.checkout.policy_links)
"""The seven link types seen on 315 live cart replies, in the order the stores listed them (report section 4)."""

SOLD_OUT_TEXT = SPEC.wording["stock.sold_out"]
"""Live Shopify's words for a cart line it will not sell, `{name}` the product's title with the variant's after ` - `
when the product has more than one (D52)."""
UNOBSERVED = "unobserved"
"""An `EVIDENCE` value starting with this word marks a behaviour no live recording shows: the sandbox keeps its
plain answer there and says so, rather than inventing the store's."""
SHOPIFY_EVIDENCE: Mapping[str, str] = MappingProxyType(SPEC.evidence())
"""Each id and argument refusal live Shopify sent, with the run, episode and seq that showed it (D29, D46), and the
neighbouring cases no recording shows, marked unobserved; and the product shape its catalog tools answer with
(`SHOPIFY_PRODUCT_VIEWS`, D55)."""


def _frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _frozen(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_frozen(v) for v in value)
    return value


SHOPIFY_PRODUCT_VIEWS: Mapping[str, Mapping[str, Any]] = _frozen(SPEC.documents.product_views)
"""The product shape live Shopify's catalog tools answer with, per view (search listing, get_product, lookup), from
`SHOPIFY_EVIDENCE` "product_view.*": the keys a product and a variant may carry, which media fields and how many
media items, which option value fields (on get_product, which ones without a selection), how many listed variants,
which categories (the taxonomy one, never the store's collections). A replica's product built from a capture's
feeds (capture v2) holds more than this (the vendor, the product type, the review summary, barcodes, every image, the
collections): that stays store truth in the capture, and the agent view shows what the store's agent view shows
(D55). A product whose own view the store recorded keeps that shape (carttrace.sandbox.surfaces.ucp.catalog `shaped`)."""
TAXONOMY = str(SHOPIFY_PRODUCT_VIEWS["listing"]["category_values"])
"""The only categories live Shopify's agent view lists: the product's standard taxonomy category."""
MARKET_TAG = str(SPEC.extra["market_tags"]["prefix"])
"""A product tag such as `NO_Australia` marks a market the product is not sold in (report section 4, store-181)."""
COUNTRY_NAMES: Mapping[str, str] = MappingProxyType(dict(SPEC.extra["market_tags"]["country_names"]))


@dataclass(frozen=True, slots=True)
class ToolDoc:
    """A platform's own words and output shape for one tool."""

    description: str
    output_schema: Mapping[str, Any] | None
    input_schema: Mapping[str, Any] | None
    """The platform's input schema, read for the inputs it adds; never used for validation."""


@cache
def tool_docs(file: str) -> Mapping[str, ToolDoc]:
    """The tool docs a platform's data file holds, by tool name."""
    found = resources.files("carttrace.sandbox.platforms").joinpath(file)
    raw = json.loads(found.read_text(encoding="utf-8")) if found.is_file() else {}
    tools = raw.get("tools") if isinstance(raw, Mapping) else None
    out: dict[str, ToolDoc] = {}
    for name, doc in (tools or {}).items():
        if not isinstance(doc, Mapping):
            continue
        output = doc.get("outputSchema")
        inputs = doc.get("inputSchema")
        out[str(name)] = ToolDoc(
            description=str(doc.get("description") or ""),
            output_schema=MappingProxyType(dict(output)) if isinstance(output, Mapping) else None,
            input_schema=MappingProxyType(dict(inputs)) if isinstance(inputs, Mapping) else None,
        )
    return MappingProxyType(out)


@dataclass(frozen=True, slots=True)
class PlatformToolList:
    """A platform's recorded tools/list: the entries in the list's own order, and the versions it was sent at."""

    mcp: str
    ucp: str
    tools: tuple[Mapping[str, Any], ...]
    source: str


@cache
def platform_tool_list(file: str) -> PlatformToolList | None:
    """The platform data file as the tools/list it was recorded from, or None for a file that names no versions or
    no order. Each entry is the file's tool with its `name`, in the recorded order."""
    found = resources.files("carttrace.sandbox.platforms").joinpath(file)
    raw = json.loads(found.read_text(encoding="utf-8")) if found.is_file() else {}
    tools = raw.get("tools") if isinstance(raw, Mapping) else None
    order = raw.get("order") if isinstance(raw, Mapping) else None
    if not isinstance(tools, Mapping) or not isinstance(order, list) or not raw.get("mcp") or not raw.get("ucp"):
        return None
    entries = tuple(
        MappingProxyType({"name": name, **tools[name]}) for name in order if isinstance(tools.get(name), Mapping)
    )
    return PlatformToolList(mcp=str(raw["mcp"]), ucp=str(raw["ucp"]), tools=entries, source=f"platform:{file}")


def market_excludes(tags: Sequence[object], country: str | None) -> bool:
    """Whether a product's market tags exclude the buyer: a `NO_<country>` tag naming the buyer's country, or any
    such tag when the buyer's country is not known (no context was sent, so the store cannot tell)."""
    limits = [str(t)[len(MARKET_TAG) :] for t in tags if isinstance(t, str) and t.upper().startswith(MARKET_TAG)]
    if not limits:
        return False
    if country is None:
        return True
    code = country.upper()
    name = COUNTRY_NAMES.get(code, "")
    return any(_fold(limit) in (code.lower(), name) for limit in limits)


def _fold(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())

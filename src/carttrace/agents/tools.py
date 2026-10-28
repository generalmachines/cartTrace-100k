"""The surfaces' tools as one list the model can call, and their results as text it can read.

Why this layer. An episode may hold several surfaces at once (a store's UCP MCP endpoint beside its own store MCP for
policies), and two of them may list a tool with the same name. The model sees one flat list of OpenAI function
definitions: each tool under its own name, prefixed with its surface kind only when that name is taken twice (or
by the harness's own `done`), so the common case reads exactly as the store wrote it. Arguments the surface fills
itself (UCP's `meta`, which carries our agent profile) are taken out of the schema the model sees, as the lab did,
because a model asked to fill them invents a profile URL.

Results go back to the model as canonical JSON, cut at a size the context can hold with a note saying so (the lab
found 95% of catalog replies cut silently at 12,000 characters, 2026-09-18); the trace keeps them whole.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from carttrace.core.canon import digest, dumps, loads
from carttrace.core.errors import ExternalServiceError, ValidationError
from carttrace.protocols.tools import ToolResult, ToolSpec, ToolSurface
from carttrace.protocols.ucp_meta import PLACEHOLDER_AGENT_PROFILE
from carttrace.trace.carts import from_ucp
from carttrace.trace.schema import Cart, EndReason

DONE = "done"
"""The harness's own tool: the agent calls it to end the conversation."""
REFUSED = "harness_blocked"
"""The result code of a call the harness refused below the agent (the live allowlist)."""
FATAL_CODES: Mapping[str, EndReason] = {"throttled": "throttled", "blocked": "blocked"}
"""Result codes no retry fixes: the surface already spent its retry budget, or the store walled us off."""
HIDDEN_ARGUMENTS = frozenset({"meta"})
"""Arguments the harness fills, so the model never sees them."""
MAX_DESCRIPTION = 2000
RESULT_CHARS = 50_000
"""What one tool result may take of the model's context before it is cut, with a note."""
_NAME_OK = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_NAME_BAD = re.compile(r"[^A-Za-z0-9_-]")

DONE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": DONE,
        "description": (
            "End the conversation. Call it once the cart is ready and you have read it back (cart_ready), when "
            "nothing in the store meets what the shopper needs (no_match), or when you cannot go on "
            "(cannot_continue). The message is your last message to the shopper."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["cart_ready", "no_match", "cannot_continue"]},
                "message": {
                    "type": "string",
                    "description": "What is in the cart and why it fits, or what is missing and why.",
                },
                "cart_id": {"type": "string", "description": "The id of the cart, exactly as the store returned it."},
            },
            "required": ["status", "message"],
        },
    },
}


DONE_CHECKOUT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": DONE,
        "description": (
            "End the conversation. Call it once the store has placed the order and you have read it back "
            "(order_placed, with the order id); when the shopper did not approve the order, or the payment failed "
            "(cannot_continue, saying why); when nothing in the store meets what the shopper needs (no_match); or "
            "when you cannot go on (cannot_continue). The message is your last message to the shopper."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["order_placed", "cart_ready", "no_match", "cannot_continue"]},
                "message": {
                    "type": "string",
                    "description": "What was ordered, its total and when it ships; or what stopped the order and why.",
                },
                "order_id": {"type": "string", "description": "The id of the order, exactly as the store returned it."},
                "cart_id": {"type": "string", "description": "The id of the cart, exactly as the store returned it."},
            },
            "required": ["status", "message"],
        },
    },
}
"""`done` on a checkout row (D13): the journey goes on to an order, so ending at the cart is no longer the goal."""


DONE_SELECTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": DONE,
        "description": (
            "End the conversation. On a store's website the journey ends at a selection: call it once you have chosen "
            "the variant with choose_variant (selection_ready, with the selection id), when nothing in the store meets "
            "what the shopper needs (no_match), or when you cannot go on (cannot_continue). The message is your last "
            "message to the shopper."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["selection_ready", "no_match", "cannot_continue"]},
                "message": {
                    "type": "string",
                    "description": "What you chose and why it fits, or what is missing and why.",
                },
                "selection_id": {
                    "type": "string",
                    "description": "The id of the selection, exactly as choose_variant returned it.",
                },
            },
            "required": ["status", "message"],
        },
    },
}
"""`done` on a store's website (D31): the journey ends at a selection, which the harness reads back."""


def done_tool(*, checkout: bool = False, selection: bool = False) -> dict[str, Any]:
    """The harness's `done` tool: the cart row's, the checkout row's with order_placed, or the website's with
    selection_ready."""
    if selection and not checkout:
        return copy.deepcopy(DONE_SELECTION_TOOL)
    return copy.deepcopy(DONE_CHECKOUT_TOOL if checkout else DONE_TOOL)


@dataclass(frozen=True, slots=True)
class BoundTool:
    exposed: str
    """The name the model sees and calls."""
    spec: ToolSpec
    surface: ToolSurface

    @property
    def name(self) -> str:
        """The surface's own name for the tool."""
        return self.spec.name


@dataclass(frozen=True, slots=True)
class ToolIndex:
    tools: tuple[BoundTool, ...]
    hidden: frozenset[str] = HIDDEN_ARGUMENTS

    def get(self, exposed: str) -> BoundTool | None:
        return next((t for t in self.tools if t.exposed == exposed), None)

    def definitions(self, *, done: bool = True) -> list[dict[str, Any]]:
        """OpenAI tool definitions for every tool, the harness's `done` last."""
        out = [_definition(t, self.hidden) for t in self.tools]
        return [*out, copy.deepcopy(DONE_TOOL)] if done else out


async def index_tools(
    surfaces: Sequence[ToolSurface],
    *,
    hidden: frozenset[str] = HIDDEN_ARGUMENTS,
    reserved: frozenset[str] = frozenset({DONE}),
) -> ToolIndex:
    """Every tool of every surface, in surface order then list order, each under a name unique in the episode."""
    labels = surface_labels(surfaces)
    listed = []
    for surface, label in zip(surfaces, labels, strict=True):
        listed += [(surface, label, spec) for spec in await surface.list_tools()]
    counts = Counter(spec.name for _, _, spec in listed)
    taken: set[str] = set(reserved)
    tools = []
    for surface, label, spec in listed:
        clash = counts[spec.name] > 1 or spec.name in reserved
        exposed = _safe(f"{label}__{spec.name}" if clash else spec.name)
        if exposed in taken:
            exposed = _safe(f"{exposed[:52]}_{digest([label, spec.name], 8)}")
        taken.add(exposed)
        tools.append(BoundTool(exposed, spec, surface))
    return ToolIndex(tuple(tools), hidden)


def surface_labels(surfaces: Sequence[ToolSurface]) -> list[str]:
    """Each surface's kind, numbered (ucp_mcp1, ucp_mcp2) when an episode holds two surfaces of one kind."""
    kinds = Counter(str(s.kind) for s in surfaces)
    running: Counter[str] = Counter()
    labels = []
    for surface in surfaces:
        kind = str(surface.kind)
        running[kind] += 1
        labels.append(kind if kinds[kind] == 1 else f"{kind}{running[kind]}")
    return labels


def _safe(name: str) -> str:
    """A function name OpenAI-compatible servers accept: letters, digits, _ and -, at most 64 characters."""
    if _NAME_OK.match(name):
        return name
    cleaned = _NAME_BAD.sub("_", name)
    return cleaned if len(cleaned) <= 64 else f"{cleaned[:55]}_{digest(name, 8)}"


def _definition(tool: BoundTool, hidden: frozenset[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.exposed,
            "description": tool.spec.description[:MAX_DESCRIPTION],
            "parameters": model_schema(tool.spec.input_schema, hidden),
        },
    }


def model_schema(schema: Mapping[str, Any], hidden: frozenset[str] = HIDDEN_ARGUMENTS) -> dict[str, Any]:
    """The input schema as the model sees it: a copy without `$schema` and without the hidden arguments."""
    out: dict[str, Any] = copy.deepcopy(dict(schema))
    out.pop("$schema", None)
    out.setdefault("type", "object")
    properties = out.get("properties")
    if isinstance(properties, dict):
        for name in hidden:
            properties.pop(name, None)
    else:
        out["properties"] = {}
    required = out.get("required")
    if isinstance(required, list):
        kept = [r for r in required if r not in hidden]
        if kept:
            out["required"] = kept
        else:
            out.pop("required")
    return out


def result_text(result: ToolResult, *, limit: int = RESULT_CHARS) -> str:
    """What the model reads back from a call: the content as text (canonical JSON unless it is text already),
    marked when it is an error, and cut at `limit` with a note saying how much was left out."""
    content = result.content
    text = content if isinstance(content, str) else _json(content)
    if result.is_error:
        text = f"error ({result.error_code or 'unknown'}): {text}"
    if len(text) <= limit:
        return text
    note = f"[result cut here: {limit} of {len(text)} characters shown. Narrow the search or ask for the next page.]"
    return f"{text[:limit]}\n{note}"


def _json(value: Any) -> str:
    try:
        return dumps(value)
    except (TypeError, ValueError):
        return str(value)


def payload_of(result: ToolResult) -> Any:
    """The result as JSON data: structured content when the surface gave it, else the content, parsed when it is
    JSON text."""
    data = result.structured if result.structured is not None else result.content
    if isinstance(data, str) and data.lstrip().startswith(("{", "[")):
        try:
            return loads(data)
        except ValueError:
            return data
    return data


def is_cart_tool(name: str) -> bool:
    """Whether a tool's result is a cart (create_cart, update_cart, get_cart, cancel_cart). A checkout also carries
    line items and an id, so only cart tools may change which cart the episode is building."""
    return "cart" in name.lower() and "checkout" not in name.lower()


def cart_in(result: ToolResult) -> Cart | None:
    """The cart a successful cart call returned (create_cart, update_cart, get_cart), or None when the result holds
    no readable cart."""
    if result.is_error:
        return None
    data = payload_of(result)
    if not isinstance(data, Mapping):
        return None
    try:
        return from_ucp(data)
    except ValidationError:
        return None


def summary(cart: Cart | None) -> dict[str, Any] | None:
    """The cart as supervision reads it: {"lines": [...], "total": minor units, "currency": ...}."""
    if cart is None:
        return None
    lines = [{"variant_id": li.variant_id, "quantity": li.quantity, "unit_price": li.unit_price} for li in cart.lines]
    return {"lines": lines, "total": cart.total, "currency": cart.currency}


def step_record(tool: str, arguments: Mapping[str, Any], result: ToolResult) -> dict[str, Any]:
    """One tool call as supervision reads it (data/decisions/supervise.yaml)."""
    text = result_text(result, limit=200)
    return {
        "tool": tool,
        "arguments": dict(arguments),
        "is_error": result.is_error,
        "error_code": result.error_code,
        "result": text,
    }


def failure_reason(exc: BaseException) -> EndReason:
    """How an episode ends when a surface raises instead of returning a result: a 429 is throttled, a 401 or 403 is
    blocked, and anything else (the live safety gate included) is a harness error, never the store's or the
    agent's fault."""
    if isinstance(exc, ExternalServiceError) and exc.status == 429:
        return "throttled"
    if isinstance(exc, ExternalServiceError) and exc.status in (401, 403):
        return "blocked"
    return "harness_error"


def transport_arguments(profile_url: str | None) -> dict[str, Any]:
    """The hidden arguments the harness fills: UCP's `meta`, naming our agent profile."""
    return {"meta": {"ucp-agent": {"profile": profile_url or PLACEHOLDER_AGENT_PROFILE}}}


def with_transport(spec: ToolSpec, arguments: Mapping[str, Any], transport: Mapping[str, Any]) -> dict[str, Any]:
    """The agent's arguments plus each transport argument the tool's listed schema declares, merged into one the call
    already carries (an idempotency key stays; the agent profile is ours). A surface that fills `meta` itself (the
    live UCP client) lists no `meta`, so nothing is added there; the sandbox lists the release's schemas as
    published, where `meta` is required."""
    declared = spec.input_schema.get("properties")
    out = dict(arguments)
    for name, value in transport.items():
        if not isinstance(declared, Mapping) or name not in declared:
            continue
        given = out.get(name)
        if isinstance(given, Mapping) and isinstance(value, Mapping):
            out[name] = {**given, **copy.deepcopy(dict(value))}  # our identity wins over one the model invented
        else:
            out[name] = copy.deepcopy(value)
    return out

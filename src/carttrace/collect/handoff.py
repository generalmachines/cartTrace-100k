"""Hand a verified live cart to a sandbox checkout (decision D3: payment runs in the sandbox, never on a live store).

Why here. A live journey ends at the cart. When its manifest row asks for a hand-off, the runner builds an envpack from
the cart exactly as the store priced it, with the profile and payment handlers the store declared in discovery, saves
it, and opens a checkout for the same lines on the sandbox's UCP surface. The bundle's outcome then names the pack,
the profile and the checkout, which is where the separate agent-payment track starts.

What never travels: the live cart id's secret key (the pack keeps only `safe_cart_ref`), and anything from the live
session beyond the lines. The pack carries the live bundle's rights in its provenance, because its snapshot is store
data under the same terms.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from carttrace.collect.ports import Discovery, OpenSandboxSurface, WireSink
from carttrace.collect.worlds import SandboxWorlds, sandbox_market
from carttrace.core.canon import loads
from carttrace.core.clock import Clock, iso
from carttrace.core.errors import ValidationError
from carttrace.protocols.profile import ProtocolProfile
from carttrace.protocols.tools import ToolResult, ToolSurface
from carttrace.sandbox import build_envpack_from_cart
from carttrace.trace.schema import Cart, Rights, StoreRef

ID_WRAPPERS = ("structuredContent", "result", "checkout", "cart")


async def hand_off(
    cart: Cart,
    *,
    store: StoreRef,
    profile: ProtocolProfile,
    discovery: Discovery | None,
    rights: Rights,
    worlds: SandboxWorlds,
    open_surface: OpenSandboxSurface,
    recorder: WireSink,
    session_id: str,
    profile_url: str,
    clock: Clock,
    opened: list[ToolSurface],
) -> dict[str, Any]:
    """The outcome's `handoff`: {"envpack", "profile", "checkout_id"}, plus "error_code" when the sandbox refused the
    checkout. The opened surface is appended to `opened`, so the caller closes it whatever happens."""
    code = sandbox_market(store.market or "US")
    if code is None:
        raise ValidationError(f"the sandbox has no market {store.market!r} to hand this cart off to")
    pack = build_envpack_from_cart(
        cart,
        store=store,
        profile=profile,
        payment_handlers=declared_handlers(discovery),
        market=code,
        source="live:get_cart",
        captured_at=iso(clock.now()),
        provenance={"handoff_of": session_id, "rights": rights.model_dump(mode="json")},
    )
    worlds.save(pack)
    surface = await open_surface(
        pack, kind="ucp_mcp", recorder=recorder, session_id=session_id, agent_profile=profile_url
    )
    opened.append(surface)
    lines = [{"item": {"id": line.variant_id}, "quantity": line.quantity} for line in cart.lines]
    result = await surface.call("create_checkout", {"checkout": {"line_items": lines}})
    handoff: dict[str, Any] = {"envpack": pack.id, "profile": profile.id, "checkout_id": None}
    if result.is_error:
        handoff["error_code"] = result.error_code or "error"
    else:
        handoff["checkout_id"] = id_in(payload_of(result), "checkout")
    return handoff


def declared_handlers(discovery: Discovery | None) -> dict[str, Any]:
    """The payment handlers a store declared in its UCP discovery document, as it declared them; {} when unknown."""
    if discovery is None:
        return {}
    doc = discovery.ucp_profile
    if isinstance(doc, BaseModel):
        doc = doc.model_dump(mode="json")
    if not isinstance(doc, Mapping):
        return {}
    ucp = doc.get("ucp", doc)
    handlers = ucp.get("payment_handlers") if isinstance(ucp, Mapping) else None
    return {str(k): v for k, v in handlers.items()} if isinstance(handlers, Mapping) else {}


def payload_of(result: ToolResult) -> Any:
    """The JSON a tool result carries: its structured content, else the first JSON text block, else its content."""
    if result.structured is not None:
        return result.structured
    content = result.content
    if isinstance(content, str):
        return _json_or_none(content)
    if not isinstance(content, list):
        return content
    for block in content:
        if not isinstance(block, Mapping):
            continue
        if block.get("type") != "text":
            return block
        parsed = _json_or_none(block.get("text"))
        if parsed is not None:
            return parsed
    return None


def _json_or_none(text: Any) -> Any:
    if not isinstance(text, str):
        return None
    try:
        return loads(text)
    except ValueError:
        return None


def id_in(payload: Any, kind: str) -> str | None:
    """The `id` of a checkout (or cart) in a reply, under any of the usual wrappers."""
    current = payload
    for _ in range(len(ID_WRAPPERS) + 1):
        if not isinstance(current, Mapping):
            return None
        nested = current.get(kind)
        if isinstance(nested, Mapping) and isinstance(nested.get("id"), str):
            return str(nested["id"])
        if isinstance(current.get("id"), str) and kind not in current:
            return str(current["id"])
        current = next((current[k] for k in ID_WRAPPERS if k in current), None)
    return None

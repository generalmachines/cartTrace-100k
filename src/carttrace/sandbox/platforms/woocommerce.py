"""WooCommerce's wire facts and wording, each from the reference store's recorded probes (decision D42).

Why a second platform. Most stores the pilots reached are Shopify, so every replica spoke Shopify's dialect (D29,
D40) and the traces taught one platform's habits. The project asked for variety (2026-10-13): the same products under
another platform's rules, so an agent is measured on what changes when the store is not Shopify. WooCommerce is the
platform we ran: WooCommerce 11.2.1 with the same 20 products as a sandbox pack (`ops/playgrounds/woocommerce/`),
225 probes to its Store API and the same operations to the sandbox, plus its MCP and two UCP/ACP plugins
(captures/diff-woo, captures/plugins-mcp, captures/plugins-ucp-acp-agent, docs/SANDBOX-GAPS.md).

What is here. The facts a surface reads: the MCP handshake WooCommerce's MCP adapter answers, the UCP plugin's
discovery extras and ACP plugin's discovery document, and WooCommerce's own words for the refusals an agent meets
(coupons, stock). The core's side (codes stack, individual use, free shipping as its own rate, the first rate
selected, a refused line keeps the cart, offline orders on hold) is `carttrace.sandbox.core.config.CORE_RULES`. Every
behaviour names its probe in `EVIDENCE`; the recorded surfaces and the quirks not ported, with why, are the data
file `woocommerce_surfaces_2026-10-12.json`, read whole so a newer recording is one file. Every fact here is read from
the pack's spec, data/sandbox/packs/woocommerce.yaml (D101).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from importlib import resources
from types import MappingProxyType
from typing import Any

from carttrace.core.money import to_major
from carttrace.sandbox.platforms.spec import pack_spec

SPEC = pack_spec("woocommerce")
"""The pack's spec (data/sandbox/packs/woocommerce.yaml): every constant below is read from it."""
SURFACES_FILE = str(SPEC.extra["surfaces_file"])
CAPTURES = "ops/playgrounds/woocommerce/captures"
WOO_MCP_VERSION = str(SPEC.serves.mcp.pinned)
"""The MCP adapter's protocol version (plugins-mcp 02)."""
WOO_ACP_VERSION = str(SPEC.serves.acp.pinned)
"""The UCP/ACP Agent plugin's ACP release (plugins-ucp-acp-agent 02)."""

EVIDENCE: Mapping[str, str] = MappingProxyType(SPEC.evidence())
"""Each ported behaviour and the recorded probe that set it (captures under CAPTURES)."""

WOO_MCP_CAPABILITIES: Mapping[str, Any] = MappingProxyType(dict(SPEC.serves.mcp.capabilities))
"""What WooCommerce's MCP initialize advertised (plugins-mcp 02)."""
WOO_SERVER_INFO: Mapping[str, str] = MappingProxyType(dict(SPEC.serves.mcp.server_info or {}))
WOO_INSTRUCTIONS = str(SPEC.serves.mcp.instructions)
WOO_ORDER_STATES: Mapping[str, str] = MappingProxyType(dict(SPEC.checkout.order_states))
"""The core's order statuses in WooCommerce's slugs (diff-woo 104, 112, 117, 120)."""

WORDING: Mapping[str, str] = MappingProxyType(dict(SPEC.wording))
"""WooCommerce's own words (diff-woo 41, 42, 63, 66, 67, 68), with the HTML entities the Store API sends decoded, as
its UCP plugin decodes titles (diff-woo 141). Coupon codes are lower case in them, as WooCommerce stores codes."""

READINESS: Mapping[str, tuple[str, str, str]] = MappingProxyType(
    {k: (r.code, r.text, r.path) for k, r in SPEC.checkout.readiness.items()}
)
"""An incomplete checkout's messages as the Finance District UCP plugin words them (diff-woo 142): its own codes,
recoverable, each at the field to send."""


def money_text(amount: int, currency: str) -> str:
    """An amount as WooCommerce writes it in a message: `$100.00` in US dollars, `100.00 EUR` otherwise."""
    major = to_major(amount, currency) or "0"
    return f"${major}" if currency.upper() == "USD" else f"{major} {currency.upper()}"


@cache
def recorded_surfaces() -> Mapping[str, Any]:
    """The recorded surfaces and quirks file, whole."""
    text = resources.files("carttrace.sandbox.platforms").joinpath(SURFACES_FILE).read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{SURFACES_FILE} is not a JSON object")
    return MappingProxyType(dict(raw))


def acp_discovery(version: str, api_base_url: str) -> dict[str, Any]:
    """The ACP discovery document as the UCP/ACP Agent plugin serves it, at the served release and base URL."""
    template = recorded_surfaces()["surfaces"]["acp"]["discovery"]
    filled = json.dumps(template).replace("{version}", version).replace("{api_base_url}", api_base_url)
    out: dict[str, Any] = json.loads(filled)
    return out


def ucp_supported_versions(base_url: str) -> dict[str, str]:
    """`supported_versions` as the Finance District plugin's discovery lists them: each older release at its URL."""
    ucp = recorded_surfaces()["surfaces"]["ucp"]
    return {v: f"{base_url}/.well-known/ucp/{v}" for v in ucp["supported_versions"]}


def quirks() -> list[Mapping[str, Any]]:
    """Every recorded quirk, ported or not, with its probe."""
    return list(recorded_surfaces()["quirks"])

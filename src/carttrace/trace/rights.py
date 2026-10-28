"""The data rights of anything taken from a live store (docs/DECISIONS.md D10): decided up front, never allowed by
default.

Why one function, here. The live client, the collector and the benchmark each need the rights of a live store, and
three copies had already drifted on what an unrecognised platform key means. The answer depends only on the store's
platform, and `Rights` lives in this package, so the one definition does too.

Shopify's API License and Terms of Use (updated 2026-10-07) cover its MCP servers, so UCP traffic with a Shopify store
is `shopify_api`: section 2.3.24 bars training on it without Shopify's or the merchant's written consent, and store
scores stay unpublished until the publication gate clears. Other platforms get their own regime and the same two false
flags. Nothing here can set either flag to true: consent is granted by people, recorded in `basis`, through a reviewed
path of its own.
"""

from __future__ import annotations

from typing import Literal

from carttrace.trace.schema import Rights

PlatformTerms = Literal["shopify_api", "woocommerce", "custom", "sandbox", "unknown"]
_TERMS: dict[str, PlatformTerms] = {"shopify": "shopify_api", "woocommerce": "woocommerce"}
_UNKNOWN = frozenset({"", "unclear", "other", "unknown"})
"""Platform keys that say nothing about whose terms apply (the queue's "Unclear (bot wall)" among them)."""

SHOPIFY_BASIS = (
    "D10: Shopify API License and Terms of Use (updated 2026-10-07) 2.3.24 bars training on data derived from its "
    "APIs or merchant stores without consent; no consent recorded"
)
LIVE_BASIS = "D10: live store, no merchant consent recorded; inference and internal benchmarking only"


def platform_terms(platform: str | None) -> PlatformTerms:
    """The terms regime of a store's platform key (carttrace.stores PlatformKey, or a Discovery's platform): Shopify and
    WooCommerce by name, unknown when the key says nothing, and the store's own terms ("custom") otherwise."""
    key = (platform or "").strip().lower()
    return _TERMS.get(key) or ("unknown" if key in _UNKNOWN else "custom")


def live_rights(platform: str | None, *, store_source: str = "team_sheet") -> Rights:
    """Rights for an episode or scan on a live store on `platform`: its terms regime, no merchant consent, and
    training and publication not allowed."""
    terms = platform_terms(platform)
    return Rights(
        store_source=store_source,
        platform_terms=terms,
        merchant_consent="none",
        training_allowed=False,
        publication_allowed=False,
        basis=SHOPIFY_BASIS if terms == "shopify_api" else LIVE_BASIS,
    )

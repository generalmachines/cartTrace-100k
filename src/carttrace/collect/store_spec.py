"""One stand-in store, one spec: the platform pack, protocol versions, rails and payment broker an open twin is served
under, the same on every row of the store.

Why. The open lane drew a platform pack, a UCP release and a broker per row (D72's `draw_platforms`, D71's draw axes),
so one stand-in store was served under three to five packs: CartTrace's public stores table had 16 rows for 4 stores,
and a store described as Magento on one row was Shopify on the next. A store is one thing: here each twin gets one pack
and one profile that declares every rail it pays on, so its worlds share one protocol set, and the exposure sheet and
experiment.json describe that store. Coverage of packs and versions comes from more stores, never from more packs per
store.

How a twin's spec is chosen (`open_specs`): deterministically from its place in its category's roster as planned (the
k-th open store planned in a category), each axis on its own rotation and each category starting at its own place, so
a category's twins are spread evenly over the packs (40 twins over 9 packs: 4 or 5 each), UCP releases, AP2 versions
and brokers, a round of one twin a category spreads over the packs too, and the same ledger gives the same specs. A
pack that pins a protocol version (bigcommerce's UCP 2026-04-08, woocommerce's MCP and ACP) keeps its pin. Only UCP
releases that define a cart and a catalog are chosen: a checkout-only release (2026-01-11, 2026-01-23) cannot serve a
cart row (D98), so those stay on the replica lane's crossing (D71).

Payment handlers follow the pack only for Shop Pay (Shopify's alone, D89). Every pack still declares the sandbox's card
handler under its vendored namespace (dev.shopify.card, the one card tokenization handler a UCP spec in vendor/
defines); a neutral card namespace is the project's call, since the agent's wallet and the WebMCP storefront name it too.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from carttrace.collect.cells import shop_pay_platform
from carttrace.collect.rails import RAIL_NAMES, at_version, rail, rail_profile
from carttrace.core.canon import digest
from carttrace.protocols.profile import ProtocolProfile
from carttrace.sandbox.core.brokers import BROKER_NAMES
from carttrace.sandbox.platforms import PLATFORM_PACKS, PLATFORMS, platform_profile

STORE_UCP: Final = ("2026-04-08", "2026-08-25")
"""UCP releases a stand-in store may speak: the ones that define cart and catalog beside checkout."""


AP2_VERSIONS: Final = ("v0.1.0", "v0.2.0")


class StoreSpec(BaseModel):
    """What one store is served as, on every one of its rows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    platform: str
    """The one platform pack the store is served under."""
    ucp: str | None = None
    """The UCP release the store speaks; None keeps the plan's preset."""
    rail_versions: dict[str, list[str]] = Field(default_factory=dict)
    """Rail to the one version of its protocol the store speaks (D71's axis, one value a store)."""
    psp: str | None = None
    """The payment broker behind the store's PSP handler (D71), one a store; None for the sandbox's own."""

    @field_validator("platform")
    @classmethod
    def _platform(cls, value: str) -> str:
        if value not in PLATFORM_PACKS:
            raise ValueError(f"not a platform pack: {value!r}; platform packs are {list(PLATFORM_PACKS)}")
        return value

    @field_validator("psp")
    @classmethod
    def _psp(cls, value: str | None) -> str | None:
        if value is not None and value not in BROKER_NAMES:
            raise ValueError(f"not a payment broker: {value!r}; brokers are {list(BROKER_NAMES)}")
        return value

    @field_validator("rail_versions")
    @classmethod
    def _one_version(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        for name, versions in value.items():
            if name not in RAIL_NAMES:
                raise ValueError(f"rail_versions: not a payment rail: {name!r}")
            if len(versions) != 1 or versions[0] not in rail(name).versions:
                raise ValueError(f"rail_versions: a store speaks one version of {name}, one of {rail(name).versions}")
        return dict(sorted(value.items()))

    def profile(self, base: ProtocolProfile, rails: Sequence[str]) -> ProtocolProfile:
        """The store's one profile: `base` at its UCP release, declaring every payment protocol of `rails` it pays on
        (at its version), Shop Pay only under the shopify pack, and the pack's own pins. Every rail's world is then
        the same profile, so the store has one protocol set."""
        profile = at_version(base, "ucp_psp", self.ucp) if self.ucp else base
        for name in rails:
            chosen = rail(name)
            if name == "ucp_psp" or (chosen.handler and not shop_pay_platform(self.platform, None)):
                continue
            versions = self.rail_versions.get(name)
            profile = rail_profile(profile, name, versions[0] if versions else None)
        return platform_profile(profile, self.platform)


def _rotation(values: Sequence[str], axis: str) -> list[str]:
    """`values` in a stable order of their own for one axis."""
    return sorted(values, key=lambda v: digest(["v1-store-spec", axis, v], 12))


def pinned_ucp(pack: str) -> str | None:
    """The UCP release every store of a platform speaks, where its pack pins one."""
    found = PLATFORMS[pack].profile_versions.get("ucp")
    return str(found) if found else None


def spec_at(k: int, start: int, *, packs: Sequence[str], psps: Sequence[str], rails: Sequence[str]) -> StoreSpec:
    """The spec of the k-th open store planned in a category whose deal starts at `start` (its place among the
    categories, so one store a category in a round spreads over the packs too): each axis in its own rotation, the
    pack by k, the UCP release by k, the AP2 version by k // 2 and the broker by k // 4, so a category's roster takes
    every pack evenly and every combination comes round."""
    at = start + k
    pack = _rotation(packs, "pack")[at % len(packs)]
    ucp = pinned_ucp(pack) or _rotation(STORE_UCP, "ucp")[at % len(STORE_UCP)]
    versions: dict[str, list[str]] = {}
    if "ucp_psp" in rails:
        versions["ucp_psp"] = [ucp]
    if "ap2_mandates" in rails:
        versions["ap2_mandates"] = [_rotation(AP2_VERSIONS, "ap2")[(start + k // 2) % len(AP2_VERSIONS)]]
    psp = _rotation(psps, "psp")[(start + k // 4) % len(psps)] if psps and "ucp_psp" in rails else None
    return StoreSpec(platform=pack, ucp=ucp, rail_versions=versions, psp=psp)


def open_specs(
    stores: Sequence[str],
    categories: Mapping[str, str],
    planned_before: Mapping[str, int],
    *,
    packs: Sequence[str],
    psps: Sequence[str],
    rails: Sequence[str],
) -> dict[str, StoreSpec]:
    """Each chosen twin's spec, by its place in its category's roster: `planned_before` holds how many open stores of
    each category earlier rounds planned, and this round's stores follow in the order given. A category's deal starts
    at its place among the categories `categories` names."""
    starts = {c: n for n, c in enumerate(sorted(set(categories.values())))}
    seen: dict[str, int] = {}
    out: dict[str, StoreSpec] = {}
    for sid in stores:
        category = categories.get(sid, "")
        k = planned_before.get(category, 0) + seen.get(category, 0)
        seen[category] = seen.get(category, 0) + 1
        out[sid] = spec_at(k, starts.get(category, 0), packs=packs, psps=psps, rails=rails)
    return out

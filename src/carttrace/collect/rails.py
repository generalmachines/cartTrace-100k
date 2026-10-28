"""Payment rails: the axis a sandbox order row is paid over, beside the platform axis (D66).

Why an axis and not a surface. A replica serves only the protocols its real store speaks, and almost every store we
captured speaks UCP alone: a trial manifest crossing store-158 with ucp_mcp, acp_rest, mpp and ap2 kept only its ucp_mcp
rows. The project asked for journeys that end with a purchase over an agentic payment protocol, so a rail serves the
store's own catalog through the rail's protocol as if the store had adopted it, and every row says whether it did
(`adopted`). Rows with adopted true are scored for the agent on that rail and kept out of the store's own ledger.

The rails, and the row surfaces each runs on:
- `ucp_psp`: UCP checkout paid with a handler token from the store's PSP, on the plan's UCP surfaces (ucp_mcp,
  ucp_rest, the storefront's webmcp);
- `acp_delegated`: an ACP checkout session paid with a delegated, vaulted token (Idempotency-Key, API-Version), on
  acp_rest with the store's UCP catalog beside it;
- `ap2_mandates`: AP2's intent, cart and payment mandates, signed by the shopper's wallet after the shopper consents, on
  ap2 with the store's UCP catalog beside it;
- `mpp`: a UCP checkout paid over MPP's HTTP 402 (challenge, credential, receipt), on the plan's UCP bindings with the
  store's MPP endpoint beside it, and UCP's own complete_checkout not offered, so the payment goes over the rail;
- `shop_pay`: a UCP checkout paid with a Shop Pay single-use token (D73), on the same UCP surfaces as ucp_psp. Shop
  Pay is a payment handler, not a protocol family: adoption adds `dev.shopify.shop_pay` 2026-04-08 to the profile's
  payment handlers, with the world's own shop_id (`adopted_shop_id`), and the wallet pays through it first.

A rail whose protocol the store's profile does not name is served at the version below (`adopt`), the newest the
sandbox serves and an agent dialect speaks. Every rail pays through the core's token PSP with opaque tokens only.

Adopting or crossing onto a curated preset composes the preset, not an override: `shopify-live-2026-10` plus ACP
2026-04-17 is registered as `live-acp-2026-04-17`, and a composition whose versions match a registered preset comes
back as that preset with source "preset" (`carttrace.protocols.presets.match_preset`). Without this the episodes a paying
twin runs were all marked override and the release funnel refused them (D74), though nothing store-derived entered
the profile. A live or override base, or a combination no preset registers, still composes an override.

Rail versions and brokers (D71). A plan may cross a rail with versions of its protocol (`versions`): ucp_psp with every
UCP release the sandbox serves (2026-01-11 and 2026-01-23 are checkout-only, D71), ap2_mandates with AP2 v0.1.0 and
v0.2.0. A version is planned for a store only where its profile allows it (`version_allowed`): a store that speaks the
protocol is served its own release or an older one, never a newer one; a store that does not (the rail adopted) any
version an agent dialect speaks. And ucp_psp rows may be crossed with the payment broker behind the store's PSP
(`carttrace.sandbox.core.brokers`: stripe, adyen, braintree, paypal), which the row and its header name as `psp`.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Final

from carttrace.collect.profiles import StoreProtocols
from carttrace.protocols.presets import match_preset
from carttrace.protocols.profile import ProtocolProfile
from carttrace.protocols.versions import version_key


@dataclass(frozen=True, slots=True)
class Rail:
    """One payment rail: its protocol family, the version a store that does not speak it is served at, the row
    surfaces it runs on, and what an episode opens or hides for it."""

    name: str
    protocol: str
    """The ProtocolProfile field that holds its version: ucp, acp, ap2 or mpp; for a handler rail, the handler's
    namespace (which the header's rail_protocol names)."""
    kind: str
    """The sandbox.surface kind that serves the protocol, for whether the store serves it itself; unused for a
    handler rail, which is read from the profile's payment handlers."""
    adopt: str | None
    """The version a store that does not speak the protocol is served at; None for UCP, which every replica speaks."""
    surfaces: tuple[str, ...]
    """The row surfaces the rail runs on."""
    own_surface: bool = False
    """Its surface is the protocol's own (acp_rest, ap2), planned whatever UCP bindings the plan names."""
    payment_surface: str | None = None
    """A surface an episode opens beside the row's, on the same served store, to pay over (MPP)."""
    hides: frozenset[str] = frozenset()
    """Tools of the row's surface the agent is not offered on this rail."""
    spoken: tuple[str, ...] = ()
    """Versions of the protocol an agent dialect speaks; a profile at another is served at `adopt` instead."""
    versions: tuple[str, ...] = ()
    """Versions a plan may cross the rail with (D71); empty where the rail has one."""
    brokers: bool = False
    """The rail pays through the store's PSP handler, so a plan may cross it with payment brokers (D71)."""
    handler: bool = False
    """The rail is a UCP payment handler, not a protocol family (D73: Shop Pay): its version lives in the profile's
    `ucp_payment_handlers`, and the store speaks it when its profile or its store-replies record declares it."""


SHOP_PAY_NS: Final = "dev.shopify.shop_pay"
"""Shop Pay's UCP payment handler namespace, as vendor/dev.shopify.shop_pay/2026-04-08 and the live stores' records
name it (D73)."""
SHOP_PAY_VERSION: Final = "2026-04-08"
"""The Shop Pay handler version the sandbox serves and an agent speaks: the one vendored release."""

RAILS: Final[dict[str, Rail]] = {
    "ucp_psp": Rail(
        "ucp_psp",
        "ucp",
        "ucp_mcp",
        None,
        ("ucp_mcp", "ucp_rest", "webmcp"),
        versions=("2026-01-11", "2026-01-23", "2026-04-08", "2026-08-25"),
        brokers=True,
    ),
    "acp_delegated": Rail(
        "acp_delegated",
        "acp",
        "acp_rest",
        "2026-04-17",
        ("acp_rest",),
        own_surface=True,
        spoken=("2025-09-29", "2025-12-12", "2026-01-16", "2026-01-30", "2026-04-17"),
    ),
    "ap2_mandates": Rail(
        "ap2_mandates",
        "ap2",
        "ap2",
        "v0.2.0",
        ("ap2",),
        own_surface=True,
        spoken=("v0.1.0", "v0.2.0"),
        versions=("v0.1.0", "v0.2.0"),
    ),
    "mpp": Rail(
        "mpp",
        "mpp",
        "mpp",
        "pympp-0.11.0",
        ("ucp_mcp", "ucp_rest"),
        payment_surface="mpp",
        hides=frozenset({"complete_checkout"}),
        spoken=("pympp-0.11.0",),
    ),
    "shop_pay": Rail(
        "shop_pay",
        SHOP_PAY_NS,
        "ucp_mcp",
        None,
        ("ucp_mcp", "ucp_rest", "webmcp"),
        handler=True,
    ),
}
"""Every rail, in the order a plan lists them."""
RAIL_NAMES: Final = tuple(RAILS)


def rail(name: str) -> Rail:
    found = RAILS.get(name)
    if found is None:
        raise ValueError(f"no payment rail {name!r}; rails are {list(RAILS)}")
    return found


CHECKOUT_ONLY_UCP: Final = frozenset({"2026-01-11", "2026-01-23"})
"""UCP releases that define checkout only (D71): their rows run on ucp_mcp, since their REST binding requires a
request signature on every checkout call that the sandbox cannot verify."""


def version_allowed(profile: ProtocolProfile, own: StoreProtocols, name: str, version: str) -> str | None:
    """Why rail `name` cannot be served at `version` for this store, or None when its profile allows it: the store's
    own release or an older one where the store speaks the protocol, any version the rail crosses where it does not."""
    chosen = rail(name)
    if chosen.handler:
        return f"{chosen.name} is a payment handler, not a protocol family: it is not crossed with versions"
    if version not in chosen.versions:
        return f"{chosen.name} is not crossed with {chosen.protocol.upper()} {version}"
    current = getattr(profile, chosen.protocol)
    speaks = isinstance(current, str) and bool(current) and not adopted_by(own, name)
    if speaks and version_key(version) > version_key(str(current)):
        return f"the store speaks {chosen.protocol.upper()} {current}, older than {version}"
    return None


def _compose(profile: ProtocolProfile, note: str, **changes: Any) -> ProtocolProfile:
    """`profile.with_versions`, except a composition over a preset base that lands exactly on a registered preset's
    versions is that preset. Adopting a public protocol version onto a curated preset embeds nothing captured from a
    store, so the composition keeps source "preset" and the release's profile_override rule stays quiet; a live or
    override base, or a combination no preset registers, stays an override."""
    composed = profile.with_versions(note, **changes)
    if profile.source != "preset":
        return composed
    found = match_preset(composed.versions())
    return found if found is not None else composed


def at_version(profile: ProtocolProfile, name: str, version: str) -> ProtocolProfile:
    """`profile` with rail `name`'s protocol moved to `version`. UCP moves its core and keeps the capabilities the
    release defines, each at the release (checkout alone where it defines none of them); handlers are unchanged."""
    chosen = rail(name)
    if getattr(profile, chosen.protocol) == version:
        return profile
    said = f"payment rail {chosen.name}: {chosen.protocol.upper()} {version}"
    note = f"{profile.note}; {said}" if profile.note else said
    if chosen.protocol != "ucp":
        return _compose(profile, note, **{chosen.protocol: version})
    from carttrace.protocols.registry import registry

    defined = set(registry().ucp_capabilities(version))
    caps = {cap: version for cap in profile.ucp_capabilities if cap in defined}
    return _compose(profile, note, ucp=version, ucp_capabilities=caps or {"dev.ucp.shopping.checkout": version})


def rail_profile(profile: ProtocolProfile, name: str, version: str | None = None) -> ProtocolProfile:
    """The profile a world serves for rail `name`: the store's own when it already speaks the rail's protocol at a
    version an agent speaks, else the same versions with the rail's protocol at its adopted version; at `version`
    when the plan crosses the rail with versions (D71). A handler rail (D73) gains the handler in the profile's
    `ucp_payment_handlers` when the store's profile does not declare it."""
    chosen = rail(name)
    if chosen.handler:
        if SHOP_PAY_NS in profile.ucp_payment_handlers:
            return profile
        said = f"payment rail {chosen.name}: {SHOP_PAY_NS} {SHOP_PAY_VERSION} adopted"
        note = f"{profile.note}; {said}" if profile.note else said
        handlers = {**profile.ucp_payment_handlers, SHOP_PAY_NS: SHOP_PAY_VERSION}
        return profile.with_versions(note, ucp_payment_handlers=handlers)
    if version is not None:
        return at_version(profile, name, version)
    current = getattr(profile, chosen.protocol)
    if chosen.adopt is None or (isinstance(current, str) and current in chosen.spoken):
        return profile
    said = f"payment rail {chosen.name}: {chosen.protocol.upper()} {chosen.adopt} adopted"
    note = f"{profile.note}; {said}" if profile.note else said
    return _compose(profile, note, **{chosen.protocol: chosen.adopt})


def adopted_by(own: StoreProtocols, name: str) -> bool:
    """Whether rail `name` is adopted for this store: the store itself (its capture, preflight or profile) does not
    serve the rail's protocol; for a handler rail, does not declare the handler."""
    chosen = rail(name)
    if chosen.handler:
        return chosen.protocol not in own.profile.ucp_payment_handlers
    return not own.serves(chosen.kind)


def rail_surfaces(name: str, planned: Sequence[str]) -> list[str | None]:
    """The row surfaces rail `name` runs on for a plan naming `planned` surfaces: its own surface whatever the plan
    names; else the plan's surfaces it runs on, or UCP over MCP unnamed when the plan names none."""
    chosen = rail(name)
    if chosen.own_surface:
        return list(chosen.surfaces)
    if not planned:
        return [None]
    return [kind for kind in planned if kind in chosen.surfaces]


def rail_version(name: str, profile: ProtocolProfile) -> str | None:
    """The version of the rail's protocol a profile names; for a handler rail, the handler's declared version."""
    chosen = rail(name)
    if chosen.handler:
        return profile.ucp_payment_handlers.get(chosen.protocol)
    found = getattr(profile, chosen.protocol)
    return found if isinstance(found, str) and found else None


def rail_kinds(names: Collection[str]) -> frozenset[str]:
    """Every row surface the named rails run on, own surfaces included."""
    return frozenset(kind for n in names for kind in rail(n).surfaces)

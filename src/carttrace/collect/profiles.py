"""A store's own protocols: the profile its sandbox replica speaks, where that profile came from, and its surfaces.

Why per store. A verifier found every sandbox twin of 50 stores planned under one preset (shopify-live-2026-10), so
the benchmark measured one imagined store fifty times over. A store speaks particular versions (D4), declares
particular payment handlers and serves particular transports, and its replica must do the same (D12). So each store's
profile is looked up in this order, and the first one found wins:
1. a profile file given for the run (`carttrace collect manifest --profiles FILE`), store id to ProtocolProfile;
2. the replica capture: a catalog snapshot that carries the store's own profile (`profile` in its
   `envpack_kwargs()`, or a `profile` attribute) and, optionally, the surfaces it serves (`surfaces`);
3. the preflight: the profile its discovery document declared (carttrace.stores.results `profile`), with the surfaces the
   preflight found (`protocols`);
4. the plan's preset, which is the fallback and says so in the manifest's notes.

Surfaces. A surface is a `sandbox.surface` kind (carttrace.sandbox PLUGINS) at the version of its protocol family in the
profile: ucp_mcp and ucp_rest speak the UCP release, acp_rest the ACP API-Version, and so on. A store serves the
surfaces its capture or preflight found; when neither says (a profile file, a preset), it serves every surface whose
family the profile names. WebMCP is the exception that D16 makes: every replica of a UCP store has a storefront, at the
draft the profile names or the sandbox's newest, so a store that serves UCP serves webmcp too. A live row only ever
runs on UCP over MCP through carttrace.live, read-only (`LIVE_SURFACES`); the other surfaces are measured on the replica.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Final, get_args

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from carttrace.agents.dialects import DIALECT_KIND, SURFACE_DIALECTS
from carttrace.core.errors import ValidationError
from carttrace.core.registry import names
from carttrace.protocols.profile import ProtocolProfile
from carttrace.protocols.tools import SurfaceKind
from carttrace.sandbox.browser.webmcp import DEFAULT_WEBMCP
from carttrace.stores.results import StoreResult

SURFACE_KINDS: Final[tuple[str, ...]] = get_args(SurfaceKind)
"""Every surface kind the ToolSurface contract names, in its order: the canonical order of a plan's surfaces."""
DEFAULT_SURFACE: Final = "ucp_mcp"
"""The surface every row ran on before rows named one: a row without a surface runs here."""
LIVE_SURFACES: Final = frozenset({"ucp_mcp", "website", "webmcp"})
"""What a live row may run on: carttrace.live's allowlisted UCP over MCP client, a store's own pages through carttrace.live's
GET-only website surface (D31), or the WebMCP tools its page registers, run in the page behind the allowlist (D67),
and nothing else."""
LIVE_WEBMCP: Final = "webmcp"
LIVE_WEBMCP_VERSION: Final = "page"
"""A live WebMCP row's surface version: the tools are whatever the store's page registered (carttrace.live.webmcp_surface)."""
WEBSITE: Final = "website"
"""The live surface of a store that declares no agent protocol: its own website, ending at a selection (D31)."""
WEBSITE_VERSION: Final = "2026-10-12"
"""The website tools' contract (carttrace.live.website.VERSION), recorded as the row's surface version."""
SANDBOX_SURFACE_KIND: Final = "sandbox.surface"
"""The registry kind sandbox surfaces register under (carttrace.sandbox.server.SURFACE_KIND)."""
FAMILY: Final[Mapping[str, str]] = {
    "ucp_mcp": "ucp",
    "ucp_rest": "ucp",
    "acp_rest": "acp",
    "mpp": "mpp",
    "ap2": "ap2",
    "webmcp": "webmcp",
    "store_mcp": "mcp",
}
"""Surface kind to the ProtocolProfile field that holds its version."""
DECLARED_ONLY: Final = frozenset({"store_mcp"})
"""Kinds a store serves only when it declares them: every UCP over MCP profile names an MCP version, which does not
make the store's own MCP server a surface."""
STOREFRONT: Final = "webmcp"
"""The kind every UCP store's replica serves on its storefront (D16), at the profile's draft or the sandbox's newest."""
UCP_SURFACES: Final = frozenset({"ucp_mcp", "ucp_rest"})


class RowSurface(BaseModel):
    """The surface a manifest row runs on: a sandbox.surface kind and the version its family speaks there."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    version: str | None = None
    """The family's version in the row's profile; None for a live row whose store has not been captured yet (it
    speaks what discovery finds)."""

    @property
    def label(self) -> str:
        return f"{self.kind}@{self.version}" if self.version else self.kind


def surface_version(kind: str, profile: ProtocolProfile) -> str | None:
    """The version `profile` gives the surface's family, or None when the profile does not speak it. The storefront
    speaks the WebMCP draft the profile names, else the sandbox's newest, on every store that speaks UCP. A website
    speaks the website tools' contract on every store: every replica serves its store's pages (D67)."""
    if kind == WEBSITE:
        return WEBSITE_VERSION
    family = FAMILY.get(kind)
    value = getattr(profile, family) if family is not None else None
    if kind == STOREFRONT and not (isinstance(value, str) and value):
        value = DEFAULT_WEBMCP if isinstance(profile.ucp, str) and profile.ucp else None
    return value if isinstance(value, str) and value else None


def profile_surfaces(profile: ProtocolProfile) -> frozenset[str]:
    """The surfaces a profile can be served on: every kind whose family it names, the declared-only kinds apart."""
    return frozenset(k for k in SURFACE_KINDS if k not in DECLARED_ONLY and surface_version(k, profile) is not None)


def registered_surfaces() -> frozenset[str]:
    """The surface kinds the sandbox has an implementation for, from carttrace.sandbox's PLUGINS, and the website: every
    sandbox app serves its replica's pages (carttrace.sandbox.surfaces.site), which the collector reads through carttrace.live's
    own website tools (carttrace.collect.bindings, D67)."""
    return frozenset(names(SANDBOX_SURFACE_KIND)) | {WEBSITE}


NO_AGENT_DIALECT: Final = "no_agent_dialect"
"""The label of a sandbox row whose surface no agent dialect speaks: it ends harness_error before the agent starts,
and the judge voids it as the harness's."""

MODEL_UNAVAILABLE: Final = "model_unavailable"
"""The label of an episode the agent's model server failed (down, no reply in time, a 5xx): a harness_error that is
the infrastructure's. The scheduler pauses its lane and runs the row again (mc-twc)."""


def spoken_surfaces() -> frozenset[str]:
    """The surface kinds an agent can shop: those whose dialect (carttrace.agents.dialects.SURFACE_DIALECTS) is registered
    under agents.dialect. MPP and AP2 have a sandbox surface and no dialect yet."""
    registered = set(names(DIALECT_KIND))
    return frozenset(kind for kind, dialect in SURFACE_DIALECTS.items() if dialect in registered)


@dataclass(frozen=True, slots=True)
class StoreProtocols:
    """One store's protocol profile, where it came from, and the surfaces it serves."""

    profile: ProtocolProfile
    origin: str
    """profiles file, replica capture, preflight, or preset <name>."""
    served: frozenset[str] | None = None
    """The surfaces the capture or the preflight found; None when nothing said, and then the profile decides."""

    @property
    def from_store(self) -> bool:
        return not self.origin.startswith("preset")

    def serves(self, kind: str) -> bool:
        """Whether the store serves `kind` at a version its profile names. A store that serves UCP serves the
        storefront's WebMCP too: its replica mounts one (D16). Every store serves its website: its replica's pages
        carry its captured JSON-LD (D67)."""
        if kind == WEBSITE:
            return True
        if surface_version(kind, self.profile) is None:
            return False
        if self.served is not None:
            return kind in self.served or (kind == STOREFRONT and bool(UCP_SURFACES & self.served))
        return kind not in DECLARED_ONLY

    def surfaces(self) -> list[str]:
        """The agent protocol surfaces the store serves, in canonical order; its website, which every store serves,
        is not one of them."""
        return [kind for kind in SURFACE_KINDS if kind != WEBSITE and self.serves(kind)]

    def note(self, store_id: str) -> str:
        if self.from_store:
            return f"{store_id}: protocol profile {self.profile.id} from the store's {self.origin}"
        return f"{store_id}: no profile captured from the store; planned under {self.origin}"


def store_protocols(
    store_id: str,
    *,
    fallback: ProtocolProfile,
    fallback_name: str,
    explicit: Mapping[str, ProtocolProfile] | None = None,
    captured: CapturedStore | None = None,
    result: StoreResult | None = None,
) -> StoreProtocols:
    """The store's protocols, in the order the module docstring gives: the profiles file, the replica capture, the
    preflight, then the plan's preset."""
    given = (explicit or {}).get(store_id)
    if given is not None:
        return StoreProtocols(given, "profiles file")
    if captured is not None:
        return StoreProtocols(captured.profile, "replica capture", captured.surfaces)
    if result is not None and result.profile is not None:
        served = frozenset(result.protocols) if result.protocols else None
        return StoreProtocols(result.profile, "preflight", served)
    return StoreProtocols(fallback, f"preset {fallback_name}")


@dataclass(frozen=True, slots=True)
class CapturedStore:
    """What a replica capture says about the store's protocols."""

    profile: ProtocolProfile
    surfaces: frozenset[str] | None = None


def captured_store(raw: Any, arguments: Mapping[str, Any]) -> CapturedStore | None:
    """The store's own profile from a catalog snapshot that carries one (a replica capture), else None.

    The snapshot's `envpack_kwargs()` (`arguments` here) may hold `profile` (a ProtocolProfile or its JSON) and
    `surfaces` (the kinds the store serves); a snapshot object may carry them as attributes instead. A licensed catalog
    carries neither, and its twins fall back to the preset."""
    profile = arguments.get("profile", getattr(raw, "profile", None))
    if profile is None:
        return None
    try:
        parsed = profile if isinstance(profile, ProtocolProfile) else ProtocolProfile.model_validate(profile)
    except PydanticValidationError as exc:
        raise ValidationError(
            f"the catalog snapshot carries a profile that is not a ProtocolProfile: {exc.errors()[0]['msg']}",
            hint="a replica capture's profile holds ProtocolProfile fields (ucp, ucp_capabilities, ...)",
        ) from None
    surfaces = arguments.get("surfaces", getattr(raw, "surfaces", None))
    kinds = _kinds(surfaces) if surfaces is not None else None
    return CapturedStore(parsed, kinds)


def _kinds(value: Any) -> frozenset[str]:
    items: Collection[Any] = value if isinstance(value, list | tuple | set | frozenset) else list(value or ())
    found = frozenset(str(item) for item in items)
    unknown = sorted(found - set(SURFACE_KINDS))
    if unknown:
        raise ValidationError(
            f"the catalog snapshot names surfaces {unknown}, which are not surface kinds",
            hint=f"surface kinds are {list(SURFACE_KINDS)}",
        )
    return found

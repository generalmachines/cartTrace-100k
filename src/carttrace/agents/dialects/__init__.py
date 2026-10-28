"""The dialects: how an agent speaks UCP, ACP and WebMCP, chosen from the tools a surface lists.

Why here. The protocol agents take any `ToolSurface` and the benchmark measures what each store's protocol can do, so
the knowledge of each protocol's tool names, argument shapes and document shapes is kept in one place per protocol
(`base.Dialect` for UCP, `acp.AcpDialect`, `webmcp.WebMcpDialect`, `website.WebsiteDialect` for a store shopped through
its own pages, D31, the payment rails' `mpp.MppDialect` and `ap2.Ap2Dialect`, D66, and D71's
`ucp_checkout.UcpCheckoutDialect` for UCP's checkout-only releases and `ap2.Ap2V1Dialect` for AP2 v0.1.0) and chosen at
run time from the tools the episode's surfaces list (`dialect_for`), or by surface kind for the collector's read-backs
(`dialect_for_surface`). Each module also holds that protocol's checkout entry (how its checkout takes the buyer's
details, `carttrace.agents.entry`) and, for ACP, MPP and AP2, the wallet's credential issuer. All three are registered in
`carttrace.agents.PLUGINS` (`agents.dialect`, `agents.checkout_entry`, `agents.credential`), so a new protocol is one module
and three lines.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from carttrace.agents.dialects.acp import AcpDialect, AcpEntry, DelegatedTokenIssuer
from carttrace.agents.dialects.ap2 import Ap2Dialect, Ap2Entry, Ap2V1Dialect, Ap2V1Entry, MandateIssuer, WalletSigner
from carttrace.agents.dialects.base import Call, Dialect
from carttrace.agents.dialects.mpp import ChallengeCredentialIssuer, MppDialect
from carttrace.agents.dialects.ucp import UcpDialect
from carttrace.agents.dialects.ucp_checkout import UcpCheckoutDialect, UcpCheckoutEntry
from carttrace.agents.dialects.webmcp import WebMcpDialect, WebMcpEntry
from carttrace.agents.dialects.website import WebsiteDialect
from carttrace.core.errors import ConfigError
from carttrace.protocols.tools import ToolSpec

DIALECT_KIND = "agents.dialect"
"""The registry kind of the dialects (`carttrace.agents.PLUGINS`)."""
SURFACE_DIALECTS: Mapping[str, str] = {
    "ucp_mcp": "ucp",
    "ucp_rest": "ucp",
    "store_mcp": "ucp",
    "acp_rest": "acp",
    "ap2": "ap2",
    "webmcp": "webmcp",
    "website": "website",
}
"""Surface kind to the dialect spoken on it."""


def dialect_for(specs: Mapping[str, ToolSpec] | Collection[str]) -> Dialect:
    """The dialect the listed tools offer, shaped by their schemas when `specs` maps names to specs: the first
    registered dialect whose tools the surfaces list, else UCP (a surface that lists none of them has no catalog or
    cart an agent could use, and the agent says so itself)."""
    from carttrace.core.registry import names, resolve

    by_name: Mapping[str, ToolSpec] = specs if isinstance(specs, Mapping) else {}
    listed = set(specs)
    for name in names(DIALECT_KIND):
        cls = resolve(DIALECT_KIND, name)
        made: Dialect = cls.from_specs(by_name) if hasattr(cls, "from_specs") else cls()
        if made.offered(listed):
            return made
    return UcpDialect()


def dialect_for_surface(kind: str, specs: Mapping[str, ToolSpec] | None = None) -> Dialect:
    """The dialect spoken on a surface kind, shaped by `specs` when given."""
    from carttrace.core.registry import resolve

    name = SURFACE_DIALECTS.get(kind)
    if name is None:
        raise ConfigError(f"no dialect is spoken on {kind} surfaces", hint=f"one of {sorted(SURFACE_DIALECTS)}")
    cls = resolve(DIALECT_KIND, name)
    made: Dialect = cls.from_specs(specs or {}) if hasattr(cls, "from_specs") else cls()
    return made


__all__ = [
    "DIALECT_KIND",
    "SURFACE_DIALECTS",
    "AcpDialect",
    "AcpEntry",
    "Ap2Dialect",
    "Ap2Entry",
    "Ap2V1Dialect",
    "Ap2V1Entry",
    "Call",
    "ChallengeCredentialIssuer",
    "DelegatedTokenIssuer",
    "Dialect",
    "MandateIssuer",
    "MppDialect",
    "UcpCheckoutDialect",
    "UcpCheckoutEntry",
    "UcpDialect",
    "WalletSigner",
    "WebMcpDialect",
    "WebMcpEntry",
    "WebsiteDialect",
    "dialect_for",
    "dialect_for_surface",
]

"""`ProtocolProfile`: the exact protocol versions one store speaks, or one journey is run under.

Why it exists. Live stores do not speak "UCP"; they speak particular versions of particular pieces of it. A Shopify
store observed on 2026-09-29 declared UCP 2026-08-25 for its cart and catalog capabilities, a Google Pay payment
handler at 2026-01-11 and Shopify's card handler at 2026-01-15, over MCP. To hand a live cart to the sandbox and pay
for it the way that store would, the sandbox must speak those same versions; to measure what a version change does,
the same journey must run under two profiles that differ in one field. A profile is that set of versions as data, with
an id that is a hash of the versions alone, so two stores that speak the same versions share a profile id.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from carttrace.core.canon import digest

Source = Literal["live", "preset", "override"]


class ProtocolProfile(BaseModel):
    """Versions per protocol family. `None` means the store or journey does not use that family at all."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ucp: str | None = None
    """UCP core version from the discovery document's `ucp.version`, e.g. "2026-08-25"."""
    ucp_capabilities: dict[str, str] = Field(default_factory=dict)
    """Capability name to version, e.g. {"dev.ucp.shopping.cart": "2026-08-25"}."""
    ucp_payment_handlers: dict[str, str] = Field(default_factory=dict)
    """Handler namespace to version, e.g. {"com.google.pay": "2026-01-11", "dev.shopify.card": "2026-01-15"}."""
    mcp: str | None = None
    """MCP protocolVersion the session negotiated, e.g. "2025-06-18"."""
    acp: str | None = None
    """ACP API-Version, e.g. "2026-04-17"."""
    ap2: str | None = None
    """AP2 release, e.g. "v0.2.0"."""
    mpp: str | None = None
    """MPP release, e.g. "pympp-0.11.0"."""
    webmcp: str | None = None
    """Dated snapshot of the WebMCP draft the page's tools follow, e.g. "2026-08-14"."""

    source: Source = "preset"
    store: str | None = None
    """The store domain a live profile was captured from."""
    observed_at: str | None = None
    """RFC 3339 time the live profile was captured."""
    note: str = ""

    def versions(self) -> dict[str, Any]:
        """Only the version fields: what the id hashes and what two profiles are compared on."""
        return self.model_dump(
            include={"ucp", "ucp_capabilities", "ucp_payment_handlers", "mcp", "acp", "ap2", "mpp", "webmcp"}
        )

    @property
    def id(self) -> str:
        return f"pp-{digest(self.versions(), 12)}"

    def with_versions(self, note: str, **changes: Any) -> ProtocolProfile:
        """A derived profile for an experiment: the same versions except `changes`, marked as an override."""
        unknown = set(changes) - set(self.versions())
        if unknown:
            raise ValueError(f"not a version field: {sorted(unknown)}")
        return self.model_copy(update={**changes, "source": "override", "note": note})

    def differs_from(self, other: ProtocolProfile) -> dict[str, tuple[Any, Any]]:
        """Fields whose versions differ, as {field: (mine, theirs)}. Empty when the profiles speak the same versions."""
        a, b = self.versions(), other.versions()
        return {k: (a[k], b[k]) for k in a if a[k] != b[k]}

    @classmethod
    def from_ucp_discovery(
        cls, doc: Mapping[str, Any], *, mcp: str | None = None, store: str | None = None, observed_at: str | None = None
    ) -> ProtocolProfile:
        """Capture a live store's profile from its `/.well-known/ucp` document or the `ucp` block of a cart reply.

        Accepts either the whole response (with a top-level "ucp" object) or the "ucp" object itself. Each capability
        and handler may be declared as a list of version entries; the newest declared version is taken, because that
        is what a negotiating client ends up speaking.
        """
        ucp = doc.get("ucp", doc)
        if not isinstance(ucp, Mapping):
            raise ValueError("UCP discovery document has no 'ucp' object")
        return cls(
            ucp=_str_or_none(ucp.get("version")),
            ucp_capabilities=_newest_versions(ucp.get("capabilities")),
            ucp_payment_handlers=_newest_versions(ucp.get("payment_handlers")),
            mcp=mcp,
            source="live",
            store=store,
            observed_at=observed_at,
        )


def _str_or_none(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None


def _newest_versions(block: Any) -> dict[str, str]:
    if not isinstance(block, Mapping):
        return {}
    out: dict[str, str] = {}
    for name, entries in block.items():
        items = entries if isinstance(entries, list) else [entries]
        versions = sorted(e["version"] for e in items if isinstance(e, Mapping) and isinstance(e.get("version"), str))
        if versions:
            out[str(name)] = versions[-1]
    return out

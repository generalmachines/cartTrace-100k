"""Sandbox website rows (D67): a replica's own pages, read through the same website tools a live website row uses.

Why. Live website sessions (D31) read a store's pages, and on 259 of the 307 stores the 2026-W42 capture read those
pages carry schema.org JSON-LD. The sandbox served no pages, so live website rows had no sandbox half to compare
with. Every sandbox app now serves its replica's pages (carttrace.sandbox.surfaces.site: the captured JSON-LD verbatim, the
theme's variant data from the state), and a sandbox row on surface `website` opens carttrace.live's website surface on them:
- under a reserved host name (RFC 2606's .test: `<store domain with dots as dashes>.replica.test`), reached through
  `SiteBridge`, which answers in process and rewrites the store's own addresses in the JSON-LD to that host;
- behind a gate that opens for that host only (carttrace.live.PlaygroundGate: a store we own; the control file still pauses
  it), with a throttle and a robots cache of their own in a directory that lives as long as the session, so sandbox
  rows on one store run side by side and leave nothing behind;
- with the selection named in the pack's own ids (`site_catalog`), the ids the row's task witnesses use.

The agent's tools, its prompt and the page reader are the live ones; only the store differs. The trace header names
the JSON-LD types the replica's pages serve (`jsonld_types`).
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from carttrace.config.schema import Settings
from carttrace.protocols.profile import ProtocolProfile
from carttrace.protocols.tools import Mode, SurfaceKind, ToolResult, ToolSpec, ToolSurface

LIVE = "carttrace.live"
SITE = "carttrace.sandbox.surfaces.site"
REPLICA_SUFFIX = "replica.test"
SITE_RPS: Mapping[str, float] = {"default": 50.0}
"""Requests per second to a replica's pages: in process, so only the event loop bounds them."""
SITE_SESSIONS = 1_000_000


def website_condition_note(store_id: str, kind: str, condition: str) -> str:
    return (
        f"{store_id}: no {kind} website rows under {condition}: a website journey ends at a selection and takes "
        "baseline and cross_border only (D31, D67)"
    )


def replica_host(domain: str) -> str:
    """The reserved host a replica's website is addressed by: store-004.example -> store-004-example.replica.test."""
    label = "".join(c if c.isalnum() else "-" for c in domain.lower().removeprefix("www.")).strip("-")[:60]
    return f"{label or 'store'}.{REPLICA_SUFFIX}"


class ReplicaWebsite:
    """carttrace.live's website surface on a replica's pages, as a sandbox row's surface: it closes its throttle directory
    with the session and names the sandbox behind it, so the row's bundle carries the store's event log."""

    def __init__(
        self, inner: Any, *, host: Any, site_host: str, catalog: Any, jsonld_types: list[str], cleanup: Path
    ) -> None:
        self._inner = inner
        self.sandbox_host = host
        """The SandboxHost the pages are served from."""
        self.site_host = site_host
        self.selection_catalog = catalog
        """The pack's products as a capture catalog: the ids a website selection is named in."""
        self.jsonld_types = jsonld_types
        self._cleanup = cleanup

    @property
    def kind(self) -> SurfaceKind:
        kind: SurfaceKind = self._inner.kind
        return kind

    @property
    def mode(self) -> Mode:
        return "sandbox"

    @property
    def version(self) -> str:
        return str(self._inner.version)

    @property
    def endpoint(self) -> str:
        return f"https://{self.site_host}/"

    def profile(self) -> ProtocolProfile:
        profile: ProtocolProfile = self._inner.profile()
        return profile

    async def list_tools(self) -> list[ToolSpec]:
        listed: list[ToolSpec] = await self._inner.list_tools()
        return listed

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result: ToolResult = await self._inner.call(name, arguments)
        return result

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            shutil.rmtree(self._cleanup, ignore_errors=True)


async def open_replica_website(
    pack: Any,
    *,
    settings: Settings,
    recorder: Any = None,
    currency: str | None = None,
    market: str | None = None,
) -> ReplicaWebsite:
    """A sandbox website row's surface on `pack`'s replica pages (the module docstring says how)."""
    from carttrace.collect.bindings import contract
    from carttrace.core.clock import SystemClock

    app = contract("carttrace.sandbox", "build_app")(pack)
    domain = str(pack.store.domain)
    host = replica_host(domain)
    clock = SystemClock()
    directory = Path(tempfile.mkdtemp(prefix="mc-replica-site-"))
    try:
        throttle = contract(f"{LIVE}.throttle", "Throttle")(directory, SITE_RPS, daily_cap=SITE_SESSIONS, clock=clock)
        robots = _memory_robots(directory, clock)
        surface = await contract(LIVE, "open_website_surface")(
            host,
            gate=contract(LIVE, "PlaygroundGate")(settings, hosts=[host]),
            recorder=recorder,
            store_id=str(pack.store.store_id),
            platform=pack.store.platform,
            currency=currency,
            market=market,
            clock=clock,
            transport=contract(SITE, "SiteBridge")(app, host, domains=[domain]),
            throttle=throttle,
            robots=robots,
        )
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return ReplicaWebsite(
        surface,
        host=app.state.host,
        site_host=host,
        catalog=site_catalog(app, host),
        jsonld_types=contract(SITE, "pack_jsonld_types")(pack),
        cleanup=directory,
    )


def site_catalog(app: Any, host: str) -> Any:
    """The replica's products as carttrace.live's CaptureCatalog, each at its page on `host`, with the pack's ids."""
    from carttrace.collect.bindings import contract

    site = app.state.site
    page_path = contract(SITE, "page_path")
    products: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    for product in site.products():
        products.append(
            {
                "product_id": product.id,
                "url": f"https://{host}{page_path(product)}",
                "title": product.title,
                "handle": product.facts.get("handle") or "",
            }
        )
        for variant in product.variants:
            doc = variant.document or {}
            variants.append(
                {
                    "product_id": product.id,
                    "variant_id": variant.id,
                    "sku": doc.get("sku") if isinstance(doc, Mapping) else None,
                    "options": dict(variant.options),
                    "title": variant.title,
                    "price_minor": variant.price,
                    "currency": variant.currency,
                    "available": variant.available,
                }
            )
    store_id = str(app.state.host.pack.store.store_id)
    return contract(f"{LIVE}.capture_catalog", "catalog_from_rows")(store_id, "replica", products, variants, [])


def _memory_robots(directory: Path, clock: Any) -> Any:
    """A robots cache that keeps the replica's robots.txt for the session only, in its throwaway directory."""
    from carttrace.collect.bindings import contract

    return contract(f"{LIVE}.robots", "RobotsCache")(directory / "robots", clock=clock)


def is_replica_website(surface: ToolSurface) -> bool:
    return isinstance(surface, ReplicaWebsite)


def site_jsonld_types(pack: Any) -> list[str]:
    """The schema.org types a pack's replica pages serve (carttrace.sandbox.surfaces.site.pack_jsonld_types)."""
    from carttrace.collect.bindings import contract

    found: list[str] = contract(SITE, "pack_jsonld_types")(pack)
    return found

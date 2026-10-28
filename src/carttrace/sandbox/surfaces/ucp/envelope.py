"""The `ucp` block every response carries, and the business profile at `/.well-known/ucp`, from the served profile.

What is declared. The capabilities are the profile's own, at exactly the versions it names (a live capture keeps the
store's versions), plus what the sandbox always serves at the release (`UcpDialect.implied`: fulfillment, which a
checkout needs to be completable, and the discounts and orders the core implements) at the profile's UCP version. A
profile that names no capabilities declares the release's defaults. The payment authentication extension is declared
only when the profile names it, because declaring it changes how a 3-D Secure challenge reaches the agent.

The payment handlers are the envpack's, exactly as the live store declared them: each entry keeps its own `id`,
`version` and `config`; an entry that declared no version takes the profile's version for its namespace. The sandbox's
credential-free handler is added when the pack's policy serves it.

What is not done. UCP businesses fetch the platform's profile named in `meta["ucp-agent"]` and intersect capabilities
with it (overview, "Negotiation Protocol"). The sandbox sends no network traffic, so it does not fetch it: every
declared capability is active for every request.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from carttrace.protocols.profile import ProtocolProfile
from carttrace.sandbox.core.config import SHOP_PAY_HANDLER_ID, SHOP_PAY_NAMESPACE
from carttrace.sandbox.envpack.doc import handler_entries
from carttrace.sandbox.envpack.pack import EnvPack
from carttrace.sandbox.platforms.woocommerce import ucp_supported_versions
from carttrace.sandbox.presented import presented
from carttrace.sandbox.surfaces.ucp.dialect import (
    CAPABILITY_DOCS,
    CART,
    LOOKUP,
    PAYMENT_AUTHENTICATION,
    RELEVANT,
    SEARCH,
    SHOPPING_SERVICE,
    ResponseKind,
    UcpDialect,
)

UCP_SITE = "https://ucp.dev"
MCP_PATH = "/api/ucp/mcp"
REST_PATH = "/api/ucp"
FAKE_HANDLER_ID = "fake"


FEED_KINDS = ("search", "lookup")
"""The product feed's documents at a checkout-only release, which are the feed release's (D71): their block is a
capability registry, never 2026-01-11's list (the pack conformance suite, D101, found the list on every catalog reply
at 2026-01-11, which the feed release's schemas refuse)."""


class Envelope:
    """The declared capabilities and payment handlers of one served profile."""

    def __init__(
        self,
        dialect: UcpDialect,
        profile: ProtocolProfile,
        pack: EnvPack,
        *,
        defined: set[str],
        full: bool = False,
        extends: bool = False,
        supported_versions: bool = False,
        unserved: tuple[str, ...] = (),
    ) -> None:
        """`defined` is every capability name the release defines (`Registry.ucp_capabilities`). `full` is the
        platform's `full_ucp_block` (live Shopify): every response lists the cart and catalog capabilities with their
        spec and schema links, the payment handlers with their configs, and the `services` registry. `extends` names
        each extension's parent in a response's block, and `supported_versions` lists the older releases and the
        store's name in the discovery document (WooCommerce's UCP plugin, captures/diff-woo 140 and 142).
        `unserved` names capabilities the platform's plugin does not serve (Medusa's has no discount, D68)."""
        self.full = full
        self.extends = extends
        self.supported_versions = supported_versions
        self.store_name = presented(pack).domain
        self.dialect = dialect
        self.version = dialect.version
        named = dict(profile.ucp_capabilities)
        base = named or dict.fromkeys(dialect.defaults, self.version)
        for name in dialect.implied:
            base.setdefault(name, self.version)
        self.capabilities: dict[str, str] = {
            name: version
            for name, version in sorted(base.items())
            if name in CAPABILITY_DOCS and name in defined and name not in unserved
        }
        self.handlers: dict[str, list[dict[str, Any]]] = _handlers(pack, profile, self.version)

    def active(self, name: str) -> bool:
        return name in self.capabilities

    @property
    def actions(self) -> bool:
        """3-D Secure is an Action: the release defines payment authentication and the profile declares it."""
        return self.dialect.payment_actions and self.active(PAYMENT_AUTHENTICATION)

    def block(self, kind: ResponseKind, *, status: str = "success") -> dict[str, Any]:
        """The `ucp` member of a response of `kind`: version, status, the relevant active capabilities, and for a
        checkout the payment handlers (required there by `response_checkout_schema`)."""
        if self.full and not self.dialect.capability_list:
            return self._full_block(kind, status)
        if self.dialect.capability_list and kind not in FEED_KINDS:
            listed = [{"name": n, "version": self.capabilities[n]} for n in RELEVANT[kind] if n in self.capabilities]
            plain: dict[str, Any] = {"version": self.version, "capabilities": listed}
            if status != "success":
                plain["status"] = status
            return plain
        caps = {n: [self._entry(n)] for n in RELEVANT[kind] if n in self.capabilities}
        out: dict[str, Any] = {"version": self.version, "status": status, "capabilities": caps}
        if self.dialect.checkout_only and status == "success":
            del out["status"]  # the checkout-only releases define no response status
        if kind == "checkout":
            out["payment_handlers"] = {
                ns: [_response_entry(e) for e in entries] for ns, entries in self.handlers.items()
            }
        return out

    def _full_block(self, kind: ResponseKind, status: str) -> dict[str, Any]:
        """The block as live Shopify sends it on every cart and catalog reply (var/live/delta-01/report.md section
        1.3): cart, catalog.lookup and catalog.search with spec and schema links, every payment handler with its
        config, and an empty `services` registry for the shopping service. Checkouts and orders list their own
        capabilities the same way."""
        names = FULL_RELEVANT.get(kind, RELEVANT[kind])
        caps = {n: [self._linked(n)] for n in names if n in self.capabilities}
        return {
            "version": self.version,
            "status": status,
            "services": {SHOPPING_SERVICE: []},
            "capabilities": caps,
            "payment_handlers": {ns: [_response_entry(e) for e in entries] for ns, entries in self.handlers.items()},
        }

    def _entry(self, name: str) -> dict[str, Any]:
        """A capability as a response's block lists it: its version, and its parent where the platform names it."""
        entry: dict[str, Any] = {"version": self.capabilities[name]}
        parent = self._parent(name) if self.extends else None
        if parent is not None:
            entry["extends"] = parent
        return entry

    def _parent(self, name: str) -> str | list[str] | None:
        parents = [p for p in CAPABILITY_DOCS[name].extends if p in self.capabilities]
        if not parents:
            return None
        return parents[0] if len(parents) == 1 else parents

    def _linked(self, name: str) -> dict[str, Any]:
        version = self.capabilities[name]
        return {
            "version": version,
            "spec": f"{UCP_SITE}/{version}/specification/{CAPABILITY_DOCS[name].spec}/",
            "schema": f"{UCP_SITE}/{version}/schemas/{_schema_path(name)}",
        }

    def payment_handlers(self) -> list[dict[str, Any]]:
        """The handlers as 2026-01-11 lists them in a checkout's `payment.handlers` and the discovery document's
        `payment.handlers`: each with its id, namespace (`name`), version, spec, config schema, instrument schemas
        and config. A declared entry's own links are kept; one that declared none points at the namespace's page."""
        out: list[dict[str, Any]] = []
        for ns, entries in self.handlers.items():
            for e in entries:
                page = str(e.get("spec") or f"{UCP_SITE}/handlers/{ns}")
                out.append(
                    {
                        "id": str(e["id"]),
                        "name": ns,
                        "version": str(e["version"]),
                        "spec": page,
                        "config_schema": str(e.get("config_schema") or f"{page.rstrip('/')}/config.json"),
                        "instrument_schemas": list(
                            e.get("instrument_schemas") or [f"{page.rstrip('/')}/instrument.json"]
                        ),
                        "config": dict(e.get("config") or {}),
                    }
                )
        return out

    def discovery(self, base_url: str) -> dict[str, Any]:
        """The business profile served at `/.well-known/ucp`: both transport bindings, every declared capability with
        its spec and schema, and the payment handlers as declared."""
        v = self.version
        if self.dialect.capability_list:
            return self._listed_discovery(base_url)
        services = [
            {
                "version": v,
                "spec": f"{UCP_SITE}/{v}/specification/overview",
                "transport": transport,
                "schema": f"{UCP_SITE}/{v}/services/shopping/{schema}",
                "endpoint": f"{base_url}{path}",
            }
            for transport, schema, path in (
                ("mcp", "mcp.openrpc.json", MCP_PATH),
                ("rest", "rest.openapi.json", REST_PATH),
            )
        ]
        capabilities: dict[str, list[dict[str, Any]]] = {}
        for name, version in self.capabilities.items():
            doc = CAPABILITY_DOCS[name]
            entry: dict[str, Any] = {
                "version": version,
                "spec": f"{UCP_SITE}/{version}/specification/{doc.spec}",
                "schema": f"{UCP_SITE}/{version}/schemas/{_schema_path(name)}",
            }
            parent = self._parent(name)
            if parent is not None:
                entry["extends"] = parent
            capabilities[name] = [entry]
        ucp: dict[str, Any] = {"version": v}
        if self.supported_versions:
            ucp["supported_versions"] = ucp_supported_versions(base_url)
        ucp.update({"services": {SHOPPING_SERVICE: services}, "capabilities": capabilities})
        ucp["payment_handlers"] = self.handlers
        out: dict[str, Any] = {"ucp": ucp}
        if self.supported_versions:
            out["name"] = self.store_name
        return out

    def _listed_discovery(self, base_url: str) -> dict[str, Any]:
        """The 2026-01-11 profile: the shopping service with one object per transport, capabilities as an array of
        named entries, and the payment handlers under `payment.handlers`."""
        v = self.version
        service = {
            "version": v,
            "spec": f"{UCP_SITE}/{v}/specification/overview",
            "mcp": {
                "schema": f"{UCP_SITE}/{v}/services/shopping/mcp.openrpc.json",
                "endpoint": f"{base_url}{MCP_PATH}",
            },
            "rest": {
                "schema": f"{UCP_SITE}/{v}/services/shopping/rest.openapi.json",
                "endpoint": f"{base_url}{REST_PATH}",
            },
        }
        capabilities: list[dict[str, Any]] = []
        for name, version in self.capabilities.items():
            entry: dict[str, Any] = {
                "name": name,
                "version": version,
                "spec": f"{UCP_SITE}/{version}/specification/{CAPABILITY_DOCS[name].spec}",
                "schema": f"{UCP_SITE}/{version}/schemas/{_schema_path(name)}",
            }
            parent = self._parent(name)
            if isinstance(parent, str):
                entry["extends"] = parent
            capabilities.append(entry)
        return {
            "ucp": {"version": v, "services": {SHOPPING_SERVICE: service}, "capabilities": capabilities},
            "payment": {"handlers": self.payment_handlers()},
        }


FULL_RELEVANT: Mapping[ResponseKind, tuple[str, ...]] = {
    "cart": (CART, LOOKUP, SEARCH),
    "search": (CART, LOOKUP, SEARCH),
    "lookup": (CART, LOOKUP, SEARCH),
}
"""What live Shopify lists on cart and catalog replies alike (315 cart replies, every catalog reply)."""

SCHEMA_FILES: Mapping[str, str] = {
    "dev.ucp.shopping.checkout": "shopping/checkout.json",
    "dev.ucp.shopping.cart": "shopping/cart.json",
    "dev.ucp.shopping.order": "shopping/order.json",
    "dev.ucp.shopping.catalog.search": "shopping/catalog_search.json",
    "dev.ucp.shopping.catalog.lookup": "shopping/catalog_lookup.json",
    "dev.ucp.shopping.fulfillment": "shopping/fulfillment.json",
    "dev.ucp.shopping.discount": "shopping/discount.json",
    "dev.ucp.common.payment.authentication": "common/payment_authentication.json",
}


def _schema_path(name: str) -> str:
    return SCHEMA_FILES[name]


def _handlers(pack: EnvPack, profile: ProtocolProfile, version: str) -> dict[str, list[dict[str, Any]]]:
    """Namespace to entries, as the store declared them; entries missing `id` or `version` are completed from the
    namespace and the profile, never invented beyond that. A profile that adopts Shop Pay without the store declaring
    it (D73) serves the spec's own entry shape with the world's own `shop_id` (`adopted_shop_id`); a store that
    declared Shop Pay keeps its recorded entry (D70)."""
    out: dict[str, list[dict[str, Any]]] = {}
    declared = pack.payment_handlers
    for ns in sorted(set(declared) | set(profile.ucp_payment_handlers)):
        entries = [dict(e) for e in handler_entries(declared.get(ns))] or [{}]
        fallback = profile.ucp_payment_handlers.get(ns, version)
        if ns == SHOP_PAY_NAMESPACE and entries == [{}]:
            out[ns] = [_adopted_shop_pay(str(fallback), presented(pack).shop_id)]
            continue
        out[ns] = [{**e, "id": str(e.get("id") or ns), "version": str(e.get("version") or fallback)} for e in entries]
    shown = presented(pack)
    if shown.card_namespace in pack.handlers():
        out[shown.card_namespace] = [
            {
                "id": shown.card_handler_id,
                "version": version,
                "available_instruments": [{"type": "card"}],
                "config": shown.card_note(),
            }
        ]
    return out


def _response_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """A handler as a checkout response carries it: its id, version, instruments and config."""
    keep = ("id", "version", "spec", "schema", "available_instruments", "config")
    return {k: entry[k] for k in keep if k in entry}


def _adopted_shop_pay(version: str, shop_id: str) -> dict[str, Any]:
    """The Shop Pay handler entry a store that adopted the handler serves (D73): the spec's own id and links
    (vendor/dev.shopify.shop_pay/2026-04-08/spec.md, "Handler configuration") with the world's `shop_id` (the pack's
    `shop_id`, `adopted_shop_id` where the store declared none)."""
    page = f"https://shopify.dev/ucp/shop-pay-handler/{version}"
    return {
        "id": SHOP_PAY_HANDLER_ID,
        "version": version,
        "spec": f"{page}/spec.md",
        "schema": f"{page}/schema.json",
        "available_instruments": [{"type": "shop_pay"}],
        "config": {"shop_id": shop_id},
    }

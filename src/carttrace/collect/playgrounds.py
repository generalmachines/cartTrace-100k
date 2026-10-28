"""Playground stores: real stores we own and run on this machine, shopped as mode "playground" (D44).

Why. The project (2026-10-13) wants trace variety from a real WooCommerce store: the WooCommerce playground
(ops/playgrounds/woocommerce, WooCommerce 11.2.1, 20 seeded products on 127.0.0.1:19300) is one we own, so agents may
shop it the whole way: through its storefront pages with the website dialect (D31, ending at a verified selection),
and through its UCP REST plugin, where a cart may be written and a checkout created and completed with its offline
gateways (cash on delivery, bank transfer). It is never a live store: its rows are mode "playground", its store id is
its own (PG-WOO), its rights are internal, and its results never enter the per-store results database.

`data/playgrounds/stores.json` (`mc.playgrounds.v1`) lists each playground: the reserved host it is addressed by on
the website road (`woo.playground.test`, RFC 2606), its loopback origin, its category and market, the surfaces rows
run on, where its UCP discovery document and Store API answer, and where its admin credentials live (the env file
pg.sh wrote; read here, never printed). This module holds what the planner and the episode need from it:
- `load_playgrounds`, `store_record`: the file, and each playground as the queue's `Store` the planner plans;
- `playground_fixtures`: its products as a shopper's Store API serves them, in the live fixtures' shape, ids the
  store's own (a variation id is the id its pages, its UCP plugin and its orders all name);
- `PlaygroundCatalog`: those ids, for naming a website selection by the store's own variant ids;
- `read_order`: an order as WooCommerce's admin API (wc/v3) holds it, the ground truth the judge reads after a
  checkout row.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast, get_args

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from carttrace.config.http import async_client, sync_client
from carttrace.config.schema import Settings
from carttrace.core.canon import loads
from carttrace.core.errors import ConfigError, NotFoundError, ValidationError
from carttrace.core.ids import stable_id
from carttrace.stores.categories import CategoryCatalog
from carttrace.stores.model import PlatformKey, Store
from carttrace.trace.schema import Cart, CartLine, Rights

PLAYGROUNDS_SCHEMA: Final = "mc.playgrounds.v1"
PLAYGROUNDS_FILE: Final = Path("playgrounds") / "stores.json"
"""Under settings.paths.data."""
PLAYGROUND_MODE: Final = "playground"
STORE_ID_PREFIX: Final = "PG-"
PER_PAGE: Final = 100
TIMEOUT_S: Final = 30.0
RECENT_ORDERS: Final = 50
"""Orders read to find one by its key: a run's orders are the newest, and a playground is reset between rounds."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PlaygroundAdmin(_Frozen):
    """Where the playground's admin API answers and which env file keys hold its Basic auth (never the values)."""

    env_file: str
    """Relative to the repository root: the file the playground's own script wrote, mode 600."""
    user_key: str
    password_key: str
    orders_path: str = "/wp-json/wc/v3/orders"


class PlaygroundStore(_Frozen):
    store_id: str = Field(pattern=r"^PG-[A-Z0-9-]+$")
    name: str
    host: str
    """The reserved name the website road addresses it by (RFC 2606 .test)."""
    origin: str
    """Its loopback origin, http://127.0.0.1:<port>."""
    platform: str
    category: str
    market: str = Field(pattern=r"^[A-Z]{2}$")
    surfaces: list[str] = Field(min_length=1)
    ucp_discovery: str = "/.well-known/ucp"
    store_api: str = "/wp-json/wc/store/v1"
    admin: PlaygroundAdmin | None = None
    directory: str = ""
    """The playground's own directory (compose file, seed, pg.sh), relative to the repository root."""
    reset: list[str] = Field(default_factory=list)
    """The command, run in `directory`, that restores the playground's seeded state (pg.sh reset fd)."""
    rights: Rights


class PlaygroundBook(_Frozen):
    schema_: str = Field(alias="schema")
    stores: list[PlaygroundStore]

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


def playgrounds_path(settings: Settings) -> Path:
    return settings.paths.data / PLAYGROUNDS_FILE


def load_playgrounds(settings: Settings, path: Path | None = None) -> dict[str, PlaygroundStore]:
    """Store id to each playground the file lists; empty when there is no file."""
    found = path or playgrounds_path(settings)
    if not found.is_file():
        return {}
    try:
        book = PlaygroundBook.model_validate(loads(found.read_text(encoding="utf-8")))
    except (ValueError, PydanticValidationError) as exc:
        raise ConfigError(f"{found} is not a {PLAYGROUNDS_SCHEMA} file: {exc}", hint="fix it by hand") from None
    if book.schema_ != PLAYGROUNDS_SCHEMA:
        raise ConfigError(f"{found} is schema {book.schema_!r}, not {PLAYGROUNDS_SCHEMA}")
    out: dict[str, PlaygroundStore] = {}
    for store in book.stores:
        if store.store_id in out:
            raise ConfigError(f"{found} lists {store.store_id} twice")
        out[store.store_id] = store
    return out


def is_playground_id(store_id: str) -> bool:
    return store_id.startswith(STORE_ID_PREFIX)


def store_record(pg: PlaygroundStore, categories: CategoryCatalog) -> Store:
    """A playground as the queue's Store, for the manifest planner: its own id, host and category, platform as the
    file says, merchant group of its own, split train."""
    category = categories.get(pg.category)
    return Store(
        id=pg.store_id,
        category=category.name,
        name=pg.name,
        website=pg.host,
        platform=pg.platform.capitalize(),
        market=pg.market,
        parent_company=None,
        status="Playground we own",
        what_we_saw="",
        what_to_do_next="",
        test_fixture=None,
        last_observed=None,
        category_index=category.index,
        url=None,
        source_status=None,
        checked_at=None,
        fit_reason=None,
        observed_evidence=None,
        parent_group=None,
        evidence_url=None,
        canonical_host=pg.host,
        duplicate_host_count=None,
        publication_permission=None,
        eligibility=None,
        platform_evidence=None,
        delivery_market=None,
        sheet_merchant_group_id=None,
        browser_preflight_result=None,
        task_fixture=None,
        collector_eligibility=None,
        sheet_split=None,
        gate_approval_reference=None,
        category_slug=category.slug,
        status_key="other",
        platform_key=cast(PlatformKey, pg.platform if pg.platform in get_args(PlatformKey) else "other"),
        market_code=pg.market,
        market_likely=False,
        currency=None,
        domain=pg.host,
        registrable_domain=pg.host,
        merchant_key=pg.host,
        merchant_basis="domain",
        merchant_group_id=stable_id("mg", pg.store_id),
        split="train",
        added_mode=None,
    )


# -- the store's products ----------------------------------------------------------------------------------------------


def playground_fixtures(pg: PlaygroundStore, *, transport: httpx.BaseTransport | None = None) -> list[dict[str, Any]]:
    """Every product the playground's Store API serves a shopper, in the live fixtures' shape (id, title, variants
    with id, title, price in minor units with its currency, availability and options, product type, vendor,
    description), with the store's own ids; variable products' variations read from the same API."""
    with sync_client(base_url=pg.origin, timeout=TIMEOUT_S, transport=transport) as client:
        products = _get_list(client, f"{pg.store_api}/products", {"per_page": PER_PAGE})
        out = []
        for product in products:
            variations: list[Mapping[str, Any]] = []
            if product.get("type") == "variable":
                variations = _get_list(
                    client,
                    f"{pg.store_api}/products",
                    {"type": "variation", "parent": product["id"], "per_page": PER_PAGE},
                )
            fixture = _fixture(product, variations)
            if fixture["variants"]:
                out.append(fixture)
    return sorted(out, key=lambda p: int(p["id"]) if str(p["id"]).isdigit() else 0)


def _fixture(product: Mapping[str, Any], variations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    named = {int(v["id"]): v.get("attributes") or [] for v in product.get("variations") or () if "id" in v}
    rows = variations or [product]
    variants = []
    for row in rows:
        options = [
            {"name": str(a.get("name")), "label": str(a.get("value"))}
            for a in named.get(int(row["id"]), [])
            if a.get("value")
        ]
        price = _price(row.get("prices"))
        if price is None:
            continue
        title = " / ".join(o["label"] for o in options) or _text(row.get("name"))
        variants.append(
            {
                "id": str(row["id"]),
                "title": title,
                "price": price,
                "availability": {"available": bool(row.get("is_in_stock")) and row.get("is_purchasable") is not False},
                "options": options,
                **({"sku": str(row["sku"])} if row.get("sku") else {}),
            }
        )
    categories = [_text(c.get("name")) for c in product.get("categories") or () if isinstance(c, Mapping)]
    return {
        "id": str(product["id"]),
        "title": _text(product.get("name")),
        "url": str(product.get("permalink") or ""),
        "variants": variants,
        "product_type": categories[0].replace("_", " ").lower() if categories else "",
        "vendor": "206 Collective" if "206 collective" in _text(product.get("name")).lower() else "",
        "description": {"html": str(product.get("description") or "")},
    }


def _price(prices: Any) -> dict[str, Any] | None:
    if not isinstance(prices, Mapping):
        return None
    amount, currency = prices.get("price"), prices.get("currency_code")
    try:
        minor = int(str(amount))
    except ValueError:
        return None
    return {"amount": minor, "currency": str(currency)} if isinstance(currency, str) and currency else None


def _get_list(client: httpx.Client, path: str, params: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    response = client.get(path, params=dict(params))
    if response.status_code != 200:
        raise NotFoundError(
            f"the playground answered {response.status_code} for {path}",
            hint="is it up? ops/playgrounds/woocommerce/pg.sh up",
        )
    body = response.json()
    return [row for row in body if isinstance(row, Mapping)] if isinstance(body, list) else []


def _text(value: Any) -> str:
    return " ".join(html.unescape(str(value or "")).split())


@dataclass(frozen=True, slots=True)
class Resolved:
    product_id: str
    variant_id: str
    id_source: str


class PlaygroundCatalog:
    """The playground's own ids, so a website selection names the variant the store, its UCP plugin and its orders
    name (carttrace.collect.selection's Catalog)."""

    def __init__(self, fixtures: Sequence[Mapping[str, Any]]) -> None:
        self._by_variant: dict[str, str] = {}
        self._by_sku: dict[str, tuple[str, str]] = {}
        for product in fixtures:
            for variant in product.get("variants") or ():
                self._by_variant[str(variant["id"])] = str(product["id"])
                if variant.get("sku"):
                    self._by_sku[str(variant["sku"])] = (str(product["id"]), str(variant["id"]))

    def resolve(
        self, url: str, *, variant_id: str | None, sku: str | None, options: Mapping[str, str]
    ) -> Resolved | None:
        del url, options
        if variant_id is not None and variant_id in self._by_variant:
            return Resolved(self._by_variant[variant_id], variant_id, "store")
        if sku is not None and sku in self._by_sku:
            product, variant = self._by_sku[sku]
            return Resolved(product, variant, "store")
        return None


# -- the admin API's word on an order ----------------------------------------------------------------------------------


def admin_auth(pg: PlaygroundStore, home: Path) -> tuple[str, str]:
    """The admin's Basic auth from the playground's env file. Raises ConfigError naming the keys, never a value."""
    if pg.admin is None:
        raise ConfigError(f"{pg.store_id} names no admin API", hint="add `admin` to data/playgrounds/stores.json")
    path = home / pg.admin.env_file
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.lstrip().startswith("#"):
                values[key.strip()] = value.strip()
    except OSError:
        raise ConfigError(
            f"{pg.store_id}: no env file at {path}", hint=f"bring the playground up: {pg.directory}/pg.sh up"
        ) from None
    user, password = values.get(pg.admin.user_key), values.get(pg.admin.password_key)
    if not user or not password:
        raise ConfigError(f"{path} sets no {pg.admin.user_key} or {pg.admin.password_key}", hint="rerun pg.sh up")
    return user, password


@dataclass(frozen=True, slots=True)
class OrderRead:
    """One wc/v3 order read: what the trace may keep of it (no names, no addresses), and the order as a cart."""

    url: str
    status: int
    summary: dict[str, Any]
    cart: Cart | None


async def read_order(
    pg: PlaygroundStore, home: Path, order: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> OrderRead:
    """The order `order` (its number, or its key as UCP shows it, wc_order_...) as WooCommerce's admin API holds it:
    status, payment method, totals and lines. The summary keeps no buyer name, email or address."""
    if pg.admin is None:
        raise ConfigError(f"{pg.store_id} names no admin API", hint="add `admin` to data/playgrounds/stores.json")
    auth = admin_auth(pg, home)
    path = pg.admin.orders_path.rstrip("/")
    async with async_client(base_url=pg.origin, timeout=TIMEOUT_S, transport=transport) as client:
        if order.isdigit():
            response = await client.get(f"{path}/{order}", auth=auth)
            body: Any = response.json() if response.status_code == 200 else None
        else:
            # An order key (wc_order_...) is no search term wc/v3 matches, so the newest orders are read and matched.
            response = await client.get(path, params={"per_page": RECENT_ORDERS, "orderby": "date"}, auth=auth)
            listed = response.json() if response.status_code == 200 else []
            body = next((o for o in listed if isinstance(o, Mapping) and o.get("order_key") == order), None)
        url = str(response.request.url)
    if not isinstance(body, Mapping):
        return OrderRead(url, response.status_code, {"found": False, "order": order}, None)
    summary = order_summary(body)
    return OrderRead(url, response.status_code, summary, order_cart(body))


def order_summary(order: Mapping[str, Any]) -> dict[str, Any]:
    """What a bundle keeps of a wc/v3 order: its number, status, payment method, how it was created, currency and
    totals, and each line's ids, quantity and totals."""
    return {
        "found": True,
        "id": order.get("id"),
        "number": order.get("number"),
        "status": order.get("status"),
        "payment_method": order.get("payment_method"),
        "payment_method_title": order.get("payment_method_title"),
        "created_via": order.get("created_via"),
        "currency": order.get("currency"),
        "total": order.get("total"),
        "total_tax": order.get("total_tax"),
        "shipping_total": order.get("shipping_total"),
        "discount_total": order.get("discount_total"),
        "lines": [
            {
                "product_id": line.get("product_id"),
                "variation_id": line.get("variation_id"),
                "quantity": line.get("quantity"),
                "subtotal": line.get("subtotal"),
                "total": line.get("total"),
            }
            for line in order.get("line_items") or ()
            if isinstance(line, Mapping)
        ],
        "shipping_lines": [
            {"method_id": s.get("method_id"), "total": s.get("total")}
            for s in order.get("shipping_lines") or ()
            if isinstance(s, Mapping)
        ],
    }


def order_cart(order: Mapping[str, Any]) -> Cart | None:
    """A wc/v3 order as the trace's Cart: one line per item at its unit price, the order's total (shipping and tax
    in), in minor units; None for an order without lines."""
    currency = str(order.get("currency") or "")
    lines = []
    for line in order.get("line_items") or ():
        if not isinstance(line, Mapping):
            continue
        quantity = int(line.get("quantity") or 0)
        subtotal = _minor(line.get("subtotal"), currency)
        if quantity < 1 or subtotal is None:
            continue
        variation = line.get("variation_id") or 0
        lines.append(
            CartLine(
                product_id=str(line.get("product_id")),
                variant_id=str(variation or line.get("product_id")),
                title=_text(line.get("name")),
                quantity=quantity,
                unit_price=subtotal // quantity,
                currency=currency,
                options={
                    str(m.get("display_key")): _text(m.get("display_value"))
                    for m in line.get("meta_data") or ()
                    if isinstance(m, Mapping) and m.get("display_key")
                },
            )
        )
    if not lines:
        return None
    subtotal = sum(line.unit_price * line.quantity for line in lines)
    return Cart(
        cart_id=f"wc-order-{order.get('id')}",
        lines=lines,
        subtotal=subtotal,
        total=_minor(order.get("total"), currency),
        currency=currency,
    )


def _minor(value: Any, currency: str) -> int | None:
    from carttrace.core.money import to_minor

    return to_minor(value, currency)


def discovery_document(pg: PlaygroundStore, *, transport: httpx.BaseTransport | None = None) -> Any:
    """The playground's UCP discovery document as it serves it now, or None when it serves none."""
    with sync_client(base_url=pg.origin, timeout=TIMEOUT_S, transport=transport) as client:
        response = client.get(pg.ucp_discovery)
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def playground_up(pg: PlaygroundStore, home: Path, *, transport: httpx.BaseTransport | None = None) -> bool:
    """Whether the playground answers its Store API now and its admin env file is present (a checkout row reads the
    order back with it)."""
    if pg.admin is not None and not (home / pg.admin.env_file).is_file():
        return False
    try:
        with sync_client(base_url=pg.origin, timeout=5.0, transport=transport) as client:
            return client.get(f"{pg.store_api}/products", params={"per_page": 1}).status_code == 200
    except httpx.HTTPError:
        return False


def reset_playground(pg: PlaygroundStore, home: Path) -> None:
    """Restore the playground's seeded state with its own script (the stores file's `reset`, run in its directory).
    Raises ConfigError when it names none, and ExternalServiceError when the script fails."""
    import subprocess

    from carttrace.core.errors import ExternalServiceError

    if not pg.reset or not pg.directory:
        raise ConfigError(f"{pg.store_id} names no reset command", hint="add `reset` to data/playgrounds/stores.json")
    done = subprocess.run(pg.reset, cwd=home / pg.directory, capture_output=True, text=True, check=False)  # noqa: S603
    if done.returncode != 0:
        raise ExternalServiceError(pg.store_id, f"reset failed ({done.returncode}): {done.stderr.strip()[-300:]}")


def playground_of(playgrounds: Mapping[str, PlaygroundStore], store_id: str) -> PlaygroundStore:
    found = playgrounds.get(store_id)
    if found is None:
        raise ValidationError(
            f"{store_id} is not a playground in data/playgrounds/stores.json", hint="add it there, or plan it live"
        )
    return found

"""A UCP catalog reply, from a live store or the sandbox, as the normalised products a capture keeps and a fidelity
check compares.

Why one reader in the trace package. A store capture keeps what a live store's search_catalog and get_product returned
(decision D12), its replica is built from that, and the fidelity check replays the same requests against the replica
and compares the answers. If the capture, the replica and the check each read UCP products their own way, a parsing
difference would read as a difference between the store and its replica. The live client and the sandbox sit side by
side on layer 4 and cannot import each other, so the one reader lives here, as the one cart converter does
(`carttrace.trace.carts`).

The shape is UCP 2026-08-25's (product.json, variant.json, catalog_search.json, catalog_lookup.json): a search or
lookup reply lists `products`, a get_product reply holds one `product`, either one possibly wrapped in MCP's
`structuredContent` or a JSON-RPC `result`. Prices are integer minor units with an ISO 4217 code; a price that is not
an integer is read as missing, never converted, because a guessed price would let a replica pass a check it fails.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from carttrace.core.canon import loads
from carttrace.core.errors import ValidationError

PRODUCT_DETAILS = (
    "description",
    "handle",
    "url",
    "categories",
    "list_price_range",
    "media",
    "rating",
    "tags",
    "metadata",
)
"""Product fields beyond id, title, price range, options and variants: what an agent reads to decide, so a replica
must carry each one the store sent."""
VARIANT_DETAILS = (
    "description",
    "sku",
    "barcodes",
    "handle",
    "url",
    "categories",
    "list_price",
    "unit_price",
    "quantity_unit",
    "media",
    "rating",
    "tags",
    "metadata",
    "seller",
)
"""Variant fields beyond id, title, price, availability and options."""
RESPONSE_ONLY = ("selected", "inputs")
"""Fields that describe one response rather than the product: get_product's effective selection, and the lookup
correlation of a variant. A replica derives them per request instead of storing them."""
_WRAPPERS = ("structuredContent", "result")


@dataclass(frozen=True, slots=True)
class OptionValue:
    label: str
    available: bool | None = None
    """get_product's availability signal relative to the effective selection; None where the reply does not say."""
    exists: bool | None = None


@dataclass(frozen=True, slots=True)
class CatalogVariant:
    id: str
    title: str
    price: int | None
    """Minor units of `currency`; None when the reply carried no integer price."""
    currency: str | None
    available: bool | None
    """None when the reply said nothing about availability."""
    status: str | None
    options: tuple[tuple[str, str], ...]
    """(option name, label) in the order the store listed them."""
    details: Mapping[str, Any]
    """VARIANT_DETAILS the reply carried, with their values as sent."""


@dataclass(frozen=True, slots=True)
class CatalogProduct:
    id: str
    title: str
    variants: tuple[CatalogVariant, ...]
    options: tuple[tuple[str, tuple[OptionValue, ...]], ...]
    """Product options in the store's order, each with its values in the store's order."""
    selected: tuple[tuple[str, str], ...] | None
    """get_product's effective selection; None for a search or lookup result."""
    details: Mapping[str, Any]
    """PRODUCT_DETAILS the reply carried, with their values as sent."""

    @property
    def currency(self) -> str | None:
        return next((v.currency for v in self.variants if v.currency), None)


def reply_payload(value: Any) -> Mapping[str, Any] | None:
    """The UCP document a reply carries: the value itself, or what MCP's structuredContent or a JSON-RPC result wraps.
    A JSON text is parsed first. None when no object is found."""
    current = _json(value)
    for _ in range(len(_WRAPPERS) + 1):
        if not isinstance(current, Mapping):
            return None
        if "products" in current or "product" in current or "ucp" in current:
            return current
        inner = next((_json(current[k]) for k in _WRAPPERS if isinstance(_json(current.get(k)), Mapping)), None)
        if inner is None:
            return current
        current = inner
    return current if isinstance(current, Mapping) else None


def product_documents(reply: Any) -> list[dict[str, Any]]:
    """The product objects of a catalog reply, in order: `products` of a search or lookup, or the one `product` of a
    get_product. Entries without a string id are skipped."""
    payload = reply_payload(reply)
    if payload is None:
        return []
    products = payload.get("products")
    if isinstance(products, list):
        return [dict(p) for p in products if isinstance(p, Mapping) and _text(p.get("id"))]
    product = payload.get("product")
    if isinstance(product, Mapping) and _text(product.get("id")):
        return [dict(product)]
    return []


def read_product(doc: Mapping[str, Any]) -> CatalogProduct:
    """One product document, normalised. Raises ValidationError when it has no id."""
    product_id = _text(doc.get("id"))
    if product_id is None:
        raise ValidationError("a catalog product has no id", hint="pass a product object from a UCP catalog reply")
    variants = tuple(read_variant(v) for v in doc.get("variants") or () if isinstance(v, Mapping) and v.get("id"))
    selected = doc.get("selected")
    return CatalogProduct(
        id=product_id,
        title=_text(doc.get("title")) or "",
        variants=variants,
        options=_product_options(doc.get("options")),
        selected=_pairs(selected) if isinstance(selected, list) else None,
        details={k: doc[k] for k in PRODUCT_DETAILS if present(doc.get(k))},
    )


def read_variant(doc: Mapping[str, Any]) -> CatalogVariant:
    """One variant document, normalised."""
    price, currency = price_of(doc.get("price"))
    availability = doc.get("availability")
    available: bool | None = None
    status: str | None = None
    if isinstance(availability, Mapping):
        flag = availability.get("available")
        available = flag if isinstance(flag, bool) else None
        status = _text(availability.get("status"))
    elif isinstance(doc.get("available"), bool):
        available = bool(doc["available"])
    return CatalogVariant(
        id=str(doc.get("id")),
        title=_text(doc.get("title")) or "",
        price=price,
        currency=currency,
        available=available,
        status=status,
        options=_pairs(doc.get("options")),
        details={k: doc[k] for k in VARIANT_DETAILS if present(doc.get(k))},
    )


def price_of(raw: Any) -> tuple[int | None, str | None]:
    """(minor units, currency) of a UCP price object; an integer is minor units without a currency. Anything else is
    (None, currency): a decimal string is never converted here."""
    if isinstance(raw, Mapping):
        amount = raw.get("amount")
        currency = _text(raw.get("currency"))
        return (amount if _is_int(amount) else None), currency
    if _is_int(raw):
        return raw, None
    return None, None


def merge_documents(primary: Mapping[str, Any], *others: Mapping[str, Any]) -> dict[str, Any]:
    """The fullest document of one product from several replies about it: `primary` (a get_product reply) wins for
    every field it holds, and each other reply (search results) adds the fields and the variants it alone carried.
    Variants keep the primary's order, then each other reply's in turn; a variant both name keeps the primary's
    fields and gains the others'."""
    merged: dict[str, Any] = dict(primary)
    variants: dict[str, dict[str, Any]] = {}
    for doc in (primary, *others):
        for key, value in doc.items():
            if key != "variants" and key not in merged:
                merged[key] = value
        for raw in doc.get("variants") or ():
            if not isinstance(raw, Mapping) or not _text(raw.get("id")):
                continue
            current = variants.setdefault(str(raw["id"]), {})
            for key, value in raw.items():
                current.setdefault(key, value)
    merged["variants"] = list(variants.values())
    return merged


def present(value: Any) -> bool:
    """A field counts as sent when it holds something: not None, not empty text, not an empty list or object."""
    return value is not None and value not in ("", [], {})


def _product_options(raw: Any) -> tuple[tuple[str, tuple[OptionValue, ...]], ...]:
    out: list[tuple[str, tuple[OptionValue, ...]]] = []
    for item in raw if isinstance(raw, list) else ():
        if not isinstance(item, Mapping) or not _text(item.get("name")):
            continue
        values: list[OptionValue] = []
        for value in item.get("values") or ():
            label = _text(value.get("label")) if isinstance(value, Mapping) else _text(value)
            if label is None:
                continue
            signals = value if isinstance(value, Mapping) else {}
            values.append(OptionValue(label, _flag(signals.get("available")), _flag(signals.get("exists"))))
        out.append((str(item["name"]), tuple(values)))
    return tuple(out)


def _pairs(raw: Any) -> tuple[tuple[str, str], ...]:
    """[{name, label}] (UCP selected options) as (name, label) pairs in order."""
    out: list[tuple[str, str]] = []
    for item in raw if isinstance(raw, list) else ():
        if isinstance(item, Mapping):
            name, label = _text(item.get("name")), _text(item.get("label")) or _text(item.get("value"))
            if name and label:
                out.append((name, label))
    return tuple(out)


def option_pairs(raw: Sequence[Any]) -> list[dict[str, str]]:
    """(name, label) pairs as UCP selected options."""
    return [{"name": name, "label": label} for name, label in raw]


def _flag(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _json(value: Any) -> Any:
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return loads(value)
        except ValueError:
            return value
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None

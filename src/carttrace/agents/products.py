"""One view of a product, whatever shape it arrived in: a UCP get_product or search_catalog reply, a sandbox
snapshot, or a Shopify products.json feed.

Why normalise. Task generation reads fixture products, the reference agent reads what a surface returned, and both
must see the same variants, prices and text, or a task would be written around a fact the reference agent cannot
find. Prices become integer minor units (carttrace.core.money): UCP and the sandbox already send integers, a feed sends
decimal strings. Description HTML is read once into lines and tables (carttrace.agents.text), with metadata and metafields
added as "key: value" lines, because stores put size charts and ingredient lists in either place.
"""

from __future__ import annotations

import json
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from carttrace.agents.memo import table
from carttrace.agents.text import Table, fold, lines_of, looks_html, pipe_tables, read_html
from carttrace.core.canon import loads
from carttrace.core.errors import ValidationError
from carttrace.core.money import to_minor
from carttrace.sandbox.catalogs.search_memory import SOURCE_KEY as SEARCH_MEMORY_KEY

Grid = tuple[tuple[str, ...], ...]
_WRAPPERS = ("structuredContent", "result", "product")
_MAX_META_DEPTH = 4


@lru_cache(maxsize=1 << 12)
def _folded(names: tuple[str, ...]) -> frozenset[str]:
    return frozenset(fold(n) for n in names)


@dataclass(frozen=True, slots=True)
class Variant:
    id: str
    product_id: str
    title: str
    price: int | None
    """Unit price in minor units of `currency`; None when the store gave no usable price."""
    currency: str | None
    available: bool
    """False only when the store said so; a variant with no availability field counts as available."""
    options: tuple[tuple[str, str], ...] = ()
    """(option name as the store names it, value), in the store's order."""
    stock: int | None = None
    """Units the store holds, when it says (a sandbox world's `stock`, or `availability.quantity`); None when
    unknown, which never limits a task's quantity."""

    def holds_units(self, quantity: int) -> bool:
        """Whether the store holds `quantity` units of the variant, as far as it says."""
        return self.stock is None or self.stock >= quantity

    def option(self, *names: str) -> str | None:
        """The value of the first option whose folded name is one of `names` (folded: lowercase, plain spaces)."""
        wanted = _folded(names)
        for name, value in self.options:
            if fold(name) in wanted:
                return value
        return None

    def label(self) -> str:
        """How a person would name the variant: its option values, or its title."""
        values = [v for _, v in self.options if fold(v) != "default title"]
        return " / ".join(values) or self.title


@dataclass(frozen=True, slots=True)
class Product:
    id: str
    title: str
    brand: str | None = None
    product_type: str | None = None
    categories: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    lines: tuple[str, ...] = ()
    """Description and metadata as text lines: what a text-only agent can read about the product."""
    tables: tuple[Grid, ...] = ()
    """Tables found in the description or metadata (HTML tables and pipe-separated text)."""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Metadata and metafields as the store sent them, for details stored as structured values."""
    variants: tuple[Variant, ...] = ()

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def variant(self, variant_id: str) -> Variant | None:
        return next((v for v in self.variants if v.id == variant_id), None)


def read_product(raw: Mapping[str, Any], *, currency: str | None = None) -> Product:
    """The product in `raw`, which may be the product itself or a reply wrapping it under "product", "result" or
    MCP's "structuredContent". `currency` prices variants whose price names none. Raises ValidationError when there
    is no product id."""
    data = _unwrap(raw)
    product_id = _text(data.get("id"))
    if product_id is None:
        raise ValidationError("product has no id", hint="pass a get_product reply or a product object with an id")
    currency = currency or _range_currency(data)
    lines, tables = _description(data.get("description"), data.get("body_html"))
    metadata = _metadata(data)
    meta_lines, meta_tables = _metadata_text(metadata)
    tags = _tags(data.get("tags"))
    option_names = _option_names(data.get("options"))
    return Product(
        id=product_id,
        title=_text(data.get("title")) or _text(data.get("name")) or "",
        brand=_brand(data),
        product_type=_text(data.get("product_type")),
        categories=_categories(data.get("categories")),
        tags=tags,
        lines=tuple(lines + meta_lines),
        tables=tuple(_grid(t) for t in tables + meta_tables),
        metadata=metadata,
        variants=tuple(_variants(data.get("variants"), product_id, currency, option_names)),
    )


def read_products(raws: Iterable[Mapping[str, Any]], *, currency: str | None = None) -> list[Product]:
    """Every product, sorted by id so the same fixtures in any order give the same tasks."""
    memo = table("read_product")
    products = [
        read_product(raw, currency=currency) if memo is None else _remembered(raw, currency, memo) for raw in raws
    ]
    ids = [p.id for p in products]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise ValidationError(f"fixture products repeat ids {repeated[:3]}", hint="pass each product once")
    return sorted(products, key=lambda p: p.id)


def _remembered(raw: Mapping[str, Any], currency: str | None, memo: dict[Hashable, Any]) -> Product:
    """`read_product` of `raw`, once per content within a memo run (carttrace.agents.memo): a store's task draws read the
    same fixtures for every persona and draw. The key is the product's whole JSON text, so changed content is read
    anew; content JSON cannot hold is read every time."""
    try:
        key = (currency, json.dumps(raw, sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError):
        return read_product(raw, currency=currency)
    found = memo.get(key)
    if found is None:
        found = memo[key] = read_product(raw, currency=currency)
    product: Product = found
    return product


def products_in_reply(reply: Any) -> list[Mapping[str, Any]]:
    """The product objects in a search_catalog or lookup_catalog reply (under "products"), or the one product of a
    get_product reply. Anything else gives an empty list."""
    data = _maybe_json(reply)
    if not isinstance(data, Mapping):
        return []
    for key in ("structuredContent", "result"):
        inner = _maybe_json(data.get(key))
        if isinstance(inner, Mapping):
            data = inner
    products = data.get("products")
    if isinstance(products, list):
        return [p for p in products if isinstance(p, Mapping) and _text(p.get("id"))]
    product = data.get("product")
    if isinstance(product, Mapping) and _text(product.get("id")):
        return [product]
    return [data] if _text(data.get("id")) and "variants" in data else []


def fixtures_from_snapshot(snapshot: Mapping[str, Any] | BaseModel) -> list[dict[str, Any]]:
    """Product dicts from a sandbox store snapshot, in UCP's shape: its `products`, each with its variants attached
    from `variants` when the product does not list them itself. Takes an envpack snapshot (variants keyed by id) or
    a catalog StoreSnapshot (variants as a list, each with its id), as a model or as JSON. A product a replica holds
    only because a live search listed it (its search memory, D39) is left out: no task is written around it."""
    data: Mapping[str, Any] = (
        snapshot.model_dump(mode="json", by_alias=True) if isinstance(snapshot, BaseModel) else snapshot
    )
    currency = _text(data.get("currency"))
    products = data.get("products") or {}
    variants = data.get("variants") or {}
    if isinstance(variants, list):
        variants = {str(v["id"]): v for v in variants if isinstance(v, Mapping) and _text(v.get("id"))}
    if not isinstance(products, Mapping) or not isinstance(variants, Mapping):
        raise ValidationError("snapshot products must be keyed by id, and variants keyed by id or listed with ids")
    by_product: dict[str, list[dict[str, Any]]] = {}
    for variant_id in sorted(variants):
        raw = variants[variant_id]
        if not isinstance(raw, Mapping):
            continue
        product_id = _text(raw.get("product_id")) or str(variant_id)
        by_product.setdefault(product_id, []).append(_ucp_variant(str(variant_id), raw, currency))
    out = []
    for product_id in sorted(set(products) | set(by_product)):
        raw_product = products.get(product_id)
        if isinstance(raw_product, Mapping) and raw_product.get(SEARCH_MEMORY_KEY):
            continue  # held only so replayed live searches can list it (D39), never a task's product
        product = dict(raw_product) if isinstance(raw_product, Mapping) else {}
        product.setdefault("id", product_id)
        product.setdefault("title", product_id)
        if not product.get("variants"):
            product["variants"] = by_product.get(product_id, [])
        out.append(product)
    return out


def _ucp_variant(variant_id: str, raw: Mapping[str, Any], currency: str | None) -> dict[str, Any]:
    options = raw.get("options")
    pairs = options.items() if isinstance(options, Mapping) else []
    price = raw.get("price")
    return {
        "id": variant_id,
        "title": _text(raw.get("title")) or variant_id,
        "price": {"amount": price, "currency": currency} if isinstance(price, int) else price,
        "availability": {"available": bool(raw.get("available", True))},
        "options": [{"name": str(k), "label": str(v)} for k, v in pairs],
        **({"stock": raw["stock"]} if isinstance(raw.get("stock"), int) and not isinstance(raw["stock"], bool) else {}),
    }


# -- shapes ----------------------------------------------------------------------------------------------------------


def _unwrap(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    data: Any = raw
    for _ in range(len(_WRAPPERS) + 1):
        data = _maybe_json(data)
        if not isinstance(data, Mapping):
            break
        if "variants" in data or ("id" in data and "title" in data):
            return data
        inner = next((data[k] for k in _WRAPPERS if isinstance(_maybe_json(data.get(k)), Mapping)), None)
        if inner is None:
            return data
        data = inner
    if not isinstance(data, Mapping):
        raise ValidationError("product reply holds no product object")
    return data


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return loads(value)
        except ValueError:
            return value
    return value


def _description(description: Any, body_html: Any) -> tuple[list[str], list[Table]]:
    lines: list[str] = []
    tables: list[Table] = []
    sources: list[tuple[str, str]] = []
    if isinstance(description, Mapping):
        for key in ("html", "markdown", "plain"):
            value = _text(description.get(key))
            if value:
                sources.append((key, value))
                break  # the richest form the store sent; the others repeat it
    elif isinstance(description, str) and description.strip():
        sources.append(("html" if looks_html(description) else "plain", description))
    if isinstance(body_html, str) and body_html.strip():
        sources.append(("html", body_html))
    for kind, value in sources:
        if kind == "html":
            text_lines, html_tables = read_html(value)
            lines += text_lines
            tables += html_tables or pipe_tables(text_lines)
        else:
            text_lines = lines_of(value)
            lines += text_lines
            tables += pipe_tables(text_lines)
    return lines, tables


def _metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    meta = data.get("metadata")
    if isinstance(meta, Mapping):
        out.update({str(k): v for k, v in meta.items()})
    fields = data.get("metafields")
    if isinstance(fields, Mapping):
        out.update({str(k): v for k, v in fields.items()})
    elif isinstance(fields, list):
        for entry in fields:
            if isinstance(entry, Mapping) and _text(entry.get("key")):
                out[str(entry["key"])] = entry.get("value")
    return out


def _metadata_text(metadata: Mapping[str, Any]) -> tuple[list[str], list[Table]]:
    lines: list[str] = []
    tables: list[Table] = []
    for key in sorted(metadata):
        _flatten(key, metadata[key], lines, tables, 0)
    return lines, tables


def _flatten(key: str, value: Any, lines: list[str], tables: list[Table], depth: int) -> None:
    label = key.replace("_", " ")
    if depth > _MAX_META_DEPTH:
        return
    if isinstance(value, str):
        if looks_html(value):
            text_lines, html_tables = read_html(value)
            lines.extend(f"{label}: {line}" for line in text_lines[:1])
            lines.extend(text_lines[1:])
            tables.extend(html_tables)
        else:
            text_lines = lines_of(value)
            if text_lines:
                lines.append(f"{label}: {text_lines[0]}")
                lines.extend(text_lines[1:])
                tables.extend(pipe_tables(text_lines))
    elif isinstance(value, bool | int | float):
        lines.append(f"{label}: {value}")
    elif isinstance(value, Mapping):
        for sub in sorted(value):
            _flatten(f"{key} {sub}", value[sub], lines, tables, depth + 1)
    elif isinstance(value, list) and all(isinstance(v, str | int | float) for v in value):
        if value:
            lines.append(f"{label}: {', '.join(str(v) for v in value)}")
    elif isinstance(value, list):
        for item in value:
            _flatten(key, item, lines, tables, depth + 1)


def _tags(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return tuple(t.strip() for t in raw.split(",") if t.strip())
    if isinstance(raw, list):
        return tuple(str(t).strip() for t in raw if isinstance(t, str | int) and str(t).strip())
    return ()


def _brand(data: Mapping[str, Any]) -> str | None:
    for key in ("brand", "vendor"):
        value = data.get(key)
        if isinstance(value, Mapping):
            value = value.get("name")
        text = _text(value)
        if text:
            return text
    return None


def _categories(raw: Any) -> tuple[str, ...]:
    out = []
    for item in raw if isinstance(raw, list) else []:
        value = item.get("value") if isinstance(item, Mapping) else item
        text = _text(value)
        if text:
            out.append(text)
    return tuple(out)


def _option_names(raw: Any) -> list[str]:
    names = []
    for item in raw if isinstance(raw, list) else []:
        name = _text(item.get("name")) if isinstance(item, Mapping) else _text(item)
        names.append(name or "")
    return names


def _range_currency(data: Mapping[str, Any]) -> str | None:
    price_range = data.get("price_range")
    if isinstance(price_range, Mapping):
        low = price_range.get("min")
        if isinstance(low, Mapping):
            return _text(low.get("currency"))
    return None


def _variants(raw: Any, product_id: str, currency: str | None, option_names: Sequence[str]) -> list[Variant]:
    items: list[tuple[str | None, Any]]
    if isinstance(raw, Mapping):
        items = [(str(k), raw[k]) for k in sorted(raw)]
    elif isinstance(raw, list):
        items = [(None, v) for v in raw]
    else:
        items = []
    out = []
    for key, value in items:
        if not isinstance(value, Mapping):
            continue
        variant_id = _text(value.get("id")) or key
        if variant_id is None:
            continue
        price, price_currency = _price(value.get("price"), currency)
        out.append(
            Variant(
                id=variant_id,
                product_id=product_id,
                title=_text(value.get("title")) or "",
                price=price,
                currency=price_currency,
                available=_available(value),
                options=_variant_options(value, option_names),
                stock=_stock(value),
            )
        )
    return out


def _price(raw: Any, currency: str | None) -> tuple[int | None, str | None]:
    """A price in minor units and its ISO 4217 code, upper case: Medusa's store API writes "usd" (D104: 2 of 45
    reference rows on an open stand-in under the medusa pack refused a $49.99 product against a $200 USD budget)."""
    currency = currency.upper() if currency else currency
    if isinstance(raw, Mapping):
        amount = raw.get("amount")
        given = _text(raw.get("currency"))
        code = given.upper() if given else currency
        if isinstance(amount, int) and not isinstance(amount, bool):
            return amount, code
        return to_minor(amount, code), code
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw, currency  # UCP and the sandbox send integer minor units
    if isinstance(raw, str | float):
        return to_minor(raw, currency), currency  # a feed's decimal price
    return None, currency


def _available(value: Mapping[str, Any]) -> bool:
    availability = value.get("availability")
    if isinstance(availability, Mapping) and isinstance(availability.get("available"), bool):
        return bool(availability["available"])
    flag = value.get("available")
    return flag if isinstance(flag, bool) else True


def _stock(value: Mapping[str, Any]) -> int | None:
    availability = value.get("availability")
    for found in (value.get("stock"), availability.get("quantity") if isinstance(availability, Mapping) else None):
        if isinstance(found, int) and not isinstance(found, bool) and found >= 0:
            return found
    return None


def _variant_options(value: Mapping[str, Any], option_names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    options = value.get("options") or value.get("selected_options")
    pairs: list[tuple[str, str]] = []
    if isinstance(options, list):
        for item in options:
            if isinstance(item, Mapping):
                name = _text(item.get("name"))
                label = _text(item.get("label")) or _text(item.get("value"))
                if name and label:
                    pairs.append((name, label))
    elif isinstance(options, Mapping):
        pairs = [(str(k), str(v)) for k, v in options.items() if _text(v)]
    if not pairs:
        for index in range(3):
            label = _text(value.get(f"option{index + 1}"))
            if label:
                name = option_names[index] if index < len(option_names) and option_names[index] else f"Option {index}"
                pairs.append((name, label))
    return tuple(pairs)


def _grid(table: Table) -> Grid:
    return tuple(tuple(row) for row in table)


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None

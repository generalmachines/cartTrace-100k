"""A store's product page as the product an agent can choose from: its variants, prices, availability and policies
links, read from what the page serves (decision D31).

Why here and why these sources. A store without an agent protocol is shopped through its own website, and an agent
there sees what a person sees, plus what the page's markup says about it. Pages say it in a few well-known ways, in
this order of trust:
- **the theme's own variant data**: Shopify themes render the variant picker from embedded JSON (`application/json`
  scripts with `variants`, each with its id, options, integer price in minor units and `available`); Magento's
  configurable products from `jsonConfig` in a `text/x-magento-init` script (attributes, their options, the child
  product ids each option leads to, and each child's final price). These carry the store's own variant ids, the ids
  its catalog and a capture of it use;
- **schema.org JSON-LD and microdata**: a Product (or a ProductGroup with `hasVariant`), its offers with price,
  currency, availability, sku and the offer's URL (Shopify's `?variant=` names the variant id);
- **OpenGraph**: `product:price:amount`, `product:price:currency`, `product:availability`, `og:title`.

Since D36 the jsonConfig's `salable` map gives each child's stock and its `sku` map each child's sku, Magento 1's
`spConfig` is read like it, a page's one theme variant meets the JSON-LD product at the page's address (a Magento
form's id is not its sku), and a category grid (`listed_products`) is never a product. carttrace.pages.product_page fills
what is still missing from the page's other sources and says where each fact came from.

A variant's availability is True only when the page says it is in stock (InStock, LimitedAvailability, OnlineOnly,
or the theme's `available`); a subscription offer ("delivery every month") is never the price of a one-time purchase.
Nothing here guesses a price or an id: a variant whose price the page does not show has none, and the selection that
needs one says so.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit

from carttrace.core.money import to_minor
from carttrace.pages.html import PageFacts, schema_name
from carttrace.pages.site_index import identity_query
from carttrace.trace.capture_v2 import available_of as normal_available

DESCRIPTION_CHARS = 12_000
MAX_VARIANTS = 300
POLICY_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("returns", ("return", "refund", "exchange")),
    ("shipping", ("shipping", "delivery")),
    ("size_guide", ("size guide", "size chart", "sizing", "size-guide", "size-chart")),
    ("terms", ("terms",)),
    ("privacy", ("privacy",)),
    ("faq", ("faq", "help", "frequently asked")),
    ("contact", ("contact",)),
    ("warranty", ("warranty", "guarantee")),
)
"""Policy kinds and the words a link's text or address carries for each."""
_SUBSCRIPTION = re.compile(r"subscri|deliver(?:y|ed) every|auto-?ship|every \d+ (?:week|month)", re.IGNORECASE)
_VARIANT_IN_URL = re.compile(r"[?&]variant=(\d{1,20})")
_VARIANT_IN_ID = re.compile(r"#variant-(\d{1,20})")
_THOUSANDS = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_DECIMAL_COMMA = re.compile(r"^\d+,\d{2}$")
_PRICE_TEXT = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SCHEMA_OPTIONS = ("color", "size", "material", "pattern", "flavor", "scent", "style")
_NO_FACTS = PageFacts(url="")
_LISTING_SWATCH = re.compile(r"swatch-option-\d+")
MIN_GRID_FORMS = 3
"""Add-to-cart forms for distinct products that make a page without schema.org products a grid (a product page
has its own form and at most a bundle's second)."""
MAX_COMBINATIONS = 20_000
"""Option combinations a Magento 1 spConfig is searched through for its children."""


@dataclass(frozen=True, slots=True)
class PolicyLink:
    kind: str
    text: str
    url: str


def product_url(url: str) -> str:
    """A product page's address without its fragment and with only the query parameters that name the page
    (carttrace.pages.site_index.identity_query: Gap's `?pid=`, never Shopify's `?variant=`): the id the product is known by in
    a session (D58)."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, identity_query(parts.query), ""))


def page_product(facts: PageFacts, *, currency: str | None = None) -> dict[str, Any] | None:
    """The product a page shows, in UCP's product shape (`id`, `title`, `url`, `brand`, `description`, `options`,
    `variants` with `id`, `title`, `sku`, `gtin`, `url`, `price`, `availability`, `options`), with `sources` naming
    the sources read and each price and availability's `from` naming the one that showed it (theme_json,
    magento_config, magento_form, jsonld, microdata, opengraph); None when the page shows no product (no schema.org
    product, no theme variants and no OpenGraph price)."""
    nodes = [*facts.products("jsonld"), *facts.products("microdata")]
    if listed_products(facts):
        return None
    theme = theme_variants(facts)
    meta = dict(facts.meta)
    og_price = meta.get("product:price:amount") or meta.get("og:price:amount")
    if not nodes and not theme and og_price is None:
        return None
    page_currency = (
        currency_of(nodes) or meta.get("product:price:currency") or meta.get("og:price:currency") or currency
    )
    canonical = product_url(facts.canonical or facts.url)
    sources: list[str] = []
    variants: list[dict[str, Any]] = []
    if theme:
        sources.append(theme[0])
        variants = _priced(theme[1], page_currency, theme[0])
    schema = schema_variants(facts.products("jsonld"), page_currency, "jsonld")
    schema_source = "jsonld"
    if not schema:
        schema = schema_variants(facts.products("microdata"), page_currency, "microdata")
        schema_source = "microdata"
    if schema:
        sources.append(schema_source)
        variants = _enriched(variants, schema, canonical) if variants else schema
    if not variants and og_price is not None:
        sources.append("opengraph")
        variants = [_og_variant(facts, meta, page_currency)]
    if og_price is not None and "opengraph" not in sources:
        sources.append("opengraph")
    variants = _product_availability(variants, nodes, "jsonld" if facts.products("jsonld") else "microdata")
    main = _main_node(nodes)
    if main is not None and len(nodes) > 1:
        main = next((n for n in nodes if _is_page(_node_url(n), canonical)), main)
    title = _text(main.get("name")) if main else None
    out: dict[str, Any] = {
        "id": canonical,
        "url": canonical,
        "title": title or meta.get("og:title") or facts.title,
        "brand": _brand(main),
        "description": {"plain": _description(main, facts)},
        "options": _options(variants, facts),
        "variants": variants[:MAX_VARIANTS],
        "sources": sources,
    }
    if page_currency:
        out["currency"] = page_currency
    return out


def endpoint_product(document: Any, url: str, *, currency: str | None = None, source: str) -> dict[str, Any] | None:
    """The product a platform's public product endpoint describes (Shopify's `/products/<handle>.js`, whose prices
    are integers in minor units and whose variants say `available`; or `.json`, whose prices are decimal text), in
    page_product's shape with `source` as each fact's `from`; None when it describes no variant."""
    if isinstance(document, Mapping) and isinstance(document.get("product"), Mapping):
        document = document["product"]
    if not isinstance(document, Mapping):
        return None
    raws = _shopify_variants(document)
    if not raws:
        return None
    variants = _priced(raws, currency, source)
    canonical = product_url(url)
    description = re.sub(r"<[^>]+>", " ", str(document.get("description") or document.get("body_html") or ""))
    out: dict[str, Any] = {
        "id": canonical,
        "url": canonical,
        "title": _text(document.get("title")) or canonical,
        "brand": _text(document.get("vendor")),
        "description": {"plain": " ".join(description.split())[:DESCRIPTION_CHARS]},
        "options": _options(variants, _NO_FACTS),
        "variants": variants[:MAX_VARIANTS],
        "sources": [source],
    }
    if currency:
        out["currency"] = currency
    return out


def _product_availability(
    variants: list[dict[str, Any]], nodes: Sequence[Mapping[str, Any]], source: str
) -> list[dict[str, Any]]:
    """Variants whose own availability the page does not show take the product's, when every one-time offer on the
    page agrees (status `product_in_stock` or `product_out_of_stock`, so a reader can tell it from a variant's own)."""
    said = {available_of(o.get("availability")) for n in nodes for o in _offers(n) if not _subscription(o)}
    said.discard(None)
    if len(said) != 1:
        return variants
    value = said.pop()
    status = "product_in_stock" if value else "product_out_of_stock"
    return [
        {**v, "availability": {"available": value, "status": status, "from": source}}
        if v["availability"].get("available") is None
        else v
        for v in variants
    ]


def policy_links(facts: PageFacts) -> list[PolicyLink]:
    """The page's links to the store's policies (returns, shipping, size guide, terms...), each address once."""
    found: dict[str, PolicyLink] = {}
    for link in facts.links:
        words = f"{link.text} {urlsplit(link.href).path}".lower()
        for kind, cues in POLICY_WORDS:
            if any(cue in words for cue in cues) and link.href not in found:
                found[link.href] = PolicyLink(kind, link.text, link.href)
                break
    return list(found.values())


def listed_products(facts: PageFacts) -> list[dict[str, Any]]:
    """The products a category or search page lists, when the page is one rather than a product's own: two or more
    schema.org products at other addresses and none at this page's ({"url", "title", "price", "currency"} each, as
    listed), or, on a page with no schema.org product, MIN_GRID_FORMS or more Magento add-to-cart forms for distinct
    products ({"product_id"} each); [] on a product page. The 2026-10-13 smoke read Store 167's eucalyptus
    collection, a grid, as one product whose variants were one tile's sizes.

    A product group at this page whose products are its own variants is a product, not a grid (D70): one store's
    mens-t-shirt-v2-0 names itself as a ProductGroup and each size and colour as a Product at a sub-page of its
    own (/s-black, /m-black), and judge-v11 voided the task as naming a listing."""
    page = product_url(facts.canonical or facts.url)
    every = [*facts.products("jsonld"), *facts.products("microdata")]
    nodes = [n for n in every if "ProductGroup" not in n["@type"]]
    if variants_of_group_here(every, page):
        return []
    elsewhere: dict[str, dict[str, Any]] = {}
    for node in nodes:
        url = _node_url(node)
        if not url:
            continue
        if _is_page(url, page):
            return []
        offer = _one_time_offer(node)
        code = (_text(offer.get("priceCurrency")) if offer else None) or None
        amount = price_minor(offer.get("price"), code) if offer else None
        entry = {"url": product_url(url), "title": _text(node.get("name")), "price": amount, "currency": code}
        elsewhere.setdefault(product_url(url), entry)
    if len(elsewhere) >= 2:
        return list(elsewhere.values())
    if nodes:
        return []
    tiles = dict.fromkeys(
        value
        for form in facts.forms
        if "checkout/cart/add" in form.action
        for value in (next((i.value for i in form.inputs if i.name == "product" and i.value.isdigit()), None),)
        if value is not None
    )
    return [{"product_id": tile} for tile in tiles] if len(tiles) >= MIN_GRID_FORMS else []


def variants_of_group_here(nodes: Sequence[Mapping[str, Any]], page: str) -> bool:
    """Whether a schema.org ProductGroup names this page as its own `url` and every Product beside it is one of its
    variants: listed in its `hasVariant`, naming its `productGroupID` in `inProductGroupWithID`, or served at a sub-page
    of this page. A group page that also shows a shelf of other products is not one (Store 103's mascara names its
    shelf at other addresses, which only its feed listing settles, D38)."""
    groups = [n for n in nodes if "ProductGroup" in n["@type"] and _is_page(_text(n.get("url")) or "", page)]
    if not groups:
        return False
    base = urlsplit(page).path.rstrip("/")
    named = {id(v) for g in groups for v in _list(g.get("hasVariant")) if isinstance(v, Mapping)}
    group_ids = {_text(g.get("productGroupID")) for g in groups} - {None, ""}
    members = [n for n in nodes if "ProductGroup" not in n["@type"]]
    if not members:
        return False
    for node in members:
        path = urlsplit(_node_url(node)).path.rstrip("/")
        under = bool(base) and path.startswith(base + "/")
        if not (id(node) in named or _text(node.get("inProductGroupWithID")) in group_ids or under or path == base):
            return False
    return True


def _node_url(node: Mapping[str, Any]) -> str:
    offer = _one_time_offer(node)
    return (_text(offer.get("url")) if offer else None) or _text(node.get("url")) or ""


def _is_page(url: str, page: str) -> bool:
    """Whether an address is this page's: the same path, or the same last segment (Magento names a product at
    /travel-test/eucalyptus-deodorant-2-6oz-e73g in its JSON-LD and serves it at /eucalyptus-deodorant-2-6oz-e73g)."""
    if not url:
        return False
    one, other = urlsplit(url).path.rstrip("/"), urlsplit(page).path.rstrip("/")
    return bool(one) and (one == other or one.rsplit("/", 1)[-1] == other.rsplit("/", 1)[-1])


def currency_of(nodes: Sequence[Mapping[str, Any]]) -> str | None:
    for node in nodes:
        for offer in _offers(node):
            code = _text(offer.get("priceCurrency"))
            if code and len(code) == 3:
                return code.upper()
    return None


def available_of(value: Any) -> bool | None:
    """schema.org availability as in stock now (True), not (False: sold out, in store only, or on pre-order or back
    order, which a shopper cannot take today), or not said (None). Read by the one normaliser of availability words
    (carttrace.trace.capture_v2.available_of), so "In Stock", "instock" and "https://schema.org/OnlineOnly" read alike."""
    return normal_available(value, orderable=False)


def price_minor(value: Any, currency: str | None) -> int | None:
    """A decimal price as the page shows it ("54.95", 54.95, "1,299.00", "$12.50") in minor units; None otherwise."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return to_minor(value, currency)
    found = _PRICE_TEXT.search(str(value))
    if found is None:
        return None
    number = found.group()
    if _THOUSANDS.match(number):
        number = number.replace(",", "")
    elif _DECIMAL_COMMA.match(number):
        number = number.replace(",", ".")
    elif "," in number:
        return None
    return to_minor(number, currency)


# -- the theme's own variant data -------------------------------------------------------------------------------------


def theme_variants(facts: PageFacts) -> tuple[str, list[dict[str, Any]]] | None:
    """Variants from the theme's embedded JSON: Shopify's product or variants JSON, else Magento's jsonConfig (the
    product page's own, never a listing tile's), else Magento 1's spConfig, else a WooCommerce variations form,
    else a Magento simple product's form."""
    for raw in facts.data_scripts:
        value = _json(raw)
        if value is None:
            continue
        shopify = _shopify_variants(value)
        if shopify:
            return "theme_json", shopify
    for raw in facts.data_scripts:
        value = _json(raw)
        for config in _magento_configs(value):
            magento = _magento_variants(config)
            if magento:
                return "magento_config", magento
    for name, raw in facts.state_scripts:
        value = _json(raw) if name == "spConfig" else None
        if isinstance(value, Mapping):
            legacy = _spconfig_variants(value)
            if legacy:
                return "magento_spconfig", legacy
    for raw in facts.data_scripts:
        woo = _woo_variants(_json(raw))
        if woo:
            return "woo_variations", woo
    simple = _magento_simple(facts)
    return ("magento_form", simple) if simple else None


def _woo_variants(value: Any) -> list[dict[str, Any]]:
    """A WooCommerce variable product's variations from its form's `data-product_variations`: each with its
    `variation_id`, its `attributes` (`attribute_color`, `attribute_pa_shoe-size`: an empty value means any), its
    `display_price` in major units and whether it is in stock and purchasable."""
    if not isinstance(value, list) or not value:
        return []
    if not all(isinstance(v, Mapping) and isinstance(v.get("variation_id"), int) for v in value):
        return []
    out = []
    for raw in value:
        attributes = raw.get("attributes") if isinstance(raw.get("attributes"), Mapping) else {}
        options = [
            {"name": _woo_attribute(str(key)), "label": str(label)}
            for key, label in attributes.items()
            if isinstance(label, str) and label.strip()
        ]
        stock = raw.get("is_in_stock")
        purchasable = raw.get("is_purchasable")
        available = stock if isinstance(stock, bool) else None
        if available is True and purchasable is False:
            available = False
        out.append(
            {
                "id": str(raw["variation_id"]),
                "title": " / ".join(o["label"] for o in options),
                "sku": _text(raw.get("sku")),
                "gtin": None,
                "_price": raw.get("display_price"),
                "_minor": False,
                "available": available,
                "options": options,
            }
        )
    return out


def _woo_attribute(key: str) -> str:
    """A WooCommerce attribute field's name as a shopper reads it: `attribute_pa_shoe-size` is "Shoe size"."""
    name = key.removeprefix("attribute_").removeprefix("pa_").replace("-", " ").replace("_", " ").strip()
    return name[:1].upper() + name[1:] if name else key


def _magento_configs(value: Any) -> Iterator[Mapping[str, Any]]:
    """Every jsonConfig in a text/x-magento-init value whose holder is not a listing tile's swatches
    (`[data-role=swatch-option-762]` is one product of a category grid; the page's own is `swatch-options`)."""
    if not isinstance(value, Mapping):
        return
    for selector, holder in value.items():
        if _LISTING_SWATCH.search(str(selector)):
            continue
        config = _find_key(holder, "jsonConfig")
        if isinstance(config, str):
            config = _json(config)
        if isinstance(config, Mapping):
            yield config


def _shopify_variants(value: Any) -> list[dict[str, Any]]:
    """A Shopify theme's variants: a product object with `variants` (and option names), or a bare variants list."""
    for candidate in _walk(value):
        listed = candidate.get("variants") if isinstance(candidate, Mapping) else None
        items = listed if isinstance(listed, list) else candidate if isinstance(candidate, list) else None
        if not isinstance(items, list) or not items or not all(_shopify_like(v) for v in items):
            continue
        names = _option_names(candidate.get("options")) if isinstance(candidate, Mapping) else []
        return [_shopify_variant(v, names) for v in items]
    return []


def _shopify_like(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("id"), int | str)
        and "price" in value
        and ("option1" in value or isinstance(value.get("options"), list))
    )


def _shopify_variant(raw: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    values = raw.get("options")
    labels = [str(v) for v in values] if isinstance(values, list) else []
    if not labels:
        labels = [str(raw[k]) for k in ("option1", "option2", "option3") if raw.get(k) not in (None, "")]
    options = [
        {"name": names[i] if i < len(names) and names[i] else f"Option {i + 1}", "label": label}
        for i, label in enumerate(labels)
        if label.lower() != "default title"
    ]
    price = raw.get("price")
    return {
        "id": str(raw["id"]),
        "title": _text(raw.get("title")) or " / ".join(labels),
        "sku": _text(raw.get("sku")),
        "gtin": _text(raw.get("barcode")),
        "_price": price,
        "_minor": isinstance(price, int) and not isinstance(price, bool),
        "available": raw.get("available") if isinstance(raw.get("available"), bool) else None,
        "options": options,
    }


def _magento_variants(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A Magento configurable product's children from jsonConfig: `index` (child id to attribute id to option id),
    `attributes` (their labels and the options' labels) and `optionPrices` (each child's final price)."""
    index = config.get("index")
    attributes = config.get("attributes")
    if not isinstance(index, Mapping) or not isinstance(attributes, Mapping):
        return []
    labels: dict[tuple[str, str], tuple[str, str]] = {}
    for attribute_id, attribute in attributes.items():
        if not isinstance(attribute, Mapping):
            continue
        name = _text(attribute.get("label")) or _text(attribute.get("code")) or str(attribute_id)
        for option in attribute.get("options") or ():
            if isinstance(option, Mapping) and option.get("id") is not None:
                labels[(str(attribute_id), str(option["id"]))] = (name, _text(option.get("label")) or "")
    prices = config.get("optionPrices") if isinstance(config.get("optionPrices"), Mapping) else {}
    skus = config.get("sku") if isinstance(config.get("sku"), Mapping) else {}
    salable = _salable(config.get("salable"))
    out: list[dict[str, Any]] = []
    for child_id in sorted(index, key=str):
        chosen = index[child_id]
        if not isinstance(chosen, Mapping):
            continue
        options = [
            {"name": labels[(str(a), str(o))][0], "label": labels[(str(a), str(o))][1]}
            for a, o in chosen.items()
            if (str(a), str(o)) in labels
        ]
        price_block = prices.get(str(child_id)) if isinstance(prices, Mapping) else None
        final = price_block.get("finalPrice") if isinstance(price_block, Mapping) else None
        amount = final.get("amount") if isinstance(final, Mapping) else None
        out.append(
            {
                "id": str(child_id),
                "title": " / ".join(o["label"] for o in options),
                "sku": _text(skus.get(str(child_id))) if isinstance(skus, Mapping) else None,
                "gtin": None,
                "_price": amount,
                "_minor": False,
                "available": None if salable is None else str(child_id) in salable,
                "options": options,
            }
        )
    return out


def _salable(value: Any) -> set[str] | None:
    """The children Magento's `salable` map lists (attribute id to option id to the salable child ids); None when
    the config carries no such map, so a child it leaves out is not taken for out of stock."""
    if not isinstance(value, Mapping) or not value:
        return None
    found: set[str] = set()
    for options in value.values():
        for children in options.values() if isinstance(options, Mapping) else ():
            if isinstance(children, list):
                found.update(str(c) for c in children if isinstance(c, int | str))
    return found or None


def _spconfig_variants(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Magento 1's configurable product (`new Product.Config({...})`): each attribute's options list the child ids
    they lead to and a price relative to `basePrice`; a child is the one id every chosen option shares."""
    attributes = config.get("attributes")
    if not isinstance(attributes, Mapping):
        return []
    base = price_number(config.get("basePrice"))
    lists: list[list[tuple[str, str, Decimal, set[str]]]] = []
    for attribute in attributes.values():
        if not isinstance(attribute, Mapping):
            continue
        name = _text(attribute.get("label")) or _text(attribute.get("code")) or "Option"
        options = []
        for option in attribute.get("options") or ():
            if not isinstance(option, Mapping) or not isinstance(option.get("products"), list):
                continue
            delta = price_number(option.get("price")) or Decimal(0)
            ids = {str(p) for p in option["products"] if isinstance(p, int | str)}
            options.append((name, _text(option.get("label")) or "", delta, ids))
        if options:
            lists.append(options)
    if not lists:
        return []
    out: list[dict[str, Any]] = []
    for combo in itertools.islice(itertools.product(*lists), MAX_COMBINATIONS):
        shared = set.intersection(*(ids for _, _, _, ids in combo))
        if len(shared) != 1:
            continue
        price = None if base is None else base + sum((delta for _, _, delta, _ in combo), Decimal(0))
        chosen = [{"name": name, "label": label} for name, label, _, _ in combo]
        out.append(
            {
                "id": shared.pop(),
                "title": " / ".join(o["label"] for o in chosen),
                "sku": None,
                "gtin": None,
                "_price": None if price is None else str(price),
                "_minor": False,
                "available": None,
                "options": chosen,
            }
        )
        if len(out) >= MAX_VARIANTS:
            break
    return out


def price_number(value: Any) -> Decimal | None:
    """A JSON price as an exact decimal ("54.95", 54.95), for sums before it becomes minor units; None otherwise."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _magento_simple(facts: PageFacts) -> list[dict[str, Any]]:
    """A Magento simple product: its id is the `product` input of the add-to-cart form (read, never posted)."""
    for form in facts.forms:
        if "checkout/cart/add" not in form.action:
            continue
        product = next((i.value for i in form.inputs if i.name == "product" and i.value.isdigit()), None)
        if product is not None and not any(i.name.startswith("super_attribute") for i in form.inputs):
            return [
                {
                    "id": product,
                    "title": facts.title,
                    "sku": form.sku or None,
                    "gtin": None,
                    "_price": None,
                    "_minor": False,
                    "available": None,
                    "options": [],
                }
            ]
    return []


def _priced(raws: Sequence[Mapping[str, Any]], currency: str | None, source: str) -> list[dict[str, Any]]:
    out = []
    for raw in raws:
        price = raw.get("_price")
        amount = price if raw.get("_minor") and isinstance(price, int) else price_minor(price, currency)
        variant = {k: v for k, v in raw.items() if not k.startswith("_")}
        variant["price"] = {"amount": amount, "currency": currency, "from": source} if amount is not None else None
        variant["availability"] = _availability(raw.get("available"), source)
        variant.pop("available", None)
        variant["id_from"] = source
        out.append(variant)
    return out


# -- schema.org --------------------------------------------------------------------------------------------------------


def schema_variants(nodes: Sequence[Mapping[str, Any]], currency: str | None, source: str) -> list[dict[str, Any]]:
    """One variant per schema.org Product that has an offer (a ProductGroup's members, or the page's products); a
    product with no own offers but a group's is the one variant of the group."""
    groups = [n for n in nodes if "ProductGroup" in (n.get("@type") or [])]
    varies = [schema_name(str(v)).lower() for g in groups for v in _list(g.get("variesBy")) if isinstance(v, str)]
    members = [n for n in nodes if "ProductGroup" not in (n.get("@type") or [])]
    for group in groups:
        members.extend(v for v in _list(group.get("hasVariant")) if isinstance(v, Mapping) and v not in members)
    out: dict[str, dict[str, Any]] = {}
    group_name = next((_text(g.get("name")) for g in groups if _text(g.get("name"))), None)
    for node in members:
        offer = _one_time_offer(node)
        if offer is None:
            continue
        variant = _schema_variant(node, offer, currency, varies, source)
        if not variant["options"] and group_name:
            variant["options"] = _named_option(variant["title"], group_name)
        out.setdefault(variant["id"], variant)
    if not out:
        for group in groups:
            offer = _one_time_offer(group)
            if offer is not None:
                variant = _schema_variant(group, offer, currency, varies, source)
                out.setdefault(variant["id"], variant)
    return list(out.values())


def _schema_variant(
    node: Mapping[str, Any], offer: Mapping[str, Any], currency: str | None, varies: Sequence[str], source: str
) -> dict[str, Any]:
    url = _text(offer.get("url")) or _text(node.get("url")) or ""
    sku = _text(offer.get("sku")) or _text(node.get("sku"))
    found = _VARIANT_IN_URL.search(url) or _VARIANT_IN_ID.search(_text(node.get("@id")) or "")
    ident = found.group(1) if found else sku or url or _text(node.get("name")) or "default"
    id_from = source if found or sku else "page_url"
    code = (_text(offer.get("priceCurrency")) or currency or "").upper() or None
    price = offer.get("price")
    if price is None:
        spec = offer.get("priceSpecification")
        price = spec.get("price") if isinstance(spec, Mapping) else None
    amount = price_minor(price if price is not None else offer.get("lowPrice"), code)
    options = []
    for name in dict.fromkeys([*varies, *_SCHEMA_OPTIONS]):
        value = node.get(name)
        if isinstance(value, Mapping):
            value = value.get("name")
        text = _text(value)
        if text:
            options.append({"name": name.title(), "label": text})
    return {
        "id": ident,
        "title": _text(node.get("name")) or "",
        "sku": sku,
        "gtin": next((_text(node.get(k)) for k in ("gtin13", "gtin12", "gtin14", "gtin8", "gtin") if node.get(k)), None)
        or next((_text(offer.get(k)) for k in ("gtin13", "gtin12", "gtin") if offer.get(k)), None),
        "url": url or None,
        "price": {"amount": amount, "currency": code, "from": source} if amount is not None else None,
        "availability": _availability(available_of(offer.get("availability")), source),
        "options": options,
        "id_from": id_from,
    }


def _named_option(title: str, group: str) -> list[dict[str, str]]:
    """The option a variant's name adds to its group's name ("Whey Protein, 3-Pack" after "Whey Protein")."""
    if not title.lower().startswith(group.lower()):
        return []
    rest = title[len(group) :].strip(" ,-/|:")
    return [{"name": "Option", "label": rest}] if rest else []


def _one_time_offer(node: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The node's offer of a one-time purchase: the first that is not a subscription."""
    offers = _offers(node)
    plain = [o for o in offers if not _subscription(o)]
    return plain[0] if plain else None


def _subscription(offer: Mapping[str, Any]) -> bool:
    spec = offer.get("priceSpecification")
    specs = spec if isinstance(spec, list) else [spec]
    if any(isinstance(s, Mapping) and s.get("billingDuration") for s in specs):
        return True
    return bool(_SUBSCRIPTION.search(_text(offer.get("name")) or ""))


def _offers(node: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for offer in _list(node.get("offers")):
        if not isinstance(offer, Mapping):
            continue
        inner = [o for o in _list(offer.get("offers")) if isinstance(o, Mapping)]
        out.extend(inner or [offer])
    return out


def _enriched(theme: list[dict[str, Any]], schema: list[dict[str, Any]], page: str) -> list[dict[str, Any]]:
    """Theme variants with what schema.org adds about the same variant (by id, by sku, else by the same option
    values; a page's one theme variant, else, is the schema.org product at this page's address, or the only one):
    availability where the theme said nothing, the sku, gtin and offer URL, and the price where the theme showed
    none. The 2026-10-13 smoke lost every Store 167 price here: the add-to-cart form's id (1165) never met the
    JSON-LD product, whose id is its sku."""
    by_key = {v["id"]: v for v in schema} | {v["sku"]: v for v in schema if v.get("sku")}
    by_labels = {_labels(v): v for v in schema if _labels(v)}
    single = None
    if len(theme) == 1:
        on_page = [v for v in schema if _is_page(str(v.get("url") or ""), page)]
        single = on_page[0] if len(on_page) == 1 else schema[0] if len(schema) == 1 else None
    out = []
    for variant in theme:
        other = by_key.get(variant["id"]) or (by_key.get(variant["sku"]) if variant.get("sku") else None)
        other = other or by_labels.get(_labels(variant)) or single
        if other is None:
            out.append(variant)
            continue
        merged = dict(variant)
        if merged["availability"].get("available") is None:
            merged["availability"] = other["availability"]
        merged["price"] = merged.get("price") or other.get("price")
        for key in ("sku", "gtin", "url"):
            merged[key] = merged.get(key) or other.get(key)
        out.append(merged)
    return out


def _labels(variant: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(" ".join(str(o["label"]).lower().split()) for o in variant.get("options") or ())


def _og_variant(facts: PageFacts, meta: Mapping[str, str], currency: str | None) -> dict[str, Any]:
    amount = price_minor(meta.get("product:price:amount") or meta.get("og:price:amount"), currency)
    stock = meta.get("product:availability") or meta.get("og:availability")
    available = available_of(stock) if stock else None
    url = product_url(facts.canonical or facts.url)
    return {
        "id": meta.get("product:retailer_item_id") or url,
        "title": meta.get("og:title") or facts.title,
        "sku": meta.get("product:retailer_item_id"),
        "gtin": None,
        "url": url,
        "price": {"amount": amount, "currency": currency, "from": "opengraph"} if amount is not None else None,
        "availability": _availability(available, "opengraph"),
        "options": [],
        "id_from": "opengraph" if meta.get("product:retailer_item_id") else "page_url",
    }


def _availability(value: Any, source: str) -> dict[str, Any]:
    """Availability as shown, with `from` naming the source that showed it."""
    if value is True:
        return {"available": True, "status": "in_stock", "from": source}
    if value is False:
        return {"available": False, "status": "out_of_stock", "from": source}
    return {"available": None, "status": "not_shown"}


def _options(variants: Sequence[Mapping[str, Any]], facts: PageFacts) -> list[dict[str, Any]]:
    """Option name to the values the variants carry, else the page's option pickers."""
    names: dict[str, list[str]] = {}
    for variant in variants:
        for option in variant.get("options") or ():
            values = names.setdefault(option["name"], [])
            if option["label"] not in values:
                values.append(option["label"])
    if not names:
        for name, picked in facts.options:
            if picked and not name.isdigit():
                names[name] = list(picked)
    return [{"name": name, "values": [{"label": v} for v in labels]} for name, labels in names.items()]


def _main_node(nodes: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    groups = [n for n in nodes if "ProductGroup" in (n.get("@type") or [])]
    if groups:
        return groups[0]
    described = [n for n in nodes if _text(n.get("description"))]
    return described[0] if described else (nodes[0] if nodes else None)


def _brand(node: Mapping[str, Any] | None) -> str | None:
    if node is None:
        return None
    brand = node.get("brand")
    if isinstance(brand, Mapping):
        return _text(brand.get("name"))
    return _text(brand)


def _description(node: Mapping[str, Any] | None, facts: PageFacts) -> str:
    """The product's own description, then the page's main text (where size charts and ingredient tables are)."""
    parts = []
    described = _text(node.get("description")) if node is not None else None
    if described:
        parts.append(described)
    if facts.main_text:
        parts.append(facts.main_text)
    for name, value in facts.specs:
        line = f"{name}: {value}"
        if line not in facts.main_text:
            parts.append(line)
    return "\n".join(parts)[:DESCRIPTION_CHARS]


# -- small readers -----------------------------------------------------------------------------------------------------


def query_variant(url: str) -> str | None:
    """The variant id a Shopify product URL names (?variant=)."""
    values = parse_qs(urlsplit(url).query).get("variant")
    return values[0] if values and values[0].isdigit() else None


def _json(raw: str) -> Any:
    text = raw.strip()
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _walk(value: Any, depth: int = 0) -> Iterator[Any]:
    """Every object and list in a JSON value, outermost first, at most 8 levels deep."""
    if depth > 8:
        return
    if isinstance(value, Mapping | list):
        yield value
    items = value.values() if isinstance(value, Mapping) else value if isinstance(value, list) else ()
    for item in items:
        yield from _walk(item, depth + 1)


def _find_key(value: Any, key: str) -> Any:
    for candidate in _walk(value):
        if isinstance(candidate, Mapping) and key in candidate:
            return candidate[key]
    return None


def _option_names(raw: Any) -> list[str]:
    names = []
    for item in raw if isinstance(raw, list) else ():
        name = item.get("name") if isinstance(item, Mapping) else item
        names.append(str(name) if isinstance(name, str) else "")
    return names


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return " ".join(value.split())
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def availability(value: Any, source: str) -> dict[str, Any]:
    """A variant's availability block (`available`, `status`, `from`) for True, False or None as a source said it."""
    return _availability(value, source)


def json_value(raw: str) -> Any:
    """A script's text as JSON when it is a JSON object or array; None otherwise."""
    return _json(raw)


__all__ = [
    "PolicyLink",
    "availability",
    "available_of",
    "currency_of",
    "json_value",
    "listed_products",
    "page_product",
    "policy_links",
    "price_minor",
    "price_number",
    "product_url",
    "query_variant",
    "schema_variants",
    "theme_variants",
]

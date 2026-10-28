"""A product page read field by field, each fact with the source that showed it and every source tried (decision
D36).

Why. The 2026-10-13 website smoke ended 5 of 12 episodes in refusals because read_product returned no price, no
metal option or no availability on pages where a person sees all three. The judge could not tell a store that
publishes no machine-readable price from a reader that did not look where the store put it. Read offline, the
pages answered: Store 167's JSON-LD, price box and OpenGraph all carried the price, and the reader dropped it
when the add-to-cart form's id did not meet the JSON-LD product's sku; Store 084's metal and ring-size variants sat
in a 475,070-character swatch config the parser cut at 400,000 characters; the eucalyptus "product" was a category
grid; only Store 454's Thai page carried no price anywhere (its buy button leaves the store). So every fact now
says where it came from, and the reader looks in more places:

- **structured sources** (carttrace.pages.site_product): the theme's variant data (Shopify JSON, Magento jsonConfig with
  its `salable` and `sku` maps, Magento 1 spConfig, the add-to-cart form), schema.org JSON-LD and microdata,
  OpenGraph;
- **embedded app state** (`__NEXT_DATA__`, `window.__INITIAL_STATE__` and the like, carttrace.pages.html.state_json): an
  object with a price and an id, sku or name the product already has. Read as JSON, never run;
- **price markup**: Magento's price box (`data-price-amount`, `data-price-type="finalPrice"`, by product id) and
  SFCC's sales value (`content` on `.sales .value` inside the `data-pid` it prices);
- **page text**, last and labelled low confidence: a price element's visible text with a currency marker, near an
  add-to-cart control, on the page's main region. A "from $1,720.00" is a lower bound and never a variant's price;
- **pickers**: rendered selects (named by their label) and swatch groups, so options the variants do not carry
  (Store 084's finish and engraving) reach the agent marked `in_variants: false`;
- **platform endpoints** (Shopify's `.js`, `.json`), read by the website surface only where it already may.

`field_sources` answers, for price, currency, availability, variant id and each option: `sources_tried`, the
`sources_with_value` on this page (whatever the merge kept), `found_in` (the source most variants took the value
from) with its `confidence`, how many variants have it, and a `verdict`: "found", "partial", "unattributed" (a
source shows a value the reader could not tie to a variant: the reader's limit) or "absent" (no source on the page
carries it: the store's). `page` adds the sources the page carries at all and its add-to-cart controls.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from carttrace.core.money import to_minor
from carttrace.pages.html import PageFacts, Picker, PriceMark
from carttrace.pages.site_product import (
    available_of,
    currency_of,
    json_value,
    listed_products,
    page_product,
    price_minor,
    product_url,
    schema_variants,
    theme_variants,
)

PUBLIC: Mapping[str, str] = {
    "theme_json": "theme-json",
    "magento_config": "magento-jsonconfig",
    "magento_spconfig": "magento-spconfig",
    "magento_form": "magento-form",
    "woo_variations": "woo-variations",
    "jsonld": "json-ld",
    "microdata": "microdata",
    "app_state": "app-state",
    "magento_price_box": "magento-price-box",
    "sfcc_price": "sfcc-price",
    "opengraph": "opengraph",
    "page_text": "page-text",
    "shopify_js": "platform-endpoint",
    "shopify_json": "platform-endpoint",
    "platform_endpoint": "platform-endpoint",
    "select": "select",
    "swatch": "swatch",
    "page_url": "page-url",
    "market": "market",
}
"""A fact's internal `from` to the name field_sources gives it."""
CONFIDENCE: Mapping[str, str] = {
    "theme-json": "high",
    "magento-jsonconfig": "high",
    "magento-spconfig": "high",
    "magento-form": "high",
    "woo-variations": "high",
    "json-ld": "high",
    "microdata": "high",
    "platform-endpoint": "high",
    "app-state": "medium",
    "magento-price-box": "medium",
    "sfcc-price": "medium",
    "opengraph": "medium",
    "select": "medium",
    "swatch": "medium",
    "page-text": "low",
    "page-url": "low",
    "market": "low",
}
FIELD_SOURCES: Mapping[str, tuple[str, ...]] = {
    "price": (
        "theme_json", "magento_config", "magento_spconfig", "woo_variations", "jsonld", "microdata", "app_state",
        "magento_price_box", "sfcc_price", "opengraph", "page_text",
    ),
    "currency": ("jsonld", "microdata", "opengraph", "app_state", "market"),
    "availability": ("theme_json", "magento_config", "woo_variations", "jsonld", "microdata", "app_state", "opengraph"),
    "variant_id": (
        "theme_json", "magento_config", "magento_spconfig", "woo_variations", "magento_form", "jsonld", "microdata",
        "opengraph", "page_url",
    ),
    "options": (
        "theme_json", "magento_config", "magento_spconfig", "woo_variations", "jsonld", "microdata", "select", "swatch",
    ),
}  # fmt: skip
"""Each field and the page sources that can carry it, in the order the reader trusts them."""
ENDPOINT_FIELDS = ("price", "availability", "variant_id", "options")
"""Fields a platform's product endpoint answers, when the website surface reads one."""
_PRICE_KEYS = ("price", "salePrice", "sale_price", "finalPrice", "final_price", "currentPrice", "current_price")
_ID_KEYS = ("id", "sku", "productId", "product_id", "variantId", "variant_id", "itemId", "code", "pid")
_NAME_KEYS = ("name", "title", "productName")
_STOCK_KEYS = ("available", "availableForSale", "inStock", "isInStock", "in_stock", "is_in_stock")
_LOWER_BOUND = re.compile(r"^\s*(?:from|starting at|starts at|as low as)\b", re.IGNORECASE)
_WAS = re.compile(r"\b(?:was|reduced|regular|save)\b", re.IGNORECASE)
_NUMBER = r"(\d{1,3}(?:[   ]\d{3})+(?:[.,]\d{1,2})?|\d[\d.,]*\d|\d)"
_AMOUNT = re.compile(
    rf"(?:[$€£¥₹₩₪₫₱฿]|\b(?:USD|EUR|GBP|CAD|AUD|INR)\b|\bRs\.?)\s*{_NUMBER}"
    rf"|{_NUMBER}\s*(?:[$€£¥₹₩₪₫₱฿]|\b(?:USD|EUR|GBP|CAD|AUD|INR|kr)\b)"
)
"""An amount beside its currency marker: "$1,720.00", "Rs. 2,450", "49,95 €", "1 299 kr"."""
MAX_STATE_NODES = 50_000
MAX_STATE_OFFERS = 500


@dataclass(frozen=True, slots=True)
class StateOffer:
    """An object of a page's app state that names a price: its ids, name, price, currency and stock as said."""

    ids: tuple[str, ...]
    name: str | None
    amount: int | None
    currency: str | None
    available: bool | None


@dataclass(slots=True)
class Evidence:
    """What each source of one page carries, whatever the merge keeps: the ground the judge's verdict stands on."""

    present: list[str] = field(default_factory=list)
    with_value: dict[str, list[str]] = field(default_factory=dict)
    option_sources: dict[str, list[str]] = field(default_factory=dict)
    currency_from: str | None = None
    lower_bound: dict[str, Any] | None = None
    cart_controls: int = 0
    states: list[str] = field(default_factory=list)
    offers: list[StateOffer] = field(default_factory=list)
    endpoint: str | None = None
    """The platform endpoint read for this page, when the website surface read one."""

    def saw(self, source: str, field_name: str) -> None:
        kept = self.with_value.setdefault(field_name, [])
        if source not in kept:
            kept.append(source)


def read_product_page(facts: PageFacts, *, currency: str | None = None) -> dict[str, Any] | None:
    """The product a page shows with every source read and `field_sources` attached; None when the page shows no
    product (no structured product, and no price near an add-to-cart control), or lists several."""
    evidence = page_evidence(facts, currency=currency)
    product = page_product_from(facts, evidence, currency=currency)
    if product is not None:
        product["field_sources"] = field_sources(product, evidence)
    return product


def page_evidence(facts: PageFacts, *, currency: str | None = None) -> Evidence:
    """Every source of the page, read on its own: which it carries and which fields each has a value for."""
    evidence = Evidence(cart_controls=facts.cart_controls)
    _theme_evidence(facts, evidence)
    for source in ("jsonld", "microdata"):
        _schema_evidence(facts, evidence, source, currency)
    _og_evidence(facts, evidence)
    _state_evidence(facts, evidence)
    _mark_evidence(facts, evidence)
    for picker in facts.pickers:
        evidence.option_sources.setdefault(_fold(picker.name), []).append(picker.kind)
        if picker.kind not in evidence.present:
            evidence.present.append(picker.kind)
    for name, values in facts.options:
        if values and not name.isdigit():
            evidence.option_sources.setdefault(_fold(name), []).append("select")
    if currency and evidence.currency_from is None:
        evidence.currency_from = "market"
        evidence.saw("market", "currency")
    return evidence


def page_product_from(facts: PageFacts, evidence: Evidence, *, currency: str | None = None) -> dict[str, Any] | None:
    """site_product's product, its gaps filled from the embedded sources (each fill naming its source), and the
    page's pickers added to its options."""
    product = page_product(facts, currency=currency)
    if product is None and not listed_products(facts):
        product = _bare_product(facts, evidence, currency)
    if product is None:
        return None
    if not product.get("currency") and evidence.offers:
        codes = {o.currency for o in evidence.offers if o.currency}
        if len(codes) == 1:
            product["currency"] = codes.pop()
            evidence.currency_from = evidence.currency_from or "app_state"
    product["variants"] = [
        {**v, "price": {**v["price"], "currency": product.get("currency")}}
        if v.get("price") and not v["price"].get("currency") and product.get("currency")
        else v
        for v in _filled(product, facts, evidence)
    ]
    product["options"] = _with_pickers(product.get("options") or [], product["variants"], facts.pickers)
    for source in ("app_state", "magento_price_box", "sfcc_price", "page_text"):
        if any(_from(v, "price") == source for v in product["variants"]) and source not in product["sources"]:
            product["sources"].append(source)
    return product


def field_sources(product: Mapping[str, Any] | None, evidence: Evidence) -> dict[str, Any]:
    """Per field: sources tried, sources with a value on the page, where the kept value came from, how sure, how
    many variants have it, and the verdict (found, partial, unattributed, absent)."""
    variants = [v for v in (product or {}).get("variants") or () if isinstance(v, Mapping)]
    out: dict[str, Any] = {}
    for name in ("price", "currency", "availability", "variant_id"):
        tried = [*FIELD_SOURCES[name]]
        if evidence.endpoint and name in ENDPOINT_FIELDS:
            tried.append(evidence.endpoint)
        taken = [s for s in (_field_from(v, name, product, evidence) for v in variants) if s]
        out[name] = _verdict(tried, evidence.with_value.get(name, []), taken, len(variants))
    if evidence.lower_bound is not None:
        out["price"]["lower_bound"] = evidence.lower_bound
    options: dict[str, Any] = {}
    for option in (product or {}).get("options") or ():
        tried = [*FIELD_SOURCES["options"]] + ([evidence.endpoint] if evidence.endpoint else [])
        source = str(option.get("from") or "")
        seen = evidence.option_sources.get(_fold(str(option["name"])), [])
        entry = _verdict(tried, seen or ([source] if source else []), [source] if source else [], 1)
        entry.pop("variants_with_value", None)
        entry.pop("variants", None)
        entry["values"] = len(option.get("values") or ())
        entry["in_variants"] = option.get("in_variants", True)
        options[str(option["name"])] = entry
    out["options"] = options
    out["page"] = {
        "sources_present": [PUBLIC.get(s, s) for s in evidence.present],
        "cart_controls": evidence.cart_controls,
        "app_state": evidence.states,
    }
    return out


def line_sources(variant: Mapping[str, Any], product: Mapping[str, Any]) -> dict[str, Any]:
    """Where a chosen variant's facts came from, for the selection: each field's source and confidence."""
    sources = product.get("field_sources") if isinstance(product.get("field_sources"), Mapping) else {}
    assert isinstance(sources, Mapping)
    currency = (sources.get("currency") or {}).get("found_in")
    out: dict[str, Any] = {}
    for name in ("price", "availability", "variant_id"):
        found = PUBLIC.get(str(_from(variant, name) or ""))
        out[name] = {"found_in": found, "confidence": CONFIDENCE.get(found or "")}
    out["currency"] = {"found_in": currency, "confidence": CONFIDENCE.get(currency or "")}
    option_sources = sources.get("options") if isinstance(sources.get("options"), Mapping) else {}
    assert isinstance(option_sources, Mapping)
    out["options"] = {
        str(o["name"]): {
            "found_in": (option_sources.get(str(o["name"])) or {}).get("found_in"),
            "confidence": (option_sources.get(str(o["name"])) or {}).get("confidence"),
        }
        for o in variant.get("options") or ()
    }
    return out


def with_endpoint(product: Mapping[str, Any] | None, described: Mapping[str, Any] | None, evidence: Evidence,
                  *, source: str) -> dict[str, Any] | None:  # fmt: skip
    """The product once the platform's endpoint was read (`source` names it): the endpoint's variants and options
    replace the page's when it described any, and the evidence says it was tried and what it had."""
    evidence.endpoint = source
    if described is not None:
        evidence.present.append(source)
        for name in ENDPOINT_FIELDS:
            if name == "options":
                for option in described.get("options") or ():
                    evidence.option_sources.setdefault(_fold(str(option["name"])), []).append(source)
            elif any(_from(v, name) == source for v in described.get("variants") or ()):
                evidence.saw(source, name)
    if described is None:
        return dict(product) if product is not None else None
    if product is None:
        return dict(described)
    return {
        **product,
        "variants": described["variants"],
        "options": [{**o, "from": source} for o in described.get("options") or ()],
        "sources": [*product.get("sources", ()), *described.get("sources", ())],
    }


# -- the sources read on their own -----------------------------------------------------------------------------------


def _theme_evidence(facts: PageFacts, evidence: Evidence) -> None:
    theme = theme_variants(facts)
    if theme is None:
        return
    source, raws = theme
    evidence.present.append(source)
    evidence.saw(source, "variant_id")
    if any(r.get("_price") is not None for r in raws):
        evidence.saw(source, "price")
    if any(isinstance(r.get("available"), bool) for r in raws):
        evidence.saw(source, "availability")
    _saw_options(evidence, source, raws)


def _schema_evidence(facts: PageFacts, evidence: Evidence, source: str, currency: str | None) -> None:
    nodes = facts.products(source)
    if not nodes:
        return
    evidence.present.append(source)
    code = currency_of(nodes)
    if code:
        evidence.saw(source, "currency")
        evidence.currency_from = evidence.currency_from or source
    variants = schema_variants(nodes, code or currency, source)
    if any(v.get("price") for v in variants):
        evidence.saw(source, "price")
    if any(v["availability"].get("available") is not None for v in variants):
        evidence.saw(source, "availability")
    if any(v.get("id_from") == source for v in variants):
        evidence.saw(source, "variant_id")
    _saw_options(evidence, source, variants)


def _saw_options(evidence: Evidence, source: str, variants: Sequence[Mapping[str, Any]]) -> None:
    names = dict.fromkeys(_fold(str(o["name"])) for v in variants for o in v.get("options") or ())
    for name in names:
        evidence.option_sources.setdefault(name, []).append(source)
    if names:
        evidence.saw(source, "options")


def _og_evidence(facts: PageFacts, evidence: Evidence) -> None:
    meta = dict(facts.meta)
    keys = {
        "price": ("product:price:amount", "og:price:amount"),
        "currency": ("product:price:currency", "og:price:currency"),
        "availability": ("product:availability", "og:availability"),
        "variant_id": ("product:retailer_item_id",),
    }
    for name, names in keys.items():
        if any(meta.get(k) for k in names):
            evidence.saw("opengraph", name)
            if "opengraph" not in evidence.present:
                evidence.present.append("opengraph")
    if (
        evidence.currency_from is None
        and "currency" in evidence.with_value
        and "opengraph" in evidence.with_value.get("currency", [])
    ):
        evidence.currency_from = "opengraph"


def _state_evidence(facts: PageFacts, evidence: Evidence) -> None:
    for name, raw in facts.state_scripts:
        if name == "spConfig":
            continue
        evidence.states.append(name)
        evidence.offers.extend(state_offers(json_value(raw)))
    evidence.offers = evidence.offers[:MAX_STATE_OFFERS]
    if evidence.states and "app_state" not in evidence.present:
        evidence.present.append("app_state")
    if any(o.amount is not None for o in evidence.offers):
        evidence.saw("app_state", "price")
    if any(o.available is not None for o in evidence.offers):
        evidence.saw("app_state", "availability")
    if any(o.currency for o in evidence.offers):
        evidence.saw("app_state", "currency")


def _mark_evidence(facts: PageFacts, evidence: Evidence) -> None:
    for mark in facts.prices:
        if mark.kind not in evidence.present:
            evidence.present.append(mark.kind)
        selling = (mark.kind == "magento_price_box" and mark.price_type in ("finalPrice", "")) or (
            mark.kind == "sfcc_price" and mark.price_type == "sales"
        )
        if selling:
            evidence.saw(mark.kind, "price")
        elif mark.kind == "page_text" and _sold_here(mark):
            if not _LOWER_BOUND.match(mark.value):
                evidence.saw(mark.kind, "price")
            elif evidence.lower_bound is None:
                evidence.lower_bound = {"text": mark.value, "found_in": "page-text"}


def _sold_here(mark: PriceMark) -> bool:
    return bool(mark.main and mark.near_cart and mark.price_type != "old" and text_amount(mark.value) is not None)


def state_offers(value: Any) -> list[StateOffer]:
    """Objects of an app state that carry a price and an id or a name, outermost first."""
    out: list[StateOffer] = []
    for node in _nodes(value):
        offer = _state_offer(node)
        if offer is not None:
            out.append(offer)
            if len(out) >= MAX_STATE_OFFERS:
                break
    return out


def _state_offer(node: Mapping[str, Any]) -> StateOffer | None:
    key = next((k for k in _PRICE_KEYS if k in node), None)
    if key is None:
        return None
    raw = node[key]
    code = _code(node.get("currency") or node.get("currencyCode") or node.get("priceCurrency"))
    amount: int | None = None
    if isinstance(raw, Mapping):
        code = code or _code(raw.get("currencyCode") or raw.get("currency"))
        cents = raw.get("centAmount")
        if isinstance(cents, int) and not isinstance(cents, bool):
            amount = cents
        else:
            value = raw.get("amount", raw.get("value"))
            amount = price_minor(value, code) if isinstance(value, int | float | str) else None
    elif isinstance(raw, int | float | str) and not isinstance(raw, bool):
        amount = price_minor(raw, code)
    if amount is None or amount < 0:
        return None
    ids = tuple(str(node[k]) for k in _ID_KEYS if isinstance(node.get(k), int | str) and str(node[k]).strip())
    name = next((str(node[k]) for k in _NAME_KEYS if isinstance(node.get(k), str) and node[k].strip()), None)
    if not ids and name is None:
        return None
    available = next((node[k] for k in _STOCK_KEYS if isinstance(node.get(k), bool)), None)
    if available is None and isinstance(node.get("availability"), str):
        available = available_of(node["availability"])
    return StateOffer(ids=ids, name=name, amount=amount, currency=code, available=available)


def _nodes(value: Any) -> Iterator[Mapping[str, Any]]:
    stack: list[tuple[Any, int]] = [(value, 0)]
    seen = 0
    while stack and seen < MAX_STATE_NODES:
        item, depth = stack.pop(0)
        seen += 1
        if isinstance(item, Mapping):
            yield item
            children = list(item.values())
        elif isinstance(item, list):
            children = list(item)
        else:
            continue
        if depth < 14:
            stack.extend((child, depth + 1) for child in children if isinstance(child, Mapping | list))


# -- filling the gaps ------------------------------------------------------------------------------------------------


def _filled(product: Mapping[str, Any], facts: PageFacts, evidence: Evidence) -> list[dict[str, Any]]:
    """Variants whose price or availability the structured sources left out, filled from the price markup and the
    app state by the variant's own id or sku; a product with one variant also from OpenGraph, an unambiguous price
    box, sales value or app-state offer, and last from the page's price text near its add-to-cart control."""
    currency = product.get("currency")
    variants = [dict(v) for v in product.get("variants") or ()]
    boxes = _by_product(facts, "magento_price_box", ("finalPrice", ""), currency)
    sales = _by_product(facts, "sfcc_price", ("sales",), currency)
    for variant in variants:
        keys = {str(variant.get("id") or ""), str(variant.get("sku") or "")} - {""}
        if variant.get("price") is None:
            found = next((boxes[k] for k in keys if k in boxes), None)
            source = "magento_price_box"
            if found is None:
                found, source = next((sales[k] for k in keys if k in sales), None), "sfcc_price"
            if found is None:
                found, source = _offer_price(evidence.offers, keys, None), "app_state"
            if found is not None:
                variant["price"] = _price(found, currency, source)
        if variant["availability"].get("available") is None:
            stock = next((o.available for o in evidence.offers if keys & set(o.ids) and o.available is not None), None)
            if stock is not None:
                variant["availability"] = {"available": stock, "status": _status(stock), "from": "app_state"}
    if len(variants) == 1:
        _fill_single(variants[0], product, facts, evidence, boxes=boxes, sales=sales)
    return variants


def _fill_single(
    variant: dict[str, Any],
    product: Mapping[str, Any],
    facts: PageFacts,
    evidence: Evidence,
    *,
    boxes: Mapping[str, int],
    sales: Mapping[str, int],
) -> None:
    currency = product.get("currency")
    meta = dict(facts.meta)
    if variant.get("price") is None:
        candidates: list[tuple[str, int | None]] = [
            ("opengraph", price_minor(meta.get("product:price:amount") or meta.get("og:price:amount"), currency)),
            ("magento_price_box", _only(boxes.values())),
            ("sfcc_price", _only(sales.values())),
            ("app_state", _offer_price(evidence.offers, {str(variant.get("id") or "")}, product.get("title"))),
            ("page_text", _page_text_price(facts, currency)),
        ]
        source, amount = next(((s, a) for s, a in candidates if a is not None), ("", None))
        if amount is not None:
            variant["price"] = _price(amount, currency, source)
    if variant["availability"].get("available") is None:
        stock = meta.get("product:availability") or meta.get("og:availability")
        said = available_of(stock) if stock else None
        if stock and said is None:
            said = stock.lower().replace(" ", "").replace("_", "") in ("instock", "available")
        if said is not None:
            variant["availability"] = {"available": said, "status": _status(said), "from": "opengraph"}
    if variant["availability"].get("available") is None:
        title = _fold(str(product.get("title") or ""))
        said_by = {
            o.available for o in evidence.offers if o.available is not None and o.name and _fold(o.name) == title
        }
        if len(said_by) == 1:
            in_stock = said_by.pop()
            variant["availability"] = {"available": in_stock, "status": _status(in_stock), "from": "app_state"}


def _by_product(facts: PageFacts, kind: str, types: tuple[str, ...], currency: Any) -> dict[str, int]:
    """Prices of one kind of markup by the product id around them; one id shown at two prices is left out."""
    seen: dict[str, set[int]] = {}
    for mark in facts.prices:
        if mark.kind == kind and mark.price_type in types and mark.product:
            amount = price_minor(mark.value, currency if isinstance(currency, str) else None)
            if amount is not None:
                seen.setdefault(mark.product, set()).add(amount)
    return {k: v.pop() for k, v in seen.items() if len(v) == 1}


def _offer_price(offers: Sequence[StateOffer], keys: set[str], title: Any) -> int | None:
    """The one price the app state gives for an object with one of `keys` as id or sku (or, with `title`, the
    product's own name); None when it gives none or several."""
    keys = keys - {""}
    folded = _fold(str(title)) if isinstance(title, str) else None
    amounts = {
        o.amount
        for o in offers
        if o.amount is not None and ((keys & set(o.ids)) or (folded and o.name and _fold(o.name) == folded))
    }
    return amounts.pop() if len(amounts) == 1 else None


def _page_text_price(facts: PageFacts, currency: Any) -> int | None:
    """The price the page's text shows near its add-to-cart control, in its main region, when every such price text
    (lower bounds and struck prices aside) shows the same amount; a text that says it was reduced counts by the last
    amount it names. None when there is none or they disagree (a grid, a bundle, related products)."""
    code = currency if isinstance(currency, str) else None
    amounts: set[int] = set()
    for mark in facts.prices:
        if mark.kind != "page_text" or not _sold_here(mark) or _LOWER_BOUND.match(mark.value):
            continue
        found = list(_AMOUNT.finditer(mark.value))
        chosen = found[-1] if _WAS.search(mark.value) else found[0]
        number = _normal_number(chosen.group(1) or chosen.group(2) or "")
        amount = to_minor(number, code) if number else None
        if amount is not None:
            amounts.add(amount)
    return amounts.pop() if len(amounts) == 1 else None


def text_amount(text: str) -> str | None:
    """The first amount next to a currency marker in a price text, as a plain decimal ("1,720.00" -> "1720.00",
    "49,95" -> "49.95"); None when there is none."""
    found = _AMOUNT.search(text)
    if found is None:
        return None
    return _normal_number(found.group(1) or found.group(2) or "")


def _normal_number(raw: str) -> str | None:
    text = re.sub(r"\s", "", raw)
    if "," in text and "." in text:
        decimal = "," if text.rfind(",") > text.rfind(".") else "."
        text = text.replace("." if decimal == "," else ",", "").replace(",", ".")
    elif "," in text:
        whole, _, tail = text.rpartition(",")
        text = f"{whole.replace(',', '')}.{tail}" if len(tail) == 2 else text.replace(",", "")
    elif text.count(".") > 1:
        text = text.replace(".", "")
    return text if re.fullmatch(r"\d+(?:\.\d+)?", text) else None


def _bare_product(facts: PageFacts, evidence: Evidence, currency: str | None) -> dict[str, Any] | None:
    """A page with no structured product but a price shown near its add-to-cart control (or one app-state offer
    named like the page): one variant at the page's address, its price labelled with where it was read."""
    if facts.cart_controls == 0:
        return None
    amount = _page_text_price(facts, currency)
    source = "page_text"
    if amount is None:
        amount, source = _offer_price(evidence.offers, set(), facts.title), "app_state"
    if amount is None:
        return None
    url = product_url(facts.canonical or facts.url)
    variant: dict[str, Any] = {
        "id": url,
        "title": facts.title,
        "sku": None,
        "gtin": None,
        "url": url,
        "price": _price(amount, currency, source),
        "availability": {"available": None, "status": "not_shown"},
        "options": [],
        "id_from": "page_url",
    }
    out: dict[str, Any] = {
        "id": url,
        "url": url,
        "title": facts.title,
        "brand": None,
        "description": {"plain": facts.main_text[:12_000]},
        "options": [],
        "variants": [variant],
        "sources": [],
    }
    if currency:
        out["currency"] = currency
    return out


def _with_pickers(
    options: Sequence[Mapping[str, Any]], variants: Sequence[Mapping[str, Any]], pickers: Sequence[Picker]
) -> list[dict[str, Any]]:
    """The product's options, each with the source that showed it, then the page's pickers no variant carries
    (`in_variants` false: a choice the page offers that is not part of the variant chosen, such as engraving)."""
    origin = Counter(str(v.get("id_from") or "") for v in variants if v.get("options"))
    variant_source = origin.most_common(1)[0][0] if origin else ""
    names = {_fold(str(o["name"])) for v in variants for o in v.get("options") or ()}
    out: list[dict[str, Any]] = []
    for option in options:
        entry = dict(option)
        if "from" not in entry:
            entry["from"] = variant_source if _fold(str(option["name"])) in names else "select"
        out.append(entry)
    known = {_fold(str(o["name"])) for o in out}
    for picker in pickers:
        if _fold(picker.name) in known:
            continue
        known.add(_fold(picker.name))
        out.append(
            {
                "name": picker.name,
                "values": [{"label": v} for v in picker.values],
                "from": picker.kind,
                "in_variants": False,
            }
        )
    return out


# -- small pieces ----------------------------------------------------------------------------------------------------


def _verdict(tried: Sequence[str], with_value: Sequence[str], taken: Sequence[str], total: int) -> dict[str, Any]:
    counts = Counter(taken)
    order = {s: i for i, s in enumerate(tried)}
    found = min(counts, key=lambda s: (-counts[s], order.get(s, 99))) if counts else None
    found_public = PUBLIC.get(found, found) if found else None
    have = len(taken)
    verdict = "found" if total and have >= total else "partial" if have else "unattributed" if with_value else "absent"
    return {
        "found_in": found_public,
        "confidence": CONFIDENCE.get(found_public or ""),
        "variants_with_value": have,
        "variants": total,
        "sources_tried": list(dict.fromkeys(PUBLIC.get(s, s) for s in tried)),
        "sources_with_value": list(dict.fromkeys(PUBLIC.get(s, s) for s in with_value)),
        "not_found": have == 0,
        "verdict": verdict,
    }


def _field_from(variant: Mapping[str, Any], name: str, product: Mapping[str, Any] | None, evidence: Evidence) -> str:
    if name == "currency":
        price = variant.get("price") or {}
        has = bool(price.get("currency")) or bool((product or {}).get("currency"))
        return (evidence.currency_from or "") if has else ""
    return str(_from(variant, name) or "")


def _from(variant: Mapping[str, Any], name: str) -> Any:
    if name == "price":
        return (variant.get("price") or {}).get("from")
    if name == "availability":
        block = variant.get("availability") or {}
        return block.get("from") if block.get("available") is not None else None
    if name == "variant_id":
        return variant.get("id_from") or ("page_url" if variant.get("id") else None)
    return None


def _price(amount: int, currency: Any, source: str) -> dict[str, Any]:
    return {"amount": amount, "currency": currency if isinstance(currency, str) else None, "from": source}


def _status(available: bool) -> str:
    return "in_stock" if available else "out_of_stock"


def _only(values: Any) -> int | None:
    distinct = set(values)
    return distinct.pop() if len(distinct) == 1 else None


def _code(value: Any) -> str | None:
    return value.upper() if isinstance(value, str) and len(value.strip()) == 3 and value.isalpha() else None


def _fold(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").split())


__all__ = [
    "CONFIDENCE",
    "FIELD_SOURCES",
    "PUBLIC",
    "Evidence",
    "StateOffer",
    "field_sources",
    "line_sources",
    "page_evidence",
    "page_product_from",
    "read_product_page",
    "state_offers",
    "text_amount",
    "with_endpoint",
]

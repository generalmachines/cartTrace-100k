"""A website journey's end (decision D31): the variant the agent chose, read back by the harness, as the judge reads it.

Why a record of its own. On a store without an agent protocol nothing is carted: the agent chooses a variant on the
store's page and the harness reads that page again with a GET (`verify_selection`). What the judge grades is the
"website:selection" state snapshot this module builds from that read-back, and the outcome carries the same block:
- `cart`, in the trace's own Cart schema (`cart_id` null): one line per chosen variant with the product and variant
  ids, the title, quantity, unit price in minor units, currency, and the page's option values (always, even when the
  page has a variant id); `subtotal` is unit price times quantity;
- `selection`, beside it: the page `url`, `availability_seen` (as the re-read showed it), `verified` (the re-read
  shows the variant in stock at the price it was chosen at), `id_source`, and `proof`: the `seq` of the verifying
  wire GET, the `method` that showed the facts ("json-ld", "microdata", "opengraph", "page-text" for the theme's
  own variant data, or "product-endpoint" for a platform's public product endpoint, D34) and the `fields_matched` on
  the re-read; and `found_via`, how the agent came to the page (search, link, sitemap, capture, or direct when it
  named the address itself): "capture" alone means the page was handed to it from our capture, not found (D34);
  and `field_sources` (D36), where the chosen variant's price, currency, availability, id and options were read on
  the page (json-ld, magento-jsonconfig, app-state, magento-price-box, sfcc-price, opengraph, page-text...) and how
  sure that source is, so the judge can tell the store's limits from the reader's.

Ids are the capture's whenever the page maps to it, because tasks and their witnesses are computed from the capture
(carttrace.live.capture_catalog: by variant id, sku, then option values); otherwise a stable synthetic id
`<product_id>#<sorted option values>` with `id_source` "synthetic".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from carttrace.trace.schema import Cart, CartLine

METHODS: Mapping[str, str] = {
    "jsonld": "json-ld",
    "microdata": "microdata",
    "opengraph": "opengraph",
    "theme_json": "page-text",
    "magento_config": "page-text",
    "magento_form": "page-text",
    "woo_variations": "page-text",
    "shopify_js": "product-endpoint",
    "shopify_json": "product-endpoint",
    "magento_spconfig": "page-text",
    "app_state": "app-state",
    "magento_price_box": "price-markup",
    "sfcc_price": "price-markup",
    "page_text": "visible-text",
}
"""Where carttrace.pages.site_product read a fact, as the proof names its method."""
SOURCE = "website:selection"


class Resolution(Protocol):
    product_id: str
    variant_id: str
    id_source: str


class Catalog(Protocol):
    """The capture's ids (carttrace.live.capture_catalog.CaptureCatalog)."""

    def resolve(
        self, url: str, *, variant_id: str | None, sku: str | None, options: Mapping[str, str]
    ) -> Resolution | None: ...


@dataclass(frozen=True, slots=True)
class WireSeen:
    """A wire record written during the read-back: its seq, method, url and status."""

    seq: int
    method: str
    url: str
    status: int | None


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    cart: Cart
    selection: dict[str, Any]
    raw_ref: int | None


def selection_record(
    doc: Mapping[str, Any], wire: Sequence[WireSeen], catalog: Catalog | None, *, unverified_reason: str | None = None
) -> SelectionRecord | None:
    """The snapshot of a verify_selection answer: its cart and its selection block; None when it holds no line."""
    raw_lines = [line for line in doc.get("line_items") or () if isinstance(line, Mapping)]
    if not raw_lines:
        return None
    checks = [c for c in doc.get("checks") or () if isinstance(c, Mapping)]
    lines: list[CartLine] = []
    blocks: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_lines):
        check = checks[index] if index < len(checks) else {}
        line, block = _line(raw, check, wire, catalog, unverified_reason)
        lines.append(line)
        blocks.append(block)
    currency = lines[0].currency
    subtotal = sum(line.unit_price * line.quantity for line in lines)
    cart = Cart(cart_id=None, lines=lines, subtotal=subtotal, total=subtotal, currency=currency)
    selection = dict(blocks[0])
    selection["verified"] = all(b["verified"] for b in blocks)
    if len(blocks) > 1:
        selection["lines"] = blocks
    return SelectionRecord(cart, selection, selection["proof"]["seq"])


def _line(
    raw: Mapping[str, Any],
    check: Mapping[str, Any],
    wire: Sequence[WireSeen],
    catalog: Catalog | None,
    unverified_reason: str | None,
) -> tuple[CartLine, dict[str, Any]]:
    item = raw.get("item") if isinstance(raw.get("item"), Mapping) else {}
    assert isinstance(item, Mapping)
    url = str(raw.get("url") or check.get("url") or "")
    options = {str(k): str(v) for k, v in (item.get("options") or {}).items()}
    page_variant = str(item.get("id") or "") or None
    resolved = (
        catalog.resolve(url, variant_id=page_variant, sku=_str(raw.get("sku")), options=options) if catalog else None
    )
    if resolved is not None:
        product_id, variant_id, id_source = resolved.product_id, resolved.variant_id, resolved.id_source
    else:
        product_id = str(item.get("product_id") or url)
        variant_id = synthetic_id(product_id, options)
        id_source = "synthetic"
    quantity: int = raw["quantity"] if isinstance(raw.get("quantity"), int) else 1
    unit_price: int = item["price"] if isinstance(item.get("price"), int) else 0
    currency = _str(item.get("currency")) or "USD"
    line = CartLine(
        product_id=product_id,
        variant_id=variant_id,
        title=str(item.get("title") or ""),
        quantity=quantity,
        unit_price=unit_price,
        currency=currency,
        options=options,
    )
    verified = bool(check.get("ok")) and unverified_reason is None
    block: dict[str, Any] = {
        "url": url,
        "availability_seen": check.get("availability_now")
        or (raw.get("availability") or {}).get("status")
        or "not_shown",
        "verified": verified,
        "id_source": id_source,
        "found_via": [str(v) for v in raw.get("found_via") or ()] or ["direct"],
        "proof": {
            "seq": _verifying_seq(check, wire),
            "method": METHODS.get(str(check.get("availability_from") or check.get("price_from") or ""), "page-text"),
            "fields_matched": _matched(check, page_variant),
        },
    }
    if isinstance(raw.get("field_sources"), Mapping):
        block["field_sources"] = dict(raw["field_sources"])
    reason = unverified_reason or (None if verified else _str(check.get("reason")))
    if reason:
        block["reason"] = reason
    return line, block


def synthetic_id(product_id: str, options: Mapping[str, str]) -> str:
    """`<product_id>#<sorted option values>`, the id of a variant the capture does not hold."""
    values = sorted(" ".join(str(v).lower().split()) for v in options.values() if v)
    return f"{product_id}#{'|'.join(values)}"


def _matched(check: Mapping[str, Any], page_variant: str | None) -> list[str]:
    fields = []
    if check.get("variant_found") is True and check.get("variant_id") == page_variant:
        fields.append("variant_id")
    if check.get("price_now") is not None and check.get("price_now") == check.get("price_shown"):
        fields.append("price")
    if check.get("available_now") is True:
        fields.append("availability")
    shown, now = check.get("options_shown") or {}, check.get("options_now") or {}
    if shown and shown == now:
        fields.append("options")
    return fields


def _verifying_seq(check: Mapping[str, Any], wire: Sequence[WireSeen]) -> int | None:
    """The seq of the GET that answered the re-read of this line's page: the last GET to its final address."""
    page = check.get("page") if isinstance(check.get("page"), Mapping) else {}
    assert isinstance(page, Mapping)
    targets = {str(page.get("final_url") or ""), str(page.get("url") or ""), str(check.get("url") or "")}
    gets = [w for w in wire if w.method == "GET"]
    hits = [w for w in gets if w.url in targets or w.url.split("?")[0] in targets]
    if hits:
        return hits[-1].seq
    return gets[-1].seq if gets else None


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None

"""The website: a store with no agent protocol, shopped through its own pages (decision D31).

What differs from UCP, and why the agents still run the same loop:
- The tools are the website surface's (carttrace.live.website): `site_search` (the store's own search), `read_product` (a
  product page's text and the variants, prices and availability its markup carries), `open_page` and `list_links`
  to browse, and `choose_variant`, which records the variant the shopper would buy. A product's id is its page's
  address, so reading a product is reading its page.
- There is no cart and no checkout: the journey ends at a selection (`boundary` "selection"). `choose_variant`
  answers with the whole selection in a cart's shape (an id, line items with the variant id and its price, totals),
  so the agents and the judge read it as they read a cart; the runner's read-back (`verify_selection`, which the
  agent is not shown) reads each chosen page again and verifies the variant is still in stock at its price.
- A store may show no search (a theme that renders it with script, or robots.txt disallowing /search): the reference
  agent then searches the store's product index (`browse_index`, its sitemaps: `index`), and only when that finds
  nothing browses, through `list_links`, the homepage's product and category links (`browse`).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from carttrace.agents.dialects.base import Call, Dialect

SELECTION_TOOLS = frozenset({"choose_variant", "verify_selection"})


@dataclass(frozen=True, slots=True)
class WebsiteDialect(Dialect):
    """How an agent speaks a store's website tools."""

    search: str = "site_search"
    product: str = "read_product"
    lookup: str = "read_product_list"
    """No such tool: a website reads one product page at a time."""
    create: str = "choose_variant"
    read: str = "verify_selection"
    create_checkout: str = "website_has_no_checkout"
    update_checkout: str = "website_has_no_checkout"
    get_checkout: str = "website_has_no_checkout"
    complete_checkout: str = "website_has_no_checkout"
    get_order: str | None = None
    checkout_tools: frozenset[str] = frozenset()
    order_tools: frozenset[str] = frozenset()
    browse: str | None = "list_links"
    page: str | None = "open_page"
    index: str | None = "browse_index"
    boundary: str = "selection"

    @property
    def name(self) -> str:
        return "website"

    def offered(self, tools: Collection[str]) -> bool:
        return {"site_search", "read_product", "choose_variant"} <= set(tools)

    def search_arguments(self, query: str) -> dict[str, Any]:
        return {"query": query}

    def next_search_arguments(self, arguments: Mapping[str, Any], payload: Any) -> dict[str, Any] | None:
        """A website's search is read one page at a time by browsing, not paged here."""
        del arguments, payload
        return None

    def product_arguments(self, product_id: str, selected: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
        """A product page by its address; the page shows every variant, so nothing narrows it."""
        del selected
        return {"url": product_id}

    def lookup_arguments(self, product_ids: Sequence[str]) -> dict[str, Any]:
        return {"url": product_ids[0] if product_ids else ""}

    def item_calls(self, items: Sequence[tuple[str, str, int]], *, currency: str | None = None) -> list[Call]:
        """One choose_variant per item: (product page, variant id, quantity); the last reply is the selection."""
        del currency
        return [
            (self.create, {"url": product_id, "variant": {"id": variant_id}, "quantity": quantity})
            for product_id, variant_id, quantity in items
        ]

    def cart_calls(self, lines: Sequence[tuple[str, int]], *, currency: str | None = None) -> list[Call]:
        raise ValueError("a website selection names the product page of each variant; use item_calls")

    def read_arguments(self, cart_id: str) -> dict[str, Any]:
        return {"id": cart_id}

    def browse_arguments(self, url: str, kind: str) -> dict[str, Any]:
        return {"url": url, "kind": kind}

    def is_cart_tool(self, name: str) -> bool:
        return name in SELECTION_TOOLS

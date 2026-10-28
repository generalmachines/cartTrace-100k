"""The one market table: per market, its currency, how tax works, what shipping costs, where pickup is, which payment
methods it takes and the return window.

Why one table. Before 2026-10-09 the lab's sandbox priced shipping from four places and tax from a fifth, and they
disagreed. The worst case was India: an 18% rate was added on top of prices that Indian storefronts already quote
tax-inclusive (MRP includes GST), so every IN order was 18% too dear. Here a market says whether its prices include
tax, and pricing adds tax only where they do not.

Every fee is in that market's own minor units (79 rupees is 7900, 600 yen is 600). The numbers are ordinary flat fees
for a small store in that market. An environment pack copies the row it uses, so changing this table is a new pack
id, never a silent change to an old episode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from carttrace.core.errors import NotFoundError


@dataclass(frozen=True, slots=True)
class ShippingOption:
    id: str
    title: str
    carrier: str
    amount: int
    free_above: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "carrier": self.carrier,
            "amount": self.amount,
            "free_above": self.free_above,
        }


@dataclass(frozen=True, slots=True)
class PickupLocation:
    id: str
    title: str
    address_country: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "address_country": self.address_country}


@dataclass(frozen=True, slots=True)
class Market:
    code: str
    currency: str
    tax_bp: int
    """Tax rate in basis points: 800 is 8%."""
    tax_inclusive: bool
    """True when listed prices already include tax, as in IN, UK, EU, AU, NZ, SG, JP, MX. Tax is then never added."""
    shipping: tuple[ShippingOption, ...]
    pickup: tuple[PickupLocation, ...] = ()
    accepts: tuple[str, ...] = ("card", "cod")
    """Payment method types the market's checkout takes."""
    return_window_days: int = 30
    unshippable: tuple[str, ...] = ()
    """Destination countries this store refuses to ship to."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "currency": self.currency,
            "tax_bp": self.tax_bp,
            "tax_inclusive": self.tax_inclusive,
            "shipping": [o.to_dict() for o in self.shipping],
            "pickup": [p.to_dict() for p in self.pickup],
            "accepts": list(self.accepts),
            "return_window_days": self.return_window_days,
            "unshippable": list(self.unshippable),
        }

    def option(self, option_id: str) -> ShippingOption | None:
        return next((o for o in self.shipping if o.id == option_id), None)

    def location(self, location_id: str) -> PickupLocation | None:
        return next((p for p in self.pickup if p.id == location_id), None)


class UnknownMarket(NotFoundError):
    """A market code that has no row in the table."""


def _ship(std: int, exp: int, free: int) -> tuple[ShippingOption, ...]:
    return (
        ShippingOption("standard", "Standard (4-7 days)", "Sandbox Post", std, free),
        ShippingOption("express", "Express (1-2 days)", "Sandbox Courier", exp, None),
    )


def _pickup(country: str) -> tuple[PickupLocation, ...]:
    return (PickupLocation("pickup-1", "Sandbox store counter", country),)


MARKETS: Mapping[str, Market] = {
    "US": Market("US", "USD", 800, False, _ship(599, 1499, 5000), _pickup("US")),
    "CA": Market("CA", "CAD", 1300, False, _ship(995, 2495, 7500), _pickup("CA")),
    "IN": Market("IN", "INR", 1800, True, _ship(7900, 19900, 99900), _pickup("IN"), accepts=("card", "upi", "cod")),
    "UK": Market("UK", "GBP", 2000, True, _ship(399, 999, 5000), _pickup("GB")),
    "EU": Market("EU", "EUR", 2000, True, _ship(495, 1295, 6000), _pickup("DE")),
    "AU": Market("AU", "AUD", 1000, True, _ship(995, 1995, 9900), _pickup("AU")),
    "NZ": Market("NZ", "NZD", 1500, True, _ship(895, 1995, 10000), _pickup("NZ")),
    "SG": Market("SG", "SGD", 900, True, _ship(599, 1499, 6000), _pickup("SG")),
    "MX": Market("MX", "MXN", 1600, True, _ship(9900, 24900, 99900), _pickup("MX")),
    "JP": Market("JP", "JPY", 1000, True, _ship(600, 1500, 5000), _pickup("JP")),
    "PH": Market("PH", "PHP", 1200, True, _ship(9900, 24900, 99900), _pickup("PH"), return_window_days=15),
}
"""The one table. Keyed by the lab's market codes (UK and EU, not GB and DE); `ALIASES` maps country codes onto them."""

ALIASES: Mapping[str, str] = {"GB": "UK", "DE": "EU", "FR": "EU", "IT": "EU", "ES": "EU", "NL": "EU", "IE": "EU"}


def countries_served(code: str) -> tuple[str, ...]:
    """The ISO 3166-1 alpha-2 countries a market code sells to: GB for the lab's UK, the EU's member rows for EU, and
    a market named by its country (US, CA, ...) itself. A market code is not a country code: comparing an address's
    `gb` with the market `uk` refused every British address on the Medusa pack (6 refusals on 2026-10-15)."""
    m = ALIASES.get(str(code).upper(), str(code).upper())
    return tuple(c for c, to in ALIASES.items() if to == m) or (m,)


def market(code: str) -> Market:
    """The row for a market code. An unknown market is an error, never a silent fall back to the US row."""
    c = str(code).upper()
    c = ALIASES.get(c, c)
    row = MARKETS.get(c)
    if row is None:
        raise UnknownMarket(
            f"no market {code!r} in the sandbox market table",
            hint=f"use one of {sorted(MARKETS)} or add a row to MARKETS in carttrace/sandbox/core/market.py",
        )
    return row


def market_for(code: str | None, currency: str | None) -> Market:
    """The market a store sells in, priced in the currency its catalog is actually in.

    A store's catalog currency decides the scale of every fee (a rupee catalog cannot be charged 5.99 in rupees), and
    its market decides tax and payment methods. When the two disagree (a test catalog in AUD served to IN shoppers),
    the shipping rows come from the market whose currency matches the catalog, and tax from the named market. No code
    means the US row; an unknown code is refused.
    """
    m = market(code) if code is not None else MARKETS["US"]
    cur = str(currency or m.currency).upper()
    if cur == m.currency:
        return m
    by_cur = next((r for r in MARKETS.values() if r.currency == cur), None)
    shipping = by_cur.shipping if by_cur else MARKETS["US"].shipping
    return replace(m, currency=cur, shipping=shipping)


def with_shipping(m: Market, options: Sequence[Mapping[str, Any]]) -> Market:
    """The same market with a store's own captured shipping facts (titles, carriers, free-above thresholds) over the
    table's fees. A captured fee is never used: a live store's page gives facts, not a fee schedule we can trust, so
    the fee stays this table's."""
    out = []
    for o in options:
        base = m.option(str(o.get("id"))) or m.shipping[0]
        free = o.get("free_above", base.free_above)
        out.append(
            ShippingOption(
                str(o.get("id") or base.id),
                str(o.get("title") or base.title),
                str(o.get("carrier") or base.carrier),
                base.amount,
                None if free is None else int(free),
            )
        )
    return replace(m, shipping=tuple(out))

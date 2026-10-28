"""Money: an integer in the currency's ISO 4217 minor units plus a currency code, and nothing else.

Why one module. The lab once carried five currency tables that disagreed about which currencies have cents. Every
package here (sandbox pricing, judge budgets, live cart parsing) takes money from this module. Rules: never a float in
arithmetic, never multiply by a hundred blindly (the yen has no minor unit, the dinar has three), and every percentage
is applied in integer arithmetic with half-up rounding so a replay is byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

ZERO_DECIMAL = frozenset(
    {"JPY", "KRW", "VND", "CLP", "ISK", "PYG", "UGX", "RWF", "XAF", "XOF", "XPF", "KMF", "DJF", "GNF", "BIF", "VUV"}
)
THREE_DECIMAL = frozenset({"BHD", "IQD", "JOD", "KWD", "LYD", "OMR", "TND"})


def exponent(currency: str | None) -> int:
    code = (currency or "USD").upper()
    return 0 if code in ZERO_DECIMAL else 3 if code in THREE_DECIMAL else 2


@dataclass(frozen=True, slots=True)
class Money:
    amount: int
    currency: str

    def __post_init__(self) -> None:
        if isinstance(self.amount, bool) or not isinstance(self.amount, int):
            raise TypeError(f"Money.amount must be an int in minor units, got {type(self.amount).__name__}")
        if not (isinstance(self.currency, str) and len(self.currency) == 3 and self.currency.isupper()):
            raise ValueError(f"Money.currency must be an ISO 4217 code, got {self.currency!r}")

    def __add__(self, other: Money) -> Money:
        self._same(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._same(other)
        return Money(self.amount - other.amount, self.currency)

    def _same(self, other: Money) -> None:
        if self.currency != other.currency:
            raise ValueError(f"cannot combine {self.currency} with {other.currency}")

    def major(self) -> str:
        return to_major(self.amount, self.currency) or "0"


def to_minor(amount: object, currency: str | None) -> int | None:
    """A decimal price (a string or a number) to integer minor units, rounding half up; None when not a price."""
    if amount is None or amount == "" or isinstance(amount, bool):
        return None
    try:
        d = Decimal(str(amount))
    except (InvalidOperation, ValueError):
        return None
    if not d.is_finite():
        return None
    return int((d * (10 ** exponent(currency))).to_integral_value(rounding=ROUND_HALF_UP))


def to_major(minor: int | None, currency: str | None) -> str | None:
    """Integer minor units to the price string a storefront shows: 2500 USD -> "25.00", 2500 JPY -> "2500"."""
    if minor is None:
        return None
    e = exponent(currency)
    if e == 0:
        return str(int(minor))
    sign = "-" if minor < 0 else ""
    q, r = divmod(abs(int(minor)), 10**e)
    return f"{sign}{q}.{r:0{e}d}"


def percent_of(amount: int, basis_points: int) -> int:
    """`amount * basis_points / 10000`, rounded half up, in integers. 10% is 1000 basis points."""
    if amount < 0 or basis_points < 0:
        raise ValueError("percent_of takes non-negative amounts and rates")
    return (amount * basis_points + 5000) // 10000


def whole_percent_of(amount: int, percent: int) -> int:
    """`amount * percent / 100`, rounded half up: a discount code's "10% off"."""
    return percent_of(amount, percent * 100)


def included_tax(gross: int, basis_points: int) -> int:
    """The tax already inside a tax-inclusive price: gross * r / (1 + r), rounded half up."""
    if gross <= 0 or basis_points <= 0:
        return 0
    return (2 * gross * basis_points + (10000 + basis_points)) // (2 * (10000 + basis_points))

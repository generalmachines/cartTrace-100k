"""How much a variant holds and what its label states per 100 g: the facts behind compare criteria such as "cheapest
per tablet" and "the most protein per 100 g".

Why its own module. A compare task is only worth judging when its criterion is exact, so task generation, the
reference agent and validate_task must all read the same number from the same text. Quantities are read from the
variant's options and title first, then the product's title, as stores write them: "32 tablets", "500 g", "1.5 l",
"6 x 330 ml". Masses become grams and volumes millilitres, so a 1 kg bag and a 500 g bag compare. Nutrients are read
from the "<Nutrient> per 100 g" lines the sandbox catalogs write from a label (Open Food Facts) and the common
"Protein: 12 g per 100 g" wording. Every number is a Fraction, so a price per unit is exact and the same everywhere.

Anything the data does not state is None, and a criterion is only offered for a set of candidates that all state
it in the same unit: a price per tablet never compares with a price per gram.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Final

from carttrace.agents.products import Product, Variant
from carttrace.agents.text import fold

COUNT_UNITS: Final[Mapping[str, str]] = {
    "tablet": "tablet",
    "tablets": "tablet",
    "caplet": "caplet",
    "caplets": "caplet",
    "capsule": "capsule",
    "capsules": "capsule",
    "softgel": "softgel",
    "softgels": "softgel",
    "gummy": "gummy",
    "gummies": "gummy",
    "sachet": "sachet",
    "sachets": "sachet",
    "pod": "pod",
    "pods": "pod",
    "bag": "bag",
    "bags": "bag",
    "wipe": "wipe",
    "wipes": "wipe",
    "roll": "roll",
    "rolls": "roll",
    "piece": "piece",
    "pieces": "piece",
    "pcs": "piece",
    "count": "item",
    "ct": "item",
    "pack": "item",
}
"""Words for things counted one by one, to the unit a price is quoted per."""
MASS_UNITS: Final[Mapping[str, Fraction]] = {"mg": Fraction(1, 1000), "g": Fraction(1), "kg": Fraction(1000)}
VOLUME_UNITS: Final[Mapping[str, Fraction]] = {
    "ml": Fraction(1),
    "cl": Fraction(10),
    "l": Fraction(1000),
    "litre": Fraction(1000),
    "liter": Fraction(1000),
    "litres": Fraction(1000),
    "liters": Fraction(1000),
}
NUTRIENTS: Final[Mapping[str, tuple[str, ...]]] = {
    "protein": ("protein", "proteins"),
    "fibre": ("fibre", "fiber"),
    "sugars": ("sugars", "sugar"),
    "salt": ("salt",),
    "saturated fat": ("saturated fat",),
}
"""The nutrients a compare task may turn on, and the names a label gives them."""
MOST: Final = ("protein", "fibre")
"""Nutrients a shopper wants more of."""
LEAST: Final = ("sugars", "salt", "saturated fat")
"""Nutrients a shopper wants less of."""

_NUMBER = r"(\d+(?:[.,]\d+)?)"
_MULTI = re.compile(rf"(?<![\w.])(\d+)\s?[x×]\s?{_NUMBER}\s?(mg|g|kg|ml|cl|l)\b", re.IGNORECASE)
_AMOUNT = re.compile(
    rf"(?<![\w.]){_NUMBER}\s?(mg|g|kg|ml|cl|l|litres?|liters?|"
    + "|".join(sorted(COUNT_UNITS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Quantity:
    """How much one variant holds: a count of units, grams or millilitres."""

    amount: Fraction
    unit: str
    """A COUNT_UNITS unit ("tablet"), "g" or "ml"."""

    @property
    def basis(self) -> Fraction:
        """What a price is quoted per: one unit counted, or 100 g or 100 ml."""
        return Fraction(100) if self.unit in ("g", "ml") else Fraction(1)

    @property
    def basis_words(self) -> str:
        """The basis as a shopper says it: "tablet", "100 g"."""
        return basis_words(self.unit)


def basis_words(unit: str) -> str:
    """What a price per unit is quoted per, as a shopper says it: "tablet", "100 g", "100 ml"."""
    return f"100 {unit}" if unit in ("g", "ml") else unit


def quantity_of(product: Product, variant: Variant) -> Quantity | None:
    """What the variant holds, from its option values and title, else the product's title; None when none says."""
    places = [value for _, value in variant.options] + [variant.title, product.title]
    for place in places:
        found = _quantity(place)
        if found is not None:
            return found
    return None


def unit_price(product: Product, variant: Variant) -> Fraction | None:
    """The variant's price per unit of what it holds (per tablet, per 100 g), in minor units, exactly."""
    held = quantity_of(product, variant)
    if variant.price is None or held is None or held.amount <= 0:
        return None
    return Fraction(variant.price) * held.basis / held.amount


def common_unit(pairs: Sequence[tuple[Product, Variant]]) -> str | None:
    """The one unit every variant of `pairs` states its quantity in, or None when one states none or they differ."""
    units = {held.unit if held is not None else None for held in (quantity_of(p, v) for p, v in pairs)}
    if len(units) != 1:
        return None
    unit = units.pop()
    return unit


def nutrient(product: Product, name: str) -> Fraction | None:
    """Grams of a NUTRIENTS nutrient per 100 g, as the product's label states it; None when it does not."""
    words = NUTRIENTS.get(name)
    if words is None:
        return None
    for line in product.lines:
        folded = fold(line).removeprefix("attributes ").strip()
        for word in words:
            found = _per_100(folded, word)
            if found is not None:
                return found
    return None


def _quantity(text: str) -> Quantity | None:
    multi = _MULTI.search(text)
    if multi is not None:
        count, each, unit = int(multi.group(1)), _fraction(multi.group(2)), multi.group(3).lower()
        if each is not None:
            return _measured(count * each, unit)
    found = _AMOUNT.search(text)
    if found is None:
        return None
    amount = _fraction(found.group(1))
    unit = found.group(2).lower()
    if amount is None or amount <= 0:
        return None
    if unit in COUNT_UNITS:
        return Quantity(amount, COUNT_UNITS[unit])
    return _measured(amount, unit)


def _measured(amount: Fraction, unit: str) -> Quantity | None:
    if unit in MASS_UNITS:
        return Quantity(amount * MASS_UNITS[unit], "g")
    if unit in VOLUME_UNITS:
        return Quantity(amount * VOLUME_UNITS[unit], "ml")
    return None


def _per_100(line: str, word: str) -> Fraction | None:
    """ "protein per 100 g, g: 12.5", "protein per 100g: 12.5 g", or "protein: 12.5 g per 100 g"."""
    name = re.escape(word)
    patterns = (
        rf"^{name} per 100 ?g(?:, ?g)?\s*[:=]\s*{_NUMBER}",
        rf"^{name}\s*[:=]\s*{_NUMBER}\s?g? per 100 ?g\b",
    )
    for pattern in patterns:
        found = re.search(pattern, line)
        if found is not None:
            return _fraction(found.group(1))
    return None


def _fraction(text: str) -> Fraction | None:
    try:
        return Fraction(Decimal(text.replace(",", ".")))
    except (InvalidOperation, ValueError):
        return None

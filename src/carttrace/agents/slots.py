"""What a task can say about a product: which variant option is which hidden-spec field, what fills a template
slot, and how a budget reads.

Why its own module. A slot that is filled must also become a predicate, or the opening would state a constraint the
judge never checks ("ibuprofen 200 mg" in the text, any painkiller in the cart). So every slot this module can fill
names the predicate it implies, and slots it cannot fill from the store's data (an occasion, a delivery date)
make their template unusable rather than invented.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

from carttrace.agents.predicates import OPTION_NAMES
from carttrace.agents.products import Product, Variant
from carttrace.agents.spec import Op
from carttrace.agents.text import fold, has_phrase
from carttrace.core.money import exponent, to_major

SYMBOLS = {
    "USD": "$",
    "CAD": "CA$",
    "AUD": "A$",
    "NZD": "NZ$",
    "SGD": "S$",
    "HKD": "HK$",
    "MXN": "MX$",
    "GBP": "£",
    "EUR": "€",
    "INR": "₹",
    "JPY": "¥",
}
DOSAGE_FORMS = (
    ("tablet", ("tablet*", "caplet*")),
    ("capsule", ("capsule*", "softgel*")),
    ("liquid", ("liquid", "syrup", "oral solution", "suspension")),
    ("chewable", ("chewable*",)),
    ("gummy", ("gummy", "gummies")),
    ("spray", ("spray*",)),
    ("cream", ("cream", "ointment", "gel")),
    ("drops", ("drops",)),
    ("powder", ("powder", "sachet*")),
)
"""The dosage_form vocabulary (data/categories/_fields.yaml) and the words a product title uses for each."""
ACTIVE_INGREDIENTS = (
    "acetaminophen",
    "aspirin",
    "cetirizine",
    "dextromethorphan",
    "diphenhydramine",
    "famotidine",
    "fexofenadine",
    "guaifenesin",
    "hydrocortisone",
    "ibuprofen",
    "loperamide",
    "loratadine",
    "melatonin",
    "naproxen",
    "omeprazole",
    "paracetamol",
    "pseudoephedrine",
)
"""Common over-the-counter active ingredients, found by name when a product states no "Active ingredient:" line."""
_STRENGTH = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?(mg|mcg|µg|g|ml|iu)\b", re.IGNORECASE)
_PACK = re.compile(
    r"(?<![\w.])(\d+)\s?(tablets|capsules|caplets|softgels|gummies|count|ct|pack|sachets|ml)\b", re.IGNORECASE
)
_ACTIVE_LINE = re.compile(r"active ingredients?\s*(?:\([^)]*\))?\s*[:\-]\s*([a-z][a-z \-]+)", re.IGNORECASE)
_SYSTEM = re.compile(r"\b(us|uk|eu|it|fr|jp|au)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SlotValue:
    """What fills a slot, and the predicate it implies."""

    text: str
    """How the shopper says it ("200 mg")."""
    op: Op
    """eq for a variant option, mentions for a fact found in the product's text, note for a qualifier."""
    value: str
    """What the predicate checks, exactly as the product writes it ("200mg"), so the check finds it again."""
    option: str | None = None
    """The store's option name, for eq."""


def option_field(name: str, fields: Collection[str]) -> str | None:
    """The hidden-spec field a variant option holds, when the category uses that field: "Colour" is color, and a
    "Size" option is pack_size in a category that has pack sizes but no garment sizes."""
    key = fold(re.sub(r"\(.*?\)", "", name)).removeprefix("select ").strip()
    for candidate, names in OPTION_NAMES.items():
        if key in names:
            if candidate == "size" and "size" not in fields and "pack_size" in fields:
                return "pack_size"
            return candidate if candidate in fields else None
    return None


def option_slots(variant: Variant, fields: Collection[str]) -> dict[str, SlotValue]:
    """Each option of the variant that the category has a field for, as a slot value (first option wins)."""
    out: dict[str, SlotValue] = {}
    for name, value in variant.options:
        field = option_field(name, fields)
        if field is None or field in out or fold(value) in ("default title", "default"):
            continue
        out[field] = SlotValue(text=value, op="eq", value=value, option=name)
    return out


def text_slots(product: Product, variant: Variant, fields: Collection[str]) -> dict[str, SlotValue]:
    """Facts a template may state that come from the product's text: the active ingredient, strength, dosage form
    and pack size of a medicine, and the sizing system of a size option."""
    out: dict[str, SlotValue] = {}
    title = f"{product.title} {variant.title}"
    if "active_ingredient" in fields:
        found = _active_ingredient(product)
        if found:
            out["active_ingredient"] = SlotValue(found, "mentions", found)
    if "strength" in fields:
        strength = _STRENGTH.search(title)
        if strength:
            text = f"{strength.group(1)} {strength.group(2).lower()}"
            out["strength"] = SlotValue(text, "mentions", strength.group(0))
    if "dosage_form" in fields:
        form = next((f for f, words in DOSAGE_FORMS if any(has_phrase(title, w) for w in words)), None)
        if form:
            out["dosage_form"] = SlotValue(form, "mentions", form)
    if "pack_size" in fields:
        pack = _PACK.search(title)
        if pack:
            text = f"{pack.group(1)} {pack.group(2).lower()}"
            out["pack_size"] = SlotValue(text, "mentions", pack.group(0))
    if "size_system" in fields:
        for name, value in variant.options:
            if "size" in fold(name):
                system = _SYSTEM.search(name) or _SYSTEM.search(value)
                if system:
                    out["size_system"] = SlotValue(system.group(1).upper(), "note", system.group(1).upper())
                break
    return out


def _active_ingredient(product: Product) -> str | None:
    for line in product.lines:
        found = _ACTIVE_LINE.search(line)
        if found:
            words = found.group(1).strip().split()
            return " ".join(words[:2]).lower() if words else None
    text = f"{product.title} {product.product_type or ''}"
    return next((name for name in ACTIVE_INGREDIENTS if has_phrase(text, name)), None)


def money(amount: int, currency: str) -> str:
    """How a shopper writes a price: $60, $59.99, £45, ¥6000, 120.00 CHF."""
    major = to_major(amount, currency) or "0"
    if exponent(currency) and major.endswith("." + "0" * exponent(currency)):
        major = major.split(".")[0]
    symbol = SYMBOLS.get(currency)
    return f"{symbol}{major}" if symbol else f"{major} {currency}"


def budget_above(price: int, currency: str, *, percent: int = 15) -> int:
    """A round budget above `price`: `percent` more (15 by default; a tight shopper's 5, a generous one's 40), rounded
    up to a step a person would choose."""
    return _round(price * (100 + percent) // 100 + 1, currency, up=True)


def budget_below(price: int, currency: str) -> int | None:
    """A round budget well below `price` (about 60% of it), or None when no positive round amount is below it."""
    amount = _round(price * 60 // 100, currency, up=False)
    return amount if 0 < amount < price else None


def _round(minor: int, currency: str, *, up: bool) -> int:
    unit = int(10 ** exponent(currency))
    major = -(-minor // unit) if up else minor // unit
    step = 1 if major < 10 else 5 * int(10 ** (len(str(major)) - 2))
    rounded = -(-major // step) * step if up else (major // step) * step
    return rounded * unit

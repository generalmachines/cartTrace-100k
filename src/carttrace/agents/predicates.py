"""Checking a hidden-spec predicate against a product variant, with the store's own data as the only evidence.

Why one checker. Task generation computes each predicate's `satisfied_by` and the witnesses from fixture data; the
reference agent checks the same predicates against what a surface returned. If the two disagreed, the reference agent
would refuse a task that has a witness, or pass one that has none, and the benchmark would blame a store for our
bug. Each check answers True, False, or None when the data cannot say (no size chart, no ingredient list, no price);
None is never counted as a pass.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any

from pydantic import JsonValue

from carttrace.agents.details import detail
from carttrace.agents.details.size_chart import size_key
from carttrace.agents.measures import nutrient, unit_price
from carttrace.agents.memo import table
from carttrace.agents.nouns import noun_of, plural_of, singular_of
from carttrace.agents.products import Product, Variant
from carttrace.agents.spec import SUPERLATIVES, Predicate
from carttrace.agents.text import fold, has_phrase

DETAIL_OPS: Mapping[str, str] = {"fits": "size_chart", "excludes": "allergens", "safe_for": "warnings"}
"""Ops a decisive detail checks, to the detail's name."""
CART_OPS = frozenset({"count", "note", "lines", "checkout", "policy", *SUPERLATIVES})
"""Ops not checked one variant at a time: a cart line's quantity, the cart's line count, a qualifying note, a wish
settled at checkout, a policy question, and the best of a set (the cheapest, the cheapest per unit, the most or least
of a nutrient), which `best` picks."""
OPTION_NAMES: Mapping[str, tuple[str, ...]] = {
    "color": ("color", "colour", "colors", "colours"),
    "size": ("size", "sizes"),
    "material": ("material", "fabric"),
    "pack_size": ("pack size", "pack", "count", "size", "quantity", "pack of"),
    "scent": ("scent", "fragrance"),
    "flavor": ("flavor", "flavour", "flavors", "flavours"),
    "strength": ("strength",),
    "width": ("width",),
    "length": ("length", "inseam", "leg length"),
    "fit": ("fit",),
    "shade": ("shade",),
    "capacity": ("capacity", "volume"),
    "format": ("format",),
    "edition": ("edition",),
    "metal": ("metal",),
    "stone": ("stone", "gemstone"),
    "dosage_form": ("form", "dosage form"),
    "condition": ("condition",),
}
"""For each hidden-spec field a variant option can hold, the option names stores use for it."""


def holds(predicate: Predicate, product: Product, variant: Variant, *, quantity: int = 1) -> bool | None:
    """Whether `variant` of `product` satisfies `predicate`: True, False, or None when the data cannot say. Ops in
    CART_OPS always give None here."""
    if predicate.op in DETAIL_OPS:
        return detail(DETAIL_OPS[predicate.op]).check(predicate.value, product, variant)
    check = _CHECKS.get(predicate.op)
    return None if check is None else check(predicate, product, variant, quantity)


def _kind(predicate: Predicate, product: Product, variant: Variant, quantity: int) -> bool | None:
    return isinstance(predicate.value, str) and kind_matches(product, predicate.value)


def _is(predicate: Predicate, product: Product, variant: Variant, quantity: int) -> bool | None:
    value = predicate.value
    return isinstance(value, str) and (fold(product.title) == fold(value) or product.id == value)


def _one_of(predicate: Predicate, product: Product, variant: Variant, quantity: int) -> bool | None:
    names = predicate.value if isinstance(predicate.value, list) else []
    wanted = {fold(str(n)) for n in names if isinstance(n, str)}
    return fold(product.title) in wanted or product.id in names


def _all_of(predicate: Predicate, product: Product, variant: Variant, quantity: int) -> bool | None:
    """The variant belongs to one of a bundle's items; the judge checks that every item is in the cart."""
    return any(item_holds(item, product, variant) for item in bundle_items(predicate.value))


def _prefer(predicate: Predicate, product: Product, variant: Variant, quantity: int) -> bool | None:
    """The named product itself. A substitute is allowed by the spec, not by the variant (carttrace.agents.tasks.holds_in)."""
    return _is(predicate, product, variant, quantity)


def bundle_items(value: object) -> list[Mapping[str, object]]:
    """The items of an all_of predicate's value: mappings with a title, a product id and variant ids."""
    return [item for item in (value if isinstance(value, list) else []) if isinstance(item, Mapping)]


def item_holds(item: Mapping[str, object], product: Product, variant: Variant) -> bool:
    variants = item.get("variants")
    ids = {str(v) for v in variants} if isinstance(variants, list) else set()
    title = item.get("title")
    return (
        variant.id in ids
        or product.id == item.get("product_id")
        or (isinstance(title, str) and fold(product.title) == fold(title))
    )


def _brand(predicate: Predicate, product: Product, variant: Variant, quantity: int) -> bool | None:
    """The product's brand field, else its title naming the brand; unknown when a store shows neither. Why (D104): a
    platform pack that serves no brand field made the reference agent refuse "AmazonBasics Anti-Theft ... Backpack"
    for a shopper who asked for the AmazonBasics backpack."""
    value = predicate.value
    if not isinstance(value, str):
        return False
    if product.brand is not None:
        return brand_key(product.brand) == brand_key(value)
    key = brand_key(value)
    return True if key and key in brand_key(product.title) else None


def brand_key(name: str) -> str:
    """A brand compared by its letters and digits: store-146.example lists most products under "Acme Brush Co" and
    a few under "Store 146", and on 2026-10-15 a brush of either was the brand the shopper named."""
    return re.sub(r"[^a-z0-9]+", "", fold(name))


def _mentions(predicate: Predicate, product: Product, variant: Variant, quantity: int) -> bool | None:
    return isinstance(predicate.value, str) and mentions(product, variant, predicate.value)


def _max_price(predicate: Predicate, product: Product, variant: Variant, quantity: int) -> bool | None:
    budget = _budget(predicate.value)
    if budget is None or variant.price is None:
        return None
    amount, currency = budget
    if currency and variant.currency and currency != variant.currency:
        return False
    return variant.price * max(1, quantity) <= amount


def _budget(value: JsonValue) -> tuple[int, JsonValue] | None:
    """(amount, currency) of a max_price predicate's value; None when it states no integer amount."""
    if not isinstance(value, Mapping):
        return None
    amount = value.get("amount")
    return (amount, value.get("currency")) if isinstance(amount, int) else None


def _eq(predicate: Predicate, product: Product, variant: Variant, quantity: int) -> bool | None:
    if predicate.value is None:
        return True  # no preference
    found = variant.option(*option_names(predicate.field, predicate.option))
    if found is None:
        return None
    return eq_key(predicate.field, found) == eq_key(predicate.field, str(predicate.value))


def eq_key(field: str, text: str) -> str:
    """What an eq check compares of an option value: a size in `size_key` form (else folded), anything else
    folded."""
    if field == "size":
        return size_key(text) or fold(text)
    return fold(text)


_CHECKS: Mapping[str, Callable[[Predicate, Product, Variant, int], bool | None]] = {
    "kind": _kind,
    "is": _is,
    "one_of": _one_of,
    "all_of": _all_of,
    "prefer": _prefer,
    "brand": _brand,
    "mentions": _mentions,
    "max_price": _max_price,
    "eq": _eq,
}
"""Ops checked from the variant and its product; CART_OPS have no entry and DETAIL_OPS go to their detail."""


def satisfied_by(
    predicate: Predicate, pool: Sequence[Product], *, quantity: int = 1, available_only: bool = False
) -> list[str] | None:
    """The variant ids in `pool` that satisfy `predicate`, in pool order; None for ops checked on the cart."""
    if predicate.op in CART_OPS:
        return None
    memo = table("passing")
    if memo is None:
        return [
            v.id
            for p in pool
            for v in p.variants
            if (v.available or not available_only) and holds(predicate, p, v, quantity=quantity) is True
        ]
    key = _key(predicate, quantity)
    shelf = tuple(pool)
    slot = (key, available_only, tuple(map(id, shelf)))
    hit = memo.get(slot)
    if hit is None:
        found = _indexed(predicate, shelf, quantity, memo)
        if found is None:
            found = [v for p in shelf for v in _passing(predicate, p, quantity, key, memo)]
        hit = memo[slot] = (shelf, [v.id for v in found if v.available or not available_only])
    ids: list[str] = hit[1]
    return list(ids)  # a copy: the caller owns the list it is given


def _indexed(
    predicate: Predicate, pool: tuple[Product, ...], quantity: int, memo: dict[Hashable, Any]
) -> list[Variant] | None:
    """The pool's variants that satisfy an eq or max_price predicate, in pool order, read without calling `holds`
    per variant (the same checks, applied to a whole pool at once); None for other ops. These two ops change value
    with every draft, so the per-product memo rarely answers them."""
    if predicate.op == "eq" and predicate.value is not None:
        index = _eq_index(predicate.field, predicate.option, pool, memo)
        return index.get(eq_key(predicate.field, str(predicate.value)), [])
    if predicate.op == "max_price":
        budget = _budget(predicate.value)
        if budget is None:
            return []
        amount, currency = budget
        units = max(1, quantity)
        return [
            v
            for p in pool
            for v in p.variants
            if v.price is not None
            and not (currency and v.currency and currency != v.currency)
            and v.price * units <= amount
        ]
    return None


def _eq_index(
    field: str, option: str | None, pool: tuple[Product, ...], memo: dict[Hashable, Any]
) -> dict[str, list[Variant]]:
    """The pool's variants by the `eq_key` of their option for `field`, each list in pool order; a variant without
    the option is in none (its check says None)."""
    slot = ("eq-index", field, option, tuple(map(id, pool)))
    hit = memo.get(slot)
    if hit is None:
        names = option_names(field, option)
        index: dict[str, list[Variant]] = {}
        for product in pool:
            for variant in product.variants:
                found = variant.option(*names)
                if found is not None:
                    index.setdefault(eq_key(field, found), []).append(variant)
        hit = memo[slot] = (pool, index)
    built: dict[str, list[Variant]] = hit[1]
    return built


def passing(predicate: Predicate, product: Product, *, quantity: int = 1) -> tuple[Variant, ...]:
    """The variants of `product` for which `holds` is True, in the product's order. Within a task generation run
    (carttrace.agents.memo.remembering) the answer is computed once per predicate and product."""
    memo = table("passing")
    if memo is None:
        return tuple(v for v in product.variants if holds(predicate, product, v, quantity=quantity) is True)
    return _passing(predicate, product, quantity, _key(predicate, quantity), memo)


_PredicateKey = tuple[str, str, str | None, str, int]


def _key(predicate: Predicate, quantity: int) -> _PredicateKey:
    """Everything `holds` reads of a predicate, with the quantity: two predicates with one key get one answer."""
    value = json.dumps(predicate.value, sort_keys=True, separators=(",", ":"))
    return (predicate.op, predicate.field, predicate.option, value, quantity)


def _passing(
    predicate: Predicate, product: Product, quantity: int, key: _PredicateKey, memo: dict[Hashable, Any]
) -> tuple[Variant, ...]:
    slot = (key, id(product))
    hit = memo.get(slot)
    if isinstance(hit, tuple) and hit[0] is product:
        found: tuple[Variant, ...] = hit[1]
        return found
    found = tuple(v for v in product.variants if holds(predicate, product, v, quantity=quantity) is True)
    memo[slot] = (product, found)  # the reference keeps id(product) from being reused while the run lasts
    return found


def best(predicate: Predicate, pairs: Sequence[tuple[Product, Variant]]) -> list[str]:
    """The variant ids among `pairs` that a superlative predicate picks, in their order: the cheapest, the cheapest
    per unit, or the most or least of a nutrient per 100 g. Ties are all kept; a variant whose data does not state
    the measure is never picked. Empty for any other op."""
    measure = _MEASURES.get(predicate.op)
    if measure is None:
        return []
    scored = [(v.id, score) for p, v in pairs if (score := measure(predicate, p, v)) is not None]
    if not scored:
        return []
    pick = max if predicate.op == "most" else min
    top = pick(score for _, score in scored)
    return [vid for vid, score in scored if score == top]


def _price(predicate: Predicate, product: Product, variant: Variant) -> Fraction | None:
    return None if variant.price is None else Fraction(variant.price)


def _per_unit(predicate: Predicate, product: Product, variant: Variant) -> Fraction | None:
    return unit_price(product, variant)


def _nutrient(predicate: Predicate, product: Product, variant: Variant) -> Fraction | None:
    value = predicate.value
    name = value.get("nutrient") if isinstance(value, Mapping) else None
    return nutrient(product, name) if isinstance(name, str) else None


_MEASURES: Mapping[str, Callable[[Predicate, Product, Variant], Fraction | None]] = {
    "cheapest": _price,
    "cheapest_per_unit": _per_unit,
    "most": _nutrient,
    "least": _nutrient,
}


def kind_matches(product: Product, noun: str) -> bool:
    """Whether the product is the kind of thing the shopper's noun names: the same noun, or the noun (or its last
    word, so "pain relief tablet" matches "Ibuprofen 200mg Tablets") in its title, type or category."""
    wanted = fold(noun)
    if noun_of(product).singular == wanted:
        return True
    head = wanted.removeprefix("pair of ").rsplit(" ", 1)[-1]
    labels = [product.title, product.product_type or "", *product.categories]
    forms = {wanted, plural_of(wanted), head, plural_of(head), singular_of(head)}
    return any(has_phrase(label, form) for label in labels for form in forms)


def mentions(product: Product, variant: Variant, phrase: str) -> bool:
    """Whether the product's title, type, tags, the variant's options or the description mention `phrase`."""
    places: Iterable[str] = (
        product.title,
        product.product_type or "",
        variant.title,
        *product.tags,
        *(value for _, value in variant.options),
        *product.lines,
    )
    return any(has_phrase(_unabbreviated(place), phrase) for place in places)


_ABBREVIATION = re.compile(r"\b(?:[A-Za-z]\.){2,}")
"""Initials with full stops ("U.S.", "U.K."), which a shopper writes without them ("us-grown hemp")."""


def _unabbreviated(text: str) -> str:
    """The text with each run of dotted initials closed up: "U.S.-grown hemp" reads "US-grown hemp". Of four
    lotions made with U.S.-grown hemp, only the one that spelled it "US-grown" was a witness (2026-10-16)."""
    return _ABBREVIATION.sub(lambda m: m.group(0).replace(".", ""), text) if "." in text else text


def option_names(field: str, store_name: str | None) -> tuple[str, ...]:
    names = OPTION_NAMES.get(field, (field.replace("_", " "),))
    return (store_name, *names) if store_name else names

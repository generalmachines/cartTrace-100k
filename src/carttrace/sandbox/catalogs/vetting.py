"""Which pooled products a stand-in store may sell, and at what price: checked once, when the pools are built (D104).

Why at pool build. A release of 20 open-lane traces (2026-10-16) served what the pools held unchecked: Lazada
descriptions as raw JSON ('{"styleType":"offer-type-1","items":...}'), a rubber car floor mat at $399.99 and a nasal
spray at $299.99 (synthetic prices drawn from a whole category's band), and products under a category that is not
what they are (a party tent among bags, paper towels among groceries). Every stand-in is cut from the pools, so a
product is vetted there once, and every drop is counted per reason in the category's summary.

The checks, per product:
- **description**: Lazada's JSON widgets are removed and its sectioned JSON read as plain lines
  (`plain_description`); a description that still holds JSON drops the product;
- **kind** (carttrace.sandbox.catalogs.kinds): what the product's own title says it is;
- **category**: a product whose kind is common in the pools (at least `MISMATCH_MIN` products) but almost never in its
  own category (under `MISMATCH_BP` of that kind's products) is dropped as filed under the wrong category;
- **price**: a synthetic price (ABO, WANDS) is drawn again inside its kind's band, so a floor mat costs what floor
  mats cost.

A real price (Lazada's pesos) stays in the pool, so a home-market store still sells it as the source did; a mixed US
store prices it with `Pricer`: the product's rank among its category's Lazada products of its kind, placed in that
kind's US band. Converted at the exchange rate alone, Lazada's prices are a Philippine marketplace's (a median shirt
at $4, a median sofa at $15: 2026-10-16, 81,636 products sampled), which a US store does not charge; the rank keeps
which listing is the cheap one and which the premium one, and the band keeps every price one a US shopper would
believe.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from carttrace.core.money import to_minor
from carttrace.sandbox.catalogs.kinds import Kind, Kinds
from carttrace.sandbox.catalogs.records import ProductRecord
from carttrace.sandbox.catalogs.synthetic import PriceRule, price_note, synthetic_price

VET_RULE: Final = "vet.v1"
"""Name of the checks; change it (v2) whenever a check or a threshold changes: it is part of the pools' key."""
MISMATCH_MIN: Final = 100
"""Products of a kind across the pools before its spread over categories is evidence."""
MISMATCH_BP: Final = 100
"""A kind under this share (basis points) of its products in a category is not that category's."""
MIN_RANKED: Final = 5
"""Products of a kind a category needs before a real price is ranked among its kind alone, not the whole category."""

JSON_DESCRIPTION: Final = "description is raw JSON"
WRONG_CATEGORY: Final = "its kind is not sold in this category"
_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class Verdict:
    record: ProductRecord | None
    """The product as the pool keeps it, or None when it is dropped."""
    reason: str | None = None
    kind: Kind | None = None


@dataclass(frozen=True, slots=True, eq=False)
class Vetting:
    kinds: Kinds
    rule: PriceRule
    spread: Mapping[str, Counter[str]]
    """Kind head to category to products, over every member's pooled products."""

    @classmethod
    def of(cls, records: Iterable[ProductRecord], kinds: Kinds, rule: PriceRule) -> Vetting:
        spread: dict[str, Counter[str]] = {}
        for record in records:
            kind = kind_of(kinds, record)
            if kind is not None:
                spread.setdefault(kind.head, Counter())[record.category] += 1
        return cls(kinds, rule, spread)

    def check(self, record: ProductRecord) -> Verdict:
        description = plain_description(record.description)
        if description is None:
            return Verdict(None, JSON_DESCRIPTION)
        derived = dict(record.provenance.derived)
        if description != record.description:
            derived["description"] = f"cleaned to plain text ({VET_RULE})"
        kind = kind_of(self.kinds, record)
        if kind is not None and self._misfiled(kind, record.category):
            return Verdict(None, WRONG_CATEGORY, kind)
        variants = record.variants
        synthetic = dict(record.provenance.synthetic)
        if kind is not None and "price" in synthetic and record.currency == self.rule.currency:
            price = synthetic_price(self.rule, kind.band, source=record.source, product_id=record.id)
            variants = [v.model_copy(update={"price": price}) for v in record.variants]
            synthetic["price"] = f"{price_note(self.rule, kind.band)}, the band of a {kind.phrase} ({self.kinds.rule})"
        provenance = record.provenance.model_copy(update={"synthetic": synthetic, "derived": derived})
        kept = record.model_copy(update={"description": description, "variants": variants, "provenance": provenance})
        return Verdict(kept, None, kind)

    def _misfiled(self, kind: Kind, category: str) -> bool:
        spread = self.spread.get(kind.head)
        if spread is None:
            return False
        total = sum(spread.values())
        return total >= MISMATCH_MIN and spread[category] * 10_000 < total * MISMATCH_BP


_OFFER_BLOCK = re.compile(r'\{\\?"styleType\\?"[^{}]*\}')
"""Lazada's image-map widget, served inside descriptions as raw JSON ('{"styleType":"offer-type-1","items":...}')."""
_SECTION_LIST = re.compile(r'\[\{\\?"content\\?".*?\}\]', re.DOTALL)
"""Lazada's sectioned description, a JSON list of {"content", "title"} objects served as text."""
_JSON_LEFT = re.compile(r'[{\[]\s*\\?"[A-Za-z_]+\\?"\s*:|\\?"\s*:\s*\\?"')
"""What is left of JSON after cleaning: an object key, or a "key":"value" pair."""


def plain_description(text: str | None) -> str | None:
    """A description as plain text: Lazada's raw JSON widgets removed, its sectioned JSON read as "Title: content"
    lines, escaped quotes and newlines unescaped, tags removed. None when JSON is still left after that, so the pool
    drops the product rather than serve "{\"styleType\":...}" to an agent (D104, a published release served it)."""
    if not text:
        return text
    out = _OFFER_BLOCK.sub(" ", text)
    out = _SECTION_LIST.sub(lambda m: _sections(m.group(0)), out)
    out = out.replace('\\"', '"').replace("\\n", "\n").replace("\\t", " ").replace("\\/", "/")
    out = "\n".join(" ".join(_TAG.sub(" ", html.unescape(line)).split()) for line in out.splitlines())
    out = "\n".join(line for line in out.splitlines() if line).strip()
    if _JSON_LEFT.search(out):
        return None
    return out


def _sections(raw: str) -> str:
    try:
        items = json.loads(raw.replace('\\"', '"'))
    except ValueError:
        try:
            items = json.loads(raw)
        except ValueError:
            return raw
    lines = []
    for item in items if isinstance(items, list) else ():
        if isinstance(item, dict) and isinstance(item.get("content"), str):
            title = str(item.get("en_title") or item.get("title") or "").strip()
            lines.append(f"{title}: {item['content'].strip()}" if title else item["content"].strip())
    return "\n".join(lines)


def kind_of(kinds: Kinds, record: ProductRecord) -> Kind | None:
    """The record's kind, its category leaf read as its type (D104)."""
    leaf = record.category_path[-1] if record.category_path else ""
    return kinds.of(record.title, [leaf] if leaf else [])


@dataclass(frozen=True, slots=True, eq=False)
class Pricer:
    """US prices for a category's real-price products in a mixed store (D104): a product's source price ranked among
    the category's products of its kind from the same source (among all of that source's products in the category
    when its kind has fewer than `MIN_RANKED`, or it has none), and the rank placed on the ladder points of its kind's
    band (its category's band when it has no kind). Integer arithmetic only: the k-th of n products, ties counted at
    their middle, takes the ladder point at k/n of the band's points."""

    kinds: Kinds
    rule: PriceRule
    ranked: Mapping[tuple[str, str], tuple[int, ...]]
    """(source, kind head or "") to the sorted lowest prices of that source's products."""

    @classmethod
    def of(cls, records: Iterable[ProductRecord], kinds: Kinds, rule: PriceRule) -> Pricer:
        found: dict[tuple[str, str], list[int]] = {}
        for record in records:
            if record.currency == rule.currency:
                continue
            low = min(v.price for v in record.variants)
            kind = kind_of(kinds, record)
            found.setdefault((record.source, ""), []).append(low)
            if kind is not None:
                found.setdefault((record.source, kind.head), []).append(low)
        return cls(kinds, rule, {k: tuple(sorted(v)) for k, v in found.items()})

    def price(self, record: ProductRecord, category_band: tuple[int, int]) -> tuple[int, str]:
        """The product's price in `rule.currency` minor units, and the note provenance keeps."""
        kind = kind_of(self.kinds, record)
        band = kind.band if kind is not None else category_band
        own = self.ranked.get((record.source, kind.head)) if kind is not None else None
        by_kind = own is not None and len(own) >= MIN_RANKED
        among = own if own is not None and by_kind else self.ranked.get((record.source, ""), ())
        low = min(v.price for v in record.variants)
        below = sum(1 for p in among if p < low)
        equal = sum(1 for p in among if p == low)
        lo, hi = (to_minor(b, self.rule.currency) or 0 for b in band)
        points = [p for p in self.rule.ladder if lo <= p <= hi] or [min(self.rule.ladder, key=lambda p: abs(p - lo))]
        n = max(1, len(among))
        index = min(len(points) - 1, ((2 * below + equal) * len(points)) // (2 * n))
        what = f"a {kind.phrase}" if kind is not None else "its category"
        scope = f"{record.source}'s {kind.head}s" if kind is not None and by_kind else record.source
        note = (
            f"{record.source}'s own price ranked among {len(among)} of {scope} in the store's pool category, placed in "
            f"the US band of {what}, {band[0]}-{band[1]} {self.rule.currency} ({self.kinds.rule}, {VET_RULE})"
        )
        return points[index], note


def dropped(verdicts: Iterable[Verdict]) -> dict[str, int]:
    """Dropped products per reason, sorted."""
    return dict(sorted(Counter(v.reason for v in verdicts if v.reason is not None).items()))


__all__ = [
    "JSON_DESCRIPTION",
    "VET_RULE",
    "WRONG_CATEGORY",
    "Pricer",
    "Verdict",
    "Vetting",
    "dropped",
    "kind_of",
    "plain_description",
]

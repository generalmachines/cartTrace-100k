"""A stand-in's products picked to look like the store it stands in for (D104): a rule match on X's captured catalog.

Why. A publishable stand-in for real store X (D85, D91) drew its 40 products from the category's pool spread over
product types, whatever X sells: a stand-in for a tea store held coffee syrup, crab cakes and frozen potatoes. The
research of 2026-10-16 (docs/research/2026-10-16-open-fidelity.md, section 4) hand-checked five stores: 2 of 25
sampled products were the kind of product X sells, in X's price band. A rule pick from X's capture got 20 of 25, and a
decision-model rerank of the rule's shortlist 21 of 25 at 120 model calls a store, so the rule stands alone.

The rule. X's profile is read from its newest capture (carttrace.trace.capture_v2): the words of its product titles, product
types and tags (stop words and X's own brand words dropped: its domain and vendors; a tag word only where a title or
type uses it too, since stores tag for their own tools; a word on nearly every product is boilerplate), each with how
many of X's products use it, and X's price band, the 10th to 90th percentile of its variants' prices in rough US
dollars. An open product scores the sum, over its words that X uses, of log(1 + X's count) times the word's rarity
in the category's pool (log((n + 1) / (pool count + 1))), divided by the square root of its word count plus one; in
X's band it scores x1.5, outside it x0.75. Its words are its title, the last step of its category path and Lazada's
`Product type` attribute. Its price is its own (Lazada's pesos at a rough rate, ABO's and WANDS' synthetic dollars),
not the repriced one a mixed store shows: the band asks whether the product costs what X's products cost.

Variety under D91's caps. Each member still gives its `member_quotas` share. Within a member, products are ranked by
score (ties by id), titles that repeat are dropped (Lazada lists one product under many sellers), and the store's n
products are a seeded sample of the top MATCH_DEPTH x n. The depth was measured (2026-10-16) on the stand-ins the v1
target would build: per category the up to 25 queue stores `choose_open` takes, each matched to its newest capture.
Taking the top n put one product in 13 haircare-fragrance and 11 home-decor-textiles stand-ins, over the ten-repeat
limit; 1.5 x n one skincare product in 11; 2 x n kept every product of every category in at most 9 (expected at most
8.0), while a shortlist deep enough for the worst case (every stand-in of a category alike: 4 x n) halved how many
sampled picks were the kind of product their store sells. Deterministic: the same pool, profile, seed and size give
the same products.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from carttrace.core.errors import NotFoundError
from carttrace.core.ids import seeded_rng
from carttrace.sandbox.catalogs.records import ProductRecord
from carttrace.trace.capture_v2 import CaptureV2, find_capture_v2, newest_captures_v2, table_rows

MATCH_RULE: Final = "match.v1"
"""The pick's rule name in a stand-in's provenance and experiment.json; change it (v2) when the scoring changes."""
SPREAD_RULE: Final = "spread"
"""Today's pick (`pools.pick`), for a stand-in whose store has no capture to match."""
USD_RATES: Final[Mapping[str, float]] = {
    "USD": 1.0,
    "PHP": 1 / 56.0,
    "GBP": 1.27,
    "EUR": 1.08,
    "CAD": 0.73,
    "AUD": 0.66,
    "NZD": 0.6,
    "SGD": 0.74,
    "JPY": 0.0067,
    "THB": 0.028,
    "INR": 0.012,
    "MYR": 0.21,
    "HKD": 0.13,
    "SEK": 0.095,
    "DKK": 0.145,
    "NOK": 0.093,
    "CHF": 1.12,
    "MXN": 0.055,
}
"""Rough rates to set prices side by side (X's dollars or pounds against Lazada's pesos): a band, never a price."""
IN_BAND: Final = 1.5
OUT_OF_BAND: Final = 0.75
_STOP_WORDS: Final = (
    "the and for with from this that your you are our new set pack size color colour black white one all free "
    "gift each per inch cm mm ml pcs piece pieces item items product products men women unisex kids edition style"
)
STOP: Final = frozenset(_STOP_WORDS.split())
_WORD: Final = re.compile(r"[a-z][a-z]{2,}")
_TITLE_KEY: Final = re.compile(r"\W+")
BOILERPLATE_SHARE: Final = 0.9
BOILERPLATE_MIN: Final = 20
"""A word on more than 90% of a store's products (of 20 or more) is dropped: one store's product types all read "... Hide
- Not Regular - Only"; a tea store's "tea" (72% of its products) stays."""
MATCH_DEPTH: Final = 2.0
"""A member's n products are sampled from its top MATCH_DEPTH x n (see the module docstring for the measurement)."""
MATCHED_ON: Final = "the words of the store's product titles, types and tags, and its price band"


@dataclass(frozen=True, slots=True)
class StoreProfile:
    """What a stand-in is matched on: X's words with how many of its products use each, and its price band."""

    store_id: str
    capture_id: str
    products: int
    words: Mapping[str, int]
    drop: frozenset[str]
    """X's brand words: its domain's and its vendors', never matched (an open product does not carry X's brand)."""
    band: tuple[float, float] | None
    """X's 10th to 90th percentile price in rough US dollars; None when no price is in a known currency."""

    def basis(self) -> dict[str, Any]:
        """The profile as the stand-in's provenance records it (`pick`): the rule and the capture it was read from.
        X's own words and prices stay out: a stand-in is publishable (D85) and a capture's content never is."""
        return {"rule": MATCH_RULE, "store": self.store_id, "capture": self.capture_id, "products": self.products,
                "on": MATCHED_ON}  # fmt: skip


def words_of(text: str, drop: Iterable[str] = ()) -> set[str]:
    dropped = set(drop)
    return {w for w in _WORD.findall(text.lower()) if w not in STOP and w not in dropped}


def usd(amount_minor: int, currency: str | None) -> float | None:
    rate = USD_RATES.get((currency or "").upper())
    return amount_minor / 100 * rate if rate is not None else None


def profile_of(capture: CaptureV2) -> StoreProfile | None:
    """X's profile from its capture's products and variants tables; None when the capture holds no titled product."""
    products = table_rows(capture, "products")
    if not any(p.get("title") for p in products):
        return None
    drop = set(_WORD.findall(capture.domain.lower().removeprefix("www.").split(".")[0]))
    for p in products:
        drop |= set(_WORD.findall(str(p.get("vendor") or "").lower()))
    counts: Counter[str] = Counter()
    tagged: Counter[str] = Counter()
    for p in products:
        named = words_of(f"{p.get('title') or ''} {p.get('product_type') or ''}", drop)
        counts.update(named)
        tagged.update(words_of(" ".join(map(str, p.get("tags") or [])), drop) - named)
    # a tag word counts only where X's titles or types use it too: stores tag for their own tools ("YGroup_tees",
    # "admin-hide", "member-tier-discount"), words an open title meets by chance
    counts.update({w: n for w, n in tagged.items() if w in counts})
    if len(products) >= BOILERPLATE_MIN:
        # a word on nearly every product is the store's boilerplate ("Hide - Not Regular - Only"), not its kind
        cut = BOILERPLATE_SHARE * len(products)
        counts = Counter({w: n for w, n in counts.items() if n <= cut})
    prices: dict[str, list[float]] = {}
    for v in table_rows(capture, "variants"):
        minor = v.get("price_minor")
        if isinstance(minor, int) and minor > 0:
            found = usd(minor, v.get("currency"))
            if found is not None:
                prices.setdefault(str(v.get("product_id")), []).append(found)
    medians = sorted(statistics.median(xs) for xs in prices.values())
    band = (medians[len(medians) // 10], medians[(9 * len(medians)) // 10]) if medians else None
    return StoreProfile(
        store_id=capture.store_id,
        capture_id=capture.id,
        products=len(products),
        words=dict(sorted(counts.items())),
        drop=frozenset(drop),
        band=band,
    )


def newest_profile_capture(root: Path, store_id: str, *, category: str | None = None) -> CaptureV2 | None:
    """The newest capture of the store that a profile can be read from (complete, with products), or None."""
    return newest_captures_v2(root, store_ids=[store_id], category=category).get(store_id)


def load_profile(root: Path, capture_id: str) -> StoreProfile | None:
    """The profile of a named capture (a world's `pick_from`); None when it holds no titled product. Raises
    NotFoundError when the capture is gone, so a world is never rebuilt on another pick."""
    try:
        capture = find_capture_v2(root, capture_id)
    except NotFoundError as exc:
        raise NotFoundError(
            f"capture {capture_id}, which a stand-in's products were matched to, is gone: {exc.message}",
            hint="build a new manifest; the stand-in will match the store's newest capture",
        ) from None
    return profile_of(capture)


def open_text(record: ProductRecord) -> str:
    path = record.category_path
    return " ".join([record.title, str(path[-1]) if path else "", record.attributes.get("Product type", "")])


def own_usd(record: ProductRecord) -> float | None:
    v = record.variants[0]
    return usd(v.price, v.currency)


@dataclass(frozen=True, slots=True)
class Scorer:
    """Scores a category's pooled products against one profile; the pool's word counts are read once."""

    profile: StoreProfile
    pool_counts: Mapping[str, int]
    pool_size: int

    @classmethod
    def of(cls, profile: StoreProfile, pool: Sequence[ProductRecord]) -> Scorer:
        counts: Counter[str] = Counter()
        for r in pool:
            counts.update(words_of(open_text(r), profile.drop))
        return cls(profile, dict(counts), len(pool))

    def score(self, record: ProductRecord) -> float:
        words = words_of(open_text(record), self.profile.drop)
        x = self.profile.words
        n = self.pool_size
        total = sum(
            math.log1p(x[w]) * math.log((n + 1) / (self.pool_counts.get(w, 0) + 1)) for w in sorted(words) if w in x
        )
        total /= math.sqrt(len(words) + 1)
        band = self.profile.band
        price = own_usd(record)
        if band is not None and price is not None:
            total *= IN_BAND if band[0] <= price <= band[1] else OUT_OF_BAND
        return total


def ranked(records: Sequence[ProductRecord], scorer: Scorer) -> list[ProductRecord]:
    """The records by score (highest first, ties by id), each title once."""
    scored = sorted(((scorer.score(r), r.id, r) for r in records), key=lambda t: (-t[0], t[1]))
    seen: set[str] = set()
    out: list[ProductRecord] = []
    for _, _, r in scored:
        key = _TITLE_KEY.sub(" ", r.title.lower()).strip()[:60]
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def shortlist_size(n: int, *, depth: float = MATCH_DEPTH) -> int:
    """How deep in the ranking a member's n products are sampled from."""
    return max(n, math.ceil(depth * n))


def match_pick(
    records: Sequence[ProductRecord],
    scorer: Scorer,
    *,
    category: str,
    seed: int,
    size: int,
    depth: float = MATCH_DEPTH,
) -> list[ProductRecord]:
    """`size` of the records most like the profile: a seeded sample of the top `shortlist_size` (deduplicated
    titles). Sorted by id."""
    if size <= 0:
        return []
    order = ranked(records, scorer)
    shortlist = order[: min(len(order), shortlist_size(size, depth=depth))]
    rng = seeded_rng("openpool-match", category, seed, size, scorer.profile.capture_id)
    picked = rng.sample(shortlist, min(size, len(shortlist)))
    return sorted(picked, key=lambda r: r.id)


__all__ = [
    "MATCH_DEPTH",
    "MATCH_RULE",
    "SPREAD_RULE",
    "USD_RATES",
    "Scorer",
    "StoreProfile",
    "load_profile",
    "match_pick",
    "newest_profile_capture",
    "open_text",
    "profile_of",
    "ranked",
    "shortlist_size",
    "usd",
    "words_of",
]

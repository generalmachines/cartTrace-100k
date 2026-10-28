"""A generic Magento-like platform pack, from the Magento 2.4.8 code reading in docs/SANDBOX-GAPS.md (decision D42).

Why generic and why marked. Magento was read at pinned tags, not run as a store, so nothing here was seen on a wire:
the behaviours below are what the code says Magento does, and each names the file and lines it was read at
(`M` is `https://github.com/magento/magento2/blob/2.4.8/app/code/Magento/`). What the notes do not settle is marked
`unverified` in `EVIDENCE` and kept as small as the reading allows; the UCP module the notes read (ooasis 0.1.3)
was read for its error mapping only, so the pack keeps the sandbox's own UCP shapes and adds Magento's behaviour
behind them.

What the pack does:
- configurable products need their options chosen: a line naming the parent product is refused in Magento's words
  (core rule `options_required`);
- cart price rules apply by priority, and a rule that stops processing ends the run; one code per cart (core rules
  `ordered_rules`, `stack_codes=False`);
- free shipping is judged on the merchandise after discounts (core rule `free_after_discount`);
- layered navigation: a search answers with facet counts per option value over sellable variants, empty values
  dropped, and automatic price ranges (`facets`, `price_ranges` beside the products).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from carttrace.sandbox.platforms.spec import pack_spec

M = "https://github.com/magento/magento2/blob/2.4.8/app/code/Magento/"

SPEC = pack_spec("magento")
"""The pack's spec (data/sandbox/packs/magento.yaml): every fact below is read from it (D101)."""

EVIDENCE: Mapping[str, str] = MappingProxyType(SPEC.evidence())
"""Each behaviour and the Magento source it was read in, or why it is unverified."""

WORDING: Mapping[str, str] = MappingProxyType(dict(SPEC.wording))
"""Magento's words (ConfigurableProduct Configurable.php#L1022)."""

RANGE_FLOOR = 10
"""Magento's automatic ranges never go below this step in major units (Auto.php: MIN_RANGE_POWER)."""


def facets(option_values: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    """Layered navigation facets over the sellable variants a search found, one entry per variant's options: each
    option name with its values and how many products carry each (a product counts once per value), values with no
    product dropped, names and values in first-seen order (Attribute.php#L94-L140)."""
    counts: dict[str, dict[str, int]] = {}
    for options in option_values:
        for name, value in options.items():
            if value:
                counts.setdefault(name, {})
                counts[name][value] = counts[name].get(value, 0) + 1
    return [
        {"name": name, "values": [{"label": label, "count": n} for label, n in values.items() if n > 0]}
        for name, values in counts.items()
        if values
    ]


def price_ranges(prices: Sequence[int], currency_exponent: int = 2) -> list[dict[str, int]]:
    """Automatic price ranges over the products' prices (minor units): a step of ten to the power of the highest
    price's digit count less one, in major units, lowered while it gives fewer than two ranges and stays above the
    floor (Auto.php#L106-L117); each range with its product count, empty ones dropped."""
    if not prices:
        return []
    unit = 10**currency_exponent
    top = max(prices) / unit
    power = max(1, len(str(int(top))) - 1)
    step = 10**power
    while step > RANGE_FLOOR and len({p // (step * unit) for p in prices}) < 2:
        step //= 10
    buckets: dict[int, int] = {}
    for p in prices:
        start = p // (step * unit)
        buckets[start] = buckets.get(start, 0) + 1
    width = step * unit
    return [
        {"from": start * width, "to": (start + 1) * width, "count": n} for start, n in sorted(buckets.items()) if n > 0
    ]

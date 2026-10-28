"""Train, validation and locked test splits, assigned per merchant group by hash.

Why by hash of the group. A split must be decided before any task exists, must never move a store once traces are
collected, and must keep every store of one business on the same side (see `carttrace.stores.merchant`). Hashing the merchant
group id with a versioned salt gives all three: no state to keep, the same answer on every machine, and one group,
one split. The shares follow the sampling plan's 2,800 / 400 / 800 groups of 4,000, that is 70 / 10 / 20; on a few
hundred groups the realised shares drift by a few points, which is the price of never moving a group to rebalance.
"""

from __future__ import annotations

from carttrace.core.canon import digest
from carttrace.stores.model import Split

SPLIT_SALT = "mc.stores.split.v1"
"""Changing the salt reshuffles every group; it is versioned so that can only happen on purpose."""
SPLIT_SHARES: tuple[tuple[Split, int], ...] = (("train", 7000), ("validation", 1000), ("test", 2000))
"""Basis points of the hash space per split, in order."""

if sum(share for _, share in SPLIT_SHARES) != 10_000:
    raise ValueError("SPLIT_SHARES must cover exactly 10,000 basis points")


def split_for(merchant_group_id: str) -> Split:
    """The split of a merchant group: the same group id always gives the same split."""
    bucket = int(digest([SPLIT_SALT, merchant_group_id], 16), 16) % 10_000
    edge = 0
    for name, share in SPLIT_SHARES:
        edge += share
        if bucket < edge:
            return name
    return SPLIT_SHARES[-1][0]

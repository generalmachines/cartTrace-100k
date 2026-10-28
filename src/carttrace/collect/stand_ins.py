"""More than one open stand-in for a queue store, where a category holds fewer queue stores than its stand-ins (D119).

Why. `choose_open` takes each open twin from a distinct queue store of its category, and gifts-collectibles holds 16
queue stores for the 40 stand-ins (100 rows each) of its 4,000 traces, so it could reach 1,600; every other category
holds 96 to 196. A category short of stores now takes a second, third, ... stand-in of the same stores. A later
stand-in is a store of its own everywhere an id is read: its world's seed and so its product draw from the category
pool, its presented identity (D106: the stand-in domain is drawn from the world's seed) and its spec (D107: dealt by
its place on the category's roster) follow from its id, and nothing of the queue store's capture is keyed by it (D104's
match, D100's recorded view), so it is drawn by the spread pick and never shares the first stand-in's products
wholesale or its identity. The release numbers it on the roster as a store of its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

from carttrace.stores.model import Store

STAND_IN_MARK: Final = "~"
"""Between a queue id and a stand-in's index: store-388~2 is the second stand-in for store-388."""


def stand_in_id(store_id: str, index: int) -> str:
    """The id of the `index`-th open stand-in for queue store `store_id`: the queue id itself for the first, the id
    and the index after STAND_IN_MARK for every later one."""
    return store_id if index <= 1 else f"{store_id}{STAND_IN_MARK}{index}"


def stand_in_of(store_id: str) -> tuple[str, int]:
    """(queue id, stand-in index) of a store id: (id, 1) for a queue id, ("store-388", 2) for store-388~2."""
    base, mark, index = store_id.rpartition(STAND_IN_MARK)
    if mark and base and index.isdigit() and int(index) >= 2:
        return base, int(index)
    return store_id, 1


def stand_in_index(store_id: str) -> int:
    return stand_in_of(store_id)[1]


def stand_ins(stores: Sequence[str], wanted: int) -> list[str]:
    """`stores` (a category's distinct queue stores, in their order) and, when they are fewer than `wanted`, second,
    third, ... stand-ins of them in the same order until `wanted` are named. Deterministic: the same stores give the
    same ids, and a category with `wanted` stores or more is unchanged."""
    out = list(dict.fromkeys(stores))
    base = list(out)
    index = 2
    while base and len(out) < wanted:
        out += [stand_in_id(s, index) for s in base][: wanted - len(out)]
        index += 1
    return out


def with_stand_ins(stores: Sequence[Store], wanted: Iterable[str]) -> list[Store]:
    """The queue with a record for each later stand-in `wanted` names, right after its queue store: the store's own
    record under the stand-in's id (its category, market and domain: the roster store it stands in for)."""
    by_id = {s.id: s for s in stores}
    extra: dict[str, list[Store]] = {}
    for sid in dict.fromkeys(wanted):
        base, index = stand_in_of(sid)
        if index > 1 and sid not in by_id and base in by_id:
            extra.setdefault(base, []).append(by_id[base].model_copy(update={"id": sid}))
    if not extra:
        return list(stores)
    out: list[Store] = []
    for store in stores:
        out.append(store)
        out += sorted(extra.get(store.id, []), key=lambda s: stand_in_of(s.id)[1])
    return out


__all__ = ["STAND_IN_MARK", "stand_in_id", "stand_in_index", "stand_in_of", "stand_ins", "with_stand_ins"]

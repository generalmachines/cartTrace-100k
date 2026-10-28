"""Remembering per-product answers for the length of one task generation run.

Why a run-scoped memo. Writing a store's tasks checks every predicate of every draft against every variant on the
shelf, and a replica shelf can hold thousands of variants of one noun: the same kind, option and detail checks ran
millions of times, and one store's manifest took over fifteen minutes (2026-10-13). The answers are pure functions
of the predicate and the product, so a run computes each once. Products are not hashable (their metadata is a
dict), so an answer is keyed by the product object itself and the memo holds a reference to it: the object cannot be
freed while the run lasts, so its id cannot be reused by another product. Outside `remembering()` nothing is
remembered, and every caller computes exactly as before.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_ACTIVE: ContextVar[dict[str, dict[Hashable, Any]] | None] = ContextVar("mc_agents_memo", default=None)


@contextmanager
def remembering() -> Iterator[None]:
    """Remember answers until the block ends. A nested block shares the outer run's memo."""
    if _ACTIVE.get() is not None:
        yield
        return
    token = _ACTIVE.set({})
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def table(name: str) -> dict[Hashable, Any] | None:
    """The running memo's table called `name`, created on first use; None outside `remembering()`."""
    tables = _ACTIVE.get()
    if tables is None:
        return None
    return tables.setdefault(name, {})


def per_product[T](name: str, product: object, key: Hashable, compute: Callable[[], T]) -> T:
    """`compute()`, a pure reading of `product` (and `key`), remembered for that product object within a run; outside
    one it is computed every time. The memo holds the product, so its id is not reused while the run lasts."""
    memo = table(name)
    if memo is None:
        return compute()
    slot = (id(product), key)
    hit = memo.get(slot)
    if hit is None or hit[0] is not product:
        hit = memo[slot] = (product, compute())
    found: T = hit[1]
    return found

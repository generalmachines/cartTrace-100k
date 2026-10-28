"""Decisive details: the facts in a store's product data that decide a task, one reader per kind of fact.

- `base`: the Detail contract and the Proposal a detail makes for a task
- `size_chart`: size charts as text, a body measurement to a size (clothing, footwear, kids, sportswear)
- `allergens`: ingredient lists and allergen statements (food, health, skin and hair care, pets, jewellery)
- `warnings`: label warnings against a condition (non-prescription health, skin care)

A category gets the details its hidden-spec fields allow (`details_for`). A new detail is a module with a class
that satisfies `Detail`, plus one line in carttrace.agents PLUGINS under "agents.detail"; task generation, the predicate
checks and the reference agent pick it up by name.
"""

from __future__ import annotations

from collections.abc import Collection

from carttrace.agents.details.allergens import AllergenDetail
from carttrace.agents.details.base import Detail, Proposal
from carttrace.agents.details.size_chart import SizeChartDetail
from carttrace.agents.details.warnings import WarningDetail
from carttrace.core.errors import NotFoundError

DETAIL_NAMES = ("size_chart", "allergens", "warnings")
"""Every detail, in the order a category prefers them when several apply."""


def detail(name: str) -> Detail:
    """The detail registered as `name` in carttrace.agents PLUGINS["agents.detail"]."""
    from carttrace.core.registry import resolve

    if name not in DETAIL_NAMES:
        raise NotFoundError(f"no decisive detail {name!r}", hint=f"choose one of {', '.join(DETAIL_NAMES)}")
    made: Detail = resolve("agents.detail", name)()
    return made


def details_for(fields: Collection[str]) -> list[Detail]:
    """The details that can decide tasks in a category with these hidden-spec fields."""
    return [d for d in (detail(n) for n in DETAIL_NAMES) if d.applies(fields)]


__all__ = [
    "DETAIL_NAMES",
    "AllergenDetail",
    "Detail",
    "Proposal",
    "SizeChartDetail",
    "WarningDetail",
    "detail",
    "details_for",
]

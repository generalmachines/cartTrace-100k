"""What a decisive detail is: a fact in a store's product data that decides whether a variant fits the shopper.

Why a contract. The benchmark asks, per category, whether a store exposes the detail an agent needs: a size chart
as text for clothing, ingredients for anyone with an allergy, warnings for anyone with a condition. One object per
detail knows how to read it from a product, how to check a shopper's value against it, and how to propose a value
for a task (one that some variant satisfies, or one that none does, for a deliberately impossible task). Task
generation, the reference agent and the predicate checks all go through the same object, so a task is never written
around a reading the reference agent would not make.

`check` answers True (the variant fits), False (it does not) or None (the product does not expose the detail). None
is never a pass: a witness must be shown to fit, and the reference agent records the detail as missing.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from random import Random
from typing import Protocol

from pydantic import JsonValue

from carttrace.agents.products import Product, Variant
from carttrace.agents.spec import Op


@dataclass(frozen=True, slots=True)
class Proposal:
    """A value the shopper can hold, and how they put it into words."""

    value: JsonValue
    say: str
    """The fact as a clause without a final stop, e.g. "my chest measures 102 cm"."""
    sentence: str
    """The fact as it appears in an opening, e.g. "My chest measures 102 cm, but I don't know my size here."."""
    slot: str
    """The fact as a category template's slot takes it: "chest 102 cm", "milk", "pregnant"."""
    aliases: tuple[str, ...] = ()
    """Fields a question may name when it is really about this fact (size, for a measurement)."""


class Detail(Protocol):
    @property
    def name(self) -> str:
        """size_chart, allergens or warnings: the name in the hidden spec's `decisive.detail`."""
        ...

    @property
    def field(self) -> str:
        """The hidden-spec field the detail decides: measurements, allergens or conditions."""
        ...

    @property
    def op(self) -> Op:
        """The predicate op it checks: fits, excludes or safe_for."""
        ...

    @property
    def missing(self) -> str:
        """What an agent records when a product does not expose the detail: size_chart, ingredients, warnings."""
        ...

    @property
    def forbidden_slots(self) -> frozenset[str]:
        """Template slots that would give the answer away (a stated size makes the size chart moot)."""
        ...

    @property
    def hypotheses(self) -> tuple[str, ...]:
        """Endings of the category hypothesis ids this detail tests, e.g. "size-chart-text"."""
        ...

    def applies(self, fields: Collection[str]) -> bool:
        """Whether a category with these hidden-spec fields can have tasks this detail decides."""
        ...

    def propose(self, target: Product, variant: Variant, shelf: Sequence[Product], rng: Random) -> Proposal | None:
        """A value `variant` satisfies and some comparable product on `shelf` (the products a shopper would weigh
        against it) fails, so the detail decides the task; None when the product does not expose the detail or no
        such value exists."""
        ...

    def impossible(self, pool: Sequence[Product], rng: Random) -> Proposal | None:
        """A value no variant in the pool satisfies, by what the pool's data shows; None when the data cannot show
        it."""
        ...

    def check(self, value: JsonValue, product: Product, variant: Variant) -> bool | None:
        """True when the variant fits the value, False when it does not, None when the product does not say."""
        ...

"""What task generation needs from a category: its slug, its hidden-spec fields, its intent templates and the ids
of its checklist hypotheses.

Why a shape and not carttrace.stores.Category. carttrace.stores sits beside carttrace.agents on layer 5, and siblings never import each
other (tests/test_layers.py). So generation takes anything with the attributes of `carttrace.stores.Category`, which the
runner (layer 6) passes as loaded and validated, or the category file itself as a mapping, which is what the
`carttrace agents tasks` command reads. Either way it becomes one frozen `CategorySpec`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from carttrace.agents.spec import INTENTS, IntentType
from carttrace.core.errors import ValidationError


class TemplateLike(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def text(self) -> str: ...

    @property
    def slots(self) -> Sequence[str]: ...


class HypothesisLike(Protocol):
    @property
    def id(self) -> str: ...


class CategoryLike(Protocol):
    """The attributes of carttrace.stores.Category that task generation reads."""

    @property
    def slug(self) -> str: ...

    @property
    def hidden_spec(self) -> Sequence[str]: ...

    @property
    def intents(self) -> Mapping[IntentType, Sequence[TemplateLike]]: ...

    @property
    def hypotheses(self) -> Sequence[HypothesisLike]: ...


@dataclass(frozen=True, slots=True)
class Template:
    id: str
    text: str
    slots: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CategorySpec:
    slug: str
    fields: tuple[str, ...]
    """The hidden-spec fields this category's tasks may use."""
    templates: Mapping[IntentType, tuple[Template, ...]]
    hypotheses: tuple[str, ...]
    """Ids of the category's checklist hypotheses, e.g. "clothing-adult.size-chart-text"."""

    def hypothesis_for(self, endings: Sequence[str]) -> str | None:
        """The hypothesis id that ends with the earliest of `endings` the category has, e.g. its size chart one."""
        return next((h for e in endings for h in self.hypotheses if h.endswith(f".{e}")), None)


def category_spec(category: CategoryLike | Mapping[str, Any]) -> CategorySpec:
    """A CategorySpec from an carttrace.stores Category or from a category file's mapping."""
    if isinstance(category, Mapping):
        return _from_mapping(category)
    templates = {
        intent: tuple(Template(t.id, t.text, tuple(t.slots)) for t in category.intents.get(intent, ()))
        for intent in INTENTS
    }
    return CategorySpec(
        slug=category.slug,
        fields=tuple(category.hidden_spec),
        templates=templates,
        hypotheses=tuple(h.id for h in category.hypotheses),
    )


def _from_mapping(data: Mapping[str, Any]) -> CategorySpec:
    slug = data.get("slug")
    fields = data.get("hidden_spec")
    if not isinstance(slug, str) or not slug or not isinstance(fields, list) or not fields:
        raise ValidationError(
            "a category needs a slug and a hidden_spec list", hint="pass a data/categories/<slug>.yaml mapping"
        )
    raw_intents = data.get("intents")
    intents: Mapping[str, Any] = raw_intents if isinstance(raw_intents, Mapping) else {}
    templates: dict[IntentType, tuple[Template, ...]] = {}
    for intent in INTENTS:
        rows = intents.get(intent) or []
        templates[intent] = tuple(
            Template(str(t["id"]), str(t["text"]), tuple(str(s) for s in t.get("slots", [])))
            for t in rows
            if isinstance(t, Mapping) and "id" in t and "text" in t
        )
    raw_hypotheses = data.get("hypotheses")
    hypotheses: list[Any] = raw_hypotheses if isinstance(raw_hypotheses, list) else []
    return CategorySpec(
        slug=slug,
        fields=tuple(str(f) for f in fields),
        templates=templates,
        hypotheses=tuple(str(h["id"]) for h in hypotheses if isinstance(h, Mapping) and "id" in h),
    )

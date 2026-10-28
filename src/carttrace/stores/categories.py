"""The 25 categories of the store queue, as data: hidden-spec fields, intent templates and checklist hypotheses.

Why data. Categories are the unit of the benchmark, and what a shopper needs differs by category: a size chart as
text for clothing, allergens and dosage by age for non-prescription health, fitment for car parts. Keeping that in
YAML lets people who know a category edit it without touching code, while `load_categories` refuses any file that
would quietly break task generation or judging.

Files:
- `data/categories/_fields.yaml`: the vocabularies every category draws from. `fields` are the hidden-spec fields:
  one definition of `budget` or `allergens` means the shopper's question mapping is one enum across categories.
  `details` are the catalog details a hypothesis may name (allergens, size_chart, care...): the facts a
  detail_hidden experiment hides one at a time (carttrace.collect.conditions), so the benchmark measures what one detail is
  worth rather than what a whole description is worth.
- `data/categories/<slug>.yaml`: one category, named exactly as in the queue, with an approximate Shopify taxonomy
  mapping, the fields its tasks may hide, two or three templates for each of the five intent types, and five to ten
  checklist hypotheses saying what a store should expose, where in the protocol data, which catalog details it is
  about, and which judge decision tests it.

Why the detail vocabulary is data here and not an import of `carttrace.sandbox.catalogs.records.DETAILS` (which the layers
would allow: stores is layer 5, the sandbox layer 4). Category files are then checked against their own directory,
the way hidden-spec fields already are; the vocabulary holds size_chart, which stores publish and agents read but no
catalog record carries; and loading categories does not load the sandbox, whose catalogs package brings its state
machine with it. Tests keep the copies in step: every catalog detail is in the vocabulary, and carttrace.collect has a
matcher for every name in it, so a hypothesis cannot name a detail that hiding would miss.

Working definitions of the intent types (ours, written down so templates stay consistent):
- replenish: buy again something the shopper already uses; the product is known, the task is to find it or its
  current equivalent;
- specify: the shopper states hard constraints up front and expects an item that meets all of them;
- compare: the shopper wants two or more candidates weighed on named criteria before choosing;
- discover: the shopper describes a situation, not a product, and expects suggestions;
- express: the shopper wants it done fast with little back and forth, delegating choices to the agent.

A hypothesis is a claim to test, never a finding. Its evidence_grade stays "hypothesis" until paired runs (the same
store, task and day, with the detail visible and hidden to our agent) support it, and any other grade must cite those
runs in `evidence`.
"""

from __future__ import annotations

import string
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.stores.text import slugify

CATEGORY_SCHEMA = "mc.category.v1"
FIELDS_SCHEMA = "mc.category-fields.v1"
FIELDS_FILE = "_fields.yaml"

IntentType = Literal["replenish", "specify", "compare", "discover", "express"]
INTENT_TYPES: tuple[IntentType, ...] = ("replenish", "specify", "compare", "discover", "express")
FieldType = Literal["text", "enum", "list", "int", "money", "date", "bool"]
"""money is an integer in minor units plus an ISO 4217 code (carttrace.core.money); date is an ISO date."""
Location = Literal[
    "get_product.description",
    "get_product.attributes",
    "get_product.options",
    "get_product.metafields",
    "policies",
]
"""Where in protocol data a detail should appear: the UCP get_product description, its attributes, its options
(variants), its metafields, or the store's policies (search_shop_policies_and_faqs)."""
EvidenceGrade = Literal["hypothesis", "supported", "refuted", "checklist"]

TEMPLATES_PER_INTENT = (2, 3)
HYPOTHESES_PER_CATEGORY = (5, 10)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SpecField(_Frozen):
    """One hidden-spec field: what a shopper may want without saying it until asked."""

    type: FieldType
    description: str
    values: list[str] = Field(default_factory=list)
    """For enum, the allowed values; for list and text, examples."""


class DetailSpec(_Frozen):
    """One catalog detail a hypothesis may name: a fact a store states about a product, which a detail_hidden
    experiment can hide on its own."""

    description: str = Field(min_length=1)


class TaxonomyPath(_Frozen):
    id: str = Field(pattern=r"^[a-z]{2}(-\d+)*$")
    """Shopify taxonomy category id without the gid prefix, e.g. "aa-1"."""
    path: str


class ShopifyTaxonomy(_Frozen):
    version: str
    """Release of Shopify's Standard Product Taxonomy the paths were checked against, e.g. "2026-08"."""
    approximate: bool
    """True: the mapping from the queue category to these parents is our best effort, not an official mapping."""
    paths: list[TaxonomyPath] = Field(min_length=1)
    note: str = ""


class IntentTemplate(_Frozen):
    id: str
    text: str
    """The shopper's opening, with {slot} placeholders filled from the hidden spec."""
    slots: list[str]

    def render(self, values: Mapping[str, str]) -> str:
        missing = [slot for slot in self.slots if slot not in values]
        if missing:
            raise ValidationError(f"template {self.id} needs values for {missing}")
        return self.text.format_map({slot: values[slot] for slot in self.slots})


class Hypothesis(_Frozen):
    id: str
    statement: str
    """What a store should expose, as a yes or no item."""
    why: str
    """Why we expect it to matter to a text-only agent. A reason to test, not a result."""
    where: list[Location] = Field(min_length=1)
    details: list[str] = Field(default_factory=list)
    """The catalog details the hypothesis is about, from the `details` vocabulary of _fields.yaml. detail_hidden
    hides exactly these wherever the agent would read them; with none, it hides the locations in `where` whole."""
    judge_decision: str = Field(pattern=r"^judge\.[a-z][a-z0-9_]*$")
    """The judge decision type whose answer, compared between paired runs, tests the hypothesis."""
    evidence_grade: EvidenceGrade
    evidence: list[str] = Field(default_factory=list)
    """Ids of the paired runs behind any grade other than hypothesis."""


class Category(_Frozen):
    schema_: Literal["mc.category.v1"] = Field(default="mc.category.v1", alias="schema")
    name: str
    """Exactly as in the queue's Category column."""
    slug: str
    index: int = Field(ge=1)
    """The evidence tab's category_id."""
    summary: str
    shopify_taxonomy: ShopifyTaxonomy
    hidden_spec: list[str] = Field(min_length=1)
    """Names from the field vocabulary that this category's tasks may hide."""
    intents: dict[IntentType, list[IntentTemplate]]
    hypotheses: list[Hypothesis]


@dataclass(frozen=True, slots=True)
class CategoryCatalog:
    """Every category, validated together with the field and detail vocabularies. Iterates in index order."""

    fields: Mapping[str, SpecField]
    categories: tuple[Category, ...]
    details: Mapping[str, DetailSpec] = field(default_factory=dict)
    """The catalog detail vocabulary, in the order _fields.yaml lists it."""

    def __iter__(self) -> Iterator[Category]:
        return iter(self.categories)

    def __len__(self) -> int:
        return len(self.categories)

    def get(self, name_or_slug: str) -> Category:
        for category in self.categories:
            if name_or_slug in (category.name, category.slug):
                return category
        raise NotFoundError(f"no category {name_or_slug!r}", hint=f"choose one of {[c.slug for c in self]}")

    def spec_fields(self, category: Category) -> dict[str, SpecField]:
        return {name: self.fields[name] for name in category.hidden_spec}

    def judge_decisions(self) -> list[str]:
        """Every judge decision type the hypotheses name, sorted: the list carttrace.judge has to define."""
        return sorted({h.judge_decision for c in self for h in c.hypotheses})


def load_categories(directory: Path | None = None) -> CategoryCatalog:
    """Load and validate every category file; the default directory is data/categories under the configured data."""
    if directory is None:
        from carttrace.stores.settings import load_settings

        directory = load_settings().categories_dir
    fields, details = _load_vocabularies(directory / FIELDS_FILE)
    paths = sorted(p for p in directory.glob("*.yaml") if not p.name.startswith("_"))
    if not paths:
        raise NotFoundError(f"no category files in {directory}", hint="restore data/categories from git")
    categories = [_load_category(path, fields, details) for path in paths]
    _check_across(categories)
    ordered = tuple(sorted(categories, key=lambda c: c.index))
    return CategoryCatalog(fields=fields, categories=ordered, details=details)


def _load_vocabularies(path: Path) -> tuple[dict[str, SpecField], dict[str, DetailSpec]]:
    """The hidden-spec fields and the catalog details. A file without `details` names none, so no hypothesis may."""
    doc = _yaml(path)
    if doc.get("schema") != FIELDS_SCHEMA or not isinstance(doc.get("fields"), dict):
        raise ValidationError(f"{path.name} is not a {FIELDS_SCHEMA} file")
    fields: dict[str, SpecField] = {}
    for name, raw in doc["fields"].items():
        _check_name(path.name, "field", name)
        fields[name] = _model(SpecField, raw, f"{path.name}: {name}")
        if fields[name].type == "enum" and not fields[name].values:
            raise ValidationError(f"{path.name}: enum field {name!r} lists no values")
    raw_details = doc.get("details", {})
    if not isinstance(raw_details, dict):
        raise ValidationError(f"{path.name}: details must map each detail name to its description")
    details: dict[str, DetailSpec] = {}
    for name, raw in raw_details.items():
        _check_name(path.name, "detail", name)
        details[name] = _model(DetailSpec, raw, f"{path.name}: details: {name}")
    return fields, details


def _check_name(where: str, kind: str, name: object) -> None:
    if not isinstance(name, str) or not name.isidentifier() or name != name.lower():
        raise ValidationError(f"{where}: {kind} name {name!r} must be a lowercase identifier")


def _load_category(path: Path, fields: Mapping[str, SpecField], details: Mapping[str, DetailSpec]) -> Category:
    category = _model(Category, _yaml(path), path.name)
    where = path.name
    if category.slug != path.stem or category.slug != slugify(category.name):
        raise ValidationError(f"{where}: slug must be {slugify(category.name)!r} and match the file name")
    if not category.shopify_taxonomy.approximate:
        raise ValidationError(f"{where}: the Shopify taxonomy mapping is a best effort; mark it approximate: true")
    unknown = [name for name in category.hidden_spec if name not in fields]
    if unknown or len(set(category.hidden_spec)) != len(category.hidden_spec):
        raise ValidationError(f"{where}: hidden_spec names unknown or repeated fields {unknown}", hint=FIELDS_FILE)
    if set(category.intents) != set(INTENT_TYPES):
        raise ValidationError(f"{where}: intents must cover exactly {list(INTENT_TYPES)}")
    for intent, templates in category.intents.items():
        if not TEMPLATES_PER_INTENT[0] <= len(templates) <= TEMPLATES_PER_INTENT[1]:
            raise ValidationError(f"{where}: {intent} needs 2 or 3 templates, has {len(templates)}")
        for template in templates:
            _check_template(where, category, intent, template)
    low, high = HYPOTHESES_PER_CATEGORY
    if not low <= len(category.hypotheses) <= high:
        raise ValidationError(f"{where}: needs {low} to {high} hypotheses, has {len(category.hypotheses)}")
    for hypothesis in category.hypotheses:
        _check_hypothesis(where, category, hypothesis, details)
    return category


def _check_hypothesis(
    where: str, category: Category, hypothesis: Hypothesis, details: Mapping[str, DetailSpec]
) -> None:
    if not hypothesis.id.startswith(f"{category.slug}."):
        raise ValidationError(f"{where}: hypothesis id {hypothesis.id!r} must start with {category.slug!r}.")
    if len(set(hypothesis.where)) != len(hypothesis.where):
        raise ValidationError(f"{where}: {hypothesis.id} lists a location twice")
    unknown = [name for name in hypothesis.details if name not in details]
    if unknown or len(set(hypothesis.details)) != len(hypothesis.details):
        raise ValidationError(
            f"{where}: {hypothesis.id} names unknown or repeated details {unknown}",
            hint=f"name details from the `details` vocabulary of {FIELDS_FILE}: {', '.join(details) or 'none'}",
        )
    if hypothesis.evidence_grade != "hypothesis" and not hypothesis.evidence:
        raise ValidationError(
            f"{where}: {hypothesis.id} is graded {hypothesis.evidence_grade!r} without evidence",
            hint="cite the paired runs that support the grade, or keep it a hypothesis",
        )


def _check_template(where: str, category: Category, intent: str, template: IntentTemplate) -> None:
    if not template.id.startswith(f"{category.slug}.{intent}."):
        raise ValidationError(f"{where}: template id {template.id!r} must start with {category.slug}.{intent}.")
    named: list[str] = []
    try:
        parsed = list(string.Formatter().parse(template.text))
    except ValueError as exc:
        raise ValidationError(f"{where}: template {template.id} has a broken placeholder: {exc}") from exc
    for _, name, spec, conversion in parsed:
        if name is None:
            continue
        if not name.isidentifier() or spec or conversion:
            raise ValidationError(f"{where}: template {template.id} placeholder {{{name}}} must be a plain slot name")
        named.append(name)
    if set(named) != set(template.slots) or len(set(template.slots)) != len(template.slots):
        raise ValidationError(f"{where}: template {template.id} slots {template.slots} do not match its text")
    outside = [slot for slot in template.slots if slot not in category.hidden_spec]
    if outside:
        raise ValidationError(f"{where}: template {template.id} uses slots {outside} not in hidden_spec")


def _check_across(categories: list[Category]) -> None:
    for attribute in ("name", "slug", "index"):
        values = [getattr(c, attribute) for c in categories]
        repeated = sorted({str(v) for v in values if values.count(v) > 1})
        if repeated:
            raise ValidationError(f"categories repeat {attribute} {repeated}")
    ids = [t.id for c in categories for templates in c.intents.values() for t in templates]
    ids += [h.id for c in categories for h in c.hypotheses]
    repeated_ids = sorted({i for i in ids if ids.count(i) > 1})
    if repeated_ids:
        raise ValidationError(f"template or hypothesis ids repeat: {repeated_ids}")


def _yaml(path: Path) -> dict[str, Any]:
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValidationError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValidationError(f"{path.name} must hold a mapping")
    return doc


def _model[M: BaseModel](model: type[M], raw: object, where: str) -> M:
    try:
        return model.model_validate(raw)
    except PydanticValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "file"
        raise ValidationError(f"{where}: {location}: {first['msg']}") from exc

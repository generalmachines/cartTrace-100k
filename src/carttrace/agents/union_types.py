"""Observed keys to data-type ids: the merge map that says which keys state the same thing, and the id every other key
gets from its own name.

Why a merge map and not a list of types. The project (2026-10-13) asked for the union of what stores show, with nothing
assumed. The same fact arrives under many names: a compare-at price is `compare_at_minor` in the capture tables,
`variants[].compare_at_price` in a Shopify feed and `variants[].list_price` on UCP; a fabric is a spec row on one site,
a "fabric: cotton" tag on another and a "Materials:" line on a third. data/bench/union.yaml lists such names under one
curated id (`product.materials`), with a plain name, a group, and the exposure row and data kinds the checklist uses.
A curated type exists in a category's union only when some store's data carries one of its names; the file never
adds a type by itself. Every key the file does not name keeps an id made from the key (`schema.product.audience`,
`feed.variants.taxable`, `label.front-rise`), so nothing a store shows is dropped for want of a name, and a labeler
can later merge it into a curated id or reject it (`labels.jsonl`, carttrace.bench.union).

The id scheme (the exposure experiments key on it):
- curated: `<area>.<name>` in lower snake case, `area` one of product, variant, price, stock, reviews, delivery,
  returns, policy, media, badge, offer, store, identity, nav; ids in data/bench/union.yaml never change meaning, and
  a renamed one is kept as an alias of the new;
- from a structured key: the source family, then the key's path with `[]` dropped and each segment in kebab case,
  an enumerated value after the field (`schema.offer.availability.pre-order`, `agent.cart.links.type.refund-policy`);
- from a free label (a spec row, a "Label:" line, a tag, an app-state field, a metafield): `label.<slug>`, so the
  same label merges across sources and stores; from a variant option's name: `variant.option.<slug>`; from a section
  heading: `heading.<slug>` (a heading is only ever a candidate until it is curated or a labeler keeps it).

It lives a layer below carttrace.bench (which re-exports it as carttrace.bench.union_types) so the sandbox's exposure sheet
(carttrace.collect.exposure_sheet, D93) resolves what a surface serves to the union's own ids.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from carttrace.agents.checks import Check, Checklists
from carttrace.agents.doc_keys import HEADING, OPTION, slug
from carttrace.core.errors import NotFoundError, ValidationError

DEFAULT_PATH: Final = Path(__file__).resolve().parents[3] / "data" / "bench" / "union.yaml"
TYPE_ID: Final = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9][a-z0-9_-]*)+$")
FREE_PREFIXES: Final = frozenset({"spec", "label", "heading", "tag", "app", "prop", "ucp_meta", "feed_meta",
                                  "product_js_meta", "graphql_meta"})  # fmt: skip
"""Key prefixes whose rest is a label the store wrote, matched against a type's `labels`."""
AUTO_PREFIX: Final[Mapping[str, str]] = {
    "table": "table",
    "schema": "schema",
    "meta": "meta",
    "feed": "feed",
    "product_js": "feed",
    "graphql": "feed",
    "ucp": "agent",
    "cart": "agent.cart",
    "payment_handler": "store.payment-handler",
    "page_field": "page.field",
    "page_table": "page.table",
    "text": "text",
    "policy": "policy.fact",
    "store_page": "store.page",
    "store_file": "store.file",
    "store_mcp": "agent.store-mcp",
    "well_known": "agent.well-known",
    "probe": "store.probe",
    "canvas": "canvas",
}
"""The family an uncurated structured key's id starts with: the capture tables, schema.org, meta tags, the platform
feeds and product documents, the agent view's fields and cart replies, the page reader, store-level reads."""
AUTO_GROUPS: Final[Mapping[str, str]] = {
    "table": "Capture table columns",
    "schema": "schema.org properties",
    "meta": "Product meta tags",
    "feed": "Platform feed fields",
    "agent": "Agent view fields",
    "store": "Store-level",
    "page": "Product page",
    "text": "Product text",
    "policy": "Policies",
    "canvas": "Canvas detector readings",
    "label": "Labels the stores wrote",
    "heading": "Section headings",
    "variant": "Variant facts",
}
Origin = Literal["curated", "auto"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TypeSpec(_Frozen):
    """One curated data type: its id and plain name, its group on the pages, what it means, the exposure row and
    brief kinds a checklist check is matched by, and the observed names it gathers."""

    id: str = Field(pattern=TYPE_ID.pattern)
    name: str
    group: str
    about: str = ""
    row: str | None = None
    kinds: list[str] = Field(default_factory=list)
    keys: list[str] = Field(default_factory=list)
    """Observed keys, exact or with `*` (any run) and `?` (one character) as wildcards (`schema:*.material`);
    brackets are literal, as in `variants[].price`."""
    labels: list[str] = Field(default_factory=list)
    """Free-label slugs (spec rows, "Label:" lines, headings, tags, app-state fields, metafields)."""
    options: list[str] = Field(default_factory=list)
    """Variant option name slugs."""
    former: list[str] = Field(default_factory=list)
    """Ids this type was known by before, so a label written against an old id still finds it."""


class TypeFile(_Frozen):
    schema_: Literal["mc.bench.union_types.v1"] = Field(alias="schema")
    version: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    types: list[TypeSpec]

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


@dataclass(frozen=True, slots=True)
class Resolved:
    """What one observed key is: its type id, whether a curated type gathered it, and its kind."""

    type_id: str
    origin: Origin
    free: bool
    """A free label or heading: uncurated, it is a type only once two stores show it (or a labeler keeps it)."""
    heading: bool


@dataclass(frozen=True, slots=True)
class TypeBook:
    """The merge map, ready to resolve keys."""

    version: str
    types: tuple[TypeSpec, ...]
    by_key: Mapping[str, str]
    patterns: tuple[tuple[re.Pattern[str], str], ...]
    by_label: Mapping[str, str]
    by_option: Mapping[str, str]
    by_former: Mapping[str, str]
    by_id: Mapping[str, TypeSpec]

    def spec(self, type_id: str) -> TypeSpec | None:
        return self.by_id.get(type_id)

    def current(self, type_id: str) -> str:
        """The id a type has now, for a label written against a former one."""
        return self.by_former.get(type_id, type_id)

    def resolve(self, key: str, kind: str) -> Resolved:
        prefix, _, rest = key.partition(":")
        curated = self.by_key.get(key) or next((t for p, t in self.patterns if p.fullmatch(key)), None)
        free = prefix in FREE_PREFIXES
        if curated is None and kind == OPTION:
            curated = self.by_option.get(rest)
        elif curated is None and free:
            curated = self.by_label.get(rest)
        if curated is not None:
            return Resolved(curated, "curated", free=False, heading=False)
        if kind == OPTION or prefix == "option":
            return Resolved(f"variant.option.{_segment(rest)}", "auto", free=False, heading=False)
        if kind == HEADING or prefix == "heading":
            return Resolved(f"heading.{_segment(rest)}", "auto", free=True, heading=True)
        if free:
            return Resolved(f"label.{_segment(rest)}", "auto", free=True, heading=False)
        return Resolved(auto_id(key), "auto", free=False, heading=False)


def _segment(text: str) -> str:
    return slug(text) or "x"


def auto_id(key: str) -> str:
    """The id an uncurated structured key gets: its family, then its path in kebab case (`feed:variants[].taxable`
    is `feed.variants.taxable`; `schema:Offer.availability=pre-order` is `schema.offer.availability.pre-order`)."""
    prefix, _, rest = key.partition(":")
    family = AUTO_PREFIX.get(prefix, _segment(prefix))
    parts = [p for p in re.split(r"[.=]", rest.replace("[]", "")) if p]
    return ".".join([family, *(_segment(p) for p in parts)]) if parts else f"{family}.x"


def auto_group(type_id: str) -> str:
    return AUTO_GROUPS.get(type_id.split(".", 1)[0], "Other")


def auto_name(type_id: str, key: str) -> str:
    """A readable name for an uncurated type: what the store calls it, and where."""
    prefix, _, rest = key.partition(":")
    if type_id.startswith("label."):
        return f'"{rest.replace("-", " ")}" (a label stores wrote)'
    if type_id.startswith("heading."):
        return f'"{rest.replace("-", " ")}" (a section heading)'
    if type_id.startswith("variant.option."):
        return f'Variant option "{rest.replace("-", " ")}"'
    return f"{rest} ({AUTO_GROUPS.get(type_id.split('.', 1)[0], prefix).lower()})"


def load_types(path: Path | None = None) -> TypeBook:
    """The merge map at `path` (default data/bench/union.yaml), checked: ids are unique and well formed, no key,
    label or option is claimed by two types, and no former id is a current one."""
    return _load((path or DEFAULT_PATH).resolve())


@functools.cache
def _load(path: Path) -> TypeBook:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NotFoundError(f"no union types at {path}", hint="restore data/bench/union.yaml from git") from None
    try:
        parsed = TypeFile.model_validate(raw)
    except PydanticValidationError as exc:
        raise ValidationError(f"{path.name} does not validate: {exc.errors()[0]['msg']}", hint=str(exc)) from exc
    by_key: dict[str, str] = {}
    patterns: list[tuple[re.Pattern[str], str]] = []
    by_label: dict[str, str] = {}
    by_option: dict[str, str] = {}
    by_former: dict[str, str] = {}
    ids: set[str] = set()
    for spec in parsed.types:
        if spec.id in ids:
            raise ValidationError(f"{path.name} lists type {spec.id} twice", hint="merge the two entries")
        ids.add(spec.id)
        for key in spec.keys:
            if "*" in key or "?" in key:
                patterns.append((_wildcard(key), spec.id))
            else:
                _claim(by_key, key, spec.id, path)
        for label in spec.labels:
            _claim(by_label, label, spec.id, path)
        for option in spec.options:
            _claim(by_option, option, spec.id, path)
        for former in spec.former:
            _claim(by_former, former, spec.id, path)
    clash = sorted(ids & set(by_former))
    if clash:
        raise ValidationError(f"{path.name}: {clash[0]} is both a type and a former id", hint="drop one")
    return TypeBook(
        version=parsed.version,
        types=tuple(parsed.types),
        by_key=by_key,
        patterns=tuple(patterns),
        by_label=by_label,
        by_option=by_option,
        by_former=by_former,
        by_id={t.id: t for t in parsed.types},
    )


def _wildcard(pattern: str) -> re.Pattern[str]:
    return re.compile(re.escape(pattern).replace(r"\*", ".*").replace(r"\?", "."))


def _claim(table: dict[str, str], name: str, type_id: str, path: Path) -> None:
    held = table.get(name)
    if held is not None and held != type_id:
        raise ValidationError(f"{path.name}: {name!r} is claimed by {held} and {type_id}", hint="keep it in one type")
    table[name] = type_id


# -- the checklist -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckMatch:
    check_id: str
    check: str
    by: Literal["row", "kind"]


def matching_checks(spec: TypeSpec | None, checks: Sequence[Check]) -> list[CheckMatch]:
    """The category's checks a type serves: those on the type's exposure row whose data kinds the type shares (or
    either names none); failing that, those that need one of the type's kinds. An uncurated type matches none: it is
    a finding for the labeler, not an error."""
    if spec is None:
        return []
    kinds = set(spec.kinds)
    by_row = [
        CheckMatch(c.id, c.check, "row")
        for c in checks
        if spec.row is not None and c.row == spec.row and (not c.kinds or not kinds or kinds & set(c.kinds))
    ]
    if by_row:
        return by_row
    return [CheckMatch(c.id, c.check, "kind") for c in checks if kinds & set(c.kinds)]


def category_checks(checklists: Checklists | None, category: str) -> list[Check]:
    return [] if checklists is None else checklists.for_category(category)


__all__ = [
    "AUTO_PREFIX",
    "DEFAULT_PATH",
    "FREE_PREFIXES",
    "TYPE_ID",
    "CheckMatch",
    "Resolved",
    "TypeBook",
    "TypeSpec",
    "auto_group",
    "auto_id",
    "auto_name",
    "category_checks",
    "load_types",
    "matching_checks",
]

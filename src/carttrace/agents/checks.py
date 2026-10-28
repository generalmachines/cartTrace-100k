"""The niche checklists as data (`data/agent/checks.yaml`, `mc.checks.v1`): the checks of every niche, each mapped to
a category, an exposure taxonomy row and the brief's data kinds, and what a store's intents and its static scan take
from them.

Why. The project's checklists (docs/research/niche-checklists-2026-10-10.md) say, per niche, what a shopper asks and
what store data answers it, with a severity and whether a static read or an agent run tests it. They covered eight of
the team sheet's 25 categories, so 70 of the 134 stores ready to shop had no checklist; the seventeen niches drafted
for the rest (docs/research/niche-checklists-2026-10-13.md, D54) carry `status: draft` until the project reviews them,
and are applied like the reviewed ones meanwhile, so a draft is tested on real stores before it is reviewed.

The intent writer turns the "Shopper asks" examples into intents varied per store and records which checks each
intent exercises; the quotas ask for every Critical and High check the store exposes the data for, and for the shared
six every store gets. Static checks are not agent tasks: `static_checks` lists them per store, with the data to look
for and whether the brief found it, for a scanner to run from the capture.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from carttrace.agents.brief import KIND_IDS, StoreBrief
from carttrace.core.canon import dumps
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.files import atomic_write_text

CHECKS_SCHEMA: Final = "mc.checks.v1"
DEFAULT_PATH: Final = Path(__file__).resolve().parents[3] / "data" / "agent" / "checks.yaml"
"""data/agent/checks.yaml in this repository."""
Severity = Literal["Critical", "High", "Medium", "Low"]
Stage = Literal["catalog", "cart", "checkout"]
SHARED: Final = ("access", "stock", "price", "total", "discounts", "returns")
"""The six shared checks every niche repeats under its own ids."""
MUST_COVER: Final[frozenset[str]] = frozenset({"Critical", "High"})


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Check(_Frozen):
    id: str = Field(pattern=r"^[A-Z]+-\d+$")
    niche: str
    shared: Literal["access", "stock", "price", "total", "discounts", "returns"] | None = None
    check: str
    shopper_asks: str
    pass_if: str
    severity: Severity
    tier: Literal["Free", "Paid"]
    test: Literal["Static", "Agent run"]
    row: str
    """The exposure taxonomy row (docs/STORE-EXPOSURE.md)."""
    kinds: list[str] = Field(default_factory=list)
    """The brief's data kinds the check needs; a store exposes the data when it exposes any of them."""
    stage: Stage = "catalog"

    @property
    def static(self) -> bool:
        return self.test == "Static"

    def applies(self, exposed: Sequence[str]) -> bool:
        """Whether the check is one of this store's: a shared check always is (every store gets the six), a niche
        check when the store exposes the data it needs (any of its kinds) or it needs none."""
        return bool(self.shared) or not self.kinds or any(k in exposed for k in self.kinds)


NicheStatus = Literal["reviewed", "draft"]


class Niche(_Frozen):
    name: str
    status: NicheStatus = "reviewed"
    """reviewed: the project's checklist; draft: written by an agent from sources, not yet reviewed by the project."""
    slugs: list[str]

    @property
    def draft(self) -> bool:
        return self.status == "draft"


class LaterNiche(_Frozen):
    name: str
    priority: Literal["P2", "P3"]
    slugs: list[str] = Field(default_factory=list)


class Checklists(_Frozen):
    schema_: Literal["mc.checks.v1"] = Field(alias="schema")
    version: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    niches: dict[str, Niche]
    later: list[LaterNiche] = Field(default_factory=list)
    checks: list[Check]

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    def niche_of(self, category: str) -> str | None:
        """The niche code whose slugs hold the category, or None when no niche covers it."""
        return next((code for code, niche in self.niches.items() if category in niche.slugs), None)

    def for_category(self, category: str) -> list[Check]:
        """The checks of the category's niche, shared ones first, in the file's order; empty for a category no niche
        covers."""
        code = self.niche_of(category)
        if code is None:
            return []
        own = [c for c in self.checks if c.niche == code]
        return [c for c in own if c.shared] + [c for c in own if not c.shared]

    def get(self, check_id: str) -> Check | None:
        return next((c for c in self.checks if c.id == check_id), None)

    def drafts(self) -> list[str]:
        """The codes of the niches still awaiting the project's review, in the file's order."""
        return [code for code, niche in self.niches.items() if niche.draft]


def load_checklists(path: Path | None = None) -> Checklists:
    """The checklists at `path` (default data/agent/checks.yaml), parsed once per process and checked: every kind a
    check names is one of the brief's, no check id repeats, every check's niche is named, a check's id carries its
    niche's code, and no category slug is in two niches (a store gets one niche's checklist)."""
    return _load((path or DEFAULT_PATH).resolve())


@functools.cache
def _load(path: Path) -> Checklists:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NotFoundError(f"no checklists at {path}", hint="restore data/agent/checks.yaml from git") from None
    except yaml.YAMLError as exc:
        raise ValidationError(f"{path.name} is not valid YAML: {exc}") from exc
    try:
        found = Checklists.model_validate(raw)
    except PydanticValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        where = ".".join(str(p) for p in first["loc"]) or "file"
        raise ValidationError(f"{path.name}: {where}: {first['msg']}", hint=f"follow {CHECKS_SCHEMA}") from None
    problems = [f"{c.id} names unknown data kind {k!r}" for c in found.checks for k in c.kinds if k not in KIND_IDS]
    ids = [c.id for c in found.checks]
    problems += [f"check {i} appears twice" for i in sorted({i for i in ids if ids.count(i) > 1})]
    problems += [
        f"{c.id} is in niche {c.niche}, which the file does not name"
        for c in found.checks
        if c.niche not in found.niches
    ]
    problems += [
        f"{c.id} is not numbered under its niche {c.niche}" for c in found.checks if not c.id.startswith(f"{c.niche}-")
    ]
    slugs = [s for niche in found.niches.values() for s in niche.slugs]
    problems += [f"category {s} is in two niches" for s in sorted({s for s in slugs if slugs.count(s) > 1})]
    if problems:
        raise ValidationError(f"{path.name}: {'; '.join(problems[:5])}", hint="fix the checklists file")
    return found


# -- what a store takes from them -------------------------------------------------------------------------------------


class StaticCheck(_Frozen):
    """One static check for the scanner: the id, what data to look for, and whether the brief found it."""

    id: str
    check: str
    severity: Severity
    row: str
    kinds: list[str]
    exposed: bool
    """Whether the brief found the data the check reads (any of its kinds) in the capture."""


class StaticChecks(_Frozen):
    schema_: Literal["mc.static-checks.v1"] = Field(default="mc.static-checks.v1", alias="schema")
    store_id: str
    category: str
    source: str
    niche: str | None
    checks: list[StaticCheck]

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


def static_checks(brief: StoreBrief, checklists: Checklists | None = None) -> StaticChecks:
    """The static checks of the store's niche, each with the data kinds to look for and whether the brief exposed
    them, for a scanner to run from the capture; empty for a category no niche covers."""
    lists = checklists or load_checklists()
    exposed = brief.exposed_kinds
    return StaticChecks(
        store_id=brief.store_id,
        category=brief.category,
        source=brief.source,
        niche=lists.niche_of(brief.category),
        checks=[
            StaticCheck(
                id=c.id, check=c.check, severity=c.severity, row=c.row, kinds=c.kinds, exposed=c.applies(exposed)
            )
            for c in lists.for_category(brief.category)
            if c.static
        ],
    )


CHECKS_DIR: Final = Path("agent") / "checks"
"""Under paths.var: <store>/<source>.json, the static check list a scanner runs from the capture."""


def save_static_checks(found: StaticChecks, var: Path) -> Path:
    path = var / CHECKS_DIR / found.store_id / f"{found.source}.json"
    atomic_write_text(path, dumps(found.to_json()) + "\n")
    return path


def agent_checks(brief: StoreBrief, checklists: Checklists | None = None) -> list[Check]:
    """The checks an intent of this store may exercise: every check of its niche whose data the brief exposes (a
    static check's "Shopper asks" is still a request an agent can be given), the shared ones first."""
    lists = checklists or load_checklists()
    exposed = brief.exposed_kinds
    return [c for c in lists.for_category(brief.category) if c.applies(exposed) and c.shared != "access"]


def must_cover(checks: Sequence[Check]) -> list[str]:
    """The check ids a store's intents must cover among `checks`: every Critical and High check, and every shared
    agent-run check (stock, price, the delivered total, discounts, returns)."""
    return [c.id for c in checks if c.severity in MUST_COVER or (c.shared and c.shared != "access")]


__all__ = [
    "CHECKS_DIR",
    "CHECKS_SCHEMA",
    "MUST_COVER",
    "SHARED",
    "Check",
    "Checklists",
    "LaterNiche",
    "Niche",
    "NicheStatus",
    "StaticCheck",
    "StaticChecks",
    "agent_checks",
    "load_checklists",
    "must_cover",
    "save_static_checks",
    "static_checks",
]

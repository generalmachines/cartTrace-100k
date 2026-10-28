"""Per-store results the sync reads: what a protocol preflight saw, and how many traces a store has.

Why records. The preflight and the collector both learn things about a store that the team sheet should show, but
neither should write the sheet itself. They write records, each a `mc.stores.result.v1` object, to the results
database (carttrace.stores.resultsdb, one row per record with its provenance beside it); the sync folds the records of a
store's own provenances (live and replica, never a twin) and plans the machine-owned cells from them. A JSONL file of
the same records is the fallback writer behind collect.results_file_fallback, and what `--results FILE` reads.

Schema `mc.stores.result.v1`, one JSON object per line:

    {"schema": "mc.stores.result.v1", "store_id": "store-004", "observed_at": "2026-10-12T09:30:00Z",
     "platform": "Shopify", "market": "US", "preflight": "ok", "robots_allowed": true, "http_status": 200,
     "protocols": {"ucp_mcp": "2026-08-25", "store_mcp": "2025-06-18"}, "profile_id": "pp-3f2a9c1b0d4e",
     "detail": "UCP discovery and catalog search answered", "traces_collected": 5, "latest_outcome": "cart_ready"}

Only `schema`, `store_id` and `observed_at` are required; a record may carry just the preflight fields, or just the
trace counts. A preflight record may also carry `profile`, the whole ProtocolProfile the store's discovery document
declared (UCP core, capabilities and payment handlers), so a store's sandbox replica speaks the store's own versions
instead of a shared preset (carttrace.collect.profiles); records written before it existed simply have none.

Folding is deterministic: a store's records are ordered by `observed_at` (then by line). A record that sets any
preflight field replaces the store's whole preflight (so a later "no_protocol" never keeps an older run's versions),
and one that sets a trace count or outcome replaces both. Text fields are short and plain, and never hold
secrets or personal data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import ValidationError as PydanticValidationError

from carttrace.core.clock import iso
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.files import read_jsonl
from carttrace.protocols.profile import ProtocolProfile
from carttrace.protocols.tools import SurfaceKind
from carttrace.trace.schema import EndReason

RESULT_SCHEMA = "mc.stores.result.v1"

PreflightOutcome = Literal["ok", "no_protocol", "blocked", "robots_disallowed", "throttled", "gated", "error"]
"""ok: an agent protocol answered; no_protocol: none found; blocked: bot wall, 401 or 403; robots_disallowed: our
token is disallowed; throttled: 429s exhausted the budget; gated: age gate, login or consultation; error: anything
else (our side, a network failure, the store's 5xx, a busy store), with the detail line saying which."""


class StoreResult(BaseModel):
    """One observation about one store, or the fold of several."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: Literal["mc.stores.result.v1"] = Field(default="mc.stores.result.v1", alias="schema")
    store_id: str = Field(min_length=1)
    observed_at: str
    """RFC 3339; stored in the one timestamp format (UTC, milliseconds, Z)."""
    platform: str | None = None
    """Platform as protocol evidence confirms it, e.g. "Shopify" from a UCP discovery document."""
    market: str | None = None
    """Delivery market the preflight could ship to, e.g. "US"."""
    preflight: PreflightOutcome | None = None
    robots_allowed: bool | None = None
    """Whether robots.txt allows our token `carttrace` on the paths the preflight used."""
    http_status: int | None = Field(default=None, ge=100, le=599)
    """HTTP status of the storefront (or its discovery endpoint) at preflight."""
    protocols: dict[SurfaceKind, str] = Field(default_factory=dict)
    """Agent protocol surfaces the preflight found, each with the version it speaks, e.g.
    {"ucp_mcp": "2026-08-25", "store_mcp": "2025-06-18"}. Empty when none answered."""
    profile_id: str | None = None
    """ProtocolProfile id (pp-...) captured from the store."""
    profile: ProtocolProfile | None = None
    """The profile itself, as discovery captured it (source "live"); the id above is its hash."""
    detail: str = Field(default="", max_length=500)
    """One plain line for people and for the decision model: what the preflight saw."""
    traces_collected: int | None = Field(default=None, ge=0)
    latest_outcome: EndReason | None = None
    """End reason of the store's most recent episode."""

    @field_validator("observed_at")
    @classmethod
    def _canonical_time(cls, value: str) -> str:
        try:
            moment = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"observed_at {value!r} is not RFC 3339") from exc
        if moment.tzinfo is None:
            raise ValueError("observed_at needs a timezone, e.g. a trailing Z")
        return iso(moment)


def load_results(path: Path) -> dict[str, StoreResult]:
    """Every store's folded result from a results file, keyed by store id."""
    return fold_results(read_results(path))


def read_results(path: Path) -> list[StoreResult]:
    """Every record of a results file, in file order, each checked against the schema."""
    if not path.is_file():
        raise NotFoundError(f"results file {path} does not exist", hint="pass the JSONL file the preflight wrote")
    records: list[StoreResult] = []
    for line_no, raw in enumerate(read_jsonl(path), start=1):
        try:
            records.append(StoreResult.model_validate(raw))
        except PydanticValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(part) for part in first["loc"]) or "record"
            raise ValidationError(
                f"{path} record {line_no}: {where}: {first['msg']}",
                hint=f"each line must be a {RESULT_SCHEMA} object; see carttrace.stores.results",
            ) from exc
    return records


def is_preflight(record: StoreResult) -> bool:
    """A record that sets any preflight field: an observation of the live store itself."""
    return bool(record.model_fields_set & set(PREFLIGHT_FIELDS))


def fold_results(records: Iterable[StoreResult]) -> dict[str, StoreResult]:
    """Fold records per store: later observations override the fields they set, earlier ones fill the rest."""
    ordered = sorted(enumerate(records), key=lambda pair: (pair[1].store_id, pair[1].observed_at, pair[0]))
    folded: dict[str, StoreResult] = {}
    for _, record in ordered:
        before = folded.get(record.store_id)
        folded[record.store_id] = record if before is None else _merge(before, record)
    return dict(sorted(folded.items()))


PREFLIGHT_FIELDS = (
    "platform",
    "market",
    "preflight",
    "robots_allowed",
    "http_status",
    "protocols",
    "profile_id",
    "profile",
    "detail",
)
COLLECTION_FIELDS = ("traces_collected", "latest_outcome")


def _merge(before: StoreResult, after: StoreResult) -> StoreResult:
    changes: dict[str, object] = {"observed_at": after.observed_at}
    for group in (PREFLIGHT_FIELDS, COLLECTION_FIELDS):
        if after.model_fields_set & set(group):
            changes.update({name: getattr(after, name) for name in group})
    return before.model_copy(update=changes)


def result_fields(result: StoreResult) -> Mapping[str, object]:
    """The folded result as plain JSON, for logs and dry-run output."""
    return result.model_dump(mode="json", by_alias=True, exclude_none=True)

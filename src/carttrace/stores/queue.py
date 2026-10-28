"""`load_queue`: a snapshot's queue and evidence tabs joined into typed `Store` records, grouped and split.

Why join two tabs. The queue tab is the people-facing working list (one row per store, what to check next); the
evidence tab keeps the nomination evidence and the machine-facing columns (merchant group, split, eligibility). Both
are keyed by the queue id. A store is the two rows together, with the merchant group and split computed over the whole
queue, because a group can only be known once every row that shares its parent or domain has been read.

Loading is strict about structure and lenient about content: a missing column or a repeated id is an error (the
snapshot is unusable), while a label no table knows, a date that does not parse, or two tabs that disagree is a
warning on the store, so one odd cell in a people-edited sheet never stops collection.

Stores added by hand (carttrace.stores.added) join after the sheet's, in the order they were added (`merge_added`). They
never take part in the sheet's merchant grouping, so adding one can never move a sheet store's group or split.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from carttrace.core.errors import ValidationError
from carttrace.stores.columns import EVIDENCE_CROSS_CHECK, EVIDENCE_FIELDS, EVIDENCE_OPTIONAL_FIELDS, QUEUE_FIELDS
from carttrace.stores.merchant import group_merchants, load_aliases, normalise_host, parent_key, registrable_domain
from carttrace.stores.model import Store, market_of, platform_key, status_key
from carttrace.stores.snapshot import read_snapshot_tab
from carttrace.stores.split import split_for
from carttrace.stores.tables import Tab
from carttrace.stores.text import slugify

DEFAULT_QUEUE_TAB = "Store queue"
DEFAULT_EVIDENCE_TAB = "Evidence"
ID_COLUMN = "ID"
QUEUE_COLUMN_OF = {name: column for column, name in QUEUE_FIELDS.items()}
"""Store field to the queue header it comes from, for messages."""


def load_queue(
    snapshot_dir: Path,
    *,
    queue_tab: str = DEFAULT_QUEUE_TAB,
    evidence_tab: str = DEFAULT_EVIDENCE_TAB,
    aliases: Mapping[str, str] | None = None,
    added: Sequence[Store] = (),
) -> list[Store]:
    """Every store in the snapshot, in queue order, then the `added` stores the snapshot does not hold (see
    `merge_added`; carttrace.stores.added.planning_queue passes them). Aliases default to `merchant-aliases.yaml` beside the
    snapshot."""
    if aliases is None:
        aliases = load_aliases(snapshot_dir.parent / "merchant-aliases.yaml")
    queue = read_snapshot_tab(snapshot_dir, queue_tab)
    evidence = read_snapshot_tab(snapshot_dir, evidence_tab)
    return merge_added(stores_from_tabs(queue, evidence, aliases=aliases), added)


def merge_added(queue: Sequence[Store], added: Sequence[Store]) -> list[Store]:
    """The sheet's stores in queue order, then each store added by hand that the sheet does not already hold, in the
    order they were added.

    The sheet's ids keep their meaning. An added store whose host or registrable domain the sheet lists (a later
    snapshot took it in) is left out, because the sheet's row is that merchant's one store; an added id the sheet also
    uses is refused, since two stores would then share one id and its traces.
    """
    ids = {store.id for store in queue}
    hosts = {store.domain for store in queue}
    domains = {store.registrable_domain for store in queue}
    merged = list(queue)
    for store in added:
        if store.id in ids:
            raise ValidationError(
                f"the store queue uses id {store.id}, which is also the id of a store added by hand",
                hint="ids of stores added by hand start MC-A; keep that prefix out of the team sheet",
            )
        if store.domain in hosts or store.registrable_domain in domains:
            continue
        ids.add(store.id)
        merged.append(store)
    return merged


def stores_from_tabs(queue: Tab, evidence: Tab, *, aliases: Mapping[str, str] | None = None) -> list[Store]:
    """Join a queue tab and an evidence tab (from a snapshot or a backend read) into stores."""
    _require_columns(queue, [ID_COLUMN, *QUEUE_FIELDS])
    _require_columns(evidence, [ID_COLUMN, *EVIDENCE_FIELDS])
    queue_rows = _by_id(queue)
    evidence_rows = _by_id(evidence)

    drafts: list[dict[str, Any]] = []
    for store_id, q in queue_rows.items():
        e = evidence_rows.get(store_id)
        drafts.append(_draft(q, e, aliases or {}))

    groups = group_merchants((d["id"], d["_parent_key"], d["registrable_domain"]) for d in drafts)
    stores: list[Store] = []
    for draft in drafts:
        group = groups[draft["id"]]
        draft.pop("_parent_key")
        stores.append(
            Store(
                **draft,
                merchant_key=group.key,
                merchant_basis=group.basis,
                merchant_group_id=group.id,
                split=split_for(group.id),
            )
        )
    return stores


def _draft(q: Mapping[str, str], e: Mapping[str, str] | None, aliases: Mapping[str, str]) -> dict[str, Any]:
    warnings: list[str] = []
    fields: dict[str, Any] = {name: _text(q.get(column)) for column, name in QUEUE_FIELDS.items()}
    fields["last_observed"] = _date(q.get("Last observed"), "Last observed", warnings)
    for column, name in {**EVIDENCE_FIELDS, **EVIDENCE_OPTIONAL_FIELDS}.items():
        fields[name] = _text(e.get(column)) if e is not None else None
    if e is None:
        warnings.append("no evidence row")
        fields["category_index"] = fields["checked_at"] = fields["duplicate_host_count"] = None
    else:
        fields["category_index"] = _int(e.get("category_id"), "category_id", warnings)
        fields["checked_at"] = _date(e.get("checked_at"), "checked_at", warnings)
        fields["duplicate_host_count"] = _int(e.get("duplicate_host_count"), "duplicate_host_count", warnings)
        for column, queue_column in EVIDENCE_CROSS_CHECK.items():
            if column in e and _text(e.get(column)) != _text(q.get(queue_column)):
                warnings.append(f"{queue_column} and evidence {column} disagree")

    for required in ("id", "category", "name", "website", "platform", "market", "status"):
        if not fields.get(required):
            raise ValidationError(
                f"queue row {fields.get('id') or '?'} has no {QUEUE_COLUMN_OF[required]}",
                hint="fill the cell in the sheet, then pull again",
            )
    fields["parent_company"] = fields["parent_company"] if parent_key(fields["parent_company"]) else None
    fields["what_we_saw"] = fields["what_we_saw"] or ""
    fields["what_to_do_next"] = fields["what_to_do_next"] or ""

    status = status_key(fields["status"])
    platform = platform_key(fields["platform"])
    market = market_of(fields["market"])
    if status is None:
        warnings.append(f"unknown status {fields['status']!r}")
    if platform is None:
        warnings.append(f"unknown platform {fields['platform']!r}")
    if market is None:
        warnings.append(f"unknown market {fields['market']!r}")
    code, currency, likely = market if market is not None else (None, None, False)
    domain = normalise_host(fields.get("canonical_host") or fields["website"])
    fields.update(
        category_slug=slugify(fields["category"]),
        status_key=status or "other",
        platform_key=platform or "other",
        market_code=code,
        market_likely=likely,
        currency=currency,
        domain=domain,
        registrable_domain=registrable_domain(domain),
        warnings=tuple(warnings),
        _parent_key=parent_key(fields["parent_company"], aliases),
    )
    return fields


def _require_columns(tab: Tab, columns: Iterable[str]) -> None:
    missing = [column for column in dict.fromkeys(columns) if column not in tab.header]
    if missing:
        raise ValidationError(
            f"tab {tab.name!r} is missing columns {missing}",
            hint="a column was renamed or removed in the sheet; restore it or update carttrace.stores.columns",
        )


def _by_id(tab: Tab) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for number, record in zip(tab.row_numbers, tab.records(), strict=True):
        key = record.get(ID_COLUMN, "")
        if not key:
            if any(record.values()):
                raise ValidationError(f"tab {tab.name!r} row {number} has data but no {ID_COLUMN}")
            continue
        if key in rows:
            raise ValidationError(f"tab {tab.name!r} repeats id {key} (row {number})", hint="ids must be unique")
        rows[key] = record
    return rows


def _text(value: str | None) -> str | None:
    return value if value else None


def _date(value: str | None, column: str, warnings: list[str]) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        warnings.append(f"{column} {value!r} is not a date")
        return None


def _int(value: str | None, column: str, warnings: list[str]) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        warnings.append(f"{column} {value!r} is not a whole number")
        return None

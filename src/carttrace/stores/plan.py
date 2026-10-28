"""Planning a sheet sync: which machine-owned cells should change, from the queue, the results and the decisions.

Why a pure planner. The sync's one risk is writing the wrong thing into a sheet people rely on. So the plan is a pure
function of its inputs (stores, folded results, eligibility decisions, the current tab and a timestamp), it only ever
produces cells in machine-owned columns, and it is printed and reviewed before `--apply` writes anything. The same
inputs give the same updates in the same order: header cells first, then by sheet row and column.

Rules the planner keeps:
- a cell is written only when its value changes, and never blanked: a missing result leaves the cell as it is;
- "Machine updated at" changes only on rows where another machine cell changed, so re-running is a no-op;
- the merchant group id and the split are locked once set: a different computed value is a conflict note;
- an eligibility decision is written only when it is effective (a live type, and no backend error); a shadow
  decision is reported as a note and changes nothing;
- rows are never added: a store the tab does not list is a note for the people who own the rows;
- a missing machine column is added (its header cell, to the right of the table) only when a row needs it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

from carttrace.core.clock import iso
from carttrace.core.errors import ValidationError
from carttrace.decide.types import Decision
from carttrace.stores.added import is_added_id
from carttrace.stores.columns import MACHINE_COLUMNS, check_writable, machine_column
from carttrace.stores.model import Store
from carttrace.stores.results import StoreResult
from carttrace.stores.service import CellUpdate
from carttrace.stores.tables import Tab
from carttrace.stores.xlsx import column_letter, parse_ref

NoteKind = Literal["new_column", "locked", "shadow", "missing_row", "unknown_store"]


@dataclass(frozen=True, slots=True)
class SyncNote:
    kind: NoteKind
    row_key: str
    column: str
    detail: str


@dataclass(frozen=True, slots=True)
class SyncPlan:
    tab: str
    updates: tuple[CellUpdate, ...]
    notes: tuple[SyncNote, ...]


@dataclass(frozen=True, slots=True)
class _Want:
    value: str
    reason: str
    decision: Decision | None = None


@dataclass(frozen=True, slots=True)
class _Change:
    """A machine cell that should change, before its column is known to exist in the tab."""

    row_key: str
    row: int
    header: str
    want: _Want
    previous: str

    def update(self, tab: str, columns: Mapping[str, int]) -> CellUpdate:
        decision = self.want.decision
        return CellUpdate(
            tab=tab,
            row_key=self.row_key,
            column=self.header,
            cell=f"{column_letter(columns[self.header])}{self.row}",
            value=self.want.value,
            previous=self.previous,
            reason=self.want.reason,
            decided_by=decision.decided_by if decision else None,
            evidence_sha=decision.evidence_sha if decision else None,
        )


def plan_sync(
    stores: Sequence[Store],
    results: Mapping[str, StoreResult],
    *,
    current: Tab,
    now: datetime,
    decisions: Mapping[str, Decision] | None = None,
    id_column: str = "ID",
) -> list[CellUpdate]:
    """The cell updates that bring the tab's machine-owned columns up to date. See `build_plan` for the notes too."""
    return list(build_plan(stores, results, current=current, now=now, decisions=decisions, id_column=id_column).updates)


def build_plan(
    stores: Sequence[Store],
    results: Mapping[str, StoreResult],
    *,
    current: Tab,
    now: datetime,
    decisions: Mapping[str, Decision] | None = None,
    id_column: str = "ID",
) -> SyncPlan:
    """The plan for one tab: its cell updates in sheet order, and notes on what was deliberately left alone."""
    if current.layout != "table" or current.header_row is None or id_column not in current.header:
        raise ValidationError(
            f"tab {current.name!r} is not a table with an {id_column!r} column", hint="check stores.sync_tab in config"
        )
    rows = current.index_by(id_column)
    notes: list[SyncNote] = []
    changes: list[_Change] = []
    listed: set[str] = set()
    for store in stores:
        listed.add(store.id)
        located = rows.get(store.id)
        if located is None:
            notes.append(SyncNote("missing_row", store.id, "", f"no row in {current.name!r}; rows are added by people"))
            continue
        wants = _wants(store, results.get(store.id), (decisions or {}).get(store.id), notes)
        changes.extend(_row_changes(store.id, located, wants, now=now, notes=notes))
    for store_id in sorted(set(results) - listed):
        if is_added_id(store_id):
            continue  # a store added by hand (carttrace.stores.added) has no sheet row to sync, by design
        notes.append(SyncNote("unknown_store", store_id, "", "results name a store the queue does not list"))

    columns = dict(zip(current.header, current.columns, strict=True))
    updates: list[CellUpdate] = []
    used = {change.header for change in changes}
    next_col = max(current.columns, default=0) + 1
    for column in MACHINE_COLUMNS:
        if column.header in columns or column.header not in used:
            continue
        columns[column.header] = next_col
        cell = f"{column_letter(next_col)}{current.header_row}"
        updates.append(
            CellUpdate(
                tab=current.name,
                row_key="",
                column=column.header,
                cell=cell,
                value=column.header,
                previous="",
                reason="header",
            )
        )
        notes.append(SyncNote("new_column", "", column.header, f"adds column {column.header!r} at {cell}"))
        next_col += 1
    updates.extend(change.update(current.name, columns) for change in changes)
    updates.sort(key=_position)
    check_writable(u.column for u in updates)
    return SyncPlan(tab=current.name, updates=tuple(updates), notes=tuple(notes))


def resolve_updates(
    fresh: Tab, updates: Sequence[CellUpdate], *, id_column: str = "ID"
) -> tuple[list[CellUpdate], list[tuple[CellUpdate, str]]]:
    """Re-address planned updates against a fresh read of the tab, just before a live write.

    Rows are found again by their id and columns by their header, so a sheet re-sorted since planning is still written
    in the right place. An update whose cell no longer holds the value it was planned against is skipped, never
    forced: a person (or another sync) changed it, and the next plan will see that.
    """
    rows = fresh.index_by(id_column)
    columns = dict(zip(fresh.header, fresh.columns, strict=True))
    present = set(columns)
    next_col = max(fresh.columns, default=0) + 1
    resolved: list[CellUpdate] = []
    skipped: list[tuple[CellUpdate, str]] = []
    for update in sorted(updates, key=lambda u: not u.is_header):
        if update.is_header:
            if update.column in present:
                skipped.append((update, "column already exists"))
                continue
            columns[update.column] = next_col
            resolved.append(replace(update, cell=f"{column_letter(next_col)}{fresh.header_row}"))
            next_col += 1
            continue
        located = rows.get(update.row_key)
        col = columns.get(update.column)
        if located is None or col is None:
            skipped.append((update, f"no row {update.row_key}" if located is None else f"no column {update.column!r}"))
            continue
        row_number, record = located
        have = record.get(update.column, "") if update.column in present else ""
        if have != update.previous:
            skipped.append((update, f"cell changed since planning: it now holds {have!r}"))
            continue
        resolved.append(replace(update, cell=f"{column_letter(col)}{row_number}"))
    resolved.sort(key=_position)
    return resolved, skipped


def preflight_text(result: StoreResult) -> str:
    """The "Protocol preflight result" cell: outcome first, then versions, then what was seen."""
    parts: list[str] = [result.preflight or "unknown"]
    if result.protocols:
        parts.append(", ".join(f"{surface} {version}" for surface, version in sorted(result.protocols.items())))
    if result.http_status is not None and result.preflight != "ok":
        parts.append(f"HTTP {result.http_status}")
    if result.robots_allowed is False:
        parts.append("robots.txt disallows carttrace")
    if result.detail:
        parts.append(result.detail)
    return "; ".join(parts)


def eligibility_text(answer: str | list[str] | bool) -> str:
    """The "Collector eligibility" cell for a decision's answer: "needs_review" -> "needs review"."""
    if isinstance(answer, bool):
        return "eligible" if answer else "not eligible"
    if isinstance(answer, list):
        return ", ".join(a.replace("_", " ") for a in answer)
    return answer.replace("_", " ")


def _wants(
    store: Store, result: StoreResult | None, decision: Decision | None, notes: list[SyncNote]
) -> dict[str, _Want]:
    wants = {
        "merchant_group_id": _Want(store.merchant_group_id, "merchant_group"),
        "split": _Want(store.split, "split"),
    }
    if result is not None:
        if result.platform:
            wants["platform_confirmed"] = _Want(result.platform, "preflight")
        if result.market:
            wants["market_confirmed"] = _Want(result.market, "preflight")
        if result.preflight:
            wants["preflight_result"] = _Want(preflight_text(result), "preflight")
        if result.traces_collected is not None:
            wants["traces_collected"] = _Want(str(result.traces_collected), "collector")
        if result.latest_outcome:
            wants["latest_outcome"] = _Want(result.latest_outcome, "collector")
    if decision is not None:
        label = eligibility_text(decision.answer)
        if decision.effective:
            reason = f"decision:{decision.type_id}@v{decision.type_version}"
            wants["collector_eligibility"] = _Want(label, reason, decision)
        else:
            why = f"backend error: {decision.error}" if decision.error else f"{decision.status} decision"
            notes.append(SyncNote("shadow", store.id, "Collector eligibility", f"{label!r} not written ({why})"))
    return wants


def _row_changes(
    row_key: str,
    located: tuple[int, dict[str, str]],
    wants: Mapping[str, _Want],
    *,
    now: datetime,
    notes: list[SyncNote],
) -> list[_Change]:
    """One row's changed machine cells, plus its "Machine updated at" stamp when anything changed."""
    row, record = located
    out: list[_Change] = []
    for column in MACHINE_COLUMNS:
        want = wants.get(column.key)
        have = record.get(column.header, "")
        if want is None or have == want.value:
            continue
        if column.locked and have:
            detail = f"sheet has {have!r}, computed {want.value!r}; left as is"
            notes.append(SyncNote("locked", row_key, column.header, detail))
            continue
        out.append(_Change(row_key, row, column.header, want, have))
    if out:
        stamp = machine_column("machine_updated_at").header
        out.append(_Change(row_key, row, stamp, _Want(iso(now), "clock"), record.get(stamp, "")))
    return out


def _position(update: CellUpdate) -> tuple[int, int, int]:
    row, col = parse_ref(update.cell)
    return (0 if update.is_header else 1, row, col)

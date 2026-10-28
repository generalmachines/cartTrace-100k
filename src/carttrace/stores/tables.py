"""From a raw tab grid to the rows a snapshot keeps: find the real header under any title rows, name every column,
drop empty rows, and remember which sheet row and column each kept value came from.

Why it is careful. The team sheet is made for people: on the queue and evidence tabs a merged title and a merged
explainer paragraph sit above the header, which is on row 3. People change that shape without warning (another
explainer row, a group header over some columns), and the sync later writes back cell by cell. So a table is found by
its anchor header (the "ID" column) rather than by position, columns are named by their header text, and every kept
row carries its sheet row number; positions from an older read are never trusted for a write.

Two layouts:
- table: a tab whose header row starts with the anchor. Title rows above it are dropped; a group header row with
  merged cells over sub-headers is tolerated (the sub-header names the column).
- sheet: any other tab (the summary, the sampling plan). Non-empty rows are kept as they are, and the columns are
  named by their sheet letters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from carttrace.stores.text import collapse_ws, nfc
from carttrace.stores.xlsx import CellRange, column_letter

Layout = Literal["table", "sheet"]

MAX_HEADER_SCAN = 50
"""Rows searched for the anchor header. The team sheet's header is on row 3."""


@dataclass(frozen=True, slots=True)
class Tab:
    """One tab as a snapshot keeps it, or as a sheet backend reads it back."""

    name: str
    layout: Layout
    header: tuple[str, ...]
    """Column names: header text for a table, sheet letters for the sheet layout."""
    columns: tuple[int, ...]
    """Sheet column (1-based) of each entry in `header`."""
    header_row: int | None
    """Sheet row of the header (the sub-header row when the header has two rows); None for the sheet layout."""
    rows: tuple[tuple[str, ...], ...]
    row_numbers: tuple[int, ...]
    """Sheet row of each entry in `rows`."""

    def column_of(self, name: str) -> int | None:
        """Sheet column of the column named `name`, or None when the tab has no such column."""
        try:
            return self.columns[self.header.index(name)]
        except ValueError:
            return None

    def records(self) -> list[dict[str, str]]:
        """Each row as {column name: value}."""
        return [dict(zip(self.header, row, strict=True)) for row in self.rows]

    def index_by(self, column: str) -> dict[str, tuple[int, dict[str, str]]]:
        """Rows keyed by the value in `column`, as {key: (sheet row, record)}. Empty keys are left out; the first row
        wins when a key repeats, because the sheet's own order is the only tie-break a person can see."""
        out: dict[str, tuple[int, dict[str, str]]] = {}
        for number, record in zip(self.row_numbers, self.records(), strict=True):
            key = record.get(column, "")
            if key and key not in out:
                out[key] = (number, record)
        return out


def normalise_cell(text: str) -> str:
    """A data cell as the snapshot keeps it: composed Unicode, Unix line breaks, no surrounding whitespace."""
    return nfc(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalise_header(text: str) -> str:
    """A header as a column name: composed Unicode and single spaces, so a header typed over two lines still matches."""
    return collapse_ws(nfc(text))


def tab_from_grid(
    name: str, grid: Sequence[Sequence[str]], *, merges: Sequence[CellRange] = (), anchor: str | None = "ID"
) -> Tab:
    """The table under the anchor header when the tab has one, otherwise the tab in the sheet layout."""
    width = max((len(row) for row in grid), default=0)
    clean = [[normalise_cell(row[c]) if c < len(row) else "" for c in range(width)] for row in grid]
    found = _find_anchor(clean, anchor) if anchor else None
    if found is None:
        return _sheet_layout(name, clean)
    header_index, anchor_col = found
    return _table_layout(name, clean, merges, header_index, anchor_col)


def _find_anchor(grid: list[list[str]], anchor: str) -> tuple[int, int] | None:
    """(row index, 1-based column) of the first row, near the top, whose first non-empty cell is the anchor."""
    wanted = normalise_header(anchor).casefold()
    for index, row in enumerate(grid[:MAX_HEADER_SCAN]):
        filled = [(col, value) for col, value in enumerate(row, start=1) if value]
        if len(filled) >= 2 and normalise_header(filled[0][1]).casefold() == wanted:
            return index, filled[0][0]
    return None


def _table_layout(
    name: str, grid: list[list[str]], merges: Sequence[CellRange], header_index: int, anchor_col: int
) -> Tab:
    width = len(grid[0]) if grid else 0
    header_row = header_index + 1
    sub_index: int | None = None
    has_group = any(m.first_row == header_row and m.last_col > m.first_col for m in merges)
    if has_group and header_index + 1 < len(grid) and not grid[header_index + 1][anchor_col - 1]:
        sub_index = header_index + 1
    first_data = (sub_index if sub_index is not None else header_index) + 1

    names: list[str] = []
    for col in range(1, width + 1):
        top = _label(grid, merges, header_index, col)
        sub = _label(grid, merges, sub_index, col) if sub_index is not None else ""
        names.append(normalise_header(sub or top))

    data: list[tuple[int, list[str]]] = [
        (index + 1, row) for index, row in enumerate(grid[first_data:], start=first_data) if any(row)
    ]
    keep = [col for col in range(1, width + 1) if names[col - 1] or any(row[col - 1] for _, row in data)]
    header = _unique([names[col - 1] or f"column_{column_letter(col)}" for col in keep])
    return Tab(
        name=name,
        layout="table",
        header=tuple(header),
        columns=tuple(keep),
        header_row=(sub_index if sub_index is not None else header_index) + 1,
        rows=tuple(tuple(row[col - 1] for col in keep) for _, row in data),
        row_numbers=tuple(number for number, _ in data),
    )


def _sheet_layout(name: str, grid: list[list[str]]) -> Tab:
    kept = [(index + 1, row) for index, row in enumerate(grid) if any(row)]
    width = max((max(c for c, v in enumerate(row, start=1) if v) for _, row in kept), default=0)
    return Tab(
        name=name,
        layout="sheet",
        header=tuple(column_letter(c) for c in range(1, width + 1)),
        columns=tuple(range(1, width + 1)),
        header_row=None,
        rows=tuple(tuple(row[:width]) for _, row in kept),
        row_numbers=tuple(number for number, _ in kept),
    )


def _label(grid: list[list[str]], merges: Sequence[CellRange], index: int | None, col: int) -> str:
    """The text shown at (row index, col): the cell itself, or the top-left of the merged range covering it."""
    if index is None:
        return ""
    value = grid[index][col - 1]
    if value:
        return value
    for merge in merges:
        if merge.contains(index + 1, col):
            top = merge.first_row - 1
            if 0 <= top < len(grid) and merge.first_col - 1 < len(grid[top]):
                return grid[top][merge.first_col - 1]
    return ""


def _unique(names: list[str]) -> list[str]:
    """Repeated names get " (2)", " (3)": a column is addressed by name, so two columns may not share one."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return out

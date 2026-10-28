"""What a sheet backend promises (Lego kind `stores.sheet`): read a tab, write machine-owned cells.

Implementations, chosen by name in config.json `components["stores.sheet"]`:
- `csv` (`carttrace.stores.sheet_csv`): reads the latest committed snapshot and writes the planned cells as files for a person
  to import. Nothing reaches the team sheet without a human; this is the default.
- `google_sheets` (`carttrace.stores.sheet_google`): reads and writes the live sheet through the Sheets API v4.

Both refuse any cell outside the machine-owned columns (`carttrace.stores.columns.check_writable`) before writing anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from carttrace.stores.tables import Tab


@dataclass(frozen=True, slots=True)
class CellUpdate:
    """One planned cell write, addressed by row key and column name; `cell` is where that was when planned."""

    tab: str
    row_key: str
    """Queue id of the row, e.g. "store-001"; "" for a header cell the sync adds."""
    column: str
    """Header of the target column."""
    cell: str
    """A1 address in the tab the plan was read from, e.g. "S4". A live backend re-resolves it before writing."""
    value: str
    previous: str
    """What the cell held when the plan was made; a live backend skips the write when the cell no longer holds it."""
    reason: str
    """What produced the value: header, merchant_group, split, preflight, collector, clock, or decision:<type>@v<n>."""
    decided_by: str | None = None
    """For decisions: rule, clef-flash, clef or abstain."""
    evidence_sha: str | None = None
    """For decisions: sha256 of the evidence, which joins the write to the decision log."""

    @property
    def is_header(self) -> bool:
        return self.row_key == ""


@dataclass(frozen=True, slots=True)
class WriteReport:
    backend: str
    target: str
    """Where the writes went: a directory of files for manual import, or the sheet's URL."""
    written: tuple[CellUpdate, ...]
    skipped: tuple[tuple[CellUpdate, str], ...] = ()
    """Updates a live backend did not write, with the reason (the row moved away, the cell changed since planning)."""


class SheetBackend(Protocol):
    @property
    def name(self) -> str: ...

    def read_tab(self, name: str) -> Tab:
        """The tab as this backend sees it: the latest snapshot for csv, the live sheet for google_sheets."""
        ...

    def write_cells(self, updates: Sequence[CellUpdate]) -> WriteReport:
        """Write the updates. Raises SafetyError, before writing anything, if one targets a column people own."""
        ...

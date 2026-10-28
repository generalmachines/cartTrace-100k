"""The team sheet's columns: which field each header fills, and who owns each column.

Why ownership is a table. The sheet is shared by people and code. People own the notes, the evidence a store was
nominated on, the publication permission and the gate approval; code owns a short list of columns it computes. The
sync writes only through `check_writable`, an allowlist of the machine-owned headers below, and every header people
own is listed explicitly too, so a test can prove the two sets never overlap and that no plan touches a human column.

Two machine columns are locked: once the merchant group id or the split is in the sheet, the sync never changes it,
because moving a store between train and the locked test set after traces exist would leak the test set. A changed
value is reported as a conflict for a person to resolve.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from carttrace.core.errors import SafetyError

QUEUE_FIELDS: dict[str, str] = {
    "ID": "id",
    "Category": "category",
    "Store": "name",
    "Website": "website",
    "Platform": "platform",
    "Market (delivery)": "market",
    "Parent company": "parent_company",
    "Status": "status",
    "What we saw": "what_we_saw",
    "What to do next": "what_to_do_next",
    "Test product / fixture": "test_fixture",
    "Last observed": "last_observed",
}
"""Store queue header to Store field. Every one is required."""

EVIDENCE_FIELDS: dict[str, str] = {
    "category_id": "category_index",
    "url": "url",
    "source_status": "source_status",
    "checked_at": "checked_at",
    "fit_reason": "fit_reason",
    "observed_evidence": "observed_evidence",
    "parent_group": "parent_group",
    "evidence_url": "evidence_url",
    "canonical_host": "canonical_host",
    "duplicate_host_count": "duplicate_host_count",
    "publication_permission": "publication_permission",
    "eligibility": "eligibility",
    "Platform evidence": "platform_evidence",
    "Delivery market": "delivery_market",
    "Browser preflight result": "browser_preflight_result",
    "Task fixture / SKU": "task_fixture",
    "Gate approval reference": "gate_approval_reference",
}
"""Evidence header to Store field, for the nomination evidence. Every one is required, as is "ID"."""

EVIDENCE_OPTIONAL_FIELDS: dict[str, str] = {
    "Merchant group ID": "sheet_merchant_group_id",
    "Collector eligibility": "collector_eligibility",
    "Split": "sheet_split",
}
"""Machine-owned evidence columns read back into the store when present; the sync creates them when they are not."""

EVIDENCE_CROSS_CHECK: dict[str, str] = {
    "store_name": "Store",
    "status": "Status",
    "Platform": "Platform",
    "canonical_host": "Website",
}
"""Evidence columns that repeat a queue column (evidence header to queue header), read only to warn when the two tabs
disagree."""


@dataclass(frozen=True, slots=True)
class MachineColumn:
    key: str
    header: str
    locked: bool = False
    """Once the sheet holds a value, the sync reports a different one as a conflict instead of writing it."""


MACHINE_COLUMNS: tuple[MachineColumn, ...] = (
    MachineColumn("platform_confirmed", "Platform confirmed"),
    MachineColumn("market_confirmed", "Delivery market confirmed"),
    MachineColumn("preflight_result", "Protocol preflight result"),
    MachineColumn("collector_eligibility", "Collector eligibility"),
    MachineColumn("merchant_group_id", "Merchant group ID", locked=True),
    MachineColumn("split", "Split", locked=True),
    MachineColumn("traces_collected", "Traces collected"),
    MachineColumn("latest_outcome", "Latest outcome"),
    MachineColumn("machine_updated_at", "Machine updated at"),
)
"""Every column the sync may write, in the order new ones are added to the right of the tab."""

MACHINE_HEADERS: frozenset[str] = frozenset(column.header for column in MACHINE_COLUMNS)

HUMAN_COLUMNS: frozenset[str] = frozenset(
    {
        *QUEUE_FIELDS,
        "ID",
        *EVIDENCE_FIELDS,
        *EVIDENCE_CROSS_CHECK,
    }
)
"""Every column people own: the whole queue tab, and the evidence tab's nomination columns, including the status
notes, what we saw, the publication permission and the gate approval reference."""

if MACHINE_HEADERS & HUMAN_COLUMNS:
    raise ValueError(f"columns owned by both people and code: {sorted(MACHINE_HEADERS & HUMAN_COLUMNS)}")


def machine_column(key: str) -> MachineColumn:
    for column in MACHINE_COLUMNS:
        if column.key == key:
            return column
    raise KeyError(key)


def check_writable(columns: Iterable[str]) -> None:
    """Refuse any write to a column that is not machine-owned. Every sheet backend calls this before writing."""
    refused = sorted({c for c in columns if c not in MACHINE_HEADERS or c in HUMAN_COLUMNS})
    if refused:
        raise SafetyError(
            f"refusing to write columns people own or nobody declared: {refused}",
            hint="only the machine-owned columns in carttrace.stores.columns.MACHINE_COLUMNS are written by the sync",
        )

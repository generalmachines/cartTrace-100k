"""Layer 5: the store queue from the team sheet, the 25 categories and their checklist hypotheses, and the sheet sync.

- `snapshot`: `carttrace stores pull`, the team sheet as dated CSVs plus a sha256 manifest (stdlib xlsx reader in `xlsx`)
- `queue`: `load_queue`, typed `Store` records with merchant groups (`merchant`) and splits (`split`)
- `added`: stores added by hand beside the sheet (`add_store`, var/stores/added.jsonl, ids store-a001...), and
  `planning_queue`, the sheet's stores then the added ones: the queue the collector and the autopilot plan from
- `categories`: `load_categories`, the validated category files under data/categories/
- `plan`, `sync`: the dry-run-first sync of machine-owned sheet columns through a `stores.sheet` backend
- `results`: the `mc.stores.result.v1` record and its fold; `resultsdb`: the results database every record goes to,
  one row per observation with its provenance (live, replica, twin, unknown) and a view per provenance that never
  mixes them; `migrate`: the one-time move of the old results file into it (`carttrace stores results migrate`)
- `eligibility`: collector eligibility through the decision engine (`sheet.eligibility`)

- `preflight`: what each store exposes to agents, through carttrace.live's discovery (`carttrace stores preflight`, a dry run
  unless --live). Import it from `carttrace.stores.preflight`, so `import carttrace.stores` never loads the live client.
"""

from carttrace.stores.added import AddedStore, add_store, added_queue, added_stores, planning_queue
from carttrace.stores.categories import Category, CategoryCatalog, Hypothesis, IntentTemplate, SpecField, load_categories
from carttrace.stores.columns import HUMAN_COLUMNS, MACHINE_COLUMNS, MACHINE_HEADERS, check_writable
from carttrace.stores.merchant import MerchantGroup, group_merchants, parent_key, registrable_domain
from carttrace.stores.model import AddedMode, PlatformKey, Split, StatusKey, Store
from carttrace.stores.plan import SyncNote, SyncPlan, build_plan, plan_sync
from carttrace.stores.queue import load_queue, merge_added
from carttrace.stores.results import StoreResult, fold_results, load_results
from carttrace.stores.resultsdb import ResultRow, ResultsDB, RunRow, export_parquet, load_store_results
from carttrace.stores.service import CellUpdate, SheetBackend, WriteReport
from carttrace.stores.settings import StoresSettings, load_settings
from carttrace.stores.snapshot import SnapshotManifest, latest_snapshot, pull_snapshot, read_snapshot_tab
from carttrace.stores.split import split_for
from carttrace.stores.tables import Tab

PLUGINS = {
    "stores.sheet": {
        "csv": "carttrace.stores.sheet_csv:CsvSheetBackend",
        "google_sheets": "carttrace.stores.sheet_google:GoogleSheetsBackend",
    },
}

__all__ = [
    "HUMAN_COLUMNS",
    "MACHINE_COLUMNS",
    "MACHINE_HEADERS",
    "PLUGINS",
    "AddedMode",
    "AddedStore",
    "Category",
    "CategoryCatalog",
    "CellUpdate",
    "Hypothesis",
    "IntentTemplate",
    "MerchantGroup",
    "PlatformKey",
    "ResultRow",
    "ResultsDB",
    "RunRow",
    "SheetBackend",
    "SnapshotManifest",
    "SpecField",
    "Split",
    "StatusKey",
    "Store",
    "StoreResult",
    "StoresSettings",
    "SyncNote",
    "SyncPlan",
    "Tab",
    "WriteReport",
    "add_store",
    "added_queue",
    "added_stores",
    "build_plan",
    "check_writable",
    "export_parquet",
    "fold_results",
    "group_merchants",
    "latest_snapshot",
    "load_categories",
    "load_queue",
    "load_results",
    "load_settings",
    "load_store_results",
    "merge_added",
    "parent_key",
    "plan_sync",
    "planning_queue",
    "pull_snapshot",
    "read_snapshot_tab",
    "registrable_domain",
    "split_for",
]

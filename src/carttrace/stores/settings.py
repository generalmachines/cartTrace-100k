"""What the stores package reads from config, as one typed value.

Why an adapter. The pull, the queue loader and the sync need a handful of settings: the sheet id and tab names
(config.json `stores`), the sheet backend (`components["stores.sheet"]`), the data and var paths, and our user agent.
Every function here takes a `StoresSettings`, so a test builds one literal and never reads the real config, and
`from_settings` is the only code that knows the shape of carttrace.config's settings. Loading, overrides and secrets stay in
carttrace.config: `load_settings` calls its `settings()`, and `secret` calls its `secret(name)`, which never logs a value.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

import carttrace.config
from carttrace.config.schema import Settings

DEFAULT_SHEET_ID = "unset"


class StoresSettings(BaseModel):
    """Every setting the stores package reads. Paths are absolute."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    home: Path
    data_dir: Path
    var_dir: Path
    sheet_id: str = DEFAULT_SHEET_ID
    """The team's store queue sheet, which the replay kit never reads: set stores.sheet_id in config.local.json."""
    queue_tab: str = "Store queue"
    evidence_tab: str = "Evidence"
    sync_tab: str | None = None
    """Tab the sync writes machine-owned columns to; the evidence tab when unset."""
    id_column: str = "ID"
    """Header of the column holding the queue id (store-001) on every table tab; it also anchors header detection."""
    credentials_env: str = "GOOGLE_APPLICATION_CREDENTIALS"
    """Name of the variable holding the service-account key (or the path to it) for API writes; never the value."""
    sheet_backend: str = "csv"
    """config.json components["stores.sheet"]: csv (files for manual import) or google_sheets (API writes)."""
    user_agent: str = "carttrace/0.1"
    http_timeout_s: float = Field(default=60.0, gt=0)
    results_db: Path | None = None
    """config.json collect.results_db; var/stores/results.sqlite under var_dir when unset."""

    @property
    def stores_dir(self) -> Path:
        return self.data_dir / "stores"

    @property
    def categories_dir(self) -> Path:
        return self.data_dir / "categories"

    @property
    def aliases_path(self) -> Path:
        return self.stores_dir / "merchant-aliases.yaml"

    @property
    def sheet_out_dir(self) -> Path:
        return self.var_dir / "sheet"

    @property
    def target_tab(self) -> str:
        return self.sync_tab or self.evidence_tab

    @property
    def results_db_path(self) -> Path:
        """The results database the sync reads (carttrace.stores.resultsdb)."""
        return self.results_db or self.var_dir / "stores" / "results.sqlite"


def from_settings(settings: Settings) -> StoresSettings:
    """The stores view of carttrace.config's settings."""
    return StoresSettings(
        home=settings.home,
        data_dir=settings.data_dir,
        var_dir=settings.var_dir,
        sheet_id=settings.stores.sheet_id,
        queue_tab=settings.stores.queue_tab,
        evidence_tab=settings.stores.evidence_tab,
        credentials_env=settings.stores.credentials_env,
        sheet_backend=settings.component("stores.sheet"),
        user_agent=settings.user_agent,
        results_db=settings.collect.results_db,
    )


def load_settings() -> StoresSettings:
    """The stores view of the process's settings (config.json, then config.local.json, then CARTTRACE__ overrides)."""
    return from_settings(carttrace.config.settings())


def secret(name: str) -> str | None:
    """A secret by variable name, from the environment or a secrets file named in config. Never printed."""
    return carttrace.config.secret(name)

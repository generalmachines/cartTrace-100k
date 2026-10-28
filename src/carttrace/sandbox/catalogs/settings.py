"""What the catalogs read from config, as one typed value.

Why an adapter. Fetching, indexing and store building need five settings: where downloads go
(sandbox.catalogs_dir), where the checked-in source manifests and category files live (paths.data), whether
restricted sources may build stores (sandbox.allow_restricted_catalogs), the default catalog, and our user agent.
Every function takes a `CatalogSettings`, so a test builds one literal and never reads the real config, and
`from_settings` is the only code that knows the shape of carttrace.config's settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from carttrace.config.schema import Settings
from carttrace.trace.capture import CAPTURES_DIR
from carttrace.trace.capture_v2 import captures_root


@dataclass(frozen=True, slots=True)
class CatalogSettings:
    """Every setting the catalogs read. Paths are absolute in production; tests may pass any directory."""

    catalogs_dir: Path
    """Downloads and indexes, one directory per source (settings.sandbox.catalogs_dir, gitignored under var/)."""
    data_dir: Path
    """Checked-in source manifests, category maps and coverage: data/catalogs/."""
    categories_dir: Path
    """The 25 category files the maps point at: data/categories/."""
    allow_restricted: bool = False
    """settings.sandbox.allow_restricted_catalogs: lets a restricted source (Lazada) build stores for internal
    research. Its bundles stay training_allowed=false and publication_allowed=false whatever this says."""
    user_agent: str = "carttrace/0.1"
    default_catalog: str = "abo"
    captures_dir: Path | None = None
    """Store captures replicas are built from (<paths.var>/captures, carttrace.trace.capture); None reads none."""
    open_pool_markets: str = "mixed"
    """settings.sandbox.open_pool_markets: mixed (every member in one US store) or home (one group a store, D91)."""
    store_captures_dir: Path | None = None
    """The capture pass's editions (<paths.var>/store/captures, carttrace.trace.capture_v2), which a stand-in's products
    are matched to (D104); None matches none."""

    @classmethod
    def from_settings(cls, settings: Settings) -> CatalogSettings:
        return cls(
            catalogs_dir=settings.sandbox.catalogs_dir,
            data_dir=settings.data_dir / "catalogs",
            categories_dir=settings.data_dir / "categories",
            allow_restricted=settings.sandbox.allow_restricted_catalogs,
            user_agent=settings.user_agent,
            default_catalog=settings.sandbox.default_catalog,
            captures_dir=settings.paths.var / CAPTURES_DIR,
            open_pool_markets=settings.sandbox.open_pool_markets,
            store_captures_dir=captures_root(settings.paths.var),
        )


def resolve(settings: Settings | CatalogSettings | None = None) -> CatalogSettings:
    """The catalog view of `settings`, or of the process's settings when none is given."""
    if isinstance(settings, CatalogSettings):
        return settings
    if settings is None:
        import carttrace.config

        settings = carttrace.config.settings()
    return CatalogSettings.from_settings(settings)

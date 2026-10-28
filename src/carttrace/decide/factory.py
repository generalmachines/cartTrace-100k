"""`build_engine`: the one way the rest of the kit gets a decision engine, rules alone (carttrace.decide.router)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from carttrace.config.schema import Settings
from carttrace.core.clock import Clock
from carttrace.core.registry import resolve
from carttrace.decide.catalog import Catalog, load_catalog
from carttrace.decide.router import Router, RouterConfig
from carttrace.decide.service import DecisionEngine, DecisionSink

SINK_KIND = "decide.sink"
DEFAULT_SINK = "jsonl"


def build_engine(
    settings: Settings | None = None,
    *,
    sink: DecisionSink | None = None,
    clock: Clock | None = None,
    catalog: Catalog | None = None,
    keep_evidence: bool = False,
) -> DecisionEngine:
    """The decision engine on the types in paths.data/decisions, writing to the configured decision log."""
    resolved = settings if settings is not None else load_settings()
    return Router(
        catalog if catalog is not None else load_catalog(catalog_dir(resolved)),
        sink=sink if sink is not None else default_sink(resolved),
        clock=clock,
        config=RouterConfig(keep_evidence=keep_evidence),
    )


def load_settings() -> Settings:
    from carttrace.config import settings

    return settings()


def catalog_dir(settings: Settings) -> Path:
    return Path(settings.data_dir) / "decisions"


def default_sink(settings: Settings) -> DecisionSink:
    """The DecisionSink config.json chooses: jsonl (the default) or memory."""
    name = settings.components.kinds().get(SINK_KIND, DEFAULT_SINK)
    return cast(DecisionSink, resolve(SINK_KIND, name)(settings))

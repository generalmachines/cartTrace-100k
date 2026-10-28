"""The decisions the shopper, the supervisor and triage make during a journey, by rule alone.

Decision types are data in data/decisions/*.yaml; each names the rule that answers it. A decision no rule settles
abstains with the type's safe default, as it did when the trace was recorded. Everything else asks for an engine with
`build_engine()` and holds the `DecisionEngine` interface.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from carttrace.decide.catalog import Catalog, load_catalog
    from carttrace.decide.factory import build_engine
    from carttrace.decide.log import DecisionLog, MemorySink, TeeSink, read_decisions
    from carttrace.decide.router import Router, RouterConfig
    from carttrace.decide.rules import RULES, RuleBook
    from carttrace.decide.service import DecisionEngine, DecisionSink
    from carttrace.decide.types import Answer, Decision, DecisionType, StageAnswer

PLUGINS = {
    "decide.sink": {
        "jsonl": "carttrace.decide.log:jsonl_sink",
        "memory": "carttrace.decide.log:memory_sink",
    },
}

_EXPORTS = {
    "Answer": "carttrace.decide.types",
    "Catalog": "carttrace.decide.catalog",
    "Decision": "carttrace.decide.types",
    "DecisionEngine": "carttrace.decide.service",
    "DecisionLog": "carttrace.decide.log",
    "DecisionSink": "carttrace.decide.service",
    "DecisionType": "carttrace.decide.types",
    "MemorySink": "carttrace.decide.log",
    "RULES": "carttrace.decide.rules",
    "Router": "carttrace.decide.router",
    "RouterConfig": "carttrace.decide.router",
    "RuleBook": "carttrace.decide.rules",
    "StageAnswer": "carttrace.decide.types",
    "TeeSink": "carttrace.decide.log",
    "build_engine": "carttrace.decide.factory",
    "load_catalog": "carttrace.decide.catalog",
    "read_decisions": "carttrace.decide.log",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'carttrace.decide' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


__all__ = [
    "PLUGINS",
    "RULES",
    "Answer",
    "Catalog",
    "Decision",
    "DecisionEngine",
    "DecisionLog",
    "DecisionSink",
    "DecisionType",
    "MemorySink",
    "Router",
    "RouterConfig",
    "RuleBook",
    "StageAnswer",
    "TeeSink",
    "build_engine",
    "load_catalog",
    "read_decisions",
]

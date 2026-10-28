"""Layer 2: protocols as versioned data.

- `profile`: ProtocolProfile, the exact versions a store speaks or a journey runs under
- `tools`: ToolSurface, the one shape every protocol endpoint takes for the agent
- `lock`, `vendor`, `fetch`: the pins in protocols.lock.toml, and fetching, checking and relocking vendor/
- `registry`: files, schemas, offline validators, UCP MCP tools and OpenAPI documents per protocol version
- `presets`: named profiles from data/protocols/presets.json
- `diff`: deterministic schema diffs between two versions

The contracts (profile, tools) import eagerly; the registry, presets and diff load on first use, so a module that
only needs `ProtocolProfile` (every trace reader) never imports jsonschema. Two functions share a name with their
module and are imported from it: `carttrace.protocols.registry.registry()` (the shared Registry) and
`carttrace.protocols.presets.profile(name)` (one preset).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from carttrace.protocols.profile import ProtocolProfile
from carttrace.protocols.tools import Mode, SurfaceKind, ToolResult, ToolSpec, ToolSurface

if TYPE_CHECKING:
    from carttrace.protocols.diff import Change, VersionDiff, diff_versions
    from carttrace.protocols.presets import load_presets
    from carttrace.protocols.registry import Registry, SchemaValidator, VersionLint

_LAZY = {
    "Change": "carttrace.protocols.diff",
    "VersionDiff": "carttrace.protocols.diff",
    "diff_versions": "carttrace.protocols.diff",
    "load_presets": "carttrace.protocols.presets",
    "Registry": "carttrace.protocols.registry",
    "SchemaValidator": "carttrace.protocols.registry",
    "VersionLint": "carttrace.protocols.registry",
}

__all__ = [
    "Change",
    "Mode",
    "ProtocolProfile",
    "Registry",
    "SchemaValidator",
    "SurfaceKind",
    "ToolResult",
    "ToolSpec",
    "ToolSurface",
    "VersionDiff",
    "VersionLint",
    "diff_versions",
    "load_presets",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'carttrace.protocols' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)

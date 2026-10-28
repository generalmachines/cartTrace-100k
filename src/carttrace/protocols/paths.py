"""Where the protocol layer's files live: the lock, the vendored tree and the presets.

Why one module. The lock (`protocols.lock.toml`) sits at the repository root and names each source's destination
relative to that root, the registry reads the vendor directory, and presets live under the data directory. Settings
decide the directories (`config.json` `paths.vendor` and `paths.data`, overridable in `config.local.json`), so every
caller asks this module instead of building paths from the working directory, and a test builds `ProtocolPaths.at`
over a temporary directory instead of touching the real tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LOCK_NAME = "protocols.lock.toml"


@dataclass(frozen=True, slots=True)
class ProtocolPaths:
    root: Path
    """Repository root: the lock lives here and every `dest` in it is relative to it."""
    vendor: Path
    data: Path

    @property
    def lock(self) -> Path:
        return self.root / LOCK_NAME

    @property
    def presets(self) -> Path:
        return self.data / "protocols" / "presets.json"

    @classmethod
    def at(cls, root: Path) -> ProtocolPaths:
        """The default layout under one root: `vendor/` and `data/` beside the lock."""
        root = root.resolve()
        return cls(root=root, vendor=root / "vendor", data=root / "data")


def configured() -> ProtocolPaths:
    """The paths the process's settings name (carttrace.config: home, paths.vendor, paths.data)."""
    from carttrace.config import settings

    s = settings()
    return ProtocolPaths(root=s.home, vendor=s.paths.vendor, data=s.paths.data)

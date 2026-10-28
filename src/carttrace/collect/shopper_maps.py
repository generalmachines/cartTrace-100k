"""The shopper's decision maps, built with the manifest (sandbox v1 final, D115; carttrace.agents.decision_map).

Each sandbox or reference row's task is decided once, at build, on the row's own sealed world, exactly as the D104
build-time replay opens it (the task in its world's presented ids, the world with the task's sealed faults, its
served catalog), and the map is sealed into the manifest under `<envpack>/<task id>`. A mask row (D96) keeps the
shopper's reading of the withheld world it is served, and an impossible task has no witness to map."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carttrace.agents.decision_map import build_map, map_key
from carttrace.agents.products import fixtures_from_snapshot
from carttrace.collect.manifest import Manifest, ManifestRow, seal
from carttrace.collect.worlds import sealed_world
from carttrace.sandbox import EnvPack

MAPPED_MODES = frozenset({"sandbox", "reference"})


def mapped(row: ManifestRow) -> bool:
    """Whether a row's shopper decides from a build-time map."""
    exp = getattr(row, "experiment", None)  # a planner's stand-in row may carry none
    return (
        row.mode in MAPPED_MODES and getattr(row, "envpack", None) is not None and (exp is None or exp.kind != "mask")
    )


def shopper_maps(manifest: Manifest, envpacks: Path) -> dict[str, dict[str, Any]]:
    """Every mapped row's decision map, by `<envpack>/<task id>`, once per world and task."""
    out: dict[str, dict[str, Any]] = {}
    packs: dict[str, EnvPack] = {}
    for row in manifest.rows:
        if not mapped(row):
            continue
        assert row.envpack is not None
        key = map_key(row.envpack, row.task_id)
        task = manifest.task(row)
        if key in out or task.hidden_spec.get("impossible") is True:
            continue
        pack = packs.get(row.envpack)
        if pack is None:
            pack = packs[row.envpack] = EnvPack.load(envpacks / f"{row.envpack}.json")
        world, _ = sealed_world(pack, task)
        out[key] = build_map(task, fixtures_from_snapshot(world.snapshot))
    return dict(sorted(out.items()))


def with_shopper_maps(manifest: Manifest, envpacks: Path) -> Manifest:
    """`manifest` sealed again with its rows' shopper decision maps."""
    maps = shopper_maps(manifest, envpacks)
    if not maps:
        return manifest
    body = manifest.body()
    body.pop("schema", None)
    return seal(**{**body, "shopper_maps": maps})


__all__ = ["mapped", "shopper_maps", "with_shopper_maps"]

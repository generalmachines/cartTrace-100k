"""Layer 6: manifests (store x task x condition x profile x surface x mode), the async runner, the status file.

- `manifest`: ManifestPlan, Manifest (`mf-<sha12>`), `build_manifest`, `load_manifest`, `save_manifest`
- `conditions`: baseline, cross_border, version_variant, detail_hidden as data
- `worlds`: the sandbox world (envpack) of a sandbox or reference row, from a catalog and a store's profile
- `profiles`: each store's own protocol profile (profiles file, replica capture, preflight, then the preset), the
  surfaces it serves, and the surface a row runs on
- `coverage`: tasks spread over a store's catalog (product type, price band, decisive detail, a cap per product, a
  rotation window) and what they cover
- `inputs`: `make_manifest` and `load_inputs`, a manifest's inputs read from disk: the planning queue (the sheet's
  stores, then those added by hand), categories, preflight results, presets (`carttrace collect manifest`, the autopilot)
- `runner`: `run_manifest`, `RunSummary`, `RunControl` (drain); `scheduler`: the run's limits, pauses and skips
- `episode`: one episode from the first request to the closed bundle; `handoff`: a live cart to a sandbox checkout
- `rights`: the data rights a bundle carries (D10); `results`: one mc.stores.result.v1 row per bundle into the
  results database (carttrace.stores.resultsdb), with the bundle's provenance, for the sheet sync
- `supervisor`, `triage`, `decisions`: the decision engine in the loop, every decision also in the bundle
- `surfaces`: wire taps and buffers, the failure observer and the detail_hidden filter
- `matchers`, `excise`: which keys and text state each catalog detail, and the cut that hides exactly those
- `status`: the collector status file contract (`mc.collector-status.v1`) and its writer
- `ports`, `bindings`: what the collector calls in carttrace.agents, carttrace.live and carttrace.sandbox, by contracted name
- `carttrace collect manifest | run | status`

Names load on first use: carttrace.ops and carttrace.media import `carttrace.collect.status`, and must not pay for the runner.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from carttrace.agents.service import SupervisorVerdict
    from carttrace.collect.bindings import Bindings, default_bindings
    from carttrace.collect.inputs import ManifestInputs, load_inputs, make_manifest
    from carttrace.collect.manifest import Manifest, ManifestPlan, ManifestRow, build_manifest, load_manifest, save_manifest
    from carttrace.collect.runner import RunControl, RunSummary, run_manifest
    from carttrace.collect.status import CollectorStatus, EndpointLoad, read_status, write_status
    from carttrace.collect.supervisor import DecisionSupervisor

_EXPORTS = {
    "Bindings": "carttrace.collect.bindings",
    "CollectorStatus": "carttrace.collect.status",
    "DecisionSupervisor": "carttrace.collect.supervisor",
    "EndpointLoad": "carttrace.collect.status",
    "Manifest": "carttrace.collect.manifest",
    "ManifestInputs": "carttrace.collect.inputs",
    "ManifestPlan": "carttrace.collect.manifest",
    "ManifestRow": "carttrace.collect.manifest",
    "RunControl": "carttrace.collect.runner",
    "RunSummary": "carttrace.collect.runner",
    "SupervisorVerdict": "carttrace.agents.service",
    "build_manifest": "carttrace.collect.manifest",
    "default_bindings": "carttrace.collect.bindings",
    "load_inputs": "carttrace.collect.inputs",
    "load_manifest": "carttrace.collect.manifest",
    "make_manifest": "carttrace.collect.inputs",
    "read_status": "carttrace.collect.status",
    "run_manifest": "carttrace.collect.runner",
    "save_manifest": "carttrace.collect.manifest",
    "write_status": "carttrace.collect.status",
}

PLUGINS: dict[str, dict[str, str]] = {}
"""Replaceable collector parts (carttrace.core.registry). The parts it drives are other packages' blocks, bound in
`bindings.py`; none of its own has a second implementation yet."""


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'carttrace.collect' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


__all__ = [
    "PLUGINS",
    "Bindings",
    "CollectorStatus",
    "DecisionSupervisor",
    "EndpointLoad",
    "Manifest",
    "ManifestInputs",
    "ManifestPlan",
    "ManifestRow",
    "RunControl",
    "RunSummary",
    "SupervisorVerdict",
    "build_manifest",
    "default_bindings",
    "load_inputs",
    "load_manifest",
    "make_manifest",
    "read_status",
    "run_manifest",
    "save_manifest",
    "write_status",
]

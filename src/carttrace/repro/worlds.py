"""The worlds a published trace ran in, as CartTrace-100k ships them: one envpack and one manifest per world.

Why ship worlds. A trace says what the agent did and what the store answered, but a replay needs the store itself: its
products, policies, prices, stock and faults (the envpack) and the episode's plan (its row of the manifest: task,
persona, surface, rail, seed, condition). Rebuilding a world from the open catalogs would need every pool and the
code that cut it; shipping the world is about 200 KB of JSON, and makes every trace replayable on its own.

What is withheld. An envpack's provenance names the queue store it stood in for and the capture it was matched to,
and a manifest names that store by its queue id; the release publishes neither (D102). `withhold` blanks those
fields of the envpack, so its bytes, and its sha256 id, change; the file keeps the published id as its name and
`carttrace.repro.published` makes the world answer to it. `world_manifest` cuts a manifest to one world's rows and puts the
store's roster pseudonym wherever its queue id or domain was.

Layout (`worlds/`):
    envpacks/<world id>.json     the world, withheld, under the id the dataset's `world_id` column names
    manifests/<world id>.json    the world's rows, tasks, profiles and shopper maps, pseudonymous
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from carttrace.core.errors import NotFoundError, ValidationError

WITHHELD: Final = "withheld"
QUEUE_ID: Final = re.compile(r"(?<![A-Za-z0-9])MC-A?\d{3,}(?![A-Za-z0-9])")
"""A queue store's id, never inside a word: catalog text quotes model numbers such as "HMC-1200XWHT"."""
CAPTURE_ID: Final = re.compile(r"cv2-[0-9a-f]{6,}")
VIEW_STORE: Final = re.compile(r"(?<![A-Za-z0-9-])asl-[a-z0-9-]+(?:\.[a-z0-9-]+)+")
"""A recorded store an agent view names by its domain (`asl-<domain>`), as a world's `policy.agent_view` carries it."""
STORE_KEY: Final = re.compile(r"MC-A?\d{3,}(?:~\d+)?")
"""A queue store's key in a manifest: a top-up round's added stand-ins (D120, round 90) carry a `~n` suffix."""
ENVPACKS: Final = "envpacks"
MANIFESTS: Final = "manifests"


def withhold(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The envpack document with every queue store and capture it was built against blanked: its provenance names
    them, and a stand-in served in a recorded store's view names that store in its policy's `presented` and
    `agent_view` blocks."""
    text = json.dumps(doc)
    text = QUEUE_ID.sub(WITHHELD, text)
    text = CAPTURE_ID.sub(WITHHELD, text)
    text = VIEW_STORE.sub(WITHHELD, text)
    out: dict[str, Any] = json.loads(text)
    return out


def world_manifest(manifest: Mapping[str, Any], world_id: str, names: Mapping[str, str]) -> dict[str, Any]:
    """The parts of `manifest` (plain JSON) one world's rows need, with each name in `names` (a queue id, a queue
    domain) replaced by its pseudonym. Unsealed: `carttrace.collect.manifest.seal` gives it its id."""
    rows = [r for r in manifest.get("rows") or [] if r.get("envpack") == world_id]
    if not rows:
        raise NotFoundError(f"no row of manifest {manifest.get('id')} runs on {world_id}")
    stores = sorted({str(r["store_id"]) for r in rows})
    profiles = sorted({str(r["profile"]) for r in rows if r.get("profile")})
    body: dict[str, Any] = {
        "plan": _own_stores(manifest["plan"], set(stores)),
        "sources": manifest.get("sources") or {},
        "stores": {s: manifest["stores"][s] for s in stores},
        "tasks": {s: manifest["tasks"][s] for s in stores},
        "profiles": {p: manifest["profiles"][p] for p in profiles},
        "worlds": {world_id: manifest["worlds"][world_id]},
        "rows": rows,
        "notes": [],
        "shopper_maps": {k: v for k, v in (manifest.get("shopper_maps") or {}).items() if k.startswith(f"{world_id}/")},
    }
    text = json.dumps(body)
    for real in sorted(names, key=len, reverse=True):
        text = text.replace(real, names[real])
    text = CAPTURE_ID.sub(WITHHELD, text)
    text = VIEW_STORE.sub(WITHHELD, text)
    if QUEUE_ID.search(text):
        found = sorted(set(QUEUE_ID.findall(text)))[:5]
        raise ValidationError(f"world {world_id}'s manifest still names queue stores: {found}")
    out: dict[str, Any] = json.loads(text)
    return out


def _own_stores(value: Any, keep: set[str]) -> Any:
    """`value` with every other queue store dropped: map entries keyed by one, and list items naming one (a round's
    plan lists every store of its round)."""
    if isinstance(value, Mapping):
        return {
            k: _own_stores(v, keep) for k, v in value.items() if not (STORE_KEY.fullmatch(str(k)) and k not in keep)
        }
    if isinstance(value, list):
        return [
            _own_stores(v, keep) for v in value if not (isinstance(v, str) and STORE_KEY.fullmatch(v) and v not in keep)
        ]
    return value


def world_files(worlds: Path, world_id: str) -> tuple[Path, Path]:
    """The envpack and manifest files of `world_id` under a worlds folder; NotFoundError says which is missing."""
    envpack, manifest = worlds / ENVPACKS / f"{world_id}.json", worlds / MANIFESTS / f"{world_id}.json"
    for path in (envpack, manifest):
        if not path.is_file():
            raise NotFoundError(
                f"no {path.parent.name[:-1]} for world {world_id} under {worlds}",
                hint="download the world from the dataset's worlds/ folder",
            )
    return envpack, manifest

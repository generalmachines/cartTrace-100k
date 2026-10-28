"""Named protocol profiles: the version sets a journey can be run under without first visiting a store.

Why presets are data. An experiment that asks "what changes when UCP moves from 2026-04-08 to 2026-08-25" must run the
same journey under two profiles that differ only in UCP, and a sandbox run must be able to name "what live Shopify
stores spoke in October 2026" without a capture at hand. Each preset is a `ProtocolProfile` in
data/protocols/presets.json, so adding one is a reviewed data change, and its id is the hash of its versions, so the
same versions under two names share one id.

Presets also name compositions: `live-acp-2026-04-17` is `shopify-live-2026-10` with ACP 2026-04-17 beside it, so a
payment rail that adopts a public protocol version onto a curated preset (`carttrace.collect.rails`) can come back as the
preset, source "preset", instead of an override the release funnel would exclude (D74).
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.protocols.paths import configured
from carttrace.protocols.profile import ProtocolProfile

PRESETS_SCHEMA: Final = "mc.protocol-presets.v1"


class PresetFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: Literal["mc.protocol-presets.v1"] = Field(default=PRESETS_SCHEMA, alias="schema")
    presets: dict[str, ProtocolProfile]


def load_presets(path: Path | None = None) -> dict[str, ProtocolProfile]:
    """Every preset by name, sorted by name. Each profile is marked `source="preset"`."""
    path = path or configured().presets
    if not path.is_file():
        raise NotFoundError(f"no protocol presets at {path}", hint="restore data/protocols/presets.json")
    try:
        raw = PresetFile.model_validate_json(path.read_bytes())
    except PydanticValidationError as exc:
        raise ValidationError(
            f"{path.name} is malformed: {exc.errors()[0]['msg']} at {'.'.join(str(p) for p in exc.errors()[0]['loc'])}",
            hint="fix the preset; every field is a ProtocolProfile field",
        ) from exc
    for name, profile in raw.presets.items():
        if profile.source != "preset":
            raise ValidationError(f"preset {name} has source {profile.source!r}", hint="presets must say 'preset'")
    return dict(sorted(raw.presets.items()))


def profile(name: str, path: Path | None = None) -> ProtocolProfile:
    """One preset by name, e.g. profile("shopify-live-2026-10")."""
    presets = load_presets(path)
    if name not in presets:
        raise NotFoundError(f"no protocol preset {name!r}", hint=f"choose one of {sorted(presets)}")
    return presets[name]


def match_preset(versions: Mapping[str, Any], path: Path | None = None) -> ProtocolProfile | None:
    """The registered preset that speaks exactly `versions`, or None. Two presets that speak the same versions share
    one id, so the first by name is answer enough."""
    wanted = dict(versions)
    resolved = (path or configured().presets).resolve()
    for preset in _cached(str(resolved)):
        if preset.versions() == wanted:
            return preset
    return None


@functools.lru_cache(maxsize=8)
def _cached(path: str) -> tuple[ProtocolProfile, ...]:
    """The presets of one file, read once per path: rail composition asks on every row of a manifest."""
    return tuple(load_presets(Path(path)).values())

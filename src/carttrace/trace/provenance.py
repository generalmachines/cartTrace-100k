"""Where an episode's store came from: a live store, a replica built from its capture, or a stand-in twin.

Why one rule, here. A sandbox run carries a real store id (store-305 says store-305.example) while shopping a twin cut
from a public catalog. Those outcomes say nothing about the store, and a results row that lost this distinction once
reached the team sheet as the store's own. Every reader that separates store truth from stand-in numbers (the results
database, the sheet sync, the dashboard tracker) derives it from the header through this one function, so the rule
never diverges.

The rule reads the header alone, so it holds for a bundle whose manifest is gone:
- `live`: the episode spoke to the store itself (`mode` live);
- `replica`: a sandbox episode whose world was built from the store's own capture (its rights are the captured
  store's, not a catalog licence);
- `twin`: a sandbox episode in a world cut from a catalog (`rights.platform_terms` sandbox, `store_source` the catalog);
- `unknown`: a header that fits none of these, which no per-store view ever counts.
"""

from __future__ import annotations

from typing import Literal, get_args

from carttrace.trace.schema import EpisodeHeader

Provenance = Literal["live", "replica", "twin", "unknown"]
PROVENANCES: tuple[Provenance, ...] = get_args(Provenance)
STORE_PROVENANCES: tuple[Provenance, ...] = ("live", "replica")
"""What counts as the store's own result: never a twin, never unknown."""
STAND_IN: Provenance = "twin"
"""The provenance labelled "stand-in" wherever twin numbers are shown in aggregate."""

ResultMode = Literal["live", "sandbox", "reference"]
"""A manifest row's mode: the header records live or sandbox, and the reference agent marks a reference row."""

REFERENCE_AGENT = "reference-agent"
CAPTURE_SOURCE = "capture"
"""The store_source of a replica result: the store's own capture, whatever source first listed the store."""


def provenance_of(header: EpisodeHeader) -> Provenance:
    """See the module docstring."""
    if header.mode == "live":
        return "live"
    if header.rights.platform_terms == "sandbox":
        return "twin"
    return "replica" if header.envpack else "unknown"


def mode_of(header: EpisodeHeader) -> ResultMode:
    """live, or reference when the reference agent ran, else sandbox."""
    if header.mode == "live":
        return "live"
    return "reference" if agent_of(header) == REFERENCE_AGENT else "sandbox"


def agent_of(header: EpisodeHeader) -> str:
    """The agent model of an episode: served when recorded, else requested."""
    model = next((m for m in header.models if m.role == "agent"), None)
    return "" if model is None else (model.served or model.requested)


def surface_of(header: EpisodeHeader) -> str:
    """The surface kinds of an episode, each once, in the header's order: "ucp_mcp", "ucp_mcp+store_mcp"."""
    return "+".join(dict.fromkeys(s.kind for s in header.surfaces))


def store_source_of(header: EpisodeHeader) -> str:
    """Where the store's data came from: the catalog a twin was cut from, `capture` for a replica, else the header's
    store source (team_sheet, manual, ...)."""
    provenance = provenance_of(header)
    if provenance == "replica":
        return CAPTURE_SOURCE
    return header.rights.store_source

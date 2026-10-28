"""One served store: the envpack and the profile its surfaces speak, the running `Sandbox`, and the current episode.

Why one object. The MCP endpoint, the REST binding, the control routes and the 3-D Secure challenge page all act on
the same store, so they share one `Sandbox`, one event log and one idea of "the current episode". A session (an MCP
session, or a REST caller) belongs to the episode it started in; a reset starts a new episode from the pack, and the
endpoint then answers the old sessions as expired.

Which pack is served. `build_app(pack, profile=P)` serves the pack under P. When P speaks other versions than the
pack's own profile, the served pack is `pack.derive(profile=P)`: a different world by its id, because the versions are
part of the world (an experiment on one store at two UCP releases is two packs). A P with the same versions keeps the
pack as it is, so its id does not change for nothing.

Which protocols are served. Exactly the ones the profile names: UCP (over MCP and REST) when it names a UCP release,
ACP when it names an API-Version, MPP and AP2 when it names their versions. A store that names no UCP release serves
no UCP at all, the way a merchant that only speaks ACP does. MPP pays for a checkout some other protocol made, so a
profile that names MPP must name UCP, ACP or AP2 beside it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carttrace.core.clock import Clock
from carttrace.core.tokens import aliases, unalias
from carttrace.protocols.profile import ProtocolProfile
from carttrace.sandbox.core import psp
from carttrace.sandbox.core.errors import CoreError
from carttrace.sandbox.core.state import State
from carttrace.sandbox.envpack.pack import EnvPack
from carttrace.sandbox.events.runtime import Outcome, Sandbox
from carttrace.sandbox.platforms import Platform, platform_named
from carttrace.sandbox.presented import Guard
from carttrace.sandbox.surfaces.errors import UnsupportedSurface, UnsupportedVersion
from carttrace.sandbox.surfaces.mcp.versions import McpVersion, mcp_version
from carttrace.sandbox.surfaces.ucp.dialect import UcpDialect, dialect_for

TIERS = ("anonymous", "signed", "token")
PROTOCOLS = ("ucp", "acp", "mpp", "ap2")
"""The protocol families the sandbox serves, as `ProtocolProfile` names them."""
CHECKOUT_MAKERS = ("ucp", "acp", "ap2")
"""Protocols that make a checkout; MPP only pays one."""


class SandboxHost:
    """The served store. Build one per app; `build_app` puts it on `app.state.host`."""

    def __init__(
        self,
        pack: EnvPack,
        *,
        profile: ProtocolProfile | None = None,
        log_path: Path | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.source = pack
        """The pack as given."""
        self._alias_tables: dict[str, Mapping[str, str]] = {}
        """Per session, each wallet alias to its test token (`carttrace.core.tokens`)."""
        self.pack = _served(pack, profile)
        """The pack the sandbox runs: the given one, or one derived for `profile`."""
        self.profile = self.pack.profile
        self.platform: Platform = platform_named(self.pack.platform_name)
        """The store's platform dialect (carttrace.sandbox.platforms): what it adds to the protocol on the wire. The pack's
        platform pack when it names one (D42), else its store's platform."""
        self._ucp: UcpDialect | None = dialect_for(self.profile.ucp) if self.profile.ucp is not None else None
        self.mcp: McpVersion = mcp_version(self.profile.mcp or self.platform.default_mcp)
        served = self.protocols()
        if not served:
            raise UnsupportedVersion(
                f"profile {self.profile.id} names no protocol the sandbox serves",
                hint=f"name one of {list(PROTOCOLS)} in the profile (presets: shopify-live-2026-10, acp-2026-04-17)",
            )
        if "mpp" in served and not any(p in served for p in CHECKOUT_MAKERS):
            raise UnsupportedVersion(
                "MPP pays for a checkout that UCP, ACP or AP2 makes; this profile names none of them",
                hint="serve MPP beside a checkout protocol, e.g. the live-mpp-0.11.0 preset",
            )
        self.sandbox = Sandbox(self.pack, log_path=log_path, clock=clock)
        self.guard: Guard | None = None
        """The row's visibility guard (D106, carttrace.sandbox.presented): the server refuses to serve a response holding
        one of its tokens. None serves everything (a store no row has named a guard for)."""

    # -- reads ---------------------------------------------------------------------------------------------------------

    @property
    def ucp(self) -> UcpDialect:
        """The UCP release served. Raises `UnsupportedSurface` for a store that serves no UCP."""
        if self._ucp is None:
            raise UnsupportedSurface(
                f"this store serves no UCP: its profile ({self.profile.id}) serves {self.describe()}",
                hint="open a surface of a protocol the store serves",
            )
        return self._ucp

    def protocols(self) -> dict[str, str]:
        """Each protocol family the store serves, with the version it serves."""
        return {f: str(getattr(self.profile, f)) for f in PROTOCOLS if getattr(self.profile, f) is not None}

    def describe(self) -> str:
        """What the store serves, as people read it: `UCP 2026-08-25, MPP pympp-0.11.0`."""
        return ", ".join(f"{f.upper()} {v}" for f, v in self.protocols().items()) or "nothing the sandbox serves"

    def serves(self, family: str) -> bool:
        return getattr(self.profile, family, None) is not None if family in PROTOCOLS else False

    def require(self, family: str) -> str:
        """The version of `family` the store serves, or `UnsupportedSurface` saying plainly what it serves."""
        if not self.serves(family):
            raise UnsupportedSurface(
                f"this store does not serve {family.upper()}: its profile ({self.profile.id}) serves {self.describe()}",
                hint=f"serve the pack under a profile that names {family} (presets list: carttrace protocols presets)",
            )
        return str(getattr(self.profile, family))

    @property
    def episode(self) -> int:
        return self.sandbox.episode

    @property
    def state(self) -> State:
        return self.sandbox.state

    def tier(self, session: str) -> str | None:
        """The trust tier a session acts at now, or None before it first acted."""
        s = self.state.sessions.get(session)
        return None if s is None else s.tier

    # -- changes -------------------------------------------------------------------------------------------------------

    def execute(self, command: Mapping[str, Any]) -> Outcome:
        """One command, with any wallet alias the agent was shown read back as its test token (`carttrace.core.tokens`): the
        token's designed fault stays here, inside the PSP, and the event log holds the token as before."""
        cmd = dict(command)
        session = cmd.get("session")
        if isinstance(session, str):
            cmd = unalias(cmd, self._aliases(session))
        return self.sandbox.execute(cmd)

    def _aliases(self, session: str) -> Mapping[str, str]:
        table = self._alias_tables.get(session)
        if table is None:
            table = self._alias_tables[session] = aliases(session, psp.TOKENS)
        return table

    def reject(self, command_type: str, error: CoreError) -> Outcome:
        """Log a call refused before it became a command, so every tool call is one entry in the log."""
        return self.sandbox.reject(command_type, error)

    def reset(self, *, seed: int | None = None, faults: Sequence[Mapping[str, Any]] | None = None) -> None:
        """A new episode from the pack, with a new fault schedule when `faults` is given (a derived pack)."""
        pack = self.pack.derive(faults=[dict(f) for f in faults]) if faults is not None else None
        if pack is not None:
            self.pack = pack
        self.sandbox.reset(pack, seed=seed)

    def open_session(self, session_id: str, tier: str, permissions: Iterable[str] = ()) -> Outcome:
        """Open (or reopen) a session at a trust tier, as the environment: logged, so a replay opens it too."""
        return self.execute(
            {
                "type": "open_session",
                "actor": "system",
                "session": session_id,
                "session_id": session_id,
                "tier": tier,
                "permissions": sorted(set(permissions)),
            }
        )


def _served(pack: EnvPack, profile: ProtocolProfile | None) -> EnvPack:
    if profile is None or profile.versions() == pack.profile.versions():
        return pack
    return pack.derive(profile=profile)

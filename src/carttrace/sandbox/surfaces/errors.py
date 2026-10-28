"""The surfaces' own refusals: a profile the sandbox cannot speak, or a surface the store does not serve.

Why typed. Building an app for a profile whose UCP release, ACP API-Version, MPP or AP2 version has no dialect, or whose
MCP version has no streamable HTTP transport, must fail when the app is built, with the versions that would work,
never as a wrong document mid-episode. Opening a surface of a protocol the store's profile does not name (an ACP surface
on a UCP-only store, a WebMCP surface on a store that serves no UCP), or of a kind the sandbox has not built, is
refused plainly the same way.
"""

from __future__ import annotations

from carttrace.core.errors import ProtocolError


class UnsupportedVersion(ProtocolError):
    """A profile names a protocol version the sandbox has no dialect or transport for."""


class UnsupportedSurface(ProtocolError):
    """A surface kind the sandbox has not built, or a protocol this store's profile does not serve."""

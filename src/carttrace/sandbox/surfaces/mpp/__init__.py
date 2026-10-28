"""MPP in the sandbox: a checkout's payment behind HTTP 402, a credential for the challenge, and a receipt.

Map:
- `method`: the `sandbox` payment method, built for the `mpp` SDK (a charge paid with an opaque test credential);
- `endpoint`: the store's 402 endpoint, challenges as core payment requests, verification on the logical clock;
- `client`: the agent's `ToolSurface` (kind `mpp`).
"""

from carttrace.sandbox.surfaces.mpp.client import MppSurface
from carttrace.sandbox.surfaces.mpp.endpoint import MPP_PATH, MppServer, served_versions
from carttrace.sandbox.surfaces.mpp.method import METHOD, SandboxMethod, credential

__all__ = ["METHOD", "MPP_PATH", "MppServer", "MppSurface", "SandboxMethod", "credential", "served_versions"]

"""Control routes: the harness's view and levers on the running store. Localhost only, never part of UCP.

- `POST /_sandbox/reset` `{"seed"?: int, "faults"?: [...]}`: a new episode from the pack, exactly as the pack holds
  it (a fault schedule given here derives a new pack); every MCP session of the old episode expires.
- `GET /_sandbox/state`: the core's full state, its hash, the event digest and the pack and profile ids.
- `GET /_sandbox/events?since=N&sealed=1`: the event log from record N; sealed records (fault firings, the scripted
  3-D Secure outcome) only with `sealed=1`.
- `GET /_sandbox/diff`: what changed in the state since the episode began, as `{path: {before, after}}`.
- `POST /_sandbox/session` `{"session_id", "tier", "permissions"?}`: open a session at a trust tier, as the
  environment, so a test can give a session the token tier without building another pack.

Every refusal is JSON with an `error` and a `hint`; a client that is not on the loopback interface gets 403.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from carttrace.core.canon import dumps_bytes, loads
from carttrace.core.errors import MCError
from carttrace.sandbox.surfaces.access import AccessRefused, require_local, valid_session_id
from carttrace.sandbox.surfaces.host import TIERS, SandboxHost

JSON = "application/json"
Handler = Callable[[Request], Awaitable[Response]]


def control_routes(host: SandboxHost) -> list[Route]:
    c = Control(host)
    return [
        Route("/_sandbox/reset", _guarded(c.reset), methods=["POST"]),
        Route("/_sandbox/state", _guarded(c.state), methods=["GET"]),
        Route("/_sandbox/events", _guarded(c.events), methods=["GET"]),
        Route("/_sandbox/diff", _guarded(c.diff), methods=["GET"]),
        Route("/_sandbox/session", _guarded(c.session), methods=["POST"]),
    ]


class BadControl(MCError):
    """A control request with a malformed body or query."""


def _guarded(handler: Handler) -> Handler:
    async def run(request: Request) -> Response:
        try:
            require_local(request)
            return await handler(request)
        except AccessRefused as e:
            return _json({"error": e.message, "hint": e.hint}, e.status)
        except MCError as e:
            return _json({"error": e.message, "hint": e.hint}, 400)

    return run


class Control:
    def __init__(self, host: SandboxHost) -> None:
        self.host = host

    async def reset(self, request: Request) -> Response:
        body = await _body(request)
        seed = body.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise BadControl("seed must be an integer", hint='send {"seed": 7} or no seed')
        faults = body.get("faults")
        if faults is not None and not isinstance(faults, list):
            raise BadControl("faults must be a list of fault knobs", hint="see carttrace.sandbox.faults.KNOBS")
        self.host.reset(seed=seed, faults=faults)
        sb = self.host.sandbox
        return _json(
            {
                "ok": True,
                "env": self.host.pack.id,
                "profile": self.host.profile.id,
                "episode": self.host.episode,
                "seed": sb.state.seed,
                "state_hash": sb.state_hash(),
                "digest": sb.digest,
            }
        )

    async def state(self, request: Request) -> Response:
        sb = self.host.sandbox
        state = sb.state
        return _json(
            {
                "env": self.host.pack.id,
                "profile": self.host.profile.id,
                "episode": self.host.episode,
                "seq": state.seq,
                "clock": state.clock,
                "state_hash": state.hash(),
                "digest": sb.digest,
                "events": sb.log_length,
                "state": state.to_dict(),
            }
        )

    async def events(self, request: Request) -> Response:
        raw = request.query_params.get("since", "0")
        if not raw.isdigit():
            raise BadControl("since must be a whole number", hint="pass the count of records already read")
        sealed = request.query_params.get("sealed", "0") in ("1", "true")
        sb = self.host.sandbox
        records = sb.events_since(int(raw), include_sealed=sealed)
        return _json(
            {
                "env": self.host.pack.id,
                "episode": self.host.episode,
                "digest": sb.digest,
                "since": int(raw),
                "events": records,
            }
        )

    async def diff(self, request: Request) -> Response:
        delta = self.host.sandbox.diff()
        return _json(
            {
                "env": self.host.pack.id,
                "episode": self.host.episode,
                "diff": {p: {"before": b, "after": a} for p, (b, a) in delta.items()},
            }
        )

    async def session(self, request: Request) -> Response:
        body = await _body(request)
        sid, tier = body.get("session_id"), body.get("tier")
        permissions = body.get("permissions") or []
        if not valid_session_id(sid):
            raise BadControl("session_id must be 1-64 letters, digits, '_', '.', ':' or '-'", hint="e.g. buyer")
        if tier not in TIERS:
            raise BadControl(f"tier must be one of {list(TIERS)}", hint='e.g. {"tier": "token"}')
        if not isinstance(permissions, list) or not all(isinstance(p, str) for p in permissions):
            raise BadControl("permissions must be a list of names", hint='e.g. ["purchase"]')
        out = self.host.open_session(str(sid), str(tier), permissions)
        return _json(
            {"ok": out.ok, "session": sid, "tier": tier, "permissions": sorted(set(permissions)), "error": out.error}
        )


async def _body(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        body = loads(raw)
    except ValueError:
        raise BadControl("the body is not JSON", hint="send a JSON object") from None
    if not isinstance(body, dict):
        raise BadControl("the body must be a JSON object", hint="send a JSON object")
    return body


def _json(value: Any, status: int = 200) -> Response:
    return Response(dumps_bytes(value), status_code=status, media_type=JSON)

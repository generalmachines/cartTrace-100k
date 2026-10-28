"""Layer 4: the protocol sandbox. State machine core, events, envpacks, faults, payment rails, versioned surfaces,
browser runtime, ASGI server.

This module is the public API of the sandbox core. Protocol surfaces (UCP over MCP and REST, ACP, MPP, AP2, WebMCP)
are built on top of it and never hold state of their own: they translate a wire call into a command, call
`Sandbox.execute`, and render the `Outcome` and the state back in their protocol's dialect.

Build a world
    `build_envpack(store=..., variants=[...], currency=..., profile=..., payment_handlers=..., market="US",
    tasks=[...], faults=[...], start={...}, policy={...})` seals an `EnvPack` named `env-<sha12>`.
    `build_envpack_from_cart(cart, store=..., profile=..., payment_handlers=...)` builds one from a live journey's
    verified cart (`carttrace.trace.schema.Cart`). `EnvPack.save(dir)` and `EnvPack.load(path)` round-trip byte for byte;
    `pack.public()` hides the fault schedule and hidden task specs. `pack.derive(faults=..., start=...)` makes a
    variant pack for an experiment. A knob with `"at_start": true` (stockout, reprice, injected_text) is true of the
    world from the episode's opening, logged as sealed events right after `episode.started`;
    `faults.opening_knobs(knobs)` turns a task's untriggered knobs into such faults.

Run it
    `sb = Sandbox(pack, log_path=..., clock=...)`, then `sb.execute({"type": "create_cart", "session": "s1",
    "lines": [{"variant_id": "v1", "quantity": 1}]})` or a typed command (`commands.CreateCart(...)`). Every
    command returns an `Outcome`: `ok`, `refs` (cart_id, checkout_id, payment_id, order_id, challenge_id,
    action_required), `error` (a typed error as JSON: kind, code, message, field, retryable, details, hint), the
    log `events`, and `messages()` for the protocol. `sb.handoff(cart, session=...)` re-creates a live cart.
    `sb.reject(type, error)` logs a wire call an adapter refused before translating it. `sb.reset()` starts a new
    episode from the pack, exactly.

Commands (`commands.TYPES`)
    Agent: read, create_cart, update_cart, cancel_cart, create_checkout, update_checkout, cancel_checkout,
    complete_checkout (`payment: {handler, instrument?, token?, request_id?}`), confirm_payment_action (`approved`),
    vault_credential (an ACP delegated token's allowance), mint_shop_token (a Shop Pay single-use token, D73).
    Merchant: fulfill_order, cancel_order, refund_order,
    edit_order, request_return, receive_return, request_payment (an MPP challenge or AP2 cart's payment request),
    record_mandate (an AP2 mandate by its digest). System: open_session (tier and profile id), revoke_token,
    set_stock, set_price, handoff_cart. Trust tiers: anonymous, signed and token; only a token session with the
    purchase permission may complete or confirm.

Payments
    Credentials are opaque test tokens (`TOKENS`), never card numbers: tok_visa_ok, tok_mc_ok, tok_amex_ok,
    tok_gpay_ok, tok_shopify_card_ok, tok_shop_pay_ok approve; tok_visa_decline_insufficient_funds,
    tok_visa_decline_do_not_honor and tok_shop_pay_declined
    decline; tok_visa_3ds_challenge and tok_visa_3ds_fail ask for a 3-D Secure challenge that passes or fails. A
    challenge parks the payment in requires_action and the checkout in requires_escalation (escalation
    payment_action_required, refs challenge_id); the surface serves the challenge and sends
    `confirm_payment_action`. Payment statuses: pending, requires_action, authenticated, authorized, captured,
    partially_refunded, refunded, declined, authentication_failed, voided (`PAYMENT`). The handlers a pack serves are
    the store's declared ones, by namespace and by entry id (`EnvPack.handlers()`), plus the fake handler when the
    policy says so, plus a payment protocol's handler when the profile names one: `DELEGATED_NAMESPACE` for ACP
    (pays only with a vaulted `vt-` credential, within its allowance: amount, currency, checkout, expiry; single use)
    and `REQUESTED_NAMESPACE` for MPP and AP2 (pays only an open payment request: exact amount, unexpired, single
    use), plus the Shop Pay handler when the store declares it or the profile adopts it (D73: pays only with a
    `shop_...` token minted for the checkout, unexpired, single use). State maps `credentials`, `payment_requests`
    and `mandates` hold them; mandates are append-only.

Replay and grading
    `replay(pack, sb.commands)` and `replay_events(pack, records)` reproduce an episode to the same state hash and
    digest; `audit_episode(pack, read_events(path, episode))` checks a log end to end. `sb.diff()` and
    `check(delta, expected, allowed)` are the grader's state-diff check; `diff_path("inventory", variant_id,
    "on_hand")` builds an exact path even when an id holds slashes, as live store gids do.

Serve it (`carttrace.sandbox.server`, `carttrace.sandbox.surfaces`)
    `build_app(pack, profile=None)` is the store as an ASGI app serving what its profile names: UCP over MCP at
    /api/ucp/mcp (streamable HTTP at the profile's MCP version), UCP REST under /api/ucp, /.well-known/ucp and the
    WebMCP storefront under /shop; ACP under /api/acp and its delegated payment PSP under /psp; MPP under /api/mpp;
    AP2's merchant under /api/ap2; and localhost-only /_sandbox/{reset, state, events, diff, session}. A protocol
    the profile does not name answers a plain 404 under its prefix. `await open_sandbox_surface(pack, kind="ucp_mcp"
    | "ucp_rest" | "acp_rest" | "mpp" | "ap2" | "webmcp", recorder=writer, session_id="buyer", options={...})` is a
    `ToolSurface` on it, in process, with no socket; a kind whose protocol the store does not serve is refused,
    saying what it serves. "webmcp" lists the tools the storefront page registers and runs them in the sandbox
    browser (`carttrace.sandbox.browser`). `host_of(app)` is the served store (its `Sandbox`, pack and profile). These names
    load on first use, so importing the core stays light.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from carttrace.sandbox.core import (
    DECLINE_CODES,
    DELEGATED_NAMESPACE,
    FAKE_HANDLERS,
    FAKE_NAMESPACE,
    MACHINES,
    MARKETS,
    PAYMENT,
    POLICY,
    REQUESTED_NAMESPACE,
    TIERS,
    TOKENS,
    TYPES,
    BusinessRule,
    Command,
    Config,
    Conflict,
    CoreError,
    Event,
    FaultRule,
    IllegalTransition,
    InvalidCommand,
    InvariantViolation,
    Market,
    NotFound,
    PolicyDenied,
    State,
    TokenSpec,
    Unavailable,
    UnknownMarket,
    apply,
    check_invariants,
    initial_state,
    market,
    market_for,
    outcome,
    parse,
)
from carttrace.sandbox.core import commands as commands
from carttrace.sandbox.core import reject as reject_state
from carttrace.sandbox.envpack import (
    DEFAULT_PROFILE,
    EnvPack,
    EnvPackError,
    build_envpack,
    build_envpack_from_cart,
    handoff_command,
    seed_state_from_cart,
)
from carttrace.sandbox.envpack import SCHEMA as ENVPACK_SCHEMA
from carttrace.sandbox.events import (
    ANY,
    BOOKKEEPING,
    EVENTS_SCHEMA,
    MESSAGE_EVENTS,
    DiffCheck,
    EventLog,
    EventRecord,
    Outcome,
    Sandbox,
    audit_episode,
    check,
    commands_from_events,
    diff,
    read_events,
    replay,
    replay_events,
    verify,
)
from carttrace.sandbox.events import path as diff_path
from carttrace.sandbox.faults import KNOBS, FaultConfigError, compile_faults

if TYPE_CHECKING:
    from carttrace.sandbox.server import build_app, host_of, open_sandbox_surface, serve
    from carttrace.sandbox.surfaces import (
        McpSurface,
        RestSurface,
        SandboxHost,
        SandboxSurface,
        UnsupportedSurface,
        UnsupportedVersion,
        WireSink,
    )
    from carttrace.sandbox.surfaces.remote import RemoteRestSurface, open_remote_rest_surface

_LAZY = {
    "build_app": "carttrace.sandbox.server",
    "host_of": "carttrace.sandbox.server",
    "open_sandbox_surface": "carttrace.sandbox.server",
    "serve": "carttrace.sandbox.server",
    "McpSurface": "carttrace.sandbox.surfaces",
    "RestSurface": "carttrace.sandbox.surfaces",
    "SandboxHost": "carttrace.sandbox.surfaces",
    "SandboxSurface": "carttrace.sandbox.surfaces",
    "UnsupportedSurface": "carttrace.sandbox.surfaces",
    "UnsupportedVersion": "carttrace.sandbox.surfaces",
    "WireSink": "carttrace.sandbox.surfaces",
    "RemoteRestSurface": "carttrace.sandbox.surfaces.remote",
    "open_remote_rest_surface": "carttrace.sandbox.surfaces.remote",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'carttrace.sandbox' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


PLUGINS: dict[str, dict[str, str]] = {
    "sandbox.surface": {
        "ucp_mcp": "carttrace.sandbox.surfaces.client:McpSurface",
        "ucp_rest": "carttrace.sandbox.surfaces.client:RestSurface",
        "acp_rest": "carttrace.sandbox.surfaces.acp.client:AcpSurface",
        "mpp": "carttrace.sandbox.surfaces.mpp.client:MppSurface",
        "ap2": "carttrace.sandbox.surfaces.ap2.client:Ap2Surface",
        "webmcp": "carttrace.sandbox.surfaces.webmcp.client:WebMcpSurface",
    },
    "sandbox.catalog": {
        "abo": "carttrace.sandbox.catalogs.abo:AboSource",
        "lazada": "carttrace.sandbox.catalogs.lazada:LazadaSource",
        "obf": "carttrace.sandbox.catalogs.openfacts:OpenFactsSource",
        "off": "carttrace.sandbox.catalogs.openfacts:OpenFactsSource",
        "replica": "carttrace.sandbox.catalogs.replica:ReplicaSource",
        "wands": "carttrace.sandbox.catalogs.wands:WandsSource",
    },
}
"""Replaceable sandbox parts (`carttrace.core.registry`), as import strings, so reading the table loads nothing.
- `sandbox.surface`: the agent's side of each protocol surface, by surface kind (UCP over MCP and REST, ACP REST, MPP,
  AP2, and WebMCP, the storefront's tools run in the sandbox browser, `carttrace.sandbox.browser`);
  `open_sandbox_surface(kind=...)` resolves it, so a new surface is one line here.
- `sandbox.catalog`: the licensed catalogs sandbox stores are cut from; adding a source is one line here plus its
  files in data/catalogs. Lazada is restricted (D11). `replica` is the one source without a manifest: real stores'
  replicas, built from captures of the stores themselves (D12, carttrace.sandbox.catalogs.replica).
The core has one implementation of each of its parts, so none of them is listed; a part with a second implementation
(a browser runtime, a challenge renderer) adds its kind here."""

__all__ = [
    "ANY",
    "BOOKKEEPING",
    "DECLINE_CODES",
    "DEFAULT_PROFILE",
    "DELEGATED_NAMESPACE",
    "ENVPACK_SCHEMA",
    "EVENTS_SCHEMA",
    "FAKE_HANDLERS",
    "FAKE_NAMESPACE",
    "KNOBS",
    "MACHINES",
    "MARKETS",
    "MESSAGE_EVENTS",
    "PAYMENT",
    "PLUGINS",
    "POLICY",
    "REQUESTED_NAMESPACE",
    "TIERS",
    "TOKENS",
    "TYPES",
    "BusinessRule",
    "Command",
    "Config",
    "Conflict",
    "CoreError",
    "DiffCheck",
    "EnvPack",
    "EnvPackError",
    "Event",
    "EventLog",
    "EventRecord",
    "FaultConfigError",
    "FaultRule",
    "IllegalTransition",
    "InvalidCommand",
    "InvariantViolation",
    "Market",
    "McpSurface",
    "NotFound",
    "Outcome",
    "PolicyDenied",
    "RemoteRestSurface",
    "RestSurface",
    "Sandbox",
    "SandboxHost",
    "SandboxSurface",
    "State",
    "TokenSpec",
    "Unavailable",
    "UnknownMarket",
    "UnsupportedSurface",
    "UnsupportedVersion",
    "WireSink",
    "apply",
    "audit_episode",
    "build_app",
    "build_envpack",
    "build_envpack_from_cart",
    "check",
    "check_invariants",
    "commands",
    "commands_from_events",
    "compile_faults",
    "diff",
    "diff_path",
    "handoff_command",
    "host_of",
    "initial_state",
    "market",
    "market_for",
    "open_remote_rest_surface",
    "open_sandbox_surface",
    "outcome",
    "parse",
    "read_events",
    "reject_state",
    "replay",
    "replay_events",
    "seed_state_from_cart",
    "serve",
    "verify",
]

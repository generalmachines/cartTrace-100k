"""UCP's `meta` argument belongs to the client surface, never to the agent.

Why. Every UCP tool takes `meta`. Its `ucp-agent.profile` names the platform's profile (Shopify requires it), and from
UCP 2026-08-25 its `idempotency-key` is required on the tools that complete or cancel (complete_checkout,
cancel_checkout, cancel_cart). Both are facts about who we are and how we retry, not choices an agent may make. So
the live and the sandbox surfaces both hide `meta` from the tool schemas the agent reads, and fill it on every call,
replacing whatever the agent sent. Before this module the live client, the sandbox client, the agent loop, the
collector and the store scan each carried their own copy, with three different placeholder URLs.
"""

from __future__ import annotations

import copy
import hashlib
import uuid
from collections.abc import Mapping
from typing import Any

from carttrace.core.cards import looks_like_card_number

PLACEHOLDER_AGENT_PROFILE = "https://carttrace.invalid/agents/ucp-profile.json"
"""Sent where no real profile URL is configured: sandbox runs and offline scans. `.invalid` never resolves (RFC 2606),
so a store that tried to fetch it would fail rather than read someone else's document."""

META = "meta"
IDEMPOTENCY_KEY = "idempotency-key"
KEYED_TOOLS = frozenset({"complete_checkout", "cancel_checkout", "cancel_cart"})
"""The tools whose `meta` requires an idempotency key at UCP 2026-08-25. A surface uses this until the store's own
tools/list says otherwise (`requires_idempotency_key`)."""


def schema_without_meta(input_schema: Mapping[str, Any]) -> dict[str, Any]:
    """A tool's input schema as the agent sees it: the client-owned `meta` property and its `required` entry gone."""
    schema: dict[str, Any] = copy.deepcopy(dict(input_schema))
    schema.setdefault("type", "object")
    properties = schema.get("properties")
    if isinstance(properties, dict):
        properties.pop(META, None)
    else:
        schema["properties"] = {}
    required = schema.get("required")
    if isinstance(required, list):
        kept = [name for name in required if name != META]
        if kept:
            schema["required"] = kept
        else:
            schema.pop("required")
    return schema


def requires_idempotency_key(input_schema: Mapping[str, Any]) -> bool:
    """True when the tool's published `meta` schema requires an idempotency key, read through allOf, anyOf and oneOf
    as UCP 2026-08-25 writes it (`allOf: [{$ref: meta}, {required: [ucp-agent, idempotency-key]}]`)."""
    properties = input_schema.get("properties")
    meta = properties.get(META) if isinstance(properties, Mapping) else None
    return _requires(meta, IDEMPOTENCY_KEY)


def _requires(schema: Any, name: str) -> bool:
    if not isinstance(schema, Mapping):
        return False
    required = schema.get("required")
    if isinstance(required, list) and name in required:
        return True
    return any(_requires(part, name) for key in ("allOf", "anyOf", "oneOf") for part in schema.get(key) or ())


def idempotency_key(*parts: object) -> str:
    """A UUID (UCP types the key `format: uuid`) derived from `parts`, so the same call in the same episode gets the
    same key on every replay, and different calls never share one.

    A UUID's hex can hold a dash-joined run of 13 or more digits that passes the Luhn check (one key in a few
    hundred). The sandbox core refuses a command whose text holds one and traces redact it, so such a key is skipped
    for the next derivation: a call is never refused, or its key echoed redacted, for looking like a card."""
    key = _uuid_of(parts)
    skips = 0
    while looks_like_card_number(key):
        skips += 1
        key = _uuid_of((*parts, skips))
    return key


def _uuid_of(parts: tuple[object, ...]) -> str:
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return str(uuid.UUID(bytes=digest[:16], version=5))


def with_meta(
    arguments: Mapping[str, Any], *, profile_url: str | None, idempotency: str | None = None
) -> dict[str, Any]:
    """The agent's arguments with its own `meta` dropped and ours put in: the profile URL, when there is one, and the
    idempotency key, when the tool needs one."""
    out = {key: value for key, value in arguments.items() if key != META}
    meta: dict[str, Any] = {}
    if profile_url:
        meta["ucp-agent"] = {"profile": profile_url}
    if idempotency:
        meta[IDEMPOTENCY_KEY] = idempotency
    if meta:
        out[META] = meta
    return out

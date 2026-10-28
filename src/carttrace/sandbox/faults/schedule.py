"""Fault knobs, as an environment pack stores them, compiled into the core's fault rules.

Why a schedule and not a switch. The lab's old server had five ad-hoc faults wired into the store's methods, each with
its own trigger logic, and a reset that forgot to undo some of them (the stock and price leak). Here a fault is data
in the environment pack: what happens, when (on a command type, from command N, after N commands, when a variant is
touched), and how many times. The pack's id covers the schedule, so two episodes with the same pack fail the same way,
and the pack's public view hides it from anything the agent can read.

Every knob is checked when the pack is built or loaded, types included, so a bad schedule is refused before an episode
starts rather than failing with a traceback in the middle of one.

Knobs (optional triggers on every knob: `on`, `at_seq`, `after`, `variant`, `times`, and an `id`):

    stockout          {"variant", "hard": false}   free stock goes to zero; hard also breaks held reservations
    reprice           {"variant", "price"}         the price changes (between search and cart, or before completion)
    conflict          {"message"}                  an HTTP 409: the command is refused, retryable
    server_error      {"status": 503, "message"}   a 5xx: refused `times` times, then the store answers normally
    rate_limit        {"after", "retry_after"}     every command after `after` is refused, retryable
    payment_decline   {"code": "generic_decline"}  the next completion's payment is declined with this reason
    three_ds          {"result": "pass"}           the next completion needs a 3-D Secure challenge that passes or fails
    token_revoked     {"session"}                  the buyer's token is revoked; completion is then denied
    duplicate_webhook {}                           the next order webhook is delivered twice
    injected_text     {"product_id", "text"}       text (a prompt injection) appears in a product's description
    refuse_codes      {}                           every discount code is refused, including one the store showed
    unshippable       {"countries": [...]}         destinations in these countries are refused

`payment_decline` and `three_ds` fire on `complete_checkout` unless `on` says otherwise; a `payment_decline` with
`"on": "confirm_payment_action"` declines after a challenge has passed, at authorization.

`"at_start": true` (stockout, reprice and injected_text only, and with no other trigger) makes the change a fact of the
world the episode opens on rather than an event in the middle of it: it fires once, as the episode starts, so the
first catalog read already shows it. A substitute task's sealed stockout is one (round 1 of the sandbox judged 71
WRONG verdicts on a named variant the replica still sold, because a stockout that waits for the variant to be touched
leaves every search and product page showing it in stock).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from carttrace.core.errors import ValidationError
from carttrace.sandbox.core.commands import TYPES
from carttrace.sandbox.core.config import FaultRule
from carttrace.sandbox.core.psp import CHALLENGE_OUTCOMES, DECLINE_CODES

ADD_COMMANDS = ("create_cart", "update_cart", "create_checkout", "update_checkout")


class FaultConfigError(ValidationError):
    """A fault knob is unknown, misses a field, or has a field of the wrong type."""


def _text(v: object) -> bool:
    return isinstance(v, str) and 0 < len(v) <= 2000


def _flag(v: object) -> bool:
    return isinstance(v, bool)


def _amount(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _status(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 500 <= v <= 599


def _decline(v: object) -> bool:
    return v in DECLINE_CODES


def _challenge(v: object) -> bool:
    return v in CHALLENGE_OUTCOMES


def _countries(v: object) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(c, str) and len(c) == 2 and c.isalpha() for c in v)


@dataclass(frozen=True, slots=True)
class Knob:
    kind: str
    """The core's name for the fault (`carttrace.sandbox.core.fault_effects`)."""
    params: Mapping[str, Callable[[object], bool]] = field(default_factory=dict)
    """Parameter name to the check its value must pass."""
    required: tuple[str, ...] = ()
    default_on: tuple[str, ...] = ()
    standing: bool = False
    """Holds for the whole episode once due, unless `times` says otherwise."""
    expects: str = ""
    """What a valid value looks like, for the error hint."""


KNOBS: Mapping[str, Knob] = {
    "stockout": Knob("stockout", {"hard": _flag}, ("variant",), ADD_COMMANDS),
    "reprice": Knob("reprice", {"price": _amount}, ("variant", "price"), ADD_COMMANDS, expects="price in minor units"),
    "conflict": Knob("conflict", {"message": _text}),
    "server_error": Knob("unavailable", {"status": _status, "message": _text}, expects="status between 500 and 599"),
    "rate_limit": Knob("rate_limited", {"retry_after": _amount}, ("after",), standing=True),
    "payment_decline": Knob(
        "payment_decline", {"code": _decline}, default_on=("complete_checkout",), expects=f"code in {DECLINE_CODES}"
    ),
    "three_ds": Knob(
        "three_ds", {"result": _challenge}, default_on=("complete_checkout",), expects=f"result in {CHALLENGE_OUTCOMES}"
    ),
    "token_revoked": Knob("revoke_token", {"session": _text}),
    "duplicate_webhook": Knob("duplicate_webhook"),
    "injected_text": Knob("injected_text", {"product_id": _text, "text": _text}, ("product_id", "text")),
    "refuse_codes": Knob("refuse_codes", standing=True),
    "unshippable": Knob(
        "unshippable", {"countries": _countries}, ("countries",), standing=True, expects="countries as 2-letter codes"
    ),
}
"""Knob name to how the core runs it."""

TRIGGERS = ("on", "at_seq", "after", "variant", "times")
START_KINDS = ("stockout", "reprice", "injected_text")
"""Knobs that may hold `at_start`: changes to the world (stock, a price, a product's text) that need no command."""


def _err(i: int, kind: str, message: str, hint: str) -> FaultConfigError:
    return FaultConfigError(f"fault {i} ({kind}): {message}", hint=hint)


def _on(i: int, kind: str, raw: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return default
    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n in TYPES for n in names):
        raise _err(i, kind, "`on` must name command types", f"use any of {', '.join(TYPES)}")
    return tuple(str(n) for n in names)


def _check_triggers(i: int, kind: str, k: Mapping[str, Any]) -> None:
    for name in ("at_seq", "after"):
        v = k.get(name)
        if v is not None and not _amount(v):
            raise _err(i, kind, f"`{name}` must be a whole number", "count commands from 1")
    times = k.get("times")
    if times is not None and not (_amount(times) and times >= 1):
        raise _err(i, kind, "`times` must be at least 1", "leave it out to fire every time it matches")
    variant = k.get("variant")
    if variant is not None and not _text(variant):
        raise _err(i, kind, "`variant` must be a variant id", "name a variant in the pack's snapshot")


def _at_start(i: int, kind: str, k: Mapping[str, Any]) -> bool:
    """Whether the knob holds from the episode's start. Refused on a kind that needs a command, and beside another
    trigger: a fact of the opening world has no other moment. A stockout's or reprice's `variant` is its target."""
    raw = k.get("at_start", False)
    if not _flag(raw):
        raise _err(i, kind, "`at_start` must be true or false", "leave it out for a fault a command triggers")
    if not raw:
        return False
    if kind not in START_KINDS:
        raise _err(i, kind, "`at_start` is for changes to the world", f"use it on one of {', '.join(START_KINDS)}")
    named = [t for t in TRIGGERS if k.get(t) is not None and not (t == "variant" and kind in ("stockout", "reprice"))]
    if named:
        raise _err(i, kind, f"`at_start` takes no other trigger, got {named}", "drop the triggers or `at_start`")
    return True


def _compile_one(i: int, k: Mapping[str, Any]) -> FaultRule:
    kind = str(k.get("kind"))
    knob = KNOBS.get(kind)
    if knob is None:
        raise FaultConfigError(f"fault {i}: unknown kind {kind!r}", hint=f"use one of {', '.join(sorted(KNOBS))}")
    extra = set(k) - {"kind", "id", "at_start", *TRIGGERS, *knob.params}
    if extra:
        raise _err(i, kind, f"unknown field(s) {sorted(extra)}", f"{kind} takes {sorted(knob.params)} and triggers")
    missing = [f for f in knob.required if k.get(f) is None]
    if missing:
        raise _err(i, kind, f"missing {missing}", knob.expects or f"{kind} needs {list(knob.required)}")
    for name, ok in knob.params.items():
        if name in k and not ok(k[name]):
            raise _err(i, kind, f"`{name}` has the wrong type or value", knob.expects or f"check `{name}`")
    _check_triggers(i, kind, k)
    at_start = _at_start(i, kind, k)
    params: dict[str, Any] = {p: k[p] for p in knob.params if p in k}
    if kind in ("stockout", "reprice"):
        params["variant"] = str(k["variant"])
    if kind == "unshippable":
        params["countries"] = sorted(str(c).upper() for c in k["countries"])
    fid = k.get("id")
    if fid is not None and not _text(fid):
        raise _err(i, kind, "`id` must be a non-empty string", "name the fault or leave the id out")
    if at_start:
        return FaultRule(id=str(fid or f"f{i}-{kind}"), kind=knob.kind, times=1, params=params, at_start=True)
    return FaultRule(
        id=str(fid or f"f{i}-{kind}"),
        kind=knob.kind,
        on=_on(i, kind, k.get("on"), knob.default_on),
        at_seq=k.get("at_seq"),
        after=k.get("after"),
        variant=None if k.get("variant") is None else str(k["variant"]),
        times=k.get("times", None if knob.standing else 1),
        params=params,
    )


def opening_knobs(knobs: Iterable[Mapping[str, Any]], *, prefix: str = "") -> list[dict[str, Any]]:
    """Knobs for a world that opens with them already true: each change to the world (`START_KINDS`) that names no
    trigger of its own holds from the episode's start (`at_start`), as a task's sealed stockout must; any other knob
    is kept as written. Every knob gets an id, `prefix` before its own or the schedule's default one, so the knobs can
    join another schedule without two ids meeting. Checked as `compile_faults` checks them."""
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(knobs, start=1):
        if not isinstance(raw, Mapping):
            raise FaultConfigError(f"fault {i} is not an object", hint='write each fault as {"kind": ...}')
        k = dict(raw)
        kind = str(k.get("kind"))
        k["id"] = f"{prefix}{k.get('id') or f'f{i}-{kind}'}"
        targeted = ("variant",) if kind in ("stockout", "reprice") else ()
        if kind in START_KINDS and "at_start" not in k and all(k.get(t) is None for t in TRIGGERS if t not in targeted):
            k["at_start"] = True
        out.append(k)
    compile_faults(out)
    return out


def compile_faults(knobs: Iterable[Mapping[str, Any]]) -> tuple[FaultRule, ...]:
    """Knobs to the core's fault rules, in schedule order. Refuses an unknown kind, an unknown or mistyped field, and
    two knobs with one id."""
    rules: list[FaultRule] = []
    for i, k in enumerate(knobs, start=1):
        if not isinstance(k, Mapping):
            raise FaultConfigError(f"fault {i} is not an object", hint='write each fault as {"kind": ...}')
        rules.append(_compile_one(i, k))
    ids = [r.id for r in rules]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        raise FaultConfigError(f"fault ids repeat: {dupes}", hint="give every fault its own id")
    return tuple(rules)

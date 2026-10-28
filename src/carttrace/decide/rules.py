"""Rules answer a decision outright whenever plain code can, so the decision model is asked only what needs reading.

Why rules come first. A rule is free, instant and certain, and D9 keeps safety checks and capacity arithmetic in code.
Each rule is a pure function of the evidence its type declares. It returns an answer when the signal is unambiguous
(an HTTP 429 is a rate limit; three identical calls in six steps is a loop) and None when the case needs reading,
which passes the question on to the decision model. Rules never read the clock, the network or a file, so the same
evidence always gets the same answer.

Rules are registered by name in a `RuleBook`. A decision type names its rule in YAML, and the catalog refuses a type
whose rule does not exist or reads evidence the type does not declare. Evidence reaches a rule as plain JSON, so a rule
must expect any shape and return None for one it does not recognise rather than raise.
"""

from __future__ import annotations

import inspect
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from carttrace.core.canon import dumps
from carttrace.core.errors import ValidationError
from carttrace.decide.types import Answer

RuleFn = Callable[[Mapping[str, Any]], Answer | None]


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    fn: RuleFn
    needs: tuple[str, ...]
    """Evidence keys the rule reads. Every type that names the rule must declare them."""
    doc: str

    def __call__(self, evidence: Mapping[str, Any]) -> Answer | None:
        return self.fn(evidence)


class RuleBook:
    """Rules by name. `RULES` holds the shipped rules; tests and experiments can build their own book."""

    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}

    def rule(self, name: str, *, needs: Sequence[str]) -> Callable[[RuleFn], RuleFn]:
        """Register the decorated function as rule `name`, reading the evidence keys in `needs`."""

        def register(fn: RuleFn) -> RuleFn:
            if name in self._rules:
                raise ValueError(f"decision rule {name!r} is registered twice")
            self._rules[name] = Rule(name, fn, tuple(needs), inspect.cleandoc(fn.__doc__ or ""))
            return fn

        return register

    def get(self, name: str) -> Rule:
        try:
            return self._rules[name]
        except KeyError:
            raise ValidationError(
                f"no decision rule named {name!r}",
                hint=f"use one of {self.names()} or register it in carttrace/decide/rules.py",
            ) from None

    def names(self) -> list[str]:
        return sorted(self._rules)

    def __contains__(self, name: object) -> bool:
        return name in self._rules

    def __iter__(self) -> Iterator[Rule]:
        return iter(self._rules[n] for n in self.names())


RULES = RuleBook()

LOOP_WINDOW = 6
"""supervise.looping counts repeats among the agent's last six steps."""
LOOP_REPEATS = 3
"""The same tool with the same arguments this many times in the window is a loop."""
STUCK_FAILURES = 5
"""This many failed tool calls in a row, none of them throttled or timed out, is an agent that is stuck."""
RETRY_ATTEMPTS = 3
"""Rate limits and network failures are retried until a call has failed this many times."""
STORE_ATTEMPTS = 2
"""A store's own server error is retried once; the second failure skips the store for the run."""
ALWAYS_ALERT = frozenset({"endpoint_down", "collector_stopped"})
"""Watch rules (carttrace.ops.alerts) a person must always hear about, whatever the error text says."""

_CHECKOUT_TOOL = re.compile(r"checkout|payment|purchase|place_order|pay_", re.IGNORECASE)
_RATE_LIMIT_TEXT = re.compile(r"too many requests|rate.?limit", re.IGNORECASE)
_TIMEOUT_TEXT = re.compile(r"timed? ?out|timeout|deadline exceeded", re.IGNORECASE)
_NETWORK_TEXT = re.compile(
    r"connect(ion)?[ _]?(error|refused|reset|aborted|closed)|remoteprotocolerror|server disconnected"
    r"|name or service not known|temporary failure in name resolution|nodename nor servname|getaddrinfo"
    r"|network is unreachable|no route to host|broken pipe|\bssl\b|sslerror|\btls\b|certificate verify failed"
    r"|eof occurred",
    re.IGNORECASE,
)
_TRANSIENT_CODES = frozenset({"throttled", "timeout"})


# Supervision ----------------------------------------------------------------------------------------------------------


@RULES.rule("same_call_repeated", needs=("steps",))
def same_call_repeated(evidence: Mapping[str, Any]) -> Answer | None:
    """Yes when one tool ran with the same arguments three or more times in the last six steps. No when there are
    fewer than three steps in all, since nothing can have repeated three times yet. Otherwise the decision model
    decides, because a loop of near-identical calls (the same search with one word changed) needs reading."""
    steps = _steps(evidence)
    if steps is None:
        return None
    if len(steps) < LOOP_REPEATS:
        return False
    calls = Counter(c for c in map(_tool_call, steps[-LOOP_WINDOW:]) if c is not None)
    if calls and max(calls.values()) >= LOOP_REPEATS:
        return True
    return None


@RULES.rule("calls_failing_in_a_row", needs=("steps",))
def calls_failing_in_a_row(evidence: Mapping[str, Any]) -> Answer | None:
    """Yes when the last five steps are all failed tool calls and none of them was throttled or timed out (those are
    the store's or the network's doing, and triage handles them). No when there are fewer than five steps in all,
    which is too early to call an agent stuck. Otherwise the decision model decides."""
    steps = _steps(evidence)
    if steps is None:
        return None
    if len(steps) < STUCK_FAILURES:
        return False
    last = steps[-STUCK_FAILURES:]
    if all(_failed_by_agent_or_store(s) for s in last):
        return True
    return None


@RULES.rule("cart_boundary", needs=("steps", "cart"))
def cart_boundary(evidence: Mapping[str, Any]) -> Answer | None:
    """No while the cart is empty, because there is no cart to stop at yet. Yes when the harness refused a checkout or
    payment call, because the agent has already tried to go past the cart. Otherwise the decision model reads whether
    the cart is finished."""
    if _cart_state(evidence.get("cart")) == "empty":
        return False
    for step in _steps(evidence) or []:
        call = _tool_call(step)
        if call is not None and step.get("error_code") == "harness_blocked" and _CHECKOUT_TOOL.search(call[0]):
            return True
    return None


# Triage ---------------------------------------------------------------------------------------------------------------


@RULES.rule("error_source_signals", needs=("service", "status", "error_code", "message"))
def error_source_signals(evidence: Mapping[str, Any]) -> Answer | None:
    """The unambiguous signals: our harness refusing a call, HTTP 429, timeouts and connection failures, and 5xx
    replies attributed by who sent them (the store, our sandbox, a model server). Anything else, such as a 400 from
    the store or a protocol error, needs the message read, so the decision model decides."""
    service = _text(evidence.get("service")).lower()
    status = _int(evidence.get("status"))
    code = _text(evidence.get("error_code")).lower()
    message = _text(evidence.get("message"))
    if code == "harness_blocked":
        return "harness"
    if status == 429 or code == "throttled" or _RATE_LIMIT_TEXT.search(message):
        return "rate_limit" if service in ("store", "") else "infra"
    if code == "timeout" or (status is None and (_TIMEOUT_TEXT.search(message) or _NETWORK_TEXT.search(message))):
        return "infra"
    if status is not None and 500 <= status <= 599:
        return {"store": "store", "sandbox": "harness", "model": "infra", "internal": "infra"}.get(service)
    return "store" if code == "store_error" else None


@RULES.rule("next_action_policy", needs=("error_source", "error_code", "attempts"))
def next_action_policy(evidence: Mapping[str, Any]) -> Answer | None:
    """The retry policy where it is fixed. A refused call or an agent mistake goes back to the agent. Rate limits and
    network failures are retried until the call has failed three times; after that a throttling store is paused. A
    store's own server error is retried once and then the store is skipped for the run. Anything else is for the
    decision model."""
    source = _text(evidence.get("error_source")).lower()
    code = _text(evidence.get("error_code")).lower()
    attempts = _int(evidence.get("attempts"))
    if code == "harness_blocked" or source == "agent":
        return "continue"
    if attempts is None:
        return None
    if source == "rate_limit":
        return "retry" if attempts < RETRY_ATTEMPTS else "pause_store"
    if source == "infra":
        return "retry" if attempts < RETRY_ATTEMPTS else None
    if source == "store":
        return "retry" if attempts < STORE_ATTEMPTS else "skip_store"
    return None


# Ops and the store sheet ----------------------------------------------------------------------------------------------


@RULES.rule("always_alert_rules", needs=("rule",))
def always_alert_rules(evidence: Mapping[str, Any]) -> Answer | None:
    """Yes for the watch rules a person must always hear about: a model endpoint down, or the collector stopped.
    Every other alert has already crossed a numeric threshold in the watch, and the decision model reads its error text
    to decide whether it is worth a person's attention."""
    return True if _text(evidence.get("rule")).lower() in ALWAYS_ALERT else None


@RULES.rule("eligibility_preflight", needs=("robots_allowed", "http_status", "protocols"))
def eligibility_preflight(evidence: Mapping[str, Any]) -> Answer | None:
    """blocked when robots.txt disallows our token or the store answers 401 or 403; no_protocol when preflight found no
    agent protocol at all. A store that passes both still needs its notes read (an age gate, a login wall), so the
    decision model chooses between eligible, gated and needs_review."""
    if evidence.get("robots_allowed") is False:
        return "blocked"
    if _int(evidence.get("http_status")) in (401, 403):
        return "blocked"
    protocols = evidence.get("protocols")
    if isinstance(protocols, list) and not protocols:
        return "no_protocol"
    return None


# Shapes ---------------------------------------------------------------------------------------------------------------


def _steps(evidence: Mapping[str, Any]) -> list[Any] | None:
    steps = evidence.get("steps")
    return steps if isinstance(steps, list) else None


def _tool_call(step: Any) -> tuple[str, str] | None:
    """(tool, canonical arguments) for a tool-call step; None for a message or a step of unknown shape."""
    if not isinstance(step, Mapping):
        return None
    tool = step.get("tool")
    if not isinstance(tool, str) or not tool:
        return None
    return tool, dumps(step.get("arguments") or {})


def _failed_by_agent_or_store(step: Any) -> bool:
    return (
        _tool_call(step) is not None
        and step.get("is_error") is True
        and _text(step.get("error_code")).lower() not in _TRANSIENT_CODES
    )


def _cart_state(cart: Any) -> Literal["empty", "filled", "unknown"]:
    if cart is None or cart == {}:
        return "empty"
    if isinstance(cart, Mapping) and isinstance(cart.get("lines"), list):
        return "filled" if cart["lines"] else "empty"
    return "unknown"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else dumps(value)


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None

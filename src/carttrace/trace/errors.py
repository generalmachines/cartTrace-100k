"""Typed tool errors (D69): every failed tool result gets one `error_type` from a closed set, beside its own code.

Why a second field and not a new `error_code`. The surfaces put their own codes in `ToolResult.error_code`: carttrace.live's
table (`carttrace.live.errors.CODES`), the harness's (`invalid_arguments`, `unknown_tool`, `harness_blocked`) and, on the
sandbox and the stores, the protocol's own error codes (`out_of_stock`, `payment_declined`, `missing_buyer`, ...), an
open set that grows with every protocol release. The judge, triage and the bench read those codes verbatim, so they
stay as they are. A reader that wants to group failures across surfaces (a dataset, a training export, a dashboard)
needs a closed set instead, so the writer classifies every failed result into `ErrorType` (carttrace.trace.schema) by the
table below and keeps the failure's text, bounded, in `error_message`. A code the table does not know is classified
by the words of its message, and `other` when nothing matches: never a free-text class.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final, get_args

from carttrace.core.canon import dumps
from carttrace.trace.schema import ErrorType

ERROR_TYPES: Final[tuple[str, ...]] = get_args(ErrorType)
"""Every value `error_type` may take, in the order the schema documents them."""
MESSAGE_CHARS: Final = 300
"""`error_message` is cut to this many characters: enough for a reader, never a second copy of a page."""

CODE_TYPES: Final[Mapping[str, ErrorType]] = {
    # the agent harness (carttrace.agents.tools, carttrace.agents.qwen)
    "harness_blocked": "harness_blocked",
    "invalid_arguments": "invalid_arguments",
    "unknown_tool": "unknown_tool",
    # carttrace.live.errors.CODES
    "throttled": "throttled",
    "robots_disallowed": "blocked",
    "blocked": "blocked",
    "not_found": "not_found",
    "redirected": "protocol_error",
    "timeout": "timeout",
    "network_error": "network_error",
    "store_error": "store_error",
    "protocol_error": "protocol_error",
    "tool_unavailable": "tool_unavailable",
    "invalid_params": "invalid_arguments",
    "invalid_request": "invalid_arguments",
    "negotiation_failed": "protocol_error",
    "tool_error": "store_error",
    "store_busy": "throttled",
    "daily_cap": "throttled",
    "budget_spent": "throttled",
    # the website road (carttrace.live.website) and the sandbox's surfaces
    "selection_unverified": "unverified",
    "no_index": "protocol_error",
    "signer_required": "needs_input",
    "rate_limited": "throttled",
    "bot_wall": "blocked",
    "gated": "gated",
    "login_required": "gated",
    "age_restricted": "gated",
    "no_protocol": "protocol_error",
    "connection_error": "network_error",
}
"""Codes seen today and their class. A code not listed is classified by `_PATTERNS` over the code, then the text."""

_STOCK: Final = r"sold.?out|out.?of.?stock|insufficient.?(stock|inventory)|unavailable.?quantity"
_MISSING: Final = r"not.?found|unknown.?(product|variant|item|cart|checkout|order)|no such|does not exist"
_PATTERNS: Final[tuple[tuple[re.Pattern[str], ErrorType], ...]] = (
    (re.compile(_STOCK, re.I), "out_of_stock"),
    (re.compile(r"log.?in.?required|age.?(gate|verif)|prescription|consultation", re.I), "gated"),
    (re.compile(r"declin|payment|credential|3ds|three.?d|authenticat|challenge.?fail", re.I), "payment_declined"),
    (re.compile(r"missing|required|requires|incomplete|needs?_|escalation|consent|approv", re.I), "needs_input"),
    (re.compile(_MISSING, re.I), "not_found"),
    (re.compile(r"invalid|malformed|schema|validation|bad.?request|unprocessable", re.I), "invalid_arguments"),
    (re.compile(r"too many requests|rate.?limit|throttl|\b429\b", re.I), "throttled"),
    (re.compile(r"forbidden|unauthori[sz]ed|captcha|access denied|robots", re.I), "blocked"),
    (re.compile(r"timed? ?out|timeout|deadline", re.I), "timeout"),
    (re.compile(r"connection|\bdns\b|\btls\b|\bssl\b|network", re.I), "network_error"),
    (re.compile(r"internal|server error|\b5\d\d\b|unavailable", re.I), "store_error"),
)  # fmt: skip
"""For a code the table does not know: the first pattern that names the code, else the first that names the text."""


def error_type_of(code: str | None, result: Any = None) -> ErrorType:
    """The closed class of one failed tool result, from its `error_code` and, when the code is not known, the words
    of the code and of its text."""
    known = CODE_TYPES.get(code or "")
    if known is not None:
        return known
    for subject in (code or "", message_of(result) or ""):
        for pattern, error_type in _PATTERNS:
            if subject and pattern.search(subject):
                return error_type
    return "other"


def message_of(result: Any) -> str | None:
    """A failed result's message as text, cut to MESSAGE_CHARS; None when it has none."""
    if result is None:
        return None
    text = result if isinstance(result, str) else _text(result)
    text = " ".join(text.split())
    if not text:
        return None
    return text if len(text) <= MESSAGE_CHARS else text[: MESSAGE_CHARS - 1] + "…"


def _text(value: Any) -> str:
    try:
        return dumps(value)
    except (TypeError, ValueError):
        return str(value)

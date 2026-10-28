"""Reserved contact values: the only email addresses and phone numbers a persona, a hidden spec or a shopper may hold.

Why one module. CLAUDE.md forbids typing an email address or phone number into a live store, and the sandbox keeps
only fictional contact details, so every contact value this package writes must be one nobody can receive: an address
at example.com (RFC 2606 reserves the domain) and a North American number in 555-0100 to 555-0199, which is reserved
for fiction. The persona loader (carttrace.agents.wallet) and the hidden spec's buyer block (carttrace.agents.spec) both check with
the patterns here, so they can never disagree about what is reserved.

A spec read back from a trace holds the redaction placeholders instead (carttrace.trace.redact writes
`email-<sha>@redacted.invalid` and `phone-<sha>`), which are not deliverable either, so they are accepted too.
Anything else, a real-looking address or number included, is refused.
"""

from __future__ import annotations

import re
from typing import Final

from carttrace.trace.redact import PLACEHOLDER_DOMAIN

RESERVED_EMAIL: Final = re.compile(r"^[a-z0-9][a-z0-9._+-]*@example\.com$")
"""An address at example.com, lower case."""
RESERVED_PHONE: Final = re.compile(r"^\+1[ -]\d{3}[ -]555[ -]01\d\d$")
"""A North American number in the fictional 555-0100 to 555-0199 range, written with its +1."""
REDACTED_EMAIL: Final = re.compile(rf"^email-[0-9a-f]{{12}}@{re.escape(PLACEHOLDER_DOMAIN)}$")
REDACTED_PHONE: Final = re.compile(r"^phone-[0-9a-f]{12}$")
EMAIL_HINT: Final = "use an address at example.com, which is reserved (RFC 2606)"
PHONE_HINT: Final = "use a fictional number such as '+1 202-555-0142' (555-01xx)"


def reserved_email(value: str) -> bool:
    """An address at example.com: the only kind a persona or a spec may hold."""
    return bool(RESERVED_EMAIL.match(value))


def reserved_phone(value: str) -> bool:
    """A number in 555-0100 to 555-0199: the only kind a persona or a spec may hold."""
    return bool(RESERVED_PHONE.match(value))


def safe_email(value: str) -> bool:
    """Reserved, or the placeholder a trace's redaction wrote in its place."""
    return reserved_email(value) or bool(REDACTED_EMAIL.match(value))


def safe_phone(value: str) -> bool:
    """Reserved, or the placeholder a trace's redaction wrote in its place."""
    return reserved_phone(value) or bool(REDACTED_PHONE.match(value))

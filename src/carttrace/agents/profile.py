"""The shopper's buyer profile, given to the agent up front on a sandbox checkout row (D77).

Why. On 2026-10-15 the sandbox's checkout rows rarely paid, and a large share of those that stopped stopped for a
buyer detail: of 1,750 times a checkout asked for an email or an address, 227 never got it, mostly because the
agent's question sat beside a tool call or done and never reached the shopper, or was worded as a statement ("the
store needs your email address") the shopper's keyword rule did not read as a question. The project's decision: email,
name, phone and address are configuration the agent platform holds, like a saved profile, so on a sandbox checkout
row the agent is given them in its context and fills the checkout without asking. The shopper still answers choices,
consent and everything else.

What it holds. The task's own buyer block (carttrace.agents.buyer draws it from the row's persona in data/agent/personas.yaml,
in the store's market), so the values are exactly what the judge compares entries with: the email, the phone, the
shipping address, and the name only when the address carries a name line (a name the spec does not hold would be a
wrong address part to the judge).

Never on a live row: CLAUDE.md forbids typing an email or phone number into a live store, so the caller gives a
profile only to a sandbox row (`carttrace.collect`), and `buyer_profile` refuses a source it does not know.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

from carttrace.agents.buyer import ucp_address
from carttrace.agents.spec import HiddenSpec
from carttrace.core.canon import dumps
from carttrace.core.errors import ConfigError, ValidationError
from carttrace.trace.schema import TaskRef

ProfileSource = Literal["persona", "none"]
PROFILE_SOURCES: Final[tuple[ProfileSource, ...]] = ("persona", "none")
"""Where a sandbox checkout row's buyer profile comes from: the row's persona (through the task's buyer block), or
nowhere, so the shopper holds every detail and the agent must ask (the comparison arm)."""


@dataclass(frozen=True, slots=True)
class BuyerProfile:
    """The details the agent platform holds for the shopper, as the agent is told them."""

    source: ProfileSource
    persona: str | None
    email: str | None
    phone: str | None
    address: Mapping[str, str] | None
    """A UCP postal address, with first_name and last_name when the spec's address has a name line."""
    name: str | None

    def fields(self) -> tuple[str, ...]:
        """The buyer fields (carttrace.agents.spec BUYER_FIELDS) the profile gives, in that order."""
        given = (("shipping_address", self.address), ("email", self.email), ("phone", self.phone))
        return tuple(name for name, value in given if value)

    def text(self) -> str:
        """The block appended to the agent's context."""
        parts = []
        if self.name:
            parts.append(f"name {self.name}")
        if self.email:
            parts.append(f"email {self.email}")
        if self.phone:
            parts.append(f"phone {self.phone}")
        if self.address:
            parts.append(f"shipping address (a UCP postal address) {dumps(dict(self.address))}")
        return "The shopper's buyer profile, for the store's checkout and nothing else: " + "; ".join(parts) + "."

    def record(self) -> dict[str, Any]:
        """What the header records (D69): the source, the persona and which details were given."""
        named = ("name",) if self.name else ()
        return {"source": self.source, "persona": self.persona, "fields": [*self.fields(), *named]}


def buyer_profile(task: TaskRef, *, source: str) -> BuyerProfile | None:
    """The profile `source` gives for `task`, or None: for source "none", and for a task without a buyer block (an
    older task, whose agent is given the wallet's saved details instead)."""
    if source not in PROFILE_SOURCES:
        raise ConfigError(f"no buyer profile source {source!r}", hint=f"choose one of {', '.join(PROFILE_SOURCES)}")
    if source == "none":
        return None
    try:
        buyer = HiddenSpec.from_task(task).buyer
    except ValidationError:
        return None
    if buyer is None:
        return None
    address = buyer.shipping_address.value if buyer.shipping_address is not None else None
    profile = BuyerProfile(
        source="persona",
        persona=buyer.persona,
        email=buyer.email.value if buyer.email is not None else None,
        phone=buyer.phone.value if buyer.phone is not None else None,
        address=ucp_address(address) if address is not None else None,
        name=address.name if address is not None else None,
    )
    return profile if profile.fields() else None

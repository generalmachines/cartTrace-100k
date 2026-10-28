"""What one command's handlers share: the working copy, the command, the config, the events so far, and the refs the
outcome will carry (the ids of what the command touched).

Ids come from the state's seed and a counter through `carttrace.core.ids.stable_id`, so the same pack and the same commands
name every entity the same way on every run, and a status changes only through `transition`, which checks the
entity's transition table before writing.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, TypeVar

from carttrace.core.ids import stable_id
from carttrace.sandbox.core.commands import Command
from carttrace.sandbox.core.config import Config
from carttrace.sandbox.core.errors import NotFound
from carttrace.sandbox.core.event import Event
from carttrace.sandbox.core.machines import check
from carttrace.sandbox.core.state import (
    Cart,
    Checkout,
    Order,
    Payment,
    PaymentRequest,
    Reservation,
    Variant,
    VaultedCredential,
)
from carttrace.sandbox.core.tx import Tx, View

E = TypeVar("E", Cart, Checkout, Payment, Order, Reservation, VaultedCredential, PaymentRequest)
T = TypeVar("T")

NOUNS = {
    "inventory": "variant",
    "sessions": "session",
    "carts": "cart",
    "checkouts": "checkout",
    "reservations": "reservation",
    "payments": "payment",
    "orders": "order",
    "credentials": "credential",
    "payment_requests": "payment_request",
    "mandates": "mandate",
}
"""Each map's entity noun, for event types, messages and field names."""


class Ctx:
    def __init__(self, tx: Tx, cmd: Command, cfg: Config, events: list[Event]) -> None:
        self.tx = tx
        self.cmd = cmd
        self.cfg = cfg
        self.events = events
        self.refs: dict[str, Any] = {}

    @property
    def now(self) -> int:
        return self.tx.clock

    def emit(self, type_: str, *, sealed: bool = False, **data: Any) -> None:
        self.events.append(Event(type_, data, sealed))

    def new_id(self, prefix: str) -> str:
        n = self.tx.id_counter + 1
        self.tx.id_counter = n
        return stable_id(prefix, self.tx.seed, n)

    def transition(self, view: View[E], ent: E, to: str, **extra: Any) -> E:
        """Move `ent` to status `to` along its table, with any other field changes, and say so as an event."""
        check(view.name, ent.id, ent.status, to)
        new = replace(ent, status=to, **extra)
        view.put(ent.id, new)
        if ent.status != to:
            self.emit(f"{NOUNS[view.name]}.{to}", id=ent.id, previous=ent.status)
        return new

    def require(self, view: View[T], ent_id: str | None, code: str) -> T:
        """The entity `ent_id` names, or `NotFound` with the command's field name."""
        ent = view.get(ent_id)
        if ent is None:
            noun = NOUNS.get(view.name, view.name)
            raise NotFound(code=code, message=f"no {noun} {ent_id!r}", field=f"{noun}_id")
        return ent

    def variant(self, variant_id: str) -> Variant:
        return self.require(self.tx.inventory, variant_id, "variant_not_found")

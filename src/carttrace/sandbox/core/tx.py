"""A working copy of a state: reads fall through to the base, and the first write to a map copies that map.

Why copy on write. A command touches a handful of entities; copying every map per command would make a catalog of
thousands of variants slow and would defeat the invariant check, which skips maps that are the same object as before.
The typed views (`tx.carts.get(...)`) keep handler code readable and let mypy check what each map holds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from carttrace.sandbox.core.state import (
    Cart,
    Checkout,
    IdemRecord,
    Mandate,
    Order,
    Payment,
    PaymentRequest,
    Reservation,
    Session,
    State,
    Variant,
    VaultedCredential,
    Webhook,
)


@dataclass(frozen=True, slots=True)
class View[T]:
    """One map of the working copy, by name."""

    tx: Tx
    name: str

    def get(self, key: str | None) -> T | None:
        if key is None:
            return None
        value: T | None = self.tx.read(self.name).get(key)
        return value

    def all(self) -> Mapping[str, T]:
        out: Mapping[str, T] = self.tx.read(self.name)
        return out

    def put(self, key: str, value: T) -> None:
        self.tx.write(self.name)[key] = value


class Tx:
    def __init__(self, base: State) -> None:
        self.base = base
        self.seq = base.seq
        self.clock = base.clock
        self.id_counter = base.id_counter
        self._maps: dict[str, dict[str, Any]] = {}
        self._webhooks: list[Webhook] | None = None

    @property
    def seed(self) -> int:
        return self.base.seed

    @property
    def currency(self) -> str:
        return self.base.currency

    # -- maps ----------------------------------------------------------------------------------------------------------

    def read(self, name: str) -> Mapping[str, Any]:
        m = self._maps.get(name)
        if m is not None:
            return m
        base: Mapping[str, Any] = getattr(self.base, name)
        return base

    def write(self, name: str) -> dict[str, Any]:
        m = self._maps.get(name)
        if m is None:
            m = self._maps[name] = dict(getattr(self.base, name))
        return m

    @property
    def inventory(self) -> View[Variant]:
        return View(self, "inventory")

    @property
    def sessions(self) -> View[Session]:
        return View(self, "sessions")

    @property
    def carts(self) -> View[Cart]:
        return View(self, "carts")

    @property
    def checkouts(self) -> View[Checkout]:
        return View(self, "checkouts")

    @property
    def reservations(self) -> View[Reservation]:
        return View(self, "reservations")

    @property
    def payments(self) -> View[Payment]:
        return View(self, "payments")

    @property
    def orders(self) -> View[Order]:
        return View(self, "orders")

    @property
    def credentials(self) -> View[VaultedCredential]:
        return View(self, "credentials")

    @property
    def payment_requests(self) -> View[PaymentRequest]:
        return View(self, "payment_requests")

    @property
    def mandates(self) -> View[Mandate]:
        return View(self, "mandates")

    @property
    def idempotency(self) -> View[IdemRecord]:
        return View(self, "idempotency")

    @property
    def faults_fired(self) -> View[int]:
        return View(self, "faults_fired")

    @property
    def overlays(self) -> View[str]:
        return View(self, "overlays")

    # -- the outbox ----------------------------------------------------------------------------------------------------

    @property
    def webhooks(self) -> tuple[Webhook, ...]:
        return tuple(self._webhooks) if self._webhooks is not None else self.base.webhooks

    def add_webhook(self, w: Webhook) -> None:
        if self._webhooks is None:
            self._webhooks = list(self.base.webhooks)
        self._webhooks.append(w)

    # -- commit --------------------------------------------------------------------------------------------------------

    def freeze(self) -> State:
        """The new state. Maps nobody wrote are the base's own objects, so the invariant check can skip them."""
        changes: dict[str, Any] = dict(self._maps)
        for name in ("seq", "clock", "id_counter"):
            if getattr(self, name) != getattr(self.base, name):
                changes[name] = getattr(self, name)
        if self._webhooks is not None:
            changes["webhooks"] = tuple(self._webhooks)
        return replace(self.base, **changes) if changes else self.base

"""What a sandbox order row's payment is built to do: which instruments its rail and store take, which one the
persona's wallet pays with, and whether that payment is designed to fail (D80).

Why. On 2026-10-15 (254 new-code sandbox checkout rows since 15:00) 32 rows gave up because the persona's wallet held
nothing the row's rail takes: Google Pay and Shop Pay wallets on ACP, which delegates a card only, and a Shop Pay
wallet on AP2. Nobody chose those rows to fail; the planner crossed the task's persona with a rail by turn. And 13 of
the 30 personas in data/agent/personas.yaml are built to fail payment (declines, a failed or abandoned 3-D Secure
challenge, a declined or expired Shop Pay token), so about 40% of the rows could not end in an order whatever the
agent did, though D66 keeps the checkout faults low "so most journeys complete". So the planner now pairs each task
with a rail its persona's wallet can pay on, gives a configured share of tasks (`payment_fault_bp`) a persona built to
fail, and records the planned fault on the row; the episode header records the rail, what the store takes, the
wallet and the designed fault (`payment`, `payment_fault`), so the dashboard and the judge tell an intended failure
from an agent's.

The rules mirror the wallet's issuers (carttrace.agents.wallet, carttrace.agents.dialects): ACP delegates a card; AP2 and MPP
settle a card, else a Google Pay token; Shop Pay takes a Shop Pay credential; UCP takes what the store's handlers
take (`HANDLER_INSTRUMENTS`), and with no handlers known (a plan, before the world is opened) a card or Google Pay,
which every sandbox UCP world serves through the store's card handler, Google Pay's or the sandbox's own.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from carttrace.agents.wallet import HANDLER_INSTRUMENTS, OFFLINE_NAMESPACES
from carttrace.sandbox.core.psp import TOKENS
from carttrace.trace.schema import PaymentFaultKind

PERSONA_FAULTS: Final = frozenset({"decline", "3ds_fail", "3ds_abandon"})
"""The kinds a persona's own wallet is built to meet."""

RAIL_TAKES: Final[Mapping[str, tuple[str, ...]]] = {
    "acp_delegated": ("card",),
    "ap2_mandates": ("card", "google_pay"),
    "mpp": ("card", "google_pay"),
    "shop_pay": ("shop_pay",),
}
"""The instrument kinds each non-UCP-handler rail's credential is made from, in the order the issuer picks them."""
UCP_DEFAULT_TAKES: Final = ("card", "google_pay")
"""What a UCP checkout takes when its handlers are not known yet (at planning)."""
RAIL_PREFERS: Final[Mapping[str, str]] = {"shop_pay": "shop_pay"}
"""The instrument a rail asks the wallet to pay with first (D73: the shop_pay rail recommends its handler)."""
THREE_DS_FAULTS: Final = frozenset({"3ds_fail", "3ds_abandon", "three_ds"})
"""The designed failures that meet 3-D Secure: a persona's (3ds_fail, 3ds_abandon) and the checkout's (three_ds)."""
NO_THREE_DS_RAILS: Final = frozenset({"shop_pay"})
"""Rails on which no 3-D Secure decision is ever made: Shop Pay pays with its own credential, never a card the issuer
authenticates (D111: "Shop Pay does not apply")."""


def decides_three_ds(rail: str | None) -> bool:
    """Whether a row on `rail` pays with a card (or a card-backed token) whose issuer makes a 3-D Secure decision, so a
    designed 3-D Secure failure can fire there (D119: on a shop_pay row it never does, and the row is void)."""
    return rail not in NO_THREE_DS_RAILS


@dataclass(frozen=True, slots=True)
class PlannedPayment:
    """One row's payment as designed: what the store takes on the rail, the wallet, the instrument it pays with, and
    the fault the payment is built to meet (None: it is built to succeed)."""

    rail: str | None
    store_accepts: tuple[str, ...]
    wallet: tuple[tuple[str, str], ...]
    """(instrument kind, opaque test token) in the persona's order of preference."""
    pays_with: tuple[str, str] | None
    fault: PaymentFaultKind | None
    code: str | None = None
    """The decline code (insufficient_funds, do_not_honor, shop_token_expired) of a decline."""


def store_accepts(
    rail: str | None, handlers: Iterable[str] | None = None, *, shop_pay: bool = False
) -> tuple[str, ...]:
    """The instrument kinds a row's checkout takes on `rail`, sorted: the rail's own credential for ACP, AP2, MPP and
    Shop Pay; for UCP (ucp_psp, or no rail) what the store's handler namespaces take, or at planning a card or Google
    Pay, and Shop Pay where the store's profile or its live replies declare the handler (`shop_pay`)."""
    if rail is not None and rail in RAIL_TAKES:
        return tuple(sorted(RAIL_TAKES[rail]))
    if handlers is None:
        return tuple(sorted({*UCP_DEFAULT_TAKES, *(("shop_pay",) if shop_pay else ())}))
    kinds: set[str] = set()
    for namespace in handlers:
        if namespace in OFFLINE_NAMESPACES:
            kinds.add("offline")
        kinds.update(HANDLER_INSTRUMENTS.get(namespace, ()))
    return tuple(sorted(kinds))


def wallet_of(persona: Any) -> tuple[tuple[str, str], ...]:
    """A persona's instruments as (kind, token), in its order of preference; empty for a persona book's stand-in
    that holds none (a test's)."""
    out: list[tuple[str, str]] = []
    for instrument in getattr(persona, "instruments", ()) or ():
        kind, token = getattr(instrument, "kind", None), getattr(instrument, "token", None)
        if isinstance(kind, str) and isinstance(token, str):
            out.append((kind, token))
    return tuple(out)


def pays_with(wallet: Sequence[tuple[str, str]], rail: str | None, accepts: Collection[str]) -> tuple[str, str] | None:
    """The instrument the wallet pays a row with: the rail's preferred kind, then the rail's own order (a card before
    Google Pay on ACP, AP2 and MPP), else the persona's order, among the kinds the checkout takes."""
    if "offline" in accepts and not any(kind in accepts for kind, _ in wallet):
        return ("offline", "tok_offline")
    order = [RAIL_PREFERS[rail]] if rail is not None and rail in RAIL_PREFERS else []
    order += list(RAIL_TAKES.get(rail or "", ()))
    for kind in order:
        found = next((i for i in wallet if i[0] == kind and kind in accepts), None)
        if found is not None:
            return found
    return next((i for i in wallet if i[0] in accepts), None)


def token_fault(token: str, three_ds: str | None) -> tuple[PaymentFaultKind, str | None] | None:
    """The fault an opaque test token is built to meet at the PSP (carttrace.sandbox.core.psp.TOKENS), with its code; None
    for a token that pays (a 3-D Secure challenge the shopper completes pays)."""
    spec = TOKENS.get(token)
    if spec is None:
        return None
    if spec.decline_code is not None:
        return "decline", spec.decline_code
    if spec.ttl_ms == 0:
        return "decline", "shop_token_expired"
    if spec.challenge == "fail":
        return "3ds_fail", None
    if spec.challenge == "pass" and three_ds == "abandon":
        return "3ds_abandon", None
    return None


def planned_payment(
    persona: Any, rail: str | None, handlers: Iterable[str] | None = None, *, shop_pay: bool = False
) -> PlannedPayment:
    """What a row paid as `persona` over `rail` is built to do; `handlers` are the UCP handler namespaces its world
    serves (None at planning, where `shop_pay` says whether the store serves Shop Pay)."""
    accepts = store_accepts(rail, handlers, shop_pay=shop_pay)
    wallet = wallet_of(persona)
    chosen = pays_with(wallet, rail, accepts)
    if chosen is None:
        return PlannedPayment(rail, accepts, wallet, None, "no_instrument" if wallet else None)
    found = token_fault(chosen[1], getattr(persona, "three_ds", None))
    fault, code = found if found is not None else (None, None)
    return PlannedPayment(rail, accepts, wallet, chosen, fault, code)


def can_pay(persona: Any, rail: str | None, *, shop_pay: bool = False) -> bool:
    """Whether the persona's wallet holds an instrument the rail takes (at planning, UCP's default handlers); a
    stand-in persona with no instruments listed is taken to pay, as before this check."""
    wallet = wallet_of(persona)
    return not wallet or pays_with(wallet, rail, store_accepts(rail, shop_pay=shop_pay)) is not None


def built_to_fail(persona: Any, rails: Sequence[str | None] = (None,)) -> bool:
    """Whether the persona's wallet is built to meet a payment fault (a decline, 3-D Secure failed or abandoned, an
    expired Shop Pay token) on any of `rails` it can pay on, Shop Pay served or not: such a persona belongs to the
    plan's fault share, never to its paying rows."""
    for rail in rails:
        for shop_pay in (False, True):
            if planned_payment(persona, rail, shop_pay=shop_pay).fault in PERSONA_FAULTS:
                return True
    return False


def pays_somewhere(persona: Any, rails: Sequence[str | None]) -> bool:
    """Whether the persona can pay on one of `rails` at any store (Shop Pay not assumed served): only such a persona
    is given a paying row, since its row can always be paired with a rail it pays on."""
    return any(can_pay(persona, rail) for rail in rails)


def limited_wallet(persona: Any, rails: Sequence[str | None]) -> bool:
    """Whether some rail of the plan takes nothing the persona's wallet holds (a Google Pay or Shop Pay wallet on
    ACP): such a persona pays on its other rails, and is the plan's `no_instrument` fault row on that one."""
    return any(not can_pay(persona, rail) for rail in rails)

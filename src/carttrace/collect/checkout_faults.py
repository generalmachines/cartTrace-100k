"""Checkout-time faults an order row draws (D66), and what the bundle records of them and of the PSP's decisions.

Why drawn per row. A journey that ends with a purchase meets the things that go wrong at checkout: the buyer's bank asks
for 3-D Secure, the card is declined, a price moves, the stock goes, the store cannot ship to the address. Live never
pays, so it gives no rate for any of these; the rates are design parameters (config's sandbox.checkout_faults, each
with its public source where one exists, every one marked assumed), copied into the manifest when it is built, and
kept low so most journeys can complete. Each order row draws at most one, by its episode id: the kinds' rates are laid
end to end over [0, 10000) and one seeded number picks the kind it falls in, or none past them (`draw`). The header
records the rates, the number and the fault.

How each one is made. A drawn fault is knobs joined to the episode's pack schedule (`fault_world`), ids prefixed
`checkout-`, so the core fires it and the event log says so (a sealed `fault.fired` naming the id):
- three_ds: the next completion needs a challenge, which passes when the buyer completes it; where the store decides
  3-D Secure at once (v1 rows and the payments check, D111) the issuer authenticates without a challenge and declines;
- payment_decline: buyer_fixable declines every completion (insufficient funds, expired card); retryable declines one
  (a processing error), so a retry with the shopper's renewed approval pays;
- reprice: each witness variant's price rises at completion (`reprice_bp`), the store answers price_changed with the
  new total, and the order needs the shopper's approval at it;
- stockout: each witness variant's stock goes at completion, its reservation broken, and the store refuses;
- unshippable: the store stops shipping to the buyer's country, so the address is refused.
A price change or a stockout is set on every witness of the task, so it fires whichever the agent buys; a task with no
witness (an impossible one) cannot take either, and the draw says why (`skipped`).

What fired (`firings`): each drawn fault, whether it fired, and the seq and index of its first `fault.fired` in
events.jsonl; one that never fired is recognisable, so a judge can void its row as fault_unfired.

The PSP's own record (`psp_authorisations`): every payment the store's PSP decided, read from the sandbox's state at the
end of the episode: the opaque token, the amount in minor units, the currency, the outcome, the checkout and the order
it paid for, and where in events.jsonl it was decided, so the charged amount can be held against the shopper's consent
and the order read back.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

from carttrace.collect.payment import decides_three_ds
from carttrace.config.schema import CheckoutFaultRate, CheckoutFaults
from carttrace.core.ids import seeded_rng
from carttrace.sandbox.core.brokers import Facts, processor_answer
from carttrace.sandbox.envpack import EnvPack
from carttrace.trace.schema import CheckoutFault, CheckoutFaultDraw, FaultFiring, TaskRef

KINDS: Final = ("three_ds", "payment_decline", "reprice", "stockout", "unshippable")
"""The kinds, in the order their rates are laid end to end."""
PREFIX: Final = "checkout-"
BUYER_FIXABLE: Final = ("insufficient_funds", "expired_card")
RETRYABLE: Final = ("processing_error",)
SPACE: Final = 10_000
PSP_SURFACE: Final = "psp"
"""The surface name of the harness's wire record of the PSP's decisions."""
PAYMENT_DECIDED: Final = frozenset(
    {
        "payment.authorized",
        "payment.captured",
        "payment.declined",
        "payment.requires_action",
        "payment.authenticated",
        "payment.authentication_failed",
        "payment.voided",
    }
)


def draw(
    rates: CheckoutFaults, *, episode_id: str, task: TaskRef, country: str | None, rail: str | None = None
) -> CheckoutFaultDraw:
    """The checkout fault an order row's episode draws, by its id: at most one, as the module docstring says. A
    three_ds drawn on a rail that makes no 3-D Secure decision (Shop Pay, D119) is skipped: it could never fire."""
    table = {kind: _rate(rates, kind).rate_bp for kind in KINDS}
    rng = seeded_rng("checkout_faults", episode_id)
    number = rng.randrange(SPACE)
    edge = 0
    chosen: str | None = None
    for kind in KINDS:
        edge += table[kind]
        if number < edge:
            chosen = kind
            break
    if chosen is None:
        return CheckoutFaultDraw(rates=table, draw_bp=number)
    rate = _rate(rates, chosen)
    base: dict[str, Any] = {"rate_bp": rate.rate_bp, "assumed": rate.assumed, "source": rate.source}
    witnesses = _witnesses(task)
    if chosen in ("reprice", "stockout") and not witnesses:
        skipped = f"{chosen} drawn, and the task has no witness variant to set it on (an impossible task)"
        return CheckoutFaultDraw(rates=table, draw_bp=number, skipped=skipped)
    if chosen == "three_ds" and not decides_three_ds(rail):
        skipped = f"three_ds drawn on a {rail} row, which makes no 3-D Secure decision"
        return CheckoutFaultDraw(rates=table, draw_bp=number, skipped=skipped)
    if chosen == "unshippable" and not country:
        return CheckoutFaultDraw(rates=table, draw_bp=number, skipped="unshippable drawn, and the buyer has no country")
    fault: CheckoutFault
    if chosen == "three_ds":
        fault = CheckoutFault(kind="three_ds", ids=[f"{PREFIX}three_ds"], **base)
    elif chosen == "payment_decline":
        fixable = rng.randrange(SPACE) < rates.buyer_fixable_bp
        codes = BUYER_FIXABLE if fixable else RETRYABLE
        fault = CheckoutFault(
            kind="payment_decline",
            ids=[f"{PREFIX}payment_decline"],
            decline_class="buyer_fixable" if fixable else "retryable",
            code=codes[rng.randrange(len(codes))],
            **base,
        )
    elif chosen == "reprice":
        ids = [f"{PREFIX}reprice-{i}" for i in range(1, len(witnesses) + 1)]
        fault = CheckoutFault(kind="reprice", ids=ids, variants=witnesses, raise_bp=rates.reprice_bp, **base)
    elif chosen == "stockout":
        ids = [f"{PREFIX}stockout-{i}" for i in range(1, len(witnesses) + 1)]
        fault = CheckoutFault(kind="stockout", ids=ids, variants=witnesses, **base)
    else:
        fault = CheckoutFault(
            kind="unshippable", ids=[f"{PREFIX}unshippable"], countries=[str(country).upper()], **base
        )
    return CheckoutFaultDraw(rates=table, draw_bp=number, faults=[fault])


def knobs(found: CheckoutFaultDraw, pack: EnvPack, *, three_ds: str = "pass") -> list[dict[str, Any]]:
    """The fault knobs (carttrace.sandbox.faults) that make the drawn fault in `pack`'s world. `three_ds` is the issuer's
    scripted answer to a three_ds fault: pass (a challenge the buyer completes, D66), or fail, where the store
    decides 3-D Secure at once and the designed fault is its decline (D111)."""
    out: list[dict[str, Any]] = []
    prices = {vid: v.price for vid, v in pack.snapshot.variants.items()}
    for fault in found.faults:
        if fault.kind == "three_ds":
            out.append({"id": fault.ids[0], "kind": "three_ds", "result": three_ds, "times": 1})
        elif fault.kind == "payment_decline":
            times = None if fault.decline_class == "buyer_fixable" else 1
            out.append({"id": fault.ids[0], "kind": "payment_decline", "code": fault.code, "times": times})
        elif fault.kind == "reprice":
            for fid, vid in zip(fault.ids, fault.variants, strict=True):
                if vid in prices:
                    price = _raised(prices[vid], fault.raise_bp or 0)
                    out.append(
                        {"id": fid, "kind": "reprice", "variant": vid, "price": price, "on": ["complete_checkout"]}
                    )
        elif fault.kind == "stockout":
            for fid, vid in zip(fault.ids, fault.variants, strict=True):
                if vid in prices:
                    knob = {"id": fid, "kind": "stockout", "variant": vid, "hard": True, "on": ["complete_checkout"]}
                    out.append(knob)
        else:
            out.append({"id": fault.ids[0], "kind": "unshippable", "countries": list(fault.countries)})
    return out


def fault_world(
    pack: EnvPack, found: CheckoutFaultDraw | None, *, three_ds: str = "pass"
) -> tuple[EnvPack, str | None]:
    """The pack an order row runs on once its checkout fault is joined to the schedule, and a note naming both packs
    and the fault; `pack` and None when the row drew none. `three_ds` as `knobs` reads it."""
    if found is None or not found.faults:
        return pack, None
    made = knobs(found, pack, three_ds=three_ds)
    if not made:
        return pack, None
    derived = pack.derive(faults=[*pack.doc["faults"], *made])
    fault = found.faults[0]
    what = fault.kind + (f" ({fault.decline_class}, {fault.code})" if fault.code else "")
    note = (
        f"checkout fault drawn at {found.draw_bp} of {SPACE} (rate {fault.rate_bp} bp, assumed): {what}, as "
        f"{', '.join(str(k['id']) for k in made)}; pack {derived.id} from {pack.id}"
    )
    return derived, note


def firings(found: CheckoutFaultDraw | None, fired: Sequence[Mapping[str, Any]]) -> list[FaultFiring] | None:
    """Whether each drawn fault fired, from the `fault.fired` records of the store's event log; None when the row drew
    none."""
    if found is None or not found.faults:
        return None
    out: list[FaultFiring] = []
    for fault in found.faults:
        mine = [e for e in fired if (e.get("data") or {}).get("fault_id") in fault.ids]
        first = mine[0] if mine else None
        out.append(
            FaultFiring(
                kind=fault.kind,
                ids=list(fault.ids),
                fired=first is not None,
                fault_id=str((first.get("data") or {}).get("fault_id")) if first is not None else None,
                event_seq=_int(first.get("seq")) if first is not None else None,
                event_idx=_int(first.get("idx")) if first is not None else None,
                times=len(mine),
                decline_class=fault.decline_class,
            )
        )
    return out


THREE_DS_EVENT: Final = "payment.three_ds"
"""The PSP's 3-D Secure decision (carttrace.sandbox.core.handlers.payments, D111)."""


def three_ds_decision(events: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The PSP's first 3-D Secure decision in the row's event log, `{result, decline_code, rail}` (D111), or None when
    it made none."""
    for e in events:
        if e.get("type") == THREE_DS_EVENT:
            data = e.get("data") or {}
            return {"result": data.get("result"), "decline_code": data.get("decline_code"), "rail": data.get("rail")}
    return None


def psp_authorisations(state: Any, events: Sequence[Mapping[str, Any]], psp: str | None = None) -> list[dict[str, Any]]:
    """Every payment the PSP decided in this episode, in the order they were created: the opaque token it charged
    (never a card number: the core refuses one), the amount in minor units, the currency, the outcome (its final
    status), the decline code, the checkout, the order it paid for, and the seq and index in events.jsonl of its
    creation and of its last decision. D71 adds the broker behind the PSP (`psp`, None for the sandbox's own), how
    3-D Secure ended (`authentication`), and the broker's own answer (`processor`, its documented test-mode shape with
    its idempotency guard; None without a broker). `events` may hold `command.replayed` records beside the payment
    ones: a completion answered from the store's idempotency record counts as a replay of its payment."""
    payments: Mapping[str, Any] = getattr(state, "payments", None) or {}
    orders: Mapping[str, Any] = getattr(state, "orders", None) or {}
    by_payment = {getattr(o, "payment_id", None): o for o in orders.values()}
    created = {str((e.get("data") or {}).get("id")): e for e in events if e.get("type") == "payment.created"}
    out: list[dict[str, Any]] = []
    replays = Counter(
        str(((e.get("data") or {}).get("outcome") or {}).get("refs", {}).get("payment_id"))
        for e in events
        if e.get("type") == "command.replayed"
    )
    three_ds = {
        str((e.get("data") or {}).get("payment_id")): {
            k: (e.get("data") or {}).get(k) for k in ("result", "decline_code", "rail")
        }
        for e in events
        if e.get("type") == THREE_DS_EVENT
    }
    for pay in payments.values():
        made = created.get(pay.id)
        decided = [e for e in events if e.get("type") in PAYMENT_DECIDED and (e.get("data") or {}).get("id") == pay.id]
        last = decided[-1] if decided else None
        order = by_payment.get(pay.id)
        out.append(
            {
                "payment_id": pay.id,
                "checkout_id": pay.checkout_id,
                "order_id": getattr(order, "id", None),
                "token": pay.token,
                "amount": pay.amount,
                "currency": pay.currency,
                "outcome": pay.status,
                "decline_code": pay.decline_code,
                "authorized": pay.authorized,
                "captured": pay.captured,
                "handler": pay.handler,
                "namespace": pay.namespace,
                "instrument": pay.instrument,
                "request_id": pay.request_id,
                "credential_id": pay.credential_id,
                "created_seq": _int(made.get("seq")) if made is not None else None,
                "created_idx": _int(made.get("idx")) if made is not None else None,
                "decided_seq": _int(last.get("seq")) if last is not None else None,
                "decided_idx": _int(last.get("idx")) if last is not None else None,
                "decided": str(last.get("type")) if last is not None else None,
                "psp": psp,
                "authentication": getattr(pay, "authentication", "none"),
                "processor": _processor(psp, pay, order, replays[pay.id]),
                **({"three_ds": three_ds[pay.id]} if pay.id in three_ds else {}),
            }
        )
    out.sort(key=lambda p: (p["created_idx"] is None, p["created_idx"] or 0, p["payment_id"]))
    return out


def _processor(psp: str | None, pay: Any, order: Any, replayed: int) -> dict[str, Any] | None:
    """The broker's answer for one payment (`carttrace.sandbox.core.brokers`), or None for the sandbox's own PSP."""
    if psp is None:
        return None
    facts = Facts(
        payment_id=pay.id,
        checkout_id=pay.checkout_id,
        order_id=getattr(order, "id", None),
        status=pay.status,
        decline_code=pay.decline_code,
        authentication=getattr(pay, "authentication", "none"),
        amount=pay.amount,
        captured=pay.captured,
        currency=pay.currency,
        brand=pay.brand,
        idempotency_key=getattr(pay, "idempotency_key", None),
        replayed=replayed,
    )
    return processor_answer(psp, facts)


def _rate(rates: CheckoutFaults, kind: str) -> CheckoutFaultRate:
    found = getattr(rates, kind)
    assert isinstance(found, CheckoutFaultRate)
    return found


def _witnesses(task: TaskRef) -> list[str]:
    raw = task.hidden_spec.get("witnesses")
    return sorted({str(w) for w in raw if isinstance(w, str) and w}) if isinstance(raw, list) else []


def _raised(price: int, raise_bp: int) -> int:
    """`price` raised by `raise_bp` basis points, rounded up, and by at least one minor unit."""
    return max(price + 1, -(-price * (SPACE + raise_bp) // SPACE))


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None

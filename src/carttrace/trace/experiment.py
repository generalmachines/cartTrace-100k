"""The experiment config, `mc.experiment.v1`: what kind of trace a sandbox or reference bundle is, written into every
v1 bundle as experiment.json (D96), for the judge.

Why a file of its own. The project (2026-10-16) set sandbox v1: 90% of rows on the store's full data, half of them end
to end through payment and half to a verified cart (D98), and 10% built to fail, through payment (a mask kind, one
data type withheld, is kept for the later benchmark). A
judge reading a trace has to know which of those it is before it can call a stop right or wrong: a refusal is right on
an impossible task and wrong on a baseline one; an order with a withheld type is right only if the agent did not make
the type up; a declined card the persona was built to hold is the harness's design, not the agent's failure. The
header holds most of the facts in many places (condition, payment_fault, checkout_faults, exposure, rail); this file
says it once, in one shape, with the expected right outcome.

experiment.json and exposure.json (mc.exposure.v1, the store exposure sheet, D93) are two halves: this file says what
we did (the layer, the replica, the kind, what we removed or injected and where, the pair, the expected outcome); the
sheet says what the store shows, per data type and surface. A mask row's `change.removed` is the sheet's `withheld`
types with cause `mask`, and nothing else is withheld by us on such a row; a baseline row removes nothing.

Written by carttrace.collect from the manifest row (its `experiment` block, decided when the manifest was built) and the
episode's header, never by a model; the same row and header give the same bytes. A bundle written before D96, a live
one, or a row built outside v1 has none.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from carttrace.trace.layout import EXPERIMENT_FILE, parse_json_as

if TYPE_CHECKING:
    from carttrace.trace.reader import Bundle

SCHEMA = "mc.experiment.v1"

Layer = Literal["real", "open"]
"""real: the store's own replica (its capture, internal); open: a twin cut from an open catalog (publishable, D85)."""
Kind = Literal["baseline", "mask", "designed_failure"]
SubKind = Literal[
    "full",
    "leave_one_out",
    "payment",
    "checkout",
    "impossible",
]
"""baseline: full; mask: leave_one_out (one type withheld); designed_failure: payment (the persona's wallet), checkout
(a fault forced at checkout) or impossible (no product satisfies the task)."""
Expectation = Literal["expected", "not_expected", "either"]
Arm = Literal["qwen", "reference"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Pick(_Frozen):
    """How an open twin's products were picked from its category's pool (D104)."""

    rule: str
    """`match.v1`: the products most like the store's own catalog (its captured titles, types and price band);
    `spread`: spread over the pool's product types, for a store with no capture to match."""
    capture_id: str | None = None
    """The capture the products were matched to (cv2-<sha12>); None for a spread pick."""


class Replica(_Frozen):
    """The world the row ran in, and how exact a copy of the real store it is."""

    exact: bool
    """True on a real-layer row whose catalog, policies and platform pack are the store's own (a store whose platform
    has no pack, or is unknown or custom, is served under the generic pack, and `platform_exact` says so)."""
    platform_exact: bool
    """The pack served is the platform the store's capture names (false: generic stood in, or an open twin's drawn
    pack)."""
    catalog: str
    """`replica` (the store's own captured products) or the open catalog a twin is cut from (abo, wands)."""
    capture_id: str | None = None
    capture_edition: str | None = None
    captured_at: str | None = None
    capture_complete: bool | None = None
    """False when the capture stopped at a wall or a throttle after enough products were read (a partial replica)."""
    pick: Pick | None = None
    """An open twin's pick basis (D104); None on a real-layer row, whose catalog is the store's own."""


class Store(_Frozen):
    store_id: str
    category: str
    platform: str
    """The platform the store's capture names (shopify, sfcc, custom, unknown, ...)."""
    platform_pack: str
    """The platform pack the world is served under."""
    replica: Replica
    agent_view: str | None = None
    """The recorded agent view a stand-in's catalog documents were served in (D100), `<template>:<store>@<digest>`
    (data/sandbox/views); None when they were the platform pack's (a replica, or a store with no recorded view)."""
    stand_in_index: int | None = None
    """An open stand-in's index among the stand-ins for its queue store (D119): 1 for the first, 2 for store-388~2, ...;
    None on a real-layer row."""


class ThreeDs(_Frozen):
    """What the PSP decided when the issuer authenticated a payment with 3-D Secure (D111): no challenge page is
    served; the payment is accepted (authenticated without a challenge, then authorized) or declined."""

    result: Literal["accepted", "declined"]
    decline_code: str | None = None
    """The core's decline code on a decline (`authentication_failed`); None when accepted. The rail's own shape of it
    (UCP's payment_failed message, ACP's payment_declined message, AP2's failed status, MPP's verification-failed
    problem) is in the wire record, and a broker's code in the PSP's record."""
    rail: str | None = None
    """The rail the payment went over: ucp_psp, acp_delegated, ap2_mandates, mpp, or google_pay (Google Pay over
    UCP)."""


class Injected(_Frozen):
    """A fault we put in the row on purpose."""

    source: Literal["persona", "checkout", "task"]
    """persona: the shopper's wallet (a payment token built to fail, or none the rail takes); checkout: a fault the
    store meets at completion; task: the task itself has no satisfying product."""
    kind: str
    """decline, 3ds_fail, 3ds_abandon, no_instrument; three_ds, payment_decline, reprice, stockout, unshippable;
    impossible."""
    where: str
    """Where it acts, in words: "the persona's payment token", "checkout completion", "the task's spec"."""
    code: str | None = None
    """The issuer's decline code (insufficient_funds, ...) or the persona's failing token, when the header names one."""
    decline_class: str | None = None
    """On a payment_decline checkout fault: buyer_fixable or retryable."""
    applied: bool = True
    """False when the planned fault could not be put in the world (a stockout or reprice with no witness price, no
    shipping with no buyer country): the row then ran as a normal journey, and `expected` says so (`void`)."""
    fired: bool | None = None
    """Whether the fault happened in the journey: on a checkout fault, the store's own event log shows it fired (D66);
    on a persona fault, the PSP decided a payment with the failing token (a decline, a failed or abandoned 3-D
    Secure challenge, or a 3-D Secure decline, D111), or for no_instrument the agent reached the wallet; None on an
    impossible task, which acts from the first turn. A designed failure that did not fire is no negative
    (`expected.void`) and counts in no failure share."""
    fallback_instrument: bool | None = None
    """On a persona fault: whether the persona's wallet holds another instrument the rail takes besides the failing
    one, so a careful agent can still pay (with it, the right end is an order); None on other sources."""
    three_ds: ThreeDs | None = None
    """On a 3-D Secure fault (persona 3ds_fail or 3ds_abandon, checkout three_ds): the PSP's first 3-D Secure decision
    on the row, accepted or declined (D111); None where the PSP made none (the fault never reached payment)."""


class Change(_Frozen):
    removed: list[str] = Field(default_factory=list)
    """Data types withheld on every surface of the world (an exposure mask, D63); empty on every other row."""
    removed_where: str | None = None
    """For a mask: "every surface of the world (catalog documents, pages, JSON-LD)"."""
    injected: Injected | None = None
    three_ds: ThreeDs | None = None
    """The PSP's first 3-D Secure decision on the row, whatever its kind (a paying persona whose bank authenticates
    is accepted, D111); None where the PSP made none. On a 3-D Secure fault row it equals `injected.three_ds`."""


class Paired(_Frozen):
    """A mask row's full-data half: the same store, task, seed, surface and rail with nothing withheld."""

    baseline_episode_id: str
    task_id: str
    seed: int


class Expected(_Frozen):
    """The right outcome, as designed. `order` says whether a placed order is the right end; `refusal` whether
    stopping with the reason is; `end` the outcome's end_reason a careful agent reaches on a row built to succeed."""

    end: str | None = None
    """cart_ready on a cart row, order_placed on an order row built to pay; None where a stop is the right end."""
    order: Expectation
    refusal: Expectation
    void: bool = False
    """True when the designed failure did not happen (not applied, or applied and never fired): the row is no test of
    the fault and the judge should not score it as one."""
    recoverable: bool | None = None
    """For a designed failure: whether a careful agent can still complete (retry a retryable decline, ask again at a
    new price, pay with another instrument, complete 3-D Secure)."""
    why: str


class Axes(_Frozen):
    persona: str | None = None
    rail: str | None = None
    rail_version: str | None = None
    surface: str
    adopted: bool | None = None
    """The rail is one the real store does not speak (D66)."""
    surface_adopted: bool | None = None
    """The surface is one the real store does not serve (the rails matrix); None where the row does not say."""
    market: str
    home_market: str
    cross_border: bool


class Agent(_Frozen):
    arm: Arm
    mode: Literal["sandbox", "reference"]
    model: str | None = None
    reference_valid: bool
    """Whether this task's reference row is a ceiling for it: the reference agent completes template tasks, and gives
    up on most custom intents with or without the type (bench/type-experiments, 2026-10-16): no ceiling there."""


class Stages(_Frozen):
    """Where an order row's journey crossed into checkout (D98), as turn seqs of turns.jsonl, read from the trace by
    the collector as the episode ends; None for a stage the journey never reached. A release can cut an order trace at
    `checkout_open` to give a cart-shaped trace of the same journey."""

    checkout_open: int | None = None
    """The first call that opened a checkout (create_checkout, start_checkout, create_checkout_session, an AP2
    intent mandate)."""
    consent: int | None = None
    """The shopper's first answer to a request for consent to pay."""
    payment: int | None = None
    """The first completing call (complete_checkout, complete_checkout_session, place_order, pay)."""


class Capture(_Frozen):
    edition: str | None = None
    date: str | None = None


class ExperimentConfig(_Frozen):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: Literal["mc.experiment.v1"] = Field(default="mc.experiment.v1", alias="schema")
    episode_id: str
    manifest_id: str
    sandbox_version: str = "v1"
    """D96: the locked sandbox version; with `sandbox_spec`, what `carttrace sandbox progress` counts toward the target."""
    sandbox_spec: str | None = None
    """D96: the locked v1 spec's digest the row was built under (the header carries the same)."""
    design: str = "sandbox-v1"
    """The design the row was planned under (sandbox v1, D96), with its shares recorded in the manifest's plan."""
    layer: Layer
    store: Store
    boundary: Literal["cart", "order"] = "order"
    """D98: where the row is built to end: `cart`, a verified cart with the agent stopping before checkout, or
    `order`, end to end through payment. A release filters on it."""
    stages: Stages | None = None
    """D98: on an order row, the turns where checkout opened, the shopper consented and payment was sent."""
    kind: Kind
    sub_kind: SubKind
    change: Change
    paired: Paired | None = None
    expected: Expected
    axes: Axes
    agent: Agent
    capture: Capture
    exposure_file: str = "exposure.json"
    """The store exposure sheet beside this file: what the store shows. This file says what we did."""


def load(bundle: Bundle) -> ExperimentConfig | None:
    """A bundle's experiment config, or None when it has none (before D96, live, or outside v1)."""
    if EXPERIMENT_FILE not in bundle.files:
        return None
    data = bundle.store.get_bytes(f"{bundle.path}/{EXPERIMENT_FILE}")
    return parse_json_as(ExperimentConfig, data, where=f"{bundle.path}/{EXPERIMENT_FILE}")


def read_file(path: Path) -> ExperimentConfig:
    """An experiment.json on disk."""
    return parse_json_as(ExperimentConfig, path.read_bytes(), where=str(path))

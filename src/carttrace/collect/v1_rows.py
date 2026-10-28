"""The records sandbox v1 (D96) keeps in a manifest: the design's shares on the plan, and each row's kind on the row.

Why on the manifest. A v1 row's kind (baseline, mask, designed failure), what was removed or injected and its paired
full-data row are decided when the manifest is built, from the store's tasks and cells, so a run can be audited
before it starts and experiment.json (carttrace.trace.experiment) is written from the row, never guessed at run time. The
shares are copied from config's sandbox.v1 when the round is built, so a manifest keeps them whatever config says
later. Neither enters an episode id: the condition, world, surface and rail that make the row already do (a forced
checkout fault is in the condition's params, so it does).
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

DESIGN: Final = "sandbox-v1"
CHECKOUT_FAULT_PARAM: Final = "checkout_fault"
"""The condition param naming the checkout fault a v1 row is built to meet (carttrace.collect.checkout_faults.KINDS)."""

Layer = Literal["real", "open"]
Kind = Literal["baseline", "mask", "designed_failure"]
SubKind = Literal["full", "leave_one_out", "payment", "checkout", "impossible"]
Source = Literal["persona", "checkout", "task"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class V1Recipe(_Frozen):
    """The v1 design a manifest was built under (ManifestPlan.v1): its layer and shares, as config gave them."""

    design: Literal["sandbox-v1"] = DESIGN
    sandbox_version: Literal["v1"] = "v1"
    spec: str | None = None
    """D96: the locked spec's digest (carttrace.collect.v1_stamp); a round whose inputs differ from the lock is refused."""
    layer: Layer
    rows_per_store: int = Field(ge=1)
    """Qwen rows a store gets (its reference rows, a `reference_share_bp` sample of them, ride beside them)."""
    personas: int = Field(ge=1)
    tasks_per_persona: int = Field(ge=1)
    cells_per_task: int = Field(ge=1)
    baseline_bp: int = Field(ge=0, le=10_000)
    mask_bp: int = Field(ge=0, le=10_000)
    designed_failure_bp: int = Field(ge=0, le=10_000)
    failure_weights: dict[str, int]
    """payment, checkout and impossible: how the designed-failure budget splits."""
    checkout_kinds: dict[str, int]
    order_share_bp: int = Field(default=10_000, ge=0, le=10_000)
    """D98: the share of a store's baseline rows that run end to end through payment; the rest end at a verified
    cart. Designed failures are always order rows."""
    cross_border_bp: int = Field(default=0, ge=0, le=10_000)
    reference_share_bp: int = Field(default=10_000, ge=0, le=10_000)
    """D111: the share of Qwen rows that got a reference twin (10000, every row, on a manifest built before D111)."""
    mask_types: dict[str, list[str]] = Field(default_factory=dict)
    """Category slug to the data types its masks may withhold (empty until the project decides, 2026-10-16)."""
    target: str | None = None
    """The target file the round was drawn toward, with its sha256."""
    round: int = Field(default=0, ge=0)


class RowExperiment(_Frozen):
    """What one v1 row is (ManifestRow.experiment): read by the episode to write experiment.json."""

    layer: Layer
    kind: Kind
    sub_kind: SubKind
    removed: list[str] = Field(default_factory=list)
    """Data types withheld on every surface (a mask row's one type)."""
    injected: Source | None = None
    """Where a designed failure acts: the persona's wallet, checkout completion, or the task itself."""
    injected_kind: str | None = None
    """decline, 3ds_fail, 3ds_abandon, no_instrument; three_ds, payment_decline, reprice, stockout, unshippable;
    impossible."""
    pairs_with: str | None = None
    """A mask row's full-data half: the episode id of the same task's row on the same store, seed, surface, rail and
    mode with nothing withheld."""
    platform: str
    """The platform the store's capture names (a twin: the queue store's, for the record)."""
    platform_exact: bool
    exact: bool
    capture_id: str | None = None
    capture_edition: str | None = None
    captured_at: str | None = None
    capture_complete: bool | None = None
    reference_valid: bool = True
    home_market: str
    cross_border: bool = False

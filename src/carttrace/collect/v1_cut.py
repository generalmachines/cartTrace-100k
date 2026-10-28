"""Cutting a planner manifest to sandbox v1's rows (D96, D98): per store 25 tasks on 4 cells, 90% baseline and 10%
designed failures in one budget, and half the baseline at a verified cart.

Why cut, not plan anew. The planner already makes every row, world and episode id honestly (D63, D66, D72, D80, D87);
the cut only chooses among its rows, forces a checkout fault on some (a condition param, so a new episode id), and turns
half the baseline into cart rows (no rail, no checkout), so every v1 row is one the planner would make and the same
inputs give the same bytes. Each kept row carries `experiment` (carttrace.collect.v1_rows.RowExperiment).

The budget per store: failure rows are designed_failure_bp of the rows, split by weight (impossible, payment,
checkout); payment faults and impossible tasks take whole tasks, rounded by seed; checkout faults are forced on single
rows and take the rest, with a top-up when a task-level source falls short and a trim on a thin store. Mask rows
(mask_bp, 0 in v1)
pair one withheld type with the same task's full-data row.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from carttrace.collect.checkout_faults import KINDS as CHECKOUT_KINDS
from carttrace.collect.exposure import BY_ID
from carttrace.collect.manifest import Manifest, ManifestRow, derived_seed, make_row, seal
from carttrace.collect.payment import THREE_DS_FAULTS, decides_three_ds
from carttrace.collect.ports import RunMode
from carttrace.collect.profiles import RowSurface
from carttrace.collect.v1_rows import CHECKOUT_FAULT_PARAM, RowExperiment, V1Recipe
from carttrace.config.schema import CheckoutFaultRate, CheckoutFaults
from carttrace.core.canon import digest, dumps
from carttrace.core.ids import seeded_rng
from carttrace.trace.schema import Condition, StoreRef

FAILURE_SOURCES: Final = ("payment", "checkout", "impossible")


EXPOSURE_BASELINE: Final = "baseline"


EXPOSURE_ONE: Final = "leave_one_out"


FORCED_SOURCE: Final = "designed failure of sandbox v1 (D96), forced on this row; not a measured rate"


INTENT_TYPES: Final[Mapping[str, str]] = {"policy": "returns.policy", "gift": "offer.gift_options"}
"""A template task's intent to the exposure type its decision turns on, where one does: a policy task asks a return
question first, a gift task wants a gift message. Other template intents turn on options, price or stock, which no
mask withholds (bench/type-experiments: 96% of exposure rows)."""


CART_SURFACES: Final = frozenset({"ucp_mcp", "ucp_rest", "webmcp"})
"""Surfaces a cart row can shop on: the catalog surfaces. An ACP or AP2 cell's cart row shops the store's UCP over MCP
catalog, the companion those rails pay beside."""


def zero_faults(rates: CheckoutFaults) -> CheckoutFaults:
    """The checkout fault table with every rate at 0, sources kept: a v1 row meets a checkout fault only when it is
    built to (its condition names one), never by a draw."""
    out: dict[str, Any] = rates.model_dump()
    for kind in CHECKOUT_KINDS:
        out[kind] = {**out[kind], "rate_bp": 0}
    return CheckoutFaults.model_validate(out)


def forced_faults(rates: CheckoutFaults, kind: str) -> CheckoutFaults:
    """The table one v1 row draws from: its built fault at the whole space, every other kind at 0."""
    out: dict[str, Any] = zero_faults(rates).model_dump()
    out[kind] = CheckoutFaultRate(rate_bp=10_000, assumed=True, source=FORCED_SOURCE).model_dump()
    return CheckoutFaults.model_validate(out)


def forced_kind(condition: Condition) -> str | None:
    kind = condition.params.get(CHECKOUT_FAULT_PARAM)
    return kind if isinstance(kind, str) and kind in CHECKOUT_KINDS else None


Cell = tuple[str, str]


@dataclass(frozen=True, slots=True)
class StoreFacts:
    """What a row's experiment block says of its store, beyond the manifest."""

    platform: str
    platform_exact: bool
    exact: bool
    capture_id: str | None = None
    capture_edition: str | None = None
    captured_at: str | None = None
    capture_complete: bool | None = None


@dataclass(slots=True)
class Budget:
    """One store's rows by kind and source, exact: the failure rows split by weight with the largest remainder, so
    every store gets its sources' shares (impossible 3, payment 4, checkout 3 of 10) and no store swings (final
    review #12; the earlier draw of whole tasks gave 0 to 16 checkout rows a store)."""

    rows: int
    baseline: int
    mask: int
    failure: int
    payment: int
    impossible: int
    checkout: int


@dataclass(slots=True)
class _Store:
    """One store's rows of the base manifest and of its exposure design."""

    store_id: str
    sandbox: list[ManifestRow] = field(default_factory=list)
    exposure: list[ManifestRow] = field(default_factory=list)


def budget(recipe: V1Recipe, store_id: str, seed: int) -> Budget:
    """A store's rows by kind and failure source."""
    rows = recipe.rows_per_store
    failure = round(rows * recipe.designed_failure_bp / 10_000)
    mask = round(rows * recipe.mask_bp / 10_000)
    share = split_failures(failure, recipe.failure_weights)
    return Budget(
        rows=rows,
        baseline=rows - mask - failure,
        mask=mask,
        failure=failure,
        payment=share["payment"],
        impossible=share["impossible"],
        checkout=share["checkout"],
    )


def split_failures(failure: int, weights: Mapping[str, int]) -> dict[str, int]:
    """`failure` rows over the sources by weight, largest remainder first (ties in FAILURE_SOURCES order)."""
    found = {s: weights.get(s, 0) for s in FAILURE_SOURCES}
    total = sum(found.values()) or 1
    exact = {s: failure * w / total for s, w in found.items()}
    share = {s: math.floor(v) for s, v in exact.items()}
    for s in sorted(exact, key=lambda s: (-(exact[s] - share[s]), FAILURE_SOURCES.index(s)))[
        : failure - sum(share.values())
    ]:
        share[s] += 1
    return share


def _fill(
    pool: Sequence[str], rows: int, limits: dict[str, int], cells: int, available: Callable[[str], int]
) -> list[str]:
    """Tasks of `pool`, in order, until they hold `rows` rows: each takes its available cells, at most `cells`, the
    last only what is left; `limits` records each chosen task's rows."""
    chosen: list[str] = []
    left = rows
    for task_id in pool:
        if left <= 0:
            break
        take = min(cells, available(task_id), left)
        if take <= 0:
            continue
        limits[task_id] = take
        chosen.append(task_id)
        left -= take
    return chosen


def _cell(row: ManifestRow) -> Cell:
    return (row.surface.kind if row.surface is not None else "ucp_mcp", row.rail or "")


def _fault_row(row: ManifestRow) -> bool:
    """Whether a row is a persona-fault row whose fault can fire: a 3-D Secure persona fault only on a rail that makes
    a 3-D Secure decision (D119: on a shop_pay row it is never decided, so the negative is void)."""
    fault = row.payment_fault
    return bool(fault) and (fault not in THREE_DS_FAULTS or decides_three_ds(row.rail))


def _sub_kind(row: ManifestRow) -> str:
    return row.experiment.sub_kind if row.experiment is not None else ""


def _persona(row: ManifestRow) -> str | None:
    found = row.condition.params.get("persona")
    return found if isinstance(found, str) else None


def remade(
    row: ManifestRow,
    condition: Condition,
    *,
    mode: RunMode | None = None,
    surface: RowSurface | None = None,
    cart: bool = False,
    repeat: tuple[int, int] | None = None,
) -> ManifestRow:
    """The row under another condition (or mode), its episode id derived again: a forced checkout fault is a new
    episode, and a reference row is the sandbox row's world, task, seed, surface and rail under the reference agent.
    `cart` makes it a cart row (D98): no rail, on `surface` (a catalog surface of its world); `repeat` (the repeat and
    its seed) makes it another run of the same cart journey."""
    if cart:
        return make_row(
            store_id=row.store_id,
            task_id=row.task_id,
            condition=condition,
            profile=row.profile,
            mode=mode or row.mode,
            repeat=repeat[0] if repeat is not None else row.repeat,
            seed=repeat[1] if repeat is not None else row.seed,
            market=row.market,
            envpack=row.envpack,
            handoff=row.handoff,
            surface=surface or row.surface,
            replica_of=row.replica_of,
            platform_pack=row.platform_pack,
            market_context=row.market_context,
        )
    return make_row(
        store_id=row.store_id,
        task_id=row.task_id,
        condition=condition,
        profile=row.profile,
        mode=mode or row.mode,
        repeat=row.repeat,
        seed=row.seed,
        market=row.market,
        envpack=row.envpack,
        handoff=row.handoff,
        surface=row.surface,
        replica_of=row.replica_of,
        platform_pack=row.platform_pack,
        market_context=row.market_context,
        rail=row.rail,
        adopted=row.adopted,
        rail_version=row.rail_version,
        psp=row.psp,
        payment_fault=row.payment_fault,
        surface_adopted=row.surface_adopted,
    )


def decisive_type(task: Any) -> str | None:
    """The exposure type a task's decision turns on: a custom intent's decisive kind's type, else its template
    intent's (INTENT_TYPES), else None."""
    kind = getattr(task, "decisive_kind", None)
    if isinstance(kind, str):
        for entry in BY_ID.values():
            if kind in entry.kinds:
                return str(entry.id)
    return INTENT_TYPES.get(str(getattr(task, "intent", "")))


@dataclass(slots=True)
class Cut:
    """One store's cut: the rows it keeps and what it could not fill."""

    rows: list[ManifestRow]
    budget: Budget
    notes: list[str]


def cut_store(  # noqa: PLR0912, PLR0915 - one pass over a store's rows, in the order the design reads
    recipe: V1Recipe,
    *,
    store_id: str,
    category: str,
    tasks: Mapping[str, Any],
    rows: Sequence[ManifestRow],
    exposure_rows: Sequence[ManifestRow] = (),
    facts: StoreFacts,
    seed: int,
    home_market: str,
    reference: bool = True,
    kind_uses: Counter[str] | None = None,
) -> Cut:
    """The v1 rows of one store, cut from the planner's rows (`rows`: baseline order rows crossed with the cells;
    `exposure_rows`: an exposure design's rows of the same tasks, for the masks). See the module docstring."""
    plan = budget(recipe, store_id, seed)
    notes: list[str] = []
    cells_per = recipe.cells_per_task
    by_task: dict[str, dict[Cell, ManifestRow]] = {}
    for row in rows:
        if row.mode == "sandbox":
            by_task.setdefault(row.task_id, {}).setdefault(_cell(row), row)

    def twin_of(row: ManifestRow) -> ManifestRow | None:
        return remade(row, row.condition, mode="reference") if reference else None

    exp_by_task: dict[str, list[ManifestRow]] = {}
    for row in exposure_rows:
        if row.mode == "sandbox":
            exp_by_task.setdefault(row.task_id, []).append(row)

    order = sorted(by_task, key=lambda t: dumps(["v1-order", seed, store_id, t]))
    rng = seeded_rng("v1-tasks", seed, store_id)
    rng.shuffle(order)
    impossible = [t for t in order if tasks[t].hidden_spec.get("impossible") is True]
    faulted = [t for t in order if t not in impossible and any(_fault_row(r) for r in by_task[t].values())]
    paying = [t for t in order if t not in impossible and t not in faulted]

    wanted = recipe.personas * recipe.tasks_per_persona
    limits: dict[str, int] = {}
    chosen_fault = _fill(
        faulted, plan.payment, limits, cells_per, lambda t: sum(1 for r in by_task[t].values() if _fault_row(r))
    )
    # an impossible task's rows are its cells without a persona fault, the ones `pick` may take (final review #12: a
    # task counted with its fault cells gave 2 impossible rows for 3 and the checkout faults took the third)
    chosen_impossible = _fill(
        impossible,
        plan.impossible,
        limits,
        cells_per,
        lambda t: sum(1 for r in by_task[t].values() if not r.payment_fault),
    )
    for name, chosen, want in (
        ("payment", chosen_fault, plan.payment),
        ("impossible", chosen_impossible, plan.impossible),
    ):
        got = sum(limits[t] for t in chosen)
        if got < want:
            notes.append(f"{store_id}: {got} of {want} {name} rows found; checkout faults take the rest")
    room = wanted - len(chosen_fault) - len(chosen_impossible)
    by_persona: dict[str, list[str]] = {}
    for task_id in paying:
        persona = next((_persona(r) for r in by_task[task_id].values()), None) or ""
        by_persona.setdefault(persona, []).append(task_id)
    chosen_paying: list[str] = []
    # the round robin starts at a persona of the store's own (final review #12: always the first persona by name)
    names = sorted(by_persona)
    turn = int(digest(["v1-persona", seed, store_id], 8), 16) % len(names) if names else 0
    names = names[turn:] + names[:turn]
    while len(chosen_paying) < room and any(by_persona.values()):
        for persona in names:
            if by_persona[persona] and len(chosen_paying) < room:
                chosen_paying.append(by_persona[persona].pop(0))
    if len(chosen_paying) < room:
        notes.append(f"{store_id}: {len(chosen_paying)} of {room} paying tasks found")

    uses: Counter[Cell] = Counter()
    picked: dict[str, list[Cell]] = {}

    def pick(task_id: str, allowed: Sequence[Cell]) -> list[Cell]:
        choice: list[Cell] = []
        pool = list(allowed)
        draw = seeded_rng("v1-cells", seed, store_id, task_id)
        while pool and len(choice) < cells_per:
            least = min(uses[c] for c in pool)
            tied = sorted(c for c in pool if uses[c] == least)
            cell = tied[draw.randrange(len(tied))]
            choice.append(cell)
            uses[cell] += 1
            pool.remove(cell)
        return choice

    for task_id in [*chosen_fault, *chosen_impossible, *chosen_paying]:
        cells = by_task[task_id]
        if task_id in chosen_fault:
            allowed = [c for c, r in cells.items() if _fault_row(r)]
        else:
            allowed = [c for c, r in cells.items() if not r.payment_fault]
        picked[task_id] = pick(task_id, allowed)[: limits.get(task_id, cells_per)]
    # a task-level source that ended part way through a task (3 impossible rows on 4-cell tasks) leaves rows to fill:
    # the next paying tasks take only what is left, so the store keeps its rows
    left = plan.rows - sum(len(c) for c in picked.values())
    for task_id in [t for t in paying if t not in picked]:
        if left <= 0:
            break
        free = [c for c, r in by_task[task_id].items() if not r.payment_fault]
        picked[task_id] = pick(task_id, free)[:left]
        left -= len(picked[task_id])
        chosen_paying.append(task_id)

    out: list[ManifestRow] = []

    def experiment(row: ManifestRow, **fields: Any) -> RowExperiment:
        return RowExperiment(
            layer=recipe.layer,
            # an open twin names only the pack it was served under, never the real store's platform (audit, c25)
            platform=(row.platform_pack or "generic") if recipe.layer == "open" else facts.platform,
            platform_exact=facts.platform_exact,
            exact=facts.exact,
            capture_id=facts.capture_id,
            capture_edition=facts.capture_edition,
            captured_at=facts.captured_at,
            capture_complete=facts.capture_complete,
            reference_valid=getattr(tasks[row.task_id], "intent_kind", None) is None,
            home_market=home_market,
            cross_border=row.market.upper() != home_market.upper(),
            **fields,
        )

    def emit(row: ManifestRow, twin: ManifestRow | None, **fields: Any) -> None:
        out.append(row.model_copy(update={"experiment": experiment(row, **fields)}))
        if twin is not None:
            out.append(twin.model_copy(update={"experiment": experiment(twin, **fields)}))

    # masks: a paying task whose decision turns on a type its category may hide, on the exposure design's cell
    allowed_types = set(recipe.mask_types.get(category, ()))
    masks_left = plan.mask
    masked: dict[str, tuple[ManifestRow, ManifestRow]] = {}
    if allowed_types and masks_left:
        for task_id in chosen_paying:
            if masks_left <= 0:
                break
            if len(picked[task_id]) < cells_per:
                continue  # a partial task has no room for a pair
            wanted_type = decisive_type(tasks[task_id])
            if wanted_type not in allowed_types:
                continue
            found = exp_by_task.get(task_id, [])
            base = next((r for r in found if r.condition.params.get("design") == EXPOSURE_BASELINE), None)
            mask = next(
                (
                    r
                    for r in found
                    if r.condition.params.get("design") == EXPOSURE_ONE
                    and r.condition.params.get("withheld") == [wanted_type]
                    and base is not None
                    and _cell(r) == _cell(base)
                ),
                None,
            )
            if base is None or mask is None:
                continue
            masked[task_id] = (base, mask)
            masks_left -= 1
    if plan.mask and masks_left:
        notes.append(
            f"{store_id}: {plan.mask - masks_left} of {plan.mask} mask rows planned"
            + ("" if allowed_types else f" ({category} has no maskable types yet)")
            + "; the rest run as baseline"
        )

    short = plan.payment + plan.impossible - sum(len(picked[t]) for t in [*chosen_fault, *chosen_impossible])
    checkout_left = plan.checkout + max(0, short)
    kinds = recipe.checkout_kinds
    kinds_used: Counter[str] = kind_uses if kind_uses is not None else Counter()
    # the checkout faults go to paying tasks in a seeded order of the store's own, never the loop's (final review
    # #12: the first persona's tasks took them in every store)
    witnessed = [t for t in chosen_paying if tasks[t].hidden_spec.get("witnesses") and t not in masked]
    fault_tasks = set(sorted(witnessed, key=lambda t: digest(["v1-fault", seed, store_id, t], 12))[: plan.checkout])

    def next_kind(row: ManifestRow) -> str:
        """The kind furthest below its weight's share over the whole round (`kind_uses` is shared by the round's
        stores, so the kinds meet their weights however few faults one store gets), ties by name; three_ds only on a
        rail that makes a 3-D Secure decision (D119: never Shop Pay)."""
        names = sorted(k for k, w in kinds.items() if w > 0) or sorted(CHECKOUT_KINDS)
        if not decides_three_ds(row.rail):
            names = [k for k in names if k not in THREE_DS_FAULTS] or [k for k in CHECKOUT_KINDS if k != "three_ds"]
        weight = sum(kinds.get(k, 0) for k in names) or 1
        total = sum(kinds_used[k] for k in names) + 1
        return min(names, key=lambda k: (kinds_used[k] / total - kinds.get(k, 0) / weight, k))

    for task_id in [*chosen_fault, *chosen_impossible, *chosen_paying]:
        mine = picked[task_id]
        rows_of = by_task[task_id]
        if task_id in chosen_fault:
            for cell in mine:
                row = rows_of[cell]
                twin = twin_of(row)
                emit(
                    row,
                    twin,
                    kind="designed_failure",
                    sub_kind="payment",
                    injected="persona",
                    injected_kind=row.payment_fault,
                )
            continue
        if task_id in chosen_impossible:
            for cell in mine:
                row = rows_of[cell]
                emit(
                    row,
                    twin_of(row),
                    kind="designed_failure",
                    sub_kind="impossible",
                    injected="task",
                    injected_kind="impossible",
                )
            continue
        pair = masked.get(task_id)
        used = list(mine)
        if pair is not None:
            base, mask = pair
            mask_cell = _cell(base)
            others = [c for c in used if c != mask_cell][: max(0, len(used) - 2)]
            base_twin = twin_of(base)
            emit(base, base_twin, kind="baseline", sub_kind="full")
            mask_twin = twin_of(mask)
            withheld = [str(t) for t in mask.condition.params.get("withheld") or ()]
            out.append(
                mask.model_copy(
                    update={
                        "experiment": experiment(
                            mask,
                            kind="mask",
                            sub_kind="leave_one_out",
                            removed=withheld,
                            pairs_with=base.episode_id,
                        )
                    }
                )
            )
            if mask_twin is not None and base_twin is not None:
                out.append(
                    mask_twin.model_copy(
                        update={
                            "experiment": experiment(
                                mask_twin,
                                kind="mask",
                                sub_kind="leave_one_out",
                                removed=withheld,
                                pairs_with=base_twin.episode_id,
                            )
                        }
                    )
                )
            used = others
        for at, cell in enumerate(used):
            row = rows_of[cell]
            twin = twin_of(row)
            if checkout_left > 0 and at == len(used) - 1 and task_id in fault_tasks:
                kind = next_kind(row)
                kinds_used[kind] += 1
                checkout_left -= 1
                condition = Condition(
                    name=row.condition.name, params={**row.condition.params, CHECKOUT_FAULT_PARAM: kind}
                )
                emit(
                    remade(row, condition),
                    remade(twin, condition) if twin is not None else None,
                    kind="designed_failure",
                    sub_kind="checkout",
                    injected="checkout",
                    injected_kind=kind,
                )
                continue
            emit(row, twin, kind="baseline", sub_kind="full")
    # the budget is exact: failure rows a task-level source could not fill (a fault task with fewer cells, no
    # impossible task in the store) become checkout faults on baseline rows, one a task, last cell first
    built = sum(1 for r in out if r.mode == "sandbox" and r.experiment and r.experiment.kind == "designed_failure")
    rows_now = sum(1 for r in out if r.mode == "sandbox")
    wanted_failures = min(plan.failure, round(rows_now * recipe.designed_failure_bp / 10_000))
    deficit = wanted_failures - built
    paired = {r.experiment.pairs_with for r in out if r.experiment and r.experiment.pairs_with}
    faulted_tasks = {r.task_id for r in out if r.experiment and r.experiment.sub_kind == "checkout"}
    for at in sorted(range(len(out)), key=lambda i: digest(["v1-deficit", seed, store_id, out[i].episode_id], 12)):
        if deficit <= 0:
            break
        row = out[at]
        exp = row.experiment
        if (
            row.mode != "sandbox"
            or exp is None
            or exp.kind != "baseline"
            or row.episode_id in paired
            or row.task_id in faulted_tasks
            or row.condition.name != "baseline"
            or not tasks[row.task_id].hidden_spec.get("witnesses")
        ):
            continue
        kind = next_kind(row)
        kinds_used[kind] += 1
        condition = Condition(name=row.condition.name, params={**row.condition.params, CHECKOUT_FAULT_PARAM: kind})
        update = exp.model_copy(
            update={"kind": "designed_failure", "sub_kind": "checkout", "injected": "checkout", "injected_kind": kind}
        )
        out[at] = remade(row, condition).model_copy(update={"experiment": update})
        twin_at = at + 1
        if twin_at < len(out) and out[twin_at].mode == "reference" and out[twin_at].task_id == row.task_id:
            out[twin_at] = remade(out[twin_at], condition).model_copy(update={"experiment": update})
        faulted_tasks.add(row.task_id)
        checkout_left -= 1
        deficit -= 1
    if deficit < 0:
        # a store with fewer rows than the design (a thin catalog) keeps its failures to the share of what it has, split
        # by the same weights (final review #12): each source's rows over its share are left out, last first, and
        # where a source is short of its share the rest go impossible first, then payment, then checkout
        rank = {"impossible": 0, "payment": 1, "checkout": 2}
        failing = [
            i
            for i, r in enumerate(out)
            if r.mode == "sandbox" and r.experiment and r.experiment.kind == "designed_failure"
        ]
        target = split_failures(wanted_failures, recipe.failure_weights)
        over: list[int] = []
        for source in FAILURE_SOURCES:
            of_source = [i for i in failing if _sub_kind(out[i]) == source]
            over += sorted(of_source, reverse=True)[: max(0, len(of_source) - target[source])]
        rest = sorted((i for i in failing if i not in over), key=lambda i: (rank.get(_sub_kind(out[i]), 3), -i))
        surplus = (over + rest)[:-deficit]
        gone = {out[i].episode_id for i in surplus}
        twins_gone = {i + 1 for i in surplus if i + 1 < len(out) and out[i + 1].mode == "reference"}
        out = [r for i, r in enumerate(out) if r.episode_id not in gone and i not in twins_gone]
        deficit = 0
    if deficit != 0:
        placed = wanted_failures - deficit
        notes.append(f"{store_id}: {placed} designed-failure rows for {wanted_failures} wanted of {rows_now} rows")
    out, even_notes = even_baseline(out, recipe, store_id=store_id, seed=seed)
    notes.extend(even_notes)
    out, cart_notes = cart_split(out, recipe, store_id=store_id, seed=seed, tasks=tasks)
    notes.extend(cart_notes)
    qwen = sum(1 for r in out if r.mode == "sandbox")
    if qwen < plan.rows:
        notes.append(f"{store_id}: {qwen} of {plan.rows} v1 rows (fewer tasks or cells than the design asks)")
    return Cut(rows=out, budget=plan, notes=notes)


def cart_split(
    rows: Sequence[ManifestRow], recipe: V1Recipe, *, store_id: str, seed: int, tasks: Mapping[str, Any] | None = None
) -> tuple[list[ManifestRow], list[str]]:
    """D98: a store's full-data rows split by code, never by a draw, and exactly: floor(n x cart
    share) of its n baseline rows go to the cart, the rest to the order. Each persona takes its own half of its rows
    (the odd rows' extra one dealt over the personas in a stable order), so personas are spread evenly; within a
    persona its tasks are dealt cart side first (`_dealt`: each intent's tasks alternate store-wide, so every intent
    sends about half its tasks to the cart) and taken while they fit (`exact_rows`), so intents and the cells the tasks
    drew are spread too.
    At most one task a persona is split between the boundaries. A cart row loses its rail and checkout and shops its
    cell's catalog surface (UCP over MCP for an ACP or AP2 cell); designed-failure rows and a mask pair stay order
    rows. The same inputs give the same split, byte for byte."""
    out = list(rows)
    by_task: dict[str, list[int]] = {}
    for i in _splittable(out):
        by_task.setdefault(out[i].task_id, []).append(i)
    wanted = sum(len(v) for v in by_task.values()) * (10_000 - recipe.order_share_bp) // 10_000
    if not wanted:
        return out, []
    dealt = _dealt(by_task, out, tasks or {}, store_id=store_id, seed=seed)
    per = {p: sum(len(by_task[t]) for t in ts) for p, ts in dealt.items()}
    share = {p: n * (10_000 - recipe.order_share_bp) // 10_000 for p, n in per.items()}
    # the rows the floors leave go to personas whose own share has a remainder, in a stable order of the store's own
    extra = sorted(
        dealt,
        key=lambda p: (
            per[p] * (10_000 - recipe.order_share_bp) % 10_000 == 0,
            digest(["v1-cart", store_id, seed, p], 12),
        ),
    )
    for persona in extra[: wanted - sum(share.values())]:
        share[persona] += 1
    chosen: dict[str, int] = {}
    for persona, order in sorted(dealt.items()):
        chosen.update(exact_rows({t: len(by_task[t]) for t in order}, share[persona]))
    catalog = next((r.surface for r in out if r.surface is not None and r.surface.kind == "ucp_mcp"), None)
    # each world's catalog surfaces, as its own rows name them (the version is the world's)
    catalogs: dict[str | None, dict[str, RowSurface]] = {}
    for r in out:
        if r.surface is not None and r.surface.kind in CART_SURFACES:
            catalogs.setdefault(r.envpack, {}).setdefault(r.surface.kind, r.surface)
    skipped = 0
    for task_id in sorted(chosen):
        used: Counter[tuple[str | None, str]] = Counter()
        for i in by_task[task_id][: chosen[task_id]]:
            row = out[i]
            surface = row.surface if row.surface is not None and row.surface.kind in CART_SURFACES else catalog
            if surface is None:
                skipped += 1
                continue
            # cells that share one world (a stand-in's one spec, carttrace.collect.store_spec) would make two cart rows on
            # one surface one episode: such a row takes a catalog surface of its world the task has not shopped yet,
            # else runs the journey again as a repeat
            own = catalogs.get(row.envpack, {})
            if used[(row.envpack, surface.kind)]:
                surface = next((own[k] for k in sorted(own) if not used[(row.envpack, k)]), surface)
            again = used[(row.envpack, surface.kind)]
            used[(row.envpack, surface.kind)] += 1
            repeat = (
                (row.repeat + again, derived_seed("row", seed, store_id, task_id, row.repeat + again))
                if again
                else None
            )
            params = {k: v for k, v in row.condition.params.items() if k != "checkout"}
            condition = Condition(name=row.condition.name, params=params)
            out[i] = remade(row, condition, surface=surface, cart=True, repeat=repeat).model_copy(
                update={"experiment": row.experiment}
            )
            twin = i + 1
            if twin < len(out) and out[twin].mode == "reference" and out[twin].task_id == row.task_id:
                cart = remade(out[twin], condition, surface=surface, cart=True, repeat=repeat)
                out[twin] = cart.model_copy(update={"experiment": out[twin].experiment})
    notes = [f"{store_id}: {skipped} cart rows left as order rows: no UCP over MCP catalog to shop"] if skipped else []
    return out, notes


def _splittable(rows: Sequence[ManifestRow]) -> list[int]:
    """The full-data rows the cart split deals: baseline order rows of the store's own Qwen arm, no mask pair."""
    paired = {r.experiment.pairs_with for r in rows if r.experiment and r.experiment.pairs_with}
    return [
        i
        for i, r in enumerate(rows)
        if r.mode == "sandbox"
        and r.experiment is not None
        and r.experiment.kind == "baseline"
        and r.condition.params.get("checkout") is True
        and r.episode_id not in paired
    ]


def even_baseline(
    rows: Sequence[ManifestRow], recipe: V1Recipe, *, store_id: str, seed: int
) -> tuple[list[ManifestRow], list[str]]:
    """A thin store whose full-data rows cannot split exactly at the order share (an odd count at 50/50) leaves out
    the fewest rows that make them split exactly, with their reference twins, chosen by a stable hash of the store's own."""
    out = list(rows)
    base = _splittable(out)
    cart_bp = 10_000 - recipe.order_share_bp
    drop = 0
    while drop < len(base) and ((len(base) - drop) * cart_bp) % 10_000:
        drop += 1
    if not drop or drop == len(base):
        return out, []
    gone = sorted(base, key=lambda i: digest(["v1-even", seed, store_id, out[i].episode_id], 12))[:drop]
    twins = {i + 1 for i in gone if i + 1 < len(out) and out[i + 1].mode == "reference"}
    kept = [r for i, r in enumerate(out) if i not in set(gone) | twins]
    return kept, [f"{store_id}: {drop} full-data row(s) left out so the rest split exactly at the order share"]


def exact_rows(sizes: Mapping[str, int], wanted: int) -> dict[str, int]:
    """Task to how many of its rows go to the cart, so exactly `wanted` rows do. Whole tasks in `sizes`' order (the deal) while they fit; the gap left, smaller than a
    task, is closed by the first task left out of exactly its size, else by part of the next task in the deal, so the
    spread over personas and intents stays the deal's."""
    chosen: dict[str, int] = {}
    left = wanted
    order = [t for t, n in sizes.items() if n > 0]
    for task_id in order:
        if sizes[task_id] > left:
            break
        chosen[task_id] = sizes[task_id]
        left -= sizes[task_id]
    if left:
        rest = [t for t in order if t not in chosen]
        exact = next((t for t in rest if sizes[t] == left), None)
        if exact is None and rest:
            chosen[rest[0]] = left
        elif exact is not None:
            chosen[exact] = left
    return chosen


def _dealt(
    by_task: Mapping[str, Sequence[int]],
    rows: Sequence[ManifestRow],
    tasks: Mapping[str, Any],
    *,
    store_id: str,
    seed: int,
) -> dict[str, list[str]]:
    """Each persona's tasks in their dealing order, the cart's first. Store-wide, each intent's tasks in hash order
    alternate between the cart and the order, so every intent sends half its tasks to the cart (an intent with an odd
    count rounds up and the next odd one down, in turn); each persona then lists its cart-side tasks first, intents
    in turn. (2026-10-17: the earlier deal took every other task of a persona's round over intents by name, and with
    about five tasks a persona every persona's first intent by name went to the cart: budget_max took all its tasks.)"""
    ranked = sorted(by_task, key=lambda t: digest(["v1-boundary", store_id, t, seed], 12))
    by_intent: dict[str, list[str]] = {}
    for task_id in ranked:
        by_intent.setdefault(str(getattr(tasks.get(task_id), "intent", "")), []).append(task_id)
    cart_side: set[str] = set()
    odd = int(digest(["v1-boundary-odd", store_id, seed], 8), 16) % 2
    for intent in sorted(by_intent):
        mine = by_intent[intent]
        start = odd if len(mine) % 2 else 0
        cart_side.update(mine[start::2])
        if len(mine) % 2:
            odd = 1 - odd
    lanes: dict[str, dict[bool, dict[str, list[str]]]] = {}
    for task_id in ranked:
        persona = _persona(rows[by_task[task_id][0]]) or ""
        intent = str(getattr(tasks.get(task_id), "intent", ""))
        side = lanes.setdefault(persona, {}).setdefault(task_id not in cart_side, {})
        side.setdefault(intent, []).append(task_id)
    per_persona: dict[str, list[str]] = {}
    for persona, sides in lanes.items():
        mixed: list[str] = []
        for later in (False, True):
            # intents in an order of the persona's own, so no intent by name leads every persona's deal
            side = sides.get(later, {})
            names = sorted(side, key=lambda i: digest(["v1-intent-turn", store_id, seed, persona, i], 8))
            queues = [list(side[i]) for i in names]
            while any(queues):
                mixed += [q.pop(0) for q in queues if q]
        per_persona[persona] = mixed
    return per_persona


def cut_manifest(
    recipe: V1Recipe,
    base: Manifest,
    *,
    exposure: Manifest | None = None,
    facts: Mapping[str, StoreFacts],
    checkout_faults: CheckoutFaults,
    reference: bool = True,
) -> Manifest:
    """The v1 manifest of `base`'s stores: each store cut by `cut_store`, the tables both manifests hold kept where a
    kept row uses them, the plan `base`'s with the recipe and the zeroed checkout fault table on it."""
    rows: list[ManifestRow] = []
    notes = list(base.notes)
    recipe_reference = reference
    kind_uses: Counter[str] = Counter()
    by_store: dict[str, _Store] = {}
    for row in base.rows:
        by_store.setdefault(row.store_id, _Store(row.store_id)).sandbox.append(row)
    for row in exposure.rows if exposure is not None else ():
        by_store.setdefault(row.store_id, _Store(row.store_id)).exposure.append(row)
    tasks: dict[str, dict[str, Any]] = {}
    for sid, table in base.tasks.items():
        tasks[sid] = dict(table)
    for sid, table in (exposure.tasks if exposure is not None else {}).items():
        for tid, task in table.items():
            tasks.setdefault(sid, {}).setdefault(tid, task)
    for store_id in sorted(by_store, key=lambda s: list(base.stores).index(s) if s in base.stores else 0):
        part = by_store[store_id]
        store = base.stores.get(store_id) or (exposure.stores[store_id] if exposure is not None else None)
        if store is None or store_id not in facts:
            continue
        cut = cut_store(
            recipe,
            store_id=store_id,
            category=store.category,
            tasks=tasks.get(store_id, {}),
            rows=part.sandbox,
            exposure_rows=part.exposure,
            facts=facts[store_id],
            seed=base.plan.seed,
            home_market=store.market or "",
            reference=recipe_reference,
            kind_uses=kind_uses,
        )
        rows.extend(cut.rows)
        notes.extend(cut.notes)
    if reference and recipe.reference_share_bp < 10_000:
        categories = {sid: ref.category for sid, ref in base.stores.items()}
        if exposure is not None:
            categories = {**{sid: ref.category for sid, ref in exposure.stores.items()}, **categories}
        rows = sample_references(rows, recipe.reference_share_bp, seed=base.plan.seed, categories=categories)
    worlds = {**(exposure.worlds if exposure is not None else {}), **base.worlds}
    profiles = {**(exposure.profiles if exposure is not None else {}), **base.profiles}
    used_worlds = {r.envpack for r in rows if r.envpack}
    used_profiles = {r.profile for r in rows} | {worlds[e].profile for e in used_worlds}
    used_tasks = {(r.store_id, r.task_id) for r in rows}
    used_stores = {r.store_id for r in rows}
    modes: list[RunMode] = ["sandbox", "reference"] if reference else ["sandbox"]
    update = {"v1": recipe, "checkout_faults": zero_faults(checkout_faults), "modes": modes}
    plan = base.plan.model_copy(update=update)
    stores = {**(exposure.stores if exposure is not None else {}), **base.stores}
    if recipe.layer == "open":
        stores = {sid: open_identity(ref) for sid, ref in stores.items()}
    return seal(
        plan=plan.model_dump(mode="json"),
        sources=dict(base.sources),
        stores={sid: ref.model_dump(mode="json") for sid, ref in sorted(stores.items()) if sid in used_stores},
        tasks={
            sid: {tid: t.model_dump(mode="json") for tid, t in sorted(table.items()) if (sid, tid) in used_tasks}
            for sid, table in sorted(tasks.items())
            if sid in used_stores
        },
        profiles={k: p.model_dump(mode="json") for k, p in sorted(profiles.items()) if k in used_profiles},
        worlds={e: w.model_dump(mode="json") for e, w in sorted(worlds.items()) if e in used_worlds},
        rows=[r.model_dump(mode="json") for r in rows],
        notes=list(dict.fromkeys(notes)),
        coverage={},
        served={},
    )


REFERENCE_AXES: Final = ("category", "boundary", "kind", "cell", "persona")
"""What the reference sample is spread over (D111)."""


def reference_axes(row: ManifestRow, category: str) -> dict[str, str]:
    """A Qwen row's place on each axis the reference sample is spread over: its store's category, its boundary
    (order or cart), its kind (baseline, or negative for a designed failure; a mask row is its own kind), its surface x
    rail cell and its persona."""
    exp = row.experiment
    kind = exp.kind if exp is not None else "baseline"
    surface, rail = _cell(row)
    return {
        "category": category,
        "boundary": "order" if row.condition.params.get("checkout") is True else "cart",
        "kind": "negative" if kind == "designed_failure" else kind,
        "cell": f"{surface}/{rail or '-'}",
        "persona": _persona(row) or "",
    }


def twin_index(rows: Sequence[ManifestRow]) -> dict[int, int]:
    """Each Qwen row's position to its reference twin's: the cut writes a twin right after its row (same task)."""
    out: dict[int, int] = {}
    for i, row in enumerate(rows):
        nxt = i + 1
        if (
            row.mode == "sandbox"
            and nxt < len(rows)
            and rows[nxt].mode == "reference"
            and rows[nxt].task_id == row.task_id
            and rows[nxt].store_id == row.store_id
        ):
            out[i] = nxt
    return out


def sample_references(
    rows: Sequence[ManifestRow], share_bp: int, *, seed: int, categories: Mapping[str, str]
) -> list[ManifestRow]:
    """D111: one reference row per 10 Qwen rows, not one for each. `rows` holds every Qwen row
    with its twin right after it (`cut_store`); the twins of the rows not sampled are left out.

    How many: store by store in the manifest's order, the running total rounded, so each store gets
    round(n x share) give or take the carry (exactly 10 of a 100-row store at 1000 bp) and the manifest exactly its
    share of all its Qwen rows. Which: one at a time, the store's row whose axes (`reference_axes`) are furthest
    below their share so far, over the whole manifest: each candidate scores the sum over the axes of (sampled + 1/2)
    / rows of its value on that axis (Sainte-Lague's divisor, so a 4% kind gets 4% of the sample, not one a store),
    lowest first, ties by a hash of the seed and the episode id. A mask pair's halves go together (the mask row is not
    a candidate; it rides with its full-data row). The same rows and seed give the same sample, byte for byte."""
    out = list(rows)
    twins = twin_index(out)
    paired: dict[str, int] = {}
    for i, row in enumerate(out):
        if row.experiment is not None and row.experiment.pairs_with is not None and row.mode == "sandbox":
            paired[row.experiment.pairs_with] = i
    by_store: dict[str, list[int]] = {}
    for i in twins:
        exp = out[i].experiment
        if exp is not None and exp.pairs_with is not None:
            continue
        by_store.setdefault(out[i].store_id, []).append(i)
    seen: Counter[tuple[str, str]] = Counter()
    sampled: Counter[tuple[str, str]] = Counter()
    axes = {i: reference_axes(out[i], categories.get(out[i].store_id, "")) for i in twins}
    keep: set[int] = set()
    total_rows = 0
    total_kept = 0

    def score(i: int) -> float:
        return sum((sampled[(a, v)] + 0.5) / seen[(a, v)] for a, v in axes[i].items())

    for store_id in dict.fromkeys(out[i].store_id for i in sorted(twins)):
        mine = by_store.get(store_id, [])
        for i in mine:
            for axis, value in axes[i].items():
                seen[(axis, value)] += 1
        total_rows += len(mine)
        wanted = (total_rows * share_bp + 5_000) // 10_000 - total_kept
        pool = sorted(mine, key=lambda i: digest(["v1-reference", seed, out[i].episode_id], 12))
        while wanted > 0 and pool:
            best = min(pool, key=score)  # min keeps the first of a tie: the hash order
            pool.remove(best)
            keep.add(best)
            for axis, value in axes[best].items():
                sampled[(axis, value)] += 1
            wanted -= 1
            total_kept += 1
    # a mask row's twin goes with its full-data row's
    for i in sorted(keep):
        mask = paired.get(out[i].episode_id)
        if mask is not None and mask in twins:
            keep.add(mask)
    drop = {twins[i] for i in twins if i not in keep}
    return [r for i, r in enumerate(out) if i not in drop]


def open_identity(ref: StoreRef) -> StoreRef:
    """An open twin's store as its bundles name it: the queue id, category and market it stands in for, under a
    reserved host and no platform, so no real domain or platform reaches a publishable bundle (audit of the c25 build,
    2026-10-16: the twin of store-001 carried store-001.example and sfcc in every header and the agent's context)."""
    return ref.model_copy(update={"domain": f"{ref.store_id.lower()}.open.sandbox.invalid", "platform": None})

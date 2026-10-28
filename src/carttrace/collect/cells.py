"""The surface x rail matrix of sandbox order rows (D89): which (surface, payment rail) cells are real, and how a plan
draws a balanced share of rows into each of them.

Why. AP2, ACP, WebMCP, UCP REST, UCP MCP and Shop Pay must all be measured, "this is very
important data". The loop's rows that day were ucp_mcp 59%, ap2 17%, acp_rest 17%, webmcp 7% and ucp_rest 0.2% of
surfaces: an exposure task took the plan's rails in turn and then the rail's surfaces in turn, and a UCP binding the
store's capture did not record (almost every replica records UCP over MCP alone) was left out, so UCP REST rows were
planned only on the few stores that recorded it. Shop Pay was not a loop rail at all (D73 left it to the project).

The cells are the rails' own surfaces (`carttrace.collect.rails.RAILS`), nothing else:
- ap2 with ap2_mandates; acp_rest with acp_delegated (each rail's own surface);
- ucp_mcp and ucp_rest with ucp_psp, mpp and shop_pay;
- webmcp with ucp_psp and shop_pay: the storefront's place_order pays through the store's UCP handlers, Shop Pay
  among them (D73), and opens no MPP payment surface beside it.

A plan that draws cells (`draw_cells`) gives each order task `cells_per_task` cells, the ones this store's rows have
used least so far among the cells its persona's wallet can pay on (D80), ties broken by seed. Every store evens its
own rows over the cells, so a round, which is the sum of its stores, holds about rows / cells in each, whatever the
stores declare: a cell's floor in a round is the sum over its stores of their rows over the cells they can serve,
rounded down. What the store itself serves is kept on every row (`adopted` for the rail, D66; `surface_adopted` for
the surface), so the analysis can read store-declared and adopted cells apart.

The draw is public (`offered_cells`, `draw_cells`, `CellDraw`) so any planner, the manifest's or another, asks it
per task or per row instead of reimplementing it: offer the cells a world can carry, say which rails the row's
persona can pay on, and count each row it plans.

Shop Pay is real only on Shopify: the shop_pay rail is planned under the shopify platform pack, or on the store's own
platform when that is Shopify; under WooCommerce, Magento and the other packs no Shop Pay handler exists to adopt.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol

from carttrace.collect.rails import RAILS
from carttrace.core.ids import seeded_rng

DEFAULT_KIND: Final = "ucp_mcp"
"""The surface a row that names none runs on (UCP over MCP)."""

Cell = tuple[str, str]
"""(surface kind, payment rail)."""

CELLS: Final[tuple[Cell, ...]] = tuple(
    (kind, name)
    for kind in ("ucp_mcp", "ucp_rest", "webmcp", "acp_rest", "ap2")
    for name in RAILS
    if kind in RAILS[name].surfaces
)
"""Every valid cell, in canonical order: by surface, then by rail."""
CELL_SET: Final = frozenset(CELLS)

SHOP_PAY_PLATFORM: Final = "shopify"
"""The one platform whose stores can declare the Shop Pay handler (dev.shopify.shop_pay)."""


class _RowLike(Protocol):
    @property
    def rail(self) -> str | None: ...

    @property
    def surface(self) -> object: ...


def cell_of(kind: str | None, rail_name: str) -> Cell:
    """The cell of a row on surface `kind` (None: UCP over MCP) paid over `rail_name`."""
    return (kind or DEFAULT_KIND, rail_name)


def valid_cell(kind: str | None, rail_name: str) -> bool:
    return cell_of(kind, rail_name) in CELL_SET


def row_cell(row: _RowLike) -> Cell | None:
    """A manifest row's cell, or None for a row with no rail."""
    if row.rail is None:
        return None
    kind = getattr(row.surface, "kind", None)
    return cell_of(kind if isinstance(kind, str) else None, row.rail)


def cell_counts(rows: Iterable[_RowLike]) -> dict[Cell, int]:
    """Rows per valid cell, every cell named (zero where none), in canonical order; rows with no rail are left out."""
    found = Counter(cell for row in rows if (cell := row_cell(row)) is not None)
    return {cell: found.get(cell, 0) for cell in CELLS}


def label(cell: Cell) -> str:
    return f"{cell[0]} x {cell[1]}"


def shop_pay_platform(pack: str | None, store_platform: str | None) -> bool:
    """Whether a world under platform pack `pack` (None: the store's own platform) can carry the Shop Pay handler."""
    return (pack or store_platform or "").lower() == SHOP_PAY_PLATFORM


def offered_cells(
    rails: Iterable[str],
    surfaces: Iterable[str | None],
    *,
    pack: str | None = None,
    store_platform: str | None = None,
) -> tuple[Cell, ...]:
    """The valid cells a world can carry, in canonical order: the plan's `rails` on the plan's `surfaces` (None: UCP
    over MCP), a rail's own surface (acp_rest, ap2) whether named or not, and Shop Pay only on a Shopify world
    (`pack`, else the store's own platform)."""
    names = set(rails)
    kinds = {kind or DEFAULT_KIND for kind in surfaces} or {DEFAULT_KIND}
    if not shop_pay_platform(pack, store_platform):
        names -= {n for n in names if n in RAILS and RAILS[n].handler}
    return tuple(c for c in CELLS if c[1] in names and (c[0] in kinds or RAILS[c[1]].own_surface))


def draw_cells(
    uses: Mapping[Cell, int],
    offered: Sequence[Cell],
    *,
    seed: object,
    key: Sequence[object] = (),
    pays: Callable[[str], bool] | None = None,
    fault: bool = False,
    count: int = 1,
) -> tuple[Cell, ...]:
    """`count` distinct cells of `offered` for one task or row: among the cells whose rail the persona can pay on
    (`pays`, D80), the ones used least so far (`uses`), ties drawn by `seed` and `key` (the store, the task, the
    repeat). A paying row no offered cell fits gets none; a designed-fault row (`fault`) no cell fits may take any,
    which makes it a planned fault row. Pure: the caller counts the rows it plans (`CellDraw`)."""
    fitting = [c for c in offered if c in CELL_SET and (pays is None or pays(c[1]))]
    pool = fitting or ([c for c in offered if c in CELL_SET] if fault else [])
    rng = seeded_rng("cells", seed, *key)
    picked: list[Cell] = []
    for _ in range(min(count, len(pool))):
        left = [c for c in pool if c not in picked]
        least = min(uses.get(c, 0) for c in left)
        tied = [c for c in left if uses.get(c, 0) == least]
        picked.append(tied[rng.randrange(len(tied))])
    return tuple(picked)


@dataclass(slots=True)
class CellDraw:
    """One quota's state (a store's rows, or a round's): rows counted per cell, and each task's drawn cells, kept so a
    task's other rows (its reference rows, its repeats) run on the cells its first rows drew."""

    seed: object
    uses: MutableMapping[Cell, int] = field(default_factory=dict)
    drawn: dict[str, tuple[Cell, ...]] = field(default_factory=dict)

    def draw(
        self,
        task: str,
        offered: Sequence[Cell],
        *,
        key: Sequence[object] = (),
        pays: Callable[[str], bool] | None = None,
        fault: bool = False,
        count: int = 1,
    ) -> tuple[Cell, ...]:
        """The task's cells: drawn the first time it is asked (`draw_cells`), the same ones after."""
        if task not in self.drawn:
            found = draw_cells(
                self.uses, offered, seed=self.seed, key=(*key, task), pays=pays, fault=fault, count=count
            )
            self.drawn[task] = found
        return self.drawn[task]

    def count(self, cell: Cell, rows: int = 1) -> None:
        """`rows` planned rows on `cell`, for the next draw."""
        self.uses[cell] = self.uses.get(cell, 0) + rows

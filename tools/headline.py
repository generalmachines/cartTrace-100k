"""Recompute CartTrace-100k's headline numbers from the released `messages` tables.

Every number is read from the rows themselves: no labels are re-derived and nothing is re-judged. Success and grounded
success are the release's own `success` and `grounded_success` columns; the 95% intervals resample whole stand-in
stores (`store_id`), as the paper's intervals do; with 1,000 draws an interval's ends can differ from the paper's by 0.1.

    uv run --extra dataset python tools/headline.py --dataset dataset
    uv run --extra dataset python tools/headline.py --dataset dataset --by category
    uv run --extra dataset python tools/headline.py --dataset dataset --json

`--dataset` is the folder that holds `data/messages/` (the layout `hf download` writes). Only a few columns are read,
so the full table set is not needed: `--include "data/messages/*"` is enough.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

COLUMNS = ["episode_id", "store_id", "category", "boundary", "kind", "sub_kind", "end_reason", "success",
           "grounded_success", "surface", "rail", "intent", "platform_pack", "labels_version"]
BY = ("category", "intent", "surface", "rail", "boundary", "platform_pack")


def load(dataset: Path) -> list[dict]:
    try:
        import pyarrow.dataset as ds
    except ImportError:
        sys.exit("pyarrow is missing: run with `uv run --extra dataset ...`")
    root = dataset / "data" / "messages"
    if not root.is_dir():
        sys.exit(f"no messages tables under {root}; download them with "
                 "`hf download generalmachines/CartTrace-100k --repo-type dataset --include \"data/messages/*\" --local-dir <dir>`")
    table = ds.dataset(str(root), format="parquet").to_table(columns=COLUMNS)
    return table.to_pylist()


def rate(rows: list[dict], field: str) -> float:
    return 100.0 * sum(bool(r[field]) for r in rows) / len(rows) if rows else float("nan")


def store_interval(rows: list[dict], field: str, draws: int, seed: int) -> tuple[float, float]:
    """95% interval of a rate when whole stores are resampled with replacement."""
    by_store: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        tally = by_store[r["store_id"]]
        tally[0] += bool(r[field])
        tally[1] += 1
    stores = list(by_store.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        hits = total = 0
        for _ in range(len(stores)):
            h, n = stores[rng.randrange(len(stores))]
            hits += h
            total += n
        estimates.append(100.0 * hits / total)
    estimates.sort()
    return estimates[int(0.025 * draws)], estimates[int(0.975 * draws) - 1]


def summary(rows: list[dict], draws: int, seed: int) -> dict:
    full = [r for r in rows if r["kind"] == "baseline"]
    out = {
        "journeys": len(rows),
        "stores": len({r["store_id"] for r in rows}),
        "categories": len({r["category"] for r in rows}),
        "labels_version": sorted({r["labels_version"] for r in rows}),
        "by_kind": dict(Counter(r["kind"] for r in rows)),
        "designed_failures": dict(Counter(r["sub_kind"] for r in rows if r["kind"] != "baseline")),
        "boundary": dict(Counter(r["boundary"] for r in rows)),
        "paid_orders": sum(r["end_reason"] == "order_placed" for r in rows),
        "end_reason": dict(Counter(r["end_reason"] for r in rows)),
    }
    for name, subset in (("all", rows), ("full_data", full)):
        block = {"journeys": len(subset)}
        for field in ("success", "grounded_success"):
            lo, hi = store_interval(subset, field, draws, seed) if draws else (None, None)
            block[field] = {"pct": round(rate(subset, field), 1),
                            "ci95": [round(lo, 1), round(hi, 1)] if lo is not None else None}
        out[name] = block
    return out


def breakdown(rows: list[dict], key: str) -> list[tuple[str, int, float, float]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    table = [(k, len(v), rate(v, "success"), rate(v, "grounded_success")) for k, v in groups.items()]
    return sorted(table, key=lambda t: -t[2])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", type=Path, required=True, help="folder holding data/messages/")
    p.add_argument("--by", choices=BY, help="also break success down by this column")
    p.add_argument("--full-data", action="store_true", help="breakdown over full-data journeys only (kind=baseline)")
    p.add_argument("--draws", type=int, default=1000, help="store resamples for the intervals (0 to skip)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")
    a = p.parse_args()

    rows = load(a.dataset)
    s = summary(rows, a.draws, a.seed)
    if a.by:
        subset = [r for r in rows if r["kind"] == "baseline"] if a.full_data else rows
        s["by_" + a.by] = [{"value": k, "journeys": n, "success": round(x, 1), "grounded_success": round(g, 1)}
                           for k, n, x, g in breakdown(subset, a.by)]
    if a.json:
        print(json.dumps(s, indent=2))
        return

    def line(label: str, value: object) -> None:
        print(f"{label:<31}{value}")

    def pct(block: dict) -> str:
        ci = block["ci95"]
        return f"{block['pct']:.1f}%" + (f"  [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "")

    line("journeys", f"{s['journeys']:,}")
    line("stand-in stores", f"{s['stores']:,}")
    line("categories", s["categories"])
    line("labels", ", ".join(s["labels_version"]))
    line("full-data journeys", f"{s['by_kind'].get('baseline', 0):,}")
    line("designed failures", ", ".join(f"{k} {v:,}" for k, v in sorted(s["designed_failures"].items())))
    line("paid orders", f"{s['paid_orders']:,}")
    print()
    for name, title in (("all", "all journeys"), ("full_data", "full-data journeys")):
        b = s[name]
        line(f"success, {title}", pct(b["success"]))
        line(f"grounded, {title}", pct(b["grounded_success"]))
    if a.by:
        print(f"\nby {a.by}{' (full-data journeys)' if a.full_data else ''}:")
        print(f"{'':<31}{'journeys':>9}{'success':>10}{'grounded':>10}")
        for row in s["by_" + a.by]:
            print(f"{row['value']:<31}{row['journeys']:>9,}{row['success']:>9.1f}%{row['grounded_success']:>9.1f}%")


if __name__ == "__main__":
    main()

"""tools/headline.py counts what the rows say: rates over the rows given, full data as kind=baseline, paid orders as
order_placed, and intervals that resample whole stores."""

from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("headline", Path(__file__).parents[1] / "tools" / "headline.py")
headline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(headline)


def row(store: str, success: bool, grounded: bool, kind: str = "baseline", end: str = "cart_ready") -> dict:
    return {"episode_id": f"ep-{store}-{success}-{grounded}-{kind}-{end}", "store_id": store, "category": "c",
            "boundary": "cart", "kind": kind, "sub_kind": "full" if kind == "baseline" else "payment",
            "end_reason": end, "success": success, "grounded_success": grounded, "surface": "ucp_mcp",
            "rail": "ucp_psp", "intent": "find", "platform_pack": "shopify", "labels_version": "1.1"}


ROWS = [row("a", True, True, end="order_placed"), row("a", True, False), row("a", False, False),
        row("b", True, True), row("b", False, False, kind="designed_failure")]


def test_rates_and_counts_come_straight_from_the_rows() -> None:
    s = headline.summary(ROWS, draws=0, seed=0)
    assert s["journeys"] == 5 and s["stores"] == 2 and s["paid_orders"] == 1
    assert s["all"]["success"]["pct"] == 60.0 and s["all"]["grounded_success"]["pct"] == 40.0
    assert s["full_data"]["journeys"] == 4 and s["full_data"]["success"]["pct"] == 75.0
    assert s["designed_failures"] == {"payment": 1}


def test_the_interval_resamples_stores_and_brackets_store_level_extremes() -> None:
    lo, hi = headline.store_interval(ROWS, "success", draws=500, seed=1)
    # store a succeeds 2 of 3, store b 1 of 2: any resample of whole stores lies between 50% and 66.7%
    assert 50.0 <= lo <= hi <= 100 * 2 / 3 + 1e-9

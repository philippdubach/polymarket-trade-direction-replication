"""Depth at the best bid from price_change rows against the book snapshots, around the cutover."""
from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path

import polars as pl

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import b0_depth_validate as b0  # noqa: E402


def _panel():
    rows = []
    for d, mult in ((dt.date(2026, 4, 27), 1.0), (dt.date(2026, 4, 28), 2.0)):
        for a in ("a", "b", "c"):
            rows.append({"asset_id": a, "day": d, "depth_bid_n": 4, "depth_bid_sum": 40.0 * mult,
                         "book_bid_best_mean": 10.0 * mult if a != "c" else None, "n_book": 2 if a != "c" else 0})
    return pl.DataFrame(rows)


def test_daily_compare_gives_both_depth_measures_and_their_paired_change_from_the_day_before_the_event():
    out = b0.daily_compare(_panel(), event=dt.date(2026, 4, 28))
    d28 = out.filter(pl.col("day") == dt.date(2026, 4, 28)).row(0, named=True)
    assert d28["n_assets"] == 3 and d28["n_with_snapshot"] == 2
    assert math.isclose(d28["median_log_ratio_pc_over_snap"], 0.0, abs_tol=1e-9)   # 20 vs 10: log ratio of 2? no: pc mean = 80/4 = 20, snap = 20
    assert math.isclose(d28["dlog_pc"], math.log(2), abs_tol=1e-9)
    assert math.isclose(d28["dlog_snap"], math.log(2), abs_tol=1e-9)

"""Cutover event study on a synthetic contract-day panel."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import b1_cutover as cu  # noqa: E402

E = dt.date(2026, 4, 28)


def panel():
    """Assets a, b quoted live every day E-8..E+7; c is a placeholder on E-3; d appears at E; b vanishes after E+2."""
    rows = []
    for k in range(-8, 8):
        day = E + dt.timedelta(days=k)
        for a in ("a", "b", "c", "d"):
            if a == "d" and k < 0:
                continue
            if a == "b" and k > 2:
                continue
            placeholder = (a == "c" and k == -3)
            base = {"a": 0.10, "b": 0.20, "c": 0.30, "d": 0.40}[a]
            spread = 0.95 if placeholder else base * (2.0 if k == 0 else 1.5 if k == 1 else 1.0)
            rows.append({"asset_id": a, "day": day, "spread_median": spread, "updates": 100 if k != 0 else 50,
                         "depth_bid_sum": 10.0, "depth_bid_n": 2, "empty_share": 1.0 if placeholder else 0.0, "hours_present": 24,
                         "trades": 3, "placeholder": placeholder, "day_complete": True})
    return pl.DataFrame(rows)


def test_cohort_requires_live_quotes_on_every_pre_day():
    c = cu.cohort(panel(), E, pre_days=7)
    assert sorted(c) == ["a", "b"]           # c has a placeholder day, d has no pre-period


def test_paired_paths_are_medians_of_within_asset_log_changes():
    p = cu.paths(panel(), ["a", "b"], E, k_range=range(-2, 3), outcome="spread_median")
    by_k = {r["k"]: r for r in p.to_dicts()}
    assert by_k[0]["median"] == pytest.approx(pl.Series([2.0, 2.0]).log()[0])
    assert by_k[1]["median"] == pytest.approx(pl.Series([1.5]).log()[0])
    assert by_k[-1]["median"] == pytest.approx(0.0) and by_k[0]["n"] == 2


def test_half_life_is_first_k_where_effect_halves():
    p = pl.DataFrame({"k": [-1, 0, 1, 2, 3], "median": [0.0, 1.0, 0.7, 0.4, 0.1]})
    assert cu.half_life(p) == 2


def test_attrition_bounds_bracket_the_point_estimate():
    b = cu.attrition_bounds(panel(), ["a", "b"], E, k=4, outcome="spread_median")
    assert b["survivors"] == 1 and b["cohort"] == 2
    assert b["worst"] >= b["point"] >= b["best"] or b["best"] >= b["point"] >= b["worst"]


def test_asset_bootstrap_gives_an_interval_around_the_median():
    d = pl.DataFrame({"asset_id": [str(i) for i in range(50)], "dlog": [0.5 + 0.01 * (i % 5) for i in range(50)]})
    lo, hi = cu.boot_median(d["dlog"], draws=200, seed=1)
    assert lo <= 0.52 <= hi and hi - lo < 0.1


def test_paths_report_the_share_of_assets_that_moved_up_and_down():
    import datetime as dt
    import polars as pl
    rows = []
    for a, s0, s1 in (("a", 0.01, 0.02), ("b", 0.01, 0.01), ("c", 0.02, 0.01), ("d", 0.01, 0.03)):
        rows += [{"asset_id": a, "day": dt.date(2026, 4, 27), "spread_median": s0, "placeholder": False},
                 {"asset_id": a, "day": dt.date(2026, 4, 28), "spread_median": s1, "placeholder": False}]
    panel = pl.DataFrame(rows)
    p = cu.paths(panel, ["a", "b", "c", "d"], dt.date(2026, 4, 28), k_range=[0], outcome="log_spread", draws=0)
    r = p.row(0, named=True)
    assert r["share_up"] == 0.5 and r["share_down"] == 0.25 and abs(r["net_up"] - 0.25) < 1e-12


def test_attrition_bound_takes_the_low_tail_where_a_fall_is_adverse():
    up = cu.attrition_bounds(panel(), ["a", "b"], E, k=4, outcome="spread_median")
    down = cu.attrition_bounds(panel(), ["a", "b"], E, k=4, outcome="log_updates")
    assert up["adverse_quantile"] == 0.9 and down["adverse_quantile"] == 0.1
    assert down["worst"] <= down["best"]

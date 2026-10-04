"""Classifying tick-change events and modelling adoption timing."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import c0_rollout_rule as ro  # noqa: E402

T = dt.datetime(2026, 5, 13, 18, 0, 0)


def events():
    return pl.DataFrame({
        "asset_id": ["r", "l1", "l2", "w", "b1", "b2", "b3", "x"],
        "condition_id": ["c1", "c2", "c3", "c4", "c5", "c5", "c5", "c6"],
        "ts": [T, T, T, dt.datetime(2026, 7, 2, 0, 31, 30), T + dt.timedelta(seconds=10), T + dt.timedelta(seconds=20),
               T + dt.timedelta(seconds=30), T + dt.timedelta(hours=3)],
        "old_tick": [0.01] * 8, "new_tick": [0.001, 0.001, 0.001, 0.0025, 0.001, 0.001, 0.001, 0.001],
        "bid_before": [0.97, 0.40, None, 0.5, 0.5, 0.5, 0.5, 0.5],
        "ask_before": [0.99, 0.60, None, 0.52, 0.52, 0.52, 0.52, 0.52],
        "quote_before_ts": [T - dt.timedelta(minutes=5), T - dt.timedelta(minutes=5), None, None, None, None, None, T - dt.timedelta(minutes=5)],
        "first_seen_ts": [T - dt.timedelta(days=10), T - dt.timedelta(hours=2), T - dt.timedelta(minutes=1), T - dt.timedelta(days=3)] + [T - dt.timedelta(days=5)] * 3 + [T - dt.timedelta(days=30)],
        "first_live_quote_ts": [T - dt.timedelta(days=10), T + dt.timedelta(minutes=1), T + dt.timedelta(minutes=10), T - dt.timedelta(days=3)] + [T - dt.timedelta(days=5)] * 3 + [T - dt.timedelta(days=30)],
        "first_trade_ts": [T - dt.timedelta(days=9), None, None, T - dt.timedelta(days=2)] + [T - dt.timedelta(days=4)] * 3 + [T - dt.timedelta(days=29)],
        "event_id": ["e1", "e2", "e3", "e4", "e5", "e5", "e5", "e6"],
    })


def test_classes_are_assigned_in_order():
    c = ro.classify(events(), batch_min=3)
    got = dict(zip(c["asset_id"].to_list(), c["klass"].to_list()))
    assert got == {"r": "R", "l1": "L", "l2": "L", "w": "W", "b1": "B", "b2": "B", "b3": "B", "x": "X"}


def test_activation_timing_relative_to_first_live_quote():
    c = ro.classify(events(), batch_min=3)
    row = c.filter(pl.col("asset_id") == "l1").row(0, named=True)
    assert row["mins_to_first_live_quote"] == pytest.approx(1.0)


def test_hazard_fit_recovers_a_strong_covariate():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 4000
    active = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(-3.0 + 2.5 * active)))
    y = rng.random(n) < p
    df = pl.DataFrame({"adopt": y, "active": active.astype(float), "age": rng.random(n)})
    fit = ro.hazard_fit(df, ["active", "age"], weight_nonadopters=1.0)
    assert fit["coef"]["active"] > 1.5 and fit["auc"] > 0.6


def test_price_at_event_describes_where_the_quote_sat_when_the_tick_changed():
    import datetime as dt
    import polars as pl
    ev = pl.DataFrame({
        "condition_id": ["m1", "m1", "m2", "m2", "m3"],
        "asset_id": ["a", "b", "c", "d", "e"],
        "ts": [dt.datetime(2026, 5, 1, 12, 0, 0)] * 2 + [dt.datetime(2026, 5, 2, 9, 0, 0), dt.datetime(2026, 5, 2, 9, 0, 30), dt.datetime(2026, 5, 3)],
        "klass": ["R", "R", "R", "R", "X"],
        "mid_before": [0.995, 0.005, 0.965, 0.035, 0.5],
    })
    p = ro.price_at_event(ev)
    assert p["R"]["n"] == 4 and p["R"]["high_side"] == 2 and p["R"]["low_side"] == 2
    assert abs(p["R"]["high_mid_median"] - 0.98) < 1e-9 and abs(p["R"]["low_mid_median"] - 0.02) < 1e-9
    assert abs(p["R"]["share_within_0.01_of_threshold"] - 0.5) < 1e-9
    assert p["R"]["share_paired_with_complement_within_60s"] == 1.0
    assert p["X"]["n"] == 1 and p["X"]["share_within_0.01_of_threshold"] == 0.0


def test_resolution_proximity_measures_hours_from_the_event_to_the_market_close_and_game_start():
    import datetime as dt
    import polars as pl
    ev = pl.DataFrame({
        "condition_id": ["m1", "m2", "m3"], "asset_id": ["a", "b", "c"], "klass": ["R", "R", "L"],
        "ts": [dt.datetime(2026, 5, 1, 20), dt.datetime(2026, 5, 1, 20), dt.datetime(2026, 5, 1, 8)],
        "bid_before": [0.99, 0.0, 0.4], "ask_before": [1.0, 0.01, 0.6],
    })
    gamma = pl.DataFrame({"condition_id": ["m1", "m2", "m3"],
                          "closed_time": [dt.datetime(2026, 5, 1, 22), dt.datetime(2026, 5, 3, 20), dt.datetime(2026, 5, 4, 8)],
                          "game_start_time": [dt.datetime(2026, 5, 1, 18), None, None]})
    r = ro.resolution_proximity(ev, gamma)
    assert r["R"]["n"] == 2 and r["R"]["one_sided_share"] == 1.0
    assert r["R"]["hours_to_close_median"] == 25.0 and r["R"]["share_closed_within_24h"] == 0.5
    assert r["R"]["sports_with_game_start"] == 1 and r["R"]["hours_after_game_start_median"] == 2.0 and r["R"]["share_after_game_start"] == 1.0
    assert r["L"]["hours_to_close_median"] == 72.0 and r["L"]["one_sided_share"] == 0.0


def test_event_structure_reports_class_overlap_untraded_share_and_market_level_count():
    import datetime as dt
    import polars as pl
    t0 = dt.datetime(2026, 5, 1, 12, 0, 0)
    ev = pl.DataFrame({
        "condition_id": ["m1", "m1", "m2", "m2"], "asset_id": ["a", "b", "c", "d"], "klass": ["R", "R", "R", "L"],
        "ts": [t0, t0 + dt.timedelta(seconds=5), t0 + dt.timedelta(hours=1), t0 + dt.timedelta(hours=1, seconds=3)],
        "first_seen_ts": [t0 - dt.timedelta(hours=2)] * 2 + [t0 - dt.timedelta(days=3)] * 2,
        "first_live_quote_ts": [t0 - dt.timedelta(hours=1)] * 4,
        "first_trade_ts": [None, None, t0 - dt.timedelta(hours=1), t0 - dt.timedelta(hours=1)],
    })
    r = ro.event_structure(ev)
    assert r["market_level_changes"] == 2
    assert r["R"]["n"] == 3 and abs(r["R"]["share_meeting_activation"] - 2 / 3) < 1e-9   # a and b are within a day of listing
    assert abs(r["R"]["share_no_prior_trade"] - 2 / 3) < 1e-9


def test_hazard_sample_risk_set_excludes_assets_already_on_the_finer_tick_and_degraded_days():
    import datetime as dt
    import polars as pl
    days = [dt.date(2026, 5, 1) + dt.timedelta(days=i) for i in range(4)]
    rows = []
    for a in ("early", "late", "never"):
        for d in days:
            rows.append({"asset_id": a, "day": d, "updates": 10, "trades": 1, "placeholder": False, "mid_last": 0.5, "spread_median": 0.02, "age_days": 3, "neg_risk": False})
    panel = pl.DataFrame(rows)
    # 'early' changed tick before the window; 'late' changes on 3 May; 'never' stays on 0.01
    events = pl.DataFrame({"asset_id": ["early", "late"], "ts": [dt.datetime(2026, 4, 20), dt.datetime(2026, 5, 3, 15)]})
    hs = ro.hazard_sample(panel, events, frac=1.0, degraded={dt.date(2026, 5, 2)})
    assert "early" not in hs["asset_id"].to_list()                      # already on 0.001: never at risk
    late = hs.filter(pl.col("asset_id") == "late").sort("t")
    assert late["t"].to_list() == [dt.date(2026, 5, 3)] and late["adopt"].to_list() == [True]   # 2 May at-risk row dropped (degraded), 4 May after adoption dropped
    never = hs.filter(pl.col("asset_id") == "never")
    assert never["t"].to_list() == [dt.date(2026, 5, 3), dt.date(2026, 5, 4), dt.date(2026, 5, 5)] and not never["adopt"].any()

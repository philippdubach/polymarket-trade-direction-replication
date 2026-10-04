"""Staggered event study with not-yet-treated controls: ATT table, aggregation, weights, exclusions."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import c2_event_study as es  # noqa: E402

D = dt.date


def panel():
    """Units u1, u2 adopt on 05-10 (cohort g); c1, c2 never adopt. y falls by 1 for adopters from g on."""
    rows = []
    for u, g in (("u1", D(2026, 5, 10)), ("u2", D(2026, 5, 10)), ("c1", None), ("c2", None)):
        for k in range(-3, 4):
            day = D(2026, 5, 10) + dt.timedelta(days=k)
            y = (1.0 if u.startswith("c") else 2.0) + (-1.0 if g and day >= g else 0.0)
            rows.append({"asset_id": u, "day": day, "y": y, "trades": 1, "g": g})
    return pl.DataFrame(rows)


def test_pairs_form_long_differences_from_the_base_day():
    pr = es.pairs(panel(), liquid=True, sample_mod=1)
    r = pr.filter((pl.col("asset_id") == "u1") & (pl.col("b") == D(2026, 5, 9)) & (pl.col("t") == D(2026, 5, 10)))
    assert r["dy"][0] == pytest.approx(-1.0)


def test_att_by_event_time_recovers_minus_one_after_and_zero_before():
    pr = es.pairs(panel(), liquid=True, sample_mod=1)
    tab = es.att_table(pr)
    att = es.aggregate(tab)
    assert att[0] == pytest.approx(-1.0) and att[1] == pytest.approx(-1.0)
    assert att[-2] == pytest.approx(0.0)


def test_control_weights_change_the_control_mean():
    p = panel().with_columns(pl.when(pl.col("asset_id") == "c2").then(pl.col("y") + 0.5).otherwise(pl.col("y")).alias("y"))
    # c2 drifts up by 0.5 only on day t = g (so its dy is +0.5 there); weighting it 3x moves the control mean
    p = p.with_columns(pl.when((pl.col("asset_id") == "c2") & (pl.col("day") == D(2026, 5, 10))).then(pl.col("y") + 0.5).otherwise(pl.col("y")).alias("y"))
    pr = es.pairs(p, liquid=True, sample_mod=1)
    w = pl.DataFrame({"asset_id": ["c1", "c2"], "w": [1.0, 3.0]})
    unweighted = es.aggregate(es.att_table(pr))[0]
    weighted = es.aggregate(es.att_table(pr, weights=w))[0]
    assert weighted < unweighted   # the control mean rose, so ATT is more negative


def test_excluded_days_drop_both_cohorts_and_control_days():
    pr = es.pairs(panel(), liquid=True, sample_mod=1, excluded_days={D(2026, 5, 10)})
    assert pr.filter((pl.col("t") == D(2026, 5, 10)) | (pl.col("b") == D(2026, 5, 10))).height == 0

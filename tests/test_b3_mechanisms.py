"""Mechanism tests around the cutover: anticipation, category DiD, archive seam."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import b3_mechanisms as mech  # noqa: E402

E = dt.date(2026, 4, 28)


def hourly():
    """Cohort median spread by hour on the event day and a reference day: flat until 11:00, then a jump."""
    rows = []
    for day, jump in ((E, True), (dt.date(2026, 4, 27), False)):
        for h in range(24):
            rows.append({"day": day, "hour": h, "spread_median": 0.10 * (3.0 if jump and h >= 11 else 1.0),
                         "empty_share": 0.9 if jump and 11 <= h <= 13 else 0.05})
    return pl.DataFrame(rows)


def test_anticipation_is_the_pre_maintenance_change_against_the_reference():
    a = mech.anticipation(hourly(), E, dt.date(2026, 4, 27), cut_hour=11)
    assert a["pre_hours_log_ratio_median"] == pytest.approx(0.0)
    assert a["post_hours_log_ratio_median"] == pytest.approx(pl.Series([3.0]).log()[0])
    assert a["max_empty_share_event_day"] == pytest.approx(0.9) and a["first_hour_empty_above_0_5"] == 11


def panel():
    rows = []
    for k in range(-8, 8):
        for a, cat, base, post in (("g1", "Geopolitics", 0.10, 1.0), ("g2", "Geopolitics", 0.20, 1.0),
                                   ("s1", "Sports", 0.10, 1.5), ("s2", "Crypto", 0.30, 1.5)):
            rows.append({"asset_id": a, "category": cat, "day": E + dt.timedelta(days=k),
                         "spread_median": base * (post if k >= 5 else 1.0), "placeholder": False})
    return pl.DataFrame(rows)


def test_category_did_compares_post_week_to_pre_week_levels():
    d = mech.category_did(panel(), ["g1", "g2", "s1", "s2"], E, treated="Geopolitics")
    assert d["treated_assets"] == 2 and d["control_assets"] == 2
    assert d["treated_change"] == pytest.approx(0.0)
    assert d["control_change"] == pytest.approx(pl.Series([1.5]).log()[0])
    assert d["did"] == pytest.approx(-pl.Series([1.5]).log()[0])


def test_seam_placebo_pairs_assets_across_the_two_archives():
    p = pl.DataFrame({"asset_id": ["a", "a", "b", "b", "c"], "era": ["v1", "v2", "v1", "v2", "v1"],
                      "day": [dt.date(2026, 4, 14), dt.date(2026, 4, 16)] * 2 + [dt.date(2026, 4, 14)],
                      "spread_median": [0.10, 0.10, 0.20, 0.22, 0.5], "placeholder": [False] * 5})
    s = mech.seam_placebo(p, dt.date(2026, 4, 14), dt.date(2026, 4, 16))
    assert s["assets"] == 2 and s["median_log_change"] == pytest.approx(0.5 * pl.Series([1.1]).log()[0])


def test_hourly_extract_gives_cohort_medians_per_hour(tmp_path):
    from test_x0_extract import write_v2_hours
    files = write_v2_hours(tmp_path)
    h = mech.hourly_extract(["2026-05-13"], ["111"], files_by_day={"2026-05-13": files})
    rows = {r["hour"]: r for r in h.to_dicts()}
    assert rows[0]["spread_median"] == pytest.approx(0.02) and rows[0]["empty_share"] == 0.0
    assert rows[1]["spread_median"] == pytest.approx(0.998) and rows[1]["empty_share"] == 1.0
    assert h["day"][0] == dt.date(2026, 5, 13)

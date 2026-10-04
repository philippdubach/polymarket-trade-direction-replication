"""Capture-outage flags for the archive."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import d0_archive_health as ah  # noqa: E402


def _series(vals):
    days = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(len(vals))]
    return pl.DataFrame({"day": days, "first_seen": vals})


def test_degraded_when_first_seen_collapses_against_trailing_median():
    vals = [1000] * 10 + [50, 40, 30, 20] + [1200] * 5
    df = _series(vals)
    flags = ah.degraded_days(df, listings=None, window=7, ratio=0.10)
    assert flags == {dt.date(2026, 6, 11), dt.date(2026, 6, 12), dt.date(2026, 6, 13), dt.date(2026, 6, 14)}


def test_not_degraded_when_the_venue_also_listed_nothing():
    vals = [1000] * 10 + [50, 40] + [1000] * 3
    df = _series(vals)
    listings = df.with_columns(pl.Series("listed", [1000] * 10 + [30, 20] + [1000] * 3))
    assert ah.degraded_days(df, listings=listings, window=7, ratio=0.10) == set()


def test_missing_hours_are_degraded_regardless():
    df = _series([1000] * 12).with_columns(pl.Series("hours_present", [24] * 5 + [21] + [24] * 6))
    assert ah.degraded_days(df, listings=None, window=7, ratio=0.10, min_hours=24) == {dt.date(2026, 6, 6)}


def test_per_day_table_counts_first_seen_once_across_eras():
    sh = pl.DataFrame({
        "asset_id": ["a", "b", "a", "c"], "condition_id": ["0x1", "0x2", "0x1", "0x3"],
        "day": [dt.date(2026, 4, 13), dt.date(2026, 4, 13), dt.date(2026, 4, 16), dt.date(2026, 4, 16)],
        "updates": [10, 20, 30, 40], "trades": [1, 2, 3, 4], "era": ["v1", "v1", "v2", "v2"],
    })
    per_day, cids = ah.per_day_table(sh)
    d13, d16 = per_day.row(0, named=True), per_day.row(1, named=True)
    assert (d13["assets"], d13["first_seen"], d13["updates"], d13["era"]) == (2, 2, 30, "v1")
    assert (d13["first_seen_markets"], d16["first_seen_markets"]) == (2, 1)
    assert (d16["assets"], d16["first_seen"], d16["trades"], d16["era"]) == (2, 1, 7, "v2")
    assert cids == {"0x1", "0x2", "0x3"}


def test_long_outage_stays_flagged_because_flagged_days_leave_the_trailing_window():
    vals = [1000] * 10 + [0] * 12 + [1200] * 3
    df = _series(vals)
    flags = ah.degraded_days(df, listings=None, window=7, ratio=0.10)
    assert flags == {dt.date(2026, 6, 11) + dt.timedelta(days=i) for i in range(12)}


def test_day_whose_listings_the_archive_mostly_never_carried_is_degraded():
    df = _series([1000] * 12)
    listings = df.with_columns(pl.Series("listed", [1000] * 12), pl.Series("carried", [950] * 8 + [300] + [950] * 3))
    assert ah.degraded_days(df, listings=listings, window=7, ratio=0.10, min_carried=0.5) == {dt.date(2026, 6, 9)}


def test_thin_hours_flag_the_day_unless_inside_the_documented_migration_window():
    rows = {f"2026-05-22T{h:02d}": 80_000_000 for h in range(24)} | {"2026-05-22T17": 220_000, "2026-05-22T18": 230_000,
            "2026-04-28T11": 1_200_000, "2026-04-28T12": 14_000_000, "2026-04-28T13": 80_000_000}
    thin = ah.thin_hours(rows, ratio=0.2, exempt={"2026-04-28T11", "2026-04-28T12"})
    assert thin == {"2026-05-22": [17, 18]}
    assert ah.thin_hours(rows, ratio=0.2, exempt=set()) == {"2026-04-28": [11, 12], "2026-05-22": [17, 18]}


def test_sensitivity_reports_the_flagged_count_under_alternative_thresholds():
    df = _series([1000] * 10 + [50, 40] + [1000] * 3).with_columns(pl.Series("hours_present", [24] * 15))
    listings = df.with_columns(pl.Series("listed", [1000] * 15), pl.Series("carried", [950] * 10 + [400, 300] + [950] * 3))
    s = ah.sensitivity(df, listings)
    assert s["baseline"]["n"] == 2
    assert {"ratio=0.333", "ratio=0.667", "min_carried=0.333", "min_carried=0.667", "window=5", "window=10"} <= set(s)
    assert s["min_carried=0.333"]["n"] == 2 and s["ratio=0.667"]["n"] >= 2

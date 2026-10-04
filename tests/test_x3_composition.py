"""Composition of the panel: the daily cross-section against the venue's listing rate."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import x3_composition as x3  # noqa: E402


def _panel():
    rows = []
    for i, d in enumerate((dt.date(2026, 4, 27), dt.date(2026, 4, 28))):
        for a, s, ph in (("a", 0.01, False), ("b", 0.02, False), ("c", 0.9, False), ("d", None, True)):
            rows.append({"asset_id": a, "day": d, "spread_median": s if s is None else s * (1 + i), "placeholder": ph, "era": "v2"})
    return pl.DataFrame(rows)


def test_daily_cross_section_reports_quantiles_over_live_quotes_and_the_tight_share():
    out = x3.daily_cross_section(_panel(), tight=0.02)
    r = out.filter(pl.col("day") == dt.date(2026, 4, 27)).row(0, named=True)
    assert r["assets"] == 4 and r["live"] == 3
    assert r["median"] == 0.02 and abs(r["p25"] - 0.015) < 1e-9 and abs(r["p75"] - 0.46) < 1e-9
    assert abs(r["tight_share"] - 2 / 3) < 1e-9


def test_both_sides_share_counts_assets_live_on_both_days():
    s = x3.both_sides(_panel(), dt.date(2026, 4, 27), dt.date(2026, 4, 28))
    assert s == {"before": 3, "after": 3, "both": 3, "share_of_before": 1.0}

"""Stacking the V1 and V2 daily shards into one contract-day panel."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import x2_build_panel as bp  # noqa: E402

D = dt.date


def _shard(era, day, asset, updates=10, spread_sum=1.0, hours=24, first=None, live=None, trades=None):
    return pl.DataFrame({
        "condition_id": ["0xa"], "asset_id": [asset], "outcome": [None], "updates": [updates], "spread_sum": [spread_sum],
        "spread_sq_sum": [0.1], "spread_max": [0.5], "spread_median": [0.1], "spread_p25": [0.05], "spread_p75": [0.2],
        "mid_last": [0.5], "hours_present": [hours], "updates_empty": [0], "depth_bid_n": [1], "depth_bid_sum": [5.0],
        "depth_ask_n": [1], "depth_ask_sum": [5.0], "n_book": [0], "book_bid_best_size": [None], "book_ask_best_size": [None],
        "book_bid_best_mean": [None], "book_ask_best_mean": [None], "book_order_mismatch": [0],
        "trades": [trades], "trade_volume": [None], "trades_buy": [None], "trades_sell": [None],
        "first_quote_ts": [first or dt.datetime.combine(day, dt.time(1))], "first_live_quote_ts": [live],
        "last_quote_ts": [dt.datetime.combine(day, dt.time(23))], "first_trade_ts": [None], "era": [era], "day": [day],
    }).with_columns(pl.col("book_bid_best_size", "book_ask_best_size", "book_bid_best_mean", "book_ask_best_mean", "trade_volume").cast(pl.Float64),
                    pl.col("trades", "trades_buy", "trades_sell").cast(pl.Int64),
                    pl.col("first_live_quote_ts", "first_trade_ts").cast(pl.Datetime("us")))


def test_panel_stacks_eras_and_derives_mean_and_completeness():
    v1 = _shard("v1", D(2026, 4, 14), "1", updates=4, spread_sum=0.4)
    v2 = _shard("v2", D(2026, 4, 16), "1", updates=10, spread_sum=2.0, hours=23, trades=3)
    labels = pl.DataFrame({"asset_id": ["1"], "condition_id": ["0xa"], "outcome": ["YES"]})
    p = bp.build(pl.concat([v1, v2]), labels)
    assert p.height == 2 and p["era"].to_list() == ["v1", "v2"]
    r1, r2 = p.to_dicts()
    assert r1["spread_mean"] == 0.1 and r2["spread_mean"] == 0.2
    assert r1["day_complete"] is True and r2["day_complete"] is False
    assert r1["outcome"] == "YES" and r2["outcome"] == "YES"
    assert r1["placeholder"] is False


def test_first_seen_is_the_minimum_across_both_eras():
    v1 = _shard("v1", D(2026, 4, 10), "1")
    v2a = _shard("v2", D(2026, 4, 16), "1")
    v2b = _shard("v2", D(2026, 5, 1), "2")
    p = bp.build(pl.concat([v1, v2a, v2b]), pl.DataFrame({"asset_id": ["1", "2"], "condition_id": ["0xa", "0xa"], "outcome": ["YES", "NO"]}))
    fs = dict(zip(p["asset_id"].to_list(), p["first_seen_day"].to_list()))
    assert fs["1"] == D(2026, 4, 10) and fs["2"] == D(2026, 5, 1)
    assert p.filter(pl.col("asset_id") == "1")["age_days"].to_list() == [0, 6]


def test_placeholder_days_are_flagged_by_median_or_mid():
    s = _shard("v2", D(2026, 5, 1), "1").with_columns(pl.lit(0.95).alias("spread_median"))
    p = bp.build(s, pl.DataFrame({"asset_id": ["1"], "condition_id": ["0xa"], "outcome": ["YES"]}))
    assert p["placeholder"].to_list() == [True]

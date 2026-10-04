"""Quote- and trade-based direction rules scored against the taker side, fill by fill."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import a2_direction_rules as dr  # noqa: E402

T = dt.datetime(2026, 5, 13, 10, 0, 0)


def trades():
    """One asset, prints with the prevailing quote already attached."""
    t = [T + dt.timedelta(seconds=i) for i in range(6)]
    return pl.DataFrame({
        "asset_id": ["1"] * 6, "ts": t,
        "price": [0.52, 0.48, 0.50, 0.50, 0.55, 0.55],
        "bid": [0.49, 0.49, 0.49, 0.49, None, 0.50],
        "ask": [0.51, 0.51, 0.51, 0.51, None, 0.60],
        "quote_ts": [T - dt.timedelta(seconds=2)] * 4 + [None, T + dt.timedelta(seconds=4)],
        "size": [10.0, 5.0, 1.0, 2.0, 3.0, 4.0],
        "taker_side": [0, 1, 0, 1, 0, 0],
    })


def test_lee_ready_signs_by_mid_then_falls_back_to_tick():
    s = dr.sign_rules(trades())
    # LR: 0.52 > mid 0.50 -> +1; 0.48 < mid -> -1; 0.50 == mid -> tick test (0.50 > 0.48 -> +1);
    # 0.50 == mid, zero tick -> carry last non-zero tick (+1); no quote -> tick (0.55 > 0.50 -> +1); 0.55 == mid 0.55 -> zero tick carry +1
    assert s["lr"].to_list() == [1, -1, 1, 1, 1, 1]


def test_tick_test_uses_previous_trade_and_carries_through_zero_ticks():
    s = dr.sign_rules(trades())
    assert s["tick"].to_list() == [None, -1, 1, 1, 1, 1]


def test_emo_uses_tick_at_the_quotes_and_mid_between():
    s = dr.sign_rules(trades())
    # EMO: at bid/ask use the quote, else tick test; 0.52 > ask? no (0.52 > 0.51 is outside) -> LR's outside rule: +1
    assert s["emo"].to_list()[:2] == [1, -1]


def test_bvc_assigns_bulk_buy_share_per_bar():
    s = dr.bvc(trades(), bar="5s", sigma_window=3)
    assert set(s.columns) >= {"asset_id", "bar", "buy_share_bvc"}
    assert 0.0 <= s["buy_share_bvc"].min() and s["buy_share_bvc"].max() <= 1.0


def test_metrics_against_taker_side_are_balanced():
    truth = pl.Series([0, 1, 0, 1, 0, 0])
    pred = pl.Series([1, -1, 1, 1, 1, 1])   # rule signs: +1 buy, -1 sell
    m = dr.metrics(truth, pred)
    assert m["n"] == 6 and m["recall_buy"] == 1.0 and m["recall_sell"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx(0.75) and m["accuracy"] == pytest.approx(5 / 6)
    assert m["majority_baseline"] == pytest.approx(4 / 6) and -1 <= m["mcc"] <= 1


def test_metrics_skip_unsigned_predictions():
    m = dr.metrics(pl.Series([0, 1, 0]), pl.Series([None, -1, 1]))
    assert m["n"] == 2 and m["coverage"] == pytest.approx(2 / 3)


def test_permutation_null_is_near_chance():
    truth = pl.Series([0] * 500 + [1] * 500)
    pred = pl.Series([1] * 500 + [-1] * 500)
    null = dr.permutation_null(truth, pred, draws=50, seed=1)
    assert abs(null["balanced_accuracy_mean"] - 0.5) < 0.05


def test_segments_table_keys():
    s = dr.sign_rules(trades()).with_columns(pl.lit(0.001).alias("tick_size"), pl.lit("std").alias("contract"))
    seg = dr.segments(s, ["tick_size", "contract"])
    assert ("tick_size", "0.001") in seg and seg[("tick_size", "0.001")]["lr"]["n"] == 6


def test_tick_at_print_uses_last_change_before_the_print_else_the_asset_default():
    ticks = pl.DataFrame({"asset_id": ["1", "1"], "tts": [T - dt.timedelta(hours=2), T + dt.timedelta(hours=1)], "old_tick": [0.01, 0.001], "new_tick": [0.001, 0.0025]})
    prints = pl.DataFrame({"asset_id": ["1", "1", "2", "3"], "ts": [T, T + dt.timedelta(hours=2), T, T]})
    defaults = pl.DataFrame({"asset_id": ["2"], "tick_default": [0.001]})
    got = dr.tick_at_print(prints, ticks=ticks, defaults=defaults).to_list()
    assert got == [0.001, 0.0025, 0.001, 0.01]     # asset 3 has neither history nor a Gamma value


def test_tick_at_print_uses_the_first_events_old_tick_before_that_event_not_the_metadata_snapshot():
    import datetime as dt
    import polars as pl
    prints = pl.DataFrame({"asset_id": ["a", "a", "b"], "ts": [dt.datetime(2026, 5, 13, 10), dt.datetime(2026, 5, 13, 14), dt.datetime(2026, 5, 13, 10)]})
    ticks = pl.DataFrame({"asset_id": ["a"], "tts": [dt.datetime(2026, 5, 13, 12)], "old_tick": [0.01], "new_tick": [0.001]})
    defaults = pl.DataFrame({"asset_id": ["a", "b"], "tick_default": [0.001, 0.001]})   # the snapshot, taken after resolution
    out = dr.tick_at_print(prints, ticks=ticks, defaults=defaults).to_list()
    assert out[0] == 0.01     # before the asset's first change: that change's old tick
    assert out[1] == 0.001    # after it: the new tick
    assert out[2] == 0.001    # no event at all: the metadata tick

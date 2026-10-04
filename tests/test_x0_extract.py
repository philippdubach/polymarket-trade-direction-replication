"""The harmonised daily extraction, checked on tiny synthetic V1 and V2 hours.

The fixtures write one hourly file per era with a handful of rows whose
aggregates are known by hand, then run the same DuckDB query the extraction
uses on the archive.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import polars as pl
import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import x0_extract_daily as x0  # noqa: E402

T0 = dt.datetime(2026, 5, 13, 0, 0, 0)


def _ts(h: int, m: int, s: int = 0) -> dt.datetime:
    return T0.replace(hour=h, minute=m, second=s)


def write_v2_hours(d: Path) -> list[str]:
    """Two hours (00 and 01) for one asset; hour 02 is missing by design."""
    cond = "0x" + "ab" * 32
    rows = [
        # hour 00: three price_change rows, spreads 0.02, 0.04, 0.02 -> median bucket 20; one at best bid
        dict(ts=_ts(0, 1), event_type="price_change", asset_id="111", bids=None, asks=None, price=0.49, size=10.0, side="BUY",
             best_bid=0.49, best_ask=0.51, tx=None, ot=None, nt=None),
        dict(ts=_ts(0, 2), event_type="price_change", asset_id="111", bids=None, asks=None, price=0.40, size=5.0, side="BUY",
             best_bid=0.48, best_ask=0.52, tx=None, ot=None, nt=None),
        dict(ts=_ts(0, 3), event_type="price_change", asset_id="111", bids=None, asks=None, price=0.52, size=7.0, side="SELL",
             best_bid=0.50, best_ask=0.52, tx=None, ot=None, nt=None),
        # a trade and a book snapshot and a tick change in hour 00
        dict(ts=_ts(0, 4), event_type="last_trade_price", asset_id="111", bids=None, asks=None, price=0.51, size=3.0, side="BUY",
             best_bid=None, best_ask=None, tx="0x01", ot=None, nt=None),
        dict(ts=_ts(0, 5), event_type="book", asset_id="111", bids=json.dumps([["0.40", "100"], ["0.50", "12.5"]]),
             asks=json.dumps([["0.60", "9"], ["0.52", "3"]]), price=None, size=None, side=None, best_bid=None, best_ask=None,
             tx=None, ot=None, nt=None),
        dict(ts=_ts(0, 6), event_type="tick_size_change", asset_id="111", bids=None, asks=None, price=None, size=None, side=None,
             best_bid=None, best_ask=None, tx=None, ot=0.01, nt=0.001),
        # hour 01: an empty book (spread >= 0.9) as the last quote of the hour
        dict(ts=_ts(1, 30), event_type="price_change", asset_id="111", bids=None, asks=None, price=0.001, size=1.0, side="BUY",
             best_bid=0.001, best_ask=0.999, tx=None, ot=None, nt=None),
    ]
    files = []
    for h in (0, 1):
        sub = [r for r in rows if r["ts"].hour == h]
        df = pl.DataFrame({
            "timestamp_received": pl.Series([r["ts"] for r in sub], dtype=pl.Datetime("ms")).dt.replace_time_zone("UTC"),
            "timestamp": pl.Series([r["ts"] for r in sub], dtype=pl.Datetime("ms")).dt.replace_time_zone("UTC"),
            "market": pl.Series([cond.encode("ascii")] * len(sub), dtype=pl.Binary),
            "event_type": [r["event_type"] for r in sub],
            "asset_id": [r["asset_id"] for r in sub],
            "bids": [r["bids"] for r in sub], "asks": [r["asks"] for r in sub],
            "price": pl.Series([r["price"] for r in sub], dtype=pl.Decimal(9, 4)),
            "size": pl.Series([r["size"] for r in sub], dtype=pl.Decimal(18, 6)),
            "side": [r["side"] for r in sub],
            "best_bid": pl.Series([r["best_bid"] for r in sub], dtype=pl.Decimal(9, 4)),
            "best_ask": pl.Series([r["best_ask"] for r in sub], dtype=pl.Decimal(9, 4)),
            "transaction_hash": [r["tx"] for r in sub],
            "old_tick_size": pl.Series([r["ot"] for r in sub], dtype=pl.Decimal(9, 4)),
            "new_tick_size": pl.Series([r["nt"] for r in sub], dtype=pl.Decimal(9, 4)),
        })
        p = d / f"polymarket_orderbook_2026-05-13T{h:02d}.parquet"
        df.write_parquet(p)
        files.append(str(p))
    return files


def write_v1_hours(d: Path) -> list[str]:
    cond = "0x" + "cd" * 32
    def pc(ts, side, bid, ask, cp, cs, cside):
        return dict(ts=ts, ut="price_change", data=json.dumps({"market_id": cond, "token_id": "222" if side == "YES" else "333",
                    "side": side, "best_bid": str(bid), "best_ask": str(ask), "timestamp": ts.timestamp(),
                    "change_price": str(cp), "change_size": str(cs), "change_side": cside}))
    rows = [
        pc(_ts(0, 1), "YES", 0.30, 0.34, 0.30, 50, "BUY"),
        pc(_ts(0, 2), "YES", 0.31, 0.33, 0.20, 5, "BUY"),
        pc(_ts(0, 1), "NO", 0.66, 0.70, 0.70, 8, "SELL"),
        dict(ts=_ts(0, 3), ut="book_snapshot", data=json.dumps({"market_id": cond, "token_id": "222", "side": "YES",
             "best_bid": "0.31", "best_ask": "0.33", "timestamp": _ts(0, 3).timestamp(),
             "bids": [["0.10", "500"], ["0.31", "42"]], "asks": [["0.90", "1"], ["0.33", "6"]]})),
    ]
    df = pl.DataFrame({
        "timestamp_received": pl.Series([r["ts"] for r in rows], dtype=pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "timestamp_created_at": pl.Series([r["ts"] for r in rows], dtype=pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "market_id": [cond] * len(rows), "update_type": [r["ut"] for r in rows], "data": [r["data"] for r in rows],
    })
    p = d / "polymarket_orderbook_2026-05-13T00.parquet"
    df.write_parquet(p)
    return [str(p)]


@pytest.fixture
def v2(tmp_path):
    return x0.extract_day("2026-05-13", "v2", files=write_v2_hours(tmp_path))


@pytest.fixture
def v1(tmp_path):
    return x0.extract_day("2026-05-13", "v1", files=write_v1_hours(tmp_path))


def test_v2_quote_aggregates(v2):
    r = v2.filter(pl.col("asset_id") == "111").row(0, named=True)
    assert r["condition_id"] == "0x" + "ab" * 32 and r["era"] == "v2"
    assert r["updates"] == 4
    assert r["spread_sum"] == pytest.approx(0.02 + 0.04 + 0.02 + 0.998)
    # bucket holding position n/2 of {20,20,40,998}, as the pilot's exact_median_hist defines it
    assert r["spread_median"] == pytest.approx(0.02, abs=0.0011)
    assert r["spread_p25"] == pytest.approx(0.02, abs=0.0011)
    assert r["spread_p75"] == pytest.approx(0.04, abs=0.0011)
    assert r["mid_last"] == pytest.approx(0.5)


def test_v2_depth_at_best_counts_only_rows_at_the_best_level(v2):
    r = v2.row(0, named=True)
    # row1: BUY at 0.49 == best_bid; row2: BUY at 0.40 != best_bid; row3: SELL at 0.52 == best_ask;
    # hour-1 row: BUY at 0.001 == best_bid 0.001
    assert r["depth_bid_n"] == 2 and r["depth_bid_sum"] == pytest.approx(11.0)
    assert r["depth_ask_n"] == 1 and r["depth_ask_sum"] == pytest.approx(7.0)


def test_v2_book_snapshot_best_level_and_ordering_flag(v2):
    r = v2.row(0, named=True)
    assert r["n_book"] == 1
    assert r["book_bid_best_size"] == pytest.approx(12.5)   # last element of bids
    assert r["book_ask_best_size"] == pytest.approx(3.0)    # last element of asks


def test_v2_hours_trades_and_first_timestamps(v2):
    r = v2.row(0, named=True)
    assert r["hours_present"] == 2 and r["updates_empty"] == 1
    assert r["trades"] == 1 and r["trade_volume"] == pytest.approx(0.51 * 3.0)
    assert r["first_quote_ts"] == _ts(0, 1) and r["first_trade_ts"] == _ts(0, 4)
    assert r["first_live_quote_ts"] == _ts(0, 1)   # first quote with spread < 0.9


def test_v2_tick_context_carries_last_quote_before_change(tmp_path):
    files = write_v2_hours(tmp_path)
    ticks = pl.DataFrame({"condition_id": ["0x" + "ab" * 32], "asset_id": ["111"], "ts": pl.Series([_ts(0, 6)], dtype=pl.Datetime("us")),
                          "old_tick": [0.01], "new_tick": [0.001]})
    ctx = x0.tick_context("2026-05-13", files=files, ticks=ticks)
    r = ctx.row(0, named=True)
    assert r["asset_id"] == "111" and r["old_tick"] == pytest.approx(0.01) and r["new_tick"] == pytest.approx(0.001)
    assert r["bid_before"] == pytest.approx(0.50) and r["ask_before"] == pytest.approx(0.52)
    assert r["quote_before_ts"] == _ts(0, 3)


def test_v1_parses_json_payload_per_token(v1):
    yes = v1.filter(pl.col("asset_id") == "222").row(0, named=True)
    no = v1.filter(pl.col("asset_id") == "333").row(0, named=True)
    assert yes["condition_id"] == "0x" + "cd" * 32 and yes["era"] == "v1" and yes["outcome"] == "YES"
    assert yes["updates"] == 2 and yes["spread_sum"] == pytest.approx(0.04 + 0.02)
    assert yes["depth_bid_n"] == 1 and yes["depth_bid_sum"] == pytest.approx(50)
    assert no["updates"] == 1 and no["depth_ask_n"] == 1 and no["depth_ask_sum"] == pytest.approx(8)
    assert yes["n_book"] == 1 and yes["book_bid_best_size"] == pytest.approx(42) and yes["book_ask_best_size"] == pytest.approx(6)
    assert yes["trades"] is None


def test_output_has_the_same_columns_in_both_eras(v1, v2):
    assert list(v1.columns) == list(v2.columns)


def _v2_quotes(d: Path, quotes: list[tuple[str, float, float]]) -> list[str]:
    """One V2 hour of price_change rows, (asset_id, best_bid, best_ask) each."""
    n = len(quotes)
    ts = [_ts(0, i + 1) for i in range(n)]
    df = pl.DataFrame({
        "timestamp_received": pl.Series(ts, dtype=pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "timestamp": pl.Series(ts, dtype=pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "market": pl.Series([("0x" + "ef" * 32).encode("ascii")] * n, dtype=pl.Binary),
        "event_type": ["price_change"] * n, "asset_id": [q[0] for q in quotes],
        "bids": [None] * n, "asks": [None] * n,
        "price": pl.Series([q[1] for q in quotes], dtype=pl.Decimal(9, 4)),
        "size": pl.Series([1.0] * n, dtype=pl.Decimal(18, 6)), "side": ["BUY"] * n,
        "best_bid": pl.Series([q[1] for q in quotes], dtype=pl.Decimal(9, 4)),
        "best_ask": pl.Series([q[2] for q in quotes], dtype=pl.Decimal(9, 4)),
        "transaction_hash": [None] * n,
        "old_tick_size": pl.Series([None] * n, dtype=pl.Decimal(9, 4)),
        "new_tick_size": pl.Series([None] * n, dtype=pl.Decimal(9, 4)),
    })
    p = d / "polymarket_orderbook_2026-05-13T00.parquet"
    df.write_parquet(p)
    return [str(p)]


def test_tick_grid_spreads_land_in_their_own_bucket_and_threshold(tmp_path):
    """0.57 - 0.56 is 0.00999... in doubles: the median must still be one cent, and a 0.05/0.95 book
    (0.8999... in doubles) must count as empty."""
    out = x0.extract_day("2026-05-13", "v2", files=_v2_quotes(tmp_path, [
        ("777", 0.56, 0.57), ("777", 0.56, 0.57), ("777", 0.56, 0.57),
        ("888", 0.05, 0.95), ("888", 0.05, 0.95)]))
    a = out.filter(pl.col("asset_id") == "777").row(0, named=True)
    b = out.filter(pl.col("asset_id") == "888").row(0, named=True)
    assert a["spread_median"] == pytest.approx(0.010)
    assert b["updates_empty"] == 2

"""Attaching the prevailing quote and the forward midpoint to prints, hour by hour."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import a2_direction_rules as dr  # noqa: E402
from test_x0_extract import _ts  # noqa: E402

T0 = dt.datetime(2026, 5, 13, 0, 0, 0)


def write_hours(d: Path) -> list[str]:
    """Hour 0: quotes at 00:01 (0.49/0.51), 00:02 (0.48/0.52); print at 00:04 and at 00:58.
    Hour 1: quote at 01:00:30 (0.40/0.42); print at 01:30 with no quote of its own in the hour before it
    except the carry from hour 0... and a quote at 01:02 (0.44/0.46) so the forward mid of the 00:58 print is 0.45."""
    cond = "0x" + "ab" * 32
    def row(ts, et, **k):
        base = dict(ts=ts, event_type=et, asset_id="111", bids=None, asks=None, price=None, size=None, side=None,
                    best_bid=None, best_ask=None, tx=None, ot=None, nt=None)
        base.update(k); return base
    rows = [
        row(_ts(0, 1), "price_change", price=0.49, size=1.0, side="BUY", best_bid=0.49, best_ask=0.51),
        row(_ts(0, 2), "price_change", price=0.48, size=1.0, side="BUY", best_bid=0.48, best_ask=0.52),
        row(_ts(0, 4), "last_trade_price", price=0.50, size=3.0, side="BUY", tx="0x01"),
        row(_ts(0, 58), "last_trade_price", price=0.51, size=2.0, side="SELL", tx="0x02"),
        row(_ts(1, 0, 30), "price_change", price=0.40, size=1.0, side="BUY", best_bid=0.40, best_ask=0.42),
        row(_ts(1, 2), "price_change", price=0.44, size=1.0, side="BUY", best_bid=0.44, best_ask=0.46),
        row(_ts(1, 30), "last_trade_price", price=0.45, size=1.0, side="BUY", tx="0x03"),
    ]
    files = []
    for h in (0, 1):
        sub = [r for r in rows if r["ts"].hour == h]
        df = pl.DataFrame({
            "timestamp_received": pl.Series([r["ts"] for r in sub], dtype=pl.Datetime("ms")).dt.replace_time_zone("UTC"),
            "timestamp": pl.Series([r["ts"] for r in sub], dtype=pl.Datetime("ms")).dt.replace_time_zone("UTC"),
            "market": pl.Series([cond.encode("ascii")] * len(sub), dtype=pl.Binary),
            "event_type": [r["event_type"] for r in sub], "asset_id": [r["asset_id"] for r in sub],
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
        df.write_parquet(p); files.append(str(p))
    return files


def test_prevailing_quote_carries_across_hours_and_forward_mid_is_five_minutes_later(tmp_path):
    pq = dr.prints_with_quotes("2026-05-13", files=write_hours(tmp_path))
    by = {r["transaction_hash"]: r for r in pq.to_dicts()}
    assert by["0x01"]["bid"] == pytest.approx(0.48) and by["0x01"]["quote_ts"] == _ts(0, 2)
    assert by["0x01"]["mid_fwd"] == pytest.approx(0.50)            # no newer quote by 00:09 -> still 0.48/0.52
    assert by["0x02"]["bid"] == pytest.approx(0.48)                 # last quote of hour 0
    assert by["0x02"]["mid_fwd"] == pytest.approx(0.45)            # at 01:03 the 01:02 quote 0.44/0.46 prevails
    assert by["0x03"]["bid"] == pytest.approx(0.44) and by["0x03"]["quote_ts"] == _ts(1, 2)
    assert pq.height == 3 and set(pq.columns) >= {"asset_id", "transaction_hash", "side", "price", "size", "ts", "bid", "ask", "quote_ts", "mid_fwd"}


def test_prevailing_quote_is_strictly_earlier_than_the_print_and_a_lag_grid_is_attached(tmp_path):
    """A quote stamped at the print's own millisecond is the post-trade book from the same batch; it must not be used."""
    files = write_hours(tmp_path)
    # add a quote at exactly the 00:04:00.000 print time with a different bid, in hour 0
    import pyarrow.parquet as pq
    t = pl.read_parquet(files[0])
    same = t.filter(pl.col("event_type") == "price_change").head(1).with_columns(
        pl.lit(T0.replace(minute=4)).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC").alias("timestamp_received"),
        pl.lit(0.30).cast(pl.Decimal(9, 4)).alias("best_bid"), pl.lit(0.32).cast(pl.Decimal(9, 4)).alias("best_ask"))
    pl.concat([t, same]).write_parquet(files[0])
    pq_ = dr.prints_with_quotes("2026-05-13", files=files)
    r = {x["transaction_hash"]: x for x in pq_.to_dicts()}
    assert r["0x01"]["bid"] == pytest.approx(0.48) and r["0x01"]["quote_ts"] == _ts(0, 2)     # not the same-ms 0.30 quote
    assert r["0x01"]["same_ms_quote"] is True and r["0x03"]["same_ms_quote"] is False
    for lag in dr.LAG_GRID_MS:
        assert f"bid_lag{lag}" in pq_.columns and f"ask_lag{lag}" in pq_.columns
    # with a 1-second lag the 00:58 print still sees the 00:02 quote; with 1000 ms the 01:30 print sees 01:02
    assert r["0x03"]["bid_lag1000"] == pytest.approx(0.44)

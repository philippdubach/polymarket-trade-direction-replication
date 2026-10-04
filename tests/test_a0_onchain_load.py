"""Loader over the extended scrape: fills, taker legs and per-transaction match types."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import a0_onchain_load as ld  # noqa: E402

EX = "0xe111180000d2663c0091e4f400237545b87b996b"
T = 1778630400  # 2026-05-13 00:00:00 UTC


def _row(kind, tx, idx, maker, taker, side, token, ma, ta, fee="0", ts=T, oh="0x" + "11" * 32):
    return dict(kind=kind, block_number=1, block_ts=ts, tx_hash=tx, log_index=idx, contract=EX, order_hash=oh,
                maker=maker, taker=taker, side=side, token_id=token, maker_amount=ma, taker_amount=ta, fee=fee)


def write_day(d: Path) -> Path:
    out = d / "day=2026-05-13"
    out.mkdir()
    rows = [
        # tx A: complementary match — taker BUY token 1, one maker SELL token 1
        _row("orderfilled", "0xa", 1, "0xm1", "0xt1", 1, "1", "50", "100"),          # maker leg (maker sells 1: side of maker order)
        _row("orderfilled", "0xa", 2, "0xt1", EX, 0, "1", "100", "50", fee="7"),     # taker leg
        _row("ordersmatched", "0xa", 3, "0xt1", None, 0, "1", "100", "50"),
        # tx B: mint — taker BUY token 1, two makers BUY token 2 (the complement)
        _row("orderfilled", "0xb", 1, "0xm2", "0xt2", 0, "2", "30", "60"),
        _row("orderfilled", "0xb", 2, "0xm3", "0xt2", 0, "2", "20", "40"),
        _row("orderfilled", "0xb", 3, "0xt2", EX, 0, "1", "100", "50", ts=T + 5),
        # tx C: merge — taker SELL token 2, one maker SELL token 1
        _row("orderfilled", "0xc", 1, "0xm4", "0xt3", 1, "1", "50", "25"),
        _row("orderfilled", "0xc", 2, "0xt3", EX, 1, "2", "50", "25", ts=T + 3601),
        # tx D: mixed — taker BUY token 1, one maker SELL token 1 and one maker BUY token 2
        _row("orderfilled", "0xd", 1, "0xm5", "0xt4", 1, "1", "10", "20"),
        _row("orderfilled", "0xd", 2, "0xm6", "0xt4", 0, "2", "10", "20"),
        _row("orderfilled", "0xd", 3, "0xt4", EX, 0, "1", "40", "20"),
        _row("feecharged", "0xa", 9, "0xfee", None, None, None, None, None, fee="7"),
    ]
    pl.DataFrame(rows, schema=ld.SCHEMA).write_parquet(out / "blocks_1_1000.parquet")
    return d


def test_fills_returns_orderfilled_rows_with_utc_time_and_bucket(tmp_path, monkeypatch):
    monkeypatch.setattr(ld, "OUT", write_day(tmp_path))
    f = ld.fills("2026-05-13")
    assert f.height == 10 and set(f["kind"].to_list()) == {"orderfilled"}
    assert f["ts"].dtype == pl.Datetime("ms") and f["ts"].min() == dt.datetime(2026, 5, 13, 0, 0, 0)
    assert f.filter(pl.col("tx_hash") == "0xb")["bucket"].max() == dt.datetime(2026, 5, 13, 0, 0, 5)


def test_taker_legs_is_one_row_per_transaction(tmp_path, monkeypatch):
    monkeypatch.setattr(ld, "OUT", write_day(tmp_path))
    t = ld.taker_legs("2026-05-13")
    assert t.height == 4 and t["tx_hash"].n_unique() == 4
    assert set(t["taker"].to_list()) == {EX}
    assert t.filter(pl.col("tx_hash") == "0xa").row(0, named=True)["side"] == 0


def test_match_types_classify_complementary_mint_merge_mixed(tmp_path, monkeypatch):
    monkeypatch.setattr(ld, "OUT", write_day(tmp_path))
    complement = {"1": "2", "2": "1"}
    m = ld.match_types("2026-05-13", complement)
    got = dict(zip(m["tx_hash"].to_list(), m["match_type"].to_list()))
    assert got == {"0xa": "COMPLEMENTARY", "0xb": "MINT", "0xc": "MERGE", "0xd": "MIXED"}
    b = m.filter(pl.col("tx_hash") == "0xb").row(0, named=True)
    assert b["maker_legs"] == 2 and b["taker_side"] == 0 and b["taker_token"] == "1"


def test_match_types_without_complement_map_falls_back_to_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(ld, "OUT", write_day(tmp_path))
    m = ld.match_types("2026-05-13", {})
    assert set(m["match_type"].to_list()) == {"COMPLEMENTARY", "UNKNOWN"}


def test_legacy_columns_match_the_pilot_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(ld, "OUT", write_day(tmp_path))
    f = ld.onchain_fills("2026-05-13")
    assert {"asset_id", "chain_side", "block_number", "chain_fee_raw", "bucket"} <= set(f.columns)
    assert f.height == 10

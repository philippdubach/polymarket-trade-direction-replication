"""Fill-level identity between feed prints and on-chain taker legs."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import a1_fill_match as fm  # noqa: E402

T = dt.datetime(2026, 5, 13, 10, 0, 0)


def prints():
    return pl.DataFrame({
        "asset_id": ["1", "1", "2", "3", "4"],
        "transaction_hash": ["0xa", "0xb", "0xc", "0xnone", "0xe"],
        "side": ["BUY", "SELL", "BUY", "BUY", "SELL"],
        "price": [0.51, 0.40, 0.70, 0.10, 0.20],
        "size": [100.0, 50.0, 10.0, 5.0, 1.0],
        "ts": [T, T + dt.timedelta(seconds=1), T + dt.timedelta(hours=1), T, T + dt.timedelta(seconds=30)],
    })


def taker():
    return pl.DataFrame({
        "tx_hash": ["0xa", "0xb", "0xc", "0xd", "0xf"],
        "token_id": ["1", "1", "2", "9", "4"],
        "side": [0, 1, 1, 0, 1],                        # 0xc disagrees on side
        # BUY: maker_amount is collateral, taker_amount shares. SELL: maker_amount shares, taker_amount collateral.
        "maker_amount": ["51000000", "50000000", "10000000", "1000000", "1000000"],
        "taker_amount": ["100000000", "20000000", "7000000", "2000000", "200000"],
        "ts": [T, T + dt.timedelta(seconds=1), T + dt.timedelta(hours=1), T, T + dt.timedelta(seconds=40)],
        "contract": ["std", "std", "neg", "std", "std"],
    })


def test_join_by_hash_keeps_one_row_per_print_and_marks_matches():
    j = fm.join_by_hash(prints(), taker())
    assert j.height == 5
    assert j.filter(pl.col("matched"))["transaction_hash"].sort().to_list() == ["0xa", "0xb", "0xc"]


def test_identity_stats_count_side_token_price_size_agreement():
    j = fm.join_by_hash(prints(), taker())
    s = fm.identity_stats(j)
    assert s["matched"] == 3
    assert s["side_equal"] == 2 and s["token_equal"] == 3
    assert s["price_within_5e4"] == 3 and s["size_within_1e6"] == 3
    assert s["abs_dprice_p50"] == 0.0 and s["abs_dprice_p99"] == pytest.approx(0.0, abs=1e-9)


def test_coverage_both_ways_and_by_hour():
    j = fm.join_by_hash(prints(), taker())
    c = fm.coverage(prints(), taker(), j)
    assert c["prints"] == 5 and c["taker_legs"] == 5
    assert c["print_matched_share"] == pytest.approx(3 / 5)
    assert c["taker_matched_share"] == pytest.approx(3 / 5)
    assert c["by_print_hour"]["10"]["prints"] == 4 and c["by_print_hour"]["11"]["matched"] == 1
    assert c["by_contract"]["neg"]["taker_legs"] == 1


def test_fuzzy_match_pairs_unmatched_by_asset_side_price_size_within_120s():
    j = fm.join_by_hash(prints(), taker())
    done = j.filter(pl.col("matched"))["transaction_hash"].to_list()
    f = fm.fuzzy_match(prints().filter(~pl.col("transaction_hash").is_in(done)),
                       taker().filter(~pl.col("tx_hash").is_in(done)))
    # print 0xe (asset 4, SELL, 0.20, 1.0 at T+30) vs taker 0xf (asset 4, SELL, price 0.2, size 1.0 at T+40)
    assert f["fuzzy_matched"] == 1 and f["pairs"][0] == ("0xe", "0xf")


def test_price_and_size_from_amounts_follow_the_side():
    t = taker()
    px, sz = fm.price_size(t)
    assert px.to_list() == pytest.approx([0.51, 0.40, 0.70, 0.50, 0.20])
    assert sz.to_list() == pytest.approx([100.0, 50.0, 10.0, 2.0, 1.0])


def test_coverage_excludes_the_next_day_tail_and_reports_hour_zero():
    t = taker().with_columns(pl.Series("ts", [T, T + dt.timedelta(seconds=1), T + dt.timedelta(hours=1), T, dt.datetime(2026, 5, 14, 0, 3)]))
    j = fm.join_by_hash(prints(), t)
    c = fm.coverage(prints(), t, j, day="2026-05-13")
    assert c["taker_legs_within_day"] == 4 and c["taker_legs"] == 5
    assert c["taker_matched_share_within_day"] == pytest.approx(3 / 4)
    assert c["hour0_share_of_unmatched_within_day"] == pytest.approx(0.0)


def test_identity_by_settlement_type_and_duplicate_prints():
    p = pl.concat([prints(), prints().head(1)])          # the first print appears twice under one hash
    j = fm.join_by_hash(p, taker())
    types = pl.DataFrame({"tx_hash": ["0xa", "0xb", "0xc"], "match_type": ["MINT", "COMPLEMENTARY", "MINT"]})
    s = fm.identity_by_type(j, types)
    assert s["MINT"]["matched"] == 3 and s["MINT"]["side_equal"] == 2 and s["COMPLEMENTARY"]["side_equal"] == 1
    assert fm.duplicate_prints(p) == {"prints": 6, "distinct_hashes": 5, "duplicates": 1}


def test_unknown_cause_counts_maker_tokens_absent_from_the_map():
    mt = pl.DataFrame({"tx_hash": ["0xa", "0xb"], "match_type": ["UNKNOWN", "UNKNOWN"], "unknown_tokens": [["9"], ["1"]]})
    known = {"1", "2"}
    r = fm.unknown_cause(mt, known)
    assert r["unknown_txs"] == 2 and r["with_token_absent_from_map"] == 1

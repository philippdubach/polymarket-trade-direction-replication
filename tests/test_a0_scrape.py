"""Decoders and slice bookkeeping for the extended on-chain scrape."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import a0_scrape_days as a0  # noqa: E402

EXCH = "0xe111180000d2663c0091e4f400237545b87b996b"


def _w(n: int) -> str:
    return f"{n:064x}"


def _log(topics, words, block=0x5000000, ts=0x68000000, idx=7):
    return {
        "address": EXCH, "blockNumber": hex(block), "blockTimestamp": hex(ts), "logIndex": hex(idx),
        "transactionHash": "0x" + "ab" * 32, "topics": topics, "data": "0x" + "".join(words),
    }


def test_decode_orderfilled_v2_reads_side_token_amounts_fee():
    order = "0x" + "11" * 32
    maker = "0x" + "00" * 12 + "22" * 20
    taker = "0x" + "00" * 12 + "33" * 20
    log = _log([a0.TOPIC_ORDERFILLED, order, maker, taker],
               [_w(1), _w(987654321), _w(5_000_000), _w(10_000_000), _w(1234), _w(0)])
    r = a0.decode_log(log)
    assert r["kind"] == "orderfilled"
    assert r["order_hash"] == order and r["maker"] == "0x" + "22" * 20 and r["taker"] == "0x" + "33" * 20
    assert r["side"] == 1 and r["token_id"] == "987654321"
    assert r["maker_amount"] == "5000000" and r["taker_amount"] == "10000000" and r["fee"] == "1234"
    assert r["block_number"] == 0x5000000 and r["block_ts"] == 0x68000000 and r["log_index"] == 7
    assert r["contract"] == EXCH and r["tx_hash"] == "0x" + "ab" * 32


def test_decode_ordersmatched_reads_taker_side_and_taker_order():
    order = "0x" + "44" * 32
    tmaker = "0x" + "00" * 12 + "55" * 20
    log = _log([a0.TOPIC_ORDERSMATCHED, order, tmaker], [_w(0), _w(42), _w(7), _w(8)])
    r = a0.decode_log(log)
    assert r["kind"] == "ordersmatched"
    assert r["order_hash"] == order and r["maker"] == "0x" + "55" * 20
    assert r["side"] == 0 and r["token_id"] == "42" and r["maker_amount"] == "7" and r["taker_amount"] == "8"


def test_decode_feecharged_reads_payer_and_amount():
    payer = "0x" + "00" * 12 + "66" * 20
    log = _log([a0.TOPIC_FEECHARGED, payer], [_w(999)])
    r = a0.decode_log(log)
    assert r["kind"] == "feecharged" and r["maker"] == "0x" + "66" * 20 and r["fee"] == "999"


def test_decode_unknown_topic_returns_none():
    assert a0.decode_log(_log(["0x" + "ff" * 32], [])) is None


def test_slices_cover_range_exactly_once():
    s = a0.slices(100, 2_350, 1_000)
    assert s == [(100, 1_099), (1_100, 2_099), (2_100, 2_350)]


def test_slice_is_done_only_when_index_hash_matches(tmp_path):
    p = tmp_path / "s.parquet"
    p.write_bytes(b"data")
    idx = {"s.parquet": {"sha256": a0.sha256_file(p)}}
    assert a0.slice_done(p, idx)
    p.write_bytes(b"other")
    assert not a0.slice_done(p, idx)
    assert not a0.slice_done(tmp_path / "missing.parquet", idx)


def test_day_index_roundtrip(tmp_path):
    idx = tmp_path / "_index.json"
    a0.index_update(idx, "s1.parquet", {"sha256": "x", "rows": {"orderfilled": 3}})
    a0.index_update(idx, "s2.parquet", {"sha256": "y", "rows": {"orderfilled": 1}})
    assert json.loads(idx.read_text()) == {"s1.parquet": {"sha256": "x", "rows": {"orderfilled": 3}},
                                           "s2.parquet": {"sha256": "y", "rows": {"orderfilled": 1}}}


def _hammer(args):
    idx, worker, n = args
    for i in range(n):
        a0.index_update(idx, f"w{worker}_s{i}.parquet", {"sha256": "x", "rows": {"orderfilled": i}})
    return worker


def test_index_update_is_safe_under_concurrent_workers(tmp_path):
    from multiprocessing import Pool

    idx = tmp_path / "_index.json"
    with Pool(4) as pool:
        pool.map(_hammer, [(idx, w, 40) for w in range(4)])
    got = json.loads(idx.read_text())
    assert len(got) == 160, f"lost {160 - len(got)} entries to the race"


def test_decode_orderfilled_v1_derives_side_and_token_from_asset_ids():
    order = "0x" + "11" * 32
    maker = "0x" + "00" * 12 + "22" * 20
    taker = "0x" + "00" * 12 + "33" * 20
    # maker paid collateral (makerAssetId 0) for token 987: the order is a BUY of 987
    log = _log([a0.TOPIC_ORDERFILLED_V1, order, maker, taker], [_w(0), _w(987), _w(5_000_000), _w(10_000_000), _w(12)])
    r = a0.decode_log(log)
    assert r["kind"] == "orderfilled" and r["side"] == 0 and r["token_id"] == "987"
    assert r["maker_amount"] == "5000000" and r["taker_amount"] == "10000000" and r["fee"] == "12"
    # maker gave token 987 for collateral: a SELL of 987
    log = _log([a0.TOPIC_ORDERFILLED_V1, order, maker, taker], [_w(987), _w(0), _w(10_000_000), _w(5_000_000), _w(0)])
    r = a0.decode_log(log)
    assert r["side"] == 1 and r["token_id"] == "987"


def test_contracts_for_era():
    assert a0.contracts_for("v2") == [a0.V2_STD, a0.V2_NEG]
    assert a0.contracts_for("v1") == [a0.V1_STD, a0.V1_NEG]


def test_search_bracket_is_ordered_on_both_sides_of_the_anchor():
    import datetime as dt
    for day in ("2026-04-26", "2026-04-29", "2026-05-13", "2026-08-05"):
        lo, hi = a0.search_bracket(dt.date.fromisoformat(day), latest=95_000_000)
        assert lo < hi, day
    lo, hi = a0.search_bracket(dt.date(2026, 4, 29), latest=95_000_000)
    assert lo <= a0.ANCHOR_BLOCK <= hi
    lo, hi = a0.search_bracket(dt.date(2026, 4, 26), latest=95_000_000)
    assert hi <= a0.ANCHOR_BLOCK and lo < 86_020_778 < hi     # the pilot's verified 26 April start block

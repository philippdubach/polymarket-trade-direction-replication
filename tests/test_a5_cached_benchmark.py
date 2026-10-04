"""History and scoring sample tests for the cached-only classifier benchmark."""
import datetime as dt
import json
import sys
from pathlib import Path

import polars as pl
import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))
import a5_cached_benchmark as bm
from a2_direction_rules import bvc, sign_rules

T = dt.datetime(2026, 5, 13, 10)


def prints(prices=(.5, .6, .55), seconds=(0, 1, 5)):
    n = len(prices)
    return pl.DataFrame({"asset_id": ["a"] * n, "ts": [T + dt.timedelta(seconds=s) for s in seconds],
                         "price": prices, "size": [2.] * n, "side": ["BUY"] + ["SELL"] * (n - 1),
                         "bid": [.49] * n, "ask": [.61] * n,
                         "transaction_hash": [f"h{i}" for i in range(n)]})


def takers():
    return pl.DataFrame({"tx_hash": ["h0", "h2"], "token_id": ["a", "a"], "side": [0, 1],
                         "maker_amount": [500000., 2000000.], "taker_amount": [1000000., 1100000.],
                         "ts": [T, T + dt.timedelta(seconds=5)]})


def test_unmatched_intervening_print_changes_next_matched_tick():
    full, _ = bm.assemble_matched(prints(), takers(), "full")
    selected, _ = bm.assemble_matched(prints(), takers(), "selected")
    assert full["tick"].to_list() == [None, -1]
    assert selected["tick"].to_list() == [None, 1]
    pair = bm.paired_history(full, selected)["tick"]
    assert pair["paired_signed_rows"] == 1
    assert pair["sign_changed"] == 1
    assert pair["full_history"]["accuracy"] == 1
    assert pair["selected_history"]["accuracy"] == 0


def test_bvc_includes_unmatched_bar_closing_price_and_volume():
    full = bm.stable_signs(prints())
    selected = bm.stable_signs(prints().filter(pl.col("transaction_hash") != "h1"))
    assert full["bvc_5s"].to_list() == [None, None, -1]
    assert selected["bvc_5s"].to_list() == [None, 1]
    direct = bvc(prints()).sort(["asset_id", "bar"])
    assert direct["volume"].to_list() == [4., 2.]
    assert full["buy_share_bvc"][2] == pytest.approx(direct["buy_share_bvc"][1])


def test_abstentions_change_common_sample_and_preserve_denominator():
    d = pl.DataFrame({"taker_side": [0, 1, 0], "lr": [1, -1, 1], "emo": [1, -1, 1],
                      "tick": [None, -1, 1], "bvc_5s": [1, None, -1]})
    scores = bm.history_scores(d)
    assert scores["own_sample"]["lr"]["n"] == 3
    assert scores["common_rows"] == 1
    assert all(scores["common_sample"][r]["n"] == 1 for r in bm.RULES)
    assert scores["common_sample"]["lr"]["coverage_of_matched"] == pytest.approx(1 / 3)
    assert scores["common_sample"]["lr"]["truth_buys"] == 1
    assert scores["common_sample"]["lr"]["truth_sells"] == 0
    assert scores["common_sample"]["lr"]["balanced_accuracy"] is None
    json.dumps(scores, allow_nan=False)


def test_tied_cache_rows_have_deterministic_signs_and_alignment():
    p = prints((.5, .6, .55), (0, 0, 0)).with_row_index(bm.ROW)
    a = bm.stable_signs(p).sort(bm.ROW)
    b = bm.stable_signs(p.reverse()).sort(bm.ROW)
    assert a["tick"].to_list() == [None, 1, -1]
    assert a.select(bm.ROW, *bm.RULES).equals(b.select(bm.ROW, *bm.RULES))
    full, audit = bm.assemble_matched(p, takers())
    assert full[bm.ROW].to_list() == [0, 2]
    assert full["tied_print"].to_list() == [True, True]
    assert audit["cached_tied_prints"] == 3


def test_missing_cache_fails_before_any_loader_or_extractor(tmp_path, monkeypatch):
    monkeypatch.setattr(bm.ld, "taker_legs", lambda _: pytest.fail("must fail before loading"))
    with pytest.raises(FileNotFoundError):
        bm.load_signed_matched("2026-05-13", ext=tmp_path)
    (tmp_path / "prints").mkdir()
    prints().write_parquet(tmp_path / "prints" / "day=2026-05-13.parquet")
    with pytest.raises(FileNotFoundError):
        bm.load_signed_matched("2026-05-13", ext=tmp_path)


def test_identity_mismatch_and_ambiguous_taker_fail_closed():
    with pytest.raises(ValueError, match="identity audit"):
        bm.assemble_matched(prints(), takers().with_columns(pl.lit("wrong").alias("token_id")))
    conflicting = pl.concat([takers(), takers().head(1).with_columns(pl.lit(1, dtype=pl.Int64).alias("side"))])
    with pytest.raises(ValueError, match="ambiguous taker"):
        bm.assemble_matched(prints(), conflicting)


def test_paired_history_rejects_missing_row_alignment():
    full, _ = bm.assemble_matched(prints(), takers())
    with pytest.raises(ValueError, match="identical matched row ids"):
        bm.paired_history(full, full.head(1))


def test_lag_sensitivity_uses_one_fixed_common_sample():
    d, _ = bm.assemble_matched(prints(), takers())
    for lag in bm.LAG_GRID_MS:
        d = d.with_columns(pl.lit(.49).alias(f"bid_lag{lag}"), pl.lit(.61).alias(f"ask_lag{lag}"))
    out = bm.lag_sensitivity(d)
    assert out["available"]
    assert len({score["n"] for score in out["scores"].values()}) == 1
    assert len({score["truth_buys"] for score in out["scores"].values()}) == 1


def test_duplicate_row_ids_rejected():
    with pytest.raises(ValueError, match="row ids"):
        bm.stable_signs(prints().with_columns(pl.lit(0).alias(bm.ROW)))


def test_legacy_rule_definitions_preserved_for_untied_multi_asset_history():
    p = prints((.5, .6, .55, .51, .57, .58), (0, 1, 5, 11, 17, 25))
    other = p.with_columns(pl.lit("b").alias("asset_id"), (pl.col("price") + .1).alias("price"))
    p = pl.concat([p, other]).with_row_index(bm.ROW)
    out = bm.stable_signs(p).sort(bm.ROW)
    legacy = sign_rules(p).sort(bm.ROW)
    assert out.select("lr", "emo", "tick").equals(legacy.select("lr", "emo", "tick"))
    bc = bvc(p).select("asset_id", "bar", "buy_share_bvc")
    direct = (p.with_columns(pl.col("ts").dt.truncate("5s").alias("bar"))
              .join(bc, on=["asset_id", "bar"], how="left").sort(bm.ROW))
    assert out["buy_share_bvc"].to_list() == pytest.approx(direct["buy_share_bvc"].to_list())


def test_common_two_class_confusions_identical_truth_and_expected_balance():
    d = pl.DataFrame({"taker_side": [0, 0, 1, 1, 0], "lr": [1, 1, -1, 1, 1],
                      "emo": [1, -1, -1, 1, 1], "tick": [1, -1, -1, -1, None],
                      "bvc_5s": [1, 1, 1, -1, 1]})
    result = bm.history_scores(d)["common_sample"]
    for r in bm.RULES:
        assert result[r]["n"] == 4
        assert result[r]["truth_buys"] == result[r]["truth_sells"] == 2
    assert result["lr"]["balanced_accuracy"] == .75
    assert result["emo"]["balanced_accuracy"] == .5
    assert result["tick"]["balanced_accuracy"] == .75
    assert result["bvc_5s"]["balanced_accuracy"] == .75

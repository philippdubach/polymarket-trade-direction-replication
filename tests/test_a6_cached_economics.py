"""Shared-row economic accounting, cached horizon and robustness tests."""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))
import a6_cached_economics as ec

DAY = "2026-05-13"
T = dt.datetime.fromisoformat(DAY) + dt.timedelta(hours=10)


def frame(n=2, **columns):
    base = {"asset_id": ["a"] * n, "transaction_hash": [f"h{i}" for i in range(n)],
            "cache_row_id": list(range(n)), "ts": [T] * n,
            "quote_ts": [T - dt.timedelta(milliseconds=500)] * n,
            "bid": [.4] * n, "ask": [.6] * n, "mid": [.5] * n,
            "price": [.55] * n, "mid_fwd": [.51] * n, "size": [1.] * n,
            "taker_side": [0] * n, "tied_print": [False] * n,
            "chain_price": [.55] * n, **{r: [1] * n for r in ec.RULES}}
    base.update(columns)
    return pl.DataFrame(base)


def test_common_rows_exclude_abstention_before_every_rule_and_preserve_funnel():
    d = frame(3, bvc_5s=[1, None, -1], tick=[1, 1, -1], taker_side=[0, 0, 1])
    out = ec.analyze_matched(d, DAY)["primary"]
    assert out["sample"]["rows"] == 2
    assert out["sample"]["truth_buys"] == out["sample"]["truth_sells"] == 1
    assert out["funnel"][1] == {"step": "all_four_rule_signs_and_valid_truth", "before": 3, "after": 2, "excluded": 1}
    for w in ec.WEIGHTS:
        assert all(v["flip_rows"] <= 2 for v in out["weights"][w]["rules"].values())
    assert out["weights"]["fill"]["taker"]["effective"] == pytest.approx(0.)
    with pytest.raises(ValueError, match="common valid sample"):
        ec.paired_accounting(d)


def test_exact_2355_cutoff_and_day_boundaries_are_exclusive():
    start = dt.datetime.fromisoformat(DAY)
    end = start + dt.timedelta(days=1)
    cutoff = end - dt.timedelta(minutes=5)
    times = [start, cutoff - dt.timedelta(microseconds=1), cutoff,
             end - dt.timedelta(microseconds=1), end, start - dt.timedelta(microseconds=1)]
    sample, funnel = ec.eligible_sample(frame(6, ts=times), DAY)
    assert sample["cache_row_id"].to_list() == [0, 1]
    assert funnel[-1]["excluded"] == 4


@pytest.mark.parametrize("field,value", [("price", np.inf), ("price", np.nan), ("price", None),
    ("price", -0.1), ("price", 1.1), ("bid", 0.), ("ask", 1.), ("bid", .7),
    ("bid", np.inf), ("ask", np.nan), ("bid", None), ("mid_fwd", np.nan),
    ("mid_fwd", np.inf), ("mid_fwd", None), ("mid_fwd", 1.1), ("size", 0.),
    ("size", -1.), ("size", np.inf), ("size", np.nan), ("size", None)])
def test_nonfinite_endpoints_crossed_missing_and_invalid_sizes_excluded(field, value):
    d = frame(**{field: [value, .5 if field != "size" else 2.]})
    sample, _ = ec.eligible_sample(d, DAY)
    assert sample["cache_row_id"].to_list() == [1]


def test_probability_price_endpoints_and_interior_locked_quotes_are_valid():
    d = frame(price=[0., 1.], mid_fwd=[0., 1.], bid=[.5, .5], ask=[.5, .5])
    sample, _ = ec.eligible_sample(d, DAY)
    assert sample.height == 2
    out = ec.paired_accounting(sample)
    assert out["identity_max_abs_error_per_fill"] < 1e-12


def test_unequal_volumes_reverse_sign_and_exact_error_mass_identities():
    d = frame(price=[.55, .45], mid_fwd=[.51, .49], size=[1., 9.], taker_side=[0, 1],
              lr=[1, 1], emo=[1, -1], tick=[-1, 1], bvc_5s=[-1, -1])
    out = ec.paired_accounting(d)
    fill, vol = (out["weights"][w] for w in ec.WEIGHTS)
    assert fill["taker"]["effective"] == pytest.approx(.1)
    assert fill["rules"]["lr"]["means"]["effective"] == pytest.approx(0.)
    assert vol["rules"]["lr"]["means"]["effective"] == pytest.approx(-.08)
    assert fill["rules"]["lr"]["flip_share"] == pytest.approx(.5)
    assert vol["rules"]["lr"]["flip_share"] == pytest.approx(.9)
    assert vol["rules"]["lr"]["error_diagnostics"]["effective"]["opposite_sign_of_taker_mean"]
    for weight in ec.WEIGHTS:
        panel = out["weights"][weight]
        tm = panel["taker"]
        assert tm["effective"] == pytest.approx(tm["realised"] + tm["impact"])
        for r in ec.RULES:
            rr = panel["rules"][r]
            mm = rr["means"]
            assert mm["effective"] == pytest.approx(mm["realised"] + mm["impact"])
            for m in ec.MEASURES:
                mass = rr["error_diagnostics"][m]["truth_signed_error_mass"]
                assert rr["difference_rule_minus_taker"][m] == pytest.approx(-2 * mass)
                absolute_mass = rr["error_diagnostics"][m]["truth_absolute_error_mass"]
                assert rr["mean_absolute_paired_distance"][m] == pytest.approx(2 * absolute_mass)
    json.dumps(out, allow_nan=False)


def test_cancellation_residue_is_numerical_zero_not_sign_reversal():
    d = frame(price=[.51, .49], mid_fwd=[.55, .45], taker_side=[0, 1],
              **{r: [-1, -1] for r in ec.RULES})
    out = ec.paired_accounting(d)
    for w in ec.WEIGHTS:
        for r in ec.RULES:
            rr = out["weights"][w]["rules"][r]
            assert all(abs(rr["means"][m]) <= ec.SIGN_ZERO_TOLERANCE for m in ec.MEASURES)
            assert not any(rr["error_diagnostics"][m]["opposite_sign_of_taker_mean"] for m in ec.MEASURES)
            assert all(rr["error_diagnostics"][m]["mean_sign_changed"] for m in ec.MEASURES)
    # The same numerical-zero convention must hold for asset means.
    aa = ec.asset_exploratory(d, minimum=2)
    for w in ec.WEIGHTS:
        for r in ec.RULES:
            assert all(aa["weights"][w][r][m]["rule_zero_mean_assets"] == 1 for m in ec.MEASURES)
            assert all(aa["weights"][w][r][m]["sign_reversal_assets"] == 0 for m in ec.MEASURES)


def test_constant_inputs_have_no_error_outcome_correlation_under_volume_weights():
    # A constant indicator remains undefined when weight normalization leaves
    # tiny summation residue; that residue must not create a Pearson variance.
    w = np.array([1., 3., 7.]) / 11
    assert ec.weighted_corr(np.ones(3), np.array([.1, .2, .3]), w) is None
    assert ec.weighted_corr(np.array([0., 1., 1.]), np.ones(3), w) is None
    assert ec.weighted_corr(np.array([0., 1., 1.]), np.array([0., 1., 1.]), w) == pytest.approx(1.)


def test_asset_ranking_reversals_same_assets_and_minimum_fills():
    assets = ["a", "a", "b", "b", "c", "c", "too_small"]
    d = frame(7, asset_id=assets, price=[.55, .55, .6, .6, .65, .65, .7],
              ask=[.8] * 7, **{r: [-1] * 7 for r in ec.RULES})
    out = ec.asset_exploratory(d, minimum=2)
    assert out["assets"] == 3
    assert out["common_fills_in_retained_assets"] == 6
    for w in ec.WEIGHTS:
        for r in ec.RULES:
            ef = out["weights"][w][r]["effective"]
            assert ef["spearman_rank_correlation"] == pytest.approx(-1.)
            assert ef["sign_reversal_assets"] == 3
            assert ef["sign_reversal_share"] == 1
            # Identical midpoint impacts have undefined rank correlation.
            assert out["weights"][w][r]["impact"]["spearman_rank_correlation"] is None


def test_asset_rank_inputs_restore_ties_from_floating_aggregation_residue():
    # Mathematical grid-equivalent costs can differ by tiny floating residue.
    # Raw accounting retains it; ranking must not invent an ordering of ties.
    d = frame(6, asset_id=["a", "a", "b", "b", "c", "c"],
              price=[.55, .55 + 1e-15, .55 - 1e-15, .55, .55, .55 + 2e-15],
              lr=[1, 1, -1, -1, 1, 1])
    out = ec.asset_exploratory(d, minimum=2)
    assert out["assets"] == 3
    for w in ec.WEIGHTS:
        for r in ec.RULES:
            assert out["weights"][w][r]["effective"]["spearman_rank_correlation"] is None


def test_fresh_quote_sensitivity_rejects_future_missing_and_stale_quotes():
    quotes = [T, T - dt.timedelta(seconds=1), T - dt.timedelta(microseconds=1000001),
              T + dt.timedelta(microseconds=1), None]
    out = ec.analyze_matched(frame(5, quote_ts=quotes), DAY)
    assert out["primary"]["sample"]["rows"] == 5
    assert out["sensitivities"]["quote_age_1s"]["sample"]["rows"] == 2


def test_tie_and_unique_hash_sensitivities_change_only_scored_rows():
    d = frame(4, tied_print=[True, False, False, False], transaction_hash=["a", "a", "b", "b"],
              price=[.6, .55, .51, .59], lr=[-1, 1, 1, -1])
    out = ec.analyze_matched(d, DAY)
    ss = out["sensitivities"]
    assert ss["tie_exclusion"]["sample"]["rows"] == 3
    assert ss["unique_hash"]["sample"]["rows"] == 2
    assert ss["unique_hash"]["weights"]["fill"]["rules"]["lr"]["flip_rows"] == 1
    assert ss["unique_hash"]["weights"]["fill"]["taker"]["effective"] == pytest.approx(.11)


def test_chain_price_sensitivity_uses_identical_valid_subset_fixed_signs_weights():
    d = frame(3, chain_price=[.54, np.nan, .5], price=[.55, .6, .55], size=[1., 100., 3.],
              lr=[-1, -1, 1])
    ss = ec.analyze_matched(d, DAY)["sensitivities"]["chain_price"]
    assert ss["sample"] == ss["print_price_comparator"]["sample"]
    assert ss["sample"]["rows"] == 2
    for w in ec.WEIGHTS:
        a, b = ss["weights"][w], ss["print_price_comparator"]["weights"][w]
        assert a["taker"]["impact"] == b["taker"]["impact"]
        for r in ec.RULES:
            assert a["rules"][r]["flip_rows"] == b["rules"][r]["flip_rows"]
            assert a["rules"][r]["means"]["impact"] == b["rules"][r]["means"]["impact"]
            assert a["rules"][r]["flip_share"] == b["rules"][r]["flip_share"]


def test_empty_sample_undefined_summaries_are_strict_json():
    out = ec.analyze_matched(frame(lr=[None, None]), DAY)
    assert out["primary"]["sample"]["rows"] == 0
    assert out["primary"]["weights"] == {}
    assert out["primary"]["asset_exploratory"]["assets"] == 0
    json.dumps(ec.summarize([out]), allow_nan=False)


def test_input_change_from_stage2_fails_closed(tmp_path):
    prints = tmp_path / "prints"
    onchain = tmp_path / "onchain" / f"day={DAY}"
    prints.mkdir()
    onchain.mkdir(parents=True)
    p = prints / f"day={DAY}.parquet"
    q = onchain / "blocks_001.parquet"
    p.write_bytes(b"cached prints")
    q.write_bytes(b"cached truth")
    expected = {"prints": {"path": str(p), "bytes": p.stat().st_size, "sha256": ec.sha256(p)},
                "onchain_slices": [{"path": str(q), "bytes": q.stat().st_size, "sha256": ec.sha256(q)}]}
    assert ec.verify_inputs(DAY, tmp_path, expected)["verified_against_stage2"]
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed since Stage 2"):
        ec.verify_inputs(DAY, tmp_path, expected)

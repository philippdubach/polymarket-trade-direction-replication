"""Signed microstructure measures under the taker sign and under inferred signs."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import a3_signed_measures as sm  # noqa: E402

T = dt.datetime(2026, 5, 13, 10, 0, 0)


def fills():
    """Two assets; asset 1 has 4 prints, asset 2 has 2. mid_fwd is the midpoint 5 minutes later."""
    return pl.DataFrame({
        "asset_id": ["1", "1", "1", "1", "2", "2"],
        "ts": [T + dt.timedelta(seconds=i * 10) for i in range(4)] + [T, T + dt.timedelta(seconds=10)],
        "price": [0.52, 0.48, 0.53, 0.47, 0.30, 0.31],
        "mid": [0.50, 0.50, 0.50, 0.50, 0.30, 0.30],
        "mid_fwd": [0.51, 0.51, 0.49, 0.49, 0.30, 0.30],
        "size": [10.0, 10.0, 10.0, 10.0, 5.0, 5.0],
        "taker": [1, -1, 1, -1, 1, 1],
        "rule": [1, -1, -1, 1, 1, -1],      # asset 1: last two wrong; asset 2: second wrong
    })


def test_effective_spread_uses_the_sign():
    m = sm.per_fill(fills(), "taker")
    # 2 s (p - m): 0.04, 0.04, 0.06, 0.06, 0.0, 0.02
    assert m["eff"].to_list() == pytest.approx([0.04, 0.04, 0.06, 0.06, 0.0, 0.02])
    r = sm.per_fill(fills(), "rule")
    assert r["eff"].to_list() == pytest.approx([0.04, 0.04, -0.06, -0.06, 0.0, -0.02])


def test_realised_spread_and_impact_split_the_effective_spread():
    m = sm.per_fill(fills(), "taker")
    assert (m["realised"] + m["impact"]).to_list() == pytest.approx(m["eff"].to_list())
    # first fill: buy at 0.52, mid 0.50 -> 0.51: impact 2(0.51-0.50)=0.02, realised 2(0.52-0.51)=0.02
    assert m["impact"][0] == pytest.approx(0.02) and m["realised"][0] == pytest.approx(0.02)


def test_asset_medians_and_ratio_to_taker():
    out = sm.compare(fills(), ["rule"], min_fills=2)
    assert out["assets"] == 2
    assert out["taker"]["eff_median"] == pytest.approx((0.05 + 0.01) / 2)   # asset medians 0.05 and 0.01
    assert out["rule"]["eff_ratio_to_taker"] < 1.0


def test_attenuation_mapping_from_two_recalls():
    # with perfect recalls the mapping is the identity; with r_b = r_s = 0.75 it is 0.5 x
    assert sm.attenuation(1.0, 1.0, x_buy=0.6, x_sell=0.4) == pytest.approx(0.6 - 0.4)
    assert sm.attenuation(0.75, 0.75, x_buy=0.6, x_sell=0.4) == pytest.approx(0.5 * (0.6 - 0.4))


def test_wrong_side_share_counts_negative_signed_spreads():
    w = sm.wrong_side(fills(), "rule")
    # rule signs: fills 3, 4 and 6 are mis-signed -> eff -0.06, -0.06, -0.02; fill 5 has eff 0
    assert w["fills"] == 6 and w["share_negative"] == pytest.approx(3 / 6) and w["share_zero"] == pytest.approx(1 / 6)


def test_compare_reports_mean_based_ratio_too():
    out = sm.compare(fills(), ["rule"], min_fills=2)
    assert "eff_mean_ratio_to_taker" in out["rule"]
    # asset means: taker (0.05, 0.01) -> 0.03; rule (0.05 - 0.06 ... ) lower
    assert out["rule"]["eff_mean_ratio_to_taker"] < 1.0


def test_accounting_identity_decomposes_the_rule_signed_mean():
    d = fills()
    acc = sm.accounting(d, "rule")
    # rule mean eff = taker mean eff + 2 * mass of wrong-side fills under the taker sign that the rule flips
    assert acc["rule_mean_eff"] == pytest.approx(sm.per_fill(d, "rule")["eff"].mean())
    assert acc["taker_mean_eff"] == pytest.approx(sm.per_fill(d, "taker")["eff"].mean())
    assert acc["flip_mass"] == pytest.approx(acc["rule_mean_eff"] - acc["taker_mean_eff"])
    assert acc["fill_level_ratio"] == pytest.approx(acc["rule_mean_eff"] / acc["taker_mean_eff"])


def test_kyle_lambda_is_no_longer_reported():
    assert not hasattr(sm, "kyle_lambda")


def test_accounting_compares_the_same_fills_when_rule_abstains():
    d = pl.DataFrame({"price": [.9, .52, .48], "mid": [.5, .5, .5],
                      "mid_fwd": [.5] * 3, "taker": [1, 1, -1], "rule": [None, 1, -1]})
    a = sm.accounting(d, "rule")
    assert a["fills"] == 2 and a["coverage"] == pytest.approx(2 / 3)
    assert a["fill_level_ratio"] == pytest.approx(1)
    assert a["flip_mass"] == pytest.approx(0)

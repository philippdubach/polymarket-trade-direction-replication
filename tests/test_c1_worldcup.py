"""The 2 July 0.0025 change as a two-arm DiD on half-day bins."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import c1_worldcup_did as wc  # noqa: E402

T0 = dt.datetime(2026, 6, 30, 12, 0)


def test_permutation_p_is_positive_even_with_no_exceedances():
    df = halfday_panel(effect=-10, n_t=10, n_c=20)
    r = wc.randomisation_p(df, draws=99)
    assert r["p"] == (r["exceedances"] + 1) / 100
    assert r["p"] >= .01


def test_event_sensitivity_preserves_contract_point_estimate_and_counts_events():
    df = halfday_panel(n_t=20, n_c=60).with_columns(
        pl.col("market_id").str.strip_prefix("m").cast(pl.Int64).floordiv(10).cast(pl.String).alias("event_id"))
    market = wc.did_twfe(df)
    event = wc.did_twfe(df, cluster_col="event_id")
    assert event["beta"] == pytest.approx(market["beta"])
    assert event["clusters"] == 8 and market["clusters"] == 80
    boot = wc.event_bootstrap(df, "y", draws=99)
    assert boot["treated_events"] == 2 and boot["control_events"] == 6


def test_seeded_paired_and_permutation_results_ignore_input_row_order():
    df = halfday_panel()
    shuffled = df.sample(fraction=1, shuffle=True, seed=42)
    assert wc.did_paired(df, draws=99) == wc.did_paired(shuffled, draws=99)
    assert wc.randomisation_p(df, draws=99) == wc.randomisation_p(shuffled, draws=99)


def halfday_panel(effect: float = -0.3, n_t: int = 20, n_c: int = 60, seed: int = 0):
    """Treated markets' log spread drops by `effect` after the change; controls flat. Bins: 3 pre, 4 post."""
    rng = np.random.default_rng(seed)
    rows = []
    bins = [T0 + dt.timedelta(hours=12 * i) for i in range(7)]
    for m in range(n_t + n_c):
        treated = m < n_t
        alpha = rng.normal(-2.0, 0.3)
        for i, b in enumerate(bins):
            post = i >= 3
            y = alpha + (effect if treated and post else 0.0) + rng.normal(0, 0.05)
            rows.append({"market_id": f"m{m}", "asset_id": f"a{m}", "bin": b, "post": post, "treated": treated, "y": y})
    return pl.DataFrame(rows)


def test_twfe_recovers_the_effect_with_market_clustered_se():
    r = wc.did_twfe(halfday_panel(effect=-0.3))
    assert r["beta"] == pytest.approx(-0.3, abs=0.05)
    assert 0 < r["se_cluster"] < 0.05 and r["clusters"] == 80


def test_paired_median_version():
    r = wc.did_paired(halfday_panel(effect=-0.3))
    assert r["treated_change"] == pytest.approx(-0.3, abs=0.05) and abs(r["control_change"]) < 0.05
    assert r["did"] == pytest.approx(-0.3, abs=0.07)


def test_randomisation_p_is_small_for_a_real_effect_and_large_for_none():
    p1 = wc.randomisation_p(halfday_panel(effect=-0.3), draws=200, seed=1)
    p0 = wc.randomisation_p(halfday_panel(effect=0.0), draws=200, seed=1)
    assert p1["p"] < 0.05 and p0["p"] > 0.2


def test_mde_formula_on_market_level_changes_has_no_bin_term():
    m = wc.mde(sigma=0.6, n_t=40, n_c=300)
    assert m == pytest.approx(2.8 * 0.6 * (1 / 40 + 1 / 300) ** 0.5)


def test_select_controls_uses_the_fee_category_not_the_empty_event_tags():
    import datetime as dt
    import polars as pl
    g = pl.DataFrame({
        "condition_id": ["0x1", "0x2", "0x3", "0x4"],
        "token_0": ["a0", "b0", "c0", "d0"], "token_1": ["a1", "b1", "c1", "d1"],
        "fee_type": ["sports_fees_v3", "sports_fees_v3", "crypto_fees_v2", "sports_fees_v3"],
        "sports_market_type": ["moneyline", "moneyline", None, "moneyline"],
        "game_start_time": [dt.datetime(2026, 7, 3, 18)] * 4,
        "event_tags": [[], [], [], []],
        "event_slug": ["mls-2026", "fifa-world-cup-2026", "btc", "wimbledon"], "event_title": ["MLS", "FIFA World Cup", "BTC", "Wimbledon"],
        "question": ["A vs. B", "C vs. D", "BTC up", "E vs. F"],
    })
    c = wc.select_controls(g, start=dt.datetime(2026, 7, 2, 12))
    assert sorted(c["condition_id"].unique().to_list()) == ["0x1", "0x4"]
    assert c.height == 4


def test_control_change_sd_is_the_spread_of_market_level_pre_to_post_changes_among_controls():
    import polars as pl
    rows = []
    for m, t, pre, post in (("c1", False, 1.0, 1.2), ("c2", False, 2.0, 1.8), ("c3", False, 1.0, 1.0), ("t1", True, 1.0, 0.2)):
        rows += [{"market_id": m, "treated": t, "post": False, "y": pre}, {"market_id": m, "treated": t, "post": True, "y": post}]
    sd = wc.control_change_sd(pl.DataFrame(rows))
    assert abs(sd - 0.2) < 1e-9   # changes +0.2, -0.2, 0.0 -> sample sd 0.2


def test_prior_tick_defaults_to_the_listing_tick_not_the_metadata_snapshot():
    """A market with no tick event before the change was on the 0.01 tick the venue lists at; the
    metadata snapshot months later carries the post-resolution tick and must not be used."""
    import datetime as dt
    import polars as pl
    assets = pl.DataFrame({"asset_id": ["a", "b"], "condition_id": ["m1", "m2"]})
    events = pl.DataFrame({"asset_id": ["a"], "ts": [dt.datetime(2026, 6, 20)], "new_tick": [0.001]})
    gamma = pl.DataFrame({"condition_id": ["m1", "m2"], "tick_size": [0.001, 0.001]})
    out = wc.prior_tick(assets, events=events, gamma=gamma, change=dt.datetime(2026, 7, 2, 0, 31))
    assert dict(zip(out["asset_id"].to_list(), out["prior_tick"].to_list())) == {"a": 0.001, "b": 0.01}


def test_select_controls_requires_strict_game_types_and_a_game_inside_the_window():
    import datetime as dt
    import polars as pl
    g = pl.DataFrame({
        "condition_id": ["0x1", "0x2", "0x3", "0x4"],
        "token_0": ["a0", "b0", "c0", "d0"], "token_1": ["a1", "b1", "c1", "d1"],
        "fee_type": ["sports_fees_v3"] * 4,
        "sports_market_type": ["moneyline", "soccer_first_to_score", "totals", "moneyline"],
        "game_start_time": [dt.datetime(2026, 7, 3, 18), dt.datetime(2026, 7, 3, 18), dt.datetime(2026, 7, 2, 20), dt.datetime(2026, 7, 9, 18)],
        "event_tags": [[], [], [], []], "event_slug": ["mls", "mls", "nba", "mls"], "event_title": ["MLS", "MLS", "NBA", "MLS"],
        "question": ["A vs. B", "First to score", "Total O/U", "C vs. D"],
    })
    c = wc.select_controls(g, start=dt.datetime(2026, 7, 2, 12), end=dt.datetime(2026, 7, 4))
    assert sorted(c["condition_id"].unique().to_list()) == ["0x1", "0x3"]   # first-to-score excluded; 9 July game outside the window



def test_paired_did_carries_a_market_bootstrap_interval_and_the_counts_it_used():
    import polars as pl
    rows = []
    for i in range(30):
        rows += [{"market_id": f"t{i}", "treated": True, "post": False, "y": 1.0}, {"market_id": f"t{i}", "treated": True, "post": True, "y": 0.0 + 0.01 * i}]
        rows += [{"market_id": f"c{i}", "treated": False, "post": False, "y": 1.0}, {"market_id": f"c{i}", "treated": False, "post": True, "y": 0.5 + 0.01 * i}]
    r = wc.did_paired(pl.DataFrame(rows), draws=200)
    assert r["treated_markets"] == 30 and r["control_markets"] == 30
    assert r["ci95"][0] < r["did"] < r["ci95"][1] and r["ci95"][1] - r["ci95"][0] < 0.3


def test_sub_cent_controls_are_excluded_from_the_finer_arm():
    import datetime as dt
    import polars as pl
    hd = pl.DataFrame({"market_id": ["a", "a", "b", "b"], "post": [False, True, False, True], "spread_median": [0.005, 0.01, 0.01, 0.02]})
    assert wc.on_listing_grid(hd, tick=0.01) == ["b"]


def test_controls_exclude_every_treated_market():
    import polars as pl
    import c1_worldcup_did as c1
    ctrl = pl.DataFrame({"asset_id": ["a1", "a2", "b1", "b2"], "condition_id": ["A", "A", "B", "B"]})
    tsets = {"D": pl.DataFrame({"asset_id": ["a1"], "condition_id": ["A"]}),
             "U": pl.DataFrame({"asset_id": ["z1"], "condition_id": ["Z"]})}
    out = c1.exclude_treated(ctrl, tsets)
    assert sorted(out["condition_id"].unique().to_list()) == ["B"]

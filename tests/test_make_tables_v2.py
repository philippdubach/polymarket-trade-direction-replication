"""Table and figure generation checks."""
from __future__ import annotations

import sys
from pathlib import Path

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import make_tables as mt  # noqa: E402


def pooled():
    return {
        "coverage": {"2026-05-13": {"prints": 2251602, "taker_legs": 2286507, "print_matched_share": 0.9991, "taker_matched_share": 0.9838,
                                    "taker_legs_within_day": 2270000, "taker_matched_share_within_day": 0.992, "duplicate_prints": 120,
                                    "unknown_txs": 0, "unknown_with_token_absent": 0}},
        "identity": {"2026-05-13": {"side_equal_share": 1.0, "token_equal_share": 1.0, "price_within_5e4_share": 0.92, "size_within_1e6_share": 0.9996,
                                    "side_equal": 2249530, "token_equal": 2249530, "matched": 2249530}},
        "match_types": {"2026-05-13": {"MINT": 0.688, "COMPLEMENTARY": 0.208, "MIXED": 0.072, "MERGE": 0.032, "UNKNOWN": 0.0}},
        "buy_share": {"2026-05-13": {"taker_legs": 0.819, "all_legs": 0.852}},
        "fuzzy": {"2026-05-13": {"unmatched_prints": 2070, "fuzzy_matched": 7}},
        "receipts": {"2026-05-13": {"reverted": 1.0, "sampled": 892}},
        "n_days": 1,
        "balanced_accuracy": {"lr": {"days": 1, "mean": 0.92, "min": 0.92, "max": 0.92}, "tick": {"days": 1, "mean": 0.74, "min": 0.74, "max": 0.74}},
        "mcc": {"lr": {"days": 1, "mean": 0.6}, "tick": {"days": 1, "mean": 0.3}},
        "recall_sell": {"lr": {"days": 1, "mean": 0.94}, "tick": {"days": 1, "mean": 0.79}},
        "recall_buy": {"lr": {"days": 1, "mean": 0.90}, "tick": {"days": 1, "mean": 0.70}},
        "kappa": {"lr": {"days": 1, "mean": 0.7}, "tick": {"days": 1, "mean": 0.3}},
        "accuracy": {"lr": {"days": 1, "mean": 0.91}, "tick": {"days": 1, "mean": 0.62}},
        "coverage_rule": {"lr": {"days": 1, "mean": 1.0}, "tick": {"days": 1, "mean": 0.97}},
        "contrasts": {"lr_vs_tick": {"mean_diff": 0.18, "min_diff": 0.15, "max_diff": 0.2, "p": 1e-11, "p_holm": 3e-11, "sign_test": {"days": 12, "positive": 12, "p_two_sided": 0.0005}}},
        "signed_ratios": {"2026-05-13": {"wrong_side_share": 0.033, "taker_eff_mean": 0.0171, "lr_mean": 1.34, "emo_mean": 1.2, "tick_mean": 0.55,
                                         "lr_fill_ratio": 1.336, "emo_fill_ratio": 1.2, "tick_fill_ratio": 0.6}},
    }


def test_fillmatch_reports_mismatch_counts_within_day_denominator_and_unknown():
    tex = mt.tab_fillmatch(pooled())
    assert "with leg" in tex and "with print" in tex and "legs in day" in tex
    assert "1.000" not in tex.split("Panel B")[0]        # no constant 1.000 columns
    assert "2{,}270{,}000" in tex and "99.2" in tex       # within-day denominator
    assert "unknown" in tex.lower() and "2{,}070" in tex and "reverted" in tex.lower()


def test_direction_table_has_kappa_accuracy_baseline_coverage_and_sign_test():
    tex = mt.tab_direction(pooled(), rules=[("lr", "Lee--Ready"), ("tick", "tick test")])
    assert "0.700" in tex and "0.910" in tex and "0.970" in tex    # kappa, raw accuracy, coverage
    assert "12 of 12" in tex and "10^{-9}" in tex


def test_signed_table_reports_fill_level_ratio_and_taker_level():
    tex = mt.tab_signed(pooled())
    assert "1.34" in tex and "0.0171" in tex and "3.3" in tex


def test_tab_cutover_reports_paired_changes_half_life_and_placebo_ranks():
    b1 = {"cohort": 51585, "outcomes": {
        "log_spread": {"half_life_days": 3, "paths": [{"k": k, "n": 50000, "median": m, "mean": m, "lo": m - 0.001, "hi": m + 0.001}
                                                       for k, m in ((-1, 0.0), (0, 0.017), (1, 0.046), (3, 0.0), (7, 0.0))]},
        "log_updates": {"half_life_days": 7, "paths": [{"k": k, "n": 50000, "median": m, "mean": m, "lo": m, "hi": m}
                                                        for k, m in ((-1, 0.0), (0, 0.025), (1, -0.146), (3, 0.124), (7, -0.005))]}},
        "attrition": {"log_spread": {"survivors": 34379, "worst": 0.108, "best": 0.0}, "log_updates": {"survivors": 34446, "worst": 0.414, "best": 0.0}}}
    b2 = {"n_dates": 66, "ranks": {"log_spread": {"rank": 1, "n": 66, "p": 0.0152}, "log_updates": {"rank": 20, "n": 66, "p": 0.303}},
          "ranks_k": {"1": {"log_spread": {"rank": 1, "n": 66, "p": 0.0152}, "log_updates": {"rank": 3, "n": 66, "p": 0.045}}},
          "strict": {"n_dates": 33, "ranks": {"log_spread": {"rank": 1, "n": 33, "p": 0.0303}, "log_updates": {"rank": 9, "n": 33, "p": 0.273}}}}
    b1["outcomes"]["log_spread"]["paths"] = [{"k": -7, "n": 51585, "median": 0.0, "mean": 0.545, "lo": 0.127, "hi": 0.141, "net_up": 0.456},
                                              {"k": -3, "n": 51585, "median": 0.0, "mean": 0.322, "lo": 0.0, "hi": 0.0, "net_up": 0.21}] + b1["outcomes"]["log_spread"]["paths"]
    tex = mt.tab_cutover(b1, b2)
    assert "log spread" in tex and "0.017" in tex and "0.046" in tex and "[0.016, 0.018]" not in tex   # the interval is in the text
    assert "0.545" in tex and "0.456" in tex   # the pre-period mean and net share are shown, not only k >= 0
    assert "34{,}379" in tex and "0.108" in tex
    assert "rank $\\Delta_0$" not in tex   # ranks live in tab:placebo
    assert "\\label{tab:cutover}" in tex


def test_tab_placebo_reports_event_statistics_with_two_sided_ranks_and_the_strict_subset():
    b2 = {"event": "2026-04-28", "n_dates": 66,
          "stats_k": {"0": {"log_spread": {"2026-04-28": 0.017, "2026-03-10": 0.0}}, "1": {"log_spread": {"2026-04-28": 0.046, "2026-03-10": 0.0}}},
          "stats_alt": {"mean": {"0": {"log_spread": {"2026-04-28": 0.308, "2026-03-10": 0.1}}, "1": {"log_spread": {"2026-04-28": 0.214, "2026-03-10": 0.05}}},
                        "net_up": {"0": {"log_spread": {"2026-04-28": 0.21, "2026-03-10": 0.02}}, "1": {"log_spread": {"2026-04-28": 0.25, "2026-03-10": 0.0}}}},
          "ranks_two_sided": {"median": {"0": {"log_spread": {"rank": 1, "n": 66, "p": 0.015}}, "1": {"log_spread": {"rank": 1, "n": 66, "p": 0.015}}},
                              "mean": {"0": {"log_spread": {"rank": 2, "n": 66, "p": 0.03}}, "1": {"log_spread": {"rank": 4, "n": 66, "p": 0.06}}},
                              "net_up": {"0": {"log_spread": {"rank": 1, "n": 66, "p": 0.015}}, "1": {"log_spread": {"rank": 1, "n": 66, "p": 0.015}}}},
          "strict": {"n_dates": 33, "ranks": {"log_spread": {"rank": 1, "n": 33, "p": 0.03}}}}
    tex = mt.tab_placebo(b2)
    assert "log spread" in tex and "0.017" in tex and "1/66" in tex and "2/66" in tex and "4/66" in tex and "0.21" in tex
    assert "1/33" in tex and "\\label{tab:placebo}" in tex


def test_tab_mechanisms_lists_recovery_by_activity_category_did_and_the_seam():
    b3 = {"cohort": 51585,
          "recovery_by_activity_quartile": {"q1": {"assets": 12899, "half_life_days": 2, "k0": 0.0153}, "q4": {"assets": 12896, "half_life_days": 3, "k0": 0.1054}},
          "category_did": {"sports": {"treated_assets": 21868, "control_assets": 16227, "treated_change": -0.17, "control_change": 0.0, "did": -0.17},
                           "politics": {"treated_assets": 8138, "control_assets": 29957, "treated_change": 0.0, "control_change": -0.094, "did": 0.094}},
          "category_fee_rates": [{"category": "sports", "fee_rate": 0.03, "assets": 1}, {"category": "politics", "fee_rate": 0.04, "assets": 1}],
          "archive_seam": {"v1_day": "2026-04-14", "v2_day": "2026-04-16", "assets": 41027, "median_log_change": -0.0282, "share_within_10pct": 0.3966}}
    tex = mt.tab_mechanisms(b3)
    assert "sports" in tex and "21{,}868" in tex and "$-0.170$" in tex and "3.0" in tex
    assert "0.105" in tex and "12{,}896" in tex
    assert "41{,}027" in tex and "$-0.028$" in tex
    assert "\\label{tab:mechanisms}" in tex


def test_tab_rollout_lists_classes_with_price_at_event_and_activation():
    c0 = {"events": 1552261, "events_0.01_to_0.001": 1551795,
          "classes": {"R": 1439570, "L": 71795, "X": 40429, "B": 1, "W": 155},
          "class_shares": {"R": 0.9277, "L": 0.0463, "X": 0.0261, "B": 0.0, "W": 0.0001},
          "price_at_event": {"R": {"n": 1439703, "high_side": 719917, "low_side": 719786, "high_mid_median": 0.995, "low_mid_median": 0.005,
                                   "high_mid_p05": 0.965, "low_mid_p95": 0.035, "share_within_0.01_of_threshold": 0.1338, "share_paired_with_complement_within_60s": 0.9997},
                             "L": {"n": 71834, "high_side": 25321, "low_side": 41192, "high_mid_median": 0.9, "low_mid_median": 0.25,
                                   "high_mid_p05": 0.505, "low_mid_p95": 0.5, "share_within_0.01_of_threshold": 0.0, "share_paired_with_complement_within_60s": 0.9997}},
          "activation": {"n": 71795, "share_first_live_quote_within_60min_after": 0.0636, "share_first_live_quote_before_event": 0.9357, "mins_to_first_live_quote_p50": -791.3},
          "hazard": {"auc": 0.7546, "predicted": {"placeholder_day": 0.163, "first_active_day": 0.299}},
          "resolution_proximity": {"R": {"n": 1439703, "one_sided_share": 0.7671, "hours_to_close_median": 2.0361, "share_closed_within_24h": 0.8646,
                                         "sports_with_game_start": 1098985, "hours_after_game_start_median": 1.5694, "share_after_game_start": 0.8074},
                                   "L": {"n": 71834, "one_sided_share": 0.0, "hours_to_close_median": 3.4267, "share_closed_within_24h": 0.7505,
                                         "sports_with_game_start": 59304, "hours_after_game_start_median": 0.7353, "share_after_game_start": 0.6514}}}
    tex = mt.tab_rollout(c0)
    assert "76.7" in tex and "2.0" in tex and "86.5" in tex and "1.6" in tex and "80.7" in tex
    assert "price rule" in tex and "1{,}439{,}570" in tex and "92.8" in tex and "0.995" in tex and "0.005" in tex and "13.4" in tex
    assert "activation" in tex and "71{,}795" in tex and "99.97" in tex
    assert "\\label{tab:rollout}" in tex


def test_tab_eventstudy_reports_gate_and_reweighted_specifications():
    c2 = {"gate": {"all": {"pre_mean_att": 0.597, "pre_trend_p": 0.0, "post_mean_att": -1.472, "post_mean_ratio": 0.229, "treated_units_e0": 241401, "cohorts": 117, "reproduced": True},
                   "liquid": {"pre_mean_att": 0.712, "pre_trend_p": 0.0, "post_mean_att": -1.320, "post_mean_ratio": 0.267, "treated_units_e0": 241686, "cohorts": 117, "reproduced": True}},
          "excluded": {"all": {"pre_mean_att": 0.6, "pre_trend_p": 0.0, "post_mean_att": -1.5, "post_mean_ratio": 0.22, "treated_units_e0": 200000, "cohorts": 81}},
          "reweighted": {"all": {"pre_mean_att": 0.422, "pre_trend_p": 0.0, "post_mean_att": -1.344, "post_mean_ratio": 0.261, "treated_units_e0": 210697, "cohorts": 81},
                         "liquid": {"pre_mean_att": 0.426, "pre_trend_p": 0.0, "post_mean_att": -1.242, "post_mean_ratio": 0.289, "treated_units_e0": 203358, "cohorts": 81}},
          "weight_summary": {"units": 3081897, "p50": 0.22, "p90": 0.399}}
    tex = mt.tab_eventstudy(c2)
    assert "241{,}401" in tex and "0.597" in tex and "$-1.472$" in tex and "0.229" in tex and "117" in tex
    assert "reweighted" in tex and "210{,}697" in tex and "81" in tex
    assert "$<0.003$" in tex and "10^{-9}" not in tex   # a bootstrap p of 0 over 300 draws is below 1/300
    assert "\\label{tab:eventstudy}" in tex


def test_tab_worldcup_reports_both_arms_with_twfe_paired_and_randomisation_inference():
    c1 = {"change": "2026-07-02 00:31:00", "arms": {
        "D": {"old_tick": 0.01, "new_tick": 0.0025, "treated_assets": 79, "treated_markets": 40, "control_assets": 316, "control_markets": 158,
              "mde_log_spread": 0.31, "descriptive_only": True,
              "outcomes": {"log_spread": {"twfe": {"beta": -0.842, "se_cluster": 0.12, "p_cluster": 0.0, "clusters": 198, "n": 1100},
                                          "event_clustered": {"se_cluster": 0.18},
                                          "paired": {"treated_change": -1.1, "control_change": -0.1, "did": -1.004, "n_treated": 40, "n_control": 158},
                                          "ri": {"observed": -1.004, "draws": 2000, "p": 1 / 2001}}}},
        "U": {"old_tick": 0.001, "new_tick": 0.0025, "treated_assets": 57, "treated_markets": 29, "control_assets": 10396, "control_markets": 5198,
              "mde_log_spread": 0.25, "descriptive_only": True,
              "outcomes": {"log_spread": {"twfe": {"beta": -0.193, "se_cluster": 0.09, "p_cluster": 0.03, "clusters": 5227, "n": 30000},
                                          "paired": {"treated_change": -0.3, "control_change": -0.05, "did": -0.255, "n_treated": 29, "n_control": 5198},
                                          "ri": {"observed": -0.255, "draws": 2000, "p": 0.0505}}}}}}
    tex = mt.tab_worldcup(c1)
    assert "0.01" in tex and "0.0025" in tex and "40" in tex and "158" in tex and "$-0.842$" in tex and "$-1.004$" in tex
    assert "5{,}198" in tex and "$-0.193$" in tex and "0.0505" in tex and "0.31" in tex
    assert "0.0005" in tex and "0.180" in tex and "event s.e." in tex
    assert tex.count("descriptive only") == 2 and "plus-one correction" in tex
    assert "\\label{tab:worldcup}" in tex


def test_tab_summary_reports_the_whole_panel_the_on_chain_days_and_the_listing_rate():
    x3 = {"summary": {"days": 171, "rows": 17009350, "rows_by_era": {"v1": 2855452, "v2": 14153898},
                      "assets_per_day_min": 26928, "assets_per_day_max": 302256, "listed_median_feb_jul": 8759.0, "listed_median_aug": 23546.5,
                      "whole_panel": {"mean": 0.2014, "p25": 0.01, "median": 0.039, "p75": 0.299, "p95": 0.86, "n_live": 13577190, "n": 17009350, "updates_median": 2163.0}}}
    pooled = {"totals": {"prints": 24500000, "taker_legs": 25000000, "matched_prints": 24460000}, "n_days": 12}
    tex = mt.tab_summary(x3, pooled)
    assert "17{,}009{,}350" in tex and "0.039" in tex and "0.860" in tex and "2{,}163" in tex and "26{,}928" in tex and "302{,}256" in tex
    assert "8{,}759" in tex and "23{,}547" in tex and "24{,}500{,}000" in tex and "\\label{tab:summary}" in tex
    assert "\\midrule \\\\" not in tex   # a rule row takes no line end

"""Pooling the per-day direction results: the day is the unit."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import a4_pool as ap  # noqa: E402


def days():
    return {
        "2026-05-13": {"rules": {"lr": {"balanced_accuracy": 0.80, "n": 100}, "tick": {"balanced_accuracy": 0.60, "n": 100}}},
        "2026-05-31": {"rules": {"lr": {"balanced_accuracy": 0.82, "n": 120}, "tick": {"balanced_accuracy": 0.62, "n": 120}}},
        "2026-06-05": {"rules": {"lr": {"balanced_accuracy": 0.78, "n": 90}, "tick": {"balanced_accuracy": 0.58, "n": 90}}},
    }


def test_day_level_summary_reports_mean_min_max_and_se():
    s = ap.summarise(days(), ["lr", "tick"], "balanced_accuracy")
    assert s["lr"]["days"] == 3 and s["lr"]["mean"] == pytest.approx(0.80)
    assert s["lr"]["min"] == 0.78 and s["lr"]["max"] == 0.82
    assert s["lr"]["se"] == pytest.approx(0.02 / 3 ** 0.5)


def test_paired_contrast_uses_day_differences():
    c = ap.contrast(days(), "lr", "tick", "balanced_accuracy")
    assert c["mean_diff"] == pytest.approx(0.20) and c["days"] == 3
    assert c["t"] > 10 and c["p"] < 0.01


def test_holm_adjusts_a_family_of_p_values():
    adj = ap.holm({"a": 0.01, "b": 0.04, "c": 0.20})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.08) and adj["c"] == pytest.approx(0.20)


def test_ranges_give_min_max_and_days_for_per_day_dicts():
    per_day = {"2026-05-13": {"a": 0.5, "b": 3}, "2026-05-31": {"a": 0.7, "b": 1}, "2026-06-05": {"a": 0.6, "b": None}}
    r = ap.ranges(per_day)
    assert r["a"] == {"min": 0.5, "max": 0.7, "argmin": "2026-05-13", "argmax": "2026-05-31", "days": 3}
    assert r["b"] == {"min": 1, "max": 3, "argmin": "2026-05-31", "argmax": "2026-05-13", "days": 2}


def test_sign_test_counts_days_with_a_positive_difference():
    s = ap.sign_test([0.1, 0.2, 0.05, 0.3])
    assert s == {"days": 4, "positive": 4, "p_two_sided": pytest.approx(2 / 16)}
    assert ap.sign_test([0.1, -0.2, 0.05])["positive"] == 2

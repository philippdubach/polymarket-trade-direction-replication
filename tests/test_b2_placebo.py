"""Placebo event dates: which dates qualify, and where 28 April ranks."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import b2_placebo_dates as pb  # noqa: E402

D = dt.date


def test_valid_dates_need_window_coverage_clean_ranked_days_and_distance_from_venue_events():
    days = [D(2026, 3, 1) + dt.timedelta(days=i) for i in range(60)]
    covered = set(days) - {D(2026, 3, 20)}
    degraded = {D(2026, 4, 10)} | {D(2026, 4, 20) + dt.timedelta(days=i) for i in range(5)}   # one thin day; one five-day outage
    venue = {D(2026, 3, 30)}
    v = pb.valid_dates(days, covered, degraded, venue, window=7, exclusion=3, clean_span=(-1, 3), outage_len=3)
    assert D(2026, 3, 10) in v
    assert D(2026, 3, 15) not in v          # 03-20 is missing inside D+7
    assert D(2026, 3, 28) not in v and D(2026, 4, 2) not in v   # +/-3 days of 03-30
    assert D(2026, 4, 5) in v               # the single degraded 04-10 is at D+5, outside the ranked days D-1..D+3
    assert D(2026, 4, 8) not in v and D(2026, 4, 11) not in v   # 04-10 inside D-1..D+3
    assert D(2026, 4, 13) not in v          # the 04-20..24 outage is inside D+7
    assert D(2026, 3, 5) not in v           # D-7 = 02-26 not covered


def test_outage_runs_are_runs_of_at_least_the_given_length():
    degraded = {D(2026, 4, 10), D(2026, 4, 20), D(2026, 4, 21), D(2026, 4, 22)}
    assert pb.outage_runs(degraded, 3) == {D(2026, 4, 20), D(2026, 4, 21), D(2026, 4, 22)}


def test_rank_of_the_event_among_placebos():
    stats = {D(2026, 4, 28): 0.9, D(2026, 3, 10): 0.1, D(2026, 3, 11): 0.3, D(2026, 3, 12): -0.2}
    r = pb.rank(stats, D(2026, 4, 28), larger_is_extreme=True)
    assert r == {"rank": 1, "n": 4, "p": 0.25}


def test_joint_rank_uses_both_outcomes():
    a = {D(2026, 4, 28): 0.9, D(2026, 3, 10): 0.5, D(2026, 3, 11): 1.0}
    b = {D(2026, 4, 28): -0.8, D(2026, 3, 10): -0.9, D(2026, 3, 11): 0.0}
    j = pb.joint_rank({"up": (a, True), "down": (b, False)}, D(2026, 4, 28))
    # only the event beats itself on both: 04-28 up 0.9 (03-11 higher but its down is 0.0, not below -0.8)
    assert j["rank"] == 1 and j["n"] == 3


def test_two_sided_rank_orders_by_absolute_size():
    stats = {D(2026, 4, 28): -0.5, D(2026, 3, 10): 0.1, D(2026, 3, 11): 0.6, D(2026, 3, 12): -0.2}
    assert pb.rank_abs(stats, D(2026, 4, 28)) == {"rank": 2, "n": 4, "p": 0.5}


def test_default_exclusion_keeps_the_event_out_of_every_placebo_window():
    days = [D(2026, 4, 1) + dt.timedelta(days=i) for i in range(60)]
    v = pb.valid_dates(days, set(days), set(), {D(2026, 4, 28)})
    assert D(2026, 4, 21) not in v and D(2026, 5, 5) not in v and D(2026, 4, 20) in v and D(2026, 5, 6) in v


def test_spaced_subset_keeps_dates_at_least_the_window_apart():
    days = [D(2026, 4, 1) + dt.timedelta(days=i) for i in range(40)]
    sub = pb.spaced(days, gap=15)
    assert sub == [D(2026, 4, 1), D(2026, 4, 16), D(2026, 5, 1)]


def test_spaced_subset_is_anchored_on_the_event_so_no_window_overlaps_it():
    days = [D(2026, 4, 1) + dt.timedelta(days=i) for i in range(70)]
    sub = pb.spaced(days, gap=15, anchor=D(2026, 4, 28))
    assert D(2026, 4, 28) not in sub
    assert all(abs((d - D(2026, 4, 28)).days) >= 15 for d in sub)
    assert all((b - a).days >= 15 for a, b in zip(sub, sub[1:]))
    assert D(2026, 4, 13) in sub and D(2026, 5, 13) in sub


def test_joint_rank_on_the_mean_is_not_decided_by_a_pinned_median():
    a = {D(2026, 4, 28): 0.3, D(2026, 3, 10): 0.35, D(2026, 3, 11): 0.1}
    b = {D(2026, 4, 28): 0.02, D(2026, 3, 10): -0.5, D(2026, 3, 11): 0.0}
    j = pb.joint_rank({"up": (a, True), "down": (b, False)}, D(2026, 4, 28))
    assert j == {"rank": 2, "n": 3, "p": 2 / 3}   # 03-10 beats the event on both

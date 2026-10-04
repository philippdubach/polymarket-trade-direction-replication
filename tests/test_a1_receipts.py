"""Classifying why a print has no settled counterpart, from its transaction receipt."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import a1_receipts as rc  # noqa: E402

EX = "0xe111180000d2663c0091e4f400237545b87b996b"
RANGE = (100, 200)


def test_missing_receipt_means_never_mined_or_replaced():
    assert rc.classify(None, RANGE, tx_found=False) == "not_mined"


def test_reverted_transaction():
    assert rc.classify({"status": "0x0", "blockNumber": hex(150), "to": EX}, RANGE) == "reverted"


def test_mined_outside_the_scraped_range():
    assert rc.classify({"status": "0x1", "blockNumber": hex(250), "to": EX}, RANGE) == "outside_range"


def test_settled_on_another_contract():
    assert rc.classify({"status": "0x1", "blockNumber": hex(150), "to": "0x" + "9" * 40}, RANGE) == "other_contract"


def test_in_range_success_on_the_exchange_is_a_scrape_gap():
    assert rc.classify({"status": "0x1", "blockNumber": hex(150), "to": EX}, RANGE) == "in_range_missed"


def test_stratified_sample_takes_per_hour_quota():
    hashes = [(f"0x{i:04x}", i % 24) for i in range(480)]
    s = rc.stratified(hashes, per_hour=5, seed=1)
    assert len(s) == 120 and all(sum(1 for _, h in s if h == k) == 5 for k in range(24))


def test_v1_exchange_counts_as_the_exchange():
    v1 = "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"
    assert rc.classify({"status": "0x1", "blockNumber": hex(150), "to": v1}, RANGE) == "in_range_missed"


def test_hour_weighted_shares_weight_each_hour_by_its_population():
    by_hour = {"0": {"reverted": 10}, "1": {"reverted": 5, "not_mined": 5}}
    pop = {"0": 100, "1": 900}
    w = rc.weighted_shares(by_hour, pop)
    assert w["reverted"] == pytest.approx((100 * 1.0 + 900 * 0.5) / 1000) and w["not_mined"] == pytest.approx(0.45)

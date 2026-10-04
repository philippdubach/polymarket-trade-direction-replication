"""Asset -> (condition, outcome) labels merged from Gamma, the V1 payload and the registry."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import x1_token_map as tm  # noqa: E402


def test_gamma_labels_win_and_sources_are_recorded():
    gamma = pl.DataFrame({"condition_id": ["0xa", "0xb"], "token_0": ["1", "3"], "token_1": ["2", "4"],
                          "outcome_0": ["Yes", "Up"], "outcome_1": ["No", "Down"]})
    v1 = pl.DataFrame({"asset_id": ["1", "2", "9"], "condition_id": ["0xa", "0xa", "0xz"], "outcome": ["YES", "NO", "NO"]})
    reg = pl.DataFrame({"condition_id": ["0xa", "0xc"], "yes": ["1", "7"], "no": ["2", "8"]})
    m = tm.build(gamma, v1, reg)
    rows = {r["asset_id"]: r for r in m.to_dicts()}
    assert rows["1"] == {"asset_id": "1", "condition_id": "0xa", "outcome": "YES", "outcome_text": "Yes", "source": "gamma",
                         "sources_agree": True, "n_sources": 3}
    assert rows["3"]["outcome"] == "OUTCOME_0" and rows["3"]["outcome_text"] == "Up" and rows["3"]["n_sources"] == 1
    assert rows["9"] == {"asset_id": "9", "condition_id": "0xz", "outcome": "NO", "outcome_text": None, "source": "v1",
                         "sources_agree": True, "n_sources": 1}
    assert rows["7"]["source"] == "registry" and rows["7"]["outcome"] == "YES"


def test_disagreement_is_flagged_not_hidden():
    gamma = pl.DataFrame({"condition_id": ["0xa"], "token_0": ["1"], "token_1": ["2"], "outcome_0": ["Yes"], "outcome_1": ["No"]})
    v1 = pl.DataFrame({"asset_id": ["1"], "condition_id": ["0xa"], "outcome": ["NO"]})
    reg = pl.DataFrame({"condition_id": [], "yes": [], "no": []}, schema={"condition_id": pl.Utf8, "yes": pl.Utf8, "no": pl.Utf8})
    m = tm.build(gamma, v1, reg)
    r = m.row(0, named=True)
    assert r["outcome"] == "YES" and r["source"] == "gamma" and r["sources_agree"] is False


def test_complement_map_pairs_the_two_tokens_of_a_condition():
    m = pl.DataFrame({"asset_id": ["1", "2", "3"], "condition_id": ["0xa", "0xa", "0xb"], "outcome": ["YES", "NO", "YES"]})
    assert tm.complement_map(m) == {"1": "2", "2": "1"}

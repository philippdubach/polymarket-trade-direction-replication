"""Every number in the manuscript traces to a JSON result, or is listed as unsourced."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import audit_numbers as au  # noqa: E402

TEX = r"""
The feed matched \SI{92.9}{\percent} of prints on 13 May, across \num{2.15} million transactions,
1{,}217{,}156 adoptions and a balanced accuracy of 0.764 (Table~\ref{tab:x}). Section~\ref{sec:y} and
2026 are not numbers to audit; nor is the \SI{95}{\percent} interval label. Kyle's $\lambda$ was 0.0002.
"""


def test_extract_finds_si_num_grouped_integers_and_decimals():
    found = au.extract(TEX)
    vals = sorted(f["value"] for f in found)
    assert 92.9 in vals and 2.15 in vals and 1217156 in vals and 0.764 in vals and 0.0002 in vals
    assert 2026 not in vals            # a year in prose is skipped
    assert all(f["line"] >= 1 for f in found)


def test_lookup_resolves_a_json_path():
    data = {"coverage": {"print_matched_share": 0.92877, "days": [{"n": 2150324}]}}
    assert au.lookup(data, "coverage.print_matched_share") == 0.92877
    assert au.lookup(data, "coverage.days[0].n") == 2150324


def test_match_applies_scale_and_tolerance():
    claims = {"92.9": {"json": "a1.json", "path": "coverage.print_matched_share", "scale": 100, "tol": 0.05},
              "0.764": {"json": "a2.json", "path": "rules.lr.balanced_accuracy", "tol": 0.0005}}
    store = {"a1.json": {"coverage": {"print_matched_share": 0.92877}}, "a2.json": {"rules": {"lr": {"balanced_accuracy": 0.7639}}}}
    rep = au.match(au.extract(TEX), claims, store)
    by = {r["text"]: r["status"] for r in rep}
    assert by["92.9"] == "ok" and by["0.764"] == "ok"
    assert by["2.15"] == "unsourced" and by["1{,}217{,}156"] == "unsourced"


def test_report_counts_and_lists_unsourced():
    rep = [{"text": "1", "status": "ok"}, {"text": "2", "status": "unsourced"}, {"text": "3", "status": "mismatch", "json_value": 5, "value": 3}]
    s = au.summary(rep)
    assert s["ok"] == 1 and s["unsourced"] == ["2"] and s["mismatch"] == 1


def test_lookup_accepts_bracket_quoted_keys_with_dots_or_equals():
    data = {"segments": {"tick_size=0.001": {"lr": {"balanced_accuracy": 0.924}}}}
    assert au.lookup(data, 'segments["tick_size=0.001"].lr.balanced_accuracy') == 0.924


def test_constant_claims_count_as_sourced():
    rep = au.match(au.extract("a tick of 0.001 and a threshold of 0.96"), {"0.001": {"constant": "tick size"}, "0.96": {"constant": "price bound"}}, {})
    assert [r["status"] for r in rep] == ["constant", "constant"]
    assert au.summary(rep)["unsourced"] == []


def test_extract_ignores_includegraphics_options():
    found = au.extract("\\includegraphics[width=0.82\\linewidth]{fig_x.pdf} and a real 0.57 here")
    assert [f["text"] for f in found] == ["0.57"]


def test_claims_can_be_scoped_to_a_file_so_one_number_text_can_carry_two_meanings():
    found = [{"text": "59", "value": 59.0, "kind": "si", "line": 1}]
    claims = {"59": {"json": "x.json", "path": "a", "scale": 100, "tol": 0.5}, "59@manuscript.tex": {"constant": "companion study"}}
    store = {"x.json": {"a": 0.589}}
    rep = au.match(found, claims, store, filename="manuscript.tex")
    assert rep[0]["status"] == "constant"
    rep2 = au.match(found, claims, store, filename="sec_other.tex")
    assert rep2[0]["status"] == "ok"


def test_urls_are_not_audited_as_numbers():
    import audit_numbers as au
    found = au.extract("deposited at \\url{https://doi.org/10.5281/zenodo.23121202} with 0.954 accuracy")
    assert [f["text"] for f in found] == ["0.954"]

"""Tests for the shared helpers of the v0.5 extension scripts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import d0_common as c  # noqa: E402


def test_decode_market_id_turns_hex_of_ascii_into_condition_id():
    cid = "0xAbC123"
    assert c.decode_market_id(cid.encode("ascii").hex()) == "0xabc123"


def test_decode_market_id_rejects_non_hex():
    with pytest.raises(ValueError):
        c.decode_market_id("zz")


def test_day_files_returns_only_existing_hours_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "V2_BASE", str(tmp_path))
    for h in (3, 0, 23):
        (tmp_path / f"polymarket_orderbook_2026-05-13T{h:02d}.parquet").write_bytes(b"")
    (tmp_path / "polymarket_orderbook_2026-05-14T00.parquet").write_bytes(b"")
    got = c.day_files("2026-05-13", "v2")
    assert [Path(p).name[-10:-8] for p in got] == ["00", "03", "23"]


def test_write_json_adds_meta_and_update_json_merges(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "HERE", tmp_path)
    p = c.write_json("x", {"a": 1}, inputs=["f1"])
    d = json.loads(p.read_text())
    assert d["a"] == 1 and d["_meta"]["inputs"] == ["f1"] and "git_head" in d["_meta"]
    c.update_json("x", "b", {"k": 2})
    d = json.loads(p.read_text())
    assert d["a"] == 1 and d["b"] == {"k": 2}
    assert c.read_json("x")["b"]["k"] == 2


def test_degraded_days_is_empty_before_health_check_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "HERE", tmp_path)
    assert c.degraded_days() == set()
    (tmp_path / "d0_archive_health.json").write_text(json.dumps({"degraded_days": ["2026-05-22"]}))
    assert c.degraded_days() == {c.dt.date(2026, 5, 22)}


def test_alchemy_key_prefers_env_and_never_returns_empty(monkeypatch):
    monkeypatch.setenv("ALCHEMY_KEY", "  abc  ")
    assert c.alchemy_key() == "abc"
    monkeypatch.setenv("ALCHEMY_KEY", "")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/nonexistent")))
    with pytest.raises(RuntimeError):
        c.alchemy_key()


def test_connect_sets_utc():
    con = c.connect(memory="256MB", threads=1)
    assert con.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC"


def test_category_from_fee_type_strips_the_version_suffix():
    assert c.category_from_fee_type("sports_fees_v3") == "sports" and c.category_from_fee_type("sports_fees_v2") == "sports"
    assert c.category_from_fee_type("crypto_fees_v2") == "crypto" and c.category_from_fee_type("finance_prices_fees") == "finance_prices"
    assert c.category_from_fee_type(None) == "unknown" and c.category_from_fee_type("general_fees") == "general"


def test_fee_rate_from_schedule_reads_the_rate():
    assert c.fee_rate_from_schedule('{"exponent": 1, "rate": 0.05, "takerOnly": true, "rebateRate": 0.15}') == 0.05
    assert c.fee_rate_from_schedule(None) is None and c.fee_rate_from_schedule("not json") is None


def test_gamma_categories_reads_the_parquet(tmp_path, monkeypatch):
    import polars as pl
    (tmp_path / "gamma").mkdir()
    pl.DataFrame({"condition_id": ["0xa", "0xb"], "fee_type": ["sports_fees_v3", None],
                  "fee_schedule": ['{"rate": 0.03}', None], "neg_risk": [True, False]}).write_parquet(tmp_path / "gamma" / "gamma_markets.parquet")
    monkeypatch.setattr(c, "EXT", tmp_path)
    g = c.gamma_categories()
    assert g.to_dicts() == [{"condition_id": "0xa", "category": "sports", "fee_rate": 0.03, "neg_risk": True},
                            {"condition_id": "0xb", "category": "unknown", "fee_rate": None, "neg_risk": False}]


def test_category_from_fee_type_folds_crypto_variants_into_crypto():
    d0 = c
    assert d0.category_from_fee_type("crypto_15_min") == "crypto"
    assert d0.category_from_fee_type("crypto_fees_v2") == "crypto"

"""The reproducibility manifest: inputs and outputs with hashes, receipts reused, no key material."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import d5_manifest as mf  # noqa: E402


def test_entry_for_a_file_has_sha256_size_and_relative_path(tmp_path):
    f = tmp_path / "a.parquet"
    f.write_bytes(b"hello")
    e = mf.entry(f, root=tmp_path)
    assert e == {"path": "a.parquet", "bytes": 5, "sha256": hashlib.sha256(b"hello").hexdigest()}


def test_receipt_hashes_are_reused_not_recomputed(tmp_path):
    f = tmp_path / "polymarket_orderbook_2026-05-13T00.parquet"
    f.write_bytes(b"x" * 100)
    (tmp_path / "polymarket_orderbook_2026-05-13T00.receipt.json").write_text(json.dumps({"sha256": "abc", "rows": 42, "bytes": 100}))
    e = mf.archive_entry(f)
    assert e["sha256"] == "abc" and e["rows"] == 42 and e["source"] == "receipt"


def test_downloader_receipt_suffix_is_appended_to_parquet_filename(tmp_path):
    f = tmp_path / "polymarket_orderbook_2026-05-13T00.parquet"
    f.write_bytes(b"archive")
    f.with_name(f.name + ".receipt.json").write_text(json.dumps({"sha256": "verified", "rows": 123, "bytes": 7}))
    e = mf.archive_entry(f)
    assert e["source"] == "receipt" and e["sha256"] == "verified" and e["rows"] == 123


def test_manifest_refuses_key_material(tmp_path):
    f = tmp_path / "alchemy.key"
    f.write_text("secret")
    try:
        mf.entry(f, root=tmp_path)
    except ValueError as err:
        assert "key" in str(err)
    else:
        raise AssertionError("a key file must not be listed")


def test_walk_lists_files_under_a_root_relative(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.json").write_text("{}")
    (tmp_path / "a.txt").write_text("a")
    got = sorted(e["path"] for e in mf.walk(tmp_path))
    assert got == ["a.txt", "sub/b.json"]

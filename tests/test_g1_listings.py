"""Archive-independent venue listings per day from a Gamma keyset sweep."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl

REV = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(REV))

import d0_archive_health as ah  # noqa: E402
import g1_gamma_listings as g1  # noqa: E402


def test_day_windows_cover_every_day_with_rfc3339_bounds():
    w = g1.day_windows(dt.date(2026, 6, 14), dt.date(2026, 6, 16))
    assert [x[0] for x in w] == ["2026-06-14", "2026-06-15", "2026-06-16"]
    assert w[1][1] == "2026-06-15T00:00:00Z" and w[1][2] == "2026-06-15T23:59:59.999Z"


def test_page_name_encodes_day_pass_and_index():
    assert g1.page_name("2026-06-15", True, 7) == "2026-06-15_closed_000007.json.gz"
    assert g1.page_name("2026-06-15", False, 0) == "2026-06-15_open_000000.json.gz"


def _rec(i, created, start, closed=True):
    return {"id": str(i), "conditionId": f"0x{i:02d}", "createdAt": created, "startDate": start, "closed": closed,
            "clobTokenIds": "[]", "outcomes": "[]"}


def test_frame_dedups_by_gamma_id_keeping_the_closed_state_last_seen():
    recs = [_rec(1, "2026-06-15T01:00:00Z", "2026-06-15T01:02:00Z", closed=False),
            _rec(1, "2026-06-15T01:00:00Z", "2026-06-15T01:02:00Z", closed=True),
            _rec(2, "2026-06-14T23:50:00Z", "2026-06-15T00:01:00Z")]
    df = g1.to_frame(recs)
    assert df.height == 2
    assert df.filter(pl.col("gamma_id") == "1")["closed"].item() is True


def test_per_day_counts_by_created_and_by_start_day():
    df = g1.to_frame([_rec(1, "2026-06-15T01:00:00Z", "2026-06-15T01:02:00Z"),
                      _rec(2, "2026-06-14T23:50:00Z", "2026-06-15T00:01:00Z"),
                      _rec(3, "2026-06-15T05:00:00Z", "2026-06-15T05:00:00Z")])
    c = g1.per_day_counts(df)
    assert c.filter(pl.col("day") == dt.date(2026, 6, 15)).row(0, named=True) == {"day": dt.date(2026, 6, 15), "created": 2, "started": 3}
    assert c.filter(pl.col("day") == dt.date(2026, 6, 14)).row(0, named=True) == {"day": dt.date(2026, 6, 14), "created": 1, "started": 0}


def test_health_prefers_the_sweep_over_the_archive_enumerated_pull(tmp_path):
    sweep = tmp_path / "gamma_listings.parquet"
    pull = tmp_path / "gamma_markets.parquet"
    pl.DataFrame({"created_at": [dt.datetime(2026, 6, 20, 1)] * 5, "start_date": [dt.datetime(2026, 6, 20, 1)] * 5}).write_parquet(sweep)
    pl.DataFrame({"created_at": [dt.datetime(2026, 6, 20, 1)]}).write_parquet(pull)
    lst, src = ah.venue_listings(sweep, pull)
    assert src == "gamma keyset sweep by startDate, counted by createdAt day"
    assert lst.filter(pl.col("day") == dt.date(2026, 6, 20))["listed"].item() == 5
    lst2, src2 = ah.venue_listings(tmp_path / "absent.parquet", pull)
    assert src2 == "gamma createdAt per day (archive-enumerated ids)" and lst2["listed"].item() == 1


def test_health_counts_how_many_of_each_days_listings_the_archive_carried(tmp_path):
    sweep = tmp_path / "gamma_listings.parquet"
    pl.DataFrame({"created_at": [dt.datetime(2026, 6, 20, 1)] * 4, "start_date": [dt.datetime(2026, 6, 20, 1)] * 4,
                  "condition_id": ["0xa", "0xb", "0xc", "0xd"]}).write_parquet(sweep)
    lst, _ = ah.venue_listings(sweep, tmp_path / "absent.parquet", archive_cids={"0xa", "0xd", "0xzz"})
    row = lst.row(0, named=True)
    assert row["listed"] == 4 and row["carried"] == 2


def test_sweep_splits_the_window_when_the_edge_rejects_a_cursor(tmp_path, monkeypatch):
    """Cloudflare returns 403 for some cursor strings; the sweep must then bisect the window
    and collect every market once, instead of retrying the same rejected request."""
    import json
    monkeypatch.setattr(g1, "PAGES", tmp_path)
    monkeypatch.setattr(g1, "PACE_S", 0.0)
    # 250 markets with startDate spread over the day; a full-day window needs cursors and is rejected on the cursor
    marks = [{"id": str(i), "startDate": f"2026-04-28T{i // 11:02d}:{(i % 11) * 5:02d}:00Z"} for i in range(250)]

    def fake_fetch(params):
        lo, hi = params["start_date_min"], params["start_date_max"]
        if params.get("after_cursor") and lo == "2026-04-28T00:00:00Z" and hi == "2026-04-28T23:59:59.999Z":
            return 403, b"<html>blocked</html>"
        sel = [m for m in marks if lo <= m["startDate"] <= hi]
        start = int(params["after_cursor"]) if params.get("after_cursor") else 0
        page = sel[start:start + 100]
        nxt = str(start + 100) if start + 100 < len(sel) else None
        return 200, json.dumps({"markets": page, "next_cursor": nxt}).encode()

    monkeypatch.setattr(g1, "fetch", fake_fetch)
    out = g1.sweep_day("2026-04-28", "2026-04-28T00:00:00Z", "2026-04-28T23:59:59.999Z", True, {})
    assert sorted(int(m["id"]) for m in out) == list(range(250))

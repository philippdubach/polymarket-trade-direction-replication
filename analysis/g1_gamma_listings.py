"""Every market the venue listed, day by day, independent of the archive.

g0_gamma_pull.py enumerates condition ids from the archive's own shards, so a
market the collector never carried is absent from it. Counting listings per
day from that pull is survivorship-biased in exactly the windows where the
collector stopped subscribing to new markets: a short-lived market created
during an outage was never carried, so it is missing from the benchmark that
is supposed to show the venue kept listing. This script sweeps Gamma's keyset
endpoint by `startDate` day instead, for closed and open markets, and keeps
every raw page gzipped with a manifest so the sweep resumes and re-parses.

Run:
    uv run python g1_gamma_listings.py                      # 2026-02-14 .. 2026-08-12
    uv run python g1_gamma_listings.py --start 2026-06-10 --end 2026-06-30
Output:
    /Volumes/data/paper-b-ext/gamma/listings/pages/*.json.gz, manifest.json
    /Volumes/data/paper-b-ext/gamma/gamma_listings.parquet
    g1_gamma_listings.json (here): per-day counts by createdAt and by startDate day
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import time

import httpx
import polars as pl

from d0_common import EXT, write_json
from g0_gamma_pull import parse_record

URL = "https://gamma-api.polymarket.com/markets/keyset"
LDIR = EXT / "gamma" / "listings"
PAGES = LDIR / "pages"
LIMIT = 100          # the endpoint caps a page at 100 whatever `limit` says
PACE_S = 0.4
START, END = dt.date(2026, 2, 14), dt.date(2026, 8, 12)


def day_windows(start: dt.date, end: dt.date) -> list[tuple[str, str, str]]:
    out = []
    d = start
    while d <= end:
        out.append((d.isoformat(), f"{d.isoformat()}T00:00:00Z", f"{d.isoformat()}T23:59:59.999Z"))
        d += dt.timedelta(days=1)
    return out


def page_name(day: str, closed: bool, i: int) -> str:
    return f"{day}_{'closed' if closed else 'open'}_{i:06d}.json.gz"


def dedup_last(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """One row per Gamma market id; a market seen twice keeps the record seen last."""
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    return lf.with_row_index("_i").sort("_i").unique(subset=["gamma_id"], keep="last").drop("_i").sort("gamma_id").collect()


def to_frame(records: list[dict]) -> pl.DataFrame:
    rows = [parse_record(m) for m in records]
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows, schema_overrides={"event_tags": pl.List(pl.Utf8)}, infer_schema_length=None).with_columns(
        pl.col("tick_size").cast(pl.Float64, strict=False))
    return dedup_last(df)


def per_day_counts(df: pl.DataFrame) -> pl.DataFrame:
    created = df.drop_nulls("created_at").group_by(pl.col("created_at").dt.date().alias("day")).len().rename({"len": "created"})
    started = df.drop_nulls("start_date").group_by(pl.col("start_date").dt.date().alias("day")).len().rename({"len": "started"})
    return created.join(started, on="day", how="full", coalesce=True).fill_null(0).sort("day")


class Blocked(Exception):
    """The edge returned 403/429 for a cursor page: the cursor string itself trips a rule."""


def fetch(params: dict) -> tuple[int, bytes]:
    last = None
    for attempt in range(6):
        try:
            r = httpx.get(URL, params=params, timeout=60)
            if r.status_code == 200:
                return r.status_code, r.content
            last = f"HTTP {r.status_code} {r.text[:120]}"
            if r.status_code in (403, 429):
                if params.get("after_cursor"):
                    return r.status_code, r.content   # deterministic for the cursor: the caller splits the window
                print(f"  HTTP {r.status_code}, sleeping {300 * (attempt + 1)} s", flush=True)
                time.sleep(300 * (attempt + 1))
                continue
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"gamma keyset page failed: {last}")


def _mid(lo: str, hi: str) -> tuple[str, str]:
    """Split an RFC3339 window at its midpoint: [lo, m] and [m + 1 ms, hi]."""
    a = dt.datetime.fromisoformat(lo.replace("Z", "+00:00"))
    b = dt.datetime.fromisoformat(hi.replace("Z", "+00:00"))
    m = a + (b - a) / 2
    m = m.replace(microsecond=(m.microsecond // 1000) * 1000)
    f = lambda t: t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"  # noqa: E731
    return f(m), f(m + dt.timedelta(milliseconds=1))


def sweep_day(day: str, lo: str, hi: str, closed: bool, manifest: dict, tag: str = "") -> list[dict]:
    """All pages of one window and one closed state; cached pages are re-read, the rest fetched.
    A cursor page the edge rejects splits the window in two and sweeps each half afresh."""
    out: list[dict] = []
    cursor = None
    i = 0
    while True:
        name = page_name(day + tag, closed, i)
        p = PAGES / name
        ent = manifest.get(name)
        if ent and ent.get("status") == 200 and p.exists():
            body = gzip.decompress(p.read_bytes())
        else:
            params = {"closed": "true" if closed else "false", "limit": str(LIMIT), "start_date_min": lo, "start_date_max": hi}
            if cursor:
                params["after_cursor"] = cursor
            status, body = fetch(params)
            if status != 200:
                if hi <= lo:
                    raise RuntimeError(f"gamma rejects the cursor on a window that cannot be split: {day}{tag}")
                m1, m2 = _mid(lo, hi)
                print(f"  {day}{tag}: cursor rejected at page {i}, splitting at {m1}", flush=True)
                return (sweep_day(day, lo, m1, closed, manifest, tag + "a") + sweep_day(day, m2, hi, closed, manifest, tag + "b"))
            p.write_bytes(gzip.compress(body))
            manifest[name] = {"status": status, "body_sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                              "after_cursor": cursor, "closed": closed, "day": day, "window": [lo, hi],
                              "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            time.sleep(PACE_S)
        d = json.loads(body)
        ms = d.get("markets") or []
        out.extend(ms)
        cursor = d.get("next_cursor")
        i += 1
        if not cursor or not ms:
            return out


PARTS = LDIR / "parts"
SCHEMA_KEYS = ["condition_id", "gamma_id", "question", "slug", "created_at", "start_date", "end_date", "closed_time",
               "game_start_time", "closed", "active", "neg_risk", "tick_size", "order_min_size", "fees_enabled", "fee_type",
               "fee_schedule", "maker_base_fee", "taker_base_fee", "rewards_min_size", "rewards_max_spread", "sports_market_type",
               "uma_resolution_status", "outcome_prices", "resolved_by", "token_0", "token_1", "outcome_0", "outcome_1",
               "yes_token", "no_token", "event_id", "event_slug", "event_title", "event_tags", "series_slug"]


def assemble() -> pl.DataFrame:
    """Concatenate the per-day parts (schemas relaxed: an all-null column in one part is typed by another)."""
    fs = sorted(PARTS.glob("day=*.parquet"))
    df = dedup_last(pl.concat([pl.read_parquet(f) for f in fs], how="vertical_relaxed"))
    df.write_parquet(EXT / "gamma" / "gamma_listings.parquet")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START.isoformat())
    ap.add_argument("--end", default=END.isoformat())
    ap.add_argument("--no-assemble", action="store_true", help="write the per-day parts only")
    a = ap.parse_args()
    PAGES.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    mpath = LDIR / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    t0 = time.time()
    windows = day_windows(dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end))
    for k, (day, lo, hi) in enumerate(windows):
        part = PARTS / f"day={day}.parquet"
        if part.exists():
            continue
        recs: list[dict] = []
        for closed in (True, False):
            recs += sweep_day(day, lo, hi, closed, manifest)
        mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        df = to_frame(recs)
        # a part's all-null column would be typed Null; cast the parts to one schema so they concatenate
        df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in ("tick_size", "order_min_size", "maker_base_fee", "taker_base_fee", "rewards_min_size", "rewards_max_spread") if c in df.columns])
        df.write_parquet(part)
        el = time.time() - t0
        print(f"  [{k+1}/{len(windows)}] {day}: {df.height:,} markets ({el/60:.1f} min, ~{len(windows)*el/(k+1)/60:.0f} min total)", flush=True)
    if a.no_assemble:
        return
    df = assemble()
    counts = per_day_counts(df)
    g0 = EXT / "gamma" / "gamma_markets.parquet"
    overlap = None
    if g0.exists():
        ids = pl.read_parquet(g0, columns=["gamma_id"])["gamma_id"]
        overlap = {"sweep_ids_in_archive_pull": round(float(df["gamma_id"].is_in(ids.implode()).mean()), 5),
                   "archive_pull_ids_in_sweep": round(float(ids.is_in(df["gamma_id"].implode()).mean()), 5)}
    out = {"start": a.start, "end": a.end, "markets": df.height, "pages": len(manifest),
           "counts": {str(r["day"]): {"created": r["created"], "started": r["started"]} for r in counts.to_dicts()},
           "overlap_with_g0": overlap, "closed_share": round(float(df["closed"].fill_null(False).mean()), 5)}
    write_json("g1_gamma_listings", out, inputs=[str(mpath)])
    print(f"{df.height:,} markets over {len(windows)} days; overlap {overlap}")


if __name__ == "__main__":
    main()

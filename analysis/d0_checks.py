"""Phase 0.1: cheap checks that could invalidate the extension design.

Each subcommand writes one key into d0_checks.json. Run:

    uv run python d0_checks.py tz          # tick events near midnight UTC (timezone exposure)
    uv run python d0_checks.py receipts    # hour-level row counts, the 04-28 stub, thin hours
    uv run python d0_checks.py v1peek      # one price_change and one book_snapshot payload
    uv run python d0_checks.py v1mirror    # are V1 YES/NO rows mirrored pairs?
    uv run python d0_checks.py v1bench     # fused single-parse query on one V1 hour, timed
    uv run python d0_checks.py v2book      # where V2 `book` events fall inside an hour
    uv run python d0_checks.py blocks      # day boundaries by block timestamp for the new days
    uv run python d0_checks.py gamma       # coverage of a 1,000-id sample by the two-pass pull
"""
from __future__ import annotations

import json
import random
import sys
import time

from d0_common import (EXT, PILOT, V2_NEG, V2_STD, alchemy_key, connect, day_files,
                       decode_market_id, ticks_utc, update_json)

NEW_DAYS = ["2026-05-13", "2026-05-31", "2026-06-05", "2026-06-09",
            "2026-07-01", "2026-07-07", "2026-07-17", "2026-08-05"]


def tz() -> dict:
    import polars as pl

    df = ticks_utc().with_columns(
        zurich_day=pl.col("timestamp_received").dt.replace_time_zone(None).dt.date(),
        utc_day=pl.col("ts").dt.date(),
        utc_hour=pl.col("ts").dt.hour(),
    )
    ad = df.filter((pl.col("old_tick") == 0.01) & (pl.col("new_tick") == 0.001))
    out = {
        "events": df.height,
        "adoptions_0.01_to_0.001": ad.height,
        "adoptions_utc_hour_ge_22": int((ad["utc_hour"] >= 22).sum()),
        "adoptions_day_shifts_zurich_vs_utc": int((ad["zurich_day"] != ad["utc_day"]).sum()),
        "share_day_shifts": float((ad["zurich_day"] != ad["utc_day"]).mean()),
        "by_utc_hour": {str(h): int(n) for h, n in ad.group_by("utc_hour").len().sort("utc_hour").iter_rows()},
    }
    return out


def receipts() -> dict:
    import glob
    from pathlib import Path

    from d0_common import V2_BASE

    rows = {}
    for f in sorted(glob.glob(f"{V2_BASE}/*.receipt.json")):
        name = Path(f).name.replace(".receipt.json", "")
        try:
            r = json.loads(Path(f).read_text())
        except Exception:
            continue
        n = r.get("rows") or r.get("row_count") or r.get("num_rows")
        if n is None:
            # find any integer-valued field that looks like a row count
            for k, v in r.items():
                if "row" in k.lower() and isinstance(v, int):
                    n = v
                    break
        rows[name[len("polymarket_orderbook_"):]] = n
    vals = [v for v in rows.values() if isinstance(v, int)]
    vals.sort()
    med = vals[len(vals) // 2] if vals else None
    thin = {k: v for k, v in rows.items() if isinstance(v, int) and med and v < 0.2 * med}
    focus = {k: rows.get(k) for k in ["2026-04-27T10", "2026-04-27T11", "2026-04-27T12",
                                       "2026-04-28T10", "2026-04-28T11", "2026-04-28T12", "2026-04-28T13"]}
    return {"hours": len(rows), "median_rows_per_hour": med, "thin_hours_lt_20pct_median": thin,
            "cutover_window": focus, "receipt_keys": sorted(json.loads(Path(glob.glob(f"{V2_BASE}/*.receipt.json")[0]).read_text()).keys())}


def _v1_hour(day="2026-03-25", hour=12) -> str:
    from d0_common import V1_BASE
    return f"{V1_BASE}/polymarket_orderbook_{day}T{hour:02d}.parquet"


def v1peek() -> dict:
    con = connect()
    f = _v1_hour()
    out = {"file": f}
    for ut in ("price_change", "book_snapshot"):
        r = con.execute(f"SELECT data FROM read_parquet('{f}') WHERE update_type='{ut}' LIMIT 1").fetchone()
        out[ut] = (r[0][:600] + "…") if r and len(r[0]) > 600 else (r[0] if r else None)
    out["update_types"] = con.execute(f"SELECT update_type, count(*) FROM read_parquet('{f}') GROUP BY 1").fetchall()
    return out


def v1mirror() -> dict:
    con = connect()
    f = _v1_hour()
    q = f"""
    WITH pc AS (
      SELECT market_id, timestamp_received,
             json_extract_string(data, ['$.side', '$.best_bid', '$.best_ask']) AS j
      FROM read_parquet('{f}') WHERE update_type = 'price_change'
    ), s AS (
      SELECT market_id, timestamp_received, j[1] AS side,
             CAST(j[2] AS DOUBLE) AS bid, CAST(j[3] AS DOUBLE) AS ask FROM pc
    )
    SELECT
      count(*) FILTER (WHERE side='YES') AS yes_rows,
      count(*) FILTER (WHERE side='NO') AS no_rows,
      (SELECT count(*) FROM s y JOIN s n USING (market_id, timestamp_received)
        WHERE y.side='YES' AND n.side='NO') AS paired_rows,
      (SELECT count(*) FROM s y JOIN s n USING (market_id, timestamp_received)
        WHERE y.side='YES' AND n.side='NO' AND abs(n.bid - (1 - y.ask)) < 1e-9 AND abs(n.ask - (1 - y.bid)) < 1e-9) AS mirrored_rows
    FROM s
    """
    r = con.execute(q).fetchone()
    return {"file": f, "yes_rows": r[0], "no_rows": r[1], "paired_rows": r[2], "mirrored_rows": r[3],
            "share_mirrored_of_paired": (r[3] / r[2]) if r[2] else None}


def v1bench() -> dict:
    """Time the fused query the V1 extraction will use. Abort criterion: > 90 s per hour."""
    con = connect()
    f = _v1_hour()
    t = time.time()
    q = f"""
    WITH pc AS (
      SELECT market_id,
             json_extract_string(data, ['$.token_id','$.side','$.best_bid','$.best_ask',
                                        '$.change_price','$.change_size','$.change_side']) AS j
      FROM read_parquet('{f}') WHERE update_type = 'price_change'
    ), s AS (
      SELECT market_id, j[1] AS asset_id, j[2] AS side,
             CAST(j[3] AS DOUBLE) AS bid, CAST(j[4] AS DOUBLE) AS ask,
             CAST(j[5] AS DOUBLE) AS cp, CAST(j[6] AS DOUBLE) AS cs, j[7] AS cside
      FROM pc
    )
    SELECT market_id, asset_id, side, count(*) AS updates,
           sum(ask - bid) AS spread_sum, sum((ask-bid)*(ask-bid)) AS spread_sq_sum, max(ask-bid) AS spread_max,
           count(*) FILTER (WHERE cside='BUY' AND cp = bid) AS depth_bid_n,
           sum(cs) FILTER (WHERE cside='BUY' AND cp = bid) AS depth_bid_sum,
           count(*) FILTER (WHERE cside='SELL' AND cp = ask) AS depth_ask_n,
           sum(cs) FILTER (WHERE cside='SELL' AND cp = ask) AS depth_ask_sum
    FROM s WHERE ask >= bid GROUP BY 1,2,3
    """
    n = con.execute(f"SELECT count(*) FROM ({q})").fetchone()[0]
    t_pc = time.time() - t
    return {"file": f, "price_change_seconds": round(t_pc, 1), "rows": n, "abort_if_over_s": 90}


def v2book() -> dict:
    con = connect()
    from d0_common import V2_BASE
    out = {}
    for f in [f"{V2_BASE}/polymarket_orderbook_2026-06-10T12.parquet", f"{V2_BASE}/polymarket_orderbook_2026-05-13T03.parquet"]:
        r = con.execute(f"""
          SELECT count(*) AS n, count(DISTINCT asset_id) AS assets,
                 count(*) FILTER (WHERE minute(timestamp_received) < 5) AS first5min,
                 count(*) FILTER (WHERE best_bid IS NULL) AS null_best,
                 min(timestamp_received)::VARCHAR, max(timestamp_received)::VARCHAR
          FROM read_parquet('{f}') WHERE event_type='book'""").fetchone()
        per_asset = con.execute(f"""
          SELECT quantile_cont(c, [0.5, 0.9, 0.99]) FROM (
            SELECT asset_id, count(*) AS c FROM read_parquet('{f}') WHERE event_type='book' GROUP BY 1)""").fetchone()[0]
        out[f.rsplit("/", 1)[1]] = {"book_rows": r[0], "assets": r[1], "in_first_5_min": r[2], "null_best": r[3],
                                     "first": r[4], "last": r[5], "book_rows_per_asset_p50_p90_p99": per_asset}
    return out


def _rpc(method: str, params: list, key: str):
    import httpx
    r = httpx.post(f"https://polygon-mainnet.g.alchemy.com/v2/{key}",
                   json={"id": 1, "jsonrpc": "2.0", "method": method, "params": params}, timeout=60)
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(j["error"])
    return j["result"]


def _block_ts(n: int, key: str) -> int:
    b = _rpc("eth_getBlockByNumber", [hex(n), False], key)
    return int(b["timestamp"], 16)


def _first_block_at_or_after(ts: int, lo: int, hi: int, key: str) -> int:
    """Smallest block number whose timestamp >= ts, searched in [lo, hi]."""
    calls = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if _block_ts(mid, key) >= ts:
            hi = mid
        else:
            lo = mid + 1
        calls += 1
    return lo


def blocks() -> dict:
    import datetime as dt
    key = alchemy_key()
    latest = int(_rpc("eth_blockNumber", [], key), 16)
    idx_path = EXT / "onchain" / "block_index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    # Known anchor from the pilot: 2026-04-29 00:00 UTC == block 86020778 (day-start in scrape_onchain_v2.py).
    anchor_block, anchor_day = 86150378, dt.date(2026, 4, 29)
    out = {}
    for d in NEW_DAYS:
        if d in idx:
            out[d] = idx[d] | {"cached": True}
            continue
        day = dt.date.fromisoformat(d)
        t0 = int(dt.datetime.combine(day, dt.time(), tzinfo=dt.timezone.utc).timestamp())
        t1 = t0 + 86400
        # bracket: block rate drifted from 2.0 s (April) to 1.5 s (July), so bound
        # the search by the fastest and slowest plausible rates rather than one guess
        secs = t0 - int(dt.datetime.combine(anchor_day, dt.time(), tzinfo=dt.timezone.utc).timestamp())
        lo, hi = anchor_block + int(secs / 2.3), min(latest, anchor_block + int(secs / 1.3))
        start = _first_block_at_or_after(t0, lo, hi, key)
        end = _first_block_at_or_after(t1, start, min(latest, start + 120_000), key) - 1
        s_ts, e_ts = _block_ts(start, key), _block_ts(end, key)
        # blockTimestamp on logs in one 10-block slice mid-day
        mid = start + (end - start) // 2
        logs = _rpc("eth_getLogs", [{"address": [V2_STD, V2_NEG], "fromBlock": hex(mid), "toBlock": hex(mid + 9)}], key)
        has_ts = all("blockTimestamp" in l for l in logs) if logs else None
        idx[d] = {"start_block": start, "end_block": end, "blocks": end - start + 1,
                  "start_ts": dt.datetime.fromtimestamp(s_ts, dt.timezone.utc).isoformat(),
                  "end_ts": dt.datetime.fromtimestamp(e_ts, dt.timezone.utc).isoformat(),
                  "sec_per_block": round(86400 / (end - start + 1), 4),
                  "sample_logs_in_10_blocks": len(logs), "logs_carry_blockTimestamp": has_ts}
        out[d] = idx[d]
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(json.dumps(idx, indent=2))
    return {"latest_block": latest, "days": out, "index_file": str(idx_path)}


def gamma() -> dict:
    import glob

    import httpx
    import polars as pl

    fs = sorted(glob.glob(str(PILOT / "shards" / "day=*.parquet")))
    ids = pl.concat([pl.read_parquet(f, columns=["market_id"]) for f in fs[::7]])["market_id"].unique().to_list()
    random.seed(20260926)
    sample = [decode_market_id(x) for x in random.sample(ids, 1000)]
    url = "https://gamma-api.polymarket.com/markets"
    found: dict[str, dict] = {}
    stats = {"pass_a_requests": 0, "pass_b_requests": 0, "max_url_bytes": 0, "errors": 0}

    def fetch(batch: list[str], closed: bool, tag: str) -> None:
        params = [("closed", "true" if closed else "false"), ("limit", "100")] + [("condition_ids", c) for c in batch]
        for attempt in range(5):
            try:
                r = httpx.get(url, params=params, timeout=60)
                stats["max_url_bytes"] = max(stats["max_url_bytes"], len(str(r.request.url)))
                stats[f"{tag}_requests"] += 1
                if r.status_code == 200:
                    for m in r.json():
                        cid = (m.get("conditionId") or "").lower()
                        if cid:
                            found[cid] = m
                    return
                stats["errors"] += 1
            except Exception:
                stats["errors"] += 1
            time.sleep(2 ** attempt)

    for i in range(0, len(sample), 90):
        fetch(sample[i:i + 90], True, "pass_a")
        time.sleep(0.25)
    after_a = sum(1 for c in sample if c in found)
    residual = [c for c in sample if c not in found]
    for i in range(0, len(residual), 90):
        fetch(residual[i:i + 90], False, "pass_b")
        time.sleep(0.25)
    after_b = sum(1 for c in sample if c in found)
    missing = [c for c in sample if c not in found]
    # field presence and token/outcome ordering on a negRisk market
    keys = {}
    for m in found.values():
        for k in m:
            keys[k] = keys.get(k, 0) + 1
    neg = next((m for m in found.values() if m.get("negRisk")), None)
    ex = None
    if neg:
        ex = {"question": neg.get("question"), "clobTokenIds": neg.get("clobTokenIds"), "outcomes": neg.get("outcomes"),
              "orderPriceMinTickSize": neg.get("orderPriceMinTickSize"), "createdAt": neg.get("createdAt"),
              "closedTime": neg.get("closedTime"), "event_keys": sorted((neg.get("events") or [{}])[0].keys())[:40]}
    return {"sample": len(sample), "coverage_after_pass_a": after_a / len(sample), "coverage_after_pass_b": after_b / len(sample),
            "missing": missing[:20], "missing_n": len(missing), **stats,
            "fields_present_share": {k: round(v / max(1, len(found)), 3) for k, v in sorted(keys.items())},
            "negrisk_example": ex}


CHECKS = {"tz": tz, "receipts": receipts, "v1peek": v1peek, "v1mirror": v1mirror, "v1bench": v1bench,
          "v2book": v2book, "blocks": blocks, "gamma": gamma}


def main() -> None:
    names = sys.argv[1:] or list(CHECKS)
    for n in names:
        t = time.time()
        res = CHECKS[n]()
        res["_seconds"] = round(time.time() - t, 1)
        update_json("d0_checks", n, res)
        print(f"== {n} ({res['_seconds']}s)")
        print(json.dumps(res, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()

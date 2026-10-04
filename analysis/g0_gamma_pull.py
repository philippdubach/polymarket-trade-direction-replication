"""Gamma API metadata for every condition id in the V2 shards.

Closed markets are only returned when `closed=true` is passed, so the pull runs
in two passes: closed first, then the residual with `closed=false`. Every raw
page is kept gzipped with a manifest entry (ids hash, body hash, status), so
the pull resumes and a reader can re-derive the parquet from the pages.

Run:
    uv run python g0_gamma_pull.py                # full pull
    uv run python g0_gamma_pull.py --limit 5000   # first N ids, for a rehearsal
Output:
    /Volumes/data/paper-b-ext/gamma/pages/*.json.gz, manifest.json
    /Volumes/data/paper-b-ext/gamma/gamma_markets.parquet
    g0_gamma_coverage.json (here)
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import gzip
import hashlib
import json
import time
from pathlib import Path

import httpx
import polars as pl

from d0_common import EXT, PILOT, decode_market_id, write_json

URL = "https://gamma-api.polymarket.com/markets"
GDIR = EXT / "gamma"
PAGES = GDIR / "pages"
BATCH = 90
PACE_S = 0.25


# ----------------------------------------------------------------------------- parsing

def parse_time(v: str | None) -> dt.datetime | None:
    """Gamma emits '2026-04-05T14:30:01.569952Z', '2026-05-02 21:53:48+00' and '...T21:00:00Z'."""
    if not v:
        return None
    s = v.strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    elif s.endswith("+00"):
        s = s + ":00"
    try:
        t = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if t.tzinfo is not None:
        t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return t


def _json_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    try:
        out = json.loads(v)
        return out if isinstance(out, list) else []
    except (TypeError, ValueError):
        return []


def parse_record(m: dict) -> dict:
    tokens = [str(t) for t in _json_list(m.get("clobTokenIds"))]
    outcomes = [str(o) for o in _json_list(m.get("outcomes"))]
    yes = no = None
    if len(tokens) == 2 and [o.lower() for o in outcomes] == ["yes", "no"]:
        yes, no = tokens
    ev = (m.get("events") or [{}])[0] if m.get("events") else {}
    tags = [t.get("label") for t in (ev.get("tags") or []) if isinstance(t, dict) and t.get("label")]
    series = ev.get("series") or []
    return {
        "condition_id": (m.get("conditionId") or "").lower() or None,
        "gamma_id": str(m["id"]) if m.get("id") is not None else None,
        "question": m.get("question"),
        "slug": m.get("slug"),
        "created_at": parse_time(m.get("createdAt")),
        "start_date": parse_time(m.get("startDate")),
        "end_date": parse_time(m.get("endDate")),
        "closed_time": parse_time(m.get("closedTime")),
        "game_start_time": parse_time(m.get("gameStartTime")),
        "closed": m.get("closed"),
        "active": m.get("active"),
        "neg_risk": m.get("negRisk"),
        "tick_size": m.get("orderPriceMinTickSize"),
        "order_min_size": m.get("orderMinSize"),
        "fees_enabled": m.get("feesEnabled"),
        "fee_type": m.get("feeType"),
        "fee_schedule": json.dumps(m["feeSchedule"]) if m.get("feeSchedule") is not None else None,
        "maker_base_fee": m.get("makerBaseFee"),
        "taker_base_fee": m.get("takerBaseFee"),
        "rewards_min_size": m.get("rewardsMinSize"),
        "rewards_max_spread": m.get("rewardsMaxSpread"),
        "sports_market_type": m.get("sportsMarketType"),
        "uma_resolution_status": m.get("umaResolutionStatus"),
        "outcome_prices": m.get("outcomePrices"),
        "resolved_by": m.get("resolvedBy"),
        "token_0": tokens[0] if len(tokens) > 0 else None,
        "token_1": tokens[1] if len(tokens) > 1 else None,
        "outcome_0": outcomes[0] if len(outcomes) > 0 else None,
        "outcome_1": outcomes[1] if len(outcomes) > 1 else None,
        "yes_token": yes,
        "no_token": no,
        "event_id": str(ev["id"]) if ev.get("id") is not None else None,
        "event_slug": ev.get("slug"),
        "event_title": ev.get("title"),
        "event_tags": tags,
        "series_slug": series[0].get("slug") if series and isinstance(series[0], dict) else None,
    }


def batches(ids: list, size: int) -> list[list]:
    return [ids[i:i + size] for i in range(0, len(ids), size)]


# ----------------------------------------------------------------------------- pulling

def shard_condition_ids() -> list[str]:
    fs = sorted(glob.glob(str(PILOT / "shards" / "day=*.parquet")))
    ids = pl.concat([pl.read_parquet(f, columns=["market_id"]) for f in fs])["market_id"].unique().sort().to_list()
    return [decode_market_id(x) for x in ids]


def _page_name(pass_tag: str, i: int) -> str:
    return f"{pass_tag}_{i:06d}.json.gz"


def fetch_page(batch: list[str], closed: bool) -> tuple[int, bytes]:
    params = [("closed", "true" if closed else "false"), ("limit", "100")] + [("condition_ids", c) for c in batch]
    last = None
    for attempt in range(6):
        try:
            r = httpx.get(URL, params=params, timeout=60)
            if r.status_code == 200:
                return r.status_code, r.content
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"gamma page failed: {last}")


def run_pass(ids: list[str], closed: bool, tag: str, manifest: dict) -> dict[str, dict]:
    found: dict[str, dict] = {}
    bs = batches(ids, BATCH)
    t0 = time.time()
    for i, b in enumerate(bs):
        name = _page_name(tag, i)
        p = PAGES / name
        ent = manifest.get(name)
        ids_sha = hashlib.sha256("\n".join(b).encode()).hexdigest()
        if ent and ent.get("ids_sha256") == ids_sha and p.exists() and ent.get("status") == 200:
            body = gzip.decompress(p.read_bytes())
        else:
            status, body = fetch_page(b, closed)
            p.write_bytes(gzip.compress(body))
            manifest[name] = {"ids_sha256": ids_sha, "body_sha256": hashlib.sha256(body).hexdigest(),
                              "status": status, "bytes": len(body), "closed": closed, "n_ids": len(b),
                              "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            if i % 50 == 0:
                (GDIR / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
            time.sleep(PACE_S)
        for m in json.loads(body):
            cid = (m.get("conditionId") or "").lower()
            if cid:
                found[cid] = m
        if i % 200 == 0:
            el = time.time() - t0
            print(f"  {tag} [{i+1}/{len(bs)}] found={len(found):,} ({el/60:.1f} min, ~{len(bs)*el/(i+1)/60:.0f} min total)", flush=True)
    (GDIR / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    PAGES.mkdir(parents=True, exist_ok=True)
    mpath = GDIR / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    ids = shard_condition_ids()
    if a.limit:
        ids = ids[:a.limit]
    print(f"condition ids: {len(ids):,}", flush=True)
    found = run_pass(ids, True, "closed", manifest)
    n_a = sum(1 for c in ids if c in found)
    residual = [c for c in ids if c not in found]
    print(f"pass A (closed=true): {n_a:,} found, {len(residual):,} residual", flush=True)
    found |= run_pass(residual, False, "open", manifest)
    n_b = sum(1 for c in ids if c in found)
    missing = [c for c in ids if c not in found]
    print(f"pass B (closed=false): {n_b:,} found, {len(missing):,} missing", flush=True)
    (GDIR / "missing.json").write_text(json.dumps(missing))

    rows = [parse_record(m) for m in found.values()]
    df = pl.DataFrame(rows, schema_overrides={"event_tags": pl.List(pl.Utf8)}).with_columns(
        pl.col("tick_size").cast(pl.Float64, strict=False),
    )
    df.write_parquet(GDIR / "gamma_markets.parquet")

    # coverage of the archive's assets by the markets' own token ids
    fs = sorted(glob.glob(str(PILOT / "shards" / "day=*.parquet")))
    assets = pl.concat([pl.read_parquet(f, columns=["market_id", "asset_id"]) for f in fs]).unique()
    assets = assets.with_columns(pl.col("market_id").map_elements(decode_market_id, return_dtype=pl.Utf8).alias("condition_id"))
    tok = df.select("condition_id", "token_0", "token_1").unpivot(index="condition_id", value_name="asset_id").drop("variable").drop_nulls()
    asset_cov = assets.join(tok, on=["condition_id", "asset_id"], how="inner").height / assets.height
    cov = {
        "condition_ids": len(ids), "found_pass_a": n_a, "found_total": n_b, "missing": len(missing),
        "coverage_conditions": round(n_b / len(ids), 5), "coverage_assets_by_token_id": round(asset_cov, 5),
        "share_yes_no_outcomes": round(float(df["yes_token"].is_not_null().mean()), 5),
        "rows": df.height, "pages": len(manifest),
        "tick_size_values": {str(k): int(v) for k, v in df.group_by("tick_size").len().sort("tick_size").iter_rows()},
        "neg_risk_share": round(float(df["neg_risk"].fill_null(False).mean()), 5),
        "created_at_range": [str(df["created_at"].min()), str(df["created_at"].max())],
    }
    write_json("g0_gamma_coverage", cov, inputs=[str(GDIR / "manifest.json")])
    print(json.dumps(cov, indent=2))


if __name__ == "__main__":
    main()

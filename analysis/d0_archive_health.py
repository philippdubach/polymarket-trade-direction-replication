"""Capture health of the V2 feed archive, day by day, and the degraded windows.

A collector whose subscription list stops refreshing shows a specific signature:
new assets stop appearing while the venue keeps listing markets, the observed
universe decays, and trades fall. Those windows are not market events and every
downstream script excludes them. The rule is applied to the archive's own
first-seen counts against a trailing median, with Gamma `createdAt` counts as
the external benchmark that separates "the venue listed nothing" from "the
collector saw nothing". Days with missing hours are degraded regardless.

Run:  uv run python d0_archive_health.py
Out:  d0_archive_health.json  (per-day table, rule parameters, degraded_days)
"""
from __future__ import annotations

import datetime as dt
import glob
import json
from pathlib import Path

import polars as pl

from d0_common import EXT, PILOT, V2_BASE, day_files, decode_market_id, write_json

WINDOW, RATIO, MIN_HOURS, MIN_CARRIED = 7, 0.10, 24, 0.5


def degraded_days(df: pl.DataFrame, listings: pl.DataFrame | None, window: int = WINDOW, ratio: float = RATIO,
                  min_hours: int | None = None, min_carried: float | None = None) -> set[dt.date]:
    """Days on which the collector's view of the venue collapsed.

    `df` has `day` and `first_seen`. A day is degraded when its first-seen count is below
    `ratio` of the median over the trailing `window` days that are not themselves degraded
    (so a long outage stays flagged); when `listings` (columns `day`, `listed`, optionally
    `carried`) is given, only if the venue's own listings did not collapse too, and, with
    `min_carried`, also when the archive carried less than that share of the day's listings.
    With `min_hours`, a day whose `hours_present` is below it is degraded regardless.
    """
    d = df.sort("day")
    fs = d["first_seen"].to_list()
    days = d["day"].to_list()
    flags: set[dt.date] = set()
    listed = dict(zip(listings["day"].to_list(), listings["listed"].to_list())) if listings is not None else None
    carried = (dict(zip(listings["day"].to_list(), listings["carried"].to_list()))
               if listings is not None and "carried" in listings.columns else None)
    for i, day in enumerate(days):
        prev_idx = [j for j in range(i - 1, -1, -1) if days[j] not in flags][:window]
        prev_days = [days[j] for j in prev_idx]
        prev = [fs[j] for j in prev_idx]
        if len(prev) >= 3:
            med = sorted(prev)[len(prev) // 2]
            if med > 0 and fs[i] < ratio * med:
                venue_ok = True
                if listed is not None:
                    lprev = [listed.get(x, 0) for x in prev_days]
                    lmed = sorted(lprev)[len(lprev) // 2]
                    venue_ok = listed.get(day, 0) >= 0.5 * lmed
                if venue_ok:
                    flags.add(day)
        if min_carried is not None and carried is not None and listed is not None and listed.get(day, 0) > 0:
            if carried.get(day, 0) / listed[day] < min_carried:
                flags.add(day)
        if min_hours is not None and "hours_present" in d.columns and d["hours_present"][i] < min_hours:
            flags.add(day)
    return flags


SWEEP_SRC = "gamma keyset sweep by startDate, counted by createdAt day"
PULL_SRC = "gamma createdAt per day (archive-enumerated ids)"


def venue_listings(sweep: Path, pull: Path, archive_cids: set[str] | None = None) -> tuple[pl.DataFrame | None, str | None]:
    """Listings per day: the archive-independent sweep (g1) when it exists, else the g0 pull.

    The g0 pull only knows condition ids the archive carried, so its per-day counts
    collapse in the same windows the collector missed; it is a fallback, not a benchmark.
    With the sweep and `archive_cids`, `carried` counts the day's listings whose condition
    id the archive carried on any day.
    """
    for p, src in ((sweep, SWEEP_SRC), (pull, PULL_SRC)):
        if not p.exists():
            continue
        cols = ["created_at"] + (["condition_id"] if src == SWEEP_SRC and archive_cids is not None else [])
        g = pl.read_parquet(p, columns=cols).drop_nulls("created_at").with_columns(pl.col("created_at").dt.date().alias("day"))
        aggs = [pl.len().alias("listed")]
        if "condition_id" in cols:
            aggs.append(pl.col("condition_id").is_in(sorted(archive_cids or ())).sum().alias("carried"))
        return g.group_by("day").agg(aggs).sort("day"), src
    return None, None


MIGRATION_HOURS = {"2026-04-28T11", "2026-04-28T12"}   # the venue's own outage during the cutover, not the collector's


def thin_hours(rows: dict[str, int], ratio: float = 0.2, exempt: set[str] = MIGRATION_HOURS) -> dict[str, list[int]]:
    """Hourly files under `ratio` of the median hour's rows, by day, except the exempt hours."""
    vals = sorted(rows.values())
    if not vals:
        return {}
    med = vals[len(vals) // 2]
    out: dict[str, list[int]] = {}
    for k, v in sorted(rows.items()):
        if v < ratio * med and k not in exempt:
            out.setdefault(k[:10], []).append(int(k[11:13]))
    return out


def sensitivity(per_day: pl.DataFrame, listings: pl.DataFrame | None) -> dict:
    """The flagged count under the baseline rule and under alternative thresholds: the first-seen
    ratio at one third and two thirds, the carried share at one third and two thirds, and trailing
    windows of five and ten days. Missing-hour and thin-hour flags are not varied."""
    def run(**kw) -> dict:
        args = {"window": WINDOW, "ratio": RATIO, "min_hours": MIN_HOURS, "min_carried": MIN_CARRIED} | kw
        f = degraded_days(per_day, listings, **args)
        return {"n": len(f), "days": sorted(str(d) for d in f)}
    out = {"baseline": run()}
    for r in (1 / 3, 2 / 3):
        out[f"ratio={r:.3f}"] = run(ratio=r)
        out[f"min_carried={r:.3f}"] = run(min_carried=r)
    for w in (5, 10):
        out[f"window={w}"] = run(window=w)
    base = set(out["baseline"]["days"])
    for k, v in out.items():
        v["added"] = sorted(set(v["days"]) - base)
        v["removed"] = sorted(base - set(v["days"]))
        del v["days"]
    return out


def _receipt_rows() -> dict[str, int]:
    rows = {}
    for f in glob.glob(f"{V2_BASE}/*.receipt.json"):
        try:
            r = json.loads(Path(f).read_text())
            rows[Path(f).name[len("polymarket_orderbook_"):-len(".parquet.receipt.json")]] = int(r["rows"])
        except Exception:  # noqa: BLE001
            pass
    return rows


def per_day_table(sh: pl.DataFrame) -> tuple[pl.DataFrame, set[str]]:
    """Per-day assets, updates, trades, first-seen assets (first day across eras) and the era
    of the day's shard; plus the set of condition ids the archive carried on any day."""
    cids = set(sh["condition_id"].drop_nulls().unique().to_list())
    first = sh.group_by("asset_id").agg(first_day=pl.col("day").min())
    per_day = sh.group_by("day").agg(assets=pl.len(), updates=pl.col("updates").sum(), trades=pl.col("trades").sum(),
                                     era=pl.col("era").first()).sort("day")
    seen = first.group_by("first_day").len().rename({"first_day": "day", "len": "first_seen"})
    firstm = sh.drop_nulls("condition_id").group_by("condition_id").agg(first_day=pl.col("day").min())
    seenm = firstm.group_by("first_day").len().rename({"first_day": "day", "len": "first_seen_markets"})
    per_day = per_day.join(seen, on="day", how="left").join(seenm, on="day", how="left")
    return per_day.with_columns(pl.col("first_seen").fill_null(0), pl.col("first_seen_markets").fill_null(0)), cids


def load_shards() -> tuple[pl.DataFrame, str]:
    """The harmonised daily shards of both eras when x0 has written them, else the V2 pilot shards."""
    hs = sorted(glob.glob(str(EXT / "shards_h" / "era=*" / "day=*.parquet")))
    if hs:
        cols = ["asset_id", "condition_id", "day", "updates", "trades", "era"]
        return pl.concat([pl.read_parquet(f, columns=cols) for f in hs], how="vertical_relaxed"), str(EXT / "shards_h")
    fs = sorted(glob.glob(str(PILOT / "shards" / "day=*.parquet")))
    sh = pl.concat([pl.read_parquet(f, columns=["asset_id", "market_id", "day", "updates", "trades"]) for f in fs])
    return sh.with_columns(pl.col("market_id").map_elements(decode_market_id, return_dtype=pl.Utf8).alias("condition_id"),
                           pl.lit("v2").alias("era")).drop("market_id"), str(PILOT / "shards")


def main() -> None:
    sh, source = load_shards()
    per_day, archive_cids = per_day_table(sh)
    rows = _receipt_rows()
    per_day = per_day.with_columns(
        pl.struct("day", "era").map_elements(lambda r: len(day_files(str(r["day"]), r["era"])), return_dtype=pl.Int64).alias("hours_present"),
        pl.col("day").map_elements(lambda d: sum(rows.get(f"{d}T{h:02d}", 0) for h in range(24)), return_dtype=pl.Int64).alias("receipt_rows"),
    )
    sweep, pull = EXT / "gamma" / "gamma_listings.parquet", EXT / "gamma" / "gamma_markets.parquet"
    listings, src = venue_listings(sweep, pull, archive_cids)
    if listings is not None:
        per_day = per_day.join(listings, on="day", how="left").with_columns(pl.col("listed").fill_null(0))
        if "carried" in per_day.columns:
            per_day = per_day.with_columns(pl.col("carried").fill_null(0),
                                           (pl.col("carried") / pl.col("listed").clip(lower_bound=1)).round(5).alias("carried_share"))
    flags = degraded_days(per_day, listings, WINDOW, RATIO, MIN_HOURS, MIN_CARRIED)
    thin = thin_hours(rows)   # V2 only: the V1 archive has no receipts
    flags |= {dt.date.fromisoformat(d) for d in thin}
    # the first archive day starts at 08:00 and the last ends at 00:41: partial by construction, not outages
    ok = per_day.filter(~pl.col("day").is_in(sorted(flags)))
    v2h = [v for v in rows.values()]
    summary = {
        "days_n": per_day.height, "degraded_n": len(flags), "nondegraded_n": ok.height,
        "listed_total": int(per_day["listed"].sum()) if "listed" in per_day.columns else None,
        "carried_total": int(per_day["carried"].sum()) if "carried" in per_day.columns else None,
        "carried_share_total": round(float(per_day["carried"].sum() / per_day["listed"].sum()), 5) if "carried" in per_day.columns else None,
        "nondegraded_median_carried_share": round(float(ok["carried_share"].median()), 5) if "carried_share" in ok.columns else None,
        "nondegraded_p10_carried_share": round(float(ok["carried_share"].quantile(0.1, "lower")), 5) if "carried_share" in ok.columns else None,
        "listed_median_feb_jul": float(per_day.filter(pl.col("day") < dt.date(2026, 8, 1))["listed"].median()) if "listed" in per_day.columns else None,
        "listed_median_aug": float(per_day.filter(pl.col("day") >= dt.date(2026, 8, 1))["listed"].median()) if "listed" in per_day.columns else None,
        "listed_peak": per_day.sort("listed", descending=True).select("day", "listed").row(0, named=True) | {} if "listed" in per_day.columns else None,
        "median_hour_rows_v2": int(sorted(v2h)[len(v2h) // 2]) if v2h else None,
        "thin_hours_n": sum(len(v) for v in thin.values()) + len(MIGRATION_HOURS & set(rows)),
    }
    if summary["listed_peak"]:
        summary["listed_peak"]["day"] = str(summary["listed_peak"]["day"])
    june = per_day.filter((pl.col("day") >= dt.date(2026, 6, 11)) & (pl.col("day") <= dt.date(2026, 6, 28)))
    mid = june.filter((pl.col("day") >= dt.date(2026, 6, 13)) & (pl.col("day") <= dt.date(2026, 6, 19)))
    july = per_day.filter((pl.col("day") >= dt.date(2026, 7, 8)) & (pl.col("day") <= dt.date(2026, 7, 14)))
    if "listed" in per_day.columns:
        summary["june_window"] = {"days": june.height, "listed_min": int(june["listed"].min()), "listed_max": int(june["listed"].max()),
                                  "first_seen_zero_days": int((june["first_seen_markets"] == 0).sum()),
                                  "carried_share_13_19_min": float(mid["carried_share"].min()), "carried_share_13_19_max": float(mid["carried_share"].max())}
        summary["july_window"] = {"days": july.height, "first_seen_markets": july["first_seen_markets"].to_list(),
                                  "first_seen_under_100_days": int((july["first_seen_markets"] < 100).sum())}
    out = {
        "summary": summary,
        "sensitivity": sensitivity(per_day, listings),
        "rule": {"window_days": WINDOW, "ratio": RATIO, "min_hours": MIN_HOURS, "min_carried": MIN_CARRIED, "venue_benchmark": src, "shards": source},
        "degraded_days": sorted(str(d) for d in flags),
        "thin_hours": thin, "thin_ratio": 0.2, "migration_hours_exempt": sorted(MIGRATION_HOURS),
        "days": per_day.with_columns(pl.col("day").cast(pl.Utf8)).to_dicts(),
    }
    write_json("d0_archive_health", out, inputs=[source, V2_BASE, str(sweep if sweep.exists() else pull)])
    print("degraded days:", out["degraded_days"])


if __name__ == "__main__":
    main()

"""Composition of the panel: the daily cross-section of spreads against the venue's listing rate.

The whole-panel median spread is a mixture statistic: the cross-section is bimodal (a tight
mode at one or two ticks and a wide mode above 0.5) and the set of markets changes every day.
This script gives the daily quantiles over live quotes, the tight share, the number of assets
observed, and, from the archive-health table, the number of markets the venue listed, so that
growth in the archive can be read against growth in the venue. It also counts the assets live
on both sides of the cutover and of the archive seam.

Run:  uv run python x3_composition.py
Out:  x3_composition.json
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from d0_common import CUTOVER, EXT, read_json, write_json

TIGHT = 0.02
COLS = ["asset_id", "day", "spread_median", "placeholder", "era", "updates"]


def _live(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.filter(~pl.col("placeholder") & (pl.col("spread_median") > 0))


def daily_cross_section(panel: pl.DataFrame, tight: float = TIGHT) -> pl.DataFrame:
    """Per day: assets observed, assets with a live quote, spread quantiles over live quotes, the
    share of live quotes at or below `tight`, and the era of the day's shard."""
    all_ = panel.group_by("day").agg(assets=pl.len(), era=pl.col("era").first())
    live = _live(panel).group_by("day").agg(
        live=pl.len(), p25=pl.col("spread_median").quantile(0.25, "linear"), median=pl.col("spread_median").median(),
        p75=pl.col("spread_median").quantile(0.75, "linear"), mean=pl.col("spread_median").mean(),
        tight_share=(pl.col("spread_median") <= tight).mean())
    return all_.join(live, on="day", how="left").sort("day")


def both_sides(panel: pl.DataFrame, before: dt.date, after: dt.date) -> dict:
    lv = _live(panel)
    b = set(lv.filter(pl.col("day") == before)["asset_id"].to_list())
    a = set(lv.filter(pl.col("day") == after)["asset_id"].to_list())
    return {"before": len(b), "after": len(a), "both": len(b & a), "share_of_before": len(b & a) / len(b) if b else None}


def distribution_weeks(panel: pl.DataFrame, before=("2026-04-21", "2026-04-27"), after=("2026-04-29", "2026-05-05")) -> dict:
    """Per-asset median spread over live quotes in the week before and after the cutover, as a
    histogram on log-spaced bins from 0.001 to 1, with the summary the text cites."""
    import numpy as np
    lv = _live(panel)
    def block(lo, hi):
        return (lv.filter((pl.col("day") >= dt.date.fromisoformat(lo)) & (pl.col("day") <= dt.date.fromisoformat(hi)))
                  .group_by("asset_id").agg(pl.col("spread_median").median().alias("s"))["s"].to_numpy())
    edges = np.logspace(np.log10(0.001), 0, 41)
    out = {"bin_edges": edges.tolist()}
    for key, (lo, hi) in (("before", before), ("after", after)):
        x = block(lo, hi)
        counts, _ = np.histogram(np.clip(x, edges[0], edges[-1]), bins=edges)
        out[key] = {"n": int(x.size), "median": float(np.median(x)), "share_above_0.5": float((x > 0.5).mean()),
                    "share_at_or_below_0.02": float((x <= 0.02).mean()), "share_by_bin": (counts / x.size).tolist()}
    return out


def main() -> None:
    panel = pl.read_parquet(EXT / "panel_v1v2.parquet", columns=COLS)
    cs = daily_cross_section(panel)
    health = read_json("d0_archive_health")
    hd = {r["day"]: r for r in health["days"]}
    degraded = set(health["degraded_days"])
    days = []
    for r in cs.with_columns(pl.col("day").cast(pl.Utf8)).to_dicts():
        h = hd.get(r["day"], {})
        days.append(r | {"listed": h.get("listed"), "first_seen_markets": h.get("first_seen_markets"), "degraded": r["day"] in degraded})
    ok = [d for d in days if not d["degraded"]]
    lv = _live(panel)
    summary = {
        "days": len(days), "rows": panel.height, "rows_by_era": {k: int(v) for k, v in panel.group_by("era").len().iter_rows()},
        "assets_per_day_min": min(d["assets"] for d in ok), "assets_per_day_max": max(d["assets"] for d in ok),
        "assets_first_ok_day": ok[0]["assets"], "assets_last_ok_day": ok[-1]["assets"], "first_ok_day": ok[0]["day"], "last_ok_day": ok[-1]["day"],
        "median_spread_min": min(d["median"] for d in ok), "median_spread_max": max(d["median"] for d in ok),
        "tight_share_min": min(d["tight_share"] for d in ok), "tight_share_max": max(d["tight_share"] for d in ok),
        "whole_panel": {"mean": float(lv["spread_median"].mean()), "p25": float(lv["spread_median"].quantile(0.25, "linear")),
                        "median": float(lv["spread_median"].median()), "p75": float(lv["spread_median"].quantile(0.75, "linear")),
                        "p95": float(lv["spread_median"].quantile(0.95, "linear")), "n_live": lv.height, "n": panel.height,
                        "updates_median": float(panel["updates"].median())},
        "listed_median_feb_jul": health["summary"]["listed_median_feb_jul"], "listed_median_aug": health["summary"]["listed_median_aug"],
        "cutover_both_sides": both_sides(panel, CUTOVER - dt.timedelta(days=1), CUTOVER),
        "seam_both_sides": both_sides(panel, dt.date(2026, 4, 14), dt.date(2026, 4, 16)),
    }
    write_json("x3_composition", {"summary": summary, "distribution_weeks": distribution_weeks(panel), "days": days}, inputs=[str(EXT / "panel_v1v2.parquet"), "d0_archive_health.json"])
    print({k: v for k, v in summary.items() if k not in ("whole_panel",)})
    print(summary["whole_panel"])


if __name__ == "__main__":
    main()

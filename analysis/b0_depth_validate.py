"""Is the depth jump at the cutover real or a change in what the feed reports?

The panel carries two measures of displayed size at the best bid: the mean size
of `price_change` rows at the best level (the panel's depth outcome) and the mean
best-level size of the `book` snapshots the feed sends when it (re)subscribes.
If the migration changed what a `price_change` size means, the two measures
diverge across 28 April; if the book itself changed, they move together.

Run:  uv run python b0_depth_validate.py
Out:  b0_depth_validate.json
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from d0_common import CUTOVER, EXT, write_json

COLS = ["asset_id", "day", "depth_bid_n", "depth_bid_sum", "book_bid_best_mean", "n_book"]


def daily_compare(panel: pl.DataFrame, event: dt.date, window: int = 7) -> pl.DataFrame:
    """Per day in [event - window, event + window]: both depth measures, their ratio, and the
    paired log change of each from the day before the event."""
    lo, hi = event - dt.timedelta(days=window), event + dt.timedelta(days=window)
    df = (panel.filter((pl.col("day") >= lo) & (pl.col("day") <= hi))
          .with_columns(
              pl.when(pl.col("depth_bid_n") > 0).then(pl.col("depth_bid_sum") / pl.col("depth_bid_n")).alias("pc"),
              pl.when(pl.col("n_book") > 0).then(pl.col("book_bid_best_mean")).alias("snap"))
          .filter(pl.col("pc") > 0))
    base = df.filter(pl.col("day") == event - dt.timedelta(days=1)).select("asset_id", pc0=pl.col("pc"), snap0=pl.col("snap"))
    j = df.join(base, on="asset_id", how="left").with_columns(
        (pl.col("pc") / pl.col("pc0")).log().alias("dlog_pc"),
        (pl.col("snap") / pl.col("snap0")).log().alias("dlog_snap"),
        (pl.col("pc") / pl.col("snap")).log().alias("lr"))
    return (j.group_by("day").agg(
        n_assets=pl.len(), n_with_snapshot=pl.col("snap").is_not_null().sum(),
        median_pc=pl.col("pc").median(), median_snap=pl.col("snap").median(),
        median_log_ratio_pc_over_snap=pl.col("lr").median(),
        dlog_pc=pl.col("dlog_pc").median(), dlog_snap=pl.col("dlog_snap").median(),
        n_paired_snap=pl.col("dlog_snap").is_not_null().sum())
        .sort("day").with_columns(((pl.col("day") - pl.lit(event)).dt.total_days()).alias("k")))


def main() -> None:
    panel = pl.read_parquet(EXT / "panel_v1v2.parquet", columns=COLS)
    out = daily_compare(panel, CUTOVER)
    rows = out.with_columns(pl.col("day").cast(pl.Utf8)).to_dicts()
    by_k = {r["k"]: r for r in rows}
    res = {"event": str(CUTOVER), "days": rows,
           "summary": {f"k={k}": {"dlog_pc": by_k[k]["dlog_pc"], "dlog_snap": by_k[k]["dlog_snap"], "n_paired_snap": by_k[k]["n_paired_snap"]}
                       for k in (0, 1, 2, 3, 7) if k in by_k}}
    write_json("b0_depth_validate", res, inputs=[str(EXT / "panel_v1v2.parquet")])
    for r in rows:
        print(r["day"], f"k={r['k']:+d}", f"n={r['n_assets']:,}", f"snap={r['n_with_snapshot']:,}", f"pc/snap={r['median_log_ratio_pc_over_snap']}",
              f"dlog_pc={r['dlog_pc']}", f"dlog_snap={r['dlog_snap']}")


if __name__ == "__main__":
    main()

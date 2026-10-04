"""Stack the harmonised daily shards of both archives into one contract-day panel.

Adds what the per-day extraction cannot know: the outcome label from the
token map, the first day an asset was seen across both eras (so V2-era ages
are not censored at 15 April), the age in days, the mean spread, whether the
day has all 24 hours, and the placeholder flag (a book quoted at or above
0.9 all day, or a midpoint pinned at exactly 0.500).

Run:  uv run python x2_build_panel.py
Out:  /Volumes/data/paper-b-ext/panel_v1v2.parquet, x2_panel_build.json
"""
from __future__ import annotations

import glob

import polars as pl

from d0_common import EXT, write_json

PLACEHOLDER_SPREAD = 0.9


def build(shards: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    lab = labels.select("asset_id", pl.col("condition_id").alias("cid_map"), pl.col("outcome").alias("outcome_map")).unique("asset_id")
    df = shards.join(lab, on="asset_id", how="left").with_columns(
        pl.coalesce("outcome_map", "outcome").alias("outcome"),
        pl.coalesce("condition_id", "cid_map").alias("condition_id"),
    ).drop("outcome_map", "cid_map")
    first = df.group_by("asset_id").agg(first_seen_day=pl.col("day").min())
    return (
        df.join(first, on="asset_id", how="left")
        .with_columns(
            (pl.col("spread_sum") / pl.col("updates")).alias("spread_mean"),
            (pl.col("hours_present") == 24).alias("day_complete"),
            (pl.col("day") - pl.col("first_seen_day")).dt.total_days().alias("age_days"),
            ((pl.col("spread_median") >= PLACEHOLDER_SPREAD) | (pl.col("updates_empty") == pl.col("updates"))).alias("placeholder"),
            (pl.col("updates_empty") / pl.col("updates")).alias("empty_share"),
        )
        .sort(["asset_id", "day"])
    )


def main() -> None:
    fs = sorted(glob.glob(str(EXT / "shards_h" / "era=*" / "day=*.parquet")))
    shards = pl.concat([pl.read_parquet(f) for f in fs], how="vertical_relaxed")
    labels = pl.read_parquet(EXT / "token_map.parquet", columns=["asset_id", "condition_id", "outcome"])
    p = build(shards, labels)
    p.write_parquet(EXT / "panel_v1v2.parquet")
    both = p.group_by("asset_id").agg(n_era=pl.col("era").n_unique()).filter(pl.col("n_era") == 2).height
    out = {
        "rows": p.height, "assets": p["asset_id"].n_unique(), "conditions": p["condition_id"].n_unique(),
        "rows_by_era": {k: int(v) for k, v in p.group_by("era").len().iter_rows()},
        "days_by_era": {k: int(v) for k, v in p.group_by("era").agg(pl.col("day").n_unique()).iter_rows()},
        "assets_in_both_eras": both,
        "labelled_share": round(float(p["outcome"].is_not_null().mean()), 5),
        "incomplete_days": {str(d): int(n) for d, n in p.filter(~pl.col("day_complete")).group_by("day").len().sort("day").iter_rows()},
        "placeholder_share": round(float(p["placeholder"].mean()), 5),
        "first_day": str(p["day"].min()), "last_day": str(p["day"].max()),
    }
    write_json("x2_panel_build", out, inputs=fs[:1] + ["…"] + fs[-1:] + [str(EXT / "token_map.parquet")])
    print(out)


if __name__ == "__main__":
    main()

"""What the 28 April change is consistent with: the order wipe, the fee model, anticipation.

Five bounded tests, none of which attributes the change:

  1. hourly window   cohort median spread and empty-book share by hour on 27 April
                     to 1 May: a rise confined to hours >= 11 with an empty-book
                     share near one at 12-13 is what the wipe predicts; an earlier
                     rise is anticipation or early cancellation
  2. recovery        half-life by pre-event activity quartile (standing market makers
                     repopulate active books faster)
  3. category DiD    days +5..+7 against -7..-1, fee-free Geopolitics against the rest:
                     the only fee evidence the data allow, reported as bounded
  4. archive seam    assets on 14 April (V1) and 16 April (V2): the paired change
                     across the format change should be about zero
  5. 30 March        the fee-category event as a dated comparison without a wipe

Run:  uv run python b3_mechanisms.py
Out:  b3_mechanisms.json, /Volumes/data/paper-b-ext/hourly_cutover_<cohort hash>.parquet
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from b1_cutover import PRE_DAYS, cohort, half_life, paths, with_category
from d0_common import CUTOVER, EXT, connect, day_files, write_json

HOURLY_DAYS = ["2026-04-27", "2026-04-28", "2026-04-29", "2026-04-30", "2026-05-01"]
REFERENCE_DAY = dt.date(2026, 4, 27)
CUT_HOUR = 11
SEAM = (dt.date(2026, 4, 14), dt.date(2026, 4, 16))
FEE_EVENT = dt.date(2026, 3, 30)


def anticipation(hourly: pl.DataFrame, event: dt.date, reference: dt.date, cut_hour: int = CUT_HOUR) -> dict:
    """Log ratio of the event day's hourly cohort median to the reference day's, before and after the cut."""
    e = hourly.filter(pl.col("day") == event).select("hour", pl.col("spread_median").alias("s_e"), pl.col("empty_share").alias("empty_e"))
    r = hourly.filter(pl.col("day") == reference).select("hour", pl.col("spread_median").alias("s_r"))
    j = e.join(r, on="hour", how="inner").with_columns((pl.col("s_e") / pl.col("s_r")).log().alias("lr")).sort("hour")
    pre, post = j.filter(pl.col("hour") < cut_hour), j.filter(pl.col("hour") >= cut_hour)
    first_empty = j.filter(pl.col("empty_e") > 0.5)["hour"]
    return {
        "cut_hour": cut_hour,
        "pre_hours_log_ratio_median": float(pre["lr"].median()) if pre.height else None,
        "post_hours_log_ratio_median": float(post["lr"].median()) if post.height else None,
        "max_empty_share_event_day": float(j["empty_e"].max()),
        "first_hour_empty_above_0_5": int(first_empty[0]) if first_empty.len() else None,
        "by_hour": j.select("hour", "lr", "empty_e").to_dicts(),
    }


def category_did(panel: pl.DataFrame, assets: list[str], event: dt.date, treated: str,
                 pre=(-PRE_DAYS, -1), post=(5, 7)) -> dict:
    sub = panel.filter(pl.col("asset_id").is_in(assets) & ~pl.col("placeholder") & (pl.col("spread_median") > 0))
    sub = sub.with_columns((pl.col("day") - pl.lit(event)).dt.total_days().alias("k"), pl.col("spread_median").log().alias("y"))
    lvl = (sub.filter((pl.col("k") >= pre[0]) & (pl.col("k") <= pre[1])).group_by("asset_id").agg(pl.col("y").median().alias("pre"), pl.col("category").first())
           .join(sub.filter((pl.col("k") >= post[0]) & (pl.col("k") <= post[1])).group_by("asset_id").agg(pl.col("y").median().alias("post")), on="asset_id", how="inner")
           .with_columns((pl.col("post") - pl.col("pre")).alias("chg"), (pl.col("category") == treated).alias("t")))
    tr, co = lvl.filter(pl.col("t")), lvl.filter(~pl.col("t"))
    t_chg = float(tr["chg"].median()) if tr.height else None
    c_chg = float(co["chg"].median()) if co.height else None
    return {"treated": treated, "treated_assets": tr.height, "control_assets": co.height,
            "treated_change": t_chg, "control_change": c_chg,
            "did": (t_chg - c_chg) if t_chg is not None and c_chg is not None else None}


def seam_placebo(panel: pl.DataFrame, d_v1: dt.date, d_v2: dt.date) -> dict:
    a = panel.filter((pl.col("day") == d_v1) & (pl.col("era") == "v1") & ~pl.col("placeholder")).select("asset_id", pl.col("spread_median").alias("s1"))
    b = panel.filter((pl.col("day") == d_v2) & (pl.col("era") == "v2") & ~pl.col("placeholder")).select("asset_id", pl.col("spread_median").alias("s2"))
    j = a.join(b, on="asset_id", how="inner").filter((pl.col("s1") > 0) & (pl.col("s2") > 0))
    d = (j["s2"] / j["s1"]).log()
    return {"v1_day": str(d_v1), "v2_day": str(d_v2), "assets": j.height,
            "median_log_change": float(d.median()) if j.height else None,
            "share_within_10pct": float((d.abs() < 0.1).mean()) if j.height else None}


def hourly_summary(hourly: pl.DataFrame) -> dict:
    """Per day: the range of the cohort's hourly median spread and empty-book share."""
    g = hourly.group_by("day").agg(spread_min=pl.col("spread_median").min(), spread_max=pl.col("spread_median").max(),
                                   empty_min=pl.col("empty_share").min(), empty_max=pl.col("empty_share").max(),
                                   empty_after_15=pl.col("empty_share").filter(pl.col("hour") >= 15).max()).sort("day")
    return {str(r["day"]): {k: v for k, v in r.items() if k != "day"} for r in g.to_dicts()}


def hourly_extract(days: list[str], assets: list[str], con=None, files_by_day: dict[str, list[str]] | None = None) -> pl.DataFrame:
    """Per (day, hour): the cohort's cross-sectional median of per-asset median spread, and the empty-book share."""
    con = con or connect()
    con.register("cohort_tmp", pl.DataFrame({"asset_id": assets}).to_arrow())
    out = []
    for d in days:
        fs = (files_by_day or {}).get(d) or day_files(d, "v2")
        files = "[" + ", ".join(f"'{f}'" for f in fs) + "]"
        q = f"""
        WITH h AS (
          SELECT asset_id, hour(timestamp_received::TIMESTAMP) AS hour,
                 quantile_cont(CAST(best_ask AS DOUBLE) - CAST(best_bid AS DOUBLE), 0.5) AS s,
                 arg_max(CAST(best_ask AS DOUBLE) - CAST(best_bid AS DOUBLE), timestamp_received) AS s_last
          FROM read_parquet({files})
          WHERE event_type = 'price_change' AND best_bid IS NOT NULL AND best_ask IS NOT NULL
            AND CAST(best_ask AS DOUBLE) >= CAST(best_bid AS DOUBLE)
            AND asset_id IN (SELECT asset_id FROM cohort_tmp)
          GROUP BY 1, 2
        )
        SELECT hour, count(*) AS assets, quantile_cont(s, 0.5) AS spread_median,
               avg(CASE WHEN round(s_last, 6) >= 0.9 THEN 1.0 ELSE 0.0 END) AS empty_share
        FROM h GROUP BY 1 ORDER BY 1
        """
        df = pl.from_arrow(con.execute(q).to_arrow_table())
        assert isinstance(df, pl.DataFrame)
        out.append(df.with_columns(pl.lit(d).str.to_date().alias("day")))
    con.unregister("cohort_tmp")
    return pl.concat(out).select("day", "hour", "assets", "spread_median", "empty_share")


def main() -> None:
    panel = with_category(pl.read_parquet(EXT / "panel_v1v2.parquet"))
    c = cohort(panel, CUTOVER)
    res = {"event": str(CUTOVER), "cohort": len(c)}
    # the cache is keyed by the cohort, so a rebuilt panel (a different cohort) never reuses an old extract
    import hashlib
    key = hashlib.sha256("\n".join(sorted(c)).encode()).hexdigest()[:12]
    hp = EXT / f"hourly_cutover_{key}.parquet"
    if hp.exists():
        hourly = pl.read_parquet(hp)
    else:
        hourly = hourly_extract(HOURLY_DAYS, c)
        hourly.write_parquet(hp)
    res["anticipation"] = anticipation(hourly, CUTOVER, REFERENCE_DAY)
    res["hourly"] = hourly.to_dicts()
    res["hourly_summary"] = hourly_summary(hourly)
    # recovery by pre-event activity quartile
    pre = (panel.filter(pl.col("asset_id").is_in(c) & (pl.col("day") < CUTOVER) & (pl.col("day") >= CUTOVER - dt.timedelta(days=PRE_DAYS)))
           .group_by("asset_id").agg(pl.col("updates").mean().alias("u")))
    qs = [pre["u"].quantile(q) for q in (0.25, 0.5, 0.75)]
    res["recovery_by_activity_quartile"] = {}
    for i, (lo, hi) in enumerate(zip([None] + qs, qs + [None])):
        g = pre.filter(((pl.col("u") > lo) if lo is not None else pl.lit(True)) & ((pl.col("u") <= hi) if hi is not None else pl.lit(True)))["asset_id"].to_list()
        if len(g) < 100:
            continue
        p = paths(panel, g, CUTOVER, k_range=range(-1, 8), outcome="log_spread", draws=300)
        res["recovery_by_activity_quartile"][f"q{i+1}"] = {"assets": len(g), "half_life_days": half_life(p), "k0": p.filter(pl.col("k") == 0)["median"][0]}
    if "category" in panel.columns:
        # the fee mechanism: the venue's categories carry different taker rates; compare each against the rest
        res["category_did"] = {cat: category_did(panel, c, CUTOVER, treated=cat)
                               for cat in panel.filter(pl.col("asset_id").is_in(c))["category"].drop_nulls().unique().to_list()}
        rates = panel.filter(pl.col("asset_id").is_in(c)).group_by("category").agg(pl.col("fee_rate").median().alias("fee_rate"), pl.len().alias("assets"))
        res["category_fee_rates"] = rates.to_dicts()
    res["archive_seam"] = seam_placebo(panel, *SEAM)
    cf = cohort(panel, FEE_EVENT)
    if len(cf) >= 1000:
        pf = paths(panel, cf, FEE_EVENT, k_range=range(-3, 8), outcome="log_spread", draws=300)
        pu = paths(panel, cf, FEE_EVENT, k_range=range(-3, 8), outcome="log_updates", draws=300)
        res["fee_event_2026_03_30"] = {"cohort": len(cf), "log_spread": pf.to_dicts(), "log_updates": pu.to_dicts()}
    write_json("b3_mechanisms", res, inputs=[str(EXT / "panel_v1v2.parquet"), str(hp)])
    a = res["anticipation"]
    print(f"cohort {len(c):,}; pre-cut log ratio {a['pre_hours_log_ratio_median']}, post-cut {a['post_hours_log_ratio_median']}, "
          f"first empty hour {a['first_hour_empty_above_0_5']}; seam {res['archive_seam']}")


if __name__ == "__main__":
    main()

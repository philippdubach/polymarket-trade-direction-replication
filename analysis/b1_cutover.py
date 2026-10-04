"""The 28 April cutover as a paired within-asset event study on the 170-day panel.

Cohort: assets live (median spread above zero and below 0.9) on each of the
seven days before the event. Outcomes per asset-day: log median spread (to
0.001), log updates, log displayed size at the best bid, the share of the day's
updates that show an empty quote, and log(1 + trades) (V2 only). For each event day k in [-7, 7] the paired
change from day E-1 is summarised by its median and mean over the cohort,
with an asset-cluster percentile bootstrap. Also: the recovery half-life,
attrition bounds at k = 7, and three pre-specified splits (age at the event,
pre-period activity quartile, category).

The same machinery serves the placebo dates in b2_placebo_dates.py, which is
where the inference lives: this script describes 28 April, it does not
attribute it.

Run:  uv run python b1_cutover.py
Out:  b1_cutover.json
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl

from d0_common import CUTOVER, EXT, gamma_categories, write_json

PRE_DAYS = 7
K_RANGE = range(-7, 8)
OUTCOMES = {
    "log_spread": lambda: pl.col("spread_median").log(),
    "log_updates": lambda: pl.col("updates").cast(pl.Float64).log(),
    "log_depth_bid": lambda: (pl.col("depth_bid_sum") / pl.col("depth_bid_n")).log(),
    "empty_share": lambda: pl.col("empty_share"),
    "log1p_trades": lambda: (1 + pl.col("trades").cast(pl.Float64)).log(),
}


def cohort(panel: pl.DataFrame, event: dt.date, pre_days: int = PRE_DAYS) -> list[str]:
    days = [event - dt.timedelta(days=i) for i in range(1, pre_days + 1)]
    ok = (panel.filter(pl.col("day").is_in(days) & ~pl.col("placeholder") & (pl.col("spread_median") > 0))
          .group_by("asset_id").agg(pl.col("day").n_unique().alias("n")).filter(pl.col("n") == pre_days))
    return sorted(ok["asset_id"].to_list())


def _expr(outcome: str) -> pl.Expr:
    """A named outcome, or the log of a raw panel column."""
    return OUTCOMES[outcome]() if outcome in OUTCOMES else pl.col(outcome).log()


def _value(panel: pl.DataFrame, assets: list[str], day: dt.date, outcome: str) -> pl.DataFrame:
    return (panel.filter((pl.col("day") == day) & pl.col("asset_id").is_in(assets))
            .select("asset_id", _expr(outcome).alias("y"))
            .filter(pl.col("y").is_finite()))


def boot_median(x: pl.Series, draws: int = 1000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = x.drop_nulls().to_numpy()
    if a.size == 0:
        return float("nan"), float("nan")
    meds = np.array([np.median(a[rng.integers(0, a.size, a.size)]) for _ in range(draws)])
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def paths(panel: pl.DataFrame, assets: list[str], event: dt.date, k_range=K_RANGE, outcome: str = "log_spread",
          draws: int = 1000) -> pl.DataFrame:
    base = _value(panel, assets, event - dt.timedelta(days=1), outcome).rename({"y": "y0"})
    rows = []
    for k in k_range:
        v = _value(panel, assets, event + dt.timedelta(days=k), outcome).join(base, on="asset_id", how="inner")
        d = (v["y"] - v["y0"])
        lo, hi = boot_median(d, draws) if draws else (None, None)
        up, down = (float((d > 0).mean()), float((d < 0).mean())) if v.height else (None, None)
        rows.append({"k": k, "n": v.height, "median": float(d.median()) if v.height else None,
                     "mean": float(d.mean()) if v.height else None, "lo": lo, "hi": hi,
                     "share_up": up, "share_down": down, "net_up": (up - down) if v.height else None})
    return pl.DataFrame(rows)


def half_life(p: pl.DataFrame) -> int | None:
    r = {row["k"]: row["median"] for row in p.to_dicts()}
    if 0 not in r or r[0] is None or r[0] == 0:
        return None
    for k in sorted(x for x in r if x > 0):
        if r[k] is not None and abs(r[k]) <= 0.5 * abs(r[0]):
            return k
    return None


# the adverse direction of each outcome for the leavers' bound: a fall in activity or depth is adverse
ADVERSE_QUANTILE = {"log_updates": 0.1, "log_depth_bid": 0.1, "log1p_trades": 0.1}


def attrition_bounds(panel: pl.DataFrame, assets: list[str], event: dt.date, k: int, outcome: str = "log_spread") -> dict:
    base = _value(panel, assets, event - dt.timedelta(days=1), outcome).rename({"y": "y0"})
    post = _value(panel, assets, event + dt.timedelta(days=k), outcome)
    j = base.join(post, on="asset_id", how="left")
    surv = j.filter(pl.col("y").is_not_null())
    point = float((surv["y"] - surv["y0"]).median()) if surv.height else float("nan")
    # worst case: a non-survivor moved to the adverse tail of the cohort's pre-period distribution (the 90th
    # percentile where a rise is adverse, the 10th where a fall is); best case: it stayed put
    q = ADVERSE_QUANTILE.get(outcome, 0.9)
    pre = panel.filter(pl.col("asset_id").is_in(assets) & (pl.col("day") < event) & (pl.col("day") >= event - dt.timedelta(days=PRE_DAYS)))
    tail = float(pre.select(_expr(outcome).alias("y")).filter(pl.col("y").is_finite())["y"].quantile(q))
    worst = j.with_columns(pl.col("y").fill_null(tail))
    best = j.with_columns(pl.col("y").fill_null(pl.col("y0")))
    return {"k": k, "cohort": j.height, "survivors": surv.height, "point": point, "adverse_quantile": q,
            "worst": float((worst["y"] - worst["y0"]).median()), "best": float((best["y"] - best["y0"]).median())}


def splits(panel: pl.DataFrame, assets: list[str], event: dt.date) -> dict[str, dict[str, list[str]]]:
    """Three pre-specified cohort splits: age at the event, pre-period activity tercile, category."""
    sub = panel.filter(pl.col("asset_id").is_in(assets))
    first = sub.group_by("asset_id").agg(pl.col("first_seen_day").min().alias("f"))
    old = first.filter(pl.col("f") < event - dt.timedelta(days=14))["asset_id"].to_list()
    out = {"age": {"listed_14d_or_more_before": old, "newer": sorted(set(assets) - set(old))}}
    pre = (sub.filter((pl.col("day") < event) & (pl.col("day") >= event - dt.timedelta(days=PRE_DAYS)))
           .group_by("asset_id").agg(pl.col("updates").mean().alias("u")))
    q1, q2 = pre["u"].quantile(1 / 3), pre["u"].quantile(2 / 3)
    out["activity"] = {"low": pre.filter(pl.col("u") <= q1)["asset_id"].to_list(),
                       "mid": pre.filter((pl.col("u") > q1) & (pl.col("u") <= q2))["asset_id"].to_list(),
                       "high": pre.filter(pl.col("u") > q2)["asset_id"].to_list()}
    if "category" in panel.columns:
        cats = sub.group_by("asset_id").agg(pl.col("category").first())
        out["category"] = {str(c): g["asset_id"].to_list() for (c,), g in cats.group_by(["category"]) if c is not None}
    return out


def with_category(panel: pl.DataFrame) -> pl.DataFrame:
    gp = EXT / "gamma" / "gamma_markets.parquet"
    if not gp.exists() or "condition_id" not in panel.columns:
        return panel
    return panel.join(gamma_categories().select("condition_id", "category", "fee_rate"), on="condition_id", how="left")


def main() -> None:
    panel = with_category(pl.read_parquet(EXT / "panel_v1v2.parquet"))
    c = cohort(panel, CUTOVER)
    res = {"event": str(CUTOVER), "cohort": len(c), "outcomes": {}, "attrition": {}, "splits": {}}
    for name in OUTCOMES:
        p = paths(panel, c, CUTOVER, outcome=name)
        res["outcomes"][name] = {"paths": p.to_dicts(), "half_life_days": half_life(p)}
        res["attrition"][name] = attrition_bounds(panel, c, CUTOVER, k=7, outcome=name)
    sp = splits(panel, c, CUTOVER)
    for dim, groups in sp.items():
        res["splits"][dim] = {}
        for g, assets in groups.items():
            if len(assets) < 100:
                continue
            p = paths(panel, assets, CUTOVER, k_range=[-7, -3, -1, 0, 1, 3, 7], outcome="log_spread", draws=300)
            res["splits"][dim][g] = {"assets": len(assets), "paths": p.to_dicts(), "half_life_days": half_life(p)}
    write_json("b1_cutover", res, inputs=[str(EXT / "panel_v1v2.parquet")])
    s = res["outcomes"]["log_spread"]
    k0 = next(r for r in s["paths"] if r["k"] == 0)
    print(f"cohort {len(c):,}; log spread at k=0: median {k0['median']:+.3f} [{k0['lo']:+.3f}, {k0['hi']:+.3f}] n={k0['n']:,}; half-life {s['half_life_days']} d")


if __name__ == "__main__":
    main()

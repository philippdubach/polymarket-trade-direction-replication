"""Staggered event study of tick adoption with not-yet-treated controls, as functions.

Reproduces t2_event_study.py (Callaway-Sant'Anna group-time ATTs aggregated
by event time, cohort bootstrap) and adds the two things the plan asks for:

  * cohorts and control days inside the degraded capture windows are excluded
  * one propensity-reweighted specification: controls reweighted to the treated
    cohort's state on the base day by p/(1-p) from the c0 hazard model, with
    its own pre-trend test

The unweighted, unexcluded run must reproduce t2_event_study.json to the
printed digits before either change is applied: that is the gate.

Run:  uv run python c2_event_study.py
Out:  c2_event_study.json
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl

from d0_common import EXT, degraded_days, read_json, write_json
from rev_common import CACHE, adoption, panel as pilot_panel

E_MIN, E_MAX = -7, 7
B = 300


def base_panel() -> pl.DataFrame:
    """The pilot's contract-day panel with the UTC adoption day, as t2 built it."""
    p = pilot_panel().filter(pl.col("s") > 0).select("asset_id", "day", pl.col("s").log().alias("y"), "trades")
    ad = adoption().select("asset_id", g=pl.col("adoption").dt.date())
    return p.join(ad, on="asset_id", how="left")


def pairs(p: pl.DataFrame, liquid: bool, sample_mod: int = 5, excluded_days: set[dt.date] | None = None) -> pl.DataFrame:
    """Every (unit, base day b, day t) with t in the event window around g = b + 1."""
    if excluded_days:
        p = p.filter(~pl.col("day").is_in(list(excluded_days)) & (pl.col("g").is_null() | ~pl.col("g").is_in(list(excluded_days))))
    base = p.rename({"day": "b", "y": "y_b", "trades": "tr_b"})
    if liquid:
        base = base.filter(pl.col("tr_b") >= 1)
    elif sample_mod > 1:
        base = base.filter(pl.col("asset_id").hash(seed=0) % sample_mod == 0)
    offsets = pl.DataFrame({"k": list(range(E_MIN + 1, E_MAX + 2))})
    bt = base.join(offsets, how="cross").with_columns(t=pl.col("b") + pl.duration(days=pl.col("k")))
    at_t = p.select("asset_id", pl.col("day").alias("t"), pl.col("y").alias("y_t"))
    return bt.join(at_t, on=["asset_id", "t"], how="inner").with_columns(dy=pl.col("y_t") - pl.col("y_b"))


def att_table(pr: pl.DataFrame, treated_ids: pl.DataFrame | None = None, weights: pl.DataFrame | None = None) -> pl.DataFrame:
    trp = pr if treated_ids is None else pr.join(treated_ids, on="asset_id", how="semi")
    tr = (trp.filter(pl.col("g") == pl.col("b") + pl.duration(days=1))
          .group_by(["g", "t"]).agg(tr_mean=pl.col("dy").mean(), n_tr=pl.len()))
    ct = pr.filter(pl.col("g").is_null() | (pl.col("g") > pl.col("t"))).with_columns(g_ref=pl.col("b") + pl.duration(days=1))
    if weights is not None:
        ct = ct.join(weights, on="asset_id", how="left").with_columns(pl.col("w").fill_null(1.0))
        ct = ct.group_by(["g_ref", "t"]).agg(ct_mean=(pl.col("dy") * pl.col("w")).sum() / pl.col("w").sum(), n_ct=pl.len())
    else:
        ct = ct.group_by(["g_ref", "t"]).agg(ct_mean=pl.col("dy").mean(), n_ct=pl.len())
    ct = ct.rename({"g_ref": "g"})
    return (tr.join(ct, on=["g", "t"], how="inner")
            .with_columns(att=pl.col("tr_mean") - pl.col("ct_mean"), e=(pl.col("t") - pl.col("g")).dt.total_days()))


def aggregate(tab: pl.DataFrame) -> dict[int, float]:
    agg = tab.group_by("e").agg(att=(pl.col("att") * pl.col("n_tr")).sum() / pl.col("n_tr").sum()).sort("e")
    return dict(zip(agg["e"].to_list(), agg["att"].to_list()))


def run(pr: pl.DataFrame, treated_ids=None, weights=None, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    tab = att_table(pr, treated_ids, weights)
    point = aggregate(tab)
    cohorts = tab["g"].unique().to_list()
    draws = {e: [] for e in point}
    pre_joint = []
    for _ in range(B):
        pick = rng.choice(len(cohorts), len(cohorts), replace=True)
        w = pl.DataFrame({"g": [cohorts[i] for i in pick]}).group_by("g").agg(w=pl.len())
        bt = tab.join(w, on="g").with_columns(nw=pl.col("n_tr") * pl.col("w"))
        agg = bt.group_by("e").agg(att=(pl.col("att") * pl.col("nw")).sum() / pl.col("nw").sum())
        d = dict(zip(agg["e"].to_list(), agg["att"].to_list()))
        for e in point:
            draws[e].append(d.get(e, np.nan))
        pre_joint.append(np.nanmean([d.get(e, np.nan) for e in range(E_MIN, -1)]))
    ci = {e: (float(np.nanpercentile(draws[e], 2.5)), float(np.nanpercentile(draws[e], 97.5))) for e in point}
    pre_mean = float(np.mean([point[e] for e in range(E_MIN, -1) if e in point]))
    pre_draws = np.array(pre_joint)
    centred = pre_draws - pre_draws.mean()
    post = [point[e] for e in range(0, E_MAX + 1) if e in point]
    return {"att_by_event_time": {int(e): point[e] for e in sorted(point)}, "ci95": {int(e): ci[e] for e in sorted(ci)},
            "pre_mean_att": pre_mean, "pre_trend_p": float(np.mean(np.abs(centred) >= abs(pre_mean))),
            "post_mean_att": float(np.mean(post)), "post_mean_ratio": float(np.exp(np.mean(post))),
            "treated_units_e0": int(tab.filter(pl.col("e") == 0)["n_tr"].sum()), "cohorts": len(cohorts)}


def propensity_weights(p: pl.DataFrame) -> pl.DataFrame | None:
    """p/(1-p) from the c0 hazard coefficients applied to each unit's state on its base days (mean over days)."""
    try:
        fit = read_json("c0_rollout")["hazard"]
    except FileNotFoundError:
        return None
    coef = fit["coef"]
    hp = EXT / "panel_v1v2.parquet"
    if not hp.exists():
        return None
    hpan = pl.read_parquet(hp, columns=["asset_id", "day", "updates", "trades", "placeholder", "mid_last", "spread_median", "age_days"])
    gp = EXT / "gamma" / "gamma_markets.parquet"
    if gp.exists():
        tm = pl.read_parquet(EXT / "token_map.parquet", columns=["asset_id", "condition_id"])
        hpan = hpan.join(tm, on="asset_id", how="left").join(pl.read_parquet(gp, columns=["condition_id", "neg_risk"]), on="condition_id", how="left")
    else:
        hpan = hpan.with_columns(pl.lit(False).alias("neg_risk"))
    z = (coef["const"]
         + coef["log_updates"] * (pl.col("updates").cast(pl.Float64) + 1).log()
         + coef["traded"] * (pl.col("trades").fill_null(0) > 0).cast(pl.Float64)
         + coef["placeholder"] * pl.col("placeholder").cast(pl.Float64)
         + coef["mid_extreme"] * ((pl.col("mid_last") > 0.96) | (pl.col("mid_last") < 0.04)).cast(pl.Float64)
         + coef["at_floor"] * (pl.col("spread_median") <= 0.0105).cast(pl.Float64)
         + coef["age_days"] * pl.col("age_days").cast(pl.Float64)
         + coef["neg_risk"] * pl.col("neg_risk").fill_null(False).cast(pl.Float64))
    pw = hpan.with_columns((1 / (1 + (-z).exp())).alias("ph")).with_columns((pl.col("ph") / (1 - pl.col("ph"))).alias("odds"))
    return pw.group_by("asset_id").agg(pl.col("odds").mean().alias("w")).with_columns(pl.col("w").clip(0.0, 20.0))


def main() -> None:
    p = base_panel()
    ev = pl.read_parquet(CACHE / "events_W7.parquet")
    stable_ids = ev.filter((pl.col("tr_pre") >= 1) & (pl.col("dlog_u").abs() <= math.log(1.25))).select("asset_id")
    specs = {"all": (False, None), "liquid": (True, None), "active_stable": (True, stable_ids)}
    res = {"gate": {}, "excluded": {}, "reweighted": {}}
    # 1. gate: reproduce t2_event_study.json
    ref = read_json("t2_event_study")
    for name, (liquid, ids) in specs.items():
        r = run(pairs(p, liquid), ids)
        res["gate"][name] = r
        ok = all(abs(r["att_by_event_time"][e] - ref[name]["att_by_event_time"][str(e)]) < 1e-9 for e in r["att_by_event_time"])
        res["gate"][name]["reproduced"] = ok
        print(f"gate {name}: {'REPRODUCED' if ok else 'MISMATCH'} pre {r['pre_mean_att']:+.3f} post {r['post_mean_att']:+.3f}")
    # 2. degraded windows excluded
    ex = degraded_days()
    for name, (liquid, ids) in specs.items():
        r = run(pairs(p, liquid, excluded_days=ex), ids)
        res["excluded"][name] = r
        print(f"excluded {name}: cohorts {r['cohorts']} pre {r['pre_mean_att']:+.3f} (p={r['pre_trend_p']:.2f}) post {r['post_mean_att']:+.3f}")
    res["excluded"]["degraded_days"] = sorted(str(d) for d in ex)
    # 3. propensity-reweighted controls (liquid spec, degraded excluded)
    w = propensity_weights(p)
    if w is not None:
        for name, (liquid, ids) in specs.items():
            r = run(pairs(p, liquid, excluded_days=ex), ids, weights=w)
            res["reweighted"][name] = r
            print(f"reweighted {name}: pre {r['pre_mean_att']:+.3f} (p={r['pre_trend_p']:.2f}) post {r['post_mean_att']:+.3f}")
        res["reweighted"]["weight_summary"] = {"units": w.height, "p50": float(w["w"].median()), "p90": float(w["w"].quantile(0.9))}
    write_json("c2_event_study", res, inputs=[str(CACHE / "events_W7.parquet"), "t2_event_study.json", "c0_rollout.json"])
    if not all(v.get("reproduced") for v in res["gate"].values()):
        raise SystemExit("event-study gate failed")


if __name__ == "__main__":
    main()

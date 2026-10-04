"""Check the venue's tick rule against the events themselves.

Polymarket's documentation of the tick_size_change message stated (archived
15 June 2025) that the tick changes when the book's price passes 0.96 or 0.04;
the current page no longer carries the sentence. Each event gets one class,
assigned in this order (the classes overlap; event_structure reports it):

  R  documented price rule: the last live quote within 60 minutes had a
     midpoint above 0.96 or below 0.04
  W  the dated 2 July World Cup decimalisation (new tick 0.0025)
  L  activation: within 24 hours of the asset's first appearance, or before
     its first live quote, or before its first trade
  B  operator batch: at least 50 events within 60 seconds on the same Gamma
     event
  X  residual

and a discrete-time hazard of adoption on the asset-day's own state the day
before (activity, placeholder book, price extremeness, floor, age, neg-risk),
fitted on all adoptions plus a 5% sample of non-adopting asset-days.
"Adoption is activation by construction" is claimed only if most L events
fall within an hour of the first live quote and the fitted hazard for a
placeholder asset-day is a small fraction of that for a first active day.

Run:  uv run python c0_rollout_rule.py
Out:  c0_rollout.json, /Volumes/data/paper-b-ext/ticks_enriched.parquet
"""
from __future__ import annotations

import datetime as dt
import glob

import numpy as np
import polars as pl

from d0_common import EXT, degraded_days, write_json

BATCH_MIN, BATCH_WINDOW = 50, "60s"
QUOTE_LOOKBACK = dt.timedelta(minutes=60)
ACTIVATION_WINDOW = dt.timedelta(hours=24)
WC_TICK = 0.0025
COVARIATES = ["log_updates", "traded", "placeholder", "mid_extreme", "at_floor", "age_days", "neg_risk"]


def classify(ev: pl.DataFrame, batch_min: int = BATCH_MIN, batch_window: str = BATCH_WINDOW) -> pl.DataFrame:
    d = ev.sort("ts")
    # rounded to 1e-6: a difference or mean of two doubles can miss a threshold it sits on
    mid = ((pl.col("bid_before") + pl.col("ask_before")) / 2).round(6)
    live = (pl.col("ask_before") - pl.col("bid_before")).round(6) < 0.9
    recent = (pl.col("ts") - pl.col("quote_before_ts")) <= QUOTE_LOOKBACK
    is_r = pl.col("bid_before").is_not_null() & live & recent & ((mid > 0.96) | (mid < 0.04))
    is_w = pl.col("new_tick") == WC_TICK
    is_l = ((pl.col("ts") - pl.col("first_seen_ts")) <= ACTIVATION_WINDOW) \
        | pl.col("first_live_quote_ts").is_null() | (pl.col("first_live_quote_ts") >= pl.col("ts")) \
        | pl.col("first_trade_ts").is_null() | (pl.col("first_trade_ts") >= pl.col("ts"))
    # batch: events within +/- window on the same Gamma event
    win = d.with_columns(pl.col("event_id").fill_null("__none__")).rolling(
        index_column="ts", period=f"{2 * int(batch_window.rstrip('s'))}s", offset=f"-{batch_window}", group_by="event_id"
    ).agg(pl.len().alias("n_window"))
    d = d.with_columns(pl.col("event_id").fill_null("__none__")).join(win, on=["event_id", "ts"], how="left").unique(["asset_id", "ts"], keep="first")
    is_b = pl.col("n_window") >= batch_min
    klass = (pl.when(is_r).then(pl.lit("R")).when(is_w).then(pl.lit("W")).when(is_l).then(pl.lit("L"))
             .when(is_b).then(pl.lit("B")).otherwise(pl.lit("X")))
    return d.with_columns(
        klass.alias("klass"), mid.alias("mid_before"),
        ((pl.col("first_live_quote_ts") - pl.col("ts")).dt.total_seconds() / 60.0).alias("mins_to_first_live_quote"),
        pl.col("ts").dt.hour().alias("hour"),
    ).sort(["asset_id", "ts"])


def hazard_fit(df: pl.DataFrame, covariates: list[str], weight_nonadopters: float = 1.0) -> dict:
    """Weighted logistic regression of `adopt` on the covariates; AUC by rank statistic."""
    import statsmodels.api as sm

    X = df.select(covariates).fill_null(0.0).to_numpy().astype(float)
    y = df["adopt"].cast(pl.Int64).to_numpy()
    w = np.where(y == 1, 1.0, weight_nonadopters)
    Xc = sm.add_constant(X, has_constant="add")
    res = sm.GLM(y, Xc, family=sm.families.Binomial(), freq_weights=w).fit()
    p = res.predict(Xc)
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")
    names = ["const"] + covariates
    return {"n": int(len(y)), "adoptions": int(n1), "coef": dict(zip(names, map(float, res.params))),
            "se": dict(zip(names, map(float, res.bse))), "auc": float(auc)}


def _first_timestamps() -> pl.DataFrame:
    fs = sorted(glob.glob(str(EXT / "shards_h" / "era=*" / "day=*.parquet")))
    sh = pl.concat([pl.read_parquet(f, columns=["asset_id", "first_quote_ts", "first_live_quote_ts", "first_trade_ts"]) for f in fs], how="vertical_relaxed")
    return sh.group_by("asset_id").agg(pl.col("first_quote_ts").min().alias("first_seen_ts"),
                                        pl.col("first_live_quote_ts").min(), pl.col("first_trade_ts").min())


def build_events() -> pl.DataFrame:
    tc = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(EXT / "tickctx" / "day=*.parquet")))])
    tc = tc.with_columns(pl.col("ts").cast(pl.Datetime("us")), pl.col("quote_before_ts").cast(pl.Datetime("us")))
    ev = tc.join(_first_timestamps(), on="asset_id", how="left")
    gp = EXT / "gamma" / "gamma_markets.parquet"
    if gp.exists():
        g = pl.read_parquet(gp, columns=["condition_id", "event_id", "created_at", "neg_risk", "end_date"])
        ev = ev.join(g, on="condition_id", how="left")
    else:
        ev = ev.with_columns(pl.lit(None, dtype=pl.Utf8).alias("event_id"))
    return ev


def hazard_sample(panel: pl.DataFrame, adoptions: pl.DataFrame, frac: float = 0.05, seed: int = 7,
                  degraded: set | None = None) -> pl.DataFrame:
    """Asset-days at risk: state on day t-1, outcome adopt on day t.

    `adoptions` carries every 0.01 -> 0.001 event of the sample, degraded days included, so that an
    asset leaves the risk set the day it changes tick and one that changed before the window is
    never in it. Days in `degraded` are dropped from the at-risk rows as well as from the outcomes."""
    ad = adoptions.select("asset_id", pl.col("ts").dt.date().alias("adopt_day")).group_by("asset_id").agg(pl.col("adopt_day").min())
    p = panel.join(ad, on="asset_id", how="left")
    # at risk on day t if not yet adopted by t-1; adopt on t
    p = p.with_columns((pl.col("day") + dt.timedelta(days=1)).alias("t"))
    p = p.filter(pl.col("adopt_day").is_null() | (pl.col("t") <= pl.col("adopt_day")))
    if degraded:   # the outcome on a degraded day is unobservable; the state the day before is kept
        p = p.filter(~pl.col("t").is_in(sorted(degraded)))
    p = p.with_columns((pl.col("t") == pl.col("adopt_day")).fill_null(False).alias("adopt"))
    feats = p.select(
        "asset_id", "t", "adopt",
        (pl.col("updates").cast(pl.Float64) + 1).log().alias("log_updates"),
        (pl.col("trades").fill_null(0) > 0).cast(pl.Float64).alias("traded"),
        pl.col("placeholder").cast(pl.Float64),
        ((pl.col("mid_last").round(6) > 0.96) | (pl.col("mid_last").round(6) < 0.04)).cast(pl.Float64).alias("mid_extreme"),
        (pl.col("spread_median") <= 0.0105).cast(pl.Float64).alias("at_floor"),
        pl.col("age_days").cast(pl.Float64),
        pl.col("neg_risk").cast(pl.Float64) if "neg_risk" in p.columns else pl.lit(0.0).alias("neg_risk"),
    )
    keep = feats.filter(pl.col("adopt")).vstack(feats.filter(~pl.col("adopt")).sample(fraction=frac, seed=seed))
    return keep


def event_structure(ev: pl.DataFrame, activation_window: dt.timedelta = ACTIVATION_WINDOW) -> dict:
    """How the classes overlap and what the events are at market level: per class, the share of
    events that also meet the activation criteria and the share on assets with no archived trade
    before the event; and the number of market-level changes (events of one market within a minute
    count once)."""
    # the same criteria as classify(): an asset with no live quote or no trade in the archive counts
    # as one whose first came after the event
    act = (((pl.col("ts") - pl.col("first_seen_ts")) <= activation_window)
           | pl.col("first_live_quote_ts").is_null() | (pl.col("first_live_quote_ts") >= pl.col("ts"))
           | pl.col("first_trade_ts").is_null() | (pl.col("first_trade_ts") >= pl.col("ts")))
    e = ev.with_columns(act.fill_null(False).alias("meets_activation"),
                        (pl.col("first_trade_ts").is_null() | (pl.col("first_trade_ts") >= pl.col("ts"))).alias("no_prior_trade"),
                        pl.col("ts").dt.truncate("1m").alias("minute"))
    out: dict = {"market_level_changes": int(e.select("condition_id", "minute").unique().height), "token_level_events": e.height}
    for k, g in e.group_by("klass"):
        out[k[0]] = {"n": g.height, "share_meeting_activation": float(g["meets_activation"].mean()),
                     "share_no_prior_trade": float(g["no_prior_trade"].mean())}
    return out


def price_at_event(ev: pl.DataFrame) -> dict:
    """Where the last live quote sat when the tick changed, by class: side of the market, the median
    midpoint on each side, the share within 0.01 of the 0.96/0.04 thresholds, and the share of events
    whose complement token in the same market changed tick within 60 seconds."""
    out: dict = {}
    pair = (ev.select("condition_id", "ts")
              .join(ev.select("condition_id", ts2=pl.col("ts")), on="condition_id")
              .filter((pl.col("ts2") - pl.col("ts")).abs() <= pl.duration(seconds=60))
              .group_by("condition_id", "ts").agg((pl.len() > 1).alias("paired")))
    e = ev.join(pair, on=["condition_id", "ts"], how="left").with_columns(pl.col("paired").fill_null(False))
    for k, g in e.group_by("klass"):
        m = g["mid_before"].drop_nulls()
        hi, lo = m.filter(m > 0.5), m.filter(m <= 0.5)
        out[k[0]] = {"n": g.height, "high_side": hi.len(), "low_side": lo.len(),
                     "high_mid_median": float(hi.median()) if hi.len() else None,
                     "low_mid_median": float(lo.median()) if lo.len() else None,
                     "high_mid_p05": float(hi.quantile(0.05)) if hi.len() else None,
                     "low_mid_p95": float(lo.quantile(0.95)) if lo.len() else None,
                     "share_within_0.01_of_threshold": float((((hi > 0.96) & (hi <= 0.97)).sum() + ((lo < 0.04) & (lo >= 0.03)).sum()) / m.len()) if m.len() else None,
                     "share_paired_with_complement_within_60s": float(g["paired"].mean())}
    return out


def resolution_proximity(ev: pl.DataFrame, gamma: pl.DataFrame) -> dict:
    """By class: the share of events whose last quote was a one-sided book at an extreme (a lone bid
    at or above 0.96 with no ask, or a lone ask at or below 0.04 with no bid), hours from the event
    to the market's closing time, and, for markets with a game start time, hours after kick-off."""
    one_sided = (((pl.col("ask_before") >= 0.999) & (pl.col("bid_before") >= 0.96))
                 | ((pl.col("bid_before") <= 0.001) & (pl.col("ask_before") <= 0.04)))
    e = (ev.join(gamma.select("condition_id", "closed_time", "game_start_time").unique("condition_id"), on="condition_id", how="left")
           .with_columns(one_sided.fill_null(False).alias("one_sided"),
                         ((pl.col("closed_time") - pl.col("ts")).dt.total_seconds() / 3600).alias("h_close"),
                         ((pl.col("ts") - pl.col("game_start_time")).dt.total_seconds() / 3600).alias("h_start")))
    out: dict = {}
    for k, g in e.group_by("klass"):
        h = g["h_close"].drop_nulls()
        st = g["h_start"].drop_nulls()
        out[k[0]] = {"n": g.height, "one_sided_share": float(g["one_sided"].mean()),
                     "with_closed_time": h.len(),
                     "hours_to_close_median": float(h.median()) if h.len() else None,
                     "hours_to_close_p25": float(h.quantile(0.25)) if h.len() else None,
                     "hours_to_close_p75": float(h.quantile(0.75)) if h.len() else None,
                     "share_closed_within_24h": float((h <= 24).mean()) if h.len() else None,
                     "share_closed_within_72h": float((h <= 72).mean()) if h.len() else None,
                     "share_event_after_close": float((h < 0).mean()) if h.len() else None,
                     "sports_with_game_start": st.len(),
                     "hours_after_game_start_median": float(st.median()) if st.len() else None,
                     "share_after_game_start": float((st > 0).mean()) if st.len() else None}
    return out


def main() -> None:
    ev = build_events()
    degraded = degraded_days()
    all_adoptions = ev.filter((pl.col("old_tick") == 0.01) & (pl.col("new_tick") == 0.001)).select("asset_id", "ts")   # every event, for the risk set
    ev = ev.filter(~pl.col("ts").dt.date().is_in(list(degraded)))
    c = classify(ev)
    c.write_parquet(EXT / "ticks_enriched.parquet")
    main_ev = c.filter((pl.col("old_tick") == 0.01) & (pl.col("new_tick") == 0.001))
    res = {"events": c.height, "events_0.01_to_0.001": main_ev.height,
           "classes": {k: int(v) for k, v in main_ev.group_by("klass").len().iter_rows()},
           "class_shares": {k: v / main_ev.height for k, v in main_ev.group_by("klass").len().iter_rows()},
           "by_hour_utc": {str(h): int(n) for h, n in main_ev.group_by("hour").len().sort("hour").iter_rows()},
           "share_with_prior_quote_same_day": float(main_ev["bid_before"].is_not_null().mean())}
    L = main_ev.filter(pl.col("klass") == "L")
    m = L["mins_to_first_live_quote"]
    res["price_at_event"] = price_at_event(main_ev)
    res["event_structure"] = event_structure(main_ev)
    gp = EXT / "gamma" / "gamma_markets.parquet"
    if gp.exists():
        res["resolution_proximity"] = resolution_proximity(main_ev, pl.read_parquet(gp, columns=["condition_id", "closed_time", "game_start_time"]))
    res["activation"] = {"n": L.height,
                         "share_first_live_quote_within_60min_after": float(((m >= 0) & (m <= 60)).mean()) if L.height else None,
                         "share_first_live_quote_before_event": float((m < 0).mean()) if L.height else None,
                         "mins_to_first_live_quote_p50": float(m.median()) if L.height else None}
    panel = pl.read_parquet(EXT / "panel_v1v2.parquet")
    gp = EXT / "gamma" / "gamma_markets.parquet"
    if gp.exists():
        panel = panel.join(pl.read_parquet(gp, columns=["condition_id", "neg_risk"]), on="condition_id", how="left")
    hs = hazard_sample(panel.filter(pl.col("era") == "v2"), all_adoptions, degraded=degraded)
    fit = hazard_fit(hs, COVARIATES, weight_nonadopters=1 / 0.05)
    res["hazard"] = fit
    b = fit["coef"]
    def pred(x):
        z = b["const"] + sum(b[k] * v for k, v in x.items())
        return 1 / (1 + np.exp(-z))
    base = {k: 0.0 for k in COVARIATES}
    res["hazard"]["predicted"] = {
        "placeholder_day": float(pred(base | {"placeholder": 1.0, "log_updates": np.log(2)})),
        "first_active_day": float(pred(base | {"log_updates": np.log(1000), "traded": 1.0, "age_days": 1.0})),
    }
    res["hazard"]["ratio_first_active_over_placeholder"] = res["hazard"]["predicted"]["first_active_day"] / max(1e-12, res["hazard"]["predicted"]["placeholder_day"])
    write_json("c0_rollout", res, inputs=[str(EXT / "tickctx"), str(EXT / "panel_v1v2.parquet")])
    print(res["class_shares"], res["activation"], "AUC", round(fit["auc"], 3), "ratio", round(res["hazard"]["ratio_first_active_over_placeholder"], 1))


if __name__ == "__main__":
    main()

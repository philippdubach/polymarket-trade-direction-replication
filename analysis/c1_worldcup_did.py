"""The 2 July 0.0025 change: the one dated, rule-based tick change, as a two-arm DiD.

Treated: the 2 July 00:31 set. Arm D (a reduction) moved 0.01 -> 0.0025,
arm U (an increase) 0.001 -> 0.0025. Controls: other sports moneyline,
spreads and totals markets, quoted on 1 July, with a game start after noon
on 2 July, at the arm's prior tick. Outcomes per asset half-day (three bins
before, four after): log median spread, spread in old ticks, the share of
quotes at the minimum tick, depth at best.

Estimators: two-way fixed effects clustered by market, the paired-median
version, and an exploratory market-label permutation over 2,000 draws.
Event-clustered standard errors and a whole-event bootstrap account for contracts sharing a game. The minimum detectable effect is reported before the
estimate. Both arms are descriptive because assignment is not random and treated markets share few events.

Run:  uv run python c1_worldcup_did.py
Out:  c1_worldcup.json, /Volumes/data/paper-b-ext/wc_halfday.parquet
"""
from __future__ import annotations

import datetime as dt
import argparse

import numpy as np
import polars as pl

from d0_common import category_from_fee_type, EXT, connect, day_files, read_json, write_json

CHANGE = dt.datetime(2026, 7, 2, 0, 31)
WINDOW_DAYS = ["2026-06-30", "2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"]
PRE_START, POST_END = dt.datetime(2026, 6, 30, 12, 0), dt.datetime(2026, 7, 4, 0, 0)
ARMS = {"D": (0.01, 0.0025), "U": (0.001, 0.0025)}
LISTING_TICK = 0.01   # the tick every market is listed with
SPORTS_TYPES = {"moneyline", "spreads", "totals"}
RI_DRAWS = 2000


# ----------------------------------------------------------------------------- estimators

def did_twfe(df: pl.DataFrame, y: str = "y", cluster_col: str = "market_id") -> dict:
    """Two-way fixed effects (market and bin) by within transformation, then OLS of the demeaned
    outcome on the demeaned treated-times-post indicator with market-clustered standard errors.
    Dummies for thousands of control markets made the design matrix singular for the solver."""
    import numpy as np
    import statsmodels.api as sm
    d = (df.select("market_id", "bin", "treated", "post", pl.col(cluster_col).alias("cluster_id"), pl.col(y).cast(pl.Float64).alias("y"))
           .filter(pl.col("y").is_finite()).drop_nulls()
           .with_columns((pl.col("treated").cast(pl.Float64) * pl.col("post").cast(pl.Float64)).alias("tp")))
    for _ in range(50):   # alternating projections converge fast on a near-balanced panel
        d = d.with_columns([(pl.col(c) - pl.col(c).mean().over("market_id")).alias(c) for c in ("y", "tp")])
        d = d.with_columns([(pl.col(c) - pl.col(c).mean().over("bin")).alias(c) for c in ("y", "tp")])
        if float(d.select(pl.col("y").mean().over("market_id").abs().max()).item()) < 1e-10:
            break
    X = d["tp"].to_numpy().reshape(-1, 1)
    res = sm.OLS(d["y"].to_numpy(), X).fit(cov_type="cluster", cov_kwds={"groups": d["cluster_id"].to_pandas().astype("category").cat.codes.to_numpy()})
    return {"beta": float(res.params[0]), "se_cluster": float(res.bse[0]), "p_cluster": float(res.pvalues[0]),
            "clusters": int(d["cluster_id"].n_unique()), "cluster_unit": cluster_col, "n": int(d.height)}


def did_paired(df: pl.DataFrame, y: str = "y", draws: int = 1000, seed: int = 7) -> dict:
    """Median over markets of the post-period median less the pre-period median, for treated and
    control markets, and their difference, with a percentile bootstrap over markets."""
    per = (df.group_by("market_id").agg(pl.col("treated").first(),
                                       pl.col(y).filter(~pl.col("post")).median().alias("pre"),
                                       pl.col(y).filter(pl.col("post")).median().alias("post_"))
           .drop_nulls().with_columns((pl.col("post_") - pl.col("pre")).alias("chg")).sort("market_id"))
    t, c = per.filter(pl.col("treated"))["chg"].to_numpy(), per.filter(~pl.col("treated"))["chg"].to_numpy()
    tc = float(np.median(t)) if t.size else None
    cc = float(np.median(c)) if c.size else None
    out = {"treated_markets": int(t.size), "control_markets": int(c.size), "treated_change": tc, "control_change": cc,
           "did": (tc - cc) if tc is not None and cc is not None else None}
    if draws and t.size > 1 and c.size > 1:
        rng = np.random.default_rng(seed)
        bs = np.array([np.median(t[rng.integers(0, t.size, t.size)]) - np.median(c[rng.integers(0, c.size, c.size)]) for _ in range(draws)])
        out["ci95"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        out["bootstrap_draws"] = draws
    return out


def randomisation_p(df: pl.DataFrame, y: str = "y", draws: int = RI_DRAWS, seed: int = 7) -> dict:
    """Exploratory label permutation; the venue did not randomise World Cup treatment."""
    rng = np.random.default_rng(seed)
    per = (df.group_by("market_id").agg(pl.col("treated").first(),
                                       pl.col(y).filter(~pl.col("post")).median().alias("pre"),
                                       pl.col(y).filter(pl.col("post")).median().alias("post_"))
           .drop_nulls().with_columns((pl.col("post_") - pl.col("pre")).alias("chg")).sort("market_id"))
    chg = per["chg"].to_numpy()
    tr = per["treated"].to_numpy()
    obs = np.median(chg[tr]) - np.median(chg[~tr])
    k = int(tr.sum())
    null = np.empty(draws)
    for i in range(draws):
        pick = np.zeros(len(chg), dtype=bool)
        pick[rng.choice(len(chg), k, replace=False)] = True
        null[i] = np.median(chg[pick]) - np.median(chg[~pick])
    exceedances = int((np.abs(null) >= abs(obs)).sum())
    return {"observed": float(obs), "draws": draws, "exceedances": exceedances,
            "p": (exceedances + 1) / (draws + 1), "p_method": "plus-one label permutation; descriptive",
            "null_p025": float(np.quantile(null, 0.025)), "null_p975": float(np.quantile(null, 0.975)),
            "degenerate": bool(np.std(null) < 1e-12),
            "null_p2_5": float(np.percentile(null, 2.5)), "null_p97_5": float(np.percentile(null, 97.5))}


def event_bootstrap(df: pl.DataFrame, y: str, draws: int = 1000, seed: int = 7) -> dict:
    """Resample whole venue events within each arm, retaining every contract of a sampled event."""
    per = (df.group_by("market_id").agg(pl.col("event_id").first(), pl.col("treated").first(),
           pl.col(y).filter(~pl.col("post")).median().alias("pre"),
           pl.col(y).filter(pl.col("post")).median().alias("after"))
           .drop_nulls().with_columns((pl.col("after") - pl.col("pre")).alias("chg")))
    groups = []
    for treated in (True, False):
        groups.append([g["chg"].to_numpy() for _, g in
                       per.filter(pl.col("treated") == treated).sort("market_id").group_by("event_id", maintain_order=True)])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(draws):
        medians = [np.median(np.concatenate([gs[i] for i in rng.integers(0, len(gs), len(gs))])) for gs in groups]
        bs.append(medians[0] - medians[1])
    return {"treated_events": len(groups[0]), "control_events": len(groups[1]), "draws": draws,
            "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}


def control_change_sd(df: pl.DataFrame, y: str = "y") -> float:
    """Standard deviation, over control markets, of the market's mean outcome after the change less
    before: the noise a market-level difference-in-differences has to beat."""
    per = (df.filter(~pl.col("treated")).group_by("market_id")
             .agg(pl.col(y).filter(~pl.col("post")).mean().alias("pre"), pl.col(y).filter(pl.col("post")).mean().alias("post_"))
             .drop_nulls().with_columns((pl.col("post_") - pl.col("pre")).alias("chg")).sort("market_id"))
    return float(per["chg"].std()) if per.height > 1 else 0.0


def mde(sigma: float, n_t: int, n_c: int) -> float:
    """Minimum detectable effect at 80% power and 5% size when `sigma` is the standard deviation of a
    market's own pre-to-post change; the change is already a mean over bins, so there is no bin term."""
    return 2.8 * sigma * (1 / n_t + 1 / n_c) ** 0.5


def on_listing_grid(hd: pl.DataFrame, tick: float = LISTING_TICK, floor: float = 0.0095) -> list[str]:
    """Markets whose pre-period spreads never fall below the listing tick: a pre-period median spread
    below 0.0095 is impossible on a 0.01 grid and marks a market that was on a finer tick."""
    per = hd.filter(~pl.col("post")).group_by("market_id").agg(pl.col("spread_median").min().alias("m"))
    return sorted(per.filter(pl.col("m") >= floor)["market_id"].to_list())


# ----------------------------------------------------------------------------- data

def treated_sets() -> dict[str, pl.DataFrame]:
    tk = pl.read_parquet(EXT / "ticks_enriched.parquet", columns=["asset_id", "condition_id", "ts", "old_tick", "new_tick"])
    win = tk.filter((pl.col("ts") >= CHANGE - dt.timedelta(minutes=5)) & (pl.col("ts") <= CHANGE + dt.timedelta(minutes=5)))
    return {arm: win.filter((pl.col("old_tick") == o) & (pl.col("new_tick") == n)).select("asset_id", "condition_id").unique("asset_id")
            for arm, (o, n) in ARMS.items()}


def prior_tick(assets: pl.DataFrame, events: pl.DataFrame | None = None, gamma: pl.DataFrame | None = None,
               change: dt.datetime = CHANGE) -> pl.DataFrame:
    """Tick size of each asset just before the change: the last tick event before it, else the 0.01
    tick every market is listed with. The metadata snapshot is not used: taken months later, it
    carries the post-resolution tick."""
    tk = events if events is not None else pl.read_parquet(EXT / "ticks_enriched.parquet", columns=["asset_id", "ts", "new_tick"])
    last = tk.filter(pl.col("ts") < change).sort("ts").group_by("asset_id").agg(pl.col("new_tick").last().alias("tick_event"))
    return assets.join(last, on="asset_id", how="left").with_columns(pl.col("tick_event").fill_null(LISTING_TICK).alias("prior_tick"))


CONTROL_COLS = ["condition_id", "token_0", "token_1", "fee_type", "sports_market_type", "game_start_time", "event_tags", "event_slug", "event_title", "question"]


def select_controls(g: pl.DataFrame, start: dt.datetime = CHANGE + dt.timedelta(hours=12), end: dt.datetime = POST_END) -> pl.DataFrame:
    """Sports markets (by the venue's fee category; Gamma's event tags are empty) of the game types the
    treated markets have (moneyline, spreads, totals; the metadata field only, no question regex), not
    World Cup, whose game starts inside the post-period like every treated market's. One row per asset."""
    tags = pl.col("event_tags").cast(pl.List(pl.Utf8)).list.join(" ").fill_null("").str.to_lowercase()
    is_wc = tags.str.contains("world cup") | pl.col("event_slug").fill_null("").str.contains("world-cup") \
        | pl.col("event_title").fill_null("").str.to_lowercase().str.contains("world cup")
    typ = pl.col("sports_market_type").fill_null("").str.to_lowercase()
    is_type = typ.is_in(list(SPORTS_TYPES))
    is_sport = pl.col("fee_type").fill_null("").map_elements(category_from_fee_type, return_dtype=pl.Utf8) == "sports"
    c = g.filter(is_sport & is_type & ~is_wc & (pl.col("game_start_time") >= start) & (pl.col("game_start_time") < end))
    return pl.concat([c.select(pl.col("token_0").alias("asset_id"), "condition_id"), c.select(pl.col("token_1").alias("asset_id"), "condition_id")]).drop_nulls("asset_id").unique("asset_id")


def exclude_treated(ctrl: pl.DataFrame, tsets: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Drop every market that either arm treats. The metadata's World Cup markers (tags, slug, title)
    miss most of the treated markets, so the treated condition ids are removed explicitly."""
    treated = pl.concat([t.select("condition_id") for t in tsets.values()])["condition_id"].unique()
    return ctrl.filter(~pl.col("condition_id").is_in(treated.implode()))


def control_candidates() -> pl.DataFrame:
    return select_controls(pl.read_parquet(EXT / "gamma" / "gamma_markets.parquet", columns=CONTROL_COLS))


def quoted_on(assets: pl.DataFrame, day: dt.date) -> pl.DataFrame:
    p = pl.read_parquet(EXT / "panel_v1v2.parquet", columns=["asset_id", "day", "placeholder"]).filter((pl.col("day") == day) & ~pl.col("placeholder"))
    return assets.join(p.select("asset_id"), on="asset_id", how="inner")


def halfday_extract(assets: pl.DataFrame, con=None) -> pl.DataFrame:
    """Per (asset, 12-hour bin): median spread, share of quotes at the asset's tick, depth at best, updates."""
    con = con or connect()
    con.register("wc_assets", assets.select("asset_id", "condition_id", "prior_tick").to_arrow())
    files = "[" + ", ".join(f"'{f}'" for d in WINDOW_DAYS for f in day_files(d, "v2")) + "]"
    q = f"""
    SELECT a.asset_id, a.condition_id, a.prior_tick,
           time_bucket(INTERVAL '12 hours', timestamp_received::TIMESTAMP) AS bin,
           count(*) AS updates,
           quantile_cont(CAST(best_ask AS DOUBLE) - CAST(best_bid AS DOUBLE), 0.5) AS spread_median,
           avg(CASE WHEN CAST(best_ask AS DOUBLE) - CAST(best_bid AS DOUBLE) <= a.prior_tick * 1.05 THEN 1.0 ELSE 0.0 END) AS at_tick_share,
           sum(CASE WHEN side = 'BUY' AND CAST(price AS DOUBLE) = CAST(best_bid AS DOUBLE) THEN CAST(size AS DOUBLE) END) AS depth_bid_sum,
           count(*) FILTER (WHERE side = 'BUY' AND CAST(price AS DOUBLE) = CAST(best_bid AS DOUBLE)) AS depth_bid_n
    FROM read_parquet({files}) r JOIN wc_assets a ON r.asset_id = a.asset_id
    WHERE event_type = 'price_change' AND best_bid IS NOT NULL AND best_ask IS NOT NULL
      AND CAST(best_ask AS DOUBLE) >= CAST(best_bid AS DOUBLE)
    GROUP BY 1, 2, 3, 4
    """
    df = pl.from_arrow(con.execute(q).to_arrow_table())
    assert isinstance(df, pl.DataFrame)
    con.unregister("wc_assets")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true", help="Reanalyse wc_halfday.parquet without scanning the raw archive or rewriting derived data")
    args = ap.parse_args()
    if args.cached:
        res = read_json("c1_worldcup")
        res.pop("_meta", None)
        summarize_halfdays(pl.read_parquet(EXT / "wc_halfday.parquet"), res)
        return
    con = connect()
    tsets = treated_sets()
    cand = quoted_on(control_candidates(), dt.date(2026, 7, 1))
    ctrl = prior_tick(exclude_treated(cand, tsets))
    res = {"change": str(CHANGE), "arms": {},
           "treated_markets_removed_from_controls": int(cand["condition_id"].n_unique() - ctrl["condition_id"].n_unique())}
    frames = []
    for arm, (old, new) in ARMS.items():
        t = tsets[arm].with_columns(pl.lit(old).alias("prior_tick"), pl.lit(True).alias("treated"), pl.lit(arm).alias("arm"))
        c = ctrl.filter(pl.col("prior_tick") == old).select("asset_id", "condition_id", "prior_tick").with_columns(pl.lit(False).alias("treated"), pl.lit(arm).alias("arm"))
        frames.append(pl.concat([t, c.select(t.columns)]))
        res["arms"][arm] = {"old_tick": old, "new_tick": new, "treated_assets": t.height, "treated_markets": t["condition_id"].n_unique(),
                            "control_assets": c.height, "control_markets": c["condition_id"].n_unique()}
    sample = pl.concat(frames)
    want = sample.unique("asset_id").select("asset_id", "condition_id", "prior_tick")
    # cache the extract itself, one row per asset-bin, before it is joined to the sample's arms
    hp = EXT / "wc_halfday_raw.parquet"
    cached = pl.read_parquet(hp) if hp.exists() else None
    if cached is not None and want["asset_id"].is_in(cached["asset_id"].implode()).all():
        hd = cached
    else:
        hd = halfday_extract(want, con)
        hd.write_parquet(hp)
    hd = hd.unique(["asset_id", "bin"])
    hd = hd.join(sample.select("asset_id", "treated", "arm"), on="asset_id", how="inner")
    hd = hd.filter((pl.col("bin") >= PRE_START) & (pl.col("bin") < POST_END)).with_columns(
        (pl.col("bin") >= CHANGE.replace(minute=0)).alias("post"),   # the 00:00 bin holds 31 minutes of pre-change quotes
        pl.col("condition_id").alias("market_id"),
        pl.col("spread_median").log().alias("log_spread"),
        (pl.col("spread_median") / pl.col("prior_tick")).alias("spread_old_ticks"),
        (pl.col("depth_bid_sum") / pl.col("depth_bid_n")).log().alias("log_depth_bid"),
    )
    hd.write_parquet(EXT / "wc_halfday.parquet")
    summarize_halfdays(hd, res)


def summarize_halfdays(hd: pl.DataFrame, res: dict) -> None:
    events = pl.read_parquet(EXT / "gamma" / "gamma_markets.parquet", columns=["condition_id", "event_id"])
    hd = hd.join(events.unique("condition_id"), left_on="market_id", right_on="condition_id", how="left")
    if hd["event_id"].null_count():
        raise ValueError("World Cup event clustering requires an event id for every market")
    # A narrow-spread screen, not proof of a two-sided book: one-sided books near resolution
    # can also have a spread below 0.5. There must be at least two asset-bins before the change.
    pre_ok = (hd.filter(~pl.col("post") & pl.col("spread_median").is_finite())
                .group_by("market_id").agg(pl.col("spread_median").median().alias("pre_med"), pl.len().alias("pre_bins"))
                .filter((pl.col("pre_med") < 0.5) & (pl.col("pre_bins") >= 2))["market_id"])
    for arm in ARMS:
        a = res["arms"][arm]
        d_all = hd.filter(pl.col("arm") == arm)
        a["control_markets_before_book_filter"] = int(d_all.filter(~pl.col("treated"))["market_id"].n_unique())
        d = d_all.filter(pl.col("treated") | pl.col("market_id").is_in(pre_ok.to_list()))
        if ARMS[arm][0] == LISTING_TICK:   # a control on the listing tick must have stayed on its grid before the change
            grid = on_listing_grid(d.filter(~pl.col("treated")))
            a["control_markets_off_grid"] = int(d.filter(~pl.col("treated"))["market_id"].n_unique() - len(grid))
            d = d.filter(pl.col("treated") | pl.col("market_id").is_in(grid))
        # capture on each day of the window: the share of the arm's markets with at least one bin that day
        a["markets_with_bins_by_day"] = {str(k): int(v) for k, v in d.group_by(pl.col("bin").dt.date().alias("d")).agg(pl.col("market_id").n_unique()).sort("d").iter_rows()}
        a["control_markets"] = int(d.filter(~pl.col("treated"))["market_id"].n_unique())
        a["control_assets"] = int(d.filter(~pl.col("treated"))["asset_id"].n_unique())
        a.pop("treated_markets_two_sided", None)
        a["treated_markets_passing_spread_screen"] = int(d.filter(pl.col("treated") & pl.col("market_id").is_in(pre_ok.to_list()))["market_id"].n_unique())
        a["bins"] = {"pre": int(d.filter(~pl.col("post"))["bin"].n_unique()), "post": int(d.filter(pl.col("post"))["bin"].n_unique())}
        # sigma is the spread of control markets' own pre-to-post change in mean log spread; the change is
        # already a mean over the bins, so the bin terms of the formula are set to one
        sd = control_change_sd(d.filter(pl.col("log_spread").is_finite()), "log_spread")
        a["mde_sigma"] = sd
        a["mde_log_spread"] = mde(sd, max(1, a["treated_markets"]), max(1, a["control_markets"]))
        a["descriptive_only"] = True  # few independent treated events; nonrandom World Cup selection
        a["outcomes"] = {}
        for y in ("log_spread", "spread_old_ticks", "at_tick_share", "log_depth_bid"):
            dd = d.filter(pl.col(y).is_finite())
            if dd.filter(pl.col("treated"))["market_id"].n_unique() < 5 or dd.filter(~pl.col("treated"))["market_id"].n_unique() < 5:
                continue
            a["outcomes"][y] = {"twfe": did_twfe(dd, y), "paired": did_paired(dd, y), "ri": randomisation_p(dd, y),
                                "event_clustered": did_twfe(dd, y, cluster_col="event_id"),
                                "event_bootstrap": event_bootstrap(dd, y)}
    write_json("c1_worldcup", res, inputs=[str(EXT / "wc_halfday.parquet"), str(EXT / "gamma" / "gamma_markets.parquet")])
    for arm, a in res["arms"].items():
        o = a["outcomes"].get("log_spread", {})
        print(f"arm {arm}: T {a['treated_markets']} C {a['control_markets']} MDE {a['mde_log_spread']:.3f}  "
              f"twfe {o.get('twfe', {}).get('beta')}  paired {o.get('paired', {}).get('did')}  RI p {o.get('ri', {}).get('p')}")


if __name__ == "__main__":
    main()

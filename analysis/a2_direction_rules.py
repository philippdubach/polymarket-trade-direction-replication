"""Direction rules scored fill by fill against the on-chain taker side.

For one UTC day, every feed print with a settled counterpart (a1's hash join)
gets its prevailing quote from the feed's own price_change rows and four
inferred signs:

  lr     Lee-Ready: above the midpoint buy, below sell, at the midpoint the tick test
  tick   tick test against the previous print of the same asset, last non-zero sign
         carried through zero ticks
  emo    quote rule at or beyond the quotes (at/above the ask buy, at/below the bid
         sell), tick test inside the spread
  bvc    bulk volume classification per 5-second bar: buy share Phi(dp / sigma)

Metrics against the taker side: accuracy, majority baseline, recall by side,
balanced accuracy, Cohen's kappa and MCC, with an asset-cluster bootstrap and a
within-day permutation null. Segments: price level, size quintile, tick regime
at the print, contract, hour, match type, quote staleness, category.

The five-second-cell comparison is reproduced twice, against all legs pooled
and against taker legs only, so the paper can show how the "89%" arose.

Run:  uv run python a2_direction_rules.py 2026-05-13
Out:  a2_direction_<day>.json
"""
from __future__ import annotations

import datetime as dt
import glob
import math
import sys

import numpy as np
import polars as pl

import a0_onchain_load as ld
from a1_fill_match import join_by_hash
from d0_common import EXT, PILOT, connect, day_files, gamma_categories, write_json

RULES = ("lr", "tick", "emo")
BUCKET = "5s"
B = 500


# ----------------------------------------------------------------------------- signing

def sign_rules(df: pl.DataFrame) -> pl.DataFrame:
    """Add mid, tick, lr and emo signs. Input sorted by (asset_id, ts) or not; output sorted."""
    d = df.sort(["asset_id", "ts"])
    prev = pl.col("price").shift(1).over("asset_id")
    raw_tick = pl.when(pl.col("price") > prev).then(1).when(pl.col("price") < prev).then(-1).otherwise(None)
    d = d.with_columns(raw_tick.alias("_t")).with_columns(pl.col("_t").forward_fill().over("asset_id").alias("tick")).drop("_t")
    mid = (pl.col("bid") + pl.col("ask")) / 2.0
    has_q = pl.col("bid").is_not_null() & pl.col("ask").is_not_null() & (pl.col("ask") >= pl.col("bid"))
    lr = (pl.when(has_q & (pl.col("price") > mid)).then(1)
          .when(has_q & (pl.col("price") < mid)).then(-1)
          .otherwise(pl.col("tick")))
    emo = (pl.when(has_q & (pl.col("price") >= pl.col("ask"))).then(1)
           .when(has_q & (pl.col("price") <= pl.col("bid"))).then(-1)
           .otherwise(pl.col("tick")))
    return d.with_columns(mid.alias("mid"), lr.cast(pl.Int64).alias("lr"), emo.cast(pl.Int64).alias("emo"),
                          pl.col("tick").cast(pl.Int64))


def bvc(df: pl.DataFrame, bar: str = BUCKET, sigma_window: int = 20) -> pl.DataFrame:
    """Bulk volume classification: per (asset, bar) the buy share Phi(dp / sigma)."""
    bars = (df.sort(["asset_id", "ts"]).with_columns(pl.col("ts").dt.truncate(bar).alias("bar"))
            .group_by(["asset_id", "bar"]).agg(pl.col("price").last().alias("p_last"), pl.col("size").sum().alias("volume"))
            .sort(["asset_id", "bar"]))
    dp = (pl.col("p_last") - pl.col("p_last").shift(1)).over("asset_id")
    bars = bars.with_columns(dp.alias("dp"))
    sig = pl.col("dp").rolling_std(window_size=sigma_window, min_samples=3).over("asset_id")
    bars = bars.with_columns(sig.alias("sigma"))
    glob_sig = float(bars["dp"].drop_nulls().std() or 0.0) if bars["dp"].drop_nulls().len() > 1 else 0.0
    bars = bars.with_columns(pl.col("sigma").fill_null(glob_sig).alias("sigma"))
    z = pl.when(pl.col("sigma") > 0).then(pl.col("dp") / pl.col("sigma")).otherwise(pl.col("dp").sign() * 10.0)
    phi = 0.5 * (1.0 + (z / math.sqrt(2.0)).map_batches(lambda s: pl.Series(np.vectorize(math.erf)(s.fill_null(0.0).to_numpy()))))
    return bars.with_columns(pl.when(pl.col("dp").is_null()).then(0.5).otherwise(phi).clip(0.0, 1.0).alias("buy_share_bvc"))


# ----------------------------------------------------------------------------- metrics

def metrics(truth: pl.Series, pred: pl.Series) -> dict:
    """truth: 0 = buy, 1 = sell (on-chain taker side). pred: +1 buy, -1 sell, null unsigned."""
    n_all = truth.len()
    mask = pred.is_not_null()
    t = (truth.filter(mask) == 0).to_numpy()   # True = buy
    p = (pred.filter(mask) > 0).to_numpy()
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "coverage": 0.0}
    tp = float((t & p).sum()); tn = float((~t & ~p).sum()); fp = float((~t & p).sum()); fn = float((t & ~p).sum())
    rb = tp / (tp + fn) if tp + fn else float("nan")
    rs = tn / (tn + fp) if tn + fp else float("nan")
    po = (tp + tn) / n
    pe = (t.mean() * p.mean()) + ((1 - t.mean()) * (1 - p.mean()))
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n": n, "coverage": n / n_all, "accuracy": po, "majority_baseline": float(max(t.mean(), 1 - t.mean())),
        "buy_share_truth": float(t.mean()), "buy_share_pred": float(p.mean()),
        "recall_buy": rb, "recall_sell": rs, "balanced_accuracy": (rb + rs) / 2,
        "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan"),
        "mcc": (tp * tn - fp * fn) / den if den else float("nan"),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def permutation_null(truth: pl.Series, pred: pl.Series, draws: int = 200, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    arr = pred.to_numpy()
    vals = []
    for _ in range(draws):
        vals.append(metrics(truth, pl.Series(rng.permutation(arr)))["balanced_accuracy"])
    return {"draws": draws, "balanced_accuracy_mean": float(np.mean(vals)), "balanced_accuracy_p95": float(np.percentile(vals, 95))}


def boot_by_asset(df: pl.DataFrame, col: str, draws: int = B, seed: int = 7) -> dict:
    """Asset-cluster percentile bootstrap of balanced accuracy for one rule."""
    rng = np.random.default_rng(seed)
    d = df.filter(pl.col(col).is_not_null()).with_columns(
        buy=(pl.col("taker_side") == 0), pbuy=(pl.col(col) > 0)
    ).group_by("asset_id").agg(
        tp=(pl.col("buy") & pl.col("pbuy")).sum(), fn=(pl.col("buy") & ~pl.col("pbuy")).sum(),
        tn=(~pl.col("buy") & ~pl.col("pbuy")).sum(), fp=(~pl.col("buy") & pl.col("pbuy")).sum(),
    )
    tp, fn, tn, fp = (d[c].to_numpy().astype(float) for c in ("tp", "fn", "tn", "fp"))
    k = len(tp)
    out = np.empty(draws)
    for b in range(draws):
        pick = rng.integers(0, k, k)
        rb = tp[pick].sum() / max(1.0, (tp[pick] + fn[pick]).sum())
        rs = tn[pick].sum() / max(1.0, (tn[pick] + fp[pick]).sum())
        out[b] = (rb + rs) / 2
    return {"clusters": k, "lo": float(np.percentile(out, 2.5)), "hi": float(np.percentile(out, 97.5))}


def segments(df: pl.DataFrame, cols: list[str], rules: tuple[str, ...] = RULES) -> dict:
    out = {}
    for c in cols:
        for val, sub in df.group_by(c):
            key = (c, str(val[0]))
            out[key] = {r: metrics(sub["taker_side"], sub[r]) for r in rules}
            out[key]["n_rows"] = sub.height
    return out


# ----------------------------------------------------------------------------- day assembly

PRINTS_DIR = EXT / "prints"
HORIZON = dt.timedelta(minutes=5)
# Lags (ms) at which the prevailing quote is also attached: the quote strictly before ts - lag.
# A quote stamped at the print's own millisecond is the post-trade book from the same batch.
LAG_GRID_MS = (1, 10, 100, 500, 1000)


def _hour_prints(con, f: str) -> pl.DataFrame:
    df = pl.from_arrow(con.execute(f"""
        SELECT asset_id, lower(transaction_hash) AS transaction_hash, side, CAST(price AS DOUBLE) AS price,
               CAST(size AS DOUBLE) AS size, timestamp_received::TIMESTAMP AS ts
        FROM read_parquet('{f}') WHERE event_type = 'last_trade_price'""").to_arrow_table())
    assert isinstance(df, pl.DataFrame)
    return df


def _hour_quotes(con, f: str, assets: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Quotes of the hour for the given assets, and the hour's last quote for every asset (the carry)."""
    con.register("pa_tbl", assets.to_arrow())
    q = pl.from_arrow(con.execute(f"""
        SELECT asset_id, timestamp_received::TIMESTAMP AS qts, CAST(best_bid AS DOUBLE) AS bid, CAST(best_ask AS DOUBLE) AS ask
        FROM read_parquet('{f}')
        WHERE event_type = 'price_change' AND best_bid IS NOT NULL AND best_ask IS NOT NULL
          AND asset_id IN (SELECT asset_id FROM pa_tbl)""").to_arrow_table())
    con.unregister("pa_tbl")
    last = pl.from_arrow(con.execute(f"""
        SELECT asset_id, max(timestamp_received::TIMESTAMP) AS qts,
               arg_max(CAST(best_bid AS DOUBLE), timestamp_received) AS bid, arg_max(CAST(best_ask AS DOUBLE), timestamp_received) AS ask
        FROM read_parquet('{f}')
        WHERE event_type = 'price_change' AND best_bid IS NOT NULL AND best_ask IS NOT NULL
        GROUP BY 1""").to_arrow_table())
    assert isinstance(q, pl.DataFrame) and isinstance(last, pl.DataFrame)
    return q, last


def prints_with_quotes(day: str, files: list[str] | None = None, con=None) -> pl.DataFrame:
    """Every print with the prevailing quote (last price_change before it) and the midpoint 5 minutes later.

    Processed hour by hour: the quotes of the hour for the assets with prints, joined as-of in
    memory, with the last quote of every asset carried into the next hour. A print whose horizon
    crosses the hour boundary waits one hour for its forward midpoint. A day-wide ASOF join sorted
    every quote of every print asset and did not finish in an hour.
    """
    cache = PRINTS_DIR / f"day={day}.parquet"
    if files is None and cache.exists():
        return pl.read_parquet(cache)
    files = files if files is not None else day_files(day, "v2")
    con = con or connect()
    carry = pl.DataFrame(schema={"asset_id": pl.Utf8, "qts": pl.Datetime("us"), "bid": pl.Float64, "ask": pl.Float64})
    pending = None
    done = []
    for i, f in enumerate(files):
        ph = _hour_prints(con, f).with_columns(pl.col("ts").cast(pl.Datetime("us")))
        need = pl.concat([ph.select("asset_id"), pending.select("asset_id") if pending is not None else ph.select("asset_id").clear()]).unique()
        q, last = _hour_quotes(con, f, need)
        q = q.with_columns(pl.col("qts").cast(pl.Datetime("us")))
        last = last.with_columns(pl.col("qts").cast(pl.Datetime("us")))
        q_all = pl.concat([carry.join(need, on="asset_id", how="semi"), q]).sort(["asset_id", "qts"])
        # prevailing quote: the last quote strictly before the print's receipt millisecond
        qq = q_all.rename({"qts": "quote_ts"})
        ph = ph.sort(["asset_id", "ts"]).join_asof(qq, left_on="ts", right_on="quote_ts", by="asset_id", strategy="backward", allow_exact_matches=False)
        # was there a quote stamped at exactly the print's millisecond?
        same = q_all.select("asset_id", pl.col("qts").alias("ts")).unique().with_columns(pl.lit(True).alias("same_ms_quote"))
        ph = ph.join(same, on=["asset_id", "ts"], how="left").with_columns(pl.col("same_ms_quote").fill_null(False))
        # the lag grid: the quote strictly before ts - lag
        for lag in LAG_GRID_MS:
            lagged = qq.select("asset_id", "quote_ts", pl.col("bid").alias(f"bid_lag{lag}"), pl.col("ask").alias(f"ask_lag{lag}"))
            ph = (ph.with_columns((pl.col("ts") - dt.timedelta(milliseconds=lag)).alias("_tl")).sort(["asset_id", "_tl"])
                  .join_asof(lagged, left_on="_tl", right_on="quote_ts", by="asset_id", strategy="backward", allow_exact_matches=False, suffix="_l")
                  .drop("_tl", "quote_ts_l"))
        ph = ph.sort(["asset_id", "ts"])
        # forward midpoint: this hour's prints plus those pending from the previous hour
        cand = pl.concat([pending, ph.with_columns((pl.col("ts") + HORIZON).alias("t_fwd"))]) if pending is not None else ph.with_columns((pl.col("ts") + HORIZON).alias("t_fwd"))
        hour_end = ph["ts"].dt.truncate("1h").max() + dt.timedelta(hours=1) if ph.height else None
        is_last = i == len(files) - 1
        ready = cand if (is_last or hour_end is None) else cand.filter(pl.col("t_fwd") < hour_end)
        pending = None if (is_last or hour_end is None) else cand.filter(pl.col("t_fwd") >= hour_end)
        fw = ready.sort(["asset_id", "t_fwd"]).join_asof(q_all.select("asset_id", pl.col("qts").alias("f_ts"), ((pl.col("bid") + pl.col("ask")) / 2).alias("mid_fwd")),
                                                          left_on="t_fwd", right_on="f_ts", by="asset_id", strategy="backward").drop("f_ts")
        done.append(fw)
        # carry: the last quote per asset up to the end of this hour
        carry = pl.concat([carry, last]).sort(["asset_id", "qts"]).unique("asset_id", keep="last")
    out = pl.concat(done).drop("t_fwd").sort(["asset_id", "ts"])
    if files == day_files(day, "v2"):
        PRINTS_DIR.mkdir(parents=True, exist_ok=True)
        out.write_parquet(cache)
    return out


def tick_at_print(df: pl.DataFrame, ticks: pl.DataFrame | None = None, defaults: pl.DataFrame | None = None) -> pl.Series:
    """The asset's tick size at each print.

    The last tick_size_change before the print wins; a print before the asset's first
    change takes that change's old tick; an asset with no change in the sample keeps the
    tick Gamma reports for it (a market created at 0.001 never emits a change; the
    snapshot is post-resolution, so it is used only then), and 0.01 when neither exists.
    """
    if ticks is None:
        from rev_common import utc_naive
        fs = sorted(glob.glob(str(PILOT / "ticks" / "day=*.parquet")))
        tk = pl.concat([pl.read_parquet(f, columns=["asset_id", "timestamp_received", "old_tick", "new_tick"]) for f in fs])
        ticks = tk.with_columns(utc_naive(tk["timestamp_received"]).alias("tts")).select("asset_id", "tts", "old_tick", "new_tick")
    if defaults is None:
        tm, gp = EXT / "token_map.parquet", EXT / "gamma" / "gamma_markets.parquet"
        if tm.exists() and gp.exists():
            defaults = (pl.read_parquet(tm, columns=["asset_id", "condition_id"])
                        .join(pl.read_parquet(gp, columns=["condition_id", "tick_size"]), on="condition_id", how="inner")
                        .select("asset_id", pl.col("tick_size").cast(pl.Float64).alias("tick_default")))
        else:
            defaults = pl.DataFrame(schema={"asset_id": pl.Utf8, "tick_default": pl.Float64})
    ticks = ticks.with_columns(pl.col("tts").cast(pl.Datetime("us")), pl.col("new_tick").cast(pl.Float64),
                               pl.col("old_tick").cast(pl.Float64)).sort(["asset_id", "tts"])
    first_old = ticks.group_by("asset_id").agg(pl.col("old_tick").first().alias("tick_before_first"))
    j = (df.select("asset_id", pl.col("ts").cast(pl.Datetime("us"))).with_row_index("_i").sort(["asset_id", "ts"])
         .join_asof(ticks.select("asset_id", "tts", "new_tick"), left_on="ts", right_on="tts", by="asset_id", strategy="backward")
         .join(first_old, on="asset_id", how="left")
         .join(defaults.unique("asset_id"), on="asset_id", how="left").sort("_i"))
    return j.select(pl.coalesce("new_tick", "tick_before_first", "tick_default", pl.lit(0.01)).alias("tick"))["tick"]


def cell_test(prints: pl.DataFrame, fills: pl.DataFrame, taker: pl.DataFrame) -> dict:
    """Asset x five-second-cell agreement of the flag, against pooled legs and taker legs."""
    feed = prints.with_columns(pl.col("ts").dt.truncate(BUCKET).cast(pl.Datetime("ms")).alias("bucket"),
                               pl.when(pl.col("side") == "BUY").then(1).otherwise(-1).alias("s")).group_by(["asset_id", "bucket"]).agg(
        pl.col("s").mean().sign().alias("f"))
    out = {}
    for name, ch in (("all_legs", fills), ("taker_legs", taker)):
        cells = ch.group_by([pl.col("token_id").alias("asset_id"), "bucket"]).agg(((pl.col("side") == 0).mean() - 0.5).sign().alias("c"))
        j = cells.join(feed, on=["asset_id", "bucket"], how="inner").filter((pl.col("c") != 0) & (pl.col("f") != 0))
        m = metrics(pl.Series((j["c"] < 0).cast(pl.Int64)), j["f"])
        out[name] = {"cells": j.height, "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                     "majority_baseline": m["majority_baseline"], "buy_share": m["buy_share_truth"]}
    return out


def main() -> None:
    day = sys.argv[1] if len(sys.argv) > 1 else "2026-05-13"
    con = connect()
    pq = prints_with_quotes(day, con=con)
    taker = ld.taker_legs(day)
    fills = ld.fills(day)
    j = join_by_hash(pq, taker).filter(pl.col("matched"))
    d = j.rename({"chain_side": "taker_side"})
    d = sign_rules(d)
    d = d.with_columns(
        tick_at_print(d).alias("tick_size"),
        (pl.col("ts") - pl.col("quote_ts")).dt.total_seconds().alias("staleness_s"),
        pl.col("ts").dt.hour().alias("hour"),
    ).with_columns(
        pl.when(pl.col("staleness_s").is_null()).then(pl.lit("no_quote")).when(pl.col("staleness_s") <= 1).then(pl.lit("<=1s"))
        .when(pl.col("staleness_s") <= 5).then(pl.lit("1-5s")).when(pl.col("staleness_s") <= 60).then(pl.lit("5-60s")).otherwise(pl.lit(">60s")).alias("staleness"),
        pl.when(pl.col("mid").is_null()).then(pl.lit("no_quote")).when((pl.col("mid") < 0.04) | (pl.col("mid") > 0.96)).then(pl.lit("tail"))
        .otherwise(pl.lit("interior")).alias("price_level"),
        (pl.col("size").rank("ordinal") * 5 / pl.len()).ceil().cast(pl.Int64).clip(1, 5).alias("size_quintile"),
    )
    # contract and match type from the chain side
    mt = ld.match_types(day, ld.complement_from_token_map())
    d = d.join(mt.select("tx_hash", "match_type"), left_on="transaction_hash", right_on="tx_hash", how="left")
    d = d.join(taker.select("tx_hash", "contract"), left_on="transaction_hash", right_on="tx_hash", how="left")
    tm = EXT / "token_map.parquet"
    gp = EXT / "gamma" / "gamma_markets.parquet"
    if tm.exists() and gp.exists():
        lab = pl.read_parquet(tm, columns=["asset_id", "condition_id"]).join(gamma_categories(), on="condition_id", how="left").select("asset_id", "category")
        d = d.join(lab, on="asset_id", how="left")
    else:
        d = d.with_columns(pl.lit(None, dtype=pl.Utf8).alias("category"))

    res = {"day": day, "prints_matched": d.height, "assets": d["asset_id"].n_unique(),
           "flag_vs_taker": metrics(d["taker_side"], d.select(pl.when(pl.col("side") == "BUY").then(1).otherwise(-1).alias("f"))["f"]),
           "rules": {}}
    for r in RULES:
        res["rules"][r] = metrics(d["taker_side"], d[r]) | {"bootstrap": boot_by_asset(d, r)}
    res["same_ms_quote_share"] = float(d["same_ms_quote"].mean()) if "same_ms_quote" in d.columns else None
    res["lag_grid"] = {}
    for lag in LAG_GRID_MS:
        if f"bid_lag{lag}" not in d.columns:
            continue
        dl = sign_rules(d.select("asset_id", "ts", "price", "taker_side", pl.col(f"bid_lag{lag}").alias("bid"), pl.col(f"ask_lag{lag}").alias("ask")))
        res["lag_grid"][str(lag)] = {"lr": metrics(dl["taker_side"], dl["lr"]), "emo": metrics(dl["taker_side"], dl["emo"]),
                                     "with_quote_share": float(dl["bid"].is_not_null().mean())}
    bars = bvc(d)
    bj = d.with_columns(pl.col("ts").dt.truncate(BUCKET).alias("bar")).join(bars.select("asset_id", "bar", "buy_share_bvc"), on=["asset_id", "bar"], how="left")
    bvc_sign = bj.select(pl.when(pl.col("buy_share_bvc") > 0.5).then(1).when(pl.col("buy_share_bvc") < 0.5).then(-1).otherwise(None).alias("s"))["s"]
    res["rules"]["bvc_5s"] = metrics(bj["taker_side"], bvc_sign)
    seg = segments(d, ["price_level", "size_quintile", "tick_size", "contract", "hour", "match_type", "staleness", "category"])
    res["segments"] = {f"{c}={v}": m for (c, v), m in seg.items()}
    res["cell_test"] = cell_test(pq, fills, taker)
    write_json(f"a2_direction_{day}", res, inputs=[str(EXT / "onchain" / f"day={day}")] + day_files(day, "v2")[:1])
    print(f"{day}: n={d.height:,}  " + "  ".join(f"{r} bal={res['rules'][r]['balanced_accuracy']:.3f}" for r in RULES)
          + f"  bvc bal={res['rules']['bvc_5s']['balanced_accuracy']:.3f}  cell(all legs) bal={res['cell_test']['all_legs']['balanced_accuracy']:.3f}"
          + f"  cell(taker) bal={res['cell_test']['taker_legs']['balanced_accuracy']:.3f}")


if __name__ == "__main__":
    main()

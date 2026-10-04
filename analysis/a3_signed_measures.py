"""What feed-only signing costs: signed measures under the taker sign and under inferred signs.

For matched prints on one UTC day, with the prevailing midpoint m at the print
and the midpoint m' five minutes later:

  effective spread   2 s (p - m)
  realised spread    2 s (p - m')
  price impact       2 s (m' - m)
  Kyle lambda        per asset-day, slope of the 5-minute mid change on signed flow

each under the on-chain taker sign and under every rule's sign from a2. The
reported quantity is the ratio of cross-sectional medians (rule / taker), and
next to it the analytical attenuation from the two recalls:

  E[s_hat x] = (2 r_b - 1) E[x 1{buy}] - (2 r_s - 1) E[x 1{sell}]

which collapses to the scalar (2p - 1) only when r_b = r_s.

Run:  uv run python a3_signed_measures.py 2026-05-13
Out:  a3_signed_<day>.json
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import polars as pl

import a0_onchain_load as ld
from a1_fill_match import join_by_hash
from a2_direction_rules import RULES, metrics, prints_with_quotes, sign_rules
from d0_common import EXT, connect, day_files, write_json

MIN_FILLS = 50


def per_fill(df: pl.DataFrame, sign_col: str) -> pl.DataFrame:
    s = pl.col(sign_col).cast(pl.Float64)
    return df.with_columns(
        (2 * s * (pl.col("price") - pl.col("mid"))).alias("eff"),
        (2 * s * (pl.col("price") - pl.col("mid_fwd"))).alias("realised"),
        (2 * s * (pl.col("mid_fwd") - pl.col("mid"))).alias("impact"),
    )


def compare(df: pl.DataFrame, rules: list[str], min_fills: int = MIN_FILLS) -> dict:
    """Cross-sectional medians of per-asset medians, under the taker sign and each rule."""
    keep = df.group_by("asset_id").len().filter(pl.col("len") >= min_fills)["asset_id"]
    d = df.filter(pl.col("asset_id").is_in(keep.to_list()))
    out = {"assets": keep.len(), "fills": d.height}
    for col in ["taker"] + rules:
        m = per_fill(d.filter(pl.col(col).is_not_null()), col).group_by("asset_id").agg(
            pl.col("eff").median().alias("eff"), pl.col("realised").median().alias("realised"), pl.col("impact").median().alias("impact"),
            pl.col("eff").mean().alias("eff_mean"),
        )
        out[col] = {"eff_median": float(m["eff"].median()), "realised_median": float(m["realised"].median()),
                    "impact_median": float(m["impact"].median()), "eff_mean_of_means": float(m["eff_mean"].mean()), "assets": m.height}
        if col != "taker":
            for k in ("eff", "realised", "impact"):
                ref = out["taker"][f"{k}_median"]
                out[col][f"{k}_ratio_to_taker"] = (out[col][f"{k}_median"] / ref) if ref else None
            ref_mean = out["taker"]["eff_mean_of_means"]
            out[col]["eff_mean_ratio_to_taker"] = (out[col]["eff_mean_of_means"] / ref_mean) if ref_mean else None
    return out


def wrong_side(df: pl.DataFrame, sign_col: str = "taker") -> dict:
    """Share of fills whose signed effective spread is negative: the price sits on the wrong side
    of the prevailing midpoint for the true aggressor. A quote-based rule cannot see these."""
    e = per_fill(df, sign_col)["eff"]
    return {"fills": e.len(), "share_negative": float((e < 0).mean()), "share_zero": float((e == 0).mean()),
            "mean_eff_negative_part": float(e.filter(e < 0).mean()) if (e < 0).any() else 0.0}


def attenuation(recall_buy: float, recall_sell: float, x_buy: float, x_sell: float) -> float:
    """Expected signed quantity under a rule with the two recalls, from the true buy/sell parts."""
    return (2 * recall_buy - 1) * x_buy - (2 * recall_sell - 1) * x_sell


def accounting(df: pl.DataFrame, sign_col: str) -> dict:
    """The rule-signed mean effective spread as the taker-signed mean plus the mass the rule flips.

    A quote rule's sign is a function of where the price sits relative to the quote, so its
    error set on off-midpoint prints is the wrong-side set; the identity holds fill by fill.
    """
    eligible = df.filter(pl.col(sign_col).is_not_null() & pl.col("taker").is_not_null()
                         & pl.col("price").is_finite() & pl.col("mid").is_finite())
    t = per_fill(eligible, "taker")["eff"]
    r = per_fill(eligible, sign_col)["eff"]
    flipped = (r != t)
    return {"fills": eligible.height, "candidate_fills": df.height, "coverage": eligible.height / df.height if df.height else 0.0,
            "taker_mean_eff": float(t.mean()), "rule_mean_eff": float(r.mean()),
            "flip_mass": float((r - t).mean()), "flip_share": float(flipped.mean()),
            "fill_level_ratio": float(r.mean() / t.mean()) if t.mean() else None,
            "wrong_side_share": float((t < 0).mean()), "wrong_side_mean_abs": float((-t.filter(t < 0)).mean()) if (t < 0).any() else 0.0}


def main() -> None:
    day = sys.argv[1] if len(sys.argv) > 1 else "2026-05-13"
    con = connect()
    pq = prints_with_quotes(day, con=con)
    taker = ld.taker_legs(day)
    d = join_by_hash(pq, taker).filter(pl.col("matched")).rename({"chain_side": "taker_side"})
    d = sign_rules(d).filter(pl.col("mid").is_not_null() & pl.col("mid_fwd").is_not_null())
    d = d.with_columns(pl.when(pl.col("taker_side") == 0).then(1).otherwise(-1).cast(pl.Int64).alias("taker"))
    res = {"day": day, "fills_with_both_mids": d.height, "measures": compare(d, list(RULES)),
           "same_ms_quote_share": float(d["same_ms_quote"].mean()) if "same_ms_quote" in d.columns else None,
           "wrong_side": {"all": wrong_side(d)}, "accounting": {r: accounting(d, r) for r in RULES}}
    # the wrong-side share with the settlement's own price in place of the print price
    dc = d.with_columns(pl.col("chain_price").alias("price"))
    res["wrong_side"]["all_chain_price"] = wrong_side(dc)
    mt = ld.match_types(day, ld.complement_from_token_map()).select("tx_hash", "match_type")
    dm = d.join(mt, left_on="transaction_hash", right_on="tx_hash", how="left")
    for (k,), g in dm.group_by(["match_type"]):
        res["wrong_side"][str(k)] = wrong_side(g)
    # the lag grid: wrong-side share and the Lee-Ready ratio with the quote strictly before ts - lag
    res["lag_grid"] = {}
    from a2_direction_rules import LAG_GRID_MS
    for lag in LAG_GRID_MS:
        if f"bid_lag{lag}" not in d.columns:
            continue
        dl = d.filter(pl.col(f"bid_lag{lag}").is_not_null()).with_columns(
            ((pl.col(f"bid_lag{lag}") + pl.col(f"ask_lag{lag}")) / 2).alias("mid"), pl.col(f"bid_lag{lag}").alias("bid"), pl.col(f"ask_lag{lag}").alias("ask"))
        dl = sign_rules(dl.drop("lr", "emo", "tick"))
        res["lag_grid"][str(lag)] = {"fills": dl.height, "wrong_side_share": float((per_fill(dl, "taker")["eff"] < 0).mean()),
                                     "lr": accounting(dl, "lr")}
    write_json(f"a3_signed_{day}", res, inputs=[str(EXT / "onchain" / f"day={day}")] + day_files(day, "v2")[:1])
    m = res["measures"]; acc = res["accounting"]
    print(f"{day}: assets {m['assets']}  taker mean eff {acc['lr']['taker_mean_eff']:.4f}  wrong-side {res['wrong_side']['all']['share_negative']:.3f}  "
          + "  ".join(f"{r} x{acc[r]['fill_level_ratio']:.2f}" for r in RULES)
          + "  lag grid: " + " ".join(f"{k}ms {v['wrong_side_share']:.3f}" for k, v in res["lag_grid"].items()))


if __name__ == "__main__":
    main()

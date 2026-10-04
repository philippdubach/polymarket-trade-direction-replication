"""Fill-level identity between the feed's trade prints and the on-chain taker legs.

Every V2 `last_trade_price` print carries the transaction hash of the
settlement it reports, and every settled transaction has exactly one
OrderFilled leg whose `taker` is the exchange contract: the taker order. This
script joins the two by hash for one UTC day and reports:

  coverage   prints with a settled counterpart, taker legs with a print,
             both by hour and by contract
  identity   on matched pairs: side, token, price and size agreement, with the
             |dprice| distribution (the print price and the on-chain average
             price differ by rounding on a minority of fills)
  residual   a second-stage match of unmatched prints to unmatched taker legs
             on (asset, side, price, size) within 120 seconds
  types      the transaction match-type composition from a0_onchain_load

Price and size from the amounts follow the side: a BUY taker paid
maker_amount of collateral for taker_amount shares; a SELL taker gave
maker_amount shares for taker_amount of collateral (checked on 75,052 pairs).

Run:  uv run python a1_fill_match.py 2026-05-13
Out:  a1_fill_match_<day>.json
"""
from __future__ import annotations

import datetime as dt
import sys

import polars as pl

import a0_onchain_load as ld
from d0_common import EXT, connect, day_files, write_json

FUZZY_WINDOW_S = 120
USDC = 1e6


def price_size(t: pl.DataFrame) -> tuple[pl.Series, pl.Series]:
    ma = t["maker_amount"].cast(pl.Float64) / USDC
    ta = t["taker_amount"].cast(pl.Float64) / USDC
    buy = t["side"] == 0
    px = pl.when(buy).then(ma / ta).otherwise(ta / ma)
    sz = pl.when(buy).then(ta).otherwise(ma)
    d = t.select(px.alias("px"), sz.alias("sz"))
    return d["px"], d["sz"]


def feed_prints(day: str, con=None) -> pl.DataFrame:
    cache = EXT / "prints" / f"day={day}.parquet"
    if cache.exists():
        return pl.read_parquet(cache, columns=["asset_id", "transaction_hash", "side", "price", "size", "ts"])
    con = con or connect()
    files = "[" + ", ".join(f"'{f}'" for f in day_files(day, "v2")) + "]"
    df = pl.from_arrow(con.execute(f"""
        SELECT asset_id, lower(transaction_hash) AS transaction_hash, side,
               CAST(price AS DOUBLE) AS price, CAST(size AS DOUBLE) AS size,
               timestamp_received::TIMESTAMP AS ts
        FROM read_parquet({files}) WHERE event_type = 'last_trade_price'""").to_arrow_table())
    assert isinstance(df, pl.DataFrame)
    return df


def join_by_hash(prints: pl.DataFrame, taker: pl.DataFrame) -> pl.DataFrame:
    px, sz = price_size(taker)
    t = taker.with_columns(px.alias("chain_price"), sz.alias("chain_size")).select(
        pl.col("tx_hash").alias("transaction_hash"), pl.col("token_id").alias("chain_token"),
        pl.col("side").alias("chain_side"), "chain_price", "chain_size", pl.col("ts").cast(pl.Datetime("us")).alias("chain_ts"),
    ).unique("transaction_hash")
    j = prints.with_columns(pl.col("ts").cast(pl.Datetime("us"))).join(t, on="transaction_hash", how="left")
    return j.with_columns(pl.col("chain_side").is_not_null().alias("matched"))


def identity_stats(j: pl.DataFrame) -> dict:
    m = j.filter(pl.col("matched"))
    if m.height == 0:
        return {"matched": 0}
    dp = (m["price"] - m["chain_price"]).abs()
    ds = (m["size"] - m["chain_size"]).abs()
    side_eq = ((m["side"] == "BUY") == (m["chain_side"] == 0))
    return {
        "matched": m.height,
        "side_equal": int(side_eq.sum()), "side_equal_share": float(side_eq.mean()),
        "token_equal": int((m["asset_id"] == m["chain_token"]).sum()),
        "token_equal_share": float((m["asset_id"] == m["chain_token"]).mean()),
        "price_within_5e4": int((dp <= 5e-4).sum()), "price_within_5e4_share": float((dp <= 5e-4).mean()),
        "price_within_1e2": int((dp <= 1e-2).sum()), "price_within_1e2_share": float((dp <= 1e-2).mean()),
        "abs_dprice_p50": float(dp.quantile(0.5)), "abs_dprice_p99": float(dp.quantile(0.99)), "abs_dprice_max": float(dp.max()),
        "size_within_1e6": int((ds <= 1e-6).sum()), "size_within_1e6_share": float((ds <= 1e-6).mean()),
        "abs_dsize_p99": float(ds.quantile(0.99)),
        "clock_print_minus_block_s_p50": float((m["ts"] - m["chain_ts"]).dt.total_seconds().quantile(0.5)),
        "clock_print_minus_block_s_p99": float((m["ts"] - m["chain_ts"]).dt.total_seconds().quantile(0.99)),
    }


def coverage(prints: pl.DataFrame, taker: pl.DataFrame, j: pl.DataFrame, day: str | None = None) -> dict:
    matched_hashes = j.filter(pl.col("matched"))["transaction_hash"].to_list()
    t = taker.with_columns(pl.col("tx_hash").is_in(matched_hashes).alias("matched"))
    within = {}
    if day is not None:
        # the scrape runs 300 blocks past midnight so the day's last prints settle; those legs
        # cannot have a print in the day's files and do not belong in the denominator
        end = dt.datetime.fromisoformat(day) + dt.timedelta(days=1)
        tw = t.filter(pl.col("ts") < end)
        un = tw.filter(~pl.col("matched"))
        within = {"taker_legs_within_day": tw.height, "taker_matched_within_day": int(tw["matched"].sum()),
                  "taker_matched_share_within_day": float(tw["matched"].mean()) if tw.height else None,
                  "hour0_share_of_unmatched_within_day": float((un["ts"].dt.hour() == 0).mean()) if un.height else 0.0,
                  "unmatched_within_day": un.height}
    by_hour = {}
    for h, n, m in j.group_by(pl.col("ts").dt.hour().alias("h")).agg(pl.len().alias("n"), pl.col("matched").sum().alias("m")).sort("h").iter_rows():
        by_hour[str(h)] = {"prints": int(n), "matched": int(m), "share": float(m / n)}
    by_fill_hour = {}
    for h, n, m in t.group_by(pl.col("ts").dt.hour().alias("h")).agg(pl.len().alias("n"), pl.col("matched").sum().alias("m")).sort("h").iter_rows():
        by_fill_hour[str(h)] = {"taker_legs": int(n), "matched": int(m), "share": float(m / n)}
    by_contract = {}
    for c, n, m in t.group_by("contract").agg(pl.len().alias("n"), pl.col("matched").sum().alias("m")).iter_rows():
        by_contract[str(c)] = {"taker_legs": int(n), "matched": int(m), "share": float(m / n)}
    return {
        "prints": prints.height, "prints_with_hash": int(prints["transaction_hash"].is_not_null().sum()),
        "taker_legs": taker.height, "taker_txs": taker["tx_hash"].n_unique(),
        "print_matched": int(j["matched"].sum()), "print_matched_share": float(j["matched"].mean()),
        "taker_matched": int(t["matched"].sum()), "taker_matched_share": float(t["matched"].mean()),
        "by_print_hour": by_hour, "by_fill_hour": by_fill_hour, "by_contract": by_contract, **within,
    }


def identity_by_type(j: pl.DataFrame, types: pl.DataFrame) -> dict:
    """Identity counts on matched pairs, by the settlement type of the transaction."""
    m = j.filter(pl.col("matched")).join(types.select("tx_hash", "match_type"), left_on="transaction_hash", right_on="tx_hash", how="left")
    out = {}
    for (k,), g in m.group_by(["match_type"]):
        out[str(k)] = {"matched": g.height,
                       "side_equal": int(((g["side"] == "BUY") == (g["chain_side"] == 0)).sum()),
                       "token_equal": int((g["asset_id"] == g["chain_token"]).sum()),
                       "price_within_5e4": int(((g["price"] - g["chain_price"]).abs() <= 5e-4).sum()),
                       "size_within_1e6": int(((g["size"] - g["chain_size"]).abs() <= 1e-6).sum())}
    return out


def duplicate_prints(prints: pl.DataFrame) -> dict:
    n, u = prints.height, prints["transaction_hash"].n_unique()
    return {"prints": n, "distinct_hashes": u, "duplicates": n - u}


def unknown_cause(mt: pl.DataFrame, known_tokens: set) -> dict:
    """How many UNKNOWN settlements involve a maker token absent from the token map (a coverage gap, not a third token)."""
    u = mt.filter(pl.col("match_type") == "UNKNOWN")
    absent = 0
    for toks in u["unknown_tokens"].to_list():
        if any(t not in known_tokens for t in (toks or [])):
            absent += 1
    return {"unknown_txs": u.height, "with_token_absent_from_map": absent}


def fuzzy_match(unmatched_prints: pl.DataFrame, unmatched_taker: pl.DataFrame) -> dict:
    """Pair leftover prints to leftover taker legs on (asset, side, price, size) within the window."""
    if unmatched_prints.height == 0 or unmatched_taker.height == 0:
        return {"unmatched_prints": unmatched_prints.height, "unmatched_taker": unmatched_taker.height, "fuzzy_matched": 0, "pairs": [], "all_pairs": []}
    px, sz = price_size(unmatched_taker)
    t = unmatched_taker.with_columns(px.round(4).alias("k_price"), sz.round(6).alias("k_size"),
                                     pl.when(pl.col("side") == 0).then(pl.lit("BUY")).otherwise(pl.lit("SELL")).alias("k_side"),
                                     pl.col("token_id").alias("k_asset"), pl.col("ts").cast(pl.Datetime("us")).alias("chain_ts")).sort("chain_ts")
    p = unmatched_prints.with_columns(pl.col("price").round(4).alias("k_price"), pl.col("size").round(6).alias("k_size"),
                                      pl.col("side").alias("k_side"), pl.col("asset_id").alias("k_asset"),
                                      pl.col("ts").cast(pl.Datetime("us"))).sort("ts")
    j = p.join_asof(t, left_on="ts", right_on="chain_ts", by=["k_asset", "k_side", "k_price", "k_size"],
                    strategy="nearest", tolerance=dt.timedelta(seconds=FUZZY_WINDOW_S))
    hit = j.filter(pl.col("tx_hash").is_not_null()).unique("tx_hash")
    return {
        "unmatched_prints": unmatched_prints.height, "unmatched_taker": unmatched_taker.height,
        "fuzzy_matched": hit.height,
        "fuzzy_share_of_unmatched_prints": hit.height / unmatched_prints.height,
        "pairs": list(zip(hit["transaction_hash"].to_list(), hit["tx_hash"].to_list()))[:50],
        "all_pairs": list(zip(hit["transaction_hash"].to_list(), hit["tx_hash"].to_list())),
        "dt_s_p50": float((hit["ts"] - hit["chain_ts"]).dt.total_seconds().abs().quantile(0.5)) if hit.height else None,
    }


def composition(mt: pl.DataFrame, all_legs: pl.DataFrame) -> dict:
    px, sz = price_size(mt.rename({"taker_side": "side", "taker_amount_in": "maker_amount", "taker_amount_out": "taker_amount"}).select("side", "maker_amount", "taker_amount"))
    mt = mt.with_columns(notional=(px * sz))
    by = mt.group_by("match_type").agg(n=pl.len(), notional=pl.col("notional").sum(), maker_legs=pl.col("maker_legs").sum(),
                                        buy_share=(pl.col("taker_side") == 0).mean())
    tot_n, tot_v = mt.height, float(mt["notional"].sum())
    types = {r["match_type"]: {"txs": int(r["n"]), "tx_share": r["n"] / tot_n, "notional_share": (r["notional"] / tot_v) if tot_v else None,
                               "maker_legs": int(r["maker_legs"] or 0), "taker_buy_share": float(r["buy_share"])} for r in by.to_dicts()}
    return {
        "txs": tot_n, "types": types,
        "taker_buy_share": float((mt["taker_side"] == 0).mean()),
        "all_legs_buy_share": float((all_legs["side"] == 0).mean()),
        "all_legs": all_legs.height,
    }


def main() -> None:
    day = sys.argv[1] if len(sys.argv) > 1 else "2026-05-13"
    con = connect()
    prints = feed_prints(day, con)
    taker = ld.taker_legs(day)
    fills = ld.fills(day)
    j = join_by_hash(prints, taker)
    matched = j.filter(pl.col("matched"))["transaction_hash"].to_list()
    unmatched_p = j.filter(~pl.col("matched")).select(prints.columns)
    unmatched_t = taker.filter(~pl.col("tx_hash").is_in(matched))
    fz = fuzzy_match(unmatched_p, unmatched_t)
    res = {"day": day, "coverage": coverage(prints, taker, j, day=day), "identity": identity_stats(j), "fuzzy": fz,
           "duplicates": duplicate_prints(prints)}
    complement = ld.complement_from_token_map()
    mt = ld.match_types(day, complement)
    res["composition"] = composition(mt, fills)
    res["composition"]["complement_map_available"] = bool(complement)
    res["composition"]["same_token_same_side_legs"] = int(mt["n_o"].fill_null(0).sum())
    res["identity_by_type"] = identity_by_type(j, mt)
    tm = EXT / "token_map.parquet"
    known = set(pl.read_parquet(tm, columns=["asset_id"])["asset_id"].to_list()) if tm.exists() else set()
    res["unknown_cause"] = unknown_cause(mt, known)
    # the residual after the second stage, for the receipt sampler (a1_receipts.py), with its hourly population
    out_dir = EXT / "onchain" / f"day={day}"
    fuzzy_hashes = [a for a, _ in fz.get("all_pairs", [])]
    remaining = unmatched_p.filter(~pl.col("transaction_hash").is_in(fuzzy_hashes))
    remaining.write_parquet(out_dir / "unmatched_prints.parquet")
    unmatched_t.write_parquet(out_dir / "unmatched_taker.parquet")
    res["fuzzy"]["remaining_prints"] = remaining.height
    res["fuzzy"]["remaining_by_hour"] = {str(h): int(n) for h, n in remaining.group_by(pl.col("ts").dt.hour().alias("h")).len().sort("h").iter_rows()}
    res["fuzzy"].pop("all_pairs", None)
    write_json(f"a1_fill_match_{day}", res, inputs=[str(EXT / "onchain" / f"day={day}")] + day_files(day, "v2")[:1])
    c, i = res["coverage"], res["identity"]
    print(f"{day}: prints {c['prints']:,} matched {c['print_matched_share']:.3f}; taker legs {c['taker_legs']:,} matched "
          f"{c['taker_matched_share']:.3f}; side equal {i.get('side_equal_share', float('nan')):.4f}; token equal "
          f"{i.get('token_equal_share', float('nan')):.4f}; |dprice|<=5e-4 {i.get('price_within_5e4_share', float('nan')):.3f}; "
          f"fuzzy {res['fuzzy']['fuzzy_matched']:,} of {res['fuzzy']['unmatched_prints']:,}")
    print({k: round(v["tx_share"], 3) for k, v in res["composition"]["types"].items()},
          "taker buy share", round(res["composition"]["taker_buy_share"], 3), "all legs", round(res["composition"]["all_legs_buy_share"], 3))


if __name__ == "__main__":
    main()

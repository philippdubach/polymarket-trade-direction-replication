"""Harmonised contract-day extraction over both feed archives (V1 and V2).

One UTC day of one archive becomes one row per (condition, asset) with only
additive aggregates plus bucket quantiles, so the two archives stack into one
170-day panel:

  quotes   updates, spread sum / sq / max, exact-to-0.001 median, p25, p75, last mid,
           hours present, updates quoting an empty book (spread >= 0.9),
           first quote, first live quote (spread < 0.9), last quote
  depth    price_change rows at the best level: count and displayed size by side
           (event-weighted, no book replay)
  book     snapshots: count, best-level size (last and mean), ordering mismatches
  trades   V2 only: count, notional, buy/sell split, first trade
  ticks    V2 only, separate output: the last quote within 60 min before each tick_size_change

V1 rows are JSON payloads (token_id, YES/NO side, best bid/ask, change fields,
bids/asks arrays); V2 rows are columnar. Both are read with receipt time only.
The median follows the pilot's `exact_median_hist.py`: the 0.001 bucket that
contains position n/2 of the sorted spreads.

Three queries per day, each a single streaming scan. A day holds ~2 billion
rows, and DuckDB materialises a CTE that is referenced more than once, so no
query references its source subquery twice: the first version did, and spilled
24 GB before it was stopped.

Run:
    uv run python x0_extract_daily.py --era v2            # every V2 day, resumable
    uv run python x0_extract_daily.py --era v1 --days 2026-03-25
Output:
    /Volumes/data/paper-b-ext/shards_h/era=<era>/day=YYYY-MM-DD.parquet
    /Volumes/data/paper-b-ext/tickctx/day=YYYY-MM-DD.parquet     (v2)
    /Volumes/data/paper-b-ext/shards_h/extract.log
"""
from __future__ import annotations

import argparse
import os
import time

import polars as pl

from d0_common import EXT, all_days, connect, day_files

SHARDS_H = EXT / "shards_h"
TICKCTX = EXT / "tickctx"
BINS = 1000
# Spreads are differences of two doubles (0.57 - 0.56 = 0.00999...), so every threshold and bucket
# test rounds the difference to 1e-6 first; a raw floor puts a tick-grid spread one bucket low.
EMPTY = 0.9   # a quoted spread at or above this is a placeholder book


def _files_sql(files: list[str]) -> str:
    return "[" + ", ".join(f"'{f}'" for f in files) + "]"


# --- one derived-row source per era; used exactly once per query ------------------------------

def source_v2(files: list[str], kinds: str) -> str:
    return f"""(
  SELECT lower(decode(market)) AS condition_id, asset_id, NULL::VARCHAR AS outcome,
         timestamp_received::TIMESTAMP AS ts, event_type,
         CAST(best_bid AS DOUBLE) AS bid, CAST(best_ask AS DOUBLE) AS ask,
         CAST(price AS DOUBLE) AS px, CAST(size AS DOUBLE) AS sz, side,
         CASE WHEN event_type = 'book' THEN json_extract_string(bids, '$[#-1][1]') END AS bid_best_sz,
         CASE WHEN event_type = 'book' THEN json_extract_string(asks, '$[#-1][1]') END AS ask_best_sz,
         CASE WHEN event_type = 'book' THEN json_extract_string(bids, '$[0][0]') END AS bid_first_px,
         CASE WHEN event_type = 'book' THEN json_extract_string(bids, '$[#-1][0]') END AS bid_last_px,
         CAST(old_tick_size AS DOUBLE) AS old_tick, CAST(new_tick_size AS DOUBLE) AS new_tick
  FROM read_parquet({_files_sql(files)})
  WHERE event_type IN ({kinds})
)"""


def source_v1(files: list[str], kinds: str) -> str:
    kinds_v1 = kinds.replace("'book'", "'book_snapshot'")
    return f"""(
  SELECT condition_id, j[1] AS asset_id, j[2] AS outcome, ts, event_type,
         CAST(j[3] AS DOUBLE) AS bid, CAST(j[4] AS DOUBLE) AS ask,
         CAST(j[5] AS DOUBLE) AS px, CAST(j[6] AS DOUBLE) AS sz, j[7] AS side,
         b[1] AS bid_best_sz, b[2] AS ask_best_sz, b[3] AS bid_first_px, b[4] AS bid_last_px,
         NULL::DOUBLE AS old_tick, NULL::DOUBLE AS new_tick
  FROM (
    SELECT lower(market_id) AS condition_id, timestamp_received::TIMESTAMP AS ts,
           CASE update_type WHEN 'book_snapshot' THEN 'book' ELSE update_type END AS event_type,
           json_extract_string(data, ['$.token_id', '$.side', '$.best_bid', '$.best_ask',
                                      '$.change_price', '$.change_size', '$.change_side']) AS j,
           CASE WHEN update_type = 'book_snapshot'
                THEN json_extract_string(data, ['$.bids[#-1][1]', '$.asks[#-1][1]', '$.bids[0][0]', '$.bids[#-1][0]']) END AS b
    FROM read_parquet({_files_sql(files)})
    WHERE update_type IN ({kinds_v1})
  )
)"""


def source(era: str, files: list[str], kinds: str) -> str:
    return source_v2(files, kinds) if era == "v2" else source_v1(files, kinds)


# --- query 1: everything at the asset level in one GROUP BY -----------------------------------

def q_asset(era: str, files: list[str]) -> str:
    src = source(era, files, "'price_change', 'book', 'last_trade_price'")
    pc = "event_type = 'price_change' AND bid IS NOT NULL AND ask IS NOT NULL AND ask >= bid"
    # presence per hour as one 24-bit mask: a single 8-byte state per group, where 24 counts,
    # an arg_max per hour or a distinct-hour set pushed 2.4M groups x 4 threads past the memory budget
    hourly = f"bit_or(CAST(1 << hour(ts) AS BIGINT)) FILTER (WHERE {pc}) AS hour_mask"
    trades = (
        "count(*) FILTER (WHERE event_type = 'last_trade_price') AS trades,\n"
        "sum(px * sz) FILTER (WHERE event_type = 'last_trade_price') AS trade_volume,\n"
        "count(*) FILTER (WHERE event_type = 'last_trade_price' AND side = 'BUY') AS trades_buy,\n"
        "count(*) FILTER (WHERE event_type = 'last_trade_price' AND side = 'SELL') AS trades_sell,\n"
        "min(ts) FILTER (WHERE event_type = 'last_trade_price') AS first_trade_ts"
        if era == "v2" else
        "NULL::BIGINT AS trades, NULL::DOUBLE AS trade_volume, NULL::BIGINT AS trades_buy, NULL::BIGINT AS trades_sell,\n"
        "NULL::TIMESTAMP AS first_trade_ts"
    )
    return f"""
SELECT any_value(condition_id) AS condition_id, asset_id, any_value(outcome) AS outcome,
       count(*) FILTER (WHERE {pc}) AS updates,
       sum(ask - bid) FILTER (WHERE {pc}) AS spread_sum,
       sum((ask - bid) * (ask - bid)) FILTER (WHERE {pc}) AS spread_sq_sum,
       max(ask - bid) FILTER (WHERE {pc}) AS spread_max,
       arg_max((bid + ask) / 2.0, ts) FILTER (WHERE {pc}) AS mid_last,
       count(*) FILTER (WHERE {pc} AND round(ask - bid, 6) >= {EMPTY}) AS updates_empty,
       {hourly},
       count(*) FILTER (WHERE {pc} AND side = 'BUY' AND px = bid) AS depth_bid_n,
       sum(sz) FILTER (WHERE {pc} AND side = 'BUY' AND px = bid) AS depth_bid_sum,
       count(*) FILTER (WHERE {pc} AND side = 'SELL' AND px = ask) AS depth_ask_n,
       sum(sz) FILTER (WHERE {pc} AND side = 'SELL' AND px = ask) AS depth_ask_sum,
       count(*) FILTER (WHERE event_type = 'book') AS n_book,
       arg_max(CAST(bid_best_sz AS DOUBLE), ts) FILTER (WHERE event_type = 'book') AS book_bid_best_size,
       arg_max(CAST(ask_best_sz AS DOUBLE), ts) FILTER (WHERE event_type = 'book') AS book_ask_best_size,
       avg(CAST(bid_best_sz AS DOUBLE)) FILTER (WHERE event_type = 'book') AS book_bid_best_mean,
       avg(CAST(ask_best_sz AS DOUBLE)) FILTER (WHERE event_type = 'book') AS book_ask_best_mean,
       count(*) FILTER (WHERE event_type = 'book' AND CAST(bid_last_px AS DOUBLE) < CAST(bid_first_px AS DOUBLE)) AS book_order_mismatch,
       {trades},
       min(ts) FILTER (WHERE {pc}) AS first_quote_ts,
       min(ts) FILTER (WHERE {pc} AND round(ask - bid, 6) < {EMPTY}) AS first_live_quote_ts,
       max(ts) FILTER (WHERE {pc}) AS last_quote_ts
FROM {src}
GROUP BY asset_id
"""


# --- query 2: bucket quantiles of the spread ------------------------------------------------------

def q_quantiles(era: str, files: list[str]) -> str:
    src = source(era, files, "'price_change'")
    return f"""
WITH hist AS (
  SELECT asset_id, least({BINS}, CAST(floor(round((ask - bid) * {BINS}, 6)) AS INTEGER)) AS bucket, count(*) AS n
  FROM {src}
  WHERE bid IS NOT NULL AND ask IS NOT NULL AND ask >= bid
  GROUP BY 1, 2
),
hc AS (
  SELECT *, sum(n) OVER (PARTITION BY asset_id ORDER BY bucket
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum,
            sum(n) OVER (PARTITION BY asset_id) AS total
  FROM hist
)
SELECT asset_id,
       min(bucket) FILTER (WHERE cum >= total * 0.25) / {BINS}.0 AS spread_p25,
       min(bucket) FILTER (WHERE cum >= total * 0.5) / {BINS}.0 AS spread_median,
       min(bucket) FILTER (WHERE cum >= total * 0.75) / {BINS}.0 AS spread_p75
FROM hc GROUP BY 1
"""


# --- query 3 (v2): last quote before each tick change ---------------------------------------------

def q_tickctx(files: list[str]) -> str:
    """Last quote within 60 minutes before each tick_size_change in `tk_tbl`.

    `tk_tbl` is a small registered table of the day's tick events, so the hash
    join builds on it and streams the quotes past it once. A CTE from a parquet
    scan had unknown cardinality, DuckDB built on the quote side and ran out of
    memory; an ASOF join sorted every quote of every tick asset and took 100 min.
    """
    fs = _files_sql(files)
    return f"""
WITH cand AS (
  SELECT tk.asset_id, tk.ts, pc.qts, pc.bid, pc.ask
  FROM tk_tbl tk JOIN (
    SELECT asset_id, timestamp_received::TIMESTAMP AS qts, CAST(best_bid AS DOUBLE) AS bid, CAST(best_ask AS DOUBLE) AS ask
    FROM read_parquet({fs})
    WHERE event_type = 'price_change' AND best_bid IS NOT NULL AND best_ask IS NOT NULL
  ) pc ON tk.asset_id = pc.asset_id AND pc.qts < tk.ts AND pc.qts >= tk.ts - INTERVAL 60 MINUTE
)
SELECT tk.condition_id, tk.asset_id, tk.ts, tk.old_tick, tk.new_tick,
       arg_max(cand.bid, cand.qts) AS bid_before, arg_max(cand.ask, cand.qts) AS ask_before, max(cand.qts) AS quote_before_ts
FROM tk_tbl tk LEFT JOIN cand ON tk.asset_id = cand.asset_id AND tk.ts = cand.ts
GROUP BY 1, 2, 3, 4, 5
"""


def day_ticks(day: str) -> pl.DataFrame:
    """The day's tick_size_change events from the pilot's extraction, in naive UTC."""
    from d0_common import decode_market_id, ticks_utc
    tk = ticks_utc().filter(pl.col("ts").dt.date() == pl.lit(day).str.to_date())
    return tk.select(
        pl.col("market_id").map_elements(decode_market_id, return_dtype=pl.Utf8).alias("condition_id"),
        "asset_id", pl.col("ts").cast(pl.Datetime("us")),
        pl.col("old_tick").cast(pl.Float64), pl.col("new_tick").cast(pl.Float64),
    )


TIMINGS: dict[str, float] = {}

COLUMNS = [
    "condition_id", "asset_id", "outcome", "updates", "spread_sum", "spread_sq_sum", "spread_max",
    "spread_median", "spread_p25", "spread_p75", "mid_last", "hours_present", "updates_empty",
    "depth_bid_n", "depth_bid_sum", "depth_ask_n", "depth_ask_sum",
    "n_book", "book_bid_best_size", "book_ask_best_size", "book_bid_best_mean", "book_ask_best_mean", "book_order_mismatch",
    "trades", "trade_volume", "trades_buy", "trades_sell",
    "first_quote_ts", "first_live_quote_ts", "last_quote_ts", "first_trade_ts", "era", "day",
]


def _frame(con, q: str) -> pl.DataFrame:
    df = pl.from_arrow(con.execute(q).to_arrow_table())
    assert isinstance(df, pl.DataFrame)
    return df


def extract_day(day: str, era: str, files: list[str] | None = None, con=None) -> pl.DataFrame:
    """One row per (condition, asset) for a UTC day of one archive."""
    files = files if files is not None else day_files(day, era)
    con = con or connect()
    t = time.time()
    a = _frame(con, q_asset(era, files)).filter(pl.col("updates") > 0)
    a = a.with_columns(
        pl.sum_horizontal([((pl.col("hour_mask") // (1 << h)) % 2).cast(pl.Int64) for h in range(24)]).alias("hours_present")
    ).drop("hour_mask")
    TIMINGS["asset"] = round(time.time() - t, 1)
    t = time.time()
    qn = _frame(con, q_quantiles(era, files))
    TIMINGS["quantiles"] = round(time.time() - t, 1)
    out = a.join(qn, on="asset_id", how="left").with_columns(
        pl.lit(era).alias("era"), pl.lit(day).str.to_date().alias("day")
    )
    return out.select(COLUMNS).sort(["condition_id", "asset_id"])


def tick_context(day: str, files: list[str] | None = None, con=None, ticks: pl.DataFrame | None = None) -> pl.DataFrame:
    """Every tick_size_change of the day with the last quote within 60 minutes before it."""
    files = files if files is not None else day_files(day, "v2")
    con = con or connect()
    tk = ticks if ticks is not None else day_ticks(day)
    t = time.time()
    if tk.height == 0:
        return tk.with_columns(pl.lit(None, dtype=pl.Float64).alias("bid_before"), pl.lit(None, dtype=pl.Float64).alias("ask_before"),
                               pl.lit(None, dtype=pl.Datetime("us")).alias("quote_before_ts"))
    con.register("tk_tbl", tk.to_arrow())
    out = _frame(con, q_tickctx(files)).sort(["asset_id", "ts"])
    con.unregister("tk_tbl")
    TIMINGS["tickctx"] = round(time.time() - t, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", choices=["v1", "v2"], required=True)
    ap.add_argument("--days", nargs="*")
    a = ap.parse_args()
    out_dir = SHARDS_H / f"era={a.era}"
    out_dir.mkdir(parents=True, exist_ok=True)
    TICKCTX.mkdir(parents=True, exist_ok=True)
    log = open(SHARDS_H / "extract.log", "a")
    days = a.days or all_days(a.era)
    todo = [d for d in days if not (out_dir / f"day={d}.parquet").exists()]
    print(f"era={a.era} days={len(days)} todo={len(todo)}", flush=True)
    con = connect(memory="8GB", threads=6)
    t_all = time.time()
    for i, d in enumerate(todo, 1):
        fs = day_files(d, a.era)
        if not fs:
            continue
        t = time.time()
        df = extract_day(d, a.era, fs, con)
        tmp = out_dir / f"day={d}.tmp.parquet"
        df.write_parquet(tmp)
        os.replace(tmp, out_dir / f"day={d}.parquet")
        n_tick = 0
        if a.era == "v2":
            tc = tick_context(d, fs, con)
            n_tick = tc.height
            tc.write_parquet(TICKCTX / f"day={d}.parquet")
        msg = (f"[{i}/{len(todo)}] {a.era} {d}: {len(fs)}h -> {df.height:,} rows, {n_tick:,} tick events "
               f"({time.time()-t:.0f}s {dict(TIMINGS)}, total {(time.time()-t_all)/60:.1f} min)")
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()


if __name__ == "__main__":
    main()

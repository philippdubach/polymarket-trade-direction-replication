"""Read the extended on-chain scrape (a0_scrape_days.py output) for one UTC day.

  fills(day)         every OrderFilled leg, with `ts` (UTC, from the block timestamp) and the
                     5-second `bucket` the direction tests use
  taker_legs(day)    the one leg per transaction whose `taker` is the exchange: the taker order
  match_types(day)   per transaction: COMPLEMENTARY / MINT / MERGE / MIXED / UNKNOWN from the legs
  onchain_fills(day) the pilot loader's columns, so t1_direction.py runs on the new days unchanged

The exchange settles a taker order against maker orders on the same token
(complementary), or against makers on the complement token by minting or
merging complete sets. The classification follows the legs alone: taker on
token a with side s; every maker leg on a with side -s is COMPLEMENTARY,
every maker leg on complement(a) with side s is MINT (s = BUY) or MERGE
(s = SELL), anything else is MIXED. Without a complement map a maker leg on a
different token is UNKNOWN.
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from a0_scrape_days import OUT, SCHEMA  # noqa: F401  (OUT is monkeypatched in tests)
from d0_common import V1_NEG, V1_STD, V2_NEG, V2_STD

BUCKET = "5s"
EXCH = [V2_STD, V2_NEG, V1_STD, V1_NEG]


_CACHE: dict[tuple[str, str], pl.DataFrame] = {}


def _day(day: str) -> pl.DataFrame:
    """The day's slices, read once per process (the fill match, the taker legs and the match types all need them)."""
    import a0_onchain_load as me
    key = (str(me.OUT), day)
    if key not in _CACHE:
        files = sorted((me.OUT / f"day={day}").glob("blocks_*.parquet"))
        if not files:
            raise SystemExit(f"no scraped slices for {day} under {me.OUT}")
        _CACHE.clear()
        _CACHE[key] = pl.concat([pl.read_parquet(f) for f in files])
    return _CACHE[key]


def fills(day: str) -> pl.DataFrame:
    df = _day(day).filter(pl.col("kind") == "orderfilled")
    return df.with_columns(
        (pl.lit(dt.datetime(1970, 1, 1)) + pl.duration(seconds=pl.col("block_ts"))).cast(pl.Datetime("ms")).alias("ts")
    ).with_columns(pl.col("ts").dt.truncate(BUCKET).alias("bucket"))


def taker_legs(day: str) -> pl.DataFrame:
    return fills(day).filter(pl.col("taker").is_in(EXCH))


def match_types(day: str, complement: dict[str, str]) -> pl.DataFrame:
    f = fills(day)
    tak = f.filter(pl.col("taker").is_in(EXCH)).select(
        "tx_hash", taker_side=pl.col("side"), taker_token=pl.col("token_id"), taker_maker=pl.col("maker"),
        taker_amount_in=pl.col("maker_amount"), taker_amount_out=pl.col("taker_amount"), taker_fee=pl.col("fee"), ts=pl.col("ts"),
    )
    mk = f.filter(~pl.col("taker").is_in(EXCH)).join(tak.select("tx_hash", "taker_side", "taker_token"), on="tx_hash", how="inner")
    comp = pl.col("taker_token").replace_strict(complement, default=None, return_dtype=pl.Utf8) if complement else pl.lit(None, dtype=pl.Utf8)
    mk = mk.with_columns(
        leg=pl.when((pl.col("token_id") == pl.col("taker_token")) & (pl.col("side") != pl.col("taker_side"))).then(pl.lit("C"))
        .when((pl.col("token_id") == comp) & (pl.col("side") == pl.col("taker_side"))).then(pl.lit("X"))
        .when(pl.col("token_id") == pl.col("taker_token")).then(pl.lit("O"))
        .otherwise(pl.lit("U"))
    )
    per_tx = mk.group_by("tx_hash").agg(
        maker_legs=pl.len(),
        n_c=(pl.col("leg") == "C").sum(), n_x=(pl.col("leg") == "X").sum(), n_u=(pl.col("leg") == "U").sum(),
        n_o=(pl.col("leg") == "O").sum(),
        unknown_tokens=pl.col("token_id").filter(pl.col("leg") == "U").unique(),
        maker_amount_sum=pl.col("maker_amount").cast(pl.Float64).sum(),
        taker_amount_sum=pl.col("taker_amount").cast(pl.Float64).sum(),
    )
    out = tak.join(per_tx, on="tx_hash", how="left").with_columns(
        match_type=pl.when(pl.col("maker_legs").is_null()).then(pl.lit("NO_MAKER"))
        .when(pl.col("n_c") == pl.col("maker_legs")).then(pl.lit("COMPLEMENTARY"))
        .when((pl.col("n_x") == pl.col("maker_legs")) & (pl.col("taker_side") == 0)).then(pl.lit("MINT"))
        .when((pl.col("n_x") == pl.col("maker_legs")) & (pl.col("taker_side") == 1)).then(pl.lit("MERGE"))
        .when(pl.col("n_u") > 0).then(pl.lit("UNKNOWN"))
        .otherwise(pl.lit("MIXED"))
    )
    return out.sort("ts")


def complement_from_token_map() -> dict[str, str]:
    """asset -> its complement, from x1's token map; empty if the map does not exist yet."""
    from d0_common import EXT
    tm = EXT / "token_map.parquet"
    if not tm.exists():
        return {}
    import x1_token_map as x1
    return x1.complement_map(pl.read_parquet(tm, columns=["asset_id", "condition_id", "outcome"]))


def onchain_fills(day: str) -> pl.DataFrame:
    """The columns direction_validation.onchain_fills() returned, from the new scrape."""
    return fills(day).select(
        pl.col("token_id").alias("asset_id"), pl.col("side").alias("chain_side"), pl.col("block_number"),
        pl.col("fee").cast(pl.Float64, strict=False).alias("chain_fee_raw"), pl.col("bucket"),
        pl.col("tx_hash"), pl.col("maker"), pl.col("taker"), pl.col("maker_amount"), pl.col("taker_amount"), pl.col("ts"),
    )

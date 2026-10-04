"""Shared event dataset for the Paper B v2 revision.

Reads the same inputs as the pilot (`../paper-b-v2-pilot/ticks`, `shards`) and
builds one row per 0.01 -> 0.001 adoption event with everything the revision
tests need. The pilot scripts are left untouched as provenance.

Definitions match the pilot exactly where the pilot is reproduced
(`did_pilot`), so Table 4 can be recovered to the printed digits before any
correction is applied.
"""
from __future__ import annotations

import datetime as dt
import glob
from functools import lru_cache
from pathlib import Path

import polars as pl

from d0_common import PILOT, WHEN

HERE = Path(__file__).parent
# Large parquet caches (events_W*.parquet) stay outside git, where the pilot
# session left them; JSON results are written next to the scripts.
CACHE = WHEN / "outputs" / "paper-b-v2-revision"
TICKS = PILOT / "ticks"
SHARDS = PILOT / "shards"
SAMPLE_START = dt.date(2026, 4, 15)
SAMPLE_END = dt.date(2026, 8, 10)
CUTOVER = dt.date(2026, 4, 28)
FLOOR = 0.01


def utc_naive(s: pl.Series) -> pl.Series:
    """A timestamp series as naive UTC.

    The pilot's `ticks/` files were written by a DuckDB session whose zone was
    Europe/Zurich, so `timestamp_received` is tz-aware Zurich time. Stripping
    the zone keeps the Zurich wall clock and moves every event after 22:00 UTC
    to the next UTC day (7.0% of adoptions). Convert first, then strip.
    """
    if s.dtype.time_zone is not None:
        s = s.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return s.cast(pl.Datetime("us"))


@lru_cache(maxsize=2)
def adoption(legacy_tz: bool = False) -> pl.DataFrame:
    """One row per asset: its first 0.01 -> 0.001 tick change.

    `legacy_tz=True` reproduces the pilot's Zurich-wall-clock timestamps and
    exists only so the printed Table 4 can still be recovered as a gate.
    """
    fs = sorted(glob.glob(str(TICKS / "day=*.parquet")))
    ad = (
        pl.concat([pl.read_parquet(f, columns=["market_id", "asset_id", "timestamp_received", "old_tick", "new_tick"]) for f in fs])
        .filter((pl.col("old_tick") == 0.01) & (pl.col("new_tick") == 0.001))
        .sort("timestamp_received")
        .group_by("asset_id")
        .agg(adoption=pl.col("timestamp_received").min(), market_id=pl.col("market_id").first())
    )
    if legacy_tz:
        return ad.with_columns(pl.col("adoption").dt.replace_time_zone(None).cast(pl.Datetime("us")))
    return ad.with_columns(utc_naive(ad["adoption"]).alias("adoption"))


@lru_cache(maxsize=1)
def panel() -> pl.DataFrame:
    fs = sorted(glob.glob(str(SHARDS / "day=*.parquet")))
    return (
        pl.concat([pl.read_parquet(f, columns=["asset_id", "day", "updates", "spread_sum", "trades", "trade_volume"]) for f in fs])
        .filter(pl.col("updates") > 0)
        .with_columns((pl.col("spread_sum") / pl.col("updates")).alias("s"))
        .filter(pl.col("s").is_finite())
    )


@lru_cache(maxsize=1)
def daily_median() -> pl.DataFrame:
    """Calendar-day cross-sectional median spread over all contracts: the pilot's control."""
    return panel().group_by("day").agg(cs=pl.col("s").median()).sort("day")


def _pilot_day_value():
    """The pilot's control lookup, including its edge clamping, reproduced exactly."""
    dm = daily_median()
    days, vals = dm["day"].to_list(), dm["cs"].to_list()
    base = dt.datetime.combine(days[0], dt.time())

    def value(ts: dt.datetime) -> float:
        idx = max(0, min(int((ts - base).days), len(vals) - 1))
        return vals[idx]

    return value


def events(W: int = 7, legacy_tz: bool = False) -> pl.DataFrame:
    """One row per adoption event with pre/post windows of half-width W days."""
    ad = adoption(legacy_tz)
    p = panel()
    first = p.group_by("asset_id").agg(first_day=pl.col("day").min(), last_day=pl.col("day").max())
    j = p.join(ad.select("asset_id", "adoption"), on="asset_id", how="inner").with_columns(
        (pl.col("day").cast(pl.Datetime("us")) - pl.col("adoption")).dt.total_days().alias("d")
    )
    near = j.filter(pl.col("d").abs() <= W)
    agg = lambda sfx: [
        pl.col("s").median().alias(f"s_{sfx}"),
        pl.col("updates").sum().alias(f"n_{sfx}"),
        pl.col("updates").mean().alias(f"u_{sfx}"),
        pl.col("trades").mean().alias(f"tr_{sfx}"),
        pl.len().alias(f"days_{sfx}"),
    ]
    pre = near.filter(pl.col("d") < 0).group_by(["asset_id", "adoption"]).agg(agg("pre"))
    post = near.filter(pl.col("d") >= 0).group_by(["asset_id", "adoption"]).agg(agg("post"))
    ev = (
        pre.join(post, on=["asset_id", "adoption"], how="inner")
        .filter((pl.col("n_pre") > 0) & (pl.col("n_post") > 0))
        .join(ad.select("asset_id", "market_id"), on="asset_id", how="left")
        .join(first, on="asset_id", how="left")
        .with_columns(adoption_day=pl.col("adoption").dt.date())
        .with_columns(age_days=(pl.col("adoption_day") - pl.col("first_day")).dt.total_days())
    )

    # Pilot control: cross-sectional median on the single days a-W and a+W, edge-clamped.
    value = _pilot_day_value()
    rows = ev.select("asset_id", "adoption").to_dicts()
    cpre = [value(r["adoption"] - dt.timedelta(days=W)) for r in rows]
    cpost = [value(r["adoption"] + dt.timedelta(days=W)) for r in rows]
    clamped = [
        not (SAMPLE_START <= (r["adoption"] - dt.timedelta(days=W)).date() and (r["adoption"] + dt.timedelta(days=W)).date() <= SAMPLE_END)
        for r in rows
    ]
    ev = ev.with_columns(
        pl.Series("ctrl_pre", cpre), pl.Series("ctrl_post", cpost), pl.Series("ctrl_clamped", clamped)
    )

    # Corrected control: median of the daily cross-sectional median over the
    # same calendar windows the treated side uses, with no edge clamping.
    # It depends only on the adoption day, so compute it per distinct day.
    dm = daily_median()
    win = ev.select("adoption_day").unique().join(dm, how="cross").with_columns(
        (pl.col("day") - pl.col("adoption_day")).dt.total_days().alias("dd")
    ).filter(pl.col("dd").abs() <= W)
    cw = win.group_by("adoption_day").agg(
        ctrlw_pre=pl.col("cs").filter(pl.col("dd") < 0).median(),
        ctrlw_post=pl.col("cs").filter(pl.col("dd") >= 0).median(),
    )
    ev = ev.join(cw, on="adoption_day", how="left")

    return ev.with_columns(
        treated_chg=pl.col("s_post") - pl.col("s_pre"),
        control_chg=pl.col("ctrl_post") - pl.col("ctrl_pre"),
        control_chg_w=pl.col("ctrlw_post") - pl.col("ctrlw_pre"),
    ).with_columns(
        did=pl.col("treated_chg") - pl.col("control_chg"),
        did_w=pl.col("treated_chg") - pl.col("control_chg_w"),
        dlog_s=(pl.col("s_post") / pl.col("s_pre")).log(),
        dlog_u=(pl.col("u_post") / pl.col("u_pre")).log(),
        at_floor=pl.col("s_pre") <= FLOOR * 1.05,
    )

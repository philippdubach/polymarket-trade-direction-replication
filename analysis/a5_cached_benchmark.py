"""Cached-only, full-history and common-sample trade-sign benchmark.

Never calls print extraction or RPC. Each UTC day starts a new history. The
stable cache row order resolves equal timestamps computationally, not as venue
event order. BVC is retrospective (bar closing price and whole-day fallback
volatility). See JFM_ANALYSIS_CONTRACT.md for the estimands and limitations.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

import a0_onchain_load as ld
from a1_fill_match import join_by_hash
from a2_direction_rules import LAG_GRID_MS, bvc, metrics, sign_rules
from d0_common import EXT, HERE, _versions, git_head

DAYS = ("2026-04-26", "2026-04-27", "2026-04-29", "2026-04-30",
        "2026-05-13", "2026-05-31", "2026-06-05", "2026-06-09",
        "2026-07-01", "2026-07-07", "2026-07-17", "2026-08-05")
RULES = ("lr", "emo", "tick", "bvc_5s")
ROW = "cache_row_id"


def clean_json(value):
    """Strict JSON: undefined one-class metrics become null, never NaN."""
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def cache_files(day: str, ext: Path = EXT) -> tuple[Path, list[Path]]:
    """Fail closed before any calculations if a required derived cache is absent."""
    dt.date.fromisoformat(day)
    prints = ext / "prints" / f"day={day}.parquet"
    slices = sorted((ext / "onchain" / f"day={day}").glob("blocks_*.parquet"))
    if not prints.is_file() or not slices:
        raise FileNotFoundError(f"required derived caches missing for {day}: {prints}, cached on-chain slices")
    return prints, slices


def stable_signs(df: pl.DataFrame) -> pl.DataFrame:
    """Reuse legacy algorithms with an explicit, stable tie-breaking cache row id.

    sign_rules only sorts its ts column: a temporary struct supplies its exact
    (timestamp, row id) order. BVC receives deterministic bar closing prices and
    summed volume, which is algebraically identical to its own bar aggregation.
    """
    d = df if ROW in df.columns else df.with_row_index(ROW)
    if d[ROW].n_unique() != d.height:
        raise ValueError("cache row ids must be unique")
    d = d.sort(["asset_id", "ts", ROW], maintain_order=True)
    d = sign_rules(d.rename({"ts": "_actual_ts"}).with_columns(
        pl.struct("_actual_ts", ROW).alias("ts"))).drop("ts").rename({"_actual_ts": "ts"})
    bars = (d.with_columns(pl.col("ts").dt.truncate("5s").alias("bar"))
            .group_by(["asset_id", "bar"]).agg(pl.col("price").last(), pl.col("size").sum())
            .rename({"bar": "ts"}))
    bc = bvc(bars).select("asset_id", "bar", "buy_share_bvc")
    d = (d.with_columns(pl.col("ts").dt.truncate("5s").alias("bar"))
         .join(bc, on=["asset_id", "bar"], how="left", validate="m:1")
         .with_columns(pl.when(pl.col("buy_share_bvc") > .5).then(1)
                       .when(pl.col("buy_share_bvc") < .5).then(-1)
                       .otherwise(None).cast(pl.Int64).alias("bvc_5s")))
    return d.drop("bar").sort(["asset_id", "ts", ROW], maintain_order=True)


def assemble_matched(prints: pl.DataFrame, taker: pl.DataFrame,
                     history: str = "full") -> tuple[pl.DataFrame, dict]:
    """Pure assembly API; returns matched rows with all signs and a truth audit."""
    if history not in ("full", "selected"):
        raise ValueError("history must be full or selected")
    if ROW not in prints.columns:
        prints = prints.with_row_index(ROW)
    prints = prints.with_columns((pl.len().over(["asset_id", "ts"]) > 1).alias("tied_print"))
    # A hash's truth cannot depend on which duplicate happens to be selected.
    truth_groups = taker.group_by("tx_hash").agg(
        pl.col("side").n_unique().alias("sides"), pl.col("token_id").n_unique().alias("tokens"))
    if truth_groups.filter((pl.col("sides") > 1) | (pl.col("tokens") > 1)).height:
        raise ValueError("ambiguous taker side/token for a transaction hash")
    duplicate_taker_hash_rows = taker.height - truth_groups.height
    taker = taker.sort(["tx_hash", "ts"]).unique("tx_hash", keep="first", maintain_order=True)
    if history == "full":
        joined = join_by_hash(stable_signs(prints), taker)
        d = joined.filter(pl.col("matched"))
    else:
        joined = join_by_hash(prints, taker)
        d = stable_signs(joined.filter(pl.col("matched")))
    d = d.rename({"chain_side": "taker_side"}).sort(ROW)
    audit = {"cached_prints": prints.height, "matched_prints": d.height,
             "matched_share": d.height / prints.height if prints.height else None,
             "cached_tied_prints": int(prints["tied_print"].sum()),
             "matched_tied_prints": int(d["tied_print"].sum()),
             "token_equal": int((d["asset_id"] == d["chain_token"]).sum()),
             "side_equal": int(((d["side"] == "BUY") == (d["taker_side"] == 0)).sum()),
             "duplicate_taker_hash_rows": duplicate_taker_hash_rows,
             "matched_repeated_hash_rows": d.height - d["transaction_hash"].n_unique()}
    if audit["token_equal"] != d.height or audit["side_equal"] != d.height:
        raise ValueError(f"settlement identity audit failed: {audit}")
    return d, audit


def load_signed_matched(day: str, *, history: str = "full", ext: Path = EXT) -> tuple[pl.DataFrame, dict]:
    """Stage 3 API: one cached day, full-column matched frame + identity audit.

    Retains bid/ask, quote_ts, mid_fwd, lag quotes, size, stable row id and tie flag.
    Does not persist frames. Caller should release each day before loading next.
    """
    path, _ = cache_files(day, ext)
    previous_out = ld.OUT
    try:
        ld.OUT = ext / "onchain"
        return assemble_matched(pl.read_parquet(path), ld.taker_legs(day), history)
    finally:
        ld.OUT = previous_out
        ld._CACHE.clear()


def common_mask(df: pl.DataFrame, rules=RULES) -> pl.Expr:
    return pl.all_horizontal([pl.col(r).is_in([-1, 1]).fill_null(False) for r in rules])


def score(df: pl.DataFrame, column: str, denominator: int | None = None) -> dict:
    out = metrics(df["taker_side"], df[column])
    n = out["n"]
    valid = df.filter(pl.col(column).is_not_null())
    out.update({"candidate_rows": df.height,
                "truth_buys": int((valid["taker_side"] == 0).sum()),
                "truth_sells": int((valid["taker_side"] == 1).sum()),
                "coverage_of_matched": n / denominator if denominator else (n / df.height if df.height else 0.0)})
    return clean_json(out)


def history_scores(df: pl.DataFrame) -> dict:
    common = df.filter(common_mask(df))
    return {"matched_rows": df.height, "common_rows": common.height,
            "common_coverage": common.height / df.height if df.height else 0.0,
            "own_sample": {r: score(df, r, df.height) for r in RULES},
            "common_sample": {r: score(common, r, df.height) for r in RULES}}


def paired_history(full: pl.DataFrame, selected: pl.DataFrame) -> dict:
    pair = full.select(ROW, "taker_side", *RULES).join(
        selected.select(ROW, *[pl.col(r).alias(f"{r}_selected") for r in RULES]),
        on=ROW, how="inner", validate="1:1")
    if pair.height != full.height or pair.height != selected.height:
        raise ValueError("history comparison does not cover identical matched row ids")
    out = {}
    for r in RULES:
        sub = pair.filter(pl.col(r).is_not_null() & pl.col(f"{r}_selected").is_not_null())
        fm, sm = score(sub, r), score(sub, f"{r}_selected")
        out[r] = {"paired_signed_rows": sub.height,
                  "sign_changed": int((sub[r] != sub[f"{r}_selected"]).sum()),
                  "sign_changed_share": float((sub[r] != sub[f"{r}_selected"]).mean()) if sub.height else None,
                  "full_only_signed": pair.filter(pl.col(r).is_not_null() & pl.col(f"{r}_selected").is_null()).height,
                  "selected_only_signed": pair.filter(pl.col(r).is_null() & pl.col(f"{r}_selected").is_not_null()).height,
                  "full_history": fm, "selected_history": sm,
                  "balanced_accuracy_difference_full_minus_selected":
                      fm.get("balanced_accuracy") - sm.get("balanced_accuracy")
                      if fm.get("balanced_accuracy") is not None and sm.get("balanced_accuracy") is not None else None}
    return out


def lag_sensitivity(df: pl.DataFrame) -> dict:
    cols = [f"lr_lag{lag}" for lag in LAG_GRID_MS]
    if not all(f"{q}_lag{lag}" in df.columns for q in ("bid", "ask") for lag in LAG_GRID_MS):
        return {"available": False}
    for lag in LAG_GRID_MS:
        bid, ask = pl.col(f"bid_lag{lag}"), pl.col(f"ask_lag{lag}")
        mid = (bid + ask) / 2
        valid = bid.is_not_null() & ask.is_not_null() & (ask >= bid)
        df = df.with_columns(pl.when(valid & (pl.col("price") > mid)).then(1)
                             .when(valid & (pl.col("price") < mid)).then(-1)
                             .otherwise(pl.col("tick")).alias(f"lr_lag{lag}"))
    common = df.filter(common_mask(df, ("lr", *cols)))
    return {"available": True, "fixed_common_rows": common.height,
            "coverage_of_matched": common.height / df.height if df.height else 0,
            "scores": {r: score(common, r, df.height) for r in ("lr", *cols)}}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_day(day: str, ext: Path = EXT) -> dict:
    print_file, slices = cache_files(day, ext)
    prints = pl.read_parquet(print_file)
    previous_out = ld.OUT
    try:
        ld.OUT = ext / "onchain"
        taker = ld.taker_legs(day)
        full, audit = assemble_matched(prints, taker, "full")
        selected, selected_audit = assemble_matched(prints, taker, "selected")
        out = {"day": day, "audit": audit, "full_history": history_scores(full),
               "selected_history": history_scores(selected),
               "paired_history": paired_history(full, selected),
               "tie_exclusion_full_history": history_scores(full.filter(~pl.col("tied_print"))),
               "unique_hash_full_history": history_scores(full.unique("transaction_hash", keep="first", maintain_order=True)),
               "lr_lag_sensitivity_full_history": lag_sensitivity(full),
               "provenance": {"prints": {"path": str(print_file), "bytes": print_file.stat().st_size,
                                          "sha256": sha256(print_file)},
                              "onchain_slices": [{"path": str(p), "bytes": p.stat().st_size,
                                                   "sha256": sha256(p)} for p in slices]}}
        assert audit == selected_audit
        return clean_json(out)
    finally:
        ld.OUT = previous_out
        ld._CACHE.clear()


def describe(values: list[float]) -> dict:
    return {"days": len(values), "equal_day_mean": float(np.mean(values)) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def summarize(days: list[dict]) -> dict:
    out = {"days": len(days), "full_history": {}, "selected_history": {},
           "full_common_paired_contrasts": {}, "paired_history": {}}
    for history in ("full_history", "selected_history"):
        for sample in ("own_sample", "common_sample"):
            out[history][sample] = {r: {m: describe([d[history][sample][r][m] for d in days
                                                    if d[history][sample][r].get(m) is not None])
                                          for m in ("balanced_accuracy", "accuracy", "mcc", "coverage_of_matched")}
                                   for r in RULES}
    for a, b in (("lr", "tick"), ("lr", "bvc_5s"), ("emo", "lr")):
        vals = [d["full_history"]["common_sample"][a]["balanced_accuracy"] -
                d["full_history"]["common_sample"][b]["balanced_accuracy"] for d in days]
        out["full_common_paired_contrasts"][f"{a}_minus_{b}"] = describe(vals) | {
            "positive_days": sum(v > 0 for v in vals), "zero_days": sum(v == 0 for v in vals),
            "daily_differences": dict(zip((d["day"] for d in days), vals))}
    for r in RULES:
        out["paired_history"][r] = describe([d["paired_history"][r][
            "balanced_accuracy_difference_full_minus_selected"] for d in days])
    return clean_json(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("days", nargs="*", default=list(DAYS))
    parser.add_argument("--output", type=Path, default=HERE / "a5_cached_benchmark.json")
    args = parser.parse_args()
    for day in args.days:
        cache_files(day)
    results = []
    for day in args.days:
        result = run_day(day)
        results.append(result)
        common = result["full_history"]["common_sample"]
        print(day, "matched", result["audit"]["matched_prints"], "common",
              result["full_history"]["common_rows"], {r: round(common[r]["balanced_accuracy"], 6) for r in RULES}, flush=True)
        gc.collect()
    payload = {"_meta": {"script": Path(__file__).name, "script_sha256": sha256(Path(__file__)),
                          "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                          "git_head": git_head(), "versions": _versions()},
               "protocol": {"rules": RULES, "history_order": ["asset_id", "ts", ROW],
                            "bvc_bar": "5s", "bvc_is_retrospective": True,
                            "tie_sensitivity": "exclude tied rows from scoring, retain all rows in construction",
                            "statistics": "equal-day descriptive summaries; selected dependent dates, no population test",
                            "history_initialization": "restart at each UTC day; no prior-day history"},
               "days": results, "summary": summarize(results)}
    args.output.write_text(json.dumps(clean_json(payload), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

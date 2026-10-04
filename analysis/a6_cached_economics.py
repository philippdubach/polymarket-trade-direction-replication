"""Paired signed-measure accounting on existing full-history daily caches.

No extraction, RPC or external-drive writes. All classifiers and settlement
truth share exactly the same eligible rows and weights. Forward quote age and
book state are not retained by the cache: these are delivered-quote accounting
quantities, not causal price impact or private-information estimates.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from a5_cached_benchmark import (DAYS, RULES, ROW, cache_files, clean_json,
                                 describe, load_signed_matched, sha256)
from d0_common import EXT, HERE, _versions, git_head

MEASURES = ("effective", "realised", "impact")
WEIGHTS = ("fill", "share_volume")
MIN_ASSET_FILLS = 50
IDENTITY_TOLERANCE = 1e-12
SIGN_ZERO_TOLERANCE = 1e-12
RANK_DECIMALS = 12


def finite_probability(column: str) -> pl.Expr:
    x = pl.col(column)
    return (x.is_not_null() & x.is_finite() & x.is_between(0., 1., closed="both")).fill_null(False)


def eligible_sample(df: pl.DataFrame, day: str) -> tuple[pl.DataFrame, list[dict]]:
    """Apply one sequential funnel, never rule-specific eligibility.

    The UTC endpoint is exclusive. A print at exactly 23:55 would need a quote
    at 00:00 in the next day's archive and is therefore excluded as well.
    """
    start = dt.datetime.fromisoformat(day)
    cutoff = start + dt.timedelta(days=1, minutes=-5)
    checks = (
        ("all_four_rule_signs_and_valid_truth", pl.all_horizontal([
            pl.col(r).is_in([-1, 1]).fill_null(False) for r in RULES]) &
            pl.col("taker_side").is_in([0, 1]).fill_null(False)),
        ("finite_print_price_in_probability_support", finite_probability("price")),
        ("strictly_interior_uncrossed_current_quotes",
         (pl.col("bid").is_finite() & pl.col("ask").is_finite() &
          (pl.col("bid") > 0) & (pl.col("bid") <= pl.col("ask")) &
          (pl.col("ask") < 1)).fill_null(False)),
        ("finite_forward_midpoint_in_probability_support", finite_probability("mid_fwd")),
        ("positive_finite_share_size", (pl.col("size").is_finite() & (pl.col("size") > 0)).fill_null(False)),
        ("timestamp_within_day_and_strictly_before_2355",
         ((pl.col("ts") >= start) & (pl.col("ts") < cutoff)).fill_null(False)),
    )
    out = df
    funnel = [{"step": "settlement_matched", "before": df.height, "after": df.height, "excluded": 0}]
    for name, mask in checks:
        before = out.height
        out = out.filter(mask)
        funnel.append({"step": name, "before": before, "after": out.height, "excluded": before - out.height})
    # Reconstruct the delivered current midpoint from the audited current book.
    out = out.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
    return out, funnel


def mean_sign(values):
    """Numerical cancellations within 1e-12 price units count as zero.

    Keep raw accounting means. This threshold prevents floating-point residue
    from being reported as an economic sign reversal, rather than defining an
    economically material minimum effect.
    """
    values = np.asarray(values)
    return np.where(np.abs(values) <= SIGN_ZERO_TOLERANCE, 0, np.sign(values))


def opposite_sign(x: float, y: float) -> bool:
    return bool(mean_sign(x) * mean_sign(y) < 0)


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float | None:
    """Descriptive Pearson correlation under the same normalized weights."""
    # Test exact constant inputs before weighted centering: the normalized
    # weights can sum to 1 +/- floating residue and otherwise create a false
    # variance for an identically-one error indicator or constant outcome.
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    dx, dy = x - np.dot(w, x), y - np.dot(w, y)
    vx, vy = np.dot(w, dx * dx), np.dot(w, dy * dy)
    return float(np.dot(w, dx * dy) / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else None


def paired_accounting(df: pl.DataFrame, *, price_column: str = "price") -> dict:
    """All signs, measures and weights share the input frame; fail on invalids.

    Caller must select eligibility first. Algebra is checked on every row and
    again after each weighting. Signed differences preserve over/understatement;
    absolute paired distances report magnitude without cancellation.
    """
    required = [price_column, "mid", "mid_fwd", "size", "taker_side", *RULES]
    if not all(c in df.columns for c in required):
        raise ValueError("paired accounting lacks required columns")
    valid = pl.all_horizontal([pl.col(c).is_finite().fill_null(False)
                               for c in (price_column, "mid", "mid_fwd", "size")])
    valid &= (pl.col("size") > 0) & pl.col("taker_side").is_in([0, 1]).fill_null(False)
    valid &= pl.all_horizontal([pl.col(r).is_in([-1, 1]).fill_null(False) for r in RULES])
    if df.filter(~valid).height:
        raise ValueError("paired accounting input is not a common valid sample")
    n = df.height
    sample = {"rows": n, "assets": df["asset_id"].n_unique(),
              "truth_buys": int((df["taker_side"] == 0).sum()),
              "truth_sells": int((df["taker_side"] == 1).sum()),
              "total_shares": float(df["size"].sum()),
              "distinct_transaction_hashes": df["transaction_hash"].n_unique(),
              "tied_prints": int(df["tied_print"].sum())}
    if not n:
        return {"sample": sample, "price_column": price_column, "weights": {},
                "identity_max_abs_error_per_fill": None}
    price, mid, fwd = (df[c].to_numpy() for c in (price_column, "mid", "mid_fwd"))
    unsigned = {"effective": 2 * (price - mid), "realised": 2 * (price - fwd),
                "impact": 2 * (fwd - mid)}
    signs = {"taker": np.where(df["taker_side"].to_numpy() == 0, 1, -1)}
    signs.update({r: df[r].to_numpy() for r in RULES})
    truth = {m: signs["taker"] * unsigned[m] for m in MEASURES}
    maximum = float(np.max(np.abs(unsigned["effective"] - unsigned["realised"] - unsigned["impact"])))
    if maximum > IDENTITY_TOLERANCE:
        raise AssertionError("per-fill effective != realised + impact")
    out = {"sample": sample, "price_column": price_column,
           "identity_max_abs_error_per_fill": maximum, "weights": {}}
    for weight in WEIGHTS:
        raw = np.ones(n) if weight == "fill" else df["size"].to_numpy()
        norm = raw / raw.sum()
        tm = {m: float(np.dot(norm, truth[m])) for m in MEASURES}
        panel = {"weight_sum": float(raw.sum()), "taker": tm, "rules": {}}
        for r in RULES:
            errors = (signs[r] != signs["taker"])
            means, differences, absolute_distance, diagnostics = {}, {}, {}, {}
            max_flip = 0.
            for m in MEASURES:
                pred = signs[r] * unsigned[m]
                delta = pred - truth[m]
                algebra = -2 * errors * truth[m]
                max_flip = max(max_flip, float(np.max(np.abs(delta - algebra))))
                means[m] = float(np.dot(norm, pred))
                differences[m] = means[m] - tm[m]
                mass = float(np.dot(norm, errors * truth[m]))
                absolute_mass = float(np.dot(norm, errors * np.abs(truth[m])))
                if abs(differences[m] + 2 * mass) > IDENTITY_TOLERANCE:
                    raise AssertionError("weighted flip distortion identity failed")
                absolute_distance[m] = float(np.dot(norm, np.abs(delta)))
                diagnostics[m] = {
                    "truth_signed_error_mass": mass,
                    "truth_absolute_error_mass": absolute_mass,
                    "error_truth_measure_correlation": weighted_corr(errors.astype(float), truth[m], norm),
                    "opposite_sign_of_taker_mean": opposite_sign(means[m], tm[m]),
                    "mean_sign_changed": bool(mean_sign(means[m]) != mean_sign(tm[m])),
                }
            mean_identity = abs(means["effective"] - means["realised"] - means["impact"])
            if max(max_flip, mean_identity) > IDENTITY_TOLERANCE:
                raise AssertionError("signed or mean accounting identity failed")
            panel["rules"][r] = {"means": means, "difference_rule_minus_taker": differences,
                                    "mean_absolute_paired_distance": absolute_distance,
                                    "flip_rows": int(errors.sum()), "flip_share": float(np.dot(norm, errors)),
                                    "error_diagnostics": diagnostics,
                                    "flip_identity_max_abs_error": max_flip,
                                    "mean_accounting_identity_abs_error": mean_identity}
        panel["taker_accounting_identity_abs_error"] = abs(tm["effective"] - tm["realised"] - tm["impact"])
        if panel["taker_accounting_identity_abs_error"] > IDENTITY_TOLERANCE:
            raise AssertionError("taker mean accounting identity failed")
        out["weights"][weight] = panel
    return clean_json(out)


def asset_exploratory(df: pl.DataFrame, minimum: int = MIN_ASSET_FILLS) -> dict:
    """Daily asset means on common fills/assets; no independent-asset inference."""
    names = ("taker", *RULES)
    sub = df.with_columns(pl.when(pl.col("taker_side") == 0).then(1).otherwise(-1).alias("taker"))
    raw = {"effective": 2 * (pl.col("price") - pl.col("mid")),
           "realised": 2 * (pl.col("price") - pl.col("mid_fwd")),
           "impact": 2 * (pl.col("mid_fwd") - pl.col("mid"))}
    aggregates = [pl.len().alias("fills"), pl.col("size").sum().alias("shares")]
    for name in names:
        for m in MEASURES:
            signed = pl.col(name) * raw[m]
            aggregates.extend([signed.mean().alias(f"fill_{name}_{m}"),
                               ((signed * pl.col("size")).sum() / pl.col("size").sum()).alias(f"share_volume_{name}_{m}")])
    assets = sub.group_by("asset_id").agg(*aggregates).filter(pl.col("fills") >= minimum)
    out = {"minimum_common_fills_per_asset": minimum, "assets": assets.height,
           "common_fills_in_retained_assets": int(assets["fills"].sum()),
           "weights": {}, "interpretation": "daily asset means; assets share markets/events; descriptive, no independent-asset inference"}
    for weight in WEIGHTS:
        out["weights"][weight] = {}
        for r in RULES:
            out["weights"][weight][r] = {}
            for m in MEASURES:
                truth = assets[f"{weight}_taker_{m}"].to_numpy()
                pred = assets[f"{weight}_{r}_{m}"].to_numpy()
                # Restore numerical ties obscured by summation residue, while
                # retaining raw accounting means. This precision convention is
                # not an economic materiality cutoff. Constant rank inputs have
                # no defined correlation, including the one-asset case.
                truth_rank, pred_rank = np.round(truth, RANK_DECIMALS), np.round(pred, RANK_DECIMALS)
                rho = float(spearmanr(truth_rank, pred_rank).statistic) if (assets.height >= 2 and
                       np.ptp(truth_rank) > 0 and np.ptp(pred_rank) > 0) else None
                truth_sign, pred_sign = mean_sign(truth), mean_sign(pred)
                reversals = int(np.count_nonzero(truth_sign * pred_sign < 0))
                changed = int(np.count_nonzero(truth_sign != pred_sign))
                out["weights"][weight][r][m] = {
                    "spearman_rank_correlation": rho, "sign_reversal_assets": reversals,
                    "sign_reversal_share": reversals / assets.height if assets.height else None,
                    "mean_sign_changed_assets": changed,
                    "taker_zero_mean_assets": int(np.count_nonzero(truth_sign == 0)),
                    "rule_zero_mean_assets": int(np.count_nonzero(pred_sign == 0)),
                }
    return clean_json(out)


def analyze_matched(df: pl.DataFrame, day: str) -> dict:
    """Pure analysis entry point for tests and cached daily runs."""
    primary, funnel = eligible_sample(df, day)
    accounting = paired_accounting(primary)
    accounting["funnel"] = funnel
    accounting["asset_exploratory"] = asset_exploratory(primary)
    sensitivities = {}
    if "quote_ts" in primary.columns:
        age_ms = (pl.col("ts") - pl.col("quote_ts")).dt.total_microseconds() / 1000
        fresh = primary.filter(((age_ms >= 0) & (age_ms <= 1000)).fill_null(False))
        sensitivities["quote_age_1s"] = paired_accounting(fresh) | {
            "available": True, "selection": "primary rows with 0 <= delivered current quote age <= 1000 ms"}
    else:
        sensitivities["quote_age_1s"] = {"available": False}
    sensitivities["tie_exclusion"] = paired_accounting(primary.filter(~pl.col("tied_print"))) | {
        "selection": "exclude tied primary scoring rows; full signing history retained"}
    sensitivities["unique_hash"] = paired_accounting(primary.sort(ROW).unique(
        "transaction_hash", keep="first", maintain_order=True)) | {
        "selection": "first cache-row-id among primary eligible rows for each transaction hash; full signing history retained"}
    if "chain_price" in primary.columns:
        valid_chain = primary.filter(finite_probability("chain_price"))
        # Price substitution and its printed-price comparator use exactly the
        # same subset, signs, posted current/future midpoints and print volumes.
        sensitivities["chain_price"] = paired_accounting(valid_chain, price_column="chain_price") | {
            "available": True, "selection": "primary rows with valid chain_price; fixed signs and print share weights",
            "print_price_comparator": paired_accounting(valid_chain),
            "print_minus_chain_price": {"mean": float((valid_chain["price"] - valid_chain["chain_price"]).mean()) if valid_chain.height else None,
                                         "max_absolute": float((valid_chain["price"] - valid_chain["chain_price"]).abs().max()) if valid_chain.height else None}}
    else:
        sensitivities["chain_price"] = {"available": False}
    return {"day": day, "primary": accounting, "sensitivities": sensitivities}


def verify_inputs(day: str, ext: Path, stage2: dict) -> dict:
    """Rehash every input and reject a change from the accepted Stage 2 inputs."""
    prints, slices = cache_files(day, ext)
    current = {"prints": {"path": str(prints), "bytes": prints.stat().st_size, "sha256": sha256(prints)},
               "onchain_slices": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in slices]}
    if current != stage2:
        raise ValueError(f"cached inputs changed since Stage 2 for {day}")
    return {"verified_against_stage2": True, "prints_sha256": current["prints"]["sha256"],
            "onchain_slices": len(slices)}


def run_day(day: str, stage2: dict, ext: Path = EXT) -> dict:
    provenance = verify_inputs(day, ext, stage2["provenance"])
    df, audit = load_signed_matched(day, history="full", ext=ext)
    if audit != stage2["audit"]:
        raise ValueError(f"matched audit changed since Stage 2 for {day}")
    out = analyze_matched(df, day)
    out.update({"audit": audit, "provenance": provenance})
    return clean_json(out)


def summarize_panel(panels: list[dict], day_names: list[str]) -> dict:
    valid = [(p, day) for p, day in zip(panels, day_names) if p.get("weights")]
    out = {"days": len(valid), "eligible_rows_total": sum(p["sample"]["rows"] for p, _ in valid),
           "sample_rows": describe([p["sample"]["rows"] for p, _ in valid]), "weights": {}}
    for weight in WEIGHTS:
        taker = {m: describe([p["weights"][weight]["taker"][m] for p, _ in valid]) for m in MEASURES}
        rules = {}
        for r in RULES:
            means, differences, distances, sign_reversals, error_diagnostics = {}, {}, {}, {}, {}
            for m in MEASURES:
                rv = [p["weights"][weight]["rules"][r] for p, _ in valid]
                means[m] = describe([v["means"][m] for v in rv])
                dif = [v["difference_rule_minus_taker"][m] for v in rv]
                differences[m] = describe(dif) | {"daily": dict(zip([day for _, day in valid], dif)),
                                                "positive_days": sum(v > SIGN_ZERO_TOLERANCE for v in dif),
                                                "negative_days": sum(v < -SIGN_ZERO_TOLERANCE for v in dif),
                                                "numerical_zero_days": sum(abs(v) <= SIGN_ZERO_TOLERANCE for v in dif)}
                distances[m] = describe([v["mean_absolute_paired_distance"][m] for v in rv])
                sign_reversals[m] = {"opposite_sign_days": sum(v["error_diagnostics"][m]["opposite_sign_of_taker_mean"] for v in rv),
                                     "mean_sign_changed_days": sum(v["error_diagnostics"][m]["mean_sign_changed"] for v in rv)}
                error_diagnostics[m] = {key: describe([v["error_diagnostics"][m][key] for v in rv
                                                      if v["error_diagnostics"][m][key] is not None])
                                         for key in ("truth_signed_error_mass", "truth_absolute_error_mass", "error_truth_measure_correlation")}
            rules[r] = {"means": means, "differences": differences, "mean_absolute_paired_distance": distances,
                        "flip_share": describe([p["weights"][weight]["rules"][r]["flip_share"] for p, _ in valid]),
                        "sign_reversals": sign_reversals, "error_diagnostics": error_diagnostics}
        out["weights"][weight] = {"taker": taker, "rules": rules}
    return clean_json(out)


def summarize(days: list[dict]) -> dict:
    names = [d["day"] for d in days]
    out = {"days": len(days), "primary": summarize_panel([d["primary"] for d in days], names), "sensitivities": {}}
    for s in ("quote_age_1s", "tie_exclusion", "unique_hash", "chain_price"):
        out["sensitivities"][s] = summarize_panel([d["sensitivities"][s] for d in days], names)
    out["asset_exploratory"] = {
        "minimum_common_fills_per_asset": MIN_ASSET_FILLS,
        "asset_day_count": sum(d["primary"]["asset_exploratory"]["assets"] for d in days),
        "daily_asset_counts": describe([d["primary"]["asset_exploratory"]["assets"] for d in days]),
        "weights": {}}
    for w in WEIGHTS:
        out["asset_exploratory"]["weights"][w] = {}
        for r in RULES:
            out["asset_exploratory"]["weights"][w][r] = {}
            for m in MEASURES:
                records = [d["primary"]["asset_exploratory"]["weights"][w][r][m] for d in days]
                out["asset_exploratory"]["weights"][w][r][m] = {
                    k: describe([v[k] for v in records if v[k] is not None])
                    for k in ("spearman_rank_correlation", "sign_reversal_share")}
    return clean_json(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("days", nargs="*", default=list(DAYS))
    parser.add_argument("--output", type=Path, default=HERE / "a6_cached_economics.json")
    parser.add_argument("--stage2", type=Path, default=HERE / "a5_cached_benchmark.json")
    args = parser.parse_args()
    stage2 = json.loads(args.stage2.read_text())
    if stage2["_meta"]["script_sha256"] != sha256(HERE / "a5_cached_benchmark.py"):
        raise ValueError("Stage 2 implementation changed since its accepted result")
    byday = {d["day"]: d for d in stage2["days"]}
    for day in args.days:
        cache_files(day)
        if day not in byday:
            raise ValueError(f"day {day} absent from accepted Stage 2 result")
    results = []
    for day in args.days:
        result = run_day(day, byday[day])
        results.append(result)
        p = result["primary"]
        print(day, "eligible", p["sample"]["rows"], "fill taker", p["weights"].get("fill", {}).get("taker"), flush=True)
        gc.collect()
    payload = {"_meta": {"script": Path(__file__).name, "script_sha256": sha256(Path(__file__)),
                          "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                          "git_head": git_head(), "versions": _versions(),
                          "stage2_json": str(args.stage2), "stage2_json_sha256": sha256(args.stage2),
                          "stage2_script_sha256": sha256(HERE / "a5_cached_benchmark.py")},
               "protocol": {"rules": RULES, "history": "full cached tape signed before settlement selection",
                            "units": "probability price per share; multiply 100 for cents per share",
                            "weights": WEIGHTS, "horizon_seconds": 300,
                            "day_boundary": "UTC day start <= ts < next UTC day start minus 5 minutes",
                            "current_quotes": "finite and 0 < bid <= ask < 1",
                            "price_support": "finite print price and forward midpoint in [0,1]",
                            "common_rows": "all four rules and truth signed, finite positive print share size",
                            "economic_meaning": "delivered-quote signed accounting, not causal impact/private information or net profitability",
                            "forward_quote_limitation": "forward quote timestamp/bid/ask absent; age, crossing and two-sided state unverified; cutoff cannot repair internal outages",
                            "statistics": "equal-day descriptive mean/range on selected dependent dates; daily asset ranks have no independent-asset inference",
                            "asset_minimum": MIN_ASSET_FILLS,
                            "identity_tolerance": IDENTITY_TOLERANCE,
                            "sign_zero_tolerance": SIGN_ZERO_TOLERANCE,
                            "sign_zero_definition": "abs(mean)<=1e-12 price units counts as numerical zero for sign diagnostics; raw means retained",
                            "rank_decimals": RANK_DECIMALS,
                            "rank_precision_definition": "round asset means to 12 decimal places only for rank-input ties; raw accounting means retained; no economic materiality threshold"},
               "days": results, "summary": summarize(results)}
    args.output.write_text(json.dumps(clean_json(payload), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

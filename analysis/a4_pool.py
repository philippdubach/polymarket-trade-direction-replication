"""Pool the per-day direction results. The honest cross-day unit is the day.

Reads every a2_direction_<day>.json, a1_fill_match_<day>.json and
a3_signed_<day>.json and reports per-day values, their mean, min, max and
a day-level standard error, plus three pre-specified paired contrasts on
balanced accuracy with Holm adjustment:

  lr vs tick, lr vs bvc_5s, lr_fresh_5s vs lr

No pooled "N million fills" interval appears anywhere; sampling uncertainty
within a day is in each day's asset-cluster bootstrap.

Run:  uv run python a4_pool.py
Out:  a4_direction_pooled.json
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

from d0_common import HERE, write_json

CONTRASTS = [("lr", "tick"), ("lr", "bvc_5s"), ("emo", "lr")]


def _t_p(t: float, df: int) -> float:
    """Two-sided p from a t statistic; scipy if present, else a normal approximation."""
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df))
    except Exception:  # noqa: BLE001
        return float(math.erfc(abs(t) / math.sqrt(2)))


def summarise(days: dict, rules: list[str], metric: str) -> dict:
    out = {}
    for r in rules:
        vals = {d: v["rules"][r][metric] for d, v in days.items() if r in v.get("rules", {}) and v["rules"][r].get(metric) is not None}
        xs = list(vals.values())
        n = len(xs)
        if n == 0:
            out[r] = {"days": 0}
            continue
        mean = sum(xs) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
        out[r] = {"days": n, "mean": mean, "min": min(xs), "max": max(xs), "sd": sd, "se": sd / math.sqrt(n) if n > 1 else float("nan"),
                  "by_day": vals}
    return out


def contrast(days: dict, a: str, b: str, metric: str) -> dict:
    diffs = [v["rules"][a][metric] - v["rules"][b][metric] for v in days.values()
             if a in v.get("rules", {}) and b in v.get("rules", {})]
    n = len(diffs)
    if n < 2:
        return {"days": n, "mean_diff": diffs[0] if diffs else None, "t": None, "p": None}
    mean = sum(diffs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1))
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    return {"days": n, "mean_diff": mean, "sd": sd, "t": t, "p": _t_p(t, n - 1) if sd > 0 else 0.0,
            "min_diff": min(diffs), "max_diff": max(diffs), "sign_test": sign_test(diffs)}


def sign_test(diffs: list[float]) -> dict:
    """Two-sided sign test on paired day differences: how many days favour the first arm."""
    n = len(diffs)
    k = sum(1 for d in diffs if d > 0)
    p_ge = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    p_le = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"days": n, "positive": k, "p_two_sided": min(1.0, 2 * min(p_ge, p_le))}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = min(1.0, running)
    return adj


def ranges(per_day: dict) -> dict:
    """min, max and the days they fall on, for each numeric key of a {day: {key: value}} dict."""
    keys = sorted({k for v in per_day.values() for k, x in v.items() if isinstance(x, (int, float)) and not isinstance(x, bool)})
    out = {}
    for k in keys:
        vals = {d: v[k] for d, v in per_day.items() if isinstance(v.get(k), (int, float)) and not isinstance(v.get(k), bool)}
        if not vals:
            continue
        lo, hi = min(vals, key=vals.get), max(vals, key=vals.get)
        out[k] = {"min": vals[lo], "max": vals[hi], "argmin": lo, "argmax": hi, "days": len(vals)}
    return out


def _load(prefix: str) -> dict:
    out = {}
    for f in sorted(glob.glob(str(HERE / f"{prefix}_*.json"))):
        d = json.loads(Path(f).read_text())
        if "day" in d:
            out[d["day"]] = d
    return out


def main() -> None:
    a2 = _load("a2_direction")
    a1 = _load("a1_fill_match")
    a3 = _load("a3_signed")
    rules = sorted({r for v in a2.values() for r in v.get("rules", {})})
    res = {"days": sorted(a2), "n_days": len(a2)}
    res["balanced_accuracy"] = summarise(a2, rules, "balanced_accuracy")
    res["mcc"] = summarise(a2, rules, "mcc")
    res["recall_sell"] = summarise(a2, rules, "recall_sell")
    res["recall_buy"] = summarise(a2, rules, "recall_buy")
    res["kappa"] = summarise(a2, rules, "kappa")
    res["accuracy"] = summarise(a2, rules, "accuracy")
    res["majority_baseline"] = summarise(a2, rules, "majority_baseline")
    res["coverage_rule"] = summarise(a2, rules, "coverage")
    cons = {f"{a}_vs_{b}": contrast(a2, a, b, "balanced_accuracy") for a, b in CONTRASTS}
    adj = holm({k: v["p"] for k, v in cons.items() if v.get("p") is not None})
    for k in cons:
        cons[k]["p_holm"] = adj.get(k)
    res["contrasts"] = cons
    res["flag_vs_taker"] = {d: v.get("flag_vs_taker", {}).get("balanced_accuracy") for d, v in a2.items()}
    res["cell_test"] = {d: v.get("cell_test") for d, v in a2.items()}
    res["coverage"] = {d: {"print_matched_share": v["coverage"]["print_matched_share"], "taker_matched_share": v["coverage"]["taker_matched_share"],
                           "prints": v["coverage"]["prints"], "taker_legs": v["coverage"]["taker_legs"],
                           "taker_legs_within_day": v["coverage"].get("taker_legs_within_day"),
                           "taker_matched_share_within_day": v["coverage"].get("taker_matched_share_within_day"),
                           "hour0_share_of_unmatched_within_day": v["coverage"].get("hour0_share_of_unmatched_within_day"),
                           "duplicate_prints": v.get("duplicates", {}).get("duplicates"),
                           "unknown_txs": v.get("unknown_cause", {}).get("unknown_txs"),
                           "unknown_with_token_absent": v.get("unknown_cause", {}).get("with_token_absent_from_map")} for d, v in a1.items()}
    res["identity_by_type"] = {d: v.get("identity_by_type") for d, v in a1.items()}
    res["identity"] = {d: {k: v["identity"].get(k) for k in ("side_equal_share", "token_equal_share", "price_within_5e4_share", "size_within_1e6_share", "abs_dprice_p99")}
                       for d, v in a1.items()}
    res["match_types"] = {d: {t: x["tx_share"] for t, x in v["composition"]["types"].items()} for d, v in a1.items()}
    res["buy_share"] = {d: {"taker_legs": v["composition"]["taker_buy_share"], "all_legs": v["composition"]["all_legs_buy_share"]} for d, v in a1.items()}
    res["signed_ratios"] = {}
    for d, v in a3.items():
        m = v["measures"]
        res["signed_ratios"][d] = {r: m.get(r, {}).get("eff_ratio_to_taker") for r in ("lr", "tick", "emo")}
        res["signed_ratios"][d] |= {f"{r}_mean": m.get(r, {}).get("eff_mean_ratio_to_taker") for r in ("lr", "tick", "emo")}
        res["signed_ratios"][d]["wrong_side_share"] = v.get("wrong_side", {}).get("all", {}).get("share_negative")
        res["signed_ratios"][d]["taker_eff_mean"] = m.get("taker", {}).get("eff_mean_of_means")
        att = v.get("attenuation", {})
        if att.get("taker_mean_eff") and att.get("lr", {}).get("predicted_mean_eff") is not None:
            res["signed_ratios"][d]["lr_pred_over_taker"] = att["lr"]["predicted_mean_eff"] / att["taker_mean_eff"]
    res["per_day"] = {}
    for d, v in a1.items():
        res["per_day"][d] = {"fills": v["composition"]["all_legs"], "settled_txs": v["composition"]["txs"]}
    for d, v in a2.items():
        seg = v.get("segments", {})
        fresh = seg.get("staleness=<=1s", {}).get("n_rows")
        res["per_day"].setdefault(d, {})["quote_within_1s_share"] = (fresh / v["prints_matched"]) if fresh and v.get("prints_matched") else None
        res["per_day"][d]["prints_matched"] = v.get("prints_matched")
        res["per_day"][d]["lr_recall_buy"] = v["rules"]["lr"].get("recall_buy")
        res["per_day"][d]["lr_boot_lo"] = v["rules"]["lr"].get("bootstrap", {}).get("lo")
        res["per_day"][d]["lr_boot_hi"] = v["rules"]["lr"].get("bootstrap", {}).get("hi")
        crypto = seg.get("category=crypto", {}).get("n_rows")
        res["per_day"][d]["crypto_prints"] = crypto
        res["per_day"][d]["same_ms_quote_share"] = v.get("same_ms_quote_share")
        for lag, g in (v.get("lag_grid") or {}).items():
            res["per_day"][d][f"lr_bal_lag{lag}"] = g["lr"]["balanced_accuracy"]
    for d, v in a3.items():
        for lag, g in (v.get("lag_grid") or {}).items():
            res["per_day"].setdefault(d, {})[f"wrong_side_lag{lag}"] = g["wrong_side_share"]
            res["per_day"][d][f"lr_ratio_lag{lag}"] = g["lr"]["fill_level_ratio"]
        acc = v.get("accounting", {})
        for r, g in acc.items():
            res["per_day"].setdefault(d, {})[f"{r}_fill_ratio"] = g.get("fill_level_ratio")
            res["per_day"][d][f"{r}_flip_mass"] = g.get("flip_mass")
        res["per_day"][d]["wrong_side_chain_price"] = v.get("wrong_side", {}).get("all_chain_price", {}).get("share_negative")
    res["totals"] = {"matched_prints": sum(v["coverage"]["print_matched"] for v in a1.values()),
                     "taker_legs": sum(v["coverage"]["taker_legs"] for v in a1.values()),
                     "prints": sum(v["coverage"]["prints"] for v in a1.values())}
    res["receipts"] = {}
    for f in sorted(glob.glob(str(HERE / "a1_receipts_*.json"))):
        rj = json.loads(Path(f).read_text())
        res["receipts"][rj["day"]] = (rj.get("weighted_shares") or rj.get("shares", {})) | {"sampled": rj.get("sampled"),
                                     "control_settled": (rj.get("control_matched") or {}).get("classes", {}).get("in_range_missed")}
    res["fuzzy"] = {d: {"unmatched_prints": v["fuzzy"]["unmatched_prints"], "fuzzy_matched": v["fuzzy"]["fuzzy_matched"]} for d, v in a1.items()}
    res["ranges"] = {
        "coverage": ranges(res["coverage"]), "identity": ranges(res["identity"]), "match_types": ranges(res["match_types"]),
        "buy_share": ranges(res["buy_share"]), "signed_ratios": ranges(res["signed_ratios"]), "receipts": ranges(res["receipts"]),
        "fuzzy": ranges(res["fuzzy"]),
        "cell_all_legs": ranges({d: v["all_legs"] for d, v in res["cell_test"].items() if v}),
        "cell_taker_legs": ranges({d: v["taker_legs"] for d, v in res["cell_test"].items() if v}),
        "lr_by_day": ranges({d: {"balanced_accuracy": v["rules"]["lr"]["balanced_accuracy"], "assets": v.get("assets"), "n": v.get("prints_matched")} for d, v in a2.items()}),
        "per_day": ranges(res["per_day"]),
    }
    write_json("a4_direction_pooled", res, inputs=sorted(glob.glob(str(HERE / "a[123]_*.json"))))
    ba = res["balanced_accuracy"]
    print(f"days={len(a2)}  " + "  ".join(f"{r}: {ba[r]['mean']:.3f} [{ba[r]['min']:.3f}, {ba[r]['max']:.3f}]" for r in rules if ba[r].get("days")))
    for k, v in cons.items():
        print(f"  {k}: diff={v['mean_diff']}, p={v.get('p')}, holm={v.get('p_holm')}")


if __name__ == "__main__":
    main()

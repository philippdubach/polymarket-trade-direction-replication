"""Where does 28 April rank among every other date the panel allows?

For every valid date D (full panel coverage on D-7..D+7, outside the
degraded capture windows, more than seven days from any documented venue
change) the same cohort rule and paired change as b1_cutover.py are computed
with D as the event. The rank of 28 April among those dates, per outcome and
jointly, is the paper's inference about the cutover: a randomisation over
dates, not a control group.

Run:  uv run python b2_placebo_dates.py
Out:  b2_placebo.json
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from b1_cutover import OUTCOMES, cohort, half_life, paths
from d0_common import CUTOVER, EXT, VENUE_EVENTS, degraded_days, write_json

WINDOW, EXCLUSION = 7, 7   # a placebo window must not contain a documented venue change
CLEAN_SPAN, OUTAGE_LEN = (-1, 3), 3   # the ranked changes are k = 0..3 from k = -1; runs of 3+ flagged days void the window
JOINT = {"log_spread": True, "log_updates": False}   # cutover signature: spreads up, updates down


def outage_runs(degraded: set[dt.date], min_len: int = 3) -> set[dt.date]:
    """Degraded days that sit in a run of at least `min_len` consecutive degraded days."""
    ds = sorted(degraded)
    out: set[dt.date] = set()
    i = 0
    while i < len(ds):
        j = i
        while j + 1 < len(ds) and (ds[j + 1] - ds[j]).days == 1:
            j += 1
        if j - i + 1 >= min_len:
            out.update(ds[i:j + 1])
        i = j + 1
    return out


def valid_dates(days: list[dt.date], covered: set[dt.date], degraded: set[dt.date], venue: set[dt.date],
                window: int = WINDOW, exclusion: int = EXCLUSION, clean_span: tuple[int, int] = CLEAN_SPAN,
                outage_len: int = OUTAGE_LEN) -> list[dt.date]:
    """Dates whose D-window..D+window is in the panel and free of multi-day outages, whose ranked
    days D+clean_span[0]..D+clean_span[1] carry no flag at all, and which lie more than
    `exclusion` days from every documented venue change."""
    runs = outage_runs(degraded, outage_len)
    out = []
    for d in days:
        span = [d + dt.timedelta(days=k) for k in range(-window, window + 1)]
        if any(x not in covered or x in runs for x in span):
            continue
        if any(d + dt.timedelta(days=k) in degraded for k in range(clean_span[0], clean_span[1] + 1)):
            continue
        if any(abs((d - v).days) <= exclusion for v in venue):
            continue
        out.append(d)
    return out


def rank(stats: dict[dt.date, float], event: dt.date, larger_is_extreme: bool = True) -> dict:
    vals = {d: (v if larger_is_extreme else -v) for d, v in stats.items() if v is not None}
    e = vals[event]
    r = 1 + sum(1 for d, v in vals.items() if d != event and v >= e)
    return {"rank": r, "n": len(vals), "p": r / len(vals)}


def spaced(dates: list[dt.date], gap: int = 15, anchor: dt.date | None = None) -> list[dt.date]:
    """A subset of dates at least `gap` days apart, and at least `gap` days from `anchor` (the
    event), taken greedily outward from the anchor: windows of half-width 7 then share no day with
    each other or with the event's."""
    ds = sorted(d for d in dates if anchor is None or abs((d - anchor).days) >= gap)
    out: list[dt.date] = []
    if anchor is not None:
        before = [d for d in ds if d < anchor][::-1]
        after = [d for d in ds if d > anchor]
        for seq in (before, after):
            last = anchor
            for d in seq:
                if abs((d - last).days) >= gap:
                    out.append(d)
                    last = d
        return sorted(out)
    for d in ds:
        if not out or (d - out[-1]).days >= gap:
            out.append(d)
    return out


def rank_abs(stats: dict[dt.date, float], event: dt.date) -> dict:
    """Two-sided: the event's position by absolute size among all dates."""
    return rank({d: abs(v) for d, v in stats.items() if v is not None}, event, larger_is_extreme=True)


def add_ranges(res: dict) -> dict:
    """The placebo distribution of each statistic at k = 0 and 1, without the event: min, median, max."""
    ev = res["event"]
    out: dict = {}
    series = {"median": res["stats_k"]} | {st: per for st, per in res.get("stats_alt", {}).items()}
    for st, per_k in series.items():
        out[st] = {}
        for k, per in per_k.items():
            out[st][k] = {}
            for name, vals in per.items():
                xs = sorted(v for d, v in vals.items() if d != ev and v is not None)
                if xs:
                    out[st][k][name] = {"n": len(xs), "min": xs[0], "median": xs[len(xs) // 2], "max": xs[-1], "event": vals.get(ev)}
    res["placebo_range"] = out
    return res


def joint_rank(spec: dict[str, tuple[dict[dt.date, float], bool]], event: dt.date) -> dict:
    """Dates at least as extreme as the event on every outcome at once."""
    dates = set.intersection(*(set(s) for s, _ in spec.values()))
    def extreme(d):
        return all((s[d] >= s[event]) if larger else (s[d] <= s[event]) for s, larger in spec.values())
    r = 1 + sum(1 for d in dates if d != event and extreme(d))
    return {"rank": r, "n": len(dates), "p": r / len(dates)}


def main() -> None:
    panel = pl.read_parquet(EXT / "panel_v1v2.parquet")
    # day-level completeness lives in the health flags (missing or thin hours); here a day is covered
    # when the panel has it at all
    per_day = panel.group_by("day").agg(pl.len().alias("n"))
    covered = set(per_day.filter(pl.col("n") > 1000)["day"].to_list())
    days = sorted(panel["day"].unique().to_list())
    venue = {dt.date.fromisoformat(k) for k in VENUE_EVENTS}
    dates = valid_dates(days, covered, degraded_days(), venue)
    if CUTOVER not in dates:
        dates.append(CUTOVER)   # the event itself is always evaluated, whatever its neighbours
    print(f"placebo dates: {len(dates)} (event included)", flush=True)
    stats = {name: {} for name in OUTCOMES}
    hl = {name: {} for name in OUTCOMES}
    stats_k: dict[int, dict[str, dict]] = {}
    stats_alt: dict[str, dict[int, dict[str, dict]]] = {}
    sizes = {}
    for i, d in enumerate(sorted(dates)):
        c = cohort(panel, d)
        sizes[str(d)] = len(c)
        if len(c) < 1000:
            continue
        for name in OUTCOMES:
            p = paths(panel, c, d, k_range=[-1, 0, 1, 2, 3, 5, 7], outcome=name, draws=0 if d != CUTOVER else 1000)
            rows_k = {r["k"]: r for r in p.to_dicts()}
            stats[name][d] = rows_k[0]["median"]
            for k in (0, 1, 3):
                stats_k.setdefault(k, {}).setdefault(name, {})[d] = rows_k[k]["median"] if k in rows_k else None
                for st in ("mean", "net_up"):
                    stats_alt.setdefault(st, {}).setdefault(k, {}).setdefault(name, {})[d] = rows_k[k][st] if k in rows_k else None
            hl[name][d] = half_life(p)
        if i % 10 == 0:
            print(f"  [{i+1}/{len(dates)}] {d} cohort {len(c):,} spread k0 {stats['log_spread'][d]:+.3f}", flush=True)
    res = {"event": str(CUTOVER), "dates": [str(d) for d in sorted(dates)], "n_dates": len(dates),
           "rule": {"window": WINDOW, "exclusion": EXCLUSION, "clean_span": list(CLEAN_SPAN), "outage_len": OUTAGE_LEN},
           "cohort_sizes": sizes, "ranks": {}, "ranks_k": {}, "half_life_days": {}, "stats": {}, "stats_k": {}}
    # one-sided ranks in the pre-specified direction (spread up, updates down, depth down) and
    # two-sided ranks by absolute size, for the median, the mean and the net share of assets moving up
    for k, per in stats_k.items():
        res["ranks_k"][str(k)] = {name: rank(st, CUTOVER, larger_is_extreme=(name not in ("log_updates", "log1p_trades", "log_depth_bid")))
                                  for name, st in per.items() if CUTOVER in st}
        res["stats_k"][str(k)] = {name: {str(d): v for d, v in st.items()} for name, st in per.items()}
    res["ranks_two_sided"] = {"median": {str(k): {name: rank_abs(st, CUTOVER) for name, st in per.items() if CUTOVER in st}
                                         for k, per in stats_k.items()}}
    res["stats_alt"] = {}
    for st_name, per_k in stats_alt.items():
        res["ranks_two_sided"][st_name] = {str(k): {name: rank_abs(st, CUTOVER) for name, st in per.items() if CUTOVER in st}
                                           for k, per in per_k.items()}
        res["stats_alt"][st_name] = {str(k): {name: {str(d): v for d, v in st.items()} for name, st in per.items()} for k, per in per_k.items()}
    for name in OUTCOMES:
        if CUTOVER in stats[name]:
            res["ranks"][name] = rank(stats[name], CUTOVER, larger_is_extreme=(name != "log_updates" and name != "log1p_trades" and name != "log_depth_bid"))
        res["stats"][name] = {str(d): v for d, v in stats[name].items()}
        res["half_life_days"][name] = {str(d): v for d, v in hl[name].items()}
    # subsets: second-generation dates only (the seam moves a median by itself), and dates spaced a
    # window apart (adjacent dates share fourteen of fifteen window days); ranks for every statistic
    def subset_ranks(keep: set) -> dict:
        r: dict = {"n_dates": len(keep), "dates": [str(d) for d in sorted(keep)]}
        r["one_sided_median_k0"] = {name: rank({d: v for d, v in stats[name].items() if d in keep}, CUTOVER,
                                               larger_is_extreme=(name not in ("log_updates", "log1p_trades", "log_depth_bid")))
                                    for name in OUTCOMES if CUTOVER in stats[name]}
        r["two_sided"] = {"median": {str(k): {name: rank_abs({d: v for d, v in st.items() if d in keep}, CUTOVER) for name, st in per.items() if CUTOVER in st}
                                     for k, per in stats_k.items()}}
        for st_name, per_k in stats_alt.items():
            r["two_sided"][st_name] = {str(k): {name: rank_abs({d: v for d, v in st.items() if d in keep}, CUTOVER) for name, st in per.items() if CUTOVER in st}
                                       for k, per in per_k.items()}
        return r
    v2_dates = {d for d in dates if d >= dt.date(2026, 4, 16)} | {CUTOVER}
    res["v2_only"] = subset_ranks(v2_dates)
    sp = spaced([d for d in dates if d != CUTOVER], anchor=CUTOVER)
    res["spaced"] = subset_ranks(set(sp) | {CUTOVER})
    res["spaced"]["min_attainable_p"] = 1 / (len(sp) + 1)
    # robustness: the strict rule (every day of the window unflagged), a subset of the dates above
    strict = set(valid_dates(days, covered, degraded_days(), venue, clean_span=(-WINDOW, WINDOW), outage_len=1)) | {CUTOVER}
    res["strict"] = subset_ranks(strict)
    res["strict"]["ranks"] = res["strict"]["one_sided_median_k0"]
    if all(CUTOVER in stats[n] for n in JOINT):
        res["ranks"]["joint_spread_up_updates_down"] = joint_rank({n: (stats[n], larger) for n, larger in JOINT.items()}, CUTOVER)
        # the median is pinned at zero on ordinary days, so the joint rank is also taken on the mean and the net share
        for st in ("mean", "net_up"):
            per = stats_alt.get(st, {}).get(0, {})
            if all(CUTOVER in per.get(n, {}) for n in JOINT):
                res["ranks"][f"joint_spread_up_updates_down_{st}"] = joint_rank({n: (per[n], larger) for n, larger in JOINT.items()}, CUTOVER)
    add_ranges(res)
    write_json("b2_placebo", res, inputs=[str(EXT / "panel_v1v2.parquet"), "d0_archive_health.json"])
    print({k: v for k, v in res["ranks"].items()})


if __name__ == "__main__":
    main()

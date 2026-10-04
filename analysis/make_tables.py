"""Generate the manuscript's tables from the result JSONs.

Nothing in a table is typed by hand: every cell is read from a JSON in this
directory and formatted here, and the same JSON paths feed claims.json for the
number audit. Each table is written as `tab_<name>.tex` here and copied to
`artifacts/tables_<name>.tex`, which the manuscript `\\input`s.

Run:  uv run python make_tables.py
"""
from __future__ import annotations

import datetime as dt
import math
import shutil

from d0_common import HERE, REPO, read_json

PAPER = REPO / "artifacts"
V1_DAYS = {"2026-04-26", "2026-04-27"}
RULES = [("lr", "Lee--Ready"), ("emo", "quote rule"), ("tick", "tick test"), ("bvc_5s", "BVC, 5 s bars")]


def fmt(x, nd: int = 3) -> str:
    if x is None:
        return "---"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float) and round(x, nd) == 0:
        x = 0.0   # no "-0.000"
    if isinstance(x, int) or (isinstance(x, float) and x.is_integer() and abs(x) >= 1000):
        s = f"{int(x):,}".replace(",", "{,}")
        return s
    s = f"{x:.{nd}f}"
    return f"${s}$" if x < 0 else s


def pct(x, nd: int = 1) -> str:
    return "---" if x is None else f"{100 * x:.{nd}f}"


def day_label(day: str) -> str:
    d = dt.date.fromisoformat(day)
    lab = f"{d.day} {d.strftime('%b')}"
    return lab + (" (V1)" if day in V1_DAYS else "")


def _table(cols: str, header: str, rows: list[str], caption: str, label: str, size: str = "footnotesize", colsep: str = "3.5pt") -> str:
    body = "\n".join(r if r.startswith("\\midrule") else r + " \\\\" for r in rows)
    return (f"\\begin{{table}}[htbp]\n\\centering\\{size}\\setlength{{\\tabcolsep}}{{{colsep}}}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
            f"{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")


def _panel(cols: str, title: str, header: str, rows: list[str]) -> str:
    body = " \\\\\n".join(rows) + " \\\\"
    n = len(cols)
    return (f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n\\multicolumn{{{n}}}{{l}}{{\\emph{{{title}}}}} \\\\\n"
            f"{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}")


def _p_label(p) -> str:
    if p is None:
        return "---"
    if p < 1e-9:
        return "$<10^{-9}$"
    if p < 1e-3:
        return f"$<10^{{{int(math.floor(math.log10(p))) + 1}}}$"
    return fmt(p)


def tab_fillmatch(pooled: dict) -> str:
    days = sorted(pooled["coverage"])
    ra, rb, rc = [], [], []
    for d in days:
        c, i, m, b = pooled["coverage"][d], pooled["identity"][d], pooled["match_types"][d], pooled["buy_share"][d]
        matched = i.get("matched")
        if matched is None:
            matched = c.get("matched") if c.get("matched") is not None else (
                round(c["prints"] * c["print_matched_share"]) if c.get("prints") is not None and c.get("print_matched_share") is not None else None)
        side_mis = (matched - i["side_equal"]) if i.get("side_equal") is not None and matched is not None else (
            round((1 - i["side_equal_share"]) * matched) if matched is not None and i.get("side_equal_share") is not None else None)
        tok_mis = (matched - i["token_equal"]) if i.get("token_equal") is not None and matched is not None else (
            round((1 - i["token_equal_share"]) * matched) if matched is not None and i.get("token_equal_share") is not None else None)
        ra.append(" & ".join([
            day_label(d), fmt(c["prints"]), pct(c["print_matched_share"]),
            fmt(c.get("taker_legs_within_day")), pct(c.get("taker_matched_share_within_day")),
            fmt(side_mis), fmt(tok_mis), pct(i["price_within_5e4_share"]), pct(i["size_within_1e6_share"], 2)]))
        rb.append(" & ".join([
            day_label(d), pct(m.get("MINT", 0)), pct(m.get("COMPLEMENTARY", 0)), pct(m.get("MERGE", 0)), pct(m.get("MIXED", 0)),
            pct(m.get("UNKNOWN", 0)), fmt(b["taker_legs"]), fmt(b["all_legs"])]))
        f, r = pooled.get("fuzzy", {}).get(d, {}), pooled.get("receipts", {}).get(d, {})
        other = sum(v for k, v in r.items() if k not in ("reverted", "not_mined", "sampled", "control_settled") and isinstance(v, float))
        rc.append(" & ".join([
            day_label(d), fmt(c.get("duplicate_prints")), fmt(f.get("unmatched_prints")), fmt(f.get("fuzzy_matched")),
            pct(r.get("reverted", 0.0)), pct(r.get("not_mined", 0.0)), pct(other), fmt(r.get("sampled"))]))
    head_a = ("day & prints & with leg (\\%) & legs in day & with print (\\%) & side $\\neq$ & token $\\neq$ & "
              "price (\\%) & size (\\%)")
    head_b = "day & mint (\\%) & compl. (\\%) & merge (\\%) & mixed (\\%) & unknown (\\%) & buy share, taker & buy share, all legs"
    head_c = "day & dup. prints & unmatched prints & 2nd-stage matched & reverted (\\%) & no tx (\\%) & other (\\%) & receipts"
    cap = ("Feed prints against on-chain taker legs, joined by transaction hash. Panel~A: the share of"
           " prints with a settled taker leg; the taker legs settled inside the day (the scrape's 300-block"
           " tail past midnight excluded) and the share of them with a print; the matched pairs on which the"
           " side or the token differ; the shares of matched pairs within $5\\times10^{-4}$ in price and"
           " $10^{-6}$ in size. Panel~B: settlement types as shares of settled transactions, read from the"
           " legs' structure; \\emph{unknown} is a transaction with a maker leg on a token absent from the"
           " market metadata; the buy share over taker legs and over all legs. Panel~C: prints that repeat a"
           " hash; the prints the hash join leaves without a settled leg; those matched in a second stage on"
           " token, side, price and size within two minutes; and the receipt classes (\\emph{reverted};"
           " \\emph{no tx}, no transaction under the hash; \\emph{other}, on these days always a transaction"
           " mined outside the scraped block range) of an hour-stratified sample of the rest, weighted by"
           " each hour's population. \\emph{receipts} is the number of hashes in the sample.")
    return (f"\\begin{{table}}[htbp]\n\\centering\\scriptsize\\setlength{{\\tabcolsep}}{{3pt}}\n"
            + _panel("l" + "r" * 8, "Panel A. Coverage and identity", head_a, ra) + "\n\\par\\medskip\n"
            + _panel("l" + "r" * 7, "Panel B. Settlement types and buy share", head_b, rb) + "\n\\par\\medskip\n"
            + _panel("l" + "r" * 7, "Panel C. The residual", head_c, rc)
            + f"\n\\caption{{{cap}}}\n\\label{{tab:fillmatch}}\n\\end{{table}}\n")


def tab_direction(pooled: dict, rules=RULES) -> str:
    rows = []
    g = lambda block, key: pooled.get(block, {}).get(key, {}).get("mean")
    for key, name in rules:
        ba = pooled["balanced_accuracy"].get(key, {})
        if not ba.get("days"):
            continue
        rows.append(" & ".join([name, str(ba["days"]), fmt(ba["mean"]), f"[{fmt(ba['min'])}, {fmt(ba['max'])}]",
                                fmt(g("recall_buy", key)), fmt(g("recall_sell", key)), fmt(g("kappa", key)), fmt(g("mcc", key)),
                                fmt(g("accuracy", key)), fmt(g("majority_baseline", key)), fmt(g("coverage_rule", key))]))
    names = dict(rules)
    notes = []
    for k, v in pooled.get("contrasts", {}).items():
        a, b = k.split("_vs_")
        st = v.get("sign_test") or {}
        notes.append(f"{names.get(a, a)} minus {names.get(b, b)}: {fmt(v['mean_diff'])} [{fmt(v.get('min_diff'))}, {fmt(v.get('max_diff'))}], "
                     f"{st.get('positive')} of {st.get('days')} days, Holm $p$ {_p_label(v.get('p_holm'))}")
    header = "signal & days & balanced & [min, max] & recall, buys & recall, sells & $\\kappa$ & MCC & accuracy & majority & coverage"
    cap = ("Direction rules on the feed's own quotes, scored fill by fill against the on-chain taker side; each cell is the mean over "
           "days of the day's value, and the day is the unit. \\emph{majority} is the accuracy of always answering the more common "
           "side; \\emph{coverage} the share of matched prints the rule signs. Scores and recalls condition on each rule's signed subset; "
           "the trade history is restricted to hash-matched settlements before signing. Paired day contrasts in these conditional scores: the mean difference, "
           "its range over days, the number of days on which it is positive, and the Holm-adjusted $p$ of a paired $t$-test over days "
           f"(the sign test on 12 of 12 days gives $p=0.0005$): {'; '.join(notes)}.")
    return _table("l" + "r" * 10, header, rows, cap, "tab:direction", size="scriptsize")


def tab_signed(pooled: dict) -> str:
    rows = []
    for d in sorted(pooled.get("signed_ratios", {})):
        r = pooled["signed_ratios"][d]
        f = pooled.get("per_day", {}).get(d, {})   # the fill-by-fill ratios live in the per-day block
        rows.append(" & ".join([day_label(d), pct(r.get("wrong_side_share")), fmt(r.get("taker_eff_mean"), 4)]
                               + [fmt(f.get(f"{k}_fill_ratio", r.get(f"{k}_fill_ratio")), 2) for k in ("lr", "emo", "tick")]))
    header = "day & wrong side (\\%) & taker-signed mean & Lee--Ready & quote rule & tick test"
    cap = ("The mean effective spread under each rule's sign relative to the mean under the taker sign, both"
           " taken fill by fill on the same prints signed by that rule. The extraction retains matched prints with both the"
           " prevailing and five-minute midpoint; unsigned prints are excluded from both means for each ratio. \\emph{wrong side}: the share of fills priced"
           " on the wrong side of the midpoint of the last quote strictly before the print, for the taker (a"
           " taker buy below the midpoint or a taker sell above it); Lee--Ready signs these fills as if they"
           " were on the right side. \\emph{taker-signed mean}: the mean, over assets with at least fifty"
           " matched prints, of each asset's mean taker-signed effective spread, in price units (the text"
           " gives it in cents); the ratios divide by the fill-level mean, not by this column.")
    return _table("lrrrrr", header, rows, cap, "tab:signed")


OUTCOME_LABELS = [("log_spread", "log spread"), ("log_updates", "log updates"), ("log_depth_bid", "log depth, best bid"),
                  ("empty_share", "empty-quote share")]   # log(1 + trades): the cohort median is zero on every day; not tabulated


def _rank(r) -> str:
    return "---" if not r else f"{r['rank']}/{r['n']}"


def tab_cutover(b1: dict, b2: dict) -> str:
    """Paired within-asset changes from 27 April by event day, the recovery half-life, attrition
    bounds at k = 7, and the rank of 28 April among placebo dates (k = 0 and k = 1; strict subset)."""
    rows = []
    for key, label in OUTCOME_LABELS:
        o = b1["outcomes"].get(key)
        if not o:
            continue
        byk = {r["k"]: r for r in o["paths"]}
        k0 = byk.get(0, {})
        ci = f"[{fmt(k0.get('lo'))}, {fmt(k0.get('hi'))}]" if k0.get("lo") is not None else "---"
        att = b1.get("attrition", {}).get(key, {})
        km7, km3 = byk.get(-7, {}), byk.get(-3, {})
        rows.append(" & ".join([
            label, fmt(km7.get("mean")), fmt(km7.get("net_up")), fmt(km3.get("mean")),
            fmt(k0.get("median")), fmt(k0.get("mean")), fmt(k0.get("net_up")),
            fmt(byk.get(1, {}).get("median")), fmt(byk.get(3, {}).get("median")),
            fmt(byk.get(7, {}).get("median")), fmt(o.get("half_life_days")) if o.get("half_life_days") is not None else "$>7$",
            fmt(att.get("survivors")), fmt(att.get("worst"))]))
    header = ("Outcome & $\\bar\\Delta_{-7}$ & net$_{-7}$ & $\\bar\\Delta_{-3}$ & $\\tilde\\Delta_0$ & $\\bar\\Delta_0$"
              " & net$_0$ & $\\tilde\\Delta_1$ & $\\tilde\\Delta_3$ & $\\tilde\\Delta_7$ & half-life & survivors & tail scenario")
    cap = (f"The 28 April cutover: paired within-asset changes from 27 April on the cohort of {fmt(b1['cohort'])} assets"
           " live on each of the seven days before. $\\Delta_k$ is the change on day $k$ from day $-1$:"
           " $\\bar\\Delta$ is the cohort mean, $\\tilde\\Delta$ the cohort median and net the share of assets"
           " whose outcome rose less the share whose outcome fell; the text gives an asset bootstrap of the"
           " median at $k=0$. The empty-quote share is the share of the day's updates that show an empty"
           " quote. The half-life is the first day after the event on which the median change, in absolute"
           " value, is at most half its value on the event day. Survivors are the cohort assets whose"
           " outcome is observed on both days of the pair at $k=7$, and `tail scenario' is the median change on that"
           " day under a sensitivity scenario that assigns each leaver the adverse tail of the cohort's pre-period"
           " distribution (the 90th percentile for spread and the empty-quote share, the 10th for updates and depth)."
           " Table~\\ref{tab:placebo} ranks the changes among placebo dates.")
    return _table("l" + "r" * 12, header, rows, cap, "tab:cutover", size="scriptsize", colsep="2.5pt")


def tab_placebo(b2: dict) -> str:
    """The event's day-0 and day-1 statistics and their two-sided rank among event dates, for the
    cohort median, the mean and the net share of assets moving up; strict-subset rank of the median."""
    ev = b2["event"]
    rows = []
    for key, label in OUTCOME_LABELS:
        if key not in b2["stats_k"].get("0", {}):
            continue
        cells = [label]
        for k in ("0", "1"):
            for st in ("median", "mean", "net_up"):
                val = (b2["stats_k"][k][key].get(ev) if st == "median" else b2["stats_alt"][st][k][key].get(ev))
                r = b2["ranks_two_sided"][st].get(k, {}).get(key)
                cells += [fmt(val, 2 if st == "net_up" else 3), _rank(r)]
        cells.append(_rank(b2.get("strict", {}).get("ranks", {}).get(key)))
        rows.append(" & ".join(cells))
    header = ("Outcome & \\multicolumn{6}{c}{day 0} & \\multicolumn{6}{c}{day 1} & strict \\\\\n"
              "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}\n"
              " & median & rank & mean & rank & net up & rank & median & rank & mean & rank & net up & rank & rank")
    cap = (f"Where 28 April ranks among the {b2['n_dates']} dates formed by the event and its placebo dates. For each"
           " outcome and statistic of the paired change from the day before (cohort median; cohort mean; net"
           " share of cohort assets whose outcome rose), the event's value and its two-sided rank by"
           " absolute size, rank 1 the most extreme. The strict column ranks the day-0 median among the"
           f" {b2.get('strict', {}).get('n_dates', '---')} dates whose whole window contains no degraded day, in the pre-specified direction"
           " (spread up, updates and depth down) and, for the empty-quote share, upward; the text gives the"
           " strict ranks of the mean and the net share.")
    return _table("l" + "r" * 13, header, rows, cap, "tab:placebo", size="scriptsize", colsep="2pt")


def tab_mechanisms(b3: dict) -> str:
    """Three panels: recovery by pre-period activity quartile; the category difference-in-differences
    for days +5..+7 against -7..-1; the archive-seam placebo."""
    q = b3["recovery_by_activity_quartile"]
    rows_a = [f"{k} & {fmt(v['assets'])} & {fmt(v['k0'])} & {fmt(v['half_life_days'])}" for k, v in sorted(q.items())]
    rates = {r["category"]: r["fee_rate"] for r in b3.get("category_fee_rates", [])}
    cd = b3["category_did"]
    rows_b = []
    for cat, v in sorted(cd.items(), key=lambda kv: -kv[1]["treated_assets"]):
        fee = rates.get(cat)
        rows_b.append(f"{cat.replace('_', ' ')} & {fmt(v['treated_assets'])} & {pct(fee) if fee is not None else '---'} & {fmt(v['treated_change'])} & {fmt(v['control_change'])} & {fmt(v['did'])}")
    sm = b3["archive_seam"]
    rows_c = [f"{day_label(sm['v1_day'])} $\\to$ {day_label(sm['v2_day'])} & {fmt(sm['assets'])} & {fmt(sm['median_log_change'])} & {pct(sm['share_within_10pct'])}"]
    pa = _panel("lrrr", "A. Recovery by pre-period activity quartile (log spread)", "quartile & assets & $\\Delta_0$ & half-life", rows_a)
    pb = _panel("lrrrrr", "B. Category difference-in-differences, days $+5$ to $+7$ against $-7$ to $-1$ (log spread)",
                "category & assets & taker fee (\\%) & treated & rest & difference", rows_b)
    pc = _panel("lrrr", "C. The archive seam: paired change across the format change, no venue event",
                "days & assets & median $\\Delta$ log spread & within 10\\% (\\%)", rows_c)
    cap = (f"What the cutover is consistent with, on the cohort of {fmt(b3['cohort'])} assets. Panel A splits the cohort"
           " by pre-period update-rate quartile (q4 the most active). Panel B compares each venue category's"
           " median change in log spread from the week before to days five to seven after with the rest of"
           " the cohort; the fee is the category's taker fee rate, the coefficient on $p(1-p)$, in percent."
           " Panel C applies a paired change from 14 to 16 April across the archive's own format change on"
           " 15 April, on the assets live on both days, without a venue event.")
    return ("\\begin{table}[htbp]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n" + pa + "\n\\vspace{4pt}\n\n" + pb +
            "\n\\vspace{4pt}\n\n" + pc + f"\n\\caption{{{cap}}}\n\\label{{tab:mechanisms}}\n\\end{{table}}\n")


CLASS_LABELS = [("R", "price rule"), ("L", "activation"), ("B", "operator batch"), ("X", "residual")]   # the World Cup class changes the tick to 0.0025 and is not in this set


def tab_rollout(c0: dict) -> str:
    """Panel A: the classes of the 0.01 to 0.001 tick events with where the quote sat; Panel B: the
    activation class against the first live quote; Panel C: the hazard fit."""
    pe = c0.get("price_at_event", {})
    rows_a = []
    for k, label in CLASS_LABELS:
        n = c0["classes"].get(k, 0)
        p = pe.get(k, {})
        rows_a.append(" & ".join([label, fmt(n), pct(c0["class_shares"].get(k, 0.0)), fmt(p.get("high_mid_median")), fmt(p.get("low_mid_median")),
                                  pct(p.get("share_within_0.01_of_threshold")) if p.get("share_within_0.01_of_threshold") is not None else "---",
                                  pct(p.get("share_paired_with_complement_within_60s"), 2) if p.get("share_paired_with_complement_within_60s") is not None else "---"]))
    act = c0["activation"]
    rows_b = [f"events in the activation class & {fmt(act['n'])}",
              f"share whose first live quote came before the event & {pct(act['share_first_live_quote_before_event'])}",
              f"share whose first live quote came within an hour after & {pct(act['share_first_live_quote_within_60min_after'])}",
              f"median minutes from event to first live quote & {fmt(act['mins_to_first_live_quote_p50'], 0)}"]
    hz = c0["hazard"]
    rows_c = [f"area under the ROC curve & {fmt(hz['auc'])}",
              f"predicted daily hazard, placeholder book the day before & {fmt(hz['predicted']['placeholder_day'])}",
              f"predicted daily hazard, first active day & {fmt(hz['predicted']['first_active_day'])}"]
    rp = c0.get("resolution_proximity", {})
    rows_d = []
    for k, label in CLASS_LABELS:
        v = rp.get(k)
        if not v:
            continue
        rows_d.append(" & ".join([label.split(":")[0], fmt(v["n"]), pct(v["one_sided_share"]), fmt(v["hours_to_close_median"], 1),
                                  pct(v["share_closed_within_24h"]), fmt(v["sports_with_game_start"]),
                                  fmt(v["hours_after_game_start_median"], 1), pct(v["share_after_game_start"])]))
    pa = _panel("lrrrrrr", f"A. Classes of the {fmt(c0['events_0.01_to_0.001'])} tick events from 0.01 to 0.001, assigned in order",
                "class & events & share (\\%) & mid, high & mid, low & near threshold (\\%) & paired (\\%)", rows_a)
    pb = _panel("lr", "B. The activation class against the asset's first live quote", "quantity & value", rows_b)
    pc = _panel("lr", "C. Discrete-time hazard of adoption on the asset-day's state the day before", "quantity & value", rows_c)
    cap = ("The tick events classified. Each event takes the first class it satisfies: price rule (the last"
           " quote within 60 minutes on the same UTC day is live and has a midpoint above 0.96 or below"
           " 0.04), activation (within a day of the asset's first appearance, or before its first live quote"
           " or first trade), operator batch (50 or more tick changes within 60 seconds in one `event', the"
           " metadata's grouping of related markets), residual. `Mid' is the median midpoint of the last"
           " quote within 60 minutes before the event, live or not, on the side of the market above and"
           " below 0.5; `near threshold' is the share of those midpoints between 0.96 and 0.97 or between"
           " 0.03 and 0.04; `paired' the share of events whose market's other token changed tick within a"
           " minute. Panel B tests activation: if an event is the venue switching a new market on, it should"
           " come before the asset's first live quote. Panel C is a logit on all adoptions and a"
           " five-percent sample of non-adopting asset-days, with the day-before update rate, trading,"
           " placeholder status, price extremeness, spread at the floor, age and neg-risk as covariates.")
    pd = _panel("lrrrrrrr", "D. The events against the market's closing time and, on sports markets, the game's start",
                "class & events & one-sided (\\%) & h to close & closed $<$24 h (\\%) & with kick-off & h after kick-off & after kick-off (\\%)", rows_d)
    cap += (" Panel D: a one-sided book is a lone bid at or above 0.96 with no ask, or a lone ask at or below 0.04 with no bid; closing"
            " and game start times are the venue's metadata.")
    return ("\\begin{table}[htbp]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n" + pa + "\n\\vspace{4pt}\n\n" + pb +
            "\n\\vspace{4pt}\n\n" + pc + "\n\\vspace{4pt}\n\n" + pd + f"\n\\caption{{{cap}}}\n\\label{{tab:rollout}}\n\\end{{table}}\n")


SPEC_LABELS = [("all", "all"), ("liquid", "traded on $g-1$"), ("active_stable", "stable activity")]


def _p_boot(p, draws: int = 300) -> str:
    """A bootstrap p-value cannot resolve below one over the number of draws."""
    if p is None:
        return "---"
    return f"$<{1 / draws:.3f}$" if p < 1 / draws else fmt(p)


def tab_eventstudy(c2: dict) -> str:
    """Group-time event study: baseline, health-screened and hazard-reweighted specifications."""
    rows = []
    for block, label in (("gate", "baseline"), ("excluded", "screened"), ("reweighted", "reweighted")):
        specs = c2.get(block, {})
        for key, sl in SPEC_LABELS:
            v = specs.get(key)
            if not v:
                continue
            rows.append(" & ".join([f"{label}, {sl}", fmt(v["cohorts"]), fmt(v["treated_units_e0"]), fmt(v["pre_mean_att"]), _p_boot(v["pre_trend_p"]),
                                    fmt(v["post_mean_att"]), fmt(v["post_mean_ratio"])]))
    header = "specification & cohorts & treated & pre mean & pre $p$ & post mean & ratio"
    ws = c2.get("reweighted", {}).get("weight_summary") or c2.get("weight_summary", {})
    cap = ("Group-time effects of tick adoption on the log of the day's mean quoted spread, with"
           " not-yet-treated controls, aggregated by event time $e\\in[-7,7]$ and weighted by cohort size;"
           " day $-1$ is the base. The pre-period mean is over $e\\le-2$ and its $p$-value is the joint"
           " pre-trend test from a bootstrap over cohorts (300 draws); the spread ratio is $\\exp$ of the"
           " post-period mean. The `all' sample is a fixed one-in-five sample of assets. `Baseline'"
           " reports the three unweighted samples; `screened'"
           " removes the archive's degraded days, and with them the cohorts whose adoption or base day is"
           " degraded; `reweighted' reweights the control units by the odds $p/(1-p)$ of the fitted daily"
           " adoption hazard of Table~\\ref{tab:rollout}, averaged over each asset's panel days and capped at"
           f" 20, on the `screened' sample (median weight {fmt(ws.get('p50'), 2)}, 90th percentile {fmt(ws.get('p90'), 2)}).")
    return _table("lrrrrrr", header, rows, cap, "tab:eventstudy", size="scriptsize")


WC_OUTCOMES = [("log_spread", "log spread"), ("spread_old_ticks", "spread in the old tick"), ("at_tick_share", "share of quotes at the tick"),
               ("log_depth_bid", "log size at best bid")]


def tab_worldcup(c1: dict) -> str:
    """The 2 July change on World Cup markets: both arms, each outcome, the two-way fixed-effects
    estimate with market-clustered standard error, the paired median difference-in-differences and
    the randomisation p-value."""
    rows = []
    for arm, label in (("D", "finer: 0.01 to 0.0025"), ("U", "coarser: 0.001 to 0.0025")):
        a = c1["arms"][arm]
        tag = "; descriptive only" if a.get("descriptive_only") else ""
        head = f"\\multicolumn{{8}}{{l}}{{\\emph{{{label}: {fmt(a['treated_markets'])} treated, {fmt(a['control_markets'])} controls, nominal MDE {fmt(a['mde_log_spread'], 2)}{tag}}}}}"
        rows.append(head)
        for key, ol in WC_OUTCOMES:
            o = a.get("outcomes", {}).get(key)
            if not o:
                continue
            t, pr, ri = o["twfe"], o["paired"], o["ri"]
            rows.append(" & ".join([ol, fmt(t["beta"]), fmt(t["se_cluster"]), fmt(o.get("event_clustered", {}).get("se_cluster")),
                                    fmt(pr["treated_change"]), fmt(pr["control_change"]), fmt(pr["did"]),
                                    "---" if ri.get("degenerate") else fmt(ri["p"], 4)]))
    header = "outcome & TWFE & market s.e. & event s.e. & treated & control & paired & perm. $p$"
    cap = ("The 2 July change of World Cup markets to a 0.0025 tick, in twelve-hour bins from 30 June 12:00"
           " to 3 July 24:00 UTC. Controls are the venue's other sports markets of the moneyline, spreads"
           " and totals types that were quoted on 1 July, whose game starts between half a day after the"
           " change and the end of the window, whose tick before the change matched the arm's and whose"
           " pre-period median spread was below 0.5, computed from at least two asset-bins. This spread screen can admit one-sided books near resolution."
           " The outcomes are the log of the half-day median spread, that spread in"
           " old ticks, the share of quotes within one old tick and the log of the mean displayed size at"
           " the best bid. TWFE is the two-way fixed-effects (market and bin) coefficient on treated times"
           " post with standard errors clustered by market and by venue event; the paired columns are the median over"
           " markets of the post-period median less the pre-period median, for treated and control markets,"
           " and their difference; perm. $p$ is an exploratory two-sided market-label permutation tail share over 2{,}000 draws,"
           " with a plus-one correction. World Cup status was not randomised, so this is not causal randomisation inference."
           " The nominal MDE assumes independent markets and 80\\% power at the 5\\% level. Both arms are descriptive:"
           " their treated markets share only six and three events, respectively. The text reports whole-event bootstrap sensitivity.")
    return _table("lrrrrrrr", header, rows, cap, "tab:worldcup", size="scriptsize")


def tab_summary(x3: dict, pooled: dict) -> str:
    """Summary statistics: the whole panel over both eras (spread quantiles over live quotes, updates,
    assets per day, the venue's listing rate) and the on-chain sample."""
    w = x3["summary"]["whole_panel"]
    sm = x3["summary"]
    rows = [
        f"\\multicolumn{{6}}{{l}}{{\\textit{{Panel A: the panel, {fmt(sm['days'])} days over both archive formats, per asset-day}}}}",
        f"Binned median quoted spread, live asset-days & {fmt(w['mean'])} & {fmt(w['p25'])} & {fmt(w['median'])} & {fmt(w['p75'])} & {fmt(w['p95'])}",
        f"Asset-days, live / all & \\multicolumn{{5}}{{c}}{{{fmt(w['n_live'])} / {fmt(w['n'])}}}",
        f"Asset-days by format, first / second & \\multicolumn{{5}}{{c}}{{{fmt(sm['rows_by_era']['v1'])} / {fmt(sm['rows_by_era']['v2'])}}}",
        f"Updates per asset-day, median & \\multicolumn{{5}}{{c}}{{{fmt(int(w['updates_median'] + 0.5))}}}",
        f"Assets observed per day, minimum and maximum over clean days & \\multicolumn{{5}}{{c}}{{{fmt(sm['assets_per_day_min'])}--{fmt(sm['assets_per_day_max'])}}}",
        f"Markets listed by the venue per day, median February--July / August & \\multicolumn{{5}}{{c}}{{{fmt(int(sm['listed_median_feb_jul'] + 0.5))} / {fmt(int(sm['listed_median_aug'] + 0.5))}}}",
        "\\midrule",
        f"\\multicolumn{{6}}{{l}}{{\\textit{{Panel B: the on-chain sample, {fmt(pooled['n_days'])} days}}}}",
        f"Feed prints & \\multicolumn{{5}}{{c}}{{{fmt(pooled['totals']['prints'])}}}",
        f"Settled taker legs & \\multicolumn{{5}}{{c}}{{{fmt(pooled['totals']['taker_legs'])}}}",
        f"Prints matched to their taker leg by transaction hash & \\multicolumn{{5}}{{c}}{{{fmt(pooled['totals']['matched_prints'])}}}",
    ]
    header = "& mean & p25 & median & p75 & p95"
    cap = ("Summary statistics. Panel A is the harmonised panel of both archive formats; `live' marks live"
           " days, asset-days whose binned median spread is above zero and below 0.9; clean days exclude the"
           " degraded days of Appendix~\\ref{app:health}. Panel B is the on-chain sample of"
           " Section~\\ref{sec:identity}.")
    return _table("lrrrrr", header, rows, cap, "tab:summary", size="footnotesize")


def main() -> None:
    pooled = read_json("a4_direction_pooled")
    if (HERE / "b1_cutover.json").exists() and (HERE / "b2_placebo.json").exists():
        tex = tab_cutover(read_json("b1_cutover"), read_json("b2_placebo"))
        (HERE / "tab_cutover.tex").write_text(tex)
        shutil.copy(HERE / "tab_cutover.tex", PAPER / "tables_cutover.tex")
        (HERE / "tab_placebo.tex").write_text(tab_placebo(read_json("b2_placebo")))
        shutil.copy(HERE / "tab_placebo.tex", PAPER / "tables_placebo.tex")
    if (HERE / "x3_composition.json").exists():
        (HERE / "tab_summary.tex").write_text(tab_summary(read_json("x3_composition"), pooled))
        shutil.copy(HERE / "tab_summary.tex", PAPER / "tables_summary.tex")
    if (HERE / "c2_event_study.json").exists():
        (HERE / "tab_eventstudy.tex").write_text(tab_eventstudy(read_json("c2_event_study")))
        shutil.copy(HERE / "tab_eventstudy.tex", PAPER / "tables_eventstudy.tex")
    if (HERE / "c1_worldcup.json").exists():
        (HERE / "tab_worldcup.tex").write_text(tab_worldcup(read_json("c1_worldcup")))
        shutil.copy(HERE / "tab_worldcup.tex", PAPER / "tables_worldcup.tex")
    if (HERE / "c0_rollout.json").exists():
        (HERE / "tab_rollout.tex").write_text(tab_rollout(read_json("c0_rollout")))
        shutil.copy(HERE / "tab_rollout.tex", PAPER / "tables_rollout.tex")
    if (HERE / "b3_mechanisms.json").exists():
        (HERE / "tab_mechanisms.tex").write_text(tab_mechanisms(read_json("b3_mechanisms")))
        shutil.copy(HERE / "tab_mechanisms.tex", PAPER / "tables_mechanisms.tex")
    out = {"fillmatch": tab_fillmatch(pooled), "direction": tab_direction(pooled), "signed": tab_signed(pooled)}
    for name, tex in out.items():
        (HERE / f"tab_{name}.tex").write_text(tex)
        if PAPER.exists():
            shutil.copy(HERE / f"tab_{name}.tex", PAPER / f"tables_{name}.tex")
        print(f"wrote tab_{name}.tex ({tex.count(chr(10))} lines)")


if __name__ == "__main__":
    main()

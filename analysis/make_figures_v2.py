"""Figures for the manuscript, generated from the result JSONs.

fig_direction.pdf   (a) balanced accuracy of the four feed-only rules against the taker side,
                        by on-chain day; (b) mean effective spread under each rule's sign relative
                        to the taker sign, by day, with the wrong-side share as text.

Colour: the first four slots of the validated reference palette, assigned in fixed order
(Lee-Ready, quote rule, tick test, BVC) and never cycled; one axis per panel; thin marks;
direct labels at the right end; a legend for four series.

Run:  uv run python make_figures_v2.py
"""
from __future__ import annotations

import datetime as dt
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

from d0_common import HERE, REPO, read_json  # noqa: E402

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": False, "lines.linewidth": 1.0,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

PAPER = REPO / "artifacts" / "figures"
SERIES = [("lr", "Lee--Ready", "#2a78d6"), ("emo", "quote rule", "#eb6834"), ("tick", "tick test", "#1baf7a"), ("bvc_5s", "BVC, 5 s bars", "#eda100")]
INK, INK2, GRID = "#000000", "#333333", "#d9d9d9"


def _publication_fonts(fig, target_width: float) -> None:
    """Keep plot text legible when LaTeX scales the canvas to its printed width."""
    from matplotlib.text import Text
    minimum = 7.5 * fig.get_size_inches()[0] / target_width
    for axis in fig.axes:
        axis.tick_params(axis="both", which="both", labelsize=minimum)
    for text in fig.findobj(Text):
        if text.get_text():
            text.set_fontsize(max(text.get_fontsize(), minimum))


def _day_label(d: str) -> str:
    x = dt.date.fromisoformat(d)
    return f"{x.day} {x.strftime('%b')}"


def fig_direction(pooled: dict, out):
    import matplotlib.dates as mdates
    days = pooled["days"]
    xs = [dt.date.fromisoformat(d) for d in days]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"wspace": 0.3, "width_ratios": [1.15, 1]})
    for a in (ax, bx):
        _style(a)
        a.xaxis.set_major_locator(mdates.MonthLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        a.set_xlim(dt.date(2026, 4, 20), dt.date(2026, 8, 12))

    def marks(a, label: bool):
        lo, hi = a.get_ylim()
        for x, lab in ((dt.date(2026, 4, 28), "V2 cutover"), (dt.date(2026, 7, 10), "sports fee")):
            a.axvline(x, color=GRID, linewidth=0.8, linestyle=(0, (3, 3)))
            if label:
                a.text(x, hi - 0.02 * (hi - lo), lab, color=INK2, fontsize=6.5, rotation=90, va="top", ha="right")

    def end_labels(a, ends, min_gap):
        ends = sorted(ends, key=lambda e: e[1])
        ys = [e[1] for e in ends]
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < min_gap:
                ys[i] = ys[i - 1] + min_gap
        for (name, _), y in zip(ends, ys):
            a.annotate(name, (xs[-1] + dt.timedelta(days=2), y), color=INK2, fontsize=8, va="center")

    ends = []
    for key, name, col in SERIES:
        ys = [pooled["balanced_accuracy"][key]["by_day"].get(d) for d in days]
        ax.plot(xs, ys, color=col, linewidth=1.2, marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.6, label=name,
                linestyle=(0, (3, 2)) if key == "emo" else "-", zorder=3 if key == "emo" else 2)
        ends.append((name, ys[-1]))
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(dt.date(2026, 4, 21), 0.505, "chance", color=INK2, fontsize=7, va="bottom")
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("balanced accuracy against the taker side", color=INK2, fontsize=8)
    _panel(ax, "a")
    ax.set_xlim(dt.date(2026, 4, 20), dt.date(2026, 8, 20))
    ax.legend(frameon=False, fontsize=7, loc="center", bbox_to_anchor=(0.45, 0.84), ncol=2)
    marks(ax, label=False)
    # (b) wrong-side share by day
    ws = [100 * pooled["signed_ratios"][d]["wrong_side_share"] for d in days]
    bx.plot(xs, ws, color=SERIES[0][2], linewidth=1.2, marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.6)
    bx.set_ylim(0, max(ws) * 1.3)
    marks(bx, label=True)
    bx.set_ylabel("fills on the wrong side of the prevailing quote (\\%)", color=INK2, fontsize=8)
    _panel(bx, "b")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _shade(a, days, lo, hi):
    """One band per run of consecutive degraded days."""
    ds = sorted(days)
    i = 0
    while i < len(ds):
        j = i
        while j + 1 < len(ds) and (ds[j + 1] - ds[j]).days == 1:
            j += 1
        a.axvspan(ds[i], ds[j] + dt.timedelta(days=1), color=GRID, alpha=0.6, linewidth=0, zorder=0)
        i = j + 1
    return lo, hi


def fig_health(health: dict, out):
    """(a) markets listed by the venue and first seen by the archive, per day; (b) the share of
    each day's listings the archive ever carried. Degraded days shaded."""
    import matplotlib.dates as mdates
    rows = [r for r in health["days"] if r.get("listed") is not None]
    xs = [dt.date.fromisoformat(r["day"]) for r in rows]
    degraded = [dt.date.fromisoformat(d) for d in health["degraded_days"]]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"hspace": 0.35, "height_ratios": [1.2, 1]})
    for a in (ax, bx):
        _style(a)
        a.xaxis.set_major_locator(mdates.MonthLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    listed = [r["listed"] for r in rows]
    seen = [r["first_seen_markets"] for r in rows]
    cap = max(listed) * 1.15
    ax.plot(xs, listed, color="#2a78d6", linewidth=1.2, label="listed by the venue (Gamma)")
    ax.plot(xs, [min(v, cap) for v in seen], color="#eb6834", linewidth=1.2, label="first seen in the archive")
    clipped = [(x, v) for x, v in zip(xs, seen) if v > cap]
    for x, v in clipped:
        ax.plot([x], [cap], marker="^", markersize=4, color="#eb6834", clip_on=False)
    if clipped:
        x, v = max(clipped, key=lambda t: t[1])
        ax.annotate(f"catch-up days off scale; {v:,} on {_day_label(x.isoformat())}", (x, cap), xytext=(-6, -4),
                    textcoords="offset points", color=INK2, fontsize=7, va="top", ha="right")
    ax.set_ylim(0, cap * 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.set_ylabel("markets per day", color=INK2, fontsize=8)
    _panel(ax, "a")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _shade(ax, degraded, *ax.get_ylim())
    share = [100 * r.get("carried_share", 0) for r in rows]
    bx.plot(xs, share, color="#2a78d6", linewidth=1.2)
    bx.set_ylim(0, 100)
    bx.set_ylabel("share of the day's listings\nthe archive carried (\\%)", color=INK2, fontsize=8)
    _panel(bx, "b")
    _shade(bx, degraded, 0, 100)
    bx.text(xs[-1], 4, "shaded: degraded days", color=INK2, fontsize=7, ha="right")
    _publication_fonts(fig, 6.27)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _style(a):
    """Journal axes: left and bottom spines only, outward ticks, no grid."""
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        a.spines[side].set_color(INK)
    a.tick_params(colors=INK, direction="out", length=2.5, width=0.6)
    a.grid(False)


def _panel(a, letter: str):
    """A bold panel letter at the top left; the description lives in the caption."""
    a.set_title(f"\\textbf{{({letter})}}", loc="left", fontsize=8, color=INK, pad=4)


def fig_cutover(b3: dict, b2: dict, out):
    """(a) the cohort's median spread and (b) its empty-book share by hour from 27 April to 1 May,
    the cutover hour marked; (c) the placebo distribution of the day-0 median change in log spread
    over event dates, 28 April marked. One measure per axis."""
    fig = plt.figure(figsize=(9.6, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[1, 0], sharex=ax)
    cx = fig.add_subplot(gs[:, 1])
    for a in (ax, bx, cx):
        _style(a)
    hourly = sorted(b3["hourly"], key=lambda r: (r["day"], r["hour"]))
    days = sorted({r["day"] for r in hourly})
    t = [days.index(r["day"]) * 24 + r["hour"] for r in hourly]
    ax.plot(t, [r["spread_median"] for r in hourly], color="#2a78d6", linewidth=1.2)
    ax.set_ylabel("median spread\n(log scale)", color=INK2, fontsize=8)
    ax.set_yscale("log")
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1"])
    ax.tick_params(labelbottom=False)
    bx.fill_between(t, 0, [r["empty_share"] for r in hourly], color="#eb6834", alpha=0.35, linewidth=0)
    bx.plot(t, [r["empty_share"] for r in hourly], color="#eb6834", linewidth=1.0)
    bx.set_ylim(0, 1)
    bx.set_ylabel("empty-quote\nshare", color=INK2, fontsize=8)
    cut = days.index("2026-04-28") * 24 + b3["anticipation"]["cut_hour"]
    for a in (ax, bx):
        a.axvline(cut, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(cut + 3, 0.42, "cutover, 11:00 UTC", color=INK2, fontsize=7, va="center", ha="left")
    bx.set_xticks([i * 24 for i in range(len(days))])
    bx.set_xticklabels([_day_label(d) for d in days])
    _panel(ax, "a")
    _panel(bx, "b")
    st = b2["stats_alt"]["net_up"]["0"]["log_spread"]   # the median is tick-pinned at zero on every placebo date
    ev = st[b2["event"]]
    others = [v for d, v in st.items() if d != b2["event"] and v is not None]
    cx.hist(others, bins=20, color="#2a78d6", alpha=0.85, edgecolor="white", linewidth=0.6)
    cx.axvline(ev, color="#eb6834", linewidth=1.4)
    cx.text(ev, cx.get_ylim()[1] * 0.95, f"28 Apr: {ev:+.2f} ", color="#eb6834", fontsize=7.5, va="top", ha="right")
    cx.set_xlabel("net share whose spread widened, day 0", color=INK2, fontsize=8)
    cx.set_ylabel("placebo dates", color=INK2, fontsize=8)
    _panel(cx, "c")
    _publication_fonts(fig, 6.27)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_series(x3: dict, out):
    """(a) daily median spread over live quotes with the interquartile range; (b) assets observed in
    the archive and markets listed by the venue, per day; (c) share of live quotes at or below 0.02.
    Degraded days shaded; the cutover marked."""
    import matplotlib.dates as mdates
    rows = x3["days"]
    xs = [dt.date.fromisoformat(r["day"]) for r in rows]
    deg = [dt.date.fromisoformat(r["day"]) for r in rows if r["degraded"]]
    fig, (ax, bx, cx) = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw={"hspace": 0.32, "height_ratios": [1.2, 1, 0.8]})
    for a in (ax, bx, cx):
        _style(a)
        a.xaxis.set_major_locator(mdates.MonthLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        a.axvline(dt.date(2026, 4, 28), color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
        _shade(a, deg, 0, 1)
    med = [r["median"] for r in rows]
    ax.fill_between(xs, [r["p25"] for r in rows], [r["p75"] for r in rows], color="#2a78d6", alpha=0.18, linewidth=0, label="interquartile range")
    ax.plot(xs, med, color="#2a78d6", linewidth=1.2, label="median")
    ax.set_yscale("log")
    ax.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    ax.set_yticklabels(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])
    ax.set_ylabel("median spread\n(log scale)", color=INK2, fontsize=8)
    _panel(ax, "a")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.text(dt.date(2026, 4, 29), 0.45, "V2 cutover", color=INK2, fontsize=7, va="top")
    bx.plot(xs, [r["assets"] for r in rows], color="#eb6834", linewidth=1.2)
    bx.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    bx.set_ylabel("assets per day", color=INK2, fontsize=8)
    _panel(bx, "b")
    cx.plot(xs, [100 * r["tight_share"] if r["tight_share"] is not None else None for r in rows], color="#1baf7a", linewidth=1.2)
    cx.set_ylim(0, 100)
    cx.set_ylabel("share at or below 0.02 (\\%)", color=INK2, fontsize=8)
    _panel(cx, "c")
    cx.text(xs[-1], 4, "shaded: degraded days", color=INK2, fontsize=7, ha="right")
    _publication_fonts(fig, 5.643)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_event_study(c2: dict, out):
    """Group-time ATT on log spread by event time: baseline samples and
    the hazard-reweighted versions, with cohort-bootstrap intervals, from c2_event_study.json."""
    specs = [("all", "all assets (one in five)", "#2a78d6", "o"), ("liquid", "traded on base day", "#eb6834", "s"), ("active_stable", "already trading, stable activity", "#1baf7a", "^")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True, gridspec_kw={"wspace": 0.12})
    for a, block, title in ((ax, "gate", "(a) Baseline samples (UTC)"), (bx, "reweighted", "(b) Hazard-reweighted samples")):
        _style(a)
        a.axhline(0, color=INK2, linewidth=0.8)
        a.axvline(-0.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
        for j, (key, name, col, mk) in enumerate(specs):
            v = c2[block][key]
            es = sorted(int(e) for e in v["att_by_event_time"])
            ys = [v["att_by_event_time"][str(e)] for e in es]
            lo = [v["ci95"][str(e)][0] for e in es]
            hi = [v["ci95"][str(e)][1] for e in es]
            xs = [e + (j - 1) * 0.18 for e in es]
            a.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]], fmt=mk, color=col, markersize=4,
                       markeredgecolor="white", markeredgewidth=0.6, elinewidth=0.8, capsize=0, label=name)
        a.set_xlabel("days relative to adoption\n(base: day $-1$)", color=INK2, fontsize=8)
        a.set_title(title, loc="left", fontsize=8, color=INK)
        a.set_xticks(range(-7, 8, 2))
    ax.set_ylabel("ATT on log quoted spread", color=INK2, fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="lower center",
               bbox_to_anchor=(0.5, -0.25), ncol=3)
    _publication_fonts(fig, 5.1414)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_distribution(x3: dict, out):
    """Distribution of the per-asset median spread over live quotes in the week before and the week
    after the cutover, from the histogram x3_composition.py stores (log-spaced bins)."""
    import numpy as np
    h = x3["distribution_weeks"]
    edges = np.array(h["bin_edges"])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    _style(ax)
    for key, col, label in (("before", "#2a78d6", f"21--27 April (n = {h['before']['n']:,})"), ("after", "#eb6834", f"29 April -- 5 May (n = {h['after']['n']:,})")):
        share = np.array(h[key]["share_by_bin"]) * 100
        ax.stairs(share, edges, color=col, linewidth=1.3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("per-asset median quoted spread\n(log scale)", color=INK2, fontsize=8)
    ax.set_ylabel("share of assets per bin (\\%)", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _publication_fonts(fig, 3.8874)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PAPER.mkdir(exist_ok=True)
    if (HERE / "b3_mechanisms.json").exists() and (HERE / "b2_placebo.json").exists():
        out = HERE / "fig_cutover.pdf"
        fig_cutover(read_json("b3_mechanisms"), read_json("b2_placebo"), out)
        shutil.copy(out, PAPER / "fig_cutover.pdf")
        print(f"wrote {out}")
    pooled = read_json("a4_direction_pooled")
    out = HERE / "fig_direction.pdf"
    fig_direction(pooled, out)
    shutil.copy(out, PAPER / "fig_direction.pdf")
    print(f"wrote {out} and {PAPER / 'fig_direction.pdf'}")
    if (HERE / "c2_event_study.json").exists():
        out = HERE / "fig_event_study.pdf"
        fig_event_study(read_json("c2_event_study"), out)
        shutil.copy(out, PAPER / "fig_event_study.pdf")
        print(f"wrote {out}")
    if (HERE / "x3_composition.json").exists():
        x3 = read_json("x3_composition")
        if "distribution_weeks" in x3:
            out = HERE / "fig_distribution.pdf"
            fig_distribution(x3, out)
            shutil.copy(out, PAPER / "fig_distribution.pdf")
            print(f"wrote {out}")
        out = HERE / "fig_series.pdf"
        fig_series(read_json("x3_composition"), out)
        shutil.copy(out, PAPER / "fig_series.pdf")
        print(f"wrote {out}")
    health = read_json("d0_archive_health")
    if health["days"] and health["days"][0].get("carried_share") is not None:
        out = HERE / "fig_health.pdf"
        fig_health(health, out)
        shutil.copy(out, PAPER / "fig_health.pdf")
        print(f"wrote {out} and {PAPER / 'fig_health.pdf'}")


if __name__ == "__main__":
    main()

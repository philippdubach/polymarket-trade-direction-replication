"""Build and cache the event datasets, then check they reproduce the pilot's Table 4.

Two passes. The legacy pass keeps the pilot's Zurich-wall-clock adoption
timestamps and must recover the printed Table 4 to the digit: that is the
reproduction gate. The UTC pass is what the paper reports from v0.5 on.
Both sets of numbers go to d0_tz_fix.json so the manuscript can say what the
correction moved.
"""
import polars as pl

from d0_common import write_json
from rev_common import CACHE, adoption, events

PRINTED = {5: (-0.091, 0.029, -0.238), 7: (-0.117, 0.068, -0.335), 10: (-0.148, 0.069, -0.444)}


def summary(ev: pl.DataFrame) -> dict:
    return {
        "n": ev.height,
        "treated": round(ev["treated_chg"].median(), 4),
        "control": round(ev["control_chg"].median(), 4),
        "did": round(ev["did"].median(), 4),
        "did_corrected_control": round(ev["did_w"].median(), 4),
        "share_control_clamped": round(float(ev["ctrl_clamped"].mean()), 4),
    }


def main() -> None:
    out = {"printed_table4": {str(k): v for k, v in PRINTED.items()}, "legacy": {}, "utc": {}}
    for W in (5, 7, 10):
        leg = events(W, legacy_tz=True)
        t, c, d = leg["treated_chg"].median(), leg["control_chg"].median(), leg["did"].median()
        pt, pc, pd = PRINTED[W]
        ok = all(abs(a - b) < 0.0006 for a, b in ((t, pt), (c, pc), (d, pd)))
        print(f"legacy W={W:>2} n={leg.height:>9,}  treated={t:+.3f} control={c:+.3f} did={d:+.3f}  "
              f"printed={PRINTED[W]}  {'REPRODUCED' if ok else 'MISMATCH'}")
        out["legacy"][str(W)] = summary(leg) | {"reproduced": ok}

        ev = events(W)
        ev.write_parquet(CACHE / f"events_W{W}.parquet")
        print(f"utc    W={W:>2} n={ev.height:>9,}  treated={ev['treated_chg'].median():+.3f} "
              f"control={ev['control_chg'].median():+.3f} did={ev['did'].median():+.3f} did_w={ev['did_w'].median():+.3f}")
        out["utc"][str(W)] = summary(ev)

    a_leg, a_utc = adoption(True), adoption(False)
    j = a_leg.join(a_utc, on="asset_id", suffix="_utc")
    shifted = int((j["adoption"].dt.date() != j["adoption_utc"].dt.date()).sum())
    out["adoptions"] = {"assets": a_utc.height, "day_shifted_by_tz_fix": shifted, "share": round(shifted / a_utc.height, 4)}
    write_json("d0_tz_fix", out, inputs=["../paper-b-v2-pilot/ticks", "../paper-b-v2-pilot/shards"])
    print(f"adoptions whose UTC day differs from the legacy day: {shifted:,} of {a_utc.height:,}")
    if not all(v["reproduced"] for v in out["legacy"].values()):
        raise SystemExit("reproduction gate failed")


if __name__ == "__main__":
    main()

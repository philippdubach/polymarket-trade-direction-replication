"""Why does a print have no settled counterpart? Receipts for a stratified sample.

a1_fill_match.py leaves two residuals per day: prints whose transaction hash
matched no scraped taker leg, and taker legs with no print. For a per-hour
sample of the unmatched prints this script fetches the transaction receipt
and classifies each:

  not_mined        no receipt: the hash was never mined (replaced or dropped)
  reverted         status 0
  outside_range    mined after the day's scraped block range (+ tail)
  other_contract   settled by a contract we did not scrape
  in_range_missed  a success on the exchange inside the range: a scrape gap

Run:  uv run python a1_receipts.py 2026-05-13 --per-hour 80
Out:  a1_receipts_<day>.json
"""
from __future__ import annotations

import argparse
import random

import polars as pl

from a0_scrape_days import TAIL_BLOCKS, day_bounds, rpc
from d0_common import EXT, V1_NEG, V1_STD, V2_NEG, V2_STD, read_json, write_json

EXCHANGES = {V2_STD, V2_NEG, V1_STD, V1_NEG}


def classify(receipt: dict | None, block_range: tuple[int, int], tx_found: bool = True) -> str:
    if receipt is None:
        return "not_mined"
    if receipt.get("status") == "0x0":
        return "reverted"
    block = int(receipt["blockNumber"], 16)
    if block < block_range[0] or block > block_range[1]:
        return "outside_range"
    if (receipt.get("to") or "").lower() not in EXCHANGES:
        return "other_contract"
    return "in_range_missed"


def weighted_shares(by_hour: dict, population: dict) -> dict:
    """Class shares weighted by each hour's population of unmatched prints, not by the sample."""
    total = sum(population.values())
    out: dict[str, float] = {}
    for h, classes in by_hour.items():
        n = sum(classes.values())
        w = population.get(str(h), 0) / total if total else 0.0
        for c, k in classes.items():
            out[c] = out.get(c, 0.0) + w * k / n
    return out


def stratified(hashes: list[tuple[str, int]], per_hour: int, seed: int = 7) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    out = []
    for h in range(24):
        pool = [x for x in hashes if x[1] == h]
        rng.shuffle(pool)
        out += pool[:per_hour]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("day")
    ap.add_argument("--per-hour", type=int, default=80)
    ap.add_argument("--control", type=int, default=100, help="matched hashes to fetch as a control")
    a = ap.parse_args()
    b = day_bounds(a.day)
    rng_blocks = (b["start_block"], b["end_block"] + TAIL_BLOCKS)
    up = pl.read_parquet(EXT / "onchain" / f"day={a.day}" / "unmatched_prints.parquet")
    hashes = list(zip(up["transaction_hash"].to_list(), up["ts"].dt.hour().to_list()))
    sample = stratified(hashes, a.per_hour)
    counts: dict[str, int] = {}
    by_hour: dict[str, dict[str, int]] = {}
    examples: dict[str, list[str]] = {}
    for i, (h, hour) in enumerate(sample):
        r = rpc("eth_getTransactionReceipt", [h])
        c = classify(r, rng_blocks)
        counts[c] = counts.get(c, 0) + 1
        by_hour.setdefault(str(hour), {})[c] = by_hour.setdefault(str(hour), {}).get(c, 0) + 1
        examples.setdefault(c, [])
        if len(examples[c]) < 5:
            examples[c].append(h)
        if i % 200 == 0:
            print(f"  [{i+1}/{len(sample)}] {counts}", flush=True)
    pop = read_json(f"a1_fill_match_{a.day}")["fuzzy"].get("remaining_by_hour", {})
    res = {"day": a.day, "unmatched_prints_after_fuzzy": up.height, "sampled": len(sample), "block_range": rng_blocks,
           "classes": counts, "shares": {k: v / len(sample) for k, v in counts.items()},
           "weighted_shares": weighted_shares(by_hour, pop) if pop else None, "by_hour": by_hour, "examples": examples}
    # control: matched hashes must come back as settled on the exchange inside the range
    if a.control:
        import a0_onchain_load as ld
        from a1_fill_match import feed_prints
        pr = feed_prints(a.day)
        legs = ld.taker_legs(a.day)
        matched = pr.join(legs.select("tx_hash"), left_on="transaction_hash", right_on="tx_hash", how="semi")["transaction_hash"].unique()
        ctrl = random.Random(11).sample(matched.to_list(), min(a.control, matched.len()))
        cc: dict[str, int] = {}
        for h in ctrl:
            c = classify(rpc("eth_getTransactionReceipt", [h]), rng_blocks)
            cc[c] = cc.get(c, 0) + 1
        res["control_matched"] = {"sampled": len(ctrl), "classes": cc}
    write_json(f"a1_receipts_{a.day}", res, inputs=[str(EXT / "onchain" / f"day={a.day}" / "unmatched_prints.parquet")])
    print(res["weighted_shares"] or res["shares"], res.get("control_matched"))


if __name__ == "__main__":
    main()

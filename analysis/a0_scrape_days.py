"""Extended on-chain scrape of the CLOB V2 exchanges for whole UTC days.

Differences from the pilot's scrape_onchain_v2.py:

  * Day boundaries come from block timestamps (binary search on
    eth_getBlockByNumber), not from a 2 s/block assumption that drifts by
    weeks after April. Each log carries `blockTimestamp`, so fill times are
    exact.
  * One request covers both exchange contracts (eth_getLogs takes an address
    list) and every log is kept: OrderFilled, OrdersMatched (the taker side
    and the taker order hash) and FeeCharged. Nothing extra is requested.
  * A slice is finished only when its sha256 matches the day index, so a
    partial or corrupt file is redone rather than trusted.
  * The key comes from ALCHEMY_KEY or ~/.config/polymarket/alchemy.key.

Run:
    uv run python a0_scrape_days.py --days 2026-05-13 2026-05-31 --workers 3
Output:
    /Volumes/data/paper-b-ext/onchain/day=YYYY-MM-DD/blocks_<lo>_<hi>.parquet
    /Volumes/data/paper-b-ext/onchain/day=YYYY-MM-DD/_index.json
    /Volumes/data/paper-b-ext/onchain/block_index.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import httpx
import polars as pl
from eth_utils import keccak, to_hex

from d0_common import EXT, V1_NEG, V1_STD, V2_NEG, V2_STD, alchemy_key

OUT = EXT / "onchain"
TOPIC_ORDERFILLED = to_hex(keccak(text="OrderFilled(bytes32,address,address,uint8,uint256,uint256,uint256,uint256,bytes32,bytes32)"))
TOPIC_ORDERSMATCHED = to_hex(keccak(text="OrdersMatched(bytes32,address,uint8,uint256,uint256,uint256)"))
TOPIC_FEECHARGED = to_hex(keccak(text="FeeCharged(address,uint256)"))
# The first-generation exchange: no side field; the side follows from which asset the maker gave.
TOPIC_ORDERFILLED_V1 = to_hex(keccak(text="OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)"))
KINDS = {TOPIC_ORDERFILLED: "orderfilled", TOPIC_ORDERSMATCHED: "ordersmatched", TOPIC_FEECHARGED: "feecharged",
         TOPIC_ORDERFILLED_V1: "orderfilled"}


def contracts_for(era: str) -> list[str]:
    return [V2_STD, V2_NEG] if era == "v2" else [V1_STD, V1_NEG]
SLICE = 1000
CHUNK = 10          # Alchemy free tier: at most 10 blocks per eth_getLogs
TAIL_BLOCKS = 300   # keep scraping ~10 min past midnight so late settlements of the day's prints are captured
ANCHOR_BLOCK, ANCHOR_DAY = 86150378, dt.date(2026, 4, 29)   # the pilot's verified 00:00:00 UTC start block of 29 April

SCHEMA = {
    "kind": pl.Utf8, "block_number": pl.Int64, "block_ts": pl.Int64, "tx_hash": pl.Utf8, "log_index": pl.Int64,
    "contract": pl.Utf8, "order_hash": pl.Utf8, "maker": pl.Utf8, "taker": pl.Utf8, "side": pl.Int64,
    "token_id": pl.Utf8, "maker_amount": pl.Utf8, "taker_amount": pl.Utf8, "fee": pl.Utf8,
}


# ----------------------------------------------------------------------------- decoding

def _words(data_hex: str) -> list[int]:
    raw = bytes.fromhex(data_hex[2:] if data_hex.startswith("0x") else data_hex)
    return [int.from_bytes(raw[i:i + 32], "big") for i in range(0, len(raw) - len(raw) % 32, 32)]


def _addr(topic: str) -> str:
    return "0x" + topic[-40:].lower()


def decode_log(log: dict) -> dict | None:
    """One flat row per log, or None for a topic we do not keep."""
    topics = log["topics"]
    kind = KINDS.get(topics[0].lower())
    if kind is None:
        return None
    w = _words(log["data"])
    base = {
        "kind": kind,
        "block_number": int(log["blockNumber"], 16),
        "block_ts": int(log["blockTimestamp"], 16) if log.get("blockTimestamp") else None,
        "tx_hash": log["transactionHash"].lower(),
        "log_index": int(log["logIndex"], 16),
        "contract": log["address"].lower(),
        "order_hash": None, "maker": None, "taker": None, "side": None,
        "token_id": None, "maker_amount": None, "taker_amount": None, "fee": None,
    }
    if kind == "orderfilled" and topics[0].lower() == TOPIC_ORDERFILLED_V1:
        # V1: (makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled, fee). A maker who gave
        # collateral (asset 0) bought the token it received; one who gave a token sold it.
        maker_asset, taker_asset = w[0], w[1]
        side = 0 if maker_asset == 0 else 1
        base |= {"order_hash": topics[1].lower(), "maker": _addr(topics[2]), "taker": _addr(topics[3]),
                 "side": side, "token_id": str(taker_asset if side == 0 else maker_asset),
                 "maker_amount": str(w[2]), "taker_amount": str(w[3]), "fee": str(w[4]) if len(w) > 4 else None}
    elif kind == "orderfilled":
        base |= {"order_hash": topics[1].lower(), "maker": _addr(topics[2]), "taker": _addr(topics[3]),
                 "side": w[0], "token_id": str(w[1]), "maker_amount": str(w[2]), "taker_amount": str(w[3]),
                 "fee": str(w[4]) if len(w) > 4 else None}
    elif kind == "ordersmatched":
        base |= {"order_hash": topics[1].lower(), "maker": _addr(topics[2]),
                 "side": w[0], "token_id": str(w[1]), "maker_amount": str(w[2]), "taker_amount": str(w[3])}
    else:  # feecharged
        base |= {"maker": _addr(topics[1]), "fee": str(w[0]) if w else None}
    return base


# ----------------------------------------------------------------------------- bookkeeping

def slices(start: int, end: int, size: int = SLICE) -> list[tuple[int, int]]:
    return [(lo, min(lo + size - 1, end)) for lo in range(start, end + 1, size)]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slice_done(p: Path, index: dict) -> bool:
    ent = index.get(p.name)
    return bool(ent) and p.exists() and sha256_file(p) == ent.get("sha256")


def index_update(index_path: Path, name: str, entry: dict) -> None:
    """Merge one slice entry into the day index.

    Several workers update the same index, so the read-modify-write runs under
    an exclusive file lock and the temporary file is per process. Without the
    lock, entries were lost and a worker could rename a temp file another had
    already renamed (the first run's crash).
    """
    lock_path = index_path.with_suffix(".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            idx = json.loads(index_path.read_text()) if index_path.exists() else {}
            idx[name] = entry
            tmp = index_path.with_name(f"{index_path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(idx, indent=2, sort_keys=True))
            os.replace(tmp, index_path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


# ----------------------------------------------------------------------------- rpc

def rpc(method: str, params: list, key: str | None = None, retries: int = 10):
    key = key or alchemy_key()
    last = None
    for attempt in range(retries):
        try:
            r = httpx.post(f"https://polygon-mainnet.g.alchemy.com/v2/{key}",
                           json={"id": 1, "jsonrpc": "2.0", "method": method, "params": params}, timeout=60)
            if r.status_code == 429:
                raise RuntimeError("429 rate limited")
            r.raise_for_status()
            j = r.json()
            if "error" in j:
                raise RuntimeError(str(j["error"]))
            return j["result"]
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(90, 2 ** attempt) + 0.5 * attempt)
    raise RuntimeError(f"{method} failed after {retries} attempts: {last}")


def block_ts(n: int, key: str | None = None) -> int:
    return int(rpc("eth_getBlockByNumber", [hex(n), False], key)["timestamp"], 16)


def first_block_at_or_after(ts: int, lo: int, hi: int, key: str | None = None) -> int:
    """Smallest block in [lo, hi] whose timestamp >= ts (hi if none)."""
    while lo < hi:
        mid = (lo + hi) // 2
        if block_ts(mid, key) >= ts:
            hi = mid
        else:
            lo = mid + 1
    return lo


def search_bracket(d: dt.date, latest: int) -> tuple[int, int]:
    """Blocks that must contain the day's first block, from the anchor and the plausible block rates.

    Polygon's block time drifted from 2.0 s to 1.5 s over the sample, so the bracket spans
    the slowest and fastest plausible rates in either direction from the anchor. The first
    version assumed the day came after the anchor and, for an earlier day, produced an
    inverted bracket that the search returned unchanged: four days were scraped three days
    early before this was caught.
    """
    secs = int(dt.datetime.combine(d, dt.time(), tzinfo=dt.timezone.utc).timestamp()) \
        - int(dt.datetime.combine(ANCHOR_DAY, dt.time(), tzinfo=dt.timezone.utc).timestamp())
    a, b = ANCHOR_BLOCK + int(secs / 2.3), ANCHOR_BLOCK + int(secs / 1.3)
    lo, hi = min(a, b) - 1000, max(a, b) + 1000
    return max(1, lo), min(latest, hi)


def day_bounds(day: str, key: str | None = None) -> dict:
    """Start and end block of a UTC day, cached in block_index.json."""
    idx_path = OUT / "block_index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    if day in idx:
        return idx[day]
    d = dt.date.fromisoformat(day)
    t0 = int(dt.datetime.combine(d, dt.time(), tzinfo=dt.timezone.utc).timestamp())
    latest = int(rpc("eth_blockNumber", [], key), 16)
    lo, hi = search_bracket(d, latest)
    start = first_block_at_or_after(t0, lo, hi, key)
    end = first_block_at_or_after(t0 + 86400, start, min(latest, start + 120_000), key) - 1
    ent = {"start_block": start, "end_block": end, "blocks": end - start + 1,
           "start_ts": dt.datetime.fromtimestamp(block_ts(start, key), dt.timezone.utc).isoformat(),
           "end_ts": dt.datetime.fromtimestamp(block_ts(end, key), dt.timezone.utc).isoformat(),
           "sec_per_block": round(86400 / (end - start + 1), 4)}
    idx[day] = ent
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(idx, indent=2))
    return ent


# ----------------------------------------------------------------------------- scraping

def scrape_slice(job: tuple[str, int, int, str]) -> dict:
    """Fetch one block slice for both exchange contracts of the era, write it atomically, update the day index."""
    day, lo, hi, era = job
    key = alchemy_key()
    out_dir = OUT / f"day={day}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"blocks_{lo}_{hi}.parquet"
    index_path = out_dir / "_index.json"
    idx = json.loads(index_path.read_text()) if index_path.exists() else {}
    if slice_done(out, idx):
        return {"slice": out.name, "skipped": True}
    t = time.time()
    rows = []
    for b in range(lo, hi + 1, CHUNK):
        for log in rpc("eth_getLogs", [{"address": contracts_for(era), "fromBlock": hex(b), "toBlock": hex(min(b + CHUNK - 1, hi))}], key):
            r = decode_log(log)
            if r is not None:
                rows.append(r)
    df = pl.DataFrame(rows, schema=SCHEMA)
    tmp = out.with_suffix(".tmp.parquet")
    df.write_parquet(tmp)
    os.replace(tmp, out)
    counts = {k: int(v) for k, v in df.group_by("kind").len().iter_rows()} if df.height else {}
    index_update(index_path, out.name, {"sha256": sha256_file(out), "rows": counts, "lo": lo, "hi": hi,
                                        "seconds": round(time.time() - t, 1)})
    return {"slice": out.name, "rows": counts, "seconds": round(time.time() - t, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-tail", action="store_true", help="do not extend past midnight")
    ap.add_argument("--era", choices=["v1", "v2"], default="v2", help="which exchange contracts to read")
    a = ap.parse_args()
    alchemy_key()  # fail early if missing
    jobs = []
    for d in a.days:
        b = day_bounds(d)
        end = b["end_block"] + (0 if a.no_tail else TAIL_BLOCKS)
        jobs += [(d, lo, hi, a.era) for lo, hi in slices(b["start_block"], end)]
    print(f"days={a.days} slices={len(jobs)} workers={a.workers}", flush=True)
    t0 = time.time()
    done = 0
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(scrape_slice, jobs):
            done += 1
            if not res.get("skipped") and done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  [{done}/{len(jobs)}] {res.get('slice')} {res.get('rows')} "
                      f"({el/60:.1f} min elapsed, ~{len(jobs)*el/done/60:.0f} min total)", flush=True)
    print(f"done in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()

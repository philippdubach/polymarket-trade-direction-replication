"""Shared helpers for the analysis scripts (d*, g*, x*, a*, b*, c*).

Every script in this directory that touches the archive goes through here so
that the clock, the paths, the DuckDB settings and the JSON output format are
the same everywhere. The rules this module enforces:

  * UTC everywhere. DuckDB sessions get `SET TimeZone='UTC'`; the pilot's
    `ticks/` files carry a Europe/Zurich timestamp and are converted on read.
  * `timestamp_received` is the only clock. The V2 native `timestamp` can be
    weeks stale on re-sent book rows.
  * Results are JSON in this directory with a `_meta` block (script, git head,
    library versions, inputs) so the number audit can trace every figure.
  * Large derived data lives on the drive under `/Volumes/data/paper-b-ext`,
    never in git.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                      # analysis
REPO = HERE.parent                                      # repository root
# Workspace containing outputs/ inputs and caches; override with PAPER_B_WHEN.
WHEN = Path(os.environ.get("PAPER_B_WHEN", str(REPO)))  # input workspace
PILOT = WHEN / "outputs" / "paper-b-v2-pilot"
# data locations; each can be overridden from the environment (a replicator points PAPER_B_EXT at the
# unpacked derived files of the deposit, and the archive paths at a copy of the PMXT archive)
EXT = Path(os.environ.get("PAPER_B_EXT", "/Volumes/data/paper-b-ext"))
V2_BASE = os.environ.get("PAPER_B_V2_ARCHIVE", "/Volumes/data/polymarket-backfill-2026-04-15_to_2026-09-07/pmxt-v2")
V1_BASE = os.environ.get("PAPER_B_V1_ARCHIVE", "/Volumes/data/polymarket")
TMP = os.environ.get("PAPER_B_DUCKDB_TMP", "/Volumes/data/duckdb_tmp")

# Exchange contracts on Polygon, lowercase.
V2_STD = "0xe111180000d2663c0091e4f400237545b87b996b"
V2_NEG = "0xe2222d279d744050d28e00520010520000310f59"
V1_STD = "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"
V1_NEG = "0xc5d563a36ae78145c45a50134d48a1215220f80a"
EXCHANGES = {"v2": (V2_STD, V2_NEG), "v1": (V1_STD, V1_NEG)}

CUTOVER = dt.date(2026, 4, 28)
V1_FIRST, V1_LAST = dt.date(2026, 2, 21), dt.date(2026, 4, 15)
V2_FIRST, V2_LAST = dt.date(2026, 4, 15), dt.date(2026, 8, 10)

# Dated changes that placebo windows must avoid (b2_placebo_dates.EXCLUSION days either side): the
# list pre-specified in the v2 plan, dated when each change took effect.
VENUE_EVENTS = {
    "2026-03-06": "taker fees and maker rebates extended to all crypto markets (changelog 1 Mar)",
    "2026-03-30": "category taker fees introduced (project record, V1 era)",
    "2026-04-13": "PMXT archive v2 begins",
    "2026-04-15": "V1 archive ends / V2 archive begins",
    "2026-04-28": "CLOB V2 cutover (changelog 17 Apr; books wiped ~11:00 UTC)",
    "2026-05-28": "taker rebate program goes live (docs.polymarket.com/programs/taker-rebates)",
    "2026-07-02": "World Cup markets decimalised to 0.0025 (changelog)",
    "2026-07-10": "sports taker fee 0.03 -> 0.05, maker rebate 25% -> 15% (changelog)",
    "2026-07-24": "async matching; order responses stop returning tx hashes (changelog 17 Jul, live 24 Jul)",
    "2026-08-07": "crypto up/down markets resolve on a Chainlink TWAP (changelog)",
}


def base_for(era: str) -> str:
    return V2_BASE if era == "v2" else V1_BASE


def day_files(day: str, era: str = "v2") -> list[str]:
    """Hourly parquet files that exist for a UTC day, in hour order."""
    base = base_for(era)
    return [p for h in range(24) if os.path.exists(p := f"{base}/polymarket_orderbook_{day}T{h:02d}.parquet")]


def all_days(era: str = "v2") -> list[str]:
    base = base_for(era)
    days = set()
    for name in os.listdir(base):
        if name.startswith("polymarket_orderbook_") and name.endswith(".parquet"):
            days.add(name[len("polymarket_orderbook_"):-len("T00.parquet")])
    return sorted(days)


def connect(memory: str = "3GB", threads: int = 4):
    """DuckDB connection with the settings every extraction uses."""
    import duckdb

    os.makedirs(TMP, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory}'")
    con.execute(f"PRAGMA temp_directory='{TMP}'")
    con.execute(f"PRAGMA threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET TimeZone='UTC'")
    return con


def alchemy_key() -> str:
    """The RPC key, from the environment or the user's key file. Never logged."""
    k = os.environ.get("ALCHEMY_KEY", "").strip()
    if not k:
        p = Path.home() / ".config" / "polymarket" / "alchemy.key"
        if p.exists():
            k = p.read_text().strip()
    if not k:
        raise RuntimeError("no Alchemy key: set ALCHEMY_KEY or write ~/.config/polymarket/alchemy.key")
    return k


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _versions() -> dict:
    v = {"python": sys.version.split()[0]}
    for mod in ("duckdb", "polars", "pyarrow", "numpy"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            pass
    return v


def write_json(name: str, obj: dict, inputs: list[str] | None = None, script: str | None = None) -> Path:
    """Write `name`.json next to the scripts with a `_meta` block. Returns the path."""
    out = HERE / (name if name.endswith(".json") else f"{name}.json")
    meta = {
        "script": script or Path(sys.argv[0]).name,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_head": git_head(),
        "versions": _versions(),
        "inputs": inputs or [],
    }
    payload = {"_meta": meta} | {k: v for k, v in obj.items() if k != "_meta"}
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out


def read_json(name: str) -> dict:
    p = HERE / (name if name.endswith(".json") else f"{name}.json")
    return json.loads(p.read_text())


def update_json(name: str, key: str, value, inputs: list[str] | None = None) -> Path:
    """Merge one top-level key into an existing JSON (or create it)."""
    p = HERE / (name if name.endswith(".json") else f"{name}.json")
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur[key] = value
    return write_json(name, cur, inputs=inputs)


def degraded_days() -> set[dt.date]:
    """UTC days flagged as capture outages by d0_archive_health.py. Empty until it has run."""
    p = HERE / "d0_archive_health.json"
    if not p.exists():
        return set()
    d = json.loads(p.read_text())
    return {dt.date.fromisoformat(x) for x in d.get("degraded_days", [])}


def decode_market_id(hex_ascii: str) -> str:
    """The pilot shards store hex() of the ASCII condition id; return the '0x…' text."""
    return bytes.fromhex(hex_ascii).decode("ascii").lower()


def ticks_utc():
    """The pilot's tick_size_change rows with `ts` converted to naive UTC."""
    import glob

    import polars as pl

    fs = sorted(glob.glob(str(PILOT / "ticks" / "day=*.parquet")))
    if not fs:
        raise FileNotFoundError(f"no pilot tick files under {PILOT / 'ticks'}; set PAPER_B_WHEN to the workspace that holds outputs/")
    df = pl.concat([pl.read_parquet(f) for f in fs])
    return df.with_columns(
        pl.col("timestamp_received").dt.convert_time_zone("UTC").dt.replace_time_zone(None).alias("ts")
    )


def category_from_fee_type(fee_type: str | None) -> str:
    """The venue's own market category, from its fee schedule name: 'sports_fees_v3' -> 'sports'."""
    if not fee_type:
        return "unknown"
    import re
    base = re.sub(r"_fees(_v\d+)?$", "", fee_type)
    return "crypto" if base.startswith("crypto") else base   # 'crypto_15_min' is a crypto schedule


def fee_rate_from_schedule(schedule: str | None) -> float | None:
    if not schedule:
        return None
    try:
        r = json.loads(schedule).get("rate")
        return float(r) if r is not None else None
    except (ValueError, AttributeError, TypeError):
        return None


def gamma_categories():
    """condition_id -> venue category, taker fee rate and neg-risk flag, from the Gamma pull."""
    import polars as pl

    g = pl.read_parquet(EXT / "gamma" / "gamma_markets.parquet", columns=["condition_id", "fee_type", "fee_schedule", "neg_risk"])
    return g.select(
        "condition_id",
        pl.col("fee_type").map_elements(category_from_fee_type, return_dtype=pl.Utf8, skip_nulls=False).alias("category"),
        pl.col("fee_schedule").map_elements(fee_rate_from_schedule, return_dtype=pl.Float64, skip_nulls=False).alias("fee_rate"),
        pl.col("neg_risk").fill_null(False),
    )

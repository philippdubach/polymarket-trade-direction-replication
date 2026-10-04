"""One label per asset: which condition it belongs to and which outcome it is.

Three sources, in order of preference:

  gamma     `clobTokenIds` with `outcomes` from g0_gamma_pull.py (all eras, current state)
  v1        the YES/NO `side` carried on every V1 feed row (assets alive before 15 April)
  registry  the static identity sqlite built from the September census (90k pairs)

Outcome codes: YES / NO for binary markets, OUTCOME_0 / OUTCOME_1 otherwise, with
the venue's text kept in `outcome_text`. Sources are merged, never averaged: the
winning source is named and disagreement is flagged.

Run:  uv run python x1_token_map.py
Out:  /Volumes/data/paper-b-ext/token_map.parquet, x1_token_map.json
"""
from __future__ import annotations

import glob
import sqlite3

import polars as pl

from d0_common import EXT, WHEN, write_json

REGISTRY = WHEN / "outputs" / "fresh-static-identity-registry" / "static-identity.sqlite"


def _gamma_long(gamma: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for i in (0, 1):
        rows.append(gamma.select(
            pl.col(f"token_{i}").alias("asset_id"), "condition_id",
            pl.col(f"outcome_{i}").alias("outcome_text"),
            pl.when(pl.col(f"outcome_{i}").str.to_lowercase() == "yes").then(pl.lit("YES"))
            .when(pl.col(f"outcome_{i}").str.to_lowercase() == "no").then(pl.lit("NO"))
            .otherwise(pl.lit(f"OUTCOME_{i}")).alias("outcome"),
        ))
    return pl.concat(rows).drop_nulls("asset_id").unique("asset_id", keep="first")


def _registry_long(reg: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([
        reg.select(pl.col("yes").alias("asset_id"), "condition_id", pl.lit("YES").alias("outcome")),
        reg.select(pl.col("no").alias("asset_id"), "condition_id", pl.lit("NO").alias("outcome")),
    ]).drop_nulls("asset_id").unique("asset_id", keep="first")


def build(gamma: pl.DataFrame, v1: pl.DataFrame, reg: pl.DataFrame) -> pl.DataFrame:
    g = _gamma_long(gamma).rename({"condition_id": "cid_g", "outcome": "out_g"})
    v = v1.select("asset_id", "condition_id", "outcome").unique("asset_id", keep="first").rename({"condition_id": "cid_v", "outcome": "out_v"})
    r = _registry_long(reg).rename({"condition_id": "cid_r", "outcome": "out_r"})
    all_ids = pl.concat([g.select("asset_id"), v.select("asset_id"), r.select("asset_id")]).unique()
    m = all_ids.join(g, on="asset_id", how="left").join(v, on="asset_id", how="left").join(r, on="asset_id", how="left")
    src = (pl.when(pl.col("out_g").is_not_null()).then(pl.lit("gamma"))
           .when(pl.col("out_v").is_not_null()).then(pl.lit("v1"))
           .otherwise(pl.lit("registry")))
    outs = [pl.col(c) for c in ("out_g", "out_v", "out_r")]
    n_sources = sum(o.is_not_null().cast(pl.Int64) for o in outs)
    agree = pl.concat_list(outs).list.drop_nulls().list.n_unique() <= 1
    return m.select(
        "asset_id",
        pl.coalesce("cid_g", "cid_v", "cid_r").alias("condition_id"),
        pl.coalesce("out_g", "out_v", "out_r").alias("outcome"),
        pl.col("outcome_text"),
        src.alias("source"),
        agree.alias("sources_agree"),
        n_sources.alias("n_sources"),
    ).sort("asset_id")


def complement_map(m: pl.DataFrame) -> dict[str, str]:
    """asset -> the other asset of the same condition, for conditions with exactly two."""
    pairs = m.group_by("condition_id").agg(pl.col("asset_id").sort()).filter(pl.col("asset_id").list.len() == 2)
    out = {}
    for a, b in pairs["asset_id"].to_list():
        out[a], out[b] = b, a
    return out


def load_registry() -> pl.DataFrame:
    if not REGISTRY.exists():
        return pl.DataFrame({"condition_id": [], "yes": [], "no": []}, schema={"condition_id": pl.Utf8, "yes": pl.Utf8, "no": pl.Utf8})
    con = sqlite3.connect(REGISTRY)
    rows = con.execute("SELECT condition, yes, no FROM pairs").fetchall()
    to_hex = lambda b: "0x" + bytes(b).hex()
    to_int = lambda b: str(int.from_bytes(bytes(b), "big"))
    return pl.DataFrame({"condition_id": [to_hex(c) for c, _, _ in rows],
                         "yes": [to_int(y) for _, y, _ in rows], "no": [to_int(n) for _, _, n in rows]})


def load_v1_labels() -> pl.DataFrame:
    fs = sorted(glob.glob(str(EXT / "shards_h" / "era=v1" / "day=*.parquet")))
    if not fs:
        return pl.DataFrame({"asset_id": [], "condition_id": [], "outcome": []},
                            schema={"asset_id": pl.Utf8, "condition_id": pl.Utf8, "outcome": pl.Utf8})
    return pl.concat([pl.read_parquet(f, columns=["asset_id", "condition_id", "outcome"]) for f in fs]).drop_nulls("outcome").unique("asset_id")


def main() -> None:
    gamma = pl.read_parquet(EXT / "gamma" / "gamma_markets.parquet",
                            columns=["condition_id", "token_0", "token_1", "outcome_0", "outcome_1"])
    v1 = load_v1_labels()
    reg = load_registry()
    m = build(gamma, v1, reg)
    m.write_parquet(EXT / "token_map.parquet")
    fs = sorted(glob.glob(str(EXT / "shards_h" / "era=v2" / "day=*.parquet")))
    v2_assets = pl.concat([pl.read_parquet(f, columns=["asset_id"]) for f in fs]).unique() if fs else None
    out = {
        "assets": m.height,
        "by_source": {k: int(v) for k, v in m.group_by("source").len().iter_rows()},
        "by_outcome": {k: int(v) for k, v in m.group_by("outcome").len().iter_rows()},
        "disagreements": int((~m["sources_agree"]).sum()),
        "multi_source": int((m["n_sources"] > 1).sum()),
        "v1_assets_labelled": v1.height, "registry_pairs": reg.height,
        "v2_asset_coverage": (v2_assets.join(m, on="asset_id", how="inner").height / v2_assets.height) if v2_assets is not None else None,
    }
    write_json("x1_token_map", out, inputs=[str(EXT / "gamma" / "gamma_markets.parquet"), str(REGISTRY)])
    print(out)


if __name__ == "__main__":
    main()

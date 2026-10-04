# Analysis pipeline

Run scripts from this directory using `uv run --frozen --project .. python <script>`.

## Inputs

- `PAPER_B_EXT`: unpacked derived-data root, including `prints/`, `onchain/`, daily shards, tick contexts and the contract-day panel.
- `PAPER_B_WHEN`: input workspace containing `outputs/paper-b-v2-pilot/` and `outputs/paper-b-v2-revision/`, used by supporting event analyses.
- `PAPER_B_V1_ARCHIVE`, `PAPER_B_V2_ARCHIVE`: raw archive directories for full extraction.
- `PAPER_B_DUCKDB_TMP`: DuckDB spill directory with sufficient disk space.
- `ALCHEMY_KEY`: RPC credential for new on-chain collection or receipt queries. It is unnecessary for cached analysis and tests; supply it through the environment.

The default data paths reflect the collection drive. Set the environment variables when reproducing the work elsewhere. Use the published Zenodo directory layout for derived inputs.

## Main measurement analysis

| Script | Role |
| --- | --- |
| `a0_onchain_load.py` | Decode settled exchange legs |
| `a1_fill_match.py` | Link public prints to taker legs using transaction hash and token |
| `a1_receipts.py` | Receipt diagnostics for unmatched prints |
| `a2_direction_rules.py` | Lee–Ready, quote, tick and bulk-volume classifications |
| `a3_signed_measures.py` | Paired signed-measure comparisons |
| `a4_pool.py` | Aggregate selected-day results |
| `a5_cached_benchmark.py` | Full-history cached-tape classification benchmark |
| `a6_cached_economics.py` | Paired economic measurement and sensitivity analyses |

`a5_cached_benchmark.py` and `a6_cached_economics.py` read existing caches, require all requested inputs and do not fetch new data. Their default run covers the twelve analysis dates.

## Supporting analyses

- `g0_gamma_pull.py`, `g1_gamma_listings.py`: market metadata and the listings benchmark.
- `x0_extract_daily.py`, `x1_token_map.py`, `x2_build_panel.py`, `x3_composition.py`: harmonised daily extraction, token mapping, panel construction and composition.
- `d0_archive_health.py`: archive coverage and collector diagnostics.
- `b0_depth_validate.py`, `b1_cutover.py`, `b2_placebo_dates.py`, `b3_mechanisms.py`: depth validation and exchange-cutover comparisons.
- `c0_rollout_rule.py`, `c1_worldcup_did.py`, `c2_event_study.py`: tick-rule classification and descriptive event comparisons. `c1_worldcup_did.py --cached` uses the deposited `wc_halfday.parquet` cache.
- `rev_common.py`, `build_events.py`: shared event-data construction and baseline validation.
- `d5_manifest.py`: input/output inventory and hashes; use a full rebuild after changing derived inputs.

Full extraction can take hours and requires the raw archive. Run archive scans sequentially. UTC is the analysis clock, and `timestamp_received` is the receipt-time field. Timestamp conversion of the event inputs is explicit in the shared helpers.

## Artifact generation

`make_tables.py`, `make_figures_v2.py` and `make_jfm_artifacts.py` read the saved result JSONs. Run them through `scripts/regenerate_artifacts.sh` from the repository root to create the required output directories first.

`audit_numbers.py` compares numeric expressions in a supplied manuscript source with `claims.json`; it handles decimal values, SI expressions and grouped integers. It is a numeric provenance check, not a complete review of every caption or equation.

# Polymarket trade direction: replication code

Replication analysis for **Measuring Trade Direction in a Prediction Market: Settlement Ground Truth and Trading-Cost Measurement on Polymarket**, by Philipp D. Dubach.

The paper validates public trade prints against settled taker legs and examines how trade-sign classification changes measured trading costs. The benchmark covers twelve selected days between April and August 2026. Supporting analyses describe exchange changes, tick-size events and archive coverage.

## Contents

| Path | Contents |
| --- | --- |
| `analysis/` | Analysis scripts, saved JSON results and generated tables and figures |
| `tests/` | Synthetic-data tests for matching, classification, measurement, inference and artifact generation |
| `scripts/regenerate_artifacts.sh` | Regenerate tables and figures from saved results |
| `pyproject.toml`, `uv.lock` | Python requirements and pinned dependency versions |
| `SHA256SUMS` | Checksums of the published source and result files |

## Install and test

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/philippdubach/polymarket-trade-direction-replication.git
cd polymarket-trade-direction-replication
uv sync --frozen --extra dev
uv run --frozen --extra dev pytest
```

The tests use synthetic fixtures and do not require the large dataset or an RPC credential.

## Regenerate tables and figures

The saved JSON results are sufficient to reproduce the table and figure artifacts:

```sh
bash scripts/regenerate_artifacts.sh
```

Outputs are written to `analysis/` and `artifacts/`. This operation does not rescan the order-book archives or modify the derived input tapes.

## Recompute the main results

Download the derived files from the [Zenodo deposit](https://zenodo.org/records/23128868), unpack them and preserve their relative directory structure. Set `PAPER_B_EXT` to the directory containing `prints/` and `onchain/`:

```sh
export PAPER_B_EXT="/path/to/paper-b-ext"
cd analysis
uv run --frozen --project .. python a5_cached_benchmark.py
uv run --frozen --project .. python a6_cached_economics.py
```

The first command signs the cached daily tape before settlement selection and reports rule-specific and common-sample classification scores. The second computes paired trading-cost quantities, error-mass decompositions and sensitivities on identical eligible observations and weights. These commands overwrite their corresponding result JSONs; regenerate the artifacts afterwards.

See [analysis/README.md](analysis/README.md) for the remaining pipeline and input configuration. The large raw and derived datasets are hosted separately on Zenodo and in the PMXT archive.

## Interpretation and provenance

The twelve dates are selected and dependent. Day means and ranges describe those dates. Receipt-time ordering and missing future-quote age limit the signed five-minute measures: they are delivered-quote accounting quantities and do not identify causal price impact or private information.

Saved JSON results retain the calculation timestamps, library versions and source commit identifiers recorded when the analyses ran. Machine-specific input prefixes are replaced with environment-variable placeholders. The public scripts use the `analysis/` layout; the numerical result values are preserved. `SHA256SUMS` identifies this repository's exported files; the Zenodo deposit preserves the archived package and its provenance.

## Citation

The published dataset and replication package can be cited using DOI [10.5281/zenodo.23128868](https://doi.org/10.5281/zenodo.23128868).

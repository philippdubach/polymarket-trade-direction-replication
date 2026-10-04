#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/figures
cd analysis
uv run --frozen --project .. python make_tables.py
uv run --frozen --project .. python make_figures_v2.py
uv run --frozen --project .. python make_jfm_artifacts.py

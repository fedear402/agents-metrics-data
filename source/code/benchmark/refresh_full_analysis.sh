#!/usr/bin/env bash
# Rebuild data/analysis/full_analysis.json from saved runs.
# Read-only over data/runs and the sweep report; safe while collection is live.
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python code/benchmark/build_full_analysis.py "$@"

#!/usr/bin/env bash
# Rebuild the principal forty-task paper from saved responses.
# Arguments pass directly to the canonical builder. No provider calls occur.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run --no-sync python code/paper/rebuild.py "$@"

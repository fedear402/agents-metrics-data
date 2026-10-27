#!/bin/bash
# Thin launcher for collect.py: nohup, log routing, and a double-launch guard.
#
# Usage:
#   bash code/benchmark/collect.sh <wave-name> [collect.py args...]
#
# Example:
#   bash code/benchmark/collect.sh sonnet_20260818 \
#     --model sonnet --reps-to 20 --shards 8 --until 19:58
#
# The log lands in data/scratch_runs/<wave-name>/collect.log. Stop with:
#   pkill -f 'code/benchmark/collect.py'
#
# This script targets bash 3.2, the system shell on macOS.
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

if [ $# -lt 1 ]; then
  echo "usage: bash code/benchmark/collect.sh <wave-name> [collect.py args...]" >&2
  exit 2
fi
WAVE="$1"
shift

if pgrep -f 'code/benchmark/collect\.py' > /dev/null 2>&1; then
  echo "REFUSED: a collect.py process is already running." >&2
  echo "Stop it first: pkill -f 'code/benchmark/collect.py'" >&2
  exit 1
fi

LOGDIR="data/scratch_runs/${WAVE}"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/collect.log"

echo "[collect.sh] $(date '+%F %T') launching wave=${WAVE} args: $*" >> "$LOG"
caffeinate -i -s nohup uv run python code/benchmark/collect.py "$@" \
  >> "$LOG" 2>&1 &
echo "[collect.sh] pid $! log $LOG"

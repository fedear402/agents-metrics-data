#!/bin/sh
# Rebuild the six-model figures and tables when the GPT-6 Luna and Opus 5.5 drivers have ended.
# Started detached; writes data/analysis/six_models/after_runs.log.
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
log=data/analysis/six_models/after_runs.log
m=data/collections/2026-10-17_full-design
until grep -q -E '"event": "(driver_done|driver_stop)"' $m/luna6_driver.jsonl 2>/dev/null \
   && grep -q -E '"event": "(driver_done|driver_stop)"' $m/opus55_driver.jsonl 2>/dev/null \
   && ! pgrep -f 'drive.py --model (luna6|opus55)' >/dev/null; do sleep 300; done
echo "$(date -u +%FT%TZ) drivers ended" >> $log
for model in luna6 opus55; do
  .venv/bin/python code/benchmark/workflows/full_design_2026_10_17/paper_rows.py --model $model >> $log 2>&1
done
.venv/bin/python code/paper/six_models.py >> $log 2>&1
echo "$(date -u +%FT%TZ) six-model build done" >> $log

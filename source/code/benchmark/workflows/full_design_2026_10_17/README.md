# Full paper design with Haiku 5.5 and GPT-6 Sol

This workflow runs the full paper design (40 tasks, 3 languages, 2 prompts, 3 modes, 10 repetitions; 7,200 runs per model) with Claude Haiku 5.5 and GPT-6 Sol, the two models that replace Claude Sonnet 4.6 and GPT-5.6 Luna. Decision record: `quality_reports/decision_records/2026-10-17_haiku55-sol-full-design.md`. Plan: `quality_reports/plans/vectorized-waddling-duckling.md`.

## Collectors

- Haiku 5.5: `code/benchmark/prospective_collect_haiku_api.py` (Claude Code 2.1.278, `claude-haiku-5-5`, effort low, API key through `code/benchmark/haiku_api_claude.sh`).
- Claude Sonnet 5.5: `code/benchmark/prospective_collect_sonnet55_api.py` (Claude Code 2.1.280, `claude-sonnet-5-5`, effort low, API key through `code/benchmark/opus_api_claude.sh`; decision record `2026-10-23_sonnet55-full-design.md`). The Opus 5.5 collector `prospective_collect_opus_api.py` uses the same binary and launcher.
- GPT-6 Sol: `code/benchmark/prospective_collect_sol.py` (pinned Codex 0.163.0, `gpt-6-sol`, effort low, ChatGPT login).

Both load `prospective_collect_unlimited_additions.py`, keep the paper command, and restore the paper-time Stata `ANSWER:` reader. Tests: `code/benchmark/tests/test_haiku_api_collect.py`, `test_sol_collect.py`, `test_full_design_drive.py`.

## Requirements

- `data/collections/haiku55_api_bin/claude-2.1.278` and `data/collections/sol_codex_bin/codex-0.163.0/` (not tracked; the collectors check their SHA-256).
- `HAIKU_API_KEY_FILE` for Haiku. The launch used `~/.config/agents-metrics/haiku55_api_key` (mode 600).
- The ChatGPT login of `~/.codex` for Sol.

## Commands (repository root)

```bash
W=code/benchmark/workflows/full_design_2026_10_17
uv run --no-sync python $W/prepare.py --model sol --reps 0 4        # also: --reps 5 9, --model haiku55
uv run --no-sync python $W/preflight_sol.py --manifest data/runs/full_2026-10-17/sol/0_4/manifest.json
HAIKU_API_KEY_FILE=~/.config/agents-metrics/haiku55_api_key \
  nohup caffeinate -i .venv/bin/python -u $W/drive.py --model haiku55 > data/collections/2026-10-17_full-design/haiku55_driver.out 2>&1 &
nohup caffeinate -i .venv/bin/python -u $W/drive.py --model sol > data/collections/2026-10-17_full-design/sol_driver.out 2>&1 &
uv run --no-sync python $W/validate.py --manifest data/runs/full_2026-10-17/sol/0_4/manifest.json --reps 0 0
uv run --no-sync python $W/paper_rows.py --model sol                 # any time; unfinished runs are left out
```

- `prepare.py` writes one manifest (one model, up to five repetitions, 720 jobs per repetition) with the final prompt set, checks each prompt against the paper's Luna prompts, and runs `prep` and `check` (no model calls).
- `preflight_sol.py` runs the manifest's Codex commands without `--ephemeral` in a temporary Codex home and checks in the rollouts that the model is `gpt-6-sol` at effort low on Codex 0.163.0. Haiku uses `code/benchmark/workflows/haiku55_api_2026_10_16/preflight.py`.
- `drive.py` runs the stages (repetition 0, then 1-2, 3-4, 5-9) for one model: preflight, collection of the pending jobs, and the stage check. It waits through usage limits, sets holds aside, and resumes from the records after any stop. Haiku runs inside the paper's write-deny profile; Sol runs in Codex's own sandbox.
- `paper_rows.py` builds paper-format rows (the columns of `data/analysis/main/data/runs.csv` plus `model_id`, labels `haiku` and `sol`), prices (the columns of `run_pricing.csv`; Haiku 5.5 and GPT-6 Sol list prices), and a summary for one model, with the paper's own row and scoring functions.
- `recover.py` resolves runs after a stopped script (decision record `2026-10-18_runaway-script-rule.md`): self-repair before round 4 gets the collector's repair request, other runs end as `execution_timeout`, and runs held by the collector stop return to the request or evaluate stage. `drive.py` runs it before each collector run; `--model M --manifest F` alone shows the changes without applying them.
- `drive.py` also stops any script that a collector of its model started after 15 minutes (SIGTERM, then SIGKILL after 30 seconds) and writes a receipt to `script_stops.jsonl`.
- `validate.py` is the stage check: completed records, holds, and for Haiku the model, route, version, and absence of the key; for Sol the pinned binary, model, and effort of every saved command, and stream errors.

## Folders

- Runs: `data/runs/full_2026-10-17/<model>/{0_4,5_9}/runs/<protocol>/`.
- Preparation, driver events (`<model>_driver.jsonl`), collector and preflight logs, stage checks, and cleared quota files: `data/collections/2026-10-17_full-design/`.

## Stop or resume

- Stop a model: end its `drive.py` process. The collector marks requests in flight as interrupted holds, as in the paper.
- Resume: start `drive.py` again. It skips checked stages and pending jobs continue.

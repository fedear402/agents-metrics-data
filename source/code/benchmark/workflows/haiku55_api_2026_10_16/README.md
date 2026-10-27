# Haiku 5.5 on the paper runtime

This workflow runs paper tasks with Claude Haiku 5.5 (`claude-haiku-5-5`, effort low), billed to an Anthropic API key. It uses the paper's collector, prompts, and `claude` command. Decision record: `quality_reports/decision_records/2026-10-16_haiku55-api-collection.md`. Plan: `quality_reports/plans/vectorized-waddling-duckling.md`. The runs stay outside the paper sample.

## Requirements

- Claude Code 2.1.278 at `data/collections/haiku55_api_bin/claude-2.1.278` (not tracked; `prospective_collect_haiku_api.py` checks its SHA-256).
- `HAIKU_API_KEY_FILE` names a file that holds the API key. Keep the file outside the repository. Only `code/benchmark/haiku_api_claude.sh` reads it.

## Commands (repository root)

```bash
export HAIKU_API_KEY_FILE=/path/outside/the/repository/key.txt
W=code/benchmark/workflows/haiku55_api_2026_10_16
uv run --no-sync python $W/prepare.py --task broad_tobit --reps 0
M=data/runs/haiku55_api_2026-10-16/broad_tobit/0_0/manifest.json
uv run --no-sync python $W/preflight.py --manifest $M
# Launch: the argv in data/collections/2026-10-16_haiku55-api/<batch>_launch_commands.json
uv run --no-sync python $W/validate.py --manifest $M
uv run --no-sync python $W/score.py --manifest $M
uv run --no-sync python $W/paper_rows.py --manifest $M
uv run --no-sync python $W/analyze.py --manifest $M
```

- `prepare.py` takes each cell's prompt from the paper's selected Sonnet 4.6 runs of the task and checks it against the canonical prompt files. It writes the prompts, catalog, configuration, manifest, authorization, and launch command. It makes no model call.
- `preflight.py` makes two calls with the manifest's chatbot and agent commands. It checks the model, the API-key route, the Claude Code version, and that the agent's shell sees no key.
- The launch command runs the collector inside the paper's write-deny profile with 4 workers.
- `validate.py` checks that every planned run finished, that every stream used `claude-haiku-5-5` on the API key with Claude Code 2.1.278, and that no saved file contains the key.
- `score.py` scores each run with the paper rule `channel_se`, prices tokens at the Haiku 5.5 list prices, and compares each cell with the paper's Sonnet 4.6 and Luna runs. It writes `quality_reports/audits/2026-10-16_haiku55-api/paper_runtime_<task>.md`.
- `paper_rows.py` builds one row per run with the paper's own functions (`saved_response_analysis.fresh_record` and `scored_row`, then the per-run update of `code/paper/dataset.py` `build`). The rows have the 51 columns of `data/analysis/main/data/runs.csv` plus `model_id`, with model label `haiku`. The pricing rows have the columns of `run_pricing.csv` and use the Claude branch of `production_paper_pricing.price_record` with Haiku 5.5 rates. A request with more than 100,000 prompt tokens is left unpriced, because Haiku 5.5 has higher prices above that length.
- `analyze.py` describes each run next to the paper's Sonnet runs of the same task: calls, turns, tool calls, shell commands that install, use the network, or name paths outside the workspace, collector access findings, `verification.txt`, stop reasons, thinking tokens, cache writes by duration, models, and Stata repairs caused by the strict `ANSWER:` reader.

## Folders

- Runs: `data/runs/haiku55_api_2026-10-16/<task>/<first rep>_<last rep>/`.
- Preparation, preflight, launch, validation, and scores: `data/collections/2026-10-16_haiku55-api/`.

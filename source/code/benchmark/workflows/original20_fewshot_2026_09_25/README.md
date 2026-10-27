# Rerun of the original tasks, the 12 added tasks, and flagged runs

This workflow reruns selected positions with new few-shot examples that show the task's own calculation, in the full example layout of the added tasks. Its scopes cover three sets: the 20 original tasks, these tasks plus the 12 added tasks without the full layout, and a list of flagged runs. The decision record is `quality_reports/decision_records/2026-09-25_original20-fewshot-rerun.md`. Its status is PROPOSED until Federico approves it.

No program in this folder calls a model provider, except `launch.py --confirm`. Only Federico runs `launch.py`.

## Design

| Scope | Tasks with new blocks | Prompts | Positions | Blocks |
|---|---|---|---|---|
| `few_shot` | 20 original | few-shot | 3,600 | 4 of 900 |
| `few_shot_and_zero_shot` | 20 original | both | 7,200 | 4 of 1,800 |
| `original20_added12_both_prompts` | 20 original and 12 added | both | 11,520 | 4 of 2,880 |
| `flagged_runs` | none | as listed | the run list (79 on 2026-09-26) | one per model and repetition half that has a listed position (2 on 2026-09-26) |
| `original20_added12_both_prompts_and_flagged_runs` | 20 original and 12 added | both, and as listed | 11,520 plus the listed positions of the other 8 added tasks (11,525 on 2026-09-26) | 4 |

- Each task scope has 2 models (Luna low, Sonnet medium), 3 languages, 3 agency conditions, and 10 repetitions for each prompt.
- A block holds one model and five repetitions (0 to 4 or 5 to 9), because one manifest accepts local repetitions 0 to 4. Each repetition wave is shuffled with the seed 20260925 plus the repetition.
- Each position replaces the selected record in the same position. `planned_positions.json` links the record path and its SHA-256 hash from `data/production/main/baseline_runs.csv`.
- The 12 added tasks are `applied_multivintage_trade_iv`, `applied_nonrecipient_population_iv`, `broad_cohort_stack`, `broad_oaxaca`, `hard_conditional_logit_welfare`, `hard_heteroskedastic_probit`, `hard_panel_lp_iv`, `hard_poisson_hurdle_effects`, `hard_truncated_normal_effects`, `lp04_ppml`, `lp11_duration`, and `pilot_probit_interaction_ame`. The other 8 added tasks already have the full layout and keep their prompts.
- A listed position of a task with new blocks gets the new prompt in the combined scope. Every other listed position gets its production prompt. In the `flagged_runs` scope, every listed position gets its production prompt.

## Files

| File | Purpose |
|---|---|
| `original20_plan.py` | Shared constants and functions: scopes, production prompts, the new example block, example checks, the run list, jobs, record links, and the plan digest. |
| `production_prompts.py` | Extracts the production prompts of the 20 added tasks from the manifests of their selected runs. `--write` writes `data/production/added20_prompts_2026-09-26/`. |
| `expected_outputs.py` | Runs the 96 examples and writes the expected output that each block shows. |
| `prepare.py` | Dry run by default. `--write` builds the collection after all checks pass. |
| `check.py` | Verifies the prepared collection and runs the central collector check. Writes `validation.json`. |
| `launch.py` | Writes `authorization.json` for each block and starts the queues, supervisors, and `caffeinate`. Requires `--confirm`. |
| `estimate.py` | Cost and run-time estimates from saved runs of the same tasks. |

Inputs:

- Examples of the 20 original tasks: `data/task_candidates/2026-09-25_original20_examples/<task>/`. Examples of the 12 added tasks: `data/task_candidates/2026-09-26_added12_examples/<task>/`. Each task folder has `example.py`, `example.R`, `example.do`, `block_parts.json` (title, context, explanation, and one note for each language), `expected_python.json`, `expected_r.json`, `expected_stata.txt`, `CHANGES.md`, and `validation.json`. `expected_outputs.py` writes the three expected output files from fresh runs.
- Production prompts (720 texts): `data/production/original10_2026-09-19/prompts/` and `data/collections/2026-09-17_live-check-and-checkpoint/work/old_prompts/prompts/` for the 20 original tasks, and `data/production/added20_prompts_2026-09-26/prompts/` for the 20 added tasks. Its `provenance.json` gives the manifests and the run count of each text. Every added-task key has one text in its selected runs.
- Run list of the flagged-run scopes: by default `quality_reports/audits/2026-09-26_rerun-throughput/work/exposure/rerun_list.csv`, the 79 flagged runs of the 2026-09-26 exposure audit. `--run-list` names another file. A CSV with a `position_id` column is the preferred form, and only that column is read. In any other text file, every position id (`task__model__language__prompt__agency__repetition`) and every selected `run_record.json` path counts, also one that the prose only mentions.
- Tasks and references: `data/production/main/catalog.json` (read only).

Outputs of `--write` (one collection folder per scope family):

| Scope | Collection folder | Raw-run folder |
|---|---|---|
| `few_shot`, `few_shot_and_zero_shot` | `data/collections/2026-09-25_original20-fewshot/` | `data/runs/original20_fewshot_2026-09-25/` |
| `original20_added12_both_prompts` | `data/collections/2026-09-26_rerun32-both-prompts/` | `data/runs/rerun32_both_prompts_2026-09-26/` |
| `flagged_runs` | `data/collections/2026-09-26_flagged-runs/` | `data/runs/flagged_runs_2026-09-26/` |
| `original20_added12_both_prompts_and_flagged_runs` | `data/collections/2026-09-26_rerun32-and-flagged-runs/` | `data/runs/rerun32_and_flagged_runs_2026-09-26/` |

- The collection folder holds `plan.json`, `collections.json`, `planned_positions.json`, `source_prompts.json`, `example_checks.json`, `preparation_receipts.json`, and one `<batch>_launch_commands.json` per block. `check.py` adds `validation_checks.json` and `validation.json`. `launch.py` adds `launch_receipt.json`, logs, and supervisor folders.
- The raw-run folder holds `<model>/<first>_<last>/` with `prompts/`, `catalog.json`, `config.json`, `manifest.json`, and later `authorization.json` and `runs/`.
- Each ledger row has `flagged` (listed in the run list) and `prompt_source` (`new_example_block`, `production_zero_shot`, or `production`).

## What prepare checks

- The decision record has the line `Status: APPROVED` and the line `Approved scope: <scope>` with the same scope as `--scope`. A flagged-run scope also needs the line `Approved run list sha256: <hash>` with the SHA-256 of the run-list file.
- Every listed position is a selected position in `baseline_runs.csv`.
- All example files of the scope's tasks exist. Each file runs alone in an empty temporary directory within the chosen limit of 600 seconds. Python runs with the project interpreter, R with `Rscript --vanilla`, and Stata in batch mode under the collectors' Stata lock.
- Python and R examples write `result.json` with finite estimates. Stata examples print parseable `ANSWER:` lines and the log has no `r(...)` error code.
- An example must not name the task data file or contain a reference value. The block text (title, context, explanation, and expected output) has the same checks. A number with four or more significant digits in the block text must not equal a reference value at its own precision.
- The example outputs must not match the reference value of every scored term within the primary tolerance max(0.01 x |reference|, 0.005).
- Each fresh example output must equal the stored expected output. Numbers may differ by a relative 1e-8 (a chosen check tolerance).
- Each new few-shot prompt equals the production prompt with only the demonstration block replaced. The prompt without its block equals the zero-shot prompt. Every other prompt equals its production prompt.

## Example block

The new block uses the layout of the added tasks with a full example block. The layout comes from the 2026-09-16 Python builder (`data/collections/2026-09-16_clearer-examples/src/prompts/build_bank.py`) and the 2026-09-17 R and Stata builder (`data/collections/2026-09-17_live-check-and-checkpoint/src/new_prompts/build.py`).

````text
Related example: <title>
Separate example context: <context>
Explanation of this example: <explanation> <note for this language>
<Python only: relation line>
```<language>
<example file>
```
Expected result.json:            (Stata: Expected printed output:)
```json                          (Stata: ```text)
<expected output>
```
````

The Python relation line is: "The example uses separate data and writes result.json. It uses the scored task's identifier and result names. The scored task below must read its own input.csv and write result.json." For `applied_multivintage_trade_iv` and `applied_nonrecipient_population_iv`, which read named CSV files and no `input.csv`, the line ends with "must read its own supplied CSV files and write result.json." The expected output of Python and R is `result.json` as indented JSON. The expected output of Stata is the `ANSWER:` lines of the log, with the padding after each equals sign removed.

Position of the block:

- Original tasks: the new block replaces the text from `Few-shot demonstration:` to the task line.
- Added tasks: the old block is the text that the few-shot prompt adds to the zero-shot prompt. When several insertion points give the same text, the block starts at the last line start. The new block takes the same place and keeps the blank line that followed the old block.

## Commands

Run every command from the project root. Use `uv run --no-sync python`, because each manifest pins the exact Python executable.

| Option | Scope value | Positions |
|---|---|---|
| A | `few_shot` | 3,600 |
| B | `few_shot_and_zero_shot` | 7,200 |
| C | `original20_added12_both_prompts` | 11,520 |
| D | `flagged_runs` | the run list (79) |
| E | `original20_added12_both_prompts_and_flagged_runs` | 11,520 plus listed positions outside the 32 tasks (11,525) |

### Step 1. Dry run

The dry run counts the positions, reads the run list, and runs the examples. It compares their outputs with the stored expected outputs, renders the prompts, and reports the block lengths by task half (`prompts.block_chars`). It writes nothing. On 2026-09-26 at 02:54 the dry run for option C took 34 seconds.

If an example file changes, run `expected_outputs.py` before the dry run. It rewrites only the expected output files of the task folders.

```bash
uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/prepare.py --scope original20_added12_both_prompts
uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/prepare.py --scope original20_added12_both_prompts_and_flagged_runs
uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/prepare.py --scope flagged_runs --run-list <file>
```

The last line must list only the decision record problems. Any other problem stops the preparation.

### Step 2. Approve the decision record

Open `quality_reports/decision_records/2026-09-25_original20-fewshot-rerun.md`. Replace the two lines `Status: PROPOSED` and `Approved scope: none`. For option C:

```text
Status: APPROVED

Approved scope: original20_added12_both_prompts
```

For option D or E, also add the SHA-256 of the run-list file. The dry run prints it in the problem line and in `run_list.sha256`.

```text
Status: APPROVED

Approved scope: original20_added12_both_prompts_and_flagged_runs

Approved run list sha256: <sha256 of the run-list file>
```

Make all other edits to the record before step 3. After step 3, do not edit the record or the run list until the collection is complete. `plan.json` pins the SHA-256 hashes of both, and `check.py` and `launch.py` refuse a changed file.

### Step 3. Prepare

```bash
uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/prepare.py --write --scope <scope>
```

`--write` runs the examples again and then writes the prompts, the manifests, the collection index, and the position ledger. The central `prep` and `check` read local files, native package lists, and CLI help. They send no model request. `--write` writes no `authorization.json`.

If `--write` stops after it creates the collection folder, read `preparation_receipts.json` in that folder. Then move the collection folder and the raw-run folder of the scope aside before a new `--write`.

### Step 4. Check

```bash
uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/check.py
```

Without `--scope`, `check.py` uses the only prepared collection. If several collections are prepared, add `--scope <scope>`. On success, it writes `validation.json` in the collection folder.

### Step 5. Collect

This step authorizes provider calls. It writes `authorization.json` for each block and starts one queue, one supervisor, and one `caffeinate` process per block. Only Federico runs it.

```bash
uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/launch.py --confirm
```

Without `--confirm`, `launch.py` only reports whether the collection is ready. `--scope` works as in `check.py`.

### Step 6. Monitor

Run this for each block folder under the raw-run folder of the scope (for option C: `luna/0_4`, `luna/5_9`, `sonnet/0_4`, and `sonnet/5_9` under `data/runs/rerun32_both_prompts_2026-09-26/`). The first three lines give the planned and completed counts.

```bash
uv run --no-sync python code/benchmark/prospective_collect_unlimited_additions.py status --manifest data/runs/rerun32_both_prompts_2026-09-26/luna/0_4/manifest.json | head -3
```

### Usage watchers

The usage watcher (`code/benchmark/usage_watch.py`) watches the Claude Max and ChatGPT Pro windows. The collection then runs until a subscription window is nearly full, waits for the reset, and continues. It never turns on extra usage and never uses a Codex rate-limit reset credit.

Sources. Neither source spends model tokens.

- Claude: the `rate_limit_event` line that Claude Code writes into every Sonnet provider stream. It gives the five-hour and seven-day utilization, the reset times, and the extra-usage flags. Claude has no status request without a model call, so readings come only from collection calls. Before a launch, check `/usage` in Claude Code.
- Codex: the `account/rateLimits/read` request of `codex app-server`. It sends no model request. On 2026-09-26 one request took 1.9 seconds. `codex exec --json` streams carry no limit data. The newest local Codex session file (`~/.codex/sessions`) is a fallback for `status`.
- Codex limit buckets. The answer has one view for each metered limit bucket (`rateLimitsByLimitId`). The watcher reads every bucket. Windows of the `codex` bucket keep plain names (`seven_day`). Windows of another bucket are named `<limit id>:<window>`. Only the `codex` bucket and a bucket whose model slug is in `model_slugs` (`gpt-5.6-luna`) can slow or pause the collection.
- Reset times. Readings of one Codex window report reset times that differ by 1 to 2 seconds. The watcher treats reset times within 600 seconds (chosen tolerance) as one window and keeps the higher usage.

Validation on 2026-09-26 (`quality_reports/audits/2026-09-26_rerun-design/work/watcher-validation/validation.md`):

- The app-server reading is live. It agrees with the `token_count` readings in non-ephemeral Codex session files and with an app-server run under the isolated Codex home: same account, same integer percent, same reset time.
- The weekly percent is an integer. One Luna collection call costs far less than one point. The fitted cost is about 0.4 points of the week for each 1,000 collection calls (estimate). The measured upper bound is 2.4 points for each 1,000 calls. Personal GPT-6-Astra use filled most of the week of 2026-09-17.
- A batch of a few hundred Luna calls does not move the reading. The fitted cost predicts this result.

Decisions for each provider, before each new provider request:

| Decision | When | Effect |
|---|---|---|
| `run` | every window below its slow limit, or no reading yet | The request starts. |
| `slow` | a window at or above its slow limit | The request starts after the chosen spacing since the last start of the same provider (30 seconds for Claude, 10 for Codex). |
| `pause` | a window at or above its pause limit, a rejection, extra usage in use, or Codex reports that included usage is not allowed | The request waits until the window reset time plus 60 seconds. Then the watcher decides again. |

The limits are chosen settings in `code/benchmark/usage_watch.json`. They are not provider values. Claude: slow at 85 percent and pause at 95 percent of the five-hour window, slow at 90 and pause at 98 percent of the seven-day window. Codex: slow at 90 and pause at 97 percent of the weekly window. The pause limits leave room for requests in flight, because a window reached during a response creates a `quota_after_model_work` hold. The watcher polls Codex every 120 seconds (300 while paused). All collector processes share one state folder, so one poll serves all Luna blocks. A replay of all 8,504 saved Sonnet readings (2026-09-17 to 2026-09-22) gives `run` for every reading. The highest readings were 69 percent (five-hour) and 83 percent (seven-day).

Records in `data/collections/usage_watch/`:

- `usage_watch.jsonl`: one line per provider when the decision changes, and at least every 300 seconds while requests start. Each line has the decision, the reset time, the window readings, the reading age and source, and the SHA-256 of the limits file.
- `status.txt`: the current state of both providers in plain text.
- `state.json`: shared readings, decisions, and request starts. Do not edit it during a collection.

Commands:

```bash
# Current state of both providers (reads saved streams, local Codex session files, and one Codex status request)
uv run --no-sync python code/benchmark/usage_watch.py status
# The same without the Codex status request
uv run --no-sync python code/benchmark/usage_watch.py status --no-poll
# Decisions that the chosen limits give for saved streams (no provider request)
uv run --no-sync python code/benchmark/usage_watch.py replay --runs data/runs/production_repairs_2026-09-21/sonnet
# Follow the watcher during a collection
tail -f data/collections/usage_watch/usage_watch.jsonl
cat data/collections/usage_watch/status.txt
```

Collector. `code/benchmark/prospective_collect_watched.py` is the additive-R collector plus the watcher. `code/benchmark/production_queue_watched.py` is the queue that starts it. The central collector and the other wrappers are unchanged. The wait happens before the collector records a request as in flight and before it takes the Stata lock, so a stop signal during a pause leaves the position unstarted. Manifests prepared with the watched collector record the watcher version and its limits in the policy and pin `usage_watch.py`. A scratch `prep` and `check` with the watched collector passed on 2026-09-26 without provider calls.

To use it in this workflow, set `COLLECTOR` to `prospective_collect_watched.py` and `QUEUE` to `production_queue_watched.py` in `original20_plan.py` before step 3. The prepared manifests then pin the watcher. After step 3, do not edit `usage_watch.py`, because the collector checks the pinned sources before each request. A changed `usage_watch.json` applies after the collector restarts.

Tests:

```bash
uv run --no-sync pytest -q code/benchmark/tests/test_usage_watch.py
```

### Refusals before approval

- `prepare.py --write` refuses when the status is not `APPROVED` or the approved scope differs from `--scope`. It also refuses a flagged-run scope without a run list or with another approved run-list hash. It writes no file.
- `prepare.py --write` also refuses when an example is missing or fails, or when a prompt does not render. It refuses a listed position that is not selected, and an existing collection folder.
- `check.py` refuses when no prepared collection exists. It also fails when the record is no longer `APPROVED`, or when the record or the run list changed after step 3.
- `launch.py` refuses without `validation.json` for the current plan, an `APPROVED` record with the prepared scope (and run list), and an unchanged record and run list. It refuses after an earlier launch.

### Other commands

Estimates (the flagged-run estimates use the default run list or `--run-list`):

```bash
uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/estimate.py
```

Production prompts of the added tasks (dry run, then write):

```bash
uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/production_prompts.py
uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/production_prompts.py --write
```

Tests:

```bash
uv run --no-sync pytest -q code/benchmark/tests/test_original20_fewshot.py
```

The tests use a temporary decision record, a temporary run list, and temporary output folders. They pass while the project record is PROPOSED.

## Estimates

These values are estimates from saved runs of the same tasks (`estimate.py`, 2026-09-26). They are not measurements of the new collection. Each task group (20 original, 12 added, 8 added with the full layout) uses its own means.

| Option | Positions | Cost, runtime-adjusted | Cost at the current regime mix | Stata lock hours | Wall time with one Stata lock | With four Stata permits |
|---|---|---|---|---|---|---|
| A, `few_shot` | 3,600 | 93.2 USD | 115.2 USD | 6.4 | about 6.4 hours | about 1.6 hours |
| B, `few_shot_and_zero_shot` | 7,200 | 179.2 USD | 223.6 USD | 12.8 | about 12.8 hours | about 3.2 hours |
| C, `original20_added12_both_prompts` | 11,520 | 471.5 USD | 516.0 USD | 30.2 | about 30.2 hours | about 7.6 hours |
| D, `flagged_runs` | 79 | 0.9 USD | 1.0 USD | 0.3 | about 0.3 hours | about 0.1 hours |
| E, combined | 11,525 | 471.6 USD | 516.1 USD | 30.3 | about 30.3 hours | about 7.6 hours |

- The 12 added tasks alone add 4,320 positions, 292.4 USD, and 17.4 Stata lock hours. All their selected runs come from the prospective runtime, so both cost columns agree for them.
- The collector holds one native Stata lock for each whole Stata agent session and for each Stata script execution. The blocks share this lock, so the Stata work sets the run time. The mean Stata agent session takes 55 seconds in the original tasks and 127 seconds in the 12 added tasks. The 90th percentiles are 80 and 183 seconds.
- Four permits need the separate 2026-09-21 scheduling adapter, which this workflow does not start.
- Python and R need about 2.4 hours for option C with 32 workers.
- Quota pauses and long sessions can add time. The new few-shot prompts are longer than the production prompts. The estimates do not include this change.

## After collection

The collection does not change `data/analysis/main`. The section "Analysis switch" of the decision record gives the full rule. In short:

1. The new runs enter only when every planned position has a terminal record. All positions switch together.
2. A separate change to `code/paper/dataset.py` substitutes the planned positions in `baseline_runs.csv`, with the same checks as the 2026-09-21 replacement.
3. The first build goes to its own `--out` folder for comparison with the canonical build.
4. The canonical build switches only after Federico reviews the comparison.

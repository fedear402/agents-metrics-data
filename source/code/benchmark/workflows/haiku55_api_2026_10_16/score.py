"""Score a Haiku 5.5 API-key collection with the paper rule and compare it with the paper runs. No model call.

Rule file: quality_reports/decision_records/2026-10-16_haiku55-api-collection.md.
Scoring: code/paper/dataset.py score_output with the default rule channel_se and the paper catalog entry,
the same call that the paper build makes for each selected run. Failure categories follow the paper build.
Cost: each call's usage priced by token type at the Haiku 5.5 list prices (prospective_collect_haiku_api.py),
as code/benchmark/production_paper_pricing.py prices Sonnet. Side-model calls are left out, as in the paper.
Comparison: the paper's Sonnet 4.6 and Luna runs of the same cells (ten repetitions each).

Outputs: data/collections/2026-10-16_haiku55-api/<batch>_scores.csv and
quality_reports/audits/2026-10-16_haiku55-api/paper_runtime_<task>.md.

Run from the repository root:
    uv run --no-sync python code/benchmark/workflows/haiku55_api_2026_10_16/score.py --manifest M
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import prospective_collect as c  # noqa: E402  (helpers only: job_id, TERMINAL)

META = ROOT / 'data/collections/2026-10-16_haiku55-api'
AUDIT = ROOT / 'quality_reports/audits/2026-10-16_haiku55-api'
CATALOG = ROOT / 'data/analysis/main/data/catalog.json'
RUNS = ROOT / 'data/analysis/main/data/runs.csv'
PRICING = ROOT / 'data/analysis/main/data/run_pricing.csv'
# USD per million tokens, prompts up to 100,000 tokens (platform.claude.com pricing page, read 2026-10-16).
HAIKU = dict(input=0.10, cache_write_5m=0.125, cache_write_1h=0.20, cache_read=0.01, output=0.50)
CELL = ['platform', 'prompt', 'condition']


def load_dataset():
    spec = importlib.util.spec_from_file_location('paper_dataset_haiku', ROOT / 'code/paper/dataset.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record_path(folder):
    done = [p for p in sorted(folder.glob('attempt_*/run_record.json'))
            if json.loads(p.read_text())['status'] in c.TERMINAL]
    if len(done) != 1:
        raise ValueError(f'{folder.name}: {len(done)} completed records')
    return done[0]


def haiku_cost(record):
    cost, out = 0.0, 0
    for call in record.get('calls') or []:
        if call.get('quota_rejected_before_response'):
            continue
        usage = call.get('usage') or {}
        split = usage.get('cache_creation') or {}
        five, hour = split.get('ephemeral_5m_input_tokens', 0), split.get('ephemeral_1h_input_tokens', 0)
        writes = usage.get('cache_creation_input_tokens', 0)
        if five + hour != writes:
            raise ValueError('Cache write split does not add up')
        cost += (usage.get('input_tokens', 0) * HAIKU['input'] + five * HAIKU['cache_write_5m']
                 + hour * HAIKU['cache_write_1h'] + usage.get('cache_read_input_tokens', 0) * HAIKU['cache_read']
                 + usage.get('output_tokens', 0) * HAIKU['output']) / 1e6
        out += usage.get('output_tokens', 0)
    return cost, out


def paper_cells(task):
    runs = pd.read_csv(RUNS, low_memory=False)
    runs = runs[runs.task == task]
    prices = pd.read_csv(PRICING, usecols=['record_path', 'cost_usd'])
    runs = runs.merge(prices, on='record_path', how='left')
    cells = runs.groupby(['model'] + CELL).agg(success=('primary', 'mean'), cost=('cost_usd', 'mean'),
                                               seconds=('provider_seconds', 'mean')).reset_index()
    return {m: g.set_index(CELL) for m, g in cells.groupby('model')}, runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    HAIKU.update(manifest['protocol']['policy'].get('haiku_cost_per_million') or {})
    batch = manifest['jobs'][0]['block']
    run_root = manifest_path.parent / 'runs' / manifest['protocol_sha256']
    dataset = load_dataset()
    runtime = dataset.analysis.runtime.load_runtime(Path(tempfile.mkdtemp(prefix='haiku_score_'))).c
    catalog = json.loads(CATALOG.read_text())
    rows = []
    for job in manifest['jobs']:
        path = record_path(run_root / c.job_id(job))
        record = json.loads(path.read_text())
        output = dataset.analysis.output_path(path, record, job['platform'])
        scored = dataset.score_output(output, path, record, job['platform'], catalog[job['task']], runtime)
        executions = record.get('executions') or []
        execution_ok = bool(executions) and bool(executions[-1].get('ok'))
        cost, output_tokens = haiku_cost(record)
        passed = scored['passed']
        rows.append(dict(task=job['task'], platform=job['platform'], prompt=job['prompt'], condition=job['condition'],
                         global_rep=job['global_rep'], primary=bool(passed), execution_ok=execution_ok,
                         output_available=scored['delivered'],
                         failure_category='Success' if passed else 'Silent failure'
                         if execution_ok and scored['delivered'] else 'Loud failure',
                         scoring_notes=' | '.join(scored['notes']), calls=len(record.get('calls') or []),
                         provider_seconds=sum(call.get('wall_seconds') or 0 for call in record.get('calls') or []),
                         output_tokens=output_tokens, cost_usd=cost,
                         record_path=str(path.relative_to(ROOT))))
    frame = pd.DataFrame(rows).sort_values(CELL)
    frame.to_csv(META / f'{batch}_scores.csv', index=False)
    task = frame.task.iloc[0]
    paper, paper_runs = paper_cells(task)
    lines = [f'# Claude model on the paper runtime: {task}', '',
             f'Batch `{batch}`, protocol `{manifest["protocol_sha256"][:12]}`. Scored with the paper rule '
             '`channel_se`. Paper columns: share of the ten paper repetitions that pass.', '',
             '| Language | Prompt | Mode | Haiku 5.5 | Sonnet 4.6 (paper) | Luna (paper) | Haiku USD | Sonnet USD |',
             '|---|---|---|---|---:|---:|---:|---:|']
    for row in frame.itertuples():
        key = (row.platform, row.prompt, row.condition)
        s, l = paper['sonnet'].loc[key], paper['luna'].loc[key]
        result = 'pass' if row.primary else row.failure_category.lower()
        lines.append(f'| {row.platform} | {row.prompt} | {row.condition} | {result} | {100 * s.success:.0f}% | '
                     f'{100 * l.success:.0f}% | {row.cost_usd:.4f} | {s.cost:.4f} |')
    totals = dict(haiku_success=frame.primary.mean(), sonnet_success=paper_runs[paper_runs.model == 'sonnet'].primary.mean(),
                  luna_success=paper_runs[paper_runs.model == 'luna'].primary.mean(),
                  haiku_cost_total=frame.cost_usd.sum(), haiku_cost_mean=frame.cost_usd.mean(),
                  sonnet_cost_mean=paper_runs[paper_runs.model == 'sonnet'].cost_usd.mean(),
                  luna_cost_mean=paper_runs[paper_runs.model == 'luna'].cost_usd.mean(),
                  haiku_seconds_mean=frame.provider_seconds.mean(),
                  sonnet_seconds_mean=paper_runs[paper_runs.model == 'sonnet'].provider_seconds.mean())
    lines += ['', f'- Success: Haiku 5.5 {frame.primary.sum()} of {len(frame)} ({100 * totals["haiku_success"]:.1f} percent). '
              f'Paper: Sonnet 4.6 {100 * totals["sonnet_success"]:.1f} percent, Luna {100 * totals["luna_success"]:.1f} percent.',
              f'- Cost per run: Haiku 5.5 USD {totals["haiku_cost_mean"]:.4f} (total USD {totals["haiku_cost_total"]:.4f}). '
              f'Paper: Sonnet 4.6 USD {totals["sonnet_cost_mean"]:.4f}, Luna USD {totals["luna_cost_mean"]:.4f}.',
              f'- Model time per run: Haiku 5.5 {totals["haiku_seconds_mean"]:.1f} s. Paper: Sonnet 4.6 '
              f'{totals["sonnet_seconds_mean"]:.1f} s.',
              f'- Failure categories: {frame.failure_category.value_counts().to_dict()}.', '']
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / (f'paper_runtime_{task}.md' if batch.startswith('haiku55_') else f'paper_runtime_{batch}.md')).write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(json.dumps({k: round(float(v), 5) for k, v in totals.items()}))


if __name__ == '__main__':
    main()

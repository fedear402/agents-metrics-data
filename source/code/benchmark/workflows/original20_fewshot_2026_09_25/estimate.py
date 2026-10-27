"""Estimate the cost and run time of the rerun scopes from saved runs.

The program reads data/analysis/main/data/runs.csv, run_pricing.csv, and the
selected raw records. It makes no model call and writes nothing unless --out
is given. All outputs are estimates from earlier runs of the same tasks.

Task groups: the 20 original tasks, the 12 added tasks with new blocks, and the
8 added tasks with the full layout. Each group uses its own means.

Cost:
- current: the priced cost of the selected runs in the positions to replace.
- runtime-adjusted: the mean cost per run of the selected prospective-runtime
  runs of the group, by model, agency, language, and prompt, times the planned
  count. Early-harness runs are left out of this mean.

Time:
- Stata lock hours: the collector holds one global Stata lock for a whole Stata
  agent session and for each Stata script execution. The estimate adds the mean
  provider plus execution seconds of the prospective Stata agent runs and the
  execution seconds of Stata chatbot and self-repair runs.
- Python and R worker hours: mean provider plus execution seconds times counts,
  divided by the chosen 32 workers.

Listed positions (flagged-run scopes) are estimated when the run-list file exists.

  uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/estimate.py
  uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/estimate.py --run-list <file>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import original20_plan as p  # noqa: E402

PRICING = p.ROOT / 'data/analysis/main/data/run_pricing.csv'
KEY = ['task', 'model', 'platform', 'prompt', 'condition', 'global_rep']
CELL = ['model', 'condition', 'platform', 'prompt']
PERMITS = (1, 4)  # One Stata permit is the collector default. Four is the tested limit of the 2026-09-21 adapter.
GROUPS = {'original20': p.ORIGINAL_TASKS, 'added12': p.ADDED12_TASKS, 'full_layout8': p.FULL_LAYOUT_TASKS}
WORKERS = p.WORKERS * len(p.MODELS) * (p.REPS // p.BLOCK_REPS)


def load(tasks=p.ALL_TASKS):
    runs = pd.read_csv(p.ANALYSIS_RUNS, low_memory=False)
    runs = runs[runs.task.isin(tasks)]
    pricing = pd.read_csv(PRICING)
    data = runs[KEY + ['record_path', 'provider_seconds', 'execution_seconds']].merge(
        pricing[KEY + ['cost_usd']], on=KEY, how='left')
    data['regime'] = [('prospective' if json.loads(Path(path).read_text()).get('protocol_sha256') else 'early')
                      for path in data.record_path]
    data['busy'] = data.provider_seconds.fillna(0) + data.execution_seconds.fillna(0)
    data['group'] = data.task.map({task: group for group, tasks in GROUPS.items() for task in tasks})
    return data


def part(data, selected):
    """Additive estimates of the selected rows, with the means of each row's task group."""
    out = dict(positions=int(len(selected)), current=0.0, adjusted=0.0, stata_seconds=0.0, other_seconds=0.0,
               by_model={model: dict(current=0.0, runtime_adjusted=0.0) for model in p.MODELS})
    for group, rows in selected.groupby('group'):
        prospective = data[(data.group == group) & (data.regime == 'prospective')]
        per_cell = prospective.groupby(CELL).cost_usd.mean()
        planned = rows.groupby(CELL).size()
        if per_cell.reindex(planned.index).isna().any():
            raise ValueError(f'A planned cell of {group} has no prospective-runtime run to price')
        adjusted = per_cell.reindex(planned.index) * planned
        out['current'] += float(rows.cost_usd.sum())
        out['adjusted'] += float(adjusted.sum())
        for model in p.MODELS:
            out['by_model'][model]['current'] += float(rows[rows.model == model].cost_usd.sum())
            if model in adjusted.index.get_level_values('model'):
                out['by_model'][model]['runtime_adjusted'] += float(adjusted.xs(model, level='model').sum())
        busy = prospective.groupby(['model', 'condition', 'platform']).busy.mean()
        execution = prospective.groupby(['model', 'condition', 'platform']).execution_seconds.mean()
        for (model, condition, platform), n in rows.groupby(['model', 'condition', 'platform']).size().items():
            if platform == 'stata' and condition == 'agent':
                out['stata_seconds'] += n * busy[(model, condition, platform)]
            elif platform == 'stata':
                out['stata_seconds'] += n * execution[(model, condition, platform)]
            else:
                out['other_seconds'] += n * busy[(model, condition, platform)]
    return out


def summarize(parts, data):
    """Add the parts. data holds the runs of the tasks whose Stata agent sessions describe the scope."""
    total = dict(positions=0, current=0.0, adjusted=0.0, stata_seconds=0.0, other_seconds=0.0,
                 by_model={model: dict(current=0.0, runtime_adjusted=0.0) for model in p.MODELS})
    for item in parts:
        for key in ('positions', 'current', 'adjusted', 'stata_seconds', 'other_seconds'):
            total[key] += item[key]
        for model in p.MODELS:
            for key in ('current', 'runtime_adjusted'):
                total['by_model'][model][key] += item['by_model'][model][key]
    stata, other = total['stata_seconds'], total['other_seconds']
    agent = data[(data.platform == 'stata') & (data.condition == 'agent') & (data.regime == 'prospective')].busy
    return dict(
        positions=total['positions'],
        cost_usd=dict(current=round(total['current'], 2), runtime_adjusted=round(total['adjusted'], 2),
                      by_model={model: {k: round(v, 2) for k, v in values.items()}
                                for model, values in total['by_model'].items()}),
        stata_lock_hours=round(stata / 3600, 2),
        python_r_worker_hours=round(other / 3600, 2),
        python_r_hours_with_workers=round(other / 3600 / WORKERS, 2),
        wall_hours_by_stata_permits={str(k): round(max(stata / k, other / WORKERS) / 3600, 1) for k in PERMITS},
        stata_agent_session_seconds=dict(mean=round(float(agent.mean()), 1), p90=round(float(agent.quantile(0.9)), 1),
                                         max=round(float(agent.max()), 1)),
        workers=WORKERS)


def estimate(data, prompts, tasks):
    """Estimate for all positions of the tasks and prompts (kept for the 2026-09-25 scopes)."""
    selected = data[data.task.isin(tasks) & data.prompt.isin(prompts)]
    return summarize([part(data, selected)], data[data.task.isin(tasks)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-list', type=Path, default=None, help=f'run-list file (default: {p.RUN_LIST})')
    parser.add_argument('--out', type=Path, help='save the estimates as JSON at this path')
    args = parser.parse_args(argv)
    data = load()
    both = ('few_shot', 'zero_shot')
    original = data[data.task.isin(p.ORIGINAL_TASKS)]
    result = dict(note='Estimates from saved runs of the same tasks. Not measured for the new examples.',
                  few_shot=estimate(data, ('few_shot',), p.ORIGINAL_TASKS),
                  zero_shot=estimate(data, ('zero_shot',), p.ORIGINAL_TASKS),
                  few_shot_and_zero_shot=estimate(data, both, p.ORIGINAL_TASKS),
                  added12_both_prompts=estimate(data, both, p.ADDED12_TASKS),
                  original20_added12_both_prompts=estimate(data, both, p.RERUN32_TASKS))
    positions, sha256, problems = p.read_run_list(args.run_list)
    listed = dict(path=str(args.run_list or p.RUN_LIST), sha256=sha256, positions=len(positions), problems=problems)
    if positions:
        ids = data.task + '__' + data.model + '__' + data.platform + '__' + data.prompt + '__' + data.condition \
            + '__' + data.global_rep.astype(str)
        flagged = data[ids.isin(positions)]
        extra = flagged[~flagged.task.isin(p.RERUN32_TASKS)]
        rerun32 = data[data.task.isin(p.RERUN32_TASKS)]
        result['flagged_runs'] = summarize([part(data, flagged)], data[data.task.isin(set(flagged.task))])
        result['original20_added12_both_prompts_and_flagged_runs'] = summarize(
            [part(data, rerun32), part(data, extra)], data[data.task.isin(set(rerun32.task) | set(extra.task))])
    result['run_list'] = listed
    regimes = original.groupby(['prompt', 'model', 'condition', 'regime']).size().unstack(fill_value=0)
    result['selected_runs_by_regime_original20'] = {f'{a}/{b}/{c_}': row.to_dict() for (a, b, c_), row in regimes.iterrows()}
    print(json.dumps(result, indent=1))
    if args.out:
        p.c.save(args.out, result)
    return 0


if __name__ == '__main__':
    sys.exit(main())

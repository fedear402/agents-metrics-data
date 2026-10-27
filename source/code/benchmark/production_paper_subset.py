"""Export Figure 4 for tasks ranked by lower success in the fixed paper sample."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.text import Text
import pandas as pd

from production_paper_assets import CELL, DEFAULT_OUT, SHORT_LABELS, load_data
from production_paper_costs import build_resource_figure, resource_points
import paper_figure_style as style


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(paper, task_count):
    paper = Path(paper).resolve()
    source = paper / 'data/runs.csv'
    source_hash = sha(source)
    data = load_data(source)
    if not 1 <= task_count < data.task.nunique():
        raise ValueError('Choose fewer tasks than the full paper sample')
    cells = data.groupby(CELL).primary.mean().reset_index(name='success')
    assert cells.groupby('task').size().eq(36).all()
    ranking = cells.groupby('task').success.mean().mul(100).reset_index(name='success_pct')
    ranking = ranking.sort_values(['success_pct', 'task'], kind='stable').reset_index(drop=True)
    ranking.insert(0, 'rank', range(1, len(ranking) + 1))
    ranking['task_label'] = ranking.task.map(SHORT_LABELS)
    ranking['selected'] = ranking['rank'].le(task_count)
    selected_tasks = ranking[ranking.selected].task.tolist()

    pricing_path = paper / 'data/run_pricing.csv'
    prices = pd.read_csv(pricing_path)
    priced = data.merge(prices[['record_path', 'record_sha256', 'cost_usd']],
        on=['record_path', 'record_sha256'], validate='one_to_one', how='left', indicator=True)
    assert priced['_merge'].eq('both').all()
    subset = priced[priced.task.isin(selected_tasks)].copy()
    assert subset.task.nunique() == task_count
    assert subset.groupby(CELL).ngroups == task_count * 36

    out = paper / f'worst{task_count}'
    out.mkdir(exist_ok=True)
    fig, points = build_resource_figure(subset)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(match=Text):
        if item.get_visible() and item.get_text().strip():
            box = item.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, item.get_text()
    target = out / 'cost_of_correctness.pdf'
    with style.figure_style():
        fig.savefig(target, metadata={'Title': f'Cost and success, {task_count} lower-success tasks'})
    plt.close(fig)
    points.to_csv(out / 'cost_of_correctness.csv', index=False)
    ranking.to_csv(out / 'task_ranking.csv', index=False)
    full = resource_points(priced)
    comparison = full[['model', 'platform', 'agency', 'success_pct', 'x_value']].merge(
        points[['model', 'platform', 'agency', 'success_pct', 'x_value']],
        on=['model', 'platform', 'agency'], suffixes=('_all_tasks', '_subset'), validate='one_to_one')
    comparison.to_csv(out / 'comparison.csv', index=False)
    receipt = dict(tasks=selected_tasks, rows=len(subset), full_rows=len(data),
        ranking='Mean primary success with equal weight on all 36 model-language-prompt-agency cells; ties use task id.',
        figure_weighting='Same as main Figure 4: available responses pooled; cost and success use the same records with observed cost.',
        interpretation='Exploratory selection on observed outcomes. This does not select or validate a final task bank.',
        source=str(source.relative_to(paper)), source_sha256=source_hash,
        pricing=str(pricing_path.relative_to(paper)), pricing_sha256=sha(pricing_path),
        figure_sha256=sha(target), provider_calls=0, main_paper_modified=False)
    (out / 'selection.json').write_text(json.dumps(receipt, indent=2) + '\n')
    task_list = ', '.join(ranking[ranking.selected].task_label)
    (out / 'README.md').write_text(
        f'# Figure 4 with {task_count} lower-success tasks\n\n'
        f'Open [the figure](cost_of_correctness.pdf). This subset contains {len(subset):,} responses from the fixed {data.task.nunique()}-task paper sample. '
        'The two model panels show costs in cents, using the same layout, markers, pricing, and scoring as the main figure.\n\n'
        'Rank tasks by average success with equal weight on all 36 design cells. Select the lowest rates across both models, all languages, both prompts, and all agency conditions. '
        'The ranking uses the same observed data as the figure, so this comparison is exploratory. Within each plotted group, pool available responses as in the main figure.\n\n'
        f'Selected tasks: {task_list}.\n\n'
        '[Task ranking](task_ranking.csv), [plotted data](cost_of_correctness.csv), [comparison with all tasks](comparison.csv), and [selection receipt](selection.json). '
        'The main paper and original responses remain unchanged.\n\n'
        f'Rebuild from the project root: `uv run --no-sync python code/benchmark/production_paper_subset.py --paper {paper} --tasks {task_count}`.\n')
    assert sha(source) == source_hash
    print(comparison[comparison.agency.eq('self_repair')].to_string(index=False))
    print(json.dumps(dict(output=str(target), tasks=selected_tasks, rows=len(subset))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--tasks', type=int, default=10)
    args = parser.parse_args()
    build(args.paper, args.tasks)

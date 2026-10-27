"""Recompute the paper's main estimates from the data repository (copied there as code/reproduce.py).

Run from the root of the data repository: python code/reproduce.py
For a local analysis directory: python code/reproduce.py --analysis PATH

1. Table 1 contrasts: OLS of task success on task fixed effects (model-by-task in the pooled column) and the 18
   language-prompt-agency cells, with CR1 standard errors clustered by task. Each contrast is compared with
   analysis/regression_contrasts.csv.
2. Task success, loud and silent failures per 100 runs, and mean cost per run, by agency condition.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA = Path(__file__).resolve().parents[1] / 'analysis'
LANGUAGES = ('python', 'r', 'stata')
PROMPTS = ('zero_shot', 'few_shot')
AGENCIES = ('chatbot', 'self_repair', 'agent')
CELLS = list(product(LANGUAGES, PROMPTS, AGENCIES))  # the first cell (python, zero_shot, chatbot) is the base


def weights() -> dict[str, np.ndarray]:
    """Each contrast as weights on the 18 cell means."""
    def w(pick):
        return np.array([pick(*cell) for cell in CELLS], dtype=float)

    def avg(agency=None, prompt=None, language=None):
        cells = [c for c in CELLS if (language is None or c[0] == language) and (prompt is None or c[1] == prompt)
                 and (agency is None or c[2] == agency)]
        return w(lambda *c: 1 / len(cells) if c in cells else 0)

    def one(language, prompt, agency):
        return w(lambda *c: 1.0 if c == (language, prompt, agency) else 0)

    gain = {a: avg(agency=a, prompt='few_shot') - avg(agency=a, prompt='zero_shot') for a in AGENCIES}
    result = {
        'H3_self_repair': avg(agency='self_repair') - avg(agency='chatbot'),
        'H3_tool_loop': avg(agency='agent') - avg(agency='chatbot'),
        'H2_fewshot': avg(prompt='few_shot') - avg(prompt='zero_shot'),
        'H1_r': one('r', 'zero_shot', 'chatbot') - one('python', 'zero_shot', 'chatbot'),
        'H1_stata': one('stata', 'zero_shot', 'chatbot') - one('python', 'zero_shot', 'chatbot'),
        'GAP_stata_avg': avg(language='stata') - avg(language='python'),
        'GAP_stata_repair_zs': one('stata', 'zero_shot', 'self_repair') - one('python', 'zero_shot', 'self_repair'),
        'GAP_stata_agent_zs': one('stata', 'zero_shot', 'agent') - one('python', 'zero_shot', 'agent'),
        'H4_self_repair': gain['self_repair'] - gain['chatbot'],
        'H4_tool_loop': gain['agent'] - gain['chatbot'],
        'AUX_fewshot_chatbot': gain['chatbot'],
        'AUX_fewshot_self_repair': gain['self_repair'],
        'AUX_fewshot_agent': gain['agent'],
        'AUX_agent_self_repair_zs': avg(agency='agent', prompt='zero_shot') - avg(agency='self_repair', prompt='zero_shot'),
        'AUX_agent_self_repair_fs': avg(agency='agent', prompt='few_shot') - avg(agency='self_repair', prompt='few_shot'),
    }
    return result


def contrasts(runs: pd.DataFrame, model: str, members=None) -> pd.DataFrame:
    members = list(members) if members is not None else (list(runs.model.unique()) if model == 'pooled' else [model])
    frame = runs[runs.model.isin(members)]
    if frame.empty:
        raise ValueError(f'No runs for contrast column {model}')
    frame = frame.assign(success=frame.primary.astype(float),
                         cell=[CELLS.index(c) for c in zip(frame.platform, frame.prompt, frame.agency)],
                         fe=frame.task if len(members) == 1 else frame.model + ':' + frame.task)
    fit = smf.ols('success ~ C(fe) + C(cell, Treatment(0))', frame).fit()
    clusters, tasks = pd.factorize(frame.task)
    n, k = fit.model.exog.shape
    if len(tasks) < 2 or n <= k:
        raise ValueError('Contrast model needs two tasks and positive residual degrees of freedom')
    correction = len(tasks) / (len(tasks) - 1) * (n - 1) / (n - k)
    names = [f'C(cell, Treatment(0))[T.{k}]' for k in range(1, len(CELLS))]
    rows = []
    for contrast, v in weights().items():
        restriction = np.zeros(len(fit.params))
        for k, name in enumerate(names, 1):
            restriction[fit.params.index.get_loc(name)] = v[k]
        # Sum squared cluster scores. This avoids a negative variance from cancellation in a zero contrast.
        influence = fit.model.exog @ (fit.normalized_cov_params.to_numpy() @ restriction)
        scores = np.bincount(clusters, weights=influence * fit.resid.to_numpy())
        standard_error = np.sqrt(correction * np.dot(scores, scores))
        rows.append({'model': model, 'contrast': contrast,
                     'estimate_pp': 100 * float(restriction @ fit.params),
                     'se_cr1_pp': 100 * float(standard_error)})
    return pd.DataFrame(rows)


def contrast_groups(runs: pd.DataFrame, paper: pd.DataFrame) -> dict[str, list[str]]:
    """Read optional group membership. Always require each model and the pooled column."""
    models = list(runs.model.unique())
    groups = {m: [m] for m in models}
    groups['pooled'] = models
    if paper[['model', 'contrast']].isna().any().any() or paper.duplicated(['model', 'contrast']).any():
        raise ValueError('Missing or duplicate contrast keys')
    if not set(groups).issubset(set(paper.model)):
        raise ValueError('Missing model or pooled contrast columns')
    for name, rows in paper.groupby('model', sort=False):
        if 'member_models' in rows:
            values = rows.member_models.dropna().unique()
            if len(values) != 1 or rows.member_models.isna().any():
                raise ValueError(f'Inconsistent contrast membership for {name}')
            members = [m.strip() for m in str(values[0]).split(',')]
            if not members or len(set(members)) != len(members) or not set(members).issubset(models):
                raise ValueError(f'Unknown or duplicate contrast members for {name}')
            if name in groups and set(members) != set(groups[name]):
                raise ValueError(f'Incorrect contrast membership for {name}')
            groups[name] = members
        elif name not in groups:
            raise ValueError(f'Unknown contrast column {name}; member_models is required')
        if set(rows.contrast) != set(weights()):
            raise ValueError(f'Missing or extra contrasts for {name}')
    return groups


def verify_contrasts(runs: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    groups = contrast_groups(runs, paper)
    ours = pd.concat([contrasts(runs, name, members) for name, members in groups.items()], ignore_index=True)
    both = ours.merge(paper[['model', 'contrast', 'estimate_pp', 'se_cr1_pp']], on=['model', 'contrast'],
                      suffixes=('', '_paper'), how='outer', validate='one_to_one', indicator=True)
    both['match'] = (both._merge.eq('both')
                     & np.isclose(both.estimate_pp, both.estimate_pp_paper, atol=1e-8, rtol=0)
                     & np.isclose(both.se_cr1_pp, both.se_cr1_pp_paper, atol=1e-6, rtol=0))
    if not both.match.all():
        raise ValueError('Some contrasts differ from analysis/regression_contrasts.csv')
    return both.drop(columns='_merge')


def priced_runs(runs: pd.DataFrame, pricing: pd.DataFrame) -> pd.DataFrame:
    """Keep every accuracy run. Leave unpriced costs missing in cost means."""
    if runs.record_path.isna().any() or runs.record_path.duplicated().any():
        raise ValueError('Missing or duplicate run record paths')
    if pricing.record_path.isna().any() or pricing.record_path.duplicated().any():
        raise ValueError('Missing or duplicate pricing record paths')
    if set(runs.record_path) != set(pricing.record_path):
        raise ValueError('Pricing coverage differs from runs.csv')
    costs = pd.to_numeric(pricing.cost_usd, errors='raise')
    if ((costs.dropna() < 0) | ~np.isfinite(costs.dropna())).any():
        raise ValueError('Costs must be nonnegative finite values or missing')
    prices = pricing[['record_path']].assign(cost_usd=costs)
    return runs.drop(columns='cost_usd', errors='ignore').merge(prices, on='record_path', validate='one_to_one')


def rates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.groupby('agency').agg(
        success=('primary', 'mean'),
        loud=('failure_category', lambda s: s.eq('Loud failure').mean()),
        silent=('failure_category', lambda s: s.eq('Silent failure').mean()),
        mean_cost_usd=('cost_usd', 'mean'),
        runs=('record_path', 'size'),
        priced_runs=('cost_usd', 'count')).reindex(list(AGENCIES))
    result[['success', 'loud', 'silent']] *= 100
    return result


def main(analysis: Path | None = None) -> None:
    data = analysis or DATA
    if (data / 'data' / 'runs.csv').is_file():
        runs = pd.read_csv(data / 'data' / 'runs.csv', low_memory=False)
        pricing = pd.read_csv(data / 'data' / 'run_pricing.csv', low_memory=False)
        paper = pd.read_csv(data / 'tables' / 'regression_contrasts.csv')
    else:
        runs = pd.read_csv(data / 'runs.csv', low_memory=False)
        pricing = pd.read_csv(data / 'run_pricing.csv', low_memory=False)
        paper = pd.read_csv(data / 'regression_contrasts.csv')
    runs = priced_runs(runs, pricing)
    both = verify_contrasts(runs, paper)
    print(both.round(4).to_string(index=False))
    for name, frame in [('all runs', runs), *[(m, g) for m, g in runs.groupby('model')]]:
        summary = rates(frame)
        print(f'\n{name} (success in percent, loud and silent failures per 100 runs)')
        print(summary.round(4).to_string())
        print('mean cost per priced run, agent / chatbot:',
              round(summary.mean_cost_usd['agent'] / summary.mean_cost_usd['chatbot'], 2))
    print('\nAll', len(both), 'contrasts match analysis/regression_contrasts.csv.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=DATA,
                        help='exported analysis/ folder or local data/analysis/<selection> directory')
    args = parser.parse_args()
    main(args.analysis)

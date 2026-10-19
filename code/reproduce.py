"""Recompute the paper's main estimates from the data repository (copied there as code/reproduce.py).

Run from the root of the data repository: python code/reproduce.py

1. Table 1 contrasts: OLS of task success on task fixed effects (model-by-task in the pooled column) and the 18
   language-prompt-agency cells, with CR1 standard errors clustered by task. Each contrast is compared with
   analysis/regression_contrasts.csv.
2. Task success, loud and silent failures per 100 runs, and mean cost per run, by agency condition.
"""
from __future__ import annotations

import sys
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
    return {
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
    }


def contrasts(runs: pd.DataFrame, model: str) -> pd.DataFrame:
    frame = runs if model == 'pooled' else runs[runs.model.eq(model)]
    frame = frame.assign(success=frame.primary.astype(float),
                         cell=[CELLS.index(c) for c in zip(frame.platform, frame.prompt, frame.agency)],
                         fe=frame.task if model != 'pooled' else frame.model + ':' + frame.task)
    fit = smf.ols('success ~ C(fe) + C(cell, Treatment(0))', frame).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(frame.task)[0]})
    names = [f'C(cell, Treatment(0))[T.{k}]' for k in range(1, len(CELLS))]
    rows = []
    for contrast, v in weights().items():
        restriction = np.zeros(len(fit.params))
        for k, name in enumerate(names, 1):
            restriction[fit.params.index.get_loc(name)] = v[k]
        test = fit.t_test(restriction)
        rows.append({'model': model, 'contrast': contrast, 'estimate_pp': 100 * float(test.effect[0]),
                     'se_cr1_pp': 100 * float(test.sd[0][0])})
    return pd.DataFrame(rows)


def main() -> None:
    runs = pd.read_csv(DATA / 'runs.csv')
    pricing = pd.read_csv(DATA / 'run_pricing.csv')
    ours = pd.concat([contrasts(runs, m) for m in ('sonnet', 'luna', 'pooled')])
    paper = pd.read_csv(DATA / 'regression_contrasts.csv')
    both = ours.merge(paper[['model', 'contrast', 'estimate_pp', 'se_cr1_pp']], on=['model', 'contrast'],
                      suffixes=('', '_paper'))
    both['match'] = (np.isclose(both.estimate_pp, both.estimate_pp_paper, atol=1e-8)
                     & np.isclose(both.se_cr1_pp, both.se_cr1_pp_paper, atol=1e-6))
    print(both.round(4).to_string(index=False))
    runs = runs.merge(pricing[['record_path', 'cost_usd']], on='record_path', how='left', validate='one_to_one')
    for name, frame in [('all runs', runs), *[(m, g) for m, g in runs.groupby('model')]]:
        rates = frame.groupby('agency').agg(
            success=('primary', 'mean'),
            loud=('failure_category', lambda s: s.eq('Loud failure').mean()),
            silent=('failure_category', lambda s: s.eq('Silent failure').mean()),
            mean_cost_usd=('cost_usd', 'mean')).reindex(list(AGENCIES))
        rates[['success', 'loud', 'silent']] *= 100
        print(f'\n{name} (success in percent, loud and silent failures per 100 runs)')
        print(rates.round(4).to_string())
        print('mean cost per run, agent / chatbot:', round(rates.mean_cost_usd['agent'] / rates.mean_cost_usd['chatbot'], 2))
    if not both.match.all() or len(both) != len(ours):
        sys.exit('Some contrasts differ from analysis/regression_contrasts.csv')
    print('\nAll', len(both), 'contrasts match analysis/regression_contrasts.csv.')


if __name__ == '__main__':
    main()

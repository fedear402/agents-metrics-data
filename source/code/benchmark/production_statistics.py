"""Supply regression helpers for the current main paper.

The retained main() function documents the historical 18-task analysis.
Direct execution is disabled because its destinations have been archived.
Use code/paper/rebuild.py to regenerate the current paper.
"""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'quality_reports/reports/2026-09-17_complete-checkpoint-package'
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import factorial_analysis as fa

LABELS = {
    'H1_r': 'R versus Python, zero-shot chatbot',
    'H1_stata': 'Stata versus Python, zero-shot chatbot',
    'GAP_stata_avg': 'Stata versus Python, design average',
    'GAP_stata_agent_zs': 'Stata versus Python, zero-shot agent',
    'H2_fewshot': 'Few-shot versus zero-shot, design average',
    'H3_self_repair': 'Self-repair versus chatbot',
    'H3_tool_loop': 'Agent versus chatbot',
    'H4_tool_loop': 'Few-shot x agent',
    'H4_self_repair': 'Few-shot x self-repair',
    'AUX_fewshot_chatbot': 'Few-shot effect, chatbot',
    'AUX_fewshot_self_repair': 'Few-shot effect, self-repair',
    'AUX_fewshot_agent': 'Few-shot effect, agent',
    'AUX_agent_self_repair_zs': 'Agent versus self-repair, zero-shot',
    'AUX_agent_self_repair_fs': 'Agent versus self-repair, few-shot',
    'GAP_stata_repair_zs': 'Stata versus Python, zero-shot self-repair',
}


def weights():
    w = fa.contrast_weights()
    w['AUX_fewshot_self_repair'] = w['AUX_fewshot_chatbot'] + w['H4_self_repair']
    w['AUX_fewshot_agent'] = w['AUX_fewshot_chatbot'] + w['H4_tool_loop']
    for prompt, suffix in [('zero_shot', 'zs'), ('few_shot', 'fs')]:
        v = np.zeros(18)
        for platform in fa.PLATFORMS:
            v[fa.cell_index(platform, prompt, 'tool_loop') - 1] += 1 / 3
            v[fa.cell_index(platform, prompt, 'self_repair') - 1] -= 1 / 3
        w['AUX_agent_self_repair_' + suffix] = v
    return w


def wild_bootstrap(frame, ws, contrasts, work, pooled):
    work.mkdir(parents=True, exist_ok=True)
    cached = work / 'bootstrap_results.json'
    csv_text = frame[['success', 'task', 'cell', 'armtask']].to_csv(index=False)
    if cached.is_file() and (work / 'input.csv').read_text() == csv_text:
        stored = json.loads(cached.read_text())
        if all(name in stored and abs(stored[name]['estimate_stata'] - contrasts[name]['estimate']) < 1e-8 for name in ws):
            return stored
    frame[['success', 'task', 'cell', 'armtask']].to_csv(work / 'input.csv', index=False)
    lines = ['clear all', 'set more off', 'set linesize 255',
             f'import delimited "{work / "input.csv"}", clear stringcols(2 4)',
             'encode task, gen(taskn)', 'encode armtask, gen(fen)',
             f'reg success i.{"fen" if pooled else "taskn"} ib1.cell, cluster(taskn)']
    for name, vector in ws.items():
        expr = fa.stata_expression(vector)
        lines += [f'lincom {expr}', f'di "PKGEST {name} " %21.15g r(estimate) " " %21.15g r(se)']
        if contrasts[name]['se_cr1'] > 1e-10:
            lines += [f'capture noisily boottest {expr}=0, reps(9999) weighttype(rademacher) seed(20260701) nograph noci',
                      'if _rc == 0 {',
                      f'di "PKGBOOT {name} " %21.15g r(p) " " %21.15g r(reps) " " %21.15g r(repsFeas)', '}',
                      'else {', f'di "PKGERROR {name} " _rc', '}']
    for name, members in {'H1_joint': ['H1_r', 'H1_stata'], 'H4_joint': ['H4_self_repair', 'H4_tool_loop']}.items():
        expr = ' '.join('(' + fa.stata_expression(ws[m]) + '=0)' for m in members)
        lines += [f'capture noisily boottest {expr}, reps(9999) weighttype(rademacher) seed(20260701) nograph noci',
                  'if _rc == 0 {', f'di "PKGBOOT {name} " %21.15g r(p) " " %21.15g r(reps) " " %21.15g r(repsFeas)', '}',
                  'else {', f'di "PKGERROR {name} " _rc', '}']
    dofile = work / 'inference.do'
    dofile.write_text('\n'.join(lines) + '\n')
    subprocess.run([fa.STATA_BIN, '-b', 'do', str(dofile)], cwd=work, check=True, timeout=1800)
    log = (work / 'inference.log').read_text()
    result = {}
    for line in log.splitlines():
        parts = line.strip().split()
        if parts and parts[0] == 'PKGEST':
            result.setdefault(parts[1], {}).update(estimate_stata=float(parts[2]),
                se_stata=float(parts[3]) if parts[3] != '.' else None)
        elif parts and parts[0] == 'PKGBOOT':
            p, requested, feasible = [float(x) if x != '.' else float('nan') for x in parts[2:5]]
            denominator = feasible if np.isfinite(feasible) and feasible > 0 else requested
            assert not np.isfinite(p) or abs(p * denominator - round(p * denominator)) < 1e-5
            adjusted = (round(p * denominator) + 1) / (denominator + 1) if np.isfinite(p) and denominator > 0 else None
            result.setdefault(parts[1], {}).update(p_wcb_raw=p if np.isfinite(p) else None,
                p_wcb=adjusted, bootstrap_draws=requested,
                feasible_draws=feasible if np.isfinite(feasible) else None,
                adjustment_denominator=denominator)
        elif parts and parts[0] == 'PKGERROR':
            result.setdefault(parts[1], {}).update(bootstrap_error=parts[2])
    for name in ws:
        assert name in result and 'estimate_stata' in result[name], (work, name, log[-1200:])
        assert abs(result[name]['estimate_stata'] - contrasts[name]['estimate']) < 1e-8
        if result[name]['se_stata'] is None:
            assert contrasts[name]['se_cr1'] <= 1e-10
        else:
            assert abs(result[name]['se_stata'] - contrasts[name]['se_cr1']) < 1e-7
        if contrasts[name]['se_cr1'] <= 1e-10:
            result[name].update(p_wcb=None, inference_note='Zero estimated cluster variance; bootstrap inference unavailable')
        elif 'p_wcb' not in result[name]:
            result[name]['p_wcb'] = None
            result[name]['inference_note'] = 'Bootstrap unavailable; see saved log'
    return result


def markdown_table(frame):
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |', '|' + '|'.join('---' for _ in frame.columns) + '|']
    lines += ['| ' + ' | '.join(map(str, r)) + ' |' for r in frame.itertuples(index=False, name=None)]
    return '\n'.join(lines)


def descriptions(data, out):
    for name, keys in [('model_agency_prompt', ['model', 'agency', 'prompt']),
                       ('model_language_agency_prompt', ['model', 'platform', 'agency', 'prompt']),
                       ('task_model_agency_prompt', ['task', 'model', 'agency', 'prompt']),
                       ('task_language_model_agency_prompt', ['task', 'platform', 'model', 'agency', 'prompt'])]:
        summary = data.groupby(keys).agg(runs=('primary', 'size'), primary=('primary', 'mean'),
            estimate_only=('estimate_only', 'mean'), estimate_se=('estimate_se', 'mean'),
            mean_cost_usd=('sonnet_cost_usd', 'mean'), median_cost_usd=('sonnet_cost_usd', 'median'),
            mean_provider_seconds=('provider_seconds', 'mean'), median_provider_seconds=('provider_seconds', 'median'),
            mean_input_tokens=('main_input_tokens', 'mean'), mean_output_tokens=('main_output_tokens', 'mean')).reset_index()
        for score in ['primary', 'estimate_only', 'estimate_se']:
            summary[score] *= 100
        summary.to_csv(out / (name + '.csv'), index=False)
    failures = data.groupby(['model', 'platform', 'agency', 'prompt', 'failure_category']).size().rename('runs').reset_index()
    failures.to_csv(out / 'failure_types.csv', index=False)
    cells = data.groupby(['task', 'model', 'platform', 'agency', 'prompt']).agg(successes=('primary', 'sum'),
                       estimate_successes=('estimate_only', 'sum'), runs=('primary', 'size')).reset_index()
    cells.to_csv(out / 'cell_variation.csv', index=False)
    task_split = data.pivot_table(index=['task', 'model', 'rep', 'platform', 'prompt'], columns='agency', values='primary').reset_index()
    task_split['agent_loses'] = (~task_split.agent.astype(bool)) & task_split.self_repair.astype(bool)
    task_split['agent_wins'] = task_split.agent.astype(bool) & (~task_split.self_repair.astype(bool))
    task_split.to_csv(out / 'agency_matched_pairs.csv', index=False)
    data.groupby(['model', 'agency', 'fresh']).agg(runs=('primary', 'size'), primary=('primary', 'mean'),
              estimate_only=('estimate_only', 'mean')).reset_index().to_csv(out / 'collection_context.csv', index=False)


def main():
    ws = weights()
    for subset in ['all_18', 'worst_10']:
        folder = REPORT / subset
        data = pd.read_csv(folder / 'data/runs.csv')
        out = folder / 'tables'
        out.mkdir(exist_ok=True)
        descriptions(data, out)
        rows = []
        for score in ['primary', 'estimate_only']:
            for model in ['sonnet', 'luna', 'pooled']:
                selected = data if model == 'pooled' else data[data.model == model]
                frame = selected.copy()
                frame['success'] = frame[score].astype(float)
                frame['cell'] = [fa.cell_index(r.platform, r.prompt, {'chatbot': 'no_feedback', 'self_repair': 'self_repair', 'agent': 'tool_loop'}[r.agency]) for r in frame.itertuples()]
                frame['armtask'] = frame.model + '__' + frame.task
                fit = smf.ols('success ~ C(armtask) + C(cell, Treatment(1))' if model == 'pooled' else 'success ~ C(task) + C(cell, Treatment(1))', frame).fit(cov_type='cluster', cov_kwds={'groups': frame.task})
                contrasts = fa.contrast_table(fit, ws)
                for value in contrasts.values():
                    if abs(value['estimate']) < 1e-12:
                        value['estimate'] = 0.0
                    if value['se_cr1'] < 1e-8:
                        value['se_cr1'] = 0.0
                        value['t_cr1'] = None
                means = frame.groupby('cell').success.mean().reindex(range(1, 19)).to_numpy()
                for name, vector in ws.items():
                    assert abs(vector @ means - contrasts[name]['estimate']) < 1e-9
                work = REPORT / 'work/inference' / subset / score / model
                boot = wild_bootstrap(frame, ws, contrasts, work, model == 'pooled')
                for name, estimate in contrasts.items():
                    se = estimate['se_cr1']
                    ntask = frame.task.nunique()
                    critical = stats.t.ppf(.975, ntask - 1)
                    rows.append(dict(scoring=score, model=model, contrast=name, label=LABELS.get(name, name),
                        estimate_pp=100 * estimate['estimate'], se_cr1_pp=100 * se,
                        ci_low_pp=100 * (estimate['estimate'] - critical * se), ci_high_pp=100 * (estimate['estimate'] + critical * se),
                        p_cluster_t=2 * stats.t.sf(abs(estimate['estimate'] / se), ntask - 1) if se > 1e-10 else None,
                        n_runs=len(frame), tasks=ntask, **boot[name]))
                (work / 'bootstrap_results.json').write_text(json.dumps(boot, indent=2) + '\n')
                print(subset, score, model, 'complete', flush=True)
        results = pd.DataFrame(rows)
        results.to_csv(out / 'regression_contrasts.csv', index=False)
        for score in ['primary', 'estimate_only']:
            selected = results[(results.scoring == score) & results.contrast.isin(list(LABELS)[:8])]
            table = selected.pivot(index='label', columns='model', values='estimate_pp').round(2).reset_index()
            (out / ('main_results_' + score + '.md')).write_text('# Main regression contrasts\n\n' + markdown_table(table) + '\n\nEffects are percentage points. Full CR1 standard errors, task-cluster t intervals, and wild-bootstrap results are in regression_contrasts.csv.\n')
            formatted = selected.copy()
            formatted['entry'] = formatted.apply(lambda r: f"{r.estimate_pp:.2f} ({r.se_cr1_pp:.2f})" + ('' if score == 'estimate_only' else '***' if pd.notna(r.p_wcb) and r.p_wcb < .01 else '**' if pd.notna(r.p_wcb) and r.p_wcb < .05 else '*' if pd.notna(r.p_wcb) and r.p_wcb < .1 else ''), axis=1)
            wide = formatted.pivot(index='label', columns='model', values='entry').reindex(columns=['sonnet', 'luna', 'pooled']).reset_index()
            tex = '\\begin{tabular}{lrrr}\n\\toprule\nContrast & Sonnet & Luna & Pooled \\\\\n\\midrule\n'
            tex += '\n'.join(' & '.join(str(v).replace('_', r'\_').replace('&', r'\&') for v in row) + r' \\' for row in wide.itertuples(index=False, name=None))
            tex += '\n\\bottomrule\n\\end{tabular}\n'
            (out / ('main_results_' + score + '.tex')).write_text(tex + '\n% Entries: effect in percentage points (CR1 SE). Stars use adjusted wild-bootstrap p-values.\n')
        pooled = results[results.model == 'pooled'].set_index(['scoring', 'contrast']).estimate_pp
        average = results[results.model.isin(['luna', 'sonnet'])].groupby(['scoring', 'contrast']).estimate_pp.mean()
        assert np.max(np.abs(pooled.sort_index() - average.sort_index())) < 1e-8
        (out / 'METHODS.md').write_text('# Estimation and descriptive tables\n\nThe sample has two responses per task, model, language, prompt, and agency cell. Main scoring preserves the saved primary outcome. Estimate-only scoring uses the same target list and tolerances without SE, observation-count, method, or integrity checks.\n\nRegressions use task fixed effects and 18 common language-prompt-agency cell effects within each model. The pooled regression uses model-by-task fixed effects and common cell effects. Standard errors use CR1 clustering by task. Task-cluster t intervals use G-1 degrees of freedom. Every contrast matches the balanced raw cell means, and Stata agrees with Python on estimates and CR1 standard errors. Pooled estimates equal the mean of model estimates.\n\nWild bootstrap uses Stata boottest, null imposed, Rademacher weights, requested 9,999 draws, seed 20260701. Actual and feasible draw counts are saved. The finite-sample adjustment uses (round(p_raw * B) + 1)/(B + 1) with the actual feasible B. Raw p-values also remain available. Zero cluster variance leaves inference unavailable.\n\nThe worst-ten selection uses these same outcomes and is exploratory. Its uncertainty calculations condition on the selected tasks and do not account for selection. Two repetitions provide limited evidence about within-cell response variation. Original and new collection contexts remain documented.\n')


if __name__ == '__main__':
    raise SystemExit('This standalone analysis used an archived 18-task sample. '
                     'Use uv run python code/paper/rebuild.py for the current paper.')

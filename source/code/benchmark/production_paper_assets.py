"""Build the figures and tables for a fixed production snapshot.

Read saved scores only. Make no provider calls. Preserve the figure style used
in the existing manuscript. Keep one PDF per figure and one CSV per data table.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

import factorial_analysis as fa
from paper_numbers import fixed
from stata_log_privacy import strip_license_details
from task_labels import HISTORICAL_LABELS, TASK_LABELS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'data/analysis/main'
STATISTICS = ROOT / 'code/benchmark/production_statistics.py'
RUNTIME_PROVENANCE = ROOT / 'code/benchmark/collection_runtime/provenance.json'
MODELS = ['sonnet', 'luna']
AGENCIES = ['chatbot', 'self_repair', 'agent']
PROMPTS = ['zero_shot', 'few_shot']
PLATFORMS = ['python', 'r', 'stata']
CELL = ['task', 'model', 'platform', 'prompt', 'agency']
TARGET_REPETITIONS = 10
EXPECTED_CELLS = set(itertools.product(MODELS, PLATFORMS, PROMPTS, AGENCIES))
FIGURE_NAMES = {
    'benchmark_design': 'benchmark_design',
    'prompt_agency': 'prompt_agency',
    'task_heterogeneity': 'task_heterogeneity',
    'cost_of_correctness': 'cost_of_correctness',
    'appendix_silent_vs_loud': 'appendix_silent_vs_loud',
    'appendix_pass_at_k': 'appendix_pass_at_k',
    'runtime_by_agency': 'runtime_by_agency',
    'prompt_agency_by_model': 'prompt_agency_by_model',
    'cost_tokens': 'cost_tokens',
    'model_agency_frontier': 'model_agency_frontier',
}
# One label set serves Figure 3, the appendix task tables, and the text (task_labels.py).
SHORT_LABELS = {**HISTORICAL_LABELS, **TASK_LABELS}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap_with_clean_log(helper, frame, ws, contrasts, work, pooled):
    """Run the wild cluster bootstrap, then save its Stata log without license lines.

    ``helper.wild_bootstrap`` keeps the body of the migrated report source, so
    the license lines are removed here after it returns.
    """
    result = helper.wild_bootstrap(frame, ws, contrasts, work, pooled)
    log = Path(work) / 'inference.log'
    if log.is_file():
        log.write_text(strip_license_details(log.read_text()))
    return result


def json_write(path, value):
    def native(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, indent=2, default=native, allow_nan=False) + '\n')


def tex_escape(value):
    return str(value).replace('&', r'\&').replace('_', r'\_').replace('%', r'\%')


def tex_table_text(headings, rows, align=None):
    """Return a booktabs tabular with one heading row."""
    align = align or ('l' + 'r' * (len(headings) - 1))
    lines = [r'\begin{tabular}{' + align + '}', r'\toprule',
             ' & '.join(headings) + r' \\', r'\midrule']
    lines.extend(' & '.join(map(str, row)) + r' \\' for row in rows)
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    return '\n'.join(lines) + '\n'


def tex_table(path, headings, rows, align=None):
    path.write_text(tex_table_text(headings, rows, align))


# Row labels of the appendix tables (Tables B1 and B2). The model name carries no
# harness suffix, as in the manuscript text (Raul Sosa, manuscript commit 3b3b7bd).
TABLE_MODEL_LABEL = {'sonnet': 'Claude Sonnet 4.6', 'luna': 'GPT-5.6 Luna'}
TABLE_AGENCY_LABEL = {'chatbot': 'Chatbot', 'self_repair': 'Self-repair', 'agent': 'Constrained agent'}
TABLE_PLATFORM_LABEL = {'python': 'Python', 'r': 'R', 'stata': 'Stata'}


def group_heading(text, span):
    """An italic group row at the left margin, across the whole table."""
    return r'\multicolumn{' + str(span) + r'}{@{}l}{\textit{' + text + r'}} \\'


def grouped_tabular(colspec, head, groups, span):
    """A booktabs tabular whose rows sit under italic group headings (2026-09-30 table-design audit).

    `groups` is a list of (heading, rows). Each row is a list of cells whose first cell is
    the row label, indented one quad under its heading.
    """
    lines = [r'\begin{tabular}{' + colspec + '}', r'\toprule', *head, r'\midrule']
    for index, (heading, rows) in enumerate(groups):
        if index:
            lines.append(r'\addlinespace[4pt]')
        lines.append(group_heading(heading, span))
        lines.extend(r'\quad ' + ' & '.join(map(str, row)) + r' \\' for row in rows)
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    return '\n'.join(lines) + '\n'


def failure_table(failures):
    """Table B1 tabular: loud and silent failures per 100 runs and the silent share, by model and agency.

    `failures` is the output of production_paper_figures.failure_rates. The model is a group
    heading, and "Failures per 100 runs" spans the loud and silent columns.
    """
    groups = {}
    for r in failures.itertuples():
        share = '--' if pd.isna(r.silent_share_of_failures_pct) else fixed(r.silent_share_of_failures_pct)
        groups.setdefault(TABLE_MODEL_LABEL[r.model], []).append(
            [TABLE_AGENCY_LABEL[r.agency], fixed(r.loud_per100), fixed(r.silent_per100), share])
    head = [r' & \multicolumn{2}{c}{Failures per 100 runs} & Silent share of \\',
            r'\cmidrule(lr){2-3}',
            r'Agency condition & Loud & Silent & failures (percent) \\']
    return grouped_tabular(r'@{}l rr r@{}', head, list(groups.items()), 4)


def software_table(data):
    """Appendix B table tabular and its data: success, loud, and silent rates by model, language, and agency.

    The language is a group heading, agency conditions are rows, and the two models are column groups.
    The last group, "All languages", pools the three programming languages; it replaces the former
    Table B1 (2026-10-01 appendix reductions, proposal B5). Each language has the same number of
    runs, so these rows are the means of the three language rows.
    """
    groups, values = [], []
    for platform in PLATFORMS + ['all']:
        rows = []
        for agency in AGENCIES:
            row = [TABLE_AGENCY_LABEL[agency]]
            for model in MODELS:
                g = data[data.model.eq(model) & data.agency.eq(agency)
                         & (data.platform.eq(platform) if platform != 'all' else True)]
                failed = ~g.primary
                silent = failed & g.execution_ok & g.output_available
                loud = failed & ~silent
                values.append(dict(model=model, platform=platform, agency=agency, n=len(g),
                    success_pct=100 * g.primary.mean(), loud_pct=100 * loud.mean(), silent_pct=100 * silent.mean()))
                row += [fixed(100 * g.primary.mean()), fixed(100 * loud.mean()), fixed(100 * silent.mean())]
            rows.append(row)
        groups.append((TABLE_PLATFORM_LABEL[platform] if platform != 'all' else 'All languages', rows))
    models = ' & '.join(r'\multicolumn{3}{c}{' + TABLE_MODEL_LABEL[m] + '}' for m in MODELS)
    head = [' & ' + models + r' \\', r'\cmidrule(lr){2-4} \cmidrule(l){5-7}',
            r'Language and agency & ' + ' & '.join(['Success & Loud & Silent'] * len(MODELS)) + r' \\']
    text = grouped_tabular(r'@{}l rrr rrr@{}', head, groups, 7)
    values = pd.DataFrame(values).sort_values(['model', 'platform', 'agency'], key=lambda c: c.map(
        {v: i for i, v in enumerate(MODELS + PLATFORMS + ['all'] + AGENCIES)})).reset_index(drop=True)
    return text, values


def task_eligibility(data):
    """Require a completed response in each design cell, including failed responses."""
    decisions = []
    for task, group in data.groupby('task', sort=True):
        observed = set(group[CELL[1:]].itertuples(index=False, name=None))
        unexpected = observed - EXPECTED_CELLS
        if unexpected:
            raise ValueError(f'Unexpected design cells for {task}: {sorted(unexpected)}')
        missing = EXPECTED_CELLS - observed
        decisions.append(dict(task=task, eligible=not missing, completed_responses=len(group),
            observed_cells=len(observed), required_cells=len(EXPECTED_CELLS),
            missing_cells=[dict(zip(CELL[1:], cell)) for cell in sorted(missing)]))
    return decisions


def filter_csv_tasks(path, eligible_tasks, task_column='task'):
    """Keep numeric and Boolean field text unchanged while selecting complete rows."""
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    selected = [row for row in rows if row[task_column].split('__')[0] in eligible_tasks]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(selected)
    return len(rows), len(selected)


def apply_task_eligibility(out):
    """Select the paper tasks once and preserve the original cutoff and source hashes."""
    source = out / 'data/runs.csv'
    receipt_path = out / 'data/task_eligibility.json'
    exclusions_path = out / 'data/paper_exclusions.json'
    exclusions = json.loads(exclusions_path.read_text()) if exclusions_path.is_file() else {'tasks': []}
    excluded_by_request = set(exclusions['tasks'])
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if set(receipt.get('requested_excluded_tasks', [])) != excluded_by_request:
            raise ValueError('Paper exclusions changed; restore the full fixed snapshot before reselection')
        if sha(source) != receipt['selected_runs_sha256']:
            raise ValueError('The selected snapshot differs from its task-eligibility receipt')
        decisions = task_eligibility(pd.read_csv(source))
        assert all(row['eligible'] for row in decisions)
        assert {row['task'] for row in decisions} == set(receipt['eligible_tasks'])
        return receipt

    original_hash = sha(source)
    data = pd.read_csv(source)
    decisions = task_eligibility(data)
    if excluded_by_request - set(data.task):
        raise ValueError('Requested paper exclusion is absent from the source snapshot')
    for row in decisions:
        row['coverage_eligible'] = row['eligible']
        row['excluded_by_request'] = row['task'] in excluded_by_request
        row['eligible'] = row['eligible'] and not row['excluded_by_request']
    eligible = {row['task'] for row in decisions if row['eligible']}
    if not eligible:
        raise ValueError('No task has a completed response in every design cell')
    excluded = data[~data.task.isin(eligible)]
    snapshot_path = out / 'data/snapshot.json'
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.is_file() else {}
    if snapshot.get('runs_sha256') and snapshot['runs_sha256'] != original_hash:
        raise ValueError('The original snapshot hash differs from its recorded source')
    receipt = dict(
        rule='Include a task when every model-language-prompt-agency cell has at least one completed response. Include failures. Retain every available repetition from zero through nine.',
        cutoff=snapshot.get('inventory_cutoff'), source_runs_sha256=original_hash,
        source_responses=len(data), required_cells=len(EXPECTED_CELLS),
        target_repetitions=TARGET_REPETITIONS, decisions=decisions,
        requested_excluded_tasks=sorted(excluded_by_request),
        exclusion_request=exclusions,
        eligible_tasks=sorted(eligible),
        excluded_tasks=sorted(set(data.task) - eligible),
        excluded_responses=len(excluded),
        excluded_records=json.loads(excluded[CELL + ['rep', 'record_path', 'record_sha256', 'output_sha256']].to_json(orient='records')),
        source_snapshot=snapshot.copy(), supporting_files={})
    total, retained = filter_csv_tasks(source, eligible)
    assert total == len(data) and retained == len(data) - len(excluded)
    receipt.update(selected_responses=retained, selected_runs_sha256=sha(source))

    terms = out / 'data/term_checks.csv'
    if terms.is_file():
        term_hash = sha(terms)
        term_total, term_kept = filter_csv_tasks(terms, eligible, 'position_id')
        receipt['supporting_files']['term_checks.csv'] = dict(source_sha256=term_hash,
            selected_sha256=sha(terms), source_rows=term_total, selected_rows=term_kept)
    catalog_path = out / 'data/catalog.json'
    if catalog_path.is_file():
        original_catalog_hash = sha(catalog_path)
        catalog = json.loads(catalog_path.read_text())
        json_write(catalog_path, {task: catalog[task] for task in sorted(eligible)})
        receipt['supporting_files']['catalog.json'] = dict(source_sha256=original_catalog_hash,
            selected_sha256=sha(catalog_path), source_tasks=len(catalog), selected_tasks=len(eligible))
        snapshot['source_catalog_sha256'] = original_catalog_hash
        snapshot['catalog_sha256'] = sha(catalog_path)
    snapshot.update(production_planned=snapshot.get('planned'), planned=len(eligible) * len(EXPECTED_CELLS) * TARGET_REPETITIONS,
        source_analyzed=len(data), analyzed=retained, source_runs_sha256=original_hash,
        runs_sha256=sha(source), task_count=len(eligible), task_eligibility_receipt='task_eligibility.json',
        selection='Tasks with at least one completed response in every design cell, excluding any explicitly requested paper-only tasks; all available repetitions retained',
        requested_excluded_tasks=sorted(excluded_by_request))
    json_write(snapshot_path, snapshot)
    inventory = out / 'data/inventory.json'
    if inventory.is_file():
        contents = json.loads(inventory.read_text())
        contents['scope'] = 'Full production collection at the saved cutoff, before paper task eligibility'
        contents['paper_selection_receipt'] = 'task_eligibility.json'
        json_write(inventory, contents)
    json_write(receipt_path, receipt)
    return receipt


def load_data(source):
    data = pd.read_csv(source)
    assert not data.duplicated(CELL + ['rep']).any()
    assert data.rep.between(0, TARGET_REPETITIONS - 1).all()
    for name in ['primary', 'execution_ok', 'output_available', 'cost_usage_complete']:
        assert data[name].isin([True, False]).all(), name
        data[name] = data[name].astype(bool)
    if 'y' in data:
        assert data.primary.astype(int).equals(data.y.astype(int))
    # runs.csv carries no cost column; price_snapshot adds the paper cost from the saved records.
    assert 'cost_usd' not in data and 'sonnet_cost_usd' not in data
    assert data.groupby(CELL).size().le(TARGET_REPETITIONS).all()
    assert all(row['eligible'] for row in task_eligibility(data))
    data['task_label'] = data.task.map(SHORT_LABELS).fillna(data.task_label)
    return data


def descriptions(data, out):
    table = out / 'tables'
    summaries = {}
    groups = {
        'model_agency_prompt': ['model', 'agency', 'prompt'],
        'model_language_agency_prompt': ['model', 'platform', 'agency', 'prompt'],
        'task_model_agency_prompt': ['task', 'model', 'agency', 'prompt'],
        'model_language_agency': ['model', 'platform', 'agency'],
        'model_agency': ['model', 'agency'],
    }
    for name, keys in groups.items():
        result = data.groupby(keys).agg(
            runs=('primary', 'size'), success_pct=('primary', lambda s: 100 * s.mean()),
            cost_observed=('cost_usd', 'count'),
            mean_cost_cents=('cost_usd', lambda s: 100 * s.mean()),
            median_cost_cents=('cost_usd', lambda s: 100 * s.median()),
            mean_input_tokens=('main_input_tokens', 'mean'),
            mean_output_tokens=('main_output_tokens', 'mean'),
            token_observed=('total_main_tokens', 'count'),
            mean_provider_seconds=('provider_seconds', 'mean'),
            median_provider_seconds=('provider_seconds', 'median')).reset_index()
        result.to_csv(table / f'{name}.csv', index=False)
        summaries[name] = json.loads(result.to_json(orient='records'))

    tasks = sorted(data.task.unique())
    grid = pd.DataFrame(itertools.product(tasks, MODELS, PLATFORMS, PROMPTS, AGENCIES), columns=CELL)
    counts = data.groupby(CELL).agg(runs=('primary', 'size'), successes=('primary', 'sum')).reset_index()
    coverage = grid.merge(counts, how='left', on=CELL)
    coverage[['runs', 'successes']] = coverage[['runs', 'successes']].fillna(0).astype(int)
    coverage['remaining'] = TARGET_REPETITIONS - coverage.runs
    coverage['success_pct'] = np.where(coverage.runs.gt(0), 100 * coverage.successes / coverage.runs, np.nan)
    coverage.to_csv(table / 'coverage_by_cell.csv', index=False)
    by_task = coverage.groupby(['task', 'model']).agg(
        runs=('runs', 'sum'), observed_cells=('runs', lambda s: int(s.gt(0).sum())),
        completed_cells=('runs', lambda s: int(s.eq(TARGET_REPETITIONS).sum())),
        remaining=('remaining', 'sum')).reset_index()
    by_task.to_csv(table / 'coverage_by_task.csv', index=False)
    coverage_rows = []
    for task in tasks:
        d = by_task[by_task.task.eq(task)].set_index('model')
        coverage_rows.append([tex_escape(SHORT_LABELS[task]), *[int(d.loc[m, 'runs']) for m in MODELS],
                              *[f"{int(d.loc[m, 'observed_cells'])}/18" for m in MODELS]])
    tex_table(table / 'task_coverage.tex', ['Task', 'Sonnet runs', 'Luna runs', 'Sonnet cells', 'Luna cells'], coverage_rows)
    json_write(out / 'data/descriptive_summary.json', summaries)
    return coverage, summaries


def paper_estimate(value, pvalue):
    """Use the original table's one-decimal, fixed-width numeric layout."""
    number = fixed(value)  # half-up rounding shared with the manuscript macros
    negative = number.startswith('-')
    body = number[1:] if negative else number
    padding = '' if negative else r'\phantom{-}'
    padding += r'\phantom{0}' * max(0, 2 - len(body.split('.')[0]))
    sign = '{-}' if negative else ''
    stars = '***' if pd.notna(pvalue) and pvalue < .01 else '**' if pd.notna(pvalue) and pvalue < .05 else '*' if pd.notna(pvalue) and pvalue < .1 else ''
    superscript = '^{' + stars + '}' if stars else ''
    star_padding = r'\phantom{^{' + '*' * (3 - len(stars)) + '}}' if len(stars) < 3 else ''
    return '$' + padding + sign + body + superscript + star_padding + '$'


def paper_se(value):
    number = fixed(value)
    padding = r'\phantom{0}' * max(0, 2 - len(number.split('.')[0]))
    return padding + '(' + number + r')\phantom{$^{***}$}'


def main_table_source():
    """Use the preserved Overleaf layout when building the current paper."""
    baseline = ROOT / 'data/production/main/manuscript_baseline/tables/main_results.tex'
    return baseline if baseline.is_file() else ROOT / 'paper/manuscript/tables/main_results.tex'


def write_main_results(result, out, bootstrap):
    """Retain the manuscript's complete table environment and approved layout."""
    template = main_table_source().read_text()
    template = template[template.index(r'\begin{table}'):]
    template = template.replace('GPT-5.4 (Codex)', 'GPT-5.6 Luna (Codex)')
    template = template.replace('Software (zero-shot chatbot, percentage points)', 'Software (percentage points)')
    names = {
        'R versus Python': 'H1_r',
        'Stata versus Python': 'H1_stata',
        'Stata versus Python, design average': 'GAP_stata_avg',
        'Stata versus Python, constrained agent': 'GAP_stata_agent_zs',
        'Few-shot versus zero-shot': 'H2_fewshot',
        'Self-repair versus chatbot': 'H3_self_repair',
        'Constrained agent versus chatbot': 'H3_tool_loop',
        r'Few-shot $\times$ constrained agent': 'H4_tool_loop',
    }
    lines = template.splitlines()
    for i, line in enumerate(lines):
        label = line.split(' & ')[0]
        if label not in names:
            continue
        selected = result[result.contrast.eq(names[label])].set_index('model')
        models = MODELS + ['pooled']
        lines[i] = label + ' & ' + ' & '.join(paper_estimate(selected.loc[m, 'estimate_pp'],
            selected.loc[m, 'p_wcb'] if bootstrap else None) for m in models) + r' \\'
        assert lines[i + 1].startswith(' & ')
        lines[i + 1] = ' & ' + ' & '.join(paper_se(selected.loc[m, 'se_cr1_pp']) for m in models) + r' \\'
    text = '\n'.join(lines) + '\n'
    observations = [int(result[result.model.eq(m)].n_runs.iloc[0]) for m in MODELS + ['pooled']]
    text = re.sub(r'^Observations & .*$', lambda _: 'Observations & ' + ' & '.join(f'{n:,}' for n in observations) + r' \\', text, flags=re.MULTILINE)
    clusters = int(result.effective_tasks.max())
    text = re.sub(r'CR1\) standard errors on the \d+ task clusters', f'CR1) standard errors on the {clusters} task clusters', text)
    clarification = (
        'Every included task has a completed response in all 36 design cells, and all available repetitions enter.\n'
        'Success requires every requested estimate to match its reference within tolerance.\n'
        'The first two software rows refer to zero-shot chatbots, and the constrained-agent software row refers to zero-shot prompts.\n')
    text = text.replace(r'\end{flushleft}', clarification + r'\end{flushleft}')
    if not bootstrap:
        text = re.sub(r'\$\^\{\*\}\$.*?The smallest reportable \$p\$ is 0\.0001\.\n',
                      'Wild-cluster-bootstrap inference is pending.\n', text, flags=re.DOTALL)
    (out / 'tables/main_results.tex').write_text('% Generated from the fixed production snapshot. Original manuscript layout retained.\n' + text)


def regression_tables(data, out, bootstrap=False):
    helper = load_module('snapshot_statistics_helpers', STATISTICS)
    ws = helper.weights()
    rows = []
    omitted = []
    inference_receipts = []
    for model in MODELS + ['pooled']:
        frame = (data if model == 'pooled' else data[data.model.eq(model)]).copy()
        frame['success'] = frame.primary.astype(float)
        frame['cell'] = [fa.cell_index(r.platform, r.prompt,
            {'chatbot': 'no_feedback', 'self_repair': 'self_repair', 'agent': 'tool_loop'}[r.agency])
            for r in frame.itertuples()]
        frame['armtask'] = frame.model + '__' + frame.task
        fe = 'armtask' if model == 'pooled' else 'task'
        informative = frame.groupby(fe).cell.transform('nunique').gt(1)
        omitted.extend(dict(model=model, task=t, reason='Only one design cell observed')
                       for t in sorted(frame.loc[~informative, 'task'].unique()))
        frame = frame[informative].copy()
        assert set(frame.cell) == set(range(1, 19))
        fit = smf.ols(f'success ~ C({fe}) + C(cell, Treatment(1))', frame).fit(
            cov_type='cluster', cov_kwds={'groups': frame.task})
        contrasts = fa.contrast_table(fit, ws)
        for estimate in contrasts.values():
            if abs(estimate['estimate']) < 1e-12:
                estimate['estimate'] = 0.0
        work = out / 'data/inference' / model
        work.mkdir(parents=True, exist_ok=True)
        boot = {}
        exact_input_hash = hashlib.sha256(frame[['success', 'task', 'cell', 'armtask']].to_csv(index=False).encode()).hexdigest()
        cache_reused = False
        if bootstrap:
            cache = work / 'bootstrap_results.json'
            previous = json.loads(cache.read_text()) if cache.is_file() else {}
            same_input = (work / 'input.csv').is_file() and (work / 'input.csv').read_text() == frame[['success', 'task', 'cell', 'armtask']].to_csv(index=False)
            matching = same_input and all(name in previous and abs(previous[name]['estimate_stata'] - contrasts[name]['estimate']) < 1e-8 for name in ws)
            if matching:
                boot = previous
                cache_reused = True
            else:
                print(f'Waiting for the shared Stata lock: {model}', flush=True)
                with open('/private/tmp/agents_metrics_native_stata.lock', 'a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    boot = bootstrap_with_clean_log(helper, frame, ws, contrasts, work, model == 'pooled')
            json_write(work / 'bootstrap_results.json', boot)
        inference_receipts.append(dict(model=model, responses=len(frame), tasks=frame.task.nunique(),
            serialized_regression_input_sha256=exact_input_hash, bootstrap_cache_reused=cache_reused,
            exact_score_and_design_identity=cache_reused if bootstrap else None))
        for name, estimate in contrasts.items():
            value, se = estimate['estimate'], estimate['se_cr1']
            clusters = frame.task.nunique()
            critical = stats.t.ppf(.975, clusters - 1)
            rows.append(dict(model=model, contrast=name, label=helper.LABELS.get(name, name),
                estimate_pp=100 * value, se_cr1_pp=100 * se,
                ci_low_pp=100 * (value - critical * se), ci_high_pp=100 * (value + critical * se),
                p_cluster_t=2 * stats.t.sf(abs(value / se), clusters - 1) if se > 1e-10 else None,
                n_runs=len(frame), effective_tasks=clusters, p_wcb=boot.get(name, {}).get('p_wcb'),
                bootstrap_draws=boot.get(name, {}).get('bootstrap_draws')))
        print(f'Estimated {model}: {len(frame)} responses, {frame.task.nunique()} task clusters', flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'tables/regression_contrasts.csv', index=False)
    write_main_results(result, out, bootstrap)
    supplements = result[result.contrast.str.startswith('AUX_')]
    supplement_rows = []
    for name in supplements.contrast.unique():
        sub = supplements[supplements.contrast.eq(name)].set_index('model')
        supplement_rows.append([tex_escape(sub.label.iloc[0])] + [f'{sub.loc[m, "estimate_pp"]:.2f} ({sub.loc[m, "se_cr1_pp"]:.2f})' for m in MODELS + ['pooled']])
    tex_table(out / 'tables/prompt_and_agency_contrasts.tex', ['Contrast', 'Sonnet', 'Luna', 'Pooled'], supplement_rows)
    json_write(out / 'data/regression_summary.json', dict(
        estimates=json.loads(result.to_json(orient='records')), omitted=omitted,
        input_verification=inference_receipts,
        bootstrap=bootstrap, weighting='Individual responses enter OLS with equal weight. Contrasts average the 18 fitted design-cell coefficients with equal weights.',
        specification='Within model: task fixed effects and 18 design cells. Pooled: model-by-task fixed effects and common design cells.',
        note=('Every task has ten responses in all 36 design cells. The balanced design gives tasks and cells equal weight.'
              if data.groupby(CELL).size().eq(10).all() else
              'Every task has responses in all 36 design cells. Cell counts may differ, so adjusted contrasts can differ from raw rate differences.')))
    return result


def figures(data, out, source):
    from matplotlib import pyplot as plt
    from matplotlib.text import Text
    from types import SimpleNamespace
    from production_paper_figures import render_paper_figures
    from production_paper_costs import render_cost_figures

    captions, inventory = {}, []
    plot = SimpleNamespace(MODEL_LABEL=TABLE_MODEL_LABEL, AGENCY_LABEL=TABLE_AGENCY_LABEL)

    def save(fig, destination, stem, plotted, caption, source_path):
        name = FIGURE_NAMES[stem]
        target = out / 'figures' / f'{name}.pdf'
        axes_count = 0
        if isinstance(fig, Path):
            shutil.copyfile(fig, target)
        else:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            clipped = []
            for item in fig.findobj(match=Text):
                if item.get_visible() and item.get_text().strip():
                    extent = item.get_window_extent(renderer)
                    if extent.width and extent.height and (extent.x0 < -1 or extent.y0 < -1
                            or extent.x1 > fig.bbox.x1 + 1 or extent.y1 > fig.bbox.y1 + 1):
                        clipped.append(item.get_text())
            assert not clipped, (name, clipped)
            axes_count = len(fig.axes)
            fig.savefig(target, metadata={'Title': name, 'Creator': 'Production paper snapshot'})
            plt.close(fig)
        plotted.to_csv(out / 'data' / f'{name}.csv', index=False)
        captions[name] = caption
        inventory.append(dict(figure=name, source_sha256=sha(source_path),
            figure_sha256=sha(target), points=len(plotted), panels=axes_count,
            text_bounds_pass=True, layout_source='Figure 2 design, code/benchmark/paper_figure_style.py'))

    failures, passes, pass_task_count = render_paper_figures(data, out, source, save, SHORT_LABELS)
    render_cost_figures(data, out, source, save)
    for old in [out / 'figures/cost_tokens_luna.pdf', out / 'data/cost_tokens_luna.csv']:
        if old.exists():
            old.unlink()

    # Table B1 (failure types by model and agency) is merged into the language table as its
    # "All languages" rows (2026-10-01 appendix reductions, proposal B5).
    (out / 'tables/appendix_failure_gates.tex').unlink(missing_ok=True)
    software_text, software_data = software_table(data)
    software_data.to_csv(out / 'tables/failure_types_by_software.csv', index=False)
    (out / 'tables/appendix_gate_by_software.tex').write_text(software_text)
    pass_rows = [[plot.MODEL_LABEL[r.model], plot.AGENCY_LABEL[r.agency], r.k, fixed(r.pass_at_k_pct)] for r in passes.itertuples()]
    tex_table(out / 'tables/pass_at_k.tex', ['Model', 'Agency', '$k$', r'Success (\%)'], pass_rows, 'llrr')
    json_write(out / 'data/figure_captions.json', captions)
    json_write(out / 'data/figure_inventory.json', inventory)
    assert len(inventory) == 10
    return failures, pass_task_count


def macros(data, regressions, failures, out, pass_tasks):
    planned = data.task.nunique() * len(EXPECTED_CELLS) * TARGET_REPETITIONS
    cell_counts = data.groupby(CELL).size()
    values = {'snapRunsTotal': len(data), 'snapPlannedRuns': planned, 'snapTasks': data.task.nunique(),
              'snapCompletion': round(100 * len(data) / planned, 1), 'snapPassTasks': pass_tasks,
              'snapPassMinReps': int(cell_counts.min()), 'snapPassMaxReps': int(cell_counts.max()),
              'snapRegressionTasks': int(regressions.effective_tasks.max())}
    pretty = {'sonnet': 'Sonnet', 'luna': 'Luna', 'chatbot': 'Chatbot', 'self_repair': 'Repair',
              'agent': 'Agent', 'zero_shot': 'ZS', 'few_shot': 'FS', 'pooled': 'Pooled'}
    for model in MODELS:
        selected = data[data.model.eq(model)]
        values[f'snapRuns{pretty[model]}'] = len(selected)
        values[f'snap{pretty[model]}Success'] = round(100 * selected.primary.mean(), 1)
        for agency in AGENCIES:
            group = selected[selected.agency.eq(agency)]
            prefix = f'snap{pretty[model]}{pretty[agency]}'
            values[prefix] = round(100 * group.primary.mean(), 1)
            values[prefix + 'Tokens'] = round(group.total_main_tokens.mean())
            values[f'snap{pretty[model]}Runtime{pretty[agency]}'] = round(group.provider_seconds.median(), 1)
            if model == 'sonnet':
                # Paper pricing (run_pricing.csv): list prices by token type.
                priced = group[group.cost_pricing_complete.astype(bool)]
                values[prefix + 'Cents'] = round(100 * priced.cost_usd.mean(), 1)
                values[f'snapSonnetCost{pretty[agency]}'] = round(priced.cost_usd.mean(), 3)
                values[f'snapSonnetCps{pretty[agency]}'] = round(priced.cost_usd.mean() / priced.primary.mean(), 3) if priced.primary.any() else '--'
            for prompt in PROMPTS:
                values[prefix + pretty[prompt]] = round(100 * group[group.prompt.eq(prompt)].primary.mean(), 1)
            prompt_rates = group.groupby('prompt').primary.mean()
            values[f'snap{pretty[model]}Prompt{pretty[agency]}'] = round(100 * (prompt_rates.few_shot - prompt_rates.zero_shot), 1)
            fail = failures[failures.model.eq(model) & failures.agency.eq(agency)].iloc[0]
            for kind in ['Loud', 'Silent']:
                values[f'snap{pretty[model]}{kind}{pretty[agency]}'] = round(fail[kind.lower() + '_per100'], 1)
            share = fail.silent_share_of_failures_pct
            values[f'snap{pretty[model]}SilentShare{pretty[agency]}'] = round(share, 1) if pd.notna(share) else '--'
    sonnet = data[data.model.eq('sonnet')]
    costs = sonnet[sonnet.cost_pricing_complete.astype(bool)].groupby('agency').cost_usd.mean()
    values['snapSonnetCostDeltaCents'] = round(100 * (costs.agent - costs.chatbot), 2)
    values['snapSonnetCostCoverage'] = round(100 * sonnet.cost_pricing_complete.astype(bool).mean(), 1)
    for agency in AGENCIES:
        values['snapPooled' + pretty[agency]] = round(100 * data[data.agency.eq(agency)].primary.mean(), 1)
    for row in regressions.itertuples():
        suffix = ''.join(piece.title() for piece in row.contrast.split('_'))
        # Macro names contain letters only, including hypothesis numbers.
        suffix = suffix.replace('1', 'One').replace('2', 'Two').replace('3', 'Three').replace('4', 'Four')
        value = round(row.estimate_pp, 2)
        values['snap' + pretty[row.model] + suffix] = 0.0 if value == 0 else value
    lines = ['% Generated from the fixed production snapshot. Do not edit by hand.']
    for name, value in values.items():
        assert name.isalpha(), name
        if isinstance(value, (float, np.floating)) and value == 0:
            value = values[name] = 0.0
        lines.append('\\newcommand{\\' + name + '}{' + str(value) + '}')
    (out / 'numbers.tex').write_text('\n'.join(lines) + '\n')
    json_write(out / 'data/numbers.json', values)
    json_write(out / 'data/summary.json', dict(
        runs=len(data), planned=planned, tasks=data.task.nunique(),
        primary='All requested estimates within tolerance; no ancillary standard-error or observation-count requirement',
        coverage=('Every task has ten responses in all 36 design cells.' if cell_counts.eq(10).all()
                  else 'Every task has responses in all 36 design cells. All available repetitions enter; cell counts may differ.'),
        task_eligibility_receipt='task_eligibility.json', minimum_repetitions=int(cell_counts.min()),
        maximum_repetitions=int(cell_counts.max()),
        pass_at_k_tasks=pass_tasks, effective_regression_tasks=int(regressions.effective_tasks.max()),
        regression_runs=int(regressions[regressions.model.eq('pooled')].n_runs.iloc[0]),
        scores=json.loads(data.groupby(['model', 'agency', 'prompt']).primary.agg(['mean', 'size']).reset_index().to_json(orient='records')),
        failures=json.loads(failures.to_json(orient='records')), input_sha256=sha(out / 'data/runs.csv')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--cost-valuation', choices=['list', 'sonnet_all_input'], default='list',
                        help='Token valuation for costs; sonnet_all_input reproduces the earlier Sonnet rule')
    args = parser.parse_args()
    out = args.out.resolve()
    for subdir in ['data', 'tables', 'figures']:
        (out / subdir).mkdir(parents=True, exist_ok=True)
    source = out / 'data/runs.csv'
    eligibility = apply_task_eligibility(out)
    initial_hash = sha(source)
    data = load_data(source)
    from production_paper_pricing import price_snapshot
    data = price_snapshot(data, out, args.cost_valuation)
    descriptions(data, out)
    regressions = regression_tables(data, out, args.bootstrap)
    failures, pass_tasks = figures(data, out, source)
    macros(data, regressions, failures, out, pass_tasks)
    assert sha(source) == initial_hash, 'The fixed snapshot changed during the build'
    json_write(out / 'data/assets_build.json', dict(
        input_sha256=initial_hash, bootstrap=args.bootstrap, figures=10,
        source_snapshot_sha256=eligibility['source_runs_sha256'],
        task_eligibility_sha256=sha(out / 'data/task_eligibility.json'),
        sources={str(path.relative_to(ROOT)): sha(path) for path in [Path(__file__).resolve(),
            STATISTICS, RUNTIME_PROVENANCE,
            ROOT / 'code/benchmark/factorial_analysis.py', main_table_source(),
            ROOT / 'code/benchmark/production_paper_figures.py',
            ROOT / 'code/benchmark/production_paper_costs.py',
            ROOT / 'code/benchmark/paper_figure_style.py',
            ROOT / 'code/benchmark/production_paper_pricing.py']},
        historical_sources={entry['source']: entry['sha256'] for name, entry in json.loads(RUNTIME_PROVENANCE.read_text()).items()
                            if name in {'production_statistics', 'historical_figures'}},
        visual_check='Inspect rendered PDF pages after each data refresh. This generation receipt does not certify a visual review.'))
    print(json.dumps(dict(runs=len(data), figures=10, bootstrap=args.bootstrap, output=str(out))), flush=True)


if __name__ == '__main__':
    main()

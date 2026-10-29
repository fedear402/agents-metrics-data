"""Build the six-model PNAS analysis from saved responses.

Run ``uv run --no-sync python code/paper/pnas_analysis.py`` from the project root.
The build checks all saved record and output hashes. It rescales no estimates and
makes no provider calls. It never executes a submitted script. Raw scoring and
raw precision tokens are cached against their input and scorer hashes. The older
analysis and manuscript are read-only inputs.

All generic, All, and Pooled number variables describe the six-model sample.
Named original-model variables retain their model scope. Hand-audit and original
repair-transition variables retain their original-pair scope, recorded in the
receipt. MODEL_gap in the robustness table is the original-pair comparison.

After the raw scoring, the build applies the wrong-method alternative check of
code/paper/scoring_alternatives.py (decision record 2026-10-29_wrong-method-alternative-check.md).
A success whose scored value is closer to a listed wrong-method value than to the nearest
accepted value becomes a silent failure. data/alternative_checks.csv lists every changed value.
The comparisons with the two-model analysis use the outcome before this check, because the
two-model analysis does not apply it.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist, f as fdist

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/paper'), str(ROOT / 'code/benchmark')]
import dataset  # noqa: E402
import six_models as six  # noqa: E402
import production_statistics as statistics  # noqa: E402
import paper_robustness as robustness  # noqa: E402
import scoring_alternatives as alternatives  # noqa: E402
from paper_numbers import fixed  # noqa: E402

OUT = ROOT / 'data/analysis/six_models'
OLD = ROOT / 'data/analysis/main'
BASE_NUMBERS = ROOT / 'paper/manuscript/numbers.tex'
TAGS = dict(sonnet='Sonnet', luna='Codex', haiku='Haiku', luna6='LunaSix', opus='Opus', sol='Sol')
AGENCY_TAGS = dict(chatbot='Chatbot', self_repair='OneRepair', agent='Agent')
CELL_KEYS = ['task', 'model', 'platform', 'prompt', 'agency']
GROUPS = {**{m: [m] for m in six.ORDER}, 'pooled': six.ORDER, **six.COST_GROUPS}
CHECKPOINT = ROOT / 'quality_reports/audits/2026-10-21_pnas-six-models/work/analysis/progress.jsonl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=native, allow_nan=False) + '\n')


def checkpoint(unit, **kw):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open('a') as stream:
        stream.write(json.dumps(dict(unit=unit, status='complete', at=datetime.now(timezone.utc).isoformat(), **kw)) + '\n')


def source_paths():
    return [OLD / 'data/runs.csv', OLD / 'data/run_pricing.csv', OLD / 'data/catalog.json',
            *[six.NEW / f'{key}_paper_{kind}.csv' for key in six.NEW_KEYS.values() for kind in ('rows', 'pricing')]]


def validate_design(runs):
    expected = set(itertools.product(sorted(runs.task.unique()), six.ORDER, six.PLATFORMS,
                                     ['zero_shot', 'few_shot'], six.AGENCIES))
    counts = runs.groupby(CELL_KEYS).size()
    if len(runs) != 43200 or runs.task.nunique() != 40 or set(counts.index) != expected or not counts.eq(10).all():
        raise ValueError('The analysis requires all 43,200 runs in 4,320 complete design cells')
    if runs.position_id.duplicated().any() or runs.record_path.duplicated().any():
        raise ValueError('Duplicate selected position or record')
    reps = runs.groupby(CELL_KEYS).rep.apply(lambda s: sorted(s.astype(int)))
    if not reps.apply(lambda x: x == list(range(10))).all():
        raise ValueError('A design cell lacks repetitions zero through nine')


def load_sources():
    runs, order, skipped = six.load()
    if order != six.ORDER:
        raise ValueError(f'All six models are required: {skipped}')
    runs = runs.drop(columns='cost_usd')
    pricing = pd.concat([pd.read_csv(OLD / 'data/run_pricing.csv'),
                         *[pd.read_csv(six.NEW / f'{key}_paper_pricing.csv') for key in six.NEW_KEYS.values()]],
                        ignore_index=True)
    validate_design(runs)
    if pricing.record_path.duplicated().any() or set(pricing.record_path) != set(runs.record_path):
        raise ValueError('The pricing ledger does not match the selected records')
    aligned = runs[['record_path', 'record_sha256']].merge(pricing, on='record_path', validate='one_to_one', suffixes=('', '_pricing'))
    if not aligned.record_sha256.eq(aligned.record_sha256_pricing).all():
        raise ValueError('The pricing ledger identifies different saved records')
    return runs, pricing, json.loads((OLD / 'data/catalog.json').read_text())


def scorer_paths():
    return [Path(dataset.__file__), Path(dataset.analysis.__file__), Path(dataset.saved.__file__),
            Path(dataset.channels.__file__), Path(dataset.se_scoring.__file__), Path(dataset.se_scoring.v3.__file__),
            dataset.se_scoring.SPEC_PATH, ROOT / 'code/benchmark/strict_posthoc.py']


def score_saved(runs, catalog, out, force=False):
    """Check immutable inputs, then compute or reuse the complete raw scoring cache."""
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [*source_paths(), *scorer_paths()]}
    work = out / 'work/pnas_scoring'
    cache = work / 'receipt.json'
    names = ('runs.csv', 'term_checks.csv', 'se_checks.csv', 'rescoring_audit.csv')
    # Recheck content hashes on every invocation, including cached scoring.
    channel_hashes = []
    for row in runs.itertuples(index=False):
        record = dataset.repo_path(row.record_path)
        if not record.is_file() or sha(record) != row.record_sha256:
            raise ValueError(f'Saved record changed: {record}')
        output = dataset.repo_path(row.output_path)
        got = sha(output) if output.is_file() else None
        expected = row.output_sha256 if pd.notna(row.output_sha256) else None
        if got != expected:
            raise ValueError(f'Saved output changed: {output}')
        if row.platform == 'stata':
            rec = json.loads(record.read_text())
            found = dataset.channels.extra_sources(record, rec, output, catalog[row.task]['estimands'])
            for label, _ in found:
                # SE parsing uses complete source text, so retain the complete hash.
                content = (dataset.channels.final_stdout(record.parent, rec).encode() if label == 'stdout'
                           else (record.parent / label).read_bytes())
                channel_hashes.append((row.position_id, label, hashlib.sha256(content).hexdigest()))
    sources['stata_channel_texts'] = hashlib.sha256(json.dumps(channel_hashes).encode()).hexdigest()
    key = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    cached = (not force and cache.is_file() and all((work / n).is_file() for n in names)
              and json.loads(cache.read_text()).get('key') == key)
    if cached:
        receipt = json.loads(cache.read_text())
        if any(sha(work / n) != receipt['outputs'][n] for n in names):
            raise ValueError('The scoring cache changed after its receipt was written')
        return [pd.read_csv(work / n, low_memory=False) for n in names], dict(receipt, reused=True)
    work.mkdir(parents=True, exist_ok=True)
    c = dataset.analysis.runtime.load_runtime(work / 'runtime').c
    rows, terms, ses, audit = [], [], [], []
    for i, row in enumerate(runs.to_dict('records')):
        path = dataset.repo_path(row['record_path'])
        record = json.loads(path.read_text())
        scored = dataset.score_output(dataset.repo_path(row['output_path']), path, record, row['platform'],
                                      catalog[row['task']], c, 'channel_se')
        passed = bool(scored['passed'])
        if passed != bool(row['primary']) or passed != bool(row['y']):
            raise ValueError(f"Saved primary differs from raw rescoring: {row['position_id']}")
        category = ('Success' if passed else 'Silent failure'
                    if row['execution_ok'] and scored['delivered'] else 'Loud failure')
        if category != row['failure_category'] or bool(scored['delivered']) != bool(row['output_available']):
            raise ValueError(f"Failure classification differs: {row['position_id']}")
        row.update(primary=passed, y=int(passed), output_rule='channel_se', failure_category=category)
        rows.append(row)
        terms.extend(dict(position_id=row['position_id'], task=row['task'], term=t, **details)
                     for t, details in scored['terms'].items())
        ses.extend(dict(position_id=row['position_id'], task=row['task'], term=t, **details)
                   for t, details in scored['se_details'].items())
        audit.append(dict(position_id=row['position_id'], previous_primary=passed, primary=passed, changed=False,
                          channel_used='|'.join(scored['channel_used']),
                          channel_conflict_terms='|'.join(scored['channel_conflicts']),
                          se_failed_terms='|'.join(scored['se_failed'])))
        if (i + 1) % 7200 == 0:
            print(f'Raw scoring checked {i + 1:,} of {len(runs):,} runs', flush=True)
    frames = [pd.DataFrame(x) for x in (rows, terms, ses, audit)]
    for name, frame in zip(names, frames):
        frame.to_csv(work / name, index=False)
    receipt = dict(key=key, input_hashes=sources, runs=len(runs), primary_disagreements=0,
                   record_hashes_checked=len(runs), output_hashes_checked=len(runs),
                   outputs={n: sha(work / n) for n in names})
    write_json(cache, receipt)
    checkpoint('raw_scoring', key=key, runs=len(runs), disagreements=0)
    return frames, dict(receipt, reused=False)


def cr1_factor(g, models=1):
    n, k = g * 180 * models, g * models + 17
    return float(np.sqrt(g / (g - 1) * (n - 1) / (n - k)))


def contrast_se(values, models=1):
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    se = cr1_factor(len(values), models) * np.sqrt(((values - mean) ** 2).sum(axis=0)) / len(values)
    return mean, se


def model_cells(runs, outcome='y'):
    tasks = sorted(runs.task.unique())
    index = pd.MultiIndex.from_product([tasks, six.PLATFORMS, ['zero_shot', 'few_shot'], six.AGENCIES])
    return {m: 100 * runs[runs.model.eq(m)].groupby(['task', 'platform', 'prompt', 'agency'])[outcome].mean()
            .reindex(index).to_numpy().reshape(len(tasks), 18) for m in six.ORDER}


def regression_contrasts(runs, legacy_check=True):
    """Table 1 contrasts. legacy_check compares Sonnet and Luna with the two-model analysis, which
    holds only for the outcome before the alternative check."""
    cells = model_cells(runs)
    ws = statistics.weights()
    rows = []
    old = pd.read_csv(OLD / 'tables/regression_contrasts.csv').set_index(['model', 'contrast'])
    for model, members in GROUPS.items():
        pooled = np.mean([cells[m] for m in members], axis=0)
        for contrast, weight in ws.items():
            task_values = pooled @ weight
            estimate, se = contrast_se(task_values, len(members))
            # Tasks whose compared cells differ. Table 1 marks the cells with fewer than three.
            informative = int((np.abs(task_values) > 1e-10).sum())
            estimate = 0.0 if abs(estimate) < 1e-10 else float(estimate)
            se = 0.0 if abs(se) < 1e-10 else float(se)
            p = float(2 * tdist.sf(abs(estimate / se), 39)) if se else None
            q = tdist.ppf(.975, 39)
            wcb = old.loc[(model, contrast), 'p_wcb'] if model in ('sonnet', 'luna') else np.nan
            rows.append(dict(model=model, contrast=contrast, label=statistics.LABELS[contrast], estimate_pp=estimate,
                             se_cr1_pp=se, ci_low_pp=estimate-q*se, ci_high_pp=estimate+q*se, p_cluster_t=p,
                             n_runs=7200*len(members), effective_tasks=40, informative_tasks=informative, p_wcb=wcb,
                             bootstrap_draws=9999 if pd.notna(wcb) else np.nan, member_models=','.join(members)))
    result = pd.DataFrame(rows)
    original = result[result.model.isin(['sonnet', 'luna'])].set_index(['model', 'contrast'])
    diff = max(float((original[c] - old.loc[original.index, c]).abs().max()) for c in ('estimate_pp', 'se_cr1_pp'))
    if legacy_check and diff > 1e-6:
        raise ValueError(f'Original model contrasts differ from the established analysis: {diff}')
    return result, diff


def joint_model_test(task_values):
    """Shared-task Wald F test that all model-specific contrasts are equal.

    The covariance equals the fully interacted model-by-task and cell regression.
    Its CR1 correction counts 18 cell effects and G-1 task effects per model.
    """
    values=np.asarray(task_values,dtype=float)
    g,models=values.shape
    differences=values[:,1:]-values[:,[0]]
    delta=differences.mean(axis=0)
    centered=differences-delta
    n,k=g*180*models,(g+17)*models
    covariance=(g/(g-1)*(n-1)/(n-k))*(centered.T@centered)/(g*g)
    rank=int(np.linalg.matrix_rank(covariance))
    if rank==0:
        return dict(F=None,df_num=0,df_den=g-1,p_cluster_F=None)
    statistic=float(delta@np.linalg.pinv(covariance)@delta/rank)
    return dict(F=statistic,df_num=rank,df_den=g-1,p_cluster_F=float(fdist.sf(statistic,rank,g-1)))


def heterogeneity_tests(runs,out):
    cells=model_cells(runs)
    rows=[]
    for contrast,weight in statistics.weights().items():
        task_values=np.column_stack([cells[m]@weight for m in six.ORDER])
        rows.append(dict(contrast=contrast,models=','.join(six.ORDER),n_runs=len(runs),
                         hypothesis='all six model-specific contrasts are equal',
                         **joint_model_test(task_values)))
    result=pd.DataFrame(rows)
    result.to_csv(out/'data/cross_model.csv',index=False)
    return result


def robust_matrix(cells):
    matrices = {m: np.column_stack([robustness.task_contrasts(cells[m].reshape(-1, 3, 2, 3)).get(
        k, np.full(len(cells[m]), np.nan)) for k in robustness.ROWS]) for m in six.ORDER}
    pooled = np.mean([matrices[m] for m in six.ORDER], axis=0)
    # Keep the original paired comparison as a clearly labelled subset row.
    pooled[:, robustness.ROWS.index('MODEL_gap')] = cells['sonnet'].mean(axis=1) - cells['luna'].mean(axis=1)
    return pooled


def cluster_se(values, groups, models):
    sums = pd.Series(values - values.mean()).groupby(groups).sum().to_numpy()
    g = len(sums)
    n, k = len(values) * 180 * models, len(values) * models + 17
    return float(np.sqrt(g / (g - 1) * (n - 1) / (n - k)) * np.sqrt((sums ** 2).sum()) / len(values)), g


def build_robustness(out, runs, catalog, contrasts, estimate_flagged=frozenset()):
    """estimate_flagged: runs with an estimate closer to a wrong-method alternative. They fail every
    tolerance scheme, as they fail the primary rule."""
    tasks = sorted(runs.task.unique())
    relation = pd.read_csv(robustness.RELATION).set_index('task')
    groups = np.array([robustness.dataset_groups(catalog)[t] for t in tasks])
    X = robust_matrix(model_cells(runs))
    sch = robustness.scheme_outcomes(out, runs, out / 'work')
    # The schemes recompute the estimate tolerance from the errors; the alternative check stays in force.
    hit = sch.position_id.isin(estimate_flagged)
    for name in (*robustness.SCHEMES, *robustness.LOOSE_SCHEMES):
        if name in sch:
            sch.loc[hit, name] = 0
    full = runs.merge(sch, on='position_id', validate='one_to_one')
    schemes = {name: robust_matrix(model_cells(full, name)) for name in robustness.SCHEMES}
    rng = np.random.default_rng(robustness.SEED)
    boot = rng.integers(0, 40, size=(robustness.N_BOOT, 40))
    subsets = np.array([rng.choice(40, robustness.SUBSET, replace=False) for _ in range(robustness.N_RANDOM)])
    rows = []
    for j, contrast in enumerate(robustness.ROWS):
        v = X[:, j]
        members = 2 if contrast == 'MODEL_gap' else 6
        mean, se = contrast_se(v, members)
        sed, g = cluster_se(v, groups, members)
        b, r = v[boot].mean(axis=1), v[subsets].mean(axis=1)
        q, qg = tdist.ppf(.975, 39), tdist.ppf(.975, g-1)
        rec = dict(contrast=contrast, label=robustness.LABELS[contrast], est=mean, se=se,
                   ci_task_lo=mean-q*se, ci_task_hi=mean+q*se, dataset_clusters=g, dataset_se=sed,
                   dataset_lo=mean-qg*sed, dataset_hi=mean+qg*sed, loo_min=((v.sum()-v)/39).min(),
                   loo_max=((v.sum()-v)/39).max(), boot_lo=np.quantile(b,.025), boot_hi=np.quantile(b,.975),
                   r20_p5=np.quantile(r,.05), r20_p95=np.quantile(r,.95),
                   r20_same_sign=float(np.mean(np.sign(r)==np.sign(mean))), models=members,
                   scope='original Sonnet-Luna pair' if members==2 else 'all six models')
        for name in robustness.SCHEMES:
            rec[f'{name}_est'], rec[f'{name}_se'] = contrast_se(schemes[name][:,j], members)
        for half in ('added', 'original'):
            vals = v[relation.reindex(tasks).half.eq(half).to_numpy()]
            rec[f'{half}_est'], rec[f'{half}_se'] = contrast_se(vals, members)
        rows.append(rec)
    table = pd.DataFrame(rows)
    ref = contrasts[contrasts.model.eq('pooled')].set_index('contrast')
    diff = max(abs(float(row.est-ref.loc[row.contrast,'estimate_pp'])) for row in table.itertuples()
               if row.contrast != 'MODEL_gap')
    if diff > 1e-6:
        raise ValueError('The pooled robustness contrasts do not reproduce the main table')
    fail = 100*(1-runs.groupby(['agency','prompt']).y.mean())
    removed = {a: float(100*(1-fail[a,'few_shot']/fail[a,'zero_shot'])) for a in six.AGENCIES}
    rates = {s: {group: float(100*full.loc[full.agency.eq(group) if group!='all' else np.ones(len(full),bool),
                                         'y' if s=='primary' else s].mean())
                 for group in ('all',*six.AGENCIES)} for s in ('primary',*robustness.SCHEMES)}
    se_rows = pd.read_csv(out / 'data/se_checks.csv')
    se_rows = se_rows[~se_rows.se_pass.astype(bool)]
    se_only, run_sets = {}, []
    for variant, column in robustness.SE_ONLY:
        mask = full[column].astype(bool) & ~full.se_ok.astype(bool)
        by_task = full.loc[mask,'task'].value_counts().sort_index()
        positions = set(full.loc[mask,'position_id'])
        run_sets.append(positions)
        failed = se_rows[se_rows.position_id.isin(positions)]
        relative = failed.se_err / failed.se_ref.abs()
        gap = relative.groupby(failed.task).min()
        # The largest SE gap of each run, then the smallest of these over the runs of a task (Appendix D).
        run_gap = relative.groupby(failed.position_id).max().groupby(failed.groupby('position_id').task.first()).min()
        se_only[variant] = dict(runs=int(mask.sum()), by_task={t:int(v) for t,v in by_task.items()},
                               missing_se_runs=int(failed[failed.se_reason.eq('se_missing')].position_id.nunique()),
                               min_relative_se_error={t:float(v) for t,v in gap.dropna().items()},
                               min_run_relative_se_error={t:float(v) for t,v in run_gap.dropna().items()})
    info = dict(table1_max_abs_diff=diff, dataset_clusters=len(set(groups)), seed=robustness.SEED,
                bootstrap_draws=robustness.N_BOOT, random_draws=robustness.N_RANDOM, subset_size=robustness.SUBSET,
                se_only_failures=se_only, success_rates={s:v['all'] for s,v in rates.items()},
                success_rates_by_agency=rates, se_only_same_runs=all(s==run_sets[0] for s in run_sets),
                failure_rates={f'{a}_{p}':float(fail[a,p]) for a in six.AGENCIES for p in robustness.PROMPTS},
                failures_removed=removed, runs=len(runs), model_gap_scope='original Sonnet-Luna pair',
                fe='model-by-task fixed effects, common language-prompt-agency cells')
    sign_columns = ['dataset_lo','dataset_hi','loo_min','loo_max','boot_lo','boot_hi','r20_p5','r20_p95'] + [f'{s}_est' for s in robustness.SCHEMES]
    info['claim_checks'] = dict(sign_changes={r.contrast:[c for c in sign_columns if np.sign(getattr(r,c))!=np.sign(r.est)]
                                              for r in table.itertuples()},
                                intervals_include_zero=[r.contrast for r in table.itertuples()
                                                        if r.dataset_lo*r.dataset_hi<=0 or r.boot_lo*r.boot_hi<=0],
                                failures_removed_range_pp=max(removed.values())-min(removed.values()))
    macros = robustness.macros(table,info)
    robustness.verify_se_claims(info)
    macros['numRobSeOnlyShiftShareMinRunGap'] = robustness.fixed(
        100*info['se_only_failures']['primary']['min_run_relative_se_error']['applied_shift_share'], 1)
    # Diagnostic only. The primary scoring tolerance remains unchanged.
    shift=se_rows[se_rows.task.eq('applied_shift_share')].copy()
    shift['relaxed_pass']=(shift.se_reason.eq('se_outside_tolerance') &
                           shift.se_err.le(.01*shift.se_ref.abs()+shift.se_allowance.fillna(0)))
    relaxed_positions=shift.groupby('position_id').relaxed_pass.all()
    eligible=full.loc[full.primary_est.astype(bool)&~full.y.astype(bool)&full.task.eq('applied_shift_share'),'position_id']
    relaxed_count=int(relaxed_positions.reindex(eligible).fillna(False).sum())
    macros['numRobShiftShareRelaxedChanges']=str(relaxed_count)
    info['shift_share_se_relaxed_1pct_changed_outcomes']=relaxed_count
    # The PNAS table names both models of the model contrast in full. The two-model table keeps its label.
    two_model_label = robustness.PRINTED['MODEL_gap']
    robustness.PRINTED['MODEL_gap'] = ('Sonnet 4.6 minus', 'GPT-5.6 Luna, all runs')
    try:
        body = robustness.render_table(table,info)
    finally:
        robustness.PRINTED['MODEL_gap'] = two_model_label
    body = body.replace('Pooled over the Two', 'Pooled over the Six')
    body = body.replace('column 3 of', 'the pooled column of')
    body = body.replace(r'\textit{Notes:}', r'\textit{Notes:} The last contrast uses the original Sonnet--Luna pair. All other contrasts and success rates pool six models.')
    body = body.replace('Generated by code/benchmark/paper_robustness.py', 'Generated by code/paper/pnas_analysis.py')
    table.to_csv(out / 'tables/robustness.csv',index=False)
    (out / 'tables/appendix_robustness.tex').write_text(body)
    write_json(out / 'data/robustness.json',dict(info,macros=macros))
    return macros, info


def cost_per_success(frame):
    """Total cost divided by successes within the same priced set of runs."""
    priced = frame[frame.cost_usd.notna()]
    return float(priced.cost_usd.sum()/priced.primary.sum())


def repair_transitions(data, out):
    """Per model: percent of self-repair silent failures whose first script had no complete output,
    and percent of incomplete first scripts that end as successes."""
    from paper_appendix_numbers import first_attempt_complete
    catalog = json.loads((OLD / 'data/catalog.json').read_text())
    c = dataset.analysis.runtime.load_runtime(out / 'work/repair_runtime').c
    repair = data[data.agency.eq('self_repair')].copy()
    repair['first_complete'] = [True if r.calls_count == 1 else first_attempt_complete(r, catalog[r.task], c, dataset)
                                for r in repair.itertuples()]
    result = {}
    for model in TAGS:
        q = repair[repair.model.eq(model)]
        silent = q[q.failure_category.eq('Silent failure')]
        incomplete = q[~q.first_complete]
        result[model] = (100 * (~silent.first_complete).mean() if len(silent) else float('nan'),
                         100 * incomplete.primary.mean() if len(incomplete) else float('nan'))
    return result


def build_numbers(runs, pricing, contrasts, out, rob_macros, legacy_check=True, write=True):
    """Number variables. With legacy_check, Sonnet and Luna variables must equal the two-model values.
    With write=False the function only runs its checks."""
    baseline = dict(re.findall(r'\\(?:newcommand|providecommand)\{\\(num[A-Za-z]+)\}\{([^{}]*)\}',BASE_NUMBERS.read_text()))
    n = dict(baseline)
    data = runs.merge(pricing[['record_path','cost_usd','cost_pricing_complete',
                               'priced_cached_input_tokens','priced_cache_write_input_tokens']],on='record_path',
                      validate='one_to_one')
    passes = pd.read_csv(out / 'data/appendix_pass_at_k.csv')
    reg = contrasts.set_index(['model','contrast'])
    overwritten = set()
    def put(name,value,decimals=1,absolute=False):
        name = 'num'+name
        n[name] = fixed(abs(value) if absolute else value,decimals,thousands=True)
        overwritten.add(name)
    def select(**kw):
        q = data
        for key,value in kw.items():
            q = q[q[key].eq(value)]
        return q
    def success(**kw): return select(**kw).primary.mean()
    def cost(stat='mean',**kw): return getattr(select(**kw).cost_usd,stat)()
    def cps(**kw): return cost_per_success(select(**kw))
    def incremental(a,b,**kw):
        # Price comparisons also use success within each priced agency subset.
        left,right = select(agency=a,**kw),select(agency=b,**kw)
        left,right = left[left.cost_usd.notna()],right[right.cost_usd.notna()]
        return (right.cost_usd.mean()-left.cost_usd.mean())/(right.primary.mean()-left.primary.mean())
    for name,value in [('Models',6),('Tasks',40),('Repetitions',10),('RunsTotal',len(data)),('RunsPerArm',7200),
                       ('RunsPerPrompt',3600),('TableTotalColumn',7),('PricedRuns',int(data.cost_usd.notna().sum())),
                       ('UnpricedRuns',int(data.cost_usd.isna().sum()))]:
        put(name,value,0)
    for label,value in [('Prompt',7200),('Language',4800),('Failure',14400),('Task',360)]:
        put('PointRuns'+label,value,0)
    costliest=data.groupby(['model','platform']).cost_usd.mean().unstack().idxmax(axis=1)
    put('StataCostliestModels',int(costliest.eq('stata').sum()),0)
    # Prose spells out counts below ten.
    words=['zero','one','two','three','four','five','six','seven','eight','nine']
    for name,count in [('Models',data.model.nunique()),('StataCostliestModels',int(costliest.eq('stata').sum()))]:
        n['num'+name+'Word']=words[count]
        n['num'+name+'WordCap']=words[count].capitalize()
        overwritten.update({'num'+name+'Word','num'+name+'WordCap'})
    for group,agency,label in [('cheap','agent','CheapAgent'),('expensive','chatbot','ExpensiveChatbot')]:
        q=data[data.model.isin(six.COST_GROUPS[group])&data.agency.eq(agency)]
        put(label+'Success',100*q.primary.mean())
        put(label+'Cost',q.cost_usd.mean(),4)
    put('PooledSuccess',100*success())
    put('ProviderHoursTotal',data.provider_seconds.sum()/3600)
    for model,tag in TAGS.items():
        usd = 3 if model in ('sonnet','opus','sol') else 4
        put(tag+'Overall',100*success(model=model))
        put('ProviderHours'+tag,select(model=model).provider_seconds.sum()/3600)
        for agency,label in AGENCY_TAGS.items():
            q = select(model=model,agency=agency)
            put(tag+label,100*q.primary.mean())
            gain = 100*(success(model=model,agency=agency,prompt='few_shot')-success(model=model,agency=agency,prompt='zero_shot'))
            put('PromptGain'+label+tag,gain,absolute=not(label=='OneRepair' and tag=='Sonnet'))
            put(tag+'MeanCost'+label,q.cost_usd.mean(),4)
            put(tag+'MedianCost'+label,q.cost_usd.median(),usd)
            put(tag+'Cps'+label,cost_per_success(q),4)
            put('Runtime'+label+'Median'+tag,q.provider_seconds.median())
            loud,silent = q.failure_category.eq('Loud failure'),q.failure_category.eq('Silent failure')
            put('Loud'+label+tag,100*loud.mean(),2 if agency=='agent' else 1)
            put('SilentPerHundred'+label+tag,100*silent.mean())
            put('Silent'+label+tag,100*silent.sum()/(q.execution_ok&q.output_available).sum())
            put('SilentShare'+label+tag,100*silent.sum()/(~q.primary).sum() if (~q.primary).any() else 0,0)
            put('PricedRuns'+label+tag,int(q.cost_usd.notna().sum()),0)
            for k,kl in [(1,'One'),(10,'Twelve')]:
                put('Pass'+kl+label+tag,float(passes[(passes.model==model)&(passes.agency==agency)&(passes.k==k)].pass_at_k_pct.iloc[0]))
            for prompt,pl in [('zero_shot','ZS'),('few_shot','FS')]:
                put(tag+'MedianCost'+label+pl,cost(stat='median',model=model,agency=agency,prompt=prompt),usd)
                put(tag+'Cps'+label+pl,cps(model=model,agency=agency,prompt=prompt),usd)
        for platform,pl in [('python','Python'),('r','R'),('stata','Stata')]:
            put(tag+pl,100*success(model=model,platform=platform))
            put(tag+'MedianCost'+pl,cost(stat='median',model=model,platform=platform),usd)
            put(tag+'Cps'+pl,cps(model=model,platform=platform),usd)
            put('LoudShareChatbot'+pl+tag,100*select(model=model,platform=platform,agency='chatbot').failure_category.eq('Loud failure').mean())
        for agency,label in [('chatbot','Chatbot'),('agent','Agent')]:
            put(tag+'Stata'+label+'ZS',100*success(model=model,platform='stata',agency=agency,prompt='zero_shot'))
        for k,kl in [(1,'One'),(10,'Twelve')]:
            put('Pass'+kl+tag,passes[(passes.model==model)&(passes.k==k)].pass_at_k_pct.mean())
        for kind in ('Loud','Silent'):
            base = select(model=model,agency='chatbot').failure_category.eq(kind+' failure').mean()
            agent = select(model=model,agency='agent').failure_category.eq(kind+' failure').mean()
            put(kind+'Reduction'+tag,100*(1-agent/base) if base else 0,0)
        rates = select(model=model).groupby('agency').primary.mean()
        put('RepairGainShare'+tag,100*(rates.self_repair-rates.chatbot)/(rates.agent-rates.chatbot),0)
        put('MeanCostAgencyRatio'+tag,cost(model=model,agency='agent')/cost(model=model,agency='chatbot'))
        put('RuntimeRatio'+tag,select(model=model,agency='agent').provider_seconds.median()/select(model=model,agency='chatbot').provider_seconds.median())
        put('RuntimeRepairPct'+tag,100*(select(model=model,agency='self_repair').provider_seconds.median()/select(model=model,agency='chatbot').provider_seconds.median()-1),0)
        put('RevShareGtOne'+tag,100*select(model=model,agency='self_repair').calls_count.gt(2).mean())
        put('RevMax'+tag,select(model=model,agency='self_repair').calls_count.max()-1,0)
        put(tag+'FewShotPct',100*abs(cost(model=model,prompt='few_shot')/cost(model=model,prompt='zero_shot')-1),0)
        for prompt,pl in [('zero_shot','ZS'),('few_shot','FS')]:
            q=select(model=model,agency='chatbot',prompt=prompt)
            for kind in ('Loud','Silent'):
                put(('LoudChatbot' if kind=='Loud' else 'SilentPerHundredChatbot')+pl+tag,100*q.failure_category.eq(kind+' failure').mean())
        p = select(model=model).groupby(['task','platform','prompt','agency']).primary.mean()
        within=(p*(1-p)).mean()*10/9
        between=p.var()-within/10
        put('Icc'+tag,between/(between+within),2)
    mappings=[('HOneR','H1_r'),('HOneStata','H1_stata'),('HTwo','H2_fewshot'),('HThreeARepair','H3_self_repair'),
              ('HThreeBAgent','H3_tool_loop'),('HFour','H4_tool_loop'),('StataGapOneRepair','GAP_stata_repair_zs'),
              ('StataGapAgent','GAP_stata_agent_zs'),('StataGapAvg','GAP_stata_avg')]
    for model,tag in [*TAGS.items(),('pooled','Pooled'),('cheap','Cheap'),('expensive','Expensive')]:
        for prefix,contrast in mappings:
            row=reg.loc[(model,contrast)]
            if prefix.startswith('StataGap'):
                # Python minus Stata: positive when Stata is below Python, negative when it is above (review I21).
                put(prefix+tag,-row.estimate_pp)
            else:
                put(prefix+tag,row.estimate_pp,absolute=True)
            put(prefix+tag+'SE',row.se_cr1_pp)
    def contrast(name): return reg.loc[('pooled',name),'estimate_pp']
    for agency,label in AGENCY_TAGS.items():
        put('All'+label,100*success(agency=agency))
        put('AllMeanCost'+label,cost(agency=agency),4)
        put('AllCps'+label,cps(agency=agency),4)
        put('AllRuntime'+label+'Median',select(agency=agency).provider_seconds.median())
        put('Pooled'+label+'Rounded',100*success(agency=agency),0)
        put('Pooled'+label,100*success(agency=agency))
        q=select(agency=agency)
        for kind in ('Loud','Silent'):
            put(kind+'PerHundred'+label+'Pooled',100*q.failure_category.eq(kind+' failure').mean())
        put('FailuresPerHundred'+label+'Pooled',100*(~q.primary).mean())
    put('PooledPromptGainChatbot',contrast('AUX_fewshot_chatbot'))
    put('AllPromptGainOneRepair',contrast('AUX_fewshot_self_repair'))
    put('AllPromptGainAgent',contrast('AUX_fewshot_agent'))
    put('AllRepairGainShare',100*contrast('H3_self_repair')/contrast('H3_tool_loop'),0)
    put('AllStataGapClosedRepairPct',100*(1-contrast('GAP_stata_repair_zs')/contrast('H1_stata')),0)
    for agency,label in [('chatbot','Chatbot'),('agent','Agent')]:
        put('AllStata'+label+'ZS',100*success(platform='stata',agency=agency,prompt='zero_shot'))
    put('AllPassTwelveChatbot',passes[passes.agency.eq('chatbot')&passes.k.eq(10)].pass_at_k_pct.mean())
    for prompt,label in [('zero_shot','ZS'),('few_shot','FS')]:
        for kind in ('Loud','Silent'):
            put('All'+kind+'Chatbot'+label,100*select(agency='chatbot',prompt=prompt).failure_category.eq(kind+' failure').mean())
    for platform,label in [('python','Python'),('r','R'),('stata','Stata')]:
        put('AllLoudShareChatbot'+label,100*select(platform=platform,agency='chatbot').failure_category.eq('Loud failure').mean())
    put('AllMeanCostAgencyRatio',cost(agency='agent')/cost(agency='chatbot'))
    put('AllMeanCostAgentPct',100*(cost(agency='agent')/cost(agency='chatbot')-1),0)
    put('AllCpsAgentPct',100*(cps(agency='agent')/cps(agency='chatbot')-1),0)
    put('AllRepairCostShare',100*(cost(agency='self_repair')-cost(agency='chatbot'))/(cost(agency='agent')-cost(agency='chatbot')),0)
    put('AllIncrementalCps',incremental('chatbot','agent'),3)
    put('AllMeanCostDeltaCents',100*(cost(agency='agent')-cost(agency='chatbot')))
    put('AllFewShotPct',100*abs(cost(prompt='few_shot')/cost(prompt='zero_shot')-1),0)
    put('AllStataVsPythonMeanRatio',cost(platform='stata')/cost(platform='python'))
    put('AllStataVsPythonCpsRatio',cps(platform='stata')/cps(platform='python'))
    put('AllRuntimeRatio',select(agency='agent').provider_seconds.median()/select(agency='chatbot').provider_seconds.median())
    put('AllRuntimeRepairPct',100*(select(agency='self_repair').provider_seconds.median()/select(agency='chatbot').provider_seconds.median()-1),0)
    agent=select(agency='agent')
    failed=agent[~agent.primary]
    by_task=failed.groupby('task').size().sort_values(ascending=False)
    put('AgentTasksNoFailure',agent.task.nunique()-len(by_task),0)
    put('AgentTopFiveFailureShare',100*by_task.head(5).sum()/len(failed),0)
    put('AgentFailureShareCodex',100*failed.model.eq('luna').mean(),0)
    put('AgentFailureShareLunaSix',100*failed.model.eq('luna6').mean(),0)
    cheap_agent=data[data.model.isin(six.COST_GROUPS['cheap'])&data.agency.eq('agent')]
    expensive_chatbot=data[data.model.isin(six.COST_GROUPS['expensive'])&data.agency.eq('chatbot')]
    put('ExpensiveChatbotToCheapAgentCostRatio',expensive_chatbot.cost_usd.mean()/cheap_agent.cost_usd.mean())
    for kind in ('Loud','Silent'):
        base=select(agency='chatbot').failure_category.eq(kind+' failure').mean()
        end=select(agency='agent').failure_category.eq(kind+' failure').mean()
        put(kind+'ReductionPooled',100*(1-end/base),0)
    pooled_delta=(select(agency='chatbot').failure_category.eq('Silent failure').mean()-select(agency='agent').failure_category.eq('Silent failure').mean())
    sonnet_delta=(select(model='sonnet',agency='chatbot').failure_category.eq('Silent failure').mean()-select(model='sonnet',agency='agent').failure_category.eq('Silent failure').mean())
    put('SilentReductionContributionSonnet',100*sonnet_delta/(6*pooled_delta),0)
    from manuscript import silent_change_intervals
    import paper_figure_style
    draws=paper_figure_style.task_draws(sorted(data.task.unique()))
    for tag,(low,high) in silent_change_intervals(data,draws,{**{tag:select(model=m) for m,tag in TAGS.items()},'Pooled':data}).items():
        put('SilentChangeLow'+tag,low)
        put('SilentChangeHigh'+tag,high)
    # PNAS figure counts replace hard-coded two-model point totals.
    for stem,label in [('cost_of_correctness','CostQuality'),('cost_tokens','CostTokens'),('model_agency_frontier','Frontier'),
                       ('runtime_by_agency','Runtime'),('appendix_pass_at_k','PassAtK'),('task_heterogeneity','TaskHeterogeneity')]:
        path=out / f'data/{stem}.csv'
        if path.is_file():
            put(label+'Points',len(pd.read_csv(path)),0)
    relation=pd.read_csv(robustness.RELATION).set_index('task').relation
    for values,label in [(['same','partial'],'Same'),(['family','related'],'Related')]:
        tasks=relation[relation.isin(values)].index
        q=select(agency='chatbot')
        q=q[q.task.isin(tasks)]
        put('FewShotTasks'+label,len(tasks),0)
        put('FewShotGainChatbot'+label,100*(q[q.prompt.eq('few_shot')].primary.mean()-q[q.prompt.eq('zero_shot')].primary.mean()))
    # Joint values cover all six models. Preserve the established Luna spelling.
    for model,tag in TAGS.items():
        for agency,label in [('chatbot','Chatbot'),('self_repair','Repair'),('agent','Agent')]:
            jointtag=('Luna' if model=='luna' else tag)+label
            cents=100*cost(model=model,agency=agency)
            put('JointSuccess'+jointtag,100*success(model=model,agency=agency))
            put('JointCents'+jointtag,cents,2 if cents<1 else 1)
        put('ShiftShareAgent'+tag,100*success(model=model,agency='agent',task='applied_shift_share'))
        delta=(select(model=model,agency='chatbot').failure_category.eq('Silent failure').mean()
               -select(model=model,agency='agent').failure_category.eq('Silent failure').mean())
        put('SilentReductionContribution'+tag,100*delta/(6*pooled_delta),0)
        put('TaskTruncated'+tag,100*success(model=model,task='hard_truncated_normal_effects'))
    truncated=select(task='hard_truncated_normal_effects',agency='self_repair')
    truncated=truncated[~truncated.primary]
    put('TruncatedRepairSilentSharePooled',100*truncated.failure_category.eq('Silent failure').mean(),0)
    # Self-repair transitions for all six models (first-attempt rule of paper_appendix_numbers.py).
    for model,(share,repaired) in repair_transitions(data,out).items():
        tag='Luna' if model=='luna' else TAGS[model]
        put('RepairSilentFromIncompletePct'+tag,share,0)
        put('RepairIncompleteToSuccessPct'+tag,repaired,0)
    # Values of Raul's Overleaf edits of 2026-10-25 and 2026-10-26.
    by_model=data.groupby(['model','agency']).primary.mean().unstack()*100
    put('ModelChatbotMinRounded',by_model.chatbot.min(),0)
    put('ModelChatbotMaxRounded',by_model.chatbot.max(),0)
    put('ModelAgentMinRounded',by_model.agent.min(),0)
    put('ModelAgentMax',by_model.agent.max())
    put('AllIncrementalCpsCents',100*incremental('chatbot','agent'),0)
    for model,tag in TAGS.items():
        put('IncrementalCps'+tag,incremental('chatbot','agent',model=model),2)
        q=select(model=model)
        put('CachedInputSharePct'+tag,100*q.priced_cached_input_tokens.sum()/q.main_input_tokens.sum(),0)
        for agency,label in [('chatbot','Chatbot'),('agent','Agent')]:
            value=cost(model=model,agency=agency)
            put(tag+'MeanCostSig'+label,value,2-math.floor(math.log10(value)))
    for agency,label in [('chatbot','Chatbot'),('agent','Agent')]:
        put('CostRatioSonnetLunaSix'+label,cost(model='sonnet',agency=agency)/cost(model='luna6',agency=agency))
    silent_agent=agent[agent.failure_category.eq('Silent failure')]
    put('AgentSilentShareLunaPair',100*silent_agent.model.isin(['luna','luna6']).mean(),0)
    for platform,label in [('python','Python'),('stata','Stata')]:
        put('SilentPerHundredAgentZS'+label,100*select(agency='agent',prompt='zero_shot',platform=platform)
            .failure_category.eq('Silent failure').mean())
    stata_loud=select(agency='chatbot',platform='stata')
    stata_loud=stata_loud[stata_loud.failure_category.eq('Loud failure')]
    stata_error=int((~stata_loud.execution_ok.astype(bool)).sum())
    put('StataChatbotLoud',len(stata_loud),0)
    put('StataChatbotLoudError',stata_error,0)
    put('StataChatbotLoudErrorPct',100*stata_error/len(stata_loud),0)
    put('StataChatbotLoudNoOutput',len(stata_loud)-stata_error,0)
    reach=agent.groupby(['task','platform']).primary.mean()
    shift=failed[failed.task.eq('applied_shift_share')]
    se_only=shift.scoring_notes.fillna('').str.split(' | ',regex=False).map(
        lambda notes:bool(notes) and all('standard error' in note for note in notes))
    put('ShiftShareAgentFailures',len(shift),0)
    put('ShiftShareAgentSeOnly',int(se_only.sum()),0)
    put('AgentMinTaskLanguageSuccess',100*reach.min(),0)
    put('SilentTotal',int(data.failure_category.eq('Silent failure').sum()),0)
    put('RobSeOnlySilentSharePct',100*int(rob_macros['numRobSeOnlyPrimary'].replace(',',''))
        /data.failure_category.eq('Silent failure').sum(),0)
    repair_up=[m for m in TAGS if select(model=m,agency='self_repair').failure_category.eq('Silent failure').mean()
               >select(model=m,agency='chatbot').failure_category.eq('Silent failure').mean()]
    n['numSilentRepairAboveChatbotModelsWord']=words[len(repair_up)]
    overwritten.add('numSilentRepairAboveChatbotModelsWord')
    # How far silent failures fall from the reference: the largest relative error of a failed estimate in each run.
    terms=pd.read_csv(out/'data/term_checks.csv',usecols=['position_id','ref_estimate','coef_err','coef_tol','coef_pass'])
    silent_ids=data.position_id[data.failure_category.eq('Silent failure')]
    wrong=terms[terms.position_id.isin(silent_ids)&terms.coef_pass.eq(False)&terms.coef_tol.gt(0)]
    worst=(wrong.coef_err/wrong.ref_estimate.abs()).groupby(wrong.position_id).max()
    put('SilentFarSharePct',100*worst.gt(.10).sum()/len(silent_ids),0)
    put('SilentMedianErrPct',100*worst.median(),0)
    signs=pd.read_csv(out/'tables/robustness.csv').set_index('contrast').r20_same_sign
    # The PNAS Table S3 omits the Sonnet 4.6 minus GPT-5.6 Luna row.
    put('RobSameSignMinMain',100*signs.drop('MODEL_gap').min())
    n.update(rob_macros)
    unchanged=sorted(k for k in baseline if k not in overwritten and k not in rob_macros)
    generic_left=[k for k in unchanged if 'Pooled' in k or k.startswith('numAll')]
    if generic_left:
        raise ValueError(f'Generic original-pair variables remain: {generic_left}')
    # The pass@k variables come from data/appendix_pass_at_k.csv, which six_models.py writes from the outcome after
    # the wrong-method check, so the comparison with the two-model values leaves them out.
    legacy_differences={k:dict(old=baseline[k],new=n[k]) for k in overwritten if k in baseline and n[k]!=baseline[k]
                        and any(t in k for t in ['Sonnet','Codex']) and k!='numAgentFailureShareCodex'
                        and not k.startswith('numPass')}
    if legacy_check and legacy_differences:
        raise ValueError(f'Original-model macro values changed: {legacy_differences}')
    if not write:
        return n,None
    source_meta=dict(path=str(BASE_NUMBERS.relative_to(ROOT)),sha256=sha(BASE_NUMBERS),
                     unchanged_original_scope_macros=unchanged,
                     original_pair_audit_macros=sorted(k for k in unchanged if k.startswith(('numAudit','numRepairSilent','numRepairIncomplete','numSpurious'))),
                     generic_scope='All, Pooled, total, and pooled failure variables use all six models',
                     silent_reduction_contribution='Sonnet chatbot-to-agent silent-rate decrease divided by six times the pooled decrease',
                     missing_cost='Success rates retain all runs. Cost per success uses priced runs for both total cost and successes.',
                     incremental_cost='Difference in mean costs divided by difference in success within each priced agency subset')
    write_json(out / 'data/numbers.json',dict(macros=n,provenance=source_meta))
    text='% Generated by code/paper/pnas_analysis.py.\n% Six-model PNAS sample. Audit subset provenance: data/numbers.json.\n'
    for key,value in sorted(n.items()):
        text+='%TC:macroword \\'+key+' 1\n\\newcommand{\\'+key+'}{'+value+'}\n'
    (out / 'numbers.tex').write_text(text)
    return n,source_meta


def build(out=OUT,force=False):
    out=Path(out).resolve()
    for sub in ('data','tables','work'):
        (out/sub).mkdir(parents=True,exist_ok=True)
    if out != OUT.resolve():
        # Reuse the recorded task-bootstrap figure inputs, including pass@k.
        for name in ('appendix_pass_at_k','cost_of_correctness','cost_tokens','model_agency_frontier',
                     'runtime_by_agency','task_heterogeneity'):
            source=OUT/f'data/{name}.csv'
            if source.is_file():
                (out/f'data/{name}.csv').write_bytes(source.read_bytes())
    runs,pricing,catalog=load_sources()
    (runs,terms,ses,audit),score_receipt=score_saved(runs,catalog,out,force)
    # Wrong-method alternative check (PNAS only). tolerance_runs keeps the outcome before it.
    tolerance_runs=runs.copy()
    runs,terms,ses,flagged=alternatives.apply_check(runs,terms,ses,catalog)
    if (runs.y>tolerance_runs.y).any():
        raise ValueError('The alternative check turned a failure into a success')
    alternative_summary=dict(file=str(alternatives.OUT.relative_to(ROOT)),sha256=sha(alternatives.OUT),
                             flagged_values=len(flagged),flagged_runs=int(flagged.position_id.nunique()),
                             successes_changed=int((tolerance_runs.y-runs.y).sum()),
                             successes_before=int(tolerance_runs.y.sum()),successes_after=int(runs.y.sum()))
    for name,frame in [('runs',runs),('run_pricing',pricing),('term_checks',terms),('se_checks',ses),('rescoring_audit',audit),
                       ('alternative_checks',flagged)]:
        frame.to_csv(out/f'data/{name}.csv',index=False)
    runs[['position_id','record_path','record_sha256','output_path','output_sha256']].to_csv(out/'data/selected_records.csv',index=False)
    write_json(out/'data/catalog.json',catalog)
    tolerance_contrasts,legacy_diff=regression_contrasts(tolerance_runs)
    contrasts,_=regression_contrasts(runs,legacy_check=False)
    contrasts.to_csv(out/'tables/regression_contrasts.csv',index=False)
    heterogeneity_tests(runs,out)
    checkpoint('contrasts',columns=list(GROUPS),legacy_max_abs_diff=legacy_diff,alternative_check=alternative_summary)
    estimate_flagged=frozenset(flagged.position_id[flagged.kind.eq('estimate')])
    rob_macros,rob_info=build_robustness(out,runs,catalog,contrasts,estimate_flagged)
    # The two-model comparison of the Sonnet and Luna variables uses the outcome before the alternative check.
    build_numbers(tolerance_runs,pricing,tolerance_contrasts,out,rob_macros,write=False)
    macros,number_provenance=build_numbers(runs,pricing,contrasts,out,rob_macros,legacy_check=False)
    priced=runs[['record_path','position_id','model','agency','primary']].merge(pricing[['record_path','cost_usd']],on='record_path',validate='one_to_one')
    summary=priced.groupby(['model','agency']).agg(runs=('primary','size'),priced_runs=('cost_usd','count'),successes=('primary','sum'),total_cost_usd=('cost_usd','sum')).reset_index()
    successes=priced[priced.cost_usd.notna()].groupby(['model','agency']).primary.sum()
    summary['priced_successes']=[int(successes.loc[(r.model,r.agency)]) for r in summary.itertuples()]
    summary['cost_per_success_usd']=summary.total_cost_usd/summary.priced_successes
    summary.to_csv(out/'data/cost_denominators.csv',index=False)
    receipt=dict(built_at=datetime.now(timezone.utc).isoformat(),runs=len(runs),tasks=40,models=six.ORDER,repetitions=10,
                 provider_calls=0,submitted_scripts_executed=0,scoring_rule='channel_se',raw_scoring=score_receipt,
                 alternative_check=alternative_summary,
                 legacy_contrast_max_abs_diff=legacy_diff,primary_successes=int(runs.primary.sum()),
                 cost=dict(priced_runs=int(priced.cost_usd.notna().sum()),unpriced_runs=priced[priced.cost_usd.isna()].drop(columns='cost_usd').to_dict('records'),
                           denominator_file='data/cost_denominators.csv'),
                 number_provenance=number_provenance,robustness_claim_checks=rob_info['claim_checks'],
                 computation_sources={str(p.relative_to(ROOT)):sha(p) for p in
                                      [Path(__file__),Path(robustness.__file__),Path(statistics.__file__),
                                       Path(alternatives.__file__),alternatives.OUT]},
                 inference=dict(cluster='task',clusters=40,fixed_effects='model-by-task in each combined column',
                                pooled_wild_bootstrap='not computed; p_wcb is missing for new and combined columns',
                                original_model_wild_bootstrap='retained from the verified original analysis'))
    files=[out/'numbers.tex',out/'tables/regression_contrasts.csv',out/'tables/robustness.csv',out/'tables/appendix_robustness.tex',
           *[out/f'data/{name}' for name in ('runs.csv','selected_records.csv','run_pricing.csv','term_checks.csv','se_checks.csv','catalog.json','numbers.json','robustness.json','cost_denominators.csv','alternative_checks.csv')]]
    files.append(out/'data/cross_model.csv')
    receipt['output_hashes']={str(p.relative_to(out)):sha(p) for p in files}
    write_json(out/'pnas_analysis_receipt.json',receipt)
    write_json(out/'data/snapshot.json',dict(planned=43200,analyzed=len(runs),source_tasks=40,models=six.ORDER,
                                           provider_calls=0,output_rule='channel_se',primary_successes=int(runs.primary.sum()),
                                           rescoring_changes=0,alternative_check_changes=alternative_summary['successes_changed'],
                                           complete_design=True))
    checkpoint('analysis_build',runs=len(runs),macros=len(macros))
    print(json.dumps(dict(runs=len(runs),success_pct=100*runs.primary.mean(),macros=len(macros),
                          scoring_cache_reused=score_receipt['reused'],robustness_claim_checks=rob_info['claim_checks'])))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--force-rescore',action='store_true')
    args=parser.parse_args()
    build(args.out,args.force_rescore)

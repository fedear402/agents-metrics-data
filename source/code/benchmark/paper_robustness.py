"""Appendix D of the manuscript: robustness of the main contrasts.

Usage (from the repository root):

    uv run --no-sync python code/benchmark/paper_robustness.py data/analysis/main

``code/paper/rebuild.py`` calls :func:`build` on the analysis folder after the paper assets.
``code/paper/manuscript.py`` copies ``<analysis>/tables/appendix_robustness.tex`` into
``paper/manuscript/tables/`` and adds the ``numRob*`` macros of ``<analysis>/data/robustness.json``
to ``numbers.tex``. The computation ports the verified audit
``quality_reports/audits/2026-09-30_robustness/src/robustness.py`` (with ``src/tokens.py``).

Identity (balanced design, saturated cells, task fixed effects): every Table 1 contrast is the
equal-weight mean over tasks of a task-level contrast d_i, and its CR1 standard error is
    sqrt(G/(G-1) * (N-1)/(N-K)) * sqrt(sum_i (d_i - dbar)^2) / G
with G tasks, N runs, and K = G + 17 per model (2G + 17 pooled). The build checks the full-sample
estimates and standard errors against ``<analysis>/tables/regression_contrasts.csv``.

Inputs (read only):

    <analysis>/data/runs.csv, term_checks.csv, se_checks.csv, catalog.json
    <analysis>/tables/regression_contrasts.csv
    data/production/main/fewshot_example_relation.csv     task half and example relation
    saved run outputs, through rescore_v2.primary_values   raw tokens for the precision allowance

Outputs:

    <analysis>/tables/appendix_robustness.tex   the table float (label tab:robustness)
    <analysis>/tables/robustness.csv            the table values at full precision
    <analysis>/data/robustness.json             macros, checks, seeds, draw counts, SE-only failures,
                                                and the dataset-cluster p-values
    <analysis>/tables/appendix_table1_pvalues_dataset.tex
                                                dataset-cluster p-values (analysis folder only,
                                                not installed in the manuscript)
    <analysis>/data/inference_dataset/<model>/  Stata input, do-file, log, and bootstrap_results.json

Table D1 columns 8 and 9 rescore the estimates at relative tolerance 1e-3 and 1e-4 and keep the
standard-error checks of the primary outcome in the nine tasks that score them (decision record
2026-09-25_se-scored-primary-outcome). robustness.json counts the runs that fail only the
standard-error check under the primary rule and under the two tight tolerances, by task.
scheme_outcomes also returns the loose columns of LOOSE_SCHEMES (relative tolerance 5 and 10
percent, never stricter than the primary rule). Table D1 does not print them. The six-model
build in code/paper/pnas_analysis.py can add them.

Table D3 repeats the wild cluster bootstrap of Table D2 (Stata boottest, Rademacher weights,
9,999 replications, null imposed, seed 20260701) with the 32 input datasets as clusters: the
tasks that share a catalog data_path form one cluster. The same Holm families apply (Panel A;
Panels B to D). The Stata run is cached in data/inference_dataset/<model>/ and reruns only when
its input.csv or a contrast estimate changes (production_statistics.wild_bootstrap). In that
input.csv the column 'task' holds the dataset id, and armtask (model__task) carries the task
fixed effects.

Checks before the outputs are written (each failure raises ValueError), because the Appendix D
prose states them:

    Table 1 is reproduced (1e-6).
    Every contrast keeps its sign in every inference, task-sampling, and scoring column.
    No dataset-cluster or bootstrap interval includes zero.
    The Holm decisions at the 5 percent level that change from task to dataset clusters are
    the ones that the Appendix D prose names (HOLM_CHANGES).
    The few-shot contrast has the largest relative change across the scoring rules and its
    smallest value at the 1e-4 tolerance.
    The example removes about the same share of failures under each agency condition (within 10 points).

robustness.csv also keeps the two task halves of fewshot_example_relation.csv (columns added_* and
original_*). The paper does not report them.

No model or provider call occurs.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
from paper_numbers import fixed  # noqa: E402

RELATION = ROOT / 'data/production/main/fewshot_example_relation.csv'
SEED = 20260930
N_BOOT = 4000
N_RANDOM = 5000
SUBSET = 20
TIGHT = (('tight_1e3', 1e-3), ('tight_1e4', 1e-4))
# Loose relative tolerances (5 and 10 percent). An estimate passes when it passes the primary rule
# or lies within rel * |reference| plus the precision allowance. Exact-integer checks, missing values,
# and the other primary checks keep their primary result, so a loose scheme never fails a run that
# the primary rule passes. The *_se columns add the standard-error checks of the primary outcome.
LOOSE = (('loose_5e2', 5e-2), ('loose_1e1', 1e-1))

MODELS, COLS = ('sonnet', 'luna'), ('sonnet', 'luna', 'pooled')
PLATS, PROMPTS, AGENCIES = ('python', 'r', 'stata'), ('zero_shot', 'few_shot'), ('chatbot', 'self_repair', 'agent')
ROWS = ('H3_tool_loop', 'H3_self_repair', 'H2_fewshot', 'H4_tool_loop', 'H1_stata', 'MODEL_gap')
TABLE1 = ('H3_self_repair', 'H3_tool_loop', 'H2_fewshot', 'H1_stata', 'H4_tool_loop')
# The prompt condition with one example is "one-shot" in every reader-facing text
# (Raul Sosa, manuscript commit fd96814). The contrast ids keep their internal names.
LABELS = {
    'H3_tool_loop': 'Constrained agent vs chatbot',
    'H3_self_repair': 'Self-repair vs chatbot',
    'H2_fewshot': 'One-shot vs zero-shot',
    'H4_tool_loop': r'One-shot $\times$ constrained agent',
    'H1_stata': 'Stata vs Python',
    'MODEL_gap': 'Sonnet 4.6 minus Luna',
}
# The printed label runs over the two lines of each contrast: the estimate line and the
# standard-error line. The second line of the last two rows names the sample.
PRINTED = {
    'H3_tool_loop': ('Constrained agent', 'vs chatbot'),
    'H3_self_repair': ('Self-repair', 'vs chatbot'),
    'H2_fewshot': ('One-shot', 'vs zero-shot'),
    'H4_tool_loop': (r'One-shot $\times$', 'constrained agent'),
    'H1_stata': ('Stata vs Python', 'zero-shot chatbot'),
    'MODEL_gap': ('Sonnet 4.6 minus', 'Luna, all runs'),
}
MACRO = {'H3_tool_loop': 'Agent', 'H3_self_repair': 'Repair', 'H2_fewshot': 'FewShot',
         'H4_tool_loop': 'Inter', 'H1_stata': 'Stata', 'MODEL_gap': 'Model'}
SCHEMES = ('tight_1e3', 'tight_1e4', 'tight_1e3_se', 'tight_1e4_se')
# Loose schemes that a table can add. They stay out of SCHEMES, so the two-model Table D1, its
# claim checks, and its macros do not change. The *_se pair applies the SE rule of the main outcome.
LOOSE_SCHEMES = ('loose_5e2', 'loose_1e1', 'loose_5e2_se', 'loose_1e1_se')
# Task-success levels of Table D1, Panel B: the main rule and every scheme, for all runs and under
# the constrained agent. Chatbot and self-repair rows made the float 30pt too tall (2026-10-02 build);
# their levels stay in robustness.json and the numRobRate* macros.
RATE_GROUPS = (('all', 'Success, all runs'), ('agent', 'Success, agent'))
# Variants whose SE-only failures robustness.json reports: name -> column with the estimate checks alone.
SE_ONLY = (('primary', 'primary_est'), ('tight_1e3', 'tight_1e3'), ('tight_1e4', 'tight_1e4'))

# Table D3: the wild cluster bootstrap of Table D2 with input-dataset clusters.
DATASET_BOOT = 'data/inference_dataset'
BOOT_DRAWS = 9999
ALPHA = 0.05
# Stata/MP 14 serves one process at a time. rebuild.py maps the shared lock to this file.
STATA_LOCK = Path('/private/tmp/agents_metrics_paper_stata_mp.lock')
# The Holm decisions at ALPHA that change from task to dataset clusters, as (column, contrast,
# decision with dataset clusters). D_robustness.tex states them; verify_dataset_claims checks them.
HOLM_CHANGES: frozenset[tuple[str, str, str]] = frozenset({
    ('pooled', 'H2_fewshot', 'keep'), ('pooled', 'H4_tool_loop', 'keep'), ('luna', 'H4_tool_loop', 'reject')})
WORDS = {2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}


# ---------------------------------------------------------------- task-level contrasts
def cells(d: pd.DataFrame, outcome: str, tasks: list[str]) -> dict[str, np.ndarray]:
    """Per model: success rate in percent, shape (task, platform, prompt, agency)."""
    out = {}
    for m in MODELS:
        s = d[d.model == m].groupby(['task', 'platform', 'prompt', 'agency'])[outcome].mean() * 100
        idx = pd.MultiIndex.from_product([tasks, PLATS, PROMPTS, AGENCIES])
        arr = s.reindex(idx).to_numpy(dtype=float).reshape(len(tasks), 3, 2, 3)
        if np.isnan(arr).any():
            raise ValueError(f'Empty design cell for {m} and outcome {outcome}')
        out[m] = arr
    return out


def task_contrasts(y: np.ndarray) -> dict[str, np.ndarray]:
    """Task-level contrast d_i. y has shape (task, platform, prompt, agency)."""
    ag = y.mean(axis=(1, 2))
    pr = y.mean(axis=(1, 3))
    fs = (y[:, :, 1, :] - y[:, :, 0, :]).mean(axis=1)
    return {'H3_self_repair': ag[:, 1] - ag[:, 0], 'H3_tool_loop': ag[:, 2] - ag[:, 0],
            'H2_fewshot': pr[:, 1] - pr[:, 0], 'H1_stata': y[:, 2, 0, 0] - y[:, 0, 0, 0],
            'H4_tool_loop': fs[:, 2] - fs[:, 0]}


def task_matrix(c: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per column: array (task, ROWS). MODEL_gap exists only in the pooled column (NaN otherwise)."""
    out = {}
    for m in MODELS:
        tc = task_contrasts(c[m])
        out[m] = np.column_stack([tc[k] if k in tc else np.full(len(c[m]), np.nan) for k in ROWS])
    pooled = (out['sonnet'] + out['luna']) / 2
    pooled[:, ROWS.index('MODEL_gap')] = c['sonnet'].mean(axis=(1, 2, 3)) - c['luna'].mean(axis=(1, 2, 3))
    out['pooled'] = pooled
    return out


def dof(g: int, col: str) -> float:
    n = g * 180 * (2 if col == 'pooled' else 1)
    k = g * (2 if col == 'pooled' else 1) + 17
    return float(np.sqrt(g / (g - 1) * (n - 1) / (n - k)))


def est_se(dmat: np.ndarray, col: str) -> tuple[np.ndarray, np.ndarray]:
    """dmat (task, contrast): mean and CR1 SE clustered by task."""
    g = dmat.shape[0]
    m = dmat.mean(axis=0)
    return m, dof(g, col) * np.sqrt(((dmat - m) ** 2).sum(axis=0)) / g


def cluster_se(v: np.ndarray, groups: np.ndarray, col: str) -> tuple[float, int]:
    """CR1 SE with clusters = groups of tasks (sum of the task residuals within a group)."""
    g_tasks = len(v)
    ss = pd.Series(v - v.mean()).groupby(groups).sum().to_numpy()
    G = len(ss)
    n = g_tasks * 180 * (2 if col == 'pooled' else 1)
    k = g_tasks * (2 if col == 'pooled' else 1) + 17
    return float(np.sqrt(G / (G - 1) * (n - 1) / (n - k)) * np.sqrt((ss ** 2).sum()) / g_tasks), G


# ---------------------------------------------------------------- scoring schemes
def term_tokens(analysis: Path, terms: pd.DataFrame, work: Path) -> pd.DataFrame:
    """Precision allowance per term row, from the raw output token (audit src/tokens.py).

    The raw token is used only when its float equals got_estimate exactly. Otherwise the token
    is the shortest repr of got_estimate. The result is cached in the work folder (default <analysis>/work), keyed by
    the hashes of runs.csv and term_checks.csv.
    """
    import rescore_v2 as rs
    key = hashlib.sha256((sha(analysis / 'data/runs.csv') + sha(analysis / 'data/term_checks.csv')
                          + sha(Path(rs.__file__))).encode()).hexdigest()
    cache = work / 'robustness_term_tokens.csv.gz'
    stamp = work / 'robustness_term_tokens.json'
    if cache.is_file() and stamp.is_file() and json.loads(stamp.read_text()).get('key') == key:
        return pd.read_csv(cache)
    runs = pd.read_csv(analysis / 'data/runs.csv', usecols=['position_id', 'task', 'platform', 'output_path'])
    catalog = json.loads((analysis / 'data/catalog.json').read_text())
    by_pos = terms.reset_index().groupby('position_id')
    rows = []
    for rec in runs.itertuples(index=False):
        if rec.position_id not in by_pos.groups:
            continue
        try:
            values, _ = rs.primary_values(rs.repo_file(rec.output_path), rec.platform, catalog[rec.task]['estimands'])
        except Exception:  # noqa: BLE001  an unreadable output falls back to repr
            values = {}
        for t in by_pos.get_group(rec.position_id).itertuples(index=False):
            got, token, source = t.got_estimate, None, 'none'
            v = values.get(t.term)
            if v is not None and v.est_token is not None and pd.notna(got):
                f = rs.to_float(v.est_token)
                if f is not None and f == float(got) and not v.token_fallback:
                    token, source = str(v.est_token), 'raw'
            if token is None and pd.notna(got):
                token, source = repr(float(got)), 'repr_fallback'
            rows.append({'row': t.index, 'token_source': source, 'allowance': rs.allowance(token) if token else 0.0})
    out = pd.DataFrame(rows).sort_values('row').reset_index(drop=True)
    if len(out) != len(terms):
        raise ValueError('Token extraction does not cover every term row')
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    stamp.write_text(json.dumps({'key': key}) + '\n')
    return out


def scheme_outcomes(analysis: Path, d: pd.DataFrame, work: Path) -> pd.DataFrame:
    """Run outcomes under the tight and loose relative tolerances (estimates only, and with the SE checks).

    Also returns primary_est (every estimate passes the primary rule) and se_ok (every scored
    standard error passes, or the task scores none), so y = primary_est * se_ok.
    """
    terms = pd.read_csv(analysis / 'data/term_checks.csv', low_memory=False).reset_index(drop=True)
    tok = term_tokens(analysis, terms, work)
    se = pd.read_csv(analysis / 'data/se_checks.csv', low_memory=False)
    se_ok = se.groupby('position_id').se_pass.apply(lambda s: s.astype(bool).all())
    cur = terms.coef_pass.combine_first(terms.exact_estimate_pass).combine_first(terms.estimate_pass)
    terms['cur_pass'] = cur.fillna(False).astype(bool)
    out = pd.DataFrame(index=d.position_id)
    rec_cur = terms.groupby('position_id').cur_pass.all().reindex(out.index).fillna(False)
    rec_se = se_ok.reindex(out.index).fillna(True)
    if not ((rec_cur & rec_se).astype(int).to_numpy() == d.y.to_numpy()).all():
        raise ValueError('The term and SE checks do not reproduce the outcome y')
    out['primary_est'] = rec_cur.astype(int)
    out['se_ok'] = rec_se.astype(int)
    numeric = terms.coef_pass.notna()
    err = (terms.got_estimate - terms.ref_estimate).abs()
    for tag, rel in TIGHT:
        p = terms.cur_pass.copy()
        p[numeric] = (err[numeric] <= rel * terms.ref_estimate[numeric].abs() + tok.allowance[numeric]).to_numpy()
        out[tag] = terms.assign(p=p).groupby('position_id').p.all().reindex(out.index).fillna(False).astype(int)
        # The same estimate check plus the standard-error checks of the primary outcome.
        out[f'{tag}_se'] = (out[tag].astype(bool) & rec_se.astype(bool)).astype(int)
    scaled = numeric & terms.coef_tol.gt(0)  # exact-integer checks (tolerance 0) keep the primary result
    for tag, rel in LOOSE:
        p = terms.cur_pass.copy()
        p[scaled] = (terms.cur_pass[scaled]
                     | (err[scaled] <= rel * terms.ref_estimate[scaled].abs() + tok.allowance[scaled])).to_numpy()
        out[tag] = terms.assign(p=p).groupby('position_id').p.all().reindex(out.index).fillna(False).astype(int)
        out[f'{tag}_se'] = (out[tag].astype(bool) & rec_se.astype(bool)).astype(int)
    return out.reset_index()


# ---------------------------------------------------------------- build
def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(analysis: Path) -> pd.DataFrame:
    d = pd.read_csv(analysis / 'data/runs.csv', usecols=['position_id', 'model', 'platform', 'prompt', 'agency',
                                                           'task', 'rep', 'y'])
    if len(d) != 14400 or d.task.nunique() != 40:
        raise ValueError('The robustness appendix expects 14,400 runs on 40 tasks')
    return d


def check_table1(D: dict[str, np.ndarray], analysis: Path) -> float:
    ref = pd.read_csv(analysis / 'tables/regression_contrasts.csv').set_index(['model', 'contrast'])
    worst = 0.0
    for col in COLS:
        m, se = est_se(D[col], col)
        for k in TABLE1:
            r = ref.loc[(col, k)]
            j = ROWS.index(k)
            worst = max(worst, abs(m[j] - r.estimate_pp), abs(se[j] - r.se_cr1_pp))
    if worst > 1e-6:
        raise ValueError(f'Table 1 is not reproduced (largest difference {worst})')
    return worst


def compute(analysis: Path, work: Path | None = None) -> dict:
    """Read the analysis folder. Only the token cache is written, to work (default <analysis>/work)."""
    analysis = Path(analysis).resolve()
    work = Path(work) if work else analysis / 'work'
    d = load(analysis)
    tasks = sorted(d.task.unique())
    relation = pd.read_csv(RELATION).set_index('task')
    if set(relation.index) != set(tasks):
        raise ValueError('fewshot_example_relation.csv does not list the analysis tasks')
    catalog = json.loads((analysis / 'data/catalog.json').read_text())
    by_task = dataset_groups(catalog)
    groups = np.array([by_task[t] for t in tasks])
    D = task_matrix(cells(d, 'y', tasks))
    worst = check_table1(D, analysis)
    col = 'pooled'
    X = D[col]
    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, 40, size=(N_BOOT, 40))
    subsets = np.array([rng.choice(40, SUBSET, replace=False) for _ in range(N_RANDOM)])
    sch = scheme_outcomes(analysis, d, work)
    x = d.merge(sch, on='position_id', validate='one_to_one')
    S = {name: task_matrix(cells(x, name, tasks))[col] for name in SCHEMES}
    half = relation.reindex(tasks).half.to_numpy()
    H = {h: np.flatnonzero(half == h) for h in ('added', 'original')}
    q = tdist.ppf(0.975, 39)
    rows = []
    for j, k in enumerate(ROWS):
        v = X[:, j]
        m, se = est_se(v[:, None], col)
        sed, G = cluster_se(v, groups, col)
        qg = tdist.ppf(0.975, G - 1)
        loo = (v.sum() - v) / 39
        b = v[boot].mean(axis=1)
        r = v[subsets].mean(axis=1)
        rec = dict(contrast=k, label=LABELS[k], est=m[0], se=se[0], ci_task_lo=m[0] - q * se[0], ci_task_hi=m[0] + q * se[0],
                   dataset_clusters=G, dataset_se=sed, dataset_lo=m[0] - qg * sed, dataset_hi=m[0] + qg * sed,
                   loo_min=loo.min(), loo_max=loo.max(), boot_lo=np.quantile(b, .025), boot_hi=np.quantile(b, .975),
                   r20_p5=np.quantile(r, .05), r20_p95=np.quantile(r, .95),
                   r20_same_sign=float(np.mean(np.sign(r) == np.sign(m[0]))))
        for name in SCHEMES:
            ms, ss = est_se(S[name][:, [j]], col)
            rec[f'{name}_est'], rec[f'{name}_se'] = ms[0], ss[0]
        for h, idx in H.items():
            mh, sh = est_se(v[idx][:, None], col)
            rec[f'{h}_est'], rec[f'{h}_se'] = mh[0], sh[0]
        rows.append(rec)
    table = pd.DataFrame(rows)
    # Failure rates by agency and prompt, and the share of zero-shot failures that the example removes.
    fail = 100 * (1 - d.groupby(['agency', 'prompt']).y.mean())
    removed = {a: 100 * (1 - fail[a, 'few_shot'] / fail[a, 'zero_shot']) for a in AGENCIES}
    same = relation.relation.isin(['same', 'partial'])
    se_only, se_only_runs = {}, {}
    se_rows = pd.read_csv(analysis / 'data/se_checks.csv', low_memory=False)
    se_rows = se_rows[~se_rows.se_pass.astype(bool)]
    for variant, column in SE_ONLY:
        mask = x[column].astype(bool) & ~x.se_ok.astype(bool)
        by_task = x.loc[mask, 'task'].value_counts().sort_index()
        se_only_runs[variant] = frozenset(x.loc[mask, 'position_id'])
        failed = se_rows[se_rows.position_id.isin(se_only_runs[variant])]
        gap = (failed.se_err / failed.se_ref.abs()).groupby(failed.task).min()
        se_only[variant] = dict(runs=int(mask.sum()), by_task={t: int(v) for t, v in by_task.items()},
                                missing_se_runs=int(failed[failed.se_reason.eq('se_missing')].position_id.nunique()),
                                min_relative_se_error={t: float(v) for t, v in gap.dropna().items()})
    rates = {'primary': float(100 * x.y.mean()), **{name: float(100 * x[name].mean()) for name in SCHEMES}}
    rates_by_group = {s: {'all': rates[s], **{a: float(100 * x.loc[x.agency.eq(a), 'y' if s == 'primary' else s].mean())
                                              for a in AGENCIES}}
                      for s in ('primary',) + SCHEMES}
    info = dict(table1_max_abs_diff=worst, dataset_clusters=int(len(set(groups))), seed=SEED, bootstrap_draws=N_BOOT,
                random_draws=N_RANDOM, subset_size=SUBSET, se_only_failures=se_only, success_rates=rates,
                success_rates_by_agency=rates_by_group,
                se_only_same_runs=len(set(se_only_runs.values())) == 1,
                estimate_failures={'primary': int((~x.primary_est.astype(bool)).sum())},
                failure_rates={f'{a}_{p}': float(fail[a, p]) for a in AGENCIES for p in PROMPTS},
                failures_removed=removed,
                same_calculation_tasks={h: int((same & relation.half.eq(h)).sum()) for h in H},
                half_tasks={h: len(idx) for h, idx in H.items()})
    return dict(table=table, info=info)


def verify_claims(table: pd.DataFrame) -> dict:
    """The statements of the Appendix D prose. A failure means the prose needs a revision."""
    t = table.set_index('contrast')
    checks = ['dataset_lo', 'dataset_hi', 'loo_min', 'loo_max', 'boot_lo', 'boot_hi', 'r20_p5', 'r20_p95'] + \
             [f'{s}_est' for s in SCHEMES]
    flips = {k: [c for c in checks if np.sign(t.loc[k, c]) != np.sign(t.loc[k, 'est'])] for k in t.index}
    if any(flips.values()):
        raise ValueError(f'A contrast changes sign in a robustness column; revise Appendix D: {flips}')
    rel = {k: max(abs(t.loc[k, f'{s}_est'] - t.loc[k, 'est']) / abs(t.loc[k, 'est']) for s in SCHEMES) for k in t.index}
    if max(rel, key=rel.get) != 'H2_fewshot':
        raise ValueError(f'The few-shot contrast is no longer the most sensitive to the scoring rule: {rel}')
    fs = {s: t.loc['H2_fewshot', f'{s}_est'] for s in SCHEMES}
    if min(fs, key=fs.get) != 'tight_1e4':
        raise ValueError(f'The few-shot contrast is no longer smallest at the 1e-4 tolerance: {fs}')
    zero = [k for k in t.index if t.loc[k, 'dataset_lo'] * t.loc[k, 'dataset_hi'] <= 0
            or t.loc[k, 'boot_lo'] * t.loc[k, 'boot_hi'] <= 0]
    if zero:
        raise ValueError(f'A dataset-cluster or bootstrap interval includes zero; revise Appendix D: {zero}')
    return dict(sign_flips=0, largest_relative_scoring_change=rel)


def verify_scale(info: dict) -> None:
    """Appendix D says the example removes about the same share of failures under each agency condition."""
    shares = info['failures_removed'].values()
    if max(shares) - min(shares) > 10:
        raise ValueError(f'The shares of failures removed differ by more than 10 points; revise Appendix D: {shares}')


def verify_se_claims(info: dict) -> None:
    """Appendix D: the same runs fail only the SE check under the primary rule and both tight
    tolerances, and every shift-share SE among them misses the reference by more than one percent."""
    if not info['se_only_same_runs']:
        raise ValueError('The SE-only failures differ across the scoring variants; revise Appendix D')
    gap = info['se_only_failures']['primary']['min_relative_se_error'].get('applied_shift_share')
    if gap is None or gap <= 0.01:
        raise ValueError(f'A shift-share SE-only failure is within one percent of the reference: {gap}')


def macros(table: pd.DataFrame, info: dict) -> dict[str, str]:
    """numRob* macros for numbers.tex. Signed values print inside $...$ in the prose."""
    t = table.set_index('contrast')
    n = {'numRobDatasetClusters': str(info['dataset_clusters']),
         'numRobRandomDraws': f"{info['random_draws']:,}",
         'numRobSubsetSize': str(info['subset_size']),
         'numRobSameSignMin': fixed(100 * t.r20_same_sign.min(), 1)}
    inc = 100 * (t.dataset_se / t.se - 1)
    n['numRobClusterSeIncreaseMin'] = fixed(inc.min(), 0)
    n['numRobClusterSeIncreaseMax'] = fixed(inc.max(), 0)
    for k, name in MACRO.items():
        n[f'numRob{name}Est'] = fixed(t.loc[k, 'est'], 1)
        for s, tag in (('tight_1e3', 'TightThree'), ('tight_1e4', 'TightFour'), ('tight_1e3_se', 'TightThreeSE'),
                       ('tight_1e4_se', 'TightFourSE')):
            n[f'numRob{name}{tag}'] = fixed(t.loc[k, f'{s}_est'], 1)
    group_tag = {'all': '', 'chatbot': 'Chatbot', 'self_repair': 'Repair', 'agent': 'Agent'}
    for s, tag in (('primary', 'Primary'), ('tight_1e3', 'TightThree'), ('tight_1e4', 'TightFour'),
                   ('tight_1e3_se', 'TightThreeSE'), ('tight_1e4_se', 'TightFourSE')):
        n[f'numRobRate{tag}'] = fixed(info['success_rates'][s], 1)
        for g, gtag in group_tag.items():
            if gtag:
                n[f'numRobRate{gtag}{tag}'] = fixed(info['success_rates_by_agency'][s][g], 1)
    for s, tag in (('primary', 'Primary'), ('tight_1e3', 'TightThree'), ('tight_1e4', 'TightFour')):
        rec = info['se_only_failures'][s]
        n[f'numRobSeOnly{tag}'] = f"{rec['runs']:,}"
        n[f'numRobSeOnlyTasks{tag}'] = WORDS.get(len(rec['by_task']), str(len(rec['by_task']))).lower()
        n[f'numRobSeOnlyShiftShare{tag}'] = f"{rec['by_task'].get('applied_shift_share', 0):,}"
    n['numRobSeOnlyShiftShareMinGap'] = fixed(
        100 * info['se_only_failures']['primary']['min_relative_se_error']['applied_shift_share'], 1)
    pretty = {'chatbot': 'Chatbot', 'self_repair': 'Repair', 'agent': 'Agent'}
    for a in AGENCIES:
        n[f'numRobFailZS{pretty[a]}'] = fixed(info['failure_rates'][f'{a}_zero_shot'], 1)
        n[f'numRobFailFS{pretty[a]}'] = fixed(info['failure_rates'][f'{a}_few_shot'], 1)
        n[f'numRobFailRemoved{pretty[a]}'] = fixed(info['failures_removed'][a], 0)
    assert all(k.isalpha() for k in n), n
    return n


def f1(v: float) -> str:
    s = fixed(v, 1)
    return f'${{-}}{s[1:]}$' if s.startswith('-') else f'${s}$'


def interval_cells(lo: float, hi: float) -> str:
    """An interval as two cells that the column pair r@{, }l aligns on the comma."""
    return f'[{f1(lo)} & {f1(hi)}]'


# Panel A: estimate, then four intervals, each split over two columns aligned on the comma.
PAIR = r'r@{\extracolsep{0pt}, }l@{\extracolsep{\fill}\hspace{2pt}}'
PANEL_A_SPEC = r'@{\extracolsep{\fill}}l r ' + ' '.join([PAIR] * 4) + '@{}'
# Panel B: five right-aligned columns at natural width. Two spanners group the tolerance columns,
# because fixed 1.6cm columns overflowed the text block by 27pt (2026-10-01 build).
PANEL_B_SPEC = r'@{\extracolsep{\fill}}l *{5}{r}@{}'
# Space after each contrast and a tighter row height keep the float inside the text block
# (with the two task-success rows of 2026-10-02, 0.25pt left the float 0.6pt too tall;
# at 3pt and arraystretch 1.0 it was 52.6pt too tall for the page in the 2026-09-30 build, and with
# the two SE-check columns of Panel B it was 16pt too tall at arraystretch 0.86 on 2026-10-01).
GAP = '0.1pt'
# Space below the column-number row, so that (2), (7) and (9) do not touch the header text below.
# The space comes from the panel headings (addlinespace 0pt), because the float has about 1.2pt of slack (2026-10-02).
NUMBER_GAP = '1.5pt'


def render_table(table: pd.DataFrame, info: dict) -> str:
    """Two panels: inference and task sampling (five columns), then scoring rule (five), with task-success levels.

    Tidy layout of the 2026-09-30 table-design audit: no scaling, one tabular* per panel in
    one float, italic panel headings at the left margin, intervals aligned on the comma, and
    each contrast label over two lines. The caption uses title case without a final period.
    """
    panel_a = [('(1)', '', 'Estimate'), ('(2)', 'Dataset', 'clusters'), ('(3)', 'Leave one', 'task out'),
               ('(4)', 'Task', 'bootstrap'), ('(5)', 'Random', f"{info['subset_size']} tasks")]
    # Panel B: number, then the spanner line (None inside a two-column spanner), then the second line.
    panel_b = [('(6)', 'Main', 'rule'), ('(7)', 'Estimates only', r'$10^{-3}$'), ('(8)', None, r'$10^{-4}$'),
               ('(9)', 'With SE checks', r'$10^{-3}$'), ('(10)', None, r'$10^{-4}$')]
    lines = [
        '% Appendix D robustness table. Generated by code/benchmark/paper_robustness.py; do not edit by hand.',
        '% Sources: <analysis>/data/runs.csv, term_checks.csv, se_checks.csv, catalog.json;',
        '% data/production/main/fewshot_example_relation.csv.',
        r'\begin{table}[!htbp]', r'\setstretch{1}', r'\centering',
        r'\caption{Robustness of the Main Contrasts, Pooled over the Two \mbox{Models}}', r'\label{tab:robustness}',
        r'\begin{footnotesize}', r'\setlength{\tabcolsep}{1pt}', r'\renewcommand{\arraystretch}{0.8}',
    ]

    def wide(text: str) -> str:
        return r'\multicolumn{2}{c}{' + text + '}'

    # Panel A.
    lines += [r'\begin{tabular*}{\textwidth}{' + PANEL_A_SPEC + '}', r'\toprule',
              r'\multicolumn{10}{@{}l}{\textit{Panel A. Inference and task sampling}} \\', r'\addlinespace[0pt]']
    for level in range(3):
        cells = [panel_a[0][level]] + [wide(c[level]) for c in panel_a[1:]]
        lines.append(('Contrast' if level == 2 else '') + ' & ' + ' & '.join(cells)
                     + (rf' \\[{NUMBER_GAP}]' if level == 0 else r' \\'))
    lines.append(r'\midrule')
    for r in table.itertuples(index=False):
        first, second = PRINTED[r.contrast]
        intervals = [interval_cells(r.dataset_lo, r.dataset_hi), interval_cells(r.loo_min, r.loo_max),
                     interval_cells(r.boot_lo, r.boot_hi), interval_cells(r.r20_p5, r.r20_p95)]
        lines.append(' & '.join([first, f1(r.est) + r'\phantom{)}', *intervals]) + r' \\')
        share = wide(f'{fixed(100 * r.r20_same_sign, 1)}\\%')
        lines.append(' & '.join([second, f'({fixed(r.se, 1)})', wide(''), wide(''), wide(''), share]) + rf' \\[{GAP}]')
    lines += [r'\midrule', r'\end{tabular*}', '']
    # Panel B.
    lines += [r'\begin{tabular*}{\textwidth}{' + PANEL_B_SPEC + '}',
              r'\multicolumn{6}{@{}l}{\textit{Panel B. Scoring rule}} \\', r'\addlinespace[0pt]']
    lines.append(' & ' + ' & '.join(c[0] for c in panel_b) + rf' \\[{NUMBER_GAP}]')
    spanners = [wide(c[1]) if c[1] and nxt[1] is None else c[1] for c, nxt in zip(panel_b, panel_b[1:] + [('', '')])
                if c[1] is not None]
    lines.append(' & ' + ' & '.join(spanners) + r' \\')
    lines.append(r'\cmidrule(lr){3-4} \cmidrule(lr){5-6}')
    lines.append('Contrast & ' + ' & '.join(c[2] for c in panel_b) + r' \\')
    lines.append(r'\midrule')
    for r in table.itertuples(index=False):
        first, second = PRINTED[r.contrast]
        estimates = [r.est, r.tight_1e3_est, r.tight_1e4_est, r.tight_1e3_se_est, r.tight_1e4_se_est]
        errors = [r.se, r.tight_1e3_se, r.tight_1e4_se, r.tight_1e3_se_se, r.tight_1e4_se_se]
        lines.append(' & '.join([first] + [f1(e) + r'\phantom{)}' for e in estimates]) + r' \\')
        lines.append(' & '.join([second] + [f'({fixed(e, 1)})' for e in errors]) + rf' \\[{GAP}]')
    # Task-success levels under each scoring rule, in percent.
    rates = info['success_rates_by_agency']
    lines.append(r'\midrule')
    for group, label in RATE_GROUPS:
        cells = [fixed(rates[s][group], 1) + r'\phantom{)}' for s in ('primary',) + SCHEMES]
        lines.append(' & '.join([label] + cells) + r' \\')
    lines += [
        r'\bottomrule', r'\end{tabular*}', r'\end{footnotesize}',
        r'\begin{flushleft}\footnotesize',
        r'\textit{Notes:} Estimates are differences in task-success rates, in percentage points.'
        r' The first five rows equal the contrasts of column 3 of \Cref{tab:estimates_three_models}.',
        'The fourth row is the change in the one-shot effect from the chatbot to the constrained agent (Panel D).',
        r'Standard errors clustered by task are in parentheses.',
        f"Column 2 is the 95 percent interval with standard errors clustered by the {info['dataset_clusters']} input datasets.",
        'Column 3 is the range of the estimates when one task is left out.',
        f"Column 4 is the 95 percent interval of a bootstrap over tasks ({info['bootstrap_draws']:,} draws).",
        f"Column 5 is the 5th to 95th percentile range of the estimates on {info['random_draws']:,} random sets of "
        f"{info['subset_size']} tasks, and the second line is the share of sets with the sign of column 1.",
        'Column 6 uses the main scoring rule, and the last two rows give task success in percent.',
        'Columns 7 to 10 score each estimate at a relative tolerance of $10^{-3}$ or $10^{-4}$ with no absolute floor, '
        'plus half a unit of the last reported decimal for values with four or more decimals, '
        'and columns 9 and 10 add the standard-error checks.',
        r'\end{flushleft}', r'\end{table}',
    ]
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------- Table D3: dataset clusters
def dataset_groups(catalog: dict) -> dict[str, str]:
    """Task -> input dataset id (the catalog data_path relative to the repository root)."""
    out = {}
    for task, rec in catalog.items():
        path = Path(rec['data_path']).resolve()
        out[task] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return out


def regression_frame(analysis: Path, model: str, groups: dict[str, str]) -> pd.DataFrame:
    """The Table 1 regression input of production_paper_assets.regression_tables plus a dataset column."""
    import factorial_analysis as fa
    runs = pd.read_csv(analysis / 'data/runs.csv', usecols=['model', 'platform', 'prompt', 'agency', 'task', 'y'])
    frame = (runs if model == 'pooled' else runs[runs.model.eq(model)]).copy()
    frame['success'] = frame.y.astype(float)
    agency = {'chatbot': 'no_feedback', 'self_repair': 'self_repair', 'agent': 'tool_loop'}
    frame['cell'] = [fa.cell_index(r.platform, r.prompt, agency[r.agency]) for r in frame.itertuples()]
    frame['armtask'] = frame.model + '__' + frame.task
    frame['dataset'] = frame.task.map(groups)
    if frame.dataset.isna().any() or set(frame.cell) != set(range(1, 19)):
        raise ValueError(f'Incomplete regression frame for {model}')
    return frame.reset_index(drop=True)


def run_dataset_bootstrap(analysis: Path) -> dict[str, dict]:
    """Run (or reuse) the Stata wild cluster bootstrap with dataset clusters for the three columns."""
    import factorial_analysis as fa
    import production_statistics as ps
    import statsmodels.formula.api as smf
    from stata_log_privacy import strip_license_details
    analysis = Path(analysis).resolve()
    groups = dataset_groups(json.loads((analysis / 'data/catalog.json').read_text()))
    ws = ps.weights()
    out = {}
    for model in COLS:
        frame = regression_frame(analysis, model, groups)
        fe = 'armtask' if model == 'pooled' else 'task'
        fit = smf.ols(f'success ~ C({fe}) + C(cell, Treatment(1))', frame).fit(
            cov_type='cluster', cov_kwds={'groups': pd.factorize(frame.dataset)[0]})
        contrasts = fa.contrast_table(fit, ws)
        for est in contrasts.values():
            if abs(est['estimate']) < 1e-12:
                est['estimate'] = 0.0
        work = analysis / DATASET_BOOT / model
        # production_statistics.wild_bootstrap clusters on its 'task' column and, with pooled=True,
        # absorbs 'armtask' fixed effects. armtask is model__task, so it equals task fixed effects
        # within one model. Passing the dataset id in the 'task' column therefore gives the Table 1
        # regression of every column with dataset clusters, and the function body stays as migrated.
        stata = frame[['success', 'cell', 'armtask']].assign(task=frame.dataset)
        STATA_LOCK.parent.mkdir(parents=True, exist_ok=True)
        with open(STATA_LOCK, 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            boot = ps.wild_bootstrap(stata, ws, contrasts, work, True)
        log = work / 'inference.log'
        if log.is_file():
            log.write_text(strip_license_details(log.read_text()))
        (work / 'bootstrap_results.json').write_text(json.dumps(boot, indent=2) + '\n')
        out[model] = boot
    return out


def dataset_pvalues(analysis: Path) -> dict[str, dict[str, dict]]:
    """Per column and Table 1 contrast: task- and dataset-cluster bootstrap p and their Holm p.

    Reads the cached dataset-cluster results and checks the estimates against Table 1.
    """
    import paper_table1 as t1
    analysis = Path(analysis).resolve()
    task = t1.load_contrasts(analysis)
    out: dict[str, dict[str, dict]] = {}
    for m in t1.MODELS:
        boot = json.loads((analysis / DATASET_BOOT / m / 'bootstrap_results.json').read_text())
        rows: dict[str, dict] = {}
        for panel in t1.PANELS:
            for cid, _ in panel['rows']:
                b = boot[cid]
                if abs(100 * b['estimate_stata'] - task[m][cid]['estimate_pp']) > 1e-6:
                    raise ValueError(f'Dataset-cluster estimate differs from Table 1: {m} {cid}')
                if b.get('p_wcb') is None or b['bootstrap_draws'] != BOOT_DRAWS:
                    raise ValueError(f'Dataset-cluster bootstrap incomplete: {m} {cid}')
                rows[cid] = dict(family=panel['family'], p_dataset=float(b['p_wcb']), se_dataset_pp=100 * b['se_stata'],
                                 bootstrap_draws=int(b['bootstrap_draws']), p_task=task[m][cid]['p_wcb'],
                                 holm_task=task[m][cid]['p_holm'])
        for family in ('primary', 'secondary'):
            ps = {cid: r['p_dataset'] for cid, r in rows.items() if r['family'] == family}
            for cid, adj in t1.holm(ps).items():
                rows[cid]['holm_dataset'] = adj
        out[m] = rows
    return out


def holm_changes(rows: dict[str, dict[str, dict]]) -> list[dict]:
    """The Holm decisions at ALPHA that differ between task and dataset clusters."""
    out = []
    for m, by_contrast in rows.items():
        for cid, r in by_contrast.items():
            task_rej, data_rej = r['holm_task'] < ALPHA, r['holm_dataset'] < ALPHA
            if task_rej != data_rej:
                out.append(dict(model=m, contrast=cid, task='reject' if task_rej else 'keep',
                                dataset='reject' if data_rej else 'keep'))
    return out


def verify_dataset_claims(rows: dict[str, dict[str, dict]]) -> list[dict]:
    """Appendix D names the Holm decisions that change with dataset clusters (HOLM_CHANGES)."""
    changes = holm_changes(rows)
    found = frozenset((c['model'], c['contrast'], c['dataset']) for c in changes)
    if found != HOLM_CHANGES:
        raise ValueError(f'The Holm decisions that change with dataset clusters differ from Appendix D: {sorted(found)}')
    return changes


def render_dataset_pvalues(rows: dict[str, dict[str, dict]], analysis: Path) -> str:
    """Table D3: the layout of Table D2 with the dataset-cluster p-values."""
    import paper_table1 as t1
    analysis = Path(analysis).resolve()
    groups = dataset_groups(json.loads((analysis / 'data/catalog.json').read_text()))
    sizes = pd.Series(groups).value_counts()
    shared = sizes[sizes > 1]
    rel = t1.display_path(analysis)
    heads = ' & '.join(r'\multicolumn{2}{c}{' + ' '.join(t1.COLUMN_HEADS[m]) + '}' for m in t1.MODELS)
    lines = [
        '% Table D3, wild cluster bootstrap p-values with dataset clusters. Generated by '
        'code/benchmark/paper_robustness.py; do not edit by hand.',
        f'% Sources: {rel}/data/inference_dataset/<model>/bootstrap_results.json, {rel}/data/catalog.json,',
        f'% and {rel}/tables/regression_contrasts.csv. Rounding: code/benchmark/paper_numbers.py (half up).',
        *t1.float_head(r'Wild Cluster Bootstrap $p$-Values with Dataset Clusters', 'tab:pvalues_dataset'),
        r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}'
        r'>{\raggedright\arraybackslash\hangindent=2em\hangafter=1}p{4.9cm} rr rr rr@{}}',
        r'\toprule',
        ' & ' + heads + r' \\',
        r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(l){6-7}',
        ' & ' + ' & '.join([r'$p$ & Holm $p$'] * 3) + r' \\',
        r'\midrule',
    ]
    for index, panel in enumerate(t1.PANELS):
        if index:
            lines.append(r'\addlinespace[3pt]')
        lines.append(t1.panel_heading(panel['title'], 7))
        for cid, label in panel['rows']:
            cells = [t1.fmt_p(rows[m][cid]['p_dataset']) + ' & ' + t1.fmt_p(rows[m][cid]['holm_dataset'])
                     for m in t1.MODELS]
            lines.append(r'\quad ' + label + ' & ' + ' & '.join(cells) + r' \\')
    # Shorter notes (2026-10-01 appendix reductions, proposal D8): Table D3 follows Table D2,
    # whose notes state the bootstrap, the Holm families, and the print rule.
    notes = (
        r'\textit{Notes:} The bootstrap, the Holm adjustment, and the print format are those of '
        r'\Cref{tab:pvalues}, with the ' + str(len(sizes)) + ' input datasets as clusters.',
        'Tasks that read the same input data file form one cluster.',
    )
    return '\n'.join(lines + t1.float_tail(notes)) + '\n'


def build(analysis: Path) -> dict:
    """Write the table, the CSV, and the macros for one analysis folder. Returns the JSON record."""
    analysis = Path(analysis).resolve()
    res = compute(analysis)
    table, info = res['table'], res['info']
    info['claims'] = verify_claims(table)
    verify_scale(info)
    verify_se_claims(info)
    (analysis / 'tables').mkdir(parents=True, exist_ok=True)
    # Table D3: dataset-cluster bootstrap (Stata, cached in data/inference_dataset/<model>/).
    run_dataset_bootstrap(analysis)
    rows = dataset_pvalues(analysis)
    changes = verify_dataset_claims(rows)
    info['holm_changes_dataset_clusters'] = changes
    n = macros(table, info)
    (analysis / 'tables/appendix_robustness.tex').write_text(render_table(table, info))
    table.to_csv(analysis / 'tables/robustness.csv', index=False)
    (analysis / 'tables/appendix_table1_pvalues_dataset.tex').write_text(render_dataset_pvalues(rows, analysis))
    record = dict(info, macros=n, dataset_cluster_pvalues=rows,
                  input_sha256={p.name: sha(p) for p in (
        analysis / 'data/runs.csv', analysis / 'data/term_checks.csv', analysis / 'data/se_checks.csv', RELATION)},
        source_sha256=sha(Path(__file__)))
    (analysis / 'data/robustness.json').write_text(json.dumps(record, indent=2, default=float) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('analysis', type=Path, nargs='?', default=ROOT / 'data/analysis/main')
    args = parser.parse_args()
    import factorial_analysis as fa
    # The saved inference used Stata/MP 14 with boottest, as in code/paper/rebuild.py.
    fa.STATA_BIN = os.environ.get('STATA_BIN', '/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp')
    rec = build(args.analysis)
    print(json.dumps({k: rec[k] for k in ('table1_max_abs_diff', 'dataset_clusters', 'bootstrap_draws', 'random_draws')}))

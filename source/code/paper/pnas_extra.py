"""Extra analyses for the six-model PNAS paper, computed from saved data only.

Run from the project root:

    uv run --no-sync python code/paper/pnas_extra.py [--data DIR] [--out DIR] [--only NAME,...]

The module reads the current outputs of ``code/paper/pnas_analysis.py`` each time it runs
(``data/analysis/six_models/data/*.csv``, ``catalog.json``), the saved run records under each
``record_path``, and the original two-model analysis in ``data/analysis/main`` for one check.
It makes no provider calls and executes no submitted script. It writes CSV files, LaTeX table
bodies, and ``extra_receipt.json`` to ``data/analysis/six_models/extra/``. The authors decide
later which results enter the SI.

Analyses (peer review 2026-10-28 and PNAS fit 6.1 identifiers in brackets):

1. ``table1_inference``  Wild cluster bootstrap, Holm, and Benjamini-Hochberg p-values and
   informative-task counts for every cell of Table 1 [M7]. In the balanced design the
   null-imposed wild cluster bootstrap of a contrast is a sign-flip test on the 40 task-level
   contrast values. ``wcb_check.csv`` compares it with the stored Stata boottest values.
2. ``se_only_silent``    Silent failures split into estimate failures and SE-only failures [M6].
3. ``majority_vote``     The most frequent answer of k chatbot runs [A1].
4. ``shared_wrong``      Wrong values shared across models and repetitions [A2].
5. ``self_checks``       What the constrained agent's own check reports [I14, A5].
6. ``small_numbers``     I04, I10, I13, I17, and I42.

Intervals are 95 percent percentile intervals of the task cluster bootstrap of the paper
figures (``paper_figure_style.task_draws``: 4,000 draws, seed 20260925).
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
from scipy.stats import t as tdist

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/paper'), str(ROOT / 'code/benchmark')]
import paper_figure_style as style  # noqa: E402
import paper_table1  # noqa: E402
import pnas_analysis as pa  # noqa: E402
import production_statistics as statistics  # noqa: E402
import six_models as six  # noqa: E402

DATA = ROOT / 'data/analysis/six_models'
OUT = DATA / 'extra'
ORIGINAL = ROOT / 'data/analysis/main'
WORK = ROOT / 'quality_reports/audits/2026-10-29_pnas-fixes/work/extra'
HAND_LABELS = WORK / 'self_check_hand_labels.csv'

MODELS = list(six.ORDER)
COLUMNS = [*MODELS, 'pooled']
GROUPS = dict(pa.GROUPS)
PROMPTS = ['zero_shot', 'few_shot']
CELL = ['model', 'task', 'platform', 'prompt', 'agency']
MODEL_NAMES = dict(sonnet='Claude Sonnet 4.6', luna='GPT-5.6 Luna', haiku='Claude Haiku 5.5',
                   luna6='GPT-6 Luna', opus='Claude Opus 5.5', sol='GPT-6 Sol', pooled='Total (pooled)',
                   cheap='Low-cost models', expensive='High-cost models', all='All models')
AGENCY_NAMES = dict(chatbot='Chatbot', self_repair='Self-repair', agent='Constrained agent', all='All conditions')

# Wild cluster bootstrap: Rademacher weights, the draw count and seed number of the Stata boottest
# calls in production_statistics.wild_bootstrap. NumPy draws differ from the Stata draws.
WCB_DRAWS = 9999
WCB_SEED = 20260701
# Ties: |t*| counts as at least |t| within this relative margin (exact ties occur with discrete rates).
TIE_MARGIN = 1e-9

# The PNAS Table 1 rows: paper_table1.PANELS with the three PNAS prompt rows after the pooled
# prompt row (paper/pnas/build_pnas.py, PROMPT_ROWS). Holm families as in paper_table1 and the
# decision record 2026-09-25_table1-p-values-and-holm.md: Panel A is the primary family, and
# Panels B to D form the secondary family, separately in each column.
PNAS_PROMPT_ROWS = (('AUX_fewshot_chatbot', 'One example, chatbot'),
                    ('AUX_fewshot_self_repair', 'One example, self-repair'),
                    ('AUX_fewshot_agent', 'One example, constrained agent'))


def table1_rows() -> list[dict]:
    """The Table 1 rows in print order, with panel title and Holm family."""
    rows = []
    for panel in paper_table1.PANELS:
        title = re.sub(r'^Panel [A-Z]\.\s*', '', panel['title'])
        extra = PNAS_PROMPT_ROWS if 'H2_fewshot' in dict(panel['rows']) else ()
        for contrast, label in (*panel['rows'], *extra):
            rows.append(dict(panel=title, family=panel['family'], contrast=contrast, label=label))
    return rows


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def truthy(series: pd.Series) -> pd.Series:
    """True where a CSV column holds True (bool or text). Missing values are False."""
    return series.map(lambda v: v is True or v == 1 or str(v).strip().lower() == 'true'
                      if not (isinstance(v, float) and math.isnan(v)) else False).astype(bool)


def falsy(series: pd.Series) -> pd.Series:
    """True where a CSV column holds False (bool or text). Missing values are False."""
    return series.map(lambda v: v is False or str(v).strip().lower() == 'false'
                      if not (isinstance(v, float) and math.isnan(v)) else False).astype(bool)


def load(data: Path = DATA) -> dict:
    """Read the current analysis outputs. Nothing is cached between runs."""
    data = Path(data)
    runs = pd.read_csv(data / 'data/runs.csv', low_memory=False)
    for column in ('primary', 'execution_ok', 'output_available'):
        runs[column] = truthy(runs[column])
    if 'y' in runs and not runs.y.astype(int).eq(runs.primary.astype(int)).all():
        raise ValueError('runs.csv: y differs from primary')
    runs['y'] = runs.primary.astype(int)
    expected = {'Success', 'Silent failure', 'Loud failure'}
    if not set(runs.failure_category) <= expected:
        raise ValueError(f'Unknown failure categories: {set(runs.failure_category) - expected}')
    if not runs.primary.eq(runs.failure_category.eq('Success')).all():
        raise ValueError('runs.csv: primary and failure_category disagree')
    terms = pd.read_csv(data / 'data/term_checks.csv', low_memory=False)
    ses = pd.read_csv(data / 'data/se_checks.csv', low_memory=False)
    pricing = pd.read_csv(data / 'data/run_pricing.csv', low_memory=False)
    catalog = json.loads((data / 'data/catalog.json').read_text())
    return dict(runs=runs, terms=terms, ses=ses, pricing=pricing, catalog=catalog, data=data)


def task_cells(runs: pd.DataFrame, models: list[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    """Success rates (percent) per task and design cell, one 40 x 18 matrix per model.

    The cell order is that of pnas_analysis.model_cells, so the contrast weights apply unchanged.
    """
    tasks = sorted(runs.task.unique())
    index = pd.MultiIndex.from_product([tasks, six.PLATFORMS, PROMPTS, six.AGENCIES])
    cells = {}
    for model in models:
        rates = runs[runs.model.eq(model)].groupby(['task', 'platform', 'prompt', 'agency']).y.mean().reindex(index)
        if rates.isna().any():
            raise ValueError(f'Missing design cells for {model}')
        cells[model] = 100 * rates.to_numpy().reshape(len(tasks), 18)
    return tasks, cells


def boot_index(tasks: list[str]) -> np.ndarray:
    """Task positions of the paper's task cluster bootstrap (4,000 draws, seed 20260925)."""
    return style.task_draws(list(tasks))


def interval(values: np.ndarray) -> tuple[float, float]:
    return style.percentile_interval(values)


def task_totals(frame: pd.DataFrame, tasks: list[str], column: str | None = None) -> np.ndarray:
    """Per-task sum of a column (or the row count), in task order."""
    if column is None:
        return frame.groupby('task').size().reindex(tasks, fill_value=0).to_numpy(dtype=float)
    return frame.groupby('task')[column].sum().reindex(tasks, fill_value=0).to_numpy(dtype=float)


def subset(frame: pd.DataFrame, **levels: str) -> pd.DataFrame:
    """Rows at the given levels. The level 'all' keeps every row of that column."""
    mask = np.ones(len(frame), dtype=bool)
    for column, value in levels.items():
        if value != 'all':
            mask &= frame[column].eq(value).to_numpy()
    return frame[mask]


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def fixed(value: float, decimals: int = 1) -> str:
    return pa.fixed(value, decimals, thousands=True)


def source_comment(name: str) -> list[str]:
    return [f'% {name}. Generated by code/paper/pnas_extra.py from data/analysis/six_models/data.',
            '% Table body only: the caption, label, and notes belong to the SI source.']


# ---------------------------------------------------------------------------
# 1. Inference for Table 1 [M7]
# ---------------------------------------------------------------------------
def rademacher(groups: int, draws: int = WCB_DRAWS, seed: int = WCB_SEED) -> np.ndarray:
    """Rademacher weights, one row per draw and one column per task cluster."""
    return 2.0 * np.random.default_rng(seed).integers(0, 2, size=(draws, groups)) - 1.0


def sign_flip_test(values: np.ndarray, models: int = 1, weights: np.ndarray | None = None) -> dict:
    """Null-imposed wild cluster bootstrap of task-level contrast values.

    In the balanced design with task (or model-by-task) fixed effects and task clusters, a
    contrast estimate is the mean of the task-level contrast values v_g, and its CR1 standard
    error is c * sqrt(sum (v_g - mean)^2) / G. The restricted residuals keep v_g as their task
    contrast, so a Rademacher draw w gives the estimate mean(w v) and the standard error
    c * sqrt(sum (w_g v_g - mean(w v))^2) / G. The test compares |t*| with |t|.

    Returns arrays over the columns of ``values`` (G x C): estimate, se, t, the count of draws
    with |t*| >= |t|, the count with |t*| > |t|, p (count + 1) / (B + 1) as in the stored Stata
    values, and p_raw = count / B.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    g = v.shape[0]
    w = rademacher(g) if weights is None else weights
    factor = pa.cr1_factor(g, models)
    mean = v.mean(axis=0)
    squares = (v ** 2).sum(axis=0)
    se = factor * np.sqrt(np.maximum(squares - g * mean ** 2, 0)) / g
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(se > 1e-10, mean / se, np.nan)
        star = w @ v / g
        se_star = factor * np.sqrt(np.maximum(squares[None, :] - g * star ** 2, 0)) / g
        t_star = np.abs(star / se_star)
    t_star = np.where(se_star > 1e-12, t_star, 0.0)
    level = np.abs(t)
    count_ge = (t_star >= level * (1 - TIE_MARGIN)).sum(axis=0)
    count_gt = (t_star > level * (1 + TIE_MARGIN)).sum(axis=0)
    draws = w.shape[0]
    undefined = ~np.isfinite(level)
    p = np.where(undefined, np.nan, (count_ge + 1) / (draws + 1))
    p_strict = np.where(undefined, np.nan, (count_gt + 1) / (draws + 1))
    return dict(estimate=mean, se=se, t=t, count_ge=count_ge, count_gt=count_gt, p=p, p_strict=p_strict,
                p_raw=np.where(undefined, np.nan, count_ge / draws), draws=draws)


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values. Missing values enter the family as 1."""
    p = np.where(np.isfinite(pvalues), pvalues, 1.0)
    order = np.argsort(p, kind='stable')
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values. Missing values enter the family as 1."""
    p = np.where(np.isfinite(pvalues), pvalues, 1.0)
    m = len(p)
    order = np.argsort(p, kind='stable')
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    adjusted = np.empty(m)
    adjusted[order] = ranked
    return adjusted


def contrast_values(runs: pd.DataFrame, members: list[str], contrasts: list[str]) -> np.ndarray:
    """Task-level contrast values (40 x C) of the mean of the member models' cells."""
    _, cells = task_cells(runs, members)
    pooled = np.mean([cells[m] for m in members], axis=0)
    weights = statistics.weights()
    return np.column_stack([pooled @ weights[c] for c in contrasts])


def column_inference(runs: pd.DataFrame, column: str, members: list[str]) -> pd.DataFrame:
    contrasts = list(statistics.weights())
    values = contrast_values(runs, members, contrasts)
    test = sign_flip_test(values, len(members))
    q = tdist.ppf(.975, values.shape[0] - 1)
    rows = []
    for j, contrast in enumerate(contrasts):
        estimate, se = float(test['estimate'][j]), float(test['se'][j])
        informative = int((np.abs(values[:, j]) > 1e-10).sum())
        p_t = float(2 * tdist.sf(abs(estimate / se), values.shape[0] - 1)) if se > 1e-10 else np.nan
        rows.append(dict(column=column, members=','.join(members), contrast=contrast,
                         estimate_pp=0.0 if abs(estimate) < 1e-10 else estimate,
                         se_cr1_pp=0.0 if se < 1e-10 else se, ci_low_pp=estimate - q * se,
                         ci_high_pp=estimate + q * se, p_cluster_t=p_t, p_wcb=float(test['p'][j]),
                         p_wcb_strict=float(test['p_strict'][j]), p_wcb_raw=float(test['p_raw'][j]),
                         tie_share=float((test['count_ge'][j] - test['count_gt'][j]) / test['draws']),
                         wcb_draws=test['draws'], wcb_seed=WCB_SEED, informative_tasks=informative,
                         tasks=values.shape[0], fewer_than_three_informative=informative < 3))
    return pd.DataFrame(rows)


def adjust_families(frame: pd.DataFrame) -> pd.DataFrame:
    """Holm and Benjamini-Hochberg within each column and Table 1 family."""
    frame = frame.copy()
    for name in ('p_holm_wcb', 'p_bh_wcb', 'p_holm_cluster_t', 'p_bh_cluster_t'):
        frame[name] = np.nan
    in_table = frame.family.notna()
    for (_, _), group in frame[in_table].groupby(['column', 'family']):
        for source, label in (('p_wcb', 'wcb'), ('p_cluster_t', 'cluster_t')):
            p = group[source].to_numpy(dtype=float)
            frame.loc[group.index, f'p_holm_{label}'] = holm(p)
            frame.loc[group.index, f'p_bh_{label}'] = benjamini_hochberg(p)
    frame['family_size'] = frame.groupby(['column', 'family']).contrast.transform('size').where(in_table)
    return frame


def wcb_check(original: Path = ORIGINAL) -> pd.DataFrame:
    """Compare the sign-flip p-values with the stored Stata boottest p-values.

    The comparison uses the original two-model analysis, whose inputs produced the stored values:
    Sonnet, Luna, and their pooled column. With discrete success rates many draws give |t*| equal
    to |t|. Boottest decides each such tie by floating-point noise of order 1e-13, so a stored value
    can fall anywhere between the strict count (|t*| > |t|) and the count with ties (|t*| >= |t|).
    ``beyond_mc_error`` flags a stored value outside that range by more than four Monte Carlo
    standard deviations of a difference of two independent estimates.
    """
    rows = []
    old_runs = pd.read_csv(original / 'data/runs.csv', low_memory=False,
                           usecols=['task', 'model', 'platform', 'prompt', 'agency', 'y'])
    old = pd.read_csv(original / 'tables/regression_contrasts.csv').set_index(['model', 'contrast'])
    for column, members in (('sonnet', ['sonnet']), ('luna', ['luna']), ('pooled', ['sonnet', 'luna'])):
        result = column_inference(old_runs, column, members)
        for rec in result.itertuples():
            ref = old.loc[(column, rec.contrast)]
            rows.append(dict(source='original two-model analysis', column=column, contrast=rec.contrast,
                             estimate_diff=rec.estimate_pp - float(ref.estimate_pp),
                             se_diff=rec.se_cr1_pp - float(ref.se_cr1_pp), p_stored=float(ref.p_wcb),
                             p_sign_flip=rec.p_wcb, p_sign_flip_strict=rec.p_wcb_strict, tie_share=rec.tie_share,
                             informative_tasks=rec.informative_tasks))
    check = pd.DataFrame(rows)
    p = check[['p_stored', 'p_sign_flip']].mean(axis=1)
    # Two independent Monte Carlo estimates of the same p: sd of the difference is sqrt(2 p (1-p) / B).
    check['mc_sd_diff'] = np.sqrt(2 * p * (1 - p) / WCB_DRAWS)
    check['abs_diff'] = (check.p_sign_flip - check.p_stored).abs()
    check['abs_diff_strict'] = (check.p_sign_flip_strict - check.p_stored).abs()
    check['z'] = check.abs_diff / np.maximum(check.mc_sd_diff, 1 / WCB_DRAWS)
    margin = 4 * check.mc_sd_diff + 2 / (WCB_DRAWS + 1)
    check['beyond_mc_error_ties_counted'] = check.abs_diff > margin
    check['beyond_mc_error'] = ((check.p_stored < check.p_sign_flip_strict - margin)
                                | (check.p_stored > check.p_sign_flip + margin))
    return check


def table1_tex(frame: pd.DataFrame) -> str:
    """SI table body: cluster-t p, wild bootstrap p, and Holm p for every Table 1 cell."""
    table = frame.set_index(['column', 'contrast'])
    heads = [('Claude', 'Sonnet 4.6'), ('GPT-5.6', 'Luna'), ('Claude', 'Haiku 5.5'), ('GPT-6', 'Luna'),
             ('Claude', 'Opus 5.5'), ('GPT-6', 'Sol'), ('Total', '(pooled)')]

    def cell(rec, name):
        if rec.informative_tasks == 0 or not np.isfinite(getattr(rec, name)):
            return r'\textendash'
        text = paper_table1.fmt_p(getattr(rec, name))
        return text + (r'\rlap{$^{\dagger}$}' if rec.informative_tasks < 3 and name == 'p_cluster_t' else '')

    lines = [*source_comment('Table 1 inference'), r'\setlength{\tabcolsep}{2pt}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}p{.30\textwidth}*{7}{r}@{}}',
             r'\toprule']
    for line in range(2):
        lines.append(' & ' + ' & '.join(h[line] for h in heads) + r' \\')
    lines += [' & ' + ' & '.join(f'({i})' for i in range(1, 8)) + r' \\', r'\midrule']
    panel = None
    for row in table1_rows():
        if row['panel'] != panel:
            if panel is not None:
                lines.append(r'\addlinespace[4pt]')
            lines.append(r'\multicolumn{8}{@{}l}{\textit{' + row['panel'] + r'}} \\')
            panel = row['panel']
        recs = [table.loc[(c, row['contrast'])] for c in COLUMNS]
        lines.append(r'\quad ' + row['label'] + ' & ' * 7 + r' \\')
        for name, label in (('p_cluster_t', 'Cluster $t$'), ('p_wcb', 'Wild bootstrap'),
                            ('p_holm_wcb', 'Holm, wild bootstrap')):
            values = [cell(r, name) for r in recs]
            lines.append(r'\qquad ' + label + ' & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}']
    return '\n'.join(lines) + '\n'


def analysis_table1(data: dict, out: Path) -> dict:
    runs = data['runs']
    frames = [column_inference(runs, column, members) for column, members in GROUPS.items()]
    frame = pd.concat(frames, ignore_index=True)
    rows = {r['contrast']: r for r in table1_rows()}
    frame['in_table1'] = frame.contrast.isin(rows)
    frame['family'] = frame.contrast.map(lambda c: rows[c]['family'] if c in rows else None)
    frame['label'] = frame.contrast.map(lambda c: rows[c]['label'] if c in rows else statistics.LABELS[c])
    frame['panel'] = frame.contrast.map(lambda c: rows[c]['panel'] if c in rows else None)
    frame = adjust_families(frame)
    stored = pd.read_csv(data['data'] / 'tables/regression_contrasts.csv')
    merged = frame.merge(stored[['model', 'contrast', 'estimate_pp', 'se_cr1_pp', 'informative_tasks']]
                         .rename(columns={'model': 'column'}), on=['column', 'contrast'], suffixes=('', '_stored'))
    stale = float(max((merged.estimate_pp - merged.estimate_pp_stored).abs().max(),
                      (merged.se_cr1_pp - merged.se_cr1_pp_stored).abs().max()))
    informative_diff = int((merged.informative_tasks != merged.informative_tasks_stored).sum())
    order = {r['contrast']: i for i, r in enumerate(table1_rows())}
    frame['row_order'] = frame.contrast.map(order)
    frame = frame.sort_values(['column', 'row_order', 'contrast'], key=lambda s: s.map(
        {c: i for i, c in enumerate(GROUPS)}) if s.name == 'column' else s).drop(columns='row_order')
    write_csv(frame, out / 'table1_inference.csv')
    (out / 'table1_inference.tex').write_text(table1_tex(frame[frame.in_table1]))
    check = wcb_check(ORIGINAL)
    write_csv(check, out / 'wcb_check.csv')
    table = frame[frame.in_table1 & frame.column.isin(COLUMNS)]
    defined = table[table.informative_tasks > 0]
    summary = dict(
        cells=len(table), cells_without_informative_task=int((table.informative_tasks == 0).sum()),
        cells_fewer_than_three_informative=int((table.informative_tasks < 3).sum()),
        cells_fewer_than_three_list=table[table.informative_tasks < 3][['column', 'contrast', 'informative_tasks']]
        .to_dict('records'),
        raw_wcb_below_05=int((defined.p_wcb < .05).sum()), holm_wcb_below_05=int((defined.p_holm_wcb < .05).sum()),
        bh_wcb_below_05=int((defined.p_bh_wcb < .05).sum()), raw_t_below_05=int((defined.p_cluster_t < .05).sum()),
        holm_t_below_05=int((defined.p_holm_cluster_t < .05).sum()),
        pooled=table[table.column.eq('pooled')].set_index('contrast')[
            ['estimate_pp', 'p_cluster_t', 'p_wcb', 'p_holm_wcb', 'p_bh_wcb']].to_dict('index'),
        wcb_check_cells=len(check), wcb_check_max_abs_diff_ties_counted=float(check.abs_diff.max()),
        wcb_check_max_abs_diff_strict=float(check.abs_diff_strict.max()),
        wcb_check_beyond_mc_ties_counted=int(check.beyond_mc_error_ties_counted.sum()),
        wcb_check_beyond_mc_tie_range=int(check.beyond_mc_error.sum()),
        wcb_check_max_tie_share=float(check.tie_share.max()),
        wcb_check_max_abs_diff_no_ties=float(check[check.tie_share.lt(.001)].abs_diff.max()),
        wcb_check_estimate_max_abs_diff=float(check.estimate_diff.abs().max()),
        current_vs_stored_contrasts_max_abs_diff=stale, current_vs_stored_informative_differences=informative_diff)
    return summary


# ---------------------------------------------------------------------------
# 2. SE-only silent failures [M6]
# ---------------------------------------------------------------------------
def outside_estimates(terms: pd.DataFrame) -> pd.Series:
    """Term rows whose estimate is outside its tolerance, fails the integer check, or is missing.

    The wrong-method check (scoring_alternatives.py) also sets coef_pass to False for values
    inside the tolerance. Those rows are not outside the tolerance.
    """
    outside = terms.coef_err > terms.coef_tol * (1 + 1e-12)
    no_alternative = terms.alternative_method.isna() if 'alternative_method' in terms else True
    return (outside | falsy(terms.exact_estimate_pass) | (falsy(terms.coef_pass) & no_alternative)
            | (terms.got_estimate.isna() & terms.ref_estimate.notna()))


def outside_ses(ses: pd.DataFrame) -> pd.Series:
    """SE rows that fail for a reason other than the wrong-method check (outside tolerance or missing)."""
    return falsy(ses.se_pass) & ses.se_reason.ne('closer_to_alternative')


def changed_by_alternative(data_dir: Path) -> set[str]:
    """Runs that the wrong-method check turned from success into failure (alternative_checks.csv)."""
    path = Path(data_dir) / 'data/alternative_checks.csv'
    if not path.is_file():
        return set()
    frame = pd.read_csv(path, low_memory=False)
    return set(frame.position_id[truthy(frame.run_changed)])


def silent_kinds(runs: pd.DataFrame, terms: pd.DataFrame, ses: pd.DataFrame,
                 changed: set[str] | None = None) -> pd.DataFrame:
    """One row per silent failure with its kind and largest relative SE deviation.

    estimate: at least one scored estimate outside tolerance (or a failed integer check).
    se_only: every estimate within tolerance and at least one SE outside tolerance or missing.
    wrong_method: every value within tolerance, and a value closer to a known wrong method
        (the runs that alternative_checks.csv marks run_changed).
    other: none of these (for example a conflict between output channels).
    """
    changed = changed or set()
    silent = runs[runs.failure_category.eq('Silent failure')][['position_id', *CELL]].copy()
    estimate_runs = set(terms.position_id[outside_estimates(terms)])
    se_fail = set(ses.position_id[outside_ses(ses)])
    silent['estimate_failure'] = silent.position_id.isin(estimate_runs)
    silent['se_failure'] = silent.position_id.isin(se_fail)
    silent['wrong_method'] = silent.position_id.isin(changed)
    if (silent.wrong_method & (silent.estimate_failure | silent.se_failure)).any():
        raise ValueError('A run changed by the wrong-method check has a value outside its tolerance')
    silent['kind'] = np.select([silent.estimate_failure, silent.se_failure, silent.wrong_method],
                               ['estimate', 'se_only', 'wrong_method'], 'other')
    alternative = pd.concat([terms.loc[terms.get('alternative_method', pd.Series(index=terms.index)).notna()
                                       & falsy(terms.coef_pass), ['position_id']].assign(on='estimate'),
                             ses.loc[ses.se_reason.eq('closer_to_alternative'), ['position_id']].assign(on='se')])
    on = alternative.groupby('position_id').on.apply(lambda s: '+'.join(sorted(set(s))))
    silent['wrong_method_on'] = silent.position_id.map(on).where(silent.wrong_method)
    rel = ses.assign(rel=(ses.se_err / ses.se_ref.abs()).where(ses.se_got.notna(), np.inf))
    worst = rel.groupby('position_id').rel.max()
    silent['largest_se_deviation'] = silent.position_id.map(worst)
    silent['se_missing'] = silent.position_id.isin(set(ses.position_id[ses.se_got.isna()]))
    return silent


def deviation_stats(values: pd.Series) -> dict:
    finite = values[np.isfinite(values)]
    return dict(se_only_runs=len(values), missing_se_runs=int((~np.isfinite(values)).sum()),
                median_largest_deviation_pct=100 * float(values.median()) if len(values) else np.nan,
                share_below_2pct=100 * float((values < .02).mean()) if len(values) else np.nan,
                share_below_5pct=100 * float((values < .05).mean()) if len(values) else np.nan,
                p25_pct=100 * float(values.quantile(.25)) if len(values) else np.nan,
                p75_pct=100 * float(values.quantile(.75)) if len(values) else np.nan,
                max_finite_pct=100 * float(finite.max()) if len(finite) else np.nan)


KINDS = (('estimate', 'Estimate outside tolerance'), ('se_only', 'SE outside tolerance only'),
         ('wrong_method', 'Closer to a wrong method'), ('other', 'Other check'))


def se_only_tex(counts: pd.DataFrame, deviations: pd.DataFrame) -> str:
    """SI table body: silent failures by kind, model, and agency, and the SE deviation of SE-only runs."""
    agencies = ['chatbot', 'self_repair', 'agent', 'all']
    table = counts.set_index(['model', 'agency'])
    kinds = [(k, label) for k, label in KINDS if k != 'other' or table.other.sum() > 0]
    lines = [*source_comment('Silent failures by kind'), r'\setlength{\tabcolsep}{2pt}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}p{.40\textwidth}*{4}{r}@{}}',
             r'\toprule', 'Kind of silent failure & ' + ' & '.join(AGENCY_NAMES[a] for a in agencies) + r' \\',
             r'\midrule']
    for index, model in enumerate(['all', *MODELS]):
        if index:
            lines.append(r'\addlinespace[3pt]')
        lines.append(r'\multicolumn{5}{@{}l}{\textit{' + ('All models' if model == 'all' else MODEL_NAMES[model]) + r'}} \\')
        for kind, label in kinds:
            lines.append(r'\quad ' + label + ' & ' + ' & '.join(f'{int(table.loc[(model, a), kind]):,}' for a in agencies) + r' \\')
        lines.append(r'\quad All silent failures & ' + ' & '.join(f'{int(table.loc[(model, a)].silent):,}' for a in agencies) + r' \\')
        if model == 'all':
            for column, label in (('se_only_share_pct', 'SE only, share of silent failures (percent)'),
                                  ('wrong_method_share_pct', 'Wrong method, share of silent failures (percent)')):
                lines.append(r'\quad ' + label + ' & ' + ' & '.join(fixed(table.loc[('all', a), column]) for a in agencies) + r' \\')
    lines += [r'\midrule', r'\multicolumn{5}{@{}l}{\textit{Largest relative SE deviation of the SE-only runs}} \\']
    dev = deviations[deviations.group.eq('agency')].set_index('key')
    for label, column in (('SE-only runs', 'se_only_runs'), ('Median (percent)', 'median_largest_deviation_pct'),
                          ('Below 2 percent (share, percent)', 'share_below_2pct'),
                          ('Below 5 percent (share, percent)', 'share_below_5pct')):
        values = []
        for a in agencies:
            if column == 'se_only_runs':
                values.append(f'{int(dev.loc[a, column]):,}')
            else:
                values.append(fixed(dev.loc[a, column]) if dev.loc[a, 'se_only_runs'] else r'\textendash')
        lines.append(r'\quad ' + label + ' & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}']
    return '\n'.join(lines) + '\n'


def analysis_se_only(data: dict, out: Path) -> dict:
    changed = changed_by_alternative(data['data'])
    silent = silent_kinds(data['runs'], data['terms'], data['ses'], changed)
    write_csv(silent, out / 'se_only_silent_runs.csv')
    rows = []
    for model in [*MODELS, 'all']:
        for agency in [*six.AGENCIES, 'all']:
            q = subset(silent, model=model, agency=agency)
            rec = dict(model=model, agency=agency, silent=len(q))
            for kind, _ in KINDS:
                rec[kind] = int(q.kind.eq(kind).sum())
            rec['wrong_method_on_estimate'] = int((q.kind.eq('wrong_method') & q.wrong_method_on.str.contains('estimate', na=False)).sum())
            rec['wrong_method_on_se'] = int((q.kind.eq('wrong_method') & q.wrong_method_on.str.contains('se', na=False)).sum())
            rec['se_only_share_pct'] = 100 * q.kind.eq('se_only').mean() if len(q) else np.nan
            rec['wrong_method_share_pct'] = 100 * q.kind.eq('wrong_method').mean() if len(q) else np.nan
            rec['estimate_share_pct'] = 100 * q.kind.eq('estimate').mean() if len(q) else np.nan
            rows.append(rec)
    counts = pd.DataFrame(rows)
    write_csv(counts, out / 'se_only_silent.csv')
    se_only = silent[silent.kind.eq('se_only')]
    dev_rows = []
    for group, keys in (('agency', [*six.AGENCIES, 'all']), ('model', [*MODELS, 'all'])):
        for key in keys:
            q = se_only if key == 'all' else se_only[se_only[group].eq(key)]
            dev_rows.append(dict(group=group, key=key, **deviation_stats(q.largest_se_deviation)))
    for (model, agency), q in se_only.groupby(['model', 'agency']):
        dev_rows.append(dict(group='model_agency', key=f'{model}|{agency}', **deviation_stats(q.largest_se_deviation)))
    deviations = pd.DataFrame(dev_rows)
    write_csv(deviations, out / 'se_only_silent_deviation.csv')
    (out / 'se_only_silent.tex').write_text(se_only_tex(counts, deviations))
    total = counts.set_index(['model', 'agency'])
    columns = ['silent', 'estimate', 'se_only', 'wrong_method', 'other', 'se_only_share_pct', 'wrong_method_share_pct']
    by_task = {kind: {k: int(v) for k, v in silent[silent.kind.eq(kind)].groupby('task').size()
                      .sort_values(ascending=False).items()} for kind in ('se_only', 'wrong_method')}
    return dict(changed_runs_listed=len(changed), all=total.loc[('all', 'all')][columns].to_dict(),
                by_agency={a: total.loc[('all', a)][columns].to_dict() for a in six.AGENCIES},
                agent_by_model={m: total.loc[(m, 'agent')][columns].to_dict() for m in MODELS},
                by_task=by_task,
                deviation_all=deviations[(deviations.group == 'agency') & (deviations.key == 'all')].iloc[0].to_dict(),
                deviation_agent=deviations[(deviations.group == 'agency') & (deviations.key == 'agent')].iloc[0].to_dict())


# ---------------------------------------------------------------------------
# Answers of a run: the scored values, for analyses 3 and 4
# ---------------------------------------------------------------------------
def term_tolerances(terms: pd.DataFrame, catalog: dict) -> pd.DataFrame:
    """The estimate tolerance of each task term: coef_tol of term_checks.csv, else the scorer rule."""
    known = terms.dropna(subset=['coef_tol']).groupby(['task', 'term']).coef_tol.first()
    rows = []
    for task, spec in catalog.items():
        for ref in spec['reference']['estimates']:
            if ref.get('estimate') is None:
                continue
            tol = known.get((task, ref['term']))
            if tol is None or not np.isfinite(tol):
                tol = max(0.005, 0.01 * abs(float(ref['estimate'])))
            rows.append(dict(task=task, term=ref['term'], ref_estimate=float(ref['estimate']), tol=float(tol)))
    return pd.DataFrame(rows)


def answer_frames(data: dict) -> dict:
    """Values of every answering run (success or silent failure), wide by term within task."""
    runs, terms, ses = data['runs'], data['terms'], data['ses']
    answered = runs[runs.failure_category.ne('Loud failure')]
    tolerances = term_tolerances(terms, data['catalog'])
    est = terms[terms.position_id.isin(set(answered.position_id))]
    est = est.merge(tolerances[['task', 'term', 'tol']], on=['task', 'term'], how='inner')
    se = ses[ses.position_id.isin(set(answered.position_id))].copy()
    se['se_tolerance'] = se.se_tol.fillna(0) + se.se_allowance.fillna(0)
    return dict(answered=answered, estimates=est, ses=se, tolerances=tolerances)


def agreement_matrix(est_values: np.ndarray, tol: np.ndarray, se_values: np.ndarray | None = None,
                     se_tol: np.ndarray | None = None) -> np.ndarray:
    """Pairwise agreement of answers: every scored value within its tolerance (n x n, bool).

    est_values is n x T. se_values is n x U with per-run tolerances se_tol (n x U). Two missing
    SEs agree, a missing and a present SE disagree.
    """
    diff = np.abs(est_values[:, None, :] - est_values[None, :, :])
    agree = np.all(diff <= tol[None, None, :] * (1 + 1e-12), axis=2)
    if se_values is not None and se_values.shape[1]:
        missing = np.isnan(se_values)
        both = missing[:, None, :] & missing[None, :, :]
        one = missing[:, None, :] ^ missing[None, :, :]
        bound = np.maximum(se_tol[:, None, :], se_tol[None, :, :])
        close = np.abs(se_values[:, None, :] - se_values[None, :, :]) <= bound * (1 + 1e-12)
        agree &= np.all(both | (~one & close), axis=2)
    return agree


def subsets(n: int, k: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(n), k)), dtype=int)


def vote(agree: np.ndarray, answered: np.ndarray, success: np.ndarray, subset_index: np.ndarray) -> tuple[float, float]:
    """Expected success of the most frequent answer over all subsets in subset_index.

    Returns (vote among answers, vote with "no answer" as a candidate). Ties are broken at random
    among the runs whose answer has the top support, so the value is the expected success.
    """
    a = (agree & answered[:, None] & answered[None, :]).astype(int)
    sub = a[subset_index[:, :, None], subset_index[:, None, :]]
    support = sub.sum(axis=2)
    ans = answered[subset_index]
    suc = success[subset_index].astype(float)
    best = support.max(axis=1)
    top = (support == best[:, None]) & ans & (best[:, None] > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        among = np.where(best > 0, (top * suc).sum(axis=1) / top.sum(axis=1), 0.0)
    none = (~ans).sum(axis=1)
    support_b = np.where(ans, support, none[:, None])
    best_b = support_b.max(axis=1)
    top_b = support_b == best_b[:, None]
    with_none = (top_b * suc * ans).sum(axis=1) / top_b.sum(axis=1)
    return float(among.mean()), float(with_none.mean())


def cell_agreements(cell_runs: pd.DataFrame, est: pd.DataFrame, se: pd.DataFrame,
                    tolerances: pd.Series) -> np.ndarray:
    """Agreement matrix of the runs of one cell (ordered as cell_runs)."""
    ids = cell_runs.position_id.tolist()
    terms = sorted(tolerances.index)
    e = est.pivot_table(index='position_id', columns='term', values='got_estimate', aggfunc='first')
    e = e.reindex(index=ids, columns=terms)
    tol = tolerances.reindex(terms).to_numpy(dtype=float)
    values = e.to_numpy(dtype=float)
    if len(se):
        s = se.pivot_table(index='position_id', columns='term', values='se_got', aggfunc='first', dropna=False)
        st = se.pivot_table(index='position_id', columns='term', values='se_tolerance', aggfunc='first', dropna=False)
        se_terms = sorted(se.term.unique())
        s, st = s.reindex(index=ids, columns=se_terms), st.reindex(index=ids, columns=se_terms)
        return agreement_matrix(values, tol, s.to_numpy(dtype=float), st.fillna(0).to_numpy(dtype=float))
    return agreement_matrix(values, tol)


def analysis_majority(data: dict, out: Path, ks: tuple[int, ...] = (1, 3, 5, 9, 10)) -> dict:
    runs = data['runs']
    frames = answer_frames(data)
    est, se = frames['estimates'], frames['ses']
    tol_by_task = {t: g.set_index('term').tol for t, g in frames['tolerances'].groupby('task')}
    est_by_cell = {k: g for k, g in est.merge(runs[['position_id', *CELL]], on=['position_id', 'task']).groupby(CELL)}
    se_by_cell = {k: g for k, g in se.merge(runs[['position_id', *CELL]], on=['position_id', 'task']).groupby(CELL)}
    index = {k: subsets(10, k) for k in ks}
    chat = runs[runs.agency.eq('chatbot')].sort_values([*CELL, 'rep'])
    agent_rate = runs[runs.agency.eq('agent')].groupby(['model', 'task', 'platform', 'prompt']).primary.mean()
    rows = []
    for key, cell in chat.groupby(CELL, sort=True):
        if len(cell) != 10:
            raise ValueError(f'Cell without ten chatbot runs: {key}')
        answered = cell.failure_category.ne('Loud failure').to_numpy()
        success = cell.primary.to_numpy()
        e = est_by_cell.get(key, est.iloc[:0])
        s = se_by_cell.get(key, se.iloc[:0])
        agree = cell_agreements(cell, e, s, tol_by_task[key[1]])
        rec = dict(zip(CELL, key), one_chatbot_run=float(success.mean()), answered_runs=int(answered.sum()),
                   one_agent_run=float(agent_rate.loc[key[:4]]), distinct_answers=distinct_count(agree, answered))
        for k in ks:
            among, with_none = vote(agree, answered, success, index[k])
            rec[f'vote_k{k}'] = among
            rec[f'vote_none_k{k}'] = with_none
        rows.append(rec)
    cells = pd.DataFrame(rows)
    write_csv(cells, out / 'majority_vote_cells.csv')
    if not np.allclose(cells.vote_k1, cells.one_chatbot_run):
        raise ValueError('Majority vote with k = 1 must equal the chatbot success rate')
    tasks = sorted(runs.task.unique())
    draws = boot_index(tasks)
    measures = ['one_chatbot_run', *[f'vote_k{k}' for k in ks], *[f'vote_none_k{k}' for k in ks], 'one_agent_run']
    summary = []
    for model in [*MODELS, 'all']:
        q = cells if model == 'all' else cells[cells.model.eq(model)]
        n = task_totals(q, tasks)
        for measure in measures:
            num = task_totals(q, tasks, measure)
            boot = 100 * num[draws].sum(axis=1) / n[draws].sum(axis=1)
            low, high = interval(boot)
            summary.append(dict(model=model, measure=measure, value_pct=100 * num.sum() / n.sum(),
                                lo_pct=low, hi_pct=high, cells=len(q)))
        for k in ks:
            diff = task_totals(q, tasks, 'one_agent_run') - task_totals(q, tasks, f'vote_k{k}')
            boot = 100 * diff[draws].sum(axis=1) / n[draws].sum(axis=1)
            low, high = interval(boot)
            summary.append(dict(model=model, measure=f'agent_minus_vote_k{k}', value_pct=100 * diff.sum() / n.sum(),
                                lo_pct=low, hi_pct=high, cells=len(q)))
    summary = pd.DataFrame(summary)
    write_csv(summary, out / 'majority_vote.csv')
    pooled = summary[summary.model.eq('all')].set_index('measure').value_pct
    return dict(cells=len(cells), **{m: float(pooled[m]) for m in measures},
                agent_minus_vote_k9=float(pooled['agent_minus_vote_k9']),
                by_model_k5={m: float(summary[(summary.model == m) & (summary.measure == 'vote_k5')].value_pct.iloc[0])
                             for m in MODELS})


def distinct_count(agree: np.ndarray, answered: np.ndarray) -> int:
    """Number of answer groups by greedy grouping on the agreement matrix."""
    left = set(np.flatnonzero(answered))
    count = 0
    while left:
        order = sorted(left)
        best = max(order, key=lambda i: sum(agree[i, j] for j in left))
        left -= {j for j in left if agree[best, j]} | {best}
        count += 1
    return count


# ---------------------------------------------------------------------------
# 4. Shared wrong values [A2]
# ---------------------------------------------------------------------------
def cluster_values(values: np.ndarray, tol: float) -> np.ndarray:
    """Group values within tolerance: take the value with the most neighbors within tol, give its
    unassigned neighbors one label, and repeat. Labels start at 0 in order of group size."""
    values = np.asarray(values, dtype=float)
    labels = np.full(len(values), -1)
    left = np.arange(len(values))
    label = 0
    while len(left):
        x = values[left]
        order = np.argsort(x, kind='stable')
        xs = x[order]
        bound = tol * (1 + 1e-12)
        counts = np.searchsorted(xs, xs + bound, side='right') - np.searchsorted(xs, xs - bound, side='left')
        centre = xs[int(np.argmax(counts))]
        members = left[np.abs(x - centre) <= bound]
        labels[members] = label
        label += 1
        left = left[labels[left] < 0]
    return labels


def shared_wrong(runs: pd.DataFrame, terms: pd.DataFrame, tolerances: pd.DataFrame,
                 include_wrong_method: bool = False) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group the failed scored estimates into distinct wrong values within each task term.

    A failed term is an estimate outside its tolerance (or failing the integer check). With
    include_wrong_method, the estimates inside the tolerance that the wrong-method check fails
    also count. Two values are the same if they fall in one group of cluster_values.
    """
    mask = outside_estimates(terms)
    if include_wrong_method:
        mask |= falsy(terms.coef_pass)
    failed = terms[mask & terms.got_estimate.notna()].merge(
        runs[['position_id', *[c for c in CELL if c != 'task'], 'failure_category', 'rep']], on='position_id').copy()
    failed['cluster'] = -1
    clusters = []
    for (task, term), group in failed.groupby(['task', 'term'], sort=True):
        tol = float(tolerances.loc[(task, term), 'tol'])
        ref = float(tolerances.loc[(task, term), 'ref_estimate'])
        labels = cluster_values(group.got_estimate.to_numpy(), tol)
        failed.loc[group.index, 'cluster'] = labels
        for label in np.unique(labels):
            g = group[labels == label]
            centre = float(g.got_estimate.median())
            clusters.append(dict(task=task, term=term, cluster=int(label), value=centre, ref_estimate=ref,
                                 tolerance=tol, relative_deviation=(centre - ref) / abs(ref) if ref else np.nan,
                                 failed_terms=len(g), runs=g.position_id.nunique(), models=g.model.nunique(),
                                 model_list=','.join(sorted(g.model.unique())), languages=g.platform.nunique(),
                                 language_list=','.join(sorted(g.platform.unique())),
                                 agencies=','.join(sorted(g.agency.unique())), prompts=g.prompt.nunique(),
                                 loud_runs=int(g[g.failure_category.eq('Loud failure')].position_id.nunique())))
    clusters = pd.DataFrame(clusters)
    failed['value_id'] = failed.task + '|' + failed.term + '|' + failed.cluster.astype(str)
    clusters['value_id'] = clusters.task + '|' + clusters.term + '|' + clusters.cluster.astype(str)
    shared = set(clusters.value_id[clusters.models >= 3])
    repeated = set(clusters.value_id[clusters.runs >= 2])
    signature = failed.groupby('position_id').value_id.apply(lambda v: '||'.join(sorted(v)))
    wrong_runs = runs[runs.position_id.isin(signature.index)].assign(signature=lambda d: d.position_id.map(signature))
    cell_rows = [dict(zip(CELL, key), wrong_value_runs=len(cell), same_wrong_answer=cell.signature.nunique() == 1)
                 for key, cell in wrong_runs.groupby(CELL) if len(cell) >= 2]
    cells = pd.DataFrame(cell_rows, columns=[*CELL, 'wrong_value_runs', 'same_wrong_answer'])
    # Second view: cells with two or more failed runs of any kind. A failed run without a wrong
    # estimate (a loud failure without values, an SE failure) breaks the identity.
    failed_runs = runs[~runs.primary].assign(signature=lambda d: d.position_id.map(signature))
    any_rows = [dict(zip(CELL, key), failed_runs=len(cell),
                     same_wrong_answer=bool(cell.signature.notna().all() and cell.signature.nunique() == 1))
                for key, cell in failed_runs.groupby(CELL) if len(cell) >= 2]
    any_cells = pd.DataFrame(any_rows, columns=[*CELL, 'failed_runs', 'same_wrong_answer'])
    # Third view: pairs of a cell and a term with two or more failed values.
    pairs = failed.groupby([*CELL, 'term']).value_id.agg(['size', 'nunique'])
    pairs = pairs[pairs['size'] >= 2]
    by_model = pd.DataFrame([dict(model=model, failed_terms=len(f), distinct_values=f.value_id.nunique(),
                                  shared_three_models_share_pct=100 * f.value_id.isin(shared).mean() if len(f) else np.nan)
                             for model, f in [*((m, failed[failed.model.eq(m)]) for m in MODELS), ('all', failed)]])
    summary = dict(failed_terms=len(failed), failed_term_runs=failed.position_id.nunique(),
                   distinct_wrong_values=len(clusters), values_in_three_or_more_models=len(shared),
                   failed_terms_shared_by_three_or_more_models=int(failed.value_id.isin(shared).sum()),
                   failed_terms_shared_by_three_or_more_models_pct=100 * float(failed.value_id.isin(shared).mean()),
                   failed_terms_in_values_of_two_or_more_runs_pct=100 * float(failed.value_id.isin(repeated).mean()),
                   cells_two_or_more_wrong_value_runs=len(cells),
                   cells_all_same_wrong_answer=int(cells.same_wrong_answer.sum()),
                   cells_all_same_wrong_answer_pct=100 * float(cells.same_wrong_answer.mean()) if len(cells) else np.nan,
                   cell_terms_two_or_more_failed_values=len(pairs),
                   cell_terms_one_wrong_value=int(pairs['nunique'].eq(1).sum()),
                   cell_terms_one_wrong_value_pct=100 * float(pairs['nunique'].eq(1).mean()) if len(pairs) else np.nan,
                   cells_two_or_more_failed_runs=len(any_cells),
                   cells_failed_all_same_wrong_answer=int(any_cells.same_wrong_answer.sum()),
                   cells_failed_all_same_wrong_answer_pct=100 * float(any_cells.same_wrong_answer.mean())
                   if len(any_cells) else np.nan)
    return summary, clusters, cells, by_model


def analysis_shared_wrong(data: dict, out: Path) -> dict:
    runs, terms = data['runs'], data['terms']
    tolerances = term_tolerances(terms, data['catalog']).set_index(['task', 'term'])
    summary, clusters, cells, by_model = shared_wrong(runs, terms, tolerances)
    wide, wide_clusters, _, wide_by_model = shared_wrong(runs, terms, tolerances, include_wrong_method=True)
    ranked = clusters.sort_values(['models', 'runs', 'failed_terms'], ascending=False)
    top = ranked.head(10)
    write_csv(ranked, out / 'wrong_value_clusters.csv')
    write_csv(top, out / 'shared_wrong_values_top10.csv')
    write_csv(cells, out / 'shared_wrong_value_cells.csv')
    long = pd.DataFrame([dict(view=view, quantity=k, value=v) for view, values in
                         (('outside tolerance', summary), ('with wrong-method values', wide)) for k, v in values.items()])
    write_csv(long, out / 'shared_wrong_values.csv')
    write_csv(pd.concat([by_model.assign(view='outside tolerance'), wide_by_model.assign(view='with wrong-method values')]),
              out / 'shared_wrong_values_by_model.csv')
    summary['with_wrong_method_values'] = wide
    summary['top10'] = top[['task', 'term', 'value', 'ref_estimate', 'relative_deviation', 'models', 'runs',
                            'failed_terms', 'language_list', 'agencies']].to_dict('records')
    summary['top10_with_wrong_method_values'] = wide_clusters.sort_values(
        ['models', 'runs', 'failed_terms'], ascending=False).head(10)[
        ['task', 'term', 'value', 'ref_estimate', 'models', 'runs']].to_dict('records')
    return summary


# ---------------------------------------------------------------------------
# 5. Agent self-checks [I14, A5]
# ---------------------------------------------------------------------------
# Keyword rule for the agent's own check (verification.txt, else the final message). It was
# written on a development sample of flagged texts and then checked by hand on a separate
# sample (self_check_hand_labels.csv).
UNRESOLVED = re.compile(
    r"\bunresolved\b|\b(?:did|could|can) ?not resolve|\bnot (?:been )?resolved\b|\bremains? (?:unresolved|incomplete)"
    r"|\b(?:are|is|remain|remains) not (?:been )?verified\b|\bcannot be (?:treated as )?verified|\brequires? (?:a )?(?:correction|further review|fix)"
    r"|\bneeds? (?:a )?(?:correction|fix|to be corrected)|\bhave not resolved|\bcannot verify|\bstill needs?\b"
    r"|\bverification (?:requirement )?(?:remains |is )?incomplete|\b(?:is|are|produces an?) incorrect\b"
    , re.I)
NOT_UNRESOLVED = re.compile(
    r"\bno (?:unresolved|remaining)\b|\bnothing (?:remains|is) unresolved|\bnone (?:remain|are) unresolved"
    r"|^\W*(?:no|none)\b|\bno (?:discrepanc\w*|mismatch\w*|difference\w*|correction\w*|change\w*)\b", re.I)
DISCREPANCY = re.compile(
    r"discrepanc|mismatch|\b(?:does|do|would) not (?:match|agree|equal|reproduce|coincide)"
    r"|\bnot (?:the same|identical|consistent|equal)\b|\bdisagree|\binconsistent\b"
    r"|\bdiffer(?:s|ed)? (?:from|by)\b|\bdid not (?:match|agree)\b|\bdisagreed\b", re.I)
NEGATED = re.compile(
    r"\b(?:no|none|without|zero|nor|never|any)\b[^.;\n]{0,40}(?:discrepanc|mismatch|disagree|inconsisten|differ)"
    r"|(?:discrepanc\w*|mismatch\w*)[^.;\n]{0,40}(?:\bnone\b|[:=~]\s*~?0\b|\bnil\b|exact match)"
    r"|\b0 (?:mismatch|discrepanc)|\bnot (?:a |any )?(?:real |true |methodological |substantive )?discrepanc"
    r"|\bdiffer\w*[^;\n]{0,40}\bby 0(?:\.0+)?(?:e[+-]?\d+)?\b(?!\.\d)", re.I)
EXPLAINED = re.compile(
    r"negligible|\bsmall\b|\btiny\b|\bminor\b|rounding|precision|\bfloat\b|convention|\bexpected\b|\bdue to\b"
    r"|\bbecause\b|attributable|consistent with|\binitial(?:ly)?\b|\bfirst (?:run|version|attempt|check)|\bearlier\b"
    r"|\bprevious(?:ly)?\b|<\s*0?\.\d+\s*%|~\s*0?\.0|\d(?:\.\d+)?e-\d|\b0\.0000|\bwithin\b|\bbelow\b|optimi[sz]er"
    r"|\bonly\b|\bhypothetical|\balternative\b|\bwould\b|\bdecimal|less than|slightly|\bclose\b|\bapprox"
    r"|\btraced\b|came from|\baris(?:es|e|ing)\b|\barose\b|\bstems?\b|numerical error|\bnot (?:a |the )?(?:model|script|estimator)"
    r"|\b(?:in|from) (?:the|my) (?:check|manual|finite)|bug in the check|check code|\bdigits\b|significant figures"
    r"|\bnote\b|\bchecked\b|\binspected\b|\bbefore the fix|\bfix\b|\bknown\b|\bscaling\b|\bfactor\b|\bdivisor\b"
    r"|\bcorrection\b|\bkept\b|\binvestigated\b|\bdiscarded\b", re.I)
RESOLVED = re.compile(
    r"\b(?:fixed|corrected|reconciled|repaired|renamed|dropped|removed)\b|(?<!un)\bresolved\b|\brerun\b|\bre-ran\b"
    r"|\breran\b|\bnow (?:match|agree|equal)|made it (?:match|agree)|\bfinal (?:script|run|version|code)\b"
    r"|\bresolv(?:es|ing)\b"
    r"|taken directly from|\bexplained\b|\bupdated\b|\bcorrecting\b|\bfixing\b|\bexcluded\b|\bretained\b"
    r"|\binstead\b|\bnot used\b", re.I)
AGREEMENT = re.compile(
    r"\bmatch(?:es|ed|ing)?\b|\bagree(?:s|d|ment)?\b|\bconsistent\b|\bidentical\b|\bconfirm(?:s|ed)?\b|\bverified\b"
    r"|\bpass(?:es|ed)?\b|\bequal(?:s)?\b|\bthe same\b|\bcoincide|\breproduc(?:e|es|ed)\b|\bheld\b|\bholds\b"
    r"|\bdifference[^.;\n]{0,40}\d(?:\.\d+)?e-\d", re.I)


def sentences(text: str) -> list[str]:
    """Split a check text into sentences. A line break before a lowercase word continues the sentence."""
    text = re.sub(r'\n(?=[ \t]*[a-z(<~])', ' ', text)
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]


def classify_check(text: str) -> dict:
    """Keyword rule for the agent's check text.

    discrepancy: a sentence states that a difference is unresolved, or names a discrepancy that is
    not negated, not explained as small or expected, and not followed by a correction.
    agreement: no such sentence, and some sentence reports a match, agreement, or no discrepancy.
    neither: otherwise. The rule is a signal; the hand check gives its accuracy.
    """
    parts = sentences(text or '')
    open_discrepancy = None
    agreement = False
    for i, part in enumerate(parts):
        if UNRESOLVED.search(part) and not NOT_UNRESOLVED.search(part):
            return dict(label='discrepancy', evidence=part[:300])
        named = bool(DISCREPANCY.search(part)) and not NEGATED.search(part)
        following = parts[i + 1] if i + 1 < len(parts) else ''
        # A heading such as "Discrepancies (Stata vs Python):" introduces numbers and states nothing.
        heading = part.endswith(':') or len(re.findall(r'[A-Za-z]+', part)) <= 3
        if (named and not heading and not EXPLAINED.search(part) and not RESOLVED.search(part)
                and not EXPLAINED.search(following)):
            open_discrepancy = i
        elif open_discrepancy is not None and RESOLVED.search(part):
            open_discrepancy = None
        if NEGATED.search(part) or (AGREEMENT.search(part) and not named):
            agreement = True
    if open_discrepancy is not None:
        return dict(label='discrepancy', evidence=parts[open_discrepancy][:300])
    return dict(label='agreement' if agreement else 'neither', evidence='')


def final_message(folder: Path) -> str:
    attempts = sorted(folder.glob('last_attempt*.txt'), key=lambda p: int(re.sub(r'\D', '', p.stem) or 0))
    if attempts:
        return attempts[-1].read_text(errors='replace')
    record = json.loads((folder / 'run_record.json').read_text())
    calls = record.get('calls') or []
    return (calls[-1].get('agent_text') or '') if calls else ''


def check_texts(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in runs.itertuples(index=False):
        folder = (ROOT / rec.record_path).parent
        path = folder / 'final_workspace/verification.txt'
        if path.is_file():
            text, source = path.read_text(errors='replace'), 'verification.txt'
        else:
            text, source = final_message(folder), 'final message'
        rows.append(dict(position_id=rec.position_id, source=source, text=text))
    return pd.DataFrame(rows)


def hand_sample(labels: pd.DataFrame, runs: pd.DataFrame, size: int = 30, seed: int = 20261030) -> pd.DataFrame:
    """A fixed hand-check sample: 10 rule discrepancies, 10 silent failures with rule agreement,
    and 10 other constrained-agent runs."""
    frame = labels
    rng = np.random.default_rng(seed)
    picks = []
    strata = [frame[frame.label.eq('discrepancy')],
              frame[frame.label.eq('agreement') & frame.failure_category.eq('Silent failure')]]
    for stratum in strata:
        picks += list(rng.choice(stratum.position_id.to_numpy(), size=min(size // 3, len(stratum)), replace=False))
    rest = frame[~frame.position_id.isin(picks)]
    picks += list(rng.choice(rest.position_id.to_numpy(), size=size - len(picks), replace=False))
    return frame.set_index('position_id').loc[picks].reset_index()


def analysis_self_checks(data: dict, out: Path, hand_labels: Path = HAND_LABELS) -> dict:
    runs = data['runs']
    agent = runs[runs.agency.eq('agent')]
    texts = check_texts(agent)
    labels = pd.concat([texts[['position_id', 'source']],
                        pd.DataFrame([classify_check(t) for t in texts.text])], axis=1)
    labels = labels.merge(agent[['position_id', 'model', 'task', 'platform', 'prompt', 'failure_category']],
                          on='position_id')
    write_csv(labels, out / 'self_checks.csv')
    summary_rows = []
    for model in [*MODELS, 'all']:
        q = labels if model == 'all' else labels[labels.model.eq(model)]
        for category in ['Success', 'Silent failure', 'Loud failure', 'all']:
            c = q if category == 'all' else q[q.failure_category.eq(category)]
            summary_rows.append(dict(model=model, outcome=category, runs=len(c),
                                     verification_file=int(c.source.eq('verification.txt').sum()),
                                     agreement=int(c.label.eq('agreement').sum()),
                                     discrepancy=int(c.label.eq('discrepancy').sum()),
                                     neither=int(c.label.eq('neither').sum())))
    summary = pd.DataFrame(summary_rows)
    write_csv(summary, out / 'self_checks_summary.csv')
    result = dict(agent_runs=len(labels), verification_files=int(labels.source.eq('verification.txt').sum()))
    silent = labels[labels.failure_category.eq('Silent failure')]
    result.update(silent=len(silent), silent_agreement=int(silent.label.eq('agreement').sum()),
                  silent_discrepancy=int(silent.label.eq('discrepancy').sum()),
                  silent_neither=int(silent.label.eq('neither').sum()),
                  discrepancy_all=int(labels.label.eq('discrepancy').sum()),
                  discrepancy_by_outcome=labels[labels.label.eq('discrepancy')].failure_category.value_counts().to_dict())
    kinds = silent_kinds(runs, data['terms'], data['ses'], changed_by_alternative(data['data']))
    by_kind = silent.merge(kinds[['position_id', 'kind']], on='position_id')
    result['silent_label_by_kind'] = pd.crosstab(by_kind.kind, by_kind.label).to_dict('index')
    result['silent_label_by_model'] = pd.crosstab(silent.model, silent.label).to_dict('index')
    # Hand check: the labelled sample is fixed by position_id once the label file exists.
    if hand_labels.is_file():
        hand = pd.read_csv(hand_labels)
        checked = hand.merge(labels[['position_id', 'label', 'failure_category']], on='position_id', how='left')
        checked = checked[checked.hand_label.notna()]
        checked['rule_correct'] = checked.label.eq(checked.hand_label)
        write_csv(checked, out / 'self_check_hand_check.csv')
        result.update(hand_checked=len(checked), hand_accuracy_pct=100 * float(checked.rule_correct.mean()),
                      hand_discrepancy_precision_pct=100 * float(
                          checked[checked.label.eq('discrepancy')].hand_label.eq('discrepancy').mean())
                      if checked.label.eq('discrepancy').any() else np.nan,
                      hand_agreement_precision_pct=100 * float(
                          checked[checked.label.eq('agreement')].hand_label.eq('agreement').mean())
                      if checked.label.eq('agreement').any() else np.nan,
                      hand_confusion=pd.crosstab(checked.label, checked.hand_label).to_dict())
    else:
        sample = hand_sample(labels, runs).merge(texts[['position_id', 'text']], on='position_id')
        write_csv(sample, out / 'self_check_hand_sample.csv')
        result['hand_checked'] = 0
    return result


# ---------------------------------------------------------------------------
# 6. Small numbers [I04, I10, I13, I17, I42]
# ---------------------------------------------------------------------------
def boot_rows(item: str, quantity: str, group: str, point: float, boot: np.ndarray, unit: str) -> dict:
    low, high = interval(boot)
    return dict(item=item, quantity=quantity, group=group, value=point, lo=low, hi=high, unit=unit)


def i04_rows(runs: pd.DataFrame, tasks: list[str], draws: np.ndarray) -> list[dict]:
    """Chatbot one-shot effect split into did-not-run, ran-but-incomplete, and silent parts."""
    chat = runs[runs.agency.eq('chatbot')].assign(
        did_not_run=lambda d: d.failure_category.eq('Loud failure') & ~d.execution_ok,
        ran_incomplete=lambda d: d.failure_category.eq('Loud failure') & d.execution_ok,
        silent=lambda d: d.failure_category.eq('Silent failure'), success=lambda d: d.primary)
    rows = []
    for model in [*MODELS, 'all']:
        q = chat if model == 'all' else chat[chat.model.eq(model)]
        zs, fs = q[q.prompt.eq('zero_shot')], q[q.prompt.eq('few_shot')]
        nz, nf = task_totals(zs, tasks), task_totals(fs, tasks)
        for part in ('success', 'did_not_run', 'ran_incomplete', 'silent'):
            a, b = task_totals(zs, tasks, part), task_totals(fs, tasks, part)
            point = 100 * (b.sum() / nf.sum() - a.sum() / nz.sum())
            boot = 100 * (b[draws].sum(1) / nf[draws].sum(1) - a[draws].sum(1) / nz[draws].sum(1))
            rows.append(boot_rows('I04', f'one-shot change, {part} per 100 chatbot runs', model, point, boot, 'pp'))
            rows.append(dict(item='I04', quantity=f'zero-shot {part} per 100 chatbot runs', group=model,
                             value=100 * a.sum() / nz.sum(), lo=np.nan, hi=np.nan, unit='percent'))
            rows.append(dict(item='I04', quantity=f'one-shot {part} per 100 chatbot runs', group=model,
                             value=100 * b.sum() / nf.sum(), lo=np.nan, hi=np.nan, unit='percent'))
    return rows


def i10_rows(runs: pd.DataFrame, tasks: list[str], draws: np.ndarray) -> list[dict]:
    """Share of zero-shot failures that the example removes, by agency, and agent minus chatbot."""
    rows = []
    for model in [*MODELS, 'all']:
        q = runs if model == 'all' else runs[runs.model.eq(model)]
        shares, boots = {}, {}
        for agency in six.AGENCIES:
            d = q[q.agency.eq(agency)].assign(fail=lambda x: ~x.primary)
            zs, fs = d[d.prompt.eq('zero_shot')], d[d.prompt.eq('few_shot')]
            fz, nz = task_totals(zs, tasks, 'fail'), task_totals(zs, tasks)
            ff, nf = task_totals(fs, tasks, 'fail'), task_totals(fs, tasks)
            with np.errstate(divide='ignore', invalid='ignore'):
                shares[agency] = 100 * (1 - (ff.sum() / nf.sum()) / (fz.sum() / nz.sum())) if fz.sum() else np.nan
                boots[agency] = 100 * (1 - (ff[draws].sum(1) / nf[draws].sum(1)) / (fz[draws].sum(1) / nz[draws].sum(1)))
            rows.append(boot_rows('I10', f'failures removed by the example, {agency}', model, shares[agency],
                                  boots[agency], 'percent'))
        rows.append(boot_rows('I10', 'failures removed, agent minus chatbot', model,
                              shares['agent'] - shares['chatbot'], boots['agent'] - boots['chatbot'], 'pp'))
    return rows


def i13_rows(runs: pd.DataFrame, pricing: pd.DataFrame, tasks: list[str], draws: np.ndarray) -> list[dict]:
    """Percent change in mean cost per run from zero-shot to one-shot, by model and pooled."""
    data = runs[['record_path', 'task', 'model', 'prompt']].merge(pricing[['record_path', 'cost_usd']],
                                                                  on='record_path', validate='one_to_one')
    data = data[data.cost_usd.notna()]
    rows = []
    groups = {**{m: [m] for m in MODELS}, 'all': MODELS, 'all_without_sonnet': [m for m in MODELS if m != 'sonnet']}
    reduction = {}
    for name, members in groups.items():
        q = data[data.model.isin(members)]
        zs, fs = q[q.prompt.eq('zero_shot')], q[q.prompt.eq('few_shot')]
        cz, nz = task_totals(zs, tasks, 'cost_usd'), task_totals(zs, tasks)
        cf, nf = task_totals(fs, tasks, 'cost_usd'), task_totals(fs, tasks)
        point = 100 * ((cf.sum() / nf.sum()) / (cz.sum() / nz.sum()) - 1)
        boot = 100 * ((cf[draws].sum(1) / nf[draws].sum(1)) / (cz[draws].sum(1) / nz[draws].sum(1)) - 1)
        rows.append(boot_rows('I13', 'mean cost change, one-shot relative to zero-shot', name, point, boot, 'percent'))
        reduction[name] = cz.sum() / nz.sum() - cf.sum() / nf.sum()
        rows.append(dict(item='I13', quantity='mean cost per run, zero-shot', group=name, value=cz.sum() / nz.sum(),
                         lo=np.nan, hi=np.nan, unit='usd'))
        rows.append(dict(item='I13', quantity='mean cost per run, one-shot', group=name, value=cf.sum() / nf.sum(),
                         lo=np.nan, hi=np.nan, unit='usd'))
    # Pooled change in total cost by model (balanced design: equal run counts).
    totals = data.groupby(['model', 'prompt']).cost_usd.sum().unstack()
    share = 100 * (totals.zero_shot - totals.few_shot)['sonnet'] / (totals.zero_shot - totals.few_shot).sum()
    rows.append(dict(item='I13', quantity='share of the pooled cost reduction from Sonnet', group='all', value=share,
                     lo=np.nan, hi=np.nan, unit='percent'))
    return rows


def i17_rows(runs: pd.DataFrame, tasks: list[str], draws: np.ndarray) -> list[dict]:
    """Chatbot pass@10 (any success in the ten runs of a cell) minus one agent run (pass@1)."""
    cells = runs.groupby(CELL).primary.agg(['sum', 'size']).reset_index()
    if not cells['size'].eq(10).all():
        raise ValueError('pass@10 requires ten runs in every cell')
    cells['pass10'] = (cells['sum'] > 0).astype(float)
    cells['pass1'] = cells['sum'] / cells['size']
    rows = []
    for model in [*MODELS, 'all']:
        q = cells if model == 'all' else cells[cells.model.eq(model)]
        chat, agent = q[q.agency.eq('chatbot')], q[q.agency.eq('agent')]
        p10, n10 = task_totals(chat, tasks, 'pass10'), task_totals(chat, tasks)
        p1, n1 = task_totals(agent, tasks, 'pass1'), task_totals(agent, tasks)
        point = 100 * (p10.sum() / n10.sum() - p1.sum() / n1.sum())
        boot = 100 * (p10[draws].sum(1) / n10[draws].sum(1) - p1[draws].sum(1) / n1[draws].sum(1))
        rows.append(boot_rows('I17', 'chatbot pass@10 minus one agent run', model, point, boot, 'pp'))
        rows.append(dict(item='I17', quantity='chatbot pass@10', group=model, value=100 * p10.sum() / n10.sum(),
                         lo=np.nan, hi=np.nan, unit='percent'))
        rows.append(dict(item='I17', quantity='one agent run', group=model, value=100 * p1.sum() / n1.sum(),
                         lo=np.nan, hi=np.nan, unit='percent'))
    return rows


NUMBER = r"(-?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def lenient_values(text: str, platform: str, terms: list[str]) -> dict[str, float]:
    """Estimates in any of the shapes the strict parser misses.

    Stata: a line '<term> ... estimate=<number>' with or without the ANSWER prefix and in any
    field order. R and Python: result JSON with estimates as a list of {term, estimate}, as a
    dict keyed by term, or with prose around the JSON object.
    """
    if platform == 'stata':
        out = {}
        for term in terms:
            for match in re.finditer(rf"(?m)^\s*(?:ANSWER:?\s*)?{re.escape(term)}\b[^\n]*?estimate\s*=\s*{NUMBER}", text):
                out[term] = float(match.group(1))
        return out
    data = None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        found = re.search(r'\{.*\}', text, re.S)
        if found:
            try:
                data = json.loads(found.group(0))
            except ValueError:
                data = None
    estimates = data.get('estimates') if isinstance(data, dict) else None
    out = {}
    if isinstance(estimates, list):
        for row in estimates:
            if isinstance(row, dict) and isinstance(row.get('term'), str):
                try:
                    out[row['term']] = float(row.get('estimate'))
                except (TypeError, ValueError):
                    pass
    elif isinstance(estimates, dict):
        for key, row in estimates.items():
            value = row.get('estimate') if isinstance(row, dict) else row
            try:
                out[key] = float(value[0] if isinstance(value, list) else value)
            except (TypeError, ValueError, IndexError):
                pass
    return out


def i42_rows(data: dict, out: Path) -> list[dict]:
    """Loud failures whose output holds every scored estimate within tolerance on a lenient read."""
    runs, catalog = data['runs'], data['catalog']
    tolerances = term_tolerances(data['terms'], catalog).set_index(['task', 'term'])
    loud = runs[runs.failure_category.eq('Loud failure')]
    rows = []
    for rec in loud.itertuples(index=False):
        wanted = [t for t in catalog[rec.task]['estimands'] if (rec.task, t) in tolerances.index]
        path = ROOT / rec.output_path if isinstance(rec.output_path, str) else None
        text = path.read_text(errors='replace') if path is not None and path.is_file() else ''
        got = lenient_values(text, rec.platform, wanted)
        within = bool(wanted) and all(
            t in got and abs(got[t] - tolerances.loc[(rec.task, t), 'ref_estimate'])
            <= tolerances.loc[(rec.task, t), 'tol'] * (1 + 1e-12) for t in wanted)
        rows.append(dict(position_id=rec.position_id, model=rec.model, platform=rec.platform, agency=rec.agency,
                         execution_ok=rec.execution_ok, output_available=rec.output_available,
                         found=sum(t in got for t in wanted), needed=len(wanted), all_within=within))
    frame = pd.DataFrame(rows)
    write_csv(frame, out / 'lenient_loud.csv')
    unread = frame.all_within & ~frame.output_available
    result = [dict(item='I42', quantity='loud failures', group='all', value=len(frame), lo=np.nan, hi=np.nan, unit='runs'),
              dict(item='I42', quantity='loud failures with all estimates within tolerance on a lenient read',
                   group='all', value=int(frame.all_within.sum()), lo=np.nan, hi=np.nan, unit='runs'),
              dict(item='I42', quantity='of these, values the strict parser does not read', group='all',
                   value=int(unread.sum()), lo=np.nan, hi=np.nan, unit='runs'),
              dict(item='I42', quantity='share of loud failures with correct values in an unread shape', group='all',
                   value=100 * float(unread.mean()), lo=np.nan, hi=np.nan, unit='percent')]
    for platform, g in frame[unread].groupby('platform'):
        result.append(dict(item='I42', quantity='correct values in an unread shape', group=platform, value=len(g),
                           lo=np.nan, hi=np.nan, unit='runs'))
    return result


def analysis_small(data: dict, out: Path) -> dict:
    runs = data['runs']
    tasks = sorted(runs.task.unique())
    draws = boot_index(tasks)
    rows = [*i04_rows(runs, tasks, draws), *i10_rows(runs, tasks, draws),
            *i13_rows(runs, data['pricing'], tasks, draws), *i17_rows(runs, tasks, draws), *i42_rows(data, out)]
    frame = pd.DataFrame(rows).assign(bootstrap_draws=style.BOOTSTRAP_DRAWS, bootstrap_seed=style.BOOTSTRAP_SEED)
    write_csv(frame, out / 'small_numbers.csv')

    def pick(item, quantity, group='all'):
        rec = frame[(frame['item'] == item) & (frame.quantity == quantity) & (frame.group == group)].iloc[0]
        return [float(rec.value), float(rec.lo), float(rec.hi)]

    return dict(
        i04={part: pick('I04', f'one-shot change, {part} per 100 chatbot runs')
             for part in ('success', 'did_not_run', 'ran_incomplete', 'silent')},
        i10={a: pick('I10', f'failures removed by the example, {a}') for a in six.AGENCIES} | dict(
            agent_minus_chatbot=pick('I10', 'failures removed, agent minus chatbot')),
        i13={g: pick('I13', 'mean cost change, one-shot relative to zero-shot', g)
             for g in [*MODELS, 'all', 'all_without_sonnet']} | dict(
            sonnet_share_of_reduction=pick('I13', 'share of the pooled cost reduction from Sonnet')[0]),
        i17={g: pick('I17', 'chatbot pass@10 minus one agent run', g) for g in [*MODELS, 'all']},
        i42={q: pick('I42', q)[0] for q in ('loud failures',
                                            'loud failures with all estimates within tolerance on a lenient read',
                                            'of these, values the strict parser does not read',
                                            'share of loud failures with correct values in an unread shape')})


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
ANALYSES = dict(table1_inference=analysis_table1, se_only_silent=analysis_se_only, majority_vote=analysis_majority,
                shared_wrong=analysis_shared_wrong, self_checks=analysis_self_checks, small_numbers=analysis_small)


def native(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def clean(value):
    """JSON-safe copy: NaN and infinity become None."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return clean(value.item())
    return value


def build(out: Path = OUT, data_dir: Path = DATA, only: list[str] | None = None) -> dict:
    """Run the selected analyses on the current data and write their outputs to ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(data_dir)
    names = list(ANALYSES) if not only else only
    unknown = set(names) - set(ANALYSES)
    if unknown:
        raise ValueError(f'Unknown analyses: {sorted(unknown)}')
    receipt_path = out / 'extra_receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() and only else {}
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in
              [Path(data_dir) / f'data/{n}' for n in ('runs.csv', 'term_checks.csv', 'se_checks.csv',
                                                       'run_pricing.csv', 'catalog.json')]
              + [Path(data_dir) / 'tables/regression_contrasts.csv', Path(__file__)]}
    receipt.update(built_at=datetime.now(timezone.utc).isoformat(), inputs=inputs, provider_calls=0,
                   runs=len(data['runs']), successes=int(data['runs'].primary.sum()),
                   bootstrap=dict(draws=style.BOOTSTRAP_DRAWS, seed=style.BOOTSTRAP_SEED),
                   wild_bootstrap=dict(draws=WCB_DRAWS, seed=WCB_SEED, weights='Rademacher'))
    results = receipt.setdefault('results', {})
    for name in names:
        results[name] = clean(ANALYSES[name](data, out))
        print(f'{name}: done', flush=True)
    receipt['outputs'] = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != receipt_path.name}
    receipt_path.write_text(json.dumps(clean(receipt), indent=2, default=native) + '\n')
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data', type=Path, default=DATA, help='analysis folder with data/ and tables/')
    parser.add_argument('--out', type=Path, default=OUT, help='output folder')
    parser.add_argument('--only', type=lambda s: [x for x in s.split(',') if x], default=None,
                        help='comma-separated analyses: ' + ','.join(ANALYSES))
    args = parser.parse_args(argv)
    receipt = build(args.out, args.data, args.only)
    print(json.dumps({k: receipt['results'][k] for k in (args.only or ANALYSES)}, indent=1, default=native)[:6000])


if __name__ == '__main__':
    main()

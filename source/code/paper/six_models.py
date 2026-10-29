"""Build the paper's figures, Table 1, and a summary table with all collected models.

Models (name with effort), in pairs of one Claude and one GPT model:
- original pair: Claude Sonnet 4.6 medium (`sonnet`), GPT-5.6 Luna low (`luna`), from the paper sample;
- low cost: Claude Haiku 5.5 low (`haiku`), GPT-6 Luna low (`luna6`);
- high cost: Claude Opus 5.5 low (`opus`), GPT-6 Sol low (`sol`).
New models come from data/collections/2026-10-17_full-design/<key>_paper_rows.csv and <key>_paper_pricing.csv
(code/benchmark/workflows/full_design_2026_10_17/paper_rows.py). A model enters only when all 7,200 runs are
complete; the build lists the models it used. The manuscript and the paper analysis are not changed.
Task success and the failure type come from data/analysis/six_models/data/runs.csv, the outcome of
code/paper/pnas_analysis.py after the wrong-method check (checked_outcome), so run that script first. It reads
data/appendix_pass_at_k.csv from this build, so rerun it after this build.

The figures use the PNAS settings of pnas_style(): 17.8 cm width, Helvetica in every figure (the design figure
through LuaLaTeX), axes from 0, and the pink accent only for the constrained agent (peer review 2026-10-28).

Figures use the paper's generators and style (code/benchmark/production_paper_figures.py,
production_paper_costs.py, paper_figure_style.py). Pooled figures pool all included models. Figures with one
panel per model use a 3 x 2 grid: one row per pair, Claude on the left, GPT on the right. Table 1 uses the paper's
regression (task fixed effects, CR1 errors clustered by task, contrasts of production_statistics.weights) for each
model and for three pooled columns (model-by-task fixed effects): original pair, low cost, high cost. Wild cluster
bootstrap p-values are not computed.

With --groups, the build pools the models into two cost groups and writes data/analysis/six_models/cheap_vs_expensive:
- cheap: GPT-5.6 Luna low, Claude Haiku 5.5 low, GPT-6 Luna low (US$16 to US$32 for 7,200 runs);
- expensive: Claude Sonnet 4.6 medium, Claude Opus 5.5 low, GPT-6 Sol low (US$251 to US$588).
Each group pools 21,600 runs with equal weight per model. Table 1 uses model-by-task fixed effects in each group.
The pass@k figure is left out, because pass@k describes repeated runs of one model.
The group cost figure is also copied to data/analysis/six_models/figures/cost_of_correctness_groups.pdf,
which the PNAS manuscript uses as its cost figure.

Run from the repository root:
    uv run --no-sync python code/paper/six_models.py [--out data/analysis/six_models] [--groups]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import factorial_analysis as fa  # noqa: E402
import paper_figure_style as style  # noqa: E402
from paper_numbers import fixed  # noqa: E402
import paper_table1 as t1  # noqa: E402
import production_paper_assets as assets  # noqa: E402
import production_paper_costs as ppc  # noqa: E402
import production_paper_figures as ppf  # noqa: E402

PAPER = ROOT / 'data/analysis/main/data'
NEW = ROOT / 'data/collections/2026-10-17_full-design'
OUT = ROOT / 'data/analysis/six_models'
NAMES = {'sonnet': 'Claude Sonnet 4.6 medium', 'luna': 'GPT-5.6 Luna low', 'haiku': 'Claude Haiku 5.5 low',
         'luna6': 'GPT-6 Luna low', 'opus': 'Claude Opus 5.5 low', 'sol': 'GPT-6 Sol low'}
NEW_KEYS = {'haiku': 'haiku55', 'luna6': 'luna6', 'opus': 'opus55', 'sol': 'sol'}
GRID = [('Original pair', ['sonnet', 'luna']), ('Low cost', ['haiku', 'luna6']), ('High cost', ['opus', 'sol'])]
COST_GROUPS = {'cheap': ['luna', 'haiku', 'luna6'], 'expensive': ['sonnet', 'opus', 'sol']}
GROUP_NAMES = {'cheap': 'Cheap models', 'expensive': 'Expensive models'}
ORDER = [m for _, pair in GRID for m in pair]
GROUP_COLOR = {'Original pair': style.MID, 'Low cost': style.WALNUT, 'High cost': style.STRONG}
# Each model has its own line pattern, so the two models of a pair stay apart in grayscale (peer review 2026-10-28).
DASHES = ['-', (0, (3, 2)), (0, (1, 1.5)), (0, (5, 1.5, 1, 1.5)), (0, (6, 2)), (0, (5, 1.5, 1, 1.5, 1, 1.5))]
LINES = {m: dict(color=GROUP_COLOR[g], ls=DASHES[k]) for k, (g, m) in enumerate((g, m) for g, pair in GRID for m in pair)}
LINES.update(cheap=dict(color=style.WALNUT, ls=(0, (3, 2))), expensive=dict(color=style.STRONG, ls='-'))
RUNS_PER_MODEL = 7200
AGENCIES = ['chatbot', 'self_repair', 'agent']
PLATFORMS = ['python', 'r', 'stata']
# Text sizes stay between 6 and 12 pt at the drawn width (PNAS digital art rules), and every text ends at least
# TEXT_MARGIN_PT inside the left and right figure edges (peer review 2026-10-28, I76 and I77).
TEXT_MARGIN_PT = 4.0
TEXT_SIZE_RANGE_PT = (6.0, 12.0)


def label(model):
    """The figure label of a model, with the effort in parentheses as in Fig. 1 and the SI tables (I77).

    NAMES keeps its form, because the tables and the receipt use it.
    """
    return re.sub(r' (medium|low)$', r' (\1)', NAMES[model])


def load():
    paper = pd.read_csv(PAPER / 'runs.csv', low_memory=False).merge(
        pd.read_csv(PAPER / 'run_pricing.csv', usecols=['record_path', 'cost_usd']), on='record_path',
        validate='one_to_one')
    frames, included, skipped = [paper], ['sonnet', 'luna'], {}
    for label, key in NEW_KEYS.items():
        rows, prices = NEW / f'{key}_paper_rows.csv', NEW / f'{key}_paper_pricing.csv'
        if not rows.is_file():
            skipped[label] = 'no paper rows yet'
            continue
        frame = pd.read_csv(rows, low_memory=False)
        if len(frame) != RUNS_PER_MODEL:
            skipped[label] = f'{len(frame)} of {RUNS_PER_MODEL} runs'
            continue
        frame = frame.drop(columns=['model_id']).merge(
            pd.read_csv(prices, usecols=['record_path', 'cost_usd']), on='record_path', validate='one_to_one')
        frames.append(frame)
        included.append(label)
    data = pd.concat(frames, ignore_index=True)
    for name in ['primary', 'execution_ok', 'output_available']:
        data[name] = data[name].astype(bool)
    data['task_label'] = data.task.map(assets.SHORT_LABELS).fillna(data.task_label)
    cells = data.groupby(['model', 'task', 'platform', 'prompt', 'agency']).size()
    if not cells.eq(10).all() or data.groupby('model').task.nunique().ne(40).any():
        raise ValueError('Every included model needs 40 tasks and 10 runs in each of its 720 cells')
    # A run without provider token counts has no cost (one GPT-6 Luna run stopped at its output-token limit and
    # Codex reported no usage). Cost means and totals use the priced runs; the receipt lists the unpriced runs.
    unpriced = data[data.cost_usd.isna()].groupby('model').size().to_dict()
    if unpriced:
        skipped['unpriced runs'] = unpriced
    order = [m for m in ORDER if m in included]
    return data, order, skipped


def checked_outcome(data, analysis=None):
    """Take task success and the failure type from the PNAS analysis, after the wrong-method check.

    code/paper/pnas_analysis.py scores the runs, applies scoring_alternatives.apply_check, and writes the
    outcome to data/analysis/six_models/data/runs.csv. A success whose value lies closer to a listed
    wrong-method value than to the reference becomes a silent failure there, and alternative_checks.csv marks
    those runs (run_changed). load() keeps the outcome before the check, because pnas_analysis.py reads it.
    This function replaces `primary` and `failure_category` and checks that only the marked runs change, each
    from success to failure.
    """
    analysis = OUT / 'data' if analysis is None else Path(analysis)
    if not (analysis / 'runs.csv').is_file():
        raise SystemExit('Run code/paper/pnas_analysis.py first: the figures use its checked outcome')
    checked = pd.read_csv(analysis / 'runs.csv', low_memory=False,
                          usecols=['record_path', 'position_id', 'primary', 'failure_category', 'execution_ok',
                                   'output_available'])
    merged = data[['record_path', 'position_id', 'primary', 'execution_ok', 'output_available']].merge(
        checked, on='record_path', how='left', suffixes=('', '_checked'), validate='one_to_one')
    if len(checked) != len(data) or merged.primary_checked.isna().any():
        raise ValueError('The checked outcome covers other runs than the six-model sample')
    if not merged.position_id.eq(merged.position_id_checked).all():
        raise ValueError('The checked outcome assigns other design positions to the records')
    for name in ('execution_ok', 'output_available'):
        if not merged[name].eq(merged[f'{name}_checked'].astype(bool)).all():
            raise ValueError(f'The checked outcome changes {name}')
    after = merged.primary_checked.astype(bool).to_numpy()
    turned = merged.primary.to_numpy() & ~after
    if (~merged.primary.to_numpy() & after).any():
        raise ValueError('The checked outcome turns a failure into a success')
    alternatives = pd.read_csv(analysis / 'alternative_checks.csv')
    marked = set(alternatives.position_id[alternatives.run_changed.astype(bool)])
    if set(merged.position_id[turned]) != marked:
        raise ValueError('The changed runs differ from the runs that alternative_checks.csv marks')
    data = data.copy()
    data['primary'] = after
    data['failure_category'] = merged.failure_category.to_numpy()
    return data, int(turned.sum())


def configure(order):
    """Point the paper generators at the included models and their names."""
    ppf.MODELS = list(order)
    ppf.MODEL_LABEL = {m: NAMES[m] for m in order}
    ppc.MODELS = list(order)
    ppc.MODEL_NAMES = {m: NAMES[m] for m in order}
    pnas_style()


def pnas_style():
    """Set the PNAS figure settings in the shared modules (peer review 2026-10-28).

    The AEA figures, which other scripts build, keep the module defaults.
    - Numerical axes go to 0 (M14). Log axes are exempt.
    - Figures are 17.8 cm wide and set every text in Helvetica, as the design figure does (I76).
    - The pink accent marks only the constrained agent, and its lines use the darker ROSE (I73). Figure 2 draws
      the one-shot prompt and R in gray and Stata with a dashed line, and Figure 3 draws silent failures in gray.
    - Axis labels give the unit in parentheses (I77).
    """
    style.AXIS_FLOOR = 0
    style.FIG_W = style.PNAS_FIG_W
    style.STYLE_OVERRIDES = dict(style.PNAS_FONT)
    style.AGENCY_LINES = {**style.AGENCY_LINES, 'agent': style.ROSE}
    ppf.PROMPT_SERIES = {'zero_shot': style.SERIES['strong'], 'few_shot': style.SERIES['mid']}
    ppf.PLATFORM_SERIES = {'python': style.SERIES['strong'], 'r': style.SERIES['mid'], 'stata': style.SERIES['open']}
    ppf.FAILURE_SERIES = {'loud': style.SERIES['strong'], 'silent': style.SERIES['mid']}
    ppf.SILENT_SHARE_LABEL = 'Silent share of failures (percent)'


def benchmark_design(data, out, order):
    """Keep the paper's design figure, with the selected model list and counts."""
    text = ppf.design_source(data, out)
    old = r'\nreps\ repetitions per cell for each model, Claude Sonnet 4.6 (medium effort) and GPT-5.6 Luna (low reasoning effort):\\[1pt]'
    if text.count(old) != 1:
        raise ValueError('The design template model line changed')
    names = [NAMES[m].replace(' medium', ' (medium)').replace(' low', ' (low)') for m in order]
    rows = [', '.join(names[i:i + 3]) for i in range(0, len(names), 3)]
    new = (rf'\nreps\ repetitions per cell for each of {len(order)} models:\\[1pt]' + '\n  '
           + ('\\\\[1pt]\n  '.join(rows)) + r'\\[1pt]')
    text = pnas_design(text.replace(old, new))
    (out / 'data/benchmark_design.tex').write_text(text)
    with tempfile.TemporaryDirectory(prefix='pnas-design-') as temporary:
        folder = Path(temporary)
        (folder / 'benchmark_design.tex').write_text(text)
        run = subprocess.run(['lualatex', '-halt-on-error', '-interaction=nonstopmode', 'benchmark_design.tex'],
                             cwd=folder, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stdout[-3000:])
        shutil.copyfile(folder / 'benchmark_design.pdf', out / 'figures/benchmark_design.pdf')
    counts = ppf.design_counts(data, out)
    counts.update(models=len(order), completed=len(data))
    pd.DataFrame([counts]).to_csv(out / 'data/benchmark_design.csv', index=False)


def check_text(fig, stem):
    """Keep every text inside the side margins, check the text rules, and return the text sizes and margins."""
    margins = style.keep_side_margins(fig, TEXT_MARGIN_PT)
    style.assert_text_outside(fig)
    sizes = style.text_sizes(fig)
    low, high = TEXT_SIZE_RANGE_PT
    if sizes[0] < low or sizes[-1] > high:
        raise AssertionError(f'{stem}: text sizes {sizes} leave {low} to {high} pt')
    return dict(width_cm=round(fig.get_figwidth() * 2.54, 2), font_pt=[sizes[0], sizes[-1]],
                margin_pt={side: round(value, 2) for side, value in margins.items()})


# The PNAS design figure: the template's text in Helvetica through fontspec (LuaLaTeX), the font of the data
# figures, and its layout widened from 135.5 mm to 177.3 mm, so the page is 17.8 cm wide with the 1 pt border
# on each side (peer review 2026-10-28, I76). Each agency column is 50.55 mm of text plus 4.55 mm of padding
# and frame, 6 mm apart, and the count strip spans the full width. The other boxes widen in the same
# proportion. The type sizes stay at 6.9 to 8.4 pt.
DESIGN_FONT = (r'\usepackage[T1]{fontenc}' '\n' r'\usepackage{helvet}' '\n' r'\renewcommand{\familydefault}{\sfdefault}'
               '\n' r'\usepackage{sansmath}' '\n' r'\sansmath' '\n',
               r'\usepackage{fontspec}' '\n'
               r'\IfFontExistsTF{Helvetica}{\setmainfont{Helvetica}\setsansfont{Helvetica}}'
               r'{\setmainfont{TeX Gyre Heros}\setsansfont{TeX Gyre Heros}}' '\n'
               r'\renewcommand{\familydefault}{\sfdefault}' '\n')
DESIGN_LAYOUT = [
    (r'% Three agency columns across the 135.5 mm text width, 6 mm apart.',
     r'% Three agency columns across the 177.3 mm PNAS width, 6 mm apart.', 1),
    (r'\def\xL{20.9} \def\xM{67.75} \def\xR{114.6}', r'\def\xL{27.55} \def\xM{88.65} \def\xR{149.75}', 1),
    ('text width=37.6mm', 'text width=50.55mm', 1),
    ('[17.5mm][t]{37.6mm}', '[17.5mm][t]{50.55mm}', 1),
    ('text width=82mm', 'text width=107mm', 2),
    ('text width=112mm', 'text width=146mm', 1),
    ('text width=131mm', 'text width=173.85mm', 1),
]


def pnas_design(text):
    """The design template in the PNAS font and width."""
    for old, new, count in [(*DESIGN_FONT, 1), *DESIGN_LAYOUT]:
        if text.count(old) != count:
            raise ValueError(f'The design template changed near {old!r}')
        text = text.replace(old, new)
    return text


def saver(out, captions, layout=None):
    def save(fig, destination, stem, plotted, caption, source):
        if layout is not None:
            layout[stem] = check_text(fig, stem)
        fig.savefig(out / 'figures' / f'{stem}.pdf', metadata={'Title': stem})
        fig.savefig(out / 'figures' / f'{stem}.png', dpi=200)
        plt.close(fig)
        plotted.to_csv(out / 'data' / f'{stem}.csv', index=False)
        captions[stem] = caption.replace('both models pooled', 'all models in the figure pooled').replace(
            'pooling models', 'pooling all models in the figure')
    return save


def grid_axes(order, row_height, extra, sharey=True, **adjust):
    """One row per pair with an included model; Claude on the left, GPT on the right."""
    rows = [(group, pair) for group, pair in GRID if any(m in order for m in pair)]
    height = row_height * len(rows) + extra
    # Margins in inches, converted to figure fractions for this height.
    adjust = dict(adjust, top=1 - adjust['top'] / height, bottom=adjust['bottom'] / height)
    fig, axes = plt.subplots(len(rows), 2, figsize=(style.FIG_W, height), sharey=sharey, squeeze=False)
    fig.subplots_adjust(**adjust)
    placed, letter = [], iter('ABCDEF')
    for r, (group, pair) in enumerate(rows):
        for c, model in enumerate(pair):
            if model in order:
                placed.append((axes[r][c], model, f'{next(letter)}. {label(model)}'))
            else:
                axes[r][c].set_axis_off()
        if pair[0] not in order and pair[1] in order:
            axes[r][1].tick_params(axis='y', labelleft=True)
    return fig, axes, placed


def prompt_by_model(data, order, out, save):
    tasks = sorted(data.task.unique())
    plotted = []
    with style.figure_style():
        fig, axes, placed = grid_axes(order, 2.35, .6, left=.085, right=.955,
                                      top=.6, bottom=.4, wspace=.10, hspace=.62)
        for ax, model, title in placed:
            values = ppf.success_rates(data[data.model.eq(model)], ['prompt', 'agency'], tasks)
            values['model'] = model
            plotted.append(values)
            for key, offset in zip(ppf.PROMPTS, ppf.dodge(2)):
                rows = values[values.prompt.eq(key)].set_index('agency').loc[AGENCIES]
                style.plot_series(ax, ppf.X + offset, rows.success_pct.to_numpy(), rows.success_pct_lo.to_numpy(),
                                  rows.success_pct_hi.to_numpy(), ppf.PROMPT_SERIES[key], gid=f'{model}_{key}')
            style.panel_title(ax, title, pad=4)
            ppf.agency_axis(ax)
        plotted = pd.concat(plotted, ignore_index=True)
        style.success_axis(placed[0][0], plotted.success_pct_lo)
        fig.legend([style.series_handle(ppf.PROMPT_SERIES[k]) for k in ppf.PROMPTS],
                   [ppf.PROMPT_LABEL[k] for k in ppf.PROMPTS], loc='upper left', bbox_to_anchor=(.085, .995),
                   ncol=2, frameon=False)
        save(fig, out, 'prompt_agency_by_model', plotted,
             'Task success by prompt and agency condition, one panel per model, languages pooled. Rows are the '
             'original pair, the low-cost pair, and the high-cost pair. ' + ppf.INTERVAL_CAPTION, None)


def pass_by_model(data, out, save, order):
    plotted, task_count = ppf.pass_at_k_data(data)
    repetitions = int(plotted.repetitions.iloc[0])
    offsets = dict(zip(AGENCIES, (-.16, 0, .16)))
    lower = 0  # PNAS asks that numerical axes go to 0 (peer review 2026-10-28).
    with style.figure_style():
        fig, axes, placed = grid_axes(order, 2.35, .6, left=.085, right=.955,
                                      top=.6, bottom=.45, wspace=.10, hspace=.62)
        for ax, model, title in placed:
            for agency in AGENCIES:
                rows = plotted[plotted.model.eq(model) & plotted.agency.eq(agency)]
                x = rows.k.to_numpy() + offsets[agency]
                ax.plot(x, rows.pass_at_k_pct, color=style.AGENCY_LINES[agency], lw=1.2, zorder=2)
                style.whiskers(ax, x, rows.pass_at_k_pct, rows.pass_at_k_pct_lo, rows.pass_at_k_pct_hi,
                               style.AGENCY_WHISKERS[agency])
                marker = {**style.AGENCY_MARKERS[agency], 'ms': style.AGENCY_MARKERS[agency]['ms'] - .8}
                ax.plot(x, rows.pass_at_k_pct, ls='none', zorder=4, clip_on=False, **marker)
            ax.set_ylim(lower, 101)
            # Even ticks that reach 100 (peer review 2026-10-28, I71).
            ax.set_yticks(range(lower, 101, 20))
            ax.set_xlim(.6, repetitions + .4)
            ax.set_xticks(range(1, repetitions + 1))
            style.hairline_grid(ax)
            style.panel_title(ax, title, pad=4)
            # k is italic, as in the legend and the text (Pass@$k$).
            ax.set_xlabel('Runs drawn ($k$)', fontsize=style.AXIS_LABEL_SIZE)
        for row in axes:
            row[0].set_ylabel('Pass@$k$ (percent)', fontsize=style.AXIS_LABEL_SIZE)
        fig.legend([style.agency_handle(a, line=True) for a in AGENCIES], [ppf.AGENCY_LABEL[a] for a in AGENCIES],
                   loc='upper left', bbox_to_anchor=(.085, .995), ncol=3, frameon=False)
        save(fig, out, 'appendix_pass_at_k', plotted,
             f'Unbiased pass@k from all {repetitions} repetitions in every cell of {task_count} tasks, one panel per '
             'model. Whiskers are 95 percent intervals from a cluster bootstrap over tasks '
             f'({style.BOOTSTRAP_DRAWS:,} draws, seed {style.BOOTSTRAP_SEED}).', None)


def runtime_by_model(data, out, save, order):
    tasks = sorted(data.task.unique())
    plotted = data.groupby(['model', 'agency']).provider_seconds.agg(['median', 'count', 'size']).reset_index()
    plotted.columns = ['model', 'agency', 'median_provider_seconds', 'n_observed', 'n_total']
    boot = style.bootstrap_medians(data, ['model', 'agency'], 'provider_seconds', tasks=tasks)
    plotted = plotted.merge(boot[['model', 'agency', 'value_lo', 'value_hi']].rename(
        columns={'value_lo': 'median_provider_seconds_lo', 'value_hi': 'median_provider_seconds_hi'}),
        on=['model', 'agency'], validate='one_to_one')
    top = 10 * math.ceil(plotted.median_provider_seconds_hi.max() * 1.05 / 10)
    with style.figure_style():
        fig, axes, placed = grid_axes(order, 2.0, .3, left=.085, right=.955,
                                      top=.25, bottom=.35, wspace=.10, hspace=.55)
        for ax, model, title in placed:
            rows = plotted[plotted.model.eq(model)].set_index('agency').loc[AGENCIES]
            style.plot_series(ax, ppf.X, rows.median_provider_seconds.to_numpy(),
                              rows.median_provider_seconds_lo.to_numpy(), rows.median_provider_seconds_hi.to_numpy(),
                              style.SERIES['strong'], gid=f'{model}_median')
            ax.set_ylim(0, top)
            ax.set_yticks(range(0, top + 1, 10 if top <= 80 else 20))
            ppf.agency_axis(ax)
            style.panel_title(ax, title, pad=4)
        for row in axes:
            # The caption and the text call the measure runtime (peer review 2026-10-28, I77).
            row[0].set_ylabel('Median runtime (seconds)', fontsize=style.AXIS_LABEL_SIZE)
        save(fig, out, 'runtime_by_agency', plotted,
             'Median provider wall time per run, one panel per model. Whiskers are 95 percent intervals for the '
             f'median from a cluster bootstrap over tasks ({style.BOOTSTRAP_DRAWS:,} draws, seed {style.BOOTSTRAP_SEED}).',
             None)


def resource_points(data, resource='cost_usd'):
    """Use all outcomes for success and observed usage for resource means."""
    points = ppc.resource_points(data, resource)
    keys = ['model', 'platform', 'agency']
    success = data.groupby(keys).primary.mean()
    points['success_pct'] = 100 * success.reindex(pd.MultiIndex.from_frame(points[keys])).to_numpy()
    return points


LANGUAGE_LINES = {'python': '-', 'r': (0, (3, 2)), 'stata': (0, (1, 1.6))}
# A log cost axis spans the points of its panel times and divided by LOG_PAD.
LOG_PAD = 1.3


def log_ticks(low, high):
    """Ticks at 1, 2, 3, and 5 times a power of ten inside the axis limits."""
    ticks = [m * 10.0 ** e for e in range(-3, 4) for m in (1, 2, 3, 5)]
    return [t for t in ticks if low * (1 - 1e-9) <= t <= high * (1 + 1e-9)]


def resource_by_model(data, out, save, order, resource, intervals=False):
    """Task success against mean resource per run, one panel per model.

    With intervals=True (the PNAS group cost figure), every point gets a 95 percent whisker from the task cluster
    bootstrap of style.bootstrap_ratios (the draws and seed of the other figures), the legend names the
    languages by line style, so no language label sits beside the panels, and each panel has a log cost axis.
    """
    points = resource_points(data, resource)
    unit = 'US cents' if resource == 'cost_usd' else 'thousands'
    if intervals:
        keys = ['model', 'platform', 'agency']
        boot = style.bootstrap_ratios(data.assign(success=data.primary.astype(float)), keys, 'success',
                                      tasks=sorted(data.task.unique()))
        points = points.merge(boot[keys + ['value', 'value_lo', 'value_hi']], on=keys, validate='one_to_one')
        if not np.allclose(points.value, points.success_pct):
            raise ValueError('The bootstrap point estimates differ from the plotted success rates')
        points = points.drop(columns='value').rename(columns={'value_lo': 'success_lo', 'value_hi': 'success_hi'})
    with style.figure_style():
        margins = dict(right=.955, wspace=.10) if intervals else dict(right=.9, wspace=.36)
        fig, axes, placed = grid_axes(order, 2.45, .6, sharey=True, left=.085,
                                      top=.55, bottom=.45, hspace=.62, **margins)
        ylim = (0, 101)  # PNAS asks that numerical axes go to 0 (peer review 2026-10-28).
        for ax, model, title in placed:
            selected = points[points.model.eq(model)]
            low, high = float(selected.x_value.min()), float(selected.x_value.max())
            if intervals:
                # The PNAS group figure (Fig. 4) uses a log cost axis in each panel. On a linear axis from 0, the
                # chatbot and self-repair points of Python and R overlap (peer review 2026-10-28, I72). A log
                # axis is exempt from the zero rule.
                ax.set_xscale('log')
                ax.set_xlim(low / LOG_PAD, high * LOG_PAD)
                ticks = log_ticks(low / LOG_PAD, high * LOG_PAD)
                ax.set_xticks(ticks, [ppc.tick_text(t) for t in ticks])
            else:
                pad = (high - low) * .08 or high * .1
                ax.set_xlim(0, high + pad)
            ax.set_ylim(*ylim)
            ax.set_yticks(range(0, 101, 20))
            ax.xaxis.set_minor_locator(NullLocator())
            style.hairline_grid(ax)
            ends = []
            for platform in PLATFORMS:
                group = selected[selected.platform.eq(platform)].set_index('agency').loc[AGENCIES]
                ax.plot(group.x_value, group.success_pct, color=style.MID, lw=.9, zorder=2,
                        ls=LANGUAGE_LINES[platform] if intervals else '-')
                if intervals:
                    style.whiskers(ax, group.x_value, group.success_pct, group.success_lo, group.success_hi,
                                   style.MID, gid=f'{model}_{platform}_interval')
                for agency, row in group.iterrows():
                    ax.plot([row.x_value], [row.success_pct], ls='none', zorder=4, clip_on=False,
                            **style.AGENCY_MARKERS[agency])
                ends.append((group.x_value.iloc[-1], group.success_pct.iloc[-1]))
            style.panel_title(ax, title, pad=4)
            ax.set_xlabel(f'Mean cost per run ({unit})' if resource == 'cost_usd'
                          else f'Mean input and output tokens ({unit})', fontsize=style.AXIS_LABEL_SIZE)
            if intervals:
                continue
            fig.canvas.draw()
            height_pt = ax.get_window_extent().height * 72 / fig.dpi
            per_point = (ylim[1] - ylim[0]) / height_pt
            label_ys = style.stacked_label_y([y for _, y in ends], ppc.LABEL_GAP_PT * per_point,
                                             bounds=(ylim[0] + 5 * per_point, ylim[1] - 5 * per_point))
            xlim = ax.get_xlim()
            width_pt = ax.get_window_extent().width * 72 / fig.dpi
            label_x = xlim[1] + ppc.LABEL_PAD_PT * (xlim[1] - xlim[0]) / width_pt
            for platform, (x, y), label_y in zip(PLATFORMS, ends, label_ys):
                ax.annotate(ppc.PLATFORM_NAMES[platform], xy=(x, y), xytext=(label_x, label_y),
                            fontsize=style.X_TICK_LABEL_SIZE, color=style.INK, ha='left', va='center',
                            annotation_clip=False, clip_on=False, zorder=5,
                            arrowprops=dict(arrowstyle='-', color=style.HAIR, lw=.5, linestyle=(0, (1, 2.2)),
                                            shrinkA=3.5, shrinkB=3.5, clip_on=False))
        for row in axes:
            row[0].set_ylabel('Task success (percent)', fontsize=style.AXIS_LABEL_SIZE)
        handles = [style.agency_handle(a) for a in AGENCIES]
        labels = [ppc.AGENCY_NAMES[a] for a in AGENCIES]
        if intervals:
            handles += [Line2D([], [], color=style.MID, lw=.9, ls=LANGUAGE_LINES[p]) for p in PLATFORMS]
            labels += [ppc.PLATFORM_NAMES[p] for p in PLATFORMS]
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.085, .995), ncol=len(labels), frameon=False)
        if intervals:
            style.assert_text_outside(fig)
        stem = 'cost_of_correctness' if resource == 'cost_usd' else 'cost_tokens'
        scale = 'logarithmic' if intervals else 'linear'
        caption = ((f'Task success and mean cost per run by language and agency, one panel per model, each with its own '
                    f'{scale} cost axis. Costs are list-price values in US cents.') if resource == 'cost_usd' else
                   'Task success and mean input plus output tokens per run, one panel per model.')
        if intervals:
            caption += (' Line styles mark the language. Whiskers are 95 percent intervals for task success from a '
                        f'cluster bootstrap over the 40 tasks ({style.BOOTSTRAP_DRAWS:,} draws, seed '
                        f'{style.BOOTSTRAP_SEED}).')
        save(fig, out, stem, points, caption, None)


def frontier(data, out, save, order, log=False):
    # log=True writes model_agency_frontier_log, the SI copy on a logarithmic cost axis (Raul, 2026-10-25).
    # The paper function needs a cost for every run. An unpriced run takes the mean of the priced runs of its model
    # and agency condition, which leaves that mean unchanged.
    cell = data.groupby(['model', 'agency']).cost_usd.transform('mean')
    points = ppc.model_agency_points(data.assign(cost_usd=data.cost_usd.fillna(cell)))
    with style.figure_style():
        fig = plt.figure(figsize=(style.FIG_W, 4.0))
        ax = fig.add_axes([.085, .12, .89, .66])
        width_pt = .89 * style.FIG_W * 72
        if log:
            low, high = float(points.cost_cents.min()), float(points.cost_cents.max())
            ax.set_xscale('log')
            ax.set_xlim(low / 1.25, high * 1.25)
            ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if low / 1.25 <= t <= high * 1.25]
        else:
            xlim = ppc.padded_linear_limits(float(points.cost_cents.min()), float(points.cost_cents.max()), width_pt)
            ax.set_xlim(*xlim)
            ticks = ppc.linear_ticks(*xlim)
        ax.set_xticks(ticks, [ppc.tick_text(t) for t in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        # PNAS asks that numerical axes go to 0 (peer review 2026-10-28).
        ax.set_ylim(0, 101)
        ax.set_yticks(range(0, 101, 10))
        style.hairline_grid(ax)
        for model in order:
            group = points[points.model.eq(model)].set_index('agency').loc[AGENCIES]
            ax.plot(group.cost_cents, group.success_pct, lw=1.1, zorder=2, **LINES[model])
            style.whiskers(ax, group.cost_cents, group.success_pct, group.success_lo, group.success_hi,
                           LINES[model]['color'])
            for agency, row in group.iterrows():
                ax.plot([row.cost_cents], [row.success_pct], ls='none', zorder=4, clip_on=False,
                        **style.AGENCY_MARKERS[agency])
        ax.set_xlabel('Mean cost per run (US cents)', fontsize=style.AXIS_LABEL_SIZE)
        ax.set_ylabel('Task success (percent)', fontsize=style.AXIS_LABEL_SIZE)
        handles = [style.agency_handle(a) for a in AGENCIES] + [Line2D([], [], lw=1.1, **LINES[m]) for m in order]
        labels = [ppc.AGENCY_NAMES[a] for a in AGENCIES] + [label(m) for m in order]
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.085, .99), ncol=3, frameon=False)
        save(fig, out, 'model_agency_frontier_log' if log else 'model_agency_frontier', points,
             'Task success and mean list-price cost per run by model and agency condition, pooled over prompts and '
             f'languages, on a {"logarithmic" if log else "linear"} cost axis. Each model has its own line pattern, and the color marks the pair (gray: '
             'original pair, brown: low cost, dark: high cost). Whiskers are 95 percent intervals from a cluster '
             'bootstrap over the 40 tasks.', None)


def table1(data, order, out):
    helper = assets.load_module('six_model_statistics', assets.STATISTICS)
    ws = helper.weights()
    columns = [(m, NAMES[m], [m]) for m in order]
    for group, pair in GRID:
        if all(m in order for m in pair):
            columns.append((f'pooled_{group.lower().replace(" ", "_")}', f'{group} pooled', pair))
    rows = []
    for key, name, models in columns:
        frame = data[data.model.isin(models)].copy()
        frame['success'] = frame.primary.astype(float)
        frame['cell'] = [fa.cell_index(r.platform, r.prompt, {'chatbot': 'no_feedback', 'self_repair': 'self_repair',
                                                              'agent': 'tool_loop'}[r.agency]) for r in frame.itertuples()]
        member = frame['member'] if 'member' in frame else frame.model
        frame['armtask'] = member + '__' + frame.task
        fe = 'armtask' if member.nunique() > 1 else 'task'
        fit = smf.ols(f'success ~ C({fe}) + C(cell, Treatment(1))', frame).fit(
            cov_type='cluster', cov_kwds={'groups': frame.task})
        for contrast, value in fa.contrast_table(fit, ws).items():
            se = value['se_cr1']
            if not np.isfinite(se) and abs(value['estimate']) < 1e-9:
                # Every run of both compared cells succeeds (Opus 5.5, zero-shot agent): the variance is zero up to
                # rounding, and its square root is NaN.
                se = 0.0
            rows.append(dict(column=key, column_name=name, contrast=contrast, estimate_pp=0.0 if abs(value['estimate']) < 1e-9 else 100 * value['estimate'],
                             se_cr1_pp=100 * se, runs=len(frame)))
        chatbot = frame[frame.agency.eq('chatbot')]
        rows.append(dict(column=key, column_name=name, contrast='chatbot_all', estimate_pp=100 * chatbot.success.mean(),
                         se_cr1_pp=np.nan, runs=len(frame)))
        zs = chatbot[chatbot.prompt.eq('zero_shot') & chatbot.platform.eq('python')]
        rows.append(dict(column=key, column_name=name, contrast='chatbot_zs_python', estimate_pp=100 * zs.success.mean(),
                         se_cr1_pp=np.nan, runs=len(frame)))
    long = pd.DataFrame(rows)
    long.to_csv(out / 'tables/table1_contrasts.csv', index=False)
    names = [name for _, name, _ in columns]
    lines = ['| Contrast | ' + ' | '.join(names) + ' |', '|---|' + '---:|' * len(names)]
    for panel in t1.PANELS:
        lines.append(f'| **{panel["title"]}** |' + ' |' * len(names))
        for contrast, label in panel['rows']:
            cells = []
            for key, _, _ in columns:
                row = long[long.column.eq(key) & long.contrast.eq(contrast)].iloc[0]
                cells.append(f'{fixed(row.estimate_pp)} ({fixed(row.se_cr1_pp)})')
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    lines.append('| **Reference levels (percent)** |' + ' |' * len(names))
    for label, contrast in t1.REFERENCE_ROWS:
        cells = [fixed(long[long.column.eq(k) & long.contrast.eq(contrast)].estimate_pp.iloc[0]) for k, _, _ in columns]
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    lines.append('| Runs | ' + ' | '.join(f'{long[long.column.eq(k)].runs.iloc[0]:,}' for k, _, _ in columns) + ' |')
    text = '\n'.join(lines) + ('\n\nEffects are differences in task-success rates in percentage points, from the '
                               "paper's regression with task fixed effects (model-by-task fixed effects in pooled "
                               'columns). Standard errors clustered by task (40 clusters) in parentheses. '
                               'Wild cluster bootstrap p-values are not computed.\n')
    (out / 'tables/table1.md').write_text(text)
    tex = [r'\begin{tabular}{l' + 'r' * len(names) + '}', r'\toprule',
           ' & ' + ' & '.join(n.replace(' ', r'\\ ', 1) if False else n for n in names) + r' \\', r'\midrule']
    for panel in t1.PANELS:
        tex.append(r'\multicolumn{' + str(len(names) + 1) + '}{l}{\\textit{' + panel['title'] + r'}} \\')
        for contrast, label in panel['rows']:
            est = [long[long.column.eq(k) & long.contrast.eq(contrast)].iloc[0] for k, _, _ in columns]
            tex.append(label + ' & ' + ' & '.join(fixed(r.estimate_pp) for r in est) + r' \\')
            tex.append(' & ' + ' & '.join(f'({fixed(r.se_cr1_pp)})' for r in est) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}']
    (out / 'tables/table1.tex').write_text('\n'.join(tex) + '\n')
    return long


def summary(data, order, out):
    rows = []
    for model in order:
        f = data[data.model.eq(model)]
        row = dict(model=NAMES[model], runs=len(f), success_pct=100 * f.primary.mean())
        for a in AGENCIES:
            row[f'{a}_pct'] = 100 * f[f.agency.eq(a)].primary.mean()
        for p in PLATFORMS:
            row[f'{p}_pct'] = 100 * f[f.platform.eq(p)].primary.mean()
        row['agent_minus_chatbot_pp'] = row['agent_pct'] - row['chatbot_pct']
        row['loud_failures'] = int((f.failure_category == 'Loud failure').sum())
        row['silent_failures'] = int((f.failure_category == 'Silent failure').sum())
        row['cost_total_usd'] = f.cost_usd.sum()
        for a in AGENCIES:
            row[f'cost_per_run_{a}_usd'] = f[f.agency.eq(a)].cost_usd.mean()
        row['median_provider_seconds'] = f.provider_seconds.median()
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'tables/summary_by_model.csv', index=False)
    shown = table.copy()
    for c in shown.columns:
        if c.endswith('_pct') or c.endswith('_pp'):
            shown[c] = shown[c].map(fixed)
        elif c.startswith('cost_per_run'):
            shown[c] = shown[c].map(lambda v: f'{v:.4f}')
        elif c == 'cost_total_usd':
            shown[c] = shown[c].map(lambda v: f'{v:,.2f}')
    (out / 'tables/summary_by_model.md').write_text(shown.to_markdown(index=False) + '\n')
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--groups', action='store_true', help='pool the models into a cheap and an expensive group')
    args = parser.parse_args(argv)
    out = args.out or (OUT / 'cheap_vs_expensive' if args.groups else OUT)
    for sub in ('figures', 'data', 'tables'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    data, order, skipped = load()
    data, alternative_failures = checked_outcome(data)
    if args.groups:
        global GRID
        missing = [m for members in COST_GROUPS.values() for m in members if m not in order]
        if missing:
            raise SystemExit(f'The cost groups need all six models; missing: {missing}')
        data['member'] = data.model
        data['model'] = data.model.map({m: g for g, members in COST_GROUPS.items() for m in members})
        NAMES.update(GROUP_NAMES)
        GRID = [('All six models', ['cheap', 'expensive'])]
        order = ['cheap', 'expensive']
    configure(order)
    captions, layout = {}, {}
    save = saver(out, captions, layout)
    if args.groups:
        # The pooled paper figures would repeat the six-model build, so each group gets its own copy.
        for group in order:
            def save_group(fig, destination, stem, plotted, caption, source, group=group):
                save(fig, destination, f'{stem}_{group}', plotted, f'{NAMES[group]}. {caption}', source)
            subset = data[data.model.eq(group)]
            ppf.task_figure(subset, out, None, save_group, assets.SHORT_LABELS)
            ppf.failure_figure(subset, out, None, save_group)
    else:
        benchmark_design(data, out, order)
        ppf.prompt_figure(data, out, None, save)
        ppf.task_figure(data, out, None, save, assets.SHORT_LABELS)
        ppf.failure_figure(data, out, None, save)
    prompt_by_model(data, order, out, save)
    if not args.groups:
        pass_by_model(data, out, save, order)
    runtime_by_model(data, out, save, order)
    resource_by_model(data, out, save, order, 'cost_usd', intervals=args.groups)
    if args.groups:
        # The PNAS cost figure (fig:cost_quality) uses the two group panels; it reads one figures folder.
        for suffix in ('pdf', 'png'):
            shutil.copyfile(out / f'figures/cost_of_correctness.{suffix}',
                            OUT / f'figures/cost_of_correctness_groups.{suffix}')
    resource_by_model(data, out, save, order, 'total_main_tokens')
    frontier(data, out, save, order)
    frontier(data, out, save, order, log=True)
    table1(data, order, out)
    table = summary(data, order, out)
    receipt = dict(built_at=datetime.now(timezone.utc).isoformat(), models=[NAMES[m] for m in order],
                   runs=len(data), skipped={NAMES.get(k, k): v for k, v in skipped.items()}, captions=captions,
                   figure_text=layout,
                   outcome=dict(source='data/analysis/six_models/data/runs.csv (after the wrong-method check)',
                                successes=int(data.primary.sum()), turned_by_alternative_check=alternative_failures))
    if args.groups:
        receipt['groups'] = {NAMES[g]: [NAMES[m] for m in members] for g, members in COST_GROUPS.items()}
    (out / 'build_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(models=receipt['models'], skipped=receipt['skipped'], runs=len(data))))
    print(table[['model', 'success_pct', 'chatbot_pct', 'agent_pct', 'cost_total_usd']].round(2).to_string(index=False))


if __name__ == '__main__':
    main()

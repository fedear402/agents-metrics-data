"""Render the saved production results as the paper figures.

Every data figure uses the Figure 2 design in paper_figure_style.py. The
prompt, failure, pass@k, and runtime figures draw its task-bootstrap
intervals. The task figure (Figure 3) and the cost figures draw no interval,
and their success axes span the plotted points with a small edge pad. The
design figure is a TikZ template in the same font and colors, and the build
fills its counts from the sample. The module makes no provider calls and
writes no manuscript files.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import paper_figure_style as style
from production_paper_costs import EDGE_PAD_PT, padded_linear_limits

ROOT = Path(__file__).resolve().parents[2]
EXHIBITS = ROOT / 'code/exhibits'
MODELS = ['sonnet', 'luna']
MODEL_LABEL = {'sonnet': 'Claude Sonnet 4.6', 'luna': 'GPT-5.6 Luna'}
PANEL_LABEL = {'sonnet': 'A. Claude Sonnet 4.6', 'luna': 'B. GPT-5.6 Luna'}
AGENCIES = ['chatbot', 'self_repair', 'agent']
AGENCY_LABEL = dict(zip(AGENCIES, ['Chatbot', 'Self-repair', 'Constrained agent']))
CELL = ['task', 'model', 'platform', 'prompt', 'agency']
PROMPTS = ['zero_shot', 'few_shot']
PLATFORMS = ['python', 'r', 'stata']
PROMPT_LABEL = {'zero_shot': 'Zero-shot', 'few_shot': 'One-shot'}  # one-shot: Raul Sosa, manuscript commit fd96814
PLATFORM_LABEL = {'python': 'Python', 'r': 'R', 'stata': 'Stata'}
TASK_LABEL_SIZE = 8.6  # Figure 3 task labels, in points before the 0.93 text-width scaling
# The treatment of each panel carries the accent: the example, and Stata.
PROMPT_SERIES = {'zero_shot': style.SERIES['strong'], 'few_shot': style.SERIES['blush']}
PLATFORM_SERIES = {'python': style.SERIES['strong'], 'r': style.SERIES['mid'], 'stata': style.SERIES['blush']}
X = np.arange(3)
# Horizontal distance between neighboring series at one agency condition, in
# axis units, so that whiskers of close rates stay apart.
DODGE = .08
INTERVAL_CAPTION = ('Whiskers are 95 percent intervals from a cluster bootstrap over tasks '
                    f'({style.BOOTSTRAP_DRAWS:,} draws, seed {style.BOOTSTRAP_SEED}).')
# Figure 2 (failure mix) constants, kept under their earlier names.
FAILURE_BOOTSTRAP_DRAWS = style.BOOTSTRAP_DRAWS
FAILURE_BOOTSTRAP_SEED = style.BOOTSTRAP_SEED


def dodge(count):
    return (np.arange(count) - (count - 1) / 2) * DODGE


def bootstrap_columns(frame, tasks):
    return frame.assign(tasks=len(tasks), bootstrap_draws=style.BOOTSTRAP_DRAWS,
                        bootstrap_seed=style.BOOTSTRAP_SEED, interval=style.BOOTSTRAP_INTERVAL)


def rates(data, grouping):
    result = data.groupby(grouping).primary.agg(['mean', 'size', 'sum']).reset_index()
    result = result.rename(columns={'size': 'n', 'sum': 'successes'})
    result['success_pct'] = 100 * result.pop('mean')
    return result


def success_rates(data, grouping, tasks):
    """Task success by group with its task-bootstrap interval."""
    points = rates(data, grouping)
    boot = style.bootstrap_ratios(data, grouping, 'primary', tasks=tasks)
    merged = points.merge(boot[grouping + ['value', 'value_lo', 'value_hi']], on=grouping, validate='one_to_one')
    if not np.allclose(merged.value, merged.success_pct, rtol=0, atol=1e-9):
        raise AssertionError('bootstrap point estimate differs from the plotted rate')
    merged = merged.drop(columns='value').rename(columns={'value_lo': 'success_pct_lo', 'value_hi': 'success_pct_hi'})
    return bootstrap_columns(merged, tasks)


def failure_rates(data):
    rows = []
    for model, agency in itertools.product(MODELS, AGENCIES):
        group = data[data.model.eq(model) & data.agency.eq(agency)]
        failed = ~group.primary
        silent = failed & group.execution_ok & group.output_available
        loud = failed & ~silent
        rows.append(dict(model=model, agency=agency, n=len(group), failures=int(failed.sum()),
            loud=int(loud.sum()), silent=int(silent.sum()), loud_per100=100 * loud.mean(),
            silent_per100=100 * silent.mean(), outcome='primary',
            silent_share_of_failures_pct=100 * silent.sum() / failed.sum() if failed.sum() else np.nan))
    return pd.DataFrame(rows)


def pass_at_k_data(data):
    """Compute unbiased pass@k from all repetitions of a complete balanced design.

    The intervals resample tasks: each draw averages the per-cell estimates of
    the drawn tasks.
    """
    if data.empty or data[CELL + ['rep']].isna().any().any():
        raise ValueError('pass@k requires nonempty data and complete cell and repetition identifiers')
    if data.duplicated(CELL + ['rep']).any():
        raise ValueError('pass@k contains duplicate cell and repetition identifiers')
    for column, levels in [('model', MODELS), ('platform', PLATFORMS),
                           ('prompt', PROMPTS), ('agency', AGENCIES)]:
        if set(data[column]) != set(levels):
            raise ValueError(f'pass@k requires all supported {column} conditions')
    repetitions = data.rep.nunique()
    if set(data.rep) != set(range(repetitions)):
        raise ValueError('pass@k repetition identifiers must be contiguous integers starting at zero')
    if not data.primary.isin([True, False]).all():
        raise ValueError('pass@k requires a binary success outcome for every run')
    cells = data.groupby(CELL, observed=True).primary.agg(['sum', 'size']).reset_index()
    task_count = data.task.nunique()
    expected_cells = task_count * len(MODELS) * len(PLATFORMS) * len(PROMPTS) * len(AGENCIES)
    if len(cells) != expected_cells or not cells['size'].eq(repetitions).all():
        raise ValueError('pass@k requires a balanced design with every cell and the same repetitions in each cell')
    per_cell = []
    for k in range(1, repetitions + 1):
        values = [1 - (math.comb(int(n - c), k) / math.comb(int(n), k) if n - c >= k else 0)
                  for c, n in zip(cells['sum'], cells['size'])]
        per_cell.append(cells[CELL].assign(k=k, value=values))
    per_cell = pd.concat(per_cell, ignore_index=True)
    tasks = sorted(data.task.unique())
    boot = style.bootstrap_ratios(per_cell, ['model', 'agency', 'k'], 'value', tasks=tasks).set_index(['model', 'agency', 'k'])
    rows = []
    for model, agency, k in itertools.product(MODELS, AGENCIES, range(1, repetitions + 1)):
        group = per_cell[per_cell.model.eq(model) & per_cell.agency.eq(agency) & per_cell.k.eq(k)]
        interval = boot.loc[(model, agency, k)]
        rows.append(dict(model=model, agency=agency, k=k, pass_at_k_pct=100 * np.mean(group.value.tolist()),
                         cells=len(group), tasks=task_count, repetitions=repetitions,
                         pass_at_k_pct_lo=interval.value_lo, pass_at_k_pct_hi=interval.value_hi,
                         bootstrap_draws=style.BOOTSTRAP_DRAWS, bootstrap_seed=style.BOOTSTRAP_SEED,
                         interval=style.BOOTSTRAP_INTERVAL))
    return pd.DataFrame(rows), task_count


def agency_axis(ax):
    ax.set_xticks(X, [AGENCY_LABEL[a] for a in AGENCIES], fontsize=style.X_TICK_LABEL_SIZE)
    ax.set_xlim(-.35, 2.35)
    style.hairline_grid(ax)


def prompt_figure(data, out, source, save, by_model=False):
    """Figure 1: task success by agency, with prompt and language series.

    Panel A draws the prompt conditions and Panel B the programming languages,
    both models pooled. by_model=True draws the prompt conditions for each
    model instead (the figure that the text does not include).
    """
    tasks = sorted(data.task.unique())
    plotted = []
    with style.figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(style.FIG_W, 2.9), sharey=True)
        fig.subplots_adjust(left=.085, right=.985, top=.80, bottom=.10, wspace=.10)
        for column, ax in enumerate(axes):
            group = data[data.model.eq(MODELS[column])] if by_model else data
            dimension = 'prompt' if by_model or column == 0 else 'platform'
            values = success_rates(group, [dimension, 'agency'], tasks)
            values['panel'] = MODEL_LABEL[MODELS[column]] if by_model else dimension
            plotted.append(values)
            keys = PROMPTS if dimension == 'prompt' else PLATFORMS
            series = PROMPT_SERIES if dimension == 'prompt' else PLATFORM_SERIES
            names = PROMPT_LABEL if dimension == 'prompt' else PLATFORM_LABEL
            for key, offset in zip(keys, dodge(len(keys))):
                rows = values[values[dimension].eq(key)].set_index('agency').loc[AGENCIES]
                style.plot_series(ax, X + offset, rows.success_pct.to_numpy(), rows.success_pct_lo.to_numpy(),
                                  rows.success_pct_hi.to_numpy(), series[key], gid=key)
            if column == 0 or not by_model:
                style.legend_above(ax, [style.series_handle(series[k]) for k in keys], [names[k] for k in keys])
            title = PANEL_LABEL[MODELS[column]] if by_model else ['A. By prompt condition', 'B. By programming language'][column]
            style.panel_title(ax, title)
            agency_axis(ax)
        plotted = pd.concat(plotted, ignore_index=True)
        style.success_axis(axes[0], plotted.success_pct_lo)
        axes[1].tick_params(axis='y', labelleft=False)
        style.assert_text_outside(fig)
        if by_model:
            caption = 'Task success by prompt and agency condition, one panel per model, languages pooled. '
        else:
            caption = ('Task success by agency condition, both models pooled. Panel A draws the prompt conditions '
                       'and Panel B the programming languages. ')
        save(fig, out, 'prompt_agency_by_model' if by_model else 'prompt_agency', plotted, caption + INTERVAL_CAPTION, source)


def task_figure(data, out, source, save, labels):
    """Figure 3: task success by task and agency, both models pooled.

    One row per task and one marker per agency condition, with no interval
    (Federico, 2026-09-25: the figure shows the task ranking). The three
    markers of a task sit on one line, and a thin line joins the lowest and
    highest marker, as in the design before the restyle (Federico,
    2026-09-29). The success axis starts and ends EDGE_PAD_PT points beyond
    the lowest and highest marker, as in Figure 4, with ticks every ten points.
    """
    plotted = rates(data, ['task', 'agency'])
    chatbot = plotted[plotted.agency.eq('chatbot')].set_index('task').success_pct
    overall = data.groupby('task').primary.mean()
    tasks = sorted(chatbot.index, key=lambda t: (chatbot[t], overall[t], labels[t]))
    with style.figure_style():
        fig = plt.figure(figsize=(style.FIG_W, 6.6))
        # The task labels print at TASK_LABEL_SIZE (2026-10-02: the earlier 7.2 pt labels were
        # too small at print size). The frame starts right of the widest label.
        probes = [fig.text(0, 0, labels[t], fontsize=TASK_LABEL_SIZE) for t in tasks]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        widest = max(p.get_window_extent(renderer).width for p in probes) / fig.dpi
        for probe in probes:
            probe.remove()
        left = (widest + 8 / 72) / style.FIG_W
        frame = [left, .065, .978 - left, .875]
        ax = fig.add_axes(frame)
        left, right = padded_linear_limits(float(plotted.success_pct.min()), float(plotted.success_pct.max()),
                                           frame[2] * style.FIG_W * 72)
        ticks = [t for t in range(0, 101, 10) if left <= t <= right]
        ax.set_xlim(left, right)
        ax.set_ylim(len(tasks) - .5, -.5)
        for row, task in enumerate(tasks):
            if row % 2 == 0:
                ax.add_patch(Rectangle((left, row - .5), right - left, 1,
                                       facecolor=style.BAND, edgecolor='none', zorder=0))
            values = plotted[plotted.task.eq(task)].set_index('agency')
            xs = [float(values.loc[agency].success_pct) for agency in AGENCIES]
            ax.plot([min(xs), max(xs)], [row, row], color=style.MID, lw=.8, zorder=2,
                    solid_capstyle='butt', gid=f'{task}_range')
            for agency in AGENCIES:
                point = values.loc[agency]
                ax.plot([point.success_pct], [row], zorder=3, clip_on=False, ls='none', gid=f'{task}_{agency}',
                        **{**style.AGENCY_MARKERS[agency], 'ms': style.AGENCY_MARKERS[agency]['ms'] - .8})
        for tick in ticks:
            ax.axvline(tick, color=style.HAIR, lw=.5, zorder=1)
        ax.set_yticks(range(len(tasks)), [labels[t] for t in tasks], fontsize=TASK_LABEL_SIZE)
        ax.tick_params(axis='y', length=0, pad=4)
        ax.set_xticks(ticks)
        ax.set_xlabel('Task success (percent)', fontsize=style.AXIS_LABEL_SIZE, labelpad=3)
        ax.spines['left'].set_visible(False)
        style.legend_above(ax, [style.agency_handle(a) for a in AGENCIES], [AGENCY_LABEL[a] for a in AGENCIES],
                           markers_only=True)
        style.assert_text_outside(fig)
        save(fig, out, 'task_heterogeneity', plotted,
             'Task success by task and agency condition, pooling models, prompts, and languages. Tasks are ordered '
             'by chatbot success.', source)


def pooled_failure_rates(data, draws=style.BOOTSTRAP_DRAWS, seed=style.BOOTSTRAP_SEED):
    """Loud and silent failures by agency condition, both models pooled.

    A run fails when `primary` is false. A silent failure executes and returns
    every selected estimand. Every other failure is loud. The intervals are 95
    percent percentile intervals from a cluster bootstrap over tasks. The three
    agency conditions use the same task draws.
    """
    failed = ~data.primary.astype(bool)
    silent = failed & data.execution_ok.astype(bool) & data.output_available.astype(bool)
    frame = data.assign(runs=1, loud=(failed & ~silent).astype(int), silent=silent.astype(int))
    tasks = sorted(frame.task.unique())
    draw_index = style.task_draws(tasks, draws, seed)
    rows = []
    for agency in AGENCIES:
        counts = (frame[frame.agency.eq(agency)].groupby('task')[['runs', 'loud', 'silent']].sum()
                  .reindex(tasks, fill_value=0))
        runs, loud, silent_runs = (counts[column].to_numpy() for column in ('runs', 'loud', 'silent'))
        point = dict(runs=runs.sum(), loud=loud.sum(), silent=silent_runs.sum())
        boot = dict(runs=runs[draw_index].sum(axis=1), loud=loud[draw_index].sum(axis=1),
                    silent=silent_runs[draw_index].sum(axis=1))
        row = dict(agency=agency, models='+'.join(MODELS), n=int(point['runs']), tasks=len(tasks),
                   failures=int(point['loud'] + point['silent']), loud=int(point['loud']), silent=int(point['silent']))
        with np.errstate(divide='ignore', invalid='ignore'):
            for values, target in [(point, row), (boot, None)]:
                failures = values['loud'] + values['silent']
                stats = dict(loud_per100=100 * values['loud'] / values['runs'],
                             silent_per100=100 * values['silent'] / values['runs'],
                             failures_per100=100 * failures / values['runs'],
                             silent_share_of_failures_pct=np.where(failures > 0, 100 * values['silent'] / np.maximum(failures, 1), np.nan))
                if target is not None:
                    target.update({key: float(value) for key, value in stats.items()})
                    continue
                for key, value in stats.items():
                    row[f'{key}_lo'], row[f'{key}_hi'] = style.percentile_interval(value)
        row.update(bootstrap_draws=draws, bootstrap_seed=seed, interval=style.BOOTSTRAP_INTERVAL)
        rows.append(row)
    return pd.DataFrame(rows)


def failure_figure(data, out, source, save):
    """Figure 2: loud and silent failures by agency condition, both models pooled.

    Panel A plots loud and silent failures per 100 runs. Panel B plots the share
    of failures that are silent, with the number of failing runs under each
    tick. Whiskers are 95 percent task-bootstrap intervals with a fixed seed.
    The LaTeX caption carries the notes, so the figure draws no note text.
    Returns the rates for each model, which feed the appendix failure table and
    the macros. This figure defines the shared design in paper_figure_style.py.
    """
    plotted = pooled_failure_rates(data)
    rows = plotted.set_index('agency').loc[AGENCIES]
    x = np.arange(3)
    with style.figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(style.FIG_W, 2.6))
        fig.subplots_adjust(left=.085, right=.985, top=.78, bottom=.175, wspace=.30)
        ax = axes[0]
        for key, series in [('loud', style.SERIES['strong']), ('silent', style.SERIES['blush'])]:
            style.plot_series(ax, x, rows[f'{key}_per100'].to_numpy(), rows[f'{key}_per100_lo'].to_numpy(),
                              rows[f'{key}_per100_hi'].to_numpy(), series, gid=f'{key}_per100')
        top = max(25, 5 * math.ceil(rows[['loud_per100_hi', 'silent_per100_hi']].max().max() / 5))
        ax.set_ylim(0, top)
        ax.set_yticks(range(0, top + 1, 5))
        ax.set_ylabel('Failures per 100 runs', fontsize=style.AXIS_LABEL_SIZE)
        style.panel_title(ax, 'A. Loud and silent failures per 100 runs')
        style.legend_above(ax, [style.series_handle(style.SERIES['strong']), style.series_handle(style.SERIES['blush'])],
                           ['Loud', 'Silent'])
        ax.set_xticks(x, [AGENCY_LABEL[a] for a in AGENCIES], fontsize=style.X_TICK_LABEL_SIZE)
        ax = axes[1]
        style.plot_series(ax, x, rows.silent_share_of_failures_pct.to_numpy(), rows.silent_share_of_failures_pct_lo.to_numpy(),
                          rows.silent_share_of_failures_pct_hi.to_numpy(), style.SERIES['blush'], gid='silent_share_of_failures_pct')
        ax.set_ylim(0, 102)
        ax.set_yticks(range(0, 101, 25))
        ax.set_ylabel('Silent share of failures, percent', fontsize=style.AXIS_LABEL_SIZE)
        style.panel_title(ax, 'B. Share of failures that are silent')
        counts = [f'{count:,} failures' if count != 1 else '1 failure' for count in rows.failures]
        ax.set_xticks(x, [f'{AGENCY_LABEL[a]}\n{n}' for a, n in zip(AGENCIES, counts)],
                      fontsize=style.X_TICK_LABEL_SIZE, linespacing=1.4)
        for ax in axes:
            style.hairline_grid(ax)
            ax.set_xlim(-.35, 2.35)
        style.assert_text_outside(fig)
        save(fig, out, 'appendix_silent_vs_loud', plotted,
             'Loud and silent failures by agency condition, both models pooled. Panel A gives failures per 100 runs '
             'and Panel B the silent share of failures, with the number of failing runs under each tick. A silent '
             'failure follows clean execution and delivery of every selected estimand. Whiskers are 95 percent '
             f'intervals from a cluster bootstrap over tasks ({style.BOOTSTRAP_DRAWS:,} draws, seed {style.BOOTSTRAP_SEED}).', source)
    return failure_rates(data)


def pass_figure(data, out, source, save):
    plotted, task_count = pass_at_k_data(data)
    repetitions = int(plotted.repetitions.iloc[0])
    offsets = dict(zip(AGENCIES, (-.16, 0, .16)))
    with style.figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(style.FIG_W, 2.8), sharey=True)
        fig.subplots_adjust(left=.085, right=.985, top=.79, bottom=.16, wspace=.10)
        lower = min(70, 5 * math.floor(plotted.pass_at_k_pct_lo.min() / 5))
        for ax, model in zip(axes, MODELS):
            for agency in AGENCIES:
                rows = plotted[plotted.model.eq(model) & plotted.agency.eq(agency)]
                x = rows.k.to_numpy() + offsets[agency]
                gid = f'{model}_{agency}'
                ax.plot(x, rows.pass_at_k_pct, color=style.AGENCY_LINES[agency], lw=1.2, zorder=2, gid=gid)
                style.whiskers(ax, x, rows.pass_at_k_pct, rows.pass_at_k_pct_lo, rows.pass_at_k_pct_hi,
                               style.AGENCY_WHISKERS[agency], gid=f'{gid}_interval')
                marker = {**style.AGENCY_MARKERS[agency], 'ms': style.AGENCY_MARKERS[agency]['ms'] - .8}
                ax.plot(x, rows.pass_at_k_pct, ls='none', zorder=4, clip_on=False, **marker)
            ax.set_ylim(lower, 101)
            ax.set_yticks(range(lower, 101, 5))
            ax.set_xlim(.6, repetitions + .4)
            ax.set_xticks(range(1, repetitions + 1))
            style.hairline_grid(ax)
            style.panel_title(ax, PANEL_LABEL[model])
            ax.set_xlabel('Runs drawn (k)', fontsize=style.AXIS_LABEL_SIZE)
        axes[1].tick_params(axis='y', labelleft=False)
        axes[0].set_ylabel('pass@k (percent)', fontsize=style.AXIS_LABEL_SIZE)
        style.legend_above(axes[0], [style.agency_handle(a, line=True) for a in AGENCIES], [AGENCY_LABEL[a] for a in AGENCIES])
        style.assert_text_outside(fig)
        save(fig, out, 'appendix_pass_at_k', plotted,
             f'Unbiased pass@k from all {repetitions} repetitions in every cell of {task_count} tasks. Each cell has '
             'equal weight. Whiskers are 95 percent intervals from a cluster bootstrap over tasks '
             f'({style.BOOTSTRAP_DRAWS:,} draws, seed {style.BOOTSTRAP_SEED}).', source)
    return plotted, task_count


def runtime_figure(data, out, source, save):
    tasks = sorted(data.task.unique())
    plotted = data.groupby(['model', 'agency']).provider_seconds.agg(['median', 'count', 'size']).reset_index()
    plotted.columns = ['model', 'agency', 'median_provider_seconds', 'n_observed', 'n_total']
    boot = style.bootstrap_medians(data, ['model', 'agency'], 'provider_seconds', tasks=tasks)
    plotted = plotted.merge(boot[['model', 'agency', 'value_lo', 'value_hi']].rename(
        columns={'value_lo': 'median_provider_seconds_lo', 'value_hi': 'median_provider_seconds_hi'}),
        on=['model', 'agency'], validate='one_to_one')
    plotted = bootstrap_columns(plotted, tasks)
    with style.figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(style.FIG_W, 2.4), sharey=True)
        fig.subplots_adjust(left=.1, right=.985, top=.87, bottom=.12, wspace=.10)
        top = 10 * math.ceil(plotted.median_provider_seconds_hi.max() * 1.05 / 10)
        for ax, model in zip(axes, MODELS):
            rows = plotted[plotted.model.eq(model)].set_index('agency').loc[AGENCIES]
            style.plot_series(ax, X, rows.median_provider_seconds.to_numpy(), rows.median_provider_seconds_lo.to_numpy(),
                              rows.median_provider_seconds_hi.to_numpy(), style.SERIES['strong'], gid=f'{model}_median')
            ax.set_ylim(0, top)
            # Ticks every ten seconds, so the axis ends on a tick, as in Figure 2.
            ax.set_yticks(range(0, top + 1, 10))
            agency_axis(ax)
            style.panel_title(ax, PANEL_LABEL[model], pad=6)
        axes[1].tick_params(axis='y', labelleft=False)
        axes[0].set_ylabel('Median provider\nseconds per run', fontsize=style.AXIS_LABEL_SIZE, linespacing=1.3)
        style.assert_text_outside(fig)
        save(fig, out, 'runtime_by_agency', plotted,
             'Median provider wall time per run, including all calls and execution inside agent sessions. External '
             'harness execution and queue waits are excluded. Whiskers are 95 percent intervals for the median from '
             f'a cluster bootstrap over tasks ({style.BOOTSTRAP_DRAWS:,} draws, seed {style.BOOTSTRAP_SEED}).', source)


# The design figure is a self-contained TikZ template in the Figure 2 style
# (code/exhibits/final_fig_design.tex). The template carries the counts of the
# sample as macros. The build replaces the macro values from the data and
# checks that the template defines the paper figure colors.
DESIGN_TEMPLATE = EXHIBITS / 'final_fig_design.tex'
DESIGN_COLORS = {'ink': style.INK, 'strong': style.STRONG, 'hair': style.HAIR, 'blush': style.BLUSH,
                 'walnut': style.WALNUT, 'mid': style.MID, 'band': style.BAND}
# Programming languages, prompt conditions, and agency conditions of one task.
CONDITIONS_PER_TASK = len(PLATFORMS) * len(PROMPTS) * len(AGENCIES)


def latex_count(value):
    return f'{int(value):,}'.replace(',', '{,}')


def count_range(counts):
    """One number when every count agrees, else the lowest and highest as a range."""
    minimum, maximum = int(min(counts)), int(max(counts))
    return latex_count(minimum) if minimum == maximum else f'{latex_count(minimum)}--{latex_count(maximum)}'


def design_counts(data, out):
    """Macro values of the design template, from the sample and the analysis folder.

    The run records give the tasks, the cells, the repetitions per cell, and
    the run counts. The datasets come from `data/catalog.json` in the analysis
    folder (distinct input paths of the selected tasks) and the standard-error
    tasks from `se_scoring.tasks` in `data/snapshot.json`. A missing file keeps
    the template's value for its counts.
    """
    tasks = sorted(data.task.unique())
    cells = data[['task', 'platform', 'prompt', 'agency']].drop_duplicates()
    counts = dict(ntasks=latex_count(len(tasks)), ncells=latex_count(len(cells)),
                  nreps=count_range(data.groupby(CELL).size()), nruns=latex_count(len(data)),
                  nrunspermodel=count_range(data.groupby('model').size()))
    catalog_path = out / 'data/catalog.json'
    if catalog_path.is_file():
        catalog = json.loads(catalog_path.read_text())
        by_path = {}
        for task in tasks:
            by_path.setdefault(catalog[task]['data_path'], []).append(task)
        shared = [group for group in by_path.values() if len(group) > 1]
        counts.update(ndatasets=latex_count(len(by_path)), nshareddatasets=latex_count(len(shared)),
                      nsharedtasks=latex_count(sum(len(group) for group in shared)))
    snapshot_path = out / 'data/snapshot.json'
    if snapshot_path.is_file():
        se_tasks = json.loads(snapshot_path.read_text()).get('se_scoring', {}).get('tasks')
        if se_tasks is not None:
            counts['nsetasks'] = latex_count(len(set(se_tasks) & set(tasks)))
    return counts


def design_source(data, out):
    """The template with the sample's counts in its macros."""
    text = DESIGN_TEMPLATE.read_text()
    for name, color in DESIGN_COLORS.items():
        if f'\\definecolor{{{name}}}{{HTML}}{{{color[1:]}}}' not in text:
            raise ValueError(f'Design template does not define {name} as {color}')
    for name, value in design_counts(data, out).items():
        pattern = r'\\newcommand\{\\' + name + r'\}\{(?:[^{}]|\{[^{}]*\})*\}'
        text, replaced = re.subn(pattern, lambda match: f'\\newcommand{{\\{name}}}{{{value}}}', text)
        if replaced != 1:
            raise ValueError(f'Design template lacks the macro \\{name}')
    return text


def design_figure(data, out, source, save):
    """Compile the TikZ design template with the counts of the sample."""
    text = design_source(data, out)
    counts = design_counts(data, out)
    tasks = data.task.nunique()
    per_cell = data.groupby(CELL).size()
    plotted = pd.DataFrame([dict(tasks=tasks, models=2, languages=3, prompts=2, agencies=3,
        cells=tasks * CONDITIONS_PER_TASK, target_reps=10, completed=len(data),
        planned=tasks * CONDITIONS_PER_TASK * 2 * 10, minimum_reps=int(per_cell.min()), maximum_reps=int(per_cell.max()),
        datasets=counts.get('ndatasets'), shared_datasets=counts.get('nshareddatasets'),
        shared_tasks=counts.get('nsharedtasks'), se_tasks=counts.get('nsetasks'))])
    with tempfile.TemporaryDirectory(prefix='paper-design-') as temporary:
        folder = Path(temporary)
        (folder / 'benchmark_design.tex').write_text(text)
        result = subprocess.run(['pdflatex', '-halt-on-error', '-interaction=nonstopmode', 'benchmark_design.tex'],
                                cwd=folder, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout[-4000:])
        save(folder / 'benchmark_design.pdf', out, 'benchmark_design', plotted,
             f'Benchmark design for {tasks} tasks, with all 36 model-language-prompt-agency cells represented.', source)


def render_paper_figures(data, out, source, save, labels):
    """Write the paper figures and the per-model prompt figure."""
    prompt_figure(data, out, source, save)
    prompt_figure(data, out, source, save, by_model=True)
    task_figure(data, out, source, save, labels)
    failures = failure_figure(data, out, source, save)
    passes, task_count = pass_figure(data, out, source, save)
    runtime_figure(data, out, source, save)
    design_figure(data, out, source, save)
    return failures, passes, task_count

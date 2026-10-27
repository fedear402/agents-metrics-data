"""Check the shared figure style, and every data figure's points and intervals against runs.csv.

The runs.csv checks recompute each plotted value with pandas and each interval
with an independent implementation of the task cluster bootstrap: the draws
become task multiplicities (np.bincount), and medians come from explicitly
resampled values. They then compare the result with the drawn lines, markers,
and whisker ends, and with the installed figure data under data/analysis/main.
The task figure and the two cost figures draw points only (Federico,
2026-09-25), so their checks also confirm that no interval is drawn.
"""
import itertools
import math
import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import paper_figure_style as style
import production_paper_assets as assets
import production_paper_costs as costs
import production_paper_figures as figures

MAIN_DATA = ROOT / 'data/analysis/main/data'
DRAWS, SEED = 4000, 20260925
requires_main = pytest.mark.skipif(not (MAIN_DATA / 'runs.csv').is_file(), reason='data/analysis/main is not present')


# Shared style ---------------------------------------------------------------

def test_style_constants_match_the_figure2_method():
    assert (style.BOOTSTRAP_DRAWS, style.BOOTSTRAP_SEED) == (DRAWS, SEED)
    assert (style.INK, style.WALNUT, style.BLUSH) == ('#021111', '#402402', '#FFAAA7')
    assert style.STYLE['font.sans-serif'][:2] == ['Helvetica Neue', 'Helvetica']
    assert style.STYLE['pdf.fonttype'] == 42


def test_figures_save_with_a_fixed_pdf_date(tmp_path):
    """Importing the style fixes the PDF date, so rebuilt figures are byte for byte reproducible."""
    script = ("import sys; sys.path.insert(0, sys.argv[1]); import paper_figure_style as style\n"
              "import matplotlib.pyplot as plt\n"
              "with style.figure_style():\n"
              "    fig, ax = plt.subplots(); ax.plot([0, 1]); fig.savefig(sys.argv[2])\n")
    environment = {k: v for k, v in os.environ.items() if k != 'SOURCE_DATE_EPOCH'}
    subprocess.run([sys.executable, '-c', script, str(ROOT / 'code/benchmark'), str(tmp_path / 'check.pdf')],
                   env=environment, check=True)
    assert b'/CreationDate (D:19700101000000Z)' in (tmp_path / 'check.pdf').read_bytes()


def small_panel():
    data = pd.DataFrame([dict(task=f't{t}', group=g, primary=(t + r + g) % 3 != 0, seconds=float(t * 10 + r + g))
                         for t, g, r in itertools.product(range(5), range(2), range(4))])
    return data


def brute_force(data, statistic, draws=50, seed=7):
    """Resample whole tasks by building each drawn data set row by row."""
    tasks = sorted(data.task.unique())
    index = np.random.default_rng(seed).integers(0, len(tasks), size=(draws, len(tasks)))
    values = []
    for row in index:
        drawn = pd.concat([data[data.task.eq(tasks[i])] for i in row], ignore_index=True)
        values.append(statistic(drawn))
    return np.percentile(values, [2.5, 97.5])


def test_bootstrap_ratios_equal_brute_force_task_resampling():
    data = small_panel()
    result = style.bootstrap_ratios(data, 'group', 'primary', draws=50, seed=7).set_index('group')
    for group in (0, 1):
        expected = brute_force(data, lambda d: 100 * d[d.group.eq(group)].primary.mean())
        assert [result.loc[group, 'value_lo'], result.loc[group, 'value_hi']] == pytest.approx(expected)
        assert result.loc[group, 'value'] == pytest.approx(100 * data[data.group.eq(group)].primary.mean())


def test_bootstrap_medians_equal_brute_force_task_resampling():
    data = small_panel()
    result = style.bootstrap_medians(data, 'group', 'seconds', draws=50, seed=7).set_index('group')
    for group in (0, 1):
        expected = brute_force(data, lambda d: d[d.group.eq(group)].seconds.median())
        assert [result.loc[group, 'value_lo'], result.loc[group, 'value_hi']] == pytest.approx(expected)


def test_label_stack_moves_inside_bounds_and_keeps_its_gaps():
    heights = style.stacked_label_y([100, 99.5, 99], gap=3, bounds=(40, 99.5))
    assert max(heights) == pytest.approx(99.5)
    ordered = sorted(heights, reverse=True)
    assert all(a - b >= 3 - 1e-9 for a, b in zip(ordered, ordered[1:]))
    assert style.stacked_label_y([50, 60], gap=1) == [50, 60]
    with pytest.raises(ValueError, match='do not fit'):
        style.stacked_label_y([1, 2, 3], gap=5, bounds=(0, 4))


def test_text_check_rejects_overlap_and_text_off_the_canvas():
    with style.figure_style():
        fig, ax = plt.subplots(figsize=(3, 2))
        fig.subplots_adjust(top=.7)
        ax.set_title('A. Panel', loc='left')
        style.assert_text_outside(fig)
        fig.text(.1, .85, 'first label')
        fig.text(.12, .85, 'second label')
        with pytest.raises(AssertionError, match='texts overlap'):
            style.assert_text_outside(fig)
        fig.texts.clear()
        fig.text(.98, .85, 'runs off the right edge')
        with pytest.raises(AssertionError, match='outside the figure'):
            style.assert_text_outside(fig)
        plt.close(fig)


# runs.csv checks ------------------------------------------------------------

@pytest.fixture(scope='module')
def runs():
    data = pd.read_csv(MAIN_DATA / 'runs.csv')
    assert data.task.nunique() == 40 and len(data) == 14400
    return data


@pytest.fixture(scope='module')
def multiplicity():
    """Times each of the 40 sorted tasks enters each draw, from the shared seed."""
    index = np.random.default_rng(SEED).integers(0, 40, size=(DRAWS, 40))
    return np.stack([np.bincount(row, minlength=40) for row in index])


def boot_ratio(frame, tasks, multiplicity, numerator, denominator=None, scale=100):
    """Ratio of sums over resampled tasks, from task totals and draw multiplicities."""
    totals = frame.groupby('task').agg(num=(numerator, 'sum'), den=(numerator, 'size') if denominator is None
                                       else (denominator, 'sum')).reindex(tasks, fill_value=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        values = scale * (multiplicity @ totals.num.to_numpy(float)) / (multiplicity @ totals.den.to_numpy(float))
    # A draw without any failure has no silent share; it leaves the interval.
    return np.nanpercentile(values, [2.5, 97.5])


def boot_median(frame, column, tasks, multiplicity):
    by_task = {task: frame.loc[frame.task.eq(task), column].dropna().to_numpy(float) for task in tasks}
    values = [np.median(np.concatenate([np.repeat(by_task[t], m) for t, m in zip(tasks, row) if m]))
              for row in multiplicity]
    return np.percentile(values, [2.5, 97.5])


def capture(render, *args, **kwargs):
    captured = {}

    def save(fig, destination, stem, plotted, caption, source):
        captured[stem] = (fig, plotted, caption)

    render(*args, save, **kwargs)
    return captured


def line_y(ax, gid):
    lines = [line for line in ax.lines if line.get_gid() == gid]
    assert len(lines) == 1, gid
    return lines[0].get_ydata()


def vertical_whiskers(ax, gid):
    """Low and high ends of vertical whiskers, one row per whisker."""
    segments = style.interval_segments(ax, gid)
    assert np.allclose(segments[:, 0], segments[:, 2])
    return segments[:, [1, 3]]


def assert_installed(stem, plotted):
    """The installed figure data equal this recomputation (after a rebuild)."""
    installed = pd.read_csv(MAIN_DATA / f'{stem}.csv')
    pd.testing.assert_frame_equal(installed, plotted.reset_index(drop=True), check_dtype=False, atol=1e-9)


@requires_main
@pytest.mark.parametrize('by_model', [False, True])
def test_prompt_figures_match_runs_csv(tmp_path, runs, multiplicity, by_model):
    tasks = sorted(runs.task.unique())
    stem = 'prompt_agency_by_model' if by_model else 'prompt_agency'
    fig, plotted, caption = capture(figures.prompt_figure, runs, tmp_path, tmp_path / 'source.csv', by_model=by_model)[stem]
    panels = [('prompt', runs[runs.model.eq(m)]) for m in figures.MODELS] if by_model else [('prompt', runs), ('platform', runs)]
    for ax, (dimension, group) in zip(fig.axes, panels):
        for key in (figures.PROMPTS if dimension == 'prompt' else figures.PLATFORMS):
            rows = [group[group[dimension].eq(key) & group.agency.eq(a)] for a in figures.AGENCIES]
            assert line_y(ax, key) == pytest.approx([100 * r.primary.mean() for r in rows], abs=1e-12)
            expected = [boot_ratio(r, tasks, multiplicity, 'primary') for r in rows]
            assert vertical_whiskers(ax, f'{key}_interval') == pytest.approx(np.array(expected), abs=1e-9)
    assert (plotted.success_pct_lo < plotted.success_pct).all() and (plotted.success_pct <= plotted.success_pct_hi).all()
    assert_installed(stem, plotted)
    plt.close(fig)


@requires_main
def test_task_figure_matches_runs_csv(tmp_path, runs):
    fig, plotted, caption = capture(figures.task_figure, runs, tmp_path, tmp_path / 'source.csv',
                                    labels=assets.SHORT_LABELS)['task_heterogeneity']
    ax = fig.axes[0]
    for (task, agency), group in runs.groupby(['task', 'agency']):
        point = [line for line in ax.lines if line.get_gid() == f'{task}_{agency}']
        assert len(point) == 1
        assert point[0].get_xdata()[0] == pytest.approx(100 * group.primary.mean(), abs=1e-12)
    # One line per task joins its lowest and highest agency rate, on the row of its markers.
    for task, group in runs.groupby('task'):
        span = [line for line in ax.lines if line.get_gid() == f'{task}_range']
        assert len(span) == 1
        rates_by_agency = 100 * group.groupby('agency').primary.mean()
        assert list(span[0].get_xdata()) == pytest.approx([rates_by_agency.min(), rates_by_agency.max()], abs=1e-12)
        markers = [line for line in ax.lines if str(line.get_gid()).startswith(f'{task}_') and line is not span[0]]
        assert {line.get_ydata()[0] for line in markers} == set(span[0].get_ydata())
    # Points only: no interval line, no interval column, no interval sentence.
    assert not [line for line in ax.lines if str(line.get_gid()).endswith('_interval')]
    assert not ax.collections
    assert not {'success_pct_lo', 'success_pct_hi', 'interval'} & set(plotted)
    assert 'interval' not in caption.lower() and 'whisker' not in caption.lower()
    assert len(plotted) == 120 and plotted.n.eq(120).all()
    # The success axis spans the points with the Figure 4 edge pad (no empty band below the lowest rate).
    left, right = ax.get_xlim()
    width_pt = ax.get_position().width * fig.get_figwidth() * 72
    low, high = plotted.success_pct.min(), plotted.success_pct.max()
    assert ((low - left) * width_pt / (right - left), (right - high) * width_pt / (right - left)) == pytest.approx(
        (figures.EDGE_PAD_PT, figures.EDGE_PAD_PT), abs=.05)
    assert_installed('task_heterogeneity', plotted)
    plt.close(fig)


@requires_main
def test_failure_figure_intervals_match_runs_csv(tmp_path, runs, multiplicity):
    tasks = sorted(runs.task.unique())
    fig, plotted, caption = capture(figures.failure_figure, runs, tmp_path, tmp_path / 'source.csv')['appendix_silent_vs_loud']
    frame = runs.assign(loud=runs.failure_category.eq('Loud failure').astype(int),
                        silent=runs.failure_category.eq('Silent failure').astype(int))
    frame['failed'] = frame.loud + frame.silent
    panel_a, panel_b = fig.axes
    for gid, ax, numerator, denominator in [('loud_per100', panel_a, 'loud', None), ('silent_per100', panel_a, 'silent', None),
                                            ('silent_share_of_failures_pct', panel_b, 'silent', 'failed')]:
        expected = [boot_ratio(frame[frame.agency.eq(a)], tasks, multiplicity, numerator, denominator) for a in figures.AGENCIES]
        assert vertical_whiskers(ax, f'{gid}_interval') == pytest.approx(np.array(expected), abs=1e-9)
    plt.close(fig)


@requires_main
def test_pass_at_k_figure_matches_runs_csv(tmp_path, runs, multiplicity):
    tasks = sorted(runs.task.unique())
    fig, plotted, caption = capture(figures.pass_figure, runs, tmp_path, tmp_path / 'source.csv')['appendix_pass_at_k']
    cells = runs.groupby(figures.CELL).primary.agg(['sum', 'size']).reset_index()
    assert cells['size'].eq(10).all()
    for ax, model in zip(fig.axes, figures.MODELS):
        for agency in figures.AGENCIES:
            group = cells[cells.model.eq(model) & cells.agency.eq(agency)]
            means, intervals = [], []
            for k in range(1, 11):
                value = group.assign(v=[1 - math.comb(n - c, k) / math.comb(n, k) for c, n in zip(group['sum'], group['size'])])
                means.append(100 * value.v.mean())
                intervals.append(boot_ratio(value, tasks, multiplicity, 'v'))
            assert line_y(ax, f'{model}_{agency}') == pytest.approx(means, abs=1e-12)
            assert vertical_whiskers(ax, f'{model}_{agency}_interval') == pytest.approx(np.array(intervals), abs=1e-9)
    assert_installed('appendix_pass_at_k', plotted)
    plt.close(fig)


@requires_main
def test_runtime_figure_matches_runs_csv(tmp_path, runs, multiplicity):
    tasks = sorted(runs.task.unique())
    fig, plotted, caption = capture(figures.runtime_figure, runs, tmp_path, tmp_path / 'source.csv')['runtime_by_agency']
    for ax, model in zip(fig.axes, figures.MODELS):
        rows = [runs[runs.model.eq(model) & runs.agency.eq(a)] for a in figures.AGENCIES]
        assert line_y(ax, f'{model}_median') == pytest.approx([r.provider_seconds.median() for r in rows], abs=1e-12)
        expected = [boot_median(r, 'provider_seconds', tasks, multiplicity) for r in rows]
        assert vertical_whiskers(ax, f'{model}_median_interval') == pytest.approx(np.array(expected), abs=1e-9)
    assert 'median' in caption and 'seed 20260925' in caption
    assert_installed('runtime_by_agency', plotted)
    plt.close(fig)


@requires_main
@pytest.mark.parametrize('stem, resource, scale', [('cost_of_correctness', 'cost_usd', 100),
                                                   ('cost_tokens', 'total_main_tokens', 1 / 1000)])
def test_cost_figures_match_runs_csv(runs, stem, resource, scale):
    prices = pd.read_csv(MAIN_DATA / 'run_pricing.csv')
    # runs.csv carries no cost column; the paper cost is in run_pricing.csv.
    data = runs.drop(columns='cost_usd', errors='ignore').merge(prices[['record_path', 'record_sha256', 'cost_usd']],
                                               on=['record_path', 'record_sha256'], validate='one_to_one')
    fig, points = costs.build_resource_figure(data, resource)
    observed = data[data[resource].notna()]
    for ax, model in zip(fig.axes, costs.MODELS):
        for platform, agency in itertools.product(costs.PLATFORMS, costs.AGENCIES):
            group = observed[observed.model.eq(model) & observed.platform.eq(platform) & observed.agency.eq(agency)]
            gid = f'{model}_{platform}_{agency}'
            marker = [line for line in ax.lines if line.get_gid() == f'{gid}_point']
            assert len(marker) == 1
            assert marker[0].get_xdata()[0] == pytest.approx(scale * group[resource].mean(), abs=1e-12)
            assert marker[0].get_ydata()[0] == pytest.approx(100 * group.primary.mean(), abs=1e-12)
        # Points only: errorbar would add line collections.
        assert not ax.collections
        # The success axis spans the points with a margin of a few points, with no blank band below.
        low, high = ax.get_ylim()
        assert points.success_pct.min() - 3 < low < points.success_pct.min()
        assert points.success_pct.max() < high < points.success_pct.max() + 3
    assert not {'success_pct_lo', 'success_pct_hi', 'x_value_lo', 'x_value_hi'} & set(points)
    assert_installed(stem, points)
    plt.close(fig)

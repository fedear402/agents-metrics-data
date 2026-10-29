"""Shared style and interval method for the paper figures.

Figure 2 (the failure mix, ``failure_figure`` in production_paper_figures.py)
set the design on 2026-09-25: Helvetica, a dark neutral (INK text, STRONG
axes and series), Powder Blush with a Deep Walnut edge for the highlighted
series, left-aligned panel titles such as "A. ...", a legend above the plot, a
hairline grid, and no text inside any plot frame. Every data figure takes its
style, its interval code, and its text check from this module.

Intervals are 95 percent percentile intervals from a cluster bootstrap over
tasks. Each draw resamples the tasks with replacement. Every group of a figure
uses the same draws, and every figure uses the same seed and draw count.
The task figure (Figure 3) and the cost figures draw no interval
(Federico, 2026-09-25).

The module defaults draw the AEA figures. The six-model PNAS build
(code/paper/six_models.py, pnas_style) sets FIG_W to PNAS_FIG_W, STYLE_OVERRIDES
to PNAS_FONT, and the agent line to ROSE, and it checks the text with
keep_side_margins and text_sizes (peer review 2026-10-28).
"""
from __future__ import annotations

from contextlib import contextmanager
import itertools
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd

# Fixed PDF dates make every rebuild write byte-identical figure files. Matplotlib
# and pdflatex (the design figure) both read this variable.
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')

# Palette of the design that Federico chose on 2026-09-25 (option F).
INK, WALNUT, BLUSH = '#021111', '#402402', '#FFAAA7'
HAIR, STRONG = '#D1C5B9', '#3A4040'
# A third neutral for a third series, and the band behind alternate task rows.
MID = '#8C9292'
BAND = '#F6F3F0'
# A darker pink for lines in the PNAS figures. BLUSH has a contrast ratio of 1.82:1 against white, and ROSE
# has 3.9:1, above the 3:1 that WCAG 2.1 asks for graphical objects (peer review 2026-10-28, I73).
# Marker fills keep BLUSH, because their WALNUT edges carry the contrast.
ROSE = '#CC5C74'

# Figure width in inches. The AEA figures use 6.5 in. The six-model PNAS build sets PNAS_FIG_W, the PNAS
# two-column width of 17.8 cm, so the text keeps its drawn point size in print (peer review 2026-10-28, I76).
FIG_W = 6.5
PNAS_FIG_W = 17.8 / 2.54
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_INTERVAL = '95 percent percentile, task cluster bootstrap'

STYLE = {
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial'],
    'mathtext.fontset': 'dejavusans', 'font.size': 8, 'text.color': INK,
    'axes.edgecolor': STRONG, 'axes.linewidth': .6, 'axes.labelcolor': INK,
    'axes.titlesize': 8.5, 'axes.titlecolor': INK, 'axes.spines.top': False,
    'axes.spines.right': False, 'xtick.color': STRONG, 'ytick.color': STRONG,
    'xtick.labelcolor': INK, 'ytick.labelcolor': INK, 'xtick.direction': 'out',
    'ytick.direction': 'out', 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.frameon': False, 'legend.fontsize': 7.4, 'pdf.fonttype': 42, 'axes.grid': False,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'savefig.bbox': 'standard'}

# Settings that figure_style() applies after STYLE. The AEA figures leave it empty. The six-model PNAS build
# sets PNAS_FONT, because PNAS asks for one font in all figures, Arial or Helvetica (peer review 2026-10-28,
# I76). The design figure uses the same Helvetica through fontspec. The math font follows the text font, so a
# label such as "Pass@$k$" sets k in Helvetica Oblique.
STYLE_OVERRIDES = {}
PNAS_FONT = {
    'font.sans-serif': ['Helvetica', 'Arial', 'TeX Gyre Heros', 'Liberation Sans', 'DejaVu Sans'],
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Helvetica', 'mathtext.sf': 'Helvetica',
    'mathtext.it': 'Helvetica:italic', 'mathtext.bf': 'Helvetica:bold'}

# Sizes of Figure 2, in points.
LINE_WIDTH = 1.5
MARKER_SIZE = 4.6
MARKER_EDGE = .7
WHISKER = dict(elinewidth=.8, capsize=2.2, capthick=.8)
TITLE_SIZE = 8.2
TITLE_PAD = 22
AXIS_LABEL_SIZE = 8
X_TICK_LABEL_SIZE = 7.3

# Series styles. `strong` and `blush` are the loud and silent series of Figure 2.
# `key_size` and `key_edge` draw the legend key, as in Figure 2.
SERIES = {
    'strong': dict(line=STRONG, marker='o', face=STRONG, edge=STRONG, whisker=STRONG, size=MARKER_SIZE,
                   key_size=4.4, key_edge=0),
    'blush': dict(line=BLUSH, marker='s', face=BLUSH, edge=WALNUT, whisker=WALNUT, size=MARKER_SIZE,
                  key_size=4.2, key_edge=MARKER_EDGE),
    'mid': dict(line=MID, marker='^', face=MID, edge=MID, whisker=MID, size=5.2, key_size=5.0, key_edge=0),
    # A third series with no accent (PNAS Figure 2): a dashed dark line and an open inverted triangle. Its
    # shape differs from the agency markers (circle, square, diamond), so it carries no agency meaning.
    'open': dict(line=STRONG, ls=(0, (3.2, 1.6)), marker='v', face='white', edge=STRONG, whisker=STRONG,
                 size=5.0, key_size=4.8, key_edge=MARKER_EDGE),
}

# One marker per agency condition. Shape and fill both differ, so the key does
# not rest on color alone. The constrained agent carries the accent.
AGENCY_MARKERS = {
    'chatbot': dict(marker='o', mfc='white', mec=STRONG, mew=.8, ms=MARKER_SIZE),
    'self_repair': dict(marker='s', mfc=MID, mec=STRONG, mew=.7, ms=4.2),
    'agent': dict(marker='D', mfc=BLUSH, mec=WALNUT, mew=.7, ms=4.0),
}
AGENCY_LINES = {'chatbot': STRONG, 'self_repair': MID, 'agent': BLUSH}
AGENCY_WHISKERS = {'chatbot': STRONG, 'self_repair': STRONG, 'agent': WALNUT}


@contextmanager
def figure_style():
    """Draw and save inside the Figure 2 style, from Matplotlib defaults.

    Starting from the defaults keeps a figure independent of any style that an
    earlier figure in the same process set.
    """
    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update(STYLE)
        plt.rcParams.update(STYLE_OVERRIDES)
        yield


def task_draws(tasks, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Task positions for each bootstrap draw, one row per draw."""
    return np.random.default_rng(seed).integers(0, len(tasks), size=(draws, len(tasks)))


def percentile_interval(values):
    """The 2.5 and 97.5 percentiles of the finite values (missing if none)."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return math.nan, math.nan
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def _groups(data, by):
    by = [by] if isinstance(by, str) else list(by)
    for key, group in data.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        yield dict(zip(by, key)), group


def bootstrap_ratios(data, by, numerator, denominator=None, scale=100.0, tasks=None,
                     draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Ratio of sums for each group, with a task cluster bootstrap interval.

    The statistic is scale * sum(numerator) / sum(denominator) over the rows of
    a group; with no denominator it divides by the number of rows, which gives
    a mean. Each draw resamples the tasks with replacement and recomputes every
    group's ratio from the totals of the drawn tasks. All groups share the
    draws. Returns one row per group with `value`, `value_lo`, `value_hi`,
    `rows`, and `tasks`.
    """
    if data[numerator].isna().any():
        raise ValueError(f'{numerator} has missing values; select the observed rows first')
    tasks = sorted(data.task.unique()) if tasks is None else list(tasks)
    index = task_draws(tasks, draws, seed)
    frame = data.assign(_num=data[numerator].astype(float),
                        _den=1.0 if denominator is None else data[denominator].astype(float))
    rows = []
    for key, group in _groups(frame, by):
        totals = group.groupby('task')[['_num', '_den']].sum().reindex(tasks, fill_value=0)
        num, den = totals['_num'].to_numpy(), totals['_den'].to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            boot = scale * num[index].sum(axis=1) / den[index].sum(axis=1)
        low, high = percentile_interval(boot)
        rows.append(dict(key, value=scale * num.sum() / den.sum(), value_lo=low, value_hi=high,
                         rows=len(group), tasks=len(tasks)))
    return pd.DataFrame(rows)


def _expanded_median(sorted_values, cumulative):
    """Median of a sample that repeats sorted_values[j] (cumulative[j] - cumulative[j-1]) times."""
    total = int(cumulative[-1])
    low = sorted_values[np.searchsorted(cumulative, (total - 1) // 2, side='right')]
    high = sorted_values[np.searchsorted(cumulative, total // 2, side='right')]
    return (low + high) / 2


def bootstrap_medians(data, by, column, tasks=None, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Median of `column` for each group, with a task cluster bootstrap interval.

    Rows with a missing value are left out. A draw repeats every row of a drawn
    task once for each time the draw selects that task, and takes the median of
    the repeated sample. All groups share the draws.
    """
    tasks = sorted(data.task.unique()) if tasks is None else list(tasks)
    position = {task: i for i, task in enumerate(tasks)}
    index = task_draws(tasks, draws, seed)
    counts = np.zeros((len(index), len(tasks)), dtype=np.int64)
    for column_index in range(index.shape[1]):
        np.add.at(counts, (np.arange(len(index)), index[:, column_index]), 1)
    rows = []
    for key, group in _groups(data[data[column].notna()], by):
        order = np.argsort(group[column].to_numpy(dtype=float), kind='stable')
        values = group[column].to_numpy(dtype=float)[order]
        owner = group.task.map(position).to_numpy()[order]
        boot = np.empty(len(index))
        for draw in range(len(index)):
            cumulative = np.cumsum(counts[draw, owner])
            boot[draw] = _expanded_median(values, cumulative) if cumulative[-1] else math.nan
        low, high = percentile_interval(boot)
        rows.append(dict(key, value=float(np.median(values)), value_lo=low, value_hi=high,
                         rows=len(group), tasks=len(tasks)))
    return pd.DataFrame(rows)


def stacked_label_y(ends, gap, bounds=None):
    """Label heights that keep a minimum gap and stay centered on the line ends.

    With bounds=(low, high), the whole stack then moves, keeping its gaps, so
    that every label height lies inside [low, high]. The stack must fit.
    """
    order = sorted(range(len(ends)), key=lambda k: ends[k], reverse=True)
    placed, previous = {}, None
    for k in order:
        y = ends[k] if previous is None else min(ends[k], previous - gap)
        placed[k] = y
        previous = y
    shift = (sum(ends) - sum(placed.values())) / len(ends)
    heights = [placed[k] + shift for k in range(len(ends))]
    if bounds is not None:
        low, high = bounds
        if max(heights) - min(heights) > high - low + 1e-9:
            raise ValueError('The labels do not fit inside the bounds with the required gap')
        move = min(0.0, high - max(heights)) + max(0.0, low - min(heights))
        heights = [y + move for y in heights]
    return heights


def panel_title(ax, text, pad=TITLE_PAD):
    """Left-aligned panel title, high enough by default to leave room for a legend."""
    ax.set_title(text, fontsize=TITLE_SIZE, loc='left', pad=pad)


def legend_above(ax, handles, labels, ncol=None, markers_only=False):
    """Legend between the panel title and the top of the plot frame.

    Keys with a line use Figure 2's handle length. Marker-only keys use a
    shorter handle and column gap, so a three-entry key fits over one panel.
    """
    spacing = dict(columnspacing=1.2, handlelength=1.0) if markers_only else dict(columnspacing=1.6, handlelength=1.8)
    return ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1.0),
                     ncol=ncol or len(labels), borderaxespad=0, **spacing)


def hairline_grid(ax, axis='y'):
    ax.grid(axis=axis, color=HAIR, lw=.5)
    ax.set_axisbelow(True)


def series_handle(style):
    """Legend key for a series drawn with plot_series (Figure 2's keys)."""
    ls = style.get('ls', '-')
    if style['key_edge']:
        return Line2D([], [], marker=style['marker'], mfc=style['face'], mec=style['edge'],
                      mew=style['key_edge'], color=style['line'], ms=style['key_size'], lw=LINE_WIDTH, ls=ls)
    return Line2D([], [], marker=style['marker'], color=style['line'], ms=style['key_size'], mew=0, lw=LINE_WIDTH,
                  ls=ls)


def agency_handle(agency, line=False):
    """Legend key for an agency marker, with its line when the figure draws one."""
    return Line2D([], [], color=AGENCY_LINES[agency], lw=LINE_WIDTH if line else 0,
                  ls='-' if line else 'none', **AGENCY_MARKERS[agency])


def whiskers(ax, x, y, low, high, color, gid=None, horizontal=False):
    """95 percent interval whiskers with Figure 2's caps and weights."""
    x, y, low, high = (np.asarray(v, dtype=float) for v in (x, y, low, high))
    centre = x if horizontal else y
    error = [centre - low, high - centre]
    container = ax.errorbar(x, y, **({'xerr': error} if horizontal else {'yerr': error}),
                            fmt='none', ecolor=color, zorder=3, **WHISKER)
    if gid:
        container.lines[2][0].set_gid(gid)
    return container


def plot_series(ax, x, y, low, high, style, gid):
    """Figure 2's series: a line, whiskers, then markers on top.

    The line carries `gid`, and the whiskers carry `gid` + '_interval'. A style with `ls` draws that line pattern.
    """
    ax.plot(x, y, color=style['line'], lw=LINE_WIDTH, ls=style.get('ls', '-'), zorder=2, gid=gid)
    whiskers(ax, x, y, low, high, style['whisker'], gid=f'{gid}_interval')
    ax.plot(x, y, marker=style['marker'], ms=style['size'], mfc=style['face'], mec=style['edge'],
            mew=MARKER_EDGE, ls='none', zorder=4, clip_on=False)


# The lower end of the success axes. None keeps the data-driven start. The six-model build sets 0,
# because PNAS asks that numerical axes go to 0 (peer review 2026-10-28).
AXIS_FLOOR = None


def success_axis(ax, lows, top=101, label=True):
    """Task-success axis from the multiple of ten below the lowest interval end, or from AXIS_FLOOR."""
    lower = min(50, 10 * math.floor(float(np.nanmin(lows)) / 10)) if AXIS_FLOOR is None else AXIS_FLOOR
    ax.set_ylim(lower, top)
    ax.set_yticks(range(lower, 101, 10))
    if label:
        ax.set_ylabel('Task success (percent)', fontsize=AXIS_LABEL_SIZE)
    return lower


def _tick_labels(axis):
    """The drawn tick labels of an axis: visible, nonempty, and inside the view."""
    low, high = sorted(axis.get_view_interval())
    tolerance = 1e-9 * max(1.0, abs(high - low))
    locations = axis.get_majorticklocs()
    labels = axis.get_majorticklabels()
    return [label for location, label in zip(locations, labels)
            if low - tolerance <= location <= high + tolerance]


def figure_texts(fig):
    """Every drawn text of a figure: titles, axis and tick labels, notes, and legends."""
    texts = list(fig.texts)
    legends = list(fig.legends)
    for ax in fig.axes:
        texts += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label]
        texts += _tick_labels(ax.xaxis) + _tick_labels(ax.yaxis) + list(ax.texts)
        if ax.get_legend() is not None:
            legends.append(ax.get_legend())
    for legend in legends:
        texts += list(legend.get_texts()) + [legend.get_title()]
    return [text for text in texts if text.get_visible() and text.get_text().strip()]


def _overlap(a, b, tolerance=.5):
    return a.x0 < b.x1 - tolerance and a.x1 > b.x0 + tolerance and a.y0 < b.y1 - tolerance and a.y1 > b.y0 + tolerance


def assert_text_outside(fig):
    """Check the text rules of the paper figures and raise AssertionError on a failure.

    1. No text sits inside a plot frame (project rule).
    2. No two texts overlap, for example a legend entry and a series label.
    3. Every text lies inside the figure canvas.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frames = [ax.get_window_extent(renderer) for ax in fig.axes]
    boxes = [(text, Text.get_window_extent(text, renderer)) for text in figure_texts(fig)]
    for text, box in boxes:
        if any(_overlap(box, frame) for frame in frames):
            raise AssertionError(f'text inside plot area: {text.get_text()!r}')
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1:
            raise AssertionError(f'text outside the figure: {text.get_text()!r}')
    for (first, a), (second, b) in itertools.combinations(boxes, 2):
        if _overlap(a, b):
            raise AssertionError(f'texts overlap: {first.get_text()!r} and {second.get_text()!r}')


def text_margins(fig):
    """Distance in points from the drawn texts to the left, right, bottom, and top canvas edges."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [Text.get_window_extent(text, renderer) for text in figure_texts(fig)]
    to_pt = 72 / fig.dpi
    return dict(left=min(b.x0 for b in boxes) * to_pt, right=(fig.bbox.x1 - max(b.x1 for b in boxes)) * to_pt,
                bottom=min(b.y0 for b in boxes) * to_pt, top=(fig.bbox.y1 - max(b.y1 for b in boxes)) * to_pt)


def keep_side_margins(fig, pad_pt, rounds=6):
    """Narrow the axes until every text ends at least pad_pt points inside the left and right canvas edges.

    The horizontal positions of all axes map linearly onto a span that is shorter by the shortfall, so the
    panels keep their order and relative spacing, and the texts that belong to an axes move with it. The
    PNAS build uses it, because one Figure 3B tick label ended 0.65 pt from the edge (peer review 2026-10-28,
    I77). Returns the final margins from text_margins.
    """
    for _ in range(rounds):
        margins = text_margins(fig)
        short_left, short_right = max(0.0, pad_pt - margins['left']), max(0.0, pad_pt - margins['right'])
        if short_left < 1e-3 and short_right < 1e-3:
            return margins
        width_pt = fig.get_figwidth() * 72
        positions = [ax.get_position() for ax in fig.axes]
        low, high = min(p.x0 for p in positions), max(p.x1 for p in positions)
        new_low, new_high = low + 1.05 * short_left / width_pt, high - 1.05 * short_right / width_pt
        scale = (new_high - new_low) / (high - low)
        for ax, position in zip(fig.axes, positions):
            ax.set_position([new_low + (position.x0 - low) * scale, position.y0, position.width * scale,
                             position.height])
    margins = text_margins(fig)
    if min(margins['left'], margins['right']) < pad_pt - 1e-3:
        raise AssertionError(f'text margin below {pad_pt} pt after {rounds} rounds: {margins}')
    return margins


def text_sizes(fig):
    """The font sizes, in points, of the drawn texts of a figure."""
    return sorted({round(text.get_fontsize(), 2) for text in figure_texts(fig)})


def interval_segments(ax, gid):
    """Whisker ends drawn under `gid`, as (x0, y0, x1, y1) rows (test helper)."""
    collections = [c for c in ax.collections if c.get_gid() == gid]
    if len(collections) != 1:
        raise AssertionError(f'expected one whisker set {gid!r}, found {len(collections)}')
    return np.array([[*segment[0], *segment[-1]] for segment in collections[0].get_segments()])

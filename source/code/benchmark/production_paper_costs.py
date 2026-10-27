"""Draw task success against mean cost or tokens per run (Figure 4 and the token figure).

The figures use the Figure 2 design in paper_figure_style.py (colors, fonts,
and agency markers) and draw points without intervals (Federico, 2026-09-25).
The task-success axis spans the plotted points with a small margin.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
import pandas as pd

import paper_figure_style as style

MODELS = ['sonnet', 'luna']
PLATFORMS = ['python', 'r', 'stata']
AGENCIES = ['chatbot', 'self_repair', 'agent']
MODEL_NAMES = {'sonnet': 'Claude Sonnet 4.6', 'luna': 'GPT-5.6 Luna'}
PANEL_TITLES = {'sonnet': 'A. Claude Sonnet 4.6', 'luna': 'B. GPT-5.6 Luna'}
AGENCY_NAMES = {'chatbot': 'Chatbot', 'self_repair': 'Self-repair', 'agent': 'Constrained agent'}
PLATFORM_NAMES = {'python': 'Python', 'r': 'R', 'stata': 'Stata'}
# Panel geometry. The margin to the right of each panel holds that panel's
# three language labels, so the gutter is sized for the widest label.
FIG_H, TOP, BOTTOM = 3.7, .835, .135
LEFT, RIGHT, GUTTER = .075, .923, .077
PANEL_W = (RIGHT - LEFT - GUTTER) / 2
WSPACE = GUTTER / PANEL_W
# Language labels: minimum gap between two label centers, and the distance from
# the frame to the start of the text, in points.
LABEL_GAP_PT, LABEL_PAD_PT = 9.5, 5.0
# Each panel's resource axis starts and ends EDGE_PAD_PT points beyond its own
# lowest and highest point, and the shared task-success axis does the same for
# all points, so every marker stays whole inside the frame and no empty stretch
# of axis remains. MIN_PAD_DECADES keeps a nonzero log axis (comparison variant
# only) when all values are equal.
EDGE_PAD_PT, MIN_PAD_DECADES = 6.0, .02


def resource_points(data, resource='cost_usd'):
    """Use the same complete records for each point's cost and success rate."""
    rows = []
    for model in MODELS:
        for platform in PLATFORMS:
            for agency in AGENCIES:
                all_rows = data[data.model.eq(model) & data.platform.eq(platform)
                                & data.agency.eq(agency)]
                group = all_rows[all_rows[resource].notna()]
                if group.empty:
                    raise ValueError(f'Missing {resource} for {model}/{platform}/{agency}')
                mean = float(group[resource].mean())
                rows.append(dict(model=model, platform=platform, agency=agency,
                    runs=len(all_rows), observed=len(group), success_pct=100 * group.primary.mean(),
                    resource=resource, mean_resource=mean,
                    x_value=100 * mean if resource == 'cost_usd' else mean / 1000))
    return pd.DataFrame(rows)


def padded_log_limits(low, high, width_pt, pad_pt=None):
    """Log-axis limits that put `low` and `high` pad_pt points inside the frame."""
    pad_pt = EDGE_PAD_PT if pad_pt is None else pad_pt
    if not 0 < low <= high:
        raise ValueError('A log axis needs positive plotted values')
    if width_pt <= 2 * pad_pt:
        raise ValueError('The panel is narrower than its padding')
    pad = max(pad_pt * math.log10(high / low) / (width_pt - 2 * pad_pt), MIN_PAD_DECADES)
    return low / 10 ** pad, high * 10 ** pad


def padded_linear_limits(low, high, width_pt, pad_pt=None):
    """Linear-axis limits that put `low` and `high` pad_pt points inside the frame."""
    pad_pt = EDGE_PAD_PT if pad_pt is None else pad_pt
    if width_pt <= 2 * pad_pt:
        raise ValueError('The panel is narrower than its padding')
    span = high - low if high > low else max(abs(high) * .1, 1e-9)
    pad = pad_pt * span / (width_pt - 2 * pad_pt)
    return low - pad, high + pad


def log_ticks(low, high):
    """Return the 1-2-5 ticks inside the axis limits (denser steps if fewer than two)."""
    for steps in ((1, 2, 5), tuple(range(1, 10))):
        ticks = [m * 10.0 ** e for e in range(-3, 7) for m in steps]
        inside = [t for t in ticks if low * (1 - 1e-9) <= t <= high * (1 + 1e-9)]
        if len(inside) >= 2:
            return inside
    raise ValueError(f'No readable log ticks between {low} and {high}')


def linear_ticks(low, high):
    """Return round linear ticks inside the axis limits."""
    ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(low, high)
    return [float(t) for t in ticks if low - 1e-12 <= t <= high + 1e-12]


def tick_text(value):
    value = round(value, 10)
    if value >= 1 and value == int(value):
        return f'{value:,.0f}'
    return f'{value:g}'


def build_resource_figure(data, resource='cost_usd', scale='linear'):
    """Draw one panel per model, each with its own linear resource axis.

    The two models differ in cost per run by more than a factor of ten, so one
    shared linear axis would compress the cheaper model into a narrow strip. Each
    panel's axis starts and ends EDGE_PAD_PT points beyond that panel's lowest
    and highest point. The panels share the task-success axis, which spans all
    points with the same margin. The points carry no interval. Language labels
    stay in the right-hand margin, outside the data frame. scale='log' draws a
    comparison variant with one shared padded log axis. The paper and the token
    figure use the linear axis.
    """
    if scale not in ('log', 'linear'):
        raise ValueError(f'Unknown axis scale: {scale}')
    points = resource_points(data, resource)
    unit = 'US cents' if resource == 'cost_usd' else 'thousands'
    with style.figure_style():
        fig = plt.figure(figsize=(style.FIG_W, FIG_H))
        gs = fig.add_gridspec(1, 2, left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM, wspace=WSPACE)
        axes = [fig.add_subplot(gs[0, 0])]
        axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
        width_pt = PANEL_W * style.FIG_W * 72
        ylim = padded_linear_limits(float(points.success_pct.min()), float(points.success_pct.max()),
                                    (TOP - BOTTOM) * FIG_H * 72)
        # Ticks every ten percentage points, as on the other success axes.
        yticks = [float(t) for t in range(0, 101, 10) if ylim[0] <= t <= ylim[1]] or linear_ticks(*ylim)
        axes[0].set_ylim(*ylim)
        axes[0].set_yticks(yticks, [tick_text(t) for t in yticks])
        axes[0].set_ylabel('Task success (percent)', fontsize=style.AXIS_LABEL_SIZE)
        for ax, model in zip(axes, MODELS):
            selected = points[points.model.eq(model)]
            if scale == 'log':
                xlim = padded_log_limits(float(points.x_value.min()), float(points.x_value.max()), width_pt)
                ax.set_xscale('log')
                xticks = log_ticks(*xlim)
                label_x = xlim[1] * 10 ** (LABEL_PAD_PT * math.log10(xlim[1] / xlim[0]) / width_pt)
            else:
                xlim = padded_linear_limits(float(selected.x_value.min()), float(selected.x_value.max()), width_pt)
                xticks = linear_ticks(*xlim)
                label_x = xlim[1] + LABEL_PAD_PT * (xlim[1] - xlim[0]) / width_pt
            ax.set_xlim(*xlim)
            ax.set_xticks(xticks, [tick_text(t) for t in xticks])
            ax.xaxis.set_minor_locator(NullLocator())
            style.hairline_grid(ax)
            ends = []
            for platform in PLATFORMS:
                group = selected[selected.platform.eq(platform)].set_index('agency').loc[AGENCIES]
                ax.plot(group.x_value, group.success_pct, color=style.MID, lw=.9, zorder=2, gid=f'{model}_{platform}')
                for agency, row in group.iterrows():
                    ax.plot([row.x_value], [row.success_pct], ls='none', zorder=4, clip_on=False,
                            gid=f'{model}_{platform}_{agency}_point', **style.AGENCY_MARKERS[agency])
                ends.append((group.x_value.iloc[-1], group.success_pct.iloc[-1]))
            points_per_y = (TOP - BOTTOM) * FIG_H * 72 / (ylim[1] - ylim[0])
            # Keep every label inside the frame's vertical range, half a gap
            # from its top and bottom, so no label reaches the legend above.
            half_gap = LABEL_GAP_PT / 2 / points_per_y
            label_ys = style.stacked_label_y([end[1] for end in ends], LABEL_GAP_PT / points_per_y,
                                             bounds=(ylim[0] + half_gap, ylim[1] - half_gap))
            for platform, (x, y), label_y in zip(PLATFORMS, ends, label_ys):
                ax.annotate(PLATFORM_NAMES[platform], xy=(x, y), xytext=(label_x, label_y),
                    fontsize=style.X_TICK_LABEL_SIZE, color=style.INK, ha='left', va='center',
                    annotation_clip=False, clip_on=False, zorder=5,
                    arrowprops=dict(arrowstyle='-', color=style.HAIR, lw=.5,
                        linestyle=(0, (1, 2.2)), shrinkA=3.5, shrinkB=3.5, clip_on=False))
            style.panel_title(ax, PANEL_TITLES[model])
            suffix = f'{unit}, log scale' if scale == 'log' else unit
            ax.set_xlabel(f'Mean cost per run ({suffix})' if resource == 'cost_usd'
                          else f'Mean input and output tokens ({suffix})', fontsize=style.AXIS_LABEL_SIZE)
        plt.setp(axes[1].get_yticklabels(), visible=False)
        style.legend_above(axes[0], [style.agency_handle(a) for a in AGENCIES],
                           [AGENCY_NAMES[a] for a in AGENCIES], markers_only=True)
        style.assert_text_outside(fig)
    return fig, points


# Joint figure of the two models and the three agency conditions (Appendix C).
JOINT_FIG_H, JOINT_LEFT, JOINT_RIGHT, JOINT_TOP, JOINT_BOTTOM = 3.3, .085, .975, .84, .145
MODEL_LINES = {'sonnet': dict(color=style.STRONG, ls='-'),
               'luna': dict(color=style.MID, ls=(0, (3, 2)))}


def model_agency_points(data):
    """One point per model and agency condition, pooled over prompts and languages.

    Success is the share of runs that succeed, with a 95 percent task cluster bootstrap
    interval (style.bootstrap_ratios). Cost is the mean paper cost per run in US cents.
    """
    if data.cost_usd.isna().any():
        raise ValueError('Every run needs a paper cost for the joint figure')
    frame = data.assign(success=data.primary.astype(float))
    success = style.bootstrap_ratios(frame, ['model', 'agency'], 'success')
    cost = (100 * frame.groupby(['model', 'agency']).cost_usd.mean()).rename('cost_cents').reset_index()
    points = success.merge(cost, on=['model', 'agency'], validate='one_to_one')
    points = points.rename(columns={'value': 'success_pct', 'value_lo': 'success_lo', 'value_hi': 'success_hi'})
    return points.sort_values(by=['model', 'agency'], key=lambda s: s.map(
        {**{m: i for i, m in enumerate(MODELS)}, **{a: i for i, a in enumerate(AGENCIES)}})).reset_index(drop=True)


def build_model_agency_figure(data):
    """Task success against mean cost per run, both models on one log cost axis.

    Each model's line joins its chatbot, self-repair, and constrained-agent points.
    Whiskers are 95 percent task cluster bootstrap intervals for task success. One key above
    the frame names the agency markers and the two model lines, so no text enters the plot.
    """
    from matplotlib.lines import Line2D
    points = model_agency_points(data)
    with style.figure_style():
        fig = plt.figure(figsize=(style.FIG_W, JOINT_FIG_H))
        ax = fig.add_axes([JOINT_LEFT, JOINT_BOTTOM, JOINT_RIGHT - JOINT_LEFT, JOINT_TOP - JOINT_BOTTOM])
        width_pt = (JOINT_RIGHT - JOINT_LEFT) * style.FIG_W * 72
        height_pt = (JOINT_TOP - JOINT_BOTTOM) * JOINT_FIG_H * 72
        xlim = padded_log_limits(float(points.cost_cents.min()), float(points.cost_cents.max()), width_pt)
        ylim = padded_linear_limits(float(points.success_lo.min()), float(points.success_hi.max()), height_pt)
        ax.set_xscale('log')
        ax.set_xlim(*xlim)
        xticks = log_ticks(*xlim)
        ax.set_xticks(xticks, [tick_text(t) for t in xticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(*ylim)
        yticks = [float(t) for t in range(0, 101, 5) if ylim[0] <= t <= ylim[1]] or linear_ticks(*ylim)
        ax.set_yticks(yticks, [tick_text(t) for t in yticks])
        style.hairline_grid(ax)
        for model in MODELS:
            group = points[points.model.eq(model)].set_index('agency').loc[AGENCIES]
            ax.plot(group.cost_cents, group.success_pct, lw=1.0, zorder=2, gid=f'{model}_line', **MODEL_LINES[model])
            style.whiskers(ax, group.cost_cents, group.success_pct, group.success_lo, group.success_hi,
                           MODEL_LINES[model]['color'], gid=f'{model}_interval')
            for agency, row in group.iterrows():
                ax.plot([row.cost_cents], [row.success_pct], ls='none', zorder=4, clip_on=False,
                        gid=f'{model}_{agency}_point', **style.AGENCY_MARKERS[agency])
        ax.set_xlabel('Mean cost per run (US cents, log scale)', fontsize=style.AXIS_LABEL_SIZE)
        ax.set_ylabel('Task success (percent)', fontsize=style.AXIS_LABEL_SIZE)
        handles = [style.agency_handle(a) for a in AGENCIES] + [
            Line2D([], [], lw=1.0, **MODEL_LINES[m]) for m in MODELS]
        style.legend_above(ax, handles, [AGENCY_NAMES[a] for a in AGENCIES] + [MODEL_NAMES[m] for m in MODELS])
        style.assert_text_outside(fig)
    return fig, points


def render_cost_figures(data, out, source, save):
    fig, points = build_resource_figure(data)
    caption = ('Task success and mean cost per run by language and agency, one panel per model. '
        'Costs are in US cents. The panels share the task-success axis and have separate linear cost axes. '
        'Each point pools both prompts and uses the same records for cost and success. '
        'The pricing receipt records the token rates.')
    # Save inside the style, which sets the embedded PDF font type.
    with style.figure_style():
        save(fig, out, 'cost_of_correctness', points, caption, source)
    fig, points = build_resource_figure(data, 'total_main_tokens')
    with style.figure_style():
        save(fig, out, 'cost_tokens', points,
            'Task success and mean input plus output tokens per run, one panel per model. '
            'Each point uses runs with complete token records. '
            'The panels share the task-success axis and have separate linear token axes.', source)
    fig, points = build_model_agency_figure(data)
    with style.figure_style():
        save(fig, out, 'model_agency_frontier', points,
            'Task success and mean cost per run by model and agency condition, pooled over prompts and languages. '
            'Costs are in US cents on a log scale. Whiskers are 95 percent intervals from a cluster bootstrap '
            'over the 40 tasks.', source)

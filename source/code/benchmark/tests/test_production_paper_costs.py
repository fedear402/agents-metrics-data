"""Check both cost panels, currency units, and the complete-record denominator."""
import itertools
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import production_paper_costs as costs


def sample():
    """Two tasks per cell with equal costs."""
    return pd.DataFrame([dict(task=task, model=model, platform=platform, agency=agency,
        cost_usd=.20 if model == 'sonnet' else .02,
        total_main_tokens=12000, primary=True)
        for task, model, platform, agency in itertools.product(
            ['task_a', 'task_b'], costs.MODELS, costs.PLATFORMS, costs.AGENCIES)])


def test_both_models_in_cents_in_the_figure2_design():
    fig, points = costs.build_resource_figure(sample())
    try:
        assert len(fig.axes) == 2
        assert [ax.get_title(loc='left') for ax in fig.axes] == list(costs.PANEL_TITLES.values())
        assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == list(costs.AGENCY_NAMES.values())
        assert all(ax.get_xlabel() == 'Mean cost per run (US cents)' for ax in fig.axes)
        assert all(ax.get_xscale() == 'linear' for ax in fig.axes)
        assert points[points.model.eq('sonnet')].x_value.eq(20).all()
        assert points[points.model.eq('luna')].x_value.eq(2).all()
        # Equal values still give each panel a nonzero axis around its value.
        for ax, value in zip(fig.axes, (20, 2)):
            low, high = ax.get_xlim()
            assert low < value < high
        assert len(points) == 18
        assert all(len(ax.texts) == 3 for ax in fig.axes)
    finally:
        plt.close(fig)


def test_missing_cost_excludes_same_observation_from_success():
    data = sample()
    missing = data.iloc[0].copy()
    missing['primary'] = False
    missing['cost_usd'] = float('nan')
    points = costs.resource_points(pd.concat([data, missing.to_frame().T], ignore_index=True))
    point = points.iloc[0]
    assert point.runs == 3
    assert point.observed == 2
    assert point.success_pct == 100


def test_no_currency_price_cannot_silently_remove_model():
    data = sample()
    data.loc[data.model.eq('luna'), 'cost_usd'] = float('nan')
    with pytest.raises(ValueError, match='Missing cost_usd for luna'):
        costs.resource_points(data)


def edge_gaps_pt(fig, ax, low, high):
    """Distance in points from the frame edges to the outermost plotted values."""
    fig.canvas.draw()
    frame = ax.get_window_extent()
    to_pt = 72 / fig.dpi
    left = (ax.transData.transform((low, 50))[0] - frame.x0) * to_pt
    right = (frame.x1 - ax.transData.transform((high, 50))[0]) * to_pt
    return left, right


def test_log_comparison_variant_shares_one_padded_axis():
    data = sample()
    data.loc[data.model.eq('luna'), 'cost_usd'] = .0017
    data.loc[data.model.eq('sonnet'), 'cost_usd'] = .173
    fig, points = costs.build_resource_figure(data, scale='log')
    try:
        assert all(ax.get_xscale() == 'log' for ax in fig.axes)
        assert all(ax.get_xlabel() == 'Mean cost per run (US cents, log scale)' for ax in fig.axes)
        low, high = fig.axes[0].get_xlim()
        assert fig.axes[1].get_xlim() == (low, high)
        assert low < points.x_value.min() and high > points.x_value.max()
        # Small padding only: the empty space at each end is EDGE_PAD_PT on the page.
        assert low > points.x_value.min() / 1.25 and high < points.x_value.max() * 1.25
        for ax in fig.axes:
            gaps = edge_gaps_pt(fig, ax, points.x_value.min(), points.x_value.max())
            assert gaps == pytest.approx((costs.EDGE_PAD_PT, costs.EDGE_PAD_PT), abs=.05)
        assert [t.get_text() for t in fig.axes[1].get_xticklabels()] == ['0.2', '0.5', '1', '2', '5', '10', '20']
    finally:
        plt.close(fig)


def test_padded_log_limits_place_the_data_pad_points_inside():
    low, high = costs.padded_log_limits(.17, 17.3, width_pt=181)
    decades = math.log10(high / low)
    assert math.log10(.17 / low) * 181 / decades == pytest.approx(costs.EDGE_PAD_PT)
    assert math.log10(high / 17.3) * 181 / decades == pytest.approx(costs.EDGE_PAD_PT)
    same_low, same_high = costs.padded_log_limits(5, 5, width_pt=181)
    assert same_low < 5 < same_high


def test_log_ticks_use_one_two_five_steps_inside_the_limits():
    assert costs.log_ticks(.15, 20.4) == pytest.approx([.2, .5, 1, 2, 5, 10, 20])
    assert costs.log_ticks(5.6, 196) == pytest.approx([10, 20, 50, 100])
    assert costs.log_ticks(2, 9) == pytest.approx([2, 5])


def test_default_linear_axis_gives_each_panel_its_own_padded_axis():
    data = sample()
    data.loc[data.model.eq('luna') & data.agency.eq('agent'), 'cost_usd'] = .0108
    data.loc[data.model.eq('sonnet') & data.agency.eq('agent'), 'cost_usd'] = .173
    fig, points = costs.build_resource_figure(data)
    try:
        assert all(ax.get_xscale() == 'linear' for ax in fig.axes)
        assert fig.axes[0].get_xlim() != fig.axes[1].get_xlim()
        assert all(ax.get_xlabel() == 'Mean cost per run (US cents)' for ax in fig.axes)
        for ax, model in zip(fig.axes, costs.MODELS):
            selected = points[points.model.eq(model)]
            low, high = selected.x_value.min(), selected.x_value.max()
            # Small padding only: the empty space at each end is EDGE_PAD_PT on the page.
            gaps = edge_gaps_pt(fig, ax, low, high)
            assert gaps == pytest.approx((costs.EDGE_PAD_PT, costs.EDGE_PAD_PT), abs=.05)
            xlim = ax.get_xlim()
            assert xlim[0] > low - .1 * (high - low) and xlim[1] < high + .1 * (high - low)
            ticks = ax.get_xticks()
            assert len(ticks) >= 2 and all(xlim[0] <= t <= xlim[1] for t in ticks)
        assert all(len(ax.texts) == 3 for ax in fig.axes)
    finally:
        plt.close(fig)


def test_token_figure_gives_each_panel_its_own_linear_axis():
    data = sample()
    data.loc[data.model.eq('luna') & data.agency.eq('agent'), 'total_main_tokens'] = 190000
    data.loc[data.model.eq('sonnet') & data.agency.eq('agent'), 'total_main_tokens'] = 120000
    fig, points = costs.build_resource_figure(data, 'total_main_tokens')
    try:
        assert all(ax.get_xscale() == 'linear' for ax in fig.axes)
        assert all(ax.get_xlabel() == 'Mean input and output tokens (thousands)' for ax in fig.axes)
        assert fig.axes[0].get_xlim() != fig.axes[1].get_xlim()
        for ax, model in zip(fig.axes, costs.MODELS):
            selected = points[points.model.eq(model)]
            gaps = edge_gaps_pt(fig, ax, selected.x_value.min(), selected.x_value.max())
            assert gaps == pytest.approx((costs.EDGE_PAD_PT, costs.EDGE_PAD_PT), abs=.05)
    finally:
        plt.close(fig)


def test_linear_ticks_are_round_values_inside_the_limits():
    assert costs.linear_ticks(3.0, 17.8) == pytest.approx([5, 10, 15])
    assert costs.linear_ticks(.15, 1.1) == pytest.approx([.2, .4, .6, .8, 1])


def test_captions_name_separate_linear_axes():
    captions = {}

    def save(fig, out, name, points, caption, source):
        captions[name] = caption
        plt.close(fig)

    costs.render_cost_figures(sample(), None, None, save)
    assert set(captions) == {'cost_of_correctness', 'cost_tokens', 'model_agency_frontier'}
    assert 'separate linear cost axes' in captions['cost_of_correctness']
    assert 'separate linear token axes' in captions['cost_tokens']
    per_model = [captions['cost_of_correctness'], captions['cost_tokens']]
    assert not any('logarithm' in caption or 'log scale' in caption for caption in per_model)
    # The per-model cost figures draw points only (Federico, 2026-09-25).
    assert not any('whisker' in caption.lower() or 'interval' in caption.lower() or 'bootstrap' in caption
                   for caption in per_model)
    # The joint figure of Appendix C puts both models on one log cost axis with bootstrap whiskers.
    assert 'log scale' in captions['model_agency_frontier']
    assert 'bootstrap' in captions['model_agency_frontier']


def test_points_carry_no_interval_and_the_success_axis_spans_the_points():
    """No whiskers, and the success axis starts EDGE_PAD_PT below the lowest point (no band from 40)."""
    data = sample()
    data.loc[data.model.eq('luna') & data.platform.eq('stata') & data.agency.eq('chatbot')
             & data.task.eq('task_a'), 'primary'] = False
    fig, points = costs.build_resource_figure(data)
    try:
        assert points.success_pct.min() == 50 and points.success_pct.max() == 100
        assert not {'success_pct_lo', 'success_pct_hi', 'x_value_lo', 'x_value_hi'} & set(points)
        fig.canvas.draw()
        to_pt = 72 / fig.dpi
        for ax in fig.axes:
            assert not ax.collections
            frame = ax.get_window_extent()
            bottom = (ax.transData.transform((0, 50))[1] - frame.y0) * to_pt
            top = (frame.y1 - ax.transData.transform((0, 100))[1]) * to_pt
            assert (bottom, top) == pytest.approx((costs.EDGE_PAD_PT, costs.EDGE_PAD_PT), abs=.05)
        assert list(fig.axes[0].get_yticks()) == [50, 60, 70, 80, 90, 100]
        assert fig.axes[0].get_ylabel() == 'Task success (percent)'
    finally:
        plt.close(fig)


def test_software_labels_stay_inside_the_frame_height_and_clear_the_legend():
    """Three close line ends at the top of the axis push the label stack down, inside the frame."""
    data = sample()
    data.loc[data.model.eq('sonnet') & data.agency.eq('agent'), 'cost_usd'] = .173
    data.loc[data.model.eq('sonnet') & data.agency.eq('chatbot'), 'primary'] = False
    fig, points = costs.build_resource_figure(data)
    try:
        for ax in fig.axes:
            low, high = ax.get_ylim()
            labels = [text.xyann[1] for text in ax.texts]
            assert len(labels) == 3 and all(low < y < high for y in labels)
        costs.style.assert_text_outside(fig)
    finally:
        plt.close(fig)


def test_cost_figures_are_saved_with_embedded_truetype_fonts():
    font_types = []

    def save(fig, out, name, points, caption, source):
        font_types.append(plt.rcParams['pdf.fonttype'])
        plt.close(fig)

    costs.render_cost_figures(sample(), None, None, save)
    assert font_types == [42, 42, 42]


def test_model_agency_points_pool_prompts_and_software():
    data = sample()
    points = costs.model_agency_points(data)
    assert list(zip(points.model, points.agency)) == [(m, a) for m in costs.MODELS for a in costs.AGENCIES]
    for r in points.itertuples():
        group = data[data.model.eq(r.model) & data.agency.eq(r.agency)]
        assert r.success_pct == pytest.approx(100 * group.primary.mean())
        assert r.cost_cents == pytest.approx(100 * group.cost_usd.mean())
        assert r.success_lo <= r.success_pct <= r.success_hi
    fig, _ = costs.build_model_agency_figure(data)
    try:
        assert len(fig.axes) == 1 and fig.axes[0].get_xscale() == 'log'
    finally:
        plt.close(fig)


MANUSCRIPT_COST_SECTION = (Path(__file__).resolve().parents[3]
                           / 'paper/manuscript/sections/05_cost_efficiency.tex')


@pytest.mark.skipif(not MANUSCRIPT_COST_SECTION.exists(), reason='paper/manuscript is not present')
def test_manuscript_caption_names_separate_linear_cost_axes():
    text = MANUSCRIPT_COST_SECTION.read_text()
    # Short caption title plus a normal-case notes block (2026-10-02); the notes end the figure.
    start = text.index('figures/cost_of_correctness.pdf')
    caption = text[start:text.index(r'\end{figure}', start)]
    assert r'\caption{Task Success and Mean Cost per Run by Programming Language and Agency Condition}' in caption
    assert 'separate linear cost axes' in caption
    assert 'logarithm' not in caption and 'share both axes' not in caption
    assert 'whisker' not in caption.lower() and 'interval' not in caption.lower()

"""Check the structure of the paper figures and the saved pass@k population.

The checks of plotted values and intervals against runs.csv are in
test_paper_figure_intervals.py.
"""
import itertools
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.text import Text
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import paper_figure_style as style
import production_paper_figures as figures


def sample(repetitions=5):
    return pd.DataFrame([dict(task='task', model=model, platform=platform, prompt=prompt,
        agency=agency, rep=rep, primary=rep != 0, execution_ok=True, output_available=True,
        provider_seconds=10 + 20 * figures.AGENCIES.index(agency))
        for model, platform, prompt, agency, rep in itertools.product(
            figures.MODELS, figures.PLATFORMS, figures.PROMPTS, figures.AGENCIES, range(repetitions))])


def test_pass_at_k_uses_all_ten_repetitions():
    data = sample(10)
    data.loc[data.rep.ge(5), 'primary'] = False
    full, task_count = figures.pass_at_k_data(data)
    assert task_count == 1
    assert set(full.k) == set(range(1, 11))
    assert full[full.k.eq(1)].pass_at_k_pct.tolist() == pytest.approx([40] * 6)
    assert full[full.k.eq(2)].pass_at_k_pct.tolist() == pytest.approx([200 / 3] * 6)
    assert full[full.k.eq(10)].pass_at_k_pct.eq(100).all()
    assert full.repetitions.eq(10).all()


def test_pass_at_k_supports_legacy_five_repetitions():
    result, _ = figures.pass_at_k_data(sample(5))
    assert set(result.k) == set(range(1, 6))
    assert result.repetitions.eq(5).all()
    assert result[result.k.eq(1)].pass_at_k_pct.eq(80).all()
    assert result[result.k.eq(2)].pass_at_k_pct.eq(100).all()


@pytest.mark.parametrize('success', [False, True])
def test_pass_at_k_all_failure_and_all_success(success):
    data = sample(10)
    data['primary'] = success
    result, _ = figures.pass_at_k_data(data)
    assert result.pass_at_k_pct.eq(100 * success).all()


def test_pass_at_k_rejects_duplicate_repetitions():
    data = sample(10)
    with pytest.raises(ValueError, match='duplicate'):
        figures.pass_at_k_data(pd.concat([data, data.iloc[[0]]]))


@pytest.mark.parametrize('missing', ['run', 'cell', 'later_repetitions'])
def test_pass_at_k_rejects_unbalanced_design(missing):
    data = sample(10)
    if missing == 'run':
        data = data.iloc[1:]
    elif missing == 'cell':
        data = data.iloc[10:]
    else:
        other = sample(5).assign(task='other_task')
        data = pd.concat([data, other])
    with pytest.raises(ValueError, match='balanced'):
        figures.pass_at_k_data(data)


def test_pass_at_k_rejects_noncontiguous_repetitions():
    data = sample(10)
    data.loc[data.rep.eq(9), 'rep'] = 10
    with pytest.raises(ValueError, match='contiguous'):
        figures.pass_at_k_data(data)


@pytest.mark.parametrize('invalid', [None, 0.5, float('inf')])
def test_pass_at_k_rejects_invalid_outcomes(invalid):
    data = sample(10)
    data['primary'] = data.primary.astype(object)
    data.loc[0, 'primary'] = invalid
    with pytest.raises(ValueError, match='binary'):
        figures.pass_at_k_data(data)


def test_pass_figure_uses_all_ten_repetitions_in_axis_and_caption(tmp_path):
    captured = []
    def save(fig, destination, stem, plotted, caption, source):
        captured.append((fig, plotted, caption))
    figures.pass_figure(sample(10), tmp_path, tmp_path / 'source.csv', save)
    fig, plotted, caption = captured[0]
    for ax in fig.axes:
        assert ax.get_xticks().tolist() == list(range(1, 11))
        assert ax.get_xlim() == pytest.approx((.6, 10.4))
    assert [ax.get_title(loc='left') for ax in fig.axes] == ['A. Claude Sonnet 4.6', 'B. GPT-5.6 Luna']
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ['Chatbot', 'Self-repair', 'Constrained agent']
    assert plotted.repetitions.eq(10).all()
    assert 'all 10 repetitions' in caption and 'cluster bootstrap over tasks' in caption
    plt.close(fig)


def test_zero_failures_have_missing_silent_share():
    data = sample()
    data['primary'] = True
    result = figures.failure_rates(data)
    assert result.silent_share_of_failures_pct.isna().all()
    assert result[['silent_per100', 'loud_per100']].eq(0).all().all()


def gids(ax):
    return {line.get_gid() for line in ax.lines if line.get_gid()}


@pytest.mark.parametrize('by_model', [False, True])
def test_prompt_figure_draws_two_panels_in_the_figure2_design(tmp_path, by_model):
    captured = []
    def save(fig, destination, stem, plotted, caption, source):
        captured.append((fig, stem, plotted, caption))
    figures.prompt_figure(sample(), tmp_path, tmp_path / 'source.csv', save, by_model=by_model)
    fig, stem, plotted, caption = captured[0]
    assert stem == ('prompt_agency_by_model' if by_model else 'prompt_agency')
    assert len(fig.axes) == 2
    assert fig.get_size_inches().tolist() == pytest.approx([6.5, 2.9])
    titles = [ax.get_title(loc='left') for ax in fig.axes]
    if by_model:
        assert titles == ['A. Claude Sonnet 4.6', 'B. GPT-5.6 Luna']
        assert [gids(ax) for ax in fig.axes] == [set(figures.PROMPTS)] * 2
        assert fig.axes[1].get_legend() is None
    else:
        assert titles == ['A. By prompt condition', 'B. By programming language']
        assert [gids(ax) for ax in fig.axes] == [set(figures.PROMPTS), set(figures.PLATFORMS)]
        assert [t.get_text() for t in fig.axes[1].get_legend().get_texts()] == ['Python', 'R', 'Stata']
        assert set(plotted.panel) == {'prompt', 'platform'}
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ['Zero-shot', 'One-shot']
    # Series labels moved to the legends: no text sits inside or next to a frame.
    assert not any(ax.texts for ax in fig.axes)
    assert {'success_pct_lo', 'success_pct_hi'} <= set(plotted)
    assert plotted.bootstrap_seed.eq(20260925).all() and plotted.bootstrap_draws.eq(4000).all()
    assert 'cluster bootstrap over tasks' in caption and 'seed 20260925' in caption
    plt.close(fig)


def test_runtime_labels_fit_inside_both_model_panels(tmp_path):
    captured = []
    def save(fig, destination, stem, plotted, caption, source):
        captured.append(fig)
    figures.runtime_figure(sample(), tmp_path, tmp_path / 'source.csv', save)
    fig = captured[0]
    assert len(fig.axes) == 2
    assert [gids(ax) for ax in fig.axes] == [{'sonnet_median'}, {'luna_median'}]
    assert [ax.get_title(loc='left') for ax in fig.axes] == ['A. Claude Sonnet 4.6', 'B. GPT-5.6 Luna']
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(match=Text):
        if text.get_visible() and text.get_text().strip():
            bounds = text.get_window_extent(renderer)
            assert bounds.x0 >= -1
            assert bounds.y0 >= -1
            assert bounds.x1 <= fig.bbox.x1 + 1
            assert bounds.y1 <= fig.bbox.y1 + 1
    plt.close(fig)


def failure_sample():
    """Four tasks with loud failures under the chatbot and self-repair and silent failures in every condition."""
    data = pd.concat([sample(10).assign(task=f'task_{i}') for i in range(4)], ignore_index=True)
    data['primary'] = True
    task_number = data.task.str[-1].astype(int)
    loud = ((data.agency.eq('chatbot') & data.rep.lt(task_number + 1))
            | (data.agency.eq('self_repair') & data.rep.eq(0) & task_number.eq(3)))
    silent = data.rep.eq(9) & task_number.isin([0, 1, 2])
    data.loc[loud | silent, 'primary'] = False
    data.loc[loud & data.platform.eq('stata'), 'execution_ok'] = False
    data.loc[loud & ~data.platform.eq('stata'), 'output_available'] = False
    return data


def run_failure_figure(data, tmp_path):
    captured = []
    def save(fig, destination, stem, plotted, caption, source):
        captured.append((fig, stem, plotted, caption))
    per_model = figures.failure_figure(data, tmp_path, tmp_path / 'source.csv', save)
    return (*captured[0], per_model)


def plotted_line(ax, gid):
    lines = [line for line in ax.lines if line.get_gid() == gid]
    assert len(lines) == 1, gid
    return lines[0].get_ydata().tolist()


def test_failure_figure_pools_models_in_rate_and_share_panels(tmp_path):
    data = failure_sample()
    fig, stem, plotted, caption, per_model = run_failure_figure(data, tmp_path)
    assert stem == 'appendix_silent_vs_loud'
    assert len(fig.axes) == 2
    assert plotted.agency.tolist() == figures.AGENCIES
    assert plotted.n.eq(len(data) // 3).all()
    panel_a, panel_b = fig.axes
    assert panel_a.get_ylabel() == 'Failures per 100 runs'
    assert panel_b.get_ylabel() == 'Silent share of failures, percent'
    assert panel_a.get_title(loc='left').startswith('A. ')
    assert panel_b.get_title(loc='left').startswith('B. ')
    assert plotted_line(panel_a, 'loud_per100') == pytest.approx(plotted.loud_per100.tolist())
    assert plotted_line(panel_a, 'silent_per100') == pytest.approx(plotted.silent_per100.tolist())
    assert plotted_line(panel_b, 'silent_share_of_failures_pct') == pytest.approx(plotted.silent_share_of_failures_pct.tolist())
    labels = [t.get_text() for t in panel_b.get_xticklabels()]
    assert labels == [f'{figures.AGENCY_LABEL[a]}\n{n:,} failures' for a, n in zip(figures.AGENCIES, plotted.failures)]
    # The notes moved to the LaTeX caption: no figure-level text remains.
    assert not fig.texts
    assert [t.get_text() for t in panel_a.get_legend().get_texts()] == ['Loud', 'Silent']
    assert 'both models pooled' in caption and 'seed 20260925' in caption
    # The appendix table and the macros keep the rates of each model.
    assert set(per_model.model) == set(figures.MODELS)
    assert 'silent_share_of_failures_pct' in per_model
    plt.close(fig)


def test_failure_figure_rejects_text_inside_a_plot_frame(tmp_path):
    fig, *_ = run_failure_figure(failure_sample(), tmp_path)
    fig.axes[0].text(1, 10, 'inside')
    with pytest.raises(AssertionError, match='inside plot area'):
        style.assert_text_outside(fig)
    plt.close(fig)


def test_failure_bootstrap_is_fixed_and_brackets_the_rates():
    data = failure_sample()
    first = figures.pooled_failure_rates(data)
    again = figures.pooled_failure_rates(data)
    pd.testing.assert_frame_equal(first, again)
    assert first.bootstrap_seed.eq(20260925).all() and first.bootstrap_draws.eq(4000).all()
    for key in ['loud_per100', 'silent_per100', 'silent_share_of_failures_pct']:
        assert (first[f'{key}_lo'] <= first[key] + 1e-12).all()
        assert (first[key] <= first[f'{key}_hi'] + 1e-12).all()
    # The chatbot loud rate varies across tasks, so its interval has width.
    chatbot = first.set_index('agency').loc['chatbot']
    assert chatbot.loud_per100_hi > chatbot.loud_per100_lo


def test_zero_failures_leave_the_silent_share_missing():
    data = failure_sample()
    data['primary'] = True
    result = figures.pooled_failure_rates(data)
    assert result.silent_share_of_failures_pct.isna().all()
    assert result.silent_share_of_failures_pct_lo.isna().all()
    assert result[['loud_per100', 'silent_per100']].eq(0).all().all()


MAIN_DATA = ROOT / 'data/analysis/main/data'


@pytest.mark.skipif(not (MAIN_DATA / 'runs.csv').is_file(), reason='data/analysis/main is not present')
def test_figure2_plotted_values_match_runs_csv(tmp_path):
    """Figure 2 plots the pooled loud and silent rates and shares of the main runs.csv."""
    runs = pd.read_csv(MAIN_DATA / 'runs.csv')
    fig, stem, plotted, caption, per_model = run_failure_figure(runs, tmp_path)
    # Recount from the stored failure category, a column the figure code does not read.
    expected = []
    for agency in figures.AGENCIES:
        group = runs[runs.agency.eq(agency)]
        counts = group.failure_category.value_counts()
        loud, silent = counts.get('Loud failure', 0), counts.get('Silent failure', 0)
        assert loud + silent == (~group.primary).sum()
        expected.append(dict(agency=agency, n=len(group), loud=loud, silent=silent,
                             loud_per100=100 * loud / len(group), silent_per100=100 * silent / len(group),
                             share=100 * silent / (loud + silent)))
    expected = pd.DataFrame(expected)
    assert expected.n.eq(4800).all()
    panel_a, panel_b = fig.axes
    assert plotted_line(panel_a, 'loud_per100') == pytest.approx(expected.loud_per100.tolist(), abs=1e-12)
    assert plotted_line(panel_a, 'silent_per100') == pytest.approx(expected.silent_per100.tolist(), abs=1e-12)
    assert plotted_line(panel_b, 'silent_share_of_failures_pct') == pytest.approx(expected.share.tolist(), abs=1e-12)
    labels = [t.get_text() for t in panel_b.get_xticklabels()]
    assert labels == [f'{figures.AGENCY_LABEL[a]}\n{n:,} failures'
                      for a, n in zip(figures.AGENCIES, expected.loud + expected.silent)]
    # The installed figure data equals this recomputation, including the bootstrap intervals.
    installed = pd.read_csv(MAIN_DATA / 'appendix_silent_vs_loud.csv')
    pd.testing.assert_frame_equal(installed, plotted.reset_index(drop=True), check_dtype=False, atol=1e-9)
    plt.close(fig)


def test_task_figure_uses_sans_labels_outside_frame(tmp_path):
    captured = []
    def save(fig, destination, stem, plotted, caption, source):
        captured.append(fig)
    data = sample(10)
    figures.task_figure(data, tmp_path, tmp_path / 'source.csv', save, {'task': 'Probit regression'})
    fig = captured[0]
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ['Probit regression']
    assert ax.get_yticklabels()[0].get_fontfamily() == ['sans-serif']
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['Chatbot', 'Self-repair', 'Constrained agent']
    # One marker per agency condition and no interval (Federico, 2026-09-25),
    # and one line from the lowest to the highest marker (Federico, 2026-09-29).
    assert gids(ax) == {f'task_{a}' for a in figures.AGENCIES} | {'task_range'}
    plt.close(fig)


def test_task_figure_success_axis_spans_the_points_with_a_small_pad(tmp_path):
    """The success axis starts and ends EDGE_PAD_PT points beyond the extreme markers, as in Figure 4."""
    captured = []
    def save(fig, destination, stem, plotted, caption, source):
        captured.append((fig, plotted))
    data = sample(10)
    data.loc[data.agency.eq('chatbot') & data.rep.lt(5), 'primary'] = False
    data.loc[data.agency.eq('agent'), 'primary'] = True
    figures.task_figure(data, tmp_path, tmp_path / 'source.csv', save, {'task': 'Probit regression'})
    fig, plotted = captured[0]
    ax = fig.axes[0]
    low, high = plotted.success_pct.min(), plotted.success_pct.max()
    assert (low, high) == pytest.approx((50, 100))
    left, right = ax.get_xlim()
    width_pt = ax.get_position().width * fig.get_figwidth() * 72
    gaps = ((low - left) * width_pt / (right - left), (right - high) * width_pt / (right - left))
    assert gaps == pytest.approx((figures.EDGE_PAD_PT, figures.EDGE_PAD_PT), abs=.05)
    # Ticks every ten points inside the axis, and the row bands fill the axis width.
    ticks = ax.get_xticks()
    assert list(ticks) == [50, 60, 70, 80, 90, 100]
    assert all(left <= t <= right for t in ticks)
    for band in ax.patches:
        assert band.get_x() == pytest.approx(left) and band.get_width() == pytest.approx(right - left)
    plt.close(fig)

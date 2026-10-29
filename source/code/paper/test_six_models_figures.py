"""Checks of the PNAS figure settings in code/paper/six_models.py (peer review 2026-10-28, I71 to I77).

The settings are PNAS-only: the AEA figures, built by other scripts, keep the module defaults.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import six_models as six  # noqa: E402
from six_models import ppf, style  # noqa: E402

SETTINGS = [(style, 'AXIS_FLOOR'), (style, 'FIG_W'), (style, 'STYLE_OVERRIDES'), (style, 'AGENCY_LINES'),
            (ppf, 'PROMPT_SERIES'), (ppf, 'PLATFORM_SERIES'), (ppf, 'FAILURE_SERIES'), (ppf, 'SILENT_SHARE_LABEL')]


def contrast(color, other='#FFFFFF'):
    def luminance(code):
        channels = [int(code[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in channels]
        return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2]
    high, low = sorted([luminance(color), luminance(other)], reverse=True)
    return (high + .05) / (low + .05)


@pytest.fixture
def pnas(monkeypatch):
    """Apply the PNAS settings for one test and restore the AEA defaults afterwards."""
    for module, name in SETTINGS:
        monkeypatch.setattr(module, name, getattr(module, name))
    six.pnas_style()


def test_aea_defaults_keep_the_earlier_look():
    assert style.FIG_W == 6.5 and style.AXIS_FLOOR is None and style.STYLE_OVERRIDES == {}
    assert style.AGENCY_LINES['agent'] == style.BLUSH
    assert ppf.FAILURE_SERIES['silent'] is style.SERIES['blush']
    assert ppf.SILENT_SHARE_LABEL == 'Silent share of failures, percent'


def test_pnas_settings_width_font_and_one_accent_meaning(pnas):
    assert style.FIG_W * 2.54 == pytest.approx(17.8)
    assert style.AXIS_FLOOR == 0
    assert style.STYLE_OVERRIDES['font.sans-serif'][0] == 'Helvetica'
    assert style.STYLE_OVERRIDES['mathtext.it'] == 'Helvetica:italic'
    # The accent marks only the constrained agent, and its line reaches 3:1 against white.
    assert contrast(style.BLUSH) < 3 <= contrast(style.ROSE)
    assert style.AGENCY_LINES['agent'] == style.ROSE
    series = [*ppf.PROMPT_SERIES.values(), *ppf.PLATFORM_SERIES.values(), *ppf.FAILURE_SERIES.values()]
    assert all(style.BLUSH not in (s['line'], s['face']) for s in series)
    assert {s['marker'] for s in series}.isdisjoint({'s', 'D'})
    assert ppf.SILENT_SHARE_LABEL == 'Silent share of failures (percent)'


def test_figure_labels_put_the_effort_in_parentheses():
    assert six.label('sonnet') == 'Claude Sonnet 4.6 (medium)'
    assert six.label('luna6') == 'GPT-6 Luna (low)'
    assert six.NAMES['sonnet'] == 'Claude Sonnet 4.6 medium'


def test_log_ticks_use_one_two_three_five():
    assert six.log_ticks(.1251 / six.LOG_PAD, .7585 * six.LOG_PAD) == pytest.approx([.1, .2, .3, .5])
    assert six.log_ticks(2.45 / six.LOG_PAD, 10.74 * six.LOG_PAD) == pytest.approx([2, 3, 5, 10])


def test_side_margins_move_a_tick_label_off_the_edge(pnas):
    with style.figure_style():
        fig, ax = plt.subplots(figsize=(style.FIG_W, 2))
        fig.subplots_adjust(left=.1, right=.995)
        ax.set_xticks([0, 1], ['Chatbot', 'Constrained agent'])
        ax.set_xlim(-.1, 1.02)
        assert style.text_margins(fig)['right'] < six.TEXT_MARGIN_PT
        result = six.check_text(fig, 'probe')
        plt.close(fig)
    assert result['margin_pt']['right'] >= six.TEXT_MARGIN_PT - 1e-3
    assert result['width_cm'] == pytest.approx(17.8)
    assert six.TEXT_SIZE_RANGE_PT[0] <= result['font_pt'][0] <= result['font_pt'][1] <= six.TEXT_SIZE_RANGE_PT[1]


def test_text_check_rejects_a_size_outside_six_to_twelve_points(pnas):
    with style.figure_style():
        fig, ax = plt.subplots(figsize=(style.FIG_W, 2))
        # 5.5 pt: FreeType fails to load Helvetica.ttc glyphs at 5 pt and 100 dpi.
        fig.subplots_adjust(bottom=.3)
        ax.set_xlabel('Too small', fontsize=5.5)
        with pytest.raises(AssertionError, match='text sizes'):
            six.check_text(fig, 'probe')
        plt.close(fig)


def write_checked(folder, data, flip=(), revive=()):
    """A checked runs.csv and alternative_checks.csv in folder, flipping the listed positions."""
    checked = data[['record_path', 'position_id', 'primary', 'failure_category', 'execution_ok',
                    'output_available']].copy()
    checked.loc[checked.position_id.isin(flip), ['primary', 'failure_category']] = [False, 'Silent failure']
    checked.loc[checked.position_id.isin(revive), 'primary'] = True
    checked.to_csv(folder / 'runs.csv', index=False)
    pd.DataFrame(dict(position_id=list(flip), run_changed=[True] * len(flip))).to_csv(
        folder / 'alternative_checks.csv', index=False)


def sample_runs():
    return pd.DataFrame(dict(record_path=[f'r{i}' for i in range(4)], position_id=[f'p{i}' for i in range(4)],
                             primary=[True, True, False, True], failure_category=['', '', 'Loud failure', ''],
                             execution_ok=True, output_available=[True, True, False, True]))


def test_checked_outcome_takes_the_marked_failures(tmp_path):
    data = sample_runs()
    write_checked(tmp_path, data, flip=['p1'])
    checked, turned = six.checked_outcome(data, tmp_path)
    assert turned == 1
    assert checked.primary.tolist() == [True, False, False, True]
    assert checked.failure_category.tolist()[1] == 'Silent failure'


def test_checked_outcome_rejects_unmarked_or_revived_runs(tmp_path):
    data = sample_runs()
    write_checked(tmp_path, data, flip=['p1'])
    marked = pd.read_csv(tmp_path / 'alternative_checks.csv').assign(position_id='p3')
    marked.to_csv(tmp_path / 'alternative_checks.csv', index=False)
    with pytest.raises(ValueError, match='marks'):
        six.checked_outcome(data, tmp_path)
    write_checked(tmp_path, data, revive=['p2'])
    with pytest.raises(ValueError, match='failure into a success'):
        six.checked_outcome(data, tmp_path)


def test_pnas_design_uses_helvetica_and_the_full_width(tmp_path):
    text = six.pnas_design(ppf.DESIGN_TEMPLATE.read_text())
    assert r'\usepackage{helvet}' not in text and r'\setmainfont{Helvetica}' in text
    assert 'text width=50.55mm' in text and 'text width=173.85mm' in text
    with pytest.raises(ValueError, match='design template changed'):
        six.pnas_design(text)
    if shutil.which('lualatex') is None or shutil.which('pdfinfo') is None:
        return
    (tmp_path / 'design.tex').write_text(text)
    subprocess.run(['lualatex', '-halt-on-error', '-interaction=nonstopmode', 'design.tex'], cwd=tmp_path,
                   check=True, capture_output=True)
    info = subprocess.run(['pdfinfo', str(tmp_path / 'design.pdf')], capture_output=True, text=True,
                          check=True).stdout
    width = float(re.search(r'^Page size:\s+([\d.]+) x', info, re.M).group(1))
    assert width / 72 * 2.54 == pytest.approx(17.8, abs=.02)
    fonts = subprocess.run(['pdffonts', str(tmp_path / 'design.pdf')], capture_output=True, text=True).stdout
    assert 'Helvetica' in fonts and 'Nimbus' not in fonts


def test_pass_at_k_figure_labels_the_vertical_axis(pnas, tmp_path, monkeypatch):
    order = ['sonnet', 'luna']
    monkeypatch.setattr(ppf, 'MODELS', order)
    rows = []
    # Success varies by task, so every bootstrap interval brackets its point.
    for t, task in enumerate(('a', 'b', 'c', 'd')):
        for model in order:
            for platform in ppf.PLATFORMS:
                for prompt in ppf.PROMPTS:
                    for a, agency in enumerate(ppf.AGENCIES):
                        for rep in range(3):
                            rows.append(dict(task=task, model=model, platform=platform, prompt=prompt, agency=agency,
                                             rep=rep, primary=rep < 1 + (t + a) % 3))
    captured = []
    def save(fig, destination, stem, plotted, caption, source):
        captured.append(fig)
    six.pass_by_model(pd.DataFrame(rows), tmp_path, save, order)
    fig = captured[0]
    ax = fig.axes[0]
    assert ax.get_ylabel() == 'Pass@$k$ (percent)'
    assert ax.get_xlabel() == 'Runs drawn ($k$)'
    assert list(ax.get_yticks()) == [0, 20, 40, 60, 80, 100]
    assert ax.get_title(loc='left') == 'A. Claude Sonnet 4.6 (medium)'
    lines = [line for line in ax.get_lines() if line.get_linestyle() == '-' and len(line.get_xdata()) == 3]
    assert style.ROSE in {matplotlib.colors.to_hex(line.get_color()).upper() for line in lines}
    six.check_text(fig, 'appendix_pass_at_k')
    plt.close(fig)

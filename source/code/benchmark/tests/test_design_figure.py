"""Check the design figure template and the counts the build fills into it."""
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import paper_figure_style as style
import production_paper_figures as figures


def sample(tasks=('alpha', 'beta'), repetitions=3):
    return pd.DataFrame([dict(task=task, model=model, platform=platform, prompt=prompt, agency=agency,
                              rep=rep, primary=True)
                         for task, model, platform, prompt, agency, rep in itertools.product(
                             tasks, figures.MODELS, figures.PLATFORMS, figures.PROMPTS, figures.AGENCIES,
                             range(repetitions))])


def write_analysis_files(out, tasks_by_path, se_tasks):
    (out / 'data').mkdir(parents=True, exist_ok=True)
    catalog = {task: dict(task_id=task, data_path=path) for path, tasks in tasks_by_path.items() for task in tasks}
    (out / 'data/catalog.json').write_text(json.dumps(catalog))
    (out / 'data/snapshot.json').write_text(json.dumps(dict(se_scoring=dict(tasks=se_tasks))))


def test_template_defines_the_paper_figure_colors():
    text = figures.DESIGN_TEMPLATE.read_text()
    defined = dict(re.findall(r'\\definecolor\{(\w+)\}\{HTML\}\{([0-9A-F]{6})\}', text))
    assert defined == {name: color[1:] for name, color in figures.DESIGN_COLORS.items()}
    assert defined['ink'] == style.INK[1:] and defined['blush'] == style.BLUSH[1:]
    assert 'helvet' in text and 'mathpazo' not in text


def test_counts_from_the_sample_and_the_analysis_files(tmp_path):
    data = sample(tasks=('alpha', 'beta', 'gamma'), repetitions=3)
    write_analysis_files(tmp_path, {'/data/one.csv': ['alpha', 'beta'], '/data/two.csv': ['gamma'],
                                    '/data/other.csv': ['omitted']}, se_tasks=['beta', 'omitted'])
    counts = figures.design_counts(data, tmp_path)
    assert counts == dict(ntasks='3', ncells='54', nreps='3', nruns='324', nrunspermodel='162',
                          ndatasets='2', nshareddatasets='1', nsharedtasks='2', nsetasks='1')


def test_counts_state_ranges_and_thousands_in_latex(tmp_path):
    data = sample(tasks=tuple(f'task{i}' for i in range(40)), repetitions=10)
    data = data[~(data.task.eq('task0') & data.model.eq('sonnet') & data.platform.eq('python')
                  & data.prompt.eq('zero_shot') & data.agency.eq('chatbot') & data.rep.eq(9))]
    counts = figures.design_counts(data, tmp_path)
    assert counts['nruns'] == '14{,}399'
    assert counts['nreps'] == '9--10'
    assert counts['nrunspermodel'] == '7{,}199--7{,}200'
    assert counts['ncells'] == '720'
    assert 'ndatasets' not in counts and 'nsetasks' not in counts


def test_source_replaces_every_macro_and_keeps_the_rest(tmp_path):
    data = sample()
    write_analysis_files(tmp_path, {'/data/one.csv': ['alpha', 'beta']}, se_tasks=['alpha'])
    text = figures.design_source(data, tmp_path)
    assert r'\newcommand{\ntasks}{2}' in text
    assert r'\newcommand{\ndatasets}{1}' in text
    assert r'\newcommand{\nshareddatasets}{1}' in text
    assert r'\newcommand{\nsharedtasks}{2}' in text
    assert r'\newcommand{\nsetasks}{1}' in text
    assert r'\newcommand{\ncells}{36}' in text
    assert r'\newcommand{\nreps}{3}' in text
    assert r'\newcommand{\nruns}{216}' in text
    assert r'\newcommand{\nrunspermodel}{108}' in text
    assert text.count(r'\newcommand{\n') == figures.DESIGN_TEMPLATE.read_text().count(r'\newcommand{\n')
    assert 'Constrained agent' in text and 'Task success' in text


def test_source_rejects_a_template_with_other_colors(tmp_path, monkeypatch):
    broken = tmp_path / 'template.tex'
    broken.write_text(figures.DESIGN_TEMPLATE.read_text().replace('FFAAA7', 'FF0000'))
    monkeypatch.setattr(figures, 'DESIGN_TEMPLATE', broken)
    with pytest.raises(ValueError, match='blush'):
        figures.design_source(sample(), tmp_path)


@pytest.mark.skipif(shutil.which('pdflatex') is None, reason='pdflatex is not installed')
def test_design_figure_compiles_one_page_at_text_width(tmp_path):
    captured = []
    def save(fig, destination, stem, plotted, caption, source):
        captured.append((fig, stem, plotted, caption, fig.read_bytes()))
    data = sample()
    write_analysis_files(tmp_path, {'/data/one.csv': ['alpha', 'beta']}, se_tasks=['alpha'])
    figures.design_figure(data, tmp_path, tmp_path / 'source.csv', save)
    fig, stem, plotted, caption, pdf = captured[0]
    assert stem == 'benchmark_design'
    assert pdf.startswith(b'%PDF')
    assert plotted.iloc[0].to_dict() == dict(tasks=2, models=2, languages=3, prompts=2, agencies=3, cells=36,
        target_reps=10, completed=216, planned=720, minimum_reps=3, maximum_reps=3, datasets='1',
        shared_datasets='1', shared_tasks='2', se_tasks='1')
    assert caption.startswith('Benchmark design for 2 tasks')
    if shutil.which('pdfinfo') is None:
        return
    # One page: the 135.5 mm text width plus the 1 pt border on each side.
    saved = tmp_path / 'design.pdf'
    saved.write_bytes(pdf)
    info = subprocess.run(['pdfinfo', str(saved)], capture_output=True, text=True, check=True).stdout
    assert re.search(r'^Pages:\s+1$', info, re.M)
    width, height = map(float, re.search(r'^Page size:\s+([\d.]+) x ([\d.]+) pts', info, re.M).groups())
    assert width == pytest.approx(135.5 / 25.4 * 72.27 + 2, abs=1.5)
    # The top-down layout of 2026-09-29 is about 306 pt high.
    assert 150 < height < 330

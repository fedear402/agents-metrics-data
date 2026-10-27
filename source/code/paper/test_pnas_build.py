"""Checks for the six-model PNAS renderer and the author's source boundaries."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('pnas_build_under_test', ROOT / 'paper/pnas/build_pnas.py')
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)
BASE = ROOT / 'quality_reports/audits/2026-10-21_pnas-six-models/work/baseline/paper/pnas'


def test_numerical_source_rejects_an_incomplete_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'ANALYSIS', tmp_path)
    (tmp_path / 'numbers.tex').write_text(r'\newcommand{\numRunsTotal}{14,400}')
    with pytest.raises(ValueError, match='43,200'):
        b.load_numbers()


def test_numerical_source_rejects_duplicate_definitions(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'ANALYSIS', tmp_path)
    (tmp_path / 'numbers.tex').write_text(r'\newcommand{\numRunsTotal}{43,200}' * 2)
    with pytest.raises(ValueError, match='Duplicate'):
        b.load_numbers()


def test_table_includes_each_model_and_the_full_sample():
    text = b.table1()
    for model in b.MODEL_ORDER:
        for head in b.TABLE_HEADS[model][:2]:
            assert head in text
    assert '43,200' in text
    assert text.count('7,200') == 6
    assert '(7)' in text


def test_all_active_placeholders_have_generated_values():
    numbers = b.load_numbers()
    text = b.front_matter(ROOT / 'paper/pnas/human_pnas.md')
    text += (ROOT / 'paper/pnas/submission/title_page.yaml').read_text()
    text += (ROOT / 'paper/pnas/submission/cover_letter.md').read_text()
    filled = b.fill_placeholders(text, numbers)
    assert '{{num' not in filled
    assert numbers['numRunsTotal'] == '43,200'
    assert numbers['numModels'] == '6'


def test_figure_denominators_follow_the_selected_sample():
    numbers = b.load_numbers()
    runs = pd.read_csv(b.ANALYSIS / 'data/runs.csv', low_memory=False)
    expected = {
        'numPointRunsPrompt': runs.groupby(['agency', 'prompt']).size(),
        'numPointRunsLanguage': runs.groupby(['agency', 'platform']).size(),
        'numPointRunsFailure': runs.groupby('agency').size(),
        'numPointRunsTask': runs.groupby(['agency', 'task']).size(),
    }
    for name, groups in expected.items():
        assert groups.nunique() == 1
        assert int(numbers[name].replace(',', '')) == groups.iloc[0]
    assert runs.loc[runs.agency.eq('agent'), 'failure_category'].eq('Loud failure').any()
    body = (ROOT / 'paper/pnas/human_pnas.md').read_text()
    assert 'removing every loud failure' not in body
    assert 'with only silent errors remaining' not in body


def test_citations_structure_and_opening_paragraph_are_preserved():
    before = (BASE / 'human_pnas.md').read_text()
    after = (ROOT / 'paper/pnas/human_pnas.md').read_text()
    # Raul's Overleaf edit of 2026-10-25 puts the cost heading in sentence case.
    assert [h.replace('Cost and Runtime', 'Cost and runtime') for h in re.findall(r'(?m)^#+ .*$', before)] == \
        re.findall(r'(?m)^#+ .*$', after)
    # Figure 3 shows the cheap and expensive model groups since 2026-10-22 (Federico's request).
    groups = 'file="cost_of_correctness_groups.pdf"'
    # Raul's Overleaf edits of 2026-10-25: Fig. 1 moves to the main text, and the SI frontier uses a log cost axis.
    design = '::: {#fig:benchmark_design .figure file="benchmark_design.pdf"}'
    blocks = [line.replace(groups, 'file="cost_of_correctness.pdf"').replace('_log.pdf"', '.pdf"')
              .replace(' width=".85"', '') for line in re.findall(r'(?m)^:::.*$', after)]
    # Federico removed Table S1 (the task list with scored outputs) on 2026-10-27.
    tasks_table = '::: {#tab:tasks .longtable widths="5.0cm,8.1cm" head="Task|Scored outputs"}'
    expected = re.findall(r'(?m)^:::.*$', before)
    expected.remove(tasks_table)
    expected.remove(':::')
    assert sorted(expected) == sorted(blocks)
    assert blocks.index(design) < blocks.index('::: {#fig:prompt_agency .figure file="prompt_agency.pdf"}')
    # Table S1, removed on 2026-10-27, held the only citation of callaway2021did.
    cited = [key for key in re.findall(r'@[A-Za-z][A-Za-z0-9_-]+', before) if key != '@callaway2021did']
    assert cited == re.findall(r'@[A-Za-z][A-Za-z0-9_-]+', after)
    assert re.findall(r'(?s)<!--.*?-->', before) == re.findall(r'(?s)<!--.*?-->', after)
    # Raul's Overleaf corrections of 2026-10-24 (commit 05e81c0), carried into the source.
    opening = before.strip().split('\n\n')[0]
    for old, new in [('established) so', 'established), so'),
                     ('reproductions made purely with AI replicate', 'AI-led teams reproduce'),
                     # Raul's Overleaf edits of 2026-10-25 (commits ca2ed7e and f8aae96).
                     ('difficult to detect and include semantic bugs, in which the code runs',
                      'difficult to detect, and they include semantic bugs, in which the code usually runs'),
                     ('a quarter of them, including studies whose',
                      'a quarter of them and can be present even when the')]:
        opening = opening.replace(old, new)
    assert opening == after.strip().split('\n\n')[0]


def test_abstract_has_only_the_required_model_scope_change():
    def abstract(path):
        return re.search(r'(?ms)^abstract: \|\n(.*?)^acknowledgements:', path.read_text()).group(1)
    before = abstract(BASE / 'draft_pnas.md')
    after = abstract(ROOT / 'paper/pnas/draft_pnas.md')
    expected = before.replace('For Claude Sonnet 4.6 and GPT-5.6 Luna,', 'Across {{numModelsWord}} models,')
    # Raul's Overleaf edits of the abstract, 2026-10-25 (commits ca2ed7e and f8aae96).
    for anchor, added in [
            ('revises it).', ' We score {{numRunsTotal}} runs against the known correct output of each task.'),
            ('{{numPooledAgentRounded}} percent.', ' Letting the chatbot revise its script after an error achieves '
             '{{numAllRepairGainShare}} percent of this gain. The agent also narrows the differences between models: '
             'task success ranges from {{numModelChatbotMinRounded}} to {{numModelChatbotMaxRounded}} percent under the '
             'chatbot and from {{numModelAgentMinRounded}} to {{numModelAgentMax}} percent under the agent.'),
            ('chatbot runs', ', but each additional successful run costs only about {{numAllIncrementalCpsCents}} cents'),
            ('wrong estimates.', ' Failures that show up as an error or a missing result fall from '
             '{{numLoudPerHundredChatbotPooled}} to {{numLoudPerHundredAgentPooled}} per 100 runs, while wrong numbers '
             'reported without an error message fall only from {{numSilentPerHundredChatbotPooled}} to '
             '{{numSilentPerHundredAgentPooled}}. Most remaining failures are therefore silent, so researchers who use AI '
             'agents still need to check the estimates.')]:
        assert expected.count(anchor) == 1
        expected = expected.replace(anchor, anchor + added)
    assert after == expected

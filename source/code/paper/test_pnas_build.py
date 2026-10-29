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
    # The 2026-10-29 review fixes rename the Methods heading, which repeated the Results heading.
    assert [h.replace('Cost and Runtime', 'Cost and runtime')
            .replace('runtime {#sec:cost_measures}', 'runtime measures {#sec:cost_measures}')
            for h in re.findall(r'(?m)^#+ .*$', before)] == re.findall(r'(?m)^#+ .*$', after)
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
    # The 2026-10-29 review fixes add Table S4, the inference of the Table 1 contrasts.
    expected += ['::: {#tab:table1_inference .table generate="table1_inference"}', ':::']
    assert sorted(expected) == sorted(blocks)
    assert blocks.index(design) < blocks.index('::: {#fig:prompt_agency .figure file="prompt_agency.pdf"}')
    # Table S1, removed on 2026-10-27, held the only citation of callaway2021did. Raul's Overleaf edit of
    # 2026-10-27 (commit 0ede81c) drops the brain-imaging sentence, the only citation of eklund2016cluster.
    cited = [key for key in re.findall(r'@[A-Za-z][A-Za-z0-9_-]+', before)
             if key not in ('@callaway2021did', '@eklund2016cluster')]
    # The 2026-10-29 review fixes add Bertran, Fogliato, and Wu (PNAS 2026).
    assert cited == [key for key in re.findall(r'@[A-Za-z][A-Za-z0-9_-]+', after) if key != '@bertran2026manyai']
    assert re.findall(r'(?s)<!--.*?-->', before) == re.findall(r'(?s)<!--.*?-->', after)
    # Raul's Overleaf corrections of 2026-10-24 (commit 05e81c0), carried into the source.
    opening = before.strip().split('\n\n')[0]
    for old, new in [('established) so', 'established), so'),
                     ('reproductions made purely with AI replicate', 'AI-led teams reproduce'),
                     # Raul's Overleaf edits of 2026-10-25 (commits ca2ed7e and f8aae96).
                     ('difficult to detect and include semantic bugs, in which the code runs',
                      'difficult to detect, and they include semantic bugs, in which the code usually runs'),
                     ('a quarter of them, including studies whose',
                      'a quarter of them and can be present even when the'),
                     # Raul's Overleaf edit of 2026-10-27 (commit b23f139).
                     ('question, we frequently', 'question, economists frequently'),
                     ('. We then write', '. They then write'),
                     ('assistant or to large language models (LLMs) through a chatbot app.',
                      'assistant or, more recently, to large language models (LLMs).'),
                     # Federico's answers to the 2026-10-28 reviews (2026-10-29).
                     ('question, economists frequently', 'question, social scientists frequently'),
                     ('code in Stata, R, or Python', 'code in Stata (a statistics package), R, or Python'),
                     ('very clearly established', 'clearly established'),
                     ('coding, these may introduce', 'coding, LLMs may introduce')]:
        opening = opening.replace(old, new)
    assert opening == after.strip().split('\n\n')[0]


# Raul's abstract (Overleaf 5d7b778), with the cost and silent-failure wording of the 2026-10-29 review fixes.
RAUL_ABSTRACT = (
    "We study how large language models write econometric code. On 40 tasks in Stata, R, and Python, each with a "
    "known correct answer, we vary whether the *prompt* includes an example and the degree of *agency*, from a chatbot "
    "that only writes the script to an agent that also runs and revises it. Across {{numModelsWord}} models, the agent "
    "raises the share of runs that return the correct answer from {{numPooledChatbotRounded}} to "
    "{{numPooledAgentRounded}} percent. Letting the chatbot revise its script after an error already achieves most of "
    "this gain. The example and the language matter more under the chatbot than under the agent, which also narrows "
    "the differences between models. Agent runs cost {{numAllMeanCostAgencyRatio}} times as much as chatbot runs, but "
    "each additional correct answer costs between \\${{numIncrementalCpsLunaSix}} and \\${{numIncrementalCpsOpus}}, depending "
    "on the model. The agent removes almost all failures in which the script does not run or does not return every "
    "result, but only about half of the silent failures, in which the script runs and returns a wrong value. Most remaining failures are therefore silent, so "
    "researchers who use AI agents still need to check the estimates.")


def test_abstract_is_the_authors_text():
    # Raul's abstract of 2026-10-27 (Overleaf commit 5d7b778). Only the authors change it.
    text = (ROOT / 'paper/pnas/draft_pnas.md').read_text()
    abstract = re.search(r'(?ms)^abstract: \|\n(.*?)^acknowledgements:', text).group(1)
    assert abstract == '  ' + RAUL_ABSTRACT + '\n'


def test_section_references_name_the_section():
    # A bare title read as a noun of the sentence ("As shown in Failures,"; peer review 2026-10-28, I59).
    text = r'As shown in \Cref{sec:a}, it holds. \Cref{sec:b} shows it (see \cref{sec:c}). \Cref{fig:x} stays.'
    assert b.section_names(text) == (
        r'As shown in the section \emph{\nameref{sec:a}}, it holds. The section \emph{\nameref{sec:b}} shows it '
        r'(see the section \emph{\nameref{sec:c}}). \Cref{fig:x} stays.')


def test_table_numbers_print_in_the_table_font():
    # Math mode set the estimates in serif digits next to sans standard errors (peer review 2026-10-28, I74).
    for name in ('table1', 'robustness', 'robustness_inference', 'robustness_scoring'):
        assert '$' not in b.TABLES[name](), name
    assert b.text_math(r'[${-}6.5$ & $10^{-3}$ & One-shot $\times$') == \
        r'[\textminus{}6.5 & 10\textsuperscript{\textminus{}3} & One-shot \texttimes{}'
    with pytest.raises(ValueError, match='Unexpected math'):
        b.text_math(r'$\alpha$')
    assert b.equal_p(0.0004) == r'\textless{}0.001' and b.equal_p(0.019) == '0.019'


def test_table1_labels_keep_estimates_next_to_their_errors():
    text = b.table1()
    assert r'\quad Self-repair & ' in text
    assert '\n' + r'\quad\hspace{1em}(revises on error or missing value) & (' in text
    assert r'\hangindent=2em' in text


def test_robustness_panels_are_separated_by_a_rule_and_space():
    # PNAS asks authors to avoid multipart tables; the two panels have different columns (I75).
    text = b.robustness()
    first, second = text.split(r'\par\vspace{6pt}')
    assert first.rstrip().endswith(r'\midrule' + '\n' + r'\end{tabular*}')
    assert second.lstrip().startswith(r'\begin{tabular*}')
    assert 'Panel B' in second and 'Panel A' in first
    for name in ('robustness_inference', 'robustness_scoring'):
        part = b.TABLES[name]()
        assert 'Panel' not in part and r'\toprule' in part and r'\bottomrule' in part

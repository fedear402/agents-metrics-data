"""One method-only label per task serves Figure 3, the appendix tables, and the text."""
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import production_paper_assets as assets
from task_labels import TASK_LABELS, tex_label

CATALOG = ROOT / 'data/analysis/main/data/catalog.json'
APPENDIX = ROOT / 'paper/manuscript/appendix/A_tasks_scoring.tex'


def test_labels_are_unique_and_shared_with_figures():
    assert len(set(TASK_LABELS.values())) == len(TASK_LABELS)
    for task, label in TASK_LABELS.items():
        assert assets.SHORT_LABELS[task] == label


def test_labels_name_methods_not_simulated_settings():
    setting_words = re.compile(r'trade|employment|wage|expenditure|stock|training|village|nonrecipient|flows', re.I)
    assert not [label for label in TASK_LABELS.values() if setting_words.search(label)]


@pytest.mark.skipif(not CATALOG.is_file(), reason='main analysis catalog not built')
def test_every_paper_task_has_a_label():
    assert set(json.loads(CATALOG.read_text())) <= set(TASK_LABELS)


@pytest.mark.skipif(not APPENDIX.is_file(), reason='manuscript checkout absent')
def test_appendix_task_tables_use_the_labels_in_order():
    text = APPENDIX.read_text()
    tables = re.findall(r'\\begin\{longtable\}.*?\\end\{longtable\}', text, re.S)
    assert len(tables) == 2
    expected = sorted((tex_label(task) for task in TASK_LABELS), key=str.lower)
    for table in tables:
        body = table[table.index(r'\endhead') + len(r'\endhead'):table.index(r'\bottomrule')]
        names = [line.split(' & ')[0] for line in body.splitlines()
                 if ' & ' in line and line.split(' & ')[0] in expected]
        assert names == expected

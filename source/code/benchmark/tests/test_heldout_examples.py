"""Held-out few-shot example set (type 2 examples) of the final bank.

These tests read the stored files only. `heldout_examples.py run` and `check` run the scripts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import final_bank_examples as fe  # noqa: E402
import final_bank_prompts as fp  # noqa: E402
import heldout_examples as he  # noqa: E402

SET = he.load_set()
POOL = list(dict.fromkeys(SET['examples'] + [r['by'] for r in SET['replacements'].values()]))


def test_set_has_three_distinct_examples_for_every_task():
    for folder in fp.package_dirs():
        ids = he.examples_for(folder.name)
        assert len(ids) == 3 and len(set(ids)) == 3
    assert he.examples_for('vif') == ['rent_distribution', 'return_normality', 'commute_quantiles']


def test_every_example_has_three_languages_and_stored_outputs_in_its_own_format():
    for example_id in POOL:
        spec = he.load_example(example_id)
        for lang in fp.LANGS:
            assert (he.SET_DIR / example_id / fp.EXAMPLE_FILES[lang]).is_file()
            stored = (he.SET_DIR / example_id / fp.EXPECTED_FILES[lang]).read_text()
            run = {'output': stored, 'raw': None if lang == 'stata' else json.loads(stored)}
            rows, header = fe.parse_rows(lang, run)
            assert fe.format_problems(spec, lang, rows, header) == []


def test_overlap_table_covers_the_bank_and_no_overlapping_example_is_shown():
    assert he.overlap_problems() == []
    table = json.loads((he.SET_DIR / 'overlap.json').read_text())
    assert set(table['tasks']) == {f.name for f in fp.package_dirs()}


def test_validation_passed_for_every_example():
    validation = json.loads((he.SET_DIR / 'validation.json').read_text())
    assert validation['ok'] and validation['overlap_problems'] == []
    assert [e['example'] for e in validation['examples']] == POOL
    for example in validation['examples']:
        assert example['problems'] == []
        assert example['leak_check']['tasks'] == len(fp.package_dirs())
        assert example['leak_check']['tasks_with_leaks'] == []


def test_text_check_finds_a_role_mapping_and_a_bank_column():
    spec = he.load_example('rent_distribution')
    spec = dict(spec, context=spec['context'] + ' flat plays the role of id.', columns=['flat', 'income'])
    problems = he.text_problems(spec, {}, he.bank_columns(), ['ols_robust'])
    assert any('role of' in p for p in problems)
    assert any('income' in p for p in problems)

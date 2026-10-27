"""Offline checks of the decision versions (decision record 2026-09-29_decision-versions).

Covers code/benchmark/decision_versions.py and the packages in data/task_candidates/2026-09-29_decision/.
No test calls a provider or runs an interpreter.
"""
from pathlib import Path
import json
import re
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import decision_versions as dv  # noqa: E402
import final_bank_prompts as fp  # noqa: E402
import final_bank_scorer as fs  # noqa: E402

FOLDERS = fp.decision_dirs()
IDS = [f.name for f in FOLDERS]


def test_seven_decision_versions_and_four_drops():
    """Decision records 2026-09-29_decision-versions and 2026-09-30_hand-audit-fixes (arima_forecast, ttest_welch)."""
    assert IDS == sorted(dv.decision_id(t) for t in dv.SPECS)
    assert len(IDS) == 7
    assert set(dv.DROPPED) == {'ols_robust', 'cluster_ols', 'arima_forecast', 'ttest_welch'}
    assert set(dv.DROPPED_SPECS) == {'arima_forecast', 'ttest_welch'}
    assert not set(dv.DROPPED) & set(dv.SPECS)
    assert all(not (dv.DECISION_DIR / dv.decision_id(t)).exists() for t in dv.DROPPED)
    assert set(dv.OUTPUT_ORDER) == set(dv.SPECS) | set(dv.DROPPED_SPECS)


def test_every_bank_task_has_a_task_type():
    types = {f.name: fp.load_task(f).get('task_type') for f in fp.all_package_dirs()}
    assert set(types.values()) == {'specified', 'decision'}
    specified_bank = {k: v for k, v in types.items() if not k.endswith(dv.SUFFIX)}
    assert len(specified_bank) == 48
    assert {k for k, v in specified_bank.items() if v == 'decision'} == {
        'poisson_offset', 'codebook_cohort', 'codebook_wellbeing'}
    assert all(types[i] == 'decision' for i in IDS)


@pytest.mark.parametrize('folder', FOLDERS, ids=IDS)
def test_same_data_and_reference_as_the_specified_task(folder):
    task = fp.load_task(folder)
    src = dv.specified_dir(task['specified_task'])
    assert task['task_id'] == task['specified_task'] + dv.SUFFIX
    for name in task['files']:
        assert (folder / 'data' / name).read_bytes() == (src / 'data' / name).read_bytes()
    ref = json.loads((folder / 'reference.json').read_text())['route_a_values']
    assert ref == json.loads((src / 'reference.json').read_text())['route_a_values']
    spec = fs.load_spec(folder)
    src_checks = {(c['term'], c['field']): c['reference'] for c in fs.load_spec(src)['checks']}
    for c in spec['checks']:
        assert src_checks[(c['term'], c['field'])] == c['reference']
    assert fs.score_rows(ref, spec)['passed']
    assert fs.validate_spec(spec) == []


@pytest.mark.parametrize('folder', FOLDERS, ids=IDS)
def test_three_checks_pass_and_are_current(folder):
    task = fp.load_task(folder)
    stored = json.loads((folder / 'decision_checks.json').read_text())
    fresh = json.loads(json.dumps(fs.jsonable(dv.checks(task['specified_task']))))
    assert stored == fresh
    assert stored['ok'] and stored['fact_check']['ok'] and stored['class_check']['ok'] and stored['word_check']['ok']
    assert stored['fact_check']['variants'], 'every decision version needs wrong-by-fact variants'
    assert len(stored['class_check']['accepted']) <= dv.MAX_ACCEPTED
    names = [c['name'] for c in json.loads((folder / 'scoring.json').read_text()).get('conventions', [])]
    assert names == stored['class_check']['accepted']


@pytest.mark.parametrize('folder', FOLDERS, ids=IDS)
def test_prompts_have_no_few_shot_and_no_decision_word(folder):
    task = fp.load_task(folder)
    assert fp.prompt_levels(task) == ('zero_shot',)
    assert task.get('few_shot_note')
    names = sorted(p.name for p in (folder / 'prompts').glob('*.txt'))
    assert names == sorted(fp.render_package(folder))
    assert len(names) == 9 and all('__zero_shot__' in n for n in names)
    assert not (folder / 'examples').exists()
    words = [re.compile(w, re.I) for w in task['decision_words']]
    zero = fp.zero_shot_prompt(task, 'python', 'chatbot')
    assert task['specified_task'] not in zero and task['output_task_id'] in zero
    for lang in fp.LANGS:
        for condition in fp.CONDITIONS:
            text = (folder / 'prompts' / fp.prompt_name(lang, 'zero_shot', condition)).read_text()
            assert [w.pattern for w in words if w.search(text)] == []
            assert fp.prompt_problems(text) == []


def test_output_task_ids_are_distinct_and_neutral():
    ids = [fp.load_task(f)['output_task_id'] for f in FOLDERS]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r'task_d\d\d', i) for i in ids)

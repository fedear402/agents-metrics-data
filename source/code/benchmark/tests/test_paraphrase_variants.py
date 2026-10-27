"""Tests of code/benchmark/paraphrase_variants.py (decision record 2026-10-02_prompt-paraphrase-variants).

Covers the stored p1 and p2 paraphrases of the 48 specified tasks, the mechanical check (it must find a lost or
added number, a lost or added identifier, and a lost formula), and the renderer (only the task text changes, and the
bank prompt files stay as they are).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_bank_prompts as fp  # noqa: E402
import paraphrase_variants as pv  # noqa: E402

PACKAGES = fp.package_dirs()
IDS = [p.name for p in PACKAGES]


def test_every_task_has_both_variants():
    assert len(PACKAGES) == 48
    for task_id in IDS:
        for variant in pv.VARIANTS:
            record = pv.load_variant(task_id, variant)
            assert record['task_id'] == task_id and record['variant'] == variant
            assert record['task_text'].strip() == record['task_text'] and '\n' not in record['task_text']


@pytest.mark.parametrize('folder', PACKAGES, ids=IDS)
def test_stored_paraphrases_pass_the_mechanical_check(folder):
    task = fp.load_task(folder)
    texts = set()
    for variant in pv.VARIANTS:
        result = pv.check_variant(task, variant)
        assert result['errors'] == [], result
        texts.add(pv.load_variant(task['task_id'], variant)['task_text'])
    assert len(texts) == 2 and task['task_text'] not in texts


def _errors_after(monkeypatch, task, variant, mutate):
    record = pv.load_variant(task['task_id'], variant)
    changed = dict(record, task_text=mutate(record['task_text']))
    monkeypatch.setattr(pv, 'load_variant', lambda t, v: changed)
    return pv.check_variant(task, variant)['errors']


def test_check_finds_a_lost_and_an_added_number(monkeypatch):
    task = fp.load_task(fp.package_dir('rdd_sharp'))
    errors = _errors_after(monkeypatch, task, 'p1', lambda t: t.replace('bandwidth 10', 'bandwidth ten'))
    assert any('number' in e for e in errors)
    errors = _errors_after(monkeypatch, task, 'p1', lambda t: t + ' Use 3 digits.')
    assert any('new numbers' in e for e in errors)


def test_check_finds_a_lost_and_an_added_identifier(monkeypatch):
    task = fp.load_task(fp.package_dir('broad_cohort_stack'))
    errors = _errors_after(monkeypatch, task, 'p2', lambda t: t.replace('att_weighted', 'the weighted average'))
    assert any('missing identifiers' in e for e in errors)
    errors = _errors_after(monkeypatch, task, 'p2', lambda t: t + ' Use cluster_id.')
    assert any('new identifiers' in e for e in errors)


def test_check_finds_a_lost_formula_and_a_new_hint_word(monkeypatch):
    task = fp.load_task(fp.package_dir('applied_cumulative_hac'))
    errors = _errors_after(monkeypatch, task, 'p1', lambda t: t.replace('N/(N-K)', 'N over N minus K'))
    assert any('missing expressions' in e for e in errors)
    errors = _errors_after(monkeypatch, task, 'p1', lambda t: t + ' Use two lags.')
    assert any('new number words' in e for e in errors)


@pytest.mark.parametrize('folder', PACKAGES[:3] + PACKAGES[-3:], ids=IDS[:3] + IDS[-3:])
def test_render_changes_only_the_task_text(folder):
    task = fp.load_task(folder)
    rendered = pv.render_variants(folder)
    assert len(rendered) == 2 * len(fp.LANGS) * len(fp.CONDITIONS)
    for variant in pv.VARIANTS:
        para = pv.load_variant(task['task_id'], variant)['task_text']
        for lang in fp.LANGS:
            for condition in fp.CONDITIONS:
                text = rendered[f'{lang}__zero_shot_{variant}__{condition}.txt']
                bank = (folder / 'prompts' / fp.prompt_name(lang, 'zero_shot', condition)).read_text()
                assert text.count(para) == 1
                assert text.replace(para, task['task_text']) == bank
                assert fp.prompt_problems(text) == []


def test_store_is_separate_from_the_bank():
    assert not str(pv.STORE).startswith(tuple(str(b) for b in fp.BANK_DIRS))
    for folder in PACKAGES:
        assert not list((folder / 'prompts').glob('*_p1__*')) and not list((folder / 'prompts').glob('*_p2__*'))


def test_stored_records_match_the_bank_text_hash():
    for folder in PACKAGES:
        task = fp.load_task(folder)
        for variant in pv.VARIANTS:
            record = json.loads(pv.variant_path(task['task_id'], variant).read_text())
            assert record['source_task_text_sha256'] == pv.sha256(task['task_text'])

"""Offline checks of the final-bank prompts and few-shot examples.

Covers code/benchmark/final_bank_prompts.py (one renderer for the 48 packages) and the stored
outputs of code/benchmark/final_bank_examples.py. No test runs an example script or calls a provider.
"""
from pathlib import Path
import json
import math
import re
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import final_bank_prompts as fp  # noqa: E402
import final_bank_examples as fe  # noqa: E402

PACKAGES = fp.package_dirs()
IDS = [p.name for p in PACKAGES]


def written(folder, lang, prompt, condition):
    return (folder / 'prompts' / fp.prompt_name(lang, prompt, condition)).read_text()


def test_bank_has_48_packages_in_two_folders():
    # Decision record 2026-09-29_hard-bank-final-changes: anova_oneway and broad_ar_confidence_set dropped,
    # weibull_stock_entry (task 50) added to the hard folder.
    assert len(PACKAGES) == 48
    assert len(set(IDS)) == 48
    assert sum(p.parent == fp.BANK_DIRS[0] for p in PACKAGES) == 38
    assert sum(p.parent == fp.BANK_DIRS[1] for p in PACKAGES) == 10
    assert 'weibull_stock_entry' in IDS
    assert not {'anova_oneway', 'broad_ar_confidence_set'} & set(IDS)


@pytest.mark.parametrize('folder', PACKAGES, ids=IDS)
def test_written_prompts_equal_the_renderer(folder):
    rendered = fp.render_package(folder)
    assert len(rendered) == len(fp.LANGS) * len(fp.PROMPTS) * len(fp.CONDITIONS) == 18
    for name, text in rendered.items():
        assert (folder / 'prompts' / name).read_text() == text, name


@pytest.mark.parametrize('folder', PACKAGES, ids=IDS)
def test_task_information_is_identical_across_languages_and_conditions(folder):
    info = fp.task_information(fp.load_task(folder))
    blocks = set()
    for lang in fp.LANGS:
        for prompt in fp.PROMPTS:
            for condition in fp.CONDITIONS:
                text = written(folder, lang, prompt, condition)
                assert text.count(info) == 1
                start = text.index('Data files in the working directory:')
                end = text.index('\n\n', text.index('\n\nTask: ', start) + 2)
                blocks.add(text[start:end])
    assert blocks == {info}


@pytest.mark.parametrize('folder', PACKAGES, ids=IDS)
def test_every_named_column_exists_in_the_data_and_every_column_is_named(folder):
    task = fp.load_task(folder)
    assert len(task['files']) == len(task['file_descriptions'])
    for name, description in zip(task['files'], task['file_descriptions']):
        assert description.startswith(name + ':')
        header = fp.csv_header(folder / 'data' / name)
        named = fp.described_columns(description)
        assert named, name
        assert [c for c in named if c not in header] == [], name
        assert [c for c in header if c not in named] == [], name


@pytest.mark.parametrize('folder', PACKAGES, ids=IDS)
def test_prompt_text_rules(folder):
    for lang in fp.LANGS:
        for prompt in fp.PROMPTS:
            for condition in fp.CONDITIONS:
                text = written(folder, lang, prompt, condition)
                assert fp.prompt_problems(text) == [], (lang, prompt, condition)
                assert 'synthetic' not in text.lower()
                if lang == 'stata':
                    assert '%24.16g' in text and '%21.12f' not in text
                    if condition == 'agent':
                        assert 'Run the do-file with stata-se -b do run.do' in text


def test_problem_detector_finds_paths_banned_words_and_missing_sentences():
    text = 'The data are synthetic. Run /Applications/StataNow/stata-se. The oracle gates.'
    problems = fp.prompt_problems(text)
    assert any('absolute path' in p for p in problems)
    assert any('banned word' in p for p in problems)
    assert sum('standard sentence' in p for p in problems) == 3
    assert fp.BANNED.search('aggregate and investigate') is None
    assert fp.BANNED.search('armed forces') is None


def test_described_columns_reads_both_layouts():
    assert fp.described_columns('a.csv: one row per unit. Columns: id (unit), x1 and x2 (regressors), '
                                'z1 to z3 (instruments), y (outcome; blank when missing).') == [
        'id', 'x1', 'x2', 'z1', 'z2', 'z3', 'y']
    assert fp.described_columns('b.csv: a panel. Codebook. pid: person identifier. wave: 1, 2, or 3. '
                                'sex: 1 male, 2 female.') == ['pid', 'wave', 'sex']


def test_insert_and_remove_block_are_inverse():
    zero = 'Role line.\n\nData files here.\n\nTask: do it.\n'
    few = fp.insert_block(zero, 'BLOCK\nlines')
    assert few == 'Role line.\n\nBLOCK\nlines\n\nData files here.\n\nTask: do it.\n'
    assert fp.remove_block(few, 'BLOCK\nlines') == zero


# Few-shot prompts and examples ----------------------------------------------------------------

@pytest.mark.parametrize('folder', PACKAGES, ids=IDS)
def test_few_shot_prompt_is_zero_shot_plus_the_block_in_one_place(folder):
    for lang in fp.LANGS:
        parts, code, expected = fp.read_example(folder, lang)
        block = fp.example_block(parts, code, expected, lang)
        for condition in fp.CONDITIONS:
            zero = written(folder, lang, 'zero_shot', condition)
            few = written(folder, lang, 'few_shot', condition)
            assert few.count(block) == 1
            assert few.startswith(fp.role_line(lang) + '\n\n' + block + '\n\n')
            assert fp.remove_block(few, block) == zero
            assert len(few) == len(zero) + len(block) + 2


@pytest.mark.parametrize('folder', PACKAGES, ids=IDS)
def test_example_folder_is_complete_and_validated(folder):
    ex = folder / 'examples'
    for name in list(fp.EXAMPLE_FILES.values()) + list(fp.EXPECTED_FILES.values()) + [
            'block_parts.json', 'validation.json']:
        assert (ex / name).is_file(), name
    report = json.loads((ex / 'validation.json').read_text())
    assert report['ok'] is True, report['problems']
    assert report['leak_check']['leaks'] == 0
    assert report['agreement']['rel_diff'] <= report['agreement']['limit'] <= 1e-3
    assert fp.parts_problems(json.loads((ex / 'block_parts.json').read_text())) == []


def stored_rows(folder, lang):
    text = (folder / 'examples' / fp.EXPECTED_FILES[lang]).read_text()
    if lang == 'stata':
        return fe.parse_rows('stata', {'output': text})
    raw = json.loads(text)
    return fe.parse_rows(lang, {'output': text, 'raw': raw})


@pytest.mark.parametrize('folder', PACKAGES, ids=IDS)
def test_stored_outputs_have_the_task_format_agree_and_do_not_leak(folder):
    task = fp.load_task(folder)
    refs = fe.reference_values(folder, task)
    assert refs
    values, by_lang = [], {}
    for lang in fp.LANGS:
        rows, header = stored_rows(folder, lang)
        assert fe.format_problems(task, lang, rows, header) == [], lang
        by_lang[lang] = {r['term']: r for r in rows}
        for r in rows:
            for field in ('estimate', 'se', 'n_obs'):
                if r.get(field) is not None:
                    values.append((f'{lang}:{r["term"]}.{field}', float(r[field])))
    leaks, closest = fe.value_leaks(values, refs)
    assert leaks == []
    assert closest['ratio'] > fe.LEAK_FACTOR
    limit = float(json.loads((folder / 'examples/block_parts.json').read_text()).get('agreement_rtol', fe.AGREE_RTOL))
    for row in task['output_rows']:
        for field in ('estimate', 'se', 'n_obs'):
            vals = [by_lang[lang][row['term']].get(field) for lang in fp.LANGS]
            if all(v is None for v in vals):
                continue
            assert all(v is not None for v in vals)
            assert max(fe.rel_diff(float(a), float(b)) for a in vals for b in vals) <= limit, (row['term'], field)


@pytest.mark.parametrize('folder', PACKAGES, ids=IDS)
def test_examples_use_no_task_data_and_carry_comments(folder):
    task = fp.load_task(folder)
    refs = fe.reference_values(folder, task)
    parts = json.loads((folder / 'examples/block_parts.json').read_text())
    prose = ' '.join([parts['title'], parts['context'], parts['explanation']]
                     + list((parts.get('platform_notes') or {}).values()))
    assert fp.BANNED.search(prose) is None
    assert '—' not in prose
    for lang in fp.LANGS:
        code = (folder / 'examples' / fp.EXAMPLE_FILES[lang]).read_text()
        assert fe.FILE_READER.search(code) is None, lang
        assert 'uniform_stream' in code, lang
        for name in task['files']:
            assert name not in code
        assert sum(bool(fe.COMMENT[lang].match(line)) for line in code.splitlines()) >= fe.MIN_COMMENTS
        for label, value in refs:
            if abs(value) >= 1e-3 and not float(value).is_integer():
                assert not fe.literal_matches(code, value), (lang, label)
                assert not fe.literal_matches(prose, value), label


def test_value_leak_rule_uses_ten_tolerances():
    refs = [('b.estimate', 2.0)]
    assert fe.value_leaks([('x', 2.19)], refs)[0]
    assert fe.value_leaks([('x', 2.21)], refs)[0] == []
    assert fe.value_leaks([('x', 1e-7)], [('t', 0.0)])[0]


def test_same_output_compares_numbers_at_the_stored_precision():
    a = 'ANSWER: b estimate=1.2345678901 se=. n_obs=10\n'
    assert fe.same_output(a, 'ANSWER: b estimate=1.23456789011 se=. n_obs=10\n')
    assert not fe.same_output(a, 'ANSWER: b estimate=1.2346 se=. n_obs=10\n')
    assert not math.isnan(fe.rel_diff(0.0, 0.0))
    assert re.match(fe.ANSWER, 'ANSWER: x estimate=   -.5 se=. n_obs=         12')


def test_version_line_names_the_manifest_versions_and_no_small_package():
    # Decision record 2026-09-29_version-line.
    manifest = json.loads((fp.ROOT / 'code/benchmark/collection_environment/environment_manifest.json').read_text())
    mm = lambda v: '.'.join(str(v).split('.')[:2])  # noqa: E731
    py = manifest['python']
    assert f"Python {mm(py['python'])}," in fp.VERSION_LINE['python']
    for name in fp.VERSION_PACKAGES['python']:
        assert f"{name} {mm(py['checks'][name]['version'])}" in fp.VERSION_LINE['python']
    r_version = re.search(r'R version (\d+\.\d+)', manifest['r']['r']).group(1)
    assert f'R {r_version} ' in fp.VERSION_LINE['r']
    for name in fp.VERSION_PACKAGES['r']:
        assert f"{name} {mm(manifest['r']['checks'][name]['version'])}" in fp.VERSION_LINE['r']
    for text in fp.VERSION_LINE.values():
        assert 'py4etrics' not in text and 'ivmodels' not in text
        assert 'sandwich' not in text and 'fixest' not in text
    assert 'stata' not in fp.VERSION_LINE
    for folder in PACKAGES:
        for lang in fp.LANGS:
            for prompt in fp.PROMPTS:
                text = written(folder, lang, prompt, 'chatbot')
                if lang == 'stata':
                    assert 'Use StataNow 19.5.' in text and 'The installed versions are' not in text
                else:
                    assert text.count(fp.VERSION_LINE[lang]) == 1
                    assert text.index(fp.LANG_LINE[lang]) < text.index(fp.VERSION_LINE[lang])

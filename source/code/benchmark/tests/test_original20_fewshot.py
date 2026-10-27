"""Tests for the rerun workflow of the original tasks, the 12 added tasks, and flagged runs.

No test calls a model provider or the central collector. The write tests
replace the collector subprocess with a local stub that writes a manifest.
"""
from __future__ import annotations

from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
FOLDER = ROOT / 'code/benchmark/workflows/original20_fewshot_2026_09_25'
sys.path.insert(0, str(FOLDER))
import original20_plan as p  # noqa: E402

c = p.c


def load(name):
    spec = importlib.util.spec_from_file_location(f'original20_{name}_tested', FOLDER / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load('prepare')
check = load('check')
launch = load('launch')

ANSWER = re.compile(r'^ANSWER:\s+(\S+)\s+estimate=(\S+)', re.MULTILINE)
SCOPE32 = 'original20_added12_both_prompts'
COMBINED = 'original20_added12_both_prompts_and_flagged_runs'


def stub_printed_estimates(text):
    return {term: dict(estimate=float(value)) for term, value in ANSWER.findall(text)}, []


def record(path, status, scope=None, run_list_sha256=None):
    lines = ['# Test record', '', f'Status: {status}']
    if scope:
        lines.append(f'Approved scope: {scope}')
    if run_list_sha256:
        lines.append(f'Approved run list sha256: {run_list_sha256}')
    path.write_text('\n'.join(lines) + '\n')
    return path


def baseline_ids(task=None, limit=None):
    with p.BASELINE.open() as stream:
        ids = [row['position_id'] for row in csv.DictReader(stream) if task is None or row['task'] == task]
    return ids[:limit] if limit else ids


# Flagged positions of a test run list: two positions of full-layout tasks and one of an original task.
FLAGGED = sorted([baseline_ids('broad_tobit')[3], baseline_ids('hard_rolling')[-1], baseline_ids('logit')[7]])


# Design -----------------------------------------------------------------------------------

@pytest.mark.parametrize('scope, positions, per_block, tasks', [
    ('few_shot', 3600, 900, 20), ('few_shot_and_zero_shot', 7200, 1800, 20), (SCOPE32, 11520, 2880, 32)])
def test_design_counts(scope, positions, per_block, tasks):
    counts = p.expected_counts(scope)
    assert counts['positions'] == counts['task_positions'] == positions and counts['blocks'] == 4
    assert counts['flagged_added'] == 0 and set(counts['per_block'].values()) == {per_block}
    jobs = p.plan_jobs(scope)
    assert jobs == p.plan_jobs(scope)  # The shuffle is seeded.
    assert len(jobs) == positions == len({p.slot(job) for job in jobs})
    cells = Counter(p.slot(job)[:5] for job in jobs)
    assert len(cells) == counts['cells'] and set(cells.values()) == {10}
    assert Counter(job['block'] for job in jobs) == {p.block_name(m, o, scope): per_block
                                                      for m in p.MODELS for o in (0, 5)}
    assert {job['task'] for job in jobs} == set(p.scope_tasks(scope)) and len(p.scope_tasks(scope)) == tasks
    assert {job['prompt'] for job in jobs} == set(p.scope_prompts(scope))
    for batch in {job['block'] for job in jobs}:
        block = [job for job in jobs if job['block'] == batch]
        assert {job['rep'] for job in block} == set(range(5))
        assert len({job['model'] for job in block}) == 1
        protocol = dict(tasks=dict.fromkeys(p.ALL_TASKS, {}),
                        prompts=dict.fromkeys(p.prompt_keys(tasks=p.ALL_TASKS), 'x'))
        c.validate_jobs(block, protocol)


def test_original_scopes_keep_their_names_and_jobs():
    assert p.block_name('luna', 0) == 'original20_luna_0' and p.TASKS == p.ORIGINAL_TASKS
    assert p.meta('few_shot') == p.meta('few_shot_and_zero_shot') == p.COLLECTIONS / '2026-09-25_original20-fewshot'
    assert p.raw('few_shot') == p.RUNS / 'original20_fewshot_2026-09-25'
    assert len({p.meta(scope) for scope in p.SCOPES}) == 4
    assert set(p.RERUN32_TASKS) == set(p.ORIGINAL_TASKS) | set(p.ADDED12_TASKS) and len(p.RERUN32_TASKS) == 32
    assert set(p.ADDED20_TASKS) | set(p.ORIGINAL_TASKS) == set(p.read(p.CATALOG)) and len(p.ALL_TASKS) == 40


def test_flagged_scopes():
    alone = p.plan_jobs('flagged_runs', FLAGGED)
    assert sorted(p.position_id(job) for job in alone) == FLAGGED
    counts = p.expected_counts('flagged_runs', FLAGGED)
    assert counts['positions'] == counts['flagged_added'] == counts['flagged_listed'] == 3
    assert counts['task_positions'] == 0 and sum(counts['per_block'].values()) == 3
    assert all(batch.startswith('flagged_') for batch in counts['per_block'])
    assert all(job['rep'] == job['global_rep'] % 5 for job in alone)
    # The combined scope adds only the listed positions outside the 32 tasks.
    combined = p.expected_counts(COMBINED, FLAGGED)
    assert combined['task_positions'] == 11520 and combined['flagged_added'] == 2
    assert combined['positions'] == 11522 and combined['flagged_listed'] == 3
    assert p.expected_counts('flagged_runs', ()) == dict(positions=0, task_positions=0, flagged_listed=0,
                                                        flagged_added=0, cells=0, blocks=0, per_block={},
                                                        prompt_keys=0)
    assert p.plan_tasks(COMBINED, FLAGGED) == tuple(sorted(set(p.RERUN32_TASKS) | {'broad_tobit', 'hard_rolling'}))
    for job in p.plan_jobs(COMBINED, FLAGGED):
        outside = job['task'] in ('broad_tobit', 'hard_rolling')
        assert (p.prompt_source(COMBINED, job) == 'production') == outside
    assert {p.prompt_source('flagged_runs', job) for job in alone} == {'production'}


def test_read_run_list(tmp_path):
    with p.BASELINE.open() as stream:
        row = next(r for r in csv.DictReader(stream) if r['task'] == 'broad_oaxaca')
    text = ('# Flagged runs\n\n| position | note |\n|---|---|\n'
            f'| `{FLAGGED[0]}` | skill read |\n| {FLAGGED[1]} | config |\n'
            f'Record: {row["record_path"]}\n'
            'Unknown: logit__luna__python__few_shot__chatbot__99 and prose with task__x words.\n')
    path = tmp_path / 'runs.md'
    path.write_text(text)
    positions, sha256, problems = p.read_run_list(path)
    assert positions == sorted([FLAGGED[0], FLAGGED[1], row['position_id']])
    assert sha256 == c.sha(path)
    assert problems == ['run list names logit__luna__python__few_shot__chatbot__99, which is not a selected position']
    assert p.read_run_list(tmp_path / 'missing.md') == ([], None, [f"run list missing: {tmp_path / 'missing.md'}"])
    empty = tmp_path / 'empty.md'
    empty.write_text('No runs yet.\n')
    assert p.read_run_list(empty)[2] == [f'run list {empty} names no selected position']
    # A CSV with a position_id column: only that column counts, not ids in other columns.
    table = tmp_path / 'runs.csv'
    table.write_text(f'position_id,reason,note\n{FLAGGED[0]},skill_read,see {FLAGGED[1]}\n'
                     f'{FLAGGED[2]},hard-coded,\nlogit__luna__python__few_shot__chatbot__99,x,\n')
    positions, sha256, problems = p.read_run_list(table)
    assert positions == sorted([FLAGGED[0], FLAGGED[2]]) and sha256 == c.sha(table)
    assert problems == ['run list names logit__luna__python__few_shot__chatbot__99, which is not a selected position']


@pytest.mark.skipif(not p.RUN_LIST.is_file(), reason='the flagged-run list of the exposure audit is not written yet')
def test_project_run_list():
    # On 2026-09-26 the list holds 79 positions, and 5 of them are outside the 32 tasks.
    positions, sha256, problems = p.read_run_list()
    assert problems == [] and positions and sha256 == c.sha(p.RUN_LIST)
    outside = [name for name in positions if name.split('__')[0] not in p.RERUN32_TASKS]
    assert {name.split('__')[0] for name in outside} <= set(p.FULL_LAYOUT_TASKS)
    assert p.expected_counts('flagged_runs', positions)['positions'] == len(positions)
    assert p.expected_counts(COMBINED, positions)['positions'] == 11520 + len(outside)


def test_decision_state(tmp_path):
    assert p.decision_state(tmp_path / 'missing.md') == (None, None)
    proposed = record(tmp_path / 'a.md', 'PROPOSED')
    assert p.decision_state(proposed) == ('PROPOSED', None)
    assert len(p.decision_problems('few_shot', proposed)) == 2
    mismatch = record(tmp_path / 'b.md', 'APPROVED', 'few_shot_and_zero_shot')
    assert p.decision_problems('few_shot', mismatch) == [
        "decision record approved scope is 'few_shot_and_zero_shot', not 'few_shot'"]
    approved = record(tmp_path / 'c.md', 'APPROVED', 'few_shot')
    assert p.decision_problems('few_shot', approved) == []
    wide = record(tmp_path / 'd.md', 'APPROVED', SCOPE32)
    assert p.decision_state(wide) == ('APPROVED', SCOPE32) and p.decision_problems(SCOPE32, wide) == []
    sha = 'a' * 64
    flagged = record(tmp_path / 'e.md', 'APPROVED', 'flagged_runs', run_list_sha256=sha)
    assert p.approved_run_list(flagged) == sha
    assert p.decision_problems('flagged_runs', flagged, run_list_sha256=sha) == []
    assert p.decision_problems('flagged_runs', flagged) == ['run list missing']
    assert p.decision_problems('flagged_runs', flagged, run_list_sha256='b' * 64) == [
        f"decision record approved run list sha256 is {sha!r}, not {'b' * 64!r}"]
    assert p.decision_problems('flagged_runs', wide, run_list_sha256=sha)[-1].startswith(
        'decision record approved run list sha256 is None')


def test_project_decision_record_parses():
    status, scope = p.decision_state()
    assert status in {'PROPOSED', 'APPROVED'}
    if status != 'APPROVED':
        for scope_name in p.SCOPES:
            assert p.decision_problems(scope_name)


# Prompts ----------------------------------------------------------------------------------

def test_production_prompts_differ_only_by_the_example_block():
    sources, provenance = p.load_source_prompts()
    assert len(sources) == len(provenance) == 720
    for key in p.prompt_keys(('few_shot',), tasks=p.ALL_TASKS):
        task, platform, _, agency, fmt = key.split('__')
        zero = sources['__'.join((task, platform, 'zero_shot', agency, fmt))]
        head, block, tail = p.split_block(sources[key], zero)
        assert head + tail == zero and block.rstrip('\n').endswith('```')
        if task in p.ORIGINAL_TASKS:
            assert block.startswith(p.BLOCK_START) and block.endswith('```\n') and tail.startswith(p.TASK_MARK)
        else:
            assert not head or head.endswith('\n')
        c.prepared_prompt(sources[key], agency)


def test_added_production_prompt_folder():
    provenance = p.read(p.ADDED_PROMPTS.parent / 'provenance.json')
    assert set(provenance['keys']) == set(p.prompt_keys(tasks=p.ADDED20_TASKS)) and not provenance['unresolved_runs']
    for key, entry in provenance['keys'].items():
        assert c.sha(p.ADDED_PROMPTS / f'{key}.txt') == entry['sha256'] == entry['texts'][0]['sha256']
        # Each key has two models and ten repetitions of selected runs.
        assert entry['selected_runs'] == sum(t['runs'] for t in entry['texts']) == 20


def test_supplied_file_tasks_match_the_catalog():
    catalog = p.read(p.CATALOG)
    several = {task for task in p.RERUN32_TASKS if set(catalog[task].get('input_files') or {}) - {'input.csv'}}
    assert several == set(p.SUPPLIED_FILE_TASKS)


PARTS = dict(title='Two toy groups', context='Ten simulated rows form two groups.',
             explanation='The example compares the two group means.',
             platform_notes=dict(stata='oneway reports F.'))
EXPECTED = {'python': '{\n  "task_id": "demo"\n}\n', 'r': '{\n  "task_id": "demo"\n}\n',
            'stata': 'ANSWER: F estimate=3.25 se=. n_obs=10\n'}


def fake_examples(tasks=p.RERUN32_TASKS):
    return {(task, platform): (f'# {task} {platform} example\nprint_line = 1\n', PARTS, EXPECTED[platform])
            for task in tasks for platform in p.PLATFORMS}


def test_render_replaces_only_the_example_block():
    sources, _ = p.load_source_prompts()
    examples = fake_examples()
    prompts = p.render_prompts(sources, examples, 'few_shot')
    assert len(prompts) == 180 and all('__few_shot__' in key for key in prompts)
    for key, text in prompts.items():
        task, platform, _, agency, fmt = key.split('__')
        old_head, _, old_tail = p.split_block(sources[key])
        block = p.example_block(platform, *examples[(task, platform)], task=task)
        assert text == old_head + block + old_tail
        assert block.startswith('Related example: Two toy groups\nSeparate example context: ')
        assert f'```{platform}\n# {task} {platform} example\n' in block
        assert (p.RELATION['python'] in block) == (platform == 'python')
        assert p.OUTPUT_HEADER[platform] in block and block.endswith('```\n')
        assert p.BLOCK_START not in text and 'related econometric task' not in text
        effective = c.prepared_prompt(text, agency)
        assert (c.AGENT_VERIFICATION_INSTRUCTION in effective) == (agency == 'agent')
    both = p.render_prompts(sources, examples, 'few_shot_and_zero_shot')
    assert len(both) == 360
    assert all(both[key] == sources[key] for key in p.prompt_keys(('zero_shot',)))


def test_render_added_tasks_keeps_the_block_position_and_spacing():
    sources, _ = p.load_source_prompts()
    examples = fake_examples()
    prompts = p.render_prompts(sources, examples, SCOPE32)
    assert len(prompts) == 576
    assert all(prompts[key] == sources[key] for key in p.prompt_keys(('zero_shot',), tasks=p.RERUN32_TASKS))
    for key in p.prompt_keys(('few_shot',), tasks=p.ADDED12_TASKS):
        task, platform, _, agency, fmt = key.split('__')
        zero = sources['__'.join((task, platform, 'zero_shot', agency, fmt))]
        old_head, old_block, old_tail = p.split_block(sources[key], zero)
        block = p.example_block(platform, *examples[(task, platform)], task=task)
        assert old_block.endswith('```\n\n')
        # The new block takes the place of the old block and keeps its blank line before the next text.
        assert prompts[key] == old_head + block + '\n' + old_tail
        head, new_block, tail = p.split_block(prompts[key], zero)
        assert head + tail == zero and new_block == block + '\n'
        relation = p.RELATION_SUPPLIED_FILES if task in p.SUPPLIED_FILE_TASKS else p.RELATION
        assert (relation['python'] in block) == (platform == 'python')
        assert 'A smaller example uses separate' not in prompts[key] and 'Example on separate small' not in prompts[key]


def test_example_block_layout():
    # Same labels and order as the added tasks with a full example block (R and Stata builder).
    block = p.example_block('stata', 'display 1\n\n', PARTS, EXPECTED['stata'])
    assert block == ('Related example: Two toy groups\n'
                     'Separate example context: Ten simulated rows form two groups.\n'
                     'Explanation of this example: The example compares the two group means. oneway reports F.\n'
                     '```stata\ndisplay 1\n```\n'
                     'Expected printed output:\n```text\nANSWER: F estimate=3.25 se=. n_obs=10\n```\n')
    # Python adds the relation line before the code, as the added-task Python builder does.
    block = p.example_block('python', 'x = 1\n', PARTS, EXPECTED['python'])
    assert block == ('Related example: Two toy groups\n'
                     'Separate example context: Ten simulated rows form two groups.\n'
                     'Explanation of this example: The example compares the two group means.\n'
                     + p.RELATION['python'] + '\n```python\nx = 1\n```\n'
                     'Expected result.json:\n```json\n{\n  "task_id": "demo"\n}\n```\n')
    supplied = p.example_block('python', 'x = 1\n', PARTS, EXPECTED['python'], task='applied_nonrecipient_population_iv')
    assert p.RELATION_SUPPLIED_FILES['python'] in supplied and 'its own input.csv' not in supplied
    r_block = p.example_block('r', 'x <- 1\n', PARTS, EXPECTED['r'])
    assert p.RELATION['python'] not in r_block and 'Expected result.json:\n```json\n' in r_block
    with pytest.raises(ValueError):
        p.example_block('python', 'x = 1\n```\n', PARTS, EXPECTED['python'])
    with pytest.raises(ValueError):
        p.example_block('python', 'x = 1\n', dict(PARTS, context='two\nlines'), EXPECTED['python'])
    with pytest.raises(ValueError):
        p.example_block('python', 'x = 1\n', dict(PARTS, title=''), EXPECTED['python'])


def test_parts_problems():
    assert p.parts_problems(PARTS) == []
    assert p.parts_problems([]) == ['block parts are not a JSON object']
    assert 'block part explanation is missing or empty' in p.parts_problems(dict(PARTS, explanation=None))
    assert any('platform_notes' in problem for problem in p.parts_problems(dict(PARTS, platform_notes=dict(julia='x'))))
    assert p.explanation_for(PARTS, 'python') == PARTS['explanation']
    assert p.explanation_for(PARTS, 'stata') == PARTS['explanation'] + ' oneway reports F.'


def test_render_output_and_same_output(tmp_path):
    (tmp_path / 'result.json').write_text('{"task_id": "demo", "estimates": [{"term": "F", "estimate": 3.25}]}')
    text = p.render_output('python', tmp_path)
    assert text == json.dumps(dict(task_id='demo', estimates=[dict(term='F', estimate=3.25)]), indent=2) + '\n'
    (tmp_path / 'example.log').write_text('. display "ANSWER: F estimate=" %21.12f 3.25\n'
                                          'ANSWER: F estimate=       3.250000000000 se=. n_obs=         75  \n')
    assert p.render_output('stata', tmp_path) == 'ANSWER: F estimate=3.250000000000 se=. n_obs=75\n'
    assert p.same_output('ANSWER: F estimate=3.25 se=. n_obs=75\n', 'ANSWER: F estimate=3.2500000000001 se=. n_obs=75')
    assert not p.same_output('ANSWER: F estimate=3.25 se=. n_obs=75', 'ANSWER: F estimate=3.26 se=. n_obs=75')
    assert not p.same_output('ANSWER: F estimate=3.25 se=. n_obs=75', 'ANSWER: G estimate=3.25 se=. n_obs=75')
    assert not p.same_output('ANSWER: F estimate=3.25', None)


def test_literal_matches():
    assert p.literal_matches('the slope is 0.6097 here', 0.609681)
    assert p.literal_matches('"estimate": 0.6096814709975316', 0.6096814709975316)
    assert p.literal_matches('value 6.097e-01', 0.609681)
    assert not p.literal_matches('the slope is 0.610 here', 0.609681)  # three significant digits
    assert not p.literal_matches('the slope is 0.6098 here', 0.609681)
    assert not p.literal_matches('240 rows and 0.975 quantile', 240.4)


def test_project_example_blocks_render_in_the_added_task_layout():
    """The 96 project example blocks load, render, and keep task data and reference values out."""
    sources, _ = p.load_source_prompts()
    catalog = p.production_catalog(tasks=p.RERUN32_TASKS)
    inventory = p.example_inventory(p.RERUN32_TASKS)
    assert all(path is not None for path in inventory.values()), [k for k, v in inventory.items() if v is None]
    examples = p.load_examples(inventory)
    prompts = p.render_prompts(sources, examples, SCOPE32)
    assert len(prompts) == 576
    labels = ('Related example: ', '\nSeparate example context: ', '\nExplanation of this example: ')
    for (task, platform), (code, parts, expected) in examples.items():
        block = p.example_block(platform, code, parts, expected, task=task)
        positions = [block.find(label) for label in labels + (f'\n```{platform}\n', '\n' + p.OUTPUT_HEADER[platform])]
        assert positions[0] == 0 and positions == sorted(positions) and min(positions[1:]) > 0, (task, platform)
        assert p.block_text_problems(parts, expected, platform, catalog[task]) == [], (task, platform)
        assert p.static_example_problems(code, catalog[task]) == [], (task, platform)
        assert not catalog[task].get('data_file') or catalog[task]['data_file'] not in block
        assert f'"task_id": "{task}"' in expected if platform != 'stata' else expected.startswith('ANSWER: ')
    lengths = p.block_lengths(prompts, sources)
    assert len(lengths) == 96 and min(lengths.values()) > 0


def test_render_rejects_a_prompt_without_zero_shot_identity():
    sources, _ = p.load_source_prompts()
    key = 'anova_oneway__python__few_shot__chatbot__json'
    broken = dict(sources)
    broken[key] = sources[key].replace('Task: ', 'Task: Changed. ', 1)
    with pytest.raises(ValueError):
        p.render_prompts(broken, fake_examples(), 'few_shot', tasks=('anova_oneway',))
    key = 'lp04_ppml__r__few_shot__agent__json'
    broken = dict(sources)
    broken[key] = sources[key].replace('Task:', 'Task (changed):', 1)
    with pytest.raises(ValueError):
        p.render_prompts(broken, fake_examples(), SCOPE32, tasks=('lp04_ppml',))


# Example checks ---------------------------------------------------------------------------

PY_GOOD = '''import json
rows = [dict(term=t, estimate=v, se=None, n_obs=75) for t, v in
        [("F", 3.25), ("df_num", 2.0), ("df_den", 72.0), ("p_value", 0.044)]]
json.dump(dict(task_id="anova_oneway", platform="python", estimates=rows), open("result.json", "w"))
'''


PY_GOOD_EXPECTED = json.dumps(dict(task_id='anova_oneway', platform='python', estimates=[
    dict(term=t, estimate=v, se=None, n_obs=75) for t, v in [('F', 3.25), ('df_num', 2.0), ('df_den', 72.0),
                                                             ('p_value', 0.044)]]), indent=2) + '\n'


def python_tools():
    return dict(python=sys.executable, env=None)


def write_block_files(folder, platform, parts=PARTS, expected=None):
    folder = Path(folder)
    if parts is not None:
        (folder / p.PARTS_FILE).write_text(json.dumps(parts))
    if expected is not None:
        (folder / p.EXPECTED_FILES[platform]).write_text(expected)


def run_python(code, spec, parts=PARTS, expected=PY_GOOD_EXPECTED):
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'example.py'
        path.write_text(code)
        write_block_files(folder, 'python', parts, expected)
        return p.check_example('anova_oneway', 'python', path, spec, python_tools(), timeout=120)


def test_python_example_passes():
    spec = p.production_catalog()['anova_oneway']
    result = run_python(PY_GOOD, spec)
    assert result['passed'], result['failures']
    assert result['prints_scored_names'] == ['F', 'df_den', 'df_num', 'p_value']
    assert result['estimates']['F'] == 3.25 and result['run']['returncode'] == 0
    assert result['output_text'] == PY_GOOD_EXPECTED


def test_python_example_block_file_checks():
    spec = p.production_catalog()['anova_oneway']
    result = run_python(PY_GOOD, spec, expected=PY_GOOD_EXPECTED.replace('3.25', '3.5'))
    assert result['failures'] == ['example output differs from expected_python.json']
    result = run_python(PY_GOOD, spec, parts=None, expected=None)
    assert result['failures'] == ['block_parts.json missing', 'expected_python.json missing']
    reference = next(row['estimate'] for row in spec['reference']['estimates'] if row['term'] == 'F')
    leaking = dict(PARTS, context=f'The F statistic is {reference}.')
    result = run_python(PY_GOOD, spec, parts=leaking)
    assert result['failures'] == ['block text contains the reference value of F']
    result = run_python(PY_GOOD, spec, expected=PY_GOOD_EXPECTED.replace('3.25', repr(reference)))
    assert 'block text contains the reference value of F' in result['failures']


def test_python_example_that_reads_the_task_input_fails():
    spec = p.production_catalog()['anova_oneway']
    code = 'import pandas as pd\npd.read_csv("input.csv")\n' + PY_GOOD
    result = run_python(code, spec)
    assert not result['passed']
    assert any('exited with status' in failure for failure in result['failures'])


def test_python_example_with_reference_values_fails():
    spec = p.production_catalog()['anova_oneway']
    reference = {row['term']: row['estimate'] for row in spec['reference']['estimates']}
    literal = PY_GOOD.replace('3.25', repr(reference['F']))
    result = run_python(literal, spec)
    assert any('contains the reference value of F' in failure for failure in result['failures'])
    copied = PY_GOOD
    for term, value in (('3.25', reference['F']), ('2.0', reference['df_num']), ('72.0', reference['df_den']),
                        ('0.044', reference['p_value'])):
        copied = copied.replace(term, f'{value} + 0', 1)
    result = run_python(copied, spec)
    assert 'example output reproduces the reference value of every scored term' in result['failures']
    named = PY_GOOD.replace('open("result.json"', 'open("pilot_v5_anova.csv.json"')
    result = run_python(named, spec)
    assert any('scored data file' in failure for failure in result['failures'])


def fake_stata_runner(log_text, returncode=0, timed_out=False):
    def runner(argv, cwd, env, timeout, platform):
        assert platform == 'stata' and argv[-2:] == ['do', 'example.do']
        (Path(cwd) / 'example.log').write_text(log_text)
        return dict(returncode=returncode, stdout='', stderr='', seconds=0.1, timed_out=timed_out)
    return runner


def test_stata_example_checks(tmp_path):
    spec = p.production_catalog()['anova_oneway']
    path = tmp_path / 'example.do'
    path.write_text('display "ANSWER: F estimate=" 3.25\n')
    write_block_files(tmp_path, 'stata', expected='ANSWER: F estimate=3.25 se=. n_obs=75\n'
                                                  'ANSWER: df_num estimate=2 se=. n_obs=75\n')
    tools = dict(stata='stata-se', env=None, printed_estimates=stub_printed_estimates)
    good = '. display\nANSWER: F estimate=   3.25 se=. n_obs=   75\nANSWER: df_num estimate=2 se=. n_obs=75\n'
    result = p.check_example('anova_oneway', 'stata', path, spec, tools, runner=fake_stata_runner(good))
    assert result['passed'], result['failures']
    changed = good.replace('3.25', '3.5')
    result = p.check_example('anova_oneway', 'stata', path, spec, tools, runner=fake_stata_runner(changed))
    assert result['failures'] == ['example output differs from expected_stata.txt']
    error = good + 'variable x not found\nr(111);\n'
    result = p.check_example('anova_oneway', 'stata', path, spec, tools, runner=fake_stata_runner(error))
    assert 'example.log contains a Stata error code' in result['failures']
    result = p.check_example('anova_oneway', 'stata', path, spec, tools, runner=fake_stata_runner('', timed_out=True))
    assert not result['passed'] and any('did not finish' in failure for failure in result['failures'])
    result = p.check_example('anova_oneway', 'stata', None, spec, tools)
    assert result['failures'] == ['example.do missing']


# Prepare, check, and launch ---------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Redirect every written path to tmp_path and supply 96 example files."""
    original, added = tmp_path / 'examples', tmp_path / 'added12_examples'
    for task in p.RERUN32_TASKS:
        for platform, name in p.EXAMPLE_FILES.items():
            target = (original if task in p.ORIGINAL_TASKS else added) / task / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f'* {task} {platform} example\n')
            write_block_files(target.parent, platform, PARTS, EXPECTED[platform])
    monkeypatch.setattr(p, 'EXAMPLES', original)
    monkeypatch.setattr(p, 'ADDED12_EXAMPLES', added)
    monkeypatch.setattr(p, 'COLLECTIONS', tmp_path / 'collections')
    monkeypatch.setattr(p, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(p, 'DECISION', record(tmp_path / 'decision.md', 'PROPOSED'))
    run_list = tmp_path / 'flagged.md'
    run_list.write_text('Flagged positions:\n' + '\n'.join(f'- {name}' for name in FLAGGED) + '\n')
    monkeypatch.setattr(p, 'RUN_LIST', run_list)
    links = p.baseline_links
    monkeypatch.setattr(p, 'baseline_links', lambda jobs, **kw: links(jobs, verify_hashes=False))

    def passing(inventory, catalog, run):
        return {key: dict(task=key[0], platform=key[1], path=str(path) if path else None, passed=path is not None,
                          failures=[] if path else ['missing'], warnings=[], sha256=c.sha(path) if path else None)
                for key, path in inventory.items()}
    monkeypatch.setattr(prepare, 'check_examples', passing)
    return tmp_path


def fake_collector(argv, capture_output, text, cwd):
    """Write a manifest the way the central prep does, without CLI or environment checks."""
    action = argv[2]
    manifest_path = Path(argv[argv.index('--manifest') + 1])
    if action == 'prep':
        config = json.loads(Path(argv[argv.index('--config') + 1]).read_text())
        catalog = json.loads(Path(config['catalog_path']).read_text())
        jobs = config['jobs']
        prompts = {key: c.prepared_prompt(Path(path).read_text(), key.split('__')[3])
                   for key, path in config['prompt_paths'].items()}
        protocol = dict(tasks={task: catalog[task] for task in sorted({job['task'] for job in jobs})},
                        prompts=prompts, decision_record=config['decision_record'], planned_jobs=jobs)
        c.save(manifest_path, dict(protocol=protocol, protocol_sha256=c.digest(protocol), jobs=jobs))
    return SimpleNamespace(returncode=0, stdout='{}', stderr='')


def test_dry_run_writes_nothing(workspace, capsys):
    assert prepare.main([]) == 0
    output = capsys.readouterr().out
    assert 'Dry run. Nothing written.' in output and 'not APPROVED' in output
    assert prepare.main(['--scope', SCOPE32]) == 0
    output = capsys.readouterr().out
    assert '"positions": 11520' in output and '"added12_new"' in output and '"original20_new"' in output
    assert prepare.main(['--scope', 'flagged_runs', '--run-list', str(workspace / 'missing.md')]) == 0
    output = capsys.readouterr().out
    assert 'run list missing' in output and 'the scope plans no position' in output
    assert not p.COLLECTIONS.exists() and not p.RUNS.exists()
    with pytest.raises(SystemExit):
        prepare.main(['--scope', 'few_shot', '--run-list', str(p.RUN_LIST)])


def test_write_refuses_without_approval_or_examples(workspace):
    with pytest.raises(SystemExit, match='status is .PROPOSED'):
        prepare.main(['--write'])
    record(p.DECISION, 'APPROVED', 'few_shot_and_zero_shot')
    with pytest.raises(SystemExit, match='approved scope'):
        prepare.main(['--write', '--scope', 'few_shot'])
    record(p.DECISION, 'APPROVED', 'few_shot')
    (p.EXAMPLES / 'vif' / 'example.do').unlink()
    with pytest.raises(SystemExit, match='59 of 60 example files pass'):
        prepare.main(['--write'])
    record(p.DECISION, 'APPROVED', SCOPE32)
    (p.ADDED12_EXAMPLES / 'lp04_ppml' / 'example.R').unlink()
    with pytest.raises(SystemExit, match='94 of 96 example files pass'):
        prepare.main(['--write', '--scope', SCOPE32])
    record(p.DECISION, 'APPROVED', 'flagged_runs', run_list_sha256='0' * 64)
    with pytest.raises(SystemExit, match='approved run list sha256'):
        prepare.main(['--write', '--scope', 'flagged_runs'])
    assert not p.COLLECTIONS.exists() and not p.RUNS.exists()
    with pytest.raises(SystemExit):
        prepare.main(['--write', '--skip-run'])


@pytest.mark.parametrize('scope, positions, per_block', [('few_shot', 3600, 900),
                                                         ('few_shot_and_zero_shot', 7200, 1800),
                                                         (SCOPE32, 11520, 2880)])
def test_write_then_check(workspace, monkeypatch, scope, positions, per_block):
    record(p.DECISION, 'APPROVED', scope)
    monkeypatch.setattr(prepare, 'subprocess', SimpleNamespace(run=fake_collector))
    assert prepare.main(['--write', '--scope', scope]) == 0
    meta = p.meta(scope)
    plan = p.read(meta / 'plan.json')
    ledger = p.read(meta / 'planned_positions.json')
    index = p.read(meta / 'collections.json')
    assert plan['scope'] == scope and plan['run_list'] is None
    assert plan['counts']['positions'] == len(ledger) == positions and len(index) == 4
    assert all(entry['planned'] == per_block for entry in index)
    assert {row['prompt'] for row in ledger} == set(p.scope_prompts(scope))
    assert all(row['supersedes_record_sha256'] and row['status'] == 'planned' and not row['flagged'] for row in ledger)
    assert {row['prompt_source'] for row in ledger} == ({'new_example_block'} if scope == 'few_shot'
                                                         else {'new_example_block', 'production_zero_shot'})
    assert not list(p.raw(scope).rglob('authorization.json'))
    assert len(list(meta.glob('*_launch_commands.json'))) == 4
    assert check.check_counts(plan, ledger, index) == []
    problems, sources, inventory, catalog = check.check_sources(plan)
    assert problems == []
    assert check.check_prompts(plan, sources, inventory, index, catalog) == []
    assert check.main(['--skip-central']) == 0
    assert not (meta / 'validation.json').exists()
    with pytest.raises(SystemExit, match='already exists'):
        prepare.main(['--write', '--scope', scope])
    assert plan['block_layout'] == json.loads(json.dumps(p.BLOCK_LAYOUT))
    tasks = p.scope_tasks(scope)
    assert set(plan['parts_sha256']) == set(tasks) and len(plan['expected_sha256']) == 3 * len(tasks)
    # A changed example, block part, or expected output after preparation stops the check and the launch.
    write_block_files(p.EXAMPLES / 'probit', 'r', dict(PARTS, title='Changed'), EXPECTED['r'] + '\n')
    problems, *_ = check.check_sources(plan)
    assert 'block parts changed since preparation' in problems
    assert 'expected outputs changed since preparation' in problems
    (p.EXAMPLES / 'logit' / 'example.R').write_text('# changed\n')
    problems, *_ = check.check_sources(plan)
    assert any('example files changed' in problem for problem in problems)
    with pytest.raises(SystemExit, match='Refusing launch'):
        launch.main(['--confirm'])
    assert not (meta / 'launch_receipt.json').exists()


@pytest.mark.parametrize('scope, positions', [('flagged_runs', 3), (COMBINED, 11522)])
def test_write_then_check_with_a_run_list(workspace, monkeypatch, scope, positions):
    sha256 = c.sha(p.RUN_LIST)
    record(p.DECISION, 'APPROVED', scope, run_list_sha256=sha256)
    monkeypatch.setattr(prepare, 'subprocess', SimpleNamespace(run=fake_collector))
    assert prepare.main(['--write', '--scope', scope]) == 0
    meta = p.meta(scope)
    plan = p.read(meta / 'plan.json')
    ledger = p.read(meta / 'planned_positions.json')
    index = p.read(meta / 'collections.json')
    assert plan['run_list']['sha256'] == sha256 and plan['run_list']['positions'] == FLAGGED
    assert len(ledger) == positions and {p.position_id(row) for row in ledger if row['flagged']} == set(FLAGGED)
    sources, _ = p.load_source_prompts()
    for row in ledger:
        if row['task'] not in p.scope_tasks(scope):
            assert row['prompt_source'] == 'production'
            key = c.prompt_key(row)
            manifest = p.read(row['manifest'])
            assert manifest['protocol']['prompts'][key] == c.prepared_prompt(sources[key], row['condition'])
    assert check.check_counts(plan, ledger, index) == []
    problems, sources, inventory, catalog = check.check_sources(plan)
    assert problems == []
    assert check.check_prompts(plan, sources, inventory, index, catalog) == []
    assert check.main(['--skip-central']) == 0
    # A changed run list after preparation stops the check and the launch.
    p.RUN_LIST.write_text(p.RUN_LIST.read_text() + f'- {baseline_ids("probit")[0]}\n')
    problems, *_ = check.check_sources(plan)
    assert 'run list changed since preparation' in problems
    c.save(meta / 'validation.json', dict(passed=True, plan_digest=plan['plan_digest']))
    with pytest.raises(SystemExit, match='run list changed'):
        launch.main(['--confirm'])


def test_launch_requires_confirm_and_validation(workspace):
    problems, plan, meta = launch.readiness()
    assert len(problems) == 1 and problems[0].startswith('run prepare.py --write and check.py first')
    assert plan is None and meta is None
    with pytest.raises(SystemExit, match='Refusing launch'):
        launch.main([])


def test_check_and_launch_find_the_prepared_collection(workspace, monkeypatch):
    monkeypatch.setattr(prepare, 'subprocess', SimpleNamespace(run=fake_collector))
    record(p.DECISION, 'APPROVED', 'few_shot')
    assert prepare.main(['--write']) == 0
    assert p.find_collection() == ('few_shot', p.meta('few_shot'))
    with pytest.raises(SystemExit, match='No prepared collection for scope'):
        p.find_collection(SCOPE32)
    record(p.DECISION, 'APPROVED', SCOPE32)
    assert prepare.main(['--write', '--scope', SCOPE32]) == 0
    with pytest.raises(SystemExit, match='Several prepared collections'):
        p.find_collection()
    assert p.find_collection(SCOPE32) == (SCOPE32, p.meta(SCOPE32))
    assert check.main(['--skip-central', '--scope', SCOPE32]) == 0


def test_check_and_launch_refuse_when_the_record_leaves_approval(workspace, monkeypatch):
    def no_process(*args, **kwargs):
        raise AssertionError('The test must not start a process')

    record(p.DECISION, 'APPROVED', 'few_shot')
    monkeypatch.setattr(prepare, 'subprocess', SimpleNamespace(run=fake_collector))
    monkeypatch.setattr(launch, 'subprocess', SimpleNamespace(run=no_process, Popen=no_process))
    assert prepare.main(['--write']) == 0
    meta = p.meta('few_shot')
    plan = p.read(meta / 'plan.json')
    c.save(meta / 'validation.json', dict(passed=True, plan_digest=plan['plan_digest']))
    # With approval and validation, launch stops before any action unless --confirm is given.
    with pytest.raises(SystemExit, match='Add --confirm'):
        launch.main([])
    record(p.DECISION, 'PROPOSED')
    problems, *_ = check.check_sources(plan)
    assert any('not APPROVED' in problem for problem in problems)
    assert any('decision record changed' in problem for problem in problems)
    assert check.main(['--skip-central']) == 1
    with pytest.raises(SystemExit, match='Refusing launch: .*not APPROVED'):
        launch.main(['--confirm'])
    assert not list(p.raw('few_shot').rglob('authorization.json')) and not (meta / 'launch_receipt.json').exists()

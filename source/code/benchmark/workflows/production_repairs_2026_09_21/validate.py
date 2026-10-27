"""Validate exact prompt changes, provenance, cells, and central runtime checks."""
from collections import Counter
import subprocess
import sys
import prepare as p

def main():
    rows = p.read(p.META / 'planned_positions.json')
    assert len(rows) == len({p.slot(row) for row in rows}) == 720
    cells = Counter(p.slot(row)[:-1] for row in rows)
    assert len(cells) == 72 and set(cells.values()) == {10}
    assert Counter(row['task'] for row in rows) == dict.fromkeys(p.TASKS, 360)
    assert set(Counter((row['task'], row['condition']) for row in rows).values()) == {120}
    assert len({row['supersedes_record'] for row in rows}) == 720
    source = p.read(p.META / 'source_prompts.json')
    assert p.c.sha(p.SOURCE) == source['source_csv_sha256']
    previous = {p.slot(row): row for row in p.source_rows()}
    for row in rows:
        old = previous[p.slot(row)]
        assert row['supersedes_record'] == old['record_path']
        assert p.c.sha(row['supersedes_record']) == row['supersedes_record_sha256'] == old['record_sha256']
        assert row['previous_success'] == old['y']
    checks = []
    for entry in p.read(p.META / 'collections.json'):
        manifest = p.read(entry['manifest'])
        assert len(manifest['jobs']) == 180
        protocol = manifest['protocol']
        assert all(value[1] is None for value in protocol['commands'].values())
        for key, text in protocol['prompts'].items():
            before = source['prompts'][key]
            expected = before.replace('Task:\n', 'Task:\n' + p.INSERTION, 1) if key.startswith(p.TASKS[0] + '__') else before
            assert text == expected, key
        for task in p.TASKS:
            before = p.read(p.BASE / task / 'source_task.json')
            after = protocol['tasks'][task]
            assert after == p.read(p.BASE / task / 'task.json')
            allowed = {'data_path', 'input_files', 'prompt', 'review_changes', 'review_version', 'input_instruction'}
            assert {k:v for k,v in before.items() if k not in allowed} == {k:v for k,v in after.items() if k not in allowed}
            assert before['reference'] == after['reference']
            assert p.c.sha(p.c.task_input_files(before)['input.csv']) == p.c.sha(p.c.task_input_files(after)['input.csv'])
            for platform in ['python', 'r', 'stata']:
                for agency in ['chatbot', 'self_repair', 'agent']:
                    job = dict(task=task, platform=platform, condition=agency, output_format='answer' if platform == 'stata' else 'json')
                    zs = protocol['prompts'][p.c.prompt_key(dict(**job, prompt='zero_shot'))]
                    fs = protocol['prompts'][p.c.prompt_key(dict(**job, prompt='few_shot'))]
                    assert zs.split('Task:\n', 1)[1] == fs.split('Task:\n', 1)[1]
        result = subprocess.run([sys.executable, str(p.ROOT / 'code/benchmark/prospective_collect_unlimited_additions.py'),
            'check', '--manifest', entry['manifest']], capture_output=True, text=True)
        checks.append(dict(batch=entry['batch'], returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        p.c.save(p.META / 'validation_checks.json', checks)
        assert result.returncode == 0, result.stdout + result.stderr
    assert p.read(p.META / 'preflight.json')['passed']
    p.c.save(p.META / 'validation.json', dict(passed=True, total=720, retained=0, new=720, cells=72, repetitions=10,
        workers_per_block=8, total_workers=32, walltime_limits=None, lp_only_input_paragraph=True,
        ppml_prompt_unchanged=True, prior_records_linked=720, fresh_per_task=360, fresh_per_task_agency=120))
    print('Passed: 720 unique fresh positions; exact prompt changes and all prior record hashes verified.')

if __name__ == '__main__':
    main()

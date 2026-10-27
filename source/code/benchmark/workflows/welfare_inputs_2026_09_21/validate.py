"""Verify complete ownership and the single authorized prompt insertion."""
from collections import Counter
from pathlib import Path
import subprocess
import sys
import prepare as p

def main():
    rows = p.read(p.META / 'planned_positions.json')
    assert len(rows) == len({p.slot(row) for row in rows}) == 360
    cells = Counter(p.slot(row)[:-1] for row in rows)
    assert len(cells) == 36 and set(cells.values()) == {10}
    assert len({row['supersedes_record'] for row in rows}) == 360
    insertion = (p.BASE / 'input_instruction.txt').read_text()
    for row in rows:
        assert p.c.sha(row['supersedes_record']) == row['supersedes_record_sha256']
        old_record = p.read(row['supersedes_record'])
        assert p.slot(old_record['job']) == p.slot(row)
        new_prompt = p.read(row['manifest'])['protocol']['prompts'][p.c.prompt_key(row)]
        old_base = new_prompt.replace('Task:\n' + insertion, 'Task:\n', 1)
        # Self-repair records retain the last request, including error feedback.
        if row['condition'] == 'self_repair':
            assert old_record['prompt'].startswith(old_base)
        else:
            assert old_record['prompt'] == old_base
    checks = []
    for entry in p.read(p.META / 'collections.json'):
        manifest = p.read(entry['manifest'])
        assert len(manifest['jobs']) == 90
        model = manifest['jobs'][0]['model']
        old = p.read(p.OLD_RAW / model / '0_4/manifest.json')['protocol']
        protocol = manifest['protocol']
        assert protocol['commands'] == old['commands']
        assert all(value[1] is None for value in protocol['commands'].values())
        assert protocol['policy'] == old['policy']
        before, after = old['tasks'][p.TASK], protocol['tasks'][p.TASK]
        assert after == p.read(p.BASE / 'task.json')
        allowed = {'data_path', 'prompt_paths', 'prompt', 'review_changes', 'review_version', 'input_instruction'}
        assert {k:v for k,v in before.items() if k not in allowed} == {k:v for k,v in after.items() if k not in allowed}
        assert before['reference'] == after['reference']
        assert p.c.sha(p.c.task_input_files(before)['input.csv']) == p.c.sha(p.c.task_input_files(after)['input.csv'])
        old_prompts = p.read(p.OLD_RAW / 'sonnet/0_4/manifest.json')['protocol']['prompts']
        for key, text in protocol['prompts'].items():
            assert text == old_prompts[key].replace('Task:\n', 'Task:\n' + insertion, 1)
        for platform in ['python', 'r', 'stata']:
            for agency in ['chatbot', 'self_repair', 'agent']:
                job = dict(task=p.TASK, platform=platform, condition=agency, output_format='answer' if platform == 'stata' else 'json')
                zs = protocol['prompts'][p.c.prompt_key(dict(**job, prompt='zero_shot'))]
                fs = protocol['prompts'][p.c.prompt_key(dict(**job, prompt='few_shot'))]
                assert zs.split('Task:\n', 1)[1] == fs.split('Task:\n', 1)[1]
        command = [sys.executable, str(p.ROOT / 'code/benchmark/prospective_collect_unlimited_additions.py'), 'check', '--manifest', entry['manifest']]
        result = subprocess.run(command, capture_output=True, text=True)
        checks.append(dict(batch=entry['batch'], returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        p.c.save(p.META / 'validation_checks.json', checks)
        assert result.returncode == 0, result.stdout + result.stderr
    p.c.save(p.META / 'validation.json', dict(passed=True, total=360, retained=0, new=360, cells=36, repetitions=10, workers_per_block=8, total_workers=32, walltime_limits=None, only_prompt_insertion=True, prior_records_linked=360))
    print('Passed: 360 unique fresh positions; prompts differ only by input paragraph.')

if __name__ == '__main__':
    main()

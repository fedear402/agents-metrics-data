"""Prepare fresh welfare responses with an explicit input filename and schema."""
from pathlib import Path
import json
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import prospective_collect as c

TASK = 'hard_conditional_logit_welfare'
BASE = ROOT / 'data/task_candidates/2026-09-21_welfare_inputs' / TASK
META = ROOT / 'data/collections/2026-09-21_welfare-inputs'
RAW = ROOT / 'data/runs/welfare_inputs_2026-09-21'
OLD = ROOT / 'data/collections/2026-09-20_three-expansion'
OLD_RAW = ROOT / 'data/runs/three_expansion_2026-09-20'
DECISION = ROOT / 'quality_reports/decision_records/2026-09-21_welfare-inputs.md'
WORKERS = 8

def read(path):
    return json.loads(Path(path).read_text())

def slot(row):
    return tuple(row[k] for k in ['task', 'model', 'platform', 'prompt', 'condition', 'global_rep'])

def old_record(row):
    if row['status'] == 'retained':
        return Path(row['source_record'])
    manifest = read(row['manifest'])
    folder = Path(row['manifest']).parent / 'runs' / manifest['protocol_sha256'] / c.job_id(row)
    records = sorted(folder.glob('attempt_*/run_record.json'))
    terminal = [path for path in records if read(path)['status'] in c.TERMINAL]
    assert terminal, str(folder)
    return terminal[-1]

def main():
    assert DECISION.exists()
    assert not (META / 'collections.json').exists(), 'Already prepared'
    insertion = (BASE / 'input_instruction.txt').read_text()
    assert insertion.endswith('\n\n') and 'input.csv' in insertion
    spec = read(BASE / 'task.json')
    previous = {slot(row): row for row in read(OLD / 'planned_positions.json') if row['task'] == TASK}
    assert len(previous) == 360
    index, ledger, receipts = [], [], []
    for model in ['luna', 'sonnet']:
        source = read(OLD_RAW / model / '0_4/manifest.json')
        old_spec = source['protocol']['tasks'][TASK]
        assert spec['reference'] == old_spec['reference']
        assert c.sha(c.task_input_files(spec)['input.csv']) == c.sha(c.task_input_files(old_spec)['input.csv'])
        for offset in [0, 5]:
            batch = f'welfare_{model}_{offset}'
            folder = RAW / model / f'{offset}_{offset + 4}'
            paths = {}
            # Luna's prior manifest omits two retained Python chatbot keys.
            # Prompts are provider-independent; Sonnet records all 18 keys.
            prompt_source = read(OLD_RAW / 'sonnet/0_4/manifest.json')['protocol']['prompts']
            for key, text in prompt_source.items():
                if not key.startswith(TASK + '__'):
                    continue
                if key in source['protocol']['prompts']:
                    assert text == source['protocol']['prompts'][key]
                assert text.count('Task:\n') == 1
                path = folder / 'prompts' / (key + '.txt')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text.replace('Task:\n', 'Task:\n' + insertion, 1))
                paths[key] = str(path)
            assert len(paths) == 18
            jobs = []
            for rep in range(offset, offset + 5):
                wave = [dict(block=batch, task=TASK, model=model, platform=platform, prompt=prompt, condition=agency, output_format='answer' if platform == 'stata' else 'json', rep=rep % 5, global_rep=rep) for platform in ['python', 'r', 'stata'] for prompt in ['zero_shot', 'few_shot'] for agency in ['chatbot', 'self_repair', 'agent']]
                random.Random(20260921 + rep).shuffle(wave)
                jobs.extend(wave)
            c.save(folder / 'catalog.json', {TASK: spec})
            c.save(folder / 'config.json', dict(catalog_path=str(folder / 'catalog.json'), prompt_paths=paths, jobs=jobs, decision_record=str(DECISION)))
            for action in ['prep', 'check']:
                argv = [sys.executable, str(ROOT / 'code/benchmark/prospective_collect_unlimited_additions.py'), action, '--manifest', str(folder / 'manifest.json')]
                if action == 'prep':
                    argv += ['--config', str(folder / 'config.json')]
                process = subprocess.run(argv, capture_output=True, text=True)
                receipts.append(dict(batch=batch, action=action, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr))
                c.save(META / 'preparation_receipts.json', receipts)
                assert process.returncode == 0, process.stdout + process.stderr
            manifest = read(folder / 'manifest.json')
            c.save(folder / 'authorization.json', dict(protocol_sha256=manifest['protocol_sha256'], allow_provider_calls=True, decision_record=str(DECISION), scope='Repeat all 360 welfare positions after adding the input filename and schema.'))
            entry = dict(batch=batch, manifest=str(folder / 'manifest.json'), planned=90, protocol_sha256=manifest['protocol_sha256'])
            index.append(entry)
            for job in jobs:
                old = previous[slot(job)]
                record = old_record(old)
                ledger.append(dict(**job, status='planned', manifest=entry['manifest'], protocol_sha256=entry['protocol_sha256'], supersedes_record=str(record), supersedes_record_sha256=c.sha(record), supersedes_manifest=old.get('source_manifest', old.get('manifest'))))
    c.save(META / 'collections.json', index)
    c.save(META / 'planned_positions.json', ledger)
    for entry in index:
        batch = entry['batch']
        launch = dict(name=batch, argv=['uv', 'run', '--no-sync', 'python', '-u', str(ROOT / 'code/benchmark/production_queue_unlimited_additions.py'), '--index', str(META / 'collections.json'), '--solo-workers', str(WORKERS), '--overlap-workers', str(WORKERS), '--drain-model', '--batches', batch], cwd=str(ROOT), env={'UV_CACHE_DIR': '/private/tmp/agents-metrics-uv-cache'}, log=str(META / (batch + '_collector.log')), index=str(META / 'collections.json'), batches=[batch])
        c.save(META / (batch + '_launch_commands.json'), [launch])
    print('Prepared four disjoint blocks, 360 fresh positions. Launch requires validation.')

if __name__ == '__main__':
    main()

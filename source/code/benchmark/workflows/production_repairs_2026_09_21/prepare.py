"""Prepare 720 fresh positions from their current effective source manifests."""
from pathlib import Path
from collections import Counter
import csv
import json
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import prospective_collect as c

TASKS = ['hard_panel_lp_iv', 'lp04_ppml']
BASE = ROOT / 'data/task_candidates/2026-09-21_production_repairs'
META = ROOT / 'data/collections/2026-09-21_production-repairs'
RAW = ROOT / 'data/runs/production_repairs_2026-09-21'
SOURCE = ROOT / 'data/production/main/pre_migration_runs.csv'
DECISION = ROOT / 'quality_reports/decision_records/2026-09-21_production-repairs.md'
WORKERS = 8
INSERTION = ('Read input.csv from the current directory. Its columns are unit (unit identifier), '
             'period (calendar period), y (outcome), x (endogenous regressor), and z (excluded instrument).\n\n')

def read(path):
    return json.loads(Path(path).read_text())

def slot(row):
    return tuple(str(row[k]) for k in ['task', 'model', 'platform', 'prompt', 'condition', 'global_rep'])

def source_rows():
    with SOURCE.open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['task'] in TASKS]
    assert len(rows) == len({slot(row) for row in rows}) == 720
    return rows

def source_material(rows):
    manifests = {row['manifest_path']: read(row['manifest_path']) for row in rows}
    prompts, specs, origins = {}, {}, {}
    for row in rows:
        protocol = manifests[row['manifest_path']]['protocol']
        key = c.prompt_key(dict(row, output_format='answer' if row['platform'] == 'stata' else 'json'))
        text = protocol['prompts'][key]
        assert key not in prompts or text == prompts[key], key
        prompts[key] = text
        origins.setdefault(key, set()).add(row['manifest_path'])
        spec = protocol['tasks'][row['task']]
        if row['task'] in specs:
            assert spec['reference'] == specs[row['task']]['reference']
            assert c.sha(c.task_input_files(spec)['input.csv']) == c.sha(c.task_input_files(specs[row['task']])['input.csv'])
        # The current collection spec preserves all original requested diagnostic outputs.
        if row['task'] not in specs or 'six_parallel_sonnet' in row['manifest_path']:
            specs[row['task']] = spec
    assert len(prompts) == 36
    return prompts, specs, {key: sorted(paths) for key, paths in origins.items()}

def main():
    assert DECISION.exists()
    assert read(META / 'preflight.json')['passed']
    assert not (META / 'collections.json').exists(), 'Already prepared'
    rows = source_rows()
    previous = {slot(row): row for row in rows}
    prompts, specs, origins = source_material(rows)
    c.save(META / 'source_prompts.json', dict(prompts=prompts, origins=origins, source_csv=str(SOURCE), source_csv_sha256=c.sha(SOURCE)))
    for task, old_spec in list(specs.items()):
        spec = dict(old_spec)
        # Resolve compatibility links to canonical physical inputs without duplicating data.
        spec['data_path'] = str(Path(spec['data_path']).resolve())
        if 'input_files' in spec:
            spec['input_files'] = {name: str(Path(path).resolve()) for name, path in spec['input_files'].items()}
        if task == TASKS[0]:
            spec['prompt'] = INSERTION + old_spec['prompt']
            spec['input_instruction'] = INSERTION
            spec['review_changes'] = 'State input.csv and its five actual columns. Preserve the task method, references, and examples.'
            spec['review_version'] = '2026-09-21_production_repairs'
            (BASE / task).mkdir(parents=True, exist_ok=True)
            (BASE / task / 'input_instruction.txt').write_text(INSERTION)
        c.save(BASE / task / 'task.json', spec)
        c.save(BASE / task / 'source_task.json', old_spec)
        specs[task] = spec
    index, ledger, receipts = [], [], []
    for model in ['luna', 'sonnet']:
        for offset in [0, 5]:
            batch = f'production_repairs_{model}_{offset}'
            folder = RAW / model / f'{offset}_{offset + 4}'
            paths = {}
            for key, text in prompts.items():
                assert text.count('Task:\n') == 1
                if key.startswith(TASKS[0] + '__'):
                    text = text.replace('Task:\n', 'Task:\n' + INSERTION, 1)
                path = folder / 'prompts' / (key + '.txt')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                paths[key] = str(path)
            jobs = []
            for rep in range(offset, offset + 5):
                wave = [dict(block=batch, task=task, model=model, platform=platform, prompt=prompt, condition=agency,
                             output_format='answer' if platform == 'stata' else 'json', rep=rep % 5, global_rep=rep)
                        for task in TASKS for platform in ['python', 'r', 'stata']
                        for prompt in ['zero_shot', 'few_shot'] for agency in ['chatbot', 'self_repair', 'agent']]
                random.Random(20260921 + rep).shuffle(wave)
                jobs.extend(wave)
            c.save(folder / 'catalog.json', specs)
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
            c.save(folder / 'authorization.json', dict(protocol_sha256=manifest['protocol_sha256'], allow_provider_calls=True,
                     decision_record=str(DECISION), scope='Repeat all 720 panel LP-IV and PPML positions after production repairs.'))
            entry = dict(batch=batch, manifest=str(folder / 'manifest.json'), planned=180, protocol_sha256=manifest['protocol_sha256'])
            index.append(entry)
            for job in jobs:
                old = previous[slot(job)]
                record = Path(old['record_path'])
                assert c.sha(record) == old['record_sha256']
                ledger.append(dict(**job, status='planned', manifest=entry['manifest'], protocol_sha256=entry['protocol_sha256'],
                    supersedes_record=str(record), supersedes_record_sha256=c.sha(record), supersedes_manifest=old['manifest_path'],
                    previous_position_id=old['position_id'], previous_success=old['y'], source_csv_sha256=c.sha(SOURCE)))
    c.save(META / 'collections.json', index)
    c.save(META / 'planned_positions.json', ledger)
    for entry in index:
        batch = entry['batch']
        launch = dict(name=batch, argv=['uv', 'run', '--no-sync', 'python', '-u',
            str(ROOT / 'code/benchmark/production_queue_unlimited_additions.py'), '--index', str(META / 'collections.json'),
            '--solo-workers', str(WORKERS), '--overlap-workers', str(WORKERS), '--drain-model', '--batches', batch],
            cwd=str(ROOT), env={'UV_CACHE_DIR': '/private/tmp/agents-metrics-uv-cache'},
            log=str(META / (batch + '_collector.log')), index=str(META / 'collections.json'), batches=[batch])
        c.save(META / (batch + '_launch_commands.json'), [launch])
    print('Prepared four disjoint blocks and 720 fresh positions. Parent review controls launch.')

if __name__ == '__main__':
    main()

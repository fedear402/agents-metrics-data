"""Prepare production inputs and collect missing positions with the existing runner.

Definitions and examples live in data/tasks. Raw responses live in data/runs.
The position index supplies the missing jobs. This program never selects by success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / 'data/production/paper22_2026-09-18'
TASKS = ROOT / 'data/tasks/paper22_2026-09-18'
RUNS = ROOT / 'data/runs/paper22_2026-09-18'
COLLECTOR = ROOT / 'code/benchmark/prospective_collect.py'
DECISION = ROOT / 'quality_reports/decision_records/2026-09-18_production-paper22.md'
ACCEPTED = ROOT / 'quality_reports/reports/2026-09-17_agent-condition-and-bank/work/accepted_bank/collector_catalog.json'
OLD_PROMPTS = ROOT / 'quality_reports/reports/2026-09-17_live-check-and-checkpoint/work/inputs/prompt_paths.json'
NEW_TASKS = ('hard_heteroskedastic_probit', 'hard_truncated_normal_effects',
             'hard_poisson_hurdle_effects', 'd1_attendance')
FIELDS = ('task', 'model', 'platform', 'prompt', 'condition', 'global_rep')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_ready(folder):
    """Check that native reference evidence still matches the saved sources."""
    receipt = read(folder / 'ready.json')
    for name, expected in receipt.get('files_sha256', {}).items():
        path = folder / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Ready source hash changed: {path}')
    native_path = receipt.get('native_validation', receipt.get('validation'))
    if not native_path:
        raise ValueError(f'Missing native validation receipt: {folder}')
    native = read(folder / native_path)
    if native.get('passed') is not True:
        raise ValueError(f'Native validation did not pass: {folder}')
    spec = read(folder / 'task.json')
    input_hash = native.get('source_input_sha256', native.get('input_sha256'))
    if not input_hash or sha(spec['data_path']) != input_hash:
        raise ValueError(f'Native input hash changed: {folder}')
    for name, expected in native.get('scripts', {}).items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Native source hash changed: {path}')
    if not receipt.get('files_sha256') and not native.get('scripts'):
        raise ValueError(f'Missing source hashes: {folder}')
    targets = {row['term'] for row in spec['reference']['estimates']}
    checks = native.get('checks', [])
    if 'native' in native:
        checks = [dict(check, language=result['platform']) for result in native['native']
                  for check in result['checks']]
        if any(result['returncode'] != 0 for result in native['native']):
            raise ValueError(f'Native reference execution failed: {folder}')
    for language in ('r', 'stata'):
        selected = [row for row in checks if row['language'] == language]
        if {row['term'] for row in selected} != targets or any(row.get('passed') is not True for row in selected):
            raise ValueError(f'Incomplete native reference checks: {folder}/{language}')
    if 'example_validation' in receipt:
        examples = read(folder / receipt['example_validation'])
        if examples.get('passed') is not True or any(row.get('passed') is not True for row in examples['examples']):
            raise ValueError(f'Example validation failed: {folder}')
        languages = {row['platform'] for row in examples['examples']}
    else:
        languages = set(native.get('examples_passed', []))
    if languages != {'python', 'r', 'stata'}:
        raise ValueError(f'Incomplete example checks: {folder}')


def prompt_key(job):
    return '__'.join(job[k] for k in ('task', 'platform', 'prompt', 'condition', 'output_format'))


def pilot(task):
    report = '2026-09-17_applied-next-eight' if task == 'd1_attendance' else '2026-09-17_twenty-harder-tasks'
    folder = ROOT / 'quality_reports/reports' / report / 'collection'
    manifest = read(folder / 'manifest.json')
    key = '__'.join((task, 'python', 'few_shot', 'agent', 'json'))
    return manifest['protocol']['tasks'][task], manifest['protocol']['prompt_provenance'][key]['source_path']


def rendered_prompt(task, platform, prompt, condition):
    folder = TASKS / task
    fragment = (folder / 'fragments' / f'{platform}_{prompt}.txt').read_text()
    if platform == 'python' and condition == 'agent':
        _, old_path = pilot(task)
        old = Path(old_path).read_text()
        few = (folder / 'fragments/python_few_shot.txt').read_text()
        if not old.startswith(few):
            raise ValueError(f'{task}: original Python example or statement changed')
        return fragment + old[len(few):]
    if platform == 'stata' and ('Write result.json' in fragment or 'write result.json' in fragment):
        raise ValueError(f'{task}: Stata fragment still requires JSON')
    output = (f'\nWrite result.json as a JSON object with task_id="{task}", platform="{platform}", '
              'and the requested estimates list.\n') if platform != 'stata' else '\nPrint the requested ANSWER lines to the Stata log.\n'
    language = {'python': 'python', 'r': 'r', 'stata': 'stata'}[platform]
    if condition != 'agent':
        return (fragment + output + f'\nWrite one self-contained {language} script.\n'
                'The evaluator executes it in a directory containing input.csv.\n'
                'Act only as a code generator. Do not use tools, inspect files, or execute code.\n'
                f'Return exactly one fenced {language} code block and nothing else.\n')
    script = {'python': 'run.py', 'r': 'run.R', 'stata': 'run.do'}[platform]
    command = {'python': f'{sys.executable} run.py', 'r': 'Rscript --vanilla run.R',
               'stata': '/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se -b do run.do'}[platform]
    return (fragment + output + '\nUse the shell to write, execute, inspect, and revise your script.\n'
            f'Run the script with {command}.\nLeave the final script as {script}.\n'
            'When the requested output is present, finish with DONE.\n'
            'Use only input.csv, installed software, and files you create in this directory.\n'
            'Do not read benchmark references, previous results, agent skills, or project instructions.\n'
            'Do not use Git, web tools, MCP servers, or other agents.\n')


def assemble():
    catalog, paths = read(ACCEPTED), read(OLD_PROMPTS)
    assert len(catalog) == 18
    ready, pending = [], []
    for task in NEW_TASKS:
        folder = TASKS / task
        receipt = folder / 'ready.json'
        if not receipt.exists() or read(receipt).get('passed') is not True:
            pending.append(task)
            continue
        validate_ready(folder)
        spec = read(folder / 'task.json')
        previous, old_prompt = pilot(task)
        if spec['reference'] != previous['reference'] or sha(spec['data_path']) != sha(previous['data_path']):
            raise ValueError(f'{task}: original references or inputs changed')
        catalog[task] = spec
        for platform in ('python', 'r', 'stata'):
            for prompt in ('zero_shot', 'few_shot'):
                for condition in ('chatbot', 'self_repair', 'agent'):
                    job = dict(task=task, platform=platform, prompt=prompt, condition=condition,
                               output_format='answer' if platform == 'stata' else 'json')
                    text = rendered_prompt(task, platform, prompt, condition)
                    key = prompt_key(job)
                    if platform == 'python' and prompt == 'few_shot' and condition == 'agent':
                        assert text == Path(old_prompt).read_text()
                        paths[key] = old_prompt
                    else:
                        path = folder / 'prompts' / (key + '.txt')
                        path.parent.mkdir(exist_ok=True)
                        if path.exists() and path.read_text() != text:
                            raise ValueError(f'Preserve a previously rendered prompt: {path}')
                        path.write_text(text)
                        paths[key] = str(path)
        ready.append(task)
    save(DESIGN / 'catalog.json', catalog)
    save(DESIGN / 'prompt_paths.json', paths)
    save(DESIGN / 'definition_status.json', dict(original=18, ready_new=ready, pending_new=pending,
         catalog_tasks=len(catalog), original_prompt_files_preserved=True))
    print(json.dumps(dict(tasks=len(catalog), ready=ready, pending=pending)))


def validate_jobs(jobs):
    seen = set()
    for job in jobs:
        key = tuple(job[k] for k in FIELDS)
        if key in seen:
            raise ValueError('Duplicate production position')
        seen.add(key)
        if type(job['global_rep']) is not int or type(job['rep']) is not int or not 0 <= job['global_rep'] < 10 or job['rep'] != job['global_rep'] % 5:
            raise ValueError('Invalid local or global repetition')
    return seen


def prepare(jobs_path, batch):
    if not batch or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in batch):
        raise ValueError('Use a flat batch name')
    out = RUNS / batch
    if (out / 'manifest.json').exists():
        raise ValueError('Preserve the existing collection manifest')
    jobs = read(jobs_path)
    requested = validate_jobs(jobs)
    if len({job['global_rep'] // 5 for job in jobs}) != 1:
        raise ValueError('Prepare repetitions one through five and six through ten in separate manifests')
    positions = read(DESIGN / 'planned_positions.json')
    missing = {tuple(row[k] for k in FIELDS) for row in positions if row['status'] == 'unplanned'}
    if not requested <= missing:
        raise ValueError('Requested positions must be unplanned in the current production index')
    catalog, paths = read(DESIGN / 'catalog.json'), read(DESIGN / 'prompt_paths.json')
    subset = {task: catalog[task] for task in sorted({j['task'] for j in jobs})}
    prompts = {prompt_key(j): paths[prompt_key(j)] for j in jobs}
    ledger = read(DESIGN / 'collections.json') if (DESIGN / 'collections.json').exists() else []
    for old in ledger:
        previous = read(Path(old['manifest']).parent / 'config.json')['jobs']
        if requested & validate_jobs(previous):
            raise ValueError('Production positions already assigned to another collection')
    # Per-batch metadata stays immutable when the design gains another ready task.
    save(out / 'catalog.json', subset)
    save(out / 'config.json', dict(catalog_path=str(out / 'catalog.json'),
         decision_record=str(DECISION), prompt_paths=prompts, jobs=jobs))
    receipts = []
    for action in ('prep', 'check'):
        argv = [sys.executable, str(COLLECTOR), action, '--manifest', str(out / 'manifest.json')]
        if action == 'prep':
            argv += ['--config', str(out / 'config.json')]
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        receipts.append(dict(command=argv, returncode=result.returncode,
                             stdout=result.stdout, stderr=result.stderr))
        save(out / 'preparation_checks.json', receipts)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
    manifest = read(out / 'manifest.json')
    save(out / 'authorization.json', dict(protocol_sha256=manifest['protocol_sha256'],
         allow_provider_calls=True, user_scope='22 tasks, all dimensions, ten repetitions',
         decision_record=str(DECISION)))
    ledger.append(dict(batch=batch, manifest=str(out / 'manifest.json'), planned=len(jobs),
                       protocol_sha256=manifest['protocol_sha256']))
    save(DESIGN / 'collections.json', ledger)
    print(json.dumps(ledger[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('assemble')
    p = sub.add_parser('prepare')
    p.add_argument('--jobs', type=Path, required=True)
    p.add_argument('--batch', required=True)
    args = parser.parse_args()
    if args.action == 'assemble':
        assemble()
    else:
        prepare(args.jobs, args.batch)


if __name__ == '__main__':
    main()

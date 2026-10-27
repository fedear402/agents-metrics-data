"""Prepare the task bank v2 collection plan. The default run is a dry run.

The dry run reads data/task_candidates/2026-09-23_v2/<task>/, checks every
package, renders the 36 prompts of each task, builds the planned positions, and
prints counts and check results. It writes nothing.

--write saves the plan under data/collections/<date>_task-bank-v2/ only. It
refuses to write when a check fails. --manifests also runs the central
``prospective_collect_unlimited_additions.py prep`` and ``check`` for each
block. That step reads CLI versions and native package lists. It sends no
model request. This script never writes authorization.json or launch commands.
A launch needs a separate decision and the central queue
(``production_queue_unlimited_additions.py``), which applies the Sonnet write
guard.

Examples:
  uv run python code/benchmark/workflows/task_bank_v2/prepare.py
  uv run python code/benchmark/workflows/task_bank_v2/prepare.py --pilot 1 --tasks ols_robust,probit
  uv run python code/benchmark/workflows/task_bank_v2/prepare.py --write --date 2026-09-30
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import prospective_collect as c  # noqa: E402
import production_queue as queue  # noqa: E402


def load_workflow(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reused without changes: agency suffixes, JSON reading, and slot identity.
THREE = load_workflow('three_expansion_prepare', ROOT / 'code/benchmark/workflows/three_expansion_2026_09_20/prepare.py')
REPAIRS = load_workflow('production_repairs_prepare', ROOT / 'code/benchmark/workflows/production_repairs_2026_09_21/prepare.py')
read = THREE.read
slot = REPAIRS.slot

V2 = ROOT / 'data/task_candidates/2026-09-23_v2'
CATALOG = ROOT / 'data/production/main/catalog.json'
SCHEMA_SOURCE = ROOT / 'data/task_candidates/2026-09-19_six_expansion'
DECISION = ROOT / 'quality_reports/decision_records/2026-09-23_task-bank-v2.md'
COLLECTOR = ROOT / 'code/benchmark/prospective_collect_unlimited_additions.py'
MODELS = ('luna', 'sonnet')
PLATFORMS = ('python', 'r', 'stata')
PROMPTS = ('zero_shot', 'few_shot')
AGENCIES = ('chatbot', 'self_repair', 'agent')
FULL_REPS = 10
BLOCK_REPS = 5  # prospective_collect.validate_jobs accepts rep 0 through 4 in one manifest.
SEED = 20260923
REQUIRED_KEYS = ('task_id', 'title', 'prompt', 'estimands', 'data_path', 'data_description', 'reference', 'validation',
                 'review_version')
LANGUAGE = {'python': 'Python', 'r': 'R', 'stata': 'Stata'}
EXAMPLE = {'python': 'example.py', 'r': 'example.R', 'stata': 'example.do'}
SCORING_FILES = ('scoring_spec.json', 'scoring.json')
STATA_LOG_NOTE = 'Print every ANSWER line to the Stata log run.log with display.'


def v2_suffix(platform, agency):
    """Agency text of the current collections, with the Stata log named (P6)."""
    text = THREE.suffix(platform, agency)
    if platform != 'stata':
        return text
    if agency == 'agent':
        old = 'Leave the final script as run.do and its requested output in this directory.\n'
        assert text.count(old) == 1
        return text.replace(old, 'Leave the final script as run.do in this directory. '
                            'Its batch run writes run.log, which must contain every ANSWER line.\n')
    return text.replace('The evaluator runs it with the supplied CSV files in its working directory.',
                        'The evaluator runs it in batch mode with the supplied CSV files in its working directory '
                        'and reads the ANSWER lines from run.log.')


def stata_contract(spec):
    lines = []
    for row in spec['reference']['estimates']:
        fields = ['estimate=<number>', 'se=<number>' if row.get('se') is not None else 'se=NA']
        if row.get('n_obs') is not None:
            fields.append('n_obs=<integer>')
        lines.append(f"ANSWER: {row['term']} " + ' '.join(fields))
    return ('Print one line per estimate. ' + STATA_LOG_NOTE + ' Use exactly this format, replacing each '
            '<number> with the computed value. Report at least twelve significant digits, for example with '
            '%21.12g. Use set linesize 255.\n' + '\n'.join(lines) + '\n')


def python_contract_start(text):
    for marker in ('Use json.dump', 'Reporting:\n'):
        if marker in text:
            return text.index(marker)
    return None


def strip_chatbot_text(text, platform, origin):
    """Remove the exact chatbot text of the current collections when a package file ends with it."""
    tail = THREE.suffix(platform, 'chatbot')
    for candidate in (tail, tail.lstrip('\n')):
        if text.endswith(candidate):
            return text[:-len(candidate)].rstrip('\n') + '\n', origin + ' (embedded chatbot text removed)'
    return text, origin


def render_body(folder, spec, platform, prompt):
    """Return (body, source). Package files win. Otherwise render from the Python prompt."""
    for name in (f'prompts/{platform}_{prompt}.txt', f'prompts/{prompt}_{platform}.txt',
                 f'prompts/{platform}/{prompt}.txt', f'{prompt}_{platform}.txt'):
        if (folder / name).is_file():
            return strip_chatbot_text((folder / name).read_text(), platform, name)
    base = folder / f'{prompt}.txt'
    if not base.is_file():
        raise ValueError(f'missing prompt {prompt}.txt or prompts/{platform}_{prompt}.txt')
    text, origin = strip_chatbot_text(base.read_text(), 'python', base.name)
    if platform == 'python':
        return text, origin
    start = python_contract_start(text)
    if start is None:
        raise ValueError(f'{base.name}: no Python contract marker to render {platform}')
    body = text[:start]
    if 'Task:\n' in body:
        # The language names appear in the header before the task statement.
        head, rest = body.split('Task:\n', 1)
        body = head.replace('Python', LANGUAGE[platform]) + 'Task:\n' + rest
    if platform == 'r':
        contract = text[start:].replace('json.dump', 'jsonlite::write_json with digits = NA and auto_unbox = TRUE')
        for old, new in (('"python"', '"r"'), ('Python', 'R')):
            contract = contract.replace(old, new)
        if 'digits = NA' not in contract:
            contract = ('Use jsonlite::write_json with digits = NA and auto_unbox = TRUE to write result.json.\n'
                        + contract)
    else:
        contract = stata_contract(spec)
    body += contract
    if prompt == 'few_shot':
        match = re.search(r'```python\n.*?```\n', body, flags=re.S)
        example = folder / EXAMPLE[platform]
        if not match or not example.is_file():
            raise ValueError(f'few_shot {platform}: no Python example block or no {example.name}')
        block = f'```{platform}\n' + example.read_text().rstrip('\n') + '\n```\n'
        body = body[:match.start()] + block + body[match.end():]
    return body, f'rendered from {base.name}'


def scoring_spec(folder, spec):
    for key in ('scoring_spec', 'scoring'):
        if isinstance(spec.get(key), dict):
            return spec[key], f'task.json:{key}'
    for name in SCORING_FILES:
        if (folder / name).is_file():
            return read(folder / name), name
    return None, None


def call_arguments(text, name):
    """Yield the argument text of every call to name(...)."""
    for match in re.finditer(re.escape(name) + r'\s*\(', text):
        depth, start = 1, match.end()
        for index in range(start, len(text)):
            depth += {'(': 1, ')': -1}.get(text[index], 0)
            if depth == 0:
                yield text[start:index]
                break


def finite(value):
    try:
        return value is not None and not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def check_package(task, runtime_base, expected_ids):
    """Return (spec, prompts, failures, warnings, notes) for one v2 package."""
    folder = V2 / task
    failures, warnings, notes = [], [], {}
    if not (folder / 'task.json').is_file():
        return None, {}, ['task.json missing'], warnings, notes
    spec = read(folder / 'task.json')
    for key in REQUIRED_KEYS:
        if key not in spec:
            failures.append(f'task.json lacks {key}')
    if failures:
        return None, {}, failures, warnings, notes
    if spec['task_id'] != task:
        failures.append(f"task_id {spec['task_id']} differs from folder name")
    if spec['review_version'] != 'v2':
        failures.append(f"review_version is {spec['review_version']!r}")
    if task not in expected_ids:
        warnings.append('task is not in the current 40-task catalog')
    if not (folder / 'CHANGES.json').is_file():
        failures.append('CHANGES.json missing')
    # References.
    rows = spec['reference'].get('estimates') if isinstance(spec['reference'], dict) else None
    if not rows:
        failures.append('reference estimates missing')
        rows = []
    terms = [row.get('term') for row in rows]
    if sorted(terms) != sorted(spec['estimands']) or len(set(terms)) != len(terms):
        failures.append('reference terms differ from estimands')
    for row in rows:
        if not finite(row.get('estimate')):
            failures.append(f"{row.get('term')}: reference estimate not finite")
        if row.get('se') is not None and not (finite(row['se']) and float(row['se']) >= 0):
            failures.append(f"{row.get('term')}: reference se invalid")
        if row.get('n_obs') is not None and not (finite(row['n_obs']) and float(row['n_obs']).is_integer()):
            failures.append(f"{row.get('term')}: reference n_obs not an integer")
    # Inputs.
    try:
        inputs = c.task_input_files(spec)
        for name, path in inputs.items():
            if not path.is_file():
                failures.append(f'input {name} missing at {path}')
            elif not path.resolve().is_relative_to(folder.resolve()):
                warnings.append(f'input {name} lives outside the v2 package')
        notes['inputs'] = {name: c.sha(path) for name, path in inputs.items() if path.is_file()}
    except ValueError as error:
        failures.append(f'input paths: {error}')
    # Scoring spec.
    scoring, source = scoring_spec(folder, spec)
    if scoring is None:
        failures.append('scoring spec missing (scoring_spec.json, scoring.json, or task.json scoring_spec)')
    else:
        notes['scoring_source'] = source
        scored = scoring.get('terms', scoring)
        if isinstance(scored, list):
            scored = {row.get('term') for row in scored if isinstance(row, dict)}
        missing = [t for t in spec['estimands'] if t not in scored]
        if missing:
            failures.append('scoring spec lacks terms: ' + ', '.join(missing))
    # Examples.
    for platform, name in EXAMPLE.items():
        if not (folder / name).is_file():
            failures.append(f'{name} missing')
    validation = folder / 'native_example_validation.json'
    verified = bool(spec['validation'].get('example_verified')) if isinstance(spec['validation'], dict) else False
    if validation.is_file():
        verified = verified or read(validation).get('passed') is True
    if not verified:
        failures.append('examples not verified (native_example_validation.json passed or validation.example_verified)')
    if (folder / 'example.R').is_file():
        example_r = (folder / 'example.R').read_text()
        calls = list(call_arguments(example_r, 'write_json')) + list(call_arguments(example_r, 'toJSON'))
        if not calls:
            failures.append('example.R writes no JSON with write_json or toJSON')
        elif not all(re.search(r'digits\s*=\s*NA\b', call) for call in calls):
            failures.append('example.R JSON call without digits = NA')
    if (folder / 'example.do').is_file():
        example_do = (folder / 'example.do').read_text()
        if 'ANSWER:' not in example_do or 'display' not in example_do and 'di ' not in example_do:
            failures.append('example.do does not display ANSWER: lines')
    if (folder / 'example.log').is_file():
        parsed, _ = runtime_base.printed_estimates((folder / 'example.log').read_text(errors='replace'))
        notes['example_log_answer_rows'] = len(parsed)
        if not parsed:
            failures.append('example.log holds no parseable ANSWER line')
    # Prompts.
    prompts = {}
    for platform in PLATFORMS:
        bodies = {}
        for prompt in PROMPTS:
            try:
                body, origin = render_body(folder, spec, platform, prompt)
            except ValueError as error:
                failures.append(str(error))
                continue
            if re.search(r'Act only as a code generator|Return exactly one fenced|finish with DONE', body):
                failures.append(f'{platform}/{prompt}: prompt body already holds agency text; '
                                'prepare.py appends the agency text')
            bodies[prompt] = body
            notes.setdefault('prompt_sources', {})[f'{platform}/{prompt}'] = origin
            for agency in AGENCIES:
                job = dict(task=task, platform=platform, prompt=prompt, condition=agency,
                           output_format='answer' if platform == 'stata' else 'json')
                text = body + v2_suffix(platform, agency)
                try:
                    c.prepared_prompt(text, agency)
                except ValueError as error:
                    failures.append(f'{platform}/{prompt}/{agency}: {error}')
                prompts[c.prompt_key(job)] = text
        if len(bodies) == 2 and all('Task:\n' in b for b in bodies.values()):
            if bodies['zero_shot'].split('Task:\n', 1)[1] != bodies['few_shot'].split('Task:\n', 1)[1]:
                warnings.append(f'{platform}: zero_shot and few_shot differ after Task:')
        elif len(bodies) == 2:
            warnings.append(f'{platform}: no Task: marker for the prompt identity check')
    for key, text in prompts.items():
        platform = key.split('__')[1]
        if platform == 'stata':
            if 'run.log' not in text:
                failures.append(f'{key}: Stata prompt does not name run.log')
            # Contract lines carry placeholders. Example output lines in few-shot prompts do not.
            contract = [line for line in text.splitlines() if line.startswith('ANSWER:') and '<' in line]
            filled = '\n'.join(line.replace('<number>', '1.25').replace('<integer>', '7')
                               .replace('<number or .>', '.').replace('<number or NA>', 'NA') for line in contract)
            estimates, _ = runtime_base.printed_estimates(filled)
            if sorted(estimates) != sorted(spec['estimands']):
                failures.append(f'{key}: contract lines do not parse for every term')
            elif c.delivery_problem(spec, estimates):
                failures.append(f'{key}: contract omits a field the delivery check requires: '
                                + c.delivery_problem(spec, estimates).replace('\n', '; '))
        if platform != 'python':
            outside = re.sub(r'```.*?```', '', text, flags=re.S)
            if re.search(r'\bpython\b|json\.dump', outside, re.I):
                failures.append(f'{key}: {platform} prompt still names Python or json.dump')
        if platform == 'r' and not re.search(r'digits\s*=\s*NA', text):
            failures.append(f'{key}: R prompt does not state digits = NA')
    # Answer leaks (warnings): literal reference values in the task text.
    task_text = spec['prompt']
    for row in rows:
        value = row.get('estimate')
        if finite(value) and abs(float(value)) >= 1e-3 and not float(value).is_integer():
            if f'{float(value):.6g}' in task_text:
                warnings.append(f"{row['term']}: reference value appears in the task text")
        if row.get('n_obs') is not None and re.search(rf'(?<![\d.]){int(row["n_obs"])}(?![\d.])', task_text):
            warnings.append(f"{row['term']}: literal n_obs {int(row['n_obs'])} appears in the task text")
    documents = [read(folder / name) for name in ('CHANGES.json', 'environment.json') if (folder / name).is_file()]
    notes['p11'] = p11_packages(*documents)
    return spec, prompts, sorted(set(failures)), sorted(set(warnings)), notes


def p11_packages(*documents):
    """Collect entries listed under any key containing 'missing' (P11 lists in CHANGES.json or environment.json)."""
    found = []

    def walk(value, inside, language):
        if isinstance(value, dict):
            if inside and ('package' in value or 'name' in value):
                found.append(dict(package=value.get('package', value.get('name')),
                                  language=value.get('language', language), install=value.get('install')))
                return
            for key, item in value.items():
                lang = key if key in ('python', 'r', 'stata') else language
                walk(item, inside or 'missing' in str(key).lower(), lang)
        elif isinstance(value, list):
            for item in value:
                walk(item, inside, language)
        elif inside and isinstance(value, str) and value:
            found.append(dict(package=value, language=language, install=None))
    for document in documents:
        walk(document, False, None)
    return found


def plan_jobs(tasks, reps, models):
    jobs = []
    for model in models:
        for offset in range(0, reps, BLOCK_REPS):
            batch = f'task_bank_v2_{model}_{offset}'
            for rep in range(offset, min(offset + BLOCK_REPS, reps)):
                wave = [dict(block=batch, task=task, model=model, platform=platform, prompt=prompt, condition=agency,
                             output_format='answer' if platform == 'stata' else 'json', rep=rep % BLOCK_REPS,
                             global_rep=rep)
                        for task in tasks for platform in PLATFORMS for prompt in PROMPTS for agency in AGENCIES]
                random.Random(SEED + rep).shuffle(wave)
                jobs.extend(wave)
    return jobs


def validate_plan(jobs, catalog, prompts):
    slots = [slot(job) for job in jobs]
    assert len(slots) == len(set(slots)), 'duplicate positions'
    for batch in sorted({job['block'] for job in jobs}):
        c.validate_jobs([job for job in jobs if job['block'] == batch], dict(tasks=catalog, prompts=prompts))


def sonnet_guard():
    path = queue.SONNET_GUARD
    text = path.read_text() if path.is_file() else ''
    return dict(path=str(path), exists=path.is_file(), sha256=c.sha(path) if path.is_file() else None,
                denies_venv=str(ROOT / '.venv') in text, launcher='code/benchmark/production_queue_unlimited_additions.py')


def write_plan(target, catalog, prompts, jobs, report, manifests):
    if target.exists():
        raise SystemExit(f'Refusing to overwrite {target}. Choose another --date or --label.')
    if not DECISION.is_file():
        raise SystemExit(f'Missing decision record {DECISION}. Write it before --write.')
    target.mkdir(parents=True)
    c.save(target / 'catalog.json', catalog)
    paths = {}
    for key, text in prompts.items():
        path = target / 'prompts' / (key + '.txt')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        paths[key] = str(path)
    index = []
    for batch in sorted({job['block'] for job in jobs}):
        folder = target / 'blocks' / batch
        block_jobs = [job for job in jobs if job['block'] == batch]
        c.save(folder / 'config.json', dict(catalog_path=str(target / 'catalog.json'), prompt_paths=paths,
                                            jobs=block_jobs, decision_record=str(DECISION)))
        if manifests:
            receipts = []
            for action in ('prep', 'check'):
                argv = [sys.executable, str(COLLECTOR), action, '--manifest', str(folder / 'manifest.json')]
                if action == 'prep':
                    argv += ['--config', str(folder / 'config.json')]
                process = subprocess.run(argv, capture_output=True, text=True)
                receipts.append(dict(action=action, returncode=process.returncode, stdout=process.stdout,
                                     stderr=process.stderr))
                c.save(folder / 'preparation_receipts.json', receipts)
                if process.returncode:
                    raise SystemExit(process.stdout + process.stderr)
            manifest = read(folder / 'manifest.json')
            index.append(dict(batch=batch, manifest=str(folder / 'manifest.json'), planned=len(block_jobs),
                              protocol_sha256=manifest['protocol_sha256']))
    c.save(target / 'planned_positions.json', [dict(**job, status='planned') for job in jobs])
    if index:
        c.save(target / 'collections.json', index)
    c.save(target / 'validation.json', report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pilot', type=int, metavar='N', help='plan N repetitions per cell (1 to 10)')
    parser.add_argument('--tasks', help='comma-separated task ids (default: all 40 catalog tasks)')
    parser.add_argument('--models', default=','.join(MODELS), help='comma-separated: luna,sonnet')
    parser.add_argument('--write', action='store_true', help='save the plan under data/collections/<date>_task-bank-v2/')
    parser.add_argument('--manifests', action='store_true', help='with --write, run central prep and check per block')
    parser.add_argument('--date', default=date.today().isoformat())
    parser.add_argument('--label', default=None, help='subfolder name (default: full or pilot_<N>reps)')
    parser.add_argument('--report', type=Path, help='also save the dry-run report as JSON at this path')
    args = parser.parse_args(argv)
    expected = sorted(read(CATALOG))
    tasks = args.tasks.split(',') if args.tasks else expected
    models = tuple(args.models.split(','))
    assert set(models) <= set(MODELS), 'unsupported model'
    reps = FULL_REPS if args.pilot is None else args.pilot
    assert 1 <= reps <= FULL_REPS, '--pilot takes 1 to 10'
    with tempfile.TemporaryDirectory(prefix='task_bank_v2_runtime_') as work:
        runtime_base = c.load_runtime(Path(work)).c.prior.base
        packages, catalog, prompts, environment = {}, {}, {}, {}
        for task in tasks:
            spec, task_prompts, failures, warnings, notes = check_package(task, runtime_base, set(expected))
            packages[task] = dict(complete=not failures, failures=failures, warnings=warnings, notes=notes)
            if spec is not None and not failures:
                catalog[task] = spec
                prompts.update(task_prompts)
            for entry in notes.get('p11', []):
                key = f"{entry['language']}:{entry['package']}"
                environment.setdefault(key, dict(install=entry['install'], tasks=[]))['tasks'].append(task)
    jobs = plan_jobs(tasks, reps, models)
    ready = [t for t in tasks if packages[t]['complete']]
    if len(ready) == len(tasks):
        validate_plan(jobs, catalog, prompts)
    counts = {field: dict(sorted(Counter(job[field] for job in jobs).items()))
              for field in ('model', 'platform', 'prompt', 'condition', 'global_rep', 'block')}
    report = dict(
        mode='write' if args.write else 'dry_run', v2_folder=str(V2), reps_per_cell=reps, models=list(models),
        tasks_requested=len(tasks), tasks_missing_folder=[t for t in tasks if not (V2 / t).is_dir()],
        tasks_complete=len(ready), tasks_incomplete=sorted(set(tasks) - set(ready)),
        extra_v2_folders=sorted(p.name for p in V2.iterdir() if p.is_dir() and p.name not in expected) if V2.is_dir() else [],
        positions=len(jobs), expected_positions=len(tasks) * len(models) * 3 * 2 * 3 * reps,
        cells=len({slot(job)[:5] for job in jobs}), counts=counts, prompts_rendered=len(prompts),
        prompt_sha256=hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest() if prompts else None,
        decision_record=dict(path=str(DECISION), exists=DECISION.is_file()),
        sonnet_guard=sonnet_guard(), p11_packages=environment, packages=packages)
    assert report['positions'] == report['expected_positions']
    if args.report:
        c.save(args.report, report)
    summary = {k: v for k, v in report.items() if k != 'packages'}
    print(json.dumps(summary, indent=1))
    print('\nPackage checks:')
    for task in tasks:
        state = packages[task]
        print(f"  {task}: {'complete' if state['complete'] else 'INCOMPLETE'}"
              f" ({len(state['failures'])} failures, {len(state['warnings'])} warnings)")
        for failure in state['failures'][:6]:
            print('    - ' + failure)
        if len(state['failures']) > 6:
            print(f"    - ... {len(state['failures']) - 6} more")
    if not args.write:
        print('\nDry run. Nothing written.')
        return 0 if len(ready) == len(tasks) else 1
    if len(ready) != len(tasks):
        raise SystemExit('Refusing --write: some packages are incomplete.')
    label = args.label or ('full' if args.pilot is None else f'pilot_{reps}reps')
    target = ROOT / 'data/collections' / f'{args.date}_task-bank-v2' / label
    write_plan(target, catalog, prompts, jobs, report, args.manifests)
    print(f'\nWrote the plan to {target}. No authorization or launch command was written.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

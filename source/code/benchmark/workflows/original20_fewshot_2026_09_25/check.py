"""Validate a prepared rerun collection before any launch.

The check reads local files and runs the central collector's check for each
block (local files and CLI help only, no model request). It confirms:

- the decision record is APPROVED for the prepared scope (and run list) and unchanged since preparation;
- the position count, cells, blocks, and repetitions match the design;
- every new few-shot prompt equals the production prompt with only its example block replaced,
  and every other prompt equals its production prompt;
- the example files, block parts, expected outputs, source prompts, run list, and catalog still
  have their prepared hashes;
- every planned position links the selected record that it replaces, with an unchanged hash;
- each manifest passes the central collector's check.

On success it writes validation.json in the collection folder. launch.py requires it.
Without --scope, the check uses the only prepared collection.

  uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/check.py
  uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/check.py --scope original20_added12_both_prompts
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import original20_plan as p  # noqa: E402

c = p.c
PATH_FIELDS = {'data_path', 'input_files'}


def flagged_of(plan):
    return (plan.get('run_list') or {}).get('positions', [])


def check_counts(plan, ledger, index):
    problems = []
    scope, flagged = plan['scope'], flagged_of(plan)
    counts = p.expected_counts(scope, flagged)
    if plan['counts'] != counts:
        problems.append('plan counts differ from the design')
    if len(ledger) != counts['positions'] or len({p.slot(row) for row in ledger}) != counts['positions']:
        problems.append(f'ledger has {len(ledger)} rows, expected {counts["positions"]} unique positions')
    tasks = set(p.scope_tasks(scope))
    cells = Counter(p.slot(row)[:5] for row in ledger if row['task'] in tasks)
    if len(cells) * p.REPS != counts['task_positions'] or (cells and set(cells.values()) != {p.REPS}):
        problems.append('cells of the scope tasks do not all have ten repetitions')
    expected = {p.slot(job) for job in p.plan_jobs(scope, flagged)}
    if {p.slot(row) for row in ledger} != expected:
        problems.append('ledger positions differ from the planned design')
    if tasks and {row['prompt'] for row in ledger if row['task'] in tasks} != set(p.scope_prompts(scope)):
        problems.append('ledger prompts differ from the scope')
    if len(index) != counts['blocks'] or {entry['batch']: entry['planned'] for entry in index} != counts['per_block']:
        problems.append('block index differs from the design')
    if len({row['supersedes_record'] for row in ledger}) != counts['positions']:
        problems.append('superseded records are not unique')
    return problems


def check_sources(plan):
    problems = []
    scope, tasks = plan['scope'], p.scope_tasks(plan['scope'])
    sources, provenance = p.load_source_prompts()
    if {key: provenance[key]['sha256'] for key in plan['source_prompt_sha256']} != plan['source_prompt_sha256']:
        problems.append('production source prompts changed since preparation')
    inventory = p.example_inventory(tasks)
    current = {f'{task}/{platform}': c.sha(path) if path else None for (task, platform), path in inventory.items()}
    if current != plan['example_sha256']:
        changed = sorted(key for key in current if current[key] != plan['example_sha256'].get(key))
        problems.append('example files changed since preparation: ' + ', '.join(changed[:6]))
    parts_sha256, expected_sha256 = p.block_file_hashes(tasks)
    if parts_sha256 != plan['parts_sha256']:
        problems.append('block parts changed since preparation')
    if expected_sha256 != plan['expected_sha256']:
        problems.append('expected outputs changed since preparation')
    run_list_sha256 = None
    if p.uses_run_list(scope):
        positions, run_list_sha256, _ = p.read_run_list(plan['run_list']['path'])
        if run_list_sha256 != plan['run_list']['sha256'] or positions != plan['run_list']['positions']:
            problems.append('run list changed since preparation')
    catalog = p.production_catalog(tasks=p.plan_tasks(scope, flagged_of(plan)))
    if c.digest(catalog) != plan['catalog_sha256']:
        problems.append('production catalog changed since preparation')
    if c.sha(p.DECISION) != plan['decision_record_sha256']:
        problems.append('decision record changed since preparation')
    problems += p.decision_problems(scope, run_list_sha256=run_list_sha256)
    if p.plan_digest(plan) != plan['plan_digest']:
        problems.append('plan digest does not match plan.json')
    return problems, sources, inventory, catalog


def check_prompts(plan, sources, inventory, index, catalog):
    """Rebuild every prompt from sources and examples and compare with each manifest."""
    problems = []
    scope = plan['scope']
    tasks = set(p.scope_tasks(scope))
    examples = p.load_examples(inventory)
    expected = p.scope_prompt_texts(scope, sources, examples, flagged_of(plan))
    if {key: p.text_sha(text) for key, text in expected.items()} != plan['prompt_sha256']:
        problems.append('rendered prompts differ from the prepared prompt hashes')
    for key, text in expected.items():
        task, platform, prompt, agency, fmt = key.split('__')
        zero = sources['__'.join((task, platform, 'zero_shot', agency, fmt))]
        if task not in tasks or prompt == 'zero_shot':
            if text != sources[key]:
                problems.append(f'{key}: prompt differs from production')
            continue
        block = p.example_block(platform, *examples[(task, platform)], task=task)
        if text != p.render_few_shot(sources[key], zero, block):
            problems.append(f'{key}: prompt differs from the production prompt with the example block replaced')
        head, _, tail = p.split_block(text, zero)
        if head + tail != zero:
            problems.append(f'{key}: prompt without its block differs from the zero-shot prompt')
    for entry in index:
        manifest = p.read(entry['manifest'])
        protocol = manifest['protocol']
        if c.digest(protocol) != manifest['protocol_sha256'] or manifest['protocol_sha256'] != entry['protocol_sha256']:
            problems.append(f"{entry['batch']}: manifest hash differs from the index")
        if manifest['jobs'] != protocol['planned_jobs'] or len(manifest['jobs']) != entry['planned']:
            problems.append(f"{entry['batch']}: planned jobs differ from the index")
        if Path(protocol['decision_record']).resolve() != p.DECISION.resolve():
            problems.append(f"{entry['batch']}: manifest pins another decision record")
        for key, text in protocol['prompts'].items():
            if key not in expected or text != c.prepared_prompt(expected[key], key.split('__')[3]):
                problems.append(f"{entry['batch']}: manifest prompt {key} differs from the rendered prompt")
        for task, spec in protocol['tasks'].items():
            wanted = catalog[task]
            if {k: v for k, v in spec.items() if k not in PATH_FIELDS} != {k: v for k, v in wanted.items() if k not in PATH_FIELDS}:
                problems.append(f"{entry['batch']}: task {task} differs from the production catalog")
            spec_inputs, wanted_inputs = c.task_input_files(spec), c.task_input_files(wanted)
            if sorted(spec_inputs) != sorted(wanted_inputs) or any(
                    c.sha(spec_inputs[name]) != c.sha(wanted_inputs[name]) for name in wanted_inputs):
                problems.append(f"{entry['batch']}: input of {task} differs from the production catalog")
    return problems


def check_links(ledger):
    problems = []
    fresh = p.baseline_links([{k: row[k] for k in p.SLOT_FIELDS} for row in ledger])
    for row, link in zip(ledger, fresh):
        if row['supersedes_record'] != link['supersedes_record'] or row['supersedes_record_sha256'] != link['supersedes_record_sha256']:
            problems.append(f'{p.slot(row)}: superseded record differs from the baseline selection')
    return problems


def central_checks(index):
    receipts, problems = [], []
    for entry in index:
        process = subprocess.run([sys.executable, str(p.COLLECTOR), 'check', '--manifest', entry['manifest']],
                                 capture_output=True, text=True, cwd=p.ROOT)
        receipts.append(dict(batch=entry['batch'], returncode=process.returncode, stdout=process.stdout,
                             stderr=process.stderr[-8000:]))
        if process.returncode:
            problems.append(f"{entry['batch']}: central check failed: {process.stderr[-500:]}")
    return receipts, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scope', choices=sorted(p.SCOPES), help='scope of the prepared collection to check')
    parser.add_argument('--skip-central', action='store_true', help='skip the central collector check (diagnosis only)')
    args = parser.parse_args(argv)
    scope, meta = p.find_collection(args.scope)
    plan = p.read(meta / 'plan.json')
    ledger = p.read(meta / 'planned_positions.json')
    index = p.read(meta / 'collections.json')
    problems = check_counts(plan, ledger, index)
    source_problems, sources, inventory, catalog = check_sources(plan)
    problems += source_problems
    if not source_problems:
        problems += check_prompts(plan, sources, inventory, index, catalog)
    problems += check_links(ledger)
    receipts = []
    if not args.skip_central:
        receipts, central = central_checks(index)
        c.save(meta / 'validation_checks.json', receipts)
        problems += central
    if problems:
        print('Check failed:')
        for problem in problems[:40]:
            print('  - ' + problem)
        return 1
    if args.skip_central:
        print('Checks passed without the central check. validation.json not written.')
        return 0
    counts = plan['counts']
    change = ('example block of the new few-shot prompts only' if p.scope_tasks(scope)
              else 'none: every prompt equals its production prompt')
    c.save(meta / 'validation.json', dict(passed=True, checked_at=datetime.now(timezone.utc).isoformat(),
        scope=scope, plan_digest=plan['plan_digest'], positions=counts['positions'], cells=counts['cells'],
        repetitions=p.REPS, blocks=counts['blocks'], workers_per_block=p.WORKERS,
        total_workers=p.WORKERS * counts['blocks'], walltime_limits=None, prior_records_linked=len(ledger),
        prompt_change=change, decision_record_sha256=plan['decision_record_sha256']))
    print(f"Passed: {counts['positions']} positions, {counts['cells']} cells, {counts['blocks']} blocks. "
          f'Prompt change: {change}. All prior record hashes verified.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

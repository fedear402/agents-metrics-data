"""Prepare the rerun of the original tasks, the 12 added tasks, or flagged runs.

The default run is a dry run. It counts the planned positions, checks the
decision record, reads the run list of the flagged-run scopes, looks for the
example files of the scope's tasks, runs every present example in an empty
temporary directory, compares each output with the expected output that its
block shows, and renders the prompts. It reports the block lengths by task
half. It writes nothing.

--write builds the collection. It refuses to write when the decision record
status is not APPROVED, when the approved scope differs from --scope, when a
flagged-run scope has no run list or the record approves another run list,
when an example file is missing or fails its check, or when the collection
exists. It then writes the prompts, catalog, and job configuration of each
block, runs the central collector's prep and check for each block (local files
and CLI help only, no model request), and writes the hashed manifests, the
collection index, and the position ledger. It never writes authorization.json.

Examples:
  uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/prepare.py
  uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/prepare.py --scope original20_added12_both_prompts
  uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/prepare.py --scope flagged_runs --run-list <file>
  uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/prepare.py --write --scope few_shot
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import original20_plan as p  # noqa: E402

c = p.c
HALVES = (('original20', p.ORIGINAL_TASKS), ('added12', p.ADDED12_TASKS))


def check_examples(inventory, catalog, run):
    """Return {(task, platform): check} for all required example files."""
    results = {}
    if not run:
        for (task, platform), path in inventory.items():
            failures = [] if path else [f'{p.EXAMPLE_FILES[platform]} missing']
            if path:
                failures += p.static_example_problems(path.read_text(), catalog[task])
                parts, expected, problems = p.block_inputs(path.parent, platform)
                failures += problems + p.block_text_problems(parts, expected, platform, catalog[task])
            results[(task, platform)] = dict(task=task, platform=platform, path=str(path) if path else None,
                                             passed=False, failures=failures + ['example not run (--skip-run)'],
                                             warnings=[], sha256=c.sha(path) if path else None)
        return results
    if not inventory:
        return results
    with tempfile.TemporaryDirectory(prefix='original20_runtime_') as work:
        tools = p.runtime_tools(work)
        for (task, platform), path in inventory.items():
            results[(task, platform)] = p.check_example(task, platform, path, catalog[task], tools)
            state = 'pass' if results[(task, platform)]['passed'] else 'FAIL'
            print(f'  example {task}/{platform}: {state}', flush=True)
    return results


def stats(values):
    values = sorted(values)
    return dict(blocks=len(values), min=values[0], median=statistics.median(values),
                mean=round(statistics.mean(values)), max=values[-1]) if values else None


def block_chars(prompts, sources, scope):
    """Block lengths in characters by task half and language (few-shot minus zero-shot, chatbot)."""
    report = {}
    new = p.block_lengths(prompts, sources, tasks=set(p.scope_tasks(scope)))
    production = {}
    for key in p.prompt_keys(('few_shot',), tasks=p.ALL_TASKS):
        task, platform, _, agency, fmt = key.split('__')
        if agency == 'chatbot':
            production[f'{task}/{platform}'] = len(sources[key]) - len(sources[key.replace('__few_shot__', '__zero_shot__')])
    groups = [(f'{half}_new', tasks, new) for half, tasks in HALVES]
    groups += [('added8_full_layout_production', p.FULL_LAYOUT_TASKS, production)]
    groups += [('added20_new_or_full_layout', p.ADDED20_TASKS, {**production, **{k: v for k, v in new.items()
                                                                                 if k.split('/')[0] in p.ADDED12_TASKS}})]
    groups += [(f'{half}_production', tasks, production) for half, tasks in HALVES]
    for name, tasks, lengths in groups:
        by_platform = {platform: stats([n for key, n in lengths.items()
                                        if key.split('/')[0] in tasks and key.endswith('/' + platform)])
                       for platform in p.PLATFORMS}
        if name.endswith('_new') and not any(by_platform.values()):
            continue
        if name == 'added20_new_or_full_layout' and not any(k.split('/')[0] in p.ADDED12_TASKS for k in new):
            continue
        report[name] = by_platform
    return report


def build_report(args, catalog, sources, inventory, checks, prompts, render_failures, jobs, run_list):
    status, approved = p.decision_state()
    counts = p.expected_counts(args.scope, run_list['positions'])
    tasks = p.scope_tasks(args.scope)
    ready_tasks = sorted({task for task in tasks if all(checks[(task, platform)]['passed'] for platform in p.PLATFORMS)})
    parts_sha256, expected_sha256 = p.block_file_hashes(tasks)
    rendered = sum(key.split('__')[0] in tasks for key in prompts)
    return dict(
        mode='write' if args.write else 'dry_run', scope=args.scope, counts=counts,
        positions=len(jobs), cells=len({p.slot(job)[:5] for job in jobs}), job_counts=p.count_report(jobs),
        decision_record=dict(path=str(p.DECISION), exists=p.DECISION.is_file(), status=status,
                             approved_scope=approved, approved_run_list_sha256=p.approved_run_list(),
                             problems=p.decision_problems(args.scope, run_list_sha256=run_list['sha256'])),
        run_list=run_list,
        examples=dict(folders=sorted({str(p.example_folder(task)) for task in tasks}),
                      present=sum(path is not None for path in inventory.values()),
                      required=len(inventory), passed=sum(check['passed'] for check in checks.values()),
                      tasks_ready=len(ready_tasks),
                      block_parts=sum(value is not None for value in parts_sha256.values()),
                      expected_outputs=sum(value is not None for value in expected_sha256.values())),
        prompts=dict(rendered=len(prompts), expected=counts['prompt_keys'], new_or_production_zero_shot=rendered,
                     production_listed_positions=len(prompts) - rendered, failures=render_failures,
                     sha256=c.digest(prompts) if prompts else None,
                     block_chars=block_chars(prompts, sources, args.scope)),
        collection=dict(folder=str(p.meta(args.scope)), raw=str(p.raw(args.scope))),
        source_prompts=len(sources), catalog_tasks=len(catalog), environment=p.environment_notes(),
        example_checks={f'{task}/{platform}': {k: v for k, v in check.items() if k != 'run'}
                        for (task, platform), check in checks.items()})


def print_report(report):
    summary = {k: v for k, v in report.items() if k not in ('example_checks', 'job_counts')}
    summary['job_counts'] = {k: v for k, v in report['job_counts'].items() if k != 'task'}
    summary['run_list'] = {k: (v if k != 'positions' else len(v)) for k, v in report['run_list'].items()}
    print(json.dumps(summary, indent=1, default=str))
    failing = {name: check for name, check in report['example_checks'].items() if not check['passed']}
    if failing:
        print(f'\nExample checks with failures ({len(failing)} of {len(report["example_checks"])}):')
        for name, check in sorted(failing.items()):
            print(f'  {name}: ' + '; '.join(check['failures'][:3]))
    warned = {name: check for name, check in report['example_checks'].items() if check.get('warnings')}
    for name, check in sorted(warned.items()):
        print(f'  warning {name}: ' + '; '.join(check['warnings'][:3]))


def write_collection(args, catalog, sources, provenance, inventory, checks, prompts, jobs, report, run_list):
    """Write prompts, catalogs, configs, manifests, the index, and the ledger."""
    counts = p.expected_counts(args.scope, run_list['positions'])
    links = p.baseline_links(jobs)
    meta, raw = p.meta(args.scope), p.raw(args.scope)
    meta.mkdir(parents=True, exist_ok=False)
    used = sorted({c.prompt_key(job) for job in jobs} | set(p.prompt_keys(tasks=p.scope_tasks(args.scope))))
    c.save(meta / 'source_prompts.json', dict(folders=[str(folder) for folder in p.PROMPT_FOLDERS],
                                              prompts={key: sources[key] for key in used},
                                              provenance={key: provenance[key] for key in used}))
    c.save(meta / 'example_checks.json', report['example_checks'])
    receipts, index, ledger = [], [], []
    baseline_sha256 = c.sha(p.BASELINE)
    flagged = set(run_list['positions'])
    by_block = {}
    for job, link in zip(jobs, links):
        by_block.setdefault(job['block'], []).append((job, link))
    for model in p.MODELS:
        for offset in range(0, p.REPS, p.BLOCK_REPS):
            batch = p.block_name(model, offset, args.scope)
            if batch not in by_block:
                continue
            folder = raw / model / f'{offset}_{offset + p.BLOCK_REPS - 1}'
            block_jobs = [job for job, _ in by_block[batch]]
            paths = {}
            for key in sorted({c.prompt_key(job) for job in block_jobs}):
                path = folder / 'prompts' / (key + '.txt')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(prompts[key])
                paths[key] = str(path)
            block_catalog = {task: catalog[task] for task in sorted({job['task'] for job in block_jobs})}
            c.save(folder / 'catalog.json', block_catalog)
            c.save(folder / 'config.json', dict(catalog_path=str(folder / 'catalog.json'), prompt_paths=paths,
                                                jobs=block_jobs, decision_record=str(p.DECISION)))
            for action in ('prep', 'check'):
                argv = [sys.executable, str(p.COLLECTOR), action, '--manifest', str(folder / 'manifest.json')]
                if action == 'prep':
                    argv += ['--config', str(folder / 'config.json')]
                process = subprocess.run(argv, capture_output=True, text=True, cwd=p.ROOT)
                receipts.append(dict(batch=batch, action=action, returncode=process.returncode,
                                     stdout=process.stdout, stderr=process.stderr[-8000:]))
                c.save(meta / 'preparation_receipts.json', receipts)
                if process.returncode:
                    raise SystemExit(f'Central {action} failed for {batch}:\n{process.stdout}{process.stderr[-4000:]}')
            manifest = p.read(folder / 'manifest.json')
            entry = dict(batch=batch, manifest=str(folder / 'manifest.json'), planned=len(block_jobs),
                         protocol_sha256=manifest['protocol_sha256'])
            index.append(entry)
            for job, link in by_block[batch]:
                ledger.append(dict(**job, status='planned', manifest=entry['manifest'],
                                   protocol_sha256=entry['protocol_sha256'], **link,
                                   flagged=p.position_id(job) in flagged, prompt_source=p.prompt_source(args.scope, job),
                                   baseline_csv_sha256=baseline_sha256))
            launch = dict(name=batch, argv=['uv', 'run', '--no-sync', 'python', '-u', str(p.QUEUE),
                                            '--index', str(meta / 'collections.json'),
                                            '--solo-workers', str(p.WORKERS), '--overlap-workers', str(p.WORKERS),
                                            '--drain-model', '--batches', batch],
                          cwd=str(p.ROOT), env={'UV_CACHE_DIR': '/private/tmp/agents-metrics-uv-cache'},
                          log=str(meta / (batch + '_collector.log')), index=str(meta / 'collections.json'),
                          batches=[batch])
            c.save(meta / (batch + '_launch_commands.json'), [launch])
    assert len(ledger) == counts['positions'] == len({p.slot(row) for row in ledger})
    c.save(meta / 'collections.json', index)
    c.save(meta / 'planned_positions.json', ledger)
    parts_sha256, expected_sha256 = p.block_file_hashes(p.scope_tasks(args.scope))
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), scope=args.scope, counts=counts,
                seed=p.SEED, block_layout=p.BLOCK_LAYOUT, blocks=index,
                run_list=run_list if p.uses_run_list(args.scope) else None,
                prompt_sha256={key: p.text_sha(text) for key, text in prompts.items()},
                source_prompt_sha256={key: provenance[key]['sha256'] for key in used},
                example_sha256={f'{task}/{platform}': c.sha(path) for (task, platform), path in inventory.items()},
                parts_sha256=parts_sha256, expected_sha256=expected_sha256,
                catalog_sha256=c.digest(catalog), decision_record=str(p.DECISION),
                decision_record_sha256=c.sha(p.DECISION), baseline=str(p.BASELINE),
                baseline_sha256=c.sha(p.BASELINE), collector=str(p.COLLECTOR), workers_per_block=p.WORKERS)
    plan['plan_digest'] = p.plan_digest(plan)
    c.save(meta / 'plan.json', plan)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scope', choices=sorted(p.SCOPES), default='few_shot',
                        help='few_shot (3,600 positions), few_shot_and_zero_shot (7,200), '
                             'original20_added12_both_prompts (11,520), flagged_runs (the run list), or '
                             'original20_added12_both_prompts_and_flagged_runs (11,520 plus listed positions of other tasks)')
    parser.add_argument('--run-list', type=Path, default=None,
                        help=f'run-list file of the flagged-run scopes (default: {p.RUN_LIST})')
    parser.add_argument('--write', action='store_true', help='write the collection (requires an APPROVED record)')
    parser.add_argument('--skip-run', action='store_true', help='dry run only: check files without running examples')
    parser.add_argument('--report', type=Path, help='also save the report as JSON at this path')
    args = parser.parse_args(argv)
    if args.write and args.skip_run:
        parser.error('--write runs every example; drop --skip-run')
    run_list = dict(path=None, sha256=None, positions=[], problems=[])
    if p.uses_run_list(args.scope):
        path = args.run_list or p.RUN_LIST
        positions, sha256, problems = p.read_run_list(path)
        run_list = dict(path=str(path), sha256=sha256, positions=positions, problems=problems)
    elif args.run_list:
        parser.error('--run-list applies only to the flagged-run scopes')
    flagged = run_list['positions']
    catalog = p.production_catalog(tasks=p.plan_tasks(args.scope, flagged))
    sources, provenance = p.load_source_prompts()
    inventory = p.example_inventory(p.scope_tasks(args.scope))
    jobs = p.plan_jobs(args.scope, flagged)
    counts = p.expected_counts(args.scope, flagged)
    assert len(jobs) == counts['positions'] == len({p.slot(job) for job in jobs})
    for batch, size in counts['per_block'].items():
        assert sum(job['block'] == batch for job in jobs) == size
    print(f"Planned positions: {counts['positions']} ({args.scope}). Checking examples.", flush=True)
    checks = check_examples(inventory, catalog, run=not args.skip_run)
    examples = p.load_examples({key: path for key, path in inventory.items() if checks[key]['passed']})
    prompts, render_failures = {}, []
    for task in p.scope_tasks(args.scope):
        if not all((task, platform) in examples for platform in p.PLATFORMS):
            continue
        try:
            prompts.update(p.render_prompts(sources, examples, args.scope, tasks=(task,)))
        except ValueError as error:
            render_failures.append(f'{task}: {error}')
    try:
        prompts.update(p.production_prompts_for(p.flagged_extra(args.scope, flagged), sources))
    except ValueError as error:
        render_failures.append(f'listed positions: {error}')
    if len(prompts) == counts['prompt_keys']:
        # One manifest holds local repetitions 0 through 4, so the collector checks each block.
        for batch in sorted({job['block'] for job in jobs}):
            block = [job for job in jobs if job['block'] == batch]
            c.validate_jobs(block, dict(tasks=catalog, prompts=prompts))
    report = build_report(args, catalog, sources, inventory, checks, prompts, render_failures, jobs, run_list)
    if args.report:
        c.save(args.report, report)
    print_report(report)
    problems = list(report['decision_record']['problems']) + list(run_list['problems'])
    if report['examples']['passed'] != report['examples']['required']:
        problems.append(f"{report['examples']['passed']} of {report['examples']['required']} example files pass")
    if len(prompts) != counts['prompt_keys']:
        problems.append(f"{len(prompts)} of {counts['prompt_keys']} prompts rendered")
    if not jobs:
        problems.append('the scope plans no position')
    if not p.SONNET_GUARD.is_file():
        problems.append(f'Sonnet guard missing: {p.SONNET_GUARD}')
    if p.meta(args.scope).exists():
        problems.append(f'collection folder already exists: {p.meta(args.scope)}')
    if not args.write:
        print('\nDry run. Nothing written.')
        print('Not ready for --write: ' + '; '.join(problems) if problems else 'Ready for --write.')
        return 0
    if problems:
        raise SystemExit('Refusing --write: ' + '; '.join(problems))
    plan = write_collection(args, catalog, sources, provenance, inventory, checks, prompts, jobs, report, run_list)
    print(f"\nWrote {counts['positions']} planned positions in {len(plan['blocks'])} blocks to {p.meta(args.scope)}.")
    print('Plan digest: ' + plan['plan_digest'])
    print('No authorization file or launch was written. Run check.py next.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

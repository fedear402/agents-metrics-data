"""Fresh 144-response example pilot. Preparation makes no provider requests."""
from __future__ import annotations
import argparse
import copy
import fcntl
import importlib.util
from ._loader import load_sibling
import importlib.metadata
import itertools
import json
from pathlib import Path
import random
import signal
import sys

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'quality_reports/reports/2026-09-16_clearer-examples'
WORK = REPORT / 'work/collector'
PROMPTS = REPORT / 'work/prompts'
PREVIOUS = ROOT / 'quality_reports/reports/2026-09-16_expanded-prompt-evaluation'
SOURCE = Path(__file__).with_name('expanded_recovery.py')
r = load_sibling(__package__, 'expanded_recovery')
m, c = r.m, r.c
BASE_ADAPT = m.adapt_record
TASKS = ['applied_shift_share', 'broad_tobit', 'applied_cumulative_hac', 'hard_rolling']
VARIANTS = ['zero', 'few_shot', 'revised']
BASELINES = [PREVIOUS / 'work/collector/applied8_v2/manifest.json',
             PREVIOUS / 'work/repair/v2/manifest.json',
             ROOT / 'quality_reports/reports/2026-09-16_eighteen-task-extension/work/collector/manifest.json']
POLICY = dict(m.POLICY, version='clearer_examples_fresh144_v1', startup=12,
              repetitions=[0, 1, 2], prompt_variants=VARIANTS, workers=8,
              shuffle_seed=2026091605,
              continuation='Review initial12 for protocol compliance. Continue fixed132 independently of effect signs.',
              pause_rule='Killed subprocesses without timeout remain null pending administrative review. Never restart them automatically.',
              primary_comparison='All144 fresh responses. No earlier observation reuse.')
m.WORK = c.WORK = WORK
m.POLICY = c.POLICY = POLICY


def jobs():
    rows = [dict(block='clearer144', task=t, model=model, platform='python', prompt=p,
                 condition=a, output_format='json', rep=rep)
            for t, model, p, a, rep in itertools.product(TASKS, ['luna', 'sonnet'], VARIANTS,
                                                         ['chatbot', 'agent'], range(3))]
    random.Random(POLICY['shuffle_seed']).shuffle(rows)
    return rows


def preflight_jobs(manifest):
    return [j for j in manifest['jobs'] if j['task'] == TASKS[0] and j['rep'] == 0]


def adapt_record(raw, destination):
    record = BASE_ADAPT(raw, destination)
    interrupted = any(isinstance(item.get('returncode'), int) and item['returncode'] < 0
                      and not item.get('timed_out')
                      for item in raw.get('calls', []) + raw.get('executions', []))
    if interrupted:
        record['status'] = 'needs_review'
        record['outcome'].update(primary_success=None, numerical_success=None, data_script_integrity=None,
                                 protocol_status='hold', measurement_status='interrupted_pending_review',
                                 retry_eligible=False)
        record['scoring'].update(success=None, numeric_success=None, integrity_pass=None)
        c.STOP.set()
    record['runtime_adapter'] = dict(source=str(Path(__file__)), source_sha256=c.base.sha(Path(__file__)),
                                    version=POLICY['version'], administrative_pause_guard=True)
    return record


def baseline_map():
    manifests = [json.loads(path.read_text()) for path in BASELINES]
    return {task: next(x for x in reversed(manifests) if task in x['protocol']['tasks']) for task in TASKS}


def make_manifest(decision):
    decision = Path(decision).resolve()
    catalog = json.loads((PROMPTS / 'catalog.json').read_text())
    pairs = json.loads((PROMPTS / 'prompt_pairs.json').read_text())['pairs']
    assert len(pairs) == 8
    lookup = {(p['task_id'], p['condition']): p for p in pairs}
    assert len(lookup) == 8
    old = baseline_map()
    specs = {task: catalog[task] for task in TASKS}
    for task in TASKS:
        assert specs[task] == old[task]['protocol']['tasks'][task], 'Task/reference changed: ' + task
    pins = {}
    for baseline in old.values():
        pins.update(baseline['protocol']['source_hashes'])
    for path in [Path(__file__), Path(__file__).with_name('test_collector.py'), SOURCE, *BASELINES, decision, PROMPTS / 'catalog.json', PROMPTS / 'prompt_pairs.json']:
        pins[str(path)] = c.base.sha(path)
    texts, diffs = {}, {}
    for task, condition in itertools.product(TASKS, ['chatbot', 'agent']):
        pair = lookup[task, condition]
        for variant, field in [('zero', 'zero_path'), ('few_shot', 'few_path'), ('revised', 'revised_path')]:
            path = Path(pair[field]).resolve()
            original = path.read_text()
            text = original.replace('Run the script with python3 run.py.', f'Run the script with {sys.executable} run.py.') if condition == 'agent' else original
            job = dict(task=task, platform='python', prompt=variant, condition=condition, output_format='json')
            key = c.prompt_key(job)
            pins[str(path)] = c.base.sha(path)
            if text != original:
                diffs[key] = ''.join(c.difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile=str(path), tofile='prior_pinned_interpreter'))
            baseline = old[task]['protocol']['prompts'][c.prompt_key(dict(job, prompt='zero'))]
            assert '\nTask:' in text and text.split('\nTask:', 1)[1] == baseline.split('\nTask:', 1)[1], 'Target/reporting change: ' + key
            if variant != 'revised':
                assert text == old[task]['protocol']['prompts'][key], 'Old control changed: ' + key
            texts[key] = text
    commands = {f'{model}/{agency}': c.command(dict(model=model, condition=agency), Path('<isolated-cwd>'), Path('<last-message>'))
                for model, agency in itertools.product(['luna', 'sonnet'], ['chatbot', 'agent'])}
    for baseline in old.values():
        assert c.digest(commands) == c.digest(baseline['protocol']['commands']), 'Provider treatment changed'
    reference = old[TASKS[0]]['protocol']
    checks = c.prior.check_tools()
    versions = {k: {field: checks[k][field] for field in ['argv', 'returncode', 'stdout', 'stderr']}
                for k in ['codex_version', 'sonnet_version']}
    protocol = dict(block='clearer144', policy=POLICY, tasks=specs, prompts=texts,
                    runtime_prompt_diffs=diffs, commands=commands, source_hashes=pins,
                    environment=reference['environment'], package_environment=reference['package_environment'],
                    scoring_fields=reference['scoring_fields'], cli_versions=versions,
                    runtime_lineage=dict(source=str(SOURCE), source_sha256=c.base.sha(SOURCE),
                                         baselines=[str(x) for x in BASELINES], prior_responses_reused=False),
                    decision_record=str(decision))
    return dict(created_at=c.now(), protocol_sha256=c.digest(protocol), protocol=protocol, jobs=jobs())


def require_review(manifest, selected, run_root, path):
    c.require_preflight_review(manifest, selected, run_root, path)
    if {c.job_id(j) for j in selected} <= {c.job_id(j) for j in preflight_jobs(manifest)}:
        return
    review = json.loads(Path(path).read_text())
    for key in ['initial12_protocol_reviewed', 'outcome_rule_confirmed', 'continuation_independent_of_effect_sign']:
        if review.get(key) is not True:
            raise ValueError('Missing startup review: ' + key)


def validate_runtime(manifest):
    c.verify_manifest(manifest)
    checks = c.prior.check_tools()
    for key, value in manifest['protocol']['cli_versions'].items():
        if m.version_identity(checks[key]) != m.version_identity(value):
            raise ValueError('CLI version changed')
    env = manifest['protocol']['package_environment']
    if env['executable'] != sys.executable or env['python'] != sys.version:
        raise ValueError('Python environment changed')
    for package, version in env['packages'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Package version changed: ' + package)
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prep', 'check', 'status', 'collect', 'adjudicate'])
    parser.add_argument('--manifest', type=Path, default=WORK / 'manifest.json')
    parser.add_argument('--decision', type=Path)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--jobs-file', type=Path)
    parser.add_argument('--preflight-review', type=Path)
    parser.add_argument('--record', type=Path)
    parser.add_argument('--reason')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    args = parser.parse_args(argv)
    if args.action == 'adjudicate':
        if not args.record or not args.reason:
            parser.error('--record and --reason are required')
        m.adjudicate(args.record, args.reason)
        return
    if args.action == 'prep':
        if not args.decision:
            parser.error('--decision is required')
        manifest = make_manifest(args.decision)
        if args.manifest.exists() and json.loads(args.manifest.read_text())['protocol_sha256'] != manifest['protocol_sha256']:
            raise ValueError('Preserve existing manifest. Use a new version after changes.')
        c.save(args.manifest, manifest)
        c.save(args.manifest.parent / 'preflight_jobs.json', preflight_jobs(manifest))
        c.save(args.manifest.parent / 'all_jobs.json', manifest['jobs'])
        print(json.dumps(dict(jobs=144, startup=12, protocol_sha256=manifest['protocol_sha256'])))
        return
    manifest = json.loads(args.manifest.read_text())
    run_root = args.manifest.parent / 'runs' / manifest['protocol_sha256']
    if args.action == 'status':
        print(json.dumps(m.status(manifest, run_root), indent=2))
        return
    checks = validate_runtime(manifest)
    if args.action == 'check':
        c.save(args.manifest.parent / 'offline_check.json', dict(provider_calls=0, tools=checks))
        print('Offline source and CLI checks passed.')
        return
    c.require_authorization(manifest, args.authorization)
    selected = c.select_jobs(manifest, args.jobs_file)
    require_review(manifest, selected, run_root, args.preflight_review)
    for signum in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(signum, c.prior.stop_children)
    with (args.manifest.parent / 'collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for job in selected:
            folder = run_root / c.job_id(job)
            if m.completed_record(folder, manifest['protocol_sha256']) is not None:
                continue
            m.validate_resume(folder, manifest['protocol_sha256'])
            pending.append(job)
        c.collect_queue(pending, manifest, run_root, args.workers, run=m.run_job)


m.adapt_record = adapt_record
m.preflight_jobs = c.preflight_jobs = preflight_jobs

if __name__ == '__main__':
    main()

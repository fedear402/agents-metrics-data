"""Wait for all 720 terminal records, then rebuild the canonical paper locally.

--once and --dry-run inspect readiness only. Neither starts a rebuild.
The continuous mode holds a wake lock, records state, and retries build errors.
This program never starts a collector or calls a model provider.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import saved_response_analysis as saved

META = ROOT / 'data/collections/2026-09-21_production-repairs'
REBUILD = ROOT / 'code/paper/rebuild.py'
TASKS = ('hard_panel_lp_iv', 'lp04_ppml')


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def assignments(meta=META):
    """Validate complete membership and both recorded protocol hash bindings."""
    positions, plan_hash = saved.read_hashed(meta / 'planned_positions.json')
    entries, index_hash = saved.read_hashed(meta / 'collections.json')
    expected = set(itertools.product(TASKS, ('luna', 'sonnet'), ('python', 'r', 'stata'),
        ('zero_shot', 'few_shot'), ('chatbot', 'self_repair', 'agent'), range(10)))
    if len(positions) != 720 or {saved.key(p) for p in positions} != expected:
        raise ValueError('The plan must contain all 720 distinct expected positions')
    indexed = {str(Path(e['manifest']).resolve()): e for e in entries}
    if len(entries) != len(indexed) or len(entries) != 4:
        raise ValueError('Expected four distinct collection manifests')
    manifests, jobs, used, receipts = {}, {}, set(), []
    for path, entry in indexed.items():
        manifest, digest = saved.read_hashed(path)
        protocol = manifest['protocol']
        identity = saved.runtime.digest(protocol)
        if identity != manifest['protocol_sha256'] or identity != entry['protocol_sha256']:
            raise ValueError(f'Manifest hash differs from collection index: {path}')
        if manifest['jobs'] != protocol['planned_jobs'] or len(manifest['jobs']) != entry['planned'] or entry['planned'] != 180:
            raise ValueError(f'Changed manifest membership: {path}')
        mapped = {saved.key(j): j for j in manifest['jobs']}
        if len(mapped) != len(manifest['jobs']):
            raise ValueError(f'Duplicate manifest position: {path}')
        manifests[path], jobs[path] = manifest, mapped
        receipts.append(dict(path=path, sha256=digest, protocol_sha256=identity))
    result = []
    for position in positions:
        path = str(Path(position['manifest']).resolve())
        if path not in manifests:
            raise ValueError(f'Position points outside collection index: {path}')
        manifest = manifests[path]
        if position['protocol_sha256'] != manifest['protocol_sha256']:
            raise ValueError(f'Position protocol hash changed: {saved.key(position)}')
        job = jobs[path].get(saved.key(position))
        if job is None:
            raise ValueError(f'Position has no assigned manifest job: {saved.key(position)}')
        if any(position.get(k) != v for k, v in job.items()):
            raise ValueError(f'Plan and manifest job fields differ: {saved.key(position)}')
        used.add((path, saved.key(job)))
        folder = Path(path).parent / 'runs' / manifest['protocol_sha256'] / saved.runtime.job_id(job)
        result.append(dict(job=job, protocol_sha256=manifest['protocol_sha256'], folder=str(folder)))
    if len(used) != sum(len(m['jobs']) for m in manifests.values()):
        raise ValueError('Some manifest jobs lack a unique planned position')
    return result, dict(plan_sha256=plan_hash, index_sha256=index_hash, manifests=receipts)


def readiness(meta=META):
    assigned, provenance = assignments(meta)
    counts, by_block, records = Counter(), {}, []
    for assignment in assigned:
        selected = saved.fresh_record(assignment)
        status = selected['analysis_status']
        counts[status] += 1
        block = assignment['job']['block']
        by_block.setdefault(block, Counter())[status] += 1
        if selected['record'] is not None:
            records.append(dict(position=list(saved.key(assignment['job'])),
                path=str(selected['record_path']), sha256=selected['record_sha256'],
                collection_status=selected['collection_status']))
    if len({r['path'] for r in records}) != len(records):
        raise ValueError('One terminal record is assigned to multiple positions')
    return dict(at=now(), ready=len(records) == 720, total=720, terminal=len(records),
        remaining=720-len(records), counts=dict(counts), by_block=by_block,
        records=records, provider_calls=0, **provenance)


def supervise(poll_seconds, retry_seconds):
    """Use configured polling policies and hold a wake lock through completion."""
    out = META / 'finalizer'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'finalizer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A finalizer or its rebuild already holds the collection lock')
        complete = out / 'complete.json'
        if complete.exists():
            print(complete.read_text())
            return
        wake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        command = ['uv', 'run', '--no-sync', 'python', str(REBUILD)]
        identity = dict(pid=os.getpid(), caffeinate_pid=wake.pid, command=command,
            provider_calls=0, poll_policy_seconds=poll_seconds, retry_policy_seconds=retry_seconds)
        write(out / 'launch.json', dict(at=now(), **identity))
        attempt = 0
        try:
            while True:
                try:
                    observed = readiness()
                    write(out / 'readiness.json', observed)
                    if not observed['ready']:
                        write(out / 'state.json', dict(at=now(), status='waiting', terminal=observed['terminal'], total=720, **identity))
                        time.sleep(poll_seconds)
                        continue
                    attempt += 1
                    log_path = out / f'rebuild_{attempt:04d}_{int(time.time())}.log'
                    write(out / 'state.json', dict(at=now(), status='building', attempt=attempt,
                        log=str(log_path), terminal=720, **identity))
                    with log_path.open('a') as log:
                        # The child inherits the lock so a parent interruption cannot start a second rebuild.
                        process = subprocess.Popen(command, cwd=ROOT,
                            env={**os.environ, 'UV_CACHE_DIR': '/private/tmp/agents-metrics-uv-cache'},
                            stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                        code = process.wait()
                    if code:
                        raise RuntimeError(f'Canonical rebuild returned {code}; see {log_path}')
                    snapshot, snapshot_hash = saved.read_hashed(ROOT / 'data/analysis/main/data/snapshot.json')
                    if not snapshot.get('production_repairs_complete') or snapshot.get('replaced_positions') != 720:
                        raise ValueError('Canonical rebuild receipt does not confirm all 720 replacements')
                    done = dict(at=now(), status='complete', attempt=attempt, log=str(log_path),
                        snapshot_sha256=snapshot_hash, replaced_positions=720, **identity)
                    write(complete, done)
                    write(out / 'state.json', done)
                    print(json.dumps(done), flush=True)
                    return
                except Exception as error:
                    failure = dict(at=now(), status='retry_wait', error=f'{type(error).__name__}: {error}', attempt=attempt, **identity)
                    write(out / 'state.json', failure)
                    with (out / 'errors.jsonl').open('a') as stream:
                        stream.write(json.dumps(failure) + '\n')
                    print(json.dumps(failure), flush=True)
                    time.sleep(retry_seconds)
        finally:
            wake.terminate()
            wake.wait()


class SelectionTests(unittest.TestCase):
    """Exercise rejection paths without providers or generated scripts."""
    def test_complete_plan_and_pinned_membership(self):
        with tempfile.TemporaryDirectory(prefix='production_finalizer_', dir='/private/tmp') as tmp:
            root = Path(tmp)
            positions, entries = [], []
            for model in ('luna','sonnet'):
                for offset in (0,5):
                    path = root / model / str(offset) / 'manifest.json'
                    jobs = [dict(task=t, model=model, platform=platform, prompt=prompt, condition=condition,
                        output_format='answer' if platform == 'stata' else 'json',
                        global_rep=rep, rep=rep-offset, block=f'{model}_{offset}')
                        for t, platform, prompt, condition, rep in itertools.product(TASKS,
                            ('python','r','stata'), ('zero_shot','few_shot'),
                            ('chatbot','self_repair','agent'), range(offset,offset+5))]
                    protocol = dict(planned_jobs=jobs)
                    identity = saved.runtime.digest(protocol)
                    write(path, dict(protocol=protocol, protocol_sha256=identity, jobs=jobs))
                    entries.append(dict(manifest=str(path), protocol_sha256=identity, planned=180))
                    positions.extend(dict(j, manifest=str(path), protocol_sha256=identity) for j in jobs)
            write(root / 'collections.json',entries)
            write(root / 'planned_positions.json',positions)
            self.assertEqual(len(assignments(root)[0]),720)
            positions[0]['protocol_sha256'] = 'changed'
            write(root / 'planned_positions.json',positions)
            with self.assertRaisesRegex(ValueError,'Position protocol hash changed'):
                assignments(root)
            positions[0] = positions[1]
            write(root / 'planned_positions.json',positions)
            with self.assertRaisesRegex(ValueError,'720 distinct'):
                assignments(root)

    def test_selection_and_duplicate_terminal(self):
        with tempfile.TemporaryDirectory(prefix='production_finalizer_', dir='/private/tmp') as tmp:
            folder = Path(tmp)
            job = dict(task=TASKS[0], model='luna', platform='python', prompt='zero_shot', condition='chatbot', global_rep=0)
            assigned = dict(folder=tmp, job=job, protocol_sha256='protocol')
            self.assertIsNone(saved.fresh_record(assigned)['record'])
            path = folder / 'attempt_0001/run_record.json'
            record = dict(job=job, protocol_sha256='protocol', status='completed')
            write(path, record)
            chosen = saved.fresh_record(assigned)
            self.assertEqual(chosen['record'], record)
            saved.read_hashed(path, chosen['record_sha256'])
            write(folder / 'attempt_0002/run_record.json', record)
            with self.assertRaisesRegex(ValueError, 'Multiple terminal'):
                saved.fresh_record(assigned)

    def test_changed_hash_and_job_rejected(self):
        with tempfile.TemporaryDirectory(prefix='production_finalizer_', dir='/private/tmp') as tmp:
            path = Path(tmp) / 'attempt_0001/run_record.json'
            job = dict(task=TASKS[0], platform='r')
            write(path, dict(job=job, protocol_sha256='wrong', status='completed'))
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                saved.fresh_record(dict(folder=tmp, job=job, protocol_sha256='expected'))
            record, digest = saved.read_hashed(path)
            record['extra'] = True
            write(path, record)
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                saved.read_hashed(path, digest)

    def test_terminal_state_waits_for_record(self):
        with tempfile.TemporaryDirectory(prefix='production_finalizer_', dir='/private/tmp') as tmp:
            job = dict(task=TASKS[0], platform='python')
            write(Path(tmp) / 'attempt_0001/state.json', dict(job=job, protocol_sha256='p', status='completed'))
            result = saved.fresh_record(dict(folder=tmp, job=job, protocol_sha256='p'))
            self.assertEqual(result['analysis_status'], 'awaiting_record')
            self.assertIsNone(result['record'])

    def test_retained_hash_and_identity(self):
        with tempfile.TemporaryDirectory(prefix='production_finalizer_', dir='/private/tmp') as tmp:
            path = Path(tmp) / 'run_record.json'
            write(path, dict(task_id=TASKS[0], platform='r', status='completed'))
            _, digest = saved.read_hashed(path)
            position = dict(record_path=str(path), record_sha256=digest, task=TASKS[0], platform='r')
            self.assertIsNotNone(saved.retained_record(position)['record'])
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                saved.retained_record(dict(position, platform='python'))
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                saved.retained_record(dict(position, record_sha256='changed'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true', help='Inspect readiness once; do not rebuild')
    parser.add_argument('--dry-run', action='store_true', help='Inspect readiness once; do not write or rebuild')
    parser.add_argument('--self-test', action='store_true', help='Run local record selection regression checks')
    parser.add_argument('--poll-seconds', type=float, default=60, help='Chosen readiness polling policy')
    parser.add_argument('--retry-seconds', type=float, default=300, help='Chosen error retry policy')
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelectionTests)
        raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    if args.poll_seconds <= 0 or args.retry_seconds <= 0:
        parser.error('Polling and retry policy intervals must be positive')
    if args.once or args.dry_run:
        result = readiness()
        if args.once and not args.dry_run:
            write(META / 'finalizer/readiness.json', result)
        print(json.dumps({k:v for k,v in result.items() if k != 'records'}, default=str))
    else:
        supervise(args.poll_seconds, args.retry_seconds)

"""Hand untouched jobs to an operational adapter while their idle collector waits.

Pinned sources, manifests, job definitions, prompts, and previous responses stay unchanged.
Each handoff uses eight workers. Three shared permits serve supplemental Stata work.
The existing collector keeps its original single Stata permit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import prospective_collect_unlimited_additions as adapter
import production_queue as queue

central = adapter.central
META = ROOT / 'data/collections/2026-09-21_production-repairs'
OPS = META / 'acceleration'
HERE = Path(__file__).resolve()
WORKERS = 8
PERMITS = 3
CURRENT_OPERATION = None
DECISION = ROOT / 'quality_reports/decision_records/2026-09-21_collection-throughput.md'
LAUNCHED_WORKERS = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    central.save(Path(path), value)


def process_snapshot():
    result = subprocess.run(['ps', '-axo', 'pid=,ppid=,stat=,lstart=,command='],
        check=True, capture_output=True, text=True)
    found = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 8)
        if len(fields) == 9:
            found[int(fields[0])] = dict(ppid=int(fields[1]), state=fields[2],
                started=' '.join(fields[3:8]), command=fields[8])
    return found


def descendants(pid, processes):
    found = set()
    for candidate, value in processes.items():
        ancestor, seen = value['ppid'], set()
        while ancestor in processes and ancestor not in seen:
            if ancestor == pid:
                found.add(candidate)
                break
            seen.add(ancestor)
            ancestor = processes[ancestor]['ppid']
    return found


def run_root(manifest_path, manifest):
    return Path(manifest_path).parent / 'runs' / manifest['protocol_sha256']


def active_states(manifest_path):
    manifest = read(manifest_path)
    result = []
    for job in manifest['jobs']:
        folder = run_root(manifest_path, manifest) / central.job_id(job)
        path = folder / 'attempt_0001/state.json'
        if path.exists():
            state = read(path)
            if state['status'] not in central.TERMINAL:
                result.append((job, state, path))
    return result


def external_stata_holder(pid, processes):
    """A live Stata agent child proves another collector still holds its permit."""
    for entry in read(META / 'collections.json'):
        for job, state, path in active_states(entry['manifest']):
            if job['platform'] != 'stata' or job['condition'] != 'agent' or state['stage'] != 'provider_in_flight':
                continue
            for active in path.parent.glob('provider_attempt*.active.json'):
                info = read(active)
                child = processes.get(info['pid'])
                if not child or child['ppid'] == pid or any(flag in child['state'] for flag in ('Z', 'T')):
                    continue
                parent = processes.get(child['ppid'], {})
                if 'prospective_collect_unlimited_additions.py collect' not in parent.get('command', ''):
                    continue
                if 'T' in parent.get('state', ''):
                    continue
                if not any(name in child['command'] for name in ('claude', 'codex')):
                    continue
                if process_cwd(info['pid']) != str(Path(info['cwd']).resolve()):
                    continue
                return dict(collector_pid=child['ppid'], provider_pid=info['pid'], job=job,
                            collector_started=parent['started'], provider_started=child['started'])
    return None


def eligible(manifest_path, pid, processes, expected_started=None):
    process = processes.get(pid)
    if not process or 'prospective_collect_unlimited_additions.py collect' not in process['command']:
        return False, 'collector identity absent'
    if str(Path(manifest_path).resolve()) not in process['command']:
        return False, 'collector manifest differs'
    if expected_started and expected_started != process['started']:
        return False, 'collector process identity changed'
    if descendants(pid, processes):
        return False, 'collector owns live descendants'
    states = active_states(manifest_path)
    if len(states) != WORKERS:
        return False, 'collector does not have eight occupied positions'
    if any(job['platform'] != 'stata' or state['status'] != 'in_progress'
           or state['stage'] not in {'evaluate', 'provider_in_flight'} for job, state, path in states):
        return False, 'occupied positions are not all Stata waiters'
    holder = external_stata_holder(pid, processes)
    if holder is None:
        return False, 'no live external Stata agent proves permit ownership'
    return True, holder


def pending_jobs(manifest_path):
    manifest = read(manifest_path)
    pending = []
    for job in manifest['jobs']:
        folder = run_root(manifest_path, manifest) / central.job_id(job)
        if any(folder.glob('attempt_*')):
            continue
        if (folder / '.lock').exists():
            with (folder / '.lock').open('r') as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                fcntl.flock(stream, fcntl.LOCK_UN)
        pending.append(job)
    # Preserve the original order within each language group.
    return sorted(pending, key=lambda job: job['platform'] == 'stata')


def quota_clear(path):
    if not Path(path).exists():
        return True
    try:
        value = read(path)
    except (ValueError, OSError):
        return False
    return isinstance(value, dict) and not value


def process_cwd(pid):
    result = subprocess.run(['lsof', '-a', '-p', str(pid), '-d', 'cwd', '-Fn'],
        capture_output=True, text=True)
    if result.returncode:
        return None
    directories = [line[1:] for line in result.stdout.splitlines() if line.startswith('n')]
    return directories[0] if len(directories) == 1 else None


def owned_live_calls(operation, processes=None):
    """Match saved process identities by their unique working directory."""
    processes = processes or process_snapshot()
    found = []
    for job in operation.get('jobs', []):
        folder = Path(operation['run_root']) / central.job_id(job)
        for path in folder.glob('attempt_*/*.active.json'):
            info = read(path)
            pid = info.get('pid')
            if pid not in processes:
                continue
            # PID values can be reused. Inspect cwd before treating a saved PID as live.
            directory = process_cwd(pid)
            if directory is None or str(Path(info['cwd']).resolve()) == directory:
                found.append(dict(pid=pid, active_file=str(path), process=processes[pid]))
    return found


def resume_problem(operation, processes=None):
    processes = processes or process_snapshot()
    if owned_live_calls(operation, processes):
        return 'supplemental calls remain alive'
    if not quota_clear(operation['quota_path']):
        return 'provider quota state remains unresolved'
    for job in operation.get('jobs', []):
        folder = Path(operation['run_root']) / central.job_id(job)
        states = list(folder.glob('attempt_*/state.json'))
        if states and central.completed(folder, operation['protocol_sha256']) is None:
            return f'unresolved supplemental record: {central.job_id(job)}'
    return None


def resume_original(operation, processes=None):
    processes = processes or process_snapshot()
    original = processes.get(operation['original_pid'])
    if original is None:
        return 'original collector exited'
    if original['started'] != operation['original_started']:
        return 'original PID was reused'
    problem = resume_problem(operation, processes)
    if problem:
        return problem
    os.kill(operation['original_pid'], signal.SIGCONT)
    return None


@contextmanager
def sidecar_stata_slot(job):
    if job['platform'] != 'stata':
        yield
        return
    permits = OPS / 'stata_permits'
    permits.mkdir(parents=True, exist_ok=True)
    acquired = None
    while acquired is None:
        if central.STOP.is_set():
            raise central.Paused('Supplemental adapter stopped before native permit')
        for index in range(PERMITS):
            stream = (permits / f'permit_{index}.lock').open('a')
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            lease = permits / f'permit_{index}.json'
            previous = read(lease) if lease.exists() else None
            if previous and previous.get('active'):
                old_operation = read(previous['operation'])
                if resume_problem(old_operation):
                    stream.close()
                    continue
            save(lease, dict(active=True, operation=str(CURRENT_OPERATION), pid=os.getpid(), job=job, at=now()))
            acquired = (stream, lease)
            break
        if acquired is None:
            time.sleep(0.25)  # Chosen local permit polling policy.
    try:
        yield
    finally:
        stream, lease = acquired
        save(lease, dict(active=False, operation=str(CURRENT_OPERATION), pid=os.getpid(), at=now()))
        stream.close()


def verify_sources(manifest):
    central.verify(manifest)
    return {path: central.sha(path) for path in manifest['protocol']['source_hashes']}


def record(operation_path, **updates):
    operation = read(operation_path)
    operation.update(at=now(), **updates)
    save(operation_path, operation)
    return operation


def retryable_operation(path, processes=None):
    """Retry a raced eligibility check only after its empty handoff fully exits."""
    operation = read(path)
    if operation.get('jobs') or operation.get('status') not in {'resumed', 'not_suspended'}:
        return False
    processes = processes or process_snapshot()
    worker = processes.get(operation['worker_pid'])
    if worker and worker['started'] == operation['worker_started']:
        return False
    if operation.get('watchdog_pid') in processes:
        return False
    original = processes.get(operation['original_pid'])
    return original is not None and original['started'] == operation['original_started'] and 'T' not in original['state']


def handoff(manifest_path, pid):
    global CURRENT_OPERATION
    manifest_path = Path(manifest_path).resolve()
    manifest = read(manifest_path)
    operation_dir = OPS / manifest_path.parent.parent.name / manifest_path.parent.name
    operation_dir.mkdir(parents=True, exist_ok=True)
    with (operation_dir / 'handoff.lock').open('a') as operation_lock:
        fcntl.flock(operation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = operation_dir / 'operation.json'
        if path.exists():
            if not retryable_operation(path):
                raise ValueError('This manifest already has a handoff operation')
            archived = operation_dir / 'previous_attempts'
            archived.mkdir(exist_ok=True)
            path.replace(archived / f'operation_{time.time_ns()}.json')
        authorization = read(manifest_path.parent / 'authorization.json')
        if not DECISION.is_file():
            raise ValueError('The operational decision record is missing')
        if authorization.get('allow_provider_calls') is not True or authorization.get('protocol_sha256') != manifest['protocol_sha256']:
            raise ValueError('Existing authorization does not match this manifest')
        pinned = verify_sources(manifest)
        runtime = central.load_runtime(manifest_path.parent)
        os.environ['PATH'] = runtime.c.environment()['PATH']
        # Apply unchanged command and interpreter checks without waiting for native Stata.
        for job in manifest['jobs']:
            current = runtime.c.command(job, Path('<cwd>'), Path('<last>'))
            expected = manifest['protocol']['commands'][f"{job['model']}/{job['condition']}"]
            if json.loads(json.dumps(current)) != expected:
                raise ValueError('Effective provider command differs from pinned manifest')
        processes = process_snapshot()
        okay, reason = eligible(manifest_path, pid, processes)
        if not okay:
            raise ValueError(reason)
        quota_path = manifest_path.parent / 'runs/provider_quota.json'
        if not quota_clear(quota_path):
            raise ValueError('Existing provider quota state prevents handoff')
        operation = dict(at=now(), status='prepared', original_pid=pid,
            original_started=processes[pid]['started'], worker_pid=os.getpid(), worker_started=processes[os.getpid()]['started'],
            original_provider_pids=sorted(descendants(pid, processes)), external_holder=reason,
            manifest=str(manifest_path), protocol_sha256=manifest['protocol_sha256'],
            run_root=str(run_root(manifest_path, manifest)), quota_path=str(quota_path),
            pinned_source_hashes=pinned, adapter_path=str(HERE), adapter_sha256=central.sha(HERE),
            decision_record=str(DECISION), decision_sha256=central.sha(DECISION),
            workers=WORKERS, supplementary_stata_permits=PERMITS, jobs=[])
        save(path, operation)
        CURRENT_OPERATION = path
        with (operation_dir / 'watchdog.log').open('a') as log:
            watcher = subprocess.Popen([sys.executable, '-u', str(HERE), '--watchdog', str(path)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record(path, watchdog_pid=watcher.pid)
        paused = False
        try:
            # The watchdog already knows the target before this signal is sent.
            record(path, status='suspending')
            os.kill(pid, signal.SIGSTOP)
            paused = True
            for _ in range(40):
                processes = process_snapshot()
                if 'T' in processes.get(pid, {}).get('state', ''):
                    break
                time.sleep(0.05)  # Chosen process-state observation policy.
            else:
                raise RuntimeError('Collector did not enter the stopped state')
            okay, reason = eligible(manifest_path, pid, processes, operation['original_started'])
            if not okay:
                raise RuntimeError('Post-stop eligibility failed: ' + str(reason))
            if (reason['provider_pid'], reason['provider_started']) != (operation['external_holder']['provider_pid'], operation['external_holder']['provider_started']):
                raise RuntimeError('The external Stata permit holder changed during suspension')
            jobs = pending_jobs(manifest_path)
            operation = record(path, status='running', jobs=jobs, paused_at=now(), post_stop_holder=reason,
                reserved_active_jobs=[job for job, state, state_path in active_states(manifest_path)])
            central.stata_slot = sidecar_stata_slot
            quotas = central.QuotaCoordinator(quota_path)
            errors = []
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                pending, active = iter(jobs), {}
                while True:
                    while len(active) < WORKERS and not central.STOP.is_set():
                        job = next(pending, None)
                        if job is None:
                            break
                        active[pool.submit(central.run_job, job, manifest, Path(operation['run_root']), runtime, quotas)] = job
                    if not active:
                        break
                    finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                    for future in finished:
                        job = active.pop(future)
                        try:
                            result = future.result()
                            event = dict(at=now(), job=job, status=result['status'])
                        except Exception as error:
                            central.STOP.set()
                            errors.append(repr(error))
                            event = dict(at=now(), job=job, error=repr(error))
                        with (operation_dir / 'jobs.jsonl').open('a') as stream:
                            stream.write(json.dumps(event) + '\n')
            verify_sources(manifest)
            record(path, status='joined', errors=errors, joined_at=now())
        except BaseException as error:
            record(path, status='error', error=repr(error))
            raise
        finally:
            if paused:
                operation = read(path)
                try:
                    problem = resume_original(operation)
                except Exception as error:
                    problem = 'Resume inspection failed: ' + repr(error)
                record(path, status='blocked' if problem else 'resumed', resume_problem=problem, finished_at=now())
            else:
                record(path, status='not_suspended', finished_at=now())


def watchdog(path):
    path = Path(path)
    wake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while True:
            try:
                operation = read(path)
                if operation['status'] in {'resumed', 'not_suspended'}:
                    return
                processes = process_snapshot()
                worker = processes.get(operation['worker_pid'])
                if worker is None or 'Z' in worker['state'] or worker['started'] != operation['worker_started']:
                    problem = resume_original(operation, processes)
                    status = 'blocked' if problem else 'resumed'
                    record(path, status=status, resume_problem=problem, watchdog_recovery=True)
                    if not problem or problem in {'original collector exited', 'original PID was reused'}:
                        return
            except Exception as error:
                with path.with_name('watchdog_errors.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(at=now(), error=repr(error), action='hold and inspect again')) + '\n')
            time.sleep(2)  # Chosen watchdog observation policy.
    finally:
        wake.terminate()
        wake.wait()


def inspect():
    processes = process_snapshot()
    result = []
    for entry in read(META / 'collections.json'):
        manifest_path = Path(entry['manifest']).resolve()
        for pid, process in processes.items():
            if 'prospective_collect_unlimited_additions.py collect' in process['command'] and str(manifest_path) in process['command']:
                okay, reason = eligible(manifest_path, pid, processes)
                result.append(dict(manifest=str(manifest_path), pid=pid, eligible=okay,
                    reason=reason, untouched=len(pending_jobs(manifest_path))))
    return result


def launch_eligible():
    OPS.mkdir(parents=True, exist_ok=True)
    for pid, process in list(LAUNCHED_WORKERS.items()):
        if process.poll() is not None:
            del LAUNCHED_WORKERS[pid]
    for candidate in inspect():
        if not candidate['eligible'] or not candidate['untouched']:
            continue
        manifest = Path(candidate['manifest'])
        out = OPS / manifest.parent.parent.name / manifest.parent.name
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'operation.json').exists() and not retryable_operation(out / 'operation.json'):
            continue
        argv = [sys.executable, '-u', str(HERE), '--handoff', str(manifest), '--pid', str(candidate['pid'])]
        if manifest.parent.parent.name == 'sonnet':
            argv = ['/usr/bin/sandbox-exec', '-f', str(queue.SONNET_GUARD), *argv]
        with (out / 'worker.log').open('a') as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        LAUNCHED_WORKERS[process.pid] = process
        print(json.dumps(dict(at=now(), launched_pid=process.pid, **candidate)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=Path)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--watchdog', type=Path)
    parser.add_argument('--launch-eligible', action='store_true')
    parser.add_argument('--watch', action='store_true', help='Inspect and launch newly eligible batches every five seconds')
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.watchdog)
    elif args.handoff:
        if not args.pid:
            parser.error('--handoff requires --pid')
        handoff(args.handoff, args.pid)
    elif args.launch_eligible or args.watch:
        while True:
            if (META / 'finalizer/complete.json').exists():
                break
            launch_eligible()
            if not args.watch:
                break
            time.sleep(5)  # Chosen handoff eligibility observation policy.
    else:
        print(json.dumps(inspect(), indent=2))

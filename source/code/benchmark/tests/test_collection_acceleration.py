"""Independent offline checks for the supplementary collection handoff."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import threading
from types import SimpleNamespace

import pytest

from benchmark import prospective_collect as central

SOURCE = Path(__file__).resolve().parents[1] / 'workflows/production_repairs_2026_09_21/accelerate.py'
SPEC = importlib.util.spec_from_file_location('collection_acceleration_tested', SOURCE)
acceleration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceleration)


@pytest.fixture(autouse=True)
def isolated_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(acceleration, 'OPS', tmp_path / 'ops')
    monkeypatch.setattr(acceleration, 'META', tmp_path / 'metadata')
    acceleration.central.STOP.clear()
    yield
    acceleration.central.STOP.clear()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def manifest_file(tmp_path, jobs):
    path = tmp_path / 'sonnet/batch/manifest.json'
    write(path, dict(protocol_sha256='same-protocol', jobs=jobs,
        protocol=dict(source_hashes={}, commands={'sonnet/agent': [['provider'], None]})))
    return path


def operation(tmp_path, jobs=()):
    return dict(status='running', original_pid=100, original_started='original-start',
        worker_pid=999, worker_started='worker-start', jobs=list(jobs),
        run_root=str(tmp_path / 'runs'), protocol_sha256='same-protocol',
        quota_path=str(tmp_path / 'quota.json'))


def original_process():
    return {100: dict(ppid=1, started='original-start', state='T', command='collector')}


def job(rep=0, platform='stata', condition='agent'):
    return dict(block='test', task='toy', model='sonnet', platform=platform,
                prompt='zero_shot', condition=condition, output_format='answer', rep=rep)


def test_existing_job_lock_blocks_duplicate_before_provider_call(tmp_path):
    selected = job()
    folder = tmp_path / central.job_id(selected)
    folder.mkdir()
    manifest = dict(protocol_sha256='same-protocol')
    runtime = SimpleNamespace(c=SimpleNamespace())
    with (folder / '.lock').open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            central.run_job(selected, manifest, tmp_path, runtime, None,
                model_call=lambda *args: pytest.fail('Duplicate provider call'),
                execute_call=lambda *args: pytest.fail('Duplicate execution'))
    assert not (folder / 'attempt_0001').exists()


def test_pending_jobs_preserve_exact_jobs_and_exclude_attempts_and_locks(tmp_path):
    jobs = [job(0), job(1, 'r'), job(2, 'python'), job(3), job(4)]
    path = manifest_file(tmp_path, jobs)
    root = acceleration.run_root(path, acceleration.read(path))
    (root / central.job_id(jobs[3]) / 'attempt_0001').mkdir(parents=True)
    locked = root / central.job_id(jobs[4]) / '.lock'
    locked.parent.mkdir(parents=True)
    with locked.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert acceleration.pending_jobs(path) == [jobs[1], jobs[2], jobs[0]]


@pytest.mark.parametrize('text,clear', [
    ('{}', True), ('{"sonnet":{"reset_at":0}}', False),
    ('{"sonnet":{"reset_at":null}}', False), ('{', False),
    ('null', False), ('[]', False), ('false', False),
])
def test_only_valid_empty_quota_state_permits_resume(tmp_path, text, clear):
    path = tmp_path / 'quota.json'
    path.write_text(text)
    assert acceleration.quota_clear(path) is clear


def test_missing_quota_file_has_no_outstanding_rejection(tmp_path):
    assert acceleration.quota_clear(tmp_path / 'quota.json') is True


def test_eligibility_requires_eight_waiters_no_children_and_same_process(tmp_path, monkeypatch):
    path = manifest_file(tmp_path, [job(i) for i in range(8)])
    states = [(item, dict(status='in_progress', stage='provider_in_flight'), tmp_path / str(i))
              for i, item in enumerate(acceleration.read(path)['jobs'])]
    monkeypatch.setattr(acceleration, 'active_states', lambda _: states)
    monkeypatch.setattr(acceleration, 'external_stata_holder', lambda *args: {'collector_pid': 200})
    processes = original_process()
    processes[100]['command'] = 'prospective_collect_unlimited_additions.py collect --manifest ' + str(path)
    assert acceleration.eligible(path, 100, processes, 'original-start')[0]
    assert not acceleration.eligible(path, 100, processes, 'reused-start')[0]
    processes[101] = dict(ppid=100, state='S', started='child', command='provider')
    assert not acceleration.eligible(path, 100, processes)[0]
    processes.pop(101)
    states.pop()
    assert not acceleration.eligible(path, 100, processes)[0]


def test_resume_blocks_live_orphan_calls_and_unfinished_attempts(tmp_path, monkeypatch):
    selected = job()
    op = operation(tmp_path, [selected])
    folder = Path(op['run_root']) / central.job_id(selected) / 'attempt_0001'
    write(folder / 'state.json', dict(stage='provider_in_flight'))
    write(folder / 'provider_attempt1.active.json', dict(pid=600, cwd=str(tmp_path)))
    processes = original_process()
    processes[600] = dict(ppid=1, state='S', started='orphan', command='provider')
    monkeypatch.setattr(acceleration.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='p600\nn' + str(tmp_path) + '\n'))
    assert acceleration.resume_problem(op, processes) == 'supplemental calls remain alive'
    processes.pop(600)
    assert acceleration.resume_problem(op, processes).startswith('unresolved supplemental record')


def test_resume_never_signals_reused_original_pid(tmp_path, monkeypatch):
    op = operation(tmp_path)
    processes = original_process()
    processes[100]['started'] = 'new-process'
    monkeypatch.setattr(acceleration.os, 'kill', lambda *args: pytest.fail('Reused PID signaled'))
    assert acceleration.resume_original(op, processes) == 'original PID was reused'


def test_watchdog_waits_for_orphan_completion_before_resume(tmp_path, monkeypatch):
    op = operation(tmp_path, [job()])
    path = tmp_path / 'operation.json'
    write(path, op)
    folder = Path(op['run_root']) / central.job_id(job()) / 'attempt_0001'
    write(folder / 'state.json', dict(stage='provider_in_flight'))
    write(folder / 'provider_attempt1.active.json', dict(pid=600, cwd=str(tmp_path)))
    processes = original_process()
    processes[600] = dict(ppid=1, state='S', started='orphan', command='provider')
    slept = []
    signals = []
    monkeypatch.setattr(acceleration, 'process_snapshot', lambda: processes)
    monkeypatch.setattr(acceleration.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='p600\nn' + str(tmp_path) + '\n'))
    monkeypatch.setattr(acceleration.subprocess, 'Popen', lambda *args, **kwargs:
                        SimpleNamespace(terminate=lambda: None, wait=lambda: None))
    monkeypatch.setattr(acceleration.central, 'completed', lambda *args: {'status': 'completed'} if slept else None)
    def sleep(_):
        assert not signals
        assert acceleration.read(path)['status'] == 'blocked'
        processes.pop(600)
        slept.append(True)
    monkeypatch.setattr(acceleration.time, 'sleep', sleep)
    monkeypatch.setattr(acceleration.os, 'kill', lambda pid, sig: signals.append((pid, sig)))
    acceleration.watchdog(path)
    assert signals == [(100, signal.SIGCONT)]
    assert acceleration.read(path)['status'] == 'resumed'


def test_stale_permit_with_unresolved_attempt_is_not_reused(tmp_path, monkeypatch):
    op = operation(tmp_path, [job()])
    op_path = tmp_path / 'previous_operation.json'
    write(op_path, op)
    folder = Path(op['run_root']) / central.job_id(job()) / 'attempt_0001'
    write(folder / 'state.json', dict(stage='provider_in_flight'))
    for i in range(acceleration.PERMITS):
        write(acceleration.OPS / 'stata_permits' / f'permit_{i}.json', dict(active=True, operation=str(op_path)))
    monkeypatch.setattr(acceleration, 'process_snapshot', original_process)
    monkeypatch.setattr(acceleration, 'owned_live_calls', lambda *args: [])
    monkeypatch.setattr(acceleration.time, 'sleep', lambda _: acceleration.central.STOP.set())
    with pytest.raises(acceleration.central.Paused):
        with acceleration.sidecar_stata_slot(job()):
            pytest.fail('Unresolved stale native permit was reused')


@pytest.mark.parametrize('provider_state,cwd_matches,accepted', [
    ('S', True, True), ('Z', True, False), ('T', True, False), ('S', False, False),
])
def test_external_holder_requires_running_provider_with_matching_cwd(tmp_path, monkeypatch, provider_state, cwd_matches, accepted):
    state_path = tmp_path / 'attempt/state.json'
    write(state_path.parent / 'provider_attempt1.active.json', dict(pid=201, cwd=str(tmp_path)))
    write(acceleration.META / 'collections.json', [{'manifest': 'manifest.json'}])
    monkeypatch.setattr(acceleration, 'active_states', lambda _: [(job(), dict(stage='provider_in_flight'), state_path)])
    monkeypatch.setattr(acceleration, 'process_cwd', lambda _: str(tmp_path) if cwd_matches else '/unrelated')
    processes = {
        200: dict(ppid=1, state='S', started='other-start', command='prospective_collect_unlimited_additions.py collect'),
        201: dict(ppid=200, state=provider_state, started='provider-start', command='claude -p'),
    }
    assert bool(acceleration.external_stata_holder(100, processes)) is accepted


def handoff_fixture(tmp_path, monkeypatch, fail_after_stop=False):
    jobs = [job(0, 'python'), job(1, 'r'), job(2, 'stata')]
    path = manifest_file(tmp_path, jobs)
    write(path.parent / 'authorization.json', dict(allow_provider_calls=True, protocol_sha256='same-protocol'))
    decision = tmp_path / 'decision.md'
    decision.write_text('Authorized test scheduling policy')
    monkeypatch.setattr(acceleration, 'DECISION', decision)
    processes = original_process()
    processes[100]['state'] = 'S'
    processes[os.getpid()] = dict(ppid=1, state='S', started='worker-start', command='sidecar')
    events, submitted, source_checks = [], [], []
    monkeypatch.setattr(acceleration, 'process_snapshot', lambda: processes)
    def eligibility(*args):
        if fail_after_stop and processes[100]['state'] == 'T':
            return False, 'state changed'
        return True, dict(provider_pid=201, provider_started='holder-start')
    monkeypatch.setattr(acceleration, 'eligible', eligibility)
    monkeypatch.setattr(acceleration, 'pending_jobs', lambda _: jobs)
    monkeypatch.setattr(acceleration, 'active_states', lambda _: [])
    monkeypatch.setattr(acceleration, 'verify_sources', lambda manifest: source_checks.append(manifest) or {})
    runtime = SimpleNamespace(c=SimpleNamespace(
        command=lambda *args: (['provider'], None), environment=lambda: {'PATH': os.environ['PATH']}))
    monkeypatch.setattr(acceleration.central, 'load_runtime', lambda _: runtime)
    # Ensure the process-local adapter replacement is restored after this test.
    monkeypatch.setattr(acceleration.central, 'stata_slot', acceleration.central.stata_slot)
    monkeypatch.setattr(acceleration, 'owned_live_calls', lambda *args: [])
    def popen(*args, **kwargs):
        events.append('watchdog')
        return SimpleNamespace(pid=888)
    monkeypatch.setattr(acceleration.subprocess, 'Popen', popen)
    def kill(pid, sig):
        assert pid == 100
        if sig == signal.SIGSTOP:
            assert events == ['watchdog']
            processes[100]['state'] = 'T'
            events.append('stop')
        elif sig == signal.SIGCONT:
            assert fail_after_stop or len(submitted) == len(jobs)
            processes[100]['state'] = 'S'
            events.append('continue')
        else:
            pytest.fail('Unexpected signal')
    monkeypatch.setattr(acceleration.os, 'kill', kill)
    def run(selected, manifest, root, actual_runtime, quotas):
        assert processes[100]['state'] == 'T'
        assert actual_runtime is runtime
        assert selected in manifest['jobs']
        assert root == acceleration.run_root(path, manifest)
        assert quotas.path == path.parent / 'runs/provider_quota.json'
        record = dict(status='completed', protocol_sha256=manifest['protocol_sha256'], job=selected)
        write(root / central.job_id(selected) / 'attempt_0001/state.json', dict(stage='done'))
        write(root / central.job_id(selected) / 'attempt_0001/run_record.json', record)
        submitted.append(selected)
        return record
    monkeypatch.setattr(acceleration.central, 'run_job', run)
    return path, jobs, events, submitted, source_checks


def test_full_handoff_joins_exact_jobs_then_resumes_original(tmp_path, monkeypatch):
    path, jobs, events, submitted, source_checks = handoff_fixture(tmp_path, monkeypatch)
    before = path.read_bytes()
    acceleration.handoff(path, 100)
    assert events == ['watchdog', 'stop', 'continue']
    assert sorted(submitted, key=lambda item: item['rep']) == jobs
    assert len(source_checks) == 2
    assert path.read_bytes() == before
    receipt = acceleration.read(acceleration.OPS / 'sonnet/batch/operation.json')
    assert receipt['status'] == 'resumed'
    assert receipt['jobs'] == jobs
    assert receipt['supplementary_stata_permits'] == 3


def test_post_stop_setup_failure_resumes_without_any_new_job(tmp_path, monkeypatch):
    path, jobs, events, submitted, source_checks = handoff_fixture(tmp_path, monkeypatch, fail_after_stop=True)
    with pytest.raises(RuntimeError, match='Post-stop eligibility failed'):
        acceleration.handoff(path, 100)
    assert events == ['watchdog', 'stop', 'continue']
    assert not submitted
    receipt = acceleration.read(acceleration.OPS / 'sonnet/batch/operation.json')
    assert receipt['status'] == 'resumed'
    assert receipt['jobs'] == []


def test_three_shared_native_permits_bound_four_contenders(tmp_path, monkeypatch):
    path = tmp_path / 'operation.json'
    write(path, operation(tmp_path))
    monkeypatch.setattr(acceleration, 'CURRENT_OPERATION', path)
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    counts = {'active': 0, 'maximum': 0}
    def contender(rep):
        with acceleration.sidecar_stata_slot(job(rep)):
            with lock:
                counts['active'] += 1
                counts['maximum'] = max(counts['maximum'], counts['active'])
                if counts['active'] == 3:
                    entered.set()
            assert release.wait(5), 'Test release did not arrive'
            with lock:
                counts['active'] -= 1
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(contender, rep) for rep in range(4)]
        try:
            assert entered.wait(5), 'Three test permits were not acquired'
            assert counts['maximum'] == 3
        finally:
            release.set()
        for future in futures:
            future.result()
    assert counts == {'active': 0, 'maximum': 3}


def test_watchdog_recovers_zombie_worker(tmp_path, monkeypatch):
    path = tmp_path / 'operation.json'
    write(path, operation(tmp_path))
    processes = original_process()
    processes[999] = dict(ppid=1, started='worker-start', state='Z', command='sidecar')
    monkeypatch.setattr(acceleration, 'process_snapshot', lambda: processes)
    monkeypatch.setattr(acceleration, 'owned_live_calls', lambda *args: [])
    monkeypatch.setattr(acceleration.subprocess, 'Popen', lambda *args, **kwargs:
                        SimpleNamespace(terminate=lambda: None, wait=lambda: None))
    monkeypatch.setattr(acceleration.time, 'sleep', lambda _: pytest.fail('Zombie worker blocked recovery'))
    signals = []
    monkeypatch.setattr(acceleration.os, 'kill', lambda pid, sig: signals.append((pid, sig)))
    acceleration.watchdog(path)
    assert signals == [(100, signal.SIGCONT)]

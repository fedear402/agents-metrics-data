"""Check queue sequencing and worker choices without starting collectors."""
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import production_queue as queue


def test_least_filled_wave_spreads_repetitions_across_cells():
    base = dict(block='b', task='t', model='sonnet', platform='python', prompt='few_shot',
                condition='agent', output_format='json')
    jobs = [dict(base, task=task, rep=rep, global_rep=rep) for task in ('a', 'b') for rep in range(5)]
    chosen = queue.least_filled_wave(jobs, {})
    assert chosen == [jobs[0], jobs[5]]


def test_full_model_queue_keeps_workers_fed_and_orders_by_coverage():
    base = dict(block='b', model='luna', platform='python', prompt='few_shot',
                condition='agent', output_format='json')
    jobs = [dict(base, task=task, rep=rep, global_rep=rep) for task in ('a', 'b') for rep in range(3)]
    jobs.append(dict(jobs[0], model='sonnet'))
    counts = {queue.cell_key(jobs[0]): 1, queue.cell_key(jobs[3]): 0}
    ordered = queue.ordered_model_jobs(jobs, counts, 'luna')
    assert len(ordered) == 6
    assert len({queue.job_id(job) for job in ordered}) == 6
    assert ordered[0]['task'] == 'b'
    projected = dict(counts)
    for job in ordered:
        remaining = [other for other in ordered[ordered.index(job):]]
        assert projected[queue.cell_key(job)] == min(projected[queue.cell_key(other)] for other in remaining)
        projected[queue.cell_key(job)] += 1


def make_index(tmp_path, models=('luna',)):
    entries = []
    for name in ('first', 'second'):
        folder = tmp_path / name
        folder.mkdir()
        protocol = {'planned_jobs': [dict(block=name, task='test', model=model, platform='python',
                    prompt='few_shot', condition='agent', output_format='json', rep=1, global_rep=6)
                    for model in models]}
        digest = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        manifest = dict(protocol=protocol, protocol_sha256=digest, jobs=protocol['planned_jobs'])
        path = folder / 'manifest.json'
        path.write_text(json.dumps(manifest))
        (folder / 'authorization.json').write_text(json.dumps(dict(protocol_sha256=digest, allow_provider_calls=True)))
        entries.append(dict(batch=name, manifest=str(path), planned=len(models), protocol_sha256=digest))
    index = tmp_path / 'collections.json'
    index.write_text(json.dumps(entries))
    return index


def snapshot(path, states=None):
    states = states or {}
    jobs = queue.read(path)['jobs']
    rows = [dict(id=queue.job_id(job), status=states.get(queue.job_id(job), 'pending')) for job in jobs]
    return dict(planned=len(rows), completed=sum(row['status'] in queue.TERMINAL for row in rows), rows=rows)


def replace_jobs(index, batch, jobs):
    entries = queue.read(index)
    entry = next(row for row in entries if row['batch'] == batch)
    path = Path(entry['manifest'])
    protocol = dict(planned_jobs=jobs)
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    queue.save(path, dict(protocol=protocol, protocol_sha256=digest, jobs=jobs))
    queue.save(path.parent / 'authorization.json', dict(protocol_sha256=digest, allow_provider_calls=True))
    entry.update(protocol_sha256=digest, planned=len(jobs))
    queue.save(index, entries)
    return path


def test_completion_counts_deduplicate_history_and_include_failures(tmp_path):
    index = make_index(tmp_path)
    path = tmp_path / 'first/manifest.json'
    job = queue.read(path)['jobs'][0]
    queue.save(tmp_path / 'planned_positions.json', [dict(job, status='retained'),
               dict(job, global_rep=0, status='retained'), dict(job, global_rep=1, status='held')])
    result = snapshot(path, {queue.job_id(job): 'model_failure'})
    result['rows'][0]['primary'] = False
    assert queue.completion_counts(index, [(path, result)])[queue.cell_key(job)] == 2


def test_least_fill_uses_count_instead_of_repetition_label():
    base = dict(block='b', model='luna', platform='r', prompt='zero_shot',
                condition='chatbot', output_format='json')
    covered = dict(base, task='covered', rep=0, global_rep=0)
    empty = dict(base, task='empty', rep=4, global_rep=9)
    assert queue.least_filled_wave([covered, empty], {queue.cell_key(covered): 3}) == [empty]


def test_least_fill_orders_across_batches_and_models(tmp_path, monkeypatch):
    index = make_index(tmp_path, ('sonnet', 'luna'))
    first = tmp_path / 'first/manifest.json'
    first_jobs = queue.read(first)['jobs']
    second_jobs = [dict(job, task='empty') for job in queue.read(tmp_path / 'second/manifest.json')['jobs']]
    replace_jobs(index, 'second', second_jobs)
    queue.save(tmp_path / 'planned_positions.json', [dict(job, status='retained', global_rep=0)
               for job in first_jobs] + [dict(second_jobs[0], status='retained', global_rep=0)])
    states, invoked = {}, []
    runner = queue.Queue(index, tmp_path / 'old.json')
    monkeypatch.setattr(queue, 'collection_active', lambda path: False)
    monkeypatch.setattr(queue, 'status', lambda path: snapshot(path, states))
    def collect(path, workers, log, model, jobs_file):
        jobs = queue.read(jobs_file)
        invoked.append((path.parent.name, model))
        assert all(job in queue.read(path)['jobs'] for job in jobs)
        states.update({queue.job_id(job): 'completed' for job in jobs})
        return 0
    monkeypatch.setattr(runner, 'collect', collect)
    assert runner.run() == 0
    assert invoked[0] == ('second', 'luna')
    assert len(invoked) == 4


def test_locked_batch_does_not_block_available_cells(tmp_path, monkeypatch):
    index = make_index(tmp_path)
    first = tmp_path / 'first/manifest.json'
    second = tmp_path / 'second/manifest.json'
    completed, invoked = set(), []
    runner = queue.Queue(index, tmp_path / 'old.json')
    monkeypatch.setattr(queue, 'collection_active', lambda path: path == first and not completed)
    monkeypatch.setattr(queue, 'status', lambda path: snapshot(path, {
        queue.job_id(job): 'completed' for job in queue.read(path)['jobs'] if path in completed}))
    def collect(path, workers, log, model, jobs_file):
        invoked.append(path)
        completed.add(path)
        return 0
    monkeypatch.setattr(runner, 'collect', collect)
    assert runner.run() == 0
    assert invoked == [second, first]


def test_repeated_cell_jobs_run_in_waves_without_duplicates(tmp_path, monkeypatch):
    index = make_index(tmp_path)
    base = queue.read(tmp_path / 'first/manifest.json')['jobs'][0]
    jobs = [dict(base, task=task, rep=rep, global_rep=rep) for task in ('a', 'b') for rep in range(3)]
    path = replace_jobs(index, 'first', jobs)
    states, invoked = {}, []
    runner = queue.Queue(index, tmp_path / 'old.json', batches=['first'])
    monkeypatch.setattr(queue, 'collection_active', lambda path: False)
    monkeypatch.setattr(queue, 'status', lambda path: snapshot(path, states))
    def collect(path, workers, log, model, jobs_file):
        chosen = queue.read(jobs_file)
        invoked.append(chosen)
        states.update({queue.job_id(job): 'model_failure' for job in chosen})
        return 0
    monkeypatch.setattr(runner, 'collect', collect)
    assert runner.run() == 0
    assert [[job['global_rep'] for job in wave] for wave in invoked] == [[0, 0], [1, 1], [2, 2]]
    assert {queue.job_id(job) for wave in invoked for job in wave} == {queue.job_id(job) for job in jobs}
    assert queue.read(path)['jobs'] == jobs


def test_drain_model_collects_all_pending_once_and_defers_holds(tmp_path, monkeypatch):
    index = make_index(tmp_path)
    base = queue.read(tmp_path / 'first/manifest.json')['jobs'][0]
    jobs = [dict(base, model=model, rep=rep, global_rep=rep)
            for model in ('luna', 'sonnet') for rep in range(4)]
    path = replace_jobs(index, 'first', jobs)
    states = {queue.job_id(jobs[0]): 'completed', queue.job_id(jobs[4]): 'needs_review'}
    invoked = []
    runner = queue.Queue(index, tmp_path / 'old.json', batches=['first'], drain_model=True)
    monkeypatch.setattr(queue, 'collection_active', lambda path: False)
    monkeypatch.setattr(queue, 'status', lambda path: snapshot(path, states))
    def collect(path, workers, log, model, jobs_file):
        chosen = queue.read(jobs_file)
        assert len(chosen) == 3
        assert all(job['model'] == model for job in chosen)
        assert all(queue.job_id(job) not in states for job in chosen)
        invoked.extend(queue.job_id(job) for job in chosen)
        states.update({queue.job_id(job): 'completed' for job in chosen})
        return 0
    monkeypatch.setattr(runner, 'collect', collect)
    assert runner.run() == 0
    assert len(invoked) == len(set(invoked)) == 6
    assert queue.read(path)['jobs'] == jobs


def test_queue_uses_four_during_overlap_and_eight_afterward(tmp_path, monkeypatch):
    index = make_index(tmp_path)
    old = tmp_path / 'old/manifest.json'
    completed, invoked = set(), []
    monkeypatch.setattr(queue, 'status', lambda path: snapshot(path, {
        queue.job_id(job): 'completed' for job in queue.read(path)['jobs']
        if path.parent.name in completed}))
    monkeypatch.setattr(queue, 'collection_active', lambda path: path == old and not completed)
    runner = queue.Queue(index, old)
    def collect(path, workers, log, model, jobs_file):
        invoked.append((path.parent.name, workers))
        completed.add(path.parent.name)
        return 0
    monkeypatch.setattr(runner, 'collect', collect)
    assert runner.run() == 0
    assert invoked == [('first', 4), ('second', 8)]


def test_zero_progress_continues_other_batches_without_retry(tmp_path, monkeypatch):
    index = make_index(tmp_path)
    monkeypatch.setattr(queue, 'status', snapshot)
    monkeypatch.setattr(queue, 'collection_active', lambda path: False)
    monkeypatch.setattr(queue, 'verify_shared', lambda path: None)
    runner = queue.Queue(index, tmp_path / 'old/manifest.json')
    invoked = []
    def collect(path, workers, log, model, jobs_file):
        invoked.append(path.parent.name)
        return 2
    monkeypatch.setattr(runner, 'collect', collect)
    assert runner.run() == 0
    assert invoked == ['first', 'second']
    assert len(queue.read(runner.out / 'checkpoint.json')['deferred_models']) == 2


def test_collection_lock_reports_live_owner(tmp_path):
    path = tmp_path / 'collection.lock'
    path.touch()
    assert not queue.collection_active(tmp_path / 'manifest.json')
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert queue.collection_active(tmp_path / 'manifest.json')
        fcntl.flock(stream, fcntl.LOCK_UN)
    assert not queue.collection_active(tmp_path / 'manifest.json')


def test_changed_manifest_and_wrong_authorization_are_rejected(tmp_path):
    index = make_index(tmp_path)
    manifest_path = tmp_path / 'first/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['protocol']['modified'] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='hash mismatch'):
        queue.checked_index(index)


def test_worker_limit_remains_at_most_eight(tmp_path):
    with pytest.raises(ValueError, match='one through eight'):
        queue.Queue(tmp_path / 'index.json', tmp_path / 'old.json', solo_workers=9)


def test_active_batch_waits_and_rechecks_status_before_selection(tmp_path, monkeypatch):
    index = make_index(tmp_path)
    runner = queue.Queue(index, tmp_path / 'old.json', batches=['first'])
    observations = []
    active = iter([True, False])
    monkeypatch.setattr(queue, 'collection_active', lambda path: next(active) if path.parent.name == 'first' else False)
    monkeypatch.setattr(queue, 'status', lambda path: observations.append(path.parent.name) or dict(planned=1, completed=1))
    monkeypatch.setattr(queue, 'sleep', lambda seconds: observations.append('wait'), raising=False)
    assert runner.run() == 0
    assert observations == ['first', 'second', 'wait', 'first', 'second']


def test_existing_hold_deferred_and_exact_model_groups_collected(tmp_path, monkeypatch):
    index = make_index(tmp_path, ('held', 'sonnet', 'luna'))
    path = tmp_path / 'first/manifest.json'
    original = queue.read(path)['jobs']
    states = {queue.job_id(original[0]): 'needs_review'}
    invoked = []
    runner = queue.Queue(index, tmp_path / 'old.json', batches=['first'])
    monkeypatch.setattr(queue, 'collection_active', lambda path: False)
    monkeypatch.setattr(queue, 'status', lambda path: snapshot(path, states))
    def collect(path, workers, log, model, jobs_file):
        jobs = queue.read(jobs_file)
        assert jobs == [job for job in original if job['model'] == model]
        invoked.append(model)
        states.update({queue.job_id(job): 'completed' for job in jobs})
        return 0
    monkeypatch.setattr(runner, 'collect', collect)
    assert runner.run() == 0
    assert invoked == ['sonnet', 'luna']
    assert queue.read(runner.out / 'checkpoint.json')['deferred_review'] == {'first': 1}
    assert queue.read(path)['jobs'] == original


def test_new_hold_continues_next_model_without_retry(tmp_path, monkeypatch):
    index = make_index(tmp_path, ('sonnet', 'luna'))
    states, invoked = {}, []
    runner = queue.Queue(index, tmp_path / 'old.json')
    monkeypatch.setattr(queue, 'collection_active', lambda path: False)
    monkeypatch.setattr(queue, 'verify_shared', lambda path: None)
    monkeypatch.setattr(queue, 'status', lambda path: snapshot(path, states))
    def collect(path, workers, log, model, jobs_file):
        invoked.append(model)
        states.update({queue.job_id(job): 'needs_review' for job in queue.read(jobs_file)})
        return 0
    monkeypatch.setattr(runner, 'collect', collect)
    assert runner.run() == 0
    assert invoked == ['sonnet', 'luna', 'sonnet', 'luna']
    assert queue.read(runner.out / 'checkpoint.json')['deferred_review'] == {'first': 2, 'second': 2}


def test_shared_environment_failure_stops_all_batches(tmp_path, monkeypatch):
    index = make_index(tmp_path)
    runner = queue.Queue(index, tmp_path / 'old.json')
    invoked = []
    monkeypatch.setattr(queue, 'collection_active', lambda path: False)
    monkeypatch.setattr(queue, 'status', snapshot)
    def collect(path, *args):
        invoked.append(path.parent.name)
        return 1
    def verify(path):
        raise RuntimeError('Native environment changed')
    monkeypatch.setattr(runner, 'collect', collect)
    monkeypatch.setattr(queue, 'verify_shared', verify)
    with pytest.raises(RuntimeError, match='environment changed'):
        runner.run()
    assert invoked == ['first']
    assert queue.read(runner.out / 'checkpoint.json')['status'] == 'shared_verification_failed'


def test_unknown_quota_defers_provider_and_continues_other_provider(tmp_path, monkeypatch):
    index = make_index(tmp_path, ('sonnet', 'luna'))
    runner = queue.Queue(index, tmp_path / 'old.json')
    invoked, states = [], {}
    monkeypatch.setattr(queue, 'collection_active', lambda path: False)
    monkeypatch.setattr(queue, 'status', lambda path: snapshot(path, states))
    monkeypatch.setattr(queue, 'verify_shared', lambda path: None)
    def collect(path, workers, log, model, jobs_file):
        invoked.append((path.parent.name, model))
        if model == 'sonnet':
            queue.save(path.parent / 'runs/provider_quota.json', {'sonnet': dict(status='rejected', reset_at=None)})
            return 2
        states.update({queue.job_id(job): 'completed' for job in queue.read(jobs_file)})
        return 0
    monkeypatch.setattr(runner, 'collect', collect)
    assert runner.run() == 0
    assert invoked == [('first', 'sonnet'), ('first', 'luna'), ('second', 'luna')]
    assert queue.read(runner.out / 'checkpoint.json')['deferred_providers'] == ['sonnet']


def test_unresolved_request_state_is_deferred(tmp_path):
    index = make_index(tmp_path)
    path = tmp_path / 'first/manifest.json'
    manifest = queue.read(path)
    identity = queue.job_id(manifest['jobs'][0])
    state_path = path.parent / 'runs' / manifest['protocol_sha256'] / identity / 'attempt_0001/state.json'
    queue.save(state_path, dict(stage='provider_in_flight'))
    assert queue.candidates(path, snapshot(path)) == ([], [identity])


def test_model_commands_keep_luna_native_and_guard_sonnet(tmp_path, monkeypatch):
    index = make_index(tmp_path)
    guard = tmp_path / 'deny.sb'
    guard.write_text('(version 1)')
    monkeypatch.setattr(queue, 'SONNET_GUARD', guard)
    calls = []
    class Process:
        def __init__(self, argv, **kwargs):
            calls.append(argv)
        def wait(self):
            return 0
    monkeypatch.setattr(queue.subprocess, 'Popen', Process)
    runner = queue.Queue(index, tmp_path / 'old.json')
    manifest = tmp_path / 'first/manifest.json'
    jobs = tmp_path / 'jobs.json'
    for model in ('luna', 'sonnet'):
        assert runner.collect(manifest, 4, tmp_path / 'run.log', model, jobs) == 0
    assert calls[1][:3] == ['/usr/bin/sandbox-exec', '-f', str(guard)]
    assert calls[1][3:] == calls[0]
    assert calls[0][0] == sys.executable
    assert calls[0][-2:] == ['--jobs-file', str(jobs)]


def test_disjoint_queue_chains_allowed_overlaps_rejected(tmp_path):
    index = make_index(tmp_path)
    with queue.queue_lock(index, {'first'}):
        with queue.queue_lock(index, {'second'}):
            pass
        with pytest.raises(BlockingIOError):
            with queue.queue_lock(index, {'first'}):
                pass


def test_altered_jobs_outside_protocol_rejected(tmp_path):
    index = make_index(tmp_path)
    path = tmp_path / 'first/manifest.json'
    manifest = queue.read(path)
    manifest['jobs'][0]['global_rep'] = 10
    queue.save(path, manifest)
    with pytest.raises(ValueError, match='identity mismatch'):
        queue.checked_index(index)

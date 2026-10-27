"""Collect indexed production batches sequentially with the existing runner.

Sonnet subsets use the shared-installation write guard. Luna retains its native
sandbox. Holds remain deferred. Position failures allow other jobs to continue.
Shared source or environment failures stop the queue.
By default, each launch takes one repetition from each least-filled eligible cell.
With --drain-model, one launch includes all pending jobs for its model and batch,
ordered by projected cell coverage. Other groups are reassessed after it returns.
Counts include retained history and current completions,
regardless of numerical success. Active collectors keep their loaded order.
Disjoint --batches lists prioritize within their scope. Use one controller for
global priority. Active batch locks remain untouched while other batches proceed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
from time import sleep, time

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / 'data/production/paper22_2026-09-18'
COLLECTOR = ROOT / 'code/benchmark/prospective_collect.py'
OLD_MANIFEST = ROOT / 'quality_reports/reports/2026-09-17_worst10-topup-and-harder-pilot/work/topup10/manifest.json'
SONNET_GUARD = DESIGN / 'work/environment_recovery/shared_python_write_deny.sb'
TERMINAL = {'completed', 'model_failure', 'provider_timeout', 'execution_timeout'}
CELL_FIELDS = ('task', 'model', 'platform', 'prompt', 'condition')


def cell_key(job):
    return tuple(job[key] for key in CELL_FIELDS)


def completion_counts(index_path, snapshots):
    """Count unique completed positions, including history outside new manifests."""
    positions = set()
    ledger = Path(index_path).parent / 'planned_positions.json'
    if ledger.exists():
        for row in read(ledger):
            if row['status'] == 'retained':
                positions.add((*cell_key(row), row['global_rep']))
    for manifest_path, snapshot in snapshots:
        manifest = read(manifest_path)
        terminal_ids = {row['id'] for row in snapshot['rows'] if row['status'] in TERMINAL}
        for job in manifest['jobs']:
            if job_id(job) in terminal_ids:
                positions.add((*cell_key(job), job.get('global_rep', job['rep'])))
    return Counter(position[:-1] for position in positions)


def least_filled_wave(jobs, counts):
    """Select one original job per cell at the lowest available completion count."""
    if not jobs:
        return []
    minimum = min(counts.get(cell_key(job), 0) for job in jobs)
    seen, selected = set(), []
    for job in sorted(jobs, key=lambda item: item.get('global_rep', item['rep'])):
        cell = cell_key(job)
        if counts.get(cell, 0) == minimum and cell not in seen:
            seen.add(cell)
            selected.append(job)
    return selected


def job_id(job):
    return '__'.join(str(job[k]) for k in ('block','task','model','platform','prompt','condition','output_format')) + f'__rep{job["rep"]:02d}'


def ordered_model_jobs(jobs, counts, model):
    """Order pending work by projected cell coverage without a wave barrier."""
    projected = dict(counts)
    ranked = []
    selected = sorted((job for job in jobs if job['model'] == model),
                      key=lambda job: job.get('global_rep', job['rep']))
    for sequence, job in enumerate(selected):
        cell = cell_key(job)
        rank = projected.get(cell, 0)
        ranked.append((rank, sequence, job))
        projected[cell] = rank + 1
    return [job for _, _, job in sorted(ranked, key=lambda item: item[:2])]


def candidates(manifest_path, snapshot):
    """Keep exact original jobs and defer unresolved provider work."""
    manifest = read(manifest_path)
    rows = {row['id']: row for row in snapshot['rows']}
    pending, deferred = [], []
    for job in manifest['jobs']:
        identity = job_id(job)
        row = rows[identity]
        if row['status'] in TERMINAL:
            continue
        state_path = Path(manifest_path).parent / 'runs' / manifest['protocol_sha256'] / identity / 'attempt_0001/state.json'
        state = read(state_path) if state_path.exists() else {}
        if row['status'] in {'hold','needs_review'} or state.get('stage') in {'hold','provider_in_flight'}:
            deferred.append(identity)
        else:
            pending.append(job)
    return pending, deferred


def read(path):
    return json.loads(Path(path).read_text())


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def collection_active(manifest):
    """Inspect the same process lock used by the central collector."""
    path = Path(manifest).parent / 'collection.lock'
    if not path.exists():
        return False
    with path.open('r') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(stream, fcntl.LOCK_UN)
            return False


def status(manifest):
    result = subprocess.run([sys.executable, str(COLLECTOR), 'status', '--manifest', str(manifest)],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def verify_shared(manifest):
    result = subprocess.run([sys.executable, str(COLLECTOR), 'check', '--manifest', str(manifest)],
                            cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Shared verification failed: ' + result.stderr[-4000:])


def quota_blocked(manifest, model):
    path = Path(manifest).parent / 'runs/provider_quota.json'
    if not path.exists():
        return False
    state = read(path).get(model, {})
    reset = state.get('reset_at')
    return state.get('status') == 'rejected' and (
        not isinstance(reset, (int, float)) or isinstance(reset, bool) or reset > time())


def checked_index(index_path):
    entries = read(index_path)
    batches, paths = set(), set()
    for entry in entries:
        path = Path(entry['manifest']).resolve()
        if entry['batch'] in batches or path in paths:
            raise ValueError('Duplicate batch or manifest in the collection index')
        batches.add(entry['batch'])
        paths.add(path)
        manifest = read(path)
        digest = hashlib.sha256(json.dumps(manifest['protocol'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if manifest['protocol_sha256'] != digest or entry['protocol_sha256'] != digest:
            raise ValueError(f'Manifest hash mismatch: {path}')
        if entry['planned'] != len(manifest['jobs']):
            raise ValueError(f'Planned count mismatch: {path}')
        if manifest['jobs'] != manifest['protocol'].get('planned_jobs'):
            raise ValueError(f'Planned job identity mismatch: {path}')
        authorization = read(path.parent / 'authorization.json')
        if authorization.get('protocol_sha256') != digest or authorization.get('allow_provider_calls') is not True:
            raise ValueError(f'Provider authorization does not match: {path}')
    return entries


@contextmanager
def queue_lock(index_path, batches):
    folder = Path(index_path).parent / 'work/queue'
    folder.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for batch in sorted(batches):
            name = hashlib.sha256(batch.encode()).hexdigest()[:16]
            stream = stack.enter_context((folder / f'batch_{name}.lock').open('a'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Queue:
    def __init__(self, index, old_manifest, batches=None, overlap_workers=4, solo_workers=8, drain_model=False):
        if not 1 <= overlap_workers <= 8 or not 1 <= solo_workers <= 8:
            raise ValueError('Worker limits must be integers from one through eight')
        self.index = Path(index).resolve()
        self.old_manifest = Path(old_manifest).resolve()
        self.batches = set(batches) if batches else None
        self.overlap_workers = overlap_workers
        self.solo_workers = solo_workers
        self.drain_model = drain_model
        self.out = self.index.parent / 'work/queue'
        if self.batches:
            name = hashlib.sha256('\n'.join(sorted(self.batches)).encode()).hexdigest()[:12]
            self.out /= name
        self.child = None
        self.stopping = False

    def stop(self, signum, frame):
        self.stopping = True
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signum)

    def collect(self, manifest, workers, log, model, jobs_file):
        argv = [sys.executable, '-u', str(COLLECTOR), 'collect', '--manifest', str(manifest),
                '--authorization', str(manifest.parent / 'authorization.json'), '--workers', str(workers),
                '--jobs-file', str(jobs_file)]
        if model == 'sonnet':
            if not SONNET_GUARD.is_file():
                raise ValueError('Sonnet shared-installation guard is missing')
            argv = ['/usr/bin/sandbox-exec', '-f', str(SONNET_GUARD), *argv]
        elif model != 'luna':
            raise ValueError('Unsupported queue model')
        with log.open('a') as stream:
            self.child = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            try:
                return self.child.wait()
            finally:
                self.child = None

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        handled = set()
        initial = checked_index(self.index)
        names = {entry['batch'] for entry in initial}
        selected = self.batches if self.batches is not None else names
        if not selected <= names:
            raise ValueError('A requested batch is absent from the index')
        deferred_counts = {}
        skipped_models, deferred_providers = set(), set()
        with queue_lock(self.index, selected):
            while not self.stopping:
                entries = checked_index(self.index)
                names = {entry['batch'] for entry in entries}
                if self.batches is not None and not self.batches <= names:
                    raise ValueError('A requested batch is absent from the index')
                pending = [entry for entry in entries if entry['batch'] not in handled
                           and entry['batch'] in selected]
                if not pending:
                    save(self.out / 'checkpoint.json', dict(status='queue_pass_complete', updated_at=timestamp(),
                         handled_batches=sorted(handled), deferred_review=deferred_counts,
                         deferred_models=[list(pair) for pair in sorted(skipped_models)],
                         deferred_providers=sorted(deferred_providers)))
                    return 0
                options, snapshots, waiting = [], [], []
                for candidate in entries:
                    path = Path(candidate['manifest']).resolve()
                    active = collection_active(path)
                    snapshot = status(path)
                    if snapshot['planned'] != candidate['planned']:
                        raise ValueError('Central status and index disagree')
                    snapshots.append((path, snapshot))
                    if active:
                        if candidate in pending:
                            waiting.append(candidate['batch'])
                        continue
                    if candidate not in pending:
                        continue
                    if snapshot['completed'] == snapshot['planned']:
                        handled.add(candidate['batch'])
                        continue
                    available, held = candidates(path, snapshot)
                    deferred_counts[candidate['batch']] = len(held)
                    available = [job for job in available
                                 if (candidate['batch'], job['model']) not in skipped_models
                                 and job['model'] not in deferred_providers]
                    if available:
                        options.append((candidate, path, snapshot, available, held))
                    else:
                        handled.add(candidate['batch'])
                if not options:
                    if waiting:
                        save(self.out / 'checkpoint.json', dict(status='waiting_for_collection_lock',
                             batches=waiting, updated_at=timestamp()))
                        sleep(2)  # Chosen local polling interval.
                    continue
                counts = completion_counts(self.index, snapshots)
                entry, manifest, before, eligible, deferred = min(
                    options, key=lambda option: min(counts.get(cell_key(job), 0) for job in option[3]))
                wave = least_filled_wave(eligible, counts)
                model = wave[0]['model']
                jobs = [job for job in wave if job['model'] == model]
                if self.drain_model:
                    jobs = ordered_model_jobs(eligible, counts, model)
                if self.stopping:
                    return 130
                jobs_file = self.out / f'{entry["batch"]}_{model}_jobs.json'
                save(jobs_file, jobs)
                overlap = collection_active(self.old_manifest)
                workers = self.overlap_workers if overlap else self.solo_workers
                receipt = dict(batch=entry['batch'], manifest=str(manifest), protocol_sha256=entry['protocol_sha256'],
                               started_at=timestamp(), workers=workers, overlapping_old_collection=overlap,
                               status='collecting', complete_before=before['completed'], planned=before['planned'])
                receipt.update(model=model, selected=len(jobs), deferred_review=len(deferred), jobs_file=str(jobs_file),
                               scheduling='projected_cell_coverage_continuous_model' if self.drain_model else 'least_completed_cells',
                               cell_completed_before=counts.get(cell_key(jobs[0]), 0))
                save(self.out / 'checkpoint.json', receipt)
                print(json.dumps(receipt), flush=True)
                code = self.collect(manifest, workers, self.out / f'{entry["batch"]}_{model}.log', model, jobs_file)
                if self.stopping:
                    return 130
                after = status(manifest)
                remaining, deferred_after = candidates(manifest, after)
                deferred_counts[entry['batch']] = len(deferred_after)
                selected_ids = {job_id(job) for job in jobs}
                finished_ids = {row['id'] for row in after['rows'] if row['status'] in TERMINAL}
                new_holds = set(deferred_after) - set(deferred)
                succeeded = code == 0 and selected_ids <= finished_ids and not new_holds
                progressed = bool(selected_ids & finished_ids) or bool(new_holds)
                if not succeeded:
                    try:
                        verify_shared(manifest)
                    except Exception as error:
                        save(self.out / 'checkpoint.json', dict(receipt, status='shared_verification_failed', error=str(error)))
                        raise
                    if quota_blocked(manifest, model):
                        deferred_providers.add(model)
                    if not progressed:
                        skipped_models.add((entry['batch'], model))
                receipt.update(finished_at=timestamp(), returncode=code, complete_after=after['completed'],
                               new_holds=sorted(new_holds), deferred_review=len(deferred_after),
                               status='completed' if succeeded else 'deferred_quota' if model in deferred_providers
                               else 'deferred_no_progress' if not progressed else 'continuing_after_position_failure')
                with (self.out / 'receipts.jsonl').open('a') as stream:
                    stream.write(json.dumps(receipt) + '\n')
                save(self.out / 'checkpoint.json', receipt)
                print(json.dumps(receipt), flush=True)
                if not remaining:
                    handled.add(entry['batch'])
        return 130


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=DESIGN / 'collections.json')
    parser.add_argument('--old-manifest', type=Path, default=OLD_MANIFEST)
    parser.add_argument('--batches', nargs='+')
    parser.add_argument('--overlap-workers', type=int, default=4)
    parser.add_argument('--solo-workers', type=int, default=8)
    parser.add_argument('--drain-model', action='store_true',
                        help='Keep workers supplied with all pending jobs for the selected model and batch.')
    args = parser.parse_args()
    queue = Queue(args.index, args.old_manifest, args.batches, args.overlap_workers, args.solo_workers, args.drain_model)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, queue.stop)
    raise SystemExit(queue.run())


if __name__ == '__main__':
    main()

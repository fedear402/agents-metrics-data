"""Keep scheduler inspection outside the inherited Sonnet execution guard.

The original adapter stays unchanged. Every Sonnet provider and submitted-script
subprocess receives the same shared-installation guard as the original queue.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve()
SOURCE = HERE.with_name('accelerate.py')
BASE_SHA256 = '470bb4c5fde076eb337c29b334f195d5296c8470e5a946447fe19f5adf3786c6'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != BASE_SHA256:
    raise RuntimeError('The original acceleration adapter changed')
spec = importlib.util.spec_from_file_location('production_acceleration_base', SOURCE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
BASE_LOAD_RUNTIME = base.central.load_runtime
GUARD = base.queue.SONNET_GUARD
GUARD_SHA256 = '178956ffc6d07ea80b5d2d20436703c529d3580531bdd6e5b4348c47ff8aff14'


def guard_runtime(runtime, guard=GUARD):
    """Wrap the shared provider/execution boundary, including its descendants."""
    guard = Path(guard).resolve()
    guard_hash = base.central.sha(guard)
    capture = runtime.c.prior.subprocess_capture

    def guarded_capture(argv, cwd, destination, stem, timeout, text=None):
        if base.central.sha(guard) != guard_hash:
            raise RuntimeError('Sonnet execution guard changed')
        wrapped = ['/usr/bin/sandbox-exec', '-f', str(guard), *argv]
        base.save(Path(destination) / f'{stem}.guard.json', dict(
            guard=str(guard), guard_sha256=guard_hash, argv=wrapped,
            adapter=str(HERE), adapter_sha256=base.central.sha(HERE)))
        return capture(wrapped, cwd, destination, stem, timeout, text)

    runtime.c.prior.subprocess_capture = guarded_capture
    return runtime


def load_runtime(work):
    if base.central.sha(GUARD) != GUARD_SHA256:
        raise RuntimeError('The original Sonnet execution guard changed')
    runtime = BASE_LOAD_RUNTIME(work)
    models = {job['model'] for job in base.read(Path(work) / 'manifest.json')['jobs']}
    if 'sonnet' in models:
        if models != {'sonnet'}:
            raise ValueError('The guarded handoff requires a single-model manifest')
        guard_runtime(runtime)
    return runtime


def launch_eligible():
    """Start only the scheduler outside the model subprocess guard."""
    base.OPS.mkdir(parents=True, exist_ok=True)
    for pid, process in list(base.LAUNCHED_WORKERS.items()):
        if process.poll() is not None:
            del base.LAUNCHED_WORKERS[pid]
    for candidate in base.inspect():
        if not candidate['eligible'] or not candidate['untouched']:
            continue
        manifest = Path(candidate['manifest'])
        out = base.OPS / manifest.parent.parent.name / manifest.parent.name
        out.mkdir(parents=True, exist_ok=True)
        operation = out / 'operation.json'
        if operation.exists() and not base.retryable_operation(operation):
            continue
        argv = [sys.executable, '-u', str(HERE), '--handoff', str(manifest), '--pid', str(candidate['pid'])]
        with (out / 'worker.log').open('a') as log:
            process = subprocess.Popen(argv, cwd=base.ROOT, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        base.LAUNCHED_WORKERS[process.pid] = process
        print(base.json.dumps(dict(at=base.now(), launched_pid=process.pid, **candidate)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=Path)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--watchdog', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    base.HERE = HERE
    base.central.load_runtime = load_runtime
    if args.watchdog:
        base.watchdog(args.watchdog)
    elif args.handoff:
        if not args.pid:
            parser.error('--handoff requires --pid')
        base.handoff(args.handoff, args.pid)
    elif args.watch:
        while not (base.META / 'finalizer/complete.json').exists():
            launch_eligible()
            time.sleep(5)
    else:
        print(base.json.dumps(base.inspect(), indent=2))


if __name__ == '__main__':
    main()

"""Run the staged full-design collection of one model (Haiku 5.5 or GPT-6 Sol) without supervision.

Decision record: quality_reports/decision_records/2026-10-17_haiku55-sol-full-design.md.
Stages: repetition 0, then 1-2, 3-4, and 5-9 (manifests data/runs/full_2026-10-17/<model>/{0_4,5_9}). For each
stage the driver runs the model check (preflight), then the collector's `collect` on the pending jobs of the stage
(Haiku inside the paper's write-deny profile, Sol in Codex's own sandbox, as in the paper), and then the stage
check (validate.py). When the collector stops:
- usage limit (the manifest's runs/provider_quota.json exists): wait until the reset time in the provider's
  message plus RESET_MARGIN, or FALLBACK_WAIT when the message gives no time; move the quota file to
  data/collections/2026-10-17_full-design/quota_cleared/; continue;
- hold (state needs_review): leave the job out, as the paper queue did; stop the model when holds exceed
  MAX_HOLD_SHARE of the stage;
- another stop without progress: wait and retry; stop the model after MAX_FAILURES such stops in a row.
Every event goes to data/collections/2026-10-17_full-design/<model>_driver.jsonl. The driver resumes where the
records end, so it can be started again after any stop.

Runaway scripts (decision record 2026-10-18_runaway-script-rule.md): a watcher thread stops any script that a
collector of this model started (a direct child running /private/tmp/prospective_task_*/run.do, run.R, or run.py)
after STOP_AFTER, with SIGTERM to its process group and SIGKILL after KILL_GRACE, and writes a receipt to
script_stops.jsonl. Before each collector run, recover.py turns stopped runs into repair requests or
execution_timeout results and returns runs held by the collector stop. The driver waits while another collector
holds the manifest's collection.lock.

Run from the repository root (Haiku needs HAIKU_API_KEY_FILE):
    caffeinate -i uv run --no-sync python code/benchmark/workflows/full_design_2026_10_17/drive.py --model sol
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import fcntl
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

from dateutil import parser as dateparser

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import prospective_collect as c  # noqa: E402  (helpers only: job_id, completed, TERMINAL)

HERE = Path(__file__).resolve().parent
RAW = ROOT / 'data/runs/full_2026-10-17'
META = ROOT / 'data/collections/2026-10-17_full-design'
GUARD = ROOT / 'data/production/paper22_2026-09-18/work/environment_recovery/shared_python_write_deny.sb'
STAGES = [(0, 0), (1, 2), (3, 4), (5, 9)]
COLLECTORS = {'haiku55': 'prospective_collect_haiku_api.py', 'sol': 'prospective_collect_sol.py',
              'luna6': 'prospective_collect_luna6.py', 'opus55': 'prospective_collect_opus_api.py',
              'sonnet55': 'prospective_collect_sonnet55_api.py'}
CODEX_MODELS = {'sol': 'gpt-6-sol', 'luna6': 'gpt-6-luna'}
# Chosen operating settings, not measured provider values.
WORKERS = {'haiku55': {0: 4, 'later': 8}, 'sol': {0: 8, 'later': 8}, 'luna6': {0: 8, 'later': 8}, 'opus55': {0: 8, 'later': 8},
           'sonnet55': {0: 8, 'later': 8}}
RESET_MARGIN = timedelta(minutes=5)
FALLBACK_WAIT = timedelta(minutes=15)
MAX_WAIT = timedelta(hours=24)
MAX_FAILURES = 3
MAX_HOLD_SHARE = 0.02
STOP_AFTER = 900      # seconds a collector-run script may run (decision record 2026-10-18_runaway-script-rule)
KILL_GRACE = 30       # seconds between SIGTERM and SIGKILL
WATCH_EVERY = 30      # seconds between process checks
SCRIPT = re.compile(r'(/private/tmp/prospective_task_[^/\s]+)/run\.(?:do|R|py)\b')
_spec = importlib.util.spec_from_file_location('full_design_recover', HERE / 'recover.py')
recover = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recover)


def etime_seconds(text):
    """Seconds from a ps etime field ([[dd-]hh:]mm:ss)."""
    days, _, rest = text.strip().rpartition('-')
    parts = [int(x) for x in rest.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def processes():
    out = subprocess.run(['ps', '-eo', 'pid=,ppid=,etime=,command='], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4:
            rows.append(dict(pid=int(fields[0]), ppid=int(fields[1]), elapsed=etime_seconds(fields[2]),
                             command=fields[3]))
    return rows


def runaway_scripts(rows, collector_name, limit=STOP_AFTER):
    """Direct children of this model's collectors that run a task script and passed the limit."""
    collectors = {r['pid'] for r in rows if collector_name in r['command'] and ' collect ' in r['command']}
    return [dict(r, cwd=SCRIPT.search(r['command'])[1]) for r in rows
            if r['ppid'] in collectors and SCRIPT.search(r['command']) and r['elapsed'] > limit]


def watch(model, stopping):
    handled = set()
    while not stopping.wait(WATCH_EVERY):
        try:
            for row in runaway_scripts(processes(), COLLECTORS[model]):
                if row['pid'] in handled:
                    continue
                handled.add(row['pid'])
                try:
                    os.killpg(row['pid'], signal.SIGTERM)
                except ProcessLookupError:
                    continue
                receipt = dict(event='script_stopped', at=now().isoformat(), model=model, pid=row['pid'],
                               collector_pid=row['ppid'], cwd=row['cwd'], command=row['command'][:300],
                               elapsed_seconds=row['elapsed'], signal='SIGTERM',
                               by='runaway rule (decision record 2026-10-18_runaway-script-rule)')
                with (META / 'script_stops.jsonl').open('a') as stream:
                    stream.write(json.dumps(receipt) + '\n')
                log(model, 'script_stopped', pid=row['pid'], cwd=row['cwd'], elapsed_seconds=row['elapsed'])
                time.sleep(KILL_GRACE)
                try:
                    os.killpg(row['pid'], signal.SIGKILL)
                    log(model, 'script_killed', pid=row['pid'])
                except ProcessLookupError:
                    pass
        except Exception as error:  # noqa: BLE001 - the watcher must keep running
            log(model, 'watch_error', error=repr(error)[:300])


def lock_free(manifest_file):
    path = Path(manifest_file).parent / 'collection.lock'
    if not path.exists():
        return True
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
    return True


def now():
    return datetime.now(timezone.utc)


def log(model, event, **fields):
    META.mkdir(parents=True, exist_ok=True)
    row = dict(at=now().isoformat(), model=model, event=event, **fields)
    with (META / f'{model}_driver.jsonl').open('a') as stream:
        stream.write(json.dumps(row, default=str) + '\n')
    print(json.dumps(row, default=str), flush=True)


def manifest_path(model, rep):
    return RAW / model / ('0_4' if rep <= 4 else '5_9') / 'manifest.json'


def states(manifest, jobs):
    run_root = manifest_path_root(manifest)
    out = {}
    for job in jobs:
        folder = run_root / c.job_id(job)
        if c.completed(folder, manifest['protocol_sha256']) is not None:
            out[c.job_id(job)] = 'done'
            continue
        state = folder / 'attempt_0001/state.json'
        if not state.is_file():
            out[c.job_id(job)] = 'pending'
            continue
        data = json.loads(state.read_text())
        # A run paused by an interrupted script that recover.py did not resolve must not replay its script.
        out[c.job_id(job)] = 'held_paused' if data.get('status') == 'paused' and data.get('hold_reason') \
            else data.get('status', 'pending')
    return out


def manifest_path_root(manifest):
    return Path(manifest['_path']).parent / 'runs' / manifest['protocol_sha256']


def reset_from_text(text, at):
    """Reset time from a usage-limit message ('try again at 1:33 PM' or 'try again in 2 hours 5 minutes')."""
    found = re.search(r'try again in ((?:\s*\d+\s*(?:days?|hours?|hrs?|minutes?|mins?|seconds?|secs?)\s*,?\s*(?:and)?)+)',
                      text, re.I)
    if found:
        delta = timedelta()
        for number, unit in re.findall(r'(\d+)\s*([a-z]+)', found[1], re.I):
            key = unit.lower()[0]
            delta += {'d': timedelta(days=int(number)), 'h': timedelta(hours=int(number)),
                      'm': timedelta(minutes=int(number)), 's': timedelta(seconds=int(number))}[key]
        return at + delta
    found = re.search(r'try again at ([^.\n"]+)', text, re.I)
    if found:
        try:
            moment = dateparser.parse(found[1], default=at.astimezone().replace(second=0, microsecond=0),
                                      fuzzy=True)
        except (ValueError, OverflowError):
            return None
        moment = moment if moment.tzinfo else moment.astimezone()
        while moment <= at:
            moment += timedelta(days=1)
        return moment
    return None


def limit_message(manifest, jobs, statuses):
    run_root = manifest_path_root(manifest)
    texts = []
    for job in jobs:
        if statuses.get(c.job_id(job)) != 'quota_wait':
            continue
        for path in (run_root / c.job_id(job)).glob('attempt_*/call_attempt*.json'):
            texts.append((path.stat().st_mtime, json.loads(path.read_text()).get('error_message') or ''))
    return max(texts)[1] if texts else ''


def wait_for_reset(model, manifest, jobs, statuses):
    quota = Path(manifest['_path']).parent / 'runs' / 'provider_quota.json'
    message = limit_message(manifest, jobs, statuses)
    reset = reset_from_text(message, now())
    until = reset + RESET_MARGIN if reset else now() + FALLBACK_WAIT
    until = min(until, now() + MAX_WAIT)
    log(model, 'usage_limit', message=message[:300], reset=reset, wait_until=until)
    while now() < until:
        time.sleep(min(300, max(1, (until - now()).total_seconds())))
    if quota.exists():
        target = META / 'quota_cleared' / f'{model}_{now().strftime("%Y%m%dT%H%M%SZ")}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(quota), target)
        log(model, 'quota_file_cleared', moved_to=target)


def run(argv, model, name):
    with (META / f'{model}_{name}.log').open('a') as stream:
        return subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode


def stage(model, first, last):
    path = manifest_path(model, first)
    manifest = json.loads(path.read_text())
    manifest['_path'] = str(path)
    jobs = [j for j in manifest['jobs'] if first <= j['global_rep'] <= last]
    batch = manifest['jobs'][0]['block']
    check = META / f'{batch}_validation_{first}_{last}.json'
    if check.is_file() and json.loads(check.read_text()).get('passed'):
        log(model, 'stage_already_checked', reps=[first, last])
        return True
    preflight = [sys.executable, str(HERE / 'preflight_sol.py'), '--manifest', str(path), '--model-id', CODEX_MODELS[model]] \
        if model in CODEX_MODELS else \
        [sys.executable, str(ROOT / 'code/benchmark/workflows/haiku55_api_2026_10_16/preflight.py'), '--manifest', str(path)]
    code = run(preflight, model, 'preflight')
    log(model, 'preflight', reps=[first, last], returncode=code)
    if code:
        return False
    workers = WORKERS[model][0 if (first, last) == (0, 0) else 'later']
    failures, last_done = 0, None
    while True:
        if not lock_free(path):
            log(model, 'waiting_for_running_collector', reps=[first, last])
            while not lock_free(path):
                time.sleep(60)
        actions = recover.resolve(model, path, apply=True)
        if actions:
            log(model, 'recovered', reps=[first, last], actions=actions)
        statuses = states(manifest, jobs)
        held = [k for k, v in statuses.items() if v in ('needs_review', 'held_paused')]
        pending = [j for j in jobs if statuses[c.job_id(j)] not in ('done', 'needs_review', 'held_paused')]
        done = sum(v == 'done' for v in statuses.values())
        if len(held) > MAX_HOLD_SHARE * len(jobs):
            log(model, 'too_many_holds', reps=[first, last], holds=len(held), held=held[:50])
            return False
        if not pending:
            break
        quota = path.parent / 'runs' / 'provider_quota.json'
        if quota.exists():
            wait_for_reset(model, manifest, jobs, statuses)
            continue
        if last_done is not None and done == last_done:
            failures += 1
            if failures >= MAX_FAILURES:
                log(model, 'stopped_without_progress', reps=[first, last], done=done, pending=len(pending))
                return False
            time.sleep(60 * failures)
        else:
            failures = 0
        last_done = done
        jobs_file = META / f'{batch}_jobs_{first}_{last}.json'
        jobs_file.write_text(json.dumps(pending))
        argv = [sys.executable, '-u', str(ROOT / 'code/benchmark' / COLLECTORS[model]), 'collect', '--manifest',
                str(path), '--authorization', str(path.parent / 'authorization.json'), '--workers', str(workers),
                '--jobs-file', str(jobs_file)]
        if model in ('haiku55', 'opus55', 'sonnet55'):
            argv = ['/usr/bin/sandbox-exec', '-f', str(GUARD), *argv]
        log(model, 'collect_start', reps=[first, last], pending=len(pending), done=done, holds=len(held),
            workers=workers)
        code = run(argv, model, 'collector')
        log(model, 'collect_end', reps=[first, last], returncode=code)
    code = run([sys.executable, str(HERE / 'validate.py'), '--manifest', str(path), '--reps', str(first), str(last)],
               model, 'validation')
    log(model, 'stage_check', reps=[first, last], returncode=code, report=check)
    return code == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=sorted(COLLECTORS), required=True)
    parser.add_argument('--parallel-halves', action='store_true',
                        help='run repetitions 0-4 and 5-9 at the same time, one collector each (8 workers each)')
    args = parser.parse_args()
    log(args.model, 'driver_start', stop_after_seconds=STOP_AFTER)
    threading.Thread(target=watch, args=(args.model, threading.Event()), daemon=True).start()
    if args.parallel_halves:
        results = {}
        def half(first, last):
            results[(first, last)] = stage(args.model, first, last)
        threads = [threading.Thread(target=half, args=r) for r in ((0, 4), (5, 9))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if not all(results.values()):
            log(args.model, 'driver_stop', results={str(k): v for k, v in results.items()})
            raise SystemExit(1)
        log(args.model, 'driver_done')
        return
    for first, last in STAGES:
        if not stage(args.model, first, last):
            log(args.model, 'driver_stop', reps=[first, last])
            raise SystemExit(1)
    log(args.model, 'driver_done')


if __name__ == '__main__':
    main()

"""Switch the Opus 5.5 collection to the spare Anthropic API key when the current key has no credit left.

Federico, 2026-10-20: "when no more credits in the one being used now use that spare one".
The launcher (code/benchmark/opus_api_claude.sh) reads HAIKU_API_KEY_FILE at every call, so new calls use the key
that the file holds. Every 60 seconds this script sends a one-token Haiku 5.5 request with the current key. When the
API answers "credit balance is too low":
- the current key file is kept as <file>.exhausted_<time>, and the spare key replaces it (mode 600);
- the spare key gets the same probe, and the swap is logged.
After the swap, runs that the collector held because the API refused the request before any response
(hold_reason infrastructure_without_response, with the credit message) return to the request stage, as a usage-limit
rejection before a response does in the collector (stage request, status quota_wait). The rejected call stays in the
state with quota_rejected_before_response=True, so pricing leaves it out. A copy of each earlier state goes to
data/collections/2026-10-17_full-design/recovery/.
The Claude Code CLI reports a refused request as a response ("Credit balance is too low", 0 tokens), so the
collector can record the run as model_failure. A finished run whose every call shows the credit message and no output
tokens is moved to recovery/<job>__credit_refused_completed/ and repeats (13 runs on 2026-10-20 at 05:41).
If the spare key also has no credit, the script stops the Opus driver and its collectors (SIGTERM, the collector's
pause), moves the refused runs, and ends.
If the Opus driver has stopped, the script starts it again with the same command, and rebuilds the six-model
outputs when that driver ends. Events go to data/collections/2026-10-17_full-design/key_swap.jsonl.

Run from the repository root, detached:
    HAIKU_API_KEY_FILE=~/.config/agents-metrics/anthropic_api_key \
    nohup .venv/bin/python -u code/benchmark/workflows/full_design_2026_10_17/key_swap.py \
        --spare ~/.config/agents-metrics/anthropic_api_key_spare &
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import shutil
import signal
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
META = ROOT / 'data/collections/2026-10-17_full-design'
RAW = ROOT / 'data/runs/full_2026-10-17/opus55'
LOG = META / 'key_swap.jsonl'
CREDIT = 'credit balance is too low'
_spec = importlib.util.spec_from_file_location('full_design_recover_swap', HERE / 'recover.py')
recover = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recover)


def now():
    return datetime.now(timezone.utc).isoformat()


def log(event, **fields):
    with LOG.open('a') as stream:
        stream.write(json.dumps(dict(at=now(), event=event, **fields)) + '\n')


def probe(key_file):
    """'ok', 'no_credit', or 'error: ...' for one one-token request."""
    key = Path(key_file).read_text().strip()
    body = json.dumps(dict(model='claude-haiku-5-5', max_tokens=1,
                           messages=[dict(role='user', content='OK')])).encode()
    request = urllib.request.Request('https://api.anthropic.com/v1/messages', data=body, headers={
        'x-api-key': key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return 'ok' if response.status == 200 else f'error: HTTP {response.status}'
    except urllib.error.HTTPError as error:
        text = error.read().decode(errors='replace')
        return 'no_credit' if CREDIT in text.lower() else f'error: HTTP {error.code} {text[:200]}'
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return f'error: {error!r}'[:300]


def swap(key_file, spare):
    key_file, spare = Path(key_file), Path(spare)
    kept = key_file.with_name(key_file.name + '.exhausted_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    os.link(key_file, kept)
    temporary = key_file.with_name(key_file.name + '.new')
    temporary.write_text(spare.read_text())
    os.chmod(temporary, 0o600)
    os.replace(temporary, key_file)
    return kept


def credit_text(call):
    return CREDIT in ' '.join(str(call.get(k, '')) for k in ('text', 'error_message', 'stderr_tail')).lower()


def restore_credit_holds():
    central = recover.load_collector('opus55')
    actions, completed = [], []
    for manifest_path in (RAW / '0_4/manifest.json', RAW / '5_9/manifest.json'):
        manifest = json.loads(manifest_path.read_text())
        root = manifest_path.parent / 'runs' / manifest['protocol_sha256']
        for job in manifest['jobs']:
            folder = root / central.job_id(job)
            destination = folder / 'attempt_0001'
            path = destination / 'state.json'
            if not path.is_file():
                continue
            record = central.completed(folder, manifest['protocol_sha256'])
            if record is not None:
                if any(credit_text(call) for call in json.loads(path.read_text()).get('calls', [])):
                    completed.append(central.job_id(job))
                continue
            state = json.loads(path.read_text())
            if not (state.get('status') == 'needs_review' and state.get('hold_reason') == 'infrastructure_without_response'
                    and state.get('calls') and credit_text(state['calls'][-1])):
                continue
            with (folder / '.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                state = json.loads(path.read_text())
                spec = manifest['protocol']['tasks'][job['task']]
                if state['job'] != job or state['protocol_sha256'] != manifest['protocol_sha256']:
                    raise ValueError(f'Identity differs: {path}')
                if not central.inputs_intact(Path(state['cwd']), central.task_input_files(spec)):
                    raise ValueError(f'Inputs changed: {path}')
                kept = recover.backup(destination, central.job_id(job), 'credit_refused')
                state['calls'][-1]['quota_rejected_before_response'] = True
                state.pop('hold_reason', None)
                state.update(stage='request', status='in_progress',
                             operational_resume=dict(at=now(), kept=str(kept), provider_calls_added=0,
                                                     local_executions_added=0,
                                                     reason='API refused the request for lack of credit before any '
                                                            'response; the request repeats with the spare key.'))
                central.snapshot_state(destination, state)
                actions.append(central.job_id(job))
    return actions, completed


def move_credit_completed():
    moved = []
    for path in sorted(RAW.glob('*/runs/*/*/attempt_0001/state.json')):
        state = json.loads(path.read_text())
        calls = state.get('calls', [])
        if not calls or not all(credit_text(call) for call in calls) or state.get('status') != 'model_failure' \
                or any((call.get('usage') or {}).get('output_tokens') for call in calls):
            continue
        attempt = path.parent
        folder = attempt.parent
        with (folder / '.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            target = META / 'recovery' / f'{folder.name}__credit_refused_completed'
            target.mkdir(parents=True)
            shutil.move(str(attempt), str(target / 'attempt_0001'))
        moved.append(folder.name)
    with (META / 'recovery_actions.jsonl').open('a') as stream:
        for job in moved:
            stream.write(json.dumps(dict(job=job, kind='credit_refused_completed', model='opus55', applied=True,
                                         action='attempt moved to recovery; run repeats', at=now())) + '\n')
    return moved


def stop_opus():
    for pattern in ('drive.py --model opus55', 'prospective_collect_opus_api.py collect'):
        found = subprocess.run(['pgrep', '-f', pattern], capture_output=True, text=True).stdout.split()
        for pid in found:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass


def driver_alive():
    return subprocess.run(['pgrep', '-f', 'drive.py --model opus55'], capture_output=True).returncode == 0


def last_driver_event():
    lines = (META / 'opus55_driver.jsonl').read_text().splitlines()
    return json.loads(lines[-1])['event'] if lines else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spare', type=Path, required=True)
    parser.add_argument('--swapped', action='store_true', help='the key file already holds the spare key')
    args = parser.parse_args()
    key_file = Path(os.environ['HAIKU_API_KEY_FILE'])
    swapped, relaunched = args.swapped, False
    log('watch_start', key_file=str(key_file), spare=str(args.spare))
    while True:
        result = probe(key_file)
        if result == 'no_credit' and not swapped:
            kept = swap(key_file, args.spare)
            swapped = True
            log('key_swapped', kept=str(kept), spare_probe=probe(key_file))
        elif result == 'no_credit':
            stop_opus()
            time.sleep(300)
            log('spare_without_credit', opus_stopped=True, moved=move_credit_completed())
            return
        elif result != 'ok':
            log('probe_error', result=result)
        if swapped and result == 'ok':
            actions, completed = restore_credit_holds()
            moved = move_credit_completed()
            if actions or moved:
                log('credit_holds', restored=actions, moved=moved)
            if not driver_alive() and last_driver_event() == 'driver_stop':
                with (META / 'opus55_driver.out').open('a') as out:
                    subprocess.Popen(['caffeinate', '-i', str(ROOT / '.venv/bin/python'), '-u',
                                      str(HERE / 'drive.py'), '--model', 'opus55', '--parallel-halves'],
                                     cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                relaunched = True
                log('driver_relaunched')
                time.sleep(120)
        if not driver_alive() and last_driver_event() == 'driver_done':
            if relaunched:
                build = ROOT / 'data/analysis/six_models/after_runs.log'
                with build.open('a') as out:
                    subprocess.run([str(ROOT / '.venv/bin/python'), str(HERE / 'paper_rows.py'), '--model', 'opus55'],
                                   cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
                    subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'code/paper/six_models.py')],
                                   cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
                log('six_models_rebuilt')
            log('watch_end', swapped=swapped, relaunched=relaunched)
            return
        time.sleep(60)


if __name__ == '__main__':
    main()

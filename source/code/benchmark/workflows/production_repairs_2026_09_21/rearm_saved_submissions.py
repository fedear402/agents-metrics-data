"""Restore saved submissions held before execution by the scoped collector stop.

This command changes only eligible nonterminal states. The normal queue performs
the local evaluation after its active pool drains. No provider or script runs here.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import prospective_collect_unlimited_additions as adapter
import accelerate

central = adapter.central
MANIFEST = ROOT / 'data/runs/production_repairs_2026-09-21/sonnet/0_4/manifest.json'
OUT = ROOT / 'data/collections/2026-09-21_production-repairs/work/production_collection/stop_fallout_recovery'
ERROR = "RuntimeError('Collection stop requested before subprocess start')"
DECISION = ROOT / 'quality_reports/decision_records/2026-09-21_ppml-r-nonterminating-execution.md'


def rearm():
    manifest = central.json.loads(MANIFEST.read_text())
    central.verify(manifest)
    runtime = central.load_runtime(MANIFEST.parent)
    root = MANIFEST.parent / 'runs' / manifest['protocol_sha256']
    changed = []
    for job in manifest['jobs']:
        folder = root / central.job_id(job)
        destination = folder / 'attempt_0001'
        path = destination / 'state.json'
        if not path.exists():
            continue
        state = central.json.loads(path.read_text())
        if state.get('status') != 'needs_review' or state.get('stage') != 'hold' or state.get('driver_error') != ERROR:
            continue
        with (folder / '.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            state = central.json.loads(path.read_text())
            assert state['driver_error'] == ERROR and state['stage'] == 'hold'
            assert state['job'] == job and state['protocol_sha256'] == manifest['protocol_sha256']
            assert state['calls'] and state['calls'][-1]['ok'] and state['calls'][-1].get('text', '').strip()
            assert len(state['calls']) <= 4 and state['round'] <= 4
            assert central.completed(folder, manifest['protocol_sha256']) is None
            if accelerate.owned_live_calls(dict(jobs=[job], run_root=str(root))):
                continue
            cwd = Path(state['cwd'])
            spec = manifest['protocol']['tasks'][job['task']]
            assert central.inputs_intact(cwd, central.task_input_files(spec))
            script = cwd / runtime.c.harness.script_name(job['platform'])
            assert script.is_file()
            if job['condition'] != 'agent':
                assert script.read_text() == runtime.c.harness.extract_code(state['calls'][-1]['text'])
            submitted = Path(state['submitted_workspace']) / script.name
            assert submitted.is_file() and central.sha(submitted) == central.sha(script)
            call_path = destination / f"call_attempt{len(state['calls'])}.json"
            assert central.json.loads(call_path.read_text()) == state['calls'][-1]
            number = state['execution_attempts_started']
            assert not (destination / f'execution_attempt{number}.active.json').exists()
            assert not (destination / f'execution_attempt{number}.json').exists()
            backup = OUT / central.job_id(job)
            assert not backup.exists(), 'Preserve an existing recovery for review'
            backup.mkdir(parents=True)
            for name in ('state.json', 'run_record.json'):
                shutil.copy2(destination / name, backup / ('held_' + name))
            evidence = dict(at=central.now(), job=job, original_state_sha256=central.sha(path),
                original_record_sha256=central.sha(destination / 'run_record.json'),
                script_sha256=central.sha(script), input_sha256={name:central.sha(source) for name,source in central.task_input_files(spec).items()},
                saved_call_sha256=central.sha(call_path), protocol_sha256=manifest['protocol_sha256'],
                decision_record=str(DECISION), decision_sha256=central.sha(DECISION),
                provider_calls_added=0, local_executions_added=0,
                reason='Saved provider response held because the collector stop preceded local subprocess creation.')
            state.pop('driver_error')
            for field in ('hold_reason', 'pause_reason'):
                state.pop(field, None)
            state.update(stage='evaluate', status='in_progress', operational_resume=evidence)
            central.snapshot_state(destination, state)
            central.save(backup / 'receipt.json', dict(evidence, resumed_state_sha256=central.sha(path)))
            changed.append(central.job_id(job))
    print(json.dumps(dict(at=central.now(), rearmed=changed, provider_calls=0, local_executions=0)), flush=True)
    return changed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch-original', type=int, help='Check while the original collector drains, then check once more')
    args = parser.parse_args()
    while True:
        rearm()
        if not args.watch_original or args.watch_original not in accelerate.process_snapshot():
            break
        time.sleep(5)  # Chosen local recovery observation policy.

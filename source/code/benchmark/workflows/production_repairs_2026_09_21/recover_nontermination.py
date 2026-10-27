"""Finalize one documented nonterminating fourth execution without replay."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import prospective_collect_unlimited_additions as unlimited

central = unlimited.central
META = ROOT / 'data/collections/2026-09-21_production-repairs'
EVIDENCE = META / 'work/production_collection'
DECISION = ROOT / 'quality_reports/decision_records/2026-09-21_ppml-r-nonterminating-execution.md'
SCRIPT_SHA256 = '3628e4a530a6817d5f87bd768f9405353ff7024d1f633bc17037913d16075706'
JOB_ID = 'production_repairs_sonnet_0__lp04_ppml__sonnet__r__zero_shot__self_repair__json__rep00'


def read(path):
    return json.loads(Path(path).read_text())


def validate_paused(state, execution, cwd):
    assert central.job_id(state['job']) == JOB_ID
    assert state['round'] == 4 and len(state['calls']) == 4
    assert state['status'] == 'paused' and state['stage'] == 'evaluate'
    assert state['hold_reason'] == 'local_execution_interrupted'
    assert state['execution_attempts_started'] == len(state['executions']) == 4
    assert state['executions'][-1] == execution
    assert execution['returncode'] == -15 and execution['timed_out'] is False
    assert execution['ok'] is False and execution['timeout_seconds'] is None
    assert central.sha(cwd / 'run.R') == SCRIPT_SHA256
    assert not (cwd / 'result.json').exists()


def score_saved(state, spec, cwd, runtime):
    c = runtime.c
    estimates, rule, notes = c.load_output(cwd, spec, state['job'], state['executions'][-1])
    numeric, terms, numeric_notes = central.enforce_integer_terms(
        estimates, spec, c.compare_numeric(estimates, spec))
    integrity, integrity_notes = c.sp.score_integrity(c.base.as_task(spec), cwd, None,
        {spec['task_id']: spec['reference']})
    integrity = bool(integrity and central.inputs_intact(cwd, central.task_input_files(spec)))
    assert not estimates and numeric is False and integrity is True
    return dict(success=False, numeric_success=False, integrity_pass=True,
        estimates=estimates, per_term=terms, integrity_scope='input_data_only; script_review_pending',
        notes=notes + numeric_notes + integrity_notes + ['Confirmed generated-code nontermination; the fourth script produced no requested output.'],
        primary_source='Preserved fourth execution and documented nontermination adjudication')


def locate():
    diagnosis = read(EVIDENCE / 'r_18087_diagnosis.json')
    destination = ROOT / diagnosis['attempt']
    manifest_path = destination.parents[3] / 'manifest.json'
    return destination, manifest_path


def self_test():
    destination, manifest_path = locate()
    manifest = read(manifest_path)
    state = read(destination / 'state.json')
    execution = read(destination / 'execution_attempt4.json')
    runtime = central.load_runtime(manifest_path.parent)
    validate_paused(state, execution, Path(state['cwd']))
    with tempfile.TemporaryDirectory(prefix='nontermination_score_', dir='/private/tmp') as folder:
        cwd = Path(folder)
        spec = manifest['protocol']['tasks'][state['job']['task']]
        central.copy_inputs(cwd, central.task_input_files(spec))
        before = copy.deepcopy(state)
        score = score_saved(state, spec, cwd, runtime)
        assert state == before
        assert score['success'] is False and score['numeric_success'] is False
        assert state['executions'][-1]['returncode'] == -15
        assert state['executions'][-1]['timed_out'] is False
        assert not list(cwd.glob('result.json'))
    altered = copy.deepcopy(state)
    altered['round'] = 3
    try:
        validate_paused(altered, execution, Path(state['cwd']))
    except AssertionError:
        pass
    else:
        raise AssertionError('An unexhausted repair budget must not be finalized')
    print('Passed: exhausted-repair guard, absent-output scoring, unchanged state, and preserved negative exit receipt.')


def finalize():
    destination, manifest_path = locate()
    manifest = read(manifest_path)
    central.verify(manifest)
    runtime = central.load_runtime(manifest_path.parent)
    folder = destination.parent
    recovery = EVIDENCE / 'nontermination_recovery'
    recovery.mkdir(exist_ok=True)
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(destination / 'state.json')
        paused_record = read(destination / 'run_record.json')
        execution = read(destination / 'execution_attempt4.json')
        cwd = Path(state['cwd'])
        validate_paused(state, execution, cwd)
        assert state['protocol_sha256'] == manifest['protocol_sha256']
        assert state['job'] in manifest['jobs']
        signal_receipt = read(EVIDENCE / 'r_18087_recovery.json')
        assert signal_receipt['status'] == 'signal_sent' and signal_receipt['signal'] == 'SIGTERM'
        assert signal_receipt['script_sha256'] == SCRIPT_SHA256
        spec = manifest['protocol']['tasks'][state['job']['task']]
        original_calls = copy.deepcopy(state['calls'])
        original_executions = copy.deepcopy(state['executions'])
        protected = [p for p in destination.iterdir() if p.is_file() and
            (p.name.startswith(('provider_', 'execution_', 'call_', 'command_', 'prompt_')))]
        protected += [cwd / 'run.R', cwd / 'input.csv', manifest_path]
        hashes = {str(p): central.sha(p) for p in protected}
        for name in ('state.json', 'run_record.json'):
            target = recovery / ('paused_' + name)
            assert not target.exists(), 'Recovery already has preserved source records'
            shutil.copy2(destination / name, target)
        adjudication = dict(kind='confirmed_generated_code_nontermination', at=central.now(),
            decision_record=str(DECISION), decision_sha256=central.sha(DECISION),
            evidence={str(p): central.sha(p) for p in [EVIDENCE / 'r_18087_diagnosis.json',
                EVIDENCE / 'r_18087_loop_probe.json', EVIDENCE / 'r_18087_recovery.json']},
            actual_returncode=-15, timed_out=False, provider_calls_added=0, local_executions_added=0,
            original_state=str(recovery / 'paused_state.json'), original_record=str(recovery / 'paused_run_record.json'))
        state['scoring'] = score_saved(state, spec, cwd, runtime)
        for field in ('provider_ok', 'all_provider_attempts_ok', 'task_censored', 'protocol_violation', 'access_review_status'):
            state[field] = paused_record[field]
        state.update(stage='done', status='completed', completed_at=central.now(), execution_adjudication=adjudication)
        for field in ('hold_reason', 'pause_reason'):
            state.pop(field, None)
        runtime.c.prior.snapshot(cwd, destination / 'final_workspace')
        state['primary_output_path'] = str(destination / 'final_workspace/result.json')
        adapt = getattr(runtime, 'BASE_ADAPT', runtime.m.adapt_record)
        record = adapt(state, destination)
        record.pop('inherited_runtime_record', None)
        record['scoring'] = state['scoring']
        record['status'] = 'completed'
        record['central_runtime'] = paused_record['central_runtime']
        record['outcome'].update(primary_success=False, numerical_success=False,
            data_script_integrity=None, input_data_integrity=True,
            protocol_status='pending_post_collection', retry_eligible=False,
            measurement_status='confirmed_generated_code_nontermination')
        assert record['calls'] == original_calls and record['executions'] == original_executions
        assert all(central.sha(p) == digest for p, digest in hashes.items())
        # Write the terminal record first. Dispatch checks this file before it can replay a job.
        central.save(destination / 'run_record.json', record)
        central.snapshot_state(destination, state)
        receipt = dict(at=central.now(), job_id=JOB_ID, status='completed', primary_success=False,
            provider_calls_added=0, local_executions_added=0, raw_receipts_unchanged=True,
            protected_sha256=hashes, record_sha256=central.sha(destination / 'run_record.json'),
            state_sha256=central.sha(destination / 'state.json'), adjudication=adjudication)
        central.save(recovery / 'finalization_receipt.json', receipt)
        assert central.completed(folder, manifest['protocol_sha256'])['status'] == 'completed'
        print(json.dumps({key:value for key,value in receipt.items() if key not in {'protected_sha256','adjudication'}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else finalize()

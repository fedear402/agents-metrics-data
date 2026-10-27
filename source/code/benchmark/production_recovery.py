"""Recover reviewed local evaluations without repeating provider calls.

Run `python code/benchmark/production_recovery.py recover` from the project root.
Original attempts remain immutable. Hash-bound derived records form an overlay.
Use `first-request` to queue the reviewed zero-call slot behind its collection
lock. A durable dispatch marker prevents a second request after interruption.
"""
from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import tempfile
import time

try:
    import prospective_collect as p
except ModuleNotFoundError:
    from benchmark import prospective_collect as p

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'data/production/paper22_2026-09-18/work/environment_recovery'
MANIFEST = ROOT / 'data/runs/paper22_2026-09-18/original8_reps6to10/manifest.json'


def read(path):
    return json.loads(Path(path).read_text())


def reject_provider_call(*args, **kwargs):
    raise RuntimeError('Provider calls are forbidden during local recovery')


def review_script(text, platform):
    """Accept only the operations observed in the four reviewed submissions."""
    if platform == 'stata':
        allowed = ('set linesize ', 'clear', 'import delimited ', 'stset ', 'stcox ', 'local ', 'display ')
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            if not any(line == prefix or line.startswith(prefix) for prefix in allowed):
                raise ValueError('Unreviewed Stata operation: ' + line)
            if line.startswith('import delimited ') and not re.match(r'import delimited "(?:\./)?input\.csv",', line):
                raise ValueError('Stata input must be the supplied CSV')
        return
    if platform != 'python':
        raise ValueError('Only reviewed Python and Stata submissions are supported')
    tree = ast.parse(text)
    calls = {'main', 'len', 'float', 'range', 'ValueError', 'open', 'json.dump',
             'np.array_equal', 'np.arange', 'np.column_stack', 'np.ones', 'np.linalg.lstsq',
             'np.array', 'np.sqrt', 'np.sum', 'np.square', 'np.mean', 'pd.read_csv'}
    methods = {'sort_values', 'reset_index', 'to_numpy', 'append'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name not in {'json', 'numpy', 'pandas'} for alias in node.names):
            raise ValueError('Unreviewed Python import')
        if isinstance(node, ast.ImportFrom):
            raise ValueError('Unreviewed Python import')
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name not in calls and not (isinstance(node.func, ast.Attribute) and node.func.attr in methods):
            raise ValueError('Unreviewed Python operation: ' + name)
        if name == 'open':
            if len(node.args) < 2 or not all(isinstance(x, ast.Constant) for x in node.args[:2]):
                raise ValueError('Unreviewed file output')
            if (node.args[0].value, node.args[1].value) != ('result.json', 'w'):
                raise ValueError('Only local result.json output is permitted')
        if name == 'pd.read_csv' and (not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value != 'input.csv'):
            raise ValueError('Python input must be the supplied CSV')


def source_hashes(folder):
    result = {}
    for path in sorted(Path(folder).rglob('*')):
        if path.is_symlink():
            raise ValueError('Original attempt contains a symlink')
        if path.is_file():
            result[str(path)] = p.sha(path)
    return result


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if not Path(path).is_file() or p.sha(path) != expected:
            raise ValueError('Source changed: ' + path)


def make_entry(record_path, position_id, kind, manifest):
    record_path = Path(record_path)
    record = read(record_path)
    if len(record.get('calls', [])) != 1 or not record['calls'][0].get('ok') or record['calls'][0].get('timed_out'):
        raise ValueError('Recovery requires one completed provider response')
    state = read(record_path.parent / 'state.json')
    job = state['job']
    if job['condition'] != 'agent' or job not in manifest['jobs']:
        raise ValueError('Recovery requires an exact planned agent job')
    if kind == 'local_recovery':
        if state['stage'] != 'hold' or state.get('executions'):
            raise ValueError('Expected a pre-execution hold')
        if state.get('driver_error') != "RuntimeError('Collection stop requested before subprocess start')":
            raise ValueError('Unreviewed hold reason')
        snapshot = Path(state['submitted_workspace'])
    elif kind == 'environment_replay':
        if state['stage'] != 'done':
            raise ValueError('Expected the reviewed completed evaluation')
        snapshot = record_path.parent / 'final_workspace'
    else:
        raise ValueError('Unknown recovery kind')
    script = snapshot / ('run.do' if job['platform'] == 'stata' else 'run.py')
    review_script(script.read_text(), job['platform'])
    spec = manifest['protocol']['tasks'][job['task']]
    if p.sha(snapshot / 'input.csv') != p.sha(spec['data_path']):
        raise ValueError('Submission input differs from the pinned input')
    return dict(position_id=position_id, kind=kind, job=job, original_record=str(record_path),
                original_record_sha256=p.sha(record_path), original_state=str(record_path.parent/'state.json'),
                original_state_sha256=p.sha(record_path.parent/'state.json'), script=str(script),
                input_path=str(snapshot/'input.csv'), source_hashes=source_hashes(record_path.parent))


def recover_entry(entry, manifest, work, runtime, execute=None):
    verify_hashes(entry['source_hashes'])
    work = Path(work)
    target = work / 'records' / entry['position_id']
    receipt_path = target / 'receipt.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        verify_hashes({receipt['recovered_record']: receipt['recovered_record_sha256']})
        if receipt['original_record_sha256'] != entry['original_record_sha256']:
            raise ValueError('Existing recovery belongs to another original record')
        return receipt
    if target.exists():
        raise ValueError('Preserve incomplete recovery for review: ' + str(target))
    original = read(entry['original_record'])
    original_state = read(entry['original_state'])
    state = copy.deepcopy(original_state)
    job = entry['job']
    script = Path(entry['script'])
    review_script(script.read_text(), job['platform'])
    run_root = target / 'runs'
    destination = run_root / p.job_id(job) / 'attempt_0001'
    destination.mkdir(parents=True)
    cwd = Path(tempfile.mkdtemp(prefix='prospective_task_recovery_', dir='/private/tmp'))
    # Only the reviewed script and unchanged input enter the fresh workspace.
    (cwd/script.name).write_bytes(script.read_bytes())
    (cwd/'input.csv').write_bytes(Path(entry['input_path']).read_bytes())
    for key in ('driver_error', 'hold_reason', 'pause_reason', 'held_workspace', 'submitted_workspace',
                'primary_output_path', 'completed_at'):
        state.pop(key, None)
    state.update(stage='evaluate', status='in_progress', cwd=str(cwd), attempt_directory=str(destination),
                 recovery_origin=entry['original_record'])
    p.save(destination/'state.json', state)
    if p.STOP.is_set():
        raise ValueError('Recovery process has a stop request')
    result = p.run_job(job, manifest, run_root, runtime, None,
                       model_call=reject_provider_call, execute_call=execute)
    if result.get('stage') != 'done' or result['status'] not in p.TERMINAL:
        raise ValueError('Local recovery did not finish; preserve derived state for review')
    if result['calls'] != original['calls']:
        raise ValueError('Provider calls changed during local recovery')
    if p.sha(cwd/script.name) != p.sha(script) or p.sha(cwd/'input.csv') != p.sha(entry['input_path']):
        raise ValueError('Recovery changed its script or input')
    verify_hashes(entry['source_hashes'])
    parity = None
    if entry['kind'] == 'environment_replay':
        old_output = Path(original['primary_output_path'])
        new_output = Path(result['primary_output_path'])
        old, new = read(old_output), read(new_output)
        old_rows = {row['term']: row for row in old['estimates']}
        new_rows = {row['term']: row for row in new['estimates']}
        parity = dict(original_output=str(old_output), original_output_sha256=p.sha(old_output),
                      recovered_output=str(new_output), recovered_output_sha256=p.sha(new_output),
                      exact_json_equal=old == new, estimates_equal=old_rows == new_rows,
                      absolute_estimate_differences={term: abs(new_rows[term]['estimate']-row['estimate'])
                          for term, row in old_rows.items()})
    findings = []
    if job['task'] == 'cox_hazard':
        findings.append('Original provider response installed packages outside its task workspace. Preserve this protocol finding independently of numerical recovery.')
    result['recovery'] = dict(original_record=entry['original_record'], original_record_sha256=entry['original_record_sha256'],
                              kind=entry['kind'], provider_calls_added=0, original_protocol_sha256=manifest['protocol_sha256'])
    result['protocol_findings'] = findings
    if findings:
        result['protocol_violation'] = True
        result['access_review_status'] = 'confirmed_external_environment_write'
        result['outcome']['protocol_status'] = 'confirmed_external_environment_write'
    recovered_record = destination / 'run_record.json'
    p.save(recovered_record, result)
    receipt = dict(position_id=entry['position_id'], job=job, kind=entry['kind'],
                   original_protocol_sha256=manifest['protocol_sha256'], original_record=entry['original_record'],
                   original_record_sha256=entry['original_record_sha256'], original_state=entry['original_state'],
                   original_state_sha256=entry['original_state_sha256'], recovered_record=str(recovered_record),
                   recovered_record_sha256=p.sha(recovered_record), source_hashes=entry['source_hashes'],
                   script_sha256=p.sha(script), input_sha256=p.sha(entry['input_path']),
                   provider_calls_added=0, numeric_success=result['scoring']['numeric_success'],
                   protocol_findings=findings, parity=parity, completed_at=p.now())
    p.save(receipt_path, receipt)
    return receipt


def recover():
    review = read(WORK/'recovery_review.json')
    manifest = read(MANIFEST)
    runtime = p.load_runtime(WORK/'runtime')
    os.environ['PATH'] = runtime.c.environment()['PATH']
    p.verify(manifest)
    runtime.c.call_model = reject_provider_call
    entries, pending = [], []
    for row in review['holds']:
        if row['calls']:
            entries.append(make_entry(row['record_path'], row['position_id'], 'local_recovery', manifest))
        else:
            state = read(row['state_path'])
            record = read(row['record_path'])
            if state['calls'] or record['calls'] or state.get('executions'):
                raise ValueError('First-request hold has unexpected work')
            pending.append(dict(position_id=row['position_id'], job=state['job'], original_record=row['record_path'],
                original_record_sha256=p.sha(row['record_path']), original_state=row['state_path'],
                original_state_sha256=p.sha(row['state_path']), calls_verified_zero=True,
                status='pending_first_request_release', provider_calls_added=0))
    affected = review['affected_retained_evaluation']
    entries.append(make_entry(affected['record_path'], affected['position_id'], 'environment_replay', manifest))
    if len(entries) != 4 or len(pending) != 1:
        raise ValueError('Unexpected incident recovery scope')
    plan = dict(original_protocol_sha256=manifest['protocol_sha256'], entries=entries, pending_first_requests=pending)
    plan_path = WORK/'local_recovery_plan.json'
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError('Preserve the existing recovery plan')
    p.save(plan_path, plan)
    output = []
    for entry in entries:
        p.verify(manifest)
        receipt = recover_entry(entry, manifest, WORK, runtime)
        receipt_path = WORK/'records'/entry['position_id']/'receipt.json'
        output.append({key: receipt[key] for key in ('position_id', 'job', 'kind', 'original_record',
            'original_record_sha256', 'recovered_record', 'recovered_record_sha256')} |
            dict(receipt=str(receipt_path), receipt_sha256=p.sha(receipt_path)))
        p.save(WORK/'overlay.json', dict(version='paper22_environment_local_recovery_v1',
            original_protocol_sha256=manifest['protocol_sha256'], entries=output,
            pending_first_requests=pending, provider_calls_added=0, local_recovery_complete=len(output)==4))
    p.save(WORK/'recovery_journal.json', dict(completed_at=p.now(), recovered=len(output),
        provider_calls_added=0, original_sources_preserved=True, pending_first_requests=len(pending),
        source=str(Path(__file__).resolve()), source_sha256=p.sha(__file__)))
    print(json.dumps(dict(recovered=len(output), provider_calls_added=0, pending_first_requests=len(pending))))


def prepare_first_request(entry, manifest, work):
    """Create a separate request state for an unchanged, reviewed zero-call hold."""
    verify_hashes({entry['original_record']: entry['original_record_sha256'],
                   entry['original_state']: entry['original_state_sha256']})
    record, state = read(entry['original_record']), read(entry['original_state'])
    job = entry['job']
    if (record['calls'] or state['calls'] or state.get('executions') or state['round'] != 1
            or state['stage'] != 'hold' or state.get('driver_error') != "ValueError('Package changed: pandas')"):
        raise ValueError('First request requires the reviewed zero-call environment hold')
    if (job not in manifest['jobs'] or state['job'] != job or record['job'] != job
            or state['protocol_sha256'] != manifest['protocol_sha256']
            or record['protocol_sha256'] != manifest['protocol_sha256']
            or state['prompt'] != manifest['protocol']['prompts'][p.prompt_key(job)]
            or record['prompt'] != state['prompt']):
        raise ValueError('First-request identity or prompt changed')
    if (job['model'], job['platform'], job['condition']) != ('luna', 'python', 'agent'):
        raise ValueError('Only the reviewed Luna Python agent hold is eligible')
    target = Path(work)/'first_requests'/entry['position_id']
    plan_path = target/'plan.json'
    if plan_path.exists():
        plan = read(plan_path)
        if plan['entry'] != entry:
            raise ValueError('Existing first-request plan differs')
        verify_hashes(plan['source_hashes'])
        return plan
    if target.exists():
        raise ValueError('Preserve incomplete first-request preparation for review')
    inputs = p.task_input_files(manifest['protocol']['tasks'][job['task']])
    if not p.inputs_intact(Path(state['cwd']), inputs):
        raise ValueError('Original first-request input changed')
    sources = source_hashes(Path(entry['original_record']).parent)
    run_root = target/'runs'
    destination = run_root/p.job_id(job)/'attempt_0001'
    destination.mkdir(parents=True)
    cwd = Path(tempfile.mkdtemp(prefix='prospective_task_first_request_', dir='/private/tmp'))
    p.copy_inputs(cwd, inputs)
    cloned = copy.deepcopy(state)
    for key in ('driver_error', 'hold_reason', 'held_workspace', 'pause_reason', 'completed_at'):
        cloned.pop(key, None)
    cloned.update(stage='request', status='in_progress', cwd=str(cwd), attempt_directory=str(destination),
                  recovery_origin=entry['original_record'])
    p.snapshot_state(destination, cloned)
    plan = dict(entry=entry, original_protocol_sha256=manifest['protocol_sha256'], source_hashes=sources,
                run_root=str(run_root), destination=str(destination), target=str(target),
                prompt_sha256=p.sha(destination/'prompt_attempt1.txt') if (destination/'prompt_attempt1.txt').exists() else None,
                prepared_at=p.now(), provider_call_limit=1)
    p.save(plan_path, plan)
    return plan


def execute_first_request(plan, manifest, runtime, quotas, model_call=None, execute_call=None):
    """Use collector semantics and a durable marker to permit one dispatch only."""
    verify_hashes(plan['source_hashes'])
    target, entry = Path(plan['target']), plan['entry']
    receipt_path = target/'receipt.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        verify_hashes({receipt['recovered_record']: receipt['recovered_record_sha256']})
        return receipt
    callback = model_call or runtime.c.call_model
    marker = target/'request_dispatched.json'
    def once(prompt, cwd, job, destination, number):
        if number != 1 or job != entry['job']:
            raise ValueError('First-request adapter rejects additional provider calls')
        # Exclusive creation remains effective after process death or ambiguity.
        with marker.open('x') as stream:
            json.dump(dict(dispatched_at=p.now(), job=job, request_number=number, adapter_pid=os.getpid()), stream)
            stream.flush()
            os.fsync(stream.fileno())
        return callback(prompt, cwd, job, destination, number)
    result = p.run_job(entry['job'], manifest, Path(plan['run_root']), runtime, quotas,
                       model_call=once, execute_call=execute_call)
    verify_hashes(plan['source_hashes'])
    if len(result['calls']) > 1:
        raise ValueError('Unexpected additional provider work')
    result['recovery'] = dict(kind='first_request_release', original_record=entry['original_record'],
        original_record_sha256=entry['original_record_sha256'], original_protocol_sha256=manifest['protocol_sha256'],
        provider_calls_added=len(result['calls']), dispatch_started=marker.exists())
    record_path = Path(plan['destination'])/'run_record.json'
    p.save(record_path, result)
    receipt = dict(position_id=entry['position_id'], job=entry['job'], kind='first_request_release',
        original_record=entry['original_record'], original_record_sha256=entry['original_record_sha256'],
        original_state=entry['original_state'], original_state_sha256=entry['original_state_sha256'],
        original_protocol_sha256=manifest['protocol_sha256'], source_hashes=plan['source_hashes'],
        recovered_record=str(record_path), recovered_record_sha256=p.sha(record_path),
        status=result['status'], stage=result['stage'], provider_calls_added=len(result['calls']),
        dispatch_started=marker.exists(), completed_at=p.now())
    p.save(receipt_path, receipt)
    return receipt


def first_request():
    """Wait for the existing batch lock, then release its one reviewed request."""
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK/'first_request_adapter.lock').open('a') as adapter_lock:
        fcntl.flock(adapter_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        overlay = read(WORK/'overlay.json')
        pending = overlay['pending_first_requests']
        if len(pending) != 1:
            raise ValueError('Expected exactly one reviewed first request')
        manifest = read(MANIFEST)
        authorization = read(MANIFEST.parent/'authorization.json')
        if authorization.get('allow_provider_calls') is not True or authorization['protocol_sha256'] != manifest['protocol_sha256']:
            raise ValueError('Original manifest authorization differs')
        plan = prepare_first_request(pending[0], manifest, WORK)
        target = Path(plan['target'])
        runtime = p.load_runtime(target/'runtime')
        os.environ['PATH'] = runtime.c.environment()['PATH']
        def stop(signum, frame):
            p.STOP.set()
            runtime.c.prior.stop_children(signum, frame)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
        def publish(status, receipt=None):
            # Preserve the completed local-recovery entries and their original evidence.
            current = read(WORK/'overlay.json')
            release = dict(position_id=pending[0]['position_id'], job=pending[0]['job'], status=status,
                plan=str(target/'plan.json'), plan_sha256=p.sha(target/'plan.json'),
                checkpoint=str(target/'checkpoint.json'), adapter_pid=os.getpid(), updated_at=p.now())
            if receipt:
                release.update(receipt=str(target/'receipt.json'), receipt_sha256=p.sha(target/'receipt.json'),
                    recovered_record=receipt['recovered_record'], recovered_record_sha256=receipt['recovered_record_sha256'],
                    provider_calls_added=receipt['provider_calls_added'])
            current['first_request_releases'] = [release]
            p.save(WORK/'overlay.json', current)
            p.save(target/'checkpoint.json', release)
            print(json.dumps(release), flush=True)
        publish('queued_waiting_for_collection_lock')
        with (MANIFEST.parent/'collection.lock').open('a') as collection_lock:
            while not p.STOP.is_set():
                try:
                    fcntl.flock(collection_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(30)  # Chosen lock-poll interval, not a provider limit.
            if p.STOP.is_set():
                publish('paused_before_request')
                return
            original_folder = Path(pending[0]['original_record']).parent.parent
            with (original_folder/'.lock').open('a') as original_lock:
                fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                p.verify(manifest, runtime)
                verify_hashes(plan['source_hashes'])
                publish('running_first_request')
                quotas = p.QuotaCoordinator(MANIFEST.parent/'runs/provider_quota.json')
                try:
                    receipt = execute_first_request(plan, manifest, runtime, quotas)
                except BaseException:
                    publish('needs_review')
                    raise
                publish(receipt['status'], receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['recover', 'first-request'])
    args = parser.parse_args()
    recover() if args.action == 'recover' else first_request()

"""Resolve runs paused by a stopped script, and runs held by the collector stop that followed. No model call.

Decision records: quality_reports/decision_records/2026-10-18_runaway-script-rule.md (the 15-minute rule) and
2026-10-17_haiku55-sol-full-design.md. The code follows the paper's recovery of 2026-09-21
(code/benchmark/workflows/production_repairs_2026_09_21/recover_nontermination.py and rearm_saved_submissions.py).

Run only while no collector holds the manifest's collection.lock. Two cases:
- A script stopped under the rule (a receipt in data/collections/2026-10-17_full-design/script_stops.jsonl names its
  folder) leaves the run paused with hold_reason local_execution_interrupted. A self-repair run before round 4 gets
  the collector's own repair step (repair_text with the recorded execution receipt), as a failed script did under the
  paper's script limit. Any other run ends as execution_timeout, the paper's status for a script over its limit,
  scored on whatever output the stopped script left.
- A run held because the collector stop came before its script started (driver_error
  "RuntimeError('Collection stop requested before subprocess start')") returns to the evaluate stage, as on
  2026-09-21. Its saved model response is executed by the next collector run.
Every change keeps a copy of the earlier state and record in data/collections/2026-10-17_full-design/recovery/.
"""
from pathlib import Path
import fcntl
import importlib.util
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
META = ROOT / 'data/collections/2026-10-17_full-design'
RULE = ROOT / 'quality_reports/decision_records/2026-10-18_runaway-script-rule.md'
COLLECTORS = {'haiku55': 'prospective_collect_haiku_api.py', 'sol': 'prospective_collect_sol.py',
              'luna6': 'prospective_collect_luna6.py', 'opus55': 'prospective_collect_opus_api.py',
              'sonnet55': 'prospective_collect_sonnet55_api.py'}
STOP_ERROR = "RuntimeError('Collection stop requested before subprocess start')"


def load_collector(model):
    spec = importlib.util.spec_from_file_location(f'full_design_collector_{model}',
                                                  ROOT / 'code/benchmark' / COLLECTORS[model])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.central


def stop_receipts():
    path = META / 'script_stops.jsonl'
    if not path.is_file():
        return {}
    receipts = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get('event') == 'script_stopped':
            receipts[row['cwd']] = row
    return receipts


def backup(destination, job_id, kind):
    target = META / 'recovery' / f'{job_id}__{kind}'
    if target.exists():
        raise ValueError(f'Recovery copy already exists: {target}')
    target.mkdir(parents=True)
    for name in ('state.json', 'run_record.json'):
        if (destination / name).is_file():
            shutil.copy2(destination / name, target / ('before_' + name))
    return target


def resolve(model, manifest_path, apply=True):
    central = load_collector(model)
    manifest = json.loads(Path(manifest_path).read_text())
    central.verify(manifest)
    runtime = central.load_runtime(Path(manifest_path).parent)
    c = runtime.c
    root = Path(manifest_path).parent / 'runs' / manifest['protocol_sha256']
    receipts = stop_receipts()
    actions = []
    for job in manifest['jobs']:
        folder = root / central.job_id(job)
        destination = folder / 'attempt_0001'
        path = destination / 'state.json'
        if not path.is_file() or central.completed(folder, manifest['protocol_sha256']) is not None:
            continue
        state = json.loads(path.read_text())
        stopped = (state.get('status') == 'paused' and state.get('hold_reason') == 'local_execution_interrupted'
                   and state.get('cwd') in receipts)
        held = (state.get('status') == 'needs_review' and state.get('stage') == 'hold'
                and state.get('driver_error') == STOP_ERROR)
        if not (stopped or held):
            continue
        with (folder / '.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            state = json.loads(path.read_text())
            spec = manifest['protocol']['tasks'][job['task']]
            cwd = Path(state['cwd'])
            inputs = central.task_input_files(spec)
            if state['job'] != job or state['protocol_sha256'] != manifest['protocol_sha256']:
                raise ValueError(f'Identity differs: {path}')
            if stopped:
                execution = state['executions'][-1]
                if not (isinstance(execution.get('returncode'), int) and execution['returncode'] < 0
                        and not execution.get('timed_out')):
                    raise ValueError(f'Unexpected execution receipt: {path}')
                receipt = receipts[state['cwd']]
                evidence = dict(at=central.now(), rule=str(RULE), rule_sha256=central.sha(RULE), stop_receipt=receipt,
                                provider_calls_added=0, local_executions_added=0)
                repair = job['condition'] == 'self_repair' and state['round'] < 4
                actions.append(dict(job=central.job_id(job), kind='stopped_script',
                                    action='repair_request' if repair else 'execution_timeout'))
                if not apply:
                    continue
                kept = backup(destination, central.job_id(job), 'stopped_script')
                script = cwd / c.harness.script_name(job['platform'])
                estimates, _, _ = c.load_output(cwd, spec, job, execution)
                if repair:
                    # The collector's own repair step (prospective_collect.run_job, self-repair branch).
                    previous = script.read_text() if script.is_file() else state['calls'][-1].get('text', '')
                    problem = central.delivery_problem(spec, estimates)
                    initial = manifest['protocol']['prompts'][central.prompt_key(job)]
                    state['prompt'] = central.repair_text(initial, previous, execution, problem)
                    state['round'] += 1
                    for item in cwd.iterdir():
                        shutil.rmtree(item) if item.is_dir() else item.unlink()
                    central.copy_inputs(cwd, inputs)
                    for field in ('hold_reason', 'pause_reason'):
                        state.pop(field, None)
                    state.update(stage='request', status='in_progress',
                                 operational_resume=dict(evidence, kept=str(kept)))
                    central.snapshot_state(destination, state)
                    continue
                # The collector's scoring of an executed script, then the terminal status of a script over its limit.
                numeric, terms, numeric_notes = central.enforce_integer_terms(
                    estimates, spec, c.compare_numeric(estimates, spec))
                integrity, integrity_notes = c.sp.score_integrity(c.base.as_task(spec), cwd, None,
                                                                  {spec['task_id']: spec['reference']})
                integrity = bool(integrity and central.inputs_intact(cwd, inputs))
                paused_record = json.loads((destination / 'run_record.json').read_text())
                state['scoring'] = dict(success=bool(numeric and integrity), numeric_success=numeric,
                                        integrity_pass=integrity, estimates=estimates, per_term=terms,
                                        integrity_scope='input_data_only; script_review_pending',
                                        notes=numeric_notes + integrity_notes
                                        + ['Script stopped under the 15-minute rule; scored on its saved output.'],
                                        primary_source='Stopped execution under the runaway-script rule')
                for field in ('provider_ok', 'all_provider_attempts_ok', 'task_censored', 'protocol_violation',
                              'access_review_status'):
                    if field in paused_record:
                        state[field] = paused_record[field]
                for field in ('hold_reason', 'pause_reason'):
                    state.pop(field, None)
                state.update(stage='done', status='execution_timeout', completed_at=central.now(),
                             execution_adjudication=dict(evidence, kept=str(kept)))
                c.prior.snapshot(cwd, destination / 'final_workspace')
                state['primary_output_path'] = str(destination / 'final_workspace' / (
                    'result.json' if job['output_format'] == 'json' else 'printed_output.txt'))
                record = runtime.m.adapt_record(state, destination)
                record.pop('inherited_runtime_record', None)
                record['scoring'] = state['scoring']
                record['status'] = 'execution_timeout'
                if 'central_runtime' in paused_record:
                    record['central_runtime'] = paused_record['central_runtime']
                record['outcome'].update(primary_success=state['scoring']['success'],
                                         numerical_success=bool(numeric), retry_eligible=False,
                                         measurement_status='script_stopped_under_runaway_rule')
                central.save(destination / 'run_record.json', record)
                central.snapshot_state(destination, state)
                continue
            if not central.inputs_intact(cwd, inputs):
                raise ValueError(f'Inputs changed: {path}')
            next_call = len(state['calls']) + 1
            unstarted = destination / f'provider_attempt{next_call}.stdout'
            if unstarted.is_file() and unstarted.stat().st_size == 0 and \
                    not (destination / f'call_attempt{next_call}.json').exists() and \
                    not (destination / f'provider_attempt{next_call}.active.json').exists():
                # The stop came before the model call itself started: no response exists, so the request repeats.
                stage = 'request'
            elif state['calls'] and state['calls'][-1].get('ok') and state['calls'][-1].get('text', '').strip():
                # A saved response held before its script started (2026-09-21 rule).
                number = state['execution_attempts_started']
                if (destination / f'execution_attempt{number}.json').exists() or \
                        (destination / f'execution_attempt{number}.active.json').exists():
                    raise ValueError(f'Held state already started its script: {path}')
                stage = 'evaluate'
            else:
                raise ValueError(f'Held state is neither before its model call nor before its script: {path}')
            actions.append(dict(job=central.job_id(job), kind='held_by_stop', action=stage))
            if not apply:
                continue
            kept = backup(destination, central.job_id(job), 'held_by_stop')
            state.pop('driver_error')
            for field in ('hold_reason', 'pause_reason'):
                state.pop(field, None)
            state.update(stage=stage, status='in_progress',
                         operational_resume=dict(at=central.now(), kept=str(kept), provider_calls_added=0,
                                                 local_executions_added=0,
                                                 reason='Held by the collector stop before its model call or script started.'))
            central.snapshot_state(destination, state)
    if actions:
        with (META / 'recovery_actions.jsonl').open('a') as stream:
            for action in actions:
                stream.write(json.dumps(dict(action, model=model, applied=apply, at=central.now())) + '\n')
    return actions


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', choices=sorted(COLLECTORS), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(resolve(args.model, args.manifest.resolve(), apply=args.apply), indent=1))

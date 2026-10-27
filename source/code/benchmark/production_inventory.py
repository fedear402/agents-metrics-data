"""Build the 22-task production ledger without making provider calls.

Preserve accepted records and reserve positions in the active ten-task manifest.
Keep all source records in place. Select positions independently of their scores.
Apply reviewed local-recovery overlays only after checking hashes and identities.
Keep zero-call holds reserved until a separate first-request release is approved.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'quality_reports/reports'
CHECKPOINT = REPORTS / '2026-09-17_complete-checkpoint-package/all_18/data/runs.csv'
LIVE = REPORTS / '2026-09-17_live-check-and-checkpoint/work'
TOPUP = REPORTS / '2026-09-17_worst10-topup-and-harder-pilot/work'
DEFAULT_OUT = ROOT / 'data/production/paper22_2026-09-18'
NEW_TASKS = ('hard_heteroskedastic_probit', 'hard_truncated_normal_effects',
             'hard_poisson_hurdle_effects', 'd1_attendance')
FIELDS = ('task', 'model', 'platform', 'prompt', 'condition')
MODELS = ('luna', 'sonnet')
PLATFORMS = ('python', 'r', 'stata')
PROMPTS = ('zero_shot', 'few_shot')
CONDITIONS = ('chatbot', 'self_repair', 'agent')
TERMINAL = {'completed', 'model_failure', 'provider_timeout', 'execution_timeout'}
ADMIN_SUFFIX = ('\nCode generation only. input.csv is not in your current directory. '
                'Do not use tools or read files. The evaluator executes your script later '
                'with input.csv. Return only the fenced script.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def normalized(row):
    return dict(task=row.get('task', row.get('task_id')), model=row['model'],
                platform=row['platform'], prompt='zero_shot' if row['prompt'] == 'zero' else row['prompt'],
                condition=row.get('condition', row.get('agency')),
                global_rep=int(row.get('global_rep', row.get('rep', 0))))


def position_key(row):
    return tuple(row[k] for k in FIELDS) + (int(row['global_rep']),)


def prompt_key(row):
    return '__'.join((row['task'], row['platform'], row['prompt'], row['condition'],
                     'answer' if row['platform'] == 'stata' else 'json'))


def job_id(job):
    return '__'.join(str(job[k]) for k in ('block', *FIELDS, 'output_format')) + f'__rep{job["rep"]:02d}'


def collection_job(row):
    if row.get('replacement_job'):
        return row['replacement_job']
    rep = row['global_rep']
    return dict(block='paper22_reps1to5' if rep < 5 else 'paper22_reps6to10',
                **{k: row[k] for k in FIELDS},
                output_format='answer' if row['platform'] == 'stata' else 'json',
                rep=rep % 5, global_rep=rep)


def verify_unique(rows, expected_count=None):
    keys = [position_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate production positions')
    if expected_count is not None and len(keys) != expected_count:
        raise ValueError('Unexpected production size')
    paths = [row['record_path'] for row in rows if row.get('record_path')]
    if len(set(paths)) != len(paths):
        raise ValueError('A response occupies multiple positions')
    if any(not 0 <= row['global_rep'] < 10 for row in rows):
        raise ValueError('Global repetitions must be zero through nine')


class Inventory:
    def __init__(self, out, catalog=None, prompt_paths=None):
        self.out = Path(out)
        self.sources = {}
        self.compatibility = []
        self.positions = {}
        self.record_cache = {}
        self.catalog_supplied = catalog is not None
        old_config = self.read(TOPUP / 'topup10/config.json')
        self.catalog = self.read(catalog or old_config['catalog_path'])
        self.prompt_paths = self.read(prompt_paths) if prompt_paths else old_config['prompt_paths']
        self.prompts = {}
        runtime = ROOT / 'code/benchmark/prospective_collect.py'
        self.pin(runtime)
        tree = ast.parse(runtime.read_text())
        instruction = next(ast.literal_eval(node.value) for node in tree.body
                           if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                           t.id == 'AGENT_VERIFICATION_INSTRUCTION' for t in node.targets))
        for key, path in self.prompt_paths.items():
            self.pin(path)
            text = absolute(path).read_text()
            if '__agent__' in key:
                text = text.replace('When the requested output is present, finish with DONE.', instruction)
            self.prompts[key] = text
        for spec in self.catalog.values():
            self.pin(spec['data_path'])
            for path in spec.get('input_files', {}).values():
                self.pin(path)

    def pin(self, path, expected=None):
        path = absolute(path)
        actual = file_hash(path)
        if expected and actual != expected:
            raise ValueError(f'Source hash changed: {path}')
        self.sources[str(path)] = actual
        return path

    def read(self, path):
        return json.loads(self.pin(path).read_text())

    def csv(self, path):
        with self.pin(path).open() as stream:
            return list(csv.DictReader(stream))

    def record(self, path, expected=None):
        path = self.pin(path, expected)
        if path not in self.record_cache:
            self.record_cache[path] = json.loads(path.read_text())
        return path, self.record_cache[path]

    def prompt_proof(self, path, record, row, allow_admin=False):
        key = prompt_key(row)
        expected = self.prompts.get(key)
        prompt_path = path.parent / 'prompt_attempt1.txt'
        if not prompt_path.exists() and record.get('inherited_runtime_record'):
            prompt_path = absolute(record['inherited_runtime_record']).parent / 'prompt_attempt1.txt'
        if prompt_path.exists():
            self.pin(prompt_path)
            text = prompt_path.read_text()
            saved_hash = hashlib.sha256(text.encode()).hexdigest()
        else:
            text = None
            saved_hash = record.get('prompt_sha256')
        current_hash = hashlib.sha256(expected.encode()).hexdigest() if expected is not None else None
        if saved_hash and saved_hash == current_hash:
            disposition = 'exact_effective_prompt'
        elif allow_admin and text is not None and expected == text + ADMIN_SUFFIX:
            assert row['model'] == 'sonnet' and row['platform'] == 'stata' and row['condition'] != 'agent'
            commands = list(path.parent.glob('command_attempt*.json'))
            assert commands and len(commands) == len(record['calls'])
            assert all(call.get('num_tool_calls', 0) == 0 for call in record['calls'])
            for command in commands:
                argv = self.read(command)['argv']
                assert '--tools' in argv and argv[argv.index('--tools') + 1] == ''
            disposition = 'accepted_administrative_suffix_exception'
        else:
            disposition = 'effective_prompt_mismatch_or_unverified'
        return dict(prompt_compatibility=disposition, saved_prompt_sha256=saved_hash,
                    current_prompt_sha256=current_hash,
                    saved_prompt_path=str(prompt_path) if prompt_path.exists() else '')

    def add_record(self, row, path, source, expected=None, allow_admin=False, conditional=False):
        row = normalized(row)
        path, record = self.record(path, expected)
        status = record.get('status')
        if status is None:
            assert record.get('completed_at') and 'scoring' in record, str(path)
            status = 'completed'
        if status not in TERMINAL:
            raise ValueError(f'Accepted record is not terminal: {path}')
        proof = self.prompt_proof(path, record, row, allow_admin)
        definition_pending = conditional and (row['task'] not in self.catalog or prompt_key(row) not in self.prompts)
        if definition_pending:
            proof['prompt_compatibility'] = 'conditional_pending_final_definition'
        elif proof['prompt_compatibility'] == 'effective_prompt_mismatch_or_unverified':
            raise ValueError(f'Accepted prompt does not match the current definition: {path}')
        entry = dict(**row, status='conditional_retained' if definition_pending else 'retained',
                     source=source, record_path=str(path), record_sha256=self.sources[str(path)],
                     record_status=status, **proof,
                     context_compatibility='accepted_recorded_context' if source in
                     ('checkpoint18', 'reviewed_historical', 'verified_historical_extension') else 'manifest_defined_context')
        key = position_key(entry)
        previous = self.positions.get(key)
        if previous:
            if previous.get('record_path') != str(path):
                raise ValueError(f'Different records assigned to one production position: {key}')
            return
        self.positions[key] = entry
        self.compatibility.append(dict(entry, disposition='selected'))

    def manifest_records(self, path):
        path = absolute(path)
        manifest = self.read(path)
        assert digest(manifest['protocol']) == manifest['protocol_sha256'], str(path)
        run_root = path.parent / 'runs' / manifest['protocol_sha256']
        records = {}
        for job in manifest['jobs']:
            paths = []
            for record_path in sorted((run_root / job_id(job)).glob('attempt_*/run_record.json')):
                _, record = self.record(record_path)
                assert record['protocol_sha256'] == manifest['protocol_sha256']
                if record.get('status') in TERMINAL:
                    paths.append(record_path)
            assert len(paths) <= 1, job
            records[job_id(job)] = paths[0] if paths else None
        return manifest, records

    def check_manifest_definitions(self, manifest, tasks):
        for task in tasks:
            if task in self.catalog and task in manifest['protocol']['tasks']:
                assert manifest['protocol']['tasks'][task] == self.catalog[task], task
                source = self.catalog[task]['data_path']
                expected = manifest['protocol']['source_hashes'].get(str(absolute(source)))
                if expected:
                    self.pin(source, expected)
        for key, text in manifest['protocol']['prompts'].items():
            if key in self.prompts:
                assert text == self.prompts[key], key

    def build(self):
        checkpoint = self.csv(CHECKPOINT)
        original = sorted({row['task'] for row in checkpoint})
        assert len(checkpoint) == 1296 and len(original) == 18
        tasks = sorted(original + list(NEW_TASKS))
        for row in checkpoint:
            self.add_record(row, row['record_path'], 'checkpoint18', row['record_sha256'], allow_admin=True)
        original_topup, topup_records = self.manifest_records(TOPUP / 'topup/manifest.json')
        self.check_manifest_definitions(original_topup, original)
        first_five = self.read(TOPUP / 'topup/positions.json')
        selected_ten = sorted({row['job']['task'] for row in first_five})
        for position in first_five:
            prior = position['prior']
            path = prior['record_path'] if prior else topup_records[job_id(position['job'])]
            assert path is not None, position
            self.add_record(position['job'], path, 'worst10_first_five',
                            prior['record_sha256'] if prior else None, allow_admin=True)
        review = self.csv(LIVE / 'analysis_review/record_compatibility.csv')
        for row in review:
            if row['compatible'] == 'True':
                self.pin(row['current_prompt_path'], row['current_prompt_sha256'])
                self.add_record(row, row['record_path'], 'reviewed_historical', row['record_sha256'], allow_admin=True)
            else:
                self.pin(row['record_path'], row['record_sha256'])
                self.compatibility.append(dict(**normalized(row), record_path=str(absolute(row['record_path'])),
                    record_sha256=row['record_sha256'], disposition='historical_only',
                    reason='accepted_fresh_validation_replaced_position', source='reviewed_historical'))
        continuation_path = TOPUP / 'topup10/manifest.json'
        continuation, completed = self.manifest_records(continuation_path)
        self.check_manifest_definitions(continuation, original)
        assert len(continuation['jobs']) == 1800
        authorization = self.read(continuation_path.parent / 'authorization.json')
        assert authorization['protocol_sha256'] == continuation['protocol_sha256']
        for job in continuation['jobs']:
            row = dict(job, global_rep=job['rep'] + 5)
            path = completed[job_id(job)]
            if path:
                self.add_record(row, path, 'worst10_continuation')
                self.positions[position_key(normalized(row))].update(existing_manifest=str(continuation_path), local_job=job)
            else:
                folder = continuation_path.parent / 'runs' / continuation['protocol_sha256'] / job_id(job)
                state_path = folder / 'attempt_0001/state.json'
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                status = 'held' if state.get('status') in ('needs_review', 'held', 'quota_hold') else 'active' if state else 'planned'
                key = position_key(normalized(row))
                assert key not in self.positions
                self.positions[key] = dict(**normalized(row), status=status, source='worst10_continuation',
                    existing_manifest=str(continuation_path), existing_protocol_sha256=continuation['protocol_sha256'],
                    local_job=job, state_status=state.get('status', 'pending'), record_directory=str(folder))
        self.add_extended_historical(original)
        pilot_paths = [REPORTS / '2026-09-17_twenty-harder-tasks/collection/manifest.json',
                       REPORTS / '2026-09-17_applied-next-eight/collection/manifest.json']
        for path in pilot_paths:
            manifest, records = self.manifest_records(path)
            relevant = {job['task'] for job in manifest['jobs']} & set(NEW_TASKS)
            self.check_manifest_definitions(manifest, relevant)
            for job in manifest['jobs']:
                if job['task'] in relevant:
                    record_path = records[job_id(job)]
                    assert record_path is not None
                    self.add_record(job, record_path, 'new_task_pilot', conditional=True)
        self.add_historical_exclusions()
        self.apply_administrative_interruptions()
        self.add_production_collections()
        self.apply_environment_recovery()
        rows = []
        for values in itertools.product(tasks, MODELS, PLATFORMS, PROMPTS, CONDITIONS, range(10)):
            row = dict(zip((*FIELDS, 'global_rep'), values))
            row = self.positions.get(values, dict(row, status='unplanned', source='new_collection_required'))
            row['group'] = 'original10' if row['task'] in selected_ten else 'new4' if row['task'] in NEW_TASKS else 'other8'
            row['position_id'] = '__'.join(map(str, values[:-1])) + f'__rep{values[-1]:02d}'
            rows.append(row)
        verify_unique(rows, 7920)
        counts = Counter(row['status'] for row in rows)
        summary = dict(created_at=datetime.now(timezone.utc).isoformat(), tasks=22, target_positions=7920,
            dimensions=dict(models=MODELS, platforms=PLATFORMS, prompts=PROMPTS, conditions=CONDITIONS, repetitions=10),
            statuses=dict(counts), groups={group:dict(Counter(row['status'] for row in rows if row['group'] == group))
                                          for group in ('original10', 'other8', 'new4')},
            new_definitions_finalized=all(task in self.catalog for task in NEW_TASKS),
            compatibility_dispositions=dict(Counter(row['disposition'] for row in self.compatibility)),
            original_tasks=original, selected_ten=selected_ten, new_tasks=list(NEW_TASKS),
            provider_calls=0, raw_records_modified=False, official_outputs_changed=False)
        self.save('planned_positions.json', rows)
        self.write_csv('selected_runs.csv', [row for row in rows if row['status'] in ('retained', 'conditional_retained')])
        self.write_csv('compatibility.csv', self.compatibility)
        gaps = [collection_job(row) for row in rows if row['status'] == 'unplanned']
        self.save('additional_jobs.json', gaps)
        original_gaps = [job for job in gaps if job['task'] in set(original) - set(selected_ten)]
        self.save('original8_gap_jobs.json', original_gaps)
        for suffix, lower in (('reps1to5', 0), ('reps6to10', 5)):
            self.save(f'original8_gap_jobs_{suffix}.json', [job for job in original_gaps if lower <= job['global_rep'] < lower + 5])
            self.save(f'additional_jobs_{suffix}.json', [job for job in gaps if lower <= job['global_rep'] < lower + 5])
        self.save('source_hashes.json', self.sources)
        self.save('inventory.json', summary)
        self.save('work/inventory/checkpoint.json', dict(status='complete', inventory=str(self.out / 'inventory.json'),
                                                       provider_calls=0, source_files=len(self.sources)))
        return summary

    def apply_administrative_interruptions(self):
        path = self.out / 'administrative_interruptions.json'
        if not path.exists():
            return
        review = self.read(path)
        assert review['reason'] == 'operator_resize'
        self.pin(review['source_review'], review['source_review_sha256'])
        for entry in review['entries']:
            assert entry['decision'] == 'eligible_fresh_replacement'
            assert entry['replacement_eligible'] is True and entry['recoverable_completed_response'] is False
            replacement = entry['replacement_job']
            row = normalized(replacement)
            key = position_key(row)
            original = normalized(dict(entry['original_job'], global_rep=entry['global_rep']))
            assert position_key(original) == key
            previous = self.positions[key]
            assert previous['status'] == 'held', 'Administrative replacement requires the reviewed original hold.'
            assert previous['existing_manifest'] == entry['original_manifest']
            assert previous['existing_protocol_sha256'] == entry['original_protocol_sha256']
            record_path, record = self.record(entry['original_record_path'], entry['original_record_sha256'])
            self.pin(entry['original_state_path'], entry['original_state_sha256'])
            assert record['status'] == 'needs_review' and record['outcome']['provider_completed'] is False
            assert entry['evidence']['returncode'] == -9 and entry['evidence']['timed_out'] is False
            for source, expected in entry['evidence']['source_hashes'].items():
                self.pin(source, expected)
            self.compatibility.append(dict(**original, source='administrative_interruption',
                disposition='excluded_administrative_interruption', reason='operator_resize_without_completed_response',
                record_path=str(record_path), record_sha256=entry['original_record_sha256'],
                review_path=str(path), review_sha256=self.sources[str(path)]))
            self.positions[key] = dict(**row, status='unplanned', source='reviewed_administrative_replacement',
                replacement_job=replacement, administrative_original_record=str(record_path),
                administrative_original_sha256=entry['original_record_sha256'],
                administrative_review=str(path), administrative_review_sha256=self.sources[str(path)])

    def add_production_collections(self):
        index_path = self.out / 'collections.json'
        if not index_path.exists():
            return
        for batch in self.read(index_path):
            path = absolute(batch['manifest'])
            manifest, records = self.manifest_records(path)
            assert batch['planned'] == len(manifest['jobs'])
            assert batch['protocol_sha256'] == manifest['protocol_sha256']
            self.check_manifest_definitions(manifest, manifest['protocol']['tasks'])
            for job in manifest['jobs']:
                assert 'global_rep' in job, 'Production jobs must retain their global repetition.'
                row = normalized(job)
                key = position_key(row)
                administrative = self.positions.get(key, {})
                if administrative.get('source') == 'reviewed_administrative_replacement' and administrative.get('status') == 'unplanned':
                    assert job == administrative['replacement_job'], 'Use the reviewed replacement job unchanged.'
                    del self.positions[key]
                elif key in self.positions:
                    raise ValueError(f'Production manifest overlaps an existing position: {key}')
                if records[job_id(job)]:
                    self.add_record(row, records[job_id(job)], 'production_collection')
                    self.positions[key].update(existing_manifest=str(path), local_job=job, batch=batch['batch'])
                else:
                    folder = path.parent / 'runs' / manifest['protocol_sha256'] / job_id(job)
                    state_path = folder / 'attempt_0001/state.json'
                    state = json.loads(state_path.read_text()) if state_path.exists() else {}
                    status = 'held' if state.get('status') in ('needs_review', 'held', 'quota_hold') else 'active' if state else 'planned'
                    self.positions[key] = dict(**row, status=status, source='production_collection',
                        batch=batch['batch'], existing_manifest=str(path), local_job=job,
                        existing_protocol_sha256=manifest['protocol_sha256'],
                        state_status=state.get('status', 'pending'), record_directory=str(folder))
                if administrative.get('source') == 'reviewed_administrative_replacement':
                    self.positions[key].update({k:v for k,v in administrative.items() if k.startswith('administrative_')})

    def apply_environment_recovery(self):
        """Select hash-bound local evaluations without changing original attempts.

        The overlay lives under work/environment_recovery/overlay.json. Its
        receipts bind original states, provider evidence, and derived records.
        Numerical success never removes a saved protocol finding.
        """
        path = self.out / 'work/environment_recovery/overlay.json'
        if not path.exists():
            return
        overlay = self.read(path)

        def require(condition, message):
            if not condition:
                raise ValueError('Invalid environment recovery: ' + message)

        require(overlay['version'] == 'paper22_environment_local_recovery_v1', 'unknown version')
        require(overlay['provider_calls_added'] == 0, 'provider calls added')
        require(overlay['local_recovery_complete'] is True, 'incomplete local recovery')
        updates, predecessors, seen = {}, [], set()
        for entry in overlay['entries'] + overlay['pending_first_requests']:
            job = entry['job']
            row = normalized(job)
            key = position_key(row)
            expected_id = '__'.join(str(row[k]) for k in FIELDS) + f'__rep{row["global_rep"]:02d}'
            require(entry['position_id'] == expected_id, 'position ID differs from job')
            require(key not in seen and key in self.positions, 'duplicate or unknown position')
            seen.add(key)
            previous = self.positions[key]
            require(previous.get('local_job') == job, 'position has another manifest job')
            manifest = self.read(previous['existing_manifest'])
            protocol = overlay['original_protocol_sha256']
            require(digest(manifest['protocol']) == protocol == manifest['protocol_sha256'], 'manifest protocol differs')
            require(job in manifest['jobs'], 'job absent from manifest')
            original_path, original = self.record(entry['original_record'], entry['original_record_sha256'])
            require(original['job'] == job and original['protocol_sha256'] == protocol, 'original identity differs')
            require(original_path.parent.parent.name == job_id(job), 'original attempt belongs to another job')
            expected_folder = absolute(previous['existing_manifest']).parent / 'runs' / protocol / job_id(job)
            require(original_path.parent.parent == expected_folder, 'original attempt is outside its manifest')
            state_path = original_path.parent / 'state.json'
            if entry in overlay['pending_first_requests']:
                self.pin(entry['original_state'], entry['original_state_sha256'])
                require(absolute(entry['original_state']) == state_path, 'pending state path differs')
                state = self.read(state_path)
                require(state['job'] == job and not state['calls'] and not original['calls']
                        and not state.get('executions'), 'pending request already has work')
                require(previous['status'] == 'held' and state['stage'] == 'hold', 'pending request is not held')
                require(entry['calls_verified_zero'] is True and entry['provider_calls_added'] == 0
                        and entry['status'] == 'pending_first_request_release', 'pending release differs')
                updates[key] = self.first_request_release(entry, previous, overlay, original, state)
                continue
            self.pin(entry['receipt'], entry['receipt_sha256'])
            receipt = self.read(entry['receipt'])
            for field in ('position_id', 'job', 'kind', 'original_record', 'original_record_sha256',
                          'recovered_record', 'recovered_record_sha256'):
                require(receipt[field] == entry[field], 'receipt differs: ' + field)
            require(receipt['provider_calls_added'] == 0 and receipt['original_protocol_sha256'] == protocol,
                    'receipt has another protocol or provider call')
            require(absolute(receipt['original_state']) == state_path, 'receipt state path differs')
            self.pin(state_path, receipt['original_state_sha256'])
            state = self.read(state_path)
            require(state['job'] == job, 'original state identity differs')
            require(receipt['source_hashes'].get(str(original_path)) == entry['original_record_sha256']
                    and receipt['source_hashes'].get(str(state_path)) == receipt['original_state_sha256'],
                    'original record or state missing from source hashes')
            for source, expected in receipt['source_hashes'].items():
                self.pin(source, expected)
            recovered_path, recovered = self.record(entry['recovered_record'], entry['recovered_record_sha256'])
            require(recovered['job'] == job and recovered['protocol_sha256'] == protocol, 'recovered identity differs')
            require(recovered['status'] in TERMINAL and recovered['stage'] == 'done', 'recovery is not terminal')
            require(len(original['calls']) == 1 and original['calls'][0].get('ok')
                    and not original['calls'][0].get('timed_out')
                    and recovered['calls'] == original['calls'], 'completed provider response changed')
            require(recovered['prompt'] == original['prompt'], 'effective prompt changed')
            require(recovered['recovery'] == dict(original_record=str(original_path),
                    original_record_sha256=entry['original_record_sha256'], kind=entry['kind'],
                    provider_calls_added=0, original_protocol_sha256=protocol), 'recovery provenance differs')
            require(recovered.get('protocol_findings') == receipt['protocol_findings'], 'protocol findings differ')
            if receipt['protocol_findings']:
                require(recovered.get('protocol_violation') is True
                        and recovered.get('access_review_status') == 'confirmed_external_environment_write'
                        and recovered['outcome']['protocol_status'] == 'confirmed_external_environment_write',
                        'protocol finding was lost')
            if entry['kind'] == 'local_recovery':
                require(previous['status'] == 'held' and state['stage'] == 'hold' and not state.get('executions'),
                        'local recovery requires the original pre-execution hold')
            elif entry['kind'] == 'environment_replay':
                require(previous['status'] == 'retained' and previous['record_path'] == str(original_path)
                        and state['stage'] == 'done', 'replay does not replace its original retained record')
                parity = receipt['parity']
                self.pin(parity['original_output'], parity['original_output_sha256'])
                self.pin(parity['recovered_output'], parity['recovered_output_sha256'])
                require(parity['exact_json_equal'] is True and parity['estimates_equal'] is True,
                        'replay differs and needs a separate review')
            else:
                require(False, 'unknown recovery kind')
            proof = self.prompt_proof(original_path, original, row)
            require(proof['prompt_compatibility'] == 'exact_effective_prompt', 'original prompt differs')
            updates[key] = dict(previous, status='retained', source='environment_local_recovery',
                record_path=str(recovered_path), record_sha256=entry['recovered_record_sha256'],
                record_status=recovered['status'], **proof, recovery_kind=entry['kind'],
                recovery_original_record=str(original_path), recovery_original_sha256=entry['original_record_sha256'],
                recovery_overlay=str(path), recovery_receipt=entry['receipt'], recovery_receipt_sha256=entry['receipt_sha256'],
                protocol_violation=recovered.get('protocol_violation', False),
                access_review_status=recovered.get('access_review_status'), protocol_findings=receipt['protocol_findings'])
            predecessors.append(dict(**row, source='environment_recovery_predecessor',
                record_path=str(original_path), record_sha256=entry['original_record_sha256'],
                disposition='preserved_recovery_predecessor', reason=entry['kind']))
        for prior in self.compatibility:
            if position_key(prior) in updates and prior.get('disposition') == 'selected':
                prior['disposition'] = 'preserved_recovery_predecessor'
        self.positions.update(updates)
        saved_predecessors = {row.get('record_path') for row in self.compatibility
                              if row.get('disposition') == 'preserved_recovery_predecessor'}
        self.compatibility.extend(row for row in predecessors if row['record_path'] not in saved_predecessors)
        self.compatibility.extend(dict(row, disposition='selected') for row in updates.values() if row['status'] == 'retained')

    def first_request_release(self, pending, previous, overlay, original, state):
        """Keep queued releases visible and select a completed first request once."""
        def require(condition, message):
            if not condition:
                raise ValueError('Invalid first-request release: ' + message)
        releases = overlay.get('first_request_releases', [])
        require(len(releases) <= 1, 'multiple releases')
        row = dict(previous, recovery_status='pending_first_request_release',
                   recovery_overlay=str(self.out/'work/environment_recovery/overlay.json'))
        if not releases:
            return row
        release = releases[0]
        require(release['position_id'] == pending['position_id'] and release['job'] == pending['job'], 'position differs')
        self.pin(release['plan'], release['plan_sha256'])
        plan = self.read(release['plan'])
        require(plan['entry'] == pending and plan['provider_call_limit'] == 1
                and plan['original_protocol_sha256'] == overlay['original_protocol_sha256'], 'plan differs')
        for source, expected in plan['source_hashes'].items():
            self.pin(source, expected)
        row.update(recovery_status=release['status'], recovery_plan=release['plan'],
                   recovery_plan_sha256=release['plan_sha256'])
        if 'receipt' not in release:
            require(release['status'] in ('queued_waiting_for_collection_lock', 'running_first_request',
                                         'paused_before_request', 'needs_review'), 'unknown incomplete status')
            if release['status'] == 'running_first_request':
                row['status'] = 'active'
            return row
        self.pin(release['receipt'], release['receipt_sha256'])
        receipt = self.read(release['receipt'])
        for field in ('position_id', 'job', 'original_record', 'original_record_sha256',
                      'original_state', 'original_state_sha256'):
            require(receipt[field] == pending[field], 'receipt differs: ' + field)
        require(receipt['original_protocol_sha256'] == overlay['original_protocol_sha256']
                and receipt['source_hashes'] == plan['source_hashes']
                and receipt['kind'] == 'first_request_release', 'receipt provenance differs')
        for field in ('recovered_record', 'recovered_record_sha256', 'provider_calls_added', 'status'):
            require(receipt[field] == release[field], 'release differs: ' + field)
        record_path, record = self.record(receipt['recovered_record'], receipt['recovered_record_sha256'])
        require(record_path == absolute(plan['destination'])/'run_record.json', 'derived record path differs')
        require(record['job'] == pending['job'] and record['protocol_sha256'] == overlay['original_protocol_sha256'],
                'derived identity differs')
        require(record['prompt'] == original['prompt'] == state['prompt'] == self.prompts[prompt_key(pending['job'])],
                'effective prompt differs')
        require(len(record['calls']) == receipt['provider_calls_added'] <= 1
                and record['status'] == receipt['status'] and record['stage'] == receipt['stage'], 'request count or status differs')
        require(record['recovery'] == dict(kind='first_request_release', original_record=pending['original_record'],
                original_record_sha256=pending['original_record_sha256'], original_protocol_sha256=overlay['original_protocol_sha256'],
                provider_calls_added=receipt['provider_calls_added'], dispatch_started=receipt['dispatch_started']),
                'derived recovery provenance differs')
        row.update(recovery_receipt=release['receipt'], recovery_receipt_sha256=release['receipt_sha256'],
                   recovery_original_record=pending['original_record'], recovery_original_sha256=pending['original_record_sha256'])
        if record['status'] not in TERMINAL:
            row.update(status='held', recovery_status='needs_review')
            return row
        require(record['stage'] == 'done' and len(record['calls']) == 1 and receipt['dispatch_started'] is True,
                'terminal request is incomplete')
        prompt_hash = hashlib.sha256(state['prompt'].encode()).hexdigest()
        row.update(status='retained', source='environment_first_request_release', record_path=str(record_path),
            record_sha256=receipt['recovered_record_sha256'], record_status=record['status'], recovery_kind=receipt['kind'],
            prompt_compatibility='exact_effective_prompt', saved_prompt_sha256=prompt_hash, current_prompt_sha256=prompt_hash,
            saved_prompt_path=pending['original_state'], protocol_violation=record.get('protocol_violation'),
            access_review_status=record.get('access_review_status'), protocol_findings=record.get('protocol_findings', []))
        return row

    def add_extended_historical(self, original):
        """Check saved repetitions five through nine against accepted same-cell context."""
        source = REPORTS / '2026-09-15_twelve-reps-extension/results/twelve_repetition_manifest.json'
        historical = self.read(source)
        by_cell = defaultdict(list)
        for entry in self.positions.values():
            if entry.get('record_path') and entry['global_rep'] < 5:
                by_cell[tuple(entry[k] for k in FIELDS)].append(entry)
        for raw in historical['runs']:
            row = normalized(raw)
            if row['task'] not in original or not 5 <= row['global_rep'] < 10:
                continue
            path = absolute(raw['run_dir']) / 'run_record.json'
            expected = raw['file_hashes'].get(str(path.relative_to(ROOT)))
            assert expected, str(path)
            path, record = self.record(path, expected)
            reason = None
            if row['condition'] == 'agent':
                reason = 'superseded_agent_instruction_without_statistical_verification'
            elif row['model'] == 'sonnet' and row['platform'] in ('python', 'r'):
                reason = 'superseded_sonnet_python_r_execution_context_replaced_by_fresh_checkpoint'
            proof = self.prompt_proof(path, record, row, allow_admin=True)
            if not reason and proof['prompt_compatibility'] == 'effective_prompt_mismatch_or_unverified':
                reason = 'superseded_prompt_or_example_version'
            data_path = path.parent / 'input.csv'
            if not reason:
                self.pin(data_path, file_hash(absolute(self.catalog[row['task']]['data_path'])))
                assert record['model'] == ('gpt-5.6-luna' if row['model'] == 'luna' else 'claude-sonnet-4-6')
                assert record['reasoning_effort'] == ('low' if row['model'] == 'luna' else 'medium')
                fields = ('model', 'reasoning_effort', 'exec_mode', 'prompt_mode',
                          'codex_timeout_seconds', 'output_contract', 'pilot_plan')
                cell = tuple(row[k] for k in FIELDS)
                baselines = by_cell.get(cell, [])
                matches = []
                for baseline in baselines:
                    _, previous = self.record(baseline['record_path'], baseline['record_sha256'])
                    command_match = True
                    if row['platform'] == 'stata':
                        current_commands = self.command_signatures(path)
                        baseline_commands = self.command_signatures(Path(baseline['record_path']))
                        command_match = bool(current_commands) and current_commands == baseline_commands
                    if all(record.get(k) == previous.get(k) for k in fields) and command_match:
                        matches.append(baseline['record_path'])
                if not matches:
                    reason = 'no_matching_accepted_provider_context'
            if reason:
                self.compatibility.append(dict(**row, record_path=str(path), record_sha256=expected,
                    disposition='historical_only', reason=reason, source='historical_repetitions5to9', **proof))
            elif position_key(row) in self.positions:
                self.compatibility.append(dict(**row, record_path=str(path), record_sha256=expected,
                    disposition='compatible_unselected_reserved_position',
                    reason='existing_authorized_continuation_owns_position',
                    source='historical_repetitions5to9', **proof))
            else:
                self.add_record(row, path, 'verified_historical_extension', expected, allow_admin=True)
                self.positions[position_key(row)]['accepted_context_record'] = matches[0]
                self.positions[position_key(row)]['recorded_concurrency'] = record.get('configured_limits')

    def command_signatures(self, record_path):
        signatures = set()
        for path in record_path.parent.glob('command_attempt*.json'):
            command = self.read(path)
            argv = list(command['argv'])
            for flag in ('-C', '-o'):
                if flag in argv:
                    argv[argv.index(flag) + 1] = '<per-request-path>'
            signatures.add(digest(dict(argv=argv, timeout_seconds=command.get('timeout_seconds'))))
        return signatures

    def add_historical_exclusions(self):
        path = REPORTS / '2026-09-17_runtime-verification/work/selection/earlier_agent_records.csv'
        for row in self.csv(path):
            self.pin(row['record_path'], row['record_sha256'])
            self.compatibility.append(dict(**normalized(row), record_path=str(absolute(row['record_path'])),
                record_sha256=row['record_sha256'], disposition='historical_only',
                reason='superseded_agent_instruction_without_statistical_verification', source='earlier_agent_records'))
        # These records document the previous provider context and example versions.
        path = REPORTS / '2026-09-16_final-design-five-reps/work/tasks/first_five_candidates.json'
        for row in self.read(path):
            reason = None
            if not row['candidate_for_selected_prompt']:
                reason = 'superseded_example_version'
            elif row['origin'] == 'old' and row['model'] == 'sonnet' and row['platform'] in ('python', 'r'):
                reason = 'superseded_sonnet_python_r_execution_context_replaced_by_fresh_checkpoint'
            if reason:
                self.pin(row['record_path'], row['record_sha256'])
                self.compatibility.append(dict(**normalized(row), record_path=str(absolute(row['record_path'])),
                    record_sha256=row['record_sha256'], disposition='historical_only', reason=reason,
                    source='prior_design_candidates'))
        # Historical recovery files preserve the same request. They are not extra repetitions.
        selected_paths = {entry.get('record_path') for entry in self.positions.values()}
        for path in sorted((LIVE / 'checkpoint').glob('*/manifest.json')):
            manifest, records = self.manifest_records(path)
            for job in manifest['jobs']:
                folder = path.parent / 'runs' / manifest['protocol_sha256'] / job_id(job)
                for record_path in folder.glob('attempt_*/run_record.json'):
                    if str(record_path) in selected_paths:
                        continue
                    _, record = self.record(record_path)
                    self.compatibility.append(dict(**normalized(job), record_path=str(record_path),
                        record_sha256=self.sources[str(record_path)], disposition='held_historical',
                        reason='recovery_predecessor_preserved_separately', record_status=record.get('status'),
                        source='checkpoint_recovery'))

    def save(self, name, value):
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    def write_csv(self, name, rows):
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k,v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--prompt-paths', type=Path)
    args = parser.parse_args()
    if bool(args.catalog) != bool(args.prompt_paths):
        parser.error('Supply both --catalog and --prompt-paths, or neither.')
    result = Inventory(args.out, args.catalog, args.prompt_paths).build()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

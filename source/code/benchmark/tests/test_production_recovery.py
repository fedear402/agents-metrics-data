"""Offline checks for recovery without repeated provider responses."""
import copy
import json
from pathlib import Path

import pytest

from benchmark import production_recovery as recovery
from test_prospective_collect import fixture, p


def test_provider_callback_always_rejects():
    with pytest.raises(RuntimeError, match='Provider calls are forbidden'):
        recovery.reject_provider_call('prompt')


@pytest.mark.parametrize('script,platform', [
    ('import subprocess\nsubprocess.run(["uv","pip","install","lifelines"])', 'python'),
    ('import pandas as pd\npd.read_csv("/etc/passwd")', 'python'),
    ('shell uv pip install lifelines', 'stata'),
    ('python: print(1)', 'stata'),
])
def test_unreviewed_external_actions_are_rejected(script, platform):
    with pytest.raises(ValueError):
        recovery.review_script(script, platform)


def test_recovery_reuses_call_and_preserves_original(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'p', p)
    p.STOP.clear()
    job, manifest, runtime, quota, call, execute, calls, _ = fixture(tmp_path, 'agent')
    def stop_before_execution(*args):
        raise RuntimeError('Collection stop requested before subprocess start')
    def provider(prompt, cwd, *args):
        (cwd/'run.py').write_text('pass\n')
        return call(prompt, cwd, *args)
    original = p.run_job(job, manifest, tmp_path/'original', runtime, quota, provider, stop_before_execution)
    assert len(calls) == 1 and original['stage'] == 'hold'
    source = Path(original['attempt_directory'])/'run_record.json'
    before = {str(f): f.read_bytes() for f in source.parent.rglob('*') if f.is_file()}
    entry = recovery.make_entry(source, 'toy_position', 'local_recovery', manifest)
    p.STOP.clear()
    result = recovery.recover_entry(entry, manifest, tmp_path/'recovery', runtime, execute=execute)
    assert result['provider_calls_added'] == 0 and len(calls) == 1
    derived = json.loads(Path(result['recovered_record']).read_text())
    assert derived['calls'] == original['calls']
    assert derived['scoring']['numeric_success'] is True
    assert all(Path(name).read_bytes() == content for name, content in before.items())


def test_zero_call_hold_is_never_locally_replayed(tmp_path):
    source = tmp_path/'run_record.json'
    source.write_text(json.dumps({'calls': [], 'job': {}}))
    with pytest.raises(ValueError, match='completed provider response'):
        recovery.make_entry(source, 'zero_call', 'local_recovery', {})


def test_changed_pinned_submission_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'p', p)
    p.STOP.clear()
    job, manifest, runtime, quota, call, execute, calls, _ = fixture(tmp_path, 'agent')
    def provider(prompt, cwd, *args):
        (cwd/'run.py').write_text('pass\n')
        return call(prompt, cwd, *args)
    original = p.run_job(job, manifest, tmp_path/'original', runtime, quota, provider,
        lambda *args: (_ for _ in ()).throw(RuntimeError('Collection stop requested before subprocess start')))
    entry = recovery.make_entry(Path(original['attempt_directory'])/'run_record.json', 'toy', 'local_recovery', manifest)
    Path(entry['script']).write_text('print("changed")')
    with pytest.raises(ValueError, match='Source changed'):
        recovery.recover_entry(entry, manifest, tmp_path/'recovery', runtime, execute=execute)


def first_request_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'p', p)
    p.STOP.clear()
    job, manifest, runtime, quota, call, execute, calls, _ = fixture(tmp_path, 'agent')
    destination = tmp_path/'original'/p.job_id(job)/'attempt_0001'
    destination.mkdir(parents=True)
    cwd = tmp_path/'workspace'
    cwd.mkdir()
    (cwd/'input.csv').write_bytes(Path(manifest['protocol']['tasks']['toy']['data_path']).read_bytes())
    state = dict(job=job, protocol_sha256=manifest['protocol_sha256'], started_at=p.now(),
        stage='hold', status='needs_review', round=1, calls=[], executions=[], access_findings=[],
        cwd=str(cwd), prompt=manifest['protocol']['prompts'][p.prompt_key(job)],
        attempt_directory=str(destination), scoring=dict(success=None, numeric_success=None, integrity_pass=None),
        driver_error="ValueError('Package changed: pandas')")
    p.snapshot_state(destination, state)
    p.save(destination/'run_record.json', runtime.m.adapt_record(state, destination))
    entry = dict(position_id='toy_zero_call', job=job, original_record=str(destination/'run_record.json'),
        original_record_sha256=p.sha(destination/'run_record.json'), original_state=str(destination/'state.json'),
        original_state_sha256=p.sha(destination/'state.json'), calls_verified_zero=True,
        status='pending_first_request_release', provider_calls_added=0)
    return entry, job, manifest, runtime, quota, call, execute, calls


def test_original_zero_call_hold_requires_review(tmp_path, monkeypatch):
    entry, job, manifest, runtime, quota, call, execute, calls = first_request_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='hold requires review'):
        p.run_job(job, manifest, tmp_path/'original', runtime, quota, call, execute)
    assert calls == []


def test_first_request_preserves_original_and_dispatches_once(tmp_path, monkeypatch):
    entry, job, manifest, runtime, quota, call, execute, calls = first_request_fixture(tmp_path, monkeypatch)
    plan = recovery.prepare_first_request(entry, manifest, tmp_path/'recovery')
    def provider(prompt, cwd, *args):
        (cwd/'run.py').write_text('pass\n')
        return call(prompt, cwd, *args)
    receipt = recovery.execute_first_request(plan, manifest, runtime, quota, provider, execute)
    again = recovery.execute_first_request(plan, manifest, runtime, quota, provider, execute)
    assert receipt == again and len(calls) == 1
    assert calls == [manifest['protocol']['prompts'][p.prompt_key(job)]]
    assert receipt['provider_calls_added'] == 1 and receipt['status'] == 'completed'
    recovery.verify_hashes(plan['source_hashes'])


def test_interrupted_first_request_cannot_dispatch_again(tmp_path, monkeypatch):
    entry, job, manifest, runtime, quota, call, execute, calls = first_request_fixture(tmp_path, monkeypatch)
    plan = recovery.prepare_first_request(entry, manifest, tmp_path/'recovery')
    dispatched = []
    def interrupted(*args):
        dispatched.append(True)
        raise RuntimeError('Provider dispatch interrupted')
    result = recovery.execute_first_request(plan, manifest, runtime, quota, interrupted, execute)
    assert result['status'] == 'needs_review'
    recovery.execute_first_request(plan, manifest, runtime, quota, interrupted, execute)
    assert len(dispatched) == 1


def test_first_request_rejects_prior_provider_work(tmp_path, monkeypatch):
    entry, job, manifest, runtime, quota, call, execute, calls = first_request_fixture(tmp_path, monkeypatch)
    path = Path(entry['original_record'])
    record = json.loads(path.read_text())
    record['calls'] = [dict(ok=True)]
    path.write_text(json.dumps(record))
    entry['original_record_sha256'] = p.sha(path)
    with pytest.raises(ValueError, match='zero-call'):
        recovery.prepare_first_request(entry, manifest, tmp_path/'recovery')

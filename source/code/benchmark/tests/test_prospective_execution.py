"""Provider timeout and interrupted local replay regression checks."""
from pathlib import Path
import shutil

import pytest

from test_prospective_collect import fixture, p


def test_provider_timeout_stops_self_repair_requests(tmp_path):
    p.STOP.clear()
    job,manifest,runtime,q,_,execute,_,_ = fixture(tmp_path, 'self_repair')
    calls=[]
    def timeout(*args):
        calls.append(True)
        return dict(ok=False,text='',num_tool_calls=0,returncode=-9,timed_out=True)
    record=p.run_job(job,manifest,tmp_path/'runs',runtime,q,timeout,execute)
    assert record['status']=='provider_timeout' and len(calls)==1
    assert len(record['calls'])==1


def test_replayed_local_interruption_allows_concurrent_queue_to_finish(tmp_path):
    p.STOP.clear()
    real=p.load_runtime(tmp_path)
    job,manifest,runtime,q,call,execute,calls,_=fixture(tmp_path)
    runtime.m=real.m
    runtime.BASE_ADAPT=real.BASE_ADAPT
    runtime.c.call_model=call
    runtime.c.prior.execute=execute
    interrupted=p.run_job(job,manifest,tmp_path/'runs',runtime,q,call,
                          lambda *args:dict(ok=False,returncode=-9,timed_out=False))
    assert interrupted['outcome']['primary_success'] is None
    p.STOP.clear()
    next_job=dict(job,rep=1)
    manifest['jobs'].append(next_job)
    p.collect(manifest['jobs'],manifest,tmp_path/'runs',runtime,2)
    assert len(calls)==2 and not p.STOP.is_set()
    assert p.status(manifest,tmp_path/'runs')['completed']==2
    resumed=p.completed(tmp_path/'runs'/p.job_id(job),manifest['protocol_sha256'])
    assert resumed['executions'][0]['returncode']==-9
    assert resumed['executions'][-1]['returncode']==0
    assert resumed['outcome']['primary_success'] is True


def test_adapter_never_clears_a_concurrent_user_pause(tmp_path):
    p.STOP.clear()
    job,manifest,runtime,q,call,execute,_,_=fixture(tmp_path)
    original=runtime.m.adapt_record
    def adapter(state,destination):
        p.STOP.set()
        return original(state,destination)
    runtime.BASE_ADAPT=adapter
    try:
        record=p.run_job(job,manifest,tmp_path/'runs',runtime,q,call,execute)
        assert record['outcome']['primary_success'] is True
        assert p.STOP.is_set()
    finally:
        p.STOP.clear()


def test_source_verification_failure_releases_probe_and_retains_rejection(tmp_path, monkeypatch):
    p.STOP.clear()
    job,manifest,runtime,q,call,execute,calls,_=fixture(tmp_path)
    manifest['protocol']['source_hashes']={}
    q.reject('luna',dict(reset_at=104,observed_at=100,status='rejected'))
    def fail(manifest):
        raise ValueError('Pinned source changed')
    monkeypatch.setattr(p,'verify',fail)
    try:
        record=p.run_job(job,manifest,tmp_path/'runs',runtime,q,call,execute)
        assert record['status']=='needs_review' and not calls
        assert q.state['luna']['reset_at']==104
        assert not q.probes
    finally:
        p.STOP.clear()


@pytest.mark.parametrize('stage',['evaluate','request'])
def test_deleted_temporary_workspace_restores_saved_snapshot(tmp_path,stage):
    p.STOP.clear()
    job,manifest,runtime,q,call,execute,calls,_=fixture(tmp_path)
    if stage=='request':
        p.STOP.set()
    row=p.run_job(job,manifest,tmp_path/'runs',runtime,q,call,
                  lambda *args:dict(ok=False,returncode=-9,timed_out=False))
    assert row['stage']==stage
    cwd=Path(row['cwd'])
    saved=Path(row['held_workspace'])
    original=(saved/'input.csv').read_bytes()
    shutil.rmtree(cwd)
    p.STOP.clear()
    row=p.run_job(job,manifest,tmp_path/'runs',runtime,q,call,execute)
    assert row['outcome']['primary_success'] and len(calls)==1
    assert Path(row['cwd'])==cwd and cwd.is_dir()
    assert (saved/'input.csv').read_bytes()==original


@pytest.mark.parametrize('defect',['symlink','changed_input','unsafe_stage'])
def test_workspace_restore_rejects_untrusted_snapshot_or_stage(tmp_path,defect):
    p.STOP.clear()
    job,manifest,runtime,q,call,execute,_,_=fixture(tmp_path)
    p.STOP.set()
    row=p.run_job(job,manifest,tmp_path/'runs',runtime,q,call,execute)
    p.STOP.clear()
    cwd=Path(row['cwd'])
    saved=Path(row['held_workspace'])
    destination=Path(row['attempt_directory'])
    if defect=='symlink':
        (saved/'outside').symlink_to(tmp_path/'data.csv')
    elif defect=='changed_input':
        (saved/'input.csv').write_text('changed')
    else:
        row['stage']='provider_in_flight'
    shutil.rmtree(cwd)
    with pytest.raises(ValueError):
        p.restore_workspace(destination,row,tmp_path/'data.csv')
    assert not cwd.exists()


def test_workspace_restore_uses_latest_completed_saved_snapshot(tmp_path):
    p.STOP.clear()
    job,manifest,runtime,q,call,execute,_,_=fixture(tmp_path)
    row=p.run_job(job,manifest,tmp_path/'runs',runtime,q,call,
                  lambda *args:dict(ok=False,returncode=-9,timed_out=False))
    p.STOP.clear()
    cwd=Path(row['cwd'])
    (Path(row['submitted_workspace'])/'marker').write_text('submitted')
    (Path(row['held_workspace'])/'marker').write_text('held')
    shutil.rmtree(cwd)
    restored=p.restore_workspace(Path(row['attempt_directory']),row,tmp_path/'data.csv')
    assert (restored/'marker').read_text()=='held'

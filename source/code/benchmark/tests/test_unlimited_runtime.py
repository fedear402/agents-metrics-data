"""Regression checks for provider and submitted-script waits without a cap."""
from pathlib import Path
import importlib.util
import json
import sys
import signal
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

SOURCE=Path(__file__).resolve().parents[1]
def load_runner():
    path=SOURCE/'prospective_collect_unlimited.py'
    if not path.exists():path=SOURCE/'prospective_collect.py'
    spec=importlib.util.spec_from_file_location('unlimited_test_runner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return getattr(module,'central',module),module

@pytest.mark.parametrize('model',['luna','sonnet'])
@pytest.mark.parametrize('condition',['chatbot','self_repair','agent'])
def test_provider_commands_have_no_wall_limit(tmp_path,model,condition):
    central,_=load_runner(); runtime=central.load_runtime(tmp_path)
    argv,limit=runtime.c.command(dict(model=model,condition=condition),tmp_path,tmp_path/'last')
    assert argv and '--max-turns' not in argv
    assert limit is None

@pytest.mark.parametrize('platform',['python','r','stata'])
def test_submitted_script_reaches_communicate_without_limit(tmp_path,monkeypatch,platform):
    central,_=load_runner();central.STOP.clear();runtime=central.load_runtime(tmp_path)
    seen=[]
    class Process:
        pid=987654321
        returncode=0
        def __init__(self,*args,**kwargs):pass
        def communicate(self,text=None,timeout=None):seen.append(timeout)
    monkeypatch.setattr(runtime.c.prior.subprocess,'Popen',Process)
    script=tmp_path/('run.'+{'python':'py','r':'R','stata':'do'}[platform]);script.write_text('')
    runtime.c.prior.execute(script,platform,tmp_path,tmp_path,1)
    assert seen==[None]


@pytest.mark.parametrize('model',['luna','sonnet'])
@pytest.mark.parametrize('condition',['chatbot','self_repair','agent'])
def test_provider_call_reaches_communicate_and_saves_null(tmp_path,monkeypatch,model,condition):
    central,_=load_runner();central.STOP.clear();runtime=central.load_runtime(tmp_path)
    seen=[]
    class Process:
        pid=987654321
        returncode=0
        def __init__(self,*args,**kwargs):pass
        def communicate(self,text=None,timeout=None):seen.append(timeout)
    monkeypatch.setattr(runtime.c.prior.subprocess,'Popen',Process)
    monkeypatch.setattr(runtime.c,'parse_call',lambda *args:dict(text='done',ok=True,timed_out=False))
    runtime.c.call_model('task',tmp_path,dict(model=model,condition=condition),tmp_path,1)
    assert seen==[None]
    receipt=json.loads((tmp_path/'command_attempt1.json').read_text())
    assert receipt['timeout_seconds'] is None


def test_real_process_outlives_short_legacy_deadline(tmp_path):
    central,_=load_runner();central.STOP.clear();runtime=central.load_runtime(tmp_path)
    result=runtime.c.prior.subprocess_capture([sys.executable,'-c','import time;time.sleep(0.05);print("complete")'],tmp_path,tmp_path,'real',0.001)
    assert result['stdout'].strip()=='complete'
    assert result['returncode']==0 and result['timed_out'] is False
    assert result['wall_seconds']>0.001 and result['timeout_seconds'] is None


def test_administrative_cancellation_still_stops_real_process(tmp_path):
    central,_=load_runner();central.STOP.clear();runtime=central.load_runtime(tmp_path)
    marker=tmp_path/'ready'
    code='from pathlib import Path;import time;Path("ready").write_text("yes");time.sleep(60)'
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(runtime.c.prior.subprocess_capture,[sys.executable,'-c',code],tmp_path,tmp_path,'cancel',None)
        end=time.monotonic()+2
        while not marker.exists() and time.monotonic()<end:time.sleep(0.005)
        try:
            assert marker.exists()
        finally:
            runtime.c.prior.stop_children(signal.SIGTERM,None)
        result=future.result(timeout=2)
    central.STOP.clear()
    assert result['returncode']<0 and result['timed_out'] is False
    assert not runtime.c.prior.CHILDREN


def test_historical_loader_keeps_finite_defaults(tmp_path):
    _,adapter=load_runner()
    runtime=adapter.BASE_LOAD_RUNTIME(tmp_path)
    assert runtime.c.command(dict(model='sonnet',condition='agent'),tmp_path,tmp_path/'last')[1]==600


def test_manifest_records_nulls_and_prepare_is_repeatable(tmp_path,monkeypatch):
    central,_=load_runner();central.STOP.clear()
    monkeypatch.setattr(central,'provider_cli_versions',lambda *args:{})
    data=tmp_path/'input.csv';data.write_text('x\n1\n')
    task=dict(task_id='toy',title='Toy',prompt='Return x.',estimands=['x'],data_path=str(data),data_description='x',reference={'estimates':[dict(term='x',estimate=1,se=None,n_obs=1)]})
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps({'toy':task}))
    decision=tmp_path/'decision.md';decision.write_text('Authorized test.')
    prompt=tmp_path/'prompt.txt';prompt.write_text('Return x.')
    job=dict(block='test',task='toy',model='sonnet',platform='python',prompt='zero_shot',condition='chatbot',output_format='json',rep=0)
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(catalog_path=str(catalog),prompt_paths={central.prompt_key(job):str(prompt)},jobs=[job],decision_record=str(decision))))
    path=tmp_path/'manifest.json';manifest=central.prepare(config,path)
    policy=manifest['protocol']['policy']
    assert policy['provider_wall_timeout_seconds'] is None
    assert policy['session_wall_timeout_seconds'] is None
    assert policy['submitted_script_wall_timeout_seconds'] is None
    assert manifest['protocol']['commands']['sonnet/chatbot'][1] is None
    assert central.prepare(config,path)['protocol_sha256']==manifest['protocol_sha256']
    central.verify(manifest,central.load_runtime(tmp_path))

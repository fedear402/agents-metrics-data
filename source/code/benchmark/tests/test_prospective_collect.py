"""Offline regression checks. Fake providers never open a network connection."""
from pathlib import Path
from types import SimpleNamespace
import copy
import importlib.util
import json
import shutil
import threading
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'prospective_collect.py'
spec = importlib.util.spec_from_file_location('prospective_collect_tested', SOURCE)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


@pytest.fixture(autouse=True)
def clear_stop():
    p.STOP.clear()
    yield
    p.STOP.clear()


class Clock:
    def __init__(self): self.value = 100.0
    def __call__(self): return self.value
    def sleep(self, seconds): self.value += seconds


def event(status='rejected', reset=104):
    return dict(type='rate_limit_event', rate_limit_info=dict(status=status, resetsAt=reset,
        overageStatus='rejected', rateLimitType='five_hour'))


def test_warning_and_allowed_overage_are_not_hard_limits():
    assert p.parse_quota([event('allowed_warning')]) is None
    assert p.parse_quota([event('allowed')]) is None
    assert p.parse_quota([event()], 100)['reset_at'] == 104
    assert p.parse_quota([dict(rate_limit_info=dict(status='rejected', resets_at=111))], 100)['reset_at'] == 111


def test_wait_reset_resume_and_restart(tmp_path):
    clock = Clock()
    path = tmp_path / 'quota.json'
    gate = p.QuotaCoordinator(path, clock=clock, sleep=clock.sleep)
    gate.reject('sonnet', p.parse_quota([event()], clock()))
    resumed = p.QuotaCoordinator(path, clock=clock, sleep=clock.sleep)
    assert resumed.acquire('luna') is False
    assert resumed.acquire('sonnet') is True
    assert clock() == 104
    resumed.release('sonnet', True)
    assert resumed.acquire('sonnet') is False


def test_unknown_reset_and_expired_at_receipt_never_guess(tmp_path):
    for reset in (None, 99):
        gate = p.QuotaCoordinator(tmp_path / str(reset), clock=lambda: 100)
        gate.reject('sonnet', p.parse_quota([event(reset=reset)], 100))
        with pytest.raises(p.QuotaUnknown): gate.acquire('sonnet')


def test_only_one_recovery_probe_and_user_pause(tmp_path):
    clock = Clock()
    stop = threading.Event()
    gate = p.QuotaCoordinator(tmp_path/'quota.json', stop, clock, clock.sleep)
    gate.reject('sonnet', p.parse_quota([event()], 100))
    assert gate.acquire('sonnet') is True
    # A second waiter cannot pass until the recovery request returns.
    gate.sleep = lambda seconds: stop.set()
    with pytest.raises(p.Paused): gate.acquire('sonnet')
    gate.release('sonnet', True)


def fixture(tmp_path, agency='chatbot'):
    data = tmp_path/'data.csv'; data.write_text('x\n1\n')
    job = dict(block='test',task='toy',model='luna',platform='python',prompt='zero',condition=agency,output_format='json',rep=0)
    task = dict(task_id='toy',data_path=str(data),reference={'estimates':[dict(term='b',estimate=123456.7,se=None,n_obs=1)]})
    manifest = dict(protocol_sha256='testhash',jobs=[job],protocol=dict(tasks={'toy':task},prompts={p.prompt_key(job):'BASE EXACT PROMPT'}))
    def extract(text):
        return text.split('```python\n',1)[1].split('```',1)[0] if '```python\n' in text else None
    def load(cwd, spec, job, execution):
        result = json.loads((cwd/'result.json').read_text()) if (cwd/'result.json').exists() else {}
        return result, 'json', []
    c = SimpleNamespace(
        harness=SimpleNamespace(extract_code=extract,script_name=lambda platform:'run.py'),
        sp=SimpleNamespace(find_script=lambda cwd, platform: cwd/'run.py' if (cwd/'run.py').exists() else None,
                           score_integrity=lambda *args:(True,[])),
        base=SimpleNamespace(as_task=lambda spec:spec),
        prior=SimpleNamespace(snapshot=lambda src,dst:shutil.copytree(src,dst)),
        access_findings=lambda text,cwd:[],tool_violations=lambda *args:[],load_output=load,
        compare_numeric=lambda values,spec:(values.get('b',{}).get('estimate')==123456.7,{},[]))
    def adapt(state, destination):
        record=copy.deepcopy(state)
        record['outcome']=dict(primary_success=state['scoring']['success'],numerical_success=state['scoring']['numeric_success'])
        return record
    runtime=SimpleNamespace(c=c,m=SimpleNamespace(adapt_record=adapt))
    clock=Clock(); gate=p.QuotaCoordinator(tmp_path/'quota.json',clock=clock,sleep=clock.sleep)
    calls=[]; executions=[]
    def provider(text,cwd,job,destination,number):
        calls.append(text)
        return dict(ok=True,text='```python\npass\n```',num_tool_calls=0,returncode=0,usage={'input_tokens':1,'output_tokens':1})
    def execute(script,platform,cwd,destination,number):
        executions.append(number)
        (cwd/'result.json').write_text(json.dumps({'b':dict(estimate=123456.7,n_obs=1)}))
        return dict(ok=True,returncode=0,stdout='',stderr='')
    return job,manifest,runtime,gate,provider,execute,calls,executions


def test_completed_slots_never_call_again(tmp_path):
    job,m,r,q,call,execute,calls,executions=fixture(tmp_path)
    a=p.run_job(job,m,tmp_path/'runs',r,q,call,execute)
    b=p.run_job(job,m,tmp_path/'runs',r,q,lambda *args:pytest.fail('duplicate provider call'),execute)
    assert a['outcome']['primary_success'] and b==a and len(calls)==1
    assert executions==[1]


def test_access_review_is_deferred_until_after_collection(tmp_path):
    job,m,r,q,call,execute,calls,executions=fixture(tmp_path)
    def forbidden_inline_scan(*args):
        pytest.fail('Access review must use saved records after collection')
    r.c.tool_violations = forbidden_inline_scan
    r.c.access_findings = forbidden_inline_scan
    def data_only_integrity(task, cwd, script, references):
        assert script is None, 'Source access and reference-literal checks must run later'
        return True, []
    r.c.sp.score_integrity = data_only_integrity
    row=p.run_job(job,m,tmp_path/'runs',r,q,call,execute)
    assert row['stage']=='done' and row['outcome']['primary_success'] is True
    assert row['access_review_status']=='pending_post_collection'
    assert row['outcome']['protocol_status']=='pending_post_collection'
    assert row['outcome']['data_script_integrity'] is None
    assert row['outcome']['input_data_integrity'] is True
    assert len(calls)==1 and executions==[1]
    assert not p.STOP.is_set()


def test_input_integrity_check_still_applies_with_deferred_access_review(tmp_path):
    job,m,r,q,call,execute,calls,executions=fixture(tmp_path)
    def alter_input(script,platform,cwd,destination,number):
        result=execute(script,platform,cwd,destination,number)
        (cwd/'input.csv').write_text('x\n999\n')
        return result
    row=p.run_job(job,m,tmp_path/'runs',r,q,call,alter_input)
    assert row['outcome']['primary_success'] is None
    assert row['hold_reason']=='integrity'


def test_model_delivery_failure_is_terminal(tmp_path):
    job,m,r,q,_,execute,_,_=fixture(tmp_path)
    call=lambda *args:dict(ok=True,text='No code.',num_tool_calls=0,returncode=0)
    row=p.run_job(job,m,tmp_path/'runs',r,q,call,execute)
    assert row['status']=='model_failure' and row['outcome']['primary_success'] is False
    p.run_job(job,m,tmp_path/'runs',r,q,lambda *args:pytest.fail('model error retried'),execute)


def test_self_repair_prefix_native_feedback_no_reference_leak(tmp_path):
    job,m,r,q,call,execute,calls,executions=fixture(tmp_path,'self_repair')
    def fail_first(script,platform,cwd,destination,number):
        if number==1:
            executions.append(number)
            return dict(ok=False,returncode=1,stderr='NameError: x',stdout='')
        return execute(script,platform,cwd,destination,number)
    row=p.run_job(job,m,tmp_path/'runs',r,q,call,fail_first)
    assert row['outcome']['primary_success'] and executions==[1,2]
    assert calls[0]=='BASE EXACT PROMPT' and calls[1].startswith(calls[0])
    assert 'NameError: x' in calls[1] and '123456.7' not in calls[1]
    destination=Path(row['attempt_directory'])
    assert (destination/'execution_workspace0001/run.py').exists()
    assert (destination/'execution_workspace0002/result.json').exists()


def test_wrong_finite_values_do_not_trigger_gold_feedback(tmp_path):
    job,m,r,q,call,_,calls,_=fixture(tmp_path,'self_repair')
    def wrong(script,platform,cwd,destination,number):
        (cwd/'result.json').write_text(json.dumps({'b':dict(estimate=0,n_obs=1)}))
        return dict(ok=True,returncode=0)
    row=p.run_job(job,m,tmp_path/'runs',r,q,call,wrong)
    assert len(calls)==1 and row['outcome']['primary_success'] is False


def test_four_attempts_are_not_provider_turn_limit(tmp_path):
    job,m,r,q,call,_,calls,_=fixture(tmp_path,'self_repair')
    row=p.run_job(job,m,tmp_path/'runs',r,q,call,lambda *args:dict(ok=False,returncode=1,stderr='syntax error'))
    assert len(calls)==4 and row['outcome']['primary_success'] is False


def test_quota_request_preserved_and_resumed_without_new_slot(tmp_path):
    job,m,r,q,call,execute,calls,_=fixture(tmp_path)
    def first_rejected(text,cwd,job,destination,number):
        if number==1:
            (destination/'quota.stdout').write_text(json.dumps(event())+'\n')
            return dict(ok=False,text='',num_tool_calls=0,returncode=1,raw_stream_file='quota.stdout',provider_limit=True)
        return call(text,cwd,job,destination,number)
    row=p.run_job(job,m,tmp_path/'runs',r,q,first_rejected,execute)
    assert row['outcome']['primary_success'] and len(row['calls'])==2 and len(calls)==1
    assert len(list((tmp_path/'runs'/p.job_id(job)).glob('attempt_*')))==1


def test_interrupted_provider_is_null_hold_never_automatic_retry(tmp_path):
    job,m,r,q,_,execute,_,_=fixture(tmp_path)
    row=p.run_job(job,m,tmp_path/'runs',r,q,lambda *args:dict(ok=False,text='',returncode=-9,timed_out=False),execute)
    assert row['outcome']['primary_success'] is None
    assert list(Path(row['attempt_directory']).glob('held_workspace_*/input.csv'))
    p.STOP.clear()
    with pytest.raises(ValueError,match='requires review'):
        p.run_job(job,m,tmp_path/'runs',r,q,lambda *args:pytest.fail('cancelled request retried'),execute)


def test_completed_response_local_pause_replays_only_locally(tmp_path):
    job,m,r,q,call,execute,calls,_=fixture(tmp_path)
    row=p.run_job(job,m,tmp_path/'runs',r,q,call,lambda *args:dict(ok=False,returncode=-9,timed_out=False))
    assert row['outcome']['primary_success'] is None and len(calls)==1
    p.STOP.clear()
    row=p.run_job(job,m,tmp_path/'runs',r,q,lambda *args:pytest.fail('response repeated'),execute)
    assert row['outcome']['primary_success']


def test_duplicate_jobs_rejected(tmp_path):
    job,m,*_=fixture(tmp_path)
    with pytest.raises(ValueError,match='Duplicate'):
        p.validate_jobs([job,dict(job)],m['protocol'])


def test_real_runtime_commands_have_no_turn_caps(tmp_path):
    runtime=p.load_runtime(tmp_path)
    for model in ('luna','sonnet'):
        for condition in ('chatbot','self_repair','agent'):
            argv,timeout=runtime.c.command(dict(model=model,condition=condition),Path('<cwd>'),Path('<last>'))
            assert '--max-turns' not in argv
            assert timeout==({'luna':(900,1500),'sonnet':(180,600)}[model][condition=='agent'])


def test_agent_verification_replaces_completion_rule_without_changing_task():
    prefix = 'Example: toy values.\nTask: estimate the requested statistic.\n'
    ending = '\nUse only input.csv. Do not read reference answers.'
    original = prefix + 'When the requested output is present, finish with DONE.' + ending
    revised = p.prepared_prompt(original, 'agent')
    assert revised == prefix + p.AGENT_VERIFICATION_INSTRUCTION + ending
    assert 'When the requested output is present' not in revised
    assert p.prepared_prompt(revised, 'agent') == revised
    for condition in ('chatbot', 'self_repair'):
        assert p.prepared_prompt(original, condition) == original


@pytest.mark.parametrize('text', ['Task only.',
    'When the requested output is present, finish with DONE.\n'
    'When the requested output is present, finish with DONE.'])
def test_agent_preparation_requires_one_completion_anchor(text):
    with pytest.raises(ValueError, match='completion instruction'):
        p.prepared_prompt(text, 'agent')


def test_five_repetition_limit_is_enforced(tmp_path):
    job, manifest, *_ = fixture(tmp_path)
    job['rep'] = 5
    with pytest.raises(ValueError, match='repetition'):
        p.validate_jobs([job], manifest['protocol'])


def test_real_adapter_timeout_with_valid_partial_output_is_primary_failure(tmp_path):
    job, manifest, runtime, quota, _, execute, _, _ = fixture(tmp_path, 'agent')
    real = p.load_runtime(tmp_path)
    runtime.BASE_ADAPT = real.BASE_ADAPT
    def timeout(text, cwd, *args):
        (cwd / 'run.py').write_text('pass\n')
        return dict(ok=False, text='', timed_out=True, returncode=-9, num_tool_calls=1)
    row = p.run_job(job, manifest, tmp_path / 'runs', runtime, quota, timeout, execute)
    assert row['status'] == 'provider_timeout'
    assert row['outcome']['numerical_success'] is True
    assert row['outcome']['primary_success'] is False
    assert row['scoring']['success'] is False


def test_real_adapter_resumed_quota_keeps_transport_attempts_separate(tmp_path):
    job, manifest, runtime, quota, call, execute, calls, _ = fixture(tmp_path)
    real = p.load_runtime(tmp_path)
    runtime.BASE_ADAPT = real.BASE_ADAPT
    def quota_then_response(text, cwd, job, destination, number):
        if number == 1:
            (destination / 'quota.stdout').write_text(json.dumps(event()) + '\n')
            return dict(ok=False, text='', num_tool_calls=0, returncode=1,
                        raw_stream_file='quota.stdout', provider_limit=True)
        return call(text, cwd, job, destination, number)
    row = p.run_job(job, manifest, tmp_path / 'runs', runtime, quota, quota_then_response, execute)
    assert row['outcome']['primary_success'] is True
    assert row['provider_ok'] is True
    assert row['all_provider_attempts_ok'] is False
    assert row['calls'][0]['quota_rejected_before_response'] is True
    assert len(row['calls']) == 2 and len(calls) == 1


def test_preparation_stores_effective_agent_prompt_and_policy(tmp_path, monkeypatch):
    data = tmp_path / 'data.csv'; data.write_text('x\n1\n')
    catalog = tmp_path / 'catalog.json'
    p.save(catalog, {'toy': {'task_id': 'toy', 'data_path': str(data),
                           'reference': {'estimates': [{'term': 'b', 'estimate': 987654.3}]}}})
    decision = tmp_path / 'decision.md'; decision.write_text('Use agent verification.')
    text = 'Task: estimate b.\nWhen the requested output is present, finish with DONE.'
    prompt = tmp_path / 'prompt.txt'; prompt.write_text(text)
    jobs = [dict(block='test', task='toy', model=model, platform='python', prompt='zero',
                 condition=condition, output_format='json', rep=0)
            for model in ('luna', 'sonnet') for condition in ('chatbot', 'self_repair', 'agent')]
    config = tmp_path / 'config.json'
    p.save(config, dict(catalog_path=str(catalog), decision_record=str(decision), jobs=jobs,
                       prompt_paths={p.prompt_key(j): str(prompt) for j in jobs}))
    check = dict(argv=['version'], returncode=0, stdout='version', stderr='')
    runtime = SimpleNamespace(c=SimpleNamespace(
        command=lambda job, cwd, last: (['fake-provider', job['condition']], 100),
        prior=SimpleNamespace(check_tools=lambda: {'codex_version': check, 'sonnet_version': check})))
    monkeypatch.setattr(p, 'load_runtime', lambda work: runtime)
    monkeypatch.setattr(p, 'runtime_sources', lambda runtime: [])
    monkeypatch.setattr(p, 'provider_cli_versions', lambda runtime, jobs:
                        {'codex_version': check, 'sonnet_version': check})
    manifest = p.prepare(config, tmp_path / 'new/manifest.json')
    protocol = manifest['protocol']
    assert protocol['policy']['agent_verification_version'] == p.AGENT_VERIFICATION_VERSION
    for job in jobs:
        effective = protocol['prompts'][p.prompt_key(job)]
        assert effective == p.prepared_prompt(text, job['condition'])
        assert '987654.3' not in effective
        entry = protocol['prompt_provenance'][p.prompt_key(job)]
        assert entry['source_sha256'] == p.sha(prompt)
        assert entry['effective_sha256'] == p.hashlib.sha256(effective.encode()).hexdigest()
    assert prompt.read_text() == text
    p.verify(manifest)

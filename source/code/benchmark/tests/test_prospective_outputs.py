"""Offline regression tests for prospective output preservation."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import math
import re

import pytest
from benchmark import prospective_outputs as outputs

SOURCE = Path(__file__).resolve().parents[1] / 'prospective_collect.py'
loader = importlib.util.spec_from_file_location('prospective_outputs_test_runner', SOURCE)
p = importlib.util.module_from_spec(loader)
loader.loader.exec_module(p)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), 10**400])
def test_invalid_numeric_output_remains_failure_and_record_is_saved(tmp_path, bad):
    p.STOP.clear()
    runtime = p.load_runtime(tmp_path)
    data = tmp_path / 'data.csv'
    data.write_text('x\n1\n2\n')
    job = dict(block='test', task='toy', model='sonnet', platform='python',
               prompt='zero', condition='chatbot', output_format='json', rep=0)
    task = dict(task_id='toy', title='Toy', prompt='Compute output.', estimands=['b'],
                data_description='Two rows.', data_path=str(data),
                reference={'estimates':[dict(term='b', estimate=2, se=None, n_obs=2)]})
    manifest = dict(protocol_sha256='test', jobs=[job],
                    protocol=dict(tasks={'toy':task}, prompts={p.prompt_key(job):'Return code.'}))
    def provider(*args):
        return dict(ok=True, returncode=0, text='```python\npass\n```', num_tool_calls=0)
    def execute(script, platform, cwd, destination, number):
        (cwd / 'result.json').write_text(json.dumps({'estimates':[dict(term='b', estimate=bad, se=None, n_obs=2)]}))
        return dict(ok=True, returncode=0)
    record = p.run_job(job, manifest, tmp_path/'runs', runtime,
                       p.QuotaCoordinator(tmp_path/'quota.json'), provider, execute)
    assert record['outcome']['primary_success'] is False
    assert record['outcome']['numerical_success'] is False
    assert record['status'] == 'completed'
    destination = Path(record['attempt_directory'])
    saved = json.loads((destination/'run_record.json').read_text())
    assert saved['scoring']['estimates']['b']['estimate'] is None
    raw = json.loads((destination/'final_workspace/result.json').read_text())
    raw_value = raw['estimates'][0]['estimate']
    assert raw_value == bad if isinstance(bad, int) else not math.isfinite(raw_value)
    assert not p.STOP.is_set()


def test_delivery_check_handles_integer_beyond_float_range():
    spec = {'reference':{'estimates':[dict(term='b',estimate=2)]}}
    assert p.delivery_problem(spec, {'b':dict(estimate=10**400)}) == 'b: missing finite estimate'


@pytest.mark.parametrize('log_text, expected_ok', [('ANSWER: b estimate=2 se=. n_obs=2\n', True), ('r(198);\n', False)])
def test_named_stata_log_is_used_and_execution_receipt_is_corrected(tmp_path, log_text, expected_ok):
    destination = tmp_path/'record'
    destination.mkdir()
    script = tmp_path/'analysis.do'
    script.write_text('display 2')
    def execute(*args):
        (tmp_path/'analysis.log').write_text(log_text)
        return dict(returncode=0, timed_out=False, ok=True)
    result = outputs.execute_submission(script, 'stata', tmp_path, destination, 1,
                                         executor=execute, harness=SimpleNamespace(STATA_ERROR_RE=re.compile(r'r\(\d+\);')))
    assert (tmp_path/'run.log').read_text() == log_text
    assert result['ok'] is expected_ok
    assert result['stata_batch_log'] == 'analysis.log'
    assert json.loads((destination/'execution_attempt1.json').read_text()) == result


def test_stale_named_stata_log_cannot_supply_results(tmp_path):
    destination = tmp_path/'record'
    destination.mkdir()
    (tmp_path/'analysis.log').write_text('ANSWER: b estimate=2 se=. n_obs=2\n')
    outputs.execute_submission(tmp_path/'analysis.do', 'stata', tmp_path, destination, 1,
        executor=lambda *args: dict(returncode=1, timed_out=False, ok=False),
        harness=SimpleNamespace(STATA_ERROR_RE=re.compile(r'r\(\d+\);')))
    assert not (tmp_path/'analysis.log').exists()
    assert not (tmp_path/'run.log').exists()


def test_record_normalization_keeps_finite_values_and_does_not_mutate():
    values = {'numeric':2.5, 'text':'NaN', 'values':[float('nan'), 3]}
    assert outputs.json_safe(values) == {'numeric':2.5, 'text':'NaN', 'values':[None, 3]}
    assert math.isnan(values['values'][0])


@pytest.mark.parametrize('platform, canonical', [('python','run.py'),('r','run.R'),('stata','run.do')])
def test_submission_selection_requires_canonical_or_unique_script(tmp_path, platform, canonical):
    extension=Path(canonical).suffix
    assert outputs.select_submission_script(tmp_path,platform,canonical) is None
    analysis=tmp_path/('analysis'+extension)
    analysis.write_text('analysis')
    assert outputs.select_submission_script(tmp_path,platform,canonical)==analysis
    verification=tmp_path/('verify'+extension)
    verification.write_text('verify')
    with pytest.raises(outputs.AmbiguousSubmissionError,match='Ambiguous submitted scripts'):
        outputs.select_submission_script(tmp_path,platform,canonical)
    submission=tmp_path/canonical
    submission.write_text('submission')
    (tmp_path/('attempt1_run'+extension)).write_text('earlier attempt')
    assert outputs.select_submission_script(tmp_path,platform,canonical)==submission


def test_ambiguous_agent_scripts_hold_before_execution(tmp_path):
    p.STOP.clear()
    runtime=p.load_runtime(tmp_path)
    data=tmp_path/'data.csv';data.write_text('x\n1\n2\n')
    job=dict(block='test',task='toy',model='sonnet',platform='python',prompt='zero',condition='agent',output_format='json',rep=0)
    task=dict(task_id='toy',title='Toy',prompt='Compute.',estimands=['b'],data_description='Two rows.',data_path=str(data),
              reference={'estimates':[dict(term='b',estimate=2,se=None,n_obs=2)]})
    manifest=dict(protocol_sha256='test',jobs=[job],protocol=dict(tasks={'toy':task},prompts={p.prompt_key(job):'Return code.'}))
    def provider(prompt,cwd,*args):
        (cwd/'analysis.py').write_text('pass\n')
        (cwd/'verify.py').write_text('pass\n')
        return dict(ok=True,returncode=0,text='DONE',num_tool_calls=0)
    def execute(*args):
        pytest.fail('Ambiguous scripts must not execute')
    record=p.run_job(job,manifest,tmp_path/'runs',runtime,p.QuotaCoordinator(tmp_path/'quota.json'),provider,execute)
    assert record['status']=='needs_review'
    assert record['hold_reason']=='ambiguous_submission'
    assert record['outcome']['primary_success'] is None
    assert record['submission_candidates']==['analysis.py','verify.py']
    assert (Path(record['held_workspace'])/'analysis.py').is_file()
    assert (Path(record['held_workspace'])/'verify.py').is_file()
    assert len(record['calls'])==1
    p.STOP.clear()

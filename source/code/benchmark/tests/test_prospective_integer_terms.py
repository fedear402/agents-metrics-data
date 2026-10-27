"""Exact count estimates retain integer semantics during prospective scoring."""
import json

import pytest

from test_prospective_collect import fixture, p


@pytest.mark.parametrize('value', [139.1, 138])
def test_declared_counts_reject_values_inside_relative_tolerance(tmp_path, value):
    p.STOP.clear()
    job, manifest, runtime, quotas, call, execute, _, _ = fixture(tmp_path)
    spec = manifest['protocol']['tasks']['toy']
    spec.update(title='Count', prompt='Count rows', estimands=['b'], data_description='Toy data')
    spec['reference']['estimates'][0]['estimate'] = 139
    spec['exact_integer_terms'] = ['b']
    runtime.c.compare_numeric = p.load_runtime(tmp_path).c.compare_numeric
    estimates = {'b': dict(estimate=value, n_obs=1)}
    assert runtime.c.compare_numeric(estimates, spec)[0] is True
    def execution(script, platform, cwd, destination, number):
        result = execute(script, platform, cwd, destination, number)
        (cwd / 'result.json').write_text(json.dumps(estimates))
        return result
    row = p.run_job(job, manifest, tmp_path / 'runs', runtime, quotas, call, execution)
    assert row['scoring']['numeric_success'] is False
    assert row['scoring']['per_term']['b']['coef_pass'] is False


@pytest.mark.parametrize('value, expected', [
    (139, True), (139.0, True), (139.1, False), (138, False),
    (True, False), (None, False), (float('nan'), False), (float('inf'), False),
])
def test_exact_integer_values(value, expected):
    spec = dict(exact_integer_terms=['count'], reference=dict(estimates=[dict(term='count', estimate=139)]))
    original = (True, {'count': {'coef_pass': True, 'se_pass': True}}, ['existing diagnostic'])
    good, terms, notes = p.enforce_integer_terms({'count': {'estimate': value}}, spec, original)
    assert good is expected
    assert terms['count']['coef_pass'] is expected
    assert terms['count']['exact_estimate_pass'] is expected
    assert terms['count']['se_pass'] is True
    assert notes[0] == 'existing diagnostic'
    assert len(notes) == (1 if expected else 2)


def test_boolean_one_is_not_an_integer_count():
    spec = dict(exact_integer_terms=['count'], reference=dict(estimates=[dict(term='count', estimate=1)]))
    assert p.enforce_integer_terms({'count': {'estimate': True}}, spec, (True, {}, []))[0] is False


def test_absent_metadata_preserves_comparison_and_diagnostics():
    comparison = (True, {'b': {'coef_pass': True}}, ['existing diagnostic'])
    assert p.enforce_integer_terms({'b': {'estimate': 139.1}}, {}, comparison) is comparison


def test_exact_success_keeps_existing_failure():
    spec = dict(exact_integer_terms=['count'], reference=dict(estimates=[dict(term='count', estimate=139)]))
    good, terms, _ = p.enforce_integer_terms({'count': {'estimate': 139}}, spec,
                                            (False, {'count': {'coef_pass': False}}, []))
    assert good is False and terms['count']['coef_pass'] is False

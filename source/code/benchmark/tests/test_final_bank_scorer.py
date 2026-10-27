"""Offline checks of the final task bank scorer (code/benchmark/final_bank_scorer.py).

The scorer reads scoring.json of a final-bank package and scores one run:
result.json for Python and R, and printed ANSWER lines for Stata (run.log,
stdout, or files written in the final run). No test calls a model provider.
"""
from pathlib import Path
import json
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import final_bank_scorer as fs  # noqa: E402

ROOT = HERE.parents[1]
BANKS = [ROOT / 'data/task_candidates/2026-09-28_final_v2', ROOT / 'data/task_candidates/2026-09-28_final_hard']


def toy_spec(**extra):
    spec = {
        'schema': fs.SCHEMA,
        'task_id': 'toy',
        'rel_tol': 0.01,
        'abs_tol': 1e-8,
        'checks': [
            {'term': 'beta', 'field': 'estimate', 'reference': 2.0, 'kind': 'continuous'},
            {'term': 'beta', 'field': 'se', 'reference': 0.5, 'kind': 'continuous'},
            {'term': 'df', 'field': 'estimate', 'reference': 196, 'kind': 'integer'},
            {'term': 'tiny', 'field': 'estimate', 'reference': 1e-9, 'kind': 'continuous'},
        ],
        'conventions': [
            {'name': 'cr0', 'description': 'alternative SE convention',
             'references': [{'term': 'beta', 'field': 'se', 'reference': 0.48}]},
        ],
        'n_obs': {'scored': False, 'reference': {'beta': 200, 'df': 200, 'tiny': 200}},
    }
    spec.update(extra)
    return spec


def rows(beta=2.0, se=0.5, df=196, tiny=1e-9, n=200):
    return {'beta': {'estimate': beta, 'se': se, 'n_obs': n},
            'df': {'estimate': df, 'se': None, 'n_obs': n},
            'tiny': {'estimate': tiny, 'se': None, 'n_obs': n}}


# Tolerance rule

def test_within_uses_relative_one_percent_with_tiny_absolute_floor():
    assert fs.within(2.0199, 2.0, 0.01, 1e-8)
    assert fs.within(1.9801, 2.0, 0.01, 1e-8)
    assert not fs.within(2.0201, 2.0, 0.01, 1e-8)
    assert fs.within(1e-9 + 5e-9, 1e-9, 0.01, 1e-8)
    assert not fs.within(1e-9 + 2e-8, 1e-9, 0.01, 1e-8)
    assert not fs.within(None, 2.0, 0.01, 1e-8)
    assert not fs.within(float('nan'), 2.0, 0.01, 1e-8)
    assert not fs.within(float('inf'), 2.0, 0.01, 1e-8)


def test_correct_rows_pass_and_record_each_check():
    result = fs.score_rows(rows(), toy_spec())
    assert result['passed'] is True
    assert result['convention'] == 'main'
    assert {(c['term'], c['field']) for c in result['checks']} == {
        ('beta', 'estimate'), ('beta', 'se'), ('df', 'estimate'), ('tiny', 'estimate')}
    assert all(c['pass'] for c in result['checks'])


def test_estimate_outside_tolerance_fails():
    result = fs.score_rows(rows(beta=2.03), toy_spec())
    assert result['passed'] is False
    failed = [c for c in result['checks'] if not c['pass']]
    assert [(c['term'], c['field']) for c in failed] == [('beta', 'estimate')]
    assert failed[0]['ratio'] == pytest.approx(1.5)


@pytest.mark.parametrize('value,ok', [(196, True), (196.0, True), (195, False), (196.0000001, False)])
def test_integer_checks_are_exact(value, ok):
    assert fs.score_rows(rows(df=value), toy_spec())['passed'] is ok


def test_missing_scored_se_fails_and_unscored_se_is_ignored():
    assert fs.score_rows(rows(se=None), toy_spec())['passed'] is False
    data = rows()
    data['df']['se'] = 123.0
    assert fs.score_rows(data, toy_spec())['passed'] is True


def test_missing_term_fails_with_reason():
    data = rows()
    del data['tiny']
    result = fs.score_rows(data, toy_spec())
    assert result['passed'] is False
    assert result['missing_terms'] == ['tiny']


def test_convention_set_passes_only_as_a_complete_set():
    result = fs.score_rows(rows(se=0.481), toy_spec())
    assert result['passed'] is True
    assert result['convention'] == 'cr0'
    spec = toy_spec()
    spec['conventions'][0]['references'].append({'term': 'beta', 'field': 'estimate', 'reference': 2.2})
    # SE matches cr0 but the estimate matches only the main reference: no complete set passes.
    assert fs.score_rows(rows(se=0.481), spec)['passed'] is False


def test_n_obs_is_diagnostic_unless_scored():
    result = fs.score_rows(rows(n=199), toy_spec())
    assert result['passed'] is True
    assert result['n_obs_mismatch'] == ['beta', 'df', 'tiny']
    assert fs.score_rows(rows(n=199), toy_spec(n_obs={'scored': True, 'reference': {'beta': 200}}))['passed'] is False


# Specification checks

def test_validate_rejects_zero_reference_p_values_and_unknown_kinds():
    assert fs.validate_spec(toy_spec()) == []
    bad = toy_spec()
    bad['checks'].append({'term': 'gamma', 'field': 'estimate', 'reference': 0.0, 'kind': 'continuous'})
    bad['checks'].append({'term': 'p_value', 'field': 'estimate', 'reference': 0.03, 'kind': 'continuous'})
    bad['checks'].append({'term': 'k', 'field': 'estimate', 'reference': 2.5, 'kind': 'integer'})
    bad['checks'].append({'term': 'z', 'field': 'estimate', 'reference': 1.0, 'kind': 'fuzzy'})
    problems = ' '.join(fs.validate_spec(bad))
    assert 'gamma' in problems and 'zero' in problems
    assert 'p_value' in problems
    assert 'k' in problems and 'integer' in problems
    assert 'fuzzy' in problems


def test_legacy_round4_targets_are_normalized():
    legacy = {'task_id': 'old', 'targets': [
        {'term': 'a', 'estimate': 1.5, 'estimate_rel_tol': 0.01, 'abs_floor': 1e-8, 'n_obs': 400,
         'se': 0.2, 'se_rel_tol': 0.01},
        {'term': 'set_type', 'estimate': 2.0, 'exact': True, 'n_obs': 400}]}
    spec = fs.normalize_spec(legacy)
    assert spec['checks'] == [
        {'term': 'a', 'field': 'estimate', 'reference': 1.5, 'kind': 'continuous'},
        {'term': 'a', 'field': 'se', 'reference': 0.2, 'kind': 'continuous'},
        {'term': 'set_type', 'field': 'estimate', 'reference': 2.0, 'kind': 'integer'}]
    assert fs.score_rows({'a': {'estimate': 1.51, 'se': 0.2, 'n_obs': 400},
                          'set_type': {'estimate': 2, 'se': None, 'n_obs': 400}}, spec)['passed']


def hard_layout():
    """scoring.json layout of the final hard-task packages (2026-09-28_final_hard/_build/build.py)."""
    return {
        'task_id': 'hard_toy', 'rule': '|value - reference| <= max(1e-8, 0.01 |reference|); counts exact',
        'relative_tolerance': 0.01, 'absolute_floor': 1e-8, 'point_of_task': 'inference',
        'scored': [
            {'term': 'b', 'field': 'estimate', 'reference': 0.5, 'check': 'relative 1 percent, floor 1e-8'},
            {'term': 'b', 'field': 'se', 'reference': 0.1, 'check': 'relative 1 percent, floor 1e-8'},
            {'term': 'set_type', 'field': 'estimate', 'reference': 2.0, 'check': 'exact'},
            {'term': 'b', 'field': 'n_obs', 'reference': 400, 'check': 'exact'},
        ],
        'diagnostic': [{'term': 'all rows', 'field': 'n_obs', 'note': 'recorded'}],
        'p_values_scored': False,
        'wrong_variants': [{'name': 'x', 'status': 'fails'}],
    }


def test_hard_layout_is_normalized_and_scored():
    spec = fs.normalize_spec(hard_layout())
    assert fs.validate_spec(spec) == []
    assert spec['rel_tol'] == 0.01 and spec['abs_tol'] == 1e-8
    assert spec['checks'] == [
        {'term': 'b', 'field': 'estimate', 'reference': 0.5, 'kind': 'continuous'},
        {'term': 'b', 'field': 'se', 'reference': 0.1, 'kind': 'continuous'},
        {'term': 'set_type', 'field': 'estimate', 'reference': 2.0, 'kind': 'integer'}]
    good = {'b': {'estimate': 0.504, 'se': 0.1009, 'n_obs': 400}, 'set_type': {'estimate': 2, 'se': None, 'n_obs': 400}}
    assert fs.score_rows(good, spec)['passed'] is True
    good['b']['n_obs'] = 399
    result = fs.score_rows(good, spec)
    assert result['passed'] is False and result['n_obs_mismatch'] == ['b']


def test_changed_input_file_fails_and_unchanged_passes(tmp_path):
    package = tmp_path / 'pkg'
    (package / 'data').mkdir(parents=True)
    (package / 'data' / 'input.csv').write_text('a,b\n1,2\n')
    (package / 'scoring.json').write_text(json.dumps(toy_spec()))
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'run.log').write_text(LOG)
    result = fs.score_run(package, 'stata', run)
    assert result['passed'] is True and result['input_files'] == {'input.csv': 'not in run folder'}
    (run / 'input.csv').write_text('a,b\n1,2\n')
    assert fs.score_run(package, 'stata', run)['input_files'] == {'input.csv': 'unchanged'}
    (run / 'input.csv').write_text('a,b\n1,3\n')
    result = fs.score_run(package, 'stata', run)
    assert result['passed'] is False and 'input file changed' in result['reason']


# Python and R output

def write_result(folder, estimates, **extra):
    data = {'task_id': 'toy', 'platform': 'python', 'estimates': estimates,
            'method_metadata': {}, 'diagnostics': {}}
    data.update(extra)
    (folder / 'result.json').write_text(json.dumps(data))


def json_rows(beta=2.0, se=0.5):
    return [{'term': 'beta', 'estimate': beta, 'se': se, 'n_obs': 200},
            {'term': 'df', 'estimate': 196, 'se': None, 'n_obs': 200},
            {'term': 'tiny', 'estimate': 1e-9, 'se': None, 'n_obs': 200}]


def test_python_result_json_is_scored(tmp_path):
    write_result(tmp_path, json_rows())
    result = fs.score_run(toy_spec(), 'python', tmp_path)
    assert result['passed'] is True
    assert result['source'] == 'result.json'


def test_r_na_strings_are_missing_values(tmp_path):
    estimates = json_rows()
    estimates[1]['se'] = 'NA'
    write_result(tmp_path, estimates)
    assert fs.score_run(toy_spec(), 'r', tmp_path)['passed'] is True


def test_missing_or_invalid_result_json_fails(tmp_path):
    result = fs.score_run(toy_spec(), 'python', tmp_path)
    assert result['passed'] is False and 'result.json missing' in result['reason']
    (tmp_path / 'result.json').write_text('{"estimates": [')
    result = fs.score_run(toy_spec(), 'python', tmp_path)
    assert result['passed'] is False and 'invalid' in result['reason']


def test_conflicting_duplicate_json_rows_fail(tmp_path):
    estimates = json_rows() + [{'term': 'beta', 'estimate': 2.5, 'se': 0.5, 'n_obs': 200}]
    write_result(tmp_path, estimates)
    result = fs.score_run(toy_spec(), 'python', tmp_path)
    assert result['passed'] is False
    assert result['conflicts'][0]['term'] == 'beta'
    write_result(tmp_path, json_rows() + [json_rows()[0]])
    assert fs.score_run(toy_spec(), 'python', tmp_path)['passed'] is True


def test_json_booleans_are_not_numbers(tmp_path):
    estimates = json_rows()
    estimates[1]['estimate'] = True
    write_result(tmp_path, estimates)
    assert fs.score_run(toy_spec(), 'python', tmp_path)['passed'] is False


# Stata output

LOG = """. display "ANSWER: beta estimate=" %24.16g b " se=" %24.16g s " n_obs=" %12.0f N
ANSWER: beta estimate=                       2 se=                     0.5 n_obs=         200
. display "ANSWER: df estimate=" %24.16g d " se=. n_obs=" %12.0f N
ANSWER: df estimate=                     196 se=. n_obs=         200
ANSWER: tiny estimate=                   1e-09 se=. n_o
> bs=         200
"""


def test_stata_run_log_is_scored_and_command_echo_is_ignored(tmp_path):
    (tmp_path / 'run.log').write_text(LOG)
    result = fs.score_run(toy_spec(), 'stata', tmp_path)
    assert result['passed'] is True
    assert result['source'] == 'run.log'
    assert result['rows']['tiny'] == {'estimate': 1e-9, 'se': None, 'n_obs': 200.0}


def test_stata_stdout_and_final_run_files_are_sources(tmp_path):
    (tmp_path / 'answers.txt').write_text(LOG)
    result = fs.score_run(toy_spec(), 'stata', tmp_path, final_files=[tmp_path / 'answers.txt'])
    assert result['passed'] is True and result['source'] == 'answers.txt'
    result = fs.score_run(toy_spec(), 'stata', tmp_path, stdout=LOG)
    assert result['passed'] is True and result['source'] == 'stdout'


def test_stata_without_answer_lines_fails(tmp_path):
    (tmp_path / 'run.log').write_text('. display 1\n1\n')
    result = fs.score_run(toy_spec(), 'stata', tmp_path)
    assert result['passed'] is False
    assert 'no ANSWER lines' in result['reason']


def test_stata_conflict_between_sources_fails(tmp_path):
    (tmp_path / 'run.log').write_text(LOG)
    (tmp_path / 'out.txt').write_text('ANSWER: beta estimate=2.5 se=0.5 n_obs=200\n')
    result = fs.score_run(toy_spec(), 'stata', tmp_path, final_files=[tmp_path / 'out.txt'])
    assert result['passed'] is False
    assert result['conflicts'][0]['term'] == 'beta'
    assert 'conflict' in result['reason']


def test_stata_conflict_within_the_log_fails(tmp_path):
    (tmp_path / 'run.log').write_text(LOG + 'ANSWER: df estimate=195 se=. n_obs=200\n')
    assert fs.score_run(toy_spec(), 'stata', tmp_path)['passed'] is False


def test_stata_same_value_at_lower_printed_precision_is_not_a_conflict(tmp_path):
    (tmp_path / 'run.log').write_text(LOG.replace('estimate=                       2 se',
                                                  'estimate=      2.0000000012345 se'))
    (tmp_path / 'out.txt').write_text('ANSWER: beta estimate=2.000000 se=0.500000 n_obs=200\n')
    result = fs.score_run(toy_spec(), 'stata', tmp_path, final_files=[tmp_path / 'out.txt'])
    assert result['conflicts'] == []
    assert result['passed'] is True
    assert result['rows']['beta']['estimate'] == 2.0000000012345


def test_printed_values_agree_rule():
    assert fs.printed_values_agree('2.000000', '2.0000000012345')
    assert fs.printed_values_agree('1.2346e-05', '1.23456789e-05')
    assert not fs.printed_values_agree('1.2347e-05', '1.23456789e-05')
    assert fs.printed_values_agree('.', 'NA')
    assert not fs.printed_values_agree('.', '0.5')
    assert not fs.printed_values_agree('2', '2.5')
    assert fs.printed_values_agree('2', '2.000')


# Command line

def test_cli_prints_json(tmp_path):
    package = tmp_path / 'pkg'
    package.mkdir()
    (package / 'scoring.json').write_text(json.dumps(toy_spec()))
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'run.log').write_text(LOG)
    out = subprocess.run([sys.executable, str(HERE / 'final_bank_scorer.py'), '--package', str(package),
                          '--language', 'stata', '--run-dir', str(run)], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout)['passed'] is True


# Real packages

def packages():
    return sorted(p for bank in BANKS if bank.exists() for p in bank.iterdir() if (p / 'scoring.json').exists())


@pytest.mark.parametrize('package', packages(), ids=lambda p: f'{p.parent.name}/{p.name}')
def test_real_package_spec_is_valid_and_reference_passes(package):
    spec = fs.load_spec(package)
    assert fs.validate_spec(spec) == []
    reference = json.loads((package / 'reference.json').read_text())
    values = reference.get('reference') or reference.get('route_a_values')
    assert fs.score_rows(values, spec)['passed'] is True

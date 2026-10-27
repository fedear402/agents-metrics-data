"""Printed ANSWER lines: the collector parser accepts the contract the prompt states.

The Stata prompts state three contract forms:
- ``ANSWER: term estimate=<number> se=<number or .> n_obs=<integer>``
- ``ANSWER: term estimate=<number> se=NA n_obs=<integer>``
- ``ANSWER: term estimate=<number>`` (cohort_stack, oaxaca, welfare)

Before the fix the parser in ``collection_runtime/runner.py`` required all
three fields in one fixed order. An estimate-only line or a reordered line
produced no row, and the self-repair loop sent a false repair request
("missing finite estimate"). The wrapped-log fixtures are copied from
StataSE 19.5 and StataMP 14 batch logs at ``set linesize 80``.
"""
import pytest

from benchmark import prospective_collect as p


@pytest.fixture(scope='module')
def base(tmp_path_factory):
    return p.load_runtime(tmp_path_factory.mktemp('runtime')).c.prior.base


def spec(terms, se=None, n_obs=None):
    return dict(task_id='toy', estimands=list(terms),
                reference=dict(estimates=[dict(term=t, estimate=1.0, se=se, n_obs=n_obs) for t in terms]))


def test_estimate_only_line_is_parsed_and_triggers_no_repair(base):
    text = 'ANSWER: att_g3 estimate=0.954222981646582\nANSWER: att_g5 estimate=       1.2377404221405\n'
    estimates, notes = base.printed_estimates(text)
    assert notes == []
    assert estimates == {'att_g3': dict(estimate=0.954222981646582, se=None, n_obs=None),
                         'att_g5': dict(estimate=1.2377404221405, se=None, n_obs=None)}
    assert p.delivery_problem(spec(['att_g3', 'att_g5']), estimates) == ''


def test_fields_in_any_order_are_parsed(base):
    text = ('ANSWER: b n_obs=300 se=0.5 estimate=1.5\n'
            'ANSWER: c se=. estimate=-2.25 n_obs=300\n'
            'ANSWER: d estimate=3 n_obs=12\n')
    estimates, notes = base.printed_estimates(text)
    assert notes == []
    assert estimates['b'] == dict(estimate=1.5, se=0.5, n_obs=300.0)
    assert estimates['c'] == dict(estimate=-2.25, se=None, n_obs=300.0)
    assert estimates['d'] == dict(estimate=3.0, se=None, n_obs=12.0)
    assert p.delivery_problem(spec(['b'], se=1.0, n_obs=300), estimates) == ''


def test_stata_wrapped_log_lines_are_joined(base):
    log = '\n'.join([
        '. display "ANSWER: reordered_term_name_long n_obs=" %12.0f 300 " se=" %21.12f 0.5 " estimate=" %21.12f 1.5',
        'ANSWER: reordered_term_name_long n_obs=         300 se=       0.500000000000 est',
        '> imate=       1.500000000000',
        'ANSWER: abcdefghijklmno estimate=       2.053051503836 se=       2.053051503836 ',
        '> n_obs=         300',
        'ANSWER: eligible_nonrecipient_effect_with_a_long_name estimate=       5.78192932011235',
        '>  se=. n_o',
        '> bs=300',
        'ANSWER: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
        '> xxx estimate=       1.500000000000',
        '',
    ])
    estimates, notes = base.printed_estimates(log)
    assert notes == []
    assert estimates['reordered_term_name_long'] == dict(estimate=1.5, se=0.5, n_obs=300.0)
    assert estimates['abcdefghijklmno'] == dict(estimate=2.053051503836, se=2.053051503836, n_obs=300.0)
    assert estimates['eligible_nonrecipient_effect_with_a_long_name'] == dict(
        estimate=5.78192932011235, se=None, n_obs=300.0)
    assert estimates['x' * 80] == dict(estimate=1.5, se=None, n_obs=None)


def test_runner_feedback_sends_no_repair_for_estimate_only_contract(base, tmp_path):
    (tmp_path / 'run.log').write_text('ANSWER: gap estimate=1.4355165973939368\n'
                                      'ANSWER: explained estimate=0.5722979841301531\n')
    assert base.feedback_output(tmp_path, spec(['gap', 'explained']), 'stata') is None


@pytest.mark.parametrize('line', [
    'ANSWER: b se=0.5 n_obs=300',                       # no estimate
    'ANSWER: b estimate=1 estimate=2',                  # repeated field
    'ANSWER: b estimate=1 ci_low=0.5',                  # field outside the contract
    'ANSWER b estimate=1 se=0.5 n_obs=300',             # no colon, outside the contract
    'ANSWER: b estimate=1 se=0.5 n_obs=300 extra',      # trailing token
    'ANSWER: b',                                        # no fields
])
def test_lines_outside_the_contract_give_no_row(base, line):
    estimates, _ = base.printed_estimates(line + '\n')
    assert estimates == {}


def test_unparseable_value_is_reported(base):
    estimates, notes = base.printed_estimates('ANSWER: b estimate=abc\n')
    assert estimates == {}
    assert notes == ['Unparseable numeric value: b']

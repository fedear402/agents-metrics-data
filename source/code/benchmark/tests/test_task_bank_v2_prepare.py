"""Offline checks of the task bank v2 plan builder. No package folder is required."""
import importlib.util
from pathlib import Path

import pytest

from benchmark import prospective_collect as p

SOURCE = Path(__file__).resolve().parents[1] / 'workflows/task_bank_v2/prepare.py'
loader = importlib.util.spec_from_file_location('task_bank_v2_prepare_test', SOURCE)
v2 = importlib.util.module_from_spec(loader)
loader.loader.exec_module(v2)


def test_full_plan_has_14400_unique_positions_in_valid_blocks():
    tasks = [f'task{i:02d}' for i in range(40)]
    jobs = v2.plan_jobs(tasks, 10, v2.MODELS)
    assert len(jobs) == 14400
    assert len({v2.slot(job) for job in jobs}) == 14400
    assert sorted({job['block'] for job in jobs}) == [
        'task_bank_v2_luna_0', 'task_bank_v2_luna_5', 'task_bank_v2_sonnet_0', 'task_bank_v2_sonnet_5']
    assert all(0 <= job['rep'] < 5 and job['rep'] == job['global_rep'] % 5 for job in jobs)
    assert all(job['output_format'] == ('answer' if job['platform'] == 'stata' else 'json') for job in jobs)


def test_pilot_plan_counts():
    jobs = v2.plan_jobs(['a', 'b'], 1, ('sonnet',))
    assert len(jobs) == 36 and {job['global_rep'] for job in jobs} == {0}


@pytest.mark.parametrize('agency', v2.AGENCIES)
def test_stata_agency_text_names_the_log_and_keeps_one_completion_line(agency):
    text = v2.v2_suffix('stata', agency)
    assert 'run.log' in text
    prompt = 'Task:\nx\n' + text
    assert p.prepared_prompt(prompt, agency)


def test_stata_contract_parses_and_satisfies_the_delivery_check(tmp_path):
    base = p.load_runtime(tmp_path).c.prior.base
    spec = dict(estimands=['a', 'b'], reference=dict(estimates=[
        dict(term='a', estimate=1.0, se=0.5, n_obs=10), dict(term='b', estimate=2.0, se=None, n_obs=None)]))
    contract = v2.stata_contract(spec)
    assert 'run.log' in contract
    filled = '\n'.join(line.replace('<number>', '1.5').replace('<integer>', '10')
                       for line in contract.splitlines() if line.startswith('ANSWER:'))
    estimates, notes = base.printed_estimates(filled)
    assert notes == [] and sorted(estimates) == ['a', 'b']
    assert p.delivery_problem(spec, estimates) == ''

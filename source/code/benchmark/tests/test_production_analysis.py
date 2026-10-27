"""Verify saved-output scoring and preserve missing operational measurements."""
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import production_analysis as analysis


@pytest.fixture
def derived_row(tmp_path, monkeypatch):
    design, out = tmp_path / 'design', tmp_path / 'analysis'
    design.mkdir()
    spec = dict(task_id='toy', title='Toy', prompt='Estimate b.', estimands=['b'],
                data_description='', data_path=str(tmp_path / 'unused.csv'),
                reference={'estimates': [dict(term='b', estimate=2, se=0.1, n_obs=20)]})
    (design / 'catalog.json').write_text(json.dumps({'toy': spec}))
    record = dict(completed_at='2026-09-18', model='gpt-5.6-luna',
        codex={'ok': False, 'usage': {'input_tokens': 10, 'output_tokens': 2}},
        execution={'ok': True}, provider_ok=False, task_censored=False, protocol_violation=False,
        scoring={'numeric_success': True, 'artifact_success': True})
    path = tmp_path / 'run_record.json'
    path.write_text(json.dumps(record))
    (tmp_path / 'result.json').write_text(json.dumps({'estimates': [{'term': 'b', 'estimate': 2}]}))
    row = dict(task='toy', model='luna', platform='python', prompt='zero_shot', condition='chatbot',
               global_rep=0, status='retained', record_path=str(path), record_sha256=analysis.sha(path),
               position_id='toy0', source='legacy')
    (design / 'planned_positions.json').write_text(json.dumps([row] + [{'status': 'planned'}] * 7919))
    monkeypatch.setattr(analysis, 'DESIGN', design)
    monkeypatch.setattr(analysis, 'OUT', out)
    analysis.build()
    return pd.read_csv(out / 'runs.csv').iloc[0]


def test_legacy_full_score_remains_available(derived_row):
    assert derived_row.recorded_full_success == True


def test_missing_durations_remain_unmeasured(derived_row):
    assert pd.isna(derived_row.provider_seconds)
    assert pd.isna(derived_row.execution_seconds)


def test_provider_failure_stays_visible_without_overriding_estimate_score(derived_row):
    assert derived_row.estimate_only == True
    assert derived_row.provider_ok == False
    assert derived_row.task_censored == False
    assert derived_row.protocol_violation == False


def test_estimate_only_ignores_auxiliary_fields_and_rejects_boolean(tmp_path):
    c = analysis.runtime.load_runtime(tmp_path).c
    spec = dict(task_id='toy', title='Toy', prompt='Estimate b.', estimands=['b'],
                data_description='', data_path='/unused.csv',
                reference={'estimates': [dict(term='b', estimate=1, se=0.1, n_obs=20)]})
    assert analysis.score_estimates({'b': dict(estimate=1, se=999, n_obs=-1)}, spec, c.compare_numeric)[0]
    assert not analysis.score_estimates({'b': dict(estimate=True)}, spec, c.compare_numeric)[0]
    assert spec['reference']['estimates'][0]['se'] == 0.1


@pytest.mark.parametrize('baseline,score,count', [
    ('2026-09-17_applied-next-eight/existing_results/runs.csv', 'estimate_only', 1800),
    ('2026-09-17_complete-checkpoint-package/all_18/data/runs.csv', 'estimate_only_success', 1296),
])
def test_saved_output_parity_with_accepted_samples(baseline, score, count):
    # This integration check reads previously generated analysis and raw-source identities.
    source = ROOT / 'data/collections' / baseline
    current = ROOT / 'data/analysis/paper22_2026-09-18/runs.csv'
    if not source.exists() or not current.exists():
        pytest.skip('The local accepted sample and derived production analysis are required.')
    def canonical(value):
        path = ROOT / value
        reports = ROOT / 'quality_reports/reports'
        if path.is_relative_to(reports):
            path = ROOT / 'data/collections' / path.relative_to(reports)
        return str(path.resolve())
    old = pd.read_csv(source)
    old['record_path'] = old.record_path.map(canonical)
    old = old[['record_path', score]].rename(columns={score: 'accepted_score'})
    new = pd.read_csv(current)[['record_path', 'estimate_only']]
    # Historical aliases and canonical data paths identify the same saved file.
    new['record_path'] = new.record_path.map(canonical)
    joined = old.merge(new, on='record_path', validate='one_to_one')
    assert len(joined) == count
    assert joined.accepted_score.eq(joined.estimate_only).all()

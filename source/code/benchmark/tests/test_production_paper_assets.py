"""Check task eligibility and preserve the existing manuscript table layout."""
import itertools
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import production_paper_assets as assets
from production_paper_figures import failure_rates


def completed_cells(task='complete', repetitions=1, success=False):
    rows = []
    for model, platform, prompt, agency, rep in itertools.product(
            assets.MODELS, assets.PLATFORMS, assets.PROMPTS, assets.AGENCIES, range(repetitions)):
        rows.append(dict(task=task, model=model, platform=platform, prompt=prompt,
            agency=agency, rep=rep, primary=success, record_path=f'/saved/{task}/{model}/{platform}/{prompt}/{agency}/{rep}',
            record_sha256='record_hash', output_sha256='output_hash'))
    return pd.DataFrame(rows)


def test_every_cell_can_fail_and_task_still_qualifies():
    decision = assets.task_eligibility(completed_cells())[0]
    assert decision['eligible'] is True
    assert decision['observed_cells'] == 36


@pytest.mark.parametrize('repetitions', [5, 10])
def test_pass_repetition_macros_match_collected_runs(tmp_path, repetitions):
    (tmp_path / 'data').mkdir()
    data = completed_cells(repetitions=repetitions, success=True).assign(
        execution_ok=True, output_available=True, total_main_tokens=100,
        provider_seconds=10, cost_usd=.01, cost_pricing_complete=True)
    data.to_csv(tmp_path / 'data/runs.csv', index=False)
    regressions = pd.DataFrame([dict(model='pooled', contrast='H1_r',
                                    effective_tasks=1, estimate_pp=0, n_runs=len(data))])
    assets.macros(data, regressions, failure_rates(data), tmp_path, pass_tasks=1)
    values = json.loads((tmp_path / 'data/numbers.json').read_text())
    for name in ['snapPassMinReps', 'snapPassMaxReps']:
        assert values[name] == repetitions
        assert f'\\newcommand{{\\{name}}}{{{repetitions}}}' in (tmp_path / 'numbers.tex').read_text()


def test_extra_repetitions_do_not_replace_a_missing_cell():
    data = completed_cells(repetitions=3)
    missing = data.iloc[0]
    remove = (data[assets.CELL[1:]] == missing[assets.CELL[1:]]).all(axis=1)
    decision = assets.task_eligibility(data[~remove])[0]
    assert len(data[~remove]) > 36
    assert decision['eligible'] is False
    assert decision['observed_cells'] == 35
    assert len(decision['missing_cells']) == 1


def test_missing_entire_model_is_ineligible_despite_many_responses():
    data = completed_cells(repetitions=10)
    decision = assets.task_eligibility(data[data.model.eq('luna')])[0]
    assert decision['eligible'] is False
    assert decision['completed_responses'] == 180
    assert len(decision['missing_cells']) == 18


def test_unknown_design_value_is_rejected():
    data = completed_cells()
    data.loc[0, 'agency'] = 'other'
    with pytest.raises(ValueError, match='Unexpected design cells'):
        assets.task_eligibility(data)


def test_selection_retains_ten_reps_and_is_idempotent(tmp_path):
    folder = tmp_path / 'data'
    folder.mkdir()
    complete = completed_cells(repetitions=10)
    partial = completed_cells(task='partial').iloc[:1]
    data = pd.concat([complete, partial], ignore_index=True)
    path = folder / 'runs.csv'
    data.to_csv(path, index=False)
    source_hash = assets.sha(path)
    cutoff = '2026-09-18T14:50:15.030671+00:00'
    (folder / 'snapshot.json').write_text(json.dumps(dict(runs_sha256=source_hash, inventory_cutoff=cutoff, planned=720)))
    receipt = assets.apply_task_eligibility(tmp_path)
    selected = pd.read_csv(path)
    assert len(selected) == 360
    assert selected.rep.max() == 9
    assert selected.primary.eq(False).all()
    assert receipt['excluded_responses'] == 1
    assert receipt['source_runs_sha256'] == source_hash
    assert receipt['cutoff'] == cutoff
    selected_hash = assets.sha(path)
    assert assets.apply_task_eligibility(tmp_path) == receipt
    assert assets.sha(path) == selected_hash
    snapshot = json.loads((folder / 'snapshot.json').read_text())
    assert snapshot['source_analyzed'] == 361
    assert snapshot['analyzed'] == 360
    assert snapshot['inventory_cutoff'] == cutoff


def test_main_table_retains_original_layout(tmp_path):
    (tmp_path / 'tables').mkdir()
    names = ['H1_r', 'H1_stata', 'GAP_stata_avg', 'GAP_stata_agent_zs',
             'H2_fewshot', 'H3_self_repair', 'H3_tool_loop', 'H4_tool_loop']
    result = pd.DataFrame([dict(contrast=name, model=model, estimate_pp=-0.000001,
        se_cr1_pp=1.234, p_wcb=.004, n_runs=1000, effective_tasks=18)
        for name, model in itertools.product(names, assets.MODELS + ['pooled'])])
    assets.write_main_results(result, tmp_path, bootstrap=True)
    tex = (tmp_path / 'tables/main_results.tex').read_text()
    assert r'\caption{Effects of software, prompting, and agency on task success.}' in tex
    assert r'\begin{table}[!ht]' in tex
    assert r'\begin{tabular*}{\linewidth}' in tex
    assert r'\setstretch{1}' in tex
    assert tex.count(r'\textbf{') == 4
    assert '{Total}' in tex
    assert 'GPT-5.6 Luna (Codex)' in tex
    assert 'Software (percentage points)' in tex
    assert 'Bootstrap $p$' not in tex
    assert '-0.0' not in tex
    assert '0.0^{***}' in tex
    assert '(1.2)' in tex
    assert 'Observations & 1,000 & 1,000 & 1,000' in tex
    assert 'Task fixed effects & Yes & Yes & Yes' in tex
    assert '18 task clusters' in tex


def test_requested_exclusion_removes_complete_task_and_preserves_receipt(tmp_path):
    folder = tmp_path / 'data'
    folder.mkdir()
    data = pd.concat([completed_cells('retained'), completed_cells('d1_attendance')])
    data.to_csv(folder / 'runs.csv', index=False)
    (folder / 'paper_exclusions.json').write_text(json.dumps({'tasks': ['d1_attendance'], 'scope': 'local version'}))
    receipt = assets.apply_task_eligibility(tmp_path)
    assert receipt['source_responses'] == 72
    assert receipt['selected_responses'] == 36
    assert receipt['excluded_tasks'] == ['d1_attendance']
    decision = next(d for d in receipt['decisions'] if d['task'] == 'd1_attendance')
    assert decision['coverage_eligible'] and not decision['eligible']
    assert set(pd.read_csv(folder / 'runs.csv').task) == {'retained'}
    assert assets.apply_task_eligibility(tmp_path) == receipt
    (folder / 'paper_exclusions.json').write_text(json.dumps({'tasks': []}))
    with pytest.raises(ValueError, match='restore the full fixed snapshot'):
        assets.apply_task_eligibility(tmp_path)

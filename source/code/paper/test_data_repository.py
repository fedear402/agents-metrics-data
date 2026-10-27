"""Local export and reproduction checks with small synthetic run collections."""
from itertools import product
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

import data_repository as export
import data_repository_reproduce as reproduce


def synthetic_runs(models=("sonnet", "luna")):
    rows = []
    for m, model in enumerate(models):
        for task, cell, rep in product(range(4), reproduce.CELLS, range(2)):
            language, prompt, agency = cell
            primary = (task + reproduce.CELLS.index(cell) + rep + m) % 5 != 0
            rows.append(dict(model=model, task=f"task_{task}", platform=language,
                             prompt=prompt, agency=agency, primary=primary,
                             record_path=f"runs/{model}/{len(rows)}/record.json",
                             failure_category="Success" if primary else "Loud failure"))
    return pd.DataFrame(rows)


def stored_contrasts(runs):
    return pd.concat([reproduce.contrasts(runs, model)
                      for model in [*runs.model.unique(), "pooled"]], ignore_index=True)


def test_reproduction_rejects_extra_stored_contrasts(tmp_path, monkeypatch):
    """The former inner join silently ignored unknown stored contrasts."""
    runs = synthetic_runs()
    paper = stored_contrasts(runs)
    extra = paper.iloc[[0]].assign(contrast="unexpected_contrast")
    pd.concat([paper, extra]).to_csv(tmp_path / "regression_contrasts.csv", index=False)
    runs.to_csv(tmp_path / "runs.csv", index=False)
    runs[["record_path"]].assign(cost_usd=0.1).to_csv(tmp_path / "run_pricing.csv", index=False)
    monkeypatch.setattr(reproduce, "DATA", tmp_path)
    with pytest.raises((ValueError, SystemExit), match="contrast"):
        reproduce.main()


@pytest.mark.parametrize("change", ["missing", "duplicate", "estimate", "standard_error"])
def test_reproduction_rejects_incomplete_or_changed_contrasts(change):
    runs = synthetic_runs()
    paper = stored_contrasts(runs)
    if change == "missing":
        paper = paper.iloc[1:]
    elif change == "duplicate":
        paper = pd.concat([paper, paper.iloc[[0]]])
    elif change == "estimate":
        paper.loc[0, "estimate_pp"] += 1e-5
    else:
        paper.loc[0, "se_cr1_pp"] += 1e-4
    with pytest.raises(ValueError, match="contrast"):
        reproduce.verify_contrasts(runs, paper)


def test_all_six_models_and_stored_group_and_pair_contrasts():
    models = ("sonnet", "luna", "haiku", "luna6", "opus", "sol")
    runs = synthetic_runs(models)
    groups = {m: [m] for m in models}
    groups.update(pooled=list(models), cheap=["luna", "haiku", "luna6"],
                  expensive=["sonnet", "opus", "sol"], original_pair=["sonnet", "luna"])
    paper = pd.concat([reproduce.contrasts(runs, name, members).assign(member_models=','.join(members))
                       for name, members in groups.items()], ignore_index=True)
    checked = reproduce.verify_contrasts(runs, paper)
    assert len(checked) == len(groups) * 15
    assert checked.match.all()
    assert set(checked.model) == set(groups)
    # Independent group calls must also agree with a direct selection of its member models.
    pair = reproduce.contrasts(runs[runs.model.isin(groups['original_pair'])], 'pooled')
    np.testing.assert_allclose(pair[['estimate_pp', 'se_cr1_pp']],
                               paper[paper.model.eq('original_pair')][['estimate_pp', 'se_cr1_pp']], atol=1e-10)


def test_missing_cost_preserves_accuracy_and_uses_priced_denominator():
    runs = pd.DataFrame(dict(record_path=['a', 'b', 'c'], agency=['agent'] * 3,
                             primary=[True, False, True],
                             failure_category=['Success', 'Loud failure', 'Success']))
    pricing = pd.DataFrame(dict(record_path=['a', 'b', 'c'], cost_usd=[2.0, np.nan, 4.0]))
    frame = reproduce.priced_runs(runs, pricing)
    result = reproduce.rates(frame).loc['agent']
    assert len(frame) == result.runs == 3
    assert result.priced_runs == 2
    assert result.mean_cost_usd == 3
    assert result.success == pytest.approx(200 / 3)
    assert result.loud == pytest.approx(100 / 3)
    with pytest.raises(ValueError, match='coverage'):
        reproduce.priced_runs(runs, pricing.iloc[:2])
    with pytest.raises(ValueError, match='duplicate'):
        reproduce.priced_runs(runs, pd.concat([pricing, pricing.iloc[[0]]]))


def test_constant_task_contrast_has_finite_zero_standard_error():
    runs = synthetic_runs(('opus',))
    runs.loc[runs.agency.eq('agent') & runs.prompt.eq('zero_shot')
             & runs.platform.isin(['python', 'stata']), 'primary'] = True
    result = reproduce.contrasts(runs, 'opus').set_index('contrast').loc['GAP_stata_agent_zs']
    assert np.isfinite(result.se_cr1_pp)
    assert result.se_cr1_pp == pytest.approx(0, abs=1e-10)


@pytest.fixture
def selected_files(tmp_path, monkeypatch):
    monkeypatch.setattr(export, 'ROOT', tmp_path)
    prompt = 'Use the input to estimate the mean.\n'
    record = tmp_path / 'runs' / 'selected' / 'attempt_0001' / 'record.json'
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps(dict(prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())))
    (record.parent / 'prompt_attempt1.txt').write_text(prompt)
    # Run outputs can contain leading-dash filenames. They are safe inside a named run folder.
    (record.parent / '--help.log').write_text('Command usage output.\n')
    runs = pd.DataFrame([dict(record_path=record.relative_to(tmp_path).as_posix(),
                             record_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
                             model='luna6', platform='python', agency='agent', task='example')])
    return tmp_path, runs


@pytest.mark.parametrize('value', ['../outside/record.json', '/absolute/record.json',
                                    'C:/absolute/record.json',
                                    'runs/../record.json', 'runs/line\nbreak/record.json',
                                    'runs/\\outside/record.json', '-option/record.json'])
def test_rejects_unsafe_paths(value):
    with pytest.raises(ValueError, match='path'):
        export.run_folder(value)


def test_selected_run_folder_and_prompt_coverage(selected_files):
    root, runs = selected_files
    assert len(export.validate_runs(runs)) == 1
    texts, mapping = export.prompt_data(runs)
    assert len(texts) == 1
    assert set(mapping.record_path) == set(runs.record_path)
    with pytest.raises(ValueError, match='duplicate'):
        export.validate_runs(pd.concat([runs, runs]))
    other = runs.copy()
    other.loc[0, 'record_path'] = 'runs/selected/attempt_0002/record.json'
    with pytest.raises(ValueError, match='Duplicate selected run folders'):
        export.validate_runs(pd.concat([runs, other]))
    (root / 'runs/selected/attempt_0001/prompt_attempt1.copy.txt').write_text('Different text')
    with pytest.raises(ValueError, match='Conflicting first prompt'):
        export.prompt_data(runs)


def test_overlapping_run_roots_are_rejected_before_reading_files():
    runs = pd.DataFrame(dict(record_path=['runs/a/record.json', 'runs/a-b/record.json',
                                          'runs/a/nested/record.json']))
    with pytest.raises(ValueError, match='overlap'):
        export.validate_runs(runs)


def test_saved_first_prompt_takes_precedence_over_later_repair_prompt(selected_files):
    root, runs = selected_files
    record = root / runs.record_path.iloc[0]
    first = (record.parent / 'prompt_attempt1.txt').read_text()
    later = 'Repair the previous code.\n'
    record.write_text(json.dumps(dict(prompt=later, prompt_sha256=hashlib.sha256(later.encode()).hexdigest())))
    texts, mapping = export.prompt_data(runs)
    assert texts == {hashlib.sha256(first.encode()).hexdigest(): first}
    assert mapping.prompt_sha256.iloc[0] == hashlib.sha256(first.encode()).hexdigest()


def test_archive_membership_and_record_hashes(selected_files):
    root, runs = selected_files
    (root / 'export/runs').mkdir(parents=True)
    group, (folders, records) = next(iter(export.validate_runs(runs).items()))
    assert export.check_sources(folders) == 3
    name, count, size = export.archive(group, folders, root / 'export', records)
    assert count == 1 and size > 0
    target = root / 'export/runs' / name
    export.validate_archive(target, folders, records)
    with pytest.raises(ValueError, match='coverage'):
        export.validate_archive(target, folders, {**records, 'runs/selected/missing.json': '0' * 64})
    with pytest.raises(ValueError, match='hash'):
        export.validate_archive(target, folders, dict.fromkeys(records, '0' * 64))
    bad = root / 'unsafe.tar.xz'
    with tarfile.open(bad, 'w:xz') as handle:
        member = tarfile.TarInfo('../escaped.txt')
        member.size = 1
        handle.addfile(member, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='path'):
        export.validate_archive(bad, folders, records)


def test_credential_checks_do_not_print_content(tmp_path, capsys):
    source = tmp_path / 'record.json'
    token = 'sk-' + 'SYNTHETICTESTVALUE' * 3
    source.write_text(json.dumps(dict(api_key=token)))
    with pytest.raises(ValueError, match='Possible credential') as caught:
        export.check_credentials(source)
    assert token not in str(caught.value)
    output = capsys.readouterr()
    assert token not in output.out + output.err


def test_symbolic_links_cannot_enter_run_archives(selected_files):
    root, runs = selected_files
    (root / 'runs/selected/linked').symlink_to(root / 'runs/selected/attempt_0001/record.json')
    with pytest.raises(ValueError, match='Links'):
        export.check_sources(runs.record_path.map(export.run_folder).tolist())


def test_export_analysis_option_and_local_output_contents(selected_files, monkeypatch):
    root, runs = selected_files
    analysis = root / 'analysis-source'
    (analysis / 'data').mkdir(parents=True)
    (analysis / 'tables').mkdir()
    runs.to_csv(analysis / 'data/runs.csv', index=False)
    runs[['record_path']].assign(cost_usd=np.nan).to_csv(analysis / 'data/run_pricing.csv', index=False)
    (root / 'input.csv').write_text('x\n1\n')
    catalog = dict(example=dict(data_path=str(root / 'input.csv'), reference=dict(estimate=1)))
    (analysis / 'data/catalog.json').write_text(json.dumps(catalog))
    (root / 'code/paper').mkdir(parents=True)
    shutil.copyfile(Path(reproduce.__file__), root / 'code/paper/data_repository_reproduce.py')
    out = root / 'export'
    monkeypatch.setattr(sys, 'argv', ['data_repository.py', str(out), '--analysis', str(analysis)])
    export.main()
    assert {p.name for p in out.iterdir()} == {'analysis', 'tasks', 'prompts', 'runs', 'code', 'CONTENTS.txt'}
    assert (out / 'tasks/example/input.csv').read_text() == 'x\n1\n'
    assert json.loads((out / 'tasks/example/reference.json').read_text()) == {'estimate': 1}
    assert len(pd.read_csv(out / 'prompts/runs.csv')) == 1
    assert len(list((out / 'runs').glob('*.tar.xz'))) == 1
    assert {p.name for p in (out / 'code').iterdir()} == {'reproduce.py', 'requirements.txt'}

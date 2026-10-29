"""Local export and reproduction checks with small synthetic run collections."""
from itertools import product
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

import data_repository as export
import data_repository_references as references
import data_repository_reproduce as reproduce
import data_repository_score as score


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
    name, count, size, extracted, files = export.archive(group, folders, root / 'export', records)
    assert count == 1 and size > 0 and files == 3 and extracted > 0
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


def export_fixture(root, runs, monkeypatch):
    """A test project with one run, one task, the code files, and a column dictionary for the fixture files."""
    analysis = root / 'analysis-source'
    (analysis / 'data').mkdir(parents=True)
    (analysis / 'tables').mkdir()
    runs.to_csv(analysis / 'data/runs.csv', index=False)
    runs[['record_path']].assign(cost_usd=np.nan).to_csv(analysis / 'data/run_pricing.csv', index=False)
    (root / 'input.csv').write_text('x\n1\n')
    catalog = dict(example=dict(data_path=str(root / 'input.csv'), reference=dict(estimate=1)))
    (analysis / 'data/catalog.json').write_text(json.dumps(catalog))
    (root / 'code/paper').mkdir(parents=True)
    for module in (reproduce, score, references):
        shutil.copyfile(Path(module.__file__), root / 'code/paper' / Path(module.__file__).name)
    columns = root / 'columns.csv'
    pd.DataFrame([dict(file=name, column=column, description='Fixture column.')
                  for name in ('runs.csv', 'run_pricing.csv')
                  for column in pd.read_csv(analysis / 'data' / name).columns]).to_csv(columns, index=False)
    monkeypatch.setattr(export, 'COLUMNS', columns)
    monkeypatch.setattr(export, 'REFERENCE_VALIDATION', root / 'missing.csv')
    monkeypatch.setattr(export, 'CODE_FILES', {name: f'code/paper/{Path(m.__file__).name}' for name, m in
                                               (('reproduce.py', reproduce), ('score.py', score),
                                                ('run_references.py', references))})
    # source/ holds the study code, so the test root is a git repository. Untracked files are exported too.
    subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
    subprocess.run(['git', 'add', 'code/paper/data_repository_reproduce.py'], cwd=root, check=True)
    return analysis


def test_export_analysis_option_and_local_output_contents(selected_files, monkeypatch):
    root, runs = selected_files
    analysis = export_fixture(root, runs, monkeypatch)
    out = root / 'export'
    monkeypatch.setattr(sys, 'argv', ['data_repository.py', str(out), '--analysis', str(analysis)])
    export.main()
    assert {p.name for p in out.iterdir()} == {'analysis', 'tasks', 'prompts', 'runs', 'code', 'source',
                                               'CONTENTS.txt', 'README.md', '.zenodo.json'}
    assert sorted(p.relative_to(out / 'source').as_posix() for p in (out / 'source').rglob('*') if p.is_file()) == [
        'code/paper/data_repository_references.py', 'code/paper/data_repository_reproduce.py',
        'code/paper/data_repository_score.py']
    assert (out / 'tasks/example/input.csv').read_text() == 'x\n1\n'
    assert json.loads((out / 'tasks/example/reference.json').read_text()) == {'estimate': 1}
    assert len(pd.read_csv(out / 'prompts/runs.csv')) == 1
    assert len(list((out / 'runs').glob('*.tar.xz'))) == 1
    assert {p.name for p in (out / 'code').iterdir()} == {'reproduce.py', 'score.py', 'run_references.py',
                                                         'scoring_spec.json', 'requirements.txt'}
    assert {p.name for p in (out / 'analysis').iterdir()} == {'runs.csv', 'run_pricing.csv', 'columns.csv'}
    readme = (out / 'README.md').read_text()
    assert export.DOI in readme and 'tar -xJf' in readme and (out / 'CONTENTS.txt').read_text() in readme
    assert 'linearmodels==6.1' in (out / 'code/requirements.txt').read_text()
    assert json.loads((out / '.zenodo.json').read_text())['title'].startswith('Data and code for')
    # A second export can reuse the archives: same bytes, same checks.
    again = root / 'export-again'
    monkeypatch.setattr(sys, 'argv', ['data_repository.py', str(again), '--analysis', str(analysis),
                                      '--reuse-runs', str(out / 'runs')])
    export.main()
    for archive in (out / 'runs').glob('*.tar.xz'):
        assert (again / 'runs' / archive.name).read_bytes() == archive.read_bytes()


def test_reused_archive_must_match_the_selected_records(selected_files, tmp_path):
    root, runs = selected_files
    (root / 'export/runs').mkdir(parents=True)
    group, (folders, records) = next(iter(export.validate_runs(runs).items()))
    export.archive(group, folders, root / 'export', records)
    (root / 'reuse/runs').mkdir(parents=True)
    with pytest.raises(ValueError, match='hash'):
        export.reuse_archive(group, folders, root / 'reuse', dict.fromkeys(records, '0' * 64), root / 'export/runs')
    assert not list((root / 'reuse/runs').iterdir())
    with pytest.raises(ValueError, match='No archive'):
        export.reuse_archive(group, folders, root / 'reuse', records, root / 'nowhere')


def test_scored_outputs_enter_the_archive_check(selected_files):
    root, runs = selected_files
    output = root / 'runs/selected/attempt_0001/result.json'
    output.write_text('{"estimates": []}')
    runs = runs.assign(output_path=output.relative_to(root).as_posix(),
                       output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    (_, (_, records)), = export.validate_runs(runs).items()
    assert records[runs.output_path.iloc[0]] == runs.output_sha256.iloc[0]
    with pytest.raises(ValueError, match='output hash'):
        export.validate_runs(runs.assign(output_sha256='0' * 64))


def test_column_dictionary_must_match_the_files(tmp_path):
    data = tmp_path / 'runs.csv'
    data.write_text('a,b\n1,2\n')
    dictionary = tmp_path / 'columns.csv'
    dictionary.write_text('file,column,description\nruns.csv,a,First.\nruns.csv,b,Second.\n')
    export.check_columns([data], dictionary)
    dictionary.write_text('file,column,description\nruns.csv,a,First.\nruns.csv,c,Other.\n')
    with pytest.raises(ValueError, match="missing \\['b'\\].*\\['c'\\]"):
        export.check_columns([data], dictionary)


def test_project_column_dictionary_covers_the_six_model_files():
    data = export.ROOT / 'data/analysis/six_models/data'
    if not (data / 'runs.csv').is_file():
        pytest.skip('six-model analysis not present')
    export.check_columns([data / name for name in export.DOCUMENTED])


def test_privacy_check_rejects_serials_and_home_paths(tmp_path):
    path = tmp_path / 'file.log'
    path.write_text('Stata license: test\nSerial number: SERIAL-REDACT-ME\n')
    export.check_privacy(path)
    path.write_text('Serial number: 123456789012\n')
    with pytest.raises(ValueError, match='serial'):
        export.check_privacy(path)
    path.write_text('python ' + '/' + 'Users/someone/project/run.py\n')
    with pytest.raises(ValueError, match='Home-directory'):
        export.check_privacy(path)
    export.check_privacy(path, home=False)


# ---------------------------------------------------------------- code/score.py
def scoring_root(tmp_path):
    """A repository with one task, an SE rule, one wrong-method alternative, and an empty runs list."""
    (tmp_path / 'tasks/t1').mkdir(parents=True)
    (tmp_path / 'tasks/t1/reference.json').write_text(json.dumps(dict(estimates=[dict(term='b', estimate=1.0)])))
    (tmp_path / 'code').mkdir()
    (tmp_path / 'code/scoring_spec.json').write_text(json.dumps(dict(tasks=dict(t1=dict(
        se_scored=True, se_terms=dict(b=dict(rel_tol=0.001, accepted_values=[0.5])))))))
    (tmp_path / 'code/scoring_alternatives.json').write_text(json.dumps(dict(tasks=dict(t1=dict(alternatives=[
        dict(term='b', kind='estimate', method='wrong method', value=1.004)])))))
    (tmp_path / 'analysis').mkdir()
    return tmp_path


def add_run(root, rows, name, platform, output_text, primary, category, ok=True, output_name=None):
    attempt = root / 'data/runs' / name / 'attempt_0001'
    attempt.mkdir(parents=True)
    record = attempt / 'run_record.json'
    record.write_text(json.dumps(dict(executions=[dict(ok=ok, wall_seconds=1.0, stdout='')])))
    output = attempt / (output_name or ('result.json' if platform != 'stata' else 'run.log'))
    if output_text is not None:
        output.write_text(output_text)
    rows.append(dict(position_id=name, task='t1', platform=platform,
                     record_path=record.relative_to(root).as_posix(),
                     record_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
                     output_path=output.relative_to(root).as_posix(),
                     output_sha256=hashlib.sha256(output.read_bytes()).hexdigest() if output.is_file() else '',
                     primary=primary, failure_category=category))
    return attempt


def result(estimate, se):
    return json.dumps(dict(estimates=[dict(term='b', estimate=estimate, se=se)]))


def test_score_reproduces_each_outcome_type(tmp_path, capsys):
    root, rows = scoring_root(tmp_path), []
    add_run(root, rows, 'success', 'python', result(1.0, 0.5), True, 'Success')
    add_run(root, rows, 'alternative', 'python', result(1.0035, 0.5), False, 'Silent failure')
    add_run(root, rows, 'outside', 'python', result(1.02, 0.5), False, 'Silent failure')
    add_run(root, rows, 'se_outside', 'r', result(1.0, 0.51), False, 'Silent failure')
    add_run(root, rows, 'se_rounded', 'r', '{"estimates": [{"term": "b", "estimate": 1.0, "se": 0.50049}]}',
            True, 'Success')
    add_run(root, rows, 'boolean', 'python', result(True, 0.5), False, 'Silent failure')
    add_run(root, rows, 'no_output', 'python', None, False, 'Loud failure', ok=False)
    add_run(root, rows, 'repaired', 'python', '```json\n' + result(1.0, 0.5) + '\n```', True, 'Success')
    add_run(root, rows, 'stata', 'stata', 'ANSWER: b estimate=1.000000 se=0.500000\n', True, 'Success')
    attempt = add_run(root, rows, 'stata_conflict', 'stata', 'ANSWER: b estimate=1.000000 se=0.500000\n',
                      False, 'Loud failure')
    # The final execution window comes from the execution files; a workspace file inside it is a second source.
    (attempt / 'execution_attempt1.stdout').write_text('')
    (attempt / 'execution_attempt1.json').write_text('{}')
    (attempt / 'final_workspace').mkdir()
    (attempt / 'final_workspace/answers.txt').write_text('ANSWER: b estimate=1.2 se=0.5\n')
    pd.DataFrame(rows).to_csv(root / 'analysis/runs.csv', index=False)
    assert score.main(['--root', str(root), '--out', str(root / 'rescored.csv')]) == 0
    rescored = pd.read_csv(root / 'rescored.csv').set_index('position_id')
    assert rescored.match.all()
    assert 'closer to the wrong method value' in rescored.loc['alternative', 'notes']
    assert 'conflicting values' in rescored.loc['stata_conflict', 'notes']
    # A stored outcome that the rule does not give is reported, and so is a changed output file.
    rows[0]['failure_category'] = 'Silent failure'
    pd.DataFrame(rows).to_csv(root / 'analysis/runs.csv', index=False)
    assert score.main(['--root', str(root)]) == 1
    rows[0]['failure_category'] = 'Success'
    (root / rows[0]['output_path']).write_text(result(1.0, 0.5) + ' ')
    pd.DataFrame(rows).to_csv(root / 'analysis/runs.csv', index=False)
    assert score.main(['--root', str(root)]) == 1
    assert 'hash differs: success output' in capsys.readouterr().out


def test_score_uses_the_project_rules_without_the_alternatives_file(tmp_path):
    root, rows = scoring_root(tmp_path), []
    (root / 'code/scoring_alternatives.json').unlink()
    add_run(root, rows, 'alternative', 'python', result(1.0035, 0.5), True, 'Success')
    pd.DataFrame(rows).to_csv(root / 'analysis/runs.csv', index=False)
    assert score.main(['--root', str(root)]) == 0


def test_score_rounding_decimals_and_alternative_rule():
    assert score.round_decimals(0.4892) == 4
    assert score.round_decimals(0.4892, '0.48920') == 5
    # 1.004 and the reference 1.0 agree at 2 decimals, so a 2-decimal value is never failed by the alternative.
    alternative = [dict(value=1.004, method='m')]
    assert score.closer_alternative(1.0, [1.0], alternative, 2) is None
    assert score.closer_alternative(1.003, [1.0], alternative, 3)['method'] == 'm'


# ---------------------------------------------------------------- code/run_references.py
def test_reference_outputs_of_both_forms_map_to_terms():
    rows = references.term_values('t', dict(estimates=[dict(term='b', estimate=1.5, se=0.2), dict(term='c', estimate=2)]))
    assert rows == {'b': 1.5, 'b:se': 0.2, 'c': 2}
    flat = references.term_values('t', dict(platform='python', x=0.5, se_x=0.1, coef_treat=0.3, n_obs=10))
    assert flat == {'x': 0.5, 'x:se': 0.1, 'treat': 0.3, 'n_obs': 10}
    log = 'ANSWER: b estimate=1.25 se=.\n. display\nJSON_RESULT: {"a":"  0.5","se_a":"0.1"}\n'
    assert references.stata_output(log) == dict(estimates=[dict(term='b', estimate='1.25', se=None)])
    assert references.stata_output('JSON_RESULT: {"a":"  0.5","se_a":"0.1"}\n') == {'a': 0.5, 'se_a': 0.1}


def test_reference_comparison_uses_the_paper_tolerances():
    reference = dict(estimates=[dict(term='accepts_zero', estimate=1), dict(term='b', estimate=2.0)])
    spec = dict(tasks=dict(broad_ar_confidence_set=dict(se_scored=True, se_terms=dict(
        b=dict(rel_tol=0.01, accepted_values=[0.5, 0.6])))))
    passed, rel, absolute, se_rel, notes = references.compare(
        'broad_ar_confidence_set', {'accepts_zero': 1.0, 'b': 2.015, 'b:se': 0.601}, reference, spec)
    assert passed and absolute == pytest.approx(0.015) and se_rel == pytest.approx(0.001 / 0.6)
    passed, *_, notes = references.compare('broad_ar_confidence_set', {'accepts_zero': 1.004, 'b': 2.0},
                                           reference, spec)
    assert not passed and 'standard error not reported' in notes[-1]


def test_reference_runner_runs_python_programs_from_the_root(tmp_path, capsys):
    (tmp_path / 'tasks/t1').mkdir(parents=True)
    (tmp_path / 'tasks/t1/input.csv').write_text('x\n2\n')
    (tmp_path / 'tasks/t1/reference.json').write_text(json.dumps(dict(estimates=[dict(term='x', estimate=2.0)])))
    program = tmp_path / 'source/reference_implementations/manual/t1/manual_python.py'
    program.parent.mkdir(parents=True)
    program.write_text('import json\nfrom pathlib import Path\n'
                       'x = float(Path("tasks/t1/input.csv").read_text().split()[1])\n'
                       'print(json.dumps({"estimates": [{"term": "x", "estimate": x}]}))\n')
    assert references.main(['--root', str(tmp_path), '--languages', 'python']) == 0
    program.write_text('print({"x": 3.0}.__repr__().replace("\'", \'"\'))\n')
    assert references.main(['--root', str(tmp_path), '--languages', 'python']) == 1
    assert 'x: 3.0 against 2.0' in capsys.readouterr().out

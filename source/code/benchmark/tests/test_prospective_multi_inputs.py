"""Offline checks for tasks with several input files."""
from pathlib import Path
import shutil
import json

import pytest

from test_prospective_collect import fixture, p


@pytest.fixture(autouse=True)
def clear_stop():
    p.STOP.clear()
    yield
    p.STOP.clear()


def multi_fixture(tmp_path, agency='chatbot'):
    parts = fixture(tmp_path, agency)
    extra = tmp_path / 'other.csv'
    extra.write_text('z\n2\n')
    parts[1]['protocol']['tasks']['toy']['input_files'] = {'other.csv': str(extra)}
    return parts


def test_all_inputs_are_available_and_preserved(tmp_path):
    job, manifest, runtime, quotas, call, execute, _, _ = multi_fixture(tmp_path)
    def check(script, platform, cwd, destination, number):
        assert (cwd / 'other.csv').read_bytes() == (tmp_path / 'other.csv').read_bytes()
        return execute(script, platform, cwd, destination, number)
    row = p.run_job(job, manifest, tmp_path / 'runs', runtime, quotas, call, check)
    assert row['outcome']['primary_success'] is True
    assert (Path(row['attempt_directory']) / 'final_workspace' / 'other.csv').is_file()


@pytest.mark.parametrize('defect', ['changed', 'missing', 'symlink'])
@pytest.mark.parametrize('when', ['provider', 'execution'])
def test_extra_input_damage_holds_collection(tmp_path, defect, when):
    job, manifest, runtime, quotas, call, execute, _, _ = multi_fixture(tmp_path, 'agent')
    def damage(cwd):
        path = cwd / 'other.csv'
        path.unlink()
        if defect == 'changed':
            path.write_text('changed')
        elif defect == 'symlink':
            path.symlink_to(tmp_path / 'other.csv')
    def provider(text, cwd, job, destination, number):
        (cwd / 'run.py').write_text('pass')
        if when == 'provider':
            damage(cwd)
        return call(text, cwd, job, destination, number)
    def execution(script, platform, cwd, destination, number):
        result = execute(script, platform, cwd, destination, number)
        if when == 'execution':
            damage(cwd)
        return result
    row = p.run_job(job, manifest, tmp_path / 'runs', runtime, quotas, provider, execution)
    assert row['hold_reason'] == 'integrity'
    assert row['outcome']['primary_success'] is None


def test_self_repair_restores_all_inputs(tmp_path):
    job, manifest, runtime, quotas, call, execute, calls, _ = multi_fixture(tmp_path, 'self_repair')
    def execution(script, platform, cwd, destination, number):
        assert (cwd / 'other.csv').read_bytes() == (tmp_path / 'other.csv').read_bytes()
        if number == 1:
            (cwd / 'scratch').write_text('remove before retry')
            return dict(ok=False, returncode=1, stderr='failed')
        assert not (cwd / 'scratch').exists()
        return execute(script, platform, cwd, destination, number)
    row = p.run_job(job, manifest, tmp_path / 'runs', runtime, quotas, call, execution)
    assert row['outcome']['primary_success'] is True and len(calls) == 2


@pytest.mark.parametrize('remove_workspace', [False, True])
@pytest.mark.parametrize('damage_saved', [False, True])
def test_resume_checks_every_input(tmp_path, remove_workspace, damage_saved):
    job, manifest, runtime, quotas, call, execute, calls, _ = multi_fixture(tmp_path)
    row = p.run_job(job, manifest, tmp_path / 'runs', runtime, quotas, call,
                    lambda *args: dict(ok=False, returncode=-9, timed_out=False))
    cwd = Path(row['cwd'])
    target = Path(row['held_workspace']) if remove_workspace else cwd
    if damage_saved:
        (target / 'other.csv').write_text('changed')
    if remove_workspace:
        shutil.rmtree(cwd)
    p.STOP.clear()
    if damage_saved:
        with pytest.raises(ValueError, match='input'):
            p.run_job(job, manifest, tmp_path / 'runs', runtime, quotas, call, execute)
    else:
        resumed = p.run_job(job, manifest, tmp_path / 'runs', runtime, quotas, call, execute)
        assert resumed['outcome']['primary_success'] is True
        assert (cwd / 'other.csv').read_bytes() == (tmp_path / 'other.csv').read_bytes()
    assert len(calls) == 1


@pytest.mark.parametrize('mapping', [
    {'../outside.csv': '/tmp/input'}, {'/absolute.csv': '/tmp/input'},
    {'other.csv': 'relative.csv'}, {'input.csv': '/tmp/different'},
    {'run.py': '/tmp/input'}, {'result.json': '/tmp/input'}, [],
])
def test_invalid_input_mapping_is_rejected(mapping):
    with pytest.raises(ValueError):
        p.task_input_files(dict(data_path='/tmp/primary', input_files=mapping))


def test_prepare_pins_every_input(tmp_path, monkeypatch):
    job, manifest, runtime, *_ = multi_fixture(tmp_path)
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps(manifest['protocol']['tasks']))
    prompt = tmp_path / 'prompt.txt'
    prompt.write_text('Exact task prompt')
    decision = tmp_path / 'decision.md'
    decision.write_text('Test decision')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(catalog_path=str(catalog), jobs=[job],
        decision_record=str(decision), prompt_paths={p.prompt_key(job): str(prompt)})))
    runtime.c.command = lambda *args: (['test-provider'], {})
    monkeypatch.setattr(p, 'load_runtime', lambda *args: runtime)
    monkeypatch.setattr(p, 'runtime_sources', lambda *args: [])
    monkeypatch.setattr(p, 'provider_cli_versions', lambda *args: {})
    monkeypatch.setattr(p, 'native_environment', lambda *args: {})
    prepared = p.prepare(config, tmp_path / 'manifest.json')
    for name in ('data.csv', 'other.csv'):
        source = tmp_path / name
        assert prepared['protocol']['source_hashes'][str(source)] == p.sha(source)
    p.verify(prepared)
    (tmp_path / 'other.csv').write_text('changed source')
    with pytest.raises(ValueError, match='Pinned source changed'):
        p.verify(prepared)

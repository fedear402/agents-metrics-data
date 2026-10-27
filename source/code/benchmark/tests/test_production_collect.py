"""Reject duplicate or already occupied production positions before preparation."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import production_collect as collect


def job(rep=0):
    return dict(block='paper22_reps1to5' if rep < 5 else 'paper22_reps6to10',
                task='toy', model='luna', platform='python', prompt='zero_shot', condition='chatbot',
                output_format='json', rep=rep % 5, global_rep=rep)


def test_duplicate_global_position_is_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        collect.validate_jobs([job(), job()])


@pytest.mark.parametrize('value', [True, 1.0])
def test_repetition_metadata_requires_integers(value):
    row = dict(job(1), global_rep=value)
    with pytest.raises(ValueError):
        collect.validate_jobs([row])


def test_mixed_global_repetition_groups_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, 'RUNS', tmp_path / 'runs')
    jobs = tmp_path / 'jobs.json'
    jobs.write_text(json.dumps([job(1), job(6)]))
    with pytest.raises(ValueError, match='separate manifests'):
        collect.prepare(jobs, 'mixed')


@pytest.mark.parametrize('status', ['retained', 'conditional_retained', 'held', 'active', 'planned'])
def test_only_unplanned_inventory_positions_can_be_prepared(tmp_path, monkeypatch, status):
    design = tmp_path / 'design'
    design.mkdir()
    monkeypatch.setattr(collect, 'DESIGN', design)
    monkeypatch.setattr(collect, 'RUNS', tmp_path / 'runs')
    jobs = tmp_path / 'jobs.json'
    jobs.write_text(json.dumps([job()]))
    (design / 'catalog.json').write_text(json.dumps({'toy': {}}))
    (design / 'prompt_paths.json').write_text(json.dumps({collect.prompt_key(job()): '/unused.txt'}))
    (design / 'planned_positions.json').write_text(json.dumps([dict(job(), status=status)]))
    def no_runtime_before_guard(*args, **kwargs):
        raise AssertionError('The collector was called before the occupied-position check.')
    monkeypatch.setattr(collect.subprocess, 'run', no_runtime_before_guard)
    with pytest.raises(ValueError, match='(?i)(unplanned|occupied|assigned|retained)'):
        collect.prepare(jobs, 'occupied')


def test_changed_ready_source_stops_assembly_before_rendering(tmp_path, monkeypatch):
    task = 'added'
    folder = tmp_path / 'tasks' / task
    folder.mkdir(parents=True)
    reference = folder / 'example.R'
    reference.write_text('changed after validation')
    data = folder / 'input.csv'
    data.write_text('x\n1\n')
    spec = dict(reference={'estimates': []}, data_path=str(data))
    (folder / 'task.json').write_text(json.dumps(spec))
    (folder / 'ready.json').write_text(json.dumps(dict(passed=True, files_sha256={'example.R': '0' * 64})))
    accepted, prompts = tmp_path / 'accepted.json', tmp_path / 'prompts.json'
    accepted.write_text(json.dumps({f'old{i}': {} for i in range(18)}))
    prompts.write_text('{}')
    monkeypatch.setattr(collect, 'ACCEPTED', accepted)
    monkeypatch.setattr(collect, 'OLD_PROMPTS', prompts)
    monkeypatch.setattr(collect, 'TASKS', folder.parent)
    monkeypatch.setattr(collect, 'NEW_TASKS', (task,))
    monkeypatch.setattr(collect, 'pilot', lambda name: (spec, '/unused.txt'))
    def no_render(*args):
        raise AssertionError('Rendering started before source readiness was verified.')
    monkeypatch.setattr(collect, 'rendered_prompt', no_render)
    with pytest.raises(ValueError, match='(?i)(hash|source|changed|ready)'):
        collect.assemble()

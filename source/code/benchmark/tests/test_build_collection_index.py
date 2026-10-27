import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('collection_index', Path(__file__).parents[1] / 'build_collection_index.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_preserves_failure_and_normalizes_only_run_specific_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    base = tmp_path / 'data/collections/example/runs/attempt_0001'
    base.mkdir(parents=True)
    path = base / 'run_record.json'
    path.write_text(json.dumps(dict(job=dict(task='task', model='sonnet', platform='python',
                                           prompt='zero', condition='self_repair', rep=0),
                                    outcome=dict(primary_success=False))))
    (base / 'prompt_attempt1.txt').write_text('An exact prompt')
    (base / 'command_attempt1.json').write_text(json.dumps(dict(
        argv=['/bin/codex', 'exec', '-C', '/tmp/one', '-c', 'features.shell_tool=false'], timeout_seconds=180)))
    row, signature = m.inspect(path)
    assert row['original_primary'] is False
    assert row['prompt'] == 'zero_shot'
    assert signature['argv'] == ['codex', 'exec', '-C', '<run-specific-path>', '-c', 'features.shell_tool=false']
    assert row['saved_prompt_sha256']
    assert 'current_selected' not in row


def test_census_does_not_follow_compatibility_aliases(tmp_path):
    base = tmp_path / 'data'
    run = base / 'collections/example/runs'
    run.mkdir(parents=True)
    (run / 'run_record.json').write_text('{}')
    (base / 'alias').symlink_to(run, target_is_directory=True)
    assert list(m.physical_records(base)) == [run / 'run_record.json']


def test_receipts_resolve_moved_and_archived_records(tmp_path):
    root = tmp_path / 'repo'
    archive = tmp_path / 'archive'
    migration = root / 'data/inventory/storage_migration_2026-09-18.json'
    migration.parent.mkdir(parents=True)
    migration.write_text(json.dumps({'items': [{'source': 'quality_reports/old', 'destination': 'data/collections/old'}]}))
    live = root / 'data/collections/old/run_record.json'
    live.parent.mkdir(parents=True)
    live.write_text('{}')
    receipt = root / 'data/maintenance/archive/archive_manifest.json'
    receipt.parent.mkdir(parents=True)
    archived = archive / 'data/runs/old/record.json'
    archived.parent.mkdir(parents=True)
    archived.write_text('{}')
    receipt.write_text(json.dumps({'archive_root': str(archive), 'trees': {'data/runs/old': [
        {'path': 'record.json', 'kind': 'file'}]}}))
    evidence = m.PathEvidence(root)
    assert evidence.resolve('quality_reports/old/run_record.json')[0] == live
    assert evidence.resolve('data/runs/old/record.json')[0] == archived
    assert archived in evidence.archived_records


def test_current_selection_does_not_inherit_old_selection(tmp_path):
    selection, analysis = tmp_path / 'selected.csv', tmp_path / 'runs.csv'
    path = tmp_path / 'new/run_record.json'
    path.parent.mkdir()
    path.write_text('{}')
    m.write_csv(selection, [{'position_id': 'current', 'record_path': str(path), 'record_sha256': 'hash'}])
    m.write_csv(analysis, [{'position_id': 'current', 'record_path': str(path), 'model': 'luna'}])
    selected = m.load_selection(selection, analysis, m.PathEvidence(tmp_path))
    assert list(selected) == [path]
    assert selected[path]['model'] == 'luna'
    assert m.classify_record({}, False, {'disposition': 'selected'}, False)[0] == 'reusable_candidate_prior_selection'
    assert m.classify_record({}, False, {}, False)[0] == 'unresolved_candidate'


def test_current_selection_rejects_two_positions_with_same_response(tmp_path):
    selection, analysis = tmp_path / 'selected.csv', tmp_path / 'runs.csv'
    rows = [{'position_id': position, 'record_path': str(tmp_path / 'record.json')} for position in ('one', 'two')]
    m.write_csv(selection, rows)
    m.write_csv(analysis, rows)
    with pytest.raises(ValueError, match='one physical record'):
        m.load_selection(selection, analysis, m.PathEvidence(tmp_path))


def test_malformed_command_does_not_destroy_record_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    record = tmp_path / 'record.json'
    record.write_text('{}')
    (tmp_path / 'command.json').write_text('unfinished')
    row, _ = m.inspect(record)
    assert row['parse_error'] == ''
    assert row['command_parse_error']


def test_centralized_collections_retain_campaign_name(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    relative = 'data/runs/collections/example/runs/attempt_0001/record.json'
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    row, _ = m.inspect(path)
    assert row['campaign'] == 'data/runs/collections/example'
    assert m.raw_storage_root(relative) == ('data/runs/collections/example', 'already_in_raw_runs')


def test_split_archive_and_summary_destinations_remain_resolvable(tmp_path):
    root = tmp_path / 'repo'
    archive = tmp_path / 'archive'
    archived = archive / 'data/runs/old/run_record.json'
    archived.parent.mkdir(parents=True)
    archived.write_text('{}')
    receipt = root / 'data/maintenance/archive/archive_manifest.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({'archive_root': str(archive), 'trees': {'data/runs/old': [
        {'path': 'run_record.json', 'kind': 'file'}]}}))
    alias = root / 'data/inventory/record_path_aliases.csv'
    alias.parent.mkdir(parents=True)
    m.write_csv(alias, [{'source': 'data/runs/old', 'destination': 'data/collections/old_summaries'}])
    summary = root / 'data/collections/old_summaries/summary.csv'
    summary.parent.mkdir(parents=True)
    summary.write_text('summary')
    evidence = m.PathEvidence(root)
    assert evidence.resolve('data/runs/old/run_record.json')[0] == archived
    assert evidence.resolve('data/runs/old/summary.csv')[0] == summary

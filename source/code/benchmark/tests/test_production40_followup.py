"""Verify scoped archive copies and explicit removal using temporary files."""
import importlib.util
import json
import os
from pathlib import Path
import shutil

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'workflows/production40/archive_followup.py'
SPEC = importlib.util.spec_from_file_location('tested_followup', SOURCE)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


@pytest.fixture
def fixture(tmp_path):
    root, external = tmp_path / 'repo', tmp_path / 'external'
    index = root / archive.INDEX
    index.mkdir(parents=True)
    (root / 'old/tree').mkdir(parents=True)
    (root / 'old/tree/raw.json').write_bytes(b'{"preserve": 1}\n')
    (root / 'old/tree/raw.json').chmod(0o640)
    (root / 'old/tree').chmod(0o750)
    (root / 'old/plan.md').write_text('Original author decisions.\n')
    (root / 'live').mkdir()
    (root / 'live/input.csv').write_text('keep this\n')
    (root / 'old/tree/link').symlink_to(root / 'live', target_is_directory=True)
    (root / 'old/dangling').symlink_to('missing-relative-target')
    plan = index / 'plan.json'
    value = dict(approved_items=['old/plan.md', 'old/tree', 'old/dangling'],
                 protected_paths=['live/input.csv'], archive_root=str(external / 'batch'))
    plan.write_text(json.dumps(value))
    return dict(root=root, external=external, plan=plan, value=value, out=external / 'batch', index=index)


def run(fixture, phase):
    return archive.run(phase, fixture['plan'], fixture['root'], fixture['external'])


def test_file_directory_links_modes_and_resume(fixture):
    run(fixture, 'inventory')
    assert not fixture['out'].exists()
    run(fixture, 'copy')
    root, out = fixture['root'], fixture['out']
    assert (root / 'old/plan.md').is_file()
    assert (out / 'payload/old/tree/link').is_symlink()
    assert os.readlink(out / 'payload/old/tree/link') == str(root / 'live')
    assert os.readlink(out / 'payload/old/dangling') == 'missing-relative-target'
    for name in ('archive_manifest.json', 'copy_receipt.json'):
        assert (fixture['index'] / name).read_bytes() == (out / name).read_bytes()
    assert fixture['plan'].read_bytes() == (out / 'plan.json').read_bytes()
    run(fixture, 'copy')
    (root / 'old/plan.md').unlink()  # Model interruption after one verified removal.
    run(fixture, 'remove')
    run(fixture, 'remove')
    run(fixture, 'copy')
    assert not os.path.lexists(root / 'old/dangling')
    assert not (root / 'old/tree').exists()
    assert (root / 'live/input.csv').read_text() == 'keep this\n'
    assert (fixture['index'] / 'removal_receipt.json').read_bytes() == (out / 'removal_receipt.json').read_bytes()


@pytest.mark.parametrize('side', ['source', 'destination'])
def test_changed_later_item_prevents_all_removal(fixture, side):
    run(fixture, 'copy')
    base = fixture['root'] if side == 'source' else fixture['out'] / 'payload'
    (base / 'old/tree/raw.json').write_text('changed')
    with pytest.raises(ValueError, match='differs from inventory'):
        run(fixture, 'remove')
    assert (fixture['root'] / 'old/plan.md').is_file()
    assert (fixture['root'] / 'old/tree').is_dir()
    with pytest.raises(ValueError, match='differs from inventory'):
        run(fixture, 'copy')


@pytest.mark.parametrize('item', ['.', '../escape', '/absolute', 'live', 'live/input.csv', 'data/maintenance'])
def test_root_traversal_and_protected_overlap_rejected(fixture, item):
    fixture['value']['approved_items'] = [item]
    fixture['plan'].write_text(json.dumps(fixture['value']))
    with pytest.raises(ValueError):
        run(fixture, 'inventory')
    assert not (fixture['index'] / 'archive_manifest.json').exists()


def test_protected_symlink_target_and_overlapping_items_rejected(fixture):
    fixture['value']['protected_paths'] = ['old/tree/link/input.csv']
    fixture['value']['approved_items'] = ['live']
    fixture['plan'].write_text(json.dumps(fixture['value']))
    with pytest.raises(ValueError, match='protected dependency'):
        run(fixture, 'inventory')
    fixture['value']['protected_paths'] = ['live/input.csv']
    fixture['value']['approved_items'] = ['old', 'old/tree']
    fixture['plan'].write_text(json.dumps(fixture['value']))
    with pytest.raises(ValueError, match='items overlap'):
        run(fixture, 'inventory')


def test_source_and_destination_symlink_parents_rejected(fixture):
    (fixture['root'] / 'alias').symlink_to(fixture['root'] / 'old', target_is_directory=True)
    fixture['value']['approved_items'] = ['alias/plan.md']
    fixture['plan'].write_text(json.dumps(fixture['value']))
    with pytest.raises(ValueError, match='path ancestry'):
        run(fixture, 'inventory')
    fixture['value']['approved_items'] = ['old/plan.md']
    fixture['plan'].write_text(json.dumps(fixture['value']))
    run(fixture, 'inventory')
    fixture['out'].mkdir(parents=True)
    (fixture['out'] / 'payload').symlink_to(fixture['root'], target_is_directory=True)
    with pytest.raises(ValueError, match='path ancestry'):
        run(fixture, 'copy')


def test_plan_changes_and_missing_sources_require_prior_verified_copy(fixture):
    run(fixture, 'inventory')
    fixture['plan'].write_text(json.dumps(fixture['value'], indent=2))
    with pytest.raises(ValueError, match='reviewed plan'):
        run(fixture, 'copy')
    fixture['plan'].write_text(json.dumps(fixture['value']))
    (fixture['root'] / 'old/plan.md').unlink()
    with pytest.raises(ValueError, match='differs from inventory'):
        run(fixture, 'copy')


def test_external_manifest_tamper_blocks_removal(fixture):
    run(fixture, 'copy')
    (fixture['out'] / 'archive_manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='metadata changed'):
        run(fixture, 'remove')
    assert (fixture['root'] / 'old/plan.md').is_file()


def test_receipt_tamper_is_not_overwritten_on_copy_resume(fixture):
    run(fixture, 'copy')
    (fixture['out'] / 'copy_receipt.json').write_text('{}')
    with pytest.raises(ValueError, match='metadata changed'):
        run(fixture, 'copy')


def test_control_temporary_symlink_is_never_followed(fixture):
    target = fixture['root'] / 'live/input.csv'
    (fixture['index'] / 'archive_manifest.json.tmp').symlink_to(target)
    with pytest.raises(ValueError, match='path ancestry'):
        run(fixture, 'inventory')
    assert target.read_text() == 'keep this\n'


def test_complete_interrupted_staging_copy_resumes(fixture):
    run(fixture, 'inventory')
    staging = fixture['out'] / 'payload/old/plan.md.copying'
    staging.parent.mkdir(parents=True)
    shutil.copy2(fixture['root'] / 'old/plan.md', staging)
    run(fixture, 'copy')
    assert not staging.exists()


def test_cli_defaults_to_inventory(fixture, monkeypatch):
    calls = []
    monkeypatch.setattr(archive, 'run', lambda *args: calls.append(args))
    monkeypatch.setattr('sys.argv', ['archive_followup.py', '--plan', str(fixture['plan'])])
    archive.main()
    assert calls == [('inventory', fixture['plan'])]


def test_history_beside_receipts_can_be_archived_without_moving_controls(fixture):
    history = fixture['index'] / 'doc_history'
    history.mkdir()
    (history / 'old.md').write_text('Preserved decisions')
    fixture['value']['approved_items'] = [str(history.relative_to(fixture['root']))]
    fixture['plan'].write_text(json.dumps(fixture['value']))
    run(fixture, 'copy')
    run(fixture, 'remove')
    assert fixture['plan'].is_file()
    assert (fixture['index'] / 'archive_manifest.json').is_file()
    assert not history.exists()

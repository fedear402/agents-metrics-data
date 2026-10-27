"""Archive checks use temporary trees and never access the external archive."""
import importlib.util
import csv
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'workflows/production40/cleanup.py'
SPEC = importlib.util.spec_from_file_location('production40_cleanup', SOURCE)
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


def setup_trees(tmp_path):
    root, out, index = tmp_path / 'project', tmp_path / 'external', tmp_path / 'index'
    for tree in cleanup.TREES:
        folder = root / tree
        folder.mkdir(parents=True)
        (folder / 'note.txt').write_bytes(b'historical note\r\n')
    data = root / 'data/collections'
    data.mkdir(parents=True)
    manifest = root / 'quality_reports/reports/manifest.json'
    manifest.write_bytes(b'{"path": "quality_reports/reports/input.csv"}\r\n')
    (data / 'manifest.json').symlink_to(manifest)
    (root / 'data/raw').mkdir()
    (root / 'data/raw/response.txt').write_bytes(b'raw provider response')
    (root / 'quality_reports/reports/runs').symlink_to('../../data/raw', target_is_directory=True)
    return root, out, index


def test_archive_preserves_raw_bytes_and_links(tmp_path):
    root, out, index = setup_trees(tmp_path)
    expected = (root / 'data/collections/manifest.json').read_bytes()
    cleanup.copy_archive(root, out, index)
    companion = root / 'data/collections/manifest.json'
    assert not companion.is_symlink()
    assert companion.read_bytes() == expected
    link = out / 'quality_reports/reports/runs'
    assert link.is_symlink()
    assert not (out / 'data').exists()
    assert (root / 'data/raw/response.txt').read_bytes() == b'raw provider response'
    cleanup.remove_originals(root, out, index)
    assert not (root / 'quality_reports/reports').exists()
    assert companion.read_bytes() == expected
    assert json.loads((index / 'copy_receipt.json').read_text())['originals_removed'] is True


def test_changed_archive_blocks_source_removal(tmp_path):
    root, out, index = setup_trees(tmp_path)
    cleanup.copy_archive(root, out, index)
    (out / 'archive/note.txt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs from inventory'):
        cleanup.remove_originals(root, out, index)
    assert all((root / tree).is_dir() for tree in cleanup.TREES)


def test_changed_source_blocks_all_source_removal(tmp_path):
    root, out, index = setup_trees(tmp_path)
    cleanup.copy_archive(root, out, index)
    (root / 'quality_reports/audits/new-note.txt').write_bytes(b'new')
    with pytest.raises(ValueError, match='differs from inventory'):
        cleanup.remove_originals(root, out, index)
    assert all((root / tree).is_dir() for tree in cleanup.TREES)


def test_archive_copies_resume_and_materialization_is_idempotent(tmp_path):
    root, out, index = setup_trees(tmp_path)
    cleanup.copy_archive(root, out, index)
    original = (index / 'materialized_companions.json').read_bytes()
    cleanup.copy_archive(root, out, index)
    assert (index / 'materialized_companions.json').read_bytes() == original


@pytest.mark.parametrize('dependency', ['selected_record', 'candidate_input'])
def test_publication_archive_rejects_current_dependencies(tmp_path, dependency):
    root, index = tmp_path / 'project', tmp_path / 'index'
    design = root / 'data/production/main'
    design.mkdir(parents=True)
    path = 'data/runs/codex_zero_shot/selected/run_record.json'
    selected = path if dependency == 'selected_record' else 'data/runs/current/run_record.json'
    with (design / 'baseline_runs.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['record_path'])
        writer.writeheader()
        writer.writerows({'record_path': selected} for _ in range(14400))
    (design / 'catalog.json').write_text('{}')
    if dependency == 'candidate_input':
        task = root / 'data/task_candidates/retained/task.json'
        task.parent.mkdir(parents=True)
        task.write_text(json.dumps({'input_files': {'input.csv': path}}))
    with pytest.raises(ValueError, match='Current selected source uses publication corpus'):
        cleanup.publication_exclusions(root, index)

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('centralize', Path(__file__).parents[1] / 'centralize_collection_data.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_preserves_raw_tree_hashes_and_relative_manifest_layout(tmp_path):
    base = tmp_path / 'quality_reports/reports/example/work'
    run = base / 'runs/protocol/job/attempt_0001'
    run.mkdir(parents=True)
    (run / 'run_record.json').write_text('{"success": false}')
    (run / 'run.py').write_text('print(42)')
    (base / 'manifest.json').write_text('{"runs": "runs"}')
    (base / 'runner.py').write_text('pass')
    plan = m.make_plan(tmp_path)
    assert len(plan['items']) == 2
    for item in plan['items']:
        m.verify_item(tmp_path, item, moved=False)
        m.move_one(tmp_path, item)
        m.move_one(tmp_path, item)
        m.verify_item(tmp_path, item)
    m.companion_links(tmp_path)
    manifest = (base / 'manifest.json').resolve()
    assert (manifest.parent / 'runs/protocol/job/attempt_0001/run_record.json').is_file()
    assert not (manifest.parent / 'runs/protocol/job/attempt_0001/run.py').is_symlink()
    assert (manifest.parent / 'runner.py').resolve() == base / 'runner.py'
    assert not (base / 'runner.py').is_symlink()
    assert (run / 'run_record.json').read_text() == '{"success": false}'


def test_rejects_overwrite_and_changed_bytes(tmp_path):
    base = tmp_path / 'quality_reports/reports/example'
    base.mkdir(parents=True)
    (base / 'sample.csv').write_text('x\n1\n')
    item = m.make_plan(tmp_path)['items'][0]
    destination = tmp_path / item['destination']
    destination.parent.mkdir(parents=True)
    destination.write_text('existing')
    with pytest.raises(FileExistsError):
        m.move_one(tmp_path, item)
    assert (base / 'sample.csv').read_text() == 'x\n1\n'
    (base / 'sample.csv').write_text('changed')
    with pytest.raises(ValueError):
        m.verify_item(tmp_path, item, moved=False)


def test_promotes_reverse_companion_without_overwriting_other_files(tmp_path):
    base = tmp_path / 'quality_reports/reports/example'
    base.mkdir(parents=True)
    source = base / 'change.diff'
    source.write_text('exact diff')
    m.companion_links(tmp_path)
    item = m.make_plan(tmp_path)['items'][0]
    m.move_one(tmp_path, item)
    m.verify_item(tmp_path, item)
    assert (tmp_path / item['destination']).read_text() == 'exact diff'

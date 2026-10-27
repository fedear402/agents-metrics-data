"""Check scoped process recognition for the unlimited queue entry point."""
import importlib.util
from pathlib import Path
import shlex
import sys

def test_supervisor_recognizes_unlimited_queue_and_preserves_scope(tmp_path):
    path=Path(__file__).resolve().parents[1]/'collection_supervisor_unlimited.py'
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('unlimited_supervisor_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    entry=dict(index=str(tmp_path/'index.json'),batches=['one','two'])
    args=['python',str(path.with_name('production_queue_unlimited.py')),'--index',entry['index'],'--batches','one','two']
    processes=[(123,shlex.join(args))]
    assert module.central_supervisor.live_controller(entry,processes)==123
    assert module.central_supervisor.live_controller(dict(entry,index=str(tmp_path/'other.json')),processes) is None
    assert module.central_supervisor.live_controller(dict(entry,batches=['three']),processes) is None

"""Keep provider output paths absolute when the CLI receives a relative manifest."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

SOURCE=Path(__file__).resolve().parents[1]/'prospective_collect.py'
spec=importlib.util.spec_from_file_location('prospective_paths_tested',SOURCE)
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_relative_manifest_produces_absolute_runtime_and_collection_paths(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder=tmp_path/'collection';folder.mkdir()
    (folder/'manifest.json').write_text(json.dumps(dict(protocol_sha256='test',protocol={},jobs=[])))
    (folder/'authorization.json').write_text(json.dumps(dict(protocol_sha256='test',allow_provider_calls=True)))
    seen={}
    runtime=SimpleNamespace(c=SimpleNamespace(environment=lambda:{'PATH':os.environ['PATH']},
        prior=SimpleNamespace(stop_children=lambda *args:None)))
    def load(work):
        seen['work']=work
        return runtime
    def collect(jobs,manifest,run_root,runtime,workers):
        seen['run_root']=run_root
    monkeypatch.setattr(p,'load_runtime',load)
    monkeypatch.setattr(p,'verify',lambda *args:None)
    monkeypatch.setattr(p,'collect',collect)
    monkeypatch.setattr(p,'status',lambda *args:{'planned':0,'completed':0})
    monkeypatch.setattr(p.signal,'signal',lambda *args:None)
    p.main(['collect','--manifest','collection/manifest.json','--authorization','collection/authorization.json'])
    assert seen['work'].is_absolute()
    assert seen['run_root'].is_absolute()
    assert seen['run_root']==folder/'runs/test'

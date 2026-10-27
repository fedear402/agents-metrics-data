from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import collection_supervisor as supervisor

def test_live_controller_requires_same_index_and_scope(tmp_path):
    entry=dict(index=str(tmp_path/'collections.json'),batches=['a','b'])
    prefix=f'python /x/production_queue.py --index {entry["index"]} --batches '
    assert supervisor.live_controller(entry,[(12,prefix+'a b --drain-model')])==12
    assert supervisor.live_controller(entry,[(12,prefix+'a c')]) is None
    assert supervisor.live_controller(entry,[(12,prefix.replace(entry['index'],'/other/index.json')+'a b')]) is None

def test_active_collection_blocks_restart(tmp_path,monkeypatch):
    entry=dict(index=str(tmp_path/'collections.json'),batches=['a'])
    monkeypatch.setattr(supervisor.queue,'checked_index',lambda p:[dict(batch='a',manifest=str(tmp_path/'manifest.json'))])
    monkeypatch.setattr(supervisor.queue,'status',lambda p:dict(rows=[dict(status='pending')]))
    monkeypatch.setattr(supervisor.queue,'candidates',lambda p,s:([{}],[]))
    monkeypatch.setattr(supervisor.queue,'collection_active',lambda p:True)
    assert not supervisor.inspect(entry,[])['restart_eligible']
    monkeypatch.setattr(supervisor.queue,'collection_active',lambda p:False)
    assert supervisor.inspect(entry,[])['restart_eligible']

def test_held_only_work_does_not_restart(tmp_path,monkeypatch):
    entry=dict(index=str(tmp_path/'collections.json'),batches=['a'])
    monkeypatch.setattr(supervisor.queue,'checked_index',lambda p:[dict(batch='a',manifest=str(tmp_path/'manifest.json'))])
    monkeypatch.setattr(supervisor.queue,'status',lambda p:dict(rows=[dict(status='needs_review')]))
    monkeypatch.setattr(supervisor.queue,'candidates',lambda p,s:([],['held']))
    monkeypatch.setattr(supervisor.queue,'collection_active',lambda p:False)
    assert not supervisor.inspect(entry,[])['restart_eligible']

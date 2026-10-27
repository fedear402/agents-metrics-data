"""Check local-recovery selection against the saved incident evidence."""
import copy
import json

import pytest

from benchmark import production_inventory as inventory


@pytest.fixture
def recovery_inventory(tmp_path):
    instance = inventory.Inventory.__new__(inventory.Inventory)
    instance.out = tmp_path
    instance.sources, instance.record_cache, instance.positions = {}, {}, {}
    instance.compatibility, instance.prompts = [], {}
    source = inventory.DEFAULT_OUT / 'work/environment_recovery/overlay.json'
    overlay = json.loads(source.read_text())
    # This fixture represents the incident before any pending request was
    # released. Later tests add their own releases inside the temporary copy.
    overlay.pop('first_request_releases', None)
    target = tmp_path / 'work/environment_recovery/overlay.json'
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(overlay))
    for entry in overlay['entries'] + overlay['pending_first_requests']:
        original = json.loads(inventory.absolute(entry['original_record']).read_text())
        job = entry['job']
        manifest = inventory.ROOT / 'data/runs/paper22_2026-09-18/original8_reps6to10/manifest.json'
        row = dict(**inventory.normalized(job), source='production_collection', local_job=job,
                   existing_manifest=str(manifest), batch='original8_reps6to10',
                   status='retained' if entry.get('kind') == 'environment_replay' else 'held')
        if row['status'] == 'retained':
            row['record_path'] = entry['original_record']
        else:
            row['record_directory'] = str(inventory.absolute(entry['original_record']).parent.parent)
        instance.positions[inventory.position_key(row)] = row
        instance.prompts[inventory.prompt_key(job)] = original['prompt']
    return instance, overlay, target


def test_selects_recovered_records_without_changing_protocol_findings(recovery_inventory):
    instance, overlay, _ = recovery_inventory
    instance.apply_environment_recovery()
    assert len(instance.positions) == 5
    for entry in overlay['entries']:
        row = instance.positions[inventory.position_key(entry['job'])]
        assert row['record_path'] == entry['recovered_record']
        assert row['status'] == 'retained'
        assert bool(row['protocol_violation']) == (entry['job']['task'] == 'cox_hazard')
    pending = instance.positions[inventory.position_key(overlay['pending_first_requests'][0]['job'])]
    assert pending['status'] == 'held'
    assert pending['recovery_status'] == 'pending_first_request_release'


@pytest.mark.parametrize('field', ['original_record_sha256', 'recovered_record_sha256', 'receipt_sha256'])
def test_rejects_changed_pinned_files(recovery_inventory, field):
    instance, overlay, target = recovery_inventory
    overlay['entries'][0][field] = '0' * 64
    target.write_text(json.dumps(overlay))
    with pytest.raises(ValueError):
        instance.apply_environment_recovery()


@pytest.mark.parametrize('change', ['position', 'job', 'duplicate', 'provider', 'source', 'protocol'])
def test_rejects_inconsistent_recovery_evidence(recovery_inventory, change):
    instance, overlay, target = recovery_inventory
    entry = overlay['entries'][0]
    if change == 'position':
        entry['position_id'] += '_wrong'
    elif change == 'job':
        entry['job']['global_rep'] = 0
    elif change == 'duplicate':
        overlay['entries'].append(copy.deepcopy(entry))
    else:
        receipt = json.loads(inventory.absolute(entry['receipt']).read_text())
        if change == 'provider':
            receipt['provider_calls_added'] = 1
        elif change == 'source':
            receipt['source_hashes'][entry['original_record']] = '0' * 64
        else:
            receipt['protocol_findings'] = []
        copied = target.parent / 'changed_receipt.json'
        copied.write_text(json.dumps(receipt))
        entry['receipt'] = str(copied)
        entry['receipt_sha256'] = inventory.file_hash(copied)
    target.write_text(json.dumps(overlay))
    with pytest.raises(ValueError):
        instance.apply_environment_recovery()


def first_request_release(recovery_inventory, status='queued_waiting_for_collection_lock'):
    instance, overlay, target = recovery_inventory
    pending = overlay['pending_first_requests'][0]
    folder = target.parent/'first_requests'
    folder.mkdir()
    sources = {pending['original_record']: pending['original_record_sha256'],
               pending['original_state']: pending['original_state_sha256']}
    plan = dict(entry=pending, original_protocol_sha256=overlay['original_protocol_sha256'],
                source_hashes=sources, destination=str(folder), provider_call_limit=1)
    path = folder/'plan.json'
    path.write_text(json.dumps(plan))
    release = dict(position_id=pending['position_id'], job=pending['job'], status=status,
                   plan=str(path), plan_sha256=inventory.file_hash(path))
    if status == 'completed':
        record = json.loads(inventory.absolute(pending['original_record']).read_text())
        record.update(status='completed', stage='done', calls=[{'ok': True}])
        record['recovery'] = dict(kind='first_request_release', original_record=pending['original_record'],
            original_record_sha256=pending['original_record_sha256'], original_protocol_sha256=overlay['original_protocol_sha256'],
            provider_calls_added=1, dispatch_started=True)
        record_path = folder/'run_record.json'
        record_path.write_text(json.dumps(record))
        receipt = {key:pending[key] for key in ('position_id','job','original_record','original_record_sha256',
                                                'original_state','original_state_sha256')}
        receipt.update(kind='first_request_release', original_protocol_sha256=overlay['original_protocol_sha256'],
            source_hashes=sources, recovered_record=str(record_path), recovered_record_sha256=inventory.file_hash(record_path),
            provider_calls_added=1, dispatch_started=True, status='completed', stage='done')
        receipt_path = folder/'receipt.json'
        receipt_path.write_text(json.dumps(receipt))
        release.update({key:receipt[key] for key in ('recovered_record','recovered_record_sha256','provider_calls_added')})
        release.update(receipt=str(receipt_path), receipt_sha256=inventory.file_hash(receipt_path))
    overlay['first_request_releases'] = [release]
    target.write_text(json.dumps(overlay))
    return release, target


@pytest.mark.parametrize('status,expected', [('queued_waiting_for_collection_lock','held'),
    ('running_first_request','active'), ('completed','retained')])
def test_first_request_status_and_selection(recovery_inventory, status, expected):
    instance, overlay, _ = recovery_inventory
    release, _ = first_request_release(recovery_inventory, status)
    instance.apply_environment_recovery()
    row = instance.positions[inventory.position_key(release['job'])]
    assert row['status'] == expected
    assert row['recovery_status'] == status
    assert len(instance.positions) == 5
    if status == 'completed':
        assert row['record_path'] == release['recovered_record']


def test_first_request_rejects_duplicate_release(recovery_inventory):
    instance, overlay, _ = recovery_inventory
    release, path = first_request_release(recovery_inventory)
    overlay['first_request_releases'].append(copy.deepcopy(release))
    path.write_text(json.dumps(overlay))
    with pytest.raises(ValueError, match='multiple releases'):
        instance.apply_environment_recovery()

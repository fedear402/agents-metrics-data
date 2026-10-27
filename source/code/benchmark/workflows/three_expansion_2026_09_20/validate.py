"""Check the exact retained responses and disjoint future ownership."""
import hashlib
from collections import Counter
from pathlib import Path
import prepare as p


def main():
    rows = p.read(p.META / 'planned_positions.json')
    assert len(rows) == 1080
    cells = Counter(tuple(r[k] for k in ['task', 'model', 'platform', 'prompt', 'condition']) for r in rows)
    assert len(cells) == 108 and set(cells.values()) == {10}
    retained = [row for row in rows if row['status'] == 'retained']
    assert len(retained) == len({r['source_record'] for r in retained}) == 75
    commands = p.read(p.RAW / 'luna/0_4/manifest.json')['protocol']['commands']
    observations = []
    for row in retained:
        record = p.read(row['source_record'])
        source = p.read(row['source_manifest'])
        assert record['protocol_sha256'] == source['protocol_sha256'] == row['source_protocol_sha256']
        assert p.c.sha(row['source_record']) == row['source_record_sha256']
        for key in ['task', 'model', 'platform', 'prompt', 'condition', 'global_rep']:
            assert record['job'][key] == row[key], (key, row)
        assert hashlib.sha256(record['prompt'].encode()).hexdigest() == row['effective_prompt_sha256']
        old_spec = source['protocol']['tasks'][row['task']]
        new_spec = p.read(p.BASE / row['task'] / 'task.json')
        assert old_spec['reference'] == new_spec['reference']
        before, after = p.c.task_input_files(old_spec), p.c.task_input_files(new_spec)
        assert before.keys() == after.keys()
        assert all(p.c.sha(before[k]) == p.c.sha(after[k]) for k in before)
        key = row['model'] + '/' + row['condition']
        assert source['protocol']['commands'][key][0] == commands[key][0], key
        observations.append(dict(source_record=row['source_record'], old_timeout=source['protocol']['commands'][key][1], new_timeout=commands[key][1]))
    for entry in p.read(p.META / 'collections.json'):
        manifest = p.read(entry['manifest'])
        p.c.validate_jobs(manifest['jobs'], manifest['protocol'])
        assert all(v[1] is None for v in manifest['protocol']['commands'].values())
    p.c.save(p.META / 'retention_validation.json', dict(passed=True, retained=75, exact_prompt_reference_inputs_provider_arguments=True, cells=108, repetitions_per_cell=10, timeout_differences=observations))
    print('Passed: 75 retained records, 108 cells, ten repetitions each.')


if __name__ == '__main__':
    main()

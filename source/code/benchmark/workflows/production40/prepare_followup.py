"""List reviewed historical outputs and protect current data dependencies."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
INDEX = ROOT / 'data/maintenance/production40_cleanup_followup'
KEEP_PLANS = {
    '2026-09-21_production40.md', '2026-09-21_paper40.md',
    '2026-09-21_paper40-independent.md', '2026-09-21_welfare-inputs.md',
}
REVIEWED_TREES = [
    'data/analysis/2026-09-20_six_vs_paper',
    'data/analysis/2026-09-19_prompt_task_review',
    'data/analysis/2026-09-19_completed_collections',
    'data/analysis/2026-09-20_timeout_audit',
    'data/analysis/2026-09-21_three-expansion',
    'data/analysis/paper22_2026-09-18',
    'data/production/paper40_2026-09-21',
    'artifacts/ui-audit', 'quality_reports/research',
    'quality_reports/run-modal-2026-09-16',
    'code/benchmark/workflows/paper40/superseded',
    'data/maintenance/production40_cleanup_followup/doc_history',
]


def prepare():
    destination = INDEX / 'plan.json'
    if destination.exists():
        raise ValueError('An existing archive plan must be reviewed before replacement')
    protected = {
        ROOT / name for name in ('data/analysis/main', 'data/production/main',
            'data/runs', 'data/collections', 'data/pilots', 'data/task_candidates',
            'data/tasks', 'code/benchmark/collection_runtime', 'paper/manuscript',
            'reference_implementations', 'literature', 'quality_reports/decision_records')
    }
    manifests = set()

    def add(value):
        if not isinstance(value, str) or not value:
            return
        path = Path(value)
        path = path if path.is_absolute() else ROOT / path
        if path.is_relative_to(ROOT):
            protected.add(path)
            protected.add(path.resolve())

    for source in ('data/production/main/baseline_runs.csv',
                   'data/analysis/main/data/runs.csv'):
        with (ROOT / source).open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 14400:
            raise ValueError(f'Incomplete selected dataset: {source}')
        for row in rows:
            for field in ('record_path', 'output_path', 'manifest_path'):
                add(row.get(field))
            if row.get('manifest_path'):
                manifests.add(Path(row['manifest_path']))
    for manifest in manifests:
        path = manifest if manifest.is_absolute() else ROOT / manifest
        data = json.loads(path.read_text())
        protocol = data['protocol']
        for source in protocol.get('source_hashes', {}):
            add(source)
        for spec in protocol.get('tasks', {}).values():
            add(spec.get('data_path'))
            for value in spec.get('input_files', {}).values():
                add(value)
    for name in ('data/production/main/catalog.json',
                 'data/analysis/main/data/catalog.json'):
        for spec in json.loads((ROOT / name).read_text()).values():
            add(spec.get('data_path'))
            for value in spec.get('input_files', {}).values():
                add(value)
    approved = REVIEWED_TREES + [str(path.relative_to(ROOT))
        for path in sorted((ROOT / 'quality_reports/plans').glob('*.md'))
        if path.name not in KEEP_PLANS]
    approved += [str(path.relative_to(ROOT))
        for path in sorted((ROOT / 'quality_reports/agent_logs').glob('*.jsonl'))
        if path.name[:10] < '2026-09-21']
    for relative in approved:
        source = ROOT / relative
        if not source.exists():
            raise ValueError(f'Reviewed source is absent: {relative}')
        if any(path == source or path.is_relative_to(source) or source.is_relative_to(path)
               for path in protected):
            raise ValueError(f'Reviewed source overlaps a retained dependency: {relative}')
    result = dict(archive_root=str(Path.home() / 'ARCHIVE/AGENTS-METRICS/2026-09-21-production40-followup'),
        approved_items=approved, protected_paths=sorted(str(path) for path in protected),
        current_selected_positions=14400, preserved_baseline_positions=14400,
        checked_manifests=len(manifests),
        rationale='Superseded generated outputs and administrative history. Selected responses, candidate data, inputs, references, runtime, and published-site analysis inputs remain local.')
    INDEX.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(plan=str(destination), approved_items=len(approved),
                          protected_paths=len(protected), manifests=len(manifests))))


if __name__ == '__main__':
    prepare()

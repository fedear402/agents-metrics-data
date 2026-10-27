"""Index saved response versions and reuse evidence without changing scores.

One row represents a physical record, not necessarily an independent response.
The index joins prior reviews and current selection. It does not grant eligibility.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/inventory'
SELECTION = ROOT / 'data/analysis/main/data/selected_records.csv'
ANALYSIS = ROOT / 'data/analysis/main/data/runs.csv'
PREVIOUS_PRODUCTION = ROOT / 'data/production/paper22_2026-09-18'
HISTORY = ROOT / 'data/collections/2026-09-16_all-runs-disposition/work'
WORK = ROOT / 'data/maintenance/project_cleanup/work/inventory'


def canonical(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def relative(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


class PathEvidence:
    """Resolve recorded storage moves while leaving the source files unchanged."""

    def __init__(self, root=ROOT):
        self.root = root
        self.moves = []
        self.archived_records = set()
        self.receipts = []
        migration = root / 'data/inventory/storage_migration_2026-09-18.json'
        if migration.is_file():
            self.receipts.append(migration)
            for item in json.loads(migration.read_text()).get('items', []):
                self.add(item['source'], item['destination'], migration)
        for receipt in sorted((root / 'data/maintenance').glob('*/archive_manifest.json')):
            self.receipts.append(receipt)
            manifest = json.loads(receipt.read_text())
            archive = Path(manifest['archive_root'])
            trees = manifest.get('trees', manifest.get('items', {}))
            for source, members in trees.items():
                self.add(source, archive / source, receipt)
                for member in members:
                    if member.get('kind') == 'file' and Path(member['path']).name in {'run_record.json', 'record.json'}:
                        self.archived_records.add(archive / source / member['path'])
        # Centralization receipts are additive. Historical source paths stay meaningful.
        centralized = root / 'data/inventory/record_path_aliases.csv'
        if centralized.is_file():
            self.receipts.append(centralized)
            for row in read_csv(centralized):
                self.add(row['source'], row['destination'], centralized)
        self.moves.sort(key=lambda item: len(item[0].parts), reverse=True)
        self.move_index = defaultdict(list)
        for source, destination, receipt in self.moves:
            self.move_index[source].append((destination, receipt))

    def add(self, source, destination, receipt):
        source, destination = Path(source), Path(destination)
        self.moves.append((source if source.is_absolute() else self.root / source,
                           destination if destination.is_absolute() else self.root / destination,
                           receipt))

    def resolve(self, value):
        path = Path(value)
        path = path if path.is_absolute() else self.root / path
        original = path
        pending = [(path, [])]
        visited = set()
        while pending:
            path, evidence = pending.pop()
            if path.exists():
                return path.resolve(), '|'.join(evidence)
            if path in visited:
                continue
            visited.add(path)
            for source in (path, *path.parents):
                if source in self.move_index:
                    # A historical collection can have archived records and a
                    # later summary-only directory at two destinations.
                    for destination, receipt in self.move_index[source]:
                        pending.append((destination / path.relative_to(source),
                                        evidence + [str(receipt.relative_to(self.root))]))
                    break
        return original.resolve(), ''


def load_selection(selection_path, analysis_path, paths):
    """Load only the principal selection. Historical choices cannot add members."""
    selected = read_csv(selection_path)
    analysis_rows = read_csv(analysis_path)
    details = {r['position_id']: r for r in analysis_rows}
    if len(details) != len(analysis_rows):
        raise ValueError('Duplicate positions in principal analysis')
    output, positions = {}, set()
    for row in selected:
        position = row['position_id']
        if position in positions:
            raise ValueError(f'Duplicate selected position: {position}')
        positions.add(position)
        path, _ = paths.resolve(row['record_path'])
        if path in output:
            raise ValueError(f'Two selected positions use one physical record: {path}')
        if position not in details:
            raise ValueError(f'Selected position absent from analysis: {position}')
        detail = details[position]
        if paths.resolve(detail['record_path'])[0] != path:
            raise ValueError(f'Selection and analysis disagree: {position}')
        output[path] = detail | row
    if positions != set(details):
        raise ValueError('Selection and analysis position sets differ')
    return output


def classify_record(row, selected, review, archived):
    """Describe evidence for retention without certifying a new treatment."""
    if selected:
        return 'selected', 'Member of the current principal selection'
    if row.get('parse_error'):
        return 'unresolved_record', 'Record could not be parsed'
    if row.get('record_role') == 'test_fixture':
        return 'test_fixture', 'Offline or test record; exclude from response counts'
    if archived:
        return 'historical_archived', 'Storage archive receipt; current reuse has not been certified'
    disposition = review.get('disposition', '')
    if disposition == 'compatible_unselected_reserved_position':
        return 'reusable_candidate_prior_review', 'Compatible in the dated 22-task review; check the intended new treatment'
    if disposition == 'selected':
        return 'reusable_candidate_prior_selection', 'Selected in the dated 22-task review; current selection uses another record or task'
    if disposition in {'historical_only', 'held_historical', 'excluded_administrative_interruption', 'preserved_recovery_predecessor'}:
        return 'historical_reviewed', 'Dated review disposition: ' + disposition
    if row.get('record_role') in {'audit_copy', 'audit_copy_or_local_diagnostic', 'recovery_record'}:
        return 'diagnostic_or_recovery', 'Review identity and source lineage before response reuse'
    return 'unresolved_candidate', 'Saved record with no current compatibility decision'


def raw_storage_root(record_path):
    """Suggest conventional raw roots; flag unusual layouts for manual review."""
    path = Path(record_path)
    if path.is_absolute():
        return str(path.parent), 'keep_external_archive'
    parts = path.parts
    if parts[:2] == ('data', 'runs'):
        depth = 4 if parts[2:3] in {('collections',), ('early_collections',)} else 3
        return str(Path(*parts[:depth])), 'already_in_raw_runs'
    if parts[:2] == ('data', 'pilots'):
        return str(Path(*parts[:3])), 'centralize_collection_preserve_metadata_and_alias'
    if parts[:2] in {('data', 'collections'), ('data', 'production')}:
        for marker in ('runs', 'records', 'recovery_records'):
            if marker in parts[3:-1]:
                return str(Path(*parts[:parts.index(marker, 3) + 1])), 'centralize_raw_tree_preserve_alias'
    return str(path.parent), 'review_layout_before_moving'


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def physical_records(base):
    for folder, dirs, files in os.walk(base, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {'.git', '.venv', '__pycache__', 'node_modules'}
                   and not (Path(folder) / d).is_symlink()]
        for name in files:
            path = Path(folder) / name
            if name in {'run_record.json', 'record.json'} and not path.is_symlink():
                yield path


def inspect(path):
    raw = path.read_bytes()
    row = dict(record_path=relative(path), record_sha256=hashlib.sha256(raw).hexdigest())
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError('Not a record object')
    except (ValueError, UnicodeError) as error:
        return row | dict(parse_error=str(error)), None
    job = value.get('job') if isinstance(value.get('job'), dict) else {}
    def get(*keys):
        for source in (job, value):
            for key in keys:
                if source.get(key) is not None and isinstance(source[key], (str, bool, int, float)):
                    return source[key]
        return ''
    outcome = value.get('outcome') if isinstance(value.get('outcome'), dict) else {}
    score = value.get('scoring') if isinstance(value.get('scoring'), dict) else {}
    agency = get('condition', 'exec_mode', 'agency')
    agency = {'codex_generate':'chatbot', 'no_feedback':'chatbot', 'tool_loop':'agent',
              'codex_tool_loop':'agent', 'codex_agent':'agent', 'codex_repair':'self_repair',
              'one_repair':'self_repair'}.get(agency, agency)
    prompt = get('prompt_mode', 'prompt')
    prompt = {'zero':'zero_shot', 'few':'few_shot'}.get(prompt, prompt)
    parts = Path(relative(path)).parts
    depth = 4 if parts[:3] in {('data', 'runs', 'collections'), ('data', 'runs', 'early_collections')} else 3
    campaign = '/'.join(parts[:depth]) if path.is_relative_to(ROOT) else str(path.parent)
    row.update(campaign=campaign, task=get('task_id', 'task'), model=get('model'),
               platform=get('platform'), prompt=prompt, agency=agency, rep=get('rep'),
               started_at=get('started_at', 'created_at', 'completed_at'),
               protocol_sha256=get('protocol_sha256'), prompt_sha256=get('prompt_sha256'),
               output_format=get('output_format', 'output_contract'), status=get('status'),
               original_primary=outcome.get('primary_success', score.get('success')),
               original_numeric=outcome.get('numerical_success', score.get('numeric_success')),
               measurement_status=outcome.get('measurement_status', ''),
               source_lineage=value.get('inherited_runtime_record', ''),
               parse_error='')
    row['record_uuid'] = get('uuid', 'run_id')
    row['recorded_cli_version'] = get('claude_version', 'codex_version')
    observed_versions = set()
    for stream_path in path.parent.glob('provider_attempt*.stdout'):
        with stream_path.open(errors='replace') as stream:
            first = stream.readline(1024 * 1024)
        try:
            event = json.loads(first)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get('claude_code_version'):
            observed_versions.add(str(event['claude_code_version']))
    row['observed_claude_versions'] = '|'.join(sorted(observed_versions))
    # Keep the initial prompt separate from later repair requests.
    for name in ('prompt_attempt1.txt', 'prompt.txt', 'prompt_0.txt'):
        candidate = path.parent / name
        if candidate.is_file():
            row['saved_prompt_path'] = relative(candidate.resolve())
            row['saved_prompt_sha256'] = hashlib.sha256(candidate.read_bytes()).hexdigest()
            break
    command = None
    for name in ('command_attempt1.json', 'command.json'):
        candidate = path.parent / name
        if candidate.is_file():
            try:
                command = json.loads(candidate.read_text())
            except (ValueError, UnicodeError) as error:
                row['command_parse_error'] = str(error)
                continue
            row['command_path'] = relative(candidate.resolve())
            break
    if not command:
        for backend in ('claude', 'codex', 'provider', 'openrouter'):
            provider = value.get(backend)
            if isinstance(provider, dict) and isinstance(provider.get('command'), list):
                command = dict(argv=provider['command'], timeout_seconds=value.get('claude_timeout_seconds'))
                row['command_path'] = row['record_path'] + '#' + backend + '.command'
                break
    signature = None
    if isinstance(command, dict):
        args = command.get('argv', command.get('command', []))
        if isinstance(args, list) and args:
            args = list(args)
            args[0] = Path(str(args[0])).name
            for i in range(1, len(args)):
                if args[i - 1] in ('-C', '-o', '--output-last-message', '--cd'):
                    args[i] = '<run-specific-path>'
            signature = dict(argv=args, timeout_seconds=command.get('timeout_seconds'))
            row['command_signature'] = digest(signature)
            row['provider_timeout_seconds'] = command.get('timeout_seconds', '')
    if any(s in row['record_path'] for s in ('/fixtures/', '/fix_regressions/', '/pytest_', '/offline_failure_cases/', '/scoring/fixed_', '/scoring/valid', '/scoring/stata_named_script/')) or get('task_id', 'task') == 'toy':
        row['record_role'] = 'test_fixture'
    elif '/samples/' in row['record_path']:
        row['record_role'] = 'audit_copy'
    elif '/recovery/' in row['record_path'] or '/resume/records/' in row['record_path']:
        row['record_role'] = 'recovery_record'
    else:
        row['record_role'] = 'saved_record_requires_membership_check'
    return row, signature


def main():
    scan_started_at = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    evidence = PathEvidence(ROOT)
    print('Loaded storage receipts', flush=True)
    aliases = []

    def indexed(path, column):
        result = {}
        if not path.is_file():
            return result
        for row in read_csv(path):
            if not row.get(column):
                continue
            resolved, receipts = evidence.resolve(row[column])
            if str(canonical(row[column])) != str(resolved):
                aliases.append(dict(source=row[column], destination=relative(resolved),
                                    receipt=receipts, source_table=relative(path)))
            result[resolved] = row
        return result

    historical = indexed(HISTORY / 'all_record_dispositions.csv', 'path')
    print(f'Resolved {len(historical)} historical records', flush=True)
    compatibility = indexed(PREVIOUS_PRODUCTION / 'compatibility.csv', 'record_path')
    views = indexed(HISTORY / 'lineage_2026-09-18.csv', 'record_path')
    selected = load_selection(SELECTION, ANALYSIS, evidence)
    # The repository census includes local fixtures and report copies. Recorded
    # archive paths preserve the historical census without searching all HOME.
    paths = set(historical) | set(selected) | set(compatibility) | set(views)
    paths.update(evidence.archived_records)
    paths.update(physical_records(ROOT))
    paths = sorted(paths)
    missing = [relative(path) for path in paths if not path.is_file()]
    existing = [path for path in paths if path.is_file()]
    print(f'Census found {len(existing)} physical records; {len(missing)} referenced paths missing', flush=True)
    rows, signatures, hash_mismatches = [], {}, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path, (row, signature) in zip(existing, pool.map(inspect, existing)):
            history = historical.get(path, {})
            review = compatibility.get(path, {})
            selection = selected.get(path, {})
            row['historical_path'] = history.get('path', '')
            row['historical_role'] = history.get('file_role', '')
            if row['historical_role'] == 'audit_copy_or_local_diagnostic' and row.get('record_role') == 'saved_record_requires_membership_check':
                row['record_role'] = 'audit_copy_or_local_diagnostic'
            elif row['historical_role'] in {'mock_or_offline_test', 'mock_provider_regression_record'}:
                row['record_role'] = 'test_fixture'
            row['historical_eligibility'] = history.get('new_main_sample_eligibility', '')
            row['historical_sonnet_review'] = history.get('sonnet_pr_proposed_eligibility', '')
            row['historical_sonnet_reasons'] = history.get('sonnet_pr_proposed_reasons', '')
            row['sample_views'] = views.get(path, {}).get('views', '')
            row['prior_review_disposition'] = review.get('disposition', '')
            row['prior_review_reason'] = review.get('reason', '')
            row['prior_review_record_sha256'] = review.get('record_sha256', '')
            row['prior_review_hash_matches'] = bool(review) and review.get('record_sha256') == row['record_sha256']
            row['current_selected'] = bool(selection)
            row['current_position'] = selection.get('position_id', '')
            row['storage_scope'] = 'repository' if path.is_relative_to(ROOT) else 'external_archive'
            row['current_disposition'], row['current_reason'] = classify_record(
                row, selection, review if row['prior_review_hash_matches'] else {},
                row['storage_scope'] == 'external_archive')
            row['prompt_compatibility'] = review.get('prompt_compatibility', '')
            row['context_compatibility'] = review.get('context_compatibility', '')
            row['compatibility_scope'] = 'dated_22_task_review' if review else ''
            if selection:
                row.update(model=selection['model'], task=selection['task'],
                           platform=selection['platform'], prompt=selection['prompt'],
                           agency=selection['agency'], selected_rep=selection['rep'],
                           current_primary=selection['primary'],
                           current_output_path=selection['output_path'],
                           current_output_sha256=selection['output_sha256'])
                row['selected_record_hash_matches'] = selection['record_sha256'] == row['record_sha256']
                if not row['selected_record_hash_matches']:
                    hash_mismatches.append(row['record_path'])
            if signature:
                signatures[row['command_signature']] = signature
            rows.append(row)
    hashes = defaultdict(list)
    for row in rows:
        hashes[row['record_sha256']].append(row)
    for row in rows:
        row['identical_record_files'] = len(hashes[row['record_sha256']])
        row['identical_selected_record'] = any(r['current_selected'] for r in hashes[row['record_sha256']])
    selected_rows = [row for row in rows if row['current_selected']]
    write_csv(OUT / 'selected_runs.csv', selected_rows)
    campaigns = defaultdict(list)
    for row in rows:
        if row['storage_scope'] == 'external_archive':
            # Keep archive collection names readable without one group per run.
            parts = Path(row['record_path']).parts
            start = parts.index('AGENTS-METRICS') + 1 if 'AGENTS-METRICS' in parts else 0
            row['campaign'] = '/'.join(parts[start:start + 4])
        campaigns[row.get('campaign', 'unparsed')].append(row)
    write_csv(OUT / 'records.csv', rows)
    write_csv(OUT / 'campaigns.csv', [dict(campaign=key, physical_records=len(values),
              selected=sum(row['current_selected'] for row in values),
              dispositions=json.dumps(dict(Counter(row['current_disposition'] for row in values)), sort_keys=True))
              for key, values in sorted(campaigns.items())])
    campaign_reviews_path = HISTORY / 'campaign_dispositions.csv'
    campaign_reviews = read_csv(campaign_reviews_path) if campaign_reviews_path.is_file() else []
    for row in campaign_reviews:
        target = ROOT / 'data/collections' / row['folder']
        row.update(review_date='2026-09-16',
                   documentation_path=str(Path('quality_reports/reports') / row['folder']),
                   data_path=relative(target) if target.exists() else '',
                   counts_are='retained assignments in the historical review; reused views must not be added')
    write_csv(OUT / 'campaign_reviews.csv', campaign_reviews)
    (OUT / 'command_versions.json').write_text(json.dumps(signatures, indent=2) + '\n')
    formats_path = HISTORY / 'replacements/format_record_index.json'
    formats = json.loads(formats_path.read_text()) if formats_path.is_file() else []
    for value in formats:
        value['canonical_directory'] = relative(evidence.resolve(value['canonical_directory'])[0])
        for entry in value.get('files', {}).values():
            entry['path'] = relative(evidence.resolve(entry['path'])[0])
    (OUT / 'format_observations.json').write_text(json.dumps(formats, indent=2) + '\n')
    cells = Counter((r['task'], r['model'], r['platform'], r['prompt'], r['agency']) for r in selected_rows)
    write_csv(OUT / 'selected_cells.csv', [dict(zip(('task', 'model', 'platform', 'prompt', 'agency'), key),
                                               selected_records=count) for key, count in sorted(cells.items())])
    duplicate_groups = [dict(record_sha256=sha, physical_files=len(values),
                             selected_files=sum(r['current_selected'] for r in values),
                             paths=json.dumps([r['record_path'] for r in values]))
                        for sha, values in sorted(hashes.items()) if len(values) > 1]
    write_csv(OUT / 'identical_record_groups.csv', duplicate_groups)
    write_csv(WORK / 'historical_path_resolutions.csv', aliases)
    write_csv(WORK / 'record_dispositions.csv', [dict(record_path=r['record_path'],
               storage_scope=r['storage_scope'], current_disposition=r['current_disposition'],
               reason=r['current_reason'], current_selected=r['current_selected'],
               record_sha256=r['record_sha256']) for r in rows])
    raw_roots = defaultdict(list)
    for row in rows:
        if row['storage_scope'] == 'repository':
            raw_roots[raw_storage_root(row['record_path'])].append(row)
    write_csv(WORK / 'raw_collection_roots.csv', [dict(storage_root=key[0], recommended_action=key[1],
              physical_records=len(values), selected_records=sum(r['current_selected'] for r in values),
              dispositions=json.dumps(dict(Counter(r['current_disposition'] for r in values)), sort_keys=True))
              for key, values in sorted(raw_roots.items())])
    write_csv(WORK / 'missing_record_sources.csv', [dict(record_path=p) for p in missing])
    missing_selected = [relative(p) for p in selected if not p.is_file()]
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(), scan_started_at=scan_started_at,
                   physical_record_files=len(rows), repository_record_files=sum(r['storage_scope'] == 'repository' for r in rows),
                   external_archive_record_files=sum(r['storage_scope'] == 'external_archive' for r in rows),
                   distinct_record_bytes=len(hashes), identical_record_groups=len(duplicate_groups),
                   current_selected=len(selected_rows), expected_selected=len(selected),
                   selected_tasks=len({r['task'] for r in selected_rows}), selected_cells=len(cells),
                   selected_repetitions=sorted(set(cells.values())),
                   selected_records_outside_raw_runs=[r['record_path'] for r in selected_rows
                                                      if not r['record_path'].startswith('data/runs/')],
                   selection_source=relative(SELECTION), selection_sha256=hashlib.sha256(SELECTION.read_bytes()).hexdigest(),
                   dispositions=dict(Counter(r['current_disposition'] for r in rows)),
                   format_observations=len(formats), missing_sources=missing, missing_selected=missing_selected,
                   selected_record_hash_mismatches=hash_mismatches,
                   parse_errors=sum(bool(r.get('parse_error')) for r in rows),
                   command_parse_errors=sum(bool(r.get('command_parse_error')) for r in rows),
                   recorded_command_versions=len(signatures), historical_alias_rows=len(aliases),
                   receipt_sources=[relative(p) for p in evidence.receipts],
                   interpretation='Physical files include copies, fixtures, recoveries and historical attempts. Record hashes do not establish independent responses.',
                   review_scope='Current selection membership plus dated prior reviews. Unresolved candidates require treatment and reference checks before reuse.',
                   provider_calls=0, sample_changed=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (WORK / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'missing_sources'}, indent=2))
    print(f'Missing historical sources: {len(missing)} (see summary.json)')
    if missing_selected or hash_mismatches or len(selected_rows) != len(selected):
        raise SystemExit('Principal selected record integrity check failed')


if __name__ == '__main__':
    main()

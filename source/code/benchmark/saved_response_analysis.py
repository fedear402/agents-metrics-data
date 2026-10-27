"""Analyze retained and fresh original-ten responses without provider calls.

Read the immutable position plan and indexed collections. Keep incomplete
positions explicit. Refresh only this design's analysis directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'code/benchmark/production_analysis.py').exists())
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import production_analysis as analysis

runtime = analysis.runtime
DEFAULT_DESIGN = ROOT / 'data/production/original10_2026-09-19'
FIELDS = ('task', 'model', 'platform', 'prompt', 'condition', 'global_rep')


def key(row):
    """Use the global repetition because local repetitions restart in each batch."""
    return tuple(row[field] for field in FIELDS)


def read_hashed(path, expected=None):
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f'File hash changed: {path}')
    return json.loads(content), digest


def index_jobs(entries, catalog, planned):
    """Deduplicate identical assignments and reject conflicting assignments."""
    indexed, receipts, duplicate_assignments = {}, [], 0
    for entry in entries:
        path = Path(entry['manifest'])
        manifest, file_hash = read_hashed(path)
        protocol = manifest['protocol']
        protocol_hash = runtime.digest(protocol)
        if protocol_hash != manifest['protocol_sha256'] or protocol_hash != entry['protocol_sha256']:
            raise ValueError(f'Manifest identity changed: {path}')
        if manifest['jobs'] != protocol['planned_jobs'] or len(manifest['jobs']) != entry['planned']:
            raise ValueError(f'Manifest membership changed: {path}')
        for task, spec in protocol['tasks'].items():
            if spec != catalog[task]:
                raise ValueError(f'Collection task differs from the analysis catalog: {task}')
            for source in runtime.task_input_files(spec).values():
                if analysis.sha(source) != protocol['source_hashes'][str(source)]:
                    raise ValueError(f'Pinned task input changed: {source}')
        for job in manifest['jobs']:
            position = key(job)
            if position not in planned or planned[position]['status'] == 'retained':
                raise ValueError(f'Fresh job conflicts with position plan: {position}')
            run_root = path.parent / 'runs' / protocol_hash
            assignment = dict(job=job, manifest_path=str(path), protocol_sha256=protocol_hash,
                              batch=entry['batch'], folder=str(run_root / runtime.job_id(job)))
            if position in indexed:
                if indexed[position] != assignment:
                    raise ValueError(f'Conflicting fresh assignments: {position}')
                duplicate_assignments += 1
            indexed[position] = assignment
        receipts.append(dict(batch=entry['batch'], manifest=str(path), manifest_sha256=file_hash,
                             protocol_sha256=protocol_hash, positions=len(manifest['jobs'])))
    return indexed, receipts, duplicate_assignments


def fresh_record(assignment):
    """Read a finished record, or preserve its current incomplete status."""
    folder = Path(assignment['folder'])
    terminal, latest = [], None
    for path in sorted(folder.glob('attempt_*/run_record.json')):
        record, digest = read_hashed(path)
        if record.get('protocol_sha256') != assignment['protocol_sha256'] or record.get('job') != assignment['job']:
            raise ValueError(f'Record identity differs from its assigned position: {path}')
        latest = (record, path, digest)
        if record.get('status') in runtime.TERMINAL:
            terminal.append(latest)
    if len(terminal) > 1:
        raise ValueError(f'Multiple terminal responses for one position: {folder}')
    if terminal:
        record, path, digest = terminal[0]
        return dict(analysis_status='analyzed', collection_status=record['status'],
                    record=record, record_path=path, record_sha256=digest)
    # State can be terminal before the collector writes the final record.
    states = sorted(folder.glob('attempt_*/state.json'))
    if states:
        state, _ = read_hashed(states[-1])
        if state.get('job') != assignment['job'] or state.get('protocol_sha256') != assignment['protocol_sha256']:
            raise ValueError(f'State identity differs from its assigned position: {states[-1]}')
        status = state.get('status', 'in_progress')
    else:
        status = latest[0].get('status', 'in_progress') if latest else 'pending'
    return dict(analysis_status='awaiting_record' if status in runtime.TERMINAL else status,
                collection_status=status, record=None,
                record_path=latest[1] if latest else folder / 'attempt_0001/run_record.json',
                record_sha256=latest[2] if latest else None)


def retained_record(position):
    path = Path(position['record_path'])
    if not path.is_file():
        return dict(analysis_status='missing_retained_record', collection_status='historical',
                    record=None, record_path=path, record_sha256=None)
    record, digest = read_hashed(path, position['record_sha256'])
    if record.get('task_id') != position['task'] or record.get('platform') != position['platform']:
        raise ValueError(f'Historical record identity differs: {path}')
    return dict(analysis_status='analyzed', collection_status=record.get('status', 'historical_completed'),
                record=record, record_path=path, record_sha256=digest)


def scored_row(position, selected, spec, c):
    """Use the same output extraction, outcome, and cost functions as production."""
    row = {field: position[field] for field in FIELDS}
    row.update(position_id='__'.join(map(str, key(position))), task_label=spec['title'],
               source='retained' if position['status'] == 'retained' else 'fresh',
               analysis_status=selected['analysis_status'], collection_status=selected['collection_status'],
               analyzed=selected['record'] is not None, record_path=str(selected['record_path']),
               record_sha256=selected['record_sha256'])
    if selected['record'] is None:
        return row, []
    record, path = selected['record'], selected['record_path']
    output = analysis.output_path(path, record, position['platform'])
    output_hash = analysis.sha(output) if output.is_file() else None
    values, extraction = analysis.read_output(output, c.base)
    if output_hash is not None and analysis.sha(output) != output_hash:
        raise ValueError(f'Output changed during analysis: {output}')
    passed, terms, notes = analysis.score_estimates(values, spec, c.compare_numeric)
    calls = analysis.attempts(record, 'provider')
    executions = analysis.attempts(record, 'execution')
    execution_ok = bool(executions and executions[-1].get('ok'))
    delivered = all(term in values and c.sp.number_or_none(values[term].get('estimate')) is not None
                    for term in spec['estimands'])
    provider_seconds, provider_time_complete = analysis.elapsed(calls)
    execution_seconds, execution_time_complete = analysis.elapsed(executions)
    row.update(primary=bool(passed), estimate_only=bool(passed),
               recorded_full_success=analysis.recorded_full_success(record),
               output_path=str(output), output_sha256=output_hash, output_file_exists=output.is_file(),
               output_available=delivered, execution_ok=execution_ok,
               failure_category='Success' if passed else 'Silent failure' if execution_ok and delivered else 'Loud failure',
               extraction=extraction, scoring_notes=' | '.join(notes),
               calls_count=len(calls), executions_count=len(executions),
               provider_seconds=provider_seconds, provider_timing_complete=provider_time_complete,
               execution_seconds=execution_seconds, execution_timing_complete=execution_time_complete,
               provider_ok=record.get('provider_ok'), task_censored=record.get('task_censored'),
               access_review_status=record.get('access_review_status', 'historical_review'),
               input_integrity=record.get('scoring', {}).get('integrity_pass'),
               **analysis.token_cost(record, position['model']))
    components = [dict(position_id=row['position_id'], **{field: row[field] for field in FIELDS},
                       term=term, **detail) for term, detail in terms.items()]
    return row, components


def write_frame(path, rows, columns=None):
    temporary = path.with_suffix(path.suffix + '.tmp')
    pd.DataFrame(rows, columns=columns).to_csv(temporary, index=False)
    temporary.replace(path)


def build(design=DEFAULT_DESIGN):
    design = Path(design).resolve()
    out = design / 'analysis'
    catalog, catalog_hash = read_hashed(design / 'catalog.json')
    positions, plan_hash = read_hashed(design / 'planned_positions.json')
    entries, index_hash = read_hashed(design / 'collections.json')
    planned = {key(position): position for position in positions}
    if len(planned) != len(positions):
        raise ValueError('Duplicate positions in the static plan')
    expected = set(itertools.product(catalog, ('luna', 'sonnet'), ('python', 'r', 'stata'),
                                    ('zero_shot', 'few_shot'), ('chatbot', 'self_repair', 'agent'), range(10)))
    if len(catalog) != 10 or set(planned) != expected:
        raise ValueError('The original-ten plan must contain the complete 3,600-position design')
    for task, spec in catalog.items():
        names = [row['term'] for row in spec['reference']['estimates']]
        if len(names) != len(set(names)) or set(names) != set(spec['estimands']):
            raise ValueError(f'Incomplete reference: {task}')
    indexed, receipts, duplicates = index_jobs(entries, catalog, planned)
    c = runtime.load_runtime(out).c
    rows, components = [], []
    for position in positions:
        assignment = indexed.get(key(position))
        if position['status'] == 'retained':
            selected = retained_record(position)
        elif assignment:
            selected = fresh_record(assignment)
        else:
            selected = dict(analysis_status='unassigned', collection_status='unassigned', record=None,
                            record_path='', record_sha256=None)
        row, checks = scored_row(position, selected, catalog[position['task']], c)
        row.update(batch=assignment['batch'] if assignment else None,
                   protocol_sha256=assignment['protocol_sha256'] if assignment else None,
                   manifest_path=assignment['manifest_path'] if assignment else None)
        rows.append(row)
        components.extend(checks)
    analyzed = [row for row in rows if row['analyzed']]
    record_paths = [row['record_path'] for row in analyzed]
    if len(record_paths) != len(set(record_paths)):
        raise ValueError('One raw response is assigned to more than one analysis position')
    cells = defaultdict(list)
    for row in rows:
        cells[tuple(row[field] for field in FIELDS[:-1])].append(row)
    coverage = []
    for cell, members in sorted(cells.items()):
        used = [row for row in members if row['analyzed']]
        successes = sum(row['estimate_only'] for row in used)
        coverage.append(dict(zip(FIELDS[:-1], cell), planned=len(members), analyzed=len(used),
                             missing=len(members)-len(used), complete=len(used)==len(members),
                             retained=sum(r['source']=='retained' for r in used),
                             fresh=sum(r['source']=='fresh' for r in used), successes=successes,
                             success_percent=100*successes/len(used) if used else None,
                             available_reps=','.join(str(r['global_rep']) for r in used),
                             missing_reps=','.join(str(r['global_rep']) for r in members if not r['analyzed'])))
    summary = dict(at=datetime.now(timezone.utc).isoformat(), planned=len(rows),
                   retained_planned=sum(p['status']=='retained' for p in positions), fresh_indexed=len(indexed),
                   analyzed=len(analyzed), analyzed_by_source=dict(Counter(r['source'] for r in analyzed)),
                   states=dict(Counter(r['analysis_status'] for r in rows)),
                   complete_cells=sum(r['complete'] for r in coverage), cells=len(coverage),
                   final_ready=len(analyzed)==len(rows), provider_calls_added=0,
                   primary='All requested estimates. No additional standard-error or observation-count requirement.',
                   missing_outcomes='Blank until a terminal record exists. A terminal record without output is a failed result.',
                   sonnet_price_per_million=dict(input=3, output=15), luna_currency_price=None,
                   duplicate_assignments_removed=duplicates,
                   input_hashes=dict(catalog=catalog_hash, planned_positions=plan_hash, collections=index_hash),
                   code_hashes={str(p.relative_to(ROOT)):analysis.sha(p) for p in
                                [Path(__file__), Path(analysis.__file__), Path(runtime.__file__), Path(c.compare_numeric.__code__.co_filename)]},
                   cost_usage_missing=sum(not r['cost_usage_complete'] for r in analyzed),
                   missing_output_files=sum(not r['output_file_exists'] for r in analyzed))
    out.mkdir(parents=True, exist_ok=True)
    write_frame(out / 'runs.csv', rows)
    write_frame(out / 'term_checks.csv', components)
    write_frame(out / 'coverage.csv', coverage)
    runtime.save(out / 'manifest_receipts.json', receipts)
    runtime.save(out / 'summary.json', summary)
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, default=DEFAULT_DESIGN)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = build(args.design)
    if args.require_complete and not result['final_ready']:
        raise SystemExit('Analysis saved. Some planned positions remain incomplete.')

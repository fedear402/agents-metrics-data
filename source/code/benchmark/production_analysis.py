"""Build one analysis table from selected production responses.

Read saved outputs without executing model scripts. Require complete coverage
before a final build. Preserve operational and full-score diagnostics separately.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd
import prospective_collect as runtime

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / 'data/production/paper22_2026-09-18'
OUT = ROOT / 'data/analysis/paper22_2026-09-18'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attempts(record, kind):
    plural = 'calls' if kind == 'provider' else 'executions'
    if isinstance(record.get(plural), list) and record[plural]:
        return record[plural]
    pattern = r'^(?:codex|claude)(?:_attempt_(\d+))?$' if kind == 'provider' else r'^execution(?:_attempt_(\d+))?$'
    items = []
    for key, value in record.items():
        match = re.match(pattern, key)
        if match and isinstance(value, dict):
            items.append((int(match[1] or 0), value))
    return [value for _, value in sorted(items)]


def token_cost(record, model):
    inputs, outputs, complete = 0, 0, True
    calls = attempts(record, 'provider')
    for call in calls:
        if call.get('quota_rejected_before_response'):
            continue
        usage = call.get('usage')
        keys = ['input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'] if model == 'sonnet' else ['input_tokens']
        valid = isinstance(usage, dict) and all(
            isinstance(usage.get(k), (int, float)) and not isinstance(usage.get(k), bool)
            and math.isfinite(usage[k]) and usage[k] >= 0 for k in keys + ['output_tokens'])
        if not valid:
            complete = False
            continue
        inputs += sum(usage[k] for k in keys)
        outputs += usage['output_tokens']
    complete = bool(calls) and complete
    return dict(cost_usage_complete=complete, main_input_tokens=inputs if complete else None,
                main_output_tokens=outputs if complete else None,
                total_main_tokens=inputs + outputs if complete else None,
                sonnet_cost_usd=(3 * inputs + 15 * outputs) / 1e6 if complete and model == 'sonnet' else None)


def elapsed(attempts_list):
    complete = bool(attempts_list) and all(
        isinstance(item.get('wall_seconds'), (int, float)) and not isinstance(item.get('wall_seconds'), bool)
        and math.isfinite(item['wall_seconds']) and item['wall_seconds'] >= 0 for item in attempts_list)
    return (sum(item['wall_seconds'] for item in attempts_list) if complete else None), complete


def recorded_full_success(record):
    outcome = record.get('outcome', {})
    if 'primary_success' in outcome:
        return outcome['primary_success']
    score = record.get('scoring', {})
    if 'success' in score:
        return score['success']
    if isinstance(score.get('numeric_success'), bool) and isinstance(score.get('artifact_success'), bool):
        return score['numeric_success'] and score['artifact_success']
    return None


def output_path(record_path, record, platform):
    explicit = record.get('primary_output_path')
    if explicit:
        return Path(explicit)
    folder = Path(record_path).parent
    if platform == 'stata':
        logs = list(folder.glob('after_attempt*_run.log'))
        if logs:
            return max(logs, key=lambda p: int(re.search(r'attempt(\d+)', p.name)[1]))
        if (folder / 'run.log').exists():
            return folder / 'run.log'
    return folder / 'result.json'


def read_output(path, base):
    if not path.is_file():
        return {}, 'missing_output'
    text = path.read_text(errors='replace')
    if path.suffix == '.json':
        data, rule = base.repair_result_text(text)
        if not isinstance(data, dict):
            return {}, 'unreadable_json'
        return {row['term']: row for row in data.get('estimates', [])
                if isinstance(row, dict) and isinstance(row.get('term'), str)}, rule
    text = re.sub(r'\n>\s?', '', text)
    values = {}
    for match in re.finditer(r'^\s*ANSWER:\s+(\S+)\s+estimate\s*=\s*([^\s]+)', text, re.MULTILINE):
        values[match[1]] = {'estimate': match[2]}
    return values, 'printed_estimate_tokens'


def score_estimates(values, spec, comparator):
    target = copy.deepcopy(spec)
    for row in target['reference']['estimates']:
        row['se'] = row['n_obs'] = None
    estimates = {term: {'estimate': row.get('estimate')} for term, row in values.items()}
    result = comparator(estimates, target)
    result = runtime.enforce_integer_terms(estimates, target, result)
    for row in target['reference']['estimates']:
        if isinstance(estimates.get(row['term'], {}).get('estimate'), bool):
            result[1][row['term']]['coef_pass'] = False
            result = (False, result[1], result[2] + [f'{row["term"]}: boolean estimate'])
    return result


def build(require_complete=False):
    catalog = json.loads((DESIGN / 'catalog.json').read_text())
    positions = json.loads((DESIGN / 'planned_positions.json').read_text())
    assert len(positions) == 7920
    if require_complete:
        if len(catalog) != 22 or any(row['status'] != 'retained' for row in positions):
            raise ValueError('The final paper requires all 7,920 verified positions and 22 ready definitions')
    OUT.mkdir(parents=True, exist_ok=True)
    c = runtime.load_runtime(OUT).c
    rows, term_rows = [], []
    for source in positions:
        if source['status'] != 'retained' or source['task'] not in catalog:
            continue
        path = Path(source['record_path'])
        assert sha(path) == source['record_sha256'], path
        record = json.loads(path.read_text())
        output = output_path(path, record, source['platform'])
        values, rule = read_output(output, c.base)
        spec = catalog[source['task']]
        passed, terms, notes = score_estimates(values, spec, c.compare_numeric)
        calls, executions = attempts(record, 'provider'), attempts(record, 'execution')
        provider_seconds, provider_timing_complete = elapsed(calls)
        execution_seconds, execution_timing_complete = elapsed(executions)
        executed = bool(executions and executions[-1].get('ok'))
        delivered = all(term in values and c.sp.number_or_none(values[term].get('estimate')) is not None
                        for term in spec['estimands'])
        row = {k: source[k] for k in ('task', 'model', 'platform', 'prompt', 'condition', 'global_rep')}
        row.update(agency=row['condition'], rep=row['global_rep'], task_label=spec['title'],
                   primary=bool(passed), estimate_only=bool(passed),
                   recorded_full_success=recorded_full_success(record),
                   execution_ok=executed, output_available=delivered,
                   failure_category='Success' if passed else 'Silent failure' if executed and delivered else 'Loud failure',
                   provider_seconds=provider_seconds, provider_timing_complete=provider_timing_complete,
                   execution_seconds=execution_seconds, execution_timing_complete=execution_timing_complete,
                   calls_count=len(calls), tools_count=sum(call.get('num_tool_calls') or 0 for call in calls),
                   record_path=str(path), record_sha256=source['record_sha256'],
                   output_path=str(output), output_sha256=sha(output) if output.is_file() else None,
                   parse_rule=rule, scoring_notes=' | '.join(notes),
                   cost_context=record.get('protocol_sha256', source.get('source')),
                   provider_ok=record.get('provider_ok'), task_censored=record.get('task_censored'),
                   record_status=record.get('status', 'historical_completed'),
                   protocol_violation=record.get('protocol_violation'),
                   access_review_status=record.get('access_review_status', 'historical_review'),
                   **token_cost(record, source['model']))
        rows.append(row)
        term_rows.extend(dict(position_id=source['position_id'], term=term, **detail) for term, detail in terms.items())
    data = pd.DataFrame(rows)
    assert not data.duplicated(['task', 'model', 'platform', 'prompt', 'condition', 'global_rep']).any()
    data.to_csv(OUT / 'runs.csv', index=False)
    pd.DataFrame(term_rows).to_csv(OUT / 'term_checks.csv', index=False)
    summary = dict(planned=7920, analyzed=len(data), final_ready=bool(len(data) == 7920 and len(catalog) == 22),
                   primary='All requested estimates; no additional standard-error or observation-count requirement',
                   full_score='Saved diagnostic only; historical definitions may differ',
                   sonnet_price_per_million=dict(input=3, output=15),
                   luna_currency_price=None, provider_calls=0,
                   index_sha256=sha(DESIGN / 'planned_positions.json'), catalog_sha256=sha(DESIGN / 'catalog.json'))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    build(args.require_complete)

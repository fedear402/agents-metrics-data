"""Build paper-format rows and prices for a Haiku 5.5 batch, with the paper's own functions. No model call.

Rule file: quality_reports/decision_records/2026-10-16_haiku55-api-collection.md.
The paper adds newly collected runs with code/paper/dataset.py replacement_rows: saved_response_analysis
fresh_record and scored_row, then the per-run update of dataset.build (score_output with channel_se, failure
category, token counts, repository paths, stale columns removed). This script applies the same steps to each
Haiku run, so its row has the columns and definitions of a fresh row in data/analysis/main/data/runs.csv.
The model label is `haiku` (column `model`); the token accounting uses the Claude branch of token_cost.
Prices follow production_paper_pricing.price_record (Claude branch, list prices by token type) with the
Haiku 5.5 list prices, and the pricing row has the columns of data/analysis/main/data/run_pricing.csv.

Outputs in data/collections/2026-10-16_haiku55-api/: <batch>_paper_rows.csv, <batch>_paper_pricing.csv.

Run from the repository root:
    uv run --no-sync python code/benchmark/workflows/haiku55_api_2026_10_16/paper_rows.py --manifest M
"""
from pathlib import Path
import argparse
import importlib.util
import json
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
META = ROOT / 'data/collections/2026-10-16_haiku55-api'
PAPER = ROOT / 'data/analysis/main/data'
LABEL = 'haiku'
# USD per million tokens, prompts up to 100,000 tokens (platform.claude.com pricing page, read 2026-10-16).
HAIKU_LIST_RATES = {'input': 0.10, 'cache_write_5m': 0.125, 'cache_write_1h': 0.20, 'cache_read': 0.01, 'output': 0.50}
LONG_PROMPT_TOKENS = 100_000  # above this prompt length per request, Haiku 5.5 has higher prices
# Columns of the earlier baseline table. dataset.build leaves them empty for fresh rows.
BASELINE_ONLY = ('tools_count', 'parse_rule', 'cost_context', 'record_status', 'protocol_violation', 'batch')


def load_dataset():
    spec = importlib.util.spec_from_file_location('paper_dataset_haiku_rows', ROOT / 'code/paper/dataset.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def largest_request(record_dir):
    """Largest prompt (uncached input, cache writes, and cache reads) of any request in the saved streams."""
    largest, seen = 0, set()
    for path in Path(record_dir).glob('provider_attempt*.stdout'):
        for line in path.read_text(errors='replace').splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            message = event.get('message') or {}
            if event.get('type') != 'assistant' or message.get('id') in seen:
                continue
            seen.add(message.get('id'))
            usage = message.get('usage') or {}
            largest = max(largest, sum(usage.get(k, 0) for k in ('input_tokens', 'cache_creation_input_tokens',
                                                                   'cache_read_input_tokens')))
    return largest


def price(record, pricing, record_dir):
    """production_paper_pricing.price_record, Claude list branch, with Haiku 5.5 rates."""
    calls = [call for call in pricing.provider_calls(record) if not call.get('quota_rejected_before_response')]
    totals = dict(input=0, output=0, cached=0, cache_write=0)
    if not calls:
        return None, totals, 'missing_provider_usage'
    if largest_request(record_dir) > LONG_PROMPT_TOKENS:
        return None, totals, 'request_above_100k_prompt_tokens'
    cost = 0.0
    for call in calls:
        usage = call.get('usage')
        keys = ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens', 'output_tokens')
        if not isinstance(usage, dict) or not all(pricing.token_number(usage.get(k)) for k in keys):
            return None, totals, 'missing_or_invalid_usage'
        uncached, writes, cached, outputs = (usage[k] for k in keys)
        split = usage.get('cache_creation') or {}
        five, hour = split.get('ephemeral_5m_input_tokens', 0), split.get('ephemeral_1h_input_tokens', 0)
        if five + hour != writes:
            return None, totals, 'invalid_cache_counts'
        cost += (uncached * HAIKU_LIST_RATES['input'] + five * HAIKU_LIST_RATES['cache_write_5m']
                 + hour * HAIKU_LIST_RATES['cache_write_1h'] + cached * HAIKU_LIST_RATES['cache_read']
                 + outputs * HAIKU_LIST_RATES['output']) / 1e6
        totals['input'] += uncached + writes + cached
        totals['output'] += outputs
        totals['cached'] += cached
        totals['cache_write'] += writes
    return cost, totals, ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    dataset = load_dataset()
    saved, analysis = dataset.saved, dataset.analysis
    spec_p = importlib.util.spec_from_file_location('paper_pricing_haiku', ROOT / 'code/benchmark/production_paper_pricing.py')
    pricing = importlib.util.module_from_spec(spec_p)
    spec_p.loader.exec_module(pricing)
    manifest_path = args.manifest.resolve()
    manifest, manifest_sha = saved.read_hashed(manifest_path)
    if analysis.runtime.digest(manifest['protocol']) != manifest['protocol_sha256'] or \
            manifest['jobs'] != manifest['protocol']['planned_jobs']:
        raise ValueError('Manifest identity changed')
    batch = manifest['jobs'][0]['block']
    catalog = json.loads((PAPER / 'catalog.json').read_text())
    c = analysis.runtime.load_runtime(Path(tempfile.mkdtemp(prefix='haiku_rows_'))).c
    rows, prices = [], []
    for job in manifest['jobs']:
        folder = manifest_path.parent / 'runs' / manifest['protocol_sha256'] / analysis.runtime.job_id(job)
        selected = saved.fresh_record(dict(job=job, protocol_sha256=manifest['protocol_sha256'], folder=str(folder)))
        if selected['record'] is None:
            raise ValueError(f'Incomplete run: {folder}')
        spec = catalog[job['task']]
        position = dict(job, status='fresh')
        row, _ = saved.scored_row(position, selected, spec, c)  # model 'sonnet': the Claude token accounting
        row.update(agency=row['condition'], rep=row['global_rep'], manifest_path=str(manifest_path),
                   protocol_sha256=manifest['protocol_sha256'])
        # The per-run update of dataset.build.
        path = Path(selected['record_path'])
        record, digest = saved.read_hashed(path, row['record_sha256'])
        output = Path(row['output_path'])
        scored = dataset.score_output(output, path, record, job['platform'], spec, c, dataset.OUTPUT_RULE)
        passed, delivered = scored['passed'], scored['delivered']
        row.update(record_path=dataset.repo_relative(path), record_sha256=digest,
                   output_path=dataset.repo_relative(output),
                   output_sha256=analysis.sha(output) if output.is_file() else None, primary=bool(passed),
                   y=int(passed), output_available=delivered, output_file_exists=output.is_file(),
                   extraction=scored['extraction'], scoring_notes=' | '.join(scored['notes']),
                   output_rule=dataset.OUTPUT_RULE, channel_used='|'.join(scored['channel_used']),
                   failure_category='Success' if passed else 'Silent failure' if row['execution_ok'] and delivered
                   else 'Loud failure', **analysis.token_cost(record, 'sonnet'))
        row['manifest_path'] = dataset.repo_relative(dataset.canonical_path(row['manifest_path']))
        for column in dataset.STALE_COLUMNS:
            row.pop(column, None)
        row['model'] = LABEL
        row['position_id'] = '__'.join(str(row[f]) for f in saved.FIELDS)
        row['model_id'] = 'claude-haiku-5-5'
        rows.append(row)
        cost, totals, issue = price(record, pricing, path.parent)
        prices.append(dict(task=job['task'], model=LABEL, platform=job['platform'], prompt=job['prompt'],
                           condition=job['condition'], global_rep=job['global_rep'], record_path=row['record_path'],
                           record_sha256=digest, main_input_tokens=totals['input'], main_output_tokens=totals['output'],
                           priced_cached_input_tokens=totals['cached'],
                           priced_cache_write_input_tokens=totals['cache_write'], cost_usd=cost,
                           cost_pricing_complete=cost is not None, cost_pricing_issue=issue))
    paper_columns = list(pd.read_csv(PAPER / 'runs.csv', nrows=0).columns)
    frame = pd.DataFrame(rows)
    for column in BASELINE_ONLY:  # empty in every fresh paper row (the 720 replacements)
        frame[column] = None
    missing = [col for col in paper_columns if col not in frame.columns]
    extra = [col for col in frame.columns if col not in paper_columns]
    if missing or extra != ['model_id']:
        raise ValueError(f'Columns differ from the paper runs: missing {missing}, extra {extra}')
    frame = frame[paper_columns + ['model_id']]
    pricing_columns = list(pd.read_csv(PAPER / 'run_pricing.csv', nrows=0).columns)
    price_frame = pd.DataFrame(prices)[pricing_columns]
    frame.to_csv(META / f'{batch}_paper_rows.csv', index=False)
    price_frame.to_csv(META / f'{batch}_paper_pricing.csv', index=False)
    print(json.dumps(dict(batch=batch, rows=len(frame), manifest_sha256=manifest_sha, success=int(frame.y.sum()),
                          priced=int(price_frame.cost_pricing_complete.sum()), cost_usd=float(price_frame.cost_usd.sum()),
                          columns_match_paper=True)))


if __name__ == '__main__':
    main()

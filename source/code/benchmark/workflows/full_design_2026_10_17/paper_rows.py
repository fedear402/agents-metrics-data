"""Paper-format rows, prices, and a summary for the full-design runs of Haiku 5.5 or GPT-6 Sol. No model call.

Decision record: quality_reports/decision_records/2026-10-17_haiku55-sol-full-design.md.
Rows are built with the paper's own functions, as code/paper/dataset.py replacement_rows does: saved_response_analysis
fresh_record and scored_row, then the per-run update of dataset.build (score_output with channel_se, failure
category, token counts, repository paths, stale columns removed). The rows have the columns of
data/analysis/main/data/runs.csv plus model_id. Labels: `haiku` (Claude token accounting) and `sol` (Codex token
accounting). Unfinished runs are listed and left out, so the script can run during the collection.
Prices follow production_paper_pricing.price_record: the Claude branch with Haiku 5.5 list prices (a request above
100,000 prompt tokens stays unpriced), and the Codex branch with GPT-6 Sol list prices (sessions above 272,000
aggregate input tokens are counted, as the paper counts them for Luna).

Outputs in data/collections/2026-10-17_full-design/: <model>_paper_rows.csv, <model>_paper_pricing.csv,
<model>_paper_summary.json.

Run from the repository root:
    uv run --no-sync python code/benchmark/workflows/full_design_2026_10_17/paper_rows.py --model sol
"""
from pathlib import Path
import argparse
import importlib.util
import json
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / 'data/runs/full_2026-10-17'
META = ROOT / 'data/collections/2026-10-17_full-design'
PAPER = ROOT / 'data/analysis/main/data'
SETTINGS = {'haiku55': dict(label='haiku', accounting='sonnet', model_id='claude-haiku-5-5'),
            'opus55': dict(label='opus', accounting='sonnet', model_id='claude-opus-5-5'),
            'sonnet55': dict(label='sonnet55', accounting='sonnet', model_id='claude-sonnet-5-5'),
            'sol': dict(label='sol', accounting='luna', model_id='gpt-6-sol'),
            'luna6': dict(label='luna6', accounting='luna', model_id='gpt-6-luna')}
# USD per million tokens, standard tier, contexts up to 272,000 input tokens
# (developers.openai.com/api/docs/pricing, read 2026-10-17).
CODEX_RATES = {'sol': {'input': 2.00, 'cached_input': 0.20, 'cache_write': 2.50, 'output': 10.00},
               'luna6': {'input': 0.10, 'cached_input': 0.01, 'cache_write': 0.125, 'output': 0.50}}
EXTRA = {'opus55': [ROOT / 'data/runs/opus55_api_2026-10-19/broad_tobit/0_0/manifest.json']}  # the one-task test, same design
BASELINE_ONLY = ('tools_count', 'parse_rule', 'cost_context', 'record_status', 'protocol_violation', 'batch')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def price_sol(record, pricing, rates):
    """production_paper_pricing.price_record, Codex branch, with GPT-6 Sol rates."""
    calls = [call for call in pricing.provider_calls(record) if not call.get('quota_rejected_before_response')]
    totals = dict(input=0, output=0, cached=0, cache_write=0)
    if not calls:
        return None, totals, 'missing_provider_usage', 0
    cost, long_sessions = 0.0, 0
    for call in calls:
        usage = call.get('usage')
        keys = ('input_tokens', 'cached_input_tokens', 'output_tokens')
        if not isinstance(usage, dict) or not all(pricing.token_number(usage.get(k)) for k in keys):
            return None, totals, 'missing_or_invalid_usage', long_sessions
        inputs, cached, outputs = (usage[k] for k in keys)
        writes = usage.get('cache_write_input_tokens', 0)
        if not pricing.token_number(writes) or cached + writes > inputs:
            return None, totals, 'invalid_cache_counts', long_sessions
        long_sessions += inputs > 272000
        cost += ((inputs - cached - writes) * rates['input'] + cached * rates['cached_input']
                 + writes * rates['cache_write'] + outputs * rates['output']) / 1e6
        totals['input'] += inputs
        totals['output'] += outputs
        totals['cached'] += cached
        totals['cache_write'] += writes
    return cost, totals, '', long_sessions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=sorted(SETTINGS), required=True)
    args = parser.parse_args()
    setting = SETTINGS[args.model]
    dataset = load('paper_dataset_full', ROOT / 'code/paper/dataset.py')
    pricing = load('paper_pricing_full', ROOT / 'code/benchmark/production_paper_pricing.py')
    haiku = load('haiku_rows', ROOT / 'code/benchmark/workflows/haiku55_api_2026_10_16/paper_rows.py')
    saved, analysis = dataset.saved, dataset.analysis
    if args.model == 'opus55':
        # Claude Opus 5.5 list prices (platform.claude.com pricing page, read 2026-10-16); no long-prompt surcharge.
        haiku.HAIKU_LIST_RATES = {'input': 4.00, 'cache_write_5m': 5.00, 'cache_write_1h': 8.00, 'cache_read': 0.20,
                                  'output': 20.00}
        haiku.LONG_PROMPT_TOKENS = 10 ** 12
    if args.model == 'sonnet55':
        # Claude Sonnet 5.5 list prices (claude-api skill model table, cached 2026-10-06); cache writes 1.25x and 2x input.
        haiku.HAIKU_LIST_RATES = {'input': 2.00, 'cache_write_5m': 2.50, 'cache_write_1h': 4.00, 'cache_read': 0.20,
                                  'output': 10.00}
        haiku.LONG_PROMPT_TOKENS = 10 ** 12
    catalog = json.loads((PAPER / 'catalog.json').read_text())
    c = analysis.runtime.load_runtime(Path(tempfile.mkdtemp(prefix='full_rows_'))).c
    rows, prices, missing, long_sessions = [], [], [], 0
    manifests = [RAW / args.model / half / 'manifest.json' for half in ('0_4', '5_9')] + EXTRA.get(args.model, [])
    for manifest_path in manifests:
        manifest, _ = saved.read_hashed(manifest_path)
        if analysis.runtime.digest(manifest['protocol']) != manifest['protocol_sha256'] or \
                manifest['jobs'] != manifest['protocol']['planned_jobs']:
            raise ValueError(f'Manifest identity changed: {manifest_path}')
        for job in manifest['jobs']:
            folder = manifest_path.parent / 'runs' / manifest['protocol_sha256'] / analysis.runtime.job_id(job)
            selected = saved.fresh_record(dict(job=job, protocol_sha256=manifest['protocol_sha256'], folder=str(folder)))
            if selected['record'] is None:
                missing.append(dict(id=analysis.runtime.job_id(job), status=selected['collection_status']))
                continue
            spec = catalog[job['task']]
            row, _ = saved.scored_row(dict(job, status='fresh'), selected, spec, c)
            row.update(agency=row['condition'], rep=row['global_rep'], manifest_path=str(manifest_path),
                       protocol_sha256=manifest['protocol_sha256'])
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
                       failure_category='Success' if passed else 'Silent failure'
                       if row['execution_ok'] and delivered else 'Loud failure',
                       **analysis.token_cost(record, setting['accounting']))
            row['manifest_path'] = dataset.repo_relative(dataset.canonical_path(row['manifest_path']))
            for column in dataset.STALE_COLUMNS:
                row.pop(column, None)
            row['model'] = setting['label']
            row['position_id'] = '__'.join(str(row[f]) for f in saved.FIELDS)
            row['model_id'] = setting['model_id']
            rows.append(row)
            if args.model in ('haiku55', 'opus55', 'sonnet55'):
                cost, totals, issue = haiku.price(record, pricing, path.parent)
            else:
                cost, totals, issue, long = price_sol(record, pricing, CODEX_RATES[args.model])
                long_sessions += long
            prices.append(dict(task=job['task'], model=setting['label'], platform=job['platform'],
                               prompt=job['prompt'], condition=job['condition'], global_rep=job['global_rep'],
                               record_path=row['record_path'], record_sha256=digest,
                               main_input_tokens=totals['input'], main_output_tokens=totals['output'],
                               priced_cached_input_tokens=totals['cached'],
                               priced_cache_write_input_tokens=totals['cache_write'], cost_usd=cost,
                               cost_pricing_complete=cost is not None, cost_pricing_issue=issue))
    paper_columns = list(pd.read_csv(PAPER / 'runs.csv', nrows=0).columns)
    frame = pd.DataFrame(rows)
    for column in BASELINE_ONLY:
        frame[column] = None
    if frame.empty:
        raise SystemExit('No completed runs yet')
    absent = [col for col in paper_columns if col not in frame.columns]
    if absent:
        raise ValueError(f'Columns missing from the paper format: {absent}')
    frame = frame[paper_columns + ['model_id']]
    pricing_frame = pd.DataFrame(prices)[list(pd.read_csv(PAPER / 'run_pricing.csv', nrows=0).columns)]
    frame.to_csv(META / f'{args.model}_paper_rows.csv', index=False)
    pricing_frame.to_csv(META / f'{args.model}_paper_pricing.csv', index=False)
    merged = frame.merge(pricing_frame[['record_path', 'cost_usd']], on='record_path')
    summary = dict(model=args.model, model_id=setting['model_id'], completed=len(frame), unfinished=len(missing),
                   unfinished_by_status=pd.Series([m['status'] for m in missing]).value_counts().to_dict(),
                   success=round(float(frame.y.mean()), 4),
                   success_by_condition=frame.groupby('condition').y.mean().round(4).to_dict(),
                   success_by_platform=frame.groupby('platform').y.mean().round(4).to_dict(),
                   failure_categories=frame.failure_category.value_counts().to_dict(),
                   cost_total_usd=round(float(merged.cost_usd.sum()), 4),
                   cost_per_run_by_condition=merged.groupby('condition').cost_usd.mean().round(5).to_dict(),
                   unpriced=int((~pricing_frame.cost_pricing_complete).sum()),
                   sessions_above_272k_input=long_sessions,
                   provider_seconds_by_condition=frame.groupby('condition').provider_seconds.mean().round(1).to_dict())
    (META / f'{args.model}_paper_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

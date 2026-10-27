"""Value saved paper tokens at fixed API rates without changing source data.

The default valuation ('list') prices every token type at the published
standard API list price for both models: uncached input, cache reads, cache
writes by duration, and output. The switch 'sonnet_all_input' reproduces the
earlier valuation, which priced all Sonnet input at the base input rate.
Luna prices are the same under both valuations. Codex records sum internal
requests, so these totals cannot establish each request's context length.
Do not apply long-context surcharges to a whole Codex session total.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRICING_DATE = '2026-09-24'
LUNA_SOURCE = 'https://developers.openai.com/api/docs/models/gpt-5.6-luna'
PRICING_SOURCE = 'https://developers.openai.com/api/docs/pricing'
SONNET_SOURCE = 'https://platform.claude.com/docs/en/about-claude/pricing'
LUNA_RATES = {'input': 0.20, 'cached_input': 0.02, 'cache_write': 0.25, 'output': 1.20}
SONNET_LIST_RATES = {'input': 3.0, 'cache_write_5m': 3.75, 'cache_write_1h': 6.0,
                     'cache_read': 0.30, 'output': 15.0}
SONNET_ALL_INPUT_RATES = {'all_input': 3.0, 'output': 15.0}
SONNET_RATES = SONNET_ALL_INPUT_RATES  # Earlier name, kept for existing imports.
VALUATIONS = ('list', 'sonnet_all_input')
DEFAULT_VALUATION = 'list'
VALUATION_TEXT = {
    'list': dict(
        valuation='Fixed-rate API list-price token cost for both models, not a provider invoice',
        sonnet_basis=('Standard Claude API list prices by token type: uncached input, cache reads, '
                      'five-minute and one-hour cache writes, and output. Decision record '
                      '2026-09-24_list-cache-pricing.')),
    'sonnet_all_input': dict(
        valuation='Fixed-rate token cost with all Sonnet input at the base input rate, not a provider invoice',
        sonnet_basis='User-authorized valuation of all input, including cache reads and writes, at USD 3 per million.'),
}


def check_valuation(valuation):
    if valuation not in VALUATIONS:
        raise ValueError(f'Unknown cost valuation {valuation!r}; choose one of {VALUATIONS}')


def provider_calls(record):
    """Read both stored record formats without importing collection code."""
    if isinstance(record.get('calls'), list) and record['calls']:
        return record['calls']
    calls = []
    for key, value in record.items():
        match = re.fullmatch(r'(?:codex|claude)(?:_attempt_(\d+))?', key)
        if match and isinstance(value, dict):
            calls.append((int(match[1] or 0), value))
    return [value for _, value in sorted(calls)]


def token_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0 and int(value) == value)


SONNET_STREAM_FILES = ('provider_attempt1.stdout', 'sonnet_events_attempt1.jsonl')
CHARS_PER_OUTPUT_TOKEN = 4  # Chosen estimate for visible text; not a measured tokenizer ratio.


def sonnet_stream_usage(record_dir):
    """Usage of a Sonnet run that stopped before its final usage event.

    Decision record 2026-09-30_stream-pricing-timed-out-runs. Each assistant
    message in the saved Claude Code stream carries its input, cache-write, and
    cache-read tokens, which are exact. Its output count stops early, so output
    tokens are estimated as the thinking-token estimates of the stream (the last
    value of each thinking block) plus the visible text and tool input divided by
    CHARS_PER_OUTPUT_TOKEN, and never below the recorded partial counts.
    """
    for name in SONNET_STREAM_FILES:
        path = Path(record_dir) / name
        if path.is_file():
            break
    else:
        return None
    messages, visible, blocks, current = {}, 0, [], None
    for line in path.read_text(errors='replace').splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get('type') == 'system' and event.get('subtype') == 'thinking_tokens':
            value = event.get('estimated_tokens', 0)
            if current is not None and value < current:
                blocks.append(current)
            current = value
        elif event.get('type') == 'assistant':
            # Claude Code writes one event per content block, so events of one
            # message share its id and usage: count the usage once, the content always.
            message = event.get('message', {})
            messages.setdefault(message.get('id'), message.get('usage') or {})
            for part in message.get('content') or []:
                if part.get('type') == 'text':
                    visible += len(part.get('text') or '')
                elif part.get('type') == 'tool_use':
                    visible += len(json.dumps(part.get('input', {})))
    if current is not None:
        blocks.append(current)
    if not messages:
        return None
    total = Counter()
    for usage in messages.values():
        for key in ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens', 'output_tokens'):
            total[key] += usage.get(key, 0)
        split = usage.get('cache_creation') or {}
        total['ephemeral_5m_input_tokens'] += split.get('ephemeral_5m_input_tokens', 0)
        total['ephemeral_1h_input_tokens'] += split.get('ephemeral_1h_input_tokens', 0)
    estimated_output = sum(blocks) + math.ceil(visible / CHARS_PER_OUTPUT_TOKEN)
    total['output_tokens'] = max(total['output_tokens'], estimated_output)
    return dict(input_tokens=total['input_tokens'], cache_creation_input_tokens=total['cache_creation_input_tokens'],
                cache_read_input_tokens=total['cache_read_input_tokens'], output_tokens=total['output_tokens'],
                cache_creation=dict(ephemeral_5m_input_tokens=total['ephemeral_5m_input_tokens'],
                                    ephemeral_1h_input_tokens=total['ephemeral_1h_input_tokens']))


def price_record(record, model, valuation=DEFAULT_VALUATION, record_dir=None):
    """Return fixed-rate costs and usage diagnostics for one saved response."""
    check_valuation(valuation)
    calls = [c for c in provider_calls(record) if not c.get('quota_rejected_before_response')]
    totals = dict(input=0, output=0, cached=0, cache_write=0)
    diagnostics = Counter(calls=len(calls))
    if not calls:
        return None, totals, diagnostics, 'missing_provider_usage'
    if model == 'sonnet' and record_dir is not None and not any(isinstance(c.get('usage'), dict) for c in calls):
        usage = sonnet_stream_usage(record_dir)
        if usage is not None:
            cost, totals, diagnostics, issue = price_record({'calls': [dict(usage=usage)]}, 'sonnet', valuation)
            return cost, totals, diagnostics, issue or 'estimated_from_stream'
    cost = 0.0
    for call in calls:
        usage = call.get('usage')
        if not isinstance(usage, dict):
            return None, totals, diagnostics, 'missing_provider_usage'
        if model == 'luna':
            keys = ('input_tokens', 'cached_input_tokens', 'output_tokens')
            if not all(token_number(usage.get(key)) for key in keys):
                return None, totals, diagnostics, 'missing_or_invalid_luna_usage'
            if call.get('model') not in (None, 'gpt-5.6-luna'):
                return None, totals, diagnostics, 'unexpected_luna_model'
            inputs, cached, outputs = (usage[key] for key in keys)
            writes = usage.get('cache_write_input_tokens', 0)
            if not token_number(writes) or cached + writes > inputs:
                return None, totals, diagnostics, 'invalid_cache_counts'
            diagnostics['legacy_calls_without_cache_write_field'] += 'cache_write_input_tokens' not in usage
            diagnostics['sessions_above_272k_aggregate_input'] += inputs > 272000
            cost += ((inputs - cached - writes) * LUNA_RATES['input']
                     + cached * LUNA_RATES['cached_input']
                     + writes * LUNA_RATES['cache_write']
                     + outputs * LUNA_RATES['output']) / 1e6
        elif model == 'sonnet':
            keys = ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens', 'output_tokens')
            if not all(token_number(usage.get(key)) for key in keys):
                return None, totals, diagnostics, 'missing_or_invalid_sonnet_usage'
            uncached, writes, cached, outputs = (usage[key] for key in keys)
            inputs = uncached + writes + cached
            if valuation == 'sonnet_all_input':
                cost += (inputs * SONNET_ALL_INPUT_RATES['all_input']
                         + outputs * SONNET_ALL_INPUT_RATES['output']) / 1e6
            else:
                split = usage.get('cache_creation')
                if not isinstance(split, dict):
                    if writes:
                        return None, totals, diagnostics, 'missing_cache_write_duration'
                    split = {}
                five = split.get('ephemeral_5m_input_tokens', 0)
                hour = split.get('ephemeral_1h_input_tokens', 0)
                if not (token_number(five) and token_number(hour)) or five + hour != writes:
                    return None, totals, diagnostics, 'invalid_cache_counts'
                cost += (uncached * SONNET_LIST_RATES['input']
                         + five * SONNET_LIST_RATES['cache_write_5m']
                         + hour * SONNET_LIST_RATES['cache_write_1h']
                         + cached * SONNET_LIST_RATES['cache_read']
                         + outputs * SONNET_LIST_RATES['output']) / 1e6
        else:
            return None, totals, diagnostics, 'unknown_model'
        totals['input'] += inputs
        totals['output'] += outputs
        totals['cached'] += cached
        totals['cache_write'] += writes
    return cost, totals, diagnostics, ''


def price_dataframe(frame, valuation=DEFAULT_VALUATION):
    """Return a priced copy and receipt. Preserve the original paper snapshot."""
    check_valuation(valuation)
    priced = frame.copy(deep=True)
    additions = []
    diagnostic_counts = Counter()
    for row in frame.to_dict('records'):
        # runs.csv stores record_path relative to the repository root.
        path = Path(row['record_path'])
        path = path if path.is_absolute() else ROOT / path
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['record_sha256']:
            raise ValueError(f'Saved record hash changed: {path}')
        record = json.loads(raw)
        cost, totals, diagnostics, issue = price_record(record, row['model'], valuation, record_dir=path.parent)
        diagnostic_counts.update(diagnostics)
        diagnostic_counts['estimated_from_stream'] += issue == 'estimated_from_stream'
        if cost is not None and issue != 'estimated_from_stream':
            for key, total_key in [('main_input_tokens', 'input'), ('main_output_tokens', 'output')]:
                if not token_number(row.get(key)) or row[key] != totals[total_key]:
                    raise ValueError(f'Saved token total differs from snapshot: {path}: {key}')
            if row['model'] == 'sonnet' and pd.notna(row.get('sonnet_cost_usd')):
                # The snapshot column holds the all-input value under every valuation.
                all_input = price_record(record, 'sonnet', 'sonnet_all_input')[0]
                if all_input is None or not math.isclose(all_input, row['sonnet_cost_usd'], abs_tol=1e-12):
                    raise ValueError(f'Sonnet cost differs from snapshot: {path}')
        additions.append(dict(cost_usd=cost, cost_pricing_complete=cost is not None,
                              cost_pricing_issue=issue,
                              priced_cached_input_tokens=totals['cached'] if cost is not None else None,
                              priced_cache_write_input_tokens=totals['cache_write'] if cost is not None else None))
    for key in additions[0] if additions else ('cost_usd', 'cost_pricing_complete', 'cost_pricing_issue'):
        priced[key] = [item[key] for item in additions]
    coverage = {}
    for model, group in priced.groupby('model'):
        coverage[model] = dict(rows=len(group), priced=int(group.cost_pricing_complete.sum()),
                               missing=int((~group.cost_pricing_complete).sum()))
    sonnet_rates = SONNET_LIST_RATES if valuation == 'list' else SONNET_ALL_INPUT_RATES
    receipt = dict(
        valuation=VALUATION_TEXT[valuation]['valuation'],
        valuation_id=valuation,
        retrieved_date=PRICING_DATE,
        luna_sources=[LUNA_SOURCE, PRICING_SOURCE],
        sonnet_sources=[SONNET_SOURCE],
        usd_per_million_tokens={'luna': LUNA_RATES, 'sonnet': sonnet_rates},
        sonnet_basis=VALUATION_TEXT[valuation]['sonnet_basis'],
        luna_basis='Standard short-context rates with recorded cache reads. Legacy missing cache-write fields contribute zero cache-write tokens.',
        excluded_adjustments=['Long-context request surcharges', 'Fast-mode and other service-tier surcharges', 'Tool fees', 'Subscription billing'],
        context_limitation='Codex usage sums internal requests. Aggregate input above 272,000 does not establish a long-context request.',
        coverage=coverage, diagnostics=dict(diagnostic_counts), provider_calls_launched=0,
    )
    return priced, receipt


def price_snapshot(frame, out, valuation=DEFAULT_VALUATION):
    """Save separate pricing data and return a priced copy for paper figures."""
    priced, receipt = price_dataframe(frame, valuation)
    destination = Path(out) / 'data'
    destination.mkdir(parents=True, exist_ok=True)
    columns = [key for key in (
        'task', 'model', 'platform', 'prompt', 'condition', 'global_rep',
        'record_path', 'record_sha256', 'main_input_tokens', 'main_output_tokens',
        'priced_cached_input_tokens', 'priced_cache_write_input_tokens',
        'cost_usd', 'cost_pricing_complete', 'cost_pricing_issue') if key in priced]
    priced[columns].to_csv(destination / 'run_pricing.csv', index=False)
    (destination / 'pricing.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return priced

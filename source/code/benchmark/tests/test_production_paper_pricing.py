"""Check fixed-rate valuation without provider calls or source edits."""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import production_paper_pricing as pricing


def luna(inputs=1000, cached=600, writes=100, outputs=200):
    return {'model': 'gpt-5.6-luna', 'usage': dict(input_tokens=inputs, cached_input_tokens=cached,
            cache_write_input_tokens=writes, output_tokens=outputs, reasoning_output_tokens=50)}


def test_luna_rates_cache_and_reasoning_are_not_double_counted():
    cost, totals, _, issue = pricing.price_record({'calls': [luna()]}, 'luna')
    assert cost == pytest.approx((300 * .20 + 600 * .02 + 100 * .25 + 200 * 1.20) / 1e6)
    assert totals['output'] == 200
    assert issue == ''


def test_cli_aggregate_does_not_trigger_request_surcharge():
    cost, _, diagnostics, _ = pricing.price_record({'calls': [luna(500000, 400000, 0)]}, 'luna')
    assert cost == pytest.approx((100000 * .20 + 400000 * .02 + 200 * 1.20) / 1e6)
    assert diagnostics['sessions_above_272k_aggregate_input'] == 1


def sonnet(uncached=10, five=5, hour=15, reads=100, outputs=5, split=True):
    usage = dict(input_tokens=uncached, cache_creation_input_tokens=five + hour,
                 cache_read_input_tokens=reads, output_tokens=outputs)
    if split:
        usage['cache_creation'] = dict(ephemeral_5m_input_tokens=five, ephemeral_1h_input_tokens=hour)
    return {'claude': {'usage': usage}}


def test_sonnet_list_prices_each_token_type_by_default():
    cost, totals, _, issue = pricing.price_record(sonnet(), 'sonnet')
    assert cost == pytest.approx((10 * 3 + 5 * 3.75 + 15 * 6 + 100 * .30 + 5 * 15) / 1e6)
    assert totals['input'] == 130 and totals['cached'] == 100 and totals['cache_write'] == 20
    assert issue == ''


def test_sonnet_all_input_switch_keeps_previous_valuation():
    cost, totals, _, _ = pricing.price_record(sonnet(), 'sonnet', valuation='sonnet_all_input')
    assert cost == (130 * 3 + 5 * 15) / 1e6
    assert totals['input'] == 130


def test_luna_price_is_the_same_under_both_valuations():
    record = {'calls': [luna()]}
    assert (pricing.price_record(record, 'luna')[0]
            == pricing.price_record(record, 'luna', valuation='sonnet_all_input')[0])


def test_sonnet_cache_write_without_duration_is_unpriced():
    cost, _, _, issue = pricing.price_record(sonnet(split=False), 'sonnet')
    assert cost is None and issue == 'missing_cache_write_duration'
    cost, _, _, issue = pricing.price_record(sonnet(five=0, hour=0, split=False), 'sonnet')
    assert cost == pytest.approx((10 * 3 + 100 * .30 + 5 * 15) / 1e6) and issue == ''
    record = sonnet()
    record['claude']['usage']['cache_creation']['ephemeral_1h_input_tokens'] = 14
    assert pricing.price_record(record, 'sonnet')[3] == 'invalid_cache_counts'


def test_unknown_valuation_is_rejected():
    with pytest.raises(ValueError, match='valuation'):
        pricing.price_record(sonnet(), 'sonnet', valuation='invoice')


def sonnet_frame(tmp_path, snapshot_cost):
    path = tmp_path / 'sonnet.json'
    path.write_text(json.dumps(sonnet()))
    return pd.DataFrame([dict(model='sonnet', record_path=str(path),
        record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        main_input_tokens=130, main_output_tokens=5, sonnet_cost_usd=snapshot_cost)])


def test_list_valuation_still_checks_the_all_input_snapshot(tmp_path):
    priced, receipt = pricing.price_dataframe(sonnet_frame(tmp_path, (130 * 3 + 5 * 15) / 1e6))
    assert priced.cost_usd.iloc[0] == pytest.approx((10 * 3 + 5 * 3.75 + 15 * 6 + 100 * .30 + 5 * 15) / 1e6)
    assert receipt['valuation_id'] == 'list'
    assert receipt['usd_per_million_tokens']['sonnet'] == pricing.SONNET_LIST_RATES
    with pytest.raises(ValueError, match='Sonnet cost differs'):
        pricing.price_dataframe(sonnet_frame(tmp_path, 1.0))
    _, old = pricing.price_dataframe(sonnet_frame(tmp_path, (130 * 3 + 5 * 15) / 1e6),
                                     valuation='sonnet_all_input')
    assert old['valuation_id'] == 'sonnet_all_input'
    assert old['usd_per_million_tokens']['sonnet'] == pricing.SONNET_ALL_INPUT_RATES


@pytest.mark.parametrize('change', [{'cached_input_tokens': 1200}, {'input_tokens': True},
                                   {'output_tokens': None}, {'cache_write_input_tokens': -1}])
def test_invalid_usage_keeps_cost_missing(change):
    call = luna()
    call['usage'].update(change)
    assert pricing.price_record({'calls': [call]}, 'luna')[0] is None


def test_legacy_cache_field_and_quota_rejection_are_explicit():
    call = luna(writes=0)
    del call['usage']['cache_write_input_tokens']
    cost, _, diagnostics, issue = pricing.price_record({'calls': [call, {'quota_rejected_before_response': True}]}, 'luna')
    assert cost is not None and issue == ''
    assert diagnostics['legacy_calls_without_cache_write_field'] == 1
    assert diagnostics['calls'] == 1


def frame(tmp_path):
    path = tmp_path / 'record.json'
    path.write_text(json.dumps({'calls': [luna()]}))
    return pd.DataFrame([dict(model='luna', record_path=str(path),
        record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        main_input_tokens=1000, main_output_tokens=200)])


def test_dataframe_and_sources_are_preserved(tmp_path):
    original = frame(tmp_path)
    expected = original.copy(deep=True)
    priced, receipt = pricing.price_dataframe(original)
    pd.testing.assert_frame_equal(original, expected)
    assert priced.cost_usd.notna().all()
    assert receipt['coverage']['luna'] == {'rows': 1, 'priced': 1, 'missing': 0}


def test_changed_record_and_snapshot_totals_are_rejected(tmp_path):
    data = frame(tmp_path)
    data.loc[0, 'main_input_tokens'] = 999
    with pytest.raises(ValueError, match='token total differs'):
        pricing.price_dataframe(data)
    data = frame(tmp_path)
    Path(data.loc[0, 'record_path']).write_text('{}')
    with pytest.raises(ValueError, match='hash changed'):
        pricing.price_dataframe(data)


def stream_event(kind, **fields):
    return json.dumps(dict(type=kind, **fields)) + '\n'


def test_timed_out_sonnet_run_is_priced_from_its_stream(tmp_path):
    # Decision record 2026-09-30_stream-pricing-timed-out-runs: a Sonnet run that
    # stopped before the final usage event is priced from the per-message usage
    # of its saved stream, with output tokens estimated from the saved text.
    usage = dict(input_tokens=2, cache_creation_input_tokens=1000, cache_read_input_tokens=5000,
                 output_tokens=3, cache_creation=dict(ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=1000))
    text = 'x' * 400
    (tmp_path / 'provider_attempt1.stdout').write_text(
        stream_event('system', subtype='thinking_tokens', estimated_tokens=50)
        + stream_event('system', subtype='thinking_tokens', estimated_tokens=120)
        + stream_event('assistant', message=dict(id='m1', usage=usage, content=[dict(type='text', text=text)]))
        + stream_event('assistant', message=dict(id='m1', usage=usage, content=[dict(type='text', text=text)]))
        + stream_event('system', subtype='thinking_tokens', estimated_tokens=30))
    record = {'calls': [{'ok': False, 'timed_out': True}]}
    cost, totals, _, issue = pricing.price_record(record, 'sonnet', record_dir=tmp_path)
    assert issue == 'estimated_from_stream'
    # One message (duplicate events share an id); thinking blocks 120 + 30; 800 visible characters / 4.
    assert totals['output'] == 120 + 30 + 200
    assert totals['cached'] == 5000 and totals['cache_write'] == 1000
    assert cost == pytest.approx((2 * 3.0 + 1000 * 6.0 + 5000 * 0.30 + 350 * 15.0) / 1e6)


def test_run_without_calls_or_stream_stays_unpriced(tmp_path):
    cost, _, _, issue = pricing.price_record({'calls': [{'ok': False}]}, 'sonnet', record_dir=tmp_path)
    assert cost is None and issue == 'missing_provider_usage'

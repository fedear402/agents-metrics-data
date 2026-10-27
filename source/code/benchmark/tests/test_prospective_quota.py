"""Preserve concurrent quota holds and evidence of generated responses."""
from pathlib import Path
import importlib.util

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'prospective_collect.py'
spec = importlib.util.spec_from_file_location('prospective_quota_tested', SOURCE)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def rejection(reset, observed=100):
    return dict(status='rejected', reset_at=reset, observed_at=observed, evidence=[])


def test_unknown_reset_survives_concurrent_known_reset_and_restart(tmp_path):
    path = tmp_path/'quota.json'
    q = p.QuotaCoordinator(path, clock=lambda: 200)
    q.reject('sonnet', rejection(None))
    q.reject('sonnet', rejection(150))
    restarted = p.QuotaCoordinator(path, clock=lambda: 200)
    for coordinator in (q, restarted):
        with pytest.raises(p.QuotaUnknown):
            coordinator.acquire('sonnet')


def test_old_probe_success_preserves_new_concurrent_rejection(tmp_path):
    q = p.QuotaCoordinator(tmp_path/'quota.json', clock=lambda: 200)
    q.reject('sonnet', rejection(150))
    probe = q.acquire('sonnet')
    q.reject('sonnet', rejection(300, 201))
    q.release('sonnet', probe)
    assert q.state['sonnet']['reset_at'] == 300
    assert not q.probes


def test_successful_current_probe_clears_only_its_provider(tmp_path):
    q = p.QuotaCoordinator(tmp_path/'quota.json', clock=lambda: 200)
    q.reject('sonnet', rejection(150))
    q.reject('luna', rejection(None))
    q.release('sonnet', q.acquire('sonnet'))
    assert 'sonnet' not in q.state and 'luna' in q.state


@pytest.mark.parametrize('call,events', [
    ({'ok':False,'text':'Generated response'}, []),
    ({'ok':False,'usage':{'output_tokens':10}}, []),
    ({'ok':False,'usage':{'completion_tokens':10}}, []),
    ({'ok':False}, [{'type':'item.completed','item':{'type':'assistant_message','text':'Generated response'}}]),
    ({'ok':False}, [{'type':'item.started','item':{'type':'reasoning','text':'Reasoning began'}}]),
])
def test_generated_response_evidence_prevents_automatic_retry(call, events):
    assert p.response_started(call, events)


def test_input_usage_alone_does_not_establish_generated_response():
    assert not p.response_started({'ok':False,'usage':{'input_tokens':100,'output_tokens':0}}, [])


@pytest.mark.parametrize('reset', [1_700_000_000_000, 1_700_000_000+86401])
def test_implausible_reset_units_and_horizon_require_review(reset):
    event={'rate_limit_info':{'status':'rejected','resetsAt':reset}}
    assert p.parse_quota([event],1_700_000_000)['reset_at'] is None


def test_reset_horizon_includes_chosen_upper_boundary():
    event={'rate_limit_info':{'status':'rejected','resetsAt':1_700_000_000+86400}}
    assert p.parse_quota([event],1_700_000_000)['reset_at']==1_700_000_000+86400


def test_invalid_persisted_reset_requires_review_without_waiting(tmp_path):
    path=tmp_path/'quota.json'
    p.save(path,{'sonnet':rejection(1_700_000_000_000,1_700_000_000)})
    q=p.QuotaCoordinator(path,clock=lambda:1_700_000_100,
                         sleep=lambda seconds:pytest.fail('Invalid reset caused a wait'))
    with pytest.raises(p.QuotaUnknown):
        q.acquire('sonnet')

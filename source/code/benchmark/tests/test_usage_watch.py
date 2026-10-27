"""Offline checks of the provider usage watcher. No test calls a model provider.

The recorded events come from saved collection streams. The Codex status poller is
replaced by a local stand-in or by a fake JSON-RPC server that runs with the test
interpreter.
"""
from pathlib import Path
import copy
import importlib
import importlib.util
import json
import sys
import threading

import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import usage_watch as uw  # noqa: E402

ROOT = HERE.parents[1]

# Recorded 2026-09-21 Sonnet stream events (production_repairs), both saved shapes.
CLAUDE_ALLOWED = {"type": "rate_limit_event", "rate_limit_info": {
    "status": "allowed", "resetsAt": 1790046000, "rateLimitType": "five_hour", "overageStatus": "rejected",
    "overageDisabledReason": "org_level_disabled", "isUsingOverage": False,
    "unifiedWindows": {"five_hour": {"utilization": 0.05, "resetsAt": 1790046000},
                       "seven_day": {"utilization": 0.75, "resetsAt": 1790204400}}}}
CLAUDE_WARNING = {"type": "rate_limit_event", "rate_limit_info": {
    "status": "allowed_warning", "resetsAt": 1790204400, "rateLimitType": "seven_day", "utilization": 0.82,
    "isUsingOverage": False, "surpassedThreshold": 0.75,
    "unifiedWindows": {"five_hour": {"utilization": 0.49, "resetsAt": 1790046000},
                       "seven_day": {"utilization": 0.82, "resetsAt": 1790204400}}}}
# Codex app-server account/rateLimits/read result, 2026-09-26 (account fields removed).
CODEX_RESULT = {"ordinaryUsageAllowed": True, "rateLimits": {
    "limitId": "codex", "primary": {"usedPercent": 3, "windowDurationMins": 10080, "resetsAt": 1790855882},
    "secondary": None, "credits": {"hasCredits": False, "unlimited": False, "balance": "0"},
    "planType": "pro", "rateLimitReachedType": None},
    "rateLimitResetCredits": {"availableCount": 3, "credits": []}}
# Codex session file token_count payload, 2026-09-24.
CODEX_SESSION = {"limit_id": "codex", "primary": {"used_percent": 3.0, "window_minutes": 10080, "resets_at": 1790855882},
                 "secondary": None, "credits": {"has_credits": False, "unlimited": False, "balance": "0"},
                 "plan_type": "pro", "rate_limit_reached_type": None}

T0 = 1790040000  # 2026-09-22 11:20 UTC, inside both recorded Claude windows.


class Clock:
    def __init__(self, value=T0):
        self.value = float(value)
        self.sleeps = []

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


def claude_event(five=0.1, seven=0.2, status='allowed', five_reset=T0 + 3600, seven_reset=T0 + 86400,
                 overage=False, limit_type='five_hour'):
    reset = five_reset if limit_type == 'five_hour' else seven_reset
    return {"type": "rate_limit_event", "rate_limit_info": {
        "status": status, "resetsAt": reset, "rateLimitType": limit_type, "isUsingOverage": overage,
        "unifiedWindows": {"five_hour": {"utilization": five, "resetsAt": five_reset},
                           "seven_day": {"utilization": seven, "resetsAt": seven_reset}}}}


def codex_result(percent=3, reset=T0 + 86400, allowed=True, reached=None):
    result = copy.deepcopy(CODEX_RESULT)
    result['ordinaryUsageAllowed'] = allowed
    result['rateLimits'].update(primary=dict(usedPercent=percent, windowDurationMins=10080, resetsAt=reset),
                                rateLimitReachedType=reached)
    return result


def slow_config():
    # The shipped settings have no slow level; these tests set one to check the spacing code.
    config = copy.deepcopy(uw.load_config())
    config['providers']['claude']['thresholds']['five_hour']['slow_at'] = 0.85
    return config


def make(tmp_path, clock=None, poller=None, stop=None, config=None):
    clock = clock or Clock()
    return uw.Watcher(config=config, state_dir=tmp_path / 'watch', clock=clock, sleep=clock.sleep,
                      stop=stop or threading.Event(), codex_poller=poller or (lambda timeout: None))


# Configuration

def test_config_labels_thresholds_as_chosen_limits():
    config = uw.load_config()
    assert 'chosen' in config['label'].lower() and 'not measured' in config['label'].lower()
    for provider in ('claude', 'codex'):
        for window, limits in config['providers'][provider]['thresholds'].items():
            # Only provider limits bind: no slow level below the pause level.
            assert 0 < limits['slow_at'] == limits['pause_at'] <= 1, (provider, window)
    # sonnet: earlier collectors; sonnet55: the final-bank runner (decision record 2026-09-30_sonnet-5-5)
    assert config['providers']['claude']['models'] == ['sonnet', 'sonnet55']
    assert config['providers']['codex']['models'] == ['luna']


def test_models_map_to_providers():
    config = uw.load_config()
    assert uw.provider_for('sonnet', config) == 'claude'
    assert uw.provider_for('luna', config) == 'codex'
    assert uw.provider_for('codex', config) == 'codex'
    assert uw.provider_for('deepseek', config) is None


# Parsing recorded events

def test_both_recorded_claude_shapes_parse_to_windows():
    first = uw.claude_observation([CLAUDE_ALLOWED], T0, 'stream')
    assert first['windows'] == {'five_hour': dict(used=0.05, resets_at=1790046000),
                                'seven_day': dict(used=0.75, resets_at=1790204400)}
    assert first['status'] == 'allowed' and first['is_using_overage'] is False
    second = uw.claude_observation([CLAUDE_ALLOWED, CLAUDE_WARNING], T0, 'stream')
    assert second['status'] == 'allowed_warning' and second['windows']['seven_day']['used'] == 0.82
    assert uw.claude_observation([{'type': 'result'}], T0, 'stream') is None


def test_top_level_window_is_used_without_unified_windows():
    event = {'type': 'rate_limit_event', 'rate_limit_info': {
        'status': 'rejected', 'resetsAt': T0 + 600, 'rateLimitType': 'five_hour', 'utilization': 1.0}}
    obs = uw.claude_observation([event], T0, 'stream')
    assert obs['windows']['five_hour'] == dict(used=1.0, resets_at=T0 + 600)
    assert obs['rejected_resets_at'] == T0 + 600


def test_millisecond_or_invalid_reset_is_not_used():
    event = claude_event(five_reset=1_790_046_000_000)
    assert uw.claude_observation([event], T0, 'stream')['windows']['five_hour']['resets_at'] is None


def test_codex_app_server_and_session_snapshots_agree():
    observed = 1790405000  # 2026-09-26, when the recorded answer was read
    polled = uw.codex_observation(CODEX_RESULT, observed, 'codex app-server')
    session = uw.codex_observation(CODEX_SESSION, observed, 'session')
    assert polled['windows'] == session['windows'] == {'seven_day': dict(used=0.03, resets_at=1790855882)}
    assert polled['plan_type'] == session['plan_type'] == 'pro'
    assert polled['ordinary_usage_allowed'] is True and session['ordinary_usage_allowed'] is None
    assert polled['reset_credits_available'] == 3
    assert 'accountId' not in json.dumps(polled)


def test_codex_limits_inside_exec_events_are_read():
    events = [{'type': 'event_msg', 'payload': {'type': 'token_count', 'rate_limits': CODEX_SESSION}}]
    obs = uw.observation_from_events('codex', events, 1790405000, 'stream')
    assert obs['windows']['seven_day']['used'] == 0.03


# Session payload of a second limit bucket (five-hour primary, weekly secondary), shaped as
# recorded on 2026-09-13; reset times moved next to CODEX_SESSION.
CODEX_OTHER_BUCKET = {"limit_id": "codex_bengalfox", "limit_name": None,
                      "primary": {"used_percent": 0.0, "window_minutes": 300, "resets_at": 1790420000},
                      "secondary": {"used_percent": 0.0, "window_minutes": 10080, "resets_at": 1790900000},
                      "plan_type": "pro", "rate_limit_reached_type": None}


def test_second_bucket_in_session_events_does_not_replace_the_codex_week():
    # Session files hold token_count events of several limit buckets. The last event can
    # belong to another bucket; its weekly window must not overwrite the codex week.
    events = [{'type': 'event_msg', 'payload': {'type': 'token_count', 'rate_limits': CODEX_SESSION}},
              {'type': 'event_msg', 'payload': {'type': 'token_count', 'rate_limits': CODEX_OTHER_BUCKET}}]
    obs = uw.observation_from_events('codex', events, 1790405000, 'stream')
    assert obs['windows']['seven_day'] == dict(used=0.03, resets_at=1790855882)
    assert obs['windows']['codex_bengalfox:five_hour'] == dict(used=0.0, resets_at=1790420000)
    assert obs['windows']['codex_bengalfox:seven_day'] == dict(used=0.0, resets_at=1790900000)


def test_session_fallback_keeps_the_newest_payload_of_each_bucket(tmp_path):
    folder = tmp_path / 'sessions/2026/09/26'
    folder.mkdir(parents=True)
    lines = [{'timestamp': '2026-09-26T07:00:00Z', 'type': 'event_msg',
              'payload': {'type': 'token_count', 'rate_limits': CODEX_SESSION}},
             {'timestamp': '2026-09-26T07:05:00Z', 'type': 'event_msg',
              'payload': {'type': 'token_count', 'rate_limits': CODEX_OTHER_BUCKET}}]
    (folder / 'rollout-2026-09-26T07-00-00-x.jsonl').write_text(''.join(json.dumps(line) + '\n' for line in lines))
    snapshot = uw.latest_codex_session_snapshot(tmp_path / 'sessions')
    assert snapshot['observed_at'] == 1790406000.0  # the codex bucket reading, 07:00 UTC
    obs = uw.codex_observation(snapshot['payload'], snapshot['observed_at'], snapshot['source'])
    assert obs['windows']['seven_day'] == dict(used=0.03, resets_at=1790855882)
    assert set(obs['buckets']) == {'codex', 'codex_bengalfox'}


def test_app_server_bucket_view_is_read_and_only_collection_buckets_pace(tmp_path):
    result = codex_result(10)
    luna = dict(limitId='codex_luna', normalModelSlug='gpt-5.6-luna', rateLimitReachedType=None,
                primary=dict(usedPercent=98, windowDurationMins=300, resetsAt=T0 + 3600))
    other = dict(limitId='codex_other', normalModelSlug='gpt-6-astra', rateLimitReachedType='rate_limit_reached',
                 primary=dict(usedPercent=100, windowDurationMins=300, resetsAt=T0 + 7200))
    result['rateLimitsByLimitId'] = {'codex': result['rateLimits'], 'codex_luna': luna, 'codex_other': other}
    obs = uw.codex_observation(result, T0, 'codex app-server')
    assert obs['windows']['seven_day']['used'] == 0.1
    assert obs['windows']['codex_luna:five_hour'] == dict(used=0.98, resets_at=T0 + 3600)
    assert obs['windows']['codex_other:five_hour']['used'] == 1.0
    w = make(tmp_path, poller=lambda timeout: result)
    decision = w.decide('codex')
    # The Luna bucket pauses the collection; the bucket of another model does not.
    assert decision['decision'] == 'pause' and decision['until'] == T0 + 3600 + 60
    assert 'codex_luna' in decision['reason'] and 'codex_other' not in decision['reason']
    assert 'codex_other:five_hour' not in decision['windows']
    del result['rateLimitsByLimitId']['codex_luna']
    w2 = make(tmp_path / 'second', poller=lambda timeout: result)
    assert w2.decide('codex')['decision'] == 'run'


def test_reset_time_jitter_keeps_the_newer_higher_reading(tmp_path):
    # Readings of one Codex window report reset times that differ by 1 to 2 seconds
    # (2026-09-26: 1791058681 and 1791058682). They describe the same window.
    w = make(tmp_path)
    reset = T0 + 86400
    w.observe(uw.codex_observation(codex_result(96, reset=reset + 1), T0, 'poll 1'))
    w.observe(uw.codex_observation(codex_result(98, reset=reset), T0 + 120, 'poll 2'))
    assert w.latest('codex')['windows']['seven_day'] == dict(used=0.98, resets_at=reset)
    assert w.decide('codex', poll=False)['decision'] == 'pause'
    # An older, lower reading that arrives late with a jittered reset does not lower usage.
    w.observe(uw.codex_observation(codex_result(90, reset=reset + 2), T0 + 60, 'late session file'))
    assert w.latest('codex')['windows']['seven_day'] == dict(used=0.98, resets_at=reset)
    # A reset more than the tolerance later is a new window.
    w.observe(uw.codex_observation(codex_result(1, reset=reset + 7 * 86400), T0 + 90000, 'next week'))
    assert w.latest('codex')['windows']['seven_day'] == dict(used=0.01, resets_at=reset + 7 * 86400)


# Decisions

@pytest.mark.parametrize('five,seven,expected', [
    (0.05, 0.75, 'run'), (0.86, 0.2, 'run'), (0.2, 0.93, 'run'), (0.95, 0.2, 'pause'), (0.2, 0.98, 'pause')])
def test_claude_thresholds(tmp_path, five, seven, expected):
    w = make(tmp_path)
    w.observe_events('claude', [claude_event(five, seven)], T0, 'stream')
    decision = w.decide('claude')
    assert decision['decision'] == expected
    if expected == 'pause':
        window = 'five_hour' if five >= 0.95 else 'seven_day'
        reset = T0 + 3600 if window == 'five_hour' else T0 + 86400
        assert decision['until'] == reset + w.config['resume_margin_seconds']
        assert window in decision['reason']


def test_claude_rejection_pauses_until_its_reset(tmp_path):
    w = make(tmp_path)
    w.observe_events('claude', [claude_event(0.5, 0.5, status='rejected')], T0, 'stream')
    decision = w.decide('claude')
    assert decision['decision'] == 'pause' and decision['until'] == T0 + 3600 + 60


def test_overage_use_pauses(tmp_path):
    w = make(tmp_path)
    w.observe_events('claude', [claude_event(0.5, 0.5, overage=True)], T0, 'stream')
    decision = w.decide('claude')
    assert decision['decision'] == 'pause' and 'extra usage' in decision['reason']


def test_expired_window_no_longer_limits(tmp_path):
    clock = Clock()
    w = make(tmp_path, clock)
    w.observe_events('claude', [claude_event(0.99, 0.2)], T0, 'stream')
    assert w.decide('claude')['decision'] == 'pause'
    clock.value = T0 + 3600 + 1
    assert w.decide('claude')['decision'] == 'pause'  # inside the chosen resume margin
    clock.value = T0 + 3600 + w.config['resume_margin_seconds']
    decision = w.decide('claude')
    assert decision['decision'] == 'run' and 'five_hour' not in decision['windows']


def test_older_reading_of_same_window_does_not_lower_usage(tmp_path):
    w = make(tmp_path)
    w.observe_events('claude', [claude_event(0.9, 0.5)], T0, 'late stream')
    w.observe_events('claude', [claude_event(0.4, 0.3)], T0 - 600, 'early stream finishing late')
    assert w.latest('claude')['windows']['five_hour']['used'] == 0.9
    w.observe_events('claude', [claude_event(0.01, 0.51, five_reset=T0 + 20000)], T0 + 10, 'new window')
    assert w.latest('claude')['windows']['five_hour'] == dict(used=0.01, resets_at=T0 + 20000)
    assert w.latest('claude')['windows']['seven_day']['used'] == 0.51


def test_no_observation_runs(tmp_path):
    decision = make(tmp_path).decide('claude')
    assert decision['decision'] == 'run' and decision['observed_at'] is None


@pytest.mark.parametrize('percent,allowed,reached,expected', [
    (3, True, None, 'run'), (91, True, None, 'run'), (97, True, None, 'pause'),
    (40, False, None, 'pause'), (40, True, 'rate_limit_reached', 'pause')])
def test_codex_decisions(tmp_path, percent, allowed, reached, expected):
    clock = Clock()
    w = make(tmp_path, clock, poller=lambda timeout: codex_result(percent, allowed=allowed, reached=reached))
    decision = w.decide('codex')
    assert decision['decision'] == expected
    if expected == 'pause':
        assert decision['until'] == T0 + 86400 + 60


def test_unknown_model_is_never_gated(tmp_path):
    w = make(tmp_path)
    assert w.gate('deepseek') is None


# Gate

def test_gate_waits_for_reset_then_resumes(tmp_path):
    clock = Clock()
    w = make(tmp_path, clock)
    w.observe_events('claude', [claude_event(0.97, 0.2, five_reset=T0 + 300)], T0, 'stream')
    decision = w.gate('sonnet')
    assert decision['decision'] == 'run'
    assert clock.value >= T0 + 300 + 60
    assert max(clock.sleeps) <= w.config['recheck_seconds']


def test_gate_stop_raises_during_pause(tmp_path):
    clock = Clock()
    stop = threading.Event()
    w = make(tmp_path, clock, stop=stop)
    w.observe_events('claude', [claude_event(0.97, 0.2)], T0, 'stream')
    w.sleep = lambda seconds: stop.set()
    with pytest.raises(uw.WatchStopped):
        w.gate('sonnet')


def test_claude_rejection_without_reset_retries_one_call_every_15_minutes(tmp_path):
    clock = Clock()
    w = make(tmp_path, clock)
    assert w.config['unknown_reset_retry_seconds'] == 900
    assert w.gate('sonnet')['decision'] == 'run'           # the call that the provider then rejects
    event = {'type': 'rate_limit_event', 'rate_limit_info': {'status': 'rejected', 'rateLimitType': 'five_hour'}}
    w.observe_events('claude', [event], clock.value, 'stream')
    first = w.gate('sonnet')
    assert first['decision'] == 'retry' and clock.value >= T0 + 900
    at = clock.value
    w.observe_events('claude', [event], clock.value, 'stream')   # the retry is rejected again
    assert w.gate('sonnet')['decision'] == 'retry' and clock.value >= at + 900
    allowed = {'type': 'rate_limit_event', 'rate_limit_info': {'status': 'allowed', 'unifiedWindows': {
        'five_hour': {'utilization': 0.1, 'resetsAt': clock.value + 3600}}}}
    w.observe_events('claude', [allowed], clock.value, 'stream')
    at = clock.value
    assert w.gate('sonnet')['decision'] == 'run' and clock.value == at


def test_claude_rejection_with_reset_pauses_until_reset_plus_margin(tmp_path):
    clock = Clock()
    w = make(tmp_path, clock)
    event = {'type': 'rate_limit_event', 'rate_limit_info': {'status': 'rejected', 'rateLimitType': 'five_hour',
                                                             'resetsAt': T0 + 1200}}
    w.observe_events('claude', [event], T0, 'stream')
    assert w.decide('claude')['decision'] == 'pause'
    assert w.gate('sonnet')['decision'] == 'run'
    assert clock.value >= T0 + 1200 + w.config['resume_margin_seconds']


def test_claude_result_error_text_is_read_as_a_rejection(tmp_path):
    w = make(tmp_path)
    reset = int(T0 + 600)
    events = [{'type': 'result', 'is_error': True, 'result': f'Claude AI usage limit reached|{reset}'}]
    observation = uw.claude_observation(events, T0, 's')
    assert observation['status'] == 'rejected' and observation['rejected_resets_at'] == reset
    no_reset = uw.claude_observation([{'type': 'result', 'is_error': True, 'result': "You've hit your limit"}], T0, 's')
    assert no_reset['status'] == 'rejected' and no_reset['rejected_resets_at'] is None
    assert uw.claude_observation([{'type': 'result', 'is_error': True, 'result': 'API Error: 500'}], T0, 's') is None
    w.observe_events('claude', events, T0, 's')
    assert w.decide('claude')['until'] == reset + w.config['resume_margin_seconds']


def test_codex_pause_without_reset_keeps_polling(tmp_path):
    clock = Clock()
    answers = [codex_result(99, reset=None, allowed=False), codex_result(99, reset=None, allowed=False), codex_result(5)]
    w = make(tmp_path, clock, poller=lambda timeout: answers.pop(0) if len(answers) > 1 else answers[0])
    assert w.gate('luna')['decision'] == 'run'
    assert clock.value >= T0 + 2 * w.config['providers']['codex']['poll_seconds_paused']


def test_slow_mode_spaces_call_starts(tmp_path):
    clock = Clock()
    w = make(tmp_path, clock, config=slow_config())
    w.observe_events('claude', [claude_event(0.9, 0.2)], T0, 'stream')
    spacing = w.config['providers']['claude']['slow_spacing_seconds']
    assert w.gate('sonnet')['decision'] == 'slow' and clock.value == T0
    assert w.gate('sonnet')['decision'] == 'slow'
    assert clock.value == pytest.approx(T0 + spacing)


def test_spacing_is_shared_through_the_state_folder(tmp_path):
    clock = Clock()
    first, second = make(tmp_path, clock, config=slow_config()), make(tmp_path, clock, config=slow_config())
    first.observe_events('claude', [claude_event(0.9, 0.2)], T0, 'stream')
    first.gate('sonnet')
    second.gate('sonnet')
    assert clock.value == pytest.approx(T0 + first.config['providers']['claude']['slow_spacing_seconds'])


def test_codex_poll_is_cached_across_watchers(tmp_path):
    clock = Clock()
    polls = []
    def poller(timeout):
        polls.append(timeout)
        return codex_result(10)
    first, second = make(tmp_path, clock, poller), make(tmp_path, clock, poller)
    first.decide('codex'); second.decide('codex')
    assert len(polls) == 1
    clock.value += first.config['providers']['codex']['poll_seconds'] + 1
    second.decide('codex')
    assert len(polls) == 2


def test_failed_poll_is_logged_and_not_repeated_at_once(tmp_path):
    clock = Clock()
    polls = []
    def poller(timeout):
        polls.append(1)
        raise uw.PollError('app-server exited')
    w = make(tmp_path, clock, poller)
    assert w.decide('codex')['decision'] == 'run'
    w.decide('codex')
    assert len(polls) == 1
    assert 'app-server exited' in json.loads((tmp_path / 'watch/state.json').read_text())['codex_poll']['last_error']


# Records

def test_log_and_status_file(tmp_path):
    clock = Clock()
    w = make(tmp_path, clock, poller=lambda timeout: codex_result(10), config=slow_config())
    w.observe_events('claude', [claude_event(0.9, 0.2)], T0, 'stream')
    w.gate('sonnet'); w.gate('sonnet')
    w.write_status()
    lines = [json.loads(line) for line in (tmp_path / 'watch/usage_watch.jsonl').read_text().splitlines()]
    claude = [line for line in lines if line['provider'] == 'claude']
    assert claude[0]['decision'] == 'slow' and claude[0]['config_sha256'] == w.config_sha256
    assert len(claude) == 2  # one gate line and one status line; the repeated gate decision is throttled
    assert {line['provider'] for line in lines} == {'claude', 'codex'}
    text = (tmp_path / 'watch/status.txt').read_text()
    assert 'claude' in text and 'codex' in text and 'chosen' in text.lower()
    assert '90%' in text and '10%' in text


def test_status_scans_saved_streams(tmp_path):
    folder = tmp_path / 'runs/sonnet/0_4/runs/abc/job/attempt_0001'
    folder.mkdir(parents=True)
    (folder / 'provider_attempt1.stdout').write_text(json.dumps(CLAUDE_WARNING) + '\n{"type":"result"}\n')
    luna = tmp_path / 'runs/luna/0_4/runs/abc/job/attempt_0001'
    luna.mkdir(parents=True)
    (luna / 'provider_attempt1.stdout').write_text('{"type":"turn.completed","usage":{}}\n')
    clock = Clock(1790040000)
    w = make(tmp_path, clock)
    found = w.scan_streams([tmp_path / 'runs'])
    assert found == 1
    assert w.latest('claude')['windows']['seven_day']['used'] == 0.82


def test_replay_of_saved_collection_streams(tmp_path):
    root = ROOT / 'data/runs/production_repairs_2026-09-21/sonnet'
    if not root.is_dir():
        pytest.skip('saved production_repairs streams are not present')
    rows = uw.replay([root], config=uw.load_config(), limit=400)
    assert rows and all(row['decision'] in ('run', 'slow', 'pause') for row in rows)
    # The 2026-09-21 window peaked at 0.69 five-hour and 0.83 seven-day utilization.
    assert all(row['decision'] == 'run' for row in rows)
    assert max(row['windows']['seven_day']['used'] for row in rows) <= 0.83


def test_poll_codex_reads_fake_app_server(tmp_path):
    answer = tmp_path / 'answer.json'
    answer.write_text(json.dumps(CODEX_RESULT))
    seen = tmp_path / 'seen.jsonl'
    server = tmp_path / 'fake_server.py'
    server.write_text(
        'import json, sys\n'
        f'answer = json.load(open({str(answer)!r}))\n'
        f'seen = open({str(seen)!r}, "a")\n'
        'for line in sys.stdin:\n'
        '    msg = json.loads(line)\n'
        '    seen.write(line); seen.flush()\n'
        '    if msg.get("method") == "initialize":\n'
        '        print(json.dumps({"id": msg["id"], "result": {}}), flush=True)\n'
        '    elif msg.get("method") == "account/rateLimits/read":\n'
        '        print(json.dumps({"method": "noise", "params": {}}), flush=True)\n'
        '        print(json.dumps({"id": msg["id"], "result": answer}), flush=True)\n')
    result = uw.poll_codex(10, command=[sys.executable, str(server)])
    assert result['rateLimits']['primary']['usedPercent'] == 3
    methods = [json.loads(line).get('method') for line in seen.read_text().splitlines()]
    assert methods == ['initialize', 'initialized', 'account/rateLimits/read']


def test_poll_codex_failure_raises_poll_error(tmp_path):
    server = tmp_path / 'broken.py'
    server.write_text('import sys\nsys.exit(3)\n')
    with pytest.raises(uw.PollError):
        uw.poll_codex(5, command=[sys.executable, str(server)])


# Collector integration

def load_watched(tmp_path, watcher):
    spec = importlib.util.spec_from_file_location('watched_collector_tested', HERE / 'prospective_collect_watched.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_watcher(watcher)
    return module


def test_collector_asks_watcher_before_each_request(tmp_path):
    clock = Clock()
    w = make(tmp_path, clock)
    central = load_watched(tmp_path, w).central
    assert w.stop is central.STOP
    central.STOP.clear()
    w.observe_events('claude', [claude_event(0.97, 0.2, five_reset=T0 + 120)], T0, 'stream')
    quotas = central.QuotaCoordinator(tmp_path / 'quota.json', clock=clock, sleep=clock.sleep)
    assert quotas.acquire('sonnet') is False
    assert clock.value >= T0 + 120 + 60
    assert quotas.acquire('luna') is False  # Codex has no observation and no poller result


def test_collector_pause_maps_to_central_exceptions(tmp_path):
    clock = Clock()
    w = make(tmp_path, clock)
    central = load_watched(tmp_path, w).central
    assert w.stop is central.STOP
    central.STOP.clear()
    quotas = central.QuotaCoordinator(tmp_path / 'quota.json', clock=clock, sleep=clock.sleep)
    w.observe_events('claude', [claude_event(0.97, 0.2)], T0, 'stream')
    w.sleep = lambda seconds: central.STOP.set()
    with pytest.raises(central.Paused):
        quotas.acquire('sonnet')
    central.STOP.clear()
    # A rejection without a reset time no longer holds the collector: one retry call may start.
    w.observe_events('claude', [{'type': 'rate_limit_event', 'rate_limit_info': {'status': 'rejected'}}], T0 + 1, 's')
    quotas.acquire('sonnet')
    lines = (w.state_dir / 'usage_watch.jsonl').read_text().splitlines()
    assert json.loads(lines[-1])['decision'] == 'retry'


def test_watcher_fault_falls_back_to_central_quota_handling(tmp_path, monkeypatch, capsys):
    clock = Clock()
    w = make(tmp_path, clock)
    central = load_watched(tmp_path, w).central
    central.STOP.clear()
    monkeypatch.setattr(w, 'gate', lambda model: 1 / 0)
    quotas = central.QuotaCoordinator(tmp_path / 'quota.json', clock=clock, sleep=clock.sleep)
    assert quotas.acquire('sonnet') is False
    assert 'usage_watch_error' in capsys.readouterr().err


def test_unexpected_poller_error_is_recorded(tmp_path):
    w = make(tmp_path, poller=lambda timeout: {}['missing'])
    assert w.decide('codex')['decision'] == 'run'
    assert 'KeyError' in json.loads((tmp_path / 'watch/state.json').read_text())['codex_poll']['last_error']


def test_finished_call_feeds_watcher_and_errors_never_break_calls(tmp_path, monkeypatch):
    clock = Clock()
    w = make(tmp_path, clock)
    central = load_watched(tmp_path, w).central
    runtime = central.load_runtime(tmp_path / 'work')
    assert getattr(runtime.c.call_model, 'usage_watch_wrapped', False)
    def fake_call(text, cwd, job, destination, attempt):
        (destination / f'provider_attempt{attempt}.stdout').write_text(json.dumps(claude_event(0.42, 0.3)) + '\n')
        return dict(ok=True, text='x', raw_stream_file=f'provider_attempt{attempt}.stdout')
    call = uw.wrap_call_model(fake_call, lambda: w)
    destination = tmp_path / 'attempt'
    destination.mkdir()
    result = call('prompt', tmp_path, dict(model='sonnet'), destination, 1)
    assert result['ok'] and w.latest('claude')['windows']['five_hour']['used'] == 0.42
    monkeypatch.setattr(w, 'observe_events', lambda *args, **kwargs: 1 / 0)
    assert call('prompt', tmp_path, dict(model='sonnet'), destination, 2)['ok']


def test_run_job_waits_for_watcher_then_completes(tmp_path):
    clock = Clock()
    w = make(tmp_path, clock)
    central = load_watched(tmp_path, w).central
    assert w.stop is central.STOP
    central.STOP.clear()
    spec = importlib.util.spec_from_file_location('prospective_collect_fixture', HERE / 'tests/test_prospective_collect.py')
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    job, manifest, runtime, _, provider, execute, calls, _ = fixtures.fixture(tmp_path)
    job = dict(job, model='sonnet'); manifest['jobs'] = [job]
    manifest['protocol']['prompts'] = {central.prompt_key(job): 'BASE EXACT PROMPT'}
    w.observe_events('claude', [claude_event(0.99, 0.2, five_reset=T0 + 500)], T0, 'stream')
    quotas = central.QuotaCoordinator(tmp_path / 'quota.json', clock=clock, sleep=clock.sleep)
    record = central.run_job(job, manifest, tmp_path / 'runs', runtime, quotas, provider, execute)
    assert record['status'] == 'completed' and len(calls) == 1
    assert clock.value >= T0 + 500 + 60


def test_watched_prepare_records_policy(tmp_path, monkeypatch):
    w = make(tmp_path)
    module = load_watched(tmp_path, w)
    central = module.central
    def base_prepare(config_path, manifest_path):
        protocol = dict(version='x', policy={}, source_hashes={})
        manifest = dict(protocol=protocol, protocol_sha256=central.digest(protocol), jobs=[])
        central.save(manifest_path, manifest)
        return manifest
    monkeypatch.setattr(module, 'BASE_PREPARE', base_prepare)
    manifest = module.prepare(tmp_path / 'config.json', tmp_path / 'out/manifest.json')
    policy = manifest['protocol']['policy']
    assert policy['usage_watch_version'] == uw.VERSION
    assert policy['usage_watch_config']['label'] == uw.load_config()['label']
    assert str(HERE / 'usage_watch.py') in manifest['protocol']['source_hashes']
    assert manifest['protocol_sha256'] == central.digest(manifest['protocol'])


def test_watched_queue_points_to_watched_collector():
    spec = importlib.util.spec_from_file_location('watched_queue_tested', HERE / 'production_queue_watched.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.central_queue.COLLECTOR.name == 'prospective_collect_watched.py'


def test_codex_hard_pause_ignores_a_raised_override(tmp_path):
    # A caller that raises the pause level to 100 percent still pauses at 99 percent (no credit use).
    config = copy.deepcopy(uw.load_config())
    for limits in config['providers']['codex']['thresholds'].values():
        limits['slow_at'] = limits['pause_at'] = 1.0
    clock = Clock()
    w = make(tmp_path, clock, poller=lambda timeout: codex_result(99), config=config)
    assert w.decide('codex')['decision'] == 'pause'
    w2 = make(tmp_path / 'b', clock, poller=lambda timeout: codex_result(98), config=config)
    assert w2.decide('codex')['decision'] == 'run'

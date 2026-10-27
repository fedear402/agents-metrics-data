"""Tests for the weekly fill-target pacing (added 2026-07-04).

The weekly-fill controller spaces runs so the weekly all-models meter reaches
the fill target exactly at the weekly reset, re-estimating the live
weekly-percent-per-run from same-week snapshots.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import harness  # noqa: E402

NOW = 1_783_170_000.0
WEEK_RESET = NOW + 50 * 3600.0  # 50h to the weekly reset


def make_controller(tmp_path: Path, **kwargs) -> harness.PacingController:
    params = dict(
        state_file=tmp_path / "pacing_state.json",
        weekly_fill_target_percent=90.0,
        default_weekly_percent_per_run=0.07,
        time_func=lambda: NOW,
        sleep_func=lambda _s: None,
    )
    params.update(kwargs)
    return harness.PacingController(**params)


def snapshot(weekly_percent: float | None) -> harness.ClaudeUsageSnapshot:
    return harness.ClaudeUsageSnapshot(
        observed_at=NOW,
        source="test",
        weekly_all_percent=weekly_percent,
        weekly_all_reset_ts=WEEK_RESET,
    )


def write_state(tmp_path: Path, snaps=None, events=None) -> None:
    (tmp_path / "pacing_state.json").write_text(
        json.dumps({"usage_snapshots": snaps or [], "events": events or []})
    )


def snap_row(pct, observed_at=NOW, reset=WEEK_RESET):
    return {"weekly_all_percent": pct, "observed_at": observed_at, "weekly_all_reset_ts": reset}


def ev(ts):
    return {"ts": ts, "cell_id": "x", "cost_usd": 0.2}


def test_failclosed_no_cap_configured_is_inert(tmp_path: Path) -> None:
    pacing = make_controller(tmp_path, weekly_all_hard_limit_percent=None)
    write_state(tmp_path, [snap_row(50.0)], [ev(NOW)])
    assert pacing.weekly_failclosed_candidate(NOW, pacing.load_events(NOW)) is None


def test_failclosed_fresh_reading_below_cap_allows(tmp_path: Path) -> None:
    pacing = make_controller(tmp_path, weekly_all_hard_limit_percent=90.0)
    write_state(tmp_path, [snap_row(60.0, observed_at=NOW)], [ev(NOW - 5)])
    assert pacing.weekly_failclosed_candidate(NOW, pacing.load_events(NOW)) is None


def test_failclosed_stale_reading_projects_forward_and_stops(tmp_path: Path) -> None:
    # Anchor 3h ago at 82%; 70 runs since, at the 0.12 pessimistic floor ->
    # projected 82 + 70*0.12 = 90.4% >= 90 cap -> stop, even though the last
    # reading itself was under the cap (a failing probe scenario).
    pacing = make_controller(tmp_path, weekly_all_hard_limit_percent=90.0, max_sleep_seconds=3600.0)
    anchor_ts = NOW - 3 * 3600.0
    events = [ev(anchor_ts + 1 + i * 100.0) for i in range(70)]
    write_state(tmp_path, [snap_row(82.0, observed_at=anchor_ts)], events)
    result = pacing.weekly_failclosed_candidate(NOW, pacing.load_events(NOW))
    assert result is not None
    delay, reason = result
    assert "fail-closed stop" in reason
    assert delay > 0


def test_failclosed_no_reading_ever_allows_blind_budget_then_pauses(tmp_path: Path) -> None:
    pacing = make_controller(tmp_path, weekly_all_hard_limit_percent=90.0, max_sleep_seconds=1800.0)
    # under blind budget -> allowed
    write_state(tmp_path, [], [ev(NOW - i) for i in range(5)])
    assert pacing.weekly_failclosed_candidate(NOW, pacing.load_events(NOW)) is None
    # over blind budget with still no reading -> pause
    write_state(tmp_path, [], [ev(NOW - i) for i in range(40)])
    result = pacing.weekly_failclosed_candidate(NOW, pacing.load_events(NOW))
    assert result is not None
    assert "no usage reading" in result[1]


def test_failclosed_ignores_previous_week_anchor(tmp_path: Path) -> None:
    # Only reading is from last week (reset in the past) -> treated as no
    # reading this week, not a stale-high anchor.
    pacing = make_controller(tmp_path, weekly_all_hard_limit_percent=90.0)
    old = snap_row(88.0, observed_at=NOW - 8 * 3600.0, reset=NOW - 24 * 3600.0)
    write_state(tmp_path, [old], [ev(NOW - i) for i in range(5)])
    # under blind budget with no current-week reading -> allowed (not stopped on the 88%)
    assert pacing.weekly_failclosed_candidate(NOW, pacing.load_events(NOW)) is None


def test_weekly_fill_spaces_runs_to_reset(tmp_path: Path) -> None:
    pacing = make_controller(tmp_path)
    events = [{"ts": NOW - 10.0, "cell_id": "x", "cost_usd": 0.2}]
    result = pacing.weekly_fill_delay(snapshot(55.0), events, NOW)
    assert result is not None
    delay, reason = result
    # headroom 35pp / 0.07pp per run = 500 runs over ~50h => interval ~360s
    seconds_to_reset = WEEK_RESET - NOW - pacing.reset_padding_seconds
    expected_interval = seconds_to_reset / (35.0 / 0.07)
    assert abs((delay + 10.0) - expected_interval) < 1.0
    assert "weekly fill target" in reason


def test_weekly_fill_sleeps_to_reset_when_target_reached(tmp_path: Path) -> None:
    pacing = make_controller(tmp_path)
    events = [{"ts": NOW - 10.0, "cell_id": "x", "cost_usd": 0.2}]
    result = pacing.weekly_fill_delay(snapshot(91.0), events, NOW)
    assert result is not None
    delay, reason = result
    assert "fill target reached" in reason
    assert delay > 3600.0  # sleeps toward the weekly reset


def test_weekly_fill_disabled_without_target(tmp_path: Path) -> None:
    pacing = make_controller(tmp_path, weekly_fill_target_percent=None)
    events = [{"ts": NOW - 10.0, "cell_id": "x", "cost_usd": 0.2}]
    assert pacing.weekly_fill_delay(snapshot(55.0), events, NOW) is None


def test_weekly_fill_no_delay_when_interval_already_elapsed(tmp_path: Path) -> None:
    pacing = make_controller(tmp_path)
    events = [{"ts": NOW - 100_000.0, "cell_id": "x", "cost_usd": 0.2}]
    assert pacing.weekly_fill_delay(snapshot(55.0), events, NOW) is None


def test_estimator_uses_same_week_snapshots(tmp_path: Path) -> None:
    pacing = make_controller(tmp_path)
    # 30 runs between two snapshots, meter moved 3pp -> 0.1pp/run.
    first_at = NOW - 4 * 3600.0
    state = {
        "usage_snapshots": [
            {
                "observed_at": first_at,
                "weekly_all_percent": 52.0,
                "weekly_all_reset_ts": WEEK_RESET,
            },
            {
                "observed_at": NOW,
                "weekly_all_percent": 55.0,
                "weekly_all_reset_ts": WEEK_RESET,
            },
        ]
    }
    (tmp_path / "pacing_state.json").write_text(json.dumps(state))
    events = [
        {"ts": first_at + i * 400.0, "cell_id": f"c{i}", "cost_usd": 0.2}
        for i in range(30)
    ]
    assert abs(pacing.estimate_weekly_percent_per_run(events) - 0.1) < 1e-9


def test_estimator_measures_small_delta_clean_window(tmp_path: Path) -> None:
    pacing = make_controller(tmp_path)
    # A quiet 25-run window that moved the meter only 1pp is a legitimate
    # measurement: max(1pp, granularity floor) / 25 = 0.04pp/run.
    state = {
        "usage_snapshots": [
            {
                "observed_at": NOW - 3600.0,
                "weekly_all_percent": 55.0,
                "weekly_all_reset_ts": WEEK_RESET,
            },
            {
                "observed_at": NOW,
                "weekly_all_percent": 56.0,
                "weekly_all_reset_ts": WEEK_RESET,
            },
        ]
    }
    (tmp_path / "pacing_state.json").write_text(json.dumps(state))
    events = [
        {"ts": NOW - 3000.0 + i * 100.0, "cell_id": f"c{i}", "cost_usd": 0.2}
        for i in range(25)
    ]
    assert abs(pacing.estimate_weekly_percent_per_run(events) - 0.04) < 1e-9


def test_estimator_recovers_via_minimum_window_after_contamination(tmp_path: Path) -> None:
    pacing = make_controller(tmp_path)
    # Window A (interactive contamination): 25 runs, meter +6pp -> 0.24pp/run.
    # Window B (quiet): 25 runs, meter +1pp -> 0.04pp/run.
    # The estimator must take the MINIMUM window, recovering immediately.
    a_start = NOW - 6 * 3600.0
    a_end = NOW - 3 * 3600.0
    state = {
        "usage_snapshots": [
            {
                "observed_at": a_start,
                "weekly_all_percent": 50.0,
                "weekly_all_reset_ts": WEEK_RESET,
            },
            {
                "observed_at": a_end,
                "weekly_all_percent": 56.0,
                "weekly_all_reset_ts": WEEK_RESET,
            },
            {
                "observed_at": NOW,
                "weekly_all_percent": 57.0,
                "weekly_all_reset_ts": WEEK_RESET,
            },
        ]
    }
    (tmp_path / "pacing_state.json").write_text(json.dumps(state))
    events = [
        {"ts": a_start + 1 + i * 400.0, "cell_id": f"a{i}", "cost_usd": 0.2}
        for i in range(25)
    ] + [
        {"ts": a_end + 1 + i * 400.0, "cell_id": f"b{i}", "cost_usd": 0.2}
        for i in range(25)
    ]
    assert abs(pacing.estimate_weekly_percent_per_run(events) - 0.04) < 1e-9


def test_estimator_ignores_previous_week_snapshots(tmp_path: Path) -> None:
    pacing = make_controller(tmp_path)
    # A stale snapshot from last week (different reset ts) must not be used
    # as the baseline even though its percent is much higher.
    state = {
        "usage_snapshots": [
            {
                "observed_at": NOW - 8 * 3600.0,
                "weekly_all_percent": 88.0,
                "weekly_all_reset_ts": WEEK_RESET - 7 * 24 * 3600.0,
            },
            {
                "observed_at": NOW - 4 * 3600.0,
                "weekly_all_percent": 52.0,
                "weekly_all_reset_ts": WEEK_RESET,
            },
            {
                "observed_at": NOW,
                "weekly_all_percent": 55.0,
                "weekly_all_reset_ts": WEEK_RESET,
            },
        ]
    }
    (tmp_path / "pacing_state.json").write_text(json.dumps(state))
    events = [
        {"ts": NOW - 4 * 3600.0 + i * 400.0, "cell_id": f"c{i}", "cost_usd": 0.2}
        for i in range(30)
    ]
    assert abs(pacing.estimate_weekly_percent_per_run(events) - 0.1) < 1e-9

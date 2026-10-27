from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import aggregate  # noqa: E402
import build_monte_carlo_plan  # noqa: E402
import export_task_registry  # noqa: E402
import failure_report  # noqa: E402
import generate_monte_carlo_realizations  # noqa: E402
import harness  # noqa: E402
from tasks import TASKS  # noqa: E402


def task_by_id(task_id: str):
    return next(task for task in TASKS if task.task_id == task_id)


def test_json_safe_replaces_nonfinite_values() -> None:
    payload = {
        "finite": 1.0,
        "bad": math.nan,
        "nested": [{"inf": math.inf, "ok": -1.25}],
    }

    safe = harness.json_safe(payload)

    assert safe == {
        "finite": 1.0,
        "bad": None,
        "nested": [{"inf": None, "ok": -1.25}],
    }
    json.dumps(safe, allow_nan=False)


def test_provider_limit_detector_matches_claude_limit_failures() -> None:
    limit_record = {
        "claude": {
            "ok": False,
            "error": "claude exited 1",
            "stderr_tail": "Credit balance is too low. Please upgrade.",
        },
        "execution_success": False,
    }
    numeric_failure = {
        "claude": {"ok": True, "total_cost_usd": 0.01, "text": "DONE"},
        "scoring": {"numeric_success": False},
    }

    assert harness.is_provider_limit_record(limit_record) is True
    assert harness.is_provider_limit_record(numeric_failure) is False
    assert harness.is_provider_limit_record({"scoring": {"notes": ["failed"]}}) is False


def test_provider_limit_detector_matches_saved_session_limit_schema() -> None:
    top_level_record = {
        "claude": {
            "ok": False,
            "error": "claude exited 1",
            "stdout_tail": "You've hit your session limit",
        }
    }
    repair_attempt_record = {
        "claude_attempt_2": {
            "ok": False,
            "error": "claude exited 1",
            "stdout_tail": "You've hit your session limit",
        }
    }

    assert harness.is_provider_limit_record(top_level_record) is True
    assert harness.is_provider_limit_record(repair_attempt_record) is True


def test_provider_limit_detector_matches_zero_token_cli_failures() -> None:
    zero_token_record = {
        "claude": {
            "ok": False,
            "error": "claude exited 1",
            "stdout_tail": '"output_tokens":0,"modelUsage":{}',
            "stderr_tail": "",
        }
    }
    repair_limit_record = {
        "claude_attempt_1": {"ok": True, "total_cost_usd": 0.02},
        "claude_attempt_2": {
            "ok": False,
            "error": "claude exited 1",
            "stdout_tail": '"output_tokens":0,"modelUsage":{}',
        },
    }

    assert harness.is_provider_limit_record(zero_token_record) is True
    assert harness.is_provider_limit_record(repair_limit_record) is True


def test_provider_limit_detector_ignores_nonzero_max_turn_failures() -> None:
    max_turn_record = {
        "claude": {
            "ok": False,
            "error": "claude exited 1",
            "stdout_tail": (
                '"modelUsage":{"claude-sonnet-4-6":{"inputTokens":14,'
                '"outputTokens":17758,"costUSD":0.5861805}},'
                '"terminal_reason":"max_turns"'
            ),
        },
        "execution_success": False,
    }

    assert harness.is_provider_limit_record(max_turn_record) is False


def test_stata_guidance_requires_json_safe_numeric_formatting() -> None:
    guidance = harness.platform_lang_hint("stata")
    demo = harness.demonstration_block("stata", "ols_robust")

    assert "leading zero" in guidance
    assert "%21.12f" in guidance
    assert "%21.12f" in demo
    assert "file write fh" in demo


def test_r_few_shot_demo_writes_diagnostics_object() -> None:
    demo = harness.demonstration_block("r", "ols_robust")

    assert "diagnostics = structure(list(), names = character(0))" in demo
    assert "diagnostics = list()" not in demo


def test_r_platform_hint_advertises_iv_packages() -> None:
    guidance = harness.platform_lang_hint("r")

    assert "AER" in guidance
    assert "ivreg" in guidance


def test_score_marks_array_boxed_terms_as_artifact_failure(tmp_path: Path) -> None:
    task = task_by_id("ols_robust")
    result = {
        "task_id": ["ols_robust"],
        "platform": ["r"],
        "estimates": [
            {"term": ["const"], "estimate": [1.0116], "se": [0.022], "n_obs": [2000]},
            {"term": ["x1"], "estimate": [0.4927], "se": [0.0228], "n_obs": [2000]},
            {"term": ["x2"], "estimate": [-0.3219], "se": [0.0213], "n_obs": [2000]},
        ],
        "method_metadata": {"estimator": ["OLS"], "vcov": ["HC1"]},
        "diagnostics": [],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))

    score = harness.score(result_path, task)

    assert score["artifact_success"] is False
    assert score["numeric_success"] is False
    assert any("non-string term" in note for note in score["notes"])


def test_ttest_prompt_declares_reference_direction() -> None:
    task = task_by_id("ttest_welch")

    assert "group 1" in task.prompt
    assert "group 0" in task.prompt
    assert "mean(y | group=1) minus mean(y | group=0)" in task.prompt


def test_dynamic_panel_reference_does_not_score_se(tmp_path: Path) -> None:
    task = task_by_id("dynpanel_ab")
    ref_by_term = {
        row["term"]: row for row in harness.REFERENCE[task.reference_key]["estimates"]
    }
    result = {
        "task_id": task.task_id,
        "platform": "python",
        "estimates": [
            {
                "term": "rho",
                "estimate": ref_by_term["rho"]["estimate"],
                "se": 123.0,
                "n_obs": 800,
            },
            {
                "term": "beta_x",
                "estimate": ref_by_term["beta_x"]["estimate"],
                "se": 456.0,
                "n_obs": 800,
            },
        ],
        "method_metadata": {"estimator": "first-difference IV", "vcov": "not scored"},
        "diagnostics": {},
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))

    score = harness.score(result_path, task)

    assert score["numeric_success"] is True
    assert score["per_term"]["rho"]["ref_se"] is None
    assert score["per_term"]["rho"]["se_pass"] is None


def test_staggered_did_reference_does_not_score_proxy_se(tmp_path: Path) -> None:
    task = task_by_id("staggered_did")
    result = {
        "task_id": task.task_id,
        "platform": "python",
        "estimates": [
            {"term": "att", "estimate": 1.5886894029357215, "se": None, "n_obs": 2000},
        ],
        "method_metadata": {"estimator": "manual cell means", "vcov": "not scored"},
        "diagnostics": {},
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))

    score = harness.score(result_path, task)

    assert score["numeric_success"] is True
    assert score["per_term"]["att"]["ref_se"] is None


def test_aggregate_sums_self_repair_attempt_usage() -> None:
    records = [
        {
            "cell_id": "cell",
            "task_id": "ols_robust",
            "platform": "python",
            "exec_mode": "self_repair",
            "prompt_mode": "zero_shot",
            "execution_success": True,
            "scoring": {
                "artifact_success": True,
                "numeric_success": True,
                "method_success": True,
            },
            "claude_attempt_1": {
                "total_cost_usd": 0.10,
                "wall_seconds": 2.5,
                "num_turns": 1,
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
            "claude_attempt_2": {
                "total_cost_usd": 0.25,
                "wall_seconds": 3.0,
                "num_turns": 1,
                "usage": {
                    "input_tokens": 30,
                    "output_tokens": 40,
                    "cache_read_input_tokens": 50,
                },
            },
        }
    ]

    built = aggregate.build_aggregate(records)
    row = built["rows"][0]

    assert built["n_runs"] == 1
    assert row["cost_usd"] == 0.35
    assert row["wall_seconds"] == 5.5
    assert row["num_turns"] == 2
    assert row["tokens_in"] == 40
    assert row["tokens_out"] == 60
    assert row["tokens_cache_read"] == 50
    assert "self_repair" in built["by_exec_mode"]
    assert "python__zero_shot" in built["by_prompt_platform"]

    md = aggregate.markdown_summary(built, target_runs=11880)
    assert "# Benchmark run summary" in md
    assert "Coverage: **1/11880**" in md
    assert "Pilot" not in md


def test_self_repair_allows_three_revisions(monkeypatch, tmp_path: Path) -> None:
    task = task_by_id("ols_robust")
    monkeypatch.setattr(harness, "RUNS_DIR", tmp_path / "runs")

    def fake_run_claude(*args, **kwargs):
        return {
            "ok": True,
            "text": "```python\n# generated script\n```",
            "total_cost_usd": 0.01,
            "usage": {},
        }

    def fake_execute_script(script_path: Path, platform: str, cwd: Path, timeout: int):
        if "attempt4_" in script_path.name:
            rows = [
                {
                    "term": row["term"],
                    "estimate": row["estimate"],
                    "se": row["se"],
                    "n_obs": 1000,
                }
                for row in harness.REFERENCE[task.reference_key]["estimates"]
            ]
            (cwd / "result.json").write_text(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "platform": platform,
                        "estimates": rows,
                        "method_metadata": {"estimator": "OLS", "vcov": "HC1"},
                        "diagnostics": {},
                    }
                )
            )
            return {"ok": True, "returncode": 0}
        return {"ok": False, "returncode": 1}

    monkeypatch.setattr(harness, "run_claude", fake_run_claude)
    monkeypatch.setattr(harness, "execute_script", fake_execute_script)

    record = harness.run_cell(
        task,
        platform="python",
        prompt_mode="zero_shot",
        exec_mode="self_repair",
        rep=0,
        overwrite=True,
    )

    assert record["execution_success"] is True
    assert "claude_attempt_4" in record
    assert "execution_attempt_4" in record
    assert record["scoring"]["numeric_success"] is True


def test_exported_task_registry_matches_active_tasks() -> None:
    registry = export_task_registry.build_registry()
    exported_ids = [task["task_id"] for task in registry["tasks"]]

    assert exported_ids == [task.task_id for task in TASKS]
    assert registry["source_of_truth"] == "code/benchmark/tasks.py"
    assert len(registry["tasks"]) == 22
    assert len(registry["candidate_tasks"]) >= 150
    assert registry["monte_carlo_design"]["status"] == "declared_not_generated"

    for task in registry["tasks"]:
        assert task["dataset"]["path"].startswith("data/tasks/inputs/")
        assert isinstance(task["dataset"]["seed"], int)
        assert task["monte_carlo_axes"]
        assert set(task["monte_carlo_axes"]) == set(task["monte_carlo_grid"])
        assert task["monte_carlo_status"] == "planned_not_generated"

    sources = {task["source"] for task in registry["candidate_tasks"]}
    assert {"StatLLM", "MetricsAI sample"} <= sources
    assert {
        "candidate_reference_needed"
    } == {task["status"] for task in registry["candidate_tasks"]}
    assert any(
        task.get("source_status") == "rewrite_then_use"
        for task in registry["candidate_tasks"]
    )

    json.dumps(registry, allow_nan=False)


def test_monte_carlo_plan_expands_one_axis_variants() -> None:
    registry = export_task_registry.build_registry()
    plan = build_monte_carlo_plan.build_plan(registry)

    assert plan["n_active_tasks"] == 22
    assert plan["n_variants"] > 500
    assert plan["cells_per_dataset"] == 18
    assert plan["implied_llm_runs_at_30_reps"] == plan["n_variants"] * 18 * 30

    variant = plan["variants"][0]
    assert set(variant) >= {
        "variant_id",
        "task_id",
        "axis",
        "value",
        "seed",
        "parameter_overrides",
        "status",
    }
    assert isinstance(variant["seed"], int)
    assert variant["parameter_overrides"] == {variant["axis"]: variant["value"]}
    assert variant["status"] == "planned_dataset_and_reference_needed"

    md = build_monte_carlo_plan.markdown_summary(plan)
    assert "Monte Carlo Expansion Plan" in md
    assert "not live-run inputs yet" in md


def test_run_cell_writes_into_an_explicit_runs_dir(monkeypatch, tmp_path: Path) -> None:
    """``run_cell`` must honor its explicit ``runs_dir`` parameter so one
    process (collect.py) can serve two arm roots without mutating the
    ``harness.RUNS_DIR`` global."""

    task = task_by_id("ols_robust")
    monkeypatch.setattr(harness, "RUNS_DIR", tmp_path / "wrong")
    monkeypatch.setattr(
        harness,
        "run_claude",
        lambda *a, **k: {"ok": False, "error": "offline test"},
    )
    arm_root = tmp_path / "arm"

    record = harness.run_cell(
        task, "python", "zero_shot", "no_feedback", rep=3, runs_dir=arm_root
    )

    cell_dir = arm_root / "ols_robust__python__zero_shot__no_feedback__rep03"
    assert cell_dir.is_dir()
    assert (cell_dir / "input.csv").exists()
    assert record["cell_id"] == cell_dir.name
    assert not (tmp_path / "wrong").exists()


def test_pacing_controller_spreads_calls_with_run_target(tmp_path: Path) -> None:
    state = tmp_path / "pacing_state.json"
    state.write_text(
        json.dumps(
            {
                "events": [
                    {"ts": 995.0, "cell_id": "a", "cost_usd": 0.10},
                ]
            }
        )
    )
    slept = []
    controller = harness.PacingController(
        state_file=state,
        window_seconds=100.0,
        target_runs=10,
        reserve_fraction=0.0,
        time_func=lambda: 1000.0,
        sleep_func=slept.append,
    )

    controller.before_run("next_cell")

    assert slept == [5.0]


def test_pacing_controller_records_cli_cost_state(tmp_path: Path) -> None:
    state = tmp_path / "pacing_state.json"
    controller = harness.PacingController(
        state_file=state,
        window_seconds=300.0,
        target_runs=10,
        reserve_fraction=0.0,
        time_func=lambda: 1000.0,
        sleep_func=lambda _: None,
    )

    controller.record(
        {
            "cell_id": "cell",
            "claude": {"total_cost_usd": 0.25},
        }
    )

    payload = json.loads(state.read_text())
    assert payload["events"] == [
        {"ts": 1000.0, "cell_id": "cell", "cost_usd": 0.25, "source": "harness"}
    ]


def test_parse_claude_usage_extracts_subscription_meter() -> None:
    now = datetime(
        2026,
        6,
        2,
        12,
        0,
        tzinfo=ZoneInfo("America/Buenos_Aires"),
    ).timestamp()
    text = """
      Current session
      ██████                                             12% used
      Resets 1pm (America/Buenos_Aires)

      Current week (all models)
      ██▌                                                5% used
      Resets Jun 8 at 8pm (America/Buenos_Aires)

      Current week (Sonnet only)
                                                         0% used
    """

    snapshot = harness.parse_claude_usage(text, now=now)

    assert snapshot.current_session_percent == 12.0
    assert snapshot.current_session_resets == "1pm (America/Buenos_Aires)"
    assert snapshot.current_session_reset_ts == now + 3600
    assert snapshot.weekly_all_percent == 5.0
    assert snapshot.weekly_all_resets == "Jun 8 at 8pm (America/Buenos_Aires)"
    assert snapshot.weekly_sonnet_percent == 0.0


def test_parse_claude_usage_handles_collapsed_tui_spacing() -> None:
    now = datetime(
        2026,
        6,
        2,
        12,
        0,
        tzinfo=ZoneInfo("America/Buenos_Aires"),
    ).timestamp()
    text = """
      Curretsession
      ███████▌15%used
      Reses1pm (America/Buenos_Aires)

      Currentweek(allmodels)
      ██▌5%used
      ResetsJun8at8pm(America/Buenos_Aires)

      Currentweek(Sonnetonly)
      0%used
    """

    snapshot = harness.parse_claude_usage(text, now=now)

    assert snapshot.current_session_percent == 15.0
    assert snapshot.current_session_reset_ts == now + 3600
    assert snapshot.weekly_all_percent == 5.0
    assert snapshot.weekly_all_reset_ts is not None
    assert snapshot.weekly_sonnet_percent == 0.0


def test_claude_usage_tui_waits_for_weekly_sections_after_session_meter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe must not exit while `/usage` still renders its weekly meters."""

    chunks = [
        b"Current session\\n32% used\\n",
        (
            b"Current week (all models)\\n40% used\\n"
            b"Current week (Sonnet only)\\n20% used\\n"
        ),
    ]
    reads: list[int] = []
    writes: list[bytes] = []

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float) -> None:
            return None

        def kill(self) -> None:
            return None

    monkeypatch.setattr(harness.pty, "openpty", lambda: (10, 11))
    monkeypatch.setattr(harness.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(harness.os, "close", lambda fd: None)
    monkeypatch.setattr(harness.os, "write", lambda fd, data: writes.append(data))
    monkeypatch.setattr(
        harness.os,
        "read",
        lambda fd, size: (reads.append(fd), chunks.pop(0))[1],
    )
    monkeypatch.setattr(harness.select, "select", lambda *args: ([10], [], []))
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(harness.time, "time", lambda: next(clock, 2.0))
    monkeypatch.setattr(harness, "claude_bin", lambda: "claude")

    captured = harness.run_claude_usage_tui(timeout_seconds=5.0)

    assert writes[0] == b"/usage\r"
    assert len(reads) == 2
    assert "Current week (all models)" in captured
    assert "Current week (Sonnet only)" in captured


def test_pacing_controller_sleeps_to_usage_reset_at_hard_limit(tmp_path: Path) -> None:
    now = datetime(
        2026,
        6,
        2,
        12,
        0,
        tzinfo=ZoneInfo("America/Buenos_Aires"),
    ).timestamp()
    usage_text = """
      Current session
      █████████████████████████████████████████████      92% used
      Resets 1pm (America/Buenos_Aires)
    """
    probe = harness.ClaudeUsageProbe(
        mode="command",
        runner=lambda: usage_text,
        interval_seconds=1,
        time_func=lambda: now,
    )
    controller = harness.PacingController(
        state_file=tmp_path / "pacing_state.json",
        usage_probe=probe,
        session_hard_limit_percent=90.0,
        reset_padding_seconds=0.0,
        max_sleep_seconds=7200.0,
        time_func=lambda: now,
        sleep_func=lambda _: None,
    )

    delay = controller.compute_sleep_seconds(now)

    assert delay == 3600.0
    payload = json.loads((tmp_path / "pacing_state.json").read_text())
    assert payload["usage_snapshots"][0]["current_session_percent"] == 92.0


def test_before_run_rechecks_usage_after_capped_hard_limit_sleep(tmp_path: Path) -> None:
    now = [datetime(2026, 6, 2, 12, 0, tzinfo=ZoneInfo("America/Buenos_Aires")).timestamp()]
    readings = iter((92.0, 92.0, 89.0))
    probe_calls: list[float] = []

    def usage_text() -> str:
        percent = next(readings)
        probe_calls.append(percent)
        return f"""
          Current session
          usage {percent:.0f}% used
          Resets 1pm (America/Buenos_Aires)
        """

    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    probe = harness.ClaudeUsageProbe(
        mode="command",
        runner=usage_text,
        interval_seconds=0,
        time_func=lambda: now[0],
    )
    controller = harness.PacingController(
        state_file=tmp_path / "pacing_state.json",
        usage_probe=probe,
        session_hard_limit_percent=90.0,
        reset_padding_seconds=0.0,
        max_sleep_seconds=10.0,
        time_func=lambda: now[0],
        sleep_func=sleep,
    )

    controller.before_run("next_cell")

    assert sleeps == [10.0, 10.0]
    assert probe_calls == [92.0, 92.0, 89.0]


def test_usage_probe_zero_interval_probes_every_call() -> None:
    calls = []

    def runner() -> str:
        calls.append(None)
        return f"""
          Current session
          ██████                                             {10 + len(calls)}% used
          Resets 1pm (America/Buenos_Aires)
        """

    now = datetime(
        2026,
        6,
        2,
        12,
        0,
        tzinfo=ZoneInfo("America/Buenos_Aires"),
    ).timestamp()
    probe = harness.ClaudeUsageProbe(
        mode="command",
        runner=runner,
        interval_seconds=0,
        time_func=lambda: now,
    )

    first = probe.snapshot()
    second = probe.snapshot()

    assert first is not None
    assert second is not None
    assert first.current_session_percent == 11.0
    assert second.current_session_percent == 12.0
    assert len(calls) == 2


def test_pacing_controller_fills_current_session_to_reset(tmp_path: Path) -> None:
    now = datetime(
        2026,
        6,
        2,
        12,
        0,
        tzinfo=ZoneInfo("America/Buenos_Aires"),
    ).timestamp()
    usage_text = """
      Current session
      ██████████                                         20% used
      Resets 1pm (America/Buenos_Aires)
    """
    state = tmp_path / "pacing_state.json"
    state.write_text(
        json.dumps(
            {
                "events": [
                    {"ts": now - 10.0, "cell_id": "previous", "cost_usd": 0.10},
                ]
            }
        )
    )
    probe = harness.ClaudeUsageProbe(
        mode="command",
        runner=lambda: usage_text,
        interval_seconds=0,
        time_func=lambda: now,
    )
    controller = harness.PacingController(
        state_file=state,
        usage_probe=probe,
        session_hard_limit_percent=80.0,
        fill_current_session=True,
        session_reserve_percent=0.0,
        default_session_percent_per_run=0.5,
        reset_padding_seconds=0.0,
        time_func=lambda: now,
        sleep_func=lambda _: None,
    )

    delay = controller.compute_sleep_seconds(now)

    assert delay == 20.0


def test_pacing_controller_backs_off_when_usage_meter_jumps(tmp_path: Path) -> None:
    base = datetime(
        2026,
        6,
        2,
        12,
        0,
        tzinfo=ZoneInfo("America/Buenos_Aires"),
    ).timestamp()
    now = base + 2400.0
    reset_ts = base + 3600.0
    usage_text = """
      Current session
      ███████████████                                    30% used
      Resets 1pm (America/Buenos_Aires)
    """
    state = tmp_path / "pacing_state.json"
    state.write_text(
        json.dumps(
            {
                "usage_snapshots": [
                    {
                        "observed_at": base,
                        "source": "test",
                        "current_session_percent": 10.0,
                        "current_session_reset_ts": reset_ts,
                    },
                    {
                        "observed_at": base + 1800.0,
                        "source": "test",
                        "current_session_percent": 15.0,
                        "current_session_reset_ts": reset_ts,
                    },
                ],
                "events": [
                    *[
                        {"ts": base + 100.0 + i * 100.0, "cell_id": f"old{i}"}
                        for i in range(10)
                    ],
                    {"ts": now - 1.0, "cell_id": "recent"},
                ],
            }
        )
    )
    probe = harness.ClaudeUsageProbe(
        mode="command",
        runner=lambda: usage_text,
        interval_seconds=0,
        time_func=lambda: now,
    )
    controller = harness.PacingController(
        state_file=state,
        usage_probe=probe,
        session_hard_limit_percent=90.0,
        fill_current_session=True,
        session_reserve_percent=0.0,
        default_session_percent_per_run=0.5,
        external_usage_tolerance_percent_per_hour=6.0,
        reset_padding_seconds=0.0,
        time_func=lambda: now,
        sleep_func=lambda _: None,
    )

    delay = controller.compute_sleep_seconds(now)

    assert delay > 40.0


def test_usage_probe_failure_falls_back_to_proxy_pacing(tmp_path: Path) -> None:
    state = tmp_path / "pacing_state.json"
    state.write_text(
        json.dumps(
            {
                "events": [
                    {"ts": 995.0, "cell_id": "a", "cost_usd": 0.10},
                ]
            }
        )
    )

    def broken_probe() -> str:
        raise RuntimeError("no usage screen")

    probe = harness.ClaudeUsageProbe(
        mode="command",
        runner=broken_probe,
        interval_seconds=1,
        time_func=lambda: 1000.0,
    )
    controller = harness.PacingController(
        state_file=state,
        window_seconds=100.0,
        target_runs=10,
        reserve_fraction=0.0,
        usage_probe=probe,
        session_hard_limit_percent=90.0,
        time_func=lambda: 1000.0,
        sleep_func=lambda _: None,
    )

    assert controller.compute_sleep_seconds(1000.0) == 5.0
    assert "no usage screen" in (probe.last_error or "")


def test_registry_json_can_feed_selected_cells(tmp_path: Path) -> None:
    registry = export_task_registry.build_registry()
    registry_path = tmp_path / "task_registry.json"
    registry_path.write_text(json.dumps(registry))

    loaded_tasks = harness.load_tasks_from_registry(registry_path)
    cells = harness.selected_cells(
        task_ids=["ols_robust"],
        platforms=["python"],
        prompt_modes=["zero_shot"],
        exec_modes=["no_feedback"],
        task_catalog=loaded_tasks,
    )

    assert [task.task_id for task in loaded_tasks] == [task.task_id for task in TASKS]
    assert len(cells) == 1
    assert cells[0][0].task_id == "ols_robust"
    assert cells[0][0].prompt == task_by_id("ols_robust").prompt


def test_resolve_data_source_accepts_project_relative_paths(tmp_path: Path) -> None:
    project_relative = Path("data/generated/test_registry_input.csv")
    resolved = harness.resolve_data_source(str(project_relative))

    assert resolved == harness.PROJECT_ROOT / project_relative
    assert harness.display_path(resolved) == str(project_relative)


def test_validate_schedule_uses_project_relative_data_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_data = tmp_path / "data" / "generated" / "input.csv"
    project_data.parent.mkdir(parents=True)
    project_data.write_text("x\n1\n")
    monkeypatch.setattr(harness, "PROJECT_ROOT", tmp_path)

    task = harness.Task(
        task_id="generated_variant",
        title="Generated variant",
        prompt="Read the generated dataset.",
        estimands=["x"],
        reference_key="ols_robust",
        data_file="data/generated/input.csv",
        data_description="Temporary generated dataset.",
    )

    validation = harness.validate_schedule(
        [(task, "python", "zero_shot", "no_feedback")],
        n_reps=1,
    )

    assert validation["ok"] is True
    assert validation["missing_data"] == []


def test_prompts_use_run_local_input_csv_before_source_filename() -> None:
    task = task_by_id("cox_hazard")
    expected = (
        "Dataset provided as `./input.csv`; "
        "original source file: `pilot_v4_cox.csv`."
    )

    prompts = [
        harness.no_feedback_prompt(task, "python", "zero_shot"),
        harness.tool_loop_prompt(task, "python", "zero_shot"),
        harness.agentic_prompt(task, "python", "zero_shot"),
        harness.repair_prompt(
            task,
            "python",
            "zero_shot",
            previous_code="print('old')",
            failure_context="missing result.json",
        ),
    ]

    for prompt in prompts:
        assert expected in prompt
        assert "Dataset: pilot_v4_cox.csv" not in prompt
        assert prompt.index("./input.csv") < prompt.index("pilot_v4_cox.csv")


def test_failure_report_groups_term_failures() -> None:
    records = [
        {
            "cell_id": "two_way_cluster__r__zero_shot__tool_loop__rep00",
            "task_id": "two_way_cluster",
            "platform": "r",
            "exec_mode": "tool_loop",
            "prompt_mode": "zero_shot",
            "execution_success": True,
            "scoring": {
                "artifact_success": True,
                "method_success": True,
                "numeric_success": False,
                "notes": [],
                "per_term": {"x": {"coef_pass": True, "se_pass": False}},
            },
        },
        {
            "cell_id": "vif__stata__few_shot__self_repair__rep00",
            "task_id": "vif",
            "platform": "stata",
            "exec_mode": "self_repair",
            "prompt_mode": "few_shot",
            "execution_success": False,
            "scoring": {
                "artifact_success": False,
                "method_success": None,
                "numeric_success": False,
                "notes": ["result.json missing"],
            },
        },
    ]

    report = failure_report.build_report(records)
    md = failure_report.markdown_summary(report)

    assert report["n_records"] == 2
    assert report["n_failed_records"] == 2
    assert ("se_failed:x", 1) in report["groups"]["all"]
    assert ("execution_failed", 1) in report["groups"]["platform:stata"]
    assert "Failure Pattern Report" in md
    assert "two_way_cluster__r__zero_shot__tool_loop__rep00" in md


def test_monte_carlo_realization_materializes_supported_variants(tmp_path: Path) -> None:
    registry = export_task_registry.build_registry()
    plan = build_monte_carlo_plan.build_plan(registry)

    payload = generate_monte_carlo_realizations.materialize(
        registry,
        plan,
        tmp_path,
        max_variants=3,
    )

    assert len(payload["registry"]["tasks"]) == 3
    assert len(payload["references"]) == 3
    assert len(payload["manifest"]["generated"]) == 3
    for task in payload["registry"]["tasks"]:
        assert task["monte_carlo_status"] == "dataset_and_reference_generated"
        assert task["reference_key"] in payload["references"]
        assert (harness.PROJECT_ROOT / task["data_file"]).exists()

    md = generate_monte_carlo_realizations.markdown_manifest(payload["manifest"])
    assert "Monte Carlo Realization Manifest" in md


def test_rdd_monte_carlo_reference_uses_in_bandwidth_n_obs() -> None:
    variant = {
        "task_id": "rdd_sharp",
        "axis": "bandwidth",
        "value": 0.5,
        "seed": 20260517,
        "variant_id": "rdd_sharp__mc__bandwidth_0_5__seed_20260517",
    }

    df, reference, _task_update = generate_monte_carlo_realizations.generate_rdd(variant)

    expected_n = int((df["z"].abs() <= 0.5).sum())
    got_n = reference["estimates"][0]["n_obs"]
    assert got_n == expected_n
    assert got_n != len(df)


def test_two_way_cluster_monte_carlo_metadata_names_cgm_not_fixest() -> None:
    variant = {
        "task_id": "two_way_cluster",
        "axis": "n_firms",
        "value": 50,
        "seed": 20260517,
        "variant_id": "two_way_cluster__mc__n_firms_50__seed_20260517",
    }

    _df, reference, _task_update = (
        generate_monte_carlo_realizations.generate_two_way_cluster(variant)
    )

    assert reference["vcov"] == "two-way cluster (CGM) on (firm, year)"
    assert "fixest" not in json.dumps(reference).lower()


def test_wcb_monte_carlo_reference_uses_exact_enumeration() -> None:
    variant = {
        "task_id": "wcb_few",
        "axis": "cluster_count",
        "value": 5,
        "seed": 20260517,
        "variant_id": "wcb_few__mc__cluster_count_5__seed_20260517",
    }

    _df, reference, task_update = generate_monte_carlo_realizations.generate_wcb(variant)

    p_row = reference["estimates"][2]
    assert p_row["term"] == "p_wcb"
    assert p_row["estimate"] == 0.0625
    assert "enumerate all 2^5 = 32" in task_update["prompt"]
    assert "random resampling" in task_update["prompt"]


def test_wcb_monte_carlo_rejects_unsupported_exact_enumeration() -> None:
    variant = {
        "task_id": "wcb_few",
        "axis": "cluster_count",
        "value": 30,
        "seed": 20260517,
        "variant_id": "wcb_few__mc__cluster_count_30__seed_20260517",
    }

    with pytest.raises(ValueError, match="exact Rademacher enumeration"):
        generate_monte_carlo_realizations.generate_wcb(variant)


def test_wcb_monte_carlo_plan_excludes_unsupported_cluster_count() -> None:
    registry = export_task_registry.build_registry()
    task = next(task for task in registry["tasks"] if task["task_id"] == "wcb_few")

    assert task["dataset"]["baseline_parameters"]["cluster_count"] == 5
    assert task["monte_carlo_grid"]["cluster_count"] == [5, 10, 12]

    plan = build_monte_carlo_plan.build_plan(registry)
    cluster_variants = [
        variant
        for variant in plan["variants"]
        if variant["task_id"] == "wcb_few" and variant["axis"] == "cluster_count"
    ]
    planned_values = {variant["value"] for variant in cluster_variants}

    assert planned_values == {10, 12}
    assert 5 not in planned_values
    assert 30 not in planned_values


def test_monte_carlo_realization_covers_all_parent_tasks(tmp_path: Path) -> None:
    registry = export_task_registry.build_registry()
    plan = build_monte_carlo_plan.build_plan(registry)
    first_by_task = {}
    for variant in plan["variants"]:
        first_by_task.setdefault(variant["task_id"], variant)
    subset_plan = {**plan, "variants": list(first_by_task.values())}

    payload = generate_monte_carlo_realizations.materialize(
        registry,
        subset_plan,
        tmp_path,
    )

    assert len(payload["registry"]["tasks"]) == len(registry["tasks"]) == 22
    assert len(payload["references"]) == 22
    assert payload["manifest"]["skipped"] == []
    assert {
        row["parent_task_id"] for row in payload["manifest"]["generated"]
    } == {task["task_id"] for task in registry["tasks"]}


def _arm_root_is_pure(runs_dir: Path) -> bool:
    # The purity rule: an arm root holds only run directories and
    # underscore-prefixed holding entries.
    for entry in runs_dir.iterdir():
        if entry.name.startswith("_"):
            continue
        if not entry.is_dir() or not (entry / "run_record.json").exists():
            return False
    return True


def _write_minimal_record(runs_dir: Path) -> None:
    cell = runs_dir / "ols_robust__python__zero_shot__no_feedback__rep00"
    cell.mkdir(parents=True)
    (cell / "run_record.json").write_text(
        json.dumps(
            {
                "cell_id": cell.name,
                "task_id": "ols_robust",
                "platform": "python",
                "prompt_mode": "zero_shot",
                "exec_mode": "no_feedback",
                "rep": 0,
                "execution_success": True,
                "scoring": {"numeric_success": True},
                "claude": {"total_cost_usd": 0.01, "usage": {}},
            }
        )
    )


def test_aggregate_default_output_stays_out_of_arm_root(
    monkeypatch, tmp_path: Path
) -> None:
    runs_dir = tmp_path / "arm"
    _write_minimal_record(runs_dir)
    monkeypatch.setattr(
        sys, "argv", ["aggregate.py", "--runs-dir", str(runs_dir)]
    )
    aggregate.main()
    assert (runs_dir / "_working" / "aggregate.json").exists()
    assert (runs_dir / "_working" / "aggregate.md").exists()
    assert _arm_root_is_pure(runs_dir)


def test_failure_report_default_output_stays_out_of_arm_root(
    monkeypatch, tmp_path: Path
) -> None:
    runs_dir = tmp_path / "arm"
    _write_minimal_record(runs_dir)
    monkeypatch.setattr(
        sys, "argv", ["failure_report.py", "--runs-dir", str(runs_dir)]
    )
    failure_report.main()
    assert (runs_dir / "_working" / "failure_report.json").exists()
    assert (runs_dir / "_working" / "failure_report.md").exists()
    assert _arm_root_is_pure(runs_dir)

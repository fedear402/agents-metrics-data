"""Offline tests for the cell execution library (``cells.py``).

Covers: claude timeout/strict-mcp provenance plumbing, provider-limit records
re-run instead of skipped on every backend's cell function, the
``rerun_provider_limit`` passthrough on the claude path (2026-08-18 defect
fix), model-derived arm labels, the Codex usage probe and monitor, summary
placement, and demonstration provenance. No live calls.

Formerly ``test_pilot_runner_flags.py``; the CLI-level tests moved to
``test_collect.py`` with the CLI itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cells as rop  # noqa: E402
import harness  # noqa: E402


def test_claude_arm_for_is_model_derived():
    assert rop.claude_arm_for("claude-sonnet-4-6") == "sonnet_pilot"
    assert rop.claude_arm_for("claude-haiku-4-5-20251001") == "haiku_pilot"


def test_extra_cli_args_reach_claude_command(monkeypatch):
    captured = {}

    class _P:
        returncode = 0
        stdout = json.dumps({"result": "OK", "usage": {}, "num_turns": 1})
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _P()

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    monkeypatch.setattr(harness, "CLAUDE_EXTRA_CLI_ARGS", ["--strict-mcp-config"])
    harness.run_claude("p", cwd=Path("."), allow_tools=False, max_turns=1, timeout=5)
    assert "--strict-mcp-config" in captured["cmd"]

    # Default (corpus) behavior: no extra args.
    monkeypatch.setattr(harness, "CLAUDE_EXTRA_CLI_ARGS", [])
    harness.run_claude("p", cwd=Path("."), allow_tools=False, max_turns=1, timeout=5)
    assert "--strict-mcp-config" not in captured["cmd"]


def test_provider_limit_record_is_rerun_not_skipped(monkeypatch, tmp_path):
    """A codex cell whose existing record is a usage-limit rejection must be
    treated as missing so a resumable driver completes the grid."""

    task = rop.TASKS_BY_ID["ols_robust"]
    cell_dir = tmp_path / "ols_robust__python__zero_shot__codex_generate__rep00"
    cell_dir.mkdir(parents=True)
    (cell_dir / "run_record.json").write_text(json.dumps({
        "cell_id": cell_dir.name, "provider_limit": True,
        "exec_mode": "codex_generate", "platform": "python",
    }))

    monkeypatch.setattr(
        harness, "prepare_cell_dir",
        lambda *a, **k: (cell_dir.name, cell_dir.name[:-7], cell_dir, tmp_path / "x.csv"),
    )
    ran = {}

    def fake_generate(record, *a, **k):
        ran["yes"] = True
        record["execution_success"] = False
        record["failure_reason"] = "fake"

    monkeypatch.setattr(rop, "_codex_generate", fake_generate)
    monkeypatch.setattr(rop.pilot_scoring, "canonical_score",
                        lambda *a, **k: {"numeric_success": False, "notes": []})
    monkeypatch.setattr(harness, "write_run_record", lambda rec, **kw: None)

    rec = rop.run_codex_cell(task, "python", "zero_shot", "codex_generate", 0,
                             "gpt-5.4", "medium", overwrite=False)
    assert ran.get("yes") is True, "provider-limit cell was skipped, not re-run"
    assert not rec.get("skipped_existing")

    # A clean existing record IS skipped.
    (cell_dir / "run_record.json").write_text(json.dumps({
        "cell_id": cell_dir.name, "execution_success": True,
    }))
    ran.clear()
    rec2 = rop.run_codex_cell(task, "python", "zero_shot", "codex_generate", 0,
                              "gpt-5.4", "medium", overwrite=False)
    assert rec2.get("skipped_existing") is True
    assert not ran


def test_direct_claude_zero_token_failure_is_marked_provider_limit(monkeypatch, tmp_path):
    """The direct-Claude pilot path must preserve the harness retry contract."""

    task = rop.TASKS_BY_ID["ols_robust"]
    record = {
        "cell_id": "ols_robust__python__few_shot__no_feedback__rep00",
        "claude": {
            "ok": False,
            "error": "claude exited 1",
            "stdout_tail": '"output_tokens":0,"modelUsage":{}',
        },
    }
    monkeypatch.setattr(harness, "run_cell", lambda *a, **k: record)
    monkeypatch.setattr(harness, "write_run_record", lambda rec, **kw: None)
    monkeypatch.setattr(rop, "claude_version", lambda: "test")
    monkeypatch.setattr(
        rop.pilot_scoring,
        "rescore_record",
        lambda *a, **k: record.setdefault("scoring", {"numeric_success": False}),
    )
    monkeypatch.setattr(harness, "RUNS_DIR", tmp_path)

    observed = rop.run_haiku_cell(
        task, "python", "few_shot", "no_feedback", 0, overwrite=False,
        model="claude-sonnet-4-6",
    )

    assert observed["provider_limit"] is True
    assert observed["failure_reason"] == "claude provider-limit or zero-token failure"


def test_run_haiku_cell_passes_rerun_provider_limit_through(monkeypatch, tmp_path):
    """Regression for the 2026-08-18 defect: the claude path never forwarded
    ``rerun_provider_limit`` to ``harness.run_cell``, so a stored rejection
    was skipped forever unless ``--overwrite`` was used. The claude path must
    honor the same contract as the codex and openrouter cell functions."""

    task = rop.TASKS_BY_ID["ols_robust"]
    seen = {}

    def fake_run_cell(*args, **kwargs):
        seen.update(kwargs)
        return {
            "cell_id": "ols_robust__python__few_shot__no_feedback__rep00",
            "skipped_existing": True,
        }

    monkeypatch.setattr(harness, "run_cell", fake_run_cell)

    rop.run_haiku_cell(
        task, "python", "few_shot", "no_feedback", 0, overwrite=False,
        model="claude-sonnet-4-6", runs_dir=tmp_path,
    )

    assert seen.get("rerun_provider_limit") is True
    assert seen.get("runs_dir") == tmp_path


def test_codex_usage_probe_maps_weekly_only_app_server_meter(monkeypatch):
    """Codex's native weekly meter must enter the shared pacing schema."""

    probe = rop.CodexUsageProbe(interval_seconds=60)
    monkeypatch.setattr(probe, "_read_rate_limits", lambda: {
        "rateLimits": {"primary": {"usedPercent": 8, "resetsAt": 1_785_818_058}}
    })

    observed = probe.snapshot()

    assert observed is not None
    assert observed.source == "codex-app-server"
    assert observed.current_session_percent is None
    assert observed.weekly_all_percent == 8.0
    assert observed.weekly_all_reset_ts == 1_785_818_058.0
    assert probe.last_error is None


def test_codex_usage_probe_selects_spark_limit_bucket(monkeypatch):
    """Spark calls must be paced against Spark's separate weekly allowance."""

    probe = rop.CodexUsageProbe(
        interval_seconds=1,
        model="gpt-5.3-codex-spark",
    )
    monkeypatch.setattr(probe, "_read_rate_limits", lambda: {
        "rateLimits": {
            "limitId": "codex",
            "primary": {"usedPercent": 15, "resetsAt": 1_786_162_246},
        },
        "rateLimitsByLimitId": {
            "codex": {
                "limitId": "codex",
                "primary": {"usedPercent": 15, "resetsAt": 1_786_162_246},
            },
            "codex_bengalfox": {
                "limitId": "codex_bengalfox",
                "limitName": "GPT-5.3-Codex-Spark",
                "primary": {"usedPercent": 3, "resetsAt": 1_786_261_010},
            },
        },
    })

    observed = probe.snapshot()

    assert observed is not None
    assert observed.source == "codex-app-server:codex_bengalfox"
    assert observed.weekly_all_percent == 3.0
    assert observed.weekly_all_reset_ts == 1_786_261_010.0
    assert probe.last_error is None


def test_codex_usage_monitor_appends_snapshot(tmp_path, monkeypatch):
    """The active-run ledger must retain frequent Spark meter readings."""

    probe = rop.CodexUsageProbe(interval_seconds=1)
    expected = harness.ClaudeUsageSnapshot(
        observed_at=123.0,
        source="codex-app-server:codex_bengalfox",
        current_session_percent=None,
        current_session_resets=None,
        current_session_reset_ts=None,
        weekly_all_percent=5.0,
        weekly_all_resets=None,
        weekly_all_reset_ts=456.0,
        weekly_sonnet_percent=None,
        weekly_sonnet_resets=None,
        weekly_sonnet_reset_ts=None,
    )
    monkeypatch.setattr(probe, "snapshot", lambda: expected)
    ledger = tmp_path / "usage.jsonl"

    rop.CodexUsageMonitor(probe, ledger).poll_once()

    assert json.loads(ledger.read_text()) == expected.to_json()


def test_pilot_summary_is_written_into_the_working_dir(tmp_path, monkeypatch):
    """`write_summary` resolves the pilot summary under the arm's `_working`
    directory and creates it when absent."""

    runs = tmp_path / "wave"
    runs.mkdir()
    monkeypatch.setattr(harness, "RUNS_DIR", runs)

    path = harness.write_summary([{"cell_id": "a"}], f"{rop.WORKING_DIRNAME}/pilot_summary.json")

    assert path == runs / rop.WORKING_DIRNAME / "pilot_summary.json"
    assert json.loads(path.read_text()) == [{"cell_id": "a"}]


def test_write_summary_accepts_explicit_runs_dir(tmp_path):
    """The explicit ``runs_dir`` parameter must bypass the module global, so
    one process can serve two arm roots without mutating harness state."""

    runs = tmp_path / "arm"
    path = harness.write_summary(
        [{"cell_id": "b"}],
        f"{rop.WORKING_DIRNAME}/collect_summary.json",
        runs_dir=runs,
    )

    assert path == runs / rop.WORKING_DIRNAME / "collect_summary.json"
    assert json.loads(path.read_text()) == [{"cell_id": "b"}]


def test_demo_provenance_fields_for_fewshot():
    """Few-shot openrouter records must carry demonstration provenance
    (respecified wave, 2026-07-25 decision record): the resolved demo key,
    the assigned demo ids, and the hash of the injected demonstration text."""

    import hashlib

    import demonstrations

    prov = rop.demo_provenance("arima_forecast", "python")
    assert prov["demo_key"] == "arima_forecast"
    assert prov["demo_ids"] == list(demonstrations.DEMO_ASSIGNMENTS["arima_forecast"])
    assert prov["n_demos"] == len(prov["demo_ids"]) > 0
    expected = hashlib.sha256(
        demonstrations.demo_for("arima_forecast", "python").encode("utf-8")
    ).hexdigest()
    assert prov["demo_sha256"] == expected
    # Monte Carlo variants resolve to the parent task's demonstration.
    assert rop.demo_provenance(
        "arima_forecast__mc__n__7", "r"
    )["demo_key"] == "arima_forecast"
    # Unregistered ids produce no fields rather than raising.
    assert rop.demo_provenance("not_a_task", "python") == {}

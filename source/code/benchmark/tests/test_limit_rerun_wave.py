"""Contract tests for the 2026-08-24 Sonnet limit re-run wave.

Cover the two harness overrides (default-inert, and disjoint in what they
reach), the two selection rules, and the wave's write-path guard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import harness  # noqa: E402
import limit_rerun_select as sel  # noqa: E402
import limit_rerun_wave as wave  # noqa: E402


class _Proc:
    returncode = 0
    stdout = json.dumps({"result": "OK", "usage": {}, "num_turns": 1})
    stderr = ""


@pytest.fixture()
def captured_cmd(monkeypatch):
    holder: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        holder["cmd"] = cmd
        holder["timeout"] = kwargs.get("timeout")
        return _Proc()

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    return holder


# --------------------------------------------------------------------------- #
# Override 1: the twelve-turn cap
# --------------------------------------------------------------------------- #
def test_corpus_default_passes_the_turn_cap(captured_cmd) -> None:
    harness.run_claude("p", cwd=Path("."), allow_tools=True, max_turns=12, timeout=5)
    cmd = captured_cmd["cmd"]
    assert "--max-turns" in cmd
    assert cmd[cmd.index("--max-turns") + 1] == "12"


def test_none_max_turns_omits_the_flag(captured_cmd) -> None:
    harness.run_claude("p", cwd=Path("."), allow_tools=True, max_turns=None, timeout=5)
    assert "--max-turns" not in captured_cmd["cmd"]


def test_turn_cap_flag_defaults_to_enabled() -> None:
    assert harness.TOOL_LOOP_TURN_CAP_ENABLED is True
    assert harness.MAX_TURNS_TOOL == 12


# --------------------------------------------------------------------------- #
# Override 2: the model-call timeout
# --------------------------------------------------------------------------- #
def test_model_call_timeout_defaults_to_the_collected_bound(monkeypatch) -> None:
    monkeypatch.setattr(harness, "MODEL_CALL_TIMEOUT_OVERRIDE", None)
    monkeypatch.setattr(harness, "TIMEOUT_NO_FEEDBACK", 180)
    assert harness.model_call_timeout() == 180


def test_model_call_timeout_override_applies(monkeypatch) -> None:
    monkeypatch.setattr(harness, "MODEL_CALL_TIMEOUT_OVERRIDE", 3600)
    monkeypatch.setattr(harness, "TIMEOUT_NO_FEEDBACK", 180)
    assert harness.model_call_timeout() == 3600


def test_script_execution_keeps_the_collected_bound(monkeypatch) -> None:
    """The override must not lengthen how long a generated script may run."""
    seen: dict[str, float | None] = {}

    def fake_run(cmd, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return _Proc()

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    monkeypatch.setattr(harness, "MODEL_CALL_TIMEOUT_OVERRIDE", 3600)
    monkeypatch.setattr(harness, "TIMEOUT_NO_FEEDBACK", 180)
    harness.execute_script(
        Path("run.py"), "python", Path("."), timeout=harness.TIMEOUT_NO_FEEDBACK
    )
    assert seen["timeout"] == 180


# --------------------------------------------------------------------------- #
# The two overrides reach disjoint conditions
# --------------------------------------------------------------------------- #
def test_overrides_are_disjoint_across_conditions() -> None:
    """A chatbot call names its own turn budget and its own timeout.

    ``run_cell`` passes ``max_turns=1`` literally for no_feedback and
    self_repair, and ``timeout=TIMEOUT_TOOL`` literally for tool_loop, so
    lifting the tool-loop cap cannot reach a chatbot run and raising the
    model-call timeout cannot reach a constrained-agent run.
    """
    source = (BENCHMARK_ROOT / "harness.py").read_text()
    body = source.split("def run_cell(", 1)[1].split("\ndef selected_cells(", 1)[0]
    tool_loop_block = body.split('elif exec_mode == "tool_loop":', 1)[1]
    tool_loop_block = tool_loop_block.split('elif exec_mode == "agentic":', 1)[0]
    assert "timeout=TIMEOUT_TOOL" in tool_loop_block
    assert "model_call_timeout()" not in tool_loop_block
    before_tool_loop = body.split('elif exec_mode == "tool_loop":', 1)[0]
    assert "MAX_TURNS_TOOL" not in before_tool_loop
    assert before_tool_loop.count("max_turns=1") == 3


def test_apply_overrides_sets_the_wave_configuration(monkeypatch) -> None:
    monkeypatch.setattr(harness, "TOOL_LOOP_TURN_CAP_ENABLED", True)
    monkeypatch.setattr(harness, "MODEL_CALL_TIMEOUT_OVERRIDE", None)
    monkeypatch.setattr(harness, "TIMEOUT_NO_FEEDBACK", 999)
    monkeypatch.setattr(harness, "CLAUDE_EXTRA_CLI_ARGS", ["--stray"])
    config = wave.apply_overrides()
    assert harness.TOOL_LOOP_TURN_CAP_ENABLED is False
    assert harness.MODEL_CALL_TIMEOUT_OVERRIDE == 3600
    assert harness.TIMEOUT_NO_FEEDBACK == 180
    assert harness.CLAUDE_EXTRA_CLI_ARGS == []
    assert config["model_pin"] == "claude-sonnet-4-6"
    assert config["tool_loop_session_wall_seconds"] == 600


def test_run_cell_plumbs_both_overrides_end_to_end(monkeypatch, tmp_path) -> None:
    """Under the wave configuration, check the bounds each call really gets."""
    from tasks import TASKS

    task = next(item for item in TASKS if item.task_id == "ols_robust")
    calls: list[dict] = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": list(cmd), "timeout": kwargs.get("timeout")})

        class _R:
            returncode = 0
            stdout = json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "```python\nprint(1)\n```",
                    "usage": {},
                    "num_turns": 1,
                }
            )
            stderr = ""

        return _R()

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    monkeypatch.setattr(harness, "TOOL_LOOP_TURN_CAP_ENABLED", True)
    monkeypatch.setattr(harness, "MODEL_CALL_TIMEOUT_OVERRIDE", None)
    monkeypatch.setattr(harness, "TIMEOUT_NO_FEEDBACK", 180)
    monkeypatch.setattr(harness, "MODEL", "the-pin")
    wave.apply_overrides()

    harness.run_cell(task, "python", "zero_shot", "no_feedback", rep=0, runs_dir=tmp_path)
    model_call, script_call = calls[0], calls[1]
    assert model_call["timeout"] == 3600  # the lifted model-call bound
    assert script_call["timeout"] == 180  # the collected script bound, unchanged
    assert model_call["cmd"][model_call["cmd"].index("--model") + 1] == "the-pin"

    calls.clear()
    harness.run_cell(task, "python", "zero_shot", "tool_loop", rep=0, runs_dir=tmp_path)
    assert "--max-turns" not in calls[0]["cmd"]
    assert calls[0]["timeout"] == 600  # the session wall stays


# --------------------------------------------------------------------------- #
# Selection rules
# --------------------------------------------------------------------------- #
def _tool_loop_record(**extra):
    record = {"exec_mode": "tool_loop", "claude": {"ok": False}}
    record["claude"].update(extra)
    return record


def test_turn_cap_rule_reads_the_stream_marker(tmp_path) -> None:
    (tmp_path / "agent_stream.jsonl").write_text(
        json.dumps({"type": "result", "subtype": "error_max_turns", "num_turns": 13})
        + "\n"
    )
    marker = sel.turn_cap_marker(tmp_path, _tool_loop_record(error="claude exited 1"))
    assert marker == "stream:error_max_turns"


def test_turn_cap_rule_excludes_a_twelve_turn_submission(tmp_path) -> None:
    """Twelve turns and a final answer is not a run stopped by the cap."""
    (tmp_path / "agent_stream.jsonl").write_text(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "num_turns": 12,
            }
        )
        + "\n"
    )
    record = {"exec_mode": "tool_loop", "claude": {"ok": True, "num_turns": 12}}
    assert sel.turn_cap_marker(tmp_path, record) is None


def test_turn_cap_rule_reads_the_legacy_stdout_marker(tmp_path) -> None:
    record = _tool_loop_record(
        error="claude exited 1",
        stdout_tail='"terminal_reason":"max_turns","errors":["Reached maximum '
        'number of turns (12)"]}',
    )
    assert sel.turn_cap_marker(tmp_path, record) == (
        "stdout_tail:terminal_reason=max_turns"
    )


def test_turn_cap_rule_ignores_a_session_limit_rejection(tmp_path) -> None:
    (tmp_path / "agent_stream.jsonl").write_text(
        json.dumps({"type": "result", "subtype": "success", "is_error": True}) + "\n"
    )
    record = _tool_loop_record(
        error="claude exited 1", stdout_tail='{"api_error_status":429}'
    )
    assert sel.turn_cap_marker(tmp_path, record) is None


def test_turn_cap_rule_ignores_other_conditions(tmp_path) -> None:
    record = {
        "exec_mode": "self_repair",
        "claude_attempt_1": {
            "ok": False,
            "error": "claude exited 1",
            "stdout_tail": '"terminal_reason":"max_turns"',
        },
    }
    assert sel.turn_cap_marker(tmp_path, record) is None


# --------------------------------------------------------------------------- #
# Write-path guard
# --------------------------------------------------------------------------- #
def test_target_root_is_inside_the_arm_working_directory() -> None:
    root = wave.target_root("sonnet_zero_shot")
    assert root.name == "limit_rerun_2026-08-24"
    assert root.parent.name == harness.WORKING_DIRNAME
    assert root.parent.parent.name == "sonnet_zero_shot"


def test_guard_refuses_a_target_outside_the_wave_directory() -> None:
    import analysis_dataset as ad

    original = ad.ARMS["sonnet_zero_shot"].path / "ols_robust__python__zero_shot"
    with pytest.raises(SystemExit):
        wave.assert_writable_target("sonnet_zero_shot", original)


def test_guard_accepts_the_wave_directory() -> None:
    good = wave.target_root("sonnet_zero_shot") / "ols_robust__python__zero_shot__rep00"
    wave.assert_writable_target("sonnet_zero_shot", good)

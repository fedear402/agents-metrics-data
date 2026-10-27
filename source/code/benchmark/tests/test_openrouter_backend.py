"""Offline tests for the OpenRouter backend and its cell-function wiring.

Covers: provider-neutral block schema + native cost extraction, retry and
provider-limit classification, tool-loop turn cap and aggregation, model-derived
arm labels + the marker convention, provider-limit re-run semantics, and secret
scrubbing of the bash-tool environment. No live calls (HTTP layer is
monkeypatched).
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harness  # noqa: E402
import openrouter_backend as orb  # noqa: E402
import cells as rop  # noqa: E402


def _ok_body(content="```python\nprint(1)\n```", cost=0.00012, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "gen-123",
        "model": "openai/gpt-oss-120b",
        "provider": "SomeProvider",
        "choices": [
            {
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 900,
            "completion_tokens": 300,
            "cost": cost,
            "prompt_tokens_details": {"cached_tokens": 100},
            "completion_tokens_details": {"reasoning_tokens": 50},
        },
    }


def test_chat_call_block_schema_and_cost(monkeypatch):
    monkeypatch.setattr(orb, "_post", lambda payload, timeout: (200, _ok_body(), ""))
    block = orb.chat_call([{"role": "user", "content": "hi"}], "openai/gpt-oss-120b")
    assert block["ok"] is True
    assert block["backend"] == "openrouter"
    assert block["cost_usd"] == pytest.approx(0.00012)
    assert block["total_cost_usd"] == pytest.approx(0.00012)  # cell_summary key
    assert block["usage"] == {
        "input_tokens": 900,
        "cached_input_tokens": 100,
        "output_tokens": 300,
        "reasoning_output_tokens": 50,
    }
    assert block["model_used"] == "openai/gpt-oss-120b"
    assert block["provider_name"] == "SomeProvider"
    assert block["generation_id"] == "gen-123"
    assert "```python" in block["text"]


def test_chat_call_normalizes_list_content(monkeypatch):
    body = _ok_body(content=[{"type": "text", "text": "part1 "}, {"type": "text", "text": "part2"}])
    monkeypatch.setattr(orb, "_post", lambda payload, timeout: (200, body, ""))
    block = orb.chat_call([{"role": "user", "content": "hi"}], "m")
    assert block["text"] == "part1 part2"


def test_chat_call_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_post(payload, timeout):
        calls["n"] += 1
        if calls["n"] <= 2:
            return 429, None, "rate limited"
        return 200, _ok_body(), ""

    monkeypatch.setattr(orb, "_post", fake_post)
    monkeypatch.setattr(orb.time, "sleep", lambda s: None)
    block = orb.chat_call([{"role": "user", "content": "hi"}], "m")
    assert block["ok"] is True
    assert block["retries"] == 2
    assert block["provider_limit"] is False


def test_chat_call_402_is_provider_limit_no_retry(monkeypatch):
    calls = {"n": 0}

    def fake_post(payload, timeout):
        calls["n"] += 1
        return 402, None, "insufficient credits"

    monkeypatch.setattr(orb, "_post", fake_post)
    block = orb.chat_call([{"role": "user", "content": "hi"}], "m")
    assert block["ok"] is False
    assert block["provider_limit"] is True
    assert calls["n"] == 1


def test_chat_call_persistent_429_is_provider_limit(monkeypatch):
    monkeypatch.setattr(orb, "_post", lambda payload, timeout: (429, None, "rate limited"))
    monkeypatch.setattr(orb.time, "sleep", lambda s: None)
    block = orb.chat_call([{"role": "user", "content": "hi"}], "m")
    assert block["ok"] is False
    assert block["provider_limit"] is True
    assert block["retries"] == orb.MAX_RETRIES


def test_chat_call_error_in_200_body(monkeypatch):
    body = {"error": {"code": 502, "message": "upstream provider error"}}
    monkeypatch.setattr(orb, "_post", lambda payload, timeout: (200, body, ""))
    monkeypatch.setattr(orb.time, "sleep", lambda s: None)
    block = orb.chat_call([{"role": "user", "content": "hi"}], "m")
    assert block["ok"] is False
    assert "upstream provider error" in block["error_message"]


def _tool_call(command, tc_id="tc1"):
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }


def test_tool_loop_completes_and_aggregates(monkeypatch, tmp_path):
    blocks = [
        # turn 1: model asks to run a command
        {
            "ok": True, "text": "", "wall_seconds": 1.0, "status_code": 200,
            "retries": 0, "error_message": "", "provider_limit": False,
            "cost_usd": 0.001, "total_cost_usd": 0.001,
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "assistant_message": {"role": "assistant", "content": None,
                                  "tool_calls": [_tool_call("touch result.json")]},
            "tool_calls": [_tool_call("touch result.json")],
            "num_tool_calls": 1, "finish_reason": "tool_calls",
            "model_used": "m", "provider_name": "P", "generation_id": "g1",
            "model_requested": "m", "backend": "openrouter",
        },
        # turn 2: model finishes
        {
            "ok": True, "text": "DONE", "wall_seconds": 0.5, "status_code": 200,
            "retries": 0, "error_message": "", "provider_limit": False,
            "cost_usd": 0.002, "total_cost_usd": 0.002,
            "usage": {"input_tokens": 200, "output_tokens": 5},
            "assistant_message": {"role": "assistant", "content": "DONE"},
            "tool_calls": [], "num_tool_calls": 0, "finish_reason": "stop",
            "model_used": "m", "provider_name": "P", "generation_id": "g2",
            "model_requested": "m", "backend": "openrouter",
        },
    ]
    it = iter(blocks)
    monkeypatch.setattr(orb, "chat_call", lambda *a, **k: next(it))
    executed = {}

    def fake_bash(command, cwd, timeout):
        executed["command"] = command
        return "exit_code: 0\nstdout:\nok\nstderr:\n"

    monkeypatch.setattr(orb, "execute_bash", fake_bash)
    out = orb.run_tool_loop("do it", tmp_path, "m")
    assert out["ok"] is True
    assert out["num_turns"] == 2
    assert out["num_tool_calls"] == 1
    assert executed["command"] == "touch result.json"
    assert out["total_cost_usd"] == pytest.approx(0.003)
    assert out["usage"]["input_tokens"] == 300
    assert out["text"] == "DONE"
    assert isinstance(out["transcript"], list)
    # tool result message appended between the two assistant turns
    roles = [m.get("role") for m in out["transcript"]]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_tool_loop_turn_cap(monkeypatch, tmp_path):
    def always_tools(*a, **k):
        return {
            "ok": True, "text": "", "wall_seconds": 0.1, "status_code": 200,
            "retries": 0, "error_message": "", "provider_limit": False,
            "cost_usd": 0.0001, "total_cost_usd": 0.0001,
            "usage": {"output_tokens": 1},
            "assistant_message": {"role": "assistant", "content": None,
                                  "tool_calls": [_tool_call("echo hi")]},
            "tool_calls": [_tool_call("echo hi")],
            "num_tool_calls": 1, "finish_reason": "tool_calls",
            "model_used": "m", "provider_name": "P", "generation_id": "g",
            "model_requested": "m", "backend": "openrouter",
        }

    monkeypatch.setattr(orb, "chat_call", always_tools)
    monkeypatch.setattr(orb, "execute_bash", lambda *a, **k: "exit_code: 0")
    out = orb.run_tool_loop("loop", tmp_path, "m", max_turns=3)
    assert out["num_turns"] == 3
    assert out["num_tool_calls"] == 3
    assert "turn cap" in out["error_message"]


def test_tool_loop_exec_semaphore_used(monkeypatch, tmp_path):
    blocks = [
        {
            "ok": True, "text": "", "wall_seconds": 0.1, "status_code": 200,
            "retries": 0, "error_message": "", "provider_limit": False,
            "cost_usd": None, "total_cost_usd": None, "usage": {},
            "assistant_message": {"role": "assistant", "content": None,
                                  "tool_calls": [_tool_call("echo hi")]},
            "tool_calls": [_tool_call("echo hi")],
            "num_tool_calls": 1, "finish_reason": "tool_calls",
            "model_used": "m", "provider_name": "P", "generation_id": "g",
            "model_requested": "m", "backend": "openrouter",
        },
        {
            "ok": True, "text": "DONE", "wall_seconds": 0.1, "status_code": 200,
            "retries": 0, "error_message": "", "provider_limit": False,
            "cost_usd": None, "total_cost_usd": None, "usage": {},
            "assistant_message": {"role": "assistant", "content": "DONE"},
            "tool_calls": [], "num_tool_calls": 0, "finish_reason": "stop",
            "model_used": "m", "provider_name": "P", "generation_id": "g2",
            "model_requested": "m", "backend": "openrouter",
        },
    ]
    it = iter(blocks)
    monkeypatch.setattr(orb, "chat_call", lambda *a, **k: next(it))

    acquired = {"n": 0}

    class Gate:
        def __enter__(self):
            acquired["n"] += 1
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(orb, "execute_bash", lambda *a, **k: "exit_code: 0")
    orb.run_tool_loop("x", tmp_path, "m", exec_semaphore=Gate())
    assert acquired["n"] == 1


def test_bash_env_scrubs_secrets(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("AGENTS_METRICS_WORKER_KEY", "wk")
    monkeypatch.setenv("SOME_HARMLESS_VAR", "1")
    env = orb._bash_env()
    assert "OPENROUTER_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "AGENTS_METRICS_WORKER_KEY" not in env
    assert env.get("SOME_HARMLESS_VAR") == "1"


def test_openrouter_arm_for_slug():
    assert rop.openrouter_arm_for("openai/gpt-oss-120b") == "openrouter_openai-gpt-oss-120b"
    assert rop.openrouter_arm_for("qwen/qwen3-coder:free") == "openrouter_qwen-qwen3-coder-free"
    assert rop.openrouter_arm_for("z-ai/glm-4.7-flash") == "openrouter_z-ai-glm-4_7-flash"


def test_openrouter_arm_marker_matches_disk_convention():
    """The collect-side marker label for the DeepSeek arms must reproduce the
    stored ``_working/.pilot_arm`` convention (provider arm, not path)."""

    import analysis_dataset as ad
    import collect

    for arm_id in ("deepseek_zero_shot", "deepseek_few_shot"):
        assert collect.arm_marker(ad.ARMS[arm_id]) == (
            "openrouter_deepseek-deepseek-v4-flash"
        )


def test_cell_summary_sums_openrouter_blocks():
    record = {
        "cell_id": "c", "task_id": "t", "platform": "python",
        "prompt_mode": "zero_shot", "exec_mode": "or_repair", "arm": "openrouter_x",
        "execution_success": True, "total_wall_seconds": 3.0,
        "scoring": {"numeric_success": True},
        "openrouter_attempt_1": {
            "ok": True, "wall_seconds": 1.0, "total_cost_usd": 0.001,
            "usage": {"output_tokens": 10},
        },
        "openrouter_attempt_2": {
            "ok": True, "wall_seconds": 2.0, "total_cost_usd": 0.002,
            "usage": {"output_tokens": 20},
        },
    }
    s = rop.cell_summary(record)
    assert s["cost_usd"] == pytest.approx(0.003)
    assert s["output_tokens"] == 30
    assert s["n_attempts"] == 2
    assert s["provider_wall_seconds"] == pytest.approx(3.0)


def test_or_provider_limit_record_is_rerun_not_skipped(monkeypatch, tmp_path):
    task = rop.TASKS_BY_ID["ols_robust"]
    cell_dir = tmp_path / "ols_robust__python__zero_shot__or_generate__rep00"
    cell_dir.mkdir(parents=True)
    (cell_dir / "run_record.json").write_text(json.dumps({
        "cell_id": cell_dir.name, "provider_limit": True,
        "exec_mode": "or_generate", "platform": "python",
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

    monkeypatch.setattr(rop, "_or_generate", fake_generate)
    monkeypatch.setattr(rop.pilot_scoring, "canonical_score",
                        lambda *a, **k: {"numeric_success": False, "notes": []})
    monkeypatch.setattr(harness, "write_run_record", lambda rec, **kw: None)

    rec = rop.run_openrouter_cell(task, "python", "zero_shot", "or_generate", 0,
                                  "openai/gpt-oss-120b", overwrite=False)
    assert ran.get("yes") is True, "provider-limit cell was skipped, not re-run"
    assert not rec.get("skipped_existing")
    assert rec["arm"] == "openrouter_openai-gpt-oss-120b"
    assert rec["backend"] == "openrouter"

    # A clean existing record IS skipped.
    (cell_dir / "run_record.json").write_text(json.dumps({
        "cell_id": cell_dir.name, "execution_success": True,
    }))
    ran.clear()
    rec2 = rop.run_openrouter_cell(task, "python", "zero_shot", "or_generate", 0,
                                   "openai/gpt-oss-120b", overwrite=False)
    assert rec2.get("skipped_existing") is True
    assert not ran


def test_slim_or_block_drops_bulk_fields():
    block = {"ok": True, "assistant_message": {"role": "assistant"},
             "tool_calls": [1], "transcript": [1, 2], "cost_usd": 0.1}
    slim = rop._slim_or_block(block)
    assert "assistant_message" not in slim
    assert "tool_calls" not in slim
    assert "transcript" not in slim
    assert slim["cost_usd"] == 0.1


def test_exec_gate_limits_stata_cap():
    assert rop.EXEC_GATE_LIMITS["stata"] <= 2
    gate = threading.Semaphore(rop.EXEC_GATE_LIMITS["stata"])
    with gate:
        pass  # context-manager protocol as used by the handlers

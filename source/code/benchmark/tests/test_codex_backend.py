"""Offline unit tests for the Codex backend. No live codex calls.

Covers the event-stream parser and ``run_codex`` control flow via a monkeypatched
subprocess for: success, tool-call-present, timeout, nonzero exit, malformed
JSON, and missing usage (Codex review #16).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex_backend  # noqa: E402


def _jsonl(*objs: dict) -> str:
    return "\n".join(json.dumps(o) for o in objs) + "\n"


SUCCESS_EVENTS = _jsonl(
    {"type": "thread.started", "thread_id": "t1"},
    {"type": "turn.started"},
    {"type": "item.completed",
     "item": {"id": "i0", "type": "agent_message",
              "text": "```python\nprint(1)\n```"}},
    {"type": "turn.completed",
     "usage": {"input_tokens": 100, "cached_input_tokens": 50,
               "output_tokens": 20, "reasoning_output_tokens": 5}},
)

TOOLCALL_EVENTS = _jsonl(
    {"type": "thread.started", "thread_id": "t2"},
    {"type": "turn.started"},
    {"type": "item.completed",
     "item": {"id": "i0", "type": "command_execution",
              "command": "python run.py"}},
    {"type": "item.completed",
     "item": {"id": "i1", "type": "agent_message", "text": "DONE"}},
    {"type": "turn.completed",
     "usage": {"input_tokens": 200, "output_tokens": 40}},
)

MISSING_USAGE_EVENTS = _jsonl(
    {"type": "thread.started", "thread_id": "t3"},
    {"type": "item.completed",
     "item": {"id": "i0", "type": "agent_message", "text": "hi"}},
)

MALFORMED_EVENTS = (
    '{"type": "thread.started"}\n'
    "this is not json\n"
    '{"type": "item.completed", "item": {"type": "agent_message", "text": "OK"}}\n'
    "{bad json again\n"
    '{"type": "turn.completed", "usage": {"output_tokens": 3}}\n'
)

UNKNOWN_ITEM_EVENTS = _jsonl(
    {"type": "item.completed", "item": {"id": "i0", "type": "todo_list"}},
    {"type": "item.completed", "item": {"id": "i1", "type": "brand_new_kind"}},
    {"type": "item.completed",
     "item": {"id": "i2", "type": "agent_message", "text": "OK"}},
)

RATE_LIMIT_EVENTS = _jsonl(
    {"type": "thread.started", "thread_id": "t4"},
    {"type": "token_count",
     "info": {"rate_limits": {"primary": {"used_percent": 12.5,
                                          "resets_in_seconds": 3600}}}},
    {"type": "item.completed",
     "item": {"id": "i0", "type": "agent_message", "text": "OK"}},
    {"type": "turn.completed", "usage": {"output_tokens": 7}},
)

USAGE_LIMIT_EVENTS = _jsonl(
    {"type": "thread.started", "thread_id": "t5"},
    {"type": "turn.started"},
    {"type": "error",
     "message": "You've hit your usage limit. Visit .../usage or try again at 1:33 PM."},
    {"type": "turn.failed",
     "error": {"message": "You've hit your usage limit. ... try again at 1:33 PM."}},
)


# --------------------------------------------------------------------------- #
# parse_events
# --------------------------------------------------------------------------- #
def test_parse_events_success():
    p = codex_backend.parse_events(SUCCESS_EVENTS)
    assert p["agent_text"] == "```python\nprint(1)\n```"
    assert p["usage"]["output_tokens"] == 20
    assert p["num_tool_calls"] == 0
    assert p["event_types"]["item.completed"] == 1


def test_parse_events_tool_call_counts():
    p = codex_backend.parse_events(TOOLCALL_EVENTS)
    assert p["num_tool_calls"] == 1
    assert p["item_types"]["command_execution"] == 1
    assert p["agent_text"] == "DONE"


def test_parse_events_missing_usage():
    p = codex_backend.parse_events(MISSING_USAGE_EVENTS)
    assert p["usage"] is None
    assert p["agent_text"] == "hi"


def test_parse_events_malformed_lines_skipped():
    p = codex_backend.parse_events(MALFORMED_EVENTS)
    # Two valid JSON objects out of the non-blank lines are parsed.
    assert p["agent_text"] == "OK"
    assert p["usage"]["output_tokens"] == 3


def test_parse_events_empty():
    p = codex_backend.parse_events("")
    assert p["agent_text"] == ""
    assert p["usage"] is None
    assert p["num_events"] == 0


def test_parse_events_unknown_item_not_counted_as_tool():
    p = codex_backend.parse_events(UNKNOWN_ITEM_EVENTS)
    # todo_list is known-non-tool; brand_new_kind is surfaced, not a tool call.
    assert p["num_tool_calls"] == 0
    assert p["unknown_item_types"] == ["brand_new_kind"]
    assert p["agent_text"] == "OK"


def test_parse_events_rate_limits_captured():
    p = codex_backend.parse_events(RATE_LIMIT_EVENTS)
    assert p["rate_limits"] == {
        "primary": {"used_percent": 12.5, "resets_in_seconds": 3600}
    }


def test_parse_events_captures_usage_limit_error():
    p = codex_backend.parse_events(USAGE_LIMIT_EVENTS)
    assert "usage limit" in p["error_message"].lower()
    assert "1:33 pm" in p["error_message"].lower()


def test_is_usage_limit():
    assert codex_backend.is_usage_limit("You've hit your usage limit; try again at 1:33 PM")
    assert codex_backend.is_usage_limit("purchase more credits")
    assert not codex_backend.is_usage_limit("everything is fine")


def test_sandbox_for():
    assert codex_backend.sandbox_for("codex_generate") == "read-only"
    assert codex_backend.sandbox_for("codex_repair") == "read-only"
    assert codex_backend.sandbox_for("codex_agent") == "workspace-write"
    with pytest.raises(ValueError):
        codex_backend.sandbox_for("nope")


# --------------------------------------------------------------------------- #
# run_codex control flow (subprocess monkeypatched)
# --------------------------------------------------------------------------- #
class _FakeProc:
    def __init__(self, stdout: str, stderr: str, returncode: int):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture(autouse=True)
def _no_version_probe(monkeypatch):
    monkeypatch.setattr(codex_backend, "_VERSION_CACHE", "codex-cli 0.144.0")


def test_run_codex_success(monkeypatch, tmp_path):
    def fake_run(argv, **kwargs):
        return _FakeProc(SUCCESS_EVENTS, "", 0)

    monkeypatch.setattr(codex_backend.subprocess, "run", fake_run)
    ev = tmp_path / "ev.jsonl"
    lm = tmp_path / "last.txt"
    res = codex_backend.run_codex(
        "prompt", tmp_path, "codex_generate", "gpt-5.4", "medium", 60, ev, lm
    )
    assert res["ok"] is True
    assert res["cost_usd"] is None
    assert res["model"] == "gpt-5.4"
    assert res["reasoning_effort"] == "medium"
    assert res["sandbox"] == "read-only"
    assert res["usage"]["output_tokens"] == 20
    assert res["text"].startswith("```python")
    assert ev.read_text() == SUCCESS_EVENTS
    assert res["argv"][-1] == "<PROMPT>"
    assert "--ephemeral" in res["argv"]
    assert "--ignore-rules" in res["argv"]


def test_run_codex_nonzero_exit(monkeypatch, tmp_path):
    def fake_run(argv, **kwargs):
        return _FakeProc("", "boom: rate limited", 1)

    monkeypatch.setattr(codex_backend.subprocess, "run", fake_run)
    res = codex_backend.run_codex(
        "p", tmp_path, "codex_generate", "gpt-5.4", "medium", 60,
        tmp_path / "e.jsonl", tmp_path / "l.txt",
    )
    assert res["ok"] is False
    assert res["exit_code"] == 1
    assert "rate limited" in res["stderr_tail"]


def test_run_codex_timeout_persists_partial(monkeypatch, tmp_path):
    # Simulate result.json already written before the kill.
    (tmp_path / "result.json").write_text("{}")

    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(
            argv, kwargs.get("timeout", 1),
            output=SUCCESS_EVENTS, stderr="partial",
        )

    monkeypatch.setattr(codex_backend.subprocess, "run", fake_run)
    ev = tmp_path / "e.jsonl"
    res = codex_backend.run_codex(
        "p", tmp_path, "codex_agent", "gpt-5.4", "medium", 1,
        ev, tmp_path / "l.txt",
    )
    assert res["ok"] is False
    assert res["timed_out"] is True
    assert res["result_json_existed_at_kill"] is True
    assert res["sandbox"] == "workspace-write"
    # Partial stdout still parsed and persisted.
    assert ev.read_text() == SUCCESS_EVENTS
    assert res["usage"]["output_tokens"] == 20


def test_run_codex_tool_flag(monkeypatch, tmp_path):
    def fake_run(argv, **kwargs):
        return _FakeProc(TOOLCALL_EVENTS, "", 0)

    monkeypatch.setattr(codex_backend.subprocess, "run", fake_run)
    res = codex_backend.run_codex(
        "p", tmp_path, "codex_agent", "gpt-5.4", "medium", 60,
        tmp_path / "e.jsonl", tmp_path / "l.txt",
    )
    assert res["used_tools"] is True
    assert res["num_tool_calls"] == 1


def test_run_codex_flags_usage_limit(monkeypatch, tmp_path):
    def fake_run(argv, **kwargs):
        return _FakeProc(USAGE_LIMIT_EVENTS, "", 1)

    monkeypatch.setattr(codex_backend.subprocess, "run", fake_run)
    res = codex_backend.run_codex(
        "p", tmp_path, "codex_generate", "gpt-5.4", "medium", 60,
        tmp_path / "e.jsonl", tmp_path / "l.txt",
    )
    assert res["provider_limit"] is True
    assert res["ok"] is False
    assert "usage limit" in res["error_message"].lower()

"""Run evidence keeps retained text and distinguishes verified reconstruction."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "falstuff_site"))
import build_run_traces as traces


def test_prompt_reconstruction_requires_matching_recorded_hash(monkeypatch):
    original = "The initial task prompt.\n" + "x" * 9000
    monkeypatch.setattr(traces, "reconstructed_prompt", lambda *args: original)
    record = {
        "task_id": "test",
        "prompt_mode": "zero_shot",
        "prompt_sha256": hashlib.sha256(original.encode()).hexdigest(),
    }
    evidence = traces.prompt_evidence(record, "python", "no_feedback")
    assert evidence["text"] == original
    assert evidence["status"] == "verified-reconstruction"
    assert "system instructions were not retained" in evidence["note"]
    record["prompt_sha256"] = "wrong"
    evidence = traces.prompt_evidence(record, "python", "no_feedback")
    assert evidence["status"] == "unavailable"
    assert evidence["text"] == ""


def test_saved_prompt_and_final_answer_preserve_exact_text(tmp_path):
    answer = "A" * 11000 + "\n/Users/private-person/file.txt"
    record = {"initial_prompt": "Task\n" + "P" * 9000, "claude": {"text": answer}}
    flow = traces.build_flow(tmp_path, "python", "no_feedback", record)
    assert flow["prompt"]["status"] == "saved"
    assert len(flow["prompt"]["text"]) > 9000
    assert flow["final"]["text"] == answer
    assert flow["process"]["events"] == []


def test_three_repair_attempts_keep_order_and_correct_final_execution(tmp_path):
    record = {}
    for number in range(1, 4):
        record[f"codex_attempt_{number}"] = {"text": f"answer {number}"}
        record[f"execution_attempt_{number}"] = {
            "returncode": number,
            "stdout_tail": f"execution {number}",
        }
        (tmp_path / f"attempt{number}_run.py").write_text(f"print({number})")
    flow = traces.build_flow(tmp_path, "python", "self_repair", record)
    assert [x["label"] for x in flow["process"]["events"]] == [
        "Attempt 1: model response",
        "Attempt 1: recorded execution",
        "Attempt 2: model response",
        "Attempt 2: recorded execution",
    ]
    assert flow["final"]["text"] == "answer 3"
    assert flow["final"]["script"] == ""
    assert json.loads(flow["executionText"])["returncode"] == 3
    assert "exact repair prompt was not retained" in flow["process"]["note"]


def test_codex_stream_preserves_commands_results_and_intermediate_messages(tmp_path):
    events = [
        {
            "type": "item.completed",
            "item": {
                "id": "a",
                "type": "agent_message",
                "text": "I will run the script.",
            },
        },
        {
            "type": "item.started",
            "item": {
                "id": "b",
                "type": "command_execution",
                "command": "python run.py",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "b",
                "type": "command_execution",
                "command": "python run.py",
                "aggregated_output": "O" * 12000,
                "exit_code": 0,
            },
        },
        {
            "type": "item.completed",
            "item": {"id": "c", "type": "agent_message", "text": "DONE"},
        },
        {
            "type": "item.started",
            "item": {"id": "d", "type": "command_execution", "command": "unfinished"},
        },
    ]
    (tmp_path / "codex_events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events)
    )
    flow = traces.build_flow(
        tmp_path, "python", "tool_loop", {"codex": {"text": "DONE"}}
    )
    steps = flow["process"]["events"]
    assert [x["kind"] for x in steps] == ["message", "tool", "log", "tool", "log"]
    assert steps[2]["text"] == "O" * 12000 + "\nExit code: 0"
    assert "No completed result" in steps[-1]["text"]
    assert all(x["text"] != "DONE" for x in steps)


def test_claude_stream_keeps_all_tool_arguments_and_results(tmp_path):
    events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t",
                        "name": "Write",
                        "input": {"file_path": "run.py", "content": "x" * 9000},
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "content": [{"type": "text", "text": "saved"}],
                    }
                ]
            },
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "DONE"}]},
        },
    ]
    (tmp_path / "agent_stream.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events)
    )
    steps = traces.stream_evidence(tmp_path, "DONE")
    assert len(steps) == 2
    assert json.loads(steps[0]["text"])["content"] == "x" * 9000
    assert steps[1]["text"] == "saved"


def test_absent_response_uses_script_without_claiming_it_is_original_answer(tmp_path):
    (tmp_path / "run.py").write_text("print(1)")
    (tmp_path / "result.json").write_text("R" * 5000)
    flow = traces.build_flow(tmp_path, "python", "tool_loop", {})
    assert flow["final"]["status"] == "script-only"
    assert flow["final"]["text"] == ""
    assert "not retained" in flow["final"]["note"]
    assert "No intermediate" in flow["process"]["note"]
    assert flow["resultText"] == "R" * 5000


def test_missing_last_attempt_script_does_not_borrow_earlier_script(tmp_path):
    (tmp_path / "attempt1_run.py").write_text("print('earlier')")
    flow = traces.build_flow(
        tmp_path,
        "python",
        "self_repair",
        {"claude_attempt_1": {"text": "earlier"}, "claude_attempt_2": {"text": "last"}},
    )
    assert flow["final"]["text"] == "last"
    assert flow["final"]["script"] == ""


def test_embedded_stata_banner_is_removed_without_removing_model_results():
    text = (
        'Earlier JSON output: {"estimate": 0.42}\n'
        "___/   /   /___/   /   /___/       SE—Standard Edition\n\n"
        " Statistics and Data Science       Copyright 1985 StataCorp LLC\n"
        " 4905 Lakeway Drive\n"
        " College Station, Texas\n"
        " user@stata.com\n\n"
        " Notes:\n     1. Stata is running in batch mode.\n\n"
        ". regress y x\nCoefficient = 0.42\n"
    )
    cleaned = traces.redact(text)
    assert "4905 Lakeway Drive" not in cleaned
    assert "Standard Edition" not in cleaned
    assert 'Earlier JSON output: {"estimate": 0.42}' in cleaned
    assert ". regress y x\nCoefficient = 0.42" in cleaned
    assert (
        traces.redact("* Use SE—Standard Edition\nregress y x")
        == "* Use SE—Standard Edition\nregress y x"
    )


def test_structured_tool_inputs_are_redacted_before_json_encoding(tmp_path):
    event = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": "Write",
                    "input": {
                        "content": "Before\nLicensed to: TEST PRIVATE NAME\nAfter"
                    },
                }
            ]
        },
    }
    (tmp_path / "agent_stream.jsonl").write_text(json.dumps(event))
    text = traces.stream_evidence(tmp_path, "DONE")[0]["text"]
    assert "TEST PRIVATE NAME" not in text
    assert json.loads(text)["content"] == "Before\nAfter"


def test_missing_combination_still_retains_available_evidence(tmp_path):
    (tmp_path / "run_record.json").write_text(
        json.dumps(
            {"initial_prompt": "Task", "codex": {"text": "Provider rejected request."}}
        )
    )
    combo = traces.build_combination(
        {
            "platform": "python",
            "prompt": "zero_shot",
            "interface": "no_feedback",
            "provenance": {"run_dir": str(tmp_path), "itt_failure": True},
        },
        1,
    )
    assert combo["missing"] is True
    assert combo["flow"]["prompt"]["text"] == "Task"
    assert combo["flow"]["final"]["text"] == "Provider rejected request."

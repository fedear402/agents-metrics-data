"""Contract tests for the mini-swe-agent backend (or_agent_swe condition)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import miniswe_backend
import openrouter_backend
import tasks as tasks_mod

TASKS_BY_ID = {t.task_id: t for t in tasks_mod.TASKS}


def test_caps_match_or_agent_contract():
    assert miniswe_backend.SWE_MAX_STEPS == openrouter_backend.DEFAULT_MAX_TURNS == 12
    assert miniswe_backend.SWE_WALL_CAP == openrouter_backend.DEFAULT_AGENT_WALL == 600
    assert miniswe_backend.SWE_BASH_TIMEOUT == openrouter_backend.DEFAULT_BASH_TIMEOUT == 240
    assert miniswe_backend.SWE_OUTPUT_CLIP == openrouter_backend.TOOL_OUTPUT_MAX_CHARS == 8000


def test_config_carries_caps_and_masks_secrets(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake2")
    config = miniswe_backend.build_config(12, 600, 240)
    assert config["agent"]["step_limit"] == 12
    assert config["agent"]["wall_time_limit_seconds"] == 600
    assert config["agent"]["cost_limit"] == 0
    assert config["agent"]["mode"] == "yolo"
    assert config["environment"]["timeout"] == 240
    env = config["environment"]["env"]
    assert env["OPENROUTER_API_KEY"] == ""
    assert env["ANTHROPIC_API_KEY"] == ""


def test_prompt_keeps_task_contract():
    task = TASKS_BY_ID["anova_oneway"]
    prompt = miniswe_backend.swe_tool_loop_prompt(task, "stata", "zero_shot")
    assert task.prompt in prompt
    assert "result.json" in prompt
    assert '"task_id": "anova_oneway"' in prompt
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in prompt
    # The few-shot variant embeds the demonstration block.
    few = miniswe_backend.swe_tool_loop_prompt(task, "stata", "few_shot")
    assert len(few) > len(prompt)


def test_classify_outcome_semantics():
    # Submitted cleanly: ok, no error.
    assert miniswe_backend.classify_outcome("Submitted", False, "", 12, 600) == (
        True, True, "",
    )
    # Step cap: a completed failing response, ok stays True (loader keeps it).
    ok, submitted, err = miniswe_backend.classify_outcome(
        "LimitsExceeded", False, "", 12, 600
    )
    assert ok and not submitted and err == "step cap reached (12)"
    # Wall cap: not ok, marked timed out for the loader's ITT failure path.
    ok, submitted, err = miniswe_backend.classify_outcome("TimeExceeded", True, "", 12, 600)
    assert not ok and not submitted and "600" in err
    # Harness failure: not ok, excluded as a collection failure.
    ok, submitted, err = miniswe_backend.classify_outcome("", False, "boom", 12, 600)
    assert not ok and not submitted and err == "boom"

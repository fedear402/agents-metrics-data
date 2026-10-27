"""Offline contract tests for the task-family-matched few-shot demonstrations.

Guards the 2026-07-25 few-shot respecification (see
``quality_reports/decision_records/2026-07-25_fewshot-respecification.md``):
every production task gets a worked demonstration of a related same-family
task on all three platforms, and no task ever receives its own solution,
dataset filename, or reference values.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import demonstrations  # noqa: E402
import harness  # noqa: E402
from tasks import EXPLORATORY_TASKS, PLATFORMS, TASKS  # noqa: E402


def task_by_id(task_id: str):
    return next(task for task in TASKS if task.task_id == task_id)


def test_every_production_task_has_demos_for_all_platforms() -> None:
    for task in TASKS:
        assert task.task_id in demonstrations.DEMONSTRATIONS
        demos = demonstrations.DEMONSTRATIONS[task.task_id]
        assert set(demos) == set(PLATFORMS)
        for platform in PLATFORMS:
            assert demos[platform].strip()
            assert "Demonstration task:" in demos[platform]


def test_exploratory_tasks_are_also_covered() -> None:
    for task in EXPLORATORY_TASKS:
        assert task.task_id in demonstrations.DEMONSTRATIONS


def test_no_demo_leaks_its_own_task() -> None:
    for task in [*TASKS, *EXPLORATORY_TASKS]:
        for platform in PLATFORMS:
            demo = demonstrations.DEMONSTRATIONS[task.task_id][platform]
            assert task.task_id not in demo, (task.task_id, platform)
            assert task.data_file not in demo, (task.task_id, platform)


def test_no_demo_contains_its_own_reference_values() -> None:
    for task in TASKS:
        reference = harness.REFERENCE[task.reference_key]
        for platform in PLATFORMS:
            demo = demonstrations.DEMONSTRATIONS[task.task_id][platform]
            for row in reference["estimates"]:
                for value in (row.get("estimate"), row.get("se")):
                    if value is None:
                        continue
                    rendered = repr(float(value))
                    if len(rendered) < 8:
                        # Short renderings (2.0, 0.0625, ...) collide with
                        # ordinary inline demo data and carry no leak risk.
                        continue
                    assert rendered not in demo, (task.task_id, platform, rendered)


def test_every_demo_declares_the_result_file_convention() -> None:
    for task_id, demos in demonstrations.DEMONSTRATIONS.items():
        for platform, demo in demos.items():
            assert "result.json" in demo, (task_id, platform)


def test_platform_demos_follow_the_shared_output_idioms() -> None:
    for task_id, demos in demonstrations.DEMONSTRATIONS.items():
        assert "json.dump" in demos["python"], task_id
        assert "write_json" in demos["r"], task_id
        assert (
            "diagnostics = structure(list(), names = character(0))" in demos["r"]
        ), task_id
        assert "diagnostics = list()" not in demos["r"], task_id
        assert "%21.12f" in demos["stata"], task_id
        assert "file write fh" in demos["stata"], task_id


def test_demonstration_block_returns_matched_demo() -> None:
    block = harness.demonstration_block("python", "staggered_did")

    assert "Few-shot demonstration" in block
    assert demonstrations.DEMONSTRATIONS["staggered_did"]["python"] in block
    assert "./input.csv" in block
    assert "./result.json" in block


def test_demonstration_block_rejects_unknown_ids_and_platforms() -> None:
    with pytest.raises(KeyError):
        harness.demonstration_block("python", "no_such_task")
    with pytest.raises(ValueError):
        harness.demonstration_block("julia", "ols_robust")


def test_monte_carlo_variant_ids_fall_back_to_parent_demo() -> None:
    variant_id = "rdd_sharp__mc__bandwidth_0_5__seed_20260517"

    assert demonstrations.resolve_demo_task_id(variant_id) == "rdd_sharp"
    assert harness.demonstration_block("r", variant_id) == (
        harness.demonstration_block("r", "rdd_sharp")
    )
    assert demonstrations.resolve_demo_task_id("no_such_task") is None


def test_few_shot_prompt_includes_matched_demo_on_every_platform() -> None:
    task = task_by_id("staggered_did")
    for platform in PLATFORMS:
        prompt = harness.no_feedback_prompt(task, platform, "few_shot")
        assert demonstrations.DEMONSTRATIONS["staggered_did"][platform] in prompt


def test_few_shot_prompts_differ_across_tasks_and_from_zero_shot() -> None:
    did = task_by_id("staggered_did")
    ols = task_by_id("ols_robust")

    few_did = harness.no_feedback_prompt(did, "python", "few_shot")
    few_ols = harness.no_feedback_prompt(ols, "python", "few_shot")
    zero_did = harness.no_feedback_prompt(did, "python", "zero_shot")

    assert demonstrations.DEMONSTRATIONS["staggered_did"]["python"] in few_did
    assert demonstrations.DEMONSTRATIONS["ols_robust"]["python"] in few_ols
    assert demonstrations.DEMONSTRATIONS["ols_robust"]["python"] not in few_did
    assert "Few-shot demonstration" not in zero_did
    assert few_did != few_ols


def test_prompt_sha256_provenance_tracks_the_demonstration_text() -> None:
    """prompt_sha256 hashes initial_prompt, so new demos stamp a new wave."""
    task = task_by_id("staggered_did")

    def sha(prompt_mode: str, exec_mode: str) -> str:
        text = harness.initial_prompt(task, "python", prompt_mode, exec_mode)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    assert sha("few_shot", "no_feedback") != sha("zero_shot", "no_feedback")
    assert sha("few_shot", "tool_loop") != sha("zero_shot", "tool_loop")

    other = task_by_id("ols_robust")
    other_text = harness.initial_prompt(other, "python", "few_shot", "no_feedback")
    other_sha = hashlib.sha256(other_text.encode("utf-8")).hexdigest()
    assert sha("few_shot", "no_feedback") != other_sha


def test_all_exec_mode_prompts_carry_the_matched_demo() -> None:
    task = task_by_id("cox_hazard")
    demo = demonstrations.DEMONSTRATIONS["cox_hazard"]["r"]

    assert demo in harness.no_feedback_prompt(task, "r", "few_shot")
    assert demo in harness.tool_loop_prompt(task, "r", "few_shot")
    assert demo in harness.agentic_prompt(task, "r", "few_shot")
    assert demo in harness.repair_prompt(
        task,
        "r",
        "few_shot",
        previous_code="print('old')",
        failure_context="missing result.json",
    )

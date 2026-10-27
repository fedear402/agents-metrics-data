"""Offline tests for the runtime section of the manuscript-numbers builder.

No test reads ``data/``. The snapshot fragment and the rows are handcrafted
in the shape ``build_full_analysis`` writes and ``analysis_dataset`` loads.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import analysis_dataset as ad  # noqa: E402
import build_full_analysis as bfa  # noqa: E402
import build_manuscript_numbers as bmn  # noqa: E402


def make_row(
    model: str, interface: str, provider_wall_seconds: float
) -> dict[str, Any]:
    return {
        "model": model,
        "model_id": model,
        "prompt": "zero_shot",
        "interface": interface,
        "interface_label": ad.INTERFACE_LABELS[interface],
        "platform": "python",
        "task": "t01",
        "rep": 0,
        "task_success": True,
        "gates": {},
        "criteria": {},
        "per_term": {},
        "cost_usd": None,
        "tokens": {},
        "wall_seconds": provider_wall_seconds + 5.0,
        "provider_wall_seconds": provider_wall_seconds,
        "turns": None,
        "provenance": {"arm_id": f"{model}_zero_shot"},
    }


def sample_rows() -> list[dict[str, Any]]:
    """Sonnet: chatbot median 10, agent median 30. Codex: 20 and 100."""
    plan = {
        "sonnet": {"no_feedback": (8.0, 12.0), "tool_loop": (20.0, 40.0)},
        "codex": {"no_feedback": (10.0, 30.0), "tool_loop": (80.0, 120.0)},
    }
    rows = []
    for model, by_interface in plan.items():
        for interface, seconds in by_interface.items():
            rows.extend(make_row(model, interface, value) for value in seconds)
    return rows


def snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"extended": {"runtime": bfa._runtime_section(rows)}}


def test_runtime_section_reports_agency_medians_and_ratios() -> None:
    rows = sample_rows()
    snap = snapshot(rows)
    section = bmn.runtime_section(rows, snap)

    sonnet = section["by_model"]["sonnet"]
    assert sonnet["by_interface"]["chatbot"]["median_provider_wall_seconds"] == 10.0
    assert (
        sonnet["by_interface"]["constrained agent"]["median_provider_wall_seconds"]
        == 30.0
    )
    assert sonnet["constrained_agent_to_chatbot_ratio"] == 3.0
    assert sonnet["constrained_agent_to_chatbot_ratio_2dp"] == "3.00"

    codex = section["by_model"]["codex"]
    assert codex["constrained_agent_to_chatbot_ratio"] == 5.0


def test_runtime_section_carries_pooled_medians_and_total_hours() -> None:
    rows = sample_rows()
    snap = snapshot(rows)
    section = bmn.runtime_section(rows, snap)

    assert section["n"] == 8
    assert section["measure"] == "provider_wall_seconds"
    assert section["pooled"]["median_provider_wall_seconds"] == 25.0
    # 320 provider-seconds in total.
    assert section["total_provider_hours"] == pytest.approx(0.09)
    assert set(section["total_provider_hours_by_model"]) == {"sonnet", "codex"}


def test_runtime_section_ratio_is_none_without_a_chatbot_median() -> None:
    rows = [
        make_row("sonnet", "tool_loop", 30.0),
        make_row("codex", "tool_loop", 30.0),
    ]
    section = bmn.runtime_section(rows, snapshot(rows))

    assert section["by_model"]["sonnet"]["constrained_agent_to_chatbot_ratio"] is None


def test_runtime_section_pools_only_the_paper_arms() -> None:
    """A third arm in the corpus snapshot never enters the pooled block."""
    rows = sample_rows()
    corpus = rows + [make_row("deepseek", "no_feedback", 900.0)]
    section = bmn.runtime_section(rows, snapshot(corpus))

    assert section["n"] == len(rows)
    assert set(section["total_provider_hours_by_model"]) == {"sonnet", "codex"}
    assert section["pooled"]["median_provider_wall_seconds"] == 25.0


def test_check_runtime_against_snapshot_accepts_the_same_rows() -> None:
    rows = sample_rows()

    bmn.check_runtime_against_snapshot(rows, snapshot(rows))


def test_check_runtime_against_snapshot_detects_a_per_model_shift() -> None:
    rows = sample_rows()
    snap = snapshot(rows)

    # Swap the sonnet and codex chatbot runtimes. The pooled multiset is
    # unchanged, so only the per-model check can catch this.
    swapped = [dict(row) for row in rows]
    chatbot = [r for r in swapped if r["interface"] == "no_feedback"]
    values = [r["provider_wall_seconds"] for r in chatbot]
    for row, value in zip(chatbot, values[2:] + values[:2], strict=True):
        row["provider_wall_seconds"] = value

    assert bmn._runtime_medians(swapped) == snap["extended"]["runtime"]["pooled"][
        "median_provider_wall_seconds"
    ]
    with pytest.raises(AssertionError, match="runtime median sonnet/chatbot"):
        bmn.check_runtime_against_snapshot(swapped, snap)

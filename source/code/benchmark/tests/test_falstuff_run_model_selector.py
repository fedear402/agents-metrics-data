"""Regression checks for model selection in the public run browser."""

from __future__ import annotations

import csv
import inspect
import sys
from collections import Counter
from pathlib import Path


SITE_DIR = Path(__file__).resolve().parents[2] / "falstuff_site"
sys.path.insert(0, str(SITE_DIR))

import build_run_traces  # noqa: E402


# The two models of the current forty-task paper sample.
EXPECTED_MODELS = {
    "sonnet": {"label": "Claude Sonnet 4.6"},
    "luna": {"label": "GPT-5.6 Luna"},
}
MAIN_RUNS = SITE_DIR.parents[1] / "data/analysis/main/data/runs.csv"


def test_trace_builder_defines_all_public_models() -> None:
    assert build_run_traces.MODEL_CONFIG == EXPECTED_MODELS
    with MAIN_RUNS.open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert {row["model"] for row in rows} == set(EXPECTED_MODELS)
    repetitions = Counter(
        (row["task"], row["model"], row["platform"], row["prompt"], row["agency"])
        for row in rows
    )
    assert set(repetitions.values()) == {build_run_traces.MAX_REPS}


def test_task_payload_and_file_are_scoped_to_one_model(tmp_path: Path) -> None:
    payload = build_run_traces.build_task_payload(
        "sonnet", "Claude Sonnet 4.6", "empty_task", []
    )

    assert payload["modelId"] == "sonnet"
    assert payload["label"] == (
        "Claude Sonnet 4.6 runs, ten repetitions per condition."
    )
    assert len(payload["combos"]) == build_run_traces.EXPECTED_COMBINATIONS

    original_trace_dir = build_run_traces.TRACE_DIR
    try:
        build_run_traces.TRACE_DIR = tmp_path
        build_run_traces.write_task_file("sonnet", payload)
    finally:
        build_run_traces.TRACE_DIR = original_trace_dir

    target = tmp_path / "sonnet" / "empty_task.js"
    source = target.read_text(encoding="utf-8")
    assert 'window.AGENTS_METRICS_RUN_TRACES["sonnet"]["empty_task"]' in source


def test_client_exposes_model_control_and_loads_one_model_file() -> None:
    source = (SITE_DIR / "app.js").read_text(encoding="utf-8")

    assert 'key: "model"' in source
    assert "options: models.map" in source
    assert "function traceScriptSrc(modelId, taskId)" in source
    assert (
        "run-traces/${encodeURIComponent(modelId)}/${encodeURIComponent(taskId)}.js"
        in source
    )
    assert "function ensureTaskRunDetails(modelId, taskId)" in source
    assert "comboChoice.model" in source


def test_builder_api_keeps_model_identity_explicit() -> None:
    assert list(inspect.signature(build_run_traces.build_task_payload).parameters) == [
        "model_id",
        "model_label",
        "task_id",
        "rows",
        "reference",
    ]

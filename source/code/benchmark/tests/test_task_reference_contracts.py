from __future__ import annotations

import json
import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import strict_posthoc  # noqa: E402
import export_task_registry  # noqa: E402
from tasks import ALL_TASKS, EXPLORATORY_TASKS, TASKS  # noqa: E402


def test_exploratory_tasks_do_not_change_primary_registry() -> None:
    primary_ids = [task.task_id for task in TASKS]
    exploratory_ids = [task.task_id for task in EXPLORATORY_TASKS]

    assert len(primary_ids) == 22
    assert exploratory_ids == ["bp_koenker", "synth_weights_fixedv"]
    assert [task.task_id for task in ALL_TASKS] == primary_ids + exploratory_ids
    assert set(exploratory_ids).isdisjoint(primary_ids)


def test_each_task_reference_key_exists_and_covers_estimands() -> None:
    references = strict_posthoc.load_references()

    missing_keys = [
        task.task_id for task in ALL_TASKS if task.reference_key not in references
    ]
    assert missing_keys == []

    missing_terms: dict[str, list[str]] = {}
    for task in ALL_TASKS:
        ref_rows = references[task.reference_key].get("estimates") or []
        ref_terms = {
            row.get("term")
            for row in ref_rows
            if isinstance(row, dict) and isinstance(row.get("term"), str)
        }
        missing = sorted(set(task.estimands) - ref_terms)
        if missing:
            missing_terms[task.task_id] = missing

    assert missing_terms == {}


def test_registry_reference_paths_match_exact_json_keys() -> None:
    references = strict_posthoc.load_references()
    registry = export_task_registry.build_registry()

    expected_paths: dict[str, set[str]] = {key: set() for key in references}
    for path in export_task_registry.REFERENCE_DIR.glob("reference*.json"):
        payload = json.loads(path.read_text())
        for key in payload:
            expected_paths.setdefault(key, set()).add(
                str(path.relative_to(export_task_registry.PROJECT_ROOT))
            )

    path_mismatches: dict[str, dict[str, list[str]]] = {}
    for task in registry["tasks"]:
        key = task["reference"]["key"]
        expected = sorted(expected_paths.get(key, set()))
        observed = sorted(task["reference"]["paths"])
        if observed != expected:
            path_mismatches[task["task_id"]] = {
                "expected": expected,
                "observed": observed,
            }

    assert path_mismatches == {}

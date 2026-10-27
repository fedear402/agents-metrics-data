"""Offline tests for canonical pilot scoring (no live runs).

Guards the R serialization regression: a numerically-correct result whose empty
``diagnostics``/``method_metadata`` serialize as ``[]`` (R jsonlite) must score
``numeric_success=True`` — the exact case ``harness.score()`` false-negatives on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pilot_scoring  # noqa: E402

TASK_ID = "ols_robust"


def _correct_result(platform: str, diagnostics, method_metadata) -> dict:
    task = pilot_scoring.TASKS_BY_ID[TASK_ID]
    ref_rows = pilot_scoring.references()[task.reference_key]["estimates"]
    ref_by_term = {r["term"]: r for r in ref_rows}
    estimates = []
    for term in task.estimands:
        r = ref_by_term[term]
        estimates.append({
            "term": term,
            "estimate": r["estimate"],
            "se": r.get("se"),
            "n_obs": r.get("n_obs"),
        })
    return {
        "task_id": TASK_ID,
        "platform": platform,
        "estimates": estimates,
        "method_metadata": method_metadata,
        "diagnostics": diagnostics,
    }


def _score(tmp_path, result: dict) -> dict:
    (tmp_path / "result.json").write_text(json.dumps(result))
    return pilot_scoring.canonical_score(
        tmp_path, pilot_scoring.TASKS_BY_ID[TASK_ID], result["platform"]
    )


def test_correct_python_result_scores_true(tmp_path):
    sc = _score(tmp_path, _correct_result("python", {}, {"estimator": "OLS"}))
    assert sc["artifact_success"] is True
    assert sc["numeric_success"] is True


def test_R_empty_diagnostics_as_array_still_correct(tmp_path):
    # R jsonlite serializes an empty object as [] — must NOT fail a correct run.
    sc = _score(tmp_path, _correct_result("r", [], []))
    assert sc["numeric_success"] is True


def test_missing_result_scores_false(tmp_path):
    # empty cell dir -> no result.json
    sc = pilot_scoring.canonical_score(
        tmp_path, pilot_scoring.TASKS_BY_ID[TASK_ID], "python"
    )
    assert sc["artifact_success"] is False
    assert sc["numeric_success"] is False


def test_fully_array_wrapped_schema_fails(tmp_path):
    # R jsonlite WITHOUT auto_unbox wraps every scalar in a length-1 array; this
    # is a non-conforming schema and must fail (matches corpus scoring — the
    # pilot is not made more lenient than strict_posthoc).
    result = {
        "task_id": ["ols_robust"],
        "platform": ["r"],
        "estimates": [{"term": ["const"], "estimate": [1.0116], "se": [0.022],
                       "n_obs": [2000]}],
        "method_metadata": {"estimator": ["OLS"]},
        "diagnostics": {},
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    # platform always comes from the record (a clean string), not the result.json.
    sc = pilot_scoring.canonical_score(
        tmp_path, pilot_scoring.TASKS_BY_ID[TASK_ID], "r"
    )
    assert sc["numeric_success"] is False

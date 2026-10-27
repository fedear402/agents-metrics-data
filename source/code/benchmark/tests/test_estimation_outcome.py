"""Offline tests for the unified primary outcome and its artifact repair.

Repair, guard, and scoring tests are hermetic: they build artifact text and run
rows in ``tmp_path`` and inject a synthetic task and reference. Only the pinned
corpus test reads the saved collection, and it skips when that collection is
absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import analysis_dataset as ad  # noqa: E402
import artifact_repair as ar  # noqa: E402
import estimation_outcome as eo  # noqa: E402
from tasks import Task  # noqa: E402

PROJECT_ROOT = BENCHMARK_ROOT.parent.parent
CORPUS_TABLES = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "estimation_outcome.json"
)

# Pinned corpus totals, measured 2026-08-25 on the twelve-rep analysis sample
# under the final-artifact primary outcome (final_artifact_2026-08-13), after
# the staggered-DiD re-collection under the clarified prompt (decision record
# 2026-08-24_task-prompt-data-conventions). The Sonnet and Codex DiD cells
# moved; the DeepSeek rows are unchanged.
# Sample decision record: 2026-08-18_twelve-repetition-sample.
PINNED = {
    "n": 13608,
    "task_success_strict_pct": 81.18,
    "task_success_pct": 83.38,
    "task_success_with_method_pct": 82.0,
    "flips_up": 300,
    "flips_down": 0,
    "flips_up_with_method": 111,
    "flips_down_with_method": 0,
    # model: (strict, final-artifact primary, with_method)
    "by_model": {
        "sonnet": (90.32, 90.78, 90.63),
        "codex": (86.16, 89.99, 87.99),
        "deepseek": (67.06, 69.38, 67.37),
    },
}

TASK = Task(
    task_id="demo_task",
    title="Demo",
    prompt="Estimate the demo model.",
    estimands=["x1"],
    reference_key="demo_task",
    data_file="demo.csv",
    data_description="Demo data.",
)
TASKS_BY_ID = {TASK.task_id: TASK}
REFERENCES: dict[str, Any] = {
    "demo_task": {
        "estimates": [{"term": "x1", "estimate": 2.5, "se": 0.25, "n_obs": 100}]
    }
}
GOOD_RESULT = {
    "task_id": "demo_task",
    "platform": "stata",
    "estimates": [{"term": "x1", "estimate": 2.5, "se": 0.25, "n_obs": 100}],
    "method_metadata": {"estimator": "ols"},
    "diagnostics": {},
}


# --- repair rules -----------------------------------------------------------


def test_rule_code_fence() -> None:
    data, rule = ar.repair_result_text('```json\n{"a": 1}\n```\n')
    assert rule == "code_fence"
    assert data == {"a": 1}


def test_rule_escaped_quotes() -> None:
    data, rule = ar.repair_result_text('{\\"a\\": 1}')
    assert rule == "escaped_quotes"
    assert data == {"a": 1}


def test_rule_doubled_quotes() -> None:
    data, rule = ar.repair_result_text('{""a"": 1}')
    assert rule == "doubled_quotes"
    assert data == {"a": 1}


def test_rule_line_wrapper() -> None:
    raw = '`"{"\'\n"a": 1\n}"\n'
    data, rule = ar.repair_result_text(raw)
    assert rule == "line_wrapper"
    assert data == {"a": 1}


def test_rule_unterminated_string() -> None:
    data, rule = ar.repair_result_text('{"a": "ols,\n"b": 1}')
    assert rule == "unterminated_string"
    assert data == {"a": "ols", "b": 1}


def test_rule_trailing_comma() -> None:
    data, rule = ar.repair_result_text('{"a": [1, 2,],}')
    assert rule == "trailing_comma"
    assert data == {"a": [1, 2]}


def test_rule_stray_quote_after_number() -> None:
    data, rule = ar.repair_result_text('{"a": 1.25", "vcov": "HC1"}')
    assert rule == "stray_quote_after_number"
    assert data == {"a": 1.25, "vcov": "HC1"}


def test_rule_empty_value() -> None:
    data, rule = ar.repair_result_text('{"n_obs": , "a": 1}')
    assert rule == "empty_value"
    assert data == {"n_obs": None, "a": 1}


def test_rule_stata_number() -> None:
    data, rule = ar.repair_result_text('{"se": .0219, "estimate": ., "n": 5.}')
    assert rule == "stata_number"
    assert data == {"se": 0.0219, "estimate": None, "n": 5.0}


def test_valid_json_needs_no_rule_and_normalizes_empty_containers() -> None:
    data, rule = ar.repair_result_text(
        '{"a": 1, "diagnostics": [], "method_metadata": []}'
    )
    assert rule == "none needed"
    assert data == {"a": 1, "diagnostics": {}, "method_metadata": {}}


def test_minimal_rule_set_is_recorded() -> None:
    data, rule = ar.repair_result_text('{""se"": .5}')
    assert rule == "doubled_quotes+stata_number"
    assert data == {"se": 0.5}


# --- digit-preservation guard ----------------------------------------------


def test_digits_preserved_accepts_padding_zero_only() -> None:
    assert ar.digits_preserved('{"se": .5}', '{"se": 0.5}')
    assert ar.digits_preserved('{"a": 12}', '{"a": 12}')
    assert not ar.digits_preserved('{"a": 12}', '{"a": 13}')
    assert not ar.digits_preserved('{"a": 12}', '{"a": 1}')
    assert not ar.digits_preserved('{"a": 12}', '{"a": 123}')


def test_driver_rejects_a_digit_altering_repair(monkeypatch: Any) -> None:
    """A rule that changes a digit is refused even when it makes the text parse."""

    def digit_changing(text: str) -> str:
        return text.replace("7", "1").replace("'", '"')

    monkeypatch.setattr(ar, "RULES", (("bad_rule", digit_changing),))
    monkeypatch.setattr(ar, "RULE_NAMES", ("bad_rule",))
    data, rule = ar.repair_result_text("{'a': 7}")
    assert data is None
    assert rule == "digit_guard_violation"


# --- documents that stay unrepairable --------------------------------------


def test_truncated_document_is_never_closed() -> None:
    raw = '{"task_id": "demo_task", "estimates": [{"term": "x1", "estimate": 2.5'
    data, rule = ar.repair_result_text(raw)
    assert data is None
    assert rule == "unrepairable: truncated"


def test_empty_document_stays_unrepairable() -> None:
    for raw in ("", "   \n\t "):
        data, rule = ar.repair_result_text(raw)
        assert data is None
        assert rule == "unrepairable: empty file"


def test_unrepairable_other_keeps_its_own_label() -> None:
    data, rule = ar.repair_result_text('{"a" 1, "b" 2}')
    assert data is None
    assert rule == "unrepairable: other"


# --- record scoring ---------------------------------------------------------


def make_row(
    tmp_path: Path,
    result_text: str | None,
    *,
    gates: dict[str, bool] | None = None,
    task_success: bool = False,
    task: str = "demo_task",
) -> dict[str, Any]:
    """An unannotated loader-shaped row; ``task_success`` is the strict verdict."""
    run_dir = tmp_path / "demo_task__stata__no_feedback__rep0"
    run_dir.mkdir(parents=True, exist_ok=True)
    if result_text is not None:
        (run_dir / "result.json").write_text(result_text)
    all_gates = {
        "execution": True,
        "artifact": True,
        "numeric": True,
        "method": True,
        "integrity": True,
    }
    all_gates.update(gates or {})
    provenance: dict[str, Any] = {
        "arm_id": "codex_zero_shot",
        "run_dir": str(run_dir),
    }
    return {
        "model": "codex",
        "prompt": "zero_shot",
        "interface": "no_feedback",
        "interface_label": ad.INTERFACE_LABELS["no_feedback"],
        "platform": "stata",
        "task": task,
        "rep": 0,
        "task_success": task_success,
        "gates": all_gates,
        "provenance": provenance,
    }


def score_one(row: dict[str, Any]) -> dict[str, Any]:
    return eo.score_rows([row], references=REFERENCES, tasks_by_id=TASKS_BY_ID)[0]


def test_known_good_record_scores_one_under_all_outcomes(tmp_path: Path) -> None:
    record = score_one(make_row(tmp_path, json.dumps(GOOD_RESULT), task_success=True))
    assert record["repair_rule"] == "none needed"
    assert record["numeric_repaired_pass"] == 1
    assert record["task_success"] == 1
    assert record["task_success_strict"] == 1
    assert record["task_success_with_method"] == 1
    assert record["numeric_fail_kind"] == ""


def test_method_gate_only_record_splits_primary_and_sensitivity(
    tmp_path: Path,
) -> None:
    """Right numbers, no recognizable method evidence: primary 1, with-method 0."""
    row = make_row(
        tmp_path,
        json.dumps(GOOD_RESULT),
        gates={"method": False},
        task_success=False,
    )
    record = score_one(row)
    assert record["task_success"] == 1
    assert record["task_success_strict"] == 0
    assert record["task_success_with_method"] == 0
    assert record["method_pass"] == 0


def test_repair_rescues_a_stata_quoted_artifact(tmp_path: Path) -> None:
    raw = json.dumps(GOOD_RESULT).replace('"', '""')
    record = score_one(make_row(tmp_path, raw, gates={"artifact": False}))
    assert record["repair_rule"] == "doubled_quotes"
    assert record["task_success"] == 1
    assert record["task_success_strict"] == 0


def test_integrity_failure_blocks_the_rescue(tmp_path: Path) -> None:
    row = make_row(tmp_path, json.dumps(GOOD_RESULT), gates={"integrity": False})
    record = score_one(row)
    assert record["numeric_repaired_pass"] == 1
    assert record["task_success"] == 0
    assert record["task_success_with_method"] == 0


def test_multi_revision_row_scores_its_final_artifact(tmp_path: Path) -> None:
    """Final-artifact rule: the stored file is judged whatever produced it."""
    row = make_row(
        tmp_path,
        json.dumps(GOOD_RESULT),
        gates={"artifact": False, "numeric": False},
        task_success=False,
    )
    row["revisions_used"] = 3
    record = score_one(row)
    assert record["task_success"] == 1
    assert record["numeric_repaired_pass"] == 1
    assert record["revisions_used"] == 3


def test_missing_file_and_truncated_file_fail_with_no_artifact(tmp_path: Path) -> None:
    missing = score_one(make_row(tmp_path / "a", None))
    assert missing["repair_rule"] == "no_file"
    assert missing["numeric_fail_kind"] == "no_artifact"
    truncated = score_one(make_row(tmp_path / "b", json.dumps(GOOD_RESULT)[:40]))
    assert truncated["repair_rule"] == "unrepairable: truncated"
    assert truncated["numeric_fail_kind"] == "no_artifact"
    assert truncated["task_success"] == 0


def test_numeric_fail_kinds_are_labeled(tmp_path: Path) -> None:
    cases = {
        "estimate_off": {"term": "x1", "estimate": 9.9, "se": 0.25, "n_obs": 100},
        "se_off": {"term": "x1", "estimate": 2.5, "se": 9.9, "n_obs": 100},
        "n_obs_only": {"term": "x1", "estimate": 2.5, "se": 0.25, "n_obs": 42},
        "other": {"term": "x1", "estimate": None, "se": 0.25, "n_obs": 100},
    }
    for index, (expected, estimate_row) in enumerate(cases.items()):
        payload = dict(GOOD_RESULT, estimates=[estimate_row])
        row = make_row(tmp_path / f"case{index}", json.dumps(payload))
        assert score_one(row)["numeric_fail_kind"] == expected
    payload = dict(GOOD_RESULT, estimates=[])
    row = make_row(tmp_path / "missing_term", json.dumps(payload))
    assert score_one(row)["numeric_fail_kind"] == "missing_term"


def test_references_are_not_read_when_no_artifact_exists(tmp_path: Path) -> None:
    """A row with no artifact never triggers a reference load."""
    loader = eo.ReferenceLoader(reference_dir=tmp_path / "does_not_exist")
    record = eo.score_row(
        make_row(tmp_path, None), tasks_by_id=TASKS_BY_ID, references=loader
    )
    assert record["repair_rule"] == "no_file"


def test_unknown_task_is_labeled_not_crashed(tmp_path: Path) -> None:
    row = make_row(tmp_path, json.dumps(GOOD_RESULT), task="not_a_task")
    record = score_one(row)
    assert record["numeric_fail_kind"] == "unknown_task"
    assert record["task_success"] == 0


def test_annotated_rows_are_formatted_without_rescoring(tmp_path: Path) -> None:
    """Rows the loader already annotated are formatted from stored fields."""
    row = make_row(tmp_path, json.dumps(GOOD_RESULT), task_success=False)
    ad.attach_unified_outcome([row], tasks_by_id=TASKS_BY_ID, references=REFERENCES)
    assert row["task_success"] is True
    assert row["task_success_strict"] is False
    (Path(row["provenance"]["run_dir"]) / "result.json").unlink()
    record = score_one(row)  # no re-read: the deleted artifact does not matter
    assert record["task_success"] == 1
    assert record["task_success_strict"] == 0
    assert record["repair_rule"] == "none needed"


# --- tables -----------------------------------------------------------------


def test_summarize_shapes_and_rates(tmp_path: Path) -> None:
    rows = []
    for index in range(4):
        rows.append(
            make_row(
                tmp_path / f"row{index}",
                json.dumps(GOOD_RESULT) if index < 3 else None,
                gates={"method": index != 2, "artifact": index < 3},
                task_success=index < 2,
            )
        )
    records = eo.score_rows(rows, references=REFERENCES, tasks_by_id=TASKS_BY_ID)
    tables = eo.summarize(records)

    overall = tables["overall"]
    assert overall == {
        "n": 4,
        "task_success_n": 3,
        "task_success_pct": 75.0,
        "task_success_strict_n": 2,
        "task_success_strict_pct": 50.0,
        "task_success_with_method_n": 2,
        "task_success_with_method_pct": 50.0,
        "delta_pp": 25.0,
        "delta_with_method_pp": 0.0,
    }
    assert tables["flips_vs_strict"]["task_success"]["n_up"] == 1
    assert tables["flips_vs_strict"]["task_success"]["n_down"] == 0
    assert tables["flips_vs_strict"]["task_success_with_method"]["n_up"] == 0
    assert set(tables["by_model"]) == {"codex"}
    assert set(tables["by_arm"]["codex"]) == set(ad.PROMPTS)
    assert set(tables["nine_cell"]["codex"]) == {"pooled", *ad.PROMPTS}
    assert len(tables["nine_cell"]["codex"]["pooled"]) == 9
    assert set(tables["by_model"]["codex"]["by_software"]) == set(ad.PLATFORMS)
    # Row 2 matched numerically and failed only the method-evidence gate.
    assert tables["flip_mechanism"]["method_gate_only"] == 1
    assert tables["flip_mechanism"]["already_valid_json"] == 1
    assert "boundary_vetoed" not in tables["flip_mechanism"]
    assert json.dumps(tables) == json.dumps(eo.summarize(records))


def test_write_records_csv_round_trip(tmp_path: Path) -> None:
    records = eo.score_rows(
        [make_row(tmp_path, json.dumps(GOOD_RESULT))],
        references=REFERENCES,
        tasks_by_id=TASKS_BY_ID,
    )
    out = tmp_path / "records.csv"
    eo.write_records_csv(records, out)
    lines = out.read_text().splitlines()
    assert lines[0] == ",".join(eo.RECORD_COLUMNS)
    assert len(lines) == 2


# --- pinned corpus totals ---------------------------------------------------


@pytest.mark.skipif(
    not CORPUS_TABLES.is_file(), reason="corpus estimation_outcome.json not built"
)
def test_corpus_totals_are_pinned() -> None:
    """A change that moves the corpus totals must move this test too."""
    tables = json.loads(CORPUS_TABLES.read_text())
    overall = tables["overall"]
    assert overall["n"] == PINNED["n"]
    assert overall["task_success_strict_pct"] == PINNED["task_success_strict_pct"]
    assert overall["task_success_pct"] == PINNED["task_success_pct"]
    assert (
        overall["task_success_with_method_pct"]
        == PINNED["task_success_with_method_pct"]
    )

    flips = tables["flips_vs_strict"]["task_success"]
    assert flips["n_up"] == PINNED["flips_up"]
    assert flips["n_down"] == PINNED["flips_down"]
    with_method = tables["flips_vs_strict"]["task_success_with_method"]
    assert with_method["n_up"] == PINNED["flips_up_with_method"]
    assert with_method["n_down"] == PINNED["flips_down_with_method"]
    assert "boundary_vetoed" not in tables["flip_mechanism"]

    for model, (strict, primary, with_gate) in PINNED["by_model"].items():
        block = tables["by_model"][model]["overall"]
        assert block["n"] == 4536
        assert block["task_success_strict_pct"] == strict
        assert block["task_success_pct"] == primary
        assert block["task_success_with_method_pct"] == with_gate

    assert "post_boundary" not in tables["repair_rule_counts"]
    assert "digit_guard_violation" not in tables["repair_rule_counts"]
    assert tables["definition"]["primary_outcome_version"] == (
        "final_artifact_2026-08-13"
    )

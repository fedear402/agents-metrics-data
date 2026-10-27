"""Offline tests for the six-arm run loader (synthetic fixtures only).

No test reads ``data/``; every loader call runs against synthetic run
directories under ``tmp_path`` with injected references, tasks, and canonical
data. Fixture helpers are local to this file so the module stays
self-contained while sibling test files are developed concurrently.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis_dataset as ad  # noqa: E402
from tasks import Task  # noqa: E402

CANONICAL_BYTES = b"id,x1,y\n1,0.5,1.0\n2,-0.3,0.2\n"
SCRIPT_TEXT = "# synthetic OLS with HC1 robust standard errors\nprint('ok')\n"
SYNTH_TASK = Task(
    task_id="ols_robust",
    title="synthetic OLS",
    prompt="synthetic",
    estimands=["const", "x1"],
    reference_key="ols_robust",
    data_file="pilot.csv",
    data_description="synthetic",
)
REFERENCES = {
    "ols_robust": {
        "estimates": [
            {"term": "const", "estimate": 1.25, "se": 0.1, "n_obs": 100},
            {"term": "x1", "estimate": -0.75, "se": 0.05, "n_obs": 100},
        ]
    }
}
TASKS_BY_ID = {"ols_robust": SYNTH_TASK}


def make_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "benchmark_data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "pilot.csv").write_bytes(CANONICAL_BYTES)
    return data_dir


def correct_result() -> dict:
    return {
        "task_id": "ols_robust",
        "platform": "python",
        "estimates": [
            {"term": "const", "estimate": 1.25, "se": 0.1, "n_obs": 100},
            {"term": "x1", "estimate": -0.75, "se": 0.05, "n_obs": 100},
        ],
        "method_metadata": {"estimator": "OLS"},
        "diagnostics": {},
    }


def passing_inline_scoring() -> dict:
    return {
        "scorer": "canonical(strict_posthoc)",
        "execution_pass": True,
        "artifact_success": True,
        "numeric_success": True,
        "method_success": True,
        "model_response": True,
        "per_term": {"const": {"coef_pass": True}},
    }


def write_wave_run(
    root: Path,
    *,
    interface: str = "no_feedback",
    rep: int = 0,
    ok: bool = True,
    error: str | None = None,
    stdout_tail: str = "",
    result: dict | None = None,
    input_bytes: bytes = CANONICAL_BYTES,
    blocks: dict | None = None,
    inline_scoring: dict | None = None,
    cost: float = 0.125,
    turns: int = 2,
) -> Path:
    name = f"ols_robust__python__few_shot__{interface}__rep{rep:02d}"
    run_dir = root / name
    run_dir.mkdir(parents=True)
    if blocks is None:
        if ok:
            block = {
                "ok": True,
                "total_cost_usd": cost,
                "wall_seconds": 3.0,
                "num_turns": turns,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 30,
                    "cache_creation_input_tokens": 40,
                },
            }
        else:
            block = {
                "ok": False,
                "error": error,
                "stdout_tail": stdout_tail,
                "wall_seconds": 3.0,
            }
        blocks = {"claude": block}
    record = {
        "cell_id": name,
        "task_id": "ols_robust",
        "platform": "python",
        "prompt_mode": "few_shot",
        "exec_mode": interface,
        "rep": rep,
        "model": "claude-sonnet-4-6",
        "prompt_sha256": "f" * 64,
        "created_at": f"2026-07-27T10:00:{rep:02d}+00:00",
        "execution_success": bool(ok),
        "total_wall_seconds": 7.0,
        **blocks,
    }
    if inline_scoring is not None:
        record["scoring"] = inline_scoring
    (run_dir / "run_record.json").write_text(json.dumps(record))
    (run_dir / "input.csv").write_bytes(input_bytes)
    (run_dir / "run.py").write_text(SCRIPT_TEXT)
    if result is not None:
        (run_dir / "result.json").write_text(json.dumps(result))
    return run_dir


def load_wave(root: Path, data_dir: Path, max_reps: int | None = 5):
    spec = replace(ad.ARMS["sonnet_few_shot"], path=root)
    return ad.load_sonnet_wave_arm(
        spec,
        max_reps,
        references=REFERENCES,
        tasks_by_id=TASKS_BY_ID,
        data_dir=data_dir,
    )


def write_codex_run(
    root: Path,
    *,
    exec_mode: str = "codex_generate",
    prompt_mode: str = "zero_shot",
    rep: int = 0,
    scoring: dict | None = None,
    ok: bool = True,
    timed_out: bool = False,
    usage: dict | None = None,
    attempts: list[dict] | None = None,
    input_bytes: bytes = CANONICAL_BYTES,
    omit_record: bool = False,
) -> Path:
    name = f"ols_robust__python__{prompt_mode}__{exec_mode}__rep{rep:02d}"
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "input.csv").write_bytes(input_bytes)
    (run_dir / "run.py").write_text(SCRIPT_TEXT)
    if omit_record:
        return run_dir
    record: dict = {
        "task_id": "ols_robust",
        "platform": "python",
        "prompt_mode": prompt_mode,
        "exec_mode": exec_mode,
        "rep": rep,
        "model": "gpt-5.4",
        "prompt_sha256": "e" * 64,
        "created_at": f"2026-07-28T09:00:{rep:02d}+00:00",
        "execution_success": True,
        "total_wall_seconds": 14.0,
        "scoring": scoring if scoring is not None else passing_inline_scoring(),
    }
    if attempts is not None:
        for idx, attempt_usage in enumerate(attempts, start=1):
            record[f"codex_attempt_{idx}"] = {
                "ok": True,
                "cost_usd": None,
                "usage": attempt_usage,
                "wall_seconds": 5.0,
                "timed_out": False,
                "provider_limit": False,
            }
    else:
        record["codex"] = {
            "ok": ok,
            "cost_usd": None,
            "usage": usage
            or {
                "input_tokens": 100,
                "cached_input_tokens": 40,
                "output_tokens": 30,
                "reasoning_output_tokens": 10,
            },
            "timed_out": timed_out,
            "provider_limit": False,
            "wall_seconds": 6.0,
        }
    (run_dir / "run_record.json").write_text(json.dumps(record))
    return run_dir


def load_codex(
    root: Path, data_dir: Path, arm: str = "codex_few_shot", max_reps: int | None = 5
):
    spec = replace(ad.ARMS[arm], path=root)
    return ad.load_codex_pilot_arm(
        spec,
        max_reps,
        references=REFERENCES,
        tasks_by_id=TASKS_BY_ID,
        data_dir=data_dir,
    )


def write_or_run(
    root: Path,
    *,
    exec_mode: str = "or_generate",
    prompt_mode: str = "zero_shot",
    rep: int = 0,
    scoring: dict | None = None,
    ok: bool = True,
    cost: float = 0.25,
    num_turns: int | None = None,
    attempts: list[dict] | None = None,
    input_bytes: bytes = CANONICAL_BYTES,
) -> Path:
    name = f"ols_robust__python__{prompt_mode}__{exec_mode}__rep{rep:02d}"
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "input.csv").write_bytes(input_bytes)
    (run_dir / "run.py").write_text(SCRIPT_TEXT)
    record: dict = {
        "task_id": "ols_robust",
        "platform": "python",
        "prompt_mode": prompt_mode,
        "exec_mode": exec_mode,
        "rep": rep,
        "model": "deepseek/deepseek-v4-flash",
        "prompt_sha256": "d" * 64,
        "created_at": f"2026-07-14T08:00:{rep:02d}+00:00",
        "execution_success": True,
        "total_wall_seconds": 5.0,
        "scoring": scoring if scoring is not None else passing_inline_scoring(),
    }
    usage = {
        "input_tokens": 700,
        "cached_input_tokens": 0,
        "output_tokens": 600,
        "reasoning_output_tokens": 300,
    }
    if attempts is not None:
        for idx, attempt in enumerate(attempts, start=1):
            record[f"openrouter_attempt_{idx}"] = {
                "ok": attempt.get("ok", True),
                "cost_usd": attempt.get("cost_usd"),
                "num_turns": attempt.get("num_turns"),
                "usage": attempt.get("usage", usage),
                "wall_seconds": 2.0,
            }
    else:
        record["openrouter"] = {
            "ok": ok,
            "cost_usd": cost,
            "num_turns": num_turns,
            "usage": usage,
            "wall_seconds": 2.0,
        }
    (run_dir / "run_record.json").write_text(json.dumps(record))
    return run_dir


def load_or(
    root: Path,
    data_dir: Path,
    arm: str = "deepseek_zero_shot",
    max_reps: int | None = 5,
):
    spec = replace(ad.ARMS[arm], path=root)
    return ad.load_openrouter_pilot_arm(
        spec,
        max_reps,
        references=REFERENCES,
        tasks_by_id=TASKS_BY_ID,
        data_dir=data_dir,
    )


def corpus_score(
    *,
    rep: int = 0,
    run_dir: str = "",
    model_response: bool = True,
    quarantined: bool = False,
    prompt_wave: str = "legacy",
    prompt_mode: str = "zero_shot",
    exec_mode: str = "no_feedback",
    numeric_pass: bool = True,
) -> dict:
    def gate(passing: bool) -> dict:
        return {"pass": passing, "notes": []}

    return {
        "task_id": "ols_robust",
        "platform": "python",
        "exec_mode": exec_mode,
        "prompt_mode": prompt_mode,
        "prompt_wave": prompt_wave,
        "rep": rep,
        "model_response": model_response,
        "quarantined": quarantined,
        "run_dir": run_dir,
        "task_success": numeric_pass,
        "cost_usd": 0.5,
        "wall_seconds": 4.0,
        "tokens": {
            "input_tokens": 1,
            "output_tokens": 2,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 4,
        },
        "gates": {
            "execution": gate(True),
            "artifact": gate(True),
            "numeric": {
                "pass": numeric_pass,
                "notes": [],
                "per_term": {"const": {"coef_pass": numeric_pass}},
            },
            "method": gate(True),
            "integrity": gate(True),
        },
        "criteria": {
            "specification_correctness": gate(True),
            "executability": gate(True),
            "output_correctness": gate(numeric_pass),
            "submission_validity": gate(True),
        },
    }


def corpus_spec(tmp_path: Path, scores: list[dict]) -> ad.ArmSpec:
    """The corpus arm pinned to the derived-report read path.

    The registry default is ``corpus_records``; these fixtures exercise the
    retained report path, so the kind is pinned here.
    """
    path = tmp_path / "task_success_report.json"
    path.write_text(json.dumps({"scores": scores}))
    return replace(ad.ARMS["sonnet_zero_shot"], kind="corpus_report", path=path)


def make_row(arm_id: str = "sonnet_few_shot", **overrides) -> dict:
    row: dict = {
        "model": "sonnet",
        "model_id": "claude-sonnet-4-6",
        "prompt": "few_shot",
        "interface": "no_feedback",
        "interface_label": "chatbot",
        "platform": "python",
        "task": "ols_robust",
        "rep": 0,
        "task_success": True,
        "gates": dict.fromkeys(ad.GATE_NAMES, True),
        "criteria": dict.fromkeys(ad.CRITERION_NAMES, True),
        "per_term": {},
        "cost_usd": 0.1,
        "tokens": dict.fromkeys(ad.TOKEN_KEYS, 0),
        "wall_seconds": 1.0,
        "turns": 1,
        "provenance": {
            "arm_id": arm_id,
            "run_dir": "synthetic",
            "source_kind": "sonnet_wave",
            "prompt_sha256": None,
            "scoring_source": "canonical_rescore",
            "usable_reason": "ok",
        },
    }
    row.update(overrides)
    return row


def stub_report(arm_id: str, present: bool = True) -> ad.ArmLoadReport:
    return ad.ArmLoadReport(arm_id, present, dict.fromkeys(ad.ATTRITION_KEYS, 0), None)


# 1. Registry invariants ------------------------------------------------------


def test_registry_has_six_unique_arms():
    assert list(ad.ARMS) == [
        "sonnet_zero_shot",
        "sonnet_few_shot",
        "codex_zero_shot",
        "codex_few_shot",
        "deepseek_zero_shot",
        "deepseek_few_shot",
    ]
    pairs = {(spec.model, spec.prompt_mode) for spec in ad.ARMS.values()}
    assert pairs == {(m, p) for m in ad.MODELS for p in ad.PROMPTS}
    for arm_id, spec in ad.ARMS.items():
        assert spec.arm_id == arm_id


def test_registry_paths_avoid_stray_dir():
    for stray in ad.EXCLUDED_STRAY_DIRS:
        for spec in ad.ARMS.values():
            assert spec.path != stray
            assert stray not in spec.path.parents


def test_no_arm_allows_an_absent_root():
    """Since the 2026-08-18 promotion every analysis arm exists on disk."""
    absent_ok = [s.arm_id for s in ad.ARMS.values() if s.allow_absent]
    assert absent_ok == []


def test_condition_maps_target_canonical_interfaces():
    for spec in ad.ARMS.values():
        assert set(spec.condition_map.values()) == set(ad.INTERFACES)


def test_analysis_task_registry_is_21():
    assert len(ad.ANALYSIS_TASK_IDS) == 21
    assert "wcb_few" not in ad.ANALYSIS_TASK_IDS
    assert "ols_robust" in ad.ANALYSIS_TASK_IDS


# 2. five_gate_success --------------------------------------------------------


def test_five_gate_success_requires_all_gates():
    gates = dict.fromkeys(ad.GATE_NAMES, True)
    assert ad.five_gate_success(gates) is True
    for gate in ad.GATE_NAMES:
        assert ad.five_gate_success({**gates, gate: False}) is False
    assert ad.five_gate_success({**gates, "method": None}) is False


# 3. criteria_from_gates ------------------------------------------------------


def test_criteria_from_gates_matches_strict_posthoc_mapping():
    gates = {
        "execution": True,
        "artifact": True,
        "numeric": False,
        "method": True,
        "integrity": True,
    }
    assert ad.criteria_from_gates(gates) == {
        "specification_correctness": True,
        "executability": True,
        "output_correctness": False,
        "submission_validity": True,
    }
    gates_fail = {
        "execution": False,
        "artifact": False,
        "numeric": True,
        "method": False,
        "integrity": False,
    }
    assert ad.criteria_from_gates(gates_fail) == dict.fromkeys(
        ad.CRITERION_NAMES, False
    )


# 4. Rep rule -----------------------------------------------------------------


def test_rep_rule_caps_at_five(tmp_path):
    root = tmp_path / "wave"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    for rep in range(8):
        write_wave_run(root, rep=rep, result=correct_result())
    rows, report = load_wave(root, data_dir, max_reps=5)
    assert [row["rep"] for row in rows] == [0, 1, 2, 3, 4]
    assert report.counts["excluded_rep_over_cap"] == 3
    rows_all, report_all = load_wave(root, data_dir, max_reps=None)
    assert len(rows_all) == 8
    assert report_all.counts["excluded_rep_over_cap"] == 0


def test_balanced_selector_backfills_a_missing_rep_with_rep_five():
    rows = [make_row(rep=rep) for rep in (0, 1, 3, 4, 5, 6)]

    selected = ad.select_first_eligible_rows(rows, max_reps=5)

    assert [row["rep"] for row in selected] == [0, 1, 3, 4, 5]


# 5. wave_record_usable -------------------------------------------------------


def _record_of(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_record.json").read_text())


def test_wave_usable_timeout_excluded(tmp_path):
    run_dir = write_wave_run(tmp_path, ok=False, error="timeout")
    assert ad.wave_record_usable(run_dir, _record_of(run_dir)) == (False, "timeout")


def test_wave_timeout_keeps_missing_cost_as_missing(tmp_path):
    root = tmp_path / "wave"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_wave_run(root, ok=False, error="timeout")

    rows, report = load_wave(root, data_dir)

    assert len(rows) == 1
    assert rows[0]["task_success"] is False
    assert rows[0]["cost_usd"] is None
    assert report.counts["itt_timeout_failure"] == 1


def test_wave_usable_session_limit_excluded(tmp_path):
    run_dir = write_wave_run(
        tmp_path,
        ok=False,
        error="claude exited 1",
        stdout_tail='{"api_error_status":429,"type":"result"}',
    )
    assert ad.wave_record_usable(run_dir, _record_of(run_dir)) == (
        False,
        "session_limit",
    )


def test_wave_usable_max_turns_rescued_by_result_json(tmp_path):
    run_dir = write_wave_run(
        tmp_path,
        ok=False,
        error="claude exited 1",
        stdout_tail='{"terminal_reason":"max_turns"}',
        result=correct_result(),
    )
    usable, reason = ad.wave_record_usable(run_dir, _record_of(run_dir))
    assert usable is True
    assert reason == "result_json_rescue"


def test_wave_usable_ok_block(tmp_path):
    run_dir = write_wave_run(tmp_path, result=correct_result())
    assert ad.wave_record_usable(run_dir, _record_of(run_dir)) == (True, "ok")


# 6. Wave loader end-to-end ---------------------------------------------------


def test_wave_loader_scores_canonically(tmp_path):
    root = tmp_path / "wave"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_wave_run(root, result=correct_result())
    rows, report = load_wave(root, data_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["task_success"] is True
    assert row["gates"] == dict.fromkeys(ad.GATE_NAMES, True)
    assert row["criteria"] == dict.fromkeys(ad.CRITERION_NAMES, True)
    assert row["cost_usd"] == 0.125
    assert row["turns"] == 2
    assert row["tokens"] == {
        "input_tokens": 10,
        "cached_input_tokens": 30,
        "cache_creation_input_tokens": 40,
        "output_tokens": 20,
        "reasoning_output_tokens": 0,
    }
    assert row["model"] == "sonnet"
    assert row["prompt"] == "few_shot"
    assert row["interface_label"] == "chatbot"
    assert row["provenance"]["scoring_source"] == "canonical_rescore"
    assert row["per_term"]["const"]["coef_pass"] is True
    assert report.counts["usable"] == 1
    assert report.max_created_at == "2026-07-27T10:00:00+00:00"


def test_wave_loader_recomputes_integrity(tmp_path):
    root = tmp_path / "wave"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_wave_run(root, result=correct_result(), input_bytes=b"tampered\n")
    rows, _ = load_wave(root, data_dir)
    row = rows[0]
    assert row["gates"]["integrity"] is False
    assert all(row["gates"][g] for g in ("execution", "artifact", "numeric", "method"))
    assert row["task_success"] is False
    assert row["criteria"]["submission_validity"] is False


# 7. Wave loader ignores inline scoring ---------------------------------------


def test_wave_loader_ignores_inline_scoring(tmp_path):
    root = tmp_path / "wave"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    wrong = correct_result()
    wrong["estimates"][0]["estimate"] = 9.99
    write_wave_run(
        root,
        result=wrong,
        inline_scoring={"numeric_success": True, "artifact_success": True},
    )
    rows, _ = load_wave(root, data_dir)
    assert rows[0]["gates"]["numeric"] is False
    assert rows[0]["task_success"] is False


def test_codex_loader_recomputes_all_inline_gates(tmp_path):
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    run_dir = write_codex_run(root, scoring=passing_inline_scoring())
    wrong = correct_result()
    wrong["estimates"][0]["estimate"] = 9.99
    (run_dir / "result.json").write_text(json.dumps(wrong))

    rows, _ = load_codex(root, data_dir, arm="codex_zero_shot")

    assert rows[0]["gates"]["numeric"] is False
    assert rows[0]["task_success"] is False
    assert rows[0]["provenance"]["scoring_source"] == "canonical_rescore"


def test_openrouter_loader_recomputes_all_inline_gates(tmp_path):
    root = tmp_path / "openrouter"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    run_dir = write_or_run(root, scoring=passing_inline_scoring())
    wrong = correct_result()
    wrong["estimates"][0]["estimate"] = 9.99
    (run_dir / "result.json").write_text(json.dumps(wrong))

    rows, _ = load_or(root, data_dir)

    assert rows[0]["gates"]["numeric"] is False
    assert rows[0]["task_success"] is False
    assert rows[0]["provenance"]["scoring_source"] == "canonical_rescore"


# 8. self_repair cost and token summation -------------------------------------


def test_wave_self_repair_sums_attempt_blocks(tmp_path):
    root = tmp_path / "wave"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    blocks = {
        "claude_attempt_1": {
            "ok": True,
            "total_cost_usd": 0.125,
            "wall_seconds": 2.0,
            "num_turns": 1,
            "usage": {
                "input_tokens": 5,
                "output_tokens": 7,
                "cache_read_input_tokens": 11,
                "cache_creation_input_tokens": 13,
            },
        },
        "claude_attempt_2": {
            "ok": True,
            "total_cost_usd": 0.25,
            "wall_seconds": 3.0,
            "num_turns": 2,
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 4,
            },
        },
    }
    write_wave_run(
        root, interface="self_repair", blocks=blocks, result=correct_result()
    )
    rows, _ = load_wave(root, data_dir)
    row = rows[0]
    assert row["cost_usd"] == 0.375
    assert row["tokens"]["input_tokens"] == 6
    assert row["tokens"]["output_tokens"] == 9
    assert row["tokens"]["cached_input_tokens"] == 14
    assert row["tokens"]["cache_creation_input_tokens"] == 17
    assert row["turns"] == 3
    assert row["interface_label"] == "one-repair"


# 9. Codex loader -------------------------------------------------------------


def test_codex_failed_run_stays_in_denominator(tmp_path):
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    scoring = {
        **passing_inline_scoring(),
        "execution_pass": False,
        "artifact_success": False,
        "numeric_success": False,
        "method_success": False,
        "model_response": False,
        "per_term": {},
    }
    write_codex_run(root, prompt_mode="few_shot", scoring=scoring)
    rows, report = load_codex(root, data_dir)
    assert len(rows) == 1
    assert rows[0]["task_success"] is False
    assert rows[0]["cost_usd"] is None
    assert rows[0]["turns"] is None
    assert report.counts["usable"] == 1
    assert report.counts["excluded_collection_failure"] == 0


def test_codex_timeout_becomes_itt_failure_row(tmp_path):
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_codex_run(root, prompt_mode="few_shot", ok=False, timed_out=True)
    rows, report = load_codex(root, data_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["task_success"] is False
    assert not any(row["gates"].values())
    assert row["provenance"]["itt_failure"] is True
    assert report.counts["itt_timeout_failure"] == 1
    assert report.counts["excluded_collection_failure"] == 0


def test_codex_cli_failure_stays_excluded(tmp_path):
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_codex_run(root, prompt_mode="few_shot", ok=False, timed_out=False)
    rows, report = load_codex(root, data_dir)
    assert rows == []
    assert report.counts["excluded_collection_failure"] == 1


def test_codex_few_shot_skips_in_flight_dir(tmp_path):
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_codex_run(root, prompt_mode="few_shot")
    write_codex_run(root, prompt_mode="few_shot", rep=1, omit_record=True)
    rows, report = load_codex(root, data_dir)
    assert len(rows) == 1
    assert report.counts["in_flight_missing_record"] == 1


def test_codex_zero_shot_missing_record_raises(tmp_path):
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_codex_run(root, prompt_mode="zero_shot", omit_record=True)
    with pytest.raises(RuntimeError, match="run_record.json"):
        load_codex(root, data_dir, arm="codex_zero_shot")


def test_codex_retired_fewshot_excluded_from_zero_shot_arm(tmp_path):
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_codex_run(root, prompt_mode="few_shot")
    write_codex_run(root, prompt_mode="zero_shot")
    rows, report = load_codex(root, data_dir, arm="codex_zero_shot")
    assert len(rows) == 1
    assert rows[0]["prompt"] == "zero_shot"
    assert report.counts["excluded_retired_fewshot"] == 1


def test_codex_underscore_dir_skipped(tmp_path):
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    (root / "_quarantine_2026-07-15").mkdir()
    write_codex_run(root, prompt_mode="few_shot")
    rows, report = load_codex(root, data_dir)
    assert len(rows) == 1
    assert report.counts["excluded_underscore_dir"] == 1


def test_codex_repair_four_attempts_keeps_whole_run_accounting(tmp_path):
    """Final-artifact rule: no boundary veto, every attempt enters the sums."""
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    attempts = [
        {
            "input_tokens": 100 * k,
            "cached_input_tokens": 10 * k,
            "output_tokens": 5 * k,
            "reasoning_output_tokens": k,
        }
        for k in (1, 2, 3, 4)
    ]
    write_codex_run(
        root, exec_mode="codex_repair", prompt_mode="few_shot", attempts=attempts
    )
    rows, _report = load_codex(root, data_dir)
    row = rows[0]
    assert row["interface"] == "self_repair"
    assert "boundary_rescored" not in row["provenance"]
    tokens = row["tokens"]
    assert tokens["input_tokens"] == 1000
    assert tokens["cached_input_tokens"] == 100
    assert tokens["output_tokens"] == 50
    assert tokens["reasoning_output_tokens"] == 10
    assert row["revisions_used"] == 3


def test_codex_repair_two_attempts_tokens_summed(tmp_path):
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    attempts = [
        {
            "input_tokens": 100 * k,
            "cached_input_tokens": 10 * k,
            "output_tokens": 5 * k,
            "reasoning_output_tokens": k,
        }
        for k in (1, 2)
    ]
    write_codex_run(
        root, exec_mode="codex_repair", prompt_mode="few_shot", attempts=attempts
    )
    rows, _report = load_codex(root, data_dir)
    row = rows[0]
    assert row["tokens"]["input_tokens"] == 300
    assert "boundary_rescored" not in row["provenance"]
    assert row["revisions_used"] == 1


# 10. OpenRouter loader -------------------------------------------------------


def test_openrouter_keeps_ok_run_with_failed_scoring(tmp_path):
    root = tmp_path / "or"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    scoring = {
        **passing_inline_scoring(),
        "numeric_success": False,
        "model_response": False,
    }
    write_or_run(root, scoring=scoring)
    rows, report = load_or(root, data_dir)
    assert len(rows) == 1
    assert rows[0]["task_success"] is False
    assert rows[0]["cost_usd"] == 0.25
    assert report.counts["usable"] == 1


def test_openrouter_excludes_run_without_ok_block(tmp_path):
    root = tmp_path / "or"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_or_run(root, ok=False)
    rows, report = load_or(root, data_dir)
    assert rows == []
    assert report.counts["excluded_collection_failure"] == 1


def test_openrouter_repair_sums_cost_and_turns(tmp_path):
    root = tmp_path / "or"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    attempts = [
        {"ok": True, "cost_usd": 0.25, "num_turns": 2},
        {"ok": True, "cost_usd": 0.5, "num_turns": 3},
    ]
    write_or_run(root, exec_mode="or_repair", attempts=attempts)
    rows, _ = load_or(root, data_dir)
    assert rows[0]["cost_usd"] == 0.75
    assert rows[0]["turns"] == 5
    assert rows[0]["interface"] == "self_repair"


def test_deepseek_arms_use_the_swe_condition_map():
    """Promoted 2026-08-18: the agent condition is stored as or_agent_swe."""
    assert ad.OPENROUTER_SWE_CONDITIONS == {
        "or_generate": "no_feedback",
        "or_repair": "self_repair",
        "or_agent_swe": "tool_loop",
    }
    for arm_id in ("deepseek_zero_shot", "deepseek_few_shot"):
        spec = ad.ARMS[arm_id]
        assert spec.condition_map == ad.OPENROUTER_SWE_CONDITIONS
        assert spec.kind == "openrouter_pilot"
        assert spec.allow_absent is False


def test_openrouter_swe_agent_record_loads_as_tool_loop(tmp_path):
    root = tmp_path / "or"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_or_run(root, exec_mode="or_agent_swe")
    rows, report = load_or(root, data_dir)
    assert len(rows) == 1
    assert rows[0]["interface"] == "tool_loop"
    assert rows[0]["interface_label"] == "constrained agent"
    assert report.counts["usable"] == 1


def test_openrouter_legacy_or_agent_record_is_wrong_arm_now(tmp_path):
    """A July or_agent record does not enter the promoted arm."""
    root = tmp_path / "or"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_or_run(root, exec_mode="or_agent")
    rows, report = load_or(root, data_dir)
    assert rows == []
    assert report.counts["excluded_wrong_arm"] == 1


def test_openrouter_retired_fewshot_excluded(tmp_path):
    root = tmp_path / "or"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_or_run(root, prompt_mode="few_shot")
    write_or_run(root, prompt_mode="zero_shot")
    rows, report = load_or(root, data_dir)
    assert len(rows) == 1
    assert rows[0]["prompt"] == "zero_shot"
    assert report.counts["excluded_retired_fewshot"] == 1


def test_openrouter_absent_root_reports_absent(tmp_path):
    # Both promoted arms exist on disk now (allow_absent=False); the
    # absent-root code path stays covered through an explicit override.
    spec = replace(
        ad.ARMS["deepseek_few_shot"],
        path=tmp_path / "missing",
        allow_absent=True,
    )
    rows, report = ad.load_openrouter_pilot_arm(
        spec,
        5,
        references=REFERENCES,
        tasks_by_id=TASKS_BY_ID,
        data_dir=tmp_path,
    )
    assert rows == []
    assert report.present is False
    assert report.counts["usable"] == 0


def test_openrouter_missing_root_raises_when_not_allowed(tmp_path):
    spec = replace(ad.ARMS["deepseek_zero_shot"], path=tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        ad.load_openrouter_pilot_arm(
            spec,
            5,
            references=REFERENCES,
            tasks_by_id=TASKS_BY_ID,
            data_dir=tmp_path,
        )


# 11. Corpus-report loader ----------------------------------------------------


def test_corpus_loader_applies_paper_filter(tmp_path):
    turn_dir = tmp_path / "runs" / "ols_robust__python__zero_shot__rep00"
    turn_dir.mkdir(parents=True)
    (turn_dir / "run_record.json").write_text(
        json.dumps({"claude": {"ok": True, "num_turns": 3}})
    )
    scores = [
        corpus_score(run_dir=str(turn_dir)),
        corpus_score(model_response=False, rep=1),
        corpus_score(quarantined=True),
        corpus_score(prompt_wave="pinned-2026-07"),
        corpus_score(prompt_mode="few_shot"),
    ]
    rows, report = ad.load_corpus_report_arm(corpus_spec(tmp_path, scores), 5)
    assert len(rows) == 1
    row = rows[0]
    assert row["task_success"] is True
    assert row["turns"] == 3
    assert row["cost_usd"] == 0.5
    assert row["tokens"]["cached_input_tokens"] == 3
    assert row["tokens"]["reasoning_output_tokens"] == 0
    assert row["per_term"] == {"const": {"coef_pass": True}}
    assert row["provenance"]["scoring_source"] == "report"
    assert row["provenance"]["prompt_sha256"] is None
    assert report.counts["excluded_no_response_other"] == 1
    assert report.counts["excluded_wrong_arm"] == 3
    assert report.counts["records_parsed"] == 2
    assert report.counts["turns_unavailable"] == 0


def test_load_runs_backfills_provider_interruption_with_later_legacy_rep(
    tmp_path, monkeypatch
):
    interrupted = tmp_path / "runs" / "ols_robust__python__zero_shot__rep02"
    interrupted.mkdir(parents=True)
    (interrupted / "run_record.json").write_text(
        json.dumps(
            {"claude": {"ok": False, "stdout_tail": 'x "api_error_status":429 y'}}
        )
    )
    scores = [
        corpus_score(
            rep=rep,
            model_response=rep != 2,
            run_dir=str(interrupted) if rep == 2 else "",
        )
        for rep in range(6)
    ]
    spec = corpus_spec(tmp_path, scores)
    monkeypatch.setitem(ad.ARMS, "sonnet_zero_shot", spec)

    result = ad.load_runs(arm_ids=["sonnet_zero_shot"], max_reps=5)

    assert [row["rep"] for row in result.rows] == [0, 1, 3, 4, 5]
    assert result.reports["sonnet_zero_shot"].counts["excluded_session_limit"] == 1
    assert result.manifest["arms"]["sonnet_zero_shot"]["usable_runs"] == 5


def test_load_runs_uses_pinned_corpus_only_for_missing_legacy_key(
    tmp_path, monkeypatch
):
    scores = [corpus_score(rep=rep) for rep in range(4)]
    scores.extend(
        [
            corpus_score(rep=3, prompt_wave="pinned-2026-07", numeric_pass=False),
            corpus_score(rep=4, prompt_wave="pinned-2026-07", numeric_pass=False),
        ]
    )
    spec = corpus_spec(tmp_path, scores)
    monkeypatch.setitem(ad.ARMS, "sonnet_zero_shot", spec)

    result = ad.load_runs(arm_ids=["sonnet_zero_shot"], max_reps=5)

    assert [row["rep"] for row in result.rows] == [0, 1, 2, 3, 4]
    assert len(result.rows) == 5
    by_rep = {row["rep"]: row for row in result.rows}
    # These fixture rows leave no result.json, so the unified primary outcome
    # fails; the loader preserves the five-gate verdict on task_success_strict.
    assert by_rep[3]["task_success_strict"] is True
    assert by_rep[3]["task_success"] is False
    assert by_rep[3]["unified"]["repair_rule"] == "no_file"
    assert by_rep[3]["provenance"]["prompt_wave"] == "legacy"
    assert by_rep[4]["task_success_strict"] is False
    assert by_rep[4]["provenance"]["prompt_wave"] == "pinned-2026-07"


def test_corpus_timeout_no_response_becomes_itt_failure_row(tmp_path):
    run_dir = tmp_path / "runs" / "ols_robust__python__zero_shot__rep01"
    run_dir.mkdir(parents=True)
    (run_dir / "run_record.json").write_text(
        json.dumps({"claude": {"ok": False, "error": "timeout"}})
    )
    scores = [corpus_score(model_response=False, run_dir=str(run_dir), rep=1)]
    rows, report = ad.load_corpus_report_arm(corpus_spec(tmp_path, scores), 5)
    assert len(rows) == 1
    row = rows[0]
    assert row["task_success"] is False
    assert not any(row["gates"].values())
    assert not any(row["criteria"].values())
    assert row["provenance"]["itt_failure"] is True
    assert row["provenance"]["scoring_source"] == "itt"
    assert report.counts["itt_timeout_failure"] == 1
    assert report.counts["excluded_no_response_other"] == 0


def test_corpus_session_limit_no_response_stays_excluded(tmp_path):
    run_dir = tmp_path / "runs" / "ols_robust__python__zero_shot__rep01"
    run_dir.mkdir(parents=True)
    (run_dir / "run_record.json").write_text(
        json.dumps(
            {"claude": {"ok": False, "stdout_tail": 'x "api_error_status":429 y'}}
        )
    )
    scores = [corpus_score(model_response=False, run_dir=str(run_dir), rep=1)]
    rows, report = ad.load_corpus_report_arm(corpus_spec(tmp_path, scores), 5)
    assert rows == []
    assert report.counts["excluded_session_limit"] == 1


def test_corpus_repair_multi_revision_keeps_stored_outcome(tmp_path):
    """Final-artifact rule: extra revisions keep the stored gates and sums."""
    run_dir = tmp_path / "runs" / "ols_robust__python__zero_shot__rep00"
    run_dir.mkdir(parents=True)
    (run_dir / "run_record.json").write_text(
        json.dumps(
            {
                "claude_attempt_1": {
                    "ok": True,
                    "num_turns": 1,
                    "total_cost_usd": 0.1,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
                "claude_attempt_2": {
                    "ok": True,
                    "num_turns": 1,
                    "total_cost_usd": 0.2,
                    "usage": {"input_tokens": 20, "output_tokens": 2},
                },
                "claude_attempt_3": {
                    "ok": True,
                    "num_turns": 1,
                    "total_cost_usd": 0.4,
                    "usage": {"input_tokens": 40, "output_tokens": 4},
                },
                "execution_attempt_2": {"ok": False},
            }
        )
    )
    scores = [
        corpus_score(exec_mode="self_repair", run_dir=str(run_dir)),
    ]
    rows, _report = ad.load_corpus_report_arm(corpus_spec(tmp_path, scores), 5)
    assert len(rows) == 1
    row = rows[0]
    assert row["task_success"] is True
    assert row["gates"]["execution"] is True
    assert row["gates"]["numeric"] is True
    assert "boundary_rescored" not in row["provenance"]
    assert row["revisions_used"] == 2


def test_corpus_repair_single_revision_untouched(tmp_path):
    run_dir = tmp_path / "runs" / "ols_robust__python__zero_shot__rep00"
    run_dir.mkdir(parents=True)
    (run_dir / "run_record.json").write_text(
        json.dumps(
            {
                "claude_attempt_1": {"ok": True, "num_turns": 1},
                "claude_attempt_2": {"ok": True, "num_turns": 1},
            }
        )
    )
    scores = [corpus_score(exec_mode="self_repair", run_dir=str(run_dir))]
    rows, _report = ad.load_corpus_report_arm(corpus_spec(tmp_path, scores), 5)
    assert rows[0]["task_success"] is True
    assert "boundary_rescored" not in rows[0]["provenance"]
    assert rows[0]["revisions_used"] == 1


def test_corpus_loader_passes_fields_through_verbatim(tmp_path):
    scores = [corpus_score(numeric_pass=False, run_dir="")]
    rows, _ = ad.load_corpus_report_arm(corpus_spec(tmp_path, scores), 5)
    row = rows[0]
    assert row["task_success"] is False
    assert row["gates"]["numeric"] is False
    assert row["criteria"]["output_correctness"] is False
    assert row["gates"]["execution"] is True


def test_corpus_loader_counts_missing_turns(tmp_path):
    scores = [corpus_score(run_dir=str(tmp_path / "no_such_run"))]
    rows, report = ad.load_corpus_report_arm(corpus_spec(tmp_path, scores), 5)
    assert rows[0]["turns"] is None
    assert report.counts["turns_unavailable"] == 1


def test_corpus_loader_include_turns_false_skips_reads(tmp_path):
    scores = [corpus_score(run_dir=str(tmp_path / "no_such_run"))]
    rows, report = ad.load_corpus_report_arm(
        corpus_spec(tmp_path, scores), 5, include_turns=False
    )
    assert rows[0]["turns"] is None
    assert report.counts["turns_unavailable"] == 0


def test_corpus_loader_rep_rule(tmp_path):
    scores = [corpus_score(rep=rep, run_dir="") for rep in range(7)]
    rows, report = ad.load_corpus_report_arm(
        corpus_spec(tmp_path, scores), 5, include_turns=False
    )
    assert [row["rep"] for row in rows] == [0, 1, 2, 3, 4]
    assert report.counts["excluded_rep_over_cap"] == 2


# 12. Completeness manifest ---------------------------------------------------


def test_manifest_classifies_cells():
    rows = [make_row(rep=rep) for rep in range(5)]
    rows += [
        make_row(rep=rep, platform="r", interface="self_repair") for rep in range(3)
    ]
    reports = {
        "sonnet_few_shot": stub_report("sonnet_few_shot"),
        "deepseek_few_shot": stub_report("deepseek_few_shot", present=False),
    }
    manifest = ad.completeness_manifest(rows, reports, 5)
    assert manifest["expected"] == {
        "tasks": 21,
        "platforms": 3,
        "interfaces": 3,
        "cells": 189,
        "runs_at_12_reps": 945,
    }
    arm = manifest["arms"]["sonnet_few_shot"]
    assert arm["status"] == "partial"
    assert arm["usable_runs"] == 8
    assert arm["cells_full"] == 1
    assert arm["cells_short"] == 1
    assert arm["cells_empty"] == 187
    assert arm["short_cells"] == [{"cell": "ols_robust__r__self_repair", "n": 3}]
    assert "ols_robust__python__no_feedback" not in arm["empty_cells"]
    assert "anova_oneway__python__no_feedback" in arm["empty_cells"]
    assert "anova_oneway" in arm["tasks_entirely_absent"]
    assert "ols_robust" not in arm["tasks_entirely_absent"]
    absent = manifest["arms"]["deepseek_few_shot"]
    assert absent["status"] == "absent"
    assert absent["usable_runs"] == 0
    assert absent["cells_empty"] == 189


def test_manifest_complete_status():
    rows = [
        make_row(task=task, platform=platform, interface=interface, rep=rep)
        for task in ad.ANALYSIS_TASK_IDS
        for platform in ad.PLATFORMS
        for interface in ad.INTERFACES
        for rep in range(5)
    ]
    reports = {"sonnet_few_shot": stub_report("sonnet_few_shot")}
    manifest = ad.completeness_manifest(rows, reports, 5)
    arm = manifest["arms"]["sonnet_few_shot"]
    assert arm["status"] == "complete"
    assert arm["usable_runs"] == 945
    assert arm["cells_full"] == 189
    assert arm["empty_cells"] == []
    assert arm["tasks_entirely_absent"] == []


# load_runs orchestration -----------------------------------------------------


def test_load_runs_reroots_against_project_root(tmp_path):
    project = tmp_path / "proj"
    wave_rel = ad.ARMS["sonnet_few_shot"].path.relative_to(ad.PROJECT_ROOT)
    root = project / wave_rel
    root.mkdir(parents=True)
    data_dir = make_data_dir(tmp_path)
    write_wave_run(root, result=correct_result())
    result = ad.load_runs(
        arm_ids=["sonnet_few_shot"],
        max_reps=5,
        project_root=project,
        references=REFERENCES,
        tasks_by_id=TASKS_BY_ID,
        data_dir=data_dir,
    )
    assert len(result.rows) == 1
    assert set(result.reports) == {"sonnet_few_shot"}
    assert result.manifest["arms"]["sonnet_few_shot"]["usable_runs"] == 1
    row = result.rows[0]
    assert row["provenance"]["arm_id"] == "sonnet_few_shot"
    # The loader annotates the unified primary outcome on every row.
    assert row["task_success"] is True
    assert row["task_success_strict"] is True
    assert row["unified"]["repair_rule"] == "none needed"


def test_load_runs_rejects_unknown_arm():
    with pytest.raises(KeyError, match="unknown arm ids"):
        ad.load_runs(arm_ids=["gemini_zero_shot"])


# Unified primary outcome ------------------------------------------------------


def test_attach_unified_outcome_rescues_repairable_artifact(tmp_path):
    root = tmp_path / "wave"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    run_dir = write_wave_run(root, result=correct_result())
    raw = (run_dir / "result.json").read_text().replace('"', '""')
    (run_dir / "result.json").write_text(raw)
    rows, _ = load_wave(root, data_dir)
    row = rows[0]
    assert row["task_success"] is False  # strict verdict before annotation
    assert row["gates"]["integrity"] is True

    ad.attach_unified_outcome(rows, tasks_by_id=TASKS_BY_ID, references=REFERENCES)

    assert row["task_success"] is True
    assert row["task_success_strict"] is False
    assert row["unified"]["repair_rule"] == "doubled_quotes"
    assert row["unified"]["numeric_repaired_pass"] is True


def test_attach_unified_outcome_requires_integrity(tmp_path):
    root = tmp_path / "wave"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_wave_run(root, result=correct_result(), input_bytes=b"tampered\n")
    rows, _ = load_wave(root, data_dir)

    ad.attach_unified_outcome(rows, tasks_by_id=TASKS_BY_ID, references=REFERENCES)

    row = rows[0]
    assert row["gates"]["integrity"] is False
    assert row["unified"]["numeric_repaired_pass"] is True
    assert row["task_success"] is False


def test_attach_unified_outcome_scores_multi_revision_final_artifact(tmp_path):
    """Final-artifact rule: the scorer judges the final saved file."""
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    attempts = [
        {
            "input_tokens": 100,
            "cached_input_tokens": 0,
            "output_tokens": 5,
            "reasoning_output_tokens": 1,
        }
        for _ in range(4)
    ]
    run_dir = write_codex_run(
        root, exec_mode="codex_repair", prompt_mode="few_shot", attempts=attempts
    )
    (run_dir / "result.json").write_text(json.dumps(correct_result()))
    rows, _ = load_codex(root, data_dir)
    row = rows[0]
    assert "boundary_rescored" not in row["provenance"]
    assert row["revisions_used"] == 3

    ad.attach_unified_outcome(rows, tasks_by_id=TASKS_BY_ID, references=REFERENCES)

    assert row["task_success"] is True
    assert row["unified"]["numeric_repaired_pass"] is True


def test_attach_unified_outcome_is_idempotent(tmp_path):
    root = tmp_path / "wave"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_wave_run(root, result=correct_result())
    rows, _ = load_wave(root, data_dir)

    ad.attach_unified_outcome(rows, tasks_by_id=TASKS_BY_ID, references=REFERENCES)
    first = {
        "task_success": rows[0]["task_success"],
        "task_success_strict": rows[0]["task_success_strict"],
        "unified": dict(rows[0]["unified"]),
    }
    ad.attach_unified_outcome(rows, tasks_by_id=TASKS_BY_ID, references=REFERENCES)

    assert rows[0]["task_success"] == first["task_success"]
    assert rows[0]["task_success_strict"] == first["task_success_strict"]
    assert rows[0]["unified"] == first["unified"]


# 15. Harmonized provider wall time -------------------------------------------


def test_provider_wall_seconds_sums_only_provider_blocks():
    record = {
        "total_wall_seconds": 100.0,
        "codex_attempt_1": {"wall_seconds": 5.0},
        "codex_attempt_2": {"wall_seconds": 7.0},
        "openrouter": {"wall_seconds": 3.0},
    }
    assert ad.sum_blocks_wall_seconds(record, "codex") == pytest.approx(12.0)
    assert ad.sum_blocks_wall_seconds(record, "openrouter") == pytest.approx(3.0)
    assert ad.sum_blocks_wall_seconds(record, "claude") is None


def test_provider_wall_seconds_honors_a_block_limit():
    record = {
        "codex_attempt_1": {"wall_seconds": 5.0},
        "codex_attempt_2": {"wall_seconds": 7.0},
        "codex_attempt_3": {"wall_seconds": 11.0},
    }
    assert ad.sum_blocks_wall_seconds(record, "codex", 2) == pytest.approx(12.0)


def test_wave_provider_wall_ignores_total_wall_seconds(tmp_path):
    root = tmp_path / "wave"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_wave_run(root, result=correct_result())
    rows, _ = load_wave(root, data_dir)
    row = rows[0]
    # The record carries total_wall_seconds 7.0 and one claude block of 3.0.
    # Both Sonnet measures report the provider-call sum.
    assert row["wall_seconds"] == pytest.approx(3.0)
    assert row["provider_wall_seconds"] == pytest.approx(3.0)


def test_codex_provider_wall_differs_from_total_wall(tmp_path):
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_codex_run(root, prompt_mode="few_shot")
    rows, _ = load_codex(root, data_dir)
    row = rows[0]
    # wall_seconds keeps the record total (local execution included); the
    # harmonized field keeps the codex-block sum.
    assert row["wall_seconds"] == pytest.approx(14.0)
    assert row["provider_wall_seconds"] == pytest.approx(6.0)


def test_openrouter_provider_wall_differs_from_total_wall(tmp_path):
    root = tmp_path / "or"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_or_run(root)
    rows, _ = load_or(root, data_dir)
    row = rows[0]
    assert row["wall_seconds"] == pytest.approx(5.0)
    assert row["provider_wall_seconds"] == pytest.approx(2.0)


def test_codex_repair_provider_wall_covers_every_attempt(tmp_path):
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    attempts = [{"input_tokens": 10 * k} for k in (1, 2, 3, 4)]
    write_codex_run(
        root, exec_mode="codex_repair", prompt_mode="few_shot", attempts=attempts
    )
    rows, _ = load_codex(root, data_dir)
    row = rows[0]
    # Four attempt blocks of 5.0 seconds each; every attempt counts.
    assert "boundary_rescored" not in row["provenance"]
    assert row["provider_wall_seconds"] == pytest.approx(20.0)
    assert row["wall_seconds"] == pytest.approx(14.0)


def test_codex_repair_two_attempts_provider_wall_untruncated(tmp_path):
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    attempts = [{"input_tokens": 10 * k} for k in (1, 2)]
    write_codex_run(
        root, exec_mode="codex_repair", prompt_mode="few_shot", attempts=attempts
    )
    rows, _ = load_codex(root, data_dir)
    row = rows[0]
    assert "boundary_rescored" not in row["provenance"]
    assert row["provider_wall_seconds"] == pytest.approx(10.0)
    assert row["wall_seconds"] == pytest.approx(14.0)


def test_openrouter_repair_provider_wall_covers_every_attempt(tmp_path):
    root = tmp_path / "or"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_or_run(root, exec_mode="or_repair", attempts=[{}, {}, {}])
    rows, _ = load_or(root, data_dir)
    row = rows[0]
    # Three attempt blocks of 2.0 seconds each; every attempt counts.
    assert "boundary_rescored" not in row["provenance"]
    assert row["provider_wall_seconds"] == pytest.approx(6.0)
    assert row["revisions_used"] == 2


def test_corpus_provider_wall_matches_stored_provider_sum(tmp_path):
    scores = [corpus_score()]
    rows, _ = ad.load_corpus_report_arm(corpus_spec(tmp_path, scores), 5)
    row = rows[0]
    assert row["wall_seconds"] == pytest.approx(4.0)
    assert row["provider_wall_seconds"] == pytest.approx(4.0)


def test_corpus_repair_provider_wall_keeps_stored_sum(tmp_path):
    run_dir = tmp_path / "runs" / "ols_robust__python__zero_shot__rep00"
    run_dir.mkdir(parents=True)
    (run_dir / "run_record.json").write_text(
        json.dumps(
            {
                "claude_attempt_1": {"ok": True, "wall_seconds": 1.0},
                "claude_attempt_2": {"ok": True, "wall_seconds": 2.0},
                "claude_attempt_3": {"ok": True, "wall_seconds": 4.0},
                "execution_attempt_2": {"ok": False},
            }
        )
    )
    scores = [corpus_score(exec_mode="self_repair", run_dir=str(run_dir))]
    rows, _ = ad.load_corpus_report_arm(corpus_spec(tmp_path, scores), 5)
    row = rows[0]
    assert "boundary_rescored" not in row["provenance"]
    # The stored corpus score carries the full claude-block sum.
    assert row["provider_wall_seconds"] == pytest.approx(4.0)
    assert row["wall_seconds"] == pytest.approx(4.0)
    assert row["revisions_used"] == 2


def test_itt_timeout_rows_carry_provider_wall(tmp_path):
    root = tmp_path / "codex"
    root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_codex_run(root, prompt_mode="few_shot", ok=False, timed_out=True)
    rows, _ = load_codex(root, data_dir)
    row = rows[0]
    assert row["provenance"]["itt_failure"] is True
    assert row["provider_wall_seconds"] == pytest.approx(6.0)
    assert row["wall_seconds"] == pytest.approx(14.0)


def test_every_loaded_row_carries_provider_wall(tmp_path):
    wave_root = tmp_path / "wave"
    wave_root.mkdir()
    data_dir = make_data_dir(tmp_path)
    write_wave_run(wave_root, result=correct_result())
    rows, _ = load_wave(wave_root, data_dir)
    assert all("provider_wall_seconds" in row for row in rows)
# 13. Pinned sample manifest --------------------------------------------------


def _manifest_project(tmp_path):
    """A minimal project root holding one sonnet_few_shot arm with two runs."""
    project = tmp_path / "proj"
    wave_rel = ad.ARMS["sonnet_few_shot"].path.relative_to(ad.PROJECT_ROOT)
    root = project / wave_rel
    root.mkdir(parents=True)
    data_dir = make_data_dir(tmp_path)
    write_wave_run(root, rep=0, result=correct_result())
    write_wave_run(root, rep=1, result=correct_result())
    return project, data_dir


def _manifest_load(project, data_dir, **kwargs):
    return ad.load_runs(
        arm_ids=["sonnet_few_shot"],
        max_reps=5,
        project_root=project,
        references=REFERENCES,
        tasks_by_id=TASKS_BY_ID,
        data_dir=data_dir,
        **kwargs,
    )


def test_sample_manifest_matching_load_passes(tmp_path):
    project, data_dir = _manifest_project(tmp_path)
    result = _manifest_load(project, data_dir)
    manifest = ad.build_sample_manifest(result.rows, 5, project_root=project)
    assert manifest["sample_version"] == ad.SAMPLE_VERSION
    assert manifest["primary_outcome_version"] == ad.PRIMARY_OUTCOME_VERSION
    assert manifest["n_rows"] == 2
    for entry in manifest["runs"]:
        assert entry["result_sha256"] is not None
        assert not Path(entry["run_dir"]).is_absolute()
    manifest_path = project / ad.SAMPLE_MANIFEST_SUBPATH
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))

    verified = _manifest_load(project, data_dir, verify_manifest=True)
    assert len(verified.rows) == 2

    ad.verify_rows_against_sample_manifest(
        result.rows, manifest_path, max_reps=5, project_root=project
    )


def test_sample_manifest_mutated_hash_raises(tmp_path):
    project, data_dir = _manifest_project(tmp_path)
    result = _manifest_load(project, data_dir)
    manifest = ad.build_sample_manifest(result.rows, 5, project_root=project)
    manifest["runs"][0]["result_sha256"] = "0" * 64
    manifest_path = project / ad.SAMPLE_MANIFEST_SUBPATH
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ad.SampleManifestError, match="hash changed"):
        _manifest_load(project, data_dir, verify_manifest=True)


def test_sample_manifest_detects_drifted_artifact(tmp_path):
    project, data_dir = _manifest_project(tmp_path)
    result = _manifest_load(project, data_dir)
    manifest = ad.build_sample_manifest(result.rows, 5, project_root=project)
    manifest_path = project / ad.SAMPLE_MANIFEST_SUBPATH
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))

    run_dir = Path(result.rows[0]["provenance"]["run_dir"])
    mutated = correct_result()
    mutated["estimates"][0]["estimate"] = 999.0
    (run_dir / "result.json").write_text(json.dumps(mutated))

    with pytest.raises(ad.SampleManifestError, match="hash changed"):
        _manifest_load(project, data_dir, verify_manifest=True)


def test_sample_manifest_missing_file_raises(tmp_path):
    project, data_dir = _manifest_project(tmp_path)
    with pytest.raises(ad.SampleManifestError, match="not found"):
        _manifest_load(project, data_dir, verify_manifest=True)


def test_sample_manifest_max_reps_mismatch_raises(tmp_path):
    project, data_dir = _manifest_project(tmp_path)
    result = _manifest_load(project, data_dir)
    manifest = ad.build_sample_manifest(result.rows, 5, project_root=project)
    manifest_path = project / ad.SAMPLE_MANIFEST_SUBPATH
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ad.SampleManifestError, match="max_reps"):
        ad.verify_rows_against_sample_manifest(
            result.rows, manifest_path, max_reps=12, project_root=project
        )


# ---------------------------------------------------------------------------
# Paper scope: the two-model main specification.
# ---------------------------------------------------------------------------


def test_paper_scope_names_two_arms_and_keeps_the_corpus_registry():
    """The paper scope selects arms; it never shrinks the arm registry."""
    assert ad.PAPER_MODELS == ("sonnet", "codex")
    assert ad.MODELS == ("sonnet", "codex", "deepseek")
    assert set(ad.ARMS) == {
        f"{model}_{prompt}" for model in ad.MODELS for prompt in ad.PROMPTS
    }
    assert ad.PAPER_ARM_IDS == (
        "sonnet_zero_shot",
        "sonnet_few_shot",
        "codex_zero_shot",
        "codex_few_shot",
    )
    assert ad.PAPER_SAMPLE_VERSION == "two_model_2026-08-21"
    assert ad.EXPECTED_PAPER_RUNS_12REP == 9072


def test_paper_rows_keeps_only_the_paper_arms():
    rows = [
        {"model": "sonnet"},
        {"model": "codex"},
        {"model": "deepseek"},
    ]
    assert ad.paper_rows(rows) == [{"model": "sonnet"}, {"model": "codex"}]


def test_paper_sample_manifest_carries_its_own_version(tmp_path):
    project, data_dir = _manifest_project(tmp_path)
    result = _manifest_load(project, data_dir)
    manifest = ad.build_paper_sample_manifest(result.rows, 5, project_root=project)

    assert manifest["sample_version"] == ad.PAPER_SAMPLE_VERSION
    assert manifest["scope"] == "paper"
    assert manifest["models"] == list(ad.PAPER_MODELS)
    assert manifest["arm_ids"] == list(ad.PAPER_ARM_IDS)
    assert manifest["n_rows"] == len(result.rows)


def test_paper_sample_manifest_drops_rows_outside_the_paper_scope(tmp_path):
    project, data_dir = _manifest_project(tmp_path)
    result = _manifest_load(project, data_dir)
    outside = dict(result.rows[0])
    outside["model"] = "deepseek"

    manifest = ad.build_paper_sample_manifest(
        [*result.rows, outside], 5, project_root=project
    )

    assert manifest["n_rows"] == len(result.rows)


def test_corpus_manifest_is_rejected_where_the_paper_pin_is_expected(tmp_path):
    """The two pins are not interchangeable."""
    project, data_dir = _manifest_project(tmp_path)
    result = _manifest_load(project, data_dir)
    corpus = ad.build_sample_manifest(result.rows, 5, project_root=project)
    manifest_path = project / ad.PAPER_SAMPLE_MANIFEST_SUBPATH
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(corpus))

    with pytest.raises(ad.SampleManifestError, match="sample_version"):
        ad.verify_rows_against_sample_manifest(
            result.rows,
            manifest_path,
            max_reps=5,
            project_root=project,
            sample_version=ad.PAPER_SAMPLE_VERSION,
        )


def test_load_paper_runs_rejects_an_arm_outside_the_paper_scope():
    with pytest.raises(KeyError, match="outside the paper scope"):
        ad.load_paper_runs(arm_ids=["deepseek_zero_shot"])

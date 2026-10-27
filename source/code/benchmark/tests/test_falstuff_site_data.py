"""Regression tests for the manuscript-facing website data sample."""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "falstuff_site"))

import build_run_traces  # noqa: E402
import build_site_data  # noqa: E402
from main_sample import load_main_sample  # noqa: E402


def score(
    prompt_wave: str,
    *,
    model_response: bool = True,
    quarantined: bool = False,
    passed: bool = True,
) -> dict:
    criteria = {
        key: {"pass": passed}
        for key in (
            "specification_correctness",
            "executability",
            "output_correctness",
            "submission_validity",
        )
    }
    gates = {
        key: {"pass": passed}
        for key in ("execution", "artifact", "numeric", "method", "integrity")
    }
    return {
        "task_id": "ols_robust",
        "platform": "python",
        "prompt_mode": "zero_shot",
        "exec_mode": "no_feedback",
        "prompt_wave": prompt_wave,
        "model_response": model_response,
        "quarantined": quarantined,
        "task_success": passed,
        "strict_success": passed,
        "criteria": criteria,
        "gates": gates,
        "cost_usd": 0.1,
        "wall_seconds": 2.0,
        "failure_labels": ["passed"] if passed else ["numeric_mismatch"],
        "failure_criteria": ["passed"] if passed else ["output_correctness"],
    }


def test_select_analysis_wave_excludes_other_waves_and_nonanalysis_records() -> None:
    report = {
        "scores": [
            score("legacy"),
            score("pinned-2026-07", passed=False),
            score("legacy", quarantined=True),
            score("legacy", model_response=False),
        ],
        "quarantined": {"tasks": ["wcb_few"]},
    }

    selected = build_site_data.select_analysis_wave(report, "legacy")

    assert selected["n_analysis_records"] == 1
    assert selected["analysis_summary"]["task_success"] == 1
    assert selected["by_exec_mode"]["no_feedback"]["n"] == 1
    assert selected["by_task"]["ols_robust"]["n"] == 1
    assert {row["prompt_wave"] for row in selected["scores"]} == {"legacy"}


def test_select_analysis_wave_fails_when_wave_is_absent() -> None:
    report = {"scores": [score("legacy")]}

    try:
        build_site_data.select_analysis_wave(report, "missing")
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("Expected a missing analysis wave to fail")


def test_every_task_trace_file_covers_all_combinations() -> None:
    """Each model and task covers 3 platforms x 2 prompts x 3 agency x 10 reps."""
    trace_dir = Path(build_site_data.TRACE_DIR)
    files = sorted(
        path
        for model_id in build_run_traces.MODEL_CONFIG
        for path in (trace_dir / model_id).glob("*.js")
    )
    catalog, _, _ = load_main_sample()
    assert len(files) == len(catalog) * len(build_run_traces.MODEL_CONFIG)
    assert {path.stem for path in files} == set(catalog)

    expected = build_run_traces.EXPECTED_COMBINATIONS
    for path in files:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(
            text.splitlines()[-1].split(" = ", maxsplit=1)[1].rstrip(";")
        )
        assert payload["modelId"] == path.parent.name
        combos = payload["combos"]
        assert len(combos) == expected, f"{path.name}: {len(combos)} combinations"
        for key, combo in combos.items():
            platform, prompt, agency, rep = key.split("|")
            assert combo["platform"] == platform
            assert combo["prompt"] == prompt
            assert combo["agency"] == agency
            assert str(combo["rep"]) == rep
            # Every combination reports an outcome, and a missing one says so.
            assert isinstance(combo["pass"], bool)
            if combo.get("missing"):
                assert combo["note"] == build_run_traces.MISSING_NOTE
        # Complete saved answers must survive without size-driven truncation.
        for combo in combos.values():
            assert set(combo["flow"]) >= {"prompt", "process", "final", "resultText"}
            assert isinstance(combo["flow"]["process"]["events"], list)



def registry_task(
    task_id: str,
    name: str,
    group: str,
    group_order: int,
    order_in_group: int,
    display_order: int,
) -> dict:
    return {
        "task_id": task_id,
        "title": name,
        "display_name": name,
        "display_group": group,
        "display_group_order": group_order,
        "display_order_in_group": order_in_group,
        "display_order": display_order,
        "analyzed": True,
        "dataset": {"source_type": "seeded_dgp", "path": "input.csv"},
        "prompt": "Estimate the declared quantity.",
        "reference": {"key": "estimate"},
        "estimands": ["estimate"],
    }


def test_build_tasks_uses_the_canonical_display_order_and_group() -> None:
    # The registry lists tasks in harness order. The site must emit them in the
    # canonical reading order and carry the group with each row.
    registry = {
        "tasks": [
            registry_task("late", "Cox regression", "Other", 8, 1, 3),
            registry_task("excluded", "Wild cluster bootstrap", "Other", 8, 2, 2),
            registry_task("early", "Probit regression", "Binary regression", 1, 1, 1),
        ]
    }
    strict = {
        "by_task": {},
        "scores": [],
        "quarantined": {"tasks": ["excluded"]},
    }

    tasks = build_site_data.build_tasks(registry, strict)

    assert [(task["order"], task["title"], task["group"]) for task in tasks] == [
        (1, "Probit regression", "Binary regression"),
        (2, "Cox regression", "Other"),
    ]
    assert [task["groupOrder"] for task in tasks] == [1, 8]


def test_site_task_names_match_the_paper() -> None:
    """Every task carries the paper's name (task_labels.py) and a method group, in group order."""
    import task_display
    from task_labels import TASK_LABELS

    catalog, _, _ = load_main_sample()
    payload = build_site_data.build_payload()
    assert {row["id"] for row in payload["tasks"]} == set(catalog)
    for task in payload["tasks"]:
        assert task["title"] == TASK_LABELS[task["id"]]
        assert task["group"] in task_display.CURRENT_GROUP_ORDER
        assert task["group"] != "Other"
    keys = [(task["groupOrder"], task["orderInGroup"]) for task in payload["tasks"]]
    assert keys == sorted(keys)
    assert [task["order"] for task in payload["tasks"]] == list(range(1, 41))
    ppml = next(task for task in payload["tasks"] if task["id"] == "lp04_ppml")
    assert "poisson" in (ppml["title"] + " " + ppml["group"] + " " + ppml["keywords"]).lower()


def test_rates_and_steps_round_half_up_like_the_paper() -> None:
    """77.25 percent prints as 77.3, and steps come from run counts, as in numbers.tex."""
    numbers = (Path(build_site_data.REPO_ROOT) / "paper/manuscript/numbers.tex").read_text(encoding="utf-8")

    def macro(name: str) -> float:
        match = re.search(r"\\newcommand\{\\" + name + r"\}\{([-0-9.]+)\}", numbers)
        assert match, name
        return float(match.group(1))

    assert build_site_data.pct(1854, 2400) == 77.3
    payload = build_site_data.build_payload()
    by_model = {group["label"]: {row["id"]: row for row in group["interfaces"]} for group in payload["costComparisons"]}
    assert by_model["GPT-5.6 Luna"]["no_feedback"]["success"] == macro("numCodexChatbot")
    assert by_model["Claude Sonnet 4.6"]["tool_loop"]["stepSuccess"] == macro("numSonnetGainRepairToAgent")
    assert payload["analysisSample"]["taskSuccessRule"] == build_site_data.OUTCOME_TEXT["definition"]


def test_task_reference_terms_mark_the_scored_standard_errors() -> None:
    sys.path.insert(0, str(Path(build_site_data.REPO_ROOT) / "code/paper"))
    import se_scoring

    spec = se_scoring.spec()
    payload = build_site_data.build_payload()
    for task in payload["tasks"]:
        scored = {row["term"] for row in task["referenceTerms"] if row["seScored"]}
        assert scored == set(spec.get(task["id"], {})), task["id"]
        assert task["scoredStandardErrors"] == [row["term"] for row in task["referenceTerms"] if row["seScored"]]


def _trace(model: str, task: str) -> dict:
    line = (Path(build_site_data.TRACE_DIR) / model / f"{task}.js").read_text(encoding="utf-8").splitlines()[2]
    return json.loads(line[line.index("] = ") + 4:].rstrip().rstrip(";"))


def test_traces_show_why_a_standard_error_failure_fails() -> None:
    trace = _trace("sonnet", "applied_shift_share")
    assert {row["term"] for row in trace["reference"] if row["seScored"]} == {"effect", "first_stage"}
    se_only = [combo for combo in trace["combos"].values()
               if not combo.get("missing") and not combo["pass"]
               and all(term["pass"] for term in combo["terms"])]
    assert se_only, "expected runs of this task that fail only on the standard error"
    for combo in se_only:
        assert any(term.get("seScored") and term.get("sePass") is False for term in combo["terms"])


def test_agent_traces_include_the_final_script() -> None:
    for model in ("sonnet", "luna"):
        trace = _trace(model, "anova_oneway")
        agent = [combo for combo in trace["combos"].values() if combo["agency"] == "tool_loop"]
        assert len(agent) == 60
        assert all(combo["flow"]["final"]["script"] for combo in agent)
        assert all(isinstance(combo["toolCalls"], int) for combo in agent)


def test_main_selection_rejects_modified_source_before_reading(tmp_path: Path) -> None:
    (tmp_path / "snapshot.json").write_text(json.dumps({"runs_sha256": "wrong"}))
    (tmp_path / "runs.csv").write_text("position_id\nchanged\n")
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        load_main_sample(tmp_path)


def test_model_costs_match_current_priced_rows() -> None:
    from main_sample import truth
    _, rows, _ = load_main_sample()
    payload = build_site_data.build_payload()
    for model in payload["models"]:
        priced = [r for r in rows if r["model"] == model["id"] and truth(r["cost_pricing_complete"]) and r["cost_usd"]]
        total = sum(float(r["cost_usd"]) for r in priced)
        assert model["nPriced"] == len(priced)
        assert model["cost"] == round(total, 6)
        assert model["meanCost"] == round(total / len(priced), 6)


def test_task_prompt_examples_are_saved_selected_messages() -> None:
    """Each example equals the saved first prompt of a selected run; no collection path is published."""
    from main_sample import repo_path

    _catalog, rows, _summary = load_main_sample()
    first_prompts: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        prompt_file = repo_path(row["record_path"]).parent / "prompt_attempt1.txt"
        if prompt_file.is_file():
            first_prompts.setdefault((row["task"], row["prompt"], row["platform"]), set()).add(
                prompt_file.read_text(encoding="utf-8"))
    payload = build_site_data.build_payload()
    for task in payload["tasks"]:
        for condition in ("zero_shot", "few_shot"):
            assert set(task["promptExamples"][condition]) == {"python", "r", "stata"}
            for platform, example in task["promptExamples"][condition].items():
                assert "source" not in example
                saved = first_prompts.get((task["id"], condition, platform))
                if saved:
                    assert example["text"] in saved



def test_client_does_not_rename_or_renumber_tasks() -> None:
    app_path = Path(build_site_data.SITE_DIR) / "app.js"
    source = app_path.read_text(encoding="utf-8")

    assert "item.title = item.title.replace" not in source
    assert re.search(r"\bT\d{2}\b", source) is None


def test_chart_payload_does_not_publish_a_dated_snapshot() -> None:
    payload = build_site_data.chart_payload({"snapshot": "July 1, 2026"})

    assert "snapshot" not in payload


def test_client_path_redaction_does_not_publish_a_literal_account_name() -> None:
    app_path = Path(build_site_data.SITE_DIR) / "app.js"
    source = app_path.read_text(encoding="utf-8")

    assert re.search(r"/\\/Users\\/[A-Za-z0-9]", source) is None
    assert "/\\/Users\\/[^/\\s]+" in source

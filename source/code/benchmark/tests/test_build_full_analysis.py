"""Offline tests for the six-arm full-analysis builder (synthetic rows only).

No test reads ``data/``. Rows, reports, and manifests are handcrafted in the
spec RunRow shape; ``analysis_dataset.load_runs`` is monkeypatched in the
``main()`` tests so no loader touches the filesystem.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import analysis_dataset as ad  # noqa: E402
import build_full_analysis as bfa  # noqa: E402

TOP_LEVEL_KEYS = [
    "generated_by",
    "decision_record",
    "primary_outcome_decision_record",
    "scoring_version",
    "primary_outcome_version",
    "sample_rule",
    "task_success_rule",
    "gpt54_pricing",
    "manifest",
    "paper_scope",
    "headline",
    "nine_cell",
    "criteria",
    "prompt_contrasts",
    "extended",
    "all_reps_sensitivity",
    "responded_only_sensitivity",
    "strict_vs_unified",
    "notes",
]
EXTENDED_KEYS = [
    "pass_at_k",
    "majority_vote",
    "silent_failure",
    "criterion_cascade",
    "interface_gain_decomposition",
    "tolerance_sensitivity",
    "per_cell_dispersion",
    "turn_distributions",
    "cost",
    "runtime",
    "revision_disclosure",
]
ATTRITION_KEYS = (
    "dirs_seen",
    "records_parsed",
    "missing_run_record",
    "invalid_json",
    "excluded_underscore_dir",
    "excluded_non_dir",
    "excluded_retired_fewshot",
    "excluded_wrong_arm",
    "excluded_rep_over_cap",
    "excluded_timeout",
    "excluded_session_limit",
    "excluded_no_response_other",
    "excluded_collection_failure",
    "in_flight_missing_record",
    "usable",
    "turns_unavailable",
)
PRICING = {"input": 2.50, "cached_input": 0.25, "output": 15.00}


def make_row(**overrides: Any) -> dict[str, Any]:
    """A valid default annotated RunRow dict, overridable field by field.

    Mirrors a loader-annotated row: ``task_success`` is the unified primary
    outcome, ``task_success_strict`` the five-gate verdict, and ``unified``
    carries the repair detail. The synthetic default keeps the two outcomes
    aligned.
    """
    success = bool(overrides.get("task_success", True))
    row: dict[str, Any] = {
        "model": "sonnet",
        "model_id": "claude-sonnet-4-6",
        "prompt": "zero_shot",
        "interface": "no_feedback",
        "platform": "python",
        "task": "t01",
        "rep": 0,
        "task_success": success,
        "task_success_strict": success,
        "unified": {
            "repair_rule": "none needed" if success else "no_file",
            "numeric_repaired_pass": success,
            "numeric_fail_kind": "" if success else "no_artifact",
        },
        "gates": {
            "execution": True,
            "artifact": True,
            "numeric": success,
            "method": True,
            "integrity": True,
        },
        "criteria": {
            "specification_correctness": True,
            "executability": True,
            "output_correctness": success,
            "submission_validity": True,
        },
        "per_term": {},
        "cost_usd": 0.01,
        "tokens": {
            "input_tokens": 1000,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 200,
            "reasoning_output_tokens": 0,
        },
        "wall_seconds": 10.0,
        "provider_wall_seconds": 8.0,
        "turns": 3,
    }
    row.update(overrides)
    if "interface_label" not in overrides:
        row["interface_label"] = ad.INTERFACE_LABELS[row["interface"]]
    if "provenance" not in overrides:
        row["provenance"] = {
            "arm_id": f"{row['model']}_{row['prompt']}",
            "run_dir": "synthetic",
            "source_kind": "synthetic",
            "prompt_sha256": None,
            "scoring_source": "synthetic",
            "usable_reason": "ok",
        }
    return row


def make_report(arm_id: str, present: bool = True, usable: int = 0) -> ad.ArmLoadReport:
    counts = dict.fromkeys(ATTRITION_KEYS, 0)
    counts["usable"] = usable
    return ad.ArmLoadReport(
        arm_id=arm_id, present=present, counts=counts, max_created_at=None
    )


def make_manifest(statuses: dict[str, str]) -> dict[str, Any]:
    arms = {}
    for arm_id, status in statuses.items():
        arms[arm_id] = {
            "present": status != "absent",
            "status": status,
            "usable_runs": 0,
            "cells_full": 0,
            "cells_short": 0,
            "cells_empty": 0,
            "empty_cells": [],
            "short_cells": [],
            "tasks_entirely_absent": [],
            "attrition": dict.fromkeys(ATTRITION_KEYS, 0),
            "max_created_at": None,
        }
    return {
        "expected": {
            "tasks": 21,
            "platforms": 3,
            "interfaces": 3,
            "cells": 189,
            "runs_at_12_reps": 2268,
        },
        "arms": arms,
    }


def six_arm_inputs() -> tuple[list[dict[str, Any]], dict[str, ad.ArmLoadReport]]:
    """Five loaded arms (deepseek few_shot absent), one task, six reps per cell.

    zero_shot rows succeed at reps 0-3 and rep 5 (five-rep pct 80.0, all-reps
    83.33); few_shot rows always succeed. Codex rows carry no cost and no
    turns, matching the real arm.
    """
    rows: list[dict[str, Any]] = []
    for model in ad.MODELS:
        for prompt in ad.PROMPTS:
            if (model, prompt) == ("deepseek", "few_shot"):
                continue
            for interface in ad.INTERFACES:
                for platform in ad.PLATFORMS:
                    for rep in range(6):
                        success = prompt == "few_shot" or rep < 4 or rep == 5
                        rows.append(
                            make_row(
                                model=model,
                                prompt=prompt,
                                interface=interface,
                                platform=platform,
                                rep=rep,
                                task_success=success,
                                cost_usd=None if model == "codex" else 0.01,
                                turns=None if model == "codex" else 3,
                            )
                        )
    reports = {
        arm_id: make_report(arm_id, present=arm_id != "deepseek_few_shot")
        for arm_id in ad.ARMS
    }
    return rows, reports


def six_arm_statuses() -> dict[str, str]:
    return {
        arm_id: "absent" if arm_id == "deepseek_few_shot" else "complete"
        for arm_id in ad.ARMS
    }


def test_paper_scope_block_pools_only_the_paper_arms() -> None:
    """The snapshot keeps three arms; paper_scope pools the two paper arms."""
    rows, reports = six_arm_inputs()
    manifest = make_manifest(six_arm_statuses())
    payload = bfa.build_payload(rows, reports, manifest, 5, PRICING)

    scope = payload["paper_scope"]
    assert scope["models"] == list(ad.PAPER_MODELS)
    assert scope["sample_version"] == ad.PAPER_SAMPLE_VERSION
    assert set(payload["headline"]) == set(ad.MODELS)

    # Two arms x two prompt conditions x 45 five-rep rows each.
    assert scope["n"] == 4 * 45
    assert scope["pooled"]["overall"]["n"] == 4 * 45
    pooled_successes = sum(
        payload["headline"][model][prompt]["overall"]["successes"]
        for model in ad.PAPER_MODELS
        for prompt in ad.PROMPTS
    )
    assert scope["pooled"]["overall"]["successes"] == pooled_successes
    for model in ad.PAPER_MODELS:
        assert (
            scope["by_model"][model]["overall"]["overall"]
            == bfa.success_block(
                [
                    row
                    for row in ad.select_first_eligible_rows(rows, 5)
                    if row["model"] == model
                ]
            )
        )


def test_build_payload_structure_and_headline() -> None:
    rows, reports = six_arm_inputs()
    manifest = make_manifest(six_arm_statuses())
    payload = bfa.build_payload(rows, reports, manifest, 5, PRICING)

    assert list(payload.keys()) == TOP_LEVEL_KEYS
    assert list(payload["extended"].keys()) == EXTENDED_KEYS
    decomposition = payload["extended"]["interface_gain_decomposition"]
    assert set(decomposition) == {"chatbot_to_agent", "chatbot_to_one_repair"}
    assert payload["manifest"] is manifest
    assert payload["gpt54_pricing"]["prices_usd_per_mtok"] == PRICING
    assert isinstance(payload["gpt54_pricing"]["source"], dict)

    sonnet_zero = payload["headline"]["sonnet"]["zero_shot"]
    assert sonnet_zero["overall"] == {"n": 45, "successes": 36, "success_pct": 80.0}
    assert sonnet_zero["by_interface"]["chatbot"] == {
        "n": 15,
        "successes": 12,
        "success_pct": 80.0,
    }
    assert sonnet_zero["by_platform"]["stata"]["success_pct"] == 80.0
    assert payload["headline"]["sonnet"]["few_shot"]["overall"] == {
        "n": 45,
        "successes": 45,
        "success_pct": 100.0,
    }

    all_reps = payload["all_reps_sensitivity"]
    assert all_reps["headline"]["sonnet"]["zero_shot"]["overall"] == {
        "n": 54,
        "successes": 45,
        "success_pct": 83.33,
    }
    assert all_reps["n_by_arm"]["sonnet_zero_shot"] == 54
    assert all_reps["n_by_arm"]["deepseek_few_shot"] == 0

    nine = payload["nine_cell"]["sonnet"]["zero_shot"]
    assert len(nine) == 9
    assert nine[0]["platform"] == ad.PLATFORMS[0]
    assert nine[0]["interface"] == ad.INTERFACES[0]
    assert all(cell["n"] == 5 and cell["success_pct"] == 80.0 for cell in nine)
    assert all(cell["median_cost_usd"] == 0.01 for cell in nine)
    assert all(cell["n_priced"] == cell["n"] for cell in nine)
    codex_nine = payload["nine_cell"]["codex"]["zero_shot"]
    assert all(cell["median_cost_usd"] is None for cell in codex_nine)
    assert all(cell["mean_cost_usd"] is None for cell in codex_nine)
    assert all(cell["median_imputed_cost_usd"] == 0.0055 for cell in codex_nine)
    assert all(cell["mean_imputed_cost_usd"] == 0.0055 for cell in codex_nine)

    criteria = payload["criteria"]["sonnet"]["zero_shot"]["chatbot"]
    assert criteria["n"] == 15
    assert criteria["output_correctness_pct"] == 80.0
    assert criteria["executability_pct"] == 100.0
    assert criteria["task_success_pct"] == 80.0
    assert criteria["task_success_strict_pct"] == 80.0

    comparison = payload["strict_vs_unified"]
    assert comparison["overall"]["task_success_pct"] == (
        comparison["overall"]["task_success_strict_pct"]
    )
    assert "boundary_vetoed" not in comparison["flip_mechanism"]

    sonnet_contrast = payload["prompt_contrasts"]["sonnet"]
    assert sonnet_contrast["status"] == "ok"
    assert sonnet_contrast["unrestricted"]["overall_pp"] == 20.0
    assert sonnet_contrast["shared_cells_only"]["n_shared_cells"] == 9
    assert sonnet_contrast["shared_cells_only"]["overall_pp"] == 20.0


def test_pricing_none_reported_as_null() -> None:
    rows, reports = six_arm_inputs()
    manifest = make_manifest(six_arm_statuses())
    payload = bfa.build_payload(rows, reports, manifest, 5, None)
    assert payload["gpt54_pricing"]["prices_usd_per_mtok"] is None


def test_absent_arm_reported_not_raised() -> None:
    rows, reports = six_arm_inputs()
    manifest = make_manifest(six_arm_statuses())
    payload = bfa.build_payload(rows, reports, manifest, 5, PRICING)

    assert payload["headline"]["deepseek"]["few_shot"] == {"status": "absent"}
    assert payload["nine_cell"]["deepseek"]["few_shot"] == {"status": "absent"}
    assert payload["criteria"]["deepseek"]["few_shot"] == {"status": "absent"}
    contrast = payload["prompt_contrasts"]["deepseek"]
    assert contrast["status"] == "few_shot_absent"
    assert contrast["zero_shot_only"] is True
    assert "unrestricted" not in contrast


def partial_codex_inputs() -> tuple[
    list[dict[str, Any]], dict[str, ad.ArmLoadReport], dict[str, Any]
]:
    """codex zero_shot covers tasks t01+t02; codex few_shot covers only t01.

    zero_shot succeeds on t01 and fails on t02, so the unrestricted contrast
    (+50.0 pp) differs from the shared-cells contrast (0.0 pp).
    """
    rows: list[dict[str, Any]] = []
    for prompt, tasks in (("zero_shot", ("t01", "t02")), ("few_shot", ("t01",))):
        for task in tasks:
            for interface in ad.INTERFACES:
                for platform in ad.PLATFORMS:
                    for rep in range(5):
                        rows.append(
                            make_row(
                                model="codex",
                                model_id="gpt-5.4",
                                prompt=prompt,
                                task=task,
                                interface=interface,
                                platform=platform,
                                rep=rep,
                                task_success=task == "t01",
                                cost_usd=None,
                                turns=None,
                            )
                        )
    reports = {
        "codex_zero_shot": make_report("codex_zero_shot", usable=90),
        "codex_few_shot": make_report("codex_few_shot", usable=45),
    }
    manifest = make_manifest(
        {"codex_zero_shot": "complete", "codex_few_shot": "partial"}
    )
    return rows, reports, manifest


def test_partial_arm_shared_cells_contrast() -> None:
    rows, reports, manifest = partial_codex_inputs()
    payload = bfa.build_payload(rows, reports, manifest, 5, PRICING)

    assert list(payload["headline"].keys()) == ["codex"]
    contrast = payload["prompt_contrasts"]["codex"]
    assert contrast["unrestricted"]["zero_shot_success_pct"] == 50.0
    assert contrast["unrestricted"]["few_shot_success_pct"] == 100.0
    assert contrast["unrestricted"]["overall_pp"] == 50.0
    shared = contrast["shared_cells_only"]
    assert shared["n_shared_cells"] == 9
    assert shared["overall_pp"] == 0.0
    assert shared["by_interface_pp"]["chatbot"] == 0.0
    assert "few_shot_partial (3 of " in contrast["status"]
    assert "task x platform cells" in contrast["status"]
    assert "zero_shot" not in contrast["status"]


def test_annotate_attrition_basis_reconciles_samples() -> None:
    rows = [make_row(rep=rep) for rep in range(7)]  # reps 5 and 6 over the cap
    manifest = make_manifest({"sonnet_zero_shot": "complete"})
    bfa.annotate_attrition_basis(manifest, rows, 5)
    arm = manifest["arms"]["sonnet_zero_shot"]
    assert arm["attrition_basis"] == "all-reps snapshot load"
    assert arm["usable_over_rep_cap"] == 2


def test_runtime_block_separates_provider_and_total_wall() -> None:
    rows = [
        make_row(interface=interface, provider_wall_seconds=seconds)
        for interface, seconds in (
            ("no_feedback", 10.0),
            ("no_feedback", 20.0),
            ("tool_loop", 40.0),
            ("tool_loop", 60.0),
        )
    ]
    block = bfa._runtime_section(rows)

    assert block["measure"] == "provider_wall_seconds"
    assert "total_wall_seconds" in block["note"]
    assert block["pooled"]["n"] == 4
    assert block["pooled"]["median_provider_wall_seconds"] == 30.0
    assert block["pooled"]["mean_provider_wall_seconds"] == 32.5
    # wall_seconds keeps its own definition and stays reported separately.
    assert block["pooled"]["median_wall_seconds"] == 10.0

    by_interface = block["models"]["sonnet"]["by_interface"]
    assert by_interface["chatbot"]["median_provider_wall_seconds"] == 15.0
    assert by_interface["constrained agent"]["median_provider_wall_seconds"] == 50.0
    assert by_interface["one-repair"]["median_provider_wall_seconds"] is None
    assert by_interface["one-repair"]["n"] == 0


def test_runtime_block_totals_provider_hours_per_model() -> None:
    rows = [
        make_row(model="sonnet", provider_wall_seconds=3600.0),
        make_row(model="codex", provider_wall_seconds=1800.0, cost_usd=None),
    ]
    block = bfa._runtime_section(rows)

    assert block["total_provider_wall_hours"] == 1.5
    assert block["total_provider_wall_hours_by_model"] == {
        "sonnet": 1.0,
        "codex": 0.5,
    }
    assert list(block["models"]) == ["sonnet", "codex"]


def test_runtime_block_tolerates_missing_provider_wall() -> None:
    rows = [
        make_row(provider_wall_seconds=None),
        make_row(provider_wall_seconds=12.0),
    ]
    block = bfa._runtime_section(rows)

    assert block["pooled"]["n"] == 2
    assert block["pooled"]["n_with_provider_wall"] == 1
    assert block["pooled"]["median_provider_wall_seconds"] == 12.0


def test_runtime_block_covers_every_software() -> None:
    rows, _ = six_arm_inputs()
    block = bfa._runtime_section(rows)

    for model in ad.MODELS:
        assert set(block["models"][model]["by_platform"]) == set(ad.PLATFORMS)
        assert set(block["models"][model]["by_interface"]) == set(
            ad.INTERFACE_LABELS.values()
        )
    assert set(block["pooled_by_platform"]) == set(ad.PLATFORMS)


def test_build_payload_deterministic_bytes() -> None:
    first = bfa.build_payload(
        *six_arm_inputs(), make_manifest(six_arm_statuses()), 5, PRICING
    )
    second = bfa.build_payload(
        *six_arm_inputs(), make_manifest(six_arm_statuses()), 5, PRICING
    )
    assert json.dumps(first, indent=2) == json.dumps(second, indent=2)


def test_build_payload_uses_later_eligible_rep_to_balance_cell() -> None:
    rows, reports = six_arm_inputs()
    target = {
        "model": "sonnet",
        "prompt": "zero_shot",
        "interface": "no_feedback",
        "platform": "python",
    }
    rows = [
        row
        for row in rows
        if not (
            all(row[key] == value for key, value in target.items()) and row["rep"] == 2
        )
    ]
    for row in rows:
        if all(row[key] == value for key, value in target.items()) and row["rep"] == 5:
            row["task_success"] = False
            row["task_success_strict"] = False
            row["unified"] = {
                "repair_rule": "no_file",
                "numeric_repaired_pass": False,
                "numeric_fail_kind": "no_artifact",
            }
            row["gates"]["numeric"] = False
            row["criteria"]["output_correctness"] = False

    payload = bfa.build_payload(
        rows, reports, make_manifest(six_arm_statuses()), 5, PRICING
    )

    zero = payload["headline"]["sonnet"]["zero_shot"]["overall"]
    assert zero["n"] == 45
    assert zero["successes"] == 35


def test_main_single_load_and_in_memory_balanced_selector(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    rows, reports = six_arm_inputs()
    calls: list[dict[str, Any]] = []

    def fake_load_runs(
        arm_ids: Any = None, max_reps: Any = 5, project_root: Any = None, **_: Any
    ) -> ad.LoadResult:
        calls.append(
            {"arm_ids": arm_ids, "max_reps": max_reps, "project_root": project_root}
        )
        return ad.LoadResult(rows=rows, reports=reports, manifest={})

    monkeypatch.setattr(ad, "load_runs", fake_load_runs)
    out_dir = tmp_path / "out"
    # The fixture holds six reps per cell, so a five-rep cap exercises the
    # in-memory balanced selector and the over-cap reconciliation.
    bfa.main(["--out-dir", str(out_dir), "--max-reps", "5"])

    assert len(calls) == 1
    assert calls[0]["max_reps"] is None
    payload = json.loads((out_dir / "full_analysis.json").read_text())
    # Attrition counters are labeled with their all-reps basis and carry the
    # reconciliation count (9 usable rows at rep 5 per loaded arm here).
    arm = payload["manifest"]["arms"]["sonnet_zero_shot"]
    assert arm["attrition_basis"] == "all-reps snapshot load"
    assert arm["usable_over_rep_cap"] == 9
    absent = payload["manifest"]["arms"]["deepseek_few_shot"]
    assert absent["usable_over_rep_cap"] == 0
    assert payload["headline"]["sonnet"]["zero_shot"]["overall"]["n"] == 45
    assert (
        payload["all_reps_sensitivity"]["headline"]["sonnet"]["zero_shot"]["overall"][
            "n"
        ]
        == 54
    )
    out = capsys.readouterr().out
    assert f"wrote {out_dir / 'full_analysis.json'}" in out


def test_cli_project_root_forwarded_and_summary_printed(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    rows, reports = six_arm_inputs()
    seen: dict[str, Any] = {}

    def fake_load_runs(
        arm_ids: Any = None, max_reps: Any = 5, project_root: Any = None, **_: Any
    ) -> ad.LoadResult:
        seen["arm_ids"] = arm_ids
        seen["project_root"] = project_root
        return ad.LoadResult(rows=rows, reports=reports, manifest={})

    monkeypatch.setattr(ad, "load_runs", fake_load_runs)
    root = tmp_path / "proj"
    root.mkdir()
    bfa.main(
        ["--project-root", str(root), "--arms", "sonnet_zero_shot", "sonnet_few_shot"]
    )

    assert seen["arm_ids"] == ["sonnet_zero_shot", "sonnet_few_shot"]
    assert seen["project_root"] == root.resolve()
    out_path = (
        root / "data" / "analysis" / "full_analysis.json"
    )
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert list(payload.keys()) == TOP_LEVEL_KEYS
    out = capsys.readouterr().out
    assert "sonnet_zero_shot" in out
    assert "gpt-5.4 pricing" in out
    assert f"wrote {out_path}" in out

"""Offline tests for factorial_fiverep and build_manuscript_numbers.

Synthetic rows only; no data/ reads, no Stata. Covers the design-cell frame
mapping, the saturation identity (balanced data: task-FE contrasts equal raw
cell-mean contrasts), and the difficulty-thirds ranking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import statsmodels.formula.api as smf

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import analysis_dataset as ad  # noqa: E402
import build_manuscript_numbers as bmn  # noqa: E402
import factorial_analysis as fa  # noqa: E402
import factorial_fiverep as ff  # noqa: E402


def make_row(**overrides) -> dict:
    row = {
        "model": "sonnet",
        "prompt": "zero_shot",
        "interface": "no_feedback",
        "interface_label": "chatbot",
        "platform": "python",
        "task": ad.ANALYSIS_TASK_IDS[0],
        "rep": 0,
        "task_success": True,
        "cost_usd": 0.1,
        "turns": 1,
    }
    row.update(overrides)
    return row


def balanced_rows(shift: int = 0) -> list[dict]:
    """Three tasks x 18 cells x 4 reps with a deterministic success pattern."""
    rows = []
    for t_idx, task in enumerate(("task_a", "task_b", "task_c")):
        for platform in fa.PLATFORMS:
            for prompt in fa.PROMPTS:
                for interface in fa.INTERFACES:
                    cell = fa.cell_index(platform, prompt, interface)
                    n_success = (cell + t_idx + shift) % 5
                    for rep in range(4):
                        rows.append(
                            make_row(
                                task=task,
                                platform=platform,
                                prompt=prompt,
                                interface=interface,
                                task_success=rep < n_success,
                            )
                        )
    return rows


def test_rows_to_frame_maps_cells() -> None:
    frame = ff.rows_to_frame(
        [
            make_row(),
            make_row(platform="stata", prompt="few_shot", interface="tool_loop"),
        ]
    )
    assert list(frame["cell"]) == [
        fa.cell_index("python", "zero_shot", "no_feedback"),
        fa.cell_index("stata", "few_shot", "tool_loop"),
    ]
    assert frame["cell"].iloc[0] == 1
    assert frame["success"].tolist() == [1, 1]


def test_model_arms_cover_the_three_scored_models() -> None:
    assert ff.MODEL_ARMS == {
        "sonnet": ("sonnet_zero_shot", "sonnet_few_shot"),
        "codex": ("codex_zero_shot", "codex_few_shot"),
        "deepseek": ("deepseek_zero_shot", "deepseek_few_shot"),
    }


def test_paper_scope_is_the_two_paper_arms() -> None:
    assert ff.PAPER_MODELS == ("sonnet", "codex")
    assert ff.CROSS_MODEL_PAIRS == (("sonnet", "codex"),)
    assert ad.PAPER_ARM_IDS == (
        "sonnet_zero_shot",
        "sonnet_few_shot",
        "codex_zero_shot",
        "codex_few_shot",
    )
    assert ad.EXPECTED_PAPER_RUNS_12REP == 9072


def test_saturation_holds_exactly_on_balanced_data() -> None:
    frame = ff.rows_to_frame(balanced_rows())
    weights = fa.contrast_weights()
    contrasts = fa.contrast_table(fa.fit_lpm(frame), weights)
    check = ff.saturation_check(frame, weights, contrasts)
    assert check["no_fe_max_abs_gap"] < ff.NO_FE_TOL
    # Balanced tasks-within-cells: the task-FE contrasts equal the raw
    # cell-mean contrasts (rounded to the reported 6 decimals).
    assert all(gap == 0.0 for gap in check["fe_minus_cell_mean"].values())


def test_cell_mean_contrasts_raise_on_missing_cell() -> None:
    frame = ff.rows_to_frame([make_row()])
    with pytest.raises(AssertionError, match="design cells with no runs"):
        ff.cell_mean_contrasts(frame, fa.contrast_weights())


def test_joint_tests_cover_h1_and_h4() -> None:
    assert ff.JOINT_TESTS == {
        "H1_joint": ("H1_r", "H1_stata"),
        "H4_joint": ("H4_self_repair", "H4_tool_loop"),
    }


def paper_frames_and_payloads() -> tuple[dict[str, object], dict[str, dict]]:
    weights = fa.contrast_weights()
    frames: dict[str, object] = {}
    payloads: dict[str, dict] = {}
    for shift, model in enumerate(ff.PAPER_MODELS):
        frame = ff.rows_to_frame(balanced_rows(shift=shift))
        frames[model] = frame
        payloads[model] = {
            "contrasts": fa.contrast_table(fa.fit_lpm(frame), weights)
        }
    return frames, payloads


def test_pooled_contrast_is_the_mean_of_the_arm_contrasts() -> None:
    """The pooled fit reproduces the mean of the two arms on balanced data."""
    frames, payloads = paper_frames_and_payloads()
    pool = ff.pooled_frame(frames)
    assert set(pool["arm"]) == set(ff.PAPER_MODELS)
    assert pool["armtask"].nunique() == 2 * pool["task"].nunique()

    weights = fa.contrast_weights()
    fit = smf.ols(
        "success ~ C(armtask) + C(cell, Treatment(1))", data=pool
    ).fit(cov_type="cluster", cov_kwds={"groups": pool["task"]})
    contrasts = fa.contrast_table(fit, weights)
    check = ff.pooled_mean_check(contrasts, payloads)

    assert check["max_abs_gap"] < ff.POOLED_MEAN_TOL
    assert check["arms"] == list(ff.PAPER_MODELS)


def test_pooled_mean_check_rejects_a_shifted_arm() -> None:
    frames, payloads = paper_frames_and_payloads()
    pool = ff.pooled_frame(frames)
    weights = fa.contrast_weights()
    fit = smf.ols(
        "success ~ C(armtask) + C(cell, Treatment(1))", data=pool
    ).fit(cov_type="cluster", cov_kwds={"groups": pool["task"]})
    contrasts = fa.contrast_table(fit, weights)
    payloads["codex"]["contrasts"]["H2_fewshot"]["estimate"] += 0.05

    with pytest.raises(RuntimeError, match="not the mean of the per-arm"):
        ff.pooled_mean_check(contrasts, payloads)


def test_software_gap_contrasts_cover_the_three_agency_conditions() -> None:
    weights = fa.contrast_weights()
    for name in ("H1_stata", "GAP_stata_repair_zs", "GAP_stata_agent_zs"):
        assert name in weights
        assert abs(weights[name].sum()) < 1e-12
    repair = weights["GAP_stata_repair_zs"]
    assert repair[fa.cell_index("stata", "zero_shot", "self_repair") - 1] == 1.0
    assert repair[fa.cell_index("python", "zero_shot", "self_repair") - 1] == -1.0


def test_cross_model_differences_match_per_arm_contrasts_on_balanced_data() -> None:
    weights = fa.contrast_weights()
    frames: dict[str, object] = {}
    payloads: dict[str, dict] = {}
    for shift, model in enumerate(ff.PAPER_MODELS):
        frame = ff.rows_to_frame(balanced_rows(shift=shift))
        frames[model] = frame
        contrasts = fa.contrast_table(fa.fit_lpm(frame), weights)
        payloads[model] = {"contrasts": contrasts}

    out = ff.cross_model_contrast_differences(frames, payloads)

    assert out["n_analysis"] == 2 * 3 * 18 * 4
    assert out["n_clusters"] == 3
    assert out["df"] == 2
    for name in ff.CROSS_MODEL_CONTRASTS:
        for a, b in ff.CROSS_MODEL_PAIRS:
            d = out["differences"][name][f"{a}_minus_{b}"]
            expected = (
                payloads[a]["contrasts"][name]["estimate"]
                - payloads[b]["contrasts"][name]["estimate"]
            )
            assert abs(d["estimate"] - expected) < 1e-9, (name, a, b)
            assert d["se_cr1"] > 0.0
            assert 0.0 <= d["p_t20"] <= 1.0


def test_cross_model_differences_reject_a_contrast_mismatch() -> None:
    frames, payloads = paper_frames_and_payloads()
    payloads["codex"]["contrasts"]["H4_tool_loop"]["estimate"] += 0.01

    with pytest.raises(RuntimeError, match="does not reproduce codex H4_tool_loop"):
        ff.cross_model_contrast_differences(frames, payloads)


def test_difficulty_thirds_ranks_and_computes_gains() -> None:
    tasks = ad.ANALYSIS_TASK_IDS
    rate_by_task = {task: (100, 50, 0)[idx // 7] for idx, task in enumerate(tasks)}
    rows = []
    for task in tasks:
        for rep in range(2):
            rows.append(
                make_row(
                    task=task,
                    interface="no_feedback",
                    interface_label="chatbot",
                    task_success=(rate_by_task[task] > rep * 50),
                )
            )
            rows.append(
                make_row(
                    task=task,
                    interface="tool_loop",
                    interface_label="constrained agent",
                    task_success=True,
                )
            )
    thirds = bmn.difficulty_thirds(rows)
    assert thirds["easiest"]["tasks"] == list(tasks[:7])
    assert thirds["hardest"]["tasks"] == list(tasks[14:])
    assert thirds["easiest"]["zero_shot"]["mean_task_gain_pp"] == 0.0
    assert thirds["middle"]["zero_shot"]["mean_task_gain_pp"] == 50.0
    assert thirds["hardest"]["zero_shot"]["mean_task_gain_pp"] == 100.0
    assert thirds["hardest"]["zero_shot"]["pooled_gain_pp"] == 100.0
    assert thirds["easiest"]["zero_shot"]["chatbot"]["success_pct"] == 100.0


def test_per_task_model_platform_preserves_empty_cells() -> None:
    rows = [
        make_row(
            model="sonnet",
            prompt="zero_shot",
            task=ad.ANALYSIS_TASK_IDS[0],
            platform="python",
            interface_label="chatbot",
            task_success=True,
        )
    ]
    payload = bmn.per_task_model_platform_section(rows)
    assert payload["sonnet"]["zero_shot"][ad.ANALYSIS_TASK_IDS[0]]["python"][
        "chatbot"
    ] == {"n": 1, "successes": 1, "success_pct": 100.0}
    assert payload["codex"]["few_shot"][ad.ANALYSIS_TASK_IDS[0]]["stata"][
        "constrained agent"
    ] == {"n": 0, "successes": 0, "success_pct": 0.0}

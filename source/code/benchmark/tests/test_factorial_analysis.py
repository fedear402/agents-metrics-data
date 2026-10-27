"""Offline tests for the factorial-analysis machinery (no Stata, no model calls).

The wild-cluster-bootstrap p-values require Stata ``boottest`` and are not
exercised here; the tests cover the deterministic layers those p-values sit
on: sample filtering, cell indexing, contrast vectors, the statsmodels fit,
and the Stata expression strings (with their fixed bootstrap configuration).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import factorial_analysis as fa  # noqa: E402


def score_record(
    task_id: str = "ols_robust",
    platform: str = "python",
    prompt_mode: str = "zero_shot",
    exec_mode: str = "no_feedback",
    *,
    task_success: bool = True,
    model_response: bool = True,
    quarantined: bool = False,
    prompt_wave: str = "legacy",
) -> dict:
    return {
        "task_id": task_id,
        "platform": platform,
        "prompt_mode": prompt_mode,
        "exec_mode": exec_mode,
        "task_success": task_success,
        "model_response": model_response,
        "quarantined": quarantined,
        "prompt_wave": prompt_wave,
    }


def write_report(tmp_path: Path, scores: list[dict]) -> Path:
    path = tmp_path / "task_success_report.json"
    path.write_text(json.dumps({"scores": scores}))
    return path


def test_zero_shot_cell_index_is_a_bijection_over_nine_cells() -> None:
    cells = [
        fa.cell_index_zero_shot(platform, interface)
        for platform in fa.PLATFORMS
        for interface in fa.INTERFACES
    ]
    assert sorted(cells) == list(range(1, 10))
    assert fa.cell_index_zero_shot("python", "no_feedback") == 1


def test_zero_shot_weights_sum_to_zero_and_target_the_right_cells() -> None:
    weights = fa.contrast_weights_zero_shot()
    assert set(weights) == {"H1_r", "H1_stata", "H3_self_repair", "H3_tool_loop"}
    for name, v in weights.items():
        assert v.shape == (9,), name
        assert abs(v.sum()) < 1e-12, name

    baseline = fa.cell_index_zero_shot("python", "no_feedback") - 1
    for platform in ("r", "stata"):
        v = weights[f"H1_{platform}"]
        assert v[fa.cell_index_zero_shot(platform, "no_feedback") - 1] == 1.0
        assert v[baseline] == -1.0
        assert np.count_nonzero(v) == 2

    for interface in ("self_repair", "tool_loop"):
        v = weights[f"H3_{interface}"]
        for platform in fa.PLATFORMS:
            assert v[fa.cell_index_zero_shot(platform, interface) - 1] == 1.0 / 3.0
            assert v[fa.cell_index_zero_shot(platform, "no_feedback") - 1] == -1.0 / 3.0
        assert np.count_nonzero(v) == 6


def test_zero_shot_frame_keeps_only_legacy_zero_shot_analysis_records(tmp_path) -> None:
    scores = [
        score_record(),  # kept
        score_record(platform="stata", exec_mode="tool_loop"),  # kept
        score_record(prompt_mode="few_shot"),  # dropped: few-shot
        score_record(model_response=False),  # dropped: no model response
        score_record(quarantined=True),  # dropped: quarantined
        score_record(prompt_wave="pinned-2026-07"),  # dropped: repopulation wave
    ]
    frame = fa.load_analysis_frame(write_report(tmp_path, scores), zero_shot_only=True)
    assert len(frame) == 2
    assert set(frame["prompt"]) == {"zero_shot"}
    assert set(frame["cell"]) == {
        fa.cell_index_zero_shot("python", "no_feedback"),
        fa.cell_index_zero_shot("stata", "tool_loop"),
    }


def test_default_frame_excludes_pinned_wave_records(tmp_path) -> None:
    scores = [
        score_record(),
        score_record(prompt_mode="few_shot"),  # kept in the 18-cell design
        score_record(prompt_wave="pinned-2026-07"),
    ]
    frame = fa.load_analysis_frame(write_report(tmp_path, scores))
    assert len(frame) == 2
    assert set(frame["prompt"]) == {"zero_shot", "few_shot"}


def synthetic_zero_shot_frame() -> pd.DataFrame:
    """Balanced 3-task x 9-cell frame with deterministic per-cell success counts.

    In a balanced panel the task fixed effects are orthogonal to the cell
    dummies, so every equal-weight cell contrast estimated by the LPM equals
    the same contrast applied to raw cell success means.
    """
    reps = 4
    rows = []
    for t, task in enumerate(("task_a", "task_b", "task_c")):
        for platform in fa.PLATFORMS:
            for interface in fa.INTERFACES:
                cell = fa.cell_index_zero_shot(platform, interface)
                n_success = (cell + t) % (reps + 1)  # deterministic 0..4
                for rep in range(reps):
                    rows.append(
                        {
                            "success": int(rep < n_success),
                            "task": task,
                            "platform": platform,
                            "prompt": "zero_shot",
                            "interface": interface,
                            "cell": cell,
                        }
                    )
    return pd.DataFrame(rows)


def test_lpm_contrasts_match_cell_mean_contrasts_on_balanced_data() -> None:
    frame = synthetic_zero_shot_frame()
    weights = fa.contrast_weights_zero_shot()
    result = fa.fit_lpm(frame)
    table = fa.contrast_table(result, weights)

    cell_means = frame.groupby("cell")["success"].mean()
    for name, v in weights.items():
        direct = sum(v[k - 1] * cell_means[k] for k in range(1, 10))
        assert abs(table[name]["estimate"] - direct) < 1e-10, name
        assert table[name]["se_cr1"] > 0.0, name


def test_cell_summary_covers_the_nine_cells_in_order() -> None:
    frame = synthetic_zero_shot_frame()
    summary = fa.cell_summary(frame)
    assert [row["cell"] for row in summary] == list(range(1, 10))
    assert all(row["n"] == 12 for row in summary)
    assert summary[0]["platform"] == "python"
    assert summary[0]["interface"] == "no_feedback"


def test_stata_expressions_omit_base_cell_and_leading_plus() -> None:
    weights = fa.contrast_weights_zero_shot()
    h1_r = fa.stata_expression(weights["H1_r"])
    assert h1_r == "1.000000000000000*4.cell"  # base cell 1 normalized away

    h3 = fa.stata_expression(weights["H3_self_repair"])
    assert not h3.startswith("+")
    assert "1.cell" not in h3
    for k in (2, 5, 8):  # self-repair cells enter positively
        assert f"0.333333333333333*{k}.cell" in h3
    for k in (4, 7):  # chatbot cells enter negatively
        assert f"- 0.333333333333333*{k}.cell" in h3


def test_bootstrap_configuration_is_pinned() -> None:
    assert fa.WCB_REPS == 9_999
    assert fa.WCB_SEED == 20260701
    assert fa.ZERO_SHOT_OUT_PATH.name == "factorial_analysis_zero_shot.json"


def test_full_design_weights_include_software_gap_diagnostics() -> None:
    weights = fa.contrast_weights()
    v = weights["GAP_stata_agent_zs"]
    assert v[fa.cell_index("stata", "zero_shot", "tool_loop") - 1] == 1.0
    assert v[fa.cell_index("python", "zero_shot", "tool_loop") - 1] == -1.0
    assert np.count_nonzero(v) == 2

    v = weights["GAP_stata_avg"]
    assert abs(v.sum()) < 1e-12
    for prompt in fa.PROMPTS:
        for interface in fa.INTERFACES:
            assert v[fa.cell_index("stata", prompt, interface) - 1] == 1.0 / 6.0
            assert v[fa.cell_index("python", prompt, interface) - 1] == -1.0 / 6.0
    assert np.count_nonzero(v) == 12


def test_finite_sample_p_counts_the_observed_draw() -> None:
    # boottest's r(p) = m / 9999 maps to (m + 1) / 10000, so zero never prints.
    assert fa.finite_sample_p(0.0) == 1 / 10_000
    assert fa.finite_sample_p(842 / 9_999) == 843 / 10_000
    assert fa.finite_sample_p(1.0) == 1.0
    with pytest.raises(ValueError, match="not a multiple"):
        fa.finite_sample_p(0.00012)


def test_default_joint_tests_cover_h1() -> None:
    assert fa.DEFAULT_JOINT_TESTS == {"H1_joint": ("H1_r", "H1_stata")}

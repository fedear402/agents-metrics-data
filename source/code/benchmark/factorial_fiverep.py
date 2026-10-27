"""Factorial inference for the twelve-rep 2x3x3 analysis samples.

Loads the balanced twelve-rep sample (the first twelve eligible repetitions in
each cell, both prompt arms) through ``analysis_dataset.load_runs`` and
estimates the saturated linear probability model with task fixed effects on
the 2x3x3 design (prompt x interface x platform; omitted cell = Python,
chatbot, zero-shot). The reported contrasts reuse the equal-cell-weight
machinery in ``factorial_analysis``:

* H1a/H1b: R - Python and Stata - Python at the chatbot zero-shot baseline,
  plus the H1 joint test;
* H2: few-shot - zero-shot, design-averaged over the 9 platform x interface
  cells;
* H3a/H3b: self-repair - chatbot and constrained agent - chatbot,
  design-averaged over the 6 platform x prompt cells;
* H4: prompt x interface interactions, averaged over the 3 platforms,
  plus the joint wild-bootstrap test that both interactions are zero;
* software-gap diagnostics: Stata - Python at the constrained-agent
  zero-shot cell, and Stata - Python design-averaged over the six
  prompt x agency cells.

Inference: CR1 cluster-by-task standard errors (21 clusters) from statsmodels
plus wild cluster bootstrap p-values (Rademacher, 9,999 replications, null
imposed, seed 20260701) from Stata ``boottest`` on the identical regression;
stored p-values use the finite-sample convention (m + 1) / (reps + 1), see
``factorial_analysis.finite_sample_p``. The script asserts Stata/statsmodels
agreement on every contrast estimate.

Scope. The paper's main specification covers two model arms, Claude Sonnet 4.6
and GPT-5.4 through Codex (decision record
2026-08-21_two-model-main-specification). ``--model paper`` is the default: it
estimates each paper arm, estimates the two-model pooled specification, and
writes ``factorial_by_model.json`` with the keys ``sonnet``, ``codex``,
``pooled``, and ``cross_model``. The paper sample is pinned by
``data/analysis/paper_sample_manifest.json`` and verified on every build.

Pooled specification: one saturated LPM over both paper arms with arm x task
fixed effects and 18 common cell dummies, CR1 clustered on the 21 shared
tasks, and the same wild cluster bootstrap. Because every arm x task x cell
holds twelve runs, the pooled contrast equals the unweighted mean of the two
per-arm contrasts; the script asserts that identity and reports the gap.

``--model all`` keeps the three-arm diagnostic run and writes
``factorial_by_model_all.json``, which the paper does not read.

The ``cross_model`` block fits the arm-specific saturated model (model x task
fixed effects plus model x cell dummies, CR1 clustered on the 21 shared tasks)
and reports the pairwise cross-arm differences of every contrast with t(20)
p-values.

Saturation check: contrasts from the saturated no-fixed-effects model must
equal the raw cell-mean contrasts exactly; the task-FE contrasts are compared
against the same cell-mean contrasts and the gap (attrition imbalance across
tasks within cells) is reported in the output.

Offline except for the local Stata batch call. Writes
``data/analysis/factorial_fiverep.json`` for Sonnet and
``factorial_by_model.json`` for the paper scope. Run from the repo root:

    uv run python code/benchmark/factorial_fiverep.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analysis_dataset as ad  # noqa: E402
import factorial_analysis as fa  # noqa: E402

MODEL_ARMS = {
    "sonnet": ("sonnet_zero_shot", "sonnet_few_shot"),
    "codex": ("codex_zero_shot", "codex_few_shot"),
    "deepseek": ("deepseek_zero_shot", "deepseek_few_shot"),
}
#: The model arms of the paper's main specification.
PAPER_MODELS = ad.PAPER_MODELS
JOINT_TESTS = {
    "H1_joint": ("H1_r", "H1_stata"),
    "H4_joint": ("H4_self_repair", "H4_tool_loop"),
}
CROSS_MODEL_CONTRASTS = (
    "H1_r",
    "H1_stata",
    "H2_fewshot",
    "H3_self_repair",
    "H3_tool_loop",
    "H4_self_repair",
    "H4_tool_loop",
    "AUX_fewshot_chatbot",
    "GAP_stata_repair_zs",
    "GAP_stata_agent_zs",
    "GAP_stata_avg",
)
#: Pairwise cross-arm differences reported for the paper scope.
CROSS_MODEL_PAIRS = (("sonnet", "codex"),)
POOLED_TOL = 1e-9
#: Tolerance of the pooled-equals-mean-of-arms identity on the balanced sample.
POOLED_MEAN_TOL = 1e-9
ARM_IDS = MODEL_ARMS["sonnet"]
MAX_REPS = 12
N_CELLS = 18
OUT_PATH = (
    ad.PROJECT_ROOT
    / "data"
    / "analysis"
    / "factorial_fiverep.json"
)
#: The paper estimation file: per paper arm, the two-model pooled fit, and the
#: cross-arm differences.
ALL_MODELS_OUT_PATH = OUT_PATH.with_name("factorial_by_model.json")
#: The three-arm diagnostic file. The paper does not read it.
DIAGNOSTIC_OUT_PATH = OUT_PATH.with_name("factorial_by_model_all.json")
SAMPLE_RULE = (
    "Twelve-rep analysis sample: first twelve eligible repetitions per "
    "task/platform/interface cell and both prompt arms"
)
POOLED_SAMPLE_RULE = (
    "Two-model paper sample: the twelve-rep sample of the Claude Sonnet 4.6 "
    "and GPT-5.4 through Codex arms, pooled"
)
NO_FE_TOL = 1e-9
STATA_TOL = 1e-6


def rows_to_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Normalized run rows -> analysis frame with 1..18 design-cell ids."""
    records = [
        {
            "success": int(bool(row["task_success"])),
            "task": str(row["task"]),
            "platform": str(row["platform"]),
            "prompt": str(row["prompt"]),
            "interface": str(row["interface"]),
            "cell": fa.cell_index(
                str(row["platform"]), str(row["prompt"]), str(row["interface"])
            ),
        }
        for row in rows
    ]
    return pd.DataFrame.from_records(records)


def cell_mean_contrasts(
    frame: pd.DataFrame, weights: Mapping[str, np.ndarray]
) -> dict[str, float]:
    """Each contrast evaluated directly on the 18 raw cell means."""
    means = frame.groupby("cell")["success"].mean()
    missing = sorted(set(range(1, N_CELLS + 1)) - set(means.index))
    if missing:
        raise AssertionError(f"design cells with no runs: {missing}")
    return {
        name: float(sum(v[k - 1] * means[k] for k in range(1, N_CELLS + 1)))
        for name, v in weights.items()
    }


def fit_no_fe(frame: pd.DataFrame):
    """Saturated cell-dummy model without task fixed effects (check model)."""
    return smf.ols("success ~ C(cell, Treatment(1))", data=frame).fit()


def saturation_check(
    frame: pd.DataFrame,
    weights: Mapping[str, np.ndarray],
    fe_contrasts: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Verify the contrast machinery against raw cell means.

    The no-FE saturated model must reproduce the cell-mean contrasts exactly
    (hard assertion). The task-FE contrasts differ from the cell means only
    through attrition imbalance across tasks within cells; that gap is
    reported per contrast, not asserted away.
    """
    cell_means = cell_mean_contrasts(frame, weights)
    no_fe = fa.contrast_table(fit_no_fe(frame), dict(weights))
    no_fe_gaps = {
        name: abs(no_fe[name]["estimate"] - cell_means[name]) for name in weights
    }
    worst = max(no_fe_gaps, key=no_fe_gaps.get)
    if no_fe_gaps[worst] > NO_FE_TOL:
        raise AssertionError(
            "saturation check failed: no-FE contrast != cell-mean contrast "
            f"for {worst} (gap {no_fe_gaps[worst]:.2e})"
        )
    return {
        "rule": (
            "no-FE saturated contrasts equal raw cell-mean contrasts to "
            f"{NO_FE_TOL:g} (asserted); task-FE minus cell-mean gap reported "
            "per contrast (attrition imbalance across tasks within cells)"
        ),
        "no_fe_max_abs_gap": max(no_fe_gaps.values()),
        "cell_mean_contrasts": {k: round(v, 6) for k, v in cell_means.items()},
        "fe_minus_cell_mean": {
            name: round(fe_contrasts[name]["estimate"] - cell_means[name], 6)
            for name in weights
        },
    }


def build_payload(
    frame: pd.DataFrame,
    contrasts: Mapping[str, Mapping[str, float]],
    h1_joint_p: float,
    h4_joint_p: float,
    check: Mapping[str, Any],
    manifest: Mapping[str, Any],
    workdir: Path,
    model: str,
    arm_ids: Sequence[str],
) -> dict[str, Any]:
    prompt_counts = frame["prompt"].value_counts()
    return {
        "generated_by": "code/benchmark/factorial_fiverep.py",
        "input": (
            f"analysis_dataset.load_runs(arm_ids={list(arm_ids)!r}, "
            f"max_reps={MAX_REPS})"
        ),
        "sample": f"{model}: {SAMPLE_RULE}",
        "model_id": model,
        "n_analysis": int(len(frame)),
        "n_zero_shot": int(prompt_counts.get("zero_shot", 0)),
        "n_few_shot": int(prompt_counts.get("few_shot", 0)),
        "n_tasks": int(frame["task"].nunique()),
        "model": (
            "OLS success ~ task FE + 18 cell dummies (saturated 2x3x3, "
            "omitted cell = python x zero_shot x no_feedback), CR1 cluster "
            "by task"
        ),
        "wcb": {
            "engine": "Stata boottest (null imposed)",
            "weights": "rademacher",
            "reps": fa.WCB_REPS,
            "seed": fa.WCB_SEED,
        },
        "contrast_convention": "equal weight per design cell",
        "contrasts": {name: dict(value) for name, value in contrasts.items()},
        "h1_joint_p_wcb": h1_joint_p,
        "h4_joint_p_wcb": h4_joint_p,
        "saturation_check": dict(check),
        "cells": fa.cell_summary(frame),
        "arm_coverage": {
            arm_id: {
                key: manifest["arms"][arm_id][key]
                for key in (
                    "status",
                    "usable_runs",
                    "cells_full",
                    "cells_short",
                    "cells_empty",
                    "empty_cells",
                )
            }
            for arm_id in arm_ids
        },
        "stata_workdir": str(workdir),
    }


def analyze_model(model: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estimate the common factorial specification for one model.

    A paper arm loads through ``load_paper_runs``, so its rows are checked
    against the pinned paper sample manifest. A non-paper arm keeps the corpus
    pin; it is a diagnostic, not a paper number.
    """
    arm_ids = MODEL_ARMS[model]
    if model in PAPER_MODELS:
        result = ad.load_paper_runs(
            max_reps=MAX_REPS, include_turns=False, arm_ids=list(arm_ids)
        )
    else:
        result = ad.load_runs(
            arm_ids=list(arm_ids),
            max_reps=MAX_REPS,
            include_turns=False,
            verify_manifest=True,
        )
    frame = rows_to_frame(result.rows)
    weights = fa.contrast_weights()
    fit = fa.fit_lpm(frame)
    contrasts = fa.contrast_table(fit, weights)
    check = saturation_check(frame, weights, contrasts)

    workdir = Path(tempfile.mkdtemp(prefix=f"factorial_{model}_wcb_"))
    stata = fa.run_stata_wcb(frame, weights, workdir, joint_tests=JOINT_TESTS)
    for name in weights:
        gap = abs(contrasts[name]["estimate"] - stata[name]["estimate_stata"])
        if gap > STATA_TOL:
            raise RuntimeError(
                f"Stata/statsmodels estimate mismatch for {name}: {gap:.2e}"
            )
        contrasts[name]["p_wcb"] = stata[name]["p_wcb"]

    payload = build_payload(
        frame,
        contrasts,
        stata["H1_joint"]["p_wcb"],
        stata["H4_joint"]["p_wcb"],
        check,
        result.manifest,
        workdir,
        model,
        arm_ids,
    )
    return payload, frame


def pooled_frame(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack the per-arm frames and add the arm and arm x task keys."""
    stacked = pd.concat(
        [frames[model].assign(arm=model) for model in frames], ignore_index=True
    )
    stacked["armtask"] = stacked["arm"] + "::" + stacked["task"]
    return stacked


def pooled_mean_check(
    contrasts: Mapping[str, Mapping[str, float]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Check the pooled contrasts against the mean of the per-arm contrasts.

    Every arm x task x cell holds the same number of runs, so the pooled fit
    with arm x task fixed effects reproduces the unweighted mean of the
    per-arm contrasts. A gap means the arms stopped being balanced.
    """
    models = list(payloads)
    gaps = {
        name: float(
            contrasts[name]["estimate"]
            - sum(
                float(payloads[model]["contrasts"][name]["estimate"])
                for model in models
            )
            / len(models)
        )
        for name in contrasts
    }
    worst = max(gaps, key=lambda name: abs(gaps[name]))
    if abs(gaps[worst]) > POOLED_MEAN_TOL:
        raise RuntimeError(
            "pooled contrast is not the mean of the per-arm contrasts for "
            f"{worst} (gap {gaps[worst]:.2e}); the arms are no longer balanced"
        )
    return {
        "rule": (
            "pooled contrast equals the unweighted mean of the per-arm "
            f"contrasts to {POOLED_MEAN_TOL:g} (asserted)"
        ),
        "max_abs_gap": abs(gaps[worst]),
        "arms": models,
    }


def analyze_pooled(
    frames: Mapping[str, pd.DataFrame],
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Estimate the two-model pooled specification with the same contrasts.

    One saturated LPM over both paper arms: arm x task fixed effects plus the
    18 common cell dummies, CR1 clustered on the 21 shared tasks. Inference is
    the same wild cluster bootstrap the per-arm runs use, on the identical
    regression in Stata.
    """
    pool = pooled_frame(frames)
    weights = fa.contrast_weights()
    fit = smf.ols(
        "success ~ C(armtask) + C(cell, Treatment(1))", data=pool
    ).fit(cov_type="cluster", cov_kwds={"groups": pool["task"]})
    contrasts = fa.contrast_table(fit, weights)
    check = saturation_check(pool, weights, contrasts)
    mean_check = pooled_mean_check(contrasts, payloads)

    workdir = Path(tempfile.mkdtemp(prefix="factorial_pooled_wcb_"))
    stata = fa.run_stata_wcb(
        pool, weights, workdir, joint_tests=JOINT_TESTS, fe_column="armtask"
    )
    for name in weights:
        gap = abs(contrasts[name]["estimate"] - stata[name]["estimate_stata"])
        if gap > STATA_TOL:
            raise RuntimeError(
                f"Stata/statsmodels estimate mismatch for pooled {name}: {gap:.2e}"
            )
        contrasts[name]["p_wcb"] = stata[name]["p_wcb"]

    prompt_counts = pool["prompt"].value_counts()
    return {
        "generated_by": "code/benchmark/factorial_fiverep.py",
        "sample": POOLED_SAMPLE_RULE,
        "model_id": "pooled",
        "models": list(frames),
        "n_analysis": int(len(pool)),
        "n_zero_shot": int(prompt_counts.get("zero_shot", 0)),
        "n_few_shot": int(prompt_counts.get("few_shot", 0)),
        "n_tasks": int(pool["task"].nunique()),
        "n_clusters": int(pool["task"].nunique()),
        "model": (
            "OLS success ~ arm x task FE + 18 cell dummies (saturated 2x3x3, "
            "omitted cell = python x zero_shot x no_feedback), CR1 cluster "
            "by task"
        ),
        "wcb": {
            "engine": "Stata boottest (null imposed)",
            "weights": "rademacher",
            "reps": fa.WCB_REPS,
            "seed": fa.WCB_SEED,
        },
        "contrast_convention": "equal weight per design cell",
        "contrasts": {name: dict(value) for name, value in contrasts.items()},
        "h1_joint_p_wcb": stata["H1_joint"]["p_wcb"],
        "h4_joint_p_wcb": stata["H4_joint"]["p_wcb"],
        "saturation_check": dict(check),
        "pooled_mean_check": mean_check,
        "cells": fa.cell_summary(pool),
        "stata_workdir": str(workdir),
    }


def model_pairs(models: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Every ordered pair of arms, in the order the arms were requested."""
    return tuple(
        (a, b)
        for index, a in enumerate(models)
        for b in list(models)[index + 1 :]
    )


def cross_model_contrast_differences(
    frames: Mapping[str, pd.DataFrame],
    payloads: Mapping[str, Mapping[str, Any]],
    pairs: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Pairwise cross-arm differences of the contrasts, under one covariance.

    Fits the arm-specific saturated model over the requested arms: model x
    task fixed effects (no intercept) plus model x cell dummies with cell 1
    omitted within each model, CR1 clustered on the 21 shared task clusters.
    The fit reproduces every per-arm contrast exactly (asserted), so the
    pairwise differences compare the per-arm contrasts under one CR1
    covariance. Two-sided p-values use the t distribution with 20 degrees of
    freedom.
    """
    weights = fa.contrast_weights()
    models = list(frames)
    if pairs is None:
        pairs = CROSS_MODEL_PAIRS if set(models) == set(PAPER_MODELS) else model_pairs(models)
    pool = pd.concat(
        [frames[m].assign(arm=m) for m in models], ignore_index=True
    )
    y = pool["success"].to_numpy(float)
    tasks = sorted(pool["task"].unique())
    cols: list[str] = []
    blocks: list[np.ndarray] = []
    for m in models:
        for task in tasks:
            cols.append(f"fe::{m}::{task}")
            blocks.append(((pool["arm"] == m) & (pool["task"] == task)).to_numpy(float))
    for m in models:
        for k in range(2, N_CELLS + 1):
            cols.append(f"mc::{m}::{k}")
            blocks.append(((pool["arm"] == m) & (pool["cell"] == k)).to_numpy(float))
    design = np.column_stack(blocks)
    n_obs, n_params = design.shape
    if np.linalg.matrix_rank(design) != n_params:
        raise RuntimeError("pooled cross-model design matrix is rank deficient")
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ (design.T @ y)
    resid = y - design @ beta
    groups = pool["task"].to_numpy()
    cluster_ids = np.unique(groups)
    n_clusters = len(cluster_ids)
    meat = np.zeros((n_params, n_params))
    for cluster in cluster_ids:
        mask = groups == cluster
        score = design[mask].T @ resid[mask]
        meat += np.outer(score, score)
    correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
    vcov = correction * xtx_inv @ meat @ xtx_inv

    def contrast_row(model: str, v: np.ndarray) -> np.ndarray:
        row = np.zeros(n_params)
        for k in range(2, N_CELLS + 1):
            if abs(v[k - 1]) > 1e-15:
                row[cols.index(f"mc::{model}::{k}")] = v[k - 1]
        return row

    for name in CROSS_MODEL_CONTRASTS:
        for m in models:
            pooled_est = float(contrast_row(m, weights[name]) @ beta)
            per_arm = float(payloads[m]["contrasts"][name]["estimate"])
            if abs(pooled_est - per_arm) > POOLED_TOL:
                raise RuntimeError(
                    f"pooled model does not reproduce {m} {name}: "
                    f"{pooled_est!r} vs {per_arm!r}"
                )

    df_t = n_clusters - 1
    differences: dict[str, dict[str, dict[str, float]]] = {}
    for name in CROSS_MODEL_CONTRASTS:
        entry: dict[str, dict[str, float]] = {}
        for a, b in pairs:
            row = contrast_row(a, weights[name]) - contrast_row(b, weights[name])
            est = float(row @ beta)
            se = float(np.sqrt(row @ vcov @ row))
            t_stat = est / se
            entry[f"{a}_minus_{b}"] = {
                "estimate": est,
                "se_cr1": se,
                "t_cr1": t_stat,
                "p_t20": float(2.0 * stats.t.sf(abs(t_stat), df_t)),
            }
        differences[name] = entry
    return {
        "generated_by": "code/benchmark/factorial_fiverep.py",
        "model": (
            "pooled OLS success ~ model x task FE + model x cell dummies "
            "(saturated within each arm, omitted cell 1 per model), CR1 "
            "cluster by task"
        ),
        "n_analysis": int(n_obs),
        "n_params": int(n_params),
        "n_clusters": int(n_clusters),
        "df": int(df_t),
        "inference": "CR1 cluster-by-task covariance; two-sided p from t(df)",
        "saturation": (
            "pooled per-arm contrasts equal the per-arm estimates to "
            f"{POOLED_TOL:g} (asserted)"
        ),
        "differences": differences,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=(*MODEL_ARMS, "paper", "all"),
        default="paper",
        help=(
            "Arms to estimate. 'paper' (default) estimates the two paper arms, "
            "the two-model pooled fit, and the cross-arm differences, and "
            "writes factorial_by_model.json. 'all' estimates the three arms as "
            "a diagnostic and writes factorial_by_model_all.json."
        ),
    )
    args = parser.parse_args()
    if args.model == "paper":
        selected = tuple(PAPER_MODELS)
    elif args.model == "all":
        selected = tuple(MODEL_ARMS)
    else:
        selected = (args.model,)
    results = {model: analyze_model(model) for model in selected}
    payloads = {model: payload for model, (payload, _frame) in results.items()}
    frames = {model: frame for model, (_payload, frame) in results.items()}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.model == "paper":
        out: dict[str, Any] = dict(payloads)
        out["pooled"] = analyze_pooled(frames, payloads)
        out["cross_model"] = cross_model_contrast_differences(frames, payloads)
        out["paper_scope"] = {
            "decision_record": (
                "quality_reports/decision_records/"
                "2026-08-21_two-model-main-specification.md"
            ),
            "models": list(PAPER_MODELS),
            "sample_version": ad.PAPER_SAMPLE_VERSION,
            "pooled_definition": (
                "the two paper arms pooled in one saturated LPM with arm x "
                "task fixed effects and 18 common cell dummies"
            ),
            "caveat": (
                "the constrained-agent condition runs under the same "
                "600-second session wall for both arms, and no retained Sonnet "
                "run is bound by a model-turn cap; the runs the cap once "
                "stopped were recollected without it (decision record "
                "2026-08-24_sonnet-rerun-replacements)"
            ),
        }
        ALL_MODELS_OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {ALL_MODELS_OUT_PATH}")
        # The Sonnet payload also refreshes the single-model file so
        # build_manuscript_numbers.py always reads the same estimation.
        OUT_PATH.write_text(json.dumps(payloads["sonnet"], indent=2) + "\n")
        print(f"wrote {OUT_PATH}")
        payloads = {**payloads, "pooled": out["pooled"]}
    elif args.model == "all":
        out = dict(payloads)
        out["cross_model"] = cross_model_contrast_differences(frames, payloads)
        DIAGNOSTIC_OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {DIAGNOSTIC_OUT_PATH}")
    else:
        OUT_PATH.write_text(json.dumps(payloads[args.model], indent=2) + "\n")
        print(f"wrote {OUT_PATH}")

    for model, payload in payloads.items():
        contrasts = payload["contrasts"]
        joint_p = payload["h1_joint_p_wcb"]
        print(
            f"\n% {model}: LaTeX fragment (estimate, CR1 SE, WCB p) in percentage points:"
        )
        print(fa.latex_fragment(contrasts, joint_p, fa.ROW_LABELS))


if __name__ == "__main__":
    main()

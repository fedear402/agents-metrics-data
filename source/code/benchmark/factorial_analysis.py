"""Factorial analysis of task success for the manuscript.

Estimates the saturated linear probability model with task fixed effects
(manuscript equation (1)) on the analysis sample in ``task_success_report.json``
and computes the pre-specified hypothesis contrasts. All contrasts are
design-averaged with equal weight on each design cell (the cells are not
balanced in run counts, so equal-cell weighting is the declared convention).

Two designs are supported:

* default: the full 2x3x3 design (18 cells; prompt x interface x platform)
  with contrasts H1-H4, written to ``factorial_analysis.json``;
* ``--zero-shot-only``: the paper's zero-shot analysis sample (decision record
  2026-07-25_fewshot-respecification), a 3x3 design (9 cells; platform x
  interface) with contrasts H1a/H1b (platform gaps at the chatbot baseline),
  the H1 joint test, and H3a/H3b (interface effects design-averaged over the
  three platforms), written to ``factorial_analysis_zero_shot.json``.

Inference: CR1 cluster-by-task standard errors from statsmodels, plus wild
cluster bootstrap p-values (Rademacher weights, 9,999 replications, null
imposed) computed by Stata ``boottest`` on the identical regression. Every
stored bootstrap p-value counts the observed statistic with the bootstrap
draws, ``(m + 1) / (reps + 1)`` where ``m`` is the number of draws at least
as extreme, so a p-value of exactly zero never prints. The script asserts
that the Stata and statsmodels contrast estimates agree before trusting the
bootstrap p-values.

This module makes no model calls and does not rerun generated code. It reads
the scored report produced by ``strict_posthoc.py`` and writes the analysis
JSON next to it, then prints a LaTeX table fragment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from stata_log_privacy import strip_license_details

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent.parent
# Derived reports and their inputs live in the arm's _working directory; the
# arm root itself holds one directory per run.
SONNET_WORKING = PROJECT_ROOT / "data" / "runs" / "sonnet_zero_shot" / "_working"
REPORT_PATH = SONNET_WORKING / "task_success_report.json"
OUT_PATH = SONNET_WORKING / "factorial_analysis.json"
ZERO_SHOT_OUT_PATH = SONNET_WORKING / "factorial_analysis_zero_shot.json"

STATA_BIN = "/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se"
WCB_REPS = 9_999
WCB_SEED = 20260701

PLATFORMS = ["python", "r", "stata"]
PROMPTS = ["zero_shot", "few_shot"]
INTERFACES = ["no_feedback", "self_repair", "tool_loop"]

INTERFACE_LABELS = {
    "no_feedback": "chatbot",
    "self_repair": "one-repair",
    "tool_loop": "constrained agent",
}


def cell_index(platform: str, prompt: str, interface: str) -> int:
    """Deterministic 1..18 cell id; cell 1 = (python, zero_shot, no_feedback)."""
    return (
        PLATFORMS.index(platform) * 6
        + PROMPTS.index(prompt) * 3
        + INTERFACES.index(interface)
        + 1
    )


def cell_index_zero_shot(platform: str, interface: str) -> int:
    """Deterministic 1..9 cell id for the 3x3 design; cell 1 = (python, no_feedback)."""
    return PLATFORMS.index(platform) * 3 + INTERFACES.index(interface) + 1


def load_analysis_frame(
    report_path: Path = REPORT_PATH, zero_shot_only: bool = False
) -> pd.DataFrame:
    """Analysis sample: scored runs with a model response, excluding quarantined tasks.

    Quarantined tasks (report key `quarantined`) were collected under a
    since-respecified contract and are excluded from the headline analysis
    pending repopulation; see strict_posthoc.QUARANTINED_TASKS. The sample is
    restricted to the legacy prompt wave: pinned-2026-07 re-collection records
    live in the same report but belong to the repopulation waves, not the
    analysis sample.

    With ``zero_shot_only`` the frame is further restricted to zero-shot runs
    (the paper's analysis sample; 2,354 runs) and cells are indexed 1..9 over
    platform x interface.
    """
    report = json.loads(report_path.read_text())
    rows = [
        {
            "success": int(bool(s["task_success"])),
            "task": s["task_id"],
            "platform": s["platform"],
            "prompt": s["prompt_mode"],
            "interface": s["exec_mode"],
            "cell": (
                cell_index_zero_shot(s["platform"], s["exec_mode"])
                if zero_shot_only
                else cell_index(s["platform"], s["prompt_mode"], s["exec_mode"])
            ),
        }
        for s in report["scores"]
        if s.get("model_response", True)
        and not s.get("quarantined", False)
        and s.get("prompt_wave", "legacy") == "legacy"
        and (not zero_shot_only or s["prompt_mode"] == "zero_shot")
    ]
    return pd.DataFrame(rows)


def contrast_weights() -> dict[str, np.ndarray]:
    """Equal-cell-weight contrast vectors over the 18 design cells.

    Every vector sums to zero, so the contrast value does not depend on the
    omitted-cell normalization of the regression coefficients.
    """
    weights: dict[str, np.ndarray] = {}

    def w() -> np.ndarray:
        return np.zeros(18)

    # H1: platform contrasts at the baseline (zero-shot, chatbot).
    for plat in ("r", "stata"):
        v = w()
        v[cell_index(plat, "zero_shot", "no_feedback") - 1] = 1.0
        v[cell_index("python", "zero_shot", "no_feedback") - 1] = -1.0
        weights[f"H1_{plat}"] = v

    # H2: design-averaged few-shot effect (average over 9 platform x interface cells).
    v = w()
    for plat in PLATFORMS:
        for iface in INTERFACES:
            v[cell_index(plat, "few_shot", iface) - 1] += 1.0 / 9.0
            v[cell_index(plat, "zero_shot", iface) - 1] -= 1.0 / 9.0
    weights["H2_fewshot"] = v

    # H3: design-averaged interface effects relative to the chatbot condition
    # (average over 6 platform x prompt cells).
    for iface in ("self_repair", "tool_loop"):
        v = w()
        for plat in PLATFORMS:
            for prompt in PROMPTS:
                v[cell_index(plat, prompt, iface) - 1] += 1.0 / 6.0
                v[cell_index(plat, prompt, "no_feedback") - 1] -= 1.0 / 6.0
        weights[f"H3_{iface}"] = v

    # H4: few-shot x interface difference-in-differences relative to chatbot
    # (average over the 3 platforms).
    for iface in ("self_repair", "tool_loop"):
        v = w()
        for plat in PLATFORMS:
            v[cell_index(plat, "few_shot", iface) - 1] += 1.0 / 3.0
            v[cell_index(plat, "zero_shot", iface) - 1] -= 1.0 / 3.0
            v[cell_index(plat, "few_shot", "no_feedback") - 1] -= 1.0 / 3.0
            v[cell_index(plat, "zero_shot", "no_feedback") - 1] += 1.0 / 3.0
        weights[f"H4_{iface}"] = v

    # Auxiliary: few-shot effect in the chatbot condition (average over platforms).
    # Not one of the pre-specified H1-H4; reported to interpret the H4 signs.
    v = w()
    for plat in PLATFORMS:
        v[cell_index(plat, "few_shot", "no_feedback") - 1] += 1.0 / 3.0
        v[cell_index(plat, "zero_shot", "no_feedback") - 1] -= 1.0 / 3.0
    weights["AUX_fewshot_chatbot"] = v

    # Software-gap diagnostics reported next to H1b. The zero-shot gap is
    # traced across the three agency conditions: the chatbot cell is H1_stata
    # itself, and these two add the self-repair and constrained-agent cells.
    # GAP_stata_avg is Stata - Python design-averaged over the six prompt x
    # agency cells (equal weight 1/6 each).
    v = w()
    v[cell_index("stata", "zero_shot", "self_repair") - 1] = 1.0
    v[cell_index("python", "zero_shot", "self_repair") - 1] = -1.0
    weights["GAP_stata_repair_zs"] = v

    v = w()
    v[cell_index("stata", "zero_shot", "tool_loop") - 1] = 1.0
    v[cell_index("python", "zero_shot", "tool_loop") - 1] = -1.0
    weights["GAP_stata_agent_zs"] = v

    v = w()
    for prompt in PROMPTS:
        for iface in INTERFACES:
            v[cell_index("stata", prompt, iface) - 1] += 1.0 / 6.0
            v[cell_index("python", prompt, iface) - 1] -= 1.0 / 6.0
    weights["GAP_stata_avg"] = v

    for name, v in weights.items():
        assert abs(v.sum()) < 1e-12, name
    return weights


def contrast_weights_zero_shot() -> dict[str, np.ndarray]:
    """Equal-cell-weight contrast vectors over the 9 zero-shot design cells.

    H1a/H1b are the platform gaps at the chatbot baseline; H3a/H3b are the
    interface effects relative to the chatbot condition, design-averaged with
    equal weight on the three platforms. Every vector sums to zero.
    """
    weights: dict[str, np.ndarray] = {}

    def w() -> np.ndarray:
        return np.zeros(9)

    # H1a/H1b: platform contrasts at the chatbot baseline.
    for plat in ("r", "stata"):
        v = w()
        v[cell_index_zero_shot(plat, "no_feedback") - 1] = 1.0
        v[cell_index_zero_shot("python", "no_feedback") - 1] = -1.0
        weights[f"H1_{plat}"] = v

    # H3a/H3b: interface effects relative to the chatbot condition,
    # averaged over the 3 platforms.
    for iface in ("self_repair", "tool_loop"):
        v = w()
        for plat in PLATFORMS:
            v[cell_index_zero_shot(plat, iface) - 1] += 1.0 / 3.0
            v[cell_index_zero_shot(plat, "no_feedback") - 1] -= 1.0 / 3.0
        weights[f"H3_{iface}"] = v

    for name, v in weights.items():
        assert abs(v.sum()) < 1e-12, name
    return weights


def fit_lpm(frame: pd.DataFrame):
    """Saturated LPM with task fixed effects and CR1 cluster-by-task errors."""
    model = smf.ols("success ~ C(task) + C(cell, Treatment(1))", data=frame)
    return model.fit(cov_type="cluster", cov_kwds={"groups": frame["task"]})


def contrast_table(result, weights: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Estimate and CR1 SE for each contrast as a linear combination of params."""
    names = list(result.params.index)
    out: dict[str, dict[str, float]] = {}
    for cname, v in weights.items():
        row = np.zeros(len(names))
        for k in range(2, len(v) + 1):
            if abs(v[k - 1]) > 1e-15:
                row[names.index(f"C(cell, Treatment(1))[T.{k}]")] = v[k - 1]
        tt = result.t_test(row)
        out[cname] = {
            "estimate": float(np.squeeze(tt.effect)),
            "se_cr1": float(np.squeeze(tt.sd)),
            "t_cr1": float(np.squeeze(tt.tvalue)),
        }
    return out


def stata_expression(v: np.ndarray) -> str:
    """Linear-combination expression for Stata ``lincom``/``boottest``.

    boottest's constraint parser rejects a leading ``+`` sign (error r(111)),
    so the first term is emitted unsigned and later terms use explicit
    ``+``/``-`` separators. The omitted base cell (cell 1) never appears: its
    coefficient is normalized to zero.
    """
    parts: list[str] = []
    for k in range(2, len(v) + 1):
        w = v[k - 1]
        if abs(w) <= 1e-15:
            continue
        magnitude = f"{abs(w):.15f}*{k}.cell"
        if not parts:
            parts.append(f"-{magnitude}" if w < 0 else magnitude)
        else:
            parts.append(f"- {magnitude}" if w < 0 else f"+ {magnitude}")
    return " ".join(parts)


DEFAULT_JOINT_TESTS: dict[str, tuple[str, ...]] = {
    "H1_joint": ("H1_r", "H1_stata"),
}


def finite_sample_p(raw_p: float, reps: int = WCB_REPS) -> float:
    """Map boottest's ``r(p)`` to the finite-sample bootstrap p-value.

    boottest reports ``m / reps`` where ``m`` counts the bootstrap draws at
    least as extreme as the observed statistic, so ``r(p)`` can be exactly
    zero. The stored p-value counts the observed draw with the bootstrap
    draws, ``(m + 1) / (reps + 1)``, bounded below by ``1 / (reps + 1)``.
    """
    exceed = round(raw_p * reps)
    if abs(exceed - raw_p * reps) > 1e-6:
        raise ValueError(f"bootstrap p {raw_p!r} is not a multiple of 1/{reps}")
    return (exceed + 1) / (reps + 1)


def run_stata_wcb(
    frame: pd.DataFrame,
    weights: dict[str, np.ndarray],
    workdir: Path,
    joint_tests: dict[str, tuple[str, ...]] | None = None,
    *,
    fe_column: str = "task",
) -> dict[str, dict[str, float]]:
    """Run the identical regression in Stata and boottest each contrast.

    Returns per-contrast Stata lincom estimates and wild-cluster-bootstrap
    p-values (finite-sample convention, see ``finite_sample_p``), plus one
    joint test per ``joint_tests`` entry (default: the H1 joint test).

    ``fe_column`` names the fixed-effect variable. The default absorbs task
    fixed effects, which is the per-arm specification. The pooled two-arm
    specification passes ``armtask`` (arm x task), so each arm keeps its own
    task baselines while the 18 cell dummies are common. Clustering stays on
    task in both cases: task is the sampling unit, and the pooled run holds
    the same 21 task clusters.
    """
    if joint_tests is None:
        joint_tests = DEFAULT_JOINT_TESTS
    for jname, members in joint_tests.items():
        missing_members = [m for m in members if m not in weights]
        if missing_members:
            raise KeyError(f"{jname}: unknown contrasts {missing_members}")
    csv_path = workdir / "factorial_input.csv"
    columns = ["success", "task", "cell"]
    if fe_column != "task":
        columns.insert(2, fe_column)
    export = frame[columns].copy()
    export.to_csv(csv_path, index=False)
    string_positions = " ".join(
        str(index + 1)
        for index, name in enumerate(columns)
        if name in {"task", fe_column}
    )

    lines = [
        f'import delimited "{csv_path}", clear stringcols({string_positions})',
        "encode task, gen(taskn)",
    ]
    if fe_column == "task":
        lines.append("reg success i.taskn ib1.cell, cluster(taskn)")
    else:
        lines.append(f"encode {fe_column}, gen(fen)")
        lines.append("reg success i.fen ib1.cell, cluster(taskn)")
    for name, v in weights.items():
        expr = stata_expression(v)
        lines += [
            f'di "LINCOM_START {name}"',
            f"lincom {expr}",
            f'di "LINCOM_EST {name} = " %20.0g r(estimate)',
            f'di "BOOTTEST_START {name}"',
            f"boottest {expr} = 0, reps({WCB_REPS}) weighttype(rademacher) seed({WCB_SEED}) nograph",
            f'di "BOOTTEST_P {name} = " %20.0g r(p)',
        ]
    for jname, members in joint_tests.items():
        joint = " ".join(f"({stata_expression(weights[n])} = 0)" for n in members)
        lines += [
            f'di "BOOTTEST_START {jname}"',
            f"boottest {joint}, reps({WCB_REPS}) weighttype(rademacher) seed({WCB_SEED}) nograph",
            f'di "BOOTTEST_P {jname} = " %20.0g r(p)',
        ]
    do_path = workdir / "factorial_wcb.do"
    do_path.write_text("\n".join(lines) + "\n")

    subprocess.run(
        [STATA_BIN, "-b", "do", str(do_path)],
        cwd=workdir,
        check=True,
        timeout=1800,
    )
    log = strip_license_details((workdir / "factorial_wcb.log").read_text())
    (workdir / "factorial_wcb.log").write_text(log)

    parsed: dict[str, dict[str, float]] = {}
    for line in log.splitlines():
        line = line.strip()
        if line.startswith("LINCOM_EST "):
            _, name, _, value = line.split()
            parsed.setdefault(name, {})["estimate_stata"] = float(value)
        elif line.startswith("BOOTTEST_P "):
            _, name, _, value = line.split()
            parsed.setdefault(name, {})["p_wcb"] = finite_sample_p(float(value))
    missing = [
        n for n in list(weights) + list(joint_tests) if "p_wcb" not in parsed.get(n, {})
    ]
    if missing:
        raise RuntimeError(f"boottest p-values missing for {missing}; see {workdir}/factorial_wcb.log")
    return parsed


def cell_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-cell n and success rate; works for both the 18- and 9-cell designs."""
    out = []
    for cell, sub in frame.groupby("cell", sort=True):
        out.append(
            {
                "cell": int(cell),
                "platform": str(sub["platform"].iloc[0]),
                "prompt": str(sub["prompt"].iloc[0]),
                "interface": str(sub["interface"].iloc[0]),
                "n": int(len(sub)),
                "success_rate": float(sub["success"].mean()),
            }
        )
    return out


ROW_LABELS = {
    "H1_r": "H1: R $-$ Python (zero-shot, chatbot)",
    "H1_stata": "H1: Stata $-$ Python (zero-shot, chatbot)",
    "H2_fewshot": "H2: few-shot $-$ zero-shot (design-averaged)",
    "H3_self_repair": "H3: self-repair $-$ chatbot (design-averaged)",
    "H3_tool_loop": "H3: constrained agent $-$ chatbot (design-averaged)",
    "H4_self_repair": "H4: few-shot $\\times$ self-repair interaction",
    "H4_tool_loop": "H4: few-shot $\\times$ constrained agent interaction",
    "AUX_fewshot_chatbot": "Aux.: few-shot $-$ zero-shot in the chatbot condition",
}

ZERO_SHOT_ROW_LABELS = {
    "H1_r": "H1a: R $-$ Python (chatbot baseline)",
    "H1_stata": "H1b: Stata $-$ Python (chatbot baseline)",
    "H3_self_repair": "H3a: self-repair $-$ chatbot (design-averaged)",
    "H3_tool_loop": "H3b: constrained agent $-$ chatbot (design-averaged)",
}


def latex_fragment(
    contrasts: dict[str, dict[str, float]],
    joint_p: float,
    row_labels: dict[str, str] = ROW_LABELS,
) -> str:
    lines = []
    for name, label in row_labels.items():
        c = contrasts[name]
        lines.append(
            f"{label} & {100 * c['estimate']:.1f} & ({100 * c['se_cr1']:.1f}) & {c['p_wcb']:.3f} \\\\"
        )
    lines.append(f"H1 joint: no platform gap at baseline & & & {joint_p:.3f} \\\\")
    return "\n".join(lines)


def run_analysis(zero_shot_only: bool = False) -> Path:
    """Fit the design, run the Stata WCB, and write the analysis JSON."""
    frame = load_analysis_frame(zero_shot_only=zero_shot_only)
    if zero_shot_only:
        weights = contrast_weights_zero_shot()
        out_path = ZERO_SHOT_OUT_PATH
        row_labels = ZERO_SHOT_ROW_LABELS
        design = "OLS success ~ task FE + 9 cell dummies (saturated 3x3, zero-shot only), CR1 cluster by task"
        sample = (
            "model_response, not quarantined, prompt_wave == legacy, "
            "prompt_mode == zero_shot"
        )
    else:
        weights = contrast_weights()
        out_path = OUT_PATH
        row_labels = ROW_LABELS
        design = "OLS success ~ task FE + 18 cell dummies (saturated 2x3x3), CR1 cluster by task"
        sample = "model_response, not quarantined, prompt_wave == legacy"
    result = fit_lpm(frame)
    contrasts = contrast_table(result, weights)

    workdir = Path(tempfile.mkdtemp(prefix="factorial_wcb_"))
    stata = run_stata_wcb(frame, weights, workdir)

    for name in weights:
        gap = abs(contrasts[name]["estimate"] - stata[name]["estimate_stata"])
        if gap > 1e-6:
            raise RuntimeError(
                f"Stata/statsmodels estimate mismatch for {name}: {gap:.2e}"
            )
        contrasts[name]["p_wcb"] = stata[name]["p_wcb"]

    payload = {
        "input": str(REPORT_PATH),
        "sample": sample,
        "n_analysis": int(len(frame)),
        "n_tasks": int(frame["task"].nunique()),
        "model": design,
        "wcb": {
            "engine": "Stata boottest (null imposed)",
            "weights": "rademacher",
            "reps": WCB_REPS,
            "seed": WCB_SEED,
        },
        "contrast_convention": "equal weight per design cell",
        "contrasts": contrasts,
        "h1_joint_p_wcb": stata["H1_joint"]["p_wcb"],
        "cells": cell_summary(frame),
        "stata_workdir": str(workdir),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")
    print("\n% LaTeX fragment (estimate, CR1 SE, WCB p) in percentage points:")
    print(latex_fragment(contrasts, stata["H1_joint"]["p_wcb"], row_labels))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zero-shot-only",
        action="store_true",
        help=(
            "Restrict to the zero-shot analysis sample and estimate the 9-cell "
            "(platform x interface) design with contrasts H1a/H1b/H1-joint/H3a/H3b."
        ),
    )
    args = parser.parse_args()
    run_analysis(zero_shot_only=args.zero_shot_only)


if __name__ == "__main__":
    main()

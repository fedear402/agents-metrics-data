"""Export the active task registry to reviewable JSON."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import task_display
from tasks import EXEC_MODES, PLATFORMS, PROMPT_MODES, TASKS, Task

ROOT = Path(__file__).parent
DEFAULT_OUT = ROOT / "task_registry.json"
PROJECT_ROOT = ROOT.parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "tasks" / "references"
ARCHIVED_GENERATORS = "code/benchmark/dgp"
STATLLM_CANDIDATE_INDEX = (
    PROJECT_ROOT
    / "data"
    / "external_corpora"
    / "statllm"
    / "LOCAL_TRANSLATABLE_TASK_CANDIDATES.csv"
)
METRICSAI_CANDIDATE_FILES = [
    (
        "problem_set",
        PROJECT_ROOT
        / "data"
        / "external_corpora"
        / "metricsai"
        / "problem_set"
        / "regression_set.csv",
    ),
    (
        "published_paper",
        PROJECT_ROOT
        / "data"
        / "external_corpora"
        / "metricsai"
        / "published_paper"
        / "regression_testset.csv",
    ),
]


DATASET_META: dict[str, dict[str, Any]] = {
    "pilot.csv": {
        "source_type": "seeded_dgp",
        "generator": f"{ARCHIVED_GENERATORS}/generate_data.py",
        "seed": 20260515,
        "baseline_parameters": {"n": 2000},
    },
    "pilot_v2.csv": {
        "source_type": "seeded_dgp",
        "generator": f"{ARCHIVED_GENERATORS}/generate_data_v2.py",
        "seed": 20260515,
        "baseline_parameters": {"n": 2000, "clusters": 50},
    },
    "pilot_v3.csv": {
        "source_type": "seeded_dgp",
        "generator": f"{ARCHIVED_GENERATORS}/generate_data_v3.py",
        "seed": 20260515,
        "baseline_parameters": {"n_firms": 200, "n_years": 10},
    },
    "pilot_v4_arima.csv": {
        "source_type": "seeded_dgp",
        "generator": f"{ARCHIVED_GENERATORS}/generate_data_v4.py",
        "seed": 20260516,
        "baseline_parameters": {"n": 250, "phi": 0.5, "theta": 0.3},
    },
    "pilot_v4_cox.csv": {
        "source_type": "seeded_dgp",
        "generator": f"{ARCHIVED_GENERATORS}/generate_data_v4.py",
        "seed": 20260516,
        "baseline_parameters": {"n": 2000, "censoring_rate": 0.30},
    },
    "pilot_v4_var.csv": {
        "source_type": "seeded_dgp",
        "generator": f"{ARCHIVED_GENERATORS}/generate_data_v4.py",
        "seed": 20260516,
        "baseline_parameters": {"n": 200, "lag_order": 2},
    },
    "pilot_v5_collinear.csv": {
        "source_type": "statllm_style_seeded_dgp",
        "generator": f"{ARCHIVED_GENERATORS}/generate_data_v5.py",
        "seed": 20260517,
        "baseline_parameters": {"n": 500},
    },
    "pilot_v5_ttest.csv": {
        "source_type": "statllm_style_seeded_dgp",
        "generator": f"{ARCHIVED_GENERATORS}/generate_data_v5.py",
        "seed": 20260517,
        "baseline_parameters": {"n_per_group": 100, "mean_difference": 0.5},
    },
    "pilot_v5_anova.csv": {
        "source_type": "statllm_style_seeded_dgp",
        "generator": f"{ARCHIVED_GENERATORS}/generate_data_v5.py",
        "seed": 20260517,
        "baseline_parameters": {"groups": 4, "n_per_group": 50},
    },
    "pilot_v5_chisq.csv": {
        "source_type": "statllm_style_seeded_dgp",
        "generator": f"{ARCHIVED_GENERATORS}/generate_data_v5.py",
        "seed": 20260517,
        "baseline_parameters": {"n": 600, "table_shape": "2x3"},
    },
    "pilot_v5_dynpanel.csv": {
        "source_type": "seeded_dgp",
        "generator": f"{ARCHIVED_GENERATORS}/generate_data_v5.py",
        "seed": 20260517,
        "baseline_parameters": {"n_firms": 100, "n_periods": 10, "rho": 0.5},
    },
    "pilot_v5_rdd.csv": {
        "source_type": "seeded_dgp",
        "generator": f"{ARCHIVED_GENERATORS}/generate_data_v5.py",
        "seed": 20260517,
        "baseline_parameters": {"n": 1000, "cutoff": 0, "late": 1.0},
    },
    "pilot_v5_lp.csv": {
        "source_type": "seeded_dgp",
        "generator": f"{ARCHIVED_GENERATORS}/generate_data_v5.py",
        "seed": 20260517,
        "baseline_parameters": {"n": 250, "rho": 0.5, "theta": 0.5},
    },
    "pilot_v5_wcb.csv": {
        "source_type": "seeded_dgp",
        "generator": f"{ARCHIVED_GENERATORS}/generate_data_v5.py",
        "seed": 20260517,
        "baseline_parameters": {
            "cluster_count": 5,
            "cluster_imbalance": "none",
            "effect_size": 0.4,
            "n_per_cluster": 80,
        },
    },
}


TASK_GRID_AXES: dict[str, list[str]] = {
    "ols_robust": ["n", "signal_to_noise", "heteroskedasticity_strength"],
    "arima_forecast": ["n", "phi", "theta", "innovation_sigma"],
    "cluster_ols": ["n_clusters", "cluster_size", "intraclass_correlation"],
    "two_way_cluster": ["n_firms", "n_years", "firm_year_error_share"],
    "vif": ["collinearity_strength", "n", "number_of_regressors"],
    "cooks_distance": ["outlier_leverage", "outlier_residual_size", "n"],
    "ttest_welch": ["mean_difference", "variance_ratio", "group_imbalance"],
    "anova_oneway": ["between_group_spread", "within_group_sd", "n_per_group"],
    "chisq_indep": ["association_strength", "cell_sparsity", "n"],
    "logit": ["base_rate", "effect_size", "separation_strength"],
    "probit": ["base_rate", "effect_size", "separation_strength"],
    "poisson_offset": ["exposure_variation", "count_overdispersion", "effect_size"],
    "cox_hazard": ["censoring_rate", "hazard_ratio", "baseline_shape"],
    "twoway_fe": ["n_firms", "n_years", "within_variation_share"],
    "iv2sls_robust": ["first_stage_strength", "endogeneity_strength", "n"],
    "dynpanel_ab": ["rho", "n_firms", "n_periods"],
    "staggered_did": ["effect_heterogeneity", "cohort_share", "pretrend_slope"],
    "rdd_sharp": ["bandwidth", "running_variable_density", "jump_size"],
    "lp_irf": ["shock_strength", "persistence", "horizon"],
    "wcb_few": ["cluster_count", "cluster_imbalance", "effect_size"],
    "var_irf": ["lag_order", "cross_lag_strength", "innovation_correlation"],
    "var_granger": ["lag_order", "cross_lag_strength", "n"],
}


MONTE_CARLO_GRID: dict[str, dict[str, list[int | float | str]]] = {
    "ols_robust": {
        "n": [500, 2000, 8000],
        "signal_to_noise": [0.5, 1.0, 2.0],
        "heteroskedasticity_strength": [0.0, 0.5, 1.0],
    },
    "arima_forecast": {
        "n": [100, 250, 500],
        "phi": [0.2, 0.5, 0.8],
        "theta": [0.0, 0.3, 0.6],
        "innovation_sigma": [0.5, 1.0, 2.0],
    },
    "cluster_ols": {
        "n_clusters": [10, 50, 100],
        "cluster_size": [20, 40, 80],
        "intraclass_correlation": [0.05, 0.2, 0.5],
    },
    "two_way_cluster": {
        "n_firms": [50, 200, 500],
        "n_years": [5, 10, 20],
        "firm_year_error_share": ["mostly_firm", "balanced", "mostly_year"],
    },
    "vif": {
        "collinearity_strength": [0.3, 0.7, 0.95],
        "n": [200, 500, 2000],
        "number_of_regressors": [5, 10, 20],
    },
    "cooks_distance": {
        "outlier_leverage": [0.0, 0.5, 0.9],
        "outlier_residual_size": [0.0, 2.5, 5.0],
        "n": [200, 500, 2000],
    },
    "ttest_welch": {
        "mean_difference": [0.0, 0.3, 0.8],
        "variance_ratio": [1.0, 2.25, 9.0],
        "group_imbalance": ["balanced", "2_to_1", "5_to_1"],
    },
    "anova_oneway": {
        "between_group_spread": [0.0, 0.5, 1.0],
        "within_group_sd": [0.5, 1.0, 2.0],
        "n_per_group": [25, 50, 200],
    },
    "chisq_indep": {
        "association_strength": [0.0, 0.1, 0.3],
        "cell_sparsity": ["none", "moderate", "high"],
        "n": [100, 600, 2000],
    },
    "logit": {
        "base_rate": [0.1, 0.5, 0.9],
        "effect_size": [0.3, 0.8, 1.5],
        "separation_strength": ["none", "partial", "near"],
    },
    "probit": {
        "base_rate": [0.1, 0.5, 0.9],
        "effect_size": [0.3, 0.6, 1.2],
        "separation_strength": ["none", "partial", "near"],
    },
    "poisson_offset": {
        "exposure_variation": ["low", "baseline", "high"],
        "count_overdispersion": [0.0, 0.5, 1.5],
        "effect_size": [0.1, 0.4, 0.9],
    },
    "cox_hazard": {
        "censoring_rate": [0.1, 0.3, 0.6],
        "hazard_ratio": [0.6, 1.0, 1.8],
        "baseline_shape": [0.7, 1.5, 2.5],
    },
    "twoway_fe": {
        "n_firms": [50, 200, 500],
        "n_years": [5, 10, 20],
        "within_variation_share": [0.1, 0.4, 0.8],
    },
    "iv2sls_robust": {
        "first_stage_strength": [3, 10, 30],
        "endogeneity_strength": [0.0, 0.4, 0.8],
        "n": [500, 2000, 8000],
    },
    "dynpanel_ab": {
        "rho": [0.2, 0.5, 0.8],
        "n_firms": [50, 100, 500],
        "n_periods": [6, 10, 20],
    },
    "staggered_did": {
        "effect_heterogeneity": [0.0, 1.0, 3.0],
        "cohort_share": ["balanced", "early_heavy", "late_heavy"],
        "pretrend_slope": [0.0, 0.05, 0.15],
    },
    "rdd_sharp": {
        "bandwidth": [0.25, 0.5, 1.0],
        "running_variable_density": ["uniform", "center_sparse", "center_dense"],
        "jump_size": [0.0, 1.0, 2.0],
    },
    "lp_irf": {
        "shock_strength": [0.2, 0.5, 1.0],
        "persistence": [0.2, 0.5, 0.8],
        "horizon": [1, 4, 8],
    },
    "wcb_few": {
        "cluster_count": [5, 10, 12],
        "cluster_imbalance": ["none", "moderate", "severe"],
        "effect_size": [0.0, 0.4, 0.8],
    },
    "var_irf": {
        "lag_order": [1, 2, 4],
        "cross_lag_strength": [0.0, 0.2, 0.5],
        "innovation_correlation": [0.0, 0.4, 0.8],
    },
    "var_granger": {
        "lag_order": [1, 2, 4],
        "cross_lag_strength": [0.0, 0.2, 0.5],
        "n": [100, 200, 500],
    },
}


CURATED_CANDIDATE_TASKS: list[dict[str, str]] = [
    {
        "candidate_id": "statllm_003_copper_anova_interaction",
        "source": "StatLLM",
        "source_row": "3",
        "family": "anova_or_variance",
        "dataset": "copper.csv",
        "proposed_output": "F, df_num, df_den, p_value for interaction terms",
        "rationale": "Portable two-way ANOVA with interaction; good StatLLM-style real-data task.",
    },
    {
        "candidate_id": "statllm_013_toxic_anova_interaction",
        "source": "StatLLM",
        "source_row": "13",
        "family": "anova_or_variance",
        "dataset": "toxic.csv",
        "proposed_output": "F, df_num, df_den, p_value for poison, treatment, and interaction",
        "rationale": "Second factorial ANOVA task with named factors and classical output.",
    },
    {
        "candidate_id": "statllm_017_sleep_mean_ci",
        "source": "StatLLM",
        "source_row": "17",
        "family": "hypothesis_test",
        "dataset": "sleeptime.csv",
        "proposed_output": "mean, t, df, p_value, ci_lower_90, ci_upper_90",
        "rationale": "Simple but useful one-sample inference task for output-schema discipline.",
    },
    {
        "candidate_id": "statllm_027_poplar_nonnormal_groups",
        "source": "StatLLM",
        "source_row": "27",
        "family": "nonparametric_test",
        "dataset": "poplar.csv",
        "proposed_output": "Kruskal-Wallis statistic, df, p_value",
        "rationale": "Adds nonparametric group comparison instead of another parametric ANOVA.",
    },
    {
        "candidate_id": "statllm_032_ecmo_small_sample_association",
        "source": "StatLLM",
        "source_row": "32",
        "family": "association_or_correlation",
        "dataset": "ecmo.csv",
        "proposed_output": "Fisher exact p_value and odds_ratio",
        "rationale": "Small-cell contingency task likely to expose default-test differences.",
    },
    {
        "candidate_id": "statllm_034_measurement_prediction",
        "source": "StatLLM",
        "source_row": "34",
        "family": "regression_or_diagnostics",
        "dataset": "measurement.csv",
        "proposed_output": "height coefficient, se, n_obs, prediction at declared height",
        "rationale": "Real-data regression plus prediction target, not just coefficient extraction.",
    },
    {
        "candidate_id": "statllm_103_imports_influence",
        "source": "StatLLM",
        "source_row": "103",
        "family": "regression_or_diagnostics",
        "dataset": "imports.csv",
        "proposed_output": "top Cook's distance observation and value",
        "rationale": "Real-data influence diagnostic complements the seeded Cook's-distance task.",
    },
    {
        "candidate_id": "metricsai_ps1_tobacco_ols",
        "source": "MetricsAI sample",
        "source_row": "problem_set/0",
        "family": "applied_ols",
        "dataset": "ps1_24S_cleaned.dta",
        "proposed_output": "tobacco coefficient, robust or declared se, p_value",
        "rationale": "Coursework-style applied OLS with categorical controls and data processing.",
    },
    {
        "candidate_id": "metricsai_ps2_panel_cluster",
        "source": "MetricsAI sample",
        "source_row": "problem_set/3",
        "family": "panel_regression",
        "dataset": "ps2_24S_cleaned.dta",
        "proposed_output": "primary coefficient and state-year clustered se",
        "rationale": "Applied panel-FE task with transformation and multiway clustering.",
    },
    {
        "candidate_id": "metricsai_birth_quarter_iv",
        "source": "MetricsAI sample",
        "source_row": "published_paper/0",
        "family": "instrumental_variables",
        "dataset": "Birth_Quarter_1980_30_39.dta",
        "proposed_output": "educ coefficient, se, p_value",
        "rationale": "Canonical applied IV task with dummy construction and instruments.",
    },
    {
        "candidate_id": "metricsai_colonial_origins_iv",
        "source": "MetricsAI sample",
        "source_row": "published_paper/1",
        "family": "instrumental_variables",
        "dataset": "Colonial_Origins_Reply.dta",
        "proposed_output": "risk coefficient, robust se, p_value",
        "rationale": "Published-paper IV task with compact data and robust inference.",
    },
    {
        "candidate_id": "metricsai_minwage_static_did",
        "source": "MetricsAI sample",
        "source_row": "published_paper/4",
        "family": "difference_in_differences",
        "dataset": "cardkrueger1994_cleaned.dta",
        "proposed_output": "treated coefficient, clustered se, p_value",
        "rationale": "Recognizable real-data DiD task with small applied dataset.",
    },
]


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned[:48] or "task"


def load_statllm_candidates() -> list[dict[str, str]]:
    if not STATLLM_CANDIDATE_INDEX.exists():
        return []

    candidates = []
    with STATLLM_CANDIDATE_INDEX.open(newline="") as fh:
        for row in csv.DictReader(fh):
            status = row["status"]
            if status not in {"use_now", "rewrite_then_use"}:
                continue
            row_id = int(row["ID"])
            candidates.append(
                {
                    "candidate_id": f"statllm_{row_id:03d}_{slug(row['DatasetName'])}",
                    "source": "StatLLM",
                    "source_row": row["ID"],
                    "source_status": status,
                    "family": row["method_family"],
                    "dataset": row["DatasetName"],
                    "proposed_output": row["TaskText"],
                    "rationale": row["translation_note"],
                    "problem_description_path": row["ProblemDescriptionPath"],
                    "data_description_path": row["DataDescriptionPath"],
                    "task_dataset_folder": row["TaskDatasetFolder"],
                    "human_verified_sas_path": row["HumanVerifiedSASPath"],
                }
            )
    return candidates


def load_metricsai_candidates() -> list[dict[str, str]]:
    candidates = []
    for subset, path in METRICSAI_CANDIDATE_FILES:
        if not path.exists():
            continue
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                row_id = row[""]
                method = row["method"]
                data_source = row["data_source"]
                candidates.append(
                    {
                        "candidate_id": (
                            f"metricsai_{subset}_{row_id}_{slug(method)}"
                        ),
                        "source": "MetricsAI sample",
                        "source_row": row_id,
                        "source_subset": subset,
                        "family": method,
                        "dataset": data_source,
                        "proposed_output": (
                            f"{row['method']} for y={row['y']} and x={row['x']}; "
                            f"target coefficient, standard error, and p-value."
                        ),
                        "rationale": (
                            "Local MetricsAI row with a published/sample reference "
                            "answer; promote only after locking platform-specific "
                            "data-loading and reference-output scripts."
                        ),
                        "control_variables": row["control_variables"],
                        "other_requirements": row["other_requirements"],
                        "reference_answer": row["answer"],
                        "article": row["article"],
                        "tags": row["tags"],
                    }
                )
    return candidates


def candidate_tasks() -> list[dict[str, str]]:
    """Return the expanded candidate pool, preserving the first copy of each id."""
    ordered = [
        *CURATED_CANDIDATE_TASKS,
        *load_statllm_candidates(),
        *load_metricsai_candidates(),
    ]
    seen = set()
    out = []
    for task in ordered:
        candidate_id = task["candidate_id"]
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        out.append(task)
    return out


def reference_paths_for_key(reference_key: str) -> list[str]:
    """Return reference files whose top-level JSON key exactly matches a task key."""
    paths = []
    for path in REFERENCE_DIR.glob("reference*.json"):
        reference_payload = json.loads(path.read_text())
        if reference_key in reference_payload:
            paths.append(str(path.relative_to(PROJECT_ROOT)))
    return sorted(paths)


def display_fields(task_id: str) -> dict[str, Any]:
    """Display name, group, and order for one task, from the canonical module.

    A task outside the analyzed set keeps a display name and a group but has no
    display position, because it never appears in a reader-facing task list.
    """
    fields: dict[str, Any] = {
        "display_name": task_display.display_name(task_id),
        "display_group": task_display.group_of(task_id),
        "display_group_order": task_display.group_order_of(task_id),
    }
    if task_display.is_analyzed(task_id):
        fields["display_order_in_group"] = task_display.order_in_group(task_id)
        fields["display_order"] = task_display.display_order(task_id)
        fields["analyzed"] = True
    else:
        fields["display_order_in_group"] = None
        fields["display_order"] = None
        fields["analyzed"] = False
    return fields


def registry_entry(task: Task) -> dict[str, Any]:
    data_meta = DATASET_META[task.data_file]
    return {
        **asdict(task),
        **display_fields(task.task_id),
        "dataset": {
            "path": f"data/tasks/inputs/{task.data_file}",
            **data_meta,
        },
        "reference": {
            "key": task.reference_key,
            "paths": reference_paths_for_key(task.reference_key),
        },
        "monte_carlo_axes": TASK_GRID_AXES[task.task_id],
        "monte_carlo_grid": MONTE_CARLO_GRID[task.task_id],
        "monte_carlo_status": "planned_not_generated",
    }


def build_registry() -> dict[str, Any]:
    return {
        "version": "2026-05-19",
        "status": "active_22_task_sonnet_sweep_registry",
        "source_of_truth": "code/benchmark/tasks.py",
        "notes": [
            "The benchmark harness imports tasks.py directly.",
            "This JSON file is the reviewable registry file for task design.",
            "Monte Carlo axes are declared here but not yet generated as extra datasets.",
            "External StatLLM and MetricsAI candidates remain candidate pools until reference outputs are built.",
        ],
        "candidate_status_rule": (
            "Candidate tasks are not eligible for live model runs until a fixed "
            "dataset copy, platform-specific reference outputs, scoring "
            "tolerances, and integrity checks exist."
        ),
        "monte_carlo_design": {
            "default_seeds": [20260519, 20260520, 20260521],
            "grid_policy": (
                "Vary one axis at a time around the baseline first. Use the "
                "complete grid only for families where the one-axis sweep "
                "shows a clear difficulty gradient."
            ),
            "status": "declared_not_generated",
        },
        "platforms": PLATFORMS,
        "prompt_modes": PROMPT_MODES,
        "execution_modes": EXEC_MODES,
        "tasks": [registry_entry(task) for task in TASKS],
        "candidate_tasks": [
            {**task, "status": "candidate_reference_needed"}
            for task in candidate_tasks()
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export task registry JSON.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path. Default: code/benchmark/task_registry.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_registry(), indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

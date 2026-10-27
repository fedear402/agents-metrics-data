"""Production task definitions for the 22-task benchmark draft.

Older exploratory slices are archived under
``code/benchmark/dgp/``.
This file is the single task import used by the benchmark harness.

``task_id`` is the internal identifier. It never changes, because the stored
run records and the pinned sample manifests reference it.

``title`` is the reader-facing display name. ``task_display.py`` is the source
of truth for every display name, group, and display order. The titles here must
match ``task_display.DISPLAY_NAME``; ``tests/test_task_display.py`` checks that.
No title carries a task number.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    task_id: str
    title: str
    prompt: str
    estimands: list[str]
    reference_key: str
    data_file: str
    data_description: str
    extra_hint: dict[str, str] = field(default_factory=dict)


PLATFORMS = ["python", "r", "stata"]
PROMPT_MODES = ["zero_shot", "few_shot"]
EXEC_MODES = ["no_feedback", "self_repair", "tool_loop", "agentic"]


PILOT_V1 = "Dataset: pilot.csv. Columns: id, x1, x2, y_lin, y_bin_logit, y_bin_probit."
PILOT_V2 = (
    "Dataset: pilot_v2.csv. Columns include id, x1, x2, y_iv, x1_iv, z, "
    "cluster, y_panel, exposure, and y_count."
)
PILOT_V3 = (
    "Dataset: pilot_v3.csv. Panel of 200 firms by 10 years. Columns include "
    "firm, year, x, y_twocluster, cohort, treat_post, y_did, z_sel, w, "
    "selected, and y_sel."
)
PILOT_V4_VAR = "Dataset: pilot_v4_var.csv. Columns: t, y1, y2."
PILOT_V5_COL = "Dataset: pilot_v5_collinear.csv. Columns: y, x1, x2, x3, x4, x5."


TASKS: list[Task] = [
    Task(
        task_id="ols_robust",
        title="OLS with robust standard errors",
        prompt=(
            "Estimate y_lin on x1 and x2 with an intercept. Use HC1 "
            "heteroskedasticity-robust standard errors. Report rows named "
            "const, x1, and x2."
        ),
        estimands=["const", "x1", "x2"],
        reference_key="ols_robust",
        data_file="pilot.csv",
        data_description=PILOT_V1,
    ),
    Task(
        task_id="arima_forecast",
        title="ARIMA forecast",
        prompt=(
            "Fit ARIMA(1,1,1) to the level series y by maximum likelihood, "
            "including no constant or drift term. Report the one-step-ahead "
            "forecast of y in levels for t = 251 and the 95 percent prediction "
            "interval bounds as rows named forecast, lower_95, and upper_95."
        ),
        estimands=["forecast", "lower_95", "upper_95"],
        reference_key="arima_forecast",
        data_file="pilot_v4_arima.csv",
        data_description="Dataset: pilot_v4_arima.csv. Columns: t, y.",
    ),
    Task(
        task_id="cluster_ols",
        title="One-way clustered OLS",
        prompt=(
            "Estimate y_panel on x1 and x2 with an intercept. Use cluster-robust "
            "standard errors clustered on cluster. Report rows const, x1, and x2."
        ),
        estimands=["const", "x1", "x2"],
        reference_key="cluster_ols",
        data_file="pilot_v2.csv",
        data_description=PILOT_V2,
    ),
    Task(
        task_id="two_way_cluster",
        title="Two-way clustered OLS",
        prompt=(
            "Estimate y_twocluster on x with an intercept. Use two-way "
            "cluster-robust standard errors clustered on firm and year using the "
            "Cameron-Gelbach-Miller formula. Report rows const and x."
        ),
        estimands=["const", "x"],
        reference_key="two_way_cluster",
        data_file="pilot_v3.csv",
        data_description=PILOT_V3,
    ),
    Task(
        task_id="vif",
        title="Variance inflation factors",
        prompt=(
            "For the linear model y on x1, x2, x3, x4, and x5, compute the VIF "
            "for each regressor. Report rows vif_x1, vif_x2, vif_x3, vif_x4, "
            "and vif_x5."
        ),
        estimands=["vif_x1", "vif_x2", "vif_x3", "vif_x4", "vif_x5"],
        reference_key="vif",
        data_file="pilot_v5_collinear.csv",
        data_description=PILOT_V5_COL,
    ),
    Task(
        task_id="cooks_distance",
        title="Cook's distance",
        prompt=(
            "Fit y on x1, x2, x3, x4, and x5 with an intercept. Compute Cook's "
            "distance for each row and report the top five row indices and values "
            "as rank1_obs_id, rank1_cooks_d, ..., rank5_obs_id, rank5_cooks_d. "
            "Use zero-based row indices."
        ),
        estimands=[
            "rank1_obs_id",
            "rank1_cooks_d",
            "rank2_obs_id",
            "rank2_cooks_d",
            "rank3_obs_id",
            "rank3_cooks_d",
            "rank4_obs_id",
            "rank4_cooks_d",
            "rank5_obs_id",
            "rank5_cooks_d",
        ],
        reference_key="cooks_distance",
        data_file="pilot_v5_collinear.csv",
        data_description=PILOT_V5_COL,
    ),
    Task(
        task_id="ttest_welch",
        title="Welch t-test",
        prompt=(
            "Compare y between group 1 and group 0 using Welch's "
            "unequal-variance two-sample t-test. Define the t statistic as "
            "mean(y | group=1) minus mean(y | group=0), divided by the Welch "
            "standard error. Use the Welch-Satterthwaite approximation for "
            "the degrees of freedom. Report rows t, df, and p_value."
        ),
        estimands=["t", "df", "p_value"],
        reference_key="ttest_welch",
        data_file="pilot_v5_ttest.csv",
        data_description="Dataset: pilot_v5_ttest.csv. Columns: group, y.",
    ),
    Task(
        task_id="anova_oneway",
        title="One-way ANOVA",
        prompt=(
            "Run a one-way ANOVA of y by group. Report the omnibus F statistic, "
            "numerator degrees of freedom, denominator degrees of freedom, and "
            "p-value as rows F, df_num, df_den, and p_value."
        ),
        estimands=["F", "df_num", "df_den", "p_value"],
        reference_key="anova_oneway",
        data_file="pilot_v5_anova.csv",
        data_description="Dataset: pilot_v5_anova.csv. Columns: group, y.",
    ),
    Task(
        task_id="chisq_indep",
        title="Chi-square independence test",
        prompt=(
            "Run Pearson's chi-square test of independence for row by col. "
            "Do not use Yates correction. Report rows chi2, df, and p_value."
        ),
        estimands=["chi2", "df", "p_value"],
        reference_key="chisq_indep",
        data_file="pilot_v5_chisq.csv",
        data_description="Dataset: pilot_v5_chisq.csv. Columns: row, col.",
    ),
    Task(
        task_id="logit",
        title="Logit regression",
        prompt=(
            "Estimate a binary logit of y_bin_logit on x1 and x2 with an "
            "intercept. Report coefficient-scale rows const, x1, and x2 with "
            "asymptotic standard errors."
        ),
        estimands=["const", "x1", "x2"],
        reference_key="logit",
        data_file="pilot.csv",
        data_description=PILOT_V1,
    ),
    Task(
        task_id="probit",
        title="Probit regression",
        prompt=(
            "Estimate a binary probit of y_bin_probit on x1 and x2 with an "
            "intercept. Report coefficient-scale rows const, x1, and x2 with "
            "asymptotic standard errors."
        ),
        estimands=["const", "x1", "x2"],
        reference_key="probit",
        data_file="pilot.csv",
        data_description=PILOT_V1,
    ),
    Task(
        task_id="poisson_offset",
        title="Poisson regression",
        prompt=(
            "Estimate a Poisson regression of y_count on x1 and x2 with an "
            "intercept and an offset equal to the natural log of exposure. "
            "Report rows const, x1, and x2 with asymptotic model-based standard "
            "errors."
        ),
        estimands=["const", "x1", "x2"],
        reference_key="poisson_offset",
        data_file="pilot_v2.csv",
        data_description=PILOT_V2,
    ),
    Task(
        task_id="cox_hazard",
        title="Cox regression",
        prompt=(
            "Fit a Cox proportional-hazards model for (t_obs, event) on treat "
            "and x. Report log-hazard-ratio rows treat and x with model-based "
            "standard errors."
        ),
        estimands=["treat", "x"],
        reference_key="cox_hazard",
        data_file="pilot_v4_cox.csv",
        data_description=(
            "Dataset: pilot_v4_cox.csv. Columns: subject, treat, x, t_obs, event."
        ),
    ),
    Task(
        task_id="twoway_fe",
        title="Two-way fixed effects",
        prompt=(
            "Estimate y_twocluster on x with firm and year fixed effects. Report "
            "the x coefficient with a standard error clustered on firm as row x."
        ),
        estimands=["x"],
        reference_key="twoway_fe",
        data_file="pilot_v3.csv",
        data_description=PILOT_V3,
    ),
    Task(
        task_id="iv2sls_robust",
        title="Two-stage least squares",
        prompt=(
            "Estimate y_iv on endogenous x1_iv with an intercept, using z as the "
            "excluded instrument. Use a proper 2SLS routine with HC1 robust "
            "standard errors. Report rows const and x1_iv."
        ),
        estimands=["const", "x1_iv"],
        reference_key="iv2sls_robust",
        data_file="pilot_v2.csv",
        data_description=PILOT_V2,
    ),
    Task(
        task_id="dynpanel_ab",
        title="Arellano-Bond GMM",
        prompt=(
            "Estimate the one-step Arellano-Bond difference GMM dynamic panel model "
            "for y on its first lag and x (equivalent to Stata `xtabond y x, lags(1)` "
            "or R `plm::pgmm(y ~ lag(y, 1) + x | lag(y, 2:99), "
            'effect = "individual", model = "onestep", transformation = "d")`): '
            "first-difference the equation, use GMM-type instruments (all available "
            "lags of y from t-2 and earlier, one instrument column per lag and "
            "period; do not collapse the instrument matrix) for the lagged dependent "
            "variable, and use x in first differences as a standard instrument. Use "
            "the standard one-step weighting matrix built from the first-difference "
            "operator. Do not use two-step GMM, system GMM, or time dummies. Report "
            "coefficient rows rho for the lagged dependent variable and beta_x for x. "
            "Standard errors are not part of this benchmark task; set se to null for "
            "both rows."
        ),
        estimands=["rho", "beta_x"],
        reference_key="dynpanel_ab",
        data_file="pilot_v5_dynpanel.csv",
        data_description="Dataset: pilot_v5_dynpanel.csv. Columns: firm, year, y, x.",
    ),
    Task(
        task_id="staggered_did",
        title="Staggered DiD",
        prompt=(
            "Compute ATT_uniform directly from group-year means as the "
            "equal-weight average over supported post-treatment cohort-time "
            "cells for cohorts 4, 6, and 8. Never-treated firms carry "
            "cohort = -1; the treated cohorts carry the year of first "
            "treatment. The year variable runs from 0 to 9 with no gaps, "
            "and every firm is observed in every year. For each cell (g,t), use "
            "[mean(y_did | cohort=g, year=t) - mean(y_did | cohort=g, "
            "year=g-1)] minus [mean(y_did | never-treated, year=t) - "
            "mean(y_did | never-treated, year=g-1)]. Do not run TWFE, "
            "package DiD aggregators, bootstrap, or standard-error routines. "
            "Report one row named att with se set to null."
        ),
        estimands=["att"],
        reference_key="staggered_did",
        data_file="pilot_v3.csv",
        data_description=PILOT_V3,
    ),
    Task(
        task_id="rdd_sharp",
        title="Regression discontinuity",
        prompt=(
            "Using bandwidth h = 0.5 and triangular kernel weights, fit local "
            "linear regressions on each side of cutoff z = 0 and report the "
            "difference in intercepts as row late."
        ),
        estimands=["late"],
        reference_key="rdd_sharp",
        data_file="pilot_v5_rdd.csv",
        data_description="Dataset: pilot_v5_rdd.csv. Columns: z, d, y.",
    ),
    Task(
        task_id="lp_irf",
        title="Local-projection impulse response",
        prompt=(
            "Estimate the local projection y_{t+4} on shock_t and y_{t-1}, with "
            "an intercept, using Newey-West standard error with lag 4. Report "
            "row irf_h4."
        ),
        estimands=["irf_h4"],
        reference_key="lp_irf",
        data_file="pilot_v5_lp.csv",
        data_description="Dataset: pilot_v5_lp.csv. Columns: t, y, shock.",
    ),
    Task(
        task_id="wcb_few",
        title="Wild cluster bootstrap",
        prompt=(
            "Estimate y on x with an intercept and cluster-robust CR1 standard "
            "error clustered on cluster, using the finite-sample correction "
            "G/(G-1) x (N-1)/(N-K). Also compute the restricted wild cluster "
            "bootstrap p-value for H0: beta_x = 0 with Rademacher weights: impose "
            "the null by regressing y on the intercept only, and enumerate all "
            "2^5 = 32 cluster sign patterns instead of random resampling. The "
            "p-value is the fraction of the 32 bootstrap t-statistics with "
            "|t*| >= |t_obs|, counting a pattern as extreme when "
            "|t*| >= |t_obs| - 1e-9. Report rows x, t_obs, and p_wcb."
        ),
        estimands=["x", "t_obs", "p_wcb"],
        reference_key="wcb_few",
        data_file="pilot_v5_wcb.csv",
        data_description="Dataset: pilot_v5_wcb.csv. Columns: cluster, x, y.",
    ),
    Task(
        task_id="var_irf",
        title="VAR impulse response",
        prompt=(
            "Fit a two-variable VAR(2) with a constant. Under Cholesky "
            "identification with y1 ordered first, report the orthogonal IRF of "
            "y1 to a one-standard-deviation shock in y2 at horizon 4 as row "
            "irf_y1_to_y2_h4."
        ),
        estimands=["irf_y1_to_y2_h4"],
        reference_key="var_irf",
        data_file="pilot_v4_var.csv",
        data_description=PILOT_V4_VAR,
    ),
    Task(
        task_id="var_granger",
        title="Granger causality",
        prompt=(
            "Test whether y2 Granger-causes y1 with two lags using the aligned "
            "OLS F-test. Report rows F, df_num, df_den, and p_value."
        ),
        estimands=["F", "df_num", "df_den", "p_value"],
        reference_key="var_granger",
        data_file="pilot_v4_var.csv",
        data_description=PILOT_V4_VAR,
    ),
]


EXPLORATORY_TASKS: list[Task] = [
    Task(
        task_id="bp_koenker",
        title="Breusch-Pagan test",
        prompt=(
            "Run OLS of y_het on x1 and x2 with an intercept. Test for "
            "heteroskedasticity with the Breusch-Pagan test in the Koenker "
            "studentized LM form: regress the squared OLS residuals on x1 and x2 "
            "with an intercept, and compute LM = n times the R-squared of that "
            "auxiliary regression. The statistic is chi-squared with 2 degrees of "
            "freedom. Report rows named bp_lm, bp_df, and bp_p."
        ),
        estimands=["bp_lm", "bp_df", "bp_p"],
        reference_key="bp_koenker",
        data_file="pilot_v6_bp_koenker.csv",
        data_description=(
            "Dataset: pilot_v6_bp_koenker.csv. Columns: id, x1, x2, y_het."
        ),
    ),
    Task(
        task_id="synth_weights_fixedv",
        title="Synthetic control weights",
        prompt=(
            "Columns: unit (0 = treated unit; 1-8 = donor units), period (1-17), "
            "y; periods 1-12 are pre-treatment. Compute synthetic-control weights "
            "for the treated unit: choose non-negative weights on the eight "
            "donors, summing to one, that minimize the sum of squared differences "
            "between the treated unit's twelve pre-treatment outcomes and the "
            "weighted combination of the donors' pre-treatment outcomes "
            "(equivalently, the Abadie estimator with all twelve pre-period "
            "outcome values as predictors and the identity predictor-weight "
            "matrix V = I). Report estimate-only rows w_3 and w_7 (the weights on "
            "donors 3 and 7), rmspe_pre (the root mean squared pre-treatment "
            "prediction error of the synthetic control), and att_post (the "
            "average of treated minus synthetic over periods 13-17). Set se to "
            "null on all rows."
        ),
        estimands=["w_3", "w_7", "rmspe_pre", "att_post"],
        reference_key="synth_weights_fixedv",
        data_file="pilot_v6_synth.csv",
        data_description="Dataset: pilot_v6_synth.csv. Columns: unit, period, y.",
    ),
]

ALL_TASKS: list[Task] = [*TASKS, *EXPLORATORY_TASKS]


def all_cells() -> list[tuple[Task, str, str, str]]:
    return [
        (t, p, prompt, feedback)
        for t in TASKS
        for p in PLATFORMS
        for prompt in PROMPT_MODES
        for feedback in EXEC_MODES
    ]

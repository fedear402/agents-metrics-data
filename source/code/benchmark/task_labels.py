"""Reader-facing task labels for the paper, its figures, and its appendix tables.

Each label names the econometric method only. It never names the simulated
application setting, because every task uses simulated data. The same label
appears in Figure 3, the appendix task tables, and the text. Change a label
here only, then rebuild the figures and edit the appendix tables to match.
"""
from __future__ import annotations

TASK_LABELS = {
    'anova_oneway': 'One-way ANOVA',
    'applied_cumulative_hac': 'Cumulative lag effects (HAC)',
    'applied_fuzzy_rd': 'Fuzzy regression discontinuity',
    'applied_multivintage_trade_iv': 'Multi-vintage exposure IV',
    'applied_nonrecipient_population_iv': 'Weighted 2SLS spillover effects',
    'applied_shift_share': 'Shift-share IV',
    'arima_forecast': 'ARIMA forecast',
    'broad_ar_confidence_set': 'Anderson–Rubin confidence set',
    'broad_cohort_stack': 'Cohort-specific difference-in-differences',
    'broad_multinomial_logit': 'Multinomial logit',
    'broad_oaxaca': 'Oaxaca decomposition',
    'broad_ordered_logit': 'Ordered logit',
    'broad_tobit': 'Tobit regression',
    'chisq_indep': 'Chi-square independence test',
    'cluster_ols': 'One-way clustered OLS',
    'cooks_distance': "Cook's distance",
    'cox_hazard': 'Cox proportional hazards',
    'hard_conditional_logit_welfare': 'Conditional logit welfare',
    'hard_heteroskedastic_probit': 'Heteroskedastic probit',
    'hard_panel_lp_iv': 'Panel local-projection IV',
    'hard_poisson_hurdle_effects': 'Poisson hurdle effects',
    'hard_rolling': 'Rolling AR(1) forecasts',
    'hard_truncated_normal_effects': 'Truncated normal regression',
    'iv2sls_robust': 'Two-stage least squares',
    'logit': 'Logit regression',
    'lp04_ppml': 'PPML with separated zeros',
    'lp11_duration': 'Discrete-time conditional logit',
    'lp_irf': 'Local-projection impulse response',
    'ols_robust': 'OLS with robust standard errors',
    'pilot_probit_interaction_ame': 'Probit interaction effects',
    'poisson_offset': 'Poisson regression',
    'probit': 'Probit regression',
    'rdd_sharp': 'Sharp regression discontinuity',
    'staggered_did': 'Staggered difference-in-differences',
    'ttest_welch': 'Welch t-test',
    'two_way_cluster': 'Two-way clustered OLS',
    'twoway_fe': 'Two-way fixed effects',
    'var_granger': 'Granger causality',
    'var_irf': 'VAR impulse response',
    'vif': 'Variance inflation factors',
}

# Earlier tasks that stay readable in historical builds. The current paper does not use them.
HISTORICAL_LABELS = {
    'd1_attendance': 'Dynamic attendance choice',
}


def task_label(task: str) -> str:
    """Return the paper label for one task, or fail when the task has none."""
    try:
        return TASK_LABELS[task]
    except KeyError:
        raise KeyError(f'No paper label for task {task!r}; add one to task_labels.py') from None


def tex_label(task: str) -> str:
    """Return the label with the LaTeX en dash, for hand-maintained tables."""
    return task_label(task).replace('–', '--')

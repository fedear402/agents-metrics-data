"""Canonical task display names, groups, and display order.

This module is the single source of truth for how a task is named and ordered
on every reader-facing surface: the project website, the README, the generated
task inventory, and the manuscript figures. Decision record
``quality_reports/decision_records/2026-08-24_presentation-renames.md`` adopted
these names, groups, and order on 2026-08-24.

Two rules govern this file.

1. Internal identifiers never change. ``task_id`` values, run-record directory
   names, and analysis data keys stay exactly as the stored corpus wrote them.
   This module maps an identifier to a display string; it never renames one.
2. No display string carries a task number. The old ``T01`` style numbering is
   gone from every display surface.

Import it from ``code/benchmark`` code as ``import task_display`` or
``from benchmark import task_display``. Site and exhibit builders put
``code/benchmark`` on ``sys.path`` and import it as ``task_display``.

``ANALYZED`` and its helpers describe the earlier 21-task sample and serve the
historical exhibits. ``CURRENT`` and the ``current_*`` helpers describe the 40
tasks of the current paper: their names come from ``task_labels.py`` and their
groups follow the method families of the paper. The website uses ``CURRENT``.

The module imports only ``task_labels`` (lazily), so any builder can import it cheaply.
"""

from __future__ import annotations

import re

# Groups in reading order. Position in this tuple is the group order.
GROUP_ORDER: tuple[str, ...] = (
    "Binary regression",
    "OLS",
    "Difference-in-differences",
    "Instrumental variables",
    "Regression discontinuity",
    "Time series",
    "Poisson regression",
    "Other",
)

# The 21 analyzed tasks, in display order: group by group, and inside each
# group in the order the decision record fixed. Position in this tuple is the
# global display order and, within a group, the within-group order.
#
# (task_id, display name, group)
ANALYZED: tuple[tuple[str, str, str], ...] = (
    ("probit", "Probit regression", "Binary regression"),
    ("logit", "Logit regression", "Binary regression"),
    ("ols_robust", "OLS with robust standard errors", "OLS"),
    ("cluster_ols", "One-way clustered OLS", "OLS"),
    ("two_way_cluster", "Two-way clustered OLS", "OLS"),
    ("staggered_did", "Staggered DiD", "Difference-in-differences"),
    ("twoway_fe", "Two-way fixed effects", "Difference-in-differences"),
    ("iv2sls_robust", "Two-stage least squares", "Instrumental variables"),
    ("rdd_sharp", "Regression discontinuity", "Regression discontinuity"),
    ("arima_forecast", "ARIMA forecast", "Time series"),
    ("var_granger", "Granger causality", "Time series"),
    ("var_irf", "VAR impulse response", "Time series"),
    ("lp_irf", "Local-projection impulse response", "Time series"),
    ("poisson_offset", "Poisson regression", "Poisson regression"),
    ("cooks_distance", "Cook's distance", "Other"),
    ("vif", "Variance inflation factors", "Other"),
    ("dynpanel_ab", "Arellano-Bond GMM", "Other"),
    ("ttest_welch", "Welch t-test", "Other"),
    ("anova_oneway", "One-way ANOVA", "Other"),
    ("chisq_indep", "Chi-square independence test", "Other"),
    ("cox_hazard", "Cox regression", "Other"),
)

# Tasks that carry a display name but stay outside the analyzed 21. They are
# defined in tasks.py and must never appear in a reader-facing task list.
#
# (task_id, display name, group)
NOT_ANALYZED: tuple[tuple[str, str, str], ...] = (
    ("wcb_few", "Wild cluster bootstrap", "Other"),
    ("bp_koenker", "Breusch-Pagan test", "Other"),
    ("synth_weights_fixedv", "Synthetic control weights", "Other"),
)

_ALL = ANALYZED + NOT_ANALYZED

DISPLAY_NAME: dict[str, str] = {
    task_id: name for task_id, name, _group in _ALL
}
GROUP: dict[str, str] = {task_id: group for task_id, _name, group in _ALL}

ANALYZED_TASK_IDS: tuple[str, ...] = tuple(
    task_id for task_id, _name, _group in ANALYZED
)

_DISPLAY_ORDER: dict[str, int] = {
    task_id: index + 1 for index, task_id in enumerate(ANALYZED_TASK_IDS)
}

_ORDER_IN_GROUP: dict[str, int] = {}
_seen_per_group: dict[str, int] = {}
for _task_id, _name, _group in ANALYZED:
    _seen_per_group[_group] = _seen_per_group.get(_group, 0) + 1
    _ORDER_IN_GROUP[_task_id] = _seen_per_group[_group]

_TASK_NUMBER_RE = re.compile(r"\bT\d{2}\b")


def display_name(task_id: str) -> str:
    """Return the reader-facing name of one task."""
    return DISPLAY_NAME[task_id]


def group_of(task_id: str) -> str:
    """Return the display group of one task."""
    return GROUP[task_id]


def group_order(group: str) -> int:
    """Return the 1-based reading position of one group."""
    return GROUP_ORDER.index(group) + 1


def group_order_of(task_id: str) -> int:
    """Return the 1-based reading position of a task's group."""
    return group_order(GROUP[task_id])


def order_in_group(task_id: str) -> int:
    """Return the 1-based position of one analyzed task inside its group."""
    return _ORDER_IN_GROUP[task_id]


def display_order(task_id: str) -> int:
    """Return the 1-based global display position of one analyzed task."""
    return _DISPLAY_ORDER[task_id]


def is_analyzed(task_id: str) -> bool:
    """Report whether one task belongs to the analyzed set."""
    return task_id in _DISPLAY_ORDER


def ordered_task_ids() -> list[str]:
    """Return the analyzed task identifiers in display order."""
    return list(ANALYZED_TASK_IDS)


def sort_key(task_id: str) -> tuple[int, int]:
    """Return a sort key that puts tasks in group order, then group position."""
    return (group_order_of(task_id), order_in_group(task_id))


def grouped_tasks() -> list[tuple[str, list[str]]]:
    """Return the analyzed tasks as (group, task identifiers) in reading order."""
    buckets: dict[str, list[str]] = {group: [] for group in GROUP_ORDER}
    for task_id in ANALYZED_TASK_IDS:
        buckets[GROUP[task_id]].append(task_id)
    return [(group, buckets[group]) for group in GROUP_ORDER if buckets[group]]


# --------------------------------------------------------------------------- #
# The 40 tasks of the current paper
# --------------------------------------------------------------------------- #
# The tables above describe the earlier 21-task sample and stay for the
# historical exhibits. The website and every other surface of the current
# paper use the tables below. Names come from task_labels.py, which also feeds
# Figure 3 and the appendix task table. Groups follow the method families of
# the paper's introduction.

CURRENT_GROUP_ORDER: tuple[str, ...] = (
    "Linear regression",
    "Classical tests",
    "Binary response",
    "Poisson and count models",
    "Censored and truncated models",
    "Discrete choice",
    "Difference-in-differences",
    "Instrumental variables",
    "Regression discontinuity",
    "Survival analysis",
    "Time series",
    "Decomposition",
)

# (task_id, group, search keywords). Position in this tuple is the reading order.
CURRENT: tuple[tuple[str, str, str], ...] = (
    ("ols_robust", "Linear regression", "OLS heteroskedasticity-robust HC1 standard errors"),
    ("cluster_ols", "Linear regression", "OLS cluster-robust standard errors"),
    ("two_way_cluster", "Linear regression", "OLS two-way clustering Cameron Gelbach Miller"),
    ("cooks_distance", "Linear regression", "influence diagnostics outliers"),
    ("vif", "Linear regression", "multicollinearity diagnostics"),
    ("ttest_welch", "Classical tests", "two-sample t test unequal variances"),
    ("anova_oneway", "Classical tests", "analysis of variance F test"),
    ("chisq_indep", "Classical tests", "contingency table Pearson chi-square"),
    ("logit", "Binary response", "logistic regression"),
    ("probit", "Binary response", "probit maximum likelihood"),
    ("pilot_probit_interaction_ame", "Binary response", "probit average marginal effects interaction"),
    ("hard_heteroskedastic_probit", "Binary response", "probit heteroskedasticity scale"),
    ("poisson_offset", "Poisson and count models", "Poisson exposure offset count"),
    ("lp04_ppml", "Poisson and count models", "Poisson pseudo maximum likelihood gravity trade fixed effects separation"),
    ("hard_poisson_hurdle_effects", "Poisson and count models", "Poisson hurdle zeros count"),
    ("broad_tobit", "Censored and truncated models", "Tobit censoring"),
    ("hard_truncated_normal_effects", "Censored and truncated models", "truncated regression conditional mean"),
    ("broad_multinomial_logit", "Discrete choice", "multinomial logit choice probabilities"),
    ("broad_ordered_logit", "Discrete choice", "ordered logit cut points"),
    ("hard_conditional_logit_welfare", "Discrete choice", "conditional logit welfare logsum"),
    ("staggered_did", "Difference-in-differences", "staggered adoption Callaway Sant'Anna"),
    ("broad_cohort_stack", "Difference-in-differences", "stacked cohort event study"),
    ("twoway_fe", "Difference-in-differences", "two-way fixed effects panel"),
    ("iv2sls_robust", "Instrumental variables", "IV 2SLS robust standard errors"),
    ("applied_shift_share", "Instrumental variables", "Bartik exposure shares shocks"),
    ("applied_multivintage_trade_iv", "Instrumental variables", "trade exposure IV employment shares"),
    ("applied_nonrecipient_population_iv", "Instrumental variables", "weighted 2SLS spillovers"),
    ("broad_ar_confidence_set", "Instrumental variables", "weak instruments Anderson-Rubin test inversion"),
    ("hard_panel_lp_iv", "Instrumental variables", "local projections IV panel impulse response"),
    ("rdd_sharp", "Regression discontinuity", "RD local linear triangular kernel"),
    ("applied_fuzzy_rd", "Regression discontinuity", "fuzzy RD Wald ratio first stage"),
    ("cox_hazard", "Survival analysis", "proportional hazards duration"),
    ("lp11_duration", "Survival analysis", "discrete-time hazard conditional logit duration"),
    ("arima_forecast", "Time series", "ARIMA forecast"),
    ("var_granger", "Time series", "VAR Granger causality test"),
    ("var_irf", "Time series", "VAR impulse response"),
    ("lp_irf", "Time series", "local projections impulse response Newey-West"),
    ("applied_cumulative_hac", "Time series", "distributed lags cumulative effect HAC Newey-West"),
    ("hard_rolling", "Time series", "rolling window AR(1) forecasts"),
    ("broad_oaxaca", "Decomposition", "Oaxaca-Blinder wage gap"),
)

CURRENT_TASK_IDS: tuple[str, ...] = tuple(task_id for task_id, _group, _keywords in CURRENT)
CURRENT_GROUP: dict[str, str] = {task_id: group for task_id, group, _keywords in CURRENT}
CURRENT_KEYWORDS: dict[str, str] = {task_id: keywords for task_id, _group, keywords in CURRENT}
_CURRENT_ORDER: dict[str, int] = {task_id: index + 1 for index, task_id in enumerate(CURRENT_TASK_IDS)}
_CURRENT_ORDER_IN_GROUP: dict[str, int] = {}
_seen_current: dict[str, int] = {}
for _task_id, _group, _keywords in CURRENT:
    _seen_current[_group] = _seen_current.get(_group, 0) + 1
    _CURRENT_ORDER_IN_GROUP[_task_id] = _seen_current[_group]


def current_name(task_id: str) -> str:
    """Return the paper's name of one task of the current sample."""
    from task_labels import task_label

    return task_label(task_id)


def current_group(task_id: str) -> str:
    """Return the method group of one task of the current sample."""
    return CURRENT_GROUP[task_id]


def current_group_order(task_id: str) -> int:
    """Return the 1-based reading position of a current task's group."""
    return CURRENT_GROUP_ORDER.index(CURRENT_GROUP[task_id]) + 1


def current_order(task_id: str) -> int:
    """Return the 1-based reading position of one current task."""
    return _CURRENT_ORDER[task_id]


def current_order_in_group(task_id: str) -> int:
    """Return the 1-based position of one current task inside its group."""
    return _CURRENT_ORDER_IN_GROUP[task_id]


def current_sort_key(task_id: str) -> tuple[int, int]:
    """Sort key that puts current tasks in group order, then group position."""
    return (current_group_order(task_id), current_order_in_group(task_id))


def _check() -> None:
    """Validate the table at import time, so a bad edit fails loudly."""
    if len(CURRENT) != 40 or len(set(CURRENT_TASK_IDS)) != 40:
        raise ValueError("task_display: the current sample must list 40 distinct tasks")
    for task_id, group, _keywords in CURRENT:
        if group not in CURRENT_GROUP_ORDER:
            raise ValueError(f"task_display: unknown current group {group!r} for {task_id}")
    current_positions = [CURRENT_GROUP_ORDER.index(group) for _task_id, group, _kw in CURRENT]
    if current_positions != sorted(current_positions):
        raise ValueError("task_display: current tasks are not in group order")
    ids = [task_id for task_id, _name, _group in _ALL]
    if len(ids) != len(set(ids)):
        raise ValueError("task_display: duplicate task identifier")
    if len(ANALYZED) != 21:
        raise ValueError(f"task_display: expected 21 analyzed tasks, got {len(ANALYZED)}")
    names = [name for _task_id, name, _group in ANALYZED]
    if len(names) != len(set(names)):
        raise ValueError("task_display: duplicate display name")
    for task_id, name, group in _ALL:
        if group not in GROUP_ORDER:
            raise ValueError(f"task_display: unknown group {group!r} for {task_id}")
        if _TASK_NUMBER_RE.search(name):
            raise ValueError(f"task_display: display name keeps a task number: {name!r}")
    positions = [GROUP_ORDER.index(group) for _task_id, _name, group in ANALYZED]
    if positions != sorted(positions):
        raise ValueError("task_display: analyzed tasks are not in group order")


_check()

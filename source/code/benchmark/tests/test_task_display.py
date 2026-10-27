"""Contract tests for the canonical task display names, groups, and order.

`code/benchmark/task_display.py` is the single source of truth. Every other
display surface must agree with it, and no display surface may carry a task
number.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "code" / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

import task_display  # noqa: E402
from tasks import ALL_TASKS  # noqa: E402

TASK_NUMBER_RE = re.compile(r"\bT\d{2}\b")

EXPECTED_GROUPS = (
    "Binary regression",
    "OLS",
    "Difference-in-differences",
    "Instrumental variables",
    "Regression discontinuity",
    "Time series",
    "Poisson regression",
    "Other",
)

# The order the decision record fixed on 2026-08-24.
EXPECTED_ORDER = (
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


def test_groups_and_order_match_the_decision_record() -> None:
    assert task_display.GROUP_ORDER == EXPECTED_GROUPS
    assert task_display.ANALYZED == EXPECTED_ORDER
    assert len(task_display.ANALYZED_TASK_IDS) == 21


def test_display_positions_are_dense_and_grouped() -> None:
    assert [
        task_display.display_order(task_id)
        for task_id in task_display.ANALYZED_TASK_IDS
    ] == list(range(1, 22))
    for group, task_ids in task_display.grouped_tasks():
        assert [task_display.order_in_group(t) for t in task_ids] == list(
            range(1, len(task_ids) + 1)
        )
        assert {task_display.group_of(t) for t in task_ids} == {group}
    assert [group for group, _ids in task_display.grouped_tasks()] == list(
        EXPECTED_GROUPS
    )


def test_no_display_name_carries_a_task_number() -> None:
    for name in task_display.DISPLAY_NAME.values():
        assert TASK_NUMBER_RE.search(name) is None


def test_task_definitions_use_the_canonical_display_names() -> None:
    for task in ALL_TASKS:
        assert task.title == task_display.DISPLAY_NAME[task.task_id], task.task_id


def test_exported_registry_agrees_with_the_module() -> None:
    registry = json.loads((BENCHMARK_DIR / "task_registry.json").read_text())
    for entry in registry["tasks"]:
        task_id = entry["task_id"]
        assert entry["title"] == task_display.display_name(task_id)
        assert entry["display_name"] == task_display.display_name(task_id)
        assert entry["display_group"] == task_display.group_of(task_id)
        assert entry["analyzed"] == task_display.is_analyzed(task_id)
        if task_display.is_analyzed(task_id):
            assert entry["display_order"] == task_display.display_order(task_id)
    assert TASK_NUMBER_RE.search(json.dumps(registry)) is None


def test_exhibit_labels_come_from_the_module() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "exhibits"))
    import style_app  # noqa: PLC0415

    assert style_app.SHORT_TASK_LABELS == {
        task_id: task_display.display_name(task_id)
        for task_id in task_display.ANALYZED_TASK_IDS
    }
    assert style_app.TASK_ORDER == list(task_display.ANALYZED_TASK_IDS)


def test_current_sample_lists_the_forty_paper_tasks_in_group_order() -> None:
    """The website's 40 tasks: the paper's names, method groups, and reading order."""
    from task_labels import TASK_LABELS  # noqa: PLC0415

    catalog = json.loads((PROJECT_ROOT / "data/analysis/main/data/catalog.json").read_text())
    assert set(task_display.CURRENT_TASK_IDS) == set(catalog) == set(TASK_LABELS)
    for task_id in task_display.CURRENT_TASK_IDS:
        assert task_display.current_name(task_id) == TASK_LABELS[task_id]
        assert task_display.current_group(task_id) in task_display.CURRENT_GROUP_ORDER
        assert task_display.CURRENT_KEYWORDS[task_id]
    keys = [task_display.current_sort_key(task_id) for task_id in task_display.CURRENT_TASK_IDS]
    assert keys == sorted(keys)
    assert "Other" not in task_display.CURRENT_GROUP_ORDER

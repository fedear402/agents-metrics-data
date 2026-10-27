"""Tests for the Appendix D robustness table (code/benchmark/paper_robustness.py).

The fixture computes the table once from data/analysis/main and keeps the token cache in a
temporary folder, so the tests write nothing under data/analysis/main.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "benchmark"))

import paper_robustness as pr  # noqa: E402

ANALYSIS = PROJECT_ROOT / "data" / "analysis" / "main"
MANUSCRIPT = PROJECT_ROOT / "paper" / "manuscript"
INSTALLED = MANUSCRIPT / "tables" / "appendix_robustness.tex"
INSTALLED_DATASET = MANUSCRIPT / "tables" / "appendix_table1_pvalues_dataset.tex"
DATASET_BOOT = ANALYSIS / "data" / "inference_dataset"

#: The nine tasks whose primary outcome scores the standard error (decision record
#: 2026-09-25_se-scored-primary-outcome), written out here.
SE_TASKS = {
    "applied_cumulative_hac", "applied_fuzzy_rd", "applied_shift_share", "cluster_ols", "iv2sls_robust",
    "lp_irf", "ols_robust", "two_way_cluster", "twoway_fe",
}

requires_analysis = pytest.mark.skipif(
    not (ANALYSIS / "data" / "runs.csv").exists(), reason="data/analysis/main is not present"
)

#: Printed row labels, written out here so that the test does not read the generator's list.
EXPECTED_ROWS = (
    "Constrained agent vs chatbot",
    "Self-repair vs chatbot",
    "One-shot vs zero-shot",
    r"One-shot $\times$ constrained agent",
    "Stata vs Python",
    "Sonnet 4.6 minus Luna",
)

#: The same contrasts as the table prints them, over two lines.
PRINTED_LABELS = (
    ("Constrained agent", "vs chatbot"),
    ("Self-repair", "vs chatbot"),
    ("One-shot", "vs zero-shot"),
    (r"One-shot $\times$", "constrained agent"),
    ("Stata vs Python", "zero-shot chatbot"),
    ("Sonnet 4.6 minus", "Luna, all runs"),
)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    work = tmp_path_factory.mktemp("robustness_work")
    first = pr.compute(ANALYSIS, work)
    second = pr.compute(ANALYSIS, work)
    return first, second


def direct_contrasts() -> dict[str, float]:
    """Pooled contrasts from runs.csv cell means, without the task-level identity."""
    d = pd.read_csv(ANALYSIS / "data" / "runs.csv", usecols=["model", "platform", "prompt", "agency", "y"])
    m = lambda **kw: 100 * d.loc[pd.concat([d[k].eq(v) for k, v in kw.items()], axis=1).all(axis=1), "y"].mean()  # noqa: E731
    fs = {a: m(agency=a, prompt="few_shot") - m(agency=a, prompt="zero_shot") for a in ("chatbot", "agent")}
    return {
        "H3_tool_loop": m(agency="agent") - m(agency="chatbot"),
        "H3_self_repair": m(agency="self_repair") - m(agency="chatbot"),
        "H2_fewshot": m(prompt="few_shot") - m(prompt="zero_shot"),
        "H4_tool_loop": fs["agent"] - fs["chatbot"],
        "H1_stata": m(platform="stata", prompt="zero_shot", agency="chatbot")
        - m(platform="python", prompt="zero_shot", agency="chatbot"),
        "MODEL_gap": m(model="sonnet") - m(model="luna"),
    }


@requires_analysis
def test_reproduces_table1_point_estimates(result):
    table = result[0]["table"].set_index("contrast")
    for contrast, value in direct_contrasts().items():
        assert table.loc[contrast, "est"] == pytest.approx(value, abs=1e-9), contrast
    ref = pd.read_csv(ANALYSIS / "tables" / "regression_contrasts.csv").set_index(["model", "contrast"])
    for contrast in pr.TABLE1:
        assert table.loc[contrast, "est"] == pytest.approx(ref.loc[("pooled", contrast), "estimate_pp"], abs=1e-9)
        assert table.loc[contrast, "se"] == pytest.approx(ref.loc[("pooled", contrast), "se_cr1_pp"], abs=1e-9)
    assert result[0]["info"]["table1_max_abs_diff"] < 1e-9


@requires_analysis
def test_table_has_expected_rows_and_columns(result):
    table, info = result[0]["table"], result[0]["info"]
    assert tuple(table.label) == EXPECTED_ROWS
    tex = pr.render_table(table, info)
    # Two panels in one float, each its own tabular* at the text width, with no scaling.
    assert r"\resizebox" not in tex
    assert tex.count(r"\begin{tabular*}{\textwidth}") == 2
    assert r"\caption{Robustness of the Main Contrasts, Pooled over the Two \mbox{Models}}" in tex
    assert r"\textit{Panel A. Inference and task sampling}" in tex
    assert r"\textit{Panel B. Scoring rule}" in tex
    assert "task half" not in tex and "example" not in tex.split("Notes:")[1]
    assert "few-shot" not in tex.lower()
    panels = tex.split(r"\begin{tabular*}")[1:]
    for index, panel in enumerate(panels):
        body = panel.split(r"\midrule", 1)[1].split(r"\end{tabular*}")[0]
        body, _, rate_block = body.partition("Success, all runs")
        rate_block = ("Success, all runs" + rate_block) if rate_block else ""
        body = body.replace(r"\midrule", "").replace(r"\bottomrule", "").strip().splitlines()
        # Each contrast prints two lines, and its label runs over both.
        labels = [(first.split(" & ")[0], second.split(" & ")[0]) for first, second in zip(body[::2], body[1::2])]
        assert labels == list(PRINTED_LABELS)
        # Panel A splits each interval into two cells aligned on the comma: 1 + 1 + 4 x 2 cells.
        # Panel B: label + five columns (main rule, two tolerances without and with the SE checks).
        cells = 10 if index == 0 else 6
        assert all(line.count(" & ") == cells - 1 for line in body[::2]), index
        if index == 1:
            # Task-success levels: all runs and the constrained agent, under every scoring rule.
            rate_rows = [line for line in rate_block.replace(r"\bottomrule", "").strip().splitlines() if line.strip()]
            assert [line.split(" & ")[0] for line in rate_rows] == ["Success, all runs", "Success, agent"]
            assert all(line.count(" & ") == 5 for line in rate_rows)
        else:
            assert not rate_block
    first_panel = panels[0].split(r"\midrule", 1)[1]
    assert "[$13.9$ & $21.7$]" in first_panel.replace(r"\phantom{)}", "")
    assert all(f"({i})" in tex for i in range(1, 11))
    assert "(11)" not in tex
    assert "SE checks" in panels[1].split(r"\midrule", 1)[0]
    assert "Per-task" not in tex and "tolerance of its task" not in tex
    assert r"\label{tab:robustness}" in tex and "|" not in tex
    assert info["dataset_clusters"] == 32
    assert (info["bootstrap_draws"], info["random_draws"], info["seed"]) == (4000, 5000, 20260930)


@requires_analysis
def test_deterministic_with_seed(result):
    first, second = result
    pd.testing.assert_frame_equal(first["table"], second["table"])
    assert pr.render_table(first["table"], first["info"]) == pr.render_table(second["table"], second["info"])


@requires_analysis
def test_prose_claims_hold(result):
    table, info = result[0]["table"], result[0]["info"]
    pr.verify_claims(table)
    pr.verify_scale(info)
    pr.verify_se_claims(info)
    macros = pr.macros(table, info)
    assert all(name.startswith("numRob") and name.isalpha() for name in macros)


@requires_analysis
@pytest.mark.skipif(not INSTALLED.exists(), reason="the manuscript table is not installed")
def test_installed_table_is_current(result):
    assert INSTALLED.read_text() == pr.render_table(result[0]["table"], result[0]["info"])


# ---------------------------------------------------------------- tight tolerance with SE checks
@requires_analysis
def test_se_kept_columns_add_the_se_check(result, tmp_path):
    d = pr.load(ANALYSIS)
    sch = pr.scheme_outcomes(ANALYSIS, d, tmp_path).set_index("position_id")
    se = pd.read_csv(ANALYSIS / "data" / "se_checks.csv", low_memory=False)
    se_ok = se.groupby("position_id").se_pass.apply(lambda s: s.astype(bool).all()).reindex(sch.index).fillna(True)
    for tag in ("tight_1e3", "tight_1e4"):
        expected = (sch[tag].astype(bool) & se_ok).astype(int)
        assert (sch[f"{tag}_se"] == expected).all(), tag
        assert (sch[f"{tag}_se"] <= sch[tag]).all()
    table = result[0]["table"].set_index("contrast")
    for tag in ("tight_1e3_se", "tight_1e4_se"):
        assert table[f"{tag}_est"].notna().all() and table[f"{tag}_se"].notna().all()


@requires_analysis
def test_loose_columns_never_fail_a_primary_success(tmp_path):
    d = pr.load(ANALYSIS)
    sch = pr.scheme_outcomes(ANALYSIS, d, tmp_path).set_index("position_id").reindex(d.position_id)
    y = d.y.to_numpy()
    assert set(pr.LOOSE_SCHEMES) <= set(sch.columns) and not set(pr.LOOSE_SCHEMES) & set(pr.SCHEMES)
    for tag in ("loose_5e2", "loose_1e1"):
        assert (sch[f"{tag}_se"] == (sch[tag].astype(bool) & sch.se_ok.astype(bool)).astype(int)).all(), tag
        assert (sch[f"{tag}_se"].to_numpy() >= y).all(), tag
        assert (sch[tag] >= sch.primary_est).all(), tag
    assert (sch.loose_1e1 >= sch.loose_5e2).all() and (sch.loose_1e1_se >= sch.loose_5e2_se).all()


@requires_analysis
def test_se_only_failures(result):
    info = result[0]["info"]
    se_only = info["se_only_failures"]
    assert set(se_only) == {"primary", "tight_1e3", "tight_1e4"}
    assert se_only["primary"]["by_task"].get("applied_shift_share", 0) > 0
    for variant, rec in se_only.items():
        assert rec["runs"] == sum(rec["by_task"].values()), variant
        assert set(rec["by_task"]) <= SE_TASKS, variant
        assert all(v > 0 for v in rec["by_task"].values())
    # A tighter estimate check can only remove runs that pass every estimate.
    assert se_only["primary"]["runs"] >= se_only["tight_1e3"]["runs"] >= 0
    assert se_only["tight_1e3"]["runs"] >= se_only["tight_1e4"]["runs"]
    # The primary SE-only failures are the runs with every estimate correct and y = 0.
    d = pd.read_csv(ANALYSIS / "data" / "runs.csv", usecols=["position_id", "y"])
    assert se_only["primary"]["runs"] == int((d.y == 0).sum()) - int(info["estimate_failures"]["primary"])
    rates = info["success_rates"]
    assert rates["primary"] == pytest.approx(100 * d.y.mean(), abs=1e-9)
    assert rates["tight_1e3_se"] <= rates["tight_1e3"] and rates["tight_1e4_se"] <= rates["tight_1e4"]


# ---------------------------------------------------------------- dataset-cluster bootstrap
def test_dataset_groups():
    catalog = __import__("json").loads((ANALYSIS / "data" / "catalog.json").read_text())
    groups = pr.dataset_groups(catalog)
    assert len(groups) == 40 and len(set(groups.values())) == 32
    sizes = pd.Series(groups).value_counts()
    assert int((sizes > 1).sum()) == 5 and int(sizes[sizes > 1].sum()) == 13
    assert groups["two_way_cluster"] == groups["twoway_fe"] == groups["staggered_did"]


requires_dataset_boot = pytest.mark.skipif(
    not all((DATASET_BOOT / m / "bootstrap_results.json").exists() for m in ("sonnet", "luna", "pooled")),
    reason="the dataset-cluster bootstrap has not run",
)


@requires_analysis
@requires_dataset_boot
def test_dataset_bootstrap_inputs_match_task_bootstrap():
    """The Stata input equals the task-cluster input with the dataset id in the cluster column.

    The task fixed effects come from armtask (model__task) in every column.
    """
    catalog = __import__("json").loads((ANALYSIS / "data" / "catalog.json").read_text())
    groups = pr.dataset_groups(catalog)
    for model in ("sonnet", "luna", "pooled"):
        task_input = pd.read_csv(ANALYSIS / "data" / "inference" / model / "input.csv")
        data_input = pd.read_csv(DATASET_BOOT / model / "input.csv")
        assert list(data_input.columns) == ["success", "task", "cell", "armtask"]
        pd.testing.assert_frame_equal(data_input[["success", "cell", "armtask"]], task_input[["success", "cell", "armtask"]])
        assert (data_input.task == task_input.task.map(groups)).all()
        assert data_input.task.nunique() == 32 and data_input.armtask.nunique() == task_input.armtask.nunique()
        do = (DATASET_BOOT / model / "inference.do").read_text()
        assert "reg success i.fen ib1.cell, cluster(taskn)" in do
        assert do.count("reps(9999) weighttype(rademacher) seed(20260701)") >= 9
        assert "adjusted for 32 clusters" in (DATASET_BOOT / model / "inference.log").read_text()


@requires_analysis
@requires_dataset_boot
def test_dataset_pvalues_and_holm():
    import paper_table1 as t1

    rows = pr.dataset_pvalues(ANALYSIS)
    assert set(rows) == set(t1.MODELS)
    task = t1.load_contrasts(ANALYSIS)
    for m in t1.MODELS:
        assert set(rows[m]) == set(t1.contrast_ids())
        for family in ("primary", "secondary"):
            ids = [cid for p in t1.PANELS if p["family"] == family for cid, _ in p["rows"]]
            ps = {cid: rows[m][cid]["p_dataset"] for cid in ids}
            for cid, adj in t1.holm(ps).items():
                assert rows[m][cid]["holm_dataset"] == pytest.approx(adj, abs=1e-15)
        for cid in t1.contrast_ids():
            r = rows[m][cid]
            assert r["bootstrap_draws"] == 9999
            assert 0 < r["p_dataset"] <= 1
            assert r["p_task"] == task[m][cid]["p_wcb"] and r["holm_task"] == task[m][cid]["p_holm"]
    changes = pr.holm_changes(rows)
    for c in changes:
        assert (rows[c["model"]][c["contrast"]]["holm_task"] < 0.05) != (rows[c["model"]][c["contrast"]]["holm_dataset"] < 0.05)
    pr.verify_dataset_claims(rows)


@requires_analysis
@requires_dataset_boot
def test_dataset_pvalue_table_layout():
    rows = pr.dataset_pvalues(ANALYSIS)
    tex = pr.render_dataset_pvalues(rows, ANALYSIS)
    assert r"\caption{Wild Cluster Bootstrap $p$-Values with Dataset Clusters}" in tex
    assert r"\label{tab:pvalues_dataset}" in tex
    assert tex.count(r"\begin{tabular*}{\textwidth}") == 1
    body = tex.split(r"\midrule", 1)[1].split(r"\bottomrule")[0]
    data = [line for line in body.splitlines() if line.startswith(r"\quad ")]
    assert len(data) == 9 and all(line.count(" & ") == 6 for line in data)
    assert "32 input datasets" in tex and "|" not in tex
    # The paper reports no p-value table; the manuscript holds no copy of it.
    assert not INSTALLED_DATASET.exists()
    appendix = (MANUSCRIPT / "appendix" / "D_robustness.tex").read_text()
    assert r"\input{tables/appendix_table1_pvalues_dataset}" not in appendix

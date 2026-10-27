"""The table generators reproduce the installed manuscript tables byte for byte.

Each test recomputes a table from data/analysis/main and compares it with the file in
paper/manuscript/tables/. Since the 2026-09-30 table-design audit, Table 1 prints the effect
over its standard error in three columns without scaling, and Tables B1 and B2 use grouped
rows with italic group headings, a note, and a title-case caption without a final period.
The model name carries no harness suffix (Raul Sosa, manuscript commit 3b3b7bd).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "benchmark"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "paper"))

import paper_table1  # noqa: E402
import production_paper_assets as assets  # noqa: E402
import production_paper_figures as figures  # noqa: E402
import manuscript  # noqa: E402  (code/paper/manuscript.py)

ANALYSIS = PROJECT_ROOT / "data" / "analysis" / "main"
TABLES = PROJECT_ROOT / "paper" / "manuscript" / "tables"

requires_manuscript = pytest.mark.skipif(
    not TABLES.is_dir(), reason="paper/manuscript is not present in this checkout"
)


@pytest.fixture(scope="module")
def runs() -> pd.DataFrame:
    data = pd.read_csv(ANALYSIS / "data" / "runs.csv")
    for column in ("primary", "execution_ok", "output_available"):
        assert data[column].isin([True, False]).all(), column
        data[column] = data[column].astype(bool)
    return data


@requires_manuscript
def test_table1_generator_reproduces_the_installed_table() -> None:
    installed = (TABLES / "main_results.tex").read_text()
    assert installed == paper_table1.render(ANALYSIS)
    assert r"\resizebox" not in installed
    assert "|" not in "\n".join(line for line in installed.splitlines() if not line.startswith("%"))
    assert r"\begin{tabular*}{\textwidth}" in installed


@requires_manuscript
def test_manuscript_holds_no_p_value_table() -> None:
    for name in ("appendix_table1_pvalues", "appendix_table1_pvalues_dataset"):
        assert not (TABLES / f"{name}.tex").exists(), name


@requires_manuscript
def test_table_b1_is_merged_into_the_software_table() -> None:
    # 2026-10-01 appendix reductions (proposal B5): the "All languages" rows replace Table B1.
    assert not (TABLES / "appendix_failure_gates.tex").exists()
    assert "appendix_failure_gates" not in manuscript.APPENDIX_TABLES


@requires_manuscript
def test_table_b2_generator_reproduces_the_installed_table(runs: pd.DataFrame) -> None:
    body, values = assets.software_table(runs)
    installed = (TABLES / "appendix_gate_by_software.tex").read_text()
    assert installed == manuscript.appendix_table("appendix_gate_by_software", body)
    assert "(Codex)" not in installed
    # The two models are column groups. The language is an italic group heading, and the last
    # group pools the three languages.
    assert r"\begin{tabular}{@{}l rrr rrr@{}}" in installed
    assert all(installed.count(rf"\textit{{{name}}}") == 1 for name in ("Python", "R", "Stata", "All languages"))
    assert r"\caption{Task Success and Failure Types by Model, Language, and Agency Condition}" in installed
    assert r"\label{tab:gate_by_software}\label{tab:app_failure_gates}" in installed
    assert len(values) == 24
    assert values[values.platform.ne("all")].n.eq(800).all() and values[values.platform.eq("all")].n.eq(2400).all()
    # The pooled rows equal the loud and silent rates of the former Table B1.
    rates = figures.failure_rates(runs).set_index(["model", "agency"])
    pooled = values[values.platform.eq("all")].set_index(["model", "agency"])
    for key, row in pooled.iterrows():
        assert row.loud_pct == pytest.approx(rates.loc[key, "loud_per100"], abs=1e-9)
        assert row.silent_pct == pytest.approx(rates.loc[key, "silent_per100"], abs=1e-9)


def test_table_labels_carry_no_harness_suffix() -> None:
    assert assets.TABLE_MODEL_LABEL == {"sonnet": "Claude Sonnet 4.6", "luna": "GPT-5.6 Luna"}

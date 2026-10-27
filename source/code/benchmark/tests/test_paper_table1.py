"""Tests for Table 1 and Table D2 (code/benchmark/paper_table1.py) and the installed manuscript tables.

Table 1 prints the effect over its standard error in three columns (Design A of the
2026-09-30 table-design audit). Table D2 in the Supplemental Appendix prints the
two-sided wild cluster bootstrap p-value and its Holm adjustment for every entry.
The installed files must be the generator output for data/analysis/main. Every printed
number must match regression_contrasts.csv, the three bootstrap_results.json files, and
runs.csv. The checks below parse the installed files and recompute the Holm adjustment
and the rates on their own.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "benchmark"))

import paper_table1  # noqa: E402
from paper_numbers import fixed  # noqa: E402

MANUSCRIPT = PROJECT_ROOT / "paper" / "manuscript"
INSTALLED = MANUSCRIPT / "tables" / "main_results.tex"
INSTALLED_PVALUES = MANUSCRIPT / "tables" / "appendix_table1_pvalues.tex"
ANALYSIS = PROJECT_ROOT / "data" / "analysis" / "main"
COLUMNS = ("sonnet", "luna", "pooled")

requires_manuscript = pytest.mark.skipif(
    not INSTALLED.exists(), reason="paper/manuscript is not present in this checkout"
)

#: Printed row label, contrast id, and Holm family, written out here on purpose
#: so that the test does not read the generator's own row list.
ROWS = (
    ("Self-repair (revises on error or missing value)", "H3_self_repair", "A"),
    ("Constrained agent (runs and revises its script)", "H3_tool_loop", "A"),
    ("One example in the prompt", "H2_fewshot", "BCD"),
    ("R, zero-shot chatbot", "H1_r", "BCD"),
    ("Stata, zero-shot chatbot", "H1_stata", "BCD"),
    ("Stata, average over prompts and agency", "GAP_stata_avg", "BCD"),
    ("Stata, zero-shot constrained agent", "GAP_stata_agent_zs", "BCD"),
    ("Under self-repair", "H4_self_repair", "BCD"),
    ("Under the constrained agent", "H4_tool_loop", "BCD"),
)

#: Panel headings of Table 1 and Table D2. The prompt condition with one example is "one-shot"
#: in every reader-facing text (Raul Sosa, manuscript commit fd96814).
PANEL_HEADINGS = (
    "Panel A. Agency, relative to the chatbot",
    "Panel B. Prompt, one-shot relative to zero-shot",
    "Panel C. Language, relative to Python",
    "Panel D. One-shot effect, change from the chatbot",
)


def plain(cell: str) -> str:
    """Remove layout commands so that only the printed characters remain."""
    cell = re.sub(r"\\phantom\{[^{}]*\}", "", cell)
    braced = r"(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*"
    cell = re.sub(r"\\multicolumn\{\d+\}\{" + braced + r"\}\{(" + braced + r")\}", r"\1", cell)
    cell = re.sub(r"\\textit\{(" + braced + r")\}", r"\1", cell)
    cell = cell.replace(r"\quad", "")
    return cell.replace("$<$", "<").replace("{-}", "-").replace("$", "").strip()


def cells(line: str) -> list[str]:
    line = line.strip()
    line = re.sub(r"\\\\(\[[\d.]+pt\])?$", "", line)
    return [plain(cell) for cell in line.split("&")]


def row_index(lines: list[str], label: str) -> int:
    """Index of the one body line whose first cell prints the label."""
    found = [i for i, line in enumerate(lines) if "&" in line and cells(line)[0] == label]
    assert len(found) == 1, label
    return found[0]


def printed_p(value: float) -> str:
    return "<0.001" if value < 0.001 else fixed(value, 3)


def step_down(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm adjustment: the k-th smallest of m p-values times (m - k + 1), made monotone, capped at 1."""
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    out: dict[str, float] = {}
    for k, key in enumerate(order):
        candidates = [min(1.0, (m - j) * pvalues[order[j]]) for j in range(k + 1)]
        out[key] = max(candidates)
    return out


@pytest.fixture(scope="module")
def contrasts() -> dict[tuple[str, str], dict[str, str]]:
    with (ANALYSIS / "tables" / "regression_contrasts.csv").open(newline="") as source:
        return {(row["model"], row["contrast"]): row for row in csv.DictReader(source)}


@pytest.fixture(scope="module")
def bootstrap() -> dict[str, dict]:
    return {
        model: json.loads((ANALYSIS / "data" / "inference" / model / "bootstrap_results.json").read_text())
        for model in COLUMNS
    }


@pytest.fixture(scope="module")
def holm(contrasts) -> dict[str, dict[str, float]]:
    """Holm within Panel A and within Panels B to D, separately in each column."""
    out: dict[str, dict[str, float]] = {}
    for model in COLUMNS:
        out[model] = {}
        for family in ("A", "BCD"):
            ps = {cid: float(contrasts[(model, cid)]["p_wcb"]) for _, cid, fam in ROWS if fam == family}
            out[model].update(step_down(ps))
    return out


def test_holm_matches_a_hand_computed_example() -> None:
    assert paper_table1.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx(
        {"a": 0.03, "b": 0.06, "c": 0.06}
    )
    # The adjustment is capped at one and never falls below an earlier value.
    assert paper_table1.holm({"a": 0.6, "b": 0.7}) == {"a": 1.0, "b": 1.0}


def test_p_values_and_effects_use_the_half_up_format() -> None:
    assert paper_table1.fmt_p(0.0006) == "$<$0.001"
    assert paper_table1.fmt_p(0.001) == "0.001"
    assert paper_table1.fmt_p(0.0125) == "0.013"
    assert paper_table1.fmt_effect(-10.75) == "${-}10.8$"
    assert paper_table1.fmt_effect(1.25) == "$1.3$"


def test_table1_prints_effect_over_standard_error_in_three_columns() -> None:
    tex = paper_table1.render(ANALYSIS)
    body = [line for line in tex.splitlines() if not line.startswith("%")]
    text = "\n".join(body)
    # No scaling, no vertical rules, no stars, and no p-values in the table body.
    assert r"\resizebox" not in text
    assert "|" not in text
    assert not re.search(r"\^\{?\*", text)
    start = next(i for i, line in enumerate(body) if line.startswith(r"\midrule"))
    stop = next(i for i, line in enumerate(body) if line.startswith(r"\bottomrule"))
    for line in body[start + 1 : stop]:
        if "&" in line:
            assert line.count("&") == 3, line
        assert "[" not in re.sub(r"\\\\\[[\d.]+pt\]$", "", line), line
        assert "<" not in line, line
    # Title-case caption without a final period (AER: Insights).
    assert r"\caption{Effects of Programming Language, Prompting, and Agency on Task Success}" in text
    assert r"\label{tab:estimates_three_models}" in text
    for heading in PANEL_HEADINGS:
        assert heading in text, heading
    assert "few-shot" not in text.lower()
    # The note keeps the Panel D wording and points to no p-value table.
    assert "In Panel D, a negative change means that the example adds less with agency." in text
    assert r"\Cref{tab:pvalues}" not in text
    assert "Holm" not in text


def test_table_d2_prints_p_and_holm_p_for_each_model() -> None:
    tex = paper_table1.render_pvalues(ANALYSIS)
    text = "\n".join(line for line in tex.splitlines() if not line.startswith("%"))
    assert r"\label{tab:pvalues}" in text
    assert r"\caption{Wild Cluster Bootstrap $p$-Values for the Contrasts of \Cref{tab:estimates_three_models}}" in text
    assert r"\resizebox" not in text and "|" not in text
    for heading in PANEL_HEADINGS:
        assert heading in text, heading
    assert "few-shot" not in text.lower()
    # The note states both Holm families and the bootstrap settings.
    assert "Rademacher weights, 9,999 replications, null imposed" in text
    assert "Panel A (the two agency contrasts) is one family" in text
    assert "Panels B to D (seven contrasts) are the other" in text


def test_command_line_writes_either_table(tmp_path: Path) -> None:
    out = tmp_path / "table.tex"
    paper_table1.main([str(ANALYSIS), "--out", str(out)])
    assert out.read_text() == paper_table1.render(ANALYSIS)
    paper_table1.main([str(ANALYSIS), "--pvalues", "--out", str(out)])
    assert out.read_text() == paper_table1.render_pvalues(ANALYSIS)


@requires_manuscript
def test_installed_tables_are_the_generator_output_for_the_main_analysis() -> None:
    hint = "rerun 'uv run --no-sync python code/paper/manuscript.py --analysis data/analysis/main'"
    assert INSTALLED.read_text() == paper_table1.render(ANALYSIS), f"main_results.tex is stale; {hint}"
    # The paper reports no p-value table; the manuscript holds no copy of it.
    assert not INSTALLED_PVALUES.exists()
    appendix = (MANUSCRIPT / "appendix" / "D_robustness.tex").read_text()
    assert r"\input{tables/appendix_table1_pvalues}" not in appendix


@requires_manuscript
def test_every_printed_number_matches_the_analysis_files(contrasts, bootstrap) -> None:
    lines = INSTALLED.read_text().splitlines()
    checked: set[int] = set()

    for label, cid, _ in ROWS:
        index = row_index(lines, label)
        first, second = cells(lines[index]), cells(lines[index + 1])
        assert first[0] == label and second[0] == ""
        for column, model in enumerate(COLUMNS):
            row = contrasts[(model, cid)]
            assert bootstrap[model][cid]["p_wcb"] == pytest.approx(float(row["p_wcb"]), abs=1e-12), (model, cid)
            assert bootstrap[model][cid]["bootstrap_draws"] == 9999, (model, cid)
            assert first[1 + column] == fixed(float(row["estimate_pp"]), 1), (model, cid)
            assert second[1 + column] == "(" + fixed(float(row["se_cr1_pp"]), 1) + ")", (model, cid)
        checked |= {index, index + 1}

    # Reference rates: chatbot mean of the 18 cell rates, and the zero-shot chatbot Python cell.
    rates: dict[tuple, list[float]] = defaultdict(list)
    with (ANALYSIS / "data" / "runs.csv").open(newline="") as source:
        for run in csv.DictReader(source):
            key = (run["model"], run["platform"], run["prompt"], run["agency"])
            rates[key].append(float(run["y"]))
    cell = {key: 100 * sum(values) / len(values) for key, values in rates.items()}
    levels = {}
    for model in ("sonnet", "luna"):
        chatbot = [value for key, value in cell.items() if key[0] == model and key[3] == "chatbot"]
        levels[model] = (sum(chatbot) / len(chatbot), cell[(model, "python", "zero_shot", "chatbot")])
    levels["pooled"] = tuple((levels["sonnet"][i] + levels["luna"][i]) / 2 for i in range(2))
    for position, label in enumerate(("Chatbot, all languages and prompts", "Zero-shot chatbot writing Python")):
        index = row_index(lines, label)
        assert cells(lines[index])[1:] == [fixed(levels[m][position], 1) for m in COLUMNS], label
        checked.add(index)

    runs_line = row_index(lines, "Runs")
    assert cells(lines[runs_line])[1:] == [f"{int(contrasts[(m, 'H3_tool_loop')]['n_runs']):,}" for m in COLUMNS]
    assert sum(len(values) for values in rates.values()) == int(contrasts[("pooled", "H3_tool_loop")]["n_runs"])
    tasks_line = row_index(lines, "Tasks (clusters)")
    assert cells(lines[tasks_line])[1:] == [contrasts[(m, "H3_tool_loop")]["effective_tasks"] for m in COLUMNS]
    checked |= {runs_line, tasks_line}

    # No other line of the table body prints a digit, apart from the column numbers in the head.
    start = next(i for i, line in enumerate(lines) if line.startswith(r"\midrule"))
    stop = next(i for i, line in enumerate(lines) if line.startswith(r"\bottomrule"))
    for i in range(start + 1, stop):
        if i not in checked:
            assert not re.search(r"\d", plain(re.sub(r"\\\\(\[[\d.]+pt\])?$", "", lines[i]))), lines[i]
    assert len(checked) == 2 * len(ROWS) + 4


def test_every_p_value_in_table_d2_matches_the_bootstrap_files(bootstrap, holm) -> None:
    lines = paper_table1.render_pvalues(ANALYSIS).splitlines()
    for label, cid, _ in ROWS:
        printed = cells(lines[row_index(lines, label)])
        expected = []
        for model in COLUMNS:
            expected += [printed_p(bootstrap[model][cid]["p_wcb"]), printed_p(holm[model][cid])]
        assert printed[1:] == expected, (label, printed)
    # Every row of Table 1 appears in Table D2, in the same order.
    order = [row_index(lines, label) for label, _, _ in ROWS]
    assert order == sorted(order)

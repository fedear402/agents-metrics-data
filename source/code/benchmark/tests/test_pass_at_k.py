"""Tests for the paper pass@k module and the appendix figure it feeds.

Three groups:

1. The estimator itself, against closed forms and against a direct
   combinatorial count on small cells.
2. The payload shape on synthetic rows: cell counts, curve length, and the
   guards that reject an unbalanced sample.
3. The installed chain: the block inside the historical
   ``manuscript_numbers.json``, the ``numbers.tex`` macros against the
   current ``data/analysis/main`` curve, and the installed figure.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path

import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

import pass_at_k as pak  # noqa: E402

PROJECT_ROOT = BENCHMARK_ROOT.parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "analysis"
CURRENT_DATA = DATA_DIR / "main" / "data"
MANUSCRIPT = PROJECT_ROOT / "paper" / "manuscript"

requires_manuscript = pytest.mark.skipif(
    not MANUSCRIPT.exists(),
    reason="paper/manuscript is not present in this checkout",
)

TASKS = ("t1", "t2")
PLATFORMS = ("python", "r", "stata")
PROMPTS = ("zero_shot", "few_shot")


# ---------------------------------------------------------------------------
# 1. The estimator
# ---------------------------------------------------------------------------
def test_pass_at_one_is_the_success_share() -> None:
    for n, c in ((12, 0), (12, 5), (12, 12), (5, 2)):
        assert pak.unbiased_pass_at_k(n, c, 1) == pytest.approx(c / n)


def test_extremes() -> None:
    for k in range(1, 13):
        assert pak.unbiased_pass_at_k(12, 0, k) == 0.0
        assert pak.unbiased_pass_at_k(12, 12, k) == 1.0


def test_k_equals_n_needs_only_one_success() -> None:
    assert pak.unbiased_pass_at_k(12, 1, 12) == 1.0


def test_matches_a_direct_enumeration() -> None:
    """Enumerate every draw of k from n and count the ones with a success."""
    for n in (5, 6, 7):
        for c in range(n + 1):
            runs = [1] * c + [0] * (n - c)
            for k in range(1, n + 1):
                draws = list(itertools.combinations(runs, k))
                direct = sum(1 for d in draws if any(d)) / len(draws)
                assert pak.unbiased_pass_at_k(n, c, k) == pytest.approx(direct)


def test_the_estimator_is_unbiased_for_a_known_binomial() -> None:
    """The mean over samples equals the population pass@k.

    With independent runs that succeed with probability p, the population
    pass@k is 1 - (1-p)^k. Averaging the cell estimator over the binomial
    distribution of c must reproduce it exactly.
    """
    n, k, p = 12, 4, 0.3
    expected = 1.0 - (1.0 - p) ** k
    got = sum(
        math.comb(n, c) * p**c * (1 - p) ** (n - c) * pak.unbiased_pass_at_k(n, c, k)
        for c in range(n + 1)
    )
    assert got == pytest.approx(expected, abs=1e-12)


def test_the_estimator_rejects_impossible_arguments() -> None:
    with pytest.raises(ValueError):
        pak.unbiased_pass_at_k(12, 3, 0)
    with pytest.raises(ValueError):
        pak.unbiased_pass_at_k(3, 1, 4)
    with pytest.raises(ValueError):
        pak.unbiased_pass_at_k(12, 13, 2)


# ---------------------------------------------------------------------------
# 2. The payload on synthetic rows
# ---------------------------------------------------------------------------
def make_rows(successes_by_model: dict[str, int]) -> list[dict]:
    """A balanced synthetic sample: twelve runs in every design cell."""
    rows = []
    for model, successes in successes_by_model.items():
        for agency in pak.AGENCY_LABELS:
            for prompt in PROMPTS:
                for platform in PLATFORMS:
                    for task in TASKS:
                        for rep in range(12):
                            rows.append(
                                {
                                    "model": model,
                                    "interface_label": agency,
                                    "prompt": prompt,
                                    "platform": platform,
                                    "task": task,
                                    "rep": rep,
                                    "task_success": rep < successes,
                                }
                            )
    return rows


def patch_cell_counts(monkeypatch) -> None:
    """Shrink the expected cell counts to the two tasks used here."""
    monkeypatch.setattr(pak, "CELLS_PER_AGENCY", len(TASKS) * len(PLATFORMS) * 2)
    monkeypatch.setattr(pak, "CELLS_PER_MODEL", pak.CELLS_PER_AGENCY * 3)


def test_cell_counts_pool_the_prompt_conditions(monkeypatch) -> None:
    patch_cell_counts(monkeypatch)
    rows = make_rows({model: 6 for model in pak.MODELS})
    counts = pak.cell_counts(rows)
    key = (pak.MODELS[0], pak.AGENCY_LABELS[0])
    assert len(counts[key]) == len(TASKS) * len(PLATFORMS) * len(PROMPTS)
    assert set(counts[key]) == {(12, 6)}


def test_payload_shape_and_values(monkeypatch) -> None:
    patch_cell_counts(monkeypatch)
    rows = make_rows({model: 6 for model in pak.MODELS})
    payload = pak.paper_pass_at_k(rows)
    assert payload["k_values"] == list(range(1, 13))
    for model in pak.MODELS:
        block = payload["by_model"][model]
        assert block["overall"]["n_runs_per_cell"] == 12
        curve = block["overall"]["pass_at_k_pct"]
        assert curve["1"] == pytest.approx(50.0)
        assert curve["12"] == pytest.approx(100.0)
        # Every cell is identical here, so the mean equals the cell value.
        for k in range(1, 13):
            assert curve[str(k)] == pytest.approx(
                round(100.0 * pak.unbiased_pass_at_k(12, 6, k), 4)
            )
        for agency in pak.AGENCY_LABELS:
            assert block["by_agency"][agency]["pass_at_k_pct"] == curve


def test_the_curve_never_falls(monkeypatch) -> None:
    patch_cell_counts(monkeypatch)
    rows = make_rows({model: 4 for model in pak.MODELS})
    payload = pak.paper_pass_at_k(rows)
    curve = payload["by_model"][pak.MODELS[0]]["overall"]["pass_at_k_pct"]
    values = [curve[str(k)] for k in range(1, 13)]
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_a_short_cell_is_rejected(monkeypatch) -> None:
    patch_cell_counts(monkeypatch)
    rows = make_rows({model: 6 for model in pak.MODELS})
    rows.pop()
    with pytest.raises(ValueError, match="twelve-repetition"):
        pak.paper_pass_at_k(rows)


def test_a_missing_agency_condition_is_rejected(monkeypatch) -> None:
    patch_cell_counts(monkeypatch)
    dropped = pak.AGENCY_LABELS[-1]
    rows = [
        row
        for row in make_rows({model: 6 for model in pak.MODELS})
        if row["interface_label"] != dropped
    ]
    with pytest.raises(ValueError, match="no cells"):
        pak.paper_pass_at_k(rows)


# ---------------------------------------------------------------------------
# 3. The installed chain
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def installed() -> dict:
    numbers = json.loads((DATA_DIR / "manuscript_numbers.json").read_text())
    return {
        "passk": numbers["pass_at_k_paper"],
        "full": json.loads((DATA_DIR / "full_analysis.json").read_text()),
    }


def test_installed_block_covers_the_paper_sample(installed: dict) -> None:
    block = installed["passk"]
    assert block["models"] == list(pak.MODELS)
    assert block["agency_conditions"] == list(pak.AGENCY_LABELS)
    assert block["k_values"] == list(range(1, 13))
    for model in pak.MODELS:
        assert block["by_model"][model]["overall"]["n_cells"] == 21 * 3 * 2 * 3
        for agency in pak.AGENCY_LABELS:
            cell = block["by_model"][model]["by_agency"][agency]
            assert cell["n_cells"] == 21 * 3 * 2
            assert cell["n_runs_per_cell"] == 12


def test_installed_pass_at_one_equals_the_observed_rate(installed: dict) -> None:
    """pass@1 collapses to the observed task-success rate of the same runs."""
    headline = installed["full"]["headline"]
    for model in pak.MODELS:
        block = installed["passk"]["by_model"][model]
        overall = 100.0 * sum(
            headline[model][p]["overall"]["successes"] for p in PROMPTS
        ) / sum(headline[model][p]["overall"]["n"] for p in PROMPTS)
        assert block["overall"]["pass_at_k_pct"]["1"] == pytest.approx(
            overall, abs=5e-4
        )
        for agency in pak.AGENCY_LABELS:
            rate = 100.0 * sum(
                headline[model][p]["by_interface"][agency]["successes"]
                for p in PROMPTS
            ) / sum(
                headline[model][p]["by_interface"][agency]["n"] for p in PROMPTS
            )
            assert block["by_agency"][agency]["pass_at_k_pct"]["1"] == pytest.approx(
                rate, abs=5e-4
            )


def test_the_standalone_file_matches_the_manuscript_block(installed: dict) -> None:
    standalone = json.loads((DATA_DIR / "pass_at_k.json").read_text())
    assert standalone["by_model"] == installed["passk"]["by_model"]


@requires_manuscript
def test_macros_carry_both_ends_of_every_curve() -> None:
    """numbers.tex carries both ends of every pass@k curve of the current analysis.

    The canonical analysis ``data/analysis/main`` has ten repetitions per cell,
    so each curve runs from k = 1 to k = 10. The macro names keep the word
    ``Twelve`` from the earlier twelve-repetition sample. They hold pass@10.
    """
    numbers = (MANUSCRIPT / "numbers.tex").read_text()
    with (CURRENT_DATA / "appendix_pass_at_k.csv").open(newline="") as source:
        curve = list(csv.DictReader(source))
    with (CURRENT_DATA / "runs.csv").open(newline="") as source:
        runs = list(csv.DictReader(source))
    model_names = {"sonnet": "Sonnet", "luna": "Codex"}
    agency_names = {"chatbot": "Chatbot", "self_repair": "OneRepair", "agent": "Agent"}
    ends = {1: "One", 10: "Twelve"}

    assert {row["model"] for row in runs} == set(model_names)
    assert {row["agency"] for row in runs} == set(agency_names)
    assert {(row["model"], row["agency"]) for row in curve} == {
        (model, agency) for model in model_names for agency in agency_names
    }
    # One curve point per k; each point pools 40 tasks x 3 languages x 2 prompts.
    assert {int(row["repetitions"]) for row in curve} == {10}
    assert {int(row["tasks"]) for row in curve} == {40}
    assert {int(row["cells"]) for row in curve} == {40 * 3 * 2}

    def macro(name: str, value: float) -> str:
        return rf"\newcommand{{\{name}}}{{{value:.1f}}}"

    def observed_rate(**criteria: str) -> float:
        selected = [
            row for row in runs if all(row[key] == value for key, value in criteria.items())
        ]
        return 100.0 * sum(row["primary"].lower() == "true" for row in selected) / len(selected)

    for model, model_name in model_names.items():
        last_points = []
        for agency, agency_name in agency_names.items():
            points = {
                int(row["k"]): float(row["pass_at_k_pct"])
                for row in curve
                if row["model"] == model and row["agency"] == agency
            }
            assert sorted(points) == list(range(1, 11)), (model, agency)
            # pass@1 collapses to the observed task-success rate of the same runs.
            assert points[1] == pytest.approx(
                observed_rate(model=model, agency=agency), abs=1e-9
            )
            for k, tex in ends.items():
                assert macro(f"numPass{tex}{agency_name}{model_name}", points[k]) in numbers
            last_points.append(points[10])
        # The balanced design gives each agency condition the same weight.
        assert macro(f"numPassOne{model_name}", observed_rate(model=model)) in numbers
        assert macro(f"numPassTwelve{model_name}", sum(last_points) / len(last_points)) in numbers


@requires_manuscript
def test_the_appendix_figure_is_installed() -> None:
    figure = MANUSCRIPT / "figures" / "appendix_pass_at_k.pdf"
    assert figure.exists()
    assert figure.stat().st_size > 0

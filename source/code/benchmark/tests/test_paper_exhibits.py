"""Focused tests for the manuscript table and figure builders."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

EXHIBIT_ROOT = Path(__file__).resolve().parents[2] / "paper_exhibits"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXHIBIT_ROOT))

import build_exhibits_full as exhibits  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "code" / "benchmark"))
from paper_numbers import fixed  # noqa: E402  (half-up rounding shared by the macros and the tables)

# The Overleaf-synced manuscript is a nested repo excluded from this one by
# design; a fresh clone lacks it. Manuscript-coupled tests skip cleanly there.
requires_manuscript = pytest.mark.skipif(
    not (PROJECT_ROOT / "paper" / "manuscript").exists(),
    reason="paper/manuscript is not present in this checkout",
)

#: Arms the paper reports, and the three columns of the main-results table
#: (decision record 2026-08-21_two-model-main-specification).
PAPER_ARMS = ("sonnet", "codex")
TABLE_COLUMNS = (*PAPER_ARMS, "pooled")

#: The prompt-by-agency interactions. Since decision record
#: 2026-09-25_table1-p-values-and-holm, Table 1 prints both in Panel D.
#: The prose discusses the constrained-agent interaction.
REPORTED_INTERACTION = "H4_tool_loop"
DIAGNOSTIC_INTERACTION = "H4_self_repair"

#: The canonical forty-task analysis that the current manuscript renders from.
MANUSCRIPT = PROJECT_ROOT / "paper" / "manuscript"
CURRENT_ANALYSIS = PROJECT_ROOT / "data" / "analysis" / "main"
CURRENT_MODELS = {"sonnet": "Claude Sonnet 4.6", "luna": "GPT-5.6 Luna"}
CURRENT_COLUMNS = (*CURRENT_MODELS, "pooled")
CURRENT_PLATFORMS = {"python": "Python", "r": "R", "stata": "Stata"}
CURRENT_AGENCIES = {
    "chatbot": "Chatbot",
    "self_repair": "Self-repair",
    "agent": "Constrained agent",
}
CURRENT_TASKS = 40
CURRENT_REPS = 10
CURRENT_RUNS = 14_400
#: Table 1 row labels and the regression contrast behind each row.
#: test_paper_table1.py checks every printed number of these rows.
CURRENT_TABLE_ROWS = {
    "Self-repair (revises on error or missing value)": "H3_self_repair",
    "Constrained agent (runs and revises its script)": "H3_tool_loop",
    "One example in the prompt": "H2_fewshot",
    "R, zero-shot chatbot": "H1_r",
    "Stata, zero-shot chatbot": "H1_stata",
    "Stata, average over prompts and agency": "GAP_stata_avg",
    "Stata, zero-shot constrained agent": "GAP_stata_agent_zs",
    "Under self-repair": DIAGNOSTIC_INTERACTION,
    "Under the constrained agent": REPORTED_INTERACTION,
}


def current_runs() -> list[dict[str, str]]:
    with (CURRENT_ANALYSIS / "data" / "runs.csv").open(newline="") as source:
        return list(csv.DictReader(source))


def is_true(value: str) -> bool:
    return value.lower() in {"true", "1", "1.0"}


def one_decimal_percent(count: int, n: int) -> str:
    # Same arithmetic as the table builder: 100 times the share.
    return fixed(100 * (count / n))


def strip_layout(tex: str) -> str:
    """Drop phantom padding, the braced minus sign, and spaces from a table cell."""
    tex = re.sub(r"\\phantom\{(?:[^{}]|\{[^{}]*\})*\}", "", tex)
    return tex.replace("{-}", "-").replace(" ", "")


def test_significance_stars_use_wild_bootstrap_thresholds() -> None:
    assert exhibits.significance_stars(0.100) == ""
    assert exhibits.significance_stars(0.099) == "*"
    assert exhibits.significance_stars(0.049) == "**"
    assert exhibits.significance_stars(0.009) == "***"
    assert exhibits.fmt_estimate_cell(0.059, 0.022, 0.012) == (
        r"\shortstack[c]{$5.9^{**}$\\{\scriptsize (2.2)}}"
    )


def model_cells(zero_successes: int, few_successes: int) -> dict:
    result = {}
    for prompt, successes in (
        ("zero_shot", zero_successes),
        ("few_shot", few_successes),
    ):
        result[prompt] = [
            {
                "interface": interface,
                "platform": platform,
                "n": 10,
                "successes": successes + interface_index,
            }
            for interface_index, interface in enumerate(exhibits.MODE_ORDER)
            for platform in ("python", "r", "stata")
        ]
    return result


def test_prompt_agency_figure_draws_claude_and_lower_opacity_codex(
    monkeypatch,
) -> None:
    captured = {}

    def capture(fig, stem):
        captured["fig"] = fig
        captured["stem"] = stem

    monkeypatch.setattr(exhibits, "save_figure", capture)
    snapshot = {
        "nine_cell": {
            "sonnet": model_cells(5, 7),
            "codex": model_cells(4, 6),
        }
    }
    exhibits.build_exhibit3(snapshot)
    ax = captured["fig"].axes[0]
    alphas = [line.get_alpha() for line in ax.lines]
    assert captured["stem"] == "exhibit3_prompt_agency"
    assert 0.25 in alphas
    assert 1.0 in alphas
    plt.close(captured["fig"])


def test_task_figure_draws_claude_and_transparent_codex_for_all_tasks(
    monkeypatch,
) -> None:
    captured = {}

    def capture(fig, stem):
        captured["fig"] = fig
        captured["stem"] = stem

    task_data = {
        prompt: {
            task: {
                platform: {
                    condition: {"n": 5, "successes": 2 + condition_index}
                    for condition_index, condition in enumerate(
                        ("chatbot", "one-repair", "constrained agent")
                    )
                }
                for platform in ("python", "r", "stata")
            }
            for task in exhibits.TASK_DISPLAY_ORDER
        }
        for prompt in ("zero_shot", "few_shot")
    }
    numbers = {
        "per_task_model_platform": {
            "sonnet": task_data,
            "codex": task_data,
        }
    }
    monkeypatch.setattr(exhibits, "save_figure", capture)

    exhibits.build_task_heterogeneity(numbers)

    ax = captured["fig"].axes[0]
    assert captured["stem"] == "exhibit4_task_heterogeneity"
    assert len(ax.get_yticklabels()) == 21
    assert len(ax.patches) == 0
    alphas = [line.get_alpha() for line in ax.lines]
    assert 1.0 in alphas
    assert 0.18 in alphas
    horizontal_paths = [line for line in ax.lines if len(line.get_ydata()) == 3]
    assert any(set(line.get_ydata()) == {20.0} for line in horizontal_paths)
    assert any(set(line.get_ydata()) == {19.89} for line in horizontal_paths)
    assert "all software and prompt conditions" in ax.get_xlabel()
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Chatbot",
        "One-repair",
        "Constrained agent",
    ]
    plt.close(captured["fig"])


@requires_manuscript
def test_results_section_keeps_prompt_and_task_figures() -> None:
    results_section = (
        PROJECT_ROOT
        / "paper"
        / "manuscript"
        / "sections"
        / "04_preliminary_results.tex"
    ).read_text()

    assert r"figures/prompt_agency.pdf" in results_section
    assert r"figures/task_heterogeneity.pdf" in results_section


@requires_manuscript
def test_coding_stages_stay_in_conceptual_prose_and_out_of_figure_one() -> None:
    # The current design figure renders from code/exhibits/final_fig_design.tex
    # through code/benchmark/production_paper_figures.py.
    figure_sources = [
        (PROJECT_ROOT / "code" / "exhibits" / "final_fig_design.tex").read_text(),
        (PROJECT_ROOT / "code" / "benchmark" / "production_paper_figures.py").read_text(),
    ]
    introduction = (MANUSCRIPT / "sections" / "01_introduction.tex").read_text()

    for figure_source in figure_sources:
        assert "Coding cycle" not in figure_source
        assert "translate request" not in figure_source
    assert "coding as involving five stages" in introduction
    # The introduction defines agency as the part of the last three stages the
    # model can do, and the three conditions give it progressively more of them.
    assert "The last three stages require running the code and reacting to what it returns" in introduction
    assert "which give the model progressively more of the last three stages" in introduction


def test_cost_figure_requests_only_sonnet(monkeypatch) -> None:
    calls = []
    captured = {}
    points = {
        (mode, platform): (0.1 + 0.02 * idx, 70.0 + 5.0 * idx)
        for idx, mode in enumerate(exhibits.MODE_ORDER)
        for platform in ("python", "r", "stata")
    }

    def fake_points(snapshot, model, cost_field):
        calls.append((model, cost_field))
        return points

    def capture(fig, stem):
        captured["fig"] = fig
        captured["stem"] = stem

    monkeypatch.setattr(exhibits, "pooled_cell_points", fake_points)
    monkeypatch.setattr(exhibits, "save_figure", capture)
    exhibits.build_exhibit4({})
    assert calls == [("sonnet", "mean_cost_usd")]
    assert captured["stem"] == "exhibit5_cost_of_correctness"
    assert captured["fig"].axes[0].get_ylim() == (50.0, 101.0)
    assert [
        text.get_text() for text in captured["fig"].axes[0].get_legend().get_texts()
    ] == ["Chatbot", "One-repair", "Constrained agent"]
    plt.close(captured["fig"])


def test_pooled_cost_uses_only_runs_with_recorded_cost() -> None:
    cells = []
    for prompt, cost, n_priced in (
        ("zero_shot", 0.10, 10),
        ("few_shot", 0.30, 5),
    ):
        for mode in exhibits.MODE_ORDER:
            for platform in ("python", "r", "stata"):
                cells.append(
                    {
                        "interface": mode,
                        "platform": platform,
                        "n": 10,
                        "n_priced": n_priced,
                        "successes": 8,
                        "mean_cost_usd": cost,
                    }
                )
    snapshot = {
        "nine_cell": {
            "sonnet": {
                prompt: [
                    cell
                    for index, cell in enumerate(cells)
                    if (index < 9) == (prompt == "zero_shot")
                ]
                for prompt in ("zero_shot", "few_shot")
            }
        }
    }

    points = exhibits.pooled_cell_points(snapshot, "sonnet", "mean_cost_usd")

    assert points[("no_feedback", "python")][0] == pytest.approx(1.0 / 6.0)


def test_legacy_builder_does_not_install_manuscript_assets_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installed = []
    monkeypatch.setattr(exhibits, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(exhibits, "apply_style", lambda: None)
    monkeypatch.setattr(exhibits, "load_json", lambda _path: {})
    monkeypatch.setattr(exhibits, "check_guards", lambda *_args: None)
    monkeypatch.setattr(exhibits.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(exhibits, "build_exhibit2", lambda *_args: None)
    monkeypatch.setattr(exhibits, "build_exhibit3", lambda *_args: None)
    monkeypatch.setattr(exhibits, "build_task_heterogeneity", lambda *_args: None)
    monkeypatch.setattr(exhibits, "build_exhibit4", lambda *_args: None)
    monkeypatch.setattr(exhibits, "verify_outputs", lambda: None)
    monkeypatch.setattr(
        exhibits, "copy_manuscript_figures", lambda: installed.append("legacy")
    )

    assert exhibits.main() == 0
    assert installed == []


def test_final_exhibit_pipeline_uses_current_analysis_tree() -> None:
    final_root = PROJECT_ROOT / "code" / "exhibits"
    files = [
        final_root / "README.md",
        final_root / "extract_estimation_cost.py",
        final_root / "verify_estimation_cost.py",
        final_root / "verify_final_exhibits.py",
    ]
    for path in files:
        text = path.read_text()
        assert "data/generated/2026-07-30_full_analysis" not in text, path
        assert "data/analysis" in text, path


@requires_manuscript
def test_manuscript_tables_match_generated_estimates_and_standard_errors() -> None:
    with (CURRENT_ANALYSIS / "tables" / "regression_contrasts.csv").open(newline="") as source:
        contrasts = {(row["model"], row["contrast"]): row for row in csv.DictReader(source)}
    results_section = (MANUSCRIPT / "sections" / "04_preliminary_results.tex").read_text()
    table1 = (MANUSCRIPT / "tables" / "main_results.tex").read_text()
    numbers = (MANUSCRIPT / "numbers.tex").read_text()
    runs = current_runs()

    # The paper reports two models; the pooled column fits both together.
    assert {model for model, _ in contrasts} == set(CURRENT_COLUMNS)
    assert Counter((row["model"], row["prompt"]) for row in runs) == {
        (model, prompt): CURRENT_RUNS // 4
        for model in CURRENT_MODELS
        for prompt in ("zero_shot", "few_shot")
    }
    for model in CURRENT_COLUMNS:
        for key in (*CURRENT_TABLE_ROWS.values(), "AUX_fewshot_chatbot"):
            row = contrasts[(model, key)]
            expected_n = CURRENT_RUNS if model == "pooled" else CURRENT_RUNS // 2
            assert int(row["n_runs"]) == expected_n, (model, key)
            assert int(row["effective_tasks"]) == CURRENT_TASKS, (model, key)
            assert float(row["bootstrap_draws"]) == 9999, (model, key)

    assert r"\input{tables/main_results}" in results_section

    # Each row label appears once. test_paper_table1.py checks every printed
    # number of Table 1 against regression_contrasts.csv, the bootstrap files,
    # and runs.csv, and checks that the installed file is the generator output.
    lines = table1.splitlines()
    for label, key in CURRENT_TABLE_ROWS.items():
        matches = [i for i, line in enumerate(lines) if line.removeprefix(r"\quad ").startswith(label + " & ")]
        assert len(matches) == 1, label
        # The pooled fit is saturated within each model on balanced cells, so
        # the pooled estimate equals the unweighted mean of the two models.
        model_mean = sum(
            float(contrasts[(model, key)]["estimate_pp"]) for model in CURRENT_MODELS
        ) / len(CURRENT_MODELS)
        assert float(contrasts[("pooled", key)]["estimate_pp"]) == pytest.approx(
            model_mean, abs=1e-9
        ), key

    # The table omits the chatbot prompt effect; the results prose carries it
    # through named numbers.tex macros.
    for model, macro_name in (
        ("sonnet", "numPromptGainChatbotSonnet"),
        ("luna", "numPromptGainChatbotCodex"),
        ("pooled", "numPooledPromptGainChatbot"),
    ):
        bare = fixed(float(contrasts[(model, 'AUX_fewshot_chatbot')]['estimate_pp']))
        assert rf"\newcommand{{\{macro_name}}}{{{bare}}}" in numbers, (model, bare)
        assert rf"\{macro_name}{{}}" in results_section, macro_name

    # Table 1 prints the effect over its standard error, with no significance stars and no
    # p-value table (2026-10-02: the paper reports standard errors only).
    assert not re.search(r"\^\{?\*", table1)
    assert r"\Cref{tab:pvalues}" not in table1
    assert r"\input{tables/appendix_table1_pvalues}" not in (MANUSCRIPT / "appendix" / "D_robustness.tex").read_text()
    assert "Holm" not in table1
    # The note defines a negative Panel D change as substitution, with the wording of manuscript commit 20ffea4.
    assert "a negative change means that the example adds less with agency." in table1
    assert "percentage-point scale" not in table1
    # The note names the reference level of each task-success rate row.
    assert "is the reference level of Panel A" in table1
    assert "zero-shot chatbot rate in Python is the reference level of the first two rows of Panel C" in table1
    assert f"Standard errors clustered by task ({CURRENT_TASKS} clusters)" in table1
    assert "H4:" not in table1
    assert "Joint $p$-value" not in table1

    def row_cells(prefix: str) -> list[str]:
        line = next(ln for ln in lines if ln.strip().startswith(prefix))
        return [strip_layout(re.sub(r"\\(?:multicolumn\{\d\}\{r\}|llap)\{(.*)\}", r"\1", cell.strip()))
                for cell in line.strip()[:-2].split("&")]

    assert row_cells("Runs &")[1:] == [
        f"{int(contrasts[(column, 'H3_tool_loop')]['n_runs']):,}" for column in CURRENT_COLUMNS
    ]
    assert sum(int(contrasts[(model, "H3_tool_loop")]["n_runs"]) for model in CURRENT_MODELS) == len(runs)
    assert row_cells("Task fixed effects")[1:] == ["Yes", "Yes", "Modelbytask"]
    # Column 3 pools the two models. Each column head runs over two lines.
    # The banned display word and the dropped model appear nowhere.
    heads = [[strip_layout(re.sub(r"\\multicolumn\{1\}\{r\}\{(.*)\}", r"\1", cell.strip()))
              for cell in line.strip()[:-2].split("&")] for line in lines if line.startswith(" & ")][:2]
    assert heads == [["", "Claude", "GPT-5.6", "Total"], ["", "Sonnet4.6", "Luna", "(pooled)"]]
    assert not re.search(r"\barms?\b", table1, flags=re.IGNORECASE)
    assert "deepseek" not in table1.lower()


BODY_SECTIONS = (
    "sections/01_introduction.tex",
    "sections/02_background.tex",
    "sections/03_experimental_design.tex",
    "sections/04_preliminary_results.tex",
    "sections/05_cost_efficiency.tex",
    "sections/06_conclusion.tex",
)


@requires_manuscript
def test_submission_counter_covers_every_body_section_and_the_main_table() -> None:
    manuscript = PROJECT_ROOT / "paper" / "manuscript"
    latexmkrc = (manuscript / "latexmkrc").read_text()

    # The exhibit count scans the body sections plus the table file that the
    # results section pulls in, so every one of them has to be listed.
    for section in BODY_SECTIONS:
        assert f"'{section}'" in latexmkrc, section
        assert (manuscript / section).exists(), section
    assert "'tables/main_results.tex'" in latexmkrc

    # The merged table retired the separate other-models section; a stale entry
    # here would make latexmk die before it could count anything.
    assert "06_other_models" not in latexmkrc

    wordcount = (manuscript / "submission_wordcount.tex").read_text()
    for section in BODY_SECTIONS:
        assert f"\\input{{{section[: -len('.tex')]}}}" in wordcount, section
    assert "06_other_models" not in wordcount


@requires_manuscript
def test_active_manuscript_avoids_rejected_reader_facing_term() -> None:
    manuscript = PROJECT_ROOT / "paper" / "manuscript"
    active_files = [
        *sorted((manuscript / "sections").glob("*.tex")),
        *sorted((manuscript / "appendix").glob("*.tex")),
        manuscript / "main.tex",
        manuscript / "main_body.tex",
        manuscript / "supplemental_appendix.tex",
    ]
    violations = [
        str(path.relative_to(manuscript))
        for path in active_files
        if "factorial" in path.read_text().lower()
    ]
    assert violations == []


def table_body_rows(table: str) -> list[list[str]]:
    """Cells of every body row between the midrule and the bottomrule.

    An italic group heading (a model or a programming language) is printed once above its
    rows. Each returned row starts with its group heading, then the row cells.
    """
    body = table[table.index(r"\midrule") + len(r"\midrule") : table.index(r"\bottomrule")]
    rows, group = [], None
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(r"\addlinespace"):
            continue
        assert line.endswith(r"\\"), line
        heading = re.fullmatch(r"\\multicolumn\{\d+\}\{@\{\}l\}\{\\textit\{(.+)\}\} \\\\", line)
        if heading:
            group = heading.group(1)
            continue
        assert line.startswith(r"\quad "), line
        rows.append([group] + [cell.strip() for cell in line[len(r"\quad ") : -2].split("&")])
    return rows


def outcome_counts(group: list[dict[str, str]]) -> Counter:
    """Count successes and loud and silent failures, and check they agree with the outcome."""
    counts = Counter(row["failure_category"] for row in group)
    assert set(counts) <= {"Success", "Loud failure", "Silent failure"}, set(counts)
    assert counts["Success"] == sum(is_true(row["primary"]) for row in group)
    return counts


@requires_manuscript
def test_gate_by_software_table_rows_sum_to_the_paper_sample() -> None:
    """The installed table by language holds 18 cells of 800 current runs, in 9 rows, plus 3 rows that
    pool the three programming languages (6 cells of 2,400 runs). The pooled rows replace the
    former Table B1 (2026-10-01 appendix reductions, proposal B5)."""
    table = (MANUSCRIPT / "tables" / "appendix_gate_by_software.tex").read_text()
    generated = (CURRENT_ANALYSIS / "tables" / "appendix_gate_by_software.tex").read_text()
    assert generated in table

    runs = current_runs()
    expected, sizes, pooled_sizes = [], [], []
    # 3 programming languages x 3 agency conditions, then all languages, with the 2 models as column groups.
    for platform, platform_label in [*CURRENT_PLATFORMS.items(), ("all", "All languages")]:
        for agency, agency_label in CURRENT_AGENCIES.items():
            row_cells = [platform_label, agency_label]
            for model in CURRENT_MODELS:
                group = [
                    row for row in runs
                    if (row["model"], row["agency"]) == (model, agency)
                    and platform in (row["platform"], "all")
                ]
                counts = outcome_counts(group)
                (pooled_sizes if platform == "all" else sizes).append(len(group))
                row_cells += [
                    one_decimal_percent(counts["Success"], len(group)),
                    one_decimal_percent(counts["Loud failure"], len(group)),
                    one_decimal_percent(counts["Silent failure"], len(group)),
                ]
            expected.append(row_cells)

    assert table_body_rows(table) == expected
    assert len(expected) == 12 and len(sizes) == 18 and len(pooled_sizes) == 6
    # Each language row holds 40 tasks x 2 prompt conditions x 10 repetitions.
    assert all(size == CURRENT_TASKS * 2 * CURRENT_REPS for size in sizes)
    assert all(size == CURRENT_TASKS * 3 * 2 * CURRENT_REPS for size in pooled_sizes)
    assert sum(sizes) == sum(pooled_sizes) == CURRENT_RUNS == len(runs)
    assert "deepseek" not in table.lower()


# --------------------------------------------------------------- figures ----
# The figure builders live in code/exhibits and read the payloads next to
# them. These tests pin the paper scope of the figures: two model arms, the
# per-point run counts that follow from it, and the design strip that states
# the sample and the turn cap of each arm.

FINAL_EXHIBITS = PROJECT_ROOT / "code" / "exhibits"
AGENCY_CONDITIONS = ("chatbot", "one-repair", "constrained agent")
REPS = 12


def exhibit_payload(name: str) -> dict:
    return json.loads((FINAL_EXHIBITS / "data" / name).read_text())


def test_design_figure_states_the_two_model_sample_without_turn_caps() -> None:
    # Decision records 2026-08-24_presentation-renames and
    # 2026-08-24_sonnet-rerun-replacements: the word "arm" left every display
    # surface, and the turn-cap line left the strip because no retained run is
    # bound by a turn cap after the re-run wave. Since 2026-09-25 the template
    # carries the sample counts as macros, which the build fills from the data.
    source = (FINAL_EXHIBITS / "final_fig_design.tex").read_text()
    rendered = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("%")
    )

    assert r"\newcommand{\ntasks}{40}" in rendered
    assert r"\newcommand{\ndatasets}{32}" in rendered
    assert r"\newcommand{\nsetasks}{9}" in rendered
    assert r"\newcommand{\nruns}{14{,}400}" in rendered
    assert r"\newcommand{\nrunspermodel}{7{,}200}" in rendered
    assert "Claude Sonnet 4.6" in rendered and "GPT-5.6 Luna" in rendered
    assert "deepseek" not in rendered.lower()
    assert "Codex" not in rendered
    assert "turn cap" not in rendered
    assert "model arm" not in rendered
    # 32 datasets; the outcome covers every scored value, standard errors included,
    # without singling out the tasks that score one (2026-10-02).
    assert "1 dataset per task" not in rendered
    assert "all requested estimates" not in rendered
    assert "every scored value" in rendered


def test_prompt_agency_payload_pools_the_two_paper_arms() -> None:
    payload = exhibit_payload("ex1_data.json")

    assert sorted(payload["model_iface_pooled"]) == ["codex", "sonnet"]
    # Row A: 21 tasks x 3 programming languages x 2 arms x 12 repetitions.
    for block in payload["prompt_iface_pooled"].values():
        for condition in AGENCY_CONDITIONS:
            assert block[condition]["n"] == 21 * 3 * 2 * REPS
    # Row B: 21 tasks x 2 prompt conditions x 2 arms x 12 repetitions.
    for block in payload["platform_iface_pooled"].values():
        for condition in AGENCY_CONDITIONS:
            assert block[condition]["n"] == 21 * 2 * 2 * REPS


def test_task_figure_payload_pools_the_two_paper_arms() -> None:
    payload = exhibit_payload("ex5_data.json")

    assert sorted(payload["per_task"]) == ["codex", "sonnet"]
    assert len(payload["pooled_all_models"]) == 21
    # One marker pools 2 arms x 2 prompt conditions x 3 programming languages
    # x 12 repetitions.
    for block in payload["pooled_all_models"].values():
        for condition in AGENCY_CONDITIONS:
            assert block[condition]["n"] == 2 * 2 * 3 * REPS


def test_cost_figure_payload_prices_every_paper_cell() -> None:
    cells = exhibit_payload("estimation_cost_data.json")["cells"]

    assert sorted(cells) == ["codex", "sonnet"]
    drawn = 0
    for software_blocks in cells.values():
        for agency_blocks in software_blocks.values():
            for cell in agency_blocks.values():
                assert cell["n"] == 21 * 2 * REPS
                # Every cell carries priced runs, so the figure omits none.
                assert cell["n_priced"] > 0
                assert cell["mean_cost_usd"] is not None
                assert cell["task_success_pct_priced"] is not None
                drawn += 1
    # 2 model arms x 3 programming languages x 3 agency conditions.
    assert drawn == 18

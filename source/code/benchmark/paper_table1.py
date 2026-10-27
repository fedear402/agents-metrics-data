"""Table 1 of the manuscript and a companion table of bootstrap p-values.

Usage (from the repository root):

    uv run --no-sync python code/benchmark/paper_table1.py data/analysis/main [--pvalues] [--out FILE]

``code/paper/manuscript.py`` calls :func:`render` on the analysis folder and writes
``paper/manuscript/tables/main_results.tex`` (Table 1). The p-value table of
:func:`render_pvalues` is not installed in the manuscript. Without ``--out`` the command
prints Table 1, or the p-value table with ``--pvalues``.

Layout (Design A of quality_reports/audits/2026-09-30_table-design/proposals.md, Federico's
choice on 2026-09-30, AER: Insights conventions):

    Table 1    one column per model and a pooled column, numbered (1) to (3). Each contrast
               prints the effect on the first line and the CR1 standard error in parentheses
               on the second line. No p-values, no stars, no vertical rules, no scaling.
    Table D2   the two-sided wild cluster bootstrap p-value and its Holm adjustment for every
               entry of Table 1, two columns per model, in the panels of Table 1.

Both captions use title case without a final period.

Inputs (read only):

    <analysis>/tables/regression_contrasts.csv                 estimates, CR1 SEs, bootstrap p
    <analysis>/data/runs.csv                                   run outcomes for the checks and rates
    <analysis>/data/inference/<model>/bootstrap_results.json   bootstrap p-values and draw counts
    code/benchmark/paper_numbers.py                            the shared half-up rounding

Checks before a table is written (each failure raises ValueError):

    Every printed contrast equals the contrast recomputed from the run-level cell means (1e-9).
    Every p-value equals bootstrap_results.json, and every test used 9,999 draws.
    The run counts equal the rows of runs.csv.

Holm step-down adjustment runs separately in each column within two families:
Panel A (two agency contrasts), and Panels B to D (seven contrasts).
Decision record: quality_reports/decision_records/2026-09-25_table1-p-values-and-holm.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_numbers import fixed  # noqa: E402  shared half-up rounding

MODELS = ("sonnet", "luna", "pooled")
# Two header lines per column in Table 1. Table 1's pooled column is "Total" (CLAUDE.md).
COLUMN_HEADS = {
    "sonnet": ("Claude", "Sonnet 4.6"),
    "luna": ("GPT-5.6", "Luna"),
    "pooled": ("Total", "(pooled)"),
}

# Panels: heading, Holm family, rows. Each row: contrast id, row label.
# The prompt condition with one example is "one-shot" in every reader-facing text
# (Raul Sosa, manuscript commit fd96814). The contrast ids keep their internal names.
PANELS = (
    {
        "title": "Panel A. Agency, relative to the chatbot",
        "family": "primary",
        "rows": (
            # Self-repair returns the output to the model when the script fails or omits a required value.
            ("H3_self_repair", "Self-repair (revises on error or missing value)"),
            ("H3_tool_loop", "Constrained agent (runs and revises its script)"),
        ),
    },
    {
        "title": "Panel B. Prompt, one-shot relative to zero-shot",
        "family": "secondary",
        "rows": (("H2_fewshot", "One example in the prompt"),),
    },
    {
        "title": "Panel C. Language, relative to Python",
        "family": "secondary",
        "rows": (
            ("H1_r", "R, zero-shot chatbot"),
            ("H1_stata", "Stata, zero-shot chatbot"),
            ("GAP_stata_avg", "Stata, average over prompts and agency"),
            ("GAP_stata_agent_zs", "Stata, zero-shot constrained agent"),
        ),
    },
    {
        "title": "Panel D. One-shot effect, change from the chatbot",
        "family": "secondary",
        "rows": (
            ("H4_self_repair", "Under self-repair"),
            ("H4_tool_loop", "Under the constrained agent"),
        ),
    },
)

REFERENCE_ROWS = (
    ("Chatbot, all languages and prompts", "chatbot_all"),
    ("Zero-shot chatbot writing Python", "chatbot_zs_python"),
)

NOTE_LINES = (
    r"\textit{Notes:} Effects are differences in task-success rates, in percentage points, from equation~\eqref{eq:main} with task fixed effects (model-by-task fixed effects in the pooled column 3), averaged over the design dimensions that the row does not name.",
    r"The chatbot rate over all languages and prompts is the reference level of Panel A, and the zero-shot chatbot rate in Python is the reference level of the first two rows of Panel C.",
    r"Standard errors clustered by task (40 clusters) are in parentheses.",
    r"In Panel D, a negative change means that the example adds less with agency.",
)

PVALUE_NOTE_LINES = (
    r"\textit{Notes:} Two-sided wild cluster bootstrap $p$-values for the entries of \Cref{tab:estimates_three_models} (Rademacher weights, 9,999 replications, null imposed, 40 task clusters).",
    r"Holm $p$ applies the Holm step-down adjustment separately in each column and within two families of contrasts.",
    r"Panel A (the two agency contrasts) is one family, and Panels B to D (seven contrasts) are the other.",
    r"Values below 0.001 print as $<$0.001.",
)

# AER: Insights captions: title case, no final period.
CAPTION = "Effects of Programming Language, Prompting, and Agency on Task Success"
LABEL = "tab:estimates_three_models"
PVALUE_CAPTION = r"Wild Cluster Bootstrap $p$-Values for the Contrasts of \Cref{tab:estimates_three_models}"
PVALUE_LABEL = "tab:pvalues"
# "p" lets Table 1 take a page of its own. In the AEA targets the table fills most of a page.
# With "!ht" only, it went to the top of the page where subsection III.B starts, and the
# heading with two lines overfilled that page (2026-10-19 "language" wording).
FLOAT_SPEC = "!htp"
BOOTSTRAP_DRAWS = 9999
TASKS = 40
TOLERANCE = 1e-9
FAMILY_SIZES = {"primary": 2, "secondary": 7}  # stated in PVALUE_NOTE_LINES

# Table 1: the three model columns are right-aligned at their natural widths, and tabular*
# spreads the free space between them. Every number has one decimal, so right alignment also
# aligns the decimal points. Fixed-width columns overflowed the text block (2026-09-30 build).
MODEL_COLUMN = "r"
GAP_BLOCK = "2pt"  # after each contrast
GAP_PANEL = "5pt"  # after the last contrast of a panel


def contrast_ids() -> list[str]:
    """Return the contrast ids in the printed row order."""
    return [cid for panel in PANELS for cid, _ in panel["rows"]]


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values, capped at one."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted


def fmt_effect(value: float) -> str:
    """Format an effect in percentage points with one decimal and a text-width minus."""
    text = fixed(value, 1)
    return f"${{-}}{text[1:]}$" if text.startswith("-") else f"${text}$"


def fmt_p(value: float) -> str:
    """Format a p-value with three decimals. Values below 0.001 print as <0.001."""
    if value < 0.001:
        return "$<$0.001"
    return fixed(value, 3)


def panel_heading(title: str, span: int) -> str:
    """An italic panel heading at the left margin, across the whole table."""
    return r"\multicolumn{" + str(span) + r"}{@{}l}{\textit{" + title + r"}} \\"


def body_rows(table: dict, levels: dict) -> list[str]:
    """Return the Table 1 panel rows, the reference rates, and the sample rows."""
    lines: list[str] = []
    for panel_index, panel in enumerate(PANELS):
        lines.append(panel_heading(panel["title"], 4))
        for row_index, (cid, label) in enumerate(panel["rows"]):
            effects = [fmt_effect(table[m][cid]["estimate_pp"]) + r"\phantom{)}" for m in MODELS]
            errors = ["(" + fixed(table[m][cid]["se_cr1_pp"], 1) + ")" for m in MODELS]
            last_in_panel = row_index == len(panel["rows"]) - 1
            last_panel = panel_index == len(PANELS) - 1
            if last_in_panel and last_panel:
                gap = ""
            elif last_in_panel:
                gap = f"[{GAP_PANEL}]"
            else:
                gap = f"[{GAP_BLOCK}]"
            lines.append(r"\quad " + label + " & " + " & ".join(effects) + r" \\")
            lines.append(" & " + " & ".join(errors) + r" \\" + gap)
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{@{}l}{Task-success rate (percent)} \\")
    for index, (label, key) in enumerate(REFERENCE_ROWS):
        gap = f"[{GAP_BLOCK}]" if index == len(REFERENCE_ROWS) - 1 else ""
        lines.append(r"\quad " + label + " & " + " & ".join(fixed(levels[m][key], 1) for m in MODELS) + r" \\" + gap)
    runs = [f"{table[m]['H3_tool_loop']['n_runs']:,}" for m in MODELS]
    lines.append("Runs & " + " & ".join(runs) + r" \\")
    lines.append("Tasks (clusters) & " + " & ".join([str(TASKS)] * 3) + r" \\")
    lines.append(r"Task fixed effects & Yes & Yes & Model by task \\")
    return lines


def display_path(path: Path) -> str:
    """Return the path relative to the repository root when possible."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def source_comment(what: str, analysis: Path) -> list[str]:
    rel = display_path(analysis)
    return [
        f"% {what}. Generated by code/benchmark/paper_table1.py; do not edit by hand.",
        f"% Sources: {rel}/tables/regression_contrasts.csv, {rel}/data/runs.csv,",
        f"% and {rel}/data/inference/<model>/bootstrap_results.json. Rounding: code/benchmark/paper_numbers.py (half up).",
    ]


def float_head(caption: str, label: str) -> list[str]:
    return [
        r"\begin{table}[" + FLOAT_SPEC + "]",
        r"\setstretch{1}",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{footnotesize}",
    ]


def float_tail(notes: tuple[str, ...]) -> list[str]:
    return [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{footnotesize}",
        r"\begin{flushleft}\footnotesize",
        *notes,
        r"\end{flushleft}",
        r"\end{table}",
    ]


def build_tex(table: dict, levels: dict, analysis: Path) -> str:
    """Return the complete Table 1 float as LaTeX source."""
    first = " & ".join(COLUMN_HEADS[m][0] for m in MODELS)
    second = " & ".join(COLUMN_HEADS[m][1] for m in MODELS)
    head = [
        *source_comment("Table 1, effect over standard error (Design A, AER: Insights style)", analysis),
        *float_head(CAPTION, LABEL),
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l*{3}{" + MODEL_COLUMN + r"}@{}}",
        r"\toprule",
        " & " + first + r" \\",
        " & " + second + r" \\",
        r" & (1) & (2) & (3) \\",
        r"\midrule",
    ]
    return "\n".join(head + body_rows(table, levels) + float_tail(NOTE_LINES)) + "\n"


def build_pvalues_tex(table: dict, analysis: Path) -> str:
    """Return the complete Table D2 float: bootstrap p and Holm p for every entry of Table 1."""
    sizes = {family: sum(len(p["rows"]) for p in PANELS if p["family"] == family) for family in FAMILY_SIZES}
    if sizes != FAMILY_SIZES:
        raise ValueError(f"The Holm families in the Table D2 note do not match the panels: {sizes}")
    groups = " & ".join(r"\multicolumn{2}{c}{" + " ".join(COLUMN_HEADS[m]) + "}" for m in MODELS)
    lines = [
        *source_comment("Table D2, wild cluster bootstrap p-values for the contrasts of Table 1", analysis),
        *float_head(PVALUE_CAPTION, PVALUE_LABEL),
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}"
        r">{\raggedright\arraybackslash\hangindent=2em\hangafter=1}p{4.9cm} rr rr rr@{}}",
        r"\toprule",
        " & " + groups + r" \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(l){6-7}",
        " & " + " & ".join([r"$p$ & Holm $p$"] * 3) + r" \\",
        r"\midrule",
    ]
    for panel_index, panel in enumerate(PANELS):
        if panel_index:
            lines.append(r"\addlinespace[3pt]")
        lines.append(panel_heading(panel["title"], 7))
        for cid, label in panel["rows"]:
            cells = [fmt_p(table[m][cid]["p_wcb"]) + " & " + fmt_p(table[m][cid]["p_holm"]) for m in MODELS]
            lines.append(r"\quad " + label + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + float_tail(PVALUE_NOTE_LINES)) + "\n"


def load_contrasts(analysis: Path) -> dict:
    """Read the printed contrasts and add the Holm-adjusted p-values."""
    frame = pd.read_csv(analysis / "tables/regression_contrasts.csv")
    table: dict[str, dict[str, dict]] = {m: {} for m in MODELS}
    for rec in frame.to_dict("records"):
        if rec["model"] not in table:
            continue
        table[rec["model"]][rec["contrast"]] = {
            "estimate_pp": float(rec["estimate_pp"]),
            "se_cr1_pp": float(rec["se_cr1_pp"]),
            "p_wcb": float(rec["p_wcb"]),
            "n_runs": int(rec["n_runs"]),
            "effective_tasks": int(rec["effective_tasks"]),
            "bootstrap_draws": int(rec["bootstrap_draws"]),
        }
    for m in MODELS:
        missing = [cid for cid in contrast_ids() if cid not in table[m]]
        if missing:
            raise ValueError(f"regression_contrasts.csv lacks {m} rows: {missing}")
        by_family: dict[str, dict[str, float]] = {"primary": {}, "secondary": {}}
        for panel in PANELS:
            for cid, _ in panel["rows"]:
                by_family[panel["family"]][cid] = table[m][cid]["p_wcb"]
        for family, ps in by_family.items():
            for cid, adj in holm(ps).items():
                table[m][cid]["p_holm"] = adj
                table[m][cid]["family"] = family
    return table


def check_against_runs(analysis: Path, table: dict) -> dict[str, dict[str, float]]:
    """Recompute every printed contrast from cell means and return the reference rates."""
    runs = pd.read_csv(analysis / "data/runs.csv", usecols=["model", "platform", "prompt", "agency", "task", "y"])
    levels: dict[str, dict[str, float]] = {}
    per_model: dict[str, dict[str, float]] = {}
    for model in ("sonnet", "luna"):
        d = runs[runs["model"] == model]
        cell = d.groupby(["platform", "prompt", "agency"])["y"].mean() * 100
        agency = cell.groupby("agency").mean()
        prompt = cell.groupby("prompt").mean()
        zs_chat = cell.xs(("zero_shot", "chatbot"), level=("prompt", "agency"))
        zs_agent = cell.xs(("zero_shot", "agent"), level=("prompt", "agency"))
        platform = cell.groupby("platform").mean()

        def oneshot_effect(cond: str) -> float:
            sub = cell.xs(cond, level="agency")
            return sub.xs("few_shot", level="prompt").mean() - sub.xs("zero_shot", level="prompt").mean()

        per_model[model] = {
            "H3_self_repair": agency["self_repair"] - agency["chatbot"],
            "H3_tool_loop": agency["agent"] - agency["chatbot"],
            "H2_fewshot": prompt["few_shot"] - prompt["zero_shot"],
            "H1_r": zs_chat["r"] - zs_chat["python"],
            "H1_stata": zs_chat["stata"] - zs_chat["python"],
            "GAP_stata_avg": platform["stata"] - platform["python"],
            "GAP_stata_agent_zs": zs_agent["stata"] - zs_agent["python"],
            "H4_self_repair": oneshot_effect("self_repair") - oneshot_effect("chatbot"),
            "H4_tool_loop": oneshot_effect("agent") - oneshot_effect("chatbot"),
        }
        levels[model] = {
            "chatbot_all": float(agency["chatbot"]),
            "chatbot_zs_python": float(zs_chat["python"]),
        }
    per_model["pooled"] = {k: (per_model["sonnet"][k] + per_model["luna"][k]) / 2 for k in per_model["sonnet"]}
    levels["pooled"] = {k: (levels["sonnet"][k] + levels["luna"][k]) / 2 for k in levels["sonnet"]}
    for model in MODELS:
        for cid, value in per_model[model].items():
            if abs(float(value) - table[model][cid]["estimate_pp"]) > TOLERANCE:
                raise ValueError(f"Cell-mean check failed for {model} {cid}: "
                                 f"{value} against {table[model][cid]['estimate_pp']}")
    counts = {"sonnet": int((runs["model"] == "sonnet").sum()), "luna": int((runs["model"] == "luna").sum()),
              "pooled": len(runs)}
    for m in MODELS:
        for cid in contrast_ids():
            if table[m][cid]["n_runs"] != counts[m]:
                raise ValueError(f"Run count mismatch for {m} {cid}: {table[m][cid]['n_runs']} against {counts[m]}")
            if table[m][cid]["effective_tasks"] != TASKS:
                raise ValueError(f"Cluster count mismatch for {m} {cid}: {table[m][cid]['effective_tasks']}")
    if runs["task"].nunique() != TASKS:
        raise ValueError(f"runs.csv has {runs['task'].nunique()} tasks, the table prints {TASKS}")
    return levels


def check_bootstrap(analysis: Path, table: dict) -> None:
    """Check every printed p-value and draw count against bootstrap_results.json."""
    for model in MODELS:
        boot = json.loads((analysis / "data/inference" / model / "bootstrap_results.json").read_text())
        for cid in contrast_ids():
            if abs(boot[cid]["p_wcb"] - table[model][cid]["p_wcb"]) > 1e-12:
                raise ValueError(f"Bootstrap p mismatch for {model} {cid}")
            if boot[cid]["bootstrap_draws"] != BOOTSTRAP_DRAWS or table[model][cid]["bootstrap_draws"] != BOOTSTRAP_DRAWS:
                raise ValueError(f"Unexpected draw count for {model} {cid}")


def checked_table(analysis: Path) -> tuple[dict, dict]:
    """Read and check the analysis folder. Return the contrasts and the reference rates."""
    analysis = Path(analysis).resolve()
    table = load_contrasts(analysis)
    levels = check_against_runs(analysis, table)
    check_bootstrap(analysis, table)
    return table, levels


def render(analysis: Path) -> str:
    """Check the analysis folder and return Table 1 as LaTeX source."""
    table, levels = checked_table(analysis)
    return build_tex(table, levels, Path(analysis).resolve())


def render_pvalues(analysis: Path) -> str:
    """Check the analysis folder and return Table D2 as LaTeX source."""
    table, _ = checked_table(analysis)
    return build_pvalues_tex(table, Path(analysis).resolve())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis", type=Path, help="analysis folder, for example data/analysis/main")
    parser.add_argument("--pvalues", action="store_true",
                        help="write Table D2 (bootstrap and Holm p-values) in place of Table 1")
    parser.add_argument("--out", type=Path, help="write the table to this file (default: print it)")
    args = parser.parse_args(argv)
    tex = render_pvalues(args.analysis) if args.pvalues else render(args.analysis)
    if args.out is None:
        sys.stdout.write(tex)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(tex)


if __name__ == "__main__":
    main()

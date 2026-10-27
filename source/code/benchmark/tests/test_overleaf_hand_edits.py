"""The generators reproduce the hand edits of manuscript commits 20ffea4 and fd96814.

Raul Sosa edited four generated texts in Overleaf (20ffea4) and renamed the prompt condition
with one example from few-shot to one-shot (fd96814). A default render must write the same text.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "benchmark"))

import paper_table1  # noqa: E402


def test_table1_note_uses_the_overleaf_wording() -> None:
    note = "\n".join(paper_table1.NOTE_LINES)
    assert "In Panel D, a negative change means that the example adds less with agency." in note
    assert "substitutes" not in note
    assert "percentage-point scale" not in note
    assert "bootstrap" not in note and "Holm" not in note
    assert "few-shot" not in note.lower()


def test_design_figure_uses_american_spelling() -> None:
    source = (PROJECT_ROOT / "code" / "exhibits" / "final_fig_design.tex").read_text()
    assert "labeled lines that the final run printed and wrote" in source
    assert "labelled" not in source
    assert r"\bul one-shot: one example, then the task statement" in source
    assert "few-shot" not in source.lower()


def test_appendix_table_titles_use_the_overleaf_wording() -> None:
    source = (PROJECT_ROOT / "code" / "paper" / "manuscript.py").read_text()
    # AER: Insights captions: title case, no final period (2026-09-30 table-design audit).
    # Table B1 was merged into this table on 2026-10-01 (appendix reductions, proposal B5).
    assert "'Failure Types by Model and Agency Condition'" not in source
    assert "'Task Success and Failure Types by Model, Language, and Agency Condition'" in source
    assert "Primary outcomes by model" not in source

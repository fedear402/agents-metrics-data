"""Regression checks for the explorer analysis-scope strip."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STYLES_PATH = PROJECT_ROOT / "code" / "falstuff_site" / "explore.css"


def test_evidence_strip_styles_outer_cells_only() -> None:
    """Keep bound values from covering their analysis-scope labels."""
    styles = STYLES_PATH.read_text()
    evidence_styles = styles.split(".evidence-strip {", maxsplit=1)[1].split(
        "/* Buttons", maxsplit=1
    )[0]

    assert ".evidence-strip > span {" in styles
    assert ".evidence-strip span {" not in styles
    assert ".evidence-strip > span > span {" in styles
    assert "font-family: Avenir" not in evidence_styles


def test_evidence_strip_gives_the_long_model_list_a_full_row() -> None:
    """Keep the model list readable without making all desktop cells narrow."""
    styles = STYLES_PATH.read_text()

    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in styles
    assert ".evidence-strip > span:last-child {\n  grid-column: 1 / -1;\n}" in styles
    assert "grid-column: auto;" in styles

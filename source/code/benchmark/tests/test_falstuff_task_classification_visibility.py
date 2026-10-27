"""Regression checks for hidden task-source classifications on the public site."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SITE_ROOT = PROJECT_ROOT / "code" / "falstuff_site"


def test_task_source_classification_is_not_rendered() -> None:
    """Keep source metadata out of the task list and task modal."""
    app = (SITE_ROOT / "app.js").read_text()
    explore = (SITE_ROOT / "explore.html").read_text()
    styles = (SITE_ROOT / "explore.css").read_text()

    assert "task-row-origin" not in app
    assert "task-row-origin" not in styles
    assert "<span>Origin</span>" not in explore
    assert '["Origin",' not in app
    assert "Designed DGP" not in app
    assert "StatLLM-style DGP" not in app

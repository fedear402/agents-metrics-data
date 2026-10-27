"""The website task universe must equal the principal manuscript selection."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/falstuff_site"))
import build_task_pages


def test_task_pages_follow_main_selection():
    selected = json.loads((ROOT / "data/analysis/main/data/catalog.json").read_text())
    entries = build_task_pages.production_entries(build_task_pages.load_site_data(), {})
    assert {e["id"] for e in entries} == set(selected)
    assert len(entries) == 40
    assert all(e["status"] == "production" for e in entries)


def test_generated_task_directories_exactly_match_selection():
    selected = json.loads((ROOT / "data/analysis/main/data/catalog.json").read_text())
    directory = ROOT / "code/falstuff_site/tasks"
    manifest = json.loads((directory / "_index.json").read_text())
    assert {e["id"] for e in manifest} == set(selected)
    assert {p.parent.name for p in directory.glob("*/index.html")} == set(selected)


def test_download_preserves_exact_success_counts():
    download = json.loads((ROOT / "code/falstuff_site/paper-results.json").read_text())
    snapshot = json.loads((ROOT / "data/analysis/main/data/snapshot.json").read_text())
    assert sum(m["records"] for m in download["models"]) == 14400
    assert sum(m["taskSuccessPassed"] for m in download["models"]) == snapshot["primary_successes"]

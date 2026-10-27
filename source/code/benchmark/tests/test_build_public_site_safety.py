"""Safety tests for falstuff public-bundle output replacement."""

from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "falstuff_site"))

import build_public_site  # noqa: E402


def empty_bundle_inputs(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    monkeypatch.setattr(build_public_site, "REPO_ROOT", repo_root)
    monkeypatch.setattr(build_public_site, "TEXT_ASSETS", ())
    monkeypatch.setattr(build_public_site, "COPY_DIRS", ())
    monkeypatch.setattr(build_public_site, "DOWNLOADS", {})


def fail_if_removed(path: Path) -> None:
    raise AssertionError(f"unsafe removal attempted: {path}")


def test_existing_unmarked_directory_is_preserved_and_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    out_dir = tmp_path / "existing-output"
    out_dir.mkdir()
    sentinel = out_dir / "user-file.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    empty_bundle_inputs(monkeypatch, repo_root)

    with pytest.raises(ValueError, match="valid bundle marker"):
        build_public_site.build(out_dir, "20260811-1")

    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_existing_directory_with_invalid_marker_is_preserved_and_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    out_dir = tmp_path / "invalid-bundle"
    out_dir.mkdir()
    marker = out_dir / ".falstuff-public-bundle"
    marker.write_text("wrong marker\n", encoding="utf-8")
    empty_bundle_inputs(monkeypatch, repo_root)

    with pytest.raises(ValueError, match="valid bundle marker"):
        build_public_site.build(out_dir, "20260811-1")

    assert marker.read_text(encoding="utf-8") == "wrong marker\n"


def test_repeat_build_replaces_only_a_marked_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    out_dir = tmp_path / "public-bundle"
    empty_bundle_inputs(monkeypatch, repo_root)

    build_public_site.build(out_dir, "20260811-1")
    marker = out_dir / ".falstuff-public-bundle"
    assert marker.read_text(encoding="utf-8") == build_public_site.BUNDLE_MARKER_CONTENT

    stale = out_dir / "stale.txt"
    stale.write_text("remove on repeat build\n", encoding="utf-8")
    build_public_site.build(out_dir, "20260811-1")

    assert not stale.exists()
    assert marker.read_text(encoding="utf-8") == build_public_site.BUNDLE_MARKER_CONTENT


def test_filesystem_root_is_rejected_before_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_bundle_inputs(monkeypatch, Path("/private/tmp/safe-repo-fixture"))
    monkeypatch.setattr(build_public_site.shutil, "rmtree", fail_if_removed)

    with pytest.raises(ValueError, match="filesystem root"):
        build_public_site.build(Path("/"), "20260811-1")


def test_repo_root_is_rejected_before_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    empty_bundle_inputs(monkeypatch, repo_root)
    monkeypatch.setattr(build_public_site.shutil, "rmtree", fail_if_removed)

    with pytest.raises(ValueError, match="contains the repository"):
        build_public_site.build(repo_root, "20260811-1")


def test_repo_ancestor_is_rejected_before_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "workspace" / "repo"
    repo_root.mkdir(parents=True)
    empty_bundle_inputs(monkeypatch, repo_root)
    monkeypatch.setattr(build_public_site.shutil, "rmtree", fail_if_removed)

    with pytest.raises(ValueError, match="contains the repository"):
        build_public_site.build(tmp_path / "workspace", "20260811-1")


def test_real_project_bundle_builds_with_marker(tmp_path: Path) -> None:
    out_dir = tmp_path / "public-bundle"

    build_public_site.build(out_dir, "20260811-1")

    assert (out_dir / "explore.html").is_file()
    assert (out_dir / "downloads" / "agents-metrics-manuscript.pdf").is_file()
    assert (out_dir / build_public_site.BUNDLE_MARKER_NAME).read_text(
        encoding="utf-8"
    ) == build_public_site.BUNDLE_MARKER_CONTENT


def test_public_bundle_contains_only_paper_results(tmp_path: Path) -> None:
    out_dir = tmp_path / "paper-bundle"
    build_public_site.build(out_dir, "20260914-2")
    traces = json.loads((out_dir / "run-traces/_index.json").read_text())
    assert set(traces["models"]) == {"sonnet", "luna"}
    assert not (out_dir / "run-traces/deepseek").exists()
    tasks = json.loads((out_dir / "tasks/_index.json").read_text())
    assert len(tasks) == 40
    catalog = json.loads((build_public_site.REPO_ROOT / "data/analysis/main/data/catalog.json").read_text())
    assert {task["id"] for task in tasks} == set(catalog)
    assert {p.name for p in (out_dir / "tasks").iterdir() if p.is_dir()} == {task["slug"] for task in tasks}
    for model in ("sonnet", "luna"):
        assert {p.stem for p in (out_dir / "run-traces" / model).glob("*.js")} == set(catalog)
    assert not (out_dir / "run-traces/codex").exists()
    assert {task["status"] for task in tasks} == {"production"}
    results = json.loads((out_dir / "downloads/paper-results.json").read_text())
    assert {row["id"] for row in results["models"]} == {"sonnet", "luna"}
    assert "extraResults" not in results
    assert not (out_dir / "downloads/full-analysis.json").exists()
    assert not (out_dir / "downloads/manuscript-numbers.json").exists()
    assert not (out_dir / "downloads/task-registry.json").exists()
    assert not (out_dir / "downloads/current-state.md").exists()


def test_bundle_excludes_stale_tasks_and_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    site = repo / "site"
    catalog = {f"task_{i}": {} for i in range(40)}
    data_dir = repo / "data/analysis/main/data"
    data_dir.mkdir(parents=True)
    (data_dir / "catalog.json").write_text(json.dumps(catalog))
    (site / "tasks").mkdir(parents=True)
    (site / "tasks/index.html").write_text("Selected tasks")
    task_index = [dict(id=t, slug=t) for t in [*catalog, "retired"]]
    (site / "tasks/_index.json").write_text(json.dumps(task_index))
    for task in task_index:
        folder = site / "tasks" / task["slug"]
        folder.mkdir()
        (folder / "index.html").write_text(task["id"])
    for model in ("sonnet", "luna", "codex", "deepseek"):
        folder = site / "run-traces" / model
        folder.mkdir(parents=True)
        for task in [*catalog, "retired"]:
            (folder / f"{task}.js").write_text("{}")
    payload = dict(tasks=[dict(id=t) for t in catalog], models=[dict(id=m) for m in ("sonnet", "luna")],
                   runTraceIndex=dict(models={m: dict(tasks=catalog) for m in ("sonnet", "luna")}))
    (site / "site-data.js").write_text("window.SITE_DATA = " + json.dumps(payload) + ";")
    empty_bundle_inputs(monkeypatch, repo)
    monkeypatch.setattr(build_public_site, "SITE_DIR", site)
    monkeypatch.setattr(build_public_site, "COPY_DIRS", ("tasks", "run-traces"))
    out = tmp_path / "bundle"
    build_public_site.build(out, "20260922-1")
    assert not (out / "tasks/retired").exists()
    assert not (out / "run-traces/codex").exists()
    assert not (out / "run-traces/deepseek").exists()
    for model in ("sonnet", "luna"):
        assert {p.stem for p in (out / "run-traces" / model).iterdir()} == set(catalog)


def test_bundle_rejects_historical_task_payload_before_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    site = repo / "site"
    site.mkdir(parents=True)
    data_dir = repo / "data/analysis/main/data"
    data_dir.mkdir(parents=True)
    (data_dir / "catalog.json").write_text(json.dumps({f"task_{i}": {} for i in range(40)}))
    (site / "site-data.js").write_text('window.SITE_DATA = {"tasks": [{"id": "old_task"}], "models": []};')
    empty_bundle_inputs(monkeypatch, repo)
    monkeypatch.setattr(build_public_site, "SITE_DIR", site)
    monkeypatch.setattr(build_public_site, "COPY_DIRS", ("tasks",))
    out = tmp_path / "bundle"
    with pytest.raises(ValueError, match="forty-task main selection"):
        build_public_site.build(out, "20260922-1")
    assert not out.exists()

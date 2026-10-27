"""Focused tests for the generated data inventory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_data_manifest as manifest  # noqa: E402


def write_record(path: Path, *, prompt_mode: str) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "model": "gpt-5.4",
                "prompt_mode": prompt_mode,
            }
        ),
        encoding="utf-8",
    )


def render_codex_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    data = tmp_path / "data"
    root = data / "runs" / "codex_zero_shot"
    write_record(root / "active" / "run_record.json", prompt_mode="zero_shot")
    write_record(root / "retired-few-shot" / "run_record.json", prompt_mode="few_shot")
    write_record(
        root / "_quarantine" / "old" / "run_record.json",
        prompt_mode="zero_shot",
    )

    monkeypatch.setattr(manifest, "ROOT", tmp_path)
    monkeypatch.setattr(manifest, "DATA", data)
    monkeypatch.setattr(
        manifest,
        "ROLES",
        {"runs/codex_zero_shot": "GPT-5.4 zero-shot production runs."},
    )
    monkeypatch.setattr(manifest, "du", lambda _path: "1K")
    monkeypatch.setattr(manifest, "tracked", lambda _rel: "gitignored")

    manifest.main()
    return (data / "MANIFEST.md").read_text(encoding="utf-8")


def test_manifest_separates_raw_records_from_active_arm_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = render_codex_manifest(tmp_path, monkeypatch)

    assert (
        "| Directory | Role | Size | Raw records | Active-arm records | Git |" in text
    )
    assert (
        "| `data/runs/codex_zero_shot` | GPT-5.4 zero-shot production runs. "
        "| 1K | 3 | 1 | gitignored |" in text
    )


def test_manifest_describes_arm_specific_score_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = render_codex_manifest(tmp_path, monkeypatch)

    assert "Only the Sonnet zero-shot corpus reads persisted `score.json`" in text
    assert "Codex and DeepSeek arms do not require `score.json`" in text
    assert (
        "Each run directory holds `run_record.json` plus the `score.json`" not in text
    )


@pytest.mark.skipif(
    not (manifest.DATA / "analysis/main/data/runs.csv").is_file(),
    reason="current analysis products are absent",
)
def test_current_manifest_lists_derived_products() -> None:
    text = manifest.render_current_manifest()
    for rel, _role in manifest.DERIVED_PRODUCTS:
        assert f"`data/analysis/main/{rel}`" in text
    assert "`data/analysis/main/data/reference_validation.csv`" in text
    assert "`data/analysis/main/data/robustness.json`" in text
    assert "data/inference_dataset/" in text
    assert "carries no cost column" in text

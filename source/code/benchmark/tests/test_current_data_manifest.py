"""Current selection inventory rejects incomplete or inconsistent samples."""

import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_data_manifest as manifest


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def sample(tmp_path):
    folder = tmp_path / "data/analysis/main/data"
    folder.mkdir(parents=True)
    rows, selected = [], []
    for task, model, language, prompt, agency, rep in product(
        ("task_a", "task_b"), ("luna", "sonnet"), ("python", "r", "stata"),
        ("zero_shot", "few_shot"), ("chatbot", "self_repair", "agent"), range(2)
    ):
        position = "__".join(map(str, (task, model, language, prompt, agency, rep)))
        record = tmp_path / "data/runs/current" / position / "run_record.json"
        record.parent.mkdir(parents=True)
        record.write_text("{}")
        item = dict(position_id=position, record_path=str(record),
                    record_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
                    output_path="", output_sha256="")
        selected.append(item)
        rows.append(dict(**item, task=task, model=model, platform=language,
                         prompt=prompt, condition=agency, global_rep=rep))
    write_csv(folder / "runs.csv", rows)
    write_csv(folder / "selected_records.csv", selected)
    (folder / "catalog.json").write_text('{}')
    snapshot = dict(analyzed=len(rows), source_tasks=2, task_count=2,
                    production_repairs_complete=True, replaced_positions=720,
                    runs_sha256=hashlib.sha256((folder / "runs.csv").read_bytes()).hexdigest())
    (folder / "snapshot.json").write_text(json.dumps(snapshot))
    for rel, _role in manifest.DERIVED_PRODUCTS:
        product_path = tmp_path / "data/analysis/main" / rel
        product_path.parent.mkdir(parents=True, exist_ok=True)
        product_path.write_text(rel)
    (tmp_path / "data/collections/reserve").mkdir(parents=True)
    return tmp_path, folder, rows, selected


def render(sample):
    return manifest.render_current_manifest(sample[0], expected_tasks=2, repetitions=2)


def test_current_inventory_counts_selected_rows_and_retained_directories(sample):
    text = render(sample)
    assert "2 tasks and 144 selected responses" in text
    assert "72 model/language/prompt/agency cells has 2 repetitions" in text
    assert "| `data/runs/current` | 144 |" in text
    assert "`data/collections/reserve`" in text
    assert "selected_records.csv" in text
    assert "9,072 runs" not in text
    assert "The complete runs, analysis, and tasks directories are tracked" not in text


def test_derived_products_are_listed_and_required(sample):
    text = render(sample)
    assert "## Derived analysis products" in text
    assert "`data/analysis/main/data/reference_validation.csv`" in text
    (sample[0] / "data/analysis/main/data/robustness.json").unlink()
    with pytest.raises(ValueError, match="Derived analysis product is missing"):
        render(sample)


def test_incomplete_sample_is_rejected(sample):
    _, folder, rows, _ = sample
    write_csv(folder / "runs.csv", rows[:-1])
    with pytest.raises(ValueError, match="row counts"):
        render(sample)


def test_missing_cell_replaced_by_duplicate_is_rejected(sample):
    _, folder, rows, _ = sample
    for field in ("task", "model", "platform", "prompt", "condition", "global_rep"):
        rows[-1][field] = rows[0][field]
    write_csv(folder / "runs.csv", rows)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        render(sample)


def test_selected_index_mismatch_is_rejected(sample):
    _, folder, _, selected = sample
    selected[0]["record_sha256"] = "0" * 64
    write_csv(folder / "selected_records.csv", selected)
    with pytest.raises(ValueError, match="record_sha256 differs"):
        render(sample)


def test_stale_snapshot_hash_is_rejected(sample):
    _, folder, _, _ = sample
    with (folder / "runs.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="hash differs"):
        render(sample)


def test_missing_raw_record_is_rejected(sample):
    Path(sample[3][0]["record_path"]).unlink()
    with pytest.raises(ValueError, match="raw record is missing"):
        render(sample)


def test_pending_replacements_are_rejected(sample):
    path = sample[1] / "snapshot.json"
    snapshot = json.loads(path.read_text())
    snapshot["production_repairs_complete"] = False
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="720 task replacements"):
        render(sample)

#!/usr/bin/env python3
"""Generate data/MANIFEST.md from the filesystem so it cannot drift.

Run: uv run python code/benchmark/build_data_manifest.py
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from itertools import product
import json
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"

ROLES = {
    "tasks/inputs": "Fixed task input datasets (one CSV per simulated dataset).",
    "tasks/references": "Stored reference outputs each run is scored against.",
    "runs/sonnet_zero_shot": (
        "Claude Sonnet 4.6 zero-shot production runs. Sweep logs and derived "
        "reports sit in `_working`."
    ),
    "runs/sonnet_few_shot": "Claude Sonnet 4.6 few-shot production runs.",
    "runs/codex_zero_shot": "GPT-5.4 (Codex CLI) zero-shot production runs.",
    "runs/codex_few_shot": "GPT-5.4 (Codex CLI) few-shot production runs.",
    "runs/deepseek_zero_shot": "DeepSeek V4 Flash zero-shot production runs.",
    "runs/deepseek_few_shot": "DeepSeek V4 Flash few-shot production runs.",
    "runs/_excluded_deepseek_fewshot_strays": (
        "Excluded DeepSeek few-shot strays. The loader excludes this path "
        "explicitly; prompt hashes collide with the valid set."
    ),
    "runs/_pending_sonnet_zeroshot_recollect": (
        "Sonnet zero-shot recollection of five tasks. Outside the analysis "
        "sample until a decision record admits it."
    ),
    "analysis": "Derived analysis products the manuscript reads.",
    "external_corpora/metricsai": "MetricsAI corpus used for task selection.",
    "external_corpora/statllm": "StatLLM corpus used for task selection.",
    "scratch_runs": "Default scratch output for local harness runs.",
    "incoming": "Default output root for new collection waves.",
}

# Derived files of data/analysis/main that the current paper and its appendices read,
# relative to data/analysis/main, with their role in the manifest.
DERIVED_PRODUCTS = (
    ("data/run_pricing.csv", "Paper cost of every run at the API list prices (Section 5, Appendix C)"),
    ("data/pricing.json", "Pricing receipt: rates, sources, and coverage"),
    ("data/term_checks.csv", "Scored comparison of every term with its reference output"),
    ("data/se_checks.csv", "Standard-error checks of the nine tasks that score the standard error"),
    ("tables/regression_contrasts.csv", "Table 1 contrasts, task-clustered standard errors, and bootstrap p-values"),
    ("data/robustness.json", "Appendix D checks, macros, seeds, and dataset-cluster p-values"),
    ("tables/robustness.csv", "Table D1 values at full precision"),
    ("tables/appendix_robustness.tex", "Table D1, robustness of the main contrasts"),
    ("tables/appendix_table1_pvalues_dataset.tex", "Table D3, bootstrap p-values with dataset clusters"),
    ("data/inference_dataset/sonnet/bootstrap_results.json", "Dataset-cluster wild bootstrap, Claude Sonnet 4.6"),
    ("data/inference_dataset/luna/bootstrap_results.json", "Dataset-cluster wild bootstrap, GPT-5.6 Luna"),
    ("data/inference_dataset/pooled/bootstrap_results.json", "Dataset-cluster wild bootstrap, pooled"),
    ("data/inference_dataset/pooled/inference.do", "Stata do-file of the dataset-cluster bootstrap (one per folder)"),
    ("data/reference_validation.csv", "Independent recomputation of the 40 reference outputs"),
)

PRODUCTION_ARM_PROMPTS = {
    "runs/sonnet_zero_shot": "zero_shot",
    "runs/sonnet_few_shot": "few_shot",
    "runs/codex_zero_shot": "zero_shot",
    "runs/codex_few_shot": "few_shot",
    "runs/deepseek_zero_shot": "zero_shot",
    "runs/deepseek_few_shot": "few_shot",
}


def tracked(rel: str) -> str:
    probe = subprocess.run(
        ["git", "check-ignore", "-q", f"data/{rel}/x"],
        cwd=ROOT,
        capture_output=True,
    )
    if probe.returncode == 0:
        return "gitignored"
    listed = subprocess.run(
        ["git", "ls-files", f"data/{rel}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return "tracked" if listed.strip() else "untracked (commit pending)"


def count_records(path: Path) -> int:
    """Count every stored run record below ``path``."""
    return sum(1 for _ in path.rglob("run_record.json"))


def count_active_arm_records(path: Path, prompt_mode: str) -> int:
    """Count records assigned to one production prompt arm.

    Underscore-prefixed directories hold quarantined or superseded records.
    Some zero-shot roots also retain old few-shot records. Neither group is
    active-arm storage coverage.
    """
    count = 0
    for record_path in path.rglob("run_record.json"):
        relative = record_path.relative_to(path)
        if any(part.startswith("_") for part in relative.parts[:-1]):
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("prompt_mode") == prompt_mode:
            count += 1
    return count


def du(path: Path) -> str:
    out = subprocess.run(
        ["du", "-sh", str(path)], capture_output=True, text=True, check=True
    ).stdout
    return out.split()[0]


def main() -> None:
    """Preserved historical inventory API. The CLI defaults to the current paper."""
    lines = [
        "# Data Manifest",
        "",
        f"Generated by `code/benchmark/build_data_manifest.py` on {date.today().isoformat()}.",
        "Do not edit by hand; regenerate instead.",
        "",
        "| Directory | Role | Size | Raw records | Active-arm records | Git |",
        "|---|---|---|---:|---:|---|",
    ]
    for rel, role in ROLES.items():
        path = DATA / rel
        arm_prompt = PRODUCTION_ARM_PROMPTS.get(rel)
        if not path.exists():
            active = "0" if arm_prompt is not None else "-"
            lines.append(f"| `data/{rel}` | {role} | absent | 0 | {active} | - |")
            continue
        raw_records = count_records(path)
        active_records = (
            str(count_active_arm_records(path, arm_prompt))
            if arm_prompt is not None
            else "-"
        )
        lines.append(
            f"| `data/{rel}` | {role} | {du(path)} | {raw_records} | "
            f"{active_records} | {tracked(rel)} |"
        )
    lines += [
        "",
        "Raw records count every recursive `run_record.json` file.",
        "This includes records in named quarantine and pending directories.",
        "Active-arm records appear only for the six production arms.",
        "They exclude underscore-prefixed subdirectories and records whose "
        "stored `prompt_mode` does not match the arm.",
        "The paper sample contains twelve repetitions per cell and 9,072 runs.",
        "The complete runs, analysis, and tasks directories are tracked in Git.",
        "Verify their original bytes with `uv run --no-project code/benchmark/verify_data.py`.",
        "",
        "Every arm directory holds one directory per run.",
        "Underscore-prefixed directories hold everything else: `_working` for "
        "sweep logs, aggregates, and derived reports, `_retired_prompt_mode` "
        "for records of a retired prompt specification.",
        "",
        "The analysis sample loads through `code/benchmark/analysis_dataset.py`.",
        "Each saved run keeps `run_record.json` and its artifacts in one directory.",
        "Only the Sonnet zero-shot corpus reads persisted `score.json` files.",
        "`strict_posthoc.py --write-run-scores` writes those files.",
        "Sonnet few-shot, Codex, and DeepSeek recompute canonical scoring gates "
        "in memory from saved artifacts.",
        "Codex and DeepSeek arms do not require `score.json`.",
        "Their inline scoring blocks are collection telemetry, not analysis scores.",
        "`task_success_report.json` is a derived summary of the Sonnet zero-shot "
        "score rows.",
        "Archived probe and screen data live under `archive/`.",
        "",
    ]
    (DATA / "MANIFEST.md").write_text("\n".join(lines))
    print(f"wrote {DATA / 'MANIFEST.md'}")


def render_current_manifest(root: Path = ROOT, *, expected_tasks: int = 40,
                            repetitions: int = 10) -> str:
    """Validate the selected sample and describe its actual local storage.

    Raw-directory listings are shallow. Selected counts are not counts of all
    stored attempts. Historical and candidate outcomes remain separate.
    """
    root = root.resolve()
    data = root / "data"
    analysis = data / "analysis/main/data"
    snapshot = json.loads((analysis / "snapshot.json").read_text())
    with (analysis / "runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (analysis / "selected_records.csv").open(newline="") as handle:
        selected = list(csv.DictReader(handle))
    expected_count = expected_tasks * 2 * 3 * 2 * 3 * repetitions
    if len(rows) != expected_count or len(selected) != expected_count:
        raise ValueError("Current sample row counts do not match the complete design")
    if snapshot.get("analyzed") != expected_count or snapshot.get("task_count", snapshot.get("source_tasks")) != expected_tasks:
        raise ValueError("Snapshot counts do not match the selected sample")
    if not snapshot.get("production_repairs_complete") or snapshot.get("replaced_positions") != 720:
        raise ValueError("The current paper requires all 720 task replacements")
    tasks = {row["task"] for row in rows}
    if len(tasks) != expected_tasks:
        raise ValueError("Unexpected task coverage")
    dimensions = (tasks, ("luna", "sonnet"), ("python", "r", "stata"),
                  ("zero_shot", "few_shot"), ("chatbot", "self_repair", "agent"))
    expected = {(*cell, rep) for cell in product(*dimensions) for rep in range(repetitions)}
    positions = [(row["task"], row["model"], row["platform"], row["prompt"],
                  row["condition"], int(row["global_rep"])) for row in rows]
    if len(set(positions)) != len(positions) or set(positions) != expected:
        raise ValueError("Missing or duplicate task/model/language/prompt/agency/repetition position")
    selected_map = {row["position_id"]: row for row in selected}
    if len(selected_map) != len(selected) or len({row["position_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate selected position identifier")
    if set(selected_map) != {row["position_id"] for row in rows}:
        raise ValueError("Selected-record index differs from analysis positions")
    record_paths = set()
    source_counts = Counter()
    for row in rows:
        item = selected_map[row["position_id"]]
        for field in ("record_path", "record_sha256", "output_path", "output_sha256"):
            if row[field] != item[field]:
                raise ValueError(f"Selected-record {field} differs at {row['position_id']}")
        path = Path(item["record_path"])
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"Selected raw record is missing: {path}")
        if path in record_paths:
            raise ValueError(f"Raw record selected more than once: {path}")
        record_paths.add(path)
        try:
            parts = path.relative_to(data).parts
        except ValueError as exc:
            raise ValueError(f"Selected raw record is outside data/: {path}") from exc
        source_counts[str(Path("data", *parts[:2]))] += 1
    run_hash = hashlib.sha256((analysis / "runs.csv").read_bytes()).hexdigest()
    if snapshot.get("runs_sha256") != run_hash:
        raise ValueError("Current runs.csv hash differs from snapshot")
    cells = expected_count // repetitions
    lines = ["# Data Manifest", "",
             f"Generated by `code/benchmark/build_data_manifest.py` on {date.today().isoformat()}.",
             "Do not edit by hand. Regenerate from the saved current selection.", "",
             "## Principal local sample", "",
             f"The current paper uses {expected_tasks} tasks and {expected_count:,} selected responses. Each of {cells:,} model/language/prompt/agency cells has {repetitions} repetitions.", "",
             "Models are Claude Sonnet 4.6 and GPT-5.6 Luna. Languages are Python, R, and Stata. Prompts are zero-shot and few-shot. Agency conditions are chatbot, self-repair, and constrained agent. Stata prints its results. All 720 September 21 LP-IV and PPML replacements are included.", "",
             "The primary score checks selected substantive estimates and, in the nine tasks whose named decision is the standard error, the scored standard errors. The other standard-error fields and all count fields remain diagnostic.", "",
             "| Selection file | SHA-256 |", "|---|---|"]
    for name in ("snapshot.json", "runs.csv", "selected_records.csv", "catalog.json"):
        path = analysis / name
        lines.append(f"| `{path.relative_to(root)}` | `{hashlib.sha256(path.read_bytes()).hexdigest()}` |")
    lines += ["", "`runs.csv`, `selected_records.csv`, and `selection_lineage.csv` store record, output, and manifest paths relative to the repository root. `runs.csv` carries no cost column. The paper cost of each run is in `run_pricing.csv`.", ""]
    lines += ["## Derived analysis products", "",
              "`uv run python code/paper/rebuild.py` regenerates these files from the selection above.", "",
              "| File | Content | SHA-256 |", "|---|---|---|"]
    for rel, role in DERIVED_PRODUCTS:
        path = data / "analysis/main" / rel
        if not path.is_file():
            raise ValueError(f"Derived analysis product is missing: {path}")
        lines.append(f"| `{path.relative_to(root)}` | {role} | `{hashlib.sha256(path.read_bytes()).hexdigest()}` |")
    lines += ["", "## Selected record locations", "",
              "These counts describe selected responses only. Each location can also contain earlier attempts, unused candidates, or historical responses.", "",
              "| Existing source directory | Selected responses |", "|---|---:|"]
    lines += [f"| `{path}` | {count:,} |" for path, count in sorted(source_counts.items())]
    lines += ["", "## Other local data", "",
              "This shallow directory listing records existing storage. It does not recursively count all raw records or certify their eligibility.", "",
              "| Existing directory | Relation to the current selection |", "|---|---|"]
    for base in (data / "runs", data / "collections", data / "production"):
        if not base.exists():
            continue
        for path in sorted(base.iterdir()):
            if not path.is_dir():
                continue
            rel = str(path.relative_to(root))
            if rel in source_counts:
                role = "Contains selected responses and can contain other saved material"
            elif path.name == "main" and base.name == "production":
                role = "Current starting selection and preserved manuscript baseline"
            else:
                role = "Collection inputs, candidate or historical material; not a selected raw-record root"
            lines.append(f"| `{rel}` | {role} |")
    lines += ["", "## Rebuild and archive", "",
              "Run `uv run python code/paper/rebuild.py` to verify saved evidence and rebuild the current paper. Run `uv run python code/benchmark/build_data_manifest.py` to refresh this inventory. Neither command calls a model provider.", "",
              "Superseded publication corpora and reports are archived at `~/ARCHIVE/AGENTS-METRICS/`. Cleanup receipts under `data/maintenance/` record hashes and restoration paths. Current selected responses and retained candidates remain under `data/`.", "",
              "`INVENTORY.jsonl.gz` describes the historical August release. It does not verify every current local collection. Git tracking is not inferred from this inventory. A fresh clone can lack local collections.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", action="store_true", help="Regenerate the earlier publication inventory")
    args = parser.parse_args()
    if args.historical:
        main()
    else:
        (DATA / "MANIFEST.md").write_text(render_current_manifest(), encoding="utf-8")
        print(f"wrote {DATA / 'MANIFEST.md'}")

"""Summarize current benchmark failure patterns from saved run records."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_RUNS_DIR = Path(__file__).parent / "runs"

DISPLAY_LABELS = {
    "artifact_failed": "result_file_failed",
    "method_metadata_missing": "specification_metadata_missing",
}


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def load_records(runs_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(runs_dir.glob("*/run_record.json")):
        records.append(json.loads(path.read_text()))
    return records


def failure_labels(record: dict[str, Any]) -> list[str]:
    scoring = record.get("scoring") or {}
    labels = []

    if not record.get("execution_success"):
        labels.append("execution_failed")
    if not scoring.get("artifact_success"):
        labels.append("artifact_failed")
    if scoring.get("method_success") is False:
        labels.append("method_metadata_missing")
    if scoring.get("numeric_success") is False:
        labels.append("numeric_failed")

    for note in scoring.get("notes") or []:
        labels.append(f"note:{note}")

    for term, details in (scoring.get("per_term") or {}).items():
        if details.get("coef_pass") is False:
            labels.append(f"coef_failed:{term}")
        if details.get("se_pass") is False:
            labels.append(f"se_failed:{term}")

    return labels or ["passed"]


def build_report(records: list[dict[str, Any]], max_examples: int = 8) -> dict[str, Any]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failed_records = []

    for record in records:
        labels = failure_labels(record)
        if labels == ["passed"]:
            continue

        failed_records.append(record)
        dimensions = {
            "task": record.get("task_id", "<unknown>"),
            "platform": record.get("platform", "<unknown>"),
            "exec_mode": record.get("exec_mode", "<unknown>"),
            "prompt_mode": record.get("prompt_mode", "<unknown>"),
        }
        for label in labels:
            for dim, value in dimensions.items():
                groups[f"{dim}:{value}"][label] += 1
            groups["all"][label] += 1
            if len(examples[label]) < max_examples:
                examples[label].append(
                    {
                        "cell_id": record.get("cell_id"),
                        "task_id": record.get("task_id"),
                        "platform": record.get("platform"),
                        "exec_mode": record.get("exec_mode"),
                        "prompt_mode": record.get("prompt_mode"),
                    }
                )

    return {
        "n_records": len(records),
        "n_failed_records": len(failed_records),
        "failure_rate": (len(failed_records) / len(records)) if records else 0.0,
        "groups": {
            group: counter.most_common()
            for group, counter in sorted(groups.items())
        },
        "examples": dict(sorted(examples.items())),
    }


def markdown_summary(report: dict[str, Any], top_n: int = 12) -> str:
    lines = [
        "# Failure Pattern Report",
        "",
        (
            f"- Records inspected: **{report['n_records']}**  "
            f"({report['n_failed_records']} with at least one non-pass label)"
        ),
        f"- Failure-labeled share: **{report['failure_rate']:.1%}**",
        "",
        (
            "This report uses the current harness' simplified execution, result-file, "
            "method-metadata, and numeric checks. It is a triage report for "
            "current saved records."
        ),
        "",
        "## Overall Labels",
        "| label | count |",
        "|---|---:|",
    ]

    for label, count in report["groups"].get("all", [])[:top_n]:
        lines.append(f"| `{display_label(label)}` | {count} |")

    lines.extend(["", "## Top Task/Platform/Mode Groups"])
    lines.extend(["| group | top labels |", "|---|---|"])
    group_rows = [
        (group, values)
        for group, values in report["groups"].items()
        if group != "all"
    ]
    group_rows.sort(key=lambda item: sum(count for _, count in item[1]), reverse=True)
    for group, values in group_rows[:top_n]:
        label_text = "; ".join(
            f"`{display_label(label)}`={count}" for label, count in values[:3]
        )
        lines.append(f"| `{group}` | {label_text} |")

    lines.extend(["", "## Example Cells"])
    for label, records in list(report["examples"].items())[:top_n]:
        cell_text = ", ".join(f"`{record['cell_id']}`" for record in records)
        lines.append(f"- `{display_label(label)}`: {cell_text}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-examples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.expanduser().resolve()
    # Derived output is collection state. The default keeps the arm root free
    # of loose files; keep the name equal to WORKING_DIRNAME in harness.py.
    out_dir = (args.out_dir or runs_dir / "_working").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(load_records(runs_dir), max_examples=args.max_examples)
    (out_dir / "failure_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "failure_report.md").write_text(markdown_summary(report))
    print(out_dir / "failure_report.md")


if __name__ == "__main__":
    main()

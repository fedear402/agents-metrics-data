"""Upload strict_posthoc score rows to the central DB, keyed by run uuid.

Reads a task_success_report.json produced by strict_posthoc.py, maps each
score row's ``run_dir`` to the run uuid by reading that directory's
run_record.json under ``--runs-dir`` (only the directory basename is used, so
reports generated on another machine map cleanly), and pushes
{run_uuid, scoring_version, task_success, quarantined, prompt_wave, score}
rows via ``QueueClient.push_scores``.

Usage:
    python code/benchmark/push_scores.py \
        --report data/generated/benchmark_runs/task_success_report.json \
        --runs-dir data/generated/benchmark_runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from queue_client import QueueClient, QueueError  # noqa: E402


def build_score_rows(
    report: dict[str, Any], runs_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Map report score rows to push rows; returns (rows, summary counts)."""
    scoring_version = report.get("scoring_version")
    counts = {
        "rows_in_report": 0,
        "mapped": 0,
        "skipped_no_record": 0,
        "skipped_no_uuid": 0,
    }
    rows: list[dict[str, Any]] = []
    for row in report.get("scores", []):
        counts["rows_in_report"] += 1
        run_dir_value = row.get("run_dir")
        record_path = (
            runs_dir / Path(str(run_dir_value)).name / "run_record.json"
            if run_dir_value
            else None
        )
        if record_path is None or not record_path.exists():
            print(
                f"WARNING no run_record.json for score row run_dir={run_dir_value}",
                file=sys.stderr,
            )
            counts["skipped_no_record"] += 1
            continue
        try:
            record = json.loads(record_path.read_text())
        except (OSError, json.JSONDecodeError) as err:
            print(f"WARNING unreadable record {record_path}: {err}", file=sys.stderr)
            counts["skipped_no_record"] += 1
            continue
        run_uuid = record.get("uuid") if isinstance(record, dict) else None
        if not run_uuid:
            print(f"WARNING record without uuid: {record_path}", file=sys.stderr)
            counts["skipped_no_uuid"] += 1
            continue
        rows.append(
            {
                "run_uuid": str(run_uuid),
                "scoring_version": scoring_version,
                "task_success": bool(row.get("task_success")),
                "quarantined": bool(row.get("quarantined")),
                "prompt_wave": row.get("prompt_wave", "legacy"),
                "score": row,
            }
        )
        counts["mapped"] += 1
    return rows, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report", required=True, type=Path, help="path to task_success_report.json"
    )
    parser.add_argument(
        "--runs-dir",
        required=True,
        type=Path,
        help="local runs dir used to map run_dir basenames to record uuids",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.runs_dir.is_dir():
        parser.error(f"--runs-dir is not a directory: {args.runs_dir}")
    try:
        report = json.loads(args.report.read_text())
    except (OSError, json.JSONDecodeError) as err:
        parser.error(f"cannot read --report {args.report}: {err}")

    rows, counts = build_score_rows(report, args.runs_dir)
    summary: dict[str, int] = dict(counts)
    if args.dry_run:
        summary["would_push"] = len(rows)
        summary["pushed"] = 0
        summary["errors"] = 0
    else:
        try:
            QueueClient().push_scores(rows)
            summary["pushed"] = len(rows)
            summary["errors"] = 0
        except QueueError as err:
            print(f"WARNING push_scores failed: {err}", file=sys.stderr)
            summary["pushed"] = 0
            summary["errors"] = len(rows)
    print("summary: " + " ".join(f"{key}={val}" for key, val in summary.items()))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

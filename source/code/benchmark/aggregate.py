"""Aggregate per-cell run records into a compact manuscript summary.

Reads all ``<runs-dir>/<cell>/run_record.json`` files, falling back to the
harness summary file only when no per-cell records are present. This matters
for self-repair, where Claude usage is split across ``claude_attempt_*`` blocks.
Produces, under ``<runs-dir>/_working/`` by default:
  - aggregate.json  (all rolled-up numbers)
  - aggregate.md    (human-readable manuscript pull-quotes)
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"

USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def number_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def median(xs: list[Any]) -> float | None:
    clean = [number_or_zero(x) for x in xs if x is not None]
    return stats.median(clean) if clean else None


def load_records(runs_dir: Path = RUNS) -> list[dict[str, Any]]:
    records = []
    for path in sorted(runs_dir.glob("*/run_record.json")):
        records.append(json.loads(path.read_text()))
    if records:
        return records

    for summary_path in (runs_dir / "_working" / "summary.json", runs_dir / "summary.json"):
        if summary_path.exists():
            data = json.loads(summary_path.read_text())
            if isinstance(data, list):
                return data
    return []


def claude_blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = []
    for key in sorted(record):
        value = record.get(key)
        if key == "claude" or key.startswith("claude_attempt_"):
            if isinstance(value, dict):
                blocks.append(value)
    return blocks


def sum_usage(record: dict[str, Any]) -> dict[str, int]:
    totals = {key: 0 for key in USAGE_KEYS}
    for block in claude_blocks(record):
        usage = block.get("usage") or {}
        for key in USAGE_KEYS:
            totals[key] += int(number_or_zero(usage.get(key)))
    return totals


def record_cost(record: dict[str, Any]) -> float:
    return sum(number_or_zero(block.get("total_cost_usd")) for block in claude_blocks(record))


def record_wall(record: dict[str, Any]) -> float:
    return sum(number_or_zero(block.get("wall_seconds")) for block in claude_blocks(record))


def record_turns(record: dict[str, Any]) -> int | None:
    total = 0
    seen = False
    for block in claude_blocks(record):
        turns = block.get("num_turns", block.get("n_turns"))
        if turns is not None:
            total += int(number_or_zero(turns))
            seen = True
    return total if seen else None


def stop_reasons(record: dict[str, Any]) -> str | None:
    reasons = [
        str(block.get("stop_reason"))
        for block in claude_blocks(record)
        if block.get("stop_reason")
    ]
    return "; ".join(reasons) if reasons else None


def rows_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        scoring = rec.get("scoring") or {}
        usage = sum_usage(rec)
        rows.append(
            {
                "cell_id": rec.get("cell_id"),
                "task": rec.get("task_id"),
                "platform": rec.get("platform"),
                "exec_mode": rec.get("exec_mode"),
                "prompt_mode": rec.get("prompt_mode"),
                "execution_success": bool(rec.get("execution_success")),
                "artifact_success": bool(scoring.get("artifact_success")),
                "numeric_success": bool(scoring.get("numeric_success")),
                "method_success": bool(scoring.get("method_success")),
                "cost_usd": record_cost(rec),
                "wall_seconds": record_wall(rec),
                "num_turns": record_turns(rec),
                "tokens_in": usage["input_tokens"],
                "tokens_out": usage["output_tokens"],
                "tokens_cache_read": usage["cache_read_input_tokens"],
                "tokens_cache_create": usage["cache_creation_input_tokens"],
                "stop_reason": stop_reasons(rec),
                "notes": scoring.get("notes", []),
            }
        )
    return rows


def success_rate(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(bool(row[key]) for row in rows) / len(rows), 3)


def build_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = rows_from_records(records)
    aggregate = {
        "n_runs": len(rows),
        "n_execution_success": sum(row["execution_success"] for row in rows),
        "n_numeric_success": sum(row["numeric_success"] for row in rows),
        "total_cost_usd": round(sum(row["cost_usd"] for row in rows), 4),
        "total_wall_seconds": round(sum(row["wall_seconds"] for row in rows), 1),
        "median_cost_per_run_usd": round(median([row["cost_usd"] for row in rows]) or 0, 4),
        "by_exec_mode": {},
        "by_platform": {},
        "by_prompt_platform": {},
        "by_mode_platform": {},
        "by_task": {},
        "rows": rows,
    }

    for mode in sorted({row["exec_mode"] for row in rows if row["exec_mode"]}):
        subset = [row for row in rows if row["exec_mode"] == mode]
        aggregate["by_exec_mode"][mode] = {
            "n": len(subset),
            "exec_success_rate": success_rate(subset, "execution_success"),
            "artifact_success_rate": success_rate(subset, "artifact_success"),
            "method_success_rate": success_rate(subset, "method_success"),
            "numeric_success_rate": success_rate(subset, "numeric_success"),
            "median_cost_usd": round(median([row["cost_usd"] for row in subset]) or 0, 4),
            "median_wall_s": round(median([row["wall_seconds"] for row in subset]) or 0, 1),
            "median_turns": median([row["num_turns"] for row in subset]),
            "median_tokens_out": median([row["tokens_out"] for row in subset]),
        }

    for platform in sorted({row["platform"] for row in rows if row["platform"]}):
        subset = [row for row in rows if row["platform"] == platform]
        aggregate["by_platform"][platform] = {
            "n": len(subset),
            "exec_success_rate": success_rate(subset, "execution_success"),
            "numeric_success_rate": success_rate(subset, "numeric_success"),
            "median_cost_usd": round(median([row["cost_usd"] for row in subset]) or 0, 4),
        }

    prompt_platform_keys = sorted(
        {
            (row["platform"], row["prompt_mode"])
            for row in rows
            if row["platform"] and row["prompt_mode"]
        }
    )
    for platform, prompt_mode in prompt_platform_keys:
        subset = [
            row
            for row in rows
            if row["platform"] == platform and row["prompt_mode"] == prompt_mode
        ]
        aggregate["by_prompt_platform"][f"{platform}__{prompt_mode}"] = {
            "n": len(subset),
            "exec_success": sum(row["execution_success"] for row in subset),
            "numeric_success": sum(row["numeric_success"] for row in subset),
            "median_cost_usd": round(median([row["cost_usd"] for row in subset]) or 0, 4),
        }

    keys = sorted(
        {
            (row["exec_mode"], row["platform"])
            for row in rows
            if row["exec_mode"] and row["platform"]
        }
    )
    for mode, platform in keys:
        subset = [
            row
            for row in rows
            if row["exec_mode"] == mode and row["platform"] == platform
        ]
        aggregate["by_mode_platform"][f"{mode}__{platform}"] = {
            "n": len(subset),
            "exec_success": sum(row["execution_success"] for row in subset),
            "numeric_success": sum(row["numeric_success"] for row in subset),
            "median_cost_usd": round(median([row["cost_usd"] for row in subset]) or 0, 4),
        }

    for task in sorted({row["task"] for row in rows if row["task"]}):
        subset = [row for row in rows if row["task"] == task]
        aggregate["by_task"][task] = {
            "n": len(subset),
            "exec_success": sum(row["execution_success"] for row in subset),
            "numeric_success": sum(row["numeric_success"] for row in subset),
        }

    return aggregate


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:.1f}%"


def markdown_summary(aggregate: dict[str, Any], target_runs: int | None = None) -> str:
    md = []
    md.append("# Benchmark run summary\n")
    md.append(f"- Runs: **{aggregate['n_runs']}**  ")
    if target_runs:
        md.append(
            f"- Coverage: **{aggregate['n_runs']}/{target_runs}** "
            f"({pct(aggregate['n_runs'], target_runs)})  "
        )
    md.append(
        "- Execution success: "
        f"**{aggregate['n_execution_success']}/{aggregate['n_runs']}**  "
    )
    md.append(
        "- Numeric success (vs reference, within tolerance): "
        f"**{aggregate['n_numeric_success']}/{aggregate['n_runs']}**  "
    )
    md.append(f"- Total cost: **${aggregate['total_cost_usd']}**  ")
    md.append(f"- Total wall time: **{aggregate['total_wall_seconds']} s**  ")
    md.append(f"- Median cost per run: **${aggregate['median_cost_per_run_usd']}**\n")

    md.append("## By execution mode")
    md.append(
        "| mode | n | exec | result file | method | numeric | median cost | "
        "median wall (s) | median turns |"
    )
    md.append(
        "|------|---|------|----------|--------|---------|-------------|"
        "------------------|---------------|"
    )
    for mode, data in aggregate["by_exec_mode"].items():
        md.append(
            f"| {mode} | {data['n']} | {data['exec_success_rate']} | "
            f"{data['artifact_success_rate']} | {data['method_success_rate']} | "
            f"{data['numeric_success_rate']} | ${data['median_cost_usd']} | "
            f"{data['median_wall_s']} | {data['median_turns']} |"
        )
    md.append("")

    md.append("## By prompt x platform")
    md.append("| cell | n | exec | numeric | median cost |")
    md.append("|------|---|------|---------|-------------|")
    for key, data in aggregate["by_prompt_platform"].items():
        md.append(
            f"| {key} | {data['n']} | {data['exec_success']} | "
            f"{data['numeric_success']} | ${data['median_cost_usd']} |"
        )
    md.append("")

    md.append("## By mode x platform")
    md.append("| cell | n | exec | numeric | median cost |")
    md.append("|------|---|------|---------|-------------|")
    for key, data in aggregate["by_mode_platform"].items():
        md.append(
            f"| {key} | {data['n']} | {data['exec_success']} | "
            f"{data['numeric_success']} | ${data['median_cost_usd']} |"
        )
    md.append("")

    md.append("## By task")
    md.append("| task | n | exec | numeric |")
    md.append("|------|---|------|---------|")
    for task, data in aggregate["by_task"].items():
        md.append(
            f"| {task} | {data['n']} | {data['exec_success']} | "
            f"{data['numeric_success']} |"
        )
    return "\n".join(md)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark run records into JSON and Markdown reports."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS,
        help="Directory containing per-cell run folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Directory for aggregate.json and aggregate.md. Default: --runs-dir.",
    )
    parser.add_argument(
        "--target-runs",
        type=int,
        default=None,
        help="Optional denominator for coverage reporting, e.g. 11880.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.expanduser().resolve()
    # Derived output is collection state. The default keeps the arm root free
    # of loose files; keep the name equal to WORKING_DIRNAME in harness.py.
    out_dir = (args.out_dir or args.runs_dir / "_working").expanduser().resolve()
    records = load_records(runs_dir)
    if not records:
        raise SystemExit(f"No run records found under {runs_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = build_aggregate(records)
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    (out_dir / "aggregate.md").write_text(
        markdown_summary(aggregate, target_runs=args.target_runs)
    )
    print((out_dir / "aggregate.md").read_text())


if __name__ == "__main__":
    main()

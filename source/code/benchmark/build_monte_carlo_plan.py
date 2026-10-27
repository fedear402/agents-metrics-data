"""Build a concrete Monte Carlo expansion plan from the task registry."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
DEFAULT_REGISTRY = ROOT / "task_registry.json"
DEFAULT_OUT_DIR = ROOT.parent.parent / "data/generated/2026-05-19_monte_carlo_plan"
CELLS_PER_DATASET = 3 * 2 * 3


def same_value(left: Any, right: Any) -> bool:
    return str(left) == str(right)


def build_variants(registry: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = registry["monte_carlo_design"]["default_seeds"]
    variants: list[dict[str, Any]] = []
    for task in registry["tasks"]:
        task_id = task["task_id"]
        baseline = task["dataset"].get("baseline_parameters", {})
        grid = task["monte_carlo_grid"]
        for axis in task["monte_carlo_axes"]:
            baseline_value = baseline.get(axis)
            for value in grid[axis]:
                if baseline_value is not None and same_value(value, baseline_value):
                    continue
                value_slug = str(value).replace(".", "p").replace("-", "m")
                for seed in seeds:
                    variants.append(
                        {
                            "variant_id": (
                                f"{task_id}__axis_{axis}__value_{value_slug}"
                                f"__seed_{seed}"
                            ),
                            "task_id": task_id,
                            "title": task["title"],
                            "source_data_file": task["data_file"],
                            "generator": task["dataset"]["generator"],
                            "axis": axis,
                            "value": value,
                            "seed": seed,
                            "baseline_parameters": baseline,
                            "parameter_overrides": {axis: value},
                            "status": "planned_dataset_and_reference_needed",
                        }
                    )
    return variants


def build_plan(registry: dict[str, Any]) -> dict[str, Any]:
    variants = build_variants(registry)
    by_task = Counter(variant["task_id"] for variant in variants)
    by_axis = Counter(variant["axis"] for variant in variants)
    return {
        "version": "2026-05-19",
        "source_registry": registry.get("source_of_truth"),
        "registry_status": registry.get("status"),
        "policy": registry["monte_carlo_design"]["grid_policy"],
        "default_seeds": registry["monte_carlo_design"]["default_seeds"],
        "cells_per_dataset": CELLS_PER_DATASET,
        "n_active_tasks": len(registry["tasks"]),
        "n_variants": len(variants),
        "implied_llm_runs_at_30_reps": len(variants) * CELLS_PER_DATASET * 30,
        "by_task": dict(sorted(by_task.items())),
        "by_axis": dict(sorted(by_axis.items())),
        "variants": variants,
    }


def markdown_summary(plan: dict[str, Any]) -> str:
    lines = [
        "# Monte Carlo Expansion Plan",
        "",
        f"- Active tasks: **{plan['n_active_tasks']}**",
        f"- Planned one-axis variants: **{plan['n_variants']}**",
        f"- Seeds: **{', '.join(str(seed) for seed in plan['default_seeds'])}**",
        f"- Cells per generated dataset: **{plan['cells_per_dataset']}**",
        (
            "- Implied LLM runs at 30 reps: "
            f"**{plan['implied_llm_runs_at_30_reps']}**"
        ),
        "",
        "Policy: vary one axis at a time around the active baseline first. "
        "Generate full-factorial grids only for task families where the "
        "one-axis pass reveals a real difficulty gradient.",
        "",
        "## Variants by Task",
        "",
        "| task | planned variants |",
        "|---|---:|",
    ]
    for task, count in plan["by_task"].items():
        lines.append(f"| {task} | {count} |")

    lines.extend(
        [
            "",
            "## Most Common Axes",
            "",
            "| axis | planned variants |",
            "|---|---:|",
        ]
    )
    for axis, count in sorted(
        plan["by_axis"].items(), key=lambda item: (-item[1], item[0])
    )[:20]:
        lines.append(f"| {axis} | {count} |")

    lines.extend(
        [
            "",
            "## Status",
            "",
            "These variants are a generation plan, not live-run inputs yet. "
            "Each variant still needs a generated dataset, platform-specific "
            "reference outputs, scoring tolerances, and integrity checks before "
            "it can be added to a live model sweep.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Monte Carlo expansion plan from task_registry.json."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = json.loads(args.registry.expanduser().read_text())
    plan = build_plan(registry)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "monte_carlo_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (out_dir / "monte_carlo_plan.md").write_text(markdown_summary(plan))
    print(out_dir / "monte_carlo_plan.md")


if __name__ == "__main__":
    main()

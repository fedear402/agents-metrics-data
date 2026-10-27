"""Primary-outcome comparison tables: strict five-gate versus unified success.

The primary outcome (``task_success`` on every loaded row) is the
final-artifact rule: the run's ``result.json``, after a mechanical model-blind
digit-preserving repair, matches the reference under the canonical per-task
tolerances, and the canonical integrity gate passes. Execution status,
artifact shape, and the method-evidence pattern are not part of the primary
outcome. The scorer judges the final saved file; model-side timeouts stay
failures. The rule is computed once in
``analysis_dataset.attach_unified_outcome`` (version
``analysis_dataset.PRIMARY_OUTCOME_VERSION``); this module never recomputes
it.

This module reports the comparison against the strict five-gate baseline that
was the primary outcome before 2026-08-12:

``task_success``
    The unified primary outcome, as loaded.

``task_success_strict``
    The five-gate diagnostic verdict: execution, artifact, numeric, method,
    and integrity all pass (scorer ``strict_posthoc.SCORING_VERSION``).

``task_success_with_method``
    Sensitivity variant: the unified outcome plus the stored method-evidence
    gate.

What still binds. The numeric comparison behind the unified outcome is the
canonical one (``strict_posthoc.score_artifact`` and
``strict_posthoc.compare_numeric``), so the per-task tolerances, skip lists,
magnitude-only comparisons, exact-integer terms, and accepted alternative
conventions all apply unchanged. The integrity gate also binds, so a run that
read a reference file or hard-coded a reference number is never rescued.

The method gate in the sensitivity variant is the stored canonical gate,
computed on the unrepaired artifact. For a record whose artifact only became
loadable after repair, that gate saw the script and the logs but not
``method_metadata``, so the variant is conservative for repaired records.

Read-only over the collection directories. No model calls and no network. Run
from the repo root:

    uv run python code/benchmark/estimation_outcome.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analysis_dataset as ad  # noqa: E402
import tasks as tasks_mod  # noqa: E402
from analysis_dataset import (  # noqa: E402
    NUMERIC_FAIL_KINDS,
    NUMERIC_FAIL_MATCHES,
    ReferenceLoader,
    classify_numeric_notes,
)
from tasks import Task  # noqa: E402

__all__ = [
    "DEFINITION",
    "NUMERIC_FAIL_KINDS",
    "NUMERIC_FAIL_MATCHES",
    "OUTCOMES",
    "RECORD_COLUMNS",
    "ReferenceLoader",
    "classify_numeric_notes",
    "main",
    "rate_block",
    "score_row",
    "score_rows",
    "summarize",
    "write_records_csv",
]

DEFAULT_OUT_SUBDIR = Path("data") / "analysis"
TABLES_NAME = "estimation_outcome.json"
RECORDS_NAME = "estimation_outcome_records.csv"
DECISION_RECORD = (
    "quality_reports/decision_records/2026-08-13_final-artifact-outcome.md"
)

DEFINITION = {
    "task_success": (
        "primary (final artifact): the mechanically repaired result.json "
        "matches the reference under the canonical per-task tolerances and "
        "the integrity gate passes; execution, artifact shape, and method "
        "evidence are not judged; the scorer judges the final saved file and "
        "model-side timeouts stay failures"
    ),
    "task_success_strict": (
        "diagnostic baseline: all five canonical gates pass (execution, "
        "artifact, numeric, method, integrity); the primary outcome before "
        "2026-08-12"
    ),
    "task_success_with_method": (
        "sensitivity: task_success plus the stored method-evidence gate"
    ),
    "repair": (
        "model-blind and digit-preserving; truncated or empty documents are "
        "never closed and stay unrepairable (code/benchmark/artifact_repair.py)"
    ),
    "primary_outcome_version": ad.PRIMARY_OUTCOME_VERSION,
    "decision_record": DECISION_RECORD,
}

RECORD_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "arm",
    "model",
    "task",
    "software",
    "prompt",
    "interface",
    "interface_label",
    "rep",
    "task_success",
    "task_success_strict",
    "task_success_with_method",
    "repair_rule",
    "integrity_pass",
    "method_pass",
    "numeric_repaired_pass",
    "numeric_fail_kind",
    "gate_execution",
    "gate_artifact",
    "gate_numeric",
    "gate_method",
    "revisions_used",
    "run_dir",
)

#: The two outcomes compared against the strict five-gate baseline.
OUTCOMES: tuple[str, ...] = ("task_success", "task_success_with_method")


def score_row(
    row: Mapping[str, Any],
    *,
    tasks_by_id: Mapping[str, Task],
    references: ReferenceLoader,
) -> dict[str, Any]:
    """Format one loaded run row as a comparison record.

    A row that ``analysis_dataset.attach_unified_outcome`` has not annotated
    yet is annotated here first, through the same single scoring path.
    """
    if "task_success_strict" not in row:
        ad.attach_unified_outcome(
            [row], tasks_by_id=tasks_by_id, references=references
        )
    run_dir = str(row["provenance"].get("run_dir") or "")
    gates = row["gates"]
    unified = row["unified"]
    primary = bool(row["task_success"])
    method_ok = bool(gates.get("method"))
    return {
        "cell_id": Path(run_dir).name if run_dir else "",
        "arm": str(row["provenance"]["arm_id"]),
        "model": str(row["model"]),
        "task": str(row["task"]),
        "software": str(row["platform"]),
        "prompt": str(row["prompt"]),
        "interface": str(row["interface"]),
        "interface_label": str(row["interface_label"]),
        "rep": int(row["rep"]),
        "task_success": int(primary),
        "task_success_strict": int(bool(row["task_success_strict"])),
        "task_success_with_method": int(primary and method_ok),
        "repair_rule": unified["repair_rule"],
        "integrity_pass": int(bool(gates.get("integrity"))),
        "method_pass": int(method_ok),
        "numeric_repaired_pass": int(bool(unified["numeric_repaired_pass"])),
        "numeric_fail_kind": unified["numeric_fail_kind"],
        "gate_execution": int(bool(gates.get("execution"))),
        "gate_artifact": int(bool(gates.get("artifact"))),
        "gate_numeric": int(bool(gates.get("numeric"))),
        "gate_method": int(method_ok),
        "revisions_used": (
            int(row["revisions_used"])
            if row.get("revisions_used") is not None
            else ""
        ),
        "run_dir": run_dir,
    }


def score_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    references: Mapping[str, Any] | None = None,
    tasks_by_id: Mapping[str, Task] | None = None,
    reference_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Format every row. ``references``/``tasks_by_id`` are test injection points."""
    loader = ReferenceLoader(reference_dir=reference_dir, references=references)
    if tasks_by_id is None:
        tasks_by_id = {task.task_id: task for task in tasks_mod.TASKS}
    return [score_row(row, tasks_by_id=tasks_by_id, references=loader) for row in rows]


# --- tables -----------------------------------------------------------------


def _pct(successes: int, n: int) -> float:
    return round(100.0 * successes / n, 2) if n else 0.0


def rate_block(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts and rates for one group, on the primary, strict, and sensitivity outcomes."""
    n = len(records)
    keys = ("task_success", "task_success_strict", "task_success_with_method")
    successes = {key: sum(int(r[key]) for r in records) for key in keys}
    block: dict[str, Any] = {"n": n}
    for key, count in successes.items():
        block[f"{key}_n"] = count
        block[f"{key}_pct"] = _pct(count, n)
    block["delta_pp"] = round(
        block["task_success_pct"] - block["task_success_strict_pct"], 2
    )
    block["delta_with_method_pp"] = round(
        block["task_success_with_method_pct"] - block["task_success_strict_pct"], 2
    )
    return block


def _subset(
    records: Sequence[Mapping[str, Any]], **conditions: str
) -> list[Mapping[str, Any]]:
    return [
        r for r in records if all(r[key] == value for key, value in conditions.items())
    ]


def _stats_for(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "overall": rate_block(records),
        "by_interface": {
            ad.INTERFACE_LABELS[interface]: rate_block(
                _subset(records, interface=interface)
            )
            for interface in ad.INTERFACES
        },
        "by_software": {
            software: rate_block(_subset(records, software=software))
            for software in ad.PLATFORMS
        },
    }


def _nine_cell_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "software": software,
            "interface": interface,
            "interface_label": ad.INTERFACE_LABELS[interface],
            **rate_block(_subset(records, software=software, interface=interface)),
        }
        for software in ad.PLATFORMS
        for interface in ad.INTERFACES
    ]


def _flip_block(records: Sequence[Mapping[str, Any]], outcome: str) -> dict[str, Any]:
    """Row flips of ``outcome`` against the strict five-gate baseline."""
    up = [r for r in records if r[outcome] == 1 and r["task_success_strict"] == 0]
    down = [r for r in records if r[outcome] == 0 and r["task_success_strict"] == 1]
    return {
        "n_up": len(up),
        "n_down": len(down),
        "up_by_model": _counts(r["model"] for r in up),
        "up_by_model_software": [
            {"model": model, "software": software, "n": count}
            for (model, software), count in sorted(
                Counter((r["model"], r["software"]) for r in up).items()
            )
        ],
        "up_by_task": _counts(r["task"] for r in up),
        "down_detail": [
            {
                "model": r["model"],
                "software": r["software"],
                "interface": r["interface"],
                "task": r["task"],
                "repair_rule": r["repair_rule"],
                "cell_id": r["cell_id"],
            }
            for r in down
        ],
    }


def _counts(values: Iterable[Any]) -> dict[str, int]:
    """Counts keyed by value, ordered by descending count then key."""
    counter = Counter(str(value) for value in values)
    return {
        key: count
        for key, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    }


def _flip_mechanism(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    up = [
        r for r in records if r["task_success"] == 1 and r["task_success_strict"] == 0
    ]
    return {
        "n_up": len(up),
        "needed_repair": sum(1 for r in up if r["repair_rule"] != "none needed"),
        "already_valid_json": sum(1 for r in up if r["repair_rule"] == "none needed"),
        "failed_gate_counts": {
            gate: sum(1 for r in up if r[gate] == 0)
            for gate in (
                "gate_execution",
                "gate_artifact",
                "gate_numeric",
                "gate_method",
            )
        },
        "method_gate_only": sum(
            1
            for r in up
            if r["gate_method"] == 0
            and r["gate_execution"] == 1
            and r["gate_artifact"] == 1
            and r["gate_numeric"] == 1
        ),
        "integrity_blocked": sum(
            1
            for r in records
            if r["integrity_pass"] == 0 and r["numeric_repaired_pass"] == 1
        ),
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Assemble the strict-versus-unified comparison tables. Pure and deterministic."""
    models = [m for m in ad.MODELS if any(r["model"] == m for r in records)]
    return {
        "generated_by": "code/benchmark/estimation_outcome.py",
        "definition": dict(DEFINITION),
        "overall": rate_block(records),
        "by_model": {
            model: _stats_for(_subset(records, model=model)) for model in models
        },
        "by_arm": {
            model: {
                prompt: _stats_for(_subset(records, model=model, prompt=prompt))
                for prompt in ad.PROMPTS
            }
            for model in models
        },
        "nine_cell": {
            model: {
                "pooled": _nine_cell_rows(_subset(records, model=model)),
                **{
                    prompt: _nine_cell_rows(
                        _subset(records, model=model, prompt=prompt)
                    )
                    for prompt in ad.PROMPTS
                },
            }
            for model in models
        },
        "flips_vs_strict": {
            outcome: _flip_block(records, outcome) for outcome in OUTCOMES
        },
        "flip_mechanism": _flip_mechanism(records),
        "repair_rule_counts": _counts(r["repair_rule"] for r in records),
        "numeric_fail_kind_counts": _counts(
            r["numeric_fail_kind"]
            for r in records
            if r["task_success"] == 0 and r["numeric_fail_kind"]
        ),
    }


def write_records_csv(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write one row per scored run, in the loaded row order."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RECORD_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in RECORD_COLUMNS})


# --- CLI --------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score the analysis sample on the primary unified outcome and "
            "write the strict-versus-unified comparison tables."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            f"output directory; default <project-root>/{DEFAULT_OUT_SUBDIR.as_posix()}"
        ),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=list(ad.ARMS),
        default=None,
        help="arm ids to load (default: all six)",
    )
    parser.add_argument(
        "--max-reps",
        type=int,
        default=12,
        help="eligible repetitions retained per design cell (default 12)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="repo root override for tests (default: the live repo root)",
    )
    args = parser.parse_args(argv)

    project_root = (args.project_root or ad.PROJECT_ROOT).resolve()
    out_dir = (
        args.out_dir if args.out_dir is not None else project_root / DEFAULT_OUT_SUBDIR
    )

    result = ad.load_runs(arm_ids=args.arms, max_reps=None, project_root=project_root)
    rows = ad.select_first_eligible_rows(result.rows, args.max_reps)
    records = score_rows(rows, reference_dir=project_root / "data" / "tasks" / "references")
    tables = summarize(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    tables_path = out_dir / TABLES_NAME
    records_path = out_dir / RECORDS_NAME
    tables_path.write_text(json.dumps(tables, indent=2) + "\n")
    write_records_csv(records, records_path)

    overall = tables["overall"]
    print(
        f"records: {overall['n']}  task_success {overall['task_success_pct']}%  "
        f"strict {overall['task_success_strict_pct']}%  "
        f"with_method {overall['task_success_with_method_pct']}%"
    )
    for outcome in OUTCOMES:
        flips = tables["flips_vs_strict"][outcome]
        print(f"{outcome} vs strict: flips up {flips['n_up']}, down {flips['n_down']}")
    print(f"wrote {tables_path}")
    print(f"wrote {records_path}")


if __name__ == "__main__":
    main()

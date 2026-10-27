"""Canonical scoring for the offline pilot arms, reusing ``strict_posthoc``.

``harness.score()`` is a simplified LIVE gate with known false-negatives that the
Sonnet corpus never relies on for official numbers:

- R's ``jsonlite`` serializes an empty ``diagnostics`` as ``[]`` not ``{}`` →
  ``harness.score()`` early-returns "diagnostics is not an object" and reports
  ``numeric_success=False`` even when the estimates are exactly correct.
- Stata result.json can need light JSON repair; Stata CLI exit codes are
  unreliable (a script can throw ``r(3000)`` after correctly writing result.json).

The corpus is scored canonically by ``strict_posthoc``. This module reproduces
that scoring for the pilot arms by reusing the SAME primitives in the SAME order
as ``strict_posthoc``'s per-record scorer (strict_posthoc.py:846-866), WITHOUT
its Sonnet-specific wave/quarantine/claude-block machinery, so it applies cleanly
to both ``codex_pilot`` and ``haiku_pilot`` records and is neither more nor less
lenient than the corpus:

- numeric is compared INDEPENDENTLY of artifact validity;
- method evidence is the corpus ``combined_evidence_text`` (scoring notes +
  method_metadata + diagnostics + generated script + logs), not just the script;
- ``execution_pass`` mirrors the corpus rule (a valid artifact rescues a nonzero
  CLI exit), and is kept separate from the raw ``execution_success``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import strict_posthoc as sp  # noqa: E402
import tasks as tasks_mod  # noqa: E402

TASKS_BY_ID = {t.task_id: t for t in tasks_mod.ALL_TASKS}

_REFS: dict[str, Any] | None = None


def references() -> dict[str, Any]:
    global _REFS
    if _REFS is None:
        _REFS = sp.load_references()
    return _REFS


def canonical_score(
    cell_dir: Path,
    task: Any,
    platform: str,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one pilot cell canonically (artifact + numeric + method + execution).

    Mirrors ``strict_posthoc``'s per-record scorer so ``numeric_success`` is the
    trustworthy correctness metric and matches corpus methodology exactly.
    """

    record = record or {}
    result_path = cell_dir / "result.json"
    try:
        data, load_notes = sp.load_result(result_path)
    except Exception as exc:  # noqa: BLE001 - never crash scoring on a bad file
        data, load_notes = None, [f"result.json read error: {exc}"]

    script_path = sp.find_script(cell_dir, platform)
    artifact_ok, estimates, art_notes = sp.score_artifact(data, task, platform)
    # Corpus scores numeric INDEPENDENT of artifact validity (strict_posthoc.py:852).
    numeric_ok, per_term, num_notes = sp.compare_numeric(estimates, task, references())

    evidence = sp.combined_evidence_text(cell_dir, record, data, script_path)
    try:
        method_ok, method_notes = sp.score_method(task.task_id, evidence, script_path)
    except Exception as exc:  # noqa: BLE001 - method scoring is informational here
        method_ok, method_notes = None, [f"method scoring error: {exc}"]

    # A parseable result.json IS a model response even if the CLI exited nonzero;
    # a valid artifact rescues execution (strict_posthoc.py:850, 862-866).
    model_response = (data is not None) or bool(record.get("execution_success"))
    execution_pass = model_response and (
        bool(record.get("execution_success")) or artifact_ok
    )

    return {
        "scorer": "canonical(strict_posthoc)",
        "artifact_success": bool(artifact_ok),
        "numeric_success": bool(numeric_ok),
        "method_success": method_ok,
        "execution_pass": execution_pass,
        "model_response": model_response,
        "per_term": per_term,
        "notes": load_notes + art_notes + num_notes,
        "method_notes": method_notes,
    }


def rescore_record(cell_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Re-score an existing pilot record in place and return it.

    Preserves the raw ``execution_success`` (a separate CLI-exit signal) and
    replaces the ``scoring`` block with the canonical one, keeping any previous
    live scoring under ``scoring_live`` for audit.
    """

    task = TASKS_BY_ID.get(record.get("task_id"))
    platform = record.get("platform")
    if task is None or platform is None:
        return record
    scoring = canonical_score(cell_dir, task, platform, record)
    scoring["execution_success"] = record.get("execution_success")
    prev = record.get("scoring")
    if isinstance(prev, dict) and prev.get("scorer") != scoring["scorer"]:
        record["scoring_live"] = prev
    record["scoring"] = scoring
    return record

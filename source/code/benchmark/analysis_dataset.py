"""Normalized six-arm run loader for the twelve-rep analysis sample.

Single source of truth for which run records exist, which are usable, and what
each run scored. Everything downstream (``extended_analyses``,
``build_full_analysis``) consumes normalized ``RunRow`` dicts produced here.

Storage layout: every arm directory under ``data/runs`` holds one directory per
run, plus underscore-prefixed directories that hold everything else. ``_working``
holds sweep logs, aggregates, and derived reports. ``_retired_prompt_mode`` holds
records of a retired prompt specification. Decision record
2026-08-12_run-layout-normalization.md.

Storage format: every arm reads per-run records from its run directories. The
Sonnet zero-shot corpus stores its scores as ``score.json`` next to each
``run_record.json`` (written by ``strict_posthoc --write-run-scores``). Those
per-run records are the source of truth. The aggregate
``task_success_report.json`` in ``_working`` is derived output, a human-readable
summary of the same score rows, and is retained as a selectable legacy read path
(``corpus_source="report"``). Decision record
2026-08-10_corpus-records-unification.md.

The six arms (``ARMS``) cover model x prompt: Sonnet, Codex (gpt-5.4), and
DeepSeek, each in zero-shot and few-shot. Every loader here reads all six; the
run corpus keeps all three model arms and the public site and the diagnostics
read them. The paper's main specification covers two of the three model arms
(``PAPER_MODELS``, decision record
2026-08-21_two-model-main-specification); the estimation and numbers chain
enters through ``load_paper_runs`` and pins its rows with
``paper_sample_manifest.json``. Sources differ by ``kind``:

- ``corpus_records``: the scored Sonnet sweep, read per run directory from
  ``run_record.json`` plus the stored ``score.json``. A run directory holding a
  record without a stored score raises (materialization tripwire).
- ``corpus_report``: the same scores read from the derived aggregate report.
  Retained for audit; selectable through ``load_runs(corpus_source="report")``.
- ``sonnet_wave``: the 2026-07-27 few-shot wave; inline scoring blocks are
  never consumed. Every usable record is re-scored in memory with the
  canonical ``strict_posthoc`` primitives, including the integrity gate
  (the wave's inline integrity value is a hardcoded placeholder).
- ``codex_pilot`` / ``openrouter_pilot``: every usable record is re-scored in
  memory with the canonical ``strict_posthoc`` primitives. Inline blocks are
  retained as source telemetry only.

Missing-data rule (intention-to-treat, review fix F2, 2026-07-30): a design
slot the model attempted without producing a scored script within its
condition's operating cap enters as a failure row (``provenance.itt_failure``,
counter ``itt_timeout_failure``). Provider-side interruptions (session and
usage limits, API errors, CLI crashes) stay excluded with their own counters.
The balanced sample keeps the first twelve eligible repetitions in each
arm/task/platform/interface cell. A later repetition therefore fills a key
whose earlier record was interrupted by the provider. Downstream code derives
the responded-only sensitivity sample by dropping flagged rows.

Self-repair accounting (decision record 2026-08-13_final-artifact-outcome):
each harness bounds the number of revisions in the self-repair condition its
own way, and the scorer judges the final saved file. Every row carries
``revisions_used`` (provider attempts minus one, floor zero, None when the
attempt count is unknown) so the paper can disclose how many self-repair runs
revised more than once. Cost, tokens, and both wall measures cover the whole
run.

Runtime fields: every row carries two wall-clock measures.
``provider_wall_seconds`` is the harmonized one. It sums ``wall_seconds`` over
the arm's provider blocks (``claude`` for Sonnet, ``codex`` for Codex,
``openrouter`` for DeepSeek), so it measures provider-call time only and is
comparable across arms. ``wall_seconds`` keeps its historical per-arm
definition: Sonnet arms report the same provider-call sum, while Codex and
DeepSeek rows prefer the record's ``total_wall_seconds``, which also covers
local script execution and harness overhead.

Primary outcome rule (``PRIMARY_OUTCOME_VERSION``, decision record
2026-08-13_final-artifact-outcome): ``task_success`` on every row returned by
``load_runs`` is the final-artifact outcome. It passes when the run's stored
``result.json``, after the model-blind digit-preserving repair in
``artifact_repair``, matches the reference under the canonical
``strict_posthoc`` tolerances AND the canonical integrity gate passes. The
execution, artifact-shape, and method-evidence gates are not part of the
primary outcome. Model-side timeouts stay failures.
The five-gate strict verdict is preserved on each row as
``task_success_strict`` for gate-anatomy diagnostics, and the repair detail
sits under ``unified``. ``attach_unified_outcome`` applies the annotation once
inside ``load_runs``; direct arm-loader calls return unannotated rows whose
``task_success`` is still the strict five-gate verdict.

Exclusion rules baked in as data, never rediscovered ad hoc:

- Underscore-prefixed subdirectories (quarantines, superseded re-collections)
  are skipped in every pilot root.
- The strays directory in ``EXCLUDED_STRAY_DIRS`` is excluded BY PATH; its
  prompt hashes match the respecified few-shot set, so a hash allowlist would
  wrongly admit it. No loader ever iterates it.
- Retired few-shot records now sit in ``_retired_prompt_mode`` inside their
  zero-shot arm, so discovery never reaches them. The ``prompt_mode`` rule
  stays as the second check: a retired record found at an arm root is excluded
  by its stored ``prompt_mode`` (counter ``excluded_retired_fewshot``), never
  by prompt hash.
- ``codex_zero_shot`` raises on a run directory without ``run_record.json``.
  That root is complete; a missing record means unexpected contamination
  (deliberate tripwire). The still-collecting roots skip such directories and
  count them as ``in_flight_missing_record``.

Token normalization: every row carries the same five keys. Anthropic
``input_tokens`` EXCLUDES cache reads; OpenAI-shaped ``input_tokens``
(codex, openrouter) INCLUDES ``cached_input_tokens``, and ``output_tokens``
INCLUDES ``reasoning_output_tokens``. There is deliberately no unified
"total input" field; ``provenance`` records the provider so cost code can
pick the right semantics.

Turns: sonnet rows sum ``num_turns`` over claude blocks (corpus rows read
their ``run_record.json``; ``include_turns=False`` skips those reads);
deepseek rows use ``num_turns`` from the openrouter block(s); codex CLI
records carry no turn field, so codex rows always have ``turns = None`` and
``turns_unavailable`` stays 0 for them (the counter tracks absences only
where the provider normally supplies turns).

Read-only over all data directories; no model calls, no network.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NotRequired, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import strict_posthoc as sp  # noqa: E402
import tasks as tasks_mod  # noqa: E402
from artifact_repair import repair_result_text  # noqa: E402
from tasks import Task  # noqa: E402

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent

MODELS = ("sonnet", "codex", "deepseek")
PROMPTS = ("zero_shot", "few_shot")
INTERFACES = ("no_feedback", "self_repair", "tool_loop")
PLATFORMS = ("python", "r", "stata")
INTERFACE_LABELS = {
    "no_feedback": "chatbot",
    "self_repair": "one-repair",
    "tool_loop": "constrained agent",
}
SONNET_CONDITIONS = {interface: interface for interface in INTERFACES}
CODEX_CONDITIONS = {
    "codex_generate": "no_feedback",
    "codex_repair": "self_repair",
    "codex_agent": "tool_loop",
}
OPENROUTER_CONDITIONS = {
    "or_generate": "no_feedback",
    "or_repair": "self_repair",
    "or_agent": "tool_loop",
}
#: The promoted DeepSeek arms (2026-08-18): the constrained-agent condition
#: runs through mini-swe-agent and is stored as ``or_agent_swe``; the other
#: two conditions are unchanged. Decision record
#: 2026-08-18_deepseek-swe-arm-replacement.
OPENROUTER_SWE_CONDITIONS = {
    "or_generate": "no_feedback",
    "or_repair": "self_repair",
    "or_agent_swe": "tool_loop",
}

ANALYSIS_TASK_IDS: tuple[str, ...] = tuple(
    sorted(
        task.task_id
        for task in tasks_mod.TASKS
        if task.task_id not in sp.QUARANTINED_TASKS
    )
)
if len(ANALYSIS_TASK_IDS) != 21:
    raise AssertionError(
        f"expected 21 analysis tasks, got {len(ANALYSIS_TASK_IDS)}: {ANALYSIS_TASK_IDS}"
    )

GATE_NAMES = ("execution", "artifact", "numeric", "method", "integrity")

#: Version stamp of the primary outcome rule. The strict five-gate scorer
#: keeps its own stamp (``strict_posthoc.SCORING_VERSION``); both are written
#: into the top level of the analysis outputs.
PRIMARY_OUTCOME_VERSION = "final_artifact_2026-08-13"

#: Version stamp of the analysis sample rule (decision record
#: 2026-08-18_twelve-repetition-sample): the first twelve eligible
#: repetitions per arm/task/platform/interface cell.
SAMPLE_VERSION = "twelve_rep_2026-08-18"

#: Default repetition cap of the balanced analysis sample.
DEFAULT_MAX_REPS = 12

#: The pinned sample manifest written by ``build_full_analysis`` and checked
#: by the builders through ``load_runs(verify_manifest=True)``.
SAMPLE_MANIFEST_SUBPATH = Path("data") / "analysis" / "sample_manifest.json"

#: Model arms of the paper's main specification (decision record
#: 2026-08-21_two-model-main-specification). "Pooled" in the estimation and
#: numbers chain means these two arms pooled. The run corpus keeps all three
#: arms on disk and every loader in this module still reads all of them; the
#: public site and the diagnostics depend on that. Only the estimation and
#: numbers chain restricts itself to ``PAPER_MODELS``.
PAPER_MODELS = ("sonnet", "codex")

#: Version stamp of the paper analysis sample: the first twelve eligible
#: repetitions per cell over the two paper arms, 9,072 runs.
PAPER_SAMPLE_VERSION = "two_model_2026-08-21"

#: The pinned paper-sample manifest. It sits next to the corpus manifest and
#: never replaces it: ``sample_manifest.json`` pins the 13,608-run corpus,
#: ``paper_sample_manifest.json`` pins the 9,072 rows the paper estimates on.
PAPER_SAMPLE_MANIFEST_SUBPATH = (
    Path("data") / "analysis" / "paper_sample_manifest.json"
)

#: Priority-ordered classification of a failed numeric comparison. The first
#: matching note wins, so a record that fails on both an estimate and a sample
#: size is labeled ``estimate_off``.
NUMERIC_FAIL_MATCHES: tuple[tuple[str, str], ...] = (
    ("missing result or reference row", "missing_term"),
    ("estimate outside tolerance", "estimate_off"),
    ("standard error outside tolerance", "se_off"),
    ("sample size outside exact match", "n_obs_only"),
)
NUMERIC_FAIL_KINDS: tuple[str, ...] = (
    "no_artifact",
    "missing_term",
    "estimate_off",
    "se_off",
    "n_obs_only",
    "unknown_task",
    "other",
)
CRITERION_NAMES = (
    "specification_correctness",
    "executability",
    "output_correctness",
    "submission_validity",
)
TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)
OPENAI_USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)
EXPECTED_CELLS_PER_ARM = 189
EXPECTED_RUNS_PER_ARM_12REP = 2268

EXCLUDED_DIR_NAMES_PREFIX = "_"
RUN_RECORD_NAME = "run_record.json"
CORPUS_REPORT_NAME = "task_success_report.json"
# Every arm root holds one directory per run. Working files and derived reports
# live in the arm's ``_working`` directory.
WORKING_DIRNAME = "_working"
CORPUS_KINDS = ("corpus_records", "corpus_report")
EXCLUDED_STRAY_DIRS = (
    PROJECT_ROOT / "data" / "runs" / "_excluded_deepseek_fewshot_strays",
)
SESSION_LIMIT_MARKER = '"api_error_status":429'
CORPUS_FALLBACK_WAVE = "pinned-2026-07"

ATTRITION_KEYS = (
    "dirs_seen",
    "records_parsed",
    "missing_run_record",
    "invalid_json",
    "excluded_underscore_dir",
    "excluded_non_dir",
    "excluded_retired_fewshot",
    "excluded_wrong_arm",
    "excluded_rep_over_cap",
    "excluded_session_limit",
    "excluded_no_response_other",
    "excluded_collection_failure",
    "in_flight_missing_record",
    "itt_timeout_failure",
    "usable",
    "turns_unavailable",
)

REPAIR_INTERFACE = "self_repair"


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    model: str
    model_id: str
    prompt_mode: str
    kind: str
    path: Path
    condition_map: Mapping[str, str]
    allow_absent: bool
    provider: str
    notes: str


ARMS: dict[str, ArmSpec] = {
    "sonnet_zero_shot": ArmSpec(
        arm_id="sonnet_zero_shot",
        model="sonnet",
        model_id="claude-sonnet-4-6",
        prompt_mode="zero_shot",
        kind="corpus_records",
        path=PROJECT_ROOT / "data" / "runs" / "sonnet_zero_shot",
        condition_map=SONNET_CONDITIONS,
        allow_absent=False,
        provider="anthropic",
        notes=(
            "paper zero-shot sample: model_response, not quarantined, "
            "prompt_wave == legacy, prompt_mode == zero_shot; scores read "
            "per run directory from the stored score.json"
        ),
    ),
    "sonnet_few_shot": ArmSpec(
        arm_id="sonnet_few_shot",
        model="sonnet",
        model_id="claude-sonnet-4-6",
        prompt_mode="few_shot",
        kind="sonnet_wave",
        path=PROJECT_ROOT / "data" / "runs" / "sonnet_few_shot",
        condition_map=SONNET_CONDITIONS,
        allow_absent=False,
        provider="anthropic",
        notes="respecified family-matched demos; canonical rescore per record",
    ),
    "codex_zero_shot": ArmSpec(
        arm_id="codex_zero_shot",
        model="codex",
        model_id="gpt-5.4",
        prompt_mode="zero_shot",
        kind="codex_pilot",
        path=PROJECT_ROOT / "data" / "runs" / "codex_zero_shot",
        condition_map=CODEX_CONDITIONS,
        allow_absent=False,
        provider="openai_codex",
        notes=(
            "retired-spec few_shot records sit in _retired_prompt_mode; "
            "missing run_record.json raises (tripwire)"
        ),
    ),
    "codex_few_shot": ArmSpec(
        arm_id="codex_few_shot",
        model="codex",
        model_id="gpt-5.4",
        prompt_mode="few_shot",
        kind="codex_pilot",
        path=PROJECT_ROOT / "data" / "runs" / "codex_few_shot",
        condition_map=CODEX_CONDITIONS,
        allow_absent=False,
        provider="openai_codex",
        notes="respecified few-shot wave; may be partial while collecting",
    ),
    "deepseek_zero_shot": ArmSpec(
        arm_id="deepseek_zero_shot",
        model="deepseek",
        model_id="deepseek/deepseek-v4-flash",
        prompt_mode="zero_shot",
        kind="openrouter_pilot",
        path=PROJECT_ROOT / "data" / "runs" / "deepseek_zero_shot",
        condition_map=OPENROUTER_SWE_CONDITIONS,
        allow_absent=False,
        provider="openrouter",
        notes=(
            "promoted 2026-08-14 mini-swe wave (or_agent_swe agent "
            "condition); the July arm sits in "
            "_retired_deepseek_or_2026-08-18/zero_shot"
        ),
    ),
    "deepseek_few_shot": ArmSpec(
        arm_id="deepseek_few_shot",
        model="deepseek",
        model_id="deepseek/deepseek-v4-flash",
        prompt_mode="few_shot",
        kind="openrouter_pilot",
        path=PROJECT_ROOT / "data" / "runs" / "deepseek_few_shot",
        condition_map=OPENROUTER_SWE_CONDITIONS,
        allow_absent=False,
        provider="openrouter",
        notes=(
            "promoted 2026-08-14 mini-swe wave (or_agent_swe agent "
            "condition); the July arm sits in "
            "_retired_deepseek_or_2026-08-18/few_shot"
        ),
    ),
}


#: Arm ids of the paper scope, in registry order (four arms: two models x two
#: prompt conditions). Derived from ``ARMS`` so the registry stays the one
#: place an arm is defined.
PAPER_ARM_IDS: tuple[str, ...] = tuple(
    arm_id for arm_id, spec in ARMS.items() if spec.model in PAPER_MODELS
)

#: Expected size of the paper analysis sample at twelve repetitions per cell.
EXPECTED_PAPER_RUNS_12REP = len(PAPER_ARM_IDS) * EXPECTED_RUNS_PER_ARM_12REP


class RunRow(TypedDict):
    model: str
    model_id: str
    prompt: str
    interface: str
    interface_label: str
    platform: str
    task: str
    rep: int
    task_success: bool
    task_success_strict: NotRequired[bool]
    unified: NotRequired[dict[str, Any]]
    gates: dict[str, bool]
    criteria: dict[str, bool]
    per_term: dict[str, Any]
    cost_usd: float | None
    tokens: dict[str, int]
    wall_seconds: float | None
    provider_wall_seconds: float | None
    turns: int | None
    revisions_used: int | None
    provenance: dict[str, Any]


def paper_rows(rows: Sequence[RunRow]) -> list[RunRow]:
    """The subset of loaded rows inside the paper scope (``PAPER_MODELS``)."""
    return [row for row in rows if row["model"] in PAPER_MODELS]


@dataclass
class ArmLoadReport:
    arm_id: str
    present: bool
    counts: dict[str, int]
    max_created_at: str | None


@dataclass
class LoadResult:
    rows: list[RunRow]
    reports: dict[str, ArmLoadReport]
    manifest: dict[str, Any]


def _new_counts() -> dict[str, int]:
    return dict.fromkeys(ATTRITION_KEYS, 0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rep_from_dir_name(name: str) -> int | None:
    match = re.search(r"rep(\d+)$", name)
    return int(match.group(1)) if match else None


def _track_created_at(record: Mapping[str, Any], current: str | None) -> str | None:
    created = record.get("created_at")
    if not isinstance(created, str) or not created:
        return current
    return created if current is None else max(current, created)


def _provider_blocks(record: Mapping[str, Any], prefix: str) -> list[Mapping[str, Any]]:
    """The primary provider block plus ``<prefix>_attempt_*`` blocks, in order."""
    blocks: list[Mapping[str, Any]] = []
    primary = record.get(prefix)
    if isinstance(primary, dict):
        blocks.append(primary)
    for key in sorted(k for k in record if k.startswith(f"{prefix}_attempt_")):
        value = record[key]
        if isinstance(value, dict):
            blocks.append(value)
    return blocks


def _revisions_used(blocks: Sequence[Mapping[str, Any]]) -> int:
    """Provider attempts minus one, floor zero (revision count for one run)."""
    return max(len(blocks) - 1, 0)


def _blocks_turns(blocks: Sequence[Mapping[str, Any]]) -> int | None:
    values = [
        block.get("num_turns")
        for block in blocks
        if isinstance(block.get("num_turns"), (int, float))
    ]
    return int(sum(values)) if values else None


def _anthropic_tokens(raw: Mapping[str, Any]) -> dict[str, int]:
    """Normalize Anthropic usage; ``input_tokens`` excludes cache reads."""
    return {
        "input_tokens": int(float(raw.get("input_tokens") or 0)),
        "cached_input_tokens": int(float(raw.get("cache_read_input_tokens") or 0)),
        "cache_creation_input_tokens": int(
            float(raw.get("cache_creation_input_tokens") or 0)
        ),
        "output_tokens": int(float(raw.get("output_tokens") or 0)),
        "reasoning_output_tokens": 0,
    }


def _openai_tokens(raw: Mapping[str, Any]) -> dict[str, int]:
    """Normalize OpenAI-shaped usage; ``input_tokens`` includes cached input."""
    return {
        "input_tokens": int(float(raw.get("input_tokens") or 0)),
        "cached_input_tokens": int(float(raw.get("cached_input_tokens") or 0)),
        "cache_creation_input_tokens": 0,
        "output_tokens": int(float(raw.get("output_tokens") or 0)),
        "reasoning_output_tokens": int(float(raw.get("reasoning_output_tokens") or 0)),
    }


def five_gate_success(gates: Mapping[str, bool | None]) -> bool:
    """All five gates truthy; a null method gate fails."""
    return all(bool(gates.get(gate)) for gate in GATE_NAMES)


def criteria_from_gates(gates: Mapping[str, Any]) -> dict[str, bool]:
    """Mirror the ``strict_posthoc.score_one_run`` gate-to-criteria mapping."""
    return {
        "specification_correctness": bool(gates.get("method")),
        "executability": bool(gates.get("execution")),
        "output_correctness": bool(gates.get("artifact"))
        and bool(gates.get("numeric")),
        "submission_validity": bool(gates.get("integrity")),
    }


class ReferenceLoader:
    """Loads the stored reference outputs once, on first use.

    Unified scoring only needs the references when a run left a loadable
    artifact, so deferring the read keeps callers that pass synthetic rows
    off the filesystem entirely.
    """

    def __init__(
        self,
        reference_dir: Path | None = None,
        references: Mapping[str, Any] | None = None,
    ) -> None:
        self._reference_dir = reference_dir
        self._references: dict[str, Any] | None = (
            dict(references) if references is not None else None
        )

    def get(self) -> dict[str, Any]:
        if self._references is None:
            directory = self._reference_dir
            if directory is None:
                directory = PROJECT_ROOT / "data" / "tasks" / "references"
            self._references = sp.load_references(directory)
        return self._references


def classify_numeric_notes(notes: Sequence[str]) -> str:
    """Label why a repaired artifact failed the canonical numeric comparison."""
    joined = " ".join(notes)
    for needle, kind in NUMERIC_FAIL_MATCHES:
        if needle in joined:
            return kind
    return "other"


def score_unified_row(
    row: Mapping[str, Any],
    *,
    tasks_by_id: Mapping[str, Task],
    references: ReferenceLoader,
) -> dict[str, Any]:
    """Primary-outcome fields for one loaded row.

    The final-artifact outcome passes when the repaired ``result.json``
    matches the reference under the canonical tolerances AND the canonical
    integrity gate passes. Repair and comparison reuse
    ``artifact_repair.repair_result_text`` and the ``strict_posthoc``
    primitives; no tolerance is reimplemented. The scorer judges the final
    saved file, whatever number of revisions produced it.
    """
    provenance = row.get("provenance") or {}
    gates = row["gates"]
    integrity_ok = bool(gates.get("integrity"))
    run_dir = str(provenance.get("run_dir") or "")
    result_path = Path(run_dir) / "result.json" if run_dir else None

    numeric_ok = False
    fail_kind = ""
    if result_path is None or not result_path.is_file():
        rule = "no_file"
        fail_kind = "no_artifact"
    else:
        data, rule = repair_result_text(result_path.read_text(errors="replace"))
        task = tasks_by_id.get(str(row["task"]))
        if data is None:
            fail_kind = "no_artifact"
        elif task is None:
            fail_kind = "unknown_task"
        else:
            _, estimates, _ = sp.score_artifact(data, task, row["platform"])
            numeric_ok, _, notes = sp.compare_numeric(
                estimates, task, dict(references.get())
            )
            if not numeric_ok:
                fail_kind = classify_numeric_notes(notes)
    return {
        "success": bool(numeric_ok and integrity_ok),
        "repair_rule": rule,
        "numeric_repaired_pass": bool(numeric_ok),
        "numeric_fail_kind": fail_kind,
    }


def attach_unified_outcome(
    rows: Sequence[RunRow],
    *,
    tasks_by_id: Mapping[str, Task] | None = None,
    references: ReferenceLoader | Mapping[str, Any] | None = None,
    reference_dir: Path | None = None,
) -> None:
    """Annotate rows in place with the primary unified outcome.

    Moves the loader-computed five-gate verdict to ``task_success_strict``,
    sets ``task_success`` to the unified outcome, and stores the repair detail
    under ``unified``. Idempotent per row: an already-annotated row (it
    carries ``task_success_strict``) is left unchanged.
    """
    if isinstance(references, ReferenceLoader):
        loader = references
    else:
        loader = ReferenceLoader(reference_dir=reference_dir, references=references)
    if tasks_by_id is None:
        tasks_by_id = {task.task_id: task for task in tasks_mod.TASKS}
    for row in rows:
        if "task_success_strict" in row:
            continue
        fields = score_unified_row(row, tasks_by_id=tasks_by_id, references=loader)
        row["task_success_strict"] = bool(row["task_success"])
        row["task_success"] = fields.pop("success")
        row["unified"] = fields


def wave_record_usable(run_dir: Path, record: Mapping[str, Any]) -> tuple[bool, str]:
    """Corpus model-response rule for the sonnet wave (decision 2026-06-30).

    Usable iff any claude block has ``ok is True`` OR the run left a parseable
    ``result.json`` (rescues tool_loop max_turns records whose CLI exited 1).
    Never exclude on ``error == "claude exited 1"`` alone; that error is
    ambiguous between a 429 session limit and max_turns.
    """
    blocks = sp.claude_blocks(dict(record))
    if any(block.get("ok") is True for block in blocks):
        return True, "ok"
    result_data, _ = sp.load_result(run_dir / "result.json")
    if result_data is not None:
        return True, "result_json_rescue"
    errors = [str(block.get("error") or "") for block in blocks]
    if any(error == "timeout" for error in errors):
        return False, "timeout"
    stdout = " ".join(str(block.get("stdout_tail") or "") for block in blocks)
    if SESSION_LIMIT_MARKER in stdout:
        return False, "session_limit"
    return False, "no_response_other"


def openrouter_api_ok(record: Mapping[str, Any]) -> bool:
    """True iff any openrouter block (primary or attempt) has ``ok is True``."""
    blocks = _provider_blocks(record, "openrouter")
    return any(block.get("ok") is True for block in blocks)


def codex_collection_ok(record: Mapping[str, Any]) -> tuple[bool, str]:
    """Collection-failure guard: exclude only when the CLI produced no response.

    Never filter on ``scoring.model_response``; that field means "result.json
    parseable OR execution_success" and dropping on it would delete legitimate
    failures from the denominator.
    """
    blocks = _provider_blocks(record, "codex")
    if not blocks:
        return False, "no_codex_block"
    if any(block.get("ok") is True for block in blocks):
        return True, "ok"
    if any(block.get("timed_out") for block in blocks):
        return False, "timed_out"
    if any(block.get("provider_limit") for block in blocks):
        return False, "provider_limit"
    return False, "cli_not_ok"


def sum_blocks_cost(record: Mapping[str, Any], prefix: str) -> float | None:
    """Sum ``cost_usd`` over provider blocks; None when no block carries one."""
    values = [
        block.get("cost_usd")
        for block in _provider_blocks(record, prefix)
        if block.get("cost_usd") is not None
    ]
    return float(sum(float(value) for value in values)) if values else None


def _blocks_wall_seconds(
    blocks: Sequence[Mapping[str, Any]], limit: int | None = None
) -> float | None:
    """Sum ``wall_seconds`` over provider blocks; None when no block carries one.

    ``limit`` restricts the sum to the first N blocks.
    """
    selected = list(blocks)[:limit] if limit is not None else blocks
    values = [
        block.get("wall_seconds")
        for block in selected
        if block.get("wall_seconds") is not None
    ]
    return float(sum(float(value) for value in values)) if values else None


def sum_blocks_wall_seconds(
    record: Mapping[str, Any], prefix: str, limit: int | None = None
) -> float | None:
    """Provider-call wall time for one record: sum over its provider blocks.

    This is the harmonized runtime measure. It never falls back to the
    record's ``total_wall_seconds``, so it excludes local script execution and
    harness overhead in every arm. ``prefix`` selects the arm's provider
    blocks: ``claude`` for Sonnet, ``codex`` for Codex, ``openrouter`` for
    DeepSeek.
    """
    return _blocks_wall_seconds(_provider_blocks(record, prefix), limit)


def sum_blocks_usage(
    record: Mapping[str, Any],
    prefix: str,
    keys: tuple[str, ...],
    limit: int | None = None,
) -> dict[str, int]:
    """Element-wise sum of ``usage`` dicts over provider blocks.

    ``limit`` restricts the sum to the first N blocks.
    """
    totals = dict.fromkeys(keys, 0)
    blocks = _provider_blocks(record, prefix)
    if limit is not None:
        blocks = blocks[:limit]
    for block in blocks:
        usage = block.get("usage") or {}
        for key in keys:
            totals[key] += int(float(usage.get(key) or 0))
    return totals


def _itt_row(
    spec: ArmSpec,
    *,
    task_id: str,
    platform: str,
    interface: str,
    rep: int,
    reason: str,
    run_dir: str,
    cost_usd: float | None,
    tokens: dict[str, int],
    wall_seconds: float | None,
    provider_wall_seconds: float | None,
    revisions_used: int | None = None,
) -> RunRow:
    """A design slot attempted without a scored script (intention-to-treat).

    The model held the slot and produced no scored script within the
    condition's operating cap, so the slot scores as a failure on every
    criterion. Provider-side interruptions never reach this constructor;
    they stay excluded with their own counters.
    """
    gates = {gate: False for gate in GATE_NAMES}
    return {
        "model": spec.model,
        "model_id": spec.model_id,
        "prompt": spec.prompt_mode,
        "interface": interface,
        "interface_label": INTERFACE_LABELS[interface],
        "platform": platform,
        "task": task_id,
        "rep": rep,
        "task_success": False,
        "gates": gates,
        "criteria": criteria_from_gates(gates),
        "per_term": {},
        "cost_usd": cost_usd,
        "tokens": tokens,
        "wall_seconds": wall_seconds,
        "provider_wall_seconds": provider_wall_seconds,
        "turns": None,
        "revisions_used": revisions_used,
        "provenance": {
            "arm_id": spec.arm_id,
            "run_dir": run_dir,
            "source_kind": spec.kind,
            "prompt_sha256": None,
            "scoring_source": "itt",
            "usable_reason": reason,
            "itt_failure": True,
        },
    }


def _record_wall_seconds(record: Mapping[str, Any], prefix: str) -> float | None:
    """Total run wall time: ``total_wall_seconds`` when present, else blocks.

    This feeds the historical ``wall_seconds`` field for the Codex and
    DeepSeek arms and therefore includes local script execution. Use
    ``sum_blocks_wall_seconds`` for the cross-arm provider-call measure.
    """
    total = record.get("total_wall_seconds")
    if total is not None:
        return float(total)
    return _blocks_wall_seconds(_provider_blocks(record, prefix))


def _canonical_gates(
    run_dir: Path,
    task: Task,
    platform: str,
    record: Mapping[str, Any],
    references: Mapping[str, Any],
    data_dir: Path,
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Five canonical gates for one saved run, with injected references.

    Mirrors ``pilot_scoring.canonical_score`` (same ``strict_posthoc``
    primitives in the same order) plus the integrity gate, without the
    module-global reference cache so tests can inject synthetic references.
    """
    try:
        result_data, _ = sp.load_result(run_dir / "result.json")
    except Exception:  # noqa: BLE001 - a bad file scores as no artifact
        result_data = None
    script_path = sp.find_script(run_dir, platform)
    artifact_ok, estimates, _ = sp.score_artifact(result_data, task, platform)
    numeric_ok, per_term, _ = sp.compare_numeric(estimates, task, dict(references))
    evidence = sp.combined_evidence_text(
        run_dir, dict(record), result_data, script_path
    )
    try:
        method_ok, _ = sp.score_method(task.task_id, evidence, script_path)
    except Exception:  # noqa: BLE001 - method evidence errors score as fail
        method_ok = False
    model_response = (result_data is not None) or bool(record.get("execution_success"))
    execution_pass = model_response and (
        bool(record.get("execution_success")) or artifact_ok
    )
    integrity_ok, _ = sp.score_integrity(
        task, run_dir, script_path, dict(references), data_dir
    )
    gates = {
        "execution": bool(execution_pass),
        "artifact": bool(artifact_ok),
        "numeric": bool(numeric_ok),
        "method": bool(method_ok),
        "integrity": bool(integrity_ok),
    }
    return gates, per_term


def _integrity_gate(
    run_dir: Path,
    task: Task,
    platform: str,
    references: Mapping[str, Any],
    data_dir: Path,
) -> bool:
    script_path = sp.find_script(run_dir, platform)
    ok, _ = sp.score_integrity(task, run_dir, script_path, dict(references), data_dir)
    return bool(ok)


def _inline_gates(scoring: Mapping[str, Any], integrity: bool) -> dict[str, bool]:
    return {
        "execution": bool(scoring.get("execution_pass")),
        "artifact": bool(scoring.get("artifact_success")),
        "numeric": bool(scoring.get("numeric_success")),
        "method": bool(scoring.get("method_success")),
        "integrity": bool(integrity),
    }


def _corpus_record(run_dir: Path) -> dict[str, Any] | None:
    record_path = run_dir / "run_record.json"
    if not record_path.exists():
        return None
    try:
        return json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


SESSION_LIMIT_TEXTS = ("session limit", "usage limit")


def _corpus_no_response_reason(record: Mapping[str, Any] | None) -> str:
    """Classify a corpus no-response record from its saved run record.

    ``timeout`` means the model held the slot and hit the operating cap
    (an intention-to-treat failure). Session and usage limits are
    provider-side interruptions. Anything else, including an unreadable
    record, stays an unclassified exclusion.
    """
    if record is None:
        return "no_response_other"
    blocks = sp.claude_blocks(record)
    if any(str(block.get("error") or "") == "timeout" for block in blocks):
        return "timeout"
    haystack = " ".join(
        str(block.get("stdout_tail") or "") + " " + str(block.get("text") or "")
        for block in blocks
    )
    if SESSION_LIMIT_MARKER in haystack:
        return "session_limit"
    lowered = haystack.lower()
    if any(marker in lowered for marker in SESSION_LIMIT_TEXTS):
        return "session_limit"
    return "no_response_other"


def _lookup_task(tasks_by_id: Mapping[str, Task], record: Mapping[str, Any]) -> Task:
    task_id = str(record.get("task_id"))
    task = tasks_by_id.get(task_id)
    if task is None:
        raise KeyError(f"task_id not in tasks_by_id: {task_id!r}")
    return task


def corpus_source_spec(spec: ArmSpec, source: str) -> ArmSpec:
    """Return the same corpus arm read through the requested storage path.

    ``records`` reads per-run ``score.json`` files under the arm root;
    ``report`` reads the derived aggregate report in the arm's ``_working``
    directory. The alternate path is derived from ``spec.path``, so a test
    fixture root resolves the same way as the live collection root.
    """
    if source not in {"records", "report"}:
        raise ValueError(f"unknown corpus source: {source!r}")
    if spec.kind not in CORPUS_KINDS:
        return spec
    if source == "records":
        path = spec.path.parent.parent if spec.kind == "corpus_report" else spec.path
        return replace(spec, kind="corpus_records", path=path)
    path = (
        spec.path / WORKING_DIRNAME / CORPUS_REPORT_NAME
        if spec.kind == "corpus_records"
        else spec.path
    )
    return replace(spec, kind="corpus_report", path=path)


def load_corpus_records_arm(
    spec: ArmSpec,
    max_reps: int | None,
    *,
    include_turns: bool = True,
) -> tuple[list[RunRow], ArmLoadReport]:
    """Load the scored sweep arm from per-run records under ``spec.path``.

    Each run directory supplies ``run_record.json`` and the ``score.json``
    written by ``strict_posthoc --write-run-scores``. Directories are read in
    sorted order, which is the order the derived report stores. A directory
    holding a record without a stored score raises; that pair is the storage
    contract, and a gap would silently shrink the sample. ``run_dir`` is taken
    from disk, so a relocated repository resolves its own paths.
    """
    if not spec.path.is_dir():
        raise FileNotFoundError(f"{spec.arm_id}: missing run root {spec.path}")
    scores: list[dict[str, Any]] = []
    excluded_underscore = 0
    for record_path in sorted(spec.path.glob(f"*/{RUN_RECORD_NAME}")):
        run_dir = record_path.parent
        if run_dir.name.startswith(EXCLUDED_DIR_NAMES_PREFIX):
            excluded_underscore += 1
            continue
        score_path = run_dir / sp.RUN_SCORE_NAME
        if not score_path.exists():
            raise RuntimeError(
                f"{spec.arm_id}: run dir without {sp.RUN_SCORE_NAME}: {run_dir} "
                "(re-run strict_posthoc.py --write-run-scores)"
            )
        try:
            score = json.loads(score_path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{spec.arm_id}: invalid {sp.RUN_SCORE_NAME}: {score_path}"
            ) from exc
        score["run_dir"] = str(run_dir)
        scores.append(score)
    rows, report = _corpus_rows_from_scores(
        spec, scores, max_reps, include_turns=include_turns
    )
    report.counts["excluded_underscore_dir"] = excluded_underscore
    return rows, report


def load_corpus_report_arm(
    spec: ArmSpec,
    max_reps: int | None,
    *,
    report_path: Path | None = None,
    include_turns: bool = True,
) -> tuple[list[RunRow], ArmLoadReport]:
    """Load the scored sweep arm from the derived aggregate report.

    The report is a summary of the same per-run score rows that
    ``load_corpus_records_arm`` reads. It is retained as an audit path.
    """
    path = report_path or spec.path
    report = json.loads(path.read_text())
    return _corpus_rows_from_scores(
        spec, list(report["scores"]), max_reps, include_turns=include_turns
    )


def _corpus_rows_from_scores(
    spec: ArmSpec,
    scores: Sequence[Mapping[str, Any]],
    max_reps: int | None,
    *,
    include_turns: bool = True,
) -> tuple[list[RunRow], ArmLoadReport]:
    """Build corpus rows from score rows, whichever storage path supplied them.

    ``dirs_seen`` counts score rows; ``records_parsed`` counts the rows passing
    the paper filter before the rep rule. With ``include_turns=False`` turns
    stay None and ``turns_unavailable`` stays 0.
    """
    counts = _new_counts()
    rows: list[RunRow] = []
    counts["dirs_seen"] = len(scores)
    scoring_source = "stored_score" if spec.kind == "corpus_records" else "report"

    # The legacy corpus remains primary. The July pinned rescore can supply a
    # missing repetition key, but it never replaces or duplicates a legacy
    # record. The stored order breaks ties between duplicate records from the
    # same wave.
    def score_key(score: Mapping[str, Any]) -> tuple[str, str, str, int] | None:
        interface = spec.condition_map.get(str(score.get("exec_mode")))
        if (
            score.get("prompt_mode") != spec.prompt_mode
            or interface is None
            or score.get("task_id") not in ANALYSIS_TASK_IDS
        ):
            return None
        return (
            str(score["task_id"]),
            str(score["platform"]),
            interface,
            int(score["rep"]),
        )

    legacy_keys = {
        key
        for score in scores
        if score.get("prompt_wave") == "legacy"
        and (key := score_key(score)) is not None
    }
    candidates: dict[tuple[str, str, str, int], tuple[int, Mapping[str, Any]]] = {}
    for index, score in enumerate(scores):
        key = score_key(score)
        wave = str(score.get("prompt_wave"))
        if (
            key is None
            or score.get("quarantined", False)
            or wave not in {"legacy", CORPUS_FALLBACK_WAVE}
            or (wave == CORPUS_FALLBACK_WAVE and key in legacy_keys)
        ):
            continue
        current = candidates.get(key)
        if current is None or index < current[0]:
            candidates[key] = (index, score)

    selected_indexes = {item[0] for item in candidates.values()}
    counts["excluded_wrong_arm"] = len(scores) - len(selected_indexes)
    for index, score in enumerate(scores):
        if index not in selected_indexes:
            continue
        interface = spec.condition_map[str(score["exec_mode"])]
        counts["records_parsed"] += 1
        rep = int(score["rep"])
        if max_reps is not None and rep >= max_reps:
            counts["excluded_rep_over_cap"] += 1
            continue
        run_dir = str(score.get("run_dir") or "")
        if not score.get("model_response", True):
            no_response_record = _corpus_record(Path(run_dir)) if run_dir else None
            reason = _corpus_no_response_reason(no_response_record)
            if reason == "timeout":
                counts["itt_timeout_failure"] += 1
                rows.append(
                    _itt_row(
                        spec,
                        task_id=str(score["task_id"]),
                        platform=str(score["platform"]),
                        interface=interface,
                        rep=rep,
                        reason="timeout",
                        run_dir=run_dir,
                        cost_usd=(
                            float(score["cost_usd"])
                            if score.get("cost_usd") is not None
                            else None
                        ),
                        tokens=_anthropic_tokens(score.get("tokens") or {}),
                        wall_seconds=(
                            float(score["wall_seconds"])
                            if score.get("wall_seconds") is not None
                            else None
                        ),
                        provider_wall_seconds=(
                            float(score["wall_seconds"])
                            if score.get("wall_seconds") is not None
                            else None
                        ),
                        revisions_used=(
                            _revisions_used(sp.claude_blocks(no_response_record))
                            if no_response_record is not None
                            else None
                        ),
                    )
                )
            elif reason == "session_limit":
                counts["excluded_session_limit"] += 1
            else:
                counts["excluded_no_response_other"] += 1
            continue
        gates = {gate: bool(score["gates"][gate]["pass"]) for gate in GATE_NAMES}
        criteria = {
            criterion: bool(score["criteria"][criterion]["pass"])
            for criterion in CRITERION_NAMES
        }
        record: dict[str, Any] | None = None
        if run_dir and (include_turns or interface == REPAIR_INTERFACE):
            record = _corpus_record(Path(run_dir))
        turns: int | None = None
        if include_turns:
            turns = _blocks_turns(sp.claude_blocks(record)) if record else None
            if turns is None:
                counts["turns_unavailable"] += 1
        row: RunRow = {
            "model": spec.model,
            "model_id": spec.model_id,
            "prompt": spec.prompt_mode,
            "interface": interface,
            "interface_label": INTERFACE_LABELS[interface],
            "platform": str(score["platform"]),
            "task": str(score["task_id"]),
            "rep": rep,
            "task_success": bool(score["task_success"]),
            "gates": gates,
            "criteria": criteria,
            "per_term": score["gates"]["numeric"].get("per_term") or {},
            "cost_usd": (
                float(score["cost_usd"])
                if score.get("cost_usd") is not None
                else None
            ),
            "tokens": _anthropic_tokens(score.get("tokens") or {}),
            "wall_seconds": float(score["wall_seconds"]),
            # The stored corpus score already carries the claude-block sum
            # (strict_posthoc.record_wall_seconds), so it is the provider-call
            # measure for this arm without re-reading the run record.
            "provider_wall_seconds": float(score["wall_seconds"]),
            "turns": turns,
            "revisions_used": (
                _revisions_used(sp.claude_blocks(record))
                if record is not None
                else None
            ),
            "provenance": {
                "arm_id": spec.arm_id,
                "run_dir": run_dir,
                "source_kind": spec.kind,
                "prompt_wave": score.get("prompt_wave"),
                "prompt_sha256": None,
                "scoring_source": scoring_source,
                "usable_reason": "ok",
            },
        }
        rows.append(row)
        counts["usable"] += 1
    return rows, ArmLoadReport(spec.arm_id, True, counts, None)


def load_sonnet_wave_arm(
    spec: ArmSpec,
    max_reps: int | None,
    *,
    references: Mapping[str, Any],
    tasks_by_id: Mapping[str, Task],
    data_dir: Path,
) -> tuple[list[RunRow], ArmLoadReport]:
    """Load the sonnet few-shot wave with a full canonical rescore per record.

    Inline ``scoring``/``scoring_live`` blocks are never consumed. Arm-identity
    fields (model, prompt_mode, prompt_sha256, dir-name rep suffix) are
    asserted, not filtered; a mismatch raises.
    """
    if not spec.path.is_dir():
        raise FileNotFoundError(f"{spec.arm_id}: missing run root {spec.path}")
    counts = _new_counts()
    rows: list[RunRow] = []
    max_created_at: str | None = None
    for entry in sorted(spec.path.iterdir()):
        if not entry.is_dir():
            counts["excluded_non_dir"] += 1
            continue
        if entry.name.startswith(EXCLUDED_DIR_NAMES_PREFIX):
            counts["excluded_underscore_dir"] += 1
            continue
        counts["dirs_seen"] += 1
        record_path = entry / "run_record.json"
        if not record_path.exists():
            counts["missing_run_record"] += 1
            continue
        try:
            record = json.loads(record_path.read_text())
        except json.JSONDecodeError:
            counts["invalid_json"] += 1
            continue
        counts["records_parsed"] += 1
        max_created_at = _track_created_at(record, max_created_at)
        _require(
            record.get("model") == spec.model_id,
            f"{entry.name}: unexpected model {record.get('model')!r}",
        )
        _require(
            record.get("prompt_mode") == spec.prompt_mode,
            f"{entry.name}: unexpected prompt_mode {record.get('prompt_mode')!r}",
        )
        _require(
            bool(record.get("prompt_sha256")),
            f"{entry.name}: missing prompt_sha256",
        )
        rep = record.get("rep")
        _require(
            _rep_from_dir_name(entry.name) == rep,
            f"{entry.name}: dir-name rep suffix != record rep {rep!r}",
        )
        interface = spec.condition_map.get(str(record.get("exec_mode")))
        if interface is None or record.get("task_id") not in ANALYSIS_TASK_IDS:
            counts["excluded_wrong_arm"] += 1
            continue
        if max_reps is not None and int(rep) >= max_reps:
            counts["excluded_rep_over_cap"] += 1
            continue
        usable, reason = wave_record_usable(entry, record)
        if not usable:
            if reason == "timeout":
                counts["itt_timeout_failure"] += 1
                rows.append(
                    _itt_row(
                        spec,
                        task_id=str(record["task_id"]),
                        platform=str(record["platform"]),
                        interface=interface,
                        rep=int(rep),
                        reason="timeout",
                        run_dir=str(entry),
                        cost_usd=sp.record_cost(record, entry),
                        tokens=_anthropic_tokens(sp.record_tokens(record)),
                        wall_seconds=sp.record_wall_seconds(record),
                        provider_wall_seconds=sum_blocks_wall_seconds(
                            record, "claude"
                        ),
                        revisions_used=_revisions_used(sp.claude_blocks(record)),
                    )
                )
            else:
                counts[f"excluded_{reason}"] += 1
            continue
        task = _lookup_task(tasks_by_id, record)
        platform = str(record["platform"])
        gates, per_term = _canonical_gates(
            entry, task, platform, record, references, data_dir
        )
        turns = _blocks_turns(sp.claude_blocks(record))
        if turns is None:
            counts["turns_unavailable"] += 1
        row: RunRow = {
            "model": spec.model,
            "model_id": spec.model_id,
            "prompt": spec.prompt_mode,
            "interface": interface,
            "interface_label": INTERFACE_LABELS[interface],
            "platform": platform,
            "task": str(record["task_id"]),
            "rep": int(rep),
            "task_success": five_gate_success(gates),
            "gates": gates,
            "criteria": criteria_from_gates(gates),
            "per_term": per_term,
            "cost_usd": sp.record_cost(record, entry),
            "tokens": _anthropic_tokens(sp.record_tokens(record)),
            "wall_seconds": sp.record_wall_seconds(record),
            "provider_wall_seconds": sum_blocks_wall_seconds(record, "claude"),
            "turns": turns,
            "revisions_used": _revisions_used(sp.claude_blocks(record)),
            "provenance": {
                "arm_id": spec.arm_id,
                "run_dir": str(entry),
                "source_kind": spec.kind,
                "prompt_sha256": record.get("prompt_sha256"),
                "scoring_source": "canonical_rescore",
                "usable_reason": reason,
            },
        }
        rows.append(row)
        counts["usable"] += 1
    return rows, ArmLoadReport(spec.arm_id, True, counts, max_created_at)


def load_codex_pilot_arm(
    spec: ArmSpec,
    max_reps: int | None,
    *,
    references: Mapping[str, Any],
    tasks_by_id: Mapping[str, Task],
    data_dir: Path,
) -> tuple[list[RunRow], ArmLoadReport]:
    """Load a Codex pilot arm and recompute every scoring gate.

    ``codex_zero_shot`` raises on a directory without a parseable
    ``run_record.json`` (contamination tripwire); the few-shot wave counts it
    as ``in_flight_missing_record`` and moves on. ``cost_usd`` is always None
    (subscription-metered CLI); ``turns`` is always None (no turn telemetry).
    """
    if not spec.path.is_dir():
        raise FileNotFoundError(f"{spec.arm_id}: missing run root {spec.path}")
    strict_root = spec.arm_id == "codex_zero_shot"
    counts = _new_counts()
    rows: list[RunRow] = []
    max_created_at: str | None = None
    for entry in sorted(spec.path.iterdir()):
        if not entry.is_dir():
            counts["excluded_non_dir"] += 1
            continue
        if entry.name.startswith(EXCLUDED_DIR_NAMES_PREFIX):
            counts["excluded_underscore_dir"] += 1
            continue
        counts["dirs_seen"] += 1
        record_path = entry / "run_record.json"
        if not record_path.exists():
            if strict_root:
                raise RuntimeError(
                    f"{spec.arm_id}: run dir without run_record.json: {entry} "
                    "(contamination tripwire; see module docstring)"
                )
            counts["in_flight_missing_record"] += 1
            continue
        try:
            record = json.loads(record_path.read_text())
        except json.JSONDecodeError as exc:
            if strict_root:
                raise RuntimeError(
                    f"{spec.arm_id}: invalid run_record.json: {entry}"
                ) from exc
            counts["invalid_json"] += 1
            continue
        counts["records_parsed"] += 1
        max_created_at = _track_created_at(record, max_created_at)
        record_mode = record.get("prompt_mode")
        if record_mode != spec.prompt_mode:
            if record_mode == "few_shot" and spec.prompt_mode == "zero_shot":
                counts["excluded_retired_fewshot"] += 1
            else:
                counts["excluded_wrong_arm"] += 1
            continue
        interface = spec.condition_map.get(str(record.get("exec_mode")))
        if (
            interface is None
            or record.get("model") != spec.model_id
            or record.get("task_id") not in ANALYSIS_TASK_IDS
        ):
            counts["excluded_wrong_arm"] += 1
            continue
        rep = int(record["rep"])
        if max_reps is not None and rep >= max_reps:
            counts["excluded_rep_over_cap"] += 1
            continue
        scoring = record.get("scoring")
        if not isinstance(scoring, dict):
            if strict_root:
                raise RuntimeError(
                    f"{spec.arm_id}: record without scoring block: {entry}"
                )
            counts["in_flight_missing_record"] += 1
            continue
        collected, reason = codex_collection_ok(record)
        if not collected:
            if reason == "timed_out":
                counts["itt_timeout_failure"] += 1
                rows.append(
                    _itt_row(
                        spec,
                        task_id=str(record["task_id"]),
                        platform=str(record["platform"]),
                        interface=interface,
                        rep=rep,
                        reason="timed_out",
                        run_dir=str(entry),
                        cost_usd=None,
                        tokens=_openai_tokens(
                            sum_blocks_usage(record, "codex", OPENAI_USAGE_KEYS)
                        ),
                        wall_seconds=_record_wall_seconds(record, "codex"),
                        provider_wall_seconds=sum_blocks_wall_seconds(
                            record, "codex"
                        ),
                        revisions_used=_revisions_used(
                            _provider_blocks(record, "codex")
                        ),
                    )
                )
            else:
                counts["excluded_collection_failure"] += 1
            continue
        task = _lookup_task(tasks_by_id, record)
        platform = str(record["platform"])
        gates, per_term = _canonical_gates(
            entry, task, platform, record, references, data_dir
        )
        row: RunRow = {
            "model": spec.model,
            "model_id": spec.model_id,
            "prompt": spec.prompt_mode,
            "interface": interface,
            "interface_label": INTERFACE_LABELS[interface],
            "platform": platform,
            "task": str(record["task_id"]),
            "rep": rep,
            "task_success": five_gate_success(gates),
            "gates": gates,
            "criteria": criteria_from_gates(gates),
            "per_term": per_term,
            "cost_usd": None,
            "tokens": _openai_tokens(
                sum_blocks_usage(record, "codex", OPENAI_USAGE_KEYS)
            ),
            "wall_seconds": _record_wall_seconds(record, "codex"),
            "provider_wall_seconds": sum_blocks_wall_seconds(record, "codex"),
            "turns": None,
            "revisions_used": _revisions_used(_provider_blocks(record, "codex")),
            "provenance": {
                "arm_id": spec.arm_id,
                "run_dir": str(entry),
                "source_kind": spec.kind,
                "prompt_sha256": record.get("prompt_sha256"),
                "scoring_source": "canonical_rescore",
                "usable_reason": reason,
            },
        }
        rows.append(row)
        counts["usable"] += 1
    return rows, ArmLoadReport(spec.arm_id, True, counts, max_created_at)


def load_openrouter_pilot_arm(
    spec: ArmSpec,
    max_reps: int | None,
    *,
    references: Mapping[str, Any],
    tasks_by_id: Mapping[str, Task],
    data_dir: Path,
) -> tuple[list[RunRow], ArmLoadReport]:
    """Load an OpenRouter pilot arm and recompute every scoring gate.

    An absent root with ``allow_absent`` reports ``present=False`` with zero
    rows and never raises. Rows failing the four inline gates stay in the
    denominator; only API collection failures (no ``ok is True`` block) and
    partially written records are excluded.
    """
    counts = _new_counts()
    if not spec.path.is_dir():
        if spec.allow_absent:
            return [], ArmLoadReport(spec.arm_id, False, counts, None)
        raise FileNotFoundError(f"{spec.arm_id}: missing run root {spec.path}")
    rows: list[RunRow] = []
    max_created_at: str | None = None
    for entry in sorted(spec.path.iterdir()):
        if not entry.is_dir():
            counts["excluded_non_dir"] += 1
            continue
        if entry.name.startswith(EXCLUDED_DIR_NAMES_PREFIX):
            counts["excluded_underscore_dir"] += 1
            continue
        counts["dirs_seen"] += 1
        record_path = entry / "run_record.json"
        if not record_path.exists():
            counts["in_flight_missing_record"] += 1
            continue
        try:
            record = json.loads(record_path.read_text())
        except json.JSONDecodeError:
            counts["invalid_json"] += 1
            continue
        counts["records_parsed"] += 1
        max_created_at = _track_created_at(record, max_created_at)
        record_mode = record.get("prompt_mode")
        if record_mode != spec.prompt_mode:
            if record_mode == "few_shot" and spec.prompt_mode == "zero_shot":
                counts["excluded_retired_fewshot"] += 1
            else:
                counts["excluded_wrong_arm"] += 1
            continue
        interface = spec.condition_map.get(str(record.get("exec_mode")))
        if (
            interface is None
            or record.get("model") != spec.model_id
            or record.get("task_id") not in ANALYSIS_TASK_IDS
        ):
            counts["excluded_wrong_arm"] += 1
            continue
        rep = int(record["rep"])
        if max_reps is not None and rep >= max_reps:
            counts["excluded_rep_over_cap"] += 1
            continue
        scoring = record.get("scoring")
        if not isinstance(scoring, dict):
            counts["in_flight_missing_record"] += 1
            continue
        if not openrouter_api_ok(record):
            blocks = _provider_blocks(record, "openrouter")
            timed_out = any(
                block.get("timed_out")
                or "timeout" in str(block.get("error") or "").lower()
                for block in blocks
            )
            if timed_out:
                counts["itt_timeout_failure"] += 1
                rows.append(
                    _itt_row(
                        spec,
                        task_id=str(record["task_id"]),
                        platform=str(record["platform"]),
                        interface=interface,
                        rep=rep,
                        reason="timed_out",
                        run_dir=str(entry),
                        cost_usd=sum_blocks_cost(record, "openrouter"),
                        tokens=_openai_tokens(
                            sum_blocks_usage(record, "openrouter", OPENAI_USAGE_KEYS)
                        ),
                        wall_seconds=_record_wall_seconds(record, "openrouter"),
                        provider_wall_seconds=sum_blocks_wall_seconds(
                            record, "openrouter"
                        ),
                        revisions_used=_revisions_used(
                            _provider_blocks(record, "openrouter")
                        ),
                    )
                )
            else:
                counts["excluded_collection_failure"] += 1
            continue
        task = _lookup_task(tasks_by_id, record)
        platform = str(record["platform"])
        gates, per_term = _canonical_gates(
            entry, task, platform, record, references, data_dir
        )
        turns = _blocks_turns(_provider_blocks(record, "openrouter"))
        if turns is None:
            counts["turns_unavailable"] += 1
        row: RunRow = {
            "model": spec.model,
            "model_id": spec.model_id,
            "prompt": spec.prompt_mode,
            "interface": interface,
            "interface_label": INTERFACE_LABELS[interface],
            "platform": platform,
            "task": str(record["task_id"]),
            "rep": rep,
            "task_success": five_gate_success(gates),
            "gates": gates,
            "criteria": criteria_from_gates(gates),
            "per_term": per_term,
            "cost_usd": sum_blocks_cost(record, "openrouter"),
            "tokens": _openai_tokens(
                sum_blocks_usage(record, "openrouter", OPENAI_USAGE_KEYS)
            ),
            "wall_seconds": _record_wall_seconds(record, "openrouter"),
            "provider_wall_seconds": sum_blocks_wall_seconds(record, "openrouter"),
            "turns": turns,
            "revisions_used": _revisions_used(
                _provider_blocks(record, "openrouter")
            ),
            "provenance": {
                "arm_id": spec.arm_id,
                "run_dir": str(entry),
                "source_kind": spec.kind,
                "prompt_sha256": record.get("prompt_sha256"),
                "scoring_source": "canonical_rescore",
                "usable_reason": "ok",
            },
        }
        rows.append(row)
        counts["usable"] += 1
    return rows, ArmLoadReport(spec.arm_id, True, counts, max_created_at)


def _effective_path(path: Path, project_root: Path) -> Path:
    if project_root == PROJECT_ROOT:
        return path
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path
    return project_root / rel


def load_runs(
    arm_ids: Sequence[str] | None = None,
    max_reps: int | None = DEFAULT_MAX_REPS,
    project_root: Path = PROJECT_ROOT,
    references: dict[str, Any] | None = None,
    tasks_by_id: dict[str, Task] | None = None,
    data_dir: Path | None = None,
    include_turns: bool = True,
    corpus_source: str | None = None,
    verify_manifest: bool | Path = False,
    manifest_sample_version: str = SAMPLE_VERSION,
) -> LoadResult:
    """Load the requested arms into normalized rows plus per-arm reports.

    ``max_reps=12`` keeps the first twelve eligible repetitions in each
    arm/task/platform/interface cell. Model-side operating-cap timeouts are
    eligible failure rows. Provider interruptions are excluded before the
    repetitions are ordered. ``max_reps=None`` keeps every eligible row.
    ``references``/``tasks_by_id``/``data_dir`` are injection points for
    tests; the defaults resolve under ``project_root``.

    ``corpus_source`` overrides the storage path of the Sonnet zero-shot arm:
    ``"records"`` reads per-run ``score.json`` files, ``"report"`` reads the
    derived aggregate report. The default follows the registry, which selects
    per-run records.

    ``verify_manifest`` checks the selected sample against the pinned sample
    manifest, so silent drift of the saved collection raises instead of
    changing numbers. ``True`` reads the manifest at
    ``<project_root>/data/analysis/sample_manifest.json``; a ``Path`` reads
    that file. Verification covers exactly the loaded arms.
    ``manifest_sample_version`` states which pin the caller expects; the paper
    chain passes ``PAPER_SAMPLE_VERSION`` with the paper manifest path.
    """
    selected = list(arm_ids) if arm_ids is not None else list(ARMS)
    unknown = [arm_id for arm_id in selected if arm_id not in ARMS]
    if unknown:
        raise KeyError(f"unknown arm ids: {unknown}")
    if tasks_by_id is None:
        tasks_by_id = {task.task_id: task for task in tasks_mod.TASKS}
    if data_dir is None:
        data_dir = project_root / "data" / "tasks" / "inputs"
    specs = {arm_id: ARMS[arm_id] for arm_id in selected}
    if corpus_source is not None:
        specs = {
            arm_id: corpus_source_spec(spec, corpus_source)
            for arm_id, spec in specs.items()
        }
    needs_references = any(spec.kind not in CORPUS_KINDS for spec in specs.values())
    if references is None and needs_references:
        references = sp.load_references(project_root / "data" / "tasks" / "references")
    rows: list[RunRow] = []
    reports: dict[str, ArmLoadReport] = {}
    for arm_id in selected:
        spec = specs[arm_id]
        effective = replace(spec, path=_effective_path(spec.path, project_root))
        if spec.kind == "corpus_records":
            arm_rows, report = load_corpus_records_arm(
                effective, None, include_turns=include_turns
            )
        elif spec.kind == "corpus_report":
            arm_rows, report = load_corpus_report_arm(
                effective, None, include_turns=include_turns
            )
        elif spec.kind == "sonnet_wave":
            arm_rows, report = load_sonnet_wave_arm(
                effective,
                None,
                references=references or {},
                tasks_by_id=tasks_by_id,
                data_dir=data_dir,
            )
        elif spec.kind == "codex_pilot":
            arm_rows, report = load_codex_pilot_arm(
                effective,
                None,
                references=references or {},
                tasks_by_id=tasks_by_id,
                data_dir=data_dir,
            )
        elif spec.kind == "openrouter_pilot":
            arm_rows, report = load_openrouter_pilot_arm(
                effective,
                None,
                references=references or {},
                tasks_by_id=tasks_by_id,
                data_dir=data_dir,
            )
        else:  # pragma: no cover - registry is static
            raise ValueError(f"unknown arm kind: {spec.kind}")
        rows.extend(arm_rows)
        reports[arm_id] = report
    # Primary outcome annotation: every consumer of load_runs inherits the
    # unified task_success and the preserved strict verdict from here.
    attach_unified_outcome(
        rows,
        tasks_by_id=tasks_by_id,
        references=references,
        reference_dir=project_root / "data" / "tasks" / "references",
    )
    rows = select_first_eligible_rows(rows, max_reps)
    manifest = completeness_manifest(
        rows, reports, max_reps if max_reps is not None else DEFAULT_MAX_REPS
    )
    if verify_manifest:
        manifest_path = (
            verify_manifest
            if isinstance(verify_manifest, Path)
            else project_root / SAMPLE_MANIFEST_SUBPATH
        )
        verify_rows_against_sample_manifest(
            rows,
            manifest_path,
            max_reps=max_reps,
            project_root=project_root,
            sample_version=manifest_sample_version,
        )
    return LoadResult(rows=rows, reports=reports, manifest=manifest)


def load_paper_runs(
    max_reps: int | None = DEFAULT_MAX_REPS,
    project_root: Path = PROJECT_ROOT,
    *,
    include_turns: bool = True,
    arm_ids: Sequence[str] | None = None,
) -> LoadResult:
    """Load the paper-scope sample and verify it against the paper manifest.

    The estimation and numbers chain enters through this function, so every
    build checks the 9,072 pinned rows before it computes a number. ``arm_ids``
    narrows the load to a subset of ``PAPER_ARM_IDS`` (one model arm, for
    example); anything outside the paper scope raises.
    """
    selected = list(PAPER_ARM_IDS) if arm_ids is None else list(arm_ids)
    outside = [arm_id for arm_id in selected if arm_id not in PAPER_ARM_IDS]
    if outside:
        raise KeyError(f"arm ids outside the paper scope: {outside}")
    return load_runs(
        arm_ids=selected,
        max_reps=max_reps,
        project_root=project_root,
        include_turns=include_turns,
        verify_manifest=project_root / PAPER_SAMPLE_MANIFEST_SUBPATH,
        manifest_sample_version=PAPER_SAMPLE_VERSION,
    )


def select_first_eligible_rows(
    rows: Sequence[RunRow], max_reps: int | None = DEFAULT_MAX_REPS
) -> list[RunRow]:
    """Select the first distinct eligible repetitions within each design cell.

    Loaders omit provider interruptions and emit model-side timeouts as
    intention-to-treat failure rows. This function therefore selects on
    repetition only, without using outcomes. A stable run-directory tie-break
    prevents duplicate records for one repetition from entering twice.
    """
    if max_reps is None:
        return list(rows)
    if max_reps < 1:
        raise ValueError("max_reps must be positive or None")

    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["provenance"]["arm_id"]),
            row["task"],
            row["platform"],
            row["interface"],
            int(row["rep"]),
            str(row["provenance"].get("run_dir") or ""),
        ),
    )
    selected: list[RunRow] = []
    counts: dict[tuple[str, str, str, str], int] = {}
    seen_reps: set[tuple[str, str, str, str, int]] = set()
    for row in ordered:
        cell = (
            str(row["provenance"]["arm_id"]),
            row["task"],
            row["platform"],
            row["interface"],
        )
        rep_key = (*cell, int(row["rep"]))
        if rep_key in seen_reps or counts.get(cell, 0) >= max_reps:
            continue
        seen_reps.add(rep_key)
        counts[cell] = counts.get(cell, 0) + 1
        selected.append(row)
    return selected


def completeness_manifest(
    rows: Sequence[RunRow],
    reports: Mapping[str, ArmLoadReport],
    max_reps: int = DEFAULT_MAX_REPS,
) -> dict[str, Any]:
    """Per-arm cell coverage over the fixed 21 x 3 x 3 design grid.

    Pure function over rows and reports; cell ids use the
    ``task__platform__interface`` convention. ``status`` is ``absent`` when the
    arm root was missing, ``complete`` when every cell holds at least
    ``max_reps`` runs, else ``partial``.
    """
    n_cells = len(ANALYSIS_TASK_IDS) * len(PLATFORMS) * len(INTERFACES)
    manifest: dict[str, Any] = {
        "expected": {
            "tasks": len(ANALYSIS_TASK_IDS),
            "platforms": len(PLATFORMS),
            "interfaces": len(INTERFACES),
            "cells": n_cells,
            "runs_at_12_reps": n_cells * max_reps,
        },
        "arms": {},
    }
    by_arm: dict[str, list[RunRow]] = {arm_id: [] for arm_id in reports}
    for row in rows:
        by_arm.setdefault(row["provenance"]["arm_id"], []).append(row)
    for arm_id, report in reports.items():
        arm_rows = by_arm.get(arm_id, [])
        cell_counts: dict[tuple[str, str, str], int] = {
            (task, platform, interface): 0
            for task in ANALYSIS_TASK_IDS
            for platform in PLATFORMS
            for interface in INTERFACES
        }
        for row in arm_rows:
            key = (row["task"], row["platform"], row["interface"])
            cell_counts[key] = cell_counts.get(key, 0) + 1
        empty_cells = sorted(
            "__".join(key) for key, count in cell_counts.items() if count == 0
        )
        short_cells = sorted(
            (
                {"cell": "__".join(key), "n": count}
                for key, count in cell_counts.items()
                if 0 < count < max_reps
            ),
            key=lambda item: item["cell"],
        )
        cells_full = sum(1 for count in cell_counts.values() if count >= max_reps)
        tasks_absent = sorted(
            task
            for task in ANALYSIS_TASK_IDS
            if all(
                cell_counts[(task, platform, interface)] == 0
                for platform in PLATFORMS
                for interface in INTERFACES
            )
        )
        if not report.present:
            status = "absent"
        elif cells_full == len(cell_counts):
            status = "complete"
        else:
            status = "partial"
        manifest["arms"][arm_id] = {
            "present": report.present,
            "status": status,
            "usable_runs": len(arm_rows),
            "cells_full": cells_full,
            "cells_short": len(short_cells),
            "cells_empty": len(empty_cells),
            "empty_cells": empty_cells,
            "short_cells": short_cells,
            "tasks_entirely_absent": tasks_absent,
            "attrition": dict(report.counts),
            "max_created_at": report.max_created_at,
        }
    return manifest


# ---------------------------------------------------------------------------
# Pinned sample manifest: the selected run directories plus the sha256 of
# each run's result.json. The analysis build writes it; every downstream
# builder verifies against it, so silent drift of the saved collection raises
# instead of changing numbers.
# ---------------------------------------------------------------------------


class SampleManifestError(RuntimeError):
    """The selected sample does not match the pinned sample manifest."""


def _manifest_run_dir(run_dir: str, project_root: Path) -> str:
    """A run directory as stored in the manifest: repo-relative when possible."""
    path = Path(run_dir)
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _result_sha256(run_dir: str) -> str | None:
    """SHA-256 of the run's ``result.json`` bytes; None when the file is absent."""
    if not run_dir:
        return None
    path = Path(run_dir) / "result.json"
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_manifest_entries(
    rows: Sequence[RunRow], project_root: Path = PROJECT_ROOT
) -> list[dict[str, Any]]:
    """One manifest entry per selected row, sorted by (arm_id, run_dir)."""
    entries = [
        {
            "arm_id": str(row["provenance"]["arm_id"]),
            "run_dir": _manifest_run_dir(
                str(row["provenance"].get("run_dir") or ""), project_root
            ),
            "result_sha256": _result_sha256(
                str(row["provenance"].get("run_dir") or "")
            ),
        }
        for row in rows
    ]
    entries.sort(key=lambda entry: (entry["arm_id"], entry["run_dir"]))
    return entries


def build_sample_manifest(
    rows: Sequence[RunRow],
    max_reps: int | None = DEFAULT_MAX_REPS,
    project_root: Path = PROJECT_ROOT,
    *,
    sample_version: str = SAMPLE_VERSION,
) -> dict[str, Any]:
    """The pinned sample manifest payload for the selected rows.

    ``sample_version`` stamps the rule the rows follow. The key order and the
    key set are part of the stored pin, so a caller that wants extra fields
    adds them after this function returns.
    """
    return {
        "generated_by": "code/benchmark/build_full_analysis.py",
        "sample_version": sample_version,
        "primary_outcome_version": PRIMARY_OUTCOME_VERSION,
        "scoring_version": sp.SCORING_VERSION,
        "max_reps": max_reps,
        "n_rows": len(rows),
        "runs": sample_manifest_entries(rows, project_root),
    }


def build_paper_sample_manifest(
    rows: Sequence[RunRow],
    max_reps: int | None = DEFAULT_MAX_REPS,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """The pinned paper-sample manifest for the two paper arms.

    ``rows`` may hold every arm; only the paper rows are pinned. The payload
    carries ``PAPER_SAMPLE_VERSION`` so a builder that verifies against the
    corpus manifest can never accept this file by accident, and the reverse.
    """
    selected = paper_rows(rows)
    payload = build_sample_manifest(
        selected,
        max_reps,
        project_root,
        sample_version=PAPER_SAMPLE_VERSION,
    )
    payload["scope"] = "paper"
    payload["models"] = list(PAPER_MODELS)
    payload["arm_ids"] = list(PAPER_ARM_IDS)
    payload["decision_record"] = (
        "quality_reports/decision_records/2026-08-21_two-model-main-specification.md"
    )
    return payload


def verify_rows_against_sample_manifest(
    rows: Sequence[RunRow],
    manifest_path: Path,
    *,
    max_reps: int | None = DEFAULT_MAX_REPS,
    project_root: Path = PROJECT_ROOT,
    sample_version: str = SAMPLE_VERSION,
) -> None:
    """Raise ``SampleManifestError`` unless the rows match the pinned manifest.

    Verification covers exactly the arms present in ``rows``: for each of
    those arms the selected run directories and their ``result.json`` hashes
    must equal the stored entries. Version strings and the repetition cap
    must match as well. A missing manifest file raises, so a builder can
    never silently run unpinned. ``sample_version`` selects which pin the
    caller expects, so the corpus manifest and the paper manifest can never
    be substituted for one another.
    """
    if not manifest_path.is_file():
        raise SampleManifestError(
            f"pinned sample manifest not found: {manifest_path} "
            "(run build_full_analysis.py to write it)"
        )
    stored = json.loads(manifest_path.read_text())
    problems: list[str] = []
    if stored.get("sample_version") != sample_version:
        problems.append(
            f"sample_version {stored.get('sample_version')!r} != {sample_version!r}"
        )
    if stored.get("primary_outcome_version") != PRIMARY_OUTCOME_VERSION:
        problems.append(
            f"primary_outcome_version {stored.get('primary_outcome_version')!r} "
            f"!= {PRIMARY_OUTCOME_VERSION!r}"
        )
    if stored.get("max_reps") != max_reps:
        problems.append(f"max_reps {stored.get('max_reps')!r} != {max_reps!r}")
    live = sample_manifest_entries(rows, project_root)
    live_arms = {entry["arm_id"] for entry in live}
    stored_by_key = {
        (entry["arm_id"], entry["run_dir"]): entry.get("result_sha256")
        for entry in stored.get("runs", [])
        if entry.get("arm_id") in live_arms
    }
    live_by_key = {
        (entry["arm_id"], entry["run_dir"]): entry["result_sha256"]
        for entry in live
    }
    for key in sorted(set(stored_by_key) - set(live_by_key)):
        problems.append(f"missing from live sample: {key[0]} {key[1]}")
    for key in sorted(set(live_by_key) - set(stored_by_key)):
        problems.append(f"not in pinned manifest: {key[0]} {key[1]}")
    for key in sorted(set(live_by_key) & set(stored_by_key)):
        if live_by_key[key] != stored_by_key[key]:
            problems.append(
                f"result.json hash changed: {key[0]} {key[1]} "
                f"(pinned {stored_by_key[key]!r}, live {live_by_key[key]!r})"
            )
    if problems:
        shown = "; ".join(problems[:10])
        more = f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""
        raise SampleManifestError(
            f"sample drift against {manifest_path}: {shown}{more}"
        )

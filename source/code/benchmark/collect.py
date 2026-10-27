"""The one collection entry point for the six benchmark arms.

Replaces the two collector CLIs (``harness.py --live`` and the pilot runner),
the three status scanners, and the seventeen driver shell scripts. Routing is
driven by the ``analysis_dataset.ARMS`` table: the arm root and the pinned
model id come from the ``ArmSpec``, so there is no ``--runs-dir`` and no way
to select a floating model alias.

Usage::

    uv run python code/benchmark/collect.py --model sonnet --reps-to 20
    uv run python code/benchmark/collect.py --model sonnet --status
    uv run python code/benchmark/collect.py --model deepseek \
        --prompt zero_shot --language python --agency chatbot \
        --tasks ols_robust --reps-to 9 --dry-run

Semantics, one rule each:

- The arm root is ``ARMS[f"{model}_{prompt}"].path``. Naming both prompt
  modes builds a two-root universe.
- Agency is named ``chatbot``, ``self_repair``, ``constrained_agent`` and is
  mapped to each arm's stored condition through ``ArmSpec.condition_map``.
- ``--reps-to N`` is a target count of eligible repetitions per cell under
  the analysis loader's own rule; a rejection occupies its index but does
  not count.
- The next repetition index is one past the highest stored index, counting
  record-less directories, so no stored record is ever overwritten.
- A cell whose stored ``prompt_sha256`` differs from the rebuilt prompt hash
  is refused; the other cells continue.
- On a provider rejection the runner stores the stamped record, waits,
  re-probes the meter, and retries the same cell at the following index. It
  never advances to another cell while the provider is refusing and never
  exits on a meter reading.
- ``--until HH:MM`` stops every shard at the local deadline.
- ``--balanced-first`` (default) fills the emptiest cells first, level by
  level, toward ``--ceiling``.
- After collection, every new run in a ``corpus_records`` arm receives its
  ``score.json`` sidecar so the analysis loader can read it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_dataset as ad  # noqa: E402
import cells  # noqa: E402
import harness  # noqa: E402
import miniswe_backend  # noqa: E402
import strict_posthoc as sp  # noqa: E402

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
MODEL_CHOICES = ("sonnet", "codex", "deepseek")
PROMPT_CHOICES = ("zero_shot", "few_shot")
LANGUAGE_CHOICES = ("python", "r", "stata")
#: CLI agency names -> internal interface keys (the Sonnet stored vocabulary).
AGENCY_CLI = {
    "chatbot": "no_feedback",
    "self_repair": "self_repair",
    "constrained_agent": "tool_loop",
}

#: Eligibility reasons that count toward ``--reps-to`` (the analysis loader's
#: rule: a model response or a model-side timeout counts; a provider
#: session-limit rejection does not).
ELIGIBLE_REASONS = frozenset({"ok", "result_json_rescue", "timeout"})

DEFAULT_CEILING = 20
#: Runaway guard on the repetition index (matches the 2026-08-17 drivers).
MAX_REP = 40
MAX_ROUNDS = 400
#: Meter thresholds: wait (never exit) while a meter sits at or above these.
SESSION_STOP_PERCENT = 98.0
WEEKLY_STOP_PERCENT = 98.0
DEFAULT_METER_WAIT_SECONDS = 120.0
#: Collection conditions for the Claude backend, recorded into every record.
#: Constants (not flags) so a future wave cannot silently differ between arms.
CLAUDE_TIMEOUT_SECONDS = harness.TIMEOUT_NO_FEEDBACK
CLAUDE_EXTRA_CLI_ARGS: list[str] = []

#: Legacy model names accepted per arm when READING stored records. The
#: zero-shot corpus predates the explicit pin and stores the alias ``sonnet``
#: (2,588 records); filtering it out once produced a phantom deficit
#: (quality_reports/reports/2026-08-17_sonnet-hammer-launch.md). The model an
#: arm RUNS is always ``ArmSpec.model_id``.
ACCEPTED_MODELS: dict[str, frozenset[str]] = {
    "sonnet_zero_shot": frozenset({"claude-sonnet-4-6", "sonnet"}),
}

#: Transition tolerance for records with no ``arm`` field. The legacy
#: zero-shot records were stamped on 2026-08-18 (decision record
#: 2026-08-18_collection-consolidation), so a missing stamp now indicates a
#: foreign record and the guard refuses it.
TOLERATE_MISSING_ARM_STAMP = False

RUN_RECORD_NAME = "run_record.json"


def accepted_models(spec: ad.ArmSpec) -> frozenset[str]:
    return ACCEPTED_MODELS.get(spec.arm_id, frozenset({spec.model_id}))


def arm_marker(spec: ad.ArmSpec) -> str:
    """The ``_working/.pilot_arm`` label convention: provider arm, not path."""
    if spec.provider == "anthropic":
        return "sonnet_pilot"
    if spec.provider == "openai_codex":
        return "codex_pilot"
    return cells.openrouter_arm_for(spec.model_id)


def stored_condition(spec: ad.ArmSpec, interface: str) -> str:
    """Invert ``ArmSpec.condition_map``: internal interface -> stored token."""
    for stored, internal in spec.condition_map.items():
        if internal == interface:
            return stored
    raise KeyError(f"{spec.arm_id} has no condition for interface {interface!r}")


# --------------------------------------------------------------------------- #
# Cells and their disk status
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cell:
    arm_id: str
    task_id: str
    platform: str
    interface: str  # internal key: no_feedback / self_repair / tool_loop

    @property
    def spec(self) -> ad.ArmSpec:
        return ad.ARMS[self.arm_id]

    @property
    def condition(self) -> str:
        return stored_condition(self.spec, self.interface)

    @property
    def prefix(self) -> str:
        spec = self.spec
        return (
            f"{self.task_id}__{self.platform}__{spec.prompt_mode}"
            f"__{self.condition}__rep"
        )

    def label(self) -> str:
        return f"{self.arm_id}/{self.prefix[:-5]}"


@dataclass
class CellStatus:
    eligible_reps: set[int] = field(default_factory=set)
    stored_reps: set[int] = field(default_factory=set)
    rejections: int = 0
    missing_score: int = 0
    prompt_hashes: set[str] = field(default_factory=set)

    @property
    def eligible(self) -> int:
        return len(self.eligible_reps)

    @property
    def next_rep(self) -> int:
        return (max(self.stored_reps) + 1) if self.stored_reps else 0


def record_is_rejection(spec: ad.ArmSpec, record: dict[str, Any]) -> bool:
    """One rejection test for every arm: the stamped flag, plus the deep
    block scan on Claude arms whose legacy records predate the stamp."""
    if record.get("provider_limit"):
        return True
    if spec.provider == "anthropic":
        return harness.is_provider_limit_record(record)
    return False


def _openrouter_blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        value
        for key, value in record.items()
        if isinstance(value, dict)
        and (key == "openrouter" or key.startswith("openrouter_attempt_"))
    ]


def record_eligibility(
    spec: ad.ArmSpec, run_dir: Path, record: dict[str, Any]
) -> tuple[bool, str]:
    """Does this stored record count toward ``--reps-to``?

    Mirrors the loader: a model response or a model-side timeout is an
    analysis row; a provider rejection or another collection failure is not.
    """
    if spec.provider == "anthropic":
        _, reason = ad.wave_record_usable(run_dir, record)
        return reason in ELIGIBLE_REASONS, reason
    if spec.provider == "openai_codex":
        ok, reason = ad.codex_collection_ok(record)
        return ok or reason == "timed_out", reason
    if ad.openrouter_api_ok(record):
        return True, "ok"
    blocks = _openrouter_blocks(record)
    timed_out = any(
        block.get("timed_out")
        or "timeout" in str(block.get("error") or "").lower()
        for block in blocks
    )
    return (True, "timed_out") if timed_out else (False, "collection_failure")


def cell_status(cell: Cell) -> CellStatus:
    """Scan one cell's run directories.

    The directory name is the key (it matches the stored identity in all
    22,764 records); nothing is inferred from record-internal id fields. A
    directory with no record still reserves its repetition index, and a
    truncated record with no model field is read as occupied, with
    eligibility decided by its surviving content.
    """
    spec = cell.spec
    status = CellStatus()
    accepted = accepted_models(spec)
    for entry in spec.path.glob(f"{cell.prefix}*"):
        if not entry.is_dir():
            continue
        suffix = entry.name.rsplit("rep", 1)[-1]
        if not suffix.isdigit():
            continue
        rep = int(suffix)
        status.stored_reps.add(rep)
        record_path = entry / RUN_RECORD_NAME
        if not record_path.exists():
            continue  # interrupted run: the index stays reserved
        try:
            record = json.loads(record_path.read_text())
        except json.JSONDecodeError:
            continue
        stored_hash = record.get("prompt_sha256")
        if stored_hash:
            status.prompt_hashes.add(str(stored_hash))
        if record_is_rejection(spec, record):
            status.rejections += 1
        if spec.kind == "corpus_records" and not (entry / sp.RUN_SCORE_NAME).exists():
            status.missing_score += 1
        model = record.get("model")
        if model is not None and str(model) not in accepted:
            continue  # foreign-model record: occupies its index, never eligible
        eligible, _ = record_eligibility(spec, entry, record)
        if eligible:
            status.eligible_reps.add(rep)
    return status


def expected_prompt_sha256(cell: Cell) -> str:
    """Rebuild the cell's initial prompt with the live harness and hash it."""
    spec = cell.spec
    task = cells.TASKS_BY_ID[cell.task_id]
    if spec.provider == "anthropic":
        text = harness.initial_prompt(
            task, cell.platform, spec.prompt_mode, cell.interface
        )
    elif spec.provider == "openai_codex":
        if cell.interface == "tool_loop":
            text = harness.tool_loop_prompt(task, cell.platform, spec.prompt_mode)
        else:
            text = (
                harness.no_feedback_prompt(task, cell.platform, spec.prompt_mode)
                + cells.CODEX_GENERATOR_NOTE
            )
    else:
        if cell.condition == "or_agent_swe":
            text = miniswe_backend.swe_tool_loop_prompt(
                task, cell.platform, spec.prompt_mode
            )
        elif cell.interface == "tool_loop":
            text = harness.tool_loop_prompt(task, cell.platform, spec.prompt_mode)
        else:
            text = harness.no_feedback_prompt(task, cell.platform, spec.prompt_mode)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Arm guard
# --------------------------------------------------------------------------- #
def assert_arm_root(
    spec: ad.ArmSpec, tolerate_missing_arm: bool | None = None
) -> None:
    """Refuse to collect into a directory that is not the resolved arm.

    The root is derived from the ``ARMS`` table and cannot be supplied, so
    the old path denylist is unnecessary. The guard checks the marker file,
    scans a sample of records for a conflicting ``arm`` stamp or a foreign
    model name, and refuses a push-capable environment.
    """
    if tolerate_missing_arm is None:
        tolerate_missing_arm = TOLERATE_MISSING_ARM_STAMP
    root = spec.path
    expected = arm_marker(spec)
    marker = root / harness.WORKING_DIRNAME / ".pilot_arm"
    if marker.exists():
        found = marker.read_text().strip()
        if found != expected:
            raise SystemExit(
                f"REFUSED: {marker} says arm={found!r} != {expected!r}."
            )
    accepted = accepted_models(spec)
    for index, path in enumerate(sorted(root.glob(f"*/{RUN_RECORD_NAME}"))):
        if index >= 500:
            break
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        arm = record.get("arm")
        if arm is None:
            if not tolerate_missing_arm:
                raise SystemExit(
                    f"REFUSED: {path} carries no arm stamp. Every record in "
                    f"this root should be stamped {expected!r} (decision "
                    "record 2026-08-18_collection-consolidation); a missing "
                    "stamp indicates a foreign record."
                )
        elif arm != expected:
            raise SystemExit(
                f"REFUSED: {path} is stamped arm={arm!r} != {expected!r}; "
                "refusing to write into a directory with foreign records."
            )
        model = record.get("model")
        if model is not None and str(model) not in accepted:
            raise SystemExit(
                f"REFUSED: {path} stores model={model!r}, outside the "
                f"accepted set {sorted(accepted)} for {spec.arm_id}."
            )
    for var in ("AGENTS_METRICS_WORKER_KEY", "AGENTS_METRICS_API"):
        if os.environ.get(var):
            raise SystemExit(
                f"REFUSED: {var} is set. Collection must not run in a "
                "push-capable environment (the queue never dispatches runs)."
            )


# --------------------------------------------------------------------------- #
# Deadline
# --------------------------------------------------------------------------- #
class Deadline:
    """A local-time HH:MM stop, checked before each round and each run.

    ``clock`` is injectable for tests. A time already past at construction
    is an immediately reached deadline: the runner collects nothing.
    """

    def __init__(self, until: str | None, clock=time.time) -> None:
        self.clock = clock
        self.stamp: float | None = None
        if until is None:
            return
        try:
            hour, minute = (int(part) for part in until.split(":"))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError(until)
        except ValueError as exc:
            raise SystemExit(f"--until wants HH:MM, got {until!r}") from exc
        now = datetime.fromtimestamp(clock())
        self.stamp = now.replace(
            hour=hour, minute=minute, second=0, microsecond=0
        ).timestamp()

    def reached(self) -> bool:
        return self.stamp is not None and self.clock() >= self.stamp


# --------------------------------------------------------------------------- #
# Meter adapter
# --------------------------------------------------------------------------- #
def meter_percentages(
    spec: ad.ArmSpec, probes: dict[str, Any]
) -> tuple[float | None, float | None] | None:
    """(session, weekly) for the arm's subscription meter.

    ``None`` means the backend has no meter (OpenRouter runs on a credit
    ledger governed by ``--max-cost-usd``). ``(None, None)`` means the probe
    failed; the caller continues rather than blocking on a broken probe.
    """
    if spec.provider == "anthropic":
        probe = probes.get("claude")
        if probe is None:
            probe = harness.ClaudeUsageProbe(mode="claude-tui", timeout_seconds=40.0)
            probes["claude"] = probe
        snapshot = probe.snapshot()
        if snapshot is None:
            return (None, None)
        return (snapshot.current_session_percent, snapshot.weekly_all_percent)
    if spec.provider == "openai_codex":
        probe = probes.get("codex")
        if probe is None:
            probe = cells.CodexUsageProbe(model=spec.model_id)
            probes["codex"] = probe
        snapshot = probe.snapshot()
        if snapshot is None:
            return (None, None)
        return (None, snapshot.weekly_all_percent)
    return None


# --------------------------------------------------------------------------- #
# The collector
# --------------------------------------------------------------------------- #
class Collector:
    """Run the universe to the target, in shards, under one deadline."""

    def __init__(
        self,
        universe: list[Cell],
        *,
        reps_to: int,
        shards: int = 1,
        balanced_first: bool = True,
        deadline: Deadline | None = None,
        pace_seconds: float = 0.0,
        max_cost_usd: float | None = None,
        meter_wait_seconds: float = DEFAULT_METER_WAIT_SECONDS,
        sleep_func=time.sleep,
    ) -> None:
        self.universe = universe
        self.reps_to = reps_to
        self.shards = max(1, shards)
        self.balanced_first = balanced_first
        self.deadline = deadline or Deadline(None)
        self.pace_seconds = pace_seconds
        self.max_cost_usd = max_cost_usd
        self.meter_wait_seconds = meter_wait_seconds
        self.sleep = sleep_func
        self.exec_gates = {
            platform: threading.Semaphore(limit)
            for platform, limit in cells.EXEC_GATE_LIMITS.items()
        }
        self.swe_gates = {
            platform: threading.Semaphore(limit)
            for platform, limit in cells.SWE_RUN_GATE_LIMITS.items()
        }
        self.probes: dict[str, Any] = {}
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.cost_total = 0.0
        self.summary: list[dict[str, Any]] = []
        self.new_run_dirs: list[tuple[str, Path]] = []
        self.refused: set[Cell] = set()
        self.collected = 0

    # -- logging ----------------------------------------------------------- #
    @staticmethod
    def log(message: str) -> None:
        stamp = datetime.now().strftime("%F %T")
        print(f"[collect] {stamp} {message}", flush=True)

    # -- backends ---------------------------------------------------------- #
    def run_cell_backend(self, cell: Cell, rep: int) -> dict[str, Any]:
        spec = cell.spec
        task = cells.TASKS_BY_ID[cell.task_id]
        condition = cell.condition
        if spec.provider == "anthropic":
            harness.TIMEOUT_NO_FEEDBACK = CLAUDE_TIMEOUT_SECONDS
            harness.CLAUDE_EXTRA_CLI_ARGS = list(CLAUDE_EXTRA_CLI_ARGS)
            return cells.run_haiku_cell(
                task,
                cell.platform,
                spec.prompt_mode,
                condition,
                rep,
                overwrite=False,
                model=spec.model_id,
                runs_dir=spec.path,
            )
        if spec.provider == "openai_codex":
            return cells.run_codex_cell(
                task,
                cell.platform,
                spec.prompt_mode,
                condition,
                rep,
                spec.model_id,
                cells.DEFAULT_CODEX_REASONING,
                False,
                runs_dir=spec.path,
            )
        gates = self.swe_gates if condition == "or_agent_swe" else self.exec_gates
        return cells.run_openrouter_cell(
            task,
            cell.platform,
            spec.prompt_mode,
            condition,
            rep,
            spec.model_id,
            False,
            exec_gate=gates[cell.platform],
            runs_dir=spec.path,
        )

    # -- one wait-and-reprobe behavior for every backend ------------------- #
    def wait_for_meter(self, spec: ad.ArmSpec) -> bool:
        """Wait after a rejection until the meter clears. Never exits on a
        meter reading; returns False only when the deadline is reached."""
        while True:
            self.sleep(self.meter_wait_seconds)
            if self.deadline.reached() or self.stop.is_set():
                return False
            reading = meter_percentages(spec, self.probes)
            if reading is None:
                return True  # no subscription meter on this backend
            session, weekly = reading
            if session is None and weekly is None:
                self.log("meter probe failed; continuing")
                return True
            if session is not None and session >= SESSION_STOP_PERCENT:
                self.log(
                    f"session {session}% at or above {SESSION_STOP_PERCENT}%;"
                    " waiting for the reset"
                )
                continue
            if weekly is not None and weekly >= WEEKLY_STOP_PERCENT:
                self.log(
                    f"weekly {weekly}% at or above {WEEKLY_STOP_PERCENT}%;"
                    " waiting"
                )
                continue
            return True

    # -- identity ---------------------------------------------------------- #
    def prompt_identity_ok(self, cell: Cell, status: CellStatus) -> bool:
        """Refuse a cell whose stored prompt hash differs from the rebuilt
        one. Legacy records with no hash never block."""
        if cell in self.refused:
            return False
        if not status.prompt_hashes:
            return True
        expected = expected_prompt_sha256(cell)
        mismatched = status.prompt_hashes - {expected}
        if not mismatched:
            return True
        self.refused.add(cell)
        self.log(
            f"REFUSED {cell.label()}: stored prompt_sha256 "
            f"{sorted(mismatched)} != rebuilt {expected}; skipping this cell"
        )
        return False

    # -- bookkeeping ------------------------------------------------------- #
    def note_record(self, cell: Cell, rep: int, record: dict[str, Any]) -> None:
        summary = cells.cell_summary(record)
        run_dir = cell.spec.path / f"{cell.prefix}{rep:02d}"
        with self.lock:
            self.summary.append(summary)
            self.collected += 1
            self.new_run_dirs.append((cell.arm_id, run_dir))
            cost = summary.get("cost_usd")
            if isinstance(cost, (int, float)):
                self.cost_total += cost
            if (
                self.max_cost_usd is not None
                and self.cost_total >= self.max_cost_usd
            ):
                self.log(
                    f"STOP: --max-cost-usd {self.max_cost_usd} reached "
                    f"(measured {self.cost_total:.4f})"
                )
                self.stop.set()

    # -- filling ----------------------------------------------------------- #
    def fill_one(self, cell: Cell, status: CellStatus) -> None:
        """One collection in a cell; on a rejection, wait and retry the same
        cell at the following index rather than advancing to another cell."""
        rep = status.next_rep
        while True:
            if self.deadline.reached() or self.stop.is_set():
                return
            if rep > MAX_REP:
                self.log(
                    f"SKIP {cell.label()}: next rep {rep} above MAX_REP {MAX_REP}"
                )
                return
            try:
                record = self.run_cell_backend(cell, rep)
            except Exception as exc:  # noqa: BLE001 - keep the batch going
                self.log(
                    f"ERROR {cell.label()} rep{rep:02d}: "
                    f"{type(exc).__name__}: {exc}"
                )
                return
            self.note_record(cell, rep, record)
            if not record_is_rejection(cell.spec, record):
                return
            self.log(
                f"provider rejection stored at {cell.label()} rep{rep:02d}; "
                "waiting on the meter before retrying this cell"
            )
            rep += 1
            if not self.wait_for_meter(cell.spec):
                return

    def shard_loop(self, shard_id: int, shard: list[Cell]) -> None:
        for _ in range(MAX_ROUNDS):
            if self.deadline.reached() or self.stop.is_set():
                return
            statuses = {cell: cell_status(cell) for cell in shard}
            short = [
                (cell, status)
                for cell, status in statuses.items()
                if status.eligible < self.reps_to
                and cell not in self.refused
                and status.next_rep <= MAX_REP
            ]
            if not short:
                self.log(f"shard {shard_id}: complete at target {self.reps_to}")
                return
            if self.balanced_first:
                level = min(status.eligible for _, status in short)
                short = [
                    (cell, status)
                    for cell, status in short
                    if status.eligible == level
                ]
                self.log(
                    f"shard {shard_id}: filling level {level}, "
                    f"{len(short)} cells"
                )
            for cell, status in short:
                if self.deadline.reached() or self.stop.is_set():
                    return
                if not self.prompt_identity_ok(cell, status):
                    continue
                self.fill_one(cell, status)
                if self.pace_seconds > 0:
                    self.sleep(self.pace_seconds)
        self.log(f"shard {shard_id}: reached MAX_ROUNDS {MAX_ROUNDS}")

    # -- scoring post-pass -------------------------------------------------- #
    def score_new_corpus_runs(self) -> int:
        """Write ``score.json`` for every new run in a ``corpus_records``
        arm. Without this the zero-shot arm collects runs the analysis
        loader cannot read."""
        with self.lock:
            pending = [
                (arm_id, run_dir)
                for arm_id, run_dir in self.new_run_dirs
                if ad.ARMS[arm_id].kind == "corpus_records"
                and run_dir.is_dir()
                and not (run_dir / sp.RUN_SCORE_NAME).exists()
            ]
        if not pending:
            return 0
        references = sp.load_references()
        tasks_by_id = {task.task_id: task for task in sp.TASKS}
        written = 0
        for _, run_dir in pending:
            record_path = run_dir / RUN_RECORD_NAME
            if not record_path.exists():
                continue
            try:
                record = json.loads(record_path.read_text())
            except json.JSONDecodeError:
                continue
            score = sp.score_one_run(run_dir, record, tasks_by_id, references)
            sp.write_run_score(run_dir, score)
            written += 1
        self.log(f"scoring post-pass: wrote {written} score sidecars")
        return written

    # -- entry -------------------------------------------------------------- #
    def split_shards(self) -> list[list[Cell]]:
        """Round-robin over the emptiest-first ordering, so every shard gets
        the same mix of levels."""
        ordered = sorted(
            self.universe,
            key=lambda cell: (
                cell_status(cell).eligible,
                cell.arm_id,
                cell.task_id,
                cell.platform,
                cell.interface,
            ),
        )
        shards: list[list[Cell]] = [[] for _ in range(self.shards)]
        for index, cell in enumerate(ordered):
            shards[index % self.shards].append(cell)
        return [shard for shard in shards if shard]

    def run(self) -> int:
        if self.deadline.reached():
            self.log("the deadline is already reached; collecting nothing")
            return 0
        shards = self.split_shards()
        threads = [
            threading.Thread(
                target=self.shard_loop, args=(shard_id, shard), daemon=True
            )
            for shard_id, shard in enumerate(shards)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.score_new_corpus_runs()
        self.write_summaries()
        self.log(
            f"done: {self.collected} runs collected, "
            f"cost_usd={self.cost_total:.4f}, {len(self.refused)} cells refused"
        )
        return 0

    def write_summaries(self) -> None:
        with self.lock:
            rows_by_arm: dict[str, list[dict[str, Any]]] = {}
            arm_ids = {arm_id for arm_id, _ in self.new_run_dirs}
            for row, (arm_id, _) in zip(self.summary, self.new_run_dirs):
                rows_by_arm.setdefault(arm_id, []).append(row)
        for arm_id in sorted(arm_ids):
            harness.write_summary(
                rows_by_arm.get(arm_id, []),
                f"{harness.WORKING_DIRNAME}/collect_summary.json",
                runs_dir=ad.ARMS[arm_id].path,
            )


# --------------------------------------------------------------------------- #
# Universe, status, dry run
# --------------------------------------------------------------------------- #
def build_universe(
    specs: list[ad.ArmSpec],
    task_ids: list[str],
    platforms: list[str],
    interfaces: list[str],
) -> list[Cell]:
    return [
        Cell(spec.arm_id, task_id, platform, interface)
        for spec in specs
        for task_id in task_ids
        for platform in platforms
        for interface in interfaces
    ]


def status_rows(universe: list[Cell]) -> list[tuple[int, Cell, CellStatus]]:
    rows = [(cell_status(cell), cell) for cell in universe]
    return sorted(
        ((status.eligible, cell, status) for status, cell in rows),
        key=lambda row: (
            row[0],
            row[1].arm_id,
            row[1].task_id,
            row[1].platform,
            row[1].interface,
        ),
    )


def print_status(universe: list[Cell]) -> int:
    interface_cli = {value: key for key, value in AGENCY_CLI.items()}
    totals: dict[str, dict[str, int]] = {}
    for eligible, cell, status in status_rows(universe):
        print(
            f"{cell.arm_id} {cell.task_id} {cell.platform} "
            f"{interface_cli[cell.interface]} eligible={eligible} "
            f"next={status.next_rep} rejections={status.rejections} "
            f"missing_score={status.missing_score}"
        )
        arm_totals = totals.setdefault(
            cell.arm_id,
            {"cells": 0, "eligible": 0, "rejections": 0, "missing_score": 0},
        )
        arm_totals["cells"] += 1
        arm_totals["eligible"] += eligible
        arm_totals["rejections"] += status.rejections
        arm_totals["missing_score"] += status.missing_score
    for arm_id, arm_totals in sorted(totals.items()):
        print(
            f"# {arm_id}: {arm_totals['cells']} cells, "
            f"{arm_totals['eligible']} eligible runs, "
            f"{arm_totals['rejections']} rejections, "
            f"{arm_totals['missing_score']} records missing score.json"
        )
    return 0


def print_dry_run(
    specs: list[ad.ArmSpec], collector: Collector, reps_to: int
) -> int:
    for spec in specs:
        print(f"arm {spec.arm_id}: root={spec.path} model_id={spec.model_id}")
    shards = collector.split_shards()
    print(
        f"universe: {len(collector.universe)} cells, target {reps_to} "
        f"eligible per cell, {len(shards)} shard(s)"
    )
    for shard_id, shard in enumerate(shards):
        for cell in shard:
            status = cell_status(cell)
            if status.eligible >= reps_to:
                continue
            print(
                f"  DRY shard={shard_id} {cell.arm_id} "
                f"{cell.prefix}{status.next_rep:02d} "
                f"(eligible {status.eligible})"
            )
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help="arm family; the pinned model id comes from the ARMS table",
    )
    parser.add_argument(
        "--prompt",
        default=",".join(PROMPT_CHOICES),
        help="comma subset of zero_shot,few_shot (default: both roots)",
    )
    parser.add_argument(
        "--language",
        default=",".join(LANGUAGE_CHOICES),
        help="comma subset of python,r,stata",
    )
    parser.add_argument(
        "--agency",
        default=",".join(AGENCY_CLI),
        help="comma subset of chatbot,self_repair,constrained_agent",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="comma subset of the 21 analyzed tasks (default: all)",
    )
    parser.add_argument(
        "--allow-quarantined",
        action="store_true",
        help="permit a quarantined task named in --tasks",
    )
    parser.add_argument(
        "--reps-to",
        type=int,
        default=None,
        help="target eligible runs per cell (default: the ceiling)",
    )
    parser.add_argument(
        "--ceiling",
        type=int,
        default=DEFAULT_CEILING,
        help=f"per-cell cap (default {DEFAULT_CEILING})",
    )
    parser.add_argument(
        "--shards", type=int, default=1, help="disjoint cell slices, one thread each"
    )
    parser.add_argument(
        "--until", default=None, metavar="HH:MM", help="local-time hard stop"
    )
    parser.add_argument(
        "--balanced-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fill the emptiest cells first, level by level (default on)",
    )
    parser.add_argument(
        "--pace-seconds",
        type=float,
        default=0.0,
        help="sleep after each collected run",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="stop scheduling once summed run cost exceeds this",
    )
    parser.add_argument(
        "--meter-wait-seconds",
        type=float,
        default=DEFAULT_METER_WAIT_SECONDS,
        help="re-probe gap while a provider meter sits at its stop",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print arms, universe, shards, and first indexes; no calls",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="print the per-cell table, emptiest first; no calls",
    )
    return parser


def resolve_arms(
    parser: argparse.ArgumentParser, model: str, prompts: list[str]
) -> list[ad.ArmSpec]:
    specs = []
    for prompt in prompts:
        key = f"{model}_{prompt}"
        if key not in ad.ARMS:
            parser.error(f"no arm named {key!r} in the ARMS table")
        specs.append(ad.ARMS[key])
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    prompts = split_csv(args.prompt)
    if not prompts or any(prompt not in PROMPT_CHOICES for prompt in prompts):
        parser.error(f"--prompt wants a subset of {PROMPT_CHOICES}")
    platforms = split_csv(args.language)
    if not platforms or any(
        platform not in LANGUAGE_CHOICES for platform in platforms
    ):
        parser.error(f"--language wants a subset of {LANGUAGE_CHOICES}")
    agency = split_csv(args.agency)
    if not agency or any(name not in AGENCY_CLI for name in agency):
        parser.error(f"--agency wants a subset of {tuple(AGENCY_CLI)}")
    interfaces = [AGENCY_CLI[name] for name in agency]

    if args.tasks is None:
        task_ids = list(ad.ANALYSIS_TASK_IDS)
    else:
        task_ids = split_csv(args.tasks)
        unknown = [task for task in task_ids if task not in cells.TASKS_BY_ID]
        if unknown:
            parser.error(f"unknown task ids: {unknown}")
        quarantined = [
            task for task in task_ids if task in sp.QUARANTINED_TASKS
        ]
        if quarantined and not args.allow_quarantined:
            parser.error(
                f"quarantined tasks {quarantined} need --allow-quarantined"
            )

    if args.ceiling < 1:
        parser.error("--ceiling must be positive")
    reps_to = args.reps_to if args.reps_to is not None else args.ceiling
    if not 1 <= reps_to <= args.ceiling:
        parser.error(
            f"--reps-to must be between 1 and the ceiling ({args.ceiling})"
        )

    specs = resolve_arms(parser, args.model, prompts)
    universe = build_universe(specs, task_ids, platforms, interfaces)

    if args.status:
        return print_status(universe)

    validation = harness.validate_schedule(
        [
            (cells.TASKS_BY_ID[cell.task_id], cell.platform,
             cell.spec.prompt_mode, cell.interface)
            for cell in universe
        ],
        n_reps=1,
    )
    if not validation["ok"]:
        raise SystemExit(
            "schedule validation failed: "
            f"missing_data={validation['missing_data']} "
            f"missing_references={validation['missing_references']}"
        )

    deadline = Deadline(args.until)
    collector = Collector(
        universe,
        reps_to=reps_to,
        shards=args.shards,
        balanced_first=args.balanced_first,
        deadline=deadline,
        pace_seconds=args.pace_seconds,
        max_cost_usd=args.max_cost_usd,
        meter_wait_seconds=args.meter_wait_seconds,
    )

    if args.dry_run:
        return print_dry_run(specs, collector, reps_to)

    for spec in specs:
        assert_arm_root(spec)
        working = spec.path / harness.WORKING_DIRNAME
        working.mkdir(parents=True, exist_ok=True)
        (working / ".pilot_arm").write_text(arm_marker(spec))
        Collector.log(
            f"arm {spec.arm_id}: root={spec.path} model_id={spec.model_id}"
        )
    return collector.run()


if __name__ == "__main__":
    raise SystemExit(main())

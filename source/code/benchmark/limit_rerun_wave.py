"""Driver for the 2026-08-24 Sonnet limit re-run wave.

Decision record: ``quality_reports/decision_records/2026-08-24_sonnet-rerun-
replacements.md``. Report: ``quality_reports/reports/2026-08-24_limit-rerun-
wave/README.md``.

The wave repeats two classes of stored Claude Sonnet 4.6 runs without the
harness limit that stopped them. Selection lives in ``limit_rerun_select.py``.
Everything else about the protocol is the collected protocol: the same prompt,
the same pinned model ``claude-sonnet-4-6``, the same permission mode, the same
tool surface, no extra CLI arguments, and no thinking flag (the corpus passes
none, so the CLI default applies).

Two overrides, and only two:

``TOOL_LOOP_TURN_CAP_ENABLED = False``
    The constrained-agent call passes no ``--max-turns``, so the session ends
    when the model submits or when the 600-second session wall stops it. The
    wall is unchanged. This reaches only ``tool_loop`` runs: the chatbot and
    self-repair calls pass ``max_turns=1`` literally.

``MODEL_CALL_TIMEOUT_OVERRIDE = 3600``
    The chatbot and self-repair model calls get a 3600-second bound instead of
    180. Three thousand six hundred seconds is a chosen safety bound, not a
    measured quantity: ``subprocess.run`` needs some bound to avoid an
    unbounded hang, and this one is twenty times the original and far above any
    observed call. It reaches only those two conditions: the constrained-agent
    call passes ``TIMEOUT_TOOL`` (600) explicitly, and script execution keeps
    ``TIMEOUT_NO_FEEDBACK`` (180), so no generated script gets longer to run
    than it had when the corpus was collected.

Because the two overrides reach disjoint conditions, both are set once for the
whole wave and no shard ever changes them, which keeps two shards free of a
configuration race.

Writing rules:

- every re-run writes to
  ``data/runs/<arm>/_working/limit_rerun_2026-08-24/<original_run_dir_name>/``;
- the driver refuses any target outside that path, so no original run
  directory is read-modified or deleted;
- ``_working`` is one level below the arm root, and every analysis loader
  globs the arm root one level deep, so the wave is invisible to the corpus
  until the swap moves it.

Pacing: the runner is serial by default and takes at most two shards. On a
provider rejection it stores the stamped record, waits, re-probes the
subscription meter, and repeats the same run. It never exits on a meter
reading and never stops at a percentage of usage.

Usage::

    uv run python code/benchmark/limit_rerun_wave.py --plan
    uv run python code/benchmark/limit_rerun_wave.py --pilot turn_cap
    uv run python code/benchmark/limit_rerun_wave.py --shards 2
    uv run python code/benchmark/limit_rerun_wave.py --status
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_dataset as ad  # noqa: E402
import cells  # noqa: E402
import collect  # noqa: E402
import harness  # noqa: E402
import limit_rerun_select as sel  # noqa: E402
import strict_posthoc as sp  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WAVE_ID = "limit_rerun_2026-08-24"
REPORT_DIR = (
    PROJECT_ROOT / "quality_reports" / "reports" / "2026-08-24_limit-rerun-wave"
)
DEFAULT_LIST = REPORT_DIR / "rerun_list.json"
DEFAULT_STATUS = REPORT_DIR / "status.json"
DEFAULT_PROGRESS = REPORT_DIR / "progress.jsonl"

#: Chosen safety bound on a model call with the 180-second timeout lifted.
#: Not a measured quantity; see the module docstring.
RERUN_MODEL_CALL_TIMEOUT_SECONDS = 3600
#: The collected script-execution bound, kept unchanged for the re-runs.
SCRIPT_TIMEOUT_SECONDS = 180
MAX_SHARDS = 2
#: Guard against an unbounded rejection loop on one run.
MAX_REJECTION_RETRIES = 40


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def target_root(arm_id: str) -> Path:
    return ad.ARMS[arm_id].path / harness.WORKING_DIRNAME / WAVE_ID


def assert_writable_target(arm_id: str, run_dir: Path) -> None:
    """Refuse any path outside the wave's working directory for this arm."""
    root = target_root(arm_id).resolve()
    resolved = run_dir.resolve()
    if resolved.parent != root:
        raise SystemExit(
            f"REFUSED: {resolved} is not directly inside {root}; the wave "
            "never writes into an original run directory."
        )


def assert_environment() -> None:
    for var in ("AGENTS_METRICS_WORKER_KEY", "AGENTS_METRICS_API"):
        if os.environ.get(var):
            raise SystemExit(
                f"REFUSED: {var} is set. Collection must not run in a "
                "push-capable environment."
            )


def apply_overrides() -> dict[str, Any]:
    """Set the wave configuration once, for every shard, and describe it."""
    harness.TOOL_LOOP_TURN_CAP_ENABLED = False
    harness.MODEL_CALL_TIMEOUT_OVERRIDE = RERUN_MODEL_CALL_TIMEOUT_SECONDS
    harness.TIMEOUT_NO_FEEDBACK = SCRIPT_TIMEOUT_SECONDS
    harness.CLAUDE_EXTRA_CLI_ARGS = []
    return {
        "wave_id": WAVE_ID,
        "tool_loop_turn_cap": "removed (--max-turns not passed)",
        "tool_loop_session_wall_seconds": harness.TIMEOUT_TOOL,
        "model_call_timeout_seconds": RERUN_MODEL_CALL_TIMEOUT_SECONDS,
        "model_call_timeout_note": (
            "chosen safety bound, not a measured limit; replaces the "
            "180-second bound on the chatbot and self-repair model calls"
        ),
        "script_execution_timeout_seconds": SCRIPT_TIMEOUT_SECONDS,
        "claude_extra_cli_args": [],
        "model_pin": ad.ARMS["sonnet_zero_shot"].model_id,
    }


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
class Status:
    """A status file the parent session can poll while the wave runs."""

    def __init__(self, path: Path, progress: Path, total: int, config: dict) -> None:
        self.path = path
        self.progress = progress
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {
            "wave_id": WAVE_ID,
            "started_at": now_iso(),
            "updated_at": now_iso(),
            "pid": os.getpid(),
            "config": config,
            "total": total,
            "done": 0,
            "skipped_existing": 0,
            "rejections": 0,
            "errors": 0,
            "waiting_on_meter": False,
            "by_class": {},
            "last": None,
            "finished_at": None,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush()

    def flush(self) -> None:
        self.state["updated_at"] = now_iso()
        self.path.write_text(json.dumps(self.state, indent=2) + "\n")

    def note(self, **fields: Any) -> None:
        with self.lock:
            for key, value in fields.items():
                if isinstance(value, int) and key in {
                    "done",
                    "skipped_existing",
                    "rejections",
                    "errors",
                }:
                    self.state[key] += value
                else:
                    self.state[key] = value
            self.flush()

    def append(self, row: dict[str, Any]) -> None:
        with self.lock:
            row = {"at": now_iso(), **row}
            with self.progress.open("a") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
            rerun_class = str(row.get("rerun_class") or "unknown")
            bucket = self.state["by_class"].setdefault(
                rerun_class, {"done": 0, "rejections": 0, "errors": 0}
            )
            outcome = str(row.get("outcome") or "")
            if outcome in bucket:
                bucket[outcome] += 1
            elif outcome == "collected":
                bucket["done"] += 1
            self.state["last"] = row
            self.flush()


# --------------------------------------------------------------------------- #
# Meter: one wait-and-reprobe behavior, shared with collect.py
# --------------------------------------------------------------------------- #
class Meter:
    def __init__(self, wait_seconds: float, status: Status | None = None) -> None:
        self.wait_seconds = wait_seconds
        self.status = status
        self.probes: dict[str, Any] = {}
        self.lock = threading.Lock()

    def wait(self, spec: ad.ArmSpec, stop: threading.Event) -> bool:
        """Wait until the meter clears. Never exits on a meter reading."""
        if self.status:
            self.status.note(waiting_on_meter=True)
        try:
            while not stop.is_set():
                time.sleep(self.wait_seconds)
                if stop.is_set():
                    return False
                with self.lock:
                    reading = collect.meter_percentages(spec, self.probes)
                if reading is None:
                    return True
                session, weekly = reading
                if session is None and weekly is None:
                    print("[wave] meter probe failed; continuing", flush=True)
                    return True
                if session is not None and session >= collect.SESSION_STOP_PERCENT:
                    print(
                        f"[wave] session {session}% at or above "
                        f"{collect.SESSION_STOP_PERCENT}%; waiting for the reset",
                        flush=True,
                    )
                    continue
                if weekly is not None and weekly >= collect.WEEKLY_STOP_PERCENT:
                    print(
                        f"[wave] weekly {weekly}% at or above "
                        f"{collect.WEEKLY_STOP_PERCENT}%; waiting",
                        flush=True,
                    )
                    continue
                return True
            return False
        finally:
            if self.status:
                self.status.note(waiting_on_meter=False)


# --------------------------------------------------------------------------- #
# One re-run
# --------------------------------------------------------------------------- #
def stamp_wave_provenance(record: dict[str, Any], item: dict[str, Any]) -> None:
    """Record what this run overrode, and what it replaces.

    ``run_haiku_cell`` stamps ``claude_timeout_seconds`` from
    ``TIMEOUT_NO_FEEDBACK``, which under the wave configuration is the
    script-execution bound rather than the model-call bound. Restate it as the
    model-call bound, which is what the field has always meant, and keep the
    script bound inside the ``limit_rerun`` block.
    """
    call_timeout = (
        harness.TIMEOUT_TOOL
        if item["exec_mode"] == "tool_loop"
        else harness.model_call_timeout()
    )
    record["claude_timeout_seconds"] = call_timeout
    record["limit_rerun"] = {
        "wave_id": WAVE_ID,
        "rerun_class": item["rerun_class"],
        "replaces_run_dir": f"data/runs/{item['arm_id']}/{item['run_dir']}",
        "original_marker": item["marker"],
        "original_in_paper_sample": item["in_paper_sample"],
        "tool_loop_turn_cap_enabled": harness.TOOL_LOOP_TURN_CAP_ENABLED,
        "model_call_timeout_seconds": call_timeout,
        "tool_loop_session_wall_seconds": harness.TIMEOUT_TOOL,
        "script_execution_timeout_seconds": harness.TIMEOUT_NO_FEEDBACK,
        "decision_record": (
            "quality_reports/decision_records/"
            "2026-08-24_sonnet-rerun-replacements.md"
        ),
    }


def score_sidecar(run_dir: Path, record: dict[str, Any]) -> None:
    references = sp.load_references()
    tasks_by_id = {task.task_id: task for task in sp.TASKS}
    score = sp.score_one_run(run_dir, record, tasks_by_id, references)
    sp.write_run_score(run_dir, score)


def run_one(
    item: dict[str, Any],
    status: Status,
    meter: Meter,
    stop: threading.Event,
    overwrite: bool,
) -> dict[str, Any]:
    arm_id = item["arm_id"]
    spec = ad.ARMS[arm_id]
    root = target_root(arm_id)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / item["run_dir"]
    assert_writable_target(arm_id, run_dir)

    if (run_dir / "run_record.json").exists() and not overwrite:
        stored = json.loads((run_dir / "run_record.json").read_text())
        if not collect.record_is_rejection(spec, stored):
            status.note(skipped_existing=1)
            status.append(
                {
                    "run_dir": item["run_dir"],
                    "arm_id": arm_id,
                    "rerun_class": item["rerun_class"],
                    "outcome": "skipped_existing",
                }
            )
            return stored

    task = cells.TASKS_BY_ID[item["task_id"]]
    record: dict[str, Any] = {}
    for attempt in range(1, MAX_REJECTION_RETRIES + 1):
        if stop.is_set():
            return record
        started = time.time()
        record = cells.run_haiku_cell(
            task,
            item["platform"],
            item["prompt_mode"],
            item["exec_mode"],
            item["rep"],
            overwrite=overwrite and attempt == 1,
            model=spec.model_id,
            rerun_provider_limit=True,
            runs_dir=root,
        )
        stamp_wave_provenance(record, item)
        harness.write_run_record(record, runs_dir=root)
        wall = round(time.time() - started, 1)
        if collect.record_is_rejection(spec, record):
            status.note(rejections=1)
            status.append(
                {
                    "run_dir": item["run_dir"],
                    "arm_id": arm_id,
                    "rerun_class": item["rerun_class"],
                    "outcome": "rejections",
                    "attempt": attempt,
                    "wall_seconds": wall,
                }
            )
            print(
                f"[wave] provider rejection on {item['run_dir']} "
                f"(attempt {attempt}); waiting on the meter",
                flush=True,
            )
            if not meter.wait(spec, stop):
                return record
            continue
        try:
            score_sidecar(run_dir, record)
            scored = True
        except Exception as exc:  # noqa: BLE001 - a scoring fault must not stop the wave
            scored = False
            print(f"[wave] scoring failed for {item['run_dir']}: {exc}", flush=True)
        status.note(done=1)
        status.append(
            {
                "run_dir": item["run_dir"],
                "arm_id": arm_id,
                "rerun_class": item["rerun_class"],
                "outcome": "collected",
                "attempt": attempt,
                "wall_seconds": wall,
                "execution_success": record.get("execution_success"),
                "scored": scored,
            }
        )
        return record
    return record


# --------------------------------------------------------------------------- #
# Wave
# --------------------------------------------------------------------------- #
def load_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(
            f"run list not found: {path}. Build it with "
            "'uv run python code/benchmark/limit_rerun_select.py "
            f"{path}'."
        )
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"run list {path} is empty or malformed")
    return rows


def split_shards(items: list[dict[str, Any]], shards: int) -> list[list[dict]]:
    buckets: list[list[dict]] = [[] for _ in range(shards)]
    for index, item in enumerate(items):
        buckets[index % shards].append(item)
    return [bucket for bucket in buckets if bucket]


def run_wave(
    items: list[dict[str, Any]],
    shards: int,
    meter_wait_seconds: float,
    overwrite: bool,
    status_path: Path,
    progress_path: Path,
) -> int:
    assert_environment()
    config = apply_overrides()
    status = Status(status_path, progress_path, len(items), config)
    meter = Meter(meter_wait_seconds, status)
    stop = threading.Event()
    print(f"[wave] {len(items)} runs, {shards} shard(s), config={config}", flush=True)

    def shard_loop(shard: list[dict[str, Any]]) -> None:
        for item in shard:
            if stop.is_set():
                return
            try:
                run_one(item, status, meter, stop, overwrite)
            except Exception as exc:  # noqa: BLE001 - keep the wave going
                status.note(errors=1)
                status.append(
                    {
                        "run_dir": item["run_dir"],
                        "arm_id": item["arm_id"],
                        "rerun_class": item["rerun_class"],
                        "outcome": "errors",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"[wave] ERROR {item['run_dir']}: {exc}", flush=True)

    threads = [
        threading.Thread(target=shard_loop, args=(shard,), daemon=True)
        for shard in split_shards(items, shards)
    ]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        stop.set()
        for thread in threads:
            thread.join(timeout=30)
    status.note(finished_at=now_iso())
    print(
        f"[wave] done: {status.state['done']} collected, "
        f"{status.state['skipped_existing']} already present, "
        f"{status.state['rejections']} rejections, "
        f"{status.state['errors']} errors",
        flush=True,
    )
    return 0


def print_plan(items: list[dict[str, Any]]) -> int:
    by_class: dict[str, int] = {}
    for item in items:
        by_class[item["rerun_class"]] = by_class.get(item["rerun_class"], 0) + 1
        print(
            f"  {item['rerun_class']:<13} {item['arm_id']} {item['run_dir']} "
            f"-> {target_root(item['arm_id'])}/{item['run_dir']}"
        )
    for name in sorted(by_class):
        print(f"{name}: {by_class[name]}")
    print(f"total: {len(items)}")
    return 0


def print_status(status_path: Path) -> int:
    if not status_path.is_file():
        print(f"no status file at {status_path}; the wave has not started")
        return 1
    state = json.loads(status_path.read_text())
    done = state["done"] + state["skipped_existing"]
    print(
        f"{WAVE_ID}: {done}/{state['total']} runs "
        f"({state['done']} collected, {state['skipped_existing']} already "
        f"present), {state['rejections']} rejections, {state['errors']} errors"
    )
    print(f"  started {state['started_at']}  updated {state['updated_at']}")
    print(f"  waiting_on_meter={state['waiting_on_meter']}")
    for name, bucket in sorted(state.get("by_class", {}).items()):
        print(f"  {name}: {bucket}")
    if state.get("last"):
        print(f"  last: {state['last']}")
    if state.get("finished_at"):
        print(f"  finished {state['finished_at']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="limit_rerun_wave.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", type=Path, default=DEFAULT_LIST)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--progress-file", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument(
        "--shards", type=int, default=1, help=f"1 or {MAX_SHARDS} (default 1)"
    )
    parser.add_argument(
        "--only",
        default=None,
        choices=(sel.CLASS_TURN_CAP, sel.CLASS_CALL_TIMEOUT),
        help="restrict the wave to one class",
    )
    parser.add_argument(
        "--pilot",
        default=None,
        choices=(sel.CLASS_TURN_CAP, sel.CLASS_CALL_TIMEOUT),
        help="run exactly one run of this class and stop",
    )
    parser.add_argument(
        "--pilot-run-dir",
        default=None,
        help="with --pilot, name the run directory to repeat",
    )
    parser.add_argument(
        "--meter-wait-seconds",
        type=float,
        default=collect.DEFAULT_METER_WAIT_SECONDS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="repeat a run that already has a record in the wave directory",
    )
    parser.add_argument("--plan", action="store_true", help="print the wave; no calls")
    parser.add_argument("--status", action="store_true", help="print the status file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.status:
        return print_status(args.status_file)

    items = load_list(args.list)
    if args.only:
        items = [item for item in items if item["rerun_class"] == args.only]
    if args.pilot:
        candidates = [item for item in items if item["rerun_class"] == args.pilot]
        if args.pilot_run_dir:
            candidates = [
                item for item in candidates if item["run_dir"] == args.pilot_run_dir
            ]
        if not candidates:
            raise SystemExit(f"no run matches --pilot {args.pilot}")
        items = candidates[:1]
    if not items:
        raise SystemExit("nothing to run")

    if args.plan:
        return print_plan(items)

    shards = max(1, min(MAX_SHARDS, args.shards))
    return run_wave(
        items,
        shards,
        args.meter_wait_seconds,
        args.overwrite,
        args.status_file,
        args.progress_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())

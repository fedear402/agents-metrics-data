"""Queue-driven worker loop: claim a cell from the central queue, run it with
the harness, push the record, repeat.

This is the distributed-worker entry point (one process per machine). It uses
``harness.run_cell`` as a library — the harness itself is unchanged — and the
central API at https://agents-metrics.falstuff.com for coordination. Records
are always written LOCALLY first (the runs dir stays the primary dataset),
then pushed; the server dedupes by uuid, so crashes never lose or double-count
work. Provider-limit failures release the slot back to the queue (no local
record is written) and the loop sleeps through the cooldown.

Never run this concurrently with ``run_balanced_sweep.sh`` on the same
machine: both call Claude under the same account and keep separate schedules.

Usage:
    python3 code/benchmark/worker_loop.py --runs-dir data/generated/2026-05-19_full_sweep/runs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import harness  # noqa: E402
import preflight  # noqa: E402
from push_records import build_item  # noqa: E402
from queue_client import QueueClient, QueueError  # noqa: E402
from tasks import EXEC_MODES, PLATFORMS, PROMPT_MODES, TASKS  # noqa: E402

# Pinned model: the sweep corpus is claude-sonnet-4-6. Aliases float across
# releases (the bare "sonnet" alias resolved to claude-sonnet-5 on 2026-07-04)
# and are therefore forbidden here.
PINNED_MODEL = "claude-sonnet-4-6"
FORBIDDEN_MODEL_ALIASES = {"sonnet", "opus", "haiku", "default"}

# Conservative pacing block — same values as run_balanced_sweep.sh /
# run_topup_both.sh (2026-06-26 playbook section 4).
PACING_DEFAULTS = dict(
    window_seconds=300.0 * 60.0,
    target_runs=120,
    reserve_fraction=0.40,
    min_delay_seconds=30.0,
    max_sleep_seconds=3600.0,
    session_soft_limit_percent=35.0,
    session_hard_limit_percent=50.0,
    session_reserve_percent=10.0,
    weekly_all_hard_limit_percent=60.0,
    weekly_sonnet_hard_limit_percent=60.0,
    default_session_percent_per_run=0.5,
    external_usage_tolerance_percent_per_hour=3.0,
    fill_current_session=True,
)
PROVIDER_LIMIT_SLEEP_SECONDS = 3600.0
EMPTY_QUEUE_SLEEP_SECONDS = 600.0


def task_by_id(task_id: str):
    for task in TASKS:
        if task.task_id == task_id:
            return task
    return None


def build_pacing(runs_dir: Path, args: argparse.Namespace) -> harness.PacingController:
    probe_mode = args.usage_probe
    if probe_mode == "auto":
        # Prefer the cross-platform OAuth endpoint probe; keep the interval at
        # or above 300s (see usage_probe_oauth.py header).
        probe_mode = "command"
    probe = None
    if probe_mode == "command":
        command = args.usage_command or (
            f"{sys.executable} {ROOT / 'usage_probe_oauth.py'}"
        )
        probe = harness.ClaudeUsageProbe(
            mode="command",
            command=command,
            interval_seconds=max(300.0, args.usage_probe_interval_seconds),
            timeout_seconds=45.0,
        )
    elif probe_mode == "claude-tui":
        if harness.pty is None:
            raise SystemExit("claude-tui probe needs a pty; use --usage-probe auto")
        probe = harness.ClaudeUsageProbe(
            mode="claude-tui",
            interval_seconds=args.usage_probe_interval_seconds,
            timeout_seconds=30.0,
        )
    overrides = dict(
        target_runs=args.pacing_target_runs_per_5h,
        reserve_fraction=args.pacing_reserve_fraction,
        min_delay_seconds=args.pacing_min_delay_seconds,
        session_soft_limit_percent=args.pacing_session_soft_limit_percent,
        session_hard_limit_percent=args.pacing_session_hard_limit_percent,
        session_reserve_percent=args.pacing_session_reserve_percent,
        weekly_all_hard_limit_percent=args.pacing_weekly_all_hard_limit_percent,
        weekly_sonnet_hard_limit_percent=args.pacing_weekly_sonnet_hard_limit_percent,
        weekly_fill_target_percent=args.pacing_weekly_fill_target_percent,
    )
    params = {**PACING_DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    if args.no_session_fill:
        params["fill_current_session"] = False
    return harness.PacingController(
        # Collection state stays out of the arm root, which holds run
        # directories only.
        state_file=runs_dir / harness.WORKING_DIRNAME / "pacing_state.json",
        usage_probe=probe,
        **params,
    )


def usage_snapshot_payload(pacing: harness.PacingController) -> dict[str, Any] | None:
    probe = getattr(pacing, "usage_probe", None)
    if probe is None:
        return None
    snap = probe.snapshot()
    if snap is None:
        return None
    return {
        "session_pct": snap.current_session_percent,
        "weekly_pct": snap.weekly_all_percent,
        "weekly_sonnet_pct": snap.weekly_sonnet_percent,
    }


def run_one(
    client: QueueClient, cell: dict[str, Any], runs_dir: Path
) -> tuple[str, dict[str, Any] | None]:
    """Run one leased queue cell. Returns (outcome, record); outcome is
    'pushed' or 'released'."""
    task = task_by_id(cell["task_id"])
    if task is None:
        client.release(cell["queue_id"], reason=f"unknown task {cell['task_id']}")
        return "released", None
    # Validate server-supplied factors before they reach a subprocess/path
    # (audit fix 9): a bad row must not crash the worker or escape the runs dir.
    if (cell["platform"] not in PLATFORMS or cell["prompt_mode"] not in PROMPT_MODES
            or cell["exec_mode"] not in EXEC_MODES):
        client.release(cell["queue_id"], reason="invalid factor value from queue")
        return "released", None
    record = harness.run_cell(
        task,
        cell["platform"],
        cell["prompt_mode"],
        cell["exec_mode"],
        rep=int(cell["rep"]),
    )
    if record.get("skipped_existing"):
        # A local record already exists (e.g. collected by the standalone
        # sweep); push it so the server marks the slot saved.
        run_dir = runs_dir / record["cell_id"]
        client.push_records([build_item(run_dir, record)])
        return "pushed", None  # no new usage event
    if harness.is_provider_limit_record(record):
        # Do not persist or push: leave the cell clean so it requeues.
        client.release(cell["queue_id"], reason="provider-limit")
        return "released", record
    harness.write_run_record(record)
    run_dir = runs_dir / record["cell_id"]
    try:
        client.push_records([build_item(run_dir, record)])
    except QueueError:
        # The record is written locally (push_records.py --all recovers it);
        # release the lease so the slot requeues cleanly instead of expiring
        # into a possible double-run (audit fix 7). Re-raise for the caller's
        # error accounting.
        try:
            client.release(cell["queue_id"], reason="push failed after local write")
        except QueueError:
            pass
        raise
    return "pushed", record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--model", default=PINNED_MODEL)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--once", action="store_true", help="claim and run one cell, then exit")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the strict environment preflight (testing only)",
    )
    parser.add_argument(
        "--usage-probe",
        choices=["auto", "command", "claude-tui", "off"],
        default="auto",
        help="auto = OAuth endpoint probe (cross-platform)",
    )
    parser.add_argument("--usage-command", default=None)
    parser.add_argument("--usage-probe-interval-seconds", type=float, default=300.0)
    parser.add_argument("--pacing-target-runs-per-5h", type=int, default=None)
    parser.add_argument("--pacing-reserve-fraction", type=float, default=None)
    parser.add_argument("--pacing-min-delay-seconds", type=float, default=None)
    parser.add_argument("--pacing-session-soft-limit-percent", type=float, default=None)
    parser.add_argument("--pacing-session-hard-limit-percent", type=float, default=None)
    parser.add_argument("--pacing-session-reserve-percent", type=float, default=None)
    parser.add_argument("--pacing-weekly-all-hard-limit-percent", type=float, default=None)
    parser.add_argument(
        "--pacing-weekly-sonnet-hard-limit-percent", type=float, default=None
    )
    parser.add_argument(
        "--pacing-weekly-fill-target-percent",
        type=float,
        default=None,
        help=(
            "adaptively space runs so weekly usage reaches this percent "
            "exactly at the weekly reset (requires a usage probe)"
        ),
    )
    parser.add_argument(
        "--no-session-fill",
        action="store_true",
        help=(
            "disable fill-before-reset (and its external-usage projection, "
            "which stalls collection whenever the owner uses Claude); the "
            "run-count spacing and session/weekly hard caps still govern"
        ),
    )
    args = parser.parse_args(argv)

    if args.model in FORBIDDEN_MODEL_ALIASES:
        parser.error(
            f"model alias '{args.model}' is forbidden; pin an exact id "
            f"(the sweep corpus is {PINNED_MODEL})"
        )

    runs_dir = args.runs_dir.expanduser().resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    harness.MODEL = args.model
    harness.RUNS_DIR = runs_dir

    client = QueueClient()

    if not args.skip_preflight:
        checks = preflight.run_checks()
        failures = preflight.strict_failures(checks)
        if failures:
            for failure in failures:
                print(f"PREFLIGHT FAIL: {failure}", file=sys.stderr)
            return 1
        client.preflight(preflight.build_payload(checks))

    pacing = build_pacing(runs_dir, args)
    completed = 0
    while args.max_runs is None or completed < args.max_runs:
        # Pace BEFORE claiming: a claim held through a long pacing sleep would
        # outlive its 45-minute lease (observed 2026-07-04). The pacer decides
        # when capacity exists; the claim is made at execution time.
        pacing.before_run("next-claim")
        try:
            resp = client.claim(max_cells=1, usage=usage_snapshot_payload(pacing))
        except QueueError as err:
            print(f"claim failed: {err}; sleeping {EMPTY_QUEUE_SLEEP_SECONDS}s", file=sys.stderr)
            if args.once:
                return 1
            time.sleep(EMPTY_QUEUE_SLEEP_SECONDS)
            continue
        cells = resp.get("cells") or []
        if not cells:
            retry_after = float(resp.get("retry_after_seconds") or EMPTY_QUEUE_SLEEP_SECONDS)
            print(f"queue empty or throttled; sleeping {retry_after:.0f}s")
            if args.once:
                return 0
            time.sleep(retry_after)
            continue
        cell = cells[0]
        cell_label = (
            f"{cell['task_id']}__{cell['platform']}__{cell['prompt_mode']}"
            f"__{cell['exec_mode']}__rep{int(cell['rep']):02d}"
        )
        try:
            outcome, record = run_one(client, cell, runs_dir)
        except QueueError as err:
            print(f"push failed for {cell_label}: {err}", file=sys.stderr)
            outcome, record = "error", None
        if record is not None:
            pacing.record(record)
        if outcome == "pushed":
            completed += 1
            print(f"[{completed}] {cell_label} pushed", flush=True)
        elif outcome == "released":
            print(f"{cell_label} released (provider limit)", flush=True)
        if args.once:
            return 0
        if outcome == "released":
            time.sleep(PROVIDER_LIMIT_SLEEP_SECONDS)
    print(f"done: {completed} runs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

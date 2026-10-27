"""Bulk-upload local run records to the central queue DB.

Scans ``<runs-dir>/*/run_record.json``, attaches the generated script and
parsed ``result.json`` when present, and pushes everything not yet
acknowledged via ``QueueClient.push_records``. Acknowledged uuids (server
inserted OR already present) are tracked in a local state file so re-runs are
incremental; the server is idempotent by uuid, so ``--all`` re-pushes are
always safe. This is also the migration tool for the existing record corpus.

Usage:
    python code/benchmark/push_records.py --runs-dir data/generated/benchmark_runs
    python code/benchmark/push_records.py --runs-dir ... --dry-run
    python code/benchmark/push_records.py --runs-dir ... --all
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from queue_client import QueueClient, QueueError, chunked  # noqa: E402

SCRIPT_NAMES = {"python": "run.py", "r": "run.R", "stata": "run.do"}
SCRIPT_EXTS = {"python": ".py", "r": ".R", "stata": ".do"}
CHUNK_SIZE = 100


def attempt_number(path: Path) -> int:
    """Attempt index of a self_repair script file (attemptN_run.*), else -1."""
    match = re.search(r"attempt(\d+)_run", path.name)
    return int(match.group(1)) if match else -1


def find_script(run_dir: Path, platform: str | None) -> Path | None:
    """Locate the generated script for a run directory.

    self_repair cells write ``attemptN_run.<ext>`` files; the LAST attempt is
    the script of record. Other modes write the canonical run.py/run.R/run.do.
    """
    if platform in SCRIPT_EXTS:
        candidates = {platform: SCRIPT_EXTS[platform]}
    else:
        candidates = SCRIPT_EXTS
    for key, ext in candidates.items():
        attempts = sorted(run_dir.glob(f"attempt*_run{ext}"), key=attempt_number)
        if attempts:
            return attempts[-1]
        canonical = run_dir / SCRIPT_NAMES[key]
        if canonical.exists():
            return canonical
    return None


def json_safe(value: Any) -> Any:
    """Replace non-finite floats with None (mirrors harness.json_safe).

    Some generated result.json files contain literal NaN/Infinity tokens —
    Python's json module tolerates them, but strict JSON consumers (Postgres
    jsonb) reject them.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(val) for val in value]
    return value


def build_item(run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Assemble one /v1/records item: record + script + parsed result.json."""
    item: dict[str, Any] = {"record": json_safe(record)}
    script_path = find_script(run_dir, record.get("platform"))
    if script_path is not None:
        try:
            item["script"] = script_path.read_text(errors="replace")
            item["script_name"] = script_path.name
        except OSError:
            item.pop("script", None)
    result_path = run_dir / "result.json"
    if result_path.exists():
        try:
            item["result_json"] = json_safe(
                json.loads(result_path.read_text(errors="replace"))
            )
        except (OSError, json.JSONDecodeError):
            pass  # unreadable/invalid result.json: push the record without it
    return item


def load_state(state_file: Path) -> set[str]:
    """Read the set of already-acknowledged uuids (missing file = empty)."""
    try:
        data = json.loads(state_file.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    if isinstance(data, list):
        return {str(u) for u in data}
    if isinstance(data, dict):
        return {str(u) for u in data.get("uuids", [])}
    return set()


def save_state(state_file: Path, uuids: set[str]) -> None:
    """Atomically persist acknowledged uuids (crash-safe incremental writes)."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_name(state_file.name + ".tmp")
    tmp.write_text(json.dumps(sorted(uuids), indent=0) + "\n")
    tmp.replace(state_file)


def sync_records(
    runs_dir: Path,
    state_file: Path,
    push_all: bool = False,
    dry_run: bool = False,
    client: Any | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, int]:
    """Push every unacknowledged run record under ``runs_dir``.

    Returns summary counts: scanned, pushed_new, already_present,
    skipped_no_uuid, errors (plus would_push under --dry-run).
    """
    state = load_state(state_file)
    summary = {
        "scanned": 0,
        "pushed_new": 0,
        "already_present": 0,
        "skipped_no_uuid": 0,
        "errors": 0,
    }
    pending: list[tuple[str, dict[str, Any]]] = []
    for record_path in sorted(runs_dir.glob("*/run_record.json")):
        run_dir = record_path.parent
        summary["scanned"] += 1
        try:
            record = json.loads(record_path.read_text())
        except (OSError, json.JSONDecodeError) as err:
            print(f"WARNING unreadable record {record_path}: {err}", file=sys.stderr)
            summary["errors"] += 1
            continue
        run_uuid = record.get("uuid") if isinstance(record, dict) else None
        if not run_uuid:
            # Known: exactly one legacy harness-exception record lacks a uuid.
            print(f"WARNING record without uuid skipped: {record_path}", file=sys.stderr)
            summary["skipped_no_uuid"] += 1
            continue
        if not push_all and str(run_uuid) in state:
            summary["already_present"] += 1
            continue
        pending.append((str(run_uuid), build_item(run_dir, record)))

    if dry_run:
        summary["would_push"] = len(pending)
        return summary

    if client is None:
        client = QueueClient()
    for chunk in chunked(pending, chunk_size):
        items = [item for _, item in chunk]
        try:
            resp = client.push_records(items)
        except QueueError as err:
            print(
                f"WARNING push failed for chunk of {len(chunk)}: {err}",
                file=sys.stderr,
            )
            summary["errors"] += len(chunk)
            continue
        # The server reports per-uuid statuses; acknowledge inserted/exists,
        # leave per-item errors unacknowledged so a re-run retries them.
        statuses: dict[str, str] = {}
        for response in resp.get("responses", []):
            for row in response.get("results", []):
                if row.get("uuid"):
                    statuses[str(row["uuid"])] = row.get("status", "")
        for run_uuid, _ in chunk:
            status = statuses.get(run_uuid, "inserted")
            if status == "inserted":
                summary["pushed_new"] += 1
                state.add(run_uuid)
            elif status == "exists":
                summary["already_present"] += 1
                state.add(run_uuid)
            else:
                summary["errors"] += 1
                print(
                    f"WARNING server rejected {run_uuid}: {status}",
                    file=sys.stderr,
                )
        save_state(state_file, state)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="default: <runs-dir>/_working/.pushed_uuids.json",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="ignore the state file and re-push everything (server is idempotent)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.runs_dir.is_dir():
        parser.error(f"--runs-dir is not a directory: {args.runs_dir}")
    # Push state is collection state, so it belongs in the arm's working
    # directory. The arm root holds run directories only.
    state_file = args.state_file or args.runs_dir / "_working" / ".pushed_uuids.json"
    summary = sync_records(
        args.runs_dir,
        state_file,
        push_all=args.all,
        dry_run=args.dry_run,
    )
    print("summary: " + " ".join(f"{key}={val}" for key, val in summary.items()))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

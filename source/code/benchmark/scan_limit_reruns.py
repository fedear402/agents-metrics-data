"""Scan the two Sonnet arms for runs stopped by a harness limit.

Two classes, per decision record 2026-08-24_sonnet-rerun-replacements.md:

- class 1 ``turn_cap``: constrained-agent (``tool_loop``) runs the Claude CLI
  stopped at the twelve-turn cap. The CLI's own ``error_max_turns`` result
  subtype is the marker; a run that spent twelve turns and still submitted a
  final answer carries a normal ``success`` result and is excluded.
- class 2 ``call_timeout``: runs where a model call hit the 180-second
  subprocess timeout, recorded as a claude block with ``error == "timeout"``.

Read-only. Writes nothing under ``data/runs/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_dataset as ad  # noqa: E402
import harness  # noqa: E402

SONNET_ARMS = ("sonnet_zero_shot", "sonnet_few_shot")
RUN_RECORD_NAME = "run_record.json"
STREAM_NAME = "agent_stream.jsonl"


def stream_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / STREAM_NAME
    if not path.exists():
        return []
    return harness.load_json_lines(path.read_text(errors="replace"))


def stream_result_event(run_dir: Path) -> dict[str, Any] | None:
    for event in reversed(stream_events(run_dir)):
        if event.get("type") == "result":
            return event
    return None


def hit_turn_cap(run_dir: Path, record: dict[str, Any]) -> tuple[bool, str]:
    """Was this tool_loop run stopped by the twelve-turn cap?"""
    if record.get("exec_mode") != "tool_loop":
        return False, "not_tool_loop"
    blocks = harness.claude_blocks(record)
    if any(block.get("ok") is True for block in blocks):
        # The CLI exited 0 with a parsed result: it stopped on its own.
        return False, "cli_ok"
    if any(str(block.get("error") or "") == "timeout" for block in blocks):
        return False, "session_wall_timeout"
    event = stream_result_event(run_dir)
    if event is None:
        return False, "no_result_event"
    subtype = str(event.get("subtype") or "")
    if subtype == "error_max_turns":
        return True, "error_max_turns"
    return False, f"result_subtype:{subtype or 'none'}"


def hit_call_timeout(record: dict[str, Any]) -> bool:
    return any(
        str(block.get("error") or "") == "timeout"
        for block in harness.claude_blocks(record)
    )


def scan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm_id in SONNET_ARMS:
        spec = ad.ARMS[arm_id]
        for entry in sorted(spec.path.iterdir()):
            if not entry.is_dir() or entry.name.startswith("_"):
                continue
            record_path = entry / RUN_RECORD_NAME
            if not record_path.exists():
                continue
            try:
                record = json.loads(record_path.read_text())
            except json.JSONDecodeError:
                continue
            suffix = entry.name.rsplit("rep", 1)[-1]
            rep = int(suffix) if suffix.isdigit() else -1
            capped, why = hit_turn_cap(entry, record)
            timed_out = hit_call_timeout(record)
            usable, reason = ad.wave_record_usable(entry, record)
            rows.append(
                {
                    "arm_id": arm_id,
                    "run_dir": entry.name,
                    "task_id": record.get("task_id"),
                    "platform": record.get("platform"),
                    "exec_mode": record.get("exec_mode"),
                    "prompt_mode": record.get("prompt_mode"),
                    "model": record.get("model"),
                    "rep": rep,
                    "turn_cap": capped,
                    "turn_cap_why": why,
                    "call_timeout": timed_out,
                    "n_turns": (harness.claude_blocks(record) or [{}])[0].get(
                        "num_turns"
                    ),
                    "usable": usable,
                    "usable_reason": reason,
                    "provider_limit": bool(record.get("provider_limit")),
                    "claude_timeout_seconds": record.get("claude_timeout_seconds"),
                    "has_result_json": (entry / "result.json").exists(),
                }
            )
    return rows


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    rows = scan()
    payload = json.dumps(rows, indent=1)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload)
        print(f"wrote {len(rows)} rows -> {out}")
    else:
        print(payload)

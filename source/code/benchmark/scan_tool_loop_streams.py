"""Survey every stored Sonnet ``tool_loop`` stream's terminal result event.

Read-only diagnostic behind the class-1 selection rule of the 2026-08-24
limit re-run wave. Prints the joint distribution of (result subtype, is_error,
stop_reason, num_turns bucket, CLI ok) so the turn-cap marker can be verified
against the corpus rather than assumed.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_dataset as ad  # noqa: E402
import harness  # noqa: E402

SONNET_ARMS = ("sonnet_zero_shot", "sonnet_few_shot")


def main() -> int:
    counter: collections.Counter = collections.Counter()
    detail: list[dict] = []
    for arm_id in SONNET_ARMS:
        spec = ad.ARMS[arm_id]
        for entry in sorted(spec.path.iterdir()):
            if not entry.is_dir() or entry.name.startswith("_"):
                continue
            record_path = entry / "run_record.json"
            if not record_path.exists():
                continue
            try:
                record = json.loads(record_path.read_text())
            except json.JSONDecodeError:
                continue
            if record.get("exec_mode") != "tool_loop":
                continue
            blocks = harness.claude_blocks(record)
            cli_ok = any(block.get("ok") is True for block in blocks)
            stream = entry / "agent_stream.jsonl"
            event = None
            if stream.exists():
                for candidate in reversed(
                    harness.load_json_lines(stream.read_text(errors="replace"))
                ):
                    if candidate.get("type") == "result":
                        event = candidate
                        break
            subtype = str((event or {}).get("subtype") or "none")
            is_error = (event or {}).get("is_error")
            stop_reason = str((event or {}).get("stop_reason") or "none")
            turns = (event or {}).get("num_turns")
            key = (
                "stream" if stream.exists() else "no_stream",
                subtype,
                str(is_error),
                stop_reason,
                "turns>=12" if isinstance(turns, int) and turns >= 12 else "turns<12",
                "cli_ok" if cli_ok else "cli_not_ok",
            )
            counter[key] += 1
            if subtype == "error_max_turns" or (
                isinstance(turns, int) and turns >= 12
            ):
                detail.append(
                    {
                        "arm_id": arm_id,
                        "run_dir": entry.name,
                        "subtype": subtype,
                        "is_error": is_error,
                        "stop_reason": stop_reason,
                        "num_turns": turns,
                        "cli_ok": cli_ok,
                    }
                )
    for key, count in sorted(counter.items(), key=lambda item: -item[1]):
        print(f"{count:6d}  {key}")
    print()
    print(f"records with subtype=error_max_turns or num_turns>=12: {len(detail)}")
    for row in detail:
        print(
            f"  {row['arm_id']} {row['run_dir']} subtype={row['subtype']} "
            f"is_error={row['is_error']} stop_reason={row['stop_reason']} "
            f"num_turns={row['num_turns']} cli_ok={row['cli_ok']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Selection rules for the 2026-08-24 Sonnet limit re-run wave.

Decision record: ``quality_reports/decision_records/2026-08-24_sonnet-rerun-
replacements.md``. Two classes of stored Claude Sonnet 4.6 runs are re-run
without the harness limit that stopped them.

Class 1, ``turn_cap``
    A constrained-agent (``tool_loop``) run the Claude CLI stopped at the
    twelve-turn cap. The CLI states this itself, in one of two encodings:

    - stream-era records (``--output-format stream-json``): the terminal
      ``result`` event carries ``subtype == "error_max_turns"``;
    - legacy records (plain ``--output-format json``, written before the
      stream capture): the CLI exited 1 and its final JSON object, kept in
      the record's ``stdout_tail``, carries ``"terminal_reason":"max_turns"``.

    A run that spent twelve turns and still submitted a final answer ends
    with ``subtype == "success"``, ``is_error == false``, and CLI exit 0. It
    was not stopped by the cap and is excluded. Two such runs exist
    (``anova_oneway__stata__few_shot__tool_loop__rep02`` and
    ``vif__stata__few_shot__tool_loop__rep11``, both ``num_turns == 12``).

Class 2, ``call_timeout``
    A run whose outcome is the 180-second model-call timeout. The harness
    stores ``{"ok": false, "error": "timeout"}`` in the claude block that hit
    ``subprocess.TimeoutExpired``, and the loader reads the record with
    ``wave_record_usable`` reason ``"timeout"``. The reason, not the raw
    marker, is the rule: a ``self_repair`` record whose second attempt timed
    out while another attempt returned a response is a completed run and is
    excluded.

Scope
    Both classes are restricted to the 21 analyzed tasks, so the quarantined
    task never enters the wave. Class 2 is further restricted to the pinned
    paper sample, because an out-of-sample timeout record is already an
    eligible row sitting past the twelve-repetition window and replacing it
    cannot change the estimation sample. Class 1 is not restricted that way:
    a capped record can be ineligible today and eligible after replacement,
    which would change which repetitions the loader selects, so every capped
    record is re-run.

Read-only. This module never writes under ``data/runs/``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_dataset as ad  # noqa: E402
import harness  # noqa: E402
import strict_posthoc as sp  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SONNET_ARMS = ("sonnet_zero_shot", "sonnet_few_shot")
RUN_RECORD_NAME = "run_record.json"
STREAM_NAME = "agent_stream.jsonl"
PAPER_SAMPLE_MANIFEST = PROJECT_ROOT / "data" / "analysis" / "paper_sample_manifest.json"

#: The legacy (pre-stream) encoding of the same CLI stop.
LEGACY_TURN_CAP_MARKER = '"terminal_reason":"max_turns"'

CLASS_TURN_CAP = "turn_cap"
CLASS_CALL_TIMEOUT = "call_timeout"


@dataclass(frozen=True)
class ReRun:
    """One stored run to repeat without the limit that stopped it."""

    rerun_class: str
    arm_id: str
    run_dir: str
    task_id: str
    platform: str
    prompt_mode: str
    exec_mode: str
    rep: int
    in_paper_sample: bool
    marker: str

    def key(self) -> tuple[str, str]:
        return (self.arm_id, self.run_dir)


def paper_sample_run_dirs() -> set[tuple[str, str]]:
    payload = json.loads(PAPER_SAMPLE_MANIFEST.read_text())
    return {
        (entry["arm_id"], entry["run_dir"].split("/")[-1])
        for entry in payload["runs"]
        if entry["arm_id"] in SONNET_ARMS
    }


def terminal_result_event(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / STREAM_NAME
    if not path.exists():
        return None
    for event in reversed(harness.load_json_lines(path.read_text(errors="replace"))):
        if event.get("type") == "result":
            return event
    return None


def turn_cap_marker(run_dir: Path, record: dict[str, Any]) -> str | None:
    """Return the CLI's own turn-cap marker, or None when the cap never fired."""
    if record.get("exec_mode") != "tool_loop":
        return None
    blocks = harness.claude_blocks(record)
    if any(block.get("ok") is True for block in blocks):
        # CLI exit 0 with a parsed result: the model ended the session itself.
        return None
    event = terminal_result_event(run_dir)
    if event is not None:
        if str(event.get("subtype") or "") == "error_max_turns":
            return "stream:error_max_turns"
        return None
    haystack = " ".join(
        str(block.get(key) or "")
        for block in blocks
        for key in ("stdout_tail", "stderr_tail")
    )
    if LEGACY_TURN_CAP_MARKER in haystack:
        return "stdout_tail:terminal_reason=max_turns"
    return None


def iter_records():
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
            yield arm_id, entry, record


def select() -> list[ReRun]:
    sample = paper_sample_run_dirs()
    analyzed = set(ad.ANALYSIS_TASK_IDS)
    picks: dict[tuple[str, str], ReRun] = {}
    for arm_id, entry, record in iter_records():
        task_id = str(record.get("task_id") or "")
        if task_id not in analyzed:
            continue  # the quarantined task never enters the wave
        suffix = entry.name.rsplit("rep", 1)[-1]
        if not suffix.isdigit():
            continue
        rep = int(suffix)
        in_sample = (arm_id, entry.name) in sample
        common = {
            "arm_id": arm_id,
            "run_dir": entry.name,
            "task_id": task_id,
            "platform": str(record.get("platform")),
            "prompt_mode": str(record.get("prompt_mode")),
            "exec_mode": str(record.get("exec_mode")),
            "rep": rep,
            "in_paper_sample": in_sample,
        }
        marker = turn_cap_marker(entry, record)
        if marker is not None:
            picks[(arm_id, entry.name)] = ReRun(
                rerun_class=CLASS_TURN_CAP, marker=marker, **common
            )
            continue  # class 1 wins the deduplication
        _, reason = ad.wave_record_usable(entry, record)
        if reason == "timeout" and in_sample:
            picks[(arm_id, entry.name)] = ReRun(
                rerun_class=CLASS_CALL_TIMEOUT,
                marker="wave_record_usable=timeout",
                **common,
            )
    return sorted(
        picks.values(),
        key=lambda item: (item.rerun_class, item.arm_id, item.run_dir),
    )


def as_rows(selection: list[ReRun]) -> list[dict[str, Any]]:
    return [asdict(item) for item in selection]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    selection = select()
    if argv:
        out = Path(argv[0])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(as_rows(selection), indent=1) + "\n")
        print(f"wrote {len(selection)} rows -> {out}")
    counts: dict[str, int] = {}
    for item in selection:
        counts[item.rerun_class] = counts.get(item.rerun_class, 0) + 1
    for name in sorted(counts):
        print(f"{name}: {counts[name]}")
    print(f"total: {len(selection)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

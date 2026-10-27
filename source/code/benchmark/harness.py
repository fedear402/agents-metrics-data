"""Benchmark harness library for the task benchmark.

This module is a library, not an entry point. It owns prompt construction,
cell execution for the Claude backend (``run_cell``), run-directory
preparation and copy-back, provider-limit detection, record and summary
writing, the subscription-meter probe, and the pacing controller. Collection
is driven by ``collect.py``; the remote worker (``worker_loop.py``) calls
``run_cell`` directly. Each live cell receives a task-specific dataset copied
into the isolated run directory as ``input.csv`` so prompts can stay
platform-neutral.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:  # pty/select power the Unix-only claude-tui usage probe; absent on Windows.
    import pty
    import select
except ImportError:  # pragma: no cover - Windows
    pty = None  # type: ignore[assignment]
    select = None  # type: ignore[assignment]

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent.parent
sys.path.insert(0, str(ROOT))

from demonstrations import demo_for  # noqa: E402
from tasks import EXEC_MODES, PLATFORMS, PROMPT_MODES, TASKS, Task  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "tasks" / "inputs"
REFERENCE_DIR = PROJECT_ROOT / "data" / "tasks" / "references"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "data" / "scratch_runs"
MODEL = "sonnet"
MAX_TURNS_TOOL = 12
MAX_TURNS_AGENTIC = 60
MAX_EXECUTIONS_REPAIR = 4
TIMEOUT_NO_FEEDBACK = 180
TIMEOUT_TOOL = 600
TIMEOUT_AGENTIC = 1800
RUNS_DIR = DEFAULT_RUNS_DIR
# --------------------------------------------------------------------------- #
# Limit overrides for the 2026-08-24 Sonnet limit re-run wave (decision record
# 2026-08-24_sonnet-rerun-replacements). Both defaults leave the collected
# protocol exactly as it stands, so no corpus path changes behavior. Only
# ``limit_rerun_wave.py`` sets them.
# --------------------------------------------------------------------------- #
#: When False, the constrained-agent (``tool_loop``) call passes no
#: ``--max-turns`` flag, so the CLI runs until the model submits or the
#: ``TIMEOUT_TOOL`` session wall stops it.
TOOL_LOOP_TURN_CAP_ENABLED = True
#: When set, replaces the model-call timeout of the ``no_feedback`` and
#: ``self_repair`` Claude calls. Script execution keeps ``TIMEOUT_NO_FEEDBACK``,
#: so raising this never lengthens how long a generated script may run.
MODEL_CALL_TIMEOUT_OVERRIDE: int | None = None
# A runs dir holds one directory per run and nothing else. Collection state and
# derived summaries live one level down, in this underscore-prefixed directory
# that every analysis loader skips. Keep equal to
# ``analysis_dataset.WORKING_DIRNAME``.
WORKING_DIRNAME = "_working"
# Extra CLI args appended to every `claude -p` invocation. Empty for the corpus
# (no behavior change); pilot runners may set e.g. ["--strict-mcp-config"] to
# stop the subject model's session from loading the operator's MCP servers
# (2026-07-13 audit hermeticity finding).
CLAUDE_EXTRA_CLI_ARGS: list[str] = []


def claude_bin() -> str:
    """Resolve the Claude CLI executable.

    On Windows the npm install is a `claude.cmd` shim, which subprocess won't
    find by bare name (CreateProcess ignores PATHEXT). `shutil.which` is
    PATHEXT-aware, so it resolves the real path on every OS. Override with the
    CLAUDE_BIN env var if needed.
    """
    return os.environ.get("CLAUDE_BIN") or shutil.which("claude") or "claude"


def detect_stata_bin() -> Path:
    """Resolve the Stata batch binary per-OS. Override with env var STATA_BIN."""
    env = os.environ.get("STATA_BIN")
    if env:
        return Path(env)
    candidates = [
        # macOS
        Path("/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se"),
        Path("/Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp"),
        Path("/Applications/Stata/StataSE.app/Contents/MacOS/stata-se"),
        Path("/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp"),
        # Windows (StataNow/Stata 18-19 default installs)
        Path("C:/Program Files/Stata19/StataSE-64.exe"),
        Path("C:/Program Files/Stata19/StataMP-64.exe"),
        Path("C:/Program Files/Stata18/StataSE-64.exe"),
        Path("C:/Program Files/Stata18/StataMP-64.exe"),
        # Linux
        Path("/usr/local/stata/stata-se"),
        Path("/usr/local/stata/stata-mp"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]  # historical default; execute_script reports if missing


STATA_BIN = detect_stata_bin()
PERMISSION_MODE = "default"
AGENTIC_PERMISSION_MODE = "default"  # was bypassPermissions; the auto-mode classifier refuses to launch a bypass agent, and an explicit allowlist gives the same file+shell surface for an econometric task
AGENTIC_ALLOWED_TOOLS = "Read Write Edit Bash"  # full default-tool surface the agent needs (read csv, write/edit/run code) under default permission mode
AGENTIC_RUN_ROOT = Path.home() / "agentic_runs"
AGENTIC_MAX_REINVOKE = 3
TOOL_LOOP_ALLOWED_TOOLS = ["Bash"]
NO_FEEDBACK_DISALLOWED_TOOLS = ["Bash", "Write", "Edit", "Read"]
DEFAULT_EXEC_MODES = ["no_feedback", "self_repair", "tool_loop"]

# Numeric tolerance for first-pass scoring. The archived strict-scoring
# prototype is reference material until TaskCard evidence rules are locked.
COEF_REL_TOL = 1e-2
COEF_ABS_TOL = 5e-3
SE_REL_TOL = 5e-2
SE_ABS_TOL = 5e-3

CODE_FENCE_RE = re.compile(r"```(?:python|r|R|stata)\s*\n(.*?)```", re.DOTALL)
STATA_ERROR_RE = re.compile(r"(?:^|\n)r\([0-9]+\);")
PROVIDER_LIMIT_PATTERNS = (
    "usage limit",
    "session limit",
    "rate limit",
    "credit balance is too low",
    "authentication",
    "not logged in",
    "login",
)
ANSI_ESCAPE_RE = re.compile(
    r"\x1b\][^\a]*(?:\a|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]"
)
USAGE_SECTION_HEADERS = (
    "Current session",
    "Current week (all models)",
    "Current week (Sonnet only)",
)
USAGE_HEADER_PATTERNS = {
    "Current session": re.compile(r"Curr[a-z]*\s*session", re.IGNORECASE),
    "Current week (all models)": re.compile(
        r"Curr[a-z]*\s*week\s*\(?\s*all\s*models\s*\)?",
        re.IGNORECASE,
    ),
    "Current week (Sonnet only)": re.compile(
        r"Curr[a-z]*\s*week\s*\(?\s*Sonnet\s*only\s*\)?",
        re.IGNORECASE,
    ),
}
MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass(frozen=True)
class ClaudeUsageSnapshot:
    """Parsed Claude Code subscription-meter state from the `/usage` screen."""

    observed_at: float
    source: str
    current_session_percent: float | None = None
    current_session_resets: str | None = None
    current_session_reset_ts: float | None = None
    weekly_all_percent: float | None = None
    weekly_all_resets: str | None = None
    weekly_all_reset_ts: float | None = None
    weekly_sonnet_percent: float | None = None
    weekly_sonnet_resets: str | None = None
    weekly_sonnet_reset_ts: float | None = None

    @property
    def has_usage(self) -> bool:
        return any(
            value is not None
            for value in (
                self.current_session_percent,
                self.weekly_all_percent,
                self.weekly_sonnet_percent,
            )
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "observed_at": self.observed_at,
            "source": self.source,
            "current_session_percent": self.current_session_percent,
            "current_session_resets": self.current_session_resets,
            "current_session_reset_ts": self.current_session_reset_ts,
            "weekly_all_percent": self.weekly_all_percent,
            "weekly_all_resets": self.weekly_all_resets,
            "weekly_all_reset_ts": self.weekly_all_reset_ts,
            "weekly_sonnet_percent": self.weekly_sonnet_percent,
            "weekly_sonnet_resets": self.weekly_sonnet_resets,
            "weekly_sonnet_reset_ts": self.weekly_sonnet_reset_ts,
        }


def strip_terminal_control(text: str) -> str:
    cleaned = ANSI_ESCAPE_RE.sub("", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    return cleaned


def _usage_section(text: str, header: str) -> str:
    header_positions = []
    for name, pattern in USAGE_HEADER_PATTERNS.items():
        match = pattern.search(text)
        if match:
            header_positions.append((match.start(), match.end(), name))
    current = next(
        (position for position in header_positions if position[2] == header),
        None,
    )
    if current is None:
        return ""
    start = current[0]
    end = len(text)
    for other_start, _, other_name in header_positions:
        if other_name == header or other_start <= start:
            continue
        end = min(end, other_start)
    return text[start:end]


def _parse_usage_percent(section: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%\s*used", section)
    if not match:
        return None
    return float(match.group(1))


def _parse_reset_label(section: str) -> str | None:
    match = re.search(r"Rese?t?s?\s*([^\n]+)", section, flags=re.IGNORECASE)
    if not match:
        return None
    label = match.group(1).strip()
    return label or None


def _parse_ampm_time(text: str) -> tuple[int, int] | None:
    match = re.search(
        r"(?<![0-9])([0-9]{1,2})(?::([0-9]{2}))?\s*(am|pm)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = match.group(3).lower()
    if hour < 1 or hour > 12 or minute > 59:
        return None
    if ampm == "pm" and hour != 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    return hour, minute


def parse_reset_timestamp(reset_label: str | None, now: float) -> float | None:
    """Parse reset labels such as `1pm (...)` or `Jun 8 at 8pm (...)`."""

    if not reset_label:
        return None
    tz_match = re.search(r"\(([^)]+)\)", reset_label)
    try:
        tz = ZoneInfo(tz_match.group(1)) if tz_match else ZoneInfo("UTC")
    except Exception:
        tz = ZoneInfo("UTC")
    local_now = datetime.fromtimestamp(now, tz)
    label = re.sub(r"\s*\([^)]+\)\s*$", "", reset_label.strip())
    lower = label.lower()

    time_parts = _parse_ampm_time(label)
    if time_parts is None:
        return None
    hour, minute = time_parts

    month_match = re.search(
        r"\b([A-Za-z]+)\s*([0-9]{1,2})(?:,\s*([0-9]{4}))?\s*(?:at)?",
        label,
        flags=re.IGNORECASE,
    )
    if month_match and month_match.group(1).lower() in MONTHS:
        month = MONTHS[month_match.group(1).lower()]
        day = int(month_match.group(2))
        year = int(month_match.group(3) or local_now.year)
        candidate = datetime(year, month, day, hour, minute, tzinfo=tz)
        if candidate <= local_now:
            candidate = datetime(year + 1, month, day, hour, minute, tzinfo=tz)
        return candidate.timestamp()

    day_offset = 0
    if lower.startswith("tomorrow"):
        day_offset = 1
    candidate_date = local_now.date()
    if day_offset:
        candidate_date = datetime.fromtimestamp(
            local_now.timestamp() + day_offset * 86400,
            tz,
        ).date()
    candidate = datetime(
        candidate_date.year,
        candidate_date.month,
        candidate_date.day,
        hour,
        minute,
        tzinfo=tz,
    )
    if day_offset == 0 and candidate <= local_now:
        candidate = datetime.fromtimestamp(candidate.timestamp() + 86400, tz)
    return candidate.timestamp()


def parse_claude_usage(
    text: str,
    *,
    now: float | None = None,
    source: str = "claude_usage",
) -> ClaudeUsageSnapshot:
    if now is None:
        now = time.time()
    cleaned = strip_terminal_control(text)

    session_section = _usage_section(cleaned, "Current session")
    weekly_all_section = _usage_section(cleaned, "Current week (all models)")
    weekly_sonnet_section = _usage_section(cleaned, "Current week (Sonnet only)")

    session_reset = _parse_reset_label(session_section)
    weekly_all_reset = _parse_reset_label(weekly_all_section)
    weekly_sonnet_reset = _parse_reset_label(weekly_sonnet_section)

    return ClaudeUsageSnapshot(
        observed_at=now,
        source=source,
        current_session_percent=_parse_usage_percent(session_section),
        current_session_resets=session_reset,
        current_session_reset_ts=parse_reset_timestamp(session_reset, now),
        weekly_all_percent=_parse_usage_percent(weekly_all_section),
        weekly_all_resets=weekly_all_reset,
        weekly_all_reset_ts=parse_reset_timestamp(weekly_all_reset, now),
        weekly_sonnet_percent=_parse_usage_percent(weekly_sonnet_section),
        weekly_sonnet_resets=weekly_sonnet_reset,
        weekly_sonnet_reset_ts=parse_reset_timestamp(weekly_sonnet_reset, now),
    )


def run_claude_usage_tui(timeout_seconds: float = 20.0) -> str:
    """Open Claude Code briefly, send `/usage`, capture output, and exit.

    The probe is intentionally best-effort. It is used only when the caller opts
    in, and callers must fall back to proxy pacing if the TUI surface changes.
    """

    if pty is None or select is None:
        raise RuntimeError(
            "claude-tui usage probe requires a pty (Unix only); "
            "use --pacing-usage-probe command on Windows"
        )
    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        [claude_bin()],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)
    chunks: list[bytes] = []
    started = time.time()
    deadline = started + timeout_seconds
    sent_usage = False
    try:
        while time.time() < deadline:
            now = time.time()
            if not sent_usage and now - started >= 1.5:
                os.write(master_fd, b"/usage\r")
                sent_usage = True
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                try:
                    data = os.read(master_fd, 8192)
                except OSError:
                    break
                if not data:
                    break
                chunks.append(data)
            text = b"".join(chunks).decode(errors="replace")
            snapshot = parse_claude_usage(text)
            # `/usage` renders the current-session meter before the weekly-all
            # meter. Do not return on that first partial screen: callers with a
            # weekly cap need the weekly-all reading to enforce it. The
            # model-specific meter is optional because current Claude plans do
            # not always display it.
            if (
                sent_usage
                and snapshot.weekly_all_percent is not None
            ):
                break
            if process.poll() is not None:
                break
    finally:
        try:
            os.write(master_fd, b"\x03/quit\r")
        except OSError:
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        os.close(master_fd)
    return b"".join(chunks).decode(errors="replace")


class ClaudeUsageProbe:
    """Cached, fail-soft `/usage` probe for subscription-aware pacing."""

    def __init__(
        self,
        *,
        mode: str = "off",
        command: str | None = None,
        interval_seconds: float = 120.0,
        timeout_seconds: float = 20.0,
        time_func: Any = time.time,
        runner: Any | None = None,
    ) -> None:
        self.mode = mode
        self.command = command
        self.interval_seconds = max(0.0, float(interval_seconds))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.time_func = time_func
        self.runner = runner
        self.last_error: str | None = None
        self._last_probe_at: float | None = None
        self._last_snapshot: ClaudeUsageSnapshot | None = None

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def snapshot(self) -> ClaudeUsageSnapshot | None:
        if not self.enabled:
            return None
        now = self.time_func()
        if (
            self._last_probe_at is not None
            and now - self._last_probe_at < self.interval_seconds
        ):
            return self._last_snapshot
        self._last_probe_at = now
        try:
            raw = self._run()
            snapshot = parse_claude_usage(raw, now=now, source=self.mode)
            if not snapshot.has_usage:
                self.last_error = "usage probe returned no parseable usage meter"
                self._last_snapshot = None
                return None
            self.last_error = None
            self._last_snapshot = snapshot
            return snapshot
        except Exception as exc:
            self.last_error = repr(exc)
            self._last_snapshot = None
            return None

    def _run(self) -> str:
        if self.runner is not None:
            return str(self.runner())
        if self.mode == "claude-tui":
            return run_claude_usage_tui(timeout_seconds=self.timeout_seconds)
        if self.mode == "command":
            if not self.command:
                raise ValueError("usage probe command mode requires a command")
            completed = subprocess.run(
                shlex.split(self.command),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            return f"{completed.stdout}\n{completed.stderr}"
        raise ValueError(f"unknown usage probe mode: {self.mode}")


def claude_record_cost(record: dict[str, Any]) -> float:
    """Sum Claude CLI cost fields across one or more Claude calls in a record."""
    total = 0.0
    for block in claude_blocks(record):
        value = block.get("total_cost_usd")
        if finite_number(value):
            total += float(value)
    return total


class PacingController:
    """Best-effort subscription pacing from `/usage` plus proxy usage events.

    When a Claude usage probe is configured, the controller uses the parsed
    subscription meter first. If the probe fails or is disabled, it falls back to
    run-count and/or CLI-reported API-equivalent cost in a rolling window.
    """

    def __init__(
        self,
        *,
        state_file: Path,
        window_seconds: float = 300 * 60,
        target_runs: int | None = None,
        target_cost_usd: float | None = None,
        reserve_fraction: float = 0.0,
        min_delay_seconds: float = 0.0,
        max_sleep_seconds: float = 1800.0,
        usage_probe: ClaudeUsageProbe | None = None,
        session_soft_limit_percent: float | None = None,
        session_hard_limit_percent: float | None = None,
        weekly_all_hard_limit_percent: float | None = None,
        weekly_sonnet_hard_limit_percent: float | None = None,
        fill_current_session: bool = False,
        session_reserve_percent: float = 0.0,
        default_session_percent_per_run: float = 0.15,
        external_usage_tolerance_percent_per_hour: float = 6.0,
        weekly_fill_target_percent: float | None = None,
        default_weekly_percent_per_run: float = 0.07,
        reset_padding_seconds: float = 60.0,
        time_func: Any = time.time,
        sleep_func: Any = time.sleep,
    ) -> None:
        self.state_file = state_file
        self.window_seconds = window_seconds
        self.target_runs = target_runs
        self.target_cost_usd = target_cost_usd
        self.reserve_fraction = max(0.0, min(float(reserve_fraction), 0.95))
        self.min_delay_seconds = max(0.0, float(min_delay_seconds))
        self.max_sleep_seconds = max(1.0, float(max_sleep_seconds))
        self.usage_probe = usage_probe
        self.session_soft_limit_percent = session_soft_limit_percent
        self.session_hard_limit_percent = session_hard_limit_percent
        self.weekly_all_hard_limit_percent = weekly_all_hard_limit_percent
        self.weekly_sonnet_hard_limit_percent = weekly_sonnet_hard_limit_percent
        self.fill_current_session = fill_current_session
        self.session_reserve_percent = max(0.0, float(session_reserve_percent))
        self.default_session_percent_per_run = max(
            0.01,
            float(default_session_percent_per_run),
        )
        self.external_usage_tolerance_percent_per_hour = max(
            0.0,
            float(external_usage_tolerance_percent_per_hour),
        )
        self.weekly_fill_target_percent = weekly_fill_target_percent
        # Measured 2026-07-04 on this account: 51% -> 55% weekly over ~50 runs
        # (0.05-0.08pp/run, attribution-confounded). The live estimator below
        # replaces this default as soon as enough same-week snapshots exist.
        self.default_weekly_percent_per_run = max(
            0.005,
            float(default_weekly_percent_per_run),
        )
        self.reset_padding_seconds = max(0.0, float(reset_padding_seconds))
        self.time_func = time_func
        self.sleep_func = sleep_func
        self._last_delay_reason = "rolling subscription guard"
        self._last_delay_requires_recheck = False
        self._last_saved_usage_observed_at: float | None = None

    @property
    def enabled(self) -> bool:
        return bool(
            self.target_runs
            or self.target_cost_usd
            or self.min_delay_seconds
            or (
                self.usage_probe
                and self.usage_probe.enabled
                and (
                    self.session_soft_limit_percent is not None
                    or self.session_hard_limit_percent is not None
                    or self.weekly_all_hard_limit_percent is not None
                    or self.weekly_sonnet_hard_limit_percent is not None
                    or self.fill_current_session
                )
            )
        )

    def _effective_target_runs(self) -> float | None:
        if self.target_runs is None:
            return None
        return max(1.0, float(self.target_runs) * (1.0 - self.reserve_fraction))

    def _effective_target_cost(self) -> float | None:
        if self.target_cost_usd is None:
            return None
        return max(0.0001, float(self.target_cost_usd) * (1.0 - self.reserve_fraction))

    def load_state(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {}
        try:
            payload = json.loads(self.state_file.read_text())
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def load_events(self, now: float | None = None) -> list[dict[str, Any]]:
        if now is None:
            now = self.time_func()
        payload = self.load_state()
        raw_events = payload.get("events", [])
        if not isinstance(raw_events, list):
            return []
        recent = []
        for event in raw_events:
            if not isinstance(event, dict):
                continue
            ts = event.get("ts")
            if not finite_number(ts):
                continue
            if float(ts) >= now - self.window_seconds:
                recent.append(event)
        return sorted(recent, key=lambda event: float(event["ts"]))

    def save_events(self, events: list[dict[str, Any]]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = self.load_state()
        payload.update(
            {
                "window_seconds": self.window_seconds,
                "target_runs": self.target_runs,
                "target_cost_usd": self.target_cost_usd,
                "reserve_fraction": self.reserve_fraction,
                "fill_current_session": self.fill_current_session,
                "session_reserve_percent": self.session_reserve_percent,
                "default_session_percent_per_run": (
                    self.default_session_percent_per_run
                ),
                "external_usage_tolerance_percent_per_hour": (
                    self.external_usage_tolerance_percent_per_hour
                ),
                "events": events,
            }
        )
        self.state_file.write_text(json.dumps(json_safe(payload), indent=2))

    def save_usage_snapshot(self, snapshot: ClaudeUsageSnapshot) -> None:
        if snapshot.observed_at == self._last_saved_usage_observed_at:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = self.load_state()
        snapshots = payload.get("usage_snapshots", [])
        if not isinstance(snapshots, list):
            snapshots = []
        snapshots.append(snapshot.to_json())
        payload["usage_snapshots"] = snapshots[-50:]
        self.state_file.write_text(json.dumps(json_safe(payload), indent=2))
        self._last_saved_usage_observed_at = snapshot.observed_at

    def estimate_next_cost(self, events: list[dict[str, Any]]) -> float:
        costs = [
            float(event["cost_usd"])
            for event in events[-20:]
            if finite_number(event.get("cost_usd")) and float(event["cost_usd"]) > 0
        ]
        if not costs:
            return 0.10
        return sum(costs) / len(costs)

    def _sleep_until_reset(self, reset_ts: float | None, now: float) -> float:
        if finite_number(reset_ts) and float(reset_ts) > now:
            return float(reset_ts) - now + self.reset_padding_seconds
        return self.max_sleep_seconds

    def _usage_soft_delay(
        self,
        *,
        percent: float,
        reset_ts: float | None,
        now: float,
    ) -> float:
        if self.session_soft_limit_percent is None:
            return 0.0
        hard = self.session_hard_limit_percent or 100.0
        span = max(1.0, hard - self.session_soft_limit_percent)
        pressure = min(
            1.0, max(0.0, (percent - self.session_soft_limit_percent) / span)
        )
        if pressure <= 0:
            return 0.0
        seconds_to_reset = (
            float(reset_ts) - now
            if finite_number(reset_ts) and float(reset_ts) > now
            else self.window_seconds
        )
        return max(
            self.min_delay_seconds,
            min(self.max_sleep_seconds, seconds_to_reset * pressure / 6.0),
        )

    def load_usage_snapshots(self) -> list[dict[str, Any]]:
        payload = self.load_state()
        raw_snapshots = payload.get("usage_snapshots", [])
        if not isinstance(raw_snapshots, list):
            return []
        snapshots = []
        for snapshot in raw_snapshots:
            if not isinstance(snapshot, dict):
                continue
            if not finite_number(snapshot.get("observed_at")):
                continue
            if not finite_number(snapshot.get("current_session_percent")):
                continue
            snapshots.append(snapshot)
        return sorted(snapshots, key=lambda item: float(item["observed_at"]))

    @staticmethod
    def _event_count_between(
        events: list[dict[str, Any]],
        start_ts: float,
        end_ts: float,
    ) -> int:
        count = 0
        for event in events:
            ts = event.get("ts")
            if finite_number(ts) and start_ts < float(ts) <= end_ts:
                count += 1
        return count

    def estimate_session_percent_per_run(
        self,
        events: list[dict[str, Any]],
    ) -> float:
        samples = []
        snapshots = self.load_usage_snapshots()
        for previous, current in zip(snapshots, snapshots[1:]):
            previous_reset = previous.get("current_session_reset_ts")
            current_reset = current.get("current_session_reset_ts")
            if (
                finite_number(previous_reset)
                and finite_number(current_reset)
                and abs(float(previous_reset) - float(current_reset)) > 60
            ):
                continue
            delta = float(current["current_session_percent"]) - float(
                previous["current_session_percent"]
            )
            if delta <= 0:
                continue
            event_count = self._event_count_between(
                events,
                float(previous["observed_at"]),
                float(current["observed_at"]),
            )
            if event_count <= 0:
                continue
            samples.append(delta / event_count)

        if not samples:
            estimate = self.default_session_percent_per_run
        else:
            recent = sorted(samples[-10:])
            estimate = recent[len(recent) // 2]
        return min(2.5, max(0.01, estimate))

    def estimate_external_usage_excess_rate(
        self,
        events: list[dict[str, Any]],
        percent_per_run: float,
    ) -> float:
        snapshots = self.load_usage_snapshots()
        tolerance_per_second = self.external_usage_tolerance_percent_per_hour / 3600.0
        excess_rates = []
        for previous, current in list(zip(snapshots, snapshots[1:]))[-5:]:
            start_ts = float(previous["observed_at"])
            end_ts = float(current["observed_at"])
            elapsed = end_ts - start_ts
            if elapsed <= 0:
                continue
            previous_reset = previous.get("current_session_reset_ts")
            current_reset = current.get("current_session_reset_ts")
            if (
                finite_number(previous_reset)
                and finite_number(current_reset)
                and abs(float(previous_reset) - float(current_reset)) > 60
            ):
                continue
            delta = float(current["current_session_percent"]) - float(
                previous["current_session_percent"]
            )
            if delta <= 0:
                continue
            event_count = self._event_count_between(events, start_ts, end_ts)
            expected_harness_delta = event_count * percent_per_run
            external_delta = max(0.0, delta - expected_harness_delta)
            rate = external_delta / elapsed
            if rate > tolerance_per_second:
                excess_rates.append(rate - tolerance_per_second)
        if not excess_rates:
            return 0.0
        return max(excess_rates[-3:])

    def session_fill_delay(
        self,
        snapshot: ClaudeUsageSnapshot,
        events: list[dict[str, Any]],
        now: float,
    ) -> tuple[float, str] | None:
        if not self.fill_current_session:
            return None
        if snapshot.current_session_percent is None:
            return None
        hard = self.session_hard_limit_percent
        if hard is None:
            return None
        reset_ts = snapshot.current_session_reset_ts
        if not finite_number(reset_ts) or float(reset_ts) <= now:
            return None

        seconds_to_reset = max(0.0, float(reset_ts) - now - self.reset_padding_seconds)
        if seconds_to_reset <= 0:
            return None

        percent_per_run = self.estimate_session_percent_per_run(events)
        excess_external_rate = self.estimate_external_usage_excess_rate(
            events,
            percent_per_run,
        )
        projected_external_usage = excess_external_rate * seconds_to_reset
        usable_percent = (
            hard
            - snapshot.current_session_percent
            - self.session_reserve_percent
            - projected_external_usage
        )
        if usable_percent <= 0:
            return (
                self._sleep_until_reset(reset_ts, now),
                (
                    "Claude /usage current-session fill budget exhausted "
                    f"({snapshot.current_session_percent:.1f}% used)"
                ),
            )

        allowed_runs = max(1.0, usable_percent / percent_per_run)
        interval = seconds_to_reset / allowed_runs
        if not events:
            return None

        elapsed_since_last = now - float(events[-1]["ts"])
        if elapsed_since_last >= interval:
            return None
        delay = interval - elapsed_since_last
        reason = (
            "current-session fill target "
            f"({usable_percent:.1f}% usable, "
            f"{seconds_to_reset / 60:.1f}m to reset, "
            f"{percent_per_run:.3f}%/run)"
        )
        if excess_external_rate > 0:
            reason += (
                ", external-use backoff "
                f"({excess_external_rate * 3600:.1f}%/h over tolerance)"
            )
        return delay, reason

    WEEKLY_ESTIMATOR_MIN_RUNS = 25
    # Fail-closed guard: pessimistic pp/run used to project the weekly meter
    # forward from the last known-good reading (~2-4x the measured clean rate),
    # and how many runs may proceed with no weekly reading at all before the
    # worker pauses to force a fresh one.
    WEEKLY_FAILCLOSED_COST_FLOOR = 0.12
    WEEKLY_FAILCLOSED_BLIND_BUDGET = 20

    def weekly_failclosed_candidate(
        self,
        now: float,
        events: list[dict[str, Any]],
    ) -> tuple[float, str] | None:
        """Fail-closed weekly cap that does NOT depend on the current probe.

        Projects the weekly meter forward from the last current-week reading
        using the local run count and a pessimistic per-run cost, and stops the
        worker if that projection reaches the hard cap. An intermittently
        failing usage probe therefore can never silently keep spending: between
        readings the projected usage still climbs and eventually halts
        collection until the next reset or a fresh reading. Only active when a
        weekly hard cap is configured.
        """
        if self.weekly_all_hard_limit_percent is None:
            return None
        payload = self.load_state()
        snaps = [
            s
            for s in payload.get("usage_snapshots", [])
            if isinstance(s, dict)
            and finite_number(s.get("weekly_all_percent"))
            and finite_number(s.get("observed_at"))
            and (
                not finite_number(s.get("weekly_all_reset_ts"))
                or float(s["weekly_all_reset_ts"]) > now
            )
        ]
        if not snaps:
            # A cap is required but the meter has never been read this week.
            # Allow a small blind budget, then pause for a reading rather than
            # spend unbounded.
            if len(events) < self.WEEKLY_FAILCLOSED_BLIND_BUDGET:
                return None
            return (
                self.max_sleep_seconds,
                "weekly fail-closed: no usage reading yet this week; pausing for a probe",
            )
        anchor = snaps[-1]
        anchor_ts = float(anchor["observed_at"])
        anchor_pct = float(anchor["weekly_all_percent"])
        runs_since = sum(
            1
            for e in events
            if finite_number(e.get("ts")) and float(e["ts"]) > anchor_ts
        )
        cost = max(
            self.estimate_weekly_percent_per_run(events),
            self.WEEKLY_FAILCLOSED_COST_FLOOR,
        )
        projected = anchor_pct + runs_since * cost
        if projected >= self.weekly_all_hard_limit_percent:
            return (
                self._sleep_until_reset(anchor.get("weekly_all_reset_ts"), now),
                (
                    f"weekly fail-closed stop (projected {projected:.1f}% >= cap "
                    f"{self.weekly_all_hard_limit_percent:.0f}%; {runs_since} runs "
                    "since last good reading)"
                ),
            )
        return None

    def estimate_weekly_percent_per_run(
        self,
        events: list[dict[str, Any]],
    ) -> float:
        """Live weekly-%-per-run: MINIMUM rate over qualifying snapshot windows.

        Interactive (non-harness) usage can only push a window's apparent rate
        UP, so the least-contaminated window approximates the sweep-only cost.
        Min-over-windows makes the estimate recover within one quiet window
        after a burst of interactive use, instead of waiting for contaminated
        snapshots to age out (observed 2026-07-05: a first-vs-latest estimator
        stayed poisoned at ~0.24pp/run for hours after the account went quiet).
        Each window's delta is floored at 1pp (meter granularity), which biases
        every window slightly UP — still conservative against overspend.
        """
        payload = self.load_state()
        snapshots = [
            snap
            for snap in payload.get("usage_snapshots", [])
            if isinstance(snap, dict)
            and finite_number(snap.get("observed_at"))
            and finite_number(snap.get("weekly_all_percent"))
        ]
        if len(snapshots) < 2:
            return self.default_weekly_percent_per_run
        latest_reset = snapshots[-1].get("weekly_all_reset_ts")
        same_week = [
            snap
            for snap in snapshots
            if not finite_number(latest_reset)
            or not finite_number(snap.get("weekly_all_reset_ts"))
            or abs(float(snap["weekly_all_reset_ts"]) - float(latest_reset)) < 3600
        ]
        event_times = sorted(
            float(event["ts"])
            for event in events
            if finite_number(event.get("ts"))
        )
        window_rates: list[float] = []
        for i in range(len(same_week)):
            for j in range(i + 1, len(same_week)):
                first, last = same_week[i], same_week[j]
                delta = float(last["weekly_all_percent"]) - float(
                    first["weekly_all_percent"]
                )
                if delta < 0:
                    continue
                runs_between = len(
                    [
                        ts
                        for ts in event_times
                        if float(first["observed_at"]) <= ts <= float(last["observed_at"])
                    ]
                )
                if runs_between < self.WEEKLY_ESTIMATOR_MIN_RUNS:
                    continue
                window_rates.append(max(delta, 1.0) / runs_between)
        if window_rates:
            return min(0.5, max(0.02, min(window_rates)))
        return self.default_weekly_percent_per_run

    def weekly_fill_delay(
        self,
        snapshot: ClaudeUsageSnapshot,
        events: list[dict[str, Any]],
        now: float,
    ) -> tuple[float, str] | None:
        """Space runs so weekly usage reaches the fill target exactly at the
        weekly reset. No external-usage projection: interactive use raises the
        meter itself, shrinking headroom and lengthening intervals naturally.
        """
        if self.weekly_fill_target_percent is None:
            return None
        percent = snapshot.weekly_all_percent
        if percent is None:
            return None
        reset_ts = snapshot.weekly_all_reset_ts
        if not finite_number(reset_ts) or float(reset_ts) <= now:
            return None
        seconds_to_reset = max(0.0, float(reset_ts) - now - self.reset_padding_seconds)
        if seconds_to_reset <= 0:
            return None
        headroom = self.weekly_fill_target_percent - percent
        if headroom <= 0:
            return (
                self._sleep_until_reset(reset_ts, now),
                f"Claude /usage weekly fill target reached ({percent:.1f}% used)",
            )
        percent_per_run = self.estimate_weekly_percent_per_run(events)
        allowed_runs = max(1.0, headroom / percent_per_run)
        interval = seconds_to_reset / allowed_runs
        if not events:
            return None
        elapsed_since_last = now - float(events[-1]["ts"])
        if elapsed_since_last >= interval:
            return None
        delay = interval - elapsed_since_last
        reason = (
            "weekly fill target "
            f"({headroom:.1f}pp headroom, {seconds_to_reset / 3600:.1f}h to reset, "
            f"{percent_per_run:.3f}pp/run, ~{allowed_runs:.0f} runs budgeted)"
        )
        return delay, reason

    def usage_probe_sleep_candidates(
        self,
        now: float,
        events: list[dict[str, Any]],
    ) -> list[tuple[float, str]]:
        if not self.usage_probe or not self.usage_probe.enabled:
            return []
        snapshot = self.usage_probe.snapshot()
        if snapshot is None:
            if self.usage_probe.last_error:
                print(
                    "    usage probe unavailable; falling back to proxy pacing "
                    f"({self.usage_probe.last_error})",
                    flush=True,
                )
            return []
        self.save_usage_snapshot(snapshot)
        delays: list[tuple[float, str]] = []

        session_percent = snapshot.current_session_percent
        if session_percent is not None:
            if (
                self.session_hard_limit_percent is not None
                and session_percent >= self.session_hard_limit_percent
            ):
                delays.append(
                    (
                        self._sleep_until_reset(snapshot.current_session_reset_ts, now),
                        (
                            "Claude /usage current-session hard limit "
                            f"({session_percent:.1f}%)"
                        ),
                    )
                )
            elif (
                self.session_soft_limit_percent is not None
                and session_percent >= self.session_soft_limit_percent
            ):
                delays.append(
                    (
                        self._usage_soft_delay(
                            percent=session_percent,
                            reset_ts=snapshot.current_session_reset_ts,
                            now=now,
                        ),
                        (
                            "Claude /usage current-session soft limit "
                            f"({session_percent:.1f}%)"
                        ),
                    )
                )
            fill_delay = self.session_fill_delay(snapshot, events, now)
            if fill_delay is not None:
                delays.append(fill_delay)

        if (
            snapshot.weekly_all_percent is not None
            and self.weekly_all_hard_limit_percent is not None
            and snapshot.weekly_all_percent >= self.weekly_all_hard_limit_percent
        ):
            delays.append(
                (
                    self._sleep_until_reset(snapshot.weekly_all_reset_ts, now),
                    f"Claude /usage weekly all-model hard limit ({snapshot.weekly_all_percent:.1f}%)",
                )
            )

        if (
            snapshot.weekly_sonnet_percent is not None
            and self.weekly_sonnet_hard_limit_percent is not None
            and snapshot.weekly_sonnet_percent >= self.weekly_sonnet_hard_limit_percent
        ):
            delays.append(
                (
                    self._sleep_until_reset(snapshot.weekly_sonnet_reset_ts, now),
                    f"Claude /usage weekly Sonnet hard limit ({snapshot.weekly_sonnet_percent:.1f}%)",
                )
            )

        weekly_fill = self.weekly_fill_delay(snapshot, events, now)
        if weekly_fill is not None:
            delays.append(weekly_fill)

        return [(delay, reason) for delay, reason in delays if delay > 0]

    def compute_sleep_seconds(self, now: float | None = None) -> float:
        if not self.enabled:
            return 0.0
        if now is None:
            now = self.time_func()
        events = self.load_events(now)
        delays: list[tuple[float, str]] = self.usage_probe_sleep_candidates(
            now,
            events,
        )

        # Fail-closed weekly guard — independent of the current probe result, so
        # a failing probe cannot silently remove the cap.
        failclosed = self.weekly_failclosed_candidate(now, events)
        if failclosed is not None:
            delays.append(failclosed)

        if events:
            elapsed_since_last = now - float(events[-1]["ts"])
            if elapsed_since_last < self.min_delay_seconds:
                delays.append(
                    (self.min_delay_seconds - elapsed_since_last, "minimum delay")
                )

        target_runs = self._effective_target_runs()
        if target_runs is not None:
            interval = self.window_seconds / target_runs
            if events:
                elapsed_since_last = now - float(events[-1]["ts"])
                if elapsed_since_last < interval:
                    delays.append((interval - elapsed_since_last, "run-count target"))
            if len(events) + 1 > target_runs:
                index = max(0, int(len(events) + 1 - target_runs) - 1)
                delays.append(
                    (
                        float(events[index]["ts"]) + self.window_seconds - now,
                        "run-count target",
                    )
                )

        target_cost = self._effective_target_cost()
        if target_cost is not None:
            estimate = self.estimate_next_cost(events)
            interval = self.window_seconds * (estimate / target_cost)
            if events:
                elapsed_since_last = now - float(events[-1]["ts"])
                if elapsed_since_last < interval:
                    delays.append((interval - elapsed_since_last, "cost target"))
            running_cost = sum(
                float(event.get("cost_usd") or 0)
                for event in events
                if finite_number(event.get("cost_usd"))
            )
            if running_cost + estimate > target_cost:
                remaining_cost = running_cost
                for event in events:
                    remaining_cost -= (
                        float(event.get("cost_usd") or 0)
                        if finite_number(event.get("cost_usd"))
                        else 0.0
                    )
                    if remaining_cost + estimate <= target_cost:
                        delays.append(
                            (
                                float(event["ts"]) + self.window_seconds - now,
                                "cost target",
                            )
                        )
                        break

        if not delays:
            self._last_delay_requires_recheck = False
            return 0.0
        blocking_markers = (
            "hard limit",
            "fail-closed stop",
            "fill budget exhausted",
            "weekly fill target reached",
        )
        self._last_delay_requires_recheck = any(
            delay > 0 and any(marker in reason for marker in blocking_markers)
            for delay, reason in delays
        )
        delay, reason = max(delays, key=lambda item: item[0])
        self._last_delay_reason = reason
        return min(delay, self.max_sleep_seconds)

    def before_run(self, cell_id: str) -> None:
        while True:
            delay = self.compute_sleep_seconds()
            if delay <= 0:
                return
            reason = self._last_delay_reason
            print(
                f"    pacing sleep {delay:.1f}s before {cell_id} ({reason})",
                flush=True,
            )
            self.sleep_func(delay)
            if not self._last_delay_requires_recheck:
                return

    def record(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        now = self.time_func()
        events = self.load_events(now)
        events.append(
            {
                "ts": now,
                "cell_id": record.get("cell_id"),
                "cost_usd": claude_record_cost(record),
                "source": "harness",
            }
        )
        self.save_events(events)


def load_references() -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for path in sorted(REFERENCE_DIR.glob("reference*.json")):
        refs.update(json.loads(path.read_text()))
    return refs


REFERENCE = load_references()


def load_reference_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_tasks_from_registry(registry_path: Path) -> list[Task]:
    """Load executable task definitions from the reviewable JSON registry."""
    registry = json.loads(registry_path.read_text())
    entries = registry.get("tasks")
    if not isinstance(entries, list):
        raise ValueError("registry JSON must contain a list at key 'tasks'")

    required = {
        "task_id",
        "title",
        "prompt",
        "estimands",
        "reference_key",
        "data_file",
        "data_description",
    }
    tasks: list[Task] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each registry task must be an object")
        missing = required - entry.keys()
        if missing:
            raise ValueError(
                f"registry task {entry.get('task_id', '<unknown>')} missing keys: "
                f"{sorted(missing)}"
            )
        tasks.append(
            Task(
                task_id=str(entry["task_id"]),
                title=str(entry["title"]),
                prompt=str(entry["prompt"]),
                estimands=[str(item) for item in entry["estimands"]],
                reference_key=str(entry["reference_key"]),
                data_file=str(entry["data_file"]),
                data_description=str(entry["data_description"]),
                extra_hint=dict(entry.get("extra_hint") or {}),
            )
        )
    return tasks


def platform_lang_hint(platform: str) -> str:
    if platform == "python":
        return (
            "Use Python 3 with pandas and statsmodels or other standard "
            "scientific packages already available in the environment. Read the "
            "CSV with pandas and write JSON with json.dump."
        )
    if platform == "r":
        return (
            "Use Rscript. Prefer base R plus common econometrics packages such as "
            "AER, ivreg, sandwich, lmtest, survival, plm, vars, forecast, and "
            "jsonlite when needed. For IV/2SLS tasks, AER::ivreg or ivreg::ivreg "
            "with sandwich::vcovHC(..., type = \"HC1\") is available."
        )
    if platform == "stata":
        return (
            "Use Stata syntax in a do-file. Read the CSV with import delimited. "
            "Write result.json with Stata file commands or another reliable JSON "
            "writing approach. JSON numeric literals must include a leading zero, "
            "so do not use Stata's %g formats for JSON output; use fixed decimal "
            "formats such as %21.12f, which write 0.123 rather than .123. "
            "Check that result.json is valid JSON before finishing."
        )
    raise ValueError(platform)


def script_language(platform: str) -> str:
    return {"python": "python", "r": "r", "stata": "stata"}[platform]


def script_name(platform: str) -> str:
    return {"python": "run.py", "r": "run.R", "stata": "run.do"}[platform]


def execution_hint(platform: str) -> str:
    if platform == "python":
        return "Run the script with python3 run.py."
    if platform == "r":
        return "Run the script with Rscript --vanilla run.R."
    if platform == "stata":
        return (
            "Run the script with "
            f"{STATA_BIN} -b do run.do and inspect run.log if needed."
        )
    raise ValueError(platform)


def output_schema_blurb(task: Task) -> str:
    rows = ",\n        ".join(
        f'{{"term": "{term}", "estimate": <float>, "se": <float or null>, '
        '"n_obs": <int>}'
        for term in task.estimands
    )
    term_list = ", ".join(f'"{term}"' for term in task.estimands)
    return f"""\
Required output:
  - Write a file `result.json` in the working directory with this shape:
    {{
      "task_id": "{task.task_id}",
      "platform": "<python|r|stata>",
      "estimates": [
        {rows}
      ],
      "method_metadata": {{"estimator": "<string>", "vcov": "<string>"}},
      "diagnostics": {{}}
    }}
  - Estimate names must be exactly: {term_list}.
  - Report numeric values to at least 6 significant digits when possible.
  - Do not include extra prose in `result.json`.
"""


def demonstration_block(platform: str, task_id: str) -> str:
    """The task-family-matched worked demonstration for the few-shot condition.

    Respecified 2026-07-25 (decision record
    ``quality_reports/decision_records/2026-07-25_fewshot-respecification.md``):
    the earlier fixed toy OLS demonstration is retired. Each scored task now
    receives a worked demonstration of a related task from the same family,
    never the scored task itself, drawn from
    ``demonstrations.DEMONSTRATIONS``. The demonstration text feeds
    ``initial_prompt`` and therefore ``prompt_sha256``, which stamps the wave
    provenance of the new few-shot collection.

    Raises KeyError for a task id with no registered demonstration and
    ValueError for an unknown platform.
    """
    demo = demo_for(task_id, platform)
    return f"""\
Few-shot demonstration:
The following worked example solves a related econometric task from the same
family as the scored task below. It shows the expected implementation style
and the result.json conventions on its own small inline data. The scored task
is a different task: do not copy the demonstration's variable names, inline
data, or numbers. The scored task must read ./input.csv and write
./result.json.

{demo}
"""


def prompt_condition_block(prompt_mode: str, platform: str, task_id: str) -> str:
    if prompt_mode == "zero_shot":
        return ""
    if prompt_mode == "few_shot":
        return demonstration_block(platform, task_id)
    raise ValueError(prompt_mode)


def dataset_context(task: Task) -> str:
    """Describe the run-local data path before any source filename metadata."""

    source = task.data_file
    detail = task.data_description.strip()
    detail = re.sub(r"^Dataset:\s*[^.]+\.csv\.\s*", "", detail, count=1)
    lines = [
        f"Dataset provided as `./input.csv`; original source file: `{source}`."
    ]
    if detail:
        lines.append(detail)
    return "\n".join(lines)


def no_feedback_prompt(task: Task, platform: str, prompt_mode: str) -> str:
    lang = script_language(platform)
    return f"""\
You are a careful econometrician writing a single self-contained {platform} script.

{dataset_context(task)}

{prompt_condition_block(prompt_mode, platform, task.task_id)}

Task: {task.prompt}

{platform_lang_hint(platform)}

{output_schema_blurb(task)}

The script will be executed with the working directory set to a folder that
contains `input.csv`. Read the data from `./input.csv` using a relative path.
The script must run end-to-end and produce `result.json` in the same folder.

OUTPUT FORMAT - STRICT:
Respond with exactly ONE fenced code block in {lang} and nothing else.
No prose before or after. The fenced block must start with ```{lang} and end
with ```."""


def tool_loop_prompt(task: Task, platform: str, prompt_mode: str) -> str:
    return f"""\
You are a careful econometrician executing an econometric task via Bash.

{dataset_context(task)}
Working directory contains `input.csv`.

{prompt_condition_block(prompt_mode, platform, task.task_id)}

Task: {task.prompt}

{platform_lang_hint(platform)}

{output_schema_blurb(task)}

You may use only the Bash tool. Use it to write one {platform} script, run it,
inspect execution errors, and iterate until `result.json` is produced.
{execution_hint(platform)} When done, your final assistant message should say
only "DONE" once `result.json` is present. Do not write any other files."""


def agentic_prompt(task: Task, platform: str, prompt_mode: str) -> str:
    return f"""\
You are a careful econometrician completing one benchmark task in a working
folder that contains `input.csv`.

{dataset_context(task)}

{prompt_condition_block(prompt_mode, platform, task.task_id)}

Task: {task.prompt}

{platform_lang_hint(platform)}

{output_schema_blurb(task)}

Produce `result.json` in this folder with the declared estimates. Use relative
paths to read `input.csv` and to write outputs. You may create and run whatever
script is useful for the task, and you should leave that script in the folder
with `result.json`."""


def repair_prompt(
    task: Task,
    platform: str,
    prompt_mode: str,
    previous_code: str,
    failure_context: str,
) -> str:
    lang = script_language(platform)
    return f"""\
You are repairing a {platform} script for an econometric task.

{dataset_context(task)}

{prompt_condition_block(prompt_mode, platform, task.task_id)}

Task: {task.prompt}

{platform_lang_hint(platform)}

{output_schema_blurb(task)}

The previous script was:
```{lang}
{previous_code}
```

It failed with this execution or artifact information:
```
{failure_context}
```

Return a corrected complete script. It must read ./input.csv and write
./result.json.

OUTPUT FORMAT - STRICT:
Respond with exactly ONE fenced code block in {lang} and nothing else."""


def extract_code(text: str) -> str | None:
    match = CODE_FENCE_RE.search(text)
    if match:
        return match.group(1)
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else None


def model_call_timeout() -> int:
    """The model-call timeout for the no_feedback and self_repair calls.

    ``TIMEOUT_NO_FEEDBACK`` unless the limit re-run wave overrides it. Script
    execution never reads this helper, so an override bounds only the call.
    """
    return (
        TIMEOUT_NO_FEEDBACK
        if MODEL_CALL_TIMEOUT_OVERRIDE is None
        else int(MODEL_CALL_TIMEOUT_OVERRIDE)
    )


def run_claude(
    prompt: str,
    cwd: Path,
    allow_tools: bool,
    max_turns: int | None,
    timeout: int,
    stream: bool = False,
) -> dict[str, Any]:
    # ``max_turns=None`` omits the flag, which is how the limit re-run wave
    # lifts the twelve-turn cap; every corpus call passes an integer.
    cmd = [
        claude_bin(),
        "-p",
        "--model",
        MODEL,
        "--output-format",
        "stream-json" if stream else "json",
        *([] if max_turns is None else ["--max-turns", str(max_turns)]),
        "--no-session-persistence",
        "--permission-mode",
        PERMISSION_MODE,
    ]
    if stream:
        # stream-json with -p requires --verbose; this preserves the full
        # turn-by-turn event log (every tool_use + tool_result) for the trace.
        cmd.append("--verbose")
    if allow_tools:
        cmd.extend(["--allowedTools", ",".join(TOOL_LOOP_ALLOWED_TOOLS)])
    else:
        cmd.extend(["--disallowedTools", ",".join(NO_FEEDBACK_DISALLOWED_TOOLS)])
    cmd.extend(CLAUDE_EXTRA_CLI_ARGS)

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        result: dict[str, Any] = {
            "ok": False,
            "error": "timeout",
            "command": cmd,
            "timeout_seconds": timeout,
            "wall_seconds": time.time() - start,
            "raw": None,
        }
        if stream:
            partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", "replace")
            if partial:
                result["raw_stdout"] = partial
        return result

    wall = time.time() - start
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        result = {
            "ok": False,
            "error": f"claude exited {proc.returncode}",
            "command": cmd,
            "wall_seconds": wall,
            "stderr_tail": stderr[-400:],
            "stdout_tail": stdout[-400:],
        }
        if stream:
            result["raw_stdout"] = stdout
        return result

    if stream:
        events = load_json_lines(stdout)
        parsed = final_stream_result(events) or final_json_object(stdout)
        num_tool_calls = recursive_tool_use_count(events)
        if not num_tool_calls and parsed:
            num_tool_calls = parsed.get("num_tool_calls") or parsed.get("tool_calls")
        num_turns = (parsed.get("num_turns") if parsed else None) or stream_turn_count(
            events
        )
    else:
        parsed = None
        for line in reversed(stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    parsed = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        num_turns = parsed.get("num_turns") if parsed else None
        num_tool_calls = (
            parsed.get("num_tool_calls") or parsed.get("tool_calls") if parsed else None
        )

    if parsed is None:
        result = {
            "ok": False,
            "error": "no JSON in stdout",
            "wall_seconds": wall,
            "stdout_tail": stdout[-400:],
        }
        if stream:
            result["raw_stdout"] = stdout
        return result

    result = {
        "ok": True,
        "command": cmd,
        "permission_mode": PERMISSION_MODE,
        "allowed_tools": TOOL_LOOP_ALLOWED_TOOLS if allow_tools else [],
        "disallowed_tools": [] if allow_tools else NO_FEEDBACK_DISALLOWED_TOOLS,
        "wall_seconds": wall,
        "session_id": parsed.get("session_id"),
        "duration_ms": parsed.get("duration_ms"),
        "num_turns": num_turns,
        "n_turns": num_turns,
        "n_tool_calls": num_tool_calls,
        "stop_reason": parsed.get("stop_reason"),
        "total_cost_usd": parsed.get("total_cost_usd"),
        "is_error": parsed.get("is_error"),
        "usage": parsed.get("usage", {}),
        "text": parsed.get("result", ""),
        "raw_meta": {
            key: parsed.get(key)
            for key in (
                "duration_api_ms",
                "ttft_ms",
                "terminal_reason",
                "modelUsage",
            )
        },
    }
    if stream:
        result["output_format"] = "stream-json"
        result["raw_stdout"] = stdout
        if stderr.strip():
            result["raw_stderr"] = stderr
    return result


def agentic_claude_cmd(cwd: Path, model: str = MODEL) -> list[str]:
    """Build the Claude Code command for R3 without executing it."""

    return [
        claude_bin(),
        "-p",
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--max-turns",
        str(MAX_TURNS_AGENTIC),
        "--no-session-persistence",
        "--permission-mode",
        AGENTIC_PERMISSION_MODE,
        "--allowedTools",
        AGENTIC_ALLOWED_TOOLS,
        "--add-dir",
        str(cwd),
    ]


def load_json_lines(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def recursive_tool_use_count(value: Any) -> int:
    if isinstance(value, dict):
        count = 1 if value.get("type") == "tool_use" else 0
        return count + sum(recursive_tool_use_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(recursive_tool_use_count(item) for item in value)
    return 0


def stream_turn_count(events: list[dict[str, Any]]) -> int | None:
    turns = sum(
        1
        for event in events
        if event.get("type") in {"assistant", "user"}
        or (event.get("message") or {}).get("role") in {"assistant", "user"}
    )
    return turns or None


def final_stream_result(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("type") == "result":
            return event
        if "total_cost_usd" in event or "num_turns" in event:
            return event
    return None


def final_json_object(text: str) -> dict[str, Any] | None:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def run_claude_agentic(
    prompt: str,
    cwd: Path,
    timeout: int,
    model: str = MODEL,
) -> dict[str, Any]:
    cmd = agentic_claude_cmd(cwd, model=model)
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        return {
            "ok": False,
            "error": "timeout",
            "permission_mode": AGENTIC_PERMISSION_MODE,
            "allowed_tools": [],
            "disallowed_tools": [],
            "output_format": "stream-json",
            "command": cmd,
            "wall_seconds": time.time() - start,
            "raw": None,
            "raw_stdout": partial,
        }

    wall = time.time() - start
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    events = load_json_lines(stdout)
    final = final_stream_result(events) or final_json_object(stdout)
    num_tool_calls = recursive_tool_use_count(events)
    if not num_tool_calls and final:
        num_tool_calls = int(
            final.get("num_tool_calls") or final.get("tool_calls") or 0
        )
    num_turns = (final.get("num_turns") if final else None) or stream_turn_count(events)

    base = {
        "permission_mode": AGENTIC_PERMISSION_MODE,
        "allowed_tools": [],
        "disallowed_tools": [],
        "output_format": "stream-json",
        "command": cmd,
        "wall_seconds": wall,
        "returncode": proc.returncode,
        "stream_events": len(events),
        "num_tool_calls": num_tool_calls,
        "n_tool_calls": num_tool_calls,
        "num_turns": num_turns,
        "n_turns": num_turns,
        "stderr_tail": stderr[-400:],
        "stdout_tail": stdout[-400:],
        "raw_stdout": stdout,
    }
    if stderr.strip():
        base["raw_stderr"] = stderr

    if final is None:
        base.update(
            {
                "ok": False,
                "error": "no JSON in stdout"
                if proc.returncode == 0
                else f"claude exited {proc.returncode}",
                "usage": {},
                "text": "",
            }
        )
        return base

    base.update(
        {
            "ok": proc.returncode == 0,
            "error": None
            if proc.returncode == 0
            else f"claude exited {proc.returncode}",
            "session_id": final.get("session_id"),
            "duration_ms": final.get("duration_ms"),
            "stop_reason": final.get("stop_reason"),
            "total_cost_usd": final.get("total_cost_usd"),
            "is_error": final.get("is_error"),
            "usage": final.get("usage", {}),
            "text": final.get("result", ""),
            "raw_meta": {
                key: final.get(key)
                for key in (
                    "duration_api_ms",
                    "ttft_ms",
                    "terminal_reason",
                    "modelUsage",
                )
            },
        }
    )
    return base


def stata_log_path(script_path: Path) -> Path:
    return script_path.with_suffix(".log")


def execute_script(
    script_path: Path,
    platform: str,
    cwd: Path,
    timeout: int,
) -> dict[str, Any]:
    if platform == "python":
        cmd = [sys.executable, str(script_path)]
    elif platform == "r":
        cmd = ["Rscript", "--vanilla", str(script_path)]
    elif platform == "stata":
        if not STATA_BIN.exists():
            return {"ok": False, "error": f"Stata binary not found: {STATA_BIN}"}
        # Batch flag differs by OS: Unix `-b do`, Windows `/e do`.
        batch_flag = "/e" if os.name == "nt" else "-b"
        cmd = [str(STATA_BIN), batch_flag, "do", str(script_path)]
    else:
        return {"ok": False, "error": f"unknown platform {platform}"}

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": "execution timeout",
            "wall_seconds": time.time() - start,
        }

    log_tail = ""
    if platform == "stata":
        log_path = stata_log_path(script_path)
        if log_path.exists():
            log_tail = log_path.read_text(errors="replace")[-3000:]

    ok = proc.returncode == 0
    if platform == "stata" and STATA_ERROR_RE.search(log_tail):
        ok = False

    return {
        "ok": ok,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1500:],
        "stderr_tail": proc.stderr[-1500:],
        "stata_log_tail": log_tail[-1500:] if log_tail else None,
        "wall_seconds": time.time() - start,
    }


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def claude_blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all saved Claude call blocks from a run record."""
    return [
        value
        for key, value in record.items()
        if key == "claude" or key.startswith("claude_attempt_")
        if isinstance(value, dict)
    ]


def is_zero_token_failure(block: dict[str, Any]) -> bool:
    """Detect Claude CLI failures that returned no model work or text."""
    if block.get("ok"):
        return False
    if block.get("total_cost_usd") not in (None, 0, 0.0):
        return False
    usage = block.get("usage")
    if isinstance(usage, dict):
        for value in usage.values():
            if finite_number(value) and float(value) > 0:
                return False

    haystack = " ".join(
        str(block.get(key, ""))
        for key in ("error", "stderr_tail", "stdout_tail", "text")
    ).lower()
    if '"output_tokens":0' in haystack or '"modelusage":{}' in haystack:
        return True
    return False


def is_provider_limit_block(block: dict[str, Any]) -> bool:
    if block.get("ok"):
        return False
    haystack = " ".join(
        str(block.get(key, "")) for key in ("error", "stderr_tail", "stdout_tail")
    ).lower()
    return any(
        pattern in haystack for pattern in PROVIDER_LIMIT_PATTERNS
    ) or is_zero_token_failure(block)


def is_provider_limit_record(record: dict[str, Any]) -> bool:
    """Detect records where Claude failed before producing a usable response."""
    blocks = claude_blocks(record)
    if not blocks:
        return False
    return any(is_provider_limit_block(block) for block in blocks)


def score(result_path: Path, task: Task) -> dict[str, Any]:
    """Simplified numeric scorer used for benchmark run records."""
    out = {
        "execution_success": None,
        "artifact_success": False,
        "method_success": None,
        "numeric_success": False,
        "integrity_success": True,
        "per_term": {},
        "notes": [],
    }

    if not result_path.exists():
        out["notes"].append("result.json missing")
        return out

    try:
        data = json.loads(result_path.read_text())
    except Exception as exc:
        out["notes"].append(f"result.json invalid JSON: {exc}")
        return out

    required = {"task_id", "platform", "estimates", "method_metadata", "diagnostics"}
    if not required.issubset(data.keys()):
        out["notes"].append(f"missing keys: {required - data.keys()}")
        return out

    if not isinstance(data.get("diagnostics"), dict):
        out["notes"].append("diagnostics is not an object")

    estimates_raw = data.get("estimates")
    if not isinstance(estimates_raw, list):
        out["notes"].append("estimates is not a list")
        return out

    estimates: dict[str, dict[str, Any]] = {}
    for row in estimates_raw:
        if not isinstance(row, dict):
            out["notes"].append("estimate row is not an object")
            continue
        term = row.get("term")
        if not isinstance(term, str):
            out["notes"].append("estimate row has missing/non-string term")
            continue
        estimates[term] = row

    if not all(term in estimates for term in task.estimands):
        out["notes"].append(f"missing terms: {set(task.estimands) - estimates.keys()}")
        return out

    if out["notes"]:
        return out

    if task.reference_key not in REFERENCE:
        out["notes"].append(f"reference key missing: {task.reference_key}")
        return out

    out["artifact_success"] = True
    ref = REFERENCE[task.reference_key]["estimates"]
    ref_by_term = {row["term"]: row for row in ref}

    all_pass = True
    for term in task.estimands:
        estimate_row = estimates[term]
        ref_row = ref_by_term[term]
        try:
            estimate = float(estimate_row["estimate"])
        except (KeyError, TypeError, ValueError):
            out["notes"].append(f"non-numeric estimate for {term}")
            all_pass = False
            continue

        ref_estimate = float(ref_row["estimate"])
        coef_err = abs(estimate - ref_estimate)
        coef_tol = max(COEF_REL_TOL * abs(ref_estimate), COEF_ABS_TOL)
        coef_pass = coef_err <= coef_tol

        got_se = estimate_row.get("se")
        ref_se = ref_row.get("se")
        se = float(got_se) if finite_number(got_se) else None
        ref_se_float = float(ref_se) if finite_number(ref_se) else None
        se_err = None
        se_pass = None
        if se is not None and ref_se_float is not None:
            se_err = abs(se - ref_se_float)
            se_tol = max(SE_REL_TOL * abs(ref_se_float), SE_ABS_TOL)
            se_pass = se_err <= se_tol

        out["per_term"][term] = {
            "ref_estimate": ref_estimate,
            "got_estimate": estimate,
            "coef_err": coef_err,
            "coef_pass": coef_pass,
            "ref_se": ref_se_float,
            "got_se": se,
            "se_err": se_err,
            "se_pass": se_pass,
        }
        if not coef_pass or se_pass is False:
            all_pass = False

    out["numeric_success"] = all_pass
    metadata = data.get("method_metadata", {})
    out["method_success"] = bool(metadata.get("estimator"))
    return out


def artifact_problem(result_path: Path) -> str | None:
    if not result_path.exists():
        return "result.json was not produced"
    try:
        data = json.loads(result_path.read_text())
    except Exception as exc:
        return f"result.json is invalid JSON: {exc}"
    required = {"task_id", "platform", "estimates"}
    missing = required - data.keys()
    if missing:
        return f"result.json is missing keys: {sorted(missing)}"
    if not isinstance(data.get("estimates"), list):
        return "result.json estimates field is not a list"
    return None


def failure_context(
    exec_res: dict[str, Any] | None,
    result_path: Path,
) -> str:
    parts = []
    if exec_res:
        parts.append(json.dumps(exec_res, indent=2, default=str)[-3000:])
    problem = artifact_problem(result_path)
    if problem:
        parts.append(problem)
    return "\n\n".join(parts)[-5000:] or "No failure details were captured."


def prepare_cell_dir(
    task: Task,
    platform: str,
    prompt_mode: str,
    exec_mode: str,
    rep: int,
    overwrite: bool,
    runs_dir: Path | None = None,
) -> tuple[str, str, Path, Path]:
    cell_id_base = f"{task.task_id}__{platform}__{prompt_mode}__{exec_mode}"
    cell_id = f"{cell_id_base}__rep{rep:02d}"
    cell_dir = (runs_dir if runs_dir is not None else RUNS_DIR) / cell_id
    if cell_dir.exists() and overwrite:
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)

    data_src = resolve_data_source(task.data_file)
    data_dst = cell_dir / "input.csv"
    if not data_src.exists():
        raise FileNotFoundError(f"missing task data: {data_src}")
    if overwrite or not data_dst.exists():
        shutil.copy2(data_src, data_dst)
    return cell_id, cell_id_base, cell_dir, data_src


def prepare_agentic_dir(cell_id: str, runs_dir: Path | None = None) -> Path:
    AGENTIC_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"{cell_id}__", dir=str(AGENTIC_RUN_ROOT)))
    input_src = (runs_dir if runs_dir is not None else RUNS_DIR) / cell_id / "input.csv"
    if not input_src.exists():
        raise FileNotFoundError(f"missing run-local input.csv: {input_src}")
    shutil.copy2(input_src, workdir / "input.csv")
    return workdir


def copy_back(workdir: Path, cell_dir: Path, platform: str | None = None) -> None:
    result_src = workdir / "result.json"
    if result_src.exists():
        shutil.copy2(result_src, cell_dir / "result.json")

    script_specs = {
        "python": ((".py",), "run.py"),
        "r": ((".r",), "run.R"),
        "stata": ((".do",), "run.do"),
    }
    copied_platforms: set[str] = set()
    for script_platform, (suffixes, dest_name) in script_specs.items():
        candidates = [
            path
            for path in workdir.iterdir()
            if path.is_file()
            and path.name != dest_name
            and path.suffix.lower() in suffixes
        ]
        canonical = workdir / dest_name
        if canonical.exists():
            candidates.append(canonical)
        if not candidates:
            continue
        script_src = sorted(
            candidates, key=lambda path: (path.stat().st_mtime, path.name)
        )[-1]
        shutil.copy2(script_src, cell_dir / dest_name)
        copied_platforms.add(script_platform)

    if platform is not None and platform not in copied_platforms:
        _, dest_name = script_specs[platform]
        ignored_names = {"input.csv", "result.json"}
        extensionless = [
            path
            for path in workdir.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.name not in ignored_names
            and path.suffix == ""
        ]
        if extensionless:
            script_src = sorted(
                extensionless, key=lambda path: (path.stat().st_mtime, path.name)
            )[-1]
            shutil.copy2(script_src, cell_dir / dest_name)

    for log_src in sorted(workdir.glob("*.log")):
        if log_src.is_file():
            shutil.copy2(log_src, cell_dir / log_src.name)


def persist_agent_stream(
    res: dict[str, Any],
    cell_dir: Path,
    stream_name: str = "agent_stream.jsonl",
    stderr_name: str = "agent_stderr.txt",
) -> None:
    """Write the full raw agent stream-json trace to disk and drop it from the
    record dict, so run_record.json stays lean while every tool_use,
    tool_result, and message is preserved on disk for full replay."""

    raw_stdout = res.pop("raw_stdout", "")
    raw_stderr = res.pop("raw_stderr", "")
    if raw_stdout:
        (cell_dir / stream_name).write_text(raw_stdout)
    if raw_stderr and raw_stderr.strip():
        (cell_dir / stderr_name).write_text(raw_stderr)


def resolve_data_source(data_file: str) -> Path:
    path = Path(data_file).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"data", "code"}:
        return PROJECT_ROOT / path
    candidates = [
        DATA_DIR / path,
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def display_path(path: Path) -> str:
    for base in (ROOT, PROJECT_ROOT):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def initial_prompt(task: Task, platform: str, prompt_mode: str, exec_mode: str) -> str:
    """The first prompt sent for a cell; hashed into the record for wave provenance."""
    if exec_mode == "tool_loop":
        return tool_loop_prompt(task, platform, prompt_mode)
    if exec_mode == "agentic":
        return agentic_prompt(task, platform, prompt_mode)
    return no_feedback_prompt(task, platform, prompt_mode)


def run_cell(
    task: Task,
    platform: str,
    prompt_mode: str,
    exec_mode: str,
    rep: int = 0,
    overwrite: bool = False,
    rerun_provider_limit: bool = False,
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    cell_id, cell_id_base, cell_dir, data_src = prepare_cell_dir(
        task,
        platform,
        prompt_mode,
        exec_mode,
        rep,
        overwrite,
        runs_dir=runs_dir,
    )
    record_path = cell_dir / "run_record.json"
    if record_path.exists() and not overwrite:
        record = json.loads(record_path.read_text())
        if not (rerun_provider_limit and is_provider_limit_record(record)):
            record["skipped_existing"] = True
            return record

        shutil.rmtree(cell_dir)
        cell_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_src, cell_dir / "input.csv")

    record: dict[str, Any] = {
        "cell_id": cell_id,
        "cell_id_base": cell_id_base,
        "rep": rep,
        "task_id": task.task_id,
        "reference_key": task.reference_key,
        "platform": platform,
        "exec_mode": exec_mode,
        "prompt_mode": prompt_mode,
        "model": MODEL,
        "data_source": display_path(data_src),
        "uuid": str(uuid.uuid4()),
        # Wave provenance (added 2026-07-04): records written before this date
        # lack both fields and are the pre-pin legacy wave.
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_sha256": hashlib.sha256(
            initial_prompt(task, platform, prompt_mode, exec_mode).encode("utf-8")
        ).hexdigest(),
    }
    agentic_workdir: Path | None = None

    if exec_mode == "no_feedback":
        prompt = no_feedback_prompt(task, platform, prompt_mode)
        claude_res = run_claude(
            prompt,
            cwd=cell_dir,
            allow_tools=False,
            max_turns=1,
            timeout=model_call_timeout(),
        )
        record["claude"] = claude_res

        if not claude_res.get("ok"):
            record["execution_success"] = False
            record["scoring"] = {"notes": ["claude call failed"]}
            return record

        code = extract_code(claude_res["text"])
        if not code:
            record["execution_success"] = False
            record["scoring"] = {"notes": ["no code block extracted"]}
            record["raw_text_tail"] = claude_res["text"][-400:]
            return record

        script_path = cell_dir / script_name(platform)
        script_path.write_text(code)
        exec_res = execute_script(
            script_path,
            platform,
            cell_dir,
            timeout=TIMEOUT_NO_FEEDBACK,
        )
        record["execution"] = exec_res
        record["execution_success"] = exec_res["ok"]

    elif exec_mode == "self_repair":
        prompt = no_feedback_prompt(task, platform, prompt_mode)
        first_claude = run_claude(
            prompt,
            cwd=cell_dir,
            allow_tools=False,
            max_turns=1,
            timeout=model_call_timeout(),
        )
        record["claude_attempt_1"] = first_claude
        if not first_claude.get("ok"):
            record["execution_success"] = False
            record["scoring"] = {"notes": ["initial claude call failed"]}
            return record

        first_code = extract_code(first_claude["text"])
        if not first_code:
            record["execution_success"] = False
            record["scoring"] = {"notes": ["no initial code block extracted"]}
            record["raw_text_tail"] = first_claude["text"][-400:]
            return record

        first_script = cell_dir / f"attempt1_{script_name(platform)}"
        first_script.write_text(first_code)
        first_exec = execute_script(
            first_script,
            platform,
            cell_dir,
            timeout=TIMEOUT_NO_FEEDBACK,
        )
        record["execution_attempt_1"] = first_exec

        result_path = cell_dir / "result.json"
        previous_code = first_code
        previous_exec = first_exec
        record["execution_success"] = (
            first_exec["ok"] and artifact_problem(result_path) is None
        )
        for attempt in range(2, MAX_EXECUTIONS_REPAIR + 1):
            if record["execution_success"]:
                break

            repair_res = run_claude(
                repair_prompt(
                    task,
                    platform,
                    prompt_mode,
                    previous_code=previous_code,
                    failure_context=failure_context(previous_exec, result_path),
                ),
                cwd=cell_dir,
                allow_tools=False,
                max_turns=1,
                timeout=model_call_timeout(),
            )
            record[f"claude_attempt_{attempt}"] = repair_res
            if not repair_res.get("ok"):
                record["execution_success"] = False
                record["scoring"] = {
                    "notes": [f"repair attempt {attempt} claude call failed"]
                }
                return record

            repaired_code = extract_code(repair_res["text"])
            if not repaired_code:
                record["execution_success"] = False
                record["scoring"] = {
                    "notes": [f"no repair attempt {attempt} code block extracted"]
                }
                record[f"raw_repair_attempt_{attempt}_text_tail"] = repair_res["text"][
                    -400:
                ]
                return record

            repaired_script = cell_dir / f"attempt{attempt}_{script_name(platform)}"
            repaired_script.write_text(repaired_code)
            repaired_exec = execute_script(
                repaired_script,
                platform,
                cell_dir,
                timeout=TIMEOUT_NO_FEEDBACK,
            )
            record[f"execution_attempt_{attempt}"] = repaired_exec
            previous_code = repaired_code
            previous_exec = repaired_exec
            record["execution_success"] = (
                repaired_exec["ok"] and artifact_problem(result_path) is None
            )

    elif exec_mode == "tool_loop":
        prompt = tool_loop_prompt(task, platform, prompt_mode)
        claude_res = run_claude(
            prompt,
            cwd=cell_dir,
            allow_tools=True,
            max_turns=MAX_TURNS_TOOL if TOOL_LOOP_TURN_CAP_ENABLED else None,
            timeout=TIMEOUT_TOOL,
            stream=True,
        )
        persist_agent_stream(claude_res, cell_dir)
        record["claude"] = claude_res
        record["execution_success"] = (
            bool(claude_res.get("ok")) and (cell_dir / "result.json").exists()
        )
    elif exec_mode == "agentic":
        prompt = agentic_prompt(task, platform, prompt_mode)
        agentic_workdir = prepare_agentic_dir(cell_id, runs_dir=runs_dir)
        final_res: dict[str, Any] = {}
        final_problem: str | None = None
        final_attempt = 0
        for attempt in range(1, AGENTIC_MAX_REINVOKE + 1):
            res = run_claude_agentic(
                prompt,
                cwd=agentic_workdir,
                timeout=TIMEOUT_AGENTIC,
                model=MODEL,
            )
            persist_agent_stream(
                res,
                cell_dir,
                stream_name=f"agent_stream_attempt{attempt}.jsonl",
                stderr_name=f"agent_stderr_attempt{attempt}.txt",
            )
            copy_back(agentic_workdir, cell_dir, platform=platform)
            final_problem = artifact_problem(cell_dir / "result.json")
            final_attempt = attempt
            if final_problem is None or attempt == AGENTIC_MAX_REINVOKE:
                final_res = res
                break
            record[f"claude_attempt_{attempt}"] = res
            record.setdefault("agentic_artifact_problems", []).append(
                {"attempt": attempt, "problem": final_problem}
            )
        record["claude"] = final_res
        record["agentic_attempts"] = final_attempt
        if final_problem:
            record["agentic_final_artifact_problem"] = final_problem
        record["execution_success"] = (
            bool(final_res.get("ok")) and final_problem is None
        )
    else:
        raise ValueError(exec_mode)

    try:
        scoring = score(cell_dir / "result.json", task)
        scoring["execution_success"] = record["execution_success"]
        record["scoring"] = scoring
        return record
    finally:
        if agentic_workdir is not None:
            shutil.rmtree(agentic_workdir, ignore_errors=True)


def selected_cells(
    task_ids: list[str] | None = None,
    platforms: list[str] | None = None,
    prompt_modes: list[str] | None = None,
    exec_modes: list[str] | None = None,
    max_cells: int | None = None,
    task_catalog: list[Task] | None = None,
) -> list[tuple[Task, str, str, str]]:
    catalog = task_catalog or TASKS
    tasks_by_id = {task.task_id: task for task in catalog}
    if task_ids:
        missing = sorted(set(task_ids) - tasks_by_id.keys())
        if missing:
            raise ValueError(f"unknown task ids: {missing}")
        tasks = [tasks_by_id[task_id] for task_id in task_ids]
    else:
        tasks = catalog

    chosen_platforms = platforms or PLATFORMS
    bad_platforms = sorted(set(chosen_platforms) - set(PLATFORMS))
    if bad_platforms:
        raise ValueError(f"unknown platforms: {bad_platforms}")

    chosen_prompts = prompt_modes or PROMPT_MODES
    bad_prompts = sorted(set(chosen_prompts) - set(PROMPT_MODES))
    if bad_prompts:
        raise ValueError(f"unknown prompt modes: {bad_prompts}")

    # R3 is intentionally opt-in because it uses a broader tool surface.
    chosen_modes = exec_modes or DEFAULT_EXEC_MODES
    bad_modes = sorted(set(chosen_modes) - set(EXEC_MODES))
    if bad_modes:
        raise ValueError(f"unknown execution modes: {bad_modes}")

    cells = [
        (task, platform, prompt_mode, mode)
        for task in tasks
        for platform in chosen_platforms
        for prompt_mode in chosen_prompts
        for mode in chosen_modes
    ]
    return cells[:max_cells] if max_cells is not None else cells


def validate_schedule(
    cells: list[tuple[Task, str, str, str]],
    n_reps: int,
) -> dict[str, Any]:
    missing_data = []
    missing_refs = []
    for task, _, _, _ in cells:
        data_path = resolve_data_source(task.data_file)
        if not data_path.exists():
            missing_data.append(display_path(data_path))
        if task.reference_key not in REFERENCE:
            missing_refs.append(task.reference_key)

    payload = {
        "tasks": len({task.task_id for task, _, _, _ in cells}),
        "cells": len(cells),
        "n_reps": n_reps,
        "total_runs": len(cells) * n_reps,
        "platforms": sorted({platform for _, platform, _, _ in cells}),
        "prompt_modes": sorted({prompt for _, _, prompt, _ in cells}),
        "exec_modes": sorted({mode for _, _, _, mode in cells}),
        "missing_data": sorted(set(missing_data)),
        "missing_references": sorted(set(missing_refs)),
    }
    payload["ok"] = not payload["missing_data"] and not payload["missing_references"]
    return payload


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats so run records stay valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [json_safe(inner) for inner in value]
    return value


def write_run_record(record: dict[str, Any], runs_dir: Path | None = None) -> None:
    cell_dir = (runs_dir if runs_dir is not None else RUNS_DIR) / record["cell_id"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "run_record.json").write_text(
        json.dumps(json_safe(record), indent=2, default=str, allow_nan=False)
    )


def write_summary(
    summary: list[dict[str, Any]],
    summary_file: str,
    runs_dir: Path | None = None,
) -> Path:
    # ``summary_file`` is relative to the runs dir and normally points into the
    # arm's ``_working`` directory, so create the parent before writing.
    summary_path = (runs_dir if runs_dir is not None else RUNS_DIR) / summary_file
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, default=str, allow_nan=False)
    )
    return summary_path

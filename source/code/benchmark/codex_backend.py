"""Codex-exec backend for the offline measurement pilot (SEPARATE ARM).

This module runs OpenAI's ``codex exec`` as an alternative code-generation
backend and returns a **provider-neutral** run block (never the ``claude``
schema). It is used ONLY by ``cells.py`` for the isolated
``codex_pilot`` arm; it must never touch the pinned ``claude-sonnet-4-6``
corpus or the central queue.

Design decisions (Codex-reviewed 2026-07-10, see
``archive/quality_reports_2026-08-10/plans/2026-07-10_codex-and-haiku-replication-arms.md``):

- ``codex exec`` hangs unless stdin is detached → ``stdin=subprocess.DEVNULL``.
- ``--ignore-user-config`` + ``--ignore-rules`` for a clean, reproducible run
  (the user's config loads MCP servers, a personality, and xhigh reasoning).
- ``--ephemeral`` so pilot runs do not persist session files under ``~/.codex``.
- ``--json`` gives typed events; ``turn.completed.usage`` carries token counts,
  and (when present) ``token_count`` / ``rate_limits`` events carry the
  message-based usage window. No per-call USD on a ChatGPT subscription →
  ``cost_usd`` is ``None`` (report NA, never $0).
- ``-o`` writes the final agent message (CLI-side, works under read-only).
- Codex is an agent even under ``--sandbox read-only``; it is NOT a no-tools
  chatbot. We audit tool calls from the event stream (allowlist of known tool
  item types) and flag any "generator" run that used tools.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

CODEX_BIN = os.environ.get("CODEX_BIN") or shutil.which("codex") or "codex"

# Base flags applied to every codex exec invocation in the pilot. Kept explicit
# so every run record can reproduce the exact contract.
BASE_FLAGS = [
    "exec",
    "--json",
    "--ignore-user-config",
    "--ignore-rules",
    "--ephemeral",
    "--skip-git-repo-check",
]

# item.completed item types that ARE tool/command activity (allowlist). Anything
# outside this set and outside NON_TOOL_ITEM_TYPES is surfaced as "unknown" for
# manual audit rather than silently counted as a tool call (Codex review #11).
KNOWN_TOOL_ITEM_TYPES = {
    "command_execution",
    "local_shell_call",
    "exec_command",
    "function_call",
    "mcp_tool_call",
    "custom_tool_call",
    "web_search",
    "file_change",
    "patch_apply",
    "apply_patch",
    "tool_call",
}
NON_TOOL_ITEM_TYPES = {
    "agent_message",
    "assistant_message",
    "reasoning",
    "todo_list",
    "plan_update",
    "error",
}

_VERSION_CACHE: str | None = None


def codex_version() -> str:
    """Return ``codex --version`` output (cached), or ``unknown:<err>``."""

    global _VERSION_CACHE
    if _VERSION_CACHE is not None:
        return _VERSION_CACHE
    try:
        out = subprocess.run(
            [CODEX_BIN, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        _VERSION_CACHE = (out.stdout or out.stderr or "").strip() or "unknown"
    except Exception as exc:  # noqa: BLE001 - best-effort probe
        _VERSION_CACHE = f"unknown:{exc}"
    return _VERSION_CACHE


def sandbox_for(condition: str) -> str:
    """Map a pilot condition to a codex sandbox policy.

    ``codex_generate`` / ``codex_repair`` are pure generators (no execution) →
    ``read-only``. ``codex_agent`` must write ``result.json`` → ``workspace-write``.
    """

    if condition in ("codex_generate", "codex_repair"):
        return "read-only"
    if condition == "codex_agent":
        return "workspace-write"
    raise ValueError(f"unknown codex condition: {condition}")


def _find_rate_limits(obj: Any) -> dict[str, Any] | None:
    """Best-effort search for a ``rate_limits`` mapping anywhere in an event.

    Codex 0.144.0 may surface message-window usage under ``rate_limits`` (e.g.
    inside a ``token_count`` event's ``info``). The exact shape is version
    dependent, so we capture whatever mapping is present for later analysis.
    """

    if isinstance(obj, dict):
        rl = obj.get("rate_limits")
        if isinstance(rl, dict):
            return rl
        for value in obj.values():
            found = _find_rate_limits(value)
            if found is not None:
                return found
    return None


def parse_events(stdout: str) -> dict[str, Any]:
    """Parse codex ``--json`` JSONL stdout into a summary.

    Returns the final agent text, token usage, event/tool counts, distinct
    item/event types, any unknown item types, and the last ``rate_limits``
    mapping seen. Robust to malformed lines and to a missing usage event.
    """

    agent_texts: list[str] = []
    usage: dict[str, Any] | None = None
    rate_limits: dict[str, Any] | None = None
    error_msgs: list[str] = []
    num_events = 0
    num_tool_calls = 0
    item_types: dict[str, int] = {}
    event_types: dict[str, int] = {}
    unknown_item_types: set[str] = set()
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        num_events += 1
        etype = obj.get("type")
        if isinstance(etype, str):
            event_types[etype] = event_types.get(etype, 0) + 1
        if etype == "item.completed":
            item = obj.get("item") or {}
            itype = item.get("type")
            if isinstance(itype, str):
                item_types[itype] = item_types.get(itype, 0) + 1
                if itype in ("agent_message", "assistant_message"):
                    text = item.get("text")
                    if isinstance(text, str):
                        agent_texts.append(text)
                elif itype in KNOWN_TOOL_ITEM_TYPES:
                    num_tool_calls += 1
                elif itype not in NON_TOOL_ITEM_TYPES:
                    unknown_item_types.add(itype)
        elif etype == "turn.completed":
            maybe_usage = obj.get("usage")
            if isinstance(maybe_usage, dict):
                usage = maybe_usage
        elif etype in ("error", "turn.failed"):
            msg = obj.get("message")
            if not msg and isinstance(obj.get("error"), dict):
                msg = obj["error"].get("message")
            if isinstance(msg, str) and msg:
                error_msgs.append(msg)
        found_rl = _find_rate_limits(obj)
        if found_rl is not None:
            rate_limits = found_rl
    return {
        "agent_text": "\n".join(agent_texts).strip(),
        "usage": usage,
        "rate_limits": rate_limits,
        "error_message": "; ".join(error_msgs),
        "num_events": num_events,
        "num_tool_calls": num_tool_calls,
        "item_types": item_types,
        "event_types": event_types,
        "unknown_item_types": sorted(unknown_item_types),
    }


# Substrings that mark a ChatGPT/codex usage-limit (message-budget) rejection.
USAGE_LIMIT_MARKERS = ("usage limit", "rate limit", "try again at", "quota", "purchase more credits")


def is_usage_limit(text: str) -> bool:
    t = (text or "").lower()
    return any(marker in t for marker in USAGE_LIMIT_MARKERS)


def _read_last_message(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def run_codex(
    prompt: str,
    cwd: Path,
    condition: str,
    model: str,
    reasoning: str,
    timeout: int,
    event_log_path: Path,
    last_message_path: Path,
) -> dict[str, Any]:
    """Run one ``codex exec`` invocation and return a provider-neutral block.

    Parameters
    ----------
    prompt:
        The full task/repair prompt (passed as a positional arg; stdin detached).
    cwd:
        Working directory codex runs in. For generator conditions this SHOULD be
        an empty temp dir (no ``input.csv``) so codex cannot read the data —
        matching Claude ``no_feedback`` which is blind to the dataset. For
        ``codex_agent`` it contains ``input.csv``.
    condition:
        ``codex_generate`` | ``codex_repair`` | ``codex_agent``.
    model, reasoning:
        e.g. ``gpt-5.4`` and ``medium`` (recorded per run).
    timeout:
        Wall-clock seconds; on expiry partial output is persisted and flagged.
    event_log_path, last_message_path:
        Where the raw JSONL event stream and the final agent message are written.
    """

    sandbox = sandbox_for(condition)
    argv = [
        CODEX_BIN,
        *BASE_FLAGS,
        "-m",
        model,
        "-c",
        f"model_reasoning_effort={reasoning}",
        "--sandbox",
        sandbox,
        "-C",
        str(cwd),
        "-o",
        str(last_message_path),
        prompt,
    ]
    # Reproducible argv with the (non-secret but bulky) prompt elided.
    argv_record = argv[:-1] + ["<PROMPT>"]

    start = time.time()
    timed_out = False
    stdout = ""
    stderr = ""
    returncode: int | None = None
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)

    wall = time.time() - start

    # Persist the raw event stream for full replay.
    try:
        event_log_path.write_text(stdout)
    except OSError:
        pass

    parsed = parse_events(stdout)
    text = _read_last_message(last_message_path) or parsed["agent_text"]
    result_json_existed = (cwd / "result.json").exists()

    ok = (returncode == 0) and not timed_out

    error_message = parsed.get("error_message", "")
    provider_limit = is_usage_limit(error_message) or is_usage_limit(stderr)

    usage = parsed["usage"] or {}
    return {
        "backend": "codex",
        "ok": ok,
        "text": text,
        "wall_seconds": round(wall, 3),
        "timed_out": timed_out,
        "exit_code": returncode,
        "provider_limit": provider_limit,
        "error_message": error_message,
        "cost_usd": None,  # subscription: no per-call USD (report NA, not $0)
        "usage": {
            "input_tokens": usage.get("input_tokens"),
            "cached_input_tokens": usage.get("cached_input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
        },
        "rate_limits": parsed["rate_limits"],
        "num_events": parsed["num_events"],
        "num_tool_calls": parsed["num_tool_calls"],
        "item_types": parsed["item_types"],
        "event_types": parsed["event_types"],
        "unknown_item_types": parsed["unknown_item_types"],
        "used_tools": parsed["num_tool_calls"] > 0,
        "result_json_existed_at_kill": result_json_existed if timed_out else None,
        "stderr_tail": stderr[-1200:],
        "event_log_path": str(event_log_path),
        "last_message_path": str(last_message_path),
        # Reproducibility metadata (Codex review #18):
        "model": model,
        "reasoning_effort": reasoning,
        "sandbox": sandbox,
        "condition": condition,
        "timeout_seconds": timeout,
        "codex_version": codex_version(),
        "codex_bin": CODEX_BIN,
        # env var NAMES (not values) that could influence a codex run, recorded
        # for reproducibility (config files are neutralized by --ignore-*).
        "env_influencing": sorted(
            k for k in os.environ if k.startswith(("CODEX_", "OPENAI_"))
        ),
        "argv": argv_record,
    }


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)

"""Constrained-agent condition through mini-swe-agent (the ``or_agent_swe`` condition).

This backend replays the ``or_agent`` design through mini-swe-agent, a
standard, pip-pinned open agent harness, instead of the project's own
``openrouter_backend.run_tool_loop``. The equivalence targets are the
caps and the task content, not the protocol text, which is
harness-specific by construction:

- same task content blocks (dataset context, prompt condition, task,
  platform hint, output schema) built from the same ``harness`` functions;
- step cap 12 = ``openrouter_backend.DEFAULT_MAX_TURNS``;
- wall cap 600 s = ``openrouter_backend.DEFAULT_AGENT_WALL``;
- per-command timeout 240 s = ``openrouter_backend.DEFAULT_BASH_TIMEOUT``;
- observation clip 8,000 chars = ``openrouter_backend.TOOL_OUTPUT_MAX_CHARS``;
- secret env vars are masked in the command environment, mirroring
  ``openrouter_backend._bash_env``.

The submit convention is mini-swe-agent's own
(``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``) in place of the
``or_agent`` "final message says DONE" convention. The full mini config is
saved next to each run record for provenance. Per-call billed cost is not
captured on this path (litellm has no price entry for the model); usage
must come from provider records if needed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import harness
import openrouter_backend

SWE_MAX_STEPS = openrouter_backend.DEFAULT_MAX_TURNS
SWE_WALL_CAP = openrouter_backend.DEFAULT_AGENT_WALL
SWE_BASH_TIMEOUT = openrouter_backend.DEFAULT_BASH_TIMEOUT
SWE_OUTPUT_CLIP = openrouter_backend.TOOL_OUTPUT_MAX_CHARS
SWE_SUBPROCESS_GRACE = 120  # extra seconds before the CLI subprocess is killed

SYSTEM_TEMPLATE = "You are a careful econometrician executing an econometric task via Bash.\n"

_MINI_BIN: str | None = None


def mini_bin() -> str:
    global _MINI_BIN
    if _MINI_BIN is None:
        found = shutil.which("mini-swe-agent")
        if not found:
            raise RuntimeError("mini-swe-agent not found on PATH (uv tool install mini-swe-agent)")
        _MINI_BIN = found
    return _MINI_BIN


def mini_version() -> str:
    try:
        out = subprocess.run(
            [mini_bin(), "--help"], capture_output=True, text=True, timeout=30, check=False,
            stdin=subprocess.DEVNULL,
        )
        for line in out.stdout.splitlines():
            if "version" in line.lower():
                return line.strip()
    except Exception:  # noqa: BLE001 - version string is best-effort provenance
        pass
    return "unknown"


def swe_tool_loop_prompt(task: Any, platform: str, prompt_mode: str) -> str:
    """Task content identical to ``harness.tool_loop_prompt`` with the
    protocol paragraph adapted to mini-swe-agent's conventions."""

    return f"""\
{harness.dataset_context(task)}
Working directory contains `input.csv`.

{harness.prompt_condition_block(prompt_mode, platform, task.task_id)}

Task: {task.prompt}

{harness.platform_lang_hint(platform)}

{harness.output_schema_blurb(task)}

You may use only bash commands. Use them to write one {platform} script, run it,
inspect execution errors, and iterate until `result.json` is produced.
{harness.execution_hint(platform)}

## Command execution rules

- Every response must include exactly one bash tool call.
- Each command runs in a fresh subshell in the working directory; directory and
  environment changes do not persist between commands.
- Once `result.json` is present, submit by issuing exactly
  `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` with no other command.
- Do not write any other files besides the script, its logs, and `result.json`."""


def _masked_env_entries() -> dict[str, str]:
    """Blank every secret-prefixed env var for model-executed commands,
    mirroring ``openrouter_backend._bash_env``. mini's LocalEnvironment merges
    these over ``os.environ`` for each command."""

    masked = {
        k: ""
        for k in os.environ
        if k.startswith(openrouter_backend.SECRET_ENV_PREFIXES)
    }
    masked["OPENROUTER_API_KEY"] = ""
    masked["PAGER"] = "cat"
    masked["MANPAGER"] = "cat"
    return masked


def build_config(max_steps: int, wall_cap: int, bash_timeout: int) -> dict[str, Any]:
    half = SWE_OUTPUT_CLIP // 2
    observation = (
        "{%- if output.output | length < " + str(SWE_OUTPUT_CLIP) + " -%}\n"
        "{\n"
        '  "returncode": {{ output.returncode }},\n'
        '  "output": {{ output.output | tojson }}\n'
        "  {%- if output.exception_info %}, \"exception_info\": {{ output.exception_info | tojson }}{% endif %}\n"
        "}\n"
        "{%- else -%}\n"
        "{\n"
        '  "returncode": {{ output.returncode }},\n'
        f'  "output_head": {{{{ output.output[:{half}] | tojson }}}},\n'
        f'  "output_tail": {{{{ output.output[-{half}:] | tojson }}}},\n'
        f'  "elided_chars": {{{{ output.output | length - {SWE_OUTPUT_CLIP} }}}},\n'
        '  "warning": "Output too long."\n'
        "  {%- if output.exception_info %}, \"exception_info\": {{ output.exception_info | tojson }}{% endif %}\n"
        "}\n"
        "{%- endif -%}"
    )
    format_error = (
        "Tool call error:\n<error>\n{{error}}\n</error>\n"
        "Every response must call the 'bash' tool exactly once with "
        '{"command": "..."}.\n'
        "To finish, issue: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` alone."
    )
    return {
        "agent": {
            "system_template": SYSTEM_TEMPLATE,
            "instance_template": "{{task}}",
            "step_limit": max_steps,
            "cost_limit": 0,
            "wall_time_limit_seconds": wall_cap,
            "mode": "yolo",
        },
        "environment": {
            "timeout": bash_timeout,
            "env": _masked_env_entries(),
        },
        "model": {
            "observation_template": observation,
            "format_error_template": format_error,
            "model_kwargs": {"drop_params": True},
        },
    }


def classify_outcome(
    exit_status: str,
    timed_out: bool,
    proc_error: str,
    max_steps: int,
    wall_cap: int,
) -> tuple[bool, bool, str]:
    """Map a mini-swe-agent terminal state onto the run_tool_loop semantics.

    ``ok`` means the provider interaction completed without an API or harness
    error. A step-cap exhaustion (``LimitsExceeded``) is a completed, failing
    model response, exactly like the or_agent turn cap, so it stays ``ok``.
    A wall-cap timeout is not ok and pairs with ``timed_out=True`` so the
    analysis loader forms an intention-to-treat failure row instead of
    excluding the record as a provider interruption.
    """

    submitted = exit_status == "Submitted"
    ok = (
        exit_status in ("Submitted", "LimitsExceeded")
        and not timed_out
        and not proc_error
    )
    if submitted and ok:
        error_message = ""
    elif exit_status == "LimitsExceeded" and ok:
        error_message = f"step cap reached ({max_steps})"
    elif timed_out:
        error_message = proc_error or f"wall cap reached ({wall_cap}s)"
    else:
        error_message = proc_error or exit_status or "mini-swe-agent run failed"
    return ok, submitted, error_message


def run_swe_agent(
    prompt: str,
    workdir: Path,
    model: str,
    max_steps: int = SWE_MAX_STEPS,
    wall_cap: int = SWE_WALL_CAP,
    bash_timeout: int = SWE_BASH_TIMEOUT,
) -> dict[str, Any]:
    """Run one constrained-agent cell through the mini-swe-agent CLI.

    Returns a block shaped like ``openrouter_backend.run_tool_loop``'s return
    so the record layout stays comparable; the message transcript is under
    ``"transcript"`` for the caller to persist, and the rendered mini config
    is under ``"config"``.
    """

    config = build_config(max_steps, wall_cap, bash_timeout)
    config_path = workdir / ".swe_config.yaml"
    config_path.write_text(json.dumps(config, indent=1))  # YAML 1.2 accepts JSON
    traj_path = workdir / ".swe_trajectory.json"

    env = dict(os.environ)
    env["OPENROUTER_API_KEY"] = openrouter_backend.resolve_api_key()
    env["MSWEA_CONFIGURED"] = "true"
    env["MSWEA_COST_TRACKING"] = "ignore_errors"

    start = time.time()
    timed_out = False
    proc_error = ""
    try:
        proc = subprocess.run(
            [
                mini_bin(),
                "-m", f"openrouter/{model}",
                "-t", prompt,
                "-y",
                "-c", str(config_path),
                "-o", str(traj_path),
                "--exit-immediately",
            ],
            cwd=str(workdir),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=wall_cap + SWE_SUBPROCESS_GRACE,
            check=False,
        )
        cli_tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        timed_out = True
        proc_error = f"mini-swe-agent subprocess killed after {wall_cap + SWE_SUBPROCESS_GRACE}s"
        cli_tail = ""
    wall = time.time() - start

    exit_status = ""
    api_calls = 0
    transcript: list[dict[str, Any]] | None = None
    if traj_path.exists():
        try:
            traj = json.loads(traj_path.read_text())
            info = traj.get("info") or {}
            exit_status = str(info.get("exit_status") or "")
            api_calls = int((info.get("model_stats") or {}).get("api_calls") or 0)
            transcript = traj.get("messages")
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            proc_error = proc_error or f"trajectory unreadable: {exc}"

    if exit_status in ("TimeExceeded",):
        timed_out = True
    ok, submitted, error_message = classify_outcome(
        exit_status, timed_out, proc_error, max_steps, wall_cap
    )
    if not ok and not exit_status and cli_tail and not timed_out:
        error_message += f" | cli tail: {cli_tail[-400:]}"

    return {
        "backend": "openrouter",
        "harness": "mini-swe-agent",
        "harness_version": mini_version(),
        "ok": ok,
        "submitted": submitted,
        "exit_status": exit_status,
        "text": "",
        "wall_seconds": round(wall, 3),
        "timed_out": timed_out,
        "provider_limit": False,
        "error_message": error_message,
        "cost_usd": None,
        "total_cost_usd": None,
        "usage": {},
        "num_turns": api_calls,
        "num_tool_calls": api_calls,
        "num_api_calls": api_calls,
        "model_requested": model,
        "model_used": None,
        "provider_name": None,
        "max_turns": max_steps,
        "wall_cap_seconds": wall_cap,
        "bash_timeout_seconds": bash_timeout,
        "transcript": transcript,
        "config": config,
    }

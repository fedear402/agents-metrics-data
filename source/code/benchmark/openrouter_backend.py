"""OpenRouter chat-completions backend for the offline measurement pilot (SEPARATE ARM).

This module sends traffic directly to OpenRouter's OpenAI-compatible
``/chat/completions`` endpoint and returns **provider-neutral** run blocks
(mirroring ``codex_backend.py``, never the ``claude`` schema). It is used ONLY
by ``cells.py`` for isolated ``openrouter_<model>`` arms; it must
never touch the pinned ``claude-sonnet-4-6`` corpus or the central queue.

Three capabilities:

- ``chat_call``: one completion (used for the generate/repair conditions).
- ``run_tool_loop``: a minimal Bash-only agent loop (OpenAI tool calling, one
  ``bash`` tool, ``max_turns`` assistant-turn cap, total wall cap) mirroring the
  corpus ``tool_loop`` condition (Bash-only + 12-turn cap) with no CLI scaffold
  on top — the cheapest possible agent rung.
- Native cost telemetry: every request sets ``"usage": {"include": true}`` so
  the response carries OpenRouter's billed ``usage.cost`` in USD. ``cost_usd``
  is therefore measured, not price-table math (per the operational-numbers rule).

Key resolution handles the Claude-Code subprocess env scrub: ``OPENROUTER_API_KEY``
from the environment first, then the macOS keychain item ``openrouter-api-key``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_URL = "https://openrouter.ai/api/v1/key"
KEYCHAIN_SERVICE = "openrouter-api-key"

DEFAULT_CALL_TIMEOUT = 300  # seconds per API call (network read cap)
MAX_RETRIES = 4  # bounded backoff on 429/5xx/transport errors
BACKOFF_BASE_SECONDS = 2.0

# Corpus tool_loop contract: Bash-only, 12 assistant turns, 600 s wall.
DEFAULT_MAX_TURNS = 12
DEFAULT_AGENT_WALL = 600
DEFAULT_BASH_TIMEOUT = 240
TOOL_OUTPUT_MAX_CHARS = 8000

BASH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the working directory and return its "
                "exit code, stdout, and stderr."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]

# Env var names stripped from bash-tool subprocesses (model-authored commands
# must not be able to read operator credentials).
SECRET_ENV_PREFIXES = (
    "OPENROUTER_",
    "ANTHROPIC_",
    "OPENAI_",
    "AGENTS_METRICS_",
    "AWS_",
    "GOOGLE_",
    "GEMINI_",
)

_KEY_CACHE: str | None = None


def resolve_api_key() -> str:
    """``OPENROUTER_API_KEY`` env var, else the macOS keychain item.

    The keychain fallback matters because Claude-Code sessions scrub subprocess
    environments (``CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1``).
    """

    global _KEY_CACHE
    if _KEY_CACHE:
        return _KEY_CACHE
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        try:
            out = subprocess.run(
                ["security", "find-generic-password", "-w", "-s", KEYCHAIN_SERVICE],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            key = (out.stdout or "").strip()
        except Exception:  # noqa: BLE001 - fall through to the error below
            key = ""
    if not key:
        raise RuntimeError(
            "no OpenRouter key: set OPENROUTER_API_KEY or add keychain item "
            f"'{KEYCHAIN_SERVICE}'"
        )
    _KEY_CACHE = key
    return key


def key_usage() -> dict[str, Any]:
    """Read key-level usage/limit from OpenRouter (the budget ledger probe)."""

    req = urllib.request.Request(
        KEY_URL, headers={"Authorization": f"Bearer {resolve_api_key()}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    d = data.get("data") or {}
    return {
        "usage": d.get("usage"),
        "limit": d.get("limit"),
        "limit_remaining": d.get("limit_remaining"),
    }


def _post(payload: dict[str, Any], timeout: int) -> tuple[int, dict[str, Any] | None, str]:
    """POST one chat-completions request. Returns (status, body_json, error_str)."""

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {resolve_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(raw), ""
            except json.JSONDecodeError as exc:
                return resp.status, None, f"invalid JSON response: {exc}"
    except urllib.error.HTTPError as exc:
        raw = ""
        try:
            raw = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        msg = raw[:2000] or str(exc)
        try:
            parsed = json.loads(raw)
            err = parsed.get("error")
            if isinstance(err, dict) and err.get("message"):
                msg = str(err["message"])[:2000]
        except (json.JSONDecodeError, AttributeError):
            pass
        return exc.code, None, msg
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, None, f"transport error: {exc}"


def _norm_content(content: Any) -> str:
    """Normalize message content (string or list-of-parts) to plain text."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "".join(parts)
    return ""


def chat_call(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]] | None = None,
    timeout: int = DEFAULT_CALL_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """One chat completion with bounded retries; returns a provider-neutral block.

    ``provider_limit`` is set on HTTP 402 (out of credits) or a 429 that
    survives all retries, so drivers can stop/resume cleanly (same contract as
    the codex arm's usage-limit records).
    """

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "usage": {"include": True},
    }
    if tools:
        payload["tools"] = tools

    start = time.time()
    retries = 0
    status = 0
    body: dict[str, Any] | None = None
    error_message = ""
    provider_limit = False
    while True:
        status, body, error_message = _post(payload, timeout)
        # A 200 body can still carry an error object (mid-stream provider fail).
        if body is not None and isinstance(body.get("error"), dict):
            err = body["error"]
            status = int(err.get("code") or status or 500)
            error_message = str(err.get("message", ""))[:2000]
            body = None
        if body is not None:
            break
        if status == 402:
            provider_limit = True
            break
        retryable = status in (0, 408, 429, 500, 502, 503, 504, 524)
        if not retryable or retries >= max_retries:
            provider_limit = status == 429
            break
        time.sleep(BACKOFF_BASE_SECONDS * (2**retries))
        retries += 1
    wall = time.time() - start

    block: dict[str, Any] = {
        "backend": "openrouter",
        "ok": body is not None,
        "wall_seconds": round(wall, 3),
        "status_code": status,
        "retries": retries,
        "error_message": error_message,
        "provider_limit": provider_limit,
        "model_requested": model,
    }
    if body is None:
        block.update(
            {
                "text": "",
                "cost_usd": None,
                "total_cost_usd": None,
                "usage": {},
                "assistant_message": None,
                "tool_calls": [],
                "finish_reason": None,
            }
        )
        return block

    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}
    cost = usage.get("cost")
    details_out = usage.get("completion_tokens_details") or {}
    details_in = usage.get("prompt_tokens_details") or {}
    tool_calls = message.get("tool_calls") or []
    block.update(
        {
            "text": _norm_content(message.get("content")),
            "cost_usd": cost,
            "total_cost_usd": cost,  # summed by cells.cell_summary
            "usage": {
                "input_tokens": usage.get("prompt_tokens"),
                "cached_input_tokens": details_in.get("cached_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "reasoning_output_tokens": details_out.get("reasoning_tokens"),
            },
            "assistant_message": message,
            "tool_calls": tool_calls,
            "num_tool_calls": len(tool_calls),
            "finish_reason": choice.get("finish_reason"),
            "model_used": body.get("model"),
            "provider_name": body.get("provider"),
            "generation_id": body.get("id"),
        }
    )
    return block


def _bash_env() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(SECRET_ENV_PREFIXES)
    }


def execute_bash(command: str, cwd: Path, timeout: int) -> str:
    """Run one model-authored bash command; return a truncated transcript."""

    start = time.time()
    try:
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_bash_env(),
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        code = proc.returncode
        note = ""
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        code = -1
        note = f"\n[command killed after {timeout}s timeout]"
    wall = time.time() - start

    def clip(s: str) -> str:
        if len(s) <= TOOL_OUTPUT_MAX_CHARS:
            return s
        half = TOOL_OUTPUT_MAX_CHARS // 2
        return s[:half] + f"\n...[{len(s) - 2 * half} chars truncated]...\n" + s[-half:]

    return (
        f"exit_code: {code} (wall {wall:.1f}s){note}\n"
        f"stdout:\n{clip(out)}\n"
        f"stderr:\n{clip(err)}"
    )


def run_tool_loop(
    prompt: str,
    workdir: Path,
    model: str,
    max_turns: int = DEFAULT_MAX_TURNS,
    wall_cap: int = DEFAULT_AGENT_WALL,
    bash_timeout: int = DEFAULT_BASH_TIMEOUT,
    call_timeout: int = DEFAULT_CALL_TIMEOUT,
    exec_semaphore: Any | None = None,
) -> dict[str, Any]:
    """Bash-only agent loop mirroring the corpus ``tool_loop`` condition.

    One user prompt, one ``bash`` tool, ``max_turns`` assistant turns, total
    ``wall_cap`` seconds. Returns an aggregate provider-neutral block; the full
    message transcript is under ``"transcript"`` (caller persists it to disk and
    pops it so run_record.json stays lean). ``exec_semaphore`` (optional)
    gates model-authored bash executions so concurrent cells respect
    per-platform local-execution caps (e.g. Stata).
    """

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    calls: list[dict[str, Any]] = []
    start = time.time()
    turns = 0
    num_tool_calls = 0
    timed_out = False
    error_message = ""
    provider_limit = False
    final_text = ""

    while turns < max_turns:
        if time.time() - start > wall_cap:
            timed_out = True
            break
        block = chat_call(messages, model, tools=BASH_TOOL, timeout=call_timeout)
        calls.append(
            {
                k: block.get(k)
                for k in (
                    "ok",
                    "wall_seconds",
                    "status_code",
                    "retries",
                    "cost_usd",
                    "usage",
                    "finish_reason",
                    "num_tool_calls",
                    "provider_name",
                    "model_used",
                    "generation_id",
                    "error_message",
                )
            }
        )
        if not block["ok"]:
            error_message = block["error_message"]
            provider_limit = provider_limit or block["provider_limit"]
            break
        turns += 1
        assistant_message = block["assistant_message"] or {}
        messages.append(assistant_message)
        tool_calls = block["tool_calls"]
        if not tool_calls:
            final_text = block["text"]
            break
        for tc in tool_calls:
            num_tool_calls += 1
            tc_id = tc.get("id") or f"call_{num_tool_calls}"
            try:
                args = json.loads((tc.get("function") or {}).get("arguments") or "{}")
                command = args.get("command", "")
            except json.JSONDecodeError:
                command = ""
            if (tc.get("function") or {}).get("name") != "bash" or not command:
                output = "error: only the bash tool with a 'command' argument is available."
            else:
                remaining = max(5, int(wall_cap - (time.time() - start)))
                if exec_semaphore is not None:
                    with exec_semaphore:
                        output = execute_bash(
                            command, workdir, min(bash_timeout, remaining)
                        )
                else:
                    output = execute_bash(command, workdir, min(bash_timeout, remaining))
            messages.append(
                {"role": "tool", "tool_call_id": tc_id, "content": output}
            )
    else:
        # Turn cap reached with the model still asking for tools.
        error_message = error_message or f"turn cap reached ({max_turns})"

    costs = [c["cost_usd"] for c in calls if isinstance(c.get("cost_usd"), (int, float))]
    usage_sum: dict[str, int] = {}
    for c in calls:
        for k, v in (c.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage_sum[k] = usage_sum.get(k, 0) + int(v)
    return {
        "backend": "openrouter",
        "ok": bool(calls) and all(c["ok"] for c in calls) and not timed_out,
        "text": final_text,
        "wall_seconds": round(time.time() - start, 3),
        "timed_out": timed_out,
        "provider_limit": provider_limit,
        "error_message": error_message,
        "cost_usd": round(sum(costs), 8) if costs else None,
        "total_cost_usd": round(sum(costs), 8) if costs else None,
        "usage": usage_sum,
        "num_turns": turns,
        "num_tool_calls": num_tool_calls,
        "num_api_calls": len(calls),
        "calls": calls,
        "model_requested": model,
        "model_used": next((c.get("model_used") for c in calls if c.get("model_used")), None),
        "provider_name": next((c.get("provider_name") for c in calls if c.get("provider_name")), None),
        "max_turns": max_turns,
        "wall_cap_seconds": wall_cap,
        "bash_timeout_seconds": bash_timeout,
        "transcript": messages,
    }

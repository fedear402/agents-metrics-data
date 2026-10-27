"""Cell execution library for the three provider backends.

Formerly ``run_offline_pilot.py``. This module holds the per-cell run
functions that collected the Codex, DeepSeek, and Sonnet few-shot corpora,
plus the provider meter probes and the cell summary. It is a library:
``collect.py`` owns arm resolution, guards, scheduling, and the CLI.

One function per backend runs a single cell and writes its record:

- ``run_haiku_cell``: any Claude model through ``harness.run_cell`` with the
  model overridden; stamps ``arm``/``backend`` and full reproducibility
  metadata, then rescores canonically.
- ``run_codex_cell``: ``codex_backend.run_codex`` plus the harness
  execute/score machinery.
- ``run_openrouter_cell``: direct OpenRouter API (and the opt-in
  mini-swe-agent condition) plus the harness execute/score machinery.

Codex conditions mirror the Claude interface conditions as faithfully as codex exec allows
(Codex- and multi-lens-reviewed 2026-07-10):

- ``codex_generate`` <-> ``no_feedback``: emit one code block; codex runs in an
  EMPTY temp dir and is told the data is not present and not to run commands
  (kills false-premise probing); execution_success = script exit only.
- ``codex_repair`` <-> ``self_repair``: generate, run, then stateless repair up
  to ``harness.MAX_EXECUTIONS_REPAIR`` executions; ``attempt{n}_`` script keys
  and artifact-gated success.
- ``codex_agent`` <-> ``tool_loop``: codex agent in a ``workspace-write`` temp
  dir writes ``result.json``; execution_success = ok and result.json exists.

Every cell function honors the provider-rejection re-run contract: an
existing record that is a provider-limit rejection is treated as missing and
re-run in place; any other existing record is returned unchanged with
``skipped_existing``. A stored record is never overwritten otherwise.

This module never imports the queue client, so there is no push path. Arm
guards live in ``collect.py``, which derives every arm root from the
``analysis_dataset.ARMS`` table.
"""

from __future__ import annotations

import hashlib
import json
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import codex_backend  # noqa: E402
import demonstrations  # noqa: E402
import harness  # noqa: E402
import miniswe_backend  # noqa: E402
import openrouter_backend  # noqa: E402
import pilot_scoring  # noqa: E402
import tasks as tasks_mod  # noqa: E402

HAIKU_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_CODEX_MODEL = "gpt-5.4"
DEFAULT_CODEX_REASONING = "medium"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b"

# Provisional codex wall-clock caps (seconds). NOT empirically calibrated yet —
# deliberately generous so the calibration batch measures true latency rather
# than being truncated. Recorded per run; tighten from calibration data.
DEFAULT_CODEX_TIMEOUT = 900
DEFAULT_CODEX_AGENT_TIMEOUT = 1500

# ``or_agent_swe`` runs the constrained-agent condition through
# mini-swe-agent; since 2026-08-18 it is the stored agent condition of the
# promoted DeepSeek arms (decision record
# 2026-08-18_deepseek-swe-arm-replacement).
OR_DEFAULT_PLAN = "2026-08-18_deepseek-swe-to12"

# Codex generator/agent temp workdirs live OUTSIDE the repo so the repo tree is
# never mutated by a workspace-write codex run (Codex review #5/#6).
CODEX_TMP_ROOT = Path.home() / "codex_pilot_runs"

# An arm root holds one directory per run and nothing else. Collection state
# (the arm marker, the cell summary, the pacing state, the usage ledger) lives
# one level down in this underscore-prefixed directory, which every analysis
# loader skips. Keep this name equal to ``analysis_dataset.WORKING_DIRNAME``.
WORKING_DIRNAME = "_working"


class CodexUsageProbe:
    """Read the requested model's weekly meter from the Codex app server."""

    def __init__(
        self,
        *,
        interval_seconds: float = 60.0,
        timeout_seconds: float = 15.0,
        model: str = DEFAULT_CODEX_MODEL,
    ) -> None:
        self.enabled = True
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.model = model
        self._last_snapshot: harness.ClaudeUsageSnapshot | None = None
        self._last_probe_at = 0.0
        self._lock = threading.Lock()
        self.last_error: str | None = None

    def snapshot(self) -> harness.ClaudeUsageSnapshot | None:
        with self._lock:
            now = time.time()
            if self._last_snapshot is not None and now - self._last_probe_at < self.interval_seconds:
                return self._last_snapshot
            try:
                payload = self._read_rate_limits()
                selected = self._select_limit(payload)
                primary = selected.get("primary") or {}
                used, reset = primary.get("usedPercent"), primary.get("resetsAt")
                if not isinstance(used, (int, float)) or not isinstance(reset, (int, float)):
                    raise RuntimeError("Codex app-server omitted primary usage or reset")
                limit_id = selected.get("limitId")
                source = "codex-app-server"
                if limit_id:
                    source = f"{source}:{limit_id}"
                self._last_snapshot = harness.ClaudeUsageSnapshot(
                    observed_at=now,
                    source=source,
                    current_session_percent=None,
                    current_session_resets=None,
                    current_session_reset_ts=None,
                    weekly_all_percent=float(used),
                    weekly_all_resets=None,
                    weekly_all_reset_ts=float(reset),
                    weekly_sonnet_percent=None,
                    weekly_sonnet_resets=None,
                    weekly_sonnet_reset_ts=None,
                )
                self.last_error = None
                return self._last_snapshot
            except Exception as exc:  # noqa: BLE001 - controller fail-closes below
                self.last_error = f"{type(exc).__name__}: {exc}"
                return None
            finally:
                self._last_probe_at = now

    def _select_limit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Select Spark's separate bucket when the requested model is Spark."""

        if self.model.lower() != "gpt-5.3-codex-spark":
            selected = payload.get("rateLimits")
            if isinstance(selected, dict):
                return selected
            raise RuntimeError("Codex app-server omitted the standard rate limit")

        limits = payload.get("rateLimitsByLimitId")
        if not isinstance(limits, dict):
            raise RuntimeError("Codex app-server omitted model-specific rate limits")
        for selected in limits.values():
            if not isinstance(selected, dict):
                continue
            name = str(selected.get("limitName") or "").lower()
            if name == "gpt-5.3-codex-spark":
                return selected
        raise RuntimeError("Codex app-server omitted the GPT-5.3-Codex-Spark limit")

    def _read_rate_limits(self) -> dict[str, Any]:
        process = subprocess.Popen(
            ["codex", "app-server", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(json.dumps({
                "id": 1,
                "method": "initialize",
                "params": {"clientInfo": {"name": "agents-metrics", "version": "1"}},
            }) + "\n")
            process.stdin.flush()
            initialized = False
            deadline = time.monotonic() + self.timeout_seconds
            while time.monotonic() < deadline:
                ready, _, _ = select.select([process.stdout], [], [], 0.2)
                if not ready:
                    continue
                line = process.stdout.readline()
                if not line:
                    break
                message = json.loads(line)
                if message.get("id") == 1 and "result" in message and not initialized:
                    process.stdin.write(json.dumps({"id": 2, "method": "account/rateLimits/read"}) + "\n")
                    process.stdin.flush()
                    initialized = True
                elif message.get("id") == 2:
                    result = message.get("result")
                    if isinstance(result, dict):
                        return result
                    raise RuntimeError("Codex app-server returned no rate-limit result")
            raise TimeoutError("Codex app-server rate-limit probe timed out")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)


class CodexUsageMonitor:
    """Persist frequent meter readings while a Codex batch is active."""

    def __init__(self, probe: CodexUsageProbe, ledger_path: Path) -> None:
        self.probe = probe
        self.ledger_path = ledger_path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.probe.timeout_seconds + 2)

    def poll_once(self) -> None:
        snapshot = self.probe.snapshot()
        row = snapshot.to_json() if snapshot is not None else {
            "observed_at": time.time(),
            "source": "codex-app-server:error",
            "error": self.probe.last_error,
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.probe.interval_seconds)


WORKDIR_OUTSIDE_REPO = (
    CODEX_TMP_ROOT.resolve() != harness.PROJECT_ROOT.resolve()
    and harness.PROJECT_ROOT.resolve() not in CODEX_TMP_ROOT.resolve().parents
)

# OpenRouter agent workdirs likewise live outside the repo (and outside any
# CLAUDE.md/context surface — 2026-07-13 invocation-audit hermeticity rule).
OR_TMP_ROOT = Path.home() / "openrouter_pilot_runs"

# Local-execution concurrency caps for --concurrency > 1. Model calls are
# network-bound and fan out freely; script execution is gated per platform.
# Stata cap is a chosen conservative limit (license/session behavior is
# unmeasured; observe before raising), kept low while other sweeps run.
EXEC_GATE_LIMITS = {"python": 6, "r": 6, "stata": 2}

# or_agent_swe holds its gate for the WHOLE run (mini-swe-agent executes the
# model's commands inside its own subprocess, so per-execution gating is not
# reachable). Runs are dominated by model wait, and the per-execution work is
# a short local script, so these run-level caps are wider than the
# per-execution caps above.
SWE_RUN_GATE_LIMITS = {"python": 8, "r": 8, "stata": 4}

# Appended to codex generator/repair prompts so codex does not act on the
# harness prompt's (for codex, false) premise that input.csv is already present.
CODEX_GENERATOR_NOTE = (
    "\n\nIMPORTANT — code generation only: input.csv is NOT in your current "
    "directory right now, and you must NOT run any commands or read any file. "
    "Your script will be executed later in a directory that contains input.csv. "
    "Return ONLY the fenced code block."
)

TASKS_BY_ID = {t.task_id: t for t in tasks_mod.ALL_TASKS}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def claude_version() -> str:
    import subprocess

    try:
        out = subprocess.run(
            [harness.claude_bin(), "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return (out.stdout or out.stderr or "").strip() or "unknown"
    except Exception as exc:  # noqa: BLE001
        return f"unknown:{exc}"


def stamp_metadata(
    record: dict[str, Any],
    arm: str,
    backend: str,
    plan: str = "2026-07-10_codex-and-haiku-replication-arms",
) -> None:
    record["arm"] = arm
    record["backend"] = backend
    record["pilot_plan"] = plan
    record.setdefault(
        "created_at", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def _mkdtemp(cell_id: str, tag: str) -> Path:
    CODEX_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{cell_id}__{tag}__", dir=str(CODEX_TMP_ROOT)))


# --------------------------------------------------------------------------- #
# Haiku arm — reuse harness.run_cell verbatim with the model overridden
# --------------------------------------------------------------------------- #
def claude_arm_for(model: str) -> str:
    """Model-derived arm label for the claude backend (2026-07-13 audit fix:
    sonnet-model pilot runs were mislabelled ``haiku_pilot``)."""

    return "sonnet_pilot" if "sonnet" in model else "haiku_pilot"


def run_haiku_cell(
    task: Any,
    platform: str,
    prompt_mode: str,
    exec_mode: str,
    rep: int,
    overwrite: bool,
    model: str = HAIKU_MODEL,
    rerun_provider_limit: bool = True,
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    # The claude backend can run any Claude model via ``model``; used for the
    # Sonnet arms and for meter-control pilots under identical conditions.
    # ``rerun_provider_limit`` defaults True so the claude path honors the same
    # rejection re-run contract as the codex and openrouter cell functions
    # (defect found 2026-08-18: the flag was never passed through, so a stored
    # rejection was skipped forever unless --overwrite was used).
    harness.MODEL = model
    started = datetime.now(timezone.utc)
    t0 = time.time()
    record = harness.run_cell(
        task,
        platform,
        prompt_mode,
        exec_mode,
        rep=rep,
        overwrite=overwrite,
        rerun_provider_limit=rerun_provider_limit,
        runs_dir=runs_dir,
    )
    # A pre-existing record was returned unchanged; do NOT re-stamp/re-write it
    # (that would corrupt a foreign record in place).
    if record.get("skipped_existing"):
        return record
    record["total_wall_seconds"] = round(time.time() - t0, 3)
    record["started_at"] = started.isoformat(timespec="seconds")
    record["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp_metadata(record, arm=claude_arm_for(model), backend="claude")
    record["claude_version"] = claude_version()
    record["model"] = model
    # Provenance for censoring analysis (2026-07-13 audit): record the caps and
    # CLI hermeticity flags this cell actually ran under.
    record["claude_timeout_seconds"] = harness.TIMEOUT_NO_FEEDBACK
    record["claude_extra_cli_args"] = list(harness.CLAUDE_EXTRA_CLI_ARGS)
    # Keep the direct-Claude pilot path resumable.  The main harness already
    # classifies failed zero-token Claude calls as provider-limit records; copy
    # that classification into this runner's top-level resume contract.
    if harness.is_provider_limit_record(record):
        limit_block = next(
            (
                block
                for block in harness.claude_blocks(record)
                if harness.is_provider_limit_block(block)
            ),
            None,
        )
        record["provider_limit"] = True
        record["provider_limit_message"] = str(
            (limit_block or {}).get("error", "")
        )
        record["failure_reason"] = "claude provider-limit or zero-token failure"
    # run_cell used harness.score() (simplified live gate); re-score canonically.
    base_dir = runs_dir if runs_dir is not None else harness.RUNS_DIR
    pilot_scoring.rescore_record(base_dir / record["cell_id"], record)
    harness.write_run_record(record, runs_dir=runs_dir)
    return record


# --------------------------------------------------------------------------- #
# Codex arm
# --------------------------------------------------------------------------- #
def _base_codex_record(
    task, platform, prompt_mode, condition, rep, cell_id, cell_id_base,
    data_src, initial_prompt, model, reasoning,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "cell_id": cell_id,
        "cell_id_base": cell_id_base,
        "rep": rep,
        "task_id": task.task_id,
        "reference_key": task.reference_key,
        "platform": platform,
        "exec_mode": condition,
        "prompt_mode": prompt_mode,
        "model": model,
        "reasoning_effort": reasoning,
        "data_source": harness.display_path(data_src),
        "prompt_sha256": hashlib.sha256(initial_prompt.encode("utf-8")).hexdigest(),
        "workdir_outside_repo": WORKDIR_OUTSIDE_REPO,
    }
    stamp_metadata(record, arm="codex_pilot", backend="codex")
    return record


def _finalize(
    record: dict[str, Any],
    task: Any,
    cell_dir: Path,
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    # Canonical scoring (strict_posthoc): normalizes R []->{}, repairs Stata JSON,
    # scores the artifact independently of CLI exit, and compares numeric
    # independent of artifact. numeric_success is the trustworthy correctness
    # metric; execution_success stays separate.
    scoring = pilot_scoring.canonical_score(cell_dir, task, record["platform"], record)
    if record.get("failure_reason"):
        scoring.setdefault("notes", []).append(record["failure_reason"])
    scoring["execution_success"] = record.get("execution_success")
    record["scoring"] = scoring
    harness.write_run_record(record, runs_dir=runs_dir)
    return record


def _codex_generate(record, task, platform, prompt_mode, cell_dir, condition,
                    model, reasoning, timeout, events_path, last_path) -> None:
    prompt = harness.no_feedback_prompt(task, platform, prompt_mode) + CODEX_GENERATOR_NOTE
    record["codex_timeout_seconds"] = timeout
    tmp = _mkdtemp(record["cell_id"], "gen")
    try:
        res = codex_backend.run_codex(
            prompt, tmp, condition, model, reasoning, timeout, events_path, last_path
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record["codex"] = res
    code = harness.extract_code(res["text"]) if res["ok"] else None
    if not res["ok"]:
        record["execution_success"] = False
        record["failure_reason"] = "codex call failed"
        return
    if not code:
        record["execution_success"] = False
        record["failure_reason"] = "no code block extracted"
        record["raw_text_tail"] = res["text"][-400:]
        return
    script = cell_dir / harness.script_name(platform)
    script.write_text(code)
    exec_res = harness.execute_script(
        script, platform, cell_dir, harness.TIMEOUT_NO_FEEDBACK
    )
    record["execution"] = exec_res
    record["execution_success"] = bool(exec_res["ok"])  # no_feedback: exit-only


def _codex_repair(record, task, platform, prompt_mode, cell_dir, condition,
                  model, reasoning, timeout, events_path, last_path) -> None:
    result_path = cell_dir / "result.json"
    prompt = harness.no_feedback_prompt(task, platform, prompt_mode) + CODEX_GENERATOR_NOTE
    record["codex_timeout_seconds"] = timeout
    tmp = _mkdtemp(record["cell_id"], "rep1")
    try:
        first = codex_backend.run_codex(
            prompt, tmp, condition, model, reasoning, timeout, events_path, last_path
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record["codex_attempt_1"] = first
    first_code = harness.extract_code(first["text"]) if first["ok"] else None
    if not first["ok"]:
        record["execution_success"] = False
        record["failure_reason"] = "initial codex call failed"
        return
    if not first_code:
        record["execution_success"] = False
        record["failure_reason"] = "no initial code block extracted"
        record["raw_text_tail"] = first["text"][-400:]
        return
    first_script = cell_dir / f"attempt1_{harness.script_name(platform)}"
    first_script.write_text(first_code)
    first_exec = harness.execute_script(
        first_script, platform, cell_dir, harness.TIMEOUT_NO_FEEDBACK
    )
    record["execution_attempt_1"] = first_exec
    prev_code, prev_exec = first_code, first_exec
    record["execution_success"] = (
        first_exec["ok"] and harness.artifact_problem(result_path) is None
    )
    for attempt in range(2, harness.MAX_EXECUTIONS_REPAIR + 1):
        if record["execution_success"]:
            break
        rp = harness.repair_prompt(
            task, platform, prompt_mode,
            previous_code=prev_code,
            failure_context=harness.failure_context(prev_exec, result_path),
        ) + CODEX_GENERATOR_NOTE
        tmp_a = _mkdtemp(record["cell_id"], f"rep{attempt}")
        try:
            res_a = codex_backend.run_codex(
                rp, tmp_a, condition, model, reasoning, timeout,
                cell_dir / f"codex_events_attempt{attempt}.jsonl",
                cell_dir / f"codex_last_attempt{attempt}.txt",
            )
        finally:
            shutil.rmtree(tmp_a, ignore_errors=True)
        record[f"codex_attempt_{attempt}"] = res_a
        code_a = harness.extract_code(res_a["text"]) if res_a["ok"] else None
        if not code_a:
            record["execution_success"] = False
            record[f"raw_repair_attempt_{attempt}_text_tail"] = res_a["text"][-400:]
            break
        script_a = cell_dir / f"attempt{attempt}_{harness.script_name(platform)}"
        script_a.write_text(code_a)
        exec_a = harness.execute_script(
            script_a, platform, cell_dir, harness.TIMEOUT_NO_FEEDBACK
        )
        record[f"execution_attempt_{attempt}"] = exec_a
        prev_code, prev_exec = code_a, exec_a
        record["execution_success"] = (
            exec_a["ok"] and harness.artifact_problem(result_path) is None
        )


def _codex_agent(record, task, platform, prompt_mode, cell_dir, condition,
                 model, reasoning, timeout, events_path, last_path) -> None:
    result_path = cell_dir / "result.json"
    prompt = harness.tool_loop_prompt(task, platform, prompt_mode)
    record["codex_agent_timeout_seconds"] = timeout
    tmp = _mkdtemp(record["cell_id"], "agent")
    try:
        shutil.copy2(cell_dir / "input.csv", tmp / "input.csv")
        res = codex_backend.run_codex(
            prompt, tmp, condition, model, reasoning, timeout, events_path, last_path
        )
        record["codex"] = res
        harness.copy_back(tmp, cell_dir, platform=platform)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record["execution_success"] = bool(res["ok"]) and result_path.exists()


def run_codex_cell(
    task, platform, prompt_mode, condition, rep, model, reasoning, overwrite,
    codex_timeout=DEFAULT_CODEX_TIMEOUT, codex_agent_timeout=DEFAULT_CODEX_AGENT_TIMEOUT,
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    cell_id, cell_id_base, cell_dir, data_src = harness.prepare_cell_dir(
        task, platform, prompt_mode, condition, rep, overwrite, runs_dir=runs_dir
    )
    record_path = cell_dir / "run_record.json"
    if record_path.exists() and not overwrite:
        rec = json.loads(record_path.read_text())
        # A usage-limit rejection is not a result: re-run the cell instead of
        # skipping it, so a resumable driver completes the grid across windows.
        if not rec.get("provider_limit"):
            rec["skipped_existing"] = True
            return rec

    events_path = cell_dir / "codex_events.jsonl"
    last_path = cell_dir / "codex_last.txt"

    if condition == "codex_agent":
        initial_prompt = harness.tool_loop_prompt(task, platform, prompt_mode)
        timeout = codex_agent_timeout
        handler = _codex_agent
    elif condition == "codex_generate":
        initial_prompt = harness.no_feedback_prompt(task, platform, prompt_mode) + CODEX_GENERATOR_NOTE
        timeout = codex_timeout
        handler = _codex_generate
    elif condition == "codex_repair":
        initial_prompt = harness.no_feedback_prompt(task, platform, prompt_mode) + CODEX_GENERATOR_NOTE
        timeout = codex_timeout
        handler = _codex_repair
    else:
        raise ValueError(f"unknown codex condition: {condition}")

    record = _base_codex_record(
        task, platform, prompt_mode, condition, rep, cell_id, cell_id_base,
        data_src, initial_prompt, model, reasoning,
    )
    started = datetime.now(timezone.utc)
    t0 = time.time()

    handler(record, task, platform, prompt_mode, cell_dir, condition, model,
            reasoning, timeout, events_path, last_path)

    # Surface a codex usage-limit (message-budget) rejection so the batch can
    # stop cleanly rather than burn the remaining cells as fake failures.
    limit_block = next(
        (b for b in (record.get("codex"),
                     *(record.get(f"codex_attempt_{i}")
                       for i in range(1, harness.MAX_EXECUTIONS_REPAIR + 1)))
         if isinstance(b, dict) and b.get("provider_limit")),
        None,
    )
    if limit_block is not None:
        record["provider_limit"] = True
        record["provider_limit_message"] = limit_block.get("error_message", "")
        record["failure_reason"] = "codex usage limit"

    record["total_wall_seconds"] = round(time.time() - t0, 3)
    record["started_at"] = started.isoformat(timespec="seconds")
    record["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return _finalize(record, task, cell_dir, runs_dir=runs_dir)


# --------------------------------------------------------------------------- #
# OpenRouter arm — direct API, conditions mirror the Claude interface conditions
# --------------------------------------------------------------------------- #
def openrouter_arm_for(model: str) -> str:
    """Model-derived arm label, e.g. ``openrouter_openai-gpt-oss-120b``."""

    slug = model.replace("/", "-").replace(":", "-").replace(".", "_")
    return f"openrouter_{slug}"


def demo_provenance(task_id: str, platform: str) -> dict[str, Any]:
    """Demonstration provenance for a few-shot cell (respecified wave).

    Mirrors the corpus wave registry: the resolved demonstration key, the
    assigned demo ids, and a hash of the joined demonstration text actually
    injected into the prompt. Empty for tasks with no registered demonstration
    (the prompt build would raise before a record is written anyway).
    """

    key = demonstrations.resolve_demo_task_id(task_id)
    if key is None:
        return {}
    demo_ids = list(demonstrations.DEMO_ASSIGNMENTS.get(key, ()))
    text = demonstrations.demo_for(task_id, platform)
    return {
        "demo_key": key,
        "demo_ids": demo_ids,
        "n_demos": len(demo_ids),
        "demo_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _or_mkdtemp(cell_id: str, tag: str) -> Path:
    OR_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{cell_id}__{tag}__", dir=str(OR_TMP_ROOT)))


def _slim_or_block(block: dict[str, Any]) -> dict[str, Any]:
    """Drop the bulky assistant message/tool fields before persisting."""

    return {
        k: v
        for k, v in block.items()
        if k not in ("assistant_message", "tool_calls", "transcript")
    }


def _or_generate(record, task, platform, prompt_mode, cell_dir,
                 call_timeout, exec_gate) -> None:
    # A direct API call has no tool surface at all, so no generator note is
    # needed (the codex note exists because codex is an agent even read-only).
    prompt = harness.no_feedback_prompt(task, platform, prompt_mode)
    block = openrouter_backend.chat_call(
        [{"role": "user", "content": prompt}], record["model"], timeout=call_timeout
    )
    record["openrouter"] = _slim_or_block(block)
    code = harness.extract_code(block["text"]) if block["ok"] else None
    if not block["ok"]:
        record["execution_success"] = False
        record["failure_reason"] = "openrouter call failed"
        return
    if not code:
        record["execution_success"] = False
        record["failure_reason"] = "no code block extracted"
        record["raw_text_tail"] = block["text"][-400:]
        return
    script = cell_dir / harness.script_name(platform)
    script.write_text(code)
    with exec_gate:
        exec_res = harness.execute_script(
            script, platform, cell_dir, harness.TIMEOUT_NO_FEEDBACK
        )
    record["execution"] = exec_res
    record["execution_success"] = bool(exec_res["ok"])  # no_feedback: exit-only


def _or_repair(record, task, platform, prompt_mode, cell_dir,
               call_timeout, exec_gate) -> None:
    result_path = cell_dir / "result.json"
    prompt = harness.no_feedback_prompt(task, platform, prompt_mode)
    first = openrouter_backend.chat_call(
        [{"role": "user", "content": prompt}], record["model"], timeout=call_timeout
    )
    record["openrouter_attempt_1"] = _slim_or_block(first)
    first_code = harness.extract_code(first["text"]) if first["ok"] else None
    if not first["ok"]:
        record["execution_success"] = False
        record["failure_reason"] = "initial openrouter call failed"
        return
    if not first_code:
        record["execution_success"] = False
        record["failure_reason"] = "no initial code block extracted"
        record["raw_text_tail"] = first["text"][-400:]
        return
    first_script = cell_dir / f"attempt1_{harness.script_name(platform)}"
    first_script.write_text(first_code)
    with exec_gate:
        first_exec = harness.execute_script(
            first_script, platform, cell_dir, harness.TIMEOUT_NO_FEEDBACK
        )
    record["execution_attempt_1"] = first_exec
    prev_code, prev_exec = first_code, first_exec
    record["execution_success"] = (
        first_exec["ok"] and harness.artifact_problem(result_path) is None
    )
    for attempt in range(2, harness.MAX_EXECUTIONS_REPAIR + 1):
        if record["execution_success"]:
            break
        rp = harness.repair_prompt(
            task, platform, prompt_mode,
            previous_code=prev_code,
            failure_context=harness.failure_context(prev_exec, result_path),
        )
        res_a = openrouter_backend.chat_call(
            [{"role": "user", "content": rp}], record["model"], timeout=call_timeout
        )
        record[f"openrouter_attempt_{attempt}"] = _slim_or_block(res_a)
        code_a = harness.extract_code(res_a["text"]) if res_a["ok"] else None
        if not code_a:
            record["execution_success"] = False
            record[f"raw_repair_attempt_{attempt}_text_tail"] = res_a["text"][-400:]
            break
        script_a = cell_dir / f"attempt{attempt}_{harness.script_name(platform)}"
        script_a.write_text(code_a)
        with exec_gate:
            exec_a = harness.execute_script(
                script_a, platform, cell_dir, harness.TIMEOUT_NO_FEEDBACK
            )
        record[f"execution_attempt_{attempt}"] = exec_a
        prev_code, prev_exec = code_a, exec_a
        record["execution_success"] = (
            exec_a["ok"] and harness.artifact_problem(result_path) is None
        )


def _or_agent(record, task, platform, prompt_mode, cell_dir,
              call_timeout, exec_gate, agent_wall) -> None:
    result_path = cell_dir / "result.json"
    prompt = harness.tool_loop_prompt(task, platform, prompt_mode)
    record["or_agent_wall_seconds_cap"] = agent_wall
    tmp = _or_mkdtemp(record["cell_id"], "agent")
    try:
        shutil.copy2(cell_dir / "input.csv", tmp / "input.csv")
        block = openrouter_backend.run_tool_loop(
            prompt, tmp, record["model"],
            wall_cap=agent_wall,
            call_timeout=call_timeout,
            exec_semaphore=exec_gate,
        )
        transcript = block.pop("transcript", None)
        if transcript is not None:
            (cell_dir / "or_agent_transcript.json").write_text(
                json.dumps(transcript, indent=1, default=str)
            )
        record["openrouter"] = _slim_or_block(block)
        harness.copy_back(tmp, cell_dir, platform=platform)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record["execution_success"] = bool(block["ok"]) and result_path.exists()


def _or_agent_swe(record, task, platform, prompt_mode, cell_dir,
                  call_timeout, exec_gate, agent_wall) -> None:
    # call_timeout is unused (litellm owns per-call networking); the argument
    # keeps the handler signature parallel with _or_agent.
    del call_timeout
    result_path = cell_dir / "result.json"
    prompt = miniswe_backend.swe_tool_loop_prompt(task, platform, prompt_mode)
    record["or_agent_wall_seconds_cap"] = agent_wall
    tmp = _or_mkdtemp(record["cell_id"], "swe")
    try:
        shutil.copy2(cell_dir / "input.csv", tmp / "input.csv")
        with exec_gate:  # whole-run gate; see SWE_RUN_GATE_LIMITS
            block = miniswe_backend.run_swe_agent(
                prompt, tmp, record["model"], wall_cap=agent_wall
            )
        transcript = block.pop("transcript", None)
        if transcript is not None:
            (cell_dir / "swe_agent_transcript.json").write_text(
                json.dumps(transcript, indent=1, default=str)
            )
        config = block.pop("config", None)
        if config is not None:
            (cell_dir / "swe_config.json").write_text(json.dumps(config, indent=1))
        record["openrouter"] = _slim_or_block(block)
        harness.copy_back(tmp, cell_dir, platform=platform)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record["execution_success"] = bool(block["ok"]) and result_path.exists()


def run_openrouter_cell(
    task, platform, prompt_mode, condition, rep, model, overwrite,
    call_timeout=openrouter_backend.DEFAULT_CALL_TIMEOUT,
    agent_wall=openrouter_backend.DEFAULT_AGENT_WALL,
    exec_gate: Any | None = None,
    plan: str = OR_DEFAULT_PLAN,
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    cell_id, cell_id_base, cell_dir, data_src = harness.prepare_cell_dir(
        task, platform, prompt_mode, condition, rep, overwrite, runs_dir=runs_dir
    )
    record_path = cell_dir / "run_record.json"
    if record_path.exists() and not overwrite:
        rec = json.loads(record_path.read_text())
        # A provider-limit rejection is not a result: re-run (resumable grids).
        if not rec.get("provider_limit"):
            rec["skipped_existing"] = True
            return rec

    if condition == "or_agent":
        initial_prompt = harness.tool_loop_prompt(task, platform, prompt_mode)
        handler_args: tuple = (call_timeout, exec_gate or _null_gate(), agent_wall)
        handler = _or_agent
    elif condition == "or_agent_swe":
        initial_prompt = miniswe_backend.swe_tool_loop_prompt(task, platform, prompt_mode)
        handler_args = (call_timeout, exec_gate or _null_gate(), agent_wall)
        handler = _or_agent_swe
    elif condition in ("or_generate", "or_repair"):
        initial_prompt = harness.no_feedback_prompt(task, platform, prompt_mode)
        handler_args = (call_timeout, exec_gate or _null_gate())
        handler = _or_generate if condition == "or_generate" else _or_repair
    else:
        raise ValueError(f"unknown openrouter condition: {condition}")

    record: dict[str, Any] = {
        "cell_id": cell_id,
        "cell_id_base": cell_id_base,
        "rep": rep,
        "task_id": task.task_id,
        "reference_key": task.reference_key,
        "platform": platform,
        "exec_mode": condition,
        "prompt_mode": prompt_mode,
        "model": model,
        "data_source": harness.display_path(data_src),
        "prompt_sha256": hashlib.sha256(initial_prompt.encode("utf-8")).hexdigest(),
        "openrouter_call_timeout_seconds": call_timeout,
        "exec_timeout_seconds": harness.TIMEOUT_NO_FEEDBACK,
    }
    if prompt_mode == "few_shot":
        record.update(demo_provenance(task.task_id, platform))
    stamp_metadata(
        record,
        arm=openrouter_arm_for(model),
        backend="openrouter",
        plan=plan,
    )
    started = datetime.now(timezone.utc)
    t0 = time.time()

    handler(record, task, platform, prompt_mode, cell_dir, *handler_args)

    limit_block = next(
        (b for b in (record.get("openrouter"),
                     *(record.get(f"openrouter_attempt_{i}")
                       for i in range(1, harness.MAX_EXECUTIONS_REPAIR + 1)))
         if isinstance(b, dict) and b.get("provider_limit")),
        None,
    )
    if limit_block is not None:
        record["provider_limit"] = True
        record["provider_limit_message"] = limit_block.get("error_message", "")
        record["failure_reason"] = "openrouter provider/credit limit"

    record["total_wall_seconds"] = round(time.time() - t0, 3)
    record["started_at"] = started.isoformat(timespec="seconds")
    record["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return _finalize(record, task, cell_dir, runs_dir=runs_dir)


class _null_gate:
    """No-op context manager standing in for an execution semaphore."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------- #
# Cell summary and canonical rescoring
# --------------------------------------------------------------------------- #
def _call_blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = []
    for k, v in record.items():
        if isinstance(v, dict) and (
            k in ("codex", "claude", "openrouter")
            or k.startswith("codex_attempt_")
            or k.startswith("claude_attempt_")
            or k.startswith("openrouter_attempt_")
        ):
            blocks.append(v)
    return blocks


def _exec_walls(record: dict[str, Any]) -> float:
    total = 0.0
    for k, v in record.items():
        if isinstance(v, dict) and (k == "execution" or k.startswith("execution_attempt_")):
            w = v.get("wall_seconds")
            if isinstance(w, (int, float)):
                total += w
    return total


def cell_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Aggregate across ALL attempt blocks so repair conditions are not
    understated (Codex/multi-lens review)."""

    blocks = _call_blocks(record)
    provider_wall = sum(b.get("wall_seconds") or 0 for b in blocks)
    costs = [b.get("total_cost_usd") for b in blocks
             if isinstance(b.get("total_cost_usd"), (int, float))]
    tokens = sum((b.get("usage") or {}).get("output_tokens") or 0 for b in blocks)
    tool_calls = 0
    for b in blocks:
        n = b.get("num_tool_calls")  # codex
        if n is None:
            n = b.get("n_tool_calls")  # claude
        if isinstance(n, int):
            tool_calls += n
    scoring = record.get("scoring") or {}
    return {
        "cell_id": record.get("cell_id"),
        "task_id": record.get("task_id"),
        "platform": record.get("platform"),
        "prompt_mode": record.get("prompt_mode"),
        "condition": record.get("exec_mode"),
        "arm": record.get("arm"),
        "execution_success": record.get("execution_success"),
        "numeric_success": scoring.get("numeric_success"),
        "n_attempts": len(blocks),
        "provider_wall_seconds": round(provider_wall, 3),
        "exec_wall_seconds": round(_exec_walls(record), 3),
        "total_wall_seconds": record.get("total_wall_seconds"),
        "cost_usd": round(sum(costs), 4) if costs else None,
        "output_tokens": tokens,
        "tool_calls": tool_calls,
        "used_tools": tool_calls > 0,
        "rate_limits": any(b.get("rate_limits") for b in blocks),
        "timed_out": any(b.get("timed_out") for b in blocks),
        "skipped_existing": record.get("skipped_existing", False),
    }


def rescore_dir(runs_dir: Path) -> int:
    """Re-score every existing record in ``runs_dir`` canonically (no live runs)."""

    runs_dir = runs_dir.resolve()
    records = sorted(runs_dir.glob("*/run_record.json"))
    print(f"[rescore] {len(records)} records in {runs_dir}")
    summary: list[dict[str, Any]] = []
    changed = 0
    for rp in records:
        try:
            rec = json.loads(rp.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skip {rp.parent.name}: {exc}")
            continue
        before = (rec.get("scoring") or {}).get("numeric_success")
        pilot_scoring.rescore_record(rp.parent, rec)
        after = rec["scoring"].get("numeric_success")
        if before != after:
            changed += 1
            print(f"  {rp.parent.name}: numeric_success {before} -> {after}")
        harness.write_run_record(rec, runs_dir=runs_dir)
        summary.append(cell_summary(rec))
    summary_path = harness.write_summary(
        summary, f"{WORKING_DIRNAME}/pilot_summary.json", runs_dir=runs_dir
    )
    ok = sum(1 for s in summary if s.get("numeric_success"))
    print(f"[rescore] numeric_success {ok}/{len(summary)}; {changed} changed; "
          f"summary -> {summary_path}")
    return 0

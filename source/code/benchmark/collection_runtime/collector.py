"""Resumable two-prompt pilot with model-specific repetition counts.

No subcommand starts by default. Root controls provider launch. Collection
requires a separate authorization file matching the complete protocol hash.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
from ._loader import load_sibling
import itertools
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'quality_reports/reports/2026-09-16_two-prompt-recheck'
WORK = REPORT / "work/collector_v3"
BASE_PATH = Path(__file__).with_name('runner.py')
base = load_sibling(__package__, 'runner')
harness, sp, codex = base.harness, base.sp, base.codex_backend

CATALOGS = ()
TASKS = ()
MODELS = {"luna": ("gpt-5.6-luna", "low"), "sonnet": ("claude-sonnet-4-6", "medium")}
MODEL_REPS = {"luna": 3, "sonnet": 2}
LANGUAGES = ("python", "r", "stata")
PROMPTS = ("zero_shot", "few_shot")
CONDITIONS = ("chatbot", "self_repair", "agent")
BUNDLES = {}
POLICY = {
    "version": "two_prompt_recheck_v3_no_turn_cap",
    "prompt_decision": "authorized pilot: zero examples or one approved exact legacy example",
    "cwd": "fresh external temporary directory for every run; empty generation directory",
    "tools": "agent native shell/file tools; external integrations disabled by requested settings; generation tool count must be zero",
    "allowed_agent_tool_item_types": ["command_execution", "local_shell_call", "exec_command", "file_change", "patch_apply", "apply_patch"],
    "allowed_sonnet_agent_tools": ["Bash"],
    "mcp": "requested empty configuration for every language; effective startup surface unverified",
    "customizations": "requested Codex feature controls and Sonnet safe mode; effective startup behavior requires first live block review",
    "filesystem_limit": "CLI settings and prompt restrictions are not proof of filesystem confinement",
    "visible_access_check": "Review actual tool commands for external skills, project instructions, benchmark references, network and Git access. Stop queue for review on a match.",
    "output": {"python": "result.json", "r": "result.json", "stata": "run.log ANSWER lines"},
    "self_repair_attempts": 4,
    "execution_timeout_seconds": 180,
    "timeouts_seconds": {"luna": {"agent": 1500, "generation": 900}, "sonnet": {"agent": 600, "generation": 180}},
    "sonnet_max_turns": None,
    "workers": 6,
    "stata_slots": 2,
    "scheduling": "continuous bounded queue, refill after each completion, stop submission on infrastructure failure",
    "limit_provenance": "chosen limits inherited from tested collectors; not provider capacities",
    "score": "final output numeric agreement and integrity, with provider/protocol/censor overrides",
    "resume": "completed runs never replaced; interrupted attempts retained in separate folders",
}
BOUNDARY = """\nUse only the supplied input.csv and files you create in this directory.
Installed software and package resources are available.
Do not read external agent skills, project instructions, prior results, or benchmark references.
Do not create journals or use Git. Do not use MCP servers, web tools, or other agents.
"""
STOP = threading.Event()
JOURNAL_LOCK = threading.Lock()
STATA_SLOT = threading.Semaphore(2)
CHILDREN = {}
CHILDREN_LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    base.save(Path(path), value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checkpoint(stage, **data):
    WORK.mkdir(parents=True, exist_ok=True)
    with JOURNAL_LOCK, (WORK / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": now(), "stage": stage, **data}) + "\n")


def load_catalog(paths=CATALOGS):
    result = {}
    for path in paths:
        current = json.loads(Path(path).read_text())
        overlap = set(result) & set(current)
        if overlap:
            raise ValueError(f"Duplicate task definitions: {sorted(overlap)}")
        result.update(current)
    return result


def example_block(bundle, platform, count):
    """Use reviewed example bytes without adding a teaching wrapper."""
    if not count:
        return ''
    if 'injection_text' in bundle:
        return bundle['injection_text'][platform]
    return 'Related example:\n' + bundle['prompt_text'][platform]


def prompt(spec, job, previous=None, feedback=None, bundle=None):
    # Generate the identical target and agency body without the old example.
    text = base.prompt(spec, dict(job, prompt='zero_shot'), previous, feedback)
    count = {'zero_shot': 0, 'few_shot': 1}[job['prompt']]
    if count:
        if bundle is None:
            bundle = json.loads(BUNDLES[spec['task_id']].read_text())
        block = example_block(bundle, job['platform'], count)
        # Match the exact location and spacing in the legacy base.prompt.
        anchor = spec['data_description'] + '\n\n\n\nTask: '
        if anchor not in text:
            raise ValueError('Legacy prompt example anchor is unavailable')
        text = text.replace(anchor, spec['data_description'] + '\n\n' + block + '\n\nTask: ', 1)
    return text + (BOUNDARY if job['condition'] == 'agent' else '')


def command(job, cwd, last):
    """The platform never changes provider permissions or available tools."""
    agent = job["condition"] == "agent"
    model, effort = MODELS[job["model"]]
    if job["model"] == "sonnet":
        args = [harness.claude_bin(), "-p", "--model", model, "--effort", effort,
                "--output-format", "stream-json", "--verbose", "--no-session-persistence",
                "--safe-mode", "--setting-sources", "", "--permission-mode", "manual",
                "--tools", "Bash" if agent else "",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
        if agent:
            args += ["--allowedTools", "Bash"]
    else:
        args = [codex.CODEX_BIN, "exec", "--json", "--ignore-user-config", "--ignore-rules",
                "--ephemeral", "--skip-git-repo-check", "-m", model,
                "--sandbox", "workspace-write" if agent else "read-only",
                "-C", str(cwd), "-o", str(last)]
        settings = [f'model_reasoning_effort="{effort}"', 'web_search="disabled"',
                    'approval_policy="never"', "project_doc_max_bytes=0", "mcp_servers={}",
                    "features.skip_host_skill_discovery=true", "features.skill_search=false",
                    "features.skill_mcp_dependency_install=false", "features.plugins=false",
                    "features.remote_plugin=false", "features.recommended_plugins=false",
                    "features.hooks=false", "features.apps=false", "features.memories=false",
                    "features.multi_agent=false", "features.multi_agent_v2=false",
                    "features.browser_use=false", "features.computer_use=false",
                    "features.image_generation=false", "features.code_mode=false",
                    "features.shell_tool=" + str(agent).lower()]
        for setting in settings:
            args += ["-c", setting]
    timeout = POLICY["timeouts_seconds"][job["model"]]["agent" if agent else "generation"]
    return args, timeout


def dependency_paths():
    files = [Path(__file__), BASE_PATH]
    for module in (harness, sp, codex, base.demonstrations, sys.modules["tasks"],
                   sys.modules["artifact_repair"]):
        files.append(Path(module.__file__))
    return sorted(set(files))


def make_manifest(tasks=None, models=tuple(MODELS), languages=LANGUAGES,
                  prompts=PROMPTS, conditions=CONDITIONS, reps=None, catalogs=CATALOGS, bundles=None):
    reps = dict(MODEL_REPS if reps is None else reps)
    if any(reps.get(model, 0) < 1 for model in models):
        raise ValueError("Repetitions must be positive")
    catalog = load_catalog(catalogs)
    tasks = list(catalog) if tasks is None else list(tasks)
    if len(tasks) != 1:
        raise ValueError('Supply exactly one reviewed task catalog or task selection')
    selected = {key: catalog[key] for key in tasks}
    pins = {str(path): base.sha(path) for path in [*dependency_paths(), *map(Path, catalogs)]}
    bundle_values = {}
    for key, spec in selected.items():
        bundle_path = (bundles or BUNDLES)[key]
        bundle = json.loads(Path(bundle_path).read_text())
        if bundle['task_id'] != key or len(bundle['examples']) != 1:
            raise ValueError(f'Expected one example for {key}')
        approved_text = bundle.get('injection_text', bundle.get('prompt_text', {}))
        if set(approved_text) != set(LANGUAGES) or not all(approved_text.values()):
            raise ValueError('Reviewed exact example text is required for each language')
        for example in bundle['examples']:
            if example.get('validation', {}).get('passed') is not True:
                raise ValueError(f'Example validation is missing: {example["id"]}')
            for language in LANGUAGES:
                if not example['code'].get(language):
                    raise ValueError(f'Example code is missing: {example["id"]}/{language}')
        bundle_values[key] = bundle
        pins[str(bundle_path)] = base.sha(Path(bundle_path))
        for baseline in bundle.get('baseline_prompt_paths', {}).values():
            pins[str(baseline)] = base.sha(Path(baseline))
        if spec["task_id"] != key:
            raise ValueError("Task identifier differs from catalog key")
        if set(spec["estimands"]) != {r["term"] for r in spec["reference"]["estimates"]}:
            raise ValueError(f"Reference term mismatch: {key}")
        path = Path(spec["data_path"])
        pins[str(path)] = base.sha(path)
    jobs = [dict(task=t, model=m, platform=l, prompt=p, condition=c, rep=r)
            for m in models for t, l, p, c, r in itertools.product(tasks, languages, prompts, conditions, range(reps[m]))]
    prompt_texts = {}
    for job in jobs:
        key = "__".join(job[k] for k in ("task", "platform", "prompt", "condition"))
        prompt_texts.setdefault(key, prompt(selected[job["task"]], job, bundle=bundle_values[job['task']]))
        baseline_key = '__'.join(job[k] for k in ('platform', 'prompt', 'condition'))
        baseline = bundle_values[job['task']].get('baseline_prompt_paths', {}).get(baseline_key)
        expected = Path(baseline).read_text() + (BOUNDARY if job['condition'] == 'agent' else '') if baseline else None
        if baseline and expected != prompt_texts[key]:
            raise ValueError(f'Initial prompt differs from reviewed archived text: {baseline_key}')
    commands = {}
    for model, condition in itertools.product(models, conditions):
        commands[f"{model}/{condition}"] = command(dict(model=model, condition=condition),
                                                   Path("<isolated-cwd>"), Path("<last-message>"))
    protocol = dict(policy=POLICY, tasks=selected, bundles=bundle_values, models={m: MODELS[m] for m in models},
                    prompts=prompt_texts, commands=commands, source_hashes=pins,
                    dimensions=dict(tasks=list(tasks), models=list(models), languages=list(languages),
                                    prompts=list(prompts), conditions=list(conditions), repetitions_by_model={m: reps[m] for m in models}))
    random.Random(20260916).shuffle(jobs)
    return {"created_at": now(), "status": "prepared; root controls authorized pilot launch",
            "protocol_sha256": digest(protocol), "protocol": protocol, "jobs": jobs}


def verify_manifest(manifest):
    if digest(manifest["protocol"]) != manifest["protocol_sha256"]:
        raise ValueError("Manifest protocol hash mismatch")
    drift = [path for path, expected in manifest["protocol"]["source_hashes"].items()
             if not Path(path).is_file() or base.sha(Path(path)) != expected]
    if drift:
        raise ValueError(f"Changed source or data. Prepare a new manifest: {drift}")


def require_authorization(manifest, path):
    if path is None:
        raise ValueError("Collection disabled without root's hash-bound pilot authorization file.")
    auth = json.loads(Path(path).read_text())
    if not (auth.get("prompt_conditions_approved") is True and auth.get("explicit_run_authorized") is True
            and auth.get("protocol_sha256") == manifest["protocol_sha256"]
            and auth.get("user_instruction_reference")):
        raise ValueError("Authorization must document the prompt decision, RUN instruction, and exact protocol hash")


def job_id(job):
    return "__".join(str(job[k]) for k in ("task", "model", "platform", "prompt", "condition")) + f"__rep{job['rep']:02d}"


def completed_record(folder, protocol_hash):
    """Only complete records count. Provider failures remain reviewable attempts."""
    for path in sorted(Path(folder).glob("attempt_*/run_record.json")):
        record = json.loads(path.read_text())
        if record.get("protocol_sha256") != protocol_hash:
            raise ValueError(f"Existing run belongs to another protocol: {path}")
        if record.get("status") == "completed":
            return record
    return None


def parse_call(model, raw, last_text, returncode, timed_out, stderr=""):
    if model == "luna":
        parsed = codex.parse_events(raw)
        error = parsed.get("error_message", "")
        result = dict(parsed, text=last_text or parsed.get("agent_text", ""))
        ok = base.codex_completed(parsed, returncode, timed_out)
        censor = timed_out
    else:
        events = harness.load_json_lines(raw)
        final = harness.final_stream_result(events) or {}
        error = "; ".join(map(str, final.get("errors") or []))
        if final.get("is_error"):
            error += " " + str(final.get("subtype", ""))
        result = dict(text=final.get("result", ""), usage=final.get("usage"), raw_meta=final,
                      num_tool_calls=harness.recursive_tool_use_count(events), num_turns=final.get("num_turns"),
                      total_cost_usd=final.get("total_cost_usd"), session_id=final.get("session_id"),
                      error_message=error)
        censor = timed_out or "max_turns" in error
        ok = returncode == 0 and bool(final) and not final.get("is_error") and not timed_out
    result.update(ok=ok, returncode=returncode, timed_out=timed_out, task_censored=censor,
                  provider_limit=not ok and (codex.is_usage_limit(error) or codex.is_usage_limit(stderr)),
                  infrastructure_failure=not ok and not censor, model=MODELS[model][0],
                  reasoning_effort=MODELS[model][1])
    events = harness.load_json_lines(raw)
    if model == 'sonnet':
        tool_items = [block for event in events if event.get('type') == 'assistant'
                      for block in event.get('message', {}).get('content', [])
                      if isinstance(block, dict) and block.get('type') in ('tool_use', 'server_tool_use', 'mcp_tool_use')]
        item_types = Counter(item.get('name', '<unnamed>') for item in tool_items)
        result.update(num_tool_calls=len(tool_items), item_types=dict(item_types),
                      unknown_item_types=sorted(set(item_types) - set(POLICY['allowed_sonnet_agent_tools'])),
                      event_types=dict(Counter(event.get('type', '<unnamed>') for event in events)), num_events=len(events))
        initialization = [event for event in events if event.get('type') == 'system' and event.get('subtype') == 'init']
    else:
        tool_items = [event.get('item') for event in events if event.get('type') == 'item.completed'
                      and event.get('item', {}).get('type') not in codex.NON_TOOL_ITEM_TYPES]
        initialization = [event for event in events if event.get('type') in ('thread.started', 'session_meta', 'session.created', 'init')]
    result['observable_runtime'] = dict(
        initialization_events=initialization,
        completed_tool_items=tool_items,
        effective_tool_catalog='not independently attested unless present in saved initialization events',
    )
    return result


def tool_violations(job, calls):
    sonnet = job['model'] == 'sonnet'
    allowed = set(POLICY['allowed_sonnet_agent_tools'] if sonnet else POLICY['allowed_agent_tool_item_types'])
    problems = []
    for index, call in enumerate(calls, 1):
        observed = set(call.get('item_types', {})) - (set() if sonnet else codex.NON_TOOL_ITEM_TYPES)
        unknown = set(call.get('unknown_item_types', []))
        if job['condition'] == 'agent':
            forbidden = (observed - allowed) | unknown
            if forbidden:
                problems.append(dict(attempt=index, unapproved_tool_item_types=sorted(forbidden)))
        elif call.get('num_tool_calls', 0) or observed or unknown:
            problems.append(dict(attempt=index, generation_tool_use=True, item_types=sorted(observed | unknown)))
        for item in call.get('observable_runtime', {}).get('completed_tool_items', []):
            command_text = str(item.get('input', {}).get('command', '')) if sonnet else str(item.get('command', item.get('arguments', '')))
            patterns = {
                'external_skill': r'\.agents/skills|\.codex/(?:skills|memories)|\.claude/(?:skills|commands)',
                'project_instruction': r'\b(?:AGENTS|CLAUDE)\.md\b',
                'benchmark_reference': r'reference_implementations|quality_reports|catalog\.json|reference_outputs?\.json',
                'network_or_git': r'https?://|\b(?:curl|wget|git)\s+',
                'parent_or_home_access': r'(?:^|[\s\x22\x27])\.\./|~/|\$HOME\b|Path\.home\(',
            }
            findings = [key for key, pattern in patterns.items() if re.search(pattern, command_text, re.I)]
            if findings:
                problems.append(dict(attempt=index, command_review_findings=findings, command=command_text))
    return problems


def subprocess_capture(argv, cwd, destination, stem, timeout, text=None):
    """Save complete streams and kill the process group after a chosen timeout."""
    stdout_path, stderr_path = destination / f"{stem}.stdout", destination / f"{stem}.stderr"
    start = time.monotonic()
    timed_out = False
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        with CHILDREN_LOCK:
            if STOP.is_set():
                raise RuntimeError('Collection stop requested before subprocess start')
            process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                       text=True, start_new_session=True)
            CHILDREN[process.pid] = process
        save(destination / f"{stem}.active.json", dict(pid=process.pid, started_at=now(), cwd=str(cwd)))
        try:
            process.communicate(text, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        finally:
            with CHILDREN_LOCK:
                CHILDREN.pop(process.pid, None)
    return dict(returncode=process.returncode, timed_out=timed_out,
                wall_seconds=time.monotonic() - start,
                stdout=stdout_path.read_text(errors="replace"), stderr=stderr_path.read_text(errors="replace"))


def call_model(text, cwd, job, destination, attempt):
    last = destination / f"last_attempt{attempt}.txt"
    argv, timeout = command(job, cwd, last)
    (destination / f"prompt_attempt{attempt}.txt").write_text(text)
    save(destination / f"command_attempt{attempt}.json", dict(argv=argv, cwd=str(cwd), timeout_seconds=timeout))
    run = subprocess_capture(argv, cwd, destination, f"provider_attempt{attempt}", timeout, text)
    result = parse_call(job["model"], run["stdout"], last.read_text() if last.exists() else "",
                        run["returncode"], run["timed_out"], run["stderr"])
    result.update(wall_seconds=run["wall_seconds"], timeout_seconds=timeout,
                  raw_stream_file=f"provider_attempt{attempt}.stdout", stderr_file=f"provider_attempt{attempt}.stderr")
    last.write_text(result["text"])
    save(destination / f"call_attempt{attempt}.json", result)
    return result


def execute(script, platform, cwd, destination, attempt):
    if platform == "python":
        argv = [sys.executable, str(script)]
    elif platform == "r":
        argv = ["Rscript", "--vanilla", str(script)]
    else:
        argv = [str(harness.STATA_BIN), "-b", "do", str(script)]
    result = subprocess_capture(argv, cwd, destination, f"execution_attempt{attempt}", 180)
    log = (cwd / "run.log").read_text(errors="replace") if (cwd / "run.log").exists() else ""
    result["ok"] = result["returncode"] == 0 and not result["timed_out"]
    if platform == "stata" and harness.STATA_ERROR_RE.search(log):
        result["ok"] = False
    result["stata_log_tail"] = log[-3000:]
    save(destination / f"execution_attempt{attempt}.json", result)
    return result


def snapshot(cwd, destination):
    """Keep symlinks as links, so snapshotting cannot traverse external files."""
    shutil.copytree(cwd, destination, symlinks=True)


@contextmanager
def preserved_workspace(destination):
    """Preserve files even when execution or parsing raises an exception."""
    with tempfile.TemporaryDirectory(prefix='two_prompt_run_', dir='/private/tmp') as temporary:
        cwd = Path(temporary)
        try:
            yield cwd
        finally:
            if not (destination / 'final_workspace').exists():
                snapshot(cwd, destination / 'final_workspace')


def reset_workspace(cwd, source):
    for path in cwd.iterdir():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    shutil.copyfile(source, cwd / 'input.csv')


def run_job(job, manifest, run_root, model_call=call_model, execute_call=execute):
    """Dependency injection permits complete offline collection tests."""
    protocol_hash = manifest["protocol_sha256"]
    folder = Path(run_root) / job_id(job)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = completed_record(folder, protocol_hash)
        if existing is not None:
            return existing
        if STOP.is_set():
            return None
        verify_manifest(manifest)
        attempt_id = len(list(folder.glob("attempt_*"))) + 1
        destination = folder / f"attempt_{attempt_id:04d}"
        destination.mkdir()
        spec = manifest["protocol"]["tasks"][job["task"]]
        record = dict(job=job, protocol_sha256=protocol_hash, started_at=now(), calls=[], executions=[],
                      attempt_directory=str(destination), status="in_progress")
        save(destination / "started.json", record)
        start = time.monotonic()
        try:
            with preserved_workspace(destination) as cwd:
                shutil.copyfile(spec["data_path"], cwd / "input.csv")
                record["input_sha256"] = base.sha(cwd / "input.csv")
                previous, feedback = None, None
                for attempt in range(1, 5 if job["condition"] == "self_repair" else 2):
                    if STOP.is_set():
                        raise RuntimeError('Collection stop requested before the next provider call')
                    verify_manifest(manifest)
                    if job['condition'] != 'agent':
                        reset_workspace(cwd, spec['data_path'])
                    text = prompt(spec, job, previous, feedback, manifest['protocol']['bundles'][job['task']])
                    if job["condition"] == "agent":
                        with STATA_SLOT if job['platform'] == 'stata' else nullcontext():
                            call = model_call(text, cwd, job, destination, attempt)
                    else:
                        with tempfile.TemporaryDirectory(prefix="two_prompt_generate_", dir="/private/tmp") as generated:
                            call = model_call(text, Path(generated), job, destination, attempt)
                    record["calls"].append(call)
                    if not call["ok"]:
                        break
                    if job["condition"] == "agent":
                        break
                    if call.get("num_tool_calls", 0) or call.get('unknown_item_types'):
                        break
                    code = harness.extract_code(call.get("text", ""))
                    if code is None:
                        feedback, previous = "No fenced script was returned.", call.get("text", "")
                        record["executions"].append(dict(ok=False, error=feedback))
                        continue
                    script = cwd / {"python": "run.py", "r": "run.R", "stata": "run.do"}[job["platform"]]
                    script.write_text(code)
                    with STATA_SLOT if job['platform'] == 'stata' else nullcontext():
                        execution = execute_call(script, job["platform"], cwd, destination, attempt)
                    record["executions"].append(execution)
                    snapshot(cwd, destination / f"workspace_attempt{attempt}")
                    problem = base.feedback_output(cwd, spec, job["platform"])
                    if execution["ok"] and problem is None:
                        break
                    previous = code
                    # Preserve full streams locally. Match prior bounded feedback tails.
                    feedback = json.dumps({key: (value[-1500:] if key in ('stdout', 'stderr') else value)
                                           for key, value in execution.items()}) + "\n" + str(problem or "")
                record["final_input_sha256"] = base.sha(cwd / "input.csv") if (cwd / "input.csv").exists() else None
                snapshot(cwd, destination / "final_workspace")
                estimates, repair, notes = base.load_output(cwd, spec, job["platform"])
                refs = {spec["task_id"]: spec["reference"]}
                numeric, terms, numeric_notes = sp.compare_numeric(estimates, base.as_task(spec), refs)
                integrity, integrity_notes = sp.score_integrity(base.as_task(spec), cwd,
                                                                sp.find_script(cwd, job["platform"]), refs)
                integrity = bool(integrity and record["final_input_sha256"] == record["input_sha256"])
                record["scoring"] = dict(numeric_success=numeric, integrity_pass=integrity, estimates=estimates,
                                         per_term=terms, repair_rule=repair, notes=notes + numeric_notes + integrity_notes)
        except Exception as error:
            record["driver_error"] = repr(error)
            record.setdefault("scoring", dict(numeric_success=False, integrity_pass=False))
        record["task_censored"] = any(c.get("task_censored") for c in record["calls"])
        record["provider_ok"] = bool(record["calls"]) and all(c["ok"] for c in record["calls"])
        record['tool_policy_findings'] = tool_violations(job, record['calls'])
        record["protocol_violation"] = bool(record['tool_policy_findings'])
        record["infrastructure_failure"] = any(c.get("infrastructure_failure") for c in record["calls"])
        record["provider_limit"] = any(c.get("provider_limit") for c in record["calls"])
        interrupted = bool(record.get("driver_error") or record["infrastructure_failure"] or record["provider_limit"]
                           or record["protocol_violation"])
        record["status"] = "needs_review" if interrupted else "completed"
        record["scoring"]["success"] = bool(record["scoring"]["numeric_success"] and record["scoring"]["integrity_pass"]
                                               and record["provider_ok"] and not record["task_censored"]
                                               and not record["protocol_violation"] and not record.get("driver_error"))
        record.update(completed_at=now(), total_wall_seconds=time.monotonic() - start)
        save(destination / "run_record.json", record)
        if interrupted:
            STOP.set()
        checkpoint("job", id=job_id(job), status=record["status"], success=record["scoring"]["success"])
        return record


def status(manifest, run_root):
    counts = Counter()
    for job in manifest["jobs"]:
        folder = Path(run_root) / job_id(job)
        done = completed_record(folder, manifest["protocol_sha256"])
        counts["completed" if done else "needs_review" if list(folder.glob("attempt_*")) else "pending"] += 1
    return dict(status="root controls authorized pilot launch", total=len(manifest["jobs"]), **counts)


def preflight_jobs(manifest):
    cells = [('luna', 'python', 'zero_shot', 'chatbot'), ('luna', 'r', 'few_shot', 'self_repair'),
             ('luna', 'stata', 'zero_shot', 'agent'), ('sonnet', 'python', 'few_shot', 'chatbot'),
             ('sonnet', 'r', 'zero_shot', 'agent'), ('sonnet', 'stata', 'few_shot', 'self_repair')]
    lookup = {(job['model'], job['platform'], job['prompt'], job['condition']): job
              for job in manifest['jobs'] if job['rep'] == 0}
    return [lookup[cell] for cell in cells if cell in lookup]


def select_jobs(manifest, path=None):
    lookup = {job_id(job): job for job in manifest['jobs']}
    if path is None:
        return list(manifest['jobs'])
    selected = json.loads(Path(path).read_text())
    ids = [job_id(job) if isinstance(job, dict) else job for job in selected]
    if len(ids) != len(set(ids)) or not set(ids) <= set(lookup):
        raise ValueError('Job subset contains duplicates or jobs outside the authorized manifest')
    return [lookup[key] for key in ids]


def require_preflight_review(manifest, selected, run_root, path):
    initial = preflight_jobs(manifest)
    initial_ids = {job_id(job) for job in initial}
    if {job_id(job) for job in selected} <= initial_ids:
        return
    if path is None:
        raise ValueError('Remaining runs require root review of the first six pilot runs')
    review = json.loads(Path(path).read_text())
    if not (review.get('approved_to_continue') is True and review.get('protocol_sha256') == manifest['protocol_sha256']):
        raise ValueError('Preflight review does not approve this protocol')
    if any(completed_record(Path(run_root) / job_id(job), manifest['protocol_sha256']) is None for job in initial):
        raise ValueError('The predefined first six runs are not all complete')


def stop_children(signum, frame):
    STOP.set()
    with CHILDREN_LOCK:
        for process in list(CHILDREN.values()):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def check_tools():
    """Local help/version inspection only. No authentication or provider requests."""
    queries = {"codex_help": [codex.CODEX_BIN, "exec", "--help"],
               "codex_features": [codex.CODEX_BIN, "features", "list"],
               "codex_version": [codex.CODEX_BIN, "--version"],
               "sonnet_help": [harness.claude_bin(), "--help"],
               "sonnet_version": [harness.claude_bin(), "--version"],
               "r_version": ["Rscript", "--version"]}
    result = {}
    for key, argv in queries.items():
        process = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        result[key] = dict(argv=argv, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
        if process.returncode:
            raise ValueError(f"Local CLI check failed: {key}")
    for key, flags in {"codex_help": ("--ignore-user-config", "--ignore-rules"),
                       "codex_features": ("skip_host_skill_discovery", "plugins"),
                       "sonnet_help": ('--safe-mode', '--setting-sources', '--strict-mcp-config', '--tools', '--effort', '--model')}.items():
        for flag in flags:
            if flag not in result[key]["stdout"]:
                raise ValueError(f"Installed CLI lacks {flag}")
    result["stata_binary"] = str(harness.STATA_BIN)
    result["stata_exists"] = harness.STATA_BIN.exists()
    if not result["stata_exists"]:
        raise ValueError("Stata executable is unavailable")
    return result


def collect_queue(pending, manifest, run_root, workers=6, run=None):
    """Refill each free worker without waiting for the other active jobs."""
    run = run_job if run is None else run
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}

        def refill():
            while len(active) < workers and not STOP.is_set():
                job = next(iterator, None)
                if job is None:
                    return
                active[pool.submit(run, job, manifest, run_root)] = job

        refill()
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                active.pop(future)
                try:
                    result = future.result()
                except BaseException:
                    STOP.set()
                    raise
                if result is not None:
                    print(json.dumps(dict(id=job_id(result['job']), status=result['status'],
                                          success=result['scoring']['success'])), flush=True)
            refill()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prep", aliases=['dry-run'], help="Save a hash-bound pilot protocol without provider calls")
    prepare.add_argument('--tasks', nargs='+')
    for key, values in (("models", tuple(MODELS)), ("languages", LANGUAGES),
                        ("prompts", PROMPTS), ("conditions", CONDITIONS)):
        prepare.add_argument("--" + key, nargs="+", choices=values, default=list(values))
    prepare.add_argument('--luna-reps', type=int, default=3)
    prepare.add_argument('--sonnet-reps', type=int, default=2)
    prepare.add_argument("--catalog", type=Path, action="append", required=True, help="Reviewed one-task catalog")
    prepare.add_argument('--bundle', type=Path, required=True, help='Reviewed exact one-example bundle')
    for action in (prepare, sub.add_parser("check", help="Verify hashes and local CLI flags, offline"),
                   sub.add_parser("status", help="Show saved run status"),
                   sub.add_parser("collect", help="Root-controlled authorized pilot launch")):
        action.add_argument("--manifest", type=Path, default=WORK / "manifest.json")
    collect = sub.choices["collect"]
    collect.add_argument("--authorization", type=Path)
    collect.add_argument("--workers", type=int, default=6, choices=range(1, 7))
    collect.add_argument("--limit", type=int, help="Run at most this many pending jobs")
    collect.add_argument('--jobs-file', type=Path, help='A predefined subset of the manifest, such as preflight_jobs.json')
    collect.add_argument('--preflight-review', type=Path, help='Root approval after inspecting the first six runs')
    collect.add_argument("--retry-reviewed", action="store_true", help="Retain and retry previously interrupted attempts")
    args = parser.parse_args(argv)
    if args.action in ("prep", "dry-run"):
        bundle = json.loads(args.bundle.read_text())
        manifest = make_manifest(args.tasks, args.models, args.languages, args.prompts, args.conditions,
                                 {'luna': args.luna_reps, 'sonnet': args.sonnet_reps}, args.catalog,
                                 {bundle['task_id']: args.bundle})
        if args.manifest.exists():
            old = json.loads(args.manifest.read_text())
            if old["protocol_sha256"] != manifest["protocol_sha256"]:
                raise ValueError("Manifest already exists with another protocol. Choose a new --manifest path.")
        save(args.manifest, manifest)
        save(args.manifest.parent / 'preflight_jobs.json', preflight_jobs(manifest))
        prompt_directory = args.manifest.parent / 'prompts'
        prompt_directory.mkdir(exist_ok=True)
        for key, text in manifest['protocol']['prompts'].items():
            (prompt_directory / (key + '.txt')).write_text(text)
        checkpoint("dry_run", manifest=str(args.manifest), jobs=len(manifest["jobs"]))
        print(json.dumps(dict(manifest=str(args.manifest), jobs=len(manifest["jobs"]),
                              protocol_sha256=manifest["protocol_sha256"], status=manifest["status"])))
        return
    manifest = json.loads(args.manifest.read_text())
    run_root = WORK / "runs" / manifest["protocol_sha256"]
    if args.action == "status":
        print(json.dumps(status(manifest, run_root), indent=2))
        return
    verify_manifest(manifest)
    if args.action == "check":
        tools = check_tools()
        save(WORK / "offline_check.json", dict(checked_at=now(), protocol_sha256=manifest["protocol_sha256"],
                                               result="passed", provider_calls=0, tools=tools))
        checkpoint("offline_check", result="passed")
        print(json.dumps(dict(result="passed", provider_calls=0, jobs=len(manifest["jobs"]))))
        return
    require_authorization(manifest, args.authorization)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Worker and run limits must be positive")
    check_tools()
    selected = select_jobs(manifest, args.jobs_file)
    require_preflight_review(manifest, selected, run_root, args.preflight_review)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop_children)
    with (WORK / "collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for job in selected:
            folder = run_root / job_id(job)
            if completed_record(folder, manifest["protocol_sha256"]):
                continue
            if list(folder.glob("attempt_*")) and not args.retry_reviewed:
                raise ValueError(f"Review interrupted attempt before retry: {folder}")
            pending.append(job)
        pending = pending[:args.limit] if args.limit else pending
        collect_queue(pending, manifest, run_root, args.workers)


if __name__ == "__main__":
    main()

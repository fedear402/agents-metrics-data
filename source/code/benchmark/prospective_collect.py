"""Collect new manifest-defined slots without changing historical collections.

prep --config CONFIG --manifest NEW_MANIFEST
check|status --manifest MANIFEST
collect --manifest MANIFEST --authorization AUTH [--jobs-file JOBS] --workers 4

AUTH contains protocol_sha256 and allow_provider_calls=true. Preparation and
checks use local files and CLI help only. Quota rejection resumes automatically
only from a recorded future reset timestamp and an unstarted model response.
Interrupted or ambiguous requests require review. Model failures are terminal.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
try:
    import prospective_outputs
    from collection_runtime import load_runtime_tree
except ModuleNotFoundError:
    from benchmark import prospective_outputs
    from benchmark.collection_runtime import load_runtime_tree

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'code/benchmark/collection_runtime/clearer_examples.py'
VERSION = 'prospective_manifest_runtime_2026_09_17_v11_post_collection_review'
AGENT_VERIFICATION_VERSION = 'agent_statistical_verification_v1'
AGENT_VERIFICATION_INSTRUCTION = (
    'Save the final solution as run.py for Python, run.R for R, or run.do for Stata.\n'
    'Before submission, verify the statistical calculation using the supplied data.\n'
    'Check the estimation sample, observation count, and variable construction against the task.\n'
    'Check the estimator and uncertainty calculation against the stated specification.\n'
    'Run at least one independent numerical check of a central calculation.\n'
    'Use a separate calculation or a mathematical identity that applies to this task.\n'
    'Inspect any discrepancy and correct errors supported by these checks.\n'
    'After edits, rerun the final script and inspect its requested output.\n'
    'Save a short plain-text account of the checks and their results in verification.txt.\n'
    'Use only the task statement, supplied data, installed software, and files you create.\n'
    'After these checks, finish with DONE.'
)
DOC = 'https://code.claude.com/docs/en/agent-sdk/python#ratelimitinfo'
STOP = threading.Event()
TERMINAL = {'completed', 'model_failure', 'provider_timeout', 'execution_timeout'}
# Chosen review policy, not a measured provider limit. Source pins cover changes.
MAX_QUOTA_RESET_HORIZON_SECONDS = 24 * 60 * 60


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(prospective_outputs.json_safe(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def prompt_key(job):
    return '__'.join(job[k] for k in ('task', 'platform', 'prompt', 'condition', 'output_format'))


def prepared_prompt(text, condition):
    """Version the agent completion instruction without editing source prompts."""
    if condition != 'agent':
        return text
    previous = 'When the requested output is present, finish with DONE.'
    if text.count(AGENT_VERIFICATION_INSTRUCTION) == 1 and previous not in text:
        return text
    if text.count(previous) != 1 or AGENT_VERIFICATION_INSTRUCTION in text:
        raise ValueError('Agent prompt needs one unambiguous completion instruction')
    return text.replace(previous, AGENT_VERIFICATION_INSTRUCTION)


def job_id(job):
    parts = [str(job[k]) for k in ('block', 'task', 'model', 'platform', 'prompt', 'condition', 'output_format')]
    if any(not part or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in part) for part in parts):
        raise ValueError('Unsafe job identifier')
    return '__'.join(parts) + f'__rep{job["rep"]:02d}'


def access_scan_text(text):
    """Keep Stata local-macro delimiters distinct from quoted file paths."""
    # The apostrophe closes a macro token. It does not open a quoted path.
    # Keep surrounding quotes and slashes so actual paths remain visible.
    return re.sub(r"`([A-Za-z_][A-Za-z0-9_]*)'", r'(STATA_LOCAL_\1)', str(text))


def scan_shell_command(text):
    """Normalize installed system commands in shell tool fields only."""
    command = r'(?P<prefix>(?:^|[\n;&|])[ \t]*)(?P<exe>/(?:usr/bin|bin)/[A-Za-z0-9_.+-]+)(?=[ \t]+\S)'
    def executable(match):
        path = Path(match.group('exe'))
        if path.is_file() and os.access(path, os.X_OK):
            return match.group('prefix') + path.name
        return match.group(0)
    normalized = re.sub(command, executable, str(text))
    shell = r'(?P<prefix>(?:^|[\n;&|])[ \t]*(?:/(?:usr/bin|bin)/)?(?:sh|bash|zsh)[ \t]+-l?c[ \t]+["\x27])(?P<exe>/(?:usr/bin|bin)/[A-Za-z0-9_.+-]+)(?=[ \t]+\S)'
    return re.sub(shell, executable, normalized)


def calls_for_access_review(calls):
    """Keep raw calls unchanged while identifying executable shell commands."""
    reviewed = copy.deepcopy(calls)
    for call in reviewed:
        runtime = call.get('observable_runtime', {})
        for field in ('all_tool_items', 'completed_tool_items'):
            for item in runtime.get(field, []):
                if item.get('type') == 'command_execution' and isinstance(item.get('command'), str):
                    item['command'] = scan_shell_command(item['command'])
                if item.get('name') == 'Bash' and isinstance(item.get('input'), dict) and isinstance(item['input'].get('command'), str):
                    item['input']['command'] = scan_shell_command(item['input']['command'])
    return reviewed


def load_runtime(work):
    module = load_runtime_tree()
    module.c.WORK = module.m.WORK = Path(work)
    module.c.STOP = module.c.prior.STOP = STOP
    scanner = module.c.access_findings
    module.c.access_findings = lambda text, cwd: scanner(access_scan_text(text), cwd)
    tool_scanner = module.c.tool_violations
    module.c.tool_violations = lambda job, calls, cwd: tool_scanner(job, calls_for_access_review(calls), cwd)
    executor = module.c.prior.execute
    module.c.prior.execute = lambda script, platform, cwd, destination, attempt: prospective_outputs.execute_submission(
        script, platform, cwd, destination, attempt, executor=executor, harness=module.c.harness)
    output_loader = module.c.load_output
    module.c.load_output = lambda cwd, spec, job, execution: prospective_outputs.load_submission_output(
        cwd, spec, job, execution, loader=output_loader)
    return module


def runtime_sources(runtime):
    """Pin imported repository modules, including modules loaded outside sys.modules."""
    seen, files = set(), {Path(__file__).resolve(), RUNTIME}
    pending = [runtime, runtime_sources]
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, types.ModuleType):
            filename, namespace = getattr(value, '__file__', None), vars(value)
        elif isinstance(value, types.FunctionType):
            filename, namespace = value.__code__.co_filename, value.__globals__
        elif isinstance(value, type):
            module = sys.modules.get(value.__module__)
            if module is not None:
                pending.append(module)
            continue
        else:
            continue
        if not filename:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(ROOT) or path.is_relative_to(ROOT / '.venv') or path.suffix != '.py':
            continue
        files.add(path)
        pending.extend(item for item in namespace.values()
                       if isinstance(item, (types.ModuleType, types.FunctionType, type)))
    return sorted(files)


def native_environment(runtime, jobs):
    """Measure required native interpreters and installed packages without task execution."""
    platforms = {job['platform'] for job in jobs}
    if not platforms & {'r', 'stata'}:
        return {}
    result = {}
    env = runtime.c.environment()
    if 'r' in platforms:
        executable = shutil.which('Rscript', path=env['PATH'])
        if executable is None:
            raise ValueError('Rscript executable is unavailable')
        expression = ('cat(R.version.string, "\\n"); '
                      'p <- installed.packages(noCache=TRUE); '
                      'p <- p[order(p[,"Package"], p[,"LibPath"]), c("Package","Version","LibPath","Built"), drop=FALSE]; '
                      'write.table(p, stdout(), sep="\\t", quote=TRUE, row.names=FALSE)')
        process = subprocess.run([executable, '--vanilla', '-e', expression], env=env,
                                 capture_output=True, text=True, timeout=30)
        if process.returncode or not process.stdout.startswith('R version '):
            raise ValueError('R environment inspection failed: ' + process.stderr)
        result['r'] = dict(executable=executable, resolved_executable=str(Path(executable).resolve()),
                           executable_sha256=sha(executable), version_and_packages=process.stdout.strip())
    if 'stata' in platforms:
        executable = runtime.c.harness.STATA_BIN
        if not Path(executable).is_file() or not os.access(executable, os.X_OK):
            raise ValueError('Stata executable is unavailable')
        with tempfile.TemporaryDirectory(prefix='prospective_stata_identity_', dir='/private/tmp') as folder:
            directory = Path(folder)
            # The identity file omits license names and contains no task data.
            script = ('clear all\nset more off\nfile open identity using "identity.txt", write replace\n'
                      'file write identity "version=" (c(stata_version)) _n\n'
                      'file write identity "born=" (c(born_date)) _n\n'
                      'file write identity "flavor=" (c(flavor)) _n\n'
                      'file write identity "bits=" (c(bit)) _n\n')
            for name in ('base', 'site', 'plus', 'personal', 'oldplace'):
                script += f'file write identity "{name}=" (c(sysdir_{name})) _n\n'
            script += 'file write identity "complete=yes" _n\nfile close identity\nexit, clear\n'
            (directory / 'identity.do').write_text(script)
            with open('/private/tmp/agents_metrics_native_stata.lock', 'a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                process = subprocess.run([str(executable), '-b', 'do', 'identity.do'], cwd=directory,
                                         env=env, capture_output=True, text=True, timeout=30)
            identity_path = directory / 'identity.txt'
            text = identity_path.read_text() if identity_path.exists() else ''
            if process.returncode or not text.endswith('complete=yes\n'):
                raise ValueError('Stata environment inspection failed')
            identity = dict(line.split('=', 1) for line in text.splitlines())
        packages = {}
        for name in ('base', 'site', 'plus', 'personal', 'oldplace'):
            path = Path(identity[name]).expanduser()
            packages[name] = {str(item.relative_to(path)): sha(item)
                              for item in sorted(path.rglob('*')) if item.is_file()}
        result['stata'] = dict(executable=str(executable), resolved_executable=str(Path(executable).resolve()),
                               executable_sha256=sha(executable), identity=identity,
                               package_files_sha256=digest(packages),
                               package_file_counts={name: len(files) for name, files in packages.items()})
    return result


def verify_native_environment(expected, runtime, jobs):
    if native_environment(runtime, jobs) != expected:
        raise ValueError('Native environment changed')


def provider_cli_versions(runtime, jobs):
    """Check only requested provider CLIs through local help and version commands."""
    models = {job['model'] for job in jobs}
    queries = {}
    flags = {}
    if 'luna' in models:
        executable = runtime.c.codex.CODEX_BIN
        queries.update(codex_version=[executable, '--version'], codex_help=[executable, 'exec', '--help'],
                       codex_features=[executable, 'features', 'list'])
        flags.update(codex_help=('--ignore-user-config', '--ignore-rules'),
                     codex_features=('skip_host_skill_discovery', 'plugins'))
    if 'sonnet' in models:
        executable = runtime.c.harness.claude_bin()
        queries.update(sonnet_version=[executable, '--version'], sonnet_help=[executable, '--help'])
        flags['sonnet_help'] = ('--safe-mode', '--setting-sources', '--strict-mcp-config', '--tools',
                                '--effort', '--model', '--disable-slash-commands')
    versions = {}
    for name, argv in queries.items():
        process = subprocess.run(argv, env=runtime.c.environment(), capture_output=True, text=True, timeout=30)
        if process.returncode:
            raise ValueError('Local CLI check failed: ' + name)
        for flag in flags.get(name, ()):
            if flag not in process.stdout:
                raise ValueError('Installed CLI lacks ' + flag)
        if name.endswith('_version'):
            versions[name] = dict(argv=argv, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
    return versions


def validate_jobs(jobs, protocol):
    identifiers = set()
    slots = set()
    for job in jobs:
        identity = job_id(job)
        slot = tuple(job[k] for k in ('task', 'model', 'platform', 'prompt', 'condition', 'output_format', 'rep'))
        if identity in identifiers or slot in slots:
            raise ValueError('Duplicate planned slot')
        identifiers.add(identity); slots.add(slot)
        if job['model'] not in {'luna', 'sonnet'} or job['condition'] not in {'chatbot', 'self_repair', 'agent'}:
            raise ValueError('Unsupported model or agency')
        if job['platform'] not in {'python', 'r', 'stata'} or type(job['rep']) is not int or not 0 <= job['rep'] < 5:
            raise ValueError('Unsupported language or repetition')
        if job['output_format'] != ('answer' if job['platform'] == 'stata' else 'json'):
            raise ValueError('Use printed Stata and JSON Python/R')
        if job['task'] not in protocol['tasks'] or prompt_key(job) not in protocol['prompts']:
            raise ValueError('Missing task or exact prompt')


def task_input_files(spec):
    """Return validated workspace filenames and absolute source paths."""
    primary = Path(spec['data_path'])
    if not primary.is_absolute():
        raise ValueError('Catalog data_path must be absolute')
    inputs = {'input.csv': primary}
    extra = spec.get('input_files', {})
    if not isinstance(extra, dict):
        raise ValueError('Catalog input_files must map filenames to absolute paths')
    for name, value in extra.items():
        if not isinstance(name, str) or not name or name in ('.', '..') or '/' in name or '\\' in name:
            raise ValueError('Catalog input_files keys must be flat filenames')
        if name in ('run.py', 'run.R', 'run.do', 'result.json', 'run.log', 'printed_output.txt'):
            raise ValueError('Catalog input filename conflicts with collector output')
        path = Path(value)
        if not path.is_absolute():
            raise ValueError('Catalog input_files paths must be absolute')
        if name == 'input.csv' and path != primary:
            raise ValueError('Catalog input.csv must match data_path')
        inputs[name] = path
    return inputs


def inputs_intact(cwd, inputs):
    """Check every required input, including missing files and file links."""
    return all(not (cwd / name).is_symlink() and (cwd / name).is_file()
               and sha(cwd / name) == sha(source) for name, source in inputs.items())


def copy_inputs(cwd, inputs):
    for name, source in inputs.items():
        shutil.copyfile(source, cwd / name)


def enforce_integer_terms(estimates, spec, comparison):
    """Apply exact integer equality only to explicitly declared estimate terms."""
    declared = spec.get('exact_integer_terms', [])
    if not declared:
        return comparison
    good, terms, notes = comparison
    references = {row['term']: row for row in spec['reference']['estimates']}
    for term in declared:
        got = estimates.get(term, {}).get('estimate')
        wanted = references.get(term, {}).get('estimate')
        try:
            passed = (not isinstance(got, bool) and not isinstance(wanted, bool)
                      and math.isfinite(float(got)) and math.isfinite(float(wanted))
                      and float(got).is_integer() and float(wanted).is_integer()
                      and got == wanted)
        except (TypeError, ValueError, OverflowError):
            passed = False
        detail = terms.setdefault(term, {})
        detail['exact_estimate_pass'] = bool(passed)
        if not passed:
            detail['coef_pass'] = False
            good = False
            notes.append(f'{term}: exact integer estimate mismatch')
    return bool(good), terms, notes


def prepare(config_path, manifest_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    def source(value):
        p = Path(value)
        return p.resolve() if p.is_absolute() else (config_path.parent / p).resolve()
    runtime = load_runtime(Path(manifest_path).parent)
    catalog_path = source(config['catalog_path'])
    catalog = json.loads(catalog_path.read_text())
    jobs = config['jobs']
    tasks = {key: catalog[key] for key in sorted({j['task'] for j in jobs})}
    prompts, pins, prompt_provenance = {}, {}, {}
    sources = [config_path, catalog_path, source(config['decision_record']),
               *sorted((Path(__file__).parent / 'tests').glob('test_prospective*.py')), *runtime_sources(runtime)]
    for key, condition in {prompt_key(job): job['condition'] for job in jobs}.items():
        path = source(config['prompt_paths'][key])
        prompts[key] = prepared_prompt(path.read_text(), condition)
        prompt_provenance[key] = dict(source_path=str(path), source_sha256=sha(path),
            effective_sha256=hashlib.sha256(prompts[key].encode()).hexdigest(),
            agent_verification_version=AGENT_VERIFICATION_VERSION if condition == 'agent' else None)
        sources.append(path)
    for task in tasks.values():
        sources.extend(task_input_files(task).values())
    for path in sources:
        pins[str(path)] = sha(path)
    commands = {f'{j["model"]}/{j["condition"]}': runtime.c.command(j, Path('<cwd>'), Path('<last>')) for j in jobs}
    if any('--max-turns' in argv for argv, _ in commands.values()):
        raise ValueError('Provider turn caps are prohibited')
    checks = provider_cli_versions(runtime, jobs)
    package_names = ['numpy', 'pandas', 'scipy', 'statsmodels', 'linearmodels']
    protocol = dict(version=VERSION, tasks=tasks, prompts=prompts, prompt_provenance=prompt_provenance,
        source_hashes=pins, commands=commands,
        decision_record=str(source(config['decision_record'])),
        policy=dict(provider_turn_cap=None, self_repair_executions=4,
                    max_repetitions=5,
                    access_review='Post-collection review of saved telemetry, script access, and reference literals; does not stop collection.',
                    quota_reset_review_horizon_seconds=MAX_QUOTA_RESET_HORIZON_SECONDS,
                    agent_verification_version=AGENT_VERIFICATION_VERSION,
                    quota='Pause provider on rejected; resume at recorded future reset only; warning is nonblocking.',
                    interruption='Hold incomplete requests; resume completed-response local work without provider repeat.',
                    score='Final submitted script executed locally; numeric and integrity; model-side timeout is primary failure; operational violations require review.',
                    provider_completion='Exclude explicit quota rejection before model response from completion status; preserve every call.',
                    sonnet_cost_per_million=dict(input=3, output=15)),
        cli_versions=checks,
        environment=dict(executable=sys.executable, python=sys.version,
                         packages={name: importlib.metadata.version(name) for name in package_names}),
        native_environment=native_environment(runtime, jobs),
        runtime_source=str(RUNTIME), quota_documentation=DOC)
    validate_jobs(jobs, protocol)
    # Job membership and order are part of the authorization hash.
    protocol['planned_jobs'] = jobs
    manifest = dict(created_at=now(), protocol=protocol, protocol_sha256=digest(protocol), jobs=jobs)
    target = Path(manifest_path)
    if target.exists():
        old = json.loads(target.read_text())
        if old['protocol_sha256'] != manifest['protocol_sha256']:
            raise ValueError('Preserve the existing manifest; select a new destination')
        return old
    save(target, manifest)
    return manifest


def verify(manifest, runtime=None):
    protocol = manifest['protocol']
    if digest(protocol) != manifest['protocol_sha256'] or manifest['jobs'] != protocol['planned_jobs']:
        raise ValueError('Manifest identity changed')
    validate_jobs(manifest['jobs'], protocol)
    for filename, expected in protocol['source_hashes'].items():
        if sha(filename) != expected:
            raise ValueError('Pinned source changed: ' + filename)
    env = protocol['environment']
    if env['executable'] != sys.executable or env['python'] != sys.version:
        raise ValueError('Python environment changed')
    for name, version in env['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Package changed: ' + name)
    if runtime is not None:
        verify_native_environment(protocol.get('native_environment', {}), runtime, manifest['jobs'])
        for job in manifest['jobs']:
            command = runtime.c.command(job, Path('<cwd>'), Path('<last>'))
            if digest(command) != digest(protocol['commands'][f'{job["model"]}/{job["condition"]}']):
                raise ValueError('Provider command changed')
        current = provider_cli_versions(runtime, manifest['jobs'])
        for key, saved in protocol['cli_versions'].items():
            if runtime.m.version_identity(current[key]) != runtime.m.version_identity(saved):
                raise ValueError('Provider CLI version changed')


def parse_quota(events, observed_at=None):
    """Use top-level status, never an overage rejection inside an allowed event."""
    observed_at = time.time() if observed_at is None else observed_at
    rejected = []
    for event in events:
        info = event.get('rate_limit_info', event.get('rateLimitInfo', {}))
        if info.get('status') != 'rejected':
            continue
        reset = next((info[k] for k in ('resetsAt', 'resetAt', 'resets_at') if info.get(k) is not None), None)
        if isinstance(reset, bool) or not isinstance(reset, (int, float)) or not math.isfinite(reset) or not 0 < reset - observed_at <= MAX_QUOTA_RESET_HORIZON_SECONDS:
            reset = None
        rejected.append(dict(reset_at=reset, info=info))
    if not rejected:
        return None
    # An unresolved rejection cannot be cleared by another window's timestamp.
    reset = None if any(x['reset_at'] is None for x in rejected) else max(x['reset_at'] for x in rejected)
    return dict(status='rejected', reset_at=reset, evidence=rejected,
                observed_at=observed_at)


class QuotaUnknown(RuntimeError):
    pass


class Paused(RuntimeError):
    pass


class QuotaCoordinator:
    """One shared provider pause, persisted across collector restarts."""
    def __init__(self, path, stop=STOP, clock=time.time, sleep=None):
        self.path, self.stop, self.clock = Path(path), stop, clock
        self.sleep = sleep or (lambda seconds: stop.wait(seconds))
        self.lock = threading.RLock()
        self.probes = set()
        self.revisions = {}
        self.probe_revisions = {}
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {}

    def reject(self, provider, quota):
        with self.lock:
            old = self.state.get(provider)
            reset = quota['reset_at']
            if old:
                # Every outstanding rejection needs a known reset.
                reset = None if old.get('reset_at') is None or reset is None else max(old['reset_at'], reset)
            self.state[provider] = dict(quota, reset_at=reset)
            self.revisions[provider] = self.revisions.get(provider, 0) + 1
            save(self.path, self.state)

    def acquire(self, provider):
        while not self.stop.is_set():
            with self.lock:
                state = self.state.get(provider)
                if not state:
                    return False
                reset = state.get('reset_at')
                if isinstance(reset, bool) or not isinstance(reset, (int, float)) or not math.isfinite(reset) or not 0 < reset - state['observed_at'] <= MAX_QUOTA_RESET_HORIZON_SECONDS:
                    raise QuotaUnknown('Rejected quota lacks a usable future reset timestamp')
                if self.clock() >= reset and provider not in self.probes:
                    self.probes.add(provider)
                    self.probe_revisions[provider] = self.revisions.get(provider, 0)
                    return True
                remaining = max(0.01, reset - self.clock())
            self.sleep(min(1.0, remaining))
        raise Paused('User requested pause')

    def release(self, provider, probe, quota=None, accepted=True):
        with self.lock:
            if quota:
                self.reject(provider, quota)
            elif accepted and probe and self.probe_revisions.get(provider) == self.revisions.get(provider, 0):
                # A successful probe cannot clear a later concurrent rejection.
                self.state.pop(provider, None)
                save(self.path, self.state)
            if probe:
                self.probes.discard(provider)
                self.probe_revisions.pop(provider, None)


def stream_events(destination, call):
    name = call.get('raw_stream_file')
    if not name or not (Path(destination) / name).exists():
        return []
    events = []
    for line in (Path(destination) / name).read_text(errors='replace').splitlines():
        try:
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
        except ValueError:
            continue
    return events


def response_started(call, events):
    """Only an explicit empty rejection qualifies for automatic request retry."""
    if call.get('ok') or call.get('num_tool_calls'):
        return True
    if str(call.get('text') or '').strip():
        return True
    usage = call.get('usage') or {}
    if isinstance(usage, dict):
        for key in ('output_tokens', 'completion_tokens'):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                return True
    for event in events:
        if event.get('type') == 'assistant':
            message = event.get('message', {})
            # Provider API-error notices are not generated assistant responses.
            if not event.get('error') and message.get('model') not in (None, '<synthetic>'):
                return True
        if event.get('type') in ('item.started', 'item.completed'):
            if event.get('item', {}).get('type') in ('agent_message', 'assistant_message', 'reasoning', 'command_execution', 'file_change'):
                return True
    return False


def delivery_problem(spec, estimates):
    """Test required finite output fields without exposing reference values."""
    missing = []
    for row in spec['reference']['estimates']:
        for field in ('estimate', 'se', 'n_obs'):
            if row.get(field) is None:
                continue
            value = estimates.get(row['term'], {}).get(field)
            try:
                finite = math.isfinite(float(value))
            except (ValueError, TypeError, OverflowError):
                finite = False
            if not finite:
                missing.append(f'{row["term"]}: missing finite {field}')
    return '\n'.join(missing)


def repair_text(initial, previous, execution, problem):
    feedback = {key: (value[-1500:] if key in ('stdout', 'stderr') and isinstance(value, str) else value)
                for key, value in execution.items()}
    return (initial + '\n\nThe previous script did not finish with all required output fields.\n'
            + 'Previous script:\n' + previous + '\nExecution feedback:\n'
            + json.dumps(feedback) + '\n' + problem + '\nReturn the complete corrected fenced script.\n')


def snapshot_state(destination, state):
    snapshots = list(Path(destination).glob('state_*.json'))
    save(Path(destination) / f'state_{len(snapshots)+1:06d}.json', state)
    save(Path(destination) / 'state.json', state)


def restore_workspace(destination, state, data_path, input_files=None):
    """Restore a missing temporary workspace from a recorded complete snapshot."""
    cwd = Path(state['cwd'])
    if state['stage'] not in ('evaluate', 'request'):
        raise ValueError('Only local evaluation or an unstarted request can restore a workspace')
    if cwd.is_symlink() or cwd.parent.resolve() != Path('/private/tmp') or not cwd.name.startswith('prospective_task_'):
        raise ValueError('Unexpected temporary workspace path')
    if cwd.exists():
        raise ValueError('Workspace restoration requires a missing directory')
    candidates = []
    for key in ('held_workspace', 'submitted_workspace'):
        if not state.get(key):
            continue
        source = Path(state[key])
        if source.is_symlink() or source.parent.resolve() != destination.resolve() or not source.name.startswith(('held_workspace_', 'submitted_round')):
            raise ValueError('Snapshot is outside the saved attempt')
        if source.is_dir():
            candidates.append(source)
    if not candidates:
        raise ValueError('No saved complete workspace snapshot')
    source = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    for path in source.rglob('*'):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError('Snapshot contains a symlink or nonregular file')
    inputs = task_input_files(dict(data_path=str(data_path), input_files=input_files or {}))
    if not inputs_intact(source, inputs):
        raise ValueError('Snapshot input differs from the pinned task data')
    staging = Path(tempfile.mkdtemp(prefix='prospective_restore_', dir='/private/tmp'))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True, symlinks=True)
        # Reject a link introduced during copying before exposing the workspace.
        if any(path.is_symlink() for path in staging.rglob('*')):
            raise ValueError('Snapshot changed during restoration')
        if not inputs_intact(staging, inputs):
            raise ValueError('Snapshot input changed during restoration')
        staging.rename(cwd)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    state['workspace_restorations'] = state.get('workspace_restorations', []) + [dict(source=str(source), cwd=str(cwd), restored_at=now())]
    snapshot_state(destination, state)
    return cwd


def completed(folder, protocol):
    files = list(Path(folder).glob('attempt_*/run_record.json'))
    done = []
    for path in files:
        record = json.loads(path.read_text())
        if record['protocol_sha256'] != protocol:
            raise ValueError('Mixed protocol records')
        if record['status'] in TERMINAL:
            done.append(record)
    if len(done) > 1:
        raise ValueError('Multiple completed responses for one planned slot')
    return done[0] if done else None


@contextmanager
def stata_slot(job):
    """Share the existing native Stata lock across agent and external execution."""
    if job['platform'] != 'stata':
        yield
        return
    with open('/private/tmp/agents_metrics_native_stata.lock', 'a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def run_job(job, manifest, run_root, runtime, quotas, model_call=None, execute_call=None):
    c = runtime.c
    model_call = model_call or c.call_model
    # Lock is shared with agent calls below, so avoid the inherited extra Stata lock.
    execute_call = execute_call or c.prior.execute
    folder = Path(run_root) / job_id(job)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = completed(folder, manifest['protocol_sha256'])
        if previous:
            return previous
        destination = folder / 'attempt_0001'
        state_path = destination / 'state.json'
        spec = manifest['protocol']['tasks'][job['task']]
        inputs = task_input_files(spec)
        initial = manifest['protocol']['prompts'][prompt_key(job)]
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['job'] != job or state['protocol_sha256'] != manifest['protocol_sha256']:
                raise ValueError('Restart identity mismatch')
            if state['stage'] in ('provider_in_flight', 'hold'):
                raise ValueError('Unresolved request or protocol hold requires review')
        else:
            if destination.exists():
                raise ValueError('Incomplete unclassified attempt directory')
            destination.mkdir()
            cwd = Path(tempfile.mkdtemp(prefix='prospective_task_', dir='/private/tmp'))
            copy_inputs(cwd, inputs)
            state = dict(job=job, protocol_sha256=manifest['protocol_sha256'], started_at=now(),
                status='in_progress', stage='request', cwd=str(cwd), prompt=initial, round=1,
                calls=[], executions=[], access_findings=[], attempt_directory=str(destination),
                scoring=dict(success=None, numeric_success=None, integrity_pass=None))
            snapshot_state(destination, state)
        cwd = Path(state['cwd'])
        if not cwd.exists():
            cwd = restore_workspace(destination, state, spec['data_path'], spec.get('input_files'))
        if not cwd.is_dir() or not inputs_intact(cwd, inputs):
            raise ValueError('Resumable workspace missing or input changed')
        try:
            while state['stage'] not in ('done', 'hold'):
                if STOP.is_set():
                    raise Paused('User requested pause')
                if state['stage'] == 'request':
                    probe = quotas.acquire(job['model'])
                    try:
                        if 'source_hashes' in manifest['protocol']:
                            verify(manifest)
                        number = len(state['calls']) + 1
                        state.update(stage='provider_in_flight', status='in_progress')
                        snapshot_state(destination, state)
                        if job['condition'] == 'agent':
                            with stata_slot(job):
                                call = model_call(state['prompt'], cwd, job, destination, number)
                        else:
                            with tempfile.TemporaryDirectory(prefix='prospective_generate_', dir='/private/tmp') as generated:
                                call = model_call(state['prompt'], Path(generated), job, destination, number)
                    except BaseException:
                        quotas.release(job['model'], probe, accepted=False)
                        raise
                    events = stream_events(destination, call)
                    quota = parse_quota(events, quotas.clock())
                    if not quota and call.get('provider_limit'):
                        quota = dict(status='rejected', reset_at=None, observed_at=quotas.clock(), evidence='CLI provider-limit signal without reset metadata')
                    call['quota_rejected_before_response'] = bool(quota and not response_started(call, events))
                    quotas.release(job['model'], probe, quota)
                    state['calls'].append(call)
                    save(destination / f'call_attempt{number}.json', call)
                    cancelled = isinstance(call.get('returncode'), int) and call['returncode'] < 0 and not call.get('timed_out')
                    if cancelled:
                        state.update(stage='hold', status='needs_review', hold_reason='interrupted_request')
                    elif quota and not response_started(call, events):
                        state.update(stage='request', status='quota_wait')
                        snapshot_state(destination, state)
                        continue
                    elif quota and not call.get('ok'):
                        state.update(stage='hold', status='needs_review', hold_reason='quota_after_model_work')
                    elif call.get('infrastructure_failure') and not response_started(call, events):
                        state.update(stage='hold', status='needs_review', hold_reason='infrastructure_without_response')
                    else:
                        state.update(stage='evaluate', status='in_progress')
                        if job['condition'] != 'agent':
                            code = c.harness.extract_code(call.get('text', ''))
                            script = cwd / c.harness.script_name(job['platform'])
                            script.unlink(missing_ok=True)
                            if code is not None:
                                script.write_text(code)
                    snapshot_state(destination, state)
                if state['stage'] == 'evaluate':
                    if spec.get('input_files') and 'source_hashes' in manifest['protocol']:
                        verify(manifest)
                    try:
                        script = prospective_outputs.select_submission_script(
                            cwd, job['platform'], c.harness.script_name(job['platform']))
                    except prospective_outputs.AmbiguousSubmissionError as error:
                        state.update(stage='hold', status='needs_review', hold_reason='ambiguous_submission',
                                     submission_candidates=error.candidates)
                        break
                    input_changed = not inputs_intact(cwd, inputs)
                    copies = len(list(destination.glob(f'submitted_round{state["round"]}_*')))
                    submitted = destination / f'submitted_round{state["round"]}_{copies+1:04d}'
                    c.prior.snapshot(cwd, submitted)
                    state['submitted_workspace'] = str(submitted)
                    if input_changed:
                        state.update(stage='hold', status='needs_review', hold_reason='integrity', input_modified=True)
                        break
                    execution, estimates = {}, {}
                    if script is not None:
                        for name in ('result.json', 'run.log', 'printed_output.txt'):
                            (cwd / name).unlink(missing_ok=True)
                        execution_number = state.get('execution_attempts_started', 0) + 1
                        state['execution_attempts_started'] = execution_number
                        snapshot_state(destination, state)
                        with stata_slot(job):
                            execution = execute_call(script, job['platform'], cwd, destination, execution_number)
                        state['executions'].append(execution)
                        c.prior.snapshot(cwd, destination / f'execution_workspace{execution_number:04d}')
                        if isinstance(execution.get('returncode'), int) and execution['returncode'] < 0 and not execution.get('timed_out'):
                            state.update(stage='evaluate', status='paused', hold_reason='local_execution_interrupted')
                            raise Paused('Local execution interrupted; preserved response needs local replay only')
                        estimates, rule, notes = c.load_output(cwd, spec, job, execution)
                        numeric, terms, numeric_notes = enforce_integer_terms(
                            estimates, spec, c.compare_numeric(estimates, spec))
                        # Passing no script limits this check to input-data identity.
                        # Source access and reference literals are reviewed from saved files later.
                        integrity, integrity_notes = c.sp.score_integrity(c.base.as_task(spec), cwd, None, {spec['task_id']: spec['reference']})
                        integrity = bool(integrity and inputs_intact(cwd, inputs))
                        state['scoring'] = dict(success=bool(numeric and integrity), numeric_success=numeric,
                            integrity_pass=integrity, estimates=estimates, per_term=terms,
                            integrity_scope='input_data_only; script_review_pending',
                            notes=notes + numeric_notes + integrity_notes, primary_source='fresh collector reexecution')
                        if not integrity:
                            state.update(stage='hold', status='needs_review', hold_reason='integrity')
                            break
                    problem = delivery_problem(spec, estimates) if script is not None else 'No fenced script was returned.'
                    if job['condition'] == 'self_repair' and state['round'] < 4 and not state['calls'][-1].get('timed_out') and (not execution.get('ok') or problem):
                        previous_code = script.read_text() if script is not None else state['calls'][-1].get('text', '')
                        state['prompt'] = repair_text(initial, previous_code, execution, problem)
                        state['round'] += 1
                        for item in cwd.iterdir():
                            if item.is_dir(): shutil.rmtree(item)
                            else: item.unlink()
                        copy_inputs(cwd, inputs)
                        state.update(stage='request', status='in_progress')
                        snapshot_state(destination, state)
                        continue
                    if script is None:
                        state['scoring'] = dict(success=False, numeric_success=None, integrity_pass=None, estimates={}, per_term={}, notes=[problem])
                    last = state['calls'][-1]
                    status = 'provider_timeout' if last.get('timed_out') else 'execution_timeout' if execution.get('timed_out') else 'completed' if script is not None else 'model_failure'
                    state.update(stage='done', status=status)
                    c.prior.snapshot(cwd, destination / 'final_workspace')
                    state['primary_output_path'] = str(destination / 'final_workspace' / ('result.json' if job['output_format'] == 'json' else 'printed_output.txt'))
                    snapshot_state(destination, state)
        except (Paused, QuotaUnknown) as error:
            state.update(status='quota_wait' if isinstance(error, QuotaUnknown) else 'paused', pause_reason=str(error))
            snapshot_state(destination, state)
            STOP.set()
        except BaseException as error:
            state.update(status='needs_review', stage='hold', driver_error=repr(error))
            snapshot_state(destination, state)
            STOP.set()
        if state['stage'] == 'hold':
            STOP.set()
        if state['stage'] != 'done' and cwd.exists():
            copies = len(list(destination.glob('held_workspace_*')))
            held = destination / f'held_workspace_{copies+1:04d}'
            c.prior.snapshot(cwd, held)
            state['held_workspace'] = str(held)
            snapshot_state(destination, state)
        substantive_calls = [ca for ca in state['calls'] if not ca.get('quota_rejected_before_response')]
        state.update(provider_ok=bool(substantive_calls) and all(ca.get('ok') for ca in substantive_calls),
                     all_provider_attempts_ok=bool(state['calls']) and all(ca.get('ok') for ca in state['calls']),
                     task_censored=any(ca.get('task_censored') for ca in substantive_calls),
                     protocol_violation=None, access_review_status='pending_post_collection', completed_at=now())
        if state['stage'] == 'done' and any(ca.get('timed_out') for ca in substantive_calls):
            state['scoring']['success'] = False
            state['scoring']['primary_failure_reason'] = 'provider_timeout'
        # This collector handles interruption state above. The older wrapper
        # stops on any historical interruption, including a completed replay.
        adapter = getattr(runtime, 'BASE_ADAPT', runtime.m.adapt_record)
        record = adapter(state, destination)
        record.pop('inherited_runtime_record', None)
        if state['stage'] != 'done':
            record['outcome'].update(primary_success=None, numerical_success=None, data_script_integrity=None,
                protocol_status='hold', measurement_status=state['status'], retry_eligible=False)
            record['scoring'].update(success=None, numeric_success=None, integrity_pass=None)
        else:
            record['scoring'] = state['scoring']
            record['outcome'].update(primary_success=state['scoring']['success'], numerical_success=state['scoring']['numeric_success'],
                data_script_integrity=None, input_data_integrity=state['scoring']['integrity_pass'],
                protocol_status='pending_post_collection', retry_eligible=False)
        record['status'] = state['status']
        record['central_runtime'] = dict(version=VERSION, source_sha256=sha(__file__), quota_calls_preserved=True)
        save(destination / 'run_record.json', record)
        return record


def status(manifest, run_root):
    rows = []
    for job in manifest['jobs']:
        folder = Path(run_root) / job_id(job)
        record = completed(folder, manifest['protocol_sha256'])
        path = folder / 'attempt_0001/state.json'
        if record is None and path.exists(): record = json.loads(path.read_text())
        rows.append(dict(id=job_id(job), status=record['status'] if record else 'pending',
                         primary=record.get('outcome', {}).get('primary_success') if record else None))
    return dict(planned=len(rows), completed=sum(r['status'] in TERMINAL for r in rows), rows=rows)


def collect(jobs, manifest, run_root, runtime, workers):
    quotas = QuotaCoordinator(Path(run_root).parent / 'provider_quota.json')
    pending = iter(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        while True:
            while len(active) < workers and not STOP.is_set():
                job = next(pending, None)
                if job is None: break
                if completed(Path(run_root) / job_id(job), manifest['protocol_sha256']): continue
                active[pool.submit(run_job, job, manifest, run_root, runtime, quotas)] = job
            if not active: break
            finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in finished:
                job = active.pop(future)
                try:
                    record = future.result()
                    print(json.dumps(dict(id=job_id(job), status=record['status'], primary=record['outcome']['primary_success'])), flush=True)
                except Exception as error:
                    STOP.set()
                    print(json.dumps(dict(id=job_id(job), infrastructure_error=repr(error))), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prep', 'check', 'status', 'collect'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--jobs-file', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    # Provider calls use a separate working directory. Output paths must be absolute.
    args.manifest = args.manifest.resolve()
    if args.action == 'prep':
        if not args.config: parser.error('prep requires --config')
        manifest = prepare(args.config, args.manifest)
        print(json.dumps(dict(protocol_sha256=manifest['protocol_sha256'], slots=len(manifest['jobs']))))
        return
    manifest = json.loads(args.manifest.read_text())
    run_root = args.manifest.parent / 'runs' / manifest['protocol_sha256']
    if args.action == 'status':
        print(json.dumps(status(manifest, run_root), indent=2)); return
    runtime = load_runtime(args.manifest.parent)
    os.environ['PATH'] = runtime.c.environment()['PATH']
    verify(manifest, runtime)
    if args.action == 'check':
        print('Local manifest, command, source, and environment checks passed. No provider calls.'); return
    if not args.authorization: parser.error('collect requires --authorization')
    authorization = json.loads(args.authorization.read_text())
    if authorization.get('protocol_sha256') != manifest['protocol_sha256'] or authorization.get('allow_provider_calls') is not True:
        raise ValueError('Authorization does not match the exact planned manifest')
    jobs = json.loads(args.jobs_file.read_text()) if args.jobs_file else manifest['jobs']
    if len({job_id(j) for j in jobs}) != len(jobs) or any(j not in manifest['jobs'] for j in jobs):
        raise ValueError('Requested jobs must be an exact unique subset')
    if not 1 <= args.workers <= 8: parser.error('workers is a chosen limit between1 and8')
    def pause(signum, frame):
        STOP.set()
        runtime.c.prior.stop_children(signum, frame)
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, pause)
    with (args.manifest.parent / 'collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collect(jobs, manifest, run_root, runtime, args.workers)
    print(json.dumps(status(manifest, run_root)))
    if any(completed(run_root / job_id(job), manifest['protocol_sha256']) is None for job in jobs):
        raise SystemExit(2)


if __name__ == '__main__':
    main()

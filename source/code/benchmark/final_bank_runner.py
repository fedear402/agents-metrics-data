"""Collection runner for the final task bank: plan, isolation, provider call, execution, and scoring.

The final bank has 48 specified packages (38 in data/task_candidates/2026-09-28_final_v2, 10 in
2026-09-28_final_hard) and 7 decision versions (data/task_candidates/2026-09-29_decision; decision record
2026-09-29_decision-versions). One collection cell is task x model x language x prompt x condition x repetition:
48 x 2 x 3 x 2 x 3 = 1,728 specified cells (prompts: zero_shot and few_shot) plus 7 x 2 x 3 x 1 x 3 = 126 decision
cells (zero_shot), 1,854 cells per repetition. The level heldout_few_shot left the bank on 2026-09-30 (decision record
2026-09-30_drop-heldout-level), and arima_forecast__decision and ttest_welch__decision were dropped (decision record
2026-09-30_hand-audit-fixes).

Harness configuration: quality_reports/decision_records/2026-09-29_harness-harmonization.md, which applies
quality_reports/audits/2026-09-29_bank-final/out/harmonize.md (Sections 4 to 7).

Parts:
- plan_cells and resolve_cell: the cell plan and the check that each cell has its prompt file, data files,
  reference, and scoring specification.
- prepare_collection: a private run root (mode 700) outside every Git worktree, the pinned binaries
  (pin_binaries, check_binaries), one PATH folder per language, and the Claude settings file.
- prepare_run: an opaque workspace runs/c<NNNNN>/ per cell, the record folder records/<cell id>/, and the
  Seatbelt profile of the cell language.
- Credentials. Sonnet uses the normal Claude login of the computer: the runner sets no CLAUDE_CONFIG_DIR and
  no token variable, and it reads no Keychain item. Luna uses one CODEX_HOME per worker that holds a symlink
  to ~/.codex/auth.json (setup_codex_home, teardown_codex_home). The runner never opens or copies that file.
- clean_env: the child environment, built from nothing, with TZ=UTC and the PATH of the cell language only
  (Python runs see python and python3; R runs see R and Rscript; Stata runs see stata and stata-se).
- Interpreter denial (decision record 2026-09-30_interpreter-exec-denial): the Seatbelt profile of a cell (script
  executions and Sonnet tool commands) denies execution of the other two languages' interpreters by any path
  (install trees, files, and command names; interpreter_rules), and the Codex permission profile denies their
  install trees and files. The own interpreter and the PATH stubs keep working.
- call_provider: one provider call. The stall rule (300 s with no completed JSON event and no tool process)
  applies to agent calls only. A generation call runs to its hard limit. The call writes a receive time for
  each stream line.
- Per-call checks: the Sonnet system/init event and modelUsage (claude_init_problems, model_problems), the
  Luna stream and rollout (codex_stream_problems, collect_rollout). A failed check stops the block.
- A usage-limit rejection is a provider stop. The run repeats the call after the usage watcher lets a new
  request start. The watcher pauses until the reset time of the rejection plus the resume margin, or retries
  one call every 15 minutes when the rejection gives no reset time.
- execute: every script execution runs under the Seatbelt execution profile (no network, writes only in the
  workspace) with the clean environment. Stata licence lines are removed from run.log, stdout, and stderr
  before any copy, repair prompt, or scoring.
- environment_check and check_binaries at each block start and end.

Dry mode (the default) makes no provider call. The runner builds and records the real claude or codex
command, then runs a local fake command in its place. It refuses a provider binary, and the usage watcher
must use a fake Codex reading. No credential is read.

Command line (repository root):
    uv run python code/benchmark/final_bank_runner.py plan [--out FILE]
    uv run python code/benchmark/final_bank_runner.py pin --bin-dir DIR
    uv run python code/benchmark/final_bank_runner.py check-bin --bin-dir DIR
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import final_bank_prompts as fp  # noqa: E402
import final_bank_scorer as fs  # noqa: E402
import usage_watch as uw  # noqa: E402
from stata_log_privacy import strip_license_details  # noqa: E402

VERSION = 'final_bank_runner_2026_09_30_v1'  # interpreter denial, Sonnet 5.5, two prompt levels
LANGS, PROMPTS, CONDITIONS = fp.LANGS, fp.PROMPTS, fp.CONDITIONS
# Benchmark models (decision record 2026-09-30_sonnet-5-5). The key names the model version. Any other model id in a
# call stops the block with ModelMismatch (_call_checks).
MODELS = {
    'sonnet55': dict(cli='claude', model_id='claude-sonnet-5-5', effort='medium'),
    'luna': dict(cli='codex', model_id='gpt-5.6-luna', effort='medium'),
}
CLAUDE_MODEL = 'sonnet55'   # the MODELS key of the Claude model
SCRIPT_NAME = {'python': 'run.py', 'r': 'run.R', 'stata': 'run.do'}

# Chosen limits. They are settings, not measured provider values.
STALL_IDLE_SECONDS = 300      # agent calls only: no completed stdout event for this long, and no tool process
STALL_PROBE_SECONDS = 10      # second reading of a stalled call, to separate runaway from hang
CALL_LIMIT_SECONDS = {'agent': 1500, 'generation': 900}   # hard limit per call, as in the pilots
EXEC_LIMIT_SECONDS = 600      # one script execution by the runner
MONITOR_SECONDS = 5.0
SELF_REPAIR_ROUNDS = 4
FEEDBACK_TAIL = 1500          # characters of stdout, stderr, and the Stata log in the repair feedback
MAX_LIMIT_REJECTIONS = 400    # usage-limit rejections of one run before the run stops (about 100 h at 15 min)
FINAL_FILE_MARGIN_SECONDS = 2  # output-channel rule of 2026-09-24
FINAL_FILE_MAX_BYTES = 5 * 1024 * 1024

# Runtime paths.
HOME = Path.home()
VENV = ROOT / '.venv'
UV_PYTHON = Path(os.path.realpath(VENV / 'bin/python')).parents[1]
R_USER_LIB = HOME / 'Library/R/arm64/4.6/library'
STATA_PLUS = HOME / 'Library/Application Support/Stata/ado/plus'
STATA_PERSONAL = HOME / 'Documents/Stata/ado/personal'
RSCRIPT = '/usr/local/bin/Rscript'
R_BIN = '/usr/local/bin/R'
STATA_BIN = '/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se'
RUN_ROOT_BASE = HOME / '.am-bench'

# Provider binaries. A collection runs pinned copies (pin_binaries). The source paths serve dry mode only.
CLAUDE_VERSION = '2.1.284'
CODEX_VERSION = '0.158.0'
CLAUDE_SOURCE = HOME / '.local/share/claude/versions' / CLAUDE_VERSION
CODEX_VENDOR = Path('/opt/homebrew/lib/node_modules/@openai/codex/node_modules/@openai/codex-darwin-arm64/'
                    'vendor/aarch64-apple-darwin')
SOURCE_SHA256 = {'claude': '50a14c2f50f56668380fdda490167f1d3630d5cc18fb8aed3073c2c7ea7314fe',
                 'codex': '788a818fbb9596869c7a487554507cb8bdca17584b8671112b23f9e225ba35c8'}
CLAUDE_PIN_NAME = f'claude-{CLAUDE_VERSION}'
CODEX_PIN_NAME = f'codex-{CODEX_VERSION}'
CLAUDE_BIN = CLAUDE_SOURCE
CODEX_BIN = CODEX_VENDOR / 'bin/codex'
CODEX_AUTH = HOME / '.codex/auth.json'   # linked, never opened or copied

LANGUAGE_READ = {'python': [VENV, UV_PYTHON], 'r': [R_USER_LIB], 'stata': [STATA_PLUS, STATA_PERSONAL]}
SHARED_READ = [R_USER_LIB, STATA_PLUS, STATA_PERSONAL]
SYSTEM_READ = ['/usr/local', '/Library/Frameworks/R.framework', '/Applications/StataNow', '/opt/homebrew']
PRIVATE_ROOTS = [HOME, Path('/private/tmp'), Path('/private/var/folders'), Path('/Users/Shared')]
SYSTEM_PATH = ['/usr/bin', '/bin', '/usr/sbin', '/sbin']
PATH_VALUE = os.pathsep.join([str(VENV / 'bin'), '/usr/local/bin', *SYSTEM_PATH])  # matplotlib seed only
# Interpreter names on the PATH of each language. The other names are stubs that answer "command not found".
LANGUAGE_TOOLS = {'python': {'python': str(VENV / 'bin/python'), 'python3': str(VENV / 'bin/python')},
                  'r': {'R': R_BIN, 'Rscript': RSCRIPT},
                  'stata': {'stata': STATA_BIN, 'stata-se': STATA_BIN}}
BLOCKED_TOOLS = ('python', 'python3', 'pip', 'pip3', 'uv', 'R', 'Rscript', 'stata', 'stata-se', 'stata-mp')
# Interpreters of each language (decision record 2026-09-30_interpreter-exec-denial). The PATH stubs stop a bare
# name only. The hand audit of 2026-09-30 found two Luna Stata agent runs that started /opt/homebrew/bin/python3 and
# /Library/Frameworks/R.framework/Resources/bin/Rscript by absolute path. A run of one language therefore may not
# execute the interpreters of the other two languages by any path:
#   - Seatbelt (script executions and Sonnet tool commands): process-exec* is denied on the install trees, the
#     files, and the command names below, and file-map-executable is denied on the trees, so a copied binary
#     cannot load its libraries;
#   - Codex (Luna agent tool commands): the permission profile denies the trees and files that exist.
HOMEBREW = Path('/opt/homebrew')
CLT = Path('/Library/Developer/CommandLineTools')
INTERPRETER_TREES = {
    'python': [VENV, UV_PYTHON.parent, *sorted(HOMEBREW.glob('Cellar/python@*')),
               *sorted(HOMEBREW.glob('Cellar/python-tk@*')), HOMEBREW / 'Frameworks/Python.framework',
               HOMEBREW / 'Cellar/uv', HOMEBREW / 'Cellar/pipx', Path('/Library/Frameworks/Python.framework'),
               CLT / 'Library/Frameworks/Python3.framework'],
    'r': [Path('/Library/Frameworks/R.framework')],
    'stata': [Path('/Applications/StataNow'), Path('/Applications/Stata')],
}
INTERPRETER_FILES = {
    'python': [Path('/usr/bin/python3'), Path('/usr/bin/pip3'), *sorted((CLT / 'usr/bin').glob('python3*')),
               *sorted((CLT / 'usr/bin').glob('pip3*'))],
    'r': [Path(R_BIN), Path(RSCRIPT)],
    'stata': [],
}
# Command names (the last path component) that no run of another language may execute, wherever the file is.
INTERPRETER_NAMES = {
    'python': [r'/(python|pip|ipython|jupyter)[^/]*$', r'/(uv|uvx)$'],
    'r': [r'/(R|Rscript)$'],
    'stata': [r'/(stata|stata-se|stata-mp|stata-be|xstata|xstata-se|xstata-mp|StataSE|StataMP|StataBE)$'],
}
PROVIDER_NAMES = {'claude', 'codex', CLAUDE_SOURCE.name, CLAUDE_PIN_NAME}

# Variables the runner never sets (harmonize.md 6.1).
FORBIDDEN_ENV = ('CLAUDE_CONFIG_DIR', 'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL',
                 'CLAUDE_CODE_OAUTH_TOKEN', 'CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR', 'CODEX_API_KEY',
                 'OPENAI_API_KEY', 'CODEX_ACCESS_TOKEN', 'CODEX_REFRESH_TOKEN_URL_OVERRIDE', 'HTTPS_PROXY')
CLAUDE_ENV = {
    'DISABLE_AUTOUPDATER': '1', 'DISABLE_UPDATES': '1', 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
    'DISABLE_TELEMETRY': '1', 'DISABLE_ERROR_REPORTING': '1', 'CLAUDE_CODE_SUBPROCESS_ENV_SCRUB': '1',
    'CLAUDE_CODE_DISABLE_BACKGROUND_TASKS': '1', 'CLAUDE_CODE_SIMPLE_SYSTEM_PROMPT': '0',
    'CLAUDE_CODE_SKIP_PROMPT_HISTORY': '1', 'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1',
    'CLAUDE_CODE_DISABLE_FAST_MODE': '1', 'BASH_DEFAULT_TIMEOUT_MS': '600000', 'BASH_MAX_TIMEOUT_MS': '600000',
}
CLAUDE_SETTINGS = {
    'permissions': {'deny': ['WebFetch', 'WebSearch']},
    'includeGitInstructions': False,
    'attribution': False,
    'totalTokensReminder': 'off',
    'autoMemoryEnabled': False,
}

ENV_DIR = HERE / 'collection_environment'
ENV_MANIFEST = ENV_DIR / 'environment_manifest.json'
SECRET_KEYS = re.compile(r'TOKEN|KEY|SECRET|PASSWORD|AUTH', re.I)
STATA_ERROR = re.compile(r'^r\(\d+\);', re.M)
CODE_FENCE = re.compile(r'```(?:python|r|R|stata)\s*\n(.*?)```', re.S)
EXPENSIVE_MODEL = re.compile(r'opus|fable', re.I)
SIDE_MODEL = re.compile(r'^claude-haiku', re.I)


class DryRunViolation(RuntimeError):
    """A dry run tried to start a provider binary or a live usage reading."""


class EnvironmentChanged(RuntimeError):
    """The package environment differs from environment_manifest.json. The block stops."""


class BinaryChanged(RuntimeError):
    """A pinned provider binary differs from binaries.json. The block stops."""


class ConfigurationCheckFailed(RuntimeError):
    """A per-call configuration check failed (init event, model, stream, rollout, credential link)."""


class ModelMismatch(ConfigurationCheckFailed):
    """A call named a model other than the collection model (for example an Opus or Fable model)."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def save_json(path: Path, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(fs.jsonable(value), indent=2, default=str) + '\n')


# Plan -----------------------------------------------------------------------------------------

def cell_id(cell: dict) -> str:
    return (f"{cell['task']}__{cell['model']}__{cell['lang']}__{cell['prompt']}__{cell['condition']}"
            f"__rep{cell['rep']:02d}")


def plan_cells(reps=(1,), tasks=None) -> list[dict]:
    """Every cell of the bank for the given repetitions, in a fixed order: the 48 specified tasks, then the decision
    versions, each with the prompt levels of its task.json (decision record 2026-09-29_decision-versions)."""
    folders = fp.all_package_dirs()
    if tasks:
        folders = [f for f in folders if f.name in set(tasks)]
    levels = {f.name: fp.prompt_levels(fp.load_task(f)) for f in folders}
    return [dict(task=f.name, bank=f.parent.name, model=m, lang=lang, prompt=p, condition=c, rep=r)
            for r in reps for f in folders for m in MODELS for lang in LANGS for p in levels[f.name]
            for c in CONDITIONS]


def reference_values(folder: Path) -> dict:
    """Reference rows {term: {estimate, se, n_obs}} of a package (route A of reference.json)."""
    reference = json.loads((folder / 'reference.json').read_text())
    return reference.get('route_a_values') or reference.get('reference') or {}


_RENDERED: dict[str, dict] = {}


def resolve_cell(cell: dict) -> dict:
    """Check that a cell resolves to its prompt, data, reference, and scoring specification."""
    folder = fp.package_dir(cell['task'])
    problems = []
    prompt = folder / 'prompts' / fp.prompt_name(cell['lang'], cell['prompt'], cell['condition'])
    if cell['task'] not in _RENDERED:
        _RENDERED[cell['task']] = fp.render_package(folder)
    if not prompt.is_file():
        problems.append(f'prompt missing: {prompt.name}')
    elif prompt.read_text() != _RENDERED[cell['task']][prompt.name]:
        problems.append(f'prompt differs from a fresh rendering: {prompt.name}')
    elif fp.prompt_problems(prompt.read_text()):
        problems += fp.prompt_problems(prompt.read_text())
    task = fp.load_task(folder)
    data = [folder / 'data' / name for name in task['files']]
    problems += [f'data file missing: {p.name}' for p in data if not p.is_file()]
    if not (folder / 'reference.json').is_file():
        problems.append('reference.json missing')
    try:
        spec = fs.load_spec(folder)
        problems += [f'scoring: {p}' for p in fs.validate_spec(spec)]
        rows = {row['term'] for row in task['output_rows']}
        problems += [f'scored term {c["term"]} is not an output row' for c in spec['checks'] if c['term'] not in rows]
        if not fs.score_rows(reference_values(folder), spec)['passed']:
            problems.append('the reference values do not pass the scoring specification')
    except (OSError, ValueError, KeyError) as error:
        problems.append(f'scoring specification: {error}')
    return dict(cell=cell_id(cell), ok=not problems, problems=problems, prompt=str(prompt.relative_to(ROOT)),
                prompt_sha256=sha(prompt) if prompt.is_file() else None,
                data={p.name: sha(p) for p in data if p.is_file()},
                scoring_sha256=sha(folder / 'scoring.json') if (folder / 'scoring.json').is_file() else None)


# Binary pins ----------------------------------------------------------------------------------

def _tree_hashes(folder: Path) -> dict:
    """sha256 of every regular file under folder, keyed by relative path. Links are recorded as links."""
    folder = Path(folder)
    out = {}
    for path in sorted(folder.rglob('*')):
        rel = str(path.relative_to(folder))
        if path.is_symlink():
            out[rel] = 'link:' + os.readlink(path)
        elif path.is_file():
            out[rel] = sha(path)
    return out


def _version(binary: Path) -> str:
    out = subprocess.run([str(binary), '--version'], capture_output=True, text=True, timeout=120,
                         env={'HOME': str(HOME), 'PATH': os.pathsep.join(SYSTEM_PATH), 'DISABLE_AUTOUPDATER': '1'})
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr).strip() else ''


def pin_binaries(bin_dir: Path, claude_source: Path | None = None, codex_vendor: Path | None = None,
                 expected: dict | None = None, versions: bool = True) -> dict:
    """Copy claude and the Codex vendor folder into bin_dir, hash every file, and write binaries.json.

    The source hashes must equal SOURCE_SHA256 (or expected). A mismatch refuses the copy.
    """
    bin_dir = Path(bin_dir)
    claude_source = Path(claude_source or CLAUDE_SOURCE)
    codex_vendor = Path(codex_vendor or CODEX_VENDOR)
    expected = expected or SOURCE_SHA256
    found = {'claude': sha(claude_source), 'codex': sha(codex_vendor / 'bin/codex')}
    for name, value in found.items():
        if value != expected[name]:
            raise BinaryChanged(f'source {name} sha256 {value} differs from the expected {expected[name]}')
    bin_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(bin_dir, 0o700)
    claude_copy, codex_copy = bin_dir / CLAUDE_PIN_NAME, bin_dir / CODEX_PIN_NAME
    if not claude_copy.exists():
        shutil.copy2(claude_source, claude_copy)
    if not codex_copy.exists():
        shutil.copytree(codex_vendor, codex_copy, symlinks=True)
    manifest = dict(created_at=now(), runner=VERSION,
                    claude=dict(source=str(claude_source), path=claude_copy.name, version=CLAUDE_VERSION,
                                sha256=sha(claude_copy)),
                    codex=dict(source=str(codex_vendor), path=codex_copy.name, version=CODEX_VERSION,
                               files=_tree_hashes(codex_copy)))
    if versions:
        manifest['claude']['version_output'] = _version(claude_copy)
        manifest['codex']['version_output'] = _version(codex_copy / 'bin/codex')
    if manifest['claude']['sha256'] != expected['claude'] or manifest['codex']['files'].get('bin/codex') != expected['codex']:
        raise BinaryChanged('the pinned copy differs from its source')
    save_json(bin_dir / 'binaries.json', manifest)
    return manifest


def check_binaries(bin_dir: Path, point: str = '') -> dict:
    """Recompute every pinned hash. Raise BinaryChanged on any difference."""
    bin_dir = Path(bin_dir)
    manifest = json.loads((bin_dir / 'binaries.json').read_text())
    problems = []
    claude_now = sha(bin_dir / manifest['claude']['path']) if (bin_dir / manifest['claude']['path']).is_file() else None
    if claude_now != manifest['claude']['sha256']:
        problems.append(f'claude sha256 {claude_now} differs from {manifest["claude"]["sha256"]}')
    codex_now = _tree_hashes(bin_dir / manifest['codex']['path'])
    for rel in sorted(set(codex_now) | set(manifest['codex']['files'])):
        if codex_now.get(rel) != manifest['codex']['files'].get(rel):
            problems.append(f'codex file {rel} changed')
    row = dict(point=point, at=now(), matches=not problems, problems=problems[:20],
               claude_sha256=claude_now, codex_bin_sha256=codex_now.get('bin/codex'),
               binaries_json_sha256=sha(bin_dir / 'binaries.json'))
    if problems:
        raise BinaryChanged(f'{point}: ' + '; '.join(problems[:5]))
    return row


def collection_config(run_root: Path) -> dict:
    path = Path(run_root) / 'tools/collection.json'
    return json.loads(path.read_text()) if path.is_file() else {}


def provider_bins(run_root: Path | None) -> tuple[Path, Path, Path | None]:
    """The claude binary, the codex binary, and the pinned Codex folder of a collection."""
    bin_dir = collection_config(run_root).get('bin_dir') if run_root else None
    if bin_dir:
        return (Path(bin_dir) / CLAUDE_PIN_NAME, Path(bin_dir) / CODEX_PIN_NAME / 'bin/codex',
                Path(bin_dir) / CODEX_PIN_NAME)
    return CLAUDE_BIN, CODEX_BIN, None


# Isolation ------------------------------------------------------------------------------------

def path_dir(run_root: Path, lang: str) -> Path:
    return Path(run_root) / 'tools/path' / lang


def write_language_paths(run_root: Path) -> dict:
    """One PATH folder per language: wrappers for its interpreter, stubs for the other interpreters."""
    out = {}
    for lang, tools in LANGUAGE_TOOLS.items():
        folder = path_dir(run_root, lang)
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        for name in BLOCKED_TOOLS:
            target = tools.get(name)
            body = (f'#!/bin/sh\nexec {_shell_quote(target)} "$@"\n' if target else
                    f'#!/bin/sh\necho "zsh: command not found: {name}" >&2\nexit 127\n')
            (folder / name).write_text(body)
            os.chmod(folder / name, 0o755)
        out[lang] = sorted(tools)
    return out


def language_path(run_root: Path | None, lang: str | None) -> str:
    if run_root is None or lang is None:
        return os.pathsep.join(SYSTEM_PATH)
    return os.pathsep.join([str(path_dir(run_root, lang)), *SYSTEM_PATH])


def clean_env(run_tmp: Path, extra: dict | None = None, path: str | None = None) -> dict:
    """The child environment, built from nothing. No host PATH and no agent variables."""
    tmp = str(run_tmp)
    env = {
        'HOME': str(HOME), 'USER': HOME.name, 'LOGNAME': HOME.name, 'SHELL': '/bin/zsh',
        'PATH': path or os.pathsep.join(SYSTEM_PATH),
        'LANG': 'en_US.UTF-8', 'LC_ALL': 'en_US.UTF-8', 'TZ': 'UTC', 'TMPDIR': tmp + '/', 'TERM': 'dumb',
        'NO_COLOR': '1', 'TMPPREFIX': tmp + '/zsh', 'ZDOTDIR': tmp,
        # Environment findings of 2026-09-28 (collection_environment/README.md).
        'R_LIBS_USER': str(R_USER_LIB), 'MPLCONFIGDIR': tmp + '/mpl',
        'R_USER_CONFIG_DIR': tmp + '/r_config', 'R_USER_DATA_DIR': tmp + '/r_data', 'R_USER_CACHE_DIR': tmp + '/r_cache',
    }
    env.update(extra or {})
    bad = [k for k in FORBIDDEN_ENV if k in env]
    if bad:
        raise ConfigurationCheckFailed(f'forbidden variables in the child environment: {bad}')
    return env


def masked_env(env: dict) -> dict:
    return {k: ('<masked>' if SECRET_KEYS.search(k) else v) for k, v in env.items()}


def sb_quote(path) -> str:
    return '"' + str(path).replace('\\', '\\\\').replace('"', '\\"') + '"'


def runtime_read(lang: str | None) -> list[Path]:
    """Private-root folders that tools of this language may read. Only Python runs see the Python venv."""
    if lang is None:
        return [VENV, UV_PYTHON, *SHARED_READ]
    extra = [p for p in LANGUAGE_READ[lang] if p not in SHARED_READ]
    return [*extra, *SHARED_READ]


def other_interpreters(lang: str) -> dict:
    """Trees, files, and name patterns of the interpreters of the two languages other than lang."""
    others = [other for other in LANGS if other != lang]
    return dict(trees=[p for o in others for p in INTERPRETER_TREES[o]],
                files=[p for o in others for p in INTERPRETER_FILES[o]],
                names=[n for o in others for n in INTERPRETER_NAMES[o]])


def interpreter_rules(lang: str | None, exec_allow=()) -> list[str]:
    """Seatbelt rules that stop a run of lang from executing another language's interpreter by any path.

    exec_allow: folders whose files stay executable (the PATH folder of the language, whose stubs for the other
    interpreters only print "command not found").
    """
    if lang is None:
        return []
    other = other_interpreters(lang)
    lines = ['; Interpreters of the other languages: no execution by any path, no loading of their libraries.']
    lines += [f'(deny process-exec* (subpath {sb_quote(p)}))' for p in other['trees']]
    lines += [f'(deny file-map-executable (subpath {sb_quote(p)}))' for p in other['trees']]
    lines += [f'(deny process-exec* (literal {sb_quote(p)}))' for p in other['files']]
    lines += [f'(deny process-exec* (regex #"{name}"))' for name in other['names']]
    lines += [f'(allow process-exec* (subpath {sb_quote(p)}))' for p in exec_allow]
    return lines


def seatbelt_profile(workspace: Path, run_tmp: Path, extra_read=(), extra_write=(), network=False,
                     lang: str | None = None, exec_allow=()) -> str:
    """Seatbelt profile. Later rules win, so the allows follow the denies."""
    lines = ['(version 1)', '(allow default)', '; Private roots: user home, shared temporary folders, other runs.']
    lines += [f'(deny file-read* file-write* (subpath {sb_quote(root)}))' for root in PRIVATE_ROOTS]
    if lang in ('r', 'stata'):
        lines += [f'(deny file-read* (subpath {sb_quote(p)}))' for p in (VENV, UV_PYTHON)]
    lines += ['; Writes: only the run workspace, the run TMPDIR, and devices.', '(deny file-write* (subpath "/"))',
              '(allow file-write* (literal "/dev/null") (literal "/dev/zero") (literal "/dev/tty")'
              ' (regex #"^/dev/fd/") (regex #"^/dev/ttys"))', '; Runtime reads inside the private roots.']
    reads = [*runtime_read(lang), *extra_read]
    parents = set()
    for path in [*reads, workspace, run_tmp, *extra_write]:
        parents.update(Path(path).parents)
    lines += [f'(allow file-read-metadata (literal {sb_quote(p)}))' for p in sorted(parents, key=str)]
    lines += [f'(allow file-read* (subpath {sb_quote(p)}))' for p in reads]
    lines += [f'(allow file-read* file-write* (subpath {sb_quote(p)}))' for p in [workspace, run_tmp, *extra_write]]
    if not network:
        lines += ['; Network: none, except local sockets that system libraries use.', '(deny network-outbound (remote ip))']
    lines += interpreter_rules(lang, exec_allow)
    return '\n'.join(lines) + '\n'


def execution_profile(workspace: Path, run_tmp: Path, lang: str | None = None, extra_read=(),
                      exec_allow=()) -> str:
    """Script executions and Sonnet tool commands: no network, writes only in the workspace, and no execution of
    the other languages' interpreters."""
    return seatbelt_profile(workspace, run_tmp, extra_read=extra_read, network=False, lang=lang,
                            exec_allow=exec_allow)


CODEX_FEATURES_OFF = ('skill_search', 'skill_mcp_dependency_install', 'plugins', 'remote_plugin',
                      'recommended_plugins', 'hooks', 'apps', 'memories', 'multi_agent', 'multi_agent_v2',
                      'browser_use', 'computer_use', 'image_generation', 'code_mode', 'goals', 'view_image',
                      'sleep_tool', 'shell_snapshot', 'daemon_auto_start', 'fast_mode',
                      'unbounded_connection_retries')


def _toml(value) -> str:
    if isinstance(value, dict):
        return '{' + ', '.join(f'{json.dumps(k)}={_toml(v)}' for k, v in value.items()) + '}'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return json.dumps(value)


def codex_argv(agent: bool, cwd: Path, last: Path, lang: str | None = None, run_root: Path | None = None) -> list[str]:
    """codex exec with the native permission profile (harmonize.md 6.2)."""
    _, codex_bin, codex_dir = provider_bins(run_root)
    entries = {':minimal': 'read', ':workspace_roots': {'.': 'write' if agent else 'read'}}
    entries.update({str(root): 'deny' for root in PRIVATE_ROOTS})
    denied = []
    if lang is not None:
        other = other_interpreters(lang)
        denied = [Path(p) for p in [*other['trees'], *other['files']] if Path(p).exists()]
    reads = [*runtime_read(lang), *[p for p in SYSTEM_READ if Path(p) not in denied]]
    if run_root is not None and lang is not None:
        reads.append(path_dir(run_root, lang))
    if codex_dir is not None:
        reads.append(codex_dir)
    entries.update({str(p): 'read' for p in reads})
    # The other languages' interpreters (decision record 2026-09-30_interpreter-exec-denial): a more specific entry
    # wins over a read root such as /opt/homebrew or /usr/local.
    entries.update({str(p): 'deny' for p in denied})
    profile = {'filesystem': entries, 'network': {'enabled': False}}
    settings = [f'model_reasoning_effort={json.dumps(MODELS["luna"]["effort"])}', 'web_search="disabled"',
                'approval_policy="never"', 'project_doc_max_bytes=0', 'mcp_servers={}',
                'skills.include_instructions=false', 'skills.bundled.enabled=false',
                'include_permissions_instructions=false', 'default_permissions="bench"',
                'permissions={bench=' + _toml(profile) + '}', 'features.shell_tool=' + _toml(agent)]
    settings += [f'features.{name}=false' for name in CODEX_FEATURES_OFF]
    settings += ['tools.experimental_request_user_input={enabled=false}', 'allow_login_shell=false',
                 'analytics.enabled=false', 'feedback.enabled=false', 'otel.exporter="none"',
                 'otel.metrics_exporter="none"', 'otel.trace_exporter="none"', 'check_for_update_on_startup=false',
                 'forced_login_method="chatgpt"', 'history.persistence="none"']
    argv = [str(codex_bin), 'exec', '--json', '--ignore-user-config', '--ignore-rules', '--skip-git-repo-check',
            '-m', MODELS['luna']['model_id'], '-C', str(cwd), '-o', str(last)]
    for setting in settings:
        argv += ['-c', setting]
    return argv


def claude_argv(agent: bool, settings_file: Path, run_root: Path | None = None) -> list[str]:
    """claude -p in safe mode with the collection settings file (harmonize.md 6.3)."""
    claude_bin, _, _ = provider_bins(run_root)
    argv = [str(claude_bin), '-p', '--model', MODELS[CLAUDE_MODEL]['model_id'], '--effort',
            MODELS[CLAUDE_MODEL]['effort'],
            '--output-format', 'stream-json', '--verbose', '--no-session-persistence', '--safe-mode',
            '--setting-sources', '', '--settings', str(settings_file), '--permission-mode', 'manual',
            '--permission-prompts', 'none', '--tools', 'Bash' if agent else '', '--strict-mcp-config',
            '--mcp-config', '{"mcpServers":{}}', '--disable-slash-commands']
    if agent:
        argv += ['--allowedTools', 'Bash']
    return argv


TOOL_WRAPPER = """#!/bin/sh
# CLAUDE_CODE_SHELL_PREFIX of one run: Claude Code calls this script with the whole Bash command
# string as the last argument. The command runs under the execution profile (no network,
# read-only libraries, writes only in the workspace) with the C.UTF-8 locale that Codex tools use.
last=""
for arg in "$@"; do last="$arg"; done
LANG=C.UTF-8; LC_ALL=C.UTF-8; export LANG LC_ALL
exec /usr/bin/sandbox-exec -f {profile} /bin/zsh -c "$last"
"""


def _shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def seed_matplotlib(run_root: Path) -> list[str]:
    """Build the matplotlib font cache once per collection. prepare_run copies it into each run's MPLCONFIGDIR."""
    seed = Path(run_root) / 'tools/mpl_seed'
    seed.mkdir(parents=True, exist_ok=True)
    if not list(seed.glob('fontlist*.json')):
        env = {'HOME': str(HOME), 'PATH': PATH_VALUE, 'MPLCONFIGDIR': str(seed), 'LANG': 'en_US.UTF-8'}
        subprocess.run([str(VENV / 'bin/python'), '-c', 'import matplotlib.font_manager'], env=env,
                       capture_output=True, text=True, timeout=600, check=False)
    return sorted(p.name for p in seed.iterdir())


def inside_git(path: Path) -> bool:
    out = subprocess.run(['git', '-C', str(path), 'rev-parse', '--is-inside-work-tree'], capture_output=True,
                         text=True, check=False, env={'PATH': os.pathsep.join(SYSTEM_PATH), 'HOME': str(HOME)})
    return out.returncode == 0 and out.stdout.strip() == 'true'


def prepare_collection(run_root: Path, seed_mpl: bool = True, bin_dir: Path | None = None) -> dict:
    """Private run root outside Git with tools, language PATH folders, and the pinned binary location.

    No credential is copied here. Stale Codex homes from a crash are removed (link first, then folder).
    """
    run_root = Path(run_root).resolve()
    for sub in ('', 'runs', 'records', 'homes', 'tools'):
        (run_root / sub).mkdir(parents=True, exist_ok=True)
        os.chmod(run_root / sub, 0o700)
    if inside_git(run_root):
        raise ConfigurationCheckFailed(f'run root {run_root} is inside a Git worktree')
    stale = [teardown_codex_home(run_root, int(p.name.split('_w')[1]))
             for p in sorted((run_root / 'homes').glob('codex_w*'))]
    settings = run_root / 'tools/claude_settings.json'
    settings.write_text(json.dumps(CLAUDE_SETTINGS, indent=2) + '\n')
    paths = write_language_paths(run_root)
    config = dict(run_root=str(run_root), bin_dir=str(Path(bin_dir).resolve()) if bin_dir else None, runner=VERSION)
    if bin_dir:
        config['binaries'] = check_binaries(bin_dir, 'prepare_collection')
    save_json(run_root / 'tools/collection.json', config)
    mpl_seed = seed_matplotlib(run_root) if seed_mpl else []
    credentials = [str(p) for p in (run_root / 'homes').rglob('*')
                   if p.name in ('auth.json', '.credentials.json') or p.is_symlink()]
    return dict(run_root=str(run_root), mode=oct(os.stat(run_root).st_mode & 0o777), homes=sorted(
        p.name for p in (run_root / 'homes').iterdir()), credentials_present=credentials,
        claude_settings=str(settings), language_paths=paths, stale_homes_removed=stale, mpl_seed=mpl_seed,
        bin_dir=config['bin_dir'])


# Codex home per worker -----------------------------------------------------------------------

def codex_home(run_root: Path, worker: int = 0) -> Path:
    return Path(run_root) / 'homes' / f'codex_w{worker}'


def setup_codex_home(run_root: Path, worker: int = 0, auth: Path | None = None) -> dict:
    """Create homes/codex_w<k> (mode 700) with one symlink to the Codex login. The file is never opened."""
    auth = Path(auth or CODEX_AUTH)
    home = codex_home(run_root, worker)
    if home.exists() or home.is_symlink():
        teardown_codex_home(run_root, worker)
    home.mkdir(parents=True)
    os.chmod(home, 0o700)
    os.symlink(auth, home / 'auth.json')
    return check_codex_link(run_root, worker, auth)


def check_codex_link(run_root: Path, worker: int = 0, auth: Path | None = None) -> dict:
    auth = Path(auth or CODEX_AUTH)
    link = codex_home(run_root, worker) / 'auth.json'
    ok = os.path.islink(link) and os.readlink(link) == str(auth)
    row = dict(home=str(codex_home(run_root, worker)), link=ok, target=str(auth))
    if not ok:
        raise ConfigurationCheckFailed(f'Codex login link changed in {link.parent}')
    return row


def teardown_codex_home(run_root: Path, worker: int = 0, auth: Path | None = None) -> dict:
    """Remove the link first (never its target), then the home. Check that the target still exists."""
    auth = Path(auth or CODEX_AUTH)
    home = codex_home(run_root, worker)
    link = home / 'auth.json'
    unlinked = False
    if os.path.islink(link):
        os.unlink(link)
        unlinked = True
    elif link.exists():
        raise ConfigurationCheckFailed(f'{link} is a regular file; the runner never copies the Codex login')
    shutil.rmtree(home, ignore_errors=True)
    return dict(home=str(home), unlinked=unlinked, removed=not home.exists(),
                target_is_file=os.path.isfile(auth) and not os.path.islink(auth))


def collect_rollout(run_root: Path, worker: int, record: Path, attempt: int) -> dict:
    """Copy the Codex rollout of one call into the record (without session_meta git fields), then delete it."""
    home = codex_home(run_root, worker)
    files = sorted((home / 'sessions').rglob('rollout-*.jsonl')) if (home / 'sessions').is_dir() else []
    out = Path(record) / f'rollout_attempt{attempt}.jsonl'
    summary = dict(files=len(files), file=out.name if files else None, models=[], efforts=[], profiles=[],
                   base_instructions_sha256=None, cli_version=None, last_token_usage=None)
    lines = []
    for path in files:
        for line in path.read_text(errors='replace').splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                lines.append(line)
                continue
            payload = event.get('payload') if isinstance(event, dict) else None
            if isinstance(payload, dict):
                if event.get('type') == 'session_meta':
                    payload.pop('git', None)
                    base = payload.get('base_instructions')
                    text = base.get('text') if isinstance(base, dict) else base
                    if isinstance(text, str):
                        summary['base_instructions_sha256'] = text_sha(text)
                    summary['cli_version'] = payload.get('cli_version')
                elif event.get('type') == 'turn_context':
                    summary['models'].append(payload.get('model'))
                    summary['efforts'].append(payload.get('effort'))
                    summary['profiles'].append(payload.get('active_permission_profile'))
                elif payload.get('type') == 'token_count' and isinstance(payload.get('info'), dict):
                    summary['last_token_usage'] = payload['info'].get('total_token_usage')
            lines.append(json.dumps(event))
        path.unlink()
    if files:
        out.write_text('\n'.join(lines) + '\n')
    for key in ('models', 'efforts'):
        summary[key] = sorted({v for v in summary[key] if v})
    summary['profiles'] = [p for p in summary['profiles'] if p][:1]
    return summary


def rollout_problems(summary: dict, block_base_sha: str | None = None) -> list[str]:
    problems = []
    if not summary['files']:
        return ['no Codex rollout was written']
    if summary['models'] != [MODELS['luna']['model_id']]:
        problems.append(f'rollout models {summary["models"]}')
    if summary['efforts'] != [MODELS['luna']['effort']]:
        problems.append(f'rollout effort {summary["efforts"]}')
    if block_base_sha and summary['base_instructions_sha256'] != block_base_sha:
        problems.append('Codex base instructions differ from the first rollout of the block')
    return problems


# Claude per-block state checks (no model call) ------------------------------------------------

def claude_login_state(claude_bin: Path, run_tmp: Path) -> dict:
    """claude auth status --json. Keep only loggedIn, authMethod, and subscriptionType."""
    env = clean_env(run_tmp, dict(CLAUDE_ENV))
    out = subprocess.run([str(claude_bin), 'auth', 'status', '--json'], capture_output=True, text=True, env=env,
                         timeout=120, check=False, stdin=subprocess.DEVNULL)
    try:
        data = json.loads(out.stdout)
    except ValueError:
        data = {}
    return dict(returncode=out.returncode, **{k: data.get(k) for k in ('loggedIn', 'authMethod', 'subscriptionType')})


def claude_state_problems(run_root: Path, claude_json: Path | None = None, projects: Path | None = None) -> dict:
    """The run root must leave no project key in ~/.claude.json and no folder under ~/.claude/projects.

    The runner loads ~/.claude.json only to list the keys of projects. It stores no value.
    """
    claude_json = Path(claude_json or HOME / '.claude.json')
    projects = Path(projects or HOME / '.claude/projects')
    root = str(Path(run_root).resolve())
    sanitized = re.sub(r'[^A-Za-z0-9]', '-', root)
    keys = []
    if claude_json.is_file():
        try:
            keys = [k for k in (json.loads(claude_json.read_text()).get('projects') or {}) if k.startswith(root)]
        except ValueError:
            keys = ['<unreadable ~/.claude.json>']
    folders = [p.name for p in projects.iterdir() if sanitized in p.name] if projects.is_dir() else []
    return dict(project_keys=len(keys), project_folders=folders,
                claude_json_mtime=claude_json.stat().st_mtime if claude_json.exists() else None,
                ok=not keys and not folders)


def codex_login_ok(run_root: Path, worker: int, run_tmp: Path) -> dict:
    """codex login status with the worker home. Keep only the exit code and the first word of the method."""
    _, codex_bin, _ = provider_bins(run_root)
    env = clean_env(run_tmp, {'CODEX_HOME': str(codex_home(run_root, worker))})
    out = subprocess.run([str(codex_bin), 'login', 'status'], capture_output=True, text=True, env=env, timeout=120,
                         check=False, stdin=subprocess.DEVNULL)
    words = (out.stdout + ' ' + out.stderr).split()
    method = next((w for w in words if w.lower() in ('chatgpt', 'api')), words[2] if len(words) > 2 else None)
    return dict(returncode=out.returncode, method=method)


# One run folder -------------------------------------------------------------------------------

@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def opaque_folder(run_root: Path, ident: str) -> str:
    """Opaque cell folder name c<NNNNN>. The mapping stays in records/cell_index.json (tools cannot read it)."""
    index = Path(run_root) / 'records/cell_index.json'
    with _locked(index.with_suffix('.lock')):
        mapping = json.loads(index.read_text()) if index.is_file() else {}
        if ident not in mapping:
            mapping[ident] = f'c{len(mapping) + 1:05d}'
            index.write_text(json.dumps(mapping, indent=1, sort_keys=True) + '\n')
        return mapping[ident]


def provider_env(cell: dict, run_root: Path, run_tmp: Path, wrapper: Path | None, worker: int = 0) -> dict:
    path = language_path(run_root, cell['lang'])
    if MODELS[cell['model']]['cli'] == 'codex':
        return clean_env(run_tmp, {'CODEX_HOME': str(codex_home(run_root, worker))}, path)
    extra = dict(CLAUDE_ENV, CLAUDE_CODE_TMPDIR=str(run_tmp))
    if wrapper is not None:
        extra['CLAUDE_CODE_SHELL_PREFIX'] = str(wrapper)
    return clean_env(run_tmp, extra, path)


def prepare_run(cell: dict, run_root: Path, worker: int = 0) -> dict:
    """Workspace, record folder, profiles, prompt, and data copies of one cell."""
    run_root = Path(run_root)
    ident = cell_id(cell)
    folder = fp.package_dir(cell['task'])
    task = fp.load_task(folder)
    opaque = opaque_folder(run_root, ident)
    base = run_root / 'runs' / opaque
    shutil.rmtree(base, ignore_errors=True)
    work, tmp = base / 'work', base / 'work/.tmp'
    tmp.mkdir(parents=True)
    os.chmod(base, 0o700)
    record = run_root / 'records' / ident
    record.mkdir(parents=True, exist_ok=True)
    for name in task['files']:
        shutil.copyfile(folder / 'data' / name, work / name)
    if (run_root / 'tools/mpl_seed').is_dir():
        shutil.copytree(run_root / 'tools/mpl_seed', tmp / 'mpl')
    prompt = (folder / 'prompts' / fp.prompt_name(cell['lang'], cell['prompt'], cell['condition'])).read_text()
    (record / 'prompt.txt').write_text(prompt)
    extra_read = [path_dir(run_root, cell['lang'])] if path_dir(run_root, cell['lang']).is_dir() else []
    exec_profile = execution_profile(work, tmp, cell['lang'], extra_read, exec_allow=extra_read)
    (record / 'execution.sb').write_text(exec_profile)
    agent = cell['condition'] == 'agent'
    wrapper = None
    if MODELS[cell['model']]['cli'] == 'claude' and agent:
        wrapper = base / 'tool_exec.sh'
        wrapper.write_text(TOOL_WRAPPER.format(profile=_shell_quote(str(record / 'execution.sb'))))
        os.chmod(wrapper, 0o700)
    cwd = work if agent else base / 'generate_1'
    argv = real_argv(cell, run_root, cwd, record, 1)
    env = provider_env(cell, run_root, tmp, wrapper, worker)
    info = dict(cell=cell, id=ident, folder=opaque, work=str(work), tmp=str(tmp), record=str(record),
                wrapper=str(wrapper or ''), worker=worker, provider_argv=argv, provider_env=masked_env(env),
                prompt_sha256=text_sha(prompt), execution_profile_sha256=text_sha(exec_profile),
                inputs={name: sha(work / name) for name in task['files']}, prepared_at=now(), runner=VERSION)
    save_json(record / 'prepared.json', info)
    return info


# Provider call ---------------------------------------------------------------------------------

def real_argv(cell: dict, run_root: Path, cwd: Path, record: Path, attempt: int) -> list[str]:
    """The claude or codex command of one attempt (recorded in dry mode, run in a real collection)."""
    agent = cell['condition'] == 'agent'
    if MODELS[cell['model']]['cli'] == 'codex':
        return codex_argv(agent, Path(cwd), Path(record) / f'last_attempt{attempt}.txt', cell['lang'], run_root)
    return claude_argv(agent, Path(run_root) / 'tools/claude_settings.json', run_root)


def real_provider(run_root: Path) -> Callable[[dict, dict, int, Path], list[str]]:
    """Provider callable for a real collection. In dry mode check_dry_argv refuses its commands."""
    def provider(cell: dict, info: dict, attempt: int, cwd: Path) -> list[str]:
        return real_argv(cell, run_root, cwd, Path(info['record']), attempt)
    return provider


def check_dry_argv(argv: list[str]) -> None:
    name = Path(argv[0]).name
    real = Path(os.path.realpath(argv[0]))
    if (name in PROVIDER_NAMES or real.name in PROVIDER_NAMES or name.startswith('claude-')
            or real.name.startswith('claude-') or 'claude/versions' in str(real)
            or re.search(r'/codex-\d', str(real)) or real == Path(os.path.realpath(CODEX_BIN))):
        raise DryRunViolation(f'dry mode refuses the provider binary {argv[0]}')


def _descendants(pid: int) -> list[str]:
    out = subprocess.run(['ps', '-A', '-o', 'pid=,ppid=,command='], capture_output=True, text=True).stdout
    children: dict[int, list] = {}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append((int(parts[0]), parts[2]))
    found, stack = [], [pid]
    while stack:
        for child, command in children.get(stack.pop(), []):
            found.append(command)
            stack.append(child)
    return found


def call_provider(argv: list[str], env: dict, cwd: Path, prompt: str, record: Path, attempt: int, *,
                  agent: bool, dry: bool = True, stall_idle: float = STALL_IDLE_SECONDS,
                  hard_limit: float | None = None, monitor: float = MONITOR_SECONDS,
                  net_probe: Callable[[int], int | None] | None = None,
                  tool_filter: Callable[[str], bool] = lambda c: 'codex-code-mode-host' not in c,
                  redact: bool = False) -> dict:
    """Run one provider call. Writes stdout, stderr, receive times, and call_attemptN.json.

    The stall rule applies to agent calls only. A generation call runs until it ends or reaches its hard limit.
    """
    if dry:
        check_dry_argv(argv)
    limit = hard_limit if hard_limit is not None else CALL_LIMIT_SECONDS['agent' if agent else 'generation']
    start = time.monotonic()
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    state = dict(last_event=start, stdout_bytes=0, events=0, partial=b'', lines=0)
    out_chunks, err_chunks, times = [], [], []
    lock = threading.Lock()

    def read_stdout():
        while True:
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                break
            received = round(time.monotonic() - start, 3)
            with lock:
                out_chunks.append(chunk)
                state['stdout_bytes'] += len(chunk)
                buffer = state['partial'] + chunk
                *lines, state['partial'] = buffer.split(b'\n')
                for line in lines:
                    state['lines'] += 1
                    times.append([state['lines'], received])
                    try:
                        complete = isinstance(json.loads(line), dict)
                    except ValueError:
                        complete = False
                    if complete:  # a completed stream event; plain text does not count
                        state['events'] += 1
                        state['last_event'] = time.monotonic()
                if len(state['partial']) > 1 << 20:
                    state['partial'] = state['partial'][-4096:]

    def read_stderr():
        for chunk in iter(lambda: process.stderr.read(65536), b''):
            err_chunks.append(chunk)

    threads = [threading.Thread(target=read_stdout, daemon=True), threading.Thread(target=read_stderr, daemon=True)]
    for thread in threads:
        thread.start()
    try:
        process.stdin.write(prompt.encode())
        process.stdin.close()
    except BrokenPipeError:
        pass
    stalled = timed_out = False
    diag = None
    while True:
        try:
            process.wait(timeout=monitor)
            break
        except subprocess.TimeoutExpired:
            pass
        elapsed = time.monotonic() - start
        with lock:
            idle = time.monotonic() - state['last_event']
        if agent and idle >= stall_idle:
            tools = [c for c in _descendants(process.pid) if tool_filter(c)]
            if not tools:
                with lock:
                    first_bytes = state['stdout_bytes']
                first_net = net_probe(process.pid) if net_probe else None
                time.sleep(min(STALL_PROBE_SECONDS, max(monitor, 0.5)))
                second_net = net_probe(process.pid) if net_probe else None
                with lock:
                    second_bytes = state['stdout_bytes']
                growing = second_bytes > first_bytes or (
                    first_net is not None and second_net is not None and second_net > first_net)
                stalled = True
                diag = dict(t=round(elapsed, 1), idle=round(idle, 1), stdout_bytes=[first_bytes, second_bytes],
                            net_bytes=[first_net, second_net], kind='runaway_generation' if growing else 'stream_hang')
                break
        if elapsed >= limit:
            timed_out = True
            diag = dict(t=round(elapsed, 1), idle=round(idle, 1),
                        tools=[c for c in _descendants(process.pid) if tool_filter(c)][:10])
            break
    if stalled or timed_out:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                continue
    for thread in threads:
        thread.join(timeout=10)
    stdout = b''.join(out_chunks).decode(errors='replace')
    stderr = b''.join(err_chunks).decode(errors='replace')
    if redact:
        stdout, stderr = redact_stream(stdout), strip_license_details(stderr)
    (record / f'provider_attempt{attempt}.stdout').write_text(stdout)
    (record / f'provider_attempt{attempt}.stderr').write_text(stderr)
    (record / f'provider_attempt{attempt}.times.json').write_text(json.dumps(times) + '\n')
    call = dict(attempt=attempt, argv_executed=argv, returncode=None if (stalled or timed_out) else process.returncode,
                wall_seconds=round(time.monotonic() - start, 3), stalled=stalled, hard_timeout=timed_out,
                stall_diag=diag, stdout_bytes=state['stdout_bytes'], events=state['events'],
                stall_rule='agent calls only', stall_idle_seconds=stall_idle if agent else None,
                hard_limit_seconds=limit, raw_stream_file=f'provider_attempt{attempt}.stdout',
                receive_times_file=f'provider_attempt{attempt}.times.json')
    call.update(parse_stream(stdout))
    call['ok'] = (call['returncode'] == 0 and not stalled and not timed_out and call['result_event']
                  and not call['stream_errors'] and not call['limit_rejected'])
    save_json(record / f'call_attempt{attempt}.json', {k: v for k, v in call.items() if k != 'init'})
    return call


LIMIT_TEXT = re.compile(r'usage limit|hit your limit|limit reached|rate limit|out of (extra )?usage', re.I)


def parse_stream(stdout: str) -> dict:
    """Text, usage, completion, and error checks of a claude stream-json or codex --json stream."""
    events = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    text, usage, models, init, result = [], None, set(), None, False
    model_usage, message_usage, errors, limit_rejected, thread = None, {}, [], False, False
    for event in events:
        kind = event.get('type')
        if kind == 'system' and event.get('subtype') == 'init':
            init = event
            models.add(event.get('model'))
        elif kind == 'assistant':
            message = event.get('message') or {}
            models.add(message.get('model'))
            text += [c.get('text', '') for c in message.get('content') or [] if c.get('type') == 'text']
            if isinstance(message.get('usage'), dict):
                message_usage[message.get('id') or len(message_usage)] = message['usage']
        elif kind == 'rate_limit_event':
            info = event.get('rate_limit_info') or event.get('rateLimitInfo') or {}
            if info.get('status') == 'rejected':
                limit_rejected = True
        elif kind == 'result':
            result = True
            usage = event.get('usage')
            model_usage = event.get('modelUsage')
            if isinstance(model_usage, dict):
                models.update(model_usage)
            if event.get('is_error') or event.get('terminal_reason') == 'api_error':
                message = str(event.get('result') or event.get('error') or event.get('subtype'))
                errors.append(message[:500])
                if LIMIT_TEXT.search(message):
                    limit_rejected = True
            elif not text and isinstance(event.get('result'), str):
                text.append(event['result'])
        elif kind == 'thread.started':
            thread = True
        elif kind == 'item.completed' and (event.get('item') or {}).get('type') == 'agent_message':
            text.append(event['item'].get('text', ''))
        elif kind == 'item.completed' and (event.get('item') or {}).get('type') == 'error':
            errors.append(str(event['item'].get('message'))[:500])
        elif kind in ('error', 'turn.failed'):
            message = str(event.get('message') or (event.get('error') or {}).get('message') or kind)
            errors.append(message[:500])
            if LIMIT_TEXT.search(message):
                limit_rejected = True
        elif kind == 'turn.completed':
            result = True
            usage = event.get('usage')
    partial = None
    if message_usage:
        partial = {}
        for item in message_usage.values():
            for key, value in item.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    partial[key] = partial.get(key, 0) + value
    return dict(text='\n'.join(text), usage=usage, model_usage=model_usage, partial_usage=partial,
                models=sorted(m for m in models if m), init_event=bool(init) or thread, init=init,
                result_event=result, stream_errors=errors, limit_rejected=limit_rejected, parsed_events=len(events))


# Per-call configuration checks -----------------------------------------------------------------

BUILTIN_AGENTS = {'claude', 'Explore', 'general-purpose', 'Plan'}


def claude_init_problems(init: dict | None, cwd: Path, agent: bool) -> list[str]:
    """The system/init check of harmonize.md 4.1. An empty list means the call passed."""
    if not init:
        return ['no system/init event']
    expect = {'claude_code_version': CLAUDE_VERSION, 'model': MODELS[CLAUDE_MODEL]['model_id'], 'cwd': str(cwd),
              'output_style': 'default', 'permissionMode': 'default', 'apiKeySource': 'none',
              'analytics_disabled': True, 'product_feedback_disabled': True, 'fast_mode_state': 'off'}
    problems = [f'{key}={init.get(key)!r}' for key, value in expect.items() if init.get(key) != value]
    if sorted(init.get('tools') or []) != (['Bash'] if agent else []):
        problems.append(f'tools={init.get("tools")!r}')
    for key in ('mcp_servers', 'slash_commands', 'skills'):
        if init.get(key):
            problems.append(f'{key}={init.get(key)!r}')
    plugins = init.get('plugins') or []
    names = [p.get('source') or p.get('name') if isinstance(p, dict) else p for p in plugins]
    if names != ['agents-md@builtin']:
        problems.append(f'plugins={names!r}')
    agents = init.get('agents') or []
    if not set(agents) <= BUILTIN_AGENTS:
        problems.append(f'agents={agents!r}')
    if init.get('memory_paths'):
        problems.append(f'memory_paths={init.get("memory_paths")!r}')
    return problems


def model_problems(call: dict, model: str) -> tuple[list[str], bool]:
    """Model names of a call. Returns the problems and whether an expensive model (Opus, Fable) appeared."""
    expected = MODELS[model]['model_id']
    names = set(call.get('models') or [])
    expensive = any(EXPENSIVE_MODEL.search(n) for n in names)
    wrong = [n for n in names if n != expected and not (MODELS[model]['cli'] == 'claude' and SIDE_MODEL.search(n))]
    return ([f'unexpected model {n}' for n in sorted(wrong)], expensive)


# Execution feedback and redaction -------------------------------------------------------------

LICENSE_IN_JSON = re.compile(r'(?:Stata[ \t]+license|Serial[ \t]+number|Licensed[ \t]+to):(?:[^"\\]|\\[^n])*'
                             r'(?:\\n[ \t]+(?:[^"\\]|\\[^n])*)*(?:\\n)?', re.I)


def redact_stream(text: str) -> str:
    """Remove Stata licence lines from a JSON stream (escaped newlines) and from plain lines."""
    return strip_license_details(LICENSE_IN_JSON.sub('', text))


def redact_workspace(work: Path) -> list[str]:
    """Strip Stata licence lines from every text log in a workspace before it is copied."""
    changed = []
    for path in Path(work).rglob('*'):
        if '.tmp' in path.relative_to(work).parts or not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() in ('.log', '.smcl', '.txt') and path.stat().st_size <= FINAL_FILE_MAX_BYTES:
            text = path.read_text(errors='replace')
            clean = strip_license_details(text)
            if clean != text:
                path.write_text(clean)
                changed.append(str(path.relative_to(work)))
    return changed


def repair_feedback(execution: dict) -> dict:
    """Fixed feedback fields (harmonize.md 6.4). No argv, environment, or profile hash."""
    feedback = dict(returncode=execution.get('returncode'), timed_out=bool(execution.get('timed_out')),
                    stdout_tail=(execution.get('stdout') or '')[-FEEDBACK_TAIL:],
                    stderr_tail=(execution.get('stderr') or '')[-FEEDBACK_TAIL:])
    if execution.get('log_tail') is not None:
        feedback['log_tail'] = strip_license_details(execution['log_tail'])[-FEEDBACK_TAIL:]
    return feedback


def repair_text(initial: str, previous: str, execution: dict, problem: str) -> str:
    """Self-repair prompt. The feedback holds only the fixed fields of repair_feedback."""
    return (initial + '\n\nThe previous script did not finish with all required output fields.\n'
            + 'Previous script:\n' + previous + '\nExecution feedback:\n'
            + json.dumps(repair_feedback(execution)) + '\n' + problem + '\nReturn the complete corrected fenced script.\n')


# Execution and scoring ---------------------------------------------------------------------------

HARNESS_FILES = {'printed_output.txt', 'verification.txt', 'result.json', 'run.log'}
SCRIPT_SUFFIXES = {'.do', '.py', '.r', '.ado'}


def written_files(work: Path, started: float, ended: float, inputs: set[str]) -> list[str]:
    """Text files that the final execution wrote (output-channel rule of 2026-09-24)."""
    out = []
    for path in sorted(Path(work).iterdir()):
        if not path.is_file() or path.is_symlink() or path.name in inputs or path.name in HARNESS_FILES:
            continue
        if path.suffix.lower() in SCRIPT_SUFFIXES or path.stat().st_size > FINAL_FILE_MAX_BYTES:
            continue
        if started - FINAL_FILE_MARGIN_SECONDS <= path.stat().st_mtime <= ended + FINAL_FILE_MARGIN_SECONDS:
            out.append(path.name)
    return out


def execute(work: Path, lang: str, record: Path, number: int, profile_text: str | None = None,
            timeout: float = EXEC_LIMIT_SECONDS, path: str | None = None, inputs: set[str] | None = None) -> dict:
    """Run the workspace script under the execution profile and the clean environment."""
    work, record = Path(work), Path(record)
    tmp = work / '.tmp'
    tmp.mkdir(exist_ok=True)
    profile_text = profile_text or execution_profile(work, tmp, lang)
    profile = record / f'execution_attempt{number}.sb'
    profile.write_text(profile_text)
    for name in ('result.json', 'run.log'):
        (work / name).unlink(missing_ok=True)
    script = SCRIPT_NAME[lang]
    command = {'python': [str(VENV / 'bin/python'), script], 'r': [RSCRIPT, '--vanilla', script],
               'stata': [STATA_BIN, '-b', 'do', script]}[lang]
    argv = ['/usr/bin/sandbox-exec', '-f', str(profile), *command]
    env = clean_env(tmp, path=path)
    start, started_at = time.monotonic(), time.time()
    try:
        process = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, timeout=timeout,
                                 stdin=subprocess.DEVNULL)
        result = dict(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr, timed_out=False)
    except subprocess.TimeoutExpired:
        result = dict(returncode=None, stdout='', stderr='', timed_out=True)
    ended_at = time.time()
    ok = result['returncode'] == 0 and not result['timed_out']
    if lang == 'stata':
        result['stdout'] = strip_license_details(result['stdout'])
        result['stderr'] = strip_license_details(result['stderr'])
        log = work / 'run.log'
        text = strip_license_details(log.read_text(errors='replace')) if log.exists() else ''
        if log.exists():
            log.write_text(text)
        result['log_tail'] = text[-FEEDBACK_TAIL:] if log.exists() else None
        result['stata_error'] = bool(STATA_ERROR.search(text))
        result['final_files'] = written_files(work, started_at, ended_at, set(inputs or ()))
        ok = ok and log.exists() and not result['stata_error']
    result.update(ok=ok, wall_seconds=round(time.monotonic() - start, 3), argv=argv, window=[started_at, ended_at],
                  profile_sha256=text_sha(profile_text), env=masked_env(env))
    save_json(record / f'execution_attempt{number}.json', result)
    return result


def score(cell: dict, work: Path, execution: dict | None) -> dict:
    folder = fp.package_dir(cell['task'])
    if cell['lang'] != 'stata':
        return fs.score_run(folder, cell['lang'], work)
    execution = execution or {}
    return fs.score_run(folder, 'stata', work, stdout=execution.get('stdout'),
                        final_files=execution.get('final_files') or [])


NUMBER = re.compile(r'(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w.])')
COMMENT = {'python': re.compile(r'#[^\n]*'), 'r': re.compile(r'#[^\n]*'),
           'stata': re.compile(r'//[^\n]*|/\*.*?\*/|^\s*\*[^\n]*', re.S | re.M)}


def literal_answers(script: str, lang: str, rows: dict, min_digits: int = 4) -> dict:
    """Static check: numeric literals in the final script (comments removed) equal to a reported estimate or
    standard error. A literal counts when it has at least min_digits significant digits and agrees with the
    reported value to the precision of the literal. Diagnostic only: it does not change the primary outcome."""
    code = COMMENT[lang].sub('', script or '')
    literals = []
    for match in NUMBER.finditer(code):
        token = match.group(0).lstrip('+-')
        digits = re.sub(r'[eE].*$', '', token).replace('.', '').lstrip('0')
        if len(digits) >= min_digits:
            literals.append(match.group(0))
    hits = []
    for term, row in (rows or {}).items():
        for field in ('estimate', 'se'):
            value = (row or {}).get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value == 0:
                continue
            for token in literals:
                number = float(token)
                decimals = len(token.split('.')[1].split('e')[0].split('E')[0]) if '.' in token else 0
                tolerance = max(0.5 * 10 ** -decimals, abs(value) * 1e-9)
                if abs(number - value) <= tolerance:
                    hits.append(dict(term=term, field=field, literal=token))
    return dict(literals_checked=len(literals), matches=hits, flagged=bool(hits))


# Environment check per block ------------------------------------------------------------------

def environment_fields() -> dict:
    manifest = json.loads(ENV_MANIFEST.read_text())
    return {'environment.manifest_sha256': manifest['manifest_sha256'],
            'environment.manifest_file_sha256': sha(ENV_MANIFEST),
            'environment.packages_json_sha256': sha(ENV_DIR / 'packages.json'),
            'environment.r_lock_sha256': sha(ENV_DIR / 'r_lock.json'),
            'environment.uv_lock_sha256': sha(ROOT / 'uv.lock')}


def environment_check(block: str, out_dir: Path, run_check: Callable[[Path], dict] | None = None) -> dict:
    """Run check_environment.py for one block point. Raise EnvironmentChanged on a hash change."""
    output = Path(out_dir) / f'environment_check_{block}.json'

    def default(path: Path) -> dict:
        subprocess.run([str(VENV / 'bin/python'), str(ENV_DIR / 'check_environment.py'), '--output', str(path)],
                       cwd=ROOT, capture_output=True, text=True, check=False, timeout=1800)
        return json.loads(path.read_text())

    checked = (run_check or default)(output)
    expected = environment_fields()
    row = dict(block=block, at=now(), checked_manifest_sha256=checked.get('manifest_sha256'),
               failures=checked.get('failures'), **expected)
    row['matches'] = checked.get('manifest_sha256') == expected['environment.manifest_sha256'] and not checked.get('failures')
    if not row['matches']:
        raise EnvironmentChanged(f'block {block}: environment hash {checked.get("manifest_sha256")} differs from '
                                 f'{expected["environment.manifest_sha256"]}')
    return row


# One cell ---------------------------------------------------------------------------------------

def gate(watcher: uw.Watcher | None, cell: dict) -> dict | None:
    """Wait until the usage watcher lets a request for this model start. Record the wait."""
    if watcher is None:
        return None
    before = watcher.clock()
    at_request = watcher.decide(cell['model'])
    decision = watcher.gate(cell['model'])
    line = watcher.line(decision) if decision else {}
    line.update(decision_at_request=at_request['decision'], reason_at_request=at_request['reason'],
                waited_seconds=round(watcher.clock() - before, 3))
    return line


def _call_checks(cell: dict, call: dict, cwd: Path, agent: bool, run_root: Path, worker: int, record_dir: Path,
                 attempt: int, block_state: dict) -> list[str]:
    """Configuration checks of one call. Writes init or rollout summaries into the record."""
    problems, expensive = model_problems(call, cell['model'])
    if expensive:
        raise ModelMismatch(f'{cell_id(cell)} attempt {attempt}: expensive model in {call.get("models")}')
    if MODELS[cell['model']]['cli'] == 'claude':
        if call.get('init'):
            save_json(record_dir / f'init_attempt{attempt}.json', call['init'])
        if not call.get('limit_rejected'):
            problems += claude_init_problems(call.get('init'), cwd, agent)
    else:
        rollout = collect_rollout(run_root, worker, record_dir, attempt)
        save_json(record_dir / f'rollout_attempt{attempt}.summary.json', rollout)
        call['rollout'] = rollout
        if rollout.get('last_token_usage') and call.get('partial_usage') is None:
            call['partial_usage'] = rollout['last_token_usage']
        if not call.get('limit_rejected'):
            base = block_state.setdefault('codex_base_sha256', rollout.get('base_instructions_sha256'))
            problems += rollout_problems(rollout, base)
            if not call.get('init_event'):
                problems.append('no thread.started event')
            problems += [f'stream error: {e}' for e in call.get('stream_errors') or []]
        check_codex_link(run_root, worker)
    return problems


def run_cell(cell: dict, run_root: Path, *, provider: Callable[[dict, dict, int, Path], list[str]],
             watcher: uw.Watcher | None = None, dry: bool = True, stall_idle: float = STALL_IDLE_SECONDS,
             hard_limit: float | None = None, monitor: float = MONITOR_SECONDS, provider_env_extra: dict | None = None,
             retry_stall: bool = True, block_fields: dict | None = None,
             tool_filter: Callable[[str], bool] | None = None, enforce_checks: bool | None = None,
             worker: int = 0, block_state: dict | None = None) -> dict:
    """Run one cell. provider(cell, info, attempt, cwd) returns the argv that runs in place of the real CLI
    (dry mode: a local fake command). The real argv is recorded in prepared.json and command_attemptN.json.

    enforce_checks (default: not dry) makes a failed configuration check stop the block.
    """
    if dry and watcher is not None and not getattr(watcher, 'dry_poller', False):
        raise DryRunViolation('dry mode needs a watcher with a fake Codex usage reader (dry_poller = True)')
    enforce = (not dry) if enforce_checks is None else enforce_checks
    block_state = block_state if block_state is not None else {}
    codex = MODELS[cell['model']]['cli'] == 'codex'
    if codex and not os.path.islink(codex_home(run_root, worker) / 'auth.json'):
        setup_codex_home(run_root, worker)
    info = prepare_run(cell, run_root, worker)
    record_dir, work = Path(info['record']), Path(info['work'])
    agent = cell['condition'] == 'agent'
    task_files = set(fp.load_task(fp.package_dir(cell['task']))['files'])
    lang_path = language_path(run_root, cell['lang'])
    record = dict(id=info['id'], folder=info['folder'], cell=cell, started_at=now(), calls=[], executions=[],
                  stop_reason=None, stalls_before=[], limit_rejections=[], config_problems=[], runner=VERSION,
                  dry=dry, **(block_fields or {}))
    initial = text = (record_dir / 'prompt.txt').read_text()
    script = SCRIPT_NAME[cell['lang']]
    attempt, rounds = 0, SELF_REPAIR_ROUNDS if cell['condition'] == 'self_repair' else 1
    round_number = 1
    while round_number <= rounds:
        attempt += 1
        record.setdefault('watch', []).append(gate(watcher, cell))
        cwd = work if agent else (Path(info['work']).parent / f'generate_{round_number}')
        if not agent:
            (cwd / '.tmp').mkdir(parents=True, exist_ok=True)
        wrapper = Path(info['wrapper']) if info['wrapper'] else None
        env = provider_env(cell, run_root, Path(info['tmp']), wrapper, worker)
        env.update(provider_env_extra or {})
        argv = provider(cell, info, attempt, cwd)
        save_json(record_dir / f'command_attempt{attempt}.json',
                  dict(real_argv=real_argv(cell, run_root, cwd, record_dir, attempt), executed_argv=argv, dry=dry,
                       cwd=str(cwd), env=masked_env(env), stall_rule='agent calls only',
                       stall_idle_seconds=stall_idle if agent else None,
                       hard_limit_seconds=hard_limit or CALL_LIMIT_SECONDS['agent' if agent else 'generation']))
        call = call_provider(argv, env, cwd, text, record_dir, attempt, agent=agent, dry=dry, stall_idle=stall_idle,
                             hard_limit=hard_limit, monitor=monitor, redact=cell['lang'] == 'stata',
                             **({'tool_filter': tool_filter} if tool_filter else {}))
        if watcher is not None:
            watcher.observe_call(cell['model'], record_dir, call)
        problems = _call_checks(cell, call, cwd, agent, Path(run_root), worker, record_dir, attempt, block_state)
        record['calls'].append({k: call.get(k) for k in ('attempt', 'ok', 'returncode', 'stalled', 'hard_timeout',
                                                         'stall_diag', 'wall_seconds', 'events', 'usage',
                                                         'partial_usage', 'model_usage', 'models', 'stream_errors',
                                                         'limit_rejected')})
        if problems:
            record['config_problems'].append(dict(attempt=attempt, problems=problems))
            if enforce:
                record['stop_reason'] = 'configuration_check'
                record['scoring'] = dict(passed=False, reason='configuration check failed')
                record['success'] = None
                record['completed_at'] = now()
                save_json(record_dir / 'record.json', record)
                message = f'{info["id"]} attempt {attempt}: ' + '; '.join(problems)
                if any(p.startswith('unexpected model') for p in problems):
                    # Any model id other than the collection model (decision record 2026-09-30_sonnet-5-5).
                    raise ModelMismatch(message)
                raise ConfigurationCheckFailed(message)
        if call.get('limit_rejected'):
            # A provider stop: not scored and not a model failure. The watcher decides when to repeat.
            record['limit_rejections'].append(dict(attempt=attempt, at=now(), errors=call.get('stream_errors')))
            record['calls'].pop()
            if len(record['limit_rejections']) >= MAX_LIMIT_REJECTIONS:
                record['stop_reason'] = 'provider_limit'
                break
            continue
        if not call['ok']:
            kind = (call.get('stall_diag') or {}).get('kind')
            record['stop_reason'] = ('runaway' if kind == 'runaway_generation' else 'stall') if call['stalled'] else (
                'timeout' if call['hard_timeout'] else 'provider_error')
            if record['stop_reason'] == 'stall' and retry_stall and not record['stalls_before']:
                # Keep the stalled attempt, then repeat the run once in a fresh workspace.
                record['stalls_before'].append(dict(calls=list(record['calls'])))
                record['calls'], record['stop_reason'] = [], None
                shutil.rmtree(Path(info['work']).parent, ignore_errors=True)
                info = prepare_run(cell, run_root, worker)
                work = Path(info['work'])
                round_number = 1
                continue
            break
        if not agent:
            code = extract_code(call['text'])
            if code is None:
                execution = dict(ok=False, returncode=None, timed_out=False, stdout='', stderr='',
                                 error='No fenced script was returned.')
                problem = 'No fenced script.'
            else:
                (work / script).write_text(code)
                execution = execute(work, cell['lang'], record_dir, attempt, info_profile(record_dir),
                                    path=lang_path, inputs=task_files)
                scored = score(cell, work, execution)
                problem = '' if execution['ok'] and not scored['missing_terms'] else (
                    'The script did not run to completion.' if not execution['ok'] else
                    'Missing rows: ' + ', '.join(scored['missing_terms']))
            record['executions'].append({k: execution.get(k) for k in ('ok', 'returncode', 'timed_out', 'wall_seconds')})
            if cell['condition'] == 'self_repair' and problem and round_number < rounds:
                text = repair_text(initial, code or call['text'], execution, problem)
                (record_dir / f'repair_prompt_round{round_number + 1}.txt').write_text(text)
                round_number += 1
                continue
        break
    if record['calls'] and record['calls'][-1]['ok'] and (work / script).exists():
        final = execute(work, cell['lang'], record_dir, 99, info_profile(record_dir), path=lang_path,
                        inputs=task_files)
        record['final_execution'] = {k: final.get(k) for k in ('ok', 'returncode', 'timed_out', 'wall_seconds',
                                                                 'final_files')}
        record['scoring'] = score(cell, work, final)
        record['success'] = bool(record['scoring']['passed'])
        record['literal_answer_check'] = literal_answers((work / script).read_text(errors='replace'), cell['lang'],
                                                         record['scoring'].get('rows') or {})
    else:
        record['scoring'] = dict(passed=False, reason=record['stop_reason'] or f'no final {script}')
        record['success'] = False
    if cell['lang'] == 'stata':
        record['redacted_files'] = redact_workspace(work)
    keep = shutil.ignore_patterns('.tmp', *task_files)
    shutil.copytree(work, record_dir / 'final_workspace', ignore=keep, dirs_exist_ok=True)
    record['completed_at'] = now()
    save_json(record_dir / 'record.json', record)
    return record


def info_profile(record_dir: Path) -> str | None:
    path = Path(record_dir) / 'execution.sb'
    return path.read_text() if path.is_file() else None


def extract_code(text: str) -> str | None:
    match = CODE_FENCE.search(text or '')
    return match.group(1) if match else None


def run_block(block: str, cells: list[dict], run_root: Path, out_dir: Path, *,
              run_check: Callable[[Path], dict] | None = None,
              per_cell: Callable[[dict], dict] | None = None, dry: bool = True, worker: int = 0,
              state_checks: bool | None = None, **cell_options) -> dict:
    """Run the cells of one block between two sets of checks.

    Start and end: environment_check, the binary hashes (required in a real collection), and in a real
    collection the Claude login and state checks and the Codex login check. A change at the start stops the
    block before any cell. The Codex home of the worker is created at the start and removed at the end.
    """
    out_dir, run_root = Path(out_dir), Path(run_root)
    bin_dir = collection_config(run_root).get('bin_dir')
    if not dry and not bin_dir:
        raise BinaryChanged('a real collection needs pinned binaries (prepare_collection bin_dir)')
    state_checks = (not dry) if state_checks is None else state_checks
    start = environment_check(f'{block}_start', out_dir, run_check)
    summary = dict(block=block, started=start, runner=VERSION, cells=[])
    summary['binaries_start'] = check_binaries(bin_dir, f'{block}_start') if bin_dir else 'not pinned (dry)'
    tmp = run_root / 'tools/check_tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    models = {MODELS[c['model']]['cli'] for c in cells}
    if 'codex' in models:
        summary['codex_home_start'] = setup_codex_home(run_root, worker)
    if state_checks:
        claude_bin = provider_bins(run_root)[0]
        if 'claude' in models:
            summary['claude_login_start'] = claude_login_state(claude_bin, tmp)
            summary['claude_state_start'] = claude_state_problems(run_root)
            if not summary['claude_login_start'].get('loggedIn') or not summary['claude_state_start']['ok']:
                raise ConfigurationCheckFailed(f'block {block}: Claude login or state check failed')
        if 'codex' in models:
            summary['codex_login_start'] = codex_login_ok(run_root, worker, tmp)
            if summary['codex_login_start']['returncode'] != 0:
                raise ConfigurationCheckFailed(f'block {block}: codex login status failed')
    fields = {'block': block, 'environment.manifest_sha256': start['environment.manifest_sha256']}
    block_state: dict = {}
    error = None
    try:
        for cell in cells:
            rec = run_cell(cell, run_root, block_fields=fields, dry=dry, worker=worker, block_state=block_state,
                           **{**cell_options, **(per_cell(cell) if per_cell else {})})
            summary['cells'].append(dict(id=rec['id'], success=rec['success'], stop_reason=rec['stop_reason']))
    except (ConfigurationCheckFailed, uw.WatchStopped) as failure:
        error = failure
        summary['stopped'] = repr(failure)
    finally:
        if 'codex' in models:
            summary['codex_home_end'] = teardown_codex_home(run_root, worker)
        if state_checks and 'claude' in models:
            summary['claude_login_end'] = claude_login_state(provider_bins(run_root)[0], tmp)
            summary['claude_state_end'] = claude_state_problems(run_root)
        summary['codex_base_instructions_sha256'] = block_state.get('codex_base_sha256')
        try:
            summary['binaries_end'] = check_binaries(bin_dir, f'{block}_end') if bin_dir else 'not pinned (dry)'
        except BinaryChanged as changed:
            summary['binaries_end'] = dict(matches=False, error=str(changed))
        try:
            summary['ended'] = environment_check(f'{block}_end', out_dir, run_check)
        except EnvironmentChanged as changed:
            summary['ended'] = dict(matches=False, error=str(changed))
        summary['environment.manifest_sha256'] = start['environment.manifest_sha256']
        save_json(out_dir / f'block_{block}.json', summary)
    if error is not None:
        raise error
    if isinstance(summary['binaries_end'], dict) and not summary['binaries_end'].get('matches', True):
        raise BinaryChanged(summary['binaries_end']['error'])
    if not summary['ended'].get('matches'):
        raise EnvironmentChanged(summary['ended']['error'])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='build and resolve the cell plan (no provider call)')
    plan.add_argument('--rep', type=int, action='append', default=None)
    plan.add_argument('--out', type=Path)
    pin = sub.add_parser('pin', help='copy and hash claude and the Codex vendor folder (no provider call)')
    pin.add_argument('--bin-dir', type=Path, required=True)
    check = sub.add_parser('check-bin', help='recompute the pinned hashes')
    check.add_argument('--bin-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'pin':
        manifest = pin_binaries(args.bin_dir)
        print(json.dumps(dict(claude=manifest['claude']['sha256'], codex=manifest['codex']['files']['bin/codex'],
                              codex_files=len(manifest['codex']['files'])), indent=2))
        return 0
    if args.command == 'check-bin':
        print(json.dumps(check_binaries(args.bin_dir, 'command line'), indent=2))
        return 0
    cells = plan_cells(tuple(args.rep or [1]))
    rows = [resolve_cell(c) for c in cells]
    summary = dict(cells=len(cells), resolved=sum(r['ok'] for r in rows),
                   problems=[r for r in rows if not r['ok']][:50])
    if args.out:
        with open(args.out, 'w') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['resolved'] == len(cells) else 1


if __name__ == '__main__':
    sys.exit(main())

"""Tests of the final-bank collection runner (code/benchmark/final_bank_runner.py). No provider call.

The provider in these tests is a local shell script. The usage watcher uses a fake clock and a fake Codex reading.
The Codex login link points at a dummy file in the test folder (autouse fixture), never at ~/.codex/auth.json.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import final_bank_prompts as fp  # noqa: E402
import final_bank_runner as r  # noqa: E402
import usage_watch as uw  # noqa: E402

DARWIN = sys.platform == 'darwin' and Path('/usr/bin/sandbox-exec').exists()
STATA = DARWIN and Path(r.STATA_BIN).exists()
CELL = dict(task='ols_robust', bank='2026-09-28_final_v2', model='sonnet55', lang='python', prompt='zero_shot',
            condition='chatbot', rep=1)
SENTINEL = 'SENTINEL-CREDENTIAL-7f3a9c'
BANNER = ('  Stata license: Single-user  perpetual\n  Serial number: 401406358220\n'
          '    Licensed to: Jane Example\n                 Example University\n')

FAKE = """#!/bin/sh
cat > /dev/null
case "$FAKE_MODE" in
  normal) cat "$FAKE_CANNED" ;;
  seq) n=$(cat "$FAKE_COUNTER" 2>/dev/null || echo 0); n=$((n+1)); echo $n > "$FAKE_COUNTER"
       if [ -f "$FAKE_CANNED.$n" ]; then cat "$FAKE_CANNED.$n"; else cat "$FAKE_CANNED"; fi ;;
  stall) echo '{"type":"system","subtype":"init"}'; exec sleep 1000 ;;
  stallusage) echo '{"type":"system","subtype":"init"}'
    echo '{"type":"assistant","message":{"id":"m1","model":"claude-sonnet-5-5","content":[],"usage":{"input_tokens":7,"output_tokens":11}}}'
    exec sleep 1000 ;;
  runaway) echo '{"type":"system","subtype":"init"}'
    exec "$PYTHON" -c 'import sys, time
while True:
    sys.stdout.write("text without end "); sys.stdout.flush(); time.sleep(0.05)' ;;
  endless) exec "$PYTHON" -c 'import time
while True:
    print("{\\"type\\": \\"x\\"}", flush=True); time.sleep(0.1)' ;;
esac
"""


@pytest.fixture(autouse=True)
def dummy_codex_login(tmp_path, monkeypatch):
    auth = tmp_path / 'dummy_codex/auth.json'
    auth.parent.mkdir()
    auth.write_text(json.dumps({'tokens': {'access_token': SENTINEL, 'refresh_token': SENTINEL}}))
    monkeypatch.setattr(r, 'CODEX_AUTH', auth)
    return auth


@pytest.fixture
def fake(tmp_path):
    path = tmp_path / 'fake_provider.sh'
    path.write_text(FAKE)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def init_event(where, agent=False, **over):
    event = {'type': 'system', 'subtype': 'init', 'claude_code_version': r.CLAUDE_VERSION,
             'model': 'claude-sonnet-5-5', 'cwd': str(where), 'tools': ['Bash'] if agent else [], 'mcp_servers': [],
             'slash_commands': [], 'skills': [], 'plugins': [{'name': 'agents-md', 'source': 'agents-md@builtin'}],
             'agents': ['claude', 'Explore', 'general-purpose', 'Plan'], 'memory_paths': None,
             'output_style': 'default', 'permissionMode': 'default', 'apiKeySource': 'none',
             'analytics_disabled': True, 'product_feedback_disabled': True, 'fast_mode_state': 'off'}
    event.update(over)
    return event


def canned(tmp_path, text, five=0.4, name='canned.jsonl', init=None, model_usage=None):
    events = [init or {'type': 'system', 'subtype': 'init', 'model': 'claude-sonnet-5-5'},
              {'type': 'assistant', 'message': {'id': 'm1', 'model': 'claude-sonnet-5-5',
                                                'content': [{'type': 'text', 'text': text}],
                                                'usage': {'input_tokens': 10, 'output_tokens': 5}}},
              {'type': 'rate_limit_event', 'rate_limit_info': {'status': 'allowed', 'unifiedWindows': {
                  'five_hour': {'utilization': five, 'resetsAt': time.time() + 1200}}}},
              {'type': 'result', 'usage': {'input_tokens': 10, 'output_tokens': 5},
               'modelUsage': model_usage or {'claude-sonnet-5-5': {'inputTokens': 10, 'outputTokens': 5}}}]
    path = tmp_path / name
    path.write_text('\n'.join(json.dumps(e) for e in events) + '\n')
    return path


def rejected(tmp_path, name, reset=None):
    info = {'status': 'rejected', 'rateLimitType': 'five_hour'}
    if reset is not None:
        info['resetsAt'] = reset
    events = [{'type': 'system', 'subtype': 'init', 'model': 'claude-sonnet-5-5'},
              {'type': 'rate_limit_event', 'rate_limit_info': info},
              {'type': 'result', 'is_error': True, 'result': "You've hit your limit"}]
    path = tmp_path / name
    path.write_text('\n'.join(json.dumps(e) for e in events) + '\n')
    return path


def python_solution():
    route = (fp.package_dir('ols_robust') / 'routes/route_python.py').read_text()
    return ('import contextlib, io, json, re\nROUTE = ' + repr(route) + '\nb = io.StringIO()\n'
            'with contextlib.redirect_stdout(b):\n    exec(compile(ROUTE, "r", "exec"), {"__name__": "__main__"})\n'
            'rows = [{"term": m[1], "estimate": float(m[2]), "se": float(m[3]), "n_obs": int(m[4])} for m in '
            're.finditer(r"^ANSWER: (\\S+) estimate=(\\S+) se=(\\S+) n_obs=(\\S+)", b.getvalue(), re.M)]\n'
            'json.dump({"estimates": rows}, open("result.json", "w"))\n')


class FakeClock:
    def __init__(self):
        self.t, self.slept = time.time(), []

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


def fake_watcher(tmp_path, clock):
    watcher = uw.Watcher(config=uw.load_config(), state_dir=tmp_path / 'watch', clock=clock, sleep=clock.sleep,
                         codex_poller=lambda timeout: None)
    watcher.dry_poller = True
    return watcher


def stub_check(sha):
    def run_check(path):
        return {'manifest_sha256': sha, 'failures': []}
    return run_check


EXPECTED = r.environment_fields()['environment.manifest_sha256']


# Plan -----------------------------------------------------------------------------------------

def test_plan_has_1854_distinct_cells_per_repetition():
    # 48 specified tasks with zero_shot and few_shot, 7 decision versions with zero_shot (decision records
    # 2026-09-29_decision-versions, 2026-09-30_hand-audit-fixes, and 2026-09-30_drop-heldout-level).
    cells = r.plan_cells((1,))
    assert len(cells) == 48 * 2 * 3 * 2 * 3 + 7 * 2 * 3 * 1 * 3 == 1854
    assert len({r.cell_id(c) for c in cells}) == 1854
    assert {c['prompt'] for c in cells} == {'zero_shot', 'few_shot'}
    decision = [c for c in cells if c['task'].endswith('__decision')]
    assert len(decision) == 126 and {c['prompt'] for c in decision} == {'zero_shot'}
    assert {c['bank'] for c in decision} == {'2026-09-29_decision'}
    assert {c['model'] for c in cells} == {'sonnet55', 'luna'}
    assert r.MODELS['sonnet55']['model_id'] == 'claude-sonnet-5-5' and r.MODELS['luna']['model_id'] == 'gpt-5.6-luna'


def test_every_cell_resolves():
    problems = [row for row in (r.resolve_cell(c) for c in r.plan_cells((1,))) if not row['ok']]
    assert problems == []


def test_resolve_reports_a_missing_prompt(monkeypatch):
    real = fp.prompt_name
    monkeypatch.setattr(fp, 'prompt_name', lambda *a: 'missing__' + real(*a))
    row = r.resolve_cell(CELL)
    assert not row['ok'] and any('prompt missing' in p for p in row['problems'])


# Harmonized configuration ----------------------------------------------------------------------

def test_both_models_run_at_effort_medium_read_from_models(monkeypatch):
    assert r.MODELS['sonnet55']['effort'] == r.MODELS['luna']['effort'] == 'medium'
    claude = r.claude_argv(False, Path('/x/s.json'))
    assert claude[claude.index('--effort') + 1] == 'medium'
    assert 'model_reasoning_effort="medium"' in r.codex_argv(False, Path('/w'), Path('/l'))
    monkeypatch.setitem(r.MODELS['sonnet55'], 'effort', 'high')
    monkeypatch.setitem(r.MODELS['luna'], 'effort', 'high')
    claude = r.claude_argv(False, Path('/x/s.json'))
    assert claude[claude.index('--effort') + 1] == 'high'
    assert 'model_reasoning_effort="high"' in r.codex_argv(False, Path('/w'), Path('/l'))


CODEX_SETTINGS = ['web_search="disabled"', 'approval_policy="never"', 'project_doc_max_bytes=0', 'mcp_servers={}',
                  'skills.include_instructions=false', 'skills.bundled.enabled=false',
                  'include_permissions_instructions=false', 'default_permissions="bench"',
                  'features.goals=false', 'features.view_image=false', 'features.sleep_tool=false',
                  'features.shell_snapshot=false', 'features.daemon_auto_start=false', 'features.fast_mode=false',
                  'features.unbounded_connection_retries=false', 'features.code_mode=false',
                  'tools.experimental_request_user_input={enabled=false}', 'allow_login_shell=false',
                  'analytics.enabled=false', 'feedback.enabled=false', 'otel.exporter="none"',
                  'otel.metrics_exporter="none"', 'otel.trace_exporter="none"',
                  'check_for_update_on_startup=false', 'forced_login_method="chatgpt"', 'history.persistence="none"']
CLAUDE_FLAGS = ['-p', '--no-session-persistence', '--safe-mode', '--strict-mcp-config', '--disable-slash-commands',
                '--verbose']


@pytest.mark.parametrize('model', ['sonnet55', 'luna'])
@pytest.mark.parametrize('condition', ['chatbot', 'self_repair', 'agent'])
def test_argv_and_environment_per_model_and_condition(tmp_path, model, condition):
    root = tmp_path / 'root'
    r.prepare_collection(root, seed_mpl=False)
    cell = dict(CELL, model=model, condition=condition)
    info = r.prepare_run(cell, root)
    argv, env = info['provider_argv'], info['provider_env']
    agent = condition == 'agent'
    # Base environment, the same for both models.
    for key, value in {'TZ': 'UTC', 'LANG': 'en_US.UTF-8', 'LC_ALL': 'en_US.UTF-8', 'TERM': 'dumb',
                       'NO_COLOR': '1', 'HOME': str(Path.home())}.items():
        assert env[key] == value, key
    assert env['PATH'].split(os.pathsep)[0] == str(r.path_dir(root.resolve(), 'python'))
    for key in r.FORBIDDEN_ENV:
        assert key not in env
    assert not any(k.startswith('ANTHROPIC_') for k in env)
    if model == 'luna':
        assert env['CODEX_HOME'] == str(r.codex_home(root.resolve(), 0))
        assert not any(k.startswith('CLAUDE') for k in env)
        text = ' '.join(argv)
        for setting in CODEX_SETTINGS:
            assert setting in argv, setting
        assert f'features.shell_tool={"true" if agent else "false"}' in argv
        assert ('"."="write"' if agent else '"."="read"') in text
        assert '"network"={"enabled"=false}' in text and '--ignore-user-config' in argv and '--ignore-rules' in argv
        assert argv[argv.index('-m') + 1] == 'gpt-5.6-luna'
        assert argv[argv.index('-C') + 1] == (info['work'] if agent else str(Path(info['work']).parent / 'generate_1'))
    else:
        for key, value in r.CLAUDE_ENV.items():
            assert env[key] == value, key
        assert env['CLAUDE_CODE_TMPDIR'] == info['tmp']
        assert ('CLAUDE_CODE_SHELL_PREFIX' in env) == agent
        assert 'CODEX_HOME' not in env
        for flag in CLAUDE_FLAGS:
            assert flag in argv, flag
        assert argv[argv.index('--model') + 1] == 'claude-sonnet-5-5'
        assert argv[argv.index('--permission-mode') + 1] == 'manual'
        assert argv[argv.index('--permission-prompts') + 1] == 'none'
        assert argv[argv.index('--setting-sources') + 1] == ''
        assert argv[argv.index('--tools') + 1] == ('Bash' if agent else '')
        assert ('--allowedTools' in argv) == agent
    # The cell identity does not appear in the cwd of the provider.
    assert r.cell_id(cell) not in info['work'] and Path(info['work']).parent.name.startswith('c')


def test_only_workspace_and_tool_permissions_differ_between_conditions(tmp_path):
    for model in ('sonnet55', 'luna'):
        chat = r.real_argv(dict(CELL, model=model), tmp_path, Path('/cwd'), tmp_path, 1)
        agent = r.real_argv(dict(CELL, model=model, condition='agent'), tmp_path, Path('/cwd'), tmp_path, 1)
        differ = [(a, b) for a, b in zip(chat, agent) if a != b]
        if model == 'sonnet55':
            assert differ == [('', 'Bash')] and agent[len(chat):] == ['--allowedTools', 'Bash']
        else:
            assert len(chat) == len(agent) and len(differ) == 2
            assert any('shell_tool' in a for a, _ in differ) and any('"write"' in b for _, b in differ)


def test_claude_settings_have_no_hook_and_no_bash_deny(tmp_path):
    info = r.prepare_collection(tmp_path / 'root', seed_mpl=False)
    settings = json.loads(Path(info['claude_settings']).read_text())
    assert settings == r.CLAUDE_SETTINGS
    assert 'hooks' not in settings and settings['permissions']['deny'] == ['WebFetch', 'WebSearch']
    assert not (Path(info['run_root']) / 'tools/deny_install_hook.sh').exists()
    assert info['homes'] == [] and info['credentials_present'] == [] and info['mode'] == '0o700'


def test_run_root_inside_git_is_refused(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    with pytest.raises(r.ConfigurationCheckFailed):
        r.prepare_collection(repo / 'collection', seed_mpl=False)


def test_clean_env_refuses_credential_variables(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', SENTINEL)
    monkeypatch.setenv('CLAUDE_CODE_OAUTH_TOKEN', SENTINEL)
    env = r.clean_env(tmp_path)
    assert SENTINEL not in json.dumps(env) and env['TZ'] == 'UTC'
    for key in ('CLAUDE_CONFIG_DIR', 'ANTHROPIC_API_KEY', 'CODEX_API_KEY'):
        with pytest.raises(r.ConfigurationCheckFailed):
            r.clean_env(tmp_path, {key: 'x'})


def test_clean_env_has_the_environment_findings(tmp_path):
    env = r.clean_env(tmp_path, path=r.language_path(tmp_path, 'r'))
    assert env['R_LIBS_USER'] == str(r.R_USER_LIB)
    for key in ('MPLCONFIGDIR', 'R_USER_CONFIG_DIR', 'R_USER_DATA_DIR', 'R_USER_CACHE_DIR', 'TMPPREFIX', 'ZDOTDIR'):
        assert env[key].startswith(str(tmp_path)), key
    assert str(r.VENV) not in env['PATH'] and '/usr/local/bin' not in env['PATH']


# Cross-language tools -------------------------------------------------------------------------

def _which(root, lang, command):
    env = r.clean_env(root / 'tmp', path=r.language_path(root, lang))
    return subprocess.run(['/bin/sh', '-c', command], env=env, capture_output=True, text=True, timeout=120)


def test_each_language_sees_only_its_interpreter_on_path(tmp_path):
    root = tmp_path / 'root'
    r.prepare_collection(root, seed_mpl=False)
    root = root.resolve()
    (root / 'tmp').mkdir()
    py = _which(root, 'python', 'python3 -c "import numpy; print(42)"')
    assert py.returncode == 0 and py.stdout.strip() == '42'
    for lang in ('r', 'stata'):
        for name in ('python3', 'python', 'pip', 'uv'):
            out = _which(root, lang, f'{name} --version')
            assert out.returncode == 127 and 'command not found' in out.stderr, (lang, name)
    for name in ('Rscript', 'R', 'stata-se', 'stata'):
        assert _which(root, 'python', f'{name} --version').returncode == 127, name
    assert _which(root, 'stata', 'Rscript --version').returncode == 127
    assert _which(root, 'r', 'stata-se -h').returncode == 127
    if Path(r.RSCRIPT).exists():
        out = _which(root, 'r', 'Rscript -e "cat(6*7)"')
        assert out.returncode == 0 and out.stdout.strip() == '42'


@pytest.mark.skipif(not DARWIN, reason='Seatbelt is macOS only')
def test_r_and_stata_profiles_deny_the_python_venv(tmp_path):
    work = tmp_path / 'work'
    (work / '.tmp').mkdir(parents=True)
    target = r.VENV / 'pyvenv.cfg'
    for lang, allowed in (('python', True), ('r', False), ('stata', False)):
        profile = tmp_path / f'{lang}.sb'
        profile.write_text(r.execution_profile(work, work / '.tmp', lang))
        out = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(profile), '/bin/cat', str(target)],
                             capture_output=True, text=True, timeout=60)
        assert (out.returncode == 0) == allowed, lang


OWN_COMMAND = {'python': [str(r.VENV / 'bin/python'), '-c', 'import numpy; print(42)'],
               'r': [r.RSCRIPT, '-e', 'cat(6*7)'],
               'stata': [r.STATA_BIN, '-q', '-b', 'do', 'own.do']}
OTHER_COMMANDS = {
    'python': [str(r.VENV / 'bin/python') + ' --version', '/opt/homebrew/bin/python3 --version',
               '/usr/bin/python3 --version', '/opt/homebrew/bin/uv --version'],
    'r': [r.RSCRIPT + ' --version', '/Library/Frameworks/R.framework/Resources/bin/Rscript --version',
          '/Library/Frameworks/R.framework/Resources/bin/exec/R --version'],
    'stata': [r.STATA_BIN + ' -q -b do own.do'],
}


def _sandboxed(profile: Path, work: Path, argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(['/usr/bin/sandbox-exec', '-f', str(profile), *argv], cwd=work,
                          env=r.clean_env(work / '.tmp'), capture_output=True, text=True, timeout=180)


@pytest.mark.skipif(not DARWIN, reason='Seatbelt is macOS only')
def test_each_profile_denies_the_other_interpreters_by_absolute_path(tmp_path):
    """Decision record 2026-09-30_interpreter-exec-denial: hand audit runs n 101 and n 174 (Luna Stata agents)."""
    work = (tmp_path / 'work').resolve()
    (work / '.tmp').mkdir(parents=True)
    (work / 'own.do').write_text('display 6*7\n')
    for lang in r.LANGS:
        profile = tmp_path / f'{lang}.sb'
        profile.write_text(r.execution_profile(work, work / '.tmp', lang))
        if Path(OWN_COMMAND[lang][0]).exists():
            own = _sandboxed(profile, work, OWN_COMMAND[lang])
            assert own.returncode == 0, (lang, own.stderr[-500:])
            if lang == 'stata':
                assert '42' in (work / 'own.log').read_text()
            else:
                assert own.stdout.strip() == '42', lang
        for other, commands in OTHER_COMMANDS.items():
            if other == lang:
                continue
            for command in commands:
                if not Path(command.split()[0]).exists():
                    continue
                out = _sandboxed(profile, work, ['/bin/zsh', '-c', command])
                assert out.returncode != 0 and 'operation not permitted' in out.stderr, (lang, command, out.stderr)


@pytest.mark.skipif(not DARWIN, reason='Seatbelt is macOS only')
def test_a_copied_python_binary_does_not_run_in_r_or_stata_profiles(tmp_path):
    work = (tmp_path / 'work').resolve()
    (work / '.tmp').mkdir(parents=True)
    binary = Path(os.path.realpath(r.VENV / 'bin/python'))
    shutil.copyfile(binary, work / 'renamed')
    os.chmod(work / 'renamed', 0o755)
    for lang in ('r', 'stata'):
        profile = tmp_path / f'{lang}.sb'
        profile.write_text(r.execution_profile(work, work / '.tmp', lang))
        out = _sandboxed(profile, work, [str(work / 'renamed'), '-c', 'print(42)'])
        assert out.returncode != 0 and out.stdout.strip() != '42', lang


@pytest.mark.skipif(not DARWIN, reason='Seatbelt is macOS only')
def test_path_stubs_still_answer_command_not_found_under_the_profile(tmp_path):
    root = tmp_path / 'root'
    r.prepare_collection(root, seed_mpl=False)
    info = r.prepare_run(dict(CELL, lang='stata', condition='agent'), root)
    work = Path(info['work'])
    env = r.clean_env(Path(info['tmp']), path=r.language_path(root.resolve(), 'stata'))
    profile = Path(info['record']) / 'execution.sb'
    out = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(profile), '/bin/zsh', '-c', 'python3 --version'],
                         cwd=work, env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 127 and 'command not found' in out.stderr
    out = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(profile), '/bin/zsh', '-c',
                          '/opt/homebrew/bin/python3 --version'], cwd=work, env=env, capture_output=True, text=True,
                         timeout=60)
    if Path('/opt/homebrew/bin/python3').exists():
        assert out.returncode != 0 and 'operation not permitted' in out.stderr


def test_interpreter_rules_name_only_the_other_languages():
    for lang in r.LANGS:
        text = '\n'.join(r.interpreter_rules(lang))
        for other in r.LANGS:
            for tree in r.INTERPRETER_TREES[other]:
                rule = f'(deny process-exec* (subpath "{tree}"))'
                assert (rule in text) == (other != lang), (lang, other, tree)
    assert r.interpreter_rules(None) == []


def test_codex_profile_denies_the_other_interpreters(tmp_path):
    for lang in r.LANGS:
        text = ' '.join(r.codex_argv(True, Path('/w'), Path('/l'), lang, tmp_path))
        for other in r.LANGS:
            for path in [*r.INTERPRETER_TREES[other], *r.INTERPRETER_FILES[other]]:
                if not Path(path).exists():
                    continue
                assert (f'"{path}"="deny"' in text) == (other != lang), (lang, path)
                if other != lang:
                    assert f'"{path}"="read"' not in text, (lang, path)


def test_codex_profile_reads_the_venv_only_for_python(tmp_path):
    for lang in ('r', 'stata'):
        text = ' '.join(r.codex_argv(True, Path('/w'), Path('/l'), lang, tmp_path))
        assert f'"{r.VENV}"="read"' not in text and f'"{r.path_dir(tmp_path, lang)}"="read"' in text
    assert f'"{r.VENV}"="read"' in ' '.join(r.codex_argv(True, Path('/w'), Path('/l'), 'python', tmp_path))


def test_execution_profile_denies_network_and_outside_writes(tmp_path):
    text = r.execution_profile(tmp_path / 'work', tmp_path / 'work/.tmp')
    assert '(deny network-outbound (remote ip))' in text
    assert '(deny file-write* (subpath "/"))' in text
    assert text.index('(deny file-write* (subpath "/"))') < text.index(f'(allow file-read* file-write* (subpath "{tmp_path}/work"))')


@pytest.mark.skipif(not DARWIN, reason='Seatbelt is macOS only')
def test_tool_wrapper_confines_the_command_and_sets_c_locale(tmp_path):
    root = tmp_path / 'root'
    r.prepare_collection(root, seed_mpl=False)
    info = r.prepare_run(dict(CELL, condition='agent'), root)
    env = r.clean_env(Path(info['tmp']), path=r.language_path(root.resolve(), 'python'))

    def run(command):
        line = f"'{info['wrapper']}' '{command}'"
        return subprocess.run(['/bin/zsh', '-c', line], cwd=info['work'], env=env, capture_output=True, text=True,
                              timeout=120)
    assert run('echo ok > inside.txt && cat inside.txt').stdout.strip() == 'ok'
    assert run('echo $LC_ALL').stdout.strip() == 'C.UTF-8'
    outside = tmp_path / 'outside.txt'
    assert run(f'echo no > {outside}').returncode != 0 and not outside.exists()
    assert run(f'touch {r.VENV}/pyvenv.cfg').returncode != 0


# Credentials ----------------------------------------------------------------------------------

def test_codex_home_links_the_login_and_removes_only_the_link(tmp_path, dummy_codex_login):
    root = tmp_path / 'root'
    r.prepare_collection(root, seed_mpl=False)
    before = dummy_codex_login.read_bytes()
    row = r.setup_codex_home(root, 0)
    link = r.codex_home(root, 0) / 'auth.json'
    assert row['link'] and link.is_symlink() and os.readlink(link) == str(dummy_codex_login)
    assert oct(os.stat(r.codex_home(root, 0)).st_mode & 0o777) == '0o700'
    end = r.teardown_codex_home(root, 0)
    assert end == dict(home=str(r.codex_home(root, 0)), unlinked=True, removed=True, target_is_file=True)
    assert dummy_codex_login.read_bytes() == before


def test_stale_codex_home_is_removed_at_collection_start(tmp_path, dummy_codex_login):
    root = tmp_path / 'root'
    r.prepare_collection(root, seed_mpl=False)
    r.setup_codex_home(root, 3)
    info = r.prepare_collection(root, seed_mpl=False)
    assert info['stale_homes_removed'][0]['unlinked'] and not r.codex_home(root, 3).exists()
    assert dummy_codex_login.is_file()


def test_a_copied_login_file_is_refused(tmp_path):
    root = tmp_path / 'root'
    home = r.codex_home(root, 0)
    home.mkdir(parents=True)
    (home / 'auth.json').write_text('{}')
    with pytest.raises(r.ConfigurationCheckFailed):
        r.teardown_codex_home(root, 0)


@pytest.mark.skipif(not DARWIN, reason='Seatbelt is macOS only')
def test_no_credential_content_appears_in_any_record(tmp_path, fake, dummy_codex_login, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', SENTINEL)
    monkeypatch.setenv('OPENAI_API_KEY', SENTINEL)
    root, out = tmp_path / 'root', tmp_path / 'out'
    r.prepare_collection(root, seed_mpl=False)
    stream = canned(tmp_path, '```python\n' + python_solution() + '```')
    cells = [CELL, dict(CELL, model='luna')]
    options = dict(provider=lambda *a: [str(fake)], provider_env_extra={'FAKE_MODE': 'normal',
                                                                        'FAKE_CANNED': str(stream)})
    block = r.run_block('cred', cells, root, out, run_check=stub_check(EXPECTED), **options)
    assert block['codex_home_start']['link'] and block['codex_home_end']['unlinked']
    scanned = 0
    for folder in (root, out):
        for path in folder.rglob('*'):
            if path.is_file() and not path.is_symlink():
                assert SENTINEL not in path.read_text(errors='replace'), path
                scanned += 1
    assert scanned > 20
    assert not any(p.is_symlink() for p in root.rglob('*'))
    assert SENTINEL in dummy_codex_login.read_text()


# Binary pins ----------------------------------------------------------------------------------

def fake_sources(tmp_path):
    claude = tmp_path / 'src/claude'
    vendor = tmp_path / 'src/vendor'
    (vendor / 'bin').mkdir(parents=True)
    (vendor / 'codex-path').mkdir()
    claude.write_text('#!/bin/sh\necho 2.1.284\n')
    (vendor / 'bin/codex').write_text('#!/bin/sh\necho codex-cli 0.158.0\n')
    (vendor / 'codex-path/rg').write_text('rg')
    os.symlink('rg', vendor / 'codex-path/rg-link')
    for path in (claude, vendor / 'bin/codex'):
        path.chmod(0o755)
    return claude, vendor, {'claude': r.sha(claude), 'codex': r.sha(vendor / 'bin/codex')}


def test_pin_binaries_copies_hashes_and_detects_a_change(tmp_path):
    claude, vendor, expected = fake_sources(tmp_path)
    bin_dir = tmp_path / 'bin'
    manifest = r.pin_binaries(bin_dir, claude, vendor, expected)
    assert manifest['claude']['sha256'] == expected['claude'] and manifest['claude']['version_output'] == '2.1.284'
    assert manifest['codex']['files']['bin/codex'] == expected['codex']
    assert manifest['codex']['files']['codex-path/rg-link'] == 'link:rg'
    assert r.check_binaries(bin_dir, 't_start')['matches']
    (bin_dir / r.CODEX_PIN_NAME / 'codex-path/rg').write_text('changed')
    with pytest.raises(r.BinaryChanged):
        r.check_binaries(bin_dir, 't_end')


def test_pin_binaries_refuses_a_source_with_another_hash(tmp_path):
    claude, vendor, expected = fake_sources(tmp_path)
    with pytest.raises(r.BinaryChanged):
        r.pin_binaries(tmp_path / 'bin', claude, vendor, dict(expected, claude='0' * 64))


def test_argv_uses_the_pinned_copies(tmp_path):
    claude, vendor, expected = fake_sources(tmp_path)
    bin_dir = tmp_path / 'bin'
    r.pin_binaries(bin_dir, claude, vendor, expected, versions=False)
    root = tmp_path / 'root'
    r.prepare_collection(root, seed_mpl=False, bin_dir=bin_dir)
    sonnet = r.real_argv(CELL, root, Path('/c'), tmp_path, 1)
    luna = r.real_argv(dict(CELL, model='luna'), root, Path('/c'), tmp_path, 1)
    assert sonnet[0] == str(bin_dir.resolve() / r.CLAUDE_PIN_NAME)
    assert luna[0] == str(bin_dir.resolve() / r.CODEX_PIN_NAME / 'bin/codex')
    assert f'"{bin_dir.resolve() / r.CODEX_PIN_NAME}"="read"' in ' '.join(luna)
    for argv in (sonnet, luna):
        with pytest.raises(r.DryRunViolation):
            r.check_dry_argv(argv)


def test_real_block_needs_pinned_binaries(tmp_path):
    root = tmp_path / 'root'
    r.prepare_collection(root, seed_mpl=False)
    with pytest.raises(r.BinaryChanged):
        r.run_block('b', [CELL], root, tmp_path, run_check=stub_check(EXPECTED), dry=False,
                    provider=lambda *a: ['/bin/true'])


def test_dry_mode_refuses_provider_binaries(tmp_path):
    for argv in ([str(r.CLAUDE_BIN)], ['claude', '-p'], [str(r.CODEX_BIN), 'exec'], ['/opt/homebrew/bin/codex'],
                 ['/x/claude-2.1.284'], ['/x/codex-0.158.0/bin/codex']):
        with pytest.raises(r.DryRunViolation):
            r.call_provider(argv, {}, tmp_path, '', tmp_path, 1, agent=False, dry=True)


# Provider call, stall rule, stream records -----------------------------------------------------

def call(fake, tmp_path, mode, agent=False, **kw):
    record = tmp_path / 'record'
    record.mkdir(exist_ok=True)
    env = r.clean_env(tmp_path, {'FAKE_MODE': mode, 'PYTHON': str(r.VENV / 'bin/python'), **kw.pop('env', {})})
    return r.call_provider([str(fake)], env, tmp_path, 'prompt', record, 1, agent=agent, dry=True, **kw)


def test_normal_call_is_recorded_with_model_usage_and_receive_times(fake, tmp_path):
    result = call(fake, tmp_path, 'normal', env={'FAKE_CANNED': str(canned(tmp_path, '```python\nprint(1)\n```'))})
    assert result['ok'] and result['init_event'] and result['result_event']
    assert r.extract_code(result['text']) == 'print(1)\n'
    assert (tmp_path / 'record/provider_attempt1.stdout').read_text().count('\n') == 4
    saved = json.loads((tmp_path / 'record/call_attempt1.json').read_text())
    assert saved['usage'] == {'input_tokens': 10, 'output_tokens': 5}
    assert saved['model_usage'] == {'claude-sonnet-5-5': {'inputTokens': 10, 'outputTokens': 5}}
    times = json.loads((tmp_path / 'record/provider_attempt1.times.json').read_text())
    assert [row[0] for row in times] == [1, 2, 3, 4] and all(row[1] >= 0 for row in times)


def test_stall_rule_cuts_an_agent_call_as_a_stream_hang(fake, tmp_path):
    result = call(fake, tmp_path, 'stall', agent=True, stall_idle=1.0, monitor=0.2, hard_limit=30)
    assert result['stalled'] and not result['ok'] and result['returncode'] is None
    assert result['stall_diag']['kind'] == 'stream_hang'
    assert result['wall_seconds'] < 15


def test_generation_call_ignores_the_idle_rule_and_runs_to_its_hard_limit(fake, tmp_path):
    result = call(fake, tmp_path, 'stall', agent=False, stall_idle=0.5, monitor=0.2, hard_limit=3)
    assert result['hard_timeout'] and not result['stalled'] and result['wall_seconds'] >= 3
    assert result['stall_idle_seconds'] is None


def test_runaway_output_is_cut_as_a_runaway_generation(fake, tmp_path):
    result = call(fake, tmp_path, 'runaway', agent=True, stall_idle=1.0, monitor=0.2, hard_limit=30)
    assert result['stalled'] and result['stall_diag']['kind'] == 'runaway_generation'
    first, second = result['stall_diag']['stdout_bytes']
    assert second > first


def test_endless_events_stop_at_the_hard_limit(fake, tmp_path):
    result = call(fake, tmp_path, 'endless', agent=True, stall_idle=1.0, monitor=0.2, hard_limit=2)
    assert result['hard_timeout'] and not result['stalled'] and result['events'] > 5


def test_a_cut_call_keeps_its_partial_usage(fake, tmp_path):
    result = call(fake, tmp_path, 'stallusage', agent=True, stall_idle=1.0, monitor=0.2, hard_limit=30)
    assert result['stalled'] and result['partial_usage'] == {'input_tokens': 7, 'output_tokens': 11}


def test_dry_mode_refuses_a_watcher_with_the_live_codex_reader(tmp_path, fake):
    live = uw.Watcher(config=uw.load_config(), state_dir=tmp_path / 'w')
    with pytest.raises(r.DryRunViolation):
        r.run_cell(CELL, tmp_path / 'root', provider=lambda *a: [str(fake)], dry=True, watcher=live)


def test_codex_stream_errors_are_recorded():
    stream = '\n'.join(json.dumps(e) for e in [
        {'type': 'thread.started'}, {'type': 'item.completed', 'item': {'type': 'error', 'message': 'bad config'}},
        {'type': 'turn.failed', 'error': {'message': "You've hit your usage limit"}}])
    parsed = r.parse_stream(stream)
    assert parsed['init_event'] and 'bad config' in parsed['stream_errors'][0] and parsed['limit_rejected']


# Per-call checks ------------------------------------------------------------------------------

def test_init_check_passes_the_harmonized_init_event():
    for agent in (False, True):
        assert r.claude_init_problems(init_event('/c', agent), Path('/c'), agent) == []


@pytest.mark.parametrize('over', [
    dict(skills=['econ-write']), dict(slash_commands=['commit']), dict(mcp_servers=[{'name': 'stata'}]),
    dict(plugins=[{'source': 'agents-md@builtin'}, {'source': 'codex@openai'}]), dict(memory_paths=['/m']),
    dict(model='claude-opus-4-8'), dict(tools=['Bash', 'Read']), dict(cwd='/elsewhere'),
    dict(agents=['claude', 'my-reviewer']), dict(claude_code_version='2.1.290'), dict(apiKeySource='ANTHROPIC_API_KEY'),
])
def test_init_check_fails_on_any_user_customization(over):
    assert r.claude_init_problems(init_event('/c', **over), Path('/c'), False) != []


def test_model_check_flags_opus_and_fable_as_expensive():
    assert r.model_problems({'models': ['claude-sonnet-5-5', 'claude-haiku-4-5']}, 'sonnet55') == ([], False)
    problems, expensive = r.model_problems({'models': ['claude-sonnet-5-5', 'claude-opus-5-5']}, 'sonnet55')
    assert problems and expensive
    assert r.model_problems({'models': ['fable-5-1']}, 'luna')[1]
    assert r.model_problems({'models': ['gpt-5.6-luna']}, 'luna') == ([], False)


@pytest.mark.skipif(not DARWIN, reason='Seatbelt is macOS only')
def test_an_expensive_model_stops_the_block(tmp_path, fake):
    root = tmp_path / 'root'
    r.prepare_collection(root, seed_mpl=False)
    stream = canned(tmp_path, '```python\nprint(1)\n```', model_usage={'claude-opus-5-5': {'outputTokens': 1}})
    with pytest.raises(r.ModelMismatch):
        r.run_block('m', [CELL], root, tmp_path, run_check=stub_check(EXPECTED), provider=lambda *a: [str(fake)],
                    provider_env_extra={'FAKE_MODE': 'normal', 'FAKE_CANNED': str(stream)})
    assert 'ModelMismatch' in json.loads((tmp_path / 'block_m.json').read_text())['stopped']


@pytest.mark.skipif(not DARWIN, reason='Seatbelt is macOS only')
@pytest.mark.parametrize('other', ['claude-sonnet-4-6', 'claude-sonnet-5'])
def test_any_other_model_id_stops_the_run(tmp_path, fake, other):
    """Decision record 2026-09-30_sonnet-5-5: the collection model is claude-sonnet-5-5 only."""
    root = tmp_path / 'root'
    r.prepare_collection(root, seed_mpl=False)
    cwd = r.prepare_run(CELL, root)
    stream = canned(tmp_path, '```python\nprint(1)\n```', init=init_event(Path(cwd['work']).parent / 'generate_1'),
                    model_usage={other: {'outputTokens': 1}})
    with pytest.raises(r.ModelMismatch):
        r.run_cell(CELL, root, provider=lambda *a: [str(fake)], enforce_checks=True,
                   provider_env_extra={'FAKE_MODE': 'normal', 'FAKE_CANNED': str(stream)})
    record = json.loads((root / 'records' / r.cell_id(CELL) / 'record.json').read_text())
    assert record['stop_reason'] == 'configuration_check'


def test_the_claude_model_key_names_the_version():
    assert r.CLAUDE_MODEL == 'sonnet55' and set(r.MODELS) == {'sonnet55', 'luna'}
    assert r.MODELS['sonnet55'] == dict(cli='claude', model_id='claude-sonnet-5-5', effort='medium')
    assert r.claude_init_problems(init_event('/c'), Path('/c'), False) == []
    problems = r.claude_init_problems(init_event('/c', model='claude-sonnet-4-6'), Path('/c'), False)
    assert problems == ["model='claude-sonnet-4-6'"]


@pytest.mark.skipif(not DARWIN, reason='Seatbelt is macOS only')
def test_enforced_init_check_stops_the_run_before_scoring(tmp_path, fake):
    root = tmp_path / 'root'
    r.prepare_collection(root, seed_mpl=False)
    cwd = r.prepare_run(CELL, root)
    bad = canned(tmp_path, 'x', init=init_event(Path(cwd['work']).parent / 'generate_1', skills=['econ-write']))
    with pytest.raises(r.ConfigurationCheckFailed):
        r.run_cell(CELL, root, provider=lambda *a: [str(fake)], enforce_checks=True,
                   provider_env_extra={'FAKE_MODE': 'normal', 'FAKE_CANNED': str(bad)})
    record = json.loads((root / 'records' / r.cell_id(CELL) / 'record.json').read_text())
    assert record['stop_reason'] == 'configuration_check' and record['success'] is None
    assert (root / 'records' / r.cell_id(CELL) / 'init_attempt1.json').is_file()
    good = canned(tmp_path, '```python\n' + python_solution() + '```', name='good.jsonl',
                  init=init_event(Path(cwd['work']).parent / 'generate_1'))
    record = r.run_cell(CELL, root, provider=lambda *a: [str(fake)], enforce_checks=True,
                        provider_env_extra={'FAKE_MODE': 'normal', 'FAKE_CANNED': str(good)})
    assert record['success'] and record['config_problems'] == []


def test_rollout_is_copied_without_git_fields_and_checked(tmp_path):
    root = tmp_path / 'root'
    sessions = r.codex_home(root, 0) / 'sessions/2026/09/29'
    sessions.mkdir(parents=True)
    events = [{'type': 'session_meta', 'payload': {'cli_version': '0.158.0', 'git': {'branch': 'main'},
                                                   'base_instructions': {'text': 'BASE'}}},
              {'type': 'turn_context', 'payload': {'model': 'gpt-5.6-luna', 'effort': 'medium'}},
              {'type': 'event_msg', 'payload': {'type': 'token_count',
                                                'info': {'total_token_usage': {'output_tokens': 9}}}}]
    (sessions / 'rollout-x.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\n')
    record = tmp_path / 'rec'
    record.mkdir()
    summary = r.collect_rollout(root, 0, record, 2)
    assert summary['files'] == 1 and not (sessions / 'rollout-x.jsonl').exists()
    copied = (record / 'rollout_attempt2.jsonl').read_text()
    assert '"git"' not in copied and 'BASE' in copied
    assert summary['base_instructions_sha256'] == r.text_sha('BASE')
    assert summary['last_token_usage'] == {'output_tokens': 9}
    assert r.rollout_problems(summary, r.text_sha('BASE')) == []
    assert r.rollout_problems(dict(summary, efforts=['low'])) == ['rollout effort [\'low\']']
    assert r.rollout_problems(summary, 'other') != []
    assert r.rollout_problems(dict(summary, files=0)) == ['no Codex rollout was written']


# Execution feedback, Stata redaction, file channel ---------------------------------------------

def test_self_repair_feedback_has_fixed_fields_only():
    execution = dict(ok=False, returncode=1, timed_out=False, stdout='o' * 3000, stderr='e' * 3000,
                     argv=['/usr/bin/sandbox-exec', '-f', '/secret/profile.sb'], env={'HOME': '/Users/x'},
                     profile_sha256='abc123', wall_seconds=1.0, log_tail=BANNER + 'r(198);\n')
    text = r.repair_text('PROMPT', 'code', execution, 'Missing rows: x')
    assert text.startswith('PROMPT\n\nThe previous script')
    feedback = json.loads(text.split('Execution feedback:\n')[1].split('\n')[0])
    assert list(feedback) == ['returncode', 'timed_out', 'stdout_tail', 'stderr_tail', 'log_tail']
    assert len(feedback['stdout_tail']) == 1500 and 'e' * 1501 not in text
    for leaked in ('sandbox-exec', 'profile', '/Users/x', 'abc123', 'Serial number', 'Licensed to', '401406358220'):
        assert leaked not in text, leaked
    assert 'r(198);' in feedback['log_tail']
    python = json.loads(r.repair_text('P', 'c', dict(returncode=None, timed_out=True), 'x')
                        .split('Execution feedback:\n')[1].split('\n')[0])
    assert python == dict(returncode=None, timed_out=True, stdout_tail='', stderr_tail='')


def test_licence_lines_are_removed_from_streams_and_workspaces(tmp_path):
    event = json.dumps({'type': 'user', 'message': {'content': [{'type': 'tool_result', 'content': 'x\n' + BANNER + 'ok'}]}})
    clean = r.redact_stream(event + '\n' + BANNER)
    assert '401406358220' not in clean and 'Jane Example' not in clean and 'Example University' not in clean
    assert json.loads(clean.splitlines()[0])['message']['content'][0]['content'].endswith('ok')
    (tmp_path / 'other.log').write_text('head\n' + BANNER + 'tail\n')
    (tmp_path / 'data.csv').write_text('Serial number: 1\n')
    assert r.redact_workspace(tmp_path) == ['other.log']
    assert (tmp_path / 'other.log').read_text() == 'head\ntail\n'


def test_written_files_follow_the_output_channel_rule(tmp_path):
    for name in ('output.txt', 'run.do', 'input.csv', 'verification.txt', 'old.txt', 'answers.log'):
        (tmp_path / name).write_text('x')
    start = time.time()
    os.utime(tmp_path / 'old.txt', (start - 100, start - 100))
    assert r.written_files(tmp_path, start - 1, start + 1, {'input.csv'}) == ['answers.log', 'output.txt']


def stata_do(values, to_file):
    lines = ['version 19', 'set linesize 255', 'local a "ANS"']
    if to_file:
        lines += ['file open fh using "output.txt", write replace']
    for term, row in values.items():
        body = f'"`a\'WER: {term} estimate=" %24.16g ({row["estimate"]!r}) " se=" %24.16g ({row["se"]!r}) " n_obs=" %12.0f ({row["n_obs"]!r})'
        lines.append((f'file write fh {body} _n') if to_file else f'display {body}')
    if to_file:
        lines += ['file close fh']
    return '\n'.join(lines) + '\n'


@pytest.mark.skipif(not STATA, reason='needs Stata and Seatbelt')
def test_stata_log_is_redacted_and_the_file_channel_is_scored(tmp_path):
    cell = dict(CELL, lang='stata')
    values = r.reference_values(fp.package_dir('ols_robust'))
    work, record = tmp_path / 'work', tmp_path / 'record'
    (work / '.tmp').mkdir(parents=True)
    record.mkdir()
    shutil_copy(fp.package_dir('ols_robust') / 'data/input.csv', work / 'input.csv')
    (work / 'run.do').write_text(stata_do(values, to_file=True))
    execution = r.execute(work, 'stata', record, 1, inputs={'input.csv'})
    assert execution['ok'], execution['stderr'][-500:]
    log = (work / 'run.log').read_text()
    assert 'Serial number' not in log and 'Licensed to' not in log
    assert 'Serial number' not in (record / 'execution_attempt1.json').read_text()
    assert execution['final_files'] == ['output.txt']
    scored = r.score(cell, work, execution)
    assert scored['passed'] and 'output.txt' in scored['source']
    assert not r.score(cell, work, dict(execution, final_files=[]))['passed']


def shutil_copy(a, b):
    Path(b).write_bytes(Path(a).read_bytes())


# Literal answers ------------------------------------------------------------------------------

def test_literal_answer_check():
    rows = {'x1': {'estimate': 0.8848096680539027, 'se': 0.6858883608259811, 'n_obs': 60}}
    hard = 'print("ANSWER: x1 estimate=0.884809668 se=0.6858883608 n_obs=60")\n'
    assert r.literal_answers(hard, 'python', rows)['flagged']
    honest = 'import pandas as pd\n# expected 0.884809668\nb = fit(1.96, 60)\n'
    assert not r.literal_answers(honest, 'python', rows)['flagged']
    stata = '* expected 0.884809668\ndisplay "ANSWER: x1 estimate=" %24.16g _b[x1]\n'
    assert not r.literal_answers(stata, 'stata', rows)['flagged']
    assert r.literal_answers('local b = 0.8848\n', 'stata', rows)['flagged']


# Cells, watcher, and blocks -----------------------------------------------------------------------

def test_environment_check_matches_and_stops_on_change(tmp_path):
    row = r.environment_check('b1_start', tmp_path, stub_check(EXPECTED))
    assert row['matches'] and row['environment.manifest_sha256'] == EXPECTED
    with pytest.raises(r.EnvironmentChanged):
        r.environment_check('b1_end', tmp_path, stub_check('0' * 64))


def test_manifest_hash_is_recorded_in_the_committed_manifest():
    manifest = json.loads(r.ENV_MANIFEST.read_text())
    assert manifest['manifest_sha256'] == '2e820d09c1bed5f19f640d1f9b5b6dcd5f9db47964dabdc4d1d8470101754e8b'
    assert manifest['failures'] == []


@pytest.mark.skipif(not DARWIN, reason='Seatbelt is macOS only')
def test_block_runs_a_cell_end_to_end_and_records_the_environment(tmp_path, fake):
    root = tmp_path / 'root'
    r.prepare_collection(root, seed_mpl=False)
    clock = FakeClock()
    watcher = fake_watcher(tmp_path, clock)
    stream = canned(tmp_path, '```python\n' + python_solution() + '```', five=0.96)
    options = dict(provider=lambda *a: [str(fake)], dry=True, watcher=watcher,
                   provider_env_extra={'FAKE_MODE': 'normal', 'FAKE_CANNED': str(stream)})
    block = r.run_block('t', [CELL, dict(CELL, rep=2)], root, tmp_path, run_check=stub_check(EXPECTED), **options)
    assert [c['success'] for c in block['cells']] == [True, True]
    records = root / 'records'
    first = json.loads((records / r.cell_id(CELL) / 'record.json').read_text())
    second = json.loads((records / r.cell_id(dict(CELL, rep=2)) / 'record.json').read_text())
    assert first['environment.manifest_sha256'] == EXPECTED and first['scoring']['passed']
    assert first['folder'] == 'c00001' and second['folder'] == 'c00002'
    assert not first['literal_answer_check']['flagged']
    for name in ('prompt.txt', 'prepared.json', 'command_attempt1.json', 'call_attempt1.json',
                 'provider_attempt1.stdout', 'provider_attempt1.times.json', 'execution_attempt1.json',
                 'execution_attempt99.json'):
        assert (records / r.cell_id(CELL) / name).is_file(), name
    command = json.loads((records / r.cell_id(CELL) / 'command_attempt1.json').read_text())
    assert command['real_argv'][0].endswith(r.CLAUDE_VERSION) and command['executed_argv'] == [str(fake)]
    assert first['watch'][0]['waited_seconds'] == 0
    assert second['watch'][0]['decision_at_request'] == 'pause' and second['watch'][0]['waited_seconds'] >= 1200


@pytest.mark.skipif(not DARWIN, reason='Seatbelt is macOS only')
@pytest.mark.parametrize('reset_after', [1800, None])
def test_a_usage_rejection_pauses_and_repeats_the_call(tmp_path, fake, reset_after):
    root = tmp_path / 'root'
    r.prepare_collection(root, seed_mpl=False)
    clock = FakeClock()
    watcher = fake_watcher(tmp_path, clock)
    good = canned(tmp_path, '```python\n' + python_solution() + '```', name='seq')
    rejected(tmp_path, 'seq.1', None if reset_after is None else clock.t + reset_after)
    record = r.run_cell(CELL, root, provider=lambda *a: [str(fake)], watcher=watcher,
                        provider_env_extra={'FAKE_MODE': 'seq', 'FAKE_CANNED': str(good),
                                            'FAKE_COUNTER': str(tmp_path / 'counter')})
    assert record['success'] and len(record['limit_rejections']) == 1 and len(record['calls']) == 1
    waited = record['watch'][1]['waited_seconds']
    if reset_after is None:
        assert record['watch'][1]['decision'] == 'retry' and waited >= 900
    else:
        assert waited >= reset_after + watcher.config['resume_margin_seconds'] - 5


def test_watcher_has_no_call_cap_below_the_chosen_levels(tmp_path):
    clock = FakeClock()
    watcher = fake_watcher(tmp_path, clock)
    watcher.observe_events('claude', [{'type': 'rate_limit_event', 'rate_limit_info': {'status': 'allowed',
                           'unifiedWindows': {'five_hour': {'utilization': 0.84, 'resetsAt': clock.t + 600}}}}])
    assert all(watcher.gate('sonnet55')['decision'] == 'run' for _ in range(500))
    assert clock.slept == []

"""Independent checks of the Sonnet subprocess guard without provider calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'workflows/production_repairs_2026_09_21/accelerate_guarded.py'
SPEC = importlib.util.spec_from_file_location('guarded_acceleration_tested', SOURCE)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def manifest(tmp_path, models):
    (tmp_path / 'manifest.json').write_text(json.dumps({'jobs': [{'model': model} for model in models]}))


def test_actual_provider_and_all_script_paths_keep_guard_and_unlimited_timeout(tmp_path, monkeypatch):
    manifest(tmp_path, ['sonnet'])
    runtime = adapter.load_runtime(tmp_path)
    launches, communications = [], []

    class Child:
        pid = 987654
        returncode = 0

        def __init__(self, argv, **kwargs):
            launches.append(list(argv))
            kwargs['stdout'].write('{}\n')

        def communicate(self, text=None, timeout=None):
            communications.append((text, timeout))

    monkeypatch.setattr(runtime.c.prior.subprocess, 'Popen', Child)
    job = {'model': 'sonnet', 'condition': 'chatbot'}
    original_command = runtime.c.command(job, tmp_path, tmp_path / 'last_attempt1.txt')
    runtime.c.call_model('preserved prompt', tmp_path, job, tmp_path, 1)
    receipt = json.loads((tmp_path / 'command_attempt1.json').read_text())
    assert receipt['argv'] == original_command[0]
    assert receipt['timeout_seconds'] is None
    assert (tmp_path / 'prompt_attempt1.txt').read_text() == 'preserved prompt'
    assert launches[0][3:] == original_command[0]
    for attempt, platform in enumerate(('python', 'r', 'stata'), 2):
        result = runtime.c.prior.execute(tmp_path / 'submitted', platform, tmp_path, tmp_path, attempt)
        assert result['ok'] and result['timeout_seconds'] is None
        assert (tmp_path / f'execution_attempt{attempt}.guard.json').is_file()
    assert len(launches) == 4
    assert all(argv[:3] == ['/usr/bin/sandbox-exec', '-f', str(adapter.GUARD.resolve())] for argv in launches)
    assert communications == [('preserved prompt', None), (None, None), (None, None), (None, None)]


def test_luna_runtime_keeps_original_capture_and_mixed_manifest_rejected(tmp_path, monkeypatch):
    capture = object()
    runtime = SimpleNamespace(c=SimpleNamespace(prior=SimpleNamespace(subprocess_capture=capture)))
    monkeypatch.setattr(adapter, 'BASE_LOAD_RUNTIME', lambda work: runtime)
    manifest(tmp_path, ['luna'])
    assert adapter.load_runtime(tmp_path).c.prior.subprocess_capture is capture
    manifest(tmp_path, ['luna', 'sonnet'])
    with pytest.raises(ValueError, match='single-model'):
        adapter.load_runtime(tmp_path)


def test_changed_guard_blocks_before_subprocess(tmp_path):
    guard = tmp_path / 'guard.sb'
    guard.write_text('original')
    called = []
    runtime = SimpleNamespace(c=SimpleNamespace(prior=SimpleNamespace(subprocess_capture=lambda *args: called.append(args))))
    adapter.guard_runtime(runtime, guard)
    guard.write_text('changed')
    with pytest.raises(RuntimeError, match='guard changed'):
        runtime.c.prior.subprocess_capture(['provider'], tmp_path, tmp_path, 'provider_attempt1', None)
    assert called == []


def test_scheduler_launch_is_outside_guard_and_reaps_workers(tmp_path, monkeypatch):
    path = tmp_path / 'sonnet/5_9/manifest.json'
    candidate = dict(manifest=str(path), pid=123, eligible=True, untouched=4)
    monkeypatch.setattr(adapter.base, 'OPS', tmp_path / 'ops')
    monkeypatch.setattr(adapter.base, 'LAUNCHED_WORKERS', {99: SimpleNamespace(poll=lambda: 0)})
    monkeypatch.setattr(adapter.base, 'inspect', lambda: [candidate])
    launches = []

    def launch(argv, **kwargs):
        launches.append((argv, kwargs))
        return SimpleNamespace(pid=321)

    monkeypatch.setattr(adapter.subprocess, 'Popen', launch)
    adapter.launch_eligible()
    assert 99 not in adapter.base.LAUNCHED_WORKERS
    assert 321 in adapter.base.LAUNCHED_WORKERS
    assert launches[0][0] == [adapter.sys.executable, '-u', str(adapter.HERE), '--handoff', str(path), '--pid', '123']
    assert launches[0][1]['start_new_session'] is True


@pytest.mark.parametrize('mode', ['handoff', 'watchdog'])
def test_main_delegates_unchanged_handoff_and_watchdog(tmp_path, monkeypatch, mode):
    called = []
    monkeypatch.setattr(adapter.base, 'HERE', adapter.base.HERE)
    monkeypatch.setattr(adapter.base.central, 'load_runtime', adapter.base.central.load_runtime)
    monkeypatch.setattr(adapter.base, mode, lambda *args: called.append(args))
    argv = ['guarded', '--' + mode, str(tmp_path)]
    if mode == 'handoff':
        argv += ['--pid', '123']
    monkeypatch.setattr(adapter.sys, 'argv', argv)
    adapter.main()
    assert called == ([(tmp_path, 123)] if mode == 'handoff' else [(tmp_path,)])
    assert adapter.base.central.load_runtime is adapter.load_runtime
    assert adapter.base.HERE == adapter.HERE
    assert adapter.base.WORKERS == 8 and adapter.base.PERMITS == 3

"""Offline checks for source and native-environment identity."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'prospective_collect.py'
spec = importlib.util.spec_from_file_location('prospective_environment_tested', SOURCE)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_runtime_sources_ignore_unrelated_caller_and_site_packages(tmp_path, monkeypatch):
    unrelated = types.ModuleType('unrelated_repository_caller')
    unrelated.__file__ = str(p.ROOT / 'code/unrelated.py')
    monkeypatch.setitem(sys.modules, unrelated.__name__, unrelated)
    runtime = p.load_runtime(tmp_path)
    sources = p.runtime_sources(runtime)
    assert Path(unrelated.__file__) not in sources
    assert not any('.venv' in path.parts for path in sources)
    assert SOURCE in sources
    for filename in ('tasks.py', 'artifact_repair.py', 'codex_backend.py', 'demonstrations.py', 'strict_posthoc.py', 'harness.py', 'prospective_outputs.py'):
        assert SOURCE.parent / filename in sources


def test_python_only_native_environment_needs_no_native_tools():
    runtime = types.SimpleNamespace(c=types.SimpleNamespace())
    assert p.native_environment(runtime, [{'platform': 'python'}]) == {}


def test_native_environment_change_is_rejected(monkeypatch):
    monkeypatch.setattr(p, 'native_environment', lambda runtime, jobs: {'r': {'version': 'changed'}})
    with pytest.raises(ValueError, match='Native environment changed'):
        p.verify_native_environment({'r': {'version': 'saved'}}, None, [{'platform': 'r'}])


def test_missing_required_native_identity_is_rejected(monkeypatch):
    monkeypatch.setattr(p, 'native_environment', lambda runtime, jobs: {'r': {'version': 'live'}})
    with pytest.raises(ValueError, match='Native environment changed'):
        p.verify_native_environment({}, None, [{'platform': 'r'}])


def test_r_identity_matches_executor_startup_and_package_inventory(tmp_path, monkeypatch):
    executable = tmp_path / 'Rscript'
    executable.write_text('local test executable')
    calls = []
    monkeypatch.setattr(p.shutil, 'which', lambda name, path: str(executable))
    def run(argv, **kwargs):
        calls.append(argv)
        return types.SimpleNamespace(returncode=0, stdout='R version test\nPackage\tVersion\njsonlite\t1\n', stderr='')
    monkeypatch.setattr(p.subprocess, 'run', run)
    runtime = types.SimpleNamespace(c=types.SimpleNamespace(environment=lambda: {'PATH': str(tmp_path)}))
    result = p.native_environment(runtime, [{'platform': 'r'}])
    assert calls[0][:3] == [str(executable), '--vanilla', '-e']
    assert 'installed.packages(noCache=TRUE)' in calls[0][3]
    assert set(result) == {'r'}
    assert result['r']['executable_sha256'] == p.sha(executable)


def test_requested_provider_check_does_not_require_other_provider_or_native_tools(monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        output = '--ignore-user-config --ignore-rules' if '--help' in argv else 'skip_host_skill_discovery plugins'
        return types.SimpleNamespace(returncode=0, stdout=output, stderr='')
    monkeypatch.setattr(p.subprocess, 'run', run)
    runtime = types.SimpleNamespace(c=types.SimpleNamespace(
        codex=types.SimpleNamespace(CODEX_BIN='/test/codex'), environment=lambda: {}))
    result = p.provider_cli_versions(runtime, [{'model': 'luna', 'platform': 'python'}])
    assert set(result) == {'codex_version'}
    assert all(argv[0] == '/test/codex' for argv in calls)


def test_missing_r_fails_before_native_execution(monkeypatch):
    monkeypatch.setattr(p.shutil, 'which', lambda *args, **kwargs: None)
    runtime = types.SimpleNamespace(c=types.SimpleNamespace(environment=lambda: {'PATH': ''}))
    with pytest.raises(ValueError, match='Rscript executable is unavailable'):
        p.native_environment(runtime, [{'platform': 'r'}])

import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
import production_cli_compat as compat


def test_exact_accepted_patch_only():
    identity = lambda value: (value['returncode'], value['stdout'])
    saved = dict(returncode=0, stdout='2.1.274 (Claude Code)\n')
    current = dict(returncode=0, stdout='2.1.277 (Claude Code)\n')
    assert compat.accepted_difference('sonnet_version', saved, current, identity)
    assert compat.accepted_difference('sonnet_version', saved, saved, identity)
    assert not compat.accepted_difference('codex_version', saved, current, identity)
    assert not compat.accepted_difference('sonnet_version', saved, dict(current, returncode=1), identity)
    assert not compat.accepted_difference('sonnet_version', saved, dict(current, stdout='2.2.0 (Claude Code)'), identity)


def test_other_verification_errors_propagate(monkeypatch):
    import pytest
    def fail(manifest, runtime):
        raise ValueError('Pinned source changed: file')
    monkeypatch.setattr(compat, 'ORIGINAL_VERIFY', fail)
    with pytest.raises(ValueError, match='Pinned source changed'):
        compat.verify({}, object())


def test_acceptance_keeps_other_provider_identity_checks(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import pytest
    saved = dict(returncode=0, stdout='2.1.274 (Claude Code)\n')
    current = dict(saved, stdout='2.1.277 (Claude Code)\n')
    codex = dict(returncode=0, stdout='codex-cli 0.154.0\n')
    manifest = dict(protocol_sha256='authorized', jobs=[], protocol=dict(cli_versions=dict(sonnet_version=saved, codex_version=codex)))
    receipt = tmp_path / 'acceptance.json'
    receipt.write_text(json.dumps(dict(accepted=True, protocol_sha256=['authorized'])))
    monkeypatch.setattr(compat, 'RECEIPT', receipt)
    def mismatch(manifest, runtime):
        raise ValueError('Provider CLI version changed')
    monkeypatch.setattr(compat, 'ORIGINAL_VERIFY', mismatch)
    versions = dict(sonnet_version=current, codex_version=codex)
    monkeypatch.setattr(compat.collector, 'provider_cli_versions', lambda runtime, jobs: versions)
    runtime = SimpleNamespace(m=SimpleNamespace(version_identity=lambda v: (v['returncode'], v['stdout'])))
    compat.verify(manifest, runtime)
    versions['codex_version'] = dict(codex, stdout='different')
    with pytest.raises(ValueError, match='Unaccepted provider CLI difference'):
        compat.verify(manifest, runtime)
    with pytest.raises(ValueError, match='does not cover'):
        compat.verify(dict(manifest, protocol_sha256='other'), runtime)

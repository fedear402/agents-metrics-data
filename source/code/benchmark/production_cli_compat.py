"""Continue production with the explicitly accepted Claude CLI patch version.

Use the same arguments as prospective_collect.py, or prefix them with queue
to start production_queue.py. Preserve the original manifest and its hashes.
Only the recorded Claude 2.1.274 to installed 2.1.277 difference is accepted.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import prospective_collect as collector

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / 'data/production/paper22_2026-09-18/accepted_cli_patch.json'
ORIGINAL_VERIFY = collector.verify
ORIGINAL_SAVE = collector.save


def accepted_difference(name, saved, current, identity):
    if identity(saved) == identity(current):
        return True
    return (name == 'sonnet_version' and saved.get('returncode') == current.get('returncode') == 0
            and saved.get('stdout', '').strip() == '2.1.274 (Claude Code)'
            and current.get('stdout', '').strip() == '2.1.277 (Claude Code)')


def verify(manifest, runtime=None):
    try:
        ORIGINAL_VERIFY(manifest, runtime)
    except ValueError as error:
        if str(error) != 'Provider CLI version changed' or runtime is None:
            raise
        receipt = json.loads(RECEIPT.read_text())
        if receipt.get('accepted') is not True or manifest['protocol_sha256'] not in receipt['protocol_sha256']:
            raise ValueError('CLI acceptance does not cover this manifest') from error
        # Original verification already checked sources, environment and commands.
        # Check every CLI identity again, with only the accepted pair exempted.
        current = collector.provider_cli_versions(runtime, manifest['jobs'])
        for name, saved in manifest['protocol']['cli_versions'].items():
            if not accepted_difference(name, saved, current[name], runtime.m.version_identity):
                raise ValueError('Unaccepted provider CLI difference: ' + name) from error


def save(path, value):
    if Path(path).name == 'run_record.json' and isinstance(value, dict):
        value = dict(value, collection_cli_acceptance=dict(
            receipt=str(RECEIPT), receipt_sha256=collector.sha(RECEIPT),
            adapter_sha256=collector.sha(__file__),
            recorded_at=datetime.now(timezone.utc).isoformat(),
            note='User accepted the Claude patch version. Raw stream versions remain unchanged.'))
    ORIGINAL_SAVE(path, value)


def main():
    collector.verify = verify
    collector.save = save
    if len(sys.argv) > 1 and sys.argv[1] == 'queue':
        import production_queue as queue
        queue.COLLECTOR = Path(__file__).resolve()
        del sys.argv[1]
        queue.main()
    else:
        collector.main()


if __name__ == '__main__':
    main()

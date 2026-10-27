"""Run new prospective collections without imposed provider or script wall limits.

The historical collector remains unchanged. This entry point configures fresh
runtime modules inside its own process and records the new policy in manifests.
Administrative signals still use the central collector's cancellation behavior.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
_spec=importlib.util.spec_from_file_location('prospective_unlimited_central',HERE/'prospective_collect.py')
central=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(central)
BASE_LOAD_RUNTIME=central.load_runtime
BASE_PREPARE=central.prepare
VERSION='prospective_unlimited_walltime_2026_09_20_v1'


def load_runtime(work):
    """Remove wall limits from fresh provider and script subprocesses."""
    runtime=BASE_LOAD_RUNTIME(work)
    original_command=runtime.c.command
    original_capture=runtime.c.prior.subprocess_capture

    def command(job,cwd,last):
        argv,_=original_command(job,cwd,last)
        return argv,None

    def capture(argv,cwd,destination,stem,timeout,text=None):
        # The inherited script executor supplies 180. The policy here has no cap.
        result=original_capture(argv,cwd,destination,stem,None,text)
        result['timeout_seconds']=None
        return result

    runtime.c.command=command
    runtime.c.prior.subprocess_capture=capture
    return runtime


def prepare(config_path,manifest_path):
    """Prepare the central protocol and explicitly record unlimited wall times."""
    target=Path(manifest_path).resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.unlimited_prep_',dir=target.parent) as temporary:
        manifest=BASE_PREPARE(config_path,Path(temporary)/'manifest.json')
    protocol=manifest['protocol']
    protocol['version']=VERSION
    protocol['policy'].update(provider_wall_timeout_seconds=None,
                              session_wall_timeout_seconds=None,
                              submitted_script_wall_timeout_seconds=None,
                              score='Final submitted script executed locally; numeric and integrity checks remain active. Estimate-only analysis and runtime primary flags remain separate.',
                              walltime_policy='No imposed provider, session, or submitted-script wall limit. Administrative cancellation remains available.')
    protocol['source_hashes'][str(HERE/'prospective_collect_unlimited.py')]=central.sha(HERE/'prospective_collect_unlimited.py')
    protocol['source_hashes'][str(HERE/'tests/test_unlimited_runtime.py')]=central.sha(HERE/'tests/test_unlimited_runtime.py')
    manifest['protocol_sha256']=central.digest(protocol)
    if target.exists():
        old=central.json.loads(target.read_text())
        if old['protocol_sha256']!=manifest['protocol_sha256']:
            raise ValueError('Preserve the existing manifest; select a new destination')
        return old
    central.save(target,manifest)
    return manifest


central.load_runtime=load_runtime
central.prepare=prepare

if __name__=='__main__':
    central.main()

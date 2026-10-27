"""Start the second repetition blocks while existing collectors continue."""
from pathlib import Path
import json
import os
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
META = ROOT / 'data/collections/2026-09-20_three-expansion'

def main():
    receipt = META / 'acceleration_receipt.json'
    assert not receipt.exists(), 'Acceleration already started'
    result = {'at': datetime.now(timezone.utc).isoformat(), 'existing_workers_per_model': 4,
              'added_workers_per_model': 8, 'chosen_total_worker_limit': 24, 'models': {}}
    for model in ('luna', 'sonnet'):
        launch = META / f'{model}_launch_commands.json'
        entries = json.loads(launch.read_text())
        for flag in ('--solo-workers', '--overlap-workers'):
            entries[0]['argv'][entries[0]['argv'].index(flag) + 1] = '8'
        launch.write_text(json.dumps(entries, indent=2) + '\n')
        manifest = ROOT / f'data/runs/three_expansion_2026-09-20/{model}/5_9/manifest.json'
        argv = [json.loads(manifest.read_text())['protocol']['environment']['executable'], '-u', str(ROOT / 'code/benchmark/prospective_collect_unlimited_additions.py'),
                'collect', '--manifest', str(manifest), '--authorization', str(manifest.parent / 'authorization.json'), '--workers', '8']
        if model == 'sonnet':
            argv = ['/usr/bin/sandbox-exec', '-f', str(ROOT / 'data/production/paper22_2026-09-18/work/environment_recovery/shared_python_write_deny.sb'), *argv]
        log = META / f'{model}_parallel_5_9.log'
        with log.open('a') as stream:
            process = subprocess.Popen(argv, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        result['models'][model] = {'pid': process.pid, 'argv': argv, 'log': str(log)}
        receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

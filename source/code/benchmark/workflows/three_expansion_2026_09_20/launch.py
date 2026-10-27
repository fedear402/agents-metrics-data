"""Start each approved model queue with its central supervisor."""
from pathlib import Path
import json
import os
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
META = ROOT / 'data/collections/2026-09-20_three-expansion'


def main():
    receipt_path = META / 'launch_receipt.json'
    assert not receipt_path.exists(), 'Already launched; use supervisors to resume'
    validation = json.loads((META / 'validation.json').read_text())
    assert validation['total'] == 1080 and validation['new'] == 1005
    receipt = dict(at=datetime.now(timezone.utc).isoformat(), models={})
    for model in ['luna', 'sonnet']:
        launch_file = META / (model + '_launch_commands.json')
        entry = json.loads(launch_file.read_text())[0]
        env = {**os.environ, **entry['env']}
        with open(entry['log'], 'a') as log:
            queue = subprocess.Popen(entry['argv'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        command = ['uv', 'run', '--no-sync', 'python', '-u', str(ROOT / 'code/benchmark/collection_supervisor_unlimited_additions.py'), '--launch-file', str(launch_file), '--out', str(META / (model + '_supervisor')), '--interval', '60', '--restart-delay', '300']
        with open(META / (model + '_supervisor.log'), 'a') as log:
            supervisor = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        caffeinate = subprocess.Popen(['caffeinate', '-i', '-w', str(supervisor.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        receipt['models'][model] = dict(queue_pid=queue.pid, supervisor_pid=supervisor.pid, caffeinate_pid=caffeinate.pid)
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

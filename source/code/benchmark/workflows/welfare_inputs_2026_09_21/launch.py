"""Launch four reviewed queues with central supervisors and wake locks."""
from datetime import datetime, timezone
import json
import os
import subprocess
import prepare as p

def main():
    receipt_path = p.META / 'launch_receipt.json'
    assert not receipt_path.exists(), 'Already launched; use central supervisors'
    validation = p.read(p.META / 'validation.json')
    assert validation['passed'] and validation['total'] == validation['new'] == 360
    receipt = dict(at=datetime.now(timezone.utc).isoformat(), blocks={})
    for item in p.read(p.META / 'collections.json'):
        batch = item['batch']
        launch_file = p.META / (batch + '_launch_commands.json')
        entry = p.read(launch_file)[0]
        env = {**os.environ, **entry['env']}
        with open(entry['log'], 'a') as log:
            queue = subprocess.Popen(entry['argv'], cwd=p.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        command = ['uv', 'run', '--no-sync', 'python', '-u', str(p.ROOT / 'code/benchmark/collection_supervisor_unlimited_additions.py'), '--launch-file', str(launch_file), '--out', str(p.META / (batch + '_supervisor')), '--interval', '60', '--restart-delay', '300']
        with open(p.META / (batch + '_supervisor.log'), 'a') as log:
            supervisor = subprocess.Popen(command, cwd=p.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        caffeinate = subprocess.Popen(['caffeinate', '-i', '-w', str(supervisor.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        receipt['blocks'][batch] = dict(queue_pid=queue.pid, supervisor_pid=supervisor.pid, caffeinate_pid=caffeinate.pid)
        p.c.save(receipt_path, receipt)
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()

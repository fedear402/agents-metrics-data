"""Authorize and launch a prepared rerun collection.

This program starts model calls. Only Federico runs it, with --confirm.
Agents must not run it.

It requires an APPROVED decision record for the prepared scope (and run list),
a passed validation.json for the current plan digest, and no earlier launch
receipt. It writes authorization.json next to each manifest, then starts one
central queue, one central supervisor, and one caffeinate process per block,
as in the 2026-09-21 collection. Without --scope it uses the only prepared collection.

  uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/launch.py --confirm
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import original20_plan as p  # noqa: E402

c = p.c
DESCRIPTION = {
    'few_shot': 'Rerun the few-shot positions of the 20 original tasks with the new examples.',
    'few_shot_and_zero_shot': 'Rerun the few-shot and zero-shot positions of the 20 original tasks with the new examples.',
    'original20_added12_both_prompts': ('Rerun both prompts of the 20 original tasks and the 12 added tasks with the '
                                        'new examples.'),
    'flagged_runs': 'Rerun the listed positions with their production prompts.',
    'original20_added12_both_prompts_and_flagged_runs': ('Rerun both prompts of the 20 original tasks and the 12 added '
                                                         'tasks with the new examples, and the listed positions of the '
                                                         'other tasks with their production prompts.'),
}


def readiness(scope=None):
    problems = []
    try:
        scope, meta = p.find_collection(scope)
    except SystemExit as error:
        return [f'run prepare.py --write and check.py first: {error}'], None, None
    plan_path, validation_path = meta / 'plan.json', meta / 'validation.json'
    if not validation_path.is_file():
        return ['run prepare.py --write and check.py first'], None, None
    plan, validation = p.read(plan_path), p.read(validation_path)
    if validation.get('passed') is not True or validation.get('plan_digest') != plan['plan_digest']:
        problems.append('validation.json does not match the current plan; run check.py again')
    run_list_sha256 = (plan.get('run_list') or {}).get('sha256')
    if p.uses_run_list(scope):
        _, current, _ = p.read_run_list(plan['run_list']['path'])
        if current != run_list_sha256:
            problems.append('run list changed since preparation')
    problems += p.decision_problems(plan['scope'], run_list_sha256=run_list_sha256)
    if c.sha(p.DECISION) != plan['decision_record_sha256']:
        problems.append('decision record changed since preparation')
    if (meta / 'launch_receipt.json').exists():
        problems.append('already launched; the central supervisors manage continuation')
    if not p.SONNET_GUARD.is_file():
        problems.append(f'Sonnet guard missing: {p.SONNET_GUARD}')
    return problems, plan, meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scope', choices=sorted(p.SCOPES), help='scope of the prepared collection to launch')
    parser.add_argument('--confirm', action='store_true', help='required: authorize provider calls and launch')
    args = parser.parse_args(argv)
    problems, plan, meta = readiness(args.scope)
    if problems:
        raise SystemExit('Refusing launch: ' + '; '.join(problems))
    blocks = p.read(meta / 'collections.json')
    if not args.confirm:
        raise SystemExit(f'Ready. Add --confirm to authorize provider calls and launch the {len(blocks)} queues.')
    receipt = dict(at=datetime.now(timezone.utc).isoformat(), scope=plan['scope'], plan_digest=plan['plan_digest'],
                   blocks={})
    for entry in blocks:
        manifest = p.read(entry['manifest'])
        c.save(Path(entry['manifest']).parent / 'authorization.json',
               dict(protocol_sha256=manifest['protocol_sha256'], allow_provider_calls=True,
                    decision_record=str(p.DECISION), decision_record_sha256=plan['decision_record_sha256'],
                    scope=DESCRIPTION[plan['scope']]))
    for entry in blocks:
        batch = entry['batch']
        launch_file = meta / (batch + '_launch_commands.json')
        command = p.read(launch_file)[0]
        env = {**os.environ, **command['env']}
        with open(command['log'], 'a') as log:
            queue = subprocess.Popen(command['argv'], cwd=p.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        supervisor_argv = ['uv', 'run', '--no-sync', 'python', '-u', str(p.SUPERVISOR), '--launch-file', str(launch_file),
                           '--out', str(meta / (batch + '_supervisor')), '--interval', '60', '--restart-delay', '300']
        with open(meta / (batch + '_supervisor.log'), 'a') as log:
            supervisor = subprocess.Popen(supervisor_argv, cwd=p.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
        caffeinate = subprocess.Popen(['caffeinate', '-i', '-w', str(supervisor.pid)], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, start_new_session=True)
        receipt['blocks'][batch] = dict(queue_pid=queue.pid, supervisor_pid=supervisor.pid, caffeinate_pid=caffeinate.pid)
        c.save(meta / 'launch_receipt.json', receipt)
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())

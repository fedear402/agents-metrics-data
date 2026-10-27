"""Model check for the GPT-6 Sol collection: two calls with the manifest's Codex commands, saved rollouts.

Decision record: quality_reports/decision_records/2026-10-17_haiku55-sol-full-design.md.
The paper's Codex command uses --ephemeral, so collected runs save no rollout and their streams do not name the
model. This probe runs the manifest's chatbot and agent commands without --ephemeral, in a temporary Codex home
that links the ChatGPT login (~/.codex/auth.json), and reads each rollout. Each must show the model (--model-id, default gpt-6-sol), effort low,
and Codex 0.163.0, and the agent call must run a shell command. The result goes to
data/collections/2026-10-17_full-design/<batch>_preflight_<time>.json.

Run from the repository root:
    uv run --no-sync python code/benchmark/workflows/full_design_2026_10_17/preflight_sol.py --manifest M
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
META = ROOT / 'data/collections/2026-10-17_full-design'
AUTH = Path.home() / '.codex/auth.json'
PROMPTS = {'chatbot': 'Reply with the single word ready.',
           'agent': 'Run the shell command `echo ready` once, then reply with its output only.'}


def probe(argv, condition):
    with tempfile.TemporaryDirectory(prefix='sol_probe_home_') as home, \
            tempfile.TemporaryDirectory(prefix='sol_probe_cwd_') as cwd:
        os.symlink(AUTH, Path(home) / 'auth.json')
        command = [Path(cwd).as_posix() if a == '<cwd>' else str(Path(cwd) / 'last.txt') if a == '<last>' else a
                   for a in argv if a != '--ephemeral']
        env = dict(os.environ, CODEX_HOME=home)
        out = subprocess.run(command, input=PROMPTS[condition], capture_output=True, text=True, cwd=cwd, env=env,
                             timeout=900)
        rollout = sorted(Path(home).glob('sessions/**/rollout-*.jsonl'))
        facts = dict(returncode=out.returncode, rollouts=len(rollout), model=None, effort=None, cli_version=None,
                     commands=0, errors=[])
        for line in out.stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('type') in ('error', 'turn.failed'):
                facts['errors'].append(json.dumps(event)[:300])
            item = event.get('item') or {}
            if event.get('type') == 'item.completed' and item.get('type') == 'command_execution':
                facts['commands'] += 1
        for path in rollout:
            for line in path.read_text(errors='replace').splitlines():
                event = json.loads(line)
                payload = event.get('payload') or {}
                if event.get('type') == 'session_meta':
                    facts['cli_version'] = payload.get('cli_version')
                elif event.get('type') == 'turn_context':
                    facts['model'] = payload.get('model')
                    facts['effort'] = payload.get('effort') or payload.get('reasoning_effort')
        return facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model-id', default='gpt-6-sol')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.resolve().read_text())
    batch = manifest['jobs'][0]['block']
    commands = manifest['protocol']['commands']
    results = {condition: probe(commands[f'luna/{condition}'][0], condition) for condition in ('chatbot', 'agent')}
    problems = []
    for condition, facts in results.items():
        if (facts['model'], facts['effort'], facts['cli_version']) != (args.model_id, 'low', '0.163.0'):
            problems.append(f'{condition}: {facts["model"]} {facts["effort"]} {facts["cli_version"]}')
        if facts['returncode'] != 0 or facts['errors']:
            problems.append(f'{condition}: return code {facts["returncode"]}, errors {facts["errors"]}')
    if results['agent']['commands'] < 1:
        problems.append('agent: no shell command ran')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report = dict(batch=batch, protocol_sha256=manifest['protocol_sha256'], at=stamp, results=results,
                  problems=problems, passed=not problems)
    META.mkdir(parents=True, exist_ok=True)
    (META / f'{batch}_preflight_{stamp}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))
    raise SystemExit(1 if problems else 0)


if __name__ == '__main__':
    main()

"""Prepare the Haiku 5.5 API-key collection of one paper task. No model call.

Rule file: quality_reports/decision_records/2026-10-16_haiku55-api-collection.md.
Prompts: the texts that the paper's selected Sonnet 4.6 runs of the task received (prompt_attempt1.txt of each
record listed in data/analysis/main/data/runs.csv). Each cell key must have one text over its ten repetitions.
When a canonical prompt file exists for the key, the text must equal it after the agent instruction update.
Task definition: the paper catalog entry (data/analysis/main/data/catalog.json).

Run from the repository root. HAIKU_API_KEY_FILE must be set, because prep and check run the launch script
for the CLI version:
    uv run --no-sync python code/benchmark/workflows/haiku55_api_2026_10_16/prepare.py --task broad_tobit --reps 0
"""
from pathlib import Path
import argparse
import json
import random
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import prospective_collect as c  # noqa: E402  (helpers only: prompt_key, prepared_prompt, sha, save)

COLLECTORS = {'haiku55': ('prospective_collect_haiku_api.py', 'claude-haiku-5-5'),
              'opus55': ('prospective_collect_opus_api.py', 'claude-opus-5-5')}
COLLECTOR = ROOT / 'code/benchmark/prospective_collect_haiku_api.py'
MODEL_ID = 'claude-haiku-5-5'
LAUNCHER = ROOT / 'code/benchmark/haiku_api_claude.sh'
RUNS = ROOT / 'data/analysis/main/data/runs.csv'
CATALOG = ROOT / 'data/analysis/main/data/catalog.json'
CANONICAL = [ROOT / 'data/production/added20_prompts_2026-09-26/prompts',
             ROOT / 'data/production/original10_2026-09-19/prompts',
             ROOT / 'data/collections/2026-09-17_live-check-and-checkpoint/work/old_prompts/prompts']
DECISION = ROOT / 'quality_reports/decision_records/2026-10-16_haiku55-api-collection.md'
RAW = ROOT / 'data/runs/haiku55_api_2026-10-16'
META = ROOT / 'data/collections/2026-10-16_haiku55-api'
GUARD = ROOT / 'data/production/paper22_2026-09-18/work/environment_recovery/shared_python_write_deny.sb'
WORKERS = 4  # chosen collection setting; the paper batches used 4 or 8
PLATFORMS, PROMPTS, CONDITIONS = ('python', 'r', 'stata'), ('zero_shot', 'few_shot'), ('chatbot', 'self_repair', 'agent')


def output_format(platform):
    return 'answer' if platform == 'stata' else 'json'


def paper_prompts(task):
    """One prompt text per cell key, from the paper's selected Sonnet records of the task."""
    runs = pd.read_csv(RUNS, low_memory=False)
    rows = runs[(runs.task == task) & (runs.model == 'sonnet')]
    if len(rows) != 180:
        raise ValueError(f'{task}: expected 180 paper Sonnet runs, found {len(rows)}')
    prompts, sources = {}, {}
    for row in rows.itertuples():
        key = c.prompt_key(dict(task=task, platform=row.platform, prompt=row.prompt, condition=row.condition,
                                output_format=output_format(row.platform)))
        text = (ROOT / row.record_path).parent.joinpath('prompt_attempt1.txt').read_text()
        if prompts.setdefault(key, text) != text:
            raise ValueError(f'{key}: the paper runs used more than one prompt text')
        sources.setdefault(key, []).append(row.record_path)
    if len(prompts) != 18:
        raise ValueError(f'{task}: expected 18 cell keys, found {len(prompts)}')
    canonical = {}
    for key, text in prompts.items():
        files = [folder / (key + '.txt') for folder in CANONICAL if (folder / (key + '.txt')).is_file()]
        if files:
            expected = c.prepared_prompt(files[0].read_text(), key.split('__')[3])
            canonical[key] = dict(file=str(files[0].relative_to(ROOT)), equal=expected == text)
            if expected != text:
                raise ValueError(f'{key}: the paper prompt differs from the canonical file {files[0]}')
    return prompts, sources, canonical


def run_collector(action, manifest, config=None):
    argv = [sys.executable, str(COLLECTOR), action, '--manifest', str(manifest)]
    if config:
        argv += ['--config', str(config)]
    process = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT)
    receipt = dict(action=action, returncode=process.returncode, stdout=process.stdout[-4000:],
                   stderr=process.stderr[-4000:])
    if process.returncode:
        raise RuntimeError(f'{action} failed: {process.stdout[-2000:]} {process.stderr[-2000:]}')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--reps', type=int, nargs='+', required=True)
    parser.add_argument('--model', choices=sorted(COLLECTORS), default='haiku55')
    args = parser.parse_args()
    global COLLECTOR, MODEL_ID
    COLLECTOR = ROOT / 'code/benchmark' / COLLECTORS[args.model][0]
    MODEL_ID = COLLECTORS[args.model][1]
    if not DECISION.is_file():
        raise SystemExit('The rule file is missing')
    if max(args.reps) - min(args.reps) > 4:
        raise SystemExit('One manifest holds at most five repetitions')
    task, reps = args.task, sorted(set(args.reps))
    prompts, sources, canonical = paper_prompts(task)
    spec = json.loads(CATALOG.read_text())[task]
    batch = f'{args.model}_{task}_{reps[0]}_{reps[-1]}'
    folder = (RAW if args.model == 'haiku55' else ROOT / f'data/runs/{args.model}_api_2026-10-19') / task / f'{reps[0]}_{reps[-1]}'
    paths = {}
    for key, text in prompts.items():
        path = folder / 'prompts' / (key + '.txt')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        paths[key] = str(path)
    jobs = []
    for rep in reps:
        wave = [dict(block=batch, task=task, model='sonnet', platform=platform, prompt=prompt, condition=condition,
                     output_format=output_format(platform), rep=rep % 5, global_rep=rep)
                for platform in PLATFORMS for prompt in PROMPTS for condition in CONDITIONS]
        random.Random(20261016 + rep).shuffle(wave)
        jobs.extend(wave)
    c.save(folder / 'catalog.json', {task: spec})
    c.save(folder / 'config.json', dict(catalog_path=str(folder / 'catalog.json'), prompt_paths=paths, jobs=jobs,
                                        decision_record=str(DECISION)))
    manifest_path = folder / 'manifest.json'
    receipts = [run_collector('prep', manifest_path, folder / 'config.json'), run_collector('check', manifest_path)]
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest['protocol']
    for key, text in prompts.items():
        if protocol['prompts'][key] != text:
            raise ValueError(f'{key}: the prepared prompt differs from the paper prompt')
    for name, (argv, timeout) in protocol['commands'].items():
        if argv[0] != str(ROOT / 'code/benchmark' / ('opus_api_claude.sh' if MODEL_ID == 'claude-opus-5-5' else 'haiku_api_claude.sh')) or argv[argv.index('--model') + 1] != MODEL_ID or timeout is not None:
            raise ValueError(f'Unexpected command for {name}')
    version = protocol['cli_versions']['sonnet_version']['stdout'].strip()
    if version != ('2.1.280 (Claude Code)' if args.model == 'opus55' else '2.1.278 (Claude Code)'):
        raise ValueError('Unexpected Claude Code version: ' + version)
    authorization = folder / 'authorization.json'
    c.save(authorization, dict(protocol_sha256=manifest['protocol_sha256'], allow_provider_calls=True,
                               decision_record=str(DECISION),
                               scope=f'Haiku 5.5 API-key collection of {task}, repetitions {reps}, '
                                     f'{len(jobs)} runs (Federico, 2026-10-16).'))
    launch = dict(name=batch, cwd=str(ROOT), env_required=['HAIKU_API_KEY_FILE'],
                  argv=['/usr/bin/sandbox-exec', '-f', str(GUARD), sys.executable, '-u', str(COLLECTOR), 'collect',
                        '--manifest', str(manifest_path), '--authorization', str(authorization),
                        '--workers', str(WORKERS)],
                  log=str(META / (batch + '_collector.log')))
    c.save(META / (batch + '_launch_commands.json'), [launch])
    c.save(META / (batch + '_preparation.json'), dict(
        task=task, reps=reps, jobs=len(jobs), manifest=str(manifest_path), protocol_sha256=manifest['protocol_sha256'],
        claude_code=version, prompt_sources=sources, canonical_prompt_check=canonical, receipts=receipts))
    print(json.dumps(dict(batch=batch, jobs=len(jobs), manifest=str(manifest_path.relative_to(ROOT)),
                          protocol_sha256=manifest['protocol_sha256'], claude_code=version,
                          canonical_equal=sum(v['equal'] for v in canonical.values()))))


if __name__ == '__main__':
    main()

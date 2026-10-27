"""Prepare one manifest of the full paper design for Haiku 5.5 or GPT-6 Sol. No model call.

Decision record: quality_reports/decision_records/2026-10-17_haiku55-sol-full-design.md.
- Prompts: the final prompt set (720 cell keys) from the three canonical folders, with the corrected LP-IV texts of
  the production repairs. Agent prompts go through prepared_prompt(). Every text must equal the prompt of the
  paper's Luna runs of that cell (prompt_attempt1.txt, the first state file, or the saved prompt hash). The script
  records where the paper's Sonnet runs used another text.
- Tasks: the paper catalog, except lp04_ppml, which uses the production-repairs manifest specification.
- Jobs: 40 tasks x 3 languages x 2 prompts x 3 modes for each repetition, shuffled within the repetition.
The script runs prep and check (no model calls) and writes the authorization.

Run from the repository root (Haiku needs HAIKU_API_KEY_FILE for the CLI version check):
    uv run --no-sync python code/benchmark/workflows/full_design_2026_10_17/prepare.py --model sol --reps 0 4
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import prospective_collect as c  # noqa: E402  (helpers only: prompt_key, prepared_prompt, save)

MODELS = {
    'sonnet55': dict(label='sonnet', collector='prospective_collect_sonnet55_api.py', model_id='claude-sonnet-5-5',
                     version_key='sonnet_version', version='2.1.280 (Claude Code)',
                     decision='quality_reports/decision_records/2026-10-23_sonnet55-full-design.md',
                     approved='2026-10-23'),
    'opus55': dict(label='sonnet', collector='prospective_collect_opus_api.py', model_id='claude-opus-5-5',
                   version_key='sonnet_version', version='2.1.280 (Claude Code)'),
    'haiku55': dict(label='sonnet', collector='prospective_collect_haiku_api.py', model_id='claude-haiku-5-5',
                    version_key='sonnet_version', version='2.1.278 (Claude Code)'),
    'luna6': dict(label='luna', collector='prospective_collect_luna6.py', model_id='gpt-6-luna',
                  version_key='codex_version', version='codex-cli 0.163.0'),
    'sol': dict(label='luna', collector='prospective_collect_sol.py', model_id='gpt-6-sol',
                version_key='codex_version', version='codex-cli 0.163.0'),
}
CANONICAL = [ROOT / 'data/runs/production_repairs_2026-09-21/luna/0_4/prompts',  # corrected LP-IV texts first
             ROOT / 'data/production/added20_prompts_2026-09-26/prompts',
             ROOT / 'data/production/original10_2026-09-19/prompts',
             ROOT / 'data/collections/2026-09-17_live-check-and-checkpoint/work/old_prompts/prompts']
CATALOG = ROOT / 'data/analysis/main/data/catalog.json'
PPML_MANIFEST = ROOT / 'data/runs/production_repairs_2026-09-21/luna/0_4/manifest.json'
RUNS = ROOT / 'data/analysis/main/data/runs.csv'
DECISION = ROOT / 'quality_reports/decision_records/2026-10-17_haiku55-sol-full-design.md'
RAW = ROOT / 'data/runs/full_2026-10-17'
META = ROOT / 'data/collections/2026-10-17_full-design'
# Runs collected earlier with the same collector, prompts, command, and settings, counted in the design.
COVERED = {'opus55': {('broad_tobit', 0)}}  # data/runs/opus55_api_2026-10-19/broad_tobit/0_0 (decision record 2026-10-19_opus55-one-task)
PLATFORMS, PROMPTS, CONDITIONS = ('python', 'r', 'stata'), ('zero_shot', 'few_shot'), ('chatbot', 'self_repair', 'agent')


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def output_format(platform):
    return 'answer' if platform == 'stata' else 'json'


def canonical_prompts(tasks):
    prompts, sources = {}, {}
    for task in tasks:
        for platform in PLATFORMS:
            for prompt in PROMPTS:
                for condition in CONDITIONS:
                    key = c.prompt_key(dict(task=task, platform=platform, prompt=prompt, condition=condition,
                                            output_format=output_format(platform)))
                    path = next(folder / (key + '.txt') for folder in CANONICAL if (folder / (key + '.txt')).is_file())
                    prompts[key] = c.prepared_prompt(path.read_text(), condition)
                    sources[key] = str(path.relative_to(ROOT))
    return prompts, sources


def paper_prompt_hash(record_path):
    folder = (ROOT / record_path).parent
    if (folder / 'prompt_attempt1.txt').is_file():
        return sha_text((folder / 'prompt_attempt1.txt').read_text())
    record = json.loads((folder / 'run_record.json').read_text())
    for value in (record.get('prompt_sha256'), (record.get('job') or {}).get('prompt_sha256')):
        if value:
            return value
    if record.get('prompt'):
        return sha_text(record['prompt'])
    states = sorted(folder.glob('state_0*.json'))
    if states and json.loads(states[0].read_text()).get('prompt'):
        return sha_text(json.loads(states[0].read_text())['prompt'])
    return None


def compare_with_paper(prompts):
    runs = pd.read_csv(RUNS, low_memory=False)
    result = {}
    for model in ('luna', 'sonnet'):
        equal, differ, missing = 0, {}, 0
        for row in runs[runs.model == model].itertuples():
            key = c.prompt_key(dict(task=row.task, platform=row.platform, prompt=row.prompt, condition=row.condition,
                                    output_format=output_format(row.platform)))
            found = paper_prompt_hash(row.record_path)
            if found is None:
                missing += 1
            elif found == sha_text(prompts[key]):
                equal += 1
            else:
                differ.setdefault(key, []).append(int(row.global_rep))
        result[model] = dict(runs_equal=equal, runs_without_prompt=missing,
                             keys_with_other_text={k: sorted(v) for k, v in sorted(differ.items())})
    return result


def run_collector(collector, action, manifest, config=None):
    argv = [sys.executable, str(ROOT / 'code/benchmark' / collector), action, '--manifest', str(manifest)]
    if config:
        argv += ['--config', str(config)]
    process = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT)
    if process.returncode:
        raise RuntimeError(f'{action} failed: {process.stdout[-3000:]} {process.stderr[-3000:]}')
    return dict(action=action, returncode=process.returncode, stdout=process.stdout[-2000:], stderr=process.stderr[-2000:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=sorted(MODELS), required=True)
    parser.add_argument('--reps', type=int, nargs=2, required=True, metavar=('FIRST', 'LAST'))
    args = parser.parse_args()
    first, last = args.reps
    decision = ROOT / MODELS[args.model].get('decision', DECISION.relative_to(ROOT))
    if not decision.is_file():
        raise SystemExit('The decision record is missing')
    if not 0 <= first <= last <= 9 or last - first > 4:
        raise SystemExit('One manifest holds at most five repetitions from 0 to 9')
    setting = MODELS[args.model]
    catalog = json.loads(CATALOG.read_text())
    if len(catalog) != 40:
        raise ValueError('The paper catalog must hold 40 tasks')
    specs = {task: spec for task, spec in catalog.items()}
    specs['lp04_ppml'] = json.loads(PPML_MANIFEST.read_text())['protocol']['tasks']['lp04_ppml']
    prompts, sources = canonical_prompts(sorted(specs))
    if len(prompts) != 720:
        raise ValueError(f'Expected 720 cell keys, found {len(prompts)}')
    comparison = compare_with_paper(prompts)
    if comparison['luna']['keys_with_other_text'] or comparison['luna']['runs_without_prompt']:
        raise ValueError('The final prompt set differs from the paper Luna prompts: '
                         + json.dumps(comparison['luna'])[:2000])
    batch = f'full_{args.model}_{first}_{last}'
    folder = RAW / args.model / f'{first}_{last}'
    paths = {}
    for key, text in prompts.items():
        path = folder / 'prompts' / (key + '.txt')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        paths[key] = str(path)
    jobs = []
    for rep in range(first, last + 1):
        wave = [dict(block=batch, task=task, model=setting['label'], platform=platform, prompt=prompt,
                     condition=condition, output_format=output_format(platform), rep=rep % 5, global_rep=rep)
                for task in sorted(specs) for platform in PLATFORMS for prompt in PROMPTS for condition in CONDITIONS]
        random.Random(20261017 + rep).shuffle(wave)
        jobs.extend(wave)
    covered = COVERED.get(args.model, set())
    jobs = [j for j in jobs if (j['task'], j['global_rep']) not in covered]
    c.save(folder / 'catalog.json', specs)
    c.save(folder / 'config.json', dict(catalog_path=str(folder / 'catalog.json'), prompt_paths=paths, jobs=jobs,
                                        decision_record=str(decision)))
    manifest_path = folder / 'manifest.json'
    receipts = [run_collector(setting['collector'], 'prep', manifest_path, folder / 'config.json'),
                run_collector(setting['collector'], 'check', manifest_path)]
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest['protocol']
    for key, text in prompts.items():
        if protocol['prompts'][key] != text:
            raise ValueError(f'{key}: the prepared prompt differs from the final prompt set')
    for name, (argv, timeout) in protocol['commands'].items():
        model = argv[argv.index('--model') + 1] if '--model' in argv else argv[argv.index('-m') + 1]
        if model != setting['model_id'] or timeout is not None:
            raise ValueError(f'Unexpected command for {name}: {model} {timeout}')
    version = protocol['cli_versions'][setting['version_key']]['stdout'].strip()
    if version != setting['version']:
        raise ValueError('Unexpected CLI version: ' + version)
    authorization = folder / 'authorization.json'
    c.save(authorization, dict(protocol_sha256=manifest['protocol_sha256'], allow_provider_calls=True,
                               decision_record=str(decision),
                               scope=f'Full design, {args.model}, repetitions {first}-{last}, {len(jobs)} runs '
                                     f"(Federico, {setting.get('approved', '2026-10-17')})."))
    c.save(META / f'{batch}_preparation.json', dict(
        model=args.model, model_id=setting['model_id'], reps=[first, last], jobs=len(jobs),
        manifest=str(manifest_path.relative_to(ROOT)), protocol_sha256=manifest['protocol_sha256'], cli=version,
        prompt_sources=sources, paper_prompt_comparison=comparison, receipts=receipts))
    print(json.dumps(dict(batch=batch, jobs=len(jobs), manifest=str(manifest_path.relative_to(ROOT)),
                          protocol_sha256=manifest['protocol_sha256'], cli=version,
                          luna_runs_equal=comparison['luna']['runs_equal'],
                          sonnet_keys_with_other_text=len(comparison['sonnet']['keys_with_other_text']))))


if __name__ == '__main__':
    main()

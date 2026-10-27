"""Stage check of the full-design collection for Haiku 5.5 or GPT-6 Sol. No model call.

Decision record: quality_reports/decision_records/2026-10-17_haiku55-sol-full-design.md.
For the jobs of the chosen repetitions in one manifest:
- every job has one completed record; holds and unfinished jobs are listed;
- Haiku: every saved Claude stream reports claude-haiku-5-5, apiKeySource ANTHROPIC_API_KEY, and Claude Code
  2.1.278, every assistant message names claude-haiku-5-5, and no file of the manifest folder or of
  data/collections/2026-10-17_full-design contains the key (HAIKU_API_KEY_FILE);
- Codex models (GPT-6 Sol, GPT-6 Luna): every saved command uses the pinned Codex binary, the manifest's model, and effort low; stream errors other than
  usage limits are listed.
The result goes to data/collections/2026-10-17_full-design/<batch>_validation_<first>_<last>.json.

Run from the repository root:
    uv run --no-sync python code/benchmark/workflows/full_design_2026_10_17/validate.py --manifest M --reps 0 0
"""
from collections import Counter
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import prospective_collect as c  # noqa: E402  (helpers only: job_id, completed, TERMINAL)
from codex_backend import is_usage_limit  # noqa: E402

META = ROOT / 'data/collections/2026-10-17_full-design'
CODEX = str(ROOT / 'data/collections/sol_codex_bin/codex-0.163.0/bin/codex')


def events(path):
    for line in path.read_text(errors='replace').splitlines():
        try:
            yield json.loads(line)
        except ValueError:
            continue


def haiku_problems(folder, model_id, version):
    problems = []
    for path in sorted(folder.glob('attempt_*/provider_attempt*.stdout')):
        init, models = {}, set()
        for event in events(path):
            if event.get('type') == 'system' and event.get('subtype') == 'init' and not init:
                init = event
            elif event.get('type') == 'assistant':
                models.add((event.get('message') or {}).get('model'))
        found = (init.get('model'), init.get('apiKeySource'), init.get('claude_code_version'))
        if found != (model_id, 'ANTHROPIC_API_KEY', version):
            problems.append(f'{path.relative_to(folder)}: init {found}')
        if any(m and m != model_id for m in models):
            problems.append(f'{path.relative_to(folder)}: messages from {sorted(m for m in models if m)}')
    return problems


def sol_problems(folder, errors, model_id):
    problems = []
    for path in sorted(folder.glob('attempt_*/command_attempt*.json')):
        argv = json.loads(path.read_text()).get('argv') or []
        if not argv or argv[0] != CODEX or argv[argv.index('-m') + 1] != model_id \
                or 'model_reasoning_effort="low"' not in argv:
            problems.append(f'{path.relative_to(folder)}: command {argv[:1]} model or effort differs')
    for path in sorted(folder.glob('attempt_*/provider_attempt*.stdout')):
        for event in events(path):
            if event.get('type') in ('error', 'turn.failed'):
                text = json.dumps(event)
                errors['usage_limit' if is_usage_limit(text) else 'other'] += 1
                if not is_usage_limit(text):
                    problems.append(f'{path.relative_to(folder)}: {text[:200]}')
    return problems


def files_with_key(folders, key):
    needle, hits = key.encode(), []
    for folder in folders:
        for path in Path(folder).rglob('*'):
            if path.is_file() and needle in path.read_bytes():
                hits.append(str(path.relative_to(ROOT)))
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--reps', type=int, nargs=2, required=True, metavar=('FIRST', 'LAST'))
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    batch = manifest['jobs'][0]['block']
    label = manifest['jobs'][0]['model']
    if label == 'sonnet':
        argv = manifest['protocol']['commands']['sonnet/chatbot'][0]
        model_id = argv[argv.index('--model') + 1]
        version = manifest['protocol']['cli_versions']['sonnet_version']['stdout'].split()[0]
    else:
        argv = manifest['protocol']['commands']['luna/chatbot'][0]
        model_id, version = argv[argv.index('-m') + 1], None
    run_root = manifest_path.parent / 'runs' / manifest['protocol_sha256']
    first, last = args.reps
    jobs = [j for j in manifest['jobs'] if first <= j['global_rep'] <= last]
    states, problems, errors = Counter(), [], Counter()
    unfinished = []
    for job in jobs:
        folder = run_root / c.job_id(job)
        record = c.completed(folder, manifest['protocol_sha256'])
        if record is None:
            state_path = folder / 'attempt_0001/state.json'
            data = json.loads(state_path.read_text()) if state_path.is_file() else {}
            status = data.get('status', 'pending')
            if status == 'paused' and data.get('hold_reason'):
                status = 'needs_review'  # paused by an interrupted script and not resolved: a hold
            states[status] += 1
            unfinished.append(dict(id=c.job_id(job), status=status))
            continue
        states[record['status']] += 1
        problems += haiku_problems(folder, model_id, version) if label == 'sonnet' else sol_problems(folder, errors, model_id)
    leaks = []
    if label == 'sonnet':
        leaks = files_with_key([manifest_path.parent, META], Path(os.environ['HAIKU_API_KEY_FILE']).read_text().strip())
        if leaks:
            problems.append(f'{len(leaks)} files contain the key')
    holds = [u for u in unfinished if u['status'] == 'needs_review']
    open_jobs = [u for u in unfinished if u['status'] != 'needs_review']
    # Holds are set aside for review, as the paper queue did. The driver stops a model with too many holds.
    report = dict(batch=batch, reps=[first, last], planned=len(jobs), states=dict(states), holds=holds,
                  unfinished=open_jobs, stream_errors=dict(errors), files_with_key=leaks, problems=problems[:200],
                  problem_count=len(problems), passed=not problems and not open_jobs)
    META.mkdir(parents=True, exist_ok=True)
    (META / f'{batch}_validation_{first}_{last}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('unfinished', 'problems', 'holds')}
                     | dict(holds=len(holds), unfinished=len(open_jobs))))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

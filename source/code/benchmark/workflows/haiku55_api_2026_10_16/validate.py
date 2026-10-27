"""Check a finished Haiku 5.5 API-key collection. No model call.

Rule file: quality_reports/decision_records/2026-10-16_haiku55-api-collection.md.
- Every planned run has one completed record.
- Every saved Claude stream reports claude-haiku-5-5, apiKeySource ANTHROPIC_API_KEY, and Claude Code 2.1.278,
  and every assistant message names claude-haiku-5-5. Other models in modelUsage (side calls) are listed.
- No file of the collection folder or of data/collections/2026-10-16_haiku55-api contains the key.
The result goes to data/collections/2026-10-16_haiku55-api/<batch>_validation.json.

Run from the repository root with HAIKU_API_KEY_FILE set:
    uv run --no-sync python code/benchmark/workflows/haiku55_api_2026_10_16/validate.py --manifest M
"""
from collections import Counter
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import prospective_collect as c  # noqa: E402  (helpers only: job_id, completed)

META = ROOT / 'data/collections/2026-10-16_haiku55-api'
MODEL, ROUTE, VERSION = 'claude-haiku-5-5', 'ANTHROPIC_API_KEY', '2.1.278'


def stream_facts(path):
    init, messages, usage = {}, set(), set()
    for line in path.read_text(errors='replace').splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') == 'system' and event.get('subtype') == 'init' and not init:
            init = event
        elif event.get('type') == 'assistant':
            messages.add((event.get('message') or {}).get('model'))
        elif event.get('type') == 'result':
            usage |= set((event.get('modelUsage') or {}).keys())
    return dict(model=init.get('model'), route=init.get('apiKeySource'), version=init.get('claude_code_version'),
                message_models=sorted(m for m in messages if m), usage_models=sorted(usage))


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
    args = parser.parse_args()
    key = Path(os.environ['HAIKU_API_KEY_FILE']).read_text().strip()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    global MODEL, VERSION
    VERSION = manifest['protocol']['cli_versions']['sonnet_version']['stdout'].split()[0]
    argv = manifest['protocol']['commands']['sonnet/chatbot'][0]
    MODEL = argv[argv.index('--model') + 1]
    batch = manifest['jobs'][0]['block']
    run_root = manifest_path.parent / 'runs' / manifest['protocol_sha256']
    problems, streams, side = [], 0, Counter()
    missing = [c.job_id(job) for job in manifest['jobs']
               if c.completed(run_root / c.job_id(job), manifest['protocol_sha256']) is None]
    if missing:
        problems.append(f'{len(missing)} planned runs have no completed record: {missing}')
    for path in sorted(run_root.rglob('provider_attempt*.stdout')):
        streams += 1
        facts = stream_facts(path)
        name = str(path.relative_to(run_root))
        if (facts['model'], facts['route'], facts['version']) != (MODEL, ROUTE, VERSION):
            problems.append(f'{name}: init {facts["model"]} {facts["route"]} {facts["version"]}')
        if any(m != MODEL for m in facts['message_models']):
            problems.append(f'{name}: assistant messages from {facts["message_models"]}')
        side.update(m for m in facts['usage_models'] if m != MODEL)
    leaks = files_with_key([manifest_path.parent, META], key)
    if leaks:
        problems.append(f'{len(leaks)} files contain the key')
    report = dict(batch=batch, protocol_sha256=manifest['protocol_sha256'], planned=len(manifest['jobs']),
                  completed=len(manifest['jobs']) - len(missing), streams=streams, side_models=dict(side),
                  files_with_key=leaks, problems=problems, passed=not problems)
    (META / f'{batch}_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))
    raise SystemExit(1 if problems else 0)


if __name__ == '__main__':
    main()

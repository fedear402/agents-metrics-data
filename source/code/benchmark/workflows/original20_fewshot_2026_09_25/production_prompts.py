"""Extract the production prompts of the 20 added tasks from the selected runs.

The 20 original tasks have fixed production prompt folders (original20_plan.PROMPT_FOLDERS).
The 20 added tasks were collected in several collections. This program reads each
selected run of an added task in data/production/main/baseline_runs.csv, finds the
manifest whose protocol hash equals the run record's hash, and takes the manifest
prompt of the run's own job key. A record without such a manifest gives its stored
prompt only for chatbot and agent runs, because a self-repair record stores its last
round. The canonical text of a prompt key is the text of most selected runs of the key.

The output folder holds one <key>.txt per prompt key (task__platform__prompt__condition__format)
and provenance.json with the texts, their run counts, and the manifests. Manifest
prompts already contain the collector's agent instruction, so prepared_prompt leaves
them unchanged. The program calls no model provider.

  uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/production_prompts.py
  uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/production_prompts.py --write
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import original20_plan as p  # noqa: E402

c = p.c


def manifest_candidates(row):
    """The collection manifest above the record's runs folder, then the baseline manifest path."""
    parts = Path(row['record_path']).parts
    last_runs = len(parts) - 1 - parts[::-1].index('runs')
    paths = [Path(*parts[:last_runs]) / 'manifest.json']
    if row.get('manifest_path'):
        paths.append(Path(row['manifest_path']))
    return paths


def record_prompt(row, cache):
    """Return (text, source) of the prompt that one selected run received, or (None, reason)."""
    record = p.read(row['record_path'])
    native = c.prompt_key(record['job'])
    for path in manifest_candidates(row):
        if not path.is_file():
            continue
        if str(path) not in cache:
            cache[str(path)] = p.read(path)
        manifest = cache[str(path)]
        if manifest.get('protocol_sha256') == record.get('protocol_sha256') and native in manifest['protocol']['prompts']:
            return manifest['protocol']['prompts'][native], str(path)
    if row['condition'] != 'self_repair' and isinstance(record.get('prompt'), str):
        return c.prepared_prompt(record['prompt'], row['condition']), 'run_record'
    return None, 'no manifest prompt and no usable record prompt'


def canonical_key(row):
    return '__'.join((row['task'], row['platform'], row['prompt'], row['condition'], p.output_format(row['platform'])))


def extract(tasks, baseline=None):
    """Return (texts, provenance, unresolved) for the prompt keys of the given tasks."""
    runs = defaultdict(Counter)
    sources = defaultdict(lambda: defaultdict(set))
    unresolved = []
    cache = {}
    with Path(baseline or p.BASELINE).open() as stream:
        for row in csv.DictReader(stream):
            if row['task'] not in tasks:
                continue
            text, source = record_prompt(row, cache)
            if text is None:
                unresolved.append(dict(position_id=row['position_id'], reason=source))
                continue
            key = canonical_key(row)
            runs[key][text] += 1
            sources[key][text].add(source)
    texts, provenance = {}, {}
    for key, counter in sorted(runs.items()):
        ranked = counter.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            raise ValueError(f'{key}: two production texts have the same run count')
        texts[key] = ranked[0][0]
        provenance[key] = dict(sha256=p.text_sha(ranked[0][0]), selected_runs=sum(counter.values()),
                               texts=[dict(sha256=p.text_sha(text), runs=n, chars=len(text),
                                           sources=sorted(sources[key][text])) for text, n in ranked])
    return texts, provenance, unresolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--write', action='store_true', help=f'write the prompt folder {p.ADDED_PROMPTS}')
    args = parser.parse_args(argv)
    texts, provenance, unresolved = extract(set(p.ADDED20_TASKS))
    expected = set(p.prompt_keys(tasks=p.ADDED20_TASKS))
    several = {key: [t['runs'] for t in value['texts']] for key, value in provenance.items() if len(value['texts']) > 1}
    print(json.dumps(dict(keys=len(texts), expected_keys=len(expected), missing=sorted(expected - set(texts)),
                          unresolved_runs=len(unresolved), keys_with_several_texts=several), indent=1))
    if set(texts) != expected:
        raise SystemExit('Prompt keys differ from the 360 expected keys of the 20 added tasks')
    if not args.write:
        return 0
    folder = p.ADDED_PROMPTS
    folder.mkdir(parents=True, exist_ok=True)
    for key, text in texts.items():
        (folder / f'{key}.txt').write_text(text)
    c.save(folder.parent / 'provenance.json', dict(
        note='Canonical production prompt of each key of the 20 added tasks: the text of most selected runs.',
        baseline=str(p.BASELINE), baseline_sha256=c.sha(p.BASELINE), keys=provenance, unresolved_runs=unresolved))
    print(f'Wrote {len(texts)} prompts to {folder}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

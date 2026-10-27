"""Run and check the held-out few-shot examples (type 2 examples) of the final task bank.

Archive since 2026-09-30: the prompt level heldout_few_shot left the bank (decision record
2026-09-30_drop-heldout-level). No package renders a held-out prompt. The set and this checker stay for the record.

The held-out set is one fixed list of short examples on calculations that no bank task scores. Every task in a
language gets the same examples in the same order, except where set.json names a replacement for one task.
Folder: data/task_candidates/2026-09-29_heldout_examples/
    set.json                       {"examples": [id, ...], "replacements": {task_id: {"replace": id, "by": id,
                                    "reason": text}}, "keywords": {id: [word, ...]}}
    <id>/example.json              example_id, title, context, explanation, optional platform_notes,
                                   columns (the toy-data column names), output_rows [{term, se, integer}]
    <id>/example.py, example.R, example.do
    <id>/expected_python.json, expected_r.json, expected_stata.txt   outputs of real runs (written by run --write)
    overlap.json                   hand classification: for each bank task and example, does the example show the
                                   task's method, or a value (correct or wrong) of its scored decision
    overlap.md                     the table of overlap.json (written by overlap --write)
    validation.json                every check below (written by run --write)

Checks of each example:
    ran, format, agree      as in final_bank_examples.py, with the example's own task_id and rows
    leak                    against every bank task: no example value within 10 tolerances of a reference value
                            of the task, and no number literal in the code or prose equal to a reference value
    no_data, comments       the code reads no file and has at least 3 comment lines
    lines                   at most MAX_LINES code lines (chosen limit, fewshot-design Section 4.1)
    words                   no banned prompt word, no bank task id, no "role of" variable mapping, and no toy-data
                            column name that a bank task file also uses
    unnamed                 the code does not name py4etrics or ivmodels

Command line (repository root):
    uv run python code/benchmark/heldout_examples.py run [--write]      run the examples and check them
    uv run python code/benchmark/heldout_examples.py check              rerun and compare with the stored outputs
    uv run python code/benchmark/heldout_examples.py scan               keyword hits in every task, for the review
    uv run python code/benchmark/heldout_examples.py overlap [--write]  validate overlap.json, write overlap.md
No provider calls.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import final_bank_examples as fe  # noqa: E402
import final_bank_prompts as fp  # noqa: E402

ROOT = fp.ROOT
SET_DIR = ROOT / 'data/task_candidates/2026-09-29_heldout_examples'
MAX_LINES = 40           # chosen limit: short examples (fewshot-design Section 4.1)
ROLE_MAPPING = re.compile(r'\brole of\b|\bplays the role\b', re.I)


def load_set(set_dir: Path = SET_DIR) -> dict[str, Any]:
    return json.loads((set_dir / 'set.json').read_text())


def load_example(example_id: str, set_dir: Path = SET_DIR) -> dict[str, Any]:
    """example.json with the fields that final_bank_examples.format_problems reads."""
    spec = json.loads((set_dir / example_id / 'example.json').read_text())
    spec['task_id'] = spec['example_id']
    spec['output_rows'] = [dict(r, se=bool(r.get('se')), integer=bool(r.get('integer'))) for r in spec['output_rows']]
    return spec


def examples_for(task_id: str, set_dir: Path = SET_DIR) -> list[str]:
    """The ordered example ids that the prompts of one task show."""
    spec = load_set(set_dir)
    ids = list(spec['examples'])
    rep = spec.get('replacements', {}).get(task_id)
    if rep:
        ids[ids.index(rep['replace'])] = rep['by']
    return ids


def prose_of(spec: dict[str, Any]) -> str:
    return ' '.join([spec['title'], spec['context'], spec['explanation']]
                    + list((spec.get('platform_notes') or {}).values()))


def bank_columns() -> set[str]:
    cols: set[str] = set()
    for folder in fp.package_dirs():
        for desc in fp.load_task(folder)['file_descriptions']:
            cols |= set(fp.described_columns(desc))
    return cols


# Running -------------------------------------------------------------------------------------

def run_script(folder: Path, lang: str) -> dict[str, Any]:
    """Run one example script in a fresh temporary folder (same capture rules as final_bank_examples.run_one)."""
    script = Path(folder) / fp.EXAMPLE_FILES[lang]
    with tempfile.TemporaryDirectory(prefix=f'heldout_{lang}_') as tmp:
        shutil.copyfile(script, Path(tmp) / script.name)
        cmd = {'python': [fe.PYTHON, script.name], 'r': [fe.RSCRIPT, '--vanilla', script.name],
               'stata': [fe.STATA, '-b', 'do', script.name]}[lang]
        start = time.time()
        try:
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=fe.TIMEOUT)
        except subprocess.TimeoutExpired:
            return {'returncode': None, 'output': None, 'raw': None, 'error': 'timeout'}
        out: dict[str, Any] = {'returncode': proc.returncode, 'wall_seconds': round(time.time() - start, 2)}
        if lang == 'stata':
            log = Path(tmp) / 'example.log'
            text = log.read_text(errors='replace') if log.is_file() else ''
            error = re.search(r'^r\((\d+)\);\s*$', text, re.M)
            lines = [re.sub(r'=\s+', '=', line.rstrip()) for line in text.splitlines() if line.startswith('ANSWER:')]
            out.update(output='\n'.join(lines) + '\n' if lines else None, raw=None,
                       error=f'Stata error r({error.group(1)})' if error else None)
        else:
            path = Path(tmp) / 'result.json'
            out['error'] = None if proc.returncode == 0 else (proc.stderr[-3000:] + proc.stdout[-2000:])
            try:
                raw = json.loads(path.read_text()) if path.is_file() else None
            except json.JSONDecodeError as exc:
                raw, out['error'] = None, f'result.json is not JSON: {exc}'
            out['raw'] = raw
            out['output'] = json.dumps(raw, indent=2, allow_nan=False) + '\n' if raw is not None else None
        return out


# Checks --------------------------------------------------------------------------------------

def text_problems(spec: dict[str, Any], code_by_lang: dict[str, str], columns: set[str],
                  task_ids: list[str]) -> list[str]:
    problems = []
    prose = prose_of(spec)
    for where, text in [('prose', prose)] + [(lang, code) for lang, code in code_by_lang.items()]:
        match = fp.BANNED.search(text)
        if match:
            problems.append(f'{where}: banned word "{match.group(0)}"')
        named = [t for t in task_ids if re.search(rf'\b{re.escape(t)}\b', text)]
        if named:
            problems.append(f'{where}: names bank task ids {named}')
        if ROLE_MAPPING.search(text):
            problems.append(f'{where}: maps variables to task variables ("role of")')
    shared = sorted(set(spec.get('columns', [])) & columns)
    if shared:
        problems.append(f'toy-data columns also used by bank task files: {shared}')
    return problems


def check_example(example_id: str, write: bool = False, set_dir: Path = SET_DIR) -> dict[str, Any]:
    folder = set_dir / example_id
    spec = load_example(example_id, set_dir)
    report: dict[str, Any] = {'example': example_id, 'languages': {}, 'problems': []}
    report['problems'] += [f'example.json: {p}' for p in fp.parts_problems(
        {k: spec.get(k) for k in ('title', 'context', 'explanation', 'platform_notes') if spec.get(k) is not None})]
    codes = {lang: (folder / fp.EXAMPLE_FILES[lang]).read_text() for lang in fp.LANGS}
    folders = fp.package_dirs()
    report['problems'] += text_problems(spec, codes, bank_columns(), [f.name for f in folders])
    rows_by_lang, values = {}, []
    for lang in fp.LANGS:
        code = codes[lang]
        run = run_script(folder, lang)
        rows, header = fe.parse_rows(lang, run)
        entry = {'returncode': run.get('returncode'), 'wall_seconds': run.get('wall_seconds'),
                 'error': run.get('error'), 'code_lines': len(code.strip('\n').splitlines()),
                 'comment_lines': sum(bool(fe.COMMENT[lang].match(line)) for line in code.splitlines())}
        report['languages'][lang] = entry
        if run.get('error') or run.get('returncode') != 0:
            report['problems'].append(f'{lang}: run failed: {(run.get("error") or "")[-800:]}')
        fmt = fe.format_problems(spec, lang, rows, header)
        report['problems'] += [f'{lang}: format: {p}' for p in fmt]
        if entry['comment_lines'] < fe.MIN_COMMENTS:
            report['problems'].append(f'{lang}: {entry["comment_lines"]} comment lines (at least {fe.MIN_COMMENTS})')
        if entry['code_lines'] > MAX_LINES:
            report['problems'].append(f'{lang}: {entry["code_lines"]} code lines (at most {MAX_LINES})')
        reader = fe.FILE_READER.search(code)
        if reader:
            report['problems'].append(f'{lang}: code reads a file ("{reader.group(0)}")')
        named = fe.UNNAMED_PACKAGES.search(code)
        if named:
            report['problems'].append(f'{lang}: code names {named.group(0)}')
        if rows is not None and not fmt:
            rows_by_lang[lang] = {r['term']: r for r in rows}
            for r in rows:
                for field in ('estimate', 'se', 'n_obs'):
                    if isinstance(r.get(field), (int, float)):
                        values.append((f'{lang}:{r["term"]}.{field}', float(r[field])))
        entry['values'] = rows_by_lang.get(lang)
        entry['output'] = run.get('output')
        if write and run.get('output'):
            (folder / fp.EXPECTED_FILES[lang]).write_text(run['output'])
    # Agreement across the three languages.
    worst: dict[str, Any] = {'rel_diff': 0.0}
    if len(rows_by_lang) == 3:
        for row in spec['output_rows']:
            for field in ('estimate', 'se', 'n_obs'):
                vals = [rows_by_lang[lang][row['term']].get(field) for lang in fp.LANGS]
                if all(v is None for v in vals):
                    continue
                if any(v is None for v in vals):
                    report['problems'].append(f'agree: {row["term"]}.{field} is missing in a language')
                    continue
                d = max(fe.rel_diff(float(a), float(b)) for a in vals for b in vals)
                if d > worst['rel_diff']:
                    worst = {'rel_diff': d, 'term': row['term'], 'field': field, 'values': vals}
        if worst['rel_diff'] > fe.AGREE_RTOL:
            report['problems'].append(f'agree: {worst["term"]}.{worst["field"]} differs by {worst["rel_diff"]:.3g}')
    else:
        report['problems'].append('agree: not all three languages produced valid output')
    report['agreement'] = dict(worst, limit=fe.AGREE_RTOL)
    # Leak check against every bank task.
    prose = prose_of(spec)
    per_task = {}
    for task_folder in folders:
        refs = fe.reference_values(task_folder, fp.load_task(task_folder))
        leaks, closest = fe.value_leaks(values, refs)
        literals = [label for label, value in refs if abs(value) >= 1e-3 and not float(value).is_integer()
                    and any(fe.literal_matches(text, value) for text in [prose, *codes.values()])]
        per_task[task_folder.name] = {'reference_values': len(refs), 'leaks': leaks, 'literals': literals,
                                      'closest_ratio': closest.get('ratio')}
        report['problems'] += [f'leak {task_folder.name}: {p}' for p in leaks[:5]]
        report['problems'] += [f'literal {task_folder.name}: code or prose contains {lab}' for lab in literals]
    report['leak_check'] = {'factor': fe.LEAK_FACTOR, 'tasks': len(per_task), 'example_values': len(values),
                            'tasks_with_leaks': sorted(t for t, v in per_task.items() if v['leaks'] or v['literals']),
                            'closest_ratio': min((v['closest_ratio'] for v in per_task.values()), default=None),
                            'per_task': per_task}
    report['ok'] = not report['problems']
    return report


def check_stored(example_id: str, set_dir: Path = SET_DIR) -> list[str]:
    problems = []
    for lang in fp.LANGS:
        path = set_dir / example_id / fp.EXPECTED_FILES[lang]
        if not path.is_file():
            problems.append(f'{example_id} {lang}: stored output missing')
        elif not fe.same_output(path.read_text(), run_script(set_dir / example_id, lang).get('output')):
            problems.append(f'{example_id} {lang}: rerun output differs from {path.name}')
    return problems


# Overlap table -------------------------------------------------------------------------------

def task_decision_space(folder: Path) -> tuple[str, list[str]]:
    """(decision text, variant descriptions) of one bank task: the variants are the other values of the decision."""
    task = json.loads((folder / 'task.json').read_text())
    decision = task.get('decision') or task.get('source_of_difficulty') or ''
    path = folder / 'variants.json'
    variants = json.loads(path.read_text()) if path.is_file() else {}
    names = list(variants) if isinstance(variants, dict) else [v.get('description') or v.get('name') for v in variants]
    return decision, [str(n) for n in names]


def scan(set_dir: Path = SET_DIR) -> dict[str, dict[str, list[str]]]:
    """Keyword hits of each example in each task's text, decision, and variant names."""
    keywords = load_set(set_dir)['keywords']
    hits: dict[str, dict[str, list[str]]] = {}
    for folder in fp.package_dirs():
        task = json.loads((folder / 'task.json').read_text())
        decision, variants = task_decision_space(folder)
        texts = {'task_text': task.get('task_text', ''), 'decision': decision}
        texts.update({f'variant: {v}': v for v in variants})
        for ex, words in keywords.items():
            found = [f'{where} ~ {w}' for where, text in texts.items() for w in words
                     if re.search(rf'\b{w}', text, re.I)]
            if found:
                hits.setdefault(folder.name, {})[ex] = found
    return hits


def overlap_problems(set_dir: Path = SET_DIR) -> list[str]:
    spec = load_set(set_dir)
    table = json.loads((set_dir / 'overlap.json').read_text())
    problems = []
    tasks = [f.name for f in fp.package_dirs()]
    pool = set(spec['examples']) | {r['by'] for r in spec.get('replacements', {}).values()}
    for task in tasks:
        row = table['tasks'].get(task)
        if row is None:
            problems.append(f'{task}: no row')
            continue
        for ex in pool:
            cell = row['examples'].get(ex)
            if cell is None or not {'method', 'decision_value'} <= set(cell):
                problems.append(f'{task}: no classification for {ex}')
                continue
            shown = examples_for(task, set_dir)
            if (cell['method'] or cell['decision_value']) and ex in shown:
                problems.append(f'{task}: {ex} overlaps the task and is still shown')
    extra = set(table['tasks']) - set(tasks)
    if extra:
        problems.append(f'rows for tasks outside the bank: {sorted(extra)}')
    return problems


def overlap_markdown(set_dir: Path = SET_DIR) -> str:
    spec = load_set(set_dir)
    table = json.loads((set_dir / 'overlap.json').read_text())
    pool = list(spec['examples']) + [r['by'] for r in spec.get('replacements', {}).values()
                                     if r['by'] not in spec['examples']]
    yes = lambda cell: 'method' if cell['method'] else ('decision' if cell['decision_value'] else 'none')  # noqa: E731
    lines = ['# Overlap of the held-out examples with the bank tasks', '',
             'Generated by `code/benchmark/heldout_examples.py overlap --write` from `overlap.json`.',
             'A cell is "method" when the example uses the task\'s estimator or test, "decision" when the example '
             'shows a value (correct or wrong) of the task\'s scored decision, and "none" otherwise.',
             'The other values of a decision are the variants in the task\'s `variants.json`.', '',
             '| Task | Decision (task.json) | Variants | ' + ' | '.join(f'`{e}`' for e in pool) + ' | Shown | Note |',
             '|---|---|---|' + '---|' * len(pool) + '---|---|']
    for folder in fp.package_dirs():
        row = table['tasks'][folder.name]
        decision, variants = task_decision_space(folder)
        cells = ' | '.join(yes(row['examples'][e]) for e in pool)
        shown = ', '.join(examples_for(folder.name, set_dir))
        note = row.get('note', '').replace('|', '/')
        lines.append(f'| `{folder.name}` | {decision.replace("|", "/")} | {len(variants)} | {cells} | {shown} | {note} |')
    return '\n'.join(lines) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--write', action='store_true')
    sub.add_parser('check')
    sub.add_parser('scan')
    ov = sub.add_parser('overlap')
    ov.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    spec = load_set()
    pool = list(spec['examples']) + [r['by'] for r in spec.get('replacements', {}).values()]
    pool = list(dict.fromkeys(pool))
    if args.command == 'run':
        reports = [check_example(e, write=args.write) for e in pool]
        for rep in reports:
            print(json.dumps({'example': rep['example'], 'ok': rep['ok'], 'problems': rep['problems'][:20],
                              'agreement': rep['agreement'].get('rel_diff'),
                              'closest_leak_ratio': rep['leak_check']['closest_ratio'],
                              'values': {lang: {t: [v.get('estimate'), v.get('se'), v.get('n_obs')]
                                                for t, v in (e.get('values') or {}).items()}
                                         for lang, e in rep['languages'].items()}}, indent=1, default=str))
        if args.write:
            summary = {'set': spec['examples'], 'replacements': spec.get('replacements', {}),
                       'bank_tasks': len(fp.package_dirs()), 'ok': all(r['ok'] for r in reports),
                       'overlap_problems': overlap_problems(), 'examples': reports}
            (SET_DIR / 'validation.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
        return 0 if all(r['ok'] for r in reports) else 1
    if args.command == 'check':
        bad = [p for e in pool for p in check_stored(e)]
        print('\n'.join(bad) if bad else f'{len(pool)} examples reproduce their stored outputs')
        return 1 if bad else 0
    if args.command == 'scan':
        print(json.dumps(scan(), indent=1))
        return 0
    problems = overlap_problems()
    print('\n'.join(problems) if problems else 'overlap.json covers every bank task and example')
    if args.write and not problems:
        (SET_DIR / 'overlap.md').write_text(overlap_markdown())
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())

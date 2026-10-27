"""Run and check the few-shot examples of the final task bank.

Each package of the final bank (see final_bank_prompts.py) has an examples/ folder:
    block_parts.json    {"title", "context", "explanation", optional "platform_notes": {language: sentence},
                         optional "agreement_rtol" and "agreement_note"}
    example.py, example.R, example.do
                        one example of the task's calculation on separate toy data. Each script generates its
                        data with the shared Park-Miller stream (the same numbers in the three languages),
                        reads no file, and writes the task's output format: result.json for Python and R,
                        ANSWER lines in the log for Stata.
This module runs the three scripts in temporary folders and writes, next to them:
    expected_python.json, expected_r.json, expected_stata.txt   the outputs that the block shows
    validation.json                                              every check below and the values

Checks (a failed check is a problem; `run --write` still writes the files so that they can be inspected):
    ran       each script exits without error and writes its output
    format    the task's task_id, platform, row names in order, se null or a number as the task says,
              integer n_obs, integer values in integer rows
    agree     the three languages agree: relative difference <= AGREE_RTOL (or the block's agreement_rtol)
    leak      no example value (estimate, se, n_obs) lies within 10 times the tolerance
              max(1e-8, 0.01 |reference|) of any reference value of the task (targets and conventions);
              no number literal in the code or the block prose equals a reference value at its printed precision
    no_data   the code names no task file and calls no file reader
    comments  each script has at least MIN_COMMENTS comment lines
    routine   the code matches a standard-routine pattern of routines.json (classes builtin, major_package, or
              method_author_package); languages without one are recorded
    unnamed   the code does not name py4etrics or ivmodels, which are installed but never named

Command line (repository root):
    uv run python code/benchmark/final_bank_examples.py run <task> [<task> ...] [--write]
    uv run python code/benchmark/final_bank_examples.py check [<task> ...]   rerun and compare with stored files
    uv run python code/benchmark/final_bank_examples.py summary
No provider calls.
"""
from __future__ import annotations

import argparse
import json
import math
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
import final_bank_prompts as fp  # noqa: E402

ROOT = fp.ROOT
PYTHON = str(ROOT / '.venv/bin/python3')
RSCRIPT = shutil.which('Rscript') or '/usr/local/bin/Rscript'
STATA = shutil.which('stata-se') or '/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se'
AGREE_RTOL = 1e-6        # chosen limit: the three languages compute the same numbers on the same data
LEAK_FACTOR = 10         # chosen limit: example values stay 10 tolerances away from every reference value
STORED_RTOL = 1e-8       # chosen limit: a rerun reproduces the stored expected output
MIN_COMMENTS = 3
NOT_STANDARD = ('small_package', 'generic', 'none')
# Installed for the agents, never named in a prompt or an example (decision 2026-09-29).
UNNAMED_PACKAGES = re.compile(r'\b(py4etrics|ivmodels)\b')
TIMEOUT = 900
NUMBER = re.compile(r'(?<![A-Za-z_])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
ANSWER = re.compile(r'^ANSWER:\s*([A-Za-z0-9_]+)\s+estimate=\s*(\S+)\s+se=\s*(\S+)\s+n_obs=\s*(\S+)\s*$')
FILE_READER = re.compile(r'read_csv|read\.csv|read\.table|fread\s*\(|read_excel|import\s+delimited|insheet|'
                         r'\binfile\b|np\.loadtxt|genfromtxt|\.csv\b|\.dta\b|open\([^)]*[\'"]r[\'"]')
COMMENT = {'python': re.compile(r'^\s*#'), 'r': re.compile(r'^\s*#'), 'stata': re.compile(r'^\s*(\*|//)')}


def examples_dir(folder: Path) -> Path:
    return Path(folder) / 'examples'


def rel_diff(a: float, b: float) -> float:
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def tolerance(reference: float) -> float:
    return max(1e-8, 0.01 * abs(reference))


# Running -------------------------------------------------------------------------------------

def run_one(folder: Path, lang: str) -> dict[str, Any]:
    """Run one example script in a fresh temporary folder. Returns the rendered output and raw values."""
    script = examples_dir(folder) / fp.EXAMPLE_FILES[lang]
    with tempfile.TemporaryDirectory(prefix=f'fbex_{lang}_') as tmp:
        target = Path(tmp) / script.name
        shutil.copyfile(script, target)
        cmd = {'python': [PYTHON, script.name], 'r': [RSCRIPT, '--vanilla', script.name],
               'stata': [STATA, '-b', 'do', script.name]}[lang]
        start = time.time()
        try:
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=TIMEOUT)
            code, stderr = proc.returncode, proc.stderr[-3000:] + proc.stdout[-2000:]
        except subprocess.TimeoutExpired:
            return {'returncode': None, 'timed_out': True, 'output': None, 'rows': None, 'error': 'timeout'}
        out = {'returncode': code, 'timed_out': False, 'wall_seconds': round(time.time() - start, 2)}
        if lang == 'stata':
            log = Path(tmp) / 'example.log'
            text = log.read_text(errors='replace') if log.is_file() else ''
            error = re.search(r'^r\((\d+)\);\s*$', text, re.M)
            lines = [re.sub(r'=\s+', '=', line.rstrip()) for line in text.splitlines() if line.startswith('ANSWER:')]
            out['output'] = '\n'.join(lines) + '\n' if lines else None
            out['error'] = f'Stata error r({error.group(1)}): ' + text[max(0, error.start() - 1500):error.end()] if error else None
            out['raw'] = None
        else:
            path = Path(tmp) / 'result.json'
            out['error'] = None if code == 0 else stderr
            try:
                raw = json.loads(path.read_text()) if path.is_file() else None
            except json.JSONDecodeError as exc:
                raw, out['error'] = None, f'result.json is not JSON: {exc}'
            out['raw'] = raw
            out['output'] = json.dumps(raw, indent=2, allow_nan=False) + '\n' if raw is not None else None
        return out


def parse_rows(lang: str, run: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """Rows [{term, estimate, se, n_obs}] and the header {task_id, platform} (None for Stata)."""
    if run.get('output') is None:
        return None, None
    if lang == 'stata':
        rows = []
        for line in run['output'].splitlines():
            m = ANSWER.match(line)
            if not m:
                rows.append({'term': None, 'line': line})
                continue
            term, est, se, n = m.groups()
            rows.append({'term': term, 'estimate': _num(est), 'se': None if se == '.' else _num(se),
                         'n_obs': _num(n), 'raw_se': se})
        return rows, None
    raw = run['raw']
    if not isinstance(raw, dict) or not isinstance(raw.get('estimates'), list):
        return None, None
    return raw['estimates'], {'task_id': raw.get('task_id'), 'platform': raw.get('platform')}


def _num(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


# Checks --------------------------------------------------------------------------------------

def format_problems(task: dict[str, Any], lang: str, rows, header) -> list[str]:
    if rows is None:
        return ['no output']
    problems = []
    terms = [r['term'] for r in task['output_rows']]
    got = [r.get('term') for r in rows]
    if got != terms:
        problems.append(f'rows {got} differ from the task rows {terms}')
    if header is not None:
        if header.get('task_id') != task['task_id']:
            problems.append(f'task_id {header.get("task_id")} differs from {task["task_id"]}')
        if header.get('platform') != lang:
            problems.append(f'platform {header.get("platform")} differs from {lang}')
    spec = {r['term']: r for r in task['output_rows']}
    for row in rows:
        s = spec.get(row.get('term'))
        if s is None:
            continue
        est, se, n = row.get('estimate'), row.get('se'), row.get('n_obs')
        if not isinstance(est, (int, float)) or isinstance(est, bool) or not math.isfinite(est):
            problems.append(f'{row["term"]}: estimate is not a finite number')
        elif s['integer'] and float(est) != round(float(est)):
            problems.append(f'{row["term"]}: estimate of an integer row is not an integer')
        if s['se'] and (not isinstance(se, (int, float)) or not math.isfinite(se)):
            problems.append(f'{row["term"]}: se must be a number')
        if not s['se'] and se is not None:
            problems.append(f'{row["term"]}: se must be null')
        if not isinstance(n, (int, float)) or isinstance(n, bool) or float(n) != round(float(n)):
            problems.append(f'{row["term"]}: n_obs is not an integer')
        elif lang != 'stata' and not isinstance(n, int):
            problems.append(f'{row["term"]}: n_obs must be written as an integer')
    return problems


def reference_values(folder: Path, task: dict[str, Any]) -> list[tuple[str, float]]:
    """[(label, value)] of every reference value of the task: targets and accepted conventions."""
    values = []
    for t in task.get('targets', []):
        for field in ('estimate', 'se', 'n_obs'):
            v = t.get(field)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                values.append((f'{t["term"]}.{field}', float(v)))
    scoring = Path(folder) / 'scoring.json'
    if scoring.is_file():
        spec = json.loads(scoring.read_text())
        for conv in spec.get('conventions') or []:
            for r in conv.get('references', []):
                if isinstance(r.get('reference'), (int, float)):
                    values.append((f'{conv["name"]}:{r["term"]}.{r["field"]}', float(r['reference'])))
    return values


def value_leaks(values: list[tuple[str, float]], refs: list[tuple[str, float]]) -> tuple[list[str], dict]:
    """Example values within LEAK_FACTOR tolerances of a reference value, and the closest pair."""
    leaks, closest = [], {'ratio': math.inf}
    for label, v in values:
        for rlabel, r in refs:
            ratio = abs(v - r) / tolerance(r)
            if ratio < closest['ratio']:
                closest = {'ratio': ratio, 'example': label, 'example_value': v, 'reference': rlabel,
                           'reference_value': r}
            if ratio <= LEAK_FACTOR:
                leaks.append(f'{label}={v!r} is {ratio:.2f} tolerances from reference {rlabel}={r!r}')
    return leaks, closest


def literal_matches(text: str, value: float) -> bool:
    """True when a number in text with at least four significant digits equals value at its printed precision."""
    for token in NUMBER.findall(text):
        mantissa, _, exponent = token.lower().partition('e')
        if len(mantissa.lstrip('-').replace('.', '').lstrip('0')) < 4:
            continue
        decimals = len(mantissa.partition('.')[2])
        step = 10.0 ** (int(exponent or 0) - decimals)
        if abs(float(token) - value) <= 0.5 * step * (1 + 1e-9):
            return True
    return False


def routine_status(folder: Path, lang: str, code: str) -> dict[str, Any]:
    """Standard-routine use of an example script.

    Only patterns of a standard class count (builtin, major_package, method_author_package in pattern_classes).
    A "small_package" routine (for example py4etrics) is installed for the agents but is not standard, and a
    "generic" routine (a generic likelihood or GMM class) is a framework. An example never needs either
    (decision record 2026-09-29_routine-classes).
    """
    spec = json.loads((Path(folder) / 'routines.json').read_text())['languages'].get(lang, {})
    classes = spec.get('pattern_classes') or {}
    patterns = {pattern: name for pattern, name in (spec.get('patterns') or {}).items()
                if classes.get(name) not in NOT_STANDARD}
    if not patterns:
        return {'routine_exists': False, 'used': None, 'note': spec.get('note', '')}
    used = [name for pattern, name in patterns.items() if re.search(pattern, code)]
    return {'routine_exists': True, 'used': used, 'routine': spec.get('routine')}


def check_task(task_id: str, write: bool = False) -> dict[str, Any]:
    folder = fp.package_dir(task_id)
    task = fp.load_task(folder)
    ex = examples_dir(folder)
    report: dict[str, Any] = {'task': task_id, 'bank': folder.parent.name, 'languages': {}, 'problems': []}
    parts_path = ex / 'block_parts.json'
    parts = json.loads(parts_path.read_text()) if parts_path.is_file() else None
    report['problems'] += [f'block_parts: {p}' for p in fp.parts_problems(parts)] if parts is not None else [
        'block_parts.json missing']
    refs = reference_values(folder, task)
    all_values = []
    rows_by_lang = {}
    prose = ' '.join([parts.get('title', ''), parts.get('context', ''), parts.get('explanation', '')]
                     + list((parts.get('platform_notes') or {}).values())) if parts else ''
    for lang in fp.LANGS:
        script = ex / fp.EXAMPLE_FILES[lang]
        entry: dict[str, Any] = {}
        report['languages'][lang] = entry
        if not script.is_file():
            report['problems'].append(f'{lang}: {script.name} missing')
            continue
        code = script.read_text()
        run = run_one(folder, lang)
        rows, header = parse_rows(lang, run)
        entry.update(returncode=run.get('returncode'), wall_seconds=run.get('wall_seconds'),
                     error=run.get('error'), code_lines=len(code.strip('\n').splitlines()),
                     comment_lines=sum(bool(COMMENT[lang].match(line)) for line in code.splitlines()))
        if run.get('error') or run.get('returncode') not in (0,):
            report['problems'].append(f'{lang}: run failed: {(run.get("error") or "")[-800:]}')
        fmt = format_problems(task, lang, rows, header)
        report['problems'] += [f'{lang}: format: {p}' for p in fmt]
        if entry['comment_lines'] < MIN_COMMENTS:
            report['problems'].append(f'{lang}: {entry["comment_lines"]} comment lines (at least {MIN_COMMENTS})')
        for name in task['files']:
            if name in code or Path(name).stem + '.csv' in code:
                report['problems'].append(f'{lang}: code names the task file {name}')
        reader = FILE_READER.search(code)
        if reader:
            report['problems'].append(f'{lang}: code reads a file ("{reader.group(0)}")')
        named = UNNAMED_PACKAGES.search(code)
        if named:
            report['problems'].append(f'{lang}: code names {named.group(0)}, which no prompt may name')
        entry['routine'] = routine_status(folder, lang, code)
        if entry['routine']['routine_exists'] and not entry['routine']['used']:
            report['problems'].append(f'{lang}: no standard routine of routines.json is used')
        for label, value in refs:
            if abs(value) >= 1e-3 and not float(value).is_integer() and literal_matches(code, value):
                report['problems'].append(f'{lang}: code contains the reference value {label}')
        if rows is not None and not fmt:
            rows_by_lang[lang] = {r['term']: r for r in rows}
            for r in rows:
                for field in ('estimate', 'se', 'n_obs'):
                    if isinstance(r.get(field), (int, float)):
                        all_values.append((f'{lang}:{r["term"]}.{field}', float(r[field])))
        entry['values'] = rows_by_lang.get(lang)
        entry['output'] = run.get('output')
        if write and run.get('output'):
            (ex / fp.EXPECTED_FILES[lang]).write_text(run['output'])
    for label, value in refs:
        if abs(value) >= 1e-3 and not float(value).is_integer() and literal_matches(prose, value):
            report['problems'].append(f'block prose contains the reference value {label}')
    # Agreement across languages.
    rtol = float(parts.get('agreement_rtol', AGREE_RTOL)) if parts else AGREE_RTOL
    worst = {'rel_diff': 0.0}
    if len(rows_by_lang) == 3:
        for row in task['output_rows']:
            for field in ('estimate', 'se', 'n_obs'):
                vals = [rows_by_lang[lang][row['term']].get(field) for lang in fp.LANGS]
                if all(v is None for v in vals):
                    continue
                if any(v is None for v in vals):
                    report['problems'].append(f'agree: {row["term"]}.{field} is missing in a language')
                    continue
                d = max(rel_diff(float(a), float(b)) for a in vals for b in vals)
                if d > worst['rel_diff']:
                    worst = {'rel_diff': d, 'term': row['term'], 'field': field, 'values': vals}
        if worst['rel_diff'] > rtol:
            report['problems'].append(f'agree: {worst.get("term")}.{worst.get("field")} differs across languages '
                                      f'by {worst["rel_diff"]:.3g} (limit {rtol:g}): {worst.get("values")}')
    else:
        report['problems'].append('agree: not all three languages produced valid output')
    report['agreement'] = dict(worst, limit=rtol, note=(parts or {}).get('agreement_note'))
    leaks, closest = value_leaks(all_values, refs)
    report['problems'] += [f'leak: {p}' for p in leaks[:20]]
    report['leak_check'] = {'factor': LEAK_FACTOR, 'reference_values': len(refs), 'example_values': len(all_values),
                            'leaks': len(leaks), 'closest': closest}
    report['ok'] = not report['problems']
    if write:
        (ex / 'validation.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    return report


def check_stored(task_id: str) -> list[str]:
    """Rerun the examples and compare with the stored expected outputs."""
    folder = fp.package_dir(task_id)
    problems = []
    for lang in fp.LANGS:
        stored_path = examples_dir(folder) / fp.EXPECTED_FILES[lang]
        if not stored_path.is_file():
            problems.append(f'{lang}: stored output missing')
            continue
        stored, fresh = stored_path.read_text(), run_one(folder, lang).get('output')
        if not same_output(stored, fresh):
            problems.append(f'{lang}: rerun output differs from {stored_path.name}')
    return problems


def same_output(expected: str | None, actual: str | None, rtol: float = STORED_RTOL) -> bool:
    if expected is None or actual is None:
        return False
    if NUMBER.sub('#', expected.strip()) != NUMBER.sub('#', actual.strip()):
        return False
    return all(math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=1e-12)
               for a, b in zip(NUMBER.findall(expected), NUMBER.findall(actual)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('tasks', nargs='+')
    run.add_argument('--write', action='store_true')
    chk = sub.add_parser('check')
    chk.add_argument('tasks', nargs='*')
    sub.add_parser('summary')
    args = parser.parse_args(argv)
    if args.command == 'run':
        bad = 0
        for task_id in args.tasks:
            rep = check_task(task_id, write=args.write)
            bad += not rep['ok']
            print(json.dumps({'task': task_id, 'ok': rep['ok'], 'problems': rep['problems'],
                              'agreement': rep['agreement'].get('rel_diff'),
                              'closest_leak_ratio': rep['leak_check']['closest'].get('ratio'),
                              'values': {lang: {t: [v.get('estimate'), v.get('se'), v.get('n_obs')]
                                                for t, v in (e.get('values') or {}).items()}
                                         for lang, e in rep['languages'].items()}}, indent=1, default=str))
        return 1 if bad else 0
    if args.command == 'check':
        tasks = args.tasks or [f.name for f in fp.package_dirs()]
        bad = {t: p for t in tasks if (p := check_stored(t))}
        print(json.dumps(bad, indent=1) if bad else f'{len(tasks)} tasks reproduce their stored outputs')
        return 1 if bad else 0
    for folder in fp.package_dirs():
        path = examples_dir(folder) / 'validation.json'
        if not path.is_file():
            print(f'{folder.name}: no validation.json')
            continue
        rep = json.loads(path.read_text())
        print(f'{folder.name}: ok={rep["ok"]} agree={rep["agreement"].get("rel_diff", 0):.2g} '
              f'closest={rep["leak_check"]["closest"].get("ratio", float("inf")):.1f} problems={len(rep["problems"])}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

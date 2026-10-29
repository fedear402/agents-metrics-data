"""Run the reference programs of the 40 tasks and compare their output with tasks/<task>/reference.json.

Run from the repository root:

    python code/run_references.py                       # every task, every language found on this machine
    python code/run_references.py --tasks ols_robust var_irf --languages python r
    python code/run_references.py --stata /path/to/stata-se --out reference_runs.csv
    python code/run_references.py --generators          # also regenerate every task input and compare bytes

The programs are in source/reference_implementations/manual/<task>/: manual_python.py, manual_r.R, and
manual_stata.do. Each one reads the task input from tasks/<task>/ of the repository. Python programs run with
the interpreter that runs this script, which needs the packages of code/requirements.txt. R programs run when
Rscript is on the PATH. Stata programs run when a Stata executable is found (--stata, the STATA environment
variable, the PATH, or the usual install folders). A Stata program runs in a temporary copy of its task
folder, so the repository receives no log file. --keep-logs DIR saves each Stata log there, without the
license lines of the Stata banner.

Each value is compared with the reference under the primary rule of the paper: an estimate passes when it is
within max(0.01 * |reference|, 0.005) of the reference, and a count term must be equal. A standard error that
the primary outcome scores (code/scoring_spec.json) passes when it is within the task tolerance of one
accepted value. A program passes when it reports every reference term and every value passes. The table also
gives the largest relative and absolute differences of the estimates and the largest relative difference
of the scored standard errors. The original-task Stata programs print 6 decimals, so a small value such as a
p-value can show a large relative difference with an absolute difference below 1e-6. The exit code is 1
when a program fails or stops with an error.

--generators also regenerates the task inputs in a temporary folder and compares them byte for byte with the
files in tasks/. The 20 added tasks have source/reference_implementations/manual/<task>/generate_data.py. The
inputs of the 20 original tasks come from the five source/code/benchmark/dgp/generate_data*.py scripts, which
write into a data/ folder next to a copy of the script.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = {'python': 'manual_python.py', 'r': 'manual_r.R', 'stata': 'manual_stata.do'}
EST_REL, EST_ABS = 0.01, 0.005  # the primary estimate tolerance of the paper
TIMEOUT = 1800  # seconds per program; chosen limit
EXACT_TERMS = {  # count terms that must equal the reference
    'cooks_distance': {f'rank{rank}_obs_id' for rank in range(1, 6)},
    'broad_ar_confidence_set': {'accepts_zero', 'unbounded'},
}
LICENSE_LINES = re.compile(r"^[ \t]*(?:Stata[ \t]+license|Serial[ \t]+number|Licensed[ \t]+to):.*\n?"
                           r"(?:[ \t]+\S.*\n?)*", re.MULTILINE | re.IGNORECASE)
ANSWER = re.compile(r"^\s*ANSWER:\s+(\S+)\s+estimate\s*=\s*(\S+)(?:\s+se\s*=\s*(\S+))?", re.MULTILINE)
NUMBER = r'(-?\d*\.?\d+(?:[eE][-+]?\d+)?)'


def number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def stata_works(executable):
    """True when the executable runs a one-line do-file in batch mode."""
    with tempfile.TemporaryDirectory(prefix='reference_stata_check_') as tmp:
        (Path(tmp) / 'check.do').write_text('display 1 + 1\n')
        try:
            subprocess.run([executable, '/e' if os.name == 'nt' else '-b', 'do', 'check.do'], cwd=tmp,
                           capture_output=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return (Path(tmp) / 'check.log').is_file()


def find_stata(explicit=None):
    """The first Stata executable that runs: --stata, STATA, the PATH, then the usual install folders."""
    candidates = [explicit, os.environ.get('STATA')]
    candidates += [shutil.which(name) for name in ('stata-mp', 'stata-se', 'stata', 'StataMP-64', 'StataSE-64',
                                                   'Stata-64')]
    for pattern in ('/Applications/StataNow/Stata*.app/Contents/MacOS/stata-*',
                    '/Applications/Stata*/Stata*.app/Contents/MacOS/stata-*', '/usr/local/stata*/stata-*',
                    'C:/Program Files/Stata*/Stata*-64.exe'):
        candidates += sorted(glob.glob(pattern), reverse=True)
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).is_file() and os.access(candidate, os.X_OK) and stata_works(candidate):
            return str(candidate)
    return None


def json_object(text):
    """The JSON object printed on stdout (from the first '{' to the last '}')."""
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        raise ValueError('no JSON object on stdout')
    return json.loads(text[start:end + 1])


def stata_output(log):
    """ANSWER lines, or the JSON_RESULT line of the earlier manual programs."""
    text = re.sub(r'\n>\s?', '', log)
    answers = {m[1]: (m[2], m[3]) for m in ANSWER.finditer(text)}
    if answers:
        rows = [dict(term=term, estimate=est, se=None if se in (None, '.') else se)
                for term, (est, se) in answers.items()]
        return dict(estimates=rows)
    lines = [line for line in text.splitlines() if line.startswith('JSON_RESULT:')]
    if not lines:
        raise ValueError('no ANSWER or JSON_RESULT line in the log')
    values = {key: float(v) for key, v in re.findall(r'"([A-Za-z0-9_]+)":"?\s*' + NUMBER, lines[-1])}
    for key, inner in re.findall(r'"([A-Za-z0-9_]+)":\[([^\]]*)\]', lines[-1]):
        values[key] = [float(v.strip().strip('"')) for v in inner.split(',') if v.strip().strip('"')]
    return values


def term_values(task, raw):
    """{term: value, 'term:se': se} from either output form."""
    if isinstance(raw.get('estimates'), list):
        out = {}
        for row in raw['estimates']:
            if isinstance(row, dict) and isinstance(row.get('term'), str):
                out[row['term']] = row.get('estimate')
                if number(row.get('se')) is not None:
                    out[row['term'] + ':se'] = row['se']
        return out
    # Flat keys of the manual programs of the original tasks.
    out = {}
    for key, value in raw.items():
        if isinstance(value, list) or number(value) is None:
            continue
        if key.startswith('se_'):
            base = {'x_cluster_firm': 'x'}.get(key[3:], key[3:])
            out[base + ':se'] = value
        elif key.startswith('coef_'):
            out[key[5:]] = value
        elif key == 'att_uniform':
            out['att'] = value
        elif key == 'se':
            out['forecast:se'] = value
        else:
            out[key] = value
    if task == 'cooks_distance' and 'top5_obs_ids' in raw:
        for i, (row, value) in enumerate(zip(raw['top5_obs_ids'], raw['top5_cooks_d']), start=1):
            out[f'rank{i}_obs_id'] = row
            out[f'rank{i}_cooks_d'] = value
    return out


def relative(value, ref):
    return abs(value - ref) / abs(ref) if ref else abs(value - ref)


def compare(task, values, reference, spec):
    """(passed, max relative and absolute estimate differences, max relative SE difference, notes)."""
    rules = spec.get('tasks', {}).get(task, {})
    exact = EXACT_TERMS.get(task, set())
    se_terms = rules.get('se_terms', {}) if rules.get('se_scored') else {}
    notes, est_rel, est_abs, se_rel, passed = [], 0.0, 0.0, None, True
    for row in reference['estimates']:
        term, ref = row['term'], number(row['estimate'])
        got = number(values.get(term))
        if got is None:
            notes.append(f'{term}: not reported')
            passed = False
            continue
        if term in exact:
            ok = got.is_integer() and ref.is_integer() and got == ref
        else:
            ok = abs(got - ref) <= max(EST_REL * abs(ref), EST_ABS)
        est_rel, est_abs = max(est_rel, relative(got, ref)), max(est_abs, abs(got - ref))
        if not ok:
            notes.append(f'{term}: {got!r} against {ref!r}')
            passed = False
        if term in se_terms:
            se = number(values.get(term + ':se'))
            if se is None:
                notes.append(f'{term}: standard error not reported')
                continue
            rule = se_terms[term]
            gaps = [relative(se, a) for a in rule['accepted_values']]
            ok_se = any(abs(se - a) <= rule['rel_tol'] * abs(a) + rule.get('abs_floor', 0.0)
                        for a in rule['accepted_values'])
            se_rel = min(gaps) if se_rel is None else max(se_rel, min(gaps))
            if not ok_se:
                notes.append(f'{term} standard error: {se!r} against {rule["accepted_values"]}')
                passed = False
    return passed, est_rel, est_abs, se_rel, notes


def run_program(task, language, program, root, stata, keep_logs):
    """Run one program; return (raw output, error message)."""
    if language == 'python':
        proc = subprocess.run([sys.executable, str(program)], cwd=root, capture_output=True, text=True,
                              timeout=TIMEOUT)
        if proc.returncode:
            return None, (proc.stderr.strip().splitlines() or ['exit ' + str(proc.returncode)])[-1]
        return json_object(proc.stdout), None
    if language == 'r':
        proc = subprocess.run(['Rscript', str(program)], cwd=root, capture_output=True, text=True, timeout=TIMEOUT)
        if proc.returncode:
            return None, (proc.stderr.strip().splitlines() or ['exit ' + str(proc.returncode)])[-1]
        return json_object(proc.stdout), None
    with tempfile.TemporaryDirectory(prefix='reference_stata_') as tmp:
        work = Path(tmp)
        relative_program = Path('source/reference_implementations/manual') / task / program.name
        (work / relative_program).parent.mkdir(parents=True)
        shutil.copyfile(program, work / relative_program)
        shutil.copytree(root / 'tasks' / task, work / 'tasks' / task)
        flag = '/e' if os.name == 'nt' else '-b'
        subprocess.run([stata, flag, 'do', relative_program.as_posix()], cwd=work, capture_output=True, text=True,
                       timeout=TIMEOUT)
        log_path = work / (program.stem + '.log')
        if not log_path.is_file():
            return None, 'Stata wrote no log'
        log = LICENSE_LINES.sub('', log_path.read_text(errors='replace'))
        if keep_logs:
            target = Path(keep_logs) / task
            target.mkdir(parents=True, exist_ok=True)
            (target / log_path.name).write_text(log)
        error = re.search(r'^r\((\d+)\);', log, re.MULTILINE)
        if error:
            return None, f'Stata error r({error[1]})'
        return stata_output(log), None


def check_generators(root, programs, tasks):
    """[(generator, task/file, status)]: regenerate every input and compare it with tasks/<task>/<file>."""
    by_name = {}
    for task in tasks:
        for path in sorted((root / 'tasks' / task).glob('*.csv')):
            by_name.setdefault(path.name, []).append(path)
    rows = []
    with tempfile.TemporaryDirectory(prefix='reference_inputs_') as tmp:
        work = Path(tmp)
        jobs = []  # (label, command, working folder, output folder, task or None)
        for task in tasks:
            script = programs / task / 'generate_data.py'
            if script.is_file():
                (work / task).mkdir()
                jobs.append((f'manual/{task}/generate_data.py', [sys.executable, str(script), str(work / task)],
                             root, work / task, task))
        for script in sorted((root / 'source/code/benchmark/dgp').glob('generate_data*.py')):
            folder = work / ('dgp_' + script.stem)
            (folder / 'data').mkdir(parents=True)
            (folder / 'reference').mkdir()
            shutil.copyfile(script, folder / script.name)
            jobs.append((f'dgp/{script.name}', [sys.executable, script.name], folder, folder / 'data', None))
        for label, command, cwd, out, task in jobs:
            proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT)
            if proc.returncode:
                rows.append((label, '', 'error: ' + (proc.stderr.strip().splitlines() or ['exit'])[-1]))
                continue
            for path in sorted(out.glob('*.csv')):
                # A dgp script also writes inputs of tasks outside the 40-task sample; those have no target.
                targets = [root / 'tasks' / task / path.name] if task else by_name.get(path.name, [])
                for target in (x for x in targets if x.is_file()):
                    status = 'identical' if path.read_bytes() == target.read_bytes() else 'differs'
                    rows.append((label, f'{target.parent.name}/{target.name}', status))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT, help='repository root (default: the parent of code/)')
    parser.add_argument('--programs', type=Path, help='program folder (default: ROOT/source/reference_implementations/manual)')
    parser.add_argument('--tasks', nargs='+', help='task names (default: every folder in tasks/)')
    parser.add_argument('--languages', nargs='+', choices=sorted(PROGRAMS), default=sorted(PROGRAMS))
    parser.add_argument('--stata', help='Stata executable (default: STATA, the PATH, or the usual install folders)')
    parser.add_argument('--keep-logs', help='folder for the Stata logs, without license lines')
    parser.add_argument('--out', type=Path, help='write the result table to this CSV file')
    parser.add_argument('--generators', action='store_true', help='also regenerate and compare the task inputs')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    programs = (args.programs or root / 'source/reference_implementations/manual').resolve()
    spec_path = root / 'code/scoring_spec.json'
    spec = json.loads(spec_path.read_text()) if spec_path.is_file() else {}
    tasks = args.tasks or sorted(p.name for p in (root / 'tasks').iterdir() if (p / 'reference.json').is_file())
    available = {'python': True, 'r': shutil.which('Rscript') is not None}
    stata = find_stata(args.stata) if 'stata' in args.languages else None
    available['stata'] = stata is not None
    rows = []
    for task in tasks:
        reference = json.loads((root / 'tasks' / task / 'reference.json').read_text())
        for language in args.languages:
            program = programs / task / PROGRAMS[language]
            row = dict(task=task, language=language, status='', terms=len(reference['estimates']),
                       max_rel_diff_estimates='', max_abs_diff_estimates='', max_rel_diff_se='', notes='')
            if not program.is_file():
                row['status'] = 'no program'
            elif not available[language]:
                row['status'] = 'not installed'
            else:
                try:
                    raw, error = run_program(task, language, program, root, stata, args.keep_logs)
                except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
                    raw, error = None, f'{type(exc).__name__}: {exc}'
                if error:
                    row.update(status='error', notes=error)
                else:
                    passed, est_rel, est_abs, se_rel, notes = compare(task, term_values(task, raw), reference, spec)
                    row.update(status='pass' if passed else 'fail', max_rel_diff_estimates=f'{est_rel:.2e}',
                               max_abs_diff_estimates=f'{est_abs:.2e}',
                               max_rel_diff_se='' if se_rel is None else f'{se_rel:.2e}', notes='; '.join(notes))
            rows.append(row)
            print(f"{task:36s} {language:6s} {row['status']:13s} {row['max_rel_diff_estimates']:>9s} "
                  f"{row['max_abs_diff_estimates']:>9s} {row['max_rel_diff_se']:>9s} {row['notes']}", flush=True)
    if args.out:
        with args.out.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    counts = {s: sum(r['status'] == s for r in rows) for s in ('pass', 'fail', 'error', 'no program', 'not installed')}
    print('Programs: ' + ', '.join(f'{v} {k}' for k, v in counts.items()))
    bad_inputs = 0
    if args.generators:
        generated = check_generators(root, programs, tasks)
        for label, name, status in generated:
            print(f'{label:48s} {name:58s} {status}')
        covered = {name for _, name, status in generated if status == 'identical'}
        missing = sorted({f'{t}/{p.name}' for t in tasks for p in (root / 'tasks' / t).glob('*.csv')} - covered)
        bad_inputs = sum(status != 'identical' for *_, status in generated) + len(missing)
        print(f'Inputs: {len(covered)} identical, {bad_inputs} differ, fail, or have no generator'
              + (f" ({', '.join(missing)})" if missing else ''))
    return 1 if counts['fail'] or counts['error'] or bad_inputs else 0


if __name__ == '__main__':
    sys.exit(main())

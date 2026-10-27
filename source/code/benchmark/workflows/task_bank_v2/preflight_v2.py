"""Check the task bank v2 execution environment. This script installs nothing.

It reads environment_v2.json and checks, in the environment the collector uses:
- Python packages in the project .venv, and the pinned versions;
- R packages through ``Rscript --vanilla`` (the collector's R call);
- Stata commands through a batch do-file in a temporary folder under /private/tmp;
- the Sonnet shared-installation write guard file.

Exit code 0 means every "required" package is present and every pin holds.
Missing "recommended" or "decision" packages are listed but do not fail.

  uv run python code/benchmark/workflows/task_bank_v2/preflight_v2.py [--json PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
import harness  # noqa: E402
import production_queue as queue  # noqa: E402

PYTHON = ROOT / '.venv/bin/python'


def python_versions(modules):
    code = ('import importlib.metadata as m, importlib.util as u, json, sys\n'
            'names = json.loads(sys.argv[1])\n'
            'out = {}\n'
            'for n in names:\n'
            '    try: out[n] = m.version(n)\n'
            '    except m.PackageNotFoundError: out[n] = None\n'
            'print(json.dumps(out))\n')
    process = subprocess.run([str(PYTHON), '-c', code, json.dumps(modules)], capture_output=True, text=True)
    return json.loads(process.stdout) if process.returncode == 0 else {n: None for n in modules}


def python_imports(modules):
    code = ('import importlib.util as u, json, sys\n'
            'print(json.dumps({n: u.find_spec(n) is not None for n in json.loads(sys.argv[1])}))\n')
    process = subprocess.run([str(PYTHON), '-c', code, json.dumps(modules)], capture_output=True, text=True)
    return json.loads(process.stdout) if process.returncode == 0 else {n: False for n in modules}


def r_versions(packages):
    vector = ', '.join(json.dumps(p) for p in packages)
    expression = (f'for (p in c({vector})) cat(p, "\\t", if (requireNamespace(p, quietly = TRUE)) '
                  'as.character(packageVersion(p)) else "NA", "\\n", sep = "")')
    process = subprocess.run(['Rscript', '--vanilla', '-e', expression], capture_output=True, text=True, timeout=120)
    found = {}
    for line in process.stdout.splitlines():
        name, _, version = line.partition('\t')
        found[name.strip()] = None if version.strip() in ('', 'NA') else version.strip()
    return {p: found.get(p) for p in packages}


def stata_commands(commands):
    lines = ['clear all', 'set more off', 'set linesize 255']
    for name in commands:
        lines += [f'capture which {name}', f'display "PREFLIGHT {name} rc=" _rc']
        lines += [f'capture noisily which {name}']
    with tempfile.TemporaryDirectory(prefix='task_bank_v2_preflight_', dir='/private/tmp') as folder:
        do = Path(folder) / 'preflight.do'
        do.write_text('\n'.join(lines) + '\n')
        process = subprocess.run([harness.STATA_BIN, '-b', 'do', 'preflight.do'], cwd=folder,
                                 capture_output=True, text=True, timeout=120)
        log = (Path(folder) / 'preflight.log').read_text(errors='replace') if (Path(folder) / 'preflight.log').exists() else ''
    result = {}
    for name in commands:
        match = re.search(rf'^PREFLIGHT {re.escape(name)} rc=(\d+)', log, re.M)
        present = bool(match) and match.group(1) == '0'
        version = None
        block = log.split(f'which {name}', 2)[-1] if present else ''
        # Header forms seen on this machine: "*! Ver 1.71", "*! v1.81", "*! version 1.2.0".
        found = re.search(r'\*!\s*(?:version|ver|v)\.?\s*([0-9][0-9.]*)', block, re.I)
        if found:
            version = found.group(1)
        result[name] = dict(present=present, version=version)
    return dict(binary=str(harness.STATA_BIN), returncode=process.returncode, commands=result)


def version_at_least(found, minimum):
    def parts(text):
        return [int(x) for x in re.findall(r'\d+', text)]
    return found is not None and parts(found) >= parts(minimum)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--json', type=Path, help='save the result as JSON at this path')
    args = parser.parse_args(argv)
    spec = json.loads((HERE / 'environment_v2.json').read_text())
    packages = spec['packages']
    pins = spec['pinned_must_not_change']['python']
    report = dict(python_executable=str(PYTHON), checks=[], pins={}, failures=[])
    versions = python_versions(list(pins) + [p['package'] for p in packages if p['language'] == 'python'])
    for name, wanted in pins.items():
        ok = versions.get(name) == wanted
        report['pins'][name] = dict(wanted=wanted, found=versions.get(name), ok=ok)
        if not ok:
            report['failures'].append(f'pin changed: {name} {versions.get(name)} (wanted {wanted})')
    imports = python_imports([p['check_import'] for p in packages if p['language'] == 'python'])
    r_found = r_versions([p['check_library'] for p in packages if p['language'] == 'r'])
    stata = stata_commands([p['check_command'] for p in packages if p['language'] == 'stata'])
    report['stata'] = dict(binary=stata['binary'], returncode=stata['returncode'])
    for item in packages:
        language, name = item['language'], item['package']
        if language == 'python':
            present, found = imports.get(item['check_import'], False), versions.get(name)
        elif language == 'r':
            found = r_found.get(item['check_library'])
            present = found is not None
        else:
            state = stata['commands'].get(item['check_command'], {})
            present, found = state.get('present', False), state.get('version')
            if present and item.get('check_min_version'):
                present = version_at_least(found, item['check_min_version'])
        row = dict(language=language, package=name, priority=item['priority'], present=present, found=found,
                   wanted=item['version'], install=item['install'])
        report['checks'].append(row)
        if item['priority'] == 'required' and not present:
            report['failures'].append(f'required {language} package missing: {name}')
    guard = queue.SONNET_GUARD
    report['sonnet_guard'] = dict(path=str(guard), exists=guard.is_file(),
                                  denies=re.findall(r'subpath "([^"]+)"', guard.read_text()) if guard.is_file() else [])
    if not guard.is_file():
        report['failures'].append('Sonnet write guard file is missing')
    report['passed'] = not report['failures']
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + '\n')
    for row in report['checks']:
        state = 'present' if row['present'] else 'MISSING'
        print(f"{row['language']:6} {row['package']:16} {row['priority']:11} {state:8} found={row['found']}")
    print('pins:', 'all hold' if all(p['ok'] for p in report['pins'].values()) else report['pins'])
    print('sonnet guard:', report['sonnet_guard'])
    print('passed' if report['passed'] else 'FAILED: ' + '; '.join(report['failures']))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())

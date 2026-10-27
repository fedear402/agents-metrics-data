"""Score one run of a final-bank task against the package's scoring.json.

One scorer for the final task bank (data/task_candidates/2026-09-28_final_v2 and
data/task_candidates/2026-09-28_final_hard). It reads the package's scoring.json and the
run output, and it returns one pass or fail with the evidence for each check.

Rule (decision of 2026-09-26, Decision 2):
    a continuous check passes when |value - reference| <= max(abs_tol, rel_tol x |reference|),
    with rel_tol = 0.01 and abs_tol = 1e-8 (the max form of Python math.isclose);
    an integer check (counts, degrees of freedom, identifiers) passes only on exact equality;
    n_obs is diagnostic unless scoring.json sets n_obs.scored to true.
    A run passes when every check passes under the main references or under one complete
    accepted convention set.

Output channels:
    Python and R: result.json in the run folder, {"estimates": [{"term", "estimate", "se", "n_obs"}, ...]}.
    Stata: printed lines "ANSWER: <term> estimate=<number> se=<number or .> n_obs=<integer>" in
    run.log, in the captured stdout, or in files written in the final run. Stata "> " continuation
    lines are joined first. Two readings of the same scored field conflict when they differ by more
    than their printed precision; a conflict fails the run.

scoring.json layout of both final banks:
    {"task_id", "rule", "relative_tolerance", "absolute_floor", "point_of_task",
     "scored": [{"term", "field": "estimate"|"se"|"n_obs", "reference", "check": "exact"|"relative 1 percent, floor 1e-8"}],
     "diagnostic": [...], "p_values_scored": false, "success": "...", "wrong_variants": [...],
     "conventions": [{"name", "description", "references": [{"term", "field", "reference"}]}]  (optional)}
normalize_spec turns it into the internal form "final_bank_scoring/1":
    {"schema", "task_id", "rule", "rel_tol", "abs_tol",
     "checks": [{"term", "field": "estimate"|"se", "reference", "kind": "continuous"|"integer"}],
     "conventions": [...], "n_obs": {"scored": bool, "scored_terms": [...], "reference": {term: integer}}}
The internal form and the round-4 layout with a "targets" list are also read.
When the spec comes from a package folder, the input files in data/ are compared with the copies in the run
folder; a changed input file fails the run.

Command line:
    uv run python code/benchmark/final_bank_scorer.py --package <package dir> --language stata \
        --run-dir <run dir> [--stdout <file>] [--final-file <file> ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = 'final_bank_scoring/1'
REL_TOL = 0.01
ABS_TOL = 1e-8
FIELDS = ('estimate', 'se')
KINDS = ('continuous', 'integer')
MISSING_TOKENS = {'', '.', 'na', 'nan', 'null', 'none'}
P_VALUE = re.compile(r'(^|_)(p|pval|pvalue|p_value)($|_)|pval', re.I)

ANSWER_LINE = re.compile(r'^ANSWER:\s*([A-Za-z0-9_]+)((?:\s+(?:estimate|se|n_obs)\s*=\s*\S+)+)\s*$')
ANSWER_FIELD = re.compile(r'(estimate|se|n_obs)\s*=\s*(\S+)')


# Specification

def normalize_spec(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the spec in the final_bank_scoring/1 form. Reads the round-4 "targets" layout too."""
    if 'checks' in raw:
        spec = dict(raw)
        spec.setdefault('schema', SCHEMA)
        spec.setdefault('rel_tol', REL_TOL)
        spec.setdefault('abs_tol', ABS_TOL)
        spec.setdefault('conventions', [])
        spec.setdefault('n_obs', {'scored': False, 'reference': {}})
        return spec
    if 'scored' in raw:
        return _normalize_scored(raw)
    if 'targets' not in raw:
        raise ValueError('scoring spec has none of "scored", "checks", or "targets"')
    targets = raw['targets']
    rel_tols = {t.get('estimate_rel_tol', REL_TOL) for t in targets if not t.get('exact')}
    rel_tols |= {t.get('se_rel_tol', REL_TOL) for t in targets if t.get('se') is not None}
    floors = {t.get('abs_floor', ABS_TOL) for t in targets if not t.get('exact')}
    rel_tol = rel_tols.pop() if len(rel_tols) == 1 else REL_TOL
    abs_tol = floors.pop() if len(floors) == 1 else ABS_TOL
    checks, n_obs = [], {}
    for t in targets:
        kind = 'integer' if t.get('exact') else 'continuous'
        check = {'term': t['term'], 'field': 'estimate', 'reference': t['estimate'], 'kind': kind}
        if kind == 'continuous' and (t.get('estimate_rel_tol', rel_tol) != rel_tol or t.get('abs_floor', abs_tol) != abs_tol):
            check.update(rel_tol=t.get('estimate_rel_tol', rel_tol), abs_tol=t.get('abs_floor', abs_tol))
        checks.append(check)
        if t.get('se') is not None:
            check = {'term': t['term'], 'field': 'se', 'reference': t['se'], 'kind': 'continuous'}
            if t.get('se_rel_tol', rel_tol) != rel_tol:
                check.update(rel_tol=t['se_rel_tol'])
            checks.append(check)
        if t.get('n_obs') is not None:
            n_obs[t['term']] = t['n_obs']
    return {'schema': SCHEMA, 'task_id': raw.get('task_id'), 'rule': raw.get('rule', ''),
            'rel_tol': rel_tol, 'abs_tol': abs_tol, 'checks': checks,
            'conventions': raw.get('conventions', []), 'n_obs': {'scored': False, 'reference': n_obs}}


def _normalize_scored(raw: dict[str, Any]) -> dict[str, Any]:
    """The package layout of both final banks: "scored" entries with term, field, reference, and check."""
    checks, n_obs_terms, n_obs_ref = [], [], {}
    for item in raw['scored']:
        exact = str(item.get('check', '')).strip().lower().startswith('exact')
        if item['field'] == 'n_obs':
            n_obs_terms.append(item['term'])
            n_obs_ref[item['term']] = item['reference']
            continue
        checks.append({'term': item['term'], 'field': item['field'], 'reference': item['reference'],
                       'kind': 'integer' if exact else 'continuous'})
    n_obs = dict(raw.get('n_obs') or {})
    n_obs.setdefault('reference', {})
    n_obs['reference'] = {**n_obs['reference'], **n_obs_ref}
    n_obs['scored'] = bool(n_obs.get('scored', False))
    n_obs['scored_terms'] = sorted(set(n_obs.get('scored_terms', [])) | set(n_obs_terms))
    return {'schema': SCHEMA, 'task_id': raw.get('task_id'), 'rule': raw.get('rule', ''),
            'rel_tol': raw.get('relative_tolerance', REL_TOL), 'abs_tol': raw.get('absolute_floor', ABS_TOL),
            'checks': checks, 'conventions': raw.get('conventions', []), 'n_obs': n_obs}


def load_spec(source: Any) -> dict[str, Any]:
    """Load a spec from a dict, a scoring.json path, or a package folder that holds scoring.json."""
    if isinstance(source, dict):
        return normalize_spec(source)
    path = Path(source)
    if path.is_dir():
        path = path / 'scoring.json'
    return normalize_spec(json.loads(path.read_text()))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_spec(spec: dict[str, Any]) -> list[str]:
    """Return a list of problems. An empty list means the spec follows the final-bank rules."""
    problems = []
    if spec.get('schema') != SCHEMA:
        problems.append(f'schema is {spec.get("schema")!r}, expected {SCHEMA!r}')
    for key in ('rel_tol', 'abs_tol'):
        if not _finite(spec.get(key)) or spec[key] < 0:
            problems.append(f'{key} is not a non-negative number')
    checks = spec.get('checks') or []
    if not checks:
        problems.append('no checks')
    seen = set()
    for check in checks:
        term, field, kind, ref = check.get('term'), check.get('field'), check.get('kind'), check.get('reference')
        label = f'{term}.{field}'
        if (term, field) in seen:
            problems.append(f'{label}: duplicate check')
        seen.add((term, field))
        if not isinstance(term, str) or not term:
            problems.append(f'{label}: term is not a string')
        elif P_VALUE.search(term):
            problems.append(f'{label}: p-values are not scored; score the statistic')
        if field not in FIELDS:
            problems.append(f'{label}: field must be one of {FIELDS}')
        if kind not in KINDS:
            problems.append(f'{label}: kind {kind!r} must be one of {KINDS}')
        if not _finite(ref):
            problems.append(f'{label}: reference is not a finite number')
        elif ref == 0:
            problems.append(f'{label}: reference is zero; change the data design')
        elif kind == 'integer' and float(ref) != round(float(ref)):
            problems.append(f'{label}: integer reference {ref} is not integral')
        if 'rel_tol' in check or 'abs_tol' in check:
            problems.append(f'{label}: per-check tolerance; the rule is one tolerance for all checks')
    for convention in spec.get('conventions') or []:
        name = convention.get('name')
        if not name or name == 'main':
            problems.append('convention without a name, or named "main"')
        for ref in convention.get('references') or []:
            key = (ref.get('term'), ref.get('field'))
            if key not in seen:
                problems.append(f'convention {name}: {key[0]}.{key[1]} is not a check')
            if not _finite(ref.get('reference')) or ref.get('reference') == 0:
                problems.append(f'convention {name}: {key[0]}.{key[1]} reference is zero or not finite')
    return problems


# Numeric rule

def within(value: Any, reference: float, rel_tol: float = REL_TOL, abs_tol: float = ABS_TOL) -> bool:
    if not _finite(value) or not _finite(reference):
        return False
    return abs(value - reference) <= max(abs_tol, rel_tol * abs(reference))


def ratio(value: Any, reference: float, rel_tol: float = REL_TOL, abs_tol: float = ABS_TOL) -> float:
    """Error divided by the allowed error. 1 is the tolerance. Missing values give infinity."""
    if not _finite(value) or not _finite(reference):
        return math.inf
    allowed = max(abs_tol, rel_tol * abs(reference))
    return abs(value - reference) / allowed


def _check(check: dict[str, Any], reference: float, value: Any, spec: dict[str, Any]) -> dict[str, Any]:
    out = {'term': check['term'], 'field': check['field'], 'kind': check['kind'], 'value': value,
           'reference': reference}
    if check['kind'] == 'integer':
        out['pass'] = _finite(value) and float(value) == float(reference)
        out['ratio'] = 0.0 if out['pass'] else math.inf
    else:
        rel_tol = check.get('rel_tol', spec['rel_tol'])
        abs_tol = check.get('abs_tol', spec['abs_tol'])
        out['pass'] = within(value, reference, rel_tol, abs_tol)
        out['ratio'] = ratio(value, reference, rel_tol, abs_tol)
    return out


def _score_set(rows: dict[str, dict[str, Any]], spec: dict[str, Any], overrides: dict[tuple, float]):
    checks = []
    for check in spec['checks']:
        key = (check['term'], check['field'])
        reference = overrides.get(key, check['reference'])
        value = rows.get(check['term'], {}).get(check['field'])
        checks.append(_check(check, reference, value, spec))
    return checks


def score_rows(rows: dict[str, dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    """Score parsed rows {term: {estimate, se, n_obs}} against a normalized spec."""
    spec = normalize_spec(spec)
    missing = sorted({c['term'] for c in spec['checks']} - set(rows))
    main = _score_set(rows, spec, {})
    chosen, checks = 'main', main
    if not all(c['pass'] for c in main):
        for convention in spec.get('conventions') or []:
            overrides = {(r['term'], r['field']): r['reference'] for r in convention.get('references') or []}
            trial = _score_set(rows, spec, overrides)
            if all(c['pass'] for c in trial):
                chosen, checks = convention['name'], trial
                break
    n_spec = spec.get('n_obs') or {}
    n_ref = n_spec.get('reference') or {}
    mismatch = sorted(term for term, n in n_ref.items()
                      if not (_finite(rows.get(term, {}).get('n_obs')) and float(rows[term]['n_obs']) == float(n)))
    passed = all(c['pass'] for c in checks)
    scored_n = set(n_ref) if n_spec.get('scored') else set(n_spec.get('scored_terms') or [])
    if scored_n & set(mismatch):
        passed = False
    return {'passed': passed, 'convention': chosen, 'checks': checks, 'missing_terms': missing,
            'n_obs_mismatch': mismatch}


# Reading printed values

def _quantum(text: str) -> float:
    """The rounding step of a printed decimal number: 10^-(decimals), scaled by its exponent.

    A mantissa without a decimal point ("2", "196", "1e-09") counts as exact, because the %g
    formats that the prompts ask for drop trailing zeros and the point.
    """
    text = text.strip().lower().lstrip('+-')
    mantissa, _, exponent = text.partition('e')
    if '.' not in mantissa:
        return 0.0
    decimals = len(mantissa.split('.', 1)[1])
    return 10.0 ** ((int(exponent) if exponent else 0) - decimals)


def _to_number(text: Any) -> float | None:
    if text is None:
        return None
    if isinstance(text, bool):
        raise ValueError('boolean is not a number')
    if isinstance(text, (int, float)):
        return float(text) if math.isfinite(text) else None
    if isinstance(text, str):
        if text.strip().lower() in MISSING_TOKENS:
            return None
        value = float(text)
        return value if math.isfinite(value) else None
    raise ValueError(f'{type(text).__name__} is not a number')


def printed_values_agree(a: Any, b: Any, exact: bool = False) -> bool:
    """True when two printed readings can be the same value at different printed precision."""
    try:
        x, y = _to_number(a), _to_number(b)
    except ValueError:
        return False
    if x is None or y is None:
        return x is None and y is None
    if exact or not isinstance(a, str) or not isinstance(b, str):
        return x == y
    slack = (_quantum(a) + _quantum(b)) / 2
    return abs(x - y) <= slack * (1 + 1e-9) + 1e-300


def _merge(readings: list[dict[str, Any]], scored: set[tuple[str, str]], exact: bool):
    """readings: [{term, source, raw: {field: text}}]. Return rows, conflicts, notes."""
    rows, conflicts, notes, best = {}, [], [], {}
    for reading in readings:
        term = reading['term']
        for field in ('estimate', 'se', 'n_obs'):
            raw = reading['raw'].get(field)
            try:
                value = _to_number(raw)
            except ValueError:
                notes.append(f'{term}.{field}: unparseable value {raw!r} in {reading["source"]}')
                value, raw = None, None
            key = (term, field)
            if key not in best:
                best[key] = (raw, value, reading['source'])
                continue
            old_raw, old_value, old_source = best[key]
            if not printed_values_agree(old_raw, raw, exact=exact):
                item = {'term': term, 'field': field, 'first': old_raw, 'first_source': old_source,
                        'second': raw, 'second_source': reading['source']}
                if key in scored:
                    conflicts.append(item)
                else:
                    notes.append(f'unscored conflict {term}.{field}: {old_raw!r} vs {raw!r}')
                continue
            if value is not None and isinstance(raw, str) and isinstance(old_raw, str) and _quantum(raw) < _quantum(old_raw):
                best[key] = (raw, value, reading['source'])
    for (term, field), (_, value, _) in best.items():
        rows.setdefault(term, {'estimate': None, 'se': None, 'n_obs': None})[field] = value
    return rows, conflicts, notes


def answer_readings(text: str, source: str) -> list[dict[str, Any]]:
    """Parse ANSWER lines. Stata continuation lines that start with "> " are joined first."""
    joined: list[str] = []
    for line in text.splitlines():
        if line.startswith('> ') and joined:
            joined[-1] += line[2:]
        else:
            joined.append(line)
    out = []
    for line in joined:
        found = ANSWER_LINE.match(line.strip())
        if not found:
            continue
        fields = ANSWER_FIELD.findall(found.group(2))
        names = [name for name, _ in fields]
        if 'estimate' not in names or len(names) != len(set(names)):
            continue
        out.append({'term': found.group(1), 'source': source, 'raw': dict(fields)})
    return out


def read_stata(sources: list[tuple[str, str]], scored: set[tuple[str, str]]):
    readings = []
    used = []
    for label, text in sources:
        found = answer_readings(text, label)
        if found:
            used.append(label)
        readings.extend(found)
    rows, conflicts, notes = _merge(readings, scored, exact=False)
    return rows, conflicts, notes, used


def read_result_json(path: Path, scored: set[tuple[str, str]]):
    """Return rows, conflicts, notes, and an error message (empty when the file parsed)."""
    if not path.exists():
        return {}, [], [], 'result.json missing'
    text = path.read_text(errors='replace')
    try:
        data = json.loads(text)
    except ValueError as error:
        return {}, [], [], f'result.json invalid: {error}'
    estimates = data.get('estimates') if isinstance(data, dict) else None
    if not isinstance(estimates, list):
        return {}, [], [], 'result.json has no estimates list'
    readings, notes = [], []
    for row in estimates:
        if not isinstance(row, dict) or not isinstance(row.get('term'), str):
            notes.append('estimate row without a string term')
            continue
        raw = {}
        for field in ('estimate', 'se', 'n_obs'):
            value = row.get(field)
            if isinstance(value, bool):
                notes.append(f'{row["term"]}.{field}: boolean is not a number')
                value = 'NA'
            elif isinstance(value, str) and value.strip().lower() not in MISSING_TOKENS:
                notes.append(f'{row["term"]}.{field}: number written as a string')
            raw[field] = value
        readings.append({'term': row['term'], 'source': 'result.json', 'raw': raw})
    rows, conflicts, more = _merge(readings, scored, exact=True)
    return rows, conflicts, notes + more, ''


def _json_repair(path: Path, spec: dict[str, Any]):
    """Diagnostic only: would the collector's minimal JSON repair have produced a passing row set?"""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from artifact_repair import repair_result_text  # noqa: PLC0415
    except Exception:  # pragma: no cover - the module is part of this repository
        return None
    data, label = repair_result_text(path.read_text(errors='replace'))
    if data is None:
        return {'label': label, 'passed': False}
    temporary = {'estimates': data.get('estimates')} if isinstance(data, dict) else {}
    rows = {}
    for row in temporary.get('estimates') or []:
        if isinstance(row, dict) and isinstance(row.get('term'), str):
            try:
                rows[row['term']] = {f: _to_number(row.get(f)) for f in ('estimate', 'se', 'n_obs')}
            except ValueError:
                continue
    return {'label': label, 'passed': score_rows(rows, spec)['passed']}


# One run

def score_run(spec_source: Any, language: str, run_dir: Any, stdout: str | None = None,
              final_files: list[Any] | None = None) -> dict[str, Any]:
    """Score one run folder. language is python, r, or stata."""
    spec = load_spec(spec_source)
    run_dir = Path(run_dir)
    data_dir = None
    if not isinstance(spec_source, dict):
        package = Path(spec_source)
        package = package if package.is_dir() else package.parent
        data_dir = package / 'data' if (package / 'data').is_dir() else None
    scored = {(c['term'], c['field']) for c in spec['checks']}
    result: dict[str, Any] = {'task_id': spec.get('task_id'), 'language': language, 'passed': False,
                              'reason': '', 'source': '', 'rows': {}, 'checks': [], 'convention': None,
                              'missing_terms': [], 'n_obs_mismatch': [], 'conflicts': [], 'notes': []}
    if language in ('python', 'r'):
        path = run_dir / 'result.json'
        rows, conflicts, notes, error = read_result_json(path, scored)
        result.update(source='result.json', notes=notes)
        if error:
            result['reason'] = error
            if path.exists():
                result['json_repair'] = _json_repair(path, spec)
            return result
    elif language == 'stata':
        sources = []
        log = run_dir / 'run.log'
        if log.exists():
            sources.append(('run.log', log.read_text(errors='replace')))
        if stdout:
            sources.append(('stdout', stdout))
        for item in final_files or []:
            item = Path(item)
            item = item if item.is_absolute() else run_dir / item
            if item.exists() and item.resolve() != log.resolve():
                sources.append((item.name, item.read_text(errors='replace')))
        rows, conflicts, notes, used = read_stata(sources, scored)
        result.update(source='+'.join(used), notes=notes)
        if not used:
            result['reason'] = 'no ANSWER lines in run.log, stdout, or final-run files'
            return result
    else:
        raise ValueError(f'unknown language {language!r}')
    result['rows'] = rows
    result['conflicts'] = conflicts
    changed = []
    if data_dir is not None:
        result['input_files'] = {}
        for original in sorted(p for p in data_dir.iterdir() if p.is_file()):
            copy = run_dir / original.name
            if not copy.exists():
                result['input_files'][original.name] = 'not in run folder'
            elif hashlib.sha256(copy.read_bytes()).digest() == hashlib.sha256(original.read_bytes()).digest():
                result['input_files'][original.name] = 'unchanged'
            else:
                result['input_files'][original.name] = 'changed'
                changed.append(original.name)
    scored_result = score_rows(rows, spec)
    result.update(checks=scored_result['checks'], convention=scored_result['convention'],
                  missing_terms=scored_result['missing_terms'], n_obs_mismatch=scored_result['n_obs_mismatch'])
    if changed:
        result['reason'] = 'input file changed: ' + ', '.join(changed)
        return result
    if conflicts:
        result['reason'] = 'conflict: ' + ', '.join(f'{c["term"]}.{c["field"]}' for c in conflicts)
        return result
    result['passed'] = scored_result['passed']
    if not result['passed']:
        if scored_result['missing_terms']:
            result['reason'] = 'missing terms: ' + ', '.join(scored_result['missing_terms'])
        else:
            failed = [f'{c["term"]}.{c["field"]}' for c in scored_result['checks'] if not c['pass']]
            result['reason'] = ('outside tolerance: ' + ', '.join(failed)) if failed else 'n_obs mismatch'
    return result


def jsonable(value: Any) -> Any:
    """Replace non-finite floats by strings so that the output is strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--package', help='package folder that holds scoring.json')
    source.add_argument('--scoring', help='path to scoring.json')
    parser.add_argument('--language', required=True, choices=('python', 'r', 'stata'))
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--stdout', help='file with the captured stdout of the final run (Stata)')
    parser.add_argument('--final-file', action='append', default=[], help='file written in the final run (Stata)')
    args = parser.parse_args(argv)
    stdout = Path(args.stdout).read_text(errors='replace') if args.stdout else None
    result = score_run(args.package or args.scoring, args.language, args.run_dir, stdout=stdout,
                       final_files=args.final_file)
    print(json.dumps(jsonable(result), indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Score the recorded outputs again and compare the result with the stored primary outcome.

Run from the repository root, after extracting one or more run archives (tar keeps the file times, which the
Stata output-channel rule reads; do not extract with an option that resets them):

    tar -xJf runs/sonnet_stata_agent.tar.xz
    python code/score.py                          # every run in analysis/runs.csv whose record is present
    python code/score.py --out rescored.csv       # also write one row per scored run

The script uses the Python standard library only. It reads analysis/runs.csv, the run folders, the reference
outputs tasks/<task>/reference.json, the standard-error rules and accepted conventions code/scoring_spec.json,
and the wrong-method alternatives code/scoring_alternatives.json. It checks the SHA-256 of every record and
output file against runs.csv, scores each run, and compares the outcome and the failure category with
runs.csv. It executes no submitted script. The exit code is 1 when a hash or an outcome differs.

The primary outcome (a run is a success when every check passes):

1. Values. Python and R runs: the estimates list of result.json, read after the mechanical, digit-preserving
   JSON repairs below. Stata runs: the ANSWER lines of the scored output (lines broken by Stata with "> " are
   joined first), plus a value printed as "ANSWER term value" when the line gives one distinct value. For Stata,
   other text files that the final execution wrote (modification time inside the final execution window,
   widened by 2 seconds) and the stdout of the final execution are read too. Values that agree within the
   rounding of the coarser token count once. Two different values for one term fail the run.
2. Estimates. Each estimand passes when |value - reference| <= max(0.01 * |reference|, 0.005). Counts
   (cooks_distance observation ids; accepts_zero and unbounded in broad_ar_confidence_set) must be equal
   integers. A boolean value fails.
3. Standard errors. In the nine tasks with se_scored in scoring_spec.json, each listed term must report a
   standard error within rel_tol (plus abs_floor) of one accepted value. A token with k >= 4 decimal places
   gets the rounding allowance 0.5 * 10^-k.
4. Wrong-method alternatives. A passing estimate or standard error fails when it is closer to a listed
   alternative of its term (scoring_alternatives.json) than to every accepted value. Accepted values are the
   reference and the conventions of scoring_spec.json. An alternative is tested only when it and every
   accepted value differ after rounding to the decimals of the reported value.

The failure category is Success, Silent failure (the last execution succeeded and every estimand has a
finite value), or Loud failure. A run that fails only check 4 is a silent failure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EST_REL, EST_ABS = 0.01, 0.005  # estimate tolerance of the primary outcome
MIN_DECIMALS = 4  # rounding allowance and same-value slack apply to tokens with at least 4 decimals
WINDOW_SLACK = 2.0  # seconds around the final execution window
MAX_BYTES = 5_000_000
SAME_VALUE = 1e-6  # an estimate convention this close to the reference is the reference
EXACT_TERMS = {
    'cooks_distance': {f'rank{rank}_obs_id' for rank in range(1, 6)},
    'broad_ar_confidence_set': {'accepts_zero', 'unbounded'},
}
TEXT_SUFFIX = {'.txt', '.log', '.json', '.out', '.md', '.csv', '.tsv', '.smcl', ''}
SCRIPT_SUFFIX = {'.do', '.py', '.r', '.ado'}
SKIP_NAMES = {'printed_output.txt', 'verification.txt', 'input.csv'}
STRICT = re.compile(r'^\s*ANSWER:\s+(\S+)\s+estimate\s*=\s*([^\s]+)(.*)$', re.MULTILINE)
PRINTED = re.compile(r'^\s*ANSWER:?\s+(\S+)\s+(?:estimate\s*=\s*)?'
                     r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?=\s|$)([^\n]*)', re.MULTILINE)
SE_TOKEN = re.compile(r'(?:^|\s)se\s*=\s*([^\s]+)')


# ---------------------------------------------------------------- numbers and tokens
def number(value):
    """Finite float or None. A boolean counts as a number here; check 2 fails it separately."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def strict_number(value):
    """Finite float or None. Booleans are not numbers."""
    return None if isinstance(value, bool) or value is None else number(value)


def decimals(token):
    if token is None:
        return None
    try:
        exponent = Decimal(str(token).strip()).as_tuple().exponent
    except (InvalidOperation, ValueError):
        return None
    return max(0, -exponent) if isinstance(exponent, int) else None


def allowance(token):
    k = decimals(token)
    return 0.0 if k is None or k < MIN_DECIMALS else 0.5 * 10.0 ** (-k)


def same_value(x, y, ta, tb):
    if x is None or y is None:
        return x is None and y is None
    ka, kb = decimals(ta), decimals(tb)
    slack = 0.5 * 10.0 ** (-min(ka, kb)) if ka is not None and kb is not None and min(ka, kb) >= MIN_DECIMALS else 0.0
    return abs(x - y) <= max(slack, 1e-9 * max(abs(x), abs(y)))


def join_stata(text):
    return re.sub(r'\n>\s?', '', text)


# ---------------------------------------------------------------- JSON repair (digit-preserving)
def _lines(text, edit):
    out = []
    for raw in text.splitlines(keepends=True):
        ending = next((e for e in ('\r\n', '\n', '\r') if raw.endswith(e)), '')
        out.append(edit(raw[:len(raw) - len(ending)]) + ending)
    return ''.join(out)


def _code_fence(text):
    stripped = text.strip()
    if not stripped.startswith('```'):
        return text
    return re.sub(r'```\s*$', '', re.sub(r'^```[A-Za-z0-9_-]*\s*', '', stripped))


def _line_wrapper(line):
    line = re.sub(r'^(\s*)`?"(?=\s*[\{\[])', r'\1', line)
    line = re.sub(r'("\'|`")\s*$', '', line)
    return re.sub(r'(?<=[\}\]])\s*"\s*(,?)\s*$', r'\1', line)


def _unterminated(line):
    if line.count('"') % 2 == 1:
        match = re.search(r'[\s,\}\]]*$', line)
        cut = match.start() if match else len(line)
        line = line[:cut] + '"' + line[cut:]
    return line


def _stata_number(text):
    text = re.sub(r'(?<=[:,\[\s])-?\.(?=\s*[,}\]])', 'null', text)
    text = re.sub(r'(?<=[:,\[\s])(-?)\.(?=\d)', r'\g<1>0.', text)
    return re.sub(r'(?<=\d)\.(?=\s*[,}\]])', '.0', text)


RULES = (
    ('code_fence', _code_fence),
    ('escaped_quotes', lambda t: t.replace('\\"', '"')),
    ('doubled_quotes', lambda t: t.replace('""', '"')),
    ('line_wrapper', lambda t: _lines(t, _line_wrapper)),
    ('unterminated_string', lambda t: _lines(t, _unterminated)),
    ('trailing_comma', lambda t: re.sub(r',(\s*[}\]])', r'\1', t)),
    ('stray_quote_after_number', lambda t: re.sub(r'(:[ \t]*-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)"(?=[ \t]*[,}\]])',
                                                  r'\1', t)),
    ('empty_value', lambda t: re.sub(r'(:[ \t]*)(?=[,}\]])', r'\1null', t)),
    ('stata_number', _stata_number),
)


def _parses(text, **kw):
    try:
        loaded = json.loads(text, **kw)
    except Exception:  # noqa: BLE001  a failed parse is the normal negative case
        return None
    return loaded if isinstance(loaded, dict) else None


def _digits_preserved(raw, repaired):
    original = [c for c in raw if c.isdigit()]
    index = 0
    for char in repaired:
        if not char.isdigit():
            continue
        if index < len(original) and char == original[index]:
            index += 1
        elif char != '0':
            return False
    return index == len(original)


def repaired_json(raw):
    """(parsed object, repaired text) under the smallest rule set that parses, or (None, None)."""
    if not raw.strip():
        return None, None
    loaded = _parses(raw)
    if loaded is not None:
        return loaded, raw
    for size in range(1, len(RULES) + 1):
        for subset in itertools.combinations(RULES, size):
            text = raw
            for _, rule in subset:
                text = rule(text)
            loaded = _parses(text)
            if loaded is None:
                continue
            if not _digits_preserved(raw, text):
                return None, None
            return loaded, text
    return None, None


class Token(str):
    """A JSON number kept as its text."""


# ---------------------------------------------------------------- values of one output
def json_rows(raw):
    """{term: (estimate, estimate token, se, se token)} of a JSON result, or None when it does not parse."""
    data, text = repaired_json(raw)
    if data is None:
        return None
    tokens = _parses(text, parse_float=Token, parse_int=Token)
    rows = data.get('estimates', [])
    if not isinstance(rows, list):
        return {}
    token_rows = tokens.get('estimates', []) if isinstance(tokens, dict) else []
    out = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get('term'), str):
            continue
        trow = token_rows[index] if index < len(token_rows) and isinstance(token_rows[index], dict) else {}

        def token(key):
            if isinstance(trow.get(key), str):
                return str(trow[key])
            value = strict_number(row.get(key))
            return None if value is None else repr(value)
        out[row['term']] = (row.get('estimate'), token('estimate'), row.get('se'), token('se'))
    return out


def se_from_tail(tail):
    match = SE_TOKEN.search(' ' + tail)
    return match[1] if match else None


def stata_lines(text, estimands):
    """Every ANSWER value per requested term: {term: [(estimate, token, se token), ...]}."""
    text = join_stata(text)
    found = {}
    for match in STRICT.finditer(text):
        if match[1] in estimands:
            found.setdefault(match[1], []).append((match[2], match[2], se_from_tail(match[3])))
    for match in PRINTED.finditer(text):
        if match[1] in estimands:
            found.setdefault(match[1], []).append((float(match[2]), match[2], se_from_tail(match[3])))
    return found


def primary_values(path, platform, estimands):
    """{term: (estimate, estimate token, se, se token)} of the scored output file."""
    if not path.is_file():
        return {}
    raw = path.read_text(errors='replace')
    if path.suffix == '.json':
        rows = json_rows(raw)
        values = {} if rows is None else rows
    else:
        values = {}
        for match in STRICT.finditer(join_stata(raw)):
            se = se_from_tail(match[3])
            values[match[1]] = (match[2], match[2], se, se)
    if platform == 'stata':
        options = {}
        for match in PRINTED.finditer(join_stata(raw)):
            if match[1] in estimands:
                options.setdefault(match[1], {}).setdefault(float(match[2]), (match[2], match[3]))
        for term, found in options.items():
            if len(found) == 1 and term not in values:
                value, (token, tail) = next(iter(found.items()))
                values[term] = (value, token, se_from_tail(tail), se_from_tail(tail))
    return values


def final_window(attempt, record):
    numbered = []
    for path in attempt.glob('execution_attempt*.json'):
        match = re.fullmatch(r'execution_attempt(\d+)\.json', path.name)
        if match:
            numbered.append((int(match[1]), path))
    if not numbered:
        return None
    n, end_file = max(numbered)
    end = end_file.stat().st_mtime
    starts = [p.stat().st_mtime for p in attempt.glob(f'execution_attempt{n}.*') if p.name != end_file.name]
    executions = record.get('executions') or []
    wall = executions[-1].get('wall_seconds') if executions and isinstance(executions[-1], dict) else None
    if isinstance(wall, (int, float)) and not isinstance(wall, bool):
        starts.append(end - wall)
    start = min(starts) if starts else end
    return start - WINDOW_SLACK, end + WINDOW_SLACK


def final_stdout(attempt, record):
    stdout = ''
    executions = record.get('executions') or []
    if executions and isinstance(executions[-1], dict):
        stdout = executions[-1].get('stdout') or ''
    files = [p for p in attempt.glob('execution_attempt*.stdout') if re.search(r'(\d+)', p.name)]
    if files:
        latest = max(files, key=lambda p: int(re.search(r'(\d+)', p.name)[1]))
        stdout += '\n' + latest.read_text(errors='replace')
    return stdout


def channel_sources(record_path, record, output, estimands):
    """[(label, {term: [(estimate, token, se token), ...]})]: final-execution files and stdout of a Stata run."""
    attempt = record_path.parent
    primary_hash = hashlib.sha256(output.read_bytes()).hexdigest() if output.is_file() else None
    workspace = next((attempt / n for n in ('final_workspace', 'submitted_workspace') if (attempt / n).is_dir()), None)
    window = final_window(attempt, record)
    texts = []
    if workspace is not None and window is not None:
        for path in sorted(workspace.rglob('*')):
            if not path.is_file() or path.name in SKIP_NAMES:
                continue
            suffix = path.suffix.lower()
            if suffix in SCRIPT_SUFFIX or suffix not in TEXT_SUFFIX:
                continue
            stat = path.stat()
            if not (window[0] <= stat.st_mtime <= window[1]) or stat.st_size > MAX_BYTES:
                continue
            if output.is_file() and path.resolve() == output.resolve():
                continue
            data = path.read_bytes()
            if primary_hash and hashlib.sha256(data).hexdigest() == primary_hash:
                continue
            texts.append((str(path.relative_to(attempt)), data.decode(errors='replace')))
    texts.append(('stdout', final_stdout(attempt, record)))
    sources = []
    for label, text in texts:
        found = stata_lines(text, estimands)
        if found:
            sources.append((label, found))
    return sources


def stored_token(values, output, term):
    """Raw token of a primary estimate. A value from the "ANSWER term value" form is found again in the file."""
    value = values[term][0]
    if isinstance(value, str):
        return value
    if output.is_file() and strict_number(value) is not None:
        for match in PRINTED.finditer(join_stata(output.read_text(errors='replace'))):
            if match[1] == term and float(match[2]) == strict_number(value):
                return match[2]
    return value


def merge_estimates(values, output, sources, estimands):
    """(estimates, conflicts): one value per term from the output and the other sources."""
    merged = {term: row[0] for term, row in values.items()}
    conflicts = []
    for term in estimands:
        candidates = []
        if term in values and strict_number(values[term][0]) is not None:
            candidates.append(('primary', stored_token(values, output, term)))
        for label, found in sources:
            candidates += [(label, item[0] if isinstance(item[0], str) else item[1])
                           for item in found.get(term, []) if strict_number(item[0]) is not None]
        if not candidates:
            continue
        distinct = []
        for label, token in candidates:
            if not any(same_value(strict_number(token), strict_number(other), token, other) for _, other in distinct):
                distinct.append((label, token))
        if len(distinct) > 1:
            conflicts.append(term)
            merged.pop(term, None)
            continue
        label, token = max(candidates, key=lambda item: decimals(item[1]) or 0)
        if label != 'primary':
            merged[term] = token
    return merged, conflicts


def merge_se(values, sources, estimands):
    """(se values {term: (se, token)}, conflicts) under the same channel rule, applied to standard errors."""
    if not sources:
        return {t: (row[2], row[3]) for t, row in values.items()}, []
    merged, conflicts = {}, []
    for term in estimands:
        candidates = [(values[term][2], values[term][3])] if term in values else []
        candidates += [(item[2], item[2]) for _, found in sources for item in found.get(term, [])]
        usable = [c for c in candidates if strict_number(c[0]) is not None]
        if not usable:
            if term in values:
                merged[term] = (values[term][2], values[term][3])
            continue
        distinct = []
        for value in usable:
            if not any(same_value(strict_number(value[0]), strict_number(o[0]), value[1], o[1]) for o in distinct):
                distinct.append(value)
        if len(distinct) > 1:
            conflicts.append(term)
            continue
        merged[term] = max(usable, key=lambda v: decimals(v[1]) or 0)
    return merged, conflicts


# ---------------------------------------------------------------- checks
def round_decimals(value, token=None):
    """Decimal places of a reported value: its token when given, otherwise the shortest repr of the float."""
    text = str(token).strip() if token is not None and str(token).strip() not in ('', 'nan', 'None') else repr(float(value))
    try:
        exponent = Decimal(text).as_tuple().exponent
    except Exception:  # noqa: BLE001  an unparsable token falls back to the float
        exponent = Decimal(repr(float(value))).as_tuple().exponent
    return -int(exponent)


def closer_alternative(got, accepted, alternatives, places):
    d_ref = min(abs(got - a) for a in accepted)
    best = None
    for alt in alternatives:
        if any(round(alt['value'], places) == round(a, places) for a in accepted):
            continue
        d_alt = abs(got - alt['value'])
        if d_alt < d_ref and (best is None or d_alt < abs(got - best['value'])):
            best = alt
    return best


def score_run(task, platform, record_path, record, output, reference, se_rules, accepted, alternatives):
    """(passed checks 1 to 3, failed check 4, delivered, notes) for one run."""
    estimands = [row['term'] for row in reference['estimates']]
    refs = {row['term']: float(row['estimate']) for row in reference['estimates']}
    values = primary_values(output, platform, estimands)
    sources = channel_sources(record_path, record, output, estimands) if platform == 'stata' else []
    estimates, conflicts = merge_estimates(values, output, sources, estimands)
    notes = [f'{t}: conflicting values in output channels' for t in conflicts]
    passed, flagged = not conflicts, False
    for term in estimands:
        raw, ref = estimates.get(term), refs[term]
        got = number(raw)
        if term not in estimates or got is None:
            notes.append(f'{term}: missing or non-finite estimate')
            passed = False
            continue
        if task == 'cooks_distance' and term in EXACT_TERMS[task]:
            coef_pass = got.is_integer() and ref.is_integer() and int(got) == int(ref)
        else:
            coef_pass = abs(got - ref) <= max(EST_REL * abs(ref), EST_ABS)
        coef_pass = coef_pass and not isinstance(raw, bool)
        exact_pass = term not in EXACT_TERMS.get(task, set()) or (got.is_integer() and got == ref)
        if not (coef_pass and exact_pass):
            notes.append(f'{term}: estimate outside tolerance')
            passed = False
        if coef_pass and (term, 'estimate') in alternatives:
            hit = closer_alternative(got, accepted[(term, 'estimate')], alternatives[(term, 'estimate')],
                                     round_decimals(got))
            if hit:
                notes.append(f"{term}: estimate closer to the {hit['method']} value")
                flagged = True
    if se_rules:
        se_values, se_conflicts = merge_se(values, sources, estimands)
        for term in sorted(se_rules):
            rule = se_rules[term]
            value, token = se_values.get(term, (None, None))
            got = strict_number(value)
            ok = got is not None and term not in se_conflicts and any(
                abs(got - a) <= rule['rel_tol'] * abs(a) + rule.get('abs_floor', 0.0) + allowance(token)
                for a in rule['accepted_values'])
            if not ok:
                notes.append(f'{term}: standard error missing, conflicting, or outside tolerance')
                passed = False
                continue
            if (term, 'se') in alternatives:
                hit = closer_alternative(got, accepted[(term, 'se')], alternatives[(term, 'se')],
                                         round_decimals(got, token))
                if hit:
                    notes.append(f"{term}: standard error closer to the {hit['method']} value")
                    flagged = True
    delivered = all(term in estimates and number(estimates[term]) is not None for term in estimands)
    return passed, flagged, delivered, notes


def execution_ok(record):
    executions = record.get('executions')
    if not (isinstance(executions, list) and executions):
        items = []
        for key, value in record.items():
            match = re.match(r'^execution(?:_attempt_(\d+))?$', key)
            if match and isinstance(value, dict):
                items.append((int(match[1] or 0), value))
        executions = [value for _, value in sorted(items)]
    return bool(executions and executions[-1].get('ok'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rules(root):
    """SE rules, accepted values, and alternatives per task."""
    spec = json.loads((root / 'code/scoring_spec.json').read_text())['tasks']
    alternatives_file = root / 'code/scoring_alternatives.json'
    # Without the alternatives file, check 4 is skipped (the two-model analysis does not use it).
    table = json.loads(alternatives_file.read_text())['tasks'] if alternatives_file.is_file() else {}
    rules = {}
    for folder in sorted((root / 'tasks').iterdir()):
        if not (folder / 'reference.json').is_file():
            continue
        task = folder.name
        reference = json.loads((folder / 'reference.json').read_text())
        rule = spec.get(task, {})
        se_rules = rule['se_terms'] if rule.get('se_scored') else {}
        accepted = {}
        for row in reference['estimates']:
            term, ref = row['term'], float(row['estimate'])
            extra = rule.get('terms', {}).get(term, {}).get('accepted_values', [])
            accepted[(term, 'estimate')] = [ref] + [float(v) for v in extra if abs(v - ref) > SAME_VALUE * abs(ref)]
            if term in se_rules:
                accepted[(term, 'se')] = [float(v) for v in se_rules[term]['accepted_values']]
        alternatives = {}
        for alt in table.get(task, {}).get('alternatives', []):
            alternatives.setdefault((alt['term'], alt['kind']), []).append(alt)
        rules[task] = (reference, se_rules, accepted, alternatives)
    return rules


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT, help='repository root (default: the parent of code/)')
    parser.add_argument('--runs', type=Path, help='the run folders\' base (default: ROOT; record_path is relative to it)')
    parser.add_argument('--out', type=Path, help='write one row per scored run to this CSV file')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    base = (args.runs or root).resolve()
    rules = load_rules(root)
    with (root / 'analysis/runs.csv').open(newline='') as handle:
        runs = list(csv.DictReader(handle))
    results, missing, hash_errors = [], 0, []
    for row in runs:
        record_path = base / row['record_path']
        if not record_path.is_file():
            missing += 1
            continue
        if sha256(record_path) != row['record_sha256']:
            hash_errors.append((row['position_id'], 'record'))
            continue
        output = base / row['output_path']
        got_hash = sha256(output) if output.is_file() else ''
        if got_hash != (row['output_sha256'] or ''):
            hash_errors.append((row['position_id'], 'output'))
            continue
        record = json.loads(record_path.read_text())
        passed, flagged, delivered, notes = score_run(row['task'], row['platform'], record_path, record, output,
                                                      *rules[row['task']])
        if passed and not flagged:
            category = 'Success'
        elif passed or (delivered and execution_ok(record)):
            category = 'Silent failure'  # check 4 alone, or a run that executed and delivered every value
        else:
            category = 'Loud failure'
        passed = passed and not flagged
        stored = row['primary'] == 'True'
        results.append(dict(position_id=row['position_id'], task=row['task'], platform=row['platform'],
                            stored_primary=stored, rescored_primary=passed,
                            stored_category=row['failure_category'], rescored_category=category,
                            match=stored == passed and category == row['failure_category'], notes=' | '.join(notes)))
    if args.out:
        with args.out.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]) if results else ['position_id'])
            writer.writeheader()
            writer.writerows(results)
    mismatches = [r for r in results if not r['match']]
    print(f'Scored {len(results):,} runs ({missing:,} of {len(runs):,} records not extracted). '
          f'Successes: stored {sum(r["stored_primary"] for r in results):,}, '
          f'rescored {sum(r["rescored_primary"] for r in results):,}. '
          f'Outcome or category differs in {len(mismatches):,} runs. Hash differences: {len(hash_errors):,}.')
    for r in mismatches[:20]:
        print('  differs:', r['position_id'], r['stored_category'], '->', r['rescored_category'], r['notes'])
    for position, kind in hash_errors[:20]:
        print('  hash differs:', position, kind)
    return 1 if mismatches or hash_errors else 0


if __name__ == '__main__':
    sys.exit(main())

"""Output-channel rule for Stata runs (decision record 2026-09-24_output-channel-rule.md).

Nine Stata prompts ask the model to print one line per estimate but do not
name the log. Some runs wrote correct ANSWER lines to another file or to
stdout. Under this rule, the scorer also reads:

- text files in the saved final workspace whose modification time falls
  inside the final execution window (plus or minus WINDOW_SLACK seconds), and
- the stdout of the final execution.

The values from all sources are merged per term. Two sources that give
different values for one term are a conflict, and a conflict fails the run.
Tolerances, standard errors, and observation counts are unchanged.

This module ports the channel part of variant V2 in
``code/benchmark/rescore_v2.py``. One difference: a workspace file counts only
when the run has a recorded final execution window. The rule reads saved files
only. It executes no script and calls no provider.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import math
from pathlib import Path
import re

STRICT = re.compile(r"^\s*ANSWER:\s+(\S+)\s+estimate\s*=\s*([^\s]+)", re.MULTILINE)
# Same pattern as dataset.PRINTED.
PRINTED = re.compile(
    r"^\s*ANSWER:?\s+(\S+)\s+(?:estimate\s*=\s*)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\s|$)",
    re.MULTILINE)
WINDOW_SLACK = 2.0  # seconds; chosen limit, the same as rescore_v2.py
MIN_DECIMALS = 4  # chosen limit for the rounding slack in the conflict check, the same as rescore_v2.py
MAX_BYTES = 5_000_000
TEXT_SUFFIX = {".txt", ".log", ".json", ".out", ".md", ".csv", ".tsv", ".smcl", ""}
SCRIPT_SUFFIX = {".do", ".py", ".r", ".ado"}
SKIP_NAMES = {"printed_output.txt", "verification.txt", "input.csv"}


def number(value):
    """Finite float or None. Booleans are not numbers."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def decimals(token):
    """Decimal places of a numeric token, or None."""
    if token is None:
        return None
    try:
        exponent = Decimal(str(token).strip()).as_tuple().exponent
    except (InvalidOperation, ValueError):
        return None
    return max(0, -exponent) if isinstance(exponent, int) else None


def join_stata(text):
    """Join Stata continuation lines ("\\n> ")."""
    return re.sub(r"\n>\s?", "", text)


def answer_values(text, estimands):
    """Every ANSWER token for each requested term: {term: [token, ...]}."""
    text = join_stata(text)
    found = {}
    for pattern in (STRICT, PRINTED):
        for match in pattern.finditer(text):
            if match[1] in estimands:
                found.setdefault(match[1], []).append(match[2])
    return found


def primary_tokens(path, values):
    """Raw tokens of the primary values, for the rounding slack in the conflict check.

    The log rule stores a value added by the printed-line pattern as a float.
    Its raw token is found again in the log so that "0.1900" keeps four places.
    """
    tokens = {term: row.get("estimate") for term, row in values.items() if isinstance(row.get("estimate"), str)}
    missing = [term for term, row in values.items() if term not in tokens and number(row.get("estimate")) is not None]
    if missing and Path(path).is_file():
        text = join_stata(Path(path).read_text(errors="replace"))
        for match in PRINTED.finditer(text):
            if match[1] in missing and match[1] not in tokens and float(match[2]) == number(values[match[1]]["estimate"]):
                tokens[match[1]] = match[2]
    return tokens


def final_execution_window(attempt, record):
    """(start, end) file times of the final harness execution, or None.

    The harness writes execution_attemptN.stdout and .stderr when the execution
    starts and execution_attemptN.json when it ends. Saved workspace copies
    keep file modification times.
    """
    numbered = []
    for path in Path(attempt).glob("execution_attempt*.json"):
        match = re.fullmatch(r"execution_attempt(\d+)\.json", path.name)
        if match:
            numbered.append((int(match[1]), path))
    if not numbered:
        return None
    n, end_file = max(numbered)
    end = end_file.stat().st_mtime
    starts = [p.stat().st_mtime for p in Path(attempt).glob(f"execution_attempt{n}.*") if p.name != end_file.name]
    executions = record.get("executions") or []
    wall = executions[-1].get("wall_seconds") if executions and isinstance(executions[-1], dict) else None
    if isinstance(wall, (int, float)) and not isinstance(wall, bool):
        starts.append(end - wall)
    start = min(starts) if starts else end
    return start - WINDOW_SLACK, end + WINDOW_SLACK


def workspace_of(attempt):
    for name in ("final_workspace", "submitted_workspace"):
        if (attempt / name).is_dir():
            return attempt / name
    return None


def extra_sources(record_path, record, output_path, estimands):
    """[(label, {term: [token, ...]})] from final-execution files and stdout."""
    attempt = Path(record_path).parent
    output = Path(output_path)
    primary_hash = hashlib.sha256(output.read_bytes()).hexdigest() if output.is_file() else None
    workspace = workspace_of(attempt)
    window = final_execution_window(attempt, record)
    sources = []
    if workspace is not None and window is not None:
        for path in sorted(workspace.rglob("*")):
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
            found = answer_values(data.decode(errors="replace"), estimands)
            if found:
                sources.append((str(path.relative_to(attempt)), found))
    found = answer_values(final_stdout(attempt, record), estimands)
    if found:
        sources.append(("stdout", found))
    return sources


def final_stdout(attempt, record):
    """Stdout of the final execution: the record copy plus the latest saved stdout file."""
    stdout = ""
    executions = record.get("executions") or []
    if executions and isinstance(executions[-1], dict):
        stdout = executions[-1].get("stdout") or ""
    files = [p for p in Path(attempt).glob("execution_attempt*.stdout") if re.search(r"(\d+)", p.name)]
    if files:
        latest = max(files, key=lambda p: int(re.search(r"(\d+)", p.name)[1]))
        stdout += "\n" + latest.read_text(errors="replace")
    return stdout


def same_value(a, b):
    """Two tokens agree when they differ by less than the rounding of the coarser one."""
    x, y = number(a), number(b)
    if x is None or y is None:
        return x is None and y is None
    ka, kb = decimals(a), decimals(b)
    slack = 0.5 * 10.0 ** (-min(ka, kb)) if ka is not None and kb is not None and min(ka, kb) >= MIN_DECIMALS else 0.0
    return abs(x - y) <= max(slack, 1e-9 * max(abs(x), abs(y)))


def merge(values, tokens, sources, estimands):
    """Merge the primary values with the extra sources.

    Returns (merged, conflicts, recovered, used). A term keeps its primary value
    unless another source gives the same value with more decimal places. A term
    with two different values is a conflict and is left out of ``merged``.
    """
    merged = {term: dict(row) for term, row in values.items()}
    conflicts, recovered, used = [], [], set()
    for term in estimands:
        candidates = []
        if term in values and number(values[term].get("estimate")) is not None:
            candidates.append(("primary", tokens.get(term, values[term].get("estimate"))))
        for label, found in sources:
            candidates.extend((label, token) for token in found.get(term, []) if number(token) is not None)
        if not candidates:
            continue
        distinct = []
        for label, token in candidates:
            if not any(same_value(token, other) for _, other in distinct):
                distinct.append((label, token))
        if len(distinct) > 1:
            conflicts.append(term)
            merged.pop(term, None)
            continue
        label, token = max(candidates, key=lambda item: decimals(item[1]) or 0)
        if label != "primary":
            merged[term] = {"estimate": token}
            used.add(label)
            if not (term in values and number(values[term].get("estimate")) is not None):
                recovered.append(term)
    return merged, sorted(conflicts), sorted(recovered), sorted(used)

"""Offline rescore of the selected forty-task sample under proposed scoring rules.

No model or provider calls. No generated script is executed. The module reads
saved outputs, run folders, the main catalog, and
``data/analysis/rescore_v2/scoring_spec.json``. It writes only under
``data/analysis/rescore_v2`` and a checkpoint folder.

Variants
--------
V0  The stored primary rule (``code/paper/dataset.py`` extraction and
    ``production_analysis.score_estimates``). It must reproduce the stored
    ``primary`` column exactly, or the run stops.
V1  Per-task tolerances, exact integers, reference overrides, accepted
    alternative conventions, and the precision rule.
V2  V1 plus the output-channel rule.
V3  V2 plus standard-error scoring for the tasks whose tested decision is the SE.

Run: ``uv run python code/benchmark/rescore_v2.py``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/benchmark"))
sys.path.insert(0, str(ROOT / "code/paper"))

import artifact_repair  # noqa: E402

MAIN = ROOT / "data/analysis/main/data"
OUT = ROOT / "data/analysis/rescore_v2"
SPEC_PATH = OUT / "scoring_spec.json"
CHECKPOINT = ROOT / "quality_reports/audits/2026-09-23_rescore-and-v2/work/rescore"
VARIANTS = ("v0", "v1", "v2", "v3")
STRICT = re.compile(r"^\s*ANSWER:\s+(\S+)\s+estimate\s*=\s*([^\s]+)(.*)$", re.MULTILINE)
# Same pattern as code/paper/dataset.py PRINTED, with the line tail kept for se=.
PRINTED = re.compile(
    r"^\s*ANSWER:?\s+(\S+)\s+(?:estimate\s*=\s*)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?=\s|$)([^\n]*)",
    re.MULTILINE)
SE_TOKEN = re.compile(r"(?:^|\s)se\s*=\s*([^\s]+)")
TEXT_SUFFIX = {".txt", ".log", ".json", ".out", ".md", ".csv", ".tsv", ".smcl", ""}
SCRIPT_SUFFIX = {".do", ".py", ".r", ".ado"}
SKIP_NAMES = {"printed_output.txt", "verification.txt", "input.csv"}
LEGACY_PREFIX = ("prompt_", "call_", "provider_", "command_", "state", "execution_", "luna_", "codex_",
                 "claude_", "active_", "run_record", "started", "inherited", "after_attempt", "attempt",
                 "last_attempt", "review_score")


# --------------------------------------------------------------------------- tokens

class Num(str):
    """A JSON number kept as its raw text."""


def to_float(value):
    """Finite float or None. Booleans are not numbers here."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def decimals(token):
    """Decimal places of a numeric token (exponent included), or None."""
    if token is None:
        return None
    text = str(token).strip()
    try:
        exponent = Decimal(text).as_tuple().exponent
    except (InvalidOperation, ValueError):
        return None
    if not isinstance(exponent, int):
        return None
    return max(0, -exponent)


def significant_digits(token):
    try:
        digits = Decimal(str(token).strip()).as_tuple().digits
    except (InvalidOperation, ValueError):
        return None
    stripped = "".join(str(d) for d in digits).lstrip("0")
    return max(1, len(stripped))


MIN_DECIMALS = 4  # chosen limit: the jsonlite write_json default is 4 decimal places


def allowance(token, min_dec=MIN_DECIMALS):
    """Extra absolute error 0.5*10^-k for a token with k >= min_dec decimal places.

    A token with fewer decimal places (for example "0" or "0.06") gets no
    allowance, because it cannot come from the 4-decimal rounding this rule
    targets and would otherwise accept almost any value.
    """
    k = decimals(token)
    if k is None or k < min_dec:
        return 0.0
    return 0.5 * 10.0 ** (-k)


@dataclass
class Value:
    estimate: object = None
    est_token: str | None = None
    se: object = None
    se_token: str | None = None
    source: str = "primary"
    token_fallback: bool = False

    @property
    def number(self):
        return to_float(self.estimate)

    @property
    def se_number(self):
        return to_float(self.se)


def token_of(raw, number):
    """Raw token when present, else the shortest repr of the parsed float."""
    if isinstance(raw, str) and not isinstance(raw, bool):
        return str(raw), False
    if number is None:
        return None, False
    return repr(float(number)), True


# --------------------------------------------------------------------------- parsing

def join_stata(text):
    return re.sub(r"\n>\s?", "", text)


def json_rows(raw):
    """Estimate rows of a JSON result text, with raw number tokens."""
    data, rule = artifact_repair.repair_result_text(raw)
    if not isinstance(data, dict):
        return None, rule
    text = raw if rule == "none needed" else artifact_repair._apply(raw, rule.split("+"))
    try:
        tokens = json.loads(text, parse_float=Num, parse_int=Num)
    except ValueError:
        tokens = None
    rows = {}
    raw_rows = data.get("estimates", [])
    token_rows = tokens.get("estimates", []) if isinstance(tokens, dict) else []
    if not isinstance(raw_rows, list):
        return {}, rule
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict) or not isinstance(row.get("term"), str):
            continue
        trow = token_rows[index] if index < len(token_rows) and isinstance(token_rows[index], dict) else {}
        est, se = row.get("estimate"), row.get("se")
        est_tok, fb1 = token_of(trow.get("estimate") if isinstance(trow.get("estimate"), str) else None, to_float(est))
        se_tok, _ = token_of(trow.get("se") if isinstance(trow.get("se"), str) else None, to_float(se))
        rows[row["term"]] = Value(est, est_tok, se, se_tok, token_fallback=fb1)
    return rows, rule


def se_from_tail(tail):
    match = SE_TOKEN.search(" " + tail)
    return match[1] if match else None


def stata_all(text, estimands):
    """Every ANSWER value per requested term: {term: [Value, ...]}."""
    text = join_stata(text)
    found = {}
    for match in STRICT.finditer(text):
        if match[1] in estimands:
            found.setdefault(match[1], []).append(Value(match[2], match[2], se_from_tail(match[3]), se_from_tail(match[3])))
    for match in PRINTED.finditer(text):
        if match[1] in estimands:
            found.setdefault(match[1], []).append(Value(float(match[2]), match[2], se_from_tail(match[3]), se_from_tail(match[3])))
    return found


def primary_values(path, platform, estimands):
    """Primary-channel values, equal to the V0 extraction (dataset.extract)."""
    path = Path(path)
    if not path.is_file():
        return {}, "missing_output"
    raw = path.read_text(errors="replace")
    if path.suffix == ".json":
        rows, rule = json_rows(raw)
        if rows is None:
            return {}, "unreadable_json"
        return rows, rule
    text = join_stata(raw)
    values = {}
    for match in STRICT.finditer(text):
        values[match[1]] = Value(match[2], match[2], se_from_tail(match[3]), se_from_tail(match[3]))
    if platform == "stata":
        alternatives = {}
        for match in PRINTED.finditer(text):
            if match[1] in estimands:
                alternatives.setdefault(match[1], {}).setdefault(float(match[2]), (match[2], match[3]))
        for term, numbers in alternatives.items():
            if len(numbers) == 1 and term not in values:
                number, (tok, tail) = next(iter(numbers.items()))
                values[term] = Value(number, tok, se_from_tail(tail), se_from_tail(tail))
    return values, "printed_estimate_tokens"


def workspace_of(attempt):
    for name in ("final_workspace", "submitted_workspace"):
        if (attempt / name).is_dir():
            return attempt / name, False
    return attempt, True


WINDOW_SLACK = 2.0  # seconds; chosen limit around the final execution window


def final_execution_window(attempt, record):
    """(start, end) file times of the final harness execution, or None.

    The harness writes execution_attemptN.stdout/.stderr/.active.json when the
    execution starts and execution_attemptN.json when it ends. Workspace copies
    keep file modification times, so a file written by the final execution has
    a modification time inside this window.
    """
    numbered = []
    for p in attempt.glob("execution_attempt*.json"):
        match = re.fullmatch(r"execution_attempt(\d+)\.json", p.name)
        if match:
            numbered.append((int(match[1]), p))
    if not numbered:
        return None
    n, end_file = max(numbered)
    end = end_file.stat().st_mtime
    starts = [p.stat().st_mtime for p in attempt.glob(f"execution_attempt{n}.*")
              if p.name != end_file.name]
    executions = record.get("executions") or []
    wall = executions[-1].get("wall_seconds") if executions and isinstance(executions[-1], dict) else None
    if isinstance(wall, (int, float)):
        starts.append(end - wall)
    start = min(starts) if starts else end
    return start - WINDOW_SLACK, end + WINDOW_SLACK


def extra_sources(record_path, output_path, platform, estimands):
    """Values from other files the final execution left, and Stata stdout."""
    attempt = Path(record_path).parent
    output = Path(output_path)
    record = json.loads(Path(record_path).read_text())
    primary_hash = hashlib.sha256(output.read_bytes()).hexdigest() if output.is_file() else None
    workspace, legacy = workspace_of(attempt)
    window = final_execution_window(attempt, record)
    sources = []
    for f in sorted(workspace.rglob("*")):
        if not f.is_file() or f.name in SKIP_NAMES or f.suffix.lower() in SCRIPT_SUFFIX:
            continue
        if window is not None and not (window[0] <= f.stat().st_mtime <= window[1]):
            continue
        if f.suffix.lower() not in TEXT_SUFFIX or f.stat().st_size > 5_000_000:
            continue
        if output.is_file() and f.resolve() == output.resolve():
            continue
        if legacy and (f.name.startswith(LEGACY_PREFIX) or f.suffix == ".jsonl"
                       or (platform == "stata" and f.name == "result.json")):
            continue
        data = f.read_bytes()
        if primary_hash and hashlib.sha256(data).hexdigest() == primary_hash:
            continue
        text = data.decode(errors="replace")
        label = str(f.relative_to(attempt))
        if platform == "stata":
            found = stata_all(text, estimands)
        elif '"estimates"' in text:
            rows, _ = json_rows(text)
            found = {t: [v] for t, v in (rows or {}).items() if t in estimands}
        else:
            found = {}
        if found:
            sources.append((label, found))
    if platform == "stata":
        stdout = ""
        executions = record.get("executions") or []
        if executions and isinstance(executions[-1], dict):
            stdout = executions[-1].get("stdout") or ""
        files = sorted(attempt.glob("execution_attempt*.stdout"),
                       key=lambda p: int(re.search(r"(\d+)", p.name)[1]) if re.search(r"(\d+)", p.name) else 0)
        if files:
            stdout += "\n" + files[-1].read_text(errors="replace")
        found = stata_all(stdout, estimands)
        if found:
            sources.append(("stdout", found))
    return sources


def same_value(a: Value, b: Value, field_name="estimate"):
    x = a.number if field_name == "estimate" else a.se_number
    y = b.number if field_name == "estimate" else b.se_number
    if x is None or y is None:
        return x is None and y is None
    ta = a.est_token if field_name == "estimate" else a.se_token
    tb = b.est_token if field_name == "estimate" else b.se_token
    ka, kb = decimals(ta), decimals(tb)
    slack = 0.5 * 10.0 ** (-min(ka, kb)) if (ka is not None and kb is not None
                                             and min(ka, kb) >= MIN_DECIMALS) else 0.0
    return abs(x - y) <= max(slack, 1e-9 * max(abs(x), abs(y)))


def merge_channels(primary, sources, estimands, field_name="estimate"):
    """Union of values; returns (values, conflict_terms, recovered_terms)."""
    merged, conflicts, recovered = {}, [], []
    for term in estimands:
        candidates = []
        if term in primary:
            candidates.append(primary[term])
        for label, found in sources:
            for value in found.get(term, []):
                candidates.append(Value(value.estimate, value.est_token, value.se, value.se_token, source=label))
        usable = [v for v in candidates if (v.number if field_name == "estimate" else v.se_number) is not None]
        if not usable:
            if term in primary:
                merged[term] = primary[term]
            continue
        distinct = []
        for value in usable:
            if not any(same_value(value, other, field_name) for other in distinct):
                distinct.append(value)
        if len(distinct) > 1:
            conflicts.append(term)
            continue
        best = max(usable, key=lambda v: decimals(v.est_token if field_name == "estimate" else v.se_token) or 0)
        merged[term] = best
        if term not in primary or (primary[term].number if field_name == "estimate" else primary[term].se_number) is None:
            recovered.append(term)
    return merged, conflicts, recovered


# --------------------------------------------------------------------------- scoring

@dataclass
class TermCheck:
    term: str
    kind: str
    got: float | None
    token: str | None
    ref: float | None
    tol: float | None
    extra: float
    err: float | None
    passed: bool
    used_allowance: bool = False
    matched_alternative: bool = False
    reason: str = ""
    source: str = "primary"


@dataclass
class Score:
    passed: bool
    checks: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def term_rule(task_spec, term):
    rule = dict(rel_tol=task_spec.get("rel_tol", 0.01), abs_floor=task_spec.get("abs_floor", 0.0),
                exact_integer=False, compare_abs=False, accepted_values=None)
    rule.update(task_spec.get("terms", {}).get(term, {}))
    return rule


def reference_map(catalog_spec, task_spec):
    refs = {r["term"]: r for r in catalog_spec["reference"]["estimates"]}
    override = task_spec.get("reference_override") or {}
    return {t: (override[t] if t in override else r.get("estimate")) for t, r in refs.items()}


def check_value(term, kind, value: Value | None, candidates, rel_tol, abs_floor, exact, compare_abs, precision, min_dec):
    got = value.number if value is not None else None
    token = value.est_token if value is not None else None
    if kind == "se":
        got = value.se_number if value is not None else None
        token = value.se_token if value is not None else None
    if value is not None and kind != "se" and isinstance(value.estimate, bool):
        got = None
    first = candidates[0] if candidates else None
    if got is None:
        return TermCheck(term, kind, None, token, first, None, 0.0, None, False,
                         reason="se_missing" if kind == "se" else "missing_value",
                         source=value.source if value else "none")
    best = None
    for index, cand in enumerate(candidates):
        if cand is None:
            continue
        if exact:
            ok = float(got).is_integer() and float(cand).is_integer() and got == cand
            check = TermCheck(term, kind, got, token, cand, 0.0, 0.0, abs(got - cand), ok, matched_alternative=index > 0)
        else:
            g, c = (abs(got), abs(cand)) if compare_abs else (got, cand)
            tol = rel_tol * abs(c) + abs_floor
            extra = allowance(token, min_dec) if precision else 0.0
            err = abs(g - c)
            ok = err <= tol + extra
            check = TermCheck(term, kind, got, token, cand, tol, extra, err, ok,
                              used_allowance=ok and err > tol, matched_alternative=index > 0 and ok)
        if check.passed:
            check.source = value.source
            return check
        if best is None or (check.err is not None and best.err is not None and check.err < best.err):
            best = check
    best.source = value.source
    best.reason = "se_outside_tolerance" if kind == "se" else "outside_tolerance"
    return best


def score_run(task, values, catalog_spec, task_spec, precision=True, score_se=False, min_dec=MIN_DECIMALS):
    refs = reference_map(catalog_spec, task_spec)
    checks = []
    terms = task_spec.get("score_terms", catalog_spec["estimands"])
    for term in terms:
        rule = term_rule(task_spec, term)
        cands = rule["accepted_values"] or [refs.get(term)]
        checks.append(check_value(term, "estimate", values.get(term), cands, rule["rel_tol"], rule["abs_floor"],
                                  rule["exact_integer"], rule["compare_abs"], precision, min_dec))
    for name, derived in task_spec.get("derived_terms", {}).items():
        checks.append(derived_check(name, derived, values, precision, min_dec))
    if score_se and task_spec.get("se_scored"):
        for term, rule in task_spec.get("se_terms", {}).items():
            checks.append(check_value(term, "se", values.get(term), rule["accepted_values"], rule["rel_tol"],
                                      rule.get("abs_floor", 0.0), False, False, precision, min_dec))
    return Score(all(c.passed for c in checks), checks)


def derived_check(name, derived, values, precision, min_dec):
    if name == "change":
        v = values.get("forecast")
        got = v.number if v else None
        extra = allowance(v.est_token, min_dec) if (precision and v) else 0.0
        if got is not None:
            got -= derived["y_250"]
        token = v.est_token if v else None
    elif name == "half_width":
        lo, hi = values.get("lower_95"), values.get("upper_95")
        got = (hi.number - lo.number) / 2 if (lo and hi and lo.number is not None and hi.number is not None) else None
        extra = (allowance(lo.est_token, min_dec) + allowance(hi.est_token, min_dec)) / 2 if (precision and lo and hi) else 0.0
        token = None
    else:
        raise ValueError(f"Unknown derived term {name}")
    ref = derived["reference"]
    if got is None:
        return TermCheck(name, "derived", None, None, ref, None, 0.0, None, False, reason="missing_value")
    tol = derived["rel_tol"] * abs(ref)
    err = abs(got - ref)
    ok = err <= tol + extra
    return TermCheck(name, "derived", got, token, ref, tol, extra, err, ok, used_allowance=ok and err > tol,
                     reason="" if ok else "outside_tolerance")


# --------------------------------------------------------------------------- V0

def v0_scorer():
    """Return (extract, score) bound to the stored primary rule."""
    import dataset
    import production_analysis as analysis
    work = OUT / "work/scoring"
    work.mkdir(parents=True, exist_ok=True)
    c = analysis.runtime.load_runtime(work).c

    def score(output, platform, spec):
        values, rule, added, ambiguous = dataset.extract(Path(output), platform, spec, c.base)
        passed, terms, notes = analysis.score_estimates(values, spec, c.compare_numeric)
        return bool(passed), values, terms, notes
    return score


# --------------------------------------------------------------------------- run loop

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_file(value):
    """Resolve a runs.csv path; runs.csv stores paths relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def classify(v0_terms, check: TermCheck, task_spec, task):
    """Reason code for a term whose status differs from V0."""
    if check.kind == "se":
        return check.reason or "se_scored"
    if check.kind == "derived":
        return "derived_term"
    rule = term_rule(task_spec, check.term)
    if check.passed:
        if check.source != "primary":
            return "channel_recovered"
        if check.matched_alternative:
            return "alternative_convention"
        if check.used_allowance:
            return "precision_allowance"
        if task_spec.get("reference_override"):
            return "reference_override"
        return "other_pass"
    if check.reason == "channel_conflict":
        return "channel_conflict"
    if check.reason == "missing_value":
        return "missing_value"
    if rule["exact_integer"]:
        return "exact_integer"
    if task == "ttest_welch" and check.term == "t" and check.got is not None and check.ref is not None \
            and abs(abs(check.got) - abs(check.ref)) <= check.tol + check.extra:
        return "signed_compare"
    if task_spec.get("reference_override") and check.term in task_spec["reference_override"]:
        return "tolerance_with_reference_override"
    return "tolerance_tightened"


def flip_rows(row, task, variant, score, v0_pass, v0_terms, catalog_spec, task_spec):
    """Term rows that explain a run whose outcome differs from V0."""
    base = dict(position_id=row["position_id"], task=task, model=row["model"], platform=row["platform"],
                prompt=row["prompt"], agency=row["condition"], variant=variant,
                direction="fail_to_pass" if score.passed else "pass_to_fail")
    out = []

    def add(check=None, term=None, reason=None, v0=None):
        v0 = v0 or {}
        item = dict(base, term=term or check.term, kind=check.kind if check else "estimate",
                    got=check.got if check else v0.get("got_estimate"), token=check.token if check else None,
                    ref=check.ref if check else v0.get("ref_estimate"), tolerance=check.tol if check else None,
                    allowance=check.extra if check else None, error=check.err if check else v0.get("coef_err"),
                    v0_got=v0.get("got_estimate"), v0_ref=v0.get("ref_estimate"), v0_tolerance=v0.get("coef_tol"),
                    v0_term_pass=v0.get("coef_pass"), source=check.source if check else "primary", reason=reason)
        out.append(item)

    if not score.passed:
        # V0 passed: every failing check in the variant is a cause.
        for check in score.checks:
            if not check.passed:
                reason = check.reason if check.kind == "channel" else classify(v0_terms, check, task_spec, task)
                add(check, reason=reason, v0=v0_terms.get(check.term, {}))
        return out
    # V0 failed: every term that failed or was missing under V0 is a cause.
    checks = {c.term: c for c in score.checks if c.kind == "estimate"}
    for term in catalog_spec["estimands"]:
        detail = v0_terms.get(term)
        if detail and detail.get("coef_pass") is not False and detail.get("estimate_pass") is not False:
            continue
        check = checks.get(term)
        if check is None:
            add(term=term, reason="term_not_scored", v0=detail)
        else:
            add(check, reason=classify(v0_terms, check, task_spec, task), v0=detail)
    for check in score.checks:
        if check.kind == "derived":
            add(check, reason="derived_term")
    return out


def process_task(task, rows, catalog, spec, v0score, min_dec=MIN_DECIMALS):
    catalog_spec = catalog[task]
    task_spec = spec["tasks"][task]
    estimands = catalog_spec["estimands"]
    out_rows, flips, mismatches = [], [], []
    for row in rows:
        output = repo_file(row["output_path"])
        if output.is_file() and pd.notna(row.get("output_sha256")) and file_sha(output) != row["output_sha256"]:
            raise ValueError(f"Saved output changed: {output}")
        v0_pass, v0_values, v0_terms, _ = v0score(output, row["platform"], catalog_spec)
        values, rule = primary_values(output, row["platform"], estimands)
        # Self-check: the primary channel equals the V0 extraction for every term.
        for term in set(values) | set(v0_values):
            a = to_float(values[term].estimate) if term in values else None
            b = to_float(v0_values[term].get("estimate")) if term in v0_values else None
            if isinstance(v0_values.get(term, {}).get("estimate"), bool):
                b = None
            if not (a == b or (a is not None and b is not None and math.isclose(a, b, rel_tol=0, abs_tol=0))):
                if not (a is None and b is None):
                    mismatches.append(dict(position_id=row["position_id"], term=term, primary=a, v0=b))
        s1 = score_run(task, values, catalog_spec, task_spec, precision=True, min_dec=min_dec)
        s1_np = score_run(task, values, catalog_spec, task_spec, precision=False)
        sources = extra_sources(repo_file(row["record_path"]), output, row["platform"], estimands)
        merged, conflicts, recovered = merge_channels(values, sources, estimands)
        se_merged, se_conflicts, _ = merge_channels(values, sources, estimands, "se")
        for term, value in se_merged.items():
            if term in merged:
                merged[term] = Value(merged[term].estimate, merged[term].est_token, value.se, value.se_token,
                                     source=merged[term].source, token_fallback=merged[term].token_fallback)
        s2 = score_run(task, merged, catalog_spec, task_spec, precision=True, min_dec=min_dec)
        if conflicts:
            s2.passed = False
            s2.checks.append(TermCheck("|".join(conflicts), "channel", None, None, None, None, 0.0, None, False,
                                       reason="channel_conflict"))
        s3 = score_run(task, merged, catalog_spec, task_spec, precision=True, score_se=True, min_dec=min_dec)
        se_conflict_scored = [t for t in se_conflicts if t in task_spec.get("se_terms", {})] if task_spec.get("se_scored") else []
        if conflicts or se_conflict_scored:
            s3.passed = False
            s3.checks.append(TermCheck("|".join(conflicts + se_conflict_scored), "channel", None, None, None, None, 0.0,
                                       None, False, reason="channel_conflict"))
        record = {k: row[k] for k in ("position_id", "task", "model", "platform", "prompt", "condition", "agency",
                                       "rep", "global_rep", "record_path", "output_path")}
        record.update(primary_stored=bool(row["primary"]), v0=v0_pass, v1=s1.passed, v2=s2.passed, v3=s3.passed,
                      v1_no_precision=s1_np.passed,
                      v1_precision_used=s1.passed and not s1_np.passed,
                      v1_allowance_terms="|".join(c.term for c in s1.checks if c.used_allowance),
                      v1_allowance_max_k=max([decimals(c.token) or 0 for c in s1.checks if c.used_allowance], default=None),
                      v1_allowance_min_sig=min([significant_digits(c.token) or 0 for c in s1.checks
                                                if c.used_allowance and c.token], default=None),
                      v1_token_fallback=any(v.token_fallback for v in values.values()),
                      v2_extra_sources="|".join(label for label, _ in sources),
                      v2_recovered_terms="|".join(recovered), v2_conflict_terms="|".join(conflicts),
                      v2_channel_used=bool(recovered) and s2.passed and not s1.passed,
                      v3_se_fail_terms="|".join(c.term for c in s3.checks if c.kind == "se" and not c.passed),
                      extraction=rule)
        for name, score in (("v1", s1), ("v2", s2), ("v3", s3)):
            record[f"{name}_fail_terms"] = "|".join(f"{c.term}:{c.reason}" for c in score.checks if not c.passed)
        out_rows.append(record)
        for name, score in (("v1", s1), ("v2", s2), ("v3", s3)):
            if score.passed != v0_pass:
                flips.extend(flip_rows(row, task, name, score, v0_pass, v0_terms, catalog_spec, task_spec))
    return out_rows, flips, mismatches


# --------------------------------------------------------------------------- results

def rates(data):
    dims = {"overall": [], "model": ["model"], "language": ["platform"], "agency": ["agency"], "prompt": ["prompt"],
            "model_x_agency": ["model", "agency"], "task": ["task"]}
    tables = {}
    for name, cols in dims.items():
        rows = []
        groups = [((), data)] if not cols else data.groupby(cols)
        for key, frame in groups:
            key = key if isinstance(key, tuple) else (key,)
            for v in VARIANTS:
                rows.append(dict(dimension=name, level=" x ".join(map(str, key)) if key else "all", variant=v,
                                 n=len(frame), successes=int(frame[v].sum()), rate=float(frame[v].mean())))
        tables[name] = pd.DataFrame(rows)
    return tables


CONTRASTS = {
    "sonnet_minus_luna": ("model", "sonnet", "luna"),
    "self_repair_minus_chatbot": ("agency", "self_repair", "chatbot"),
    "agent_minus_chatbot": ("agency", "agent", "chatbot"),
    "few_shot_minus_zero_shot": ("prompt", "few_shot", "zero_shot"),
}


def contrasts(data):
    import statsmodels.formula.api as smf
    rows = []
    for v in VARIANTS:
        for name, (col, treated, control) in CONTRASTS.items():
            sub = data[data[col].isin([treated, control])].copy()
            sub["d"] = (sub[col] == treated).astype(int)
            sub["y"] = sub[v].astype(int)
            diff = sub.loc[sub.d == 1, "y"].mean() - sub.loc[sub.d == 0, "y"].mean()
            fit = smf.ols("y ~ d + C(task)", sub).fit(cov_type="cluster", cov_kwds={"groups": sub["task"]})
            rows.append(dict(variant=v, contrast=name, n=len(sub), clusters=sub.task.nunique(),
                             treated_rate=float(sub.loc[sub.d == 1, "y"].mean()),
                             control_rate=float(sub.loc[sub.d == 0, "y"].mean()),
                             diff_means=float(diff), coef_task_fe=float(fit.params["d"]),
                             se_cluster_task=float(fit.bse["d"]),
                             ci_low=float(fit.conf_int().loc["d", 0]), ci_high=float(fit.conf_int().loc["d", 1])))
    return pd.DataFrame(rows)


def code_hash():
    """Hash of the scoring code only, so table changes reuse task checkpoints."""
    text = Path(__file__).read_text()
    return hashlib.sha256(text.split("# " + "-" * 75 + " results")[0].encode()).hexdigest()[:16]


def run(out=OUT, checkpoint=CHECKPOINT, tasks=None, min_dec=MIN_DECIMALS):
    started = time.time()
    out = Path(out)
    checkpoint = Path(checkpoint)
    (checkpoint / "tasks").mkdir(parents=True, exist_ok=True)
    spec = json.loads(SPEC_PATH.read_text())
    catalog = json.loads((MAIN / "catalog.json").read_text())
    runs = pd.read_csv(MAIN / "runs.csv")
    if len(runs) != 14400:
        raise ValueError("Expected the 14,400 selected runs")
    missing = set(catalog) - set(spec["tasks"])
    if missing:
        raise ValueError(f"Spec lacks tasks: {sorted(missing)}")
    stamp = dict(code=code_hash(), spec=file_sha(SPEC_PATH)[:16], min_dec=min_dec)
    progress_path = checkpoint / "progress.jsonl"
    done = {}
    if progress_path.is_file():
        for line in progress_path.read_text().splitlines():
            item = json.loads(line)
            if item.get("unit", "").startswith("task:") and all(item.get(k) == v for k, v in stamp.items()):
                done[item["unit"][5:]] = item
    v0score = v0_scorer()
    all_rows, all_flips = [], []
    for task in sorted(tasks or catalog):
        cache = checkpoint / "tasks" / f"{task}.json"
        if task in done and cache.is_file():
            stored = json.loads(cache.read_text())
            all_rows += stored["rows"]
            all_flips += stored["flips"]
            continue
        t0 = time.time()
        rows = runs[runs.task == task].to_dict("records")
        out_rows, flips, mismatches = process_task(task, rows, catalog, spec, v0score, min_dec)
        if mismatches:
            raise ValueError(f"Primary extraction differs from V0 for {task}: {mismatches[:3]}")
        bad = [r["position_id"] for r in out_rows if r["v0"] != r["primary_stored"]]
        if bad:
            raise ValueError(f"V0 does not reproduce the stored primary for {task}: {bad[:5]}")
        cache.write_text(json.dumps(dict(rows=out_rows, flips=flips), default=str) + "\n")
        with progress_path.open("a") as handle:
            handle.write(json.dumps(dict(unit=f"task:{task}", n=len(out_rows), seconds=round(time.time() - t0, 2),
                                         **{v: int(sum(r[v] for r in out_rows)) for v in VARIANTS}, **stamp)) + "\n")
        all_rows += out_rows
        all_flips += flips
    data = pd.DataFrame(all_rows)
    flips = pd.DataFrame(all_flips)
    if tasks is None:
        write_outputs(data, flips, spec, out, started, stamp)
    return data, flips


def write_outputs(data, flips, spec, out, started, stamp):
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(MAIN / "runs.csv")
    stored = runs.set_index("position_id").primary.astype(bool)
    mine = data.set_index("position_id").v0.reindex(stored.index)
    if len(data) != len(runs) or mine.isna().any() or (mine.astype(bool) != stored).any():
        raise ValueError("V0 does not reproduce the stored primary outcome")
    data.to_csv(out / "runs_rescored.csv", index=False)
    flips.to_csv(out / "flips.csv", index=False)
    tables = rates(data)
    for name, table in tables.items():
        table.to_csv(out / f"rates_{name}.csv", index=False)
    con = contrasts(data)
    con.to_csv(out / "contrasts.csv", index=False)
    run_flips = []
    for v in ("v1", "v2", "v3"):
        changed = data[data[v] != data.v0]
        run_flips.append(dict(variant=v, pass_to_fail=int((changed.v0 & ~changed[v]).sum()),
                              fail_to_pass=int((~changed.v0 & changed[v]).sum())))
    reason_counts = {}
    if len(flips):
        per_run = flips.groupby(["variant", "direction", "position_id"]).reason.agg(lambda s: "|".join(sorted(set(s))))
        reason_counts = {f"{v}:{d}": per_run.loc[(v, d)].value_counts().to_dict()
                         for v, d in per_run.index.droplevel(2).unique()}
    se_tasks = sorted(t for t, s in spec["tasks"].items() if s.get("se_scored"))
    precision_runs = data[data.v1_precision_used]
    results = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"), provider_calls=0, runs=len(data), stamp=stamp,
        runtime_seconds=round(time.time() - started, 1),
        v0_reproduces_stored_primary=True,
        success=dict({v: dict(successes=int(data[v].sum()), rate=float(data[v].mean())) for v in VARIANTS}),
        by_model={v: data.groupby("model")[v].mean().round(6).to_dict() for v in VARIANTS},
        by_task_successes={t: {v: int(g[v].sum()) for v in VARIANTS} for t, g in data.groupby("task")},
        run_flips_vs_v0=run_flips,
        flip_reasons_per_run=reason_counts,
        precision_rule=dict(
            runs_passing_only_with_allowance_v1=int(len(precision_runs)),
            by_platform=precision_runs.platform.value_counts().to_dict(),
            by_task=precision_runs.task.value_counts().to_dict(),
            by_max_decimals=precision_runs.v1_allowance_max_k.value_counts().sort_index().to_dict(),
            by_min_significant_digits=precision_runs.v1_allowance_min_sig.value_counts().sort_index().to_dict(),
            runs_with_token_fallback=int(data.v1_token_fallback.sum())),
        channel_rule=dict(
            runs_with_extra_sources=int((data.v2_extra_sources.fillna("") != "").sum()),
            runs_recovered_v1_fail_to_v2_pass=int((~data.v1 & data.v2).sum()),
            runs_lost_v1_pass_to_v2_fail=int((data.v1 & ~data.v2).sum()),
            runs_with_conflicts=int((data.v2_conflict_terms.fillna("") != "").sum()),
            recovered_by_platform=data[~data.v1 & data.v2].platform.value_counts().to_dict(),
            recovered_by_task=data[~data.v1 & data.v2].task.value_counts().to_dict(),
            conflicts_by_task=data[data.v2_conflict_terms.fillna("") != ""].task.value_counts().to_dict()),
        se_scored_tasks=se_tasks,
        se_rule=dict(runs_lost_v2_pass_to_v3_fail=int((data.v2 & ~data.v3).sum()),
                     lost_by_task=data[data.v2 & ~data.v3].task.value_counts().to_dict()),
        contrasts=con.to_dict("records"),
        outputs=["runs_rescored.csv", "flips.csv", "contrasts.csv"] + [f"rates_{n}.csv" for n in tables],
    )
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(json.dumps({k: results[k] for k in ("success", "run_flips_vs_v0", "runtime_seconds")}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="*", help="Score only these tasks (no final tables).")
    parser.add_argument("--min-decimals", type=int, default=MIN_DECIMALS,
                        help="Smallest number of decimal places that earns the precision allowance.")
    args = parser.parse_args()
    run(tasks=args.tasks, min_dec=args.min_decimals)

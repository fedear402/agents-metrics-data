"""Standard-error check for the "channel_se" rule, the canonical outcome.

Decision record 2026-09-25_se-scored-primary-outcome.md made "channel_se" the
primary outcome. This module adds the standard-error check of variant V3 in
``code/benchmark/rescore_v2.py``. It applies to the nine tasks whose named
decision is the standard error (``se_scored`` in
``data/analysis/rescore_v2/scoring_spec.json``). Each scored SE term must match
one accepted convention within the V3 relative tolerance. V3 also gives an SE
token with at least four decimal places the rounding allowance 0.5*10^-k.

Standard errors are read from the same sources as the estimates:

- the ``se`` field of a JSON result row, or the ``se=`` token of an ANSWER line;
- for Stata, the files and stdout that the canonical output-channel rule reads
  (``output_channels.extra_sources``). Two different SE values for one scored
  term fail the run, as in V3.

The V1 changes to point-estimate tolerances, references, and conventions are
not part of this rule. The module reads saved files only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescore_v2 as v3  # noqa: E402
import output_channels as channels  # noqa: E402

SPEC_PATH = ROOT / "data/analysis/rescore_v2/scoring_spec.json"
PRECISION_ALLOWANCE = True  # the V3 SE check applies the precision allowance to SE tokens
_SPEC = None


def spec():
    """The per-task SE rules of variant V3: {task: {term: rule}} for the nine tasks."""
    global _SPEC
    if _SPEC is None:
        tasks = json.loads(SPEC_PATH.read_text())["tasks"]
        _SPEC = {task: rule["se_terms"] for task, rule in tasks.items() if rule.get("se_scored")}
        if len(_SPEC) != 9:
            raise ValueError(f"Expected nine SE-scored tasks, found {len(_SPEC)}")
    return _SPEC


def spec_sha256():
    return hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()


def se_terms_for(estimate_spec):
    """(task_id, SE rules) for a catalog task spec; empty rules for the other 31 tasks."""
    task = estimate_spec.get("task_id")
    terms = spec().get(task, {})
    if terms and not set(terms) <= set(estimate_spec["estimands"]):
        raise ValueError(f"SE terms are not estimands of {task}")
    return task, terms


def _stata_sources(record_path, record, output, estimands, platform):
    """SE-bearing values from the sources that the canonical channel rule reads."""
    if platform != "stata":
        return []
    attempt = Path(record_path).parent
    sources = []
    for label, _ in channels.extra_sources(record_path, record, output, estimands):
        if label == "stdout":
            text = channels.final_stdout(attempt, record)
        else:
            text = (attempt / label).read_text(errors="replace")
        found = v3.stata_all(text, estimands)
        if found:
            sources.append((label, found))
    return sources


def check(output, record_path, record, platform, estimate_spec, use_channels=True):
    """Score the standard errors of one saved run.

    Returns dict(task, checked, failed, conflicts, notes, details). ``failed``
    is empty when the task has no SE decision.
    """
    task, terms = se_terms_for(estimate_spec)
    result = dict(task=task, checked=sorted(terms), failed=[], conflicts=[], notes=[], details={})
    if not terms:
        return result
    output = Path(output)
    estimands = estimate_spec["estimands"]
    values, _ = v3.primary_values(output, platform, estimands)
    sources = _stata_sources(record_path, record, output, estimands, platform) if use_channels else []
    if sources:
        values, conflicts, _ = v3.merge_channels(values, sources, estimands, "se")
        result["conflicts"] = sorted(t for t in conflicts if t in terms)
    for term in sorted(terms):
        rule = terms[term]
        item = v3.check_value(term, "se", values.get(term), rule["accepted_values"], rule["rel_tol"],
                              rule.get("abs_floor", 0.0), False, False, PRECISION_ALLOWANCE, v3.MIN_DECIMALS)
        if term in result["conflicts"]:
            item.passed, item.reason = False, "se_channel_conflict"
        result["details"][term] = dict(se_got=item.got, se_token=item.token, se_ref=item.ref, se_tol=item.tol,
                                       se_allowance=item.extra, se_err=item.err, se_pass=bool(item.passed),
                                       se_reason=item.reason, se_source=item.source,
                                       se_alternative=bool(item.matched_alternative))
        if not item.passed:
            result["failed"].append(term)
            result["notes"].append(f"{term}: standard error {item.reason or 'outside_tolerance'}")
    return result

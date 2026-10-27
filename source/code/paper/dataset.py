"""Recompute the selected forty-task sample from immutable saved outputs.

No provider calls or generated-script execution occurs here. A replacement
collection enters the sample only when all 720 planned responses are terminal.

The output rule sets what the scorer checks:
"channel_se" (default and canonical, decision record
2026-09-25_se-scored-primary-outcome.md) is the "channel" rule plus the V3
standard-error check (se_scoring.py) in the nine tasks whose named decision is
the standard error.
"channel" (decision record 2026-09-24_output-channel-rule.md) scores the
estimates only. For a Stata run it also reads ANSWER lines that the final
execution wrote to another file or to stdout.
"log" reproduces the earlier estimate-only outcome, which read the Stata log only.

Every rule compares each scored estimate in the same way (decision record
2026-10-02_scoring-without-exceptions.md): the Welch t statistic keeps its sign,
and the Granger denominator degrees of freedom must equal the reference value.
general_rule() switches off the two exceptions of strict_posthoc.py for the
paper scoring only.

runs.csv, selected_records.csv, and selection_lineage.csv store record_path,
output_path, and manifest_path relative to the repository root; resolve them
with repo_path(). runs.csv carries no cost column: the paper cost of every run
is in run_pricing.csv (production_paper_pricing.py). Run alone, this module
builds the paper sample with the 720 September 21 replacements; --baseline
builds the earlier sample without them.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import itertools
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import production_analysis as analysis
import saved_response_analysis as saved
import output_channels as channels
import se_scoring

OUTPUT_RULES = ("channel", "log", "channel_se")
OUTPUT_RULE = "channel_se"
DESIGN = ROOT / "data/production/main"
OUT = ROOT / "data/analysis/main"
REPAIRS = ROOT / "data/collections/2026-09-21_production-repairs"
CELL = ["task", "model", "platform", "prompt", "condition"]
PRINTED = re.compile(r"^\s*ANSWER:?\s+(\S+)\s+(?:estimate\s*=\s*)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\s|$)", re.MULTILINE)
ALIASES = json.loads((DESIGN / "path_aliases.json").read_text())
# Columns of baseline_runs.csv and of token_cost() that hold earlier cost ledgers.
# The paper prices every run in run_pricing.csv, so runs.csv drops them.
STALE_COLUMNS = ("cost_usd", "sonnet_cost_usd", "cost_pricing_complete", "cost_pricing_issue",
                 "priced_cached_input_tokens", "priced_cache_write_input_tokens", "estimate_only")


def repo_relative(value):
    """A path under the repository as a string relative to its root; other paths unchanged."""
    path = Path(value)
    if path.is_absolute() and path.is_relative_to(ROOT):
        return path.relative_to(ROOT).as_posix()
    return str(value)


def repo_path(value):
    """Resolve a stored path, relative to the repository root unless absolute."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def canonical_path(value):
    """Resolve saved absolute and relative report aliases without editing records."""
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    mapped = ALIASES.get(str(path), ALIASES.get(str(value)))
    if mapped:
        path = Path(mapped)
    reports = ROOT / "quality_reports/reports"
    if path.is_relative_to(reports):
        # The data-centralization migration retained each collection suffix.
        # Callers verify source/output hashes before using the mapped file.
        path = ROOT / "data/collections" / path.relative_to(reports)
    return path.resolve()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def extract(path, platform, spec, base):
    """Retain the audited printed-value rule used in the forty-task build."""
    values, rule = analysis.read_output(path, base)
    added, ambiguous = [], []
    if platform == "stata" and path.is_file():
        alternatives = {}
        text = re.sub(r"\n>\s?", "", path.read_text(errors="replace"))
        for match in PRINTED.finditer(text):
            if match[1] in spec["estimands"]:
                alternatives.setdefault(match[1], set()).add(float(match[2]))
        for term, numbers in alternatives.items():
            if len(numbers) > 1:
                ambiguous.append(term)
            elif term not in values:
                values[term] = {"estimate": next(iter(numbers))}
                added.append(term)
    return values, rule, added, ambiguous


@contextmanager
def general_rule(sp):
    """Score without the magnitude-only Welch comparison and the Granger 2x df alternative."""
    saved_rules = (sp.ABS_COMPARE_ESTIMANDS, sp.MULTIPLE_CONVENTION_ESTIMANDS)
    sp.ABS_COMPARE_ESTIMANDS, sp.MULTIPLE_CONVENTION_ESTIMANDS = set(), {}
    try:
        yield
    finally:
        sp.ABS_COMPARE_ESTIMANDS, sp.MULTIPLE_CONVENTION_ESTIMANDS = saved_rules


def score_output(output, record_path, record, platform, spec, c, output_rule=OUTPUT_RULE):
    """Score one saved run under the selected output rule.

    The "log" rule is the earlier outcome. The "channel" rule adds, for Stata
    runs only, the ANSWER lines of the final execution in other files and on
    stdout (output_channels.py). A conflict between sources fails the run.
    The "channel_se" rule (the default) is the "channel" rule plus the
    standard-error check of se_scoring.py. It changes only the nine tasks with
    an SE decision.
    """
    if output_rule not in OUTPUT_RULES:
        raise ValueError(f"Unknown output rule: {output_rule}")
    output = Path(output)
    values, rule, added, ambiguous = extract(output, platform, spec, c.base)
    sources, conflicts, recovered, used = [], [], [], []
    use_channels = output_rule in ("channel", "channel_se")
    if use_channels and platform == "stata":
        sources = channels.extra_sources(record_path, record, output, spec["estimands"])
        if sources:
            tokens = channels.primary_tokens(output, values)
            values, conflicts, recovered, used = channels.merge(values, tokens, sources, spec["estimands"])
    with general_rule(c.sp):
        passed, terms, notes = analysis.score_estimates(values, spec, c.compare_numeric)
    if conflicts:
        passed = False
        notes = list(notes) + [f"{term}: conflicting values in output channels" for term in conflicts]
    se = dict(checked=[], failed=[], conflicts=[], notes=[], details={})
    if output_rule == "channel_se":
        se = se_scoring.check(output, record_path, record, platform, spec, use_channels=True)
        if se["failed"]:
            passed = False
            notes = list(notes) + se["notes"]
    delivered = all(term in values and c.sp.number_or_none(values[term].get("estimate")) is not None
                    for term in spec["estimands"])
    return dict(passed=bool(passed), terms=terms, notes=notes, values=values, extraction=rule, added=added,
                ambiguous=ambiguous, delivered=delivered, output_rule=output_rule,
                channel_sources=[label for label, _ in sources], channel_used=used,
                channel_recovered=recovered, channel_conflicts=conflicts,
                se_checked=se["checked"], se_failed=se["failed"], se_conflicts=se["conflicts"],
                se_details=se["details"])


def replacement_rows(catalog, runtime):
    positions = json.loads((REPAIRS / "planned_positions.json").read_text())
    if len(positions) != 720 or len({saved.key(p) for p in positions}) != 720:
        raise ValueError("Replacement plan must contain 720 distinct positions")
    expected = set(itertools.product(("hard_panel_lp_iv", "lp04_ppml"), ("luna", "sonnet"),
        ("python", "r", "stata"), ("zero_shot", "few_shot"), ("chatbot", "self_repair", "agent"), range(10)))
    if {saved.key(p) for p in positions} != expected:
        raise ValueError("Replacement plan differs from the complete approved design")
    entries = {str(Path(e["manifest"]).resolve()): e for e in json.loads((REPAIRS / "collections.json").read_text())}
    rows, manifests, collection_specs = [], {}, {}
    for p in positions:
        mp = Path(p["manifest"]).resolve()
        if mp not in manifests:
            m, digest = saved.read_hashed(mp)
            if analysis.runtime.digest(m["protocol"]) != m["protocol_sha256"]:
                raise ValueError(f"Changed protocol: {mp}")
            if m["protocol_sha256"] != entries[str(mp)]["protocol_sha256"]:
                raise ValueError(f"Protocol differs from the collection index: {mp}")
            if m["jobs"] != m["protocol"]["planned_jobs"]:
                raise ValueError(f"Changed assignments: {mp}")
            manifests[mp] = (m, digest)
        m, _ = manifests[mp]
        if m["protocol_sha256"] != p["protocol_sha256"]:
            raise ValueError(f"Protocol differs from the selected position: {p}")
        spec = m["protocol"]["tasks"][p["task"]]
        old = catalog[p["task"]]
        # The collection also asks for diagnostic quantities. Retain the
        # substantive primary targets that were selected before these calls.
        collection_specs[p["task"]] = spec
        for term in old["reference"]["estimates"]:
            matches = [r for r in spec["reference"]["estimates"] if r["term"] == term["term"]]
            if len(matches) != 1 or matches[0]["estimate"] != term["estimate"]:
                raise ValueError(f"Changed reference: {p['task']}/{term['term']}")
        jobs = [j for j in m["jobs"] if saved.key(j) == saved.key(p)]
        if len(jobs) != 1:
            raise ValueError(f"Position has no unique assignment: {p}")
        selected = saved.fresh_record(dict(job=jobs[0], protocol_sha256=m["protocol_sha256"],
            folder=str(mp.parent / "runs" / m["protocol_sha256"] / analysis.runtime.job_id(jobs[0]))))
        if selected["record"] is None:
            raise ValueError(f"Replacement collection is incomplete: {saved.key(p)}")
        row, _ = saved.scored_row(dict(p, status="fresh"), selected, old, runtime)
        row.update(agency=row["condition"], rep=row["global_rep"], manifest_path=str(mp),
                   protocol_sha256=m["protocol_sha256"])
        rows.append(row)
    for task, spec in collection_specs.items():
        updated = copy.deepcopy(spec)
        for field in ("estimands", "reference", "exact_integer_terms", "collection_estimands", "primary_scoring_note"):
            if field in catalog[task]:
                updated[field] = catalog[task][field]
        catalog[task] = updated
    return pd.DataFrame(rows), [dict(path=repo_relative(p), sha256=h) for p, (_, h) in manifests.items()]


def build(out=OUT, use_replacements=True, output_rule=OUTPUT_RULE):
    if output_rule not in OUTPUT_RULES:
        raise ValueError(f"Unknown output rule: {output_rule}")
    out = Path(out)
    destination = out / "data"
    destination.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(DESIGN / "baseline_runs.csv")
    catalog = json.loads((DESIGN / "catalog.json").read_text())
    c = analysis.runtime.load_runtime(out / "work/scoring").c
    data, manifests = baseline.copy(), []
    if use_replacements:
        fresh, manifests = replacement_rows(catalog, c)
        tasks = set(fresh.task)
        if tasks != {"hard_panel_lp_iv", "lp04_ppml"}:
            raise ValueError("Unexpected replacement tasks")
        data = pd.concat([baseline[~baseline.task.isin(tasks)], fresh], ignore_index=True)
    if len(data) != 14400 or data.task.nunique() != 40 or not data.groupby(CELL).size().eq(10).all():
        raise ValueError("The sample must contain forty complete tasks and ten repetitions per cell")
    if data.duplicated(CELL + ["global_rep"]).any() or data.record_path.nunique() != len(data):
        raise ValueError("Duplicate assignment or raw response")
    rows, checks, audit, se_checks = [], [], [], []
    for row in data.to_dict("records"):
        path = canonical_path(row["record_path"])
        record, digest = saved.read_hashed(path, row["record_sha256"])
        output = canonical_path(row["output_path"])
        output_hash = analysis.sha(output) if output.is_file() else None
        expected = row.get("output_sha256")
        if pd.notna(expected) and expected != output_hash:
            raise ValueError(f"Saved output changed: {output}")
        spec = catalog[row["task"]]
        scored = score_output(output, path, record, row["platform"], spec, c, output_rule)
        passed, terms, notes, delivered = scored["passed"], scored["terms"], scored["notes"], scored["delivered"]
        rule, added, ambiguous = scored["extraction"], scored["added"], scored["ambiguous"]
        previous = bool(row["primary"])
        audit.append(dict(position_id=row["position_id"], previous_primary=previous, primary=bool(passed),
                          changed=previous != bool(passed), added_terms="|".join(added), ambiguous_terms="|".join(ambiguous),
                          channel_sources="|".join(scored["channel_sources"]),
                          channel_used="|".join(scored["channel_used"]),
                          channel_recovered_terms="|".join(scored["channel_recovered"]),
                          channel_conflict_terms="|".join(scored["channel_conflicts"])))
        if output_rule == "channel_se":
            audit[-1].update(se_failed_terms="|".join(scored["se_failed"]),
                             se_conflict_terms="|".join(scored["se_conflicts"]))
        se_checks.extend(dict(position_id=row["position_id"], task=row["task"], term=term, **detail)
                         for term, detail in scored["se_details"].items())
        row.update(record_path=repo_relative(path), record_sha256=digest, output_path=repo_relative(output),
                   output_sha256=output_hash, primary=bool(passed), y=int(passed), output_available=delivered,
                   output_file_exists=output.is_file(), extraction=rule, scoring_notes=" | ".join(notes),
                   output_rule=output_rule, channel_used="|".join(scored["channel_used"]),
                   failure_category="Success" if passed else "Silent failure" if row["execution_ok"] and delivered else "Loud failure",
                   **analysis.token_cost(record, row["model"]))
        if pd.notna(row.get("manifest_path")):
            row["manifest_path"] = repo_relative(canonical_path(row["manifest_path"]))
        for column in STALE_COLUMNS:
            row.pop(column, None)
        rows.append(row)
        checks.extend(dict(position_id=row["position_id"], task=row["task"], term=term, **detail) for term, detail in terms.items())
    rescoring = pd.DataFrame(audit)
    rescoring.to_csv(destination / "rescoring_audit.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(destination / "runs.csv", index=False)
    pd.DataFrame(checks).to_csv(destination / "term_checks.csv", index=False)
    if output_rule == "channel_se":
        pd.DataFrame(se_checks).to_csv(destination / "se_checks.csv", index=False)
    write_json(destination / "catalog.json", catalog)
    write_json(destination / "paper_exclusions.json", dict(tasks=[], already_excluded=["d1_attendance", "dynpanel_ab"]))
    receipt = dict(planned=14400, analyzed=len(result), source_tasks=40, provider_calls=0,
        production_repairs_complete=use_replacements, replaced_positions=720 if use_replacements else 0,
        primary_successes=int(result.primary.sum()), rescoring_changes=int(rescoring.changed.sum()),
        scoring="All selected substantive estimates; audited Stata printed-value parsing."
                + (" Standard errors in the nine tasks whose named decision is the standard error."
                   if output_rule == "channel_se" else ""),
        output_rule=output_rule,
        channel_used_runs=int((rescoring.channel_used != "").sum()),
        channel_conflict_runs=int((rescoring.channel_conflict_terms != "").sum()),
        baseline_sha256=analysis.sha(DESIGN / "baseline_runs.csv"), catalog_sha256=analysis.sha(DESIGN / "catalog.json"),
        sources=[dict(path=str(p.relative_to(ROOT)), sha256=analysis.sha(p)) for p in
                 (Path(__file__), Path(analysis.__file__), Path(saved.__file__), Path(channels.__file__))]
                + ([dict(path=str(p.relative_to(ROOT)), sha256=analysis.sha(p)) for p in
                    (Path(se_scoring.__file__), Path(se_scoring.v3.__file__))] if output_rule == "channel_se" else []),
        manifests=manifests)
    if output_rule == "channel_se":
        receipt["se_scoring"] = dict(
            rule="Primary outcome: V3 standard-error check (code/paper/se_scoring.py), "
                 "decision record 2026-09-25_se-scored-primary-outcome.md",
            tasks=sorted(se_scoring.spec()), spec=str(se_scoring.SPEC_PATH.relative_to(ROOT)),
            spec_sha256=se_scoring.spec_sha256(),
            se_failed_runs=int((rescoring.se_failed_terms != "").sum()),
            se_conflict_runs=int((rescoring.se_conflict_terms != "").sum()))
    write_json(destination / "snapshot.json", receipt)
    result[["position_id", "record_path", "record_sha256", "output_path", "output_sha256"]].to_csv(destination / "selected_records.csv", index=False)
    previous = baseline[CELL + ["global_rep", "record_path", "record_sha256"]].copy()
    previous["record_path"] = [repo_relative(p) for p in previous.record_path]
    lineage = result[CELL + ["global_rep", "record_path", "record_sha256"]].merge(
        previous, on=CELL + ["global_rep"], suffixes=("", "_previous"), validate="one_to_one")
    lineage["replaced"] = lineage.record_sha256.ne(lineage.record_sha256_previous)
    if int(lineage.replaced.sum()) != (720 if use_replacements else 0):
        raise ValueError("Replacement lineage does not match the approved 720 positions")
    lineage.to_csv(destination / "selection_lineage.csv", index=False)
    print(json.dumps(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--baseline", action="store_true",
                        help="Build the earlier sample without the 720 September 21 replacements "
                             "(default: the paper sample with them)")
    parser.add_argument("--output-rule", choices=OUTPUT_RULES, default=OUTPUT_RULE,
                        help="channel_se (default): channel plus the V3 standard-error check in nine tasks; "
                             "channel: estimates only, also read final-execution files and stdout for Stata; "
                             "log: earlier estimate-only rule")
    args = parser.parse_args()
    build(args.out, not args.baseline, args.output_rule)

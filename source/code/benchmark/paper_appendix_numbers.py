"""Numbers for three appendix passages: the joint model and agency figure, the self-repair
silent failures, and the causes of the silent failures in the hand audit.

No model calls and no script execution. Reads, from the analysis folder:

    data/runs.csv                      outcomes, agency, model, failure category, model calls
    data/model_agency_frontier.csv     the six points of the Appendix C joint figure
    saved run records                  the first attempt of each self-repair run
and quality_reports/audits/2026-10-01_hand-audit-paper/out/verdicts.csv (hand-audit verdicts).

Writes data/appendix_numbers.json with the counts and the num* macros that
code/paper/manuscript.py merges into numbers.tex.

Self-repair first attempts: an attempt is complete when it ran without error and its output
holds a finite value for every scored estimand (the same reading as the final scoring). A
self-repair run that ends as a silent failure either had a complete but wrong first attempt
(one model call) or had an incomplete first attempt that the repairs turned into a complete
but wrong output.

Usage: uv run --no-sync python code/benchmark/paper_appendix_numbers.py [analysis folder]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/benchmark'))
sys.path.insert(0, str(ROOT / 'code/paper'))
from paper_numbers import fixed  # noqa: E402

HAND_AUDIT = ROOT / 'quality_reports/audits/2026-10-01_hand-audit-paper/out/verdicts.csv'
MODELS = {'sonnet': 'Sonnet', 'luna': 'Luna'}
AGENCIES = {'chatbot': 'Chatbot', 'self_repair': 'Repair', 'agent': 'Agent'}
# Hand-audit causes of silent failures, in the order the appendix names them.
CAUSES = (('coding_slip', 'CodingSlip'), ('wrong_sample_or_variables', 'Sample'),
          ('wrong_se_convention', 'SeConvention'), ('wrong_quantity_reported', 'Quantity'),
          ('wrong_estimator', 'Estimator'))


def cents(value: float) -> str:
    """Two decimals below one cent, one decimal above."""
    return fixed(value, 2 if value < 1 else 1)


def joint_macros(analysis: Path) -> tuple[dict, dict]:
    points = pd.read_csv(analysis / 'data/model_agency_frontier.csv')
    if len(points) != 6:
        raise ValueError('The joint figure needs one point per model and agency condition')
    macros, rows = {}, {}
    for r in points.itertuples():
        tag = MODELS[r.model] + AGENCIES[r.agency]
        macros[f'numJointSuccess{tag}'] = fixed(r.success_pct, 1)
        macros[f'numJointCents{tag}'] = cents(r.cost_cents)
        rows[tag] = dict(success_pct=float(r.success_pct), cost_cents=float(r.cost_cents))
    return macros, rows


def first_attempt_complete(row, spec, c, dataset) -> bool:
    """Complete when attempt 1 ran without error and output a finite value for every scored estimand."""
    folder = dataset.repo_path(row.record_path).parent
    record = json.loads(dataset.repo_path(row.record_path).read_text())
    executions = record.get('executions') or []
    if not executions:
        return False
    first = executions[0]
    if first.get('timed_out') or first.get('returncode') not in (0, None):
        return False
    workspace = folder / 'execution_workspace0001'
    output = workspace / Path(row.output_path).name
    if not output.is_file():
        # Early collections keep no per-attempt workspace; the recorded check then decides.
        return bool(first.get('ok'))
    values, _, _, _ = dataset.extract(output, row.platform, spec, c.base)
    return all(term in values and c.sp.number_or_none(values[term].get('estimate')) is not None
               for term in spec['estimands'])


def self_repair_counts(analysis: Path) -> tuple[dict, dict]:
    import dataset
    runs = pd.read_csv(analysis / 'data/runs.csv', low_memory=False)
    catalog = json.loads((analysis / 'data/catalog.json').read_text())
    c = dataset.analysis.runtime.load_runtime(analysis / 'work/scoring').c
    repair = runs[runs.agency.eq('self_repair')].copy()
    repair['first_complete'] = [
        True if r.calls_count == 1 else first_attempt_complete(r, catalog[r.task], c, dataset)
        for r in repair.itertuples()]
    macros, counts = {}, {}
    for model, tag in MODELS.items():
        m = repair[repair.model.eq(model)]
        silent = m[m.failure_category.eq('Silent failure')]
        silent_from_incomplete = silent[~silent.first_complete.astype(bool)]
        incomplete_first = m[~m.first_complete.astype(bool)]
        repaired_success = incomplete_first[incomplete_first.y.eq(1)]
        counts[model] = dict(self_repair_runs=len(m), silent=len(silent),
                             silent_first_incomplete=len(silent_from_incomplete),
                             silent_first_complete=len(silent) - len(silent_from_incomplete),
                             first_incomplete=len(incomplete_first), first_incomplete_success=len(repaired_success))
        share = 100 * len(silent_from_incomplete) / len(silent) if len(silent) else math.nan
        success = 100 * len(repaired_success) / len(incomplete_first) if len(incomplete_first) else math.nan
        macros[f'numRepairSilent{tag}'] = f'{len(silent):,}'
        macros[f'numRepairSilentFromIncomplete{tag}'] = f'{len(silent_from_incomplete):,}'
        macros[f'numRepairSilentFromIncompletePct{tag}'] = fixed(share, 0)
        macros[f'numRepairIncompleteFirst{tag}'] = f'{len(incomplete_first):,}'
        macros[f'numRepairIncompleteToSuccessPct{tag}'] = fixed(success, 0)
    return macros, counts


def hand_audit_macros() -> tuple[dict, dict]:
    verdicts = pd.read_csv(HAND_AUDIT)
    silent = verdicts[verdicts.group.eq('fail') & verdicts.failure_category.eq('Silent failure')]
    total = len(silent)
    shares = {cause: 100 * silent.fail_cause.eq(cause).sum() / total for cause, _ in CAUSES}
    shares['other'] = 100 - sum(shares.values())
    macros = {'numAuditSilentRuns': f'{total:,}', 'numAuditRuns': f'{len(verdicts):,}'}
    for cause, tag in CAUSES:
        macros[f'numAuditSilent{tag}Pct'] = fixed(shares[cause], 0)
    return macros, dict(silent_runs=total, sampled_runs=len(verdicts), shares_pct=shares)


def build(analysis: Path) -> dict:
    analysis = Path(analysis).resolve()
    joint, points = joint_macros(analysis)
    repair, counts = self_repair_counts(analysis)
    audit, causes = hand_audit_macros()
    macros = {**joint, **repair, **audit}
    assert all(k.isalpha() for k in macros), macros
    record = dict(joint_points=points, self_repair=counts, hand_audit_silent_causes=causes, macros=macros)
    (analysis / 'data/appendix_numbers.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('analysis', type=Path, nargs='?', default=ROOT / 'data/analysis/main')
    result = build(parser.parse_args().analysis)
    print(json.dumps({k: result[k] for k in ('self_repair', 'hand_audit_silent_causes')}, indent=2))
